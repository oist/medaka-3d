"""Bounded view and node correspondence proposals, never accepted identities."""

from __future__ import annotations

from itertools import permutations, product

import numpy as np
from geometry import geometry_metrics

PERMUTATIONS = tuple(permutations(range(3)))
PAIRS = ((0, 1), (0, 2), (1, 2))


def huber(value):
    a = np.abs(value)
    return np.where(a <= 2, 0.5 * a**2, 2 * (a - 1))


def fit_flank(xyz, frames):
    """Fit 17 native rows; missing rows never become fictitious adjacent steps."""
    xyz, frames = np.asarray(xyz, float), np.asarray(frames)
    if xyz.shape != (len(frames), 3, 3) or frames.ndim != 1:
        raise ValueError("flank needs (frames, 3 fish, xyz) and a one-dimensional schedule")
    if not np.isfinite(frames).all() or np.any(frames != np.floor(frames)):
        raise ValueError("flank frame indices must be finite integers")
    models = []
    for slot in range(3):
        valid = np.isfinite(xyz[:, slot]).all(-1)
        t, x = frames[valid], xyz[valid, slot]
        model = dict(valid=False, slot=slot, count=int(valid.sum()))
        adjacent = valid[1:] & valid[:-1] & (np.diff(frames) == 1)
        native_steps = np.linalg.norm(np.diff(xyz[:, slot], axis=0), axis=-1)[adjacent]
        if len(frames) != 17:
            model["reason"] = "incomplete_17_frame_flank"
        elif np.any(np.diff(frames) != 1):
            model["reason"] = "non_native_flank_schedule"
        elif len(t) < 14:
            model["reason"] = "insufficient_pectoral_support"
        elif np.any(native_steps > 4):
            model["reason"] = "unstable_reference_step"
        else:
            ii, jj = np.triu_indices(len(t), k=1)
            held = t[jj] - t[ii] >= 3
            if not held.any():
                model["reason"] = "insufficient_slope_pairs"
                models.append(model)
                continue
            velocity = np.median(
                (x[jj[held]] - x[ii[held]]) / (t[jj[held]] - t[ii[held]])[:, None], axis=0
            )
            center = float(np.median(t))
            intercept = np.median(x - (t - center)[:, None] * velocity, axis=0)
            error = float(
                np.median(np.linalg.norm(x - intercept - (t - center)[:, None] * velocity, axis=-1))
            )
            model.update(
                center=center,
                intercept=intercept.tolist(),
                velocity=velocity.tolist(),
                median_error_mm=error,
                valid=error <= 1.5,
                reason="stable" if error <= 1.5 else "curved_or_noisy_reference",
            )
        models.append(model)
    return models


def predictions(models, frame):
    out = np.full((3, 3), np.nan)
    for s, model in enumerate(models):
        if model["valid"]:
            out[s] = np.array(model["intercept"]) + (frame - model["center"]) * np.array(
                model["velocity"]
            )
    return out


def assemble(rows_by_view, owners):
    """Gather per-node sources; each owner is a frame-local detection row."""
    owners = _owner_array(owners)
    if len(rows_by_view) != 3:
        raise ValueError("exactly three camera detection arrays are required")
    uv = np.full((*owners.shape, 2), np.nan)
    for v, rows in enumerate(rows_by_view):
        rows = np.asarray(rows, dtype=float)
        if rows.ndim != 3 or rows.shape[1:] != (3, 2):
            raise ValueError("each camera detection array must have shape (rows, 3 nodes, 2)")
        for node in range(3):
            indices = owners[..., v, node]
            valid = indices >= 0
            if np.any(indices[valid] >= len(rows)):
                raise ValueError("candidate refers to nonexistent detection")
            uv[..., v, node, :][valid] = rows[indices[valid], node]
    return uv


def _owner_array(owners):
    owners = np.asarray(owners)
    if owners.ndim < 3 or owners.shape[-3:] != (3, 3, 3):
        raise ValueError("owners must have shape (..., 3 slots, 3 views, 3 nodes)")
    if not np.isfinite(owners).all() or np.any(owners != np.floor(owners)):
        raise ValueError("owners must contain integer detection row numbers")
    if np.any(owners < -1):
        raise ValueError("only -1 denotes an absent detection")
    return owners.astype(np.int64, copy=False)


def ownership_bijective(owners):
    """Nonmissing rows may be owned once per view/node; missing -1 may repeat."""
    ordered = np.sort(_owner_array(owners), axis=-3)
    duplicates = (ordered[..., 1:, :, :] == ordered[..., :-1, :, :]) & (ordered[..., 1:, :, :] >= 0)
    return ~duplicates.any(axis=(-3, -2, -1))


def owner_candidates(source, rows_by_view):
    source = np.asarray(source)
    if source.shape != (3, 3):
        raise ValueError("source must have shape (3 slots, 3 views)")
    base = _owner_array(np.repeat(source[..., None], 3, axis=-1))
    # Validate source ranges and missing-value conventions before enumeration.
    assemble(rows_by_view, base)
    result, families = [], []
    for per_view in product(PERMUTATIONS, repeat=3):
        candidate = base.copy()
        for v, order in enumerate(per_view):
            candidate[:, v] = base[list(order), v]
        result.append(candidate)
        families.append("whole_view_permutation")
    for v, rows in enumerate(rows_by_view):
        unused = sorted(set(range(len(rows))) - set(source[:, v]))
        for row in unused:
            for slot in range(3):
                candidate = base.copy()
                candidate[slot, v] = row
                result.append(candidate)
                families.append("unused_whole_detection")
    return np.stack(result), families


def node_candidates(base, best):
    result, families = [], []
    for origin, root in (("baseline", base), ("whole_best", best)):
        for view in range(3):
            for a, b in PAIRS:
                for nodes in ((0,), (1,), (2,), (0, 1)):
                    candidate = root.copy()
                    for node in nodes:
                        candidate[a, view, node], candidate[b, view, node] = (
                            root[b, view, node],
                            root[a, view, node],
                        )
                    result.append(candidate)
                    families.append(
                        f"{origin}_view{view}_pair{a}{b}_nodes" + "".join(map(str, nodes))
                    )
    return np.stack(result), families


def score(metrics, owners, base, prior, forward, backward, baseline_uv):
    pose = metrics["pose"]
    ray = metrics["ray_rms_mm"]
    finite_nodes = np.isfinite(pose).all(-1)
    geometry = np.mean(np.where(finite_nodes, np.minimum((ray / 1.0) ** 2, 100), 0), axis=(-2, -1))
    lengths = np.stack([metrics["mp_mm"], metrics["pt_mm"]], axis=-1)
    body = np.mean(
        np.where(np.isfinite(lengths), huber((lengths - prior["median"]) / prior["scale"]), 0),
        axis=(-2, -1),
    )
    motion_parts = []
    directional_errors = []
    for predicted in (forward, backward):
        good = np.isfinite(predicted).all(-1)
        error = np.linalg.norm(pose[..., 1, :] - predicted, axis=-1)
        directional_errors.append(error)
        if good.any():
            motion_parts.append(
                np.mean(
                    np.where(np.isfinite(error[..., good]), huber(error[..., good] / 3), 50),
                    axis=-1,
                )
            )
    motion = np.mean(motion_parts, axis=0) if motion_parts else np.zeros(len(owners))
    changed = np.count_nonzero(owners != base, axis=(-3, -2, -1))
    missing = 9 - finite_nodes.sum(axis=(-2, -1))
    cost = geometry + body + 0.25 * motion + 0.05 * changed + 50 * missing
    # A missing projection or fewer supported2Dnodes never wins by reducing residuals.
    base_support = np.isfinite(baseline_uv).all(-1).sum()
    support = metrics["view_counts"].sum(axis=(-2, -1))
    finite_projections = np.isfinite(metrics["reprojection_px"]).sum(axis=-2)
    projection_bad = np.any(
        (metrics["view_counts"] >= 2) & (finite_projections < metrics["view_counts"]),
        axis=(-2, -1),
    )
    bijective = ownership_bijective(owners)
    cost = np.where(
        (support >= base_support) & ~projection_bad & bijective & np.isfinite(cost),
        cost,
        np.inf,
    )
    return cost, dict(
        geometry=geometry,
        body=body,
        motion=motion,
        changed_nodes=changed,
        missing_nodes=missing,
        forward_error_mm=directional_errors[0],
        backward_error_mm=directional_errors[1],
        ownership_bijective=bijective,
        projections_valid=~projection_bad,
    )


def propose(rig, source, rows_by_view, prior, forward, backward):
    owners, families = owner_candidates(source, rows_by_view)
    base = np.repeat(np.asarray(source, int)[..., None], 3, axis=-1)
    base_uv = assemble(rows_by_view, base)
    uv = assemble(rows_by_view, owners)
    metrics = geometry_metrics(rig, uv)
    costs, components = score(metrics, owners, base, prior, forward, backward, base_uv)
    whole_best = int(np.argmin(costs))
    extra_owners, extra_families = node_candidates(base, owners[whole_best])
    extra_uv = assemble(rows_by_view, extra_owners)
    extra_metrics = geometry_metrics(rig, extra_uv)
    extra_costs, extra_components = score(
        extra_metrics, extra_owners, base, prior, forward, backward, base_uv
    )
    all_owners = np.concatenate([owners, extra_owners])
    costs = np.r_[costs, extra_costs]
    families += extra_families
    # Same hypothesis may occur repeatedly with missing rows; margins use distinct owners.
    unique = {}
    for i, candidate in enumerate(all_owners):
        unique.setdefault(candidate.tobytes(), i)
    order = sorted(unique.values(), key=lambda i: (costs[i], i))
    chosen = order[0]
    if not np.isfinite(costs[chosen]):
        chosen = 0  # Preserve the baseline when all candidates are unsupported.
    finite_order = [i for i in order if np.isfinite(costs[i])]
    runnerup_margin = (
        float(costs[finite_order[1]] - costs[chosen]) if len(finite_order) >= 2 else None
    )
    all_uv = np.concatenate([uv, extra_uv])
    all_metrics = {key: np.concatenate([metrics[key], extra_metrics[key]]) for key in metrics}
    all_components = {
        key: np.concatenate([components[key], extra_components[key]]) for key in components
    }
    best_pose = all_metrics["pose"][chosen]
    segments = np.stack([all_metrics["mp_mm"][chosen], all_metrics["pt_mm"][chosen]], axis=-1)
    support_preserved = np.all(
        ~np.isfinite(metrics["pose"][0]).all(-1) | np.isfinite(best_pose).all(-1)
    )
    in_tank = (
        np.isfinite(best_pose).all()
        and np.all(best_pose >= 0)
        and np.all(best_pose <= np.array([192, 192, 186]))
    )
    cohesive = np.isfinite(segments).all() and np.all(
        np.abs((segments - prior["median"]) / prior["scale"]) <= 3
    )
    gain = costs[0] - costs[chosen] if np.isfinite(costs[0]) else np.nan
    worth = bool(
        chosen != 0
        and np.isfinite(gain)
        and gain >= 0.5
        and gain >= 0.25 * costs[0]
        and support_preserved
        and in_tank
        and cohesive
    )
    detail = dict(
        family=families[chosen],
        chosen_index=chosen,
        candidate_count=len(unique),
        finite_candidate_count=len(finite_order),
        original_cost=float(costs[0]),
        whole_best_cost=float(costs[whole_best]),
        candidate_cost=float(costs[chosen]),
        gain=float(gain),
        runnerup_margin=runnerup_margin,
        review_worthy=worth,
        accepted_identity_correction=False,
        coverage_preserved=bool(support_preserved),
        in_tank=bool(in_tank),
        cohesive=bool(cohesive),
        whole_best_source=owners[whole_best].tolist(),
        original_components={k: v[0].tolist() for k, v in all_components.items()},
        candidate_components={k: v[chosen].tolist() for k, v in all_components.items()},
        alternatives=[
            dict(family=families[i], cost=float(costs[i]), source=all_owners[i].tolist())
            for i in order[:5]
        ],
    )
    return (
        all_owners[chosen],
        all_uv[chosen],
        {key: val[chosen] for key, val in all_metrics.items()},
        detail,
    )

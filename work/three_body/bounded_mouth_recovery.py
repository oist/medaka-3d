"""Additional mouth proposals only for eligible free intersections outside the box.

The frozen node_recovery helper is called for validation and original diagnostic
gates; its proposals are never returned here. A caller must retain its original
pool separately and apply its existing temporal age gate. No fit, age inference,
identity reset, review change or earlier winner coordinate is used by this helper.
"""

from __future__ import annotations

import copy
from itertools import combinations, product

import bounded_ray_seed
import node_recovery as original
import numpy as np
from node_geometry import supported_nodes

from medaka3d.calib import triangulate
from medaka3d.types import VIEWS

VERSION = "bounded_mouth_recovery_v1"


def _bounded_candidate(pose, plan, lengths, fish, source, origins, directions, rays, rig, cfg):
    """Solve from rays/box only, then apply the unchanged geometric gates."""
    try:
        solved = bounded_ray_seed.solve_bounded_ray_seed(
            origins, directions, cfg.lower_mm, cfg.upper_mm
        )
    except (ValueError, RuntimeError) as error:
        return None, dict(
            status="bounded_solver_failed", error_type=type(error).__name__, message=str(error)
        )
    point = solved.point_xyz
    # Preserve the old raw-ray error/ranking arithmetic. The convex solver uses
    # normalized directions internally; actual calibrated rays already are unit.
    delta = point - origins
    depth = (delta * directions).sum(-1)
    perpendicular = delta - depth[:, None] * directions
    rms = float(np.sqrt(np.sum(perpendicular * perpendicular) / len(origins)))
    info = dict(
        status="admitted",
        point_xyz=point.tolist(),
        ray_rms_mm=rms,
        solver_rms_mm=solved.rms_mm,
        ray_depth_mm=depth.tolist(),
        active_bounds=solved.active_bounds.tolist(),
        solver_status=solved.solver_status,
        kkt_violation=solved.kkt_violation,
        kkt_tolerance=solved.kkt_tolerance,
        stacked_matrix_condition_number=solved.condition_number,
    )
    if (
        not np.isfinite(point).all()
        or not np.isfinite(rms)
        or not np.isfinite(depth).all()
        or np.any(depth < 0)
    ):
        info["status"] = "bounded_nonfinite_or_behind"
        return None, info
    if rms > cfg.ray_rms_mm:
        info["status"] = "bounded_ray_rms"
        return None, info
    if np.any(point < cfg.lower_mm) or np.any(point > cfg.upper_mm):
        info["status"] = "bounded_outside_box"
        return None, info
    length = float(np.linalg.norm(point - pose[fish, 1]))
    ratio = length / lengths[fish, 0]
    info.update(mouth_pectoral_mm=length, length_ratio=float(ratio))
    if not cfg.minimum_length_ratio <= ratio <= cfg.maximum_length_ratio:
        info["status"] = "bounded_anatomy"
        return None, info
    candidate_pose, candidate_plan = pose.copy(), plan.copy()
    candidate_pose[fish, 0] = point
    candidate_plan[fish, :, 0] = source
    if not original._injective(candidate_plan):
        raise AssertionError("bounded candidate stole a protected current-frame mouth row")
    sources, observed, _ = supported_nodes(candidate_pose, candidate_plan, rays, rig)
    info.update(
        strict_sources=sources[fish, :, 0].tolist(),
        mouth_direct=bool(observed[fish, 0]),
        pectoral_direct=bool(observed[fish, 1]),
    )
    if not observed[fish, 0] or not observed[fish, 1]:
        info["status"] = "bounded_strict_support"
        return None, info
    proposal = original.NodeRecoveryProposal(
        pose_xyz=candidate_pose,
        plan=candidate_plan,
        fish=fish,
        node=0,
        displacement_mm=float(np.linalg.norm(point - pose[fish, 0])),
        mouth_pectoral_mm=length,
        length_ratio=float(ratio),
        ray_rms_mm=rms,
        supported_views=int((sources[fish, :, 0] >= 0).sum()),
    )
    return proposal, info


def bounded_mouth_recovery_proposals(scene_xyz, base_plan, rays, rig, initial_lengths, config=None):
    """Return only new bounded proposals, with the original result/proposal types.

    Eligibility, source protection, 3-view then 2-view enumeration, combination
    limit, anatomy bounds, 2-mm/12-pixel support and per-fish ranking/cap retain
    the original rules. Only free intersections that are finite, forward, pass
    the original RMS gate and lie outside the box can call the bounded solver.
    The original in-box candidates are neither added to this result nor replaced.

    Diagnostics include the original helper's report and every bounded attempt.
    A returned plan is the trial assignment, not a promise that every assigned
    third ray survives strict filtering. No cross-candidate deduplication is
    added: the original lexicographic ranking and stable ties are preserved.
    """
    cfg = config or original.NodeRecoveryConfig()
    legacy = original.node_recovery_proposals(scene_xyz, base_plan, rays, rig, initial_lengths, cfg)
    pose = np.array(scene_xyz, dtype=float, copy=True)
    plan = np.array(base_plan, dtype=np.int32, copy=True)
    lengths = np.array(initial_lengths, dtype=float, copy=True)
    lower, upper = np.asarray(cfg.lower_mm), np.asarray(cfg.upper_mm)
    all_proposals, fish_diagnostics = [], []
    for fish, old_info in enumerate(legacy.diagnostics["fish"]):
        info = dict(
            fish=fish,
            status="no_outside_box_intersections",
            pectoral_direct=old_info["pectoral_direct"],
            mouth_direct=old_info["mouth_direct"],
            combinations=old_info["combinations"],
            bounded_attempts=0,
            admitted=0,
            returned=0,
            attempts=[],
            rejection_counts={},
        )
        fish_diagnostics.append(info)
        expected_outside = old_info.get("rejection_counts", {}).get("outside_box", 0)
        if not expected_outside:
            if old_info["status"] not in ("proposals_available", "no_eligible_intersection"):
                info["status"] = old_info["status"]
            continue
        free = []
        for view, name in enumerate(VIEWS):
            finite = np.isfinite(rays.points[name][:, 0]).all(-1)
            finite &= np.isfinite(rays.origins[view][:, 0]).all(-1)
            finite &= np.isfinite(rays.directions[view][:, 0]).all(-1)
            protected = plan[np.arange(3) != fish, view, 0]
            free.append(np.flatnonzero(finite & ~np.isin(np.arange(len(finite)), protected)))
        admitted = []
        combination_index = 0
        for count in (3, 2):
            for views in combinations(range(3), count):
                choices = list(product(*(free[v] for v in views)))
                if not choices:
                    continue
                origins = np.array(
                    [
                        [rays.origins[v][row, 0] for v, row in zip(views, rows, strict=True)]
                        for rows in choices
                    ]
                )
                directions = np.array(
                    [
                        [rays.directions[v][row, 0] for v, row in zip(views, rows, strict=True)]
                        for rows in choices
                    ]
                )
                points, errors = triangulate(origins, directions)
                for index, (point, error) in enumerate(zip(points, errors, strict=True)):
                    number = combination_index
                    combination_index += 1
                    if (
                        not np.isfinite(point).all()
                        or not np.isfinite(error)
                        or np.any(((point - origins[index]) * directions[index]).sum(-1) < 0)
                        or error > cfg.ray_rms_mm
                    ):
                        continue
                    if not (np.any(point < lower) or np.any(point > upper)):
                        continue
                    source = np.full(3, -1, np.int32)
                    source[list(views)] = choices[index]
                    candidate, attempt = _bounded_candidate(
                        pose,
                        plan,
                        lengths,
                        fish,
                        source,
                        origins[index],
                        directions[index],
                        rays,
                        rig,
                        cfg,
                    )
                    attempt.update(
                        combination_index=number,
                        source_rows=source.tolist(),
                        free_point_xyz=point.tolist(),
                        free_ray_rms_mm=float(error),
                        returned=False,
                    )
                    info["attempts"].append(attempt)
                    if candidate is None:
                        status = attempt["status"]
                        info["rejection_counts"][status] = (
                            info["rejection_counts"].get(status, 0) + 1
                        )
                    else:
                        admitted.append((candidate, attempt))
        if (
            combination_index != old_info["combinations"]
            or len(info["attempts"]) != expected_outside
        ):
            raise RuntimeError("bounded enumeration differs from frozen outside-box diagnostics")
        admitted.sort(
            key=lambda item: (
                -item[0].supported_views,
                item[0].ray_rms_mm,
                item[0].displacement_mm,
                tuple(item[0].plan[fish, :, 0]),
            )
        )
        selected = admitted[: cfg.proposals_per_fish]
        for candidate, attempt in selected:
            attempt["returned"] = True
            all_proposals.append(candidate)
        info.update(
            status="proposals_available" if selected else "no_eligible_bounded_intersection",
            bounded_attempts=len(info["attempts"]),
            admitted=len(admitted),
            returned=len(selected),
        )
    return original.NodeRecoveryResult(
        all_proposals,
        dict(
            schema=VERSION,
            solver_version=bounded_ray_seed.VERSION,
            target_node="mouth",
            scope="additional current-frame bounded mouth geometry; no identity certification",
            original_candidates_returned=False,
            age_gate_applied=False,
            base_direct_support=copy.deepcopy(legacy.diagnostics["base_direct_support"]),
            legacy_diagnostics=copy.deepcopy(legacy.diagnostics),
            fish=fish_diagnostics,
        ),
    )

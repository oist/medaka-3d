"""Clean 2D anchors, refractive geometry, and a separate cross-view join layer.

No saved SLEAP, C27, C28, or C34 identity is an input. Missing observations stay
missing; predicted positions are only assignment hypotheses. This is a local
experimental estimator, not a declaration of biological identity correctness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise, product

import numpy as np

from medaka3d.calib import triangulate
from medaka3d.types import VIEWS


@dataclass(frozen=True)
class MultiViewConfig:
    ray_gate_mm: float = 1.5
    missing_view_cost_mm: float = 1.5
    missing_mouth_cost_mm: float = 4.0
    absent_slot_cost_mm: float = 8.0
    maximum_speed_mm_per_frame: float = 4.0
    prediction_gate_mm: float = 5.0
    gap_prediction_growth_mm: float = 1.0
    prediction_weight: float = 0.25
    single_clean_vote_bonus_mm: float = 0.5
    correspondence_margin_mm: float = 0.5
    identity_gap_frames: int = 5
    bridge_gap_frames: int = 166
    bridge_projection_gate_px: float = 8.0
    bridge_competitor_margin_px: float = 4.0
    tank_lower_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tank_upper_mm: tuple[float, float, float] = (192.0, 192.0, 186.0)


def _cameras(rig):
    if hasattr(rig, "camera"):
        return {v: rig.camera(v) for v in VIEWS}
    if isinstance(rig, dict):
        return {v: rig[v] for v in VIEWS}
    cameras = {c.view: c for c in rig}
    if set(cameras) != set(VIEWS):
        raise ValueError("camera views must be top_cam, side1_cam, side2_cam")
    return cameras


def _prepare(frames, views, cameras):
    prepared = {}
    for view in VIEWS:
        data = views[view]
        f = np.asarray(data["frame_idx"], dtype=np.int64)
        points = np.asarray(data["points"], dtype=float)
        labels = np.asarray(data["labels"], dtype=np.int64)
        clean = np.asarray(data["clean_edge"], dtype=bool)
        if f.ndim != 1 or points.shape != (len(f), 3, 2):
            raise ValueError("per-view points must have shape (N, 3 nodes, 2)")
        if labels.shape != f.shape or clean.shape != f.shape or np.any(labels < 0):
            raise ValueError("labels/clean_edge must cover every row; labels are nonnegative")
        if not np.isin(f, frames).all():
            raise ValueError("all detection rows must belong to the requested native frames")
        groups = {int(frame): np.flatnonzero(f == frame) for frame in np.unique(f)}
        for rows in groups.values():
            if len(np.unique(labels[rows])) != len(rows):
                raise ValueError("one clean label cannot occupy two rows in the same frame")
        label_rows = {}
        for row in np.argsort(f, kind="stable"):
            label_rows.setdefault(int(labels[row]), []).append(int(row))
        for rows in label_rows.values():
            if clean[rows[0]]:
                raise ValueError("the first row of a clean tracklet cannot have an incoming edge")
            for before, after in pairwise(rows):
                if not clean[after] or f[after] != f[before] + 1:
                    raise ValueError("a clean label must be a contiguous component of clean edges")
        origins = np.full((len(f), 3, 3), np.nan)
        directions = origins.copy()
        valid = np.isfinite(points).all(-1)
        if valid.any():
            origins[valid], directions[valid] = cameras[view].rays(points[valid])
        prepared[view] = dict(
            frames=f,
            points=points,
            labels=labels,
            clean=clean,
            groups=groups,
            label_rows=label_rows,
            origins=origins,
            directions=directions,
        )
    return prepared


def _hypotheses(frame, prepared, config):
    """All 2/3-view tuples; source indexes refer to original per-frame row order."""
    rows = [prepared[v]["groups"].get(int(frame), np.empty(0, dtype=int)) for v in VIEWS]
    tuples = [p for p in product(*[range(-1, len(r)) for r in rows]) if sum(i >= 0 for i in p) >= 2]
    if not tuples:
        return None
    source = np.asarray(tuples, dtype=np.int64)
    count = len(source)
    uv = np.full((count, 3, 3, 2), np.nan)
    origins = np.full((count, 3, 3, 3), np.nan)
    directions = origins.copy()
    global_rows = np.full(source.shape, -1, dtype=np.int64)
    for vi, view in enumerate(VIEWS):
        have = source[:, vi] >= 0
        selected = rows[vi][source[have, vi]]
        global_rows[have, vi] = selected
        uv[have, vi] = prepared[view]["points"][selected]
        origins[have, vi] = prepared[view]["origins"][selected]
        directions[have, vi] = prepared[view]["directions"][selected]
    origins, directions = origins.swapaxes(1, 2), directions.swapaxes(1, 2)
    valid = np.isfinite(origins).all(-1) & np.isfinite(directions).all(-1)
    pose, residual = triangulate(origins, directions, valid)
    inside = ((pose >= config.tank_lower_mm) & (pose <= config.tank_upper_mm)).all(-1)
    usable = np.isfinite(pose[:, 1]).all(-1) & (residual[:, 1] <= config.ray_gate_mm)
    mouth = np.isfinite(residual[:, 0])
    usable &= ~mouth | (residual[:, 0] <= config.ray_gate_mm)
    usable &= inside[:, 1] & (~mouth | inside[:, 0])
    usable &= ((source < 0) | valid[:, :2].any(axis=1)).all(axis=-1)
    # Tail does not choose correspondence. Unsupported tail intersections are
    # unavailable coordinates; the raw residual remains a separate diagnostic.
    raw_residual = residual.copy()
    tail_triangulated = np.isfinite(pose[:, 2]).all(-1)
    tail_outside = tail_triangulated & ~inside[:, 2]
    tail_high_residual = tail_triangulated & (residual[:, 2] > config.ray_gate_mm)
    tail_supported = inside[:, 2] & (residual[:, 2] <= config.ray_gate_mm)
    pose[:, 2] = np.where(tail_supported[:, None], pose[:, 2], np.nan)
    residual[:, 2] = np.where(tail_supported, residual[:, 2], np.nan)
    # A missing mouth never lowers the two-node geometry cost.
    mouth_cost = np.where(mouth, residual[:, 0], config.missing_mouth_cost_mm)
    cost = (mouth_cost + residual[:, 1]) / 2
    cost += config.missing_view_cost_mm * (source < 0).sum(-1)
    chosen = np.flatnonzero(usable)
    if not len(chosen):
        return None
    # World coordinates name previously unseen slots. Pixel values settle only
    # exact world ties, whose observation ambiguity is still rejected by margin.
    xyz = pose[chosen, 1]
    chosen = chosen[np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))]
    return {
        "source": source[chosen],
        "rows": global_rows[chosen],
        "uv": uv[chosen],
        "pose": pose[chosen],
        "residual": residual[chosen],
        "raw_residual": raw_residual[chosen],
        "tail_outside": tail_outside[chosen],
        "tail_high_residual": tail_high_residual[chosen],
        "cost": cost[chosen],
    }


def _vote_counts(hypotheses, prepared, owners):
    votes = np.zeros((len(hypotheses["source"]), 3), dtype=int)
    for vi, view in enumerate(VIEWS):
        for i, row in enumerate(hypotheses["rows"][:, vi]):
            if row < 0 or not prepared[view]["clean"][row]:
                continue
            owner = owners[view].get(int(prepared[view]["labels"][row]))
            if owner is not None:
                votes[i, owner] += 1
    return votes


def _assignment(hypotheses, histories, votes, frame, config):
    """Exact joint three-slot assignment over gated hypotheses, including absence."""
    n = len(hypotheses["source"])
    costs = np.broadcast_to(hypotheses["cost"], (3, n)).copy()
    conflicted = (votes > 0).sum(-1) > 1
    costs[:, conflicted] = np.inf
    for slot in range(3):
        strong_other = (votes >= 2).any(-1) & (votes[:, slot] < 2)
        costs[slot, strong_other] = np.inf
        costs[slot] -= config.single_clean_vote_bonus_mm * (votes[:, slot] == 1)
        history = histories[slot]
        if not history:
            continue
        last_frame, last = history[-1]
        gap = int(frame - last_frame)
        velocity = np.zeros(3)
        if len(history) >= 2:
            samples = [(b - a) / (fb - fa) for (fa, a), (fb, b) in pairwise(history)]
            velocity = np.median(samples, axis=0)
        xyz = hypotheses["pose"][:, 1]
        distance = np.linalg.norm(xyz - (last + velocity * gap), axis=-1)
        step = np.linalg.norm(xyz - last, axis=-1) / gap
        recent = gap <= config.identity_gap_frames
        # A long unanchored gap may reacquire geometry, but gets a persistent
        # identity-review flag in reconstruct; no extrapolated point is emitted.
        if recent:
            gate = config.prediction_gate_mm + config.gap_prediction_growth_mm * (gap - 1)
            costs[slot, (distance > gate) | (step > config.maximum_speed_mm_per_frame)] = np.inf
            costs[slot] += config.prediction_weight * distance
        else:
            costs[slot] += config.prediction_weight * np.minimum(distance, 12.0)
            anchored = votes[:, slot] >= 2
            costs[slot, anchored & (step > config.maximum_speed_mm_per_frame)] = np.inf
    # Costs above a missing slot are exactly dominated: absence consumes no rows.
    choices = [np.r_[np.flatnonzero(costs[s] <= config.absent_slot_cost_mm), n] for s in range(3)]
    state = np.stack(np.meshgrid(*choices, indexing="ij"), axis=-1).reshape(-1, 3)
    source = np.vstack([hypotheses["source"], np.full((1, 3), -1)])
    valid = np.ones(len(state), dtype=bool)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        left, right = source[state[:, a]], source[state[:, b]]
        valid &= ~np.any((left >= 0) & (left == right), axis=-1)
        if not histories[a] and not histories[b]:
            # Quotient out arbitrary permutations among never-observed slots.
            valid &= state[:, a] <= state[:, b]
    state = state[valid]
    padded_cost = np.column_stack([costs, np.full(3, config.absent_slot_cost_mm)])
    total = padded_cost[np.arange(3)[None, :], state].sum(-1)
    best_row = int(np.argmin(total))
    best = state[best_row]
    margins = np.full(3, np.inf)
    for slot in range(3):
        rival = state[:, slot] != best[slot]
        if rival.any():
            margins[slot] = float(total[rival].min() - total[best_row])
    return best, margins, n, int(conflicted.sum()), len(state)


def _project(camera, xyz):
    out = np.full((*xyz.shape[:-1], 2), np.nan)
    finite = np.isfinite(xyz).all(-1)
    if finite.any():
        out[finite] = camera.project(xyz[finite])
    return out


def _bridge_evidence(start, stop, slot, vi, rows, frames, prepared, cameras, config):
    """Validate third-view endpoints using only the other two views' rays."""
    other = [i for i in range(3) if i != vi]
    selected = rows[start : stop + 1, slot]
    if np.any(selected[:, other] < 0):
        return dict(reason="two_clean_views_not_observed_through_gap")
    support = {}
    ray_o, ray_d = [], []
    for k in other:
        view = VIEWS[k]
        data = prepared[view]
        rr = selected[:, k]
        labels = data["labels"][rr]
        if len(np.unique(labels)) != 1 or not data["clean"][rr[1:]].all():
            return dict(reason="two_clean_tracklets_not_contiguous")
        support[view] = int(labels[0])
        ray_o.append(data["origins"][rr, :2])
        ray_d.append(data["directions"][rr, :2])
    origins, directions = np.stack(ray_o, axis=-2), np.stack(ray_d, axis=-2)
    valid = np.isfinite(origins).all(-1) & np.isfinite(directions).all(-1)
    pose, residual = triangulate(origins, directions, valid)
    if not np.isfinite(pose).all() or np.any(residual > config.ray_gate_mm):
        return dict(reason="independent_two_view_geometry_unsupported", supporting_labels=support)
    if np.any((pose < config.tank_lower_mm) | (pose > config.tank_upper_mm)):
        return dict(reason="independent_two_view_outside_tank", supporting_labels=support)
    speed = np.linalg.norm(np.diff(pose[:, 1], axis=0), axis=-1) / np.diff(frames[start : stop + 1])
    if np.any(speed > config.maximum_speed_mm_per_frame):
        return dict(reason="independent_two_view_motion_gate", supporting_labels=support)
    view = VIEWS[vi]
    data = prepared[view]
    projection = _project(cameras[view], pose[[0, -1]])
    errors, margins = [], []
    for endpoint, time in enumerate((start, stop)):
        row = rows[time, slot, vi]
        target = data["points"][row, :2]
        if not np.isfinite(target).all() or not np.isfinite(projection[endpoint]).all():
            return dict(reason="endpoint_mouth_and_pectoral_required", supporting_labels=support)
        error = float(np.sqrt(np.mean(np.sum((target - projection[endpoint]) ** 2, axis=-1))))
        alternatives = data["groups"][int(frames[time])]
        competitors = []
        for rival in alternatives[alternatives != row]:
            delta = np.linalg.norm(data["points"][rival, :2] - projection[endpoint], axis=-1)
            finite = delta[np.isfinite(delta)]
            # Unknown competitors cannot manufacture a favorable margin.
            competitors.append(float(np.sqrt(np.mean(finite**2))) if finite.size else 0.0)
        margin = min(competitors) - error if competitors else np.inf
        errors.append(error)
        margins.append(float(margin))
    evidence = dict(
        supporting_labels=support,
        endpoint_projection_error_px=errors,
        endpoint_competitor_margin_px=margins,
        independent_ray_rms_max_mm=float(residual.max()),
        independent_speed_max_mm_per_frame=float(speed.max(initial=0)),
    )
    if max(errors) > config.bridge_projection_gate_px:
        return dict(reason="third_view_projection_gate", **evidence)
    if min(margins) < config.bridge_competitor_margin_px:
        return dict(reason="third_view_competitor_margin", **evidence)
    return dict(reason="two_clean_views_and_unique_endpoint_geometry", accepted=True, **evidence)


def _join_layer(frames, rows, identity_review, prepared, cameras, config):
    proposals = []
    for vi, view in enumerate(VIEWS):
        data = prepared[view]
        for slot in range(3):
            last = None
            for ti, row in enumerate(rows[:, slot, vi]):
                if row < 0:
                    continue
                if last is not None:
                    before, previous = last
                    left, right = int(data["labels"][previous]), int(data["labels"][row])
                    if left != right and frames[ti] - frames[before] <= config.bridge_gap_frames:
                        event = dict(
                            view=view,
                            slot=slot,
                            left_label=left,
                            right_label=right,
                            from_frame=int(frames[before]),
                            to_frame=int(frames[ti]),
                            from_row=int(previous),
                            to_row=int(row),
                            accepted=False,
                        )
                        if (
                            data["label_rows"][left][-1] != previous
                            or data["label_rows"][right][0] != row
                        ):
                            event["reason"] = "not_tracklet_endpoints"
                        elif identity_review[before : ti + 1, slot].any():
                            event["reason"] = "three_dimensional_identity_requires_review"
                        else:
                            event.update(
                                _bridge_evidence(
                                    before, ti, slot, vi, rows, frames, prepared, cameras, config
                                )
                            )
                        proposals.append(event)
                last = (ti, row)
    # Never resolve conflicting endpoint joins with event iteration order.
    for event in proposals:
        if not event["accepted"]:
            continue
        conflicts = [
            e
            for e in proposals
            if e is not event
            and e["accepted"]
            and e["view"] == event["view"]
            and (e["left_label"] == event["left_label"] or e["right_label"] == event["right_label"])
        ]
        if conflicts:
            event["conflicting_endpoints"] = True
            for conflict in conflicts:
                conflict["conflicting_endpoints"] = True
    for event in proposals:
        if event.get("conflicting_endpoints"):
            event.update(accepted=False, reason="non_unique_one_to_one_tracklet_join")
    joined = {}
    for view in VIEWS:
        labels = prepared[view]["labels"]
        parent = {int(label): int(label) for label in np.unique(labels)}

        def find(label, parent=parent):
            while parent[label] != label:
                label = parent[label]
            return label

        for event in proposals:
            if event["view"] == view and event["accepted"]:
                a, b = find(event["left_label"]), find(event["right_label"])
                parent[max(a, b)] = min(a, b)
        joined[view] = np.asarray([find(int(label)) for label in labels], dtype=np.int64)
        for frame_rows in prepared[view]["groups"].values():
            if len(np.unique(joined[view][frame_rows])) != len(frame_rows):
                raise RuntimeError("auxiliary joins assigned one identity to two simultaneous rows")
    return joined, proposals


def reconstruct(frames, views, cameras, *, config=None):
    """Estimate 3D from independent clean tracklets and audit auxiliary 2D joins.

    ``views[v]`` contains frame_idx[N], points[N,M/P/T,xy], labels[N] and
    clean_edge[N]. Input row order need not be sorted. Output source indexes are
    offsets among this input view's rows at that native frame, not label IDs.
    The original labels and incoming clean edges are returned unchanged.
    """
    config = MultiViewConfig() if config is None else config
    frames = np.asarray(frames, dtype=np.int64)
    if frames.ndim != 1 or (len(frames) > 1 and not np.all(np.diff(frames) == 1)):
        raise ValueError("frames must be unique contiguous native frames")
    camera_map = _cameras(cameras)
    prepared = _prepare(frames, views, camera_map)
    f = len(frames)
    pose = np.full((f, 3, 3, 3), np.nan)
    uv = np.full((f, 3, 3, 3, 2), np.nan)
    source = np.full((f, 3, 3), -1, dtype=np.int64)
    selected_rows = source.copy()
    residual = np.full((f, 3, 3), np.nan)
    raw_residual = residual.copy()
    tail_outside = np.zeros((f, 3), dtype=bool)
    tail_high_residual = tail_outside.copy()
    margins = np.full((f, 3), np.nan)
    identity_review = np.ones((f, 3), dtype=bool)
    sticky_review = np.zeros(3, dtype=bool)
    histories = [[] for _ in range(3)]
    owners = {view: {} for view in VIEWS}
    events = []
    max_hypotheses = max_states = 0
    for ti, frame in enumerate(frames):
        h = _hypotheses(frame, prepared, config)
        if h is None:
            events.append(dict(frame=int(frame), reason="no_supported_two_view_geometry"))
            continue
        votes = _vote_counts(h, prepared, owners)
        chosen, margin, missing, conflicts, states = _assignment(h, histories, votes, frame, config)
        max_hypotheses = max(max_hypotheses, missing)
        max_states = max(max_states, states)
        margins[ti] = margin
        if conflicts:
            events.append(
                dict(
                    frame=int(frame),
                    reason="contradicting_clean_anchors",
                    rejected_hypotheses=conflicts,
                )
            )
        for slot, index in enumerate(chosen):
            if index == missing:
                events.append(
                    dict(frame=int(frame), slot=slot, reason="no_joint_supported_assignment")
                )
                continue
            if margin[slot] < config.correspondence_margin_mm:
                events.append(
                    dict(
                        frame=int(frame),
                        slot=slot,
                        reason="correspondence_margin",
                        margin_mm=float(margin[slot]),
                    )
                )
                continue
            history = histories[slot]
            gap = int(frame - history[-1][0]) if history else 0
            if gap > config.identity_gap_frames and votes[index, slot] < 2:
                sticky_review[slot] = True
                events.append(
                    dict(
                        frame=int(frame),
                        slot=slot,
                        reason="long_gap_unanchored_identity_reacquisition",
                        gap_frames=gap,
                    )
                )
            identity_review[ti, slot] = sticky_review[slot]
            source[ti, slot] = h["source"][index]
            selected_rows[ti, slot] = h["rows"][index]
            pose[ti, slot] = h["pose"][index]
            uv[ti, slot] = h["uv"][index]
            residual[ti, slot] = h["residual"][index]
            raw_residual[ti, slot] = h["raw_residual"][index]
            tail_outside[ti, slot] = h["tail_outside"][index]
            tail_high_residual[ti, slot] = h["tail_high_residual"][index]
            history.append((int(frame), pose[ti, slot, 1].copy()))
            if len(history) > 4:
                del history[0]
            for vi, view in enumerate(VIEWS):
                row = selected_rows[ti, slot, vi]
                if row >= 0:
                    label = int(prepared[view]["labels"][row])
                    # Keep the first owner as an anchor rather than rewriting a
                    # clean tracklet to follow a later uncertain reconstruction.
                    owners[view].setdefault(label, slot)
    joined, bridges = _join_layer(
        frames, selected_rows, identity_review, prepared, camera_map, config
    )
    projection = np.stack([_project(camera_map[v], pose) for v in VIEWS], axis=2)
    return dict(
        xyz=pose[:, :, 1].copy(),
        pose=pose,
        source=source,
        uv=uv,
        residual_mm=residual,
        raw_residual_mm=raw_residual,
        projected_uv=projection,
        identity_review=identity_review,
        assignment_margin_mm=margins,
        labels_by_view={v: prepared[v]["labels"].copy() for v in VIEWS},
        clean_edge_by_view={v: prepared[v]["clean"].copy() for v in VIEWS},
        joined_labels_by_view=joined,
        bridges=bridges,
        events=events,
        metadata=dict(
            config=asdict(config),
            node_order=["mouth", "pectoral", "tail"],
            view_order=list(VIEWS),
            input_detections={v: len(prepared[v]["frames"]) for v in VIEWS},
            selected_detections={v: int((source[:, :, i] >= 0).sum()) for i, v in enumerate(VIEWS)},
            clean_edges_unchanged=True,
            saved_temporal_ids_used=False,
            historical_reconstruction_used=False,
            identity_accuracy_claim=False,
            accepted_bridges=sum(e["accepted"] for e in bridges),
            tail_masked_fish_frames=int((tail_outside | tail_high_residual).sum()),
            tail_outside_tank_fish_frames=int(tail_outside.sum()),
            tail_high_ray_residual_fish_frames=int(tail_high_residual.sum()),
            maximum_frame_hypotheses=max_hypotheses,
            maximum_joint_assignments=max_states,
            bridge_layer="auxiliary 2D joins; initial clean edges and 3D observations unchanged",
            identity_review_rule=(
                "missing observations and sticky unanchored reacquisition after "
                f">{config.identity_gap_frames} frames"
            ),
            source_index_rule=(
                "original input row order within each native frame; -1 means no observation"
            ),
        ),
    )

"""Conservative native-frame 2D tracklets from untracked detection rows.

The parameters below were fixed before running on the current recordings.
There is no SLEAP tracking-ID, 3D correspondence, identity annotation, or old-lab
input. The detector's within-row M/P/T grouping is still an input assumption.
Only adjacent native frames may link. Missing, close, inconsistent, and ambiguous
observations start new fragments; they are never removed. 'Clean' means passing
these internal checks, not human-validated biological identity.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TrackingConfig:
    native_fps: float = 166.0
    top_body_px: float = 120.0
    side_body_px: float = 100.0
    speed_fraction: float = 0.30
    prediction_fraction: float = 0.12
    assignment_margin_fraction: float = 0.025
    proximity_fraction: float = 0.08
    mouth_relative_motion_fraction: float = 0.12
    tail_relative_motion_fraction: float = 0.25
    length_change_fraction: float = 0.12
    max_primary_span_fraction: float = 0.80
    max_tail_span_fraction: float = 1.80
    min_primary_span_fraction: float = 0.005
    axis_change_deg: float = 70.0
    axis_min_length_fraction: float = 0.03
    history_observations: int = 4
    heading_speed_fraction: float = 0.02
    heading_step_fraction: float = 0.06
    heading_error_fraction: float = 0.06
    heading_change_deg: float = 85.0
    flow_error_fraction: float = 0.06
    flow_prediction_weight: float = 0.5

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("All fixed configuration values must be finite and positive")
        if self.history_observations < 3 or not 0 < self.flow_prediction_weight < 1:
            raise ValueError("Need three history observations and a fractional flow weight")
        if max(self.axis_change_deg, self.heading_change_deg) > 180:
            raise ValueError("Angles cannot exceed 180 degrees")

    def span(self, view):
        if view not in ("top_cam", "side1_cam", "side2_cam"):
            raise ValueError(f"Unknown view: {view}")
        return self.top_body_px if view == "top_cam" else self.side_body_px


@dataclass
class TrackletResult:
    labels: np.ndarray
    clean_edge: np.ndarray
    events: list[dict]
    metadata: dict


def _angle(first, second):
    product = np.linalg.norm(first) * np.linalg.norm(second)
    if product <= 1e-12:
        return 0.0  # A stationary observation has no motion heading.
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second) / product, -1, 1))))


def _inputs(frame_idx, points, view, feature_mode, flow, config):
    frames, points = np.asarray(frame_idx), np.asarray(points, dtype=float)
    span = config.span(view)
    if frames.ndim != 1 or frames.dtype.kind not in "iu":
        raise ValueError("frame_idx must be a one-dimensional native integer vector")
    if np.any(frames < 0) or np.any(np.diff(frames.astype(np.int64)) < 0):
        raise ValueError("Native frames must be nonnegative and sorted")
    if points.shape != (len(frames), 3, 2) or np.isinf(points).any():
        raise ValueError("points must be (N,3,2) M/P/T; missing nodes must be NaN")
    if feature_mode not in ("pectoral", "nodes"):
        raise ValueError("feature_mode must be pectoral or nodes")
    held_flow = {}
    if flow is not None:
        for direction in ("forward", "backward"):
            predicted = np.asarray(flow[f"predicted_{direction}"], dtype=float)
            valid = np.asarray(flow[f"valid_{direction}"], dtype=bool)
            if predicted.shape != points.shape or valid.shape != points.shape[:2]:
                raise ValueError("Flow predictions/validity must correspond to every input node")
            held_flow[f"predicted_{direction}"] = predicted
            held_flow[f"valid_{direction}"] = valid & np.isfinite(predicted).all(axis=-1)
    return frames.astype(np.int64, copy=False), points, span, held_flow


def _row_quality(points, groups, feature_mode, config, span):
    reasons = [[] for _ in points]
    for row, nodes in enumerate(points):
        if not np.isfinite(nodes[1]).all():
            reasons[row].append("missing_pectoral")
        if feature_mode == "nodes":
            if not np.isfinite(nodes[:2]).all():
                reasons[row].append("missing_primary_node")
                continue
            mouth_span = np.linalg.norm(nodes[0] - nodes[1])
            if not (
                config.min_primary_span_fraction * span
                <= mouth_span
                <= config.max_primary_span_fraction * span
            ):
                reasons[row].append("unsafe_primary_shape")
            if np.isfinite(nodes[2]).all() and (
                np.linalg.norm(nodes[2] - nodes[1]) > config.max_tail_span_fraction * span
            ):
                reasons[row].append("unsafe_tail_shape")
    for _, rows in groups:
        xy = points[rows, 1]
        distances = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        np.fill_diagonal(distances, np.inf)
        close = np.any(distances < config.proximity_fraction * span, axis=1)
        for row in rows[close]:
            reasons[row].append("nearby_detection")
    return reasons


def _prediction(history, target_frame, frames, points, config):
    origin = history[-1]
    predicted = points[origin].copy()
    velocity = np.full((3, 2), np.nan)
    if len(history) >= 3:
        selected = np.asarray(history[-config.history_observations :])
        dt = np.diff(frames[selected])
        differences = np.diff(points[selected], axis=0) / dt[:, None, None]
        for node in range(2):
            values = differences[:, node]
            good = np.isfinite(values).all(axis=-1)
            if good.sum() >= 2:
                velocity[node] = np.median(values[good], axis=0)
                predicted[node] += velocity[node] * (target_frame - frames[origin])
    return predicted, velocity


def _pair(origin, target, prediction, velocity, frames, points, feature_mode, flow, config, span):
    a, b = points[origin], points[target]
    support = np.array([1]) if feature_mode == "pectoral" else np.array([0, 1])
    step = b - a
    tail_veto = None
    if np.any(np.linalg.norm(step[support], axis=-1) > config.speed_fraction * span):
        return np.inf, "large_primary_step", {}
    if feature_mode == "nodes":
        old_axis, new_axis = a[0] - a[1], b[0] - b[1]
        if np.linalg.norm(new_axis - old_axis) > config.mouth_relative_motion_fraction * span:
            return np.inf, "mouth_pectoral_motion_disagrees", {}
        if abs(np.linalg.norm(new_axis) - np.linalg.norm(old_axis)) > (
            config.length_change_fraction * span
        ):
            return np.inf, "primary_length_changes", {}
        if (
            min(np.linalg.norm(old_axis), np.linalg.norm(new_axis))
            > (config.axis_min_length_fraction * span)
            and _angle(old_axis, new_axis) > config.axis_change_deg
        ):
            return np.inf, "body_axis_disagrees", {}
        if np.isfinite(a[2]).all() and np.isfinite(b[2]).all():
            # Tail is a cohesion veto only: its score can never select a fish.
            relative = (b[2] - b[1]) - (a[2] - a[1])
            if np.linalg.norm(relative) > config.tail_relative_motion_fraction * span:
                tail_veto = "tail_pectoral_motion_disagrees"
    expected_step = velocity[1] * (frames[target] - frames[origin])
    if (
        np.isfinite(expected_step).all()
        and np.linalg.norm(expected_step) >= config.heading_speed_fraction * span
        and np.linalg.norm(step[1]) >= config.heading_step_fraction * span
        and np.linalg.norm(step[1] - expected_step) >= config.heading_error_fraction * span
        and _angle(expected_step, step[1]) >= config.heading_change_deg
    ):
        return np.inf, "motion_heading_disagrees", {}
    prediction_errors = np.linalg.norm(b[support] - prediction[support], axis=-1)
    if np.any(prediction_errors > config.prediction_fraction * span):
        return np.inf, "motion_prediction_error", {}
    weights = np.ones(len(support))
    weights[support == 1] = 2
    motion_cost = float(np.average(prediction_errors, weights=weights))
    details = {"motion_error_px": motion_cost, "flow_nodes": [], "tail_veto": tail_veto}
    if not flow:
        return motion_cost, None, details
    direct = "forward" if frames[target] > frames[origin] else "backward"
    inverse = "backward" if direct == "forward" else "forward"
    # A validity mask from only one direction is insufficient positive evidence.
    both = flow[f"valid_{direct}"][origin] & flow[f"valid_{inverse}"][target]
    flow_nodes = support[both[support]]
    if not len(flow_nodes):
        return motion_cost, None, details
    direct_error = np.linalg.norm(
        flow[f"predicted_{direct}"][origin, flow_nodes] - b[flow_nodes], axis=-1
    )
    inverse_error = np.linalg.norm(
        flow[f"predicted_{inverse}"][target, flow_nodes] - a[flow_nodes], axis=-1
    )
    if max(direct_error.max(), inverse_error.max()) > config.flow_error_fraction * span:
        return np.inf, "bidirectional_flow_disagrees", {}
    flow_weights = np.where(flow_nodes == 1, 2.0, 1.0)
    flow_cost = float(np.average((direct_error + inverse_error) / 2, weights=flow_weights))
    details.update(flow_nodes=flow_nodes.tolist(), flow_error_px=flow_cost)
    # Flow can rank already motion-admissible edges, never override their gates.
    weight = config.flow_prediction_weight
    return (1 - weight) * motion_cost + weight * flow_cost, None, details


def _directional(frames, points, groups, reasons, feature_mode, flow, config, span, reverse):
    ordered = list(reversed(groups)) if reverse else groups
    histories, previous_frame, edges, evidence, events = {}, None, set(), {}, []
    direction = "backward" if reverse else "forward"
    gate = config.prediction_fraction * span
    for frame, current in ordered:
        if previous_frame is None or abs(frame - previous_frame) != 1:
            histories = {}
        origins = sorted(histories)
        n, m = len(origins), len(current)
        costs = np.full((n, m + n), 1e12)
        pair_info = {}
        for i, origin in enumerate(origins):
            costs[i, m + i] = gate
            if any(reason != "unsafe_tail_shape" for reason in reasons[origin]):
                continue
            prediction, velocity = _prediction(histories[origin], frame, frames, points, config)
            for j, target in enumerate(current):
                if any(reason != "unsafe_tail_shape" for reason in reasons[target]):
                    continue
                value, reason, details = _pair(
                    origin,
                    target,
                    prediction,
                    velocity,
                    frames,
                    points,
                    feature_mode,
                    flow,
                    config,
                    span,
                )
                pair_info[(i, j)] = (value, reason, details)
                if np.isfinite(value):
                    costs[i, j] = value
        continued, assigned = {}, {}
        if n:
            row_ind, col_ind = linear_sum_assignment(costs)
            best = float(costs[row_ind, col_ind].sum())
            for i, j in zip(row_ind, col_ind, strict=True):
                if j >= m or costs[i, j] >= 1e12:
                    continue
                alternate = costs.copy()
                alternate[i, j] = 1e12
                ai, aj = linear_sum_assignment(alternate)
                margin = float(alternate[ai, aj].sum() - best)
                origin, target = origins[i], int(current[j])
                if margin < config.assignment_margin_fraction * span:
                    assigned[target] = (origin, "assignment_margin", margin)
                    continue
                tail_veto = pair_info[(i, j)][2]["tail_veto"]
                if "unsafe_tail_shape" in reasons[origin] + reasons[target]:
                    tail_veto = "unsafe_tail_shape"
                if tail_veto:
                    # A bad tail may cut the primary-selected edge, but must not
                    # remove its competitors and manufacture primary confidence.
                    assigned[target] = (origin, tail_veto, margin)
                    continue
                edge = (min(origin, target), max(origin, target))
                edges.add(edge)
                evidence[edge] = dict(
                    score_px=float(costs[i, j]), margin_px=margin, **pair_info[(i, j)][2]
                )
                continued[target] = [*histories[origin], target][-config.history_observations :]
        for j, target in enumerate(current):
            target = int(target)
            if target in continued or not origins:
                continue
            if reasons[target]:
                why, origin = reasons[target][0], None
            elif target in assigned:
                origin, why, _ = assigned[target]
            else:
                distances = np.linalg.norm(points[origins, 1] - points[target, 1], axis=-1)
                origin = None
                why = "no_safe_candidate"
                if np.isfinite(distances).any():
                    i = int(np.nanargmin(distances))
                    origin = origins[i]
                    why = (
                        reasons[origin][0]
                        if reasons[origin]
                        else pair_info.get((i, j), (None, None, {}))[1] or "one_to_one_competition"
                    )
            events.append(
                dict(
                    kind="directional_link_refused",
                    direction=direction,
                    reason=why,
                    from_row=origin,
                    to_row=target,
                    from_frame=int(frames[origin]) if origin is not None else None,
                    to_frame=frame,
                )
            )
        # All detections enter exactly one new history; stale competitors cannot persist.
        histories = {int(row): continued.get(int(row), [int(row)]) for row in current}
        previous_frame = frame
    return edges, evidence, events


def build_tracklets(frame_idx, points, *, view, feature_mode="nodes", flow=None, config=None):
    """Build independent view tracklets; all rows survive, including isolated rows.

    ``clean_edge[row]`` is true only for an agreed link from the previous native
    frame into this row. No gap filling and no deferred identity guesses occur.
    Tail is ignored in pectoral mode and is only a veto in nodes mode. LK evidence
    requires validity in both directions and cannot rescue motion/shape failures.
    """
    config = config or TrackingConfig()
    frames, xy, span, held_flow = _inputs(frame_idx, points, view, feature_mode, flow, config)
    unique, starts, lengths = np.unique(frames, return_index=True, return_counts=True)
    groups = [
        (int(f), np.arange(s, s + n)) for f, s, n in zip(unique, starts, lengths, strict=True)
    ]
    reasons = _row_quality(xy, groups, feature_mode, config, span)
    forward, f_info, f_events = _directional(
        frames, xy, groups, reasons, feature_mode, held_flow, config, span, False
    )
    backward, b_info, b_events = _directional(
        frames, xy, groups, reasons, feature_mode, held_flow, config, span, True
    )
    agreed = forward & backward
    parent = np.arange(len(frames), dtype=np.int64)

    def find(row):
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = parent[row]
        return int(row)

    clean_edge = np.zeros(len(frames), dtype=bool)
    events = f_events + b_events
    for row, row_reasons in enumerate(reasons):
        if row_reasons:
            events.append(
                dict(kind="unsafe_detection", row=row, frame=int(frames[row]), reasons=row_reasons)
            )
    for a, b in sorted(agreed):
        parent[find(b)] = find(a)
        clean_edge[b] = True
        events.append(
            dict(
                kind="clean_link",
                from_row=a,
                to_row=b,
                from_frame=int(frames[a]),
                to_frame=int(frames[b]),
                forward=f_info[(a, b)],
                backward=b_info[(a, b)],
            )
        )
    for a, b in sorted(forward ^ backward):
        events.append(
            dict(
                kind="directional_disagreement",
                from_row=a,
                to_row=b,
                from_frame=int(frames[a]),
                to_frame=int(frames[b]),
            )
        )
    roots = np.array([find(row) for row in range(len(frames))], dtype=np.int64)
    labels = np.unique(roots, return_inverse=True)[1].astype(np.int64)
    for _, rows in groups:
        if len(np.unique(labels[rows])) != len(rows):
            raise AssertionError("A tracklet cannot contain two detections in one frame")
    counts = np.unique(labels, return_counts=True)[1]
    metadata = dict(
        algorithm_revision="clean_tracklets_v1",
        view=view,
        feature_mode=feature_mode,
        config=asdict(config),
        body_scale_px=span,
        input_detections=len(frames),
        retained_detections=len(frames),
        output_tracklets=len(counts),
        isolated_detections=int(np.count_nonzero(counts == 1)),
        clean_links=len(agreed),
        forward_links=len(forward),
        backward_links=len(backward),
        disagreement_links=len(forward ^ backward),
        unsafe_detections=sum(bool(r) for r in reasons),
        unsafe_reasons=dict(Counter(reason for held in reasons for reason in held)),
        refusal_reasons=dict(Counter(e["reason"] for e in f_events + b_events)),
        flow_supplied=flow is not None,
        clean_links_with_flow=sum(bool(f_info[e]["flow_nodes"]) for e in agreed),
        gap_links=0,
        sleap_tracking_ids_used=False,
        tail_used_for_identity=False,
        detector_instance_grouping_retained=True,
        identity_accuracy_claim=False,
        quality_definition=(
            "adjacent native, safe support, competition margin, bidirectional agreement"
        ),
    )
    return TrackletResult(labels, clean_edge, events, metadata)

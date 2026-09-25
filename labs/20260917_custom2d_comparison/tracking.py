"""Detection-preserving experimental 2D association, with no 3D/identity input.

Thresholds were fixed before inspecting these arms' results. Pixel scales are
twice the historical C24b gate, not fitted to the reviewed scenes. All time
differences use native absolute frame indices. Tail coordinates never affect ID.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TrackingConfig:
    top_body_span_px: float = 120.0
    side_body_span_px: float = 100.0
    hard_speed_fraction: float = 0.5
    prediction_error_fraction: float = 0.20
    assignment_margin_fraction: float = 0.04
    history_observations: int = 5
    velocity_min_intervals: int = 2
    max_missing_frames: int = 3
    gap_error_growth: float = 0.25
    pectoral_weight: float = 2.0
    heading_min_speed_fraction: float = 0.015
    lateral_min_speed_fraction: float = 0.08
    lateral_error_fraction: float = 0.10
    lateral_angle_deg: float = 80.0
    consistent_flank_angle_deg: float = 45.0

    def __post_init__(self):
        positive = (
            self.top_body_span_px,
            self.side_body_span_px,
            self.hard_speed_fraction,
            self.prediction_error_fraction,
            self.assignment_margin_fraction,
            self.pectoral_weight,
            self.heading_min_speed_fraction,
            self.lateral_min_speed_fraction,
            self.lateral_error_fraction,
        )
        if not all(np.isfinite(x) and x > 0 for x in positive):
            raise ValueError("Scales, gates and weights must be finite and positive")
        if (
            self.history_observations < 3
            or not 1 <= self.velocity_min_intervals < self.history_observations
        ):
            raise ValueError("Velocity support must fit inside the history")
        if self.max_missing_frames < 0 or self.gap_error_growth < 0:
            raise ValueError("Gap settings must not be negative")
        if not 0 < self.lateral_angle_deg <= 180 or not 0 < self.consistent_flank_angle_deg <= 180:
            raise ValueError("Angles must be in (0, 180]")

    def span(self, view):
        if view not in ("top_cam", "side1_cam", "side2_cam"):
            raise ValueError(f"Unknown camera: {view}")
        return self.top_body_span_px if view == "top_cam" else self.side_body_span_px


@dataclass
class TrackingResult:
    labels: np.ndarray
    events: list[dict]
    metadata: dict


def _inputs(frame_idx, points, view, config):
    config = config or TrackingConfig()
    config.span(view)
    frames = np.asarray(frame_idx)
    xy = np.asarray(points, dtype=float)
    if frames.ndim != 1 or frames.dtype.kind not in "iu":
        raise ValueError("frame_idx must be an integer vector")
    if xy.shape != (len(frames), 3, 2):
        raise ValueError("points must have shape (N, 3, 2), ordered mouth/pectoral/tail")
    if np.any(frames < 0) or np.any(np.diff(frames.astype(np.int64)) < 0):
        raise ValueError("Native absolute frames must be nonnegative and sorted")
    if np.isinf(xy).any():
        raise ValueError("Missing points must be NaN, not infinity")
    return frames.astype(np.int64, copy=False), xy, config


def _weights(config):
    return np.array([1.0, config.pectoral_weight])


def _error(a, b, config):
    distance = np.linalg.norm(a[:2] - b[:2], axis=-1)
    live = np.isfinite(distance)
    if not live.any():
        return np.inf
    return float(np.average(distance[live], weights=_weights(config)[live]))


def _vector(points, config):
    live = np.isfinite(points).all(axis=-1)
    if not live.any():
        return np.full(2, np.nan)
    return np.average(points[live], axis=0, weights=_weights(config)[live])


def _velocity(rows, frames, points, config, span):
    """Per-node median finite-difference velocity; the tested edge is absent."""
    velocity = np.full((2, 2), np.nan)
    if len(rows) < 2:
        return velocity
    rows = np.asarray(rows[-config.history_observations :], dtype=np.int64)
    dt = np.diff(frames[rows])
    usable_time = (np.abs(dt) > 0) & (np.abs(dt) <= config.max_missing_frames + 1)
    diff = np.diff(points[rows, :2], axis=0)
    values = np.full_like(diff, np.nan)
    np.divide(diff, dt[:, None, None], out=values, where=usable_time[:, None, None])
    for node in range(2):
        good = np.isfinite(values[:, node]).all(axis=-1)
        good &= np.linalg.norm(values[:, node], axis=-1) <= span * config.hard_speed_fraction
        if good.sum() >= config.velocity_min_intervals:
            velocity[node] = np.median(values[good, node], axis=0)
    return velocity


def _prediction(row, target_frame, history, frames, points, config, span):
    velocity = _velocity(history, frames, points, config, span)
    predicted = points[row, :2].copy()
    supported = np.isfinite(velocity).all(axis=-1)
    predicted[supported] += velocity[supported] * (target_frame - frames[row])
    return predicted, velocity


def _angle(a, b, minimum):
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    norms = np.linalg.norm(a), np.linalg.norm(b)
    if min(norms) < minimum:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / np.prod(norms), -1, 1))))


def _groups(frames):
    unique, first, count = np.unique(frames, return_index=True, return_counts=True)
    return [(int(f), np.arange(a, a + n)) for f, a, n in zip(unique, first, count, strict=True)]


def _labels_from_edges(n, edges):
    parent = np.arange(n, dtype=np.int64)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in sorted(edges):
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)
    roots = [find(i) for i in range(n)]
    return np.unique(roots, return_inverse=True)[1].astype(np.int64)


def _result(frames, labels, events, config, view, **extra):
    for _, rows in _groups(frames):
        if len(np.unique(labels[rows])) != len(rows):
            raise AssertionError("A result label occurs twice in one frame")
    if labels.shape != frames.shape or labels.dtype != np.int64 or np.any(labels < 0):
        raise AssertionError("Every original detection must receive a nonnegative int64 ID")
    lengths = np.unique(labels, return_counts=True)[1]
    metadata = dict(
        config=asdict(config),
        view=view,
        body_span_px=config.span(view),
        input_detections=len(frames),
        retained_detections=len(labels),
        output_tracklets=len(lengths),
        singleton_tracklets=int((lengths == 1).sum()),
        fraction_detections_in_tracklets_ge_5=float(lengths[lengths >= 5].sum() / len(labels))
        if len(labels)
        else 0.0,
        event_counts=dict(Counter(e["kind"] for e in events)),
        identity_accuracy_claim=False,
        tail_used_for_identity=False,
        **extra,
    )
    return TrackingResult(labels, events, metadata)


def cut_sleap_links(frame_idx, points, baseline_labels, *, view, config=None):
    """B: split original ID paths using independent past and future evidence.

    This never merges distinct baseline IDs. Negative/duplicate input labels are
    retained as detections but isolated rather than silently omitted or joined.
    Ordinary turns are not rejected by angle alone: the two flank headings must
    agree, the step must be large, and both independent predictions must err.
    """
    frames, xy, config = _inputs(frame_idx, points, view, config)
    baseline = np.asarray(baseline_labels)
    if baseline.shape != frames.shape or baseline.dtype.kind not in "iu":
        raise ValueError("baseline_labels must be an integer vector parallel to frame_idx")
    span = config.span(view)
    gate, margin = span * config.prediction_error_fraction, span * config.assignment_margin_fraction
    groups = {f: rows for f, rows in _groups(frames)}
    paths = {}
    for row, label in enumerate(baseline):
        if label >= 0:
            paths.setdefault(int(label), []).append(row)
    edges, events, total = set(), [], 0
    for label, rows in paths.items():
        duplicated_frames = {f for f, n in Counter(frames[rows]).items() if n > 1}
        for k, (i, j) in enumerate(pairwise(rows)):
            total += 1
            dt = int(frames[j] - frames[i])
            reasons = []
            detail = dict(
                c27_label=label,
                from_row=i,
                to_row=j,
                from_frame=int(frames[i]),
                to_frame=int(frames[j]),
            )
            if frames[i] in duplicated_frames or frames[j] in duplicated_frames:
                reasons.append("duplicate_baseline_label_in_frame")
            elif dt > config.max_missing_frames + 1:
                reasons.append("gap_too_long")
            else:
                displacement = xy[j, :2] - xy[i, :2]
                raw_speed = np.linalg.norm(displacement, axis=-1) / dt
                if not np.isfinite(raw_speed).any():
                    reasons.append("missing_primary_points")
                elif np.any(raw_speed[np.isfinite(raw_speed)] > span * config.hard_speed_fraction):
                    reasons.append("hard_speed_gate")
                left = rows[max(0, k + 1 - config.history_observations) : k + 1]
                right = rows[k + 1 : k + 1 + config.history_observations][::-1]
                forward, vl = _prediction(i, frames[j], left, frames, xy, config, span)
                backward, vr = _prediction(j, frames[i], right, frames, xy, config, span)
                # Audit only nodes with independently estimated flank velocity.
                forward[~np.isfinite(vl).all(axis=-1)] = np.nan
                backward[~np.isfinite(vr).all(axis=-1)] = np.nan
                fe, re = _error(forward, xy[j], config), _error(backward, xy[i], config)
                have_f, have_r = np.isfinite(fe), np.isfinite(re)
                error_gate = gate * (1 + config.gap_error_growth * (dt - 1))
                if have_f and have_r and min(fe, re) > error_gate:
                    reasons.append("both_flank_prediction_errors")
                motion = _vector(displacement / dt, config)
                lv, rv = _vector(vl, config), _vector(vr, config)
                minimum = span * config.heading_min_speed_fraction
                la, ra = _angle(motion, lv, minimum), _angle(motion, rv, minimum)
                agreement = _angle(lv, rv, minimum)
                if (
                    agreement is not None
                    and agreement <= config.consistent_flank_angle_deg
                    and la is not None
                    and ra is not None
                    and max(la, ra) >= config.lateral_angle_deg
                    and np.linalg.norm(motion) >= span * config.lateral_min_speed_fraction
                    and have_f
                    and have_r
                    and min(fe, re) >= span * config.lateral_error_fraction
                ):
                    reasons.append("lateral_jump_between_consistent_flanks")
                competing = {}
                for side, prediction, current, candidates, error in (
                    ("forward", forward, j, groups[int(frames[j])], fe),
                    ("reverse", backward, i, groups[int(frames[i])], re),
                ):
                    alternatives = [
                        _error(prediction, xy[row], config) for row in candidates if row != current
                    ]
                    best = min(alternatives, default=np.inf)
                    competing[side] = float(best) if np.isfinite(best) else None
                    if np.isfinite(error) and error > error_gate and best + margin < error:
                        reasons.append(f"{side}_prediction_prefers_other_detection")
                detail.update(
                    node_speed_px_per_native_frame=[
                        float(x) if np.isfinite(x) else None for x in raw_speed
                    ],
                    forward_prediction_error_px=fe if have_f else None,
                    reverse_prediction_error_px=re if have_r else None,
                    forward_angle_deg=la,
                    reverse_angle_deg=ra,
                    flank_heading_angle_deg=agreement,
                    competing_detection_error_px=competing,
                )
            if reasons:
                events.append(dict(kind="cut", reasons=reasons, **detail))
            else:
                edges.add((i, j))
    labels = _labels_from_edges(len(frames), edges)
    return _result(
        frames,
        labels,
        events,
        config,
        view,
        arm="B",
        baseline_candidate_links=total,
        retained_links=len(edges),
        cut_links=total - len(edges),
        baseline_ids_used=True,
    )


def _directional(frames, xy, config, view, reverse):
    span = config.span(view)
    gate, margin = span * config.prediction_error_fraction, span * config.assignment_margin_fraction
    histories, edges, events = [], set(), []
    groups = _groups(frames)
    if reverse:
        groups.reverse()
    direction = "reverse" if reverse else "forward"
    for frame, current in groups:
        histories = [
            h for h in histories if abs(frame - frames[h[-1]]) <= config.max_missing_frames + 1
        ]
        accepted = set()
        if histories:
            n, m = len(histories), len(current)
            blocked = 1e12
            costs = np.full((n, m + n), blocked)
            raw = np.full((n, m), np.inf)
            for h_index, history in enumerate(histories):
                previous = history[-1]
                dt = abs(frame - frames[previous])
                prediction, _ = _prediction(previous, frame, history, frames, xy, config, span)
                limit = gate * (1 + config.gap_error_growth * (dt - 1))
                costs[h_index, m + h_index] = limit
                for c, row in enumerate(current):
                    speed = np.linalg.norm(xy[row, :2] - xy[previous, :2], axis=-1) / dt
                    error = _error(prediction, xy[row], config)
                    raw[h_index, c] = error
                    if (
                        np.isfinite(speed).any()
                        and np.all(speed[np.isfinite(speed)] <= span * config.hard_speed_fraction)
                        and error <= limit
                    ):
                        costs[h_index, c] = error
            row_ind, col_ind = linear_sum_assignment(costs)
            optimum = float(costs[row_ind, col_ind].sum())
            continued_histories = set()
            for h_index, col in zip(row_ind, col_ind, strict=True):
                if col >= m:
                    continue
                previous, row = histories[h_index][-1], int(current[col])
                alternate = costs.copy()
                alternate[h_index, col] = blocked
                a, b = linear_sum_assignment(alternate)
                advantage = float(alternate[a, b].sum() - optimum)
                if advantage < margin:
                    events.append(
                        dict(
                            kind="ambiguous_directional_assignment",
                            direction=direction,
                            from_row=previous,
                            to_row=row,
                            from_frame=int(frames[previous]),
                            to_frame=frame,
                            prediction_error_px=float(raw[h_index, col]),
                            global_assignment_margin_px=advantage,
                        )
                    )
                    continue
                edges.add((min(previous, row), max(previous, row)))
                histories[h_index].append(row)
                histories[h_index] = histories[h_index][-config.history_observations :]
                continued_histories.add(h_index)
                accepted.add(row)
            remaining = []
            for h_index, history in enumerate(histories):
                admissible = costs[h_index, :m] < blocked
                if h_index in continued_histories or not admissible.any():
                    # Keep genuinely unobserved tracks briefly, but do not let
                    # a refused track compete with its own newly born successor.
                    remaining.append(history)
                else:
                    previous = history[-1]
                    events.append(
                        dict(
                            kind="ended_competing_history",
                            direction=direction,
                            from_row=previous,
                            from_frame=int(frames[previous]),
                            to_frame=frame,
                            admissible_detection_rows=current[admissible].tolist(),
                            reason="unmatched_history_has_admissible_current_detection",
                        )
                    )
            histories = remaining
        histories.extend([[int(row)] for row in current if row not in accepted])
    return edges, events


def track_bidirectional(frame_idx, points, *, view, config=None):
    """C: retain links independently accepted in both temporal directions.

    Hungarian assignment includes explicit no-match choices. A link must beat
    the globally best assignment with that link forbidden by a fixed margin.
    Gaps are limited in native frames. Disagreements become separate tracklets,
    never forced identities or discarded detections; there is no SLEAP ID input.
    An unmatched history with an admissible current detection is ended, because
    retaining it beside newly born histories would create duplicate hypotheses
    for the same observation. Only histories without admissible current detections
    remain active across a short gap.
    """
    frames, xy, config = _inputs(frame_idx, points, view, config)
    forward, forward_events = _directional(frames, xy, config, view, False)
    backward, backward_events = _directional(frames, xy, config, view, True)
    agreed = forward & backward
    events = forward_events + backward_events
    for i, j in sorted(forward ^ backward):
        events.append(
            dict(
                kind="bidirectional_disagreement",
                from_row=i,
                to_row=j,
                from_frame=int(frames[i]),
                to_frame=int(frames[j]),
                accepted_forward=(i, j) in forward,
                accepted_reverse=(i, j) in backward,
            )
        )
    labels = _labels_from_edges(len(frames), agreed)
    return _result(
        frames,
        labels,
        events,
        config,
        view,
        arm="C",
        algorithm_revision="v2_retire_unmatched_competing_histories",
        baseline_ids_used=False,
        forward_links=len(forward),
        reverse_links=len(backward),
        retained_links=len(agreed),
        disagreement_links=len(forward ^ backward),
        ambiguous_directional_assignments=sum(
            e["kind"] == "ambiguous_directional_assignment" for e in events
        ),
        ended_competing_histories=sum(e["kind"] == "ended_competing_history" for e in events),
    )

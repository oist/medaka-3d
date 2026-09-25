"""Variable-count, P-primary native-frame tracklets without temporal input IDs.

The declared scales reuse the frozen P0 experiment's physical pixel references;
this implementation has no lab imports. Directional provisional histories inform
competition, but only mutual adjacent links supported by both passes are joined.
Missing-frame relations remain candidate records and predictions never edit data.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, replace

import numpy as np


@dataclass(frozen=True)
class P1Config:
    native_fps: float = 166.0
    top_body_px: float = 120.0
    side_body_px: float = 100.0
    step_fraction: float = 0.30
    prediction_fraction: float = 0.12
    margin_fraction: float = 0.025
    relative_margin: float = 0.15
    endpoint_lifetime_seconds: float = 0.10
    uncertainty_fraction: float = 0.01
    uncertainty_growth_fraction: float = 0.005
    history_observations: int = 4
    heading_change_deg: float = 80.0
    heading_min_step_fraction: float = 0.02
    max_mouth_span_fraction: float = 0.80
    max_tail_span_fraction: float = 1.80
    mouth_relative_motion_fraction: float = 0.12
    tail_relative_motion_fraction: float = 0.25
    auxiliary_bonus_fraction: float = 0.50

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("All P1 configuration values must be positive and finite")
        if not isinstance(self.history_observations, int) or self.history_observations < 2:
            raise ValueError("history_observations must be an integer of at least two")
        if self.heading_change_deg > 180 or self.auxiliary_bonus_fraction > 1:
            raise ValueError("Invalid heading angle or auxiliary bonus fraction")

    def span(self, view):
        if view not in ("top_cam", "side1_cam", "side2_cam"):
            raise ValueError(f"Unknown view: {view}")
        return self.top_body_px if view == "top_cam" else self.side_body_px

    @property
    def endpoint_lifetime_frames(self):
        return math.ceil(self.endpoint_lifetime_seconds * self.native_fps)


def sensitivity_configs():
    """Prespecified one-parameter sensitivities; never select by scene results."""
    base = P1Config()
    return {
        "P1_margin_loose": replace(base, margin_fraction=0.015),
        "P1_margin_strict": replace(base, margin_fraction=0.035),
    }


@dataclass
class P1Result:
    labels: np.ndarray
    edges: list[dict]
    events: list[dict]
    predictions: list[dict]
    reconnect_candidates: list[dict]
    node_quality: list[dict]
    metadata: dict


def _angle(first, second):
    product = float(np.linalg.norm(first) * np.linalg.norm(second))
    if product <= 1e-12:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second) / product, -1, 1))))


def _quality(frames, points, rows, span, config):
    quality = []
    for frame, nodes, row in zip(frames, points, rows, strict=True):
        observed = np.isfinite(nodes).all(axis=1)
        flags = []
        for node, name, limit in (
            (0, "mouth", config.max_mouth_span_fraction),
            (2, "tail", config.max_tail_span_fraction),
        ):
            if observed[1] and observed[node]:
                distance = float(np.linalg.norm(nodes[node] - nodes[1]))
                if distance > limit * span:
                    flags.append(f"{name}_span_suspect")
        quality.append({
            "global_row": int(row), "row": int(row), "frame": int(frame),
            "observed_mask": observed.tolist(),
            "missing_nodes": [name for name, ok in zip("MPT", observed, strict=True) if not ok],
            "geometry_flags": flags,
            "temporal_cohesion_flags": [],
            "reflection_status": "unknown", "reflection_evidence": [],
            "node_scores": None,
            "node_score_source": "not_supplied; no synthetic node scores",
        })
    return quality


def _predict(history, target_frame, frames, points, config):
    selected = np.asarray(history[-config.history_observations:], dtype=np.int64)
    velocity = np.zeros(2, dtype=float)
    if len(selected) >= 2:
        delta = np.diff(frames[selected]).astype(float)
        velocity = np.median(np.diff(points[selected, 1], axis=0) / delta[:, None], axis=0)
    origin = int(selected[-1])
    predicted = points[origin, 1] + velocity * (target_frame - frames[origin])
    return predicted, velocity


def _auxiliary(origin, target, points, span, config):
    """Coherent M/T adds bounded support; missing/bad nodes do not veto P."""
    supported, flags, values = [], [], []
    for node, name, size, motion in (
        (0, "M", config.max_mouth_span_fraction, config.mouth_relative_motion_fraction),
        (2, "T", config.max_tail_span_fraction, config.tail_relative_motion_fraction),
    ):
        a, b = points[origin, node], points[target, node]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            continue
        first, second = a - points[origin, 1], b - points[target, 1]
        change = float(np.linalg.norm(second - first))
        if max(np.linalg.norm(first), np.linalg.norm(second)) > size * span:
            flags.append(f"{name}_span_suspect")
        elif change > motion * span:
            flags.append(f"{name}_relative_motion_suspect")
        else:
            supported.append(name)
            values.append(max(0.0, 1.0 - change / (motion * span)))
    return (float(np.mean(values)) if values else 0.0), supported, flags


def _record(origin, target, frames, rows):
    if frames[origin] > frames[target]:
        origin, target = target, origin
    return {
        "from_global_row": int(rows[origin]), "to_global_row": int(rows[target]),
        "from_row": int(rows[origin]), "to_row": int(rows[target]),
        "from_frame": int(frames[origin]), "to_frame": int(frames[target]),
    }


def _directional(frames, points, rows, groups, span, config, auxiliary, reverse):
    active, edges, evidence, events, predictions = {}, set(), {}, [], []
    direction = "backward" if reverse else "forward"
    ordered = groups[::-1] if reverse else groups
    valid = np.isfinite(points[:, 1]).all(axis=1)
    gate = config.prediction_fraction * span
    lifetime = config.endpoint_lifetime_frames
    for frame, current in ordered:
        active = {
            origin: history for origin, history in active.items()
            if abs(frame - frames[origin]) <= lifetime
        }
        origins = sorted(active, key=lambda origin: int(rows[origin]))
        predicted, velocities, ages, uncertainties = [], [], [], []
        for origin in origins:
            prediction, velocity = _predict(active[origin], frame, frames, points, config)
            age = abs(int(frame - frames[origin]))
            uncertainty = span * (
                config.uncertainty_fraction + config.uncertainty_growth_fraction * (age - 1)
            )
            predicted.append(prediction)
            velocities.append(velocity)
            ages.append(age)
            uncertainties.append(uncertainty)
            if age > 1:
                predictions.append({
                    "from_global_row": int(rows[origin]), "from_row": int(rows[origin]),
                    "from_frame": int(frames[origin]), "target_frame": int(frame),
                    "direction": direction, "position": prediction.tolist(),
                    "uncertainty_px": float(uncertainty), "age_frames": age,
                    "age_seconds": age / config.native_fps,
                    "kind": "unobserved_prediction", "observed": False,
                    "history_scope": "provisional_directional",
                })
        n, m = len(origins), len(current)
        costs = np.full((n, m), np.inf)
        errors = np.full((n, m), np.inf)
        steps = np.full((n, m), np.inf)
        aux_info = {}
        if n and m:
            errors = np.linalg.norm(
                points[current, 1][None, :, :] - np.asarray(predicted)[:, None, :], axis=-1
            )
            steps = np.linalg.norm(
                points[current, 1][None, :, :] - points[origins, 1][:, None, :], axis=-1
            )
            for i, origin in enumerate(origins):
                for j, target in enumerate(current):
                    if not valid[target]:
                        continue
                    age = ages[i]
                    extra = uncertainties[i] if age > 1 else 0.0
                    if (steps[i, j] > config.step_fraction * span * age
                            or errors[i, j] > gate + extra):
                        continue
                    # An unseen endpoint competes at the nearest edge of its
                    # uncertainty radius; uncertainty must not manufacture safety.
                    costs[i, j] = max(0.0, errors[i, j] - extra)
                    if age == 1 and auxiliary:
                        aux_info[i, j] = _auxiliary(origin, target, points, span, config)
                        support = aux_info[i, j][0]
                        costs[i, j] -= (
                            config.auxiliary_bonus_fraction
                            * config.margin_fraction * span * support
                        )
        continued, consumed = {}, set()
        for j, target in enumerate(current):
            candidates = []
            incoming = costs[:, j]
            for i, origin in enumerate(origins):
                key = (min(origin, int(target)), max(origin, int(target)))
                if not np.isfinite(incoming[i]):
                    reason = (
                        "missing_pectoral" if not valid[target] else
                        "large_step" if steps[i, j] > config.step_fraction * span * ages[i]
                        else "prediction_gate"
                    )
                    info = {
                        "from_global_row": int(rows[origin]),
                        "to_global_row": int(rows[target]),
                        "prediction_error_px": (float(errors[i, j])
                                                if np.isfinite(errors[i, j]) else None),
                        "step_px": float(steps[i, j]) if np.isfinite(steps[i, j]) else None,
                        "score_px": None, "origin_margin_px": None, "target_margin_px": None,
                        "required_margin_px": config.margin_fraction * span,
                        "origin_rank": None, "target_rank": None,
                        "heading_change_deg": None, "heading_review": False,
                        "uncertainty_px": float(uncertainties[i]), "age_frames": ages[i],
                        "competing_endpoints": [int(rows[origins[k]]) for k in range(n)
                                                if k != i and np.isfinite(incoming[k])],
                        "auxiliary_support": 0.0, "observed_support_nodes": [],
                        "node_cohesion_flags": [], "reason": reason,
                    }
                    candidates.append(info)
                    if ages[i] == 1:
                        evidence[key] = info
                    continue
                own = float(costs[i, j])
                outgoing = costs[i]
                other_targets = np.delete(outgoing, j)
                other_origins = np.delete(incoming, i)
                origin_margin = min(gate, float(np.min(other_targets, initial=np.inf))) - own
                target_margin = min(gate, float(np.min(other_origins, initial=np.inf))) - own
                origin_rank = 1 + int(np.count_nonzero(outgoing < own - 1e-9))
                target_rank = 1 + int(np.count_nonzero(incoming < own - 1e-9))
                primary_tie = (
                    np.count_nonzero(np.isfinite(outgoing)
                                     & (np.abs(errors[i] - errors[i, j]) <= 1e-9)) > 1
                    or np.count_nonzero(np.isfinite(incoming)
                                        & (np.abs(errors[:, j] - errors[i, j]) <= 1e-9)) > 1
                )
                minimum_margin = config.margin_fraction * span + (
                    config.relative_margin * max(0.0, own)
                )
                expected = velocities[i] * (frame - frames[origin])
                observed_step = points[target, 1] - points[origin, 1]
                angle = _angle(expected, observed_step)
                heading_review = bool(
                    angle is not None and angle >= config.heading_change_deg
                    and min(np.linalg.norm(expected), np.linalg.norm(observed_step))
                    >= config.heading_min_step_fraction * span
                )
                support, support_nodes, flags = aux_info.get((i, j), (0.0, [], []))
                competing = [int(rows[origins[k]]) for k in range(n)
                             if k != i and np.isfinite(incoming[k])]
                if ages[i] > 1:
                    reason = "missing_frame_candidate_only"
                elif primary_tie:
                    reason = "primary_distance_tie"
                elif origin_rank != 1 or target_rank != 1:
                    reason = "nonmutual_candidate"
                elif min(origin_margin, target_margin) < minimum_margin:
                    reason = "candidate_margin"
                else:
                    reason = "directional_agreement"
                info = {
                    "from_global_row": int(rows[origin]), "to_global_row": int(rows[target]),
                    "prediction_error_px": float(errors[i, j]),
                    "step_px": float(steps[i, j]), "score_px": own,
                    "origin_margin_px": float(origin_margin),
                    "target_margin_px": float(target_margin),
                    "required_margin_px": float(minimum_margin),
                    "origin_rank": origin_rank, "target_rank": target_rank,
                    "heading_change_deg": angle, "heading_review": heading_review,
                    "uncertainty_px": float(uncertainties[i]),
                    "age_frames": ages[i], "competing_endpoints": competing,
                    "auxiliary_support": support, "observed_support_nodes": support_nodes,
                    "node_cohesion_flags": flags, "reason": reason,
                }
                candidates.append(info)
                if ages[i] == 1:
                    evidence[key] = info
                if reason == "directional_agreement":
                    edges.add(key)
                    continued[int(target)] = [*active[origin], int(target)][
                        -config.history_observations:
                    ]
                    consumed.add(origin)
                elif (ages[i] > 1 and not primary_tie and origin_rank == target_rank == 1
                      and min(origin_margin, target_margin) >= minimum_margin):
                    # The old hypothesis is now represented by this new observed
                    # endpoint. Retire its redundant forecast without transferring
                    # the history or identity. This avoids a gap hypothesis forever
                    # competing with its own new fragment; ambiguous ghosts persist.
                    consumed.add(origin)
                    info["forecast_retired_as_represented"] = True
            if int(target) not in continued:
                if not valid[target]:
                    reason = "missing_pectoral"
                elif candidates:
                    candidates.sort(key=lambda item: (
                        item["score_px"] if item["score_px"] is not None else np.inf,
                        item["prediction_error_px"]
                        if item["prediction_error_px"] is not None else np.inf,
                        item["from_global_row"],
                    ))
                    reason = candidates[0]["reason"]
                    if candidates[0]["age_frames"] > 1:
                        reason = "open_endpoint_competition"
                elif origins:
                    finite_steps = steps[:, j][np.isfinite(steps[:, j])]
                    reason = (
                        "large_step" if len(finite_steps)
                        and np.min(finite_steps) > config.step_fraction * span
                        else "prediction_gate"
                    )
                else:
                    reason = "new_or_unobserved_endpoint"
                events.append({
                    "kind": "directional_link_refused", "direction": direction,
                    "row": int(rows[target]), "global_row": int(rows[target]),
                    "frame": int(frame), "reason": reason, "candidates": candidates,
                })
        for origin in consumed:
            del active[origin]
        for target in current:
            if valid[target]:
                active[int(target)] = continued.get(int(target), [int(target)])
    return edges, evidence, events, predictions


def _reconnect(labels, frames, points, rows, span, config):
    histories = {}
    for index, label in enumerate(labels):
        histories.setdefault(int(label), []).append(index)
    starts = sorted((held[0], label) for label, held in histories.items())
    ends = sorted((held[-1], label) for label, held in histories.items())
    candidates, open_events = [], []
    valid = np.isfinite(points[:, 1]).all(axis=1)
    for end, label in ends:
        open_events.append({
            "kind": "open_endpoint", "global_row": int(rows[end]), "row": int(rows[end]),
            "frame": int(frames[end]), "local_id": label,
            "lifetime_frames": config.endpoint_lifetime_frames,
            "lifetime_seconds": config.endpoint_lifetime_seconds,
            "effective_lifetime_seconds": config.endpoint_lifetime_frames / config.native_fps,
            "valid_pectoral": bool(valid[end]),
            "reason": "context_boundary" if frames[end] == frames[-1] else "unconnected_endpoint",
        })
    # Native-frame index keeps reconnect enumeration bounded by the declared lifetime.
    by_frame = {}
    for end, label in ends:
        by_frame.setdefault(int(frames[end]), []).append((end, label))
    for start, start_label in starts:
        if not valid[start]:
            continue
        frame = int(frames[start])
        for old_frame in range(max(0, frame - config.endpoint_lifetime_frames), frame - 1):
            for end, end_label in by_frame.get(old_frame, []):
                if not valid[end] or start_label == end_label:
                    continue
                elapsed = frame - old_frame
                forward, _ = _predict(histories[end_label], frame, frames, points, config)
                backward, _ = _predict(
                    list(reversed(histories[start_label])), old_frame, frames, points, config
                )
                f_error = float(np.linalg.norm(forward - points[start, 1]))
                b_error = float(np.linalg.norm(backward - points[end, 1]))
                uncertainty = span * (config.uncertainty_fraction
                                      + config.uncertainty_growth_fraction * (elapsed - 1))
                gate = config.prediction_fraction * span + uncertainty
                if min(f_error, b_error) > gate:
                    continue
                candidates.append({
                    **_record(end, start, frames, rows),
                    "from_local_id": end_label, "to_local_id": start_label,
                    "elapsed_frames": elapsed, "elapsed_seconds": elapsed / config.native_fps,
                    "forward_error_px": f_error, "backward_error_px": b_error,
                    "uncertainty_px": float(uncertainty), "prediction_gate_px": float(gate),
                    "forward_prediction": forward.tolist(),
                    "backward_prediction": backward.tolist(),
                    "accepted": False, "reason": "candidate_only_not_merged",
                })
    target_sources, source_targets = {}, {}
    for candidate in candidates:
        target_sources.setdefault(candidate["to_global_row"], set()).add(
            candidate["from_global_row"]
        )
        source_targets.setdefault(candidate["from_global_row"], set()).add(
            candidate["to_global_row"]
        )
    for candidate in candidates:
        competitors = sorted(target_sources[candidate["to_global_row"]]
                             - {candidate["from_global_row"]})
        candidate["competing_endpoints"] = competitors
        candidate["competing_startpoints"] = sorted(
            source_targets[candidate["from_global_row"]] - {candidate["to_global_row"]}
        )
        if max(candidate["forward_error_px"], candidate["backward_error_px"]) > (
            candidate["prediction_gate_px"]
        ):
            candidate["reason"] = "bidirectional_prediction_disagreement"
        elif competitors or candidate["competing_startpoints"]:
            candidate["reason"] = "competing_endpoints"
    return candidates, open_events


def track(frame_idx, points, global_row, view, config=None, auxiliary=False):
    """Retain every row and return labels aligned to arbitrary input row order.

    ``global_row`` is a unique nonnegative C01 row identifier, never an identity.
    Scores remain the caller's unchanged arrays: this tracker neither consumes
    detection scores nor invents node scores. All table row references are global.
    """
    config = config or P1Config()
    span = config.span(view)
    frames, xy, rows = np.asarray(frame_idx), np.asarray(points), np.asarray(global_row)
    if frames.ndim != 1 or frames.dtype.kind not in "iu" or np.any(frames < 0):
        raise ValueError("frame_idx must be nonnegative integer native frames")
    if rows.shape != frames.shape or rows.dtype.kind not in "iu" or np.any(rows < 0):
        raise ValueError("global_row must contain nonnegative integer row identifiers")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("global_row must be unique")
    if xy.shape != (len(frames), 3, 2) or np.isinf(xy).any():
        raise ValueError("points must be (N,3,2) M/P/T with NaN for missing coordinates")
    order = np.lexsort((rows, frames))
    frames = frames[order].astype(np.int64, copy=True)
    xy = xy[order].astype(float, copy=True)
    rows = rows[order].astype(np.int64, copy=True)
    unique, starts, lengths = np.unique(frames, return_index=True, return_counts=True)
    held_groups = {int(frame): np.arange(start, start + length)
                   for frame, start, length in zip(unique, starts, lengths, strict=True)}
    groups = ([(frame, held_groups.get(frame, np.empty(0, dtype=np.int64)))
               for frame in range(int(frames[0]), int(frames[-1]) + 1)] if len(frames) else [])
    quality = _quality(frames, xy, rows, span, config)
    forward, f_info, f_events, f_predictions = _directional(
        frames, xy, rows, groups, span, config, auxiliary, False
    )
    backward, b_info, b_events, b_predictions = _directional(
        frames, xy, rows, groups, span, config, auxiliary, True
    )
    agreed = forward & backward
    parent = np.arange(len(frames), dtype=np.int64)

    def find(row):
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = int(parent[row])
        return row

    edges, events = [], f_events + b_events
    for a, b in sorted(agreed):
        parent[find(b)] = find(a)
        edge = {
            **_record(a, b, frames, rows), "reason": "bidirectional_agreement",
            "forward": f_info[a, b], "backward": b_info[a, b], "accepted": True,
        }
        edges.append(edge)
        for origin, target in ((a, b), (b, a)):
            _, _, flags = _auxiliary(origin, target, xy, span, config)
            quality[target]["temporal_cohesion_flags"].extend(flags)
    for a, b in sorted(forward ^ backward):
        events.append({
            **_record(a, b, frames, rows), "kind": "directional_disagreement",
            "reason": "forward_backward_disagreement", "frame": int(frames[b]),
            "forward": f_info.get((a, b)), "backward": b_info.get((a, b)),
        })
    roots = np.asarray([find(row) for row in range(len(frames))], dtype=np.int64)
    labels = np.unique(roots, return_inverse=True)[1].astype(np.int64)
    for _, group in groups:
        if len(np.unique(labels[group])) != len(group):
            raise AssertionError("One tracklet cannot contain two rows in one native frame")
    reconnects, endpoints = _reconnect(labels, frames, xy, rows, span, config)
    events.extend(endpoints)
    for item, label in zip(quality, labels, strict=True):
        item["local_id"] = int(label)
        item["temporal_cohesion_flags"] = sorted(set(item["temporal_cohesion_flags"]))
    aligned = np.empty_like(labels)
    aligned[order] = labels
    aligned_quality = [None] * len(quality)
    for index, original in enumerate(order):
        aligned_quality[int(original)] = quality[index]
    counts = np.unique(labels, return_counts=True)[1]
    metadata = {
        "algorithm_revision": "variable_count_p1_v1", "view": view,
        "auxiliary": bool(auxiliary), "config": asdict(config), "body_scale_px": span,
        "endpoint_lifetime_frames": config.endpoint_lifetime_frames,
        "endpoint_lifetime_seconds": config.endpoint_lifetime_seconds,
        "effective_endpoint_lifetime_seconds": config.endpoint_lifetime_frames / config.native_fps,
        "input_detections": len(frames), "retained_detections": len(frames),
        "output_tracklets": len(counts), "isolated_detections": int(np.sum(counts == 1)),
        "clean_links": len(edges), "forward_links": len(forward), "backward_links": len(backward),
        "disagreement_links": len(forward ^ backward), "gap_links": 0,
        "reconnect_candidates": len(reconnects), "prediction_records": len(f_predictions)
        + len(b_predictions), "reflection_status": "unknown_for_all_rows",
        "refusal_reasons": dict(Counter(event["reason"] for event in f_events + b_events)),
        "sleap_tracking_ids_used": False, "detector_instance_grouping_retained": True,
        "identity_accuracy_claim": False, "row_reference": "C01 global_row",
        "heading_rule": "diagnostic only; angle never independently vetoes a link",
        "auxiliary_rule": "bounded coherent M/T support; missing/suspect nodes never veto P",
        "prediction_rule": "separate unobserved provisional directional endpoint forecasts",
        "represented_endpoint_rule": (
            "unique gap hypothesis forecast retires when represented by a new observation; "
            "no history transfer, label merge, or gap link"
        ),
        "empty_frames_between_observations": len(groups) - len(unique),
    }
    return P1Result(
        aligned, edges, events, f_predictions + b_predictions, reconnects, aligned_quality, metadata
    )

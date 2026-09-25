"""Bounded two-hypothesis continuation of fixed revision-2 tracklets.

See deferred_protocol.md. Ambiguous interior detections remain unassigned to the
reconnected flank identities; their existing nonnegative tracklet IDs survive.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import permutations

import numpy as np
from tracking import TrackingConfig, TrackingResult, _inputs, _result


@dataclass(frozen=True)
class DeferredConfig:
    max_horizon_frames: int = 5
    future_observations: int = 2
    competing_tracks: int = 2

    def __post_init__(self):
        if (self.max_horizon_frames, self.future_observations, self.competing_tracks) != (5, 2, 2):
            raise ValueError(
                "This frozen revision evaluates two hypotheses within five native frames"
            )


def _distance(a, b, nodes, weights):
    error = np.linalg.norm(a - b, axis=-1)
    valid = nodes & np.isfinite(error)
    if not valid.any():
        return np.inf
    return float(np.average(error[valid], weights=weights[valid]))


def _flank(label, rows, side, frames, points, native, span):
    if len(rows) < 3:
        return None
    selected = rows[-3:] if side == "left" else rows[:3]
    times = frames[selected]
    if not np.all(np.diff(times) == 1):
        return None
    xy = points[selected, :2]
    valid = np.isfinite(xy).all(axis=(0, 2))
    speeds = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    valid &= np.all(speeds <= span * native.hard_speed_fraction, axis=0)
    if not valid.any():
        return None
    velocity = np.full((2, 2), np.nan)
    velocity[valid] = np.median(np.diff(xy[:, valid], axis=0), axis=0)
    boundary = -1 if side == "left" else 0
    return dict(
        label=int(label),
        rows=selected,
        times=times,
        xy=xy,
        valid=valid,
        velocity=velocity,
        boundary_frame=int(times[boundary]),
        boundary_xy=xy[boundary],
    )


def _boundary_compatible(left, right, native, span, weights):
    nodes = left["valid"] & right["valid"]
    gap = right["boundary_frame"] - left["boundary_frame"]
    if not nodes.any() or gap <= 0:
        return False
    speed = np.linalg.norm(right["boundary_xy"] - left["boundary_xy"], axis=-1) / gap
    if np.any(speed[nodes] > span * native.hard_speed_fraction):
        return False
    f = left["boundary_xy"] + left["velocity"] * gap
    r = right["boundary_xy"] - right["velocity"] * gap
    limit = span * native.prediction_error_fraction * (1 + native.gap_error_growth * (gap - 1))
    return (
        min(
            _distance(f, right["boundary_xy"], nodes, weights),
            _distance(r, left["boundary_xy"], nodes, weights),
        )
        <= limit
    )


def _components(left, right, native, span, weights):
    graph = defaultdict(set)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if _boundary_compatible(a, b, native, span, weights):
                graph[("L", i)].add(("R", j))
                graph[("R", j)].add(("L", i))
    seen = set()
    for seed in sorted(graph):
        if seed in seen:
            continue
        todo, component = [seed], set()
        while todo:
            node = todo.pop()
            if node in component:
                continue
            component.add(node)
            todo.extend(graph[node] - component)
        seen |= component
        li = sorted(i for side, i in component if side == "L")
        ri = sorted(i for side, i in component if side == "R")
        yield [left[i] for i in li], [right[j] for j in ri], graph, li, ri


def _has_ambiguity(events, labels, left_frame, right_frame, candidate_labels):
    kinds = {
        "ambiguous_directional_assignment",
        "bidirectional_disagreement",
        "ended_competing_history",
    }
    for event in events:
        if event.get("kind") not in kinds:
            continue
        a, b = event.get("from_frame"), event.get("to_frame")
        if a is None or b is None or max(a, b) < left_frame or min(a, b) > right_frame:
            continue
        rows = [event.get("from_row"), event.get("to_row")]
        rows += event.get("admissible_detection_rows", [])
        if any(
            row is not None and 0 <= row < len(labels) and int(labels[row]) in candidate_labels
            for row in rows
        ):
            return True
    return False


def _hypothesis(left, right, mapping, nodes, frames, native, span, weights):
    fish_scores, errors, gates, speeds = [], [], [], []
    for i, j in enumerate(mapping):
        a, b = left[i], right[j]
        pair_errors = []
        gap = b["boundary_frame"] - a["boundary_frame"]
        speed = np.linalg.norm(b["boundary_xy"] - a["boundary_xy"], axis=-1) / gap
        speeds.extend(speed[nodes].tolist())
        for origin, targets in ((a, b), (b, a)):
            dt = targets["times"] - origin["boundary_frame"]
            prediction = origin["boundary_xy"][None] + dt[:, None, None] * origin["velocity"][None]
            for predicted, observed, distance in zip(
                prediction, targets["xy"], np.abs(dt), strict=True
            ):
                error = _distance(predicted, observed, nodes, weights)
                limit = (
                    span
                    * native.prediction_error_fraction
                    * (1 + native.gap_error_growth * (int(distance) - 1))
                )
                pair_errors.append(error)
                errors.append(error)
                gates.append(limit)
        fish_scores.append(float(np.mean(pair_errors)))
    return dict(
        mapping=[int(j) for j in mapping],
        label_pairs=[[left[i]["label"], right[j]["label"]] for i, j in enumerate(mapping)],
        score_px=float(sum(fish_scores)),
        fish_mean_prediction_error_px=fish_scores,
        observation_prediction_errors_px=errors,
        observation_prediction_gates_px=gates,
        bridge_node_speeds_px_per_native_frame=speeds,
        within_gates=bool(
            np.all(np.asarray(errors) <= gates)
            and np.all(np.asarray(speeds) <= span * native.hard_speed_fraction)
        ),
    )


def _nearby_count(left, right, nodes, frames, points, native, span, weights):
    first = int(left[0]["times"][0])
    last = int(right[0]["times"][-1])
    maximum, at_frame = 0, None
    for frame in range(first, last + 1):
        lo, hi = np.searchsorted(frames, [frame, frame + 1])
        count = 0
        for point in points[lo:hi, :2]:
            near = False
            for flank in left + right:
                dt = abs(frame - flank["boundary_frame"])
                limit = (
                    span
                    * native.prediction_error_fraction
                    * (1 + native.gap_error_growth * max(0, dt - 1))
                )
                predicted = flank["boundary_xy"] + flank["velocity"] * (
                    frame - flank["boundary_frame"]
                )
                near |= _distance(point, predicted, nodes, weights) <= limit
            count += int(near)
        if count > maximum:
            maximum, at_frame = count, frame
    return maximum, at_frame


def refine_deferred(frame_idx, points, base_result: TrackingResult, *, view, config=None):
    """Reconsider two continuation hypotheses without altering confirmed edges.

    Only fixed tracklet components are joined. The reported decision interval is
    the whole observed support, and unresolved detections inside a bridge retain
    their existing tracklet membership. Refusals leave the base partition intact.
    """
    config = config or DeferredConfig()
    native = TrackingConfig(**base_result.metadata.get("config", {}))
    frames, xy, native = _inputs(frame_idx, points, view, native)
    labels = np.asarray(base_result.labels)
    if labels.shape != frames.shape or labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("Base labels must retain every detection with a nonnegative integer ID")
    for frame in np.unique(frames):
        held = labels[frames == frame]
        if np.unique(held).size != held.size:
            raise ValueError("Base labels must be unique within each frame")
    span, weights = native.span(view), np.array([1.0, native.pectoral_weight])
    tracks = {int(label): np.flatnonzero(labels == label) for label in np.unique(labels)}
    left_by_frame, right_by_frame = defaultdict(list), defaultdict(list)
    for label, rows in tracks.items():
        for side, destination in (("left", left_by_frame), ("right", right_by_frame)):
            flank = _flank(label, rows, side, frames, xy, native, span)
            if flank is not None:
                destination[flank["boundary_frame"]].append(flank)
    decisions = []
    for lframe, left_group in sorted(left_by_frame.items()):
        for rframe in range(lframe + 1, lframe + config.max_horizon_frames + 1):
            right_group = right_by_frame.get(rframe, [])
            if not right_group:
                continue
            for left, right, graph, li, ri in _components(
                left_group, right_group, native, span, weights
            ):
                if len(left) == len(right) == 1:
                    continue  # This revision does not invent a one-hypothesis bridge.
                involved = {f["label"] for f in left + right}
                if not _has_ambiguity(base_result.events, labels, lframe, rframe, involved):
                    continue
                decision = dict(
                    kind="deferred_candidate",
                    status="refused",
                    left_frame=lframe,
                    right_frame=rframe,
                    from_frame=lframe,
                    to_frame=rframe,
                    left_labels=[f["label"] for f in left],
                    right_labels=[f["label"] for f in right],
                    left_support_frames=[f["times"].tolist() for f in left],
                    right_support_frames=[f["times"].tolist() for f in right],
                )
                decisions.append(decision)
                if len(left) != 2 or len(right) != 2:
                    decision["reason"] = "not_exactly_two_competing_tracks_on_both_sides"
                    continue
                if (
                    rframe - int(left[0]["times"][0]) > config.max_horizon_frames
                    or int(right[0]["times"][-1]) - lframe > config.max_horizon_frames
                ):
                    decision["reason"] = "native_deadline_exceeded"
                    continue
                if not all(("R", j) in graph[("L", i)] for i in li for j in ri):
                    decision["reason"] = "not_two_possible_bijections"
                    continue
                nodes = np.logical_and.reduce([f["valid"] for f in left + right])
                decision["support_nodes"] = [
                    name for name, yes in zip(("mouth", "pectoral"), nodes, strict=True) if yes
                ]
                if not nodes.any():
                    decision["reason"] = "no_shared_primary_node_support"
                    continue
                count, at_frame = _nearby_count(
                    left, right, nodes, frames, xy, native, span, weights
                )
                decision.update(max_nearby_detections=count, max_nearby_at_frame=at_frame)
                if count > 2:
                    decision["reason"] = "third_detection_in_candidate_neighbourhood"
                    continue
                hypotheses = [
                    _hypothesis(left, right, p, nodes, frames, native, span, weights)
                    for p in permutations(range(2))
                ]
                order = np.argsort([h["score_px"] for h in hypotheses])
                winner, runnerup = (hypotheses[int(i)] for i in order)
                margin = runnerup["score_px"] - winner["score_px"]
                decision.update(
                    hypotheses=hypotheses,
                    winner_index=int(order[0]),
                    score_margin_px=float(margin),
                    required_margin_px=span * native.assignment_margin_fraction,
                )
                if not winner["within_gates"]:
                    decision["reason"] = "winner_exceeds_prediction_or_speed_gate"
                elif margin < span * native.assignment_margin_fraction:
                    decision["reason"] = "insufficient_two_hypothesis_margin"
                else:
                    decision.update(
                        status="proposed",
                        reason="future_support_resolves_two_hypotheses",
                        proposed_label_pairs=winner["label_pairs"],
                    )
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for i, decision in enumerate(decisions):
        if decision["status"] == "proposed":
            for a, b in decision["proposed_label_pairs"]:
                outgoing[a].append(i)
                incoming[b].append(i)
    conflicts = {
        i
        for groups in (outgoing, incoming)
        for values in groups.values()
        if len(values) > 1
        for i in values
    }
    parent = {label: label for label in tracks}

    def find(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    bridges = 0
    for i, decision in enumerate(decisions):
        if decision["status"] != "proposed":
            continue
        if i in conflicts:
            decision.update(status="refused", reason="competing_endpoint_proposals")
            continue
        for a, b in decision["proposed_label_pairs"]:
            pa, pb = find(a), find(b)
            parent[max(pa, pb)] = min(pa, pb)
            bridges += 1
        decision["status"] = "accepted"
    roots = np.array([find(int(label)) for label in labels], dtype=np.int64)
    result_labels = np.unique(roots, return_inverse=True)[1].astype(np.int64)
    events = [*base_result.events, *decisions]
    return _result(
        frames,
        result_labels,
        events,
        native,
        view,
        arm="C",
        baseline_ids_used=False,
        algorithm_revision="v3_deferred_two_hypothesis_refinement",
        base_algorithm_revision=base_result.metadata.get("algorithm_revision"),
        base_metadata=base_result.metadata.copy(),
        deferred_config=asdict(config),
        deferred_candidate_episodes=len(decisions),
        deferred_accepted_episodes=sum(d["status"] == "accepted" for d in decisions),
        deferred_refused_episodes=sum(d["status"] == "refused" for d in decisions),
        deferred_links_added=bridges,
        base_tracklets=len(tracks),
        retained_links=len(labels) - len(np.unique(result_labels)),
        confirmed_base_links_preserved=True,
        interior_ambiguous_detections="keep existing tracklet IDs; no guessed per-frame assignment",
    )

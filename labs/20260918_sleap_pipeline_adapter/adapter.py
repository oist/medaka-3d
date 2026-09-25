"""SLEAP provisional histories with separately certified adjacent 2D fragments.

The installed Tracker.track/get_scores, official Hungarian matcher and native
LocalQueueCandidates.update_tracks are executed. Extension points supply fixed
P-only OKS, per-observation native-age expiry, unique dummy assignments, and
monotonic provisional IDs. No installed source or global scorer is modified.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import asdict, dataclass
from functools import partial

import numpy as np
import sleap_io as sio
from sleap_nn.evaluation import compute_oks
from sleap_nn.tracking.candidates.local_queues import LocalQueueCandidates
from sleap_nn.tracking.tracker import Tracker


@dataclass(frozen=True)
class AdapterConfig:
    history: int = 5
    match_threshold: float = 0.05
    margin: float = 0.05
    max_age_seconds: float = 0.10
    native_fps: float = 166.0
    top_body_px: float = 120.0
    side_body_px: float = 100.0
    oks_stddev: float = 0.025
    max_step_fraction: float = 0.30

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("Configuration values must be positive and finite")
        if not isinstance(self.history, int) or self.history < 1:
            raise ValueError("history must be a positive integer")
        if self.match_threshold >= 1 or self.margin >= 1:
            raise ValueError("Similarity threshold and margin must be less than one")

    def span(self, view):
        if view not in ("top_cam", "side1_cam", "side2_cam"):
            raise ValueError(f"Unknown camera view: {view}")
        return self.top_body_px if view == "top_cam" else self.side_body_px

    @property
    def max_age_frames(self):
        return math.ceil(self.max_age_seconds * self.native_fps)


@dataclass
class AdapterResult:
    labels: np.ndarray
    provisional_labels: np.ndarray
    edges: list[dict]
    events: list[dict]
    reconnect_candidates: list[dict]
    metadata: dict


class _ExpiringQueues(LocalQueueCandidates):
    """Retain native SLEAP observation queues without ever reusing an ID."""

    def __init__(self, config):
        super().__init__(window_size=config.history, max_tracks=None, min_new_track_points=0)
        self.config = config
        self.next_provisional_id = 0
        self.expired_tracks = 0
        self.expired_observations = 0

    def get_new_track_id(self):
        identity = self.next_provisional_id
        self.next_provisional_id += 1
        self.tracker_queue[identity] = deque(maxlen=self.window_size)
        return identity

    def expire(self, native_frame):
        expired = []
        for identity in self.current_tracks:
            original = self.tracker_queue[identity]
            held = deque(
                (item for item in original
                 if 0 < native_frame - item.frame_idx <= self.config.max_age_frames),
                maxlen=self.window_size,
            )
            self.expired_observations += len(original) - len(held)
            if held:
                self.tracker_queue[identity] = held
            else:
                expired.append(identity)
                del self.tracker_queue[identity]
        self.current_tracks = [identity for identity in self.current_tracks
                               if identity not in expired]
        self.expired_tracks += len(expired)
        return expired


class _EvidenceTracker(Tracker):
    """Change assignment policy while executing native Tracker.track/get_scores."""

    def configure(self, config, span):
        self.adapter_config = config
        self.scoring_method = "fixed_pectoral_oks"
        self._scoring_functions = {
            **self._scoring_functions,
            "fixed_pectoral_oks": partial(compute_oks, scale=span**2, stddev=config.oks_stddev),
        }
        self.assignment_calls = 0
        self.score_calls = 0
        self.last_scores = np.empty((0, 0), dtype=float)
        self.last_selected_columns = {}
        self.last_candidate_ids = []

    def get_scores(self, current_instances, candidates_feature_dict):
        self.score_calls += 1
        scores = super().get_scores(current_instances, candidates_feature_dict)
        self.last_scores = scores.copy()
        return scores

    def assign_tracks(self, current_instances, cost_matrix):
        self.assignment_calls += 1
        identities = list(self.candidate.current_tracks)
        self.last_candidate_ids = identities
        rows, columns = cost_matrix.shape
        extended = np.full((rows, columns + rows), np.inf, dtype=float)
        valid = np.isfinite(cost_matrix) & (-cost_matrix >= self.adapter_config.match_threshold)
        extended[:, :columns] = np.where(valid, cost_matrix, np.inf)
        extended[np.arange(rows), columns + np.arange(rows)] = -self.adapter_config.match_threshold
        # The stock matcher is executed; dummy columns never reach the candidate updater.
        selected_rows, selected_columns = self._track_matching_methods["hungarian"](extended)
        real = selected_columns < columns
        matched_rows, matched_columns = selected_rows[real], selected_columns[real]
        if np.any(~valid[matched_rows, matched_columns]):
            raise AssertionError("A below-threshold or invalid provisional match escaped its dummy")
        self.last_selected_columns = dict(zip(matched_rows.tolist(), matched_columns.tolist(),
                                             strict=True))
        actual_ids = np.array([identities[column] for column in matched_columns], dtype=np.int64)
        tracking_scores = [-float(cost_matrix[row, column])
                           for row, column in zip(matched_rows, matched_columns, strict=True)]
        return self.candidate.update_tracks(
            current_instances, matched_rows, actual_ids, tracking_scores
        )


def _number(value):
    return float(value) if np.isfinite(value) else None


def _references(previous, current, frames, global_rows):
    return {
        "from_global_row": int(global_rows[previous]),
        "to_global_row": int(global_rows[current]),
        "from_frame": int(frames[previous]), "to_frame": int(frames[current]),
    }


def _evidence(position, scores, identities, prior, instance_rows, frame, global_rows):
    candidates = []
    for column, identity in enumerate(identities):
        queue = prior[identity]
        last = queue[-1]
        previous = instance_rows[id(last.src_instance)]
        candidates.append({
            "provisional_id": int(identity),
            "from_global_row": int(global_rows[previous]), "from_frame": int(last.frame_idx),
            "age_frames": int(frame - last.frame_idx),
            "history_frames": [int(item.frame_idx) for item in queue],
            "history_observations": len(queue), "score": _number(scores[position, column]),
        })
    return candidates


def _inputs(frame_idx, points, score, global_row):
    frames, xy, scores, rows = map(np.asarray, (frame_idx, points, score, global_row))
    if frames.ndim != 1 or frames.dtype.kind not in "iu" or np.any(frames < 0):
        raise ValueError("frame_idx must contain nonnegative integer native frames")
    if rows.shape != frames.shape or rows.dtype.kind not in "iu" or np.any(rows < 0):
        raise ValueError("global_row must contain nonnegative integer row identifiers")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("global_row must be unique")
    if xy.shape != (len(frames), 3, 2) or np.isinf(xy).any():
        raise ValueError("points must have (N,3,2) M/P/T shape, with NaN for missing coordinates")
    if scores.shape != frames.shape or scores.dtype.kind not in "fiu":
        raise ValueError("score must contain one numeric instance score per input row")
    order = np.lexsort((rows, frames))
    frames, xy, scores, rows = (array[order] for array in (frames, xy, scores, rows))
    return frames, xy, scores, rows, order


def track(frame_idx, points, score, global_row, view, config=None):
    """Return every row with separate provisional-history and confirmed-fragment IDs.

    Input order may be arbitrary; decisions use (native frame, global row) order.
    Original points/scores are read only. Missing-P rows remain singleton fragments
    with provisional ID -1. Provisional assignment is never biological identity GT.
    """
    config = config or AdapterConfig()
    span = config.span(view)
    frames, xy, scores, rows, order = _inputs(frame_idx, points, score, global_row)
    valid = np.isfinite(xy[:, 1]).all(axis=-1)
    labels = np.arange(len(frames), dtype=np.int64)
    provisional = np.full(len(frames), -1, dtype=np.int64)
    events = [None] * len(frames)
    edges, reconnects, instance_rows = [], [], {}
    queue = _ExpiringQueues(config)
    backend = _EvidenceTracker(candidate=queue, is_local_queue=True, features="keypoints",
                               scoring_reduction="mean", track_matching_method="hungarian",
                               oks_stddev=config.oks_stddev)
    backend.configure(config, span)
    skeleton = sio.Skeleton(["pectoral"])
    native_track_calls, max_active, births, max_history_age = 0, 0, 0, 0
    unique, starts, lengths = np.unique(frames, return_index=True, return_counts=True)
    for frame, start, length in zip(unique, starts, lengths, strict=True):
        frame = int(frame)
        expired = queue.expire(frame)
        current = np.arange(start, start + length)
        kept = current[valid[current]]
        prior = {identity: list(queue.tracker_queue[identity]) for identity in queue.current_tracks}
        for held in prior.values():
            max_history_age = max(max_history_age, max(frame - item.frame_idx for item in held))
        batch = [sio.PredictedInstance.from_numpy(
            points_data=xy[row, 1:2].astype(float), skeleton=skeleton,
            point_scores=np.ones(1), score=float(scores[row]),
        ) for row in kept]
        for row, instance in zip(kept, batch, strict=True):
            instance_rows[id(instance)] = int(row)
        backend.last_selected_columns = {}
        backend.last_candidate_ids = []
        backend.last_scores = np.empty((len(batch), 0), dtype=float)
        returned = backend.track(batch, frame_idx=frame) if batch else []
        native_track_calls += bool(batch)
        if (len(returned) != len(batch)
                or {id(item) for item in returned} != {id(item) for item in batch}):
            raise AssertionError("Native SLEAP readout dropped or replaced a P-observed input row")
        for instance in returned:
            provisional[instance_rows[id(instance)]] = int(
                instance.track.name.removeprefix("track_")
            )
        max_active = max(max_active, len(queue.current_tracks))
        for row in current[~valid[current]]:
            events[row] = {
                "global_row": int(rows[row]), "frame": frame, "native_frame": frame,
                "kind": "observation_decision", "provisional_id": -1,
                "confirmed": False, "reason": "missing_pectoral", "score": None,
                "row_margin": None, "column_margin": None, "candidates": [],
                "reflection_status": "unknown", "expired_provisional_ids": expired,
            }
        matrix, identities = backend.last_scores, backend.last_candidate_ids
        for position, row in enumerate(kept):
            identity = int(provisional[row])
            candidates = _evidence(position, matrix, identities, prior, instance_rows, frame, rows)
            chosen = backend.last_selected_columns.get(position)
            event = {
                "global_row": int(rows[row]), "frame": frame, "native_frame": frame,
                "kind": "observation_decision", "provisional_id": identity,
                "confirmed": False, "reason": "new_provisional_history", "score": None,
                "row_margin": None, "column_margin": None, "candidates": candidates,
                "reflection_status": "unknown", "expired_provisional_ids": expired,
            }
            if chosen is None:
                births += 1
                event["reason"] = "dummy_unmatched" if identities else "new_provisional_history"
            else:
                if identities[chosen] != identity:
                    raise AssertionError("Candidate column confused with a noncontiguous track ID")
                old = prior[identity][-1]
                previous = instance_rows[id(old.src_instance)]
                age = frame - int(old.frame_idx)
                own = float(matrix[position, chosen])
                alternatives_row = np.delete(matrix[position], chosen)
                alternatives_col = np.delete(matrix[:, chosen], position)
                row_next = max(config.match_threshold,
                               float(np.max(alternatives_row, initial=-np.inf)))
                column_next = max(config.match_threshold,
                                  float(np.max(alternatives_col, initial=-np.inf)))
                row_margin, column_margin = own - row_next, own - column_next
                step = float(np.linalg.norm(xy[row, 1] - xy[previous, 1]))
                evidence = {
                    **_references(previous, row, frames, rows), "provisional_id": identity,
                    "score": own, "row_margin": row_margin, "column_margin": column_margin,
                    "required_margin": config.margin, "match_threshold": config.match_threshold,
                    "age_frames": age, "age_seconds": age / config.native_fps, "step_px": step,
                }
                if age > 1:
                    reason = "gap_candidate_only"
                    reconnects.append({**evidence, "accepted": False, "provisional_match": True,
                                       "reason": reason,
                                       "competing_provisional_ids": [value["provisional_id"]
                                           for value in candidates if value["provisional_id"]
                                           != identity and value["score"] is not None
                                           and value["score"] >= config.match_threshold]})
                elif step > config.max_step_fraction * span:
                    reason = "large_pectoral_step"
                elif min(row_margin, column_margin) <= 1e-12:
                    reason = "nonunique_or_nonmutual_candidate"
                elif min(row_margin, column_margin) < config.margin:
                    reason = "insufficient_similarity_margin"
                else:
                    reason = "confirmed_adjacent"
                    labels[row] = labels[previous]
                    edges.append({**evidence, "accepted": True, "reason": reason})
                    event["confirmed"] = True
                event.update(evidence, reason=reason)
            events[row] = event
    labels = np.unique(labels, return_inverse=True)[1].astype(np.int64)
    for start, length in zip(starts, lengths, strict=True):
        if len(np.unique(labels[start:start + length])) != length:
            raise AssertionError("Confirmed fragments must be one-to-one within a native frame")
        active_labels = provisional[start:start + length]
        active_labels = active_labels[active_labels >= 0]
        if len(np.unique(active_labels)) != len(active_labels):
            raise AssertionError("Provisional histories must be one-to-one within a native frame")
    for index, event in enumerate(events):
        event["local_id"] = int(labels[index])
    aligned_labels, aligned_provisional = np.empty_like(labels), np.empty_like(provisional)
    aligned_labels[order], aligned_provisional[order] = labels, provisional
    aligned_events = [None] * len(events)
    for index, original in enumerate(order):
        aligned_events[int(original)] = events[index]
    metadata = {
        "algorithm_revision": "sleap_pipeline_adapter_v1", "view": view,
        "config": asdict(config), "body_scale_px": span, "fixed_oks_area_px2": span**2,
        "max_age_frames": config.max_age_frames,
        "effective_max_age_seconds": config.max_age_frames / config.native_fps,
        "input_detections": len(frames), "retained_detections": len(frames),
        "output_tracklets": len(np.unique(labels)), "clean_links": len(edges), "gap_links": 0,
        "provisional_tracks_created": queue.next_provisional_id, "birth_count": births,
        "max_active_tracks": max_active, "expired_tracks": queue.expired_tracks,
        "expired_history_observations": queue.expired_observations,
        "maximum_scored_history_age_frames": max_history_age,
        "native_tracker_track_calls": native_track_calls,
        "native_get_scores_calls": backend.score_calls,
        "native_hungarian_calls": backend.assignment_calls,
        "sleap_paths_called": [
            "Tracker.track", "Tracker.get_features", "Tracker.generate_candidates",
            "Tracker.update_candidates", "Tracker.get_scores", "Tracker.scores_to_cost_matrix",
            "sleap_nn.evaluation.compute_oks(scale=span**2, stddev=oks_stddev)",
            "sleap_nn.tracking.utils.hungarian_matching(extended dummy cost matrix)",
            "LocalQueueCandidates.get_features_from_track_id", "LocalQueueCandidates.update_tracks",
            "LocalQueueCandidates.add_new_tracks",
        ],
        "sleap_extension_points": ["fixed P scorer", "assign_tracks", "expiry", "get_new_track_id"],
        "history_key": "provisional SLEAP ID, independent of confirmed fragment labels",
        "soft_refusal_updates_provisional_history": True,
        "provisional_identity_is_ground_truth": False, "identity_accuracy_claim": False,
        "sleap_input_tracking_ids_used": False, "max_tracks": None,
        "reflection_status": "unknown_for_all_rows", "input_points_or_scores_modified": False,
        "node_score_policy": "SLEAP adapter uses synthetic unit P confidence; not a measured score",
        "reconnect_candidates": len(reconnects),
        "reconnect_candidate_scope": (
            "selected provisional gap matches; unselected history alternatives remain in events"
        ),
        "decision_reasons": dict(Counter(event["reason"] for event in events)),
        "zero_frames_between_observations": (int(frames[-1] - frames[0] + 1 - len(unique))
                                             if len(frames) else 0),
    }
    return AdapterResult(aligned_labels, aligned_provisional, edges, aligned_events,
                         reconnects, metadata)


def native_pectoral(frame_idx, points, score, global_row, view, history=5):
    """Native SLEAP control: P-only fixed OKS, no dummy/expiry/output margin.

    Native IDs may span gaps. All rows, including missing-P singleton outputs,
    survive. These labels are a comparison arm, never biological identity truth.
    """
    config = AdapterConfig(history=history)
    span = config.span(view)
    frames, xy, scores, rows, order = _inputs(frame_idx, points, score, global_row)
    backend = Tracker.from_config(
        window_size=history, candidates_method="local_queues", features="keypoints",
        scoring_method="oks", scoring_reduction="mean", track_matching_method="hungarian",
        min_match_points=0, max_tracks=None, tracking_target_instance_count=None,
        oks_stddev=config.oks_stddev,
    )
    backend.scoring_method = "fixed_pectoral_oks"
    backend._scoring_functions = {
        **backend._scoring_functions,
        "fixed_pectoral_oks": partial(compute_oks, scale=span**2, stddev=config.oks_stddev),
    }
    labels = np.full(len(frames), -1, dtype=np.int64)
    valid = np.isfinite(xy[:, 1]).all(axis=-1)
    skeleton = sio.Skeleton(["pectoral"])
    _, starts, lengths = np.unique(frames, return_index=True, return_counts=True)
    calls = 0
    for start, length in zip(starts, lengths, strict=True):
        current = np.arange(start, start + length)
        kept = current[valid[current]]
        batch = [sio.PredictedInstance.from_numpy(
            points_data=xy[row, 1:2].astype(float), skeleton=skeleton,
            point_scores=np.ones(1), score=float(scores[row]),
        ) for row in kept]
        mapping = {id(instance): int(row) for row, instance in zip(kept, batch, strict=True)}
        returned = backend.track(batch, frame_idx=int(frames[start])) if batch else []
        calls += bool(batch)
        if {id(instance) for instance in returned} != set(mapping):
            raise AssertionError("Native P control changed its input rows")
        for instance in returned:
            labels[mapping[id(instance)]] = int(instance.track.name.removeprefix("track_"))
    next_label = int(labels.max(initial=-1)) + 1
    labels[~valid] = np.arange(next_label, next_label + int((~valid).sum()))
    last, edges = {}, []
    for row, identity in enumerate(labels):
        if int(identity) in last:
            previous = last[int(identity)]
            edges.append({
                **_references(previous, row, frames, rows), "provisional_id": int(identity),
                "accepted": True, "reason": "native_sleap_history_continuity",
                "gap_frames": int(frames[row] - frames[previous]),
            })
        last[int(identity)] = row
    aligned = np.empty_like(labels)
    aligned[order] = labels
    metadata = {
        "algorithm_revision": "native_sleap_pectoral_fixed_oks_v1", "view": view,
        "history": history, "fixed_oks_area_px2": span**2, "oks_stddev": config.oks_stddev,
        "scoring_reduction": "mean", "dummy_used": False, "native_age_expiry_used": False,
        "output_margin_used": False, "max_tracks": None,
        "input_detections": len(frames), "retained_detections": len(frames),
        "output_tracklets": len(np.unique(labels)),
        "clean_links": sum(edge["gap_frames"] == 1 for edge in edges),
        "gap_links": sum(edge["gap_frames"] > 1 for edge in edges),
        "native_tracker_track_calls": calls,
        "sleap_paths_called": ["Tracker.track", "Tracker.get_scores", "Tracker.assign_tracks",
                               "LocalQueueCandidates.update_tracks", "compute_oks(fixed P scale)"],
        "missing_p_policy": "retained isolated output label; not inserted into native P history",
        "identity_accuracy_claim": False, "provisional_identity_is_ground_truth": False,
        "sleap_input_tracking_ids_used": False, "reflection_status": "unknown_for_all_rows",
    }
    return AdapterResult(aligned, aligned.copy(), edges, [], [], metadata)

"""Frame-boundary continuation of the frozen persistent three-body algorithm.

Checkpoints retain every beam parent, not merely the current best trajectory.
The numerical step is copied from the frozen loop; no detector, score, proposal,
review rule or geometric initialization is added at a continuation boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from itertools import pairwise

import numpy as np
from motion import FlowObservations
from node_geometry import RayObservations, supported_nodes
from observations import FrameObservation, sample_pixels
from persistent_node_tracker import (
    AcceleratedPersistentNodeTracker,
    PersistentNodeState,
    PersistentNodeThreeBodyTracker,
    updated_ages,
)

from medaka3d.types import VIEWS

SCHEMA = "persistent_tracker_full_state_v1"
STATE_FIELDS = ("pose", "velocity", "missing", "review", "cost",
                "node_missing", "direct_pectoral_age")
RECORD_FIELDS = (
    "frame_idx", "pose_xyz", "observed", "sources", "identity_review",
    "mask_supported", "body_supported", "missing", "costs", "path_costs",
    "loss_parts", "score_margin", "parent", "proposal_kind", "hypotheses",
    "proposal_flow_nodes", "proposal_flow_parent", "node_missing", "direct_pectoral_age",
)


def _native_frame(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("native frame must be an integer")
    value = int(value)
    if value < 0 or value >= np.iinfo(np.int64).max:
        raise ValueError("native frame is outside supported int64 range")
    return value


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _observation_copy(frame, *, require_images=True):
    """The iterator/caller may reuse buffers after advance returns."""
    native = _native_frame(frame.frame_idx)
    saved = {}
    for field in ("detections", "images_bgr", "masks", "backgrounds_bgr"):
        values = getattr(frame, field)
        if not require_images and field != "detections" and not values:
            saved[field] = {}
            continue
        if set(values) != set(VIEWS):
            raise ValueError("observation must contain exactly the three camera views")
        saved[field] = {}
        for view in VIEWS:
            array = values[view]
            if field == "masks" and array is None:
                saved[field][view] = None
                continue
            array = np.asarray(array)
            if array.dtype.hasobject:
                raise ValueError("observation arrays cannot contain objects")
            saved[field][view] = array.copy()
    for view in VIEWS:
        points = saved["detections"][view]
        if points.ndim != 3 or points.shape[1:] != (3, 2) or points.dtype.kind != "f":
            raise ValueError("detections must be floating instance,node,xy arrays")
        if not require_images:
            continue
        image, bg = saved["images_bgr"][view], saved["backgrounds_bgr"][view]
        masks = saved["masks"][view]
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3
                or min(image.shape[:2]) < 1 or bg.dtype != np.uint8 or bg.shape != image.shape):
            raise ValueError("previous observation requires matching native uint8 BGR/background")
        if masks is not None and (masks.dtype != bool or masks.ndim != 3
                                  or masks.shape[1:] != image.shape[:2]):
            raise ValueError("masks must retain native boolean instances, including known-empty")
    return FrameObservation(native, **saved)


class ContinuousPersistentNodeTracker(PersistentNodeThreeBodyTracker):
    """One logical initialization with explicit completed-frame boundaries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._states = None
        self._history = []
        self._previous_frame = None
        self._previous_observation = None
        self._finished = False
        self._failed = False
        self.restore_count = 0
        self._initial_context = self._context()

    def _context(self):
        return json.loads(_canonical({
            "implementation": type(self).__name__,
            "tracker_config": asdict(self.config),
            "recovery_config": asdict(self.recovery_config),
            "persistence_config": asdict(self.persistence_config),
            "initial_pose": self.initial, "initial_lengths": self.lengths,
            "lower": self.lower, "upper": self.upper,
            "views": list(self.rig.views), "world_units": self.rig.world_units,
            "cameras": [asdict(camera) for camera in self.rig.cameras],
        }))

    @property
    def processed_frames(self):
        return len(self._history)

    @property
    def previous_frame(self):
        return self._previous_frame

    def begin(self):
        if self.initialize_count or self._finished or self._failed:
            raise RuntimeError("a tracker cannot be initialized twice")
        self.initialize_count = 1
        self._states = [PersistentNodeState(
            self.initial.copy(), np.zeros((3, 3)), np.zeros(3, int),
            np.zeros(3, bool), 0.0, np.zeros((3, 3), int), np.zeros(3, int))]

    def _require_active(self):
        if self._failed or self._finished or self._states is None:
            raise RuntimeError("tracker requires an active, nonfailed continuation")

    def advance(self, frame):
        self._require_active()
        native = _native_frame(frame.frame_idx)
        if self._previous_frame is not None and native != self._previous_frame + 1:
            raise ValueError("tracking input must be continuous native frames")
        # Validate and own frame buffers before mutable proposal diagnostics run.
        owned_frame = _observation_copy(
            frame, require_images=self.config.flow_proposals or self.config.mode != "nodes")
        try:
            self._advance_owned(owned_frame)
        except Exception:
            self._failed = True
            raise

    def _advance_owned(self, frame):
        cfg, states = self.config, self._states
        previous_observation = self._previous_observation
        rays = RayObservations.from_points(self.rig, frame.detections)
        targets = (
            None
            if cfg.mode == "nodes"
            else sample_pixels(
                frame, cfg.mode, cfg.positive_samples, cfg.near_samples, cfg.global_samples
            )
        )
        flow = None
        if cfg.flow_proposals and previous_observation is not None:
            flow = FlowObservations.from_frames(
                previous_observation, frame, self.rig, states[0].pose
            )
        proposals = (
            self._proposals(states, rays)
            if flow is None
            else self._proposals(states, rays, flow)
        )
        poses, local_cost, parts, mask_support = self._fit(proposals, states, rays, targets)
        path_cost = local_cost + np.array([states[p.parent].cost for p in proposals])
        order = np.argsort(path_cost, kind="stable")
        selected = []
        for candidate in order:
            if any(
                proposals[candidate].parent == proposals[j].parent
                and proposals[candidate].permutation == proposals[j].permutation
                and np.max(np.linalg.norm(poses[candidate] - poses[j], axis=-1)) < 0.5
                for j in selected
            ):
                continue
            selected.append(int(candidate))
            if len(selected) == cfg.beam_size:
                break
        next_states, records = [], []
        for candidate in selected:
            proposal = proposals[candidate]
            parent = states[proposal.parent]
            pose = poses[candidate]
            sources, observed, _ = supported_nodes(pose, proposal.plan, rays, self.rig)
            has_head_ray = (sources[:, :, :2] >= 0).any(axis=(1, 2))
            body_supported = observed[:, 1] | (mask_support[candidate] & has_head_ray)
            missing = np.where(body_supported, 0, parent.missing + 1)
            review = parent.review | (missing >= cfg.review_gap_frames)
            identity_margin = float("inf")
            for other in order:
                if other == candidate:
                    continue
                affected = self._different_identity(pose, poses[other])
                # At a spatial tie, permutation argmin alone chooses identity
                # lexically. Preserve explicit same-parent symmetry candidates.
                if proposals[other].parent == proposal.parent:
                    tied = (
                        np.max(np.linalg.norm(pose[:, 1] - poses[other, :, 1], axis=-1)) < 0.5
                    )
                    if tied:
                        affected |= np.asarray(proposal.permutation) != np.asarray(
                            proposals[other].permutation
                        )
                if not affected.any():
                    continue
                gap = float(path_cost[other] - path_cost[candidate])
                identity_margin = min(identity_margin, gap)
                if gap <= cfg.ambiguity_margin:
                    review |= affected
            displacement = pose[:, 1] - parent.pose[:, 1]
            velocity = 0.5 * parent.velocity + 0.5 * displacement
            speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
            velocity *= np.minimum(1.0, 2.0 / np.maximum(speed, 1e-12))
            velocity = np.where(body_supported[:, None], velocity, 0.8 * parent.velocity)
            node_missing, direct_pectoral_age = updated_ages(parent, observed)
            next_states.append(
                PersistentNodeState(
                    pose, velocity, missing, review, float(path_cost[candidate]),
                    node_missing, direct_pectoral_age
                )
            )
            records.append(
                dict(
                    frame_idx=frame.frame_idx,
                    pose_xyz=pose,
                    observed=observed,
                    sources=sources,
                    identity_review=review,
                    mask_supported=mask_support[candidate],
                    body_supported=body_supported,
                    missing=missing,
                    costs=local_cost[candidate],
                    path_costs=path_cost[candidate],
                    loss_parts=parts[candidate],
                    score_margin=identity_margin,
                    parent=proposal.parent,
                    proposal_kind=proposal.kind,
                    hypotheses=len(proposals),
                    proposal_flow_nodes=(
                        flow.node_valid if flow is not None else np.zeros((3, 3), bool)
                    ),
                    proposal_flow_parent=(0 if flow is not None else -1),
                    node_missing=node_missing,
                    direct_pectoral_age=direct_pectoral_age,
                )
            )
        self._history.append(records)
        self._states = next_states
        self._previous_frame = int(frame.frame_idx)
        self._previous_observation = frame if cfg.flow_proposals else None

    def finish(self):
        self._require_active()
        if not self._history:
            raise ValueError("empty tracking sequence")
        chosen, state_index = [], 0
        for records in reversed(self._history):
            record = records[state_index]
            chosen.append(record)
            state_index = record["parent"]
        chosen.reverse()
        result = {
            key: np.asarray([record[key] for record in chosen])
            for key in chosen[0] if key != "parent"
        }
        result["xyz"] = np.where(
            result["observed"][:, :, 1, None], result["pose_xyz"][:, :, 1], np.nan)
        result["accepted_xyz"] = np.where(
            result["identity_review"][:, :, None], np.nan, result["xyz"])
        result["body_xyz"] = np.where(
            result["body_supported"][:, :, None], result["pose_xyz"][:, :, 1], np.nan)
        self._finished = True
        return result

    def track(self, frames):
        self.begin()
        for frame in frames:
            self.advance(frame)
        return self.finish()

    def export_state(self):
        self._require_active()
        if not self._history:
            raise ValueError("checkpoint must follow a completed frame")
        if self._context() != self._initial_context:
            raise ValueError("tracker configuration or calibration changed after construction")
        flat = [record for records in self._history for record in records]
        arrays = {"history_offsets": np.r_[0, np.cumsum([len(r) for r in self._history])]}
        arrays.update({"history_" + key: np.asarray([r[key] for r in flat])
                       for key in RECORD_FIELDS})
        arrays.update({"state_" + key: np.asarray([getattr(s, key) for s in self._states])
                       for key in STATE_FIELDS})
        previous = self._previous_observation
        presence = {}
        if previous is not None:
            for field in ("detections", "images_bgr", "masks", "backgrounds_bgr"):
                for view, value in getattr(previous, field).items():
                    key = "previous_" + field + "_" + view
                    presence[key] = value is not None
                    if value is not None:
                        arrays[key] = value.copy()
        metadata = {
            "schema": SCHEMA, "context": self._context(),
            "initialize_count": self.initialize_count, "restore_count": self.restore_count,
            "processed_frames": len(self._history), "previous_frame": self._previous_frame,
            "record_fields": list(RECORD_FIELDS), "state_fields": list(STATE_FIELDS),
            "previous_observation": presence,
            "node_recovery_diagnostics": self.node_recovery_diagnostics,
        }
        # Also catch invalid retained structure before a successful save can occur.
        self._decode_state(arrays, metadata)
        return ({key: value.copy() for key, value in arrays.items()},
                json.loads(_canonical(metadata)))

    def _decode_state(self, supplied_arrays, metadata):
        if (metadata.get("schema") != SCHEMA or metadata.get("context") != self._context()
                or self._context() != self._initial_context
                or type(metadata.get("initialize_count")) is not int
                or metadata.get("initialize_count") != 1
                or metadata.get("record_fields") != list(RECORD_FIELDS)
                or metadata.get("state_fields") != list(STATE_FIELDS)):
            raise ValueError("checkpoint context, schema or algorithm differs")
        count = metadata.get("processed_frames")
        restores = metadata.get("restore_count")
        if (type(count) is not int or count < 1 or type(restores) is not int or restores < 0):
            raise ValueError("invalid checkpoint lifecycle counts")
        last = _native_frame(metadata.get("previous_frame"))
        presence = metadata.get("previous_observation")
        expected_presence = {
            "previous_" + field + "_" + view
            for field in ("detections", "images_bgr", "masks", "backgrounds_bgr")
            for view in VIEWS
        } if self.config.flow_proposals else set()
        if (not isinstance(presence, dict) or set(presence) != expected_presence
                or any(type(flag) is not bool for flag in presence.values())
                or any(not flag and not key.startswith("previous_masks_")
                       for key, flag in presence.items())):
            raise ValueError("checkpoint previous-frame evidence is incomplete")
        expected = ({"history_offsets"} | {"history_" + k for k in RECORD_FIELDS}
                    | {"state_" + k for k in STATE_FIELDS}
                    | {key for key, flag in presence.items() if flag})
        if set(supplied_arrays) != expected:
            raise ValueError("checkpoint arrays are missing or unexpected")
        arrays = {}
        for key, value in supplied_arrays.items():
            if not isinstance(value, np.ndarray) or value.dtype.hasobject:
                raise ValueError("checkpoint requires non-object NumPy arrays")
            arrays[key] = value.copy()
        offsets = arrays["history_offsets"]
        if (offsets.dtype != np.int64 or offsets.shape != (count + 1,)
                or offsets[0] != 0 or (np.diff(offsets) < 1).any()
                or (np.diff(offsets) > self.config.beam_size).any()):
            raise ValueError("checkpoint history beam offsets are invalid")
        nrows, nbeam = int(offsets[-1]), int(offsets[-1] - offsets[-2])
        field_specs = {
            "frame_idx": ((), "i"), "pose_xyz": ((3, 3, 3), "f"),
            "observed": ((3, 3), "b"), "sources": ((3, 3, 3), "i"),
            "identity_review": ((3,), "b"), "mask_supported": ((3,), "b"),
            "body_supported": ((3,), "b"), "missing": ((3,), "i"),
            "costs": ((), "f"), "path_costs": ((), "f"), "loss_parts": ((5,), "f"),
            "score_margin": ((), "f"), "parent": ((), "i"),
            "proposal_kind": ((), "U"), "hypotheses": ((), "i"),
            "proposal_flow_nodes": ((3, 3), "b"), "proposal_flow_parent": ((), "i"),
            "node_missing": ((3, 3), "i"), "direct_pectoral_age": ((3,), "i"),
        }
        for key, (shape, kind) in field_specs.items():
            value = arrays["history_" + key]
            if value.shape != (nrows, *shape) or value.dtype.kind != kind:
                raise ValueError("checkpoint history field shape/dtype is invalid: " + key)
            if kind == "f" and (np.isnan(value).any() or (
                    key != "score_margin" and not np.isfinite(value).all())):
                raise ValueError("nonfinite checkpoint history: " + key)
        history = []
        previous = [PersistentNodeState(self.initial.copy(), np.zeros((3, 3)),
                    np.zeros(3, int), np.zeros(3, bool), 0.0,
                    np.zeros((3, 3), int), np.zeros(3, int))]
        for index, (left, right) in enumerate(pairwise(offsets)):
            records, next_previous = [], []
            expected_frame = last - count + index + 1
            _native_frame(expected_frame)
            for row in range(int(left), int(right)):
                record = {key: arrays["history_" + key][row] for key in RECORD_FIELDS}
                parent_index = int(record["parent"])
                if (int(record["frame_idx"]) != expected_frame
                        or not 0 <= parent_index < len(previous)):
                    raise ValueError("checkpoint native sequence or parent chain is invalid")
                parent = previous[parent_index]
                observed, sources = record["observed"], record["sources"]
                if ((sources < -1).any() or (observed & ((sources >= 0).sum(1) < 2)).any()
                        or (record["pose_xyz"] < self.lower).any()
                        or (record["pose_xyz"] > self.upper).any()):
                    raise ValueError("checkpoint sources/support or engineering bounds invalid")
                for view in range(3):
                    for node in range(3):
                        used = sources[:, view, node]
                        used = used[used >= 0]
                        if len(used) != len(np.unique(used)):
                            raise ValueError("checkpoint reuses a source node")
                body = observed[:, 1] | (record["mask_supported"] &
                                        (sources[:, :, :2] >= 0).any(axis=(1, 2)))
                missing = np.where(body, 0, parent.missing + 1)
                node_age, pectoral_age = updated_ages(parent, observed)
                if (not np.array_equal(body, record["body_supported"])
                        or not np.array_equal(missing, record["missing"])
                        or not np.array_equal(node_age, record["node_missing"])
                        or not np.array_equal(pectoral_age, record["direct_pectoral_age"])
                        or (parent.review & ~record["identity_review"]).any()
                        or ((missing >= self.config.review_gap_frames) &
                            ~record["identity_review"]).any()
                        or not 1 <= int(record["hypotheses"]) <= self.config.max_hypotheses + 6
                        or int(record["proposal_flow_parent"]) !=
                        (0 if self.config.flow_proposals and index else -1)):
                    raise ValueError("checkpoint parent-specific ages/support/review are invalid")
                if not np.isclose(record["path_costs"], parent.cost + record["costs"],
                                  rtol=0, atol=0):
                    raise ValueError("checkpoint cumulative path cost disagrees with parent")
                displacement = record["pose_xyz"][:, 1] - parent.pose[:, 1]
                velocity = 0.5 * parent.velocity + 0.5 * displacement
                speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
                velocity *= np.minimum(1.0, 2.0 / np.maximum(speed, 1e-12))
                velocity = np.where(body[:, None], velocity, 0.8 * parent.velocity)
                next_previous.append(PersistentNodeState(
                    record["pose_xyz"], velocity, missing, record["identity_review"],
                    float(record["path_costs"]), node_age, pectoral_age))
                records.append(record)
            history.append(records)
            previous = next_previous
        for key in STATE_FIELDS:
            expected_value = np.asarray([getattr(state, key) for state in previous])
            actual = arrays["state_" + key]
            if (actual.dtype != expected_value.dtype or actual.shape != expected_value.shape
                    or actual.tobytes() != expected_value.tobytes()):
                raise ValueError("checkpoint active beam differs from retained history: " + key)
        if len(previous) != nbeam:
            raise ValueError("checkpoint active beam count differs")
        observation = None
        if presence:
            fields = {field: {view: arrays.get("previous_" + field + "_" + view)
                              for view in VIEWS}
                      for field in ("detections", "images_bgr", "masks", "backgrounds_bgr")}
            observation = _observation_copy(FrameObservation(last, **fields))
        diagnostic = metadata.get("node_recovery_diagnostics", {})
        expected_diagnostic = set(self.node_recovery_diagnostics)
        if set(diagnostic) != expected_diagnostic:
            raise ValueError("checkpoint recovery diagnostics are incomplete")
        scalar_counters = ("calls", "calls_with_age_eligible_fish",
                           "calls_with_additional_proposals")
        if any(type(diagnostic[k]) is not int or not 0 <= diagnostic[k] <= count
               for k in scalar_counters) or diagnostic["calls"] != count:
            raise ValueError("checkpoint recovery counters are invalid")
        if (not isinstance(diagnostic["proposed_by_fish"], list)
                or len(diagnostic["proposed_by_fish"]) != 3
                or any(type(v) is not int or v < 0 for v in diagnostic["proposed_by_fish"])
                or not isinstance(diagnostic["fish_status_counts"], dict)
                or any(not isinstance(k, str) or type(v) is not int or v < 0
                       for k, v in diagnostic["fish_status_counts"].items())):
            raise ValueError("checkpoint recovery per-fish counters are invalid")
        for key in ("original_budget", "additional_budget", "scope"):
            if diagnostic[key] != self.node_recovery_diagnostics[key]:
                raise ValueError("checkpoint recovery diagnostic contract changed")
        return history, previous, observation, diagnostic

    def restore_state(self, arrays, metadata):
        if self.initialize_count or self._states is not None or self._finished or self._failed:
            raise RuntimeError("restore requires a fresh, never-started tracker")
        history, states, observation, diagnostic = self._decode_state(arrays, metadata)
        self._history, self._states = history, states
        self._previous_frame = int(metadata["previous_frame"])
        self._previous_observation = observation
        self.initialize_count = 1
        self.restore_count = metadata["restore_count"] + 1
        self._recovery_calls = diagnostic["calls"]
        self._age_eligible_calls = diagnostic["calls_with_age_eligible_fish"]
        self._recovery_proposed_calls = diagnostic["calls_with_additional_proposals"]
        self._recovery_proposed_by_fish = np.asarray(
            diagnostic["proposed_by_fish"], np.int64).copy()
        self._recovery_status_counts = Counter(diagnostic["fish_status_counts"])


class AcceleratedContinuousPersistentNodeTracker(
    ContinuousPersistentNodeTracker, AcceleratedPersistentNodeTracker
):
    """The same continuation state; CUDA fit caches are rebuilt after process restore."""

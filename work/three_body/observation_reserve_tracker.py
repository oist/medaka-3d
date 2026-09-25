"""Reserve at most the sixth beam slot for a strictly supported observation.

The owned-frame scope explicitly copies the frozen Nodewise/Sparse guards.
The temporal method explicitly copies ContinuousPersistentNodeTracker's step,
replacing only admission and caching each candidate's unchanged support test.
Fitting, costs, proposals, flow, ambiguity, state updates and the 39-field codec
remain inherited or use their original expressions. This is a search policy,
not a physical identity label or a guarantee of later retention.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from time import perf_counter

import continuous_tracker as continuous
import nodewise_body_row_tracker as nodewise
import numpy as np
import sparse_instance_rescore_tracker as sparse
from observation_reserve_selection import (
    observation_eligibility,
    select_with_observation_reserve,
)

NODEWISE_TRACKER_SHA = "b574ce26481c2c1457ceb1428659aa7bce25783e36957fd2eb9b6399f52b8d16"
TEMPORAL_SOURCE_SHA = "91a811cab27a8a26e605aa00159fdb2a2b450bbcca1a2505e49543f58ef7c7e2"
SPARSE_TRACKER_SHA = "d041aff3a0d8cbd71341dbe67dd9b5a7a1e2e7aeb165f55e4b08f8fabdb8fb79"
SELECTION_SOURCE_SHA = "b4b69b67e488db00fdff49b278a0c39f3966cc24e55749e2b1ca8251d1196853"


@dataclass(frozen=True)
class ObservationReservePolicy:
    effective_from_native: int

    def __post_init__(self):
        if type(self.effective_from_native) is not int or self.effective_from_native < 0:
            raise ValueError("nonnegative integer observation-reserve boundary required")


class ObservationReserveContinuousTracker(nodewise.BodyRowContinuousTracker):
    def __init__(self, *args, observation_reserve_effective_from_native, **kwargs):
        self.observation_reserve_policy = ObservationReservePolicy(
            observation_reserve_effective_from_native
        )
        self.last_observation_reserve_diagnostics = None
        self._observation_reserve_frame = None
        super().__init__(*args, **kwargs)
        if type(self.config.beam_size) is not int or self.config.beam_size != 6:
            raise ValueError("observation reserve requires the unchanged six-candidate beam")

    def _context(self):
        context = super()._context()
        context["observation_reserve_policy"] = dict(
            schema="observation_reserve_policy_v1",
            **asdict(self.observation_reserve_policy),
            nodewise_tracker_source_sha256=NODEWISE_TRACKER_SHA,
            temporal_source_sha256=TEMPORAL_SOURCE_SHA,
            sparse_tracker_source_sha256=SPARSE_TRACKER_SHA,
            selection_source_sha256=SELECTION_SOURCE_SHA,
            eligibility=(
                "all three views/all MCT post-fit supported sources; whole exclusive rows; "
                "node and silhouette no worse than baseline best"
            ),
            fitted_support_thresholds_unchanged=True,
            candidate_support_evaluations=(
                "once per fitted candidate; selected records reuse these values"
            ),
            protected_prefix_count=5,
            maximum_reserved_slots=1,
            original_fit_local_path_and_five_plus_sixth_unchanged=True,
            baseline_state0_and_first_five_unchanged=True,
            no_op_uses_original_selection=True,
            same_parent_permutation_duplicate_distance_mm=0.5,
            ambiguity_uses_all_candidates_and_original_path_order=True,
            original_support_age_velocity_review_expressions_unchanged=True,
            admission_overrides_inherited_body_row_temporal_policy_only=True,
            reservation_does_not_guarantee_future_retention=True,
            costs_reweighted=False,
            original_history_rewritten=False,
            clears_identity_review=False,
            exact_global_search=False,
            physical_ID_truth=False,
        )
        return context

    def _clear_frame_diagnostics(self):
        self.last_observation_reserve_diagnostics = None
        self.last_body_row_diagnostics = None
        self.last_instance_diagnostics = None

    def advance(self, frame):
        self._clear_frame_diagnostics()
        try:
            return super().advance(frame)
        except BaseException:
            self._clear_frame_diagnostics()
            raise

    def _advance_owned(self, frame):
        """Explicit Nodewise/Sparse scope; inherited explicit base call is bypassed."""
        self._clear_frame_diagnostics()
        if any(
            value is not None
            for value in (
                self._body_row_frame,
                self._instance_frame,
                self._observation_reserve_frame,
            )
        ):
            raise RuntimeError("owned frame scope is already active")
        native = continuous._native_frame(frame.frame_idx)
        if native < max(
            self.body_row_policy.effective_from_native,
            self.sparse_backend_policy.effective_from_native,
            self.observation_reserve_policy.effective_from_native,
        ):
            raise ValueError("execution precedes a declared policy boundary")
        self._body_row_frame = self._instance_frame = self._observation_reserve_frame = native
        self._instance_observations, self._instance_gate_seconds = {}, {}
        try:
            if native >= sparse.POLICY.effective_from_native:
                for view in sparse.VIEWS:
                    started = perf_counter()
                    self._instance_observations[view] = sparse.prepare_sparse_instance_observation(
                        frame.masks.get(view), frame.detections.get(view), sparse.DEFAULT_CONFIG
                    )
                    self._instance_gate_seconds[view] = perf_counter() - started
            self._advance_temporal_with_reserve(frame)
            if (
                self.last_body_row_diagnostics is None
                or self.last_instance_diagnostics is None
                or self.last_observation_reserve_diagnostics is None
            ):
                raise RuntimeError("completed frame lacks proposal, fit or reserve diagnostics")
            self.last_body_row_diagnostics["frame_completed"] = True
            self.last_observation_reserve_diagnostics["frame_completed"] = True
        except BaseException:
            self._clear_frame_diagnostics()
            raise
        finally:
            self._body_row_frame = self._instance_frame = self._observation_reserve_frame = None
            self._instance_observations = self._instance_gate_seconds = None

    def _advance_temporal_with_reserve(self, frame):
        # Source-copy: continuous_tracker.py:165-280; only selection/support caching differs.
        if not (
            self._observation_reserve_frame
            == self._instance_frame
            == self._body_row_frame
            == continuous._native_frame(frame.frame_idx)
        ):
            raise RuntimeError("temporal reserve requires the current owned frame")
        cfg, states = self.config, self._states
        previous_observation = self._previous_observation
        rays = continuous.RayObservations.from_points(self.rig, frame.detections)
        targets = (
            None
            if cfg.mode == "nodes"
            else continuous.sample_pixels(
                frame, cfg.mode, cfg.positive_samples, cfg.near_samples, cfg.global_samples
            )
        )
        flow = None
        if cfg.flow_proposals and previous_observation is not None:
            flow = continuous.FlowObservations.from_frames(
                previous_observation, frame, self.rig, states[0].pose
            )
        proposals = (
            self._proposals(states, rays) if flow is None else self._proposals(states, rays, flow)
        )
        poses, local_cost, parts, mask_support = self._fit(proposals, states, rays, targets)
        path_cost = local_cost + np.array([states[p.parent].cost for p in proposals])
        order = np.argsort(path_cost, kind="stable")
        support_started = perf_counter()
        support_cache = [
            continuous.supported_nodes(pose, proposal.plan, rays, self.rig)
            for pose, proposal in zip(poses, proposals, strict=True)
        ]
        supported_sources = np.stack([value[0] for value in support_cache])
        observed_nodes = np.stack([value[1] for value in support_cache])
        support_seconds = perf_counter() - support_started
        eligible, eligibility_checks = observation_eligibility(
            supported_sources, observed_nodes, parts, int(order[0])
        )
        selection_started = perf_counter()
        selected, selection = select_with_observation_reserve(
            poses,
            path_cost,
            np.asarray([p.parent for p in proposals], dtype=np.int64),
            np.asarray([p.permutation for p in proposals], dtype=np.int64),
            eligible,
            cfg.beam_size,
        )
        self.last_observation_reserve_diagnostics = dict(
            schema="observation_reserve_frame_diagnostics_v1",
            frame_idx=int(frame.frame_idx),
            frame_completed=False,
            policy=copy.deepcopy(self._context()["observation_reserve_policy"]),
            candidate_index=np.arange(len(proposals), dtype=np.int64),
            candidate_kind=[p.kind for p in proposals],
            candidate_parent=np.asarray([p.parent for p in proposals], dtype=np.int64),
            parent_pose=np.asarray([state.pose for state in states]).copy(),
            parent_velocity=np.asarray([state.velocity for state in states]).copy(),
            parent_missing=np.asarray([state.missing for state in states]).copy(),
            parent_node_missing=np.asarray([state.node_missing for state in states]).copy(),
            parent_direct_pectoral_age=np.asarray(
                [state.direct_pectoral_age for state in states]
            ).copy(),
            candidate_mask_support=mask_support.copy(),
            supported_sources=supported_sources,
            observed=observed_nodes,
            eligible=eligible,
            eligibility_checks=eligibility_checks,
            local_cost=local_cost.copy(),
            path_cost=path_cost.copy(),
            original_five_loss_parts=parts.copy(),
            baseline_best_index=int(order[0]),
            selection=selection,
            support_evaluation_count=len(support_cache),
            selected_support_reused=True,
            support_seconds=support_seconds,
            selection_seconds=perf_counter() - selection_started,
            current_beam_not_final_path=True,
            physical_ID_truth=False,
        )
        next_states, records = [], []
        for candidate in selected:
            proposal = proposals[candidate]
            parent = states[proposal.parent]
            pose = poses[candidate]
            sources, observed, _ = support_cache[candidate]
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
                    tied = np.max(np.linalg.norm(pose[:, 1] - poses[other, :, 1], axis=-1)) < 0.5
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
            node_missing, direct_pectoral_age = continuous.updated_ages(parent, observed)
            next_states.append(
                continuous.PersistentNodeState(
                    pose,
                    velocity,
                    missing,
                    review,
                    float(path_cost[candidate]),
                    node_missing,
                    direct_pectoral_age,
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

    def export_observation_reserve_diagnostics(self):
        value = self.last_observation_reserve_diagnostics
        if (
            value is None
            or not value["frame_completed"]
            or self._observation_reserve_frame is not None
            or value["frame_idx"] != self.previous_frame
        ):
            raise RuntimeError("no completed observation-reserve diagnostic is available")
        return copy.deepcopy(value)


def import_nodewise_state(target, original, arrays, metadata):
    """Real Nodewise decode, context-only migration and exactly one target restore."""
    if (
        type(target) is not ObservationReserveContinuousTracker
        or type(original) is not nodewise.BodyRowContinuousTracker
    ):
        raise ValueError("explicit original Nodewise and target observation-reserve required")
    for tracker in (target, original):
        if (
            tracker.initialize_count
            or tracker._states is not None
            or tracker._history
            or tracker.previous_frame is not None
            or tracker._finished
            or tracker._failed
            or tracker.restore_count
        ):
            raise ValueError("migration requires fresh uninitialized trackers")
    old_context, new_context = original._context(), target._context()
    if (
        metadata.get("context") != old_context
        or original._initial_context != old_context
        or target._initial_context != new_context
    ):
        raise ValueError("source or constructor-finalized context differs")
    expected = copy.deepcopy(new_context)
    policy = expected.pop("observation_reserve_policy")
    expected["implementation"] = old_context["implementation"]
    if expected != old_context:
        raise ValueError("migration may add reserve policy only; all Nodewise context must match")
    previous = metadata.get("previous_frame")
    if type(previous) is not int or previous + 1 != policy["effective_from_native"]:
        raise ValueError("reserve boundary must immediately follow source previous frame")
    if len(arrays) != 39:
        raise ValueError("migration requires all39 source state arrays")
    before, original_metadata = nodewise.array_signature(arrays), copy.deepcopy(metadata)
    original._decode_state(arrays, metadata)
    adapted = copy.deepcopy(metadata)
    adapted["context"] = new_context
    target.restore_state(arrays, adapted)
    restored_arrays, restored_metadata = target.export_state()
    expected_metadata = copy.deepcopy(adapted)
    expected_metadata["restore_count"] += 1
    if (
        before != nodewise.array_signature(arrays)
        or before != nodewise.array_signature(restored_arrays)
        or metadata != original_metadata
        or restored_metadata != expected_metadata
        or target.initialize_count != 1
        or original.initialize_count != 0
        or original.restore_count != 0
        or target.restore_count != metadata["restore_count"] + 1
        or target.processed_frames != metadata["processed_frames"]
        or target.previous_frame != previous
    ):
        raise ValueError("reserve migration changed historical bytes or undeclared metadata")
    return dict(
        schema="observation_reserve_nodewise_state_import_v1",
        status="ok",
        old_context=old_context,
        new_context=new_context,
        original_metadata=original_metadata,
        imported_metadata=restored_metadata,
        arrays=before,
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        logical_initialize_count=1,
        new_begin_calls=0,
        added_restore_count=1,
        processed_frames=target.processed_frames,
        previous_frame=previous,
        effective_from_native=policy["effective_from_native"],
        old_history_rewritten=False,
        review_cleared_by_migration=False,
        companion_diagnostics_required_each_future_frame=True,
    )

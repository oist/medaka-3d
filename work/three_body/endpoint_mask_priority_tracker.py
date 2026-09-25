"""Prefer post-fit non-split bodies without changing fitted or cumulative costs.

The finite old objective remains in every state and record. Current-frame mask
priority changes beam admission/order, so the selected state0 need not minimize
that objective. Split candidates remain backups and all-split frames use the
original order. The original all-candidate ambiguity calculation is retained.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from time import perf_counter

import continuous_tracker as continuous
import endpoint_mask_conflict_score as endpoint_score
import nodewise_body_row_tracker as nodewise
import numpy as np
import sparse_instance_rescore_tracker as sparse

NODEWISE_TRACKER_SHA = "b574ce26481c2c1457ceb1428659aa7bce25783e36957fd2eb9b6399f52b8d16"
TEMPORAL_SOURCE_SHA = "91a811cab27a8a26e605aa00159fdb2a2b450bbcca1a2505e49543f58ef7c7e2"
ENDPOINT_SCORE_SHA = "8739a48c7fa72fa2f58cb720051fcea4ab81da609612763f42e959cc7ad33d60"


@dataclass(frozen=True)
class EndpointMaskPriorityPolicy:
    effective_from_native: int
    endpoint_tolerance_px: float = 6.0

    def __post_init__(self):
        if type(self.effective_from_native) is not int or self.effective_from_native < 0:
            raise ValueError("nonnegative integer endpoint-mask boundary required")
        if type(self.endpoint_tolerance_px) is not float or self.endpoint_tolerance_px != 6.0:
            raise ValueError("the fixed6px endpoint policy is required")


def candidate_orders(path_cost, split_by_body):
    path = np.asarray(path_cost)
    split = np.asarray(split_by_body)
    if (
        path.ndim != 1
        or not len(path)
        or not np.isfinite(path).all()
        or split.shape != (len(path), 3)
        or split.dtype != np.dtype(bool)
    ):
        raise ValueError("finite candidate costs and boolean body split flags required")
    cost_order = np.argsort(path, kind="stable")
    has_split = split.any(axis=1)
    priority = np.concatenate(
        (cost_order[~has_split[cost_order]], cost_order[has_split[cost_order]])
    )
    return cost_order, priority, bool(has_split.all())


def retain_candidates(proposals, poses, order, beam_size):
    """Frozen same-parent/permutation and0.5mm dedup, applied to the supplied order."""
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
        if len(selected) == beam_size:
            break
    return selected


class EndpointMaskPriorityTracker(nodewise.BodyRowContinuousTracker):
    def __init__(self, *args, endpoint_mask_effective_from_native, **kwargs):
        self.endpoint_mask_policy = EndpointMaskPriorityPolicy(endpoint_mask_effective_from_native)
        self.last_endpoint_mask_diagnostics = None
        self._endpoint_mask_frame = None
        self._endpoint_mask_observations = None
        self._endpoint_mask_prepare_seconds = None
        kwargs.setdefault("body_row_effective_from_native", endpoint_mask_effective_from_native)
        super().__init__(*args, **kwargs)

    def _context(self):
        context = super()._context()
        context["endpoint_mask_priority_policy"] = dict(
            schema="endpoint_mask_priority_policy_v1",
            **asdict(self.endpoint_mask_policy),
            nodewise_tracker_source_sha256=NODEWISE_TRACKER_SHA,
            temporal_source_sha256=TEMPORAL_SOURCE_SHA,
            endpoint_score_source_sha256=ENDPOINT_SCORE_SHA,
            observation_scope="all supplied nonempty masks, independent of SLEAP rows",
            candidate_scope="all post-fit candidates, all bodies and all views",
            split_requires_three_uniquely_covered_endpoints=True,
            uncovered_ambiguous_or_unavailable_is_not_split=True,
            not_split_does_not_certify_body_or_identity=True,
            selection_order="stable old path order among not_split, then among split backups",
            all_candidates_split="exact old stable path order fallback; unresolved",
            beam_size_and_same_parent_permutation_dedup_unchanged=True,
            original_fit_local_path_costs_and_five_plus_sixth_terms_unchanged=True,
            rank_or_penalty_added_to_cost=False,
            selected_state0_need_not_minimize_numeric_cost=True,
            temporal_selection_overrides_inherited_body_row_policy_only=True,
            original_support_ages_velocity_math_unchanged=True,
            ambiguity_uses_all_candidates_and_original_cost_order=True,
            selected_split_body_review_is_ORed=True,
            original_history_rewritten=False,
            exact_global_search=False,
            physical_ID_truth=False,
            clears_identity_review=False,
        )
        return context

    def _clear_frame_diagnostics(self):
        self.last_endpoint_mask_diagnostics = None
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
        """Explicit Nodewise/Sparse frame scope, calling the new temporal loop."""
        self._clear_frame_diagnostics()
        if (
            self._body_row_frame is not None
            or self._instance_frame is not None
            or self._endpoint_mask_frame is not None
        ):
            raise RuntimeError("owned frame scope is already active")
        native = continuous._native_frame(frame.frame_idx)
        if native < max(
            self.body_row_policy.effective_from_native,
            self.sparse_backend_policy.effective_from_native,
            self.endpoint_mask_policy.effective_from_native,
        ):
            raise ValueError("execution precedes a declared policy boundary")
        self._body_row_frame = self._instance_frame = self._endpoint_mask_frame = native
        self._instance_observations, self._instance_gate_seconds = {}, {}
        try:
            if native >= sparse.POLICY.effective_from_native:
                for view in sparse.VIEWS:
                    started = perf_counter()
                    self._instance_observations[view] = sparse.prepare_sparse_instance_observation(
                        frame.masks.get(view), frame.detections.get(view), sparse.DEFAULT_CONFIG
                    )
                    self._instance_gate_seconds[view] = perf_counter() - started
            self._prepare_endpoint_masks(frame)
            self._advance_temporal_with_priority(frame)
            if (
                self.last_body_row_diagnostics is None
                or self.last_endpoint_mask_diagnostics is None
            ):
                raise RuntimeError("completed frame lacks proposal or endpoint diagnostics")
            self.last_body_row_diagnostics["frame_completed"] = True
            self.last_endpoint_mask_diagnostics["frame_completed"] = True
        except BaseException:
            self._clear_frame_diagnostics()
            raise
        finally:
            self._body_row_frame = self._instance_frame = self._endpoint_mask_frame = None
            self._instance_observations = self._instance_gate_seconds = None
            self._endpoint_mask_observations = None
            self._endpoint_mask_prepare_seconds = None

    def _prepare_endpoint_masks(self, frame):
        if self._endpoint_mask_frame != continuous._native_frame(frame.frame_idx):
            raise RuntimeError("endpoint masks must belong to the current owned frame")
        observations, seconds = {}, {}
        for view in sparse.VIEWS:
            started = perf_counter()
            observations[view] = endpoint_score.prepare_endpoint_mask_observation(
                frame.masks.get(view)
            )
            seconds[view] = perf_counter() - started
        self._endpoint_mask_observations = observations
        self._endpoint_mask_prepare_seconds = seconds

    def _fit(self, proposals, states, rays, targets):
        self.last_endpoint_mask_diagnostics = None
        if (
            self._endpoint_mask_frame is None
            or self._endpoint_mask_observations is None
            or self._endpoint_mask_frame != self._instance_frame
            or self._endpoint_mask_frame != self._body_row_frame
        ):
            raise RuntimeError("endpoint-mask priority requires a current owned frame")
        result = super()._fit(proposals, states, rays, targets)
        poses, local_cost, parts, mask_support = result
        count = len(proposals)
        if np.shape(poses) != (count, 3, 3, 3):
            raise ValueError("fitted candidate count or body pose shape differs")
        preserved = dict(poses=poses, local_cost=local_cost, parts=parts, mask_support=mask_support)
        preserved.update({f"plan_{i}": p.plan for i, p in enumerate(proposals)})
        before = nodewise.array_signature(preserved)
        views, seconds = {}, {}
        for view in sparse.VIEWS:
            started = perf_counter()
            scored = endpoint_score.score_endpoint_mask_conflicts(
                poses,
                self.rig.camera(view).project,
                self._endpoint_mask_observations[view],
                tolerance_px=self.endpoint_mask_policy.endpoint_tolerance_px,
            )
            for key in ("split_by_body", "unknown_by_body", "geometry_finite_by_body"):
                value = scored[key]
                if np.shape(value) != (count, 3) or np.asarray(value).dtype != np.dtype(bool):
                    raise ValueError("endpoint scorer candidate/body boolean shape differs")
            views[view] = scored
            seconds[view] = perf_counter() - started
        if before != nodewise.array_signature(preserved):
            raise ValueError("endpoint scoring modified original fit arrays or plans")
        split = np.stack([views[v]["split_by_body"] for v in sparse.VIEWS], axis=-1)
        unknown = np.stack([views[v]["unknown_by_body"] for v in sparse.VIEWS], axis=-1)
        self.last_endpoint_mask_diagnostics = dict(
            schema="endpoint_mask_frame_diagnostics_v1",
            frame_idx=self._endpoint_mask_frame,
            frame_completed=False,
            policy=copy.deepcopy(self._context()["endpoint_mask_priority_policy"]),
            candidate_index=np.arange(count, dtype=np.int64),
            candidate_count=count,
            candidate_parent=np.asarray([p.parent for p in proposals], dtype=np.int64),
            candidate_kind=[p.kind for p in proposals],
            candidate_local_cost=np.asarray(local_cost).copy(),
            view_order=list(sparse.VIEWS),
            views=views,
            split_by_candidate_body_view=split,
            candidate_split_by_body=split.any(axis=-1),
            candidate_has_split=split.any(axis=(1, 2)),
            unknown_by_candidate_body_view=unknown,
            candidate_unknown_by_body=unknown.any(axis=-1),
            candidate_has_unknown=unknown.any(axis=(1, 2)),
            original_fit_arrays_and_plans=before,
            fit_arrays_and_plans_byte_preserved=True,
            preparation_seconds_by_view=copy.deepcopy(self._endpoint_mask_prepare_seconds),
            scoring_seconds_by_view=seconds,
            observation_preparations_per_view=1,
            postfit_scorer_calls_per_view=1,
            original_costs_unchanged=True,
            masks_are_identity_truth=False,
            not_split_includes_unknown=True,
            unknown_does_not_certify_body_correctness=True,
            unknown_causes_automatic_review=False,
            observed_support_unchanged=True,
            clears_identity_review=False,
        )
        # Return exactly the inherited tuple and its original array objects.
        return result

    def _advance_temporal_with_priority(self, frame):
        """Explicit frozen continuous165-272 loop; selection/review deltas are marked."""
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
        split = self.last_endpoint_mask_diagnostics["candidate_split_by_body"]
        cost_order, selection_order, fallback = candidate_orders(path_cost, split)
        selected = retain_candidates(proposals, poses, selection_order, cfg.beam_size)
        baseline_selected = retain_candidates(proposals, poses, cost_order, cfg.beam_size)
        next_states, records, added_review = [], [], []
        for candidate in selected:
            proposal = proposals[candidate]
            parent = states[proposal.parent]
            pose = poses[candidate]
            sources, observed, _ = continuous.supported_nodes(pose, proposal.plan, rays, self.rig)
            has_head_ray = (sources[:, :, :2] >= 0).any(axis=(1, 2))
            body_supported = observed[:, 1] | (mask_support[candidate] & has_head_ray)
            missing = np.where(body_supported, 0, parent.missing + 1)
            review = parent.review | (missing >= cfg.review_gap_frames)
            identity_margin = float("inf")
            # All original candidates remain competitors, including split backups.
            for other in cost_order:
                if other == candidate:
                    continue
                affected = self._different_identity(pose, poses[other])
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
            # The only new review rule; preserve it in record and active state.
            added_review.append(split[candidate] & ~review)
            review = review | split[candidate]
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
                    proposal_flow_nodes=flow.node_valid
                    if flow is not None
                    else np.zeros((3, 3), bool),
                    proposal_flow_parent=0 if flow is not None else -1,
                    node_missing=node_missing,
                    direct_pectoral_age=direct_pectoral_age,
                )
            )
        self.last_endpoint_mask_diagnostics.update(
            candidate_path_cost=path_cost.copy(),
            cost_order=cost_order,
            selection_order=selection_order,
            selected_candidate_indices=np.asarray(selected, np.int64),
            baseline_selected_candidate_indices=np.asarray(baseline_selected, np.int64),
            all_candidates_split_fallback=fallback,
            selection_changed_from_baseline=selected != baseline_selected,
            selected_split_by_body=split[selected].copy(),
            added_identity_review_by_selected_body=np.asarray(added_review, dtype=bool),
            selected_original_score_margin=np.asarray([r["score_margin"] for r in records]),
        )
        self._history.append(records)
        self._states = next_states
        self._previous_frame = int(frame.frame_idx)
        self._previous_observation = frame if cfg.flow_proposals else None

    def export_endpoint_mask_diagnostics(self):
        value = self.last_endpoint_mask_diagnostics
        if (
            value is None
            or not value["frame_completed"]
            or self._endpoint_mask_frame is not None
            or value["frame_idx"] != self.previous_frame
        ):
            raise RuntimeError("no completed endpoint-mask frame diagnostic is available")
        return copy.deepcopy(value)


def import_sparse_state(target, original, arrays, metadata):
    """Validate the real Sparse state before adding only the two new policies."""
    if (
        type(target) is not EndpointMaskPriorityTracker
        or type(original) is not sparse.SparseInstanceRescoreContinuousTracker
    ):
        raise ValueError(
            "explicit original Sparse and target EndpointMaskPriority implementations required"
        )
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
    body_policy = expected.pop("body_row_policy")
    mask_policy = expected.pop("endpoint_mask_priority_policy")
    expected["implementation"] = old_context["implementation"]
    if expected != old_context:
        raise ValueError(
            "only body-row/endpoint-mask policies may be added; seed/config/rig/bounds unchanged"
        )
    previous = metadata.get("previous_frame")
    if (
        type(previous) is not int
        or previous + 1 != body_policy["effective_from_native"]
        or previous + 1 != mask_policy["effective_from_native"]
    ):
        raise ValueError("both new boundaries must immediately follow source previous frame")
    if len(arrays) != 39:
        raise ValueError("all39 source state arrays required")
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
        or target.restore_count != metadata["restore_count"] + 1
        or target.processed_frames != metadata["processed_frames"]
        or target.previous_frame != previous
    ):
        raise ValueError("transition changed historical bytes or undeclared metadata")
    return dict(
        schema="endpoint_mask_sparse_state_import_v1",
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
        effective_from_native=mask_policy["effective_from_native"],
        old_history_rewritten=False,
        review_cleared_by_migration=False,
        companion_diagnostics_required_each_future_frame=True,
    )

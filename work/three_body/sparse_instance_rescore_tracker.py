"""Execution-only sparse distance backend for the frozen instance-rescore policy.

The owned-frame and post-fit hooks are explicit source copies of the frozen
f625513e... implementation, using statically imported sparse distance functions
and the same FastTail fitter/temporal loop. There is no runtime substitution.
The sixth-term arithmetic and original scientific policy remain unchanged.

Backend metadata declares its first native frame; the imported historical
arrays were produced under their recorded source context and are not rewritten.
Every future frame still requires companion diagnostics alongside the39 arrays.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from directional_chest_tracker import array_signature
from fast_tail_recovery_tracker import FastTailRecoveryContinuousTracker
from instance_body_score import DEFAULT_CONFIG
from instance_rescore_tracker import POLICY, InstanceRescoreContinuousTracker
from sparse_instance_body_score import (
    prepare_sparse_instance_observation,
    score_sparse_instance_bodies,
)

from medaka3d.types import VIEWS

DENSE_TRACKER_SHA = "f625513ed379e17241ad120c8a454c56b398764a1341ab7756790062231863e5"
DENSE_DISTANCE_SHA = "cb9e04a20160f2b681aa1b7cca46ed5b4017c7b05760f9dbbeff691bf058cb87"
SPARSE_DISTANCE_SHA = "68c41510d64a41098c6c1daf5e98621723f37c2f0f35730a68bd0e896517a634"
QUALIFICATION_SHA = "941ca436a5e59fca5a63794483d315da3950402aafdb4ce13c34630e98014125"


@dataclass(frozen=True)
class SparseDistanceBackendPolicy:
    effective_from_native: int = 78020

    def __post_init__(self):
        if type(self.effective_from_native) is not int or self.effective_from_native < 0:
            raise ValueError("nonnegative integer sparse backend transition frame required")


class SparseInstanceRescoreContinuousTracker(InstanceRescoreContinuousTracker):
    def __init__(self, *args, sparse_effective_from_native=78020, **kwargs):
        # The base constructor captures _context(), so bind the transition first.
        self.sparse_backend_policy = SparseDistanceBackendPolicy(sparse_effective_from_native)
        super().__init__(*args, **kwargs)

    def _context(self):
        context = super()._context()
        context["sparse_distance_backend"] = dict(
            schema="sparse_instance_distance_backend_v1",
            **asdict(self.sparse_backend_policy),
            backend="cKDTree exact lattice corners then scipy linear interpolation",
            dense_tracker_source_sha256=DENSE_TRACKER_SHA,
            dense_distance_source_sha256=DENSE_DISTANCE_SHA,
            sparse_distance_source_sha256=SPARSE_DISTANCE_SHA,
            saved_workload_qualification_sha256=QUALIFICATION_SHA,
            qualified_saved_interval=[76360, 78020],
            qualification_scope="all retained dense diagnostics/costs on saved1660 candidate banks",
            prior_history_rewritten=False,
            backend_applied_retroactively_to_prior_history=False,
            science_policy_unchanged=True,
            original_five_parts_and_sixth_term_unchanged=True,
            gradient_objective_and_optimizer_unchanged=True,
            continuous_rollout_equivalence_preclaimed=False,
            paired_or_end_to_end_speedup_preclaimed=False,
            clears_identity_review=False,
        )
        return context

    def _advance_owned(self, frame):
        if int(frame.frame_idx) < self.sparse_backend_policy.effective_from_native:
            raise ValueError("sparse execution precedes its declared transition boundary")
        # advance() in the frozen base owns/validates this frame before reaching
        # here. Never prepare from _previous_observation or keep EDTs across time.
        if self._instance_frame is not None:
            raise RuntimeError("instance observation scope is already active")
        self.last_instance_diagnostics = None
        self._instance_frame = int(frame.frame_idx)
        self._instance_observations = {}
        self._instance_gate_seconds = {}
        try:
            if self._instance_frame >= POLICY.effective_from_native:
                for view in VIEWS:
                    started = perf_counter()
                    self._instance_observations[view] = prepare_sparse_instance_observation(
                        frame.masks.get(view), frame.detections.get(view), DEFAULT_CONFIG
                    )
                    self._instance_gate_seconds[view] = perf_counter() - started
            return FastTailRecoveryContinuousTracker._advance_owned(self, frame)
        except Exception:
            # A completed fit is not a completed temporal frame. Never export a
            # stale/success-looking companion if inherited support/update fails.
            self.last_instance_diagnostics = None
            raise
        finally:
            self._instance_frame = None
            self._instance_observations = None
            self._instance_gate_seconds = None

    def _fit(self, proposals, states, rays, targets):
        if self._instance_frame is None:
            raise RuntimeError("post-fit instance rescoring requires a current owned frame")
        # Exactly the same fitting call and objects; no proposal/plan/pose edits.
        poses, base_cost, parts, support = FastTailRecoveryContinuousTracker._fit(
            self, proposals, states, rays, targets
        )
        count = len(poses)
        if (
            np.shape(base_cost) != (count,)
            or np.shape(parts) != (count, 5)
            or np.shape(support) != (count, 3)
            or not np.isfinite(base_cost).all()
        ):
            raise ValueError("fitter must return finite costs and the original five parts")
        contributions = np.zeros((count, len(VIEWS)), dtype=np.float64)
        view_details, score_seconds = {}, {}
        for vi, view in enumerate(VIEWS):
            if self._instance_frame < POLICY.effective_from_native:
                view_details[view] = dict(
                    eligible=False, reason="policy_not_yet_effective", neutral=True
                )
                score_seconds[view] = 0.0
                continue
            started = perf_counter()
            scored = score_sparse_instance_bodies(
                poses, self.rig.camera(view).project, self._instance_observations[view]
            )
            raw = np.asarray(scored["best_cost"], dtype=np.float64)
            if raw.shape != (count,) or np.any(np.isfinite(raw) & (raw < 0)):
                raise ValueError("instance scorer must return nonnegative per-candidate costs")
            normalized = raw / POLICY.bodies_per_view
            invalid = ~np.isfinite(raw)
            capped = np.clip(normalized, 0.0, POLICY.cap_per_view)
            capped[invalid] = POLICY.cap_per_view
            if not scored["eligible"]:
                # Eligibility belongs only to the observation, never candidate quality.
                capped[:] = 0.0
            contributions[:, vi] = capped
            # Compact companion payload: raw severe costs, all six assignments,
            # ties/invalids and capped contributions remain separately visible.
            view_details[view] = {
                key: copy.deepcopy(scored[key])
                for key in (
                    "eligible",
                    "reason",
                    "neutral",
                    "observation_evidence",
                    "best_assignment",
                    "assignment_costs",
                    "optimal_assignment_mask",
                    "body_instance_costs",
                )
            }
            view_details[view].update(
                raw_uncapped_scene_cost=raw.copy(),
                uncapped_mean_body_cost=normalized.copy(),
                capped_view_contribution=capped.copy(),
                nonfinite_score=invalid.copy(),
                candidate_projection_finite=copy.deepcopy(
                    scored.get("candidate_projection_finite")
                ),
                cap_applied=(normalized > POLICY.cap_per_view) | invalid,
            )
            score_seconds[view] = perf_counter() - started
        penalty = (
            float(self.config.mask_weight)
            * contributions.sum(axis=1)
            / POLICY.fixed_camera_denominator
        )
        adjusted = np.asarray(base_cost, dtype=np.float64) + penalty
        if not np.isfinite(adjusted).all():
            raise ValueError("instance-adjusted candidate costs must remain finite")
        parents = np.asarray([proposal.parent for proposal in proposals], dtype=np.int64)
        parent_cost = np.asarray([states[parent].cost for parent in parents], dtype=np.float64)
        self.last_instance_diagnostics = dict(
            schema="instance_rescore_frame_diagnostics_v1",
            frame_idx=self._instance_frame,
            distance_backend=copy.deepcopy(self._context()["sparse_distance_backend"]),
            policy=copy.deepcopy(self._context()["instance_rescore_policy"]),
            candidate_index=np.arange(count, dtype=np.int64),
            candidate_pose_xyz=poses.copy(),
            candidate_parent=parents,
            candidate_plan=np.asarray([proposal.plan for proposal in proposals]).copy(),
            candidate_kind=[proposal.kind for proposal in proposals],
            candidate_permutation=np.asarray([proposal.permutation for proposal in proposals]),
            base_local_cost=np.asarray(base_cost).copy(),
            original_five_loss_parts=parts.copy(),
            sixth_instance_term=penalty.copy(),
            adjusted_local_cost=adjusted.copy(),
            parent_path_cost=parent_cost,
            adjusted_path_cost=parent_cost + adjusted,
            capped_contributions_by_view=contributions.copy(),
            views=view_details,
            gate_preparation_seconds_by_view=copy.deepcopy(self._instance_gate_seconds),
            gate_preparation_seconds=sum(self._instance_gate_seconds.values()),
            candidate_scoring_seconds_by_view=score_seconds,
            candidate_scoring_seconds=sum(score_seconds.values()),
            temporal_state=False,
            post_fit_only=True,
            observed_flags_used=False,
            permanent_IDs_assigned=False,
            clears_identity_review=False,
        )
        # Preserve the exact three original returned objects. Only local costs
        # change; inherited ranking/support/ages/review/history code consumes them.
        return poses, adjusted, parts, support


def import_dense_instance_state(target, dense_original, arrays, metadata):
    """Actual Dense decoder first, then only implementation/backend context + one restore.

    The source's legitimate frame/count/lifecycle are validated by its frozen
    decoder. The new backend boundary must equal previous_frame+1; it is not
    hardcoded to a particular scientific history length or restore count.
    """
    if (
        type(target) is not SparseInstanceRescoreContinuousTracker
        or type(dense_original) is not InstanceRescoreContinuousTracker
    ):
        raise ValueError("explicit dense-instance and sparse-instance implementations required")
    if target.initialize_count or dense_original.initialize_count:
        raise ValueError("migration requires fresh uninitialized trackers")
    old_context, new_context = dense_original._context(), target._context()
    if metadata.get("context") != old_context:
        raise ValueError("source metadata does not match its DenseInstanceRescore implementation")
    expected = copy.deepcopy(new_context)
    backend = expected.pop("sparse_distance_backend")
    expected["implementation"] = old_context["implementation"]
    if expected != old_context:
        raise ValueError("migration cannot change calibration, seed, scientific policy or fitter")
    previous = metadata.get("previous_frame")
    if type(previous) is not int or previous + 1 != backend["effective_from_native"]:
        raise ValueError("source previous frame must immediately precede sparse backend boundary")
    if len(arrays) != 39:
        raise ValueError("this transition requires all39 DenseInstanceRescore state arrays")
    before = array_signature(arrays)
    original_metadata = copy.deepcopy(metadata)
    dense_original._decode_state(arrays, metadata)
    adapted = copy.deepcopy(metadata)
    adapted["context"] = new_context
    target.restore_state(arrays, adapted)
    restored_arrays, restored_metadata = target.export_state()
    expected_metadata = copy.deepcopy(adapted)
    expected_metadata["restore_count"] += 1
    if (
        before != array_signature(arrays)
        or before != array_signature(restored_arrays)
        or metadata != original_metadata
        or restored_metadata != expected_metadata
    ):
        raise ValueError("sparse transition changed historical array bytes or undeclared metadata")
    if (
        target.initialize_count != 1
        or dense_original.initialize_count != 0
        or target.restore_count != metadata["restore_count"] + 1
        or target.processed_frames != metadata["processed_frames"]
        or target.previous_frame != previous
    ):
        raise ValueError("sparse transition changed the single-boundary lifecycle")
    return dict(
        schema="sparse_instance_dense_state_import_v1",
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
        previous_frame=target.previous_frame,
        effective_from_native=backend["effective_from_native"],
        backend_applied_retroactively_to_prior_history=False,
        review_cleared_by_migration=False,
        companion_diagnostics_required_each_future_frame=True,
    )

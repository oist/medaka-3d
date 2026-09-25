"""Experimental post-fit instance rescoring of the unchanged FastTail candidate pool.

The original fitter and five loss parts are unchanged. A finite sixth term is
added only to candidate local costs, before inherited temporal beam ranking.
Its scale/cap are initial experimental choices, not a calibrated equivalence
to the original mask loss. This does not optimize instance loss by gradients.

The ordinary 39-array checkpoint continues to carry costs and the original five
parts. Those five parts therefore cannot reconstruct the new local cost. An
actual rollout caller MUST persist ``export_instance_diagnostics()`` for every
processed frame as a companion artifact; this prototype has no rollout caller.
That frame-local payload is logging only, never future tracking/identity state.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from time import perf_counter

import numpy as np
from directional_chest_tracker import array_signature
from fast_tail_recovery_tracker import FastTailRecoveryContinuousTracker
from instance_body_score import DEFAULT_CONFIG, prepare_instance_observation, score_instance_bodies

from medaka3d.types import VIEWS


@dataclass(frozen=True)
class InstanceRescorePolicy:
    """Fixed first experiment; these choices were not tuned on diagnostic frames."""

    effective_from_native: int = field(default=76360, init=False)
    bodies_per_view: int = field(default=3, init=False)
    fixed_camera_denominator: int = field(default=3, init=False)
    cap_per_view: float = field(default=9.0, init=False)
    weight_source: str = field(default="tracker_config.mask_weight", init=False)


POLICY = InstanceRescorePolicy()


class InstanceRescoreContinuousTracker(FastTailRecoveryContinuousTracker):
    """Preserve all proposals/updates; add observation-gated scene cost after fit."""

    def __init__(self, *args, **kwargs):
        self._instance_frame = None
        self._instance_observations = None
        self._instance_gate_seconds = None
        self.last_instance_diagnostics = None
        super().__init__(*args, **kwargs)
        if self.config.mode != "detectron2" or tuple(self.rig.views) != tuple(VIEWS):
            raise ValueError("instance rescoring requires D2 and the fixed three camera slots")
        if not np.isfinite(self.config.mask_weight) or self.config.mask_weight < 0:
            raise ValueError("instance weight requires finite nonnegative cfg.mask_weight")

    def _context(self):
        context = super()._context()
        context["instance_rescore_policy"] = json.loads(
            json.dumps(
                dict(
                    schema="instance_rescore_policy_v1",
                    **asdict(POLICY),
                    observation_config=asdict(DEFAULT_CONFIG),
                    camera_order=list(VIEWS),
                    weight=float(self.config.mask_weight),
                    normalization="weight * sum_views(clip(best_scene_cost / 3, 0, 9)) / 3",
                    unavailable_view_contribution=0.0,
                    nonfinite_projection_contribution=POLICY.cap_per_view,
                    initial_experimental_choices=True,
                    calibrated_equivalence_to_original_mask_loss=False,
                    post_fit_only=True,
                    gradient_objective_unchanged=True,
                    original_five_loss_parts_unchanged=True,
                    five_parts_reconstruct_new_local_cost=False,
                    companion_diagnostics_required_each_frame=True,
                    existing_proposals_and_temporal_update_code_unchanged=True,
                    permanent_IDs_assigned=False,
                    clears_identity_review=False,
                )
            )
        )
        return context

    def _advance_owned(self, frame):
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
                    self._instance_observations[view] = prepare_instance_observation(
                        frame.masks.get(view), frame.detections.get(view), DEFAULT_CONFIG
                    )
                    self._instance_gate_seconds[view] = perf_counter() - started
            return super()._advance_owned(frame)
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
        poses, base_cost, parts, support = super()._fit(proposals, states, rays, targets)
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
            scored = score_instance_bodies(
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

    def export_instance_diagnostics(self):
        """Caller must save this copied payload per frame, apart from base state."""
        if self.last_instance_diagnostics is None:
            raise RuntimeError("no completed fit diagnostic is available for this frame")
        return copy.deepcopy(self.last_instance_diagnostics)


def import_fast_tail_state(target, original, arrays, metadata):
    """Validate with actual FastTail decoder, add policy, preserve all 39 arrays.

    The boundary is fixed at native 76360. Lifecycle counts come from decoded
    source metadata, not assumptions about a prior experiment. A scientific
    caller must independently bind the real source/seal and require its expected
    26560 processed frames, initialization 1 and restore count 4 (becoming 5).
    No actual scientific checkpoint is read by this prototype.
    """
    if (
        type(target) is not InstanceRescoreContinuousTracker
        or type(original) is not FastTailRecoveryContinuousTracker
    ):
        raise ValueError("explicit FastTail and instance-rescore implementations required")
    if target.initialize_count or original.initialize_count:
        raise ValueError("migration requires fresh uninitialized trackers")
    old_context, new_context = original._context(), target._context()
    if metadata.get("context") != old_context:
        raise ValueError("source metadata does not match its FastTail implementation")
    expected = copy.deepcopy(new_context)
    expected.pop("instance_rescore_policy")
    expected["implementation"] = old_context["implementation"]
    if expected != old_context:
        raise ValueError("migration cannot change calibration, device, seed or existing policies")
    previous = metadata.get("previous_frame")
    if type(previous) is not int or previous + 1 != POLICY.effective_from_native:
        raise ValueError("source previous frame must immediately precede policy effective frame")
    if len(arrays) != 39:
        raise ValueError("this migration requires all 39 FastTail state arrays")
    before = array_signature(arrays)
    original_metadata = copy.deepcopy(metadata)
    # This is the frozen source decoder, not merely target decoding with new context.
    original._decode_state(arrays, metadata)
    adapted = copy.deepcopy(metadata)
    adapted["context"] = new_context
    target.restore_state(arrays, adapted)
    restored, restored_metadata = target.export_state()
    expected_metadata = copy.deepcopy(adapted)
    expected_metadata["restore_count"] += 1
    if (
        before != array_signature(arrays)
        or before != array_signature(restored)
        or metadata != original_metadata
        or restored_metadata != expected_metadata
    ):
        raise ValueError("instance migration changed source array bytes or undeclared metadata")
    if (
        target.initialize_count != 1
        or original.initialize_count != 0
        or target.restore_count != metadata["restore_count"] + 1
        or target.processed_frames != metadata["processed_frames"]
        or target.previous_frame != previous
    ):
        raise ValueError("instance migration changed the single-boundary lifecycle")
    return dict(
        schema="instance_rescore_fast_tail_state_import_v1",
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
        effective_from_native=POLICY.effective_from_native,
        processed_frames=target.processed_frames,
        previous_frame=target.previous_frame,
        original_implementation_not_claimed_for_new_tracking=True,
        review_cleared_by_migration=False,
        companion_diagnostics_required_each_future_frame=True,
    )

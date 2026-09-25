"""Append legacy and nodewise-view bodies to the frozen Sparse instance tracker.

The existing fitted objective, support tests and temporal beam are inherited.
Only parent zero supplies the shape/motion template for additional proposals.
Historical states enter through the real Sparse decoder and one context-only
restore; no old proposal is replaced and the existing decoder cap is retained.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from body_row_proposals import BodyRowConfig, joint_body_row_scenes
from directional_chest_tracker import array_signature
from nodewise_body_row_proposals import nodewise_body_row_pool as body_row_pool
from sparse_instance_rescore_tracker import SparseInstanceRescoreContinuousTracker
from tracker import Proposal

SPARSE_TRACKER_SHA = "d041aff3a0d8cbd71341dbe67dd9b5a7a1e2e7aeb165f55e4b08f8fabdb8fb79"
BODY_ROW_SOURCE_SHA = "c4dbca4c44c6b2cee2f68a2175c809987e489b5e131a1fb792f9bbd36de918fb"
NODEWISE_BODY_ROW_SOURCE_SHA = "ce8c3f7716ef36d87f1dc32e10faa4686575d5ed78b2f102064a78a5d9577f06"
LEGACY_BODY_ROW_TRACKER_SHA = "a9e1d59cc93e81590a06175e763939980438aa1e0fc5e879dff00bb0ca921b1b"
DECODER_EXTRA_CAPACITY = 6


@dataclass(frozen=True)
class BodyRowPolicy:
    effective_from_native: int

    def __post_init__(self):
        if type(self.effective_from_native) is not int or self.effective_from_native < 0:
            raise ValueError("nonnegative integer body-row transition frame required")


class BodyRowContinuousTracker(SparseInstanceRescoreContinuousTracker):
    def __init__(self, *args, body_row_effective_from_native, body_row_config=None, **kwargs):
        # The inherited constructor captures the dynamic _context before begin.
        self.body_row_policy = BodyRowPolicy(body_row_effective_from_native)
        self.body_row_config = BodyRowConfig() if body_row_config is None else body_row_config
        if type(self.body_row_config) is not BodyRowConfig:
            raise ValueError("explicit immutable BodyRowConfig required")
        self.last_body_row_diagnostics = None
        self._body_row_frame = None
        super().__init__(*args, **kwargs)

    def _context(self):
        context = super()._context()
        context["body_row_policy"] = dict(
            schema="body_row_proposal_policy_v1",
            **asdict(self.body_row_policy),
            config=asdict(self.body_row_config),
            sparse_tracker_source_sha256=SPARSE_TRACKER_SHA,
            body_row_source_sha256=BODY_ROW_SOURCE_SHA,
            nodewise_body_row_source_sha256=NODEWISE_BODY_ROW_SOURCE_SHA,
            legacy_body_row_tracker_source_sha256=LEGACY_BODY_ROW_TRACKER_SHA,
            pool_variant="legacy prefix plus nodewise-view complete bodies",
            legacy_pool_prefix_preserved=True,
            supplemental_nodes_all_triangulated_without_template=True,
            supplemental_all_available_views_preferred_when_valid=True,
            supplemental_pair_fallback_only_after_three_view_failure=True,
            unused_node_view_source=-1,
            excluded_views_are_not_verified=True,
            parent_scope="existing parent0 predict.predicted only",
            total_candidate_cap=int(self.config.max_hypotheses) + DECODER_EXTRA_CAPACITY,
            total_candidate_cap_rule="max_hypotheses+6; unchanged decoder limit",
            additional_candidate_limit="min(body_row_config.scene_limit, remaining cap)",
            preserve_existing_proposal_order_and_objects=True,
            same_row_proposed_sources_used_without_reassignment=True,
            partial_completion="legacy partial body only: one node from rotated parent template",
            bounds_and_calibration_unchanged=True,
            fitter_costs_support_and_temporal_update_unchanged=True,
            old_history_rewritten=False,
            retroactive_proposals=False,
            exact_global_search=False,
            physical_IDs_assigned=False,
            clears_identity_review=False,
        )
        return context

    def advance(self, frame):
        # Also clear on native/observation validation failures before _advance_owned.
        self.last_body_row_diagnostics = None
        self.last_instance_diagnostics = None
        try:
            return super().advance(frame)
        except BaseException:
            self.last_body_row_diagnostics = None
            self.last_instance_diagnostics = None
            raise

    def _advance_owned(self, frame):
        self.last_body_row_diagnostics = None
        self.last_instance_diagnostics = None
        if self._body_row_frame is not None:
            raise RuntimeError("body-row owned frame scope already active")
        native = int(frame.frame_idx)
        if native < self.body_row_policy.effective_from_native:
            raise ValueError("body-row execution precedes its declared transition boundary")
        self._body_row_frame = native
        try:
            result = super()._advance_owned(frame)
            if self.last_body_row_diagnostics is None:
                raise RuntimeError("completed frame has no body-row proposal diagnostic")
            self.last_body_row_diagnostics["frame_completed"] = True
            return result
        except BaseException:
            self.last_body_row_diagnostics = None
            self.last_instance_diagnostics = None
            raise
        finally:
            self._body_row_frame = None

    def _proposals(self, states, rays, flow=None):
        self.last_body_row_diagnostics = None
        if self._body_row_frame is None:
            raise RuntimeError("body-row proposals require a current owned frame")
        inherited_started = perf_counter()
        original = super()._proposals(states, rays, flow)
        inherited_seconds = perf_counter() - inherited_started
        incoming = len(original)
        capacity = int(self.config.max_hypotheses) + DECODER_EXTRA_CAPACITY
        if not 0 < incoming <= capacity:
            raise ValueError("inherited proposals exceed the unchanged decoder cap or are empty")
        main = original[0]
        if main.parent != 0 or main.kind != "predict" or main.permutation != (0, 1, 2):
            raise ValueError("first inherited proposal must be parent0 predict with P012")
        predicted = main.predicted
        if np.shape(predicted) != (3, 3, 3) or not np.isfinite(predicted).all():
            raise ValueError("finite parent0 prediction required")
        started = perf_counter()
        pool, pool_stats = body_row_pool(
            self.rig, rays, self.lower, self.upper, self.body_row_config
        )
        pool_seconds = perf_counter() - started
        scene_started = perf_counter()
        scenes, scene_stats = joint_body_row_scenes(
            pool, predicted, self.lengths, self.lower, self.upper, self.body_row_config
        )
        scene_seconds = perf_counter() - scene_started
        if len(scenes) > self.body_row_config.scene_limit:
            raise ValueError("whole-row helper exceeded its declared scene limit")
        headroom = capacity - incoming
        admitted = min(len(scenes), self.body_row_config.scene_limit, headroom)
        extra, filled = [], []
        for scene in scenes[:admitted]:
            pose = np.asarray(scene["pose_xyz"])
            plan = np.asarray(scene["proposed_sources"])
            measured = np.asarray(scene["triangulated_nodes"])
            rows = np.asarray(scene["source_rows"])
            if (
                pose.shape != (3, 3, 3)
                or not np.isfinite(pose).all()
                or np.any(pose < self.lower)
                or np.any(pose > self.upper)
                or plan.shape != (3, 3, 3)
                or plan.dtype.kind not in "iu"
                or np.any(plan < -1)
                or measured.shape != (3, 3)
                or measured.dtype != np.bool_
                or np.any(measured.sum(-1) < 2)
                or rows.shape != (3, 3)
                or rows.dtype.kind not in "iu"
                or np.any(rows < -1)
            ):
                raise ValueError("whole-row scene pose/source/completion schema differs")
            if not np.all((plan < 0) | (plan == rows[..., None])):
                raise ValueError("whole-row source plan must keep one row in each view")
            for view in range(3):
                used = rows[:, view]
                used = used[used >= 0]
                if len(set(used.tolist())) != len(used):
                    raise ValueError("whole-row scene reuses a view row across bodies")
            extra.append(Proposal(pose.copy(), predicted, plan.copy(), 0, "body_row", (0, 1, 2)))
            filled.append(~measured)
        outgoing = original + extra
        self.last_body_row_diagnostics = dict(
            schema="body_row_frame_diagnostics_v1",
            frame_idx=self._body_row_frame,
            frame_completed=False,
            policy=copy.deepcopy(self._context()["body_row_policy"]),
            incoming_proposal_count=incoming,
            outgoing_proposal_count=len(outgoing),
            decoder_candidate_cap=capacity,
            headroom=headroom,
            zero_headroom=headroom == 0,
            generated_scene_count=len(scenes),
            added_proposal_count=admitted,
            excluded_by_capacity=len(scenes) - admitted,
            excluded_by_scene_limit=max(0, int(scene_stats["admissible_scenes"]) - len(scenes)),
            pool=copy.deepcopy(pool),
            pool_stats=copy.deepcopy(pool_stats),
            scenes=copy.deepcopy(scenes),
            scene_stats=copy.deepcopy(scene_stats),
            parent_index=0,
            parent_prediction=np.asarray(predicted).copy(),
            new_proposal_indices=np.arange(incoming, len(outgoing), dtype=np.int64),
            new_proposal_plans=(
                np.stack([p.plan for p in extra]) if extra else np.empty((0, 3, 3, 3), np.int32)
            ),
            new_proposal_poses=(
                np.stack([p.pose for p in extra]) if extra else np.empty((0, 3, 3, 3), np.float64)
            ),
            filled_nodes_by_proposal=(np.stack(filled) if filled else np.empty((0, 3, 3), bool)),
            new_proposal_kind="body_row",
            new_proposal_permutation=(0, 1, 2),
            inherited_proposal_seconds=inherited_seconds,
            pool_generation_seconds=pool_seconds,
            scene_generation_seconds=scene_seconds,
            generation_seconds=perf_counter() - started,
            support_recomputed=False,
            permanent_IDs_assigned=False,
            preserves_all_original_proposals=True,
            old_history_rewritten=False,
        )
        return outgoing

    def export_body_row_diagnostics(self):
        value = self.last_body_row_diagnostics
        if (
            value is None
            or not value["frame_completed"]
            or self._body_row_frame is not None
            or value["frame_idx"] != self.previous_frame
        ):
            raise RuntimeError("no completed body-row frame diagnostic is available")
        return copy.deepcopy(value)


def import_sparse_state(target, original, arrays, metadata):
    """Validate the real Sparse checkpoint, then migrate context and restore once."""
    if (
        type(target) is not BodyRowContinuousTracker
        or type(original) is not SparseInstanceRescoreContinuousTracker
    ):
        raise ValueError("explicit original Sparse and target BodyRow implementations required")
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
    policy = expected.pop("body_row_policy")
    expected["implementation"] = old_context["implementation"]
    if expected != old_context:
        raise ValueError(
            "migration may add body-row policy only; seed/config/rig/bounds must match"
        )
    previous = metadata.get("previous_frame")
    if type(previous) is not int or previous + 1 != policy["effective_from_native"]:
        raise ValueError("body-row boundary must immediately follow source previous frame")
    if len(arrays) != 39:
        raise ValueError("migration requires all39 source state arrays")
    before, original_metadata = array_signature(arrays), copy.deepcopy(metadata)
    original._decode_state(arrays, metadata)
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
        or target.initialize_count != 1
        or original.initialize_count != 0
        or target.restore_count != metadata["restore_count"] + 1
        or target.processed_frames != metadata["processed_frames"]
        or target.previous_frame != previous
    ):
        raise ValueError("body-row transition changed historical bytes or undeclared metadata")
    return dict(
        schema="body_row_sparse_state_import_v1",
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

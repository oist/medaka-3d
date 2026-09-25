"""Explicit continuous variant appending free-only tails after the bounded pool.

Prediction, fitting, beam selection, temporal updates and sticky review remain
inherited. The tail policy is separate checkpoint context, not a legacy alias.
"""

from __future__ import annotations

import copy

import numpy as np
import tail_recovery
from bounded_mouth_tracker import (
    BoundedMouthContinuousTracker,
    import_original_bounded_state,
)
from continuous_tracker import AcceleratedContinuousPersistentNodeTracker
from directional_chest_tracker import array_signature
from tracker import Proposal


def _proposal_signature(proposals):
    return [
        (
            array_signature(dict(pose=p.pose, predicted=p.predicted, plan=p.plan)),
            p.parent,
            p.kind,
            p.permutation,
        )
        for p in proposals
    ]


class TailRecoveryContinuousTracker(BoundedMouthContinuousTracker):
    def __init__(self, *args, tail_recovery_enabled=True, **kwargs):
        if type(tail_recovery_enabled) is not bool:
            raise ValueError("tail_recovery_enabled must be boolean")
        self.tail_recovery_enabled = tail_recovery_enabled
        self.last_tail_recovery_diagnostic = None
        super().__init__(*args, **kwargs)

    def _context(self):
        context = super()._context()
        context["tail_recovery_policy"] = dict(
            schema="tail_recovery_continuous_policy_v1",
            enabled=self.tail_recovery_enabled,
            helper_version=tail_recovery.VERSION,
            target_node=2,
            allow_bounded=False,
            parent="state0; original first predict with identity permutation",
            age_gate="persistence enabled; tail missing and direct pectoral configured ages",
            final_gates="unchanged free intersection, box, anatomy, strict support and exclusivity",
            ordering="after every original bounded proposal; helper order, then age, then budget",
            total_cap="max_hypotheses+6; preserve every original proposal",
            clears_identity_review=False,
        )
        return context

    def _proposals(self, states, rays, flow=None):
        original = super()._proposals(states, rays, flow)
        if not original:
            raise ValueError("original pool required")
        main = original[0]
        if main.kind != "predict" or main.parent != 0 or main.permutation != (0, 1, 2):
            raise ValueError("first-parent identity predict required")
        capacity = self.config.max_hypotheses + 6
        if type(capacity) is not int or len(original) > capacity:
            raise ValueError("original pool exceeds capacity")
        cfg = self.persistence_config
        eligible = (
            (states[0].node_missing[:, 2] >= cfg.minimum_missing_frames)
            & (states[0].direct_pectoral_age >= cfg.minimum_pectoral_frames)
            & cfg.enabled
        )
        enabled = bool(self.tail_recovery_enabled and eligible.any())
        diagnostic = dict(
            configured_enabled=self.tail_recovery_enabled,
            enabled=enabled,
            allow_bounded=False,
            age_eligible_slots=np.flatnonzero(eligible).tolist(),
            original_count=len(original),
            helper_called=False,
            helper_returned=0,
            after_age=0,
            admitted=0,
            excluded_by_age=0,
            excluded_by_budget=0,
            capacity=capacity,
            helper_diagnostics=None,
            target_node=2,
        )
        self.last_tail_recovery_diagnostic = diagnostic
        if not enabled:
            return original
        before = _proposal_signature(original)
        result = tail_recovery.tail_recovery_proposals(
            main.pose,
            main.plan,
            rays,
            self.rig,
            self.lengths,
            self.recovery_config,
            allow_bounded=False,
        )
        if not all(p.node == 2 and p.seed_method == "free_intersection" for p in result.proposals):
            raise ValueError("free-only native tail proposals required")
        allowed = [p for p in result.proposals if eligible[p.fish]]
        admitted = allowed[: capacity - len(original)]
        extra = [
            Proposal(
                p.pose_xyz.copy(),
                main.predicted.copy(),
                p.plan.copy(),
                main.parent,
                "tail_recovery",
                main.permutation,
            )
            for p in admitted
        ]
        if before != _proposal_signature(original):
            raise ValueError("tail helper mutated original pool")
        diagnostic.update(
            helper_called=True,
            helper_returned=len(result.proposals),
            after_age=len(allowed),
            admitted=len(admitted),
            excluded_by_age=len(result.proposals) - len(allowed),
            excluded_by_budget=len(allowed) - len(admitted),
            helper_diagnostics=result.diagnostics,
        )
        return original + extra


def import_original_tail_state(target, original, arrays, metadata):
    """Validate through the frozen bounded importer, then restore original metadata.

    The intermediate proves the original state. Its incremented restore count is
    not fed to the target: this logical migration adds exactly one restore.
    """
    if (
        type(target) is not TailRecoveryContinuousTracker
        or type(original) is not AcceleratedContinuousPersistentNodeTracker
    ):
        raise ValueError("explicit original and tail implementations required")
    if target.initialize_count or original.initialize_count:
        raise ValueError("migration requires fresh uninitialized trackers")
    before = array_signature(arrays)
    original_metadata = copy.deepcopy(metadata)
    intermediate = BoundedMouthContinuousTracker(
        target.rig,
        target.initial,
        target.config,
        recovery_config=target.recovery_config,
        persistence_config=target.persistence_config,
        bounded_mouth_enabled=target.bounded_mouth_enabled,
    )
    new_context, bounded_context = target._context(), intermediate._context()
    expected = copy.deepcopy(new_context)
    expected.pop("tail_recovery_policy")
    expected["implementation"] = bounded_context["implementation"]
    if expected != bounded_context:
        raise ValueError("migration may add only tail policy and implementation context")
    validation = import_original_bounded_state(intermediate, original, arrays, metadata)
    validated_arrays, validated_metadata = intermediate.export_state()
    if before != array_signature(validated_arrays):
        raise ValueError("validated bounded migration changed array bytes")
    adapted = copy.deepcopy(original_metadata)
    adapted["context"] = new_context
    target.restore_state(arrays, adapted)
    restored, restored_metadata = target.export_state()
    if (
        before != array_signature(arrays)
        or before != array_signature(restored)
        or metadata != original_metadata
    ):
        raise ValueError("tail migration changed original arrays or metadata")
    if (
        target.initialize_count != 1
        or original.initialize_count != 0
        or target.restore_count != original_metadata["restore_count"] + 1
        or target.restore_count != validated_metadata["restore_count"]
        or target.processed_frames != original_metadata["processed_frames"]
        or target.previous_frame != original_metadata["previous_frame"]
    ):
        raise ValueError("tail migration changed the single-boundary lifecycle")
    return dict(
        schema="tail_recovery_original_state_import_v1",
        status="ok",
        original_metadata=original_metadata,
        imported_metadata=restored_metadata,
        old_context=original_metadata["context"],
        bounded_context=bounded_context,
        new_context=new_context,
        bounded_validation=validation,
        arrays=before,
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        logical_initialize_count=1,
        new_begin_calls=0,
        added_restore_count=1,
        original_implementation_not_claimed_for_new_tracking=True,
        review_cleared_by_migration=False,
    )

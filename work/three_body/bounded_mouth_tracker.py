"""Explicit continuous variant appending outside-box mouth seeds after the old pool.

Temporal prediction, fit, beam selection, strict support and sticky review remain
inherited. The extra policy is part of checkpoint context, never a legacy alias.
"""

from __future__ import annotations

import copy

import bounded_mouth_recovery
import numpy as np
from continuous_tracker import AcceleratedContinuousPersistentNodeTracker
from directional_chest_tracker import (
    DirectionalChestContinuousTracker,
    array_signature,
)
from directional_chest_tracker import (
    import_original_state as import_directional_state,
)
from tracker import Proposal


class BoundedMouthContinuousTracker(DirectionalChestContinuousTracker):
    def __init__(self, *args, bounded_mouth_enabled=True, **kwargs):
        if type(bounded_mouth_enabled) is not bool:
            raise ValueError("bounded_mouth_enabled must be boolean")
        self.bounded_mouth_enabled = bounded_mouth_enabled
        self.last_bounded_mouth_diagnostic = None
        super().__init__(*args, **kwargs)

    def _context(self):
        context = super()._context()
        context["bounded_mouth_policy"] = dict(
            schema="bounded_mouth_continuous_policy_v1",
            enabled=self.bounded_mouth_enabled,
            helper_version=bounded_mouth_recovery.VERSION,
            parent="state0; original first predict with identity permutation",
            age_gate="existing persistence enabled and mouth missing/direct pectoral ages",
            free_intersection_gate="finite, forward, original RMS gate, outside original box",
            final_gates="unchanged box/anatomy/mouth and chest direct support/source exclusivity",
            ordering="after every original directional proposal; original helper rank per fish",
            total_cap="max_hypotheses+6; preserve every original proposal",
            clears_identity_review=False,
        )
        return context

    def _proposals(self, states, rays, flow=None):
        original = super()._proposals(states, rays, flow)
        main = original[0]
        if main.kind != "predict" or main.parent != 0 or main.permutation != (0, 1, 2):
            raise ValueError("original state0 prediction must precede additional mouths")
        eligible = self._age_eligible(states[0])
        enabled = bool(
            self.bounded_mouth_enabled and self.persistence_config.enabled and eligible.any()
        )
        capacity = self.config.max_hypotheses + 6
        if len(original) > capacity:
            raise ValueError("original directional pool exceeds registered total capacity")
        diagnostic = dict(
            enabled=enabled,
            age_eligible_slots=np.flatnonzero(eligible).tolist(),
            original_count=len(original),
            capacity=capacity,
            helper_returned=0,
            after_age=0,
            admitted=0,
            excluded_by_age=0,
            excluded_by_budget=0,
        )
        self.last_bounded_mouth_diagnostic = diagnostic
        if not enabled:
            return original
        result = bounded_mouth_recovery.bounded_mouth_recovery_proposals(
            main.pose, main.plan, rays, self.rig, self.lengths, self.recovery_config
        )
        allowed = [p for p in result.proposals if eligible[p.fish]]
        admitted = allowed[: capacity - len(original)]
        extra = [
            Proposal(
                p.pose_xyz.copy(),
                main.predicted.copy(),
                p.plan.copy(),
                main.parent,
                "bounded_mouth_recovery",
                main.permutation,
            )
            for p in admitted
        ]
        diagnostic.update(
            helper_returned=len(result.proposals),
            after_age=len(allowed),
            admitted=len(admitted),
            excluded_by_age=len(result.proposals) - len(allowed),
            excluded_by_budget=len(allowed) - len(admitted),
            helper_diagnostics=result.diagnostics,
        )
        return original + extra


def import_original_bounded_state(target, original, arrays, metadata):
    """Validate via the frozen Directional import, then migrate context explicitly.

    The intermediate validates the old 39-array state. The final restore uses the
    original metadata (not the intermediate's incremented restore count), so this
    single logical boundary adds exactly one restore, never two.
    """
    if (
        type(target) is not BoundedMouthContinuousTracker
        or type(original) is not AcceleratedContinuousPersistentNodeTracker
    ):
        raise ValueError("explicit original and bounded implementations required")
    if target.initialize_count or original.initialize_count:
        raise ValueError("migration requires fresh uninitialized trackers")
    before = array_signature(arrays)
    original_metadata = copy.deepcopy(metadata)
    intermediate = DirectionalChestContinuousTracker(
        target.rig,
        target.initial,
        target.config,
        recovery_config=target.recovery_config,
        persistence_config=target.persistence_config,
    )
    new_context, directional_context = target._context(), intermediate._context()
    expected = copy.deepcopy(new_context)
    expected.pop("bounded_mouth_policy")
    expected["implementation"] = directional_context["implementation"]
    if expected != directional_context:
        raise ValueError("migration may add only bounded policy and implementation context")
    validation = import_directional_state(intermediate, original, arrays, metadata)
    validated_arrays, validated_metadata = intermediate.export_state()
    if before != array_signature(validated_arrays):
        raise ValueError("validated Directional migration changed array bytes")
    adapted = copy.deepcopy(original_metadata)
    adapted["context"] = new_context
    target.restore_state(arrays, adapted)
    restored, restored_metadata = target.export_state()
    if (
        before != array_signature(arrays)
        or before != array_signature(restored)
        or metadata != original_metadata
    ):
        raise ValueError("bounded migration changed original arrays or metadata")
    if (
        target.initialize_count != 1
        or original.initialize_count != 0
        or target.restore_count != original_metadata["restore_count"] + 1
        or target.restore_count != validated_metadata["restore_count"]
        or target.processed_frames != original_metadata["processed_frames"]
        or target.previous_frame != original_metadata["previous_frame"]
    ):
        raise ValueError("bounded migration changed the single-boundary lifecycle")
    return dict(
        schema="bounded_mouth_original_state_import_v1",
        status="ok",
        original_metadata=original_metadata,
        imported_metadata=restored_metadata,
        old_context=original_metadata["context"],
        directional_context=directional_context,
        new_context=new_context,
        directional_validation=validation,
        arrays=before,
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        logical_initialize_count=1,
        new_begin_calls=0,
        added_restore_count=1,
        original_implementation_not_claimed_for_new_tracking=True,
        review_cleared_by_migration=False,
    )

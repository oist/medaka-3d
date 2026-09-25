"""Explicit execution-only FastTail variant of the frozen continuous tail tracker.

The CUDA fitter passed the sealed fixed301 whole-pool comparison. Proposals,
serial flow, beam selection, review and temporal updates remain inherited.
Continuous replay qualification is a separate requirement.
"""

from __future__ import annotations

import copy

from continuous_tracker import AcceleratedContinuousPersistentNodeTracker
from directional_chest_tracker import array_signature
from directional_device_fitter import DirectionalDeviceFitter
from fast_directional_device_fitter import FastDirectionalDeviceFitter
from tail_recovery_tracker import TailRecoveryContinuousTracker, import_original_tail_state

QUALIFICATION_SHA = "21244298988acd69ac6bc59edc0a833298144af80f5582b6b06de888ac0bb58c"
FAST_FITTER_SHA = "5804003d537fe5ad144927c9629bf12927fc0ce9dc460615a78125cfe16c4abd"
ORIGINAL_FITTER_SHA = "3228e3815966a5695fc8647a9b8310b28dea9368e177fd4941d20b35024c45aa"


class FastTailRecoveryContinuousTracker(TailRecoveryContinuousTracker):
    def __init__(self, *args, fast_fitter_enabled=True, **kwargs):
        if type(fast_fitter_enabled) is not bool:
            raise ValueError("fast_fitter_enabled must be boolean")
        self.fast_fitter_enabled = fast_fitter_enabled
        super().__init__(*args, **kwargs)

    def _context(self):
        context = super()._context()
        context["fast_execution_policy"] = dict(
            schema="fast_tail_execution_policy_v1",
            enabled=self.fast_fitter_enabled,
            cuda_fitter="FastDirectionalDeviceFitter"
            if self.fast_fitter_enabled
            else "DirectionalDeviceFitter",
            cpu_fitter="DirectionalDeviceFitter",
            fast_fitter_sha256=FAST_FITTER_SHA,
            original_fitter_sha256=ORIGINAL_FITTER_SHA,
            qualification_seal_sha256=QUALIFICATION_SHA,
            qualification_scope="fixed301 all77 CUDA arrays byte exact; independent frames",
            execution_only=True,
            objective_and_optimizer_unchanged=True,
            serial_flow_unchanged=True,
            clears_identity_review=False,
        )
        return context

    def _fit(self, proposals, states, rays, targets):
        # This class is not a FastDirectionalDeviceFitter subclass: its CPU
        # zero-argument super branch must never be invoked with this instance.
        if self.fast_fitter_enabled and self.device.type == "cuda":
            return FastDirectionalDeviceFitter._fit(self, proposals, states, rays, targets)
        return DirectionalDeviceFitter._fit(self, proposals, states, rays, targets)


def import_original_fast_tail_state(target, original, arrays, metadata):
    """Validate via frozen Tail migration and add only one logical restore."""
    if (
        type(target) is not FastTailRecoveryContinuousTracker
        or type(original) is not AcceleratedContinuousPersistentNodeTracker
    ):
        raise ValueError("explicit original and fast tail implementations required")
    if target.initialize_count or original.initialize_count:
        raise ValueError("migration requires fresh uninitialized trackers")
    before = array_signature(arrays)
    original_metadata = copy.deepcopy(metadata)
    intermediate = TailRecoveryContinuousTracker(
        target.rig,
        target.initial,
        target.config,
        recovery_config=target.recovery_config,
        persistence_config=target.persistence_config,
        bounded_mouth_enabled=target.bounded_mouth_enabled,
        tail_recovery_enabled=target.tail_recovery_enabled,
    )
    new_context, tail_context = target._context(), intermediate._context()
    expected = copy.deepcopy(new_context)
    expected.pop("fast_execution_policy")
    expected["implementation"] = tail_context["implementation"]
    if expected != tail_context:
        raise ValueError("migration may add only fast execution policy and implementation context")
    validation = import_original_tail_state(intermediate, original, arrays, metadata)
    validated_arrays, validated_metadata = intermediate.export_state()
    if before != array_signature(validated_arrays):
        raise ValueError("validated tail migration changed array bytes")
    adapted = copy.deepcopy(original_metadata)
    adapted["context"] = new_context
    target.restore_state(arrays, adapted)
    restored, restored_metadata = target.export_state()
    if (
        before != array_signature(arrays)
        or before != array_signature(restored)
        or metadata != original_metadata
    ):
        raise ValueError("fast tail migration changed original arrays or metadata")
    expected_metadata = copy.deepcopy(adapted)
    expected_metadata["restore_count"] += 1
    if (
        target.initialize_count != 1
        or original.initialize_count != 0
        or target.restore_count != original_metadata["restore_count"] + 1
        or target.restore_count != validated_metadata["restore_count"]
        or target.processed_frames != original_metadata["processed_frames"]
        or target.previous_frame != original_metadata["previous_frame"]
        or restored_metadata != expected_metadata
    ):
        raise ValueError("fast tail migration changed the single-boundary lifecycle or metadata")
    return dict(
        schema="fast_tail_original_state_import_v1",
        status="ok",
        original_metadata=original_metadata,
        imported_metadata=restored_metadata,
        old_context=original_metadata["context"],
        tail_context=tail_context,
        new_context=new_context,
        tail_validation=validation,
        arrays=before,
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        logical_initialize_count=1,
        new_begin_calls=0,
        added_restore_count=1,
        original_implementation_not_claimed_for_new_tracking=True,
        review_cleared_by_migration=False,
    )

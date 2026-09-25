"""One explicit original66399 -> unmasked Nodewise66400 state migration.

The scientific caller must bind the successful original checkpoint and require
16600 historical frames starting at49800. This pure adapter also admits a short
synthetic history ending66399, validated by the actual frozen source decoder.
It never reads a checkpoint, begins tracking, advances a frame, or changes the
frozen instance sixth-term schedule (effective76360).
"""

from __future__ import annotations

import copy

from continuous_tracker import AcceleratedContinuousPersistentNodeTracker
from directional_chest_tracker import array_signature
from fast_tail_recovery_tracker import (
    FastTailRecoveryContinuousTracker,
    import_original_fast_tail_state,
)
from instance_rescore_tracker import POLICY
from nodewise_body_row_tracker import BodyRowContinuousTracker

EFFECTIVE_FROM_NATIVE = 66400
SOURCE_RESTORE_COUNT = 3
SOURCE_PINS = {
    "fast_tail_recovery_tracker.py": (
        "fbaa1a81f337e291f487886ace1fbceb7ac9adf95db86b82896904d42cafa3ad"
    ),
    "nodewise_body_row_tracker.py": (
        "b574ce26481c2c1457ceb1428659aa7bce25783e36957fd2eb9b6399f52b8d16"
    ),
    "sparse_instance_rescore_tracker.py": (
        "d041aff3a0d8cbd71341dbe67dd9b5a7a1e2e7aeb165f55e4b08f8fabdb8fb79"
    ),
    "instance_rescore_tracker.py": (
        "f625513ed379e17241ad120c8a454c56b398764a1341ab7756790062231863e5"
    ),
}


def import_original_state(target, original, arrays, metadata):
    """Validate the original through frozen FastTail, then restore target once.

    The temporary FastTail restore is validation only. Its metadata is never
    used as target input; target receives the original restore3 metadata with
    the explicitly checked new context and exports restore4.
    """
    if (
        type(target) is not BodyRowContinuousTracker
        or type(original) is not AcceleratedContinuousPersistentNodeTracker
    ):
        raise ValueError(
            "exact original Accelerated and unmasked Nodewise implementations required"
        )
    for tracker in (target, original):
        if (
            tracker.initialize_count
            or tracker.restore_count
            or tracker._states is not None
            or tracker._history
            or tracker.previous_frame is not None
            or tracker._previous_observation is not None
            or tracker._finished
            or tracker._failed
        ):
            raise ValueError("migration requires fresh uninitialized objects")
    old_context, new_context = original._context(), target._context()
    if (
        metadata.get("context") != old_context
        or original._initial_context != old_context
        or target._initial_context != new_context
    ):
        raise ValueError("source or constructor-finalized context differs")
    if (
        type(metadata.get("previous_frame")) is not int
        or metadata["previous_frame"] != EFFECTIVE_FROM_NATIVE - 1
        or type(metadata.get("restore_count")) is not int
        or metadata["restore_count"] != SOURCE_RESTORE_COUNT
        or metadata.get("initialize_count") != 1
    ):
        raise ValueError("original66399 initialization1/restore3 required")
    if len(arrays) != 39:
        raise ValueError("all39 original state arrays required")
    expected_fast = copy.deepcopy(new_context)
    instance_policy = expected_fast.pop("instance_rescore_policy")
    backend = expected_fast.pop("sparse_distance_backend")
    body_policy = expected_fast.pop("body_row_policy")
    if (
        backend["effective_from_native"] != EFFECTIVE_FROM_NATIVE
        or body_policy["effective_from_native"] != EFFECTIVE_FROM_NATIVE
        or instance_policy["effective_from_native"] != POLICY.effective_from_native
        or POLICY.effective_from_native != 76360
    ):
        raise ValueError("body/sparse66400 and original sixth-term76360 boundaries required")

    # Uninitialized construction only. Frozen FastTail performs the original
    # decoder and its own strict original -> chest -> bounded -> tail checks.
    validator = FastTailRecoveryContinuousTracker(
        target.rig,
        target.initial,
        target.config,
        recovery_config=target.recovery_config,
        persistence_config=target.persistence_config,
        bounded_mouth_enabled=target.bounded_mouth_enabled,
        tail_recovery_enabled=target.tail_recovery_enabled,
        fast_fitter_enabled=target.fast_fitter_enabled,
    )
    fast_context = validator._context()
    expected_fast["implementation"] = fast_context["implementation"]
    if expected_fast != fast_context:
        raise ValueError(
            "only instance, sparse backend, body policy and implementation may be added"
        )
    before, original_metadata = array_signature(arrays), copy.deepcopy(metadata)
    validation = import_original_fast_tail_state(validator, original, arrays, metadata)
    if (
        validation["status"] != "ok"
        or validation["arrays"] != before
        or not validation["all_arrays_byte_preserved"]
        or not validation["original_metadata_unchanged"]
        or validation["imported_metadata"]["restore_count"] != SOURCE_RESTORE_COUNT + 1
        or validation["original_metadata"] != original_metadata
        or validation["new_context"] != fast_context
    ):
        raise ValueError("frozen original-to-Fast validation differs")
    fast_validation = dict(
        schema=validation["schema"],
        status="ok",
        source_sha256=SOURCE_PINS["fast_tail_recovery_tracker.py"],
        context=fast_context,
        all_arrays_byte_preserved=True,
        temporary_restore_count=SOURCE_RESTORE_COUNT + 1,
        validation_only=True,
        temporary_metadata_used_as_target_input=False,
    )
    # Drop the large temporary decoded history before decoding the target.
    del validation, validator
    adapted = copy.deepcopy(original_metadata)
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
        or target.initialize_count != 1
        or original.initialize_count != 0
        or target.restore_count != SOURCE_RESTORE_COUNT + 1
        or target.processed_frames != metadata["processed_frames"]
        or target.previous_frame != EFFECTIVE_FROM_NATIVE - 1
    ):
        raise ValueError("migration changed historical bytes or undeclared metadata")
    del restored
    return dict(
        schema="nodewise_original_state_import_v1",
        status="ok",
        original_metadata=original_metadata,
        imported_metadata=restored_metadata,
        old_context=old_context,
        new_context=new_context,
        arrays=before,
        fast_validation=fast_validation,
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        logical_initialize_count=1,
        added_restore_count=1,
        new_begin_calls=0,
        previous_frame=target.previous_frame,
        processed_frames=target.processed_frames,
        effective_from_native=EFFECTIVE_FROM_NATIVE,
        instance_policy_effective_from_native=POLICY.effective_from_native,
        old_history_rewritten=False,
        review_cleared_by_migration=False,
        mask_split_priority_applied=False,
        companion_diagnostics_required_each_future_frame=True,
        scientific_caller_must_bind_original49800_to66400=True,
    )

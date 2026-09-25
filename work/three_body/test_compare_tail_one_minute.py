"""Synthetic completion and migration gates; no real result, fit or advance."""

import copy
from types import SimpleNamespace

import compare_tail_one_minute as m
import numpy as np
import pytest
import test_tail_recovery_tracker as state_fixture


def complete():
    return dict(
        schema="tail_one_minute_v1",
        status="ok",
        one_minute_completed=True,
        native_interval=[66400, 76360],
        native_frames=9960,
        logical_initialization_count=1,
        restore_count=4,
        external_pose_resets=0,
        checkpoint_count=6,
        diagnostic_frames=572,
        periodic_checkpoints_are_not_reinitializations=True,
    )


def prior_summary():
    names = ("original", "directional", "bounded")
    return dict(
        schema="bounded_one_minute_comparison_v1",
        status="ok",
        native_interval=[66400, 76360],
        native_frames=9960,
        horizon_stop=76360,
        same_parent_and_horizon=True,
        one_minute_completed=True,
        candidate_paths={name: "/synthetic/" + name + "/candidate.npz" for name in names},
        candidate_sha256={name: letter * 64 for name, letter in zip(names, "abc", strict=True)},
        all39_arrays_semantically_decoded=True,
        all21_arrays_exact_backtrace=True,
        original66400_parent_and_migration_verified=True,
        bounded_all_six_checkpoint_prefixes_exact=True,
        logical_initialization_count=1,
        checkpoint_restore_count=4,
        restores_during_one_minute=0,
        external_pose_resets=0,
        methods={name: {} for name in names},
        all_seven_C34_conditions_retained=True,
        requested_female_anchors=11,
        requested_female_intervals=10,
    )


@pytest.fixture(scope="module")
def migrated():
    # This fixture constructs the 39 arrays directly and invokes only the frozen
    # decoder and migration. The legacy fixture that fits/advances is never used.
    inputs = state_fixture.inputs.__wrapped__()
    arrays, parent = state_fixture.checkpoint.__wrapped__(inputs)
    parent["restore_count"] = 3
    tracker = state_fixture.m
    original = tracker.AcceleratedContinuousPersistentNodeTracker(*inputs)
    target = tracker.TailRecoveryContinuousTracker(*inputs)
    migration = tracker.import_original_tail_state(target, original, arrays, parent)
    return migration, arrays, parent, target.export_state()[1]


def test_complete_tail_and_three_method_prior_headers_are_accepted_without_mutation():
    tail, prior = complete(), prior_summary()
    before = copy.deepcopy((tail, prior))
    assert m.validate_completion(tail) is None
    assert m.validate_prior_summary(prior) is None
    assert (tail, prior) == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema", "bounded_mouth_one_minute_v1"),
        ("status", "incomplete"),
        ("one_minute_completed", False),
        ("native_frames", 1660),
        ("native_interval", [66400, 68060]),
        ("restore_count", 5),
        ("logical_initialization_count", 2),
        ("external_pose_resets", 1),
        ("checkpoint_count", 5),
        ("diagnostic_frames", 321),
        ("periodic_checkpoints_are_not_reinitializations", False),
    ],
)
def test_tail_header_rejects_old_scope_and_incomplete_lifecycle(key, value):
    summary = complete()
    summary[key] = value
    with pytest.raises(ValueError):
        m.validate_completion(summary)


@pytest.mark.parametrize(
    "session,exit_code",
    [(0, 0), (-1, 0), (True, 0), (None, 0), ("123", 0), (1.5, 0), (123, 1), (123, -1)],
)
def test_owned_success_gate_precedes_path_or_payload_access(session, exit_code):
    # Path attributes are intentionally absent: reaching one would fail this test.
    args = SimpleNamespace(owned_session=session, owned_exit_code=exit_code)
    with pytest.raises(ValueError, match="CLOSED0"):
        m.main(args)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema", "directional_minute_comparison_v1"),
        ("status", "incomplete"),
        ("native_interval", [66400, 68060]),
        ("native_frames", 1660),
        ("horizon_stop", 68060),
        ("same_parent_and_horizon", False),
        ("one_minute_completed", False),
        ("all39_arrays_semantically_decoded", False),
        ("all21_arrays_exact_backtrace", False),
        ("original66400_parent_and_migration_verified", False),
        ("bounded_all_six_checkpoint_prefixes_exact", False),
        ("logical_initialization_count", 2),
        ("checkpoint_restore_count", 5),
        ("restores_during_one_minute", 1),
        ("external_pose_resets", 1),
        ("all_seven_C34_conditions_retained", False),
        ("requested_female_anchors", 10),
        ("requested_female_intervals", 9),
    ],
)
def test_three_method_reuse_rejects_scope_evidence_or_lifecycle_drift(key, value):
    summary = prior_summary()
    summary[key] = value
    with pytest.raises(ValueError):
        m.validate_prior_summary(summary)


@pytest.mark.parametrize("key", ["methods", "candidate_paths", "candidate_sha256"])
@pytest.mark.parametrize("change", ["missing", "extra", "renamed"])
def test_three_method_reuse_requires_exact_corresponding_method_sets(key, change):
    summary = prior_summary()
    if change == "missing":
        summary[key].pop("bounded")
    elif change == "extra":
        summary[key]["tail"] = summary[key]["bounded"]
    else:
        summary[key]["tail"] = summary[key].pop("bounded")
    with pytest.raises(ValueError):
        m.validate_prior_summary(summary)


def test_actual_synthetic_39_array_migration_validates_with_nested_single_restores(migrated):
    migration, arrays, parent, metadata = copy.deepcopy(migrated)
    before = copy.deepcopy((migration, parent, metadata))
    signatures = state_fixture.m.array_signature(arrays)
    assert len(arrays) == 39
    assert m.verify_tail_import(migration, arrays, parent, metadata) is None
    assert (migration, parent, metadata) == before
    assert state_fixture.m.array_signature(arrays) == signatures
    assert parent["restore_count"] == 3
    assert migration["imported_metadata"]["restore_count"] == 4
    assert migration["bounded_validation"]["imported_metadata"]["restore_count"] == 4


@pytest.mark.parametrize(
    "case",
    [
        "array_bytes",
        "array_dtype",
        "array_shape",
        "array_missing",
        "tail_signature",
        "bounded_signature",
        "directional_signature",
        "nested_original_context",
        "nested_original_metadata",
        "tail_double_restore",
        "bounded_double_restore",
        "directional_double_restore",
        "old_context",
        "new_context",
        "metadata_field",
        "undeclared_context",
        "bounded_context",
        "schema",
        "nested_schema",
    ],
)
def test_tail_migration_rejects_nested_origin_hash_context_and_restore_drift(migrated, case):
    migration, arrays, parent, metadata = copy.deepcopy(migrated)
    bounded = migration["bounded_validation"]
    directional = bounded["directional_validation"]
    if case == "array_bytes":
        arrays["previous_images_bgr_top_cam"].flat[0] += 1
    elif case == "array_dtype":
        arrays["state_pose"] = arrays["state_pose"].astype(np.float32)
    elif case == "array_shape":
        arrays["state_pose"] = arrays["state_pose"].reshape(-1)
    elif case == "array_missing":
        arrays.pop("previous_masks_top_cam")
    elif case.endswith("_signature"):
        owner = {
            "tail_signature": migration,
            "bounded_signature": bounded,
            "directional_signature": directional,
        }[case]
        owner["arrays"]["state_pose"]["sha256"] = "0" * 64
    elif case == "nested_original_context":
        directional["old_context"]["tracker_config"]["motion_weight"] += 1
    elif case == "nested_original_metadata":
        bounded["original_metadata"]["previous_frame"] += 1
    elif case.endswith("_double_restore"):
        owner = {
            "tail_double_restore": migration,
            "bounded_double_restore": bounded,
            "directional_double_restore": directional,
        }[case]
        owner["imported_metadata"]["restore_count"] += 1
    elif case == "old_context":
        migration["old_context"]["implementation"] = "WrongOriginal"
    elif case == "new_context":
        metadata["context"] = copy.deepcopy(metadata["context"])
        metadata["context"]["tracker_config"]["motion_weight"] += 1
    elif case == "metadata_field":
        migration["imported_metadata"]["processed_frames"] += 1
    elif case == "undeclared_context":
        migration["new_context"]["extra_policy"] = True
        migration["imported_metadata"]["context"] = copy.deepcopy(migration["new_context"])
        metadata["context"] = copy.deepcopy(migration["new_context"])
    elif case == "bounded_context":
        migration["bounded_context"]["tracker_config"]["motion_weight"] += 1
    elif case == "schema":
        migration["schema"] = "bounded_mouth_original_state_import_v1"
    else:
        bounded["schema"] = "directional_chest_original_state_import_v1"
    with pytest.raises(ValueError):
        m.verify_tail_import(migration, arrays, parent, metadata)


@pytest.mark.parametrize(
    "key,value",
    [
        ("enabled", False),
        ("allow_bounded", True),
        ("target_node", 0),
        ("clears_identity_review", True),
    ],
)
def test_tail_migration_requires_enabled_free_only_native_tail_policy(migrated, key, value):
    migration, arrays, parent, metadata = copy.deepcopy(migrated)
    # Keep all repeated declarations consistent so the policy gate itself must fail.
    for context in (
        migration["new_context"],
        migration["imported_metadata"]["context"],
        metadata["context"],
    ):
        context["tail_recovery_policy"][key] = value
    with pytest.raises(ValueError):
        m.verify_tail_import(migration, arrays, parent, metadata)


@pytest.mark.parametrize(
    "key,value",
    [
        ("all_arrays_byte_preserved", False),
        ("original_metadata_unchanged", False),
        ("logical_initialize_count", 2),
        ("new_begin_calls", 1),
        ("added_restore_count", 2),
        ("original_implementation_not_claimed_for_new_tracking", False),
        ("review_cleared_by_migration", True),
    ],
)
def test_tail_migration_requires_every_preservation_declaration(migrated, key, value):
    migration, arrays, parent, metadata = copy.deepcopy(migrated)
    migration[key] = value
    with pytest.raises(ValueError):
        m.verify_tail_import(migration, arrays, parent, metadata)


def test_expected_completion_seal_cannot_drift_during_output_verification(monkeypatch):
    from pathlib import Path

    import compare_tail_one_minute as c

    root = Path("synthetic_completed").resolve()

    def changed(root, filename, stage, hashes, expected):
        hashes[str(root / filename)] = "b" * 64
        return {"status": "ok"}

    monkeypatch.setattr(c.old, "successful", changed)
    with pytest.raises(ValueError, match="expected seal changed"):
        c.successful(root, "tracking.provenance.json", "completed", {}, "a" * 64)

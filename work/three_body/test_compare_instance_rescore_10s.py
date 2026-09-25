"""Synthetic CPU checks; no recording, checkpoint, model or biological labels."""

import copy
from pathlib import Path

import compare_instance_rescore_10s as caller
import numpy as np
import pytest


@pytest.fixture
def small_scope(monkeypatch):
    for name, value in dict(FIRST=10, START=13, STOP=16).items():
        monkeypatch.setattr(caller, name, value)


def imported_state():
    old = dict(implementation="FastTail", seed="same", calibration="same", old_policy={"x": 1})
    policy = dict(effective_from_native=caller.START, clears_identity_review=False)
    new = {**copy.deepcopy(old), "implementation": "Instance", "instance_rescore_policy": policy}
    before = dict(
        context=old,
        previous_frame=caller.START - 1,
        processed_frames=caller.START - caller.FIRST,
        initialize_count=1,
        restore_count=4,
    )
    after = {**copy.deepcopy(before), "context": new, "restore_count": 5}
    imported = dict(
        schema="instance_rescore_fast_tail_state_import_v1",
        status="ok",
        old_context=old,
        new_context=new,
        original_metadata=before,
        imported_metadata=after,
        arrays={str(k): {} for k in range(39)},
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        logical_initialize_count=1,
        new_begin_calls=0,
        added_restore_count=1,
        effective_from_native=caller.START,
        processed_frames=caller.START - caller.FIRST,
        previous_frame=caller.START - 1,
        review_cleared_by_migration=False,
        companion_diagnostics_required_each_future_frame=True,
    )
    return imported, old, dict(context=new), policy


def test_single_import_preserves_source_and_exact_metadata(small_scope):
    args = imported_state()
    before = copy.deepcopy(args)
    caller.validate_import(*args)
    assert args == before


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(new_begin_calls=1),
        lambda d: d.update(effective_from_native=14),
        lambda d: d.update(review_cleared_by_migration=True),
        lambda d: d.update(all_arrays_byte_preserved=1),
        lambda d: d["arrays"].pop("38"),
        lambda d: d["imported_metadata"].update(restore_count=6),
        lambda d: d["new_context"].update(calibration="different"),
        lambda d: d["new_context"].update(old_policy={"x": 2}),
    ],
)
def test_import_rejects_undeclared_change(small_scope, change):
    imported, old, final, policy = imported_state()
    # Keep expected contexts detached so changes cannot alter both operands.
    old, final, policy = copy.deepcopy((old, final, policy))
    change(imported)
    with pytest.raises(ValueError):
        caller.validate_import(imported, old, final, policy)


@pytest.mark.parametrize(
    "document",
    [
        dict(status="ok", native_frames=2, finish_matches_checkpoint_backtrace=True),
        dict(status="failed", native_frames=3, finish_matches_checkpoint_backtrace=True),
        dict(status="ok", native_frames=3, finish_matches_checkpoint_backtrace=1),
        dict(status="ok", native_frames=3, finish_matches_checkpoint_backtrace=False),
    ],
)
def test_partial_or_false_completion_is_rejected(document):
    with pytest.raises(ValueError):
        caller.exact_fields(
            document,
            dict(status="ok", native_frames=3, finish_matches_checkpoint_backtrace=True),
            "incomplete",
        )


def tiny_candidate():
    result = {"frame_idx": np.arange(10, 16, dtype=np.int64)}
    result.update({f"field{k}": np.arange(6, dtype=np.float64) for k in range(20)})
    result["field1"] = np.array(["a"] * 6, dtype="U1")
    result["field2"][2] = np.nan
    return result


def test_prefix_comparison_keeps_both_terminal_horizons(small_scope):
    old = tiny_candidate()
    new = {k: a.copy() for k, a in old.items()}
    new["field0"][-1] += 8  # A changed suffix does not imply changed selected prefix.
    new["field1"] = new["field1"].astype("U4")
    before = caller.signature(old), caller.signature(new)
    result, arrays = caller.same_horizon_prefix(old, new)
    assert result["compared_interval"] == [10, 13]
    assert result["control_terminal_horizon"] == result["variant_terminal_horizon"] == 16
    assert "snapshot_horizon" not in result
    assert not any(row["changed_rows"] for row in result["fields"].values())
    assert result["fields"]["field1"]["unicode_values_ignoring_capacity"] is True
    assert (caller.signature(old), caller.signature(new)) == before
    assert np.array_equal(arrays["frame_idx"], [10, 11, 12])


def test_prefix_exact_bytes_include_signed_zero(small_scope):
    old = tiny_candidate()
    new = {k: a.copy() for k, a in old.items()}
    new["field0"][0] = -0.0
    result, arrays = caller.same_horizon_prefix(old, new)
    assert result["fields"]["field0"]["changed_rows"] == 1
    assert arrays["field0_row_changed"].tolist() == [True, False, False]
    assert result["fields"]["field2"]["changed_rows"] == 0  # Equal NaN bytes stay equal.


def test_geometry_uses_fixed_interval_and_retains_latent_nodes(small_scope):
    pose = np.zeros((6, 3, 3, 3), dtype=np.float64)
    pose[:, :, :, 0] = np.arange(6)[:, None, None]
    pose[:, 1, :, 1] = 5
    pose[:, 2, :, 1] = 12
    observed = np.ones((6, 3, 3), dtype=bool)
    observed[:, 2] = False
    old = dict(frame_idx=np.arange(10, 16), pose_xyz=pose, observed=observed)
    new = {k: a.copy() for k, a in old.items()}
    new["pose_xyz"][:3] += 1000  # Outside interval; cannot inflate internal velocity.
    new["pose_xyz"][3:, 2, 0, 2] += 4  # Latent point remains in geometric diagnostics.
    before = caller.signature(old), caller.signature(new)
    summary, arrays = caller.geometry_deltas(old, new)
    assert np.array_equal(arrays["frame_idx"], [13, 14, 15])
    assert arrays["pose_delta_mm"][:, 2, 0].tolist() == [4, 4, 4]
    assert np.all(arrays["all_estimated_speed_mm_s"] == 166)
    assert np.all(arrays["all_estimated_acceleration_mm_s2"] == 0)
    assert not arrays["speed_both_direct"][:, :, 2].any()
    assert np.all(arrays["chest_pair_distance_mm"] == [5, 12, 7])
    assert summary["no_anatomical_or_identity_threshold_introduced"] is True
    assert (caller.signature(old), caller.signature(new)) == before


def test_execution_guard_precedes_any_path_or_payload_access(monkeypatch):
    monkeypatch.setattr(caller, "EXECUTION_READY", False)
    with pytest.raises(ValueError, match="source review"):
        caller.run(Path("no-such-branch"), "a" * 64, Path("no-output-must-be-created"))

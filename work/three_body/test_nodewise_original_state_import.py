"""Synthetic original39 migration only: no actual checkpoint, readers or fit."""

# ruff: noqa: E402

import copy
import sys
import weakref
from dataclasses import replace
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"),
    str(Path(__file__).resolve().parents[2] / "src"),
]

import continuous_tracker as continuous
import nodewise_original_state_import as m
import numpy as np
import pytest
import test_tail_recovery_tracker as toy
from directional_device_fitter import DirectionalDeviceFitter
from fast_directional_device_fitter import FastDirectionalDeviceFitter


@pytest.fixture(scope="module")
def inputs():
    return toy.inputs.__wrapped__()


@pytest.fixture(scope="module")
def checkpoint(inputs):
    arrays, metadata = copy.deepcopy(toy.checkpoint.__wrapped__(inputs))
    arrays["history_frame_idx"] += 66399 - metadata["previous_frame"]
    metadata["previous_frame"], metadata["restore_count"] = 66399, 3
    assert metadata["context"]["implementation"] == "AcceleratedContinuousPersistentNodeTracker"
    return arrays, metadata


@pytest.fixture(autouse=True)
def synthetic_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("real arrays, begin/advance or optimizer must not be used")

    monkeypatch.setattr(np, "load", forbidden)
    for key in ("begin", "advance", "_advance_owned"):
        monkeypatch.setattr(continuous.ContinuousPersistentNodeTracker, key, forbidden)
    monkeypatch.setattr(DirectionalDeviceFitter, "_fit", forbidden)
    monkeypatch.setattr(FastDirectionalDeviceFitter, "_fit", forbidden)
    monkeypatch.setattr(continuous.AcceleratedContinuousPersistentNodeTracker, "_fit", forbidden)


def target(inputs, **kwargs):
    options = dict(body_row_effective_from_native=66400, sparse_effective_from_native=66400)
    options.update(kwargs)
    return m.BodyRowContinuousTracker(*inputs, **options)


def test_real39_source_decoder_once_target_restore_once_and_validator_released(
    inputs, checkpoint, monkeypatch
):
    arrays, metadata = copy.deepcopy(checkpoint)
    source = m.AcceleratedContinuousPersistentNodeTracker(*inputs)
    dest = target(inputs)
    before, old_meta = m.array_signature(arrays), copy.deepcopy(metadata)
    source_calls, target_calls, temporary = [], [], []
    real_decode, real_restore = source._decode_state, dest.restore_state
    real_fast = m.FastTailRecoveryContinuousTracker

    def decode(*args):
        source_calls.append(True)
        return real_decode(*args)

    def make_fast(*args, **kwargs):
        result = real_fast(*args, **kwargs)
        temporary.append(weakref.ref(result))
        return result

    def restore(a, md):
        assert source_calls and temporary[0]() is None
        assert md["restore_count"] == 3  # never consume the validator's restore4
        target_calls.append(True)
        return real_restore(a, md)

    monkeypatch.setattr(source, "_decode_state", decode)
    monkeypatch.setattr(m, "FastTailRecoveryContinuousTracker", make_fast)
    monkeypatch.setattr(dest, "restore_state", restore)
    report = m.import_original_state(dest, source, arrays, metadata)
    exported, actual_meta = dest.export_state()
    assert len(source_calls) == len(target_calls) == 1
    assert len(exported) == 39 and m.array_signature(exported) == before == m.array_signature(
        arrays
    )
    assert metadata == old_meta
    assert (
        dest.initialize_count,
        dest.restore_count,
        dest.previous_frame,
        dest.processed_frames,
    ) == (1, 4, 66399, 5)
    assert source.initialize_count == source.restore_count == source.processed_frames == 0
    expected = copy.deepcopy(metadata)
    expected["context"] = dest._context()
    expected["restore_count"] = 4
    assert actual_meta == expected == report["imported_metadata"]
    assert np.all(exported["state_review"][:, 2])
    assert np.all(exported["history_identity_review"][:, 2])
    assert np.any(~exported["history_observed"]) and np.any(exported["history_sources"] == -1)
    assert report["fast_validation"]["validation_only"]
    assert not report["fast_validation"]["temporary_metadata_used_as_target_input"]
    assert report["effective_from_native"] == 66400
    assert report["instance_policy_effective_from_native"] == 76360
    assert report["scientific_caller_must_bind_original49800_to66400"]
    assert not report["mask_split_priority_applied"] and not report["review_cleared_by_migration"]
    # The resulting real metadata is consumable by the frozen target codec.
    resumed = target(inputs)
    resumed.restore_state(exported, actual_meta)
    assert resumed.restore_count == 5
    assert m.array_signature(resumed.export_state()[0]) == before
    with pytest.raises(ValueError, match="context"):
        source.restore_state(exported, actual_meta)


@pytest.mark.parametrize("key", ["body_row_effective_from_native", "sparse_effective_from_native"])
def test_rejects_wrong_new_boundary_before_source_decode(inputs, checkpoint, monkeypatch, key):
    source = m.AcceleratedContinuousPersistentNodeTracker(*inputs)
    monkeypatch.setattr(source, "_decode_state", lambda *a: pytest.fail("boundary must fail first"))
    with pytest.raises(ValueError, match="boundaries"):
        m.import_original_state(target(inputs, **{key: 78020}), source, *copy.deepcopy(checkpoint))


@pytest.mark.parametrize("key,value", [("previous_frame", 66400), ("restore_count", 4)])
def test_rejects_wrong_original_boundary_or_restore_count(inputs, checkpoint, key, value):
    arrays, metadata = copy.deepcopy(checkpoint)
    metadata[key] = value
    dest = target(inputs)
    with pytest.raises(ValueError, match="original66399"):
        m.import_original_state(
            dest, m.AcceleratedContinuousPersistentNodeTracker(*inputs), arrays, metadata
        )
    assert dest.initialize_count == 0


@pytest.mark.parametrize("damage", ["config", "bounds", "source_context"])
def test_rejects_configuration_or_context_change(inputs, checkpoint, damage):
    arrays, metadata = copy.deepcopy(checkpoint)
    source = m.AcceleratedContinuousPersistentNodeTracker(*inputs)
    dest = target(inputs)
    if damage == "config":
        dest = target((*inputs[:2], replace(inputs[2], mask_weight=inputs[2].mask_weight + 1)))
    elif damage == "bounds":
        dest.lower = dest.lower.copy()
        dest.lower[2] -= 1
    else:
        metadata["context"]["implementation"] = "FastTailRecoveryContinuousTracker"
    with pytest.raises(ValueError):
        m.import_original_state(dest, source, arrays, metadata)
    assert dest.initialize_count == 0 and dest._states is None


@pytest.mark.parametrize("damage", ["missing_array", "parent", "review", "age"])
def test_original_decoder_rejects_corrupt_history_before_target_restore(
    inputs, checkpoint, monkeypatch, damage
):
    arrays, metadata = copy.deepcopy(checkpoint)
    if damage == "missing_array":
        arrays.pop("previous_images_bgr_top_cam")
    elif damage == "parent":
        arrays["history_parent"][-1] = 999
    elif damage == "review":
        arrays["state_review"][0, 2] = False
    else:
        arrays["state_node_missing"][0, 0, 2] = 0
    dest = target(inputs)
    monkeypatch.setattr(dest, "restore_state", lambda *a: pytest.fail("bad source reached target"))
    with pytest.raises(ValueError):
        m.import_original_state(
            dest, m.AcceleratedContinuousPersistentNodeTracker(*inputs), arrays, metadata
        )


def test_exact_source_type_and_freshness_required(inputs, checkpoint):
    with pytest.raises(ValueError, match="exact original"):
        m.import_original_state(
            target(inputs), m.FastTailRecoveryContinuousTracker(*inputs), *copy.deepcopy(checkpoint)
        )
    for stale in ("source", "target"):
        dest, source = target(inputs), m.AcceleratedContinuousPersistentNodeTracker(*inputs)
        (source if stale == "source" else dest).restore_count = 1
        with pytest.raises(ValueError, match="fresh"):
            m.import_original_state(dest, source, *copy.deepcopy(checkpoint))


def test_target_extra_restore_is_not_reported_as_success(inputs, checkpoint, monkeypatch):
    dest = target(inputs)
    real = dest.restore_state

    def bad_restore(*args):
        real(*args)
        dest.restore_count += 1

    monkeypatch.setattr(dest, "restore_state", bad_restore)
    with pytest.raises(ValueError, match="undeclared metadata"):
        m.import_original_state(
            dest, m.AcceleratedContinuousPersistentNodeTracker(*inputs), *copy.deepcopy(checkpoint)
        )

"""Synthetic CPU dispatch/context/migration checks; no real fit or temporal step."""

# ruff: noqa: E402

import copy
import sys
from dataclasses import replace
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"),
    str(Path(__file__).resolve().parents[2] / "src"),
]

import fast_tail_recovery_tracker as m
import numpy as np
import pytest
import test_tail_recovery_tracker as old
import torch
from continuous_tracker import ContinuousPersistentNodeTracker
from persistent_node_tracker import PersistentNodeConfig

QUALIFICATION_SHA = "21244298988acd69ac6bc59edc0a833298144af80f5582b6b06de888ac0bb58c"


@pytest.fixture(scope="module")
def inputs():
    return old.inputs.__wrapped__()


@pytest.fixture(scope="module")
def checkpoint(inputs):
    return old.checkpoint.__wrapped__(inputs)


@pytest.fixture(autouse=True)
def forbid_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic checks must not execute a fit, initialization or temporal step")

    for method in ("begin", "advance", "_advance_owned"):
        monkeypatch.setattr(ContinuousPersistentNodeTracker, method, forbidden)
    monkeypatch.setattr(m.DirectionalDeviceFitter, "_fit", forbidden)
    monkeypatch.setattr(m.FastDirectionalDeviceFitter, "_fit", forbidden)
    monkeypatch.setattr(m.AcceleratedContinuousPersistentNodeTracker, "_fit", forbidden)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("raise_error", [False, True])
def test_dispatch_exact_arguments_return_and_exception(
    monkeypatch, inputs, enabled, device, raise_error
):
    tracker = m.FastTailRecoveryContinuousTracker(*inputs, fast_fitter_enabled=enabled)
    # Constructing a device descriptor does not initialize CUDA or allocate tensors.
    tracker.device = torch.device(device)
    args = tuple(object() for _ in range(4))
    returned = object()
    failure = RuntimeError("synthetic delegated failure")
    calls = []

    def invoke(label):
        def stub(self, proposals, states, rays, targets):
            calls.append((label, self, proposals, states, rays, targets))
            if raise_error:
                raise failure
            return returned

        return stub

    monkeypatch.setattr(m.DirectionalDeviceFitter, "_fit", invoke("frozen"))
    monkeypatch.setattr(m.FastDirectionalDeviceFitter, "_fit", invoke("fast"))
    if raise_error:
        with pytest.raises(RuntimeError) as caught:
            tracker._fit(*args)
        assert caught.value is failure
    else:
        assert tracker._fit(*args) is returned
    assert len(calls) == 1
    assert calls[0][0] == ("fast" if enabled and device == "cuda" else "frozen")
    assert calls[0][1] is tracker
    assert all(a is b for a, b in zip(calls[0][2:], args, strict=True))
    assert tracker.initialize_count == tracker.restore_count == tracker.processed_frames == 0


@pytest.mark.parametrize("value", [0, 1, None, "true", np.bool_(True)])
def test_fast_switch_rejects_non_python_booleans(inputs, value):
    with pytest.raises(ValueError, match="boolean"):
        m.FastTailRecoveryContinuousTracker(*inputs, fast_fitter_enabled=value)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("tail_enabled", [False, True])
def test_context_adds_execution_policy_only_and_inherits_scientific_methods(
    inputs, enabled, tail_enabled
):
    kwargs = dict(tail_recovery_enabled=tail_enabled, bounded_mouth_enabled=False)
    baseline = m.TailRecoveryContinuousTracker(*inputs, **kwargs)._context()
    tracker = m.FastTailRecoveryContinuousTracker(*inputs, fast_fitter_enabled=enabled, **kwargs)
    context = tracker._context()
    assert context["implementation"] == "FastTailRecoveryContinuousTracker"
    policy = context.pop("fast_execution_policy")
    context["implementation"] = baseline["implementation"]
    assert context == baseline
    assert policy["schema"] == "fast_tail_execution_policy_v1"
    assert policy["enabled"] is enabled and policy["execution_only"] is True
    assert policy["clears_identity_review"] is False
    assert policy["qualification_seal_sha256"] == QUALIFICATION_SHA
    assert policy["cuda_fitter"] == (
        "FastDirectionalDeviceFitter" if enabled else "DirectionalDeviceFitter"
    )
    assert policy["cpu_fitter"] == "DirectionalDeviceFitter"
    for name in (
        "_proposals",
        "_age_eligible",
        "advance",
        "_advance_owned",
        "restore_state",
        "finish",
    ):
        assert getattr(type(tracker), name) is getattr(m.TailRecoveryContinuousTracker, name)


@pytest.mark.parametrize("restore_count,enabled", [(0, True), (3, True), (3, False)])
def test_migration_preserves_all39_bytes_metadata_and_sticky_review_once(
    inputs, checkpoint, restore_count, enabled
):
    arrays, metadata = copy.deepcopy(checkpoint)
    metadata["restore_count"] = restore_count
    before = m.array_signature(arrays)
    old_metadata = copy.deepcopy(metadata)
    original = m.AcceleratedContinuousPersistentNodeTracker(*inputs)
    target = m.FastTailRecoveryContinuousTracker(*inputs, fast_fitter_enabled=enabled)
    report = m.import_original_fast_tail_state(target, original, arrays, metadata)
    restored, result_meta = target.export_state()
    assert len(restored) == 39 and m.array_signature(restored) == before
    assert m.array_signature(arrays) == before and metadata == old_metadata
    assert target.initialize_count == 1 and target.restore_count == restore_count + 1
    assert target.processed_frames == 5 and target.previous_frame == 1004
    assert original.initialize_count == original.restore_count == original.processed_frames == 0
    assert target._states[0].review[2]
    assert target._states[0].node_missing[0, 2] == 5
    assert report["schema"] == "fast_tail_original_state_import_v1"
    assert report["tail_validation"]["status"] == "ok"
    assert report["added_restore_count"] == 1 and report["new_begin_calls"] == 0
    assert report["all_arrays_byte_preserved"] is True
    assert report["review_cleared_by_migration"] is False
    expected_meta = copy.deepcopy(metadata)
    expected_meta["context"] = target._context()
    expected_meta["restore_count"] += 1
    assert result_meta == expected_meta
    for key in restored:
        assert not np.shares_memory(restored[key], arrays[key])
    second = m.FastTailRecoveryContinuousTracker(*inputs, fast_fitter_enabled=enabled)
    second.restore_state(restored, result_meta)
    assert second.restore_count == restore_count + 2
    assert m.array_signature(second.export_state()[0]) == before


def test_restored_fast_context_cannot_masquerade_as_tail_or_opposite_switch(inputs, checkpoint):
    target = m.FastTailRecoveryContinuousTracker(*inputs)
    original = m.AcceleratedContinuousPersistentNodeTracker(*inputs)
    m.import_original_fast_tail_state(target, original, *copy.deepcopy(checkpoint))
    arrays, metadata = target.export_state()
    for other in (
        m.TailRecoveryContinuousTracker(*inputs),
        m.AcceleratedContinuousPersistentNodeTracker(*inputs),
        m.FastTailRecoveryContinuousTracker(*inputs, fast_fitter_enabled=False),
    ):
        with pytest.raises(ValueError, match="context"):
            other.restore_state(arrays, metadata)
        assert other.initialize_count == other.restore_count == other.processed_frames == 0


@pytest.mark.parametrize(
    "fault",
    [
        "seed",
        "config",
        "persistence",
        "context",
        "history",
        "age",
        "target_started",
        "original_started",
    ],
)
def test_migration_rejects_state_or_context_drift_without_target_restore(inputs, checkpoint, fault):
    arrays, metadata = copy.deepcopy(checkpoint)
    rig, seed, config = inputs
    target = m.FastTailRecoveryContinuousTracker(
        rig,
        seed + 0.01 if fault == "seed" else seed,
        replace(config, motion_weight=0.3) if fault == "config" else config,
        persistence_config=PersistentNodeConfig(minimum_missing_frames=7)
        if fault == "persistence"
        else None,
    )
    original = m.AcceleratedContinuousPersistentNodeTracker(*inputs)
    if fault == "context":
        metadata["context"]["unexpected"] = 1
    elif fault == "history":
        arrays["history_parent"][-1] = 999
    elif fault == "age":
        arrays["state_node_missing"][0, 0, 2] = 0
    elif fault == "target_started":
        target.initialize_count = 1
    elif fault == "original_started":
        original.initialize_count = 1
    with pytest.raises(ValueError):
        m.import_original_fast_tail_state(target, original, arrays, metadata)
    assert target.restore_count == target.processed_frames == 0
    assert target.initialize_count == int(fault == "target_started")


def test_migration_requires_exact_tracker_types(inputs, checkpoint):
    class DerivedFast(m.FastTailRecoveryContinuousTracker):
        pass

    class DerivedOriginal(m.AcceleratedContinuousPersistentNodeTracker):
        pass

    for target, original in (
        (DerivedFast(*inputs), m.AcceleratedContinuousPersistentNodeTracker(*inputs)),
        (m.FastTailRecoveryContinuousTracker(*inputs), DerivedOriginal(*inputs)),
        (
            m.TailRecoveryContinuousTracker(*inputs),
            m.AcceleratedContinuousPersistentNodeTracker(*inputs),
        ),
    ):
        with pytest.raises(ValueError):
            m.import_original_fast_tail_state(target, original, *copy.deepcopy(checkpoint))
        assert target.initialize_count == original.initialize_count == 0

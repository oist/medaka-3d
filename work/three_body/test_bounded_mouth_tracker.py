"""Synthetic policy and full-state migration tests; no actual tracking or CUDA."""


import copy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"))

import bounded_mouth_tracker as m
import numpy as np
import pytest
import test_directional_chest_tracker as toy
import torch
from persistent_node_tracker import PersistentNodeConfig, PersistentNodeState
from tracker import Proposal


@pytest.fixture(scope="module")
def inputs():
    return toy.inputs.__wrapped__()


@pytest.fixture(scope="module")
def checkpoint(inputs):
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return toy.checkpoint(inputs)
    finally:
        torch.set_num_threads(before)


def setup_policy(monkeypatch, inputs, *, enabled=True, count=6):
    tracker = m.BoundedMouthContinuousTracker(*inputs, bounded_mouth_enabled=enabled)
    pose = inputs[1]
    plan = np.zeros((3, 3, 3), int)
    original = [
        Proposal(pose.copy(), pose.copy(), plan.copy(), i % 6, "predict") for i in range(count)
    ]
    state = PersistentNodeState(
        pose.copy(),
        np.zeros((3, 3)),
        np.zeros(3, int),
        np.ones(3, bool),
        10.0,
        np.array([[4, 0, 0], [3, 0, 0], [9, 0, 0]]),
        np.array([4, 9, 3]),
    )
    result = SimpleNamespace(
        proposals=[
            SimpleNamespace(fish=fish, pose_xyz=pose + k + 1, plan=plan.copy())
            for k, fish in enumerate([0, 1, 2, 0])
        ],
        diagnostics={},
    )
    called = []
    monkeypatch.setattr(m.DirectionalChestContinuousTracker, "_proposals", lambda *args: original)

    def helper(*args):
        called.append(args)
        return result

    monkeypatch.setattr(m.bounded_mouth_recovery, "bounded_mouth_recovery_proposals", helper)
    return tracker, state, original, result, called


@pytest.mark.parametrize("case", ["disabled", "persistence", "mouth_age", "chest_age"])
def test_no_helper_when_original_age_gate_closed(monkeypatch, inputs, case):
    tracker, state, original, _, called = setup_policy(
        monkeypatch, inputs, enabled=case != "disabled"
    )
    if case == "persistence":
        tracker.persistence_config = PersistentNodeConfig(enabled=False)
    if case == "mouth_age":
        state.node_missing[:, 0] = 3
    if case == "chest_age":
        state.direct_pectoral_age[:] = 3
    assert tracker._proposals([state], None) is original
    assert called == [] and not tracker.last_bounded_mouth_diagnostic["enabled"]


def test_exact_original_prefix_single_helper_age_filter_owned_append(monkeypatch, inputs):
    tracker, state, original, result, called = setup_policy(monkeypatch, inputs)
    before = state.pose.copy(), state.review.copy()
    proposals = tracker._proposals([state], None)
    assert len(called) == 1 and len(proposals) == len(original) + 2
    assert all(a is b for a, b in zip(proposals[: len(original)], original, strict=True))
    for p in proposals[len(original) :]:
        assert p.kind == "bounded_mouth_recovery" and p.parent == 0 and p.permutation == (0, 1, 2)
        assert np.array_equal(p.predicted, original[0].predicted)
    proposals[-1].pose.fill(-99)
    assert np.all(result.proposals[-1].pose_xyz > 0)
    assert np.array_equal(state.pose, before[0]) and np.array_equal(state.review, before[1])
    assert tracker.last_bounded_mouth_diagnostic["excluded_by_age"] == 2


@pytest.mark.parametrize("remaining", [0, 1])
def test_cap_preserves_full_original_pool(monkeypatch, inputs, remaining):
    cap = inputs[2].max_hypotheses + 6
    tracker, state, original, _, called = setup_policy(monkeypatch, inputs, count=cap - remaining)
    actual = tracker._proposals([state], None)
    assert len(actual) == cap and len(called) == 1
    assert all(a is b for a, b in zip(actual[: len(original)], original, strict=True))
    assert tracker.last_bounded_mouth_diagnostic["excluded_by_budget"] == 2 - remaining


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_enabled_type_and_context_scope(inputs, value):
    with pytest.raises(ValueError, match="boolean"):
        m.BoundedMouthContinuousTracker(*inputs, bounded_mouth_enabled=value)


def test_migration_39arrays_sticky_review_and_one_restore(inputs, checkpoint):
    arrays, metadata = checkpoint
    before, old_meta = m.array_signature(arrays), copy.deepcopy(metadata)
    old = m.AcceleratedContinuousPersistentNodeTracker(*inputs)
    target = m.BoundedMouthContinuousTracker(*inputs)
    report = m.import_original_bounded_state(target, old, arrays, metadata)
    restored, new_meta = target.export_state()
    assert len(restored) == 39 and m.array_signature(restored) == before
    assert m.array_signature(arrays) == before and metadata == old_meta
    assert target.restore_count == metadata["restore_count"] + 1 == 1
    assert target.initialize_count == 1 and old.initialize_count == 0
    assert target._states[0].review[2] and target._states[0].node_missing[2, 1] == 5
    assert report["new_begin_calls"] == 0 and report["added_restore_count"] == 1
    assert new_meta["context"]["bounded_mouth_policy"]["enabled"]
    resumed = m.BoundedMouthContinuousTracker(*inputs)
    resumed.restore_state(restored, new_meta)
    assert resumed.restore_count == 2 and m.array_signature(resumed.export_state()[0]) == before
    off = m.BoundedMouthContinuousTracker(*inputs, bounded_mouth_enabled=False)
    with pytest.raises(ValueError, match="context"):
        off.restore_state(restored, new_meta)


@pytest.mark.parametrize("change", ["config", "history", "age", "already_started"])
def test_migration_cannot_skip_original_validation(inputs, checkpoint, change):
    arrays, metadata = copy.deepcopy(checkpoint)
    rig, pose, cfg = inputs
    target = m.BoundedMouthContinuousTracker(
        rig, pose, replace(cfg, motion_weight=0.3) if change == "config" else cfg
    )
    old = m.AcceleratedContinuousPersistentNodeTracker(*inputs)
    if change == "history":
        arrays["history_parent"][-1] = 999
    if change == "age":
        arrays["state_node_missing"][0, 2, 1] = 0
    if change == "already_started":
        target.begin()
    with pytest.raises(ValueError):
        m.import_original_bounded_state(target, old, arrays, metadata)
    assert target.initialize_count == (1 if change == "already_started" else 0)

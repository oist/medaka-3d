"""Small synthetic integration checks; no original data, optimizer or GPU.

Only proposals/fitted results and flow/pixel targets are fixed. The original
support geometry, both temporal admission loops and 39-array codecs run.
"""

# ruff: noqa: E402

import copy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"),
    str(Path(__file__).resolve().parents[2] / "src"),
]

import continuous_tracker as continuous
import numpy as np
import observation_reserve_tracker as m
import pytest
import test_body_row_tracker as memory
from directional_device_fitter import DirectionalDeviceFitter
from fast_directional_device_fitter import FastDirectionalDeviceFitter

NATIVE = 82718
FROZEN_ADVANCE = continuous.ContinuousPersistentNodeTracker._advance_owned
FROZEN_SUPPORT = continuous.supported_nodes


@pytest.fixture(autouse=True)
def no_actual_inputs_or_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("real input, public rollout or optimizer is forbidden")

    monkeypatch.setattr(np, "load", forbidden)
    for name in ("begin", "advance", "_advance_owned", "track"):
        monkeypatch.setattr(continuous.ContinuousPersistentNodeTracker, name, forbidden)
    monkeypatch.setattr(DirectionalDeviceFitter, "_fit", forbidden)
    monkeypatch.setattr(FastDirectionalDeviceFitter, "_fit", forbidden)


@pytest.fixture
def inputs():
    rig, pose, config = memory.inputs.__wrapped__()
    return rig, pose, replace(config, beam_size=6, max_hypotheses=8)


def old_tracker(inputs):
    return m.nodewise.BodyRowContinuousTracker(*inputs, body_row_effective_from_native=78020)


def new_tracker(inputs, boundary=NATIVE):
    return m.ObservationReserveContinuousTracker(
        *inputs,
        body_row_effective_from_native=78020,
        observation_reserve_effective_from_native=boundary,
    )


@pytest.fixture
def checkpoint(inputs):
    arrays, metadata = copy.deepcopy(memory.checkpoint.__wrapped__(inputs))
    metadata["context"] = old_tracker(inputs)._context()
    metadata["restore_count"] = 7
    old_tracker(inputs)._decode_state(arrays, metadata)
    assert len(arrays) == 39
    return arrays, metadata


def exact_arrays(first, second):
    assert first.keys() == second.keys()
    for key in first:
        assert first[key].dtype == second[key].dtype and first[key].shape == second[key].shape
        assert first[key].tobytes() == second[key].tobytes(), key


def fixed_bank(monkeypatch, inputs, checkpoint, *, reserve):
    old, new = old_tracker(inputs), new_tracker(inputs)
    old.restore_state(*copy.deepcopy(checkpoint))
    m.import_nodewise_state(new, old_tracker(inputs), *copy.deepcopy(checkpoint))
    poses = np.stack([inputs[1] + [0.6 * i, 0, 0] for i in range(8)])
    plans = [
        np.broadcast_to(np.arange(3, dtype=np.int32)[:, None, None] + 3 * i, (3, 3, 3)).copy()
        for i in range(8)
    ]
    proposals = [
        memory.Proposal(p.copy(), p.copy(), plan, 0, "predict", (0, 1, 2))
        for p, plan in zip(poses, plans, strict=True)
    ]
    points = {
        view: inputs[0].camera(view).project(poses).reshape(24, 3, 2).copy()
        for view in memory.VIEWS
    }
    # The proposed plan looks complete. Actual supported sources must lose the
    # first two views' tail in candidates0..6 (all candidates for a no-op).
    for view in memory.VIEWS[:2]:
        points[view][: 21 if reserve else 24, 2] = np.nan
    images = {view: np.zeros((4, 5, 3), np.uint8) for view in memory.VIEWS}
    frame = memory.FrameObservation(
        NATIVE,
        points,
        images,
        {view: np.empty((0, 4, 5), bool) for view in memory.VIEWS},
        images,
    )
    costs = np.array([2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 7.0])
    parts = np.tile([2.0, 0.4, 0.3, 0.2, 2.0], (8, 1))
    parts[7, [0, 4]] = 1.0
    masks = np.zeros((8, 3), bool)
    calls = dict(propose=[], fit=[], support=[])

    def propose(self, states, rays, flow):
        assert states is self._states
        self._recovery_calls += 1
        self.last_body_row_diagnostics = dict(frame_idx=NATIVE, frame_completed=False)
        calls["propose"].append(self)
        return proposals

    def fit(self, actual, states, rays, targets):
        assert actual is proposals and states is self._states
        self.last_instance_diagnostics = dict(frame_idx=NATIVE, synthetic_fixed_fit=True)
        calls["fit"].append(self)
        return poses, costs, parts, masks

    def support(pose, plan, rays, rig):
        index = next(i for i, proposed in enumerate(plans) if plan is proposed)
        calls["support"].append(index)
        return FROZEN_SUPPORT(pose, plan, rays, rig)

    monkeypatch.setattr(m.nodewise.BodyRowContinuousTracker, "_proposals", propose)
    monkeypatch.setattr(m.nodewise.BodyRowContinuousTracker, "_fit", fit)
    monkeypatch.setattr(
        m.sparse.FastTailRecoveryContinuousTracker, "_advance_owned", FROZEN_ADVANCE
    )
    monkeypatch.setattr(continuous, "sample_pixels", lambda *args: object())
    monkeypatch.setattr(
        continuous.FlowObservations,
        "from_frames",
        lambda *args: SimpleNamespace(node_valid=np.zeros((3, 3), bool)),
    )
    monkeypatch.setattr(continuous, "supported_nodes", support)
    return SimpleNamespace(
        old=old,
        new=new,
        frame=frame,
        poses=poses,
        costs=costs,
        parts=parts,
        masks=masks,
        plans=plans,
        calls=calls,
    )


def test_real_nodewise39_import_preserves_bytes_and_adds_one_restore(inputs, checkpoint):
    arrays, metadata = copy.deepcopy(checkpoint)
    original_meta = copy.deepcopy(metadata)
    target, validator = new_tracker(inputs), old_tracker(inputs)
    report = m.import_nodewise_state(target, validator, arrays, metadata)
    saved, current_meta = target.export_state()
    exact_arrays(arrays, saved)
    assert metadata == original_meta and validator.initialize_count == validator.restore_count == 0
    assert target.initialize_count == 1 and target.restore_count == 8
    assert target.previous_frame == NATIVE - 1 and target.processed_frames == 5
    assert report["added_restore_count"] == 1 and report["all_arrays_byte_preserved"]
    assert not report["review_cleared_by_migration"]
    assert saved["state_review"][:, 2].all() and saved["history_identity_review"][:, 2].all()
    expected = copy.deepcopy(metadata)
    expected["context"] = target._context()
    expected["restore_count"] += 1
    assert current_meta == expected
    assert target.last_observation_reserve_diagnostics is None
    wrong = new_tracker(inputs, NATIVE + 1)
    with pytest.raises(ValueError, match="immediately follow"):
        m.import_nodewise_state(wrong, old_tracker(inputs), arrays, metadata)
    assert wrong.initialize_count == wrong.restore_count == 0


def test_noop_matches_original_temporal_loop_all39(monkeypatch, inputs, checkpoint):
    bank = fixed_bank(monkeypatch, inputs, checkpoint, reserve=False)
    bank.old._advance_owned(bank.frame)
    old_calls = bank.calls["support"].copy()
    bank.calls["support"].clear()
    bank.new._advance_owned(bank.frame)
    old_arrays, old_meta = bank.old.export_state()
    new_arrays, new_meta = bank.new.export_state()
    exact_arrays(old_arrays, new_arrays)
    new_meta["context"] = old_meta["context"]
    assert new_meta == old_meta
    diag = bank.new.export_observation_reserve_diagnostics()
    assert not diag["selection"]["changed"] and not diag["eligible"].any()
    assert old_calls == [2, 3, 0, 1, 4, 5]
    assert bank.calls["support"] == list(range(8)) and diag["support_evaluation_count"] == 8
    assert diag["selected_support_reused"] and diag["baseline_best_index"] == 2


def test_last_slot_only_actual_support_cached_costs_prefix_and_review_preserved(
    monkeypatch, inputs, checkpoint
):
    bank = fixed_bank(monkeypatch, inputs, checkpoint, reserve=True)
    parents = bank.new._states.copy()
    parent_before = {
        "parent_" + key: np.asarray([getattr(state, key) for state in parents]).copy()
        for key in ("pose", "velocity", "missing", "node_missing", "direct_pectoral_age")
    }
    untouched = [x.copy() for x in (bank.poses, bank.costs, bank.parts, bank.masks)]
    bank.old._advance_owned(bank.frame)
    bank.calls["support"].clear()
    bank.new._advance_owned(bank.frame)
    diagnostic = bank.new.export_observation_reserve_diagnostics()
    for key, value in parent_before.items():
        exact_arrays({key: diagnostic[key]}, {key: value})
        actual = np.asarray([getattr(state, key.removeprefix("parent_")) for state in parents])
        exact_arrays({key: actual}, {key: value})
    np.testing.assert_array_equal(diagnostic["candidate_mask_support"], bank.masks)
    assert not np.shares_memory(
        bank.new.last_observation_reserve_diagnostics["candidate_mask_support"], bank.masks
    )
    assert diagnostic["selection"]["baseline_indices"] == [2, 3, 0, 1, 4, 5]
    assert diagnostic["selection"]["selected_indices"] == [2, 3, 0, 1, 4, 7]
    assert diagnostic["baseline_best_index"] == 2
    np.testing.assert_array_equal(diagnostic["eligible"], [False] * 7 + [True])
    assert all((plan >= 0).all() for plan in bank.plans)
    assert (diagnostic["supported_sources"][:7, :, :2, 2] == -1).all()
    assert not diagnostic["observed"][:7, :, 2].any()
    assert diagnostic["observed"][7].all()
    assert bank.calls["support"] == list(range(8))
    old_arrays, _ = bank.old.export_state()
    new_arrays, _ = bank.new.export_state()
    for key in old_arrays:
        if key.startswith("history_") and key != "history_offsets":
            # Last row is the one permitted replacement; all earlier history
            # and current first five rows must remain byte-identical.
            exact_arrays({key: old_arrays[key][:-1]}, {key: new_arrays[key][:-1]})
        elif key.startswith("state_"):
            exact_arrays({key: old_arrays[key][:5]}, {key: new_arrays[key][:5]})
        else:
            exact_arrays({key: old_arrays[key]}, {key: new_arrays[key]})
    record = bank.new._history[-1][-1]
    np.testing.assert_array_equal(record["sources"], diagnostic["supported_sources"][7])
    np.testing.assert_array_equal(record["observed"], diagnostic["observed"][7])
    assert record["costs"] == bank.costs[7] and record["path_costs"] == 5.0 + bank.costs[7]
    np.testing.assert_array_equal(record["loss_parts"], bank.parts[7])
    assert record["identity_review"][2] and bank.new._states[-1].review[2]
    np.testing.assert_array_equal(record["node_missing"], np.zeros((3, 3), int))
    for actual, before in zip(
        (bank.poses, bank.costs, bank.parts, bank.masks), untouched, strict=True
    ):
        assert actual.tobytes() == before.tobytes()
    diagnostic["supported_sources"][:] = 999
    for key in parent_before:
        diagnostic[key][:] = 999
        exact_arrays(
            {key: bank.new.export_observation_reserve_diagnostics()[key]},
            {key: parent_before[key]},
        )
    diagnostic["candidate_mask_support"][:] = True
    assert not bank.new.export_observation_reserve_diagnostics()["candidate_mask_support"].any()
    assert not (bank.new.export_observation_reserve_diagnostics()["supported_sources"] == 999).any()


@pytest.mark.parametrize("fault", ["boundary", "temporal", "missing_instance"])
def test_failure_clears_all_companions_and_owned_buffers(monkeypatch, inputs, fault):
    tracker = new_tracker(inputs)
    tracker.last_body_row_diagnostics = {"stale": True}
    tracker.last_instance_diagnostics = {"stale": True}
    tracker.last_observation_reserve_diagnostics = {"stale": True}

    def temporal(self, frame):
        self.last_body_row_diagnostics = dict(frame_idx=NATIVE, frame_completed=False)
        self.last_observation_reserve_diagnostics = dict(frame_idx=NATIVE, frame_completed=False)
        if fault == "temporal":
            raise ValueError("synthetic temporal failure")

    monkeypatch.setattr(
        m.ObservationReserveContinuousTracker, "_advance_temporal_with_reserve", temporal
    )
    frame = SimpleNamespace(frame_idx=NATIVE - int(fault == "boundary"), masks={}, detections={})
    with pytest.raises((ValueError, RuntimeError)):
        tracker._advance_owned(frame)
    assert tracker.last_body_row_diagnostics is None
    assert tracker.last_instance_diagnostics is None
    assert tracker.last_observation_reserve_diagnostics is None
    assert (
        tracker._body_row_frame
        is tracker._instance_frame
        is tracker._observation_reserve_frame
        is None
    )
    assert tracker._instance_observations is tracker._instance_gate_seconds is None
    with pytest.raises(RuntimeError, match="no completed"):
        tracker.export_observation_reserve_diagnostics()


def test_constructor_and_temporal_scope_reject_invalid_controls(inputs):
    with pytest.raises(ValueError, match="six-candidate"):
        new_tracker((*inputs[:2], replace(inputs[2], beam_size=5)))
    tracker = new_tracker(inputs)
    with pytest.raises(RuntimeError, match="current owned"):
        tracker._advance_temporal_with_reserve(SimpleNamespace(frame_idx=NATIVE))

"""CPU contracts for appended proposals and a real synthetic 39-array resume.

The integration fixture fixes the old proposal bank and optimizer output; actual
new row geometry, completion, support, temporal update and checkpoint codecs run.
No scientific files, video readers, GPU or existing test suites are executed.
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

import body_row_tracker as m
import continuous_tracker as continuous
import numpy as np
import pytest
import sparse_instance_rescore_tracker as sparse
import test_instance_rescore_tracker as toy
from directional_device_fitter import DirectionalDeviceFitter
from fast_directional_device_fitter import FastDirectionalDeviceFitter
from observations import FrameObservation
from tracker import Proposal

from medaka3d.types import VIEWS

NATIVE = 82718
FROZEN_ADVANCE_OWNED = continuous.ContinuousPersistentNodeTracker._advance_owned


@pytest.fixture(scope="module")
def inputs():
    return toy.inputs.__wrapped__()


@pytest.fixture(autouse=True)
def forbid_real_execution(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("synthetic test reached a scientific reader, optimizer or public rollout")

    monkeypatch.setattr(np, "load", fail)
    for name in ("begin", "advance", "_advance_owned"):
        monkeypatch.setattr(continuous.ContinuousPersistentNodeTracker, name, fail)
    monkeypatch.setattr(DirectionalDeviceFitter, "_fit", fail)
    monkeypatch.setattr(FastDirectionalDeviceFitter, "_fit", fail)


@pytest.fixture(scope="module")
def checkpoint(inputs):
    # The existing fixture constructs arrays in memory; it does not run tracking.
    arrays, meta = copy.deepcopy(toy.checkpoint.__wrapped__(inputs))
    arrays["history_frame_idx"] += NATIVE - 1 - meta["previous_frame"]
    meta["previous_frame"], meta["restore_count"] = NATIVE - 1, 6
    validator = sparse.SparseInstanceRescoreContinuousTracker(*inputs)
    meta["context"] = validator._context()
    validator._decode_state(arrays, meta)
    assert len(arrays) == 39 and validator.initialize_count == 0
    return arrays, meta


def make(inputs, **kwargs):
    return m.BodyRowContinuousTracker(*inputs, body_row_effective_from_native=NATIVE, **kwargs)


def plans():
    return np.broadcast_to(np.arange(3, dtype=np.int32)[:, None, None], (3, 3, 3)).copy()


def hook_fixture(monkeypatch, inputs, incoming=2, scene_count=3):
    tracker = make(inputs)
    tracker._body_row_frame = NATIVE
    pose = inputs[1]
    prediction = pose + np.array([0.5, 0.25, 0.125])
    original = [
        Proposal(
            pose.copy(),
            prediction if i == 0 else pose + 9,
            plans(),
            0 if i == 0 else 1,
            "predict" if i == 0 else "sentinel",
            (0, 1, 2),
        )
        for i in range(incoming)
    ]
    scenes = []
    for i in range(scene_count):
        plan = plans()
        plan[0, :2, 2] = -1
        measured = np.ones((3, 3), bool)
        measured[0, 2] = False
        scenes.append(
            dict(
                pose_xyz=pose + i * 0.01,
                proposed_sources=plan,
                triangulated_nodes=measured,
                source_rows=plans()[..., 0],
            )
        )
    pool, rays, flow = [{"sentinel": np.arange(3)}], object(), object()
    states = [SimpleNamespace(pose=pose - 9), SimpleNamespace(pose=pose + 99)]

    def inherited(self, actual_states, actual_rays, actual_flow):
        assert self is tracker and actual_states is states
        assert actual_rays is rays and actual_flow is flow
        return original

    def pool_fn(rig, actual_rays, lower, upper, config):
        assert rig is inputs[0] and actual_rays is rays
        assert lower is tracker.lower and upper is tracker.upper
        assert config is tracker.body_row_config
        return pool, {"synthetic_pool": True}

    def scenes_fn(actual_pool, predicted, lengths, lower, upper, config):
        assert actual_pool is pool and predicted is prediction
        assert lengths is tracker.lengths  # constructor-fixed body lengths
        assert predicted is not states[0].pose and predicted is not original[-1].predicted
        return scenes, {"admissible_scenes": scene_count + 2}

    monkeypatch.setattr(sparse.SparseInstanceRescoreContinuousTracker, "_proposals", inherited)
    monkeypatch.setattr(m, "body_row_pool", pool_fn)
    monkeypatch.setattr(m, "joint_body_row_scenes", scenes_fn)
    return SimpleNamespace(
        tracker=tracker,
        original=original,
        scenes=scenes,
        states=states,
        rays=rays,
        flow=flow,
        prediction=prediction,
    )


def test_appends_without_replacing_objects_or_using_other_parent(monkeypatch, inputs):
    s = hook_fixture(monkeypatch, inputs)
    before = m.array_signature({str(i): p.pose for i, p in enumerate(s.original)})
    out = s.tracker._proposals(s.states, s.rays, s.flow)
    assert all(out[i] is p for i, p in enumerate(s.original))
    assert len(out) == 5
    assert before == m.array_signature({str(i): p.pose for i, p in enumerate(s.original)})
    for proposal, scene in zip(out[2:], s.scenes, strict=True):
        assert proposal.parent == 0 and proposal.kind == "body_row"
        assert proposal.permutation == (0, 1, 2) and proposal.predicted is s.prediction
        np.testing.assert_array_equal(proposal.plan, scene["proposed_sources"])
        assert not np.shares_memory(proposal.pose, scene["pose_xyz"])
        assert not np.shares_memory(proposal.plan, scene["proposed_sources"])
        assert np.count_nonzero(proposal.plan[0, :, 2] >= 0) == 1
    d = s.tracker.last_body_row_diagnostics
    assert d["filled_nodes_by_proposal"][:, 0, 2].all()
    assert not d["support_recomputed"] and not d["frame_completed"]
    with pytest.raises(RuntimeError, match="no completed"):
        s.tracker.export_body_row_diagnostics()


@pytest.mark.parametrize("incoming,scenes,added", [(9, 3, 1), (10, 3, 0), (2, 0, 0)])
def test_capacity_and_empty_pool_record_zero_additions(
    monkeypatch, inputs, incoming, scenes, added
):
    s = hook_fixture(monkeypatch, inputs, incoming, scenes)
    out = s.tracker._proposals(s.states, s.rays, s.flow)
    d = s.tracker.last_body_row_diagnostics
    assert len(out) == incoming + added <= inputs[2].max_hypotheses + 6
    assert all(out[i] is p for i, p in enumerate(s.original))
    assert d["added_proposal_count"] == added
    assert d["excluded_by_capacity"] == scenes - added
    assert d["headroom"] == 10 - incoming and d["zero_headroom"] == (incoming == 10)
    assert d["new_proposal_plans"].shape == (added, 3, 3, 3)


@pytest.mark.parametrize("damage", ["over_cap", "wrong_parent", "row_reuse", "split_row"])
def test_bad_inherited_or_scene_contract_fails_instead_of_silent_repair(
    monkeypatch, inputs, damage
):
    s = hook_fixture(monkeypatch, inputs, incoming=11 if damage == "over_cap" else 2)
    if damage == "wrong_parent":
        s.original[0].parent = 1
    elif damage == "row_reuse":
        s.scenes[0]["source_rows"][1, 0] = 0
        s.scenes[0]["proposed_sources"][1, 0] = 0
    elif damage == "split_row":
        s.scenes[0]["proposed_sources"][0, 0, 1] = 2
    with pytest.raises(ValueError):
        s.tracker._proposals(s.states, s.rays, s.flow)
    assert s.tracker.last_body_row_diagnostics is None


def test_real_sparse_decoder_migration_all39_and_second_resume(inputs, checkpoint):
    arrays, meta = copy.deepcopy(checkpoint)
    before, original_meta = m.array_signature(arrays), copy.deepcopy(meta)
    source, target = sparse.SparseInstanceRescoreContinuousTracker(*inputs), make(inputs)
    result = m.import_sparse_state(target, source, arrays, meta)
    saved, saved_meta = target.export_state()
    assert m.array_signature(saved) == before == m.array_signature(arrays)
    assert meta == original_meta and source.initialize_count == source.restore_count == 0
    assert target.initialize_count == 1 and target.restore_count == 7
    assert target.previous_frame == NATIVE - 1 and target.processed_frames == 5
    assert result["all_arrays_byte_preserved"] and not result["review_cleared_by_migration"]
    expected = copy.deepcopy(meta)
    expected["context"] = target._context()
    expected["restore_count"] += 1
    assert saved_meta == expected
    old_context = copy.deepcopy(saved_meta["context"])
    policy = old_context.pop("body_row_policy")
    old_context["implementation"] = meta["context"]["implementation"]
    assert old_context == meta["context"] and policy["effective_from_native"] == NATIVE
    assert not policy["retroactive_proposals"] and not policy["old_history_rewritten"]
    resumed = make(inputs)
    resumed.restore_state(saved, saved_meta)
    assert resumed.restore_count == 8 and m.array_signature(resumed.export_state()[0]) == before
    assert saved["history_identity_review"][:, 2].all()
    assert resumed.last_body_row_diagnostics is None
    with pytest.raises(ValueError, match="context"):
        source.restore_state(saved, saved_meta)


@pytest.mark.parametrize("damage", ["boundary", "source_review", "config", "missing_field"])
def test_migration_rejects_before_target_initialization(inputs, checkpoint, damage):
    arrays, meta = copy.deepcopy(checkpoint)
    target = make(inputs)
    if damage == "boundary":
        target = m.BodyRowContinuousTracker(*inputs, body_row_effective_from_native=NATIVE + 1)
    elif damage == "source_review":
        arrays["state_review"][0, 2] = False
    elif damage == "config":
        target = make((*inputs[:2], replace(inputs[2], shape_weight=99)))
    else:
        arrays.pop("state_node_missing")
    with pytest.raises(ValueError):
        m.import_sparse_state(
            target, sparse.SparseInstanceRescoreContinuousTracker(*inputs), arrays, meta
        )
    assert target.initialize_count == 0 and target._states is None


def test_real_row_geometry_support_temporal_update_and_appended39_restore(
    monkeypatch, inputs, checkpoint
):
    tracker = make(inputs, body_row_config=m.BodyRowConfig(per_slot_limit=1, scene_limit=1))
    m.import_sparse_state(
        tracker, sparse.SparseInstanceRescoreContinuousTracker(*inputs), *copy.deepcopy(checkpoint)
    )
    pose = inputs[1]
    detections = {v: inputs[0].camera(v).project(pose).copy() for v in VIEWS}
    for v in VIEWS[:2]:
        detections[v][0, 2] = np.nan  # one-view tail must stay latent after completion
    images = {v: np.zeros((192, 192, 3), np.uint8) for v in VIEWS}
    frame = FrameObservation(
        NATIVE, detections, images, {v: np.empty((0, 192, 192), bool) for v in VIEWS}, images
    )
    original = [Proposal(pose.copy(), pose.copy(), plans(), 0, "predict", (0, 1, 2))]

    def inherited_proposals(self, states, rays, flow):
        # Bookkeeping performed by the replaced frozen proposal generator.
        self._recovery_calls += 1
        return original

    monkeypatch.setattr(
        sparse.SparseInstanceRescoreContinuousTracker, "_proposals", inherited_proposals
    )
    monkeypatch.setattr(continuous, "sample_pixels", lambda *args: object())
    monkeypatch.setattr(
        continuous.FlowObservations,
        "from_frames",
        lambda *args: SimpleNamespace(node_valid=np.zeros((3, 3), bool)),
    )
    fitted_banks = []

    def fit(self, proposals, states, rays, targets):
        assert proposals[0] is original[0] and len(proposals) == 2
        assert proposals[1].kind == "body_row"
        fitted_banks.append(proposals)
        return (
            np.stack([p.pose for p in proposals]),
            np.array([100.0, 0.0]),
            np.zeros((2, 5)),
            np.zeros((2, 3), bool),
        )

    monkeypatch.setattr(sparse.FastTailRecoveryContinuousTracker, "_fit", fit)
    monkeypatch.setattr(
        sparse.FastTailRecoveryContinuousTracker, "_advance_owned", FROZEN_ADVANCE_OWNED
    )
    tracker._advance_owned(frame)
    record = tracker._history[-1][0]
    assert record["proposal_kind"] == "body_row"
    np.testing.assert_allclose(record["pose_xyz"], pose, rtol=0, atol=1e-12)
    expected_observed = np.ones((3, 3), bool)
    expected_observed[0, 2] = False
    np.testing.assert_array_equal(record["observed"], expected_observed)
    assert np.count_nonzero(record["sources"][0, :, 2] >= 0) == 1
    assert record["node_missing"][0, 2] == checkpoint[0]["state_node_missing"][0, 0, 2] + 1
    assert record["identity_review"][2] and tracker._states[0].review[2]
    np.testing.assert_array_equal(record["loss_parts"], np.zeros(5))
    instance = tracker.export_instance_diagnostics()
    np.testing.assert_array_equal(instance["sixth_instance_term"], np.zeros(2))
    diagnostic = tracker.export_body_row_diagnostics()
    assert diagnostic["frame_completed"] and diagnostic["added_proposal_count"] == 1
    assert diagnostic["filled_nodes_by_proposal"][0, 0, 2]
    diagnostic["new_proposal_plans"][...] = 999
    assert not np.any(tracker.export_body_row_diagnostics()["new_proposal_plans"] == 999)
    assert tracker._body_row_frame is tracker._instance_frame is None
    assert len(fitted_banks) == 1
    arrays, metadata = tracker.export_state()
    resumed = make(inputs, body_row_config=tracker.body_row_config)
    resumed.restore_state(arrays, metadata)
    assert len(arrays) == 39 and m.array_signature(arrays) == m.array_signature(
        resumed.export_state()[0]
    )
    assert resumed.previous_frame == NATIVE and resumed.processed_frames == 6
    for key, old in checkpoint[0].items():
        if key.startswith("history_"):
            np.testing.assert_array_equal(arrays[key][: len(old)], old)


@pytest.mark.parametrize("where", ["owned", "before_boundary", "public_validation"])
def test_failure_clears_both_diagnostics_and_owned_scope(monkeypatch, inputs, where):
    tracker = make(inputs)
    tracker.last_body_row_diagnostics = {"stale": True}
    tracker.last_instance_diagnostics = {"stale": True}

    def fail(*args):
        raise ValueError("synthetic failure")

    if where == "public_validation":
        monkeypatch.setattr(sparse.SparseInstanceRescoreContinuousTracker, "advance", fail)
        call, argument = tracker.advance, object()
    elif where == "before_boundary":
        call, argument = tracker._advance_owned, SimpleNamespace(frame_idx=NATIVE - 1)
    else:
        monkeypatch.setattr(sparse.SparseInstanceRescoreContinuousTracker, "_advance_owned", fail)
        call, argument = tracker._advance_owned, SimpleNamespace(frame_idx=NATIVE)
    with pytest.raises(ValueError):
        call(argument)
    assert tracker._body_row_frame is None
    assert tracker.last_body_row_diagnostics is tracker.last_instance_diagnostics is None


def test_owned_scope_and_constructor_context_are_not_mutable_runtime_switches(inputs, checkpoint):
    tracker = make(inputs)
    with pytest.raises(RuntimeError, match="current owned"):
        tracker._proposals([], object())
    m.import_sparse_state(
        tracker, sparse.SparseInstanceRescoreContinuousTracker(*inputs), *copy.deepcopy(checkpoint)
    )
    tracker.body_row_config = replace(tracker.body_row_config, scene_limit=1)
    with pytest.raises(ValueError, match=r"configuration|context|changed"):
        tracker.export_state()

"""Synthetic priority-loop and real 39-field migration contracts; no scientific IO."""

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
import endpoint_mask_priority_tracker as m
import numpy as np
import pytest
import sparse_instance_rescore_tracker as sparse
import test_body_row_tracker as toy
from directional_device_fitter import DirectionalDeviceFitter
from fast_directional_device_fitter import FastDirectionalDeviceFitter
from observations import FrameObservation
from tracker import Proposal

from medaka3d.types import VIEWS

NATIVE = toy.NATIVE
OLD_TEMPORAL_LOOP = continuous.ContinuousPersistentNodeTracker._advance_owned


@pytest.fixture(scope="module")
def inputs():
    return toy.inputs.__wrapped__()


@pytest.fixture(scope="module")
def checkpoint(inputs):
    return toy.checkpoint.__wrapped__(inputs)


@pytest.fixture(autouse=True)
def synthetic_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("scientific IO, optimizer or public rollout is forbidden")

    monkeypatch.setattr(np, "load", forbidden)
    for method in ("begin", "advance", "_advance_owned"):
        monkeypatch.setattr(continuous.ContinuousPersistentNodeTracker, method, forbidden)
    monkeypatch.setattr(DirectionalDeviceFitter, "_fit", forbidden)
    monkeypatch.setattr(FastDirectionalDeviceFitter, "_fit", forbidden)


def make(inputs, **kwargs):
    return m.EndpointMaskPriorityTracker(
        *inputs, endpoint_mask_effective_from_native=NATIVE, **kwargs
    )


def test_priority_preserves_old_ties_and_all_split_fallback():
    costs = np.array([5.0, 1.0, 1.0, 3.0])
    flags = np.zeros((4, 3), bool)
    flags[[1, 3], 2] = True
    original = costs.copy(), flags.copy()
    old, priority, fallback = m.candidate_orders(costs, flags)
    np.testing.assert_array_equal(old, [1, 2, 3, 0])
    np.testing.assert_array_equal(priority, [2, 0, 1, 3])
    assert not fallback
    flags[:] = True
    old, priority, fallback = m.candidate_orders(costs, flags)
    np.testing.assert_array_equal(priority, old)
    assert fallback
    np.testing.assert_array_equal(costs, original[0])
    for invalid in (np.array([np.inf]), np.array([np.nan]), np.empty(0)):
        with pytest.raises(ValueError):
            m.candidate_orders(invalid, np.zeros((len(invalid), 3), bool))


def test_dedup_keeps_distinct_parent_or_permutation():
    poses = np.zeros((4, 3, 3, 3))
    poses[1] += 0.01
    proposals = [
        SimpleNamespace(parent=0, permutation=(0, 1, 2)),
        SimpleNamespace(parent=0, permutation=(0, 1, 2)),
        SimpleNamespace(parent=1, permutation=(0, 1, 2)),
        SimpleNamespace(parent=0, permutation=(0, 2, 1)),
    ]
    assert m.retain_candidates(proposals, poses, [1, 0, 2, 3], 4) == [1, 2, 3]


def loop_fixture(monkeypatch, inputs, checkpoint, *, all_split=False):
    tracker = make(inputs)
    m.import_sparse_state(
        tracker, sparse.SparseInstanceRescoreContinuousTracker(*inputs), *copy.deepcopy(checkpoint)
    )
    pose = inputs[1]
    permutations = [(0, 2, 1), (0, 1, 2)]
    poses = np.stack([pose[list(p)] for p in permutations])
    proposals = [
        Proposal(poses[k].copy(), pose.copy(), toy.plans()[list(p)], 0, "predict", p)
        for k, p in enumerate(permutations)
    ]
    local = np.array([0.25, 8.5])
    parts = np.arange(10, dtype=np.float64).reshape(2, 5)
    mask_support = np.zeros((2, 3), bool)
    split = np.zeros((2, 3), bool)
    split[0, 0] = True  # independent of which identity alternative moves
    if all_split:
        split[1, 1] = True
    detections = {view: inputs[0].camera(view).project(pose).copy() for view in VIEWS}
    for view in VIEWS[:2]:
        detections[view][0, 2] = np.nan
    images = {view: np.zeros((192, 192, 3), np.uint8) for view in VIEWS}
    frame = FrameObservation(
        NATIVE,
        detections,
        images,
        {view: np.empty((0, 192, 192), bool) for view in VIEWS},
        images,
    )

    def proposal_hook(states, rays, flow=None):
        tracker._recovery_calls += 1  # frozen replaced generator's lifecycle bookkeeping
        tracker.last_body_row_diagnostics = {"frame_idx": NATIVE, "frame_completed": False}
        return proposals

    def fit_hook(actual_proposals, states, rays, targets):
        assert actual_proposals is proposals
        tracker.last_endpoint_mask_diagnostics = {
            "frame_idx": NATIVE,
            "frame_completed": False,
            "candidate_split_by_body": split.copy(),
        }
        tracker.last_instance_diagnostics = {"synthetic_fixed_fit": True}
        return poses, local, parts, mask_support

    monkeypatch.setattr(tracker, "_proposals", proposal_hook)
    monkeypatch.setattr(tracker, "_fit", fit_hook)
    monkeypatch.setattr(tracker, "_prepare_endpoint_masks", lambda frame: None)
    monkeypatch.setattr(continuous, "sample_pixels", lambda *args: None)
    monkeypatch.setattr(
        continuous.FlowObservations,
        "from_frames",
        lambda *args: SimpleNamespace(node_valid=np.zeros((3, 3), bool)),
    )
    return SimpleNamespace(
        tracker=tracker,
        frame=frame,
        proposals=proposals,
        poses=poses,
        local=local,
        parts=parts,
        mask_support=mask_support,
        split=split,
    )


def test_actual_loop_preserves_cost_support_and_negative_ambiguity_gap(
    monkeypatch, inputs, checkpoint
):
    s = loop_fixture(monkeypatch, inputs, checkpoint)
    parent = copy.deepcopy(s.tracker._states[0])
    immutable = m.nodewise.array_signature(
        {"poses": s.poses, "cost": s.local, "parts": s.parts, "mask": s.mask_support}
    )
    s.tracker._advance_owned(s.frame)
    d = s.tracker.export_endpoint_mask_diagnostics()
    np.testing.assert_array_equal(d["selected_candidate_indices"], [1, 0])
    np.testing.assert_array_equal(d["baseline_selected_candidate_indices"], [0, 1])
    assert d["selection_changed_from_baseline"]
    assert s.tracker._states[0].cost > s.tracker._states[1].cost
    assert s.tracker._history[-1][0]["score_margin"] == s.local[0] - s.local[1] < 0
    assert s.tracker._history[-1][0]["identity_review"][1:].all()
    assert s.tracker._history[-1][1]["identity_review"][0]  # split OR in retained backup
    assert s.tracker._states[1].review[0]
    for rank, candidate in enumerate((1, 0)):
        record = s.tracker._history[-1][rank]
        np.testing.assert_array_equal(record["pose_xyz"], s.poses[candidate])
        np.testing.assert_array_equal(record["loss_parts"], s.parts[candidate])
        assert record["costs"] == s.local[candidate]
        assert record["path_costs"] == parent.cost + s.local[candidate]
        assert not record["observed"][0, 2]
        assert record["node_missing"][0, 2] == parent.node_missing[0, 2] + 1
        assert record["identity_review"][2]  # prior sticky review retained
    assert immutable == m.nodewise.array_signature(
        {"poses": s.poses, "cost": s.local, "parts": s.parts, "mask": s.mask_support}
    )
    arrays, metadata = s.tracker.export_state()  # real decoder permits non-cost-sorted beam
    resumed = make(inputs)
    resumed.restore_state(arrays, metadata)
    assert m.nodewise.array_signature(resumed.export_state()[0]) == m.nodewise.array_signature(
        arrays
    )
    assert resumed.restore_count == 8 and resumed.previous_frame == NATIVE
    for key, previous in checkpoint[0].items():
        if key.startswith("history_"):
            np.testing.assert_array_equal(arrays[key][: len(previous)], previous)
    assert s.tracker._endpoint_mask_observations is None
    assert (
        s.tracker._body_row_frame
        is s.tracker._instance_frame
        is s.tracker._endpoint_mask_frame
        is None
    )
    d["selected_candidate_indices"][:] = 99
    assert s.tracker.export_endpoint_mask_diagnostics()["selected_candidate_indices"][0] == 1


def test_all_split_actual_loop_uses_old_order_and_preserves_review(monkeypatch, inputs, checkpoint):
    s = loop_fixture(monkeypatch, inputs, checkpoint, all_split=True)
    s.tracker._advance_owned(s.frame)
    d = s.tracker.export_endpoint_mask_diagnostics()
    assert d["all_candidates_split_fallback"]
    assert not d["selection_changed_from_baseline"]
    np.testing.assert_array_equal(d["selected_candidate_indices"], [0, 1])
    assert s.tracker._states[0].review[0] and s.tracker._history[-1][0]["identity_review"][0]
    assert s.tracker._states[1].review[1] and s.tracker._history[-1][1]["identity_review"][1]


def test_no_split_matches_real_frozen_temporal_loop_all39(monkeypatch, inputs, checkpoint):
    new = loop_fixture(monkeypatch, inputs, checkpoint)
    new.split[:] = False
    new.tracker._advance_owned(new.frame)
    old = loop_fixture(monkeypatch, inputs, checkpoint)
    old.split[:] = False
    OLD_TEMPORAL_LOOP(old.tracker, old.frame)
    old_arrays, old_metadata = old.tracker.export_state()
    new_arrays, new_metadata = new.tracker.export_state()
    assert len(old_arrays) == len(new_arrays) == 39
    assert m.nodewise.array_signature(old_arrays) == m.nodewise.array_signature(new_arrays)
    assert old_metadata == new_metadata
    assert not new.tracker.export_endpoint_mask_diagnostics()["selection_changed_from_baseline"]


def test_real_fit_hook_classifies_split_without_changing_sparse_five_plus_sixth(
    monkeypatch, inputs
):
    tracker = make(inputs)
    pose = inputs[1]
    fitted = np.stack([pose.copy(), pose.copy()])
    fitted[0, 0, 2] = pose[1, 2]  # mouth/chest in fish0, tail in separate fish1 mask
    costs = np.array([0.25, 8.5])
    parts = np.arange(10, dtype=np.float64).reshape(2, 5)
    supports = np.array([[True, False, True], [False, True, False]])
    proposals = [
        Proposal(p.copy(), pose.copy(), toy.plans(), 0, "predict", (0, 1, 2)) for p in fitted
    ]
    states = [SimpleNamespace(cost=7.0)]
    detections = {view: inputs[0].camera(view).project(pose).copy() for view in VIEWS}
    masks = {}
    for view in VIEWS:
        masks[view] = np.zeros((3, 192, 192), bool)
        for body, points in enumerate(detections[view]):
            lo = np.floor(points.min(axis=0)).astype(int) - 2
            hi = np.ceil(points.max(axis=0)).astype(int) + 3
            masks[view][body, lo[1] : hi[1], lo[0] : hi[0]] = True
    images = {view: np.zeros((192, 192, 3), np.uint8) for view in VIEWS}
    frame = FrameObservation(NATIVE, detections, images, masks, images)
    result = {}

    def fixed_optimizer(self, *args):
        return fitted, costs, parts, supports

    def only_fit(owned_frame):
        assert owned_frame is frame
        baseline = sparse.SparseInstanceRescoreContinuousTracker._fit(
            tracker, proposals, states, object(), object()
        )
        actual = tracker._fit(proposals, states, object(), object())
        for index in range(4):
            np.testing.assert_array_equal(actual[index], baseline[index])
        for index in (0, 2, 3):
            assert actual[index] is baseline[index]
        result["fit"] = actual
        tracker.last_body_row_diagnostics = {"frame_idx": NATIVE, "frame_completed": False}
        tracker._previous_frame = NATIVE  # synthetic completed-frame association only

    monkeypatch.setattr(sparse.FastTailRecoveryContinuousTracker, "_fit", fixed_optimizer)
    monkeypatch.setattr(tracker, "_advance_temporal_with_priority", only_fit)
    before = m.nodewise.array_signature({"poses": fitted, "costs": costs, "parts": parts})
    tracker._advance_owned(frame)
    classified = tracker.export_endpoint_mask_diagnostics()
    assert classified["candidate_split_by_body"][0, 0]
    assert not classified["candidate_split_by_body"][1].any()
    instance = tracker.export_instance_diagnostics()
    np.testing.assert_array_equal(
        instance["base_local_cost"] + instance["sixth_instance_term"], result["fit"][1]
    )
    np.testing.assert_array_equal(instance["adjusted_path_cost"], 7.0 + result["fit"][1])
    assert before == m.nodewise.array_signature({"poses": fitted, "costs": costs, "parts": parts})
    assert tracker._endpoint_mask_observations is None


def test_real_sparse_migration_preserves39_and_rejects_source_damage(inputs, checkpoint):
    arrays, metadata = copy.deepcopy(checkpoint)
    original = copy.deepcopy(metadata)
    signatures = m.nodewise.array_signature(arrays)
    source, target = sparse.SparseInstanceRescoreContinuousTracker(*inputs), make(inputs)
    imported = m.import_sparse_state(target, source, arrays, metadata)
    saved, current = target.export_state()
    assert signatures == m.nodewise.array_signature(saved) == m.nodewise.array_signature(arrays)
    assert metadata == original and imported["all_arrays_byte_preserved"]
    assert target.restore_count == 7 and source.initialize_count == source.restore_count == 0
    assert target.initialize_count == 1 and target.processed_frames == 5
    current_context = copy.deepcopy(current["context"])
    for key in ("body_row_policy", "endpoint_mask_priority_policy"):
        assert current_context.pop(key)["effective_from_native"] == NATIVE
    current_context["implementation"] = metadata["context"]["implementation"]
    assert current_context == metadata["context"]
    for damage in ("review", "boundary", "config"):
        a, meta = copy.deepcopy(checkpoint)
        target = make(inputs)
        if damage == "review":
            a["state_review"][0, 2] = False
        elif damage == "boundary":
            target = make(inputs, body_row_effective_from_native=NATIVE + 1)
        else:
            target = make((*inputs[:2], replace(inputs[2], shape_weight=9.0)))
        with pytest.raises(ValueError):
            m.import_sparse_state(
                target, sparse.SparseInstanceRescoreContinuousTracker(*inputs), a, meta
            )
        assert target.initialize_count == 0


@pytest.mark.parametrize("where", ["prepare", "temporal", "boundary", "public"])
def test_failure_clears_all_diagnostics_and_ephemeral_scopes(monkeypatch, inputs, where):
    tracker = make(inputs)
    for name in (
        "last_instance_diagnostics",
        "last_body_row_diagnostics",
        "last_endpoint_mask_diagnostics",
    ):
        setattr(tracker, name, {"stale": True})

    def fail(*args):
        raise ValueError("synthetic injected failure")

    frame = SimpleNamespace(frame_idx=NATIVE, masks={}, detections={})
    if where == "public":
        monkeypatch.setattr(m.nodewise.BodyRowContinuousTracker, "advance", fail)
        call = tracker.advance
    else:
        call = tracker._advance_owned
        if where == "boundary":
            frame.frame_idx -= 1
        elif where == "prepare":
            monkeypatch.setattr(tracker, "_prepare_endpoint_masks", fail)
        else:
            monkeypatch.setattr(tracker, "_prepare_endpoint_masks", lambda frame: None)
            monkeypatch.setattr(tracker, "_advance_temporal_with_priority", fail)
    with pytest.raises(ValueError):
        call(frame)
    assert tracker.last_instance_diagnostics is tracker.last_body_row_diagnostics is None
    assert tracker.last_endpoint_mask_diagnostics is None
    assert (
        tracker._body_row_frame is tracker._instance_frame is tracker._endpoint_mask_frame is None
    )
    assert tracker._endpoint_mask_observations is None

"""Synthetic CPU hook/39-state equivalence; no optimizer, actual data or GPU."""

# ruff: noqa: E402

import copy
import hashlib
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"),
    str(Path(__file__).resolve().parents[2] / "src"),
]

import continuous_tracker as continuous
import instance_rescore_tracker as dense
import numpy as np
import pytest
import sparse_instance_body_score as distance
import sparse_instance_rescore_tracker as sparse
import test_instance_rescore_tracker as toy
from directional_device_fitter import DirectionalDeviceFitter
from fast_directional_device_fitter import FastDirectionalDeviceFitter
from observations import FrameObservation

from medaka3d.types import VIEWS

FROZEN_ADVANCE_OWNED = continuous.ContinuousPersistentNodeTracker._advance_owned


def exact(a, b):
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray)
        assert a.dtype.str == b.dtype.str and a.shape == b.shape and a.tobytes() == b.tobytes()
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            exact(x, y)
    else:
        assert type(a) is type(b) and a == b


@pytest.fixture(scope="module")
def inputs():
    return toy.inputs.__wrapped__()


@pytest.fixture(autouse=True)
def forbid_optimizer_and_public_rollout(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic tests may not optimize, begin or run a public rollout")

    for method in ("begin", "advance", "_advance_owned"):
        monkeypatch.setattr(continuous.ContinuousPersistentNodeTracker, method, forbidden)
    monkeypatch.setattr(DirectionalDeviceFitter, "_fit", forbidden)
    monkeypatch.setattr(FastDirectionalDeviceFitter, "_fit", forbidden)


@pytest.fixture(scope="module")
def checkpoint(inputs):
    # Five synthetic retained frames only, with the actual frozen Dense decoder.
    arrays, meta = copy.deepcopy(toy.checkpoint.__wrapped__(inputs))
    offset = 78019 - meta["previous_frame"]
    arrays["history_frame_idx"] += offset
    meta["previous_frame"] += offset
    meta["restore_count"] = 5
    original = dense.InstanceRescoreContinuousTracker(*inputs)
    meta["context"] = original._context()
    original._decode_state(arrays, meta)
    assert original.initialize_count == 0 and len(arrays) == 39
    return arrays, meta


def test_frozen_sources_static_functions_and_unchanged_inherited_methods(inputs):
    assert hashlib.sha256(Path(dense.__file__).read_bytes()).hexdigest() == sparse.DENSE_TRACKER_SHA
    assert (
        hashlib.sha256(Path(distance.__file__).read_bytes()).hexdigest()
        == sparse.SPARSE_DISTANCE_SHA
    )
    assert (
        sparse.prepare_sparse_instance_observation is distance.prepare_sparse_instance_observation
    )
    assert sparse.score_sparse_instance_bodies is distance.score_sparse_instance_bodies
    old, new = (
        dense.InstanceRescoreContinuousTracker(*inputs),
        sparse.SparseInstanceRescoreContinuousTracker(*inputs),
    )
    for method in (
        "_proposals",
        "begin",
        "advance",
        "restore_state",
        "export_state",
        "export_instance_diagnostics",
    ):
        assert getattr(type(old), method) is getattr(type(new), method)
    context = new._context()
    backend = context.pop("sparse_distance_backend")
    context["implementation"] = old._context()["implementation"]
    assert context == old._context()
    assert backend["effective_from_native"] == 78020
    assert backend["science_policy_unchanged"] and not backend["prior_history_rewritten"]
    assert not backend["backend_applied_retroactively_to_prior_history"]
    with pytest.raises(FrozenInstanceError):
        new.sparse_backend_policy.effective_from_native = 0


@pytest.mark.parametrize("boundary,restores", [(76360, 0), (78020, 5), (100000, 8)])
def test_arbitrary_valid_transition_preserves_all39_and_actual_decoder_lifecycle(
    inputs, checkpoint, boundary, restores
):
    arrays, meta = copy.deepcopy(checkpoint)
    arrays["history_frame_idx"] += boundary - 1 - meta["previous_frame"]
    meta["previous_frame"], meta["restore_count"] = boundary - 1, restores
    before, old_meta = sparse.array_signature(arrays), copy.deepcopy(meta)
    validator = dense.InstanceRescoreContinuousTracker(*inputs)
    new = sparse.SparseInstanceRescoreContinuousTracker(
        *inputs, sparse_effective_from_native=boundary
    )
    result = sparse.import_dense_instance_state(new, validator, arrays, meta)
    saved, metadata = new.export_state()
    assert sparse.array_signature(saved) == sparse.array_signature(arrays) == before
    assert meta == old_meta and validator.initialize_count == 0
    assert new.initialize_count == 1 and new.restore_count == restores + 1
    assert new.previous_frame == boundary - 1 and new.processed_frames == 5
    assert result["all_arrays_byte_preserved"] and result["added_restore_count"] == 1
    assert not result["backend_applied_retroactively_to_prior_history"]
    assert not result["review_cleared_by_migration"]
    assert np.all(saved["history_identity_review"][:, 2])
    expected = copy.deepcopy(meta)
    expected["context"] = new._context()
    expected["restore_count"] += 1
    assert metadata == expected
    resumed = sparse.SparseInstanceRescoreContinuousTracker(
        *inputs, sparse_effective_from_native=boundary
    )
    resumed.restore_state(saved, metadata)
    assert resumed.restore_count == restores + 2
    assert before == sparse.array_signature(resumed.export_state()[0])
    assert resumed.last_instance_diagnostics is None


@pytest.mark.parametrize("mode", ["eligible", "neutral", "invalid_projection"])
def test_real_temporal_selection_flow_ages_reviews_and39_export_match_dense(
    monkeypatch, inputs, checkpoint, mode
):
    old = dense.InstanceRescoreContinuousTracker(*inputs)
    old.restore_state(*checkpoint)
    new = sparse.SparseInstanceRescoreContinuousTracker(*inputs)
    sparse.import_dense_instance_state(
        new, dense.InstanceRescoreContinuousTracker(*inputs), *checkpoint
    )
    masks, points, poses = toy.scene()
    frame = FrameObservation(
        78020,
        {v: points.copy() for v in VIEWS},
        {v: np.zeros((100, 180, 3), np.uint8) for v in VIEWS},
        {v: masks.copy() if mode != "neutral" else np.empty((0, 100, 180), bool) for v in VIEWS},
        {v: np.ones((100, 180, 3), np.uint8) for v in VIEWS},
    )

    def project(x):
        y = x[..., :2].copy()
        if mode == "invalid_projection":
            y[0, 0, 0] = np.nan
        return y

    monkeypatch.setattr(inputs[0], "camera", lambda view: SimpleNamespace(project=project))
    plan = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).copy()
    proposals = [
        SimpleNamespace(
            parent=0, kind="predict", permutation=(0, 1, 2), pose=p.copy(), plan=plan.copy()
        )
        for p in poses
    ]
    base_cost = np.array([0.0, 0.5])
    parts = np.arange(10, dtype=float).reshape(2, 5)
    support = np.array([[False, False, True], [False, True, False]])
    bank_before = sparse.array_signature(
        dict(pose=poses, cost=base_cost, parts=parts, support=support)
    )
    observed = np.array([[True, True, False], [True, False, True], [False, True, True]])
    rays, pixels = object(), object()
    flow = SimpleNamespace(node_valid=np.zeros((3, 3), bool))
    calls = {"fit": [], "flow": [], "support": []}
    monkeypatch.setattr(continuous.RayObservations, "from_points", lambda *args: rays)
    monkeypatch.setattr(continuous, "sample_pixels", lambda *args: pixels)

    def flow_from(previous, current, rig, parent_pose):
        assert previous.frame_idx == 78019 and current is frame and rig is inputs[0]
        calls["flow"].append(previous)
        return flow

    def propose(self, states, actual_rays, actual_flow):
        assert states is self._states and actual_rays is rays and actual_flow is flow
        self._recovery_calls += 1
        return proposals

    def fit(self, actual_proposals, states, actual_rays, targets):
        assert actual_proposals is proposals and states is self._states
        assert actual_rays is rays and targets is pixels
        calls["fit"].append(self)
        self.last_diagnostics = {"frozen_fit_marker": True}
        return poses, base_cost, parts, support

    def supported(pose, actual_plan, actual_rays, rig):
        assert actual_rays is rays and rig is inputs[0]
        assert any(actual_plan is p.plan for p in proposals)
        calls["support"].append(pose.copy())
        return np.where(observed[:, None, :], actual_plan, -1), observed.copy(), None

    monkeypatch.setattr(continuous.FlowObservations, "from_frames", flow_from)
    monkeypatch.setattr(continuous, "supported_nodes", supported)
    monkeypatch.setattr(sparse.FastTailRecoveryContinuousTracker, "_proposals", propose)
    monkeypatch.setattr(sparse.FastTailRecoveryContinuousTracker, "_fit", fit)
    monkeypatch.setattr(
        sparse.FastTailRecoveryContinuousTracker, "_advance_owned", FROZEN_ADVANCE_OWNED
    )
    old._advance_owned(frame)
    new._advance_owned(frame)
    assert calls["fit"] == [old, new] and len(calls["flow"]) == 2 and len(calls["support"]) == 4
    exact(old._history, new._history)
    for a, b in zip(old._states, new._states, strict=True):
        for field in continuous.STATE_FIELDS:
            exact(getattr(a, field), getattr(b, field))
        assert a.review[2] and b.review[2]
    a, am = old.export_state()
    b, bm = new.export_state()
    assert len(a) == len(b) == 39 and sparse.array_signature(a) == sparse.array_signature(b)
    stripped = copy.deepcopy(bm)
    backend = stripped["context"].pop("sparse_distance_backend")
    stripped["context"]["implementation"] = am["context"]["implementation"]
    assert stripped == am and backend["effective_from_native"] == 78020
    assert old.previous_frame == new.previous_frame == 78020
    assert old.restore_count == new.restore_count == 6
    assert old.processed_frames == new.processed_frames == 6
    assert (
        sparse.array_signature(dict(pose=poses, cost=base_cost, parts=parts, support=support))
        == bank_before
    )
    da, db = old.export_instance_diagnostics(), new.export_instance_diagnostics()
    assert db.pop("distance_backend") == bm["context"]["sparse_distance_backend"]
    for key in (
        "gate_preparation_seconds_by_view",
        "gate_preparation_seconds",
        "candidate_scoring_seconds_by_view",
        "candidate_scoring_seconds",
    ):
        da.pop(key)
        db.pop(key)
    exact(da, db)
    assert new._instance_frame is new._instance_observations is None


@pytest.mark.parametrize(
    "damage", ["boundary", "context", "age", "review", "source", "missing", "framegap"]
)
def test_transition_rejects_malformed_source_without_initializing_target(
    inputs, checkpoint, damage
):
    arrays, meta = copy.deepcopy(checkpoint)
    if damage == "boundary":
        meta["previous_frame"] -= 1
    elif damage == "context":
        meta["context"]["implementation"] = "ForgedDense"
    elif damage == "age":
        arrays["state_node_missing"][0, 0, 2] = 0
    elif damage == "review":
        arrays["state_review"][0, 2] = False
    elif damage == "source":
        arrays["history_sources"][0, 1, :, 0] = arrays["history_sources"][0, 0, :, 0]
    elif damage == "missing":
        arrays.pop("previous_images_bgr_" + VIEWS[0])
    else:
        arrays["history_frame_idx"][0] -= 1
    target = sparse.SparseInstanceRescoreContinuousTracker(*inputs)
    with pytest.raises(ValueError):
        sparse.import_dense_instance_state(
            target, dense.InstanceRescoreContinuousTracker(*inputs), arrays, meta
        )
    assert target.initialize_count == 0


def test_science_policy_drift_and_wrong_implementation_are_rejected(inputs, checkpoint):
    old = dense.InstanceRescoreContinuousTracker(*inputs)
    target = sparse.SparseInstanceRescoreContinuousTracker(
        *inputs[:2], replace(inputs[2], mask_weight=0.7)
    )
    with pytest.raises(ValueError, match="cannot change"):
        sparse.import_dense_instance_state(target, old, *checkpoint)
    with pytest.raises(ValueError, match="implementations"):
        sparse.import_dense_instance_state(
            target, sparse.FastTailRecoveryContinuousTracker(*inputs), *checkpoint
        )


@pytest.mark.parametrize("boundary", [True, -1, 78020.0])
def test_bad_backend_boundary_is_rejected(inputs, boundary):
    with pytest.raises(ValueError, match="transition frame"):
        sparse.SparseInstanceRescoreContinuousTracker(
            *inputs, sparse_effective_from_native=boundary
        )


def test_frame_before_declared_backend_is_rejected(inputs):
    tracker = sparse.SparseInstanceRescoreContinuousTracker(*inputs)
    with pytest.raises(ValueError, match="precedes"):
        tracker._advance_owned(SimpleNamespace(frame_idx=78019))
    assert tracker._instance_frame is tracker._instance_observations is None


@pytest.mark.parametrize("stage", ["prepare", "after_fit"])
def test_failed_owned_frame_clears_ephemeral_gates_and_companion(monkeypatch, inputs, stage):
    tracker = sparse.SparseInstanceRescoreContinuousTracker(*inputs)
    tracker.last_instance_diagnostics = {"frame_idx": 78019, "stale": True}
    frame = SimpleNamespace(frame_idx=78020, masks={}, detections={})

    def fail_prepare(*args):
        raise RuntimeError("synthetic gate failure")

    def fail_after_fit(self, current):
        assert current is frame and self._instance_frame == frame.frame_idx
        assert set(self._instance_observations) == set(VIEWS)
        self.last_instance_diagnostics = {"frame_idx": 78020, "fit_only": True}
        raise RuntimeError("synthetic post-fit failure")

    if stage == "prepare":
        monkeypatch.setattr(sparse, "prepare_sparse_instance_observation", fail_prepare)
    else:
        monkeypatch.setattr(
            sparse.FastTailRecoveryContinuousTracker, "_advance_owned", fail_after_fit
        )
    with pytest.raises(RuntimeError, match="synthetic"):
        tracker._advance_owned(frame)
    assert tracker.last_instance_diagnostics is None
    assert tracker._instance_frame is None
    assert tracker._instance_observations is None
    assert tracker._instance_gate_seconds is None

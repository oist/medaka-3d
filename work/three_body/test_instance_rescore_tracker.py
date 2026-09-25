"""CPU synthetic rescoring and real frozen 39-array decoder; no fitting/rollout."""

# ruff: noqa: E402

import copy
import json
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"),
    str(Path(__file__).resolve().parents[2] / "src"),
]

import continuous_tracker as continuous
import instance_rescore_tracker as m
import numpy as np
import pytest
import test_tail_recovery_tracker as toy
from continuous_tracker import ContinuousPersistentNodeTracker
from directional_device_fitter import DirectionalDeviceFitter
from fast_directional_device_fitter import FastDirectionalDeviceFitter
from observations import FrameObservation
from persistent_node_tracker import PersistentNodeState

from medaka3d.types import VIEWS

FROZEN_ADVANCE_OWNED = ContinuousPersistentNodeTracker._advance_owned


@pytest.fixture(scope="module")
def inputs():
    return toy.inputs.__wrapped__()


@pytest.fixture(autouse=True)
def forbid_fitting_or_rollout(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("this test must not begin, advance, execute a temporal step or optimize")

    for method in ("begin", "advance", "_advance_owned"):
        monkeypatch.setattr(ContinuousPersistentNodeTracker, method, forbidden)
    monkeypatch.setattr(DirectionalDeviceFitter, "_fit", forbidden)
    monkeypatch.setattr(FastDirectionalDeviceFitter, "_fit", forbidden)


def scene():
    masks = np.zeros((3, 100, 180), dtype=bool)
    points = []
    for k, x in enumerate((10, 70, 130)):
        masks[k, 30:46, x : x + 30] = True
        points.append([[x + 4, 35], [x + 15, 37], [x + 25, 40]])
    points = np.asarray(points, dtype=float)
    good = np.concatenate([points, np.ones((3, 3, 1))], axis=-1)
    mixed = good.copy()
    mixed[2, :2] = good[0, :2]
    return masks, points, np.stack([mixed, good])


def setup_scoring(monkeypatch, inputs, *, available=VIEWS, base_cost=(0.0, 0.5)):
    tracker = m.InstanceRescoreContinuousTracker(*inputs)
    masks, points, poses = scene()
    # Synthetic projection only; never CUDA, camera files or actual images.
    monkeypatch.setattr(
        tracker.rig, "camera", lambda view: SimpleNamespace(project=lambda x: x[..., :2])
    )
    frame = FrameObservation(
        m.POLICY.effective_from_native,
        {v: points.copy() for v in VIEWS},
        {v: np.zeros((100, 180, 3), np.uint8) for v in VIEWS},
        {v: masks.copy() if v in available else None for v in VIEWS},
        {v: np.ones((100, 180, 3), np.uint8) for v in VIEWS},
    )
    proposals = [
        SimpleNamespace(
            parent=0,
            kind="predict",
            permutation=(0, 1, 2),
            pose=p.copy(),
            plan=np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).copy(),
        )
        for p in poses
    ]
    states = [SimpleNamespace(cost=7.0, review=np.array([False, True, True]))]
    parts = np.arange(10, dtype=float).reshape(2, 5)
    support = np.array([[True, False, True], [False, True, False]])
    costs = np.asarray(base_cost, dtype=float)
    original = (poses, costs, parts, support)
    rays, targets = object(), object()
    fit_calls, parent_calls = [], []
    marker = {"original_fitter_diagnostic": True}

    def frozen_fit(self, *args):
        assert self is tracker
        assert all(a is b for a, b in zip(args, (proposals, states, rays, targets), strict=True))
        fit_calls.append(args)
        self.last_diagnostics = marker
        return original

    def frozen_parent(self, owned_frame):
        assert owned_frame is frame
        parent_calls.append(owned_frame)
        return self._fit(proposals, states, rays, targets)

    monkeypatch.setattr(m.FastTailRecoveryContinuousTracker, "_fit", frozen_fit)
    monkeypatch.setattr(m.FastTailRecoveryContinuousTracker, "_advance_owned", frozen_parent)
    return SimpleNamespace(
        tracker=tracker,
        frame=frame,
        original=original,
        proposals=proposals,
        states=states,
        fit_calls=fit_calls,
        parent_calls=parent_calls,
        marker=marker,
    )


def test_fixed_policy_context_and_original_inheritance(inputs):
    tracker = m.InstanceRescoreContinuousTracker(*inputs)
    original = m.FastTailRecoveryContinuousTracker(*inputs)
    new = tracker._context()
    assert json.loads(json.dumps(new)) == new
    policy = new.pop("instance_rescore_policy")
    new["implementation"] = original._context()["implementation"]
    assert new == original._context()
    assert policy["effective_from_native"] == 76360
    assert policy["cap_per_view"] == 9 and policy["weight"] == inputs[2].mask_weight
    assert policy["bodies_per_view"] == policy["fixed_camera_denominator"] == 3
    assert policy["post_fit_only"] and policy["gradient_objective_unchanged"]
    assert policy["companion_diagnostics_required_each_frame"]
    assert not policy["five_parts_reconstruct_new_local_cost"]
    assert not policy["calibrated_equivalence_to_original_mask_loss"]
    assert not policy["clears_identity_review"]
    for method in ("_proposals", "advance", "begin", "restore_state", "export_state"):
        assert getattr(type(tracker), method) is getattr(type(original), method)
    with pytest.raises(FrozenInstanceError):
        m.POLICY.cap_per_view = 42
    with pytest.raises(TypeError):
        m.InstanceRescorePolicy(cap_per_view=42)


@pytest.mark.parametrize("weight", [-1.0, np.nan, np.inf])
def test_invalid_weight_is_rejected(inputs, weight):
    with pytest.raises(ValueError, match="weight"):
        m.InstanceRescoreContinuousTracker(*inputs[:2], replace(inputs[2], mask_weight=weight))


def test_mixed_fish_candidate_loses_without_changing_original_fit_outputs(monkeypatch, inputs):
    s = setup_scoring(monkeypatch, inputs)
    before = m.array_signature({str(k): a for k, a in enumerate(s.original)})
    proposals_before = np.stack([p.pose for p in s.proposals]).copy()
    review_before = s.states[0].review.copy()
    result = s.tracker._advance_owned(s.frame)
    assert len(s.fit_calls) == len(s.parent_calls) == 1
    assert np.argmin(s.original[1]) == 0 and np.argmin(result[1]) == 1
    for index in (0, 2, 3):
        assert result[index] is s.original[index]
    assert before == m.array_signature({str(k): a for k, a in enumerate(s.original)})
    np.testing.assert_array_equal(np.stack([p.pose for p in s.proposals]), proposals_before)
    np.testing.assert_array_equal(s.states[0].review, review_before)
    assert s.tracker.last_diagnostics is s.marker
    d = s.tracker.export_instance_diagnostics()
    np.testing.assert_array_equal(d["base_local_cost"] + d["sixth_instance_term"], result[1])
    np.testing.assert_array_equal(d["adjusted_path_cost"], result[1] + 7.0)
    np.testing.assert_array_equal(d["original_five_loss_parts"], s.original[2])
    assert d["sixth_instance_term"][1] == 0
    for view in VIEWS:
        v = d["views"][view]
        assert v["eligible"] and v["raw_uncapped_scene_cost"][0] > 0
        assert v["raw_uncapped_scene_cost"][1] == 0
        assert not v["observation_evidence"]["candidate_pose_used_for_eligibility"]
        np.testing.assert_array_equal(v["best_assignment"][1], [0, 1, 2])
    assert d["gate_preparation_seconds"] >= 0 and d["candidate_scoring_seconds"] >= 0
    assert not d["observed_flags_used"] and not d["clears_identity_review"]
    assert s.tracker._instance_frame is s.tracker._instance_observations is None
    d["sixth_instance_term"][:] = 999
    assert np.all(s.tracker.export_instance_diagnostics()["sixth_instance_term"] < 999)


def test_real_inherited_selection_changes_rank_and_keeps_support_ages_sticky_review(
    monkeypatch,
    inputs,
):
    """One synthetic selection step per arm; no begin/advance/optimizer or real data."""
    s = setup_scoring(monkeypatch, inputs)
    base = m.FastTailRecoveryContinuousTracker(*inputs)
    parent = PersistentNodeState(
        s.original[0][1].copy(),
        np.zeros((3, 3)),
        np.array([0, 2, 0]),
        np.array([False, True, True]),
        7.0,
        np.full((3, 3), 2),
        np.full(3, 3),
    )
    parent_before = copy.deepcopy(parent)
    for tracker in (base, s.tracker):
        tracker._states = [copy.deepcopy(parent)]
    sentinel_rays, sentinel_targets = object(), object()
    monkeypatch.setattr(continuous.RayObservations, "from_points", lambda *args: sentinel_rays)
    monkeypatch.setattr(continuous, "sample_pixels", lambda *args: sentinel_targets)
    monkeypatch.setattr(
        m.FastTailRecoveryContinuousTracker, "_proposals", lambda *args: s.proposals
    )
    support_calls = []
    observed = np.array([[True, True, False], [True, False, True], [False, True, True]])

    def support(pose, plan, rays, rig):
        assert rays is sentinel_rays and rig is inputs[0]
        candidate = next(i for i, p in enumerate(s.proposals) if plan is p.plan)
        assert pose is not s.proposals[candidate].pose
        np.testing.assert_array_equal(pose, s.original[0][candidate])
        sources = np.where(observed[:, None, :], plan, -1)
        support_calls.append(candidate)
        return sources, observed.copy(), None

    monkeypatch.setattr(continuous, "supported_nodes", support)

    def fitted(self, proposals, states, rays, targets):
        assert proposals is s.proposals and states is self._states
        assert rays is sentinel_rays and targets is sentinel_targets
        return s.original

    monkeypatch.setattr(m.FastTailRecoveryContinuousTracker, "_fit", fitted)
    monkeypatch.setattr(m.FastTailRecoveryContinuousTracker, "_advance_owned", FROZEN_ADVANCE_OWNED)
    FROZEN_ADVANCE_OWNED(base, s.frame)
    s.tracker._advance_owned(s.frame)
    assert support_calls == [0, 1, 1, 0]
    baseline, rescored = base._history[0], s.tracker._history[0]
    np.testing.assert_array_equal(baseline[0]["pose_xyz"], s.original[0][0])
    np.testing.assert_array_equal(rescored[0]["pose_xyz"], s.original[0][1])
    # Match the same candidate in both arms: support and age computation stay inherited.
    for old, new in ((baseline[1], rescored[0]), (baseline[0], rescored[1])):
        for field in (
            "pose_xyz",
            "sources",
            "observed",
            "mask_supported",
            "body_supported",
            "missing",
            "node_missing",
            "direct_pectoral_age",
            "loss_parts",
        ):
            np.testing.assert_array_equal(old[field], new[field])
        assert np.all(new["identity_review"][parent.review])
    for old, new in (
        (base._states[1], s.tracker._states[0]),
        (base._states[0], s.tracker._states[1]),
    ):
        np.testing.assert_array_equal(old.velocity, new.velocity)
    for name in continuous.STATE_FIELDS:
        np.testing.assert_array_equal(getattr(parent, name), getattr(parent_before, name))
    d = s.tracker.export_instance_diagnostics()
    np.testing.assert_array_equal(d["candidate_plan"], np.stack([p.plan for p in s.proposals]))
    for record in rescored:
        candidate = next(
            i for i, p in enumerate(s.original[0]) if np.array_equal(p, record["pose_xyz"])
        )
        assert record["costs"] == d["adjusted_local_cost"][candidate]
        assert record["path_costs"] == d["adjusted_path_cost"][candidate]
    assert s.tracker.previous_frame == s.frame.frame_idx


@pytest.mark.parametrize("available", [(), VIEWS[:1], VIEWS[:2], VIEWS])
def test_fixed_camera_denominator_no_rescaling_missing_views(monkeypatch, inputs, available):
    s = setup_scoring(monkeypatch, inputs, available=available)
    result = s.tracker._advance_owned(s.frame)
    d = s.tracker.export_instance_diagnostics()
    expected = np.zeros(2)
    for view in VIEWS:
        v = d["views"][view]
        if view in available:
            expected += np.minimum(v["raw_uncapped_scene_cost"] / 3, 9)
        else:
            assert not v["eligible"] and v["neutral"]
            assert v["reason"] == "mask_view_unavailable"
            assert v["best_assignment"] is None
    expected *= inputs[2].mask_weight / 3
    np.testing.assert_allclose(result[1], s.original[1] + expected, rtol=0, atol=1e-14)
    if not available:
        np.testing.assert_array_equal(result[1], s.original[1])


def test_cap_retains_raw_severity_ties_and_invalid_projections(monkeypatch, inputs):
    s = setup_scoring(monkeypatch, inputs)
    s.original[0][1] += 1000
    s.original[0][0] += 10000
    s.tracker._advance_owned(s.frame)
    d = s.tracker.export_instance_diagnostics()
    np.testing.assert_array_equal(d["sixth_instance_term"], np.full(2, 9 * inputs[2].mask_weight))
    for v in d["views"].values():
        assert np.all(v["cap_applied"])
        assert v["raw_uncapped_scene_cost"][0] > v["raw_uncapped_scene_cost"][1] > 27
        assert v["assignment_costs"].shape == v["optimal_assignment_mask"].shape == (2, 6)

    # Nonfinite projection in one camera remains eligible with a finite penalty.
    def good_project(x):
        return x[..., :2]

    def invalid_project(x):
        out = x[..., :2].copy()
        out[0, 0, 0] = np.nan
        return out

    monkeypatch.setattr(
        s.tracker.rig,
        "camera",
        lambda view: SimpleNamespace(project=invalid_project if view == VIEWS[0] else good_project),
    )
    actual = s.tracker._advance_owned(s.frame)
    d = s.tracker.export_instance_diagnostics()
    invalid = d["views"][VIEWS[0]]
    assert invalid["eligible"] and invalid["nonfinite_score"][0]
    assert np.isinf(invalid["raw_uncapped_scene_cost"][0])
    assert invalid["capped_view_contribution"][0] == 9
    np.testing.assert_array_equal(invalid["best_assignment"][0], [-1, -1, -1])
    assert np.isfinite(actual[1]).all()


def test_gate_prepared_once_per_owned_frame_and_never_from_previous(monkeypatch, inputs):
    s = setup_scoring(monkeypatch, inputs)
    s.tracker._previous_observation = object()
    prepare = m.prepare_instance_observation
    calls = []

    def recording(masks, detections, config):
        calls.append((masks, detections))
        return prepare(masks, detections, config)

    monkeypatch.setattr(m, "prepare_instance_observation", recording)
    s.tracker._advance_owned(s.frame)
    assert len(calls) == 3
    for pair, view in zip(calls, VIEWS, strict=True):
        assert pair[0] is s.frame.masks[view] and pair[1] is s.frame.detections[view]
    # Different current masks close gates, even though the previous fit was eligible.
    for view in VIEWS:
        s.frame.masks[view] = None
    s.tracker._advance_owned(s.frame)
    assert len(calls) == 6
    assert all(v["neutral"] for v in s.tracker.export_instance_diagnostics()["views"].values())


@pytest.mark.parametrize("stage", ["prepare", "fit", "score", "after_fit"])
def test_ephemeral_scope_and_stale_diagnostics_cleared_on_failure(monkeypatch, inputs, stage):
    s = setup_scoring(monkeypatch, inputs)
    s.tracker._advance_owned(s.frame)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    if stage == "prepare":
        monkeypatch.setattr(m, "prepare_instance_observation", fail)
    elif stage == "fit":
        monkeypatch.setattr(m.FastTailRecoveryContinuousTracker, "_fit", fail)
    elif stage == "score":
        monkeypatch.setattr(m, "score_instance_bodies", fail)
    else:
        original_parent = m.FastTailRecoveryContinuousTracker._advance_owned

        def fail_after_fit(self, frame):
            original_parent(self, frame)
            assert self.last_instance_diagnostics is not None
            fail()

        monkeypatch.setattr(m.FastTailRecoveryContinuousTracker, "_advance_owned", fail_after_fit)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        s.tracker._advance_owned(s.frame)
    assert s.tracker._instance_frame is s.tracker._instance_observations is None
    assert s.tracker._instance_gate_seconds is None
    with pytest.raises(RuntimeError, match="no completed fit"):
        s.tracker.export_instance_diagnostics()


def test_before_policy_is_neutral_and_direct_fit_cannot_use_stale_scope(monkeypatch, inputs):
    s = setup_scoring(monkeypatch, inputs)
    object.__setattr__(s.frame, "frame_idx", m.POLICY.effective_from_native - 1)

    def forbidden(*args, **kwargs):
        pytest.fail("pre-policy frame must not prepare or score instance observations")

    monkeypatch.setattr(m, "prepare_instance_observation", forbidden)
    monkeypatch.setattr(m, "score_instance_bodies", forbidden)
    result = s.tracker._advance_owned(s.frame)
    np.testing.assert_array_equal(result[1], s.original[1])
    with pytest.raises(RuntimeError, match="current owned frame"):
        s.tracker._fit(None, None, None, None)
    assert all(
        v["reason"] == "policy_not_yet_effective"
        for v in s.tracker.export_instance_diagnostics()["views"].values()
    )


@pytest.fixture(scope="module")
def checkpoint(inputs):
    arrays, metadata = copy.deepcopy(toy.checkpoint.__wrapped__(inputs))
    original = m.FastTailRecoveryContinuousTracker(*inputs)
    delta = m.POLICY.effective_from_native - 1 - metadata["previous_frame"]
    arrays["history_frame_idx"] += delta
    metadata["previous_frame"] += delta
    metadata["restore_count"] = 4
    metadata["context"] = original._context()
    original._decode_state(arrays, metadata)
    assert len(arrays) == 39
    return arrays, metadata


@pytest.mark.parametrize("restores", [0, 4, 8])
def test_all39_real_decoder_migration_preserves_source_history_reviews_and_counts(
    inputs,
    checkpoint,
    restores,
):
    arrays, metadata = copy.deepcopy(checkpoint)
    metadata["restore_count"] = restores
    original = m.FastTailRecoveryContinuousTracker(*inputs)
    target = m.InstanceRescoreContinuousTracker(*inputs)
    before = m.array_signature(arrays)
    before_metadata = copy.deepcopy(metadata)
    report = m.import_fast_tail_state(target, original, arrays, metadata)
    actual, meta = target.export_state()
    assert len(actual) == 39 and before == m.array_signature(actual) == m.array_signature(arrays)
    assert metadata == before_metadata
    assert target.initialize_count == 1 and target.restore_count == restores + 1
    assert target.previous_frame == 76359 and target.processed_frames == 5
    assert original.initialize_count == original.restore_count == original.processed_frames == 0
    assert report["added_restore_count"] == 1 and report["new_begin_calls"] == 0
    assert report["all_arrays_byte_preserved"] and not report["review_cleared_by_migration"]
    assert report["effective_from_native"] == 76360
    assert np.all(actual["state_review"][:, 2])
    assert np.all(actual["history_identity_review"][:, 2])
    assert np.any(actual["history_sources"] == -1)
    assert np.any(~actual["history_observed"])
    assert target._instance_frame is target._instance_observations is None
    assert target.last_instance_diagnostics is None
    expected = copy.deepcopy(metadata)
    expected["context"] = target._context()
    expected["restore_count"] += 1
    assert meta == expected
    resumed = m.InstanceRescoreContinuousTracker(*inputs)
    resumed.restore_state(actual, meta)
    assert resumed.restore_count == restores + 2
    assert before == m.array_signature(resumed.export_state()[0])
    with pytest.raises(ValueError, match="context"):
        original.restore_state(actual, meta)


@pytest.mark.parametrize(
    "damage",
    [
        "metadata_context",
        "wrong_boundary",
        "missing_array",
        "extra_array",
        "active_review",
        "history_review",
        "active_age",
        "source_rows",
        "frame_gap",
        "processed_count",
        "init_count",
    ],
)
def test_migration_rejects_tampering_before_target_initialization(inputs, checkpoint, damage):
    arrays, metadata = copy.deepcopy(checkpoint)
    if damage == "metadata_context":
        metadata["context"]["implementation"] = "ForgedFastTail"
    elif damage == "wrong_boundary":
        metadata["previous_frame"] -= 1
    elif damage == "missing_array":
        arrays.pop("previous_images_bgr_" + VIEWS[0])
    elif damage == "extra_array":
        arrays["extra"] = np.zeros(1)
    elif damage == "active_review":
        arrays["state_review"][0, 2] = False
    elif damage == "history_review":
        arrays["history_identity_review"][4, 2] = False
    elif damage == "active_age":
        arrays["state_node_missing"][0, 0, 2] = 0
    elif damage == "source_rows":
        arrays["history_sources"][0, 1, :, 0] = arrays["history_sources"][0, 0, :, 0]
    elif damage == "frame_gap":
        arrays["history_frame_idx"][0] -= 1
    elif damage == "processed_count":
        metadata["processed_frames"] += 1
    elif damage == "init_count":
        metadata["initialize_count"] = 2
    original, target = (
        m.FastTailRecoveryContinuousTracker(*inputs),
        m.InstanceRescoreContinuousTracker(*inputs),
    )
    with pytest.raises(ValueError):
        m.import_fast_tail_state(target, original, arrays, metadata)
    assert target.initialize_count == original.initialize_count == 0


@pytest.mark.parametrize("drift", ["motion_weight", "fast_fitter", "tail_policy"])
def test_migration_rejects_existing_context_drift(inputs, checkpoint, drift):
    original = m.FastTailRecoveryContinuousTracker(*inputs)
    args, kwargs = inputs, {}
    if drift == "motion_weight":
        args = (*inputs[:2], replace(inputs[2], motion_weight=0.987))
    elif drift == "fast_fitter":
        kwargs["fast_fitter_enabled"] = False
    else:
        kwargs["tail_recovery_enabled"] = False
    target = m.InstanceRescoreContinuousTracker(*args, **kwargs)
    with pytest.raises(ValueError, match="cannot change"):
        m.import_fast_tail_state(target, original, *checkpoint)
    assert target.initialize_count == 0

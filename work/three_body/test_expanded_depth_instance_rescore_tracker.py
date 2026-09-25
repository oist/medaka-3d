"""CPU synthetic depth sensitivity and frozen-state migration; no real rollout."""

# ruff: noqa: E402

import copy
import hashlib
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"),
    str(Path(__file__).resolve().parents[2] / "src"),
]

import continuous_tracker as continuous
import expanded_depth_instance_rescore_tracker as expanded
import fast_tail_recovery_tracker as fast
import numpy as np
import pytest
import sparse_instance_rescore_tracker as sparse
import test_instance_rescore_tracker as toy
from directional_device_fitter import DirectionalDeviceFitter
from fast_directional_device_fitter import FastDirectionalDeviceFitter
from node_geometry import RayObservations, supported_nodes
from node_recovery import node_recovery_proposals
from test_sparse_instance_rescore_tracker import exact

from medaka3d.types import VIEWS


@pytest.fixture(scope="module")
def inputs():
    return toy.inputs.__wrapped__()


@pytest.fixture(scope="module")
def checkpoint(inputs):
    # Five synthetic frames, with source restore 4 and previous native 76359.
    return toy.checkpoint.__wrapped__(inputs)


@pytest.fixture(autouse=True)
def forbid_optimization_and_public_rollout(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("this test may not optimize or run an actual rollout")

    for method in ("begin", "advance", "_advance_owned"):
        monkeypatch.setattr(continuous.ContinuousPersistentNodeTracker, method, forbidden)
    monkeypatch.setattr(DirectionalDeviceFitter, "_fit", forbidden)
    monkeypatch.setattr(FastDirectionalDeviceFitter, "_fit", forbidden)


def test_frozen_sources_constructor_finalization_and_only_declared_context_changes(inputs):
    for module, expected in (
        (sparse, expanded.SPARSE_TRACKER_SHA),
        (fast, expanded.FAST_TAIL_SHA),
    ):
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected
    old = sparse.SparseInstanceRescoreContinuousTracker(*inputs, sparse_effective_from_native=76360)
    new = expanded.ExpandedDepthInstanceRescoreContinuousTracker(*inputs)
    assert new.initialize_count == 0 and new.processed_frames == 0 and new._states is None
    np.testing.assert_array_equal(new.lower, [0, 0, -12])
    np.testing.assert_array_equal(old.lower, [0, 0, -6])
    np.testing.assert_array_equal(new.upper, old.upper)
    assert tuple(new.lower) == new.recovery_config.lower_mm
    old_recovery, new_recovery = asdict(old.recovery_config), asdict(new.recovery_config)
    new_recovery["lower_mm"] = old_recovery["lower_mm"]
    assert old_recovery == new_recovery
    context = new._context()
    assert context == new._initial_context
    policy = context.pop("expanded_depth_policy")
    old_context = old._context()
    context["implementation"] = old_context["implementation"]
    context["lower"] = old_context["lower"]
    context["recovery_config"]["lower_mm"][2] = -6.0
    assert context == old_context
    assert (
        policy["effective_from_native"] == new.sparse_backend_policy.effective_from_native == 76360
    )
    assert policy["scientific_search_domain_changed"] and policy["development_diagnostics_reused"]
    assert not policy["depth_policy_applied_retroactively_to_prior_history"]
    assert not policy["measured_floor_or_physical_depth_truth_claimed"]
    for method in (
        "_proposals",
        "_fit",
        "_advance_owned",
        "advance",
        "restore_state",
        "export_state",
    ):
        assert getattr(type(new), method) is getattr(type(old), method)
    with pytest.raises(FrozenInstanceError):
        expanded.DEPTH_POLICY.expanded_lower_z_mm = -20


def test_all39_source_decoder_restore4_to5_review_history_and_resume(inputs, checkpoint):
    arrays, metadata = copy.deepcopy(checkpoint)
    before, old_meta = expanded.array_signature(arrays), copy.deepcopy(metadata)
    old, new = (
        fast.FastTailRecoveryContinuousTracker(*inputs),
        expanded.ExpandedDepthInstanceRescoreContinuousTracker(*inputs),
    )
    report = expanded.import_fast_tail_state(new, old, arrays, metadata)
    restored, meta = new.export_state()
    assert len(restored) == 39 and expanded.array_signature(restored) == before
    assert expanded.array_signature(arrays) == before and metadata == old_meta
    assert old.initialize_count == 0 and new.initialize_count == 1 and new.restore_count == 5
    assert new.processed_frames == 5 and new.previous_frame == 76359
    expected = copy.deepcopy(metadata)
    expected["context"] = new._context()
    expected["restore_count"] = 5
    assert meta == expected and report["all_arrays_byte_preserved"]
    assert not report["review_cleared_by_migration"] and report["new_begin_calls"] == 0
    assert np.all(restored["state_review"][:, 2])
    assert np.all(restored["history_identity_review"][:, 2])
    resumed = expanded.ExpandedDepthInstanceRescoreContinuousTracker(*inputs)
    resumed.restore_state(restored, meta)
    assert resumed.restore_count == 6
    assert before == expanded.array_signature(resumed.export_state()[0])
    with pytest.raises(ValueError, match="context"):
        old.restore_state(restored, meta)


@pytest.mark.parametrize(
    "damage", ["old_z", "review", "age", "source", "framegap", "missing", "boundary", "context"]
)
def test_source_tampering_rejected_before_target_initialization(inputs, checkpoint, damage):
    arrays, metadata = copy.deepcopy(checkpoint)
    if damage == "old_z":
        # New domain accepts -9, but the old source decoder must reject this history.
        arrays["history_pose_xyz"][0, 0, 0, 2] = -9.0
    elif damage == "review":
        arrays["state_review"][0, 2] = False
    elif damage == "age":
        arrays["state_node_missing"][0, 0, 2] = 0
    elif damage == "source":
        arrays["history_sources"][0, 1, :, 0] = arrays["history_sources"][0, 0, :, 0]
    elif damage == "framegap":
        arrays["history_frame_idx"][0] -= 1
    elif damage == "missing":
        arrays.pop("previous_images_bgr_" + VIEWS[0])
    elif damage == "boundary":
        metadata["previous_frame"] -= 1
    else:
        metadata["context"]["implementation"] = "forged"
    old, new = (
        fast.FastTailRecoveryContinuousTracker(*inputs),
        expanded.ExpandedDepthInstanceRescoreContinuousTracker(*inputs),
    )
    with pytest.raises(ValueError):
        expanded.import_fast_tail_state(new, old, arrays, metadata)
    assert old.initialize_count == new.initialize_count == 0


@pytest.mark.parametrize("drift", ["lower_x", "upper_z", "ray_rms", "mask_weight", "fast_fitter"])
def test_migration_rejects_any_undeclared_configuration_change(inputs, checkpoint, drift):
    old = fast.FastTailRecoveryContinuousTracker(*inputs)
    args, kwargs = inputs, {}
    if drift == "ray_rms":
        kwargs["recovery_config"] = replace(old.recovery_config, ray_rms_mm=2.0)
    elif drift == "mask_weight":
        args = (*inputs[:2], replace(inputs[2], mask_weight=0.7))
    elif drift == "fast_fitter":
        kwargs["fast_fitter_enabled"] = False
    new = expanded.ExpandedDepthInstanceRescoreContinuousTracker(*args, **kwargs)
    if drift == "lower_x":
        new.lower[0] = -1
    elif drift == "upper_z":
        new.upper[2] = 187
    with pytest.raises(ValueError):
        expanded.import_fast_tail_state(new, old, *checkpoint)
    assert new.initialize_count == 0


def test_existing_guard_rejects_operational_configuration_mutation(inputs, checkpoint):
    new = expanded.ExpandedDepthInstanceRescoreContinuousTracker(*inputs)
    expanded.import_fast_tail_state(
        new, fast.FastTailRecoveryContinuousTracker(*inputs), *checkpoint
    )
    new.lower[2] = -13
    with pytest.raises(ValueError, match=r"config|context"):
        new.export_state()


@pytest.mark.parametrize("boundary", [78020, True, 76360.0])
def test_boundary_is_fixed_not_a_hidden_parameter(inputs, boundary):
    with pytest.raises(ValueError, match="76360"):
        expanded.ExpandedDepthInstanceRescoreContinuousTracker(
            *inputs, sparse_effective_from_native=boundary
        )


@pytest.mark.parametrize("mouth_z,admitted", [(-9.0, True), (-12.01, False)])
def test_real_recovery_geometry_changes_only_depth_admission(inputs, mouth_z, admitted):
    rig, initial, _ = inputs
    old, new = (
        fast.FastTailRecoveryContinuousTracker(*inputs),
        expanded.ExpandedDepthInstanceRescoreContinuousTracker(*inputs),
    )
    current = initial.copy()
    current[1] = [[95, 90, 0], [90, 90, 0], [78, 90, 0]]
    actual = current.copy()
    actual[1, 0, 2] = mouth_z
    points = {view: rig.camera(view).project(actual) for view in VIEWS}
    rays = RayObservations.from_points(rig, points)
    plan = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).copy()
    plan[1, :, 0] = -1
    before = expanded.array_signature(dict(current=current, plan=plan))
    baseline = node_recovery_proposals(current, plan, rays, rig, old.lengths, old.recovery_config)
    result = node_recovery_proposals(current, plan, rays, rig, new.lengths, new.recovery_config)
    assert not baseline.proposals
    assert baseline.diagnostics["fish"][1]["rejection_counts"]["outside_box"] > 0
    assert bool(result.proposals) is admitted
    for proposal in result.proposals:
        assert proposal.fish == 1 and proposal.node == 0 and proposal.ray_rms_mm <= 1.5
        assert 0.5 <= proposal.length_ratio <= 1.5
        np.testing.assert_allclose(proposal.pose_xyz[1, 0], actual[1, 0], rtol=0, atol=1e-12)
        sources, observed, _ = supported_nodes(proposal.pose_xyz, proposal.plan, rays, rig)
        assert observed[1, :2].all() and (sources[1, :, 0] >= 0).sum() >= 2
        np.testing.assert_array_equal(proposal.pose_xyz[[0, 2]], current[[0, 2]])
        np.testing.assert_array_equal(proposal.pose_xyz[1, 1:], current[1, 1:])
        np.testing.assert_array_equal(proposal.plan[[0, 2]], plan[[0, 2]])
        np.testing.assert_array_equal(proposal.plan[1, :, 1:], plan[1, :, 1:])
    assert before == expanded.array_signature(dict(current=current, plan=plan))


def test_same_fitted_pool_retains_mask_costs_rank_and_original_objects(monkeypatch, inputs):
    setup = toy.setup_scoring(monkeypatch, inputs)
    baseline = setup.tracker._advance_owned(setup.frame)
    old = setup.tracker.export_instance_diagnostics()
    new = expanded.ExpandedDepthInstanceRescoreContinuousTracker(*inputs)

    def same_fitted_pool(self, proposals, states, rays, targets):
        assert self is new and proposals is setup.proposals and states is setup.states
        return setup.original

    monkeypatch.setattr(fast.FastTailRecoveryContinuousTracker, "_fit", same_fitted_pool)
    result = new._advance_owned(setup.frame)
    for index in (0, 2, 3):
        assert result[index] is setup.original[index]
    exact(baseline, result)
    diagnostic = new.export_instance_diagnostics()
    policy = diagnostic.pop("expanded_depth_policy")
    assert policy["expanded_lower_z_mm"] == -12.0
    assert diagnostic.pop("distance_backend")["effective_from_native"] == 76360
    for key in (
        "gate_preparation_seconds_by_view",
        "gate_preparation_seconds",
        "candidate_scoring_seconds_by_view",
        "candidate_scoring_seconds",
    ):
        old.pop(key)
        diagnostic.pop(key)
    exact(old, diagnostic)
    assert np.argmin(result[1]) == np.argmin(baseline[1]) == 1
    assert new._instance_frame is new._instance_observations is None
    assert not diagnostic["clears_identity_review"]

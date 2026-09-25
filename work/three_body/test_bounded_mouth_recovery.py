"""Synthetic calibrated rays only; no real fish, saved winner, fit or GPU."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tests"), str(REPO / "labs/20260909_three_body")]

import bounded_mouth_recovery as module  # noqa: E402
from node_geometry import RayObservations, supported_nodes  # noqa: E402
from node_recovery import NodeRecoveryConfig, node_recovery_proposals  # noqa: E402

from conftest import project, rig  # noqa: E402, F401
from medaka3d.types import VIEWS  # noqa: E402
from test_three_body_node_geometry import assert_injective, observations, pose  # noqa: E402, F401


def setup(rig, pose, shift=68):  # noqa: F811
    actual = pose.copy()
    actual[1] += [shift, 0, 0]
    points, true_plan = observations(rig, actual)
    current = actual.copy()
    current[1, 0] = current[1, 1] + [-5, 0, 0]
    plan = true_plan.copy()
    plan[1, :, 0] = -1
    lengths = np.linalg.norm(np.diff(actual, axis=1), axis=-1)
    return current, plan, RayObservations.from_points(rig, points), lengths, true_plan


def recover(rig, inputs, config=None):  # noqa: F811
    current, plan, rays, lengths, _ = inputs
    return module.bounded_mouth_recovery_proposals(current, plan, rays, rig, lengths, config)


def test_outside_free_point_adds_bounded_supported_mouth(rig, pose):  # noqa: F811
    inputs = setup(rig, pose)
    current, plan, rays, lengths, _ = inputs
    legacy = node_recovery_proposals(current, plan, rays, rig, lengths)
    assert not legacy.proposals
    assert legacy.diagnostics["fish"][1]["rejection_counts"]["outside_box"] == 4
    result = recover(rig, inputs)
    assert len(result.proposals) == 2
    info = result.diagnostics["fish"][1]
    assert info["bounded_attempts"] == 4 and info["admitted"] >= 2
    assert result.diagnostics["legacy_diagnostics"] == legacy.diagnostics
    assert not result.diagnostics["age_gate_applied"]
    assert not result.diagnostics["original_candidates_returned"]
    assert all(attempt["free_point_xyz"][0] > 192 for attempt in info["attempts"])
    for proposal in result.proposals:
        assert proposal.fish == 1 and proposal.node == 0
        assert proposal.pose_xyz[1, 0, 0] == 192
        assert proposal.ray_rms_mm <= 1.5
        assert 0.5 <= proposal.length_ratio <= 1.5
        assert_injective(proposal.plan)
        sources, observed, _ = supported_nodes(proposal.pose_xyz, proposal.plan, rays, rig)
        assert observed[1, :2].all()
        assert np.count_nonzero(sources[1, :, 0] >= 0) == proposal.supported_views
        np.testing.assert_array_equal(
            proposal.pose_xyz[np.arange(3) != 1], current[np.arange(3) != 1]
        )
        np.testing.assert_array_equal(proposal.pose_xyz[1, 1:], current[1, 1:])
        np.testing.assert_array_equal(proposal.plan[np.arange(3) != 1], plan[np.arange(3) != 1])
        np.testing.assert_array_equal(proposal.plan[1, :, 1:], plan[1, :, 1:])
    returned = [a for a in info["attempts"] if a["returned"]]
    assert len(returned) == len(result.proposals)


def test_inside_pool_is_unchanged_and_not_returned_or_sent_to_solver(rig, pose, monkeypatch):  # noqa: F811
    inputs = setup(rig, pose, shift=0)
    current, plan, rays, lengths, _ = inputs
    before = node_recovery_proposals(current, plan, rays, rig, lengths)
    assert before.proposals

    def forbidden(*args):
        raise AssertionError("an inside intersection must not call BVLS")

    monkeypatch.setattr(module.bounded_ray_seed, "solve_bounded_ray_seed", forbidden)
    result = recover(rig, inputs)
    assert not result.proposals
    assert all(not row["bounded_attempts"] for row in result.diagnostics["fish"])
    after = node_recovery_proposals(current, plan, rays, rig, lengths)
    assert before.diagnostics == after.diagnostics
    for a, b in zip(before.proposals, after.proposals, strict=True):
        np.testing.assert_array_equal(a.pose_xyz, b.pose_xyz)
        np.testing.assert_array_equal(a.plan, b.plan)


def test_mixed_inside_outside_pool_returns_only_outside_rescues(rig, pose):  # noqa: F811
    current, plan, rays, lengths, _ = setup(rig, pose)
    points = {view: values.copy() for view, values in rays.points.items()}
    for view in VIEWS:
        extra = np.full((1, 3, 2), np.nan)
        extra[0, 0] = project(rig.camera(view), current[1, 1] + [-4, 0, 0])
        points[view] = np.concatenate((points[view], extra))
    rays = RayObservations.from_points(rig, points)
    result = module.bounded_mouth_recovery_proposals(current, plan, rays, rig, lengths)
    assert result.proposals
    assert result.diagnostics["legacy_diagnostics"]["fish"][1]["admitted"] > 0
    for info in result.diagnostics["fish"]:
        for attempt in info["attempts"]:
            point = np.array(attempt["free_point_xyz"])
            assert np.any(point < [0, 0, -6]) or np.any(point > [192, 192, 186])


def test_other_fish_unsupported_base_mouth_rows_still_protected(rig, pose):  # noqa: F811
    inputs = list(setup(rig, pose))
    inputs[1][2, :, 0] = inputs[4][1, :, 0]
    _, support, _ = supported_nodes(inputs[0], inputs[1], inputs[2], rig)
    assert not support[2, 0]
    result = recover(rig, inputs)
    assert not any(proposal.fish == 1 for proposal in result.proposals)
    for proposal in result.proposals:
        assert_injective(proposal.plan)


def test_bad_current_chest_does_not_anchor_recovery(rig, pose):  # noqa: F811
    inputs = list(setup(rig, pose))
    inputs[0][1, 1] += [0, 20, 0]
    result = recover(rig, inputs)
    assert not any(p.fish == 1 for p in result.proposals)
    assert result.diagnostics["fish"][1]["status"] == "pectoral_not_directly_supported"


def test_bounded_point_must_retain_anatomy(rig, pose):  # noqa: F811
    inputs = list(setup(rig, pose))
    inputs[3][1, 0] = 0.1
    result = recover(rig, inputs)
    assert not result.proposals
    assert result.diagnostics["fish"][1]["rejection_counts"]["bounded_anatomy"] == 4


def test_bounded_point_still_needs_native_pixel_support(rig, pose):  # noqa: F811
    inputs = setup(rig, pose)
    for view, name in enumerate(VIEWS):
        inputs[2].points[name][inputs[4][1, view, 0], 0] += [100, 100]
    result = recover(rig, inputs)
    assert not result.proposals
    assert result.diagnostics["fish"][1]["rejection_counts"]["bounded_strict_support"] == 4


def test_bounded_solution_cannot_widen_rms_threshold(rig, pose):  # noqa: F811
    result = recover(rig, setup(rig, pose, shift=71))
    assert not result.proposals
    info = result.diagnostics["fish"][1]
    assert info["rejection_counts"]["bounded_ray_rms"] > 0
    for attempt in info["attempts"]:
        if attempt["ray_rms_mm"] > 1.5:
            assert attempt["status"] == "bounded_ray_rms"


def test_current_direct_mouth_does_not_get_an_addition(rig, pose):  # noqa: F811
    inputs = list(setup(rig, pose))
    inputs[0][1, 0] = [192, 90, 100]
    inputs[1][1, :, 0] = inputs[4][1, :, 0]
    _, observed, _ = supported_nodes(inputs[0], inputs[1], inputs[2], rig)
    assert observed[1, 0]
    result = recover(rig, inputs)
    assert not result.proposals
    assert result.diagnostics["fish"][1]["status"] == "not_needed"


def test_fewer_than_two_free_views_is_neutral(rig, pose):  # noqa: F811
    inputs = setup(rig, pose)
    for view in (1, 2):
        inputs[2].points[VIEWS[view]][inputs[4][1, view, 0], 0] = np.nan
    result = recover(rig, inputs)
    assert not result.proposals
    assert result.diagnostics["fish"][1]["status"] == "fewer_than_two_unclaimed_views"


def test_original_combination_limit_skips_whole_fish(rig, pose, monkeypatch):  # noqa: F811
    def forbidden(*args):
        raise AssertionError("budget-gated fish must not call BVLS")

    monkeypatch.setattr(module.bounded_ray_seed, "solve_bounded_ray_seed", forbidden)
    result = recover(rig, setup(rig, pose), NodeRecoveryConfig(maximum_combinations_per_fish=3))
    assert not result.proposals
    assert result.diagnostics["fish"][1]["status"] == "combination_limit_exceeded"


@pytest.mark.parametrize("bad", ["behind", "parallel"])
def test_free_ray_gate_precedes_bounded_solver(rig, pose, monkeypatch, bad):  # noqa: F811
    inputs = setup(rig, pose)
    for view in range(3):
        row = inputs[4][1, view, 0]
        if bad == "behind":
            inputs[2].directions[view][row, 0] *= -1
        else:
            inputs[2].directions[view][row, 0] = [0, 0, 1]

    def forbidden(*args):
        raise AssertionError("free ray gate must not call BVLS")

    monkeypatch.setattr(module.bounded_ray_seed, "solve_bounded_ray_seed", forbidden)
    result = recover(rig, inputs)
    assert not result.proposals


def test_solver_failure_is_explicit_not_an_admitted_candidate(rig, pose, monkeypatch):  # noqa: F811
    def failed(*args):
        raise RuntimeError("synthetic nonconvergence")

    monkeypatch.setattr(module.bounded_ray_seed, "solve_bounded_ray_seed", failed)
    result = recover(rig, setup(rig, pose))
    assert not result.proposals
    info = result.diagnostics["fish"][1]
    assert info["rejection_counts"]["bounded_solver_failed"] == 4
    assert all(row["message"] == "synthetic nonconvergence" for row in info["attempts"])


def test_per_fish_limit_and_ranking_retain_original_rules(rig, pose):  # noqa: F811
    inputs = setup(rig, pose)
    all_result = recover(rig, inputs, NodeRecoveryConfig(proposals_per_fish=10))
    limited = recover(rig, inputs)
    keys = [
        (-p.supported_views, p.ray_rms_mm, p.displacement_mm, tuple(p.plan[p.fish, :, 0]))
        for p in all_result.proposals
    ]
    assert keys == sorted(keys)
    for a, b in zip(limited.proposals, all_result.proposals[:2], strict=True):
        np.testing.assert_array_equal(a.pose_xyz, b.pose_xyz)
        np.testing.assert_array_equal(a.plan, b.plan)


def test_all_input_arrays_and_returned_ownership(rig, pose):  # noqa: F811
    inputs = setup(rig, pose)
    arrays = [
        inputs[0],
        inputs[1],
        inputs[3],
        *inputs[2].points.values(),
        *inputs[2].origins,
        *inputs[2].directions,
    ]
    before = [(a.dtype, a.shape, a.tobytes()) for a in arrays]
    for array in arrays:
        array.flags.writeable = False
    result = recover(rig, inputs)
    assert [(a.dtype, a.shape, a.tobytes()) for a in arrays] == before
    first, second = result.proposals
    saved_second = copy.deepcopy(second)
    first.pose_xyz[:] = 0
    first.plan[:] = -1
    assert [(a.dtype, a.shape, a.tobytes()) for a in arrays] == before
    np.testing.assert_array_equal(second.pose_xyz, saved_second.pose_xyz)
    np.testing.assert_array_equal(second.plan, saved_second.plan)


def test_original_input_validation_is_retained(rig, pose):  # noqa: F811
    inputs = list(setup(rig, pose))
    inputs[1][1, 0, 1] = inputs[1][0, 0, 1]
    with pytest.raises(ValueError, match="not injective"):
        recover(rig, inputs)

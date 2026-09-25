"""Node recovery requires current anatomy and exclusive observations, never an ID reset."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import project
from medaka3d.types import VIEWS
from test_three_body_node_geometry import (
    _GEOMETRY,
    assert_injective,
    observations,
    pose,  # noqa: F401
)

_PATH = Path(__file__).resolve().parents[1] / "labs/20260909_three_body/node_recovery.py"
_SPEC = importlib.util.spec_from_file_location("three_body_node_recovery_test_module", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_OLD = sys.modules.get("node_geometry")
sys.modules["node_geometry"] = _GEOMETRY
try:
    _SPEC.loader.exec_module(_MODULE)
finally:
    if _OLD is None:
        sys.modules.pop("node_geometry", None)
    else:
        sys.modules["node_geometry"] = _OLD
recover = _MODULE.node_recovery_proposals
Config = _MODULE.NodeRecoveryConfig


def setup_scene(rig, pose, seed=932):  # noqa: F811
    points, plan = observations(rig, pose, seed)
    predicted = pose.copy()
    predicted[1, 0] = pose[1, 1] + [-5, 0, 0]
    plan[1, :, 0] = -1
    lengths = np.linalg.norm(np.diff(pose, axis=1), axis=-1)
    return predicted, points, plan, lengths


def test_recovers_missing_mouth_with_healthy_chest_preserving_all_other_state(rig, pose):  # noqa: F811
    predicted, points, plan, lengths = setup_scene(rig, pose)
    original_pose, original_plan = predicted.copy(), plan.copy()
    rays = _GEOMETRY.RayObservations.from_points(rig, points)
    output = recover(predicted, plan, rays, rig, lengths)
    assert output.proposals
    best = output.proposals[0]
    assert best.fish == 1 and best.node == 0 and best.supported_views == 3
    np.testing.assert_allclose(best.pose_xyz[1, 0], pose[1, 0], atol=1e-7)
    assert best.displacement_mm == pytest.approx(10)
    assert best.length_ratio == pytest.approx(1)
    unchanged = np.ones((3, 3), bool)
    unchanged[1, 0] = False
    np.testing.assert_array_equal(best.pose_xyz[unchanged], predicted[unchanged])
    plan_unchanged = np.ones(plan.shape, bool)
    plan_unchanged[1, :, 0] = False
    np.testing.assert_array_equal(best.plan[plan_unchanged], plan[plan_unchanged])
    np.testing.assert_array_equal(predicted, original_pose)
    np.testing.assert_array_equal(plan, original_plan)
    for proposal in output.proposals:
        assert_injective(proposal.plan)
        assert not np.shares_memory(proposal.pose_xyz, predicted)
        assert not np.shares_memory(proposal.plan, plan)
        _, supported, _ = _GEOMETRY.supported_nodes(proposal.pose_xyz, proposal.plan, rays, rig)
        assert supported[1, :2].all()


def test_unobserved_center_cannot_anchor_a_plausible_mouth_candidate(rig, pose):  # noqa: F811
    predicted, points, plan, lengths = setup_scene(rig, pose)
    predicted[1, 1] += [15, 0, 0]
    result = recover(
        predicted, plan, _GEOMETRY.RayObservations.from_points(rig, points), rig, lengths
    )
    assert not result.proposals
    assert result.diagnostics["fish"][1]["status"] == "pectoral_not_directly_supported"


def test_one_unclaimed_view_does_not_create_an_observed_mouth(rig, pose):  # noqa: F811
    predicted, points, plan, lengths = setup_scene(rig, pose)
    _, expected = observations(rig, pose)
    for v in (1, 2):
        points[VIEWS[v]][expected[1, v, 0], 0] = np.nan
    result = recover(
        predicted, plan, _GEOMETRY.RayObservations.from_points(rig, points), rig, lengths
    )
    assert not result.proposals
    assert result.diagnostics["fish"][1]["status"] == "fewer_than_two_unclaimed_views"


def test_even_an_unsupported_other_slot_base_row_is_protected(rig, pose):  # noqa: F811
    predicted, points, plan, lengths = setup_scene(rig, pose)
    _, expected = observations(rig, pose)
    plan[2, :, 0] = expected[1, :, 0]
    rays = _GEOMETRY.RayObservations.from_points(rig, points)
    _, held, _ = _GEOMETRY.supported_nodes(predicted, plan, rays, rig)
    assert not held[2, 0]  # Protect the base assignment regardless of its filtered support.
    result = recover(predicted, plan, rays, rig, lengths)
    assert not any(p.fish == 1 for p in result.proposals)
    for candidate in result.proposals:
        assert_injective(candidate.plan)
        if candidate.fish != 2:
            np.testing.assert_array_equal(candidate.plan[2, :, 0], expected[1, :, 0])


def test_far_ray_intersection_fails_anatomy_instead_of_stretching_a_healthy_body(rig, pose):  # noqa: F811
    predicted, points, plan, lengths = setup_scene(rig, pose)
    _, expected = observations(rig, pose)
    wrong_mouth = pose[1, 1] + [25, 0, 0]
    for v, view in enumerate(VIEWS):
        points[view][expected[1, v, 0], 0] = project(rig.camera(view), wrong_mouth)
    result = recover(
        predicted, plan, _GEOMETRY.RayObservations.from_points(rig, points), rig, lengths
    )
    assert not result.proposals
    assert result.diagnostics["fish"][1]["rejection_counts"]["anatomy"] == 4


def test_outside_wall_candidates_are_rejected_without_clipping(rig, pose):  # noqa: F811
    truth = pose.copy()
    truth[1] += [68, 0, 0]  # Pectoral x188, anatomically consistent mouth x193 > box192.
    predicted, points, plan, lengths = setup_scene(rig, truth)
    result = recover(
        predicted, plan, _GEOMETRY.RayObservations.from_points(rig, points), rig, lengths
    )
    assert not result.proposals
    assert result.diagnostics["fish"][1]["rejection_counts"]["outside_box"] == 4


def test_wrong_third_view_preserves_the_two_view_alternative(rig, pose):  # noqa: F811
    predicted, points, plan, lengths = setup_scene(rig, pose)
    _, expected = observations(rig, pose)
    points[VIEWS[0]][expected[1, 0, 0], 0] += [240, -160]
    result = recover(
        predicted, plan, _GEOMETRY.RayObservations.from_points(rig, points), rig, lengths
    )
    assert result.proposals
    best = result.proposals[0]
    assert best.supported_views == 2 and best.plan[1, 0, 0] == -1
    np.testing.assert_allclose(best.pose_xyz[1, 0], pose[1, 0], atol=1e-7)


def test_parallel_mouth_rays_cannot_create_a_finite_recovery(rig, pose):  # noqa: F811
    predicted, points, plan, lengths = setup_scene(rig, pose)
    _, expected = observations(rig, pose)
    rays = _GEOMETRY.RayObservations.from_points(rig, points)
    for view in range(3):
        row = expected[1, view, 0]
        rays.directions[view][row, 0] = [0.0, 0.0, 1.0]
    result = recover(predicted, plan, rays, rig, lengths)
    assert not result.proposals
    assert result.diagnostics["fish"][1]["rejection_counts"]["nonfinite_or_behind"] == 4


def test_current_row_permutations_do_not_change_recovered_geometry(rig, pose):  # noqa: F811
    returned = []
    for seed in (41, 19):
        predicted, points, plan, lengths = setup_scene(rig, pose, seed)
        result = recover(
            predicted, plan, _GEOMETRY.RayObservations.from_points(rig, points), rig, lengths
        )
        returned.append(result.proposals[0].pose_xyz)
    np.testing.assert_allclose(returned[0], returned[1], atol=1e-8)


def test_combination_budget_skips_whole_fish_with_explicit_diagnostic(rig, pose):  # noqa: F811
    predicted, points, plan, lengths = setup_scene(rig, pose)
    result = recover(
        predicted,
        plan,
        _GEOMETRY.RayObservations.from_points(rig, points),
        rig,
        lengths,
        Config(maximum_combinations_per_fish=3),
    )
    assert not result.proposals
    assert result.diagnostics["fish"][1]["combinations"] == 4
    assert result.diagnostics["fish"][1]["status"] == "combination_limit_exceeded"


def test_healthy_mouths_need_no_recovery(rig, pose):  # noqa: F811
    points, plan = observations(rig, pose)
    result = recover(
        pose,
        plan,
        _GEOMETRY.RayObservations.from_points(rig, points),
        rig,
        np.linalg.norm(np.diff(pose, axis=1), axis=-1),
    )
    assert not result.proposals
    assert all(row["status"] == "not_needed" for row in result.diagnostics["fish"])


def test_nonexclusive_base_plan_fails_loudly(rig, pose):  # noqa: F811
    predicted, points, plan, lengths = setup_scene(rig, pose)
    plan[1, 0, 1] = plan[0, 0, 1]
    with pytest.raises(ValueError, match="not injective"):
        recover(predicted, plan, _GEOMETRY.RayObservations.from_points(rig, points), rig, lengths)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ray_rms_mm": np.nan},
        {"minimum_length_ratio": 2},
        {"proposals_per_fish": True},
        {"maximum_combinations_per_fish": 0},
        {"upper_mm": (0, 0, 0)},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)

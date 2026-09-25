"""Physical tail recovery from calibrated synthetic views; no real ID labels."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tests"), str(REPO / "labs/20260909_three_body")]

import tail_recovery as module  # noqa: E402
from node_geometry import RayObservations, supported_nodes  # noqa: E402
from node_recovery import NodeRecoveryConfig  # noqa: E402

from conftest import rig  # noqa: E402,F401
from medaka3d.types import VIEWS  # noqa: E402
from test_three_body_node_geometry import assert_injective, observations, pose  # noqa: E402,F401


def setup(rig, pose, *, outside=False, seed=932):  # noqa: F811
    truth = pose.copy()
    if outside:
        truth[1, :, 0] += -1 - truth[1, 2, 0]
    points, true_plan = observations(rig, truth, seed)
    current, plan = truth.copy(), true_plan.copy()
    current[1, 2] = 2 * truth[1, 1] - truth[1, 2]
    plan[1, :, 2] = -1
    return (
        current,
        plan,
        RayObservations.from_points(rig, points),
        np.linalg.norm(np.diff(truth, axis=1), axis=-1),
        truth,
        true_plan,
    )


def recover(rig, data, **kwargs):  # noqa: F811
    return module.tail_recovery_proposals(*data[:3], rig, data[3], **kwargs)


def test_reversing_twice_preserves_all_native_node_data_and_length_order(rig, pose):  # noqa: F811
    data = setup(rig, pose)
    p, plan, rays, lengths = module.reversed_inputs(*data[:3], data[3])
    p, plan, rays, lengths = module.reversed_inputs(p, plan, rays, lengths)
    np.testing.assert_array_equal(p, data[0])
    np.testing.assert_array_equal(plan, data[1])
    np.testing.assert_array_equal(lengths, data[3])
    for i, view in enumerate(VIEWS):
        np.testing.assert_array_equal(rays.points[view], data[2].points[view])
        np.testing.assert_array_equal(rays.origins[i], data[2].origins[i])
        np.testing.assert_array_equal(rays.directions[i], data[2].directions[i])
        assert not np.shares_memory(rays.points[view], data[2].points[view])


def test_missing_tail_recovery_uses_tail_length_and_preserves_head_other_fish_and_inputs(rig, pose):  # noqa: F811
    data = setup(rig, pose)
    before = copy.deepcopy(data)
    result = recover(rig, data)
    assert result.proposals
    best = result.proposals[0]
    assert best.node == 2 and best.fish == 1 and best.seed_method == "free_intersection"
    np.testing.assert_allclose(best.pose_xyz[1, 2], data[4][1, 2], atol=1e-7)
    assert best.length_ratio == pytest.approx(1)
    assert best.tail_pectoral_mm == pytest.approx(data[3][1, 1])
    assert best.tail_pectoral_mm != pytest.approx(data[3][1, 0])
    for proposal in result.proposals:
        np.testing.assert_array_equal(proposal.pose_xyz[:, :2], data[0][:, :2])
        np.testing.assert_array_equal(proposal.plan[:, :, :2], data[1][:, :, :2])
        np.testing.assert_array_equal(proposal.pose_xyz[[0, 2]], data[0][[0, 2]])
        np.testing.assert_array_equal(proposal.plan[[0, 2]], data[1][[0, 2]])
        assert_injective(proposal.plan)
        assert not np.shares_memory(proposal.pose_xyz, data[0])
        assert not np.shares_memory(proposal.plan, data[1])
        sources, observed, _ = supported_nodes(proposal.pose_xyz, proposal.plan, data[2], rig)
        assert observed[1, 1:].all()
        assert int((sources[1, :, 2] >= 0).sum()) == proposal.supported_views
    for index in (0, 1, 3):
        np.testing.assert_array_equal(data[index], before[index])
    for vi, view in enumerate(VIEWS):
        np.testing.assert_array_equal(data[2].points[view], before[2].points[view])
        np.testing.assert_array_equal(data[2].origins[vi], before[2].origins[vi])
        np.testing.assert_array_equal(data[2].directions[vi], before[2].directions[vi])
    assert not result.diagnostics["age_gate_applied"]
    assert not result.diagnostics["tracker_budget_applied"]


def test_outside_tail_can_use_bounded_seed_without_changing_the_box(rig, pose):  # noqa: F811
    data = setup(rig, pose, outside=True)
    assert not recover(rig, data, allow_bounded=False).proposals
    result = recover(rig, data)
    assert result.proposals
    assert all(p.seed_method == "bounded_intersection" for p in result.proposals)
    assert all(p.pose_xyz[1, 2, 0] == 0 for p in result.proposals)
    assert len(result.proposals) <= 2


@pytest.mark.parametrize(
    "fault", ["healthy", "chest_missing", "one_view", "protected", "bad_length"]
)
def test_no_supported_tail_is_manufactured_when_recovery_requirements_fail(rig, pose, fault):  # noqa: F811
    data = list(setup(rig, pose))
    if fault == "healthy":
        data[0][1, 2] = data[4][1, 2]
        data[1][1, :, 2] = data[5][1, :, 2]
    elif fault == "chest_missing":
        data[1][1, :, 1] = -1
    elif fault == "one_view":
        for vi in (1, 2):
            data[2].points[VIEWS[vi]][data[5][1, vi, 2], 2] = np.nan
    elif fault == "protected":
        data[1][2, :, 2] = data[5][1, :, 2]
    else:
        data[3][1, 1] = 1
    result = recover(rig, data)
    assert not any(p.fish == 1 for p in result.proposals)


def test_wrong_third_view_keeps_two_view_tail_with_native_source_indices(rig, pose):  # noqa: F811
    data = setup(rig, pose)
    data[2].points[VIEWS[0]][data[5][1, 0, 2], 2] += [240, -160]
    rays = RayObservations.from_points(rig, data[2].points)
    data = (*data[:2], rays, *data[3:])
    result = recover(rig, data)
    assert result.proposals[0].supported_views == 2
    assert result.proposals[0].plan[1, 0, 2] == -1
    np.testing.assert_allclose(result.proposals[0].pose_xyz[1, 2], data[4][1, 2], atol=1e-7)


def test_combination_cap_is_whole_fish_and_no_new_angle_gate_is_imposed(rig, pose):  # noqa: F811
    data = setup(rig, pose)
    result = recover(rig, data, config=NodeRecoveryConfig(maximum_combinations_per_fish=3))
    assert not result.proposals
    assert (
        result.diagnostics["original_delegate"]["fish"][1]["status"] == "combination_limit_exceeded"
    )
    assert result.diagnostics["no_angle_constraint_added"]


def test_source_row_order_does_not_change_the_recovered_tail_geometry(rig, pose):  # noqa: F811
    a = recover(rig, setup(rig, pose, seed=41)).proposals[0]
    b = recover(rig, setup(rig, pose, seed=19)).proposals[0]
    np.testing.assert_allclose(a.pose_xyz, b.pose_xyz, atol=1e-7)


def test_nonboolean_bounded_option_and_wrong_native_axis_are_rejected(rig, pose):  # noqa: F811
    data = setup(rig, pose)
    with pytest.raises(ValueError, match="boolean"):
        recover(rig, data, allow_bounded=1)
    with pytest.raises(ValueError, match="shapes"):
        module.reversed_inputs(data[0][:2], data[1], data[2], data[3])

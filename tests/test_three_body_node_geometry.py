"""Independent synthetic correspondences, missing nodes, and source exclusion."""

import importlib.util
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from conftest import project
from medaka3d.types import VIEWS

_PATH = Path(__file__).resolve().parents[1] / "labs/20260909_three_body/node_geometry.py"
_SPEC = importlib.util.spec_from_file_location("three_body_node_geometry_test_module", _PATH)
_GEOMETRY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _GEOMETRY
_SPEC.loader.exec_module(_GEOMETRY)
RayObservations = _GEOMETRY.RayObservations
assignment_plans = _GEOMETRY.assignment_plans
assigned_rays = _GEOMETRY.assigned_rays
exclusive = _GEOMETRY.exclusive
node_pools = _GEOMETRY.node_pools
recovery_poses = _GEOMETRY.recovery_poses
supported_nodes = _GEOMETRY.supported_nodes


@pytest.fixture
def pose():
    centers = np.array([[40., 60., 30.], [120., 90., 100.], [70., 150., 160.]])
    return centers[:, None] + np.array([[5., 0., 0.], [0., 0., 0.], [-12., 1., -1.]])[None]


def observations(rig, pose, seed=932):
    """Each node has its own row permutation: grouped instances are deliberately wrong."""
    rng = np.random.default_rng(seed)
    points, expected = {}, np.empty((3, 3, 3), dtype=np.int32)
    for vi, view in enumerate(VIEWS):
        raw = project(rig.camera(view), pose)  # Closed-form pinhole, not camera.project().
        output = np.empty_like(raw)
        for node in range(3):
            order = rng.permutation(3)
            output[:, node] = raw[order, node]
            expected[:, vi, node] = np.argsort(order)
        points[view] = output
    return points, expected


def assert_injective(plan):
    for view in range(3):
        for node in range(3):
            used = plan[:, view, node]
            used = used[used >= 0]
            assert len(used) == len(set(used))


def test_node_mixing_assignments_recover_independent_row_permutations(rig, pose):
    points, expected = observations(rig, pose)
    rays = RayObservations.from_points(rig, points)
    plans = assignment_plans(pose, rays, keep=4)
    np.testing.assert_array_equal(plans[0][1], expected)
    assert plans[0][0] < 1e-12
    for _, plan in plans:
        assert_injective(plan)
    sources, supported, distance = supported_nodes(pose, plans[0][1], rays, rig)
    np.testing.assert_array_equal(sources, expected)
    assert supported.all()
    assert distance.max() < 1e-4  # assigned_rays stores geometry at float32 precision.


def test_triangulation_proposals_recover_true_positions_from_independent_pinhole_pixels(rig, pose):
    points, expected = observations(rig, pose)
    pools = node_pools(rig, RayObservations.from_points(rig, points),
                      np.array([0, 0, -6]), np.array([192, 192, 186]))
    for node, pool in enumerate(pools):
        for fish in range(3):
            matching = [item for item in pool if np.array_equal(item[1], expected[fish, :, node])]
            assert len(matching) == 1
            np.testing.assert_allclose(matching[0][0], pose[fish, node], atol=1e-7)
            assert matching[0][2] < 1e-8


def test_empty_view_still_allows_two_view_geometry_without_fabricating_source(rig, pose):
    points, expected = observations(rig, pose)
    points[VIEWS[2]] = np.empty((0, 3, 2))
    expected[:, 2] = -1
    rays = RayObservations.from_points(rig, points)
    best = assignment_plans(pose, rays)[0][1]
    np.testing.assert_array_equal(best, expected)
    sources, supported, _ = supported_nodes(pose, best, rays, rig)
    assert supported.all()
    assert (sources[:, 2] == -1).all()
    pools = node_pools(rig, rays, np.zeros(3), np.array([192, 192, 186]))
    for node, pool in enumerate(pools):
        for fish in range(3):
            matching = [item for item in pool if np.array_equal(item[1], expected[fish, :, node])]
            assert matching
            np.testing.assert_allclose(matching[0][0], pose[fish, node], atol=1e-7)


def test_bad_third_view_does_not_remove_true_two_view_candidate(rig, pose):
    points, expected = observations(rig, pose)
    bad_row = expected[1, 0, 1]
    points[VIEWS[0]][bad_row, 1] += [230, -150]
    rays = RayObservations.from_points(rig, points)
    pools = node_pools(rig, rays, np.zeros(3), np.array([192, 192, 186]))
    source = expected[1, :, 1].copy()
    source[0] = -1
    matching = [item for item in pools[1] if np.array_equal(item[1], source)]
    assert matching
    np.testing.assert_allclose(matching[0][0], pose[1, 1], atol=1e-7)
    best = assignment_plans(pose, rays)[0][1]
    assert best[1, 0, 1] == -1
    _, supported, _ = supported_nodes(pose, best, rays, rig)
    assert supported[1, 1]


def test_only_one_view_support_never_becomes_a_measured_node(rig, pose):
    points, expected = observations(rig, pose)
    for vi in (1, 2):
        points[VIEWS[vi]][expected[2, vi, 2], 2] = np.nan
    rays = RayObservations.from_points(rig, points)
    best = assignment_plans(pose, rays)[0][1]
    sources, supported, _ = supported_nodes(pose, best, rays, rig)
    assert not supported[2, 2]
    assert sources[2, 0, 2] >= 0
    assert (sources[2, 1:, 2] == -1).all()
    assert supported[:2].all() and supported[2, :2].all()


def test_all_missing_has_no_support_no_proposals_and_zero_unheld_rays(rig, pose):
    points = {view: np.empty((0, 3, 2)) for view in VIEWS}
    rays = RayObservations.from_points(rig, points)
    best = assignment_plans(pose, rays)[0][1]
    assert (best == -1).all()
    origin, direction, held = assigned_rays(rays, best[None])
    assert not held.any() and not origin.any() and not direction.any()
    _, supported, _ = supported_nodes(pose, best, rays, rig)
    assert not supported.any()
    assert node_pools(rig, rays, np.zeros(3), np.full(3, 192)) == [[], [], []]


def test_exclusive_best_matches_complete_enumeration_with_conflicts_and_missing():
    rng = np.random.default_rng(221)
    missing = 9.0
    for _ in range(15):
        costs = rng.uniform(0, 12, (3, 5))
        costs[rng.random(costs.shape) < 0.2] = np.nan
        reference = []
        for assignment in product(range(-1, 5), repeat=3):
            used = [row for row in assignment if row >= 0]
            if len(used) != len(set(used)):
                continue
            values = [missing if row == -1 else costs[fish, row]
                      for fish, row in enumerate(assignment)]
            if np.isfinite(values).all():
                reference.append((sum(values), assignment))
        reference.sort()
        actual = exclusive(costs, missing, keep=3)
        assert actual[0][0] == pytest.approx(reference[0][0])
        assert tuple(actual[0][1]) == reference[0][1]
        for _, assignment in actual:
            used = assignment[assignment >= 0]
            assert len(used) == len(set(used))


def test_exclusive_retains_fourth_detection_needed_for_the_second_global_solution():
    # Fish 1 and 2 occupy fish 0's cheapest two detections. Its third-best row
    # completes the best global plan; its fourth-best row is essential to keep
    # the next plausible plan instead of incorrectly declaring fish 0 missing.
    costs = np.array([[0., 1., 2., 3.], [0., 100., 100., 100.], [100., 0., 100., 100.]])
    for column_order in (np.arange(4), np.array([3, 0, 2, 1])):
        candidates = exclusive(costs[:, column_order], missing_cost=9, keep=2)
        np.testing.assert_array_equal([candidate[0] for candidate in candidates], [2, 3])
        physical_rows = np.stack([column_order[candidate[1]] for candidate in candidates])
        np.testing.assert_array_equal(physical_rows, [[2, 0, 1], [3, 0, 1]])
        assert all((candidate[1] >= 0).all() for candidate in candidates)


def test_frame_local_row_changes_do_not_change_geometric_fish_assignment(rig, pose):
    recovered = []
    for seed in (41, 19):
        points, expected = observations(rig, pose, seed=seed)
        rays = RayObservations.from_points(rig, points)
        best = assignment_plans(pose, rays)[0][1]
        np.testing.assert_array_equal(best, expected)
        gathered = np.stack([points[view][best[:, vi, 1], 1]
                             for vi, view in enumerate(VIEWS)], axis=1)
        recovered.append(gathered)
    np.testing.assert_array_equal(recovered[0], recovered[1])


def test_support_requires_reprojection_even_when_ray_threshold_is_relaxed(rig, pose):
    points, expected = observations(rig, pose)
    rays = RayObservations.from_points(rig, points)
    wrong_pose = pose + np.array([20, 30, 40])
    _, supported, _ = supported_nodes(wrong_pose, expected, rays, rig,
                                      maximum_ray_mm=1000, maximum_pixel_error=12)
    assert not supported.any()


def test_pool_bounds_are_respected_without_clipping_candidates_onto_boundary(rig, pose):
    points, _ = observations(rig, pose)
    lower, upper = np.array([80., 0., 0.]), np.array([192., 192., 186.])
    pools = node_pools(rig, RayObservations.from_points(rig, points), lower, upper)
    for pool in pools:
        assert all((item[0] >= lower).all() and (item[0] <= upper).all() for item in pool)
        assert all(not np.isclose(item[0][0], lower[0]) for item in pool)


def test_recovery_uses_unused_current_sources_and_keeps_other_fish_unchanged(rig, pose):
    points, _ = observations(rig, pose)
    rays = RayObservations.from_points(rig, points)
    lower, upper = np.array([0, 0, -6]), np.array([192, 192, 186])
    predicted = pose.copy()
    predicted[2] -= [22, 0, 0]
    lengths = np.linalg.norm(np.diff(pose, axis=1), axis=-1)
    base = assignment_plans(predicted, rays)[0][1]
    pools = node_pools(rig, rays, lower, upper)
    candidates = recovery_poses(predicted, np.array([0, 0, 8]), lengths, pools,
                                base, lower, upper, keep=8)
    assert candidates
    assert any(np.allclose(candidate[1][2], pose[2], atol=1e-6) for candidate in candidates)
    for _, candidate, fish in candidates:
        assert fish == 2
        np.testing.assert_array_equal(candidate[:2], predicted[:2])
        candidate_plan = assignment_plans(candidate, rays)[0][1]
        assert_injective(candidate_plan)
        # Claimed sources for the two still-observed bodies must remain available.
        np.testing.assert_array_equal(candidate_plan[:2], base[:2])


def test_view_presence_and_node_shapes_are_checked(rig):
    points = {view: np.empty((0, 3, 2)) for view in VIEWS}
    with pytest.raises(ValueError, match="all three views"):
        RayObservations.from_points(rig, {VIEWS[0]: points[VIEWS[0]]})
    points[VIEWS[1]] = np.zeros((3, 2))
    with pytest.raises(ValueError, match="node array"):
        RayObservations.from_points(rig, points)

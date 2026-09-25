"""Independent synthetic camera tests; no saved observation or tracking inputs.

The orthographic cameras have perpendicular unit rays and exact projections.
They isolate forward-ray, shared-coordinate, and assigned-view pixel checks.
"""

import copy
from dataclasses import replace
from types import SimpleNamespace

import body_row_proposals as proposals
import numpy as np
import pytest

from medaka3d.types import VIEWS

LOWER = np.array([-50.0, -50.0, -50.0])
UPPER = np.array([50.0, 50.0, 50.0])


class Camera:
    def __init__(self, axis, plane=-100.0, projection_bias=0.0):
        self.axis = axis
        self.plane = plane
        self.xy_axes = tuple(i for i in range(3) if i != axis)
        self.projection_bias = projection_bias

    def project(self, xyz):
        return np.asarray(xyz)[..., self.xy_axes] + self.projection_bias

    def rays(self, points):
        points = np.asarray(points)
        origins = np.zeros((*points.shape[:-1], 3))
        directions = np.zeros_like(origins)
        origins[..., self.axis] = self.plane
        origins[..., self.xy_axes] = points
        directions[..., self.axis] = 1.0
        missing = ~np.isfinite(points).all(axis=-1)
        origins[missing] = np.nan
        directions[missing] = np.nan
        return origins, directions


def fixture(body=None, point_changes=None, camera_axes=(2, 0, 1), plane=-100.0):
    if body is None:
        body = np.array([[0.0, 0.0, 10.0], [3.0, 1.0, 11.0], [8.0, 2.0, 12.0]])
    bodies = body[None] if body.ndim == 2 else body
    cameras = {v: Camera(a, plane) for v, a in zip(VIEWS, camera_axes, strict=True)}
    points = {v: cameras[v].project(bodies).copy() for v in VIEWS}
    if point_changes is not None:
        point_changes(points)
    origins, directions = [], []
    for v in VIEWS:
        origin, direction = cameras[v].rays(points[v])
        origins.append(origin)
        directions.append(direction)
    rig = SimpleNamespace(views=VIEWS, camera=cameras.__getitem__)
    rays = SimpleNamespace(points=points, origins=origins, directions=directions)
    return bodies, rig, rays


def snapshot(rays):
    return [x.tobytes() for x in [*rays.points.values(), *rays.origins, *rays.directions]]


def pool(rig, rays, **kwargs):
    return proposals.body_row_pool(rig, rays, LOWER, UPPER, **kwargs)


def no_third_view(points):
    points[VIEWS[2]] = np.empty((0, 3, 2))


def test_complete_body_reconstructed_from_three_views_and_all_pairs_without_mutation():
    bodies, rig, rays = fixture()
    before = snapshot(rays)
    actual, report = pool(rig, rays)
    assert report["row_combinations"] == 4 and len(actual) == 4
    assert {tuple(x["view_rows"]) for x in actual} == {
        (0, 0, 0),
        (0, 0, -1),
        (0, -1, 0),
        (-1, 0, 0),
    }
    for candidate in actual:
        np.testing.assert_allclose(candidate["pose_xyz"], bodies[0], atol=1e-12, rtol=0)
        assert candidate["triangulated_nodes"].all()
        assert candidate["same_row_within_each_view"]
        assert candidate["direct_support_not_recomputed"]
        for view, row in enumerate(candidate["view_rows"]):
            np.testing.assert_array_equal(candidate["proposed_sources"][view], [row] * 3)
    assert snapshot(rays) == before


def test_wrong_third_view_rejected_while_consistent_pair_survives():
    def corrupt(points):
        points[VIEWS[2]] += 40.0

    bodies, rig, rays = fixture(point_changes=corrupt)
    actual, report = pool(rig, rays)
    assert report["row_combinations"] == 4
    assert len(actual) == 1 and actual[0]["view_rows"].tolist() == [0, 0, -1]
    np.testing.assert_allclose(actual[0]["pose_xyz"], bodies[0], atol=1e-12)


def test_partial_two_nodes_and_one_view_source_remain_explicitly_latent():
    prediction = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 1.0, 0.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expected = prediction @ rotation.T + [4.0, -2.0, 10.0]

    def missing(points):
        for v in VIEWS[1:]:
            points[v][:, 2] = np.nan

    _, rig, rays = fixture(expected, point_changes=missing)
    actual, _ = pool(rig, rays)
    candidate = next(x for x in actual if np.array_equal(x["view_rows"], [0, 0, 0]))
    assert candidate["triangulated_nodes"].tolist() == [True, True, False]
    assert np.isnan(candidate["pose_xyz"][2]).all()
    assert candidate["proposed_sources"][:, 2].tolist() == [0, -1, -1]
    assert candidate["used_ray_count"] == 7
    before = copy.deepcopy(candidate)
    completed = proposals.complete_from_prediction(candidate, prediction)
    np.testing.assert_allclose(completed, expected, atol=1e-12)
    np.testing.assert_array_equal(candidate["pose_xyz"], before["pose_xyz"])
    np.testing.assert_array_equal(candidate["triangulated_nodes"], before["triangulated_nodes"])
    strict, _ = pool(
        rig, rays, config=replace(proposals.BodyRowConfig(), minimum_triangulated_nodes=3)
    )
    assert strict == []


def test_bad_third_node_is_not_dropped_to_rescue_two_good_nodes():
    def corrupt(points):
        no_third_view(points)
        points[VIEWS[1]][0, 2, 0] += 10.0  # Contradict the shared Y coordinate.

    _, rig, rays = fixture(point_changes=corrupt)
    actual, report = pool(rig, rays)
    assert report["row_combinations"] == 1
    assert actual == []


@pytest.mark.parametrize("shared_coordinate_error, accepted", [(3.0, True), (3.1, False)])
def test_ray_rms_boundary_is_applied_to_each_measured_node(shared_coordinate_error, accepted):
    def corrupt(points):
        no_third_view(points)
        points[VIEWS[1]][0, 2, 0] += shared_coordinate_error

    _, rig, rays = fixture(point_changes=corrupt)
    actual, _ = pool(rig, rays)
    assert bool(actual) == accepted
    if accepted:
        assert actual[0]["ray_rms_mm"][2] == 1.5


def test_every_assigned_view_reprojection_is_checked_even_with_zero_ray_rms():
    _, rig, rays = fixture(point_changes=no_third_view)
    # Rays are exact, but the projector disagrees by 13 px on both coordinates.
    rig.camera(VIEWS[1]).projection_bias = 13.0
    actual, report = pool(rig, rays)
    assert report["row_combinations"] == 1 and actual == []


@pytest.mark.parametrize("failure", ["parallel", "behind", "bounds"])
def test_degenerate_or_physically_invalid_intersections_are_not_candidates(failure):
    _, rig, rays = fixture(
        camera_axes=(2, 2, 2) if failure == "parallel" else (2, 0, 1),
        plane=100.0 if failure == "behind" else -100.0,
    )
    lower = [1.0, -50.0, -50.0] if failure == "bounds" else LOWER
    actual, report = proposals.body_row_pool(rig, rays, lower, UPPER)
    assert report["row_combinations"] == 4 and actual == []


def test_row_combination_limit_fails_without_silent_partial_enumeration():
    body = np.array([[0.0, 0.0, 10.0], [3.0, 1.0, 11.0], [8.0, 2.0, 12.0]])
    _, rig, rays = fixture(np.stack([body, body + 5.0]))
    before = snapshot(rays)
    with pytest.raises(ValueError, match="row combination bound exceeded"):
        pool(rig, rays, config=replace(proposals.BodyRowConfig(), maximum_row_combinations=7))
    assert snapshot(rays) == before


def test_nonunit_finite_ray_is_rejected_and_no_observations_returns_empty_pool():
    _, rig, rays = fixture()
    rays.directions[0][0, 0] *= 2.0
    with pytest.raises(ValueError, match="unit directions"):
        pool(rig, rays)

    def missing(points):
        for value in points.values():
            value[:] = np.nan

    _, rig, rays = fixture(point_changes=missing)
    actual, report = pool(rig, rays)
    assert actual == [] and report["row_combinations"] == 0


@pytest.mark.parametrize("missing_node", [0, 1, 2])
def test_rigid_template_completion_preserves_measured_nodes_for_each_latent_position(missing_node):
    prediction = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 1.0, 0.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expected = prediction @ rotation.T + [4.0, -2.0, 10.0]
    measured = np.ones(3, dtype=bool)
    measured[missing_node] = False
    pose = expected.copy()
    pose[missing_node] = np.nan
    before = pose.tobytes(), prediction.tobytes()
    completed = proposals.complete_from_prediction(
        {"pose_xyz": pose, "triangulated_nodes": measured}, prediction
    )
    np.testing.assert_array_equal(completed[measured], pose[measured])
    np.testing.assert_allclose(completed, expected, atol=1e-12, rtol=0)
    assert before == (pose.tobytes(), prediction.tobytes())


def test_antiparallel_rotation_is_proper_deterministic_and_degenerate_pair_fails():
    direction = np.array([1.0, 2.0, -3.0])
    matrix = proposals._minimum_rotation(direction, -direction)
    np.testing.assert_allclose(matrix @ direction, -direction, atol=1e-12)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)
    np.testing.assert_array_equal(matrix, proposals._minimum_rotation(direction, -direction))
    np.testing.assert_array_equal(proposals._minimum_rotation(direction, 2 * direction), np.eye(3))
    with pytest.raises(ValueError, match="degenerate"):
        proposals._minimum_rotation(np.zeros(3), direction)


def test_joint_scene_cannot_reuse_one_row_for_three_bodies():
    bodies, rig, rays = fixture()
    geometry, _ = pool(rig, rays)
    one = [next(x for x in geometry if np.array_equal(x["view_rows"], [0, 0, 0]))]
    prediction = np.repeat(bodies, 3, axis=0)
    lengths = np.linalg.norm(np.diff(prediction, axis=1), axis=-1)
    scenes, report = proposals.joint_body_row_scenes(one, prediction, lengths, LOWER, UPPER)
    assert scenes == [] and report["conflicting_scene_combinations"] == 1
    assert report["admissible_scenes"] == 0


def test_scene_pruning_reports_approximation_and_keeps_source_rows_disjoint():
    body = np.array([[0.0, 0.0, 10.0], [3.0, 1.0, 11.0], [8.0, 2.0, 12.0]])
    prediction = np.stack([body + delta for delta in ([-15, -15, -5], [0, 0, 0], [15, 15, 5])])
    _, rig, rays = fixture(prediction)
    geometry, _ = pool(rig, rays)
    original = copy.deepcopy(geometry)
    lengths = np.linalg.norm(np.diff(prediction, axis=1), axis=-1)
    before = prediction.tobytes(), lengths.tobytes()
    cfg = replace(proposals.BodyRowConfig(), per_slot_limit=2, scene_limit=1)
    scenes, report = proposals.joint_body_row_scenes(
        geometry, prediction, lengths, LOWER, UPPER, cfg
    )
    assert len(scenes) == 1 and report["scenes_returned"] == 1
    assert report["retained_body_choices_by_slot"] == [2, 2, 2]
    assert all(n > 0 for n in report["per_slot_pruned"])
    assert report["admissible_scenes"] > report["scenes_returned"]
    assert not report["exact_global_search"] and report["existing_proposals_must_be_retained"]
    for scene in scenes:
        assert not scene["physical_IDs_assigned"] and scene["parent_prediction_used"]
        for rows in scene["source_rows"].T:
            used = rows[rows >= 0]
            assert len(used) == len(set(used.tolist()))
        for slot, index in enumerate(scene["pool_indices"]):
            np.testing.assert_array_equal(
                scene["proposed_sources"][slot], geometry[index]["proposed_sources"]
            )
    assert before == (prediction.tobytes(), lengths.tobytes())
    for saved, current in zip(original, geometry, strict=True):
        for key in ("pose_xyz", "view_rows", "proposed_sources", "triangulated_nodes"):
            np.testing.assert_array_equal(saved[key], current[key])


def test_latent_completion_must_still_fit_scene_bounds_and_segment_lengths():
    template = np.array([[0.0, 0.0, 10.0], [2.0, 0.0, 10.0], [100.0, 1.0, 10.0]])
    measured = np.array([True, True, False])
    partial = template.copy()
    partial[2] = np.nan
    body = {
        "pose_xyz": partial,
        "triangulated_nodes": measured,
        "ray_rms_mm": np.array([0.0, 0.0, np.nan]),
        "view_rows": np.array([0, 0, -1], np.int32),
        "proposed_sources": np.array([[0, 0, -1], [0, 0, -1], [-1, -1, -1]], np.int32),
    }
    prediction = np.repeat(template[None], 3, axis=0)
    lengths = np.linalg.norm(np.diff(prediction, axis=1), axis=-1)
    scenes, report = proposals.joint_body_row_scenes([body], prediction, lengths, LOWER, UPPER)
    assert scenes == [] and report["admissible_body_choices_by_slot"] == [0, 0, 0]
    # With broad bounds, implausible fixed segment lengths also reject each option.
    scenes, report = proposals.joint_body_row_scenes(
        [body], prediction, np.ones((3, 2)), LOWER, [200.0, 200.0, 200.0]
    )
    assert scenes == [] and report["admissible_body_choices_by_slot"] == [0, 0, 0]

"""World-to-image prediction through water, independently of tracking assets."""

from dataclasses import replace

import numpy as np
import pytest

from medaka3d.calib import CalibrationError, RefractiveCamera, rodrigues, triangulate


def camera(view="top", distorted=False):
    if view == "top":
        rotation = rodrigues([0.0, np.pi, 0.0])
        center = np.array([96.0, 96.0, 500.0])
        normal = np.array([0.0, 0.0, -1.0])
        point = np.array([0.0, 0.0, 186.0])
    else:
        rotation = rodrigues([np.pi / 2, 0.0, 0.0])
        center = np.array([96.0, -400.0, 93.0])
        normal = np.array([0.0, 1.0, 0.0])
        point = np.array([0.0, -4.0, 0.0])
    return RefractiveCamera(
        name=view, view=view, fx=2000.0, fy=2100.0, cx=600.0, cy=610.0,
        distortion=(-0.222, 0.204, 0.0044, 0.0019, -0.112) if distorted else (0,) * 5,
        rotation=rotation, center=center, interface_normal=normal, interface_point=point,
        glass_thickness_mm=4.0, index_air=1.0003, index_glass=1.52,
        index_water=1.3325, layer_count=1 if view == "top" else 2,
    )


@pytest.mark.parametrize("view", ["top", "side"])
@pytest.mark.parametrize("distorted", [False, True])
def test_ray_points_at_many_depths_return_to_the_original_pixel(view, distorted):
    cam = camera(view, distorted)
    rng = np.random.default_rng(328)
    pixels = rng.uniform([250, 260], [950, 960], size=(4, 5, 2))
    origins, directions = cam.rays(pixels)
    depths = np.array([1e-5, 1.0, 50.0, 180.0])
    points = origins[None] + depths[:, None, None, None] * directions[None]
    predicted = cam.project(points)
    assert predicted.shape == (4, 4, 5, 2)
    np.testing.assert_allclose(predicted, np.broadcast_to(pixels, predicted.shape), atol=2e-6,
                               rtol=0)


@pytest.mark.parametrize("view", ["top", "side"])
def test_matches_independent_scalar_snell_construction(view):
    cam = camera(view)
    normal = cam.interface_normal
    tangent = np.array([1.0, 0.0, 0.0])
    air_depth = (cam.interface_point - cam.center) @ normal
    glass_depth = 4.0 if view == "side" else 0.0
    for air_angle in [0.0, 0.15, 0.6, 1.25]:
        glass_angle = np.arcsin(cam.index_air * np.sin(air_angle) / cam.index_glass)
        water_angle = np.arcsin(cam.index_air * np.sin(air_angle) / cam.index_water)
        for water_depth in [0.0, 15.0, 180.0]:
            transverse = (air_depth * np.tan(air_angle)
                          + glass_depth * np.tan(glass_angle)
                          + water_depth * np.tan(water_angle))
            point = (cam.center + normal * (air_depth + glass_depth + water_depth)
                     + tangent * transverse)
            air_direction = normal * np.cos(air_angle) + tangent * np.sin(air_angle)
            expected_cam = cam.rotation @ air_direction
            expected = np.array([cam.fx, cam.fy]) * expected_cam[:2] / expected_cam[2]
            expected += [cam.cx, cam.cy]
            np.testing.assert_allclose(cam.project(point), expected, atol=2e-6, rtol=0)


def test_tilted_parallel_planes_and_distortion_still_roundtrip():
    cam = camera("side", distorted=True)
    normal = rodrigues([0.08, -0.13, 0.04]) @ cam.interface_normal
    cam = replace(cam, interface_normal=normal, interface_point=cam.center + 396 * normal)
    uv = np.array([[300.0, 850.0], [600.0, 610.0], [900.0, 350.0]])
    origins, directions = cam.rays(uv)
    np.testing.assert_allclose(cam.project(origins + 85.0 * directions), uv, atol=2e-6, rtol=0)


def test_equal_indices_reduce_to_pinhole_and_its_analytic_derivative():
    cam = replace(camera("side"), index_air=1.0, index_glass=1.0, index_water=1.0)
    point = np.array([70.0, 85.0, 40.0])
    q = cam.rotation @ (point - cam.center)
    expected = [cam.fx * q[0] / q[2] + cam.cx, cam.fy * q[1] / q[2] + cam.cy]
    np.testing.assert_allclose(cam.project(point), expected, atol=1e-8, rtol=0)
    derivative = (np.array([[cam.fx / q[2], 0, -cam.fx * q[0] / q[2] ** 2],
                            [0, cam.fy / q[2], -cam.fy * q[1] / q[2] ** 2]])
                  @ cam.rotation)
    delta = np.eye(3) * 1e-3
    numerical = ((cam.project(point + delta) - cam.project(point - delta)) / 2e-3).T
    np.testing.assert_allclose(numerical, derivative, atol=1e-7, rtol=0)


def test_refractive_derivative_is_stable_for_local_optimisation():
    cam = camera("top", distorted=True)
    point = np.array([42.0, 134.0, 20.0])
    derivatives = []
    for step in [1e-2, 1e-3]:
        delta = np.eye(3) * step
        derivatives.append(((cam.project(point + delta) - cam.project(point - delta))
                            / (2 * step)).T)
    np.testing.assert_allclose(*derivatives, atol=1e-7, rtol=0)
    movement = np.array([0.001, -0.002, 0.001])
    expected = cam.project(point) + derivatives[1] @ movement
    np.testing.assert_allclose(cam.project(point + movement), expected, atol=1e-7, rtol=0)


def test_refractive_images_triangulate_back_to_known_world_points():
    points = np.array([[20.0, 170.0, 5.0], [96.0, 96.0, 93.0], [180.0, 15.0, 185.0]])
    pairs = [cam.rays(cam.project(points)) for cam in [camera("top", True), camera("side", True)]]
    origins, directions = [np.stack(items, axis=1) for items in zip(*pairs, strict=True)]
    recovered, rms = triangulate(origins, directions)
    np.testing.assert_allclose(recovered, points, atol=1e-7, rtol=0)
    assert (rms < 1e-7).all()


def test_top_ignores_glass_and_side_models_it():
    point = np.array([35.0, 60.0, 45.0])
    top = camera()
    np.testing.assert_array_equal(
        top.project(point), replace(top, glass_thickness_mm=99).project(point)
    )
    side = camera("side")
    assert np.linalg.norm(side.project(point) - replace(side, layer_count=1).project(point)) > 0.01


def test_invalid_points_do_not_poison_valid_rows_or_emit_warnings():
    cam = camera()
    points = np.array([[70.0, 120.0, 50.0], [96.0, 96.0, 187.0],
                       [np.nan, 50.0, 80.0], [96.0, np.inf, 80.0],
                       [96.0, 96.0, 186.0]])
    with np.errstate(all="raise"):
        projected = cam.project(points)
    assert np.isfinite(projected[[0, 4]]).all()
    assert np.isnan(projected[1:4]).all()
    np.testing.assert_allclose(projected[4], [cam.cx, cam.cy], atol=1e-10, rtol=0)
    assert np.isnan(camera("side").project([96.0, -2.0, 93.0])).all()  # in glass


def test_behind_camera_and_camera_in_water_are_invalid():
    cam = camera()
    upwards = replace(cam, rotation=np.eye(3))
    assert np.isnan(upwards.project([96.0, 96.0, 50.0])).all()
    submerged = replace(cam, center=np.array([96.0, 96.0, 100.0]))
    assert np.isnan(submerged.project([96.0, 96.0, 50.0])).all()


def test_surface_point_cannot_require_total_internal_reflection():
    # An unusual but valid positive-index configuration exposes the boundary
    # case: a zero water distance must not bypass the water's critical angle.
    cam = replace(camera(), index_air=1.5, index_water=1.0)
    assert np.isnan(cam.project([1096.0, 96.0, 186.0])).all()
    assert np.isfinite(cam.project([196.0, 96.0, 186.0])).all()


def test_output_shape_and_invalid_calibration():
    cam = camera()
    assert cam.project(np.empty((2, 0, 3))).shape == (2, 0, 2)
    assert cam.project([96, 96, 50]).shape == (2,)
    for bad_shape in [1, [1, 2], np.empty((3, 4))]:
        with pytest.raises(ValueError, match="shape"):
            cam.project(bad_shape)
    with pytest.raises(CalibrationError, match="indices"):
        replace(cam, index_water=0).project([96, 96, 50])
    with pytest.raises(CalibrationError, match="thickness"):
        replace(camera("side"), glass_thickness_mm=-1).project([96, 96, 50])

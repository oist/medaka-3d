"""Batched admission versus frozen per-pose geometry, including missing evidence."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from medaka3d.types import VIEWS

_LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"three_body_support_batch_test_{name}",
                                                 _LAB / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_GEOMETRY = load_module("node_geometry")
_PREVIOUS = sys.modules.get("node_geometry")
sys.modules["node_geometry"] = _GEOMETRY
try:
    supported_nodes_batch = load_module("support_batch").supported_nodes_batch
finally:
    if _PREVIOUS is None:
        sys.modules.pop("node_geometry", None)
    else:
        sys.modules["node_geometry"] = _PREVIOUS


@pytest.fixture
def case(rig):
    rig = replace(rig, cameras=tuple(
        replace(camera, index_air=1.0003, index_glass=1.52, index_water=1.3325,
                layer_count=1 if camera.view == VIEWS[0] else 2,
                distortion=(-0.15, 0.07, 0.003, -0.002, -0.01))
        for camera in rig.cameras
    ))
    centers = np.array([[40., 50., 30.], [110., 85., 110.], [70., 150., 160.]])
    pose = centers[:, None] + np.array([[5., 0., 0.], [0., 0., 0.], [-12., 2., -1.]])[None]
    rng = np.random.default_rng(9113)
    points, plan = {}, np.empty((3, 3, 3), np.int32)
    for vi, view in enumerate(VIEWS):
        raw = rig.camera(view).project(pose)
        out = np.empty_like(raw)
        for node in range(3):
            order = rng.permutation(3)
            out[:, node] = raw[order, node]
            plan[:, vi, node] = np.argsort(order)
        points[view] = out
    rays = _GEOMETRY.RayObservations.from_points(rig, points)
    return rig, pose, plan, rays


def compare_reference(rig, poses, plans, rays, **kwargs):
    expected = list(zip(*[
        _GEOMETRY.supported_nodes(pose, plan, rays, rig, **kwargs)
        for pose, plan in zip(poses, plans, strict=True)
    ], strict=True))
    actual = supported_nodes_batch(poses, plans, rays, rig, **kwargs)
    for got, want in zip(actual, expected, strict=True):
        # Distance arithmetic also remains bitwise identical, including NaNs.
        np.testing.assert_array_equal(got, np.stack(want))
    return actual


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("batch", [1, 6])
def test_refractive_distorted_batch_matches_separate_calls(case, dtype, batch):
    rig, pose, plan, rays = case
    poses = np.broadcast_to(pose, (batch, 3, 3, 3)).copy().astype(dtype)
    if batch > 1:
        poses[1] += [0.12, -0.08, 0.04]
        poses[2, 0, 1] += [6., 8., 9.]  # Reject a displaced observed node.
        poses[3, 1, 0] = np.nan
        poses[4, 2] += [0., 0., 400.]  # Above top water interface: projection invalid.
    plans = np.broadcast_to(plan, poses.shape).copy()
    plans[-1, 1, :, 2] = -1
    _, observed, _ = compare_reference(rig, poses, plans, rays)
    assert observed[0, 0].all()
    assert not observed[-1, 1, 2]
    if batch > 1:
        assert not observed[2, 0, 1]
        assert not observed[3, 1, 0]


@pytest.mark.parametrize("thresholds", [(0.1, 12.), (20., 0.2), (2., 12.)])
def test_both_existing_admission_gates_are_preserved(case, thresholds):
    rig, pose, plan, rays = case
    poses = np.repeat(pose[None], 3, axis=0)
    poses[1] += [0.2, -0.1, 0.15]
    poses[2] += [4., -3., 5.]
    _, observed, _ = compare_reference(
        rig, poses, np.repeat(plan[None], 3, axis=0), rays,
        maximum_ray_mm=thresholds[0], maximum_pixel_error=thresholds[1])
    assert observed[0].all()
    assert not observed[2].any()


def test_missing_views_never_become_observed_background_or_latent_support(case):
    rig, pose, plan, rays = case
    points = dict(rays.points)
    points[VIEWS[1]] = np.empty((0, 3, 2))
    points[VIEWS[2]] = np.full((3, 3, 2), np.nan)
    plan[:, 1:] = -1
    rays = _GEOMETRY.RayObservations.from_points(rig, points)
    poses = np.repeat(pose[None], 6, axis=0)
    plans = np.repeat(plan[None], 6, axis=0)
    plans[-1] = -1
    sources, observed, _ = compare_reference(rig, poses, plans, rays)
    assert not observed.any()
    assert (sources[:, :, 1:] == -1).all()
    assert (sources[-1] == -1).all()


def test_batch_projects_once_per_camera_without_changing_projection(case):
    rig, pose, plan, rays = case
    rng = np.random.default_rng(9914)
    poses = pose[None] + rng.uniform(-0.01, 0.01, size=(6, 3, 3, 3))
    calls = []

    def camera(view):
        original = rig.camera(view)

        def project(value):
            calls.append((view, value.shape))
            return original.project(value)

        return SimpleNamespace(project=project)

    supported_nodes_batch(poses, np.repeat(plan[None], 6, axis=0), rays,
                          SimpleNamespace(camera=camera))
    assert calls == [(view, (6, 3, 3, 3)) for view in VIEWS]
    for view in VIEWS:
        cam = rig.camera(view)
        separate = np.stack([cam.project(p) for p in poses])
        np.testing.assert_allclose(cam.project(poses), separate, rtol=0, atol=1e-7)


def test_pixel_threshold_is_inclusive_without_rounding_or_tolerance_adjustment(case):
    rig, pose, plan, rays = case
    # Use the exact original numerical projection residual as the threshold;
    # moving it one representable float either way must preserve the oracle.
    poses = np.repeat(pose[None], 1, axis=0)
    shifted = {view: values.copy() for view, values in rays.points.items()}
    row = plan[0, 0, 0]
    shifted[VIEWS[0]][row, 0] += [12., 0.]
    rays = _GEOMETRY.RayObservations.from_points(rig, shifted)
    error = np.linalg.norm(rig.camera(VIEWS[0]).project(pose)[0, 0]
                           - shifted[VIEWS[0]][row, 0])
    for threshold in (np.nextafter(error, 0), error, np.nextafter(error, np.inf)):
        sources, _, _ = compare_reference(
            rig, poses, plan[None], rays, maximum_ray_mm=100.,
            maximum_pixel_error=threshold)
        assert (sources[0, 0, 0, 0] >= 0) == (threshold >= error)


def test_empty_batch_and_invalid_shapes(case):
    rig, pose, plan, rays = case
    result = supported_nodes_batch(np.empty((0, 3, 3, 3)),
                                   np.empty((0, 3, 3, 3), np.int32), rays, rig)
    assert [x.shape for x in result] == [(0, 3, 3, 3), (0, 3, 3), (0, 3, 3, 3)]
    with pytest.raises(ValueError, match="poses must have shape"):
        supported_nodes_batch(pose, plan[None], rays, rig)
    with pytest.raises(ValueError, match="plans must have shape"):
        supported_nodes_batch(pose[None], np.repeat(plan[None], 2, axis=0), rays, rig)

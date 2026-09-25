"""Geometric evidence must preserve nodes, camera order and missingness."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from medaka3d.types import VIEWS


@pytest.fixture(scope="module")
def metrics():
    path = (Path(__file__).resolve().parents[1] / "labs"
            / "20260916_multiview_node_review" / "geometry.py")
    if not path.exists():
        pytest.skip("multiview node review experiment has been retired")
    spec = importlib.util.spec_from_file_location("_multiview_node_geometry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.geometry_metrics


class PinholeCamera:
    def __init__(self, center):
        self.center = np.asarray(center, dtype=float)

    def rays(self, uv):
        assert np.isfinite(uv).all(), "invalid pixels should not reach camera solver"
        xy = (np.asarray(uv) - [400, 300]) / 600
        direction = np.concatenate([xy, np.ones((*xy.shape[:-1], 1))], axis=-1)
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        return np.broadcast_to(self.center, direction.shape).copy(), direction

    def project(self, xyz):
        assert np.isfinite(xyz).all(), "invalid world points should not reach camera solver"
        relative = np.asarray(xyz) - self.center
        return 600 * relative[..., :2] / relative[..., 2, None] + [400, 300]


class PinholeRig:
    views = VIEWS
    world_units = "mm"

    def __init__(self):
        self.cameras = [PinholeCamera(c) for c in ([-40, 0, 0], [40, 0, 0], [0, 40, 0])]

    def camera(self, view):
        return self.cameras[VIEWS.index(view)]


def world_pose():
    centers = np.array([[-12, 2, 90], [8, -10, 100], [18, 14, 110]], dtype=float)
    return centers[:, None] + np.array([[-5, 0, 0], [0, 0, 0], [10, 0, 0]])


def pixels(rig, pose):
    return np.stack([rig.camera(view).project(pose) for view in VIEWS], axis=-3)


def test_perfect_three_view_nodes_reconstruct_and_have_zero_residuals(metrics):
    rig, pose = PinholeRig(), world_pose()
    uv = pixels(rig, pose)
    result = metrics(rig, uv)
    np.testing.assert_allclose(result["pose"], pose, atol=1e-10)
    for key in ("ray_rms_mm", "reprojection_px", "leave_one_out_px", "bend_deg"):
        np.testing.assert_allclose(result[key], 0, atol=1e-6)
    np.testing.assert_allclose(result["mp_mm"], 5)
    np.testing.assert_allclose(result["pt_mm"], 10)
    np.testing.assert_allclose(result["mt_mm"], 15)
    assert result["observed"].all()
    assert (result["view_counts"] == 3).all()
    assert result["reprojection_px"].shape == (3, 3, 3)
    assert "frozen_reprojection_px" not in result


def test_one_bad_node_in_one_view_cannot_contaminate_other_nodes(metrics):
    rig, pose = PinholeRig(), world_pose()
    uv = pixels(rig, pose)
    uv[1, 2, 0, 0] += 30
    result = metrics(rig, uv)
    # Held-out view 2 predicts the original point using the untouched pair.
    assert result["leave_one_out_px"][1, 2, 0] == pytest.approx(30)
    assert result["ray_rms_mm"][1, 0] > 0.1
    assert result["reprojection_px"][1, 2, 0] > 1
    np.testing.assert_allclose(result["pose"][[0, 2]], pose[[0, 2]], atol=1e-10)
    np.testing.assert_allclose(result["pose"][1, 1:], pose[1, 1:], atol=1e-10)


def test_two_view_reconstruction_has_no_fabricated_leave_one_out_evidence(metrics):
    rig, pose = PinholeRig(), world_pose()
    uv = pixels(rig, pose)
    uv[0, 2, 1] = np.nan
    result = metrics(rig, uv)
    np.testing.assert_allclose(result["pose"], pose, atol=1e-10)
    assert result["view_counts"][0, 1] == 2
    assert result["observed"][0, 1]
    assert np.isnan(result["leave_one_out_px"][0, :, 1]).all()
    assert np.isnan(result["reprojection_px"][0, 2, 1])
    np.testing.assert_allclose(result["reprojection_px"][0, :2, 1], 0, atol=1e-10)


def test_missing_and_infinite_pixels_never_become_zero_cost_evidence(metrics):
    rig, pose = PinholeRig(), world_pose()
    uv = pixels(rig, pose)
    uv[0, :, 0] = np.nan
    uv[1, 1:, 2] = np.inf
    result = metrics(rig, uv)
    assert result["view_counts"][0, 0] == 0
    assert result["view_counts"][1, 2] == 1
    for slot, node in ((0, 0), (1, 2)):
        assert np.isnan(result["pose"][slot, node]).all()
        assert np.isnan(result["ray_rms_mm"][slot, node])
        assert not result["observed"][slot, node]
        assert np.isnan(result["reprojection_px"][slot, :, node]).all()
        assert np.isnan(result["leave_one_out_px"][slot, :, node]).all()
    assert np.isnan(result["mp_mm"][0])
    assert np.isnan(result["pt_mm"][1])
    assert np.isnan(result["bend_deg"][:2]).all()
    assert np.isfinite(result["pose"][2]).all()


def test_frozen_pose_projection_is_separate_and_never_overwrites_fitted_pose(metrics):
    rig, pose = PinholeRig(), world_pose()
    uv = pixels(rig, pose)
    frozen = pose.copy()
    frozen[:, 1, 1] += 4
    original_uv, original_frozen = uv.copy(), frozen.copy()
    result = metrics(rig, uv, frozen)
    np.testing.assert_allclose(result["pose"], pose, atol=1e-10)
    np.testing.assert_allclose(result["ray_rms_mm"], 0, atol=1e-10)
    assert (result["frozen_reprojection_px"][:, :, 1] > 10).all()
    np.testing.assert_allclose(result["frozen_reprojection_px"][:, :, [0, 2]], 0, atol=1e-10)
    np.testing.assert_array_equal(uv, original_uv)
    np.testing.assert_array_equal(frozen, original_frozen)


def test_bending_degrees_distinguishes_right_angle_fold_and_zero_segment(metrics):
    rig, pose = PinholeRig(), world_pose()
    pose[0, 2] = pose[0, 1] + [0, 10, 0]
    pose[1, 2] = pose[1, 1] + [-10, 0, 0]
    pose[2, 0] = pose[2, 1]
    result = metrics(rig, pixels(rig, pose))
    np.testing.assert_allclose(result["bend_deg"][:2], [90, 180], atol=1e-6)
    assert np.isnan(result["bend_deg"][2])
    np.testing.assert_allclose(result["mt_mm"][0], np.sqrt(125))


def test_arbitrary_batch_axes_and_empty_batch_are_preserved(metrics):
    rig = PinholeRig()
    pose = np.broadcast_to(world_pose(), (2, 4, 3, 3, 3)).copy()
    pose[1, :, :, :, 2] += 3
    result = metrics(rig, pixels(rig, pose), pose)
    assert result["pose"].shape == (2, 4, 3, 3, 3)
    assert result["reprojection_px"].shape == (2, 4, 3, 3, 3)
    assert result["view_counts"].shape == (2, 4, 3, 3)
    assert result["bend_deg"].shape == (2, 4, 3)
    np.testing.assert_allclose(result["pose"], pose, atol=1e-10)
    empty = metrics(rig, np.empty((0, 3, 3, 3, 2)))
    assert empty["pose"].shape == (0, 3, 3, 3)
    assert empty["leave_one_out_px"].shape == (0, 3, 3, 3)


def test_degenerate_parallel_cameras_do_not_claim_observed_points(metrics):
    rig = PinholeRig()
    rig.cameras = [PinholeCamera([0, 0, 0]) for _ in VIEWS]
    result = metrics(rig, pixels(rig, world_pose()))
    assert (result["view_counts"] == 3).all()
    assert not result["observed"].any()
    assert np.isnan(result["ray_rms_mm"]).all()
    assert np.isnan(result["pose"]).all()
    assert np.isnan(result["leave_one_out_px"]).all()


def test_axis_and_calibration_mismatch_fail_loudly(metrics):
    rig = PinholeRig()
    with pytest.raises(ValueError, match="uv must"):
        metrics(rig, np.zeros((3, 3, 2)))
    uv = pixels(rig, world_pose())
    with pytest.raises(ValueError, match="frozen_pose"):
        metrics(rig, uv, np.zeros((3, 3)))
    rig.views = VIEWS[::-1]
    with pytest.raises(ValueError, match="view order"):
        metrics(rig, uv)
    rig.views = VIEWS
    rig.world_units = "m"
    with pytest.raises(ValueError, match="world_units"):
        metrics(rig, uv)


def test_production_refractive_cameras_preserve_missing_nodes(metrics, rig):
    cameras = tuple(
        replace(
            camera,
            index_air=1.0003,
            index_glass=1.52,
            index_water=1.3325,
            layer_count=1 if camera.view == "top_cam" else 2,
            distortion=(-0.10, 0.04, 0.002, -0.001, 0.0),
        )
        for camera in rig.cameras
    )
    refractive = replace(rig, cameras=cameras)
    pose = world_pose() + np.array([80, 80, 0])
    uv = pixels(refractive, pose)
    uv[1, 2, 2] = np.nan
    result = metrics(refractive, uv)
    np.testing.assert_allclose(result["pose"], pose, atol=1e-6)
    np.testing.assert_allclose(result["ray_rms_mm"], 0, atol=1e-6)
    assert np.isnan(result["leave_one_out_px"][1, :, 2]).all()
    finite_errors = result["leave_one_out_px"][np.isfinite(result["leave_one_out_px"])]
    np.testing.assert_allclose(finite_errors, 0, atol=1e-6)

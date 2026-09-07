"""Refractive geometry, pinned.

These are hermetic: every camera here is synthesised in ``tmp_path``. The check
that the *real* promoted calibration is loaded and used correctly is
``tools/c02_tracking_probe.py``, which reproduces the calibration's own
point-to-ray accuracy on its own held-out evidence — that needs the bucket and so
cannot be a unit test.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from medaka3d.calib import (
    CalibrationError,
    RefractiveCamera,
    load_rig,
    refract,
    rodrigues,
    triangulate,
)


def bundle(**overrides) -> dict:
    base = {
        "name": "cam0",
        "fx": 2000.0,
        "fy": 2000.0,
        "cx": 600.0,
        "cy": 600.0,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "k5": 0.0,
        "k6": 0.0,
        "rotation_vector": [0.0, np.pi, 0.0],
        "camera_center": [96.0, 96.0, 500.0],
        "interface_normal": [0.0, 0.0, -1.0],
        "interface_point": [0.0, 0.0, 186.0],
        "glass_thickness_mm": 4.0,
        "index_air": 1.0,
        "index_glass": 1.0,
        "index_water": 1.0,
        "layer_count": 1,
    }
    return base | overrides


def write_rig(tmp_path, camera_overrides=(), manifest_overrides=None):
    names = ["cam0", "cam1", "cam2"]
    views = ["top_cam", "side1_cam", "side2_cam"]
    poses = [
        {"rotation_vector": [0.0, np.pi, 0.0], "camera_center": [96.0, 96.0, 500.0],
         "interface_normal": [0.0, 0.0, -1.0], "interface_point": [0.0, 0.0, 186.0],
         "layer_count": 1},
        {"rotation_vector": [np.pi / 2, 0.0, 0.0], "camera_center": [96.0, -400.0, 93.0],
         "interface_normal": [0.0, 1.0, 0.0], "interface_point": [0.0, -4.0, 0.0],
         "layer_count": 2},
        {"rotation_vector": [1.2091996, -1.2091996, 1.2091996],
         "camera_center": [500.0, 96.0, 93.0],
         "interface_normal": [-1.0, 0.0, 0.0], "interface_point": [196.0, 0.0, 0.0],
         "layer_count": 2},
    ]
    files = []
    for name, view, pose in zip(names, views, poses, strict=True):
        raw = bundle(name=name, **pose)
        for override in camera_overrides:
            if override.get("name") == name:
                raw |= {k: v for k, v in override.items() if k != "name"}
        path = tmp_path / f"{name}_bundle_refractive.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        files.append({"camera": name, "view": view, "path": path.name})
    manifest = {
        "backend": "refractive",
        "world_units": "mm",
        "camera_order": names,
        "view_order": views,
        "model_files": files,
    } | (manifest_overrides or {})
    (tmp_path / "calibration_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


class TestRodrigues:
    def test_pi_about_y_matches_the_promoted_top_camera(self):
        # cam0's real rotation vector is a pi-rotation about +y. Under
        # x_cam = R @ x_world + t that puts its optical axis, R's third row,
        # along -z: looking down into the tank. If the convention were the
        # transpose this test would fail, and every 3D point would be wrong in a
        # way that still looked like a fish.
        r = rodrigues([0.0, np.pi, 0.0])
        assert np.allclose(r, np.diag([-1.0, 1.0, -1.0]), atol=1e-12)
        assert np.allclose(r[2, :], [0.0, 0.0, -1.0], atol=1e-12)

    def test_zero_vector_is_identity(self):
        assert np.allclose(rodrigues([0.0, 0.0, 0.0]), np.eye(3))

    def test_is_a_rotation(self):
        r = rodrigues([0.3, -1.1, 0.7])
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(r), 1.0)


class TestRefract:
    normal = np.array([0.0, 0.0, -1.0])

    def test_normal_incidence_does_not_bend(self):
        out = refract(self.normal, self.normal, 1.0 / 1.3325)
        assert np.allclose(out, self.normal)

    def test_obeys_snell(self):
        theta_i = np.deg2rad(35.0)
        d = np.array([np.sin(theta_i), 0.0, -np.cos(theta_i)])
        eta = 1.0003 / 1.3325
        out = refract(d, self.normal, eta)
        theta_t = np.arccos(float(out @ self.normal))
        assert np.isclose(np.sin(theta_i) * 1.0003, np.sin(theta_t) * 1.3325, atol=1e-12)
        assert theta_t < theta_i  # into the denser medium, so it bends toward the normal

    def test_total_internal_reflection_is_nan_not_clamped(self):
        # Water -> air past the critical angle. A clamped ray would produce a
        # confident 3D point from light that never reached the camera.
        theta = np.deg2rad(75.0)
        d = np.array([np.sin(theta), 0.0, -np.cos(theta)])
        assert np.isnan(refract(d, self.normal, 1.3325 / 1.0003)).all()

    def test_vectorises(self):
        d = np.tile(self.normal, (4, 3, 1))
        assert refract(d, self.normal, 0.75).shape == (4, 3, 3)


class TestUndistort:
    def test_inverts_the_brown_model(self):
        # The real cam0 coefficients, so the inverse is exercised where it is used.
        raw = bundle(k1=-0.222, k2=0.204, p1=0.0044, p2=0.0019, k3=-0.112)
        camera = _camera_from(raw)
        ideal = np.array([[0.10, -0.05], [-0.22, 0.31], [0.0, 0.0]])
        x, y = ideal[:, 0], ideal[:, 1]
        r2 = x * x + y * y
        k1, k2, p1, p2, k3 = camera.distortion
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        pixels = np.stack([xd * camera.fx + camera.cx, yd * camera.fy + camera.cy], axis=-1)
        assert np.allclose(camera.undistort(pixels), ideal, atol=1e-10)


def _camera_from(raw: dict, view: str = "top_cam") -> RefractiveCamera:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "b.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return RefractiveCamera.from_bundle(path, view)


class TestRays:
    def test_without_refraction_the_ray_is_a_straight_line_from_the_centre(self):
        # index_air == index_water, so the interface is optically invisible and
        # the ray must still pass through the camera centre.
        camera = _camera_from(bundle())
        uv = np.array([[600.0, 600.0], [900.0, 400.0]])
        origin, direction = camera.rays(uv)
        offset = camera.center - origin
        cross = np.cross(offset, direction)
        assert np.allclose(np.linalg.norm(cross, axis=-1), 0.0, atol=1e-9)
        assert np.allclose(np.linalg.norm(direction, axis=-1), 1.0)

    def test_principal_ray_of_the_top_camera_points_down(self):
        camera = _camera_from(bundle())
        _, direction = camera.rays(np.array([[600.0, 600.0]]))
        assert np.allclose(direction[0], [0.0, 0.0, -1.0], atol=1e-12)

    def test_refraction_bends_an_oblique_ray_toward_the_normal(self):
        straight = _camera_from(bundle())
        bent = _camera_from(bundle(index_air=1.0003, index_water=1.3325))
        uv = np.array([[1000.0, 600.0]])
        _, d_straight = straight.rays(uv)
        _, d_bent = bent.rays(uv)
        down = np.array([0.0, 0.0, -1.0])
        assert float(d_bent[0] @ down) > float(d_straight[0] @ down)

    def test_nan_pixels_give_nan_rays(self):
        camera = _camera_from(bundle())
        origin, direction = camera.rays(np.array([[np.nan, 600.0]]))
        assert np.isnan(origin).any() and np.isnan(direction).any()

    def test_glass_layer_is_only_applied_when_the_bundle_says_so(self):
        # The top camera images through a free surface; applying the side
        # cameras' 4 mm pane there would shift every depth.
        one = _camera_from(bundle(index_air=1.0003, index_glass=1.52, index_water=1.3325))
        two = _camera_from(
            bundle(index_air=1.0003, index_glass=1.52, index_water=1.3325, layer_count=2)
        )
        uv = np.array([[1000.0, 600.0]])
        assert not np.allclose(one.rays(uv)[0], two.rays(uv)[0])


class TestTriangulate:
    def test_recovers_a_point_seen_by_three_cameras(self, tmp_path):
        rig = load_rig(write_rig(tmp_path))
        target = np.array([70.0, 120.0, 45.0])
        origins, directions = [], []
        for camera in rig.cameras:
            direction = target - camera.center
            origins.append(camera.center)
            directions.append(direction / np.linalg.norm(direction))
        xyz, rms = triangulate(np.asarray(origins), np.asarray(directions))
        assert np.allclose(xyz, target, atol=1e-9)
        assert rms < 1e-9

    def test_residual_grows_with_a_wrong_correspondence(self, tmp_path):
        rig = load_rig(write_rig(tmp_path))
        a, b = np.array([70.0, 120.0, 45.0]), np.array([120.0, 60.0, 140.0])
        origins, directions = [], []
        for camera, target in zip(rig.cameras, [a, a, b], strict=True):
            direction = target - camera.center
            origins.append(camera.center)
            directions.append(direction / np.linalg.norm(direction))
        _, rms = triangulate(np.asarray(origins), np.asarray(directions))
        assert rms > 10.0  # mm — a wrong triple is not a near miss

    def test_parallel_rays_are_nan_not_a_least_squares_guess(self):
        origins = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
        directions = np.tile([1.0, 0.0, 0.0], (3, 1))
        xyz, _ = triangulate(origins, directions)
        assert np.isnan(xyz).all()

    def test_nan_input_propagates(self):
        origins = np.zeros((3, 3))
        directions = np.tile([1.0, 0.0, 0.0], (3, 1)).astype(float)
        directions[1] = np.nan
        xyz, _ = triangulate(origins, directions)
        assert np.isnan(xyz).all()

    def test_batches(self):
        origins = np.zeros((5, 7, 3, 3))
        directions = np.zeros((5, 7, 3, 3))
        directions[..., 0, 0] = 1.0
        directions[..., 1, 1] = 1.0
        directions[..., 2, 2] = 1.0
        xyz, rms = triangulate(origins, directions)
        assert xyz.shape == (5, 7, 3) and rms.shape == (5, 7)

    def test_rejects_a_single_ray(self):
        with pytest.raises(ValueError, match="at least two rays"):
            triangulate(np.zeros((1, 3)), np.tile([1.0, 0.0, 0.0], (1, 1)))

    def test_a_masked_ray_is_left_out_instead_of_poisoning_the_row(self, tmp_path):
        """A camera that lost this animal must not cost the other two cameras."""
        rig = load_rig(write_rig(tmp_path))
        target = np.array([70.0, 120.0, 45.0])
        origins, directions = [], []
        for camera in rig.cameras:
            direction = target - camera.center
            origins.append(camera.center)
            directions.append(direction / np.linalg.norm(direction))
        origins, directions = np.asarray(origins), np.asarray(directions)
        directions[2] = np.nan

        xyz, rms = triangulate(origins, directions)
        assert np.isnan(xyz).all()  # unmasked, a NaN ray is still fatal

        valid = np.array([True, True, False])
        xyz, rms = triangulate(origins, directions, valid=valid)
        assert np.allclose(xyz, target, atol=1e-9)
        assert rms < 1e-9

    def test_fewer_than_two_valid_rays_is_nan(self):
        origins = np.zeros((3, 3))
        directions = np.eye(3)
        xyz, _ = triangulate(origins, directions, valid=np.array([True, False, False]))
        assert np.isnan(xyz).all()

    def test_the_residual_averages_over_the_rays_that_took_part(self):
        """Two rays 2 mm apart give 1 mm rms whether or not a third exists on paper."""
        origins = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 50.0]])
        directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        _, rms = triangulate(origins, directions, valid=np.array([True, True, False]))
        assert np.isclose(rms, 1.0)

    def test_valid_must_match_the_ray_shape(self):
        with pytest.raises(ValueError, match="valid"):
            triangulate(np.zeros((4, 3, 3)), np.zeros((4, 3, 3)), valid=np.ones((4, 2), bool))


class TestLoadRig:
    def test_loads_views_in_manifest_order(self, tmp_path):
        rig = load_rig(write_rig(tmp_path))
        assert rig.views == ("top_cam", "side1_cam", "side2_cam")
        assert rig.camera("side2_cam").name == "cam2"

    def test_refuses_a_non_refractive_backend(self, tmp_path):
        # The archived polynomial backend is a historical-reproduction asset and
        # must never be an automatic fallback.
        path = write_rig(tmp_path, manifest_overrides={"backend": "polynomial"})
        with pytest.raises(CalibrationError, match="not 'refractive'"):
            load_rig(path)

    def test_refuses_a_reordered_view_list(self, tmp_path):
        # side1 and side2 share every frame index; swapping them here would be
        # silent and would corrupt every reconstruction.
        path = write_rig(
            tmp_path, manifest_overrides={"view_order": ["top_cam", "side2_cam", "side1_cam"]}
        )
        with pytest.raises(CalibrationError):
            load_rig(path)

    def test_refuses_rational_distortion_terms(self, tmp_path):
        path = write_rig(tmp_path, camera_overrides=[{"name": "cam1", "k4": 0.01}])
        with pytest.raises(CalibrationError, match="k4"):
            load_rig(path)

    def test_refuses_a_non_unit_interface_normal(self, tmp_path):
        path = write_rig(
            tmp_path, camera_overrides=[{"name": "cam2", "interface_normal": [0, 0, -2]}]
        )
        with pytest.raises(CalibrationError, match="unit vector"):
            load_rig(path)

    def test_rays_demands_every_view_exactly_once(self, tmp_path):
        rig = load_rig(write_rig(tmp_path))
        with pytest.raises(CalibrationError, match="missing"):
            rig.rays({"top_cam": np.zeros((1, 2)), "side1_cam": np.zeros((1, 2))})

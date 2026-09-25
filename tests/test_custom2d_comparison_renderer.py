"""Review pixels must distinguish 2D detections, 3D projection and missingness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def renderer():
    path = (
        Path(__file__).resolve().parents[1]
        / "labs"
        / "20260917_custom2d_comparison"
        / "render_reviews.py"
    )
    if not path.is_file():
        pytest.skip("Custom 2D comparison experiment was retired")
    spec = importlib.util.spec_from_file_location("_custom2d_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lines_follow_saved_2d_not_projected_3d(renderer):
    image = np.zeros((140, 140, 3), np.uint8)
    measured, fitted = np.full((3, 3, 2), np.nan), np.full((3, 3, 2), np.nan)
    measured[0] = [[20, 30], [70, 30], [110, 30]]
    fitted[0] = [[20, 100], [70, 100], [110, 100]]
    original = measured.copy()
    renderer.overlay(image, measured, fitted, 1.0, np.zeros(2))
    assert image[30, 45].any(), "2D M-P line should be visible"
    assert not image[100, 45].any(), "No artificial 3D connecting line is requested"
    assert image[100, 20].any(), "3D projection should have a cross"
    np.testing.assert_array_equal(measured, original)


def test_projection_preserves_missing_nodes_and_passes_only_finite_world_points(renderer):
    class Camera:
        def project(self, xyz):
            assert np.isfinite(xyz).all()
            return xyz[:, :2] + 100

    class Rig:
        def camera(self, view):
            return Camera()

    pose = np.full((2, 3, 3, 3), np.nan)
    pose[0, 1, 2] = [10, 20, 30]
    arms = {name: {"pose": pose.copy()} for name in renderer.CONDITIONS}
    values = renderer.project_arms(Rig(), arms)
    for projected in values.values():
        assert projected.shape == (2, 3, 3, 3, 2)
        np.testing.assert_array_equal(projected[0, 1, :, 2], [[110, 120]] * 3)
        assert np.isfinite(projected).all(-1).sum() == 3
        assert np.isnan(projected[1]).all()


def test_common_crop_includes_all_slots_and_all_conditions(renderer):
    arms, projected = {}, {}
    for column, condition in enumerate(renderer.CONDITIONS):
        uv = np.full((2, 3, 3, 3, 2), np.nan)
        uv[:, column, :, 1] = [[100, 200], [600, 700], [1100, 1000]][column]
        arms[condition] = {"uv": uv}
        projected[condition] = np.full_like(uv, np.nan)
    crops = renderer.common_crops(arms, projected, np.arange(2))
    for crop in crops.values():
        assert crop[0] <= 100 and crop[1] <= 200
        assert crop[2] > 1100 and crop[3] > 1000
        assert all(0 <= value <= 1200 for value in crop)


def test_three_equal_arms_have_identical_camera_cells(renderer):
    uv = np.full((1, 3, 3, 3, 2), np.nan)
    uv[0, 0, :, :] = [[450, 510], [470, 510], [510, 515]]
    arms = {condition: {"uv": uv.copy()} for condition in renderer.CONDITIONS}
    projected = {condition: uv.copy() for condition in renderer.CONDITIONS}
    images = {
        view: np.full((1200, 1200, 3), 70 + vi * 20, np.uint8)
        for vi, view in enumerate(renderer.VIEWS)
    }
    crops = renderer.common_crops(arms, projected, np.array([0]))
    canvas = renderer.render_frame(
        images, arms, projected, 0, 76791, crops, "synthetic", "final", "renderer test"
    )
    assert canvas.shape == renderer.CANVAS_SHAPE
    for row in range(3):
        first = renderer.HEADER + row * renderer.ROW_HEIGHT
        cells = [
            canvas[
                first : first + renderer.ROW_HEIGHT,
                col * renderer.WIDTH : (col + 1) * renderer.WIDTH,
            ]
            for col in range(3)
        ]
        np.testing.assert_array_equal(cells[0], cells[1])
        np.testing.assert_array_equal(cells[0], cells[2])


def test_selected_native_frames_are_never_silently_nearest_matched(renderer):
    frames = np.arange(100, 110)
    np.testing.assert_array_equal(
        renderer.selected_indices(frames, np.array([100, 103, 109])), [0, 3, 9]
    )
    with pytest.raises(ValueError, match="absent"):
        renderer.selected_indices(frames, np.array([100, 110]))
    with pytest.raises(ValueError, match="increase"):
        renderer.selected_indices(frames, np.array([100, 100]))

"""Optical-flow evidence must follow images, native time, and input coordinates."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "labs/20260917_clean_tracklets/flow.py"
SPEC = importlib.util.spec_from_file_location("clean_flow_for_test", MODULE_PATH)
flow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flow)


def _translated_video(path, *, textured=True):
    rng = np.random.default_rng(341)
    image = rng.integers(20, 230, (160, 160), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0.5) if textured else np.full_like(image, 100)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 166, (160, 160))
    assert writer.isOpened()
    for frame in range(9):
        matrix = np.float32([[1, 0, 2 * frame], [0, 1, frame]])
        shifted = cv2.warpAffine(image, matrix, (160, 160), borderMode=cv2.BORDER_REFLECT)
        writer.write(cv2.cvtColor(shifted, cv2.COLOR_GRAY2BGR))
    writer.release()


def _points(frames):
    base = np.array([[55, 60], [80, 70], [100, 100]], dtype=float)
    return base[None] + np.asarray(frames)[:, None, None] * np.array([2, 1])[None, None]


def test_translation_uses_actual_adjacent_frames_and_preserves_rows(tmp_path):
    path = tmp_path / "moving.avi"
    _translated_video(path)
    # A missing detection frame must not turn t->t+1 into t->next detection.
    # Nonmonotone order and duplicate frame rows must also remain unchanged.
    frames = np.array([5, 1, 5, 7], dtype=np.int64)
    points = _points(frames)
    result = flow.extract_flow(path, frames, points)
    assert result["valid_forward"].all()
    assert result["valid_backward"].all()
    np.testing.assert_allclose(result["predicted_forward"], points + np.array([2, 1]), atol=0.22)
    np.testing.assert_allclose(result["predicted_backward"], points - [2, 1], atol=0.22)
    assert result["diagnostics"]["decoded_range_half_open"] == [0, 9]
    assert result["diagnostics"]["decoded_native_frames"] == 9
    assert result["metadata"]["identity_labels_used"] is False


def test_textureless_images_cannot_claim_reliable_motion(tmp_path):
    path = tmp_path / "flat.avi"
    _translated_video(path, textured=False)
    result = flow.extract_flow(path, np.array([3]), _points([3]))
    for direction in ("forward", "backward"):
        assert not result[f"valid_{direction}"].any()
        assert np.isnan(result[f"predicted_{direction}"]).all()


def test_boundaries_invalid_coordinates_and_empty_input(tmp_path):
    path = tmp_path / "bounds.avi"
    _translated_video(path)
    frames = np.array([-1, 0, 8, 9, 4])
    points = _points(frames)
    points[4, 0] = np.nan
    points[4, 1] = [-100, 40]
    result = flow.extract_flow(path, frames, points)
    assert result["valid_forward"][1].all()
    assert not result["valid_backward"][1].any()
    assert result["valid_backward"][2].all()
    assert not result["valid_forward"][2].any()
    for direction in ("forward", "backward"):
        valid = result[f"valid_{direction}"]
        assert not valid[[0, 3]].any()
        assert not valid[4, :2].any()
        assert np.isnan(result[f"predicted_{direction}"][~valid]).all()
    empty = flow.extract_flow(path, np.array([], dtype=int), np.empty((0, 3, 2)))
    assert empty["predicted_forward"].shape == (0, 3, 2)
    assert empty["diagnostics"]["decoded_native_frames"] == 0


def test_cache_checks_coordinates_and_video_signature(tmp_path, monkeypatch):
    path, cache = tmp_path / "cache_source.avi", tmp_path / "flow.npz"
    _translated_video(path)
    frames, points = np.array([3]), _points([3])
    first = flow.extract_flow(path, frames, points, cache_path=cache)
    assert not first["metadata"]["cache"]["hit"]
    assert cache.with_name("flow_flow.provenance.json").exists()
    real_extract = flow._extract

    def no_decode(*_args):
        pytest.fail("Matching flow cache should avoid decoding the video")

    monkeypatch.setattr(flow, "_extract", no_decode)
    cached = flow.extract_flow(path, frames, points.copy(), cache_path=cache)
    assert cached["metadata"]["cache"]["hit"]
    np.testing.assert_array_equal(cached["predicted_forward"], first["predicted_forward"])
    monkeypatch.setattr(flow, "_extract", real_extract)
    changed_points = points + np.array([1, 0])
    changed = flow.extract_flow(path, frames, changed_points, cache_path=cache)
    assert not changed["metadata"]["cache"]["hit"]
    assert (changed["metadata"]["input_signature"]["points_sha256"]
            != first["metadata"]["input_signature"]["points_sha256"])
    np.testing.assert_allclose(
        changed["predicted_forward"], changed_points + np.array([2, 1]), atol=0.22
    )
    _translated_video(path, textured=False)
    flat = flow.extract_flow(path, frames, changed_points, cache_path=cache)
    assert not flat["metadata"]["cache"]["hit"]
    assert not flat["valid_forward"].any()


def test_wrong_input_shapes_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="integer"):
        flow.extract_flow(tmp_path / "unused.avi", np.array([1.5]), _points([1]))
    with pytest.raises(ValueError, match="shape"):
        flow.extract_flow(tmp_path / "unused.avi", np.array([1]), np.zeros((1, 2, 2)))

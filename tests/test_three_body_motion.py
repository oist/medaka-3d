"""Image-motion evidence is geometric support, never an inferred identity label."""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from medaka3d.types import VIEWS

MODULE = Path(__file__).resolve().parents[1] / "labs/20260909_three_body/motion.py"
SPEC = importlib.util.spec_from_file_location("_three_body_motion_test", MODULE)
motion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = motion
SPEC.loader.exec_module(motion)


@dataclass
class OrthographicCamera:
    axes: tuple[int, int]

    def project(self, xyz):
        return np.asarray(xyz)[..., self.axes]

    def rays(self, uv):
        uv = np.asarray(uv)
        origins = np.zeros((*uv.shape[:-1], 3))
        origins[..., self.axes] = uv
        directions = np.zeros_like(origins)
        normal = next(iter(set(range(3)) - set(self.axes)))
        directions[..., normal] = 1
        origins[~np.isfinite(uv).all(axis=-1)] = np.nan
        return origins, directions


class OrthographicRig:
    def __init__(self):
        self.cameras = dict(
            zip(VIEWS, map(OrthographicCamera, ((0, 1), (0, 2), (1, 2))), strict=True)
        )

    def camera(self, view):
        return self.cameras[view]


class ImageFrame(SimpleNamespace):
    @property
    def detections(self):
        raise AssertionError("motion must not inspect current or past SLEAP rows")


@pytest.fixture
def textured_translation():
    rig = OrthographicRig()
    pose = np.array([[45.0, 48.0, 50.0], [95.0, 100.0, 92.0], [150.0, 145.0, 148.0]])[:, None]
    pose = pose + np.array([[8.0, 6.0, 4.0], [0.0, 0.0, 0.0], [-14.0, -10.0, -6.0]])[None]
    shift = np.array([1.5, -2.0, 0.7])
    images, targets, masks, shifted_masks = {}, {}, {}, {}
    for index, view in enumerate(VIEWS):
        projected = rig.camera(view).project(pose)
        support = np.zeros((3, 200, 200), np.uint8)
        for fish in range(3):
            points = np.rint(projected[fish]).astype(np.int32)
            cv2.polylines(support[fish], [points], False, 1, 16)
            for point in points:
                cv2.circle(support[fish], tuple(point), 8, 1, -1)
        rng = np.random.default_rng(400 + index)
        texture = rng.integers(30, 240, (200, 200), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0.5)
        images[view] = np.where(support.any(axis=0), texture, 15).astype(np.uint8)
        dx, dy = rig.camera(view).project(shift)
        transform = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        targets[view] = cv2.warpAffine(images[view], transform, (200, 200), borderValue=15)
        masks[view] = support.astype(bool)
        shifted_masks[view] = np.stack(
            [
                cv2.warpAffine(mask, transform, (200, 200), flags=cv2.INTER_NEAREST).astype(bool)
                for mask in support
            ]
        )
    previous = ImageFrame(frame_idx=10, images_bgr=images, masks=masks)
    current = ImageFrame(frame_idx=11, images_bgr=targets, masks=shifted_masks)
    return rig, pose, shift, previous, current


def test_textured_translation_recovers_3d_motion_without_sleap_or_identity(textured_translation):
    rig, pose, shift, previous, current = textured_translation
    result = motion.FlowObservations.from_frames(previous, current, rig, pose)
    assert result.node_valid.all()
    np.testing.assert_allclose(result.pose_xyz, pose + shift, atol=0.25)
    assert result.valid.all()
    assert (result.feature_count >= 3).all()
    assert np.nanmax(result.fb_error_px) < 1
    assert (result.pose_quality > 0).all()
    assert not result.pose_xyz.flags.writeable
    np.testing.assert_array_equal(result.used_views.sum(axis=1), 3)


def test_blank_texture_and_missing_masks_are_neutral(textured_translation):
    rig, pose, _, previous, current = textured_translation
    for frame in (previous, current):
        frame.images_bgr = {view: np.full((200, 200), 30, np.uint8) for view in VIEWS}
    result = motion.FlowObservations.from_frames(previous, current, rig, pose)
    assert not result.valid.any() and not result.node_valid.any()
    assert np.isnan(result.pose_xyz).all() and (result.quality == 0).all()
    previous.masks = dict.fromkeys(VIEWS)
    result = motion.FlowObservations.from_frames(previous, current, rig, pose)
    assert not result.valid.any()


def test_one_inconsistent_view_is_excluded_by_multiview_geometry(textured_translation):
    rig, pose, shift, previous, current = textured_translation
    view = VIEWS[2]
    transform = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, 7.0]])
    current.images_bgr[view] = cv2.warpAffine(
        previous.images_bgr[view], transform, (200, 200), borderValue=15
    )
    current.masks[view] = np.stack(
        [
            cv2.warpAffine(
                mask.astype(np.uint8), transform, (200, 200), flags=cv2.INTER_NEAREST
            ).astype(bool)
            for mask in previous.masks[view]
        ]
    )
    result = motion.FlowObservations.from_frames(previous, current, rig, pose)
    assert result.valid[:, 2].any(), "the third view has valid 2D tracks but inconsistent geometry"
    assert result.node_valid.all()
    assert not result.used_views[:, 2].any()
    np.testing.assert_allclose(result.pose_xyz, pose + shift, atol=0.3)


def test_forward_backward_inconsistency_rejects_apparent_flow(textured_translation, monkeypatch):
    rig, pose, _, previous, current = textured_translation
    actual = cv2.calcOpticalFlowPyrLK
    calls = 0

    def corrupted_return(*args, **kwargs):
        nonlocal calls
        calls += 1
        points, status, error = actual(*args, **kwargs)
        if calls % 2 == 0 and points is not None:
            points = points + np.array([8.0, 0.0], dtype=np.float32)
        return points, status, error

    monkeypatch.setattr(motion.cv2, "calcOpticalFlowPyrLK", corrupted_return)
    result = motion.FlowObservations.from_frames(previous, current, rig, pose)
    assert not result.valid.any() and not result.node_valid.any()


def test_overlapping_previous_bodies_cannot_duplicate_the_same_texture(textured_translation):
    rig, pose, _, previous, current = textured_translation
    pose[1] = pose[0]
    result = motion.FlowObservations.from_frames(previous, current, rig, pose)
    assert not result.valid[:2].any()
    assert not result.node_valid[:2].any()
    assert result.node_valid[2].all()


def test_native_frame_gap_is_not_silently_treated_as_one_step(textured_translation):
    rig, pose, _, previous, current = textured_translation
    current.frame_idx += 1
    with pytest.raises(ValueError, match="consecutive"):
        motion.FlowObservations.from_frames(previous, current, rig, pose)


def test_failed_nonfinite_forward_points_are_not_sent_to_backward_lk(
    textured_translation, monkeypatch
):
    rig, pose, _, previous, current = textured_translation

    def failed_forward(before, after, points, *args, **kwargs):
        assert np.isfinite(points).all(), "a failed forward track reached backward LK"
        return (
            np.full_like(points, np.nan),
            np.ones((len(points), 1), np.uint8),
            np.zeros((len(points), 1), np.float32),
        )

    monkeypatch.setattr(motion.cv2, "calcOpticalFlowPyrLK", failed_forward)
    result = motion.FlowObservations.from_frames(previous, current, rig, pose)
    assert not result.valid.any() and not result.node_valid.any()

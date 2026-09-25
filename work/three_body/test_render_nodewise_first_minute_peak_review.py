"""Focused new multicamera seek, native-row and drawing checks; synthetic only."""

import io
from pathlib import Path

import numpy as np
import pytest
import render_nodewise_first_minute_peak_review as m


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    for module in (m, m.prep):
        monkeypatch.setattr(module, "WORK", tmp_path)
        monkeypatch.setattr(module, "REPO", tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("real input access forbidden")

    monkeypatch.setattr(m, "load_context", forbidden)
    monkeypatch.setattr(m.prep, "load_context", forbidden)
    monkeypatch.setattr(m.prep, "video_guard", forbidden)
    monkeypatch.setattr(m.calib, "load_rig", forbidden)
    monkeypatch.setattr(m.cv2, "VideoCapture", forbidden)
    actual_load = m.np.load

    def synthetic_load(value, *args, **kwargs):
        assert isinstance(value, io.BytesIO) or Path(value).is_relative_to(tmp_path)
        return actual_load(value, *args, **kwargs)

    monkeypatch.setattr(m.np, "load", synthetic_load)


class Capture:
    def __init__(self, images, fault=None):
        self.images, self.fault = images, fault
        self.position, self.reads, self.released, self.seeks = 0, 0, False, []

    def isOpened(self):
        return True

    def get(self, key):
        return {
            m.cv2.CAP_PROP_FRAME_COUNT: 348600,
            m.cv2.CAP_PROP_FPS: 166,
            m.cv2.CAP_PROP_FRAME_HEIGHT: m.SHAPE[0],
            m.cv2.CAP_PROP_FRAME_WIDTH: m.SHAPE[1],
            m.cv2.CAP_PROP_POS_FRAMES: self.position,
        }[key]

    def set(self, key, value):
        assert key == m.cv2.CAP_PROP_POS_FRAMES
        self.seeks.append(value)
        self.position = value - 74700 if self.fault == "relative" else value
        return True

    def read(self):
        image = self.images[self.position].copy()
        self.reads += 1
        self.position += 2 if self.fault == "post_position" else 1
        if self.fault == "sha":
            image[0, 0, 0] ^= 1
        return True, image

    def release(self):
        self.released = True


def videos_fixture(monkeypatch, fault=None):
    monkeypatch.setattr(m, "SHAPE", (4, 5, 3))
    guards = []
    monkeypatch.setattr(m.prep, "video_guard", lambda ctx: guards.append(ctx["video_path"]))
    videos, captures, expected = {}, {}, {}
    for v, view in enumerate(m.VIEWS):
        photos = {f: np.full(m.SHAPE, v * 30 + t, np.uint8) for t, f in enumerate(m.FRAMES)}
        videos[view] = dict(
            video_path=view,
            frames=[
                dict(frame_idx=f, expected_bgr_sha256=m.prep.signature(photos[f])["sha256"])
                for f in m.FRAMES
            ],
        )
        captures[view] = Capture(photos, fault if v == 1 else None)
        expected[view] = photos
    return videos, captures, expected, guards


def test_three_captures_three_absolute_seeks_each(monkeypatch):
    videos, captures, expected, guards = videos_fixture(monkeypatch)
    opened = []

    def factory(path):
        opened.append(path)
        return captures[path]

    images = m.extract_native(videos, capture_factory=factory)
    assert opened == list(m.VIEWS)
    assert guards == [view for view in m.VIEWS for _ in range(2)]
    for view, cap in captures.items():
        assert cap.seeks == [75053, 75054, 75055] and cap.reads == 3 and cap.released
        for frame in m.FRAMES:
            np.testing.assert_array_equal(images[frame][view], expected[view][frame])
    assert [f - 74700 for f in m.FRAMES] == [353, 354, 355]


@pytest.mark.parametrize("fault", ["relative", "post_position", "sha"])
def test_second_camera_failure_releases_and_stops(monkeypatch, fault):
    videos, captures, _, guards = videos_fixture(monkeypatch, fault)
    opened = []

    def factory(path):
        opened.append(path)
        return captures[path]

    with pytest.raises(ValueError):
        m.extract_native(videos, capture_factory=factory)
    assert opened == list(m.VIEWS[:2])
    assert captures[m.VIEWS[0]].released and captures[m.VIEWS[1]].released
    assert not captures[m.VIEWS[2]].released
    assert guards == [view for view in m.VIEWS[:2] for _ in range(2)]


@pytest.mark.parametrize("bad_frame", [353, 75052])
def test_wrong_native_request_rejected_before_open(monkeypatch, bad_frame):
    videos, _, _, _ = videos_fixture(monkeypatch)
    videos[m.VIEWS[0]]["frames"][0]["frame_idx"] = bad_frame
    with pytest.raises(ValueError, match="exact absolute frames"):
        m.extract_native(videos)


def test_native_science_slice_and_readonly(tmp_path):
    count = m.STOP - m.START
    frames = np.arange(m.START, m.STOP, dtype=np.int64)
    pose = np.broadcast_to(frames[:, None, None, None], (count, 3, 3, 3)).astype(np.float64)
    observed = np.zeros((count, 3, 3), bool)
    observed[75054 - m.START, 2] = True
    arrays = dict(frame_idx=frames, pose_xyz=pose, observed=observed)
    path = tmp_path / "science.npz"
    np.savez_compressed(path, **arrays)
    data = path.read_bytes()
    ref = dict(
        archive=dict(path=str(path), sha256=m.prep.sha(data), size_bytes=len(data)),
        signatures={key: m.prep.signature(value) for key, value in arrays.items()},
    )
    actual = m.load_science(ref, m.prep.Evidence())
    np.testing.assert_array_equal(actual["frame_idx"], m.FRAMES)
    np.testing.assert_array_equal(actual["pose_xyz"][:, 2, 0, 0], m.FRAMES)
    np.testing.assert_array_equal(actual["observed"][:, 2, 0], [False, True, False])
    assert all(not value.flags.writeable for value in actual.values())
    ref["signatures"]["frame_idx"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="signature differs"):
        m.load_science(ref, m.prep.Evidence())


def test_common_crop_and_global_support_labels(monkeypatch):
    crop = m.crop_for(np.array([[300, 400], [500, 600], [np.nan, np.nan]]))
    assert crop == [220, 320, 360, 360]
    assert m.support_label(2, 0, True) == "S2M:D"
    assert m.support_label(2, 2, False) == "S2T:N"
    projected = np.full((3, 3, 2), np.nan)
    projected[2, 0] = [310, 500]
    observed = np.zeros((3, 3), bool)
    observed[2, 0] = True
    circles = []
    original = m.cv2.circle

    def circle(image, center, radius, color, thickness, line_type):
        circles.append((center, thickness))
        return original(image, center, radius, color, thickness, line_type)

    monkeypatch.setattr(m.cv2, "circle", circle)
    panel = m.draw_panel(
        np.zeros((1200, 1200, 3), np.uint8), crop, "synthetic", projected, observed
    )
    assert panel.shape == (638, 600, 3)
    assert circles == [((150, 300), -1)]
    observed[2, 0] = False
    m.draw_panel(np.zeros((1200, 1200, 3), np.uint8), crop, "synthetic", projected, observed)
    assert circles[-1] == ((150, 300), 2)

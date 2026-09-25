"""Small synthetic native-seek, hash, median and fresh-output checks; no real video."""

import json
from pathlib import Path

import numpy as np
import prepare_top_background_5to6_minutes as m
import pytest


@pytest.fixture(autouse=True)
def isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "IMAGE_SHAPE", (4, 5, 3))

    def forbidden(*args, **kwargs):
        pytest.fail("real input access forbidden")

    monkeypatch.setattr(m, "load_context", forbidden)
    monkeypatch.setattr(m, "video_guard", forbidden)
    monkeypatch.setattr(m.cv2, "VideoCapture", forbidden)


def fixture_inputs():
    base = np.arange(np.prod(m.IMAGE_SHAPE), dtype=np.uint8).reshape(m.IMAGE_SHAPE)
    values = (80, 0, 160, 20, 60, 100, 40, 120, 140)
    images = [base + value for value in values]
    rows = [
        dict(frame_idx=frame, expected_bgr_sha256=m.signature(image)["sha256"])
        for frame, image in zip(m.FRAMES, images, strict=True)
    ]
    return dict(video_path="synthetic.mp4", frames=rows), images


class Capture:
    def __init__(self, images, fault=None):
        self.images = dict(zip(m.FRAMES, images, strict=True))
        self.seeks, self.reads, self.position, self.released = [], 0, 0, False
        self.fault = fault

    def isOpened(self):
        return True

    def get(self, key):
        return {
            m.cv2.CAP_PROP_FRAME_COUNT: m.TOTAL,
            m.cv2.CAP_PROP_FPS: 20 if self.fault == "fps" else m.FPS,
            m.cv2.CAP_PROP_FRAME_HEIGHT: m.IMAGE_SHAPE[0],
            m.cv2.CAP_PROP_FRAME_WIDTH: m.IMAGE_SHAPE[1],
            m.cv2.CAP_PROP_POS_FRAMES: self.position,
        }[key]

    def set(self, key, value):
        assert key == m.cv2.CAP_PROP_POS_FRAMES
        self.seeks.append(value)
        self.position = value - m.FIRST if self.fault == "relative" else value
        return True

    def read(self):
        image = self.images[self.position].copy()
        self.reads += 1
        self.position += 2 if self.fault == "post_position" else 1
        if self.fault == "sha":
            image[0, 0, 0] ^= 1
        if self.fault == "dtype":
            image = image.astype(np.float64)
        if self.fault == "shape":
            image = image[:-1]
        return True, image

    def release(self):
        self.released = True


def test_one_capture_nine_absolute_reads_pixel_hash_and_odd_median():
    context, images = fixture_inputs()
    original = [image.copy() for image in images]
    calls, capture = [], Capture(images)

    def factory(path):
        calls.append(path)
        return capture

    actual = m.extract_frames(context, capture_factory=factory)
    assert calls == ["synthetic.mp4"]
    assert capture.seeks == list(m.FRAMES) and capture.reads == 9 and capture.released
    for output, reference in zip(actual, original, strict=True):
        np.testing.assert_array_equal(output, reference)
    # Independent odd order-statistic oracle; uint8's fifth sorted sample.
    expected = np.sort(np.stack(original), axis=0)[4]
    background = m.median_background(actual)
    assert background.dtype == np.uint8
    np.testing.assert_array_equal(background, expected)
    encoded = m.png_bytes(background)
    restored = m.cv2.imdecode(np.frombuffer(encoded, np.uint8), m.cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(restored, expected)
    for image, before in zip(images, original, strict=True):
        np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("fault", ["fps", "relative", "post_position", "sha", "dtype", "shape"])
def test_bad_native_or_pixels_rejected_and_capture_closed(fault):
    context, images = fixture_inputs()
    capture = Capture(images, fault)
    with pytest.raises(ValueError):
        m.extract_frames(context, capture_factory=lambda path: capture)
    assert capture.released


def test_fixed_order_and_nine_images_required():
    context, images = fixture_inputs()
    context["frames"][0]["frame_idx"] = 0
    with pytest.raises(ValueError, match="fixed native"):
        m.extract_frames(context)
    with pytest.raises(ValueError, match="nine native"):
        m.median_background(images[:-1])


def test_synthetic_run_saves_ten_png_and_seal(tmp_path, monkeypatch):
    context, images = fixture_inputs()
    context.update(hardlinks=[], expected_stat=[1, 2, 3, 4])
    monkeypatch.setattr(m, "load_context", lambda evidence: context)
    monkeypatch.setattr(m, "video_guard", lambda context: None)
    capture = Capture(images)
    monkeypatch.setattr(m.cv2, "VideoCapture", lambda path: capture)
    # Source provenance is synthetic too; no existing workspace file read here.
    fake_source = tmp_path / "source.py"
    fake_source.write_text("# synthetic source\n", encoding="utf-8")
    monkeypatch.setattr(m, "__file__", str(fake_source))
    output = tmp_path / "output"
    summary = m.run(output)
    assert summary["requested_BGR_reads"] == 9 and summary["whole_video_rehashes"] == 0
    assert not summary["background_adopted_for_tracking"]
    seal = json.loads((output / "preparation.provenance.json").read_bytes())
    assert seal["status"] == "ok" and len(seal["outputs"]) == 14
    assert seal["video_reference"]["whole_bytes_rehashed"] is False
    assert capture.seeks == list(m.FRAMES)
    for record in seal["outputs"].values():
        path = Path(record["path"])
        assert path.is_relative_to(tmp_path)
        data = path.read_bytes()
        assert len(data) == record["size_bytes"] and m.sha(data) == record["sha256"]
    with pytest.raises(ValueError, match="fresh workspace"):
        m.run(output)


def test_decode_failure_preserves_failed_seal(tmp_path, monkeypatch):
    context, images = fixture_inputs()
    monkeypatch.setattr(m, "load_context", lambda evidence: context)
    monkeypatch.setattr(m, "video_guard", lambda context: None)
    capture = Capture(images, "sha")
    monkeypatch.setattr(m.cv2, "VideoCapture", lambda path: capture)
    fake_source = tmp_path / "source.py"
    fake_source.write_text("# synthetic source\n", encoding="utf-8")
    monkeypatch.setattr(m, "__file__", str(fake_source))
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="BGR SHA"):
        m.run(output)
    seal = json.loads((output / "preparation.provenance.json").read_bytes())
    assert seal["status"] == "failed"
    assert "summary.json" not in seal["outputs"] and capture.released
    assert not (output / "background_top_cam.png").exists()

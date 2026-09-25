"""Bounded synthetic absolute-frame requests and real lossless PNG encoding."""

import hashlib
from pathlib import Path

import numpy as np
import prepare_endpoint_negative_native_photos as m
import pytest


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    work = root / "work" / "three_body"
    work.mkdir(parents=True)
    for module in (m, m.frozen, m.prior.raw, m.proof):
        monkeypatch.setattr(module, "REPO", root, raising=False)
        monkeypatch.setattr(module, "WORK", work, raising=False)
    monkeypatch.setattr(m, "PRODUCER", work / "producer")
    monkeypatch.setattr(m, "RAWPACK", work / "rawpack")

    def forbidden(*args, **kwargs):
        pytest.fail("actual source, metadata, arrays or video access is forbidden")

    monkeypatch.setattr(m, "source_code", forbidden)
    monkeypatch.setattr(m, "load_context", forbidden)
    monkeypatch.setattr(m.frozen, "load_chunk", forbidden)
    monkeypatch.setattr(m.previous, "read", forbidden)
    monkeypatch.setattr(m.previous, "video_stat_guard", forbidden)
    monkeypatch.setattr(m.prior.raw, "iter_native", forbidden)
    monkeypatch.setattr(m.prior.cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    return work


@pytest.mark.parametrize("frame", [90141, 90258, 90259])
def test_request_is_absolute_native_not_mask_offset_and_reader_closes(frame, isolate, monkeypatch):
    context = dict(
        videos={v: isolate / (v + ".mp4") for v in m.VIEWS},
        manifest=dict(frames=list(range(89640, 91300))),
        chunk=dict(index=24, start=89640, stop=91300),
    )
    images = {v: object() for v in m.VIEWS}
    calls, guards = [], []

    def reader(videos, manifest, start, stop):
        assert videos is context["videos"] and manifest is context["manifest"]
        assert (start, stop) == (frame, frame + 1)
        assert start not in (501, 618, 619)
        calls.append((start, stop))
        try:
            yield frame, images
        finally:
            calls.append("closed")

    monkeypatch.setattr(m.prior.raw, "iter_native", reader)
    monkeypatch.setattr(
        m.previous, "video_stat_guard", lambda videos, chunk: guards.append((videos, chunk))
    )
    actual = m.native_frame(context, frame)
    assert actual is images and calls == [(frame, frame + 1), "closed"]
    assert guards == [(context["videos"], context["chunk"])] * 2


@pytest.mark.parametrize("wrong", [501, 618])
def test_chunk_offsets_are_rejected_before_any_reader_or_stat_access(wrong):
    with pytest.raises(ValueError, match="absolute frames"):
        m.native_frame({}, wrong)


def test_wrong_returned_native_still_closes_reader_and_checks_final_stat(monkeypatch):
    calls = []

    def reader(*args):
        try:
            yield 90142, {v: object() for v in m.VIEWS}
        finally:
            calls.append("closed")

    monkeypatch.setattr(m.prior.raw, "iter_native", reader)
    monkeypatch.setattr(m.previous, "video_stat_guard", lambda *args: calls.append("stat"))
    with pytest.raises(ValueError, match="requested native"):
        m.native_frame(dict(videos={}, manifest={}, chunk={}), 90141)
    assert calls == ["stat", "closed", "stat"]


class TemporaryOutput:
    def __init__(self, root):
        self.root, self.calls = root, []

    def write(self, name, data):
        path = self.root / name
        assert path.resolve().is_relative_to(self.root.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        self.calls.append(name)
        return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def test_real_png_preserves_exact_asymmetric_uint8_BGR_and_native_metadata(isolate):
    yy, xx = np.indices((1200, 1200), dtype=np.uint16)
    image = np.stack((xx % 256, yy % 256, (xx + 3 * yy) % 256), axis=-1).astype(np.uint8)
    original = image.copy()
    image.flags.writeable = False
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    out = TemporaryOutput(isolate)
    record = m.write_native_png(out, 90258, "side2_cam", image, digest)
    assert out.calls == ["native/frame_90258_side2_cam.png"]
    payload = Path(record["image"]["path"]).read_bytes()
    restored = m.prior.cv2.imdecode(np.frombuffer(payload, np.uint8), m.prior.cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(restored, original)
    np.testing.assert_array_equal(image, original)
    assert hashlib.sha256(restored.tobytes()).hexdigest() == digest
    assert record["native_absolute_seek"] == record["frame_idx"] == 90258
    assert record["consumed_chunk_offset"] == 618
    assert record["camera_sync_offset_applied"] == 0
    assert record["lossless_PNG_roundtrip_verified"] is True


def test_wrong_BGR_digest_rejected_before_encoding_or_write(isolate, monkeypatch):
    image = np.zeros((1200, 1200, 3), np.uint8)
    monkeypatch.setattr(
        m.prior.cv2, "imencode", lambda *args: pytest.fail("mismatched pixels must not encode")
    )
    out = TemporaryOutput(isolate)
    with pytest.raises(ValueError, match="native BGR pixels"):
        m.write_native_png(out, 90141, "top_cam", image, "0" * 64)
    assert not out.calls

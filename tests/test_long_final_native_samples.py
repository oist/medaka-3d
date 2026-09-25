"""Fixed selection and mock native decoder tests; no real video/run is opened."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "work/three_body/long_final_native_samples.py"
SPEC = importlib.util.spec_from_file_location("long_final_native_samples_cpu_test", PATH)
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)
from prepare_sequence import PackedMasks  # noqa: E402


def test_fixed_plan_exact_count_and_every_boundary_without_result_inputs():
    values = mod.fixed_sample_frames()
    assert values.dtype == np.int64 and len(values) == 543 and len(set(values)) == 543
    np.testing.assert_array_equal(values[:3], [49800, 49801, 49802])
    np.testing.assert_array_equal(values[-3:], [348597, 348598, 348599])
    for boundary in range(51460, 348600, 1660):
        assert {boundary - 1, boundary, boundary + 1} <= set(values)
    values[:] = 0
    assert mod.fixed_sample_frames()[0] == 49800


@pytest.mark.parametrize("mutation", ["short", "gap", "duplicate", "order", "bool"])
def test_consumed_exact180_order(mutation):
    records = [
        dict(index=i, start=49800 + i * 1660, stop=49800 + (i + 1) * 1660) for i in range(180)
    ]
    mod.validate_consumed_ranges(records)
    if mutation == "short":
        records.pop()
    elif mutation == "gap":
        records[17]["start"] += 1
    elif mutation == "duplicate":
        records[17] = records[16]
    elif mutation == "order":
        records[17], records[18] = records[18], records[17]
    else:
        records[0]["index"] = False
    with pytest.raises(ValueError, match="consumed"):
        mod.validate_consumed_ranges(records)


@pytest.fixture
def packed():
    encoder = PackedMasks((4, 5))
    for frame, count in ((49800, 2), (49801, 0), (49802, 1)):
        masks = np.zeros((count, 4, 5), bool)
        if count:
            masks[0, 1:3, 2:4] = True
        encoder.append(
            frame,
            masks,
            np.full(count, 0.75, np.float32),
            np.tile(np.array([1.0, 1.0, 4.0, 3.0], np.float32), (count, 1)),
            np.zeros(count, np.int64),
        )
    return encoder.arrays()


def test_selected13fields_keep_row_order_empty_crops_and_rebase(packed):
    first, masks = mod.select_packed_frame(packed, 0, 49800, 2)
    assert len(first) == 13 and masks.shape == (2, 4, 5)
    assert masks[0].sum() == 4 and masks[1].sum() == 0  # Empty cropped instance retained.
    last, masks = mod.select_packed_frame(packed, 2, 49802, 1)
    np.testing.assert_array_equal(last["frame_offsets"], [0, 1])
    np.testing.assert_array_equal(last["mask_byte_offsets"], [0, 1])
    assert masks.sum() == 4 and last["frame_idx"].tolist() == [49802]
    for key in ("scores", "pred_boxes", "classes", "instance_id", "source_track_id"):
        np.testing.assert_array_equal(last[key], packed[key][2:3])
        assert not np.shares_memory(last[key], packed[key])


def test_zero_detection_and_unavailable_are_different(packed):
    selected, available = mod.select_packed_frame(packed, 1, 49801, 0)
    assert available is not None and available.shape == (0, 4, 5)
    packed["view_available"][1] = False
    missing, absent = mod.select_packed_frame(packed, 1, 49801, 0)
    assert absent is None and selected["view_available"][0] and not missing["view_available"][0]
    np.testing.assert_array_equal(selected["frame_mask_sha256"], missing["frame_mask_sha256"])


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("native", "native offset"),
        ("count", "count"),
        ("offset", "offsets"),
        ("hash", "SHA256"),
        ("payload", "SHA256"),
        ("missing_with_instances", "unavailable"),
    ],
)
def test_sample_corruption_rejected(packed, mutation, reason):
    index, native, count = 0, 49800, 2
    if mutation == "native":
        native += 1
    elif mutation == "count":
        count += 1
    elif mutation == "offset":
        packed["mask_byte_offsets"][0] = 1
    elif mutation == "hash":
        packed["frame_mask_sha256"][0] = b"0" * 64
    elif mutation == "payload":
        packed["mask_bits"][0] ^= 1
    else:
        packed["view_available"][0] = False
    with pytest.raises(ValueError, match=reason):
        mod.select_packed_frame(packed, index, native, count)


class Capture:
    def __init__(self, image, fault=None):
        self.image, self.fault, self.pos, self.closed, self.seeks = image, fault, 0, False, []

    def isOpened(self):
        return self.fault != "open"

    def get(self, prop):
        return {
            cv2.CAP_PROP_FPS: 165 if self.fault == "fps" else 166,
            cv2.CAP_PROP_FRAME_COUNT: 1660 if self.fault == "count" else 348600,
            cv2.CAP_PROP_FRAME_HEIGHT: self.image.shape[0],
            cv2.CAP_PROP_FRAME_WIDTH: self.image.shape[1],
            cv2.CAP_PROP_POS_FRAMES: self.pos,
        }[prop]

    def set(self, prop, value):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.seeks.append(value)
        self.pos = value + (1 if self.fault == "seek_position" else 0)
        return self.fault != "seek"

    def read(self):
        self.pos += 2 if self.fault == "decode_position" else 1
        return self.fault != "EOF", self.image

    def release(self):
        self.closed = True


@pytest.mark.parametrize(
    "fault", ["open", "fps", "count", "seek", "seek_position", "EOF", "decode_position", "BGR"]
)
def test_mock_capture_offset_eof_metadata_and_hash_rejection(fault):
    image = np.full((4, 5, 3), 19, np.uint8)
    capture = Capture(image, fault)
    with pytest.raises(ValueError):
        mod.validate_capture(capture, shape=(4, 5))
        mod.read_native_frame(
            capture, 51460, "0" * 64 if fault == "BGR" else mod.array_sha(image), shape=(4, 5)
        )


def test_native_seek_is_absolute_not_relative_to_chunk():
    image = np.full((4, 5, 3), 19, np.uint8)
    capture = Capture(image)
    mod.validate_capture(capture, shape=(4, 5))
    result = mod.read_native_frame(capture, 51460, mod.array_sha(image), shape=(4, 5))
    assert capture.seeks == [51460] and capture.pos == 51461
    np.testing.assert_array_equal(result, image)


def test_scan_releases_on_failure_before_any_output(tmp_path):
    capture = Capture(np.zeros((1200, 1200, 3), np.uint8), "fps")
    context = SimpleNamespace(videos={"top_cam": tmp_path / "fake.mp4"})
    with pytest.raises(ValueError, match="fps"):
        mod._scan_view(context, "top_cam", lambda _: capture, None)
    assert capture.closed and not capture.seeks


def test_changed_fixed_sample_set_is_rejected_before_capture():
    context = SimpleNamespace(samples=np.array([49800]))

    def forbidden(_):
        raise AssertionError("must not open decoder")

    with pytest.raises(ValueError, match="selection"):
        mod.read_native_samples(context, capture_factory=forbidden)


@pytest.mark.parametrize("fault", [None, "EOF", "mask_hash"])
def test_reader_native_masks_callback_and_failure_release(tmp_path, fault):
    image = np.full((1200, 1200, 3), 19, np.uint8)
    encoder = PackedMasks((1200, 1200))
    for frame, count in ((49800, 1), (49801, 0), (49802, 0)):
        masks = np.zeros((count, 1200, 1200), bool)
        if count:
            masks[0, 2:5, 3:7] = True
        encoder.append(
            frame,
            masks,
            np.full(count, 0.75, np.float32),
            np.tile(np.array([3.0, 2.0, 7.0, 5.0], np.float32), (count, 1)),
            np.zeros(count, np.int64),
        )
    data = encoder.arrays()
    data["view_available"][2] = False
    if fault == "mask_hash":
        data["frame_mask_sha256"][0] = b"0" * 64
    np.savez_compressed(tmp_path / "masks.npz", **data)
    manifest = dict(
        frames=[49800, 49801, 49802],
        image_shape_hw=[1200, 1200],
        raw={"top_cam": dict(frame_bgr_sha256=[mod.array_sha(image)] * 3)},
        masks={"top_cam": dict(path="masks.npz", instances_per_frame=[1, 0, 0])},
    )
    context = SimpleNamespace(
        videos={"top_cam": tmp_path / "fake.mp4"},
        samples=np.arange(49800, 49803),
        chunks=[dict(root=tmp_path, manifest=manifest, record=dict(start=49800))],
        nodes=SimpleNamespace(at=lambda view, frame: np.full((2, 3, 2), frame, np.float32)),
    )
    capture, saved = Capture(image, fault), []

    def callback(row, arrays):
        saved.append((row, arrays))
        row["selected_packed_arrays"]["scores"]["sha256"] = "callback mutation"
        arrays["image_bgr"][:] = 0

    if fault:
        with pytest.raises(ValueError):
            mod._scan_view(context, "top_cam", lambda _: capture, callback)
        assert saved == []
    else:
        rows = mod._scan_view(context, "top_cam", lambda _: capture, callback)
        assert len(rows) == 3 and capture.seeks == [49800, 49801, 49802]
        assert [row["mask_count"] for row in rows] == [1, 0, None]
        assert [row["mask_available"] for row in rows] == [True, True, False]
        assert rows[-1]["mask_decoded_sha256"] is None
        assert rows[-1]["packed_frame_mask_sha256"] == mod.array_sha(
            np.empty((0, 1200, 1200), bool)
        )
        assert np.all(image == 19)  # Callback owns its pixels and cannot alter the decoder buffer.
        assert rows[0]["selected_packed_arrays"]["scores"]["sha256"] != "callback mutation"
    assert capture.closed


def test_unfinished_chain_stops_before_chunk_or_decoder_access(monkeypatch):
    monkeypatch.syspath_prepend(str(PATH.parent))
    import audit_long_final as final

    def unfinished(*args):
        raise ValueError("completed successful guarded stage required")

    monkeypatch.setattr(final, "load_supervisor_chain", unfinished)
    monkeypatch.setattr(mod.np, "load", lambda *args, **kwargs: pytest.fail("opened payload"))
    evidence = SimpleNamespace(add=lambda *args: None)
    with pytest.raises(ValueError, match="completed successful"):
        mod.load_native_sample_inputs(Path("not-a-completed-run"), evidence)

"""Lossless raw and mask staging must preserve frames, counts and empty views."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "labs/20260909_three_body/prepare_sequence.py"
SPEC = importlib.util.spec_from_file_location("_three_body_staging", MODULE)
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


def test_mask_packing_roundtrip_handles_borders_fragments_empty_and_four_instances(tmp_path):
    rng = np.random.default_rng(35)
    shape = (11, 13)
    masks = rng.random((4, *shape)) > 0.7
    masks[0] = False
    masks[1] = False
    masks[1, 0, 0] = True
    masks[1, -1, -1] = True
    encoder = stage.PackedMasks(shape)
    encoder.append(42, masks, np.array([0.5, 0.8, 0.9, 1]), np.zeros((4, 4)), np.zeros(4))
    encoder.append(43, np.zeros((0, *shape), bool), np.zeros(0), np.zeros((0, 4)), np.zeros(0))
    path = tmp_path / "masks.npz"
    np.savez_compressed(path, **encoder.arrays())
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(stage.decode_frame_masks(data, 0), masks)
    assert stage.decode_frame_masks(data, 1).shape == (0, *shape)
    assert data["view_available"].all()
    np.testing.assert_array_equal(data["frame_offsets"], [0, 4, 4])
    np.testing.assert_array_equal(data["mask_boxes"][0], [0, 0, 0, 0])
    np.testing.assert_array_equal(data["mask_boxes"][1], [0, 0, 13, 11])
    np.testing.assert_array_equal(data["instance_id"], [0, 1, 2, 3])
    assert np.all(data["source_track_id"] == -1)
    assert data["frame_mask_sha256"][0].decode() == stage.array_sha256(masks)


def test_bad_prediction_counts_scores_frames_and_crop_offsets_fail():
    encoder = stage.PackedMasks((2, 3))
    fields = (np.ones((1, 2, 3), bool), np.array([0.9]), np.zeros((1, 4)), np.zeros(1))
    encoder.append(0, *fields)
    with pytest.raises(ValueError, match="consecutive"):
        encoder.append(2, *fields)
    with pytest.raises(ValueError, match="threshold"):
        encoder.append(1, fields[0], np.array([0.49]), fields[2], fields[3])
    with pytest.raises(ValueError, match="counts differ"):
        encoder.append(1, fields[0], np.ones(2), fields[2], fields[3])
    data = encoder.arrays()
    data["mask_boxes"][0, 2] = 4
    with pytest.raises(ValueError, match="outside image"):
        stage.decode_frame_masks(data, 0)
    data = encoder.arrays()
    data["mask_byte_offsets"][-1] += 1
    with pytest.raises(ValueError, match="byte count"):
        stage.decode_frame_masks(data, 0)


def test_lossless_clip_all_decoded_pixels_and_frame_count_are_verified(tmp_path):
    path = tmp_path / "lossless.avi"
    rng = np.random.default_rng(7)
    images = rng.integers(0, 256, (5, 32, 34, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 166.0, (34, 32))
    assert writer.isOpened(), "the experiment requires FFV1 availability"
    for image in images:
        writer.write(image)
    writer.release()
    hashes = [stage.array_sha256(image) for image in images]
    qa = stage.verify_lossless_clip(path, hashes, (32, 34))
    assert qa == {"decoded_frames": 5, "pixel_sha256_matches": 5, "extra_frames": 0}
    bad_hashes = hashes.copy()
    bad_hashes[3] = "0" * 64
    with pytest.raises(ValueError, match="pixel mismatch"):
        stage.verify_lossless_clip(path, bad_hashes, (32, 34))
    with pytest.raises(ValueError, match="count or fps"):
        stage.verify_lossless_clip(path, hashes[:-1], (32, 34))


def test_background_guard_does_not_use_sequence_neighbors():
    selected = stage.select_background_frames(79800, 80100)
    assert 80000 not in selected
    assert 60000 in selected and 100000 in selected
    for name, (start, stop) in stage.WINDOWS.items():
        chosen = stage.select_background_frames(start, stop)
        assert len(chosen) == (8 if name == "continuous_one_minute" else 9)
        assert all(not start - 332 <= f < stop + 332 for f in chosen)


def test_continuous_minute_contains_9960_native_frames_and_excludes_80000_background():
    start, stop = stage.WINDOWS["continuous_one_minute"]
    assert (start, stop) == (76620, 86580)
    assert stop - start == 9960
    chosen = stage.select_background_frames(start, stop)
    assert 80000 not in chosen
    assert chosen == [value for value in stage.BACKGROUND_CANDIDATES if value != 80000]


def test_existing_output_is_untouched(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="overwrite"):
        stage.run(tmp_path, out, "transient")
    assert sentinel.read_bytes() == b"keep"


def test_source_frame_position_mismatch_cannot_be_silently_staged():
    class WrongPositionCapture:
        def set(self, prop, value):
            return True

        def read(self):
            return True, np.zeros((2, 3, 3), np.uint8)

        def get(self, prop):
            # A decoder that reports it skipped one source frame must fail.
            return 44

    with pytest.raises(RuntimeError, match="position mismatch"):
        stage.read_frame(WrongPositionCapture(), 42, seek=True, shape=(2, 3))

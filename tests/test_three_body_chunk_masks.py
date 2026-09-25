"""Packed chunk joins preserve the encoder bytes and reject ambiguous or corrupt data."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load(name):
    spec = importlib.util.spec_from_file_location(f"three_body_chunk_masks_test_{name}",
                                                 LAB / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREPARE = load("prepare_sequence")
previous = sys.modules.get("prepare_sequence")
sys.modules["prepare_sequence"] = PREPARE
try:
    CHUNKS = load("chunk_masks")
finally:
    if previous is None:
        sys.modules.pop("prepare_sequence", None)
    else:
        sys.modules["prepare_sequence"] = previous


def encode(frames, shape=(7, 9)):
    encoder = PREPARE.PackedMasks(shape)
    counts = (4, 0, 1, 2, 0, 1, 0, 3)
    for frame in frames:
        count = counts[(frame - 510) % len(counts)]
        masks = np.zeros((count, *shape), bool)
        for instance in range(count):
            if instance == 0 and count == 4:
                continue  # Empty individual detection, distinct from zero detections.
            if instance == 1:
                masks[instance, 3, 2] = True  # Seven canonical padding bits.
            else:
                masks[instance, 1:5, 2:7] = True
                masks[instance, 2, 3] = False
        scores = np.linspace(.5, 1., count, dtype=np.float32)
        if count:
            scores[0] = np.nextafter(np.float32(.5), np.float32(1.))
        boxes = np.tile(np.array([-0., .125, 7.25, 5.875], np.float32), (count, 1))
        encoder.append(frame, masks, scores, boxes, np.arange(count, dtype=np.int64))
    result = encoder.arrays()
    result["view_available"][np.isin(result["frame_idx"], [511, 516])] = False
    return result


def assert_bytes_equal(first, second):
    assert list(first) == list(second)
    for key in first:
        assert first[key].dtype == second[key].dtype, key
        assert first[key].shape == second[key].shape, key
        assert first[key].tobytes() == second[key].tobytes(), key


def test_mixed_path_and_mapping_join_equals_one_continuous_encoder(tmp_path):
    chunks = [encode(range(510, 513)), encode([513]), encode(range(514, 518))]
    path = tmp_path / "middle.npz"
    np.savez_compressed(path, **chunks[1])
    actual = CHUNKS.concatenate_packed_masks(iter((chunks[0], path, chunks[2])))
    expected = encode(range(510, 518))
    assert_bytes_equal(actual, expected)
    CHUNKS.validate_packed_masks(actual)
    output = tmp_path / "joined.npz"
    np.savez_compressed(output, **actual)
    with np.load(output, allow_pickle=False) as arrays:
        assert_bytes_equal({key: arrays[key] for key in arrays.files}, expected)
    for index in range(8):
        np.testing.assert_array_equal(PREPARE.decode_frame_masks(actual, index),
                                      PREPARE.decode_frame_masks(expected, index))
    assert actual["frame_offsets"].tolist() == [0, 4, 4, 5, 7, 7, 8, 8, 11]
    assert actual["view_available"].tolist() == [True, False, True, True, True, True, False, True]


def test_all_empty_detections_and_unavailable_remain_distinct():
    first, second = encode([511]), encode([512])
    # Use the producer for an available empty second frame as well.
    encoder = PREPARE.PackedMasks((7, 9))
    encoder.append(512, np.zeros((0, 7, 9), bool), np.empty(0, np.float32),
                   np.empty((0, 4), np.float32), np.empty(0, np.int64))
    second = encoder.arrays()
    result = CHUNKS.concatenate_packed_masks([first, second])
    assert result["view_available"].tolist() == [False, True]
    assert result["frame_offsets"].tolist() == [0, 0, 0]
    assert result["mask_byte_offsets"].tolist() == [0]
    assert result["mask_bits"].size == 0
    assert result["frame_mask_sha256"][0] == result["frame_mask_sha256"][1]


def test_empty_individual_crop_has_no_payload_and_single_chunk_result_does_not_alias():
    data = encode([510])
    result = CHUNKS.concatenate_packed_masks([data])
    assert result["mask_boxes"][0].tolist() == [0, 0, 0, 0]
    assert result["mask_byte_offsets"][:2].tolist() == [0, 0]
    assert not PREPARE.decode_frame_masks(result, 0)[0].any()
    assert_bytes_equal(result, data)
    result["scores"][0] = .75
    result["image_shape_hw"][0] = 11
    assert data["scores"][0] != .75
    assert data["image_shape_hw"][0] == 7


@pytest.mark.parametrize("frames", [[510, 511], [513, 514], [507, 508]])
def test_duplicate_gap_or_out_of_order_boundary_is_rejected(frames):
    with pytest.raises(ValueError, match="boundary"):
        CHUNKS.concatenate_packed_masks([encode([510, 511]), encode(frames)])


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.pop("scores"), "schema"),
    (lambda d: d.update(schema_version=np.array(1)), "schema"),
    (lambda d: d.update(scores=d["scores"].astype(np.float64)), "dtype"),
    (lambda d: d["frame_idx"].__setitem__(1, 510), "continuous"),
    (lambda d: d["frame_idx"].__setitem__(1, 513), "continuous"),
    (lambda d: d["frame_idx"].__setitem__(0, -1), "nonnegative"),
    (lambda d: d["frame_offsets"].__setitem__(0, 1), "frame_offsets"),
    (lambda d: d["frame_offsets"].__setitem__(-1, 3), "frame_offsets"),
    (lambda d: d["frame_offsets"].__setitem__(1, -1), "frame_offsets"),
    (lambda d: d["mask_byte_offsets"].__setitem__(0, 1), "mask_byte_offsets"),
    (lambda d: d["mask_byte_offsets"].__setitem__(-1, 0), "mask_byte_offsets"),
    (lambda d: d["mask_byte_offsets"].__setitem__(2, 2), "payload length"),
    (lambda d: d.update(mask_bits=d["mask_bits"][:-1]), "mask_byte_offsets"),
    (lambda d: d["mask_boxes"].__setitem__((1, 2), 99), "outside"),
    (lambda d: d["mask_boxes"].__setitem__(0, [1, 1, 1, 1]), "canonical"),
    (lambda d: d["scores"].__setitem__(0, np.nan), "scores"),
    (lambda d: d["scores"].__setitem__(0, np.inf), "scores"),
    (lambda d: d["scores"].__setitem__(0, .49), "scores"),
    (lambda d: d["pred_boxes"].__setitem__((0, 0), np.nan), "pred_boxes"),
    (lambda d: d["pred_boxes"].__setitem__((0, 0), np.inf), "pred_boxes"),
    (lambda d: d["classes"].__setitem__(0, -1), "classes"),
    (lambda d: d["instance_id"].__setitem__(1, 0), "instance_id"),
    (lambda d: d["source_track_id"].__setitem__(0, 7), "source_track_id"),
    (lambda d: d["view_available"].__setitem__(0, False), "zero instances"),
    (lambda d: d["frame_mask_sha256"].__setitem__(0, b"0" * 64), "SHA256 mismatch"),
    (lambda d: d["frame_mask_sha256"].__setitem__(0, b"g" * 64), "lowercase"),
])
def test_invalid_chunk_is_rejected_before_join(mutation, match):
    data = encode([510, 511])
    mutation(data)
    with pytest.raises(ValueError, match=match):
        CHUNKS.concatenate_packed_masks([data])


def test_meaningful_payload_corruption_fails_decoded_hash():
    data = encode([510])
    data["mask_bits"][0] ^= np.uint8(1)
    with pytest.raises(ValueError, match="decoded mask SHA256"):
        CHUNKS.validate_packed_masks(data)


def test_unused_padding_corruption_is_detected_even_when_decoded_hash_still_matches():
    data = encode([510])
    before = PREPARE.array_sha256(PREPARE.decode_frame_masks(data, 0))
    data["mask_bits"][0] |= np.uint8(128)
    assert PREPARE.array_sha256(PREPARE.decode_frame_masks(data, 0)) == before
    with pytest.raises(ValueError, match="padding"):
        CHUNKS.validate_packed_masks(data)


def test_non_tight_crop_is_rejected_even_if_decoded_mask_hash_is_unchanged():
    data = encode([510])
    # Extend a one-pixel bbox by a zero column: same one-byte payload and full
    # decoded image, but not the canonical encoder representation.
    data["mask_boxes"][1, 2] += 1
    assert PREPARE.array_sha256(PREPARE.decode_frame_masks(data, 0)).encode() == (
        data["frame_mask_sha256"][0])
    with pytest.raises(ValueError, match="exact nonempty-mask bounding box"):
        CHUNKS.validate_packed_masks(data)


def test_open_npz_mapping_is_supported_without_changing_values(tmp_path):
    expected = encode([510, 511])
    path = tmp_path / "mapping.npz"
    np.savez_compressed(path, **expected)
    with np.load(path, allow_pickle=False) as arrays:
        actual = CHUNKS.concatenate_packed_masks([arrays])
    assert_bytes_equal(actual, expected)


def test_unavailable_frame_requires_empty_decoded_hash():
    data = encode([511])
    data["frame_mask_sha256"][0] = b"0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        CHUNKS.validate_packed_masks(data)


def test_image_shape_and_empty_chunk_schema_are_not_silently_normalized():
    with pytest.raises(ValueError, match="image shapes"):
        CHUNKS.concatenate_packed_masks([encode([510]), encode([511], (8, 9))])
    with pytest.raises(ValueError, match="nonempty"):
        CHUNKS.concatenate_packed_masks([PREPARE.PackedMasks((7, 9)).arrays()])
    with pytest.raises(ValueError, match="at least one"):
        CHUNKS.concatenate_packed_masks([])


def test_pickle_arrays_are_never_loaded(tmp_path):
    data = encode([510])
    data["scores"] = data["scores"].astype(object)
    path = tmp_path / "object.npz"
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="Object arrays"):
        CHUNKS.concatenate_packed_masks([path])

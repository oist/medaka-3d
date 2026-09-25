"""Strict concatenation of the frozen PackedMasks format, without decoding all frames at once.

The format has thirteen fixed array fields and no explicit version/bitorder key.
Those keys/dtypes and little-bit-order encoding define the supported implicit
schema. Unknown additions are rejected instead of silently losing metadata.
Input file checksums, view/experiment/fps identity (absent from these arrays), and
chunk-manifest provenance remain the caller's duty.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path

import numpy as np
from prepare_sequence import array_sha256, decode_frame_masks

SCHEMA_DTYPES = {
    "image_shape_hw": np.dtype(np.int32),
    "frame_idx": np.dtype(np.int64),
    "frame_offsets": np.dtype(np.int64),
    "mask_boxes": np.dtype(np.int32),
    "mask_byte_offsets": np.dtype(np.int64),
    "mask_bits": np.dtype(np.uint8),
    "scores": np.dtype(np.float32),
    "pred_boxes": np.dtype(np.float32),
    "classes": np.dtype(np.int64),
    "instance_id": np.dtype(np.int64),
    "source_track_id": np.dtype(np.int64),
    "view_available": np.dtype(bool),
    "frame_mask_sha256": np.dtype("S64"),
}


def validate_packed_masks(data: Mapping[str, np.ndarray]) -> None:
    """Validate one nonempty native-frame chunk, including every decoded frame SHA.

    Zero detections in an available frame and an unavailable frame stay distinct.
    Unavailable frames must have zero instances and the empty decoded-stack hash.
    Empty individual masks use the encoder's [0,0,0,0] crop and zero payload bytes.
    Scores/predicted boxes must be finite, matching the frozen encoder. Class
    labels are nonnegative integers; instance_id is within-frame 0..N-1 and
    source_track_id is always -1, never an inferred animal identity.
    """
    if not isinstance(data, Mapping) or set(data) != set(SCHEMA_DTYPES):
        raise ValueError("packed-mask schema must contain exactly the frozen thirteen fields")
    for name, dtype in SCHEMA_DTYPES.items():
        value = data[name]
        if not isinstance(value, np.ndarray) or value.dtype != dtype:
            raise ValueError(f"packed-mask {name} must retain dtype {dtype}")
    image_shape = data["image_shape_hw"]
    if image_shape.shape != (2,) or np.any(image_shape <= 0):
        raise ValueError("invalid packed-mask image_shape_hw")
    frames = data["frame_idx"]
    if frames.ndim != 1 or not len(frames) or np.any(frames < 0):
        raise ValueError("packed-mask frame_idx must be nonempty and nonnegative")
    if np.any(np.diff(frames) != 1):
        raise ValueError("packed-mask frames must be continuous native frames without duplicates")
    boxes = data["mask_boxes"]
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError("invalid mask_boxes shape")
    instances, frame_count = len(boxes), len(frames)
    shapes = {
        "frame_offsets": (frame_count + 1,),
        "mask_byte_offsets": (instances + 1,),
        "scores": (instances,),
        "pred_boxes": (instances, 4),
        "classes": (instances,),
        "instance_id": (instances,),
        "source_track_id": (instances,),
        "view_available": (frame_count,),
        "frame_mask_sha256": (frame_count,),
    }
    for name, shape in shapes.items():
        if data[name].shape != shape:
            raise ValueError(f"invalid packed-mask {name} shape")
    if data["mask_bits"].ndim != 1:
        raise ValueError("mask_bits must be a one-dimensional byte array")
    for name, end in (("frame_offsets", instances),
                      ("mask_byte_offsets", len(data["mask_bits"]))):
        offsets = data[name]
        if (offsets[0] != 0 or offsets[-1] != end or np.any(offsets < 0)
                or np.any(offsets > end) or np.any(np.diff(offsets) < 0)):
            raise ValueError(f"invalid packed-mask {name}: origin, terminal or monotonicity")
    height, width = map(int, image_shape)
    x0, y0, x1, y1 = boxes.astype(np.int64, copy=False).T
    if np.any((x0 < 0) | (x0 > x1) | (x1 > width)
              | (y0 < 0) | (y0 > y1) | (y1 > height)):
        raise ValueError("mask crop is outside image or reversed")
    area = (x1 - x0) * (y1 - y0)
    if np.any(boxes[area == 0] != 0):
        raise ValueError("empty cropped masks must have canonical zero boxes")
    byte_counts = np.diff(data["mask_byte_offsets"])
    if np.any(byte_counts != (area + 7) // 8):
        raise ValueError("mask payload length does not match crop area")
    # A changed padding bit would evade a decoded-frame hash. The encoder always
    # pads with zero, so check those bits separately as part of the payload schema.
    partial = np.flatnonzero(area % 8)
    if len(partial):
        last = data["mask_bits"][data["mask_byte_offsets"][partial + 1] - 1]
        if np.any(last.astype(np.uint16) >> (area[partial] % 8)):
            raise ValueError("nonzero padding bits in packed-mask payload")
    scores = data["scores"]
    if not np.isfinite(scores).all() or np.any((scores < .5) | (scores > 1)):
        raise ValueError("scores must be finite and within the encoder threshold [0.5,1]")
    if not np.isfinite(data["pred_boxes"]).all():
        raise ValueError("pred_boxes must be finite")
    if np.any(data["classes"] < 0):
        raise ValueError("classes must be nonnegative integer labels")
    if np.any(data["source_track_id"] != -1):
        raise ValueError("source_track_id must be -1 in the frozen framewise mask schema")
    for index in range(frame_count):
        lo, hi = map(int, data["frame_offsets"][index:index + 2])
        if not np.array_equal(data["instance_id"][lo:hi], np.arange(hi - lo, dtype=np.int64)):
            raise ValueError("instance_id must retain within-frame detection order 0..N-1")
        if not data["view_available"][index] and lo != hi:
            raise ValueError("unavailable frames must contain zero instances")
        expected = bytes(data["frame_mask_sha256"][index])
        if len(expected) != 64 or any(c not in b"0123456789abcdef" for c in expected):
            raise ValueError("frame_mask_sha256 must contain lowercase SHA256 hex strings")
        decoded = decode_frame_masks(data, index)
        if array_sha256(decoded).encode("ascii") != expected:
            raise ValueError(f"decoded mask SHA256 mismatch at native frame {int(frames[index])}")
        for local, instance in enumerate(range(lo, hi)):
            if not area[instance]:
                continue
            crop = decoded[local, y0[instance]:y1[instance], x0[instance]:x1[instance]]
            if not (crop[0].any() and crop[-1].any() and crop[:, 0].any() and crop[:, -1].any()):
                raise ValueError("mask crop is not the exact nonempty-mask bounding box")


def _read_chunk(chunk) -> Mapping[str, np.ndarray]:
    if isinstance(chunk, (str, PathLike)):
        archive = np.load(Path(chunk), allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("packed-mask chunk path must refer to an NPZ archive")
        try:
            if len(set(archive.files)) != len(archive.files):
                raise ValueError("duplicate array names in packed-mask NPZ archive")
            return {name: archive[name] for name in archive.files}
        finally:
            archive.close()
    if not isinstance(chunk, Mapping):
        raise ValueError("chunks must be NPZ paths or mappings of numpy arrays")
    # Freeze the mapping's entries once. This also avoids repeatedly decompressing
    # arrays if a caller supplies an already-open NpzFile mapping.
    return {name: chunk[name] for name in chunk}


def concatenate_packed_masks(
    chunks: Iterable[Mapping[str, np.ndarray] | str | PathLike],
) -> dict[str, np.ndarray]:
    """Join ordered chunks after strict schema, continuity and decoded-data checks.

    Only frame_offsets and mask_byte_offsets are rebased. Every other per-frame
    or per-instance value, float bit pattern, dtype and order is preserved. The
    returned arrays own their concatenated data and contain no object arrays.
    This function does not checksum input files or create provenance records.
    It decodes only one frame at a time for verification, never the whole video.
    """
    collected = []
    frame_offsets = [np.zeros(1, dtype=np.int64)]
    byte_offsets = [np.zeros(1, dtype=np.int64)]
    instance_total, byte_total, previous_frame = 0, 0, None
    image_shape = None
    for number, chunk in enumerate(chunks):
        try:
            data = _read_chunk(chunk)
            validate_packed_masks(data)
        except (ValueError, IndexError) as error:
            raise ValueError(f"invalid packed-mask chunk {number}: {error}") from error
        if image_shape is None:
            image_shape = data["image_shape_hw"].copy()
        elif not np.array_equal(image_shape, data["image_shape_hw"]):
            raise ValueError("packed-mask chunk image shapes differ")
        if previous_frame is not None and int(data["frame_idx"][0]) != previous_frame + 1:
            raise ValueError(
                "chunk boundary is not continuous: duplicate, missing or reordered frames")
        previous_frame = int(data["frame_idx"][-1])
        frame_offsets.append(data["frame_offsets"][1:] + instance_total)
        byte_offsets.append(data["mask_byte_offsets"][1:] + byte_total)
        instance_total += len(data["mask_boxes"])
        byte_total += len(data["mask_bits"])
        collected.append(data)
    if not collected:
        raise ValueError("at least one nonempty packed-mask chunk is required")
    result = {}
    for name in SCHEMA_DTYPES:
        if name == "image_shape_hw":
            result[name] = image_shape
        elif name == "frame_offsets":
            result[name] = np.concatenate(frame_offsets)
        elif name == "mask_byte_offsets":
            result[name] = np.concatenate(byte_offsets)
        else:
            result[name] = np.concatenate([data[name] for data in collected], axis=0)
    return result

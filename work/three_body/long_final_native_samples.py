"""Fixed final raw samples only; no flow, checkpoint comparison, CLI or GPU.

The successful final chain is required before chunk payloads are opened. The
caller owns output snapshots/seals, global detector assembly verification and
the final evidence.final_guard(). No candidate/checkpoint arrays are read here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from chunk_masks import SCHEMA_DTYPES, validate_packed_masks  # noqa: E402
from observations import CachedNodes  # noqa: E402
from prepare_sequence import decode_frame_masks  # noqa: E402
from source_sequence import validate_source_manifest  # noqa: E402

from medaka3d.types import VIEWS  # noqa: E402

FIRST, STOP, CHUNK, TOTAL = 49800, 348600, 1660, 348600
PLAN = REPO / "work/three_body/LONG_FINAL_VALIDATION_PLAN.md"
PLAN_SHA = "a3cc8deffdf9b18db3a34ed3ee4c1112f406b5aa69ebe616e9b3fe38010a221d"
EXPERIMENT = "20251225_K_1_2_FCMC"
RAW_CHECK_ORIGIN = {
    "path": "work/three_body/audit_reflection_temporal.py",
    "sha256": "a8b6da73b522877c03bd9d3632b14f05471789b95490e68c304a44e45697814a",
    "reuse": "absolute native seek, position/uint8 BGR/SHA checks; no main import",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def fixed_sample_frames():
    """543 native frames, independent of predictions/loss/labels/mask counts."""
    boundaries = np.arange(FIRST + CHUNK, STOP, CHUNK, dtype=np.int64)
    frames = np.unique(
        np.r_[
            np.arange(FIRST, FIRST + 3),
            np.arange(STOP - 3, STOP),
            (boundaries[:, None] + [-1, 0, 1]).ravel(),
        ]
    ).astype(np.int64)
    require(len(boundaries) == 179 and len(frames) == 543, "fixed sample plan changed")
    return frames


def checked_relative(root, value):
    require(isinstance(value, str), "relative path must be text")
    value = value.replace("\\", "/")
    rel = PurePosixPath(value)
    require(
        rel.parts and not rel.is_absolute() and ".." not in rel.parts and ":" not in value,
        "unsafe sample input path",
    )
    path = (Path(root) / rel).resolve(strict=True)
    require(path.is_relative_to(Path(root).resolve(strict=True)), "sample input escaped root")
    return path


def validate_consumed_ranges(records):
    require(isinstance(records, list) and len(records) == 180, "180 consumed chunks required")
    for index, row in enumerate(records):
        require(
            type(row.get("index")) is int
            and row["index"] == index
            and type(row.get("start")) is int
            and row["start"] == FIRST + index * CHUNK
            and type(row.get("stop")) is int
            and row["stop"] == FIRST + (index + 1) * CHUNK,
            "consumed chunk order/native range differs",
        )


def select_packed_frame(data, index, native, expected_count):
    """Preserve all 13 fields; rebase only offsets, then validate this decoded frame."""
    require(set(data) == set(SCHEMA_DTYPES), "packed sample requires exactly 13 fields")
    for name, dtype in SCHEMA_DTYPES.items():
        require(
            isinstance(data[name], np.ndarray) and data[name].dtype == dtype,
            "packed sample dtype differs: " + name,
        )
    frames, offsets, byte_offsets = (
        data[k] for k in ("frame_idx", "frame_offsets", "mask_byte_offsets")
    )
    require(
        type(index) is int and 0 <= index < len(frames) and frames[index] == native,
        "packed sample native offset differs",
    )
    count = len(data["mask_boxes"])
    require(
        offsets.shape == (len(frames) + 1,)
        and offsets[0] == 0
        and offsets[-1] == count
        and np.all(np.diff(offsets) >= 0)
        and byte_offsets.shape == (count + 1,)
        and byte_offsets[0] == 0
        and byte_offsets[-1] == len(data["mask_bits"])
        and np.all(np.diff(byte_offsets) >= 0),
        "packed sample offsets corrupt",
    )
    lo, hi = map(int, offsets[index : index + 2])
    require(
        type(expected_count) is int and hi - lo == expected_count,
        "manifest and packed instance count differ",
    )
    b0, b1 = map(int, byte_offsets[[lo, hi]])
    frame_fields = {"frame_idx", "view_available", "frame_mask_sha256"}
    result = {}
    for key, value in data.items():
        if key == "frame_offsets":
            result[key] = np.array([0, hi - lo], dtype=value.dtype)
        elif key == "mask_byte_offsets":
            result[key] = value[lo : hi + 1].copy() - b0
        elif key == "mask_bits":
            result[key] = value[b0:b1].copy()
        elif key == "image_shape_hw":
            result[key] = value.copy()
        elif key in frame_fields:
            result[key] = value[index : index + 1].copy()
        else:
            result[key] = value[lo:hi].copy()
    validate_packed_masks(result)
    decoded = decode_frame_masks(result, 0)
    available = bool(result["view_available"][0])
    return result, decoded if available else None


def validate_capture(capture, *, shape=(1200, 1200)):
    require(capture.isOpened(), "cannot open original source video")
    require(abs(capture.get(cv2.CAP_PROP_FPS) - 166) <= 1e-6, "source fps differs")
    require(capture.get(cv2.CAP_PROP_FRAME_COUNT) == TOTAL, "source native count differs")
    require(
        (capture.get(cv2.CAP_PROP_FRAME_HEIGHT), capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        == tuple(shape),
        "source image dimensions differ",
    )


def read_native_frame(capture, native, expected_sha, *, shape=(1200, 1200)):
    require(type(native) is int and 0 <= native < TOTAL, "invalid native source offset")
    require(capture.set(cv2.CAP_PROP_POS_FRAMES, native), "absolute native seek failed")
    require(
        abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - native) <= 1e-6, "native seek position differs"
    )
    ok, image = capture.read()
    require(
        ok
        and isinstance(image, np.ndarray)
        and image.shape == (*shape, 3)
        and image.dtype == np.uint8,
        "native frame EOF/shape/dtype differs",
    )
    require(
        abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - native - 1) <= 1e-6,
        "native decode position differs",
    )
    require(array_sha(image) == expected_sha, "native BGR SHA differs")
    return image


@dataclass
class NativeSampleInputs:
    samples: np.ndarray
    chunks: tuple
    videos: dict
    nodes: object
    evidence: object
    chain: dict
    metadata: dict


def load_native_sample_inputs(supervisor_root, evidence):
    """Verify completed chain and consumed attempts; no raw decode or candidate load."""
    import audit_long_final as final

    evidence.add(PLAN, PLAN_SHA)
    evidence.add(Path(__file__))
    chain = final.load_supervisor_chain(supervisor_root, evidence)
    evidence.add(Path(final.__file__))
    record, run = chain["final_record"], chain["final_run"]
    records = final.json_output(record, "consumed_chunks", run / "consumed_chunks.json")
    validate_consumed_ranges(records)
    fixed = chain["binding"]["inputs"]
    for path, digest in fixed["detector_binding"]["code_sha256"].items():
        evidence.add(REPO / path, digest)
    stage_root = Path(fixed["input_stage_root"]).resolve(strict=True)
    chunks, videos = [], {}
    for row in records:
        attempt = checked_relative(stage_root, row["attempt_path"])
        manifest_path, seal_path = attempt / "manifest.json", attempt / "staging.provenance.json"
        evidence.add(manifest_path, row["manifest_sha256"])
        evidence.add(seal_path, row["provenance_sha256"])
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        require(
            seal.get("status") == "ok"
            and seal.get("stage") == "prepare_independent_mask_chunk"
            and seal.get("finished_at"),
            "sample chunk lacks successful detector seal",
        )
        final.equal(
            seal["params"]["binding"], fixed["detector_binding"], "detector binding differs"
        )
        # The fixed binding stores path/SHA only; provenance also has size/mtime.
        declared_inputs = {
            key: dict(path=str(Path(value["path"]).resolve(strict=True)), sha256=value["sha256"])
            for key, value in seal["inputs"].items()
        }
        final.equal(declared_inputs, fixed["detector_inputs"], "detector inputs differ")
        for item in seal["inputs"].values():
            require(
                Path(item["path"]).stat().st_size == item["size"], "detector input size differs"
            )
            evidence.add(item["path"], item["sha256"])
        outputs = {
            key: dict(path=str(Path(v["path"]).resolve(strict=True)), sha256=v["sha256"])
            for key, v in seal["outputs"].items()
        }
        final.equal(outputs, row["output_hashes"], "consumed chunk output proofs differ")
        require(
            Path(outputs["manifest"]["path"]) == manifest_path
            and outputs["manifest"]["sha256"] == row["manifest_sha256"],
            "manifest is not bound to chunk output",
        )
        for item in outputs.values():
            require(Path(item["path"]).is_relative_to(attempt), "chunk output escaped attempt")
            evidence.add(item["path"], item["sha256"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(
            manifest["experiment"] == EXPERIMENT
            and manifest["fps"] == 166
            and manifest["image_shape_hw"] == [1200, 1200]
            and manifest["frames"] == list(range(row["start"], row["stop"]))
            and manifest["protocol_sha256"] == fixed["protocol_sha256"]
            and manifest["tracking_performed"] is False
            and manifest["human_truth_created"] is False,
            "chunk native metadata differs",
        )
        final.equal(
            manifest["binding"], fixed["detector_binding"], "manifest detector binding differs"
        )
        final.equal(
            manifest["runtime"],
            fixed["detector_binding"]["runtime"],
            "manifest detector runtime differs",
        )
        validate_source_manifest(attempt, manifest)
        for view in VIEWS:
            for category, suffix in (
                ("raw", "raw"),
                ("masks", "masks"),
                ("backgrounds", "background"),
            ):
                item = manifest[category][view]
                path = checked_relative(attempt, item["path"])
                final.equal(
                    outputs[view + "_" + suffix],
                    dict(path=str(path), sha256=item["sha256"]),
                    "manifest payload/output binding differs",
                )
            raw = manifest["raw"][view]
            require(
                len(raw["frame_bgr_sha256"]) == CHUNK
                and all(
                    isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)
                    for x in raw["frame_bgr_sha256"]
                ),
                "invalid native BGR SHA list",
            )
            link = row["raw_hardlinks"][view]
            original, alias = (
                Path(link["source_path"]).resolve(strict=True),
                checked_relative(attempt, raw["path"]),
            )
            stat = alias.stat()
            original_input = fixed["detector_inputs"]["video:" + view]
            require(
                Path(link["path"]).resolve() == alias
                and os.path.samefile(original, alias)
                and Path(original_input["path"]).resolve(strict=True) == original
                and original_input["sha256"] == raw["sha256"]
                and (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                == (link["device"], link["inode"], link["bytes"], link["mtime_ns"])
                and link["sha256"] == raw["sha256"],
                "raw source hardlink differs",
            )
            evidence.add(original, raw["sha256"])
            require(view not in videos or videos[view] == original, "chunk changed canonical video")
            videos[view] = original
            require(
                manifest["backgrounds"][view]["sha256"] == fixed["background_sha256"][view],
                "registered median background differs",
            )
        chunks.append(dict(root=attempt, manifest=manifest, record=row))
    seed = fixed["seed"]
    input_root = Path(seed["input_root"]).resolve(strict=True)
    for path, digest in seed["SL_and_calibration_sha256"].items():
        evidence.add(path, digest)
    nodes = CachedNodes(input_root, EXPERIMENT, np.arange(FIRST, STOP, dtype=np.int64))
    for path in nodes.paths.values():
        require(str(path.resolve()) in seed["SL_and_calibration_sha256"], "unbound SL node input")
    return NativeSampleInputs(
        fixed_sample_frames(),
        tuple(chunks),
        videos,
        nodes,
        evidence,
        chain,
        dict(
            schema="long_final_native_sample_inputs_v1",
            native_samples=543,
            raw_images=1629,
            independent_ID_truth=False,
            candidate_or_checkpoint_arrays_opened=False,
            flow_boundary_checks=False,
            checkpoint_previous_checks=False,
            global_detector_assembly_verified=False,
        ),
    )


def _scan_view(context, view, capture_factory, on_sample):
    capture = capture_factory(str(context.videos[view]))
    rows, packed, chunk_index = [], None, None
    try:
        validate_capture(capture)
        for native in context.samples:
            native = int(native)
            index = (native - FIRST) // CHUNK
            chunk = context.chunks[index]
            manifest, root = chunk["manifest"], chunk["root"]
            offset = native - chunk["record"]["start"]
            if index != chunk_index:
                path = checked_relative(root, manifest["masks"][view]["path"])
                with np.load(path, allow_pickle=False) as data:
                    require(
                        len(data.files) == 13 and len(set(data.files)) == 13,
                        "duplicate/missing packed fields",
                    )
                    packed = {key: data[key] for key in data.files}
                require(
                    np.array_equal(packed["frame_idx"], manifest["frames"])
                    and np.array_equal(packed["image_shape_hw"], manifest["image_shape_hw"]),
                    "packed native range/image shape differs",
                )
                chunk_index = index
            image = read_native_frame(
                capture, native, manifest["raw"][view]["frame_bgr_sha256"][offset]
            )
            selected, masks = select_packed_frame(
                packed, offset, native, manifest["masks"][view]["instances_per_frame"][offset]
            )
            detections = np.asarray(context.nodes.at(view, native))
            require(
                detections.ndim == 3
                and detections.shape[1:] == (3, 2)
                and detections.dtype.kind == "f"
                and not np.isinf(detections).any(),
                "invalid native SL node array",
            )
            row = dict(
                frame_idx=native,
                view=view,
                chunk_index=index,
                chunk_frame_offset=offset,
                video_offset=native,
                raw_bgr_sha256=array_sha(image),
                mask_available=masks is not None,
                mask_count=None if masks is None else len(masks),
                mask_decoded_sha256=None if masks is None else array_sha(masks),
                packed_frame_mask_sha256=bytes(selected["frame_mask_sha256"][0]).decode(),
                detection_shape=list(detections.shape),
                detection_dtype=str(detections.dtype),
                detection_sha256=array_sha(detections),
                selected_packed_arrays={
                    key: dict(
                        shape=list(value.shape), dtype=str(value.dtype), sha256=array_sha(value)
                    )
                    for key, value in selected.items()
                },
            )
            rows.append(row)
            if on_sample is not None:
                on_sample(
                    copy.deepcopy(row),
                    dict(
                        image_bgr=image.copy(),
                        masks=None if masks is None else masks.copy(),
                        detections=detections.copy(),
                        packed=selected,
                    ),
                )
    finally:
        capture.release()
    return rows


def read_native_samples(context, *, on_sample=None, capture_factory=cv2.VideoCapture):
    """Read each original MP4 once, seek 543 fixed positions, return 1629 hash rows.

    Callback receives owned per-view arrays for optional saving/next-stage
    checkpoint comparison. All arrays are bounded to one chunk or one sample.
    No frame after the final sampled frame is read to manufacture an EOF test.
    """
    require(
        np.array_equal(context.samples, fixed_sample_frames()), "native sample selection changed"
    )
    require(
        len(context.chunks) == 180 and set(context.videos) == set(VIEWS),
        "complete native sample context required",
    )
    rows = []
    for view in VIEWS:
        rows.extend(_scan_view(context, view, capture_factory, on_sample))
    require(len(rows) == 1629, "incomplete fixed native sample read")
    return rows

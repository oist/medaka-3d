"""CPU-only execution diagnostic on sealed masks; no detector or tracker is run."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(LAB))
import prepare_sequence as original  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

CHUNK = (REPO / "work/three_body/inputs/long_source_masks_49800-348600_v1/chunks"
         / "f049800-051460/attempt_0001")
OUT = REPO / "work/three_body/detector_cpu_execution_profile_v1"
BLOCKS = [(0, 20), (820, 840), (1640, 1660)]


def buffer_hash(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


class RectPackedMasks(original.PackedMasks):
    """Diagnostic copy of append; bbox extraction and optional hash buffer only."""
    use_buffer_hash = False

    def append(self, frame, masks, scores, pred_boxes, classes):
        masks = np.asarray(masks)
        count = len(masks)
        scores = np.asarray(scores, dtype=np.float32)
        pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
        classes = np.asarray(classes, dtype=np.int64)
        if self.frames and frame != self.frames[-1] + 1:
            raise ValueError("staging requires consecutive source frames")
        if masks.dtype != bool or masks.shape != (count, *self.shape):
            raise ValueError("masks must be bool in full source image coordinates")
        if scores.shape != (count,) or pred_boxes.shape != (count, 4) or classes.shape != (count,):
            raise ValueError("prediction field counts differ")
        if not np.isfinite(scores).all() or np.any((scores < 0.5) | (scores > 1)):
            raise ValueError("score is invalid for the fixed threshold 0.5")
        if not np.isfinite(pred_boxes).all():
            raise ValueError("predicted boxes contain nonfinite coordinates")
        for index, mask in enumerate(masks):
            x, y, width, height = cv2.boundingRect(mask.view(np.uint8))
            box = [x, y, x + width, y + height] if width and height else [0] * 4
            x0, y0, x1, y1 = box
            bits = np.packbits(mask[y0:y1, x0:x1].reshape(-1), bitorder="little")
            self.bits.append(bits)
            self.boxes.append(box)
            self.byte_offsets.append(self.byte_offsets[-1] + len(bits))
            self.ids.append(index)
        self.frames.append(int(frame))
        self.offsets.append(len(self.boxes))
        self.scores.extend(scores)
        self.pred_boxes.extend(pred_boxes)
        self.classes.extend(classes)
        hashing = buffer_hash if self.use_buffer_hash else original.array_sha256
        self.mask_hashes.append(hashing(masks))


class RectBufferPackedMasks(RectPackedMasks):
    use_buffer_hash = True


def assert_arrays_equal(first, second):
    assert set(first) == set(second)
    for key in first:
        a, b = first[key], second[key]
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), key


def segment(data, lo, hi):
    first, last = map(int, data["frame_offsets"][[lo, hi]])
    begin, end = map(int, data["mask_byte_offsets"][[first, last]])
    result = {}
    for key, value in data.items():
        if key == "image_shape_hw":
            result[key] = value
        elif key == "frame_offsets":
            result[key] = value[lo:hi + 1] - first
        elif key == "mask_byte_offsets":
            result[key] = value[first:last + 1] - begin
        elif key == "mask_bits":
            result[key] = value[begin:end]
        elif key in ("frame_idx", "view_available", "frame_mask_sha256"):
            result[key] = value[lo:hi]
        else:
            result[key] = value[first:last]
    return result


def encode_block(data, cls, lo, hi):
    encoder = cls()
    decode_s = append_s = 0.0
    for t in range(lo, hi):
        started = time.perf_counter()
        masks = original.decode_frame_masks(data, t)
        decode_s += time.perf_counter() - started
        first, last = data["frame_offsets"][t:t + 2]
        started = time.perf_counter()
        encoder.append(int(data["frame_idx"][t]), masks, data["scores"][first:last],
                       data["pred_boxes"][first:last], data["classes"][first:last])
        append_s += time.perf_counter() - started
    assert_arrays_equal(encoder.arrays(), segment(data, lo, hi))
    return decode_s, append_s


def synthetic():
    rng = np.random.default_rng(1917)
    cases = [np.zeros((0, 64, 64), bool), np.zeros((3, 64, 64), bool),
             np.ones((1, 64, 64), bool), rng.random((4, 64, 64)) < .01]
    corners = np.zeros((2, 64, 64), bool)
    corners[0, 0, 63] = corners[0, 63, 0] = True
    corners[1, 31, 31] = True
    cases.extend([corners, corners[:, ::-1, ::-1]])
    for masks in cases:
        count = len(masks)
        outputs = []
        for cls in (original.PackedMasks, RectPackedMasks, RectBufferPackedMasks):
            encoder = cls(shape=(64, 64))
            encoder.append(0, masks, np.full(count, .75), np.zeros((count, 4)),
                           np.zeros(count, int))
            outputs.append(encoder.arrays())
        for result in outputs[1:]:
            assert_arrays_equal(outputs[0], result)
    return len(cases)


def raw_components(manifest, view):
    info = manifest["raw"][view]
    source = CHUNK / info["path"]
    before = source.stat()
    cap = cv2.VideoCapture(str(source))
    decode_s = hash_s = 0.0
    raw_count = 0
    try:
        original.validate_capture(cap, source)
        for lo, hi in BLOCKS:
            for t in range(lo, hi):
                frame = manifest["frames"][t]
                started = time.perf_counter()
                image = original.read_frame(cap, frame, seek=t == lo)
                decode_s += time.perf_counter() - started
                started = time.perf_counter()
                digest = original.array_sha256(image)
                hash_s += time.perf_counter() - started
                assert digest == info["frame_bgr_sha256"][t]
                raw_count += 1
        started = time.perf_counter()
        background_images = [original.read_frame(cap, f, seek=True)
                             for f in manifest["background_frames"]]
        background_decode_s = time.perf_counter() - started
        started = time.perf_counter()
        background = np.median(np.stack(background_images), axis=0).astype(np.uint8)
        background_median_s = time.perf_counter() - started
        stored = cv2.imread(str(CHUNK / manifest["backgrounds"][view]["path"]))
        assert background.tobytes() == stored.tobytes()
    finally:
        cap.release()
    after = source.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_ino, after.st_size, after.st_mtime_ns)
    return dict(raw_images=raw_count, raw_decode_s=decode_s, raw_hash_s=hash_s,
                background_decode_nine_s=background_decode_s,
                background_median_s=background_median_s, background_exact=True)


def main():
    OUT.mkdir(exist_ok=False)
    manifest_path, seal_path = CHUNK / "manifest.json", CHUNK / "staging.provenance.json"
    manifest = json.loads(manifest_path.read_text())
    seal = json.loads(seal_path.read_text())
    assert seal["status"] == "ok"
    assert original.sha256(manifest_path) == seal["outputs"]["manifest"]["sha256"]
    source_hashes = dict(manifest["binding"]["code_sha256"])
    source_hashes[str(Path(__file__).relative_to(REPO))] = original.sha256(__file__)
    assert all(original.sha256(REPO / name) == value for name, value in source_hashes.items())
    input_hashes = {str(manifest_path): original.sha256(manifest_path),
                    str(seal_path): original.sha256(seal_path)}
    masks = {}
    for view in original.VIEWS:
        for category in ("masks", "backgrounds"):
            info = manifest[category][view]
            path = CHUNK / info["path"]
            assert original.sha256(path) == info["sha256"]
            input_hashes[str(path)] = info["sha256"]
        with np.load(CHUNK / manifest["masks"][view]["path"], allow_pickle=False) as arrays:
            masks[view] = {key: arrays[key] for key in arrays.files}
    with provenance("cpu_detector_execution_diagnostic", OUT, name="profile") as record:
        for name, value in source_hashes.items():
            path = REPO / name
            target = OUT / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            record.input("code:" + name, path, sha256=value)
            record.output(target.relative_to(OUT).as_posix(), target, sha256=value)
        for path, value in input_hashes.items():
            record.input(path, path, sha256=value)
        synthetic_cases = synthetic()
        methods = {"original": original.PackedMasks, "rect": RectPackedMasks,
                   "rect_buffer_hash": RectBufferPackedMasks}
        repeats = {name: [] for name in methods}
        for repeat in range(3):
            names = list(methods) if repeat % 2 == 0 else list(reversed(methods))
            for name in names:
                decode_s = append_s = 0.0
                for data in masks.values():
                    for lo, hi in BLOCKS:
                        decode, append = encode_block(data, methods[name], lo, hi)
                        decode_s += decode
                        append_s += append
                repeats[name].append(dict(decode_s=decode_s, append_s=append_s))
        raw = {view: raw_components(manifest, view) for view in original.VIEWS}
        wall = (datetime.fromisoformat(seal["finished_at"])
                - datetime.fromisoformat(seal["started_at"])).total_seconds()
        synchronized = sum(sum(v["inference_s_by_frame"]) for v in manifest["masks"].values())
        result = dict(
            scope="CPU-only proposal diagnostic; no production code or outputs changed",
            manifest_sha256=input_hashes[str(manifest_path)],
            sample_native_blocks=[[manifest["frames"][lo], manifest["frames"][hi-1]+1]
                                  for lo, hi in BLOCKS],
            sample_view_images=180, repeats=repeats, synthetic_cases=synthetic_cases,
            all_13_arrays_dtype_shape_bytes_exact=True, raw=raw,
            full_chunk_wall_s=wall, full_chunk_predictor_to_cpu_sync_s=synchronized,
            full_chunk_unattributed_s=wall-synchronized,
            environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                             opencv=cv2.__version__, opencv_threads=cv2.getNumThreads()),
            limitations=[
                "CPU timings overlap another GPU staging process; not idle-machine speed.",
                "Raw full SHA was not repeated; sealed metadata and native BGR SHA checked.",
                "Sample CPU timings do not fully attribute the original residual wall time.",
                "New bbox/hash code is a diagnostic prototype, not integrated into frozen staging.",
                "No detector batching, GPU model, alternate algorithm, GT or ID evaluation ran.",
            ],
        )
        path = OUT / "results.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        record.output("results", path, sha256=original.sha256(path))
        for name, value in source_hashes.items():
            assert original.sha256(REPO / name) == value
        for path, value in input_hashes.items():
            assert original.sha256(path) == value
        for info in record.outputs.values():
            assert original.sha256(info["path"]) == info["sha256"]
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

"""ID-free, adjacent-native-frame optical-flow evidence at detection nodes.

Predictions are native source-pixel coordinates, not displacements. A successful
LK/round-trip check is local image support, not proof of fish identity. No nodes,
frames, or missing detections are interpolated. The video cache signature uses
size/mtime and hashed boundary samples; it is explicitly not a full-video hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from medaka3d.provenance import provenance

VERSION = "adjacent_native_lk_v1"
PARAMETERS = {
    "image_scale": 0.5,
    "downsampling": "gaussian_pyrDown",
    "window_working_pixels": 15,
    "pyramid_max_level": 3,
    "max_iterations": 30,
    "epsilon_working_pixels": 0.01,
    "min_eigen_threshold": 1e-4,
    "max_patch_l1_8bit": 20.0,
    "max_roundtrip_native_pixels": 1.5,
    "video_decode_threads": 2,
    "video_boundary_sample_bytes": 1024 * 1024,
}
ARRAY_KEYS = (
    "predicted_forward", "valid_forward", "predicted_backward", "valid_backward",
    "patch_error_forward", "patch_error_backward",
    "roundtrip_error_forward", "roundtrip_error_backward",
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _video_signature(path: Path) -> dict:
    stat = path.stat()
    sample_bytes = PARAMETERS["video_boundary_sample_bytes"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for offset in sorted({0, max(0, stat.st_size - sample_bytes)}):
            handle.seek(offset)
            block = handle.read(sample_bytes)
            digest.update(offset.to_bytes(8, "little"))
            digest.update(len(block).to_bytes(8, "little"))
            digest.update(block)
    return {
        "path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "boundary_sample_sha256": digest.hexdigest(),
        "sample_bytes_per_boundary": sample_bytes,
        "hash_scope": "size_mtime_and_boundary_samples_not_full_video",
    }


def _inside(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    margin = PARAMETERS["window_working_pixels"] // 2 + 1
    height, width = shape
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= margin) & (points[:, 0] < width - margin)
        & (points[:, 1] >= margin) & (points[:, 1] < height - margin)
    )


def _lk(first: np.ndarray, second: np.ndarray, points: np.ndarray):
    return cv2.calcOpticalFlowPyrLK(
        first, second, points.reshape(-1, 1, 2).astype(np.float32), None,
        winSize=(PARAMETERS["window_working_pixels"],) * 2,
        maxLevel=PARAMETERS["pyramid_max_level"],
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                  PARAMETERS["max_iterations"], PARAMETERS["epsilon_working_pixels"]),
        flags=0, minEigThreshold=PARAMETERS["min_eigen_threshold"],
    )


def _flow_pair(
    first: np.ndarray, second: np.ndarray, points_native: np.ndarray,
    scale_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Track one batch in prepared grayscale images, checking both LK directions."""
    points = np.asarray(points_native, dtype=np.float32).reshape(-1, 2)
    predicted = np.full_like(points, np.nan)
    valid = np.zeros(len(points), dtype=bool)
    patch_error = np.full(len(points), np.nan, dtype=np.float32)
    roundtrip_error = np.full(len(points), np.nan, dtype=np.float32)
    working = points * scale_xy
    initial = np.flatnonzero(_inside(working, first.shape))
    if not len(initial):
        return predicted, valid, patch_error, roundtrip_error
    target, status, error = _lk(first, second, working[initial])
    if target is None or status is None or error is None:
        return predicted, valid, patch_error, roundtrip_error
    target = target.reshape(-1, 2)
    status = status.ravel().astype(bool)
    error = error.ravel()
    good = status & _inside(target, second.shape) & np.isfinite(error)
    patch_error[initial[good]] = error[good]
    good &= error <= PARAMETERS["max_patch_l1_8bit"]
    local = np.flatnonzero(good)
    if not len(local):
        return predicted, valid, patch_error, roundtrip_error
    returned, back_status, back_error = _lk(second, first, target[local])
    if returned is None or back_status is None or back_error is None:
        return predicted, valid, patch_error, roundtrip_error
    returned = returned.reshape(-1, 2)
    back_error = back_error.ravel()
    native_error = np.linalg.norm(
        (returned - working[initial[local]]) / scale_xy, axis=1
    )
    checked = (
        back_status.ravel().astype(bool) & _inside(returned, first.shape)
        & np.isfinite(back_error) & np.isfinite(native_error)
    )
    roundtrip_error[initial[local[checked]]] = native_error[checked]
    checked &= (
        (back_error <= PARAMETERS["max_patch_l1_8bit"])
        & (native_error <= PARAMETERS["max_roundtrip_native_pixels"])
    )
    accepted = initial[local[checked]]
    predicted[accepted] = target[local[checked]] / scale_xy
    valid[accepted] = True
    return predicted, valid, patch_error, roundtrip_error


def _empty(n: int) -> dict:
    result = {}
    for direction in ("forward", "backward"):
        result[f"predicted_{direction}"] = np.full((n, 3, 2), np.nan, np.float32)
        result[f"valid_{direction}"] = np.zeros((n, 3), bool)
        result[f"patch_error_{direction}"] = np.full((n, 3), np.nan, np.float32)
        result[f"roundtrip_error_{direction}"] = np.full((n, 3), np.nan, np.float32)
    return result


def _load_cache(path: Path, signature: dict, n: int) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata["input_signature"] != signature:
                return None
            result = {key: archive[key].copy() for key in ARRAY_KEYS}
        for direction in ("forward", "backward"):
            p, v = result[f"predicted_{direction}"], result[f"valid_{direction}"]
            if p.shape != (n, 3, 2) or v.shape != (n, 3) or v.dtype != np.bool_:
                return None
            if not np.isfinite(p[v]).all() or not np.isnan(p[~v]).all():
                return None
            for prefix in ("patch_error", "roundtrip_error"):
                if result[f"{prefix}_{direction}"].shape != (n, 3):
                    return None
        metadata["cache"] = {"hit": True, "path": str(path), "sha256": _file_hash(path)}
        result["metadata"] = metadata
        result["diagnostics"] = metadata["diagnostics"]
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return None


def extract_flow(video_path, frame_idx, points, *, cache_path=None) -> dict:
    """Return forward/backward t±1 predictions for each input row's three nodes.

    Input row order and repeated frame indices are retained. Nonfinite nodes,
    source/neighbour frames outside the video, failed LK checks, and patch-border
    exclusions have NaN predictions and false validity. Unexpected decode/seek
    failures raise rather than silently introducing missing data.
    """
    video_path = Path(video_path).resolve()
    frame_idx = np.asarray(frame_idx)
    if frame_idx.ndim != 1 or frame_idx.dtype.kind not in "iu":
        raise ValueError("frame_idx must be a one-dimensional integer array")
    if frame_idx.dtype.kind == "u" and len(frame_idx) and frame_idx.max() > 2**63 - 1:
        raise ValueError("Frame indices exceed int64")
    frame_idx = np.ascontiguousarray(frame_idx, dtype=np.int64)
    points_input = np.asarray(points, dtype=np.float64)
    if points_input.shape != (len(frame_idx), 3, 2):
        raise ValueError("points must have shape (N, 3, 2), matching frame_idx")
    points_input = np.ascontiguousarray(points_input)
    signature = {
        "version": VERSION, "parameters": PARAMETERS,
        "video": _video_signature(video_path),
        "frame_idx_sha256": _array_hash(frame_idx),
        "points_sha256": _array_hash(points_input),
        "points_hash_dtype": "float64", "input_rows": len(frame_idx),
        "code_sha256": _file_hash(Path(__file__)),
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
    }
    cache = Path(cache_path).resolve() if cache_path is not None else None
    if cache is not None and cache.is_file():
        loaded = _load_cache(cache, signature, len(frame_idx))
        if loaded is not None:
            return loaded
    context = (
        provenance("clean_tracklet_optical_flow", cache.parent, name=cache.stem + "_flow")
        if cache is not None else nullcontext(None)
    )
    start_time = time.perf_counter()
    with context as record:
        if record:
            record.input("raw_video", video_path)
            record.input("flow_code", Path(__file__), sha256=signature["code_sha256"])
            record.param("input_signature", signature)
        result, diagnostics = _extract(video_path, frame_idx, points_input)
        if _video_signature(video_path) != signature["video"]:
            raise RuntimeError("Video source changed during optical-flow extraction")
        diagnostics["elapsed_seconds"] = time.perf_counter() - start_time
        metadata = {
            "input_signature": signature, "diagnostics": diagnostics,
            "created_utc": datetime.now(UTC).isoformat(),
            "coordinates": "native source pixels; predictions at input frame plus/minus one",
            "node_order": ["mouth", "pectoral", "tail"],
            "identity_labels_used": False,
            "interpretation": "Local LK image support, not fish identity or detection truth",
            "cache": {"hit": False, "path": str(cache) if cache else None},
        }
        if cache is not None:
            temporary = cache.with_name(cache.name + f".{os.getpid()}.tmp")
            try:
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle, **result,
                        metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)),
                    )
                temporary.replace(cache)
            finally:
                if temporary.exists():
                    temporary.unlink()
            metadata["cache"]["sha256"] = _file_hash(cache)
            record.output("flow_cache", cache, sha256=metadata["cache"]["sha256"])
        result["metadata"] = metadata
        result["diagnostics"] = diagnostics
        return result


def _extract(video_path: Path, frame_idx: np.ndarray, points: np.ndarray):
    result = _empty(len(frame_idx))
    capture = cv2.VideoCapture(
        str(video_path), cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_N_THREADS, PARAMETERS["video_decode_threads"]],
    )
    read_count, pair_count = 0, 0
    first = last = None
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = (int(capture.get(prop)) for prop in
                         (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count < 1 or width < 2 or height < 2:
            raise ValueError("Video has no usable dimensions/frame count")
        working_size = (max(1, round(width * PARAMETERS["image_scale"])),
                        max(1, round(height * PARAMETERS["image_scale"])))
        scale_xy = np.asarray((working_size[0] / width, working_size[1] / height), np.float32)
        in_video = (frame_idx >= 0) & (frame_idx < frame_count)
        ordered = np.flatnonzero(in_video)
        ordered = ordered[np.argsort(frame_idx[ordered], kind="stable")]
        unique, starts = np.unique(frame_idx[ordered], return_index=True)
        groups = (
            dict(zip(unique.tolist(), np.split(ordered, starts[1:]), strict=True))
            if len(unique) else {}
        )

        def process(frame, previous, current, following):
            nonlocal pair_count
            row_indices = groups.get(frame)
            if row_indices is None:
                return
            for direction, neighbour in (("forward", following), ("backward", previous)):
                if neighbour is None:
                    continue
                values = _flow_pair(current, neighbour, points[row_indices], scale_xy)
                for key, value in zip(
                    ("predicted", "valid", "patch_error", "roundtrip_error"),
                    values, strict=True,
                ):
                    result[f"{key}_{direction}"][row_indices] = value.reshape(
                        (len(row_indices), 3, 2) if key == "predicted"
                        else (len(row_indices), 3)
                    )
                pair_count += 1

        if len(unique):
            first, last = max(0, int(unique[0]) - 1), min(frame_count - 1, int(unique[-1]) + 1)
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
                raise RuntimeError("Could not seek to the first native context frame")
            if capture.get(cv2.CAP_PROP_POS_FRAMES) != first:
                raise RuntimeError("Native frame seek position does not match")
            previous = current = None
            for frame in range(first, last + 1):
                ok, image = capture.read()
                if not ok or image is None or capture.get(cv2.CAP_PROP_POS_FRAMES) != frame + 1:
                    raise RuntimeError(f"Native frame decode/position failed at {frame}")
                if image.shape[:2] != (height, width) or image.dtype != np.uint8:
                    raise RuntimeError("Decoded image dimensions/dtype changed")
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                following = cv2.pyrDown(gray, dstsize=working_size)
                if current is not None:
                    process(frame - 1, previous, current, following)
                previous, current = current, following
                read_count += 1
            process(last, previous, current, None)
    finally:
        capture.release()
    diagnostics = {
        "source_frame_count": frame_count, "source_fps": fps,
        "source_size_wh": [width, height], "working_size_wh": list(working_size),
        "native_scale_xy": scale_xy.tolist(), "input_rows": len(frame_idx),
        "unique_in_range_frames": len(unique), "out_of_range_rows": int((~in_video).sum()),
        "finite_input_nodes": int(np.isfinite(points).all(axis=2).sum()),
        "decoded_native_frames": read_count,
        "decoded_range_half_open": [first, last + 1] if first is not None else None,
        "directional_frame_batches": pair_count,
        "valid_forward_nodes": int(result["valid_forward"].sum()),
        "valid_backward_nodes": int(result["valid_backward"].sum()),
        "point_rows_with_gaps_are_not_linked": True,
        "opencv_threads": cv2.getNumThreads(),
    }
    return result, diagnostics

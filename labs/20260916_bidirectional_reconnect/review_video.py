"""Read-only raw-camera review of native measured 3D tracking samples.

These displays provide visual evidence, not human identity ground truth. Source
pixels, coordinates and native timestamps are never interpolated. The caller
owns provenance and supplies the calibrated rig and explicitly selected tracks.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from medaka3d.aquarium import movie_indices
from medaka3d.aquarium_render import SLOT_COLOURS, render_frame
from medaka3d.types import VIEWS

SOURCE_TOTAL_FRAMES = 348600
SOURCE_SHAPE = (1200, 1200, 3)
PANEL_SIZE = (640, 480)


def _validate_inputs(raw_paths, rig, frames, pose, *, contiguous):
    frames = np.asarray(frames)
    pose = np.asarray(pose, dtype=float)
    if frames.ndim != 1 or not len(frames) or frames.dtype.kind not in "iu":
        raise ValueError("frames must be a nonempty one-dimensional integer array")
    if np.any(frames[1:] <= frames[:-1]):
        raise ValueError("frames must be strictly increasing")
    if contiguous and np.any(np.diff(frames) != 1):
        raise ValueError("review movies require consecutive native frames")
    if frames[0] < 0 or frames[-1] >= SOURCE_TOTAL_FRAMES:
        raise ValueError("native source frame is outside the registered recording")
    if pose.shape != (len(frames), 3, 3, 3):
        raise ValueError("pose must have shape (len(frames), 3 slots, 3 nodes, 3 coordinates)")
    if np.isinf(pose).any():
        raise ValueError("pose may contain NaNs for missing observations, but not infinities")
    if set(raw_paths) != set(VIEWS) or tuple(rig.views) != VIEWS:
        raise ValueError("raw_paths and calibrated rig must explicitly contain the three views")
    return frames.astype(np.int64), pose


def _new_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"review output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _source_captures(raw_paths, first_frame, source_fps):
    """Seek once, then let callers verify the native position after every read."""
    captures, metadata = {}, {}
    try:
        for view in VIEWS:
            path = Path(raw_paths[view]).resolve(strict=True)
            capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
            captures[view] = capture
            if not capture.isOpened():
                raise RuntimeError(f"cannot open source video for {view}: {path}")
            count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv2.CAP_PROP_FPS)
            shape = (
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                3,
            )
            if count != SOURCE_TOTAL_FRAMES or shape != SOURCE_SHAPE:
                raise ValueError(f"source metadata mismatch for {view}: {count}, {shape}")
            if not np.isclose(fps, source_fps, rtol=0, atol=1e-6):
                raise ValueError(f"source FPS mismatch for {view}: {fps} != {source_fps}")
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(first_frame)):
                raise RuntimeError(f"native source seek failed for {view}")
            if capture.get(cv2.CAP_PROP_POS_FRAMES) != first_frame:
                raise RuntimeError(f"native source seek position mismatch for {view}")
            metadata[view] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "source_frames": int(count),
                "source_fps": fps,
                "source_shape": list(shape),
                "initial_seek_frame": int(first_frame),
            }
        yield captures, metadata
    finally:
        for capture in captures.values():
            capture.release()


def _read_native(captures, frame):
    images = {}
    for view, capture in captures.items():
        ok, image = capture.read()
        if not ok or capture.get(cv2.CAP_PROP_POS_FRAMES) != frame + 1:
            raise RuntimeError(f"source native frame alignment failed: {view}/{frame}")
        if image is None or image.shape != SOURCE_SHAPE or image.dtype != np.uint8:
            raise ValueError(f"decoded source image shape/dtype mismatch: {view}/{frame}")
        images[view] = image
    return images


def _text(image, text, xy, *, scale=0.48, color=(235, 235, 235), thickness=1):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _letterbox(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    resized = cv2.resize(
        image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )
    output = np.full((height, width, 3), (22, 22, 22), dtype=np.uint8)
    x, y = (width - size[0]) // 2, (height - size[1]) // 2
    output[y : y + size[1], x : x + size[0]] = resized
    return output, scale, np.array([x, y])


def _draw_pose(image, pixels, *, scale=1.0, offset=None, focus_slots=None):
    offset = np.zeros(2) if offset is None else np.asarray(offset)
    projected = np.asarray(pixels) * scale + offset
    for slot in range(3):
        if focus_slots is not None and slot not in focus_slots:
            continue
        points = projected[slot]
        color = SLOT_COLOURS[slot]
        for a, b in ((0, 1), (1, 2)):
            if np.isfinite(points[[a, b]]).all():
                cv2.line(
                    image,
                    tuple(np.rint(points[a]).astype(int)),
                    tuple(np.rint(points[b]).astype(int)),
                    color,
                    1,
                    cv2.LINE_AA,
                )
        for node, point in enumerate(points):
            if not np.isfinite(point).all():
                continue
            xy = tuple(np.rint(point).astype(int))
            if node == 0:
                cv2.circle(image, xy, 4, color, 1, cv2.LINE_AA)
            elif node == 1:
                cv2.drawMarker(image, xy, color, cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
            else:
                cv2.circle(image, xy, 2, color, 1, cv2.LINE_AA)
        finite = points[np.isfinite(points).all(axis=1)]
        if len(finite):
            x, y = np.rint(finite[0]).astype(int)
            _text(image, str(slot), (int(x + 7), int(y - 7)), color=color, scale=0.45)


def _review_frame(images, pixels, pose, frames, index, *, source_fps, labels, preserve_native):
    width, height = PANEL_SIZE
    canvas = np.full((2 * height, 2 * width, 3), (22, 22, 22), dtype=np.uint8)
    frame = int(frames[index])
    for panel_index, view in enumerate(VIEWS):
        panel = np.full((height, width, 3), (22, 22, 22), dtype=np.uint8)
        content, scale, offset = _letterbox(images[view], width, height - 64)
        _draw_pose(content, pixels[view][index], scale=scale, offset=offset)
        panel[32 : height - 32] = content
        _text(panel, f"{view} | projected measured nodes", (12, 22))
        _text(panel, f"frame {frame} | source {frame / source_fps:.4f} s", (12, height - 11))
        y, x = divmod(panel_index, 2)
        canvas[y * height : (y + 1) * height, x * width : (x + 1) * width] = panel
    trails = [pose[max(0, index - 49) : index + 1, slot, 1].copy() for slot in range(3)]
    cg = render_frame(
        pose[index],
        trails,
        frame=frame,
        source_fps=source_fps,
        elapsed_seconds=(frame - int(frames[0])) / source_fps,
        duration_seconds=len(frames) / source_fps,
        alerts=[],
        title="Measured 3D pose",
        width=1280,
        height=720,
    )
    panel = np.full((height, width, 3), (22, 22, 22), dtype=np.uint8)
    panel[46:406] = cv2.resize(cg, (640, 360), interpolation=cv2.INTER_AREA)
    _text(panel, "Native samples | no temporal interpolation", (12, 23))
    mode = "All native frames / slow replay" if preserve_native else "Native samples / real time"
    _text(panel, mode, (12, 430))
    label = (labels or {}).get(frame, "")
    if label:
        _text(panel, str(label)[:78], (12, 455), color=(120, 180, 255), scale=0.43)
    else:
        _text(panel, "Colour denotes input slot; not biological identity", (12, 455), scale=0.43)
    canvas[height:, width:] = panel
    return canvas


def render_review_clip(
    raw_paths,
    rig,
    frames,
    pose,
    out_path,
    labels=None,
    output_fps=30,
    source_fps=166,
    preserve_native=False,
):
    """Render three cameras and CG; decode every native frame before subsampling.

    ``preserve_native=True`` plays every native sample at ``output_fps``. Otherwise
    nearest native samples produce approximately real-time duration. Coordinates
    are never smoothed. Labels must be ASCII for OpenCV's built-in font.
    """
    frames, pose = _validate_inputs(raw_paths, rig, frames, pose, contiguous=True)
    if not np.isfinite(output_fps) or not 0 < output_fps <= source_fps:
        raise ValueError("require finite 0 < output_fps <= source_fps")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg must already be available on PATH")
    out_path = _new_output(out_path)
    log_path = _new_output(out_path.with_suffix(".ffmpeg.log"))
    indices = (
        np.arange(len(frames))
        if preserve_native
        else movie_indices(len(frames), source_fps, output_fps)
    )
    selected = set(map(int, indices))
    pixels = {view: rig.camera(view).project(pose) for view in VIEWS}
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "1280x960",
        "-r",
        str(output_fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    with (
        _source_captures(raw_paths, int(frames[0]), source_fps) as (captures, source_metadata),
        log_path.open("xb") as log,
    ):
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for index, frame in enumerate(frames):
                images = _read_native(captures, int(frame))
                if index not in selected:
                    continue
                rendered = _review_frame(
                    images,
                    pixels,
                    pose,
                    frames,
                    index,
                    source_fps=source_fps,
                    labels=labels,
                    preserve_native=preserve_native,
                )
                process.stdin.write(rendered.tobytes())
            process.stdin.close()
            if process.wait(timeout=120) != 0:
                raise RuntimeError(f"ffmpeg failed; inspect {log_path}")
        except BaseException:
            process.kill()
            process.wait()
            raise
    decoded = 0
    capture = cv2.VideoCapture(str(out_path))
    try:
        if not capture.isOpened() or not np.isclose(capture.get(cv2.CAP_PROP_FPS), output_fps):
            raise RuntimeError("rendered review movie failed FPS/open validation")
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image is None or image.shape != (960, 1280, 3):
                raise RuntimeError("rendered review movie has unexpected dimensions")
            decoded += 1
    finally:
        capture.release()
    if decoded != len(indices):
        raise RuntimeError(f"movie decoded {decoded} frames; expected {len(indices)}")
    return {
        "output": str(out_path.resolve()),
        "source": source_metadata,
        "native_frame_range_half_open": [int(frames[0]), int(frames[-1]) + 1],
        "all_native_frames_read_per_view": len(frames),
        "output_native_frames": frames[indices].tolist(),
        "output_fps": float(output_fps),
        "output_frames": decoded,
        "output_duration_seconds": decoded / output_fps,
        "native_duration_seconds": len(frames) / source_fps,
        "preserve_native": bool(preserve_native),
        "coordinates_interpolated": False,
        "ffmpeg_command": command,
        "movie_readback_verified": True,
    }


def render_contact_sheet(raw_paths, rig, frames, pose, out_path, focus_slots=None):
    """Fixed source-pixel crops: paired clean/overlay rows per view, frame columns.

    Native frames may be sparse. Every frame from the first through the last is
    decoded sequentially, and the requested frames are retained. A view's crop is
    fixed across all selected frames so apparent motion is not camera recentering.
    """
    frames, pose = _validate_inputs(raw_paths, rig, frames, pose, contiguous=False)
    if len(frames) > 8:
        raise ValueError("contact sheets accept at most eight native frames")
    slots = list(range(3)) if focus_slots is None else sorted(set(focus_slots))
    if not slots or any(slot not in range(3) for slot in slots):
        raise ValueError("focus_slots must be a nonempty subset of 0, 1, 2")
    out_path = _new_output(out_path)
    pixels = {view: rig.camera(view).project(pose) for view in VIEWS}
    retained, hashes, selected = {}, {}, set(map(int, frames))
    with _source_captures(raw_paths, int(frames[0]), 166) as (captures, source_metadata):
        for frame in range(int(frames[0]), int(frames[-1]) + 1):
            images = _read_native(captures, frame)
            if frame in selected:
                retained[frame] = images
                hashes[frame] = {
                    view: hashlib.sha256(image.tobytes()).hexdigest()
                    for view, image in images.items()
                }
    cell_width, cell_height, header = 320, 350, 30
    sheet = np.full((6 * cell_height, len(frames) * cell_width, 3), 22, dtype=np.uint8)
    crops = {}
    for view_index, view in enumerate(VIEWS):
        visible = pixels[view][:, slots].reshape(-1, 2)
        visible = visible[np.isfinite(visible).all(axis=1)]
        if len(visible):
            low = np.maximum(0, np.floor(visible.min(axis=0) - 60)).astype(int)
            high = np.minimum(1200, np.ceil(visible.max(axis=0) + 60)).astype(int)
            if np.any(high <= low):
                low, high = np.array([0, 0]), np.array([1200, 1200])
        else:
            low, high = np.array([0, 0]), np.array([1200, 1200])
        x0, y0 = map(int, low)
        x1, y1 = map(int, high)
        crops[view] = [x0, y0, x1, y1]
        for column, frame in enumerate(frames):
            raw = retained[int(frame)][view][y0:y1, x0:x1]
            clean, scale, offset = _letterbox(raw, cell_width, cell_height - header)
            overlay = clean.copy()
            _draw_pose(
                overlay, pixels[view][column] - low, scale=scale, offset=offset, focus_slots=slots
            )
            for kind_index, (kind, image) in enumerate((("clean", clean), ("overlay", overlay))):
                x = column * cell_width
                y = (2 * view_index + kind_index) * cell_height
                sheet[y + header : y + cell_height, x : x + cell_width] = image
                _text(sheet, f"{view} {kind} | f{int(frame)}", (x + 6, y + 20), scale=0.42)
    if not cv2.imwrite(str(out_path), sheet):
        raise RuntimeError(f"could not write contact sheet {out_path}")
    decoded = cv2.imread(str(out_path))
    if decoded is None or decoded.shape != sheet.shape or not np.array_equal(decoded, sheet):
        raise RuntimeError("contact-sheet lossless PNG readback verification failed")
    return {
        "output": str(out_path.resolve()),
        "source": source_metadata,
        "native_frames": frames.tolist(),
        "focus_slots": slots,
        "crops_xyxy_source_pixels": crops,
        "shape": list(sheet.shape),
        "all_native_frames_read_per_view": int(frames[-1] - frames[0] + 1),
        "selected_source_bgr_sha256": hashes,
        "paired_rows": "clean then overlay for top_cam, side1_cam, side2_cam",
        "nodes": "mouth circle; pectoral cross; tail small circle",
        "identity_ground_truth": False,
    }

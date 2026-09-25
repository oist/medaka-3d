"""Native-frame raw-camera and scientific-plot review of unaccepted node proposals.

No algorithm or artifact from another lab is imported. Callers own provenance and
provide the saved baseline/candidate arrays. This module never changes tracks or
interprets a proposal's geometric support as accepted biological identity.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from medaka3d.types import VIEWS

SOURCE_TOTAL_FRAMES = 348600
SOURCE_SHAPE = (1200, 1200, 3)
COLOURS = ((83, 190, 255), (234, 216, 83), (186, 132, 247))
METHODS = ("baseline", "candidate")


def _validate(arrays, rig=None, raw_paths=None):
    frames = np.asarray(arrays["frames"])
    if frames.ndim != 1 or not len(frames) or frames.dtype.kind not in "iu":
        raise ValueError("frames must be a nonempty integer vector")
    if np.any(frames[1:] != frames[:-1] + 1):
        raise ValueError("evaluation frames must be consecutive native source frames")
    if frames[0] < 0 or frames[-1] >= SOURCE_TOTAL_FRAMES:
        raise ValueError("source frame is outside the registered recording")
    shapes = {
        "baseline_pose": (len(frames), 3, 3, 3),
        "candidate_pose": (len(frames), 3, 3, 3),
        "original_uv": (len(frames), 3, 3, 3, 2),
        "candidate_uv": (len(frames), 3, 3, 3, 2),
        "original_source": (len(frames), 3, 3, 3),
        "candidate_source": (len(frames), 3, 3, 3),
        "tested": (len(frames),),
        "changed": (len(frames),),
        "supported": (len(frames),),
    }
    for key, shape in shapes.items():
        if np.asarray(arrays[key]).shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
    for key in ("baseline_pose", "candidate_pose", "original_uv", "candidate_uv"):
        if np.isinf(arrays[key]).any():
            raise ValueError(f"{key} may contain missing NaNs, but not infinities")
    if rig is not None and tuple(rig.views) != VIEWS:
        raise ValueError("rig must use the registered top, side1, side2 order")
    if raw_paths is not None and set(raw_paths) != set(VIEWS):
        raise ValueError("raw_paths must explicitly identify all three cameras")
    return frames.astype(np.int64)


def _focus(focus_slots):
    slots = sorted(set(focus_slots))
    if not slots or any(
        type(s) not in (int, np.int64, np.int32) or s not in range(3) for s in slots
    ):
        raise ValueError("focus_slots must be a nonempty subset of 0, 1, 2")
    return slots


def _indices(frames, selected_frames):
    selected = np.asarray(selected_frames)
    if selected.ndim != 1 or not len(selected) or selected.dtype.kind not in "iu":
        raise ValueError("selected_frames must be a nonempty integer vector")
    if np.any(selected[1:] <= selected[:-1]):
        raise ValueError("selected_frames must be strictly increasing")
    indices = np.searchsorted(frames, selected)
    if np.any(indices >= len(frames)) or not np.array_equal(frames[indices], selected):
        raise ValueError("selected frame is absent from evaluation arrays")
    return indices


def _new_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"review output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _captures(raw_paths, first, source_fps):
    captures, metadata = {}, {}
    try:
        for view in VIEWS:
            path = Path(raw_paths[view]).resolve(strict=True)
            capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
            captures[view] = capture
            if not capture.isOpened():
                raise RuntimeError(f"cannot open source video: {view}")
            shape = (
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                3,
            )
            count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv2.CAP_PROP_FPS)
            if shape != SOURCE_SHAPE or count != SOURCE_TOTAL_FRAMES:
                raise ValueError(f"unregistered source dimensions/frame count: {view}")
            if not np.isclose(fps, source_fps, rtol=0, atol=1e-6):
                raise ValueError(f"source FPS mismatch: {view}")
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(first)):
                raise RuntimeError(f"source seek failed: {view}")
            if capture.get(cv2.CAP_PROP_POS_FRAMES) != first:
                raise RuntimeError(f"source seek frame mismatch: {view}")
            metadata[view] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "source_frames": int(count),
                "source_fps": fps,
                "shape": list(shape),
            }
        yield captures, metadata
    finally:
        for capture in captures.values():
            capture.release()


def _read_native(captures, frame):
    images = {}
    for view, capture in captures.items():
        ok, image = capture.read()
        if not ok or image is None or capture.get(cv2.CAP_PROP_POS_FRAMES) != frame + 1:
            raise RuntimeError(f"native frame alignment failed: {view}/{frame}")
        if image.shape != SOURCE_SHAPE or image.dtype != np.uint8:
            raise ValueError(f"native image shape/dtype mismatch: {view}/{frame}")
        images[view] = image
    return images


def _project(rig, arrays):
    return {
        method: np.stack(
            [rig.camera(view).project(arrays[f"{method}_pose"]) for view in VIEWS], axis=2
        )
        for method in METHODS
    }


def _changed_nodes(arrays):
    original, candidate = arrays["original_uv"], arrays["candidate_uv"]
    original_ok, candidate_ok = (
        np.isfinite(original).all(axis=-1),
        np.isfinite(candidate).all(axis=-1),
    )
    changed = original_ok != candidate_ok
    changed |= original_ok & candidate_ok & (np.linalg.norm(original - candidate, axis=-1) > 1e-6)
    changed |= np.asarray(arrays["original_source"]) != np.asarray(arrays["candidate_source"])
    return changed


def _crops(arrays, projected, indices, slots, *, min_size):
    crops = {}
    for view_index, view in enumerate(VIEWS):
        points = np.concatenate(
            [
                np.asarray(values)[indices][:, slots, view_index].reshape(-1, 2)
                for values in (
                    arrays["original_uv"],
                    arrays["candidate_uv"],
                    projected["baseline"],
                    projected["candidate"],
                )
            ]
        )
        points = points[np.isfinite(points).all(axis=1)]
        # Out-of-image projections remain visible as diagnostic metadata, but
        # cannot expand a crop beyond the actual source image.
        points = points[(points >= 0).all(axis=1) & (points < 1200).all(axis=1)]
        if len(points):
            low, high = points.min(axis=0) - 45, points.max(axis=0) + 45
            extent = np.maximum(high - low, min_size)
            center = (high + low) / 2
            low = np.clip(np.floor(center - extent / 2), 0, 1200 - np.minimum(extent, 1200))
            high = np.minimum(1200, np.ceil(low + extent))
            low, high = low.astype(int), high.astype(int)
        else:
            low, high = np.array([0, 0]), np.array([1200, 1200])
        crops[view] = [int(low[0]), int(low[1]), int(high[0]), int(high[1])]
    return crops


def _text(image, text, xy, *, scale=0.45, color=(240, 240, 240), thickness=1):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _letterbox(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    resized = cv2.resize(
        image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    offset = np.array([(width - size[0]) // 2, (height - size[1]) // 2])
    canvas[offset[1] : offset[1] + size[1], offset[0] : offset[0] + size[0]] = resized
    return canvas, scale, offset


def _point(point):
    return tuple(np.rint(np.clip(point, -100000, 100000)).astype(int))


def _overlay(image, measured, projected, changed, slots, *, scale, offset, candidate):
    measured = measured * scale + offset
    projected = projected * scale + offset
    for slot in slots:
        color = COLOURS[slot]
        for a, b in ((0, 1), (1, 2)):
            if np.isfinite(projected[slot, [a, b]]).all():
                cv2.line(
                    image,
                    _point(projected[slot, a]),
                    _point(projected[slot, b]),
                    color,
                    1,
                    cv2.LINE_AA,
                )
        for node in range(3):
            observed, fitted = measured[slot, node], projected[slot, node]
            if np.isfinite(observed).all():
                cv2.circle(image, _point(observed), 5, color, 1, cv2.LINE_AA)
                if candidate and changed[slot, node]:
                    x, y = _point(observed)
                    cv2.rectangle(image, (x - 8, y - 8), (x + 8, y + 8), (250, 250, 250), 1)
            if np.isfinite(fitted).all():
                cv2.drawMarker(
                    image, _point(fitted), color, cv2.MARKER_TILTED_CROSS, 7, 1, cv2.LINE_AA
                )
            if np.isfinite([observed, fitted]).all() and np.linalg.norm(observed - fitted) > 4:
                cv2.line(image, _point(observed), _point(fitted), (165, 165, 165), 1, cv2.LINE_AA)
        finite = measured[slot][np.isfinite(measured[slot]).all(axis=1)]
        if len(finite):
            x, y = _point(finite[0])
            _text(image, f"S{slot}", (x + 7, y - 9), scale=0.42, color=color)


def _panel(raw, crop, arrays, projected, changes, index, view_index, method, slots, size):
    width, height = size
    x0, y0, x1, y1 = crop
    header, footer = 28, 22
    panel = np.full((height, width, 3), 18, dtype=np.uint8)
    content, scale, offset = _letterbox(raw[y0:y1, x0:x1], width, height - header - footer)
    uv_key = "original_uv" if method == "baseline" else "candidate_uv"
    _overlay(
        content,
        np.asarray(arrays[uv_key])[index, :, view_index] - (x0, y0),
        projected[method][index, :, view_index] - (x0, y0),
        changes[index, :, view_index],
        slots,
        scale=scale,
        offset=offset,
        candidate=method == "candidate",
    )
    panel[header : height - footer] = content
    title = "BASELINE" if method == "baseline" else "PROPOSAL (UNACCEPTED)"
    _text(panel, f"{VIEWS[view_index]} | {title}", (6, 19), scale=0.4)
    frame = int(arrays["frames"][index])
    suffix = "tested" if arrays["tested"][index] else "context"
    _text(panel, f"f{frame}  {frame / 166:.4f}s  {suffix}", (6, height - 7), scale=0.4)
    return panel


def _save_png(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write {path}")
    reread = cv2.imread(str(path))
    if reread is None or not np.array_equal(reread, image):
        raise RuntimeError("PNG lossless readback failed")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_comparison_sheet(
    raw_paths, rig, arrays, selected_frames, out_path, focus_slots=(0, 1, 2)
):
    """Up to five columns and baseline/proposal row pairs for each raw camera.

    Rings are selected detector nodes; x marks and segments are projected 3D.
    White boxes mark changed node selections, never accepted identity changes.
    Fixed crops are shared across every native frame and both methods per view.
    """
    frames = _validate(arrays, rig, raw_paths)
    indices, slots = _indices(frames, selected_frames), _focus(focus_slots)
    if len(indices) > 5:
        raise ValueError("comparison sheets accept at most five frames")
    out_path = _new_output(out_path)
    projected, changes = _project(rig, arrays), _changed_nodes(arrays)
    crops = _crops(arrays, projected, indices, slots, min_size=220)
    width, row_height, banner = 420, 365, 72
    sheet = np.full((6 * row_height + banner, width * len(indices), 3), 18, dtype=np.uint8)
    _text(sheet, "Node correspondence review | proposal is unaccepted", (10, 23), scale=0.6)
    _text(
        sheet,
        "Ring: selected 2D node   X + line: 3D reprojection   White box: changed selection",
        (10, 46),
        scale=0.48,
    )
    _text(
        sheet,
        "Node order along line: mouth - pectoral - tail | source pixels unchanged",
        (10, 64),
        scale=0.42,
    )
    selected = {int(frames[index]): (column, int(index)) for column, index in enumerate(indices)}
    hashes = {}
    first, last = int(frames[indices[0]]), int(frames[indices[-1]])
    with _captures(raw_paths, first, 166) as (captures, source):
        for frame in range(first, last + 1):
            images = _read_native(captures, frame)
            if frame not in selected:
                continue
            column, index = selected[frame]
            hashes[frame] = {
                view: hashlib.sha256(image.tobytes()).hexdigest() for view, image in images.items()
            }
            for view_index, view in enumerate(VIEWS):
                for method_index, method in enumerate(METHODS):
                    panel = _panel(
                        images[view],
                        crops[view],
                        arrays,
                        projected,
                        changes,
                        index,
                        view_index,
                        method,
                        slots,
                        (width, row_height),
                    )
                    y = banner + (2 * view_index + method_index) * row_height
                    x = column * width
                    sheet[y : y + row_height, x : x + width] = panel
    digest = _save_png(out_path, sheet)
    return {
        "output": str(out_path.resolve()),
        "sha256": digest,
        "shape": list(sheet.shape),
        "source": source,
        "native_frames": frames[indices].tolist(),
        "focus_slots": slots,
        "source_bgr_sha256": hashes,
        "crops_xyxy_source_pixels": crops,
        "all_native_frames_read_per_view": last - first + 1,
        "png_readback_verified": True,
        "rows": "baseline then unaccepted proposal for each of top, side1, side2",
        "identity_corrections_accepted": False,
    }


def render_comparison_video(
    raw_paths,
    rig,
    arrays,
    first_frame,
    stop_frame,
    out_path,
    focus_slots=(0, 1, 2),
    output_fps=30,
    source_fps=166,
):
    """Three view rows, baseline/proposal columns; all native samples retained.

    Range is [first_frame, stop_frame). Native 166 fps samples replay at 30 fps
    by default. Each view uses one fixed source crop of at least 400 px per axis.
    """
    frames = _validate(arrays, rig, raw_paths)
    if int(first_frame) != first_frame or int(stop_frame) != stop_frame:
        raise ValueError("video boundaries must be native integer frames")
    indices = _indices(frames, np.arange(first_frame, stop_frame, dtype=np.int64))
    slots = _focus(focus_slots)
    if not np.isfinite(output_fps) or not 0 < output_fps <= source_fps:
        raise ValueError("require finite 0 < output_fps <= source_fps")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg must already be installed")
    out_path = _new_output(out_path)
    log_path = _new_output(out_path.with_suffix(".ffmpeg.log"))
    projected, changes = _project(rig, arrays), _changed_nodes(arrays)
    crops = _crops(arrays, projected, indices, slots, min_size=400)
    width, row_height, banner = 640, 460, 60
    canvas_size = (2 * width, 3 * row_height + banner)
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
        f"{canvas_size[0]}x{canvas_size[1]}",
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
        _captures(raw_paths, int(first_frame), source_fps) as (captures, source),
        log_path.open("xb") as log,
    ):
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for index in indices:
                frame = int(frames[index])
                images = _read_native(captures, frame)
                canvas = np.full((canvas_size[1], canvas_size[0], 3), 18, dtype=np.uint8)
                _text(
                    canvas,
                    "BASELINE / UNACCEPTED NODE PROPOSAL | every native frame retained",
                    (12, 23),
                    scale=0.58,
                )
                _text(
                    canvas,
                    "Rings: selected 2D  |  X + lines: projected 3D  |  White boxes: changed nodes",
                    (12, 47),
                    scale=0.48,
                )
                for view_index, view in enumerate(VIEWS):
                    y = banner + view_index * row_height
                    for method_index, method in enumerate(METHODS):
                        panel = _panel(
                            images[view],
                            crops[view],
                            arrays,
                            projected,
                            changes,
                            index,
                            view_index,
                            method,
                            slots,
                            (width, row_height),
                        )
                        x = method_index * width
                        canvas[y : y + row_height, x : x + width] = panel
                process.stdin.write(canvas.tobytes())
            process.stdin.close()
            if process.wait(timeout=120) != 0:
                raise RuntimeError(f"ffmpeg failed; see {log_path}")
        except BaseException:
            process.kill()
            process.wait()
            raise
    decoded = 0
    capture = cv2.VideoCapture(str(out_path))
    try:
        if not capture.isOpened() or not np.isclose(capture.get(cv2.CAP_PROP_FPS), output_fps):
            raise RuntimeError("comparison movie open/FPS verification failed")
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image is None or image.shape != (canvas_size[1], canvas_size[0], 3):
                raise RuntimeError("comparison movie dimensions are wrong")
            decoded += 1
    finally:
        capture.release()
    if decoded != len(indices):
        raise RuntimeError(f"decoded {decoded} movie frames, expected {len(indices)}")
    return {
        "output": str(out_path.resolve()),
        "source": source,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "native_frame_range_half_open": [int(first_frame), int(stop_frame)],
        "native_frames": frames[indices].tolist(),
        "output_frames": decoded,
        "source_fps": float(source_fps),
        "output_fps": float(output_fps),
        "native_duration_seconds": len(indices) / source_fps,
        "output_duration_seconds": decoded / output_fps,
        "crops_xyxy_source_pixels": crops,
        "focus_slots": slots,
        "all_native_frames_read_per_view": len(indices),
        "movie_readback_verified": True,
        "coordinates_interpolated": False,
        "identity_corrections_accepted": False,
        "ffmpeg_command": command,
    }


def _finite_max(values):
    if not len(values):
        return np.empty(0, dtype=float)
    values = np.asarray(values, dtype=float).reshape(len(values), -1)
    result = np.where(np.isfinite(values), values, -np.inf).max(axis=1)
    result[~np.isfinite(result)] = np.nan
    return result


def plot_case(arrays, event_frame, out_path, core=None, focus_slots=(0, 1, 2)):
    """Export an inspectable eight-panel PNG of native diagnostic measurements.

    ``core`` optionally supplies [first, stop) for the plot; otherwise all context
    is shown. Lengths are plotted per slot; error maxima are across focused slots
    and, where applicable, views/nodes. Supplied body median +/- scale is shown as
    a descriptive reference band, never a confidence interval.
    No improvement, identity accuracy, or statistical uncertainty is inferred.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames, slots = _validate(arrays), _focus(focus_slots)
    indices = (
        np.arange(len(frames))
        if core is None
        else _indices(frames, np.arange(int(core[0]), int(core[1]), dtype=np.int64))
    )
    x = frames[indices]
    out_path = _new_output(out_path)
    measures = {}
    for method in METHODS:
        method_traces = {}
        for short, key in (
            ("reprojection_px", f"{method}_reprojection_px"),
            ("loo_px", f"{method}_loo_px"),
            ("ray_mm", f"{method}_ray_mm"),
        ):
            method_traces[short] = _finite_max(np.asarray(arrays[key])[:, slots])[indices]
        pose = np.asarray(arrays[f"{method}_pose"])
        jumps = np.linalg.norm(np.diff(pose[:, slots, 1], axis=0), axis=-1)
        method_traces["pectoral_step_mm"] = np.r_[np.nan, _finite_max(jumps)][indices]
        lengths = np.asarray(arrays[f"{method}_lengths"])
        method_traces["mouth_pectoral_mm"] = lengths[:, slots, 0][indices]
        method_traces["pectoral_tail_mm"] = lengths[:, slots, 1][indices]
        score_key = "score_original" if method == "baseline" else "score_candidate"
        method_traces["score"] = np.asarray(arrays[score_key], dtype=float)[indices]
        uv_key = "original_uv" if method == "baseline" else "candidate_uv"
        method_traces["selected_node_count"] = (
            np.isfinite(np.asarray(arrays[uv_key])[:, slots])
            .all(axis=-1)
            .sum(axis=(1, 2, 3))[indices]
        )
        measures[method] = method_traces
    panels = [
        ("reprojection_px", "Maximum reprojection discrepancy", "px"),
        ("loo_px", "Maximum leave-one-view-out discrepancy", "px"),
        ("ray_mm", "Maximum point-to-ray discrepancy", "mm"),
        ("pectoral_step_mm", "Maximum pectoral displacement / native step", "mm / frame"),
        ("mouth_pectoral_mm", "Mouth-to-pectoral segment by slot", "mm"),
        ("pectoral_tail_mm", "Pectoral-to-tail segment by slot", "mm"),
        ("score", "Saved objective score", "score (provided)"),
        ("selected_node_count", "Available selected 2D nodes", f"count / {len(slots) * 9}"),
    ]
    reference = None
    if "body_reference_median" in arrays and "body_reference_scale" in arrays:
        median = np.asarray(arrays["body_reference_median"], dtype=float)
        scale = np.asarray(arrays["body_reference_scale"], dtype=float)
        if median.shape not in ((2,), (3, 2)) or scale.shape != median.shape:
            raise ValueError("body reference median and scale must both have shape (2,) or (3,2)")
        if not np.isfinite([median, scale]).all() or np.any(scale < 0):
            raise ValueError("body reference median/scale must be finite with nonnegative scale")
        reference = {
            "median": median.tolist(),
            "scale": scale.tolist(),
            "meaning": "supplied descriptive median +/- scale; not a confidence interval",
        }
    # Keep the source overlay hues while darkening them for contrast on white.
    slot_colours = {0: "#ac771b", 1: "#178796", 2: "#9c58a3"}
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#252525",
            "text.color": "#252525",
            "axes.edgecolor": "#888888",
            "figure.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(4, 2, figsize=(14, 13), sharex=True)
        tested = np.asarray(arrays["tested"], dtype=bool)[indices]
        edges = np.diff(np.r_[False, tested, False].astype(int))
        spans = [
            (x[a] - 0.5, x[b - 1] + 0.5)
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
        ]
        for axis, (key, title, unit) in zip(axes.flat, panels, strict=True):
            for low, high in spans:
                axis.axvspan(low, high, color="#eeeeee", zorder=0)
            if key in ("mouth_pectoral_mm", "pectoral_tail_mm"):
                segment = 0 if key == "mouth_pectoral_mm" else 1
                if reference is not None and median.shape == (2,):
                    axis.axhspan(
                        max(0, median[segment] - scale[segment]),
                        median[segment] + scale[segment],
                        color="#d9d1ad",
                        alpha=0.5,
                    )
                    axis.axhline(median[segment], color="#88846b", lw=0.7, ls=":")
                for slot_index, slot in enumerate(slots):
                    color = slot_colours[slot]
                    if reference is not None and median.shape == (3, 2):
                        axis.axhspan(
                            max(0, median[slot, segment] - scale[slot, segment]),
                            median[slot, segment] + scale[slot, segment],
                            color=color,
                            alpha=0.06,
                        )
                    axis.plot(
                        x,
                        measures["baseline"][key][:, slot_index],
                        color=color,
                        ls="--",
                        lw=1.6,
                        zorder=3,
                    )
                    axis.plot(
                        x,
                        measures["candidate"][key][:, slot_index],
                        color=color,
                        lw=1,
                        label=f"S{slot}",
                        zorder=2,
                    )
                axis.legend(
                    loc="lower right",
                    bbox_to_anchor=(1, 1.005),
                    ncol=len(slots),
                    frameon=False,
                    fontsize=9,
                    borderaxespad=0,
                )
            else:
                axis.plot(
                    x,
                    measures["baseline"][key],
                    color="#17638b",
                    ls="--",
                    lw=1.35,
                    label="Baseline",
                    zorder=3,
                )
                axis.plot(
                    x,
                    measures["candidate"][key],
                    color="#c56a20",
                    lw=1.15,
                    label="Proposal (unaccepted)",
                    zorder=2,
                )
            if not any(np.isfinite(measures[method][key]).any() for method in METHODS):
                axis.text(
                    0.5,
                    0.5,
                    "No measured values",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="#666666",
                )
            axis.axvline(event_frame, color="#555555", ls=":", lw=1)
            axis.set_title(title, loc="left", fontsize=11, pad=9)
            axis.set_ylabel(unit)
            axis.grid(axis="y", color="#e2e2e2", linewidth=0.6)
            axis.set_xlim(x[0] - 0.5, x[-1] + 0.5)
            if key != "score":
                axis.set_ylim(bottom=0)
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        ticks = np.unique(np.rint(np.linspace(x[0], x[-1], min(6, len(x)))).astype(int))
        axes[-1, 0].set_xticks(ticks)
        axes[-1, 1].set_xticks(ticks)
        axes[-1, 0].set_xlabel("Native source frame (166 fps)")
        axes[-1, 1].set_xlabel("Native source frame (166 fps)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper right", bbox_to_anchor=(0.975, 0.995), ncol=2, frameon=False
        )
        fig.suptitle(
            f"Multiview node review | event frame {event_frame}",
            x=0.08,
            y=0.987,
            ha="left",
            fontsize=15,
        )
        fig.text(
            0.08,
            0.954,
            f"Slots {slots}; body lengths per slot; other errors show maxima. "
            "Grey bands: tested frames.",
            fontsize=10,
        )
        fig.text(
            0.08,
            0.035,
            "Body panels: dashed baseline / solid proposal. "
            + (
                "Body band: reference median +/- supplied scale (not a confidence interval)."
                if reference is not None
                else "No body reference band was supplied."
            ),
            fontsize=9,
        )
        fig.text(
            0.08,
            0.018,
            "Missing measurements remain missing. Lower geometric error does not "
            "establish biological identity. No proposal is accepted in this review.",
            fontsize=9,
        )
        fig.subplots_adjust(
            left=0.08, right=0.975, bottom=0.096, top=0.914, hspace=0.42, wspace=0.2
        )
        fig.savefig(out_path, dpi=140, facecolor="white")
        plt.close(fig)
    image = cv2.imread(str(out_path))
    if image is None or image.shape[:2] != (1820, 1960):
        raise RuntimeError("scientific PNG render dimensions/readback failed")
    series = {
        method: {
            key: np.where(np.isfinite(values), values, None).tolist()
            for key, values in traces.items()
        }
        for method, traces in measures.items()
    }
    return {
        "output": str(out_path.resolve()),
        "shape": list(image.shape),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "native_frames": x.tolist(),
        "event_frame": int(event_frame),
        "focus_slots": slots,
        "aggregation": "body lengths per slot; other errors max of finite values; count explicit",
        "body_reference": reference,
        "series": series,
        "identity_corrections_accepted": False,
        "png_decoded": True,
        "visual_inspection_required": True,
    }

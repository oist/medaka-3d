"""Render native-pixel three-camera comparisons of common-downstream A/B/C arms.

No other lab is imported. The saved arm coordinates and their one-time slot
alignment are used verbatim; rendering never relinks, smooths or fills a gap.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from medaka3d.aquarium import movie_indices, sha256
from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

CONDITIONS = ("A", "B", "C")
TITLES = (
    "A | saved SLEAP labels + local 3D",
    "B | bidirectional connection cuts",
    "C | custom 2D tracker",
)
COLOURS = ((83, 190, 255), (234, 216, 83), (186, 132, 247))
SOURCE_FPS, OUTPUT_FPS = 166.0, 30.0
SOURCE_TOTAL, SOURCE_SHAPE = 348600, (1200, 1200, 3)
WIDTH, ROW_HEIGHT, HEADER, FOOTER = 640, 430, 110, 64
CANVAS_SHAPE = (HEADER + 3 * ROW_HEIGHT + FOOTER, 3 * WIDTH, 3)
CALIBRATION = (
    "calibration/fixed_refractive_192x192x186/"
    "phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration"
)
REVIEWS = (
    ("cg20s", 76791),
    ("cg20s", 76988),
    ("cg20s", 76235),
    ("cg20s", 76930),
    ("known_30735", 30735),
    ("known_31479", 31479),
    ("known_323819", 323819),
    ("cg20s", 77415),
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def selected_indices(frames, selected):
    selected = np.asarray(selected)
    if selected.ndim != 1 or not len(selected) or selected.dtype.kind not in "iu":
        raise ValueError("Selected native frames must be a nonempty integer vector")
    if np.any(np.diff(selected) <= 0):
        raise ValueError("Selected native frames must strictly increase")
    index = np.searchsorted(frames, selected)
    if np.any(index >= len(frames)) or not np.array_equal(frames[index], selected):
        raise ValueError("A requested native frame is absent from a saved arm")
    return index


def load_arms(directory, stage):
    arms, hashes, frames = {}, {}, None
    for condition in CONDITIONS:
        path = directory / f"{condition}.npz"
        with np.load(path, allow_pickle=False) as saved:
            native = saved["frames"]
            if native.ndim != 1 or not len(native) or native.dtype.kind not in "iu":
                raise ValueError("Saved native frames must be an integer vector")
            if np.any(np.diff(native) != 1) or native[0] < 0 or native[-1] >= SOURCE_TOTAL:
                raise ValueError("Saved frames must be consecutive within the source recording")
            if frames is not None and not np.array_equal(frames, native):
                raise ValueError("A/B/C native frame ranges differ")
            frames = native.copy()
            arrays = {key: saved[f"{stage}__{key}"].copy() for key in ("pose", "uv", "source")}
        shapes = {
            "pose": (len(frames), 3, 3, 3),
            "uv": (len(frames), 3, 3, 3, 2),
            "source": (len(frames), 3, 3),
        }
        for key, expected in shapes.items():
            if arrays[key].shape != expected:
                raise ValueError(f"{path.name}/{key}: expected {expected}")
        if any(np.isinf(arrays[key]).any() for key in ("pose", "uv")):
            raise ValueError("Coordinates may be missing NaNs, but not infinities")
        arms[condition], hashes[condition] = arrays, sha256(path)
    return frames, arms, hashes


def project_arms(rig, arms):
    projected = {}
    for condition, arrays in arms.items():
        pose = arrays["pose"]
        finite = np.isfinite(pose).all(axis=-1)
        values = np.full((len(pose), 3, 3, 3, 2), np.nan)
        for column, view in enumerate(VIEWS):
            here = values[:, :, column]
            if finite.any():
                here[finite] = rig.camera(view).project(pose[finite])
        projected[condition] = values
    return projected


def common_crops(arms, projected, indices):
    """One time-fixed crop per view, shared by all columns and all three slots."""
    crops = {}
    for column, view in enumerate(VIEWS):
        points = np.concatenate(
            [
                values[indices, :, column].reshape(-1, 2)
                for condition in CONDITIONS
                for values in (arms[condition]["uv"], projected[condition])
            ]
        )
        points = points[np.isfinite(points).all(axis=-1)]
        points = points[(points >= 0).all(axis=-1) & (points < 1200).all(axis=-1)]
        if len(points):
            low, high = points.min(0) - 50, points.max(0) + 50
            extent = np.minimum(1200, np.maximum(high - low, [500, 400]))
            low = np.floor(np.clip((low + high - extent) / 2, 0, 1200 - extent)).astype(int)
            high = np.ceil(low + extent).astype(int)
            crops[view] = [int(low[0]), int(low[1]), int(high[0]), int(high[1])]
        else:
            crops[view] = [0, 0, 1200, 1200]
    return crops


def text(image, value, xy, scale=0.48, colour=(235, 235, 235), thickness=1):
    cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def letterbox(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    resized = cv2.resize(
        image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )
    offset = np.array([(width - size[0]) // 2, (height - size[1]) // 2])
    canvas = np.full((height, width, 3), 20, np.uint8)
    canvas[offset[1] : offset[1] + size[1], offset[0] : offset[0] + size[0]] = resized
    return canvas, scale, offset


def point(value):
    return tuple(np.rint(np.clip(value, -100000, 100000)).astype(int))


def overlay(image, measured, projected, scale, offset):
    """Lines/rings use saved 2D points; crosses use independently projected 3D."""
    measured, projected = measured * scale + offset, projected * scale + offset
    for slot, colour in enumerate(COLOURS):
        for first, second in ((0, 1), (1, 2)):
            if np.isfinite(measured[slot, [first, second]]).all():
                cv2.line(
                    image,
                    point(measured[slot, first]),
                    point(measured[slot, second]),
                    colour,
                    1,
                    cv2.LINE_AA,
                )
        for node in range(3):
            if np.isfinite(measured[slot, node]).all():
                cv2.circle(image, point(measured[slot, node]), 5, colour, 1, cv2.LINE_AA)
            if np.isfinite(projected[slot, node]).all():
                cv2.drawMarker(
                    image,
                    point(projected[slot, node]),
                    colour,
                    cv2.MARKER_TILTED_CROSS,
                    8,
                    1,
                    cv2.LINE_AA,
                )
        valid = measured[slot][np.isfinite(measured[slot]).all(-1)]
        if len(valid):
            x, y = point(valid[0])
            text(image, f"S{slot}", (x + 8, y - 8), colour=colour, scale=0.48, thickness=1)


def render_frame(images, arms, projected, index, native_frame, crops, case, stage, mode):
    canvas = np.full(CANVAS_SHAPE, 20, np.uint8)
    text(
        canvas,
        f"{case} | frame {native_frame} | source {native_frame / SOURCE_FPS:.4f} s",
        (14, 28),
        scale=0.66,
        thickness=2,
    )
    text(canvas, f"Stage: {stage} | {mode}", (14, 54), scale=0.52)
    text(
        canvas,
        "Rings + M-P-T lines: saved 2D detections   |   X: 3D reprojection",
        (14, 79),
        scale=0.53,
    )
    for column, title in enumerate(TITLES):
        text(canvas, title, (column * WIDTH + 14, 104), scale=0.49, thickness=1)
    for view_index, view in enumerate(VIEWS):
        x0, y0, x1, y1 = crops[view]
        raw = images[view][y0:y1, x0:x1]
        base, scale, offset = letterbox(raw, WIDTH - 8, ROW_HEIGHT - 66)
        offset = offset - np.array([x0, y0]) * scale
        for column, condition in enumerate(CONDITIONS):
            panel = np.full((ROW_HEIGHT, WIDTH, 3), 20, np.uint8)
            text(panel, view, (12, 23), scale=0.52)
            content = base.copy()
            measured = arms[condition]["uv"][index, :, view_index]
            fitted = projected[condition][index, :, view_index]
            overlay(content, measured, fitted, scale, offset)
            panel[30 : ROW_HEIGHT - 36, 4 : WIDTH - 4] = content
            for slot, colour in enumerate(COLOURS):
                n2d = int(np.isfinite(measured[slot]).all(-1).sum())
                n3d = int(np.isfinite(fitted[slot]).all(-1).sum())
                text(
                    panel,
                    f"S{slot} 2D:{n2d}/3 X:{n3d}/3",
                    (10 + slot * 210, ROW_HEIGHT - 13),
                    scale=0.40,
                    colour=colour,
                )
            y, x = HEADER + view_index * ROW_HEIGHT, column * WIDTH
            canvas[y : y + ROW_HEIGHT, x : x + WIDTH] = panel
    y = HEADER + 3 * ROW_HEIGHT
    text(
        canvas,
        "All three slots shown. A is a local rerun, not the saved full-series C34.",
        (14, y + 23),
        scale=0.51,
    )
    text(
        canvas,
        "Missing 2D rings stay absent; 3D crosses may use other views. No time interpolation.",
        (14, y + 48),
        scale=0.48,
    )
    return canvas


@contextmanager
def source_captures(paths, first):
    captures = {}
    try:
        for view in VIEWS:
            capture = cv2.VideoCapture(
                str(paths[view]), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2]
            )
            captures[view] = capture
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open raw video: {view}")
            shape = (
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                3,
            )
            if shape != SOURCE_SHAPE or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != SOURCE_TOTAL:
                raise ValueError(f"Unexpected source recording shape or length: {view}")
            if not np.isclose(capture.get(cv2.CAP_PROP_FPS), SOURCE_FPS, atol=1e-6, rtol=0):
                raise ValueError(f"Unexpected source frame rate: {view}")
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(first)):
                raise RuntimeError(f"Native frame seek failed: {view}")
            if capture.get(cv2.CAP_PROP_POS_FRAMES) != first:
                raise RuntimeError(f"Native frame seek mismatch: {view}")
        yield captures
    finally:
        for capture in captures.values():
            capture.release()


def read_selected(captures, current, target):
    for frame in range(current, target):
        for view, capture in captures.items():
            if not capture.grab() or capture.get(cv2.CAP_PROP_POS_FRAMES) != frame + 1:
                raise RuntimeError(f"Native frame skip mismatch: {view}/{frame}")
    images = {}
    for view, capture in captures.items():
        ok, image = capture.read()
        if not ok or image is None or capture.get(cv2.CAP_PROP_POS_FRAMES) != target + 1:
            raise RuntimeError(f"Native frame read mismatch: {view}/{target}")
        if image.shape != SOURCE_SHAPE or image.dtype != np.uint8:
            raise ValueError("Unexpected decoded source image")
        images[view] = image
    return images


def full_decode(path, expected):
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    count = 0
    try:
        if not capture.isOpened() or not np.isclose(capture.get(cv2.CAP_PROP_FPS), OUTPUT_FPS):
            raise RuntimeError("Encoded movie open or FPS check failed")
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image is None or image.shape != CANVAS_SHAPE:
                raise RuntimeError("Encoded movie shape mismatch")
            count += 1
    finally:
        capture.release()
    if count != expected:
        raise RuntimeError(f"Full movie decode found {count} frames, expected {expected}")
    return {"verified": True, "decoded_frames": count, "shape": list(CANVAS_SHAPE)}


def render_video(paths, frames, arms, projected, selected, output, *, case, center, stage, mode):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg must already be available")
    if output.exists() or output.with_suffix(".ffmpeg.log").exists():
        raise FileExistsError(f"Preserve existing render: {output}")
    index = selected_indices(frames, selected)
    crops = common_crops(arms, projected, index)
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
        f"{CANVAS_SHAPE[1]}x{CANVAS_SHAPE[0]}",
        "-r",
        str(OUTPUT_FPS),
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
        str(output),
    ]
    still_path = output.with_suffix(".png") if center is not None else None
    if still_path is not None and still_path.exists():
        raise FileExistsError(f"Preserve existing still: {still_path}")
    with (
        source_captures(paths, int(selected[0])) as captures,
        output.with_suffix(".ffmpeg.log").open("xb") as log,
    ):
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        current = int(selected[0])
        try:
            for i, frame in zip(index, selected, strict=True):
                images = read_selected(captures, current, int(frame))
                current = int(frame) + 1
                canvas = render_frame(
                    images, arms, projected, int(i), int(frame), crops, case, stage, mode
                )
                process.stdin.write(canvas.tobytes())
                if (
                    still_path is not None
                    and frame == center
                    and not cv2.imwrite(str(still_path), canvas)
                ):
                    raise RuntimeError("Representative PNG write failed")
            process.stdin.close()
            if process.wait(timeout=60) != 0:
                raise RuntimeError(f"ffmpeg failed; inspect {output.with_suffix('.ffmpeg.log')}")
        except BaseException:
            process.kill()
            process.wait(timeout=10)
            raise
    decoded = full_decode(output, len(selected))
    if still_path is not None:
        image = cv2.imread(str(still_path))
        if image is None or image.shape != CANVAS_SHAPE:
            raise RuntimeError("Representative PNG readback failed")
    return {
        "crops_xyxy_source_pixels": crops,
        "full_decode": decoded,
        "native_sample_frames": selected.tolist(),
        "ffmpeg_command": command,
    }, still_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("work/custom2d_comparison_v1"))
    parser.add_argument(
        "--analysis", type=Path, help="Completed analysis directory; default ROOT/analysis"
    )
    parser.add_argument("--output", type=Path, help="Review directory; default ROOT/review")
    parser.add_argument("--source-cache", type=Path, default=Path("work/three_body/source_cache"))
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument("--centers", type=int, nargs="*", default=[c for _, c in REVIEWS])
    parser.add_argument("--no-overview", action="store_true")
    parser.add_argument("--include-c28-31479", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    analysis = args.analysis or args.root / "analysis"
    output = args.output or args.root / "review"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {
            "schema_version": 1,
            "entries": [],
            "conditions": dict(zip(CONDITIONS, TITLES, strict=True)),
            "legend": "Rings and M-P-T lines: saved 2D points; X: 3D reprojection",
            "accepted_identity_corrections": 0,
        }
    )
    if manifest.get("analysis_directory", str(analysis.resolve())) != str(analysis.resolve()):
        raise ValueError("Use a different review output directory for a different analysis version")
    manifest["analysis_directory"] = str(analysis.resolve())
    for entry in manifest["entries"]:
        path = output / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Previously verified review artifact is missing or changed: {path}")
    cache_path = args.source_cache / "source_cache_manifest.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache["status"] != "verified" or cache["experiment"] != "20251225_K_1_2_FCMC":
        raise ValueError("Wrong or unverified raw source cache")
    known_centers = {center for _, center in REVIEWS}
    if not set(args.centers).issubset(known_centers):
        raise ValueError("Review center must belong to the eight fixed centers")
    jobs = [
        (case, center, center - 83, center + 83, "final", False)
        for case, center in REVIEWS
        if center in args.centers
    ]
    if not args.no_overview:
        jobs.append(("cg20s", None, 75270, 78590, "final", True))
    if args.include_c28_31479:
        jobs.append(("known_31479", 31479, 31396, 31562, "c28", False))
    # Fail before hashing raw videos if any required method output is incomplete.
    for case, *_ in jobs:
        for condition in CONDITIONS:
            if not (analysis / case / f"{condition}.npz").is_file():
                raise FileNotFoundError(f"Missing completed arm: {case}/{condition}.npz")
    run_name = "render_reviews_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    with (
        threadpool_limits(limits=2),
        provenance("custom2d_comparison_reviews", output, name=run_name) as record,
    ):
        record.params_from(args)
        record.input("renderer", __file__, sha256=sha256(Path(__file__)))
        record.input("source_cache_manifest", cache_path, sha256=sha256(cache_path))
        if manifest_path.exists():
            record.param("previous_manifest_sha256", sha256(manifest_path))
        paths = {}
        for view, filename in zip(
            VIEWS, ("cam0_top.mp4", "cam1_side.mp4", "cam2_side.mp4"), strict=True
        ):
            relative = f"medaka_202512/{cache['experiment']}/{view}/{filename}"
            path = args.source_cache / relative
            expected = cache["files"][relative]
            print(f"Hashing raw source: {view}", flush=True)
            digest = sha256(path)
            if path.stat().st_size != expected["bytes"] or digest != expected["local_full_sha256"]:
                raise ValueError(f"Raw source cache changed: {view}")
            if digest != expected["source_stream_sha256"]:
                raise ValueError(f"Raw source differs from originally verified recording: {view}")
            record.input(f"raw_{view}", path, sha256=digest)
            paths[view] = path
        calibration = args.input_root / CALIBRATION
        calibration_manifest = calibration / "calibration_manifest.json"
        models = json.loads(calibration_manifest.read_text(encoding="utf-8"))
        record.input(
            "calibration_manifest", calibration_manifest, sha256=sha256(calibration_manifest)
        )
        for entry in models["model_files"]:
            path = calibration / entry["path"]
            digest = sha256(path)
            if digest != entry["sha256"]:
                raise ValueError("Calibration model hash mismatch")
            record.input(entry["camera"], path, sha256=digest)
        rig = load_rig(calibration)
        loaded = {}
        for case, center, first, stop, stage, overview in jobs:
            key = (case, stage)
            if key not in loaded:
                frames, arms, hashes = load_arms(analysis / case, stage)
                projected = project_arms(rig, arms)
                loaded[key] = frames, arms, hashes, projected
                for condition, digest in hashes.items():
                    record.input(
                        f"{case}/{condition}",
                        analysis / case / f"{condition}.npz",
                        sha256=digest,
                    )
            frames, arms, hashes, projected = loaded[key]
            stem = f"{case}_{'overview' if overview else 'f' + str(center)}_{stage}"
            name = stem + ".mp4"
            previous = next((entry for entry in manifest["entries"] if entry["path"] == name), None)
            if previous is not None:
                if (
                    previous["input_npz_sha256"] != hashes
                    or sha256(output / name) != previous["sha256"]
                ):
                    raise ValueError(f"Completed render/input mismatch: {name}")
                print(f"Preserved existing verified movie: {name}", flush=True)
                continue
            selected = (
                first + movie_indices(stop - first, SOURCE_FPS, OUTPUT_FPS)
                if overview
                else np.arange(first, stop)
            )
            mode = (
                "20 s real-time overview; nearest native samples"
                if overview
                else "All 166 native frames; 5.53 s slow replay"
            )
            print(f"Rendering {name} ({len(selected)} frames)", flush=True)
            metadata, still_path = render_video(
                paths,
                frames,
                arms,
                projected,
                selected,
                output / name,
                case=case,
                center=center,
                stage=stage,
                mode=mode,
            )
            entry = {
                "case": case,
                "center_frame": center,
                "path": name,
                "sha256": sha256(output / name),
                "native_frame_range_half_open": [first, stop],
                "output_frames": len(selected),
                "source_fps": SOURCE_FPS,
                "output_fps": OUTPUT_FPS,
                "columns": list(CONDITIONS),
                "stage": stage,
                "kind": "video",
                "input_npz_sha256": hashes,
                "analysis_directory": str(analysis.resolve()),
                "input_npz_paths": {
                    condition: str((analysis / case / f"{condition}.npz").resolve())
                    for condition in CONDITIONS
                },
                "metadata": metadata,
                "coordinates_interpolated": False,
                "all_slots_shown": [0, 1, 2],
            }
            manifest["entries"].append(entry)
            record.output(name, output / name, sha256=entry["sha256"])
            log_path = (output / name).with_suffix(".ffmpeg.log")
            record.output(log_path.name, log_path, sha256=sha256(log_path))
            if still_path is not None:
                still = {
                    **entry,
                    "path": still_path.name,
                    "sha256": sha256(still_path),
                    "kind": "image",
                    "native_frame_range_half_open": [center, center + 1],
                    "output_frames": 1,
                    "output_fps": None,
                    "metadata": {
                        "crops_xyxy_source_pixels": metadata["crops_xyxy_source_pixels"],
                        "png_decoded": True,
                        "shape": list(CANVAS_SHAPE),
                    },
                }
                manifest["entries"].append(still)
                record.output(still_path.name, still_path, sha256=still["sha256"])
            write_json(manifest_path, manifest)
            print(f"Verified full decode and saved: {name}", flush=True)
        write_json(manifest_path, manifest)
        record.output("manifest", manifest_path, sha256=sha256(manifest_path))
        for name, entry in record.inputs.items():
            if not name.startswith("raw_") and sha256(Path(entry["path"])) != entry["sha256"]:
                raise RuntimeError(f"Input changed while rendering: {name}")


if __name__ == "__main__":
    main()

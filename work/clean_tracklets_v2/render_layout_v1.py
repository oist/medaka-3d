"""Render old C34 / independent clean 2D / new 3D plus auxiliary joins.

Generic video utilities were copied from the earlier renderer; no lab imports
are used. Saved arrays are never modified. A fixed prefix permutation affects
only 3D display names. All original detection rows remain visible in column 2.
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

SOURCE_FPS, OUTPUT_FPS = 166.0, 30.0
SOURCE_TOTAL, SOURCE_SHAPE = 348600, (1200, 1200, 3)
WIDTH, ROW_HEIGHT, HEADER, FOOTER = 640, 440, 118, 72
CANVAS_SHAPE = (HEADER + 3 * ROW_HEIGHT + FOOTER, 3 * WIDTH, 3)
COLOURS = ((83, 190, 255), (234, 216, 83), (186, 132, 247))
CLEAN_COLOUR = (130, 230, 110)
JOIN_COLOUR = (70, 180, 255)
TITLES = (
    "Old saved C34 | display slots",
    "New clean 2D | F: local fragment",
    "New 3D | S: slot, F/J: clean/joined",
)
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


def register(record, name, path):
    record.input(name, path, sha256=sha256(path))


def projected_pose(rig, pose):
    output = np.full((len(pose), 3, 3, 3, 2), np.nan)
    finite = np.isfinite(pose).all(-1)
    for vi, view in enumerate(VIEWS):
        if finite.any():
            output[:, :, vi][finite] = rig.camera(view).project(pose[finite])
    return output


def load_case(analysis, previous, case, rig, record):
    directory = analysis / case
    path = directory / "reconstruction.npz"
    register(record, case + "_reconstruction", path)
    with np.load(path, allow_pickle=False) as z:
        frames = z["frames"].copy()
        new = {
            key: z[key].copy()
            for key in ("pose", "source", "uv", "projected_uv", "identity_review")
        }
    path = previous / case / "saved_baseline.npz"
    register(record, case + "_old_C34", path)
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["frames"], frames)
        old = {key: z[f"final__{key}"].copy() for key in ("pose", "source")}
    path = directory / "summary.json"
    register(record, case + "_summary", path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    alignment = summary["display_alignment"]
    permutation = alignment["permutation"] if alignment["valid"] else [0, 1, 2]
    for key in new:
        new[key] = new[key][:, permutation].copy()
    tracks = {}
    old["uv"] = np.full((len(frames), 3, 3, 3, 2), np.nan)
    for vi, view in enumerate(VIEWS):
        path = directory / f"tracking_F_{view}.npz"
        register(record, case + "_clean_" + view, path)
        with np.load(path, allow_pickle=False) as z:
            data = {key: z[key].copy() for key in ("frame_idx", "points", "labels", "clean_edge")}
        path = directory / f"joined_{view}.npz"
        register(record, case + "_joined_" + view, path)
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["frame_idx"], data["frame_idx"])
            np.testing.assert_array_equal(z["clean_labels"], data["labels"])
            data["joined_labels"] = z["joined_labels"].copy()
        data["start"] = np.searchsorted(data["frame_idx"], frames)
        data["stop"] = np.searchsorted(data["frame_idx"], frames + 1)
        for index in range(len(frames)):
            for slot in range(3):
                row = old["source"][index, slot, vi]
                if row >= 0:
                    flat = data["start"][index] + row
                    if flat >= data["stop"][index]:
                        raise ValueError("Old C34 source exceeds the native detections")
                    old["uv"][index, slot, vi] = data["points"][flat]
                chosen = new["source"][index, slot, vi]
                if chosen >= 0:
                    flat = data["start"][index] + chosen
                    if flat >= data["stop"][index]:
                        raise ValueError("New source exceeds the native detections")
                    np.testing.assert_array_equal(new["uv"][index, slot, vi], data["points"][flat])
        data["previous"] = np.full(len(data["labels"]), -1, np.int64)
        last = {}
        for row, label in enumerate(data["labels"]):
            if data["clean_edge"][row]:
                before = last.get(int(label))
                if before is None or data["frame_idx"][row] - data["frame_idx"][before] != 1:
                    raise ValueError("Clean incoming edge must refer to the previous native frame")
                data["previous"][row] = before
            last[int(label)] = row
        tracks[view] = data
    old["projected_uv"] = projected_pose(rig, old["pose"])
    path = directory / "joins.json"
    register(record, case + "_join_evidence", path)
    joins = json.loads(path.read_text(encoding="utf-8"))["bridges"]
    accepted = {(e["view"], e["to_frame"], e["to_row"]): e for e in joins if e["accepted"]}
    return dict(frames=frames, old=old, new=new, tracks=tracks, joins=accepted, alignment=alignment)


def common_crops(data, selected):
    indices = selected_indices(data["frames"], selected)
    crops = {}
    for vi, view in enumerate(VIEWS):
        tracks = data["tracks"][view]
        lo, hi = np.searchsorted(tracks["frame_idx"], [selected[0], selected[-1] + 1])
        points = [tracks["points"][lo:hi].reshape(-1, 2)]
        for name in ("old", "new"):
            points += [
                data[name][key][indices, :, vi].reshape(-1, 2) for key in ("uv", "projected_uv")
            ]
        points = np.concatenate(points)
        points = points[np.isfinite(points).all(-1)]
        points = points[(points >= 0).all(-1) & (points < 1200).all(-1)]
        if not len(points):
            crops[view] = [0, 0, 1200, 1200]
            continue
        low, high = points.min(0) - 55, points.max(0) + 55
        extent = np.minimum(1200, np.maximum(high - low, [500, 400]))
        low = np.floor(np.clip((low + high - extent) / 2, 0, 1200 - extent)).astype(int)
        high = np.ceil(low + extent).astype(int)
        crops[view] = [int(low[0]), int(low[1]), int(high[0]), int(high[1])]
    return crops


def fragment_colour(label):
    hue = (int(label) * 47 + 17) % 180
    hsv = np.array([[[hue, 140, 250]]], dtype=np.uint8)
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def skeleton(image, measured, projected, colour, label, scale, offset):
    measured, projected = measured * scale + offset, projected * scale + offset
    for a, b in ((0, 1), (1, 2)):
        if np.isfinite(measured[[a, b]]).all():
            cv2.line(image, point(measured[a]), point(measured[b]), colour, 1, cv2.LINE_AA)
    for node in range(3):
        if np.isfinite(measured[node]).all():
            cv2.circle(image, point(measured[node]), 5, colour, 1, cv2.LINE_AA)
        if np.isfinite(projected[node]).all():
            cv2.drawMarker(image, point(projected[node]), colour, cv2.MARKER_TILTED_CROSS, 9, 1)
    live = measured[np.isfinite(measured).all(-1)]
    if not len(live):
        live = projected[np.isfinite(projected).all(-1)]
    if len(live):
        x, y = point(live[0])
        text(image, label, (x + 7, y - 8), scale=0.43, colour=colour)


def link_mark(image, data, row, scale, offset, joined=None):
    current = data["points"][row, 1]
    if not np.isfinite(current).all():
        return
    current = current * scale + offset
    before = data["previous"][row]
    if before >= 0:
        a = data["points"][before, 1] * scale + offset
        if np.isfinite(a).all():
            cv2.line(image, point(a), point(current), CLEAN_COLOUR, 3, cv2.LINE_AA)
            cv2.circle(image, point(current), 8, CLEAN_COLOUR, 1)
    elif joined is None:
        x, y = point(current)
        cv2.rectangle(image, (x - 7, y - 7), (x + 7, y + 7), (170, 170, 170), 1)
    if joined is not None:
        a = data["points"][joined["from_row"], 1] * scale + offset
        if np.isfinite(a).all():
            length = max(1, int(np.linalg.norm(current - a) / 4))
            for i in range(0, length, 2):
                cv2.line(
                    image,
                    point(a + (current - a) * i / length),
                    point(a + (current - a) * min(i + 1, length) / length),
                    JOIN_COLOUR,
                    2,
                )
        cv2.circle(image, point(current), 11, JOIN_COLOUR, 2)
        x, y = point(current)
        text(image, "+JOIN", (x + 10, y + 16), colour=JOIN_COLOUR, scale=0.43)


def render_frame(images, data, index, native_frame, crops, case, mode):
    canvas = np.full(CANVAS_SHAPE, 20, np.uint8)
    text(
        canvas, f"{case} | native frame {native_frame} | {mode}", (14, 27), scale=0.65, thickness=2
    )
    text(
        canvas,
        "Rings: observations  X: projections  F/J: view-local fragments; "
        "arbitrary fragment colors do not identify 3D slots",
        (14, 51),
        scale=0.52,
    )
    text(
        canvas,
        "Green line/ring: initial clean edge   Gray square: new fragment   "
        "Orange dashed/+JOIN: separate cross-view join",
        (14, 75),
        scale=0.47,
    )
    for column, title in enumerate(TITLES):
        text(canvas, title, (column * WIDTH + 12, 106), scale=0.48)
    for vi, view in enumerate(VIEWS):
        x0, y0, x1, y1 = crops[view]
        base, scale, offset = letterbox(images[view][y0:y1, x0:x1], WIDTH - 8, ROW_HEIGHT - 76)
        offset = offset - np.array([x0, y0]) * scale
        tracks = data["tracks"][view]
        lo, hi = int(tracks["start"][index]), int(tracks["stop"][index])
        for column in range(3):
            panel = np.full((ROW_HEIGHT, WIDTH, 3), 20, np.uint8)
            text(panel, view, (12, 23), scale=0.52)
            content = base.copy()
            if column == 1:
                for row in range(lo, hi):
                    label = int(tracks["labels"][row])
                    skeleton(
                        content,
                        tracks["points"][row],
                        np.full((3, 2), np.nan),
                        fragment_colour(label),
                        f"F{label}",
                        scale,
                        offset,
                    )
                    link_mark(content, tracks, row, scale, offset)
                count = int(tracks["clean_edge"][lo:hi].sum())
                foot = f"All {hi - lo} detections; {count} clean incoming links"
                text(panel, foot, (10, ROW_HEIGHT - 25), scale=0.46)
            else:
                arm = data["old" if column == 0 else "new"]
                for slot, colour in enumerate(COLOURS):
                    measured, projected = (
                        arm["uv"][index, slot, vi],
                        arm["projected_uv"][index, slot, vi],
                    )
                    label = f"S{slot}"
                    selected = int(arm["source"][index, slot, vi])
                    if column == 2:
                        if arm["identity_review"][index, slot]:
                            label += "?"
                        if selected >= 0:
                            row = lo + selected
                            label += f" F{tracks['labels'][row]}/J{tracks['joined_labels'][row]}"
                            event = data["joins"].get((view, native_frame, row))
                            link_mark(content, tracks, row, scale, offset, event)
                    skeleton(content, measured, projected, colour, label, scale, offset)
                    n2d = int(np.isfinite(measured).all(-1).sum())
                    n3d = int(np.isfinite(projected).all(-1).sum())
                    text(
                        panel,
                        f"S{slot}: obs{n2d}/3 X{n3d}/3",
                        (10 + slot * 210, ROW_HEIGHT - 25),
                        colour=colour,
                        scale=0.39,
                    )
            panel[30 : ROW_HEIGHT - 46, 4 : WIDTH - 4] = content
            y, x = HEADER + vi * ROW_HEIGHT, column * WIDTH
            canvas[y : y + ROW_HEIGHT, x : x + WIDTH] = panel
    bottom = HEADER + 3 * ROW_HEIGHT
    text(
        canvas,
        "All three 3D slots and all 2D detections shown. '?' requests identity review; "
        "smooth motion is not identity truth.",
        (14, bottom + 24),
        scale=0.49,
    )
    text(
        canvas,
        "Missing detections remain absent. X projections do not create observations. "
        "No temporal interpolation or smoothing.",
        (14, bottom + 50),
        scale=0.49,
    )
    return canvas


def render_video(paths, data, selected, output, *, case, center, mode):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    for path in (output, output.with_suffix(".png"), output.with_suffix(".ffmpeg.log")):
        if path.exists():
            raise FileExistsError(f"Preserve existing render: {path}")
    indices = selected_indices(data["frames"], selected)
    crops = common_crops(data, selected)
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
    still = output.with_suffix(".png")
    still_frame = center if center is not None else int(selected[len(selected) // 2])
    with (
        source_captures(paths, int(selected[0])) as captures,
        output.with_suffix(".ffmpeg.log").open("xb") as log,
    ):
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        current = int(selected[0])
        try:
            for index, frame in zip(indices, selected, strict=True):
                images = read_selected(captures, current, int(frame))
                current = int(frame) + 1
                canvas = render_frame(images, data, int(index), int(frame), crops, case, mode)
                process.stdin.write(canvas.tobytes())
                if frame == still_frame and not cv2.imwrite(str(still), canvas):
                    raise RuntimeError("Still image write failed")
            process.stdin.close()
            if process.wait(timeout=60) != 0:
                raise RuntimeError(f"ffmpeg failed: {output}")
        except BaseException:
            process.kill()
            process.wait(timeout=10)
            raise
    verified = full_decode(output, len(selected))
    image = cv2.imread(str(still))
    if image is None or image.shape != CANVAS_SHAPE:
        raise RuntimeError("PNG readback failed")
    return dict(
        crops_xyxy_source_pixels=crops,
        full_decode=verified,
        native_sample_frames=selected.tolist(),
        ffmpeg_command=command,
    ), still


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=Path("work/clean_tracklets_v1"))
    parser.add_argument(
        "--previous", type=Path, default=Path("work/custom2d_comparison_v1/analysis_v3")
    )
    parser.add_argument("--source-cache", type=Path, default=Path("work/three_body/source_cache"))
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--centers", type=int, nargs="*", default=[c for _, c in REVIEWS])
    parser.add_argument("--no-overview", action="store_true")
    args = parser.parse_args()
    output = args.output or args.analysis / "review"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {
            "analysis_directory": str(args.analysis.resolve()),
            "assets": [],
            "interpretation": (
                "Local fragment numbers and 3D display slots are not biological identities"
            ),
            "overlap_note": "f76930 and f76988 clips overlap by 108 native frames",
        }
    )
    if manifest["analysis_directory"] != str(args.analysis.resolve()):
        raise ValueError("Use a separate review directory for another analysis revision")
    if set(args.centers) - {c for _, c in REVIEWS}:
        raise ValueError("Only the eight preregistered review anchors are allowed")
    jobs = [
        (case, center, center - 83, center + 83)
        for case, center in REVIEWS
        if center in args.centers
    ]
    if not args.no_overview:
        jobs.append(("cg20s", None, 75270, 78590))
    for case, *_ in jobs:
        if not (args.analysis / case / "summary.json").is_file():
            raise FileNotFoundError(f"Case incomplete: {case}")
    cache_path = args.source_cache / "source_cache_manifest.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache["status"] != "verified" or cache["experiment"] != "20251225_K_1_2_FCMC":
        raise ValueError("Raw source cache is not verified for this experiment")
    run_name = "render_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    with (
        threadpool_limits(limits=2),
        provenance("clean_tracklets_review", output, name=run_name) as record,
    ):
        record.params_from(args)
        register(record, "renderer", Path(__file__))
        register(record, "source_cache_manifest", cache_path)
        paths = {}
        for view, filename in zip(
            VIEWS, ("cam0_top.mp4", "cam1_side.mp4", "cam2_side.mp4"), strict=True
        ):
            relative = f"medaka_202512/{cache['experiment']}/{view}/{filename}"
            path = args.source_cache / relative
            expected = cache["files"][relative]
            print(f"Verifying raw bytes: {view}", flush=True)
            digest = sha256(path)
            if (
                path.stat().st_size != expected["bytes"]
                or digest != expected["local_full_sha256"]
                or digest != expected["source_stream_sha256"]
            ):
                raise ValueError("Raw video differs from verified source")
            record.input("raw_" + view, path, sha256=digest)
            paths[view] = path
        calibration = args.input_root / CALIBRATION
        for path in calibration.iterdir():
            if path.is_file():
                register(record, "calibration_" + path.name, path)
        rig = load_rig(calibration)
        loaded = {}
        for case, center, start, stop in jobs:
            name = f"review_f{center}" if center is not None else "cg20s_overview"
            previous = next((a for a in manifest["assets"] if a["name"] == name), None)
            if previous:
                for kind in ("video", "still"):
                    path = output / previous[kind]
                    if sha256(path) != previous[kind + "_sha256"]:
                        raise ValueError("Existing reviewed output changed")
                print(f"Verified existing {name}", flush=True)
                continue
            if case not in loaded:
                loaded[case] = load_case(args.analysis, args.previous, case, rig, record)
            data = loaded[case]
            selected = (
                np.arange(start, stop)
                if center is not None
                else start + movie_indices(stop - start, SOURCE_FPS, OUTPUT_FPS)
            )
            mode = (
                "All native frames, slowed to 30 fps"
                if center is not None
                else "20 s overview, 30 fps sampling"
            )
            print(f"Rendering {name}: {len(selected)} frames", flush=True)
            path = output / f"{name}.mp4"
            detail, still = render_video(
                paths, data, selected, path, case=case, center=center, mode=mode
            )
            asset = dict(
                name=name,
                case=case,
                center_frame=center,
                video=path.name,
                still=still.name,
                video_sha256=sha256(path),
                still_sha256=sha256(still),
                native_frame_range_half_open=[start, stop],
                output_frames=len(selected),
                output_fps=OUTPUT_FPS,
                source_fps=SOURCE_FPS,
                output_duration_seconds=len(selected) / OUTPUT_FPS,
                display_alignment=data["alignment"],
                **detail,
            )
            manifest["assets"].append(asset)
            record.output(name, path, sha256=asset["video_sha256"])
            record.output(name + "_still", still, sha256=asset["still_sha256"])
            write_json(manifest_path, manifest)
        record.output("manifest", manifest_path, sha256=sha256(manifest_path))


if __name__ == "__main__":
    main()

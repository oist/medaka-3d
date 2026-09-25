"""Per-camera direct comparison of frozen SLEAP and independent P/N/F labels.

No 3D coordinates, reconstructed sources, joined labels, or other lab imports.
Each panel receives exactly the same raw video frame and detection rows.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from medaka3d.aquarium import movie_indices, sha256
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

ARMS = ("SLEAP", "P", "N", "F")
TITLES = (
    "Saved SLEAP 2D tracking",
    "P: independent pectoral",
    "N: independent nodes",
    "F: independent nodes + LK",
)
SOURCE_FPS, OUTPUT_FPS = 166.0, 30.0
SOURCE_TOTAL = 348600
SOURCE_SHAPE = (1200, 1200, 3)
WIDTH, HEADER, PANEL_HEIGHT, FOOTER = 640, 84, 698, 88
CANVAS_SHAPE = (HEADER + PANEL_HEIGHT + FOOTER, 4 * WIDTH, 3)
TRAIL_FRAMES = 20
VIDEO_NAMES = ("cam0_top.mp4", "cam1_side.mp4", "cam2_side.mp4")
DETAIL_CENTERS = (76235, 76791, 76930, 76988, 77415)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def text(image, value, xy, scale=0.48, colour=(235, 235, 235), thickness=1):
    (width, height), baseline = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = int(np.clip(xy[0], 2, max(2, image.shape[1] - width - 2)))
    y = int(np.clip(xy[1], height + 2, max(height + 2, image.shape[0] - baseline - 2)))
    cv2.putText(
        image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA
    )


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


def validate_arm(arm, arrays):
    frames = arrays["frame_idx"]
    labels, previous, gap = (arrays[k] for k in ("labels", "previous_row", "link_gap"))
    count = len(frames)
    if (
        frames.ndim != 1
        or frames.dtype.kind not in "iu"
        or np.any(np.diff(frames.astype(np.int64)) < 0)
    ):
        raise ValueError("Native detection rows must be sorted integer frames")
    if arrays["points"].shape != (count, 3, 2) or np.isinf(arrays["points"]).any():
        raise ValueError("M/P/T points must retain NaNs for missing nodes")
    for key in ("labels", "clean_edge", "previous_row", "link_gap", "global_rows"):
        if arrays[key].shape != frames.shape:
            raise ValueError(f"{arm}/{key}: every detection row must be retained")
    if labels.dtype.kind not in "iu" or previous.dtype.kind not in "iu":
        raise ValueError("Labels and previous_row must be integers")
    if arrays["clean_edge"].dtype.kind != "b" or gap.dtype.kind not in "iu":
        raise ValueError("Clean-edge flags must be boolean and native gaps integer")
    if np.any(previous >= np.arange(count)) or np.any(previous < -1):
        raise ValueError("Previous-row references must point strictly backwards")
    linked = previous >= 0
    rows = np.flatnonzero(linked)
    if np.any(labels[rows] < 0) or np.any(labels[rows] != labels[previous[rows]]):
        raise ValueError("Unassigned/different IDs must not be connected")
    expected_gap = frames[rows].astype(np.int64) - frames[previous[rows]]
    if np.any(expected_gap <= 0) or not np.array_equal(gap[rows], expected_gap):
        raise ValueError("Link gaps must use native frame coordinates")
    if np.any(arrays["clean_edge"] & (~linked | (gap != 1))):
        raise ValueError("Clean incoming links must be native-adjacent")
    if arm != "SLEAP":
        if np.any(labels < 0) or not np.array_equal(arrays["clean_edge"], linked):
            raise ValueError("Clean fragments must preserve all rows and exactly their clean edges")
        if len(np.unique(np.c_[frames, labels], axis=0)) != count:
            raise ValueError("A clean fragment contains multiple detections in one frame")


def validate_shared(arms):
    reference = arms["SLEAP"]
    for name, arrays in arms.items():
        validate_arm(name, arrays)
        for key in ("frame_idx", "points", "global_rows"):
            if not np.array_equal(reference[key], arrays[key], equal_nan=True):
                raise ValueError(f"{name} changed the shared detection rows: {key}")
    return {
        "same_native_frames": True,
        "same_detection_points_including_nan": True,
        "same_global_rows": True,
        "all_input_rows_preserved": True,
        "native_gap_semantics_checked": True,
    }


def load_case(directory, case, view, record):
    arms = {}
    keys = (
        "frame_idx",
        "points",
        "labels",
        "clean_edge",
        "global_rows",
        "previous_row",
        "link_gap",
    )
    hashes, metadata = {}, {}
    for arm in ARMS:
        path = directory / case / f"tracking_{arm}_{view}.npz"
        digest = sha256(path)
        record.input(f"{case}_{view}_{arm}", path, sha256=digest)
        with np.load(path, allow_pickle=False) as z:
            arms[arm] = {key: z[key].copy() for key in keys}
        hashes[arm] = digest
        info = path.with_suffix(".json")
        record.input(f"{case}_{view}_{arm}_metadata", info, sha256=sha256(info))
        metadata[arm] = json.loads(info.read_text(encoding="utf-8"))["metadata"]
    invariants = validate_shared(arms)
    return {"arms": arms, "hashes": hashes, "invariants": invariants, "metadata": metadata}


def fragment_colour(label, arm, view):
    if label < 0:
        return (190, 190, 190)
    hue = (int(label) * 47 + ARMS.index(arm) * 31 + VIEWS.index(view) * 19 + 17) % 180
    hsv = np.array([[[hue, 150, 250]]], dtype=np.uint8)
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def tag(image, label, location, colour, occupied, scale=0.56):
    (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(np.clip(location[0], 4, max(4, image.shape[1] - width - 4)))
    step = height + baseline + 9
    candidates = [0] + [signed for n in range(1, 25) for signed in (-n * step, n * step)]
    for delta in candidates:
        y = int(np.clip(location[1] + delta, height + 4, image.shape[0] - baseline - 4))
        box = (x - 3, y - height - 3, x + width + 3, y + baseline + 2)
        if all(
            box[2] < old[0] or box[0] > old[2] or box[3] < old[1] or box[1] > old[3]
            for old in occupied
        ):
            break
    occupied.append(box)
    if delta:
        cv2.line(image, point(location), (x, y - height // 2), colour, 1, cv2.LINE_AA)
    cv2.rectangle(
        image, (x - 3, y - height - 3), (x + width + 3, y + baseline + 2), (18, 18, 18), -1
    )
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def edge_segments(arrays, row, frame):
    """Return only the stored label's native links; never cross a new fragment."""
    if arrays["labels"][row] < 0:
        return []
    segments = []
    child = int(row)
    cutoff = frame - TRAIL_FRAMES
    while arrays["previous_row"][child] >= 0:
        parent = int(arrays["previous_row"][child])
        if arrays["frame_idx"][parent] < cutoff:
            break
        segments.append((parent, child, int(arrays["link_gap"][child])))
        child = parent
    return segments


def dashed_line(image, a, b, colour):
    length = max(1, int(np.linalg.norm(b - a) / 5))
    for i in range(0, length, 2):
        cv2.line(
            image,
            point(a + (b - a) * i / length),
            point(a + (b - a) * min(i + 1, length) / length),
            colour,
            2,
        )


def overlay_detection(image, arrays, row, frame, arm, view, scale, offset, occupied):
    label = int(arrays["labels"][row])
    colour = fragment_colour(label, arm, view)
    coords = arrays["points"][row] * scale + offset
    for parent, child, gap in reversed(edge_segments(arrays, row, frame)):
        endpoints = arrays["points"][[parent, child], 1] * scale + offset
        if np.isfinite(endpoints).all():
            if gap == 1:
                cv2.line(image, point(endpoints[0]), point(endpoints[1]), colour, 2, cv2.LINE_AA)
            else:
                dashed_line(image, endpoints[0], endpoints[1], colour)
    for a, b in ((0, 1), (1, 2)):
        if np.isfinite(coords[[a, b]]).all():
            cv2.line(image, point(coords[a]), point(coords[b]), colour, 1, cv2.LINE_AA)
    for node in coords:
        if np.isfinite(node).all():
            cv2.circle(image, point(node), 4, colour, 1, cv2.LINE_AA)
    previous = int(arrays["previous_row"][row])
    gap = int(arrays["link_gap"][row])
    if label < 0:
        name = "UNASSIGNED"
    elif previous < 0:
        name = f"ID{label} NEW"
    elif gap > 1:
        name = f"ID{label} GAP{gap}"
    else:
        name = f"ID{label}"
    if previous < 0 and np.isfinite(coords[1]).all():
        x, y = point(coords[1])
        cv2.rectangle(image, (x - 7, y - 7), (x + 7, y + 7), (180, 180, 180), 1)
    live = coords[np.isfinite(coords).all(-1)]
    if len(live):
        x, y = point(live[0])
        tag(image, name, (x + 8, y - 9), colour, occupied)


def render_frame(raw, data, frame, case, view, mode):
    if raw.shape != SOURCE_SHAPE or raw.dtype != np.uint8:
        raise ValueError("Raw pixels changed shape or dtype")
    canvas = np.full(CANVAS_SHAPE, 18, np.uint8)
    text(
        canvas,
        f"{case} | {view} | native frame {frame} | {frame / SOURCE_FPS:.4f} s",
        (12, 27),
        scale=0.68,
        thickness=2,
    )
    text(canvas, mode, (12, 52), scale=0.53)
    base, scale, offset = letterbox(raw, WIDTH - 8, WIDTH - 8)
    frames = data["arms"]["SLEAP"]["frame_idx"]
    lo, hi = np.searchsorted(frames, [frame, frame + 1])
    for column, arm in enumerate(ARMS):
        panel = np.full((PANEL_HEIGHT, WIDTH, 3), 18, np.uint8)
        text(canvas, TITLES[column], (column * WIDTH + 12, 77), scale=0.56, thickness=1)
        content = base.copy()
        arrays = data["arms"][arm]
        occupied = []
        for row in range(lo, hi):
            overlay_detection(content, arrays, row, frame, arm, view, scale, offset, occupied)
        panel[0 : WIDTH - 8, 4 : WIDTH - 4] = content
        labels = arrays["labels"][lo:hi]
        assigned = labels >= 0
        new = assigned & (arrays["previous_row"][lo:hi] < 0)
        gaps = assigned & (arrays["link_gap"][lo:hi] > 1)
        linked = arrays["clean_edge"][lo:hi]
        nodes = int(np.isfinite(arrays["points"][lo:hi]).all(-1).sum())
        text(
            panel,
            f"{hi - lo} detections | {int(linked.sum())} adjacent links | "
            f"{nodes}/{3 * (hi - lo)} nodes",
            (10, WIDTH + 16),
            scale=0.49,
        )
        text(
            panel,
            f"NEW {int(new.sum())} | UNASSIGNED {int((~assigned).sum())} | "
            f"ID gap {int(gaps.sum())}",
            (10, WIDTH + 42),
            scale=0.47,
        )
        canvas[HEADER : HEADER + PANEL_HEIGHT, column * WIDTH : (column + 1) * WIDTH] = panel
    bottom = HEADER + PANEL_HEIGHT
    text(
        canvas,
        "Exactly the same raw frame and detected points in all four panels. "
        "IDs and colors are arbitrary within each method/view.",
        (12, bottom + 24),
        scale=0.57,
    )
    text(
        canvas,
        "NEW = fragment start / connection cut, not a missing detection. "
        "UNASSIGNED = a detected SLEAP row without a tracking ID.",
        (12, bottom + 49),
        scale=0.55,
    )
    text(
        canvas,
        f"Pectoral trails: last {TRAIL_FRAMES} native frames. "
        "Dashed lines: saved ID across a gap; no points are invented. "
        "M-P-T lines: measured nodes.",
        (12, bottom + 73),
        scale=0.52,
    )
    return canvas


def open_capture(path, first):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    if not cap.isOpened():
        raise RuntimeError("Cannot open original raw video")
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)
    if shape != SOURCE_SHAPE or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != SOURCE_TOTAL:
        cap.release()
        raise ValueError("Unexpected raw source dimensions/frame count")
    if not np.isclose(cap.get(cv2.CAP_PROP_FPS), SOURCE_FPS, atol=1e-6, rtol=0):
        cap.release()
        raise ValueError("Unexpected raw source FPS")
    if (
        not cap.set(cv2.CAP_PROP_POS_FRAMES, int(first))
        or cap.get(cv2.CAP_PROP_POS_FRAMES) != first
    ):
        cap.release()
        raise RuntimeError("Native raw video seek failed")
    return cap


def read_selected(cap, current, target):
    for native in range(current, target):
        if not cap.grab() or cap.get(cv2.CAP_PROP_POS_FRAMES) != native + 1:
            raise RuntimeError("Raw native-frame skip mismatch")
    ok, raw = cap.read()
    if not ok or cap.get(cv2.CAP_PROP_POS_FRAMES) != target + 1:
        raise RuntimeError("Raw native-frame decode mismatch")
    return raw


def full_decode(path, count):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    found = 0
    try:
        if not cap.isOpened() or not np.isclose(cap.get(cv2.CAP_PROP_FPS), OUTPUT_FPS):
            raise RuntimeError("Encoded comparison video open/FPS check failed")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != CANVAS_SHAPE:
                raise RuntimeError("Encoded comparison shape differs")
            found += 1
    finally:
        cap.release()
    if found != count:
        raise RuntimeError("Encoded comparison lost frames")
    return {"verified": True, "decoded_frames": found, "shape": list(CANVAS_SHAPE)}


def render_video(source, data, selected, output, *, case, view, center):
    if np.any(np.diff(selected) <= 0) or selected[0] < 0 or selected[-1] >= SOURCE_TOTAL:
        raise ValueError("Native sample frames must be strictly increasing inside the recording")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    still = output.with_suffix(".png")
    log_path = output.with_suffix(".ffmpeg.log")
    if any(p.exists() for p in (output, still, log_path)):
        raise FileExistsError("Preserve previous rendered output")
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
        "24",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    mode = (
        "All 166 native frames at 30 fps (slow motion)"
        if center is not None
        else "20 seconds at 30 fps (native frames sampled; no time interpolation)"
    )
    still_frame = center if center is not None else int(selected[len(selected) // 2])
    cap = open_capture(source, int(selected[0]))
    with log_path.open("xb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        current = int(selected[0])
        try:
            for frame in selected:
                raw = read_selected(cap, current, int(frame))
                current = int(frame) + 1
                canvas = render_frame(raw, data, int(frame), case, view, mode)
                process.stdin.write(canvas.tobytes())
                if frame == still_frame and not cv2.imwrite(str(still), canvas):
                    raise RuntimeError("Comparison PNG write failed")
            process.stdin.close()
            if process.wait(timeout=60) != 0:
                raise RuntimeError("Comparison encoder failed")
        except BaseException:
            process.kill()
            process.wait(timeout=10)
            raise
        finally:
            cap.release()
    verified = full_decode(output, len(selected))
    if cv2.imread(str(still)).shape != CANVAS_SHAPE:
        raise RuntimeError("Comparison still readback failed")
    return {
        "full_decode": verified,
        "native_sample_frames": selected.tolist(),
        "ffmpeg_command": command,
    }, still


def make_jobs(protocol):
    """Main view overviews, historical anchors, then full-rate primary anchors."""
    main = next(case for case in protocol["cases"] if case["name"] == "cg20s")
    specs = [(main, None, main["start"], main["stop"])]
    specs += [
        (case, center, center - 83, center + 83)
        for case in protocol["cases"]
        if case.get("render") and case["name"] != "cg20s"
        for center in case["centers"]
    ]
    specs += [(main, center, center - 83, center + 83) for center in main["centers"]]
    return [
        dict(case=case["name"], view=view, center=center, start=start, stop=stop)
        for case, center, start, stop in specs
        for view in VIEWS
    ]


def wait_for_inputs(directory, case, view, timeout=1800):
    paths = [
        directory / case / f"tracking_{arm}_{view}{suffix}"
        for arm in ARMS
        for suffix in (".npz", ".json")
    ]
    started = time.monotonic()
    next_notice = 0
    while not all(path.is_file() for path in paths):
        elapsed = time.monotonic() - started
        if elapsed > timeout:
            raise TimeoutError(f"Input case/view not complete: {case}/{view}")
        if elapsed >= next_notice:
            print(f"Waiting for input: {case}/{view}", flush=True)
            next_notice = elapsed + 30
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("work/clean2d_video_comparison_v1"))
    parser.add_argument("--source-cache", type=Path, default=Path("work/three_body/source_cache"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--views", choices=VIEWS, nargs="+", default=list(VIEWS))
    parser.add_argument("--centers", type=int, nargs="*")
    parser.add_argument("--no-overview", action="store_true")
    args = parser.parse_args()
    output = args.output or args.data / "review"
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.data / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["fps"] != SOURCE_FPS:
        raise ValueError("This renderer is specified for 166 native fps")
    all_jobs = make_jobs(protocol)
    allowed_centers = {job["center"] for job in all_jobs if job["center"] is not None}
    if args.centers is not None and set(args.centers) - allowed_centers:
        raise ValueError("Requested anchor not in the fixed protocol")
    jobs = [
        job
        for job in all_jobs
        if job["view"] in args.views
        and not (args.no_overview and job["center"] is None)
        and (job["center"] is None or args.centers is None or job["center"] in args.centers)
    ]
    manifest_path = output / "manifest.json"
    code_hash = sha256(Path(__file__))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["data_directory"] != str(args.data.resolve())
            or manifest["renderer_sha256"] != code_hash
        ):
            raise ValueError("Use a new review directory for changed data path or renderer")
    else:
        manifest = dict(
            status="in_progress",
            data_directory=str(args.data.resolve()),
            assets=[],
            renderer_sha256=code_hash,
            protocol_sha256=sha256(protocol_path),
            expected_assets=len(all_jobs),
            arm_order=list(ARMS),
            interpretation="IDs/colors are method- and view-local; new fragments retain detections",
            crop="Full 1200x1200 raw view, shared identically by all four panels",
            trail_native_frames=TRAIL_FRAMES,
        )
    if manifest["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Fixed data protocol changed")
    cache_path = args.source_cache / "source_cache_manifest.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache["status"] != "verified" or cache["experiment"] != protocol["experiment"]:
        raise ValueError("Verified raw cache does not match the experiment")
    run_name = "render_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    with (
        threadpool_limits(limits=2),
        provenance("clean2d_direct_video_comparison", output, name=run_name) as record,
    ):
        record.params_from(args)
        record.input("renderer", Path(__file__), sha256=code_hash)
        record.input("protocol", protocol_path, sha256=sha256(protocol_path))
        record.input("raw_cache_manifest", cache_path, sha256=sha256(cache_path))
        paths = {}
        for view, filename in zip(VIEWS, VIDEO_NAMES, strict=True):
            if view not in args.views:
                continue
            relative = f"medaka_202512/{protocol['experiment']}/{view}/{filename}"
            path = args.source_cache / relative
            expected = cache["files"][relative]
            print(f"Verifying original raw video: {view}", flush=True)
            digest = sha256(path)
            if (
                path.stat().st_size != expected["bytes"]
                or digest != expected["local_full_sha256"]
                or digest != expected["source_stream_sha256"]
            ):
                raise ValueError("Raw video changed from the verified source")
            record.input("raw_" + view, path, sha256=digest)
            paths[view] = path
        loaded = {}
        for job in jobs:
            case, view, center, start, stop = (
                job[k] for k in ("case", "view", "center", "start", "stop")
            )
            name = f"{view}__f{center}" if center is not None else f"{view}__cg20s_overview"
            existing = next((a for a in manifest["assets"] if a["name"] == name), None)
            if existing:
                for kind in ("video", "still"):
                    if sha256(output / existing[kind]) != existing[kind + "_sha256"]:
                        raise ValueError("Previously reviewed output changed")
                print(f"Verified existing {name}", flush=True)
                continue
            key = (case, view)
            if key not in loaded:
                wait_for_inputs(args.data, case, view)
                loaded[key] = load_case(args.data, case, view, record)
            data = loaded[key]
            selected = (
                np.arange(start, stop, dtype=np.int64)
                if center is not None
                else start + movie_indices(stop - start, SOURCE_FPS, OUTPUT_FPS)
            )
            print(f"Rendering {name}: {len(selected)} frames", flush=True)
            path = output / f"{name}.mp4"
            details, still = render_video(
                paths[view], data, selected, path, case=case, view=view, center=center
            )
            source_frames = data["arms"]["SLEAP"]["frame_idx"]
            native = np.arange(start, stop)
            counts = np.searchsorted(source_frames, native + 1) - np.searchsorted(
                source_frames, native
            )
            asset = dict(
                name=name,
                case=case,
                view=view,
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
                arm_order=list(ARMS),
                input_npz_sha256=data["hashes"],
                input_metadata=data["metadata"],
                renderer_invariants=data["invariants"],
                input_detection_count_range=[int(counts.min()), int(counts.max())],
                source_pixels=(
                    "Full native image, spatially reduced only; no temporal interpolation"
                ),
                **details,
            )
            manifest["assets"].append(asset)
            manifest["status"] = (
                "complete" if len(manifest["assets"]) == len(all_jobs) else "in_progress"
            )
            write_json(manifest_path, manifest)
            record.output(name, path, sha256=asset["video_sha256"])
            record.output(name + "_still", still, sha256=asset["still_sha256"])
        record.output("manifest", manifest_path, sha256=sha256(manifest_path))


if __name__ == "__main__":
    main()

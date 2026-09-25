"""Same-observation C27/P0/P1 films, full-view context and magnified event evidence.

Drawing/decoder conventions adapted from frozen clean2d renderer, SHA256
228c92dd7896e6289d8977e2f2ddd1d08a13ecc8f3111af88c692b287abd2e96.
No cross-lab imports. No interpolated observations, identity matching, or 3D inputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
VIEWS = ("top_cam", "side1_cam", "side2_cam")
METHODS = ("C27", "S1", "S2", "P0", "P1", "P1aux")
ARMS = ("C27", "P0", "P1")
SOURCE_FPS = 166.0
OUTPUT_FPS = 30.0
SOURCE_SHAPE = (1200, 1200, 3)
WIDTH = 448
HEADER = 88
FOOTER = 92
VIDEO_NAMES = ("cam0_top.mp4", "cam1_side.mp4", "cam2_side.mp4")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate_shared(arms):
    """Rendering is forbidden unless every method preserves the same source rows."""
    reference = arms["C27"]
    for method, values in arms.items():
        frames, labels = values["frame_idx"], values["labels"]
        count = len(frames)
        if frames.dtype.kind not in "iu" or np.any(np.diff(frames.astype(np.int64)) < 0):
            raise ValueError("Native frames must be sorted integers")
        if values["points"].shape != (count, 3, 2) or np.isinf(values["points"]).any():
            raise ValueError("M/P/T points must retain source NaNs")
        for key in ("frame_idx", "global_rows", "points"):
            if not np.array_equal(reference[key], values[key], equal_nan=True):
                raise ValueError(f"{method} changed shared detection {key}")
        if labels.shape != frames.shape or labels.dtype.kind not in "iu":
            raise ValueError("Every input row needs a stored local label")
        assigned = labels >= 0
        if len(np.unique(np.c_[frames[assigned], labels[assigned]], axis=0)) != assigned.sum():
            raise ValueError("One local ID cannot consume two rows in a native frame")
        previous = values["previous_row"]
        if previous.shape != frames.shape or np.any(previous >= np.arange(count)):
            raise ValueError("Previous links must point strictly backwards")
        if np.any(previous < -1):
            raise ValueError("Invalid previous-row sentinel")
        rows = np.flatnonzero(previous >= 0)
        if np.any(labels[rows] < 0) or np.any(labels[rows] != labels[previous[rows]]):
            raise ValueError("Previous links must retain same nonnegative local identity")
        gaps = frames[rows] - frames[previous[rows]]
        if np.any(gaps <= 0):
            raise ValueError("Previous links must have positive native frame gaps")
        if method.startswith("P") and np.any(gaps != 1):
            raise ValueError("Independent confirmed links cannot cross a gap")
    return dict(
        same_native_frames=True,
        same_points_including_nan=True,
        same_global_rows=True,
        all_input_rows_preserved=True,
    )


def load_case(directory, case, view, methods=METHODS):
    arms, hashes = {}, {}
    for method in methods:
        path = directory / case / f"tracking_{method}_{view}.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        for canonical, alternate in (
            ("frame_idx", "native_frame"),
            ("global_rows", "global_row"),
            ("labels", "local_id"),
        ):
            if canonical not in arrays:
                arrays[canonical] = arrays[alternate]
        arms[method] = arrays
        hashes[method] = sha256(path)
    invariants = validate_shared(arms)
    return dict(arms=arms, hashes=hashes, invariants=invariants)


def colour(label, method, view):
    if label < 0:
        return (210, 210, 210)
    hue = (int(label) * 47 + METHODS.index(method) * 31 + VIEWS.index(view) * 19 + 17) % 180
    return tuple(
        int(v)
        for v in cv2.cvtColor(np.array([[[hue, 150, 255]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    )


def text(image, value, xy, scale=0.43, color=(240, 240, 240)):
    cv2.putText(
        image,
        str(value),
        tuple(map(int, xy)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def point(value):
    return tuple(np.rint(np.clip(value, -100000, 100000)).astype(int))


def edge_segments(arrays, row, frame, trail=20):
    segments = []
    child = int(row)
    while arrays["previous_row"][child] >= 0:
        parent = int(arrays["previous_row"][child])
        if arrays["frame_idx"][parent] < frame - trail:
            break
        segments.append(
            (parent, child, int(arrays["frame_idx"][child] - arrays["frame_idx"][parent]))
        )
        child = parent
    return segments


def identity_tag(image, name, location, color, occupied):
    """Use compact outlined glyphs and collision avoidance without covering raw fish."""
    (width, height), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
    x = int(np.clip(location[0] + 9, 3, image.shape[1] - width - 4))
    for shift in [0, -20, 20, -40, 40, -60, 60, -80, 80, -100, 100]:
        y = int(np.clip(location[1] - 9 + shift, height + 4, image.shape[0] - baseline - 4))
        box = (x - 2, y - height - 2, x + width + 2, y + baseline + 2)
        if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in occupied):
            break
    occupied.append(box)
    if shift:
        cv2.line(image, point(location), (x, y - height // 2), color, 1, cv2.LINE_AA)
    cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)


def overlay_detection(image, arrays, row, frame, method, view, scale, offset, slot, occupied=None):
    """Draw observed nodes only; every row also appears in the text strip."""
    label = int(arrays["labels"][row])
    color = colour(label, method, view)
    coords = arrays["points"][row] * scale + offset
    for parent, child, gap in edge_segments(arrays, row, frame):
        ends = arrays["points"][[parent, child], 1] * scale + offset
        if np.isfinite(ends).all():
            if gap == 1:
                cv2.line(image, point(ends[0]), point(ends[1]), color, 2, cv2.LINE_AA)
            else:
                for fraction in np.arange(0, 1, 0.2):
                    cv2.line(
                        image,
                        point(ends[0] + fraction * (ends[1] - ends[0])),
                        point(ends[0] + (fraction + 0.1) * (ends[1] - ends[0])),
                        color,
                        1,
                        cv2.LINE_AA,
                    )
    for a, b in ((0, 1), (1, 2)):
        if np.isfinite(coords[[a, b]]).all():
            cv2.line(image, point(coords[a]), point(coords[b]), color, 1, cv2.LINE_AA)
    for node_index, node in enumerate(coords):
        if np.isfinite(node).all():
            cv2.circle(image, point(node), 4 if node_index == 1 else 3, color, 1, cv2.LINE_AA)
    finite = coords[np.isfinite(coords).all(-1)]
    finite = finite[
        (finite[:, 0] >= 0)
        & (finite[:, 0] < image.shape[1])
        & (finite[:, 1] >= 0)
        & (finite[:, 1] < image.shape[0])
    ]
    if len(finite):
        name = f"L{label}" if label >= 0 else "UNASSIGNED"
        if arrays["previous_row"][row] < 0:
            name += " NEW"
        identity_tag(image, name, finite[0], color, occupied if occupied is not None else [])


def choose_crop(arrays, center):
    """Magnify the closest observed P pair; the full view is always retained above."""
    frames = arrays["frame_idx"]
    lo, hi = np.searchsorted(frames, [center, center + 1])
    points = arrays["points"][lo:hi, 1]
    points = points[np.isfinite(points).all(-1)]
    if len(points) > 1:
        distance = np.linalg.norm(points[:, None] - points[None], axis=-1)
        np.fill_diagonal(distance, np.inf)
        i, j = np.unravel_index(np.argmin(distance), distance.shape)
        middle = (points[i] + points[j]) / 2
        size = min(800, max(440, int(np.max(np.abs(points[i] - points[j]))) + 160))
    else:
        middle = points[0] if len(points) else np.array([600, 600])
        size = 440
    left, top = np.clip(middle - size / 2, 0, 1200 - size).astype(int)
    return [int(left), int(top), int(size), int(size)]


def render_frame(raw, data, frame, case, view, crop=None):
    if raw.shape != SOURCE_SHAPE or raw.dtype != np.uint8:
        raise ValueError("Unexpected original source pixels")
    body_height = WIDTH * (2 if crop is not None else 1)
    canvas = np.full((HEADER + body_height + FOOTER, WIDTH * 3, 3), 18, np.uint8)
    text(canvas, f"{case} | {view} | native {frame} | {frame / SOURCE_FPS:.4f} sec", (12, 25), 0.63)
    mode = (
        "30 fps sampled / original duration"
        if crop is None
        else "Every native frame / 5.53x slow / full view + zoom"
    )
    text(canvas, mode, (12, 49), 0.52)
    reference = data["arms"]["C27"]
    lo, hi = np.searchsorted(reference["frame_idx"], [frame, frame + 1])
    full = cv2.resize(raw, (WIDTH, WIDTH), interpolation=cv2.INTER_AREA)
    for column, method in enumerate(ARMS):
        x = column * WIDTH
        text(
            canvas,
            method + (" saved full-recording" if method == "C27" else " independent 2D"),
            (x + 10, 76),
            0.53,
        )
        arrays = data["arms"][method]
        panel = full.copy()
        occupied = []
        for slot, row in enumerate(range(lo, hi)):
            overlay_detection(
                panel,
                arrays,
                row,
                frame,
                method,
                view,
                WIDTH / 1200,
                np.zeros(2),
                slot,
                occupied,
            )
        if crop is not None:
            cx, cy, cw, ch = crop
            cv2.rectangle(
                panel,
                point(np.array([cx, cy]) * WIDTH / 1200),
                point(np.array([cx + cw, cy + ch]) * WIDTH / 1200),
                (240, 240, 240),
                1,
            )
            zoom = cv2.resize(
                raw[cy : cy + ch, cx : cx + cw], (WIDTH, WIDTH), interpolation=cv2.INTER_LINEAR
            )
            occupied = []
            for slot, row in enumerate(range(lo, hi)):
                overlay_detection(
                    zoom,
                    arrays,
                    row,
                    frame,
                    method,
                    view,
                    WIDTH / cw,
                    -np.array([cx, cy]) * WIDTH / cw,
                    slot,
                    occupied,
                )
            text(zoom, f"Zoom [{cx},{cy},{cw},{ch}]", (10, WIDTH - 12))
            canvas[HEADER + WIDTH : HEADER + 2 * WIDTH, x : x + WIDTH] = zoom
        canvas[HEADER : HEADER + WIDTH, x : x + WIDTH] = panel
        bottom = HEADER + body_height
        nodes = np.isfinite(arrays["points"][lo:hi]).all(-1)
        linked = int(arrays["clean_edge"][lo:hi].sum())
        text(
            canvas,
            f"{hi - lo} rows | {linked} adjacent | M/P/T observed {nodes.sum()}/{nodes.size}",
            (x + 8, bottom + 21),
        )
        labels = [
            f"r{int(arrays['global_rows'][row])}:L{int(arrays['labels'][row])}"
            for row in range(lo, hi)
        ]
        text(canvas, " ".join(labels[:3]), (x + 8, bottom + 42), 0.34)
        text(canvas, " ".join(labels[3:]), (x + 8, bottom + 59), 0.34)
    text(
        canvas,
        "Same raw pixels/rows. IDs/colors are method-local. "
        "Solid=observed. No predicted points shown.",
        (12, canvas.shape[0] - 10),
        0.48,
    )
    return canvas


def make_jobs(protocol):
    main = next(c for c in protocol["cases"] if c["name"] == "cg20s")
    specs = [(main, None, main["start"], main["stop"])]
    for case in protocol["cases"]:
        if case.get("render", bool(case.get("centers"))):
            specs.extend(
                (case, center, center - 83, center + 83) for center in case.get("centers", [])
            )
    return [
        dict(case=case["name"], view=view, center=center, start=start, stop=stop)
        for case, center, start, stop in specs
        for view in VIEWS
    ]


def open_capture(path, first):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not cap.isOpened() or not np.isclose(cap.get(cv2.CAP_PROP_FPS), SOURCE_FPS):
        raise RuntimeError("Cannot decode expected 166-fps original movie")
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(first)):
        raise RuntimeError("Original native-frame seek failed")
    return cap


def full_decode(path, expected_count, shape):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    count = 0
    try:
        if not cap.isOpened() or not np.isclose(cap.get(cv2.CAP_PROP_FPS), OUTPUT_FPS):
            raise RuntimeError("Encoded comparison movie failed to open")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != shape:
                raise RuntimeError("Encoded shape changed")
            count += 1
    finally:
        cap.release()
    if count != expected_count:
        raise RuntimeError(f"Full decode count {count} differs from {expected_count}")
    return dict(verified=True, decoded_frames=count, shape=list(shape), fps=OUTPUT_FPS)


def render_job(job, directory, output, source):
    case, view, center = (job[k] for k in ("case", "view", "center"))
    name = f"{view}__f{center}" if center is not None else f"{view}__cg20s_overview"
    receipt = output / (name + ".json")
    if receipt.is_file():
        asset = json.loads(receipt.read_text(encoding="utf-8"))
        for kind in ("video", "before", "center", "after"):
            if sha256(output / asset[kind]) != asset[kind + "_sha256"]:
                raise ValueError("Existing media changed")
        return asset
    data = load_case(directory, case, view)
    crop = choose_crop(data["arms"]["C27"], center) if center is not None else None
    start, stop = job["start"], job["stop"]
    selected = (
        np.arange(start, stop, dtype=np.int64)
        if center is not None
        else start
        + np.floor(
            np.arange(round((stop - start) / SOURCE_FPS * OUTPUT_FPS)) * SOURCE_FPS / OUTPUT_FPS
        ).astype(np.int64)
    )
    still_frames = [
        int(selected[0]),
        int(center if center is not None else selected[len(selected) // 2]),
        int(selected[-1]),
    ]
    shape = (HEADER + WIDTH * (2 if crop else 1) + FOOTER, WIDTH * 3, 3)
    movie = output / (name + ".mp4")
    if movie.exists():
        raise FileExistsError(f"Unreceipted media: {movie}; inspect before retrying")
    command = [
        shutil.which("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{shape[1]}x{shape[0]}",
        "-r",
        str(OUTPUT_FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-threads",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(movie),
    ]
    print(f"Rendering {name}: {len(selected)} frames", flush=True)
    cap = open_capture(source, int(selected[0]))
    stills = {}
    with movie.with_suffix(".ffmpeg.log").open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        current = int(selected[0])
        try:
            for native in selected:
                for _ in range(current, int(native)):
                    if not cap.grab():
                        raise RuntimeError("Source frame skip failed")
                ok, raw = cap.read()
                if not ok or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != int(native) + 1:
                    raise RuntimeError("Native source alignment failed")
                current = int(native) + 1
                frame = render_frame(raw, data, int(native), case, view, crop)
                process.stdin.write(frame.tobytes())
                for suffix, target in zip(("before", "center", "after"), still_frames, strict=True):
                    if native == target:
                        path = output / f"{name}__{suffix}.png"
                        if not cv2.imwrite(str(path), frame):
                            raise RuntimeError("Still export failed")
                        stills[suffix] = path.name
                        stills[suffix + "_sha256"] = sha256(path)
            process.stdin.close()
            if process.wait(timeout=60):
                raise RuntimeError("Movie encoder failed")
        except BaseException:
            process.kill()
            process.wait(timeout=10)
            raise
        finally:
            cap.release()
    decoded = full_decode(movie, len(selected), shape)
    asset = dict(
        name=name,
        case=case,
        view=view,
        center_frame=center,
        video=movie.name,
        video_sha256=sha256(movie),
        **stills,
        still_native_frames=still_frames,
        crop_xywh=crop,
        zoom_policy="closest P pair at center; full view also displayed",
        native_frame_range_half_open=[start, stop],
        native_sample_frames=selected.tolist(),
        output_frames=len(selected),
        output_fps=OUTPUT_FPS,
        output_duration_seconds=len(selected) / OUTPUT_FPS,
        source_fps=SOURCE_FPS,
        arm_order=list(ARMS),
        input_npz_sha256=data["hashes"],
        renderer_invariants=data["invariants"],
        full_decode=decoded,
        renderer_sha256=sha256(Path(__file__)),
    )
    write_json(receipt, asset)
    return asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "work/variable_count_tracklets_v1")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    parser.add_argument("--centers", nargs="*", type=int)
    parser.add_argument("--no-overview", action="store_true")
    args = parser.parse_args()
    output = args.data / "review"
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.data / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    all_jobs = make_jobs(protocol)
    jobs = [
        j
        for j in all_jobs
        if j["view"] in args.views
        and not (args.no_overview and j["center"] is None)
        and (j["center"] is None or args.centers is None or j["center"] in args.centers)
    ]
    source_root = ROOT / "work/three_body/source_cache"
    cache_path = source_root / "source_cache_manifest.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    sources = {
        view: source_root / "medaka_202512" / protocol["experiment"] / view / name
        for view, name in zip(VIEWS, VIDEO_NAMES, strict=True)
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    with provenance("variable_count_comparison_media", output, name="render_" + stamp) as record:
        record.params_from(args)
        record.input("renderer", Path(__file__), sha256=sha256(Path(__file__)))
        record.input("protocol", protocol_path, sha256=sha256(protocol_path))
        record.input("source_cache_manifest", cache_path, sha256=sha256(cache_path))
        for view in args.views:
            source = sources[view]
            relative = source.relative_to(source_root).as_posix()
            expected = cache["files"][relative]
            # Cached cryptographic receipt is preserved, then size and decoder are checked here.
            if source.stat().st_size != expected["bytes"]:
                raise ValueError("Original source size changed")
            record.input("source_" + view, source, sha256=expected["local_full_sha256"])
        record.param("source_hash_policy", "previous full-copy SHA receipt + current size/decode")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(render_job, j, args.data, output, sources[j["view"]]) for j in jobs
            ]
            for future in as_completed(futures):
                asset = future.result()
                record.output(asset["name"], output / asset["video"], sha256=asset["video_sha256"])
                print(f"Decoded and verified {asset['name']}", flush=True)
        assets = [
            json.loads(p.read_text(encoding="utf-8")) for p in sorted(output.glob("*__*.json"))
        ]
        manifest = dict(
            status="complete" if len(assets) == len(all_jobs) else "partial",
            expected_assets=len(all_jobs),
            assets=assets,
            arm_order=list(ARMS),
            protocol_sha256=sha256(protocol_path),
            renderer_sha256=sha256(Path(__file__)),
            all_videos_fully_decoded=all(a["full_decode"]["verified"] for a in assets),
        )
        write_json(output / "manifest.json", manifest)
        record.output("manifest", output / "manifest.json", sha256=sha256(output / "manifest.json"))


if __name__ == "__main__":
    main()

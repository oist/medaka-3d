"""Render a fixed-crop six-panel C34/v3 video without changing tracking results."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260908_generative_tracking"
sys.path.insert(0, str(LAB))

from benchmark import (  # noqa: E402 - this workspace helper imports the isolated lab
    BASELINE_DIR,
    CALIBRATION_DIR,
    EXPERIMENT,
    open_videos,
    seek_read,
    sha256,
)

COLORS = ((255, 185, 45), (65, 205, 240), (240, 90, 170))
STEP = 4
OUTPUT_FPS = 166 / 16


def fixed_crops(rig, pose, baseline, metadata):
    crops = {}
    for view in VIEWS:
        points = np.concatenate((rig.camera(view).project(pose.reshape(-1, 3)),
                                 rig.camera(view).project(baseline.reshape(-1, 3))))
        points = points[np.isfinite(points).all(axis=1)]
        width, height = int(metadata[view]["width"]), int(metadata[view]["height"])
        if len(points):
            lower = np.maximum(np.floor(points.min(axis=0) - 85).astype(int), 0)
            upper = np.minimum(np.ceil(points.max(axis=0) + 85).astype(int), [width, height])
        else:
            lower, upper = np.array([0, 0]), np.array([width, height])
        if np.any(upper <= lower):
            lower, upper = np.array([0, 0]), np.array([width, height])
        crops[view] = [int(lower[0]), int(lower[1]), int(upper[0]), int(upper[1])]
    return crops


def render_frame(rgb, index, frame, rig, data, baseline, crops):
    rows = []
    for label, candidate in (("C34 lambda 0.767", False), ("3D-driven v3 candidate", True)):
        panels = []
        for view in VIEWS:
            panel = rgb[view].copy()
            for fish, color in enumerate(COLORS):
                if candidate:
                    uv = rig.camera(view).project(data["pose_xyz"][index, fish])
                    for node, pixel in enumerate(uv):
                        if np.isfinite(pixel).all():
                            known = data["observed"][index, fish, node]
                            cv2.circle(panel, tuple(np.rint(pixel).astype(int)),
                                       5 if known else 8, color, -1 if known else 1)
                    for a, b in ((0, 1), (1, 2)):
                        if np.isfinite(uv[[a, b]]).all():
                            cv2.line(panel, tuple(np.rint(uv[a]).astype(int)),
                                     tuple(np.rint(uv[b]).astype(int)), color, 2)
                    pixel = uv[1]
                    suffix = " review" if data["review"][index, fish] else ""
                else:
                    pixel = rig.camera(view).project(baseline[index, fish])
                    suffix = ""
                if np.isfinite(pixel).all():
                    point = tuple(np.rint(pixel).astype(int))
                    cv2.circle(panel, point, 10, color, 2)
                    cv2.putText(panel, f"S{fish}{suffix}", (point[0] + 12, point[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            x0, y0, x1, y1 = crops[view]
            panel = panel[y0:y1, x0:x1]
            # Reserve the header outside the image instead of obscuring fish.
            scale = min(480 / panel.shape[1], 412 / panel.shape[0])
            image = cv2.resize(panel, (round(panel.shape[1] * scale),
                                       round(panel.shape[0] * scale)))
            panel = np.full((480, 480, 3), 15, np.uint8)
            top = 68 + (412 - image.shape[0]) // 2
            left = (480 - image.shape[1]) // 2
            panel[top:top + image.shape[0], left:left + image.shape[1]] = image
            for text, y, size in ((f"{label} | {view}", 20, 0.46),
                                  (f"frame {frame} | 4x slower", 41, 0.48),
                                  ("hollow = prediction; review = unresolved ID", 60, 0.36)):
                cv2.putText(panel, text, (7, y), cv2.FONT_HERSHEY_SIMPLEX,
                            size, (245, 245, 245), 1, cv2.LINE_AA)
            panels.append(panel)
        rows.append(np.concatenate(panels, axis=1))
    return np.concatenate(rows, axis=0)


def decode_qa(path, expected):
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot decode rendered video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        reported = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (960, 1440, 3):
                raise RuntimeError(f"unexpected decoded frame size: {frame.shape}")
            count += 1
        if count != expected or reported != expected or not np.isclose(fps, OUTPUT_FPS, atol=0.01):
            raise RuntimeError(f"rendered video length/fps mismatch: {count}, {reported}, {fps}")
        return {"decoded_frames": count, "reported_frames": reported, "fps": fps,
                "width": 1440, "height": 960, "decode_ok": True}
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {args.out}")
    candidate_path = args.run / "candidate.npz"
    with np.load(candidate_path, allow_pickle=False) as arrays:
        data = {key: arrays[key] for key in ("frame_idx", "pose_xyz", "observed", "review")}
    frames = data["frame_idx"]
    if not len(frames) or not np.all(np.diff(frames) == 1):
        raise ValueError("comparison requires a nonempty contiguous candidate window")
    baseline_path = args.input_root / BASELINE_DIR / f"C34-label__{EXPERIMENT}__lam0.767.npz"
    with np.load(baseline_path, allow_pickle=False) as arrays:
        xyz = arrays["xyz"][frames]
        baseline = np.where((arrays["slot_track"][frames] >= 0)[..., None], xyz, np.nan)
    rig = load_rig(args.input_root / CALIBRATION_DIR)
    sample_indices = np.arange(0, len(frames), STEP)
    args.out.mkdir(parents=True, exist_ok=False)
    with provenance("render_generative_comparison_video", args.out, name="render") as record:
        record.params_from(args)
        record.input("candidate", candidate_path, sha256=sha256(candidate_path))
        record.input("baseline", baseline_path, sha256=sha256(baseline_path))
        code_paths = (Path(__file__), LAB / "benchmark.py", REPO / "src/medaka3d/calib.py")
        code_hashes = {str(path): sha256(path) for path in code_paths}
        record.param("code_hashes", code_hashes)
        code_dir = args.out / "code_snapshot"
        code_dir.mkdir()
        for path in code_paths:
            shutil.copyfile(path, code_dir / path.name)
        _, captures, metadata = open_videos(args.raw_root, EXPERIMENT, int(frames[-1]) + 1)
        writer = None
        video = args.out / "comparison_4x_slow.mp4"
        preview_indices = set(map(int, np.linspace(0, len(sample_indices) - 1,
                                                    min(6, len(sample_indices)))))
        previews = []
        try:
            if any(not np.isclose(meta["fps"], 166) for meta in metadata.values()):
                raise ValueError("the requested 4x playback protocol requires 166 fps sources")
            crops = fixed_crops(rig, data["pose_xyz"], baseline, metadata)
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                                     OUTPUT_FPS, (1440, 960))
            video_enabled = writer.isOpened()
            previous_frame = None
            for sequence, index in enumerate(sample_indices):
                frame = int(frames[index])
                rgb = {}
                for view, capture in captures.items():
                    if previous_frame is None:
                        rgb[view] = seek_read(capture, frame, view)
                    else:
                        for _ in range(frame - previous_frame - 1):
                            if not capture.grab():
                                raise RuntimeError(f"video ended while skipping to {view}/{frame}")
                        ok, image = capture.read()
                        if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
                            raise RuntimeError(f"video frame mismatch at {view}/{frame}")
                        rgb[view] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                previous_frame = frame
                panel = render_frame(rgb, int(index), frame, rig, data, baseline, crops)
                bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
                if video_enabled:
                    writer.write(bgr)
                if sequence in preview_indices:
                    preview = args.out / f"preview_{frame:09d}.jpg"
                    if not cv2.imwrite(str(preview), bgr):
                        raise RuntimeError(f"could not write preview: {preview}")
                    previews.append(preview.name)
                if sequence % 25 == 0:
                    print(f"render {frame}; panel {sequence + 1}/{len(sample_indices)}", flush=True)
        finally:
            if writer is not None:
                writer.release()
            for capture in captures.values():
                capture.release()
        qa = decode_qa(video, len(sample_indices)) if video_enabled else {
            "decode_ok": False, "fallback": "VideoWriter unavailable; six static previews saved"}
        if video_enabled:
            record.output("video", video, sha256=sha256(video))
        record.param("raw_videos", metadata)
        record.param("sampling", {"every_nth_frame": STEP, "output_fps": OUTPUT_FPS,
                                   "slowdown": 4, "source_frames": frames[sample_indices].tolist()})
        result = {
            "experiment": EXPERIMENT, "source_window": [int(frames[0]), int(frames[-1]) + 1],
            "candidate_npz": {"path": str(candidate_path), "sha256": sha256(candidate_path)},
            "baseline_npz": {"path": str(baseline_path), "sha256": sha256(baseline_path)},
            "code_hashes": code_hashes, "source_videos": metadata,
            "crop_rule": "fixed per view over union of both methods across the full window",
            "fixed_crops_xyxy": crops, "source_fps": 166, "sample_step": STEP,
            "output_fps": OUTPUT_FPS, "slowdown": 4,
            "sampled_source_frames": frames[sample_indices].tolist(),
            "video": video.name if video_enabled else None, "previews": previews, "qa": qa,
            "interpretation": "Top C34; bottom v3. Slots share initial names, not verified IDs. "
                              "Hollow nodes are latent; review means unresolved identity.",
        }
        meta_path = args.out / "video_metadata.json"
        meta_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        record.output("metadata", meta_path, sha256=sha256(meta_path))
        html = ("<!doctype html><meta charset='utf-8'><title>4x slower tracking comparison</title>"
                "<style>body{background:#171b21;color:#eee;font-family:sans-serif}"
                "video,img{display:block;width:min(100%,1440px);margin:16px 0}</style>"
                "<h1>4x slower | C34 and v3 comparison</h1><p>Top: C34. Bottom: v3. "
                "Hollow nodes are latent predictions; review means unresolved ID. "
                "The shared crop stays fixed throughout the window. This is not ground truth.</p>")
        if video_enabled:
            html += f'<video controls preload="metadata" src="{video.name}"></video>'
        html += "".join(f'<img src="{file}" alt="{file}">' for file in previews)
        (args.out / "index.html").write_text(html, encoding="utf-8")
        record.output("viewer", args.out / "index.html")
        print(json.dumps({"out": str(args.out), "qa": qa}, indent=2), flush=True)


if __name__ == "__main__":
    main()

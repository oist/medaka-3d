"""Raw three-camera review of latent three-body fits; overlays are not truth."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from observations import SequenceReader

from medaka3d.calib import load_rig
from medaka3d.types import VIEWS

CALIBRATION = (
    "calibration/fixed_refractive_192x192x186/"
    "phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration"
)
COLORS = ((40, 90, 255), (30, 230, 50), (255, 170, 20))  # BGR, arbitrary seeded IDs


def panel(frame, result, t, rig, view, width=600):
    image = frame.images_bgr[view].copy()
    masks = frame.masks[view]
    if masks is not None and len(masks):
        union = np.any(masks, axis=0).astype(np.uint8)
        contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 255, 0), 1)
    pose = result["pose_xyz"][t]
    pixels = rig.camera(view).project(pose)
    for fish in range(3):
        color = COLORS[fish]
        if not np.isfinite(pixels[fish]).all():
            continue
        xy = np.rint(pixels[fish]).astype(np.int32)
        cv2.polylines(image, [xy], False, color, 2, cv2.LINE_AA)
        for node, point in enumerate(xy):
            radius = 5 if node == 1 else 3
            thickness = -1 if result["observed"][t, fish, node] else 1
            cv2.circle(image, tuple(point), radius, color, thickness, cv2.LINE_AA)
        review = " ?" if result["identity_review"][t, fish] else ""
        cv2.putText(image, f"S{fish}{review}", tuple(xy[1] + [8, -8]),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
    resized = cv2.resize(image, (width, width), interpolation=cv2.INTER_AREA)
    canvas = np.full((width + 54, width, 3), 25, np.uint8)
    canvas[54:] = resized
    cv2.putText(canvas, f"{view}  frame {frame.frame_idx}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "cyan: D2; lines: latent pose; ?: ID review", (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, .43, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--frames", type=int, nargs="*")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.tracks, allow_pickle=False) as arrays:
        result = {k: arrays[k] for k in arrays.files}
    frames = result["frame_idx"]
    indexes = {int(f): t for t, f in enumerate(frames)}
    selected = set(args.frames or frames[np.linspace(0, len(frames) - 1, 6, dtype=int)].tolist())
    if not selected.issubset(indexes):
        raise ValueError("requested review frame outside candidate output")
    rig = load_rig(args.input_root / CALIBRATION)
    reader = SequenceReader(args.sequence, args.input_root)
    writer = None
    if args.video:
        writer = cv2.VideoWriter(str(args.output / "three_views.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 20., (1800, 654))
        if not writer.isOpened():
            raise RuntimeError("video writer unavailable")
    count = 0
    try:
        for frame in reader:
            if frame.frame_idx not in indexes:
                continue
            if writer is None and frame.frame_idx not in selected:
                continue
            t = indexes[frame.frame_idx]
            montage = np.concatenate([panel(frame, result, t, rig, v) for v in VIEWS], axis=1)
            if frame.frame_idx in selected:
                path = args.output / f"frame_{frame.frame_idx}.png"
                if not cv2.imwrite(str(path), montage):
                    raise RuntimeError(f"failed to save {path}")
            if writer is not None:
                writer.write(montage)
                count += 1
    finally:
        if writer is not None:
            writer.release()
    if writer is not None and count != len(frames):
        raise RuntimeError("review video lost candidate frames")
    print(f"saved {len(selected)} stills and {count} video frames in {args.output}")


if __name__ == "__main__":
    main()

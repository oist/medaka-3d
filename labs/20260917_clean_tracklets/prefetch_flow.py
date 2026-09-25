"""Prepare immutable image evidence independently for each camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from flow import extract_flow

VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIDEOS = ("cam0_top.mp4", "cam1_side.mp4", "cam2_side.mp4")
EXPERIMENT = "20251225_K_1_2_FCMC"
CASES = ("cg20s", "known_30735", "known_31479", "known_323819", "control_50000", "holdout20s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True, choices=VIEWS)
    parser.add_argument("--output", type=Path, default=Path("work/clean_tracklets_v1"))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    protocol = Path("work/custom2d_comparison_v1/analysis_v3/protocol.json")
    cases = json.loads(protocol.read_text(encoding="utf-8"))["cases"]
    source = Path("work/generative_tracking_inputs/work/c01_s16_predictions/bottomup")
    with np.load(source / f"{EXPERIMENT}__{args.view}.npz", allow_pickle=False) as data:
        frame_idx, points = data["frame_idx"], data["points"]
    video = (
        Path("work/three_body/source_cache/medaka_202512")
        / EXPERIMENT
        / args.view
        / VIDEOS[VIEWS.index(args.view)]
    )
    for name in CASES:
        case = next(c for c in cases if c["name"] == name)
        lo, hi = np.searchsorted(frame_idx, [case["start"] - 332, case["stop"] + 332])
        output = args.output / name
        output.mkdir(parents=True, exist_ok=True)
        print(f"Flow {name} {args.view}: {hi - lo} detection rows", flush=True)
        result = extract_flow(
            video, frame_idx[lo:hi], points[lo:hi], cache_path=output / f"flow_{args.view}.npz"
        )
        print(f"Flow complete {name} {args.view}: {result['diagnostics']}", flush=True)


if __name__ == "__main__":
    main()

"""Bounded actual-image v3 fit timings under explicit, scoped thread limits."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from accelerated_observations import CudaEvidenceBuilder
from benchmark import (
    BACKGROUND_FRAMES,
    BASELINE_DIR,
    CALIBRATION_DIR,
    EXPERIMENT,
    CachedDetections,
    make_backgrounds,
    open_videos,
    seek_read,
    sha256,
    skeleton_from_sources,
)
from observations import EvidenceConfig
from threadpoolctl import threadpool_info, threadpool_limits
from tracker import FrameInput, GenerativeTracker, TrackerConfig

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite thread profile: {args.out}")
    if not 1 <= args.start < args.stop or args.stop - args.start > 16:
        raise ValueError("use 1-16 consecutive frames with a preceding frame")
    args.out.mkdir(parents=True)
    with provenance("profile_fit_threads", args.out, name="profile") as record:
        record.params_from(args)
        code_dir = args.out / "code_snapshot"
        code_dir.mkdir()
        paths = [Path(__file__), Path(__file__).with_name("accelerated_observations.py"),
                 Path(__file__).with_name("observations.py"),
                 Path(__file__).with_name("tracker.py"), Path(__file__).with_name("benchmark.py"),
                 Path(__file__).resolve().parents[2] / "src/medaka3d/calib.py"]
        record.param("code_hashes", {str(path): sha256(path) for path in paths})
        for path in paths:
            shutil.copyfile(path, code_dir / path.name)
        rig = load_rig(args.input_root / CALIBRATION_DIR)
        detections = CachedDetections(args.input_root, EXPERIMENT)
        for view, path in detections.paths.items():
            record.input(view, path, sha256=sha256(path))
        seed_path = args.input_root / BASELINE_DIR / f"C34-label__{EXPERIMENT}__lam0.767.npz"
        with np.load(seed_path, allow_pickle=False) as data:
            source = data["source"][args.start:args.start + 1].copy()
        initial, observed, _ = skeleton_from_sources(rig, detections, [args.start], source)
        if not observed.all():
            raise ValueError("complete seed required")
        record.input("seed", seed_path, sha256=sha256(seed_path))
        frames = np.arange(args.start, args.stop)
        inputs = []
        _, captures, metadata = open_videos(args.raw_root, EXPERIMENT, detections.n_frames)
        record.param("videos", metadata)
        backgrounds_at = [f for f in BACKGROUND_FRAMES
                          if f < args.start - 332 or f >= args.stop + 332]
        try:
            backgrounds = make_backgrounds(captures, backgrounds_at)
            previous = {v: seek_read(cap, args.start - 1, v) for v, cap in captures.items()}
            with CudaEvidenceBuilder(EvidenceConfig(flow_max_side=320)) as builder:
                for frame in frames:
                    rgb = {}
                    for view, cap in captures.items():
                        ok, bgr = cap.read()
                        if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
                            raise RuntimeError(f"unaligned read {view}/{frame}")
                        rgb[view] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    evidence = builder.from_frames(rgb, backgrounds, previous)
                    inputs.append(FrameInput(int(frame), {
                        view: detections.points(view, int(frame)) for view in VIEWS}, evidence))
                    previous = rgb
        finally:
            for cap in captures.values():
                cap.release()
        original_torch, original_opencv = torch.get_num_threads(), cv2.getNumThreads()
        environment = {"pools": threadpool_info(), "torch_threads": original_torch,
                       "opencv_threads": original_opencv}
        variants = (("default", None, original_torch, original_opencv),
                    ("blas1", 1, original_torch, original_opencv),
                    ("blas2", 2, original_torch, original_opencv),
                    ("blas1_torch1_opencv1", 1, 1, 1))
        rows, reference = {}, None
        try:
            for name, blas, torch_threads, opencv_threads in variants:
                torch.set_num_threads(torch_threads)
                cv2.setNumThreads(opencv_threads)
                with threadpool_limits(limits=blas, user_api="blas"):
                    print(f"fit thread profile {name}", flush=True)
                    GenerativeTracker(rig, initial[0], TrackerConfig()).track(inputs)  # warm
                    durations = []
                    profiler = cProfile.Profile()
                    for _ in range(2):
                        started = time.perf_counter()
                        profiler.enable()
                        result = GenerativeTracker(rig, initial[0], TrackerConfig()).track(inputs)
                        profiler.disable()
                        durations.append(time.perf_counter() - started)
                    profile_path = args.out / f"{name}.prof"
                    profiler.dump_stats(str(profile_path))
                    with (args.out / f"{name}_functions.txt").open("w", encoding="utf-8") as stream:
                        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                            "cumulative").print_stats(45)
                    if reference is None:
                        reference = result
                    rows[name] = {
                        "seconds": durations, "median_seconds": float(np.median(durations)),
                        "fps": len(frames) / float(np.median(durations)),
                        "pools_inside_scope": threadpool_info(), "torch_threads": torch_threads,
                        "opencv_threads": opencv_threads,
                        "pose_max_difference_mm": float(np.max(np.linalg.norm(
                            reference.pose_xyz - result.pose_xyz, axis=-1))),
                        "node_source_difference_count": int(
                            (reference.node_source != result.node_source).sum()),
                        "review_difference_count": int((reference.review != result.review).sum()),
                        "fit_failure_frames": result.diagnostics["fitting_limit_or_failure_frames"],
                    }
                    record.output(name, profile_path, sha256=sha256(profile_path))
        finally:
            torch.set_num_threads(original_torch)
            cv2.setNumThreads(original_opencv)
        output = {"frames": [args.start, args.stop], "environment": environment, "variants": rows,
                  "scope": "Same precomputed CUDA320 image observations and unchanged v3 tracker. "
                           "Warm, fit-only timings include cProfile overhead; no raw decoding or "
                           "image preprocessing. Threadpool limits are scoped and restored.",
                  "official_reference": "https://github.com/joblib/threadpoolctl"}
        path = args.out / "thread_profiles.json"
        path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=sha256(path))
        print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()

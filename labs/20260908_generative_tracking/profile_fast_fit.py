"""Save a few actual images and compare frozen/block-Jacobian fits without a profiler."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
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
from fast_fit import FastFitTracker
from observations import EvidenceConfig
from tracker import FrameInput, GenerativeTracker, TrackerConfig

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--saved-images", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite fast-fit comparison: {args.out}")
    if not 1 <= args.start < args.stop or args.stop - args.start > 8:
        raise ValueError("use 1-8 consecutive frames with a preceding frame")
    if (args.raw_root is None) == (args.saved_images is None):
        raise ValueError("provide exactly one of --raw-root and --saved-images")
    args.out.mkdir(parents=True)
    with provenance("profile_fast_fit", args.out, name="profile") as record:
        record.params_from(args)
        code_dir = args.out / "code_snapshot"
        code_dir.mkdir()
        lab = Path(__file__).resolve().parent
        paths = [Path(__file__), lab / "accelerated_observations.py",
                 lab / "observations.py", lab / "tracker.py", lab / "fast_fit.py",
                 lab / "benchmark.py", lab.parents[1] / "src/medaka3d/calib.py"]
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
            raise ValueError("complete first-frame seed required")
        record.input("seed", seed_path, sha256=sha256(seed_path))
        record.param("seed_source", source.tolist())
        frames = np.arange(args.start, args.stop)
        rgb_frames = []
        if args.saved_images:
            with np.load(args.saved_images, allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["frame_idx"], frames)
                backgrounds = {v: arrays[f"background_{v}"] for v in VIEWS}
                previous = {v: arrays[f"previous_{v}"] for v in VIEWS}
                for i in range(len(frames)):
                    rgb_frames.append({v: arrays[f"rgb_{v}"][i] for v in VIEWS})
            record.input("saved_images", args.saved_images, sha256=sha256(args.saved_images))
        else:
            _, captures, metadata = open_videos(args.raw_root, EXPERIMENT, detections.n_frames)
            record.param("videos", metadata)
            backgrounds_at = [f for f in BACKGROUND_FRAMES
                              if f < args.start - 332 or f >= args.stop + 332]
            record.param("background_frames", backgrounds_at)
            try:
                backgrounds = make_backgrounds(captures, backgrounds_at)
                previous = {v: seek_read(cap, args.start - 1, v) for v, cap in captures.items()}
                for frame in frames:
                    rgb = {}
                    for view, cap in captures.items():
                        ok, bgr = cap.read()
                        if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
                            raise RuntimeError(f"unaligned read {view}/{frame}")
                        rgb[view] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb_frames.append(rgb)
            finally:
                for cap in captures.values():
                    cap.release()
            archive = args.out / "raw_frames.npz"
            np.savez_compressed(archive, frame_idx=frames,
                                **{f"background_{v}": backgrounds[v] for v in VIEWS},
                                **{f"previous_{v}": previous[v] for v in VIEWS},
                                **{f"rgb_{v}": np.stack([rgb[v] for rgb in rgb_frames])
                                   for v in VIEWS})
            record.output("saved_images", archive, sha256=sha256(archive))
        inputs = []
        with CudaEvidenceBuilder(EvidenceConfig(flow_max_side=320)) as builder:
            record.param("evidence_backend", builder.metadata)
            for frame, rgb in zip(frames, rgb_frames, strict=True):
                evidence = builder.from_frames(rgb, backgrounds, previous)
                inputs.append(FrameInput(int(frame), {
                    view: detections.points(view, int(frame)) for view in VIEWS}, evidence))
                previous = rgb
        config = TrackerConfig()
        reference = GenerativeTracker(rig, initial[0], config)
        fast = FastFitTracker(rig, initial[0], config)
        fits, jacobians = [], []
        for frame in inputs:
            _, plan = reference._plans(initial[0], frame)[0]
            problem = fast.fit_problem(initial[0], plan, frame, 1)
            flat = problem.initial
            step = np.where(flat + 1e-3 < problem.upper, 1e-3, -1e-3)
            trials = np.broadcast_to(flat, (28, 27)).copy()
            trials[1:] += np.eye(27) * step
            values = problem.residual_batch(trials)
            full_jac = ((values[1:] - values[0]) / step[:, None]).T
            compressed = problem.jacobian(flat)
            jacobians.append(float(np.max(np.abs(full_jac - compressed))))
            old = reference._fit(initial[0], plan, frame, 1)
            new = fast._fit(initial[0], plan, frame, 1)
            fits.append({"pose_max_difference_mm": float(np.max(np.abs(old[0] - new[0]))),
                         "cost_difference": float(abs(old[3] - new[3])),
                         "source_differences": int((old[1] != new[1]).sum()),
                         "observed_differences": int((old[2] != new[2]).sum()),
                         "status_same": old[4:] == new[4:]})
        rows, outputs = {}, {}
        # Warm both paths, then alternate their timed calls to reduce drift bias.
        classes = {"frozen": GenerativeTracker, "compressed": FastFitTracker}
        for cls in classes.values():
            cls(rig, initial[0], config).track(inputs)
        durations = {name: [] for name in classes}
        for _ in range(3):
            for name, cls in classes.items():
                started = time.perf_counter()
                outputs[name] = cls(rig, initial[0], config).track(inputs)
                durations[name].append(time.perf_counter() - started)
        old = outputs["frozen"]
        for name, result in outputs.items():
            rows[name] = {
                "seconds": durations[name], "median_seconds": float(np.median(durations[name])),
                "fit_only_fps": len(frames) / float(np.median(durations[name])),
                "pose_max_difference_mm": float(np.max(np.abs(old.pose_xyz - result.pose_xyz))),
                "source_difference_count": int((old.node_source != result.node_source).sum()),
                "review_difference_count": int((old.review != result.review).sum()),
                "observed_difference_count": int((old.observed != result.observed).sum()),
            }
            output = args.out / f"{name}.npz"
            np.savez_compressed(output, frame_idx=result.frame_idx, pose_xyz=result.pose_xyz,
                                node_source=result.node_source, observed=result.observed,
                                review=result.review, xyz=result.xyz,
                                accepted_xyz=result.accepted_xyz)
            record.output(name, output, sha256=sha256(output))
        summary = {
            "frames": [args.start, args.stop], "variants": rows, "fit_checks": fits,
            "jacobian_max_abs_differences": jacobians,
            "speed_ratio": rows["frozen"]["median_seconds"] / rows["compressed"]["median_seconds"],
            "scope": "Unprofiled warm fit-only wall time, identical precomputed CUDA320 image "
                     "evidence, unchanged 27D joint solver/objective/tolerances. Excludes video "
                     "decode and image preparation. No GT or later baseline poses used.",
        }
        path = args.out / "comparison.json"
        path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=sha256(path))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Local-only saved-image QA for batched projection and compressed fit paths."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from accelerated_observations import CudaEvidenceBuilder
from batched_image_score import BatchedImageTracker, image_point_groups, project_image_groups
from benchmark import (
    BASELINE_DIR,
    CALIBRATION_DIR,
    EXPERIMENT,
    CachedDetections,
    sha256,
    skeleton_from_sources,
)
from fast_fit import FastFitTracker
from observations import EvidenceConfig
from reacquisition import ReacquisitionTracker
from tracker import FrameInput, GenerativeTracker, _body_samples

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--saved-images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timing-context", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    with provenance("profile_batched_image_score", args.out, name="profile") as record:
        record.params_from(args)
        lab = Path(__file__).resolve().parent
        paths = [Path(__file__), lab / "fast_fit.py", lab / "batched_image_score.py",
                 lab / "accelerated_observations.py", lab / "observations.py",
                 lab / "tracker.py", lab / "reacquisition.py", lab / "benchmark.py",
                 lab.parents[1] / "src/medaka3d/calib.py"]
        code = args.out / "code_snapshot"
        code.mkdir()
        record.param("code_hashes", {str(p): sha256(p) for p in paths})
        for path in paths:
            shutil.copyfile(path, code / path.name)
        record.input("saved_images", args.saved_images, sha256=sha256(args.saved_images))
        rig = load_rig(args.input_root / CALIBRATION_DIR)
        detections = CachedDetections(args.input_root, EXPERIMENT)
        with np.load(args.saved_images, allow_pickle=False) as arrays:
            frames = arrays["frame_idx"]
            if not 1 <= len(frames) <= 8:
                raise ValueError("saved archive must contain 1-8 frames")
            backgrounds = {v: arrays[f"background_{v}"] for v in VIEWS}
            previous_rgb = {v: arrays[f"previous_{v}"] for v in VIEWS}
            rgbs = [{v: arrays[f"rgb_{v}"][i] for v in VIEWS} for i in range(len(frames))]
        seed = args.input_root / BASELINE_DIR / f"C34-label__{EXPERIMENT}__lam0.767.npz"
        with np.load(seed, allow_pickle=False) as arrays:
            source = arrays["source"][frames[:1]]
        initial, observed, _ = skeleton_from_sources(rig, detections, frames[:1], source)
        if not observed.all():
            raise ValueError("complete seed required")
        record.input("seed", seed, sha256=sha256(seed))
        record.param("seed_source", source.tolist())
        for view, path in detections.paths.items():
            record.input(view, path, sha256=sha256(path))
        inputs = []
        with CudaEvidenceBuilder(EvidenceConfig(flow_max_side=320)) as builder:
            record.param("backend", builder.metadata)
            for frame, rgb in zip(frames, rgbs, strict=True):
                evidence = builder.from_frames(rgb, backgrounds, previous_rgb)
                inputs.append(FrameInput(int(frame), {
                    view: detections.points(view, int(frame)) for view in VIEWS}, evidence))
                previous_rgb = rgb

        def make(name):
            cls = ReacquisitionTracker if name == "frozen" else BatchedImageTracker
            model = cls(rig, initial[0])
            fitter = FastFitTracker if name == "batched_and_fast_fit" else GenerativeTracker
            # Explicit assignment keeps the contrast fixed if the surrounding
            # research tracker later adopts a different default fitter.
            model.local = fitter(rig, initial[0], model.local.config)
            model.recovery_fit = fitter(rig, initial[0], model.recovery_fit.config)
            return model

        models = {name: make(name) for name in ("frozen", "batched", "batched_and_fast_fit")}
        reference = models["frozen"].track(inputs)
        cases, checks = [], []
        previous_pose, templates = initial[0], {}
        for i, frame in enumerate(inputs):
            pose, observed = reference.pose_xyz[i], reference.observed[i]
            cases.append((frame, pose, previous_pose, templates, observed))
            old = models["frozen"]._image_score(*cases[-1])
            new = models["batched"]._image_score(*cases[-1])
            points, shape, sizes = image_point_groups(pose, previous_pose)
            errors = []
            for camera in rig.cameras:
                expected = (camera.project(pose), camera.project(_body_samples(pose)),
                            camera.project(previous_pose))
                actual = project_image_groups(camera, points, shape, sizes)
                for original, grouped in zip(expected, actual, strict=True):
                    np.testing.assert_array_equal(np.isnan(original), np.isnan(grouped))
                    errors.append(float(np.nanmax(np.abs(original - grouped))))
            assert old[1].keys() == new[1].keys()
            template_error = max((float(np.max(np.abs(old[1][k] - new[1][k])))
                                  for k in old[1]), default=0.)
            checks.append({"projection_max_difference_px": max(errors),
                           "score_difference": abs(old[0] - new[0]),
                           "template_max_difference": template_error,
                           "feature_max_difference": float(np.max(np.abs(old[2] - new[2]))),
                           "quality_max_difference": float(np.max(np.abs(old[3] - new[3])))})
            templates, previous_pose = old[1], pose
        score_times = {name: [] for name in ("frozen", "batched")}
        for _ in range(3):
            for name in score_times:
                started = time.perf_counter()
                for _ in range(20):
                    for case in cases:
                        models[name]._image_score(*case)
                score_times[name].append(time.perf_counter() - started)
        for name in models:
            make(name).track(inputs)
        track_times, outputs = {name: [] for name in models}, {}
        for _ in range(3):
            for name in models:
                started = time.perf_counter()
                outputs[name] = make(name).track(inputs)
                track_times[name].append(time.perf_counter() - started)
        rows = {}
        for name, result in outputs.items():
            rows[name] = {
                "seconds": track_times[name],
                "median_seconds": float(np.median(track_times[name])),
                "fit_only_fps": len(frames) / float(np.median(track_times[name])),
                "pose_max_difference_mm": float(np.max(np.abs(
                    reference.pose_xyz - result.pose_xyz))),
                "source_difference_count": int((reference.node_source != result.node_source).sum()),
                "review_difference_count": int((reference.review != result.review).sum()),
                "appearance_difference": float(np.max(np.abs(
                    reference.appearance - result.appearance))),
                "appearance_quality_difference": float(np.max(np.abs(
                    reference.appearance_quality - result.appearance_quality))),
            }
            output = args.out / f"{name}.npz"
            np.savez_compressed(output, frame_idx=result.frame_idx, pose_xyz=result.pose_xyz,
                                node_source=result.node_source, observed=result.observed,
                                review=result.review, appearance=result.appearance,
                                appearance_quality=result.appearance_quality)
            record.output(name, output, sha256=sha256(output))
        summary = {"frames": [int(frames[0]), int(frames[-1]) + 1],
                   "timing_context": args.timing_context,
                   "projection_and_image_score_checks": checks,
                   "image_score_seconds_per_60_calls": score_times,
                   "tracking": rows,
                   "scope": "Unprofiled warm wall time using identical saved images and "
                            "precomputed CUDA320 observations. Three interleaved repetitions. "
                            "Tracking times exclude image preprocessing and video decode."}
        path = args.out / "comparison.json"
        path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=sha256(path))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

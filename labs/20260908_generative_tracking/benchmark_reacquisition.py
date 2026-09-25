"""Run a preregistered reacquisition window with exactly one initialization.

The previous benchmark remains frozen. This command streams native-rate raw
images and untracked detections into one ReacquisitionTracker.track invocation.
Baseline trajectories and female labels are evaluated only after inference.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path, PurePosixPath

import numpy as np

from medaka3d.calib import load_rig
from medaka3d.eval.generative import (
    match_mouth_anchors,
    mouth_reprojection_summary,
    observed_motion_summary,
    score_female_identity,
    skeleton_shape_summary,
)
from medaka3d.eval.reacquisition import (
    assess_availability,
    availability_episodes,
    nominal_tank_summary,
)
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import (
    BASELINE_DIR,
    CALIBRATION_DIR,
    EXPERIMENT,
    CachedDetections,
    make_backgrounds,
    open_videos,
    seek_read,
    sha256,
    skeleton_from_sources,
    write_previews,
)
from observations import EvidenceConfig
from ordered_prefetch import ordered_prefetch
from stage_inputs import RELATIVE_INPUTS
from tracker import FrameInput

BACKGROUND_FRAMES = (20000, 40000, 60000, 80000, 100000, 160000, 220000, 280000, 340000)


def load_registered_window(path: Path, name: str) -> tuple[dict, dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["experiment"] != EXPERIMENT or manifest["fps"] != 166:
        raise ValueError("registration has the wrong experiment or frame rate")
    matches = [window for window in manifest["windows"] if window["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"window must identify exactly one preregistered interval: {name}")
    window = matches[0]
    if not 0 <= window["start"] < window["stop"]:
        raise ValueError("invalid registered interval")
    if window["stop"] - window["start"] != window["n_frames"]:
        raise ValueError("registered frame count is inconsistent")
    if name == "continuous_one_minute" and window["n_frames"] != 9960:
        raise ValueError("continuous one minute must contain exactly 9960 native-rate frames")
    return manifest, window


def verify_inputs(input_root: Path) -> Path:
    root = input_root.resolve(strict=True)
    manifest_path = root / "staging_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    found = set()
    for entry in manifest["files"]:
        relative = PurePosixPath(entry["relative"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
            raise ValueError(f"unsafe input manifest path: {relative}")
        path = root.joinpath(*relative.parts).resolve(strict=True)
        if not path.is_relative_to(root) or relative.as_posix() in found:
            raise ValueError(f"duplicate or escaped input manifest path: {relative}")
        found.add(relative.as_posix())
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"input checksum or size mismatch: {relative}")
    if not set(RELATIVE_INPUTS).issubset(found):
        raise ValueError("input manifest does not cover the fixed comparison bundle")
    return manifest_path


def candidate_metrics(result, frames, initial, anchors, rig):
    """Use the existing shared ID score and separate new availability diagnostics."""
    raw = score_female_identity(
        frames, result.node_source[:, :, :, 0], result.observed[:, :, 0], anchors)
    accepted = score_female_identity(
        frames, result.node_source[:, :, :, 0], result.observed[:, :, 0], anchors,
        review=result.review)
    return {
        "motion": observed_motion_summary(result.xyz, frames),
        "accepted_motion": observed_motion_summary(result.accepted_xyz, frames),
        "observed_episodes": availability_episodes(result.xyz, frames),
        "accepted_episodes": availability_episodes(result.accepted_xyz, frames),
        "availability_assessment": assess_availability(result.accepted_xyz, frames),
        "skeleton": skeleton_shape_summary(result.pose_xyz, result.observed, initial),
        "nominal_tank": nominal_tank_summary(result.pose_xyz, result.observed),
        "accepted_nominal_tank": nominal_tank_summary(
            result.pose_xyz, result.observed & ~result.review[..., None]),
        "female_identity_raw": raw, "female_identity_accepted": accepted,
        "mouth_reprojection": mouth_reprojection_summary(
            result.pose_xyz, frames, anchors, raw,
            lambda view, xyz: rig.camera(view).project(xyz)),
        "review_animal_frames": int(result.review.sum()),
        "review_fraction": float(result.review.mean()),
        "accepted_observed_animal_frames": int(
            np.isfinite(result.accepted_xyz).all(axis=-1).sum()),
        "diagnostics": result.diagnostics,
    }


def baseline_metrics(input_root, frames, rig, detections, anchors, seed_path):
    arms, preview_xyz = {}, None
    paths = sorted((input_root / BASELINE_DIR).glob(f"C34-label__{EXPERIMENT}__lam*.npz"))
    if {path.stem.split("__lam")[-1] for path in paths} != {
        "0", "0.1", "0.25", "0.5", "0.767", "2", "8"
    }:
        raise ValueError("the comparison requires all seven fixed baseline arms")
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            source = arrays["source"][frames]
            accepted = arrays["slot_track"][frames] >= 0
            xyz = np.where(accepted[..., None], arrays["xyz"][frames], np.nan)
        pose, observed, _ = skeleton_from_sources(rig, detections, frames, source)
        observed &= accepted[..., None]
        identity = score_female_identity(frames, source, observed[:, :, 0], anchors)
        arms[path.stem.split("__lam")[-1]] = {
            "path": str(path), "sha256": sha256(path),
            "motion": observed_motion_summary(xyz, frames),
            "accepted_episodes": availability_episodes(xyz, frames),
            "female_identity": identity,
            "skeleton": skeleton_shape_summary(pose, observed, pose[0]),
            "nominal_tank": nominal_tank_summary(pose, observed),
            "mouth_reprojection": mouth_reprojection_summary(
                pose, frames, anchors, identity,
                lambda view, xyz: rig.camera(view).project(xyz)),
        }
        if path.resolve() == seed_path.resolve():
            preview_xyz = xyz
    if preview_xyz is None:
        raise ValueError("seed arm was not among the fixed baselines")
    return arms, preview_xyz


def main(argv=None):
    from accelerated_observations import CudaEvidenceBuilder
    from reacquisition import ReacquisitionTracker, RecoveryConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--floor-policy", choices=("nominal", "tolerance6"), default="nominal",
                        help="Registered floor-gauge sensitivity; same algorithm, explicit bounds")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(f"refusing to overwrite an existing output: {args.out}")
    lab = Path(__file__).resolve().parent
    repo = lab.parents[1]
    registration_path = lab / "reacquisition_windows.json"
    registration, window = load_registered_window(registration_path, args.window)
    staged = verify_inputs(args.input_root)
    args.out.mkdir(parents=True, exist_ok=False)
    with provenance("reacquisition_registered_window", args.out, name="benchmark") as record:
        for key, value in vars(args).items():
            record.param(key, str(value) if isinstance(value, Path) else value)
        record.input("staged_manifest", staged, sha256=sha256(staged))
        record.input("registration", registration_path, sha256=sha256(registration_path))
        record.param("registered_window", window)
        code = [
            Path(__file__), lab / "reacquisition.py", lab / "tracker.py",
            lab / "fast_fit.py",
            lab / "observations.py", lab / "accelerated_observations.py",
            lab / "ordered_prefetch.py",
            lab / "benchmark.py", lab / "stage_inputs.py",
            repo / "src/medaka3d/calib.py", repo / "src/medaka3d/types.py",
            repo / "src/medaka3d/provenance.py", repo / "src/medaka3d/eval/generative.py",
            repo / "src/medaka3d/eval/reacquisition.py", registration_path,
            lab / "REACQUISITION_PLAN.md",
        ]
        hashes = {str(path): sha256(path) for path in code}
        record.param("code_hashes", hashes)
        for path in code:
            target = args.out / "code_snapshot" / path.relative_to(repo)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        record.param("package_versions", {
            package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "opencv-python", "torch")
        })
        rig = load_rig(args.input_root / CALIBRATION_DIR)
        detections = CachedDetections(args.input_root, EXPERIMENT)
        frames = np.arange(window["start"], window["stop"], dtype=np.int64)
        if frames[-1] >= detections.n_frames:
            raise ValueError("registered interval extends past detection data")
        seed_path = args.input_root / registration["seed_baseline_relative"]
        if sha256(seed_path) != registration["seed_baseline_sha256"]:
            raise ValueError("seed baseline differs from registration")
        with np.load(seed_path, allow_pickle=False) as arrays:
            source = arrays["source"][frames[:1]].copy()
            seed_accepted = arrays["slot_track"][frames[:1]] >= 0
        initial, initial_observed, _ = skeleton_from_sources(rig, detections, frames[:1], source)
        if not initial_observed.all() or not seed_accepted.all():
            raise ValueError("registered seed lacks complete accepted observations")
        np.testing.assert_array_equal(source[0], window["seed_source"])
        np.testing.assert_array_equal(initial[0], window["seed_pose_xyz"])
        record.param("seed", {
            "frame": int(frames[0]), "baseline": str(seed_path),
            "sha256": registration["seed_baseline_sha256"], "source": source.tolist(),
            "pose_xyz": initial[0].tolist(), "initialization_count": 1,
            "later_baseline_predictions_used": False,
            "optimistic": "first seed came from the whole-recording baseline",
        })
        config = RecoveryConfig()
        if args.floor_policy == "tolerance6":
            config = dataclasses.replace(config, local=dataclasses.replace(
                config.local, lower_mm=(0.0, 0.0, -6.0)))
        evidence_config = EvidenceConfig(flow_max_side=320)
        record.param("tracker_config", dataclasses.asdict(config))
        record.param("evidence_config", dataclasses.asdict(evidence_config))
        record.param("prefetch", {"producer_threads": 1, "capacity": 2,
                                  "frame_order": "preserved; no frame subsampling"})
        backgrounds_at = [frame for frame in BACKGROUND_FRAMES
                          if frame < frames[0] - 332 or frame >= frames[-1] + 1 + 332]
        record.param("background_frames_used", backgrounds_at)
        record.param("background_rule",
                     "fixed offline samples excluding target +/-332 frames; no GT")
        _, captures, video_metadata = open_videos(args.raw_root, EXPERIMENT, detections.n_frames)
        record.param("videos", video_metadata)
        snapshots = {}
        preview_at = set(map(int, np.linspace(frames[0], frames[-1], 6)))
        if window["event"]:
            first, last = window["event"]
            preview_at.update(f for f in (first - 1, first, last, last + 1)
                              if frames[0] <= f <= frames[-1])
        preparation_started = time.perf_counter()
        progress_path = args.out / "progress.jsonl"
        try:
            backgrounds = make_backgrounds(captures, backgrounds_at)
            previous = {view: seek_read(cap, int(frames[0]) - 1, view)
                        for view, cap in captures.items()}
            with CudaEvidenceBuilder(config=evidence_config, workers=3) as builder:
                record.param("image_backend", builder.metadata)
                preparation_seconds = time.perf_counter() - preparation_started
                tracking_started = time.perf_counter()
                with progress_path.open("x", encoding="utf-8", buffering=1) as progress:
                    def inputs():
                        nonlocal previous
                        import cv2
                        for frame in frames:
                            rgb = {}
                            for view, capture in captures.items():
                                ok, bgr = capture.read()
                                position = round(capture.get(cv2.CAP_PROP_POS_FRAMES))
                                if not ok or position != frame + 1:
                                    raise RuntimeError(
                                        f"sequential video alignment failed: {view}/{frame}")
                                rgb[view] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            evidence = builder.from_frames(
                                rgb, backgrounds=backgrounds, previous_frames=previous)
                            previous = rgb
                            if int(frame) in preview_at:
                                snapshots[int(frame)] = {
                                    view: image.copy() for view, image in rgb.items()}
                            yield FrameInput(int(frame), {
                                view: detections.points(view, int(frame)) for view in VIEWS
                            }, evidence)
                    with ordered_prefetch(inputs(), capacity=2) as prepared:
                        def tracked_inputs():
                            for index, frame_input in enumerate(prepared):
                                yield frame_input
                                if index % 50 == 0 or index + 1 == len(frames):
                                    update = {
                                        "frame": frame_input.frame_idx, "processed": index + 1,
                                        "requested": len(frames),
                                        "elapsed_seconds": time.perf_counter() - tracking_started,
                                    }
                                    progress.write(json.dumps(update) + "\n")
                                    print(json.dumps(update), flush=True)
                        # One object and one call for the entire requested interval.
                        result = ReacquisitionTracker(rig, initial[0], config=config).track(
                            tracked_inputs())
                tracking_seconds = time.perf_counter() - tracking_started
                record.param("image_backend_timings", builder.timings)
        finally:
            for capture in captures.values():
                capture.release()
        np.testing.assert_array_equal(result.frame_idx, frames)
        np.testing.assert_array_equal(result.accepted_xyz,
                                      np.where(result.review[..., None], np.nan, result.xyz))
        candidate_path = args.out / "candidate.npz"
        np.savez_compressed(
            candidate_path, frame_idx=result.frame_idx, pose_xyz=result.pose_xyz,
            xyz=result.xyz, accepted_xyz=result.accepted_xyz, observed=result.observed,
            node_source=result.node_source, review=result.review, margin=result.margin,
            identity_margin=result.identity_margin,
            appearance=result.appearance, appearance_quality=result.appearance_quality,
            geometry_review=result.geometry_review, identity_review=result.identity_review,
        )
        record.output("candidate", candidate_path, sha256=sha256(candidate_path))
        # Human labels and baseline predictions after the first frame enter only here.
        gt_path = args.input_root / registration["female_annotations_relative"]
        if sha256(gt_path) != registration["female_annotations_sha256"]:
            raise ValueError("female evaluation labels differ from registration")
        with gt_path.open(encoding="utf-8-sig", newline="") as stream:
            anchors = match_mouth_anchors(csv.DictReader(stream), detections.mouth,
                                         int(frames[0]), int(frames[-1]) + 1)
        candidate = candidate_metrics(result, frames, initial[0], anchors, rig)
        baselines, baseline_xyz = baseline_metrics(
            args.input_root, frames, rig, detections, anchors, seed_path)
        previews = write_previews(args.out, snapshots, result, baseline_xyz, rig)
        unchanged = all(sha256(path) == hashes[str(path)] for path in code)
        if not unchanged:
            raise RuntimeError("source changed during inference; preserve this run as invalid")
        summary = {
            "protocol": "registered_reacquisition_single_initialization",
            "experiment": EXPERIMENT, "n_groups": 1, "window": window["name"],
            "frames": [int(frames[0]), int(frames[-1]) + 1],
            "technical_completion": True, "initialization_count": 1,
            "requested_frames": len(frames), "returned_frames": len(result.frame_idx),
            "preparation_seconds": preparation_seconds, "tracking_seconds": tracking_seconds,
            "tracking_fps": len(frames) / tracking_seconds,
            "candidate": candidate, "baseline_arms": baselines, "previews": previews,
            "code_unchanged_during_run": unchanged,
            "limitations": [
                "Technical completion and availability are not identity validation.",
                "A single optimistic first-frame seed came from the full-recording baseline.",
                "No baseline predictions or GT labels after initialization enter inference.",
                "Female endpoint marks cannot validate male identities or count returning swaps.",
                "The minute includes a known development failure and runs regardless "
                "of short-clip quality.",
                "Only candidate wall time is measured; frozen baseline compute time is unknown.",
            ],
        }
        comparison = args.out / "comparison.json"
        comparison.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("comparison", comparison, sha256=sha256(comparison))
        record.output("preview", args.out / "preview.html")
        record.output("progress", progress_path)
        record.param("code_unchanged_during_run", unchanged)
        # Keep validation outcomes sealed until the independent postpass is frozen.
        print(json.dumps({"out": str(args.out), "tracking_seconds": tracking_seconds,
                          "technical_completion": True, "returned_frames": len(frames)},
                         indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

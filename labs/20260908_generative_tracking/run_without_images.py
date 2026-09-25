"""Image-evidence ablation of the frozen generative tracker, using local inputs.

This separate command changes only FrameInput.evidence to None. The seed,
detections, calibration, TrackerConfig and shared evaluation are unchanged.
It does not replace comparison with the seven existing C34 baseline arms.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
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
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import (
    BASELINE_DIR,
    CALIBRATION_DIR,
    EXPERIMENT,
    CachedDetections,
    sha256,
    skeleton_from_sources,
)
from stage_inputs import RELATIVE_INPUTS
from tracker import FrameInput, GenerativeTracker, TrackerConfig


def run_without_images(input_root: Path, start: int, stop: int, out: Path) -> dict:
    """Run one local window with no masks, appearance observations or optical flow."""
    input_root, out = Path(input_root).resolve(), Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing ablation output: {out}")
    if not 0 <= start < stop:
        raise ValueError("require 0 <= start < stop")
    preparation_started = time.perf_counter()
    staged = input_root / "staging_manifest.json"
    manifest = json.loads(staged.read_text(encoding="utf-8-sig"))
    relatives = set()
    for entry in manifest["files"]:
        # Accept the original Windows manifest and portable staging CLI manifests.
        relative = PurePosixPath(entry["relative"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
            raise ValueError(f"unsafe staged input path: {relative}")
        path = input_root.joinpath(*relative.parts).resolve(strict=True)
        if not path.is_relative_to(input_root):
            raise ValueError(f"staged input escapes input root: {relative}")
        if relative.as_posix() in relatives:
            raise ValueError(f"duplicate staged input: {relative}")
        relatives.add(relative.as_posix())
        if sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"staging checksum or size mismatch: {path}")
    if not set(RELATIVE_INPUTS).issubset(relatives):
        raise ValueError("staging manifest does not contain all fixed comparison inputs")

    out.mkdir(parents=True, exist_ok=False)
    with provenance("generative_ablation_without_images", out, name="ablation") as record:
        record.param("input_root", str(input_root))
        record.param("experiment", EXPERIMENT)
        record.param("frames", [start, stop])
        record.param("evidence", None)
        record.param("algorithm_change",
                     "FrameInput.evidence=None; default TrackerConfig unchanged")
        record.input("staged_manifest", staged, sha256=sha256(staged))
        code_paths = (
            Path(__file__), Path(__file__).with_name("benchmark.py"),
            Path(__file__).with_name("tracker.py"), Path(__file__).with_name("observations.py"),
            Path(__file__).with_name("stage_inputs.py"),
            Path(__file__).resolve().parents[2] / "src/medaka3d/calib.py",
            Path(__file__).resolve().parents[2] / "src/medaka3d/eval/generative.py",
            Path(__file__).resolve().parents[2] / "src/medaka3d/provenance.py",
            Path(__file__).resolve().parents[2] / "src/medaka3d/types.py",
        )
        code_hashes = {str(path): sha256(path) for path in code_paths}
        record.param("code_hashes", code_hashes)
        code_dir = out / "code_snapshot"
        code_dir.mkdir()
        for path in code_paths:
            shutil.copyfile(path, code_dir / path.name)
        rig = load_rig(input_root / CALIBRATION_DIR)
        detections = CachedDetections(input_root, EXPERIMENT)
        if stop > detections.n_frames:
            raise ValueError("requested stop extends past the detection cache")
        frames = np.arange(start, stop, dtype=np.int64)
        seed_path = input_root / BASELINE_DIR / f"C34-label__{EXPERIMENT}__lam0.767.npz"
        with np.load(seed_path, allow_pickle=False) as arrays:
            # Copy one frame, so the complete baseline array is not retained.
            seed_source = arrays["source"][start:start + 1].copy()
        initial, initial_observed, _ = skeleton_from_sources(
            rig, detections, frames[:1], seed_source)
        if not initial_observed.all():
            raise ValueError("the initial frame lacks a complete three-animal skeleton")
        config = TrackerConfig()
        record.param("tracker_config", dataclasses.asdict(config))
        record.param("seed", {
            "frame": start, "baseline": str(seed_path), "sha256": sha256(seed_path),
            "source": seed_source.tolist(), "pose_xyz": initial[0].tolist(),
            "uses_future_baseline": False, "optimistic": "baseline seed used the whole recording",
        })
        preparation_seconds = time.perf_counter() - preparation_started

        def inputs():
            for frame in frames:
                if (frame - start) % 50 == 0:
                    print(f"no-image tracking {int(frame)} / {stop}", flush=True)
                yield FrameInput(
                    frame_idx=int(frame),
                    detections={view: detections.points(view, int(frame)) for view in VIEWS},
                    evidence=None,
                )

        tracking_started = time.perf_counter()
        result = GenerativeTracker(rig, initial[0], config=config).track(inputs())
        tracking_seconds = time.perf_counter() - tracking_started
        if not np.array_equal(result.frame_idx, frames):
            raise ValueError("tracker output frames differ from the requested window")
        if not np.isnan(result.accepted_xyz[result.review]).all():
            raise ValueError("reviewed identities leaked into accepted coordinates")
        candidate_path = out / "candidate.npz"
        np.savez_compressed(
            candidate_path, frame_idx=result.frame_idx, pose_xyz=result.pose_xyz, xyz=result.xyz,
            accepted_xyz=result.accepted_xyz, observed=result.observed,
            node_source=result.node_source, review=result.review,
            margin=result.margin, identity_margin=result.identity_margin,
        )
        record.output("candidate", candidate_path, sha256=sha256(candidate_path))

        # Ground-truth labels are opened only after inference and output are complete.
        gt_path = input_root / "assets/groundtruth/fish_sex_gt" / EXPERIMENT / "female_sex_gt.csv"
        with gt_path.open(encoding="utf-8-sig", newline="") as stream:
            anchors = match_mouth_anchors(csv.DictReader(stream), detections.mouth, start, stop)
        raw_score = score_female_identity(
            frames, result.node_source[:, :, :, 0], result.observed[:, :, 0], anchors)
        accepted_score = score_female_identity(
            frames, result.node_source[:, :, :, 0], result.observed[:, :, 0], anchors,
            review=result.review)

        def project(view, xyz):
            return rig.camera(view).project(xyz)

        candidate = {
            "motion": observed_motion_summary(result.xyz, frames),
            "accepted_motion": observed_motion_summary(result.accepted_xyz, frames),
            "skeleton": skeleton_shape_summary(result.pose_xyz, result.observed, initial[0]),
            "female_identity_raw": raw_score, "female_identity_accepted": accepted_score,
            "mouth_reprojection": mouth_reprojection_summary(
                result.pose_xyz, frames, anchors, raw_score, project),
            "review_animal_frames": int(result.review.sum()),
            "review_fraction": float(result.review.mean()),
            "accepted_observed_animal_frames": int(
                np.isfinite(result.accepted_xyz).all(axis=-1).sum()),
            "beam_margin": "retained hypotheses only; not a calibrated probability",
            "diagnostics": result.diagnostics,
        }
        summary = {
            "protocol": "window_initialized_recovery_without_images",
            "experiment": EXPERIMENT, "n_groups": 1, "frames": [start, stop],
            "preparation_seconds": preparation_seconds, "tracking_seconds": tracking_seconds,
            "tracking_fps": len(frames) / tracking_seconds,
            "candidate": candidate, "baseline_arms": {}, "previews": [],
            "limitations": [
                "Ablation only: image evidence removed from the same default tracker and seed.",
                "Not a replacement for comparison against the seven frozen C34 baseline arms.",
                "Fit-only timing excludes video decoding and image preprocessing; not a fair "
                "speed comparison against the image-enabled command.",
                "First frame initialized from the whole-recording baseline; optimistic local test.",
                "No baseline predictions after initialization enter inference; GT labels are "
                "used only after tracking.",
                "Female endpoint intervals cannot count transient swaps or validate male IDs.",
                "Known failure windows are development diagnostics, not held-out generalization.",
                "Low jumps, residuals or bounded skeleton lengths do not establish correct IDs.",
            ],
        }
        summary_path = out / "comparison.json"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("comparison", summary_path, sha256=sha256(summary_path))
        record.param("code_unchanged_during_run",
                     all(sha256(path) == code_hashes[str(path)] for path in code_paths))
        print(json.dumps({"out": str(out), "tracking_seconds": tracking_seconds,
                          "candidate": candidate}, indent=2), flush=True)
        return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run_without_images(args.input_root, args.start, args.stop, args.out)


if __name__ == "__main__":
    main()

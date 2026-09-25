"""Run one continuous three-body experiment, then evaluate seven frozen baselines.

Only the previously registered first pose initializes tracking. Later baseline
arrays and human female labels are opened after the single tracker call returns.
The detector masks are precomputed independently; reported tracking time does
not include their preparation. A prefix smoke test never becomes a full-window
or identity certification. No other experimental lab is imported.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import shutil
import sys
import time
from itertools import islice
from pathlib import Path, PurePosixPath

import numpy as np

from medaka3d.calib import load_rig, triangulate
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

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
sys.path.insert(0, str(LAB))
from observations import SequenceReader  # noqa: E402

EXPERIMENT = "20251225_K_1_2_FCMC"
BASELINE_DIR = f"work/c34_microflip/{EXPERIMENT}__chain-c28-c33-c30v4-c34"
BASELINE_LAMBDAS = ("0", "0.1", "0.25", "0.5", "0.767", "2", "8")
CALIBRATION_DIR = (
    "calibration/fixed_refractive_192x192x186/"
    "phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration"
)
REGISTRATION = REPO / "labs/20260908_generative_tracking/reacquisition_windows.json"
REGISTRATION_SHA256 = "ea752603a8088b461bbca0fcf6f310ea181ab10769f3694bc7510a4635d2822b"
CODE_ORIGIN = {
    "labs/20260908_generative_tracking/benchmark.py": {
        "sha256": "a119bdbe4a3401c0510e07fbeb306472664e27cd0e167a822b448d5caa499d9a",
        "reused": "fixed asset paths and source-selected skeleton triangulation, adapted in place",
    },
    "labs/20260908_generative_tracking/benchmark_reacquisition.py": {
        "sha256": "72affc8d1a276a8d52775440aa205ba1c56ffd71f8ae8b6320ed77b89c0eb341",
        "reused": "shared evaluator calls and one-initialization provenance pattern",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value):
    value = str(value).replace("\\", "/")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or ":" in value or not relative.parts:
        raise ValueError(f"unsafe relative input path: {value}")
    return relative


def resolve_input(root, relative):
    path = (Path(root) / safe_relative(relative)).resolve(strict=True)
    if not path.is_relative_to(Path(root).resolve(strict=True)) or not path.is_file():
        raise ValueError(f"input escaped its root or is not a file: {relative}")
    return path


def staging_catalog(input_root):
    """Read manifest metadata only; do not open baseline arrays or human labels."""
    path = Path(input_root) / "staging_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = {}
    for entry in manifest["files"]:
        key = safe_relative(entry["relative"]).as_posix()
        if key in entries:
            raise ValueError(f"duplicate staging entry: {key}")
        entries[key] = entry
    return entries, path


def verified_input(root, relative, catalog):
    key = safe_relative(relative).as_posix()
    if key not in catalog:
        raise ValueError(f"required file is absent from staging manifest: {key}")
    entry = catalog[key]
    path = resolve_input(root, key)
    if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
        raise ValueError(f"staged file differs from its checksum or size: {key}")
    return path


def sequence_frames(sequence_manifest, max_frames=None):
    if sequence_manifest["experiment"] != EXPERIMENT or sequence_manifest["fps"] != 166:
        raise ValueError("sequence experiment or native frame rate is wrong")
    values = sequence_manifest["frames"]
    if not values or any(type(value) is not int for value in values):
        raise ValueError("sequence frames must be nonempty integers")
    frames = np.asarray(values, dtype=np.int64)
    if np.any(np.diff(frames) != 1):
        raise ValueError("sequence must contain every consecutive native-rate frame")
    if frames[0] not in (76620, 333519) or frames[-1] >= 348600:
        raise ValueError("sequence has no authorized registered first pose")
    if frames[0] == 333519 and frames[-1] >= 333719:
        raise ValueError("transient sequence extends beyond the registered development interval")
    if max_frames is not None:
        if max_frames < 1 or max_frames > len(frames):
            raise ValueError(
                "--max-frames must select a nonempty prefix within the staged sequence"
            )
        frames = frames[:max_frames]
    return frames


def registered_seed(first_frame):
    """Use the frozen JSON pose, never reopen a baseline trajectory for seeding."""
    if sha256(REGISTRATION) != REGISTRATION_SHA256:
        raise ValueError("the historical seed registration has changed")
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    name = "development_interval" if first_frame == 76620 else "development_transient"
    entries = [item for item in registration["windows"] if item["name"] == name]
    if len(entries) != 1 or entries[0]["start"] != first_frame:
        raise ValueError("source frame does not match a registered development seed")
    source = entries[0]
    pose = np.asarray(source["seed_pose_xyz"], dtype=np.float64)
    if pose.shape != (3, 3, 3) or not np.isfinite(pose).all():
        raise ValueError("registered initial pose is incomplete")
    seed = {
        "frame": int(first_frame),
        "pose_xyz": pose.tolist(),
        "source": source["seed_source"],
        "registration_path": str(REGISTRATION),
        "registration_sha256": REGISTRATION_SHA256,
        "registration_window": name,
        "baseline_relative": registration["seed_baseline_relative"],
        "baseline_sha256": registration["seed_baseline_sha256"],
        "initialization_count": 1,
        "later_baseline_input": False,
        "limitation": "optimistic first pose from the offline whole-recording baseline",
    }
    evaluation = {
        "female_annotations_relative": registration["female_annotations_relative"],
        "female_annotations_sha256": registration["female_annotations_sha256"],
    }
    return pose, seed, evaluation


def skeleton_from_sources(rig, nodes, frames, source):
    """Reconstruct already-selected baseline source rows; no relinking or scoring."""
    uv = np.full((len(frames), 3, 3, 3, 2), np.nan)
    for vi, view in enumerate(VIEWS):
        for t, frame in enumerate(frames):
            points = nodes.at(view, int(frame))
            for fish in range(3):
                row = int(source[t, fish, vi])
                if row >= 0:
                    if row >= len(points):
                        raise ValueError(f"baseline source row out of range: {view}/{frame}/{row}")
                    uv[t, fish, vi] = points[row]
    rays = [rig.camera(view).rays(uv[:, :, vi]) for vi, view in enumerate(VIEWS)]
    valid = np.stack(
        [np.isfinite(o).all(axis=-1) & np.isfinite(d).all(axis=-1) for o, d in rays], axis=-1
    )
    pose, _ = triangulate(
        np.stack([o for o, _ in rays], axis=-2),
        np.stack([d for _, d in rays], axis=-2),
        valid=valid,
    )
    return pose, (valid.sum(axis=-1) >= 2) & np.isfinite(pose).all(axis=-1)


def validate_result(result, frames, nodes):
    required = {
        "frame_idx": (len(frames),),
        "pose_xyz": (len(frames), 3, 3, 3),
        "xyz": (len(frames), 3, 3),
        "accepted_xyz": (len(frames), 3, 3),
        "observed": (len(frames), 3, 3),
        "sources": (len(frames), 3, 3, 3),
        "identity_review": (len(frames), 3),
        "mask_supported": (len(frames), 3),
        "costs": (len(frames),),
        "score_margin": (len(frames),),
    }
    for key, shape in required.items():
        if key not in result or np.asarray(result[key]).shape != shape:
            raise ValueError(
                f"tracker result has missing or wrong shape for {key}: expected {shape}"
            )
    np.testing.assert_array_equal(result["frame_idx"], frames)
    for name in ("observed", "identity_review", "mask_supported"):
        if np.asarray(result[name]).dtype != bool:
            raise ValueError(f"tracker result {name} must be bool")
    source = np.asarray(result["sources"])
    if not np.issubdtype(source.dtype, np.integer) or np.any(source < -1):
        raise ValueError("tracker sources must be integer rows or -1")
    observed = result["observed"]
    if np.any(observed & ((source >= 0).sum(axis=2) < 2)):
        raise ValueError("observed node lacks two directly supported views")
    if np.any(observed & ~np.isfinite(result["pose_xyz"]).all(axis=-1)):
        raise ValueError("observed node has nonfinite latent coordinates")
    for name in ("xyz", "accepted_xyz"):
        points = result[name]
        finite = np.isfinite(points).all(axis=-1)
        if np.any(np.isfinite(points).any(axis=-1) != finite):
            raise ValueError(f"{name} must contain whole finite points or whole NaN points")
        if np.any(finite & ~observed[:, :, 1]):
            raise ValueError(f"{name} exposes an unsupported pectoral point")
        np.testing.assert_allclose(
            points[finite], result["pose_xyz"][:, :, 1][finite], rtol=0, atol=1e-8
        )
    accepted = np.isfinite(result["accepted_xyz"]).all(axis=-1)
    if np.any(accepted & (result["identity_review"] | ~np.isfinite(result["xyz"]).all(axis=-1))):
        raise ValueError("accepted output exposes a reviewed identity or unavailable point")
    for t, frame in enumerate(frames):
        for vi, view in enumerate(VIEWS):
            count = len(nodes.at(view, int(frame)))
            if np.any(source[t, :, vi] >= count):
                raise ValueError(f"tracker source row out of range at {view}/{frame}")
            for node in range(3):
                rows = source[t, :, vi, node]
                rows = rows[rows >= 0]
                if len(np.unique(rows)) != len(rows):
                    raise ValueError(
                        f"tracker reused one node observation across fish: {view}/{frame}"
                    )


def candidate_metrics(result, frames, initial, anchors, rig):
    raw = score_female_identity(
        frames, result["sources"][:, :, :, 0], result["observed"][:, :, 0], anchors
    )
    accepted = score_female_identity(
        frames,
        result["sources"][:, :, :, 0],
        result["observed"][:, :, 0],
        anchors,
        review=result["identity_review"],
    )
    return {
        "motion": observed_motion_summary(result["xyz"], frames),
        "accepted_motion": observed_motion_summary(result["accepted_xyz"], frames),
        "observed_episodes": availability_episodes(result["xyz"], frames),
        "accepted_episodes": availability_episodes(result["accepted_xyz"], frames),
        "availability_assessment": assess_availability(result["accepted_xyz"], frames),
        "skeleton": skeleton_shape_summary(result["pose_xyz"], result["observed"], initial),
        "nominal_tank": nominal_tank_summary(result["pose_xyz"], result["observed"]),
        "accepted_nominal_tank": nominal_tank_summary(
            result["pose_xyz"], result["observed"] & ~result["identity_review"][..., None]
        ),
        "female_identity_raw": raw,
        "female_identity_accepted": accepted,
        "mouth_reprojection": mouth_reprojection_summary(
            result["pose_xyz"],
            frames,
            anchors,
            raw,
            lambda view, xyz: rig.camera(view).project(xyz),
        ),
        "review_animal_frames": int(result["identity_review"].sum()),
        "review_fraction": float(result["identity_review"].mean()),
        "accepted_observed_animal_frames": int(
            np.isfinite(result["accepted_xyz"]).all(axis=-1).sum()
        ),
        "mask_supported_animal_frames": int(result["mask_supported"].sum()),
        "mask_support_scope": "rendered support diagnostic; not direct node or identity truth",
    }


def baseline_metrics(input_root, catalog, frames, rig, nodes, anchors, seed):
    arms = {}
    for value in BASELINE_LAMBDAS:
        relative = f"{BASELINE_DIR}/C34-label__{EXPERIMENT}__lam{value}.npz"
        path = verified_input(input_root, relative, catalog)
        if value == "0.767" and sha256(path) != seed["baseline_sha256"]:
            raise ValueError("seed baseline differs from the historical registration")
        with np.load(path, allow_pickle=False) as arrays:
            source = arrays["source"][frames]
            accepted = arrays["slot_track"][frames] >= 0
            xyz = np.where(accepted[..., None], arrays["xyz"][frames], np.nan)
        if value == "0.767":
            np.testing.assert_array_equal(source[0], seed["source"])
        pose, observed = skeleton_from_sources(rig, nodes, frames, source)
        observed &= accepted[..., None]
        identity = score_female_identity(frames, source, observed[:, :, 0], anchors)
        arms[value] = {
            "path": str(path),
            "sha256": sha256(path),
            "motion": observed_motion_summary(xyz, frames),
            "accepted_episodes": availability_episodes(xyz, frames),
            "female_identity": identity,
            "skeleton": skeleton_shape_summary(pose, observed, pose[0]),
            "nominal_tank": nominal_tank_summary(pose, observed),
            "mouth_reprojection": mouth_reprojection_summary(
                pose, frames, anchors, identity, lambda view, xyz: rig.camera(view).project(xyz)
            ),
        }
    return arms


def frozen_code_paths():
    local = (
        "run_experiment.py",
        "tracker.py",
        "renderer.py",
        "fast_union.py",
        "node_geometry.py",
        "observations.py",
        "prepare_sequence.py",
        "motion.py",
        "V1_PROTOCOL.md",
        "README.md",
    )
    shared = (
        "calib.py",
        "types.py",
        "provenance.py",
        "eval/generative.py",
        "eval/reacquisition.py",
    )
    return [*(LAB / name for name in local), *(REPO / "src/medaka3d" / name for name in shared)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("detectron2", "top_contrast", "nodes"), default="detectron2"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--opt-steps", type=int)
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--max-hypotheses", type=int)
    parser.add_argument("--flow-proposals", action="store_true")
    parser.add_argument(
        "--max-frames", type=int, help="Development smoke: process only this native-rate prefix"
    )
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite experiment output: {args.output}")
    for key in ("opt_steps", "beam_size", "max_hypotheses", "max_frames"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    sequence_path = args.sequence / "manifest.json"
    sequence_manifest = json.loads(sequence_path.read_text(encoding="utf-8"))
    frames = sequence_frames(sequence_manifest, args.max_frames)
    initial, seed, evaluation = registered_seed(int(frames[0]))
    catalog, staging_manifest = staging_catalog(args.input_root)
    # Verify only inference assets here. Baseline/GT content is strictly deferred.
    inference_inputs = {}
    for name in (
        "calibration_manifest.json",
        *(f"cam{i}_bundle_refractive.json" for i in range(3)),
    ):
        relative = f"{CALIBRATION_DIR}/{name}"
        inference_inputs[relative] = verified_input(args.input_root, relative, catalog)
    for view in VIEWS:
        relative = f"work/c01_s16_predictions/bottomup/{EXPERIMENT}__{view}.npz"
        inference_inputs[relative] = verified_input(args.input_root, relative, catalog)
    for category in ("raw", "masks", "backgrounds"):
        for view in VIEWS:
            resolve_input(args.sequence, sequence_manifest[category][view]["path"])
    import cv2
    import scipy
    import torch
    from tracker import ThreeBodyTracker, TrackerConfig

    torch.set_num_threads(2)
    overrides = {
        key: getattr(args, key)
        for key in ("mode", "device", "opt_steps", "beam_size", "max_hypotheses", "flow_proposals")
        if getattr(args, key) is not None
    }
    config = TrackerConfig(**overrides)
    code_paths = frozen_code_paths()
    hashes = {str(path.relative_to(REPO)): sha256(path) for path in code_paths}
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance("three_body_single_initialization", args.output, name="experiment") as record:
        preparation_started = time.perf_counter()
        record.params_from(args)
        record.param("tracker_config", dataclasses.asdict(config))
        if config.flow_proposals:
            from motion import FlowConfig

            record.param("flow_config", dataclasses.asdict(FlowConfig()))
        record.param("torch_cpu_threads", torch.get_num_threads())
        record.param("code_hashes", hashes)
        record.param("code_origin", CODE_ORIGIN)
        record.param("seed", seed)
        record.param("baseline_and_gt_opened_only_after_tracking", True)
        record.param(
            "package_versions",
            {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "opencv": cv2.__version__,
                "torch": torch.__version__,
            },
        )
        record.input("sequence_manifest", sequence_path, sha256=sha256(sequence_path))
        record.input("staging_manifest", staging_manifest, sha256=sha256(staging_manifest))
        record.input("seed_registration", REGISTRATION, sha256=REGISTRATION_SHA256)
        input_hashes = {str(path): sha256(path) for path in inference_inputs.values()}
        input_hashes.update(
            {
                str(sequence_path): sha256(sequence_path),
                str(staging_manifest): sha256(staging_manifest),
            }
        )
        for relative, path in inference_inputs.items():
            record.input(relative, path, sha256=input_hashes[str(path)])
        for category in ("raw", "masks", "backgrounds"):
            for view in VIEWS:
                entry = sequence_manifest[category][view]
                path = resolve_input(args.sequence, entry["path"])
                input_hashes[str(path)] = entry["sha256"]
                record.input(category + "_" + view, path, sha256=entry["sha256"])
        for path in code_paths:
            target = args.output / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for name, source in (
            ("sequence_manifest.json", sequence_path),
            ("staging_manifest.json", staging_manifest),
        ):
            shutil.copyfile(source, args.output / name)
        for relative, source in inference_inputs.items():
            if relative.startswith(CALIBRATION_DIR):
                destination = args.output / "calibration_snapshot" / source.name
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, destination)
        seed_path = args.output / "seed_registration.json"
        seed_path.write_text(json.dumps(seed, indent=2, allow_nan=False), encoding="utf-8")
        np.save(args.output / "initial_pose_xyz.npy", initial, allow_pickle=False)
        record.output("seed_registration", seed_path, sha256=sha256(seed_path))
        rig = load_rig(args.input_root / CALIBRATION_DIR)
        reader = SequenceReader(args.sequence, args.input_root)
        preparation_seconds = time.perf_counter() - preparation_started
        progress_path = args.output / "progress.jsonl"
        observations = iter(reader)
        started = time.perf_counter()
        processed = 0
        with progress_path.open("x", encoding="utf-8", buffering=1) as progress:

            def inputs():
                nonlocal processed
                for index, observation in enumerate(islice(observations, len(frames))):
                    if observation.frame_idx != frames[index]:
                        raise ValueError("sequence yielded an unexpected source frame")
                    yield observation
                    processed += 1
                    if processed % 25 == 0 or processed == len(frames):
                        event = {
                            "processed": processed,
                            "requested": len(frames),
                            "source_frame": observation.frame_idx,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                        progress.write(json.dumps(event) + "\n")
                        print(json.dumps(event), flush=True)

            try:
                tracker = ThreeBodyTracker(rig, initial.copy(), config=config)
                result = tracker.track(inputs())
                if config.device == "cuda":
                    torch.cuda.synchronize()
            finally:
                observations.close()
        tracking_seconds = time.perf_counter() - started
        if processed != len(frames):
            raise ValueError("tracker did not consume the complete requested native-rate sequence")
        if tracker.initialize_count != 1:
            raise ValueError("tracker did not report exactly one initialization")
        validate_result(result, frames, reader.nodes)
        arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
        if any(value.dtype.hasobject for value in arrays.values()):
            raise ValueError("tracker output contains object arrays; pickle storage is forbidden")
        candidate_path = args.output / "candidate.npz"
        np.savez_compressed(candidate_path, **arrays)
        record.output("candidate", candidate_path, sha256=sha256(candidate_path))
        # This boundary is deliberately below the only call to tracker.track.
        gt_path = verified_input(
            args.input_root, evaluation["female_annotations_relative"], catalog
        )
        if sha256(gt_path) != evaluation["female_annotations_sha256"]:
            raise ValueError("female labels differ from the registered evaluation source")
        record.input("female_annotations_post_inference", gt_path, sha256=sha256(gt_path))
        with gt_path.open(encoding="utf-8-sig", newline="") as stream:
            anchors = match_mouth_anchors(
                csv.DictReader(stream),
                lambda view, frame: reader.nodes.at(view, frame)[:, 0],
                int(frames[0]),
                int(frames[-1]) + 1,
            )
        candidate = candidate_metrics(result, frames, initial, anchors, rig)
        baselines = baseline_metrics(
            args.input_root, catalog, frames, rig, reader.nodes, anchors, seed
        )
        code_after = {str(path.relative_to(REPO)): sha256(path) for path in code_paths}
        inputs_after = {path: sha256(path) for path in input_hashes}
        if (
            code_after != hashes
            or inputs_after != input_hashes
            or sha256(REGISTRATION) != REGISTRATION_SHA256
        ):
            raise RuntimeError("code or input changed during the run; preserved output is invalid")
        summary = {
            "protocol": "three_body_single_initialization",
            "experiment": EXPERIMENT,
            "n_groups": 1,
            "mode": args.mode,
            "window": sequence_manifest.get("window", "continuous_from_76620"),
            "frames": [int(frames[0]), int(frames[-1]) + 1],
            "source_sequence_frames": [
                sequence_manifest["frames"][0],
                sequence_manifest["frames"][-1] + 1,
            ],
            "requested_frames": len(frames),
            "returned_frames": len(result["frame_idx"]),
            "native_fps": 166,
            "temporal_subsampling": False,
            "development_prefix_smoke": args.max_frames is not None,
            "technical_completion": True,
            "initialization_count": 1,
            "track_invocation_count": 1,
            "reset_count": 0,
            "reset_count_scope": "external seed resets; recovery hypotheses remain internal",
            "preparation_seconds": preparation_seconds,
            "tracking_seconds": tracking_seconds,
            "tracking_fps": len(frames) / tracking_seconds,
            "tracking_timing_includes": "local raw clip decode, packed-mask decode and tracking",
            "tracking_timing_excludes": "D2 staging, input hashes and evaluation",
            "precomputed_sequence_staging_seconds": sequence_manifest.get("staging_elapsed_s"),
            "tracker_config": dataclasses.asdict(config),
            "seed": seed,
            "candidate": candidate,
            "baseline_arms": baselines,
            "code_unchanged_during_run": True,
            "inputs_unchanged_during_run": True,
            "code_sha256_before": hashes,
            "code_sha256_after": code_after,
            "input_sha256_before": input_hashes,
            "input_sha256_after": inputs_after,
            "output_array_shapes": {key: list(value.shape) for key, value in arrays.items()},
            "limitations": [
                "Completion, mask fit and availability do not certify persistent identity.",
                "One optimistic initial pose originated from the whole-recording baseline.",
                "Baseline trajectories and human labels were opened only after tracking returned.",
                "Female endpoints cannot establish male identity or exclude returning swaps.",
                "No independent all-three-animal identity verdict is produced by this runner.",
                "Only candidate runtime is measured; frozen baseline computation time is unknown.",
            ],
        }
        output = args.output / "summary.json"
        output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", output, sha256=sha256(output))
        record.output("progress", progress_path, sha256=sha256(progress_path))
        record.param("code_unchanged_during_run", True)
        record.param("inputs_unchanged_during_run", True)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "frames": len(frames),
                    "tracking_seconds": tracking_seconds,
                    "development_prefix_smoke": args.max_frames is not None,
                }
            ),
            flush=True,
        )
    return summary


if __name__ == "__main__":
    main()

"""Refine a bounded offline window of a sealed, complete forward experiment.

All twelve endpoint-permutation/start hypotheses are retained. Boundaries are
internal forward estimates, not truth. No baseline or human-label arrays are
opened, and this diagnostic never publishes accepted identity coordinates.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(LAB))
from node_geometry import RayObservations  # noqa: E402
from observations import SequenceReader, sample_pixels  # noqa: E402
from run_experiment import (  # noqa: E402
    CALIBRATION_DIR,
    EXPERIMENT,
    resolve_input,
    sha256,
)

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.eval.generative import (  # noqa: E402
    observed_motion_summary,
    skeleton_shape_summary,
)
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

PIXEL_SAMPLING = {"n_positive": 96, "n_near": 96, "n_global": 64}
LOSS_PART_NAMES = ("node", "shape", "acceleration", "speed", "repulsion", "image")


def verify_file(path, digest, size=None):
    if not isinstance(digest, str) or len(digest) != 64 or sha256(path) != digest:
        raise ValueError(f"sealed input checksum mismatch: {path}")
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"sealed input size mismatch: {path}")
    return path


def select_window(frames, pose, observed, start, stop):
    if not 5 <= stop - start <= 50:
        raise ValueError("offline refinement requires 5 to 50 consecutive native frames")
    frames = np.asarray(frames)
    if frames.ndim != 1 or len(frames) == 0 or np.any(np.diff(frames) != 1):
        raise ValueError("forward result must contain consecutive source frames")
    if pose.shape != (len(frames), 3, 3, 3) or observed.shape != (len(frames), 3, 3):
        raise ValueError("forward pose/observed shape mismatch")
    if observed.dtype != bool:
        raise ValueError("forward observed flags must be boolean")
    wanted = np.arange(start, stop, dtype=np.int64)
    indices = np.searchsorted(frames, wanted)
    if np.any(indices >= len(frames)) or not np.array_equal(frames[indices], wanted):
        raise ValueError("requested refinement window is not fully present in forward output")
    selected_pose, selected_observed = pose[indices].copy(), observed[indices].copy()
    if not np.isfinite(selected_pose).all():
        raise ValueError("refinement window requires finite forward latent poses")
    boundary = np.r_[0:2, len(wanted) - 2 : len(wanted)]
    if not selected_observed[boundary, :, :2].all():
        raise ValueError("first/last two frames require directly observed mouths and pectorals")
    return wanted, selected_pose, selected_observed


def load_sealed_forward(run, sequence, input_root, start, stop):
    paths = {
        name: run / name
        for name in (
            "experiment.provenance.json",
            "summary.json",
            "candidate.npz",
            "seed_registration.json",
            "initial_pose_xyz.npy",
            "sequence_manifest.json",
        )
    }
    frozen = {str(path): sha256(path) for path in paths.values()}
    forward_record = json.loads(paths["experiment.provenance.json"].read_text(encoding="utf-8"))
    if forward_record["status"] != "ok":
        raise ValueError("forward experiment has not completed successfully")
    for key, name in (
        ("summary", "summary.json"),
        ("candidate", "candidate.npz"),
        ("seed_registration", "seed_registration.json"),
    ):
        entry = forward_record["outputs"][key]
        verify_file(paths[name], entry["sha256"], entry["size"])
    summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
    if (
        summary["experiment"] != EXPERIMENT
        or not summary["technical_completion"]
        or summary["development_prefix_smoke"]
        or summary["temporal_subsampling"]
        or summary["native_fps"] != 166
        or summary["initialization_count"] != 1
        or summary["track_invocation_count"] != 1
        or summary["reset_count"] != 0
        or summary["mode"] not in ("detectron2", "top_contrast", "nodes")
        or not summary["code_unchanged_during_run"]
        or not summary["inputs_unchanged_during_run"]
        or summary["code_sha256_before"] != summary["code_sha256_after"]
        or summary["input_sha256_before"] != summary["input_sha256_after"]
    ):
        raise ValueError("forward experiment is incomplete or has an incompatible protocol")
    for relative, digest in summary["code_sha256_before"].items():
        path = resolve_input(run / "code_snapshot", relative)
        verify_file(path, digest)
        frozen[str(path)] = digest
    sequence_path = sequence / "manifest.json"
    sequence_sha = forward_record["inputs"]["sequence_manifest"]["sha256"]
    verify_file(sequence_path, sequence_sha)
    verify_file(paths["sequence_manifest.json"], sequence_sha)
    sequence_manifest = json.loads(sequence_path.read_text(encoding="utf-8"))
    if sequence_manifest["fps"] != 166 or sequence_manifest["experiment"] != EXPERIMENT:
        raise ValueError("sequence does not match the forward native-rate experiment")
    frozen[str(sequence_path)] = sequence_sha
    with np.load(paths["candidate.npz"], allow_pickle=False) as arrays:
        source_frames = arrays["frame_idx"]
        if not np.array_equal(source_frames, sequence_manifest["frames"]):
            raise ValueError("forward output is not the complete staged sequence")
        if summary["frames"] != [int(source_frames[0]), int(source_frames[-1]) + 1]:
            raise ValueError("forward summary frame bounds differ from its arrays")
        if summary["returned_frames"] != len(source_frames):
            raise ValueError("forward summary frame count differs from its arrays")
        frames, pose, observed = select_window(
            source_frames, arrays["pose_xyz"], arrays["observed"], start, stop
        )
    initial = np.load(paths["initial_pose_xyz.npy"], allow_pickle=False)
    seed = json.loads(paths["seed_registration.json"].read_text(encoding="utf-8"))
    if initial.shape != (3, 3, 3) or not np.isfinite(initial).all():
        raise ValueError("initial pose is incomplete")
    if not np.array_equal(initial, seed["pose_xyz"]) or seed != summary["seed"]:
        raise ValueError("initial pose is not the sealed forward seed")
    inference_inputs = {}
    for name in (
        "calibration_manifest.json",
        *(f"cam{i}_bundle_refractive.json" for i in range(3)),
    ):
        relative = f"{CALIBRATION_DIR}/{name}"
        inference_inputs[relative] = resolve_input(input_root, relative)
    for view in VIEWS:
        relative = f"work/c01_s16_predictions/bottomup/{EXPERIMENT}__{view}.npz"
        inference_inputs[relative] = resolve_input(input_root, relative)
    for key, path in inference_inputs.items():
        entry = forward_record["inputs"][key]
        verify_file(path, entry["sha256"], entry["size"])
        frozen[str(path)] = entry["sha256"]
    for category in ("raw", "masks", "backgrounds"):
        for view in VIEWS:
            info = sequence_manifest[category][view]
            entry = forward_record["inputs"][f"{category}_{view}"]
            if info["sha256"] != entry["sha256"]:
                raise ValueError("forward input and sequence manifest disagree")
            path = resolve_input(sequence, info["path"])
            verify_file(path, info["sha256"], entry["size"])
            frozen[str(path)] = info["sha256"]
    return summary, frames, pose, observed, initial, frozen, inference_inputs


def validate_hypotheses(arrays, frames, rays, *, expected_kinds=("linear", "scene")):
    if not expected_kinds or len(set(expected_kinds)) != len(expected_kinds):
        raise ValueError("expected initialization kinds must be nonempty and distinct")
    count = 6 * len(expected_kinds)
    required = {
        "pose_xyz": (count, len(frames), 3, 3, 3),
        "sources": (count, len(frames), 3, 3, 3),
        "observed": (count, len(frames), 3, 3),
        "costs": (count,),
        "loss_parts": (count, 6),
        "permutations": (count, 3),
        "start_kind": (count,),
        "boundary_max_error_mm": (count,),
        "original_cost": (1,),
        "original_loss_parts": (1, 6),
    }
    for key, shape in required.items():
        if key not in arrays or arrays[key].shape != shape or arrays[key].dtype.hasobject:
            raise ValueError(f"invalid refinement output: {key}")
    if arrays["observed"].dtype != bool:
        raise ValueError("refined observed flags must be boolean")
    numeric = (
        "pose_xyz",
        "costs",
        "loss_parts",
        "boundary_max_error_mm",
        "original_cost",
        "original_loss_parts",
        "checkpoint_costs",
    )
    if any(not np.isfinite(arrays[key]).all() for key in numeric):
        raise ValueError("refinement returned a nonfinite pose or cost")
    if arrays["checkpoint_costs"].ndim != 2 or arrays["checkpoint_costs"].shape[1] != count:
        raise ValueError("checkpoint costs must retain all registered competitors")
    if set(arrays["start_kind"].tolist()) != set(expected_kinds):
        raise ValueError("unknown temporal refinement initialization kind")
    source = arrays["sources"]
    if not np.issubdtype(source.dtype, np.integer) or np.any(source < -1):
        raise ValueError("refined source rows must be integers or missing (-1)")
    if np.any(arrays["observed"] & ((source >= 0).sum(axis=3) < 2)):
        raise ValueError("refined observed node has fewer than two views")
    perms = arrays["permutations"]
    if not np.issubdtype(perms.dtype, np.integer):
        raise ValueError("endpoint permutations must be integer slots")
    for perm in permutations(range(3)):
        selected = np.all(perms == perm, axis=1)
        if selected.sum() != len(expected_kinds) or set(arrays["start_kind"][selected]) != set(
            expected_kinds
        ):
            raise ValueError("need all six permutations with each registered start kind")
    for t, ray in enumerate(rays):
        for vi, view in enumerate(VIEWS):
            if np.any(source[:, t, :, vi] >= len(ray.points[view])):
                raise ValueError("refined source row outside its current observation array")
            for batch in range(count):
                for node in range(3):
                    rows = source[batch, t, :, vi, node]
                    rows = rows[rows >= 0]
                    if len(np.unique(rows)) != len(rows):
                        raise ValueError("one refined node observation was reused across fish")


def validate_boundaries(arrays, forward_pose, forward_observed):
    boundary = [0, 1, len(forward_pose) - 2, len(forward_pose) - 1]
    for b, perm in enumerate(arrays["permutations"]):
        wanted = np.concatenate((forward_pose[:2], forward_pose[-2:, perm]), axis=0)
        held = np.concatenate((forward_observed[:2], forward_observed[-2:, perm]), axis=0)
        error = np.abs(arrays["pose_xyz"][b, boundary] - wanted)[held]
        if error.max(initial=0) > 1e-10:
            raise ValueError("refinement moved a fixed observed boundary coordinate")
        if abs(error.max(initial=0) - arrays["boundary_max_error_mm"][b]) > 1e-10:
            raise ValueError("reported boundary error disagrees with actual refined coordinates")


def motion_diagnostics(pose, observed, frames, initial):
    latent_step = np.linalg.norm(np.diff(pose[:, :, 1], axis=0), axis=-1)
    fish = []
    for index in range(3):
        step = latent_step[:, index]
        counts = observed[:, index].sum(axis=0)
        fish.append(
            {
                "slot": index,
                "frame_denominator": len(frames),
                "observed_node_counts": counts.tolist(),
                "observed_node_fractions": (counts / len(frames)).tolist(),
                "all_nodes_observed_fraction": float(observed[:, index].all(axis=1).mean()),
                "latent_pectoral_step_mm_median": float(np.median(step)),
                "latent_pectoral_step_mm_p99": float(np.percentile(step, 99)),
                "latent_pectoral_step_mm_max": float(step.max()),
                "latent_pectoral_speed_mm_s_p99": float(np.percentile(step, 99) * 166),
            }
        )
    points = np.where(observed[:, :, 1, None], pose[:, :, 1], np.nan)
    return {
        "per_fish": fish,
        "observed_nodes": int(observed.sum()),
        "node_denominator": int(observed.size),
        "observed_node_fraction": float(observed.mean()),
        "observed_pectoral_motion": observed_motion_summary(points, frames),
        "skeleton_shape": skeleton_shape_summary(pose, observed, initial),
        "scope": "separate hypothesis diagnostics; no pooling across B and no identity score",
    }


def freeze_code_paths():
    local = (
        "run_temporal_refine.py",
        "temporal_refine.py",
        "temporal_renderer.py",
        "renderer.py",
        "fast_union.py",
        "node_geometry.py",
        "support_batch.py",
        "observations.py",
        "prepare_sequence.py",
        "run_experiment.py",
        "V2_PROTOCOL.md",
    )
    shared = (
        "calib.py",
        "types.py",
        "provenance.py",
        "eval/generative.py",
        "eval/reacquisition.py",
    )
    return [*(LAB / p for p in local), *(REPO / "src/medaka3d" / p for p in shared)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True, help="Exclusive source-frame stop")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite refinement output: {args.output}")
    if args.steps < 1:
        raise ValueError("refinement steps must be positive")
    forward, frames, pose, observed, initial, input_hashes, inference_inputs = load_sealed_forward(
        args.run, args.sequence, args.input_root, args.start, args.stop
    )
    import cv2
    import scipy
    import torch
    from temporal_refine import RefinementConfig, refine_window

    torch.set_num_threads(2)
    config = RefinementConfig(mode=forward["mode"], device=args.device, steps=args.steps)
    sampling_mode = "detectron2" if forward["mode"] == "nodes" else forward["mode"]
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "requested_device": args.device,
        "cuda_device_name": torch.cuda.get_device_name() if args.device == "cuda" else None,
    }
    code_paths = freeze_code_paths()
    code_hashes = {path.relative_to(REPO).as_posix(): sha256(path) for path in code_paths}
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance(
        "three_body_offline_temporal_refinement", args.output, name="refinement"
    ) as record:
        record.params_from(args)
        record.param("config", dataclasses.asdict(config))
        record.param("pixel_sampling", PIXEL_SAMPLING)
        record.param("actual_pixel_sampling_mode", sampling_mode)
        record.param("runtime", environment)
        record.param("code_sha256", code_hashes)
        record.param("torch_cpu_threads", torch.get_num_threads())
        record.param("forward_initialization_count", 1)
        record.param("external_reset_count", 0)
        record.param("boundaries_are_internal_forward_estimates", True)
        for path, digest in input_hashes.items():
            record.input(path, path, sha256=digest)
        for source in code_paths:
            destination = args.output / "code_snapshot" / source.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        for name in (
            "summary.json",
            "experiment.provenance.json",
            "seed_registration.json",
            "initial_pose_xyz.npy",
            "sequence_manifest.json",
        ):
            target = args.output / "forward_snapshot" / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(args.run / name, target)
        for relative, source in inference_inputs.items():
            if relative.startswith(CALIBRATION_DIR):
                target = args.output / "calibration_snapshot" / source.name
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, target)
        rig = load_rig(args.input_root / CALIBRATION_DIR)
        reader = SequenceReader(args.sequence, args.input_root)
        rays, targets, found = [], [], []
        iterator = iter(reader)
        preparation_started = time.perf_counter()
        try:
            for observation in iterator:
                if observation.frame_idx < args.start:
                    continue
                if observation.frame_idx >= args.stop:
                    break
                found.append(observation.frame_idx)
                rays.append(RayObservations.from_points(rig, observation.detections))
                targets.append(sample_pixels(observation, mode=sampling_mode, **PIXEL_SAMPLING))
        finally:
            iterator.close()
        if not np.array_equal(found, frames):
            raise ValueError("selected source frames were not fully read in native order")
        target_path = args.output / "sampled_targets.npz"
        np.savez_compressed(
            target_path,
            frame_idx=frames,
            views=np.asarray(VIEWS),
            pixels=np.stack([np.stack([target.pixels[v] for v in VIEWS]) for target in targets]),
            target=np.stack([target.target for target in targets]),
            weights=np.stack([target.weights for target in targets]),
            foreground=np.stack([target.foreground for target in targets]),
        )
        target_sha256 = sha256(target_path)
        record.output("sampled_targets", target_path, sha256=target_sha256)
        preparation_seconds = time.perf_counter() - preparation_started
        before_pose, before_observed = pose.copy(), observed.copy()
        started = time.perf_counter()
        arrays = refine_window(rig, pose, observed, rays, targets, initial.copy(), config)
        if config.device == "cuda":
            torch.cuda.synchronize()
        refinement_seconds = time.perf_counter() - started
        if not np.array_equal(pose, before_pose) or not np.array_equal(observed, before_observed):
            raise ValueError("optimizer mutated its supplied forward arrays")
        validate_hypotheses(arrays, frames, rays)
        validate_boundaries(arrays, before_pose, before_observed)
        saved = {key: value for key, value in arrays.items() if isinstance(value, np.ndarray)}
        if any(value.dtype.hasobject for value in saved.values()):
            raise ValueError("pickle-valued refinement output is forbidden")
        output = args.output / "hypotheses.npz"
        np.savez_compressed(output, frame_idx=frames, **saved)
        record.output("hypotheses", output, sha256=sha256(output))
        rankings = []
        for perm in permutations(range(3)):
            selected = np.flatnonzero(np.all(arrays["permutations"] == perm, axis=1))
            best = selected[np.argmin(arrays["costs"][selected])]
            rankings.append(
                {
                    "permutation": list(perm),
                    "best_hypothesis": int(best),
                    "best_cost": float(arrays["costs"][best]),
                    "starts": [
                        {
                            "hypothesis": int(i),
                            "kind": str(arrays["start_kind"][i]),
                            "cost": float(arrays["costs"][i]),
                        }
                        for i in selected
                    ],
                }
            )
        rankings.sort(key=lambda entry: (entry["best_cost"], entry["permutation"]))
        for index, entry in enumerate(rankings):
            entry["rank"] = index + 1
            entry["cost_above_best"] = entry["best_cost"] - rankings[0]["best_cost"]
        code_after = {path.relative_to(REPO).as_posix(): sha256(path) for path in code_paths}
        inputs_after = {path: sha256(path) for path in input_hashes}
        if code_after != code_hashes or inputs_after != input_hashes:
            raise RuntimeError("refinement source or input changed during this run")
        result = {
            "protocol": "separate_offline_bounded_window_refinement",
            "mode": forward["mode"],
            "frames": [args.start, args.stop],
            "returned_native_frames": len(frames),
            "native_fps": 166,
            "temporal_subsampling": False,
            "hypotheses": len(arrays["costs"]),
            "config": dataclasses.asdict(config),
            "pixel_sampling": PIXEL_SAMPLING,
            "actual_pixel_sampling_mode": sampling_mode,
            "sampled_targets_sha256": target_sha256,
            "sampled_targets_scope": "one fixed target per frame/view, shared by all 12 hypotheses",
            "runtime": environment,
            "forward_candidate_sha256": input_hashes[str(args.run / "candidate.npz")],
            "forward_initialization_count": forward["initialization_count"],
            "forward_external_reset_count": forward["reset_count"],
            "new_external_reset_count": 0,
            "accepted_identity_output_written": False,
            "boundary_source": "first/last two internal forward estimates; not ground truth",
            "preparation_seconds": preparation_seconds,
            "refinement_seconds": refinement_seconds,
            "permutation_rankings": rankings,
            "permutation_semantics": "left identity -> original forward right-boundary slot",
            "loss_part_names": LOSS_PART_NAMES,
            "original_recomputed_cost": float(arrays["original_cost"][0]),
            "original_recomputed_loss_parts": arrays["original_loss_parts"][0].tolist(),
            "original_cost_scope": "same 256 pixel targets/objective, fresh SLEAP assignments",
            "forward_diagnostics": motion_diagnostics(pose, observed, frames, initial),
            "hypothesis_diagnostics": [
                {
                    "hypothesis": i,
                    "permutation": arrays["permutations"][i].tolist(),
                    "start_kind": str(arrays["start_kind"][i]),
                    "cost": float(arrays["costs"][i]),
                    "loss_parts": arrays["loss_parts"][i].tolist(),
                    **motion_diagnostics(
                        arrays["pose_xyz"][i], arrays["observed"][i], frames, initial
                    ),
                }
                for i in range(12)
            ],
            "code_sha256_before": code_hashes,
            "code_sha256_after": code_after,
            "input_sha256_before": input_hashes,
            "input_sha256_after": inputs_after,
            "code_unchanged_during_run": True,
            "inputs_unchanged_during_run": True,
            "limitations": [
                "Endpoint permutation ranks and fitted support do not establish biological IDs.",
                "Future internal estimates constrain this offline, noncausal window.",
                "No new baseline arrays or human labels are opened or supplied to refinement.",
                "Existing evaluation fields in the forward summary are not optimization inputs.",
                "Forward accepted coordinates and identity-review decisions remain untouched.",
            ],
        }
        path = args.output / "summary.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=sha256(path))
        print(
            json.dumps(
                {
                    "frames": len(frames),
                    "hypotheses": 12,
                    "refinement_seconds": refinement_seconds,
                    "rankings": rankings,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

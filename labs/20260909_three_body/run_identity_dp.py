"""Infer proposed IDs over a full frozen geometry sequence; no evaluation labels.

The only discrete initial condition is identity permutation zero at frame zero.
No geometry fit, external reset, template update or original review clearing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import run_experiment as base
import scipy
import torch
from identity_dp import PERMUTATIONS, build_adjacent_motion_cost, solve_identity_dp
from identity_dp import _frames as native_frames
from identity_emissions import PERMUTATIONS as EMISSION_PERMUTATIONS
from identity_emissions import IdentityEmissionConfig, score_identity_permutations
from identity_inputs import load_identity_inputs
from identity_output import build_identity_output
from pixel_appearance_loss import PixelAppearanceConfig

from medaka3d.provenance import provenance

STAGE = "three_body_fixed_geometry_identity_dp"


def registered_config():
    return {
        "emission": dataclasses.asdict(IdentityEmissionConfig()),
        "pixel": dataclasses.asdict(PixelAppearanceConfig()),
        "training_frames": 166, "appearance_weight": 1.0,
        "motion_scale_mm": 0.8, "motion_weight": 0.2,
        "initial_state": 0, "tie_atol": 1e-10, "tie_rtol": 1e-12,
    }


def package_versions():
    return {
        "numpy": np.__version__, "scipy": scipy.__version__,
        "opencv": cv2.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "cuda_device_name": torch.cuda.get_device_name(0),
    }


def code_paths():
    """Capture actual imported workspace dependencies, including this entrypoint."""
    result = {Path(__file__).resolve(), base.LAB / "IDENTITY_DP_PROTOCOL.md"}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(base.LAB) or path.is_relative_to(base.REPO / "src")
            ):
                result.add(path)
    return sorted(result)


def collect_scores(reader, rig, geometry, enrollment, config, *, device, progress=None):
    """Consume every native frame exactly once, scoring only after enrollment."""
    frames = native_frames(geometry["frame_idx"])
    if not np.array_equal(native_frames(reader.frames), frames):
        raise ValueError("reader and frozen geometry must have identical full native frames")
    np.testing.assert_array_equal(PERMUTATIONS, EMISSION_PERMUTATIONS)
    n = len(frames)
    scores = {
        "frame_idx": frames.copy(), "unary": np.zeros((n, 6)),
        "per_view_costs": np.zeros((n, 6, 3)),
        "enabled_views": np.zeros((n, 3), bool),
        "sampled_pixels": np.zeros((n, 3), np.int64),
        "valid_pixels": np.zeros((n, 3), np.int64),
        "available_core_pixels": np.zeros((n, 3), np.int64),
        "visible_mass": np.zeros((n, 3, 3)),
        "rendering_performed": np.zeros(n, bool),
    }
    emission = IdentityEmissionConfig(**config["emission"])
    pixel = PixelAppearanceConfig(**config["pixel"])
    observations = iter(reader)
    processed = 0
    started = time.perf_counter()
    try:
        for index, frame in enumerate(observations):
            if index >= n or frame.frame_idx != frames[index]:
                raise ValueError("extra or incorrectly aligned native observation")
            if index >= config["training_frames"]:
                result = score_identity_permutations(
                    rig, frame, geometry["pose_xyz"][index], enrollment["templates"],
                    enrollment["enabled_views"], config=emission, pixel_config=pixel, device=device,
                )
                scores["unary"][index] = config["appearance_weight"] * result.costs
                for key in scores.keys() - {"frame_idx", "unary"}:
                    scores[key][index] = getattr(result, key)
            processed += 1
            if progress and (processed % 100 == 0 or processed == n):
                progress({
                    "processed": processed, "requested": n, "source_frame": int(frame.frame_idx),
                    "elapsed_seconds": time.perf_counter() - started,
                })
    finally:
        if hasattr(observations, "close"):
            observations.close()
    if processed != n:
        raise ValueError("reader ended before the complete geometry interval")
    if not np.isfinite(scores["unary"]).all():
        raise ValueError("image emissions must be finite on the full interval")
    return scores


def optimize_scores(geometry, scores, config):
    frames = native_frames(geometry["frame_idx"])
    if not np.array_equal(native_frames(scores.get("frame_idx", ())), frames):
        raise ValueError("image score frames must match the complete frozen geometry interval")
    motion = build_adjacent_motion_cost(
        frames, geometry["xyz"], geometry["observed"][:, :, 1],
        scale_mm=config["motion_scale_mm"], weight=config["motion_weight"],
    )
    for key in ("transitions", "supported_pairs", "supported_pair_count", "squared_distance_mm2"):
        scores[key] = getattr(motion, key)
    result = solve_identity_dp(
        frames, scores["unary"], motion.transitions,
        initial_state=config["initial_state"], tie_atol=config["tie_atol"],
        tie_rtol=config["tie_rtol"],
    )
    if result.status != "optimal":
        raise ValueError("registered finite-cost identity problem must be feasible")
    return result, build_identity_output(geometry, result.permutation_by_id)


def json_scalar(value):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def assert_unchanged(hashes, code, versions, *, snapshots=None):
    if any(base.sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("verified inference evidence changed during identity inference")
    if any(base.sha256(base.REPO / path) != digest for path, digest in code.items()):
        raise RuntimeError("identity inference source changed during execution")
    if any(base.sha256(path) != digest for path, digest in (snapshots or {}).items()):
        raise RuntimeError("identity inference source snapshot changed during execution")
    if package_versions() != versions:
        raise RuntimeError("identity inference runtime versions changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--enrollment-run", type=Path, required=True)
    parser.add_argument("--enrollment-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for key in ("geometry_run", "enrollment_run", "enrollment_audit"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output = args.output.resolve()
    if (not args.output.is_relative_to(base.REPO)
            or args.output.exists() or args.output.is_symlink()):
        raise ValueError("a new output directory inside the workspace is required")
    if not torch.cuda.is_available():
        raise RuntimeError("registered image-scoring experiment requires CUDA")
    torch.set_num_threads(2)
    config, sources, versions = registered_config(), code_paths(), package_versions()
    code = {str(path.relative_to(base.REPO)): base.sha256(path) for path in sources}
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance(STAGE, args.output, name="experiment") as record:
        record.params_from(args)
        record.param("config", config)
        record.param("code_hashes", code)
        record.param("package_versions", versions)
        record.param("evaluation_data_opened_during_inference", False)
        record.param("torch_cpu_threads", torch.get_num_threads())
        prepared = time.perf_counter()
        inputs = load_identity_inputs(args.geometry_run, args.enrollment_run, args.enrollment_audit)
        hashes = inputs.verified_hashes
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        snapshots = {}
        for path in sources:
            target = args.output / "code_snapshot" / path.relative_to(base.REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            digest = code[str(path.relative_to(base.REPO))]
            if base.sha256(target) != digest:
                raise RuntimeError("source changed while snapshotting")
            snapshots[str(target)] = digest
            record.output(str(target.relative_to(args.output)), target, sha256=digest)
        frames = inputs.frame_idx
        reader = base.SequenceReader(inputs.geometry_sequence_root, inputs.input_root)
        rig = base.load_rig(inputs.input_root / base.CALIBRATION_DIR)
        preparation_seconds = time.perf_counter() - prepared
        progress_path = args.output / "progress.jsonl"
        inference_started = time.perf_counter()
        with progress_path.open("x", encoding="utf-8", buffering=1) as stream:
            def progress(event):
                stream.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)

            scores = collect_scores(
                reader, rig, inputs.geometry, inputs.enrollment, config,
                device="cuda", progress=progress,
            )
            torch.cuda.synchronize()
        image_seconds = time.perf_counter() - inference_started
        dp_started = time.perf_counter()
        result, identity = optimize_scores(inputs.geometry, scores, config)
        dp_seconds = time.perf_counter() - dp_started
        inference_seconds = time.perf_counter() - inference_started
        result_values = {
            field.name: getattr(result, field.name) for field in dataclasses.fields(result)
        }
        dp_arrays = {
            key: value for key, value in result_values.items() if isinstance(value, np.ndarray)
        }
        dp_scalars = {key: json_scalar(value) for key, value in result_values.items()
                      if not isinstance(value, np.ndarray)}
        for role, arrays in (("identity", identity), ("dp", dp_arrays), ("scores", scores)):
            path = args.output / f"{role}.npz"
            np.savez_compressed(path, **arrays)
            record.output(role, path, sha256=base.sha256(path))
        prefix_path = args.output / "prefix_audit.json"
        write_json(prefix_path, inputs.prefix_audit)
        record.output("prefix_audit", prefix_path, sha256=base.sha256(prefix_path))
        source_path = args.output / "source_manifest.json"
        write_json(source_path, {path: {
            "sha256": digest, "snapshot": str(Path("code_snapshot") / path),
        } for path, digest in code.items()})
        record.output("source_manifest", source_path, sha256=base.sha256(source_path))
        record.output("progress", progress_path, sha256=base.sha256(progress_path))
        assert_unchanged(hashes, code, versions, snapshots=snapshots)
        chosen_pairs = scores["supported_pair_count"][
            np.arange(len(frames) - 1), result.path_state[:-1], result.path_state[1:]
        ]
        summary = {
            "protocol": STAGE, "technical_completion": True,
            "frames": [int(frames[0]), int(frames[-1] + 1)],
            "requested_frames": len(frames), "returned_frames": len(frames), "native_fps": 166,
            "geometry_initialization_count": 1, "external_reset_count": 0,
            "identity_initial_state": 0, "geometry_fits_in_this_stage": 0,
            "temporal_subsampling": False, "config": config, "dp": dp_scalars,
            "preparation_seconds": preparation_seconds,
            "image_scoring_seconds": image_seconds, "dp_and_export_seconds": dp_seconds,
            "identity_inference_seconds": inference_seconds,
            "timing_includes": (
                "full native image/mask decode, six image costs, motion, DP, export arrays"
            ),
            "timing_excludes": (
                "upstream geometry/D2, input verification, saving, final hashes, evaluation"
            ),
            "scored_frames": int(scores["enabled_views"].any(axis=1).sum()),
            "sampled_pixels": scores["sampled_pixels"].sum(axis=0).tolist(),
            "mapping_changes": int(np.count_nonzero(np.diff(result.path_state))),
            "nonidentity_mapping_frames": int(np.count_nonzero(result.path_state)),
            "selected_motion_supported_pair_histogram": (
                np.bincount(chosen_pairs, minlength=4).tolist()
            ),
            "raw_geometry_and_review_preserved": True,
            "code_sha256_before": code, "code_sha256_after": code,
            "input_sha256_before": hashes, "input_sha256_after": hashes,
            "code_unchanged_during_run": True, "inputs_unchanged_during_run": True,
            "evaluation_data_opened_during_inference": False,
            "limitations": [
                "Exact only for six fixed-geometry permutations and this first-order objective.",
                "Unsupported adjacent motion pairs cost zero and can favor missing connections.",
                "Templates come from provisional online labels; prefix color costs are zero.",
                "Original reviews remain; no newly confident ID output or human correction.",
                "Mapping changes may be corrections or errors, not measured ID switches.",
                "Ties and alternative gaps are model-conditional costs, not ID probabilities.",
                "Geometry duplicates, reflections and missing bodies cannot be repaired here.",
                "No female evaluation labels or full-three identity truth were read in this stage.",
            ],
        }
        summary_path = args.output / "summary.json"
        write_json(summary_path, summary)
        record.output("summary", summary_path, sha256=base.sha256(summary_path))
        assert_unchanged(hashes, code, versions, snapshots=snapshots)
        record.param("code_unchanged_during_run", True)
        record.param("inputs_unchanged_during_run", True)
    print(json.dumps({"output": str(args.output), "frames": len(frames),
                      "identity_inference_seconds": inference_seconds}), flush=True)
    return summary


if __name__ == "__main__":
    main()

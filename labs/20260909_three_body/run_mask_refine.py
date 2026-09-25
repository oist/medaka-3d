"""Compare six mask-grid starts to a sealed twelve-hypothesis v2 experiment."""

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
from mask_initialization import (  # noqa: E402
    FrameMaskVoxels,
    MaskInitializationConfig,
    build_mask_initializations,
)
from node_geometry import RayObservations  # noqa: E402
from observations import CachedNodes, PixelTargets  # noqa: E402
from run_temporal_refine import (  # noqa: E402
    CALIBRATION_DIR,
    LOSS_PART_NAMES,
    freeze_code_paths,
    load_sealed_forward,
    motion_diagnostics,
    sha256,
    validate_boundaries,
    validate_hypotheses,
)
from temporal_refine import (  # noqa: E402
    RefinementConfig,
    TemporalObjective,
    optimize_starts,
)

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402


def sealed_record(root, name, hashes):
    path = root / f"{name}.provenance.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["status"] != "ok":
        raise ValueError(f"source experiment not complete: {root}")
    def remember(source, digest):
        key = str(source.resolve())
        if key in hashes and hashes[key] != digest:
            raise ValueError(f"inconsistent sealed source digests: {source}")
        hashes[key] = digest

    remember(path, sha256(path))
    for entry in record["inputs"].values():
        source = Path(entry["path"]).resolve()
        if not entry.get("sha256") or sha256(source) != entry["sha256"]:
            raise ValueError(f"source input checksum mismatch: {source}")
        remember(source, entry["sha256"])
    for entry in record["outputs"].values():
        source = Path(entry["path"]).resolve()
        if sha256(source) != entry["sha256"]:
            raise ValueError(f"source output checksum mismatch: {source}")
        remember(source, entry["sha256"])
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--base-refinement", type=Path, required=True)
    parser.add_argument("--mask-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    input_hashes = {}
    base_record = sealed_record(args.base_refinement, "refinement", input_hashes)
    grid_record = sealed_record(args.mask_grid, "diagnostic", input_hashes)
    base_summary = json.loads((args.base_refinement / "summary.json").read_text(encoding="utf-8"))
    start, stop = base_summary["frames"]
    forward, frames, pose, observed, initial, sealed, _ = load_sealed_forward(
        args.forward, args.sequence, args.input_root, start, stop
    )
    for path, digest in sealed.items():
        resolved = str(Path(path).resolve())
        if input_hashes.get(resolved, digest) != digest:
            raise ValueError("forward and grid/refinement source inputs disagree")
        for source_record in (base_record, grid_record):
            sources = {
                str(Path(entry["path"]).resolve()): entry.get("sha256")
                for entry in source_record["inputs"].values()
            }
            if sources.get(resolved) != digest:
                raise ValueError(f"grid/refinement not bound to this forward input: {path}")
        input_hashes[resolved] = digest
    if base_summary["forward_candidate_sha256"] != sha256(args.forward / "candidate.npz"):
        raise ValueError("refinement uses another forward result")
    grid_summary = json.loads((args.mask_grid / "diagnostic.json").read_text(encoding="utf-8"))
    if grid_summary["frames"] != [start, stop] or forward["mode"] != "detectron2":
        raise ValueError("grid and forward observations must share this Detectron2 window")
    if base_summary["config"]["steps"] != 400:
        raise ValueError("the registered comparison requires the sealed 400-step base")
    config = RefinementConfig(**{**base_summary["config"], "device": args.device})
    initialization_config = MaskInitializationConfig()
    rig = load_rig(args.input_root / CALIBRATION_DIR)
    with np.load(args.base_refinement / "sampled_targets.npz", allow_pickle=False) as data:
        if not np.array_equal(data["frame_idx"], frames) or tuple(data["views"]) != VIEWS:
            raise ValueError("fixed fitting pixels do not match frames or view order")
        targets = [
            PixelTargets(
                {v: data["pixels"][t, vi].copy() for vi, v in enumerate(VIEWS)},
                data["target"][t].copy(),
                data["weights"][t].copy(),
                data["foreground"][t].copy(),
            )
            for t in range(len(frames))
        ]
    with np.load(args.base_refinement / "hypotheses.npz", allow_pickle=False) as data:
        base = {k: data[k] for k in data.files if k != "frame_idx"}
        if not np.array_equal(data["frame_idx"], frames):
            raise ValueError("base hypotheses use different frames")
    with np.load(args.mask_grid / "fixed_grid.npz", allow_pickle=False) as data:
        grid_xyz = data["xyz"]
        if tuple(data["views"]) != VIEWS:
            raise ValueError("grid camera order mismatch")
    evidence = []
    for frame in frames:
        with np.load(args.mask_grid / f"f{frame}_support.npz", allow_pickle=False) as data:
            if int(data["frame_idx"]) != frame:
                raise ValueError("grid evidence frame mismatch")
            evidence.append(
                FrameMaskVoxels(
                    int(frame),
                    data["two_view_or_more_voxel_indices"],
                    data["active_view_support"],
                    data["active_novelty_weight"],
                )
            )
    nodes = CachedNodes(args.input_root, forward["experiment"], frames)
    rays = [RayObservations.from_points(rig, {v: nodes.at(v, f) for v in VIEWS}) for f in frames]
    validate_hypotheses(base, frames, rays)
    validate_boundaries(base, pose, observed)
    code_paths = list(
        dict.fromkeys(
            [
                *freeze_code_paths(),
                Path(__file__),
                LAB / "mask_initialization.py",
                LAB / "mask_grid.py",
                LAB / "V3_PROTOCOL.md",
            ]
        )
    )
    code_hashes = {str(p.relative_to(REPO)): sha256(p) for p in code_paths}
    import torch

    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance("mask_grid_temporal_starts", args.output, name="refinement") as record:
        record.params_from(args)
        record.param("config", dataclasses.asdict(config))
        record.param("initialization_config", dataclasses.asdict(initialization_config))
        record.param("external_reset_count", 0)
        record.param("accepted_identity_output_written", False)
        record.param("code_sha256", code_hashes)
        record.param(
            "runtime",
            {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": sys.version,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            },
        )
        for path, digest in input_hashes.items():
            record.input(path, path, sha256=digest)
        for source in code_paths:
            destination = args.output / "code_snapshot" / source.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        shutil.copyfile(
            args.base_refinement / "sampled_targets.npz", args.output / "sampled_targets.npz"
        )
        record.output(
            "sampled_targets",
            args.output / "sampled_targets.npz",
            sha256=sha256(args.output / "sampled_targets.npz"),
        )
        objective = TemporalObjective(rig, rays, targets, initial, config)
        with torch.no_grad():
            original_plans = objective.assignments(base["pose_xyz"])
            repeated, _ = objective(
                objective.tensor(base["pose_xyz"]), objective.ray_tensors(original_plans)
            )
        base_cost_error = float(np.max(np.abs(repeated.cpu().numpy() - base["costs"])))
        if not np.isfinite(base_cost_error) or base_cost_error > 1e-10:
            raise ValueError(f"base objective changed: max error {base_cost_error}")
        started = time.perf_counter()
        proposed = build_mask_initializations(
            pose, observed, grid_xyz, evidence, initialization_config
        )
        initialization_seconds = time.perf_counter() - started
        starts = proposed.starts
        for name in (
            "pose_xyz",
            "permutations",
            "path_costs",
            "selected_grid_indices",
            "selected_kind",
            "scene_mapping",
            "missing_slot",
        ):
            value = np.asarray(getattr(proposed, name))
            if value.dtype.hasobject:
                raise ValueError(f"object array forbidden in initialization: {name}")
        path = args.output / "initializations.npz"
        initialization_arrays = dict(proposed.arrays)
        initialization_arrays["frame_idx"] = frames
        if any(np.asarray(value).dtype.hasobject for value in initialization_arrays.values()):
            raise ValueError("object arrays forbidden in saved finite-path candidates")
        np.savez_compressed(path, **initialization_arrays)
        record.output("initializations", path, sha256=sha256(path))
        path = args.output / "initialization_details.json"
        path.write_text(json.dumps(proposed.metadata, indent=2, allow_nan=False), encoding="utf-8")
        record.output("initialization_details", path, sha256=sha256(path))
        before_pose, before_observed = pose.copy(), observed.copy()
        started = time.perf_counter()
        result = optimize_starts(rig, pose, observed, rays, targets, initial.copy(), starts, config)
        if args.device == "cuda":
            torch.cuda.synchronize()
        refinement_seconds = time.perf_counter() - started
        if not np.array_equal(before_pose, pose) or not np.array_equal(before_observed, observed):
            raise ValueError("new starts mutated forward observations")
        validate_hypotheses(result, frames, rays, expected_kinds=("mask_grid",))
        validate_boundaries(result, pose, observed)
        path = args.output / "hypotheses.npz"
        np.savez_compressed(path, frame_idx=frames, **result)
        record.output("hypotheses", path, sha256=sha256(path))
        combined = {
            key: np.concatenate((base[key], result[key]))
            for key in (
                "pose_xyz",
                "sources",
                "observed",
                "costs",
                "loss_parts",
                "permutations",
                "start_kind",
            )
        }
        path = args.output / "combined_hypotheses.npz"
        np.savez_compressed(path, frame_idx=frames, **combined)
        record.output("combined_hypotheses", path, sha256=sha256(path))
        rankings = []
        for permutation in permutations(range(3)):
            selected = np.flatnonzero(np.all(combined["permutations"] == permutation, axis=1))
            best = int(selected[np.argmin(combined["costs"][selected])])
            rankings.append(
                {
                    "permutation": list(permutation),
                    "best_hypothesis": best,
                    "best_kind": str(combined["start_kind"][best]),
                    "best_cost": float(combined["costs"][best]),
                    "all_costs": combined["costs"][selected].tolist(),
                }
            )
        rankings.sort(key=lambda item: (item["best_cost"], item["permutation"]))
        if any(sha256(path) != value for path, value in input_hashes.items()) or (
            {str(p.relative_to(REPO)): sha256(p) for p in code_paths} != code_hashes
        ):
            raise ValueError("code or input changed during mask-start refinement")
        summary = {
            "protocol": "six_additional_mask_grid_starts",
            "frames": [start, stop],
            "native_fps": 166,
            "temporal_subsampling": False,
            "new_hypotheses": 6,
            "preserved_base_hypotheses": 12,
            "config": dataclasses.asdict(config),
            "initialization_config": dataclasses.asdict(initialization_config),
            "base_cost_recompute_max_error": base_cost_error,
            "initialization_seconds": initialization_seconds,
            "refinement_seconds": refinement_seconds,
            "permutation_rankings": rankings,
            "loss_part_names": LOSS_PART_NAMES,
            "hypothesis_diagnostics": [
                {
                    "hypothesis": i,
                    "kind": "mask_grid",
                    "permutation": result["permutations"][i].tolist(),
                    "cost": float(result["costs"][i]),
                    "loss_parts": result["loss_parts"][i].tolist(),
                    **motion_diagnostics(
                        result["pose_xyz"][i], result["observed"][i], frames, initial
                    ),
                }
                for i in range(6)
            ],
            "new_external_reset_count": 0,
            "accepted_identity_output_written": False,
            "code_unchanged_during_run": True,
            "inputs_unchanged_during_run": True,
            "limitations": [
                "All eighteen are hypotheses, not independent identity truth.",
                "Grid centres are starts, not anatomical landmark measurements.",
                "Healthy-body downweight uses current SLEAP-supported internal estimates.",
                "One missing pectoral per frame only; finite spatial pools may omit true bodies.",
            ],
        }
        path = args.output / "summary.json"
        path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=sha256(path))
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "rankings": rankings,
                    "refinement_seconds": refinement_seconds,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

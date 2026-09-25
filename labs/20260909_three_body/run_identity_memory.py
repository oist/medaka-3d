"""Full frozen-geometry ID inference with last-direct-observation memory.

The original adjacent-only producer remains frozen. Its validated image reader,
sampling and source-preservation helpers are reused; the new search retains
direct observation events through missing intervals. Evaluation is separate.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import time
from pathlib import Path

import numpy as np
import run_identity_dp as common
from identity_memory_dp import solve_identity_memory_dp
from identity_memory_links import chosen_memory_links

base, torch = common.base, common.torch
STAGE = "three_body_fixed_geometry_identity_memory_dp"


def registered_config():
    return {
        **common.registered_config(),
        "memory": {
            "model": "brownian_direct_event", "max_layer_states": 100000,
            "max_total_states": 1000000, "max_total_edges": 6000000,
        },
    }


def code_paths():
    return sorted(set(common.code_paths()) | {
        Path(__file__).resolve(), base.LAB / "IDENTITY_MEMORY_PROTOCOL.md",
    })


def optimize_memory(geometry, scores, config):
    if not np.array_equal(scores["frame_idx"], geometry["frame_idx"]):
        raise ValueError("memory DP scores and geometry frames differ")
    memory = config["memory"]
    if memory["model"] != "brownian_direct_event":
        raise ValueError("unregistered observation-memory model")
    result = solve_identity_memory_dp(
        geometry["frame_idx"], scores["unary"], geometry["xyz"],
        geometry["observed"][:, :, 1], scale_mm=config["motion_scale_mm"],
        weight=config["motion_weight"], initial_state=config["initial_state"],
        tie_atol=config["tie_atol"], tie_rtol=config["tie_rtol"],
        **{key: memory[key] for key in ("max_layer_states", "max_total_states", "max_total_edges")},
    )
    if result.status != "optimal":
        raise ValueError("registered finite image/memory objective did not complete optimally")
    identity = common.build_identity_output(geometry, result.permutation_by_id)
    links = chosen_memory_links(
        geometry, result.permutation_by_id,
        scale_mm=config["motion_scale_mm"], weight=config["motion_weight"],
    )
    independently_summed = float(
        scores["unary"][np.arange(len(geometry["frame_idx"])), result.path_state].sum()
        + links["brownian_motion_cost_by_id"].sum()
    )
    if not np.isclose(result.path_cost, independently_summed, rtol=1e-11, atol=1e-10):
        raise ValueError("memory DP cost differs from independent chosen direct-event links")
    return result, identity, links


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
        raise RuntimeError("registered memory experiment uses CUDA image scoring")
    torch.set_num_threads(2)
    config, sources, versions = registered_config(), code_paths(), common.package_versions()
    code = {str(path.relative_to(base.REPO)): base.sha256(path) for path in sources}
    args.output.mkdir(parents=True, exist_ok=False)
    with common.provenance(STAGE, args.output, name="experiment") as record:
        record.params_from(args)
        record.param("config", config)
        record.param("code_hashes", code)
        record.param("package_versions", versions)
        record.param("evaluation_data_opened_during_inference", False)
        record.param("torch_cpu_threads", torch.get_num_threads())
        prepared = time.perf_counter()
        inputs = common.load_identity_inputs(
            args.geometry_run, args.enrollment_run, args.enrollment_audit
        )
        hashes, snapshots = inputs.verified_hashes, {}
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path in sources:
            relative = path.relative_to(base.REPO)
            target = args.output / "code_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if base.sha256(target) != code[str(relative)]:
                raise RuntimeError("source changed while snapshotting")
            snapshots[str(target)] = code[str(relative)]
            record.output(str(target.relative_to(args.output)), target, sha256=code[str(relative)])
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

            scores = common.collect_scores(
                reader, rig, inputs.geometry, inputs.enrollment, config,
                device="cuda", progress=progress,
            )
            torch.cuda.synchronize()
        image_seconds = time.perf_counter() - inference_started
        dp_started = time.perf_counter()
        result, identity, links = optimize_memory(inputs.geometry, scores, config)
        dp_seconds = time.perf_counter() - dp_started
        inference_seconds = time.perf_counter() - inference_started
        fields = {field.name: getattr(result, field.name) for field in dataclasses.fields(result)}
        dp_arrays = {key: value for key, value in fields.items() if isinstance(value, np.ndarray)}
        dp_scalars = {
            key: common.json_scalar(value) for key, value in fields.items() if key not in dp_arrays
        }
        for role, arrays in (
            ("identity", identity), ("dp", dp_arrays), ("scores", scores), ("memory_links", links)
        ):
            path = args.output / f"{role}.npz"
            np.savez_compressed(path, **arrays)
            record.output(role, path, sha256=base.sha256(path))
        prefix_path = args.output / "prefix_audit.json"
        common.write_json(prefix_path, inputs.prefix_audit)
        record.output("prefix_audit", prefix_path, sha256=base.sha256(prefix_path))
        source_path = args.output / "source_manifest.json"
        common.write_json(source_path, {path: {
            "sha256": digest, "snapshot": str(Path("code_snapshot") / path),
        } for path, digest in code.items()})
        record.output("source_manifest", source_path, sha256=base.sha256(source_path))
        record.output("progress", progress_path, sha256=base.sha256(progress_path))
        common.assert_unchanged(hashes, code, versions, snapshots=snapshots)
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
                "native image/mask decode, six color scores, memory DP, export arrays"
            ),
            "timing_excludes": (
                "upstream geometry/D2, input verification, saving, hashes, evaluation"
            ),
            "scored_frames": int(scores["enabled_views"].any(axis=1).sum()),
            "sampled_pixels": scores["sampled_pixels"].sum(axis=0).tolist(),
            "mapping_changes": int(np.count_nonzero(np.diff(result.path_state))),
            "nonidentity_mapping_frames": int(np.count_nonzero(result.path_state)),
            "reobservation_links": int(links["motion_cost_applied"].sum()),
            "terminal_unobserved_age_frames": links["terminal_unobserved_age_frames"].tolist(),
            "terminal_unsettled_gap": links["terminal_unsettled_gap"].tolist(),
            "raw_geometry_and_review_preserved": True,
            "code_sha256_before": code, "code_sha256_after": code,
            "input_sha256_before": hashes, "input_sha256_after": hashes,
            "code_unchanged_during_run": True, "inputs_unchanged_during_run": True,
            "evaluation_data_opened_during_inference": False,
            "limitations": [
                "Exact over the bounded completed memory-state graph and fixed geometry only.",
                "Resource overflow raises an error; there is no automatic beam approximation.",
                "Brownian motion is unvalidated for these fish and has no visibility model.",
                "Unreturned terminal gaps have no later motion evidence and remain unresolved.",
                "Causal color labels are provisional; initial166 color scores are zero.",
                "Original review and NaN remain; there is no human correction or ID certification.",
                "Mapping changes and conditional cost gaps are not measured biological ID errors.",
                "Fixed duplicate or absent bodies cannot be repaired by label assignment.",
            ],
        }
        summary_path = args.output / "summary.json"
        common.write_json(summary_path, summary)
        record.output("summary", summary_path, sha256=base.sha256(summary_path))
        common.assert_unchanged(hashes, code, versions, snapshots=snapshots)
        for entry in record.outputs.values():
            if base.sha256(entry["path"]) != entry["sha256"]:
                raise RuntimeError("saved memory inference output changed before final seal")
        record.param("code_unchanged_during_run", True)
        record.param("inputs_unchanged_during_run", True)
    print(json.dumps({"output": str(args.output), "frames": len(frames),
                      "identity_inference_seconds": inference_seconds}), flush=True)
    return summary


if __name__ == "__main__":
    main()

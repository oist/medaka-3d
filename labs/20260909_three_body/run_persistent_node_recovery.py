"""Persistent mouth-recovery algorithm arm; frozen v1 inputs and one registered seed.

The original proposal set remains intact. Up to six mouth-only alternatives are
added, while the validated CUDA Graph changes only fit execution. Completed v1
results, baseline trajectories and human labels are opened after tracking.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import shutil
import time
from pathlib import Path

import numpy as np
import run_experiment as base

from medaka3d.provenance import provenance

FULL_WINDOWS = {
    (333519, 333719): "v1_transient_detectron2",
    (76620, 77286): "v1_interval_detectron2",
    (76620, 86580): "v1_continuous_one_minute_detectron2",
}
ORIGINAL_ARRAYS = {
    "frame_idx",
    "pose_xyz",
    "observed",
    "sources",
    "identity_review",
    "mask_supported",
    "body_supported",
    "missing",
    "costs",
    "path_costs",
    "loss_parts",
    "score_margin",
    "proposal_kind",
    "hypotheses",
    "proposal_flow_nodes",
    "proposal_flow_parent",
    "xyz",
    "accepted_xyz",
    "body_xyz",
}
ALGORITHM_ARM = "v1_plus_persistent_mouth_recovery_graph"


def full_registered_frames(manifest):
    frames = base.sequence_frames(manifest)
    bounds = int(frames[0]), int(frames[-1] + 1)
    if bounds not in FULL_WINDOWS:
        raise ValueError("only complete registered 200, 666 or 9960-frame windows are allowed")
    return frames, FULL_WINDOWS[bounds]


def normalized_hashes(values):
    result = {}
    for name, digest in values.items():
        path = Path(name)
        path = (path if path.is_absolute() else base.REPO / path).resolve(strict=True)
        key = str(path)
        if key in result and result[key] != digest:
            raise ValueError("conflicting resolved input digests")
        result[key] = digest
    return result


def validate_arm_arrays(result, frames):
    if not ORIGINAL_ARRAYS.issubset(result):
        raise ValueError("new arm must preserve all original 19 result arrays")
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    if not ORIGINAL_ARRAYS.issubset(arrays) or any(
        value.dtype.hasobject for value in arrays.values()
    ):
        raise ValueError("original arrays must remain non-object NumPy arrays")
    if any(value.ndim < 1 or len(value) != len(frames) for value in arrays.values()):
        raise ValueError("every saved result array must cover the full native window")
    hypotheses = arrays["hypotheses"]
    if hypotheses.shape != (len(frames),) or hypotheses.dtype.kind not in "iu":
        raise ValueError("hypotheses must be one integer count per native frame")
    if np.any(hypotheses < 1) or np.any(hypotheses > 24):
        raise ValueError("new arm must retain at most 18 original plus 6 mouth proposals")
    validate_node_ages(arrays)
    return arrays


def validate_node_ages(arrays):
    """Ages follow the returned native backtrace, never another beam parent."""
    observed = arrays["observed"]
    node_age = arrays.get("node_missing")
    pectoral_age = arrays.get("direct_pectoral_age")
    if (
        node_age is None or pectoral_age is None
        or node_age.shape != observed.shape
        or pectoral_age.shape != observed.shape[:2]
        or node_age.dtype.kind not in "iu"
        or pectoral_age.dtype.kind not in "iu"
    ):
        raise ValueError("explicit per-parent node age arrays are required")
    previous_node = np.zeros((3, 3), np.int64)
    previous_pectoral = np.zeros(3, np.int64)
    for index, held in enumerate(observed):
        previous_node = np.where(held, 0, previous_node + 1)
        previous_pectoral = np.where(held[:, 1], previous_pectoral + 1, 0)
        if not np.array_equal(node_age[index], previous_node) or not np.array_equal(
            pectoral_age[index], previous_pectoral
        ):
            raise ValueError("node ages do not follow strict support on the saved backtrace")


def code_paths():
    additions = (
        "run_persistent_node_recovery.py",
        "PERSISTENT_NODE_PROTOCOL.md",
        "node_recovery.py",
        "persistent_node_tracker.py",
        "accelerated_tracker.py",
        "graph_fit.py",
        "graph_union.py",
    )
    return list(
        dict.fromkeys([*base.frozen_code_paths(), *(base.LAB / name for name in additions)])
    )


def compare_completed_v1(root, frames, initial, seed, input_hashes, config, anchors, rig, nodes):
    """Only called below the single tracker invocation and candidate save boundary."""
    paths = {
        name: root / name
        for name in ("experiment.provenance.json", "summary.json", "candidate.npz")
    }
    hashes = {str(path): base.sha256(path) for path in paths.values()}
    record = json.loads(paths["experiment.provenance.json"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
    if record.get("status") != "ok" or summary.get("technical_completion") is not True:
        raise ValueError("reference v1 run is not complete")
    for key, name in (("candidate", "candidate.npz"), ("summary", "summary.json")):
        if hashes[str(paths[name])] != record["outputs"][key]["sha256"]:
            raise ValueError("reference v1 output differs from its completed seal")
    if (
        summary["frames"] != [int(frames[0]), int(frames[-1] + 1)]
        or summary["returned_frames"] != len(frames)
        or summary["initialization_count"] != 1
        or summary["reset_count"] != 0
        or summary["track_invocation_count"] != 1
        or summary["development_prefix_smoke"] is not False
        or summary["temporal_subsampling"] is not False
        or summary["native_fps"] != 166
    ):
        raise ValueError("reference v1 must be the same full native single-initialization run")
    if summary["seed"] != seed or summary["tracker_config"] != dataclasses.asdict(config):
        raise ValueError("new arm changes the frozen v1 seed or base tracker configuration")
    if normalized_hashes(summary["input_sha256_before"]) != normalized_hashes(
        input_hashes
    ) or normalized_hashes(summary["input_sha256_after"]) != normalized_hashes(input_hashes):
        raise ValueError("new arm and completed v1 do not share every inference input path and SHA")
    with np.load(paths["candidate.npz"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    base.validate_result(arrays, frames, nodes)
    if not ORIGINAL_ARRAYS.issubset(arrays):
        raise ValueError("reference v1 lacks original result arrays")
    metrics = base.candidate_metrics(arrays, frames, initial, anchors, rig)
    if metrics != summary["candidate"]:
        raise ValueError("shared re-evaluation differs from frozen v1 metrics")
    return {
        "path": str(root),
        "candidate_sha256": hashes[str(paths["candidate.npz"])],
        "same_seed_config_and_all_inference_inputs": True,
        "shared_metrics_exact_to_original_summary": True,
        "candidate": metrics,
    }, hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("new algorithm arm requires a new output directory")
    args.sequence = args.sequence.resolve(strict=True)
    args.input_root = args.input_root.resolve(strict=True)
    sequence_path = args.sequence / "manifest.json"
    manifest = json.loads(sequence_path.read_text(encoding="utf-8"))
    frames, reference_name = full_registered_frames(manifest)
    initial, seed, evaluation = base.registered_seed(int(frames[0]))
    catalog, staging_manifest = base.staging_catalog(args.input_root)
    inference_inputs = {}
    for name in (
        "calibration_manifest.json",
        *(f"cam{i}_bundle_refractive.json" for i in range(3)),
    ):
        relative = f"{base.CALIBRATION_DIR}/{name}"
        inference_inputs[relative] = base.verified_input(args.input_root, relative, catalog)
    for view in base.VIEWS:
        relative = f"work/c01_s16_predictions/bottomup/{base.EXPERIMENT}__{view}.npz"
        inference_inputs[relative] = base.verified_input(args.input_root, relative, catalog)
    input_hashes = {str(path): base.sha256(path) for path in inference_inputs.values()}
    input_hashes.update(
        {
            str(sequence_path): base.sha256(sequence_path),
            str(staging_manifest): base.sha256(staging_manifest),
        }
    )
    for category in ("raw", "masks", "backgrounds"):
        for view in base.VIEWS:
            entry = manifest[category][view]
            path = base.resolve_input(args.sequence, entry["path"])
            digest = base.sha256(path)
            if digest != entry["sha256"]:
                raise ValueError("sequence asset differs from its declared SHA")
            input_hashes[str(path)] = digest
    import cv2
    import scipy
    import torch
    from motion import FlowConfig
    from node_recovery import NodeRecoveryConfig
    from persistent_node_tracker import AcceleratedPersistentNodeTracker, PersistentNodeConfig
    from tracker import TrackerConfig

    if not torch.cuda.is_available():
        raise RuntimeError("this fixed Graph arm requires CUDA; no automatic backend change")
    torch.set_num_threads(2)
    config = TrackerConfig(mode="detectron2", device="cuda", flow_proposals=True)
    recovery_config = NodeRecoveryConfig()
    persistence_config = PersistentNodeConfig()
    sources = code_paths()
    code_before = {str(path.relative_to(base.REPO)): base.sha256(path) for path in sources}
    evaluation_hashes = {}
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance(
        "three_body_persistent_node_recovery_algorithm_arm", args.output, name="experiment"
    ) as record:
        record.params_from(args)
        record.param("algorithm_arm", ALGORITHM_ARM)
        record.param("tracker_config", dataclasses.asdict(config))
        record.param("node_recovery_config", dataclasses.asdict(recovery_config))
        record.param("persistence_config", dataclasses.asdict(persistence_config))
        record.param("flow_config", dataclasses.asdict(FlowConfig()))
        record.param("seed", seed)
        record.param("code_hashes", code_before)
        record.param("base_candidates", 18)
        record.param("additional_mouth_candidates_max", 6)
        record.param("old_results_and_gt_opened_only_after_tracking", True)
        record.param("torch_cpu_threads", torch.get_num_threads())
        record.param(
            "package_versions",
            {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "opencv": cv2.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_device_name": torch.cuda.get_device_name(0),
            },
        )
        record.input("seed_registration", base.REGISTRATION, sha256=base.REGISTRATION_SHA256)
        for path, digest in input_hashes.items():
            record.input(path, path, sha256=digest)
        for path in sources:
            target = args.output / "code_snapshot" / path.relative_to(base.REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for name, path in (
            ("sequence_manifest.json", sequence_path),
            ("staging_manifest.json", staging_manifest),
        ):
            shutil.copyfile(path, args.output / name)
        for relative, path in inference_inputs.items():
            if relative.startswith(base.CALIBRATION_DIR):
                target = args.output / "calibration_snapshot" / path.name
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(path, target)
        seed_path = args.output / "seed_registration.json"
        seed_path.write_text(json.dumps(seed, indent=2, allow_nan=False), encoding="utf-8")
        np.save(args.output / "initial_pose_xyz.npy", initial, allow_pickle=False)
        record.output("seed_registration", seed_path, sha256=base.sha256(seed_path))
        preparation_started = time.perf_counter()
        rig = base.load_rig(args.input_root / base.CALIBRATION_DIR)
        reader = base.SequenceReader(args.sequence, args.input_root)
        preparation_seconds = time.perf_counter() - preparation_started
        observations = iter(reader)
        progress_path = args.output / "progress.jsonl"
        processed = 0
        started = time.perf_counter()
        with progress_path.open("x", encoding="utf-8", buffering=1) as progress:

            def inputs():
                nonlocal processed
                for index, observation in enumerate(observations):
                    if index >= len(frames) or observation.frame_idx != frames[index]:
                        raise ValueError("reader yielded an extra or incorrect native frame")
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
                tracker = AcceleratedPersistentNodeTracker(
                    rig, initial.copy(), config=config, recovery_config=recovery_config,
                    persistence_config=persistence_config,
                )
                result = tracker.track(inputs())
                torch.cuda.synchronize()
            finally:
                observations.close()
        tracking_seconds = time.perf_counter() - started
        if processed != len(frames) or tracker.initialize_count != 1:
            raise ValueError("one initialization must consume the complete registered window")
        base.validate_result(result, frames, reader.nodes)
        arrays = validate_arm_arrays(result, frames)
        candidate_path = args.output / "candidate.npz"
        np.savez_compressed(candidate_path, **arrays)
        record.output("candidate", candidate_path, sha256=base.sha256(candidate_path))
        # Evaluation-only boundary: human marks, completed v1, and C34 arrays are first opened here.
        gt_path = base.verified_input(
            args.input_root, evaluation["female_annotations_relative"], catalog
        )
        if base.sha256(gt_path) != evaluation["female_annotations_sha256"]:
            raise ValueError("GT differs from registered source")
        evaluation_hashes[str(gt_path)] = base.sha256(gt_path)
        with gt_path.open(encoding="utf-8-sig", newline="") as stream:
            anchors = base.match_mouth_anchors(
                csv.DictReader(stream),
                lambda view, frame: reader.nodes.at(view, frame)[:, 0],
                int(frames[0]),
                int(frames[-1] + 1),
            )
        candidate = base.candidate_metrics(result, frames, initial, anchors, rig)
        reference, reference_hashes = compare_completed_v1(
            base.REPO / "work/three_body" / reference_name,
            frames,
            initial,
            seed,
            input_hashes,
            config,
            anchors,
            rig,
            reader.nodes,
        )
        evaluation_hashes.update(reference_hashes)
        baselines = base.baseline_metrics(
            args.input_root, catalog, frames, rig, reader.nodes, anchors, seed
        )
        evaluation_hashes.update({arm["path"]: arm["sha256"] for arm in baselines.values()})
        for path, digest in evaluation_hashes.items():
            record.input("post_inference:" + path, path, sha256=digest)
        code_after = {str(path.relative_to(base.REPO)): base.sha256(path) for path in sources}
        inputs_after = {path: base.sha256(path) for path in input_hashes}
        if code_before != code_after or input_hashes != inputs_after:
            raise RuntimeError("code or inference input changed; arm cannot seal")
        diagnostics = tracker.node_recovery_diagnostics
        summary = {
            "protocol": "three_body_persistent_node_recovery_algorithm_arm",
            "algorithm_arm": ALGORITHM_ARM,
            "experiment": base.EXPERIMENT,
            "n_groups": 1,
            "mode": "detectron2",
            "window": manifest.get("window", "registered_full_window"),
            "frames": [int(frames[0]), int(frames[-1] + 1)],
            "source_sequence_frames": [int(frames[0]), int(frames[-1] + 1)],
            "requested_frames": len(frames),
            "returned_frames": len(result["frame_idx"]),
            "native_fps": 166,
            "temporal_subsampling": False,
            "development_prefix_smoke": False,
            "technical_completion": True,
            "initialization_count": 1,
            "track_invocation_count": 1,
            "reset_count": 0,
            "reset_count_scope": "external seeds; mouth candidates are internal",
            "preparation_seconds": preparation_seconds,
            "tracking_seconds": tracking_seconds,
            "tracking_fps": len(frames) / tracking_seconds,
            "tracking_timing_includes": (
                "local native decode, packed-mask decode, proposals and Graph tracking"
            ),
            "tracking_timing_excludes": "D2 staging, input hashes and post-inference evaluation",
            "precomputed_sequence_staging_seconds": manifest.get("staging_elapsed_s"),
            "tracker_config": dataclasses.asdict(config),
            "node_recovery_config": dataclasses.asdict(recovery_config),
            "persistence_config": dataclasses.asdict(persistence_config),
            "node_recovery_diagnostics": diagnostics,
            "seed": seed,
            "candidate": candidate,
            "original_v1_reference": reference,
            "baseline_arms": baselines,
            "code_unchanged_during_run": True,
            "inputs_unchanged_during_run": True,
            "code_sha256_before": code_before,
            "code_sha256_after": code_after,
            "input_sha256_before": input_hashes,
            "input_sha256_after": inputs_after,
            "output_array_shapes": {key: list(value.shape) for key, value in arrays.items()},
            "original_19_arrays_preserved": True,
            "limitations": [
                "New persistent-mouth arm; four prior mouth-missing/direct-pectoral frames.",
                "Same opened development windows; no independent held-out identity evaluation.",
                "One optimistic first seed from the registered whole-recording baseline.",
                "Completed v1, C34 trajectories and GT were opened only after inference returned.",
                "Mouth support, mask fit, availability and margins are not ID correctness.",
                "Female endpoints cannot establish male identity or exclude returning swaps.",
                "Review is never manually cleared and no human correction is applied.",
                "Graph fit equivalence does not imply unchanged paths with added proposals.",
                "Only new-arm tracking runtime is measured; baseline timings are not rerun.",
            ],
        }
        summary_path = args.output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", summary_path, sha256=base.sha256(summary_path))
        record.output("progress", progress_path, sha256=base.sha256(progress_path))
        if (
            any(
                base.sha256(path) != digest
                for path, digest in {**input_hashes, **evaluation_hashes}.items()
            )
            or any(base.sha256(base.REPO / path) != digest for path, digest in code_before.items())
            or base.sha256(base.REGISTRATION) != base.REGISTRATION_SHA256
        ):
            raise RuntimeError("input or code changed while saving; arm cannot seal")
        record.param("code_unchanged_during_run", True)
        record.param("inputs_unchanged_during_run", True)
    print(
        json.dumps(
            {
                "algorithm_arm": ALGORITHM_ARM,
                "output": str(args.output),
                "native_frames": len(frames),
                "tracking_seconds": tracking_seconds,
            }
        )
    )
    return summary


if __name__ == "__main__":
    main()

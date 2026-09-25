"""Run the new CUDA Graph fitter once on 200 native frames, then compare frozen v1.

This command has one fixed algorithm/configuration. No old source is overwritten.
Later archived positions are opened only after the accelerated tracker returns;
human identity annotations are not opened. Equivalence is not tracking quality.
"""

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy
import torch

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from accelerated_tracker import AcceleratedThreeBodyTracker  # noqa: E402
from motion import FlowConfig  # noqa: E402
from observations import SequenceReader  # noqa: E402
from run_experiment import (  # noqa: E402
    CALIBRATION_DIR,
    EXPERIMENT,
    frozen_code_paths,
    registered_seed,
    sha256,
    staging_catalog,
    validate_result,
    verified_input,
)
from tracker import TrackerConfig  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402


def compare_arrays(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("output array names differ")
    comparisons = {}
    for key, value in actual.items():
        reference = expected[key]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            comparisons[key] = dict(exact=False, reason="shape or dtype differs")
            continue
        floating = np.issubdtype(value.dtype, np.inexact)
        equal = np.array_equal(value, reference, equal_nan=True) if floating else np.array_equal(
            value, reference)
        row = dict(exact=bool(equal), shape=list(value.shape), dtype=str(value.dtype))
        if floating:
            both_finite = np.isfinite(value) & np.isfinite(reference)
            row["maximum_finite_difference"] = float(np.max(
                np.abs(value[both_finite] - reference[both_finite]), initial=0))
            row["same_finite_mask"] = bool(np.array_equal(np.isfinite(value),
                                                        np.isfinite(reference)))
            row["same_nan_and_infinity"] = bool(
                np.array_equal(np.isnan(value), np.isnan(reference))
                and np.array_equal(np.isposinf(value), np.isposinf(reference))
                and np.array_equal(np.isneginf(value), np.isneginf(reference)))
        else:
            row["different_elements"] = int(np.count_nonzero(value != reference))
        comparisons[key] = row
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sequence = REPO / "work/three_body/inputs/transient_333519-333719"
    input_root = REPO / "work/generative_tracking_inputs"
    sealed = REPO / "work/three_body/v1_transient_detectron2"
    config = TrackerConfig(flow_proposals=True)
    torch.set_num_threads(2)
    code = [Path(__file__), *frozen_code_paths(), *(LAB / name for name in (
        "graph_union.py", "graph_fit.py", "accelerated_tracker.py"))]
    hashes = {str(p.relative_to(REPO)): sha256(p) for p in code}
    with provenance("three_body_graph_fit_single_initialization", args.output,
                    name="experiment") as record:
        record.param("tracker_config", dataclasses.asdict(config))
        record.param("flow_config", dataclasses.asdict(FlowConfig()))
        record.param("source_hashes", hashes)
        record.param("torch_cpu_threads", torch.get_num_threads())
        record.param("no_baseline_or_gt_used_during_inference", True)
        record.param("only_override", "ThreeBodyTracker._fit")
        versions = dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                        opencv=cv2.__version__, gpu=torch.cuda.get_device_name())
        record.param("runtime", versions)
        initial, seed, _ = registered_seed(333519)
        record.param("seed", seed)
        catalog, staging = staging_catalog(input_root)
        paths = {"sequence_manifest": sequence / "manifest.json", "staging_manifest": staging,
                 "seed_registration": Path(seed["registration_path"])}
        for name in ("calibration_manifest.json", *(f"cam{i}_bundle_refractive.json"
                                                   for i in range(3))):
            relative = f"{CALIBRATION_DIR}/{name}"
            paths[relative] = verified_input(input_root, relative, catalog)
        for view in VIEWS:
            relative = f"work/c01_s16_predictions/bottomup/{EXPERIMENT}__{view}.npz"
            paths[relative] = verified_input(input_root, relative, catalog)
        input_hashes = {str(path): sha256(path) for path in paths.values()}
        for name, path in paths.items():
            record.input(name, path, sha256=input_hashes[str(path)])
        for path in code:
            destination = args.output / "code_snapshot" / path.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        for source in paths.values():
            if source.parent == input_root / CALIBRATION_DIR:
                target = args.output / "calibration_snapshot" / source.name
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, target)
        (args.output / "seed_registration.json").write_text(
            json.dumps(seed, indent=2, allow_nan=False), encoding="utf-8")
        reader = SequenceReader(sequence, input_root)
        np.testing.assert_array_equal(reader.frames, np.arange(333519, 333719))
        for category in ("raw", "masks", "backgrounds"):
            for view in VIEWS:
                entry = reader.manifest[category][view]
                path = sequence / entry["path"]
                input_hashes[str(path)] = entry["sha256"]
                record.input(category + "_" + view, path, sha256=entry["sha256"])
        rig = load_rig(input_root / CALIBRATION_DIR)
        tracker = AcceleratedThreeBodyTracker(rig, initial, config)
        start = time.perf_counter()
        observations = iter(reader)
        processed = 0
        with (args.output / "progress.jsonl").open("x", encoding="utf-8", buffering=1) as log:

            def stream():
                nonlocal processed
                for index, frame in enumerate(observations):
                    if frame.frame_idx != reader.frames[index]:
                        raise ValueError("native frame alignment changed")
                    yield frame
                    processed += 1
                    if processed % 25 == 0:
                        event = dict(processed=processed, source_frame=frame.frame_idx,
                                     elapsed_seconds=time.perf_counter() - start)
                        log.write(json.dumps(event) + "\n")
                        print(json.dumps(event), flush=True)

            try:
                result = tracker.track(stream())
                torch.cuda.synchronize()
            finally:
                observations.close()
        elapsed = time.perf_counter() - start
        if processed != 200 or tracker.initialize_count != 1:
            raise ValueError("tracking was not one uninterrupted 200-frame initialization")
        validate_result(result, reader.frames, reader.nodes)
        actual = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
        candidate = args.output / "candidate.npz"
        np.savez_compressed(candidate, **actual)
        record.output("candidate", candidate, sha256=sha256(candidate))

        # Explicit post-inference comparison boundary: no archived pose was opened above.
        reference_path = sealed / "candidate.npz"
        with np.load(reference_path, allow_pickle=False) as arrays:
            expected = {key: arrays[key] for key in arrays.files}
        previous_summary = json.loads((sealed / "summary.json").read_text(encoding="utf-8"))
        if previous_summary["tracker_config"] != dataclasses.asdict(config):
            raise ValueError("sealed v1 configuration differs from fixed accelerated config")
        comparisons = compare_arrays(actual, expected)
        source_after = {str(p.relative_to(REPO)): sha256(p) for p in code}
        if source_after != hashes:
            raise RuntimeError("frozen source changed during accelerated experiment")
        all_exact = all(row["exact"] for row in comparisons.values())
        summary = dict(status="ok", all_arrays_exact=all_exact,
                       initialization_count=tracker.initialize_count, frame_count=processed,
                       source_frame_start=333519, source_frame_stop_exclusive=333719,
                       tracking_seconds=elapsed, tracker_config=dataclasses.asdict(config),
                       runtime=versions, code_hashes=hashes, source_unchanged=True,
                       input_hashes=input_hashes, candidate_sha256=sha256(candidate),
                       reference_opened_only_after_inference=True,
                       reference_path=str(reference_path), reference_sha256=sha256(reference_path),
                       comparisons=comparisons, cached_shapes=[str(k[:2])
                                                              for k in tracker._fit_graph_cache],
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       limitations=["Concurrent D2 staging: elapsed time is descriptive.",
                                    "Identical output preserves both successes and failures.",
                                    "No all-three-fish identity ground truth was used or scored.",
                                    "No restart/resume occurred; this is 200 native frames."])
        summary_path = args.output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.input("v1_post_inference_comparison", reference_path, sha256=sha256(reference_path))
        record.output("summary", summary_path, sha256=sha256(summary_path))
        print(json.dumps(dict(all_arrays_exact=all_exact, tracking_seconds=elapsed,
                              mismatches=[key for key, row in comparisons.items()
                                          if not row["exact"]])), flush=True)


if __name__ == "__main__":
    main()

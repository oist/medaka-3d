"""Paired CPU-only OpenCV thread experiment on the already-fixed twenty flow pairs."""

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

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from motion import FlowConfig, FlowObservations  # noqa: E402
from observations import SequenceReader  # noqa: E402
from profile_flow_projection_timer import sha256  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixed = REPO / "work/three_body/flow_projection_timer_v1"
    fixed_metadata = json.loads((fixed / "results.json").read_text(encoding="utf-8"))
    fixed_arrays = fixed / "inputs_and_outputs.npz"
    if sha256(fixed_arrays) != fixed_metadata["input_and_output_sha256"]:
        raise ValueError("fixed flow baseline changed")
    with np.load(fixed_arrays, allow_pickle=False) as data:
        baseline = {key: data[key] for key in data.files}
    code = [Path(__file__), REPO / "work/three_body/profile_flow_projection_timer.py",
            LAB / "motion.py", LAB / "observations.py", LAB / "prepare_sequence.py",
            REPO / "src/medaka3d/calib.py"]
    hashes = {str(path.relative_to(REPO)): sha256(path) for path in code}
    for path in code:
        recorded = fixed_metadata["source_hashes"].get(str(path.relative_to(REPO)))
        if recorded is not None and recorded != sha256(path):
            raise ValueError("source differs from fixed projection-timer baseline")
        target = args.output / "code_snapshot" / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    sequence = REPO / "work/three_body/inputs/transient_333519-333719"
    calibration = REPO / "work/three_body/graph_accelerated_transient_200_v1/calibration_snapshot"
    rig = load_rig(calibration)
    reader = SequenceReader(sequence, REPO / "work/generative_tracking_inputs")
    wanted = set(map(int, baseline["previous_frame"])) | set(map(int, baseline["frame"]))
    frames = {}
    iterator = iter(reader)
    try:
        for observation in iterator:
            if observation.frame_idx in wanted:
                frames[observation.frame_idx] = observation
    finally:
        iterator.close()
    if set(frames) != wanted:
        raise ValueError("missing one of the fixed image pairs")
    cfg = FlowConfig(**fixed_metadata["config"])
    original_threads = cv2.getNumThreads()
    conditions = (1, 2, 4, 32)
    orders = (conditions, conditions[::-1])
    fields = dataclasses.fields(FlowObservations)
    trials, saved = [], {field.name: [] for field in fields}
    try:
        for repeat, order in enumerate(orders):
            for threads in order:
                cv2.setNumThreads(threads)
                if cv2.getNumThreads() != threads:
                    raise RuntimeError("OpenCV did not apply requested thread count")
                for pair, (before, after) in enumerate(zip(
                    baseline["previous_frame"], baseline["frame"], strict=True)):
                    start = time.perf_counter()
                    result = FlowObservations.from_frames(
                        frames[int(before)], frames[int(after)], rig,
                        baseline["previous_pose_xyz"][pair], config=cfg)
                    elapsed = time.perf_counter() - start
                    comparison = {}
                    for field in fields:
                        actual = np.asarray(getattr(result, field.name))
                        expected = baseline["original_" + field.name][pair]
                        same_schema = (actual.dtype == expected.dtype
                                       and actual.shape == expected.shape)
                        floating = np.issubdtype(actual.dtype, np.inexact)
                        equal = (np.array_equal(actual, expected, equal_nan=True) if floating
                                 else np.array_equal(actual, expected))
                        maximum = 0.
                        if floating and same_schema:
                            finite = np.isfinite(actual) & np.isfinite(expected)
                            maximum = float(np.max(np.abs(actual[finite] - expected[finite]),
                                                   initial=0))
                        comparison[field.name] = dict(exact=bool(same_schema and equal),
                                                       maximum_finite_difference=maximum)
                        saved[field.name].append(actual)
                    trials.append(dict(repeat=repeat, threads=threads,
                                       actual_threads=cv2.getNumThreads(), pair=pair,
                                       previous_frame=int(before), frame=int(after),
                                       seconds=elapsed, fields=comparison,
                                       all_fields_exact=all(row["exact"]
                                                            for row in comparison.values())))
                group = trials[-20:]
                print(json.dumps(dict(repeat=repeat, threads=threads,
                                      median_seconds=float(np.median(
                                          [r["seconds"] for r in group])),
                                      all_fields_exact=all(r["all_fields_exact"] for r in group))),
                      flush=True)
    finally:
        cv2.setNumThreads(original_threads)
    restored = cv2.getNumThreads()
    if restored != original_threads:
        raise RuntimeError("OpenCV thread count was not restored")
    source_after = {str(path.relative_to(REPO)): sha256(path) for path in code}
    if source_after != hashes:
        raise RuntimeError("source changed during thread experiment")
    baseline_times = {(r["repeat"], r["pair"]): r["seconds"]
                      for r in trials if r["threads"] == 32}
    summaries = {}
    for threads in conditions:
        selected = [r for r in trials if r["threads"] == threads]
        ratios = [baseline_times[(r["repeat"], r["pair"])] / r["seconds"] for r in selected]
        summaries[str(threads)] = dict(
            trials=len(selected), median_seconds=float(np.median([r["seconds"] for r in selected])),
            total_seconds=sum(r["seconds"] for r in selected),
            paired_median_32_over_condition=float(np.median(ratios)),
            all_15_fields_exact=all(r["all_fields_exact"] for r in selected),
            max_finite_difference=max(info["maximum_finite_difference"]
                                      for r in selected for info in r["fields"].values()),
            repeats={str(repeat): dict(
                median_seconds=float(np.median([r["seconds"] for r in selected
                                                if r["repeat"] == repeat])),
                total_seconds=sum(r["seconds"] for r in selected if r["repeat"] == repeat))
                for repeat in range(2)})
    output_arrays = args.output / "outputs.npz"
    np.savez_compressed(output_arrays, **{key: np.asarray(value) for key, value in saved.items()},
                        trial_threads=[r["threads"] for r in trials],
                        trial_repeat=[r["repeat"] for r in trials],
                        trial_pair=[r["pair"] for r in trials])
    report = dict(
        status="ok", device="CPU only", cuda_operations=False, algorithm_changes=False,
        original_threads=original_threads, restored_threads=restored, conditions=conditions,
        orders=orders, pair_count=20, repeat_count=2, total_flow_calls=len(trials),
        all_15_fields_exact=all(r["all_fields_exact"] for r in trials),
        configuration=dataclasses.asdict(cfg), opencv_version=cv2.__version__,
        summaries=summaries, trials=trials, source_hashes=hashes, source_unchanged=True,
        inputs={str(p): sha256(p) for p in (
            fixed_arrays, fixed / "results.json", sequence / "manifest.json",
            *calibration.glob("*.json"))},
        output_arrays_sha256=sha256(output_arrays),
        limitations=[
            "Fixed twenty real pairs and saved backtraced poses; not a new tracking run.",
            "Thread setting and image preparation are outside each timed flow call.",
            "Two reversed condition orders reduce ordering bias but do not remove "
            "concurrent load or cache effects; repeated pairs are not independent samples.",
            "Changing OpenCV threads can affect every OpenCV operation in that process. "
            "This measures the complete original flow function only, not full tracking.",
            "All output fields are compared to the exact previously saved 32-thread baseline."])
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                             encoding="utf-8")
    print(json.dumps(dict(status="ok", all_15_fields_exact=report["all_15_fields_exact"],
                          original_threads=original_threads, restored_threads=restored,
                          summaries=summaries)), flush=True)


if __name__ == "__main__":
    main()

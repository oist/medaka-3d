"""Seven fixed lambdas on the primary window, reusing final A/B/C 2D labels."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

# This separate process shares a machine with the main comparison.
THREAD_SETTINGS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for variable in THREAD_SETTINGS:
    os.environ[variable] = "1"

import numpy as np  # noqa: E402
from downstream import prepare_window, solve_window  # noqa: E402
from run import (  # noqa: E402
    EXPERIMENT,
    clean,
    fixed_prefix_alignment,
    load_inputs,
    score_stage,
    shape_metrics,
    write_json,
)

from medaka3d.aquarium import sha256  # noqa: E402
from medaka3d.eval.generative import match_mouth_anchors  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

LAMBDAS = (0.0, 0.1, 0.25, 0.5, 0.767, 2.0, 8.0)
CORE = (75270, 78590)
CONTEXT = 332


def wait_for(paths, timeout=1800):
    """Wait only for explicitly named main-run completion files."""
    started = time.monotonic()
    last_notice = -60.0
    while not all(path.is_file() for path in paths):
        elapsed = time.monotonic() - started
        if elapsed > timeout:
            raise TimeoutError(
                f"Main result not ready: {[str(p) for p in paths if not p.is_file()]}"
            )
        if elapsed - last_notice >= 30:
            print(f"Waiting for main: {[p.name for p in paths if not p.is_file()]}", flush=True)
            last_notice = elapsed
        time.sleep(2)


def register(record, name, path):
    record.input(name, path, sha256=sha256(path))


def verify_main(reconstructed, main_path, calculated, main_summary):
    checks = []
    with np.load(main_path, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["frames"], reconstructed["frames"])
        for stage in ("c28", "final"):
            for key, actual in reconstructed[stage].items():
                if not isinstance(actual, np.ndarray):
                    continue
                expected = stored[f"{stage}__{key}"]
                equal = np.array_equal(actual, expected, equal_nan=True)
                if not equal:
                    raise AssertionError(f"lambda .767 does not reproduce main: {stage}/{key}")
                checks.append(f"{stage}/{key}")
            for key in ("metrics", "alignment", "female"):
                expected = main_summary[stage] if key == "metrics" else main_summary[key][stage]
                if clean(calculated[stage][key]) != expected:
                    raise AssertionError(f"lambda .767 main summary differs: {stage}/{key}")
    return dict(
        status="exactly_reproduced",
        arrays_equal_including_nan=checks,
        shape_metrics_equal=True,
        female_scores_equal=True,
        prefix_alignment_equal=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument("--staged", type=Path, default=Path("work/custom2d_comparison_v1/inputs"))
    parser.add_argument(
        "--analysis", type=Path, default=Path("work/custom2d_comparison_v1/analysis_v2")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("work/custom2d_comparison_v1/sensitivity")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with provenance(
        "custom2d_primary_lambda_sensitivity", args.output, name="sensitivity"
    ) as record:
        record.params_from(args)
        record.param("lambda_grid_mm", LAMBDAS)
        record.param("threads", {name: os.environ[name] for name in THREAD_SETTINGS})
        record.param("core_interval_half_open", CORE)
        record.param("context_native_frames", CONTEXT)
        record.param("lambda_selection", "none; all seven inherited values are reported")
        for name in ("sensitivity.py", "run.py", "downstream.py", "tracking.py", "README.md"):
            register(record, name, Path(__file__).with_name(name))
        protocol_path = args.analysis / "protocol.json"
        register(record, "main_protocol", protocol_path)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        is_deferred = protocol.get("revision") == "v3_deferred_hypotheses"
        expected_revision = (
            "v3_deferred_two_hypothesis_refinement"
            if is_deferred
            else "v2_retire_unmatched_competing_histories"
        )
        if is_deferred:
            for name in ("run_deferred.py", "deferred.py", "deferred_protocol.md"):
                register(record, name, Path(__file__).with_name(name))
        record.param("label_algorithm_revision", expected_revision)
        case = next(c for c in protocol["cases"] if c["name"] == "cg20s")
        if (case["start"], case["stop"]) != CORE or protocol["context_frames"] != CONTEXT:
            raise ValueError("Primary core or context differs from the fixed sensitivity protocol")
        if protocol["lambda_mm"] != 0.767:
            raise ValueError("Expected main lambda 0.767")
        arrays, detections, rig, _ = load_inputs(args, record)
        frames = np.arange(CORE[0] - CONTEXT, CORE[1] + CONTEXT)
        mask = (frames >= CORE[0]) & (frames < CORE[1])
        reference = {"c28": arrays["C28"], "final": arrays["C34"]}
        reference = {
            stage: {key: value[frames] for key, value in data.items()}
            for stage, data in reference.items()
        }
        prefix = {
            stage: np.flatnonzero((frames < CORE[0]) & np.isfinite(data["xyz"]).all(axis=(1, 2)))[
                :17
            ]
            for stage, data in reference.items()
        }
        gt = args.input_root / "assets/groundtruth/fish_sex_gt" / EXPERIMENT / "female_sex_gt.csv"
        register(record, "female_gt", gt)
        with gt.open(encoding="utf-8-sig", newline="") as handle:
            gt_rows = list(csv.DictReader(handle))

        def mouths(view, frame):
            data = detections[view]
            lo, hi = np.searchsorted(data.frame_idx, [frame, frame + 1])
            return data.points[lo:hi, 0]

        anchors = match_mouth_anchors(gt_rows, mouths, *CORE)
        if len(anchors) != 4:
            raise ValueError("Primary female anchor denominator differs from main protocol")
        summary = dict(
            status="in_progress",
            analysis_directory=str(args.analysis),
            label_version=expected_revision,
            experiment=EXPERIMENT,
            case="cg20s",
            core_interval_half_open=CORE,
            context_interval_half_open=[int(frames[0]), int(frames[-1] + 1)],
            native_frames_with_context=len(frames),
            core_native_frames=int(mask.sum()),
            lambda_grid_mm=LAMBDAS,
            selected_lambda=None,
            female_requested_anchors=len(anchors),
            female_possible_intervals=len(anchors) - 1,
            prefix_reference_frames={k: frames[v] for k, v in prefix.items()},
            threads={name: os.environ[name] for name in THREAD_SETTINGS},
            interpretation_limits=[
                "Selected primary development window from one fish group; "
                "no independent accuracy claim.",
                "Female endpoint intervals do not count individual swap events "
                "or validate male identities.",
                "All lambdas reuse exactly the final main-run 2D labels; "
                "only downstream lambda changes.",
                "One naming permutation is fitted from the pre-core reference prefix only.",
                "No lambda is selected or adopted from this table.",
            ],
            main_reproduction={},
            rows=[],
            elapsed_seconds=0.0,
        )
        main_dir = args.analysis / "cg20s"
        for condition in ("A", "B", "C"):
            wait_for(
                [
                    main_dir / f"tracking_{condition}_{v}{suffix}"
                    for v in VIEWS
                    for suffix in (".npz", ".json")
                ]
            )
            labels = {}
            for view in VIEWS:
                tracking_path = main_dir / f"tracking_{condition}_{view}.npz"
                tracking_json = tracking_path.with_suffix(".json")
                register(record, f"tracking_{condition}_{view}", tracking_path)
                register(record, f"tracking_metadata_{condition}_{view}", tracking_json)
                if condition == "C":
                    metadata = json.loads(tracking_json.read_text(encoding="utf-8"))["metadata"]
                    if metadata["algorithm_revision"] != expected_revision:
                        raise ValueError("C label revision differs from sensitivity protocol")
                original = detections[view]
                lo, hi = np.searchsorted(original.frame_idx, [frames[0], frames[-1] + 1])
                with np.load(tracking_path, allow_pickle=False) as data:
                    np.testing.assert_array_equal(data["global_rows"], np.arange(lo, hi))
                    np.testing.assert_array_equal(data["frame_idx"], original.frame_idx[lo:hi])
                    held = data["labels"]
                if held.shape != (hi - lo,) or np.any(held < 0):
                    raise ValueError("Tracking rows missing or unassigned")
                labels[view] = np.full(len(original.frame_idx), -1, np.int64)
                labels[view][lo:hi] = held
            prepared = prepare_window(detections, labels, rig, int(frames[0]), int(frames[-1] + 1))
            np.testing.assert_array_equal(prepared.frames, frames)
            # Verify common methods before interpreting any other lambda.
            for penalty in (0.767, *(x for x in LAMBDAS if x != 0.767)):
                arm_started = time.perf_counter()
                print(f"Sensitivity {condition} lambda={penalty:g}: solving", flush=True)
                reconstructed = solve_window(
                    prepared, lambda_mm=penalty, core_interval=CORE, reference_c28=arrays["C28"]
                )
                calculated, saved = {}, {"frames": frames}
                for stage in ("c28", "final"):
                    data = reconstructed[stage]
                    alignment = fixed_prefix_alignment(data, reference[stage]["xyz"], prefix[stage])
                    metrics = shape_metrics(data, mask)
                    metrics["source_rows_different_from_saved"] = int(
                        np.count_nonzero(data["source"][mask] != reference[stage]["source"][mask])
                    )
                    female = score_stage(data, frames, mask, anchors)
                    calculated[stage] = dict(metrics=metrics, alignment=alignment, female=female)
                    saved.update(
                        {
                            f"{stage}__{key}": value
                            for key, value in data.items()
                            if isinstance(value, np.ndarray)
                        }
                    )
                if penalty == 0.767:
                    main_path = main_dir / f"{condition}.npz"
                    main_summary_path = main_dir / f"{condition}_summary.json"
                    wait_for([main_path, main_summary_path])
                    register(record, f"main_{condition}", main_path)
                    register(record, f"main_summary_{condition}", main_summary_path)
                    main_summary = json.loads(main_summary_path.read_text(encoding="utf-8"))
                    summary["main_reproduction"][condition] = verify_main(
                        reconstructed, main_path, calculated, main_summary
                    )
                    output_path = main_path
                else:
                    output_path = args.output / f"{condition}__lambda{penalty:g}.npz"
                    np.savez_compressed(output_path, **saved)
                    record.output(
                        f"{condition}_lambda{penalty:g}", output_path, sha256=sha256(output_path)
                    )
                elapsed = time.perf_counter() - arm_started
                detail = dict(
                    condition=condition,
                    lambda_mm=penalty,
                    stages=calculated,
                    diagnostics=reconstructed["diagnostics"],
                    elapsed_seconds=elapsed,
                    reconstruction_path=str(output_path),
                )
                detail_path = args.output / f"{condition}__lambda{penalty:g}.json"
                write_json(detail_path, detail)
                record.output(
                    f"detail_{condition}_lambda{penalty:g}", detail_path, sha256=sha256(detail_path)
                )
                for stage, data in calculated.items():
                    metrics, female = data["metrics"], data["female"]
                    summary["rows"].append(
                        dict(
                            condition=condition,
                            lambda_mm=penalty,
                            stage=stage,
                            steps_over_10mm=metrics["steps_over_10mm"],
                            steps_over_4mm=metrics["steps_over_4mm"],
                            missing_fish_frames=metrics["missing_fish_frames"],
                            observed_native_steps=metrics["observed_native_steps"],
                            possible_native_steps=metrics["possible_native_steps"],
                            female_preserved=female["interval_counts"]["preserved"],
                            female_swap=female["interval_counts"]["swap"],
                            female_unscored=female["interval_counts"]["unscored"],
                            available_anchors=female["available_anchors"],
                            requested_anchors=female["requested_anchors"],
                            source_rows_different_from_saved=metrics[
                                "source_rows_different_from_saved"
                            ],
                            residual_median_mm=metrics["residual_mm"]["median"],
                            residual_p95_mm=metrics["residual_mm"]["p95"],
                            alignment_valid=data["alignment"]["valid"],
                            elapsed_seconds=elapsed,
                            reconstruction_path=str(output_path),
                            details_path=str(detail_path),
                        )
                    )
                summary["rows"].sort(
                    key=lambda row: (row["condition"], row["lambda_mm"], row["stage"])
                )
                summary["elapsed_seconds"] = time.perf_counter() - started
                write_json(args.output / "summary.json", summary)
                print(
                    f"Sensitivity {condition} lambda={penalty:g}: saved ({elapsed:.1f}s)",
                    flush=True,
                )
        summary["status"] = "complete"
        summary["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "summary.json", summary)
        record.output(
            "summary", args.output / "summary.json", sha256=sha256(args.output / "summary.json")
        )


if __name__ == "__main__":
    main()

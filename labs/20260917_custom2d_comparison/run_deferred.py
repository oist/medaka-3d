"""Complete condition C with deferred 2D hypotheses; reuse frozen A/B bytes."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import run as baseline
from deferred import DeferredConfig, refine_deferred
from downstream import prepare_window, solve_window
from tracking import TrackingResult

from medaka3d.aquarium import sha256
from medaka3d.eval.generative import match_mouth_anchors
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def equivalent_partitions(first, second):
    pairs = np.unique(np.c_[first, second], axis=0)
    return len(pairs) == len(np.unique(first)) == len(np.unique(second))


def verified_copy(source, target, rec, name):
    digest = sha256(source)
    if target.exists():
        if sha256(target) != digest:
            raise ValueError(f"Existing destination differs: {target}")
    else:
        shutil.copy2(source, target)
    if sha256(target) != digest:
        raise ValueError(f"Copied bytes differ: {target}")
    rec.input(name + "_source", source, sha256=digest)
    rec.output(name, target, sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("work/custom2d_comparison_v1/analysis_v2")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("work/custom2d_comparison_v1/analysis_v3")
    )
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument("--staged", type=Path, default=Path("work/custom2d_comparison_v1/inputs"))
    args = parser.parse_args()
    source_summary = read(args.source / "summary.json")
    if source_summary["status"] != "complete":
        raise ValueError("The complete frozen v2 comparison is required")
    args.output.mkdir(parents=True, exist_ok=True)
    deferred_config = DeferredConfig()
    protocol = copy.deepcopy(source_summary["protocol"])
    protocol["deferred_config"] = asdict(deferred_config)
    protocol["arms"]["C"] = "Independent bidirectional 2D with deferred short hypotheses"
    protocol["revision"] = "v3_deferred_hypotheses"
    protocol["base_summary_sha256"] = sha256(args.source / "summary.json")
    protocol_target = args.output / "protocol.json"
    if protocol_target.exists() and read(protocol_target) != baseline.clean(protocol):
        raise ValueError("Protocol changed; use another output directory")
    baseline.write_json(protocol_target, protocol)
    with provenance("custom2d_deferred_comparison", args.output) as rec:
        rec.params_from(args)
        for name in (
            "run_deferred.py",
            "run.py",
            "tracking.py",
            "downstream.py",
            "deferred.py",
            "deferred_protocol.md",
        ):
            p = Path(__file__).parent / name
            rec.input(name, p, sha256=sha256(p))
        rec.input(
            "base_summary", args.source / "summary.json", sha256=protocol["base_summary_sha256"]
        )
        rec.input("protocol", protocol_target, sha256=sha256(protocol_target))
        arrays, detections, rig, _ = baseline.load_inputs(args, rec)
        gt = args.input_root / "assets/groundtruth/fish_sex_gt" / baseline.EXPERIMENT
        gt /= "female_sex_gt.csv"
        rec.input("female_gt", gt, sha256=sha256(gt))
        with gt.open(encoding="utf-8-sig", newline="") as stream:
            gt_rows = list(csv.DictReader(stream))

        def mouths(view, frame):
            data = detections[view]
            lo, hi = np.searchsorted(data.frame_idx, [frame, frame + 1])
            return data.points[lo:hi, 0]

        priority = ["cg20s", "known_30735", "known_31479", "known_323819"]
        cases = sorted(
            source_summary["cases"],
            key=lambda c: priority.index(c["name"]) if c["name"] in priority else len(priority),
        )
        for case in cases:
            origin = args.source / case["name"]
            dest = args.output / case["name"]
            dest.mkdir(exist_ok=True)
            if (dest / "summary.json").exists():
                print(f"Reusing completed {case['name']}", flush=True)
                continue
            began = time.perf_counter()
            result = copy.deepcopy(case)
            result["conservative_C"] = copy.deepcopy(case["arms"]["C"])
            rec.input(
                case["name"] + "_summary",
                origin / "summary.json",
                sha256=sha256(origin / "summary.json"),
            )
            for path in origin.iterdir():
                if (
                    path.name == "saved_baseline.npz"
                    or path.name in ("A.npz", "B.npz", "A_summary.json", "B_summary.json")
                    or path.name.startswith(("tracking_A_", "tracking_B_"))
                ):
                    verified_copy(path, dest / path.name, rec, case["name"] + "_" + path.name)
            labels, tracking_stats, same_partitions, refinements = {}, {}, {}, {}
            for view in VIEWS:
                track_path = origin / f"tracking_C_{view}.npz"
                json_path = origin / f"tracking_C_{view}.json"
                rec.input(case["name"] + "_base_C_" + view, track_path, sha256=sha256(track_path))
                rec.input(
                    case["name"] + "_base_C_events_" + view, json_path, sha256=sha256(json_path)
                )
                with np.load(track_path, allow_pickle=False) as loaded:
                    native, rows, held = [loaded[k] for k in ("frame_idx", "global_rows", "labels")]
                data = detections[view]
                np.testing.assert_array_equal(native, data.frame_idx[rows])
                previous = read(json_path)
                tracked = refine_deferred(
                    native,
                    data.points[rows],
                    TrackingResult(held, previous["events"], previous["metadata"]),
                    view=view,
                    config=deferred_config,
                )
                if tracked.labels.shape != held.shape or np.any(tracked.labels < 0):
                    raise ValueError("Deferred processing discarded detections")
                same_partitions[view] = equivalent_partitions(held, tracked.labels)
                if same_partitions[view]:
                    # Preserve numeric names too, so reused reconstruction and
                    # later fresh lambda solves consume byte-identical labels.
                    tracked.labels = held.copy()
                labels[view] = np.full(len(data.frame_idx), -1, dtype=np.int64)
                labels[view][rows] = tracked.labels
                tracking_stats[view] = baseline.tracking_metrics(
                    native, data.points[rows], tracked.labels, case["start"], case["stop"], view
                )
                refinements[view] = tracked.metadata
                np.savez_compressed(
                    dest / f"tracking_C_{view}.npz",
                    frame_idx=native,
                    global_rows=rows,
                    labels=tracked.labels,
                )
                baseline.write_json(
                    dest / f"tracking_C_{view}.json",
                    dict(metadata=tracked.metadata, events=tracked.events),
                )
                for suffix in ("npz", "json"):
                    path = dest / f"tracking_C_{view}.{suffix}"
                    rec.output(
                        case["name"] + "_tracking_C_" + view + "_" + suffix,
                        path,
                        sha256=sha256(path),
                    )
            print(f"{case['name']} deferred: unchanged partitions={same_partitions}", flush=True)
            start, stop = case["context_start"], case["context_stop"]
            frames = np.arange(start, stop)
            mask = (frames >= case["start"]) & (frames < case["stop"])
            if all(same_partitions.values()):
                verified_copy(origin / "C.npz", dest / "C.npz", rec, case["name"] + "_C")
                arm_result = copy.deepcopy(case["arms"]["C"])
                arm_result["diagnostics"]["reconstruction_reused_exact_partition"] = True
            else:
                prepared = prepare_window(detections, labels, rig, start, stop)
                reconstructed = solve_window(
                    prepared,
                    lambda_mm=protocol["lambda_mm"],
                    core_interval=(case["start"], case["stop"]),
                    reference_c28=arrays["C28"],
                )
                arm_result = dict(alignment={}, female={}, diagnostics=reconstructed["diagnostics"])
                anchors = match_mouth_anchors(gt_rows, mouths, case["start"], case["stop"])
                saved = {"frames": frames}
                for stage, key in (("c28", "C28"), ("final", "C34")):
                    reference = {k: v[frames] for k, v in arrays[key].items()}
                    prefix = np.flatnonzero(
                        (frames < case["start"]) & np.isfinite(reference["xyz"]).all(axis=(1, 2))
                    )[:17]
                    data = reconstructed[stage]
                    arm_result["alignment"][stage] = baseline.fixed_prefix_alignment(
                        data, reference["xyz"], prefix
                    )
                    arm_result[stage] = baseline.shape_metrics(data, mask)
                    arm_result[stage]["source_rows_different_from_saved"] = int(
                        np.count_nonzero(data["source"][mask] != reference["source"][mask])
                    )
                    arm_result["female"][stage] = baseline.score_stage(data, frames, mask, anchors)
                    saved.update(
                        {stage + "__" + k: v for k, v in data.items() if isinstance(v, np.ndarray)}
                    )
                np.savez_compressed(dest / "C.npz", **saved)
                rec.output(case["name"] + "_C", dest / "C.npz", sha256=sha256(dest / "C.npz"))
            arm_result["tracking"] = tracking_stats
            arm_result["deferred"] = dict(
                equivalent_base_partitions=same_partitions,
                views=refinements,
                fixed_base_sha256=sha256(origin / "C.npz"),
            )
            arm_result["elapsed_seconds"] = time.perf_counter() - began
            result["arms"]["C"] = arm_result
            baseline.write_json(dest / "C_summary.json", arm_result)
            baseline.write_json(dest / "summary.json", result)
            rec.output(case["name"], dest / "summary.json", sha256=sha256(dest / "summary.json"))
            print(
                f"{case['name']} C: final >10mm={arm_result['final']['steps_over_10mm']}",
                flush=True,
            )
        summary = copy.deepcopy(source_summary)
        summary["protocol"] = protocol
        summary["cases"] = [
            read(args.output / c["name"] / "summary.json") for c in source_summary["cases"]
        ]
        baseline.write_json(args.output / "summary.json", summary)
        rec.output(
            "summary", args.output / "summary.json", sha256=sha256(args.output / "summary.json")
        )


if __name__ == "__main__":
    main()

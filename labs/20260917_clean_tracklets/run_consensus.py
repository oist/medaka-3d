"""Evaluate consensus two-view observations with byte-identical clean 2D inputs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
from evaluate import geometry_metrics, prefix_permutation, validate_sources
from multiview import MultiViewConfig
from multiview_consensus import reconstruct
from run import CALIBRATION, EXPERIMENT, load_reference, write_json

from medaka3d.aquarium import sha256
from medaka3d.calib import load_rig
from medaka3d.eval.generative import match_mouth_anchors, score_female_identity
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, default=Path("work/clean_tracklets_v1"))
    parser.add_argument("--output", type=Path, default=Path("work/clean_tracklets_v2"))
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument(
        "--previous", type=Path, default=Path("work/custom2d_comparison_v1/analysis_v3")
    )
    args = parser.parse_args()
    initial = json.loads((args.initial / "summary.json").read_text(encoding="utf-8"))
    if initial["status"] != "complete":
        raise ValueError("The initial experiment must be complete and retained")
    if args.output.exists():
        raise ValueError("Use a new destination; preserved outputs are immutable")
    args.output.mkdir(parents=True)
    protocol = dict(initial["protocol"])
    protocol["initial_protocol_sha256"] = sha256(args.initial / "protocol.json")
    protocol["condition_3d"] = "consensus_of_near_optimal_sources_with_two_view_support"
    protocol["difference_from_initial"] = (
        "Only the unit of assignment abstention changes. "
        "All 2D bytes and numerical gates unchanged. "
        "Keep two agreed measured views when only a third view is ambiguous."
    )
    pipeline = (
        "run_consensus.py",
        "multiview_consensus.py",
        "multiview.py",
        "evaluate.py",
        "run.py",
    )
    protocol["code_sha256"] = {name: sha256(Path(__file__).parent / name) for name in pipeline}
    write_json(args.output / "protocol.json", protocol)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for p in Path(__file__).parent.iterdir():
        if p.suffix in (".py", ".md"):
            shutil.copy2(p, snapshot / p.name)
    rig = load_rig(args.input_root / CALIBRATION)
    gt_path = args.input_root / "assets/groundtruth/fish_sex_gt" / EXPERIMENT / "female_sex_gt.csv"
    with gt_path.open(encoding="utf-8-sig", newline="") as file:
        gt = list(csv.DictReader(file))
    complete = []
    with provenance("clean_tracklets_consensus", args.output) as rec:
        rec.params_from(args)
        rec.input(
            "protocol", args.output / "protocol.json", sha256=sha256(args.output / "protocol.json")
        )
        rec.input(
            "initial_summary",
            args.initial / "summary.json",
            sha256=sha256(args.initial / "summary.json"),
        )
        rec.input("female_evaluation_only", gt_path, sha256=sha256(gt_path))
        for p in (args.input_root / CALIBRATION).iterdir():
            if p.is_file():
                rec.input("calibration_" + p.name, p, sha256=sha256(p))
        for name in pipeline:
            p = Path(__file__).parent / name
            rec.input(name, p, sha256=sha256(p))
        for case in initial["cases"]:
            began = time.perf_counter()
            dest, old_dir = args.output / case["name"], args.initial / case["name"]
            dest.mkdir()
            views = {}
            for view in VIEWS:
                for arm in ("P", "N", "F"):
                    for suffix in (".npz", ".json"):
                        p = old_dir / f"tracking_{arm}_{view}{suffix}"
                        shutil.copy2(p, dest / p.name)
                        assert sha256(p) == sha256(dest / p.name)
                        rec.input(case["name"] + "_" + p.name, p, sha256=sha256(p))
                with np.load(dest / f"tracking_F_{view}.npz", allow_pickle=False) as z:
                    views[view] = dict(z)
            frames = np.arange(case["context_start"], case["context_stop"], dtype=np.int32)
            core = (frames >= case["start"]) & (frames < case["stop"])
            print(f"Consensus {case['name']}: {len(frames)} native frames", flush=True)
            stage = reconstruct(frames, views, rig, config=MultiViewConfig())
            source_audit = validate_sources(stage, frames, views)
            arrays = {k: v for k, v in stage.items() if isinstance(v, np.ndarray)}
            np.savez_compressed(dest / "reconstruction.npz", frames=frames, **arrays)
            for view in VIEWS:
                np.savez_compressed(
                    dest / f"joined_{view}.npz",
                    frame_idx=views[view]["frame_idx"],
                    clean_labels=views[view]["labels"],
                    joined_labels=stage["joined_labels_by_view"][view],
                )
            write_json(
                dest / "joins.json",
                {k: v for k, v in stage.items() if not isinstance(v, np.ndarray)},
            )
            references = load_reference(args.previous / case["name"], frames)
            for name in ("saved_baseline.npz", "C.npz"):
                p = args.previous / case["name"] / name
                rec.input("comparison_" + case["name"] + "_" + name, p, sha256=sha256(p))
            p = old_dir / "reconstruction.npz"
            rec.input("initial_reconstruction_" + case["name"], p, sha256=sha256(p))
            with np.load(old_dir / "reconstruction.npz", allow_pickle=False) as z:
                references["initial_clean_v1"] = dict(z)

            def mouth_at(view, frame, views=views):
                data = views[view]
                lo, hi = np.searchsorted(data["frame_idx"], [frame, frame + 1])
                return data["points"][lo:hi, 0]

            anchors = match_mouth_anchors(gt, mouth_at, case["start"], case["stop"])
            comparison = {}
            for name, output in {"new": stage, **references}.items():
                review = output.get("identity_review")
                female = score_female_identity(
                    frames[core],
                    output["source"][core],
                    np.isfinite(output["pose"][core, :, 0]).all(-1),
                    anchors,
                    review=review[core] if review is not None else None,
                )
                comparison[name] = {"geometry": geometry_metrics(output, core), "female": female}
            result = {
                **case,
                "comparison": comparison,
                "display_alignment": prefix_permutation(
                    stage, references["saved_C34"], frames, case["start"]
                ),
                "source_audit": source_audit,
                "multiview": stage["metadata"],
                "bridges": stage["bridges"],
                "elapsed_seconds": time.perf_counter() - began,
                "initial_2d_bytes_unchanged": True,
            }
            write_json(dest / "summary.json", result)
            for name in ("reconstruction.npz", "joins.json", "summary.json"):
                rec.output(case["name"] + "_" + name, dest / name, sha256=sha256(dest / name))
            complete.append(result)
            print(f"Consensus {case['name']}: {comparison['new']}", flush=True)
        write_json(
            args.output / "summary.json",
            {
                "protocol": protocol,
                "cases": complete,
                "status": "complete",
                "completed_cases": len(complete),
                "requested_cases": len(complete),
            },
        )
        rec.output(
            "summary", args.output / "summary.json", sha256=sha256(args.output / "summary.json")
        )


if __name__ == "__main__":
    main()

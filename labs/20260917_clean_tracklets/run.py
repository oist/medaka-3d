"""Run the frozen clean-fragment experiment; never read SLEAP temporal IDs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import c01_s16_predictions as c01  # noqa: E402
from evaluate import (  # noqa: E402
    geometry_metrics,
    prefix_permutation,
    tracklet_metrics,
    validate_sources,
)
from flow import extract_flow  # noqa: E402
from multiview import MultiViewConfig, reconstruct  # noqa: E402
from tracking import TrackingConfig, build_tracklets  # noqa: E402

from medaka3d.aquarium import sha256  # noqa: E402
from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.eval.generative import match_mouth_anchors, score_female_identity  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

EXPERIMENT = "20251225_K_1_2_FCMC"
CALIBRATION = (
    "calibration/fixed_refractive_192x192x186/"
    "phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration"
)
CASE_NAMES = ("cg20s", "known_30735", "known_31479", "known_323819", "control_50000", "holdout20s")
VIDEO_NAMES = ("cam0_top.mp4", "cam1_side.mp4", "cam2_side.mp4")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    path.write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_reference(directory, frames):
    with np.load(directory / "saved_baseline.npz", allow_pickle=False) as z:
        if not np.array_equal(z["frames"], frames):
            raise ValueError("New and historical comparison ranges differ")
        old = {key.split("__", 1)[1]: z[key].copy() for key in z.files if key.startswith("final__")}
    with np.load(directory / "C.npz", allow_pickle=False) as z:
        previous = {
            key.split("__", 1)[1]: z[key].copy() for key in z.files if key.startswith("final__")
        }
    return {"saved_C34": old, "previous_custom_C_v3": previous}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument(
        "--previous", type=Path, default=Path("work/custom2d_comparison_v1/analysis_v3")
    )
    parser.add_argument(
        "--videos", type=Path, default=Path("work/three_body/source_cache/medaka_202512")
    )
    parser.add_argument("--output", type=Path, default=Path("work/clean_tracklets_v1"))
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    if set(args.case) - set(CASE_NAMES):
        parser.error("Unknown case")
    args.output.mkdir(parents=True, exist_ok=True)
    previous_protocol = args.previous / "protocol.json"
    original_cases = json.loads(previous_protocol.read_text(encoding="utf-8"))["cases"]
    cases = [next(c for c in original_cases if c["name"] == name) for name in CASE_NAMES]
    tracking_config, geometry_config = TrackingConfig(), MultiViewConfig()
    code = {
        name: sha256(Path(__file__).parent / name)
        for name in ("run.py", "evaluate.py", "tracking.py", "flow.py", "multiview.py")
    }
    protocol = {
        "experiment": EXPERIMENT,
        "fps": 166,
        "context": 332,
        "cases": cases,
        "tracking_config": asdict(tracking_config),
        "multiview_config": asdict(geometry_config),
        "code_sha256": code,
        "conditions_2d": {"P": "pectoral", "N": "nodes", "F": "nodes_plus_optical_flow"},
        "condition_3d": "F_clean_fragments_geometry_motion_then_cross_view_joins",
        "identity_claim": (
            "No all-fish identity accuracy claim; female endpoints and manual anchors only"
        ),
        "old_tracks_used_for_estimation": False,
    }
    target = args.output / "protocol.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != clean(protocol):
        raise ValueError("Frozen protocol/code changed; use a fresh output directory")
    write_json(target, protocol)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in (".py", ".md"):
            shutil.copy2(p, snapshot / p.name)
    selected = [c for c in cases if not args.case or c["name"] in args.case]
    with provenance(
        "clean_tracklets", args.output, name="run_" + ("_".join(args.case) or "all")
    ) as rec:
        rec.params_from(args)
        rec.input("protocol", target, sha256=sha256(target))
        detections = {}
        for view in VIEWS:
            p = args.input_root / "work/c01_s16_predictions/bottomup" / f"{EXPERIMENT}__{view}.npz"
            rec.input("untracked_" + view, p, sha256=sha256(p))
            detections[view] = c01.load(args.input_root / "work", EXPERIMENT, view)
            if tuple(detections[view].nodes) != ("mouth", "pectoral", "tail"):
                raise ValueError("Unexpected node order")
        calibration = args.input_root / CALIBRATION
        for p in calibration.iterdir():
            if p.is_file():
                rec.input("calibration_" + p.name, p, sha256=sha256(p))
        rig = load_rig(calibration)
        gt_path = (
            args.input_root / "assets/groundtruth/fish_sex_gt" / EXPERIMENT / "female_sex_gt.csv"
        )
        rec.input("female_evaluation_only", gt_path, sha256=sha256(gt_path))
        with gt_path.open(encoding="utf-8-sig", newline="") as file:
            gt = list(csv.DictReader(file))

        def mouth_at(view, frame):
            data = detections[view]
            lo, hi = np.searchsorted(data.frame_idx, [frame, frame + 1])
            return data.points[lo:hi, 0]

        for case in selected:
            dest = args.output / case["name"]
            dest.mkdir(exist_ok=True)
            if (dest / "summary.json").exists():
                print(f"Reusing completed {case['name']}", flush=True)
                continue
            began = time.perf_counter()
            start, stop = case["start"] - 332, case["stop"] + 332
            frames = np.arange(start, stop, dtype=np.int32)
            core = (frames >= case["start"]) & (frames < case["stop"])
            views, stats = {}, {arm: {} for arm in ("P", "N", "F")}
            flow_stats = {}
            for view, video_name in zip(VIEWS, VIDEO_NAMES, strict=True):
                data = detections[view]
                lo, hi = np.searchsorted(data.frame_idx, [start, stop])
                native, points = data.frame_idx[lo:hi], data.points[lo:hi]
                video = args.videos / EXPERIMENT / view / video_name
                print(
                    f"{case['name']} {view}: extracting image flow ({len(frames)} frames)",
                    flush=True,
                )
                flow = extract_flow(video, native, points, cache_path=dest / f"flow_{view}.npz")
                flow_stats[view] = flow.get("metadata", flow.get("diagnostics", {}))
                for arm in ("P", "N", "F"):
                    tracked = build_tracklets(
                        native,
                        points,
                        view=view,
                        feature_mode="pectoral" if arm == "P" else "nodes",
                        flow=flow if arm == "F" else None,
                        config=tracking_config,
                    )
                    path = dest / f"tracking_{arm}_{view}.npz"
                    np.savez_compressed(
                        path,
                        frame_idx=native,
                        points=points,
                        labels=tracked.labels,
                        clean_edge=tracked.clean_edge,
                        global_rows=np.arange(lo, hi),
                    )
                    rec.output(f"{case['name']}_{arm}_{view}", path, sha256=sha256(path))
                    write_json(
                        path.with_suffix(".json"),
                        {"events": tracked.events, "metadata": tracked.metadata},
                    )
                    stats[arm][view] = tracklet_metrics(
                        native,
                        points,
                        tracked.labels,
                        tracked.clean_edge,
                        case["start"],
                        case["stop"],
                        view,
                    )
                    if arm == "F":
                        views[view] = dict(
                            frame_idx=native,
                            points=points,
                            labels=tracked.labels,
                            clean_edge=tracked.clean_edge,
                            global_rows=np.arange(lo, hi),
                        )
                    print(
                        f"{case['name']} {view} {arm}: {stats[arm][view]['fragments']} fragments",
                        flush=True,
                    )
            print(f"{case['name']}: reconstructing from clean fragments", flush=True)
            stage = reconstruct(frames, views, rig, config=geometry_config)
            audit = validate_sources(stage, frames, views)
            arrays = {k: v for k, v in stage.items() if isinstance(v, np.ndarray)}
            np.savez_compressed(dest / "reconstruction.npz", frames=frames, **arrays)
            for view in VIEWS:
                joined = stage.get("joined_labels_by_view", stage.get("labels_by_view", {}))[view]
                np.savez_compressed(
                    dest / f"joined_{view}.npz",
                    frame_idx=views[view]["frame_idx"],
                    clean_labels=views[view]["labels"],
                    joined_labels=joined,
                )
            write_json(
                dest / "joins.json",
                {k: v for k, v in stage.items() if not isinstance(v, np.ndarray)},
            )
            # Historical trajectories enter only after the estimated outputs are saved.
            references = load_reference(args.previous / case["name"], frames)
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
            alignment = prefix_permutation(stage, references["saved_C34"], frames, case["start"])
            result = {
                **case,
                "context_start": start,
                "context_stop": stop,
                "tracking": stats,
                "flow": flow_stats,
                "comparison": comparison,
                "display_alignment": alignment,
                "source_audit": audit,
                "multiview": stage.get("metadata", {}),
                "bridges": stage.get("bridges", []),
                "elapsed_seconds": time.perf_counter() - began,
            }
            write_json(dest / "summary.json", result)
            for name in ("reconstruction.npz", "joins.json", "summary.json"):
                rec.output(f"{case['name']}_{name}", dest / name, sha256=sha256(dest / name))
            for name in ("saved_baseline.npz", "C.npz"):
                p = args.previous / case["name"] / name
                rec.input(f"comparison_only_{case['name']}_{name}", p, sha256=sha256(p))
            print(f"{case['name']}: {comparison['new']}", flush=True)
        complete = []
        for case in cases:
            p = args.output / case["name"] / "summary.json"
            if p.exists():
                complete.append(json.loads(p.read_text(encoding="utf-8")))
        write_json(
            args.output / "summary.json",
            {
                "protocol": protocol,
                "cases": complete,
                "status": "complete" if len(complete) == len(cases) else "in_progress",
                "completed_cases": len(complete),
                "requested_cases": len(cases),
            },
        )
        rec.output(
            "summary", args.output / "summary.json", sha256=sha256(args.output / "summary.json")
        )


if __name__ == "__main__":
    main()

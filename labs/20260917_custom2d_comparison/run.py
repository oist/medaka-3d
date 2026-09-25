"""Run the frozen A/B/C comparison, preserving all native detections and stages."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import c01_s16_predictions as c01  # noqa: E402
from downstream import prepare_window, solve_window  # noqa: E402
from tracking import TrackingConfig, cut_sleap_links, track_bidirectional  # noqa: E402

from medaka3d.aquarium import sha256  # noqa: E402
from medaka3d.calib import load_rig, triangulate  # noqa: E402
from medaka3d.eval.generative import match_mouth_anchors, score_female_identity  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

EXPERIMENT = "20251225_K_1_2_FCMC"
CALIBRATION = (
    "calibration/fixed_refractive_192x192x186/"
    "phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration"
)
PERMS = np.asarray(list(permutations(range(3))))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, obj):
    path.write_text(
        json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def describe(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return dict(
        n=int(a.size),
        median=float(np.median(a)) if a.size else None,
        p95=float(np.quantile(a, 0.95)) if a.size else None,
        p99=float(np.quantile(a, 0.99)) if a.size else None,
        maximum=float(a.max()) if a.size else None,
    )


def tracking_metrics(frame_idx, points, labels, start, stop, view):
    """Count links ending in core; never reward discarded/unnamed detections."""
    selected = (frame_idx >= start) & (frame_idx < stop)
    live = labels >= 0
    previous = {}
    distances, gaps, edges = [], [], []
    starts = 0
    for row, (frame, label) in enumerate(zip(frame_idx, labels, strict=True)):
        if label < 0:
            continue
        before = previous.get(int(label))
        if selected[row]:
            if before is None:
                starts += 1
            else:
                dt = int(frame - frame_idx[before])
                assert dt > 0, "Duplicate identity inside one frame"
                step = np.linalg.norm(points[row, :2] - points[before, :2], axis=-1)
                finite = step[np.isfinite(step)]
                distance = float(finite.max()) if finite.size else np.nan
                distances.append(distance)
                gaps.append(dt)
                edges.append([before, row])
        previous[int(label)] = row
    held = selected & live
    _, lengths = np.unique(labels[held], return_counts=True)
    gate = 60 if view == "top_cam" else 50
    distance = np.asarray(distances)
    dt = np.asarray(gaps)
    return dict(
        detection_rows=int(selected.sum()),
        assigned_rows=int(held.sum()),
        detection_retention=float(held.sum() / selected.sum()) if selected.any() else None,
        fragments_in_core=int(lengths.size),
        new_tracks_in_core=starts,
        connected_edges=len(edges),
        consecutive_edges=int((dt == 1).sum()),
        gap_links=int((dt > 1).sum()),
        hidden_frames_in_links=int((dt - 1).sum()),
        large_link_steps=int(np.count_nonzero(distance > gate * dt)),
        large_consecutive_steps=int(np.count_nonzero((distance > gate) & (dt == 1))),
        motion_px=describe(distance),
        fragment_length_frames=describe(lengths),
        legacy_step_gate_px=gate,
        scope="Motion/fragmentation diagnostics; not observed identity accuracy",
    )


def shape_metrics(stage, mask):
    xyz = stage["xyz"][mask].astype(float)
    held = np.isfinite(xyz).all(-1)
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
    possible = max(len(xyz) - 1, 0) * 3
    finite_steps = np.isfinite(step)
    alternative = []
    for i in np.flatnonzero((np.nan_to_num(step, nan=0) > 4).any(-1)):
        if not (held[i].all() and held[i + 1].all()):
            continue
        costs = np.array([np.linalg.norm(xyz[i] - xyz[i + 1, p], axis=-1).sum() for p in PERMS])
        if np.argmin(costs) != 0 and costs[0] - costs.min() > 5:
            alternative.append(i + 1)
    pose = stage.get("pose")
    node_available = None
    length = None
    if pose is not None:
        pose = pose[mask]
        node_available = np.isfinite(pose).all(-1).sum(axis=(0, 1))
        length = describe(np.linalg.norm(pose[:, :, 2] - pose[:, :, 0], axis=-1))
    return dict(
        native_frames=len(xyz),
        possible_fish_frames=int(held.size),
        observed_fish_frames=int(held.sum()),
        missing_fish_frames=int((~held).sum()),
        availability=float(held.mean()) if held.size else None,
        possible_native_steps=possible,
        observed_native_steps=int(finite_steps.sum()),
        steps_over_4mm=int((step > 4).sum()),
        steps_over_10mm=int((step > 10).sum()),
        step_mm=describe(step),
        residual_mm=describe(stage["residual_mm"][mask]),
        alternative_permutation_edges=len(alternative),
        alternative_permutation_relative_frames=alternative,
        node_observed_counts=node_available,
        mouth_tail_length_mm=length,
        identity_metric_note="Alternative permutations and step counts are not confirmed swaps",
    )


def selected_pose(detections, rig, source, xyz, frames):
    uv = np.full((len(frames), 3, 3, 3, 2), np.nan)
    origins, directions, present = [], [], []
    for vi, view in enumerate(VIEWS):
        data = detections[view]
        starts = np.searchsorted(data.frame_idx, frames)
        rows = source[:, :, vi]
        good = rows >= 0
        flat = starts[:, None] + np.maximum(rows, 0)
        uv[:, :, vi][good] = data.points[flat[good]]
        o, d = rig.camera(view).rays(uv[:, :, vi])
        origins.append(o)
        directions.append(d)
        present.append(np.isfinite(o).all(-1) & np.isfinite(d).all(-1))
    pose, _ = triangulate(
        np.stack(origins, axis=-2), np.stack(directions, axis=-2), np.stack(present, axis=-1)
    )
    pose[:, :, 1] = xyz
    return pose


def fixed_prefix_alignment(stage, reference, prefix_rows):
    """One naming permutation from pre-core frames; no core/GT optimisation."""
    valid = prefix_rows[
        np.isfinite(stage["xyz"][prefix_rows]).all(axis=(1, 2))
        & np.isfinite(reference[prefix_rows]).all(axis=(1, 2))
    ]
    if len(valid) < 5:
        return dict(
            valid=False,
            reason="fewer_than_5_joint_prefix_frames",
            permutation=[0, 1, 2],
            frames_used=len(valid),
        )
    costs = np.array(
        [
            np.median(np.linalg.norm(reference[valid] - stage["xyz"][valid][:, p], axis=-1))
            for p in PERMS
        ]
    )
    order = np.argsort(costs)
    choice = PERMS[order[0]]
    for key, value in list(stage.items()):
        if (
            isinstance(value, np.ndarray)
            and value.ndim >= 2
            and value.shape[:2] == stage["xyz"].shape[:2]
        ):
            stage[key] = value[:, choice].copy()
    return dict(
        valid=True,
        permutation=choice,
        frames_used=len(valid),
        median_distance_mm=costs[order[0]],
        runnerup_margin_mm=costs[order[1]] - costs[order[0]],
        scope="One pre-core naming permutation; not identity truth",
    )


def score_stage(stage, frames, mask, anchors):
    pose = stage["pose"]
    return score_female_identity(
        frames[mask], stage["source"][mask], np.isfinite(pose[mask, :, 0]).all(-1), anchors
    )


def load_inputs(args, rec):
    manifest_path = args.staged / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rec.input("staging_manifest", manifest_path, sha256=sha256(manifest_path))
    arrays = {}
    for name, item in manifest.items():
        path = Path(item["path"])
        digest = sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"Staged hash differs: {name}")
        rec.input(name, path, sha256=digest)
        if name.startswith("C27_"):
            arrays[name] = np.load(path, allow_pickle=False)
        elif name == "C28":
            with np.load(path, allow_pickle=False) as data:
                arrays[name] = {k: data[k] for k in ("xyz", "source", "residual_mm")}
    old = json.loads(
        (ROOT / "docs/reviews/2026-09-16_multiview_identity_review.json").read_text(
            encoding="utf-8"
        )
    )
    c34 = ROOT / old["baseline"]["path"]
    if sha256(c34) != old["baseline"]["sha256"]:
        raise ValueError("Original C34 changed")
    rec.input("C34", c34, sha256=sha256(c34))
    with np.load(c34, allow_pickle=False) as data:
        arrays["C34"] = {k: data[k] for k in ("xyz", "source", "residual_mm")}
    detections = {}
    for view in VIEWS:
        path = args.input_root / "work/c01_s16_predictions/bottomup" / f"{EXPERIMENT}__{view}.npz"
        rec.input(f"C01_{view}", path, sha256=sha256(path))
        detections[view] = c01.load(args.input_root / "work", EXPERIMENT, view)
        if len(detections[view].frame_idx) != len(arrays[f"C27_{view}"]):
            raise ValueError("C01/C27 row alignment differs")
    calibration = args.input_root / CALIBRATION
    for p in sorted(calibration.iterdir()):
        if p.is_file():
            rec.input("calibration_" + p.name, p, sha256=sha256(p))
    return arrays, detections, load_rig(calibration), old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument("--staged", type=Path, default=Path("work/custom2d_comparison_v1/inputs"))
    parser.add_argument("--output", type=Path, default=Path("work/custom2d_comparison_v1/analysis"))
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--context", type=int, default=332)
    parser.add_argument("--lambda-mm", type=float, default=0.767)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / "work/bidirectional_reconnect_v1/analysis_v2/protocol.json"
    cases = json.loads(protocol_path.read_text(encoding="utf-8"))["cases"]
    cases.append(
        dict(
            name="holdout20s",
            start=240000,
            stop=243320,
            category="temporally_separate_unlabelled20s_same_group",
        )
    )
    config = TrackingConfig()
    protocol = dict(
        experiment=EXPERIMENT,
        fps=166,
        cases=cases,
        context_frames=args.context,
        lambda_mm=args.lambda_mm,
        tracking_config=asdict(config),
        arms={
            "A": "Frozen serial SLEAP",
            "B": "Bidirectional cuts of A",
            "C": "Independent bidirectional 2D motion tracker",
        },
        all_identity_corrections_unaccepted=True,
    )
    protocol_target = args.output / "protocol.json"
    if protocol_target.exists():
        if json.loads(protocol_target.read_text(encoding="utf-8")) != clean(protocol):
            raise ValueError("Protocol changed; use a different output directory")
    else:
        write_json(protocol_target, protocol)
    selected_cases = [c for c in cases if not args.case or c["name"] in args.case]
    with provenance(
        "custom2d_comparison",
        args.output,
        name="run_" + ("_".join(args.case) if args.case else "all"),
    ) as rec:
        rec.params_from(args)
        for name in ("run.py", "tracking.py", "downstream.py", "README.md"):
            p = Path(__file__).parent / name
            rec.input(name, p, sha256=sha256(p))
        rec.input("protocol", protocol_target, sha256=sha256(protocol_target))
        arrays, detections, rig, review = load_inputs(args, rec)
        gt = args.input_root / "assets/groundtruth/fish_sex_gt" / EXPERIMENT / "female_sex_gt.csv"
        rec.input("female_gt", gt, sha256=sha256(gt))
        with gt.open(encoding="utf-8-sig", newline="") as f:
            gt_rows = list(csv.DictReader(f))

        def mouths(view, frame):
            data = detections[view]
            lo, hi = np.searchsorted(data.frame_idx, [frame, frame + 1])
            return data.points[lo:hi, 0]

        for case in selected_cases:
            dest = args.output / case["name"]
            dest.mkdir(exist_ok=True)
            if (dest / "summary.json").exists():
                print(f"Reusing completed {case['name']}", flush=True)
                continue
            start = max(0, case["start"] - args.context)
            stop = min(len(arrays["C34"]["xyz"]), case["stop"] + args.context)
            frames = np.arange(start, stop)
            mask = (frames >= case["start"]) & (frames < case["stop"])
            ref = {
                "c28": {k: v[frames] for k, v in arrays["C28"].items()},
                "final": {k: v[frames] for k, v in arrays["C34"].items()},
            }
            anchors = match_mouth_anchors(gt_rows, mouths, case["start"], case["stop"])
            result = {
                **case,
                "context_start": start,
                "context_stop": stop,
                "arms": {},
                "saved_baseline": {},
                "female_baseline": {},
            }
            prefix = {}
            for stage, data in ref.items():
                data["pose"] = selected_pose(detections, rig, data["source"], data["xyz"], frames)
                prefix[stage] = np.flatnonzero(
                    (frames < case["start"]) & np.isfinite(data["xyz"]).all(axis=(1, 2))
                )[:17]
                result["saved_baseline"][stage] = shape_metrics(data, mask)
                result["female_baseline"][stage] = score_stage(data, frames, mask, anchors)
            save_ref = {"frames": frames}
            for stage, data in ref.items():
                save_ref.update({stage + "__" + k: v for k, v in data.items()})
            np.savez_compressed(dest / "saved_baseline.npz", **save_ref)
            for arm in ("A", "B", "C"):
                began = time.perf_counter()
                labels, tracking_stats = {}, {}
                for view in VIEWS:
                    data = detections[view]
                    lo, hi = np.searchsorted(data.frame_idx, [start, stop])
                    frame_idx, points = data.frame_idx[lo:hi], data.points[lo:hi]
                    original = arrays[f"C27_{view}"][lo:hi]
                    if arm == "A":
                        held, events, metadata = original.copy(), [], {"frozen_serial": True}
                    else:
                        tracked = (
                            cut_sleap_links(frame_idx, points, original, view=view, config=config)
                            if arm == "B"
                            else track_bidirectional(frame_idx, points, view=view, config=config)
                        )
                        held, events, metadata = tracked.labels, tracked.events, tracked.metadata
                    assert held.shape == original.shape and (held >= 0).all()
                    labels[view] = np.full(len(data.frame_idx), -1, dtype=np.int64)
                    labels[view][lo:hi] = held
                    tracking_stats[view] = tracking_metrics(
                        frame_idx, points, held, case["start"], case["stop"], view
                    )
                    np.savez_compressed(
                        dest / f"tracking_{arm}_{view}.npz",
                        frame_idx=frame_idx,
                        labels=held,
                        global_rows=np.arange(lo, hi),
                    )
                    write_json(
                        dest / f"tracking_{arm}_{view}.json", dict(metadata=metadata, events=events)
                    )
                print(
                    f"{case['name']} {arm}: 2D complete; reconstructing {len(frames)} frames",
                    flush=True,
                )
                prepared = prepare_window(detections, labels, rig, start, stop)
                reconstructed = solve_window(
                    prepared,
                    lambda_mm=args.lambda_mm,
                    core_interval=(case["start"], case["stop"]),
                    reference_c28=arrays["C28"],
                )
                arm_result = dict(
                    tracking=tracking_stats,
                    alignment={},
                    female={},
                    diagnostics=reconstructed["diagnostics"],
                )
                saved = {"frames": frames}
                for stage in ("c28", "final"):
                    data = reconstructed[stage]
                    arm_result["alignment"][stage] = fixed_prefix_alignment(
                        data, ref[stage]["xyz"], prefix[stage]
                    )
                    arm_result[stage] = shape_metrics(data, mask)
                    arm_result[stage]["source_rows_different_from_saved"] = int(
                        np.count_nonzero(data["source"][mask] != ref[stage]["source"][mask])
                    )
                    arm_result["female"][stage] = score_stage(data, frames, mask, anchors)
                    saved.update(
                        {stage + "__" + k: v for k, v in data.items() if isinstance(v, np.ndarray)}
                    )
                arm_result["elapsed_seconds"] = time.perf_counter() - began
                result["arms"][arm] = arm_result
                path = dest / f"{arm}.npz"
                np.savez_compressed(path, **saved)
                rec.output(f"{case['name']}_{arm}", path, sha256=sha256(path))
                write_json(dest / f"{arm}_summary.json", arm_result)
                print(
                    f"{case['name']} {arm}: final >10mm={arm_result['final']['steps_over_10mm']}; "
                    f"female={arm_result['female']['final']['interval_counts']}",
                    flush=True,
                )
            write_json(dest / "summary.json", result)
            rec.output(case["name"], dest / "summary.json", sha256=sha256(dest / "summary.json"))
        completed = []
        for case in cases:
            p = args.output / case["name"] / "summary.json"
            if p.exists():
                completed.append(json.loads(p.read_text(encoding="utf-8")))
        summary = dict(
            experiment=EXPERIMENT,
            fps=166,
            protocol=protocol,
            cases=completed,
            review_events=review["events"],
            reviewed_panel=review["reviewed_panel"],
            status="complete" if len(completed) == len(cases) else "in_progress",
            identity_accuracy_claim="No verified all-three-fish swap accuracy is available",
            counts=dict(completed_cases=len(completed), requested_cases=len(cases)),
        )
        write_json(args.output / "summary.json", summary)
        rec.output(
            "summary", args.output / "summary.json", sha256=sha256(args.output / "summary.json")
        )


if __name__ == "__main__":
    main()

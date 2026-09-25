"""Reproducible fixed-detection direction audit on development/known-error windows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from directions import DirectionConfig, diagnose, episodes
from reconnect import audit_transient, audit_window

from medaka3d.aquarium import sha256
from medaka3d.calib import load_rig, triangulate
from medaka3d.eval.generative import (
    match_mouth_anchors,
    observed_motion_summary,
    score_female_identity,
)
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
BASELINE_SHA = "a025af23eab568620ca1e2ae5f2e2fa941a788a17c881c70de94936e56930de0"
CALIBRATION = (
    "calibration/fixed_refractive_192x192x186/"
    "phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration"
)


def clean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


class Inputs:
    def __init__(self, root, record):
        self.root = Path(root)
        self.baseline = (
            self.root
            / "work/c34_microflip"
            / (f"{EXPERIMENT}__chain-c28-c33-c30v4-c34")
            / f"C34-label__{EXPERIMENT}__lam0.767.npz"
        )
        digest = sha256(self.baseline)
        if digest != BASELINE_SHA:
            raise ValueError("Frozen C34 baseline hash differs")
        record.input("C34", self.baseline, sha256=digest)
        with np.load(self.baseline, allow_pickle=False) as data:
            self.xyz = data["xyz"]
            self.source = data["source"]
            self.slot_track = data["slot_track"]
        self.rig = load_rig(self.root / CALIBRATION)
        manifest_path = self.root / CALIBRATION / "calibration_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record.input("calibration_manifest", manifest_path, sha256=sha256(manifest_path))
        for entry in manifest["model_files"]:
            path = manifest_path.parent / entry["path"]
            digest = sha256(path)
            if digest != entry["sha256"]:
                raise ValueError(f"Calibration hash changed: {path}")
            record.input(entry["camera"], path, sha256=digest)
        self.cache = {}
        for view in VIEWS:
            path = self.root / "work/c01_s16_predictions/bottomup" / f"{EXPERIMENT}__{view}.npz"
            record.input(f"detections_{view}", path, sha256=sha256(path))
            with np.load(path, allow_pickle=False) as data:
                if str(data["experiment"]) != EXPERIMENT or str(data["view"]) != view:
                    raise ValueError("Cache experiment/view mismatch")
                if tuple(data["nodes"]) != ("mouth", "pectoral", "tail"):
                    raise ValueError("Cache node order mismatch")
                frame_idx, points = data["frame_idx"], data["points"]
                if np.any(np.diff(frame_idx) < 0):
                    raise ValueError("Cache is not sorted")
                counts = np.bincount(frame_idx, minlength=len(self.xyz))
                self.cache[view] = points, np.r_[0, np.cumsum(counts)]

    def mouth(self, view, frame):
        points, offsets = self.cache[view]
        return points[offsets[frame] : offsets[frame + 1], 0]

    def poses(self, frames):
        source = self.source[frames]
        origins, directions, valid = [], [], []
        uv_all = []
        for vi, view in enumerate(VIEWS):
            points, offsets = self.cache[view]
            rows = source[:, :, vi]
            present = rows >= 0
            counts = np.diff(offsets)[frames]
            if np.any(present & (rows >= counts[:, None])):
                raise ValueError("Stored source row is outside its frame")
            uv = np.full((len(frames), 3, 3, 2), np.nan)
            flat = offsets[frames, None] + rows
            uv[present] = points[flat[present]]
            o, d = self.rig.camera(view).rays(uv)
            origins.append(o)
            directions.append(d)
            valid.append(np.isfinite(o).all(-1) & np.isfinite(d).all(-1))
            uv_all.append(uv)
        held = np.stack(valid, axis=-1)
        pose, residual = triangulate(
            np.stack(origins, axis=-2), np.stack(directions, axis=-2), held
        )
        pose[:, :, 1] = self.xyz[frames]
        pose[~np.isfinite(self.xyz[frames]).all(-1)] = np.nan
        observed = (held.sum(-1) >= 2) & np.isfinite(pose).all(-1)
        return pose, observed, residual, np.stack(uv_all, axis=2)


def cases(root):
    note_path = root / "assets/groundtruth/swap_event_gt" / EXPERIMENT / "swap_events.csv"
    with note_path.open(encoding="utf-8-sig", newline="") as stream:
        notes = list(csv.DictReader(stream))
    panel = [dict(name="cg20s", start=75270, stop=78590, category="development20s")]
    for row in notes:
        first, last = int(row["frame_start"]), int(row["frame_end"])
        panel.append(
            dict(
                name=f"known_{first}",
                start=first - 166,
                stop=last + 167,
                category="historical_note_neighbourhood",
                note=row,
                note_chain="pre-v4 C30; slot names cannot be carried as C34 truth",
            )
        )
    for first, last in ((82121, 83265), (90200, 90687), (212276, 213061)):
        panel.append(
            dict(
                name=f"regression_{first}",
                start=first,
                stop=last + 1,
                category="historical_v3_false_repair_interval_C34_control",
            )
        )
    for first in (50000, 150000, 280000):
        panel.append(
            dict(
                name=f"control_{first}",
                start=first,
                stop=first + 332,
                category="unlabelled_control_not_assumed_error_free",
            )
        )
    return panel, note_path


def method_summary(result, case, method):
    held = (result["frames"] >= case["start"]) & (result["frames"] < case["stop"])
    flags = result[method] & held[:, None]
    groups = episodes(result, case["start"], case["stop"], method)
    # Padded adjacent groups may overlap; count the union of source frames.
    review_frames = {
        frame
        for group in groups
        for frame in range(group["start_frame"], group["stop_frame_exclusive"])
    }
    return dict(
        slot_steps=int(flags.sum()),
        native_frames=int(flags.any(-1).sum()),
        episodes=len(groups),
        flagged_seconds=float(flags.any(-1).sum() / 166),
        inspection_seconds_with_padding=float(len(review_frames) / 166),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument(
        "--output", type=Path, default=Path("work/bidirectional_reconnect_v1/analysis_v2")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance("bidirectional_development", args.output) as record:
        record.params_from(args)
        lab = Path(__file__).resolve().parent
        for name in ("README.md", "run.py", "directions.py", "reconnect.py"):
            record.input(name, lab / name, sha256=sha256(lab / name))
        panel, note_path = cases(args.input_root)
        record.input("historical_notes", note_path, sha256=sha256(note_path))
        write_json(
            args.output / "protocol.json",
            dict(
                cases=panel,
                scope="development only",
                baseline_sha256=BASELINE_SHA,
                settings=vars(DirectionConfig()),
                context_frames=53,
                revision=(
                    "v2: complete 50-step boundary context, union review duration, "
                    "explicit source-triangulation residual name; primary thresholds unchanged"
                ),
            ),
        )
        inputs = Inputs(args.input_root, record)
        gt_path = (
            args.input_root / "assets/groundtruth/fish_sex_gt" / EXPERIMENT / "female_sex_gt.csv"
        )
        record.input("female_gt", gt_path, sha256=sha256(gt_path))
        with gt_path.open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        anchors = match_mouth_anchors(records, inputs.mouth, 0, len(inputs.xyz))
        gt_frames = np.array([a.frame for a in anchors])
        _, gt_observed, _, _ = inputs.poses(gt_frames)
        female = score_female_identity(
            gt_frames, inputs.source[gt_frames], inputs.slot_track[gt_frames] >= 0, anchors
        )
        female["availability_semantics"] = "C34-label slot_track>=0, matches archived label audit"
        female["mouth_geometry_support_score"] = score_female_identity(
            gt_frames, inputs.source[gt_frames], gt_observed[:, :, 0], anchors
        )
        write_json(args.output / "female_baseline.json", female)
        report = dict(
            experiment=EXPERIMENT,
            baseline_sha256=BASELINE_SHA,
            female_baseline=female["interval_counts"],
            cases=[],
        )
        for case in panel:
            print(
                f"Analyzing {case['name']} ({case['stop'] - case['start']} core frames)", flush=True
            )
            frames = np.arange(max(0, case["start"] - 53), min(len(inputs.xyz), case["stop"] + 53))
            xyz = inputs.xyz[frames]
            source = inputs.source[frames]
            pose, observed, residual, uv = inputs.poses(frames)
            primary = diagnose(xyz, frames)
            sensitivity = []
            for window in (17, 33, 50):
                result = (
                    primary
                    if window == 17
                    else diagnose(xyz, frames, DirectionConfig(window=window))
                )
                for threshold in (60, 80, 100):
                    sensitivity_result = dict(result)
                    distance = result["distance"]
                    big = result["step_mm"] > 1
                    forward = distance | (big & (result["past_angle_deg"] >= threshold))
                    sensitivity_result["forward"] = forward
                    sensitivity_result["bidirectional"] = forward | (
                        big & (result["future_angle_deg"] >= threshold)
                    )
                    sensitivity.append(
                        dict(
                            window=window,
                            angle_deg=threshold,
                            **{
                                method: method_summary(sensitivity_result, case, method)
                                for method in ("distance", "forward", "bidirectional")
                            },
                        )
                    )
            groups = episodes(primary, case["start"], case["stop"])
            for group in groups:
                lo = group["start_frame"] - int(frames[0])
                hi = group["stop_frame_exclusive"] - int(frames[0]) - 1
                group["persistent_audit"] = audit_window(xyz, frames, lo, hi, observed[:, :, 1])
                # Test short excursions at each flagged edge without forcing the
                # padded episode boundaries to coincide with transient entry/exit.
                trials = []
                for entry in sorted({e["frame"] for e in group["edges"]}):
                    for length in (1, 2, 3):
                        start = entry - int(frames[0])
                        stop = start + length - 1
                        trials.append(
                            dict(
                                entry_frame=entry,
                                length=length,
                                audit=audit_transient(
                                    xyz,
                                    frames,
                                    start,
                                    stop,
                                    observed[:, :, 1],
                                    reference_window=(lo, hi),
                                ),
                            )
                        )
                group["transient_audits"] = trials
            core = (frames >= case["start"]) & (frames < case["stop"])
            held_edges = (
                (primary["frames"] >= case["start"]) & (primary["frames"] < case["stop"])
            )[:, None]
            extra = primary["bidirectional"] & ~primary["forward"] & held_edges
            diagnostic = dict(
                case=case,
                methods={
                    method: method_summary(primary, case, method)
                    for method in ("distance", "forward", "bidirectional")
                },
                bidirectional_extra_slot_steps=int(extra.sum()),
                sensitivity=sensitivity,
                episodes=groups,
                motion=observed_motion_summary(xyz[core], frames[core]),
                full_pose_fraction=np.isfinite(pose[core]).all(axis=(2, 3)).mean(0),
                accepted_identity_corrections=0,
                status="proposals_only_pending_independent_identity_evidence",
                residual_semantics=(
                    "triangulated_source_residual_mm describes triangulated source rays, "
                    "not frozen C34 pectoral coordinates; unused in detection or reconnection gates"
                ),
            )
            if "note" in case:
                first = int(case["note"]["frame_start"])
                last = int(case["note"]["frame_end"])
                nearby = (primary["frames"] >= first - 2) & (primary["frames"] <= last + 2)
                diagnostic["historical_note_proximity"] = {
                    method: bool(primary[method][nearby].any())
                    for method in ("distance", "forward", "bidirectional")
                }
            case_dir = args.output / case["name"]
            case_dir.mkdir()
            np.savez_compressed(
                case_dir / "poses.npz",
                frames=frames,
                pose=pose,
                xyz=xyz,
                source=source,
                observed=observed,
                triangulated_source_residual_mm=residual,
                uv=uv,
                slot_track=inputs.slot_track[frames],
                core=core,
                **{f"diag_{k}": v for k, v in primary.items()},
            )
            write_json(case_dir / "analysis.json", diagnostic)
            row = {
                key: value
                for key, value in diagnostic.items()
                if key not in ("sensitivity", "episodes")
            }
            row["persistent_statuses"] = dict(
                Counter(g["persistent_audit"]["status"] for g in groups)
            )
            row["transient_statuses"] = dict(
                Counter(t["audit"]["status"] for g in groups for t in g["transient_audits"])
            )
            row["candidate_geometry_proposals"] = [
                dict(
                    kind="persistent",
                    episode=g["index"],
                    frame=g["start_frame"],
                    audit=g["persistent_audit"],
                )
                for g in groups
                if g["persistent_audit"].get("suggested_permutation") is not None
            ]
            row["candidate_geometry_proposals"] += [
                dict(
                    kind="transient",
                    episode=g["index"],
                    frame=t["entry_frame"],
                    length=t["length"],
                    audit=t["audit"],
                )
                for g in groups
                for t in g["transient_audits"]
                if t["audit"].get("suggested_permutation") is not None
            ]
            report["cases"].append(row)
            for path in case_dir.iterdir():
                record.output(f"{case['name']}/{path.name}", path, sha256=sha256(path))
            print(
                f"  methods={[(m, row['methods'][m]['slot_steps']) for m in row['methods']]} "
                f"proposals={len(row['candidate_geometry_proposals'])}",
                flush=True,
            )
        write_json(args.output / "summary.json", report)
        for name in ("protocol.json", "female_baseline.json", "summary.json"):
            path = args.output / name
            record.output(name, path, sha256=sha256(path))
        print("Completed development panel", flush=True)


if __name__ == "__main__":
    main()

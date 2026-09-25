"""Evaluate frozen ID reconnection on immutable raw recovery arrays.

This separate command never reruns raw tracking or changes its output. Human
labels and the raw comparison metrics are parsed only after reconnection.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from medaka3d.calib import load_rig
from medaka3d.eval.generative import match_mouth_anchors
from medaka3d.provenance import provenance

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import (
    CALIBRATION_DIR,
    EXPERIMENT,
    CachedDetections,
    open_videos,
    seek_read,
    sha256,
    write_previews,
)
from benchmark_reacquisition import candidate_metrics, verify_inputs
from reacquisition import RecoveryTrackResult
from reconnect import ReconnectConfig, reconnect


def load_raw(source: Path):
    """Load geometry and provenance, without parsing the comparison/GT labels."""
    provenance_path = source / "benchmark.provenance.json"
    record = json.loads(provenance_path.read_text(encoding="utf-8"))
    if record["status"] != "ok":
        raise ValueError("raw source run did not complete successfully")
    candidate_path, comparison_path = source / "candidate.npz", source / "comparison.json"
    for name, path in (("candidate", candidate_path), ("comparison", comparison_path)):
        if sha256(path) != record["outputs"][name]["sha256"]:
            raise ValueError(f"raw {name} differs from its recorded output checksum")
    with np.load(candidate_path, allow_pickle=False) as arrays:
        fields = {field.name for field in dataclasses.fields(RecoveryTrackResult)}
        values = {key: arrays[key].copy() for key in arrays.files if key in fields}
        values["diagnostics"] = {
            "algorithm": "refractive-generative-reacquisition",
            "source_candidate_sha256": sha256(candidate_path),
            "initializations": 1,
        }
        raw = RecoveryTrackResult(**values)
        np.testing.assert_array_equal(raw.accepted_xyz, arrays["accepted_xyz"])
    expected = record["params"]["registered_window"]
    np.testing.assert_array_equal(raw.frame_idx, np.arange(expected["start"], expected["stop"]))
    initial = np.asarray(record["params"]["seed"]["pose_xyz"], dtype=float)
    return raw, initial, record


def assert_only_relabelled(raw, connected):
    """Ensure the postpass did not silently refit, interpolate or duplicate fish."""
    np.testing.assert_array_equal(connected.frame_idx, raw.frame_idx)
    permutation = connected.permutation
    np.testing.assert_array_equal(np.sort(permutation, axis=1),
                                  np.tile(np.arange(3), (len(raw.frame_idx), 1)))
    rows = np.arange(len(raw.frame_idx))[:, None]
    for name in ("pose_xyz", "observed", "node_source", "xyz"):
        np.testing.assert_array_equal(getattr(connected, name),
                                      np.asarray(getattr(raw, name))[rows, permutation])
    if np.any(raw.geometry_review[rows, permutation] & ~connected.geometry_review):
        raise ValueError("postpass removed a geometry review flag")


def json_safe_summary(summary):
    """Keep an infeasible model result serializable without disguising its status."""
    replaced = []

    def convert(value, path):
        if isinstance(value, dict):
            return {key: convert(item, f"{path}.{key}" if path else str(key))
                    for key, item in value.items()}
        if isinstance(value, np.ndarray):
            return convert(value.tolist(), path)
        if isinstance(value, (list, tuple)):
            return [convert(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            replaced.append(path)
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    safe = convert(summary, "")
    safe["serialization"] = {
        "nonfinite_values_written_as_null": replaced,
        "scope": "NPZ retains numeric infinities; model feasibility/review flags are unchanged",
    }
    return safe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path,
                        help="Optional read-only original videos for post-inference overlays")
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(f"refusing to overwrite an existing postpass output: {args.out}")
    raw, initial, source_record = load_raw(args.source_run)
    staged = verify_inputs(args.input_root)
    lab, repo = Path(__file__).resolve().parent, Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True, exist_ok=False)
    with provenance("frozen_geometry_id_reconnection", args.out, name="benchmark") as record:
        record.param("source_run", str(args.source_run))
        record.param("input_root", str(args.input_root))
        record.param("raw_root", str(args.raw_root) if args.raw_root else None)
        record.param("experiment", EXPERIMENT)
        record.param("frames", [int(raw.frame_idx[0]), int(raw.frame_idx[-1]) + 1])
        record.param("seed", source_record["params"]["seed"])
        record.param("raw_registered_window", source_record["params"]["registered_window"])
        record.input("staged_manifest", staged, sha256=sha256(staged))
        for name in ("candidate.npz", "comparison.json", "benchmark.provenance.json"):
            path = args.source_run / name
            record.input("raw_" + name, path, sha256=sha256(path))
        code = [
            Path(__file__), lab / "reconnect.py", lab / "tracker.py", lab / "reacquisition.py",
            lab / "fast_fit.py", lab / "observations.py", lab / "ordered_prefetch.py",
            lab / "stage_inputs.py",
            lab / "benchmark_reacquisition.py", lab / "benchmark.py",
            repo / "src/medaka3d/eval/generative.py", repo / "src/medaka3d/eval/reacquisition.py",
            repo / "src/medaka3d/calib.py", repo / "src/medaka3d/provenance.py",
            repo / "src/medaka3d/types.py",
        ]
        hashes = {str(path): sha256(path) for path in code}
        record.param("code_hashes", hashes)
        record.param("snapshot_scope",
                     "local imported implementation files; project environment required")
        for path in code:
            target = args.out / "code_snapshot" / path.relative_to(repo)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        config = ReconnectConfig()
        record.param("reconnect_config", dataclasses.asdict(config))
        started = time.perf_counter()
        result = reconnect(raw, initial, config)
        postpass_seconds = time.perf_counter() - started
        assert_only_relabelled(raw, result)
        candidate_path = args.out / "candidate.npz"
        np.savez_compressed(
            candidate_path, frame_idx=result.frame_idx, pose_xyz=result.pose_xyz, xyz=result.xyz,
            accepted_xyz=result.accepted_xyz, observed=result.observed,
            node_source=result.node_source, review=result.review, margin=result.margin,
            identity_margin=result.identity_margin, appearance=result.appearance,
            appearance_quality=result.appearance_quality, geometry_review=result.geometry_review,
            identity_review=result.identity_review, permutation=result.permutation,
            state_min_cost=result.state_min_cost,
            appearance_margin_gain=result.appearance_margin_gain,
            epoch_appearance_margin=result.epoch_appearance_margin,
            appearance_verified=result.appearance_verified,
            connection_review=result.connection_review,
        )
        record.output("candidate", candidate_path, sha256=sha256(candidate_path))
        # Labels and earlier scoring outcomes cannot influence the completed postpass.
        source_summary = json.loads((args.source_run / "comparison.json").read_text())
        result.diagnostics["raw_algorithm"] = source_summary["candidate"]["diagnostics"]
        gt = args.input_root / "assets/groundtruth/fish_sex_gt" / EXPERIMENT / "female_sex_gt.csv"
        detections, rig = CachedDetections(args.input_root, EXPERIMENT), load_rig(
            args.input_root / CALIBRATION_DIR)
        with gt.open(encoding="utf-8-sig", newline="") as stream:
            anchors = match_mouth_anchors(csv.DictReader(stream), detections.mouth,
                                         int(raw.frame_idx[0]), int(raw.frame_idx[-1]) + 1)
        candidate = candidate_metrics(result, result.frame_idx, initial, anchors, rig)
        previews = []
        if args.raw_root:
            chosen = [int(Path(name).stem.split("_")[-1]) for name in source_summary["previews"]]
            _, captures, videos = open_videos(args.raw_root, EXPERIMENT, detections.n_frames)
            try:
                snapshots = {
                    frame: {view: seek_read(capture, frame, view)
                            for view, capture in captures.items()}
                    for frame in chosen
                }
            finally:
                for capture in captures.values():
                    capture.release()
            record.param("videos", videos)
            with np.load(source_record["params"]["seed"]["baseline"], allow_pickle=False) as base:
                available = base["slot_track"][raw.frame_idx] >= 0
                baseline_xyz = np.where(available[..., None], base["xyz"][raw.frame_idx], np.nan)
            previews = write_previews(args.out, snapshots, result, baseline_xyz, rig)
        if not all(sha256(path) == hashes[str(path)] for path in code):
            raise RuntimeError(
                "postpass source changed during run; preserve this output as invalid")
        summary = {
            "protocol": "fixed_raw_geometry_global_id_reconnection",
            "experiment": EXPERIMENT, "n_groups": 1,
            "window": source_summary["window"], "frames": source_summary["frames"],
            "technical_completion": True, "source_run": str(args.source_run),
            "postpass_seconds": postpass_seconds,
            "raw_tracking_seconds": source_summary["tracking_seconds"],
            "coordinate_refitting_or_interpolation": False,
            "candidate": candidate, "raw_candidate": source_summary["candidate"],
            "baseline_arms": source_summary["baseline_arms"], "previews": previews,
            "limitations": [
                "Reconnection chooses permutations of fixed raw geometry, not new 3D hypotheses.",
                "Shared female endpoint scoring cannot validate male IDs or returning swaps.",
                "Model cost gaps and appearance margin gains are not calibrated probabilities.",
                "Availability and technical completion do not establish correct biological IDs.",
            ],
        }
        comparison = args.out / "comparison.json"
        comparison.write_text(json.dumps(json_safe_summary(summary), indent=2, allow_nan=False),
                              encoding="utf-8")
        record.output("comparison", comparison, sha256=sha256(comparison))
        if previews:
            record.output("preview", args.out / "preview.html")
        print(json.dumps({"out": str(args.out), "postpass_seconds": postpass_seconds,
                          "technical_completion": True, "returned_frames": len(result.frame_idx)}),
              flush=True)


if __name__ == "__main__":
    main()

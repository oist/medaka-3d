"""Post-hoc development-only appearance-emission ablation on fixed raw geometry."""

from __future__ import annotations

import csv
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260908_generative_tracking"
sys.path.insert(0, str(LAB))

from benchmark import CALIBRATION_DIR, EXPERIMENT, CachedDetections, sha256
from benchmark_reacquisition import candidate_metrics
from benchmark_reconnect import assert_only_relabelled, load_raw
from reconnect import ReconnectConfig, reconnect

from medaka3d.calib import load_rig
from medaka3d.eval.generative import match_mouth_anchors
from medaka3d.provenance import provenance

SOURCE = REPO / "work/generative_tracking/recovery_v1_nominal_dev_transient"
INPUTS = REPO / "work/generative_tracking_inputs"
OUT = REPO / "work/generative_tracking/appearance_ablation_v1_transient"


def changes(frames, permutation):
    start = np.r_[0, np.flatnonzero(np.any(np.diff(permutation, axis=0), axis=1)) + 1]
    ends = np.r_[start[1:], len(frames)]
    return [{"start": int(frames[a]), "stop": int(frames[b - 1]) + 1,
             "persistent_id_to_raw_slot": permutation[a].tolist()}
            for a, b in zip(start, ends, strict=True)]


def main():
    raw, initial, source_record = load_raw(SOURCE)
    np.testing.assert_array_equal(raw.frame_idx, np.arange(333519, 333719))
    OUT.mkdir(parents=True, exist_ok=False)
    with provenance("posthoc_development_appearance_emission_ablation", OUT,
                    name="benchmark") as record:
        record.param("scope", "Post-hoc v1 development transient only, never formal validation or minute.")
        record.param("source_run", str(SOURCE))
        record.param("information_barrier", "Both default and zero-appearance inference finish before "
                     "GT labels or source comparison outcomes are parsed. No core/config file changes.")
        record.param("seed", source_record["params"]["seed"])
        for name in ("candidate.npz", "comparison.json", "benchmark.provenance.json"):
            path = SOURCE / name
            record.input("raw_" + name, path, sha256=sha256(path))
        paths = [Path(__file__), LAB / "reconnect.py", LAB / "benchmark_reconnect.py",
                 LAB / "benchmark_reacquisition.py", LAB / "benchmark.py", LAB / "tracker.py",
                 LAB / "reacquisition.py", REPO / "src/medaka3d/eval/generative.py",
                 REPO / "src/medaka3d/eval/reacquisition.py", REPO / "src/medaka3d/calib.py",
                 REPO / "src/medaka3d/provenance.py"]
        hashes = {str(path): sha256(path) for path in paths}
        record.param("code_hashes", hashes)
        for path in paths:
            target = OUT / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        default = ReconnectConfig()
        zero = dataclasses.replace(default, appearance_weight=0.0)
        configs = {"default": default, "without_appearance_emission": zero}
        record.param("configs", {name: dataclasses.asdict(cfg) for name, cfg in configs.items()})
        results, seconds = {}, {}
        for name, config in configs.items():
            start = time.perf_counter()
            result = reconnect(raw, initial, config)
            seconds[name] = time.perf_counter() - start
            assert_only_relabelled(raw, result)
            np.testing.assert_array_equal(result.permutation[0], np.arange(3))
            np.testing.assert_array_equal(result.accepted_xyz,
                                          np.where(result.review[..., None], np.nan, result.xyz))
            results[name] = result
            output_dir = OUT / name
            output_dir.mkdir()
            candidate = output_dir / "candidate.npz"
            fields = {field.name: getattr(result, field.name)
                      for field in dataclasses.fields(result)
                      if field.name != "diagnostics" and getattr(result, field.name) is not None}
            np.savez_compressed(candidate, **fields, accepted_xyz=result.accepted_xyz)
            record.output(name + "_candidate", candidate, sha256=sha256(candidate))
        # No label or source score is accessed above this point.
        gt = INPUTS / "assets/groundtruth/fish_sex_gt" / EXPERIMENT / "female_sex_gt.csv"
        detections = CachedDetections(INPUTS, EXPERIMENT)
        rig = load_rig(INPUTS / CALIBRATION_DIR)
        with gt.open(encoding="utf-8-sig", newline="") as stream:
            anchors = match_mouth_anchors(csv.DictReader(stream), detections.mouth, 333519, 333719)
        record.input("female_labels_evaluation_only", gt, sha256=sha256(gt))
        raw_summary = json.loads((SOURCE / "comparison.json").read_text(encoding="utf-8"))
        summaries = {}
        for name, result in results.items():
            metrics = candidate_metrics(result, result.frame_idx, initial, anchors, rig)
            summaries[name] = {
                "config": dataclasses.asdict(configs[name]), "postpass_seconds": seconds[name],
                "permutation_epochs": changes(result.frame_idx, result.permutation),
                "candidate": metrics,
                "identity_review_animal_frames": int(result.identity_review.sum()),
                "identity_review_by_id": result.identity_review.sum(0).tolist(),
                "connection_review_by_id": result.connection_review.sum(0).tolist(),
                "accepted_by_id": np.isfinite(result.accepted_xyz).all(-1).sum(0).tolist(),
                "final_permutation": result.permutation[-1].tolist(),
                "appearance_margin_gain_max": float(np.max(np.abs(result.appearance_margin_gain))),
            }
            path = OUT / name / "comparison.json"
            path.write_text(json.dumps(summaries[name], indent=2, allow_nan=False), encoding="utf-8")
            record.output(name + "_comparison", path, sha256=sha256(path))
        assert all(sha256(path) == hashes[str(path)] for path in paths)
        original, ablated = results.values()
        summary = {
            "protocol": "posthoc_development_appearance_emission_ablation",
            "frames": [333519, 333719], "n_groups": 1,
            "permuted_id_frame_difference_count": int((original.permutation != ablated.permutation).sum()),
            "source_difference_count": int((original.node_source != ablated.node_source).sum()),
            "variants": summaries, "raw_candidate": raw_summary["candidate"],
            "limitations": [
                "Both variants relabel exactly the same fixed v1 geometry; no new detections or 3D fit.",
                "Only appearance_weight differs; length, motion, bounds, review rules and templates are fixed.",
                "Zero appearance weight removes image emissions from the path objective. Image diagnostics "
                "still run, but appearance-margin gain is zero, so prior ID uncertainty cannot be cleared by it.",
                "One female anchor cannot establish interval tracking accuracy or either male identity.",
                "This post-hoc development contrast does not modify the preregistered default used on validation/minute.",
            ],
        }
        path = OUT / "comparison.json"
        path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("comparison", path, sha256=sha256(path))
        print(json.dumps({name: {key: row[key] for key in (
            "postpass_seconds", "final_permutation", "identity_review_by_id", "connection_review_by_id",
            "accepted_by_id", "permutation_epochs")} for name, row in summaries.items()}, indent=2))
        print(json.dumps({name: {key: row["candidate"][key] for key in (
            "female_identity_raw", "female_identity_accepted", "accepted_observed_animal_frames")}
                          for name, row in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()

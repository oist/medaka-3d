"""Independent read-only checks of the sealed fixed57 evaluation.

No evaluator/scorer imports; only the authoritative frozen refractive projector
is reused. Output is a separate QA receipt outside the sealed evaluation folder.
"""

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from medaka3d.calib import load_rig  # noqa: E402

WORK = REPO / "work/three_body"
ROOT = WORK / "resumed_nodewise_beam_female57_comparison_v1"
OUT = WORK / "resumed_nodewise_beam_female57_comparison_independent_qa_v1.json"
START, STOP = 66400, 116200
VIEWS = ("top_cam", "side1_cam", "side2_cam")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    assert not OUT.exists(), "fresh QA artifact required"
    seal_path = ROOT / "evaluation.provenance.json"
    seal, summary = read(seal_path), read(ROOT / "summary.json")
    assert seal["status"] == summary["status"] == "ok"
    input_checks = {}
    for path, expected in seal["inputs"].items():
        actual = digest(Path(path))
        assert actual == expected, path
        input_checks[path] = {"expected": expected, "actual": actual, "matched": True}
    output_checks = {}
    for name, descriptor in seal["outputs"].items():
        path = ROOT / name
        assert path.resolve() == Path(descriptor["path"]).resolve()
        actual = digest(path)
        assert actual == descriptor["sha256"] and path.stat().st_size == descriptor["size_bytes"]
        output_checks[name] = {"sha256": actual, "hash_and_size_matched": True}

    anchor_root = WORK / "female_57_five_minute_endpoint_evaluation_v1"
    frames = read(anchor_root / "fixed_female_anchor_set.json")["frames"]
    anchors = read(anchor_root / "anchor_observations.json")["matches"]
    assert len(frames) == len(set(frames)) == 57
    assert frames == [anchor["frame"] for anchor in anchors]
    assert all(START <= frame < STOP for frame in frames)
    requested_support = {(a["frame"], view) for a in anchors for view in a["pixels"]}
    assert len(requested_support) == 112
    manifest = [Path(path) for path in seal["inputs"] if Path(path).name == "calibration_manifest.json"]
    assert len(manifest) == 1
    rig = load_rig(manifest[0].parent)

    paths = {
        "sparse_frozen": WORK / "sparse_instance_five_minute_continuation_v1/candidate.npz",
        "nodewise_raw": WORK / "nodewise_body_row_five_minute_resume_099600_v1/candidate.npz",
        "nodewise_beam": WORK / "resumed_nodewise_five_minute_identity_beam256_v1/beam_remapped_stream.npz",
    }
    modes = ("raw_slots", "review_gated")
    recomputed = {mode: {} for mode in modes}
    endpoint_checks, coverage_checks = {}, {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            allframes = data["frame_idx"]
            keep = (allframes >= START) & (allframes < STOP)
            assert np.array_equal(allframes[keep], np.arange(START, STOP))
            sources = data["sources"][keep]
            observed = data["observed"][keep]
            pose = data["pose_xyz"][keep]
            review = data["identity_review"][keep]
        coverage = summary["methods"][name]["coverage"]
        assert coverage["total_fish_frames"] == 49800 * 3
        assert coverage["total_node_frames"] == 49800 * 9
        assert coverage["observed_node_frames"] == int(np.count_nonzero(observed))
        assert coverage["observed_node_fraction"] == float(np.count_nonzero(observed) / observed.size)
        assert coverage["accepted_identity_frames"] == int(np.count_nonzero(~review))
        assert coverage["accepted_identity_fraction"] == float(np.count_nonzero(~review) / review.size)
        assert coverage["accepted_observed_mouth_fraction"] == float(np.count_nonzero(~review & observed[:, :, 0]) / review.size)
        for slot in range(3):
            assert coverage["slots"][slot]["accepted_identity_frames"] == int(np.count_nonzero(~review[:, slot]))
            assert coverage["slots"][slot]["observed_node_fraction"] == float(np.count_nonzero(observed[:, slot]) / observed[:, slot].size)
        coverage_checks[name] = {"native_frame_and_node_denominators_recomputed": True,
                                 "accepted_id_fraction": coverage["accepted_identity_fraction"],
                                 "observed_node_fraction": coverage["observed_node_fraction"]}

        endpoint_checks[name] = {}
        for mode in modes:
            identities, errors = [], {}
            for anchor in anchors:
                index = anchor["frame"] - START
                matches = []
                duplicated = False
                for view, row in anchor["detection_rows"].items():
                    selected = [slot for slot in range(3) if sources[index, slot, VIEWS.index(view), 0] == row]
                    duplicated |= len(selected) > 1
                    matches.extend(selected)
                identity = None
                if not duplicated and len(set(matches)) == 1:
                    slot = matches[0]
                    if observed[index, slot, 0] and (mode == "raw_slots" or not review[index, slot]):
                        identity = slot
                identities.append(identity)
                if identity is not None:
                    for view, target in anchor["pixels"].items():
                        predicted = rig.camera(view).project(pose[index, identity, 0])
                        if np.isfinite(predicted).all():
                            residual = predicted - target
                            errors[(anchor["frame"], view)] = float(np.sqrt(np.dot(residual, residual)))
            saved = summary["methods"][name][mode]
            assert identities == [row["identity"] for row in saved["anchors"]]
            assert frames == [row["frame"] for row in saved["anchors"]]
            outcomes = ["unscored" if left is None or right is None else "preserved" if left == right else "swap"
                        for left, right in zip(identities[:-1], identities[1:], strict=True)]
            counts = {key: Counter(outcomes)[key] for key in ("preserved", "swap", "unscored")}
            assert saved["interval_counts"] == counts
            assert saved["available_anchors"] == sum(item is not None for item in identities)
            assert outcomes == [row["outcome"] for row in saved["intervals"]]
            assert [[left, right] for left, right in zip(frames[:-1], frames[1:], strict=True)] == [[row["start"], row["stop"]] for row in saved["intervals"]]
            endpoint_checks[name][mode] = {"all_57_assignments_and_56_adjacent_intervals_recomputed": True, "interval_counts": counts}
            recomputed[mode][name] = errors

    with (ROOT / "human_mouth_reprojection.csv").open(encoding="utf-8", newline="") as handle:
        csv_data = list(csv.DictReader(handle))
    csv_keys = [(row["mode"], int(row["frame"]), row["view"]) for row in csv_data]
    assert len(csv_keys) == len(set(csv_keys)), "duplicate paired CSV observations"
    common_checks = {}
    for mode in modes:
        sets = [set(errors) for errors in recomputed[mode].values()]
        support = set.intersection(*sets)
        saved = summary["human_mouth_reprojection_paired_support"][mode]
        assert support == {tuple(row) for row in saved["common_frame_view_support"]}
        assert saved["common_view_observations"] == len(support)
        assert saved["common_anchor_frames"] == len({frame for frame, _ in support})
        assert saved["requested_view_observations"] == len(requested_support)
        rows = [row for row in csv_data if row["mode"] == mode]
        assert support == {(int(row["frame"]), row["view"]) for row in rows}
        metrics = {}
        for name in paths:
            for row in rows:
                key = (int(row["frame"]), row["view"])
                assert np.isclose(float(row[name + "_error_px"]), recomputed[mode][name][key], rtol=0, atol=1e-12)
            values = np.array([recomputed[mode][name][key] for key in sorted(support)])
            statistics = {"mean_px": float(np.mean(values)), "median_px": float(np.median(values)),
                          "rmse_px": float(np.sqrt(np.sum(values * values) / len(values))),
                          "p95_px": float(np.quantile(values, .95)), "max_px": float(np.max(values))}
            assert saved["methods"][name]["view_observations"] == len(support)
            for key, value in statistics.items():
                assert np.isclose(value, saved["methods"][name][key], rtol=0, atol=1e-12), (mode, name, key)
            metrics[name] = statistics
        common_checks[mode] = {"support_exact": True, "csv_errors_and_summary_statistics_recomputed": True,
                               "common_anchor_frames": saved["common_anchor_frames"], "common_view_observations": len(support),
                               "metrics": metrics}

    result = {
        "stage": "resumed_nodewise_beam_female57_comparison_independent_qa_v1", "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_provenance_sha256": digest(seal_path),
        "qa_source_sha256": digest(Path(__file__)),
        "input_hash_checks": input_checks, "output_hash_checks": output_checks,
        "fixed_57_frames": frames, "fixed_57_frames_exact": True,
        "endpoint_checks": endpoint_checks, "coverage_checks": coverage_checks,
        "paired_reprojection_checks": common_checks,
        "scope": "Independent direct array computations, no evaluator/scorer imports. Frozen authoritative refractive projector reused; no recalibration.",
        "sealed_artifacts_modified": False, "human_reviews_created": False,
        "promotion_implications": {
            "fixed_female_endpoint_gate": "pass for Nodewise raw and beam",
            "confirmed_three_fish_failure_gate": "pending: no human review results, male identity remains unknown",
            "accepted_id_coverage_gate": "fail: Nodewise 33.33% overall; slot 0 and slot 2 accepted 0%",
            "observed_node_coverage": "aggregate and all three node-type aggregates exceed Sparse on the identical 49,800 frames; method-local slots are not biological truth",
            "human_mouth_reprojection_gate": "mixed: raw paired mean/RMSE/max improve, median and P95 slightly worsen; no predeclared tolerance allows a blanket no-worse claim",
            "raw_beam_disagreement_gate": "no raw/beam disagreements; conditional beam ties are still review risks, not truth",
            "overall_promotion": "not passed",
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "inputs_verified": len(input_checks), "outputs_verified": len(output_checks),
                      "paired_support": {mode: [item["common_anchor_frames"], item["common_view_observations"]] for mode, item in common_checks.items()},
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()

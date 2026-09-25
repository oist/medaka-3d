"""Independently check saved detection retention and accepted cross-view joins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from evaluate import geometry_metrics, validate_sources

from medaka3d.aquarium import sha256
from medaka3d.calib import load_rig, triangulate
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
CALIBRATION = (
    "calibration/fixed_refractive_192x192x186/"
    "phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration"
)


def check_join(event, data, stage, frames, rig, config):
    vi, slot = VIEWS.index(event["view"]), event["slot"]
    a, b = np.searchsorted(frames, [event["from_frame"], event["to_frame"]])
    assert 0 <= a < b < len(frames)
    assert (frames[a], frames[b]) == (event["from_frame"], event["to_frame"])
    assert frames[b] - frames[a] <= config["bridge_gap_frames"]
    assert np.all(np.diff(frames[a : b + 1]) == 1)
    assert not stage["identity_review"][a : b + 1, slot].any()
    target = data[event["view"]]
    left_label, right_label = event["left_label"], event["right_label"]
    left_rows = np.flatnonzero(target["labels"] == left_label)
    right_rows = np.flatnonzero(target["labels"] == right_label)
    assert left_label != right_label and len(left_rows) and len(right_rows)
    assert left_rows[-1] == event["from_row"]
    assert right_rows[0] == event["to_row"]
    assert target["frame_idx"][left_rows[-1]] == frames[a]
    assert target["frame_idx"][right_rows[0]] == frames[b]
    assert not target["clean_edge"][right_rows[0]]
    other = [k for k in range(3) if k != vi]
    origins, directions = [], []
    for k in other:
        view = VIEWS[k]
        local = stage["source"][a : b + 1, slot, k]
        assert np.all(local >= 0)
        rows = np.searchsorted(data[view]["frame_idx"], frames[a : b + 1]) + local
        labels = data[view]["labels"][rows]
        assert np.all(labels == labels[0])
        assert labels[0] == event["supporting_labels"][view]
        assert data[view]["clean_edge"][rows[1:]].all()
        o, d = rig.camera(view).rays(data[view]["points"][rows, :2])
        origins.append(o)
        directions.append(d)
    origins, directions = np.stack(origins, -2), np.stack(directions, -2)
    pose, residual = triangulate(origins, directions)
    assert np.isfinite(pose).all()
    assert np.all(residual <= config["ray_gate_mm"] + 1e-8)
    assert np.all(pose >= np.asarray(config["tank_lower_mm"]) - 1e-8)
    assert np.all(pose <= np.asarray(config["tank_upper_mm"]) + 1e-8)
    speed = np.linalg.norm(np.diff(pose[:, 1], axis=0), axis=-1)
    assert np.all(speed <= config["maximum_speed_mm_per_frame"] + 1e-8)
    view = VIEWS[vi]
    projected = rig.camera(view).project(pose[[0, -1]])
    errors, margins = [], []
    for index, t in enumerate((a, b)):
        begin, end = np.searchsorted(data[view]["frame_idx"], [frames[t], frames[t] + 1])
        assert stage["source"][t, slot, vi] >= 0
        selected = begin + stage["source"][t, slot, vi]
        assert selected == event["from_row" if index == 0 else "to_row"]
        assert data[view]["labels"][selected] == (left_label if index == 0 else right_label)
        points = data[view]["points"][begin:end, :2]
        error = np.sqrt(np.mean(np.sum((points[selected - begin] - projected[index]) ** 2, -1)))
        assert np.isfinite(error) and error <= config["bridge_projection_gate_px"] + 1e-8
        competitors = []
        for row in range(begin, end):
            if row == selected:
                continue
            distance = np.linalg.norm(data[view]["points"][row, :2] - projected[index], axis=-1)
            finite = distance[np.isfinite(distance)]
            competitors.append(float(np.sqrt(np.mean(finite**2))) if len(finite) else 0.0)
        margin = min(competitors) - error if competitors else np.inf
        assert margin >= config["bridge_competitor_margin_px"] - 1e-8
        errors.append(error)
        margins.append(margin)
    np.testing.assert_allclose(errors, event["endpoint_projection_error_px"], atol=1e-6)
    expected_margins = [np.inf if m is None else m for m in event["endpoint_competitor_margin_px"]]
    np.testing.assert_allclose(margins, expected_margins, atol=1e-6)


def expected_joined_labels(labels, view, events):
    """Rebuild exactly the recorded auxiliary union; no undocumented joins."""
    labels = np.asarray(labels)
    parent = {int(label): int(label) for label in np.unique(labels)}

    def find(label):
        while parent[label] != label:
            label = parent[label]
        return label

    joins = [event for event in events if event["accepted"] and event["view"] == view]
    assert len({event["left_label"] for event in joins}) == len(joins)
    assert len({event["right_label"] for event in joins}) == len(joins)
    for event in joins:
        left, right = int(event["left_label"]), int(event["right_label"])
        assert left in parent and right in parent and left != right
        a, b = find(left), find(right)
        assert a != b, "Duplicate or cyclic accepted join"
        parent[max(a, b)] = min(a, b)
    return np.asarray([find(int(label)) for label in labels], dtype=labels.dtype)


def observation_identity_counts(stage, core):
    """Partition fish frames; missing coordinates are not an observed ID result."""
    observed = np.isfinite(stage["xyz"][core]).all(-1)
    review = stage["identity_review"][core]
    return {
        "possible_fish_frames": int(observed.size),
        "missing_fish_frames": int((~observed).sum()),
        "observed_but_identity_review_fish_frames": int((observed & review).sum()),
        "observed_and_confident_fish_frames": int((observed & ~review).sum()),
        "confident_scope": "No estimator review flag; not verified biological identity",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("work/clean_tracklets_v1"))
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    args = parser.parse_args()
    summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete":
        raise ValueError("Run all frozen cases before final audit")
    config = summary["protocol"]["multiview_config"]
    originals = {}
    original_paths = {}
    for view in VIEWS:
        p = args.input_root / "work/c01_s16_predictions/bottomup" / f"{EXPERIMENT}__{view}.npz"
        original_paths[view] = p
        with np.load(p, allow_pickle=False) as z:
            originals[view] = {key: z[key] for key in ("frame_idx", "points")}
    rig = load_rig(args.input_root / CALIBRATION)
    results = []
    with provenance("clean_tracklet_independent_audit", args.output) as rec:
        for name in ("audit.py", "evaluate.py"):
            path = Path(__file__).parent / name
            rec.input(name, path, sha256=sha256(path))
        for view, path in original_paths.items():
            rec.input("original_C01_" + view, path, sha256=sha256(path))
        for path in (args.input_root / CALIBRATION).iterdir():
            if path.is_file():
                rec.input("calibration_" + path.name, path, sha256=sha256(path))
        rec.input(
            "summary", args.output / "summary.json", sha256=sha256(args.output / "summary.json")
        )
        for case in summary["cases"]:
            directory = args.output / case["name"]
            for name in ("reconstruction.npz", "joins.json", "summary.json"):
                path = directory / name
                rec.input(case["name"] + "_" + name, path, sha256=sha256(path))
            joins = json.loads((directory / "joins.json").read_text(encoding="utf-8"))
            assert joins["bridges"] == case["bridges"]
            with np.load(directory / "reconstruction.npz", allow_pickle=False) as z:
                stage = dict(z)
            frames = stage.pop("frames")
            core = (frames >= case["start"]) & (frames < case["stop"])
            assert geometry_metrics(stage, core) == case["comparison"]["new"]["geometry"]
            data, retained = {}, 0
            for view in VIEWS:
                original = originals[view]
                lo, hi = np.searchsorted(original["frame_idx"], [frames[0], frames[-1] + 1])
                for arm in ("P", "N", "F"):
                    p = directory / f"tracking_{arm}_{view}.npz"
                    with np.load(p, allow_pickle=False) as z:
                        item = dict(z)
                    np.testing.assert_array_equal(item["global_rows"], np.arange(lo, hi))
                    np.testing.assert_array_equal(item["frame_idx"], original["frame_idx"][lo:hi])
                    np.testing.assert_array_equal(item["points"], original["points"][lo:hi])
                    assert np.all(item["labels"] >= 0)
                    for label in np.unique(item["labels"]):
                        rows = np.flatnonzero(item["labels"] == label)
                        assert not item["clean_edge"][rows[0]]
                        assert item["clean_edge"][rows[1:]].all()
                        assert np.all(np.diff(item["frame_idx"][rows]) == 1)
                    if arm == "F":
                        data[view] = item
                    rec.input(case["name"] + "_" + p.stem, p, sha256=sha256(p))
                retained += hi - lo
                path = directory / f"joined_{view}.npz"
                rec.input(case["name"] + "_joined_" + view, path, sha256=sha256(path))
                with np.load(path, allow_pickle=False) as z:
                    np.testing.assert_array_equal(z["frame_idx"], data[view]["frame_idx"])
                    np.testing.assert_array_equal(z["clean_labels"], data[view]["labels"])
                    expected = expected_joined_labels(data[view]["labels"], view, joins["bridges"])
                    np.testing.assert_array_equal(z["joined_labels"], expected)
                    pairs = np.column_stack([z["frame_idx"], z["joined_labels"]])
                    assert len(np.unique(pairs, axis=0)) == len(pairs)
            validate_sources(stage, frames, data)
            accepted = [e for e in case["bridges"] if e["accepted"]]
            for event in accepted:
                check_join(event, data, stage, frames, rig, config)
            result = {
                "case": case["name"],
                "retained_rows_per_arm": int(retained),
                "accepted_joins_recomputed": len(accepted),
                "identity_observation_counts_core": observation_identity_counts(stage, core),
                "status": "pass",
            }
            results.append(result)
            print(result, flush=True)
        report = {
            "status": "pass",
            "cases": results,
            "scope": (
                "All original detector rows, clean components, source uniqueness, metrics, "
                "exact logged join unions, and independent two-view evidence for every "
                "accepted join. "
                "Observed-but-review and missing fish frames are reported separately. "
                "Confident means no estimator flag, not validated fish identity."
            ),
        }
        target = args.output / "audit.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        rec.output("audit", target, sha256=sha256(target))


if __name__ == "__main__":
    main()

"""Independently validate retained observations, metrics, and frozen artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from tracking import track

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/variable_count_tracklets_v1"
VIEWS = ("top_cam", "side1_cam", "side2_cam")
EXPECTED_COUNTS = {
    ("known_30735", "side2_cam", 30734): 1,
    ("known_30735", "side2_cam", 30735): 1,
    ("known_30735", "side2_cam", 30736): 1,
    ("known_30735", "side2_cam", 30746): 2,
    ("known_30735", "side2_cam", 30747): 2,
    ("cg20s", "top_cam", 76930): 2,
    ("cg20s", "side1_cam", 76930): 2,
    ("cg20s", "side2_cam", 76930): 3,
    ("known_31479", "side1_cam", 31479): 4,
    ("known_31479", "side1_cam", 31486): 5,
    ("known_31479", "side1_cam", 31488): 5,
    ("known_31479", "side1_cam", 31499): 5,
    ("known_260434", "side1_cam", 260434): 2,
    ("known_260434", "side2_cam", 260434): 4,
    ("known_323819", "side1_cam", 323819): 3,
    ("known_323819", "side1_cam", 323820): 2,
    ("known_323819", "side1_cam", 323821): 3,
    ("known_323819", "side1_cam", 323822): 3,
    ("known_323819", "side1_cam", 323823): 3,
    ("known_323819", "side1_cam", 323824): 3,
    ("known_323819", "side1_cam", 323825): 1,
}


def same(a, b):
    assert np.array_equal(a, b, equal_nan=True)


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    protocol = summary["protocol"]
    counts, verified, shuffled, checks = [], [], [], 0
    records = {(r["case"], r["view"], r["method"]): r for r in summary["records"]}
    with provenance("variable_count_independent_audit", OUT) as rec:
        rec.input("summary", OUT / "summary.json", sha256=sha256(OUT / "summary.json"))
        rec.input("audit_code", Path(__file__), sha256=sha256(Path(__file__)))
        for view in VIEWS:
            source = ROOT / "work/generative_tracking_inputs/work/c01_s16_predictions/bottomup"
            source /= f"20251225_K_1_2_FCMC__{view}.npz"
            with np.load(source, allow_pickle=False) as z:
                raw = {k: z[k] for k in ("frame_idx", "points", "score")}
            saved_path = ROOT / "work/custom2d_comparison_v1/inputs/c27_uncapped_tracklets_any"
            saved_path /= "20251225_K_1_2_FCMC/mouth-pectoral_oks_serial"
            saved = np.load(saved_path / f"track2d__{view}.npy", allow_pickle=False)
            rec.input(view + "_c01", source, sha256=sha256(source))
            for case in protocol["cases"]:
                dest = OUT / case["name"]
                with np.load(dest / f"detections_{view}.npz") as z:
                    original = {k: z[k] for k in z.files}
                rows, frames, points = (original[k] for k in ("global_row", "frame_idx", "points"))
                mask = (raw["frame_idx"] >= case["start"] - 332) & (
                    raw["frame_idx"] < case["stop"] + 332
                )
                same(rows, np.flatnonzero(mask))
                for key in ("frame_idx", "points", "score"):
                    same(original[key], raw[key][rows])
                same(original["observed_mask"], np.isfinite(points).all(axis=-1))
                assert len(np.unique(rows)) == len(rows)
                core = (frames >= case["start"]) & (frames < case["stop"])
                eligible = core & (frames > case["start"])
                for (name, camera, frame), expected in EXPECTED_COUNTS.items():
                    if name == case["name"] and camera == view:
                        actual = int((frames == frame).sum())
                        assert actual == expected
                        counts.append(dict(case=name, view=view, frame=frame, count=actual))
                for method in protocol["conditions"]:
                    path = dest / f"tracking_{method}_{view}.npz"
                    with np.load(path) as data:
                        for key in ("frame_idx", "points", "score", "observed_mask", "global_row"):
                            same(data[key], original[key])
                        labels = data["labels"]
                        if method == "C27":
                            same(labels, saved[rows])
                        previous = np.full(len(rows), -1, dtype=int)
                        last, frame_labels = {}, set()
                        for i, (frame, label) in enumerate(zip(frames, labels, strict=True)):
                            if label < 0:
                                continue
                            assert (int(frame), int(label)) not in frame_labels
                            frame_labels.add((int(frame), int(label)))
                            previous[i] = last.get(int(label), -1)
                            last[int(label)] = i
                        gap = np.zeros(len(rows), dtype=int)
                        valid = previous >= 0
                        gap[valid] = frames[valid] - frames[previous[valid]]
                        same(data["previous_row"], previous)
                        same(data["clean_edge"], gap == 1)
                        stats = records[case["name"], view, method]
                        calculated = {
                            "core_rows": int(core.sum()),
                            "retained_rows": int(core.sum()),
                            "eligible_observations": int(eligible.sum()),
                            "adjacent_connections": int(((gap == 1) & eligible).sum()),
                            "gap_connections": int(((gap > 1) & eligible).sum()),
                            "unassigned_rows": int(((labels < 0) & core).sum()),
                        }
                        calculated["contiguous_runs"] = (
                            calculated["core_rows"] - (calculated["adjacent_connections"])
                        )
                        calculated["unconnected_observations"] = (
                            calculated["eligible_observations"] - calculated["adjacent_connections"]
                        )
                        for key, value in calculated.items():
                            assert stats[key] == value, (case["name"], view, method, key)
                        assert (
                            sum(s["eligible_observations"] for s in stats["strata"])
                            == (calculated["eligible_observations"])
                        )
                        assert (
                            sum(s["adjacent_connections"] for s in stats["strata"])
                            == (calculated["adjacent_connections"])
                        )
                        count_by_frame = Counter(frames[core].tolist())
                        count_histogram = Counter(
                            min(4, count_by_frame.get(f, 0))
                            for f in range(case["start"], case["stop"])
                        )
                        for k in range(5):
                            assert (
                                stats["detection_count_frames"][str(k) if k < 4 else "4+"]
                                == (count_histogram[k])
                            )
                        if method.startswith("P"):
                            assert not np.any(gap > 1)
                        if method.startswith("P1"):
                            edges = json.loads(
                                (dest / f"edges_{method}_{view}.json").read_text(encoding="utf-8")
                            )
                            actual_edges = {
                                (e["from_global_row"], e["to_global_row"])
                                for e in edges
                                if e["accepted"]
                            }
                            expected_edges = {
                                (int(rows[previous[i]]), int(rows[i]))
                                for i in np.flatnonzero(gap == 1)
                            }
                            assert actual_edges == expected_edges
                        verified.append(
                            dict(case=case["name"], view=view, method=method, **calculated)
                        )
                        checks += 1
                # Full-context real-input shuffle, including initialization and all known anchors.
                if case["name"] == "cg20s":
                    permutation = np.random.default_rng(20260917).permutation(len(rows))
                    for auxiliary, method in ((False, "P1"), (True, "P1aux")):
                        result = track(
                            frames[permutation],
                            points[permutation],
                            rows[permutation],
                            view,
                            auxiliary=auxiliary,
                        )
                        aligned = np.empty_like(result.labels)
                        aligned[permutation] = result.labels
                        with np.load(dest / f"tracking_{method}_{view}.npz") as z:
                            same(aligned, z["labels"])
                        shuffled.append(
                            dict(
                                case=case["name"],
                                view=view,
                                method=method,
                                rows=len(rows),
                                exact_labels_and_edges=True,
                            )
                        )
                print(f"audit {case['name']} {view}: all conditions OK", flush=True)
        preserved = json.loads((OUT / "preservation_snapshot.json").read_text(encoding="utf-8"))
        for relative, digest in preserved.items():
            assert sha256(ROOT / relative) == digest, f"Changed existing file: {relative}"
        for relative, digest in protocol["code_sha256"].items():
            assert sha256(Path(__file__).parent / relative) == digest
        result = dict(
            status="passed",
            method_case_view_arrays=checks,
            preserved_existing_files=len(preserved),
            known_count_regressions=counts,
            real_input_permutations=shuffled,
            independently_recomputed_metrics=verified,
            no_new_human_truth_created=True,
            confirmed_identity_denominator=0,
        )
        path = OUT / "audit.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        rec.output("audit", path, sha256=sha256(path))
        print(
            f"PASS: {checks} arrays, {len(counts)} count regressions, "
            f"{len(shuffled)} real shuffles, {len(preserved)} preserved files",
            flush=True,
        )


if __name__ == "__main__":
    main()

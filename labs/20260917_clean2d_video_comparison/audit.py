"""Independently check row preservation, baseline identity and comparison totals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/clean2d_video_comparison_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    summary = read(OUT / "summary.json")
    protocol = summary["protocol"]
    receipts = []
    with provenance("direct_2d_integrity_audit", OUT) as rec:
        rec.input("summary", OUT / "summary.json", sha256=sha256(OUT / "summary.json"))
        rec.input(
            "runner",
            Path(__file__).with_name("run.py"),
            sha256=sha256(Path(__file__).with_name("run.py")),
        )
        for view in ("top_cam", "side1_cam", "side2_cam"):
            source = ROOT / "work/generative_tracking_inputs/work/c01_s16_predictions/bottomup"
            with np.load(source / f"{protocol['experiment']}__{view}.npz") as z:
                full_frames, full_points = z["frame_idx"], z["points"]
            baseline = np.load(ROOT / protocol["baseline_source"] / f"track2d__{view}.npy")
            for record in [r for r in summary["records"] if r["view"] == view]:
                case = record["case"]
                expected_rows = np.flatnonzero(
                    (full_frames >= record["frame_start"] - 332)
                    & (full_frames < record["frame_stop"] + 332)
                )
                for arm, stats in record["arms"].items():
                    path = OUT / case / f"tracking_{arm}_{view}.npz"
                    with np.load(path, allow_pickle=False) as z:
                        frames, points, labels = z["frame_idx"], z["points"], z["labels"]
                        assert np.array_equal(z["global_rows"], expected_rows)
                        assert np.array_equal(frames, full_frames[expected_rows])
                        assert np.array_equal(points, full_points[expected_rows], equal_nan=True)
                        if arm == "SLEAP":
                            assert np.array_equal(labels, baseline[expected_rows])
                        else:
                            assert np.all(labels >= 0)
                            assert np.all((z["link_gap"] == 0) | (z["link_gap"] == 1))
                        last, previous = {}, np.full(len(labels), -1, np.int64)
                        for row, (f, identity) in enumerate(zip(frames, labels, strict=True)):
                            if identity < 0:
                                continue
                            before = last.get(int(identity))
                            if before is not None:
                                assert frames[before] < f
                                previous[row] = before
                            last[int(identity)] = row
                        assert np.array_equal(previous, z["previous_row"])
                        gaps = np.zeros(len(labels), np.int64)
                        has_previous = previous >= 0
                        gaps[has_previous] = frames[has_previous] - frames[previous[has_previous]]
                        assert np.array_equal(gaps, z["link_gap"])
                        assert np.array_equal(gaps == 1, z["clean_edge"])
                        core = (frames >= record["frame_start"]) & (frames < record["frame_stop"])
                        incoming = core & (frames > record["frame_start"]) & (gaps == 1)
                        assert int(core.sum()) == stats["detection_rows"] == stats["retained_rows"]
                        assert int(incoming.sum()) == stats["adjacent_links"]
                        assert int(core.sum() - incoming.sum()) == stats["contiguous_runs"]
                        prior_path = ROOT / "work/clean_tracklets_v2" / case / path.name
                        reused = arm in ("P", "N", "F") and prior_path.exists()
                        if reused:
                            with np.load(prior_path, allow_pickle=False) as prior:
                                for key in (
                                    "frame_idx",
                                    "points",
                                    "labels",
                                    "global_rows",
                                    "clean_edge",
                                ):
                                    assert np.array_equal(z[key], prior[key], equal_nan=True)
                        receipts.append(
                            dict(
                                case=case,
                                view=view,
                                arm=arm,
                                rows=len(labels),
                                path=str(path.relative_to(ROOT)),
                                sha256=sha256(path),
                                existing_arrays_equal=reused,
                            )
                        )
        old_reports = {
            "work/clean_tracklets_v1/clean_tracklets_report.html": (
                "86775db2e14e1dc74c68f36f090050f107abf4e1a018126e9a89d10e8be28814"
            ),
            "work/custom2d_comparison_v1/comparison_report.html": (
                "7ffae76213478802eb09362ce4861f7ae7a93b43317a2575e93435b240913c44"
            ),
        }
        for relative, expected in old_reports.items():
            assert sha256(ROOT / relative) == expected
        for name in ("tracking.py", "flow.py"):
            assert (
                sha256(Path(__file__).with_name(name))
                == protocol[name.removesuffix(".py") + "_sha256"]
            )
        result = dict(
            status="passed",
            audited_arrays=len(receipts),
            identical_rows_to_C01=True,
            saved_C27_labels_unchanged=True,
            frozen_existing_PNF_arrays_unchanged=True,
            previous_reports_unchanged=True,
            receipts=receipts,
        )
        target = OUT / "audit.json"
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        rec.output("audit", target, sha256=sha256(target))
        print(f"PASS: {len(receipts)} arrays; identical detections and frozen previous results")


if __name__ == "__main__":
    main()

"""Audit measured arrays and frozen comparisons independently of the HTML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", type=Path, default=Path("work/custom2d_comparison_v1/analysis_v3")
    )
    parser.add_argument(
        "--base", type=Path, default=Path("work/custom2d_comparison_v1/analysis_v2")
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=Path("work/generative_tracking_inputs/work/c01_s16_predictions/bottomup"),
    )
    args = parser.parse_args()
    summary_path = args.analysis / "summary.json"
    summary = read(summary_path)
    assert summary["status"] == "complete"
    assert summary["counts"] == {"completed_cases": 20, "requested_cases": 20}
    rows = []
    with provenance("audit_custom2d_delivery", args.analysis) as rec:
        rec.input("code", Path(__file__), sha256=sha256(Path(__file__)))
        rec.input("summary", summary_path, sha256=sha256(summary_path))
        native_by_view = {}
        for view in VIEWS:
            path = args.detections / f"{summary['experiment']}__{view}.npz"
            rec.input(view, path, sha256=sha256(path))
            with np.load(path, allow_pickle=False) as data:
                native_by_view[view] = data["frame_idx"]
        for case in summary["cases"]:
            dest, old = args.analysis / case["name"], args.base / case["name"]
            for arm in "AB":
                for suffix in (".npz", "_summary.json"):
                    assert sha256(dest / (arm + suffix)) == sha256(old / (arm + suffix))
            partitions_changed = []
            for arm in "ABC":
                for view in VIEWS:
                    path = dest / f"tracking_{arm}_{view}.npz"
                    rec.input(case["name"] + "_" + path.name, path, sha256=sha256(path))
                    with np.load(path, allow_pickle=False) as data:
                        native, labels, global_rows = [
                            data[k] for k in ("frame_idx", "labels", "global_rows")
                        ]
                    assert len(labels) == len(global_rows)
                    assert np.all(labels >= 0)
                    assert len(np.unique(np.c_[native, labels], axis=0)) == len(labels)
                    np.testing.assert_array_equal(native, native_by_view[view][global_rows])
                    with np.load(old / path.name, allow_pickle=False) as previous:
                        np.testing.assert_array_equal(native, previous["frame_idx"])
                        np.testing.assert_array_equal(global_rows, previous["global_rows"])
                        if arm in "AB":
                            np.testing.assert_array_equal(labels, previous["labels"])
                        else:
                            pairs = np.unique(np.c_[previous["labels"], labels], axis=0)
                            # Each old tracklet remains whole; only merges are allowed.
                            assert len(pairs) == len(np.unique(previous["labels"]))
                            partitions_changed.append(
                                len(np.unique(previous["labels"])) != len(np.unique(labels))
                            )
                path = dest / f"{arm}.npz"
                rec.input(case["name"] + "_" + arm, path, sha256=sha256(path))
                with np.load(path, allow_pickle=False) as data:
                    frames = data["frames"]
                    np.testing.assert_array_equal(
                        frames, np.arange(case["context_start"], case["context_stop"])
                    )
                    core = (frames >= case["start"]) & (frames < case["stop"])
                    for stage in ("c28", "final"):
                        source, xyz = data[stage + "__source"], data[stage + "__xyz"]
                        for v, view in enumerate(VIEWS):
                            ids = source[:, :, v]
                            valid = ids >= 0
                            first = np.searchsorted(native_by_view[view], frames)
                            stop = np.searchsorted(native_by_view[view], frames + 1)
                            count = np.broadcast_to((stop - first)[:, None], ids.shape)
                            # C28/C34 source is an index within each native frame,
                            # not a full-file C01 row index.
                            assert np.all(ids[valid] < count[valid])
                            global_rows = ids + first[:, None]
                            expected = np.broadcast_to(frames[:, None], ids.shape)
                            np.testing.assert_array_equal(
                                native_by_view[view][global_rows[valid]], expected[valid]
                            )
                            for left in range(3):
                                for right in range(left + 1, 3):
                                    assert not np.any(
                                        (ids[:, left] >= 0) & (ids[:, left] == ids[:, right])
                                    )
                        held = xyz[core].astype(float)
                        steps = np.linalg.norm(np.diff(held, axis=0), axis=-1)
                        measured = case["arms"][arm][stage]
                        assert int((steps > 10).sum()) == measured["steps_over_10mm"]
                        assert int((steps > 4).sum()) == measured["steps_over_4mm"]
                        assert (
                            int((~np.isfinite(held).all(-1)).sum())
                            == measured["missing_fish_frames"]
                        )
            rows.append(
                dict(
                    case=case["name"],
                    passed=True,
                    A_B_bytes_preserved=True,
                    C_partitions_changed_by_view=dict(zip(VIEWS, partitions_changed, strict=True)),
                )
            )
        payload = dict(
            passed=True,
            cases=rows,
            checked=[
                "A/B reconstruction and summary bytes equal frozen comparison",
                "All input detection rows and native frames retained",
                "No per-frame duplicate 2D identity",
                "Deferred C never splits or rewires a confirmed base tracklet",
                "Every chosen 3D source row belongs to the correct view and native frame",
                "No detection reused in two 3D slots of a view",
                "3D jump and missingness metrics recomputed from saved arrays",
            ],
            identity_accuracy_verified=False,
        )
        output = args.analysis / "validation.json"
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        rec.output("validation", output, sha256=sha256(output))
        print(f"Validated {len(rows)} cases: {output}")


if __name__ == "__main__":
    main()

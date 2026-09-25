"""Describe changes in reviewed clips without treating old output as ID truth."""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance


def compare(first, candidate):
    options = list(permutations(range(3)))
    errors = [int(np.count_nonzero(first != candidate[:, p])) for p in options]
    best = options[int(np.argmin(errors))]
    aligned = candidate[:, best]
    changed = np.any(first != aligned, axis=(1, 2))
    physical_changed = []
    for before, after in zip(first, candidate, strict=True):
        physical_changed.append(
            sorted(map(tuple, before.tolist())) != sorted(map(tuple, after.tolist()))
        )
    return {
        "comparison_permutation_only": list(best),
        "source_entry_differences": min(errors),
        "source_frames_different": int(changed.sum()),
        "source_exact_frame_fraction": float((~changed).mean()),
        "instantaneous_correspondence_sets_different_frames": int(np.sum(physical_changed)),
        "reading": (
            "No source change within this clip after one constant renaming"
            if not changed.any()
            else "Changed assignments; visual identity adjudication still required"
        ),
        "not_an_identity_accuracy_measure": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", type=Path, default=Path("work/custom2d_comparison_v1/analysis_v2")
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("docs/reviews/2026-09-16_multiview_identity_review.json"),
    )
    args = parser.parse_args()
    output = args.analysis / "review_windows.json"
    if output.exists():
        raise FileExistsError("Preserve the existing reviewed-window comparison")
    review = json.loads(args.review.read_text(encoding="utf-8"))
    with provenance("compare_human_review_windows", args.analysis) as rec:
        rec.input("human_review", args.review, sha256=sha256(args.review))
        rec.input("code", Path(__file__), sha256=sha256(Path(__file__)))
        cached = {}
        rows = []
        for event in review["events"]:
            center = event["frame"]
            case = "cg20s" if 75270 <= center < 78590 else f"known_{center}"
            if case not in cached:
                cached[case] = {}
                for arm in ("A", "B", "C"):
                    p = args.analysis / case / f"{arm}.npz"
                    rec.input(f"{case}_{arm}", p, sha256=sha256(p))
                    with np.load(p, allow_pickle=False) as data:
                        cached[case][arm] = {
                            k: data[k] for k in ("frames", "final__source", "final__xyz")
                        }
            case_data = cached[case]
            frames = case_data["A"]["frames"]
            mask = (frames >= center - 83) & (frames < center + 83)
            assert mask.sum() == 166
            original = case_data["A"]["final__source"][mask]
            row = {
                "frame": center,
                "case": case,
                "human_kind": event["kind"],
                "human_note": event["verbatim_note"],
                "native_frame_range_half_open": [center - 83, center + 83],
                "arms": {},
            }
            for arm, data in case_data.items():
                assert np.array_equal(frames, data["frames"])
                result = compare(original, data["final__source"][mask])
                xyz = data["final__xyz"][mask].astype(float)
                steps = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
                finite = steps[np.isfinite(steps)]
                result.update(
                    steps_over_10mm=int((steps > 10).sum()),
                    maximum_step_mm=float(finite.max()) if finite.size else None,
                    missing_fish_frames=int((~np.isfinite(xyz).all(-1)).sum()),
                )
                row["arms"][arm] = result
            rows.append(row)
        payload = {
            "reviewed_panel": review["reviewed_panel"],
            "rows": rows,
            "interpretation": "Compare methods within each fixed clip; A is not ID ground truth",
            "overlap": (
                "f76930 and f76988 clips overlap by 108 frames; "
                "do not sum as independent events"
            ),
        }
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        rec.output("review_window_comparison", output, sha256=sha256(output))
        print(output)


if __name__ == "__main__":
    main()

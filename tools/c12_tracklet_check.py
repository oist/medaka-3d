"""C12 — would 2D tracklets alone have carried the female through the swaps?

`c09` measured 8 intervals where the female's identity was handed to another
fish. `c11` showed one of them in detail: the two animals never occluded in any
camera, and the 3D correspondence flipped on one to three frames out of 1,055.

If a tracklet built **inside one camera**, with no 3D, spans such an interval
unbroken and holds the same animal at both human anchors, then the information
needed to avoid that swap was present and the old architecture discarded it.
That is the claim `docs/decisions/2026-08-18-2d-first-tracking.md` rests on, and
this tool tests it rather than assuming it.

=============================================================================
THE TEST — fixed before the first number was read
=============================================================================

For each of `c09`'s 296 intervals, in each view independently:

**spanned**      the female's detection at the opening anchor and at the closing
                 anchor belong to the **same tracklet**. The camera carried her
                 across on its own.
**broken**       both anchors matched a detection, but the tracklets differ. The
                 camera stopped — honestly, since a tracklet only ends where the
                 linker refused.
**unmatched**    an anchor had no detection within ``--match-px``, or the frame
                 did not hold exactly three, so this view cannot be asked.

A view that spans a swap interval is evidence **for** the 2D-first order. A view
that breaks it is not evidence against — a break is a refusal to guess, and the
association stage exists to resolve refusals with the other cameras. What would
be evidence against is a view that spans the interval while **changing which
animal** it holds, and that is exactly what "same tracklet at both anchors, and
the anchors are the female" tests: the human says both ends are her.

**The headline is the swap subset**, 8 intervals, and eight is a small number.
The 269 preserved intervals are reported beside them as the control: tracklets
should span those at least as often, or the measure is picking up interval
length rather than difficulty.

**n_groups = 1, n_experiments = 1, female only.**

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c12_tracklet_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from s10_shared_node_20k import median_span_px, tail_jump_threshold_px

from medaka3d import load_catalog, load_paths
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.reconstruct import N_ANIMALS
from medaka3d.track2d import UNTRACKED, build
from medaka3d.types import FPS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN = "bottomup"
GT = Path(r"Z:/StephensU/medaka/archive/medaka-mating-pipeline/annotations"
          r"/fish_sex_gt/20251225_K_1_2_FCMC/female_sex_gt.csv")
SIDE = (("side1_cam", "side1"), ("side2_cam", "side2"))


def anchor_rows(detections, gt: pd.DataFrame, prefix: str, match_px: float) -> dict[int, int]:
    """``frame -> detection row`` for the female, matched on the annotated mouth."""
    start, stop = detections.start, detections.stop
    mouth = detections.node("mouth")
    found = {}
    for _, record in gt[gt[f"{prefix}_selection_status"].eq("sleap_linked")].iterrows():
        frame = int(record["frame"])
        lo, hi = int(start[frame]), int(stop[frame])
        target = np.asarray([record[f"{prefix}_mouth_x"], record[f"{prefix}_mouth_y"]], float)
        if lo == hi or not np.isfinite(target).all():
            continue
        distance = np.linalg.norm(mouth[lo:hi] - target, axis=1)
        distance = np.where(np.isfinite(distance), distance, np.inf)
        index = int(np.argmin(distance))
        if distance[index] <= match_px:
            found[frame] = lo + index
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--match-px", type=float, default=5.0)
    parser.add_argument("--margin-scale", type=float, default=0.5,
                        help="Ambiguity margin as a fraction of the view's gate.")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    out_dir = paths.work_dir / "c12_tracklet_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("tracklet_check", out_dir, name="c12") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        record.input("female_sex_gt", GT)

        validation = json.loads(
            (paths.work_dir / "c09_gt_swap_validation" / "c09_validation.json").read_text(
                encoding="utf-8"))
        intervals = validation["intervals"]
        gt = pd.read_csv(GT, low_memory=False)
        gt = gt[gt.gt_status.eq("confirmed")].sort_values("frame")

        print(f"{'view':<10} {'gate':>5} {'margin':>7} {'tracklets':>10} "
              f"{'>=1s':>7} {'frames in >=1s':>15}   cuts")
        tracklets, rows_by_view, stats = {}, {}, {}
        for view, prefix in SIDE:
            key = f"labels.round8.{'top' if view == 'top_cam' else 'side'}.train"
            span, _ = median_span_px(catalog.get(key))
            gate = tail_jump_threshold_px(span)
            margin = args.margin_scale * gate
            detections = c01.load(paths.work_dir, EXPERIMENT, view, RUN)
            result = build(detections.frame_idx, detections.node("pectoral"),
                           detections.n_frames, experiment=EXPERIMENT, view=view,
                           gate_px=gate, margin_px=margin)
            lengths = result.lengths()
            long = lengths[lengths >= FPS]
            tracklets[view] = result
            rows_by_view[view] = anchor_rows(detections, gt, prefix, args.match_px)
            stats[view] = {
                "gate_px": gate, "margin_px": margin, "tracklets": result.count,
                "over_one_second": int(long.size),
                "generations": result.count // N_ANIMALS,
                "frames_in_over_one_second":
                    float(long.sum() / N_ANIMALS / detections.n_frames),
                "cuts": result.cuts,
                "anchors_matched": len(rows_by_view[view]),
            }
            print(f"{view:<10} {gate:>5.0f} {margin:>7.0f} {result.count // N_ANIMALS:>10,} "
                  f"{long.size:>7,} {long.sum() / N_ANIMALS / detections.n_frames:>14.1%}   "
                  + "  ".join(f"{k} {v:,}" for k, v in result.cuts.items()))
        record.param("per_view", stats)

        print(f"\n{'outcome':<11} {'n':>4}   " + "   ".join(
            f"{v.replace('_cam', ''):<24}" for v, _ in SIDE))
        print(f"{'':<11} {'':>4}   " + "   ".join(
            f"{'spanned':>8} {'broken':>7} {'n/a':>5}   " for _ in SIDE))
        table, detail = {}, []
        for outcome in ("swap", "break", "preserved"):
            cell = [r for r in intervals if r["outcome"] == outcome]
            line, per_view = f"{outcome:<11} {len(cell):>4}   ", {}
            for view, _ in SIDE:
                rows, result = rows_by_view[view], tracklets[view]
                verdicts = Counter()
                for item in cell:
                    a, b = rows.get(item["t1"]), rows.get(item["t2"])
                    if a is None or b is None:
                        verdicts["unmatched"] += 1
                        continue
                    first, second = result.tracklet[a], result.tracklet[b]
                    if first == UNTRACKED or second == UNTRACKED:
                        verdicts["unmatched"] += 1
                    else:
                        verdicts["spanned" if first == second else "broken"] += 1
                per_view[view] = dict(verdicts)
                line += (f"{verdicts['spanned']:>8} {verdicts['broken']:>7} "
                         f"{verdicts['unmatched']:>5}   ")
            table[outcome] = per_view
            print(line)

        for item in [r for r in intervals if r["outcome"] == "swap"]:
            entry = {"t1": item["t1"], "t2": item["t2"], "gap": item["gap"],
                     "encounters": item["encounters"]}
            for view, _ in SIDE:
                a, b = rows_by_view[view].get(item["t1"]), rows_by_view[view].get(item["t2"])
                result = tracklets[view]
                if a is None or b is None:
                    entry[view] = "unmatched"
                else:
                    first, second = int(result.tracklet[a]), int(result.tracklet[b])
                    entry[view] = ("unmatched" if UNTRACKED in (first, second)
                                   else "spanned" if first == second
                                   else f"broken {first}->{second}")
            detail.append(entry)
        print("\nthe 8 swaps, one line each")
        for entry in detail:
            print(f"  {entry['t1']:>7}-{entry['t2']:<7} gap {entry['gap']:>5} "
                  f"enc {entry['encounters']:>2}   "
                  + "   ".join(f"{v.replace('_cam', '')} {entry[v]}" for v, _ in SIDE))

        record.param("outcomes", table)
        (out_dir / "c12_tracklet_check.json").write_text(
            json.dumps({"experiment": EXPERIMENT, "n_groups": 1,
                        "groups": [ExperimentId.parse(EXPERIMENT).group],
                        "per_view": stats, "outcomes": table, "swaps": detail},
                       indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

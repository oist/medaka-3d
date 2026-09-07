"""C09 — which of the automatic metrics actually indicates an ID swap?

Every number in `c04` and `c08` is a property of the prediction. They can prove
a tracker wrong; none of them can prove one right, and none of them has ever
been checked against a human. This tool does that check, on the only human
identity annotation that exists in this project.

=============================================================================
THE ANNOTATION, AND WHY IT TRANSFERS
=============================================================================

`annotations/fish_sex_gt/20251225_K_1_2_FCMC/female_sex_gt.csv` in the retired
repo's archive holds **300 confirmed identifications of the female**, spread
across frames 4,439-348,388 of one experiment, with her mouth and tail clicked
in side1 and side2.

It was made on round-7 top-down output under a refractive calibration.
Production is now round-8 bottom-up 3-node under a pinhole DLT rig, so instance
indices, geometry IDs and 3D coordinates do not transfer. **Pixel coordinates
and frame numbers do**, because they are properties of the video rather than of
the model that was run on it. Measured, not assumed: the nearest current mouth
to an annotated mouth is a median **1.27 px** away in side1 and **1.60 px** in
side2, 96.9% / 95.5% within 5 px, and the runner-up is a median 163 / 176 px
further — so the match is close *and* unambiguous. Frame indexing agrees
exactly; offsets of -1 and +1 both make it worse.

=============================================================================
DEFINITIONS — fixed before the first number was read
=============================================================================

**Anchor.** One annotated frame where the female's mouth matches a current
detection within ``--match-px`` in a side view, and that detection is the one
the reconstruction used for a slot. The slot gives her 3D track ID. Anchors
where the two side views name different slots are dropped, not adjudicated.

**Interval.** A consecutive pair of anchors. The median gap is ~1,024 frames
(6.2 s), which is long; this is a property of the annotation and is reported
rather than worked around. ``--max-gap`` restricts to shorter intervals.

**Outcome of an interval**, from the tracker's own output plus the human anchors:

* ``preserved`` — the female carries the same track ID at both ends.
* ``break`` — her ID changed, and the old ID is **not alive** at the far end.
  The tracker dropped an identity and started a new one. Costly, but visible to
  every automatic metric and not a scientific error.
* ``swap`` — her ID changed, and the old ID **is alive at the far end, on
  another animal**. The tracker gave her name to a different fish. This is the
  failure that corrupts behaviour scoring, and the one no self-consistency check
  can see. **This is the positive class.**

The break/swap split is the whole point. `c04`'s ``preserved`` and `c08`'s
``held`` both collapse them, and a metric that predicts breaks while missing
swaps would look useful and be worthless.

**Predictors**, all computed on the interval with no human input:

===========================  ===================================================
``encounters``               `c08` encounter windows overlapping the interval
``max_stratum``              most cameras any of them crossed in (0-3)
``stratum3_encounters``      how many crossed in all three
``id_changes``               `c08` ID changes summed over those encounters
``track_jumps``              per-view links exceeding the jump gate
``min_margin_mm``            smallest runner-up margin of the correspondence
``unresolved_fraction``      frames with no reconstruction
``min_separation_mm``        closest approach of any pair
===========================  ===================================================

**How a predictor is judged.** For each one, the swap rate in its top decile
against its bottom decile, and the fraction of all swaps captured by the top
decile. A predictor that indicates swaps concentrates them; one that indicates
breaks, or nothing, does not. **Rates are reported with their counts**, because
the positive class is small: 300 anchors over one experiment yield single-digit
to low-double-digit swaps, and no ranking of predictors will be stable at that
size. That limit is the finding's headline, not a footnote.

=============================================================================
FOUND AFTER THE FACT — NOT PRE-REGISTERED
=============================================================================

Two quantities below were **not** in the predictor list above. They were found
by inspecting where the swaps fell after the pre-registered comparison had been
run and printed. They are reported because they are much stronger than anything
that was registered, and labelled because that is exactly the circumstance in
which a number is most likely to be an artefact of the sample it was found in.
Neither may be quoted as validated until it is tested on data it was not found
in.

* ``chain_column_changed`` — the geometric reference chain holds the female in
  a different canonical column at the two anchors.
* ``encounters`` in bands rather than deciles.

A caution specific to the first. The chain and the `gated` tracker are both
minimum-displacement assignments over the same slots, so a chain flip and a
tracker flip are not independent events; this indicator partly measures "the
geometry was ambiguous here" rather than checking the tracker from outside. It
still needs no human, which is what it is for.

**n_groups = 1, n_experiments = 1.** K_1_2 alone. The female alone — there is
still no M1/M2 annotation, so nothing here measures a male's swap rate.
**Female-vs-male crossings, however, are spanned by the anchors** and are
reported separately; see ``attributable``.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c09_gt_swap_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from c02_tracking_probe import CALIBRATION
from c04_score_tracking import reference_chain
from s10_shared_node_20k import median_span_px, tail_jump_threshold_px

from medaka3d import load_catalog, load_paths
from medaka3d.calib import load_rig
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.track import UNTRACKED
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN, METHOD = "bottomup", "gated"
SIDE_VIEWS = (("side1_cam", "side1"), ("side2_cam", "side2"))
GT_RELATIVE = Path("annotations/fish_sex_gt/20251225_K_1_2_FCMC/female_sex_gt.csv")
PREDICTORS = (
    "encounters", "max_stratum", "stratum3_encounters", "id_changes",
    "track_jumps", "min_margin_mm", "unresolved_fraction", "min_separation_mm",
)
#: Predictors where a *small* value is the suspicious one.
LOWER_IS_WORSE = frozenset({"min_margin_mm", "min_separation_mm"})


def female_anchors(paths, gt: pd.DataFrame, npz, rig_views, match_px: float) -> list[dict]:
    """Frame, 3D track ID and match quality for each usable female annotation."""
    source = npz["source"]
    slot_track = npz["slot_track"]

    per_view: dict[str, dict[int, tuple[int, float]]] = {}
    for view, prefix in SIDE_VIEWS:
        detections = c01.load(paths.work_dir, EXPERIMENT, view, RUN)
        start, stop = detections.start, detections.stop
        mouth = detections.node("mouth")
        column = rig_views.index(view)
        found: dict[int, tuple[int, float]] = {}
        rows = gt[gt[f"{prefix}_selection_status"].eq("sleap_linked")]
        for _, row in rows.iterrows():
            frame = int(row["frame"])
            lo, hi = int(start[frame]), int(stop[frame])
            target = np.asarray([row[f"{prefix}_mouth_x"], row[f"{prefix}_mouth_y"]], float)
            if lo == hi or not np.isfinite(target).all():
                continue
            distance = np.linalg.norm(mouth[lo:hi] - target, axis=1)
            distance = np.where(np.isfinite(distance), distance, np.inf)
            index = int(np.argmin(distance))
            if distance[index] > match_px:
                continue
            # The reconstruction records, per slot, which detection row of this
            # view it consumed. Invert that to go from the matched detection
            # back to the slot the human's fish occupies.
            slots = np.flatnonzero(source[frame, :, column] == index)
            if slots.size != 1:
                continue
            found[frame] = (int(slots[0]), float(distance[index]))
        per_view[view] = found

    anchors = []
    disagreed = 0
    for frame in sorted(set(per_view["side1_cam"]) | set(per_view["side2_cam"])):
        seen = {v: per_view[v][frame] for v, _ in SIDE_VIEWS if frame in per_view[v]}
        slots = {s for s, _ in seen.values()}
        if len(slots) != 1:
            disagreed += 1
            continue
        slot = slots.pop()
        track = int(slot_track[frame, slot])
        if track == UNTRACKED:
            continue
        anchors.append({
            "frame": frame, "slot": slot, "track": track,
            "views": sorted(seen), "match_px": max(d for _, d in seen.values()),
        })
    return anchors, disagreed


def outcome(npz, first: dict, second: dict) -> str:
    """``preserved`` / ``break`` / ``swap`` for one interval."""
    if first["track"] == second["track"]:
        return "preserved"
    alive = set(npz["slot_track"][second["frame"]].tolist())
    return "swap" if first["track"] in alive else "break"


def predictors_for(interval, encounters, npz, chain, jumps_at) -> dict:
    """Every automatic quantity, measured on ``[t1, t2]`` with no human input."""
    lo, hi = interval
    inside = [e for e in encounters if e["exit_frame"] >= lo and e["entry_frame"] <= hi]
    rows = np.arange(lo, hi + 1)[:, None]
    ok = chain[lo : hi + 1] >= 0
    canonical = np.where(ok[:, :, None], npz["xyz"][rows, np.maximum(chain[lo:hi + 1], 0)], np.nan)
    separations = [
        np.linalg.norm(canonical[:, a, :] - canonical[:, b, :], axis=1)
        for a, b in ((0, 1), (0, 2), (1, 2))
    ]
    finite = np.concatenate([d[np.isfinite(d)] for d in separations])
    margin = npz["margin_mm"][lo : hi + 1]
    resolved = npz["slot_track"][lo : hi + 1].max(axis=1) >= 0
    return {
        "frames": hi - lo,
        "encounters": len(inside),
        "max_stratum": max((e["stratum"] for e in inside), default=0),
        "stratum3_encounters": sum(e["stratum"] == 3 for e in inside),
        "id_changes": sum(e["id_changes"] for e in inside),
        "track_jumps": int(jumps_at[lo:hi].sum()),
        "min_margin_mm": float(np.nanmin(margin)) if np.isfinite(margin).any() else np.nan,
        "unresolved_fraction": float(1.0 - resolved.mean()),
        "min_separation_mm": float(finite.min()) if finite.size else np.nan,
    }


def attributable(rows: list[dict], encounters: list[dict]) -> tuple[list[dict], dict]:
    """Intervals whose outcome can be pinned on **one** encounter, labelled F-M or M-M.

    Male-vs-male identity is unmeasurable here — there is no M1/M2 annotation.
    **Female-vs-male is not.** Three animals means exactly one female, so any
    encounter whose participants include the female's canonical column is an
    F-M crossing, and the human anchors on either side say whether she came out
    of it carrying the same track ID. That is a real per-crossing swap rate for
    one of the two crossing types, from annotation that already exists.

    An interval qualifies only if the female sits in the **same canonical column**
    at both ends and exactly **one** encounter falls inside it. Both conditions
    are needed for attribution: a column change makes "which pair crossed"
    ambiguous, and two encounters make it impossible to say which one cost her
    the identity. Everything excluded is counted.

    The M-M intervals are the control. A crossing the female was not part of
    should cost her identity far less often; if it does not, the attribution is
    measuring something else and the F-M number should not be believed.
    """
    kept, excluded = [], Counter()
    for row in rows:
        if row["column_1"] != row["column_2"]:
            excluded["female changed canonical column"] += 1
            continue
        inside = [
            e for e in encounters
            if e["exit_frame"] >= row["t1"] and e["entry_frame"] <= row["t2"]
        ]
        if len(inside) != 1:
            excluded["zero encounters" if not inside else "several encounters"] += 1
            continue
        encounter = inside[0]
        kept.append({
            **row,
            "pair": "F-M" if row["column_1"] in encounter["participants"] else "M-M",
            "encounter_entry": encounter["entry_frame"],
            "encounter_stratum": encounter["stratum"],
            "encounter_outcome": encounter["outcome"],
            "encounter_min_mm": encounter["min_distance_mm"],
        })
    return kept, dict(excluded)


def decile_table(rows: list[dict], name: str) -> dict:
    """Swap rate in the suspicious decile against the benign one."""
    usable = [r for r in rows if np.isfinite(r[name])]
    if len(usable) < 20:
        return {"n": len(usable), "note": "too few intervals to decile"}
    values = np.asarray([r[name] for r in usable], float)
    swap = np.asarray([r["outcome"] == "swap" for r in usable])
    worse = -values if name in LOWER_IS_WORSE else values
    order = np.argsort(-worse)  # most suspicious first
    k = max(1, len(usable) // 10)
    top, bottom = order[:k], order[-k:]
    return {
        "n": len(usable),
        "swaps": int(swap.sum()),
        "top_decile_n": int(k),
        "top_decile_swaps": int(swap[top].sum()),
        "top_decile_rate": float(swap[top].mean()),
        "bottom_decile_swaps": int(swap[bottom].sum()),
        "bottom_decile_rate": float(swap[bottom].mean()),
        "recall_of_top_decile": float(swap[top].sum() / swap.sum()) if swap.any() else np.nan,
        "median_when_swap": float(np.median(values[swap])) if swap.any() else np.nan,
        "median_when_not": float(np.median(values[~swap])) if (~swap).any() else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--match-px", type=float, default=5.0)
    parser.add_argument("--npz", type=Path, default=None,
                        help="Score a different reconstruction, e.g. c13's. The encounter "
                             "predictors still come from c08 and are stale against it; the "
                             "swap count, which is what the decision rule asks for, is not.")
    parser.add_argument("--max-gap", type=int, default=0,
                        help="Drop intervals longer than this many frames. 0 keeps all.")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    rig = load_rig(CALIBRATION)
    out_dir = paths.work_dir / "c09_gt_swap_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = paths.archive_root / GT_RELATIVE if hasattr(paths, "archive_root") else None
    if gt_path is None or not gt_path.exists():
        gt_path = Path(r"Z:/StephensU/medaka/archive/medaka-mating-pipeline") / GT_RELATIVE
    if not gt_path.exists():
        raise SystemExit(f"female ground truth not found: {gt_path}")

    with provenance("gt_swap_validation", out_dir, name="c09") as record:
        record.params_from(args)
        record.input("female_sex_gt", gt_path)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        print(f"ground truth : {gt_path}")
        print("n_groups = 1, n_experiments = 1 — K_1_2, female only\n")

        gate = {}
        for view in VIEWS:
            key = f"labels.round8.{'top' if view == 'top_cam' else 'side'}.train"
            span, _ = median_span_px(catalog.get(key))
            gate[view] = tail_jump_threshold_px(span)

        gt = pd.read_csv(gt_path, low_memory=False)
        gt = gt[gt.gt_status.eq("confirmed")].sort_values("frame")
        tracks_path = args.npz or (
            paths.work_dir / "c03_track_3d" / RUN / f"{METHOD}__{EXPERIMENT}.npz")
        record.input("tracks", tracks_path)
        print(f"tracks       : {tracks_path}")
        npz = np.load(tracks_path, allow_pickle=False)
        chain = reference_chain(npz["xyz"].astype(float), npz["slot_track"].max(axis=1) >= 0)

        anchors, disagreed = female_anchors(paths, gt, npz, rig.views, args.match_px)
        # Which canonical column holds her. `c08`'s encounter participants are
        # canonical columns, so this is what makes a crossing nameable as F-M.
        for anchor in anchors:
            column = np.flatnonzero(chain[anchor["frame"]] == anchor["slot"])
            anchor["column"] = int(column[0]) if column.size == 1 else -1
        print(f"confirmed annotations : {len(gt)}")
        print(f"usable anchors        : {len(anchors)}  "
              f"(dropped {disagreed} where the two side views named different slots)")
        record.param("anchors", len(anchors))
        record.param("anchors_dropped_view_disagreement", disagreed)

        # Per-frame track-jump indicator, pooled over views, for the predictors.
        jumps_at = np.zeros(c01.N_FRAMES, dtype=np.int32)
        for view in VIEWS:
            detections = c01.load(paths.work_dir, EXPERIMENT, view, RUN)
            track = npz[f"track__{view}"]
            keep = track != UNTRACKED
            frame = detections.frame_idx[keep]
            mouth = detections.node("mouth")[keep]
            order = np.lexsort((frame, track[keep]))
            frame, ids, mouth = frame[order], track[keep][order], mouth[order]
            same = ids[1:] == ids[:-1]
            step = frame[1:] - frame[:-1]
            moved = np.linalg.norm(mouth[1:] - mouth[:-1], axis=1)
            hit = same & (step == 1) & (moved > gate[view])
            np.add.at(jumps_at, frame[1:][hit], 1)

        strata = json.loads(
            (paths.work_dir / "c08_crossing_strata" / "c08_strata.json").read_text(
                encoding="utf-8")
        )
        encounters = [e for e in strata["encounters"] if e["experiment"] == EXPERIMENT]
        record.param("c08_encounters_this_experiment", len(encounters))

        rows = []
        for first, second in pairwise(anchors):
            gap = second["frame"] - first["frame"]
            if args.max_gap and gap > args.max_gap:
                continue
            values = predictors_for(
                (first["frame"], second["frame"]), encounters, npz, chain, jumps_at
            )
            rows.append({
                "t1": first["frame"], "t2": second["frame"], "gap": gap,
                "track_1": first["track"], "track_2": second["track"],
                "column_1": first["column"], "column_2": second["column"],
                "outcome": outcome(npz, first, second), **values,
            })

        counts = Counter(r["outcome"] for r in rows)
        print(f"\nintervals             : {len(rows)}   median gap "
              f"{np.median([r['gap'] for r in rows]):.0f} frames")
        for key in ("preserved", "break", "swap"):
            print(f"  {key:<10} {counts[key]:>4}  {counts[key] / max(1, len(rows)):6.1%}")
        record.param("outcomes", dict(counts))

        # --- female-male crossings, which the anchors DO span --------------
        attributed, dropped = attributable(rows, encounters)
        print(f"\nattributable intervals: {len(attributed)} of {len(rows)}   "
              + "  ".join(f"[-{v} {k}]" for k, v in sorted(dropped.items())))
        print(f"{'pair':<6} {'n':>5} {'preserved':>10} {'break':>7} {'swap':>7}"
              f" {'swap rate':>10}")
        pair_table = {}
        for pair in ("F-M", "M-M"):
            cell = [r for r in attributed if r["pair"] == pair]
            if not cell:
                continue
            tally = Counter(r["outcome"] for r in cell)
            pair_table[pair] = {"n": len(cell),
                                **{k: tally[k] for k in ("preserved", "break", "swap")}}
            print(f"{pair:<6} {len(cell):>5} {tally['preserved']:>10} {tally['break']:>7}"
                  f" {tally['swap']:>7} {tally['swap'] / len(cell):>10.1%}")
        record.param("pair_attribution", pair_table)
        record.param("pair_attribution_dropped", dropped)

        by_stratum = {}
        print("\nF-M crossings by stratum")
        print(f"{'views':>5} {'n':>5} {'preserved':>10} {'break':>7} {'swap':>7}")
        for stratum in (0, 1, 2, 3):
            cell = [r for r in attributed
                    if r["pair"] == "F-M" and r["encounter_stratum"] == stratum]
            if not cell:
                continue
            tally = Counter(r["outcome"] for r in cell)
            by_stratum[str(stratum)] = {"n": len(cell),
                                        **{k: tally[k] for k in
                                           ("preserved", "break", "swap")}}
            print(f"{stratum:>5} {len(cell):>5} {tally['preserved']:>10} "
                  f"{tally['break']:>7} {tally['swap']:>7}")
        record.param("fm_by_stratum", by_stratum)

        # --- post-hoc; see the docstring. Printed apart from the registered
        # comparison so the two are never read as one result.
        print("\nPOST-HOC, not pre-registered — hypotheses, not results")
        posthoc = {}
        for moved in (False, True):
            cell = [r for r in rows if (r["column_1"] != r["column_2"]) is moved]
            tally = Counter(r["outcome"] for r in cell)
            posthoc[f"chain_column_changed_{moved}"] = {
                "n": len(cell), **{k: tally[k] for k in ("preserved", "break", "swap")}
            }
            print(f"  chain moved her = {moved!s:<5} n={len(cell):>4}  "
                  f"swap {tally['swap']:>2} ({tally['swap'] / max(1, len(cell)):5.1%})  "
                  f"break {tally['break']:>2}")
        bands = {}
        for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 6), (7, 999)):
            cell = [r for r in rows if lo <= r["encounters"] <= hi]
            if not cell:
                continue
            tally = Counter(r["outcome"] for r in cell)
            bands[f"{lo}-{hi}"] = {"n": len(cell),
                                   **{k: tally[k] for k in ("preserved", "break", "swap")}}
            print(f"  encounters {lo}-{hi:<3} n={len(cell):>4}  "
                  f"swap {tally['swap']:>2} ({tally['swap'] / len(cell):5.1%})  "
                  f"break {tally['break']:>2}")
        record.param("posthoc_chain_column", posthoc)
        record.param("posthoc_encounter_bands", bands)

        print(f"\n{'predictor':<22} {'swap med':>9} {'other med':>10} "
              f"{'top10% swaps':>13} {'rate':>7} {'bottom rate':>12} {'recall':>7}")
        table = {}
        for name in PREDICTORS:
            stats = decile_table(rows, name)
            table[name] = stats
            if "note" in stats:
                print(f"{name:<22} {stats['note']}")
                continue
            print(f"{name:<22} {stats['median_when_swap']:>9.2f} "
                  f"{stats['median_when_not']:>10.2f} "
                  f"{stats['top_decile_swaps']:>7}/{stats['top_decile_n']:<5} "
                  f"{stats['top_decile_rate']:>7.1%} {stats['bottom_decile_rate']:>12.1%} "
                  f"{stats['recall_of_top_decile']:>7.1%}")
        record.param("predictors", table)

        (out_dir / "c09_validation.json").write_text(
            json.dumps({
                "experiment": EXPERIMENT, "n_groups": 1,
                "groups": [ExperimentId.parse(EXPERIMENT).group],
                "match_px": args.match_px, "max_gap": args.max_gap,
                "anchors": anchors, "outcomes": dict(counts),
                "pair_attribution": pair_table, "pair_attribution_dropped": dropped,
                "fm_by_stratum": by_stratum, "attributed": attributed,
                "posthoc_chain_column": posthoc,
                "posthoc_encounter_bands": bands,
                "predictors": table, "intervals": rows,
            }, indent=2, default=float),
            encoding="utf-8",
        )
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

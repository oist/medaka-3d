"""C10 — swaps as the experimenter defines them, and the scenes to check them by eye.

`c09` used consecutive annotations as interval ends, which put them a median 5.8 s
apart with several crossings inside. That measures *whether* the female was lost,
never *where*. This tool adopts the definition that makes the question local:

    A swap is: a **clean** stretch of track, a crossing, another **clean**
    stretch, and a different ID on the far side.

Clean means the animals were apart — inter-fish distance above the separation
threshold, held long enough that no one could confuse them. Transient exchanges
*during* the approach are deliberately out of scope: they are not decidable from
a still, and the experimenter does not treat them as errors.

=============================================================================
DEFINITIONS — fixed before the first number was read
=============================================================================

**Clean segment.** A maximal run of at least ``--min-clean`` frames in which the
female's distance to *both* other animals is at least ``--separation-mm``. This
is the same "sustained separation" that bounds an encounter in
`medaka3d.eval.encounters`, applied to one animal rather than to a pair, because
the female is the one the annotation names.

**Transition.** Two consecutive clean segments and the gap between them. The gap
holds one crossing or several; ``--max-encounters`` restricts to gaps holding at
most that many, since a gap with four crossings cannot say which one cost her
the identity.

**Anchored transition.** One with a human female annotation in the clean segment
on each side, and a track ID that is constant across each of those segments. The
constancy requirement matters: an anchor in a segment whose ID churns does not
establish "her ID on this side of the crossing".

**Outcome.** ``preserved`` / ``break`` / ``swap`` exactly as `c09` defines them —
swap being the case where her old ID is still alive at the far anchor, carried by
another animal.

**Two annotation sources, and only one of them is independent.**

* ``female_sex_gt.csv`` — 300 confirmed frames, sampled from *clean* three-view
  candidates, which is why they land in clean segments at all.
* ``jump6_gate_branch_female_review.csv`` — 25 frames the previous project chose
  **at crossings**, each carrying the human's decision between two ID mappings.
  `c09`'s strongest indicator was found on the first source; **this second one is
  the data it was not found in**, and is scored separately for that reason.

**Transient steps, the experimenter's heuristic, made computable.** A swap tends
to show as a step in the inter-fish-distance or speed trace rather than a smooth
excursion, because two identities exchanging teleports both traces at one frame.
Two quantities per gap, both with no human input:

``ifd_step_mm``    largest one-frame change in the female's distance to the
                   nearest other animal, over the gap.
``speed_step_mm``  largest one-frame change in her own speed.

They are reported beside the pre-registered predictors and beside `c09`'s
post-hoc chain-column indicator, which this tool re-tests rather than assumes.

**n_groups = 1, n_experiments = 1.** K_1_2, female only. Male-vs-male crossings
remain unmeasurable; female-vs-male ones are exactly what this measures.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c10_clean_to_clean_swaps.py
    .venv\\Scripts\\python.exe tools/c10_clean_to_clean_swaps.py --no-clips
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from c02_tracking_probe import CALIBRATION
from c04_score_tracking import reference_chain
from c07_render_tracking import INK, OTHER, PALETTE, PAPER, ViewData, colour_map, crop_box
from c08_crossing_strata import canonical, row, tiles_at
from c09_gt_swap_validation import female_anchors, outcome
from s10_shared_node_20k import median_span_px, tail_jump_threshold_px

from medaka3d import load_catalog, load_paths
from medaka3d.calib import load_rig
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.track import UNTRACKED
from medaka3d.types import FPS, VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN, METHOD = "bottomup", "gated"
ARCHIVE = Path(r"Z:/StephensU/medaka/archive/medaka-mating-pipeline/annotations")
SOURCES = {
    "clean_sample": ARCHIVE / "fish_sex_gt/20251225_K_1_2_FCMC/female_sex_gt.csv",
    "gate_branch": ARCHIVE / "fish_sex_gt/20251225_K_1_2_FCMC"
                             "/jump6_gate_branch_female_review.csv",
}
CLIP_FPS, CLIP_CONTEXT, CLIP_MAX = 30, 166, 900


def clean_segments(distance: np.ndarray, separation_mm: float, min_frames: int):
    """Maximal runs where the female is at least ``separation_mm`` from both others.

    NaN is *not* clean: an unreconstructed frame is not evidence of separation,
    the same asymmetry `encounters` applies.
    """
    apart = np.isfinite(distance) & (distance >= separation_mm)
    padded = np.concatenate(([False], apart, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)
            if b - a >= min_frames]


def steps_in(data: dict, column: int, lo: int, hi: int) -> dict:
    """The experimenter's heuristic: one-frame steps in inter-fish distance and speed."""
    others = [c for c in range(3) if c != column]
    her = data["xyz"][lo:hi, column, :]
    nearest = np.nanmin(np.stack([
        np.linalg.norm(her - data["xyz"][lo:hi, o, :], axis=1) for o in others
    ]), axis=0)
    speed = np.linalg.norm(np.diff(her, axis=0), axis=1) * FPS
    d_ifd = np.abs(np.diff(nearest))
    d_speed = np.abs(np.diff(speed))
    return {
        "ifd_step_mm": float(np.nanmax(d_ifd)) if np.isfinite(d_ifd).any() else np.nan,
        "speed_step_mm": float(np.nanmax(d_speed)) if np.isfinite(d_speed).any() else np.nan,
        "min_ifd_mm": float(np.nanmin(nearest)) if np.isfinite(nearest).any() else np.nan,
    }


def transition_figure(item, data, column_by_frame, path, title):
    """IFD and speed over the transition, with the two clean segments shaded."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    lo = max(0, item["clean_a"][1] - CLIP_CONTEXT * 2)
    hi = min(data["xyz"].shape[0], item["clean_b"][0] + CLIP_CONTEXT * 2)
    frames = np.arange(lo, hi)
    seconds = (frames - item["gap"][0]) / FPS
    column = item["column"]
    others = [c for c in range(3) if c != column]
    her = data["xyz"][lo:hi, column, :]
    nearest = np.nanmin(np.stack([
        np.linalg.norm(her - data["xyz"][lo:hi, o, :], axis=1) for o in others]), axis=0)
    speed = np.concatenate(([np.nan], np.linalg.norm(np.diff(her, axis=0), axis=1) * FPS))

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 0.7], "hspace": 0.18})
    fig.patch.set_facecolor(PAPER)
    for ax in axes:
        ax.set_facecolor(PAPER)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e6e5e2", linewidth=0.8)
        ax.set_axisbelow(True)
        for a, b in (item["clean_a"], item["clean_b"]):
            ax.axvspan((max(a, lo) - item["gap"][0]) / FPS, (min(b, hi) - item["gap"][0]) / FPS,
                       color="#eef3ee", zorder=0)

    axes[0].plot(seconds, nearest, color=PALETTE[0], linewidth=2.0)
    axes[0].axhline(item["separation_mm"], color="#8a8a85", linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("female IFD, nearest other (mm)")
    axes[0].set_ylim(bottom=0)
    axes[0].text(seconds[0], item["separation_mm"], " clean threshold", fontsize=8.5,
                 va="bottom", color="#52514e")

    axes[1].plot(seconds, speed, color=PALETTE[1], linewidth=1.4)
    axes[1].set_ylabel("female speed (mm/s)")
    axes[1].text(seconds[-1], np.nanmax(speed), f"max 1-frame step "
                 f"{item['speed_step_mm']:.0f} mm/s   IFD step {item['ifd_step_mm']:.1f} mm",
                 ha="right", va="top", fontsize=8.5, color="#52514e")

    ids = data["track"][lo:hi, column]
    counts = Counter(int(t) for t in ids if int(t) != UNTRACKED)
    hue = {t: (PALETTE[i] if i < len(PALETTE) else OTHER)
           for i, (t, _) in enumerate(counts.most_common())}
    edges = np.flatnonzero(np.diff(ids)) + 1
    for start, stop in zip(np.r_[0, edges], np.r_[edges, len(ids)], strict=True):
        value = int(ids[start])
        right = seconds[min(stop, len(ids) - 1)]
        axes[2].add_patch(Rectangle((seconds[start], 0.1), right - seconds[start], 0.8,
                                    color="#dcdbd7" if value == UNTRACKED
                                    else hue.get(value, OTHER), linewidth=0))
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([0.5], ["female ID"])
    axes[2].grid(False)
    axes[2].set_xlabel("seconds from the end of the first clean segment")
    for frame in (item["anchor_a"], item["anchor_b"]):
        for ax in axes:
            ax.axvline((frame - item["gap"][0]) / FPS, color=INK, linewidth=1.0, alpha=0.5)

    fig.suptitle(title, x=0.012, ha="left", fontsize=12)
    fig.text(0.012, 0.005, "Green bands are the clean segments; vertical rules are the human "
             "anchors. A swap should show as a step, not a smooth excursion.",
             fontsize=8, color="#52514e")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def bracketed(encounters: list[dict], anchors: list[dict], npz, column_of_anchor) -> list[dict]:
    """Each crossing, with the human annotation **before** it and the one **after**.

    This is the unit the annotation was made for. There is no ground truth
    *during* a crossing and there cannot be — the animals are overlapping, which
    is the whole reason the crossing is hard. What a human can say is which fish
    is the female on either side of it, and that is exactly what brackets the
    event.

    `c10`'s first pass demanded that the anchors sit inside a *clean segment* of
    its own construction, which threw away most of the annotation and kept only
    the easiest crossings. Bracketing needs no such construction: an encounter is
    already bounded on both sides by sustained separation, so an anchor before
    its entry and one after its exit are in clean conditions by definition.

    **Attribution.** A bracket counts only if no *other* encounter falls between
    the two anchors. Otherwise the crossing that cost her the identity is not
    identified, and a swap would be attributed to the wrong event. Brackets
    rejected on that ground are counted, not dropped silently.
    """
    frames = np.asarray([a["frame"] for a in anchors])
    spans = sorted(encounters, key=lambda e: (e["entry_frame"], e["exit_frame"]))
    out = []
    for encounter in spans:
        entry, exit_ = encounter["entry_frame"], encounter["exit_frame"]
        before = np.searchsorted(frames, entry, side="left") - 1
        after = np.searchsorted(frames, exit_, side="right")
        if before < 0 or after >= len(anchors):
            continue
        first, second = anchors[before], anchors[after]
        others = [
            e for e in encounters
            if e is not encounter
            and e["exit_frame"] >= first["frame"] and e["entry_frame"] <= second["frame"]
        ]
        column = column_of_anchor(first)
        out.append({
            "entry_frame": entry, "exit_frame": exit_,
            "stratum": encounter["stratum"], "crossed_views": encounter["crossed_views"],
            "min_distance_mm": encounter["min_distance_mm"],
            "c08_outcome": encounter["outcome"], "c08_id_changes": encounter["id_changes"],
            "anchor_before": first["frame"], "anchor_after": second["frame"],
            "lead_frames": entry - first["frame"], "lag_frames": second["frame"] - exit_,
            "source_before": first["source"], "source_after": second["source"],
            "other_encounters_between": len(others),
            "attributable": not others,
            "column": column,
            "pair": "F-M" if column in encounter["participants"] else "M-M",
            "chain_column_changed": first["column"] != second["column"],
            "track_before": first["track"], "track_after": second["track"],
            "outcome": outcome(npz, first, second),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--separation-mm", type=float, default=30.0)
    parser.add_argument("--min-clean", type=int, default=166)
    parser.add_argument("--match-px", type=float, default=5.0)
    parser.add_argument("--max-encounters", type=int, default=0,
                        help="Skip gaps holding more crossings than this. 0 keeps all.")
    parser.add_argument("--no-clips", action="store_true")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    rig = load_rig(CALIBRATION)
    out_dir = paths.work_dir / "c10_clean_to_clean_swaps"
    for sub in ("transitions", "montage", "clips"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    removed = 0
    for sub in ("transitions", "montage", "clips"):
        for stale in (out_dir / sub).iterdir():
            if stale.is_file() and stale.suffix in (".png", ".mp4"):
                stale.unlink()
                removed += 1
    if removed:
        print(f"cleared   : {removed} files from a previous run")

    with provenance("clean_to_clean_swaps", out_dir, name="c10") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        for name, path in SOURCES.items():
            if not path.exists():
                raise SystemExit(f"missing annotation source {name}: {path}")
            record.input(f"female_gt.{name}", path)

        span, gate = {}, {}
        for view in VIEWS:
            key = f"labels.round8.{'top' if view == 'top_cam' else 'side'}.train"
            span[view], _ = median_span_px(catalog.get(key))
            gate[view] = tail_jump_threshold_px(span[view])

        npz = np.load(paths.work_dir / "c03_track_3d" / RUN / f"{METHOD}__{EXPERIMENT}.npz",
                      allow_pickle=False)
        chain = reference_chain(npz["xyz"].astype(float), npz["slot_track"].max(axis=1) >= 0)
        data = canonical(npz, chain, rig.views)

        anchors = []
        for name, path in SOURCES.items():
            gt = pd.read_csv(path, low_memory=False)
            gt = gt[gt.gt_status.eq("confirmed")].sort_values("frame")
            found, dropped = female_anchors(paths, gt, npz, rig.views, args.match_px)
            for anchor in found:
                col = np.flatnonzero(chain[anchor["frame"]] == anchor["slot"])
                anchor["column"] = int(col[0]) if col.size == 1 else -1
                anchor["source"] = name
            print(f"{name:<14}: {len(gt):>4} confirmed -> {len(found):>4} anchors "
                  f"({dropped} dropped on view disagreement)")
            anchors += found
        anchors = sorted({a["frame"]: a for a in anchors}.values(), key=lambda a: a["frame"])
        record.param("anchors", len(anchors))

        strata = json.loads((paths.work_dir / "c08_crossing_strata"
                             / "c08_strata.json").read_text(encoding="utf-8"))
        encounters = [e for e in strata["encounters"] if e["experiment"] == EXPERIMENT]

        # --- clean segments, per the experimenter's definition ---------------
        n = data["xyz"].shape[0]
        nearest = np.full(n, np.nan)
        column_of = np.full(n, -1, dtype=np.int8)
        for anchor in anchors:
            column_of[anchor["frame"]] = anchor["column"]
        # The female's column is only known at anchors, so cleanliness is
        # evaluated on the *minimum* pairwise distance: if every pair is apart,
        # she is apart from both regardless of which column she occupies.
        pairwise_min = np.nanmin(np.stack([
            np.linalg.norm(data["xyz"][:, a, :] - data["xyz"][:, b, :], axis=1)
            for a, b in ((0, 1), (0, 2), (1, 2))]), axis=0)
        nearest[:] = pairwise_min
        segments = clean_segments(nearest, args.separation_mm, args.min_clean)
        print(f"\nclean segments : {len(segments)}  "
              f"({sum(b - a for a, b in segments) / FPS / 60:.1f} min of "
              f"{n / FPS / 60:.1f} min)")
        record.param("clean_segments", len(segments))

        by_segment: dict[int, list] = {}
        for anchor in anchors:
            for index, (a, b) in enumerate(segments):
                if a <= anchor["frame"] < b:
                    by_segment.setdefault(index, []).append(anchor)
                    break

        rows_out = []
        for index, (segment_a, segment_b) in enumerate(pairwise(segments)):
            left, right = by_segment.get(index, []), by_segment.get(index + 1, [])
            if not left or not right:
                continue
            first, second = left[-1], right[0]
            track_a = npz["slot_track"][segment_a[0]:segment_a[1]]
            track_b = npz["slot_track"][segment_b[0]:segment_b[1]]
            ids_a = track_a[:, chain[first["frame"], first["column"]]] \
                if first["column"] >= 0 else None
            gap = (segment_a[1], segment_b[0])
            inside = [e for e in encounters
                      if e["exit_frame"] >= gap[0] and e["entry_frame"] <= gap[1]]
            if args.max_encounters and len(inside) > args.max_encounters:
                continue
            steps = steps_in(data, first["column"], gap[0], gap[1]) \
                if first["column"] >= 0 else {"ifd_step_mm": np.nan,
                                              "speed_step_mm": np.nan, "min_ifd_mm": np.nan}
            rows_out.append({
                "clean_a": list(segment_a), "clean_b": list(segment_b), "gap": list(gap),
                "gap_frames": gap[1] - gap[0],
                "anchor_a": first["frame"], "anchor_b": second["frame"],
                "source_a": first["source"], "source_b": second["source"],
                "column": first["column"], "column_b": second["column"],
                "chain_column_changed": first["column"] != second["column"],
                "track_a": first["track"], "track_b": second["track"],
                "outcome": outcome(npz, first, second),
                "encounters": len(inside),
                "max_stratum": max((e["stratum"] for e in inside), default=0),
                "separation_mm": args.separation_mm,
                "unused_a": int(ids_a.size) if ids_a is not None else 0,
                "unused_b": int(track_b.size),
                **steps,
            })

        counts = Counter(r["outcome"] for r in rows_out)
        print(f"\nanchored transitions : {len(rows_out)}   median gap "
              f"{np.median([r['gap_frames'] for r in rows_out]):.0f} frames "
              f"({np.median([r['gap_frames'] for r in rows_out]) / FPS:.1f} s)")
        for key in ("preserved", "break", "swap"):
            print(f"  {key:<10} {counts[key]:>4}  {counts[key] / max(1, len(rows_out)):6.1%}")
        record.param("outcomes", dict(counts))

        print("\nby crossings inside the gap")
        for lo, hi in ((1, 1), (2, 3), (4, 999)):
            cell = [r for r in rows_out if lo <= r["encounters"] <= hi]
            if not cell:
                continue
            tally = Counter(r["outcome"] for r in cell)
            print(f"  {lo}-{hi:<3} n={len(cell):>3}  swap {tally['swap']:>2} "
                  f"({tally['swap'] / len(cell):5.1%})  break {tally['break']:>2}")

        print("\nindicators on this (differently defined) unit")
        for name in ("chain_column_changed", "ifd_step_mm", "speed_step_mm", "min_ifd_mm"):
            swaps = [r[name] for r in rows_out if r["outcome"] == "swap"]
            other = [r[name] for r in rows_out if r["outcome"] != "swap"]
            if name == "chain_column_changed":
                hit = [r for r in rows_out if r[name]]
                tally = Counter(r["outcome"] for r in hit)
                print(f"  {name:<22} flagged {len(hit):>3}  swaps in flagged "
                      f"{tally['swap']}/{counts['swap']}  precision "
                      f"{tally['swap'] / max(1, len(hit)):.1%}")
            else:
                swaps = [v for v in swaps if np.isfinite(v)]
                other = [v for v in other if np.isfinite(v)]
                print(f"  {name:<22} median when swap "
                      f"{np.median(swaps) if swaps else float('nan'):8.2f}   "
                      f"otherwise {np.median(other) if other else float('nan'):8.2f}")

        gate_rows = [r for r in rows_out
                     if "gate_branch" in (r["source_a"], r["source_b"])]
        print(f"\nindependent subset (gate_branch anchors): {len(gate_rows)} transitions  "
              + str(dict(Counter(r["outcome"] for r in gate_rows))))
        record.param("gate_branch_transitions", len(gate_rows))

        # --- the unit the annotation was made for: bracket each crossing ----
        brackets = bracketed(encounters, anchors, npz,
                             lambda a: a["column"] if a["column"] >= 0 else -1)
        good = [b for b in brackets if b["attributable"]]
        print(f"\nbracketed crossings   : {len(brackets)} of {len(encounters)}   "
              f"attributable {len(good)}  "
              f"(rejected {len(brackets) - len(good)} with another crossing between "
              f"the two anchors)")
        print(f"  median lead {np.median([b['lead_frames'] for b in good]):.0f} frames, "
              f"lag {np.median([b['lag_frames'] for b in good]):.0f} frames")
        tally = Counter(b["outcome"] for b in good)
        for key in ("preserved", "break", "swap"):
            print(f"  {key:<10} {tally[key]:>4}  {tally[key] / max(1, len(good)):6.1%}")
        record.param("bracketed", {"n": len(brackets), "attributable": len(good),
                                   **{k: tally[k] for k in
                                      ("preserved", "break", "swap")}})

        print(f"\n{'pair':<5} {'views':>6} {'n':>5} {'preserved':>10} {'break':>7} {'swap':>7}")
        bracket_table = {}
        for pair in ("F-M", "M-M"):
            for stratum in (0, 1, 2, 3):
                cell = [b for b in good if b["pair"] == pair and b["stratum"] == stratum]
                if not cell:
                    continue
                c = Counter(b["outcome"] for b in cell)
                bracket_table[f"{pair}_s{stratum}"] = {
                    "n": len(cell), **{k: c[k] for k in ("preserved", "break", "swap")}}
                print(f"{pair:<5} {stratum:>6} {len(cell):>5} {c['preserved']:>10} "
                      f"{c['break']:>7} {c['swap']:>7}")
        record.param("bracketed_by_pair_stratum", bracket_table)

        flagged = [b for b in good if b["chain_column_changed"]]
        caught = sum(b["outcome"] == "swap" for b in flagged)
        swaps = tally["swap"]
        print(f"\nchain_column_changed on this unit: flagged {len(flagged)} of {len(good)}   "
              f"swaps caught {caught}/{swaps}   "
              f"precision {caught / max(1, len(flagged)):.1%}")
        record.param("bracketed_chain_indicator", {
            "flagged": len(flagged), "swaps_total": swaps,
            "swaps_caught": caught})

        # --- render for direct human verification --------------------------
        review = sorted(
            [r for r in rows_out if r["outcome"] == "swap" or r["chain_column_changed"]],
            key=lambda r: (r["outcome"] != "swap", r["gap"][0]),
        )
        print(f"\nrendering {len(review)} transitions for review")
        views = {v: ViewData(paths.work_dir, RUN, EXPERIMENT, v) for v in VIEWS}
        import sleap_io

        videos = {v: sleap_io.Video.from_filename(str(paths.video(EXPERIMENT, v)))
                  for v in VIEWS}
        shape = videos[VIEWS[0]].shape[1:3]
        manifest = []
        for item in review:
            stem = f"{item['outcome']}_f{item['gap'][0]}_to_f{item['gap'][1]}"
            title = (f"{item['outcome']}  ·  {item['encounters']} crossing(s) in the gap  ·  "
                     f"{EXPERIMENT}  ·  track {item['track_a']} -> {item['track_b']}")
            transition_figure(item, data, column_of, out_dir / "transitions" / f"{stem}.png",
                              title)
            window = np.arange(max(0, item["gap"][0] - CLIP_CONTEXT),
                               min(c01.N_FRAMES, min(item["gap"][1] + CLIP_CONTEXT,
                                                     item["gap"][0] + CLIP_MAX)))
            colours = colour_map({v: views[v] for v in VIEWS}, window[::5])
            boxes = {v: crop_box({RUN: views[v]}, window[::5], shape, span[v]) for v in VIEWS}
            note = title + "  |  crop 6 body lengths; labels are track IDs"
            still = row(tiles_at(int(np.mean(item["gap"])), videos, views, boxes, colours), note)
            cv2.imwrite(str(out_dir / "montage" / f"{stem}.png"), still)
            if not args.no_clips:
                writer = cv2.VideoWriter(str(out_dir / "clips" / f"{stem}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS,
                                         (still.shape[1], still.shape[0]))
                for f in window:
                    writer.write(row(tiles_at(int(f), videos, views, boxes, colours), note))
                writer.release()
            manifest.append({**item, "files": stem})
            print(f"  {stem}")

        (out_dir / "c10_transitions.json").write_text(
            json.dumps({"experiment": EXPERIMENT, "n_groups": 1,
                        "groups": [ExperimentId.parse(EXPERIMENT).group],
                        "separation_mm": args.separation_mm, "min_clean": args.min_clean,
                        "outcomes": dict(counts), "transitions": rows_out,
                        "bracketed": brackets,
                        "bracketed_by_pair_stratum": bracket_table,
                        "rendered": manifest}, indent=2, default=float),
            encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

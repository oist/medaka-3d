"""C08 — stratify encounters by *how many cameras actually see the crossing*, and
pull the scenes for each stratum.

`c04` scored tracking on one 3D distance and reported a single encounter-
preservation rate. That pools two physically different events: a pair that
passes at 15 mm but stays separated in every image, and a pair that occludes
each other in all three cameras at once. The tracker's job is not the same in
those two cases, and neither is its failure rate.

=============================================================================
DEFINITIONS — fixed in code before the first number was read
=============================================================================

**Encounters** are `medaka3d.eval.encounters`' — windows bounded on both sides
by sustained 3D separation, on the geometric reference chain so the window is
identical for every method. Unchanged from `c04`.

**A view crossing** is the two participants' *pectorals* staying within that
view's gate — half the median labelled mouth-tail span, 60 px top / 50 px side,
the same gate `s10` and `c04` derive — for at least ``--min-crossing-frames``
**consecutive** frames inside ``[entry_frame, exit_frame)``. Pectoral rather
than mouth because it is the node bottom-up never fails to place (0.000% of
complete frames, 2026-08-13), so the stratum is a property of the geometry
rather than of a dropped detection.

The persistence requirement exists because a plain minimum lets one bad frame
promote an encounter a whole stratum, and 6 ms of proximity is not an occlusion.
The default of 3 frames (18 ms) is stated rather than tuned. **Its effect was
measured, not assumed**: ``--min-crossing-frames 1`` moves 65 of 1,227
encounters up a stratum (stratum 3 goes 555 -> 620) and changes no ``held`` rate
by more than 8 points, leaving the monotone ordering across strata unchanged.
The headline does not depend on this choice.

**The stratum** is the number of views (0-3) in which that happened. Stratum 0
is real and is kept: two animals 20 mm apart in 3D can stay resolved in every
camera.

**The outcome**, comparing ``entry_frame - 1`` with ``exit_frame``:

* ``held`` — each participant column carries **one** track ID continuously
  across the whole window, and the two IDs differ.
* ``exchanged`` — the ID column A held at entry is on column B at exit and vice
  versa. Read this as **the tracker and the geometric reference chain disagree
  about which column is which**, not as a confessed swap: both are geometric
  assignments and either could be the one that flipped. `c04`'s docstring makes
  the same caveat about the chain and it applies here unchanged.
* ``broken`` — any other discontinuity: an ID died, a new one appeared, or a
  boundary was untracked.

``held`` is strictly stronger than `c04`'s ``preserved``, which compares only
the two endpoints and scores a window containing a hundred ID changes as
preserved if the endpoints happen to match. Both are reported so the gap is
visible rather than argued about.

**What this still cannot do.** There is no human M1/M2 track in this project, so
none of this measures whether the *right* animal came out — only whether the
tracker maintained an identity to be right or wrong about. ``exchanged`` is the
tracker's own claim, not a verified swap. A swap that the tracker performs
confidently and continuously reads as ``held``.

=============================================================================
WHAT IS RENDERED
=============================================================================

Selection is deterministic: within each (stratum, outcome) cell, encounters are
sorted by closest approach and sampled at even intervals. No randomness, and no
picking the scenes that flatter a conclusion. Cells that exist but are not
rendered are printed with their counts, so nothing is silently dropped.

Three outputs per selected encounter:

1. ``strata/`` — the diagnostic figure. 3D separation, then the **per-view
   separation in body lengths** for all three cameras on one axis with the
   0.5-body-length crossing line, then the two participants' track-ID strips.
   This is the panel that shows a pair crossing in two cameras and staying clear
   in the third.
2. ``montage/`` — a 1 x 3 still at closest approach, views across, cropped to a
   fixed 6 body lengths so two scenes are directly comparable.
3. ``clips/`` — the same row as MP4 across the encounter plus context, at 30 fps
   from 166 fps source (1/5.5 speed). Tracking is temporal; a still cannot show
   a swap happening.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c08_crossing_strata.py
    .venv\\Scripts\\python.exe tools/c08_crossing_strata.py --per-cell 2 --no-clips
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from c02_tracking_probe import CALIBRATION
from c04_score_tracking import reference_chain
from c07_render_tracking import (
    INK,
    OTHER,
    PALETTE,
    PAPER,
    TILE,
    ViewData,
    bgr,
    caption,
    colour_map,
    crop_box,
    draw,
    frame_bgr,
)
from s10_shared_node_20k import median_span_px, tail_jump_threshold_px

from medaka3d import load_catalog, load_paths
from medaka3d.calib import load_rig
from medaka3d.cohort import ExperimentId
from medaka3d.eval import EncounterConfig, find_encounters
from medaka3d.provenance import provenance
from medaka3d.track import UNTRACKED
from medaka3d.types import FPS, VIEWS

RUN, METHOD = "bottomup", "gated"
OUTCOMES = ("held", "exchanged", "broken")
STRATA = (0, 1, 2, 3)

#: Playback rate. Source is 166 fps, so this is 1/5.5 speed.
CLIP_FPS = 30
#: Context frames on each side of the encounter in the clip.
CLIP_CONTEXT = 166
#: Frames a clip may span before it is truncated. Truncation is printed.
CLIP_MAX_FRAMES = 900


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def canonical(npz, chain: np.ndarray, rig_views: tuple[str, ...]) -> dict:
    """The reference-chain view of one run: xyz, track ids, and per-view pectorals.

    **The per-view node arrays are indexed by detection row within the frame, not
    by reconstruction slot.** `c03` writes them straight from the detector, and
    the slot -> row map is ``source``. Indexing them by slot silently pairs one
    animal's 3D position with another animal's pixels — which is how the first
    version of this tool reported two fish 13 mm apart in 3D as eight body
    lengths apart in both side cameras. That is impossible, and noticing it is
    the only reason the bug was found.
    """
    rows = np.arange(chain.shape[0])[:, None]
    ok = chain >= 0
    take = np.maximum(chain, 0)
    source = npz["source"][rows, take]  # (n_frames, 3, n_views), canonical order
    out = {
        "xyz": np.where(ok[:, :, None], npz["xyz"].astype(float)[rows, take], np.nan),
        "track": np.where(ok, npz["slot_track"][rows, take], UNTRACKED),
    }
    for view in VIEWS:
        node = npz[f"pectoral__{view}"].astype(float)
        detection_row = source[:, :, rig_views.index(view)]
        gathered = node[rows, np.maximum(detection_row, 0)]
        out[view] = np.where((ok & (detection_row >= 0))[:, :, None], gathered, np.nan)
    return out


def classify(track: np.ndarray, a: int, b: int, lo: int, hi: int) -> tuple[str, int]:
    """``(outcome, id changes)`` for one encounter's two participant columns."""
    seg_a, seg_b = track[lo - 1 : hi + 1, a], track[lo - 1 : hi + 1, b]
    entry_a, entry_b, exit_a, exit_b = seg_a[0], seg_b[0], seg_a[-1], seg_b[-1]
    changes = int((seg_a[1:] != seg_a[:-1]).sum() + (seg_b[1:] != seg_b[:-1]).sum())
    held = (seg_a == entry_a).all() and (seg_b == entry_b).all()
    if held and entry_a != UNTRACKED and entry_b != UNTRACKED and entry_a != entry_b:
        return "held", changes
    if UNTRACKED not in (entry_a, entry_b) and exit_a == entry_b and exit_b == entry_a:
        return "exchanged", changes
    return "broken", changes


def sustained_below(distance: np.ndarray, gate: float, min_frames: int) -> bool:
    """Does ``distance`` stay under ``gate`` for ``min_frames`` consecutive frames?

    NaN counts as *not* below: an unreconstructed frame is not evidence that two
    animals overlapped, the same asymmetry `encounters` applies to separation.
    """
    below = np.isfinite(distance) & (distance <= gate)
    if not below.any():
        return False
    if min_frames <= 1:
        return True
    padded = np.concatenate(([False], below, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return bool(((edges[1::2] - edges[::2]) >= min_frames).any())


def analyse(paths, experiment: str, gate: dict, span: dict, min_crossing: int,
            rig_views: tuple[str, ...]) -> list[dict]:
    """Every encounter of one experiment, with its stratum and its outcome."""
    npz = np.load(
        paths.work_dir / "c03_track_3d" / RUN / f"{METHOD}__{experiment}.npz",
        allow_pickle=False,
    )
    chain = reference_chain(npz["xyz"].astype(float), npz["slot_track"].max(axis=1) >= 0)
    data = canonical(npz, chain, rig_views)
    encounters = find_encounters(data["xyz"], experiment=experiment, config=EncounterConfig())

    out = []
    for e in encounters:
        a, b = e.participants
        lo, hi = e.entry_frame, min(e.exit_frame, chain.shape[0] - 1)
        separation, crossed = {}, []
        for view in VIEWS:
            d = np.linalg.norm(data[view][lo:hi, a, :] - data[view][lo:hi, b, :], axis=1)
            finite = d[np.isfinite(d)]
            separation[view] = float(finite.min()) if finite.size else float("nan")
            if sustained_below(d, gate[view], min_crossing):
                crossed.append(view)
        outcome, changes = classify(data["track"], a, b, lo, hi)

        # `c04`'s endpoint-only rule, recomputed here so the two are comparable
        # on identical windows rather than across two runs of two tools.
        endpoints = data["track"][[lo - 1, hi]][:, [a, b]]
        preserved = bool(
            (endpoints[0] == endpoints[1]).all() and UNTRACKED not in endpoints[0]
        )
        centre = lo + int(np.nanargmin(np.linalg.norm(
            data["xyz"][lo:hi, a, :] - data["xyz"][lo:hi, b, :], axis=1)))
        out.append({
            "experiment": experiment,
            "entry_frame": lo,
            "exit_frame": hi,
            "closest_frame": centre,
            "participants": [int(a), int(b)],
            "stratum": len(crossed),
            "crossed_views": crossed,
            "min_separation_px": separation,
            "min_separation_body_lengths": {v: separation[v] / span[v] for v in VIEWS},
            "min_distance_mm": e.min_distance_mm,
            "duration_frames": hi - lo,
            "outcome": outcome,
            "id_changes": changes,
            "preserved_endpoints_only": preserved,
        })
    return out


def a_slot(record: dict) -> int:
    return int(record["participants"][0])


def b_slot(record: dict) -> int:
    return int(record["participants"][1])


def evenly(items: list, n: int) -> list:
    """``n`` items sampled at even intervals from an already-sorted list.

    Even sampling rather than the head, so a cell is represented across its whole
    range instead of only by its most extreme members.
    """
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    step = (len(items) - 1) / (n - 1) if n > 1 else 0.0
    return [items[round(i * step)] for i in range(n)]


def tabulate(records: list[dict]) -> dict:
    """The stratum x outcome table, plus `c04`'s endpoint rule beside it."""
    cells = Counter((r["stratum"], r["outcome"]) for r in records)
    table = {}
    for stratum in STRATA:
        rows = [r for r in records if r["stratum"] == stratum]
        if not rows:
            continue
        table[str(stratum)] = {
            "encounters": len(rows),
            **{o: cells[(stratum, o)] / len(rows) for o in OUTCOMES},
            "median_id_changes": float(np.median([r["id_changes"] for r in rows])),
            "median_duration_s": float(np.median([r["duration_frames"] for r in rows]) / FPS),
            "preserved_endpoints_only": float(
                np.mean([r["preserved_endpoints_only"] for r in rows])
            ),
        }
    return table


# --------------------------------------------------------------------------
# The diagnostic figure
# --------------------------------------------------------------------------
def strata_figure(record: dict, data: dict, span: dict, path: Path) -> None:
    """3D separation, per-view separation in body lengths, and the two ID strips."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    a, b = record["participants"]
    lo = max(0, record["entry_frame"] - CLIP_CONTEXT)
    hi = min(data["xyz"].shape[0], record["exit_frame"] + CLIP_CONTEXT)
    frames = np.arange(lo, hi)
    seconds = (frames - record["entry_frame"]) / FPS

    fig, axes = plt.subplots(
        3, 1, figsize=(11.5, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 2.4, 0.8], "hspace": 0.18},
    )
    fig.patch.set_facecolor(PAPER)
    for ax in axes:
        ax.set_facecolor(PAPER)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e6e5e2", linewidth=0.8)
        ax.set_axisbelow(True)

    # --- 3D separation: one series, so no legend; the title names it.
    d3 = np.linalg.norm(data["xyz"][lo:hi, a, :] - data["xyz"][lo:hi, b, :], axis=1)
    axes[0].plot(seconds, d3, color=INK, linewidth=2.0)
    axes[0].axhline(EncounterConfig().close_mm, color="#8a8a85", linewidth=1.0, linestyle="--")
    axes[0].text(seconds[-1], EncounterConfig().close_mm, "  20 mm", va="center",
                 fontsize=8.5, color="#52514e")
    axes[0].set_ylabel("3D separation (mm)")
    axes[0].set_ylim(bottom=0)

    # --- per-view separation, expressed in body lengths so one axis serves all
    # three cameras and the crossing gate is a single line at 0.5.
    for i, view in enumerate(VIEWS):
        d = np.linalg.norm(data[view][lo:hi, a, :] - data[view][lo:hi, b, :], axis=1) / span[view]
        colour = PALETTE[i]
        crossed = view in record["crossed_views"]
        axes[1].plot(seconds, d, color=colour, linewidth=2.0 if crossed else 1.3,
                     alpha=1.0 if crossed else 0.55)
        finite = np.flatnonzero(np.isfinite(d))
        if finite.size:
            axes[1].annotate(
                f" {view.replace('_cam', '')}" + ("  ✓crossed" if crossed else ""),
                (seconds[finite[-1]], d[finite[-1]]), fontsize=8.5,
                color=colour if crossed else "#52514e", va="center",
            )
    axes[1].axhline(0.5, color="#8a8a85", linewidth=1.0, linestyle="--")
    axes[1].text(seconds[0], 0.5, "0.5 body length — crossing gate  ", ha="left", va="bottom",
                 fontsize=8.5, color="#52514e")
    axes[1].set_ylabel("in-view separation (body lengths)")
    axes[1].set_ylim(0, max(3.0, float(np.nanmax([
        np.nanmax(np.linalg.norm(data[v][lo:hi, a, :] - data[v][lo:hi, b, :], axis=1) / span[v])
        for v in VIEWS
    ]))))
    axes[1].legend(
        handles=[plt.Line2D([], [], color=PALETTE[i], linewidth=2.0, label=v.replace("_cam", ""))
                 for i, v in enumerate(VIEWS)],
        loc="upper right", frameon=False, fontsize=8.5, ncols=3,
    )

    # --- the two participants' track IDs. Colours rank the window's longest
    # tracks; a fourth folds into the neutral rather than inventing a hue.
    strip = data["track"][lo:hi][:, [a, b]]
    counts = Counter(int(t) for t in strip.ravel() if int(t) != UNTRACKED)
    ranked = [t for t, _ in counts.most_common()]
    hue = {t: (PALETTE[i] if i < len(PALETTE) else OTHER) for i, t in enumerate(ranked)}
    for row in (0, 1):
        ids = strip[:, row]
        edges = np.flatnonzero(np.diff(ids)) + 1
        for start, stop in zip(np.r_[0, edges], np.r_[edges, len(ids)], strict=True):
            value = int(ids[start])
            right = seconds[min(stop, len(ids) - 1)]
            axes[2].add_patch(Rectangle(
                (seconds[start], 0.55 - row * 0.5), right - seconds[start], 0.4,
                color="#dcdbd7" if value == UNTRACKED else hue.get(value, OTHER),
                linewidth=0,
            ))
    axes[2].set_ylim(0, 1.05)
    axes[2].set_yticks([0.75, 0.25], [f"slot {a}", f"slot {b}"])
    axes[2].grid(False)
    axes[2].set_xlabel("seconds from encounter entry")
    axes[2].text(seconds[-1], 1.02, f"{record['id_changes']} ID changes",
                 ha="right", va="bottom", fontsize=8.5, color="#52514e")

    for ax in axes:
        ax.axvspan(0.0, (record["exit_frame"] - record["entry_frame"]) / FPS,
                   color="#f0efec", zorder=0)

    crossed = ", ".join(v.replace("_cam", "") for v in record["crossed_views"]) or "none"
    fig.suptitle(
        f"{record['outcome']}  ·  crossed in {record['stratum']} view(s) [{crossed}]  ·  "
        f"{record['experiment']}  ·  frames {record['entry_frame']:,}-{record['exit_frame']:,}",
        x=0.012, ha="left", fontsize=12,
    )
    fig.text(0.012, 0.005,
             "Shaded band is the encounter (entry to exit); the flanks are context. "
             "Strip colours rank this window's longest tracks and are not a correspondence "
             "with any other figure; pale grey is untracked.",
             fontsize=8, color="#52514e")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


# --------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------
def tiles_at(frame: int, videos: dict, views: dict, boxes: dict, colours: dict) -> dict:
    """One frame drawn for all three cameras, cropped and scaled identically."""
    out = {}
    for view in VIEWS:
        image = frame_bgr(videos[view], frame)
        points, tracks = views[view].at(int(frame))
        _x0, _y0, width, _h = boxes[view]
        out[view] = draw(image, points, tracks, colours, boxes[view], TILE / width)
    return out


def row(tiles: dict, note: str) -> np.ndarray:
    """1 run x 3 views with a wrapped header."""
    from c07_render_tracking import _wrap

    body = np.hstack([caption(tiles[v], f"{RUN}  ·  {v}") for v in VIEWS])
    lines = _wrap(note, body.shape[1] - 20, 0.58)
    header = np.full((14 + 24 * len(lines), body.shape[1], 3), bgr(PAPER), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(header, line, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, bgr(INK), 1, cv2.LINE_AA)
    return np.vstack([header, body])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--experiments", nargs="+", default=list(c01.EXPERIMENTS))
    parser.add_argument("--per-cell", type=int, default=2,
                        help="Encounters rendered per (stratum, outcome) cell.")
    parser.add_argument("--clips-per-cell", type=int, default=1)
    parser.add_argument("--no-clips", action="store_true")
    parser.add_argument("--min-crossing-frames", type=int, default=3,
                        help="Consecutive frames under the gate before a view counts "
                             "as crossed. 1 reproduces the unfiltered minimum.")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    rig = load_rig(CALIBRATION)
    out_dir = paths.work_dir / "c08_crossing_strata"
    # Clear the render directories first. A run with different thresholds — or
    # different code — writes different stems, and the survivors of the previous
    # run then sit beside the new ones looking equally current. This directory
    # already accumulated 22 figures carrying a stratum that a later bug fix
    # changed, which is exactly the failure `MANIFEST.json` exists to prevent
    # elsewhere. Removed counts are printed, never silent.
    removed = 0
    for sub in ("strata", "montage", "clips"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
        for stale in (out_dir / sub).iterdir():
            if stale.is_file() and stale.suffix in (".png", ".mp4"):
                stale.unlink()
                removed += 1
    if removed:
        print(f"cleared   : {removed} files from a previous run")

    with provenance("crossing_strata", out_dir, name="c08") as record:
        record.params_from(args)
        groups = sorted({ExperimentId.parse(e).group for e in args.experiments})
        record.param("n_groups", len(groups))

        gate, span = {}, {}
        for view in VIEWS:
            key = f"labels.round8.{'top' if view == 'top_cam' else 'side'}.train"
            record.input(key, catalog.get(key))
            span[view], _ = median_span_px(catalog.get(key))
            gate[view] = tail_jump_threshold_px(span[view])
        print(f"n_groups = {len(groups)} {groups}")
        print("gates    : " + "  ".join(f"{v} {gate[v]:.0f}px" for v in VIEWS))

        records = []
        for experiment in args.experiments:
            found = analyse(paths, experiment, gate, span, args.min_crossing_frames,
                            rig.views)
            print(f"{experiment}: {len(found)} encounters")
            records += found

        table = tabulate(records)
        record.param("strata", table)
        print(f"\n{'views':>5} {'n':>6} {'held':>8} {'exchanged':>10} {'broken':>8}"
              f" {'c04 preserved':>14} {'med ID chg':>11}")
        for stratum, cell in table.items():
            print(f"{stratum:>5} {cell['encounters']:>6} {cell['held']:>7.1%}"
                  f" {cell['exchanged']:>10.1%} {cell['broken']:>8.1%}"
                  f" {cell['preserved_endpoints_only']:>13.1%}"
                  f" {cell['median_id_changes']:>11.0f}")

        # --- selection: deterministic, hardest first, evenly sampled -------
        chosen, skipped = [], {}
        for stratum in STRATA:
            for outcome in OUTCOMES:
                cell = sorted(
                    (r for r in records if r["stratum"] == stratum and r["outcome"] == outcome),
                    key=lambda r: (r["min_distance_mm"], r["entry_frame"]),
                )
                if not cell:
                    continue
                picked = evenly(cell, args.per_cell)
                chosen += picked
                if len(cell) > len(picked):
                    skipped[f"{stratum}_{outcome}"] = len(cell) - len(picked)
        record.param("not_rendered", skipped)
        print(f"\nrendering {len(chosen)} of {len(records)} encounters; "
              f"not rendered: {sum(skipped.values()):,}")

        import sleap_io

        manifest = []
        for experiment in args.experiments:
            here = [r for r in chosen if r["experiment"] == experiment]
            if not here:
                continue
            npz = np.load(
                paths.work_dir / "c03_track_3d" / RUN / f"{METHOD}__{experiment}.npz",
                allow_pickle=False,
            )
            chain = reference_chain(
                npz["xyz"].astype(float), npz["slot_track"].max(axis=1) >= 0
            )
            data = canonical(npz, chain, rig.views)
            views = {v: ViewData(paths.work_dir, RUN, experiment, v) for v in VIEWS}
            videos = {v: sleap_io.Video.from_filename(str(paths.video(experiment, v)))
                      for v in VIEWS}
            shape = videos[VIEWS[0]].shape[1:3]

            for rank, item in enumerate(here):
                stem = (f"s{item['stratum']}_{item['outcome']}_"
                        f"{experiment.split('_FCMC')[0][9:]}_f{item['entry_frame']}")
                strata_figure(item, data, span, out_dir / "strata" / f"{stem}.png")

                lo = max(0, item["entry_frame"] - CLIP_CONTEXT)
                hi = min(c01.N_FRAMES, item["exit_frame"] + CLIP_CONTEXT)
                truncated = max(0, (hi - lo) - CLIP_MAX_FRAMES)
                window = np.arange(lo, min(hi, lo + CLIP_MAX_FRAMES))
                colours = colour_map({v: views[v] for v in VIEWS}, window[::5])
                # Centre each tile on the two participants at closest approach —
                # not on the window's centroid, which for a multi-second encounter
                # is a place neither animal ever was.
                boxes = {}
                for view in VIEWS:
                    pair = data[view][item["closest_frame"], [a_slot(item), b_slot(item)], :]
                    focus = tuple(np.nanmean(pair, axis=0).tolist()) \
                        if np.isfinite(pair).any() else None
                    boxes[view] = crop_box({RUN: views[view]}, window[::5], shape,
                                           span[view], focus)
                crossed = ", ".join(v.replace("_cam", "") for v in item["crossed_views"]) or "none"
                note = (f"{item['outcome']}  ·  crossed in {item['stratum']} view(s) [{crossed}]"
                        f"  ·  {experiment}  ·  frames {item['entry_frame']:,}-"
                        f"{item['exit_frame']:,}  ·  closest 3D {item['min_distance_mm']:.1f} mm"
                        f"  ·  {item['id_changes']} ID changes  |  crop 6 body lengths; "
                        f"colours are track IDs, labels are the IDs themselves")

                still = row(
                    tiles_at(item["closest_frame"], videos, views, boxes, colours), note
                )
                cv2.imwrite(str(out_dir / "montage" / f"{stem}.png"), still)

                every = max(1, args.per_cell // max(1, args.clips_per_cell))
                wrote_clip = False
                if not args.no_clips and rank % every == 0:
                    writer = cv2.VideoWriter(
                        str(out_dir / "clips" / f"{stem}.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS,
                        (still.shape[1], still.shape[0]),
                    )
                    for f in window:
                        writer.write(
                            row(tiles_at(int(f), videos, views, boxes, colours), note)
                        )
                    writer.release()
                    wrote_clip = True

                manifest.append({**item, "files": stem, "clip": wrote_clip,
                                 "clip_truncated_frames": int(truncated)})
                print(f"  {stem}" + (f"  (clip truncated {truncated} frames)" if truncated else ""))

        record.param("rendered", len(manifest))
        (out_dir / "c08_strata.json").write_text(
            json.dumps({"n_groups": len(groups), "groups": groups, "run": RUN,
                        "method": METHOD, "gate_px": gate, "span_px": span,
                        "strata": table, "not_rendered": skipped,
                        "encounters": records, "rendered": manifest},
                       indent=2, default=float),
            encoding="utf-8",
        )
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

"""C14 — per-frame 2D displacement of every instance, across the eight measured swaps.

At 166 fps a pectoral moves ~1.5 px per frame, and the *ratio* of a step to its
own local neighbourhood has median exactly 1.00. A single frame that moves far
more than its neighbourhood is either a swap or a misdetection, and the two are
separable: a swap lands one animal where another was, so its displacement is
close to the distance to the nearest other instance. A misdetection is not
constrained that way.

This draws that directly, for every instance in every camera, over each of the
eight intervals `c09` measured a female swap in. Nothing is summarised here; the
point is to look.

=============================================================================
WHAT IS DRAWN
=============================================================================

One figure per interval, three panels — one camera each, on a log y axis because
the median step is 1.5 px and the events are 50-200 px:

* **thick coloured line per track** — that track's displacement between adjacent
  frames, in pixels. One per animal the tracker is holding; a track that dies
  simply stops.
* **thin line of the same colour** — that track's distance to the nearest other
  instance in the same frame. **Where the thick line touches its own thin line
  the animal moved by exactly the gap to its neighbour**, which is what a swap
  looks like.
* **red rings** — frames passing the detector: step at least ``--excess`` times
  the local median, and step over nearest-neighbour distance inside
  ``--ratio-band``.
* **vertical rules** — the two human female annotations bracketing the interval.
  The swap is somewhere between them and nowhere else.
* **shaded bands** — `c08` encounters inside the interval.

The baseline is the median of the same track's steps over +/- ``--window``
frames, excluding the frame itself, so a step is judged against its own
neighbourhood rather than a global constant.

The thresholds were chosen **after** seeing which intervals they hit. They are a
display setting, not a validated detector, and are exposed as flags so the
picture can be read at other settings.

**n_groups = 1, n_experiments = 1, female only.**

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c14_step_plots.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from c07_render_tracking import INK, OTHER, PALETTE, PAPER
from s10_shared_node_20k import median_span_px

from medaka3d import load_catalog, load_paths
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.types import FPS, VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN, METHOD = "bottomup", "gated"
ALERT = "#e34948"


def per_view_series(detections, track, window: int) -> dict:
    """Step, local baseline and nearest-neighbour distance, per track, per frame."""
    pectoral = detections.node("pectoral")
    start, stop = detections.start, detections.stop
    nearest = np.full(detections.frame_idx.shape[0], np.nan)
    for frame in range(detections.n_frames):
        lo, hi = int(start[frame]), int(stop[frame])
        if hi - lo < 2:
            continue
        points = pectoral[lo:hi]
        distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        np.fill_diagonal(distance, np.inf)
        nearest[lo:hi] = np.nanmin(distance, axis=1)

    out = {}
    for identifier in np.unique(track[track >= 0]):
        rows = np.flatnonzero(track == identifier)
        frames = detections.frame_idx[rows]
        order = np.argsort(frames)
        rows, frames = rows[order], frames[order]
        if frames.size < 2:
            continue
        adjacent = np.diff(frames) == 1
        moved = np.linalg.norm(pectoral[rows][1:] - pectoral[rows][:-1], axis=1)
        step = np.where(adjacent, moved, np.nan)
        padded = np.pad(step, (window, window), constant_values=np.nan)
        stack = np.stack([padded[i:i + step.size] for i in range(2 * window + 1)])
        stack[window] = np.nan
        with np.errstate(all="ignore"):
            baseline = np.nanmedian(stack, axis=0)
        out[int(identifier)] = {
            "frame": frames[1:], "step": step, "baseline": baseline,
            "nearest": nearest[rows][:-1],
        }
    return out


def figure(item, series, encounters, path, title, args) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = item["t1"], item["t2"]
    fig, axes = plt.subplots(len(VIEWS), 1, figsize=(12.5, 8.4), sharex=True,
                             gridspec_kw={"hspace": 0.16})
    fig.patch.set_facecolor(PAPER)
    flagged_total = 0
    for panel, view in zip(axes, VIEWS, strict=True):
        panel.set_facecolor(PAPER)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
        panel.grid(axis="y", color="#e6e5e2", linewidth=0.8)
        panel.set_axisbelow(True)
        for encounter in encounters:
            panel.axvspan((encounter["entry_frame"] - lo) / FPS,
                          (encounter["exit_frame"] - lo) / FPS,
                          color="#f0efec", zorder=0)
        ranked = sorted(series[view].items(), key=lambda kv: -kv[1]["frame"].size)
        drawn = 0
        for identifier, data in ranked:
            inside = (data["frame"] >= lo) & (data["frame"] <= hi)
            if inside.sum() < 2:
                continue
            colour = PALETTE[drawn] if drawn < len(PALETTE) else OTHER
            drawn += 1
            seconds = (data["frame"][inside] - lo) / FPS
            step, near = data["step"][inside], data["nearest"][inside]
            panel.plot(seconds, near, color=colour, linewidth=0.8, alpha=0.35)
            panel.plot(seconds, step, color=colour, linewidth=1.4,
                       label=f"track {identifier}")
            with np.errstate(all="ignore"):
                ratio = step / np.where(near > 0, near, np.nan)
                excess = step / np.maximum(data["baseline"][inside], 1e-6)
            hit = ((excess >= args.excess) & (ratio >= args.ratio_band[0])
                   & (ratio <= args.ratio_band[1]))
            hit = np.where(np.isfinite(hit), hit, False).astype(bool)
            flagged_total += int(hit.sum())
            panel.plot(seconds[hit], step[hit], "o", markerfacecolor="none",
                       markeredgecolor=ALERT, markersize=9, markeredgewidth=1.6)
        panel.set_yscale("log")
        panel.set_ylabel(f"{view.replace('_cam', '')}\nstep (px)")
        if drawn:
            panel.legend(loc="upper right", frameon=False, fontsize=8, ncols=4)
        for anchor in (item["t1"], item["t2"]):
            panel.axvline((anchor - lo) / FPS, color=INK, linewidth=1.1, alpha=0.6)
    axes[-1].set_xlabel("seconds from the opening annotation")
    fig.suptitle(title, x=0.012, ha="left", fontsize=12)
    fig.text(0.012, 0.004,
             "Thick line: that track's per-frame displacement. Thin line of the same colour: "
             "its distance to the nearest other instance - where the two meet, the animal "
             "moved by exactly the gap to its neighbour. Red rings pass the step detector. "
             "Vertical rules are the human annotations; the swap lies between them.",
             fontsize=8, color="#52514e")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return flagged_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--excess", type=float, default=10.0)
    parser.add_argument("--ratio-band", type=float, nargs=2, default=(0.7, 1.3))
    parser.add_argument("--npz", type=Path, default=None)
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    out_dir = paths.work_dir / "c14_step_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()

    with provenance("step_plots", out_dir, name="c14") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        tracks_path = args.npz or (
            paths.work_dir / "c03_track_3d" / RUN / f"{METHOD}__{EXPERIMENT}.npz")
        record.input("tracks", tracks_path)
        npz = np.load(tracks_path, allow_pickle=False)
        validation = json.loads(
            (paths.work_dir / "c09_gt_swap_validation" / "c09_validation.json").read_text(
                encoding="utf-8"))
        swaps = [r for r in validation["intervals"] if r["outcome"] == "swap"]
        strata = json.loads(
            (paths.work_dir / "c08_crossing_strata" / "c08_strata.json").read_text(
                encoding="utf-8"))
        encounters = [e for e in strata["encounters"] if e["experiment"] == EXPERIMENT]
        print(f"tracks : {tracks_path}")
        print(f"swaps  : {len(swaps)}")

        series = {}
        for view in VIEWS:
            key = f"labels.round8.{'top' if view == 'top_cam' else 'side'}.train"
            median_span_px(catalog.get(key))
            detections = c01.load(paths.work_dir, EXPERIMENT, view, RUN)
            series[view] = per_view_series(detections, npz[f"track__{view}"], args.window)
            print(f"  {view:<10} {len(series[view]):>6,} tracks with a step series")

        manifest = []
        for item in swaps:
            inside = [e for e in encounters
                      if e["exit_frame"] >= item["t1"] and e["entry_frame"] <= item["t2"]]
            stem = f"swap_f{item['t1']}_to_f{item['t2']}"
            title = (f"measured female swap  ·  frames {item['t1']:,}-{item['t2']:,}"
                     f"  ({item['gap'] / FPS:.1f} s)  ·  {len(inside)} crossing(s)"
                     f"  ·  female track {item['track_1']} -> {item['track_2']}")
            flagged = figure(item, series, inside, out_dir / f"{stem}.png", title, args)
            manifest.append({**item, "encounters_inside": len(inside),
                             "flagged_points": flagged, "file": stem})
            print(f"  {stem}   {len(inside)} crossings, {flagged} flagged points")

        record.param("plots", len(manifest))
        (out_dir / "c14_step_plots.json").write_text(
            json.dumps({"experiment": EXPERIMENT, "n_groups": 1,
                        "groups": [ExperimentId.parse(EXPERIMENT).group],
                        "tracks": str(tracks_path), "swaps": manifest},
                       indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

"""C11 — the scenes where a swap may have happened, one crossing at a time.

`c09` found that every one of the 8 measured female swaps fell in an interval
where the geometric reference chain moved her to a different column — 17 such
intervals out of 296. `c10` then showed those intervals cannot be split by the
existing annotation: they hold several crossings each, and the human anchors sit
1.6-2.0 s outside, so the swap is real but unlocalised.

This tool renders **every crossing inside those intervals**, separately, so the
experimenter can look and say which one it was. It also emits the annotation
worklist for the frames that would settle it without looking again.

=============================================================================
WHAT IS DRAWN, AND WHY THOSE PANELS
=============================================================================

Per crossing, three panels sharing one time axis:

1. **The three pairwise inter-fish distances in mm.** The crossing is shaded.
   Which pair actually closed is the first thing to establish.
2. **The three animals' speeds.** This is the experimenter's heuristic made
   visible: an identity exchange teleports two trajectories at one frame, so it
   shows as a *step* in two speed traces at once, not as a smooth excursion. A
   genuine fast turn shows in one trace and is smooth on the scale of a frame.
3. **The three track-ID strips**, one per canonical column. A swap is two strips
   exchanging colour; a break is a strip changing to a colour no one else has.

Vertical rules mark the human female annotations, labelled with the column she
occupied there. **Between two rules the female's identity is known at the ends
and unknown in between** — which is exactly the judgement being asked for.

Selection is not a sample. **Every** crossing inside the 17 flagged intervals is
drawn; the count is printed. Intervals containing a measured swap are rendered
first and are the only ones given clips by default, because those are the ones
known to contain an error.

Nothing here decides anything. `c09`'s indicator was found after the fact and is
still a hypothesis; these scenes are how it gets tested against a person.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c11_swap_review_pack.py --no-clips
    .venv\\Scripts\\python.exe tools/c11_swap_review_pack.py
"""

from __future__ import annotations

import argparse
import csv
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
from c07_render_tracking import INK, OTHER, PALETTE, PAPER, ViewData, colour_map
from c08_crossing_strata import canonical, row, tiles_at
from s10_shared_node_20k import median_span_px, tail_jump_threshold_px

from medaka3d import load_catalog, load_paths
from medaka3d.calib import load_rig
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.track import UNTRACKED
from medaka3d.types import FPS, VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN, METHOD = "bottomup", "gated"
PAIRS = ((0, 1), (0, 2), (1, 2))
CONTEXT = 166
CLIP_FPS, CLIP_MAX = 30, 900
#: Frames of moving average on the follow-crop centre. 25 = 0.15 s.
SMOOTH_FRAMES = 25


def follow_boxes(data, view, participants, window, shape, span_px):
    """A crop that **follows the two animals**, one box per frame.

    A crop fixed for the whole clip is wrong here for a simple reason: the clip
    spans several seconds and the fish swim out of it, which is what the first
    version did and why most of its frames showed empty water.

    Centre is the midpoint of the two participants in this view, gaps filled by
    interpolation and then smoothed over ``SMOOTH_FRAMES`` so the view pans
    rather than jitters. **The side is constant within a clip** — chosen once
    from the pair's 90th-percentile separation, floored at 6 body lengths — so
    two moments inside one clip stay at the same scale even though the box
    moves. It is printed in the caption, because a crop that changes between
    clips is only honest if the scale is stated.
    """
    points = data[view][window[0]:window[-1] + 1, list(participants), :]
    centre = np.nanmean(points, axis=1)
    frames = np.arange(centre.shape[0])
    for axis in (0, 1):
        finite = np.flatnonzero(np.isfinite(centre[:, axis]))
        if finite.size == 0:
            centre[:, axis] = shape[1 - axis] / 2
        else:
            centre[:, axis] = np.interp(frames, finite, centre[finite, axis])
    kernel = np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES
    padded = np.pad(centre, ((SMOOTH_FRAMES // 2, SMOOTH_FRAMES // 2), (0, 0)), mode="edge")
    centre = np.stack([np.convolve(padded[:, a], kernel, mode="valid")[:len(frames)]
                       for a in (0, 1)], axis=-1)

    separation = np.linalg.norm(points[:, 0, :] - points[:, 1, :], axis=1)
    finite = separation[np.isfinite(separation)]
    wanted = (np.percentile(finite, 90) + 2 * span_px) if finite.size else 6 * span_px
    side = int(min(max(wanted, 6 * span_px), shape[0], shape[1]))
    x0 = np.clip(centre[:, 0] - side / 2, 0, shape[1] - side).astype(int)
    y0 = np.clip(centre[:, 1] - side / 2, 0, shape[0] - side).astype(int)
    return x0, y0, side


def crossing_figure(item, data, anchors, path, title):
    """Pairwise IFD, per-animal speed, and the three ID strips over one crossing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # The window must reach BOTH bracketing anchors, not just the crossing.
    # The first version showed the crossing +/- 1 s and the ID change fell
    # outside it, which made every scene look clean whether it was or not.
    lo = max(0, min(item["entry_frame"] - CONTEXT, item["anchor_before"] - 20))
    hi = min(data["xyz"].shape[0],
             max(item["exit_frame"] + CONTEXT, item["anchor_after"] + 20))
    seconds = (np.arange(lo, hi) - item["entry_frame"]) / FPS

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1], "hspace": 0.18})
    fig.patch.set_facecolor(PAPER)
    for ax in axes:
        ax.set_facecolor(PAPER)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e6e5e2", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axvspan(0.0, (item["exit_frame"] - item["entry_frame"]) / FPS,
                   color="#f0efec", zorder=0)

    for i, (a, b) in enumerate(PAIRS):
        distance = np.linalg.norm(data["xyz"][lo:hi, a, :] - data["xyz"][lo:hi, b, :], axis=1)
        participant = sorted(item["participants"]) == sorted((a, b))
        axes[0].plot(seconds, distance, color=PALETTE[i],
                     linewidth=2.2 if participant else 1.2,
                     alpha=1.0 if participant else 0.5)
        finite = np.flatnonzero(np.isfinite(distance))
        if finite.size:
            axes[0].annotate(f" {a}-{b}" + ("  ✓this crossing" if participant else ""),
                             (seconds[finite[-1]], distance[finite[-1]]), fontsize=8.5,
                             color=PALETTE[i] if participant else "#52514e", va="center")
    axes[0].axhline(20.0, color="#8a8a85", linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("inter-fish distance (mm)")
    axes[0].set_ylim(bottom=0)

    for column in range(3):
        track = data["xyz"][lo:hi, column, :]
        speed = np.concatenate(([np.nan], np.linalg.norm(np.diff(track, axis=0), axis=1) * FPS))
        axes[1].plot(seconds, speed, color=PALETTE[column], linewidth=1.3, alpha=0.9)
    axes[1].set_ylabel("speed (mm/s)")
    axes[1].legend(handles=[plt.Line2D([], [], color=PALETTE[c], linewidth=2.0,
                                       label=f"column {c}") for c in range(3)],
                   loc="upper right", frameon=False, fontsize=8.5, ncols=3)

    strip = data["track"][lo:hi]
    counts = Counter(int(t) for t in strip.ravel() if int(t) != UNTRACKED)
    hue = {t: (PALETTE[i] if i < len(PALETTE) else OTHER)
           for i, (t, _) in enumerate(counts.most_common())}
    for column in range(3):
        ids = strip[:, column]
        edges = np.flatnonzero(np.diff(ids)) + 1
        for start, stop in zip(np.r_[0, edges], np.r_[edges, len(ids)], strict=True):
            value = int(ids[start])
            right = seconds[min(stop, len(ids) - 1)]
            axes[2].add_patch(Rectangle(
                (seconds[start], 0.72 - column * 0.32), right - seconds[start], 0.26,
                color="#dcdbd7" if value == UNTRACKED else hue.get(value, OTHER),
                linewidth=0))
    axes[2].set_ylim(0, 1.05)
    axes[2].set_yticks([0.85, 0.53, 0.21], ["column 0", "column 1", "column 2"])
    axes[2].grid(False)
    axes[2].set_xlabel("seconds from crossing entry")

    for anchor in anchors:
        x = (anchor["frame"] - item["entry_frame"]) / FPS
        if seconds[0] <= x <= seconds[-1]:
            for ax in axes:
                ax.axvline(x, color=INK, linewidth=1.2, alpha=0.6)
            axes[0].annotate(f"female = column {anchor['column']}", (x, axes[0].get_ylim()[1]),
                             fontsize=8.5, rotation=90, va="top", ha="right", color=INK)

    fig.suptitle(title, x=0.012, ha="left", fontsize=12)
    fig.text(0.012, 0.005,
             "Shaded band is the crossing. A swap is two ID strips exchanging colour and a "
             "one-frame STEP in two speed traces at once; a fast turn is smooth. Strip colours "
             "rank this window's longest tracks and are not a correspondence with other figures.",
             fontsize=8, color="#52514e")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--no-clips", action="store_true")
    parser.add_argument("--clips-for", choices=("swap", "all", "none"), default="swap")
    parser.add_argument("--max-frames", type=int, default=CLIP_MAX,
                        help="Cap on clip length. A clip that does not reach the far "
                             "anchor is reported, because a scene that stops short of "
                             "the anchor cannot show where the identity changed.")
    parser.add_argument("--npz", type=Path, default=None,
                        help="Tracking output to render. Defaults to c03's. The track "
                             "IDs drawn on the video come from here.")
    parser.add_argument("--validation", type=Path, default=None,
                        help="c09 output to read the anchors and outcomes from. Must "
                             "have been produced against the same --npz.")
    parser.add_argument("--tag", default=None,
                        help="Subdirectory for the output, so rendering another arm "
                             "does not clear the baseline's figures.")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    rig = load_rig(CALIBRATION)
    out_dir = paths.work_dir / "c11_swap_review_pack"
    if args.tag:
        out_dir = out_dir / args.tag
    tracks_npz = args.npz or (
        paths.work_dir / "c03_track_3d" / RUN / f"{METHOD}__{EXPERIMENT}.npz")
    validation_path = args.validation or (
        paths.work_dir / "c09_gt_swap_validation" / "c09_validation.json")
    for sub in ("crossings", "montage", "clips"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    removed = 0
    for sub in ("crossings", "montage", "clips"):
        for stale in (out_dir / sub).iterdir():
            if stale.is_file() and stale.suffix in (".png", ".mp4"):
                stale.unlink()
                removed += 1
    if removed:
        print(f"cleared   : {removed} files from a previous run")

    with provenance("swap_review_pack", out_dir, name="c11") as record:
        record.params_from(args)
        record.param("n_groups", 1)

        record.input("tracks", tracks_npz)
        record.input("validation", validation_path)
        print(f"tracks     : {tracks_npz}")
        print(f"validation : {validation_path}")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        strata = json.loads(
            (paths.work_dir / "c08_crossing_strata" / "c08_strata.json").read_text(
                encoding="utf-8"))
        encounters = [e for e in strata["encounters"] if e["experiment"] == EXPERIMENT]
        anchors = validation["anchors"]

        # Every measured swap, plus every interval the chain moved her across.
        # `c09`'s chain indicator caught all 8 of `c03`'s swaps, so for the
        # baseline this union is the same 17 intervals it always was; it is a
        # union because another arm's swap need not be chain-flagged, and a swap
        # that is not drawn is a swap that cannot be checked.
        flagged = [r for r in validation["intervals"]
                   if r["column_1"] != r["column_2"] or r["outcome"] == "swap"]
        flagged.sort(key=lambda r: (r["outcome"] != "swap", r["t1"]))
        print(f"flagged intervals : {len(flagged)}   "
              f"of which measured swaps: {sum(r['outcome'] == 'swap' for r in flagged)}")
        record.param("flagged_intervals", len(flagged))

        span = {}
        for view in VIEWS:
            key = f"labels.round8.{'top' if view == 'top_cam' else 'side'}.train"
            span[view], _ = median_span_px(catalog.get(key))
            tail_jump_threshold_px(span[view])

        npz = np.load(tracks_npz, allow_pickle=False)
        chain = reference_chain(npz["xyz"].astype(float), npz["slot_track"].max(axis=1) >= 0)
        data = canonical(npz, chain, rig.views)

        scenes = []
        for interval in flagged:
            inside = [e for e in encounters
                      if e["exit_frame"] >= interval["t1"] and e["entry_frame"] <= interval["t2"]]
            for encounter in sorted(inside, key=lambda e: e["entry_frame"]):
                scenes.append({
                    "interval_t1": interval["t1"], "interval_t2": interval["t2"],
                    "interval_outcome": interval["outcome"],
                    "track_before": interval["track_1"], "track_after": interval["track_2"],
                    "female_column_before": interval["column_1"],
                    "female_column_after": interval["column_2"],
                    "anchor_before": interval["t1"], "anchor_after": interval["t2"],
                    **{k: encounter[k] for k in
                       ("entry_frame", "exit_frame", "participants", "stratum",
                        "crossed_views", "min_distance_mm", "outcome", "id_changes")},
                })
        print(f"crossings to draw : {len(scenes)}  "
              f"({sum(s['interval_outcome'] == 'swap' for s in scenes)} inside swap intervals)")
        record.param("scenes", len(scenes))

        views = {v: ViewData(paths.work_dir, RUN, EXPERIMENT, v, tracks_npz)
                 for v in VIEWS}
        import sleap_io

        videos = {v: sleap_io.Video.from_filename(str(paths.video(EXPERIMENT, v)))
                  for v in VIEWS}
        shape = videos[VIEWS[0]].shape[1:3]

        worklist = []
        for scene in scenes:
            swap = scene["interval_outcome"] == "swap"
            # Two encounters can share an entry frame with different participants,
            # and without the pair in the name the second figure silently
            # overwrote the first.
            stem = (f"{'SWAP' if swap else 'flag'}_i{scene['interval_t1']}"
                    f"_x{scene['entry_frame']}"
                    f"_p{''.join(str(c) for c in scene['participants'])}")
            crossed = ", ".join(v.replace("_cam", "") for v in scene["crossed_views"]) or "none"
            title = (f"{'inside a MEASURED SWAP interval' if swap else 'flagged interval'}"
                     f"  ·  crossing {scene['entry_frame']:,}-{scene['exit_frame']:,}"
                     f"  ·  columns {tuple(scene['participants'])}"
                     f"  ·  {scene['stratum']} view(s) [{crossed}]"
                     f"  ·  closest {scene['min_distance_mm']:.1f} mm"
                     f"  ·  anchors {scene['anchor_before']:,} / {scene['anchor_after']:,}"
                     f"  ·  track {scene['track_before']} -> {scene['track_after']}")
            crossing_figure(scene, data, anchors, out_dir / "crossings" / f"{stem}.png", title)

            # The clip must reach both anchors for the same reason the figure
            # must: the ID change can sit outside the crossing itself.
            start = max(0, min(scene["entry_frame"] - CONTEXT, scene["anchor_before"]))
            window = np.arange(start, min(c01.N_FRAMES,
                                          max(scene["exit_frame"] + CONTEXT,
                                              scene["anchor_after"] + 1),
                                          start + args.max_frames))
            if window[-1] < scene["anchor_after"]:
                print(f"    !! clip stops at {window[-1]:,}, short of the far anchor "
                      f"{scene['anchor_after']:,} — raise --max-frames to see the join")
            colours = colour_map({v: views[v] for v in VIEWS}, window[::5])
            middle = int((scene["entry_frame"] + scene["exit_frame"]) // 2)
            follow = {v: follow_boxes(data, v, scene["participants"], window, shape, span[v])
                      for v in VIEWS}

            def boxes_at(frame, follow=follow, window=window):
                index = int(np.clip(frame - window[0], 0, len(window) - 1))
                return {v: (int(follow[v][0][index]), int(follow[v][1][index]),
                            follow[v][2], follow[v][2]) for v in VIEWS}

            sides = "  ".join(f"{v.replace('_cam', '')} {follow[v][2] / span[v]:.1f}"
                              for v in VIEWS)
            note = (title + f"  |  crop follows the pair; side in body lengths: {sides}"
                    "; labels are track IDs")
            still = row(tiles_at(middle, videos, views, boxes_at(middle), colours), note)
            cv2.imwrite(str(out_dir / "montage" / f"{stem}.png"), still)

            want_clip = (args.clips_for == "all") or (args.clips_for == "swap" and swap)
            if want_clip and not args.no_clips:
                writer = cv2.VideoWriter(str(out_dir / "clips" / f"{stem}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS,
                                         (still.shape[1], still.shape[0]))
                for f in window:
                    writer.write(row(tiles_at(int(f), videos, views, boxes_at(int(f)),
                                              colours), note))
                writer.release()

            # The two frames that would settle this crossing without looking again.
            for side, frame in (("before", scene["entry_frame"] - 1),
                                ("after", scene["exit_frame"] + 1)):
                worklist.append({
                    "experiment": EXPERIMENT, "frame": int(frame), "side": side,
                    "crossing_entry": scene["entry_frame"], "crossing_exit": scene["exit_frame"],
                    "stratum": scene["stratum"], "participants": str(scene["participants"]),
                    "interval_outcome": scene["interval_outcome"],
                    "min_distance_mm": round(scene["min_distance_mm"], 2),
                    "scene_files": stem,
                })
            print(f"  {stem}")

        fields = list(worklist[0]) if worklist else []
        with (out_dir / "c11_annotation_worklist.csv").open("w", newline="",
                                                            encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(worklist)
        record.param("worklist_rows", len(worklist))

        (out_dir / "c11_scenes.json").write_text(
            json.dumps({"experiment": EXPERIMENT, "n_groups": 1,
                        "groups": [ExperimentId.parse(EXPERIMENT).group],
                        "flagged_intervals": len(flagged), "scenes": scenes},
                       indent=2, default=float), encoding="utf-8")
        print(f"\nworklist: {len(worklist)} frames to annotate "
              f"({len(worklist) // 2} crossings x before/after)")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

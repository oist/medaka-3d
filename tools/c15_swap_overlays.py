"""C15 — the eight measured swaps, with the SLEAP poses and track IDs drawn thin.

`c14` located candidate frames numerically: a one-frame displacement that is
10-600x its own local median **and** equal to the distance to the nearest other
instance, appearing on two tracks at once. Six of the eight intervals have such
frames. This renders them, so the numbers can be checked against pixels.

=============================================================================
DRAWING
=============================================================================

**Thin marks throughout.** `c07`'s convention was built for a 460 px tile seen
at a glance and its 2 px strokes with 4 px haloes cover the animal at these crop
scales — the mark then hides the thing being judged. Here every stroke is 1 px
over a 2 px halo, the mouth is a small ring, the tail a small square, the
pectoral a dot, and the track ID is drawn at a small size beside the mouth. The
halo stays because the background is video and its brightness is not under our
control.

Identity is never colour alone: the **track ID is written on every instance**.
Colours rank the window's longest tracks and fold everything past the third into
one neutral, so a fourth hue never implies a fourth animal.

**The crop follows the interacting pair** — the two nearest instances in that
frame — with the centre interpolated across gaps and smoothed, and a side that
is constant within a clip so two moments stay at one scale. The side is printed
in body lengths on the caption.

Two outputs per interval:

* ``stills/`` — one PNG per candidate frame `c14` flagged, at that exact frame,
  three cameras across. Flagged instances are ringed.
* ``clips/`` — one MP4 over the interval, capped at ``--max-frames`` and the cap
  reported, at 30 fps from 166 fps source.

**n_groups = 1, n_experiments = 1, female only.**

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c15_swap_overlays.py --no-clips
    .venv\\Scripts\\python.exe tools/c15_swap_overlays.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from c07_render_tracking import (
    INK,
    OTHER,
    PALETTE,
    PAPER,
    TILE,
    ViewData,
    _wrap,
    bgr,
    caption,
    frame_bgr,
)
from c14_step_plots import per_view_series
from s10_shared_node_20k import median_span_px

from medaka3d import load_catalog, load_paths
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.track import UNTRACKED
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN, METHOD = "bottomup", "gated"
ALERT = "#e34948"
CLIP_FPS = 30
SMOOTH_FRAMES = 25


def draw_thin(image, points, tracks, colours, box, scale, flagged: set[int]):
    """One camera's instances, drawn with 1 px strokes over a 2 px halo."""
    x0, y0, w, h = box
    tile = cv2.resize(image[y0:y0 + h, x0:x0 + w], (TILE, TILE),
                      interpolation=cv2.INTER_AREA)
    halo = bgr(INK)

    def to_tile(xy):
        return (int((xy[0] - x0) * scale), int((xy[1] - y0) * scale))

    for index in range(points.shape[0]):
        track = int(tracks[index]) if index < len(tracks) else UNTRACKED
        colour = bgr(colours.get(track, OTHER))
        p = points[index]
        ok = np.isfinite(p).all(axis=-1)
        chain = [to_tile(p[i]) for i in range(3) if ok[i]]
        for u, v in pairwise(chain):
            cv2.line(tile, u, v, halo, 2, cv2.LINE_AA)
            cv2.line(tile, u, v, colour, 1, cv2.LINE_AA)
        if ok[1]:
            cv2.circle(tile, to_tile(p[1]), 2, colour, -1, cv2.LINE_AA)
        if ok[0]:
            centre = to_tile(p[0])
            cv2.circle(tile, centre, 5, halo, 2, cv2.LINE_AA)
            cv2.circle(tile, centre, 5, colour, 1, cv2.LINE_AA)
        if ok[2]:
            c = to_tile(p[2])
            cv2.rectangle(tile, (c[0] - 4, c[1] - 4), (c[0] + 4, c[1] + 4), halo, 2, cv2.LINE_AA)
            cv2.rectangle(tile, (c[0] - 4, c[1] - 4), (c[0] + 4, c[1] + 4), colour, 1,
                          cv2.LINE_AA)
        if track in flagged and ok[0]:
            cv2.circle(tile, to_tile(p[0]), 12, bgr(ALERT), 1, cv2.LINE_AA)
        if ok[0] and track != UNTRACKED:
            centre = to_tile(p[0])
            for thickness, ink in ((2, halo), (1, colour)):
                cv2.putText(tile, str(track), (centre[0] + 8, centre[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, ink, thickness, cv2.LINE_AA)
    return tile


def pair_boxes(view_data, window, shape, span_px):
    """Follow the two nearest instances, one box per frame, constant side."""
    centres = np.full((window.size, 2), np.nan)
    gaps = np.full(window.size, np.nan)
    for i, frame in enumerate(window):
        points, _ = view_data.at(int(frame))
        mouths = points[:, 0, :]
        finite = mouths[np.isfinite(mouths).all(axis=1)]
        if finite.shape[0] < 2:
            if finite.shape[0] == 1:
                centres[i] = finite[0]
            continue
        distance = np.linalg.norm(finite[:, None, :] - finite[None, :, :], axis=-1)
        np.fill_diagonal(distance, np.inf)
        a, b = np.unravel_index(np.argmin(distance), distance.shape)
        centres[i] = (finite[a] + finite[b]) / 2.0
        gaps[i] = distance[a, b]
    frames = np.arange(window.size)
    for axis in (0, 1):
        good = np.flatnonzero(np.isfinite(centres[:, axis]))
        if good.size == 0:
            centres[:, axis] = shape[1 - axis] / 2
        else:
            centres[:, axis] = np.interp(frames, good, centres[good, axis])
    kernel = np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES
    padded = np.pad(centres, ((SMOOTH_FRAMES // 2, SMOOTH_FRAMES // 2), (0, 0)), mode="edge")
    centres = np.stack([np.convolve(padded[:, a], kernel, mode="valid")[:window.size]
                        for a in (0, 1)], axis=-1)
    finite = gaps[np.isfinite(gaps)]
    wanted = (np.percentile(finite, 90) + 2 * span_px) if finite.size else 5 * span_px
    side = int(min(max(wanted, 5 * span_px), shape[0], shape[1]))
    x0 = np.clip(centres[:, 0] - side / 2, 0, shape[1] - side).astype(int)
    y0 = np.clip(centres[:, 1] - side / 2, 0, shape[0] - side).astype(int)
    return x0, y0, side


def grid(tiles, note):
    body = np.hstack([caption(tiles[v], f"{RUN}  ·  {v}") for v in VIEWS])
    lines = _wrap(note, body.shape[1] - 20, 0.52)
    header = np.full((14 + 22 * len(lines), body.shape[1], 3), bgr(PAPER), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(header, line, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, bgr(INK), 1, cv2.LINE_AA)
    return np.vstack([header, body])


def flags_for(series, lo, hi, excess, band):
    """`c14`'s detector, re-evaluated here so the stills and the numbers agree."""
    out = []
    for view in VIEWS:
        for identifier, data in series[view].items():
            inside = (data["frame"] >= lo) & (data["frame"] <= hi)
            if inside.sum() < 2:
                continue
            step, near = data["step"][inside], data["nearest"][inside]
            frames = data["frame"][inside]
            with np.errstate(all="ignore"):
                ratio = step / np.where(near > 0, near, np.nan)
                relative = step / np.maximum(data["baseline"][inside], 1e-6)
            hit = ((relative >= excess) & (ratio >= band[0]) & (ratio <= band[1]))
            hit = np.where(np.isfinite(hit), hit, False).astype(bool)
            for index in np.flatnonzero(hit):
                out.append({"frame": int(frames[index]), "view": view,
                            "track": int(identifier), "step_px": float(step[index]),
                            "nearest_px": float(near[index]),
                            "ratio": float(ratio[index])})
    out.sort(key=lambda r: (r["frame"], r["view"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--excess", type=float, default=10.0)
    parser.add_argument("--ratio-band", type=float, nargs=2, default=(0.7, 1.3))
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--no-clips", action="store_true")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    out_dir = paths.work_dir / "c15_swap_overlays"
    for sub in ("stills", "clips"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    removed = 0
    for sub in ("stills", "clips"):
        for stale in (out_dir / sub).iterdir():
            if stale.is_file() and stale.suffix in (".png", ".mp4"):
                stale.unlink()
                removed += 1
    if removed:
        print(f"cleared : {removed} files from a previous run")

    with provenance("swap_overlays", out_dir, name="c15") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        npz = np.load(paths.work_dir / "c03_track_3d" / RUN / f"{METHOD}__{EXPERIMENT}.npz",
                      allow_pickle=False)
        validation = json.loads(
            (paths.work_dir / "c09_gt_swap_validation" / "c09_validation.json").read_text(
                encoding="utf-8"))
        swaps = [r for r in validation["intervals"] if r["outcome"] == "swap"]
        print(f"swaps : {len(swaps)}")

        span, series, views = {}, {}, {}
        for view in VIEWS:
            key = f"labels.round8.{'top' if view == 'top_cam' else 'side'}.train"
            span[view], _ = median_span_px(catalog.get(key))
            detections = c01.load(paths.work_dir, EXPERIMENT, view, RUN)
            series[view] = per_view_series(detections, npz[f"track__{view}"], args.window)
            views[view] = ViewData(paths.work_dir, RUN, EXPERIMENT, view)

        import sleap_io

        videos = {v: sleap_io.Video.from_filename(str(paths.video(EXPERIMENT, v)))
                  for v in VIEWS}
        shape = videos[VIEWS[0]].shape[1:3]

        manifest = []
        for item in swaps:
            lo, hi = item["t1"], item["t2"]
            hits = flags_for(series, lo, hi, args.excess, tuple(args.ratio_band))
            window = np.arange(lo, min(hi + 1, lo + args.max_frames))
            truncated = max(0, (hi + 1 - lo) - window.size)
            boxes = {v: pair_boxes(views[v], window, shape, span[v]) for v in VIEWS}

            counts: Counter = Counter()
            for view in VIEWS:
                for frame in window[::5]:
                    _, tracks = views[view].at(int(frame))
                    counts.update(int(t) for t in tracks if int(t) != UNTRACKED)
            colours = {t: (PALETTE[i] if i < len(PALETTE) else OTHER)
                       for i, (t, _) in enumerate(counts.most_common())}

            def tiles_at(frame, flagged, boxes=boxes, colours=colours, window=window):
                index = int(np.clip(frame - window[0], 0, window.size - 1))
                out = {}
                for view in VIEWS:
                    image = frame_bgr(videos[view], frame)
                    x0, y0, side = boxes[view]
                    box = (int(x0[index]), int(y0[index]), side, side)
                    points, tracks = views[view].at(int(frame))
                    out[view] = draw_thin(image, points, tracks, colours, box,
                                          TILE / side, flagged.get(view, set()))
                return out

            sides = "  ".join(f"{v.replace('_cam', '')} {boxes[v][2] / span[v]:.1f}"
                              for v in VIEWS)
            head = (f"measured female swap  ·  frames {lo:,}-{hi:,}  ·  female track "
                    f"{item['track_1']} -> {item['track_2']}  ·  {item['encounters']} "
                    f"crossing(s)  ·  {len(hits)} flagged points")
            note = head + (f"  |  crop follows the closest pair; side in body lengths: "
                           f"{sides}; thin marks, labels are track IDs")

            by_frame: dict[int, dict[str, set]] = {}
            for hit in hits:
                by_frame.setdefault(hit["frame"], {}).setdefault(hit["view"], set()).add(
                    hit["track"])
            for frame, flagged in sorted(by_frame.items()):
                detail = "  ".join(
                    f"{h['view'].replace('_cam','')} t{h['track']} {h['step_px']:.0f}px "
                    f"nn {h['nearest_px']:.0f}px r{h['ratio']:.2f}"
                    for h in hits if h["frame"] == frame)
                still = grid(tiles_at(frame, flagged), f"{head}  ·  FRAME {frame:,}  ·  {detail}")
                cv2.imwrite(str(out_dir / "stills" / f"swap_f{lo}_flag_f{frame}.png"), still)

            stem = f"swap_f{lo}_to_f{hi}"
            if not by_frame:
                middle = int((lo + hi) // 2)
                still = grid(tiles_at(middle, {}), f"{head}  ·  no flagged frame")
                cv2.imwrite(str(out_dir / "stills" / f"{stem}_mid.png"), still)
            if not args.no_clips:
                sample = grid(tiles_at(int(window[0]), {}), note)
                writer = cv2.VideoWriter(str(out_dir / "clips" / f"{stem}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS,
                                         (sample.shape[1], sample.shape[0]))
                for frame in window:
                    writer.write(grid(tiles_at(int(frame), by_frame.get(int(frame), {})), note))
                writer.release()
            manifest.append({**item, "flags": hits, "clip_truncated_frames": int(truncated),
                             "files": stem})
            print(f"  {stem}   {len(hits)} flags, {len(by_frame)} stills"
                  + (f", clip truncated {truncated}" if truncated else ""))

        record.param("rendered", len(manifest))
        (out_dir / "c15_swaps.json").write_text(
            json.dumps({"experiment": EXPERIMENT, "n_groups": 1,
                        "groups": [ExperimentId.parse(EXPERIMENT).group],
                        "swaps": manifest}, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

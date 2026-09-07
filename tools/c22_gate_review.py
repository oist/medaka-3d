"""C22 — the links the gate refused, rendered so they can be judged by eye.

=============================================================================
WHAT AN EVENT IS
=============================================================================

`max_step_mm` bounds **one** quantity: the 3D distance in mm between an open
track's last known pectoral position and the candidate pectoral position on the
current frame. The per-track ceiling is ``max_step_mm x frames since last seen``,
so a track absent two frames is allowed twice the distance.

When the nearest continuation lies beyond that ceiling, `gated` refuses the link
and opens a new identity. **That refusal is what this tool renders.** An event
here is:

* an identity **A** holding at least ``--min-seconds`` ends at frame *t*,
* a different identity **B**, also holding at least ``--min-seconds``, starts at
  *t + lag* for lag in 1-3,
* B starts within ``--max-mm`` of where A ended, and **beyond A's ceiling at
  that lag** — so the gate, not the absence of a fish, is what split them.

Measured on `c21`'s `gated-partial-unbounded` at a 4.0 mm gate, the 128 such
events fall out by how large the step was, which is the same as asking which
gate would still have cut them:

===============  =======  =====================================================
distance         events   read as
===============  =======  =====================================================
>= 12 mm               6  cut even by a 12 mm gate
8 - 12 mm              8  cut by 8 mm, survived by 12 mm
4 - 8 mm             113  cut by 4 mm, survived by 8 mm
===============  =======  =====================================================

120 of the 128 are at **lag 1** — no frame missing at all, the animal simply
moved further in one frame (6.02 ms) than the gate allows. At 4.0 mm that is
0.66 m/s, and the median event is 4.50 mm. So these are darts being refused, not
short dropouts: a 1-3 frame gap is bridged 1,788 times inside these same
identities.

**The question a reviewer answers here is one thing only: at this moment, is the
new identity the same fish as the old one?** If yes, the gate is too tight and
the link should have been made. If the two identities are different fish, the
refusal was correct and a looser gate would have produced a swap.

=============================================================================
DRAWING
=============================================================================

`c15`'s convention, unchanged: 1 px strokes over a 2 px halo, the mouth a ring,
the tail a square, the pectoral a dot, and **the track ID written on every
instance** so identity is never colour alone. A is blue, B is orange, every
other animal neutral grey; both are ringed in red on the frames they appear.

The crop follows A then B, at a side constant within one event so the two
moments are at one scale, and `frame_bgr` is used so the channels are not
exchanged — the experimenter's own fish descriptors are colours.

Two outputs per event, named so they sort worst-first:

``stills/``  one PNG per event: three cameras across, ``--montage`` frames
             spanning the refusal, the transition marked in the caption.
``clips/``   one MP4 over a wider window at 30 fps from 166 fps source, so the
             motion can be seen rather than inferred.

**n_groups = 1, n_experiments = 1.** This is a viewing aid, not a measurement.
Nothing here scores anything and no threshold is chosen from it.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c22_gate_review.py --min-mm 12
    .venv\\Scripts\\python.exe tools/c22_gate_review.py --min-mm 8
    .venv\\Scripts\\python.exe tools/c22_gate_review.py --min-mm 4 --no-clips
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c07_render_tracking import INK, PALETTE, PAPER, TILE, ViewData, bgr, frame_bgr
from c15_swap_overlays import CLIP_FPS, SMOOTH_FRAMES, draw_thin, grid
from s10_shared_node_20k import median_span_px

from medaka3d import load_catalog, load_paths
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.types import FPS, VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN = "bottomup"
ARM = Path("c21_link_refusal") / f"gated-partial-unbounded__{EXPERIMENT}.npz"

#: The gate the arm was tracked with. Needed to say whether a step was *refused*
#: rather than merely large, so it is a parameter and not a guess.
GATE_MM = 4.0

#: Bands the events are filed under: the largest gate that would still cut them.
BANDS = (12.0, 8.0, 4.0)


def identities(slot_track: np.ndarray, xyz: np.ndarray, min_frames: int) -> list[dict]:
    """Every track ID holding at least ``min_frames``, with its ends in 3D."""
    held = slot_track >= 0
    frames, slots = np.nonzero(held)
    ids = slot_track[held]
    order = np.lexsort((frames, ids))
    ids, frames, slots = ids[order], frames[order], slots[order]
    edges = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, ids.size]
    out = []
    for a, b in pairwise(edges):
        if b - a < min_frames:
            continue
        out.append({
            "track": int(ids[a]),
            "first": int(frames[a]), "last": int(frames[b - 1]),
            "first_slot": int(slots[a]), "last_slot": int(slots[b - 1]),
            "first_xyz": xyz[frames[a], slots[a]],
            "last_xyz": xyz[frames[b - 1], slots[b - 1]],
            "frames": int(b - a),
        })
    return out


def refusals(records: list[dict], gate_mm: float, max_lag: int, max_mm: float) -> list[dict]:
    """Where one identity ends and another takes over beyond the ceiling."""
    ends: dict[int, list[dict]] = {}
    for r in records:
        ends.setdefault(r["last"], []).append(r)

    out = []
    for b in records:
        best = None
        for lag in range(1, max_lag + 1):
            for a in ends.get(b["first"] - lag, ()):
                if a["track"] == b["track"]:
                    continue
                if not (np.isfinite(a["last_xyz"]).all() and np.isfinite(b["first_xyz"]).all()):
                    continue
                d = float(np.linalg.norm(a["last_xyz"] - b["first_xyz"]))
                if best is None or d < best["mm"]:
                    best = {"mm": d, "lag": lag, "a": a}
        if best is None or best["mm"] > max_mm:
            continue
        ceiling = gate_mm * best["lag"]
        if best["mm"] <= ceiling:
            continue  # the gate allowed this; something else ended the track
        out.append({
            "frame_end": best["a"]["last"], "frame_start": b["first"], "lag": best["lag"],
            "mm": best["mm"], "ceiling_mm": ceiling,
            "mm_per_frame": best["mm"] / best["lag"],
            "speed_m_s": best["mm"] / best["lag"] / 1000.0 * FPS,
            "track_before": best["a"]["track"], "track_after": b["track"],
            "slot_before": best["a"]["last_slot"], "slot_after": b["first_slot"],
            "seconds_before": best["a"]["frames"] / FPS,
            "seconds_after": b["frames"] / FPS,
            "band": next((f"ge{int(x)}mm" for x in BANDS if best["mm"] >= x), "below"),
        })
    out.sort(key=lambda e: -e["mm"])
    return out


#: Frames either side of the refusal over which the crop is pinned to the two
#: identities' midpoint. Without it the crop follows A, then jumps to B, and the
#: transition itself can leave the frame — which is exactly what happened on
#: `ge12mm_024.14mm_f113382` and hid the moment the clip exists to show.
HOLD_FRAMES = 40


def _mouth_of(view_data, frame: int, track: int) -> np.ndarray | None:
    points, tracks = view_data.at(int(frame))
    rows = np.flatnonzero(tracks == track)
    for row in rows:
        if np.isfinite(points[row, 0]).all():
            return points[row, 0]
    return None


def follow(view_data, window, shape, span_px, event) -> tuple[np.ndarray, np.ndarray, int]:
    """Crop boxes that hold BOTH identities in frame across the refusal.

    Away from the refusal the crop follows whichever identity is live. Within
    ``HOLD_FRAMES`` of it the centre is pinned to the midpoint of where A ended
    and where B began, and the side is widened to span them, so the transition
    cannot fall outside the tile.
    """
    a_end = _mouth_of(view_data, event["frame_end"], event["track_before"])
    b_start = _mouth_of(view_data, event["frame_start"], event["track_after"])
    if a_end is not None and b_start is not None:
        midpoint, apart = (a_end + b_start) / 2.0, float(np.linalg.norm(a_end - b_start))
    elif a_end is not None:
        midpoint, apart = a_end, 0.0
    elif b_start is not None:
        midpoint, apart = b_start, 0.0
    else:
        midpoint, apart = np.array([shape[1] / 2, shape[0] / 2]), 0.0

    centres = np.full((window.size, 2), np.nan)
    for i, frame in enumerate(window):
        if abs(int(frame) - event["frame_end"]) <= HOLD_FRAMES:
            centres[i] = midpoint
            continue
        wanted = (event["track_before"] if frame < event["frame_end"]
                  else event["track_after"])
        found = _mouth_of(view_data, int(frame), wanted)
        if found is None:
            found = _mouth_of(view_data, int(frame), event["track_after"]
                              if wanted == event["track_before"] else event["track_before"])
        if found is not None:
            centres[i] = found

    index = np.arange(window.size)
    for axis in (0, 1):
        good = np.flatnonzero(np.isfinite(centres[:, axis]))
        if good.size == 0:
            centres[:, axis] = midpoint[axis]
        else:
            centres[:, axis] = np.interp(index, good, centres[good, axis])
    kernel = np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES
    padded = np.pad(centres, ((SMOOTH_FRAMES // 2, SMOOTH_FRAMES // 2), (0, 0)), mode="edge")
    centres = np.stack([np.convolve(padded[:, a], kernel, mode="valid")[: window.size]
                        for a in (0, 1)], axis=-1)

    side = int(min(max(6.0 * span_px, apart + 3.0 * span_px), shape[0], shape[1]))
    x0 = np.clip(centres[:, 0] - side / 2, 0, shape[1] - side).astype(int)
    y0 = np.clip(centres[:, 1] - side / 2, 0, shape[0] - side).astype(int)
    return x0, y0, side


def montage(views, videos, span, event, frames: np.ndarray, shape) -> np.ndarray:
    """Three cameras across, one column per frame, the refusal in the middle."""
    colours = {event["track_before"]: PALETTE[0], event["track_after"]: PALETTE[1]}
    flagged = {event["track_before"], event["track_after"]}
    rows = []
    for view in VIEWS:
        x0, y0, side = follow(views[view], frames, shape, span[view], event)
        tiles = []
        for i, frame in enumerate(frames):
            points, tracks = views[view].at(int(frame))
            tile = draw_thin(frame_bgr(videos[view], frame), points, tracks, colours,
                             (int(x0[i]), int(y0[i]), side, side), TILE / side, flagged)
            mark = "  <-- refused here" if frame == event["frame_end"] else ""
            band = np.full((26, TILE, 3), bgr(PAPER), np.uint8)
            cv2.putText(band, f"{int(frame):,}{mark}", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr(INK), 1, cv2.LINE_AA)
            tiles.append(np.vstack([band, tile]))
        label = np.full((tiles[0].shape[0], 96, 3), bgr(PAPER), np.uint8)
        cv2.putText(label, view.replace("_cam", ""), (6, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr(INK), 1, cv2.LINE_AA)
        rows.append(np.hstack([label, *tiles]))
    note = (f"gate refusal  ·  {event['mm']:.2f} mm over {event['lag']} frame(s) "
            f"= {event['speed_m_s']:.2f} m/s  ·  ceiling was {event['ceiling_mm']:.1f} mm  ·  "
            f"track {event['track_before']} (blue, {event['seconds_before']:.1f} s) -> "
            f"{event['track_after']} (orange, {event['seconds_after']:.1f} s)  ·  "
            f"same fish? then the gate is too tight; different fish? the refusal was right")
    body = np.vstack(rows)
    header = np.full((30, body.shape[1], 3), bgr(PAPER), np.uint8)
    cv2.putText(header, note[: int(body.shape[1] / 8)], (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr(INK), 1, cv2.LINE_AA)
    return np.vstack([header, body])


def clip(path, views, videos, span, event, frames: np.ndarray, shape) -> None:
    colours = {event["track_before"]: PALETTE[0], event["track_after"]: PALETTE[1]}
    flagged = {event["track_before"], event["track_after"]}
    boxes = {v: follow(views[v], frames, shape, span[v], event) for v in VIEWS}
    writer = None
    for i, frame in enumerate(frames):
        tiles = {}
        for view in VIEWS:
            x0, y0, side = boxes[view]
            points, tracks = views[view].at(int(frame))
            tiles[view] = draw_thin(frame_bgr(videos[view], frame), points, tracks, colours,
                                    (int(x0[i]), int(y0[i]), side, side), TILE / side, flagged)
        note = (f"frame {int(frame):,}   "
                + ("REFUSED HERE   " if frame == event["frame_end"] else "")
                + f"{event['mm']:.2f} mm / {event['lag']} frame(s) = "
                  f"{event['speed_m_s']:.2f} m/s, ceiling {event['ceiling_mm']:.1f} mm   "
                  f"track {event['track_before']} (blue) -> {event['track_after']} (orange)")
        image = grid(tiles, note)
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     CLIP_FPS, (image.shape[1], image.shape[0]))
        writer.write(image)
    if writer is not None:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--arm", type=Path, default=None,
                        help=f"tracking npz to review; default {ARM}")
    parser.add_argument("--gate-mm", type=float, default=GATE_MM,
                        help="the gate the arm was tracked with, for the ceiling")
    parser.add_argument("--min-mm", type=float, default=8.0,
                        help="only events at least this far. 12 gives the handful cut "
                             "even by a 12 mm gate; 4 gives all of them")
    parser.add_argument("--max-mm", type=float, default=60.0,
                        help="beyond this the two identities are not plausibly one fish "
                             "and the frame is a different question")
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--montage", type=int, default=9, help="frames in the still")
    parser.add_argument("--clip-seconds", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--no-clips", action="store_true")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    arm = args.arm or (paths.work_dir / ARM)
    if not arm.is_file():
        raise SystemExit(f"no tracking npz at {arm}")
    out_dir = paths.work_dir / "c22_gate_review"
    (out_dir / "stills").mkdir(parents=True, exist_ok=True)
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)

    with provenance("gate_review", out_dir, name="c22") as record:
        record.params_from(args)
        record.input("arm", arm)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        print(f"arm  {arm}\ngate {args.gate_mm} mm   n_groups = 1, n_experiments = 1\n")

        run = np.load(arm, allow_pickle=False)
        records = identities(run["slot_track"], run["xyz"].astype(float),
                             round(args.min_seconds * FPS))
        events = refusals(records, args.gate_mm, args.max_lag, args.max_mm)
        print(f"identities >= {args.min_seconds} s : {len(records):,}")
        print(f"gate refusals between them  : {len(events):,}")
        for band in BANDS:
            n = sum(1 for e in events if e["mm"] >= band)
            print(f"  at least {band:>4.0f} mm : {n:>4}   "
                  f"(a {band:.0f} mm gate would still cut these)")
        record.param("identities", len(records))
        record.param("refusals", len(events))

        shown = [e for e in events if e["mm"] >= args.min_mm][: args.limit]
        print(f"\nrendering {len(shown)} of {sum(1 for e in events if e['mm'] >= args.min_mm)} "
              f"events at or above {args.min_mm} mm, largest first")
        if not shown:
            record.param("rendered", 0)
            return

        import sleap_io

        views = {v: ViewData(paths.work_dir, RUN, EXPERIMENT, v, arm) for v in VIEWS}
        videos = {v: sleap_io.Video.from_filename(str(paths.video(EXPERIMENT, v)))
                  for v in VIEWS}
        shape = videos[VIEWS[0]].shape[1:3]
        span = {}
        for view in VIEWS:
            key = f"labels.round8.{'top' if view == 'top_cam' else 'side'}.train"
            span[view], _ = median_span_px(catalog.get(key))

        half = args.montage // 2
        clip_half = round(args.clip_seconds * FPS)
        manifest = []
        for event in shown:
            stem = (f"{event['band']}_{event['mm']:06.2f}mm_f{event['frame_end']}"
                    f"_t{event['track_before']}to{event['track_after']}")
            still = np.arange(max(0, event["frame_end"] - half),
                              min(c01_frames(run), event["frame_end"] + half + 1))
            cv2.imwrite(str(out_dir / "stills" / f"{stem}.png"),
                        montage(views, videos, span, event, still, shape))
            if not args.no_clips:
                window = np.arange(max(0, event["frame_end"] - clip_half),
                                   min(c01_frames(run), event["frame_end"] + clip_half + 1))
                clip(out_dir / "clips" / f"{stem}.mp4", views, videos, span, event,
                     window, shape)
            manifest.append({**event, "stem": stem})
            print(f"  {stem}   {event['mm']:6.2f} mm  lag {event['lag']}  "
                  f"{event['speed_m_s']:5.2f} m/s   "
                  f"{event['seconds_before']:6.1f} s -> {event['seconds_after']:5.1f} s")

        record.param("rendered", len(manifest))
        (out_dir / "c22_events.json").write_text(
            json.dumps({"experiment": EXPERIMENT, "n_groups": 1,
                        "groups": [ExperimentId.parse(EXPERIMENT).group],
                        "arm": str(arm), "gate_mm": args.gate_mm,
                        "identities": len(records), "refusals": len(events),
                        "bands": {f"ge{int(b)}mm": sum(1 for e in events if e["mm"] >= b)
                                  for b in BANDS},
                        "rendered": manifest,
                        "all_events": events}, indent=2, default=float),
            encoding="utf-8")

    print(f"\nwrote {out_dir}")
    print("  stills/  three cameras across, the refusal frame captioned")
    print("  clips/   the same window at 30 fps")
    print("\nfor each one, the only question is: are the blue and the orange track the "
          "same fish?\n  same fish  -> the gate is too tight at that speed\n"
          "  different  -> the refusal was correct and a looser gate would have swapped")


def c01_frames(run) -> int:
    return int(run["slot_track"].shape[0])


if __name__ == "__main__":
    main()

"""C07 — render both architectures' tracked output on the same frames, side by side.

`c06` says top-down duplicates a tail 900-2,100x more often than bottom-up and
asserts 11x the impossible track motion. Those are numbers. This draws the
pixels, so the numbers can be checked by eye rather than trusted — the same
reason `k02` exists beside `k01`.

**Both architectures are drawn on the identical video frame.** The source video
is the same file; only the predictions differ, so a difference on screen is a
difference in the model and nothing else.

=============================================================================
WHAT IS RENDERED, AND HOW THE FRAMES ARE CHOSEN
=============================================================================

Selection is **deterministic and stated before rendering** — no randomness, and
no picking the frames that flatter a conclusion. Four categories, each sorted by
a fixed key and sampled at even intervals:

* ``shared_tail`` — top-down places two tails within 5 px and bottom-up does not,
  in the same view and frame. This is `c06`'s headline difference; the sort key
  is the tail-to-tail distance, so the montage shows the clearest cases first.
* ``track_jump`` — one architecture's `gated` track moved further than the
  per-view gate between adjacent frames and the other's did not. This is the
  metric that decided both the tracker and the architecture.
* ``encounter`` — the closest-approach frame of the longest encounters, where
  identity is genuinely at risk for *both* architectures.
* ``routine`` — evenly spaced frames with no anomaly in either architecture.
  **Included so the montage is not only failures**; a reader who sees only
  broken frames cannot judge how often they occur.

Three outputs per selected moment:

1. ``montage/`` — a 2 x 3 still: architectures down, views across, cropped to
   the animals with margin.
2. ``clips/`` — the same grid as an MP4 over a window around the moment, written
   at 30 fps from 166 fps source, so it plays at 1/5.5 speed. Tracking is
   temporal and a still cannot show it.
3. ``continuity/`` — the reconstructed 3D pairwise separations over the window,
   one panel per architecture, with a track-identity band underneath. A break
   or a swap shows as a colour change in the band; an encounter shows as a dip
   in the separation trace.

=============================================================================
COLOUR, AND WHY ONLY THREE HUES
=============================================================================

Track identity is **categorical**, so hues are assigned in fixed order and never
cycled. The three longest tracks in a window take categorical slots 1-3 (blue,
orange, aqua); **everything else folds into one neutral "other"** rather than
generating a fourth hue. That is the documented rule for marks where every pair
is on screen at once, and it is honest here for a second reason: three animals
are present, so a fourth colour would imply a fourth identity that does not
exist. Debris looks like debris.

Those three slots are the ones `references/palette.md` records as clearing the
all-pairs colour-vision floors in both modes. **The validator was not re-run
here** — this machine has no `node` — so that is a documented result being
relied on, not a measurement made today.

Identity is never colour alone: every instance carries its **track ID as a
direct label**, which is also what the light-surface contrast relief rule
requires. Every mark is drawn with a dark halo, because the "surface" is a video
frame rather than a chart background and its brightness is not under our control.

Marks follow `k02`'s convention so the two tools can be read together: **mouth a
circle, tail a square**, joined by the body axis, with the pectoral a small dot
between them. A node shared between two instances is ringed in red.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c07_render_tracking.py
    .venv\\Scripts\\python.exe tools/c07_render_tracking.py --per-category 2 --no-clips
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from c04_score_tracking import reference_chain
from s10_shared_node_20k import SHARED_NODE_PX, median_span_px, tail_jump_threshold_px

from medaka3d import load_catalog, load_paths
from medaka3d.cohort import ExperimentId
from medaka3d.eval import EncounterConfig, find_encounters
from medaka3d.provenance import provenance
from medaka3d.track import UNTRACKED
from medaka3d.types import FPS, VIEWS

RUNS = ("bottomup", "topdown")
METHOD = "gated"

#: Categorical slots 1-3 of the reference palette — the set documented as
#: clearing the all-pairs colour-vision floors. Never cycled; a fourth track
#: folds into OTHER rather than inventing a hue.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a")
OTHER = "#8a8a85"
ALERT = "#e34948"
INK = "#0b0b0b"
PAPER = "#fcfcfb"

#: Half a window, in frames. 166 = 1 s at 166 fps, so a clip spans 2 s.
WINDOW = 166
#: Playback rate. Source is 166 fps, so this is 1/5.5 speed — fast enough to
#: watch, slow enough to see a swap happen.
CLIP_FPS = 30
TILE = 460


def bgr(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (4, 2, 0))


def frame_bgr(video, index) -> np.ndarray:
    """One video frame in **BGR**, the only order OpenCV writes correctly.

    `sleap_io` hands back **RGB**; `cv2.imwrite` and `cv2.VideoWriter` interpret
    what they are given as **BGR**. Reading one straight into the other exchanges
    red and blue, and every renderer in this project did exactly that. Measured
    on K_1_2 side1 frame 214,398: the channel-reversed `sleap_io` frame equals
    `cv2.VideoCapture`'s byte for byte, mean absolute difference **0.000**.

    It is not a cosmetic bug. **The experimenter's own fish descriptors are
    colours** — `group_list.csv` names group K's males *white* and *red* and the
    female *yellow* — so a red male rendered blue is an invitation to
    misidentify an animal by eye, which is what these renders exist for. Every
    still and clip produced before 2026-08-19 has the two channels exchanged.
    """
    image = np.asarray(video[int(index)])
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


class ViewData:
    """One (run, experiment, view): detections plus the track ID of each instance."""

    def __init__(self, work_dir: Path, run: str, experiment: str, view: str,
                 tracks_npz: Path | None = None):
        """``tracks_npz`` selects which run's per-view track IDs are drawn.

        It defaults to `c03`'s, which is what every caller wanted while `c03` was
        the only tracker. **A renderer given a different arm's npz must pass it
        here**, or the video is drawn with `c03`'s IDs over another arm's
        geometry and every scene looks like whatever `c03` did.
        """
        self.detections = c01.load(work_dir, experiment, view, run)
        source = tracks_npz or (
            work_dir / "c03_track_3d" / run / f"{METHOD}__{experiment}.npz")
        tracks = np.load(source, allow_pickle=False)
        self.track = tracks[f"track__{view}"]
        self.start = np.searchsorted(
            self.detections.frame_idx, np.arange(self.detections.n_frames), side="left"
        )
        self.stop = np.searchsorted(
            self.detections.frame_idx, np.arange(self.detections.n_frames), side="right"
        )
        if self.track.shape[0] != self.detections.frame_idx.shape[0]:
            raise SystemExit(
                f"{run}/{experiment}/{view}: {self.track.shape[0]} track ids for "
                f"{self.detections.frame_idx.shape[0]} instances"
            )

    def at(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """(k, n_nodes, 2) points and (k,) track ids for one frame."""
        a, b = int(self.start[frame]), int(self.stop[frame])
        return self.detections.points[a:b].astype(float), self.track[a:b]


def shared_pairs(points: np.ndarray, node: int) -> list[tuple[int, int]]:
    """Instance pairs placing ``node`` within the shared-node radius."""
    out = []
    for a in range(points.shape[0]):
        for b in range(a + 1, points.shape[0]):
            pa, pb = points[a, node], points[b, node]
            if (
                np.isfinite(pa).all()
                and np.isfinite(pb).all()
                and float(np.linalg.norm(pa - pb)) <= SHARED_NODE_PX
            ):
                out.append((a, b))
    return out


def colour_map(view_data: dict, frames: np.ndarray) -> dict[int, str]:
    """Track ID -> hex, the three longest in this window taking slots 1-3.

    Longest-first rather than first-seen, so the three animals get the three
    hues and the fragments get the neutral, whichever order they appear in.
    """
    counts: dict[int, int] = {}
    for data in view_data.values():
        for frame in frames:
            _, tracks = data.at(int(frame))
            for t in tracks:
                if int(t) != UNTRACKED:
                    counts[int(t)] = counts.get(int(t), 0) + 1
    ranked = sorted(counts, key=lambda t: (-counts[t], t))
    return {t: PALETTE[i] if i < len(PALETTE) else OTHER for i, t in enumerate(ranked)}


#: Crop side, in median body lengths. A 20 cm tank at 1200 px makes one fish
#: ~110 px, so a whole-tank crop renders it at 40 px in a tile and nothing can be
#: judged. Six body lengths keeps the animal legible at a fixed, stated scale —
#: which also means two tiles are directly comparable, where a
#: crop-to-fit-the-animals box silently rescales every moment.
CROP_BODY_LENGTHS = 6.0


def crop_box(view_data: dict, frames: np.ndarray, shape: tuple[int, int],
             span_px: float, focus: tuple[float, float] | None = None):
    """A fixed-scale square crop, centred on ``focus`` or on the animals.

    ``focus`` is the event location when the moment has one — a shared node or a
    teleporting mouth — because that is what the tile exists to show. Without one
    the crop centres on the instances' centroid over the window.
    """
    if focus is None:
        xs, ys = [], []
        for data in view_data.values():
            for frame in frames:
                points, _ = data.at(int(frame))
                finite = points[np.isfinite(points).all(axis=-1)]
                if finite.size:
                    xs.extend(finite[:, 0].tolist())
                    ys.extend(finite[:, 1].tolist())
        if not xs:
            return 0, 0, min(shape), min(shape)
        focus = (float(np.median(xs)), float(np.median(ys)))
    # Clamp the side to the frame *before* placing the origin. The other order
    # makes `shape - side` negative when the box exceeds the frame, and
    # `np.clip(v, 0, negative)` returns the negative bound, which slices to an
    # empty crop rather than failing.
    side = int(min(CROP_BODY_LENGTHS * span_px, shape[0], shape[1]))
    x0 = int(np.clip(focus[0] - side / 2, 0, shape[1] - side))
    y0 = int(np.clip(focus[1] - side / 2, 0, shape[0] - side))
    return x0, y0, side, side


def focus_of(moment: dict, data: dict, view: str) -> tuple[float, float] | None:
    """Where the tile should be centred: the event, when the moment names one."""
    if moment["view"] != view:
        return None
    frame = moment["frame"]
    points, _tracks = data[("topdown" if moment["category"] == "shared_tail" else
                           moment.get("run", "topdown"), view)].at(frame)
    if moment["category"] == "shared_tail":
        pairs = shared_pairs(points, 2)
        if pairs:
            a, b = pairs[0]
            return tuple(((points[a, 2] + points[b, 2]) / 2).tolist())
    if moment["category"] == "track_jump":
        finite = points[np.isfinite(points[:, 0]).all(axis=-1)]
        if finite.size:
            return tuple(finite[:, 0].mean(axis=0).tolist())
    return None


def draw(image: np.ndarray, points: np.ndarray, tracks: np.ndarray, colours: dict,
         box, scale: float) -> np.ndarray:
    """Draw one architecture's instances onto a cropped, scaled tile."""
    x0, y0, w, h = box
    tile = cv2.resize(image[y0 : y0 + h, x0 : x0 + w], (TILE, TILE),
                      interpolation=cv2.INTER_AREA)
    halo = bgr(INK)

    flagged = set()
    for node in (0, 2):
        for a, b in shared_pairs(points, node):
            flagged.update({(a, node), (b, node)})

    def to_tile(xy):
        return (int((xy[0] - x0) * scale), int((xy[1] - y0) * scale))

    for index in range(points.shape[0]):
        track = int(tracks[index]) if index < len(tracks) else UNTRACKED
        colour = bgr(colours.get(track, OTHER))
        p = points[index]
        ok = np.isfinite(p).all(axis=-1)
        # Body axis first, so the node marks sit on top of it.
        chain = [to_tile(p[i]) for i in range(3) if ok[i]]
        for u, v in pairwise(chain):
            cv2.line(tile, u, v, halo, 4, cv2.LINE_AA)
            cv2.line(tile, u, v, colour, 2, cv2.LINE_AA)
        if ok[1]:
            cv2.circle(tile, to_tile(p[1]), 3, colour, -1, cv2.LINE_AA)
        if ok[0]:  # mouth: circle
            c = to_tile(p[0])
            cv2.circle(tile, c, 7, halo, 4, cv2.LINE_AA)
            cv2.circle(tile, c, 7, colour, 2, cv2.LINE_AA)
        if ok[2]:  # tail: square
            c = to_tile(p[2])
            cv2.rectangle(tile, (c[0] - 6, c[1] - 6), (c[0] + 6, c[1] + 6), halo, 4, cv2.LINE_AA)
            cv2.rectangle(tile, (c[0] - 6, c[1] - 6), (c[0] + 6, c[1] + 6), colour, 2, cv2.LINE_AA)
        for node in (0, 2):
            if (index, node) in flagged and ok[node]:
                cv2.circle(tile, to_tile(p[node]), 14, bgr(ALERT), 2, cv2.LINE_AA)
        # Identity is never colour alone: the track ID is drawn on the animal.
        if ok[0] and track != UNTRACKED:
            c = to_tile(p[0])
            label = str(track)
            cv2.putText(tile, label, (c[0] + 10, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, halo, 3, cv2.LINE_AA)
            cv2.putText(tile, label, (c[0] + 10, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return tile


def caption(tile: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    band = np.full((34 if not sub else 50, tile.shape[1], 3), bgr(PAPER), np.uint8)
    cv2.putText(band, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, bgr(INK), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(band, sub, (8, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    bgr("#52514e"), 1, cv2.LINE_AA)
    return np.vstack([band, tile])


def grid(images: dict, frame: int, note: str) -> np.ndarray:
    """2 runs x 3 views, with a header naming the moment."""
    rows = []
    for run in RUNS:
        row = [caption(images[(run, v)], f"{run}  ·  {v}") for v in VIEWS]
        rows.append(np.hstack(row))
    body = np.vstack(rows)
    lines = _wrap(note, body.shape[1] - 20, 0.58)
    header = np.full((14 + 24 * len(lines), body.shape[1], 3), bgr(PAPER), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(header, line, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, bgr(INK), 1, cv2.LINE_AA)
    return np.vstack([header, body])


def _wrap(text: str, width_px: int, scale: float) -> list[str]:
    """Greedy wrap against the measured text width, so a caption cannot run off
    the edge — which it silently did, truncating the sentence that explains what
    the colours mean."""
    lines, current = [], ""
    for word in text.split(" "):
        trial = f"{current} {word}".strip()
        (w, _), _ = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if w > width_px and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


# --------------------------------------------------------------------------
# Selection — deterministic, and fixed before anything is drawn
# --------------------------------------------------------------------------
def find_shared_tail(data: dict, frames: np.ndarray) -> list[dict]:
    """Frames where topdown shares a tail and bottomup does not, same view."""
    out = []
    for view in VIEWS:
        td, bu = data[("topdown", view)], data[("bottomup", view)]
        for frame in frames:
            p_td, _ = td.at(int(frame))
            if p_td.shape[0] < 2:
                continue
            pairs = shared_pairs(p_td, 2)
            if not pairs:
                continue
            p_bu, _ = bu.at(int(frame))
            if shared_pairs(p_bu, 2):
                continue
            a, b = pairs[0]
            out.append({
                "category": "shared_tail", "view": view, "frame": int(frame),
                "key": float(np.linalg.norm(p_td[a, 2] - p_td[b, 2])),
                "detail": f"top-down places two tails "
                          f"{float(np.linalg.norm(p_td[a, 2] - p_td[b, 2])):.2f} px apart; "
                          f"bottom-up does not",
            })
    return out


def find_track_jump(data: dict, frames: np.ndarray, gate: dict) -> list[dict]:
    """Frames where one architecture's track teleports and the other's does not."""
    out = []
    for view in VIEWS:
        for run, other in (("topdown", "bottomup"), ("bottomup", "topdown")):
            a, b = data[(run, view)], data[(other, view)]
            for frame in frames[:-1]:
                moved = _max_track_step(a, int(frame))
                if moved is None or moved <= gate[view]:
                    continue
                counter = _max_track_step(b, int(frame))
                if counter is not None and counter > gate[view]:
                    continue
                out.append({
                    "category": "track_jump", "view": view, "frame": int(frame),
                    "key": -moved,
                    "detail": f"{run} moves one track {moved:.0f} px in 6 ms "
                              f"(gate {gate[view]:.0f} px); {other} does not",
                })
    return out


def _max_track_step(data: ViewData, frame: int) -> float | None:
    here, here_t = data.at(frame)
    there, there_t = data.at(frame + 1)
    best = None
    for i, t in enumerate(here_t):
        if int(t) == UNTRACKED:
            continue
        match = np.flatnonzero(there_t == t)
        if match.size != 1:
            continue
        a, b = here[i, 0], there[int(match[0]), 0]
        if np.isfinite(a).all() and np.isfinite(b).all():
            d = float(np.linalg.norm(a - b))
            best = d if best is None else max(best, d)
    return best


def find_encounters_for(work_dir: Path, run: str, experiment: str) -> list:
    npz = np.load(
        work_dir / "c03_track_3d" / run / f"{METHOD}__{experiment}.npz", allow_pickle=False
    )
    xyz = npz["xyz"].astype(float)
    chain = reference_chain(xyz, npz["slot_track"].max(axis=1) >= 0)
    rows = np.arange(chain.shape[0])[:, None]
    canonical = np.where((chain >= 0)[:, :, None], xyz[rows, np.maximum(chain, 0)], np.nan)
    return find_encounters(canonical, experiment=experiment, config=EncounterConfig())


def evenly(items: list[dict], n: int) -> list[dict]:
    """``n`` items spread across the sorted list — not the ``n`` most extreme."""
    if not items or n <= 0:
        return []
    items = sorted(items, key=lambda d: (d["key"], d["view"], d["frame"]))
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


# --------------------------------------------------------------------------
def continuity_figure(work_dir: Path, experiment: str, frames: np.ndarray,
                      colours_by_run: dict, out: Path, title: str) -> None:
    """3D pairwise separation over the window, one panel per architecture.

    The form follows the job: the question is *did identity hold*, so the plot
    shows the physical quantity identity is at risk from — how close the animals
    came, in mm — with the tracker's own identity assignment as a band in its own
    strip beneath. A break or a swap is a colour change in the band; an encounter
    is a dip in the trace. One y-axis, one unit, no second scale.

    The identity strip gets its **own axes** rather than being drawn inside the
    trace. Inside, it collided with the legend and sat at a y-position that read
    as a separation value it does not have.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        4, 1, figsize=(11, 6.6), sharex=True, facecolor=PAPER,
        gridspec_kw={"height_ratios": [5, 1, 5, 1], "hspace": 0.22},
    )
    seconds = (frames - frames[0]) / FPS
    for row, run in enumerate(RUNS):
        trace, strip = axes[row * 2], axes[row * 2 + 1]
        npz = np.load(
            work_dir / "c03_track_3d" / run / f"{METHOD}__{experiment}.npz", allow_pickle=False
        )
        xyz = npz["xyz"].astype(float)[frames]
        slot_track = npz["slot_track"][frames]
        pairs = [(0, 1), (0, 2), (1, 2)]
        separation = np.stack(
            [np.linalg.norm(xyz[:, a] - xyz[:, b], axis=-1) for a, b in pairs], axis=-1
        )
        # Frames the reconstruction could not resolve have no separation. NaN is
        # the honest value and leaves a visible gap in the trace; suppressing the
        # all-NaN warning is not the same as filling the gap in.
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered")
            low = np.nanmin(separation, axis=-1)
            high = np.nanmax(separation, axis=-1)

        trace.plot(seconds, high, lw=1.2, color="#b8b7b0", label="furthest pair", zorder=2)
        trace.plot(seconds, low, lw=2, color=PALETTE[0], label="closest pair", zorder=3)
        trace.axhline(EncounterConfig().close_mm, lw=1, ls=(0, (4, 3)),
                      color="#8a8a85", zorder=1)
        trace.annotate("20 mm — identity at risk below here",
                       xy=(seconds[-1], EncounterConfig().close_mm), xytext=(-4, 4),
                       textcoords="offset points", ha="right", fontsize=7.5, color="#52514e")
        trace.set_ylabel("separation (mm)", fontsize=9, color=INK)
        trace.set_title(run, fontsize=10, loc="left", color=INK, pad=4)
        trace.set_ylim(0, float(np.nanmax(high)) * 1.12 if np.isfinite(high).any() else 40)
        trace.spines[["top", "right"]].set_visible(False)
        trace.tick_params(labelsize=8, colors="#52514e")
        trace.grid(axis="y", lw=0.5, color="#e6e5e0", zorder=0)
        trace.set_facecolor(PAPER)
        if row == 0:
            # Inside the axes at upper right, where the traces never reach — a
            # legend under the suptitle collided with it.
            trace.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)

        colours = colours_by_run[run]
        breaks = 0
        for slot in range(slot_track.shape[1]):
            ids = slot_track[:, slot]
            edges = np.flatnonzero(np.diff(ids) != 0) + 1
            breaks += len(edges)
            for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(ids)], strict=True):
                track = int(ids[lo])
                strip.axhspan(
                    slot + 0.12, slot + 0.88, xmin=lo / len(ids), xmax=hi / len(ids),
                    color=colours.get(track, OTHER) if track != UNTRACKED else "#dcdbd5",
                    lw=0,
                )
        strip.set_ylim(0, slot_track.shape[1])
        strip.set_yticks([])
        strip.set_xlim(seconds[0], seconds[-1])
        strip.set_facecolor(PAPER)
        for side in ("top", "right", "left"):
            strip.spines[side].set_visible(False)
        strip.tick_params(labelsize=8, colors="#52514e")
        strip.set_ylabel("identity", fontsize=8, color="#52514e", rotation=0,
                         ha="right", va="center", labelpad=8)
        # The count is the point of the strip, so it is stated rather than
        # left to be counted off the stripes by eye.
        strip.annotate(f"{breaks} identity changes in {len(frames)} frames",
                       xy=(1.0, 1.15), xycoords="axes fraction", ha="right",
                       fontsize=7.5, color="#52514e")

    axes[-1].set_xlabel("seconds within the window", fontsize=9, color=INK)
    fig.suptitle(title, fontsize=11, x=0.012, ha="left", color=INK)
    fig.text(0.012, 0.005,
             "Identity strips are the tracker's own assignment, one stripe per 3D slot: a "
             "colour change is a track break or swap, grey is unresolved. Colours rank the "
             "window's longest tracks and are per-architecture, not a correspondence.",
             fontsize=7.5, color="#52514e")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--experiment", default=c01.EXPERIMENTS[0])
    parser.add_argument("--per-category", type=int, default=3,
                        help="Moments rendered per category.")
    parser.add_argument("--scan", type=int, default=40_000,
                        help="Contiguous frames scanned for candidate moments.")
    parser.add_argument("--scan-start", type=int, default=120_000,
                        help="Where the scan begins. Deterministic; off the handling period.")
    parser.add_argument("--no-clips", action="store_true", help="Stills and plots only.")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    out_dir = paths.work_dir / "c07_render_tracking" / args.experiment
    for sub in ("montage", "clips", "continuity"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    with provenance("render_tracking", out_dir, name="c07") as record:
        record.params_from(args)
        groups = sorted({ExperimentId.parse(args.experiment).group})
        record.param("n_groups", len(groups))

        gate, span_px = {}, {}
        for view in VIEWS:
            key = f"labels.round8.{'top' if view == 'top_cam' else 'side'}.train"
            record.input(key, catalog.get(key))
            span, _ = median_span_px(catalog.get(key))
            gate[view] = tail_jump_threshold_px(span)
            span_px[view] = span
        print(f"experiment : {args.experiment}   n_groups = {len(groups)} {groups}")
        print("gates      : " + "  ".join(f"{v} {gate[v]:.0f}px" for v in VIEWS))

        data = {
            (run, view): ViewData(paths.work_dir, run, args.experiment, view)
            for run in RUNS for view in VIEWS
        }
        scan = np.arange(args.scan_start, args.scan_start + args.scan)
        print(f"scanning   : frames {scan[0]:,}-{scan[-1]:,}")

        started = time.time()
        candidates = find_shared_tail(data, scan) + find_track_jump(data, scan, gate)
        counts = {}
        for c in candidates:
            counts[c["category"]] = counts.get(c["category"], 0) + 1

        # Encounters come from the incumbent's reconstruction so both
        # architectures are shown the same moments.
        encounters = [
            e for e in find_encounters_for(paths.work_dir, RUNS[0], args.experiment)
            if scan[0] <= e.entry_frame and e.exit_frame <= scan[-1]
        ]
        candidates += [
            {"category": "encounter", "view": "top_cam",
             "frame": int((e.entry_frame + e.exit_frame) // 2), "key": -e.duration_frames,
             "detail": f"encounter of {e.duration_s:.1f} s, closest approach "
                       f"{e.min_distance_mm:.1f} mm, slots {e.participants}"}
            for e in encounters
        ]
        counts["encounter"] = len(encounters)

        anomalous = {c["frame"] for c in candidates}
        routine = [
            {"category": "routine", "view": "top_cam", "frame": int(f), "key": float(f),
             "detail": "no shared node and no track jump in either architecture"}
            for f in scan[:: max(1, len(scan) // 400)]
            if not any(abs(int(f) - a) < WINDOW for a in anomalous)
        ]
        candidates += routine
        counts["routine"] = len(routine)
        print("candidates : " + "  ".join(f"{k} {v:,}" for k, v in sorted(counts.items()))
              + f"   ({time.time() - started:.0f}s)")
        record.param("candidate_counts", counts)

        chosen = []
        for category in ("shared_tail", "track_jump", "encounter", "routine"):
            chosen += evenly([c for c in candidates if c["category"] == category],
                             args.per_category)
        if not chosen:
            raise SystemExit("no moments matched; widen --scan")

        videos = {v: __import__("sleap_io").Video.from_filename(
            str(paths.video(args.experiment, v))) for v in VIEWS}
        shape = videos[VIEWS[0]].shape[1:3]

        manifest = []
        for moment in chosen:
            frame = moment["frame"]
            window = np.arange(max(0, frame - WINDOW),
                               min(c01.N_FRAMES, frame + WINDOW))
            colours_by_run = {
                run: colour_map({v: data[(run, v)] for v in VIEWS}, window) for run in RUNS
            }
            boxes = {
                v: crop_box({r: data[(r, v)] for r in RUNS}, window, shape, span_px[v],
                            focus_of(moment, data, v))
                for v in VIEWS
            }
            stem = f"{moment['category']}_f{frame}_{moment['view']}"

            tiles = {}
            for view in VIEWS:
                image = frame_bgr(videos[view], frame)
                _x0, _y0, w, _h = boxes[view]
                for run in RUNS:
                    points, tracks = data[(run, view)].at(frame)
                    tiles[(run, view)] = draw(image, points, tracks,
                                              colours_by_run[run], boxes[view], TILE / w)
            note = (f"{moment['category']}  ·  {args.experiment}  ·  frame {frame:,}  "
                    f"·  {moment['detail']}")
            note += f"   |  crop {CROP_BODY_LENGTHS:.0f} body lengths; "
            note += "colours are per-architecture track IDs, not a correspondence"
            still = grid(tiles, frame, note)
            cv2.imwrite(str(out_dir / "montage" / f"{stem}.png"), still)

            continuity_figure(
                paths.work_dir, args.experiment, window, colours_by_run,
                out_dir / "continuity" / f"{stem}.png",
                f"{moment['category']} · frame {frame:,} · {args.experiment}",
            )

            if not args.no_clips:
                writer = cv2.VideoWriter(
                    str(out_dir / "clips" / f"{stem}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS,
                    (still.shape[1], still.shape[0]),
                )
                for f in window:
                    tiles = {}
                    for view in VIEWS:
                        image = frame_bgr(videos[view], f)
                        _x0, _y0, w, _h = boxes[view]
                        for run in RUNS:
                            points, tracks = data[(run, view)].at(int(f))
                            tiles[(run, view)] = draw(image, points, tracks,
                                                      colours_by_run[run], boxes[view], TILE / w)
                    writer.write(grid(tiles, int(f), note))
                writer.release()

            manifest.append({**moment, "window": [int(window[0]), int(window[-1])],
                             "files": stem})
            print(f"  {stem}")

        record.param("moments", manifest)
        (out_dir / "c07_moments.json").write_text(
            json.dumps({"experiment": args.experiment, "n_groups": len(groups),
                        "gate_px": gate, "candidate_counts": counts,
                        "moments": manifest}, indent=2, default=float),
            encoding="utf-8",
        )
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

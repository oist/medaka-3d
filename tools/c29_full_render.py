"""C29 - the whole of K_1_2 rendered at native resolution, 2x2, with the pipeline's own IDs.

Every render in this project so far has been a *selection*: `c07` picks four
categories of moment, `c11` builds review packs, `c15` draws the eight measured
swaps. Each answers a question that was already framed numerically. **None of
them lets anyone watch the recording.** 35 minutes at 166 fps is 348,600 frames
and the eye is the only instrument that has never been pointed at more than a
few thousand of them at a time.

This renders all of them.

=============================================================================
WHAT IS DRAWN, AND FROM WHICH ARM
=============================================================================

Stage A is `c27` - the uncapped 2D tracker run **serially**, not `c24`'s 30-way
chunked run, whose partition-equivalence claim was retracted on 2026-08-26.
The correspondence is `c28`, one arm of the inherited lambda grid.

**`c28` deliberately selects no lambda** (2026-08-26: the pre-registered
selector resolves a factor of 3.16 at 0.09-0.41% of its own score, so it does
not identify lambda on this data). A video needs one arm, so exactly one of
``--lam`` (a stored `c28` arm) and ``--npz`` (any reconstruction npz in `c28`'s
schema, e.g. a `c30` stitched arm) must be given - neither has a default - and
the choice is written into the frame, the file name and the provenance record.
Choosing an arm to look at is not choosing a lambda to report.

=============================================================================
THE ID ON SCREEN IS THE SLOT, AND THAT IS A MEASURED CHOICE
=============================================================================

Two identity labels exist in the `c28` output and they are very different
objects:

* ``slot_track`` - the **gated 3D tracker**'s id. On K_1_2 at lambda 0.767 it
  emits **770 tracks** over the recording; the top three cover 11.4% of it and
  the median run is **3 frames**. Drawn as the primary label it would be a
  number changing several times a second.
* the **slot** itself - the column of ``xyz``. `c28`'s own ``coverage_label``
  records **3** ids covering **98.77%** of frames.

The slot is documented in ``medaka3d.reconstruct`` as "arbitrary and stable only
within a frame", so it is fair to ask whether colouring by it invents a
continuity that is not there. Measured on this arm, over frames where both
neighbours resolved: the same slot's 3D position moves a median of **0.294 mm**
between adjacent frames, p99 **1.67 mm**, and only **0.239%** of steps exceed
`c19`'s 4.0 mm gate - against a median inter-animal separation of **95.1 mm**.
A slot permutation would appear as a step of that second size. It essentially
does not happen, and lambda is the penalty that buys it.

So: **colour and the large label are the slot** (``S0``/``S1``/``S2``), and the
gated track id is printed small beneath it, so a break is visible rather than
hidden. Neither is an animal's name. There is no male-vs-male ground truth in
this project, and `female_sex_gt.csv` names one fish in one experiment.

Instances the correspondence did not use - the uncapped detector reports up to
eight per frame - are drawn in one neutral grey with no label, following the
rule `c07` fixed: a fourth hue would imply a fourth animal.

=============================================================================
THE 3D PANEL
=============================================================================

Fixed orthographic view of the tank, which the calibration puts at
**192 x 192 x 186 mm** (x, y, z; the water surface is the top camera's
refracting interface at z = 186). No rotation: a moving camera makes two frames
incomparable by eye, which is the one thing this render exists to allow.

All **three nodes** are triangulated per slot, not just the anchor. `c28` stores
only the anchor's ``xyz``, but it stores ``source`` - the detector's own row per
slot per view - so mouth and tail follow from the same correspondence with no
new assumption. **The tail is triangulated for display only.** It is the node
`s10` measured part-mixing on and it is excluded from every cost that decides
identity; seeing it is not the same as trusting it.

The trail is **5 s = 830 frames** of the **pectoral** node, the most reliable of
the three in every view (3.9 / 6.3 / 5.3% of frames missing it, against the
tail's 10.3 / 11.7 / 11.6%), faded toward the panel ground with age.

=============================================================================
WHY SEGMENTING THE RENDER IS NOT THE MISTAKE `c24` MADE
=============================================================================

`c24` ran the *tracker* in chunks and that changed the computation, because
``LocalQueueCandidates.current_tracks`` carries unbounded history. Here every
per-frame input - detections, slot assignment, track ids, 3D points, projected
pixels - is a **precomputed global array**, and the 5 s trail is read backwards
out of that same array rather than accumulated. A segment therefore draws the
identical pixels wherever its boundaries fall. ``--check-segmenting`` proves it
rather than asserting it: it renders a boundary frame inside a segment and alone
and compares the two images byte for byte.

Colour order: OpenCV's ``VideoCapture`` already returns **BGR**, which is what
``ffmpeg -pix_fmt bgr24`` and every drawing call here expect. `c07` measured
that this is byte-identical to the channel-reversed ``sleap_io`` frame (K_1_2
side1 frame 214,398, mean absolute difference 0.000). Nothing converts anything,
which is why the red male stays red.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c29_full_render.py --lam 0.767 --check-alignment
    .venv\\Scripts\\python.exe tools/c29_full_render.py --lam 0.767 --frames 100000 102000
    .venv\\Scripts\\python.exe tools/c29_full_render.py --lam 0.767 --workers 6
    .venv\\Scripts\\python.exe tools/c29_full_render.py --npz <arm> --tag chain-c28-c33-c30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from c02_tracking_probe import CALIBRATION
from c07_render_tracking import INK, OTHER, PALETTE, PAPER, bgr

from medaka3d import load_paths
from medaka3d.calib import load_rig, triangulate
from medaka3d.provenance import provenance
from medaka3d.reconstruct import N_ANIMALS
from medaka3d.track import UNTRACKED
from medaka3d.types import FPS, VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN = "bottomup"

#: Native camera resolution. "100% resolution" means no resize anywhere, so a
#: 2x2 grid of these is exactly 2400x2400.
TILE = 1200

#: The detector's skeleton, in its own order. Asserted against `c01`, not assumed.
NODES = ("mouth", "pectoral", "tail")
MOUTH, PECTORAL, TAIL = 0, 1, 2

#: The tank interior in world mm, from the promoted refractive calibration:
#: 192 x 192 inside, 186 mm of water. The top camera's interface plane sits at
#: z = 186 and the side cameras' at y = -4 and x = 196 (outer glass), which is
#: what fixes these numbers rather than the nominal "20 cm" of the protocol.
TANK = ((0.0, 192.0), (0.0, 192.0), (0.0, 186.0))

#: Trail length, in seconds. 5 s at 166 fps is 830 frames.
TRAIL_SECONDS = 5.0

#: Fixed view direction for the 3D panel, matplotlib's convention.
AZIM, ELEV = -58.0, 20.0

#: Quadrants. Three cameras and the reconstruction they produce, reading order.
LAYOUT = ((VIEWS[0], VIEWS[1]), (VIEWS[2], "reconstruction"))

FFMPEG = "ffmpeg"


# --------------------------------------------------------------------------
# Geometry for the 3D panel
# --------------------------------------------------------------------------
def basis(azim_deg: float, elev_deg: float) -> np.ndarray:
    """(3, 3) rows: screen right, screen up, depth toward the viewer."""
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    ca, sa, ce, se = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
    return np.stack([
        np.array([-sa, ca, 0.0]),
        np.array([-ca * se, -sa * se, ce]),
        np.array([ca * ce, sa * ce, se]),
    ])


class Panel:
    """Fixed orthographic projection of the tank into a TILE x TILE image.

    The scale is set from the tank's own projected bounding box, so the whole
    volume is always on screen and a point never silently leaves the panel.
    """

    def __init__(self, margin: int = 74):
        self.rotation = basis(AZIM, ELEV)
        corners = np.array([[x, y, z] for x in TANK[0] for y in TANK[1] for z in TANK[2]])
        self.centre = corners.mean(axis=0)
        flat = (corners - self.centre) @ self.rotation.T
        span = max(np.ptp(flat[:, 0]), np.ptp(flat[:, 1]))
        self.scale = (TILE - 2 * margin) / span
        self.origin = np.array([TILE / 2.0, TILE / 2.0])

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """(..., 3) world mm -> (..., 2) float pixels. NaN stays NaN."""
        flat = (np.asarray(xyz, float) - self.centre) @ self.rotation.T
        out = np.empty((*flat.shape[:-1], 2), float)
        out[..., 0] = self.origin[0] + flat[..., 0] * self.scale
        out[..., 1] = self.origin[1] - flat[..., 1] * self.scale
        return out

    def depth(self, xyz: np.ndarray) -> np.ndarray:
        return (np.asarray(xyz, float) - self.centre) @ self.rotation[2]


def tank_background(panel: Panel) -> np.ndarray:
    """The panel ground with the tank drawn once, so the loop only copies it."""
    image = np.full((TILE, TILE, 3), bgr(PAPER), np.uint8)
    corners = np.array([[x, y, z] for x in TANK[0] for y in TANK[1] for z in TANK[2]])
    pixels = panel.project(corners).astype(np.int32)
    depths = panel.depth(corners)
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if bin(i ^ j).count("1") == 1]
    # Far edges first and lighter, so the box reads as a box rather than a mesh.
    edges.sort(key=lambda e: depths[e[0]] + depths[e[1]])
    for k, (i, j) in enumerate(edges):
        near = k >= len(edges) // 2
        cv2.line(image, tuple(pixels[i]), tuple(pixels[j]),
                 bgr("#9a9a95") if near else bgr("#d6d6d1"), 1, cv2.LINE_AA)
    # The floor plane's mid-lines, at 96 mm, as a scale the eye can use.
    for a, b in (([96, 0, 0], [96, 192, 0]), ([0, 96, 0], [192, 96, 0])):
        u, v = panel.project(np.array([a, b], float)).astype(np.int32)
        cv2.line(image, tuple(u), tuple(v), bgr("#e4e4df"), 1, cv2.LINE_AA)
    label(image, "192 x 192 x 186 mm", (16, TILE - 18), "#8a8a85", 0.46)
    return image


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def label(image, text, origin, colour="#0b0b0b", scale=0.5, halo=True):
    origin = (int(origin[0]), int(origin[1]))
    if halo:
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    bgr(PAPER), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                bgr(colour), 1, cv2.LINE_AA)


def stroke(image, a, b, colour):
    """1 px over a 2 px halo - `c15`'s convention, thin enough not to cover the fish."""
    cv2.line(image, a, b, bgr(INK), 2, cv2.LINE_AA)
    cv2.line(image, a, b, colour, 1, cv2.LINE_AA)


def marks(image, pixels, ok, colour):
    """mouth ring, pectoral dot, tail square, joined along the body axis."""
    chain = [(int(pixels[i][0]), int(pixels[i][1])) for i in range(3) if ok[i]]
    for u, v in pairwise(chain):
        stroke(image, u, v, colour)
    if ok[MOUTH]:
        p = (int(pixels[MOUTH][0]), int(pixels[MOUTH][1]))
        cv2.circle(image, p, 5, bgr(INK), 2, cv2.LINE_AA)
        cv2.circle(image, p, 5, colour, 1, cv2.LINE_AA)
    if ok[PECTORAL]:
        cv2.circle(image, (int(pixels[PECTORAL][0]), int(pixels[PECTORAL][1])),
                   2, colour, -1, cv2.LINE_AA)
    if ok[TAIL]:
        x, y = int(pixels[TAIL][0]), int(pixels[TAIL][1])
        cv2.rectangle(image, (x - 4, y - 4), (x + 4, y + 4), bgr(INK), 2, cv2.LINE_AA)
        cv2.rectangle(image, (x - 4, y - 4), (x + 4, y + 4), colour, 1, cv2.LINE_AA)


def draw_view(image, points, slots, gated, view):
    """One camera's frame, at native size, with every detected instance on it."""
    order = np.argsort(np.where(slots < 0, 0, 1))  # unassigned first, under the rest
    for index in order:
        slot = int(slots[index])
        colour = bgr(PALETTE[slot]) if slot >= 0 else bgr(OTHER)
        p = points[index]
        ok = np.isfinite(p).all(axis=-1)
        if not ok.any():
            continue
        pixels = np.nan_to_num(p)
        marks(image, pixels, ok, colour)
        if slot >= 0 and ok[MOUTH]:
            x, y = pixels[MOUTH]
            label(image, f"S{slot}", (x + 9, y - 6), PALETTE[slot], 0.62)
            track = int(gated[index])
            label(image, "t--" if track == UNTRACKED else f"t{track}",
                  (x + 9, y + 11), "#6f6e6a", 0.42)
    label(image, view, (16, 34), INK, 0.66)


def runs(mask: np.ndarray):
    """(start, stop) pairs of contiguous True - a trail must not bridge a gap."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2], strict=True))


def draw_panel(background, proj, ok, slot_track, frame, trail_frames, note):
    """The reconstruction: 3 nodes x 3 slots, plus the pectoral's 5 s trail."""
    image = background.copy()
    paper = np.array(bgr(PAPER), float)
    lo = max(0, frame - trail_frames + 1)

    for slot in range(N_ANIMALS):
        base = np.array(bgr(PALETTE[slot]), float)
        history = np.asarray(proj[lo:frame + 1, slot, PECTORAL])
        alive = np.asarray(ok[lo:frame + 1, slot, PECTORAL])
        for start, stop in runs(alive):
            piece = history[start:stop]
            if piece.shape[0] < 2:
                continue
            # Age fades toward the ground. Drawn in blocks rather than per
            # vertex: OpenCV has no per-vertex alpha and 830 line calls a slot
            # a frame is the whole frame budget.
            for block in np.array_split(np.arange(piece.shape[0]), 12):
                if block.size < 2:
                    continue
                age = (start + block[-1]) / max(alive.size - 1, 1)
                mix = 0.16 + 0.84 * age
                colour = tuple(round(c) for c in paper + (base - paper) * mix)
                cv2.polylines(image, [piece[block]], False, colour, 1, cv2.LINE_AA)

    for slot in range(N_ANIMALS):
        here = np.asarray(ok[frame, slot])
        if not here.any():
            continue
        colour = bgr(PALETTE[slot])
        pixels = np.asarray(proj[frame, slot])
        marks(image, pixels, here, colour)
        if here[MOUTH]:
            x, y = pixels[MOUTH]
            label(image, f"S{slot}", (x + 9, y - 6), PALETTE[slot], 0.62)
            track = int(slot_track[frame, slot])
            label(image, "t--" if track == UNTRACKED else f"t{track}",
                  (x + 9, y + 11), "#6f6e6a", 0.42)

    label(image, "reconstruction", (16, 34), INK, 0.66)
    for i, line in enumerate(note):
        label(image, line, (16, 62 + 22 * i), "#52514e", 0.46)
    return image


# --------------------------------------------------------------------------
# Precomputation - every per-frame input, built once for the whole recording
# --------------------------------------------------------------------------
def arm_path(paths, experiment, lam) -> Path:
    path = (paths.work_dir / "c28_lambda_grid_transfer" / experiment
            / f"C28-gated__{experiment}__lam{lam:g}.npz")
    if not path.is_file():
        raise SystemExit(
            f"no c28 arm at {path}. The stored grid is 0, 0.1, 0.25, 0.5, "
            f"0.767, 2, 8 - run c28 first for anything else.")
    return path


def build_state(paths, experiment, npz_path, panel, out_dir) -> Path:
    """Triangulate all three nodes per slot, project them, and cache the lot.

    Written once by the parent and memory-mapped by every worker, which is what
    makes a segmented render identical to a serial one: no worker computes
    anything that depends on where its segment starts.
    """
    arm = np.load(npz_path)
    source = arm["source"]                       # (F, slots, views) row within frame
    n_frames = source.shape[0]
    rig = load_rig(paths.asset_root / CALIBRATION)

    points, starts, slot_of = {}, {}, {}
    for view in VIEWS:
        detections = c01.load(paths.work_dir, experiment, view, RUN)
        if tuple(detections.nodes) != NODES:
            raise SystemExit(f"{view}: skeleton {detections.nodes} is not {NODES}")
        points[view], starts[view] = detections.points, detections.start
        rows = source[:, :, VIEWS.index(view)]
        valid = rows >= 0
        flat = starts[view][:, None] + np.where(valid, rows, 0)
        assigned = np.full(detections.frame_idx.size, -1, np.int8)
        for slot in range(N_ANIMALS):
            take = valid[:, slot]
            assigned[flat[take, slot]] = slot
        slot_of[view] = assigned
        print(f"  {view:<10} {int((assigned >= 0).sum()):,} of {assigned.size:,} "
              f"instances used by the correspondence")

    # (F, slots, nodes, 3): the same correspondence, all three nodes. Built in
    # frame blocks - `triangulate` forms a (..., n_rays, 3, 3) projector, which
    # over the whole recording at once is tens of gigabytes for no benefit.
    xyz = np.full((n_frames, N_ANIMALS, len(NODES), 3), np.nan, np.float32)
    block = 25_000
    for lo in range(0, n_frames, block):
        hi = min(lo + block, n_frames)
        span = hi - lo
        origins = np.empty((span, N_ANIMALS, len(NODES), len(VIEWS), 3))
        directions = np.empty_like(origins)
        present = np.zeros((span, N_ANIMALS, len(NODES), len(VIEWS)), bool)
        for column, view in enumerate(VIEWS):
            rows = source[lo:hi, :, column]
            valid = rows >= 0
            flat = starts[view][lo:hi, None] + np.where(valid, rows, 0)
            uv = np.where(valid[..., None, None],
                          points[view][flat], np.nan)       # (span, slots, nodes, 2)
            present[..., column] = np.isfinite(uv).all(axis=-1)
            ray_o, ray_d = rig.camera(view).rays(np.nan_to_num(uv).reshape(-1, 2))
            origins[..., column, :] = ray_o.reshape((*uv.shape[:-1], 3))
            directions[..., column, :] = ray_d.reshape((*uv.shape[:-1], 3))
        point, _ = triangulate(origins, directions, valid=present)
        xyz[lo:hi] = point.astype(np.float32)
    for node in range(len(NODES)):
        held = np.isfinite(xyz[:, :, node]).all(axis=-1)
        print(f"  {NODES[node]:<9} triangulated on {held.mean():.2%} of slot-frames")

    projected = panel.project(xyz)
    ok = np.isfinite(projected).all(axis=-1) & np.isfinite(xyz).all(axis=-1)
    # Off-panel points would wrap through int32; the tank bound makes that
    # impossible for a real fish, but a NaN-free outlier must still not draw.
    inside = (projected > -TILE).all(axis=-1) & (projected < 2 * TILE).all(axis=-1)
    ok &= inside
    projected = np.where(ok[..., None], projected, 0.0).astype(np.int32)

    state = out_dir / "c29_state.npz"
    np.savez(state, proj=projected, ok=ok, xyz=xyz,
             slot_track=arm["slot_track"], residual_mm=arm["residual_mm"],
             margin_mm=arm["margin_mm"], n_views_used=arm["n_views_used"],
             slot_rays=arm["slot_rays"],
             **{f"slot_of__{v}": slot_of[v] for v in VIEWS})
    return state


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
class Sources:
    """Everything one worker reads. Opened once, per process."""

    def __init__(self, paths, experiment, npz_path, state_path):
        self.paths, self.experiment = paths, experiment
        self.source_name = Path(npz_path).stem
        state = np.load(state_path, mmap_mode="r")
        self.proj, self.ok = state["proj"], state["ok"]
        self.slot_track = np.asarray(state["slot_track"])
        self.residual = np.asarray(state["residual_mm"])
        self.margin = np.asarray(state["margin_mm"])
        self.n_views = np.asarray(state["n_views_used"])
        self.n_gated = int(np.unique(self.slot_track[self.slot_track >= 0]).size)
        self.resolved_share = float((self.n_views >= 2).mean())
        arm = np.load(npz_path)
        self.detections, self.bounds, self.slot_of, self.gated = {}, {}, {}, {}
        for view in VIEWS:
            detections = c01.load(paths.work_dir, experiment, view, RUN)
            self.detections[view] = detections.points
            self.bounds[view] = (detections.start, detections.stop)
            self.slot_of[view] = np.asarray(state[f"slot_of__{view}"])
            self.gated[view] = arm[f"track__{view}"]
        self.panel = Panel()
        self.background = tank_background(self.panel)
        self.trail = round(TRAIL_SECONDS * FPS)

    def caption(self, frame: int) -> list[str]:
        seconds = frame / FPS
        clock = f"{int(seconds) // 60:02d}:{seconds % 60:06.3f}"
        residual = self.residual[frame]
        finite = residual[np.isfinite(residual)]
        return [
            f"{self.experiment}   frame {frame:,} / {self.n_views.size:,}   {clock}",
            f"{self.source_name}   gate 4.0 mm / 73 fr (post-hoc)   "
            f"nothing promoted",
            (f"resolved, {int(self.n_views[frame])} views   residual "
             f"{finite.mean():.2f} mm   margin {self.margin[frame]:.1f} mm"
             if self.n_views[frame] >= 2 and finite.size
             else "correspondence unresolved"),
            f"S = slot (3 ids, {self.resolved_share:.1%} of frames).  "
            f"t = gated track id ({self.n_gated} ids).",
        ]

    def compose(self, frame: int, images: dict) -> np.ndarray:
        tiles = {}
        for view in VIEWS:
            image = images[view]
            lo, hi = self.bounds[view][0][frame], self.bounds[view][1][frame]
            draw_view(image, self.detections[view][lo:hi],
                      self.slot_of[view][lo:hi], self.gated[view][lo:hi], view)
            tiles[view] = image
        tiles["reconstruction"] = draw_panel(
            self.background, self.proj, self.ok, self.slot_track, frame,
            self.trail, self.caption(frame))
        return np.vstack([np.hstack([tiles[a], tiles[b]]) for a, b in LAYOUT])


def open_captures(paths, experiment, frame: int) -> dict:
    captures = {}
    for view in VIEWS:
        capture = cv2.VideoCapture(str(paths.video(experiment, view)))
        if not capture.isOpened():
            raise SystemExit(f"cannot open {view}")
        if frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            landed = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if landed != frame:
                raise SystemExit(f"{view}: seek asked {frame}, landed {landed}")
        captures[view] = capture
    return captures


def read_frame(captures) -> dict:
    images = {}
    for view, capture in captures.items():
        ok, image = capture.read()
        if not ok:
            return {}
        images[view] = image                      # already BGR - nothing converts
    return images


def encoder(out_path: Path, fps: float, codec: str, quality: int):
    size = f"{2 * TILE}x{2 * TILE}"
    video = (["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
              "-rc", "vbr", "-cq", str(quality), "-b:v", "0"] if codec == "nvenc"
             else ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(quality)])
    command = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", size,
               "-r", f"{fps:g}", "-i", "-", "-an", *video,
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def render_segment(paths, experiment, npz_path, state_path, lo, hi, out_path,
                   fps, codec, quality, progress_every=2000) -> int:
    sources = Sources(paths, experiment, npz_path, state_path)
    captures = open_captures(paths, experiment, lo)
    process = encoder(out_path, fps, codec, quality)
    started, drawn = time.time(), 0
    try:
        for frame in range(lo, hi):
            images = read_frame(captures)
            if not images:
                print(f"  [{lo}-{hi}] decode stopped at {frame}", flush=True)
                break
            process.stdin.write(sources.compose(frame, images).tobytes())
            drawn += 1
            if drawn % progress_every == 0:
                rate = drawn / (time.time() - started)
                left = (hi - lo - drawn) / max(rate, 1e-6)
                print(f"  [{lo:,}-{hi:,}] {drawn:,}/{hi - lo:,}  "
                      f"{rate:.1f} fps  {left / 60:.0f} min left", flush=True)
    finally:
        process.stdin.close()
        process.wait()
        for capture in captures.values():
            capture.release()
    return drawn


def render_one(paths, experiment, npz_path, state_path, frame) -> np.ndarray:
    """One frame, drawn on its own - the reference `--check-segmenting` compares to."""
    sources = Sources(paths, experiment, npz_path, state_path)
    captures = open_captures(paths, experiment, frame)
    try:
        images = read_frame(captures)
        if not images:
            raise SystemExit(f"cannot decode frame {frame}")
        return sources.compose(frame, images)
    finally:
        for capture in captures.values():
            capture.release()


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check_alignment(paths, experiment, frames) -> bool:
    """`cv2.VideoCapture` frame N must be `sleap_io` frame N, channel-reversed.

    Everything drawn here is indexed by the detector's ``frame_idx``, which came
    from ``sleap_io``. A one-frame offset between the two readers would put every
    overlay on the wrong picture and look entirely plausible.
    """
    import sleap_io as sio
    good = True
    for view in VIEWS:
        path = paths.video(experiment, view)
        video = sio.Video.from_filename(str(path))
        capture = cv2.VideoCapture(str(path))
        for frame in frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, mine = capture.read()
            theirs = np.asarray(video[int(frame)])[..., ::-1]
            difference = float(np.abs(mine.astype(np.int16)
                                      - theirs.astype(np.int16)).mean())
            print(f"  {view:<10} frame {frame:>7,}  mean |diff| {difference:.4f}"
                  f"{'' if difference == 0 else '   MISMATCH'}")
            good &= bool(ok) and difference == 0.0
        capture.release()
    return good


def check_segmenting(paths, experiment, npz_path, state_path, boundary, back=64) -> bool:
    """A boundary frame drawn inside a segment and alone must be the same pixels."""
    alone = render_one(paths, experiment, npz_path, state_path, boundary)
    sources = Sources(paths, experiment, npz_path, state_path)
    captures = open_captures(paths, experiment, boundary - back)
    try:
        inside = None
        for frame in range(boundary - back, boundary + 1):
            images = read_frame(captures)
            inside = sources.compose(frame, images)
    finally:
        for capture in captures.values():
            capture.release()
    worst = int(np.abs(alone.astype(np.int16) - inside.astype(np.int16)).max())
    same = worst == 0
    print(f"  frame {boundary:,}: rendered from {back} frames back vs alone -> "
          f"{'identical' if same else 'DIFFERENT'} (max |diff| {worst})")
    return same


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--lam", type=float, default=None,
                        help="which c28 arm to draw. No default: c28 selects no "
                             "lambda and this tool does not select one either")
    parser.add_argument("--npz", type=Path, default=None,
                        help="render this reconstruction npz (c28 schema, e.g. "
                             "a c30 stitched arm) instead of naming a c28 arm "
                             "by --lam. Exactly one of the two must be given")
    parser.add_argument("--tag", default=None,
                        help="name for this run, used for the output directory "
                             "and the video's filename. Give one whenever the "
                             "npz's own filename does not identify the chain "
                             "that produced it - see the run-name rule above")
    parser.add_argument("--frames", nargs=2, type=int, default=None,
                        help="frame range; a partial render never lands where a "
                             "whole one does")
    parser.add_argument("--fps", type=float, default=float(FPS),
                        help="playback rate. The default is the source rate, so "
                             "the video runs in real time")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--scratch", type=Path, default=None,
                        help="where the per-worker segments are staged before "
                             "they are concatenated. Defaults to the output "
                             "directory; a local path keeps 2x the finished "
                             "size off the bucket, and the segments are deleted "
                             "either way")
    parser.add_argument("--codec", choices=("nvenc", "x264"), default="nvenc")
    parser.add_argument("--quality", type=int, default=21)
    parser.add_argument("--check-alignment", action="store_true")
    parser.add_argument("--check-segmenting", action="store_true")
    parser.add_argument("--segment", nargs=3, default=None,
                        help="internal: lo hi out_path, used by the parent's workers")
    parser.add_argument("--state", default=None,
                        help="internal: the parent's c29_state.npz")
    args = parser.parse_args()

    paths = load_paths()
    if (args.npz is None) == (args.lam is None):
        parser.error("give exactly one of --lam and --npz")
    npz_path = (Path(args.npz) if args.npz
                else arm_path(paths, args.experiment, args.lam))
    if not npz_path.is_file():
        raise SystemExit(f"no reconstruction npz at {npz_path}")

    if args.segment:
        lo, hi, out = int(args.segment[0]), int(args.segment[1]), Path(args.segment[2])
        render_segment(paths, args.experiment, npz_path, Path(args.state),
                       lo, hi, out, args.fps, args.codec, args.quality)
        return

    leaf = (args.experiment if not args.frames
            else f"{args.experiment}_frames{args.frames[0]}-{args.frames[1]}")
    # The run name must identify the *chain*, not just the file. `c30` writes
    # ``C30-gated__<experiment>__lam<x>.npz`` under a per-chain directory, so
    # two chains hand over npz files with identical names; deriving the output
    # directory from the stem alone landed the c28->c33->c30 render on top of
    # the c28->c30 one on 2026-09-01 and overwrote its state file. So: the
    # parent directory's name joins the stem, and --tag overrides both.
    if args.npz is None:
        run_name = f"lam{args.lam:g}"
    elif args.tag:
        run_name = args.tag
    else:
        parent = npz_path.parent.name
        run_name = f"{parent}__{npz_path.stem}" if parent else npz_path.stem
    out_dir = paths.work_dir / "c29_full_render" / run_name / leaf
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("full_render", out_dir, name="c29") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        record.param("stage_a", "c27 serial (NOT c24 chunked)")
        record.param("lambda_is_chosen", False)
        record.param("run_name", run_name)
        record.param("id_drawn", "slot (3 ids); gated track id annotated small")
        record.param("tail_node_is_display_only", True)
        record.input("correspondence", npz_path)

        panel = Panel()
        started = time.time()
        state = build_state(paths, args.experiment, npz_path, panel, out_dir)
        print(f"state built in {time.time() - started:.0f}s -> {state.name}")
        record.output("state", state)

        n_frames = int(np.load(state, mmap_mode="r")["slot_track"].shape[0])
        lo, hi = args.frames if args.frames else (0, n_frames)

        if args.check_alignment:
            print("\nreader alignment (cv2 vs sleap_io):")
            ok = check_alignment(paths, args.experiment,
                                 [lo, (lo + hi) // 2, hi - 1])
            record.param("reader_alignment_exact", ok)
            if not ok:
                raise SystemExit("readers disagree - nothing rendered")

        if args.check_segmenting:
            print("\nsegment equivalence:")
            same = check_segmenting(paths, args.experiment, npz_path, state,
                                    (lo + hi) // 2)
            record.param("segmenting_is_exact", same)
            if not same:
                raise SystemExit("segments are not equivalent - nothing rendered")

        workers = max(1, min(args.workers, (hi - lo) // 200 or 1))
        edges = np.linspace(lo, hi, workers + 1).round().astype(int)
        staging = Path(args.scratch) if args.scratch else out_dir
        staging.mkdir(parents=True, exist_ok=True)
        segments = [(int(a), int(b), staging / f"segment_{i:03d}.mp4")
                    for i, (a, b) in enumerate(pairwise(edges))]
        print(f"\n{hi - lo:,} frames, {workers} workers, "
              f"{2 * TILE}x{2 * TILE} at {args.fps:g} fps, codec {args.codec}")

        started = time.time()
        running = []
        for a, b, path in segments:
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--experiment", args.experiment,
                       "--npz", str(npz_path), "--tag", run_name,
                       "--fps", str(args.fps), "--codec", args.codec,
                       "--quality", str(args.quality),
                       "--state", str(state),
                       "--segment", str(a), str(b), str(path)]
            running.append(subprocess.Popen(command))
        codes = [p.wait() for p in running]
        if any(codes):
            raise SystemExit(f"worker exit codes {codes}")
        elapsed = time.time() - started
        print(f"\nrendered in {elapsed / 60:.1f} min "
              f"({(hi - lo) / elapsed:.1f} fps)")

        listing = staging / "segments.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for _, _, p in segments),
                           encoding="utf-8")
        final = out_dir / (f"c29__{args.experiment}__{run_name}"
                           f"__{2 * TILE}x{2 * TILE}__{args.fps:g}fps.mp4")
        subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", str(final)], check=True)
        for _, _, path in segments:
            path.unlink()
        listing.unlink()
        record.output("video", final)
        record.param("frames_rendered", int(hi - lo))
        record.param("render_seconds", elapsed)
        print(f"\nwrote {final}\n{final.stat().st_size / 1024 ** 3:.2f} GB   "
              f"{(hi - lo) / args.fps / 60:.1f} min of video")
        (out_dir / "c29_summary.json").write_text(json.dumps({
            "experiment": args.experiment, "n_groups": 1, "n_experiments": 1,
            "stage_a": "c27 serial", "correspondence": str(npz_path),
            "run_name": run_name, "lambda_is_chosen": False,
            "frames": [int(lo), int(hi)], "fps": args.fps,
            "resolution": [2 * TILE, 2 * TILE], "layout": "2x2",
            "id_drawn": "slot", "gated_track_id_annotated": True,
            "tail_node_is_display_only": True,
            "trail_seconds": TRAIL_SECONDS, "trail_node": "pectoral",
            "video": str(final), "size_bytes": final.stat().st_size,
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

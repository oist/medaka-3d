"""C31 - the doubt channel: per-frame signals that concentrate the known failures.

=============================================================================
WHAT THIS IS, AND WHAT IT IS NOT
=============================================================================

Step D of the 2026-08-26 plan, built after the experimenter's frame-by-frame
review of the `c29` render diagnosed four failure scenes and every one of them
turned out to leave a signature in quantities the pipeline already stores.
Measured before this tool was written (K_1_2, stitched lambda 0.767, halo
+-2 frames):

    residual > 3 mm on a 3-ray slot   fires 0.35%   captures 93.7% of spikes
    slot ray-count changed            fires 3.2%    captures 94.8% of exchanges
    view best score < 1.5             fires 0.61%   captures 61.0% of exchanges
    margin < 1.5 mm                   fires 7.6%    captures 87.1% of exchanges
    OR of the four                    fires 12.0%   captures 100% of both

**This tool changes nothing.** It reads a reconstruction npz and the `c01`
cache, writes flags, and ranks contiguous doubt zones for human review. It is
the instrument the 2026-08-26 followup promoted: where there is no ground
truth, a per-frame confidence is not a router - it is how a run on an
unannotated experiment can be called good or bad at all.

**Thresholds are calibrated on K_1_2 and are hypotheses anywhere else.**
`n_groups = 1`. The 1.5 score floor, in particular, is read off one calm
stretch (best-score median 1.94, p10 1.79) against one wave scene (1.13-1.29).

Residual is blind on 2-ray slots - a two-camera triangulation has no
redundancy, so `residual_spike` deliberately requires 3 rays. The 30735 class
(a mis-assigned ray on a 2-ray frame) is carried by `margin_low`, `ray_change`
and the per-view occlusion counts instead.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c31_doubt_channel.py --npz <reconstruction npz>
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01

from medaka3d import load_paths
from medaka3d.provenance import provenance
from medaka3d.types import FPS, VIEWS

RUN = "bottomup"

#: Fixed in `decisions/2026-09-01-doubt-channel.md`, before the first run.
RESIDUAL_SPIKE_MM = 3.0
MARGIN_LOW_MM = 1.5
SCORE_DEGRADED = 1.5
SPIKE_STEP_MM = 10.0
SPIKE_RETURN_MM = 3.0
EXCHANGE_MARGIN_MM = 5.0

#: Body-length wobble floor: |mouth-tail length - its 2D track's rolling
#: median| / median. Proposed by the experimenter on 2026-09-03 ("the body
#: length is also very fluctuating" at the 31479 wave scene) and measured
#: before adoption: calm-stretch dev median 0.024-0.033 / p99 0.34-0.48 per
#: view; at 0.3 the any-view signal fires on 5.0% of frames and captures 83.4%
#: of exchange boundaries - more than the score floor's 74.1%. It sees
#: *malformed* inference (part mixing, node switching, ripple distortion, two
#: animals fused into one detection); the score floor sees *unconfident*
#: inference; a stable detection of the wrong animal (the 30735 class) is
#: invisible to both, deliberately left to the occlusion counts.
WOBBLE_DEV = 0.3
#: Rolling-median window for the wobble, in samples of one 2D track (~0.15 s).
#: Short enough that a genuine turn toward the camera moves the median with
#: the fish; a detector glitch cannot.
WOBBLE_WINDOW = 25

#: Flagged frames closer than this merge into one zone. 1 s at 166 fps.
ZONE_GAP = 166

PERMS = tuple(permutations(range(3)))


def best_scores(detections, n_frames: int) -> np.ndarray:
    """(n_frames,) the best instance score per frame, -1 where nothing detected."""
    best = np.full(n_frames, -1.0)
    np.maximum.at(best, detections.frame_idx, detections.score)
    return best


def view_degraded(detections, n_frames: int) -> np.ndarray:
    """(n_frames,) bool - the view detected something and its best is below the floor.

    A view with no detections is absent, not degraded: absence is already
    handled by the correspondence's own missing-node machinery, and demoting
    it twice would double-count.
    """
    best = best_scores(detections, n_frames)
    return (best >= 0) & (best < SCORE_DEGRADED)


def view_wobble(detections, labels: np.ndarray, n_frames: int) -> np.ndarray:
    """(n_frames,) the worst body-length deviation among this view's tracked
    instances, as a fraction of that instance's own rolling median length.

    Length is mouth to tail in pixels. The tail is the least reliable node,
    which is the point: this is a detector-quality signal, not a measurement
    of the animal.
    """
    import pandas as pd

    length = np.linalg.norm(detections.points[:, 0] - detections.points[:, 2],
                            axis=-1)
    table = pd.DataFrame({"frame": detections.frame_idx, "label": labels,
                          "length": length})
    table = table[(table.label >= 0) & np.isfinite(table.length)]
    table = table.sort_values(["label", "frame"])
    median = table.groupby("label")["length"].transform(
        lambda run: run.rolling(WOBBLE_WINDOW, center=True,
                                min_periods=5).median())
    deviation = (table.length - median).abs() / median
    out = np.zeros(n_frames)
    np.maximum.at(out, table.frame.values, np.nan_to_num(deviation.values))
    return out


def spikes(xyz: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """(n_frames,) bool - out-and-back excursions, per the pre-registered rule."""
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
    back = np.linalg.norm(xyz[2:] - xyz[:-2], axis=-1) < SPIKE_RETURN_MM
    hit = (step[:-1] > SPIKE_STEP_MM) & (step[1:] > SPIKE_STEP_MM) & back
    out = np.zeros(xyz.shape[0], bool)
    out[1:-1] = hit.any(axis=1)
    return out & ok


def exchange_boundaries(xyz: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """(n_frames,) bool - a non-identity slot assignment is decisively cheaper."""
    idx = np.flatnonzero(ok)
    a, b = xyz[idx[:-1]], xyz[idx[1:]]
    d = np.linalg.norm(a[:, :, None, :] - b[:, None, :, :], axis=-1)
    costs = np.stack(
        [np.nan_to_num(d[:, (0, 1, 2), pm], nan=0.0).sum(axis=1) for pm in PERMS],
        axis=1)
    decisive = ((costs.argmin(axis=1) != 0)
                & (costs[:, 0] - costs.min(axis=1) > EXCHANGE_MARGIN_MM))
    out = np.zeros(xyz.shape[0], bool)
    out[idx[1:][decisive]] = True
    return out


def zones(flags: np.ndarray, severity: np.ndarray, gap: int = ZONE_GAP) -> list[dict]:
    """Contiguous doubt zones, most severe first - the review queue."""
    frames = np.flatnonzero(flags)
    if not frames.size:
        return []
    breaks = np.flatnonzero(np.diff(frames) > gap)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks, [frames.size - 1]])
    out = []
    for lo, hi in zip(starts, stops, strict=True):
        first, last = int(frames[lo]), int(frames[hi])
        out.append({
            "first_frame": first, "last_frame": last,
            "seconds": round(first / FPS, 1),
            "flagged_frames": int(hi - lo + 1),
            "severity": round(float(severity[first:last + 1].max()), 2),
        })
    out.sort(key=lambda z: -z["severity"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--npz", type=Path, required=True,
                        help="a reconstruction npz in c28's schema")
    parser.add_argument("--experiment", default="20251225_K_1_2_FCMC")
    args = parser.parse_args()

    paths = load_paths()
    arm = np.load(args.npz)
    xyz = arm["xyz"].astype(float)
    residual, margin = arm["residual_mm"], arm["margin_mm"]
    rays, n_views = arm["slot_rays"], arm["n_views_used"]
    n_frames = xyz.shape[0]
    ok = (n_views >= 2) & np.isfinite(xyz).all(axis=(1, 2))

    out_dir = paths.work_dir / "c31_doubt" / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("doubt_channel", out_dir, name="c31") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        record.param("thresholds", {
            "residual_spike_mm": RESIDUAL_SPIKE_MM, "margin_low_mm": MARGIN_LOW_MM,
            "score_degraded": SCORE_DEGRADED, "spike_step_mm": SPIKE_STEP_MM,
            "spike_return_mm": SPIKE_RETURN_MM,
            "exchange_margin_mm": EXCHANGE_MARGIN_MM})
        record.param("thresholds_are_k12_calibrated", True)
        record.input("reconstruction", args.npz)

        flags = {}
        flags["residual_spike"] = (
            (residual > RESIDUAL_SPIKE_MM) & (rays == 3)).any(axis=1)
        flags["margin_low"] = margin < MARGIN_LOW_MM
        change = np.zeros(n_frames, bool)
        change[1:] = (np.diff(rays.astype(int), axis=0) != 0).any(axis=1)
        flags["ray_change"] = change

        label_dir = (paths.work_dir / "c27_uncapped_tracklets_any"
                     / args.experiment / "mouth-pectoral_oks_serial")
        degraded, wobbled = {}, {}
        for view in VIEWS:
            detections = c01.load(paths.work_dir, args.experiment, view, RUN)
            degraded[view] = view_degraded(detections, n_frames)
            flags[f"view_degraded__{view}"] = degraded[view]
            labels = np.load(label_dir / f"track2d__{view}.npy")
            wobbled[view] = view_wobble(detections, labels, n_frames)
            flags[f"view_wobble__{view}"] = wobbled[view] > WOBBLE_DEV
        any_degraded = np.stack(list(degraded.values())).any(axis=0)
        any_wobble = np.stack([flags[f"view_wobble__{v}"] for v in VIEWS]).any(axis=0)

        flags["spike"] = spikes(xyz, ok)
        flags["exchange_boundary"] = exchange_boundaries(xyz, ok)
        doubt = (flags["residual_spike"] | flags["margin_low"]
                 | flags["ray_change"] | any_degraded | any_wobble)
        flags["doubt"] = doubt
        record.param("doubt_version", 2)
        record.param("doubt_v2_adds", "view_wobble (annotator-proposed)")

        # Severity for ranking zones: worst step or residual inside the zone,
        # with an exchange boundary dominating everything - identity is the
        # failure that corrupts scoring, a fat residual only blurs it.
        step = np.zeros(n_frames)
        step[1:] = np.nan_to_num(
            np.linalg.norm(np.diff(xyz, axis=0), axis=-1), nan=0.0).max(axis=1)
        severity = np.maximum(step, np.nan_to_num(residual, nan=0.0).max(axis=1))
        severity[flags["exchange_boundary"]] += 1_000.0

        queue = zones(doubt, severity)
        # The parent directory joins the stem: `c30` writes identically named
        # arms under per-chain directories, and the stem alone made the second
        # chain's doubt file overwrite the first's on 2026-09-01 (same fault
        # `c29` had, fixed the same way).
        leaf = (f"{args.npz.parent.name}__{args.npz.stem}"
                if args.npz.parent.name else args.npz.stem)
        path = out_dir / f"doubt__{leaf}.npz"
        np.savez_compressed(path, **flags)
        record.output("doubt", path)

        print(f"{args.npz.stem}   {n_frames:,} frames")
        for name, flag in flags.items():
            print(f"  {name:28s} fires on {flag[ok].mean():7.3%} of resolved frames")
        print(f"\n  doubt zones: {len(queue)}   "
              f"frames inside one: {int(doubt.sum()):,}")
        print("  worst ten (severity > 1000 means an exchange is inside):")
        for zone in queue[:10]:
            print(f"    f{zone['first_frame']:>7,}-{zone['last_frame']:>7,}  "
                  f"{zone['seconds']:7.1f}s  severity {zone['severity']:8.1f}  "
                  f"{zone['flagged_frames']} flagged")

        (out_dir / f"doubt__{leaf}.json").write_text(
            json.dumps({"experiment": args.experiment, "n_groups": 1,
                        "input": str(args.npz),
                        "fires": {k: float(v[ok].mean()) for k, v in flags.items()},
                        "zones": queue}, indent=2), encoding="utf-8")
        record.output("zones", out_dir / f"doubt__{leaf}.json")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

"""C03 — reconstruct the s16 detections in 3D and run three trackers over them.

=============================================================================
WHY 3D, decided by tools/c02_tracking_probe.py before this file was written
=============================================================================

`c02` measured the two correspondence problems against each other on 15,000
frames. Its **pre-registered rule was degenerate** — both temporal ambiguity
rates came out at exactly 0.000%, so the rule chose 2D on which side of a tie the
comparison fell — and the choice was made on the post-hoc measurements it added,
plus one structural fact. Recorded in
`docs/decisions/2026-08-13-tracking-3d-gated.md`; in short:

* frame-to-frame linking is **not** the hard part in either representation
  (median winner 4.2 px against a 245 px runner-up in 2D; 1.1 mm against 79.0 mm
  in 3D), so a bare Hungarian is already close to right in both;
* **coverage is the real difference.** One camera reports three detections in
  94.2-96.6% of frames; at least two cameras do in **98.96%** on `c02`'s block
  sample (95.9-97.1% over the whole recording, which includes the handling
  period), and 76-90% of the frames where a view loses an animal have both other
  views intact. Those are exactly the frames where a 2D track breaks and a 3D
  track need not;
* the cross-view correspondence is **not a cost 3D pays and 2D avoids.** The
  project needs 3D output — encounters are defined in mm — so both routes must
  solve it. 3D solves it per frame on raw detections, where the winner beats the
  runner-up by 12x (1.27 mm against 15.38 mm); the 2D route defers it until after
  three view-trackers have already broken independently, and then has less to
  work with.

This tool therefore reconstructs first and tracks second. It does **not**
implement a 2D route: the brief asked for one, and building both would make
neither answerable.

=============================================================================
THE THREE METHODS AND THE THRESHOLDS THEY USE
=============================================================================

`medaka3d.track` holds them; each differs from the last by one mechanism —
`nearest` (Hungarian, no gate, no memory), `gated` (+ distance gate + gap
bridging), `predictive` (+ constant-velocity prediction). They are three
functions, not one function with a flag, because design rule 5 forbids a flag
that changes which algorithm runs.

**Every threshold is derived here, from this data, and printed. None is
hardcoded.**

* ``max_step_mm`` — the p99.9 of the per-animal 3D mouth displacement between
  adjacent resolved frames under the geometrically forced assignment. Derived
  before any tracking runs, so no tracker's output can influence its own gate.
  It came out at 10.2 mm against a median step of 0.36 mm, which is what a
  distribution with a 204 mm tail of detector failures looks like.
* ``max_gap_frames`` — the p99 of the observed run-lengths of unresolved frames,
  so bridging covers the gaps that actually occur and does not invent links
  across the long ones.
* ``velocity_frames = 3`` — 18 ms, long enough to average out ~0.4 mm of
  detector noise and short enough not to lag a turn. This one is a **choice, not
  a derivation**, and is stated as such.

The same three thresholds are given to every method that takes them, so a
difference between methods is the mechanism and never the tuning.

**n_groups = 1.** Group K, one trio, three consecutive days. Whatever this
measures about tracking, it measures on one of the corpus's 14 groups.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c03_track_3d.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from c02_tracking_probe import CALIBRATION

from medaka3d import load_paths
from medaka3d.calib import load_rig
from medaka3d.cohort import ExperimentId
from medaka3d.eval.tracking import PERMUTATIONS
from medaka3d.provenance import provenance
from medaka3d.reconstruct import N_ANIMALS, reconstruct
from medaka3d.track import METHODS, TrackConfig, propagate_to_views
from medaka3d.types import VIEWS

#: Percentile of observed per-animal motion the link gate must admit.
GATE_PERCENTILE = 99.9
#: Percentile of observed unresolved-run lengths that bridging must cover.
GAP_PERCENTILE = 99.0
#: Frames over which `predictive` estimates velocity. A choice, not a derivation.
VELOCITY_FRAMES = 3


def frame_arrays(detections: c01.Detections) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """``node -> (n_frames, 3, 2)`` pixels, and the (n_frames,) complete mask.

    Rows for frames that did not report exactly three instances are NaN, not
    absent: the reconstruction indexes by frame and a shifted array would pair
    one frame's pixels with another frame's index.

    **A view counts as complete only if all three mouths are finite.** Three
    instances is not enough: the bottom-up model leaves a mouth unplaced in
    2.9-4.2% of otherwise complete frames per view, and one missing mouth makes
    every one of the 36 correspondence hypotheses non-finite — so without this
    the whole frame is lost, in all three cameras, because one camera missed one
    node. Excluding the view instead lets the other two carry the frame.
    """
    counts = detections.counts()
    three = counts == N_ANIMALS
    start = np.searchsorted(detections.frame_idx, np.flatnonzero(three), side="left")
    rows = start[:, None] + np.arange(N_ANIMALS)[None, :]

    out = {}
    for node in detections.nodes:
        xy = np.full((detections.n_frames, N_ANIMALS, 2), np.nan)
        xy[three] = detections.node(node)[rows]
        out[node] = xy
    return out, three & np.isfinite(out["mouth"]).all(axis=(1, 2))


def run_lengths(flags: np.ndarray) -> np.ndarray:
    """Lengths of the maximal runs of True."""
    if not flags.any():
        return np.zeros(0, dtype=int)
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[1::2] - edges[::2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--experiments", nargs="*", default=list(c01.EXPERIMENTS))
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--methods", nargs="*", default=sorted(METHODS))
    parser.add_argument("--run", choices=sorted(c01.RUNS), default=c01.DEFAULT_RUN,
                        help="Which detection run to reconstruct and track.")
    args = parser.parse_args()

    paths = load_paths()
    out_dir = paths.work_dir / "c03_track_3d" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    rig = load_rig(args.calibration)

    with provenance("track_3d", out_dir, name=f"c03_{args.run}") as record:
        record.params_from(args)
        record.input("calibration", rig.manifest)
        groups = sorted({ExperimentId.parse(e).group for e in args.experiments})
        record.param("n_groups", len(groups))
        record.param("run", args.run)
        print(f"detections  : {args.run}")
        print(f"calibration : {rig.manifest.parent.name}")
        print(f"views       : {rig.views}")
        print(f"n_groups    : {len(groups)} {groups} — one trio, three consecutive days\n")

        # --- reconstruct ---------------------------------------------------
        reconstructions, nodes, starts, counts = {}, {}, {}, {}
        for experiment in args.experiments:
            started = time.time()
            uv, complete, by_node = {}, {}, {}
            frame_start, n_instances = {}, {}
            for view in VIEWS:
                det = c01.load(paths.work_dir, experiment, view, args.run)
                by_node[view], complete[view] = frame_arrays(det)
                uv[view] = by_node[view]["mouth"]
                frame_start[view] = np.searchsorted(
                    det.frame_idx, np.arange(det.n_frames), side="left"
                )
                n_instances[view] = int(det.frame_idx.shape[0])
            # Frames the detector supplied enough views for, before geometry has
            # its say. The difference against `resolved` is triangulation
            # failing on detections that were present — a distinct problem from
            # a missing detection, and one that would be invisible if only the
            # final rate were printed.
            enough = sum(complete[v].astype(int) for v in VIEWS) >= 2
            out = reconstruct(rig, experiment, uv, complete, c01.N_FRAMES)
            reconstructions[experiment] = out
            nodes[experiment] = by_node
            starts[experiment] = frame_start
            counts[experiment] = n_instances
            resolved = out.resolved
            record.param(f"coverage_{experiment}", {
                "two_views_available": float(enough.mean()),
                "resolved": float(resolved.mean()),
                "three_views_used": float((out.n_views_used == 3).mean()),
                "two_views_used": float((out.n_views_used == 2).mean()),
                "triangulation_failed": int((enough & ~resolved).sum()),
                "residual_median_mm": float(np.nanmedian(out.residual_mm)),
                "margin_median_mm": float(np.nanmedian(out.margin_mm)),
            })
            print(
                f"  {experiment}  >=2 views {100 * enough.mean():5.2f}%  "
                f"resolved {100 * resolved.mean():5.2f}%  "
                f"(3 views {100 * (out.n_views_used == 3).mean():5.2f}%, "
                f"2 views {100 * (out.n_views_used == 2).mean():5.2f}%, "
                f"triangulation failed {int((enough & ~resolved).sum()):,})  "
                f"residual median {np.nanmedian(out.residual_mm):.3f} mm  "
                f"margin median {np.nanmedian(out.margin_mm):.2f} mm  "
                f"{time.time() - started:.0f}s"
            )

        # --- thresholds, derived before any tracker runs -------------------
        steps, gaps = [], []
        for experiment in args.experiments:
            out = reconstructions[experiment]
            resolved = out.resolved
            adjacent = np.flatnonzero(resolved[:-1] & resolved[1:])
            # The forced assignment, not a tracker's: slots are already aligned
            # frame to frame only by geometry at this point.
            here, there = out.xyz[adjacent], out.xyz[adjacent + 1]
            cost = np.linalg.norm(here[:, :, None, :] - there[:, None, :, :], axis=-1)
            # With three animals the minimum over the six permutations *is* the
            # Hungarian answer, and enumerating keeps the per-animal terms that a
            # solver would collapse into one total.
            totals = np.stack([cost[:, [0, 1, 2], p].sum(axis=1) for p in PERMUTATIONS], axis=-1)
            perms = np.asarray(PERMUTATIONS)[np.argmin(totals, axis=-1)]
            steps.append(
                cost[np.arange(cost.shape[0])[:, None], np.arange(3)[None, :], perms].ravel()
            )
            gaps.append(run_lengths(~resolved))

        step = np.concatenate(steps)
        gap = np.concatenate(gaps)
        max_step_mm = float(np.round(np.percentile(step, GATE_PERCENTILE), 1))
        max_gap_frames = int(np.percentile(gap, GAP_PERCENTILE)) if gap.size else 0

        config = TrackConfig(
            max_step_mm=max_step_mm,
            max_gap_frames=max_gap_frames,
            velocity_frames=VELOCITY_FRAMES,
        )
        print("\nthresholds — derived here, from this data, before any tracker ran")
        print(
            f"  per-animal step        median {np.median(step):6.3f} mm   "
            f"p99 {np.percentile(step, 99):6.3f}   p{GATE_PERCENTILE} "
            f"{np.percentile(step, GATE_PERCENTILE):6.3f}   max {step.max():7.2f}"
        )
        print(f"  max_step_mm            {max_step_mm} mm")
        print(
            f"  unresolved run length  n {gap.size:,}  median {np.median(gap):.0f}  "
            f"p{GAP_PERCENTILE:.0f} {np.percentile(gap, GAP_PERCENTILE):.0f}  max {gap.max()}"
        )
        print(f"  max_gap_frames         {max_gap_frames} frames "
              f"({1000 * max_gap_frames / 166:.0f} ms)")
        print(f"  velocity_frames        {VELOCITY_FRAMES} (a choice, not a derivation)")
        record.param("max_step_mm", max_step_mm)
        record.param("max_gap_frames", max_gap_frames)
        record.param("velocity_frames", VELOCITY_FRAMES)
        record.param("step_percentiles", {
            "median": float(np.median(step)),
            "p99": float(np.percentile(step, 99)),
            "p99.9": float(np.percentile(step, 99.9)),
            "max": float(step.max()),
        })

        # --- track ---------------------------------------------------------
        summary: dict[str, dict] = {}
        for method in args.methods:
            print(f"\n--- {method}")
            for experiment in args.experiments:
                started = time.time()
                out = reconstructions[experiment]
                slot_track = METHODS[method](out, config)
                per_view = propagate_to_views(
                    out, slot_track, starts[experiment], counts[experiment]
                )
                path = out_dir / f"{method}__{experiment}.npz"
                np.savez_compressed(
                    path,
                    slot_track=slot_track,
                    xyz=out.xyz.astype(np.float32),
                    residual_mm=out.residual_mm.astype(np.float32),
                    margin_mm=out.margin_mm.astype(np.float32),
                    n_views_used=out.n_views_used,
                    source=out.source,
                    **{f"track__{v}": per_view[v] for v in VIEWS},
                    **{
                        f"{node}__{v}": nodes[experiment][v][node].astype(np.float32)
                        for v in VIEWS
                        for node in ("pectoral", "tail")
                    },
                )
                record.output(f"{method}__{experiment}", path)
                live = slot_track[slot_track >= 0]
                summary.setdefault(method, {})[experiment] = {
                    "tracks": int(np.unique(live).size),
                    "tracks_per_animal": float(np.unique(live).size / N_ANIMALS),
                    "assigned": int(live.size),
                }
                print(
                    f"  {experiment}  {np.unique(live).size:>7,} tracks  "
                    f"({np.unique(live).size / N_ANIMALS:8.1f} per animal)  "
                    f"{time.time() - started:5.0f}s"
                )
        record.param("summary", summary)
        result = {
            "n_groups": len(groups),
            "groups": groups,
            "max_step_mm": max_step_mm,
            "max_gap_frames": max_gap_frames,
            "velocity_frames": VELOCITY_FRAMES,
            "summary": summary,
        }
        (out_dir / "c03_tracking.json").write_text(
            json.dumps(result, indent=2, default=float), encoding="utf-8"
        )
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

"""C13 — anchor the cross-view correspondence on the pectoral, and see if the
third camera comes back.

=============================================================================
THE PROBLEM THIS ADDRESSES
=============================================================================

Two cameras triangulate *any* correspondence: two rays always come close in the
least-squares sense, so the residual says nothing about whether the pairing was
right. The third camera is what makes a wrong pairing impossible to fit. That
redundancy is the reason this rig has three cameras rather than a stereo pair.

Measured on K_1_2 with the current mouth anchor (2026-08-18):

| cameras | frames | share | residual median | **runner-up margin p1** |
|---|---|---|---|---|
| 3 | 262,751 | 75.4% | 0.420 mm | **1.110 mm** |
| 2 | 71,747 | **20.6%** | 0.228 mm | **0.009 mm** |

On two-view frames the correct correspondence and its runner-up are 0.009 mm
apart at the first percentile — 120x closer than on three-view frames. **The
check is absent for a fifth of the recording**, and the smaller residual is not
a better fit, it is fewer constraints. The eight measured female swaps sit
there: median 43.7% two-view frames inside a swap interval against 20.1%
elsewhere.

**Why a camera drops out is a one-node problem.** `c03.frame_arrays` marks a
view complete only if all three **mouths** are finite, and bottom-up leaves a
mouth unplaced in 1.065-1.507% of instances (2026-08-15). One missing node costs
the whole camera for that frame, and with it the correspondence check. The
pectoral is missing in **0.000%** of complete frames (2026-08-13), and switching
to it was recommended in that decision and never done.

=============================================================================
WHAT THIS RUNS, AND WHAT IT DOES NOT TOUCH
=============================================================================

A separate command rather than a flag on `c03`, per design rule 5. It rebuilds
the reconstruction for one experiment with ``--anchor`` as the correspondence
node, tracks it with the same `gated` method and thresholds derived the same
way, and writes an npz in `c03`'s format so `c09` can score it unchanged.

`c03`'s own output is not modified. Nothing is promoted here; the decision rule
in ``docs/decisions/2026-08-18-2d-first-tracking.md`` decides, and it needs
`c09`'s swap count, which is a separate run against the same female annotation.

**n_groups = 1, n_experiments = 1.**

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c13_pectoral_anchor.py
    .venv\\Scripts\\python.exe tools/c13_pectoral_anchor.py --anchor mouth   # baseline
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
from c03_track_3d import GAP_PERCENTILE, GATE_PERCENTILE, VELOCITY_FRAMES, run_lengths

from medaka3d import load_paths
from medaka3d.calib import load_rig
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.reconstruct import N_ANIMALS, PERMUTATIONS, reconstruct
from medaka3d.track import METHODS, TrackConfig, propagate_to_views
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN, METHOD = "bottomup", "gated"


def frame_arrays(detections, anchor: str):
    """``node -> (n_frames, 3, 2)`` pixels, and completeness judged on ``anchor``.

    `c03`'s version hardcodes the mouth. Here the node that decides whether a
    view takes part is the same node the correspondence is solved on, which is
    the only self-consistent choice: a view is complete exactly when it can
    supply the quantity being matched.
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
    return out, three & np.isfinite(out[anchor]).all(axis=(1, 2))


def report(npz: dict, label: str) -> dict:
    """Coverage, camera count, and how forced the correspondence was."""
    views_used = npz["n_views_used"]
    residual = npz["residual_mm"]
    margin = npz["margin_mm"]
    resolved = views_used >= 2
    stats = {"label": label, "frames": int(views_used.size),
             "resolved": float(resolved.mean())}
    for k in (2, 3):
        sel = resolved & (views_used == k)
        m = margin[sel]
        m = m[np.isfinite(m)]
        r = residual[sel]
        r = r[np.isfinite(r)]
        stats[f"views_{k}"] = {
            "frames": int(sel.sum()), "share": float(sel.mean()),
            "residual_median": float(np.median(r)) if r.size else float("nan"),
            "margin_p1": float(np.percentile(m, 1)) if m.size else float("nan"),
            "margin_p10": float(np.percentile(m, 10)) if m.size else float("nan"),
            "margin_median": float(np.median(m)) if m.size else float("nan"),
            "margin_under_1mm": float(np.mean(m < 1.0)) if m.size else float("nan"),
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--anchor", default="pectoral", choices=("pectoral", "mouth", "tail"))
    parser.add_argument("--experiment", default=EXPERIMENT)
    args = parser.parse_args()

    paths = load_paths()
    rig = load_rig(CALIBRATION)
    out_dir = paths.work_dir / "c13_pectoral_anchor" / args.anchor
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("pectoral_anchor", out_dir, name=f"c13_{args.anchor}") as record:
        record.params_from(args)
        record.input("calibration", rig.manifest)
        record.param("n_groups", 1)
        record.param("anchor", args.anchor)
        print(f"experiment : {args.experiment}   anchor : {args.anchor}   n_groups = 1")

        started = time.time()
        uv, complete, nodes, starts, counts = {}, {}, {}, {}, {}
        for view in VIEWS:
            detections = c01.load(paths.work_dir, args.experiment, view, RUN)
            nodes[view], complete[view] = frame_arrays(detections, args.anchor)
            uv[view] = nodes[view][args.anchor]
            starts[view] = np.searchsorted(
                detections.frame_idx, np.arange(detections.n_frames), side="left")
            counts[view] = int(detections.frame_idx.shape[0])
            print(f"  {view:<10} complete {complete[view].mean():6.2%}")
        record.param("complete_per_view", {v: float(complete[v].mean()) for v in VIEWS})

        out = reconstruct(rig, args.experiment, uv, complete, c01.N_FRAMES)
        print(f"  reconstructed in {time.time() - started:.0f}s   "
              f"resolved {out.resolved.mean():.2%}")

        # Thresholds derived from this reconstruction, the same way `c03` does,
        # because a gate carried over from the mouth anchor would be a threshold
        # tuned on other data.
        frames = np.flatnonzero(out.resolved)
        adjacent = frames[:-1][np.diff(frames) == 1]
        here, there = out.xyz[adjacent], out.xyz[adjacent + 1]
        cost = np.linalg.norm(here[:, :, None, :] - there[:, None, :, :], axis=-1)
        totals = np.stack([cost[:, [0, 1, 2], p].sum(axis=1) for p in PERMUTATIONS], axis=-1)
        perms = np.asarray(PERMUTATIONS)[np.argmin(totals, axis=-1)]
        step = cost[np.arange(cost.shape[0])[:, None], np.arange(3)[None, :], perms].ravel()
        gap = run_lengths(~out.resolved)
        config = TrackConfig(
            max_step_mm=float(np.round(np.percentile(step, GATE_PERCENTILE), 1)),
            max_gap_frames=int(np.percentile(gap, GAP_PERCENTILE)) if gap.size else 0,
            velocity_frames=VELOCITY_FRAMES,
        )
        print(f"  max_step_mm {config.max_step_mm}   max_gap_frames {config.max_gap_frames}")
        record.param("max_step_mm", config.max_step_mm)
        record.param("max_gap_frames", config.max_gap_frames)

        slot_track = METHODS[METHOD](out, config)
        per_view = propagate_to_views(out, slot_track, starts, counts)
        path = out_dir / f"{METHOD}__{args.experiment}.npz"
        np.savez_compressed(
            path,
            slot_track=slot_track,
            xyz=out.xyz.astype(np.float32),
            residual_mm=out.residual_mm.astype(np.float32),
            margin_mm=out.margin_mm.astype(np.float32),
            n_views_used=out.n_views_used,
            source=out.source,
            **{f"track__{v}": per_view[v] for v in VIEWS},
            **{f"{node}__{v}": nodes[v][node].astype(np.float32)
               for v in VIEWS for node in ("pectoral", "tail")},
        )
        record.output(f"{METHOD}__{args.experiment}", path)

        stats = report(np.load(path), args.anchor)
        record.param("reconstruction", stats)
        print(f"\n{'cameras':>8} {'frames':>9} {'share':>7} {'residual med':>13} "
              f"{'margin p1':>10} {'margin p10':>11} {'margin<1mm':>11}")
        for k in (2, 3):
            cell = stats[f"views_{k}"]
            print(f"{k:>8} {cell['frames']:>9,} {cell['share']:>7.2%} "
                  f"{cell['residual_median']:>13.3f} {cell['margin_p1']:>10.3f} "
                  f"{cell['margin_p10']:>11.3f} {cell['margin_under_1mm']:>10.2%}")
        live = slot_track[slot_track >= 0]
        print(f"\nresolved {out.resolved.mean():.2%}   tracks {np.unique(live).size:,}   "
              f"per animal {np.unique(live).size / N_ANIMALS:.1f}")
        record.param("tracks", int(np.unique(live).size))

        (out_dir / "c13_summary.json").write_text(
            json.dumps({"experiment": args.experiment, "anchor": args.anchor,
                        "n_groups": 1,
                        "groups": [ExperimentId.parse(args.experiment).group],
                        "reconstruction": stats,
                        "max_step_mm": config.max_step_mm,
                        "tracks": int(np.unique(live).size)},
                       indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

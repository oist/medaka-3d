"""C18 — solve the cross-view correspondence on two keypoints instead of one.

`reconstruct` matches detections across cameras on **one point per animal** —
its input contract has no node axis — while the per-camera 2D tracker matches a
whole pose. `medaka3d.correspond.reconstruct_multinode` closes that gap, and
`docs/decisions/2026-08-18-2d-first-tracking.md` records what it is worth: the
runner-up margin at the first percentile rises 0.989 -> 3.635 mm on three-view
frames and 0.006 -> 0.252 mm on two-view frames, and the winner changes on
6-11% of frames.

A view counts as complete on the **anchor** alone, so a frame missing one mouth
is solved on the pectoral rather than losing a camera. That is the failure `c13`
measured: judging completeness on the mouth cost the third camera on a fifth of
the recording.

A separate command rather than a flag on `c03`, per design rule 5. `c03` is not
modified. Nothing is promoted here; the four-clause rule decides, and clause 1
needs `c09` run against this output.

**n_groups = 1, n_experiments = 1.**

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c18_multinode_correspondence.py
    .venv\\Scripts\\python.exe tools/c18_multinode_correspondence.py --nodes pectoral
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
from c13_pectoral_anchor import frame_arrays, report

from medaka3d import load_paths
from medaka3d.calib import load_rig
from medaka3d.cohort import ExperimentId
from medaka3d.correspond import reconstruct_multinode
from medaka3d.provenance import provenance
from medaka3d.reconstruct import N_ANIMALS, PERMUTATIONS
from medaka3d.track import METHODS, TrackConfig, propagate_to_views
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN, METHOD = "bottomup", "gated"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("Run::")[0].strip())
    parser.add_argument("--anchor", default="pectoral", choices=("pectoral", "mouth"))
    parser.add_argument("--nodes", nargs="+", default=["mouth", "pectoral"])
    parser.add_argument("--experiment", default=EXPERIMENT)
    args = parser.parse_args()

    paths = load_paths()
    rig = load_rig(CALIBRATION)
    tag = "-".join(sorted(args.nodes)) + f"_anchor-{args.anchor}"
    out_dir = paths.work_dir / "c18_multinode_correspondence" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("multinode_correspondence", out_dir, name=f"c18_{tag}") as record:
        record.params_from(args)
        record.input("calibration", rig.manifest)
        record.param("n_groups", 1)
        print(f"experiment {args.experiment}   anchor {args.anchor}   "
              f"nodes {args.nodes}   n_groups = 1")

        uv, complete, nodes_by_view, starts, counts = {}, {}, {}, {}, {}
        for view in VIEWS:
            detections = c01.load(paths.work_dir, args.experiment, view, RUN)
            nodes_by_view[view], complete[view] = frame_arrays(detections, args.anchor)
            uv[view] = nodes_by_view[view]
            starts[view] = np.searchsorted(
                detections.frame_idx, np.arange(detections.n_frames), side="left")
            counts[view] = int(detections.frame_idx.shape[0])
            print(f"  {view:<10} complete on {args.anchor} {complete[view].mean():6.2%}")

        started = time.time()
        out, tally = reconstruct_multinode(
            rig, args.experiment, uv, complete, c01.N_FRAMES,
            anchor=args.anchor, nodes=tuple(args.nodes))
        print(f"  reconstructed in {time.time() - started:.0f}s   "
              f"resolved {out.resolved.mean():.2%}")
        print(f"  frames {tally['frames']:,}   "
              + "  ".join(f"{k.split('.')[1]} contributed on {v:,} "
                          f"({v / max(1, tally['frames']):.1%})"
                          for k, v in tally.items() if k.startswith("node_frames."))
              + f"   **winner differs from {args.anchor}-only on "
                f"{tally['changed_vs_anchor_only']:,} "
                f"({tally['changed_vs_anchor_only'] / max(1, tally['frames']):.2%})**")
        record.param("rule", tally)

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
        record.param("max_step_mm", config.max_step_mm)
        record.param("max_gap_frames", config.max_gap_frames)

        slot_track = METHODS[METHOD](out, config)
        per_view = propagate_to_views(out, slot_track, starts, counts)
        path = out_dir / f"{METHOD}__{args.experiment}.npz"
        np.savez_compressed(
            path, slot_track=slot_track, xyz=out.xyz.astype(np.float32),
            residual_mm=out.residual_mm.astype(np.float32),
            margin_mm=out.margin_mm.astype(np.float32),
            n_views_used=out.n_views_used, source=out.source,
            **{f"track__{v}": per_view[v] for v in VIEWS},
            **{f"{node}__{v}": nodes_by_view[v][node].astype(np.float32)
               for v in VIEWS for node in ("pectoral", "tail")},
        )
        record.output(f"{METHOD}__{args.experiment}", path)

        stats = report(np.load(path), tag)
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
              f"per animal {np.unique(live).size / N_ANIMALS:.1f}   "
              f"max_step_mm {config.max_step_mm}")
        record.param("tracks", int(np.unique(live).size))

        (out_dir / "c18_summary.json").write_text(
            json.dumps({"experiment": args.experiment, "anchor": args.anchor,
                        "nodes": args.nodes, "n_groups": 1,
                        "groups": [ExperimentId.parse(args.experiment).group],
                        "rule": tally, "reconstruction": stats,
                        "tracks": int(np.unique(live).size),
                        "max_step_mm": config.max_step_mm}, indent=2, default=float),
            encoding="utf-8")
    print(f"\nwrote {out_dir}\nnow score it:  tools/c09_gt_swap_validation.py --npz {path}")


if __name__ == "__main__":
    main()

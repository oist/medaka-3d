"""C32 - `c28`'s solve with one change: a momentarily degraded view carries no rays.

=============================================================================
THE CASE THIS EXISTS FOR
=============================================================================

K_1_2 frame 31,479: two males fight near the surface, the waves wreck
`top_cam`'s inference for a few frames (best instance score 1.13-1.29 against
a calm median of 1.94), the reconstruction teleports two animals by ~174 mm,
and both the raw and the stitched labelling leave the scene **physically
exchanged**. `c30` fired two decisive-looking joins (margin ~336 mm) that net
to identity - continuity cannot repair a stretch where the geometry itself is
corrupted. The repair has to happen where the corruption enters.

So: on frames where **exactly one** view is degraded - best instance score
below `c31`'s floor of 1.5 - that view's rays are marked absent before the
solve. The machinery is `c25`'s existing absent-node handling; **no cost
function is edited and no constant is added.** A slot the two healthy views
support triangulates from them and pays no mu; the degraded view's labels are
still assigned (lambda's continuity term), they just carry no geometric weight
that frame. Frames with two or more degraded views are left untouched and
counted: demotion needs two healthy views, and a frame where it cannot have
them is a refusal, not a guess.

This is **not** the per-view weight table rejected on 2026-08-24. That was a
static weight measured on calm frames, and calm frames are exactly where the
cameras are equivalent. This is a per-frame drop, triggered by the detector's
own reported confidence, firing on 0.61% (top) / 1.4% / 1.7% (sides) of
frames.

Everything else reproduces `c28` exactly - same stage A labels (`c27`
serial), same lambda grid (inherited, none selected), same held post-hoc gate,
same output schema plus a ``demoted__<view>`` flag per frame. Scored by
re-stitching (`c30`) and `c09`, per the reading rules fixed in
`decisions/2026-09-01-doubt-channel.md` before this ran.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c32_view_demotion.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
import c25_variable_cardinality as c25
from c02_tracking_probe import CALIBRATION
from c19_viterbi_correspondence import derive_gate
from c28_lambda_grid_transfer import LAMBDA_GRID, residual_tail
from c31_doubt_channel import SCORE_DEGRADED, view_degraded, view_wobble

from medaka3d import load_paths
from medaka3d.calib import load_rig
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.reconstruct import N_ANIMALS, Reconstruction
from medaka3d.track import METHODS, UNTRACKED, propagate_to_views
from medaka3d.types import VIEWS

RUN = "bottomup"

#: `c19`'s gate, held for every arm since `c21`. POST-HOC.
GATE_MM = 4.0
GAP_FRAMES = 73

#: Wobble floor for DEMOTION - deliberately above `c31`'s 0.3 review floor.
#: Review wants sensitivity; demotion wants certainty that the view's
#: inference is malformed. 0.5 sits beyond every view's calm p99 (0.34-0.48)
#: and fires on 1.4% of frames across the three views together.
WOBBLE_DEMOTE = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--experiment", default="20251225_K_1_2_FCMC")
    parser.add_argument("--nodes", nargs="+", default=["pectoral", "mouth"])
    parser.add_argument("--mu", type=float, default=4.0)
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(LAMBDA_GRID))
    parser.add_argument("--frames", nargs=2, type=int, default=None,
                        help="frame range; for validation runs. A partial run "
                             "never lands where a whole one does")
    parser.add_argument("--demote-views", nargs="+", default=list(VIEWS),
                        choices=list(VIEWS),
                        help="which views may be demoted. The experimenter's "
                             "original proposal named top_cam alone (surface "
                             "waves); the all-views default is the "
                             "generalisation, and the 2026-09-01 grid showed "
                             "that generalisation failing c09's gate")
    parser.add_argument("--trigger", choices=("score", "wobble"), default="score",
                        help="what marks a view degraded on a frame: the SLEAP "
                             "best-instance score floor (1.5), or the "
                             "annotator-proposed body-length wobble floor "
                             "(0.5). A threshold choice, not an algorithm "
                             "switch: the demotion machinery is identical")
    args = parser.parse_args()

    paths = load_paths()
    experiment = args.experiment
    nodes = tuple(args.nodes)
    stem = "-".join(["mouth", "pectoral"]) + "_oks"
    label_dir = (paths.work_dir / "c27_uncapped_tracklets_any" / experiment
                 / f"{stem}_serial")
    if not label_dir.is_dir():
        raise SystemExit(f"no serial stage A at {label_dir}; run c27 first")

    leaf = (experiment if not args.frames
            else f"{experiment}_frames{args.frames[0]}-{args.frames[1]}")
    if set(args.demote_views) != set(VIEWS):
        leaf += "__" + "-".join(v[:-4].rstrip("_") for v in args.demote_views)
    if args.trigger != "score":
        leaf += f"__{args.trigger}"
    out_dir = paths.work_dir / "c32_view_demotion" / leaf
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("view_demotion", out_dir, name="c32") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        record.param("stage_a", "c27 serial (NOT c24 chunked)")
        record.param("stage_a_labels", str(label_dir))
        record.param("gate_mm", GATE_MM)
        record.param("max_gap_frames", GAP_FRAMES)
        record.param("gate_is_post_hoc", True)
        record.param("lambda_is_chosen", False)
        record.param("trigger", args.trigger)
        record.param("score_degraded_floor", SCORE_DEGRADED)
        record.param("wobble_demote_floor", WOBBLE_DEMOTE)
        record.param("demotion_needs_two_healthy_views", True)
        rig = load_rig(paths.asset_root / CALIBRATION)
        record.input("calibration", rig.manifest)

        lo, hi = args.frames if args.frames else (0, c01.N_FRAMES)
        frames = np.arange(lo, hi)
        uv, label_at, row_at, k = {}, {}, {}, {}
        starts, counts, n_labels = {}, {}, 0
        degraded = {}
        for view in VIEWS:
            detections = c01.load(paths.work_dir, experiment, view, RUN)
            labels = np.load(label_dir / f"track2d__{view}.npy")
            uv[view], label_at[view], row_at[view], k[view] = c25.compact_layout(
                detections, labels, nodes)
            starts[view], counts[view] = detections.start, int(labels.size)
            n_labels = max(n_labels, int(labels.max()) + 1)
            if args.trigger == "score":
                degraded[view] = view_degraded(detections, c01.N_FRAMES)
            else:
                degraded[view] = (view_wobble(detections, labels, c01.N_FRAMES)
                                  > WOBBLE_DEMOTE)

        n_degraded = np.stack([degraded[v] for v in VIEWS]).sum(axis=0)
        demoted = {v: degraded[v] & (n_degraded == 1)
                   & (v in args.demote_views) for v in VIEWS}
        undemotable = int((n_degraded >= 2).sum())
        for view in VIEWS:
            print(f"  {view:<10} degraded {int(degraded[view].sum()):,} frames"
                  f"   demoted {int(demoted[view].sum()):,}")
        print(f"  refused (>=2 views degraded at once): {undemotable:,} frames\n")
        record.param("demoted_frames",
                     {v: int(demoted[v].sum()) for v in VIEWS})
        record.param("undemotable_frames", undemotable)

        rays = c25.compact_rays(rig, uv, nodes)
        for view in VIEWS:
            rows = np.flatnonzero(demoted[view])
            for node in nodes:
                _origins, _directions, present = rays[(view, node)]
                present[rows, :] = False
        base = n_labels + 2

        arms = {}
        for penalty in args.lambdas:
            started = time.time()
            chosen, objective, inverse, tables, margin, pruned = c25.solve(
                rays, nodes, label_at, k, frames, penalty, args.mu, base)
            sizes = np.array([t.shape[0] for t in tables])[inverse]

            packed = np.zeros((frames.size, N_ANIMALS), np.int64)
            for index, table in enumerate(tables):
                member = np.flatnonzero(inverse == index)
                for start in range(0, member.size, c25.BLOCK):
                    take = member[start:start + c25.BLOCK]
                    packed[take] = c25.chosen_identities(
                        table[chosen[take]], label_at, frames[take], base)
            changed = (packed[1:] != packed[:-1]).sum(axis=1)

            xyz, residual, source_rows, slot_rays = c25.chosen_geometry(
                rays, nodes, tables, inverse, chosen, label_at, row_at,
                frames, nodes[0])
            out = Reconstruction(
                experiment=experiment, views=tuple(VIEWS), xyz=xyz,
                source=source_rows, residual_mm=residual,
                margin_mm=margin * N_ANIMALS,
                n_views_used=slot_rays.min(axis=1).astype(np.int8))

            derived = derive_gate(out)
            config = dataclasses.replace(derived, max_step_mm=GATE_MM,
                                         max_gap_frames=GAP_FRAMES)
            slot_track = METHODS["gated"](out, config)
            per_view = propagate_to_views(out, slot_track, starts, counts)
            path = out_dir / f"C32-gated__{experiment}__lam{penalty:g}.npz"
            np.savez_compressed(
                path, slot_track=slot_track, xyz=out.xyz.astype(np.float32),
                residual_mm=out.residual_mm.astype(np.float32),
                margin_mm=out.margin_mm.astype(np.float32),
                n_views_used=out.n_views_used, source=out.source,
                slot_rays=slot_rays, state=chosen.astype(np.int32),
                **{f"demoted__{v}": demoted[v][lo:hi] for v in VIEWS},
                **{f"track__{v}": per_view[v] for v in VIEWS})
            record.output(f"C32-gated__lam{penalty:g}", path)

            label_track = np.where(
                out.resolved[:, None], np.arange(N_ANIMALS)[None, :],
                UNTRACKED).astype(np.int32)
            arms[f"{penalty:g}"] = {
                "lambda_mm": float(penalty),
                "path": str(path),
                "objective": float(objective),
                "states_per_frame_mean": float(sizes.mean()),
                "frames_pruned": len(pruned),
                "resolved": float(out.resolved.mean()),
                "three_cameras_every_animal": float(
                    (slot_rays[out.resolved] == 3).all(axis=1).mean()),
                "slot_label_changes": int(changed.sum()),
                "gate_derived_mm": derived.max_step_mm,
                "residual": residual_tail(out.residual_mm, slot_rays),
                "coverage_gated": c25.coverage(slot_track, frames.size),
                "coverage_label": c25.coverage(label_track, frames.size),
                "seconds": time.time() - started,
            }
            a = arms[f"{penalty:g}"]
            r = a["residual"]
            print(f"  lambda {penalty:<6g} solved in {a['seconds']:.0f}s   "
                  f"resolved {a['resolved']:.4%}   "
                  f"3-cam median {r['median_mm']:.3f}  p99.9 {r['p99_9_mm']:.2f}  "
                  f">4mm {r['over_4mm']:,}   slot changes "
                  f"{a['slot_label_changes']:,}", flush=True)

        record.param("arms", arms)
        (out_dir / "c32_summary.json").write_text(
            json.dumps({"experiment": experiment, "n_groups": 1,
                        "n_experiments": 1,
                        "groups": [ExperimentId.parse(experiment).group],
                        "stage_a": "c27 serial", "nodes": list(nodes),
                        "mu": args.mu, "lambda_grid": list(args.lambdas),
                        "lambda_is_chosen": False,
                        "score_degraded_floor": SCORE_DEGRADED,
                        "demoted_frames": {v: int(demoted[v].sum())
                                           for v in VIEWS},
                        "undemotable_frames": undemotable,
                        "gate_mm": GATE_MM, "max_gap_frames": GAP_FRAMES,
                        "gate_is_post_hoc": True, "arms": arms},
                       indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

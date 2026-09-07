"""C28 - `c25`'s correspondence across the whole lambda grid, on serial stage A.

=============================================================================
NO LAMBDA IS CHOSEN HERE, AND THAT IS THE POINT
=============================================================================

The 2026-08-18 rule picks `lambda` by held-out node cross-validation. The audit
of 2026-08-26 found that on K_1_2 it resolves a factor of 3.16 in `lambda` at
0.09-0.41% of its own score - one direction of one run being a tie at 0.002% -
and that the two runs said to have agreed on ``kappa = 0.2371`` in fact chose
grid points 3.16x apart in **each** direction. Only the geometric mean
coincided, and only because the two grid indices happened to sum to the same
number: on a geometric grid ``sqrt(K[i] * K[j])`` depends on ``i + j`` alone.

So there is no selector to trust and none is used. **The whole grid is run and a
claim is carried forward only if it holds at every point of it.** On K_1_1 and
K_1_3 there is no `c09`, so a `lambda` picked after the fact could never be
checked - but a statement true across the range needs no pick.

The grid is **inherited, not selected**: the seven values already stored for
K_1_2 on 2026-08-25. Taking K_1_2's own set is what makes the columns comparable
and removes the choice.

=============================================================================
WHY THE LABELS COME FROM `c27` AND NOT `c24`
=============================================================================

`c24` ran stage A in 30 parallel chunks. On 2026-08-26 that was measured against
a serial run, **within each chunk** so that a label stopping at a boundary is
not counted as a disagreement:

    K_1_2   top_cam     47.15% of instances in a different partition
            side1_cam   46.63%
            side2_cam   53.82%       exact agreement on 2-4 of 30 chunks

Half the labelling differs. `c24` is therefore not "`c16`'s tracker with the cap
off" - it is a different computation - and any comparison of `c16` against
`c24` confounds the cap with the chunking. `c27` runs the same tracker serially,
which is the only arm that isolates the cap.

The gate is held at `c19`'s **4.0 mm / 73 frames** for every arm, as it has been
since `c21`. That is POST-HOC and is recorded as such: `derive_gate` takes the
99.9th percentile of the 3D step, and a step across a correspondence change is a
**renaming** rather than a movement, so a gate derived on one reconstruction
cannot be compared with one derived on another.

See `docs/decisions/2026-08-26-transfer-to-k11-k13.md`, written before this ran.
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

from medaka3d import load_paths
from medaka3d.calib import load_rig
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.reconstruct import N_ANIMALS, Reconstruction
from medaka3d.track import METHODS, UNTRACKED, propagate_to_views
from medaka3d.types import VIEWS

RUN = "bottomup"

#: The seven arms already stored for K_1_2 on 2026-08-25. Inherited, not chosen.
LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 0.767, 2.0, 8.0)

#: `c19`'s gate, held for every arm since `c21`. POST-HOC.
GATE_MM = 4.0
GAP_FRAMES = 73


def residual_tail(residual: np.ndarray, slot_rays: np.ndarray) -> dict:
    """The tail over three-camera animal-frames, computed as the reference was.

    Restricting to ``slot_rays == 3`` is what makes the number comparable: an
    animal seen by two cameras triangulates with no redundancy, so its residual
    is near zero by construction and says nothing about whether the
    correspondence is right.
    """
    values = residual[slot_rays == 3]
    values = values[np.isfinite(values)]
    return {
        "three_cam_animal_frames": int(values.size),
        "median_mm": float(np.median(values)),
        "p99_mm": float(np.percentile(values, 99)),
        "p99_9_mm": float(np.percentile(values, 99.9)),
        "max_mm": float(values.max()),
        "over_4mm": int((values > 4.0).sum()),
        "over_4mm_fraction": float((values > 4.0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--nodes", nargs="+", default=["pectoral", "mouth"],
                        help="the first is the anchor, as in c13/c18/c25")
    parser.add_argument("--mu", type=float, default=4.0,
                        help="mm charged for a (slot, node) no two cameras support")
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(LAMBDA_GRID),
                        help="the inherited grid. Changing it makes the columns "
                             "incomparable to K_1_2's stored arms")
    parser.add_argument("--frames", nargs=2, type=int, default=None,
                        help="frame range; for validation runs")
    args = parser.parse_args()

    paths = load_paths()
    experiment = args.experiment
    nodes = tuple(args.nodes)
    stem = "-".join(["mouth", "pectoral"]) + "_oks"
    label_dir = (paths.work_dir / "c27_uncapped_tracklets_any" / experiment
                 / f"{stem}_serial")
    if not label_dir.is_dir():
        raise SystemExit(
            f"no serial stage A at {label_dir}. Run c27 first:\n"
            f"    .venv\\Scripts\\python.exe tools/c27_uncapped_tracklets_any.py "
            f"--experiment {experiment}")

    # A partial run never lands where a whole one does, so it can never be
    # mistaken for one - the same rule c27 applies to its own frame range.
    leaf = (experiment if not args.frames
            else f"{experiment}_frames{args.frames[0]}-{args.frames[1]}")
    out_dir = paths.work_dir / "c28_lambda_grid_transfer" / leaf
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("lambda_grid_transfer", out_dir, name="c28") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        record.param("stage_a", "c27 serial (NOT c24 chunked)")
        record.param("stage_a_labels", str(label_dir))
        record.param("gate_mm", GATE_MM)
        record.param("max_gap_frames", GAP_FRAMES)
        record.param("gate_is_post_hoc", True)
        record.param("lambda_is_chosen", False)
        rig = load_rig(paths.asset_root / CALIBRATION)
        record.input("calibration", rig.manifest)

        lo, hi = args.frames if args.frames else (0, c01.N_FRAMES)
        frames = np.arange(lo, hi)
        print(f"{experiment}   stage A: c27 SERIAL   nodes {nodes}   mu {args.mu} mm")
        print(f"frames {lo:,}-{hi:,}   n_groups = 1, n_experiments = 1")
        print(f"lambda grid {list(args.lambdas)} mm - inherited, none selected")
        print(f"gate held at {GATE_MM} mm / {GAP_FRAMES} frames - POST-HOC\n")

        uv, label_at, row_at, k = {}, {}, {}, {}
        starts, counts, n_labels = {}, {}, 0
        for view in VIEWS:
            detections = c01.load(paths.work_dir, experiment, view, RUN)
            labels = np.load(label_dir / f"track2d__{view}.npy")
            uv[view], label_at[view], row_at[view], k[view] = c25.compact_layout(
                detections, labels, nodes)
            starts[view], counts[view] = detections.start, int(labels.size)
            n_labels = max(n_labels, int(labels.max()) + 1)
            census = np.bincount(k[view][frames], minlength=6)
            print(f"  {view:<10} {int(labels.max()) + 1:2d} labels   per frame  "
                  + "  ".join(f"{i}:{c:,}" for i, c in enumerate(census) if c))
        rays = c25.compact_rays(rig, uv, nodes)
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
            path = out_dir / f"C28-gated__{experiment}__lam{penalty:g}.npz"
            np.savez_compressed(
                path, slot_track=slot_track, xyz=out.xyz.astype(np.float32),
                residual_mm=out.residual_mm.astype(np.float32),
                margin_mm=out.margin_mm.astype(np.float32),
                n_views_used=out.n_views_used, source=out.source,
                slot_rays=slot_rays, state=chosen.astype(np.int32),
                **{f"track__{v}": per_view[v] for v in VIEWS})
            record.output(f"C28-gated__lam{penalty:g}", path)

            label_track = np.where(
                out.resolved[:, None], np.arange(N_ANIMALS)[None, :],
                UNTRACKED).astype(np.int32)
            arms[f"{penalty:g}"] = {
                "lambda_mm": float(penalty),
                "path": str(path),
                "objective": float(objective),
                "states_per_frame_mean": float(sizes.mean()),
                "states_per_frame_max": int(sizes.max()),
                "frames_pruned": len(pruned),
                "resolved": float(out.resolved.mean()),
                "three_cameras_every_animal": float(
                    (slot_rays[out.resolved] == 3).all(axis=1).mean()),
                "slot_label_changes": int(changed.sum()),
                "slot_changes_1": int((changed == 1).sum()),
                "slot_changes_2": int((changed == 2).sum()),
                "slot_changes_3": int((changed == 3).sum()),
                "gate_derived_mm": derived.max_step_mm,
                "gate_derived_gap_frames": derived.max_gap_frames,
                "residual": residual_tail(out.residual_mm, slot_rays),
                "coverage_gated": c25.coverage(slot_track, frames.size),
                "coverage_label": c25.coverage(label_track, frames.size),
                "seconds": time.time() - started,
            }
            a = arms[f"{penalty:g}"]
            r = a["residual"]
            print(f"\n  lambda {penalty:<6g} solved in {a['seconds']:.0f}s   "
                  f"states/frame mean {a['states_per_frame_mean']:.0f}")
            print(f"    resolved {a['resolved']:.4%}   three cams on every animal "
                  f"{a['three_cameras_every_animal']:.2%}")
            print(f"    residual 3-cam  median {r['median_mm']:.3f}  "
                  f"p99 {r['p99_mm']:.3f}  p99.9 {r['p99_9_mm']:.2f}  "
                  f"max {r['max_mm']:.1f}  >4mm {r['over_4mm']:,} "
                  f"({r['over_4mm_fraction']:.3%})")
            print(f"    slot changes {a['slot_label_changes']:,}  "
                  f"(1: {a['slot_changes_1']:,}  2: {a['slot_changes_2']:,}  "
                  f"3: {a['slot_changes_3']:,})   "
                  f"derive_gate would say {a['gate_derived_mm']} mm")

        record.param("arms", arms)
        (out_dir / "c28_summary.json").write_text(
            json.dumps({"experiment": experiment, "n_groups": 1,
                        "n_experiments": 1,
                        "groups": [ExperimentId.parse(experiment).group],
                        "stage_a": "c27 serial", "stage_a_labels": str(label_dir),
                        "nodes": list(nodes), "mu": args.mu,
                        "lambda_grid": list(args.lambdas),
                        "lambda_is_chosen": False,
                        "gate_mm": GATE_MM, "max_gap_frames": GAP_FRAMES,
                        "gate_is_post_hoc": True,
                        "frames": [int(lo), int(hi)], "arms": arms},
                       indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

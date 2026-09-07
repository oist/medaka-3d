"""C26 - the two lambda selectors, side by side, on the one bench where they agree.

=============================================================================
THE QUESTION
=============================================================================

`c19.select_kappa` returned ``kappa = 0.2371``. `c20` imports that same function
and returned ``0.2371`` again on a cost scale 3.6x apart - two independent
confirmations of the quantity ``kappa`` exists to make transferable.
`c25.sweep_kappa`, a separate reimplementation of the same rule on `c25`'s state
space, returns 0.75-1.33 and prefers values in the degrading half of its own
sweep. **Every lambda in the 2026-08-25 record is therefore post-hoc**, and on
K_1_1 and K_1_3 there is no `c09` to justify one after the fact.

This tool does not fix anything and does not choose a lambda. It runs both
selectors on the same frames and reports which term produces the disagreement.
`c25` and `c19` are imported, never modified.

=============================================================================
WHY `c16`'S LABELS, AND ONLY THOSE
=============================================================================

`c25 --check-reduction` proved its per-state cost equals `c20`'s `label_costs`
to 1.846e-06 mm with 0 winner disagreements, on `c16`'s labels. The Step A run
of 2026-08-26 measured `c25`'s state space there at mean 214, max 216 - its
floor, which is `c19`'s 36 correspondences times the 6 slot renamings `c19` does
not carry because it pins slot naming to ``top_cam``.

On `c24`'s labels the two selectors do not see the same geometry, which is why
the disagreement was never localised.

=============================================================================
THE CHECK THAT COMES FIRST, AND CAN FAIL
=============================================================================

`c19`'s selector is **re-run on the window** rather than read out of the
whole-recording store, so the comparison is like-for-like. It must reproduce the
stored whole-recording answer:

    kappa* 0.2371   (fit mouth 0.3162, fit pectoral 0.1778)
    margins  mouth 3.228   pectoral 3.197   total 3.233 mm

Landing on the same grid point is the criterion, because the grid is the
selector's own resolution: ``geomspace(1e-4, 1e2, 25)``, 10^0.25 = 1.78 a step.
**If the window does not reproduce it, the window is not a bench** and the tool
says so rather than printing a comparison that cannot be read.

See `docs/decisions/2026-08-26-lambda-selector.md`, written before this ran.
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
import c20_uncapped_correspondence as c20
import c25_variable_cardinality as c25
from c02_tracking_probe import CALIBRATION
from c19_viterbi_correspondence import KAPPA, select_kappa

from medaka3d import load_paths
from medaka3d.calib import load_rig
from medaka3d.correspond import label_costs
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN = "bottomup"

#: The stored whole-recording answer this window has to reproduce, read off
#: `c20`'s provenance. Not a tolerance - a grid point.
ORACLE = {"kappa": 0.23713737056616555,
          "best_kappa": {"mouth": 0.31622776601683794,
                         "pectoral": 0.1778279410038923},
          "margins_mm": {"mouth": 3.2282791137695312,
                         "pectoral": 3.19677996635437,
                         "total": 3.2331314086914062}}


def window_costs(costs, window: np.ndarray):
    """`label_costs` restricted to the window, without recomputing it.

    `select_kappa` reads `margin`, `usable` and `fit_cost`, all of them keyed by
    frame, so slicing every field is the whole of it. Recomputing on a slice
    would be a different measurement, because `label_costs` batches.
    """
    return dataclasses.replace(
        costs,
        cost=costs.cost[window],
        by_node={node: array[window] for node, array in costs.by_node.items()},
        usable={node: array[window] for node, array in costs.usable.items()},
        margin={node: array[window] for node, array in costs.margin.items()},
        terms=costs.terms[window],
        slot_rays=costs.slot_rays[window],
        n_views_used=costs.n_views_used[window],
    )


def grid_index(value: float) -> int:
    """Which point of `c19`'s own grid a kappa sits on."""
    return int(np.argmin(np.abs(np.log(KAPPA) - np.log(value))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--frames", nargs=2, type=int, default=[100_000, 120_000],
                        help="the window both selectors read. A DISCLOSED "
                             "deviation; the tool records it and checks that "
                             "c19's selector still reproduces its stored answer")
    parser.add_argument("--nodes", nargs="+", default=["pectoral", "mouth"],
                        help="the first is the anchor, as in c13/c18/c25")
    parser.add_argument("--mu", type=float, default=4.0,
                        help="c25's charge for a (slot, node) no two cameras support")
    args = parser.parse_args()

    paths = load_paths()
    nodes = tuple(args.nodes)
    lo, hi = args.frames
    window = np.arange(lo, hi)
    out_dir = paths.work_dir / "c26_lambda_selector_bench" / f"c16_{lo}-{hi}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("lambda_selector_bench", out_dir, name="c26") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        record.param("labels", "c16")
        rig = load_rig(paths.asset_root / CALIBRATION)
        record.input("calibration", rig.manifest)

        print(f"{EXPERIMENT}   labels c16   nodes {nodes}   mu {args.mu} mm")
        print(f"window {lo:,}-{hi:,}   n_groups = 1, n_experiments = 1")
        print("no human annotation is read by either selector\n")

        label_path = paths.work_dir / c25.LABEL_SOURCES["c16"]
        by_label, uv, label_at, row_at, k = {}, {}, {}, {}, {}
        n_labels = 0
        for view in VIEWS:
            detections = c01.load(paths.work_dir, EXPERIMENT, view, RUN)
            labels = np.load(label_path / f"track2d__{view}.npy")
            by_label[view], _, _, _ = c20.label_arrays(detections, labels)
            uv[view], label_at[view], row_at[view], k[view] = c25.compact_layout(
                detections, labels, nodes)
            n_labels = max(n_labels, int(labels.max()) + 1)

        # --- the bench check: c19's selector, re-run on the window ------------
        started = time.time()
        costs = label_costs(rig, by_label, c01.N_FRAMES, anchor=nodes[0], nodes=nodes)
        here = window_costs(costs, window)
        evidence = here.terms.max(axis=1) > 0
        old = select_kappa(here, evidence)
        print(f"c19/c20's select_kappa on the window, {time.time() - started:.0f}s")
        for node in ("mouth", "pectoral"):
            curve = old["curves"][node]
            print(f"  fit {node:<9} kappa* {curve['best_kappa']:.4f}  "
                  f"(stored {ORACLE['best_kappa'][node]:.4f})  "
                  f"margin {curve['margin_mm']:.3f} mm  "
                  f"(stored {ORACLE['margins_mm'][node]:.3f})  "
                  f"{'interior' if curve['interior'] else 'AT THE GRID EDGE'}")
        print(f"  kappa {old['kappa']:.4f}   (stored {ORACLE['kappa']:.4f})   "
              f"lambda {old['lambda_mm']:.3f} mm")
        record.param("c19_on_window", old)

        agrees = all(grid_index(old["curves"][node]["best_kappa"])
                     == grid_index(ORACLE["best_kappa"][node])
                     for node in ("mouth", "pectoral"))
        record.param("window_reproduces_stored_oracle", bool(agrees))
        if not agrees:
            print("\n  !! the window does not reproduce the stored whole-recording "
                  "answer.\n     It is not a bench, and no comparison below could "
                  "be read. Stopping.")
            raise SystemExit(
                "window is not a bench; widen --frames or use the whole recording")
        print("  the window reproduces the stored answer on both directions\n")

        # --- c25's selector, verbatim, on the same frames ---------------------
        started = time.time()
        rays = c25.compact_rays(rig, uv, nodes)
        new = c25.sweep_kappa(rays, nodes, label_at, k, window, KAPPA, args.mu,
                              n_labels + 2)
        print(f"c25's sweep_kappa on the same window, {time.time() - started:.0f}s")
        for node in nodes:
            curve = new["curves"][node]
            print(f"  fit {node:<9} kappa* {curve['best_kappa']:.4f}  "
                  f"margin {curve['margin_mm']:.3f} mm  "
                  f"{'interior' if curve['interior'] else 'AT THE GRID EDGE'}")
        print(f"  kappa {new['kappa']:.4f}   lambda {new['lambda_mm']:.3f} mm")
        record.param("c25_on_window", new)

        # --- (1) the margins --------------------------------------------------
        print("\n(1) margins - kappa = lambda / m, so a margin k times too small "
              "inflates kappa by k")
        ratios = {}
        for key in ("mouth", "pectoral", "total"):
            a = old["margins_mm"][key]
            b = new["margins_mm"][key]
            ratios[key] = a / b if b else float("nan")
            print(f"  {key:<9} c19 {a:7.3f} mm   c25 {b:7.3f} mm   "
                  f"c19/c25 {ratios[key]:6.3f}x")
        record.param("margin_ratio_c19_over_c25", ratios)

        # --- (2) the curves ---------------------------------------------------
        print("\n(2) held-out curves on the same 25-point grid "
              "(kappa, then lambda in mm)")
        for node in ("mouth", "pectoral"):
            a, b = old["curves"][node], new["curves"][node]
            print(f"  fit {node}")
            print(f"    c19  kappa* {a['best_kappa']:8.4f}  index {a['best_index']:2d}"
                  f"   lambda* {a['best_kappa'] * a['margin_mm']:7.3f} mm")
            print(f"    c25  kappa* {b['best_kappa']:8.4f}  index {b['best_index']:2d}"
                  f"   lambda* {b['best_kappa'] * b['margin_mm']:7.3f} mm")
            print(f"    grid points apart: {b['best_index'] - a['best_index']:+d}")
            print("    c19 held_mean_mm  "
                  + " ".join(f"{v:.5f}" for v in a["held_mean_mm"]))
            print("    c25 held_mean_mm  "
                  + " ".join(f"{v:.5f}" for v in b["held_mean_mm"]))

        # --- (3) the penalty unit, measured ------------------------------------
        print("\n(3) slot changes along c25's curve - the penalty-unit conversion, "
              "measured rather than assumed")
        for node in nodes:
            changes = np.asarray(new["curves"][node]["slot_changes"], float)
            index = new["curves"][node]["best_index"]
            print(f"  fit {node:<9} at its own kappa* {changes[index]:12,.0f} "
                  f"slot changes; at kappa = 0.2371 "
                  f"{changes[grid_index(ORACLE['kappa'])]:12,.0f}")
        record.param("kappa_grid", KAPPA.tolist())

        (out_dir / "c26_bench.json").write_text(
            json.dumps({"experiment": EXPERIMENT, "n_groups": 1, "n_experiments": 1,
                        "labels": "c16", "frames": [lo, hi], "nodes": list(nodes),
                        "mu": args.mu, "oracle": ORACLE,
                        "c19_on_window": old, "c25_on_window": new,
                        "margin_ratio_c19_over_c25": ratios},
                       indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

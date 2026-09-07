"""C19 — stage B: solve the cross-view correspondence as a piecewise-constant
sequence, not one frame at a time.

=============================================================================
WHAT THIS IS
=============================================================================

`reconstruct`, `c13` and `c18` all solve the correspondence **independently at
every frame**. Three attempts to add a temporal term on top of that are retired
in ``docs/decisions/2026-08-18-2d-first-tracking.md`` Addendum 2, and the best
of them failed for an arithmetic reason: a modal vote over a window follows a
genuine change only after it has held half a window, so it spent most of its
effect delaying correct reassignments and swaps went 8 -> 67.

This is the lag-free version. Over the whole recording, minimise

    SUM_t cost_t(s_t)  +  lambda * (number of times s changes)

exactly, by Viterbi over 36 states x 348,600 frames. No window, no vote, no
direction of time: a change at frame t is paid for by evidence on both sides of
t at once.

The 36 states are the pairs of permutations ``top label -> side1 label`` and
``top label -> side2 label``, where the labels are `c16`'s per-camera 2D tracks.
That is what makes "the correspondence stayed the same" mean anything between
frames — detection rows are the detector's own ordering and mean nothing across
a frame boundary.

**At lambda = 0 this must reproduce `c18` frame for frame**, and the tool
asserts it rather than hoping. If it does not, the label ordering or the
row map is wrong, which is silent on real data.

=============================================================================
LAMBDA IS CHOSEN WITHOUT THE GROUND TRUTH, AND NOT THE WAY THE HANDOFF SAID
=============================================================================

The handoff asked for lambda swept until the change count matches the 635
changes per 284,305 adjacent pairs measured on 2026-08-18. **That is provably
unsatisfiable**: the change count of the minimising sequence is non-increasing
in lambda and at lambda -> 0 it already equals the unconstrained per-frame
count, so any target equal to the unconstrained rate is reached only by not
smoothing at all. The tool that produced 635 was also deleted with retired
design 3 and nothing reproduces it, and it was measured on a labelling that
predates `c16`. The quantity is recomputed here and **reported as a diagnostic,
not used as a selector**.

Lambda is instead chosen by **held-out node cross-validation**, fixed in
``docs/decisions/2026-08-18-viterbi-correspondence.md`` before this file
existed: fit the sequence on the mouth alone and score it by the pectoral's
residual, and the other way round; report lambda in the scale-free unit
``kappa = lambda / m`` where *m* is the median runner-up margin of the cost used
in that fit, so a kappa selected on a one-node fit transfers to the deployed
two-node fit. Nothing in the selection reads `female_sex_gt.csv`, `c09`, `c14`
or `c15`.

=============================================================================
TWO ARMS, BOTH SCORED
=============================================================================

``B-gated``   `gated` on the stage-B reconstruction, gate derived from it the
              same way `c13`/`c18` derive theirs. Comparable to `c03`/`c13`/`c18`.
``B-label``   track ID = the ``top_cam`` 2D label, held constant. The cheapest
              possible stage C, and the arm clause 5 exists to test. Its known
              weakness is stated in advance: at a state change at least one
              camera relabelled, and naming identity after ``top_cam`` is wrong
              whenever the camera that relabelled was ``top_cam``.

This is **not** retired design 1, which forced one constant correspondence on
the whole recording. Here the correspondence is free at every frame and only
*changes* are charged.

Nothing is promoted. `c03` is not modified. The five-clause rule decides, and
clause 1 needs `c09` run against these outputs — which overwrites `c09`'s fixed
output path, so re-run `c09` with no ``--npz`` afterwards to restore the
baseline that `c11`/`c14`/`c15` read.

**n_groups = 1, n_experiments = 1, female only. Swap counts of 2-8 are small
samples and no ranking built on them is stable.**

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c19_viterbi_correspondence.py
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
from c02_tracking_probe import CALIBRATION
from c03_track_3d import GAP_PERCENTILE, GATE_PERCENTILE, VELOCITY_FRAMES, run_lengths
from c04_score_tracking import node_pairing
from c13_pectoral_anchor import frame_arrays, report

from medaka3d import load_paths
from medaka3d.calib import load_rig
from medaka3d.cohort import ExperimentId
from medaka3d.correspond import (
    reconstruct_from_states,
    state_costs,
    viterbi_path,
    viterbi_sweep,
)
from medaka3d.eval.tracking import modal_agreement
from medaka3d.provenance import provenance
from medaka3d.reconstruct import N_ANIMALS, PERMUTATIONS
from medaka3d.track import METHODS, UNTRACKED, TrackConfig, propagate_to_views
from medaka3d.types import FPS, VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN, METHOD = "bottomup", "gated"
TRACKER_2D = Path("c16_sleap_tracker_2d") / "mouth-pectoral_oks"

#: Grid for the scale-free penalty. Wide enough on both sides that a minimum in
#: the middle is a minimum and not an edge of the grid, which is the one way a
#: cross-validated choice fails quietly.
KAPPA = np.geomspace(1e-4, 1e2, 25)


def label_ordered(detections, labels: np.ndarray, anchor: str):
    """Reindex a view's detections from detector row to 2D track label.

    Returns ``(by_label, by_row, row_of_label, complete)``. **Both orders are
    kept and they are not interchangeable.** The solver needs label order, or a
    state means something different at every frame. `c04`'s cross-view agreement
    and `c09`'s anchor lookup both read detector order. Mixing them pairs one
    animal's 3D position with another animal's pixels, silently.
    """
    by_row, complete = frame_arrays(detections, anchor)
    counts = detections.counts()
    three = counts == N_ANIMALS
    start = np.searchsorted(detections.frame_idx, np.flatnonzero(three), side="left")
    rows = start[:, None] + np.arange(N_ANIMALS)[None, :]

    label_of_row = labels[rows]  # (n_three, 3) label carried by each detector row
    clean = (label_of_row >= 0).all(axis=1) & (
        np.sort(label_of_row, axis=1) == np.arange(N_ANIMALS)[None, :]).all(axis=1)
    # argsort of a permutation inverts it: label -> the row that carries it.
    row_of_label = np.zeros((detections.n_frames, N_ANIMALS), np.int8)
    row_of_label[np.flatnonzero(three)] = np.argsort(label_of_row, axis=1).astype(np.int8)

    frames = np.arange(detections.n_frames)[:, None]
    by_label = {node: value[frames, row_of_label] for node, value in by_row.items()}
    labelled = np.zeros(detections.n_frames, bool)
    labelled[np.flatnonzero(three)] = clean
    return by_label, by_row, row_of_label, complete & labelled, float(clean.mean())


def slot_signature(source: np.ndarray) -> np.ndarray:
    """(n_frames, 3) frame-order-independent encoding of a correspondence.

    Slots are arbitrary within a frame — `medaka3d.reconstruct` says so — and
    this stage indexes them by ``top_cam`` 2D label where `c18` indexed them by
    ``top_cam`` detection row, so the two agree on the correspondence while
    listing the slots in a different order. Comparing them element-wise would
    report a disagreement that is not one. Encode each slot's per-view row
    triple as one integer and sort, so only the *set* of triples is compared.
    """
    packed = ((source + 1) * (4 ** np.arange(source.shape[2]))).sum(axis=2)
    return np.sort(packed, axis=1)


def coverage(slot_track: np.ndarray, n_frames: int) -> dict:
    """Clause 5, both definitions, so nothing is ever compared across definitions.

    ``top3_track_coverage``  frames held by the three largest track IDs, over
                             3 x T. The handoff's 15.6% for `gated`.
    ``top3_run_coverage``    frames inside uninterrupted runs of >= 1 s, over
                             3 x T, summed over every track. The definition
                             `c16.tracklet_runs` uses, so it is comparable with
                             the 2D tracker's 94.6-97.0%.
    """
    held = slot_track >= 0
    if not held.any():
        return {"tracks": 0, "top3_track_coverage": 0.0, "top3_run_coverage": 0.0}
    frames, _ = np.nonzero(held)
    identifiers = slot_track[held]
    counts = np.bincount(identifiers)
    top3 = np.sort(counts)[::-1][:N_ANIMALS].sum()

    # Runs, in one sort rather than one pass per track: a new run starts wherever
    # the track changes or the frame is not the previous one plus one.
    order = np.lexsort((frames, identifiers))
    identifiers, frames = identifiers[order], frames[order]
    starts = np.r_[True, (identifiers[1:] != identifiers[:-1]) | (np.diff(frames) != 1)]
    lengths = np.bincount(np.cumsum(starts) - 1)
    in_long_runs = int(lengths[lengths >= FPS].sum())
    return {
        "tracks": int(counts[counts > 0].size),
        "top3_track_coverage": float(top3 / (N_ANIMALS * n_frames)),
        "top3_run_coverage": float(in_long_runs / (N_ANIMALS * n_frames)),
    }


def cross_view_agreement(rig, run: dict) -> dict:
    """Clause 3. Re-derive the pairing from a node the tracker did not link on."""
    out = {}
    for node in ("pectoral", "tail"):
        for other in rig.views[1:]:
            stats = modal_agreement(node_pairing(rig, run, node, rig.views[0], other))
            out[f"{node}:{rig.views[0]}~{other}"] = stats
    return out


def derive_gate(out) -> TrackConfig:
    """`c03`'s gate, re-derived on this reconstruction — never carried over."""
    frames = np.flatnonzero(out.resolved)
    adjacent = frames[:-1][np.diff(frames) == 1]
    here, there = out.xyz[adjacent], out.xyz[adjacent + 1]
    cost = np.linalg.norm(here[:, :, None, :] - there[:, None, :, :], axis=-1)
    totals = np.stack([cost[:, [0, 1, 2], p].sum(axis=1) for p in PERMUTATIONS], axis=-1)
    perms = np.asarray(PERMUTATIONS)[np.argmin(totals, axis=-1)]
    step = cost[np.arange(cost.shape[0])[:, None], np.arange(3)[None, :], perms].ravel()
    gap = run_lengths(~out.resolved)
    return TrackConfig(
        max_step_mm=float(np.round(np.percentile(step, GATE_PERCENTILE), 1)),
        max_gap_frames=int(np.percentile(gap, GAP_PERCENTILE)) if gap.size else 0,
        velocity_frames=VELOCITY_FRAMES,
    )


def select_kappa(costs, evidence: np.ndarray) -> dict:
    """The pre-registered choice of lambda. Reads no human annotation.

    Fit on one node, score on the other, both ways. Reported in kappa = lambda/m
    so the number transfers from the one-node selection fit to the two-node
    deployed fit, whose margins are roughly three times larger.
    """
    margins = {
        node: float(np.nanmedian(costs.margin[node][costs.usable[node]]))
        for node in ("mouth", "pectoral")
    }
    margins["total"] = float(np.nanmedian(costs.margin["total"][evidence]))

    curves = {}
    for fit, held in (("mouth", "pectoral"), ("pectoral", "mouth")):
        scale = margins[fit]
        sweep = viterbi_sweep(costs.fit_cost(fit), KAPPA * scale,
                              held_out=costs.fit_cost(held))
        scored = int(costs.usable[held].sum())
        curves[fit] = {
            "held_out": held, "margin_mm": scale,
            "kappa": KAPPA.tolist(),
            "changes": sweep["changes"].tolist(),
            "held_mean_mm": (sweep["held_cost"] / scored).tolist(),
            "best_index": int(np.argmin(sweep["held_cost"])),
            "best_kappa": float(KAPPA[int(np.argmin(sweep["held_cost"]))]),
            "interior": bool(0 < int(np.argmin(sweep["held_cost"])) < KAPPA.size - 1),
        }
    kappa = float(np.sqrt(curves["mouth"]["best_kappa"] * curves["pectoral"]["best_kappa"]))
    return {
        "margins_mm": margins, "curves": curves, "kappa": kappa,
        "lambda_mm": kappa * margins["total"],
        "rule": "held-out node cross-validation, pre-registered 2026-08-18",
        "interior_minimum_both_directions": bool(
            curves["mouth"]["interior"] and curves["pectoral"]["interior"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--anchor", default="pectoral", choices=("pectoral", "mouth"))
    parser.add_argument("--nodes", nargs="+", default=["mouth", "pectoral"])
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--lambda-mm", type=float, default=None,
                        help="Override the cross-validated penalty. Using this makes the "
                             "run a post-hoc choice and it must be reported as one.")
    args = parser.parse_args()

    paths = load_paths()
    rig = load_rig(CALIBRATION)
    tag = "-".join(sorted(args.nodes)) + f"_anchor-{args.anchor}"
    out_dir = paths.work_dir / "c19_viterbi_correspondence" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("viterbi_correspondence", out_dir, name=f"c19_{tag}") as record:
        record.params_from(args)
        record.input("calibration", rig.manifest)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        print(f"experiment {args.experiment}   anchor {args.anchor}   nodes {args.nodes}")
        print("n_groups = 1, n_experiments = 1 — K_1_2, female only\n")

        by_label, by_row, row_of_label, complete = {}, {}, {}, {}
        starts, counts = {}, {}
        for view in VIEWS:
            detections = c01.load(paths.work_dir, args.experiment, view, RUN)
            labels = np.load(paths.work_dir / TRACKER_2D / f"track2d__{view}.npy")
            (by_label[view], by_row[view], row_of_label[view],
             complete[view], clean) = label_ordered(detections, labels, args.anchor)
            starts[view] = np.searchsorted(
                detections.frame_idx, np.arange(detections.n_frames), side="left")
            counts[view] = int(detections.frame_idx.shape[0])
            print(f"  {view:<10} complete {complete[view].mean():6.2%}   "
                  f"2D labels a clean permutation on {clean:.4%} of 3-detection frames")

        started = time.time()
        costs = state_costs(rig, by_label, complete, c01.N_FRAMES, nodes=tuple(args.nodes))
        evidence = costs.n_views_used >= 2
        print(f"\n  {costs.n_states} states x {c01.N_FRAMES:,} frames built in "
              f"{time.time() - started:.0f}s   evidence on {evidence.mean():.2%}")

        # --- the assertion that the state space is the one c18 solved ---------
        c18_path = (paths.work_dir / "c18_multinode_correspondence"
                    / f"{'-'.join(sorted(args.nodes))}_anchor-{args.anchor}"
                    / f"{METHOD}__{args.experiment}.npz")
        if c18_path.exists():
            zero = viterbi_path(costs.cost, 0.0)
            check = reconstruct_from_states(
                rig, args.experiment, by_label, complete, row_of_label, zero,
                c01.N_FRAMES, anchor=args.anchor, nodes=tuple(args.nodes))
            incumbent = np.load(c18_path)["source"]
            same = (slot_signature(check.source)[evidence]
                    == slot_signature(incumbent)[evidence]).all(axis=1)
            print(f"  lambda = 0 reproduces c18 on {same.mean():.4%} of frames with "
                  "evidence (comparing the correspondence, not the slot order)")
            record.param("zero_penalty_matches_c18", float(same.mean()))
            if same.mean() < 0.999:
                raise SystemExit(
                    f"lambda = 0 must reproduce c18; it agrees on only {same.mean():.2%}. "
                    "The label ordering or the row map is wrong.")
            n0 = int((np.diff(zero[evidence]) != 0).sum())
        else:
            print(f"  c18 output not found at {c18_path} — the lambda = 0 check is skipped")
            n0 = int((np.diff(costs.cost.argmin(axis=1)[evidence]) != 0).sum())

        # --- diagnostics that decide nothing ---------------------------------
        pairs = int(evidence.sum()) - 1
        reacquired = sum(
            int((complete[v][1:] & ~complete[v][:-1]).sum()) for v in VIEWS)
        print("\ndiagnostics, deciding nothing")
        print(f"  N0, changes of the unconstrained per-frame winner : {n0:,} "
              f"over {pairs:,} pairs ({n0 / max(1, pairs):.4%})")
        print("  handoff's figure, on a labelling that predates c16 : 635 over "
              "284,305 (0.2234%) — tool deleted, not reproducible")
        print(f"  2D label re-acquisitions across the three views    : {reacquired:,} "
              f"(an upper bound on legitimate changes, not an estimate)")
        record.param("diagnostics", {
            "n0_changes": n0, "n0_pairs": pairs,
            "n0_rate": n0 / max(1, pairs), "reacquisitions_2d": reacquired,
            "handoff_figure_not_reproducible": {"changes": 635, "pairs": 284305},
        })

        # --- lambda, chosen without the ground truth --------------------------
        started = time.time()
        choice = select_kappa(costs, evidence)
        print(f"\nlambda by held-out node cross-validation ({time.time() - started:.0f}s)")
        print(f"{'fit':>9} {'held out':>9} {'margin m':>9} {'best kappa':>11} "
              f"{'lambda mm':>10} {'changes':>9} {'interior':>9}")
        for fit, cell in choice["curves"].items():
            index = cell["best_index"]
            print(f"{fit:>9} {cell['held_out']:>9} {cell['margin_mm']:>9.2f} "
                  f"{cell['best_kappa']:>11.4g} "
                  f"{cell['best_kappa'] * cell['margin_mm']:>10.3f} "
                  f"{cell['changes'][index]:>9,} {cell['interior']!s:>9}")
        if not choice["interior_minimum_both_directions"]:
            print("  !! a minimiser sits on the edge of the grid. The pre-registered "
                  "rule has failed and its fallback applies; see the decision document.")
        penalty = choice["lambda_mm"] if args.lambda_mm is None else args.lambda_mm
        if args.lambda_mm is not None:
            print(f"  !! --lambda-mm given: {penalty} mm is a POST-HOC choice, "
                  "not the pre-registered one")
        print(f"  kappa* {choice['kappa']:.4g}   m(two-node) "
              f"{choice['margins_mm']['total']:.2f} mm   **lambda* {penalty:.3f} mm**")
        record.param("lambda_choice", choice)
        record.param("lambda_mm_used", float(penalty))
        record.param("lambda_is_post_hoc", args.lambda_mm is not None)

        # --- the deployed sequence -------------------------------------------
        started = time.time()
        chosen = viterbi_path(costs.cost, float(penalty))
        kept = int((np.diff(chosen[evidence]) != 0).sum())
        out = reconstruct_from_states(
            rig, args.experiment, by_label, complete, row_of_label, chosen,
            c01.N_FRAMES, anchor=args.anchor, nodes=tuple(args.nodes))
        out = dataclasses.replace(out, margin_mm=costs.margin["total"])
        print(f"\nsequence solved in {time.time() - started:.0f}s   "
              f"resolved {out.resolved.mean():.2%}   "
              f"changes kept {kept:,} of N0 {n0:,}   "
              f"segments {kept + 1:,}, median {(int(evidence.sum())) / (kept + 1):,.0f} frames")
        record.param("changes_kept", kept)

        config = derive_gate(out)
        record.param("max_step_mm", config.max_step_mm)
        record.param("max_gap_frames", config.max_gap_frames)
        print(f"max_step_mm {config.max_step_mm}   max_gap_frames {config.max_gap_frames}")

        # --- the two arms ------------------------------------------------------
        label_track = np.where(
            out.resolved[:, None], np.arange(N_ANIMALS)[None, :], UNTRACKED
        ).astype(np.int32)
        arms = {"B-gated": METHODS[METHOD](out, config), "B-label": label_track}

        summary = {}
        for name, slot_track in arms.items():
            per_view = propagate_to_views(out, slot_track, starts, counts)
            path = out_dir / f"{name}__{args.experiment}.npz"
            np.savez_compressed(
                path, slot_track=slot_track, xyz=out.xyz.astype(np.float32),
                residual_mm=out.residual_mm.astype(np.float32),
                margin_mm=out.margin_mm.astype(np.float32),
                n_views_used=out.n_views_used, source=out.source,
                state=chosen.astype(np.int16),
                **{f"track__{v}": per_view[v] for v in VIEWS},
                # Detector row order, because `c04` and `c09` both read it that way.
                **{f"{node}__{v}": by_row[v][node].astype(np.float32)
                   for v in VIEWS for node in ("pectoral", "tail")},
            )
            record.output(f"{name}__{args.experiment}", path)
            run = dict(np.load(path))
            summary[name] = {
                "path": str(path),
                "reconstruction": report(run, f"c19_{name}"),
                "coverage": coverage(slot_track, c01.N_FRAMES),
                "cross_view_agreement": cross_view_agreement(rig, run),
                "max_step_mm": config.max_step_mm,
            }

        # --- the incumbents, on the identical definitions ----------------------
        for name, npz_path in (
            ("c03 mouth 1 node", paths.work_dir / "c03_track_3d" / RUN
             / f"{METHOD}__{args.experiment}.npz"),
            ("c13 pectoral 1 node", paths.work_dir / "c13_pectoral_anchor" / "pectoral"
             / f"{METHOD}__{args.experiment}.npz"),
            ("c18 2 nodes", c18_path),
        ):
            if not npz_path.exists():
                continue
            run = dict(np.load(npz_path))
            summary[name] = {
                "path": str(npz_path),
                "reconstruction": report(run, name),
                "coverage": coverage(run["slot_track"], c01.N_FRAMES),
                "cross_view_agreement": cross_view_agreement(rig, run),
            }

        print(f"\n{'arm':<22} {'resolved':>9} {'tracks':>7} {'top3 track':>11} "
              f"{'top3 runs>=1s':>14} {'pectoral agree':>15} {'tail agree':>11}")
        for name, cell in summary.items():
            agree = cell["cross_view_agreement"]
            pect = np.mean([agree[k]["agreement_rate"] for k in agree if k.startswith("pect")])
            tail = np.mean([agree[k]["agreement_rate"] for k in agree if k.startswith("tail")])
            print(f"{name:<22} {cell['reconstruction']['resolved']:>9.2%} "
                  f"{cell['coverage']['tracks']:>7,} "
                  f"{cell['coverage']['top3_track_coverage']:>11.2%} "
                  f"{cell['coverage']['top3_run_coverage']:>14.2%} "
                  f"{pect:>15.4f} {tail:>11.4f}")
        print("\nclause 5's 2D standard, from c16 on the identical run definition: "
              "94.6-97.0%")
        record.param("arms", summary)

        (out_dir / "c19_summary.json").write_text(
            json.dumps({
                "experiment": args.experiment, "n_groups": 1, "n_experiments": 1,
                "groups": [ExperimentId.parse(args.experiment).group],
                "anchor": args.anchor, "nodes": args.nodes,
                "lambda_choice": choice, "lambda_mm_used": float(penalty),
                "lambda_is_post_hoc": args.lambda_mm is not None,
                "diagnostics": {"n0_changes": n0, "n0_pairs": pairs,
                                "changes_kept": kept, "reacquisitions_2d": reacquired},
                "max_step_mm": config.max_step_mm, "arms": summary,
            }, indent=2, default=float),
            encoding="utf-8")

    print(f"\nwrote {out_dir}\nnow score both arms, then restore the baseline:")
    for name in arms:
        print(f"  tools/c09_gt_swap_validation.py --npz "
              f"{out_dir / f'{name}__{args.experiment}.npz'}")
    print("  tools/c09_gt_swap_validation.py"
          "        # no --npz: restores the JSON c11/c14/c15 read")


if __name__ == "__main__":
    main()

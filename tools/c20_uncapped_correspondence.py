"""C20 — stage C: the cross-view correspondence without the "exactly three
detections per view" rule.

=============================================================================
WHAT IS BEING REMOVED
=============================================================================

`c03`, `c13`, `c18` and `c19` all require a view to report **exactly three**
detections before it may take part in a frame. Nothing about the experiment says
that: three animals are in the tank at all times, but the detector answers with
however many instances it found. On K_1_2's bottom-up run:

    instances per frame    0      1       2         3      4      5+
    top_cam              115    559  12,841   327,977  6,925    183
    side1_cam              0  3,770  18,215   321,538  4,972    105
    side2_cam             71  4,146  14,146   327,423  2,715     99

So a frame with four detections is thrown away although all three animals are in
it, and a frame with two is thrown away although two animals are. The rule is a
consequence of indexing detections by **row**: with three rows and three slots,
a correspondence is a permutation, and any other count has no permutation to be.

`c16` already supplies the replacement. Its per-camera 2D tracker holds exactly
three track labels per view, so the *label* says which detection is which, and a
label is either present in a frame or absent — no counting. A fourth detection
carries no label and is ignored; a two-detection frame supplies two labels and
the third is absent. `c19` already indexes its state space by those labels, so
**the state space does not change and the algorithm does not change**: same 36
states, same objective

    J(s) = SUM_t cost_t(s_t)  +  lambda * #{t : s_t != s_{t-1}},

same exact Viterbi, same held-out-node choice of lambda. The only difference is
which detections are allowed to reach it.

Participation becomes per animal rather than per camera. The slot for top label
*a* is triangulated from whichever views hold its label under the state being
scored, so one camera losing one animal weakens **that animal's** slot by one ray
and leaves the other two untouched — `c13`'s "a missing node costs a node, never
a camera", along the other axis.

=============================================================================
WHAT THIS MUST REPRODUCE, AND THE TOOL ASSERTS IT
=============================================================================

On a frame where every view holds every label the new cost is the old cost
divided by three — a mean over nine terms where `c19` took a mean over three
sums of three — so the per-frame winner, the runner-up *ratio* and therefore the
cross-validated kappa are unchanged, and lambda simply moves by the same factor.
That is checked against a live `state_costs` run on the 83.7% of frames where
the evidence is identical, and, through `c18`'s stored output, so is the
correspondence itself. If either fails, the label bookkeeping is wrong, which is
silent on real data.

=============================================================================
TWO ARMS, THE SAME TWO AS C19
=============================================================================

``C-gated``   `gated` on the stage-C reconstruction, gate re-derived from it.
``C-label``   track ID = the ``top_cam`` 2D label, held constant.

Nothing is promoted. `c03`, `c13`, `c16`, `c18` and `c19` are not modified. The
five-clause rule decides, and clause 1 needs `c09` run against these outputs —
which overwrites `c09`'s fixed output path, so re-run `c09` with no ``--npz``
afterwards to restore the baseline `c11`/`c14`/`c15` read.

**n_groups = 1, n_experiments = 1, female only. Swap counts of 2-8 are small
samples and no ranking built on them is stable.**

=============================================================================
WHAT IT FOUND (2026-08-19, K_1_2) — NOTHING IS PROMOTED
=============================================================================

The reconstruction half succeeded: resolved 97.53% -> **98.55%** against a hard
ceiling of 98.93%, frames with three cameras on every animal 83.68% -> **87.50%**,
and the reduction to `c19` is exact where the evidence is identical.

The sequence half did not. The admitted frames triple the state changes the
Viterbi keeps, 973 -> **3,090**, and that is not lambda's rescaling — at `c19`'s
own effective penalty it is still 2,939. Every frame enters the objective with
the same weight because the cost is a *mean* over its terms, so a frame resting
on four rays pushes as hard as one resting on nine.

The derived gate then goes 4.0 -> **72.2 mm**, because `derive_gate` takes the
99.9th percentile of the 3D step and a step across a correspondence change is a
**renaming, not a movement** (p99.9 over changed pairs: 177.7 mm). At that gate
`gated` stops refusing anything and swaps go 2 -> 8; with the gate held at
`c19`'s 4.0 mm the same reconstruction takes **1 swap**, the lowest in the
project, at 37 breaks. See the decision document for the two repairs this
implies, both of which are separate decisions.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c20_uncapped_correspondence.py
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
from c13_pectoral_anchor import report
from c19_viterbi_correspondence import (
    KAPPA,
    TRACKER_2D,
    coverage,
    derive_gate,
    label_ordered,
    select_kappa,
    slot_signature,
)

from medaka3d import load_paths
from medaka3d.calib import load_rig, triangulate
from medaka3d.cohort import ExperimentId
from medaka3d.correspond import (
    label_costs,
    reconstruct_from_label_states,
    state_costs,
    viterbi_path,
)
from medaka3d.eval.tracking import modal_agreement
from medaka3d.provenance import provenance
from medaka3d.reconstruct import N_ANIMALS, PERMUTATIONS, UNRESOLVED, hypotheses
from medaka3d.track import METHODS, UNTRACKED, propagate_to_views
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN, METHOD = "bottomup", "gated"


def label_arrays(detections, labels: np.ndarray):
    """Lay a view's detections out by 2D track label, with no cardinality rule.

    Returns ``(by_label, row_of_label, present, stats)``:

    ``by_label``      ``node -> (n_frames, 3, 2)`` pixels, NaN where this frame
                      holds no detection with that label.
    ``row_of_label``  (n_frames, 3) the detector's own row offset within the
                      frame for each label, ``UNRESOLVED`` where absent. `c09`
                      inverts this through ``source``; handing it anything but a
                      row would pair one animal's 3D position with another
                      animal's pixels.
    ``present``       (n_frames, 3) which labels the frame holds at all.

    Two labels on one detection, or one label twice in a frame, would make the
    layout ambiguous. Neither can happen with `sleap_nn`'s tracker, so both are
    checked here rather than assumed — a silently overwritten row is exactly the
    kind of error this whole file exists to avoid.
    """
    n_frames = detections.n_frames
    start = detections.start
    live = labels >= 0
    frame = detections.frame_idx[live]
    label = labels[live]
    offset = (np.arange(labels.size) - start[detections.frame_idx])[live]

    key = frame.astype(np.int64) * N_ANIMALS + label
    if np.unique(key).size != key.size:
        raise SystemExit("a 2D track label appears twice in one frame; the layout is ambiguous")

    row_of_label = np.full((n_frames, N_ANIMALS), UNRESOLVED, np.int8)
    row_of_label[frame, label] = offset.astype(np.int8)
    by_label = {}
    for node in detections.nodes:
        xy = np.full((n_frames, N_ANIMALS, 2), np.nan)
        xy[frame, label] = detections.node(node)[live]
        by_label[node] = xy

    counts = detections.counts()
    present = row_of_label >= 0
    stats = {
        "instances": int(labels.size),
        "unlabelled_instances": int((~live).sum()),
        "instances_per_frame": {int(k): int(v) for k, v in
                                zip(*np.unique(counts, return_counts=True), strict=True)},
        "labels_per_frame": {int(k): float(v / n_frames) for k, v in
                             zip(*np.unique(present.sum(axis=1), return_counts=True),
                                 strict=True)},
        "frames_not_exactly_three": float((counts != N_ANIMALS).mean()),
        "frames_with_all_three_labels": float(present.all(axis=1).mean()),
    }
    return by_label, row_of_label, present, stats


def node_pairing_by_slot(rig, run: dict, node: str, reference: str, other: str) -> np.ndarray:
    """`c04.node_pairing`, expressed in slots because rows are no longer a permutation.

    `c04`'s version indexes each view's three points by detector row and inverts
    ``source`` with an argsort — which is only an inverse when the rows are
    exactly {0, 1, 2}. Once a frame may hold four detections, or two, that is no
    longer true. Storing the points already in slot order says the same thing
    without the inversion: the correspondence claims slot *i* of ``reference`` is
    slot *i* of ``other``, so re-deriving the pairing from a node the tracker did
    not link on and asking whether the winner is the identity is exactly `c04`'s
    question.
    """
    slot_track = run["slot_track"]
    out = np.full(slot_track.shape, -1, dtype=np.int64)
    columns = {view: i for i, view in enumerate(rig.views)}
    source = run["source"]
    frames = np.flatnonzero(
        (slot_track.max(axis=1) >= 0)
        & (source[:, :, columns[reference]] >= 0).all(axis=1)
        & (source[:, :, columns[other]] >= 0).all(axis=1)
    )
    if frames.size == 0:
        return out

    xy_ref = run[f"{node}_by_slot__{reference}"][frames].astype(float)
    xy_other = run[f"{node}_by_slot__{other}"][frames].astype(float)
    present = np.isfinite(xy_ref).all(axis=(1, 2)) & np.isfinite(xy_other).all(axis=(1, 2))
    frames, xy_ref, xy_other = frames[present], xy_ref[present], xy_other[present]
    if frames.size == 0:
        return out

    o_ref, d_ref = rig.camera(reference).rays(xy_ref)
    o_oth, d_oth = rig.camera(other).rays(xy_other)
    costs = []
    for perm in PERMUTATIONS:
        _, rms = triangulate(
            np.stack([o_ref, o_oth[:, perm, :]], axis=-2),
            np.stack([d_ref, d_oth[:, perm, :]], axis=-2),
        )
        costs.append(rms.sum(axis=-1))
    winner = np.asarray(PERMUTATIONS)[np.argmin(np.where(np.isfinite(costs), costs, np.inf), 0)]
    out[frames] = (winner == np.arange(N_ANIMALS)[None, :]).astype(np.int64)
    return out


def cross_view_agreement(rig, run: dict) -> dict:
    """Clause 3. Re-derive the pairing from a node the tracker did not link on."""
    out = {}
    for node in ("pectoral", "tail"):
        for other in rig.views[1:]:
            stats = modal_agreement(node_pairing_by_slot(rig, run, node, rig.views[0], other))
            out[f"{node}:{rig.views[0]}~{other}"] = stats
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--anchor", default="pectoral", choices=("pectoral", "mouth"))
    parser.add_argument("--nodes", nargs="+", default=["mouth", "pectoral"])
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--lambda-mm", type=float, default=None,
                        help="Override the cross-validated penalty. Using this makes the "
                             "run a post-hoc choice and it must be reported as one.")
    parser.add_argument("--gate-mm", type=float, default=None,
                        help="Hold `max_step_mm` at this value instead of deriving it. A "
                             "threshold, not a switch: the same `gated` runs either way. "
                             "Using it makes the gate a POST-HOC choice and the run is "
                             "written to its own directory so the pre-registered one stands.")
    parser.add_argument("--max-gap-frames", type=int, default=None,
                        help="Hold `max_gap_frames` likewise.")
    parser.add_argument("--skip-equivalence", action="store_true",
                        help="Skip the check against c19's cost. Only for a machine "
                             "without the memory for both cost matrices at once.")
    args = parser.parse_args()

    paths = load_paths()
    rig = load_rig(CALIBRATION)
    nodes = tuple(args.nodes)
    tag = "-".join(sorted(nodes)) + f"_anchor-{args.anchor}"
    held = "" if args.gate_mm is None else f"_gate-{args.gate_mm:g}"
    out_dir = paths.work_dir / "c20_uncapped_correspondence" / (tag + held)
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("uncapped_correspondence", out_dir, name=f"c20_{tag}") as record:
        record.params_from(args)
        record.input("calibration", rig.manifest)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        print(f"experiment {args.experiment}   anchor {args.anchor}   nodes {list(nodes)}")
        print("n_groups = 1, n_experiments = 1 — K_1_2, female only\n")

        by_label, row_of_label, present, capped = {}, {}, {}, {}
        old_uv, old_complete, by_row = {}, {}, {}
        starts, counts, census = {}, {}, {}
        for view in VIEWS:
            detections = c01.load(paths.work_dir, args.experiment, view, RUN)
            labels = np.load(paths.work_dir / TRACKER_2D / f"track2d__{view}.npy")
            by_label[view], row_of_label[view], present[view], census[view] = label_arrays(
                detections, labels)
            # The incumbent's own layout, for the equivalence check only.
            old_uv[view], by_row[view], _, old_complete[view], _ = label_ordered(
                detections, labels, args.anchor)
            capped[view] = old_complete[view]
            starts[view] = detections.start
            counts[view] = int(labels.size)
            anchor_present = np.isfinite(by_label[view][args.anchor]).all(axis=2)
            not_three = census[view]["frames_not_exactly_three"]
            print(f"  {view:<10} not exactly three detections on {not_three:6.2%} of frames"
                  f"   usable: c19 {capped[view].mean():6.2%} -> c20 "
                  f"{anchor_present.all(axis=1).mean():6.2%}")

        started = time.time()
        costs = label_costs(rig, by_label, c01.N_FRAMES, anchor=args.anchor, nodes=nodes)
        evidence = costs.terms.max(axis=1) > 0
        resolvable = costs.resolvable()
        print(f"\n  {costs.n_states} states x {c01.N_FRAMES:,} frames built in "
              f"{time.time() - started:.0f}s")
        print(f"  frames with any evidence                 {evidence.mean():7.2%}")
        print(f"  frames where some state gives every animal two cameras "
              f"{resolvable.mean():7.2%}")
        terms_vary = costs.terms.min(axis=1) != costs.terms.max(axis=1)
        print(f"  frames whose states differ in how much they explain "
              f"{terms_vary.mean():7.2%}  (the stated bias)")
        record.param("frames_with_evidence", float(evidence.mean()))
        record.param("frames_resolvable", float(resolvable.mean()))
        record.param("frames_terms_vary", float(terms_vary.mean()))
        record.param("cardinality_census", census)

        # --- the assertion that this is c19's problem with more of the data ---
        # The evidence is identical only where every view holds every label **of
        # every node**. Where a view is short of one mouth, `c19` drops the mouth
        # from the whole frame while stage C keeps it for the two animals whose
        # mouths are there, so the two must differ — that difference is measured
        # below rather than asserted away.
        old_all = np.stack([capped[v] for v in VIEWS], axis=1).all(axis=1)
        identical = old_all.copy()
        for view in VIEWS:
            for node in nodes:
                identical &= np.isfinite(by_label[view][node]).all(axis=2).all(axis=1)
        if not args.skip_equivalence:
            started = time.time()
            old = state_costs(rig, old_uv, capped, c01.N_FRAMES, nodes=nodes)
            gap = float(np.abs(costs.cost[identical] * N_ANIMALS - old.cost[identical]).max())
            same_pick = float((costs.cost[identical].argmin(axis=1)
                               == old.cost[identical].argmin(axis=1)).mean())
            partial = old_all & ~identical
            moved = float((costs.cost[partial].argmin(axis=1)
                           != old.cost[partial].argmin(axis=1)).mean())
            print(f"\n  against c19's cost ({time.time() - started:.0f}s)")
            print(f"    identical evidence, {identical.mean():6.2%} of frames: "
                  f"max |{N_ANIMALS} x new - old| {gap:.2e} mm, winner equal {same_pick:.4%}")
            print(f"    a view short of one mouth, {partial.mean():6.2%} of frames: "
                  f"winner moves on {moved:.2%} — stage C keeps the mouths that are there, "
                  "c19 drops all three")
            record.param("equivalence", {
                "identical_frames": float(identical.mean()), "max_abs_diff_mm": gap,
                "winner_equal": same_pick, "partial_node_frames": float(partial.mean()),
                "winner_moves_on_partial_node_frames": moved})
            if not (gap < 1e-3 and same_pick == 1.0):
                raise SystemExit(
                    "stage C must reduce to c19 where the evidence is the same; it does not. "
                    "The label layout or the masked triangulation is wrong.")
            del old

        c18_path = (paths.work_dir / "c18_multinode_correspondence" / tag
                    / f"{METHOD}__{args.experiment}.npz")
        zero = viterbi_path(costs.cost, 0.0)
        if c18_path.exists():
            check, _ = reconstruct_from_label_states(
                rig, args.experiment, by_label, row_of_label, zero, c01.N_FRAMES,
                anchor=args.anchor, nodes=nodes)
            incumbent = np.load(c18_path)["source"]
            here = identical & check.resolved
            same = (slot_signature(check.source)[here]
                    == slot_signature(incumbent)[here]).all(axis=1)
            print(f"  lambda = 0 reproduces c18's correspondence on {same.mean():.4%} of "
                  "those frames")
            record.param("zero_penalty_matches_c18", float(same.mean()))
            if same.mean() < 0.999:
                raise SystemExit(
                    f"lambda = 0 must reproduce c18 where the evidence is identical; it "
                    f"agrees on only {same.mean():.2%}.")
        n0 = int((np.diff(zero[evidence]) != 0).sum())
        pairs = int(evidence.sum()) - 1
        print(f"\n  N0, changes of the unconstrained per-frame winner : {n0:,} over "
              f"{pairs:,} pairs ({n0 / max(1, pairs):.4%})   [c19: 1,941 / 340,001]")
        record.param("diagnostics", {"n0_changes": n0, "n0_pairs": pairs})

        # --- lambda, chosen without the ground truth, exactly as in c19 -------
        started = time.time()
        choice = select_kappa(costs, evidence)
        print(f"\nlambda by held-out node cross-validation ({time.time() - started:.0f}s)")
        print(f"{'fit':>9} {'held out':>9} {'margin m':>9} {'best kappa':>11} "
              f"{'lambda mm':>10} {'changes':>9} {'interior':>9}")
        for fit, cell in choice["curves"].items():
            index = cell["best_index"]
            print(f"{fit:>9} {cell['held_out']:>9} {cell['margin_mm']:>9.3f} "
                  f"{cell['best_kappa']:>11.4g} "
                  f"{cell['best_kappa'] * cell['margin_mm']:>10.4f} "
                  f"{cell['changes'][index]:>9,} {cell['interior']!s:>9}")
        if not choice["interior_minimum_both_directions"]:
            print("  !! a minimiser sits on the edge of the grid. The pre-registered rule "
                  "has failed and its fallback applies; see the decision document.")
        penalty = choice["lambda_mm"] if args.lambda_mm is None else args.lambda_mm
        if args.lambda_mm is not None:
            print(f"  !! --lambda-mm given: {penalty} mm is a POST-HOC choice")
        print(f"  kappa* {choice['kappa']:.4g}   m(two-node) {choice['margins_mm']['total']:.3f} mm"
              f"   **lambda* {penalty:.4f} mm**   (c19, on its x{N_ANIMALS} scale: "
              f"kappa* 0.2371, lambda* 2.766 mm)")
        record.param("lambda_choice", choice)
        record.param("lambda_mm_used", float(penalty))
        record.param("lambda_is_post_hoc", args.lambda_mm is not None)
        record.param("kappa_grid", KAPPA.tolist())

        # --- the deployed sequence --------------------------------------------
        started = time.time()
        chosen = viterbi_path(costs.cost, float(penalty))
        kept = int((np.diff(chosen[evidence]) != 0).sum())
        out, slot_rays = reconstruct_from_label_states(
            rig, args.experiment, by_label, row_of_label, chosen, c01.N_FRAMES,
            anchor=args.anchor, nodes=nodes)
        # Reported on the incumbents' scale — their margin is the sum over the
        # three animals where this solver's is the mean — so the percentiles in
        # the table below compare with c18's and c19's without a conversion.
        out = dataclasses.replace(out, margin_mm=costs.margin["total"] * N_ANIMALS)
        print(f"\nsequence solved in {time.time() - started:.0f}s   "
              f"resolved {out.resolved.mean():.2%}   changes kept {kept:,} of N0 {n0:,}")
        print(f"  cameras per animal on resolved frames: "
              f"3 in {(slot_rays[out.resolved] == 3).mean():.2%}, "
              f"2 in {(slot_rays[out.resolved] == 2).mean():.2%}")
        record.param("changes_kept", kept)
        record.param("slot_rays_three", float((slot_rays[out.resolved] == 3).mean()))

        config = derive_gate(out)
        if args.gate_mm is not None or args.max_gap_frames is not None:
            # The gate `derive_gate` produces is a property of *this*
            # reconstruction, so comparing two reconstructions through it
            # compares gates as much as methods — the confound clause 2 already
            # names. Holding it at another run's value is the controlled version
            # of the same comparison, and it is post-hoc.
            config = dataclasses.replace(
                config,
                max_step_mm=config.max_step_mm if args.gate_mm is None else args.gate_mm,
                max_gap_frames=(config.max_gap_frames if args.max_gap_frames is None
                                else args.max_gap_frames))
            print(f"  !! gate held at {config.max_step_mm} mm / {config.max_gap_frames} "
                  f"frames — a POST-HOC choice; the derived one was "
                  f"{derive_gate(out).max_step_mm} mm")
        record.param("gate_is_post_hoc", args.gate_mm is not None)
        record.param("max_step_mm", config.max_step_mm)
        record.param("max_gap_frames", config.max_gap_frames)
        print(f"max_step_mm {config.max_step_mm}   max_gap_frames {config.max_gap_frames}")

        # --- the two arms ------------------------------------------------------
        label_track = np.where(
            out.resolved[:, None], np.arange(N_ANIMALS)[None, :], UNTRACKED
        ).astype(np.int32)
        arms = {"C-gated": METHODS[METHOD](out, config), "C-label": label_track}

        by_slot = {}
        rows = np.arange(c01.N_FRAMES)[:, None]
        state_labels = hypotheses(len(VIEWS))[chosen]  # (n_frames, n_animals, n_views)
        for index, view in enumerate(VIEWS):
            for node in ("pectoral", "tail"):
                by_slot[f"{node}_by_slot__{view}"] = by_label[view][node][
                    rows, state_labels[:, :, index]].astype(np.float32)

        summary = {}
        for name, slot_track in arms.items():
            per_view = propagate_to_views(out, slot_track, starts, counts)
            path = out_dir / f"{name}__{args.experiment}.npz"
            np.savez_compressed(
                path, slot_track=slot_track, xyz=out.xyz.astype(np.float32),
                residual_mm=out.residual_mm.astype(np.float32),
                margin_mm=out.margin_mm.astype(np.float32),
                n_views_used=out.n_views_used, source=out.source,
                slot_rays=slot_rays, state=chosen.astype(np.int16), **by_slot,
                **{f"track__{v}": per_view[v] for v in VIEWS},
            )
            record.output(f"{name}__{args.experiment}", path)
            run = dict(np.load(path))
            summary[name] = {
                "path": str(path),
                "reconstruction": report(run, f"c20_{name}"),
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
            ("c19 B-gated", paths.work_dir / "c19_viterbi_correspondence" / tag
             / f"B-gated__{args.experiment}.npz"),
            ("c19 B-label", paths.work_dir / "c19_viterbi_correspondence" / tag
             / f"B-label__{args.experiment}.npz"),
        ):
            if not npz_path.exists():
                continue
            run = dict(np.load(npz_path))
            summary[name] = {
                "path": str(npz_path),
                "reconstruction": report(run, name),
                "coverage": coverage(run["slot_track"], c01.N_FRAMES),
            }

        print(f"\n{'arm':<22} {'resolved':>9} {'3 cams':>8} {'tracks':>7} {'top3 track':>11} "
              f"{'top3 runs>=1s':>14} {'margin p1':>10}")
        for name, cell in summary.items():
            stats = cell["reconstruction"]
            three = stats.get("views_3", {}).get("share", float("nan"))
            print(f"{name:<22} {stats['resolved']:>9.2%} {three:>8.2%} "
                  f"{cell['coverage']['tracks']:>7,} "
                  f"{cell['coverage']['top3_track_coverage']:>11.2%} "
                  f"{cell['coverage']['top3_run_coverage']:>14.2%} "
                  f"{stats.get('views_3', {}).get('margin_p1', float('nan')):>10.3f}")
        print("\nclause 5's 2D standard, from c16 on the identical run definition: 94.6-97.0%")
        record.param("arms", summary)

        (out_dir / "c20_summary.json").write_text(
            json.dumps({
                "experiment": args.experiment, "n_groups": 1, "n_experiments": 1,
                "groups": [ExperimentId.parse(args.experiment).group],
                "anchor": args.anchor, "nodes": list(nodes),
                "cardinality_census": census,
                "frames_with_evidence": float(evidence.mean()),
                "frames_resolvable": float(resolvable.mean()),
                "frames_terms_vary": float(terms_vary.mean()),
                "lambda_choice": choice, "lambda_mm_used": float(penalty),
                "lambda_is_post_hoc": args.lambda_mm is not None,
                "diagnostics": {"n0_changes": n0, "n0_pairs": pairs, "changes_kept": kept},
                "max_step_mm": config.max_step_mm, "arms": summary,
            }, indent=2, default=float),
            encoding="utf-8")

    print(f"\nwrote {out_dir}\nnow score both arms, then restore the baseline:")
    for name in arms:
        print(f"  tools/c09_gt_swap_validation.py --npz "
              f"{out_dir / f'{name}__{args.experiment}.npz'}")
    print("  tools/c09_gt_swap_validation.py        # no --npz: restores the JSON "
          "c11/c14/c15 read")


if __name__ == "__main__":
    main()

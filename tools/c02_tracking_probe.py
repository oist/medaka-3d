"""C02 — track in 2D or in 3D? Measured on the two correspondence problems, then decided.

=============================================================================
SPECIFICATION — fixed in code before the first measurement
=============================================================================

**The fork.** ``sleap_nn.predict`` takes a ``tracker_config`` and tracks each 2D
view independently; the project also holds a validated refractive calibration of
all three cameras. Both are real options and they are not variations of one
method, so only one is built.

* **2D then lift.** Each view solves its own frame-to-frame correspondence, and
  the three views are reconciled afterwards. Simple, and it is what the detector
  hands you. Its failure mode is structural: two animals that overlap in *this*
  camera are ambiguous in *this* camera, and each view fails at different
  moments, so the three views disagree about who is who exactly when it matters.
* **3D.** Solve the cross-view correspondence per frame, triangulate, then track
  three points in a 20 x 20 x 20 cm volume. Two animals that overlap in one
  camera are usually metres apart in depth, so the temporal problem should be
  strictly easier — but it is paid for with a *new* per-frame problem the 2D
  route never has to solve, and with the frames where any one view failed to
  produce three detections.

**Neither is assumed. The two correspondence problems are measured against each
other, on the same frames, in this tool, and the decision rule below is applied
as written.**

**What is measured.** Four quantities, on contiguous blocks of the s16
predictions (see ``BLOCKS``), using the ``mouth`` node throughout — it is the
detector's anchor and the node ``s10``'s permutation metric already links on:

1. **Usable fraction.** Per view, the rate of frames with exactly three
   detections; and the rate of *moments* where all three views have exactly
   three at once. 3D pays the joint rate; 2D pays the per-view rate. Context,
   not a criterion — a fallback for degenerate frames is a tracker feature, and
   this tool does not pretend the fallback does not exist.

2. **2D temporal margin**, per view. Between adjacent frames that both hold
   exactly three instances, enumerate all six assignments, score each by total
   mouth displacement in px, and take the best and the runner-up.

3. **3D cross-view margin.** On moments where all three views hold exactly
   three, enumerate all 36 correspondences (3! for side1 x 3! for side2, with
   top fixed as the reference), score each by the summed point-to-ray residual
   in mm of its three triangulated mouths, and take the best and the runner-up.
   This is the problem 3D adds.

4. **3D temporal margin.** Between adjacent resolvable moments, the same
   enumeration as (2) but over triangulated 3D mouths in mm. This is the
   problem 3D improves.

**Ambiguity is defined once, scale-free, and derived from the data.** A
correspondence is **ambiguous** when the runner-up costs less than one typical
winning cost more than the winner::

    ambiguous  <=>  (second_best - best) < median(best over this sample)

The units differ between the metrics (px for 2D, mm for 3D) and the definition
does not care, because the yardstick is each metric's own median winning cost.
Every yardstick used is printed. Nothing is hardcoded — in particular the 2D
thresholds are not the 60/50 px tail-jump thresholds, which measure something
else.

**Decision rule, fixed before the run.** Let ``a2d`` be the *lowest* of the three
views' 2D temporal ambiguity rates — the best case for 2D, not the average, so
2D is given the benefit of the doubt. Then:

    Adopt 3D  iff  (3D temporal ambiguity < a2d)  AND  (3D cross-view ambiguity < a2d)

    Adopt 2D  otherwise.

The second clause is the one that matters. A 3D tracker whose *own* per-frame
correspondence is more ambiguous than the 2D temporal problem it was meant to
replace has imported a harder problem than it solved, no matter how clean its
temporal step looks.

**The rule floored, and section 5 exists because of that.** On the first run both
temporal rates came out at exactly 0.000%: with a median winning cost of 4.2 px
and a median runner-up of 245 px, the yardstick "within one typical winning cost"
can never be met, so the statistic measured its own denominator rather than the
two routes. That is precisely the failure `s10` documented for shared-node across
architectures, and the response here is the one that decision prescribes — the
pre-registered rule's output is printed **unaltered**, and the quantities that
can discriminate are added below it and labelled as what they are: chosen after
seeing the floor.

Section 5 measures three things the pre-registered rule could not:

* **Credible-rival rate.** A runner-up is a real rival only if it costs no more
  than the winning hypothesis usually costs — concretely, the 99.9th percentile
  of the winners in that same metric. One criterion for all four metrics, derived
  from each metric's own data, so px and mm need no common threshold. `s10`'s
  half-body-span gate is also reported for the two *motion* metrics, unchanged,
  for continuity; it is deliberately **not** applied to the cross-view metric,
  because a triangulation residual is not a displacement and gating one with the
  other is a category error.
* **Margin percentiles.** The low tail is where crossings live; a median hides it.
* **Recoverability.** The real 2D failure is not the assignment in an easy frame,
  it is a view losing an animal and having to re-acquire it across a gap. A view
  that drops to two detections while the other two views still hold three is
  recoverable in 3D and unrecoverable in that view alone. That rate, and the
  two-view-sufficient fraction (triangulation needs two cameras, not three), is
  what actually separates the routes.

**n_groups = 1.** Every frame here is group K. ``K_1_1``, ``K_1_2``, ``K_1_3``
are the same trio on three consecutive days — repeated measures, not replicates.
Whichever route this picks, it is picked on one trio out of the corpus's 14
groups.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c02_tracking_probe.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from s10_shared_node_20k import (
    block_frames,
    block_starts,
    median_span_px,
    tail_jump_threshold_px,
)

from medaka3d import load_catalog, load_paths
from medaka3d.calib import load_rig, triangulate
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

#: The promoted fixed-refractive calibration. `load_rig` refuses any other backend.
CALIBRATION = Path(
    r"Z:\StephensU\medaka\calibration\fixed_refractive_192x192x186"
    r"\phase1-calibration-preserve-20260802T180200JST-60e13a7746d1\calibration"
)

#: Same block geometry as s10: five contiguous blocks at the centre of each fifth.
#: Contiguity is required — every metric here reads adjacent frames — and centring
#: keeps the sample off the handling period at frame 0, which s16 showed is the one
#: stretch where detection genuinely degrades.
BLOCKS = 5
BLOCK_LEN = 1000

#: Top is the reference view: its three instances define slots 0, 1, 2, and the
#: 36 hypotheses permute the two side views against it.
PERMUTATIONS = tuple(itertools.permutations(range(3)))


def three_instance_mask(detections: c01.Detections) -> np.ndarray:
    return detections.counts() == 3


def nodes_by_frame(detections: c01.Detections, frames: np.ndarray, node: str) -> np.ndarray:
    """(n_frames, 3, 2) pixels of one node, for frames known to hold three instances."""
    start = np.searchsorted(detections.frame_idx, frames, side="left")
    xy = detections.node(node)
    return np.stack([xy[s : s + 3] for s in start], axis=0).astype(float)


def best_two(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and second-lowest along the last axis, NaN-safe."""
    ordered = np.sort(np.where(np.isfinite(costs), costs, np.inf), axis=-1)
    best = np.where(np.isfinite(ordered[..., 0]), ordered[..., 0], np.nan)
    second = np.where(np.isfinite(ordered[..., 1]), ordered[..., 1], np.nan)
    return best, second


def assignment_costs(here: np.ndarray, there: np.ndarray) -> np.ndarray:
    """(n, 6) total displacement of each of the six assignments between two frames.

    ``here`` and ``there`` are (n, 3, d). No Hungarian needed: with three animals
    the six permutations enumerate exhaustively, and the runner-up — which a
    Hungarian solver discards — is the entire point of this measurement.
    """
    per = np.linalg.norm(here[:, :, None, :] - there[:, None, :, :], axis=-1)  # (n, 3, 3)
    return np.stack([per[:, [0, 1, 2], p].sum(axis=1) for p in PERMUTATIONS], axis=-1)


#: A runner-up counts as a *credible rival* if it costs no more than the 99.9th
#: percentile of what the winning hypothesis costs. One criterion for all four
#: metrics, derived from each metric's own winners, so px and mm need no
#: conversion and no imported threshold. Chosen post-hoc — see the module
#: docstring — and stated once here rather than tuned per metric.
CREDIBLE_PERCENTILE = 99.9


def ambiguity(best: np.ndarray, second: np.ndarray, span_gate: float | None = None) -> dict:
    """The pre-registered ambiguity statistic, plus the post-hoc credible-rival rate.

    ``span_gate`` is `s10`'s half-body-span threshold and applies only to the two
    *motion* metrics; a triangulation residual is not a displacement and gating
    one with the other is a category error, so the cross-view metric passes None.
    """
    finite = np.isfinite(best) & np.isfinite(second)
    best, second = best[finite], second[finite]
    yardstick = float(np.median(best)) if best.size else float("nan")
    margin = second - best
    credible = float(np.percentile(best, CREDIBLE_PERCENTILE)) if best.size else float("nan")
    out = {
        "n": int(best.size),
        "yardstick_median_best": yardstick,
        "median_best": yardstick,
        "median_second": float(np.median(second)) if second.size else float("nan"),
        "median_margin": float(np.median(margin)) if margin.size else float("nan"),
        "ambiguous": int((margin < yardstick).sum()),
        "ambiguous_rate": float((margin < yardstick).mean()) if margin.size else float("nan"),
        "credible_threshold": credible,
        "credible_rivals": int((second <= credible).sum()),
        "credible_rate": float((second <= credible).mean()) if second.size else float("nan"),
    }
    for label, q in (("p0.1", 0.1), ("p1", 1.0), ("p5", 5.0)):
        out[f"margin_{label}"] = float(np.percentile(margin, q)) if margin.size else float("nan")
    if span_gate is not None:
        rival = second < 3.0 * span_gate  # three animals, so the total-cost gate is 3x
        out["span_gate"] = float(span_gate)
        out["span_gated_rate"] = float(rival.mean()) if rival.size else float("nan")
    return out


def show(label: str, stats: dict, unit: str) -> None:
    print(
        f"  {label:<30} n={stats['n']:>7,}  best {stats['median_best']:7.3f} {unit}  "
        f"2nd {stats['median_second']:8.2f}  margin p0.1 {stats['margin_p0.1']:8.2f} "
        f"p1 {stats['margin_p1']:8.2f} p5 {stats['margin_p5']:8.2f}  "
        f"ambig {100 * stats['ambiguous_rate']:6.3f}%  "
        f"credible rival {100 * stats['credible_rate']:6.3f}% "
        f"(<= {stats['credible_threshold']:.1f} {unit})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--experiments", nargs="*", default=list(c01.EXPERIMENTS))
    parser.add_argument("--blocks", type=int, default=BLOCKS)
    parser.add_argument("--block-len", type=int, default=BLOCK_LEN)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--batch", type=int, default=2000, help="Frames per triangulation batch.")
    args = parser.parse_args()

    paths = load_paths()
    out_dir = paths.work_dir / "c02_tracking_probe"
    rig = load_rig(args.calibration)

    with provenance("tracking_probe", out_dir, name="c02") as record:
        record.params_from(args)
        record.input("calibration", rig.manifest)
        groups = sorted({ExperimentId.parse(e).group for e in args.experiments})
        starts = block_starts(c01.N_FRAMES, args.blocks, args.block_len)
        plan = np.asarray(
            [i for block in block_frames(starts, args.block_len) for i in block], dtype=np.int64
        )
        record.param("block_starts", starts)
        record.param("n_groups", len(groups))
        print(f"calibration : {rig.manifest}")
        print(f"views       : {rig.views}")
        print(f"frames      : {args.blocks} x {args.block_len} per experiment at {starts}")
        print(f"experiments : {args.experiments}")
        print(f"n_groups    : {len(groups)} {groups} — one trio, three consecutive days\n")

        detections = {
            (e, v): c01.load(paths.work_dir, e, v) for e in args.experiments for v in VIEWS
        }

        # --- 1. usable fraction -------------------------------------------
        eq3 = {key: three_instance_mask(det) for key, det in detections.items()}
        usable = {}
        for view in VIEWS:
            rate = float(np.mean([eq3[(e, view)][plan].mean() for e in args.experiments]))
            usable[view] = rate
        joint = {
            e: eq3[(e, VIEWS[0])] & eq3[(e, VIEWS[1])] & eq3[(e, VIEWS[2])]
            for e in args.experiments
        }
        usable["all_three_views"] = float(
            np.mean([joint[e][plan].mean() for e in args.experiments])
        )
        # Triangulation needs two cameras, not three. Charging 3D the three-view
        # rate would be charging it for a constraint it does not have.
        pairwise = {
            e: (
                eq3[(e, VIEWS[0])].astype(int)
                + eq3[(e, VIEWS[1])].astype(int)
                + eq3[(e, VIEWS[2])].astype(int)
            )
            >= 2
            for e in args.experiments
        }
        usable["at_least_two_views"] = float(
            np.mean([pairwise[e][plan].mean() for e in args.experiments])
        )
        print("1. usable fraction — frames with exactly three detections")
        for name, rate in usable.items():
            print(f"  {name:<34} {100 * rate:6.2f}%")
        print(
            f"  2D pays the per-view rate; 3D pays {100 * usable['all_three_views']:.2f}% "
            f"for three views, {100 * usable['at_least_two_views']:.2f}% for two.\n"
        )

        # --- thresholds, derived from the labels, printed -----------------
        catalog = load_catalog(paths.asset_root)
        gate_px: dict[str, float] = {}
        print("gates — half a median body span, from the labels, per view (s10's derivation)")
        for view in VIEWS:
            key = f"labels.round8.{'top' if view == 'top_cam' else 'side'}.train"
            label_path = catalog.get(key)
            record.input(key, label_path)
            span, n_span = median_span_px(label_path)
            gate_px[view] = tail_jump_threshold_px(span)
            print(
                f"  {view:<12} median span {span:7.2f} px over {n_span:>5} instances "
                f"-> gate {gate_px[view]:.0f} px"
            )
            record.param(f"median_span_px_{view}", round(span, 3))
            record.param(f"gate_px_{view}", gate_px[view])
        print()

        # --- 2. 2D temporal margin ----------------------------------------
        print("2. 2D temporal margin — adjacent frames, six assignments, mouth displacement")
        temporal_2d = {}
        for view in VIEWS:
            best_all, second_all = [], []
            for experiment in args.experiments:
                det = detections[(experiment, view)]
                mask = eq3[(experiment, view)]
                pairs = plan[mask[plan] & mask[np.minimum(plan + 1, det.n_frames - 1)]]
                pairs = pairs[np.isin(pairs + 1, plan)]  # stay inside a block
                if pairs.size == 0:
                    continue
                here = nodes_by_frame(det, pairs, "mouth")
                there = nodes_by_frame(det, pairs + 1, "mouth")
                best, second = best_two(assignment_costs(here, there))
                best_all.append(best)
                second_all.append(second)
            temporal_2d[view] = ambiguity(
                np.concatenate(best_all), np.concatenate(second_all), span_gate=gate_px[view]
            )
            show(view, temporal_2d[view], "px")
        a2d = min(temporal_2d[v]["ambiguous_rate"] for v in VIEWS)
        best_2d_view = min(VIEWS, key=lambda v: temporal_2d[v]["ambiguous_rate"])
        print(f"  best 2D view: {best_2d_view} at {100 * a2d:.3f}%\n")

        # --- 3 & 4. the 3D pair -------------------------------------------
        print("3. 3D cross-view margin — 36 correspondences, summed point-to-ray residual")
        print("4. 3D temporal margin — adjacent resolvable moments, 3D mouth displacement")
        corr_best, corr_second = [], []
        temp_best, temp_second = [], []
        body_mm: list[np.ndarray] = []
        recoverable = {view: [] for view in VIEWS}
        for experiment in args.experiments:
            others = {
                view: eq3[(experiment, VIEWS[(i + 1) % 3])] & eq3[(experiment, VIEWS[(i + 2) % 3])]
                for i, view in enumerate(VIEWS)
            }
            for view in VIEWS:
                lost = ~eq3[(experiment, view)][plan]
                recoverable[view].append(
                    np.array([lost.sum(), (lost & others[view][plan]).sum()])
                )

            frames = plan[joint[experiment][plan]]
            if frames.size == 0:
                continue
            mouths = {
                view: nodes_by_frame(detections[(experiment, view)], frames, "mouth")
                for view in VIEWS
            }
            tails = {
                view: nodes_by_frame(detections[(experiment, view)], frames, "tail")
                for view in VIEWS
            }
            rays = {view: rig.camera(view).rays(mouths[view]) for view in VIEWS}
            tail_rays = {view: rig.camera(view).rays(tails[view]) for view in VIEWS}

            resolved = np.full((frames.size, 3, 3), np.nan)  # per frame, 3 slots, xyz
            for lo in range(0, frames.size, args.batch):
                hi = min(lo + args.batch, frames.size)
                n = hi - lo
                # (n, 36, 3 slots, 3 cameras, 3) — top fixed, the sides permuted.
                origins = np.empty((n, 36, 3, 3, 3))
                directions = np.empty((n, 36, 3, 3, 3))
                for h, (p1, p2) in enumerate(itertools.product(PERMUTATIONS, repeat=2)):
                    for cam, view, order in (
                        (0, VIEWS[0], (0, 1, 2)),
                        (1, VIEWS[1], p1),
                        (2, VIEWS[2], p2),
                    ):
                        o, d = rays[view]
                        origins[:, h, :, cam, :] = o[lo:hi][:, order, :]
                        directions[:, h, :, cam, :] = d[lo:hi][:, order, :]
                xyz, rms = triangulate(origins, directions)  # (n, 36, 3), (n, 36, 3)
                cost = rms.sum(axis=-1)  # (n, 36)
                best, second = best_two(cost)
                corr_best.append(best)
                corr_second.append(second)
                winner = np.argmin(np.where(np.isfinite(cost), cost, np.inf), axis=-1)
                resolved[lo:hi] = xyz[np.arange(n), winner]

                # Body length in mm, under the *same* correspondence — the gate
                # for the 3D temporal metric has to come from the animals, not
                # from a pixel threshold carried over from a camera.
                won = np.asarray(list(itertools.product(PERMUTATIONS, repeat=2)), dtype=int)[winner]
                tail_o = np.empty((n, 3, 3, 3))
                tail_d = np.empty((n, 3, 3, 3))
                rows = np.arange(n)[:, None]
                for cam, view in enumerate(VIEWS):
                    order = (
                        np.tile(np.arange(3), (n, 1)) if cam == 0 else won[:, cam - 1, :]
                    )
                    o, d = tail_rays[view]
                    tail_o[:, :, cam, :] = o[lo:hi][rows, order]
                    tail_d[:, :, cam, :] = d[lo:hi][rows, order]
                tail_xyz, _ = triangulate(tail_o, tail_d)
                body_mm.append(
                    np.linalg.norm(tail_xyz - resolved[lo:hi], axis=-1).ravel()
                )

            # Temporal margin over consecutive resolvable moments only.
            step = np.flatnonzero(np.diff(frames) == 1)
            usable_step = step[np.isfinite(resolved[step]).all(axis=(1, 2))]
            usable_step = usable_step[np.isfinite(resolved[usable_step + 1]).all(axis=(1, 2))]
            if usable_step.size:
                best, second = best_two(
                    assignment_costs(resolved[usable_step], resolved[usable_step + 1])
                )
                temp_best.append(best)
                temp_second.append(second)

        lengths = np.concatenate(body_mm)
        lengths = lengths[np.isfinite(lengths)]
        body_median = float(np.median(lengths))
        gate_mm = body_median / 2.0
        # No span gate on the cross-view metric: its unit is a point-to-ray
        # residual, not a displacement, and half a body length says nothing
        # about how far three rays may miss each other.
        cross_view = ambiguity(np.concatenate(corr_best), np.concatenate(corr_second))
        temporal_3d = ambiguity(
            np.concatenate(temp_best), np.concatenate(temp_second), span_gate=gate_mm
        )
        show("3D cross-view", cross_view, "mm")
        show("3D temporal", temporal_3d, "mm")
        print(
            f"  reconstructed mouth-tail length: median {body_median:.2f} mm "
            f"(p5 {np.percentile(lengths, 5):.2f}, p95 {np.percentile(lengths, 95):.2f}, "
            f"n = {lengths.size:,}) -> 3D gate {gate_mm:.2f} mm"
        )
        record.param("body_length_median_mm", body_median)
        record.param("gate_mm", gate_mm)

        # --- the pre-registered decision, printed unaltered ----------------
        adopt_3d = bool(
            temporal_3d["ambiguous_rate"] < a2d and cross_view["ambiguous_rate"] < a2d
        )
        print("\n" + "=" * 78)
        print("DECISION RULE, AS PRE-REGISTERED — 3D iff both its rates beat the best 2D view")
        print("=" * 78)
        print(f"  best 2D temporal ambiguity ({best_2d_view:<10}) : {100 * a2d:7.3f}%")
        print(f"  3D temporal ambiguity                    : "
              f"{100 * temporal_3d['ambiguous_rate']:7.3f}%  "
              f"{'PASS' if temporal_3d['ambiguous_rate'] < a2d else 'FAIL'}")
        print(f"  3D cross-view ambiguity                  : "
              f"{100 * cross_view['ambiguous_rate']:7.3f}%  "
              f"{'PASS' if cross_view['ambiguous_rate'] < a2d else 'FAIL'}")
        print(f"  -> the rule says TRACK IN {'3D' if adopt_3d else '2D'}")

        floored = a2d == 0.0 and temporal_3d["ambiguous_rate"] == 0.0
        if floored:
            print(
                "\n  THE RULE IS DEGENERATE HERE. Both temporal rates are exactly 0, so the\n"
                "  comparison decided on which side of a tie the '<' fell, not on the data.\n"
                "  Section 5 below was added after seeing that, and is labelled as post-hoc."
            )

        # --- 5. added after the rule floored -------------------------------
        print("\n" + "=" * 78)
        print("5. POST-HOC — chosen after the pre-registered rule floored")
        print("=" * 78)
        print(
            f"  credible-rival rate — runner-up costs no more than the p{CREDIBLE_PERCENTILE} "
            f"of winners"
        )
        for view in VIEWS:
            print(
                f"    2D {view:<12} {100 * temporal_2d[view]['credible_rate']:7.4f}%   "
                f"[s10 span gate: {100 * temporal_2d[view]['span_gated_rate']:6.2f}%]"
            )
        print(
            f"    3D temporal      {100 * temporal_3d['credible_rate']:7.4f}%   "
            f"[s10 span gate: {100 * temporal_3d['span_gated_rate']:6.2f}%]"
        )
        print(f"    3D cross-view    {100 * cross_view['credible_rate']:7.4f}%")
        best_2d_credible = min(temporal_2d[v]["credible_rate"] for v in VIEWS)
        print(
            f"    3D temporal vs the best 2D view: "
            f"{100 * temporal_3d['credible_rate']:.4f}% vs {100 * best_2d_credible:.4f}%"
        )

        print("\n  recoverability: a view that lost an animal while the other two held three")
        recovery = {}
        for view in VIEWS:
            lost, covered = np.sum(recoverable[view], axis=0)
            recovery[view] = {
                "frames_lost": int(lost),
                "covered_by_other_two": int(covered),
                "covered_rate": float(covered / lost) if lost else float("nan"),
            }
            print(
                f"    {view:<12} lost {int(lost):>6} frames, "
                f"{100 * recovery[view]['covered_rate']:5.1f}% of them with both other "
                f"views intact"
            )
        print(
            "    Those are the frames where a 2D track in that view must be re-acquired\n"
            "    across a gap and a 3D track need not be. They are the actual difference\n"
            "    between the two routes; neither temporal margin could see them."
        )
        print(f"\n     n_groups = {len(groups)}. One trio of the corpus's 14 groups.")

        result = {
            "n_groups": len(groups),
            "groups": groups,
            "experiments": list(args.experiments),
            "block_starts": starts,
            "block_len": args.block_len,
            "usable_fraction": usable,
            "temporal_2d": temporal_2d,
            "best_2d_view": best_2d_view,
            "best_2d_ambiguity": a2d,
            "cross_view_3d": cross_view,
            "temporal_3d": temporal_3d,
            "adopt_3d": adopt_3d,
            "rule_degenerate": floored,
            "gate_px": gate_px,
            "gate_mm": gate_mm,
            "body_length_median_mm": body_median,
            "recoverability": recovery,
        }
        record.param("result", result)
        out = out_dir / "c02_probe.json"
        out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        record.output("probe", out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

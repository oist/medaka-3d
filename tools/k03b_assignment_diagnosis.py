"""K03b — why the pre-registered assignment run failed. Diagnosis, not tuning.

`k03_assignment_heldout.py` ran the specification frozen in the lab README and
the rule returned REJECT: coverage 88.79% -> 78.47%, both-node recall
74.93% -> 60.77%. Something has to explain a 35-instance coverage loss, and the
two candidates are not equally interesting:

  1. **My error.** The edge rule uses 68.7 px, which k01 measured as the minimum
     distance from an *instance centroid* to the frame edge. It is applied here
     to *mouths*. A mouth can sit half a body span (~56 px) nearer the edge than
     its centroid, so the threshold is being applied to the wrong quantity.
  2. **The specification's error.** The cost is span-deviation alone. The
     labelled span runs 35--153 px, so a great many wrong pairings are as
     "plausible" as the right one, and the solver is free to swap tails between
     animals whenever that lands closer to the median.

These are separable. This script attributes the loss to each, and counts how
often the assignment moves a tail off the pairing the model itself decoded and
whether that move helps or hurts against ground truth.

Nothing here changes the verdict in k03. It explains it.

Run:
    .venv\\Scripts\\python.exe tools/k03b_assignment_diagnosis.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sleap_io as sio
from b03_heldout_comparison import VIEWS, build_heldout, points, run_inference
from k01_misdetection_taxonomy import label_stats, regime_of
from k03_assignment_heldout import SPAN_LABELS, evaluate, totals
from scipy.optimize import linear_sum_assignment

from medaka3d import load_catalog, load_paths

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "labs" / "20260803_mouth_anchored_assignment")
)
from assignment import MOUTH, TAIL, Params, assign_frame, span_model_from_labels

MATCH_TOL_PX = 20.0


def gt_node_edge_distances(frames) -> dict[str, np.ndarray]:
    """Distance to the nearest frame edge for GT mouths, tails and centroids.

    k01 reported the centroid figure (min 68.7 px). The assignment rule gates
    *mouths*, so the mouth figure is the one it should have used.
    """
    out = {"mouth": [], "tail": [], "centroid": []}
    for frame in frames:
        height, width = frame.video.shape[1:3]
        for instance in frame.instances:
            p = points(instance)
            items = {"mouth": p[MOUTH], "tail": p[TAIL], "centroid": np.nanmean(p, axis=0)}
            for name, point in items.items():
                if np.isnan(point).any():
                    continue
                out[name].append(
                    min(point[0], point[1], width - point[0], height - point[1])
                )
    return {k: np.array(v) for k, v in out.items()}


def pairing_audit(frames, baseline, produced, tol=MATCH_TOL_PX) -> dict:
    """Did assignment move a tail off the model's own pairing, and did it help?

    For each baseline instance with both nodes, find the assignment output whose
    mouth is the same mouth (< 1 px). Compare which tail each holds, and score
    both against the nearest ground-truth instance by mouth.
    """
    tally = {
        "compared": 0, "same_tail": 0, "moved": 0, "lost_tail": 0,
        "moved_correct_to_wrong": 0, "moved_wrong_to_correct": 0,
        "moved_wrong_to_wrong": 0, "moved_correct_to_correct": 0,
        "lost_from_correct": 0, "lost_from_wrong": 0,
    }
    for frame in frames:
        gt = [p for p in (points(i) for i in frame.instances) if not np.isnan(p).any()]
        base = baseline.get(frame.key, [])
        out = produced.get(frame.key, [])
        by_mouth = {}
        for instance in out:
            if not np.isnan(instance[MOUTH]).any():
                by_mouth[tuple(np.round(instance[MOUTH], 3))] = instance

        for instance in base:
            if np.isnan(instance).any():
                continue
            after = by_mouth.get(tuple(np.round(instance[MOUTH], 3)))
            if after is None:
                continue
            tally["compared"] += 1
            # The ground-truth instance this mouth belongs to.
            truth = min(
                gt, key=lambda g: float(np.hypot(*(g[MOUTH] - instance[MOUTH]))), default=None
            )
            if truth is None or float(np.hypot(*(truth[MOUTH] - instance[MOUTH]))) > tol:
                continue
            was_right = float(np.hypot(*(truth[TAIL] - instance[TAIL]))) <= tol
            if np.isnan(after[TAIL]).any():
                tally["lost_tail"] += 1
                tally["lost_from_correct" if was_right else "lost_from_wrong"] += 1
                continue
            if float(np.hypot(*(after[TAIL] - instance[TAIL]))) < 1.0:
                tally["same_tail"] += 1
                continue
            tally["moved"] += 1
            now_right = float(np.hypot(*(truth[TAIL] - after[TAIL]))) <= tol
            key = f"moved_{'correct' if was_right else 'wrong'}_to_" \
                  f"{'correct' if now_right else 'wrong'}"
            tally[key] += 1
    return tally


def coverage(frames, predictions, body) -> tuple[float, int]:
    """Animal coverage @20 px, k03's definition, pooled."""
    covered = total = 0
    for frame in frames:
        gt = [p for p in (points(i) for i in frame.instances) if not np.isnan(p).all()]
        pr = predictions.get(frame.key, [])
        total += len(gt)
        if not gt or not pr:
            continue
        from b03_heldout_comparison import mean_visible_error

        cost = np.array([[mean_visible_error(g, p) for p in pr] for g in gt])
        rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e6))
        covered += sum(
            1 for r, c in zip(rows, cols, strict=False)
            if np.isfinite(cost[r, c]) and cost[r, c] <= MATCH_TOL_PX
        )
    _ = body, regime_of
    return (100.0 * covered / total if total else 0.0), covered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    print(__doc__.split("Run:")[0])

    lo, hi, body = label_stats(catalog)
    span = span_model_from_labels(sio.load_slp(str(catalog.get(SPAN_LABELS))))
    frames, _ = build_heldout(catalog, "top")
    shape = tuple(frames[0].video.shape[1:3])
    baseline = run_inference(frames, VIEWS["top"]["topdown"], args.device, args.batch_size)

    print(f"\n{'=' * 78}\n1. IS THE EDGE THRESHOLD APPLIED TO THE WRONG QUANTITY?\n{'=' * 78}")
    dist = gt_node_edge_distances(frames)
    for name, values in dist.items():
        print(f"  GT {name:9s} n={len(values):4d}  min {values.min():7.2f}  "
              f"p1 {np.percentile(values, 1):7.2f}  median {np.median(values):7.2f}")
    below = (dist["mouth"] < 68.7).sum()
    print(f"\n  GT mouths closer to the edge than the 68.7 px centroid threshold: "
          f"{below}/{len(dist['mouth'])} ({100 * below / len(dist['mouth']):.1f}%)")
    print("  -> the rule gates mouths with a bound measured on centroids. Mine, not the")
    print("     specification's. The mouth-derived minimum is printed above.")

    print(f"\n{'=' * 78}\n2. ATTRIBUTING THE COVERAGE LOSS\n{'=' * 78}")
    base_cov, base_n = coverage(frames, baseline, body)
    print(f"  baseline                                  coverage {base_cov:6.2f}%  ({base_n})")

    variants = {
        "as pre-registered (edge 68.7, span cost)": Params(),
        "edge rule OFF, span cost unchanged": Params(edge_min_px=0.0),
        "edge 68.7, no reassignment (keep model pairing)": None,
    }
    produced_cache = {}
    for name, params in variants.items():
        if params is None:
            continue
        produced = {}
        for frame in frames:
            produced[frame.key] = assign_frame(
                baseline.get(frame.key, []), shape, span, params
            ).instances
        produced_cache[name] = produced
        cov, n = coverage(frames, produced, body)
        print(f"  {name:41s} coverage {cov:6.2f}%  ({n})   "
              f"delta {cov - base_cov:+.2f}")

    print(f"\n{'=' * 78}\n3. WHAT DOES REASSIGNMENT ACTUALLY DO TO THE TAILS?\n{'=' * 78}")
    for name in ("as pre-registered (edge 68.7, span cost)", "edge rule OFF, span cost unchanged"):
        audit = pairing_audit(frames, baseline, produced_cache[name])
        print(f"\n  [{name}]")
        print(f"    complete baseline instances whose mouth survived: {audit['compared']}")
        print(f"      tail unchanged            {audit['same_tail']:4d}")
        print(f"      tail MOVED to another     {audit['moved']:4d}   "
              f"correct->wrong {audit['moved_correct_to_wrong']}, "
              f"wrong->correct {audit['moved_wrong_to_correct']}, "
              f"wrong->wrong {audit['moved_wrong_to_wrong']}, "
              f"correct->correct {audit['moved_correct_to_correct']}")
        print(f"      tail REJECTED (mouth-only){audit['lost_tail']:4d}   "
              f"was correct {audit['lost_from_correct']}, was wrong {audit['lost_from_wrong']}")

    print(f"\n{'=' * 78}\n4. HOW DISCRIMINATIVE IS A SPAN-ONLY COST?\n{'=' * 78}")
    right, wrong = [], []
    for frame in frames:
        gt = [p for p in (points(i) for i in frame.instances) if not np.isnan(p).any()]
        for truth in gt:
            for other in gt:
                d = float(np.hypot(*(truth[MOUTH] - other[TAIL])))
                (right if other is truth else wrong).append(float(span.cost(np.array([d]))[0]))
    right, wrong = np.array(right), np.array(wrong)
    plausible = (wrong < -np.log(0.01)).mean()
    print(f"  cost of the TRUE mouth-tail pairing:  median {np.median(right):.3f}")
    print(f"  cost of a FOREIGN mouth-tail pairing: median {np.median(wrong):.3f}  "
          f"(n={len(wrong)})")
    print(f"  foreign pairings the ceiling accepts as plausible: {100 * plausible:.1f}%")
    print("  -> a span-only cost cannot rank the right tail above a wrong one; it ranks")
    print("     both against the median span, and most wrong tails sit inside the band.")

    print(f"\n{'=' * 78}\n5. TAXONOMY OF THE PRE-REGISTERED ARM, FOR THE RECORD\n{'=' * 78}")
    k_arena_pts = np.vstack([points(i) for f in frames for i in f.instances])
    finite = k_arena_pts[~np.isnan(k_arena_pts).any(axis=1)]
    arena = (float(finite[:, 0].min() - body), float(finite[:, 1].min() - body),
             float(finite[:, 0].max() + body), float(finite[:, 1].max() + body))
    agg = totals(evaluate(frames, produced_cache["as pre-registered (edge 68.7, span cost)"],
                          body, lo, hi, arena))
    print("  (arena recomputed from the 113 frames, so this differs slightly from k03)")
    print(f"  coverage {agg['coverage']:.2f}%  recall {agg['recall']:.2f}%  FP {agg['fp']}")


if __name__ == "__main__":
    main()

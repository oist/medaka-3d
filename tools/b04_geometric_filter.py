"""B0.4 — geometric plausibility filter, scored on the B0.3 held-out frames.

B0.3 localised the reported failure. On the held-out set, round-7 **top-down**
emits predicted instances that share a node with another instance — 5 pairs on
top, 3 on side — while full-resolution bottom-up emits **zero**. "One tail joined
to two mouths" is that shared-node event, and it is architecture-specific.
Top-down also over-predicts: 227 instances for 225 ground-truth ones while
recalling only 73%, so a large share of its output is wrong rather than missing.

This filter is a cheap post-process aimed at exactly that.

Two rules, both derived from the human labels rather than assumed:

1. **Shared node.** If two predicted instances have any node pair within
   ``SHARE_PX``, drop the lower-scoring one. Human labels never do this: B0.1
   found **0** of 4,146 training frames with two instances within 1 px on a node.
2. **Implausible span.** Drop an instance whose mouth-tail span falls outside the
   central range of the labelled distribution. Bounds are computed from the
   ground-truth labels of the *training* package, never from the evaluation
   frames or from predictions.

Note on units: bounds are in **pixels**, taken from the labelled span
distribution. They are deliberately *not* converted to mm — B0.2's mm/px is an
operational screening proxy, not calibration, and the labelled mouth-tail span
(~101 px) is shorter than the animal's 25-35 mm body because the ``tail`` node
sits at the tail base. Converting would import an unvalidated assumption into a
production filter.

**Decision rule, fixed before running:** the filter is worth keeping for an
architecture if it raises precision@20px at crossings **without** reducing
recall@20px at crossings by more than 1.0 point.

Run:
    .venv\\Scripts\\python.exe tools/b04_geometric_filter.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import sleap_io as sio
from b03_heldout_comparison import (
    VIEWS,
    accepted_mask,
    build_heldout,
    points,
    run_inference,
    score,
)

from medaka3d import load_catalog, load_paths

SHARE_PX = 1.0
SPAN_LO_PCT, SPAN_HI_PCT = 0.5, 99.5

#: Training packages the span bounds are derived from, per view.
SPAN_SOURCE = {"top": "labels.round8.top.train", "side": "labels.round8.side.train"}


def labelled_span_bounds(catalog, view: str) -> tuple[float, float, dict]:
    """Percentile bounds of the human-labelled mouth-tail span, in pixels."""
    labels = sio.load_slp(str(catalog.get(SPAN_SOURCE[view])))
    spans = []
    for frame in labels.labeled_frames:
        for instance in frame.instances:
            p = points(instance)
            if not np.isnan(p).any():
                spans.append(float(np.hypot(*(p[0] - p[1]))))
    array = np.array(spans)
    lo = float(np.percentile(array, SPAN_LO_PCT))
    hi = float(np.percentile(array, SPAN_HI_PCT))
    stats = {
        "n": len(array),
        "median": float(np.median(array)),
        "lo_pct": SPAN_LO_PCT,
        "hi_pct": SPAN_HI_PCT,
        "lo_px": lo,
        "hi_px": hi,
    }
    return lo, hi, stats


def span_of(instance: np.ndarray) -> float:
    if np.isnan(instance).any():
        return float("nan")
    return float(np.hypot(*(instance[0] - instance[1])))


def apply_filter(
    predictions: dict, lo: float, hi: float
) -> tuple[dict, dict[str, int]]:
    """Drop shared-node duplicates and implausible spans."""
    counts = {"input": 0, "dropped_shared": 0, "dropped_span": 0, "kept": 0}
    out: dict = {}
    for key, instances in predictions.items():
        counts["input"] += len(instances)
        survivors: list[np.ndarray] = []
        for candidate in instances:
            span = span_of(candidate)
            # Rule 2: implausible span. NaN span means a node is missing, which
            # this rule does not judge — rule 1 and the matcher handle those.
            if not np.isnan(span) and not (lo <= span <= hi):
                counts["dropped_span"] += 1
                continue
            survivors.append(candidate)

        # Rule 1: shared node. Compare pairwise; drop the later of any pair that
        # shares a node. Without instance scores here, "later" is the tie-break —
        # recorded as a limitation rather than dressed up as a ranking.
        kept: list[np.ndarray] = []
        for candidate in survivors:
            clash = False
            for existing in kept:
                gaps = np.hypot(*(candidate - existing).T)
                if np.isfinite(gaps).any() and float(np.nanmin(gaps)) < SHARE_PX:
                    clash = True
                    break
            if clash:
                counts["dropped_shared"] += 1
            else:
                kept.append(candidate)
        counts["kept"] += len(kept)
        out[key] = kept
    return out, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    print(__doc__.split("Run:")[0])

    results: dict = {}
    for view in ("top", "side"):
        lo, hi, stats = labelled_span_bounds(catalog, view)
        frames, meta = build_heldout(catalog, view)
        print(f"\n{'=' * 78}\n{view.upper()}  ({len(frames)} held-out frames, "
              f"n_groups={meta['n_groups']} {meta['groups']})\n{'=' * 78}")
        print(
            f"span bounds from {SPAN_SOURCE[view]} labels: "
            f"[{lo:.1f}, {hi:.1f}] px  (median {stats['median']:.1f}, n={stats['n']})"
        )

        preds = {
            arch: run_inference(frames, VIEWS[view][arch], args.device, args.batch_size)
            for arch in ("topdown", "bottomup")
        }
        accepted = accepted_mask(frames, preds["topdown"])
        view_out: dict = {"span_bounds": stats, "meta": meta}

        for arch in ("topdown", "bottomup"):
            filtered, counts = apply_filter(preds[arch], lo, hi)
            print(
                f"\n{arch}: {counts['input']} predicted -> {counts['kept']} kept "
                f"(dropped {counts['dropped_shared']} shared-node, "
                f"{counts['dropped_span']} implausible span)"
            )
            arch_out = {"filter_counts": counts}
            for arm_name, mask in (("ALL", None), ("CORRECTED", accepted)):
                before = score(frames, preds[arch], mask)["crossing"]
                after = score(frames, filtered, mask)["crossing"]
                arch_out[arm_name] = {"before": before, "after": after}
                d_rec = after["recall"] - before["recall"]
                d_pre = after["precision"] - before["precision"]
                verdict = (
                    "KEEP" if d_pre > 0 and d_rec >= -1.0 else "REJECT"
                )
                print(
                    f"  [{arm_name:9s}] crossing recall {before['recall']:6.2f}% ->"
                    f" {after['recall']:6.2f}% ({d_rec:+.2f})   "
                    f"precision {before['precision']:6.2f}% -> {after['precision']:6.2f}%"
                    f" ({d_pre:+.2f})   shared {before['shared_node_pairs']} ->"
                    f" {after['shared_node_pairs']}   {verdict}"
                )
            view_out[arch] = arch_out
        results[view] = view_out

    out_path = paths.work_dir / "b04_geometric_filter.json"
    out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

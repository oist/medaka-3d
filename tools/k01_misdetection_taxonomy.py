"""Group K — classify detector misdetections during approaches.

Two anchors make this possible with **no new annotation**:

1. **Cardinality is known and exact.** Every one of the 1,402 labelled top frames
   holds exactly 3 instances. So in any prediction frame, ``n > 3`` implies at
   least ``n - 3`` false positives, and ``n < 3`` implies missed animals. This is
   a free, unsupervised label for *counting* errors across all 2.1 M frames.
2. **Group K is the only labelled group.** All 1,382 round-8 top training frames
   are K (K_1_1 125, K_1_2 1,132, K_1_3 125), plus 20 val. So FP status can be
   determined exactly on those frames by matching predictions to ground truth.

**Taxonomy.** Applied to each false-positive instance in priority order, so the
categories are mutually exclusive and every FP lands in exactly one:

  A  shared_node    shares a node with another prediction (< SHARE_PX)
                    — the reported "one tail joined to two mouths"
  B  duplicate      centroid within DUP_FRAC body lengths of another prediction,
                    no shared node — the same animal detected twice
  C  chimera        mouth-tail span above the labelled 99.5th percentile
                    — mouth of one animal joined to the tail of another
  D  collapsed      span below the labelled 0.5th percentile
  E  partial        one node missing
  F  out_of_arena   centroid outside the arena extent — wall reflections and the
                    entry phantom of PRIOR_FINDINGS 4.6
  G  unclassified   none of the above; the residue that needs eyes or model work

Each category maps to a distinct remedy, which is the point of separating them:
A and C/D are already handled by the B0.4 filter, B needs centroid NMS, E needs a
completeness rule, F needs an arena mask, and **G is the only part that requires
model work or new annotation**.

Regimes are computed from ground truth where available, otherwise from
predictions, and are reported separately because the question is specifically
about approaches.

Run:
    .venv\\Scripts\\python.exe tools/k01_misdetection_taxonomy.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import numpy as np
import sleap_io as sio
from b03_heldout_comparison import (
    VIEWS,
    leaf_video,
    mean_visible_error,
    points,
    run_inference,
)
from scipy.optimize import linear_sum_assignment

from medaka3d import load_catalog, load_paths

GROUP = "K"
LABEL_KEYS = ("labels.round8.top.train", "labels.round8.top.val")
ROUND7_KEY = "labels.round7.top"

SHARE_PX = 5.0
DUP_FRAC = 0.5
MATCH_TOL_PX = 20.0
#: Regime edges in body lengths, from the closest pair of complete GT instances.
REGIMES = ((0.0, 0.25, "contact"), (0.25, 1.0, "crossing"), (1.0, 2.0, "approach"))

CATEGORIES = (
    "A_shared_node",
    "B_duplicate",
    "C_chimera",
    "D_collapsed",
    "E_partial",
    "F_out_of_arena",
    "G_unclassified",
)


def experiment_of(path: str) -> str | None:
    m = re.search(r"/(\d{8}_[A-Z]+_\d+_\d+_\w+)/", path)
    return m.group(1) if m else None


def span(instance: np.ndarray) -> float:
    if np.isnan(instance).any():
        return float("nan")
    return float(np.hypot(*(instance[0] - instance[1])))


def centroid(instance: np.ndarray) -> np.ndarray:
    return np.nanmean(instance, axis=0)


def min_node_gap(a: np.ndarray, b: np.ndarray) -> float:
    gaps = [
        float(np.hypot(*(x - y)))
        for x in a
        for y in b
        if not (np.isnan(x).any() or np.isnan(y).any())
    ]
    return min(gaps) if gaps else float("inf")


def classify(
    instance: np.ndarray,
    others: list[np.ndarray],
    lo: float,
    hi: float,
    body: float,
    arena: tuple[float, float, float, float],
) -> str:
    """Assign one false-positive instance to exactly one category."""
    for other in others:
        if other is instance:
            continue
        if min_node_gap(instance, other) < SHARE_PX:
            return "A_shared_node"

    if np.isnan(instance).any():
        # Partial before duplicate: a one-node instance has no meaningful span,
        # and its centroid is just that node, so duplicate/​span tests mislead.
        if not np.isnan(instance).all():
            c = centroid(instance)
            x0, y0, x1, y1 = arena
            if not (x0 <= c[0] <= x1 and y0 <= c[1] <= y1):
                return "F_out_of_arena"
        return "E_partial"

    c = centroid(instance)
    x0, y0, x1, y1 = arena
    if not (x0 <= c[0] <= x1 and y0 <= c[1] <= y1):
        return "F_out_of_arena"

    s = span(instance)
    if s > hi:
        return "C_chimera"
    if s < lo:
        return "D_collapsed"

    for other in others:
        if other is instance or np.isnan(other).all():
            continue
        if float(np.hypot(*(c - centroid(other)))) < DUP_FRAC * body:
            return "B_duplicate"

    return "G_unclassified"


def label_stats(catalog) -> tuple[float, float, float]:
    """(0.5th pct span, 99.5th pct span, median span) from labelled K frames."""
    spans: list[float] = []
    for key in LABEL_KEYS:
        labels = sio.load_slp(str(catalog.get(key)))
        for frame in labels.labeled_frames:
            for instance in frame.instances:
                p = points(instance)
                if not np.isnan(p).any():
                    spans.append(float(np.hypot(*(p[0] - p[1]))))
    arr = np.array(spans)
    return float(np.percentile(arr, 0.5)), float(np.percentile(arr, 99.5)), float(np.median(arr))


def build_k_frames(catalog):
    """All labelled Group-K top frames, with a flag for round-7 hold-out status."""
    round7 = {
        (leaf_video(f.video).lower(), int(f.frame_idx))
        for f in sio.load_slp(str(catalog.get(ROUND7_KEY))).labeled_frames
    }
    out, seen = [], set()
    for key in LABEL_KEYS:
        labels = sio.load_slp(str(catalog.get(key)))
        index: dict[int, object] = {}
        for video in labels.videos:
            src = getattr(video.backend, "source_inds", None)
            if src is None:
                continue
            for original in np.asarray(src):
                index.setdefault(int(original), video)
        for frame in labels.labeled_frames:
            video = index.get(int(frame.frame_idx), frame.video)
            path = leaf_video(video)
            exp = experiment_of(path)
            if not exp or f"_{GROUP}_" not in exp:
                continue
            canonical = (path.lower(), int(frame.frame_idx))
            if canonical in seen:
                continue
            seen.add(canonical)
            out.append(
                {
                    "key": canonical,
                    "video": video,
                    "frame_idx": int(frame.frame_idx),
                    "instances": list(frame.instances),
                    "experiment": exp,
                    "heldout": canonical not in round7,
                }
            )
    return out


def regime_of(gt: list[np.ndarray], body: float) -> str:
    full = [g for g in gt if not np.isnan(g).any()]
    if len(full) < 2:
        return "unknown"
    closest = min(
        float(np.hypot(*(centroid(full[i]) - centroid(full[j]))))
        for i in range(len(full))
        for j in range(i + 1, len(full))
    )
    for low, high, name in REGIMES:
        if low * body <= closest < high * body:
            return name
    return "separated"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    print(__doc__.split("Run:")[0])

    lo, hi, body = label_stats(catalog)
    print(f"labelled K span: median {body:.1f} px, 0.5-99.5 pct [{lo:.1f}, {hi:.1f}] px")

    frames = build_k_frames(catalog)
    if args.limit:
        frames = frames[: args.limit]
    by_exp = Counter(f["experiment"] for f in frames)
    n_held = sum(f["heldout"] for f in frames)
    print(f"labelled K frames: {len(frames)}  {dict(by_exp)}")
    print(f"  held out from round 7: {n_held}   in round-7 training: {len(frames) - n_held}")

    # Arena extent from the labelled animals themselves, padded by one body
    # length. Deliberately generous: this rule should fire only on detections
    # well outside where fish have ever been observed.
    all_pts = np.vstack([points(i) for f in frames for i in f["instances"]])
    finite = all_pts[~np.isnan(all_pts).any(axis=1)]
    arena = (
        float(finite[:, 0].min() - body),
        float(finite[:, 1].min() - body),
        float(finite[:, 0].max() + body),
        float(finite[:, 1].max() + body),
    )
    print(f"arena extent (labelled positions +/- 1 body): "
          f"x {arena[0]:.0f}..{arena[2]:.0f}, y {arena[1]:.0f}..{arena[3]:.0f}")

    simple = [
        type("F", (), {"key": f["key"], "video": f["video"], "frame_idx": f["frame_idx"]})()
        for f in frames
    ]
    preds = run_inference(simple, VIEWS["top"]["topdown"], args.device, args.batch_size)

    tally: dict[str, Counter] = {}
    counts = Counter()
    per_regime_frames = Counter()
    examples: dict[str, list] = {c: [] for c in CATEGORIES}
    # 1,288 of these frames are round-7 TRAINING data, so top-down is partly
    # scored on what it was fit to and the FP rate there is biased DOWN. Only the
    # 113 held-out frames give an honest rate; both are reported.
    split_fp = {True: Counter(), False: Counter()}
    split_frames = Counter()

    for frame in frames:
        gt = [p for p in (points(i) for i in frame["instances"]) if not np.isnan(p).all()]
        pr = preds.get(frame["key"], [])
        regime = regime_of(gt, body)
        per_regime_frames[regime] += 1
        counts[f"n_pred_{min(len(pr), 6)}"] += 1

        matched_pr: set[int] = set()
        if gt and pr:
            cost = np.array([[mean_visible_error(g, p) for p in pr] for g in gt])
            rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e6))
            for r, c in zip(rows, cols, strict=False):
                if np.isfinite(cost[r, c]) and cost[r, c] <= MATCH_TOL_PX:
                    matched_pr.add(c)

        split_frames[frame["heldout"]] += 1
        for i, instance in enumerate(pr):
            if i in matched_pr:
                continue
            category = classify(instance, pr, lo, hi, body, arena)
            tally.setdefault(regime, Counter())[category] += 1
            split_fp[frame["heldout"]][category] += 1
            if len(examples[category]) < 3:
                examples[category].append(
                    f"{frame['experiment']} f{frame['frame_idx']} "
                    f"span={span(instance):.0f}px"
                )

    print(f"\n{'=' * 78}\nFALSE POSITIVES BY REGIME AND CATEGORY\n{'=' * 78}")
    print("regime is the closest pair of complete GT instances, in body lengths\n")
    header = f"{'regime':11s} {'frames':>7s} {'FPs':>6s} " + " ".join(
        f"{c.split('_', 1)[0]:>4s}" for c in CATEGORIES
    )
    print(header)
    for regime in ("contact", "crossing", "approach", "separated", "unknown"):
        if regime not in tally and per_regime_frames[regime] == 0:
            continue
        row = tally.get(regime, Counter())
        total = sum(row.values())
        cells = " ".join(f"{row.get(c, 0):>4d}" for c in CATEGORIES)
        print(f"{regime:11s} {per_regime_frames[regime]:>7d} {total:>6d} {cells}")

    grand = Counter()
    for row in tally.values():
        grand.update(row)
    total_fp = sum(grand.values())
    print(f"\n{'category':16s} {'count':>6s} {'share':>7s}   remedy")
    remedies = {
        "A_shared_node": "B0.4 shared-node rule (already removes 100%)",
        "B_duplicate": "centroid NMS at ~0.5 body length",
        "C_chimera": "B0.4 span rule (upper bound)",
        "D_collapsed": "B0.4 span rule (lower bound)",
        "E_partial": "require both nodes, or impute from the other view",
        "F_out_of_arena": "arena mask from calibration",
        "G_unclassified": "NEEDS MODEL WORK OR ANNOTATION",
    }
    for c in CATEGORIES:
        n = grand.get(c, 0)
        share = 100.0 * n / total_fp if total_fp else 0.0
        print(f"{c:16s} {n:>6d} {share:>6.1f}%   {remedies[c]}")
    print(f"{'TOTAL':16s} {total_fp:>6d}")

    addressable = sum(grand.get(c, 0) for c in CATEGORIES if c != "G_unclassified")
    if total_fp:
        print(
            f"\nGeometrically addressable: {addressable}/{total_fp} "
            f"= {100 * addressable / total_fp:.1f}%. "
            f"Residue needing model work: {grand.get('G_unclassified', 0)}."
        )

    print(f"\n{'=' * 78}\nRATE, SPLIT BY ROUND-7 TRAINING MEMBERSHIP\n{'=' * 78}")
    for heldout, name in ((False, "in round-7 training"), (True, "HELD OUT (honest)")):
        n_fr = split_frames[heldout]
        n_fp = sum(split_fp[heldout].values())
        rate = n_fp / n_fr if n_fr else 0.0
        print(f"  {name:22s} {n_fr:5d} frames  {n_fp:4d} FPs  = {rate:.3f} FP/frame")
    if split_frames[True] and split_frames[False]:
        r_in = sum(split_fp[False].values()) / split_frames[False]
        r_out = sum(split_fp[True].values()) / split_frames[True]
        if r_in:
            print(f"  -> held-out FP rate is {r_out / r_in:.1f}x the in-sample rate.")
        print("  Use the held-out rate. The in-sample one scores the model on its own")
        print("  training frames and understates the failure.")
    held = split_fp[True]
    held_total = sum(held.values())
    print(f"\n  held-out category mix ({held_total} FPs) — this is the honest taxonomy:")
    for c in CATEGORIES:
        n = held.get(c, 0)
        if n:
            print(f"    {c:16s} {n:>4d}  {100 * n / held_total:>5.1f}%")
    held_addressable = sum(held.get(c, 0) for c in CATEGORIES if c != "G_unclassified")
    if held_total:
        print(
            f"    geometrically addressable: {held_addressable}/{held_total} "
            f"= {100 * held_addressable / held_total:.1f}%"
        )

    print("\nexamples per category:")
    for c in CATEGORIES:
        if examples[c]:
            print(f"  {c:16s} {'; '.join(examples[c])}")

    out = paths.work_dir / "k01_misdetection_taxonomy.json"
    out.write_text(
        json.dumps(
            {
                "span_px": {"lo": lo, "hi": hi, "median": body},
                "arena": arena,
                "frames": len(frames),
                "by_regime": {k: dict(v) for k, v in tally.items()},
                "totals": dict(grand),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""K03 — mouth-anchored assignment, scored on the 113 held-out Group-K top frames.

Runs the lab in ``labs/20260803_mouth_anchored_assignment`` against unfiltered
round-7 top-down output, under the criterion frozen in that lab's README before
anything was run. Nothing is defined here: the frame set, the matcher and the
both-node recall come from ``b03_heldout_comparison``, the regimes and the
false-positive taxonomy from ``k01_misdetection_taxonomy``. That is the point —
a number that is not comparable to Gate B0 cannot inform a decision about it.

Three arms, all pre-specified:

  baseline    round-7 top-down, unfiltered
  b04         + the Gate B0 geometric filter (shared-node suppression + span)
  assign      mouth-anchored one-to-one assignment

``b04`` is included because it is what Gate B0 actually adopted, so it, not the
raw model, is the thing assignment has to beat in production.

Run:
    .venv\\Scripts\\python.exe tools/k03_assignment_heldout.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sleap_io as sio
from b03_heldout_comparison import (
    VIEWS,
    Frame,
    build_heldout,
    experiment_of,
    leaf_video,
    mean_visible_error,
    points,
    run_inference,
)
from b04_geometric_filter import apply_filter
from k01_misdetection_taxonomy import CATEGORIES, build_k_frames, classify, label_stats, regime_of
from scipy.optimize import linear_sum_assignment

from medaka3d import load_catalog, load_paths

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "labs" / "20260803_mouth_anchored_assignment")
)
from assignment import (
    MOUTH,
    TAIL,
    Params,
    assign_frame,
    span_model_from_labels,
)

#: Fitted on round 7, which is disjoint from the evaluation frames by
#: construction. round8.top.train — what B0.4 used — contains all 113 of them.
SPAN_LABELS = "labels.round7.top"

MATCH_TOL_PX = 20.0
SHARE_PX = 5.0
SHARE_PX_STRICT = 1.0
REGIME_ORDER = ("contact", "crossing", "approach", "separated", "unknown")


def min_node_gap(a: np.ndarray, b: np.ndarray) -> float:
    gaps = [
        float(np.hypot(*(x - y)))
        for x in a
        for y in b
        if not (np.isnan(x).any() or np.isnan(y).any())
    ]
    return min(gaps) if gaps else float("inf")


def evaluate(frames, predictions, body, lo, hi, arena) -> dict:
    """The frozen criterion. Definitions are k01's and b03's, unchanged."""
    blank = {
        "frames": 0, "n_gt": 0, "n_pred": 0, "covered": 0, "both_node": 0,
        "fp": 0, "shared5": 0, "shared1": 0, "shared_tail": 0, "mouth_only": 0,
        "tail_only": 0, "taxonomy": dict.fromkeys(CATEGORIES, 0),
    }
    out = {r: {k: (dict(v) if isinstance(v, dict) else v) for k, v in blank.items()}
           for r in REGIME_ORDER}

    for frame in frames:
        gt = [p for p in (points(i) for i in frame.instances) if not np.isnan(p).all()]
        pr = predictions.get(frame.key, [])
        bucket = out[regime_of(gt, body)]
        bucket["frames"] += 1
        bucket["n_gt"] += len(gt)
        bucket["n_pred"] += len(pr)

        for instance in pr:
            has_mouth = not np.isnan(instance[MOUTH]).any()
            has_tail = not np.isnan(instance[TAIL]).any()
            bucket["mouth_only"] += int(has_mouth and not has_tail)
            bucket["tail_only"] += int(has_tail and not has_mouth)

        for a in range(len(pr)):
            for b in range(a + 1, len(pr)):
                gap = min_node_gap(pr[a], pr[b])
                bucket["shared5"] += int(gap < SHARE_PX)
                bucket["shared1"] += int(gap < SHARE_PX_STRICT)
                ta, tb = pr[a][TAIL], pr[b][TAIL]
                if not (np.isnan(ta).any() or np.isnan(tb).any()):
                    bucket["shared_tail"] += int(float(np.hypot(*(ta - tb))) < SHARE_PX)

        matched: set[int] = set()
        if gt and pr:
            cost = np.array([[mean_visible_error(g, p) for p in pr] for g in gt])
            rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e6))
            for r, c in zip(rows, cols, strict=False):
                if not np.isfinite(cost[r, c]) or cost[r, c] > MATCH_TOL_PX:
                    continue
                matched.add(c)
                bucket["covered"] += 1
                g, p = gt[r], pr[c]
                if not (np.isnan(g).any() or np.isnan(p).any()):
                    bucket["both_node"] += int(bool((np.hypot(*(g - p).T) <= MATCH_TOL_PX).all()))

        for i, instance in enumerate(pr):
            if i in matched:
                continue
            bucket["fp"] += 1
            bucket["taxonomy"][classify(instance, pr, lo, hi, body, arena)] += 1

    return out


def totals(result: dict) -> dict:
    keys = ("frames", "n_gt", "n_pred", "covered", "both_node", "fp",
            "shared5", "shared1", "shared_tail", "mouth_only", "tail_only")
    agg = {k: sum(r[k] for r in result.values()) for k in keys}
    agg["taxonomy"] = {
        c: sum(r["taxonomy"][c] for r in result.values()) for c in CATEGORIES
    }
    agg["coverage"] = 100.0 * agg["covered"] / agg["n_gt"] if agg["n_gt"] else 0.0
    agg["recall"] = 100.0 * agg["both_node"] / agg["n_gt"] if agg["n_gt"] else 0.0
    agg["precision"] = 100.0 * agg["both_node"] / agg["n_pred"] if agg["n_pred"] else 0.0
    return agg


def print_block(name: str, result: dict) -> None:
    agg = totals(result)
    print(f"\n--- {name}")
    print(f"{'regime':11s} {'fr':>4s} {'GT':>4s} {'pred':>5s} "
          f"{'cover%':>7s} {'both%':>7s} {'FP':>4s} {'shr5':>5s} {'m-only':>7s}")
    for regime in REGIME_ORDER:
        r = result[regime]
        if not r["frames"]:
            continue
        cover = 100.0 * r["covered"] / r["n_gt"] if r["n_gt"] else 0.0
        both = 100.0 * r["both_node"] / r["n_gt"] if r["n_gt"] else 0.0
        print(f"{regime:11s} {r['frames']:>4d} {r['n_gt']:>4d} {r['n_pred']:>5d} "
              f"{cover:>7.2f} {both:>7.2f} {r['fp']:>4d} {r['shared5']:>5d} "
              f"{r['mouth_only']:>7d}")
    print(f"{'ALL':11s} {agg['frames']:>4d} {agg['n_gt']:>4d} {agg['n_pred']:>5d} "
          f"{agg['coverage']:>7.2f} {agg['recall']:>7.2f} {agg['fp']:>4d} "
          f"{agg['shared5']:>5d} {agg['mouth_only']:>7d}")
    print(f"  precision {agg['precision']:.2f}%   shared-node pairs: "
          f"{agg['shared5']} @5px / {agg['shared1']} @1px / "
          f"{agg['shared_tail']} shared TAIL   tail-only {agg['tail_only']}")
    mix = ", ".join(f"{c.split('_', 1)[0]}={n}" for c, n in agg["taxonomy"].items() if n)
    print(f"  taxonomy: {mix or 'none'}")


def neighbour_nodes(frames, paths, baseline, device, batch) -> tuple[dict, dict]:
    """Decoded mouths and tails at f-1 and f+1, pooled, per evaluation frame.

    The 113 held-out frames are scattered singletons and the round-8 package
    embeds *only* the labelled images, so ``f+-1`` does not exist inside it and
    must come from the raw recording. ``frame_idx`` is an absolute index into
    that recording (verified: 17,853--338,872 against 348,600 frames), so the
    two are directly addressable.

    Because this crosses from the embedded package to the mp4, the same 113
    frames are also re-predicted from the raw video and compared. If the two
    disagree, the temporal arm is measuring the container, not the model, and
    the caller is told so rather than left to assume.
    """
    experiments = {experiment_of(leaf_video(f.video)) for f in frames}
    if len(experiments) != 1:
        raise SystemExit(f"temporal arm expects one recording, got {experiments}")
    video = sio.load_video(str(paths.video(experiments.pop(), "top_cam")))
    n_video = int(video.shape[0])

    wanted = sorted(
        {
            i
            for f in frames
            for i in (f.frame_idx - 1, f.frame_idx, f.frame_idx + 1)
            if 0 <= i < n_video
        }
    )
    print(f"  raw-video inference on {len(wanted)} frames (f-1, f, f+1)")
    extra = [Frame((f"raw:{i}", i), video, i, []) for i in wanted]
    preds = run_inference(extra, VIEWS["top"]["topdown"], device, batch)
    by_index = {int(key[1]): value for key, value in preds.items()}

    agree = {"frames": 0, "same_count": 0, "matched_within_1px": 0, "package": 0, "raw": 0}
    out: dict[tuple[str, int], tuple] = {}
    for frame in frames:
        pool = [i for o in (-1, 1) for i in by_index.get(frame.frame_idx + o, [])]
        stack = np.array(pool) if pool else np.empty((0, 2, 2))
        mouths = stack[:, MOUTH] if len(stack) else np.empty((0, 2))
        tails = stack[:, TAIL] if len(stack) else np.empty((0, 2))
        out[frame.key] = (
            mouths[~np.isnan(mouths).any(axis=1)],
            tails[~np.isnan(tails).any(axis=1)],
        )
        pkg, raw = baseline.get(frame.key, []), by_index.get(frame.frame_idx, [])
        agree["frames"] += 1
        agree["package"] += len(pkg)
        agree["raw"] += len(raw)
        agree["same_count"] += int(len(pkg) == len(raw))
        for a in pkg:
            if any(min_node_gap(a, b) < 1.0 for b in raw):
                agree["matched_within_1px"] += 1
    print(
        f"  container check: package vs raw-video predictions on the same 113 frames — "
        f"{agree['same_count']}/{agree['frames']} frames agree on count "
        f"({agree['package']} vs {agree['raw']} instances), "
        f"{agree['matched_within_1px']}/{agree['package']} instances match within 1 px"
    )
    return out, agree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-temporal", action="store_true")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    print(__doc__.split("Run:")[0])

    lo, hi, body = label_stats(catalog)
    span = span_model_from_labels(sio.load_slp(str(catalog.get(SPAN_LABELS))))
    q01 = span.bounds(0.01)
    q10 = span.bounds(0.10)
    print(f"regime body length (k01, round-8 labels): {body:.1f} px")
    print(f"span model from {SPAN_LABELS}: n={span.spans.size}, median {span.median:.1f} px")
    print(f"  reject ceiling q<0.01 -> [{q01[0]:.1f}, {q01[1]:.1f}] px "
          f"(B0.4 adopted [{lo:.1f}, {hi:.1f}] px from round-8 labels)")
    print(f"  sensitivity   q<0.10 -> [{q10[0]:.1f}, {q10[1]:.1f}] px")

    frames, meta = build_heldout(catalog, "top")
    print(f"\nheld-out frames: {len(frames)}  experiments {meta['experiments']}  "
          f"n_groups {meta['n_groups']} {meta['groups']}")

    k_frames = build_k_frames(catalog)
    all_pts = np.vstack([points(i) for f in k_frames for i in f["instances"]])
    finite = all_pts[~np.isnan(all_pts).any(axis=1)]
    arena = (
        float(finite[:, 0].min() - body), float(finite[:, 1].min() - body),
        float(finite[:, 0].max() + body), float(finite[:, 1].max() + body),
    )

    shape = tuple(frames[0].video.shape[1:3])
    print(f"frame shape (H, W): {shape}")

    baseline = run_inference(frames, VIEWS["top"]["topdown"], args.device, args.batch_size)
    filtered, counts = apply_filter(baseline, lo, hi)
    print(f"\nB0.4 filter: {counts['input']} -> {counts['kept']} instances "
          f"(dropped {counts['dropped_shared']} shared-node, "
          f"{counts['dropped_span']} span)")

    neighbours: dict = {}
    container_check: dict = {}
    if not args.skip_temporal:
        neighbours, container_check = neighbour_nodes(
            frames, paths, baseline, args.device, args.batch_size
        )

    arms: dict[str, dict] = {
        "baseline (round-7 top-down, unfiltered)": baseline,
        "b04 (Gate B0 geometric filter)": filtered,
    }
    diagnostics: dict[str, dict] = {}
    variants = [("assign q<0.01 (PRIMARY)", Params(reject_q=0.01))]
    variants.append(("assign q<0.10 (sensitivity)", Params(reject_q=0.10)))
    if neighbours:
        variants.append(("assign q<0.01 + temporal gate", Params(reject_q=0.01, temporal=True)))

    for name, params in variants:
        produced: dict = {}
        tally = {
            "n_anchors_raw": 0, "n_dropped_edge": 0, "n_dropped_dedup_mouth": 0,
            "n_dropped_temporal_mouth": 0, "n_tail_candidates": 0,
            "n_dropped_temporal_tail": 0, "n_assigned": 0, "n_mouth_only": 0,
            "n_tails_unclaimed": 0,
        }
        for frame in frames:
            nm, nt = neighbours.get(frame.key, (None, None))
            result = assign_frame(
                baseline.get(frame.key, []), shape, span, params,
                neighbour_mouths=nm, neighbour_tails=nt,
            )
            produced[frame.key] = result.instances
            for key in tally:
                tally[key] += getattr(result, key)
        arms[name] = produced
        diagnostics[name] = tally

    results: dict[str, dict] = {}
    for name, preds in arms.items():
        result = evaluate(frames, preds, body, lo, hi, arena)
        results[name] = result
        print_block(name, result)
        if name in diagnostics:
            d = diagnostics[name]
            print(f"  anchors {d['n_anchors_raw']} raw -> "
                  f"-{d['n_dropped_edge']} edge -{d['n_dropped_dedup_mouth']} dedup"
                  f" -{d['n_dropped_temporal_mouth']} temporal;  "
                  f"tail candidates {d['n_tail_candidates']} "
                  f"(-{d['n_dropped_temporal_tail']} temporal), "
                  f"{d['n_assigned']} claimed, {d['n_tails_unclaimed']} unclaimed")

    decide(results)

    out = paths.work_dir / "k03_assignment_heldout.json"
    out.write_text(
        json.dumps(
            {
                "meta": meta,
                "span_model": {"source": SPAN_LABELS, "n": int(span.spans.size),
                               "median": span.median, "q01": q01, "q10": q10},
                "body_px": body,
                "arms": {k: totals(v) for k, v in results.items()},
                "by_regime": {k: {r: {kk: vv for kk, vv in d.items()}
                                  for r, d in v.items() if d["frames"]}
                              for k, v in results.items()},
                "diagnostics": diagnostics,
                "container_check": container_check,
            },
            indent=2, default=float,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


def decide(results: dict) -> None:
    """The rule from the lab README, applied verbatim."""
    print(f"\n{'=' * 78}\nPRE-REGISTERED DECISION RULE (labs/20260803_mouth_anchored_"
          f"assignment)\n{'=' * 78}")
    base_name = next(k for k in results if k.startswith("baseline"))
    base = totals(results[base_name])
    primary = next((k for k in results if "PRIMARY" in k), None)
    if primary is None:
        print("primary arm absent; cannot evaluate")
        return
    arm = totals(results[primary])

    checks = [
        ("1. shared-node pairs @5px == 0",
         arm["shared5"] == 0, f"{base['shared5']} -> {arm['shared5']}"),
        ("2. animal coverage >= baseline",
         arm["coverage"] >= base["coverage"],
         f"{base['coverage']:.2f}% -> {arm['coverage']:.2f}% "
         f"({arm['coverage'] - base['coverage']:+.2f})"),
        ("3. both-node recall >= baseline - 2.0",
         arm["recall"] >= base["recall"] - 2.0,
         f"{base['recall']:.2f}% -> {arm['recall']:.2f}% "
         f"({arm['recall'] - base['recall']:+.2f})"),
        ("4. false positives <= baseline",
         arm["fp"] <= base["fp"], f"{base['fp']} -> {arm['fp']}"),
    ]
    for label, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label:38s} {detail}")

    if all(p for _, p, _ in checks):
        verdict = "PROMOTE into src/medaka3d/"
    elif not checks[0][1] or arm["coverage"] < base["coverage"] - 2.0:
        verdict = "REJECT"
    else:
        verdict = "PARK — name the blocker"
    print(f"\n  -> {verdict}")


if __name__ == "__main__":
    main()

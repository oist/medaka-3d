"""K03c — origin-preferred assignment, on the same 113 held-out Group-K top frames.

The corrected specification from the lab README's 2026-08-03 amendment, which was
written after run 1 returned REJECT and before this ran. That makes it weaker
evidence than the original pre-registration, and it is labelled so here as well
as there. Run 1's verdict stands; this does not revise it.

What changed, and only this:

  * cost = ``origin_penalty + d / hi`` instead of span-deviation alone, so the
    pairing the centered-instance head decoded wins whenever it is feasible and
    only a **dispossessed** mouth searches for another tail;
  * the edge rule uses the ground-truth **mouth** minimum (28.8 px) instead of
    the **centroid** minimum (68.7 px) it was mistakenly given.

Every metric, frame set, matcher and taxonomy is imported from k03/k01/b03
unchanged, so the two runs and Gate B0 are directly comparable.

Run:
    .venv\\Scripts\\python.exe tools/k03c_assignment_origin.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sleap_io as sio
from b03_heldout_comparison import VIEWS, build_heldout, points, run_inference
from b04_geometric_filter import apply_filter
from k01_misdetection_taxonomy import build_k_frames, label_stats
from k03_assignment_heldout import SPAN_LABELS, evaluate, print_block, totals

from medaka3d import load_catalog, load_paths

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "labs" / "20260803_mouth_anchored_assignment")
)
from assignment import (
    OriginParams,
    Params,
    assign_frame,
    assign_frame_origin,
    span_model_from_labels,
)

DIAG_FIELDS = (
    "n_anchors_raw", "n_dropped_edge", "n_dropped_dedup_mouth", "n_tail_candidates",
    "n_assigned", "n_mouth_only", "n_tails_unclaimed",
)


def run(frames, baseline, shape, span, params, origin: bool) -> tuple[dict, dict]:
    produced, tally = {}, dict.fromkeys(DIAG_FIELDS, 0)
    for frame in frames:
        result = (
            assign_frame_origin(baseline.get(frame.key, []), shape, span, params)
            if origin
            else assign_frame(baseline.get(frame.key, []), shape, span, params)
        )
        produced[frame.key] = result.instances
        for key in tally:
            tally[key] += getattr(result, key)
    return produced, tally


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
    frames, meta = build_heldout(catalog, "top")
    shape = tuple(frames[0].video.shape[1:3])
    print(f"held-out frames: {len(frames)}  n_groups {meta['n_groups']} {meta['groups']}")
    print(f"feasible span band: [{span.bounds(0.01)[0]:.1f}, {span.bounds(0.01)[1]:.1f}] px")

    k_frames = build_k_frames(catalog)
    all_pts = np.vstack([points(i) for f in k_frames for i in f["instances"]])
    finite = all_pts[~np.isnan(all_pts).any(axis=1)]
    arena = (float(finite[:, 0].min() - body), float(finite[:, 1].min() - body),
             float(finite[:, 0].max() + body), float(finite[:, 1].max() + body))

    baseline = run_inference(frames, VIEWS["top"]["topdown"], args.device, args.batch_size)
    filtered, _ = apply_filter(baseline, lo, hi)

    arms: dict[str, dict] = {
        "baseline (round-7 top-down, unfiltered)": baseline,
        "b04 (Gate B0 geometric filter)": filtered,
    }
    diagnostics: dict[str, dict] = {}

    # Carried forward so the amendment's two changes stay separable: the same
    # rejected cost with only the edge threshold corrected.
    variants = [
        ("run 1 cost, edge corrected to 28.8", Params(edge_min_px=28.8), False),
        ("origin cost, abstain 2.0 (PRIMARY)", OriginParams(), True),
        ("origin cost, abstain 1.5 (sensitivity)", OriginParams(abstain_cost=1.5), True),
    ]
    for name, params, origin in variants:
        arms[name], diagnostics[name] = run(frames, baseline, shape, span, params, origin)

    results = {}
    for name, preds in arms.items():
        results[name] = evaluate(frames, preds, body, lo, hi, arena)
        print_block(name, results[name])
        if name in diagnostics:
            d = diagnostics[name]
            print(f"  anchors {d['n_anchors_raw']} raw -> -{d['n_dropped_edge']} edge "
                  f"-{d['n_dropped_dedup_mouth']} dedup;  tails {d['n_tail_candidates']}, "
                  f"{d['n_assigned']} claimed, {d['n_tails_unclaimed']} unclaimed, "
                  f"{d['n_mouth_only']} mouth-only")

    decide(results)

    # Does the method reach its own stated limit? k01 predicted a ceiling of
    # "9 of 19 recovered, 10 abstained" from the fact that in 10 of those frames
    # no true tail was ever decoded. A method that beats its ceiling has a bug;
    # one far below it has a cost function still to fix. Same audit function that
    # convicted run 1, so the two are on one scale.
    from k03b_assignment_diagnosis import pairing_audit

    print(f"\n{'=' * 78}\nCEILING CHECK — what assignment did to each pairing\n{'=' * 78}")
    ceiling: dict[str, dict] = {}
    for name in [k for k in arms if k in diagnostics]:
        audit = pairing_audit(frames, baseline, arms[name])
        ceiling[name] = audit
        moved_good = audit["moved_wrong_to_correct"] + audit["moved_correct_to_correct"]
        moved_bad = audit["moved_correct_to_wrong"] + audit["moved_wrong_to_wrong"]
        print(f"\n  [{name}]  {audit['compared']} complete baseline instances kept their mouth")
        print(f"    unchanged {audit['same_tail']:>4d}   "
              f"moved {audit['moved']:>3d} ({moved_good} to a correct tail, {moved_bad} to a "
              f"wrong one)   abstained {audit['lost_tail']:>3d} "
              f"({audit['lost_from_correct']} of them had been correct)")

    out = paths.work_dir / "k03c_assignment_origin.json"
    out.write_text(
        json.dumps(
            {"meta": meta, "arms": {k: totals(v) for k, v in results.items()},
             "by_regime": {k: {r: d for r, d in v.items() if d["frames"]}
                           for k, v in results.items()},
             "diagnostics": diagnostics, "ceiling": ceiling},
            indent=2, default=float,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


def decide(results: dict) -> None:
    """The amended rule, applied verbatim. Criterion 1 is on shared TAILS."""
    print(f"\n{'=' * 78}\nAMENDED DECISION RULE (lab README, 2026-08-03 amendment)\n{'=' * 78}")
    base = totals(results[next(k for k in results if k.startswith("baseline"))])
    arm = totals(results[next(k for k in results if "PRIMARY" in k)])
    checks = [
        ("1. shared-TAIL pairs @5px == 0", arm["shared_tail"] == 0,
         f"{base['shared_tail']} -> {arm['shared_tail']}   "
         f"(any-node @5px: {base['shared5']} -> {arm['shared5']}, not gated)"),
        ("2. animal coverage >= baseline", arm["coverage"] >= base["coverage"],
         f"{base['coverage']:.2f}% -> {arm['coverage']:.2f}% "
         f"({arm['coverage'] - base['coverage']:+.2f})"),
        ("3. both-node recall >= baseline - 2.0", arm["recall"] >= base["recall"] - 2.0,
         f"{base['recall']:.2f}% -> {arm['recall']:.2f}% "
         f"({arm['recall'] - base['recall']:+.2f})"),
        ("4. false positives <= baseline", arm["fp"] <= base["fp"],
         f"{base['fp']} -> {arm['fp']}"),
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
    print(f"\n  mouth-only instances: {base['mouth_only']} -> {arm['mouth_only']}. This is the "
          f"quantity Task 4's\n  retrain has to reduce; it is not scored good or bad here.")


if __name__ == "__main__":
    main()

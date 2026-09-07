"""K08 — the locked HITL-v2 benchmark: a frame-disjoint test set that already exists.

Gate B0 concluded "no unbiased evaluation set exists" and made building one the
strongest argument for Phase A. That conclusion was drawn without reading the
archive. A human-annotated, frame-locked benchmark was built by the retired repo
on 2026-07-26/27 and preserved during the migration:

    Z:\\StephensU\\medaka\\benchmarks\\sleap_hitl_v2\\
        phase1-hitl-v2-preserve-20260802T174200JST-60e13a7746d1\\final\\
        shared_benchmark_40_hitl_v2_locked.pkg.slp

40 synchronised source frames x 3 views = 120 view-frames, **360 human instances,
exactly 3 per view-frame**, 345 complete and 15 partial, spanning all three K days
(K_1_1 14, K_1_2 13, K_1_3 13). Verified here: its 40 **top** frames are disjoint
from round-7 training, from round-8 train, and from round-8 val — all three, 0
overlap.

**It is not in `MANIFEST.json`.** A human-made artefact of exactly the kind design
rule 3 exists to checksum, and rule 7 exists to keep on `/bucket`, is sitting
outside the catalogue where nothing verifies it. That is reported as a defect.

## The question this tool exists to answer honestly

Frame-disjoint is not the same as annotation-independent. Gate B0's governing
finding was that round-8 labels were made by *correcting round-7 predictions*, so
~48% of them are byte-identical to round-7 output. The archive's own evaluation
of this benchmark reports round-8 median errors of 0.0012 px (mouth) and 0.0004 px
(tail) — the same verbatim signature, pointing the other way. The HITL runbook
offered an "optional Round-8 prediction review" path alongside the manual one.

So the contamination is measured here for **every** model before any score is
read, by the same <= 0.01 px test `b03_heldout_comparison.accepted_mask` uses. A
benchmark that is verbatim one model's output cannot rank that model.

Metrics are k03/k01/b03's, unchanged.

Run:
    .venv\\Scripts\\python.exe tools/k08_locked_benchmark.py
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
    experiment_of,
    leaf_video,
    load_keys,
    points,
    run_inference,
)
from k01_misdetection_taxonomy import build_k_frames, label_stats
from k03_assignment_heldout import SPAN_LABELS, evaluate, print_block, totals
from k04_retrain_centered_instance import RUN_NAME
from k07_production_baseline import ROUND8_TOP
from k11_retrain_ci_highres import LEGACY_RUN_NAMES
from k11_retrain_ci_highres import RUN_NAME as K11_RUN

from medaka3d import load_catalog, load_paths
from medaka3d.cohort import CohortError, ExperimentId

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "labs" / "20260803_mouth_anchored_assignment")
)
from assignment import OriginParams, assign_frame_origin, span_model_from_labels

LOCKED = Path(
    r"Z:\StephensU\medaka\benchmarks\sleap_hitl_v2"
    r"\phase1-hitl-v2-preserve-20260802T174200JST-60e13a7746d1\final"
    r"\shared_benchmark_40_hitl_v2_locked.pkg.slp"
)
VERBATIM_PX = 0.01


def load_top_frames() -> list[Frame]:
    labels = sio.load_slp(str(LOCKED))
    frames = []
    for frame in labels.labeled_frames:
        path = leaf_video(frame.video)
        if "top_cam" not in path:
            continue
        frames.append(
            Frame((path.lower(), int(frame.frame_idx)), frame.video, int(frame.frame_idx),
                  list(frame.instances))
        )
    return frames


def contamination(frames, predictions) -> tuple[int, int]:
    """GT instances byte-identical (<= 0.01 px) to one of this model's predictions."""
    hits = total = 0
    for frame in frames:
        gt = [p for p in (points(i) for i in frame.instances) if not np.isnan(p).all()]
        pr = predictions.get(frame.key, [])
        for g in gt:
            total += 1
            for p in pr:
                both = ~(np.isnan(g).any(axis=1) | np.isnan(p).any(axis=1))
                if both.any() and float(np.hypot(*(g[both] - p[both]).T).max()) <= VERBATIM_PX:
                    hits += 1
                    break
    return hits, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not LOCKED.is_file():
        raise SystemExit(f"locked benchmark not found: {LOCKED}")

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    print(__doc__.split("Run:")[0])

    frames = load_top_frames()
    experiments = {experiment_of(leaf_video(f.video)) for f in frames}
    groups = set()
    for name in experiments:
        try:
            groups.add(ExperimentId.parse(name).group)
        except CohortError:
            continue
    keys = {f.key for f in frames}
    overlaps = {
        "round7 top training": len(keys & load_keys(catalog, "labels.round7.top")),
        "round8 top train": len(keys & load_keys(catalog, "labels.round8.top.train")),
        "round8 top val": len(keys & load_keys(catalog, "labels.round8.top.val")),
    }
    n_gt = sum(len(f.instances) for f in frames)
    print(f"top frames: {len(frames)}   GT instances: {n_gt}")
    print(f"experiments: {sorted(experiments)}   n_groups: {len(groups)} {sorted(groups)}")
    print(f"frame overlap with training sets: {overlaps}  <- all zero = frame-disjoint")

    lo, hi, body = label_stats(catalog)
    span = span_model_from_labels(sio.load_slp(str(catalog.get(SPAN_LABELS))))
    shape = tuple(frames[0].video.shape[1:3])

    k_frames = build_k_frames(catalog)
    all_pts = np.vstack([points(i) for f in k_frames for i in f["instances"]])
    finite = all_pts[~np.isnan(all_pts).any(axis=1)]
    arena = (float(finite[:, 0].min() - body), float(finite[:, 1].min() - body),
             float(finite[:, 0].max() + body), float(finite[:, 1].max() + body))

    models = {
        "round 7 top-down": VIEWS["top"]["topdown"],
        "round 8 top-down (PRODUCTION)": ROUND8_TOP,
    }
    k04 = paths.work_dir / "k04_retrain" / RUN_NAME
    if not (k04 / "best.ckpt").is_file():
        k04 = Path("C:/sleap_cache/k04_runs") / RUN_NAME
    if (k04 / "best.ckpt").is_file():
        models["k04 retrain (ms32/f32/intensity)"] = [VIEWS["top"]["topdown"][0], k04]
    # k11 pairs with ROUND 8's centroid, not round 7's — the centroid head is the
    # variable held fixed in that lab, so it must be round 8's in every arm.
    # Two layouts: the original top run predates the per-view subdirectory that
    # `k11 --view` now creates. Check both so a re-run does not go unnoticed.
    names = [K11_RUN, LEGACY_RUN_NAMES["top"]]
    candidates = [
        root / sub / name
        for name in names
        for root in (paths.work_dir / "k11_retrain", Path("C:/sleap_cache/k11_runs"))
        for sub in (Path(), Path("top"))
    ]
    k11 = next((c for c in candidates if (c / "best.ckpt").is_file()), candidates[0])
    if (k11 / "best.ckpt").is_file():
        models["round10 hires top (+r8 centroid)"] = [ROUND8_TOP[0], k11]

    predictions = {
        name: run_inference(frames, model_paths, args.device, args.batch_size)
        for name, model_paths in models.items()
    }

    print(f"\n{'=' * 78}\nCONTAMINATION — read this before any score below\n{'=' * 78}")
    contam = {}
    for name, preds in predictions.items():
        hits, total = contamination(frames, preds)
        contam[name] = {"verbatim": hits, "total": total, "pct": 100.0 * hits / max(total, 1)}
        print(f"  {name:34s} {hits:>4d}/{total} = {contam[name]['pct']:5.1f}% "
              f"of GT instances are byte-identical to its own output")
    print("  A model cannot be ranked by ground truth that is its own prediction.")

    # Gate B0's bracket, applied to whichever model this benchmark was seeded
    # from. The CORRECTED frames are those holding no instance that is verbatim
    # that model's output, so the arm keeps only evidence the annotator produced
    # independently of it. Frame-level rather than instance-level because the
    # shared harness scores whole frames; the cost is a smaller n, which is
    # stated rather than hidden.
    seeded = max(contam, key=lambda k: contam[k]["pct"])
    corrected: list[Frame] = []
    if contam[seeded]["pct"] > 0:
        preds = predictions[seeded]
        for frame in frames:
            gt = [p for p in (points(i) for i in frame.instances) if not np.isnan(p).all()]
            pr = preds.get(frame.key, [])
            verbatim = False
            for g in gt:
                for p in pr:
                    both = ~(np.isnan(g).any(axis=1) | np.isnan(p).any(axis=1))
                    if both.any() and float(
                        np.hypot(*(g[both] - p[both]).T).max()
                    ) <= VERBATIM_PX:
                        verbatim = True
                        break
                if verbatim:
                    break
            if not verbatim:
                corrected.append(frame)
        print(f"\n  CORRECTED arm: {len(corrected)} of {len(frames)} frames contain no "
              f"instance verbatim to {seeded}.")
        print("  Scored on both arms below. The truth is bracketed by them, and only a "
              "conclusion\n  that holds at both ends is trustworthy.")

    results = {}
    for name, preds in predictions.items():
        results[f"{name} — raw"] = evaluate(frames, preds, body, lo, hi, arena)
        produced = {}
        for frame in frames:
            produced[frame.key] = assign_frame_origin(
                preds.get(frame.key, []), shape, span, OriginParams()
            ).instances
        results[f"{name} — K03c assignment"] = evaluate(frames, produced, body, lo, hi, arena)
        if corrected:
            results[f"{name} — raw [CORRECTED arm]"] = evaluate(
                corrected, preds, body, lo, hi, arena
            )

    for name, result in results.items():
        print_block(name, result)

    print(f"\n{'=' * 78}\nSUMMARY — 40 frame-disjoint frames, 3 K days\n{'=' * 78}")
    print(f"{'arm':46s} {'cover%':>7s} {'both%':>7s} {'FP':>4s} {'FP/fr':>6s} "
          f"{'shrTail':>8s} {'m-only':>7s}")
    for name, result in results.items():
        agg = totals(result)
        n = agg["frames"] or len(frames)
        print(f"{name:46s} {agg['coverage']:>7.2f} {agg['recall']:>7.2f} {agg['fp']:>4d} "
              f"{agg['fp'] / n:>6.3f} {agg['shared_tail']:>8d} "
              f"{agg['mouth_only']:>7d}")

    out = paths.work_dir / "k08_locked_benchmark.json"
    out.write_text(json.dumps({
        "benchmark": str(LOCKED), "frames": len(frames), "gt_instances": n_gt,
        "experiments": sorted(experiments), "n_groups": len(groups),
        "frame_overlaps": overlaps, "contamination": contam,
        "arms": {k: totals(v) for k, v in results.items()},
        "by_regime": {k: {r: d for r, d in v.items() if d["frames"]} for k, v in results.items()},
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    print("\nn_groups is still 1. 40 frames of one trio is a locked test set, not a "
          "population.")


if __name__ == "__main__":
    main()

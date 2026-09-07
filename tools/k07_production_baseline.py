"""K07 — round 8 is the production top model. Re-baseline everything against it.

## The correction

`MANIFEST.json` records `production_models.sleap_top = "round7"`. That figure is
not a measurement: it is a hardcoded constant, `PRODUCTION = {"top": "round7"}`,
in `tools/build_asset_manifest.py`, and the manifest merely echoes it back. The
chronology contradicts it:

* round-7 top models trained **2026-07-04**;
* every retained corpus prediction was produced **2026-07-07 to 07-13** — the
  `.slp` provenance names `round7_260704.centroid_n1295` +
  `.centered_instance_n1295` explicitly;
* round-8 models, **both top and side**, trained **2026-07-23**.

So the corpus predictions are round 7 because round 8 did not exist yet, not
because anyone chose round 7 for top. Round 8 postdates round 7 for both views
and is the model the experimenter reported as failing at crossings — which is the
premise Phase B0 was opened on.

Round-8 top is **architecturally identical** to round 7 (`filters=24`,
`max_stride=64`, `crop_size=512`, `anchor_part='mouth'`, 70,755,866 params); only
the label set differs (1,382 vs 1,295, "strictmouth").

## The evaluation problem this creates, and the only honest way out

The 113 frames used throughout B0.3, k01 and k03 are *round-8 (train+val) minus
round-7*. Measured against the packages:

* **93 of the 113 are in round-8's training data.** Scoring round 8 there
  measures memorisation, exactly as k01 showed for round 7 (0.033 FP/frame
  in-sample against 0.460 held out — 13.8x).
* **20 of the 113 are round-8's validation split**, and **none of those 20 are in
  round 7**.

Those 20 frames are the only frames in this project held out from *both*
checkpoints. They are a tiny sample and are reported as such — but a small honest
number beats a large memorised one, and the difference between the two arms below
is itself the measurement.

Arms: round-7 top-down, round-8 top-down (production), and the k04 retrain if it
has finished. Each raw and under K03c assignment. Metrics are k03/k01/b03's,
unchanged.

Run:
    .venv\\Scripts\\python.exe tools/k07_production_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sleap_io as sio
from b03_heldout_comparison import ROUND7, VIEWS, build_heldout, load_keys, points, run_inference
from k01_misdetection_taxonomy import build_k_frames, label_stats
from k03_assignment_heldout import SPAN_LABELS, evaluate, totals
from k04_retrain_centered_instance import RUN_NAME

from medaka3d import load_catalog, load_paths

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "labs" / "20260803_mouth_anchored_assignment")
)
from assignment import OriginParams, assign_frame_origin, span_model_from_labels

ROUND8 = Path(r"Z:\StephensU\medaka\sleap_models\retrain_20260722_round8")
ROUND8_TOP = [
    ROUND8 / "top" / "runs" / "round8_260723_strictmouth.centroid_n1382",
    ROUND8 / "top" / "runs" / "round8_260723_strictmouth.centered_instance_n1382",
]


def subsets(catalog, frames) -> dict[str, list]:
    """The 113, split by whether round 8 was trained on them."""
    train_keys = load_keys(catalog, "labels.round8.top.train")
    val_keys = load_keys(catalog, "labels.round8.top.val")
    return {
        "all 113 (93 in round-8 TRAIN — in-sample for production)": frames,
        "20 held out from BOTH round 7 and round 8 (the honest set)": [
            f for f in frames if f.key in val_keys
        ],
        "93 in round-8 training (memorisation check)": [
            f for f in frames if f.key in train_keys
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    print(__doc__.split("Run:")[0])

    missing = [m for m in ROUND8_TOP if not m.is_dir()]
    if missing:
        raise SystemExit(f"round-8 top models not found: {[str(m) for m in missing]}")

    lo, hi, body = label_stats(catalog)
    span = span_model_from_labels(sio.load_slp(str(catalog.get(SPAN_LABELS))))
    frames, meta = build_heldout(catalog, "top")
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
    k04_run = paths.work_dir / "k04_retrain" / RUN_NAME
    if not (k04_run / "best.ckpt").is_file():
        k04_run = Path("C:/sleap_cache/k04_runs") / RUN_NAME
    if (k04_run / "best.ckpt").is_file():
        models["k04 retrain (ms32/f32/intensity)"] = [VIEWS["top"]["topdown"][0], k04_run]
        print(f"including the k04 retrain from {k04_run}")
    else:
        print("k04 retrain not available yet; running the two existing checkpoints only")

    predictions = {
        name: run_inference(frames, paths_, args.device, args.batch_size)
        for name, paths_ in models.items()
    }
    assigned = {}
    for name, preds in predictions.items():
        produced, only = {}, 0
        for frame in frames:
            result = assign_frame_origin(preds.get(frame.key, []), shape, span, OriginParams())
            produced[frame.key] = result.instances
            only += result.n_mouth_only
        assigned[name] = (produced, only)

    results: dict = {}
    for subset_name, subset in subsets(catalog, frames).items():
        keys = {f.key for f in subset}
        print(f"\n{'=' * 92}\n{subset_name.upper()}  —  {len(subset)} frames\n{'=' * 92}")
        print(f"{'model':34s} {'arm':11s} {'GT':>4s} {'pred':>5s} {'cover%':>7s} "
              f"{'both%':>7s} {'FP':>4s} {'FP/fr':>6s} {'shrTail':>8s} {'m-only':>7s}")
        for name in models:
            for arm, source in (("raw", predictions[name]), ("assigned", assigned[name][0])):
                scoped = {k: v for k, v in source.items() if k in keys}
                agg = totals(evaluate(subset, scoped, body, lo, hi, arena))
                results.setdefault(subset_name, {})[f"{name} — {arm}"] = agg
                print(f"{name:34s} {arm:11s} {agg['n_gt']:>4d} {agg['n_pred']:>5d} "
                      f"{agg['coverage']:>7.2f} {agg['recall']:>7.2f} {agg['fp']:>4d} "
                      f"{agg['fp'] / max(len(subset), 1):>6.3f} "
                      f"{agg['shared_tail']:>8d} {agg['mouth_only']:>7d}")

    print(f"\n{'=' * 92}\nWHAT THIS SHOWS\n{'=' * 92}")
    honest = next(k for k in results if k.startswith("20 held"))
    insample = next(k for k in results if k.startswith("93 in"))
    for name in models:
        h = results[honest][f"{name} — raw"]
        i = results[insample][f"{name} — raw"]
        h_rate = h["fp"] / 20.0
        i_rate = i["fp"] / 93.0
        print(f"  {name}")
        print(f"    honest 20   {h['fp']:>3d} FP = {h_rate:.3f}/frame   "
              f"recall {h['recall']:6.2f}%   shared tails {h['shared_tail']}")
        print(f"    in-sample 93 {i['fp']:>3d} FP = {i_rate:.3f}/frame   "
              f"recall {i['recall']:6.2f}%   shared tails {i['shared_tail']}")
        if i_rate:
            print(f"    -> honest rate is {h_rate / i_rate:.1f}x the in-sample rate")

    out = paths.work_dir / "k07_production_baseline.json"
    out.write_text(json.dumps({"meta": meta, "results": results,
                               "mouth_only": {k: v[1] for k, v in assigned.items()},
                               "models": {k: [str(p) for p in v] for k, v in models.items()}},
                              indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    print("\n20 frames is a small sample: a single instance moves recall ~1.7 points.")
    print("It is reported because it is the only subset neither model was fitted to.")
    _ = ROUND7


if __name__ == "__main__":
    main()

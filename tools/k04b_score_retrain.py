"""K04b — score the retrained centered-instance head on the task metric.

Applies the rule frozen in `labs/20260803_centered_instance_capacity` before
training started. The criterion is the **mouth-only count under K03c
assignment**, because that is the quantity the representational gap shows up in:
if the retrain puts a peak on the ~10 tails the old model never represented at
any threshold, mouths stop having to abstain.

``val/loss`` is deliberately not a criterion and is reported only as evidence
about *training*. The round-7 run sat flat at 0.00023/0.00029 from epoch 95 to
200 while failing exactly this way; a confmap MSE over a 256x256 grid is
dominated by empty background and is nearly blind to a misplaced tail peak.

Only the centered-instance head changed. The centroid head is round-7's,
unmodified, so any difference is attributable to the one thing that moved.

Overfitting is checked directly rather than inferred from the loss: the same
model is scored on the 1,288 frames that **were** in round-7 training and on the
113 that were not. The gap between those two numbers is the overfitting, and it
is a gap the validation split cannot show because the split is drawn from the
same three fish, tank and lighting.

Run:
    .venv\\Scripts\\python.exe tools/k04b_score_retrain.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sleap_io as sio
from b03_heldout_comparison import VIEWS, build_heldout, points, run_inference
from k01_misdetection_taxonomy import build_k_frames, label_stats
from k03_assignment_heldout import SPAN_LABELS, evaluate, print_block, totals
from k04_retrain_centered_instance import RUN_NAME

from medaka3d import load_catalog, load_paths

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "labs" / "20260803_mouth_anchored_assignment")
)
from assignment import OriginParams, assign_frame_origin, span_model_from_labels

#: The rule, from the lab front matter. Baselines are round-7's measured values.
BASELINE_MOUTH_ONLY = 59
BASELINE_COVERAGE = 88.79
BASELINE_RECALL = 74.93
REJECT_RECALL = BASELINE_RECALL - 2.0
REJECT_COVERAGE = BASELINE_COVERAGE - 2.0

CENTROID = VIEWS["top"]["topdown"][0]


def find_run(paths, override: str | None) -> Path:
    """Prefer the copy on /bucket; fall back to the local scratch it came from."""
    if override:
        return Path(override)
    for candidate in (
        paths.work_dir / "k04_retrain" / RUN_NAME,
        Path("C:/sleap_cache/k04_runs") / RUN_NAME,
    ):
        if (candidate / "best.ckpt").is_file():
            return candidate
    raise SystemExit(
        f"no trained run found. Expected best.ckpt under "
        f"{paths.work_dir / 'k04_retrain' / RUN_NAME} or C:/sleap_cache/k04_runs/{RUN_NAME}. "
        f"Run tools/k04_retrain_centered_instance.py first."
    )


def loss_curve(run: Path) -> dict:
    """Summarise the loss, to report as training evidence and nothing more."""
    path = run / "training_log.csv"
    if not path.is_file():
        return {}
    rows = [line.split(",") for line in path.read_text(encoding="utf-8").splitlines()[1:]]
    train, val = [], []
    for row in rows:
        if len(row) > 2 and row[1]:
            train.append(float(row[1]))
        if len(row) > 2 and row[2]:
            val.append(float(row[2]))
    if not val:
        return {}
    return {
        "epochs": len(val),
        "train_final": train[-1] if train else None,
        "val_final": val[-1],
        "val_best": min(val),
        "val_best_epoch": int(np.argmin(val)),
        "train_val_gap": (train[-1] - val[-1]) if train else None,
    }


def assign(frames, predictions, shape, span) -> tuple[dict, int]:
    produced, mouth_only = {}, 0
    for frame in frames:
        result = assign_frame_origin(
            predictions.get(frame.key, []), shape, span, OriginParams()
        )
        produced[frame.key] = result.instances
        mouth_only += result.n_mouth_only
    return produced, mouth_only


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="path to the trained run directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-insample", action="store_true")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    run = find_run(paths, args.run)
    print(__doc__.split("Run:")[0])
    print(f"retrained run: {run}")

    curve = loss_curve(run)
    if curve:
        print(f"loss (evidence about training, NOT a criterion): {curve['epochs']} epochs, "
              f"train {curve['train_final']:.6f}, val {curve['val_final']:.6f}, "
              f"best val {curve['val_best']:.6f} @ epoch {curve['val_best_epoch']}")

    lo, hi, body = label_stats(catalog)
    span = span_model_from_labels(sio.load_slp(str(catalog.get(SPAN_LABELS))))
    frames, meta = build_heldout(catalog, "top")
    shape = tuple(frames[0].video.shape[1:3])
    print(f"held-out frames: {len(frames)}  n_groups {meta['n_groups']} {meta['groups']}")

    k_frames = build_k_frames(catalog)
    all_pts = np.vstack([points(i) for f in k_frames for i in f["instances"]])
    finite = all_pts[~np.isnan(all_pts).any(axis=1)]
    arena = (float(finite[:, 0].min() - body), float(finite[:, 1].min() - body),
             float(finite[:, 0].max() + body), float(finite[:, 1].max() + body))

    models = {
        "round-7 centered_instance": VIEWS["top"]["topdown"],
        "k04 centered_instance (ms32/f32/intensity)": [CENTROID, run],
    }
    results, mouth_only, raw = {}, {}, {}
    for name, model_paths in models.items():
        preds = run_inference(frames, model_paths, args.device, args.batch_size)
        raw[name] = preds
        results[f"{name} — raw"] = evaluate(frames, preds, body, lo, hi, arena)
        produced, only = assign(frames, preds, shape, span)
        results[f"{name} — K03c assignment"] = evaluate(frames, produced, body, lo, hi, arena)
        mouth_only[name] = only

    for name, result in results.items():
        print_block(name, result)

    print(f"\n{'=' * 78}\nPRE-REGISTERED RULE "
          f"(labs/20260803_centered_instance_capacity)\n{'=' * 78}")
    key = "k04 centered_instance (ms32/f32/intensity)"
    arm = totals(results[f"{key} — K03c assignment"])
    only = mouth_only[key]
    checks = [
        (f"1. mouth-only < {BASELINE_MOUTH_ONLY}", only < BASELINE_MOUTH_ONLY,
         f"{BASELINE_MOUTH_ONLY} -> {only}"),
        (f"2. coverage >= {BASELINE_COVERAGE}%", arm["coverage"] >= BASELINE_COVERAGE,
         f"{BASELINE_COVERAGE:.2f}% -> {arm['coverage']:.2f}%"),
        (f"3. both-node recall >= {BASELINE_RECALL}%", arm["recall"] >= BASELINE_RECALL,
         f"{BASELINE_RECALL:.2f}% -> {arm['recall']:.2f}%"),
    ]
    for label, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label:38s} {detail}")
    if all(p for _, p, _ in checks):
        verdict = "ADOPT the retrained head"
    elif arm["recall"] < REJECT_RECALL or arm["coverage"] < REJECT_COVERAGE:
        verdict = "REJECT"
    else:
        verdict = "PARK — name the blocker"
    print(f"\n  -> {verdict}")

    insample = {}
    if not args.skip_insample:
        print(f"\n{'=' * 78}\nOVERFITTING — in-sample versus held-out, measured not inferred\n"
              f"{'=' * 78}")
        held_keys = {f.key for f in frames}
        train_frames = [
            type("F", (), {"key": f["key"], "video": f["video"],
                           "frame_idx": f["frame_idx"], "instances": f["instances"]})()
            for f in k_frames if f["key"] not in held_keys
        ]
        print(f"  frames in round-7 training: {len(train_frames)}")
        for name, model_paths in models.items():
            preds = run_inference(train_frames, model_paths, args.device, args.batch_size)
            agg = totals(evaluate(train_frames, preds, body, lo, hi, arena))
            out_agg = totals(results[f"{name} — raw"])
            in_rate = agg["fp"] / max(agg["frames"], 1)
            out_rate = out_agg["fp"] / max(out_agg["frames"], 1)
            insample[name] = {"in_sample": agg, "in_rate": in_rate, "out_rate": out_rate}
            print(f"  {name}")
            print(f"    in-sample  {agg['fp']:>4d} FP / {agg['frames']:>4d} fr = "
                  f"{in_rate:.3f} FP/frame   recall {agg['recall']:.2f}%")
            print(f"    held out   {out_agg['fp']:>4d} FP / {out_agg['frames']:>4d} fr = "
                  f"{out_rate:.3f} FP/frame   recall {out_agg['recall']:.2f}%")
            print(f"    ratio      {out_rate / in_rate:.1f}x   "
                  f"recall gap {agg['recall'] - out_agg['recall']:+.2f} pts")
        print("  A widening ratio against round 7 is overfitting, whatever the loss says.")

    out = paths.work_dir / "k04b_score_retrain.json"
    out.write_text(
        json.dumps({"run": str(run), "loss_curve": curve, "meta": meta,
                    "arms": {k: totals(v) for k, v in results.items()},
                    "mouth_only": mouth_only,
                    "overfitting": {k: {"in_rate": v["in_rate"], "out_rate": v["out_rate"],
                                        "in_sample": v["in_sample"]}
                                    for k, v in insample.items()}},
                   indent=2, default=float),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

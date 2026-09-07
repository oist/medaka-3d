"""K09 — what does ROUND 8 actually get wrong? The first corpus run of production.

Every retained prediction under `paths.sleap_output_dir` is **round 7**, produced
2026-07-07 to 07-13, before round 8 existed. So production has never been run
across this corpus and its failure rate outside its own 1,382 labelled frames is
unknown. That is the gap this closes, and it blocks two things at once:

* **"Train a model that beats round 8"** needs somewhere round 8 measurably
  fails. On the locked benchmark it is at 117/117 matched with 0 false positives,
  so there is nothing there to beat — roughly 7 imperfect instances in 120, on a
  benchmark that is 30% its own output.
* **The tracking work** needs the detection stream it will actually consume,
  characterised. Fixing a linker on top of an uncharacterised detector is how the
  retired repo produced results it could not conclude.

The measure is the cardinality anchor — one female and two males, so `n > 3`
implies false positives and `n < 3` implies misses, with no annotation required.
It is blind to a swapped tail or a mouth 40 px off the animal; **a count of 3 is
not a correct frame**. It is a bound on one failure mode, and the only free one.

Round 7 comes from the retained `.slp` chunks, so the two arms are scored on the
**same frames** with no re-inference of round 7. Round-8 inference matches the
retained run's settings (`peak_threshold=0.2`, `max_instances=6`) so the
comparison is of models, not of decoder configuration.

Run:
    .venv\\Scripts\\python.exe tools/k09_round8_corpus.py [--experiments A B] [--stride N]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sleap_io as sio
from b02_crossing_eval_set import chunk_files
from k05_cardinality_corpus import EXPECTED, frames_of, tally_counts
from k07_production_baseline import ROUND8_TOP

from medaka3d import load_paths
from medaka3d.cohort import CohortError, ExperimentId

#: Matches the retained round-7 inference config recorded in the chunk provenance.
PEAK_THRESHOLD = 0.2
MAX_INSTANCES = 6

#: Chosen before running, to span the corpus rather than confirm a hunch: K is the
#: only group round 8 trained on; I and H are the worst two groups under round 7
#: (84.96% and 89.52% exact-3, and H has the fattest predicted-span tail); G is
#: the best (94.13%). One experiment each, so a difference is between groups.
DEFAULT_EXPERIMENTS = (
    "20251225_K_1_2_FCMC",
    "20251224_I_1_2_FCMC",
    "20260106_H_1_2_FCMC",
    "20251223_G_1_2_FCMC",
)


def round8_counts(video_path: Path, indices: list[int], device: str, batch: int) -> dict[int, int]:
    """Instances per frame from round-8 top-down, keyed by absolute frame index."""
    import sleap_nn

    video = sio.load_video(str(video_path))
    labels = sio.Labels(
        labeled_frames=[sio.LabeledFrame(video=video, frame_idx=i, instances=[])
                        for i in indices],
        videos=[video],
        skeletons=[sio.Skeleton(["mouth", "tail"])],
    )
    predicted = sleap_nn.predict(
        labels,
        model_paths=[str(m) for m in ROUND8_TOP],
        device=device,
        batch_size=batch,
        peak_threshold=PEAK_THRESHOLD,
        max_instances=MAX_INSTANCES,
    )
    out: dict[int, int] = {}
    for frame in predicted.labeled_frames:
        kept = 0
        for instance in frame.instances:
            array = np.asarray(instance.numpy(), dtype=float)[:, :2]
            if not np.isnan(array).all():
                kept += 1
        out[int(frame.frame_idx)] = out.get(int(frame.frame_idx), 0) + kept
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="*", default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    paths = load_paths()
    print(__doc__.split("Run:")[0])
    print(f"stride {args.stride}; round-8 decoder matched to the retained run "
          f"(peak {PEAK_THRESHOLD}, max_instances {MAX_INSTANCES})\n")

    rows = []
    for name in args.experiments:
        directory = paths.sleap_output_dir / name
        if not chunk_files(directory):
            print(f"  {name}: no retained round-7 chunks, skipped")
            continue
        try:
            group = ExperimentId.parse(name).group
        except CohortError:
            group = "?"

        started = time.time()
        r7_frames, _ = frames_of(directory)
        indices = list(range(0, len(r7_frames), args.stride))
        r7 = np.array([len(r7_frames[i]) for i in indices], dtype=int)

        video = paths.video(name, "top_cam")
        r8_map = round8_counts(video, indices, args.device, args.batch_size)
        r8 = np.array([r8_map.get(i, 0) for i in indices], dtype=int)

        row = {
            "experiment": name, "group": group, "frames": len(indices),
            "round7": tally_counts(r7), "round8": tally_counts(r8),
            "seconds": round(time.time() - started, 1),
        }
        rows.append(row)
        a, b = row["round7"], row["round8"]
        star = " *" if group == "K" else "  "
        print(f"  {name:22s}{star} {len(indices):>6d} fr | "
              f"=3 r7 {a['pct_exact']:5.2f} r8 {b['pct_exact']:5.2f} | "
              f"<3 r7 {a['pct_under']:5.2f} r8 {b['pct_under']:5.2f} | "
              f">3 r7 {a['pct_over']:5.2f} r8 {b['pct_over']:5.2f}  ({row['seconds']:.0f}s)")

    if not rows:
        raise SystemExit("nothing analysed")

    print(f"\n{'=' * 92}\nROUND 7 vs ROUND 8, same frames, no annotation\n{'=' * 92}")
    print(f"{'group':6s} {'frames':>8s} | {'=3 r7':>7s} {'=3 r8':>7s} {'delta':>7s} | "
          f"{'<3 r7':>7s} {'<3 r8':>7s} | {'>3 r7':>7s} {'>3 r8':>7s}")
    pooled = {"round7": Counter(), "round8": Counter()}
    for row in rows:
        a, b = row["round7"], row["round8"]
        for arm, t in (("round7", a), ("round8", b)):
            for field in ("frames", "under", "exact", "over"):
                pooled[arm][field] += t[field]
        star = "*" if row["group"] == "K" else " "
        print(f"{row['group']:2s}{star}    {row['frames']:>8d} | "
              f"{a['pct_exact']:>7.2f} {b['pct_exact']:>7.2f} "
              f"{b['pct_exact'] - a['pct_exact']:>+7.2f} | "
              f"{a['pct_under']:>7.2f} {b['pct_under']:>7.2f} | "
              f"{a['pct_over']:>7.2f} {b['pct_over']:>7.2f}")
    n = pooled["round7"]["frames"]
    print(f"{'ALL':6s} {n:>8d} | "
          f"{100 * pooled['round7']['exact'] / n:>7.2f} "
          f"{100 * pooled['round8']['exact'] / n:>7.2f} "
          f"{100 * (pooled['round8']['exact'] - pooled['round7']['exact']) / n:>+7.2f} | "
          f"{100 * pooled['round7']['under'] / n:>7.2f} "
          f"{100 * pooled['round8']['under'] / n:>7.2f} | "
          f"{100 * pooled['round7']['over'] / n:>7.2f} "
          f"{100 * pooled['round8']['over'] / n:>7.2f}")

    print(f"\n{'=' * 92}\nWHERE ROUND 8 HAS HEADROOM\n{'=' * 92}")
    miss = 100 * pooled["round8"]["under"] / n
    over = 100 * pooled["round8"]["over"] / n
    print(f"  round 8 fails the count in {miss + over:.2f}% of frames "
          f"({miss:.2f}% miss an animal, {over:.2f}% over-detect).")
    print(f"  That is {int(pooled['round8']['under'] + pooled['round8']['over']):,d} frames of "
          f"{n:,d} sampled, and it is the target a retrain has to improve on.")
    print("  Cardinality is blind to localisation, so this is a floor on the error, "
          "not the error.")

    out = paths.work_dir / "k09_round8_corpus.json"
    out.write_text(json.dumps({"stride": args.stride, "expected": EXPECTED,
                               "peak_threshold": PEAK_THRESHOLD,
                               "max_instances": MAX_INSTANCES, "by_experiment": rows},
                              indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

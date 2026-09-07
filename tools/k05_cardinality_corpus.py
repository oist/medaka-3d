"""K05 — cardinality check on the retained corpus predictions, before and after assignment.

There are no labels outside Group K, so nothing here can measure *localisation*.
What it can measure, exactly and without annotation, is **counting**: every trial
holds one female and two males, every one of the 1,402 labelled frames holds
exactly 3 instances, so in any frame ``n > 3`` implies at least ``n - 3`` false
positives and ``n < 3`` implies at least ``3 - n`` missed animals.

That is the only free label this corpus offers and it is a weak one. It is blind
to a swapped tail, blind to a mouth 40 px off the animal, and blind to three
detections on two fish plus a reflection. **A count of 3 is not a correct frame.**
Read every number below as a bound on one specific failure mode.

The direction of the asymmetry matters and is reported rather than netted out.
K03c assignment can only ever *reduce* the instance count — it drops mouths near
the frame edge, merges mouths within 8 px, and turns a tail-only prediction into
a candidate rather than an instance. So it must improve ``>3`` and can only worsen
``<3``. Netting the two into one "count error" figure would hide exactly the
trade the method makes, so the two are kept apart and the causes decomposed.

The unit is the **group**, never the experiment: `K_1_1..3` are one trio on three
consecutive days. Experiment rows are printed for transparency; the group rows are
the result.

Run:
    .venv\\Scripts\\python.exe tools/k05_cardinality_corpus.py [--groups K E ...] [--limit-frames N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sleap_io as sio
from b02_crossing_eval_set import chunk_files
from b04_geometric_filter import apply_filter
from k01_misdetection_taxonomy import label_stats
from k03_assignment_heldout import SPAN_LABELS

from medaka3d import load_catalog, load_paths
from medaka3d.cohort import CohortError, ExperimentId

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "labs" / "20260803_mouth_anchored_assignment")
)
from assignment import OriginParams, assign_frame_origin, span_model_from_labels

EXPECTED = 3


def frames_of(experiment_dir: Path) -> tuple[list[list[np.ndarray]], tuple[int, int]]:
    """Per-frame instance arrays for one experiment, indexed by absolute frame."""
    chunks = chunk_files(experiment_dir)
    if not chunks:
        return [], (0, 0)
    per_frame: dict[int, list[np.ndarray]] = {}
    total, shape = 0, (0, 0)
    for _, path in chunks:
        labels = sio.load_slp(str(path))
        if labels.videos:
            video_shape = labels.videos[0].shape
            total = max(total, int(video_shape[0]))
            shape = (int(video_shape[1]), int(video_shape[2]))
        for frame in labels.labeled_frames:
            kept = []
            for instance in frame.instances:
                array = np.asarray(instance.numpy(), dtype=float)[:, :2]
                if not np.isnan(array).all():
                    kept.append(array)
            if kept:
                per_frame[int(frame.frame_idx)] = kept
    if not total:
        total = (max(per_frame) + 1) if per_frame else 0
    return [per_frame.get(i, []) for i in range(total)], shape


def tally_counts(counts: np.ndarray) -> dict:
    n = len(counts)
    if not n:
        return {}
    return {
        "frames": n,
        "zero": int((counts == 0).sum()),
        "under": int((counts < EXPECTED).sum()),
        "exact": int((counts == EXPECTED).sum()),
        "over": int((counts > EXPECTED).sum()),
        "pct_under": 100.0 * float((counts < EXPECTED).mean()),
        "pct_exact": 100.0 * float((counts == EXPECTED).mean()),
        "pct_over": 100.0 * float((counts > EXPECTED).mean()),
        "surplus": int(np.maximum(counts - EXPECTED, 0).sum()),
        "deficit": int(np.maximum(EXPECTED - counts, 0).sum()),
    }


def analyse(
    experiment: str, directory: Path, span, params, limit: int | None, stride: int,
    bounds: tuple[float, float],
) -> dict | None:
    started = time.time()
    frames, shape = frames_of(directory)
    if not frames:
        print(f"  {experiment}: no top_cam chunks")
        return None
    if limit:
        frames = frames[:limit]
    # Deterministic decimation, never random. At 166 fps successive frames are
    # near-duplicates, so a fixed stride costs a rate estimate almost nothing and
    # a random sample would not be reproducible across runs.
    if stride > 1:
        frames = frames[::stride]

    lo, hi = bounds
    before = np.array([len(f) for f in frames], dtype=int)
    after = np.empty(len(frames), dtype=int)
    b04 = np.empty(len(frames), dtype=int)
    causes = Counter()
    for index, instances in enumerate(frames):
        result = assign_frame_origin(instances, shape, span, params)
        after[index] = len(result.instances)
        causes["edge"] += result.n_dropped_edge
        causes["dedup"] += result.n_dropped_dedup_mouth
        causes["mouth_only"] += result.n_mouth_only
        causes["tail_only_dropped"] += max(len(instances) - result.n_anchors_raw, 0)
        # The filter Gate B0 actually adopted, run through its own implementation
        # rather than re-derived here. It *suppresses* — so its effect on the two
        # count errors is the comparison that matters: assignment keeps every
        # anchor and rations tails, b04 deletes the loser outright.
        kept, counts = apply_filter({0: instances}, lo, hi)
        b04[index] = len(kept[0])
        causes["b04_dropped_shared"] += counts["dropped_shared"]
        causes["b04_dropped_span"] += counts["dropped_span"]

    row = {
        "experiment": experiment,
        "shape": shape,
        "before": tally_counts(before),
        "after": tally_counts(after),
        "b04": tally_counts(b04),
        "causes": dict(causes),
        "seconds": round(time.time() - started, 1),
    }
    b, a, f = row["before"], row["after"], row["b04"]
    print(
        f"  {experiment:22s} {b['frames']:>7d} fr | <3 {b['pct_under']:5.2f} "
        f"b04 {f['pct_under']:5.2f} asg {a['pct_under']:5.2f} | "
        f"=3 {b['pct_exact']:5.2f} b04 {f['pct_exact']:5.2f} asg {a['pct_exact']:5.2f} | "
        f">3 {b['pct_over']:5.2f} b04 {f['pct_over']:5.2f} asg {a['pct_over']:5.2f} "
        f"({row['seconds']}s)"
    )
    return row


def aggregate(rows: list[dict], key) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        bucket = out.setdefault(
            key(row),
            {"experiments": [], "before": Counter(), "b04": Counter(), "after": Counter()},
        )
        bucket["experiments"].append(row["experiment"])
        for arm in ("before", "b04", "after"):
            for field in ("frames", "zero", "under", "exact", "over", "surplus", "deficit"):
                bucket[arm][field] += row[arm][field]
    for bucket in out.values():
        for arm in ("before", "b04", "after"):
            total = bucket[arm]["frames"]
            for field in ("under", "exact", "over"):
                bucket[arm][f"pct_{field}"] = 100.0 * bucket[arm][field] / total if total else 0.0
            bucket[arm] = dict(bucket[arm])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="*", default=None, help="restrict to these groups")
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every Nth frame; deterministic, reported in the output")
    parser.add_argument("--out", default="k05_cardinality_corpus.json")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    root = paths.sleap_output_dir
    if root is None:
        raise SystemExit("sleap_output_dir is not configured")

    span = span_model_from_labels(sio.load_slp(str(catalog.get(SPAN_LABELS))))
    lo, hi, _ = label_stats(catalog)
    params = OriginParams()
    print(__doc__.split("Run:")[0])
    print(f"span band [{span.bounds(0.01)[0]:.1f}, {span.bounds(0.01)[1]:.1f}] px, "
          f"edge {params.edge_min_px} px, dedup {params.dedup_px} px, "
          f"abstain {params.abstain_cost}")
    print(f"b04 arm (Gate B0's adopted filter): span [{lo:.1f}, {hi:.1f}] px + shared-node "
          f"suppression")
    if args.stride > 1:
        print(f"frame stride {args.stride} (every {args.stride}th frame, "
              f"{166 / args.stride:.1f} Hz effective) — percentages are unaffected by "
              f"decimation, absolute surplus/deficit counts scale by 1/{args.stride}.")

    directories = sorted(p for p in root.iterdir() if p.is_dir())
    selected: list[tuple[str, Path, str]] = []
    for directory in directories:
        try:
            group = ExperimentId.parse(directory.name).group
        except CohortError:
            print(f"  skipping unparseable {directory.name}")
            continue
        if args.groups and group not in args.groups:
            continue
        selected.append((directory.name, directory, group))

    groups = sorted({g for _, _, g in selected})
    print(f"\n{len(selected)} experiments across {len(groups)} groups: {groups}")
    print("K is the only group the detector was trained on; every other group is an")
    print("untested transfer, and this count check is the only handle on it.\n")

    by_experiment: dict[str, str] = {}
    rows: list[dict] = []
    for name, directory, group in selected:
        row = analyse(name, directory, span, params, args.limit_frames, args.stride,
                      (lo, hi))
        if row:
            by_experiment[name] = group
            rows.append(row)

    if not rows:
        raise SystemExit("nothing analysed")

    print(f"\n{'=' * 100}\nBY GROUP — the independent unit\n{'=' * 100}")
    grouped = aggregate(rows, lambda r: by_experiment[r["experiment"]])
    print(f"{'grp':4s} {'exp':>3s} {'frames':>9s} |"
          f"{'  <3: raw':>10s} {'b04':>6s} {'asgn':>6s} |"
          f"{'  =3: raw':>10s} {'b04':>6s} {'asgn':>6s} |"
          f"{'  >3: raw':>10s} {'b04':>6s} {'asgn':>6s}")
    for group in sorted(grouped):
        b, f, a = grouped[group]["before"], grouped[group]["b04"], grouped[group]["after"]
        mark = "*" if group == "K" else " "
        print(f"{group:2s}{mark}  {len(grouped[group]['experiments']):>3d} {b['frames']:>9d} |"
              f"{b['pct_under']:>10.2f} {f['pct_under']:>6.2f} {a['pct_under']:>6.2f} |"
              f"{b['pct_exact']:>10.2f} {f['pct_exact']:>6.2f} {a['pct_exact']:>6.2f} |"
              f"{b['pct_over']:>10.2f} {f['pct_over']:>6.2f} {a['pct_over']:>6.2f}")
    print("  * = the only group with labels, and the only group the detector saw.")

    k_rows = [r for r in rows if by_experiment[r["experiment"]] == "K"]
    other = [r for r in rows if by_experiment[r["experiment"]] != "K"]
    if k_rows and other:
        print(f"\n{'=' * 100}\nK VERSUS THE REST — the rollout question\n{'=' * 100}")
        for label, subset in (("K (trained on)", k_rows), ("non-K (transfer)", other)):
            agg = aggregate(subset, lambda _: "all")["all"]
            b, f, a = agg["before"], agg["b04"], agg["after"]
            print(f"  {label:18s} {len(subset):2d} exp, {b['frames']:>9d} frames")
            for arm_name, arm in (("raw", b), ("b04", f), ("assignment", a)):
                print(f"      {arm_name:11s} =3 {arm['pct_exact']:6.2f}%   "
                      f"<3 {arm['pct_under']:6.2f}%   >3 {arm['pct_over']:6.2f}%   "
                      f"surplus {arm['surplus']:>7d}  deficit {arm['deficit']:>8d}")
        spread = [
            (g, grouped[g]["before"]["pct_exact"]) for g in sorted(grouped)
        ]
        lo = min(spread, key=lambda x: x[1])
        hi = max(spread, key=lambda x: x[1])
        print(f"\n  spread across groups, before assignment: "
              f"{lo[0]} {lo[1]:.2f}% .. {hi[0]} {hi[1]:.2f}% exact-3")

    causes = Counter()
    for row in rows:
        causes.update(row["causes"])
    print(f"\n{'=' * 100}\nWHY THE COUNT CHANGED (pooled instance-level)\n{'=' * 100}")
    for name, value in causes.most_common():
        print(f"  {name:20s} {value:>12,d}")
    print("  tail_only_dropped and edge are pure count reductions; mouth_only instances")
    print("  are retained as anchors and still count, so they do not reduce it.")

    out = paths.work_dir / args.out
    out.write_text(
        json.dumps(
            {"params": {"edge_min_px": params.edge_min_px, "dedup_px": params.dedup_px,
                        "abstain_cost": params.abstain_cost, "reject_q": params.reject_q},
             "stride": args.stride, "limit_frames": args.limit_frames,
             "by_experiment": rows, "group_of": by_experiment,
             "by_group": grouped, "causes": dict(causes)},
            indent=2, default=float,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

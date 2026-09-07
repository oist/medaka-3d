"""K06 — enumerate the partial instances that are worth completing by hand.

Task 5 of the Group-K detection-quality pass, and the only annotation on the list
that survives scrutiny. **Not** more crossing frames: B0.1 showed the training set
is already 53.02% crossing frames under the primary definition, and the failures
the model actually makes are *easier* than 44--60% of what it already trains on.

The case for this one instead. 5.9% of round-7 top training instances are missing
a node (1.7% mouth-only, 3.9% tail-only), and the model reproduces that behaviour
as **36.5% of its held-out false positives** — category E, 19 of 52, the second
largest and the one no geometric rule can remove, because a one-node instance is
not implausible, it is merely incomplete. K03c assignment removes the *tail-only*
half by construction (a lone tail becomes a candidate, not an instance) and can do
nothing about the rest.

This does not do the annotation. It produces the work list, sized and ordered, so
the experimenter can decide whether to spend the hours — and so nobody has to
re-derive which instances they are.

Output, under ``work_dir/k06_partial_instances/``:

* ``worklist.csv``     one row per partial instance: package, video, frame, which
                       node is missing, the other node's position, and the
                       nearest complete neighbour, for context
* ``worklist.json``    the same, plus the per-package summary
* ``frames.txt``       the distinct frame indices, for driving a labelling GUI

Run:
    .venv\\Scripts\\python.exe tools/k06_partial_instance_worklist.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

import numpy as np
import sleap_io as sio
from b03_heldout_comparison import experiment_of, leaf_video, points

from medaka3d import load_catalog, load_paths
from medaka3d.cohort import CohortError, ExperimentId

#: Every top package that feeds a top model. round9b is side-only and is listed
#: for contrast: it completed *instances per frame*, never *nodes per instance*,
#: which is why the side partial rate did not move.
PACKAGES = (
    "labels.round7.top",
    "labels.round8.top.train",
    "labels.round8.top.val",
    "labels.round8.side.train",
    "labels.round8.side.val",
    "training_labels:training/sleap/legacy/retrain_20260728_round9b/side/train_labels.pkg.slp",
)

#: Packages whose completion would change a *top* model. The others are reported
#: for context and are not part of the recommended job.
TOP_TRAINING = ("labels.round7.top", "labels.round8.top.train")

MOUTH, TAIL = 0, 1


def scan(labels) -> tuple[list[dict], Counter]:
    """Every incomplete instance in one package, with enough context to find it."""
    rows: list[dict] = []
    tally = Counter()
    for frame in labels.labeled_frames:
        arrays = [points(i) for i in frame.instances]
        video = leaf_video(frame.video)
        experiment = experiment_of(video) or ""
        complete = [a for a in arrays if not np.isnan(a).any()]
        for index, array in enumerate(arrays):
            missing_mouth = bool(np.isnan(array[MOUTH]).any())
            missing_tail = bool(np.isnan(array[TAIL]).any())
            if missing_mouth and missing_tail:
                tally["empty"] += 1
                kind, present, position = "empty", None, (None, None)
            elif missing_mouth:
                tally["mouth_missing"] += 1
                kind, present, position = "mouth_missing", "tail", tuple(array[TAIL])
            elif missing_tail:
                tally["tail_missing"] += 1
                kind, present, position = "tail_missing", "mouth", tuple(array[MOUTH])
            else:
                tally["complete"] += 1
                continue
            neighbour = float("nan")
            if position[0] is not None and complete:
                point = np.array(position, dtype=float)
                neighbour = min(
                    float(np.hypot(*(point - c.mean(axis=0)))) for c in complete
                )
            rows.append(
                {
                    "experiment": experiment,
                    "video": video,
                    "frame_idx": int(frame.frame_idx),
                    "instance_index": index,
                    "missing": kind,
                    "present_node": present,
                    "x": None if position[0] is None else round(float(position[0]), 2),
                    "y": None if position[1] is None else round(float(position[1]), 2),
                    "n_instances_in_frame": len(arrays),
                    "n_complete_in_frame": len(complete),
                    "px_to_nearest_complete": (
                        None if np.isnan(neighbour) else round(neighbour, 1)
                    ),
                }
            )
    return rows, tally


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes-per-instance", type=float, default=0.5,
                        help="budget assumption, stated rather than hidden")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    print(__doc__.split("Run:")[0])

    out_dir = paths.work_dir / "k06_partial_instances"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    all_rows: list[dict] = []
    print(f"{'package':58s} {'inst':>6s} {'complete':>9s} {'mouth-':>7s} {'tail-':>6s} "
          f"{'empty':>6s} {'partial%':>9s}")
    for key in PACKAGES:
        try:
            path = catalog.get(key)
        except Exception as error:
            print(f"  {key}: unavailable ({error})")
            continue
        labels = sio.load_slp(str(path))
        rows, tally = scan(labels)
        total = sum(tally.values())
        partial = total - tally["complete"]
        summary[key] = {
            "path": str(path),
            "frames": len(labels.labeled_frames),
            "instances": total,
            **dict(tally),
            "partial": partial,
            "partial_pct": 100.0 * partial / total if total else 0.0,
            "is_top_training": key in TOP_TRAINING,
        }
        for row in rows:
            row["package"] = key
        all_rows.extend(rows)
        print(f"  {key[:56]:56s} {total:>6d} {tally['complete']:>9d} "
              f"{tally['mouth_missing']:>7d} {tally['tail_missing']:>6d} "
              f"{tally['empty']:>6d} {summary[key]['partial_pct']:>8.2f}%")

    job = [r for r in all_rows if r["package"] in TOP_TRAINING]
    print(f"\n{'=' * 78}\nTHE RECOMMENDED JOB — top training packages only\n{'=' * 78}")
    by_package = Counter(r["package"] for r in job)
    for key in TOP_TRAINING:
        if key in summary:
            s = summary[key]
            print(f"  {key:28s} {by_package[key]:>4d} partial instances "
                  f"across {len({r['frame_idx'] for r in job if r['package'] == key})} frames "
                  f"({s['partial_pct']:.2f}% of {s['instances']})")
    kinds = Counter(r["missing"] for r in job)
    print("\n  by kind: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))

    groups: set[str] = set()
    for row in job:
        try:
            groups.add(ExperimentId.parse(row["experiment"]).group)
        except CohortError:
            continue
    print(f"  groups represented: {sorted(groups) or 'unparseable'}  (n = {len(groups)})")

    minutes = len(job) * args.minutes_per_instance
    print(f"\n  budget at {args.minutes_per_instance} min/instance: "
          f"{minutes:.0f} min = {minutes / 60:.1f} h")
    print("  for comparison, annotating every B0.2 encounter candidate is 367.1 h.")

    contact = [r for r in job if (r["px_to_nearest_complete"] or 1e9) < 100.0]
    print(f"\n  {len(contact)} of {len(job)} sit within 100 px of a complete instance, i.e. at")
    print("  a crossing — those are the ones the annotator will find hardest and the")
    print("  ones whose absence the model most visibly reproduces.")

    csv_path = out_dir / "worklist.csv"
    fields = ["package", "experiment", "video", "frame_idx", "instance_index", "missing",
              "present_node", "x", "y", "n_instances_in_frame", "n_complete_in_frame",
              "px_to_nearest_complete"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in job)

    frames_path = out_dir / "frames.txt"
    frames_path.write_text(
        "\n".join(str(i) for i in sorted({r["frame_idx"] for r in job})), encoding="utf-8"
    )
    json_path = out_dir / "worklist.json"
    json_path.write_text(
        json.dumps({"summary": summary, "job": job,
                    "minutes_per_instance": args.minutes_per_instance}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {csv_path}\n      {frames_path}\n      {json_path}")
    print("\nThis enumerates the work. It does not do it — completing a node is a human")
    print("judgement about which pixels are the animal, and nothing here should guess it.")


if __name__ == "__main__":
    main()

"""K14 — recover the occluded-keypoint coordinates that round 8 dropped.

## What was lost

A SLEAP point carries `xy`, a `visible` flag and a `complete` flag.
`Instance.numpy()` returns NaN for anything flagged `visible=False`, so that view
cannot distinguish *never annotated* from *annotated but occluded*. Reading the
underlying `PointsArray` instead:

| package | points | not visible, **with xy** | not visible, **NaN** |
|---|---|---|---|
| round 7 top | 7,770 | **226** | 0 |
| locked HITL-v2 benchmark | 720 | **15** | 0 |
| round 8 top train | 8,292 | 0 | **248** |
| round 8 side train | 16,586 | 0 | **986** |

**Round 7 and the locked benchmark have no missing points at all.** Every node was
annotated; the "partial" ones are simply flagged occluded, positioned where the
annotator estimated them. **Round 8 kept the flag and dropped the coordinate.**
It was built by correcting round-7 output, and the coordinates were lost in that
pass.

These are human estimates of where an occluded node is, so they are worth
recovering: as a prior for a future skeleton (pectoral fins), as a completeness
audit, and because `k06_partial_instance_worklist.py` counts NaN via `numpy()`
and therefore **overstates the annotation job** — much of it is already done.

## What this does

For every round-8 frame with a round-7 counterpart, match instances by Hungarian
assignment on the nodes visible in both, then copy `xy` into any round-8 point
that is NaN where round 7 has a coordinate.

**The `visible=False` flag is preserved.** Making these points visible would
change what a trained model is asked to predict — a node it cannot see — which is
a modelling decision, not a data repair, and is deliberately not taken here.
sleap-nn builds confmaps from visible points only, so the recovered packages
train identically to the originals; what changes is that the information is no
longer thrown away.

Outputs go to `work_dir/k14_recovered_labels/` (on `/bucket`, per design rule 7).
Promoting them into `asset_root` needs `tools/build_asset_manifest.py` re-run so
`verify_all()` stays green — that is left as an explicit step, not done silently.

Run:
    .venv\\Scripts\\python.exe tools/k14_recover_hidden_keypoints.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

import numpy as np
import sleap_io as sio
from b03_heldout_comparison import leaf_video
from scipy.optimize import linear_sum_assignment

from medaka3d import load_catalog, load_paths

#: round-8 package -> the round-7 package holding the same annotation pass.
PAIRS = {
    "labels.round8.top.train": "labels.round7.top",
    "labels.round8.top.val": "labels.round7.top",
    "labels.round8.side.train": "labels.round7.side",
    "labels.round8.side.val": "labels.round7.side",
}

#: Round 8 was made by correcting round 7, so matched instances sit almost on top
#: of each other. Generous bound; the realised distances are reported so a bad
#: match would be visible rather than silent.
MATCH_TOL_PX = 20.0


def frame_index(labels) -> dict:
    """(leaf video, frame_idx) -> LabeledFrame, keyed on the frame's own video.

    **Deliberately does not use the ``source_inds`` remap that
    ``b03_heldout_comparison.build_heldout`` applies.** That remap re-attaches a
    frame to the first video whose ``source_inds`` contains its index, which is
    wrong whenever one package holds more than one camera — measured on
    ``labels.round8.side.train``:

    | keying | result |
    |---|---|
    | ``frame.video`` | side1 1382 / side2 1382, **2,764 unique keys** (correct) |
    | ``source_inds`` remap | side1 2764, **1,381 unique keys** |

    The remap collapses side2 onto side1 and silently discards half the package.
    It also loses one key on the single-camera top package (1,382 -> 1,381).
    ``frame.video`` reproduces the camera split exactly, so it is used here.
    """
    out = {}
    for frame in labels.labeled_frames:
        out[(leaf_video(frame.video).lower(), int(frame.frame_idx))] = frame
    return out


def visible_xy(points) -> np.ndarray:
    """(n_nodes, 2) with NaN where the point has no usable coordinate."""
    out = np.full((len(points), 2), np.nan)
    for i, row in enumerate(points):
        if np.isfinite(row["xy"]).all():
            out[i] = row["xy"]
    return out


def pair_cost(a: np.ndarray, b: np.ndarray) -> float:
    both = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
    if not both.any():
        return np.inf
    return float(np.hypot(*(a[both] - b[both]).T).mean())


def recover(donor_frame, target_frame, stats: Counter, distances: list,
            log: list | None = None, video: str = "", frame_idx: int = -1) -> int:
    """Copy coordinates into NaN points of ``target_frame``. Returns n recovered."""
    donors = [i.points for i in donor_frame.instances]
    targets = [i.points for i in target_frame.instances]
    if not donors or not targets:
        return 0
    # Match on any node with a coordinate in both — critically including
    # instances whose *mouth* is the missing node, which an earlier
    # mouth-anchored pass silently skipped and undercounted.
    cost = np.array([[pair_cost(visible_xy(t), visible_xy(d)) for d in donors]
                     for t in targets])
    rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e6))

    recovered = 0
    for r, c in zip(rows, cols, strict=False):
        if not np.isfinite(cost[r, c]) or cost[r, c] > MATCH_TOL_PX:
            stats["unmatched_instance"] += 1
            continue
        distances.append(cost[r, c])
        target, donor = targets[r], donors[c]
        for n in range(len(target)):
            if np.isfinite(target[n]["xy"]).all():
                continue
            stats["nan_points"] += 1
            if np.isfinite(donor[n]["xy"]).all():
                # Preserve visible=False. This restores the coordinate, it does
                # not assert the node became observable.
                target[n]["xy"] = donor[n]["xy"]
                recovered += 1
                stats["recovered"] += 1
                if log is not None:
                    log.append({
                        "video": video, "frame_idx": frame_idx,
                        "instance_index": r, "node": str(target[n]["name"]),
                        "x": round(float(donor[n]["xy"][0]), 2),
                        "y": round(float(donor[n]["xy"][1]), 2),
                        "visible": False, "source": "round7",
                        "match_px": round(float(cost[r, c]), 4),
                    })
            else:
                stats["donor_also_missing"] += 1
    return recovered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be recovered, write nothing")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    out_dir = paths.work_dir / "k14_recovered_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(__doc__.split("Run:")[0])

    summary = {}
    for target_key, donor_key in PAIRS.items():
        target_labels = sio.load_slp(str(catalog.get(target_key)))
        donor_labels = sio.load_slp(str(catalog.get(donor_key)))
        donor_index = frame_index(donor_labels)
        target_index = frame_index(target_labels)

        before = Counter()
        for frame in target_labels.labeled_frames:
            for inst in frame.instances:
                for row in inst.points:
                    before["points"] += 1
                    if not np.isfinite(row["xy"]).all():
                        before["nan"] += 1

        stats, distances, log = Counter(), [], []
        shared = set(target_index) & set(donor_index)
        for key in shared:
            recover(donor_index[key], target_index[key], stats, distances,
                    log, key[0], key[1])

        after_nan = 0
        for frame in target_labels.labeled_frames:
            for inst in frame.instances:
                for row in inst.points:
                    if not np.isfinite(row["xy"]).all():
                        after_nan += 1

        no_donor = before["nan"] - stats["nan_points"]
        median_d = float(np.median(distances)) if distances else float("nan")
        summary[target_key] = {
            "points": before["points"], "nan_before": before["nan"],
            "nan_after": after_nan, "recovered": stats["recovered"],
            "donor_also_missing": stats["donor_also_missing"],
            "no_donor_frame": no_donor,
            "frames_target": len(target_index), "frames_shared": len(shared),
            "match_median_px": median_d,
            "unmatched_instance": stats["unmatched_instance"],
        }
        print(f"\n{target_key}")
        print(f"  frames {len(target_index)} ({len(shared)} have a {donor_key} counterpart)")
        print(f"  instance match median {median_d:.3f} px, "
              f"{stats['unmatched_instance']} unmatched")
        print(f"  NaN points {before['nan']} -> {after_nan}   "
              f"RECOVERED {stats['recovered']}")
        if no_donor:
            print(f"    {no_donor} unreachable: frame has no round-7 counterpart")
        if stats["donor_also_missing"]:
            print(f"    {stats['donor_also_missing']} unreachable: round 7 lacks it too")

        if not args.dry_run and stats["recovered"]:
            stem = target_key.replace(".", "_")
            # CSV first: it is small, immediately auditable, and directly usable
            # as a reference list when placing nodes in a new skeleton. The .slp
            # re-embeds every labelled image and is large and slow by comparison.
            csv_path = out_dir / f"{stem}.recovered_points.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                fields = ["video", "frame_idx", "instance_index", "node", "x", "y",
                          "visible", "source", "match_px"]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(log)
            print(f"  wrote {csv_path}  ({len(log)} rows)")
            out = out_dir / f"{stem}.recovered.pkg.slp"
            print(f"  embedding {len(target_labels.labeled_frames)} frames -> {out.name} ...")
            sio.save_slp(target_labels, str(out), embed=True)
            summary[target_key]["output"] = str(out)
            summary[target_key]["csv"] = str(csv_path)
            print(f"  wrote {out}  ({out.stat().st_size / 1e9:.2f} GB)")

    total = sum(v["recovered"] for v in summary.values())
    print(f"\n{'=' * 78}\nRECOVERED {total} human-placed coordinates that were flagged "
          f"occluded\n{'=' * 78}")
    print("`visible=False` is preserved on every one, so these packages train")
    print("identically to the originals — sleap-nn builds confmaps from visible")
    print("points only. What changed is that the annotator's position estimate is")
    print("no longer discarded, which is what makes it usable as a prior for a")
    print("skeleton with more nodes.")
    if not args.dry_run:
        record = out_dir / "k14_recovery_summary.json"
        record.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {record}")
        print("\nTo promote into asset_root, move the packages there and re-run")
        print("tools/build_asset_manifest.py so verify_all() stays green.")


if __name__ == "__main__":
    main()

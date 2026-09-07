"""C16 — run SLEAP's own per-view instance tracker, and see whether it disagrees
with the 3D correspondence where the 3D correspondence is wrong.

=============================================================================
WHY THIS EXISTS
=============================================================================

`c02` (2026-08-13) decided this pipeline would reconstruct first and track
second, and stated plainly that it would **not** implement a 2D route. That
decision is retracted in `docs/decisions/2026-08-18-2d-first-tracking.md`: its
pre-registered rule came out degenerate (both ambiguity rates were exactly
0.000%), and it was taken while believing no human identity ground truth
existed. It does exist, and against it the current arrangement hands the
female's identity to another fish **8 times** in K_1_2.

The mechanism is now measured. `c14`/`c15` located candidate frames — a
one-frame displacement 10-600x its own local median whose size equals the
distance to the nearest instance, appearing on two tracks at once — and the
experimenter confirmed by eye that those frames are ID swaps. At those same
frames the 2D frame-to-frame assignment is **not ambiguous**: its winner beats
the runner-up by 0.46-1.02 body lengths, and adding the second and third node
never changes the winner (0 of 10, 0 of 15, 0 of 12). The 2D evidence was
unambiguous and was discarded, because per-view track IDs are the 3D
correspondence pushed back onto the detections and nothing exists to disagree
with it.

So the value of a 2D tracker here is not a cleverer match. It is **an
independent series that can refuse a 3D reassignment.**

=============================================================================
WHAT THIS RUNS
=============================================================================

`sleap_nn.tracking.Tracker`, per camera, on the cached `s16` detections, with:

``features='keypoints'``      multi-node similarity, which is the experimenter's
                              proposal — a whole pose is matched, not one point.
``scoring_method``            ``oks`` by default; ``euclidean_dist`` available,
                              and both are reported because OKS needs a keypoint
                              spread this project has never measured.
``track_matching_method``     ``hungarian``.
``max_tracks=3``              three animals are present in every frame of every
                              experiment, so this is a fact, not a guess.

The **tail is excluded by default** (``--nodes mouth pectoral``). It is the node
`s10` measured part-mixing on, and a tracker that trusts it will be least
reliable exactly at a crossing. Including it is a flag so the difference can be
measured rather than argued.

Nothing is promoted. This writes per-instance 2D track IDs beside the existing
output and reports one number: **at the frames the experimenter confirmed as
swaps, does the 2D tracker keep its identity while the 3D-derived one flips?**

**n_groups = 1, n_experiments = 1.**

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c16_sleap_tracker_2d.py
    .venv\\Scripts\\python.exe tools/c16_sleap_tracker_2d.py --nodes mouth pectoral tail
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01

from medaka3d import load_paths
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.track import UNTRACKED
from medaka3d.types import FPS, VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN, METHOD = "bottomup", "gated"
N_ANIMALS = 3


def run_tracker(detections, nodes: list[str], args) -> np.ndarray:
    """Per-instance 2D track ID for one camera, from `sleap_nn`'s tracker."""
    import sleap_io as sio
    from sleap_nn.tracking.tracker import Tracker

    tracker = Tracker.from_config(
        window_size=args.window_size,
        candidates_method=args.candidates,
        features="keypoints",
        scoring_method=args.scoring,
        scoring_reduction="mean",
        track_matching_method="hungarian",
        max_tracks=N_ANIMALS,
        tracking_target_instance_count=N_ANIMALS,
        min_match_points=args.min_match_points,
    )
    # Resolve the columns *before* building the skeleton: `sio.Skeleton`
    # replaces the strings in the list it is given with `Node` objects in place,
    # so a later `.index(name)` against the detector's node names fails.
    columns = [detections.nodes.index(name) for name in nodes]
    skeleton = sio.Skeleton(list(nodes))
    points = detections.points[:, columns, :].astype(float)
    start, stop = detections.start, detections.stop

    assigned = np.full(detections.frame_idx.shape[0], UNTRACKED, dtype=np.int64)
    names: dict[str, int] = {}
    for frame in range(detections.n_frames):
        lo, hi = int(start[frame]), int(stop[frame])
        if lo == hi:
            continue
        batch = [
            sio.PredictedInstance.from_numpy(
                points_data=points[row], skeleton=skeleton,
                point_scores=np.ones(len(nodes)), score=float(detections.score[row]),
            )
            for row in range(lo, hi)
        ]
        for offset, instance in enumerate(tracker.track(batch, frame_idx=frame)):
            if instance.track is None:
                continue
            key = instance.track.name
            if key not in names:
                names[key] = len(names)
            assigned[lo + offset] = names[key]
    return assigned


def tracklet_runs(frame_idx: np.ndarray, track: np.ndarray, n_frames: int) -> dict:
    """How long each 2D track holds without a break, in frames."""
    live = track != UNTRACKED
    lengths: Counter = Counter()
    for identifier in np.unique(track[live]):
        frames = np.sort(frame_idx[track == identifier])
        if frames.size == 0:
            continue
        cuts = np.flatnonzero(np.diff(frames) != 1)
        starts = np.r_[0, cuts + 1]
        stops = np.r_[cuts + 1, frames.size]
        for a, b in zip(starts, stops, strict=True):
            lengths[(int(identifier), int(a))] = int(b - a)
    values = np.asarray(sorted(lengths.values(), reverse=True)) if lengths else np.zeros(0, int)
    long = values[values >= FPS]
    return {
        "tracks": int(np.unique(track[live]).size),
        "runs": int(values.size),
        "median_run": float(np.median(values)) if values.size else float("nan"),
        "runs_over_one_second": int(long.size),
        "frames_in_runs_over_one_second": float(long.sum() / N_ANIMALS / n_frames),
        "coverage": float(live.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--nodes", nargs="+", default=["mouth", "pectoral"])
    parser.add_argument("--scoring", default="oks", choices=("oks", "euclidean_dist"))
    parser.add_argument("--candidates", default="fixed_window",
                        choices=("fixed_window", "local_queue"))
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-match-points", type=int, default=0)
    parser.add_argument("--views", nargs="+", default=list(VIEWS))
    args = parser.parse_args()

    paths = load_paths()
    tag = "-".join(args.nodes) + f"_{args.scoring}"
    out_dir = paths.work_dir / "c16_sleap_tracker_2d" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("sleap_tracker_2d", out_dir, name="c16") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        print(f"nodes {args.nodes}   scoring {args.scoring}   "
              f"candidates {args.candidates}   window {args.window_size}")

        npz = np.load(paths.work_dir / "c03_track_3d" / RUN / f"{METHOD}__{EXPERIMENT}.npz",
                      allow_pickle=False)
        swaps = json.loads(
            (paths.work_dir / "c15_swap_overlays" / "c15_swaps.json").read_text(
                encoding="utf-8"))
        flags = [
            {**hit, "interval": (item["t1"], item["t2"])}
            for item in swaps["swaps"] for hit in item["flags"]
        ]
        print(f"confirmed swap frames to test: {len(flags)}")

        summary, verdicts = {}, []
        for view in args.views:
            detections = c01.load(paths.work_dir, EXPERIMENT, view, RUN)
            started = time.time()
            track2d = run_tracker(detections, args.nodes, args)
            summary[view] = tracklet_runs(detections.frame_idx, track2d,
                                          detections.n_frames)
            np.save(out_dir / f"track2d__{view}.npy", track2d)
            record.output(f"track2d__{view}", out_dir / f"track2d__{view}.npy")
            cell = summary[view]
            print(f"  {view:<10} {time.time() - started:5.0f}s  tracks {cell['tracks']:>3}  "
                  f"runs {cell['runs']:>7,}  median {cell['median_run']:>6.0f}f  "
                  f">=1s {cell['runs_over_one_second']:>6,}  "
                  f"frames in >=1s {cell['frames_in_runs_over_one_second']:>6.1%}  "
                  f"coverage {cell['coverage']:.2%}")

            # The test, and it is not "did the 2D id change". The 3D-derived
            # assignment asserts that `row_before` at t-1 and `row_after` at t
            # are the same animal; that assertion IS the swap the experimenter
            # confirmed. A correct 2D tracker must therefore give them different
            # ids. What distinguishes agreement from disagreement is **where the
            # 2D tracker put the animal instead**.
            #
            #   refused   the id from `row_before` reappears at t on the OTHER
            #             row - the camera followed the same fish and
            #             contradicted the 3D link. This is the outcome that
            #             says the 2D evidence was there to be used.
            #   followed  it is on `row_after` - the 2D tracker made the same
            #             swap, and a 2D series would not have helped here.
            #   lost      it is absent at t.
            #
            # An earlier version of this compared the two ids directly and
            # reported every refusal as agreement, i.e. exactly backwards.
            track3d = npz[f"track__{view}"]
            start, stop = detections.start, detections.stop
            for hit in [f for f in flags if f["view"] == view]:
                frame = hit["frame"]
                previous = list(range(int(start[frame - 1]), int(stop[frame - 1])))
                current = list(range(int(start[frame]), int(stop[frame])))
                before = {int(track3d[r]): r for r in previous if track3d[r] != UNTRACKED}
                after = {int(track3d[r]): r for r in current if track3d[r] != UNTRACKED}
                row_before, row_after = before.get(hit["track"]), after.get(hit["track"])
                if row_before is None or row_after is None:
                    verdicts.append({**hit, "verdict": "not comparable"})
                    continue
                identifier = int(track2d[row_before])
                if identifier == UNTRACKED:
                    verdicts.append({**hit, "verdict": "2D untracked before"})
                    continue
                landed = [r for r in current if int(track2d[r]) == identifier]
                if not landed:
                    verdict = "2D lost it"
                elif landed[0] == row_after:
                    verdict = "2D followed the swap"
                else:
                    verdict = "2D refused the swap"
                verdicts.append({
                    **hit, "verdict": verdict, "track2d": identifier,
                    "row_before": int(row_before), "row_after": int(row_after),
                    "row_landed": int(landed[0]) if landed else -1,
                })

        record.param("per_view", summary)
        tally = Counter(v["verdict"] for v in verdicts)
        print(f"\nat the {len(verdicts)} confirmed swap frames:")
        for key, count in tally.most_common():
            print(f"  {key:<18} {count:>4}  {count / max(1, len(verdicts)):6.1%}")
        record.param("swap_frame_verdicts", dict(tally))

        (out_dir / "c16_summary.json").write_text(
            json.dumps({"experiment": EXPERIMENT, "n_groups": 1,
                        "groups": [ExperimentId.parse(EXPERIMENT).group],
                        "nodes": args.nodes, "scoring": args.scoring,
                        "per_view": summary, "verdicts": verdicts,
                        "tally": dict(tally)}, indent=2, default=float),
            encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

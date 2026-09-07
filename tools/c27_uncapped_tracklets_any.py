"""C27 - stage A uncapped, run SERIALLY, on any experiment.

=============================================================================
WHY SERIAL, AND WHY THAT IS NOT A TUNING CHOICE
=============================================================================

`c24` ran this tracker in 30 parallel chunks per camera and recorded that the
parallel run is equivalent to a serial one, on the strength of one measured
cell: K_1_2, ``side1_cam``, frames 12,000-24,000, **0 of 35,580 instances in a
different partition**. That measurement is real and this file reproduces it
exactly. **The generalisation from it is false.** Run across days and views on
2026-08-26:

    day     view          window 12,000-24,000     window 24,000-48,000
    K_1_1   top_cam                          0     -
    K_1_1   side1_cam            34.8% differ      -
    K_1_1   side2_cam            56.3% differ      -
    K_1_2   top_cam                          0            55.2% differ
    K_1_2   side1_cam                        0            87.7% differ
    K_1_2   side2_cam            86.6% differ      -
    K_1_3   top_cam              37.3% differ      -
    K_1_3   side1_cam            61.3% differ      -
    K_1_3   side2_cam            63.5% differ      -

Three passes in eleven measurements, all three at the same window, and **K_1_2
itself fails on its third camera and at its next boundary out.**

Raising the warm-up makes it worse, not better - K_1_3 ``top_cam`` differs on
37.3% at 500-vs-2,000 and on 45.9% at 2,000-vs-8,000 - and `c24`'s own decision
document says why, two sections before the claim this refutes:
``LocalQueueCandidates.current_tracks`` is appended to and never emptied, so a
track last seen 300,000 frames ago is still a candidate. More history means a
larger candidate set, so **a chunked run is not an approximation of a serial run
that a longer warm-up converges to. It is a different computation.**

Design rule 5 - *no flag may change which algorithm runs*. ``--chunk`` changes
which algorithm runs, so it is not offered here. The chunked production path is
**deleted** rather than kept behind a flag; `c24` still holds it, frozen, and
the numbers it produced are what the 2026-08-25 record describes.

=============================================================================
WHAT IS IMPORTED, AND WHAT THAT BUYS
=============================================================================

The tracker itself, the presence map and the census come from `c24` unchanged:

    c24.run_tracker  c24.presence  c24.label_census  c24.maximal_states
    c24.slice_detections  c24.partition_key   (the diagnostic only)

So the only difference between this and `c24` is that the whole recording is
tracked in one pass, per camera, and the three cameras are independent
processes. If the two ever disagree beyond that, `c24` changed.

`c24` measured the serial tracker at **56 minutes per camera**. The cameras do
not interact, so one experiment costs about an hour of wall clock.

See `docs/decisions/2026-08-26-transfer-to-k11-k13.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
import c24_uncapped_tracklets as c24

from medaka3d import load_paths
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.track import UNTRACKED
from medaka3d.types import VIEWS

RUN = "bottomup"


def track_one_view(payload: tuple) -> dict:
    """One camera, one pass over ``frames``, in its own process."""
    view, detections, nodes, window, scoring, min_match, lo, hi = payload
    started = time.time()
    labels, effective = c24.run_tracker(
        detections, nodes, window_size=window, scoring=scoring,
        min_match_points=min_match, frames=range(lo, hi))
    return {"view": view, "labels": labels, "effective": effective,
            "seconds": time.time() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--experiment", required=True,
                        help="e.g. 20251225_K_1_1_FCMC. Required and never "
                             "defaulted; a default is how c24's output would be "
                             "overwritten")
    parser.add_argument("--nodes", nargs="+", default=["mouth", "pectoral"])
    parser.add_argument("--scoring", default="oks", choices=("oks", "euclidean_dist"))
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-match-points", type=int, default=0)
    parser.add_argument("--views", nargs="+", default=list(VIEWS))
    parser.add_argument("--frames", nargs=2, type=int, default=None,
                        metavar=("LO", "HI"),
                        help="track only [LO, HI). FOR VALIDATION RUNS - it is "
                             "not a different algorithm, but it IS a different "
                             "amount of history, which this tracker is "
                             "sensitive to. The tool records it and tags the "
                             "output directory, so a partial run can never be "
                             "mistaken for a whole one")
    parser.add_argument("--check-chunking", nargs=2, type=int, default=None,
                        metavar=("LO", "HI"),
                        help="DIAGNOSTIC ONLY, and nothing here runs chunked. "
                             "Track frames [LO, HI) at warm-up 500 and at 2,000 "
                             "and compare the partitions. This is the "
                             "measurement that established chunking is a "
                             "different computation; it is per-experiment, "
                             "where c24's is fixed to K_1_2")
    args = parser.parse_args()

    paths = load_paths()
    experiment = args.experiment
    nodes = tuple(args.nodes)
    stem = "-".join(args.nodes) + f"_{args.scoring}"
    checking = args.check_chunking is not None
    if checking:
        leaf = f"{stem}_chunkcheck"
    elif args.frames:
        leaf = f"{stem}_serial_frames{args.frames[0]}-{args.frames[1]}"
    else:
        leaf = f"{stem}_serial"
    out_dir = (paths.work_dir / "c27_uncapped_tracklets_any" / experiment / leaf)
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("uncapped_tracklets_serial", out_dir, name="c27") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        record.param("chunked", False)
        record.param("reuses", "c24.run_tracker / presence / label_census / "
                               "maximal_states / slice_detections / partition_key")
        print(f"{experiment}   nodes {args.nodes}   scoring {args.scoring}   "
              f"window {args.window_size}")
        print("stage A with max_tracks = None, candidates local_queues "
              "(c24's configuration, imported)")
        print("SERIAL: the whole recording in one pass per camera, no chunking")
        print("n_groups = 1, n_experiments = 1\n")

        loaded = {view: c01.load(paths.work_dir, experiment, view, RUN)
                  for view in args.views}

        if checking:
            lo, hi = args.check_chunking
            view = args.views[0]
            detections = loaded[view]
            frames = np.arange(lo, hi)
            runs = {}
            for warmup in (500, 2_000):
                chunk, keep = c24.slice_detections(detections, lo - warmup, hi)
                labels, _ = c24.run_tracker(
                    chunk, nodes, window_size=args.window_size,
                    scoring=args.scoring, min_match_points=args.min_match_points,
                    frames=range(lo - warmup, hi))
                full = np.full(detections.frame_idx.size, UNTRACKED, np.int64)
                full[keep] = labels
                runs[warmup] = c24.partition_key(detections, full, frames)
            differ = int((runs[500] != runs[2_000]).sum())
            total = runs[500].size
            print(f"chunking diagnostic on {view} {lo:,}-{hi:,}: warm-up 500 vs "
                  f"2,000 -> {differ:,} of {total:,} instances "
                  f"({differ / total:.4%}) land in a different partition")
            print("  0 would mean the chunk's content does not depend on how much "
                  "history preceded it. Anything else means it does.")
            record.param("chunking_diagnostic",
                         {"view": view, "frames": [lo, hi], "instances": total,
                          "differ": differ, "fraction": differ / total})
            return

        lo, hi = args.frames if args.frames else (0, c01.N_FRAMES)
        tracked = {view: (c24.slice_detections(loaded[view], lo, hi)[0]
                          if args.frames else loaded[view])
                   for view in args.views}
        payloads = [(view, tracked[view], nodes, args.window_size, args.scoring,
                     args.min_match_points, lo, hi) for view in args.views]
        record.param("frames_tracked", [lo, hi])
        record.param("whole_recording", not args.frames)
        print(f"{len(payloads)} cameras, one process each, "
              f"frames {lo:,}-{hi:,} apiece"
              + ("" if not args.frames else "   !! A PARTIAL RUN, for validation"))
        started = time.time()
        with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
            done = list(pool.map(track_one_view, payloads))
        wall = time.time() - started
        core_seconds = sum(view["seconds"] for view in done)
        print(f"tracked in {wall:.0f}s wall clock, {core_seconds:.0f}s of core "
              f"time ({core_seconds / wall:.1f}x)\n")
        record.param("wall_seconds", wall)
        record.param("core_seconds", core_seconds)

        census, states, effective, unlabelled = {}, {}, {}, {}
        for result in done:
            view = result["view"]
            detections = tracked[view]
            labels = result["labels"]
            effective[view] = result["effective"]
            here = c24.presence(detections, labels)
            census[view] = c24.label_census(detections, labels, here)
            census[view]["chunks"] = 1
            census[view]["boundary_renames"] = 0
            census[view]["seconds"] = result["seconds"]
            states[view] = c24.maximal_states(here.sum(axis=1))
            unlabelled[view] = int((labels == UNTRACKED).sum())

            np.save(out_dir / f"track2d__{view}.npy", labels)
            record.output(f"track2d__{view}", out_dir / f"track2d__{view}.npy")

            print(f"  {view:<10} labels {census[view]['labels_created']:5,}   "
                  f"boundary renames 0   "
                  f"unlabelled {unlabelled[view]:,}   "
                  f"mean states/frame {states[view].mean():.1f}   "
                  f"{result['seconds'] / 60:.0f} min")

        record.param("census", census)
        record.param("unlabelled_instances", unlabelled)
        record.param("effective_tracker_config", effective)
        total_unlabelled = sum(unlabelled.values())
        record.param("unlabelled_total", total_unlabelled)
        record.param("mean_states_per_frame",
                     {v: float(states[v].mean()) for v in args.views})
        print(f"\n  instances left unlabelled, all cameras: {total_unlabelled:,}")
        print("  (c24 on K_1_2, 30-way chunked: 0.  c16 on K_1_2, capped: 15,487)")
        if total_unlabelled:
            print("  !! the uncapped arm left instances unlabelled. "
                  "Pre-registered as a failure of the arm; recorded, not repaired.")

        (out_dir / "c27_summary.json").write_text(
            json.dumps({"experiment": experiment, "n_groups": 1,
                        "n_experiments": 1, "chunked": False,
                        "groups": [ExperimentId.parse(experiment).group],
                        "nodes": list(nodes), "scoring": args.scoring,
                        "window_size": args.window_size,
                        "census": census,
                        "unlabelled_instances": unlabelled,
                        "unlabelled_total": total_unlabelled,
                        "effective_tracker_config": effective,
                        "mean_states_per_frame": {v: float(states[v].mean())
                                                  for v in args.views}},
                       indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

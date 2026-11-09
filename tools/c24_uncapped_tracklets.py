"""C24 — stage A without a track cap: every detection gets a label.

=============================================================================
WHY THIS EXISTS
=============================================================================

`c16` runs `sleap_nn`'s tracker with ``max_tracks=3`` because three animals are
present in every frame. That is true about the animals and false about the
detections. `c23` measured what it costs: on K_1_2 113,378-113,383 `side1_cam`
reports **four** instances, the cap hands out three labels, and for five frames
label 0 sits on a reflection in the tank wall while the real third fish carries
**no label at all**. A cap does not refuse a false positive; it makes one
animal pay for it.

That scene is not a single event. Measured before this tool was written, over
K_1_2's 14,501 frames where some view reports four or more detections: on
**4,515 animal-frames in 1,003 runs** the cheapest correspondence built only
from the instances `c16` labelled is more than 4 mm worse than the cheapest
correspondence over *all* the detector's instances, and the better answer uses
an instance the cap left unlabelled. The longest such run is 127 frames.

Measured on `side1_cam` 100,000-120,000:

* uncapped, the same 20,000 frames create **5** labels and leave **0**
  instances unlabelled; the real fish keeps its label across the reflection.
* the extra labels are not a fourth animal. They are **handovers**: label sets
  {0,1,3}, {0,2,4}, {1,2,3}, {0,2,3}, {1,2,4} account for **12.6%** of frames,
  against **0.91%** where four labels are genuinely present at once.
* **detection score does not separate them.** 92.7% of the instances carrying a
  non-primary label score above the 5th percentile of the primary labels'
  scores (means 1.683 vs 1.702). A cull by score removes real animals at nearly
  the rate it removes artefacts, so no cull by score is defensible here.

=============================================================================
WHAT THIS RUNS, AND THE ONE VARIABLE
=============================================================================

`c16`'s tracker, same nodes, same scoring, same window, with two changes and
nothing else:

``max_tracks=None``                   no cap on how many labels may exist.
``candidates_method='local_queues'``  stated explicitly. `c16` passes
                                      ``'fixed_window'``, but `sleap_nn`
                                      **silently switches to local queues**
                                      whenever ``max_tracks`` is set
                                      (`tracker.from_config`), so `c16`'s
                                      provenance records a candidate maker that
                                      did not run. Dropping the cap alone would
                                      hand the run back to `fixed_window` and
                                      change two things at once: on the same
                                      20,000 frames that arm fragments into
                                      **14** labels against local queues' 5.

The tracker's *effective* configuration is read back off the object and
recorded, rather than the arguments it was asked for. That is the whole lesson
of the paragraph above.

=============================================================================
CHUNKED, AND WHAT THAT COSTS
=============================================================================

`sleap_nn`'s tracker is sequential within a camera — frame *t*'s assignment is
what *t+1* matches against — and single-threaded, so one recording is one core
for **56 minutes**, measured. This machine has 16 cores. The recording is
therefore cut into chunks and the chunks run in parallel, each starting
``--warmup`` frames early and discarding that warm-up, so no chunk begins with
an empty candidate queue.

**The cost is real and is stated rather than hidden.** Label ids do not carry
across a chunk boundary, so each boundary renames three labels: with the
default chunk that is 29 renames per camera, against the 12.6% of frames that
already carry a handover. `c25` charges one lambda per slot for a rename, so
the boundaries cost 29 lambda per camera and nothing else — they cannot create
a swap, only a break the sequence model pays for once.

``--check-chunking`` runs one chunk twice, at the stated warm-up and at four
times it, and reports how many instances land in a different **partition** —
not a different id, which is meaningless across runs. If the warm-up is long
enough the answer is zero, and the parallelisation is then equivalent to the
serial run away from the boundaries.

**What none of this fixes.** Uncapping removes "there was no id left for the
real fish". It does not remove "the id was put on the wrong instance":
`sleap_nn` has no similarity threshold — `scores_to_cost_matrix` is `-scores`
and the Hungarian answer is taken whatever it costs — and `LocalQueueCandidates`
never retires a track. Measured on the same window, one-frame mouth
displacement inside a label: p99.9 of **12.5 px** but **18 links over 30 px**,
five of them over 100 px, which is 3.5 body spans in 6 ms. Uncapping cuts those
from 54 to 18; it does not reach zero. `c24b` gates them.

Nothing is promoted. `c16` is not modified and its output is not overwritten;
this writes beside it so the two can be compared instance by instance.

**n_groups = 1, n_experiments = 1.**

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c24_uncapped_tracklets.py
    .venv\\Scripts\\python.exe tools/c24_uncapped_tracklets.py --check-chunking
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from math import perm
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
RUN = "bottomup"
N_ANIMALS = 3
CAPPED_2D = Path("c16_sleap_tracker_2d") / "mouth-pectoral_oks"

#: The scene the experimenter read by eye and `c23` measured: `side1_cam` holds
#: four detections, one of them a reflection standing still against the wall.
REFLECTION = (113_374, 113_384)
REFLECTION_VIEW = "side1_cam"


def run_tracker(detections, nodes, *, window_size: int, scoring: str,
                min_match_points: int, frames: range) -> tuple[np.ndarray, dict]:
    """Per-instance 2D track label over ``frames``, with no cap on label count.

    A label id is issued the moment a detection matches no existing track;
    `sleap_nn` keeps issuing them and never retires one, so the ids only grow.
    The count is reported rather than bounded — bounding it is what `c16` did.
    """
    import sleap_io as sio
    from sleap_nn.tracking.tracker import Tracker

    tracker = Tracker.from_config(
        window_size=window_size,
        candidates_method="local_queues",
        features="keypoints",
        scoring_method=scoring,
        scoring_reduction="mean",
        track_matching_method="hungarian",
        max_tracks=None,
        tracking_target_instance_count=None,
        min_match_points=min_match_points,
    )
    effective = {
        "candidate_maker": type(tracker.candidate).__name__,
        "max_tracks": getattr(tracker.candidate, "max_tracks", None),
        "target_instance_count": tracker.tracking_target_instance_count,
        "pre_cull_to_target": tracker.tracking_pre_cull_to_target,
        "window_size": window_size,
        "scoring_method": scoring,
        "row_assignment": "input_instance_identity_v1",
    }
    if effective["max_tracks"] is not None or effective["target_instance_count"]:
        raise SystemExit(f"the tracker is still capped: {effective}")

    # `sio.Skeleton` replaces the strings in the list it is given with `Node`
    # objects in place, so resolve the columns before building it.
    columns = [detections.nodes.index(name) for name in nodes]
    skeleton = sio.Skeleton(list(nodes))
    points = detections.points[:, columns, :].astype(float)
    start, stop = detections.start, detections.stop

    assigned = np.full(detections.frame_idx.shape[0], UNTRACKED, dtype=np.int64)
    names: dict[str, int] = {}
    for frame in frames:
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
        # The tracker returns matched instances followed by newly created tracks,
        # not necessarily in input order. Capture the source rows before calling
        # it: even an in-place list reorder must not change row correspondence.
        source_rows = {
            id(instance): row for row, instance in zip(range(lo, hi), batch, strict=True)
        }
        returned = tracker.track(batch, frame_idx=frame)
        returned_ids = [id(instance) for instance in returned]
        if len(returned_ids) != len(source_rows) or set(returned_ids) != set(source_rows):
            raise ValueError(
                f"Tracker did not return each input instance exactly once at frame {frame}"
            )
        for instance in returned:
            if instance.track is None:
                continue
            key = instance.track.name
            if key not in names:
                names[key] = len(names)
            assigned[source_rows[id(instance)]] = names[key]
    return assigned, effective


def slice_detections(detections, lo: int, hi: int):
    """The instances of frames ``[lo, hi)``, keeping the global frame numbering."""
    keep = (detections.frame_idx >= lo) & (detections.frame_idx < hi)
    return replace(detections, frame_idx=detections.frame_idx[keep],
                   points=detections.points[keep], score=detections.score[keep]), keep


def track_one_chunk(payload: tuple) -> dict:
    """One camera, one chunk, in its own process; the warm-up is discarded."""
    view, index, lo, hi, warm, chunk, nodes, window, scoring, min_match = payload
    started = time.time()
    labels, effective = run_tracker(
        chunk, nodes, window_size=window, scoring=scoring,
        min_match_points=min_match, frames=range(warm, hi))
    inside = chunk.frame_idx >= lo
    return {"view": view, "index": index, "lo": lo, "hi": hi,
            "labels": labels[inside], "effective": effective,
            "seconds": time.time() - started}


def stitch(chunks: list[dict], n_instances: int, keep: list[np.ndarray]) -> np.ndarray:
    """Chunk-local label ids -> one dense labelling for the whole recording.

    Ids are never carried across a boundary: a chunk knows nothing about the one
    before it, so pretending its label 0 is the previous chunk's label 0 would
    be an assertion, not a measurement. Each chunk's ids are offset instead, and
    the boundary shows up as what it is — a handover.
    """
    out = np.full(n_instances, UNTRACKED, np.int64)
    offset = 0
    for chunk, mask in zip(sorted(chunks, key=lambda c: c["index"]), keep, strict=True):
        labels = chunk["labels"]
        live = labels >= 0
        out[np.flatnonzero(mask)[live]] = labels[live] + offset
        offset += int(labels.max()) + 1 if live.any() else 0
    live = out >= 0
    out[live] = np.unique(out[live], return_inverse=True)[1]
    return out


def presence(detections, labels: np.ndarray) -> np.ndarray:
    """(n_frames, n_labels) True where the frame holds a detection with that label."""
    live = labels >= 0
    n_labels = int(labels.max()) + 1 if live.any() else 0
    out = np.zeros((detections.n_frames, n_labels), bool)
    out[detections.frame_idx[live], labels[live]] = True
    return out


def maximal_states(k: np.ndarray) -> np.ndarray:
    """How many ways one view's ``k`` present labels fill three slots, injectively.

    Every present label is used — a slot is left empty only when the frame is
    short of labels. That is the state space `c25` enumerates, and this is the
    number that decides whether it is affordable: ``k = 3`` gives 6, which is
    exactly `c19`'s six permutations per view.
    """
    table = np.array([perm(N_ANIMALS, j) if j < N_ANIMALS else perm(int(j), N_ANIMALS)
                      for j in range(int(k.max()) + 1)], dtype=np.int64)
    return table[k]


def label_census(detections, labels: np.ndarray, present: np.ndarray) -> dict:
    """Everything about the labelling that the next stage's state space rests on."""
    n_frames = detections.n_frames
    live = labels >= 0
    k = present.sum(axis=1)
    counts = detections.counts()

    lifetime = []
    for label in range(present.shape[1]):
        frames = np.flatnonzero(present[:, label])
        gaps = np.diff(frames)
        lifetime.append({
            "label": label,
            "frames_present": int(frames.size),
            "first": int(frames[0]), "last": int(frames[-1]),
            "longest_gap": int(gaps.max() - 1) if gaps.size else 0,
            "share": float(frames.size / n_frames),
        })

    # A birth or a death: the set of present labels differs from the frame
    # before. That is a handover where the count did not change, and a genuine
    # appearance or loss where it did.
    changed = (present[1:] != present[:-1]).any(axis=1)
    same_count = k[1:] == k[:-1]
    return {
        "labels_created": int(present.shape[1]),
        "labels_over_one_second": int(sum(x["frames_present"] >= FPS for x in lifetime)),
        "instances": int(labels.size),
        "unlabelled_instances": int((~live).sum()),
        "labels_per_frame": {int(a): int(b) for a, b in
                             zip(*np.unique(k, return_counts=True), strict=True)},
        "detections_per_frame": {int(a): int(b) for a, b in
                                 zip(*np.unique(counts, return_counts=True), strict=True)},
        "frames_with_four_or_more_labels": float((k >= 4).mean()),
        "label_set_changes": float(changed.mean()),
        "handovers": float((changed & same_count).mean()),
        "lifetime": lifetime,
    }


def reflection_check(detections, labels: np.ndarray) -> dict:
    """Does the real third fish carry a label through the reflection scene?"""
    start, stop = detections.start, detections.stop
    rows = []
    for frame in range(*REFLECTION):
        lo, hi = int(start[frame]), int(stop[frame])
        for row in range(lo, hi):
            mouth = detections.node("mouth")[row]
            rows.append({"frame": frame, "row": row - lo,
                         "mouth": [float(mouth[0]), float(mouth[1])],
                         "label": int(labels[row])})
    return {"frames": list(REFLECTION), "instances": rows,
            "unlabelled": int(sum(r["label"] == UNTRACKED for r in rows))}


def partition_key(detections, labels: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Which instances share a label, expressed without using the ids themselves.

    Two runs of the tracker number their labels differently even when they agree
    completely, so ids cannot be compared. What can be compared is the partition
    they induce: for each instance, the id of the *first* instance carrying its
    label within the compared range.
    """
    keep = np.isin(detections.frame_idx, frames) & (labels >= 0)
    rows = np.flatnonzero(keep)
    first: dict[int, int] = {}
    out = np.full(rows.size, -1, np.int64)
    for position, row in enumerate(rows):
        label = int(labels[row])
        out[position] = first.setdefault(label, row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--nodes", nargs="+", default=["mouth", "pectoral"])
    parser.add_argument("--scoring", default="oks", choices=("oks", "euclidean_dist"))
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-match-points", type=int, default=0)
    parser.add_argument("--views", nargs="+", default=list(VIEWS))
    parser.add_argument("--chunk", type=int, default=12_000,
                        help="frames per parallel chunk")
    parser.add_argument("--warmup", type=int, default=500,
                        help="frames tracked before a chunk starts, then discarded")
    parser.add_argument("--workers", type=int, default=min(30, os.cpu_count() or 1))
    parser.add_argument("--check-chunking", action="store_true")
    args = parser.parse_args()

    paths = load_paths()
    tag = "-".join(args.nodes) + f"_{args.scoring}"
    out_dir = paths.work_dir / "c24_uncapped_tracklets" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = tuple(args.nodes)

    with provenance("uncapped_tracklets", out_dir, name="c24") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        print(f"{EXPERIMENT}   nodes {args.nodes}   scoring {args.scoring}   "
              f"window {args.window_size}")
        print(f"chunk {args.chunk:,} frames   warm-up {args.warmup}   "
              f"{args.workers} workers   n_groups = 1, n_experiments = 1\n")

        loaded = {view: c01.load(paths.work_dir, EXPERIMENT, view, RUN)
                  for view in args.views}

        if args.check_chunking:
            view = args.views[0]
            detections = loaded[view]
            lo, hi = args.chunk, 2 * args.chunk
            frames = np.arange(lo, hi)
            runs = {}
            for warmup in (args.warmup, 4 * args.warmup):
                chunk, keep = slice_detections(detections, lo - warmup, hi)
                labels, _ = run_tracker(chunk, nodes, window_size=args.window_size,
                                        scoring=args.scoring,
                                        min_match_points=args.min_match_points,
                                        frames=range(lo - warmup, hi))
                full = np.full(detections.frame_idx.size, UNTRACKED, np.int64)
                full[keep] = labels
                runs[warmup] = partition_key(detections, full, frames)
            differ = int((runs[args.warmup] != runs[4 * args.warmup]).sum())
            total = runs[args.warmup].size
            print(f"chunking check on {view} {lo:,}-{hi:,}: warm-up {args.warmup} vs "
                  f"{4 * args.warmup} frames -> {differ:,} of {total:,} instances "
                  f"({differ / total:.4%}) land in a different partition")
            record.param("chunking_check", {"view": view, "frames": [lo, hi],
                                            "instances": total, "differ": differ})
            if differ:
                raise SystemExit("the warm-up is too short; the chunks are not equivalent")
            print("passed — the warm-up is long enough for the chunks to be equivalent")
            return

        payloads, keeps = [], {}
        for view in args.views:
            detections = loaded[view]
            keeps[view] = []
            for index, lo in enumerate(range(0, detections.n_frames, args.chunk)):
                hi = min(lo + args.chunk, detections.n_frames)
                warm = max(0, lo - args.warmup)
                chunk, keep = slice_detections(detections, warm, hi)
                keeps[view].append(keep & (detections.frame_idx >= lo))
                payloads.append((view, index, lo, hi, warm, chunk, nodes,
                                 args.window_size, args.scoring, args.min_match_points))
        print(f"{len(payloads)} chunks over {len(args.views)} cameras")

        started = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            done = list(pool.map(track_one_chunk, payloads))
        wall = time.time() - started
        core_seconds = sum(chunk["seconds"] for chunk in done)
        print(f"tracked in {wall:.0f}s wall clock, {core_seconds:.0f}s of core time "
              f"({core_seconds / wall:.1f}x)\n")
        record.param("wall_seconds", wall)
        record.param("core_seconds", core_seconds)

        census, states, effective = {}, {}, {}
        for view in args.views:
            detections = loaded[view]
            mine = [chunk for chunk in done if chunk["view"] == view]
            labels = stitch(mine, detections.frame_idx.size, keeps[view])
            effective[view] = mine[0]["effective"]
            here = presence(detections, labels)
            census[view] = label_census(detections, labels, here)
            census[view]["chunks"] = len(mine)
            census[view]["boundary_renames"] = (len(mine) - 1) * N_ANIMALS
            states[view] = maximal_states(here.sum(axis=1))

            np.save(out_dir / f"track2d__{view}.npy", labels)
            record.output(f"track2d__{view}", out_dir / f"track2d__{view}.npy")

            capped_path = paths.work_dir / CAPPED_2D / f"track2d__{view}.npy"
            if capped_path.is_file():
                capped = np.load(capped_path)
                census[view]["capped_unlabelled"] = int((capped == UNTRACKED).sum())
                census[view]["rescued_by_uncapping"] = int(
                    ((capped == UNTRACKED) & (labels != UNTRACKED)).sum())

            cell = census[view]
            print(f"  {view:<10} labels {cell['labels_created']:>4} "
                  f"(>=1s {cell['labels_over_one_second']:>3}, "
                  f"{cell['boundary_renames']} from chunk boundaries)  "
                  f"unlabelled {cell['unlabelled_instances']:>6,}  "
                  f">=4 labels {cell['frames_with_four_or_more_labels']:6.3%}  "
                  f"handovers {cell['handovers']:6.3%}")
            if "capped_unlabelled" in cell:
                print(f"             c16 left {cell['capped_unlabelled']:>6,} instances "
                      f"unlabelled; this labels {cell['rescued_by_uncapping']:>6,} of them")
            if view == REFLECTION_VIEW:
                scene = reflection_check(detections, labels)
                record.param("reflection_scene", scene)
                print(f"             reflection scene {REFLECTION[0]:,}-{REFLECTION[1]:,}: "
                      f"{scene['unlabelled']} unlabelled instances "
                      f"(c16 left the real fish unlabelled for 10 frames)")

        # The number `c25` is affordable or not by: ordered states per frame,
        # the product over views, with no quotient for slot naming because the
        # slots carry identity between frames and so cannot be renamed freely.
        if len(states) == len(VIEWS):
            per_frame = np.prod(np.stack([states[v] for v in VIEWS]), axis=0)
            summary = {
                "mean": float(per_frame.mean()), "median": float(np.median(per_frame)),
                "p99_9": float(np.percentile(per_frame, 99.9)),
                "max": int(per_frame.max()),
                "frames_over_1000": float((per_frame > 1000).mean()),
                "c19_constant": 6 ** len(VIEWS),
            }
            record.param("states_per_frame", summary)
            print(f"\nordered states per frame  mean {summary['mean']:.0f}  "
                  f"median {summary['median']:.0f}  p99.9 {summary['p99_9']:.0f}  "
                  f"max {summary['max']:,}   (c19 is a constant {summary['c19_constant']})")

        record.param("per_view", {v: {k: x for k, x in c.items() if k != "lifetime"}
                                  for v, c in census.items()})
        record.param("effective_tracker", effective)
        (out_dir / "c24_summary.json").write_text(
            json.dumps({"experiment": EXPERIMENT, "n_groups": 1, "n_experiments": 1,
                        "groups": [ExperimentId.parse(EXPERIMENT).group],
                        "nodes": args.nodes, "scoring": args.scoring,
                        "chunk": args.chunk, "warmup": args.warmup,
                        "effective_tracker": effective, "per_view": census},
                       indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

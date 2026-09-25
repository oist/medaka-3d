"""Crop-replay the installed C27 tracker, preserving legacy and faithful row reads.

This is a diagnostic of history and serialization, not replacement tracking.
All original full-recording C27 arrays remain immutable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from scipy.optimize import linear_sum_assignment

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
CASES = (
    "cg20s",
    "known_21232",
    "known_30735",
    "known_31479",
    "known_105541",
    "known_260434",
    "known_302762",
    "known_333619",
    "known_323819",
)


def edge_set(frames, labels, global_rows, start, stop):
    previous, edges = {}, set()
    for row, (frame, label) in enumerate(zip(frames, labels, strict=True)):
        if label < 0:
            continue
        before = previous.get(int(label))
        if before is not None and frames[before] + 1 == frame and start <= frame < stop:
            edges.add((int(global_rows[before]), int(global_rows[row])))
        previous[int(label)] = row
    return edges


def compare(frames, original, replay, global_rows, start, stop):
    reference_edges = edge_set(frames, original, global_rows, start, stop)
    replay_edges = edge_set(frames, replay, global_rows, start, stop)
    common = reference_edges & replay_edges
    core = (frames >= start) & (frames < stop)
    keep = core & (original >= 0) & (replay >= 0)
    _, x = np.unique(original[keep], return_inverse=True)
    _, y = np.unique(replay[keep], return_inverse=True)
    matrix = np.zeros((int(x.max(initial=-1) + 1), int(y.max(initial=-1) + 1)), dtype=int)
    np.add.at(matrix, (x, y), 1)
    a, b = linear_sum_assignment(-matrix)
    matched = int(matrix[a, b].sum())
    return dict(
        core_rows=int(core.sum()),
        full_serial_unique_ids=int(np.unique(original[core & (original >= 0)]).size),
        cropped_unique_ids=int(np.unique(replay[core & (replay >= 0)]).size),
        unassigned_cropped_rows=int((core & (replay < 0)).sum()),
        full_serial_adjacent_edges=len(reference_edges),
        cropped_adjacent_edges=len(replay_edges),
        common_adjacent_edges=len(common),
        adjacent_edge_jaccard=len(common) / len(reference_edges | replay_edges)
        if reference_edges | replay_edges
        else None,
        reference_edges_missing_in_crop=len(reference_edges - replay_edges),
        crop_edges_absent_from_reference=len(replay_edges - reference_edges),
        matched_rows_after_one_global_id_mapping=matched,
        exact_core_partition_up_to_one_global_mapping=matched == int(core.sum()),
        mapping_scope="diagnostic renaming over core only; not biological identity accuracy",
        edge_scope="adjacent native links whose endpoint is in core; includes context-to-core edge",
    )


def replay(frames, points, scores, start, stop):
    import sleap_io as sio
    from sleap_nn.tracking.tracker import Tracker

    tracker = Tracker.from_config(
        window_size=5,
        candidates_method="local_queues",
        features="keypoints",
        scoring_method="oks",
        scoring_reduction="mean",
        track_matching_method="hungarian",
        max_tracks=None,
        tracking_target_instance_count=None,
        min_match_points=0,
    )
    skeleton = sio.Skeleton(["mouth", "pectoral"])
    legacy = np.full(len(frames), -1, dtype=np.int64)
    faithful = legacy.copy()
    names, reordered = {}, []
    for frame in range(start, stop):
        lo, hi = np.searchsorted(frames, [frame, frame + 1])
        if lo == hi:
            continue
        batch = [
            sio.PredictedInstance.from_numpy(
                points_data=points[row, :2].astype(float),
                skeleton=skeleton,
                point_scores=np.ones(2),
                score=float(scores[row]),
            )
            for row in range(lo, hi)
        ]
        original_row = {id(instance): row for row, instance in enumerate(batch)}
        returned = tracker.track(batch, frame_idx=frame)
        order = []
        for offset, instance in enumerate(returned):
            source_row = original_row[id(instance)]
            order.append(source_row)
            if instance.track is None:
                continue
            name = instance.track.name
            if name not in names:
                names[name] = len(names)
            legacy[lo + offset] = names[name]
            faithful[lo + source_row] = names[name]
        if order != list(range(len(batch))):
            reordered.append(
                dict(frame=int(frame), input_rows=int(hi - lo), returned_input_row_order=order)
            )
    effective = dict(
        tracker_type=type(tracker).__name__,
        candidate_type=type(tracker.candidate).__name__,
        min_match_points=tracker.min_match_points,
        min_new_track_points=tracker.candidate.min_new_track_points,
        window_size=tracker.candidate.window_size,
        max_tracks=tracker.candidate.max_tracks,
        scoring_method=tracker.scoring_method,
        scoring_reduction=tracker.scoring_reduction,
        oks_stddev=tracker.oks_stddev,
        use_flow=tracker.use_flow,
    )
    return legacy, faithful, reordered, effective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("work/clean2d_video_comparison_v1/c27_replay")
    )
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    protocol_path = Path("work/custom2d_comparison_v1/analysis_v3/protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    selected = [c for c in protocol["cases"] if c["name"] in (args.case or CASES)]
    if set(args.case) - set(CASES):
        parser.error("Unknown replay case")
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = (
        Path("work/custom2d_comparison_v1/inputs/c27_uncapped_tracklets_any")
        / EXPERIMENT
        / "mouth-pectoral_oks_serial"
    )
    cached = Path("work/generative_tracking_inputs/work/c01_s16_predictions/bottomup")
    records = []
    with provenance("local_c27_replay_diagnostic", args.output) as rec:
        rec.params_from(args)
        rec.input("code", Path(__file__), sha256=sha256(Path(__file__)))
        rec.input("ranges", protocol_path, sha256=sha256(protocol_path))
        raw, saved = {}, {}
        for view in VIEWS:
            path = cached / f"{EXPERIMENT}__{view}.npz"
            with np.load(path, allow_pickle=False) as z:
                raw[view] = {k: z[k] for k in ("frame_idx", "points", "score")}
            rec.input("untracked_" + view, path, sha256=sha256(path))
            path = baseline / f"track2d__{view}.npy"
            saved[view] = np.load(path, allow_pickle=False)
            rec.input("saved_full_serial_" + view, path, sha256=sha256(path))
        # Record the actually installed implementation; the old provenance did
        # not pin package file hashes, so this cannot certify historical replay.
        import sleap_nn.tracking.tracker as installed

        package = Path(installed.__file__).parent
        for relative in ("tracker.py", "candidates/local_queues.py", "utils.py"):
            path = package / relative
            rec.input("installed_" + relative, path, sha256=sha256(path))
        for case in selected:
            start, stop = case["start"] - 332, case["stop"] + 332
            dest = args.output / case["name"]
            dest.mkdir(exist_ok=True)
            for view in VIEWS:
                output = dest / f"{view}.npz"
                report = dest / f"{view}.json"
                if output.exists() or report.exists():
                    raise ValueError(f"Refusing to replace existing replay: {output}")
                data = raw[view]
                lo, hi = np.searchsorted(data["frame_idx"], [start, stop])
                frames, points, scores = (data[k][lo:hi] for k in ("frame_idx", "points", "score"))
                began = time.perf_counter()
                legacy, faithful, reordered, effective = replay(frames, points, scores, start, stop)
                global_rows = np.arange(lo, hi)
                original = saved[view][lo:hi]
                np.savez_compressed(
                    output,
                    frame_idx=frames,
                    global_rows=global_rows,
                    legacy_labels=legacy,
                    row_faithful_labels=faithful,
                    saved_full_serial_labels=original,
                )
                core = (frames >= case["start"]) & (frames < case["stop"])
                result = dict(
                    case=case["name"],
                    view=view,
                    core_half_open=[case["start"], case["stop"]],
                    context_half_open=[start, stop],
                    effective_config=effective,
                    legacy_vs_saved=compare(
                        frames, original, legacy, global_rows, case["start"], case["stop"]
                    ),
                    row_faithful_vs_saved=compare(
                        frames, original, faithful, global_rows, case["start"], case["stop"]
                    ),
                    reordered_returns=reordered,
                    readout_different_rows_context=int((legacy != faithful).sum()),
                    readout_different_rows_core=int(((legacy != faithful) & core).sum()),
                    elapsed_seconds=time.perf_counter() - began,
                    historical_reproduction_claim=False,
                    limitations=[
                        "Cropped history differs from full serial history.",
                        "Historical package file hashes were not recorded.",
                        "Faithful row readout changes serialization only, not tracker state.",
                    ],
                )
                report.write_text(
                    json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
                rec.output(case["name"] + "_" + view, output, sha256=sha256(output))
                rec.output(case["name"] + "_" + view + "_report", report, sha256=sha256(report))
                records.append(result)
                print(
                    json.dumps(
                        {
                            "case": case["name"],
                            "view": view,
                            "seconds": round(result["elapsed_seconds"], 2),
                            "legacy_edge_jaccard": result["legacy_vs_saved"][
                                "adjacent_edge_jaccard"
                            ],
                            "row_readout_differences_core": result["readout_different_rows_core"],
                        }
                    ),
                    flush=True,
                )
        target = args.output / "summary.json"
        target.write_text(
            json.dumps(
                dict(
                    status="complete",
                    records=records,
                    scope="cropped current-installation diagnostic",
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        rec.output("summary", target, sha256=sha256(target))


if __name__ == "__main__":
    main()

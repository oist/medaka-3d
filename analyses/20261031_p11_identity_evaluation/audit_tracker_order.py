"""Observe the existing tracker's return-order contract on a short real prefix.

This audit returns the library output unchanged. It does not repair or replace
the C24/C27 tracker and does not estimate full-recording identity performance.
"""

import argparse
import importlib
from pathlib import Path

import numpy as np
from run_pipeline import settings, sha256, write_json

from medaka3d.provenance import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument(
        "--view", choices=["all", "top_cam", "side1_cam", "side2_cam"], default="all"
    )
    args = parser.parse_args()
    cfg, _, _, output = settings()
    suffix = "" if args.view == "all" else f"_{args.view}"
    destination = output / f"tracker_order_audit_{args.frames}frames{suffix}"
    destination.mkdir(exist_ok=False)
    with np.load(output / "native_timing.npz") as timing:
        first_scored = int(
            np.flatnonzero(timing["top_cam"] >= cfg["evaluation"]["preparation_seconds"])[0]
        )
    c24 = importlib.import_module("c24_uncapped_tracklets")
    c01 = importlib.import_module("c01_s16_predictions")
    from sleap_nn.tracking.tracker import Tracker

    original = Tracker.track
    results = {}
    active = None

    def observed_track(self, instances, *a, **kw):
        returned = original(self, instances, *a, **kw)
        input_row = {id(instance): i for i, instance in enumerate(instances)}
        order = [input_row[id(instance)] for instance in returned]
        active["frames_with_detections"] += 1
        active["returned_instances"] += len(returned)
        frame = int(kw["frame_idx"])
        in_window = frame >= first_scored
        active["scored_frames_with_detections"] += int(in_window)
        mismatch = [i for i, row in enumerate(order) if i != row]
        if mismatch:
            active["reordered_frames"] += 1
            active["labels_written_to_wrong_input_row"] += len(mismatch)
            active["scored_reordered_frames"] += int(in_window)
            active["scored_labels_written_to_wrong_input_row"] += len(mismatch) * int(in_window)
            example = {
                "frame": frame,
                "input_rows_in_return_order": order,
                "returned_track_names": [instance.track.name for instance in returned],
            }
            if len(active["examples"]) < 10:
                active["examples"].append(example)
            if in_window and len(active["scored_examples"]) < 10:
                active["scored_examples"].append(example)
        return returned

    with provenance("p11_tracker_return_order_audit", destination) as rec:
        rec.input("audit_code", Path(__file__), sha256=sha256(Path(__file__)))
        rec.input("c24", Path(c24.__file__), sha256=sha256(Path(c24.__file__)))
        library = Path(importlib.import_module(Tracker.__module__).__file__)
        rec.input("tracker_library", library, sha256=sha256(library))
        rec.param("frames", [0, args.frames])
        rec.param("scope", "Return-order contract audit with continuous history from frame zero")
        rec.param("first_scored_frame", first_scored)
        Tracker.track = observed_track
        try:
            views = ["top_cam", "side1_cam", "side2_cam"] if args.view == "all" else [args.view]
            for view in views:
                data = c01.load(output / "pipeline", cfg["experiment"], view, "bottomup")
                active = {
                    "frames_with_detections": 0,
                    "returned_instances": 0,
                    "reordered_frames": 0,
                    "labels_written_to_wrong_input_row": 0,
                    "examples": [],
                    "scored_frames_with_detections": 0,
                    "scored_reordered_frames": 0,
                    "scored_labels_written_to_wrong_input_row": 0,
                    "scored_examples": [],
                }
                labels, _ = c24.run_tracker(
                    data,
                    ["mouth", "pectoral"],
                    window_size=5,
                    scoring="oks",
                    min_match_points=0,
                    frames=range(args.frames),
                )
                active["assigned_labels"] = int(np.count_nonzero(labels >= 0))
                results[view] = active
        finally:
            Tracker.track = original
        write_json(destination / "results.json", results)
        rec.output("results", destination / "results.json")
    print(results)


if __name__ == "__main__":
    main()

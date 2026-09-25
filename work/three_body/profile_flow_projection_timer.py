"""Direct CPU flow and transparent projection timing on twenty real image pairs.

No CUDA operation, cProfile, optimization, new observation or tracking run is used.
Saved backtraced poses are diagnostic inputs, not a replay of online beam[0].
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from motion import FlowConfig, FlowObservations  # noqa: E402
from observations import SequenceReader  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class CameraTimer:
    """Delegate the exact original project call and return its original object."""

    def __init__(self, camera, records):
        self._camera, self._records = camera, records

    def __getattr__(self, name):
        return getattr(self._camera, name)

    def project(self, xyz):
        start = time.perf_counter()
        result = self._camera.project(xyz)
        elapsed = time.perf_counter() - start
        caller = sys._getframe(1)
        self._records.append(dict(view=self._camera.view, input_shape=list(np.shape(xyz)),
                                  seconds=elapsed, caller=caller.f_code.co_name,
                                  caller_line=caller.f_lineno))
        return result


class RigTimer:
    def __init__(self, rig):
        self._rig, self.records = rig, []
        self._cameras = {view: CameraTimer(rig.camera(view), self.records) for view in rig.views}

    def __getattr__(self, name):
        return getattr(self._rig, name)

    def camera(self, view):
        return self._cameras[view]


def compare_flow(first, second):
    comparisons = {}
    for field in dataclasses.fields(first):
        a, b = np.asarray(getattr(first, field.name)), np.asarray(getattr(second, field.name))
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f"proxy changed schema: {field.name}")
        floating = np.issubdtype(a.dtype, np.inexact)
        exact = np.array_equal(a, b, equal_nan=True) if floating else np.array_equal(a, b)
        maximum = 0.
        if floating:
            selected = np.isfinite(a) & np.isfinite(b)
            maximum = float(np.max(np.abs(a[selected] - b[selected]), initial=0))
        comparisons[field.name] = dict(exact=bool(exact), maximum_finite_difference=maximum)
        if not exact:
            raise AssertionError(f"proxy changed output: {field.name}")
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    code = [Path(__file__), LAB / "motion.py", LAB / "observations.py",
            LAB / "prepare_sequence.py", REPO / "src/medaka3d/calib.py"]
    hashes = {str(path.relative_to(REPO)): sha256(path) for path in code}
    for path in code:
        copy = args.output / "code_snapshot" / path.relative_to(REPO)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy)
    sequence = REPO / "work/three_body/inputs/transient_333519-333719"
    forward = REPO / "work/three_body/graph_accelerated_transient_200_v1"
    with np.load(forward / "candidate.npz", allow_pickle=False) as data:
        frame_idx = data["frame_idx"]
        poses = data["pose_xyz"]
        recorded_flow_node_valid = data["proposal_flow_nodes"]
    pair_indexes = np.rint(np.linspace(1, len(frame_idx) - 1, 20)).astype(int)
    assert len(np.unique(pair_indexes)) == 20
    rig = load_rig(forward / "calibration_snapshot")
    reader = SequenceReader(sequence, REPO / "work/generative_tracking_inputs")
    np.testing.assert_array_equal(reader.frames, frame_idx)
    cfg = FlowConfig()
    rows, saved = [], defaultdict(list)
    previous = None
    iterator = iter(reader)
    try:
        for index, frame in enumerate(iterator):
            if index in pair_indexes:
                timer = RigTimer(rig)
                order = ("original", "proxy") if len(rows) % 2 == 0 else ("proxy", "original")
                output, timing = {}, {}
                for method in order:
                    start = time.perf_counter()
                    output[method] = FlowObservations.from_frames(
                        previous, frame, rig if method == "original" else timer,
                        poses[index - 1], config=cfg)
                    timing[method] = time.perf_counter() - start
                comparison = compare_flow(output["original"], output["proxy"])
                original = output["original"]
                counts = Counter(tuple(row["input_shape"]) for row in timer.records)
                by_shape = {str(shape): dict(calls=count, seconds=sum(
                    row["seconds"] for row in timer.records if tuple(row["input_shape"]) == shape))
                    for shape, count in sorted(counts.items())}
                project_seconds = sum(row["seconds"] for row in timer.records)
                row = dict(previous_frame=int(previous.frame_idx), frame=int(frame.frame_idx),
                           order=order, seconds=timing, projection_calls=len(timer.records),
                           projection_seconds=project_seconds,
                           proxy_seconds_outside_project=timing["proxy"] - project_seconds,
                           projection_fraction_of_proxy=project_seconds / timing["proxy"],
                           per_input_shape=by_shape, fields=comparison,
                           node_valid_count=int(original.node_valid.sum()),
                           node_valid_matches_recorded_online_flow=bool(np.array_equal(
                               original.node_valid, recorded_flow_node_valid[index])),
                           project_events=timer.records)
                rows.append(row)
                saved["previous_frame"].append(int(previous.frame_idx))
                saved["frame"].append(int(frame.frame_idx))
                saved["previous_pose_xyz"].append(poses[index - 1])
                for method, result in output.items():
                    for field in dataclasses.fields(result):
                        saved[method + "_" + field.name].append(getattr(result, field.name))
                print(json.dumps({key: value for key, value in row.items()
                                  if key not in ("project_events", "fields")}), flush=True)
            previous = frame
    finally:
        iterator.close()
    if len(rows) != 20:
        raise RuntimeError("did not evaluate all twenty image pairs")
    source_after = {str(path.relative_to(REPO)): sha256(path) for path in code}
    if source_after != hashes:
        raise RuntimeError("frozen source changed during CPU timing")
    arrays = args.output / "inputs_and_outputs.npz"
    np.savez_compressed(arrays, **{key: np.asarray(value) for key, value in saved.items()})
    shapes, callers = {}, {}
    for row in rows:
        for event in row["project_events"]:
            for target, key in ((shapes, str(event["input_shape"])),
                                (callers, f'{event["caller"]}:{event["caller_line"]}')):
                bucket = target.setdefault(key, dict(calls=0, seconds=0.))
                bucket["calls"] += 1
                bucket["seconds"] += event["seconds"]
    sum_project = sum(row["projection_seconds"] for row in rows)
    sum_proxy = sum(row["seconds"]["proxy"] for row in rows)
    report = dict(
        status="ok", device="CPU only", cuda_operations=False, pair_count=20,
        pair_indexes=pair_indexes.tolist(), config=dataclasses.asdict(cfg),
        all_fields_exact=True, field_count=len(dataclasses.fields(FlowObservations)),
        sum_seconds={method: sum(row["seconds"][method] for row in rows)
                     for method in ("original", "proxy")},
        median_seconds={method: float(np.median([row["seconds"][method] for row in rows]))
                        for method in ("original", "proxy")},
        projection_total_calls=sum(row["projection_calls"] for row in rows),
        projection_total_seconds=sum_project,
        projection_fraction_of_proxy_total=sum_project / sum_proxy,
        proxy_seconds_outside_projection=sum_proxy - sum_project,
        projection_by_input_shape=shapes, projection_by_caller=callers,
        node_valid_matches_online_count=sum(row["node_valid_matches_recorded_online_flow"]
                                           for row in rows),
        opencv_version=cv2.__version__, opencv_threads=cv2.getNumThreads(),
        source_hashes=hashes, source_unchanged=True,
        inputs={str(path): sha256(path) for path in (
            sequence / "manifest.json", forward / "candidate.npz",
            *(forward / "calibration_snapshot").glob("*.json"),
            *reader.nodes.paths.values())},
        input_and_output_sha256=sha256(arrays), rows=rows,
        limitations=[
            "These are saved backtraced pose inputs, not an archive of the original "
            "online beam[0].",
            "The proxy calls and returns the original camera.project without changing its math.",
            "Projection timers measure native call wall time; per-event bookkeeping is outside "
            "that inner timer but remains included in total proxy flow time.",
            "Original/proxy order alternates, but these are one call per method per pair; "
            "cache state and concurrent machine load remain descriptive limitations.",
            "No new tracking, optimization, ground-truth evaluation or GPU inference occurred.",
        ])
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                             encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("rows", "inputs", "source_hashes")}), flush=True)


if __name__ == "__main__":
    main()

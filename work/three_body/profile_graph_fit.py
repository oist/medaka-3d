"""Whole-objective and complete-Adam parity on two real frames, then paired timing."""

import argparse
import ast
import dataclasses
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from accelerated_tracker import AcceleratedThreeBodyTracker  # noqa: E402
from graph_fit import FitInputs, GraphFit  # noqa: E402
from node_geometry import RayObservations  # noqa: E402
from observations import SequenceReader, sample_pixels  # noqa: E402
from run_experiment import frozen_code_paths, registered_seed, sha256  # noqa: E402
from tracker import Proposal, State, ThreeBodyTracker, TrackerConfig  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402


def reference_scope(tracker, proposals, states, rays, targets):
    tree = ast.parse((LAB / "tracker.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ThreeBodyTracker")
    fit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_fit")
    stop = next(i for i, n in enumerate(fit.body)
                if isinstance(n, ast.FunctionDef) and n.name == "objective")
    fit.body = [*fit.body[:stop + 1], ast.Return(ast.Call(ast.Name("locals", ast.Load()), [], []))]
    fit.name = "reference_prefix"
    namespace = dict(ThreeBodyTracker._fit.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module([fit], [])), str(LAB / "tracker.py"),
                 "exec"), namespace)
    return namespace[fit.name](tracker, proposals, states, rays, targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), *frozen_code_paths(), *(LAB / name for name in (
        "graph_union.py", "graph_fit.py", "accelerated_tracker.py"))]
    hashes = {p.relative_to(REPO).as_posix(): sha256(p) for p in paths}
    snapshot = args.output / "code_snapshot"
    for path in paths:
        target = snapshot / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    sequence = REPO / "work/three_body/inputs/transient_333519-333719"
    forward = REPO / "work/three_body/v1_transient_detectron2"
    reader = SequenceReader(sequence, REPO / "work/generative_tracking_inputs")
    iterator = iter(reader)
    try:
        frames = [next(iterator), next(iterator)]
    finally:
        iterator.close()
    with np.load(forward / "candidate.npz", allow_pickle=False) as data:
        actual = {key: data[key][:2] for key in ("pose_xyz", "sources", "missing", "frame_idx")}
    np.testing.assert_array_equal(actual["frame_idx"], [f.frame_idx for f in frames])
    initial, seed, _ = registered_seed(frames[0].frame_idx)
    rig = load_rig(forward / "calibration_snapshot")
    config = TrackerConfig(flow_proposals=True)
    torch.set_num_threads(2)
    original = ThreeBodyTracker(rig, initial, config)
    accelerated = AcceleratedThreeBodyTracker(rig, initial, config)
    targets = [sample_pixels(f) for f in frames]
    rays = [RayObservations.from_points(rig, f.detections) for f in frames]
    rng = np.random.default_rng(123393)
    perturbation = rng.uniform(-.01, .01, (18, 3, 3, 3))
    perturbation[0] = 0
    output_arrays = dict(frame_idx=actual["frame_idx"],
                         poses=actual["pose_xyz"][:, None] + perturbation[None],
                         plans=actual["sources"], initial=initial,
                         pixels=np.stack([np.stack([t.pixels[v] for v in VIEWS]) for t in targets]),
                         targets=np.stack([t.target for t in targets]),
                         weights=np.stack([t.weights for t in targets]))
    records = []
    for batch in (1, 6, 18):
        graph = None
        parity, timing = [], []
        for frame in (0, 1, 0):
            prior = initial if frame == 0 else actual["pose_xyz"][frame - 1]
            states = [State(prior, np.zeros((3, 3)), actual["missing"][frame],
                            np.zeros(3, bool), 0.)]
            proposals = [Proposal(output_arrays["poses"][frame, b], prior,
                                  actual["sources"][frame], 0, "recorded_pose_perturbation")
                         for b in range(batch)]
            scope = reference_scope(original, proposals, states, rays[frame], targets[frame])
            constants = FitInputs(**{f.name: scope.get(f.name)
                                     for f in dataclasses.fields(FitInputs)})
            if graph is None:
                graph = GraphFit(config, scope["pose"], constants, scope["renderer"])
            else:
                graph.configure(constants, scope["renderer"])
            expected_score, expected_parts, _ = scope["objective"](scope["pose"])
            gradient, = torch.autograd.grad(expected_score.sum(), scope["pose"])
            result = graph.evaluate(scope["pose"])
            differences = {}
            for name, a, b in zip(("objective", "parts", "gradient"),
                                 (result.scores, result.parts, result.pose_gradient),
                                 (expected_score, expected_parts, gradient), strict=True):
                differences[name] = float((a - b).abs().max())
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            fits = {}
            for method, tracker in (("original", original), ("captured", accelerated)):
                fits[method] = tracker._fit(proposals, states, rays[frame], targets[frame])
            for name, a, b in zip(("pose", "score", "parts", "mask_support"),
                                 fits["captured"], fits["original"], strict=True):
                np.testing.assert_array_equal(a, b)
                differences["fit_" + name] = (int(np.count_nonzero(a != b)) if a.dtype == bool
                                               else float(np.max(np.abs(a - b))))
                output_arrays[f"B{batch}_frame{frame}_fit_{name}"] = a
            parity.append(dict(frame=int(frames[frame].frame_idx), differences=differences))
            if frame == 1:
                for repeat in range(args.repeats):
                    methods = (("original", original), ("captured", accelerated))
                    if repeat % 2:
                        methods = methods[::-1]
                    for name, tracker in methods:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        tracker._fit(proposals, states, rays[frame], targets[frame])
                        torch.cuda.synchronize()
                        timing.append(dict(repeat=repeat, method=name,
                                           milliseconds=1000 * (time.perf_counter() - start)))
        records.append(dict(batch=batch, parity=parity, timing=timing,
                            median_ms={name: float(np.median([r["milliseconds"] for r in timing
                                                            if r["method"] == name]))
                                       for name in ("original", "captured")}))
        print(json.dumps(records[-1]), flush=True)
    np.savez_compressed(args.output / "samples_and_fits.npz", **output_arrays)
    after = {p.relative_to(REPO).as_posix(): sha256(p) for p in paths}
    if after != hashes:
        raise RuntimeError("code changed during fitter parity run")
    source = (LAB / "tracker.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_fit")
    source_segment = ast.get_source_segment(source, method)
    report = dict(status="ok", source_hashes=hashes,
                  original_fit_method_sha256=hashlib.sha256(source_segment.encode()).hexdigest(),
                  config=dataclasses.asdict(config), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(), torch_cpu_threads=torch.get_num_threads(),
                  seed=seed, records=records, all_values_and_gradients_exact=True,
                  limitations=["Concurrent D2 staging: timing is descriptive, not isolated.",
                               "Selected archived poses with perturbations, not archived beams.",
                               "Numerical equivalence is not fish identity validation."],
                  inputs={str(p): sha256(p) for p in (
                      sequence / "manifest.json", forward / "candidate.npz",
                      *(forward / "calibration_snapshot").glob("*.json"),
                      *reader.nodes.paths.values())},
                  sample_sha256=sha256(args.output / "samples_and_fits.npz"))
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()

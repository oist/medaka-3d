"""CPU admission batching benchmark on preserved development-frame outputs only."""

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from node_geometry import RayObservations, supported_nodes  # noqa: E402
from observations import CachedNodes  # noqa: E402
from support_batch import supported_nodes_batch  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), LAB / "support_batch.py", LAB / "node_geometry.py",
               LAB / "observations.py", LAB / "prepare_sequence.py", REPO / "src/medaka3d/calib.py"]
    source_hashes = {str(p.relative_to(REPO)): digest(p) for p in sources}
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshot / path.name)
    frame = 333544
    roots = [REPO / f"work/three_body/v0_transient_{arm}"
             for arm in ("nodes", "detectron2", "top_contrast")]
    rig = load_rig(roots[0] / "calibration_snapshot")
    inputs, poses, plans = {}, [], []
    for root in roots:
        path = root / "candidate.npz"
        inputs[str(path.relative_to(REPO))] = digest(path)
        with np.load(path, allow_pickle=False) as data:
            t = np.flatnonzero(data["frame_idx"] == frame)
            if len(t) != 1:
                raise ValueError("expected exactly one frame in each preserved output")
            poses.append(data["pose_xyz"][t[0]])
            plans.append(data["sources"][t[0]])
    for path in (roots[0] / "calibration_snapshot").glob("*.json"):
        inputs[str(path.relative_to(REPO))] = digest(path)
    cache = CachedNodes(REPO / "work/generative_tracking_inputs", "20251225_K_1_2_FCMC",
                        np.array([frame]))
    points = {view: cache.at(view, frame) for view in VIEWS}
    for path in cache.paths.values():
        inputs[str(path.relative_to(REPO))] = digest(path)
    rays = RayObservations.from_points(rig, points)
    original_poses, original_plans = np.stack(poses), np.stack(plans)
    sample = args.output / "samples.npz"
    np.savez_compressed(sample, frame_idx=frame, poses=original_poses, plans=original_plans,
                        **{f"pixels_{v}": points[v] for v in VIEWS})
    rng = np.random.default_rng(933544)
    cases = [
        ("three_actual_states", original_poses, original_plans),
        ("six_repeated_actual_states", np.tile(original_poses, (2, 1, 1, 1)),
         np.tile(original_plans, (2, 1, 1, 1))),
        ("six_perturbed_actual_states", np.tile(original_poses, (2, 1, 1, 1))
         + rng.uniform(-0.01, 0.01, (6, 3, 3, 3)),
         np.tile(original_plans, (2, 1, 1, 1))),
    ]
    results = []
    for name, poses, plans in cases:
        def evaluate(method, poses=poses, plans=plans):
            if method == "batch":
                return supported_nodes_batch(poses, plans, rays, rig)
            return tuple(np.stack(values) for values in zip(*[
                supported_nodes(p, s, rays, rig) for p, s in zip(poses, plans, strict=True)
            ], strict=True))

        reference, actual = evaluate("separate"), evaluate("batch")
        for got, want in zip(actual, reference, strict=True):
            np.testing.assert_array_equal(got, want)
        maximum_pixel_difference = 0.
        for view in VIEWS:
            cam = rig.camera(view)
            uv = cam.project(poses)
            separate_uv = np.stack([cam.project(p) for p in poses])
            np.testing.assert_array_equal(np.isfinite(uv), np.isfinite(separate_uv))
            maximum_pixel_difference = max(
                maximum_pixel_difference, float(np.nanmax(np.abs(uv - separate_uv))))
        timings = {"separate": [], "batch": []}
        for _ in range(5):
            evaluate("separate")
            evaluate("batch")
        for repeat in range(args.repeats):
            for method in (("separate", "batch") if repeat % 2 == 0 else ("batch", "separate")):
                start = time.perf_counter_ns()
                evaluate(method)
                timings[method].append((time.perf_counter_ns() - start) / 1e6)
        separate_median = float(np.median(timings["separate"]))
        batch_median = float(np.median(timings["batch"]))
        results.append({
            "case": name, "batch": len(poses), "sources_exact": True,
            "observed_exact": True, "distance_exact": True,
            "max_distance_difference_mm": float(np.nanmax(np.abs(actual[2] - reference[2]))),
            "max_projection_difference_px": maximum_pixel_difference,
            "observed_nodes": int(actual[1].sum()),
            "separate_median_ms": separate_median, "batch_median_ms": batch_median,
            "paired_median_speedup": float(np.median(
                np.asarray(timings["separate"]) / np.asarray(timings["batch"]))),
            "milliseconds_saved_per_call": separate_median - batch_median,
            "milliseconds": timings,
        })
    after = {str(p.relative_to(REPO)): digest(p) for p in sources}
    if after != source_hashes:
        raise RuntimeError("source changed while benchmarking")
    report = {
        "frame": frame, "device": "CPU only", "python": sys.version,
        "platform": platform.platform(), "processor": platform.processor(),
        "numpy": np.__version__, "repeats": args.repeats, "warmup": 5,
        "source_sha256": source_hashes, "input_sha256": inputs,
        "samples_sha256": digest(sample), "results": results,
        "limitations": [
            "Six-state timing repeats three real selected states; "
            "these are not archived beam states.",
            "Plans are already filtered preserved sources and use "
            "this frame's original untracked rows.",
            "Microbenchmark excludes ray creation, source loading, tracking, GPU and evaluation.",
            "A concurrent GPU tracking job may create CPU scheduling noise; "
            "timings are descriptive.",
            "Batch projection may differ in last bits; no threshold tolerance is added.",
            "No baseline or identity labels are read; no tracking-quality claim follows.",
        ],
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    printed = [{k: v for k, v in r.items() if k != "milliseconds"} for r in results]
    print(json.dumps(printed, indent=2))


if __name__ == "__main__":
    main()

"""CPU-only 6x18 geometric identity-comparison parity/timing on assembled v1 poses."""

import argparse
import ast
import hashlib
import json
import platform
import shutil
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from identity_batch import different_identity_batch, identity_costs_batch  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_method():
    tree = ast.parse((LAB / "tracker.py").read_text(encoding="utf-8"))
    body = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "ThreeBodyTracker")
    method = next(node for node in body.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_different_identity")
    method.decorator_list = []
    scope = {"np": np, "permutations": permutations}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(LAB / "tracker.py"), "exec"),
         scope)
    return scope["_different_identity"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=333544)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    files = [LAB / "identity_batch.py", LAB / "tracker.py", Path(__file__)]
    hashes = {str(p.relative_to(REPO)): digest(p) for p in files}
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for path in files:
        shutil.copyfile(path, snapshot / path.name)
    selected, inputs = [], {}
    for mode in ("detectron2", "top_contrast"):
        path = REPO / f"work/three_body/v1_transient_{mode}/candidate.npz"
        inputs[str(path.relative_to(REPO))] = digest(path)
        with np.load(path, allow_pickle=False) as data:
            index = np.flatnonzero(data["frame_idx"] == args.frame)
            if len(index) != 1:
                raise ValueError("frame absent or duplicated in preserved forward result")
            selected.append(data["pose_xyz"][index[0]])
    order = np.asarray(list(permutations(range(3))))
    first = selected[0][order]
    rng = np.random.default_rng(934544)
    second = np.concatenate((selected[0][order], selected[1][order],
                             selected[0][order] + rng.uniform(-0.01, 0.01, first.shape)))
    sample = args.output / "samples.npz"
    np.savez_compressed(sample, frame_idx=args.frame, first=first, second=second,
                        actual_selected=np.stack(selected))
    original = original_method()

    def evaluate(method):
        if method == "batch":
            return different_identity_batch(first, second)
        return np.stack([np.stack([original(a, b) for b in second]) for a in first])

    expected, actual = evaluate("separate"), evaluate("batch")
    np.testing.assert_array_equal(actual, expected)
    scalar_costs = np.asarray([[
        [sum(np.linalg.norm(a[f, 1] - b[p[f], 1]) ** 2 for f in range(3))
         for p in permutations(range(3))] for b in second] for a in first])
    batch_costs = identity_costs_batch(first, second)
    np.testing.assert_array_equal(batch_costs, scalar_costs)
    for _ in range(10):
        evaluate("separate")
        evaluate("batch")
    timing = {"separate": [], "batch": []}
    for repeat in range(args.repeats):
        for method in (("separate", "batch") if repeat % 2 == 0 else ("batch", "separate")):
            started = time.perf_counter_ns()
            evaluate(method)
            timing[method].append((time.perf_counter_ns() - started) / 1e6)
    medians = {key: float(np.median(values)) for key, values in timing.items()}
    if {str(p.relative_to(REPO)): digest(p) for p in files} != hashes:
        raise RuntimeError("source changed during CPU benchmark")
    record = dict(
        status="ok", frame=args.frame, pairs=[6, 18], device="CPU only", numpy=np.__version__,
        python=sys.version, platform=platform.platform(), source_sha256=hashes,
        input_sha256=inputs, samples_sha256=digest(sample), warmup=10, repeats=args.repeats,
        affected_bitwise_exact=actual.tobytes() == expected.tobytes(),
        costs_bitwise_exact=batch_costs.tobytes() == scalar_costs.tobytes(),
        maximum_absolute_cost_difference=float(np.max(np.abs(batch_costs - scalar_costs))),
        milliseconds=timing, median_milliseconds=medians,
        median_paired_speedup=float(np.median(
            np.asarray(timing["separate"]) / np.asarray(timing["batch"]))),
        estimated_milliseconds_saved_per_6x18_comparison=medians["separate"] - medians["batch"],
        construction="First: six fish permutations of v1 D2 selected pose. Second: six "
                     "permutations from each of D2 and top-contrast, plus six D2 poses "
                     "with deterministic +/-0.01 mm perturbations. All use one frame.",
        limits=["These arrays are assembled inputs, not archived actual beam histories.",
                "This is the identity-comparison microbenchmark, not end-to-end tracking.",
                "Concurrent CPU preparation may affect timings; calls alternate method order.",
                "The frozen lexical result at coincident pectorals is preserved, not resolved.",
                "No images, identity labels or baseline trajectories enter this measurement."],
    )
    (args.output / "results.json").write_text(json.dumps(record, indent=2, allow_nan=False),
                                             encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items()
                      if key not in ("milliseconds", "input_sha256", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()

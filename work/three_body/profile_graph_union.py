"""Short CUDA Graph parity/timing experiment; concurrent staging makes timing descriptive."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from fast_union import render_union  # noqa: E402
from graph_union import PROD_BACKWARD_ORIGIN, GraphUnion  # noqa: E402
from observations import SequenceReader, sample_pixels  # noqa: E402
from renderer import RefractiveFishRenderer  # noqa: E402

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
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), *(LAB / name for name in (
        "graph_union.py", "fast_union.py", "renderer.py", "observations.py",
        "prepare_sequence.py")), REPO / "src/medaka3d/calib.py"]
    hashes = {p.relative_to(REPO).as_posix(): digest(p) for p in paths}
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for path in paths:
        shutil.copyfile(path, snapshot / path.name)
    sequence = REPO / "work/three_body/inputs/transient_333519-333719"
    forward_run = REPO / "work/three_body/v1_transient_detectron2"
    reader = SequenceReader(sequence, REPO / "work/generative_tracking_inputs")
    iterator = iter(reader)
    try:
        frames = [next(iterator), next(iterator)]
    finally:
        iterator.close()
    targets = [sample_pixels(frame) for frame in frames]
    with np.load(forward_run / "candidate.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["frame_idx"][:2], [f.frame_idx for f in frames])
        actual_poses = data["pose_xyz"][:2]
    rng = np.random.default_rng(933521)
    perturbation = rng.uniform(-0.01, 0.01, (18, 3, 3, 3))
    perturbation[0] = 0
    poses_np = actual_poses[:, None] + perturbation[None]
    sample = args.output / "samples.npz"
    np.savez_compressed(sample, frame_idx=[f.frame_idx for f in frames], poses=poses_np,
                        pixels=np.stack([np.stack([t.pixels[v] for v in VIEWS]) for t in targets]),
                        targets=np.stack([t.target for t in targets]),
                        weights=np.stack([t.weights for t in targets]))
    rig = load_rig(forward_run / "calibration_snapshot")
    renderers = [RefractiveFishRenderer(rig, t.pixels, device="cuda", dtype=torch.float64)
                 for t in targets]
    poses = torch.tensor(poses_np, device="cuda", dtype=torch.float64)
    target = [torch.as_tensor(t.target, device="cuda", dtype=torch.float64) for t in targets]
    weights = [torch.as_tensor(t.weights, device="cuda", dtype=torch.float64) for t in targets]
    records = []
    for batch in (1, 6, 18):
        started = time.perf_counter()
        graph = GraphUnion(renderers[0], poses[0, :batch], warmup=3)
        torch.cuda.synchronize()
        capture_seconds = time.perf_counter() - started

        def evaluate(method, frame, graph=graph, batch=batch):
            if method == "graph_public_copy":
                result = graph.evaluate(poses[frame, :batch], target[frame], weights[frame],
                                        renderer=renderers[frame], copy_outputs=True)
                return result.union_alpha, result.loss, result.pose_gradient
            pose = poses[frame, :batch].detach().clone().requires_grad_(True)
            alpha = render_union(renderers[frame], pose)
            loss = ((alpha - target[frame]).square()
                    * (weights[frame] * renderers[frame].ray_valid)).sum((1, 2)) / 3
            gradient, = torch.autograd.grad(loss.sum(), pose)
            return alpha.detach(), loss.detach(), gradient

        differences = []
        for frame in (0, 1, 0):
            expected = evaluate("eager", frame)
            actual = evaluate("graph_public_copy", frame)
            maximum = []
            for got, want in zip(actual, expected, strict=True):
                maximum.append(float((got - want).abs().max()))
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            differences.append(dict(frame=frames[frame].frame_idx,
                                    union=maximum[0], loss=maximum[1], gradient=maximum[2]))
        del expected, actual
        for frame in (0, 1, 0):
            evaluate("eager", frame)
            evaluate("graph_public_copy", frame)
        timing = []
        for repeat in range(args.repeats):
            frame = repeat % 2
            methods = ("eager", "graph_public_copy")
            for method in (methods if repeat % 2 == 0 else methods[::-1]):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                started = time.perf_counter()
                result = evaluate(method, frame)
                torch.cuda.synchronize()
                milliseconds = (time.perf_counter() - started) * 1000
                timing.append(dict(repeat=repeat, frame=frames[frame].frame_idx, method=method,
                                   milliseconds=milliseconds,
                                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   peak_extra_bytes=torch.cuda.max_memory_allocated() - before))
                del result
        medians = {method: float(np.median([row["milliseconds"] for row in timing
                                            if row["method"] == method])) for method in methods}
        speedup = []
        for repeat in range(args.repeats):
            row = {item["method"]: item["milliseconds"] for item in timing
                   if item["repeat"] == repeat}
            speedup.append(row["eager"] / row["graph_public_copy"])
        record = dict(batch=batch, capture_and_warmup_seconds=capture_seconds,
                      positive_product_bound_used=graph.uses_positive_product_bound,
                      transmission_lower_bound=graph.transmission_lower_bound,
                      strict_bitwise_parity=True, max_absolute_differences=differences,
                      median_milliseconds=medians, paired_median_speedup=float(np.median(speedup)),
                      measurements=timing)
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "measurements"}), flush=True)
        del graph, evaluate
        gc.collect()
    if {p.relative_to(REPO).as_posix(): digest(p) for p in paths} != hashes:
        raise RuntimeError("source changed during short GPU test")
    inputs = {str(path): digest(path) for path in (
        forward_run / "candidate.npz", sequence / "manifest.json")}
    inputs.update({str(path): digest(path) for path in (forward_run / "calibration_snapshot").glob(
        "*.json")})
    for category in ("raw", "masks", "backgrounds"):
        for view in VIEWS:
            item = reader.manifest[category][view]
            inputs[str(sequence / item["path"])] = item["sha256"]
    report = dict(
        status="ok", source_sha256=hashes, input_sha256=inputs,
        samples_sha256=digest(sample), torch=torch.__version__,
        torch_git_version=torch.version.git_version, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), dtype="float64", numpy=np.__version__,
        python=sys.version, platform=platform.platform(), results=records,
        prod_backward_source=PROD_BACKWARD_ORIGIN,
        baseline="frozen render_union, original Gaussian validation, fresh pose gradient",
        graph_timing_includes="public pose/target/weight/width/ray validation, all input buffer "
                              "copies including changed frame rays, replay, "
                              "independent output clones",
        frame_shape=[3, 1024], poses="v1 selected forward poses plus fixed +/-0.01mm perturbation",
        limits=["D2 staging ran concurrently; paired timings are descriptive, "
                "not isolated speedups.",
                "Graph is fixed per batch/N/config/dtype and does not implement "
                "Adam or full fitting.",
                "Pose gradient is explicit, not attached to caller autograd; "
                "no width/higher gradients.",
                "An explicit zero-safe product backward matches the installed PyTorch commit; "
                "revalidate if its reduction/backward implementation changes.",
                "Only rendering plus image loss is compared; no end-to-end runtime claim.",
                "Public output copies prevent later replay from overwriting a retained result."],
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()

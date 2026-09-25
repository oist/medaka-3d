"""Isolated CUDA64 full/union-only benchmark on one preserved real observation.

Run only after other scheduled GPU jobs finish. This is a timing/numerical-parity
experiment, not an accuracy evaluation. No trajectory after the first seed or
identity annotation is read. Construction, input loading and hashing are excluded
from synchronized forward-plus-backward timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fast_union import render_union
from observations import SequenceReader, sample_pixels
from renderer import RefractiveFishRenderer, RendererConfig

from medaka3d.calib import load_rig
from medaka3d.types import VIEWS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True,
                        help="Contains first seed_registration.json and calibration_snapshot only")
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 6, 18])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.repeats < 1 or any(batch < 1 for batch in args.batches):
        parser.error("warmup, repeats and every batch size must be positive")
    return args


def main(argv=None):
    args = arguments(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("this isolated benchmark requires CUDA; --help never initializes CUDA")
    args.output.mkdir(parents=True, exist_ok=False)
    lab = Path(__file__).resolve().parent
    source_paths = [lab / name for name in (
        "profile_fast_union.py", "fast_union.py", "renderer.py", "observations.py",
        "prepare_sequence.py")]
    source_paths.append(Path("src/medaka3d/calib.py"))
    source_hashes = {str(path): digest(path) for path in source_paths}
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for path in source_paths:
        shutil.copyfile(path, snapshot / path.name)
    seed_path = args.reference_run / "seed_registration.json"
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    initial = np.asarray(seed["pose_xyz"], dtype=np.float64)
    if initial.shape != (3, 3, 3) or not np.isfinite(initial).all():
        raise ValueError("reference seed must contain three finite three-node bodies")
    calibration = args.reference_run / "calibration_snapshot"
    rig = load_rig(calibration)
    reader = SequenceReader(args.sequence, args.input_root)
    iterator = iter(reader)
    try:
        frame = next(iterator)
    finally:
        iterator.close()
    if frame.frame_idx != seed["frame"]:
        raise ValueError("the first observation and preserved first seed must have the same frame")
    targets = sample_pixels(frame, "detectron2", 384, 384, 256)
    if targets.target.shape != (3, 1024):
        raise ValueError("benchmark requires exactly 1024 pixels per view")
    rng = np.random.default_rng(931204)
    perturbations = rng.uniform(-0.01, 0.01, (max(args.batches), 3, 3, 3))
    perturbations[0] = 0  # Keep the actual seed as the first candidate in every batch.
    poses = initial[None] + perturbations
    widths = np.full((max(args.batches), 3), 0.06)
    sample_path = args.output / "samples.npz"
    np.savez_compressed(sample_path, frame_idx=frame.frame_idx, poses=poses, width_ratio=widths,
                        pixels=np.stack([targets.pixels[view] for view in VIEWS]),
                        target=targets.target, weights=targets.weights)
    setup = time.perf_counter()
    renderer = RefractiveFishRenderer(rig, targets.pixels, RendererConfig(),
                                     device="cuda", dtype=torch.float64)
    target = torch.as_tensor(targets.target, dtype=torch.float64, device="cuda")
    weights = torch.as_tensor(targets.weights, dtype=torch.float64, device="cuda")
    weights = weights * renderer.ray_valid
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup
    results = []
    for batch in args.batches:
        pose = torch.tensor(poses[:batch], dtype=torch.float64, device="cuda", requires_grad=True)
        width = torch.tensor(widths[:batch], dtype=torch.float64, device="cuda", requires_grad=True)

        def evaluate(method, pose=pose, width=width):
            alpha = (renderer.render(pose, width).union_alpha if method == "full"
                     else render_union(renderer, pose, width))
            loss = ((alpha - target) ** 2 * weights).sum() / 3
            return alpha, loss

        reference, reference_loss = evaluate("full")
        expected_gradients = torch.autograd.grad(reference_loss, (pose, width))
        actual, actual_loss = evaluate("fast")
        actual_gradients = torch.autograd.grad(actual_loss, (pose, width))
        arrays = [(actual, reference), *zip(actual_gradients, expected_gradients, strict=True)]
        differences = [float((candidate - expected).abs().max()) for candidate, expected in arrays]
        for candidate, expected in arrays:
            torch.testing.assert_close(candidate, expected, rtol=0, atol=0)
        del reference, reference_loss, expected_gradients, actual, actual_loss
        del actual_gradients, arrays, candidate, expected

        def timed(method, pose=pose, width=width, evaluate=evaluate):
            pose.grad = None
            width.grad = None
            torch.cuda.synchronize()
            before_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            alpha, loss = evaluate(method)
            loss.backward()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            peak_delta = torch.cuda.max_memory_allocated() - before_memory
            del alpha, loss
            return seconds, peak_delta

        for repeat in range(args.warmup):
            order = ("full", "fast") if repeat % 2 == 0 else ("fast", "full")
            for method in order:
                timed(method)
        measurements = []
        for repeat in range(args.repeats):
            order = ("full", "fast") if repeat % 2 == 0 else ("fast", "full")
            for method in order:
                seconds, memory = timed(method)
                measurements.append(dict(repeat=repeat, method=method, seconds=seconds,
                                         peak_extra_allocated_bytes=memory))
        timing = {}
        for method in ("full", "fast"):
            seconds = [row["seconds"] for row in measurements if row["method"] == method]
            timing[method] = dict(median_s=float(np.median(seconds)),
                                  p95_s=float(np.quantile(seconds, .95)))
        pairs = [next(row["seconds"] for row in measurements
                      if row["repeat"] == repeat and row["method"] == "full")
                 / next(row["seconds"] for row in measurements
                        if row["repeat"] == repeat and row["method"] == "fast")
                 for repeat in range(args.repeats)]
        record = dict(batch=batch, strict_bitwise_parity=True,
                      max_abs_diff=dict(union=differences[0], pose_gradient=differences[1],
                                        width_gradient=differences[2]),
                      timing=timing, paired_speedup_median=float(np.median(pairs)),
                      measurements=measurements)
        results.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != "measurements"}),
              flush=True)
    if {str(path): digest(path) for path in source_paths} != source_hashes:
        raise RuntimeError("benchmark source changed during the run")
    inputs = {str(args.sequence / "manifest.json"): digest(args.sequence / "manifest.json"),
              str(seed_path): digest(seed_path)}
    inputs.update({str(path): digest(path) for path in calibration.glob("*.json")})
    record = dict(
        status="ok", purpose="isolated numerical-parity and timing comparison; no accuracy claim",
        isolation="run only after other scheduled GPU jobs finish; external scheduling required",
        torch=torch.__version__, numpy=np.__version__, python=sys.version,
        platform=platform.platform(), gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda, dtype="float64", seed=931204, frame_idx=frame.frame_idx,
        pixels_per_view=1024, width_ratio=0.06, perturbation_max_abs_mm=0.01,
        warmup_per_method=args.warmup, timed_repeats_per_method=args.repeats,
        alternating_method_order=True, synchronize_before_and_after=True,
        renderer_setup_s=setup_seconds, source_hashes=source_hashes, inputs=inputs,
        samples_sha256=digest(sample_path), results=results,
        scope="Forward plus weighted union-MSE backward over all pose/width parameters; "
              "input loading, ray caching and diagnostics used after optimization are excluded.",
    )
    (args.output / "results.json").write_text(json.dumps(record, indent=2, allow_nan=False),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()

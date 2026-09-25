"""Stage fixed development samples, then profile temporal CUDA64 rendering in isolation.

Preparation is CPU-only. GPU execution requires an external scheduling signal;
this script does not detect whether other agents have jobs running.
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

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from fast_union import render_union  # noqa: E402
from observations import SequenceReader, sample_pixels  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402
from temporal_renderer import TemporalUnionRenderer  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sources():
    return [Path(__file__), *(LAB / name for name in (
        "renderer.py", "fast_union.py", "temporal_renderer.py", "observations.py",
        "prepare_sequence.py")), REPO / "src/medaka3d/calib.py"]


def source_hashes():
    return {str(path.relative_to(REPO)): digest(path) for path in sources()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for path in sources():
        shutil.copyfile(path, snapshot / path.name)
    candidate = args.reference_run / "candidate.npz"
    frames = np.arange(args.frame_start, args.frame_start + args.frame_count)
    with np.load(candidate, allow_pickle=False) as data:
        indexes = np.searchsorted(data["frame_idx"], frames)
        if not np.array_equal(data["frame_idx"][indexes], frames):
            raise ValueError("forward output must cover exactly the requested native frames")
        forward = data["pose_xyz"][indexes]
    if not np.isfinite(forward).all() or forward.shape != (args.frame_count, 3, 3, 3):
        raise ValueError("forward poses must be finite three-node latent coordinates")
    calibration = args.output / "calibration_snapshot"
    calibration.mkdir()
    inputs = {str(candidate): digest(candidate)}
    for path in (args.reference_run / "calibration_snapshot").glob("*.json"):
        shutil.copyfile(path, calibration / path.name)
        inputs[str(path)] = digest(path)
    reader = SequenceReader(args.sequence, args.input_root)
    iterator = iter(reader)
    targets, found = [], []
    try:
        for frame in iterator:
            if frame.frame_idx < frames[0]:
                continue
            if frame.frame_idx > frames[-1]:
                break
            targets.append(sample_pixels(frame, "detectron2", 96, 96, 64))
            found.append(frame.frame_idx)
    finally:
        iterator.close()
    if not np.array_equal(found, frames):
        raise ValueError("native images must cover the full requested window")
    pixels = np.stack([np.stack([t.pixels[v] for v in VIEWS]) for t in targets])
    if pixels.shape != (args.frame_count, 3, 256, 2):
        raise ValueError("expected 256 independent samples per view and frame")
    sample = args.output / "samples.npz"
    np.savez_compressed(sample, frame_idx=frames, forward_pose=forward,
                        poses=np.repeat(forward[None], args.batch, axis=0),
                        width_ratio=np.full((args.batch, args.frame_count, 3), 0.06),
                        pixels=pixels, target=np.stack([t.target for t in targets]),
                        weights=np.stack([t.weights for t in targets]))
    manifest = args.sequence / "manifest.json"
    inputs[str(manifest)] = digest(manifest)
    for category in ("raw", "masks", "backgrounds"):
        for view in VIEWS:
            item = reader.manifest[category][view]
            inputs[str(args.sequence / item["path"])] = item["sha256"]
    if source_hashes() != hashes:
        raise RuntimeError("source changed during CPU preparation")
    setup = dict(
        status="prepared_cpu_only", source_sha256=hashes, input_sha256=inputs,
        samples_sha256=digest(sample), batch=args.batch, frames=frames.tolist(),
        samples_per_view=256, sampling=dict(positive=96, near=96, global_background=64),
        pose_construction="identical repeated selected v1 forward latent poses; no new seed",
        time_chunk_size=5, dtype="float64", mode="detectron2",
        targets="sampled independently of every pose/ID candidate from native raw and masks",
        purpose="renderer numerical parity, memory and timing only; no tracking-quality claim",
    )
    write_json(args.output / "setup.json", setup)
    print(f"Prepared {args.batch} x {args.frame_count} x 3 x 256 CPU samples.", flush=True)


def run(args):
    setup = json.loads((args.output / "setup.json").read_text(encoding="utf-8"))
    if source_hashes() != setup["source_sha256"]:
        raise RuntimeError("current source differs from prepared frozen source")
    if digest(args.output / "samples.npz") != setup["samples_sha256"]:
        raise RuntimeError("prepared samples changed")
    if not torch.cuda.is_available():
        raise RuntimeError("the isolated run requires CUDA")
    with np.load(args.output / "samples.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    rig = load_rig(args.output / "calibration_snapshot")
    pixels = [{v: values[vi] for vi, v in enumerate(VIEWS)} for values in arrays["pixels"]]
    config = RendererConfig()
    methods = {
        "checkpoint_off_chunk5": TemporalUnionRenderer(
            rig, pixels, config, device="cuda", time_chunk_size=5, checkpoint_chunks=False),
        "checkpoint_on_chunk5": TemporalUnionRenderer(
            rig, pixels, config, device="cuda", time_chunk_size=5, checkpoint_chunks=True),
    }
    per_frame = [RefractiveFishRenderer(rig, p, config, device="cuda", dtype=torch.float64)
                 for p in pixels]
    pose = torch.tensor(arrays["poses"], device="cuda", dtype=torch.float64, requires_grad=True)
    width = torch.tensor(arrays["width_ratio"], device="cuda", dtype=torch.float64,
                         requires_grad=True)
    target = torch.as_tensor(arrays["target"], device="cuda", dtype=torch.float64)
    weights = torch.as_tensor(arrays["weights"], device="cuda", dtype=torch.float64)
    weights = weights * methods["checkpoint_on_chunk5"].ray_valid
    times = len(pixels)

    def evaluate(method):
        if method == "per_frame_loop":
            alpha = torch.stack([render_union(r, pose[:, t], width[:, t])
                                 for t, r in enumerate(per_frame)], dim=1)
        else:
            alpha = methods[method].render(pose, width)
        loss = ((alpha - target).square() * weights).sum() / (times * 3)
        return alpha, loss

    def once(method, keep=False):
        pose.grad = None
        width.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        started = time.perf_counter()
        alpha, loss = evaluate(method)
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        record = dict(method=method, seconds=elapsed,
                      baseline_allocated_bytes=before,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      extra_peak_allocated_bytes=torch.cuda.max_memory_allocated() - before)
        held = None
        if keep:
            held = (alpha.detach().cpu(), pose.grad.detach().cpu(), width.grad.detach().cpu())
        del alpha, loss
        return record, held

    reference_record, reference = once("per_frame_loop", keep=True)
    correctness = [reference_record]
    print(json.dumps(reference_record), flush=True)
    for method in methods:
        record, actual = once(method, keep=True)
        differences = {}
        for name, got, want in zip(("union", "pose_gradient", "width_gradient"),
                                   actual, reference, strict=True):
            differences[name] = float((got - want).abs().max())
            torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-11)
        record["max_absolute_differences"] = differences
        record["float64_parity_atol_rtol"] = 1e-11
        correctness.append(record)
        print(json.dumps(record), flush=True)
        del actual
    del reference
    for _ in range(args.warmup):
        for method in methods:
            once(method)
    measurements = []
    names = list(methods)
    for repeat in range(args.repeats):
        for method in (names if repeat % 2 == 0 else names[::-1]):
            record, _ = once(method)
            record["repeat"] = repeat
            measurements.append(record)
    summaries = {}
    for method in methods:
        rows = [row for row in measurements if row["method"] == method]
        summaries[method] = dict(
            median_seconds=float(np.median([r["seconds"] for r in rows])),
            p95_seconds=float(np.quantile([r["seconds"] for r in rows], .95)),
            maximum_peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in rows),
            maximum_extra_peak_allocated_bytes=max(r["extra_peak_allocated_bytes"] for r in rows),
        )
    if source_hashes() != setup["source_sha256"]:
        raise RuntimeError("source changed during isolated measurement")
    report = dict(
        status="ok", setup=setup, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        python=sys.version, platform=platform.platform(), numpy=np.__version__,
        external_isolation_required=True, no_other_scheduled_gpu_jobs=True,
        synchronized_forward_backward=True, timed_repeats=args.repeats,
        warmup_per_method=args.warmup, correctness=correctness,
        measurements=measurements, summaries=summaries,
        scope="Full union plus weighted image MSE backward for all pose/width parameters; "
              "not a full temporal optimizer, assignment or tracking benchmark.",
        limitations=["Forward poses are estimates, not anatomical or identity truth.",
                     "Reported memory is torch allocated/reserved memory, not driver-wide usage.",
                     "Reference loop is timed once for correctness, without warmup.",
                     "T31 memory test does not establish safety for every T50 workload."],
    )
    write_json(args.output / "results.json", report)
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path,
                        default=Path("work/three_body/inputs/transient_333519-333719"))
    parser.add_argument("--reference-run", type=Path,
                        default=Path("work/three_body/v1_transient_detectron2"))
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=333528)
    parser.add_argument("--frame-count", type=int, default=31)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--prepare-only", action="store_true")
    options.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    if min(args.batch, args.frame_count, args.warmup, args.repeats) < 1:
        parser.error("batch, frames, warmup and repeats must be positive")
    if not args.run_prepared:
        prepare(args)
    if not args.prepare_only:
        run(args)


if __name__ == "__main__":
    main()

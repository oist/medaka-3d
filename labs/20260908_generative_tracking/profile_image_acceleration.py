"""Measure actual three-camera CPU/CUDA image processing and explicit flow320 differences."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from accelerated_observations import CudaEvidenceBuilder, ParallelEvidenceBuilder
from benchmark import (
    BACKGROUND_FRAMES,
    BASELINE_DIR,
    CALIBRATION_DIR,
    EXPERIMENT,
    CachedDetections,
    make_backgrounds,
    open_videos,
    seek_read,
    sha256,
    skeleton_from_sources,
)
from observations import EvidenceConfig, SceneEvidence, _sample
from tracker import FrameInput, GenerativeTracker, TrackerConfig

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS


def same_maps(reference, actual, include_flow=True):
    fields = ["rgb", "foreground", "distance", "labels", "component_areas"]
    if include_flow:
        fields += ["flow", "flow_confidence"]
    for view in reference.views:
        for field in fields:
            if not np.array_equal(getattr(reference.views[view], field),
                                  getattr(actual.views[view], field)):
                raise AssertionError(f"maps differ: {view}/{field}")
        if reference.views[view].quality != actual.views[view].quality:
            raise AssertionError(f"foreground quality differs: {view}")


def flow_pixels(view, uv):
    scale = np.asarray(view.flow_scale)
    pixels = (uv + 0.5) * scale - 0.5
    return np.column_stack([_sample(view.flow[..., axis], pixels) / scale[axis]
                            for axis in range(2)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite profile output: {args.out}")
    if not 1 <= args.start < args.stop or args.stop - args.start > 32:
        raise ValueError("use 1-32 consecutive frames with an available preceding frame")
    args.out.mkdir(parents=True)
    with provenance("profile_image_acceleration", args.out, name="profile") as record:
        record.params_from(args)
        code_dir = args.out / "code_snapshot"
        code_dir.mkdir()
        paths = [Path(__file__), Path(__file__).with_name("accelerated_observations.py"),
                 Path(__file__).with_name("observations.py"),
                 Path(__file__).with_name("tracker.py"),
                 Path(__file__).with_name("benchmark.py"),
                 Path(__file__).resolve().parents[2] / "src/medaka3d/calib.py"]
        record.param("code_hashes", {str(path): sha256(path) for path in paths})
        for path in paths:
            shutil.copyfile(path, code_dir / path.name)
        detections = CachedDetections(args.input_root, EXPERIMENT)
        rig = load_rig(args.input_root / CALIBRATION_DIR)
        for view, path in detections.paths.items():
            record.input(view, path, sha256=sha256(path))
        seed_path = args.input_root / BASELINE_DIR / f"C34-label__{EXPERIMENT}__lam0.767.npz"
        with np.load(seed_path, allow_pickle=False) as arrays:
            source = arrays["source"][args.start:args.start + 1].copy()
        initial, observed, _ = skeleton_from_sources(rig, detections, [args.start], source)
        if not observed.all():
            raise ValueError("the first frame must provide three complete seed skeletons")
        record.input("seed", seed_path, sha256=sha256(seed_path))
        record.param("seed_pose", initial[0].tolist())
        frames = np.arange(args.start, args.stop)
        _, captures, metadata = open_videos(args.raw_root, EXPERIMENT, detections.n_frames)
        record.param("source_videos", metadata)
        background_frames = [f for f in BACKGROUND_FRAMES
                             if f < args.start - 332 or f >= args.stop + 332]
        record.param("background_frames", background_frames)
        try:
            backgrounds = make_backgrounds(captures, background_frames)
            previous = {view: seek_read(cap, args.start - 1, view)
                        for view, cap in captures.items()}
            pairs = []
            for frame in frames:
                rgb = {}
                for view, capture in captures.items():
                    ok, bgr = capture.read()
                    if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
                        raise RuntimeError(f"unaligned source read: {view}/{frame}")
                    rgb[view] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pairs.append((rgb, previous))
                previous = rgb
        finally:
            for capture in captures.values():
                capture.release()
        variants, held = {}, {}
        factories = (("serial640", None, 640), ("parallel640", ParallelEvidenceBuilder, 640),
                     ("cuda640", CudaEvidenceBuilder, 640),
                     ("parallel320", ParallelEvidenceBuilder, 320),
                     ("cuda320", CudaEvidenceBuilder, 320))
        for name, factory, size in factories:
            config = EvidenceConfig(flow_max_side=size)
            builder = None if factory is None else factory(config)
            call = SceneEvidence.from_frames if builder is None else builder.from_frames
            times, gpu_times, scenes = [], [], []
            print(f"benchmark preprocessing {name}", flush=True)
            try:
                call(pairs[0][0], backgrounds, pairs[0][1], config)  # warm kernels/pool
                for rgb, previous in pairs:
                    started = time.perf_counter()
                    scene = call(rgb, backgrounds, previous, config)
                    times.append(time.perf_counter() - started)
                    gpu_times.append(0.0 if builder is None else
                                     builder.last_timings["gpu_foreground_seconds"])
                    scenes.append(scene)
                variants[name] = {
                    "config": dataclasses.asdict(config),
                    "metadata": ({"backend": "v3-sequential"} if builder is None
                                 else builder.metadata),
                    "seconds": times, "median_seconds": float(np.median(times)),
                    "fps": len(times) / sum(times),
                    "median_gpu_seconds": float(np.median(gpu_times)),
                }
                if name == "serial640":
                    held[name] = scenes
                elif name in ("parallel640", "cuda640"):
                    for expected, actual in zip(held["serial640"], scenes, strict=True):
                        same_maps(expected, actual)
                    variants[name]["all_v3_maps_exact_equal"] = True
                    if name == "cuda640":
                        held[name] = scenes
                elif name == "parallel320":
                    held[name] = scenes
                else:
                    for expected, actual in zip(held["parallel320"], scenes, strict=True):
                        same_maps(expected, actual)
                    variants[name]["all_parallel320_maps_exact_equal"] = True
                    held[name] = scenes
            finally:
                if builder is not None:
                    builder.close()
        flow_difference = []
        for frame, high, low in zip(frames, held["serial640"], held["cuda320"], strict=True):
            same_maps(high, low, include_flow=False)
            for view in VIEWS:
                uv = detections.points(view, int(frame) - 1).reshape(-1, 2)
                difference = np.linalg.norm(flow_pixels(high.views[view], uv)
                                            - flow_pixels(low.views[view], uv), axis=1)
                flow_difference.extend(difference[np.isfinite(difference)].tolist())
        track_outputs = {}
        for name in ("serial640", "cuda640", "cuda320"):
            print(f"benchmark unchanged v3 tracker {name}", flush=True)
            inputs = [FrameInput(int(frame), {view: detections.points(view, int(frame))
                                             for view in VIEWS}, scene)
                      for frame, scene in zip(frames, held[name], strict=True)]
            started = time.perf_counter()
            result = GenerativeTracker(rig, initial[0], TrackerConfig()).track(inputs)
            variants[name]["tracker_seconds"] = time.perf_counter() - started
            output = args.out / f"tracks_{name}.npz"
            np.savez_compressed(output, pose_xyz=result.pose_xyz, node_source=result.node_source,
                                observed=result.observed, review=result.review,
                                accepted_xyz=result.accepted_xyz, frame_idx=result.frame_idx)
            record.output(name, output, sha256=sha256(output))
            track_outputs[name] = result
        high, same, low = (track_outputs[name] for name in ("serial640", "cuda640", "cuda320"))
        for field in ("pose_xyz", "node_source", "observed", "review"):
            if not np.array_equal(getattr(high, field), getattr(same, field)):
                raise AssertionError(f"CUDA640 tracker differs despite equal maps: {field}")
        comparison = {
            "cuda640_tracker_exact_equal_to_serial640": True,
            "cuda320_max_pose_difference_mm": float(np.max(np.linalg.norm(
                high.pose_xyz - low.pose_xyz, axis=-1))),
            "cuda320_node_source_difference_count": int(
                (high.node_source != low.node_source).sum()),
            "cuda320_review_difference_count": int((high.review != low.review).sum()),
            "flow320_vs640_original_pixels": {
                "samples": len(flow_difference), "median": float(np.median(flow_difference)),
                "p95": float(np.quantile(flow_difference, 0.95)),
                "max": float(np.max(flow_difference)),
            },
        }
        summary = {"frames": [args.start, args.stop], "variants": variants,
                   "comparison": comparison,
                   "scope": "Timing includes GPU transfers and synchronization but excludes video "
                            "decoding/background preparation. No identity-accuracy claim. Flow320 "
                            "changes the observation algorithm; source frames remain 166 fps.",
                   "official_references": [
                       "https://docs.pytorch.org/docs/main/generated/torch.cuda.synchronize.html",
                       "https://docs.pytorch.org/docs/2.9/generated/torch.histc.html",
                       "https://docs.opencv.org/4.12.0/d9/d30/classcv_1_1cuda_1_1FarnebackOpticalFlow.html",
                   ]}
        result_path = args.out / "acceleration.json"
        result_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", result_path, sha256=sha256(result_path))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Score frozen temporal poses on unused pixels of the same development images.

This is spatial resampling within the fitted images, not held-out frames, an
independent biological replicate, anatomical truth or identity validation.
Sampling is fixed before scores are computed and excludes every fitted coordinate
in the same frame/view. All original and candidate poses share the same targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from observations import SequenceReader  # noqa: E402
from renderer import RendererConfig  # noqa: E402
from temporal_renderer import TemporalUnionRenderer  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

COUNTS = (96, 96, 64)
SEED_OFFSET = 8803010


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unused_samples(frame, fitted_pixels):
    pixels, target, weights, membership, audit = [], [], [], [], []
    for vi, view in enumerate(VIEWS):
        height, width = frame.images_bgr[view].shape[:2]
        fitted = np.asarray(fitted_pixels[vi])
        if not np.isfinite(fitted).all() or not np.array_equal(fitted, np.rint(fitted)):
            raise ValueError("fitted samples must be finite native integer pixel centers")
        fitted_xy = fitted.astype(int)
        fitted_flat = np.ravel_multi_index((fitted_xy[:, 1], fitted_xy[:, 0]), (height, width))
        allowed = np.ones(height * width, bool)
        allowed[fitted_flat] = False
        masks = frame.masks[view]
        union = (np.zeros((height, width), bool) if masks is None
                 else np.any(masks, axis=0))
        dilation = cv2.dilate(union.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        classes = (union, dilation & ~union, ~union)
        pools = [np.flatnonzero(group.ravel() & allowed) for group in classes]
        fallback = np.flatnonzero(allowed)
        if not len(fallback):
            raise ValueError("no unused pixel remains")
        rng = np.random.default_rng(frame.frame_idx * 7 + vi + SEED_OFFSET)
        selections = [rng.choice(pool if len(pool) else fallback, count,
                                 replace=len(pool if len(pool) else fallback) < count)
                      for pool, count in zip(pools, COUNTS, strict=True)]
        selected = np.concatenate(selections)
        y, x = np.unravel_index(selected, (height, width))
        t = np.zeros(sum(COUNTS), np.float32)
        w = np.zeros_like(t)
        # Preserve the fitting sampler's neutral empty-foreground rule and its
        # class-normalized weights. Exclusion can also leave a class unavailable.
        if len(pools[0]):
            t[:COUNTS[0]] = 1
            w[:COUNTS[0]] = 1 / COUNTS[0]
            if len(pools[1]):
                w[COUNTS[0]:COUNTS[0] + COUNTS[1]] = 0.75 / COUNTS[1]
            if len(pools[2]):
                w[-COUNTS[2]:] = 0.25 / COUNTS[2]
        overlaps = np.isin(selected, fitted_flat)
        if overlaps.any():
            raise AssertionError("evaluation sample reused a fitted frame/view/pixel")
        held = w > 0
        if not np.array_equal(union[y[held], x[held]], t[held].astype(bool)):
            raise AssertionError("sample target contradicts the independent frame mask")
        pixels.append(np.column_stack((x, y)).astype(float))
        target.append(t)
        weights.append(w)
        membership.append(overlaps)
        audit.append(dict(
            frame=frame.frame_idx, view=view, source_mask_available=masks is not None,
            source_foreground_pixels=int(union.sum()),
            available_unused_pool_sizes=[len(pool) for pool in pools],
            selected_unique_by_class=[len(np.unique(s)) for s in selections],
            fitted_unique_pixels=len(np.unique(fitted_flat)),
            overlap_count=int(overlaps.sum()), known_weight_sum=float(w.sum()),
            sampler_seed=frame.frame_idx * 7 + vi + SEED_OFFSET,
        ))
    return (np.stack(pixels), np.stack(target), np.stack(weights),
            np.stack(membership), audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=Path("work/three_body/temporal_v2_transient_333528-333559"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-samples", type=Path,
                        help="Reuse exact saved evaluation NPZ pixels/targets; do not resample")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), *(LAB / name for name in (
        "observations.py", "prepare_sequence.py", "renderer.py", "temporal_renderer.py")),
        REPO / "src/medaka3d/calib.py"]
    hashes = {p.relative_to(REPO).as_posix(): digest(p) for p in paths}
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for path in paths:
        shutil.copyfile(path, snapshot / path.name)
    provenance_path = args.run / "refinement.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    summary_path = args.run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if provenance["status"] != "ok" or summary["mode"] != "detectron2":
        raise ValueError("requires the completed fixed Detectron2 temporal run")
    # The intended camera/renderer must be the same frozen source as the run.
    for relative, expected in summary["code_sha256_before"].items():
        relative = relative.replace("\\", "/")
        if relative in hashes and hashes[relative] != expected:
            raise ValueError(f"changed frozen dependency: {relative}")
    hypotheses_path, fitted_path = args.run / "hypotheses.npz", args.run / "sampled_targets.npz"
    with np.load(hypotheses_path, allow_pickle=False) as data:
        hypotheses = {key: data[key] for key in data.files}
    with np.load(fitted_path, allow_pickle=False) as data:
        fitted = {key: data[key] for key in data.files}
    np.testing.assert_array_equal(hypotheses["frame_idx"], fitted["frame_idx"])
    if digest(fitted_path) != summary["sampled_targets_sha256"]:
        raise ValueError("fitted target file differs from run provenance")
    params = provenance["params"]
    forward_path = Path(params["run"]) / "candidate.npz"
    if digest(forward_path) != summary["forward_candidate_sha256"]:
        raise ValueError("forward candidate changed")
    frames = fitted["frame_idx"]
    with np.load(forward_path, allow_pickle=False) as data:
        indexes = np.searchsorted(data["frame_idx"], frames)
        np.testing.assert_array_equal(data["frame_idx"][indexes], frames)
        original = data["pose_xyz"][indexes]
    poses = np.concatenate((original[None], hypotheses["pose_xyz"]))
    reader = SequenceReader(Path(params["sequence"]), Path(params["input_root"]))
    if args.reuse_samples is None:
        iterator = iter(reader)
        selected = []
        try:
            for frame in iterator:
                if frame.frame_idx < frames[0]:
                    continue
                if frame.frame_idx > frames[-1]:
                    break
                t = len(selected)
                if frame.frame_idx != frames[t]:
                    raise ValueError("raw image times do not match the frozen window")
                selected.append(unused_samples(frame, fitted["pixels"][t]))
        finally:
            iterator.close()
        if len(selected) != len(frames):
            raise ValueError("incomplete native-frame samples")
        pixels, target, weights, overlaps = [
            np.stack([row[k] for row in selected]) for k in range(4)]
        membership = [entry for row in selected for entry in row[4]]
    else:
        with np.load(args.reuse_samples, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["frame_idx"], frames)
            np.testing.assert_array_equal(data["fitted_pixels"], fitted["pixels"])
            np.testing.assert_array_equal(data["views"], VIEWS)
            pixels, target, weights, overlaps = [data[k] for k in (
                "pixels", "target", "weights", "reused_fitted_coordinate")]
        old = json.loads((args.reuse_samples.parent / "results.json").read_text(encoding="utf-8"))
        if digest(args.reuse_samples) != old["samples_sha256"]:
            raise ValueError("saved disjoint samples changed")
        membership = old["membership_audit"]
        if overlaps.any():
            raise ValueError("saved evaluation target overlaps fitted samples")
    torch.set_num_threads(2)
    rig = load_rig(args.run / "calibration_snapshot")
    width = summary["config"]["width_ratio"]
    renderer = TemporalUnionRenderer(
        rig, [{view: value[vi] for vi, view in enumerate(VIEWS)} for value in pixels],
        RendererConfig(width_ratio=width), device="cpu", dtype=torch.float64,
        time_chunk_size=5, checkpoint_chunks=False)
    started = time.perf_counter()
    with torch.no_grad():
        alpha = renderer.render(poses).numpy()
    render_seconds = time.perf_counter() - started
    valid = renderer.ray_valid.numpy()
    known_weights = weights * valid
    view_frame_error = ((alpha - target) ** 2 * known_weights).sum(axis=-1)
    scores = view_frame_error.mean(axis=(1, 2))
    labels = ["original_v1", *[f"H{i}" for i in range(len(hypotheses["pose_xyz"]))]]
    fitted_scores = np.r_[hypotheses["original_loss_parts"][0, 5],
                          hypotheses["loss_parts"][:, 5]]
    sample_path = args.output / "evaluation_samples_and_scores.npz"
    np.savez_compressed(
        sample_path, frame_idx=frames, views=np.asarray(VIEWS), labels=np.asarray(labels),
        pixels=pixels, target=target, weights=weights, ray_valid=valid,
        reused_fitted_coordinate=overlaps, fitted_pixels=fitted["pixels"],
        poses=poses, width_ratio=width, union_alpha=alpha,
        view_frame_error=view_frame_error, mean_error=scores)
    inputs = {str(p): digest(p) for p in (
        provenance_path, summary_path, hypotheses_path, fitted_path, forward_path)}
    if args.reuse_samples is not None:
        inputs[str(args.reuse_samples)] = digest(args.reuse_samples)
        inputs[str(args.reuse_samples.parent / "results.json")] = digest(
            args.reuse_samples.parent / "results.json")
    inputs[str(Path(params["sequence"]) / "manifest.json")] = digest(
        Path(params["sequence"]) / "manifest.json")
    for category in ("raw", "masks", "backgrounds"):
        for view in VIEWS:
            item = reader.manifest[category][view]
            inputs[str(Path(params["sequence"]) / item["path"])] = item["sha256"]
    for path in (args.run / "calibration_snapshot").glob("*.json"):
        inputs[str(path)] = digest(path)
    if {p.relative_to(REPO).as_posix(): digest(p) for p in paths} != hashes:
        raise RuntimeError("source changed during CPU scoring")
    records = [dict(label=label, mean_image_error=float(scores[i]),
                    mean_image_error_by_view=view_frame_error[i].mean(axis=0).tolist(),
                    mean_image_error_by_frame=view_frame_error[i].mean(axis=1).tolist(),
                    fitted_pixel_image_error=float(fitted_scores[i]))
               for i, label in enumerate(labels)]
    h1, h11 = labels.index("H1"), labels.index("H11")
    report = dict(
        status="ok", frames=[int(frames[0]), int(frames[-1]) + 1],
        device="CPU only", dtype="float64", torch_cpu_threads=2,
        render_seconds=render_seconds, source_sha256=hashes, input_sha256=inputs,
        samples_sha256=digest(sample_path), candidates=records,
        image_only_ranking=[labels[index] for index in np.argsort(scores, kind="stable")],
        H1_vs_original=dict(error_difference=float(scores[h1] - scores[0]),
                            lower_error=bool(scores[h1] < scores[0]),
                            lower_error_views=(view_frame_error[h1].mean(0)
                                               < view_frame_error[0].mean(0)).tolist(),
                            lower_error_frame_views=int((view_frame_error[h1]
                                                         < view_frame_error[0]).sum())),
        H1_vs_H11=dict(error_difference=float(scores[h1] - scores[h11]),
                       lower_error=bool(scores[h1] < scores[h11]),
                       lower_error_views=(view_frame_error[h1].mean(0)
                                          < view_frame_error[h11].mean(0)).tolist(),
                       lower_error_frame_views=int((view_frame_error[h1]
                                                    < view_frame_error[h11]).sum())),
        sampling=dict(counts=dict(positive=96, near=96, global_background=64),
                      exact_reuse_from=str(args.reuse_samples) if args.reuse_samples else None,
                      seed_rule=f"frame_idx * 7 + view_index + {SEED_OFFSET}",
                      exclusion="all fitted native coordinates in the same frame and view",
                      overlap_count=int(overlaps.sum()),
                      empty_foreground="all fitting weights zero; absence is neutral",
                      group_weight_sums=[1.0, 0.75, 0.25],
                      error_definition="mean over T,V of sum_N weight*(union-target)^2; "
                                       "same normalization as the original image objective"),
        membership_audit=membership,
        limits=["Same images and masks as fitting; only sampled coordinates are disjoint.",
                "Adjacent pixels are spatially correlated; no statistical independence is claimed.",
                "The masks may be wrong, and silhouette fit does not locate anatomical points.",
                "Union is invariant to fish ID permutation; these scores cannot validate IDs.",
                "No poses, widths, thresholds, seeds or models were refitted from these scores.",
                "Fitted-pixel scores are copied from the completed frozen run for context."],
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                             encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "render_seconds", "image_only_ranking", "H1_vs_original", "H1_vs_H11")},
        indent=2))
    print(json.dumps([{k: record[k] for k in ("label", "mean_image_error",
                                               "mean_image_error_by_view")}
                      for record in records], indent=2))


if __name__ == "__main__":
    main()

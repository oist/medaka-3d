"""Fixed-node curve-only image diagnostic; never changes tracked geometry or IDs."""

from __future__ import annotations

import argparse
import dataclasses
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
sys.path[:0] = [str(LAB), str(REPO / "src"), str(Path(__file__).parent)]
import identity_inputs  # noqa: E402
import measure_persistent_silhouette as measured  # noqa: E402
import measure_scene0_silhouette as attribution  # noqa: E402
import run_experiment as base  # noqa: E402
from body_pixel_evaluation import sample_body_evaluation, summarize_pixel_error  # noqa: E402
from curved_renderer import CurvedFishRenderer, tail_curve_diagnostics  # noqa: E402
from observations import FrameObservation, sample_pixels  # noqa: E402
from prepare_sequence import VIEWS, array_sha256, decode_frame_masks, sha256  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

FRAMES = measured.FRAMES
LOWER = (0.0, 0.0, -6.0)
UPPER = (192.0, 192.0, 186.0)
PROTOCOL = LAB / "BODY_PROPOSAL_PROTOCOL.md"


def constrain_offsets(pose, offset, lower=LOWER, upper=UPPER):
    """Perpendicular projection then ONE scalar for norm and control-box limits.

    Returns the constrained tensor and exact action diagnostics. The offset is
    not clipped by coordinate, and pose is not modified. A collapsed chord has
    only the zero feasible offset. This projection is applied after Adam steps.
    """
    if pose.shape != (3, 3, 3) or offset.shape != (3, 3):
        raise ValueError("pose/offset require fixed three-body shapes")
    if not bool(torch.isfinite(pose).all() and torch.isfinite(offset).all()):
        raise ValueError("constraint inputs must be finite")
    lo, hi = (torch.as_tensor(x, dtype=pose.dtype, device=pose.device) for x in (lower, upper))
    if not bool(((pose >= lo) & (pose <= hi)).all()):
        raise ValueError("fixed original pose is outside registered bounds")
    chord = pose[:, 2] - pose[:, 1]
    length = torch.linalg.vector_norm(chord, dim=-1)
    direction = chord / length.clamp_min(1e-12)[:, None]
    parallel = (offset * direction).sum(dim=-1)
    planar = offset - parallel[:, None] * direction
    planar = torch.where((length > 1e-12)[:, None], planar, torch.zeros_like(planar))
    norm = torch.linalg.vector_norm(planar, dim=-1)
    norm_scale = torch.minimum(torch.ones_like(norm), length / norm.clamp_min(1e-12))
    midpoint = (pose[:, 1] + pose[:, 2]) / 2
    positive = (hi - midpoint) / planar.clamp_min(1e-300)
    negative = (midpoint - lo) / (-planar).clamp_min(1e-300)
    limits = torch.where(planar > 0, positive, torch.where(planar < 0, negative, torch.inf))
    box_scale = torch.minimum(torch.ones_like(norm), limits.amin(dim=-1)).clamp_min(0)
    scalar = torch.minimum(norm_scale, box_scale)
    bounded = planar * scalar[:, None]
    return bounded, dict(
        parallel_removed_mm=parallel,
        norm_scale=norm_scale,
        box_scale=box_scale,
        applied_scalar=scalar,
        norm_constraint_active=norm_scale < 1,
        box_constraint_active=box_scale < 1,
        collapsed_chord=length <= 1e-12,
    )


def image_loss(union, target, weights, valid):
    return ((union - target) ** 2 * weights * valid).sum(dim=-1).mean(dim=-1)


def optimize_curve_offsets(engine, pose, targets, *, steps=100, learning_rate_mm=0.25):
    """Choose a finite iterate by fitting loss only, including the zero offset."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if not np.isfinite(learning_rate_mm) or learning_rate_mm <= 0:
        raise ValueError("positive finite learning rate required")
    pose = torch.as_tensor(pose, dtype=engine.dtype, device=engine.device)
    before_pose = pose.detach().clone()
    target = torch.as_tensor(targets.target, dtype=engine.dtype, device=engine.device)
    weights = torch.as_tensor(targets.weights, dtype=engine.dtype, device=engine.device)
    offset = torch.zeros((3, 3), dtype=engine.dtype, device=engine.device, requires_grad=True)
    constrained, _ = constrain_offsets(pose, offset)
    if not torch.equal(constrained, offset):
        raise AssertionError("zero offset must remain unchanged")
    optimizer = torch.optim.Adam([offset], lr=learning_rate_mm)
    best, best_loss, best_iteration = offset.detach().clone(), np.inf, -1
    losses, offsets, actions = [], [], []
    status = "completed"
    for iteration in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = engine.render(pose, tail_control_offset_mm=offset, sigma_reference_pose_xyz=pose)
        loss = image_loss(output.union_alpha, target, weights, output.ray_valid)[0]
        value = float(loss.detach())
        if not np.isfinite(value):
            status = "nonfinite_fit_stopped_with_best_finite_retained"
            break
        losses.append(value)
        offsets.append(offset.detach().cpu().numpy().copy())
        if value < best_loss:
            best_loss, best_iteration, best = value, iteration, offset.detach().clone()
        if iteration == steps:
            break
        if not bool(weights.any()):
            status = "neutral_no_fitting_foreground"
            break
        loss.backward()
        if offset.grad is None or not bool(torch.isfinite(offset.grad).all()):
            status = "nonfinite_gradient_stopped_with_best_finite_retained"
            break
        optimizer.step()
        if not bool(torch.isfinite(offset).all()):
            status = "nonfinite_adam_stopped_with_best_finite_retained"
            break
        with torch.no_grad():
            projected, action = constrain_offsets(pose, offset)
            offset.copy_(projected)
        actions.append({key: value.cpu().numpy().copy() for key, value in action.items()})
    if best_iteration < 0:
        raise FloatingPointError("no finite initial curve fitting loss")
    if not torch.equal(before_pose, pose):
        raise AssertionError("curve fitting modified fixed nodes")
    return dict(
        offset_mm=best.cpu().numpy(),
        best_loss=best_loss,
        best_iteration=best_iteration,
        loss_history=np.asarray(losses),
        offset_history_mm=np.asarray(offsets),
        constraint_history=actions,
        status=status,
        finite_iterates=len(losses),
        requested_steps=steps,
        completed_adam_steps=len(actions),
    )


def tensor_arrays(rendered):
    return dict(
        alpha=rendered.per_fish_alpha[0].detach().cpu().numpy(),
        visible=rendered.visible_alpha[0].detach().cpu().numpy(),
        ownership=rendered.ownership[0].detach().cpu().numpy(),
        union=rendered.union_alpha[0].detach().cpu().numpy(),
        valid=rendered.ray_valid.detach().cpu().numpy(),
    )


def all_source_paths():
    return sorted(set(measured.code_paths()) | {Path(__file__).resolve(), PROTOCOL})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=REPO / "work/three_body/persistent_mouth_continuous_one_minute_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    measured.require(output.is_relative_to(REPO), "output escaped workspace")
    if output.exists():
        raise FileExistsError("output exists; no overwrite")
    codes = {str(path): sha256(path) for path in all_source_paths()}
    print("Verifying sealed original geometry and no-GT inputs...", flush=True)
    source = identity_inputs._verify_run(args.run, appearance=False)
    hashes = {**source.hashes, **codes}
    original = identity_inputs._npz(source.root / "candidate.npz")
    measured.require(
        set(original) == identity_inputs.GEOMETRY_ARRAYS, "expected fixed 21-array geometry"
    )
    measured.require(
        np.array_equal(source.frames, np.arange(76620, 86580)), "expected completed minute"
    )
    cfg = source.params["tracker_config"]
    measured.require(
        cfg["mode"] == "detectron2"
        and [cfg[k] for k in ("positive_samples", "near_samples", "global_samples")]
        == [384, 384, 256],
        "fitting mode/samples changed",
    )
    for name in ("renderer.py", "observations.py", "prepare_sequence.py"):
        measured.require(
            sha256(LAB / name) == source.code[(LAB / name).relative_to(REPO).as_posix()],
            "live original renderer/sampling differs",
        )
    for name in ("src/medaka3d/calib.py", "src/medaka3d/types.py"):
        measured.require(
            sha256(REPO / name) == source.code[name], "live original calibration arithmetic differs"
        )
    rig = load_rig(source.root / "calibration_snapshot")
    measured.require(tuple(rig.views) == tuple(VIEWS), "wrong camera order")
    packed = {
        v: identity_inputs._npz(
            base.resolve_input(source.sequence, source.manifest["masks"][v]["path"])
        )
        for v in VIEWS
    }
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable; no silent backend switch")
    render_config = RendererConfig(width_ratio=cfg["width_ratio"])
    output.mkdir(parents=True, exist_ok=False)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    frame_results = []
    try:
        with provenance("fixed_node_curve_spatial_diagnostic", output, name="curve") as rec:
            for path, digest in hashes.items():
                rec.input(path, path, sha256=digest)
            for path, digest in codes.items():
                dst = output / "code_snapshot" / Path(path).relative_to(REPO)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dst)
                measured.require(sha256(dst) == digest, "snapshot mismatch")
                rec.output(str(dst.relative_to(output)), dst, sha256=digest)
            for frame in FRAMES:
                started = time.perf_counter()
                t = frame - int(source.frames[0])
                pose = original["pose_xyz"][t].copy()
                masks, pixels, foreground_flat = {}, {}, {}
                for view in VIEWS:
                    decoded = decode_frame_masks(packed[view], t)
                    measured.require(
                        array_sha256(decoded) == packed[view]["frame_mask_sha256"][t].decode(),
                        "native mask SHA mismatch",
                    )
                    masks[view] = decoded if packed[view]["view_available"][t] else None
                    measured.require(
                        packed[view]["view_available"][t] or not len(decoded),
                        "missing view contains instances",
                    )
                    union = decoded.any(axis=0) if len(decoded) else np.zeros((1200, 1200), bool)
                    flat = np.flatnonzero(union)
                    foreground_flat[view] = flat
                    pixels[view] = np.column_stack((flat % 1200, flat // 1200)).astype(float)
                dummy = np.zeros((1200, 1200, 3), np.uint8)
                obs = FrameObservation(frame, {}, dict.fromkeys(VIEWS, dummy), masks, {})
                fit_targets = sample_pixels(obs, "detectron2", 384, 384, 256)
                evaluation = sample_body_evaluation(obs, fit_targets.pixels)
                # Save the actual evaluation coordinates BEFORE the first optimizer update.
                sample_arrays = dict(
                    frame_idx=np.array(frame),
                    fit_uv=np.stack([fit_targets.pixels[v] for v in VIEWS]),
                    fit_target=fit_targets.target,
                    fit_weights=fit_targets.weights,
                    eval_target=evaluation.target,
                    eval_weights=evaluation.weights,
                    eval_strata=evaluation.strata,
                    eval_counts=evaluation.sampled_counts,
                    eval_pool_counts=evaluation.pool_counts,
                    eval_excluded_fit_unique_count=evaluation.excluded_fit_unique_count,
                    eval_view_available=evaluation.view_available,
                    eval_foreground_nonempty=evaluation.foreground_nonempty,
                    **{v + "_eval_uv": evaluation.pixels[v] for v in VIEWS},
                )
                sample_path = output / f"f{frame}_pixels.npz"
                np.savez_compressed(sample_path, **sample_arrays)
                rec.output(sample_path.name, sample_path, sha256=sha256(sample_path))
                engine = CurvedFishRenderer(
                    rig, fit_targets.pixels, render_config, dtype=torch.float64, device=device
                )
                old_engine = RefractiveFishRenderer(
                    rig, fit_targets.pixels, render_config, dtype=torch.float64, device=device
                )
                pose_tensor = torch.as_tensor(pose, dtype=torch.float64, device=device)
                with torch.no_grad():
                    zero_render = engine.render(
                        pose_tensor,
                        tail_control_offset_mm=np.zeros((3, 3)),
                        sigma_reference_pose_xyz=pose_tensor,
                    )
                    old_render = old_engine.render(pose_tensor)
                for key in (
                    "per_fish_alpha",
                    "union_alpha",
                    "visible_alpha",
                    "ownership",
                    "ray_valid",
                ):
                    measured.require(
                        torch.equal(getattr(zero_render, key), getattr(old_render, key)),
                        "zero curve differs from original renderer",
                    )
                zero = tensor_arrays(zero_render)
                zero_metrics = attribution.fixed_sample_metrics(
                    zero["alpha"],
                    zero["visible"],
                    zero["ownership"],
                    zero["valid"],
                    fit_targets.target,
                    fit_targets.weights,
                )
                saved_loss = float(original["loss_parts"][t, 4])
                measured.require(
                    abs(zero_metrics["mean_view_loss"] - saved_loss) < 1e-10,
                    "zero curve does not replay saved image loss",
                )
                fit_started = time.perf_counter()
                fitted = optimize_curve_offsets(engine, pose_tensor, fit_targets)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                fit_seconds = time.perf_counter() - fit_started
                offset = fitted["offset_mm"]
                with torch.no_grad():
                    best_render = engine.render(
                        pose_tensor,
                        tail_control_offset_mm=offset,
                        sigma_reference_pose_xyz=pose_tensor,
                    )
                for key in ("axial_sigma_mm", "radial_sigma_mm", "segment_lengths_mm"):
                    measured.require(
                        torch.equal(
                            getattr(best_render.bodies, key), getattr(old_render.bodies, key)
                        ),
                        "curve changed frozen Gaussian scales",
                    )
                best = tensor_arrays(best_render)
                fit_metrics = attribution.fixed_sample_metrics(
                    best["alpha"],
                    best["visible"],
                    best["ownership"],
                    best["valid"],
                    fit_targets.target,
                    fit_targets.weights,
                )
                measured.require(
                    abs(fit_metrics["mean_view_loss"] - fitted["best_loss"]) < 1e-12,
                    "best fit loss replay differs",
                )
                measured.require(
                    fitted["best_loss"] <= fitted["loss_history"][0],
                    "best retained result worse than zero",
                )
                if evaluation.target.shape[1]:
                    evaluator = CurvedFishRenderer(
                        rig, evaluation.pixels, render_config, dtype=torch.float64, device=device
                    )
                    with torch.no_grad():
                        before_eval = evaluator.render(
                            pose_tensor, sigma_reference_pose_xyz=pose_tensor
                        )
                        after_eval = evaluator.render(
                            pose_tensor,
                            tail_control_offset_mm=offset,
                            sigma_reference_pose_xyz=pose_tensor,
                        )
                    eval_before = tensor_arrays(before_eval)
                    eval_after = tensor_arrays(after_eval)
                else:
                    eval_before = eval_after = dict(
                        union=np.empty((3, 0)), valid=np.empty((3, 0), bool)
                    )
                metrics_before = summarize_pixel_error(
                    eval_before["union"], eval_before["valid"], evaluation
                )
                metrics_after = summarize_pixel_error(
                    eval_after["union"], eval_after["valid"], evaluation
                )
                diagnostics = tail_curve_diagnostics(pose, offset)
                diag = {
                    field.name: getattr(diagnostics, field.name).numpy()[0]
                    for field in dataclasses.fields(diagnostics)
                }
                diag["arc_chord_ratio"] = np.divide(
                    diag["arc_length_mm"],
                    diag["chord_length_mm"],
                    out=np.full(3, np.nan),
                    where=diag["chord_length_mm"] > 1e-12,
                )
                control = (pose[:, 1] + pose[:, 2]) / 2 + offset
                dot = np.sum(offset * (pose[:, 2] - pose[:, 1]), axis=-1)
                measured.require(np.max(np.abs(dot)) < 1e-9, "final offset no longer perpendicular")
                measured.require(
                    np.all(np.linalg.norm(offset, axis=-1) <= diag["chord_length_mm"] + 1e-10),
                    "offset exceeded chord bound",
                )
                measured.require(
                    np.all(control >= np.asarray(LOWER) - 1e-10)
                    and np.all(control <= np.asarray(UPPER) + 1e-10),
                    "control point outside box",
                )
                dense_views = {}
                dense_arrays = {}
                if any(len(value) for value in pixels.values()):
                    dense_engine = CurvedFishRenderer(
                        rig, pixels, render_config, dtype=torch.float64, device=device
                    )
                    with torch.no_grad():
                        dense0 = tensor_arrays(
                            dense_engine.render(pose_tensor, sigma_reference_pose_xyz=pose_tensor)
                        )
                        dense1 = tensor_arrays(
                            dense_engine.render(
                                pose_tensor,
                                tail_control_offset_mm=offset,
                                sigma_reference_pose_xyz=pose_tensor,
                            )
                        )
                    for v, view in enumerate(VIEWS):
                        n = len(pixels[view])
                        mask = (
                            masks[view]
                            if masks[view] is not None
                            else np.empty((0, 1200, 1200), bool)
                        )
                        rows_by_kind = {}
                        for name, value in (("original", dense0), ("curve", dense1)):
                            a, vis, valid = (
                                value["alpha"][v, :n],
                                value["visible"][v, :n],
                                value["valid"][v, :n],
                            )
                            rows, iou = attribution.instance_rows(
                                mask, foreground_flat[view], a, vis, valid
                            )
                            _, core, core_union = measured.core_metrics(
                                mask, foreground_flat[view], a, vis, valid
                            )
                            rows_by_kind[name] = dict(
                                instances=rows, cores=core, core_union=core_union, instance_iou=iou
                            )
                            dense_arrays[view + "_" + name + "_alpha"] = a
                        dense_views[view] = rows_by_kind
                        dense_arrays[view + "_foreground_uv"] = pixels[view]
                fit_arrays = dict(
                    pose_xyz=pose,
                    observed=original["observed"][t],
                    sources=original["sources"][t],
                    prior_identity_review=original["identity_review"][t],
                    best_offset_mm=offset,
                    control_xyz=control,
                    offset_dot_chord=dot,
                    loss_history=fitted["loss_history"],
                    offset_history_mm=fitted["offset_history_mm"],
                    original_fit_union=zero["union"],
                    curve_fit_union=best["union"],
                    fit_ray_valid=best["valid"],
                    original_eval_union=eval_before["union"],
                    curve_eval_union=eval_after["union"],
                    eval_ray_valid=eval_after["valid"],
                    **dense_arrays,
                    **{key: value for key, value in diag.items()},
                )
                for key in (
                    "applied_scalar",
                    "norm_scale",
                    "box_scale",
                    "parallel_removed_mm",
                    "norm_constraint_active",
                    "box_constraint_active",
                    "collapsed_chord",
                ):
                    fit_arrays["constraint_" + key] = np.asarray(
                        [action[key] for action in fitted["constraint_history"]]
                    )
                path = output / f"f{frame}_curve.npz"
                np.savez_compressed(path, **fit_arrays)
                rec.output(path.name, path, sha256=sha256(path))
                frame_result = dict(
                    frame_idx=frame,
                    status=fitted["status"],
                    best_iteration=fitted["best_iteration"],
                    completed_adam_steps=fitted["completed_adam_steps"],
                    fit_original=zero_metrics,
                    fit_curve=fit_metrics,
                    saved_original_loss=saved_loss,
                    spatial_evaluation_original=metrics_before,
                    spatial_evaluation_curve=metrics_after,
                    spatial_evaluation_counts=evaluation.sampled_counts,
                    spatial_evaluation_pool_counts=evaluation.pool_counts,
                    offset_mm=offset,
                    control_xyz=control,
                    curve_diagnostics=diag,
                    gaussian_nondegenerate=best_render.bodies.nondegenerate[0]
                    .detach()
                    .cpu()
                    .numpy(),
                    norm_constraint_active_iterations=np.sum(
                        fit_arrays["constraint_norm_constraint_active"], axis=0
                    ),
                    box_constraint_active_iterations=np.sum(
                        fit_arrays["constraint_box_constraint_active"], axis=0
                    ),
                    per_instance=dense_views,
                    zero_renderer_torch_equal=True,
                    frozen_scales_torch_equal=True,
                    fit_seconds=fit_seconds,
                    frame_seconds=time.perf_counter() - started,
                )
                frame_results.append(frame_result)
                print(
                    json.dumps(
                        measured.safe(
                            dict(
                                frame=frame,
                                zero_loss=zero_metrics["mean_view_loss"],
                                curve_loss=fit_metrics["mean_view_loss"],
                                evaluation_before=metrics_before["mean_view_loss"],
                                evaluation_after=metrics_after["mean_view_loss"],
                                best_iteration=fitted["best_iteration"],
                                elapsed_s=frame_result["frame_seconds"],
                            )
                        )
                    ),
                    flush=True,
                )
            summary = dict(
                status="completed_curve_only_diagnostic",
                frames=FRAMES,
                measurements=frame_results,
                device=str(device),
                dtype="float64",
                torch_threads=2,
                source_run=str(source.root),
                candidate_sha256=hashes[str(source.root / "candidate.npz")],
                source_manifest_sha256=hashes[str(source.sequence / "manifest.json")],
                protocol_sha256=codes[str(PROTOCOL)],
                renderer_config=dataclasses.asdict(render_config),
                curve_config=dict(
                    steps=100,
                    learning_rate_mm=0.25,
                    initial_offset="zero",
                    selection="minimum finite fitting loss including zero",
                    constraint="perpendicular, norm<=chord, single scalar control-box shrinking",
                    lower=LOWER,
                    upper=UPPER,
                ),
                all_saved_nodes_sources_reviews_preserved=True,
                sampled_evaluation_used_for_selection=False,
                dummy_image_usage="D2 mode image shape only; no RGB/background content read",
                native_raw_decodes=0,
                gt_or_baseline_bytes_opened=False,
                accepted_identity_or_trajectory_output=False,
                input_sha256_before=hashes,
                input_sha256_after=hashes,
                runtime=dict(
                    python=sys.version,
                    torch=str(torch.__version__),
                    opencv=cv2.__version__,
                    numpy=np.__version__,
                    cuda_device=torch.cuda.get_device_name() if device.type == "cuda" else None,
                ),
                limitations=[
                    "Pixels share the same D2 detector/frame; "
                    "not independent temporal or identity truth.",
                    "Only one quadratic tail curve per fish; "
                    "not every U/S shape or torsion can be represented.",
                    "Fixed possibly unsupported nodes remain fixed; "
                    "better silhouettes do not validate their 3D positions.",
                    "Arc length is not conserved or penalized; "
                    "this cannot reject all possible curved-body models.",
                    "Instance masks can overlap/merge/reflect; no instance-to-fish identity gate.",
                    "Original reviews remain unchanged; no new accepted track is produced.",
                ],
            )
            path = output / "summary.json"
            path.write_text(
                json.dumps(measured.safe(summary), indent=2, allow_nan=False), encoding="utf-8"
            )
            rec.output("summary", path, sha256=sha256(path))
            print("Final source/input/output SHA guards...", flush=True)
            measured.require(
                all(sha256(path) == digest for path, digest in hashes.items()),
                "input/source changed during diagnostic",
            )
            measured.require(
                all(sha256(item["path"]) == item["sha256"] for item in rec.outputs.values()),
                "output/snapshot changed during diagnostic",
            )
            rec.param("inputs_and_code_unchanged", True)
            rec.param("gt_or_baseline_bytes_opened", False)
    finally:
        torch.set_num_threads(old_threads)


if __name__ == "__main__":
    main()

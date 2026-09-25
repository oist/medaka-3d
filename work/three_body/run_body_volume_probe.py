"""Thirteen registered local body fits: image-volume starts versus random starts.

No annotations, baseline trajectories, identity repair or continuous tracking
are performed. Every outside-box trial remains recorded. Both arms share one
freshly assigned/fitted original scene and the exact frozen objective.
"""

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
sys.path.insert(0, str(LAB))

import run_experiment as base  # noqa: E402
import run_persistent_node_recovery as persistent  # noqa: E402
from body_local_fit import LocalBodyFitter, replay_chosen_parents  # noqa: E402
from body_pixel_evaluation import sample_body_evaluation, summarize_pixel_error  # noqa: E402
from fast_union import render_union  # noqa: E402
from identity_inputs import GEOMETRY_ARRAYS, _npz, _verify_run  # noqa: E402
from instance_volume_proposals import InstanceVolumeConfig, InstanceVolumeProposer  # noqa: E402
from mask_grid import MaskGrid, MaskGridConfig  # noqa: E402
from node_geometry import RayObservations, supported_nodes  # noqa: E402
from observations import CachedNodes, FrameObservation, sample_pixels  # noqa: E402
from prepare_sequence import array_sha256, decode_frame_masks  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402
from rigid_body_starts import (  # noqa: E402
    RigidStartConfig,
    random_scene_starts,
    volume_scene_starts,
)

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

FRAMES = (79316, 79318, 79322, 79355, 79370, 79372, 79381, 79394, 79396, 79405, 79547, 82719, 82720)
PARTS = ("node", "shape", "pectoral_motion", "repulsion", "silhouette")
DEFAULT_RUN = REPO / "work/three_body/persistent_mouth_continuous_one_minute_v1"
PROTOCOL = LAB / "BODY_PROPOSAL_PROTOCOL.md"


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if dataclasses.is_dataclass(value):
        return clean(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def source_paths():
    paths = {Path(__file__).resolve(), PROTOCOL}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(LAB) or path.is_relative_to(REPO / "src")
            ):
                paths.add(path)
    return sorted(paths)


def assert_hashes(hashes):
    for path, digest in hashes.items():
        if base.sha256(path) != digest:
            raise RuntimeError("body probe input/source/output changed: " + path)


def save_arrays(record, root, role, filename, arrays):
    if any(not isinstance(v, np.ndarray) or v.dtype.hasobject for v in arrays.values()):
        raise TypeError("saved body arrays must be explicit non-object NumPy arrays")
    path = root / filename
    np.savez_compressed(path, **arrays)
    record.output(role, path, sha256=base.sha256(path))
    return path


def save_json(record, root, role, filename, value):
    path = root / filename
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")
    record.output(role, path, sha256=base.sha256(path))
    return path


def decoded_probes(run, nodes):
    """Seek only these diagnostic frames; verify each decoded native BGR hash."""
    masks, backgrounds, captures = {}, {}, {}
    try:
        for view in base.VIEWS:
            masks[view] = _npz(
                base.resolve_input(run.sequence, run.manifest["masks"][view]["path"])
            )
            np.testing.assert_array_equal(masks[view]["frame_idx"], run.frames)
            backgrounds[view] = cv2.imread(
                str(base.resolve_input(run.sequence, run.manifest["backgrounds"][view]["path"]))
            )
            if backgrounds[view] is None or backgrounds[view].shape != (1200, 1200, 3):
                raise ValueError("background native shape differs")
            path = base.resolve_input(run.sequence, run.manifest["raw"][view]["path"])
            capture = cv2.VideoCapture(str(path))
            captures[view] = capture
            if (
                not capture.isOpened()
                or capture.get(cv2.CAP_PROP_FRAME_COUNT) != len(run.frames)
                or not np.isclose(capture.get(cv2.CAP_PROP_FPS), 166, rtol=0, atol=1e-6)
                or capture.get(cv2.CAP_PROP_FRAME_HEIGHT) != 1200
                or capture.get(cv2.CAP_PROP_FRAME_WIDTH) != 1200
            ):
                raise ValueError("raw clip does not match the declared native minute")
        for frame in FRAMES:
            t = int(np.searchsorted(run.frames, frame))
            if t >= len(run.frames) or run.frames[t] != frame:
                raise ValueError("registered probe is absent from the completed native geometry")
            images, decoded, evidence = {}, {}, {}
            for view in base.VIEWS:
                capture = captures[view]
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, t):
                    raise RuntimeError("raw decoder rejected native seek")
                ok, image = capture.read()
                if (
                    not ok
                    or image is None
                    or image.shape != (1200, 1200, 3)
                    or image.dtype != np.uint8
                    or not np.isclose(
                        capture.get(cv2.CAP_PROP_POS_FRAMES), t + 1, rtol=0, atol=1e-6
                    )
                ):
                    raise ValueError("native probe frame failed decoder shape/position guard")
                digest = array_sha256(image)
                if digest != run.manifest["raw"][view]["frame_bgr_sha256"][t]:
                    raise ValueError("native probe BGR SHA differs from original inference")
                data = masks[view]
                current = decode_frame_masks(data, t)
                mask_sha = array_sha256(current)
                if mask_sha != data["frame_mask_sha256"][t].decode("ascii"):
                    raise ValueError("decoded masks differ from original row-ordered mask SHA")
                available = bool(data["view_available"][t])
                if not available and len(current):
                    raise ValueError("unavailable view contains detector instances")
                images[view], decoded[view] = image, current if available else None
                lo, hi = data["frame_offsets"][t : t + 2]
                evidence[view] = dict(
                    bgr_sha256=digest,
                    decoded_mask_sha256=mask_sha,
                    available=available,
                    instances=len(current),
                    native_shape=list(image.shape),
                    instance_id=data["instance_id"][lo:hi].tolist(),
                    scores=data["scores"][lo:hi].tolist(),
                    predicted_boxes=data["pred_boxes"][lo:hi].tolist(),
                    classes=data["classes"][lo:hi].tolist(),
                    source_track_id=data["source_track_id"][lo:hi].tolist(),
                )
            yield (
                t,
                FrameObservation(
                    frame, {v: nodes.at(v, frame) for v in base.VIEWS}, images, decoded, backgrounds
                ),
                evidence,
            )
    finally:
        for capture in captures.values():
            capture.release()


def combine_admissible(mask, random):
    """Keep one common original, explicit rejected trials, and arm-to-fit maps."""
    np.testing.assert_array_equal(mask["pose_xyz"][0], random["pose_xyz"][0])
    np.testing.assert_array_equal(mask["component_slot_sign_roll"][1:, 1], random["raw_slot"][1:])
    if not mask["admissible"][0] or not random["admissible"][0]:
        raise ValueError("the common original scene must be admitted")
    poses, mappings = [mask["pose_xyz"][:1]], {}
    cursor = 1
    for name, arm in (("mask", mask), ("random", random)):
        if not np.array_equal(arm["admissible"], ~arm["outside_coordinates"].any((1, 2, 3))):
            raise ValueError("outside-box admission flags disagree")
        selected = np.flatnonzero(arm["admissible"])[1:]
        mapping = np.full(len(arm["pose_xyz"]), -1, np.int64)
        mapping[0] = 0
        mapping[selected] = np.arange(cursor, cursor + len(selected))
        poses.append(arm["pose_xyz"][selected])
        cursor += len(selected)
        mappings[name] = mapping
    return np.concatenate(poses), mappings


def unique_pose_count(poses):
    """Rounded geometric diversity only, not statistically independent starts."""
    return len(np.unique(np.rint(np.asarray(poses).reshape(len(poses), -1) / 1e-9), axis=0))


def component_payload(pool):
    offsets = np.r_[0, np.cumsum([len(c.voxel_indices) for c in pool.components])].astype(np.int64)
    arrays = dict(
        voxel_offsets=offsets,
        voxel_indices=(
            np.concatenate([c.voxel_indices for c in pool.components])
            if pool.components
            else np.empty(0, np.int64)
        ),
    )
    descriptions = []
    for component in pool.components:
        item = dataclasses.asdict(component)
        item["voxel_indices"] = dict(
            count=len(component.voxel_indices), sha256=array_sha256(component.voxel_indices)
        )
        descriptions.append(item)
    return arrays, dict(
        components=descriptions,
        top_diagnostics=pool.top_diagnostics,
        omitted=pool.omitted,
        view_available=pool.view_available,
        view_has_foreground=pool.view_has_foreground,
        diagnostics=pool.diagnostics,
    )


def target_arrays(fitting, evaluation):
    arrays = dict(
        fit_target=fitting.target, fit_weights=fitting.weights, fit_foreground=fitting.foreground
    )
    for view in base.VIEWS:
        arrays["fit_pixels_" + view] = fitting.pixels[view]
        arrays["evaluation_pixels_" + view] = evaluation.pixels[view]
        if len(evaluation.pixels[view]):
            fit = fitting.pixels[view]
            held = evaluation.pixels[view]
            if np.intersect1d(fit[:, 1] * 1200 + fit[:, 0], held[:, 1] * 1200 + held[:, 0]).size:
                raise ValueError("evaluation contains a fitting pixel")
    for field in dataclasses.fields(evaluation):
        value = getattr(evaluation, field.name)
        if isinstance(value, np.ndarray):
            arrays["evaluation_" + field.name] = value
    return arrays


def evaluate_poses(rig, poses, targets, config):
    if targets.target.shape[1] == 0:
        union, valid = np.zeros((len(poses), 3, 0)), np.zeros((3, 0), bool)
    else:
        renderer = RefractiveFishRenderer(
            rig,
            targets.pixels,
            RendererConfig(width_ratio=config["width_ratio"]),
            device=config["device"],
            dtype=torch.float64,
        )
        with torch.no_grad():
            union = np.concatenate(
                [
                    render_union(
                        renderer,
                        torch.as_tensor(
                            poses[start : start + 24], dtype=torch.float64, device=config["device"]
                        ),
                    )
                    .cpu()
                    .numpy()
                    for start in range(0, len(poses), 24)
                ]
            )
        valid = renderer.ray_valid.cpu().numpy()
    return {**summarize_pixel_error(union, valid, targets), "union": union, "ray_valid": valid}


def initial_support(rig, fitted, rays):
    """Direct geometry evidence at each start under its own new, fixed plan."""
    rows = [
        supported_nodes(pose, plan, rays, rig)
        for pose, plan in zip(fitted["input_pose_xyz"], fitted["plans"], strict=True)
    ]
    return dict(
        initial_sources=np.stack([row[0] for row in rows]),
        initial_observed=np.stack([row[1] for row in rows]),
        initial_ray_distance_mm=np.stack([row[2] for row in rows]),
    )


def instance_mass(rig, poses, frame, config):
    """All foreground pixels of each D2 union, reused across overlapping instances."""
    pixels = {}
    for view in base.VIEWS:
        masks = frame.masks[view]
        union = np.zeros((1200, 1200), bool) if masks is None else masks.any(0)
        y, x = np.nonzero(union)
        pixels[view] = np.column_stack((x, y)).astype(float)
    if not any(len(value) for value in pixels.values()):
        return []
    renderer = RefractiveFishRenderer(
        rig,
        pixels,
        RendererConfig(width_ratio=config["width_ratio"]),
        device=config["device"],
        dtype=torch.float64,
    )
    with torch.no_grad():
        rendered = renderer.render(
            torch.as_tensor(poses, dtype=torch.float64, device=config["device"])
        )
    alpha = rendered.per_fish_alpha.cpu().numpy()
    visible = rendered.visible_alpha.cpu().numpy()
    union = rendered.union_alpha.cpu().numpy()
    valid = rendered.ray_valid.cpu().numpy()
    if not np.allclose(visible.sum(-1), union, rtol=1e-12, atol=1e-12):
        raise ValueError("visible body mass does not reproduce union mass")
    rows = []
    for vi, view in enumerate(base.VIEWS):
        masks = frame.masks[view]
        if masks is None:
            continue
        xy = pixels[view].astype(int)
        for row, mask in enumerate(masks):
            member = np.zeros(union.shape[2], bool)
            member[: len(xy)] = mask[xy[:, 1], xy[:, 0]]
            member &= valid[vi]
            a = alpha[:, vi, member]
            contribution = np.stack(
                [
                    a[..., fish] * np.prod(1 - a[..., [f for f in range(3) if f != fish]], axis=-1)
                    for fish in range(3)
                ],
                axis=-1,
            ).sum(1)
            rows.append(
                dict(
                    view=view,
                    instance_row=row,
                    mask_pixels=int(mask.sum()),
                    valid_pixels=int(member.sum()),
                    unexplained_mass=(1 - union[:, vi, member]).sum(1),
                    per_body_alpha_mass=a.sum(1),
                    visible_mass=visible[:, vi, member].sum(1),
                    unique_union_mass=contribution,
                    pair_shared_mass=np.stack(
                        [(a[..., i] * a[..., j]).sum(1) for i, j in ((0, 1), (0, 2), (1, 2))],
                        axis=-1,
                    ),
                    note="overlapping D2 instances are not additive or independent fish labels",
                )
            )
    return rows


def preview(rig, frame, poses, supports, labels):
    colors = ((255, 110, 0), (0, 210, 255), (255, 40, 210))
    panels, native = [[] for _ in poses], {}
    for view in base.VIEWS:
        uv = rig.camera(view).project(poses)
        finite = uv.reshape(-1, 2)
        finite = finite[np.isfinite(finite).all(1)]
        masks = frame.masks[view]
        if masks is not None and masks.any():
            y, x = np.nonzero(masks.any(0))
            finite = np.concatenate([finite, [[x.min(), y.min()], [x.max(), y.max()]]])
        if len(finite):
            low = np.maximum(np.floor(finite.min(0) - 35).astype(int), 0)
            high = np.minimum(np.ceil(finite.max(0) + 35).astype(int), [1200, 1200])
        else:
            low, high = np.array([0, 0]), np.array([1200, 1200])
        if np.any(high <= low):
            low, high = np.array([0, 0]), np.array([1200, 1200])
        native[view] = dict(crop_xyxy=[*low, *high], projected_nodes=uv)
        for candidate, pose_uv in enumerate(uv):
            image = frame.images_bgr[view].copy()
            if masks is not None:
                for row, mask in enumerate(masks):
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(image, contours, -1, (130, 130, 130), 1)
                    if contours:
                        corner = min(
                            (point[0] for c in contours for point in c), key=lambda p: p[0]
                        )
                        cv2.putText(
                            image,
                            f"D2 row {row}",
                            tuple(corner),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.33,
                            (230, 230, 230),
                            1,
                        )
            for fish in range(3):
                points = pose_uv[fish]
                for a, b in ((0, 1), (1, 2)):
                    if np.isfinite(points[[a, b]]).all():
                        cv2.line(
                            image,
                            tuple(np.rint(points[a]).astype(int)),
                            tuple(np.rint(points[b]).astype(int)),
                            colors[fish],
                            2,
                        )
                for node, point in enumerate(points):
                    if not np.isfinite(point).all():
                        continue
                    xy = tuple(np.rint(point).astype(int))
                    if supports[candidate, fish, node]:
                        cv2.circle(image, xy, 4, colors[fish], 1)
                    else:
                        cv2.drawMarker(image, xy, colors[fish], cv2.MARKER_TILTED_CROSS, 8, 1)
                if np.isfinite(points[1]).all():
                    cv2.putText(
                        image,
                        f"S{fish}",
                        tuple(np.rint(points[1] + 5).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        colors[fish],
                        1,
                    )
            image = image[low[1] : high[1], low[0] : high[0]]
            scale = min(600 / image.shape[1], 310 / image.shape[0])
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            panel = np.full((350, 610, 3), 25, np.uint8)
            panel[35 : 35 + image.shape[0], : image.shape[1]] = image
            cv2.putText(
                panel,
                f"{frame.frame_idx} {view} {labels[candidate]}",
                (4, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (240, 240, 240),
                1,
            )
            panels[candidate].append(panel)
    return np.vstack([np.hstack(row) for row in panels]), native


def write_report(path, summary):
    lines = [
        "# Image-volume body starts: fixed local comparison",
        "",
        "Each row uses one unchanged saved scene and fresh exclusive SLEAP assignments.",
        "Scores choose candidates; evaluation pixels do not select them.",
        "",
        "| Frame | Mask fit / attempts | Random fit / attempts | Original score | "
        "Refit score | Mask best | Random best | Eval original/refit/mask/random |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["frames"]:
        arm = row["arms"]
        scores = row["selected"]["fit_scores"]
        evaluations = "/".join(f"{v:.5f}" for v in row["selected"]["evaluation_mean_view_loss"])
        lines.append(
            f"| {row['frame']} | {arm['mask']['fitted']}/{arm['mask']['nominal']} | "
            f"{arm['random']['fitted']}/{arm['random']['nominal']} | "
            f"{scores[0]:.5f} | {scores[1]:.5f} | {scores[2]:.5f} | {scores[3]:.5f} | "
            f"{evaluations} |"
        )
    lines += [
        "",
        "Per-camera disjoint foreground / background weighted error follows.",
        "The order is original, refitted original, mask best and random best.",
        "",
        "| Frame | Camera | Foreground (four scenes) | Background (four scenes) |",
        "|---|---|---|---|",
    ]
    for row in summary["frames"]:
        foreground = np.asarray(row["selected"]["evaluation_foreground_loss_by_view"])
        background = np.asarray(row["selected"]["evaluation_background_loss_by_view"])
        for vi, view in enumerate(base.VIEWS):
            fg = "/".join(f"{v:.5f}" for v in foreground[:, vi])
            bg = "/".join(f"{v:.5f}" for v in background[:, vi])
            lines.append(f"| {row['frame']} | {view} | {fg} | {bg} |")
    lines += [
        "",
        "- Nominal attempts include the shared original. Outside trials are not redrawn.",
        "- One trial changes one body; the other bodies can move during the joint fit.",
        "- Evaluation pixels share the same frame and detector; this is not independent ID truth.",
        "- The 2-mm components may fragment, merge or miss fish and include reflections/ghosts.",
        "- No identity, review flag, saved trajectory or anatomical measurement is corrected.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(REPO) or output.exists():
        raise FileExistsError("a new workspace output directory is required")
    output.mkdir(parents=True)
    old_torch_threads, old_cv_threads = torch.get_num_threads(), cv2.getNumThreads()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    try:
        with provenance("three_body_local_volume_start_probe", output, name="experiment") as record:
            record.params_from(args)
            code = {str(path): base.sha256(path) for path in source_paths()}
            for path, digest in code.items():
                target = output / "code_snapshot" / Path(path).relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                if base.sha256(target) != digest:
                    raise RuntimeError("source changed during the initial snapshot")
                record.output("code:" + str(Path(path).relative_to(REPO)), target, sha256=digest)
            run = _verify_run(args.run, appearance=False)
            hashes = {**run.hashes, **code}
            for path, digest in hashes.items():
                record.input(path, path, sha256=digest)
            if not np.array_equal(run.frames, np.arange(76620, 86580, dtype=np.int64)):
                raise ValueError("body experiment requires the full registered persistent minute")
            for relative, digest in run.code.items():
                live = str(REPO / relative)
                if live in code and code[live] != digest:
                    raise ValueError("a reused frozen tracker source differs from the producer")
            geometry = _npz(run.root / "candidate.npz")
            if set(geometry) != GEOMETRY_ARRAYS:
                raise ValueError("body probe requires exactly the current 21 geometry arrays")
            persistent.validate_arm_arrays(geometry, run.frames)
            nodes = CachedNodes(run.input_root, base.EXPERIMENT, run.frames)
            base.validate_result(geometry, run.frames, nodes)
            cfg = run.params["tracker_config"]
            expected = dict(
                mode="detectron2",
                device="cuda",
                opt_steps=14,
                learning_rate_mm=0.25,
                width_ratio=0.06,
                positive_samples=384,
                near_samples=384,
                global_samples=256,
            )
            if any(cfg.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    "saved tracker does not match the registered fourteen-step objective"
                )
            record.param("tracker_config", cfg)
            record.param("registered_native_probes", list(FRAMES))
            record.param("batch_size", 24)
            parents = replay_chosen_parents(geometry, run.initial)
            rig = load_rig(run.root / "calibration_snapshot")
            grid_started = time.perf_counter()
            grid = MaskGrid(rig, MaskGridConfig())
            proposer = InstanceVolumeProposer(
                grid.xyz,
                grid.projected_uv,
                grid.grid_shape,
                image_shape_hw=grid.config.image_shape_hw,
            )
            grid_seconds = time.perf_counter() - grid_started
            save_arrays(
                record,
                output,
                "grid",
                "grid.npz",
                dict(
                    xyz=grid.xyz,
                    projected_uv=grid.projected_uv,
                    grid_shape=np.asarray(grid.grid_shape, np.int64),
                ),
            )
            fitter = LocalBodyFitter(rig, run.initial, cfg, batch_size=24)
            reports = []
            for t, frame, raw_evidence in decoded_probes(run, nodes):
                started = time.perf_counter()
                root = output / f"frame_{frame.frame_idx}"
                root.mkdir()
                scene = geometry["pose_xyz"][t]
                targets = sample_pixels(
                    frame,
                    mode=cfg["mode"],
                    n_positive=cfg["positive_samples"],
                    n_near=cfg["near_samples"],
                    n_global=cfg["global_samples"],
                )
                evaluation = sample_body_evaluation(frame, targets.pixels)
                save_arrays(
                    record,
                    root,
                    f"targets/{frame.frame_idx}",
                    "sampled_targets.npz",
                    target_arrays(targets, evaluation),
                )
                observations = {}
                for view in base.VIEWS:
                    observations["sleap_" + view] = frame.detections[view]
                    value = frame.masks[view]
                    observations["masks_" + view] = (
                        np.empty((0, 1200, 1200), bool) if value is None else value
                    )
                    observations["available_" + view] = np.asarray(value is not None)
                save_arrays(
                    record,
                    root,
                    f"observations/{frame.frame_idx}",
                    "observations.npz",
                    observations,
                )
                pool = proposer.evaluate(frame.masks)
                component_arrays, component_info = component_payload(pool)
                save_arrays(
                    record,
                    root,
                    f"components/{frame.frame_idx}",
                    "components.npz",
                    component_arrays,
                )
                save_json(
                    record,
                    root,
                    f"component_metadata/{frame.frame_idx}",
                    "components.json",
                    component_info,
                )
                usable = [i for i, c in enumerate(pool.components) if c.principal_axis_defined]
                starts = volume_scene_starts(
                    scene,
                    [
                        dict(
                            centroid=pool.components[i].centroid_xyz,
                            axes=pool.components[i].pca_axes,
                        )
                        for i in usable
                    ],
                )
                random = random_scene_starts(
                    scene,
                    starts["component_slot_sign_roll"][1:, 1],
                    seed=frame.frame_idx * 7 + 9101,
                )
                all_poses, arm_maps = combine_admissible(starts, random)
                rays = RayObservations.from_points(rig, frame.detections)
                fitted = fitter.run(
                    all_poses,
                    parents["predicted_pose_xyz"][t],
                    parents["parent_missing"][t],
                    rays,
                    targets,
                )
                fitted.update(initial_support(rig, fitted, rays))
                np.testing.assert_array_equal(fitted["input_pose_xyz"][0], scene)
                choices = {
                    name: int(
                        mapping[mapping >= 0][np.argmin(fitted["scores"][mapping[mapping >= 0]])]
                    )
                    for name, mapping in arm_maps.items()
                }
                evaluation_poses = np.concatenate((scene[None], fitted["pose_xyz"]))
                metrics = evaluate_poses(rig, evaluation_poses, evaluation, cfg)
                indices = [0, 1, choices["mask"] + 1, choices["random"] + 1]
                selected_poses = evaluation_poses[indices]
                selected_support = np.stack(
                    (
                        fitted["initial_observed"][0],
                        fitted["observed"][0],
                        fitted["observed"][choices["mask"]],
                        fitted["observed"][choices["random"]],
                    )
                )
                selected_labels = [
                    "original fresh sources",
                    "refit original",
                    "mask best",
                    "random best",
                ]
                mass = instance_mass(rig, selected_poses, frame, cfg)
                image, projection = preview(
                    rig, frame, selected_poses, selected_support, selected_labels
                )
                image_path = root / "preview.jpg"
                if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise RuntimeError("preview write failed")
                record.output(
                    f"preview/{frame.frame_idx}", image_path, sha256=base.sha256(image_path)
                )
                arrays = {
                    "fit_" + key: value
                    for key, value in fitted.items()
                    if isinstance(value, np.ndarray)
                }
                arrays.update({"evaluation_" + key: value for key, value in metrics.items()})
                arrays.update(
                    predicted_pose_xyz=parents["predicted_pose_xyz"][t],
                    parent_velocity=parents["parent_velocity"][t],
                    parent_missing=parents["parent_missing"][t],
                    original_pose_xyz=scene,
                    original_observed=geometry["observed"][t],
                    original_review=geometry["identity_review"][t],
                    oriented_component_indices=np.asarray(usable, np.int64),
                    selected_evaluation_indices=np.asarray(indices, np.int64),
                )
                for name, arm in (("mask", starts), ("random", random)):
                    arrays.update(
                        {
                            name + "_" + key: value
                            for key, value in arm.items()
                            if isinstance(value, np.ndarray)
                        }
                    )
                    arrays[name + "_fit_index"] = arm_maps[name]
                save_arrays(record, root, f"candidates/{frame.frame_idx}", "candidates.npz", arrays)
                fit_scores = [
                    float(fitted["initial_scores"][0]),
                    float(fitted["scores"][0]),
                    float(fitted["scores"][choices["mask"]]),
                    float(fitted["scores"][choices["random"]]),
                ]
                summary = dict(
                    frame=frame.frame_idx,
                    raw_evidence=raw_evidence,
                    selected_labels=selected_labels,
                    component_count=len(pool.components),
                    oriented_components=len(usable),
                    skipped_undefined_axis_components=[
                        i for i in range(len(pool.components)) if i not in usable
                    ],
                    arms={
                        name: dict(
                            nominal=len(arm["pose_xyz"]),
                            fitted=int(arm["admissible"].sum()),
                            outside_rejected=int((~arm["admissible"]).sum()),
                            unique_rounded_1e_minus_9_mm_nominal=unique_pose_count(arm["pose_xyz"]),
                            unique_rounded_1e_minus_9_mm_admitted=unique_pose_count(
                                arm["pose_xyz"][arm["admissible"]]
                            ),
                        )
                        for name, arm in (("mask", starts), ("random", random))
                    },
                    fit_seconds=fitted["elapsed_seconds"],
                    joint_admitted_fit_count=len(all_poses),
                    frame_total_seconds=time.perf_counter() - started,
                    random_seed=random["seed"],
                    batch_size=fitted["batch_size"],
                    selected=dict(
                        fit_indices=choices,
                        fit_scores=fit_scores,
                        direct_node_counts=selected_support.sum((1, 2)),
                        evaluation_mean_view_loss=metrics["mean_view_loss"][indices],
                        evaluation_loss_by_view=metrics["loss_by_view"][indices],
                        evaluation_loss_by_stratum=metrics["loss_by_stratum"][indices],
                        evaluation_foreground_loss_by_view=(
                            metrics["foreground_loss_by_view"][indices]
                        ),
                        evaluation_background_loss_by_view=(
                            metrics["background_loss_by_view"][indices]
                        ),
                    ),
                    selected_instance_mass=mass,
                    preview_native_coordinates=projection,
                    candidate_ranking_uses_fitting_scores_only=True,
                    both_arms_share_one_refitted_original=True,
                    old_review_cleared=False,
                    preview_direct_support="new local plans; old source support saved separately",
                    biological_id_verdict="unverified",
                )
                save_json(record, root, f"frame_summary/{frame.frame_idx}", "summary.json", summary)
                reports.append(summary)
                print(
                    json.dumps(dict(frame=frame.frame_idx, completed=len(reports), requested=13)),
                    flush=True,
                )
            if [row["frame"] for row in reports] != list(FRAMES):
                raise ValueError("not all thirteen registered probes completed")
            summary = dict(
                status="completed_registered_local_body_volume_probe",
                frames=reports,
                native_frame_probes=list(FRAMES),
                original_geometry_run=str(run.root),
                inputs_verified_without_loading_gt_or_baseline=True,
                tracker_config=cfg,
                grid_config=dataclasses.asdict(grid.config),
                component_config=dataclasses.asdict(InstanceVolumeConfig()),
                rigid_config=dataclasses.asdict(RigidStartConfig()),
                grid_seconds=grid_seconds,
                fit_loss_parts=PARTS,
                fit_score_replay_tolerance=dict(rtol=1e-12, atol=1e-12),
                gpu_device=torch.cuda.get_device_name(),
                torch_version=str(torch.__version__),
                numpy_version=np.__version__,
                opencv_version=cv2.__version__,
                torch_cpu_threads=torch.get_num_threads(),
                opencv_threads=cv2.getNumThreads(),
                continuous_tracking_performed=False,
                original_reviews_cleared=False,
                sampling_holdout="within-frame disjoint pixels, same D2 detector; not ID truth",
                arm_compute_budget="matched nominal attempts; admissible computation differs",
            )
            save_json(record, output, "summary", "summary.json", summary)
            report_path = output / "RESULT.md"
            write_report(report_path, clean(summary))
            record.output("report", report_path, sha256=base.sha256(report_path))
            assert_hashes(
                {**hashes, **{item["path"]: item["sha256"] for item in record.outputs.values()}}
            )
            record.param("source_input_snapshot_output_sha_unchanged", True)
            record.param("ground_truth_and_baseline_trajectories_loaded", False)
    finally:
        torch.set_num_threads(old_torch_threads)
        cv2.setNumThreads(old_cv_threads)
    return summary


if __name__ == "__main__":
    main()

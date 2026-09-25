"""Isolated CPU view concurrency for the frozen local-flow observation equations.

The caller owns a three-worker ThreadPoolExecutor and keeps both frames, pose,
rig and configuration unchanged until this call returns. Workers only write
their private five-field image-motion results. The caller merges VIEWS in order
and performs the original serial ray/subset geometry. No OpenCV global setting,
tracker integration, GPU operation, file read or performance claim occurs here.
All submitted workers finish before a result or worker exception is returned.

Source binding is the caller's responsibility, as for the frozen flow helper.
"""

from concurrent.futures import wait
from itertools import combinations

import motion as frozen
import numpy as np

from medaka3d.calib import triangulate
from medaka3d.types import VIEWS

VERSION = "parallel_flow_views_v1"
FROZEN_MOTION_SHA256 = "3ed3c4fbfa39a7c1df5804453423a76dd9d9ff064e66ab45a5beda1696852f85"


def _view(previous, current, rig, view, pose, config):
    """Exact old within-view order; return only private fish/node arrays."""
    uv = np.full((3, 3, 2), np.nan)
    quality = np.zeros((3, 3))
    count = np.zeros((3, 3), dtype=np.int32)
    fb = np.full((3, 3), np.nan)
    spread = np.full_like(fb, np.nan)
    before, after = frozen._gray(previous.images_bgr[view]), frozen._gray(current.images_bgr[view])
    if before.shape != after.shape:
        raise ValueError("consecutive raw image dimensions differ")
    independent = frozen._mask_union(previous.masks[view], before.shape)
    current_union = frozen._mask_union(current.masks[view], before.shape)
    if independent is None or not independent.any():
        return uv, quality, count, fb, spread
    projected = rig.camera(view).project(pose)
    capsules = frozen._capsules(projected, before.shape, config.body_radius_px)
    for fish in range(3):
        tracks = frozen._tracked_features(before, after, capsules[fish] & independent, config)
        if tracks is None:
            continue
        for node in range(3):
            result = frozen._local_warp(projected[fish, node], tracks, current_union, config)
            if result is None:
                continue
            (
                uv[fish, node],
                quality[fish, node],
                count[fish, node],
                fb[fish, node],
                spread[fish, node],
            ) = result
    return uv, quality, count, fb, spread


def _views(executor, previous, current, rig, pose, config):
    futures = []
    try:
        for view in VIEWS:
            futures.append(executor.submit(_view, previous, current, rig, view, pose, config))
    finally:
        # Also drains already submitted work if a later submit/camera lookup fails.
        wait(futures)
    # All completed first: the first exception is chosen in fixed VIEWS order,
    # even if a different camera failed or completed earlier in wall-clock time.
    return [future.result() for future in futures]


def from_frames_parallel(previous, current, rig, previous_pose_xyz, config=None, *, executor):
    """Return the original FlowObservations type, using caller-owned CPU workers.

    No frame buffers are copied: the same input-lifetime guarantee required by
    the synchronous frozen function extends until all three workers have joined.
    The executor is neither created nor shut down here and must not be the same
    saturated pool whose worker is calling this function.
    """
    config = config or frozen.FlowConfig()
    if current.frame_idx != previous.frame_idx + 1:
        raise ValueError("flow proposals require consecutive native source frames")
    pose = np.asarray(previous_pose_xyz, dtype=float)
    if pose.shape != (3, 3, 3):
        raise ValueError("previous pose must have shape (three fish, three nodes, xyz)")
    uv = np.full((3, 3, 3, 2), np.nan)
    quality = np.zeros((3, 3, 3))
    count = np.zeros((3, 3, 3), dtype=np.int32)
    fb = np.full((3, 3, 3), np.nan)
    spread = np.full_like(fb, np.nan)
    results = _views(executor, previous, current, rig, pose, config)
    for vi, values in enumerate(results):
        for destination, value in zip((uv, quality, count, fb, spread), values, strict=True):
            destination[:, vi] = value
    # Original serial post-view equations and enumeration order, unchanged.
    valid = np.isfinite(uv).all(axis=-1) & (quality > 0)
    rays = [rig.camera(view).rays(uv[:, vi]) for vi, view in enumerate(VIEWS)]
    origins = np.stack([value[0] for value in rays], axis=1)
    directions = np.stack([value[1] for value in rays], axis=1)
    valid &= np.isfinite(origins).all(axis=-1) & np.isfinite(directions).all(axis=-1)
    quality = np.where(valid, quality, 0.0)
    node_pose = np.full((3, 3, 3), np.nan)
    node_valid = np.zeros((3, 3), bool)
    used = np.zeros((3, 3, 3), bool)
    pose_quality = np.zeros((3, 3))
    residuals = np.full((3, 3), np.nan)
    for fish in range(3):
        for node in range(3):
            available = np.flatnonzero(valid[fish, :, node])
            candidates = []
            for number in (3, 2):
                for subset in combinations(available, number):
                    selected = np.asarray(subset)
                    origin, direction = (
                        origins[fish, selected, node],
                        directions[fish, selected, node],
                    )
                    xyz, residual = triangulate(origin, direction)
                    if (
                        not np.isfinite(xyz).all()
                        or not np.isfinite(residual)
                        or residual > config.maximum_ray_residual_mm
                        or np.linalg.norm(xyz - pose[fish, node]) > config.maximum_node_step_mm
                        or np.any(np.sum((xyz - origin) * direction, axis=-1) < 0)
                    ):
                        continue
                    errors = [
                        np.linalg.norm(rig.camera(VIEWS[vi]).project(xyz) - uv[fish, vi, node])
                        for vi in selected
                    ]
                    if (
                        not np.isfinite(errors).all()
                        or max(errors) > config.maximum_reprojection_px
                    ):
                        continue
                    factor = float(quality[fish, selected, node].mean() * number / 3)
                    candidates.append((-number, float(residual), -factor, xyz, selected, factor))
            if candidates:
                candidates.sort(key=lambda value: value[:3])
                _, residual, _, xyz, selected, factor = candidates[0]
                node_pose[fish, node], node_valid[fish, node] = xyz, True
                used[fish, selected, node] = True
                pose_quality[fish, node], residuals[fish, node] = factor, residual
    arrays = (
        uv,
        quality,
        count,
        fb,
        spread,
        valid,
        origins,
        directions,
        node_pose,
        node_valid,
        used,
        pose_quality,
        residuals,
    )
    for array in arrays:
        array.setflags(write=False)
    return frozen.FlowObservations(int(previous.frame_idx), int(current.frame_idx), *arrays)

"""Actual descending-frame optical flow; proposals and audit records, never IDs.

The numerical LK/triangulation rules match motion.py. This diagnostic retains
rejected-feature information and requires current.frame_idx == previous.frame_idx - 1.
It never renumbers frames to pass the forward implementation's guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
from motion import FlowConfig, FlowObservations, _capsules, _gray, _local_warp, _mask_union

from medaka3d.calib import triangulate
from medaka3d.types import VIEWS


def _readonly(value):
    value = np.asarray(value)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class FeatureAudit:
    view: str
    fish: int
    reason: str
    usable: bool
    previous_uv: np.ndarray
    current_uv: np.ndarray
    returned_uv: np.ndarray
    forward_valid: np.ndarray
    backward_valid: np.ndarray
    lk_error: np.ndarray
    fb_error_px: np.ndarray
    step_px: np.ndarray
    feature_valid: np.ndarray


def _tracked_features_audit(previous, current, feature_mask, cfg, view, fish):
    """Same crop, LK and acceptance arithmetic as motion._tracked_features."""
    start = np.empty((0, 2))
    stop = returned = start.copy()
    fgood = bgood = valid = np.empty(0, bool)
    error = fb = step = np.empty(0)

    def finish(reason, usable=False):
        arrays = (start, stop, returned, fgood, bgood, error, fb, step, valid)
        audit = FeatureAudit(view, fish, reason, usable, *map(_readonly, arrays))
        tracks = (start[valid], stop[valid], fb[valid], error[valid]) if usable else None
        return tracks, audit

    left, top, width, height = cv2.boundingRect(feature_mask.astype(np.uint8))
    if (
        width == 0
        or np.count_nonzero(feature_mask[top : top + height, left : left + width])
        < cfg.node_min_features
    ):
        return finish("empty_or_too_small_feature_mask")
    pad = cfg.search_padding_px
    x0, y0 = max(0, left - pad), max(0, top - pad)
    x1, y1 = min(previous.shape[1], left + width + pad), min(previous.shape[0], top + height + pad)
    before = np.ascontiguousarray(previous[y0:y1, x0:x1])
    after = np.ascontiguousarray(current[y0:y1, x0:x1])
    allowed = np.ascontiguousarray(feature_mask[y0:y1, x0:x1], dtype=np.uint8)
    points = cv2.goodFeaturesToTrack(
        before,
        cfg.max_corners,
        cfg.corner_quality,
        cfg.corner_spacing_px,
        mask=allowed,
        blockSize=3,
    )
    if points is None:
        return finish("no_texture_corners")
    offset = np.array([x0, y0])
    start = points[:, 0] + offset
    stop = np.full_like(start, np.nan)
    returned = np.full_like(start, np.nan)
    fgood, bgood, valid = (np.zeros(len(start), bool) for _ in range(3))
    error, fb, step = (np.full(len(start), np.nan, np.float32) for _ in range(3))
    if len(points) < cfg.node_min_features:
        return finish("insufficient_texture_corners")
    options = dict(
        winSize=(cfg.window_size, cfg.window_size),
        maxLevel=cfg.pyramid_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        minEigThreshold=cfg.min_eigenvalue,
    )
    target, forward_status, optical_error = cv2.calcOpticalFlowPyrLK(
        before, after, points, None, **options
    )
    if target is None:
        return finish("forward_lk_failed")
    error = optical_error.reshape(-1).copy()
    fgood = (
        forward_status.ravel().astype(bool)
        & np.isfinite(target[:, 0]).all(axis=-1)
        & np.isfinite(error)
    )
    stop[fgood] = target[fgood, 0] + offset
    if fgood.sum() < cfg.node_min_features:
        return finish("insufficient_forward_lk")
    back, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        after, before, target[fgood], None, **options
    )
    if back is None:
        return finish("backward_lk_failed")
    p0, p1 = points[fgood, 0], target[fgood, 0]
    returned[fgood] = back[:, 0] + offset
    bgood[fgood] = backward_status.ravel().astype(bool)
    fb[fgood] = np.linalg.norm(back[:, 0] - p0, axis=-1)
    step[fgood] = np.linalg.norm(p1 - p0, axis=-1)
    # The bounds are relative to this common crop, exactly as in motion.py.
    valid[fgood] = (
        bgood[fgood]
        & np.isfinite(p1).all(axis=-1)
        & np.isfinite(fb[fgood])
        & np.isfinite(error[fgood])
        & (fb[fgood] <= cfg.max_fb_error_px)
        & (error[fgood] <= cfg.max_lk_error)
        & (step[fgood] <= cfg.max_feature_step_px)
        & (p1[:, 0] >= 0)
        & (p1[:, 0] < before.shape[1])
        & (p1[:, 1] >= 0)
        & (p1[:, 1] < before.shape[0])
    )
    return finish(
        "usable" if valid.sum() >= cfg.node_min_features else "insufficient_fb_features",
        bool(valid.sum() >= cfg.node_min_features),
    )


@dataclass(frozen=True)
class ReverseFlowObservations(FlowObservations):
    feature_audits: tuple[FeatureAudit, ...]

    @classmethod
    def from_frames(cls, previous, current, rig, previous_pose_xyz, config=None):
        cfg = config or FlowConfig()
        for frame in (previous.frame_idx, current.frame_idx):
            if isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, np.integer)):
                raise ValueError("reverse flow requires integer native frame numbers")
        if current.frame_idx != previous.frame_idx - 1:
            raise ValueError("reverse flow requires consecutive descending native source frames")
        pose = np.asarray(previous_pose_xyz, float)
        if pose.shape != (3, 3, 3) or not np.isfinite(pose).all():
            raise ValueError("previous internal pose must be finite three-fish three-node XYZ")
        uv = np.full((3, 3, 3, 2), np.nan)
        quality = np.zeros((3, 3, 3))
        count = np.zeros((3, 3, 3), np.int32)
        fb = np.full((3, 3, 3), np.nan)
        spread = np.full_like(fb, np.nan)
        audits = []
        for vi, view in enumerate(VIEWS):
            before, after = _gray(previous.images_bgr[view]), _gray(current.images_bgr[view])
            if before.shape != after.shape:
                raise ValueError("consecutive reverse raw image dimensions differ")
            independent = _mask_union(previous.masks[view], before.shape)
            current_union = _mask_union(current.masks[view], before.shape)
            projected = rig.camera(view).project(pose)
            capsules = _capsules(projected, before.shape, cfg.body_radius_px)
            for fish in range(3):
                allowed = (
                    capsules[fish] & independent
                    if independent is not None
                    else np.zeros(before.shape, bool)
                )
                tracks, audit = _tracked_features_audit(before, after, allowed, cfg, view, fish)
                audits.append(audit)
                if tracks is None:
                    continue
                for node in range(3):
                    result = _local_warp(projected[fish, node], tracks, current_union, cfg)
                    if result is not None:
                        (
                            uv[fish, vi, node],
                            quality[fish, vi, node],
                            count[fish, vi, node],
                            fb[fish, vi, node],
                            spread[fish, vi, node],
                        ) = result
        valid = np.isfinite(uv).all(axis=-1) & (quality > 0)
        rays = [rig.camera(view).rays(uv[:, vi]) for vi, view in enumerate(VIEWS)]
        origins = np.stack([value[0] for value in rays], axis=1)
        directions = np.stack([value[1] for value in rays], axis=1)
        valid &= np.isfinite(origins).all(axis=-1) & np.isfinite(directions).all(axis=-1)
        quality = np.where(valid, quality, 0)
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
                            or residual > cfg.maximum_ray_residual_mm
                            or np.linalg.norm(xyz - pose[fish, node]) > cfg.maximum_node_step_mm
                            or np.any(np.sum((xyz - origin) * direction, axis=-1) < 0)
                        ):
                            continue
                        errors = [
                            np.linalg.norm(rig.camera(VIEWS[v]).project(xyz) - uv[fish, v, node])
                            for v in selected
                        ]
                        if (
                            not np.isfinite(errors).all()
                            or max(errors) > cfg.maximum_reprojection_px
                        ):
                            continue
                        factor = float(quality[fish, selected, node].mean() * number / 3)
                        candidates.append(
                            (-number, float(residual), -factor, xyz, selected, factor)
                        )
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
        return cls(
            int(previous.frame_idx), int(current.frame_idx), *map(_readonly, arrays), tuple(audits)
        )


@dataclass(frozen=True)
class ReverseStep:
    pose_xyz: np.ndarray
    node_flow_valid: np.ndarray
    mode: np.ndarray
    fallback_translation_xyz: np.ndarray


def propagate_reverse(
    flow, previous_pose_xyz, earlier_in_reverse_pose_xyz=None, velocity_cap_mm=2.0
):
    """Keep valid flow nodes; translate missing nodes, or extrapolate an unsupported fish.

    `earlier_in_reverse_pose_xyz` is the state visited before `previous_pose_xyz`,
    i.e. the state at a *larger* source frame number. No baseline reseed occurs here.
    """
    previous = np.asarray(previous_pose_xyz, float)
    older = (
        previous
        if earlier_in_reverse_pose_xyz is None
        else np.asarray(earlier_in_reverse_pose_xyz, float)
    )
    if (
        previous.shape != (3, 3, 3)
        or older.shape != previous.shape
        or not np.isfinite([previous, older]).all()
    ):
        raise ValueError("reverse propagation needs finite same-shape internal poses")
    if not np.isfinite(velocity_cap_mm) or velocity_cap_mm <= 0:
        raise ValueError("reverse extrapolation cap must be finite and positive")
    valid = np.asarray(flow.node_valid)
    if valid.shape != (3, 3) or valid.dtype != bool or not np.isfinite(flow.pose_xyz[valid]).all():
        raise ValueError("flow validity and finite node coordinates disagree")
    pose, translation = previous.copy(), np.zeros((3, 3))
    mode = np.empty(3, dtype="U24")
    for fish in range(3):
        if valid[fish].any():
            delta = np.median(
                flow.pose_xyz[fish, valid[fish]] - previous[fish, valid[fish]], axis=0
            )
            pose[fish] += delta
            pose[fish, valid[fish]] = flow.pose_xyz[fish, valid[fish]]
            mode[fish] = "flow_with_translation"
        else:
            delta = previous[fish, 1] - older[fish, 1]
            delta = delta * min(1.0, velocity_cap_mm / max(float(np.linalg.norm(delta)), 1e-12))
            pose[fish] += delta
            mode[fish] = "capped_velocity_fallback"
        translation[fish] = delta
    return ReverseStep(*map(_readonly, (pose, valid.copy(), mode, translation)))

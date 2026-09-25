"""Local image-motion proposals from consecutive native-rate raw frames.

Texture features are selected inside the previous projected body and independent
segmentation. Lucas-Kanade tracks are checked forward and backward. Nearby feature
displacements warp projected anatomical nodes; these are local-flow approximations,
not matched stereo corners or independent animal identities. Missing evidence is
neutral, and current SLEAP rows and identity labels are never read.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from medaka3d.calib import triangulate
from medaka3d.types import VIEWS


@dataclass(frozen=True)
class FlowConfig:
    body_radius_px: int = 8
    search_padding_px: int = 28
    max_corners: int = 64
    corner_quality: float = 0.02
    corner_spacing_px: float = 3.0
    window_size: int = 21
    pyramid_levels: int = 2
    min_eigenvalue: float = 1e-4
    max_fb_error_px: float = 1.0
    max_lk_error: float = 35.0
    max_feature_step_px: float = 20.0
    node_radius_px: float = 22.0
    node_max_features: int = 12
    node_min_features: int = 3
    min_node_quality: float = 0.08
    maximum_ray_residual_mm: float = 1.0
    maximum_reprojection_px: float = 3.0
    maximum_node_step_mm: float = 6.0

    def __post_init__(self):
        integer_fields = (
            "body_radius_px",
            "search_padding_px",
            "max_corners",
            "window_size",
            "node_max_features",
            "node_min_features",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.window_size % 2 != 1 or self.pyramid_levels < 0:
            raise ValueError("LK window must be odd and pyramid levels nonnegative")
        if self.node_min_features > self.node_max_features:
            raise ValueError("minimum node features exceeds maximum")
        for field in (
            "corner_quality",
            "corner_spacing_px",
            "min_eigenvalue",
            "max_fb_error_px",
            "max_lk_error",
            "max_feature_step_px",
            "node_radius_px",
            "min_node_quality",
            "maximum_ray_residual_mm",
            "maximum_reprojection_px",
            "maximum_node_step_mm",
        ):
            value = getattr(self, field)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if self.corner_quality > 1 or self.min_node_quality > 1:
            raise ValueError("corner and node quality thresholds must be at most one")


def _gray(image):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise ValueError("motion expects native uint8 BGR or grayscale images")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("motion expects BGR or grayscale images")


def _mask_union(masks, shape):
    if masks is None:
        return None
    masks = np.asarray(masks)
    if masks.ndim != 3 or masks.shape[1:] != shape or masks.dtype != bool:
        raise ValueError("instance masks must be bool N,H,W in source image coordinates")
    return np.any(masks, axis=0)


def _capsules(uv, shape, radius):
    masks = np.zeros((3, *shape), np.uint8)
    for fish in range(3):
        if not np.isfinite(uv[fish]).all():
            continue
        points = np.rint(uv[fish]).astype(np.int32)
        # Skip remote projections before passing coordinates into OpenCV's int API.
        if (np.abs(points) > max(shape) * 4).any():
            continue
        cv2.polylines(masks[fish], [points], False, 1, 2 * radius)
        for point in points:
            cv2.circle(masks[fish], tuple(point), radius, 1, -1)
    # Overlapping projected bodies cannot assign the same texture to two fish.
    # Three binary masks sum to at most three. Keep the same exclusion rule
    # without widening the full raster to uint64 or using boolean scatter.
    unique = masks.sum(axis=0, dtype=np.uint8) == 1
    masks *= unique[None]
    return masks.astype(bool)


def _tracked_features(previous, current, feature_mask, config):
    """Small common crop; return only forward/backward-consistent texture tracks."""
    # BoundingRect scans the binary raster without materializing every nonzero
    # coordinate. Its integer [x,x+w), [y,y+h) box is exactly min/max + one.
    left, top, width, height = cv2.boundingRect(feature_mask.astype(np.uint8))
    if (
        width == 0
        or np.count_nonzero(feature_mask[top : top + height, left : left + width])
        < config.node_min_features
    ):
        return None
    padding = config.search_padding_px
    x0, y0 = max(0, left - padding), max(0, top - padding)
    x1, y1 = (
        min(previous.shape[1], left + width + padding),
        min(previous.shape[0], top + height + padding),
    )
    before = np.ascontiguousarray(previous[y0:y1, x0:x1])
    after = np.ascontiguousarray(current[y0:y1, x0:x1])
    allowed = np.ascontiguousarray(feature_mask[y0:y1, x0:x1], dtype=np.uint8)
    features = cv2.goodFeaturesToTrack(
        before,
        config.max_corners,
        config.corner_quality,
        config.corner_spacing_px,
        mask=allowed,
        blockSize=3,
    )
    if features is None or len(features) < config.node_min_features:
        return None
    options = dict(
        winSize=(config.window_size, config.window_size),
        maxLevel=config.pyramid_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        minEigThreshold=config.min_eigenvalue,
    )
    target, forward_status, error = cv2.calcOpticalFlowPyrLK(
        before, after, features, None, **options
    )
    if target is None:
        return None
    # Failed forward LK points may contain undefined coordinates. Never feed
    # those into a second OpenCV call; lack of a track remains neutral.
    forward_good = (
        forward_status.ravel().astype(bool)
        & np.isfinite(target[:, 0]).all(axis=-1)
        & np.isfinite(error.reshape(-1))
    )
    if forward_good.sum() < config.node_min_features:
        return None
    features, target = features[forward_good], target[forward_good]
    error = error[forward_good]
    returned, backward_status, _ = cv2.calcOpticalFlowPyrLK(after, before, target, None, **options)
    if returned is None:
        return None
    p0, p1 = features[:, 0], target[:, 0]
    fb = np.linalg.norm(returned[:, 0] - p0, axis=-1)
    photometric = error.reshape(-1)
    step = np.linalg.norm(p1 - p0, axis=-1)
    valid = (
        backward_status.ravel().astype(bool)
        & np.isfinite(p1).all(axis=-1)
        & np.isfinite(fb)
        & np.isfinite(photometric)
        & (fb <= config.max_fb_error_px)
        & (photometric <= config.max_lk_error)
        & (step <= config.max_feature_step_px)
        & (p1[:, 0] >= 0)
        & (p1[:, 0] < before.shape[1])
        & (p1[:, 1] >= 0)
        & (p1[:, 1] < before.shape[0])
    )
    if valid.sum() < config.node_min_features:
        return None
    offset = np.array([x0, y0])
    return p0[valid] + offset, p1[valid] + offset, fb[valid], photometric[valid]


def _local_warp(node_uv, tracks, current_union, config):
    start, stop, fb, error = tracks
    distance = np.linalg.norm(start - node_uv, axis=-1)
    chosen = np.flatnonzero(distance <= config.node_radius_px)
    chosen = chosen[np.argsort(distance[chosen], kind="stable")[: config.node_max_features]]
    if len(chosen) < config.node_min_features:
        return None
    flow = stop[chosen] - start[chosen]
    center = np.median(flow, axis=0)
    deviation = np.linalg.norm(flow - center, axis=-1)
    chosen = chosen[deviation <= max(0.75, 2.5 * float(np.median(deviation)))]
    if len(chosen) < config.node_min_features:
        return None
    flow = stop[chosen] - start[chosen]
    center = np.median(flow, axis=0)
    spread = float(np.median(np.linalg.norm(flow - center, axis=-1)))
    fb_error = float(np.median(fb[chosen]))
    quality = (
        min(1.0, len(chosen) / 8)
        * np.exp(-((fb_error / config.max_fb_error_px) ** 2))
        * np.exp(-((spread / 3.0) ** 2))
        * np.exp(-((np.median(error[chosen]) / config.max_lk_error) ** 2))
    )
    # A current independent mask can discount a track, but mask silence never
    # vetoes raw optical evidence. This score is an engineering quality factor.
    if current_union is not None and current_union.any():
        xy = np.rint(stop[chosen]).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, current_union.shape[1] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, current_union.shape[0] - 1)
        quality *= 0.5 + 0.5 * current_union[xy[:, 1], xy[:, 0]].mean()
    if quality < config.min_node_quality:
        return None
    return node_uv + center, float(quality), len(chosen), fb_error, spread


@dataclass(frozen=True)
class FlowObservations:
    previous_frame_idx: int
    frame_idx: int
    warped_uv: np.ndarray  # fish, view, node, xy; NaN when no reliable local flow
    quality: np.ndarray  # fish, view, node; engineering factor, not ID probability
    feature_count: np.ndarray
    fb_error_px: np.ndarray
    spread_px: np.ndarray
    valid: np.ndarray
    ray_origins: np.ndarray  # fish, view, node, xyz
    ray_directions: np.ndarray
    pose_xyz: np.ndarray  # fish, node, xyz; only nodes supported by >=2 consistent views
    node_valid: np.ndarray
    used_views: np.ndarray  # fish, view, node; selected geometry-consistent subset
    pose_quality: np.ndarray  # fish, node
    ray_residual_mm: np.ndarray  # fish, node

    @classmethod
    def from_frames(cls, previous, current, rig, previous_pose_xyz, config=None):
        config = config or FlowConfig()
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
        for vi, view in enumerate(VIEWS):
            before, after = _gray(previous.images_bgr[view]), _gray(current.images_bgr[view])
            if before.shape != after.shape:
                raise ValueError("consecutive raw image dimensions differ")
            independent = _mask_union(previous.masks[view], before.shape)
            current_union = _mask_union(current.masks[view], before.shape)
            if independent is None or not independent.any():
                continue
            projected = rig.camera(view).project(pose)
            capsules = _capsules(projected, before.shape, config.body_radius_px)
            for fish in range(3):
                tracks = _tracked_features(before, after, capsules[fish] & independent, config)
                if tracks is None:
                    continue
                for node in range(3):
                    result = _local_warp(projected[fish, node], tracks, current_union, config)
                    if result is None:
                        continue
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
        for array in arrays:
            array.setflags(write=False)
        return cls(int(previous.frame_idx), int(current.frame_idx), *arrays)

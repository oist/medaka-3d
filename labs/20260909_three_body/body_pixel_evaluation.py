"""Fixed spatial evaluation pixels, disjoint from the current fitting pixels.

This is within-frame D2-mask evaluation, not a temporal or identity holdout.
The same detector provides training and evaluation labels. Images supply shape
only; no RGB/background content is read. Ring and all-background are separate,
intentionally overlapping strata. Missing or detector-empty views stay neutral.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from observations import FrameObservation
from prepare_sequence import VIEWS

SAMPLE_COUNTS = (768, 768, 512)
STRATUM_WEIGHTS = (1.0, 0.75, 0.25)
STRATUM_NAMES = ("foreground", "dilation_ring", "all_background")
SEED_OFFSET = 6137


@dataclass(frozen=True)
class SpatialEvaluationTargets:
    frame_idx: int
    pixels: dict[str, np.ndarray]  # Native xy, variable N/view, no dummy pixels.
    target: np.ndarray  # V,Nmax, padding target/weight zero.
    weights: np.ndarray
    foreground: np.ndarray
    strata: np.ndarray  # int8: 0/1/2; -1 is padding.
    sampled_counts: np.ndarray  # V,3; actual draws, no replacement per stratum.
    pool_counts: np.ndarray  # V,3 after excluding ALL fitting coordinates.
    excluded_fit_unique_count: np.ndarray  # V; includes zero-weight fit pixels.
    view_available: np.ndarray  # Missing mask differs from available zero detections.
    foreground_nonempty: np.ndarray


def sample_body_evaluation(frame: FrameObservation, fitting_pixels) -> SpatialEvaluationTargets:
    if (
        isinstance(frame.frame_idx, bool)
        or not isinstance(frame.frame_idx, (int, np.integer))
        or frame.frame_idx < 0
    ):
        raise ValueError("frame_idx must be a nonnegative native integer")
    if (
        set(frame.images_bgr) != set(VIEWS)
        or set(frame.masks) != set(VIEWS)
        or set(fitting_pixels) != set(VIEWS)
    ):
        raise ValueError("all three views must be explicit and in the registered schema")
    counts = np.zeros((3, 3), np.int64)
    pools = np.zeros_like(counts)
    excluded_counts = np.zeros(3, np.int64)
    available = np.zeros(3, bool)
    nonempty = np.zeros(3, bool)
    coordinates, labels = {}, []
    for v, view in enumerate(VIEWS):
        image = np.asarray(frame.images_bgr[view])
        if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("images must supply nonempty native H,W,3 shape")
        height, width = image.shape[:2]
        fit = np.asarray(fitting_pixels[view])
        if (
            fit.ndim != 2
            or fit.shape[1] != 2
            or not np.isfinite(fit).all()
            or np.any(fit != np.floor(fit))
            or np.any(fit < 0)
            or np.any(fit[:, 0] >= width)
            or np.any(fit[:, 1] >= height)
        ):
            raise ValueError("fitting pixels must be finite native integer xy within image")
        flat_fit = np.unique(fit[:, 1].astype(np.int64) * width + fit[:, 0].astype(np.int64))
        excluded_counts[v] = len(flat_fit)
        mask = frame.masks[view]
        available[v] = mask is not None
        union = np.zeros((height, width), bool)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.dtype != bool or mask.ndim != 3 or mask.shape[1:] != (height, width):
                raise ValueError("D2 masks must be native boolean N,H,W")
            union = mask.any(axis=0)
        nonempty[v] = bool(union.any())
        eligible = np.ones(height * width, bool)
        eligible[flat_fit] = False
        dilation = cv2.dilate(union.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        sets = (union, dilation & ~union, ~union)
        selected, strata = [], []
        rng = np.random.default_rng(int(frame.frame_idx) * 7 + v + SEED_OFFSET)
        for k, selector in enumerate(sets):
            pool = (
                np.flatnonzero(selector.ravel() & eligible)
                if available[v]
                else np.empty(0, np.int64)
            )
            pools[v, k] = len(pool)
            count = min(len(pool), SAMPLE_COUNTS[k]) if nonempty[v] else 0
            counts[v, k] = count
            chosen = rng.choice(pool, count, replace=False) if count else np.empty(0, np.int64)
            selected.append(chosen)
            strata.append(np.full(count, k, np.int8))
        flat = np.concatenate(selected)
        coordinates[view] = np.column_stack((flat % width, flat // width)).astype(np.float64)
        labels.append(np.concatenate(strata))
    maximum = max(map(len, coordinates.values()))
    target = np.zeros((3, maximum), np.float64)
    weights = np.zeros_like(target)
    foreground = np.zeros_like(target, bool)
    strata = np.full(target.shape, -1, np.int8)
    for v, label in enumerate(labels):
        strata[v, : len(label)] = label
        for k, weight in enumerate(STRATUM_WEIGHTS):
            selected = strata[v] == k
            count = int(counts[v, k])
            if count:
                weights[v, selected] = weight / count
        foreground[v] = strata[v] == 0
        target[v, foreground[v]] = 1
    return SpatialEvaluationTargets(
        int(frame.frame_idx),
        coordinates,
        target,
        weights,
        foreground,
        strata,
        counts,
        pools,
        excluded_counts,
        available,
        nonempty,
    )


def summarize_pixel_error(union, ray_valid, targets: SpatialEvaluationTargets):
    """Keep fixed stratum denominators; invalid rays contribute zero, counted separately.

    Returns a leading batch dimension even for a single scene. Fully neutral
    targets can be evaluated with explicit arrays (B,3,0)/(3,0), without invoking
    a renderer or inventing positive pixels.
    """
    union = np.asarray(union)
    valid = np.asarray(ray_valid)
    if union.ndim == 2:
        union = union[None]
    if (
        union.ndim != 3
        or union.shape[1:] != targets.target.shape
        or not len(union)
        or not np.isfinite(union).all()
        or np.any((union < 0) | (union > 1))
        or valid.dtype != bool
        or valid.shape != targets.target.shape
    ):
        raise ValueError("union/valid shape or values differ from fixed evaluation targets")
    loss = (union - targets.target[None]) ** 2 * targets.weights[None] * valid[None]
    parts = np.stack(
        [np.sum(loss * (targets.strata == k)[None], axis=-1) for k in range(3)], axis=-1
    )
    by_view = parts.sum(axis=-1)
    return dict(
        loss_by_view=by_view,
        loss_by_stratum=parts,
        mean_view_loss=by_view.mean(axis=1),
        foreground_loss_by_view=parts[..., 0],
        background_loss_by_view=parts[..., 1:].sum(axis=-1),
        valid_counts=np.stack(
            [np.sum(valid & (targets.strata == k), axis=1) for k in range(3)], axis=-1
        ),
    )

"""Conditional color objective on fixed rendered shapes; no identity decisions.

Pixel features share the scale of appearance_diagnostic's aggregate descriptors,
but squared pixel-to-template distances are a different, unfitted surrogate.
Unexplained image pixels and background do not contribute through this objective.
"""

from itertools import permutations

import cv2
import numpy as np

PERMUTATIONS = np.array(list(permutations(range(3))), dtype=np.int64)
FEATURES = (
    "r_fraction", "g_fraction", "s_cos_h_scaled", "s_sin_h_scaled", "saturation", "value"
)


def union_core(masks):
    """Union every independent mask, then erode once by the fixed 5x5 ellipse."""
    masks = np.asarray(masks)
    if masks.dtype != bool or masks.ndim != 3 or min(masks.shape[1:]) < 1:
        raise ValueError("full-image boolean masks with shape N,H,W required")
    union = masks.any(axis=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core = cv2.erode(
        union.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)
    return union, core


def pixel_features(bgr_pixels):
    """Map native BGR uint8 colors to six bounded values, without spatial gating."""
    pixels = np.asarray(bgr_pixels)
    if pixels.dtype != np.uint8 or pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError("BGR uint8 pixel colors with shape N,3 required")
    if not len(pixels):
        return np.empty((0, 6), dtype=float)
    bgr = pixels.astype(np.float32) / 255
    normalized = bgr / np.maximum(bgr.sum(-1, keepdims=True), 1e-8)
    hsv = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    angle, saturation, value = np.deg2rad(hsv[:, 0]), hsv[:, 1], hsv[:, 2]
    return np.column_stack(
        (
            normalized[:, 2], normalized[:, 1],
            (1 + saturation * np.cos(angle)) / 2,
            (1 + saturation * np.sin(angle)) / 2, saturation, value,
        )
    ).astype(float)


def assignment_costs(features, templates, visible_alpha):
    """Return all six shape-slot -> template assignments, never a chosen ID.

The denominator is shared across permutations. Zero-mass slots retain zero raw
cost and undefined per-slot normalized costs. If all masses vanish, all six
scores are NaN/unscored and rank_order is -1 rather than a lexical winner.
"""
    features = np.asarray(features, dtype=float)
    templates = np.asarray(templates, dtype=float)
    alpha = np.asarray(visible_alpha, dtype=float)
    if features.ndim != 2 or features.shape[1] != 6 or templates.shape != (3, 6):
        raise ValueError("expected N,6 pixel features and 3,6 frozen templates")
    if alpha.shape != (len(features), 3):
        raise ValueError("visible_alpha must have shape N,3")
    for name, values in (("features", features), ("templates", templates), ("alpha", alpha)):
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"{name} must be finite and in [0,1]")
    distance_sq = np.sum((features[:, None] - templates[None]) ** 2, axis=-1)
    raw_cost = alpha.T @ distance_sq
    mass = alpha.sum(axis=0)
    denominator = float(mass.sum())
    permutation_raw = raw_cost[np.arange(3)[None], PERMUTATIONS].sum(axis=1)
    scores = (
        permutation_raw / denominator if denominator > 0 else np.full(6, np.nan)
    )
    normalized = np.divide(
        raw_cost, mass[:, None], out=np.full((3, 3), np.nan), where=mass[:, None] > 0
    )
    return {
        "permutations": PERMUTATIONS.copy(),
        "fish_effective_mass": mass,
        "fish_template_raw_cost": raw_cost,
        "fish_template_normalized_cost": normalized,
        "permutation_raw_cost": permutation_raw,
        "common_denominator": denominator,
        "scores": scores,
        "rank_order": np.argsort(scores, kind="stable") if denominator > 0 else np.full(6, -1),
        "scored": denominator > 0,
    }

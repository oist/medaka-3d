"""Independent foreground pixels for candidate color comparison.

Sampling uses native images, every D2 instance, and the source frame number.
It never uses a SLEAP row, pose, anatomical support flag, or an assigned fish ID.
Templates and their initial slot labels belong to a separate enrollment stage.
"""

from dataclasses import dataclass

import numpy as np
from pixel_template_assignment import pixel_features, union_core

from medaka3d.types import VIEWS


@dataclass(frozen=True)
class AppearancePixels:
    pixels: dict[str, np.ndarray]
    features: np.ndarray  # view,padded pixels,6
    weights: np.ndarray  # view,padded pixels; only sampled core pixels carry weight
    enabled: np.ndarray  # view; this frame supplies pixels for an enrolled view
    counts: np.ndarray  # view; sampled pixels
    available_core_counts: np.ndarray  # view; before the fixed sampling cap


def sample_appearance_pixels(frame, enabled_views, maximum_samples=512):
    enabled = np.asarray(enabled_views)
    if enabled.shape != (3,) or enabled.dtype != bool:
        raise ValueError("three explicit boolean view flags are required")
    if type(maximum_samples) is not int or maximum_samples < 1:
        raise ValueError("maximum_samples must be a positive integer")
    pixels = {view: np.empty((0, 2), float) for view in VIEWS}
    values = {view: np.empty((0, 6), float) for view in VIEWS}
    counts, core_counts = np.zeros(3, int), np.zeros(3, int)
    for index, view in enumerate(VIEWS):
        if not enabled[index]:
            continue
        masks = frame.masks.get(view)
        if masks is None:
            continue
        _, core = union_core(masks)
        locations = np.argwhere(core)
        core_counts[index] = len(locations)
        if not len(locations):
            continue
        image = frame.images_bgr.get(view)
        if image is None or image.dtype != np.uint8 or image.shape != (*core.shape, 3):
            raise ValueError("native uint8 BGR image and full-size masks must align")
        if len(locations) > maximum_samples:
            rng = np.random.default_rng(int(frame.frame_idx) * 7 + index + 42017)
            chosen = np.sort(rng.choice(len(locations), maximum_samples, replace=False))
            locations = locations[chosen]
        pixels[view] = locations[:, ::-1].astype(float)
        values[view] = pixel_features(image[locations[:, 0], locations[:, 1]])
        counts[index] = len(locations)
    size = int(counts.max(initial=0))
    features, weights = np.zeros((3, size, 6)), np.zeros((3, size))
    for index, view in enumerate(VIEWS):
        features[index, :counts[index]] = values[view]
        weights[index, :counts[index]] = 1.0
    return AppearancePixels(pixels, features, weights, enabled & (counts > 0), counts, core_counts)

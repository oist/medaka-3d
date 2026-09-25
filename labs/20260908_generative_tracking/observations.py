"""Image observations for the quarantined generative tracker.

All coordinates are pixels, in (x, y) order; input color is RGB, not OpenCV BGR.
Backgrounds must come from the same camera and must not be made from ID labels.
This deliberately modest image model uses background subtraction and flow, not
a learned fish segmenter. Reflections can look exactly like fish: these image
terms alone cannot distinguish them, and are never a hard detection veto.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType

import cv2
import numpy as np
from scipy.ndimage import map_coordinates


def _readonly(value: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(value)
    value.setflags(write=False)
    return value


def _rgb(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
        raise ValueError("frames must be RGB [H,W,3] or gray [H,W], with H,W >= 2")
    # Raw video is already uint8. Rounding it first creates a large floating
    # temporary, and range/finite checks cannot fail for this dtype. Own the
    # copy so freezing observations does not freeze the caller's video buffer.
    if image.dtype == np.uint8:
        return _readonly(image.copy(order="C"))
    if not np.isfinite(image).all():
        raise ValueError("frames must contain finite pixels")
    if np.issubdtype(image.dtype, np.floating):
        if image.min() < 0 or image.max() > 1:
            raise ValueError("floating frames must lie in [0,1]")
        image = image * 255.0
    elif image.min() < 0 or image.max() > 255:
        raise ValueError("integer frames must lie in [0,255]")
    return _readonly(np.rint(image).astype(np.uint8))


def _sample(image: np.ndarray, points: np.ndarray, outside: float = np.nan) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pixel coordinates must have shape [N,2]")
    finite = np.isfinite(points).all(axis=1)
    safe = np.where(finite[:, None], points, -1.0)
    values = map_coordinates(
        image, safe[:, ::-1].T, order=1, mode="constant", cval=outside, output=np.float64
    )
    return np.where(finite, values, outside)


@dataclass(frozen=True)
class EvidenceConfig:
    """Fixed image processing parameters, independent of evaluation labels."""

    difference_threshold: float = 12.0
    minimum_component_area: int = 6
    maximum_foreground_fraction: float = 0.35
    distance_scale_px: float = 5.0
    maximum_distance_px: float = 25.0
    histogram_bins: int = 8
    body_radius_fraction: float = 0.10
    flow_max_side: int = 640
    flow_consistency_px: float = 1.5
    motion_scale_px: float = 3.0

    def __post_init__(self) -> None:
        if not all(np.isfinite(v) and v > 0 for v in self.__dict__.values()):
            raise ValueError("evidence parameters must be positive and finite")
        if self.maximum_foreground_fraction >= 1:
            raise ValueError("maximum_foreground_fraction must be less than 1")
        if self.histogram_bins < 2 or self.flow_max_side < 16:
            raise ValueError("histogram_bins >= 2 and flow_max_side >= 16 are required")


@dataclass(frozen=True)
class EvidenceScore:
    silhouette_cost: float = 0.0
    appearance_cost: float = 0.0
    motion_cost: float = 0.0
    silhouette_quality: float = 0.0
    appearance_quality: float = 0.0
    motion_quality: float = 0.0
    appearance: np.ndarray | None = None

    @property
    def cost(self) -> float:
        """Quality-weighted sum; the tracker may instead weight terms separately."""
        return (
            self.silhouette_quality * self.silhouette_cost
            + self.appearance_quality * self.appearance_cost
            + self.motion_quality * self.motion_cost
        )


@dataclass(frozen=True)
class ViewEvidence:
    rgb: np.ndarray
    foreground: np.ndarray
    distance: np.ndarray
    labels: np.ndarray
    component_areas: np.ndarray
    quality: float
    flow: np.ndarray | None = None
    flow_confidence: np.ndarray | None = None
    flow_scale: tuple[float, float] = (1.0, 1.0)

    def distance_at(self, uv: np.ndarray) -> np.ndarray:
        """Bilinear distance to foreground, in original image pixels."""
        return _sample(self.distance, uv)


def _integer_channel_median(delta: np.ndarray) -> np.ndarray:
    """Exact median of uint8 differences, including the mean of two middle values.

    Counting 511 possible integer values avoids partitioning every full image.
    The half-integer answer is exactly representable as float32, matching the
    previous np.median(float32_delta, axis=(0, 1)) implementation.
    """
    count = delta.shape[0] * delta.shape[1]
    ranks = ((count - 1) // 2, count // 2)
    result = np.empty(3, dtype=np.float32)
    for channel in range(3):
        histogram = np.bincount((delta[..., channel] + 255).ravel(), minlength=511)
        middle = np.searchsorted(np.cumsum(histogram), ranks, side="right") - 255
        result[channel] = middle.mean()
    return result


def _foreground(
    rgb: np.ndarray, background: np.ndarray | None, config: EvidenceConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    shape = rgb.shape[:2]
    unknown = (
        np.zeros(shape, np.float32),
        np.zeros(shape, np.float32),
        np.zeros(shape, np.int32),
        np.zeros(1, np.int32),
        0.0,
    )
    if background is None:
        return unknown
    if background.shape != rgb.shape:
        raise ValueError("a background must match its camera frame shape")
    integer_delta = np.subtract(rgb, background, dtype=np.int16)
    illumination = _integer_channel_median(integer_delta)
    delta = integer_delta.astype(np.float32)
    # Remove uniform illumination changes; this is not an adaptive background
    # model that can silently absorb a stationary fish into the background.
    delta -= illumination
    magnitude = np.max(np.abs(delta), axis=2)
    probability = np.clip(
        (magnitude - config.difference_threshold) / (2 * config.difference_threshold), 0, 1
    ).astype(np.float32)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (probability >= 0.25).astype(np.uint8), connectivity=8
    )
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = areas >= config.minimum_component_area
    keep[0] = False
    probability *= keep[labels]
    mask = probability >= 0.25
    if not mask.any():
        return unknown
    fraction = float(mask.mean())
    quality = min(1.0, config.maximum_foreground_fraction / fraction)
    # A nearly flooded mask contains little localization information.
    quality *= max(0.0, (1.0 - fraction) / (1.0 - config.maximum_foreground_fraction))
    quality = min(quality, 1.0)
    distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    return probability, distance, labels, areas, quality


def _flow(
    current: np.ndarray, previous: np.ndarray | None, config: EvidenceConfig
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[float, float]]:
    if previous is None:
        return None, None, (1.0, 1.0)
    if previous.shape != current.shape:
        raise ValueError("previous frames must match current camera frame shapes")
    height, width = current.shape[:2]
    scale = min(1.0, config.flow_max_side / max(height, width))
    size = (max(2, round(width * scale)), max(2, round(height * scale)))
    gray = [
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA)
        for frame in (previous, current)
    ]
    forward = cv2.calcOpticalFlowFarneback(gray[0], gray[1], None, 0.5, 3, 21, 4, 7, 1.5, 0)
    backward = cv2.calcOpticalFlowFarneback(gray[1], gray[0], None, 0.5, 3, 21, 4, 7, 1.5, 0)
    yy, xx = np.indices(gray[0].shape, dtype=np.float32)
    target_x = xx + forward[..., 0]
    target_y = yy + forward[..., 1]
    reverse = cv2.remap(backward, target_x, target_y, cv2.INTER_LINEAR)
    next_gray = cv2.remap(gray[1].astype(np.float32), target_x, target_y, cv2.INTER_LINEAR)
    consistency = np.linalg.norm(forward + reverse, axis=2)
    prev_float = gray[0].astype(np.float32)
    variance = np.maximum(
        0.0, cv2.blur(prev_float**2, (9, 9)) - cv2.blur(prev_float, (9, 9)) ** 2
    )
    confidence = np.minimum(1.0, np.sqrt(variance) / 10.0)
    confidence *= np.exp(-0.5 * (consistency / config.flow_consistency_px) ** 2)
    confidence *= np.exp(-0.5 * ((next_gray - prev_float) / 20.0) ** 2)
    confidence *= (
        (target_x >= 0) & (target_x <= size[0] - 1)
        & (target_y >= 0) & (target_y <= size[1] - 1)
    )
    return forward, confidence, (size[0] / width, size[1] / height)


def _body_samples(uv: np.ndarray) -> np.ndarray:
    samples = []
    for start, end in pairwise(uv):
        if not np.isfinite([start, end]).all():
            continue
        count = min(25, max(2, int(np.linalg.norm(end - start) / 2) + 1))
        samples.append(np.linspace(start, end, count))
    return np.concatenate(samples) if samples else uv[np.isfinite(uv).all(axis=1)]


def _appearance(
    view: ViewEvidence, uv: np.ndarray, config: EvidenceConfig
) -> tuple[np.ndarray | None, float]:
    if view.quality == 0 or not np.isfinite(uv).all():
        return None, 0.0
    height, width = view.rgb.shape[:2]
    length = float(np.linalg.norm(np.diff(uv, axis=0), axis=1).sum())
    radius = max(2.0, min(12.0, length * config.body_radius_fraction))
    lower = np.maximum(np.floor(uv.min(axis=0) - radius - 1), 0).astype(int)
    upper = np.minimum(np.ceil(uv.max(axis=0) + radius + 2), [width, height]).astype(int)
    if np.any(upper <= lower):
        return None, 0.0
    x0, y0 = lower
    x1, y1 = upper
    tube = np.zeros((y1 - y0, x1 - x0), np.uint8)
    local = np.rint(uv - lower).astype(np.int32)
    cv2.polylines(tube, [local], False, 1, max(1, round(2 * radius)))
    for point in local:
        cv2.circle(tube, tuple(point), round(radius), 1, -1)
    weights = tube * view.foreground[y0:y1, x0:x1]
    support = float(weights.sum())
    if support < config.minimum_component_area:
        return None, 0.0
    pixel_values = view.rgb[y0:y1, x0:x1]
    histogram = np.concatenate([
        np.histogram(pixel_values[..., channel], bins=config.histogram_bins,
                     range=(0, 256), weights=weights)[0]
        for channel in range(3)
    ]).astype(float)
    histogram /= histogram.sum()
    # A component much larger than the predicted body may contain overlapping
    # animals, a reflection or a wave. It supplies weak individual appearance.
    labels = view.labels[y0:y1, x0:x1][weights > 0]
    _, label_counts = np.unique(labels, return_counts=True)
    component_area = float(np.median(view.component_areas[labels]))
    expected_area = max(1.0, length * 2 * radius + np.pi * radius**2)
    isolation = min(1.0, 2.0 * expected_area / max(1.0, component_area))
    # Multiple disconnected pieces also reduce confidence in the crop.
    coherence = float(label_counts.max() / label_counts.sum())
    quality = view.quality * min(1.0, support / max(1, tube.sum()) / 0.5)
    return _readonly(histogram), float(quality * isolation * coherence)


@dataclass(frozen=True)
class SceneEvidence:
    views: Mapping[str, ViewEvidence]
    config: EvidenceConfig

    @classmethod
    def from_frames(
        cls,
        frames: Mapping[str, np.ndarray],
        backgrounds: Mapping[str, np.ndarray] | None = None,
        previous_frames: Mapping[str, np.ndarray] | None = None,
        config: EvidenceConfig | None = None,
    ) -> SceneEvidence:
        """Precompute maps once; missing per-view backgrounds/previous frames are neutral."""
        config = config or EvidenceConfig()
        backgrounds = {} if backgrounds is None else backgrounds
        previous_frames = {} if previous_frames is None else previous_frames
        if set(backgrounds) - set(frames) or set(previous_frames) - set(frames):
            raise ValueError("background/previous camera names must be present in frames")
        views = {}
        for name, frame in frames.items():
            rgb = _rgb(frame)
            background = _rgb(backgrounds[name]) if name in backgrounds else None
            previous = _rgb(previous_frames[name]) if name in previous_frames else None
            foreground, distance, labels, areas, quality = _foreground(rgb, background, config)
            flow, confidence, scale = _flow(rgb, previous, config)
            views[name] = ViewEvidence(
                rgb, _readonly(foreground), _readonly(distance), _readonly(labels),
                _readonly(areas), quality,
                None if flow is None else _readonly(flow),
                None if confidence is None else _readonly(confidence), scale,
            )
        return cls(MappingProxyType(views), config)

    def silhouette_residuals(
        self, view: str, uv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalized distances and separate confidence weights for least squares.

        Multiply residuals by sqrt(weights) before least squares. Unobservable
        points contribute zero. Leaving the frame needs a separate geometry
        prior; the image observation itself does not claim what lies outside.
        """
        evidence = self.views[view]
        distances = evidence.distance_at(uv)
        finite = np.isfinite(distances)
        residuals = np.minimum(
            np.where(finite, distances, 0), self.config.maximum_distance_px
        ) / self.config.distance_scale_px
        return residuals, finite.astype(float) * evidence.quality

    def score(
        self,
        view: str,
        uv: np.ndarray,
        reference_appearance: np.ndarray | None = None,
        previous_uv: np.ndarray | None = None,
        occluded: bool = False,
    ) -> EvidenceScore:
        """Score projected mouth/pectoral/tail; lower costs support the hypothesis.

        References must be from this same camera and confidently unoccluded
        observations. The caller, which knows other projected fish, must mark
        overlapping bodies as occluded: a single image cannot establish purity.
        """
        uv = np.asarray(uv, dtype=float)
        if uv.shape != (3, 2):
            raise ValueError("projected skeleton must have shape [3,2]")
        evidence = self.views[view]
        residuals, weights = self.silhouette_residuals(view, _body_samples(uv))
        silhouette_quality = float(weights.mean()) if len(weights) else 0.0
        valid = weights > 0
        silhouette_cost = float(np.mean(residuals[valid] ** 2)) if valid.any() else 0.0
        if occluded:
            return EvidenceScore(
                silhouette_cost=silhouette_cost, silhouette_quality=0.25 * silhouette_quality
            )
        appearance, appearance_quality = _appearance(evidence, uv, self.config)
        appearance_cost = 0.0
        if reference_appearance is not None:
            reference = np.asarray(reference_appearance, dtype=float)
            if (reference.shape != (3 * self.config.histogram_bins,)
                    or not np.isfinite(reference).all() or (reference < 0).any()
                    or reference.sum() <= 0):
                raise ValueError("appearance reference must be a finite nonnegative histogram")
            if appearance is not None:
                reference = reference / reference.sum()
                appearance_cost = float(
                    0.5 * ((np.sqrt(appearance) - np.sqrt(reference)) ** 2).sum()
                )
        motion_cost = motion_quality = 0.0
        if previous_uv is not None and evidence.flow is not None:
            previous_uv = np.asarray(previous_uv, dtype=float)
            if previous_uv.shape != (3, 2):
                raise ValueError("previous projected skeleton must have shape [3,2]")
            scale = np.asarray(evidence.flow_scale)
            # OpenCV resize maps pixel centers as (u + .5) * scale - .5.
            small_uv = (previous_uv + 0.5) * scale - 0.5
            flow = np.column_stack([
                _sample(evidence.flow[..., axis], small_uv) / scale[axis] for axis in range(2)
            ])
            confidence = _sample(evidence.flow_confidence, small_uv, outside=0)
            valid = np.isfinite(uv).all(axis=1) & np.isfinite(flow).all(axis=1)
            confidence = np.where(valid, confidence, 0.0)
            error = np.linalg.norm(uv - previous_uv - flow, axis=1)
            if confidence.sum() > 1e-6:
                normalized = np.minimum(np.where(valid, error, 0) / self.config.motion_scale_px, 5)
                motion_cost = float(np.average(normalized**2, weights=confidence))
                motion_quality = float(confidence.mean())
        return EvidenceScore(
            silhouette_cost, appearance_cost, motion_cost, silhouette_quality,
            appearance_quality, motion_quality, appearance,
        )

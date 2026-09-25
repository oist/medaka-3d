"""CPU image-derived spatial proposals from a fixed refractive projection grid.

Candidate centres are occupied-space hypotheses, never pectoral landmarks or
identity labels. Two-view support can be contradicted by the third camera; it
is deliberately retained and distinguished from strict three-view support.
No SLEAP rows, previous S2 position or animal IDs are accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from medaka3d.types import VIEWS


def readonly(value):
    result = np.asarray(value)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MaskGridConfig:
    spacing_mm: float = 2.0
    bounds_mm: tuple = ((0.0, 192.0), (0.0, 192.0), (-6.0, 186.0))
    image_shape_hw: tuple = (1200, 1200)
    support_threshold: float = 0.5
    healthy_sigma_mm: float = 4.0
    healthy_downweight: float = 0.8
    nms_radius_mm: float = 2.0
    max_candidates: int = 48

    def __post_init__(self):
        bounds = np.asarray(self.bounds_mm, float)
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
            raise ValueError("bounds must be three finite lower/upper pairs")
        for key in ("spacing_mm", "healthy_sigma_mm", "nms_radius_mm"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        counts = np.diff(bounds, axis=1).ravel() / self.spacing_mm
        if np.any(counts <= 0) or not np.allclose(counts, np.rint(counts), atol=1e-9, rtol=0):
            raise ValueError("positive grid extents must be integer multiples of cell spacing")
        if len(self.image_shape_hw) != 2 or any(
            type(v) is not int or v < 1 for v in self.image_shape_hw
        ):
            raise ValueError("image shape must be two positive integers")
        if not np.isfinite(self.support_threshold) or not 0 < self.support_threshold <= 1:
            raise ValueError("support threshold must be in (0,1]")
        if not np.isfinite(self.healthy_downweight) or not 0 <= self.healthy_downweight < 1:
            raise ValueError("healthy downweight must be in [0,1), never a hard exclusion")
        if type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 48:
            raise ValueError("finite spatial proposal pool is bounded to 1..48 candidates")


@dataclass(frozen=True)
class MaskSamples:
    view_support: np.ndarray  # K,V float64; NaN means unknown, zero known background
    view_available: np.ndarray  # V; an empty detector output is still available
    view_informative: np.ndarray  # V; empty/zero masks are neutral to the likelihood
    known_view_count: np.ndarray  # K
    count_supported: np.ndarray  # K
    at_least_two: np.ndarray  # includes points with a known third-view contradiction
    strict_three: np.ndarray
    third_view_contradiction: np.ndarray
    mean_known_support: np.ndarray  # K, NaN if entirely unknown; not a probability


@dataclass(frozen=True)
class GridSupport:
    samples: MaskSamples
    healthy_distance_mm: np.ndarray
    novelty_weight: np.ndarray
    ranking_score: np.ndarray


@dataclass(frozen=True)
class CandidatePool:
    grid_indices: np.ndarray
    xyz: np.ndarray
    view_support: np.ndarray
    known_view_count: np.ndarray
    count_supported: np.ndarray
    third_view_contradiction: np.ndarray
    healthy_distance_mm: np.ndarray
    novelty_weight: np.ndarray
    ranking_score: np.ndarray


def bilinear_sample(image, uv):
    """Native pixel centres: (0,0) is the first pixel, outside is unknown.

    No rounding or nearest-neighbour mask admission is hidden in this sampler.
    Float64 coordinates/weights preserve fractional mask edges.
    """
    image, uv = np.asarray(image), np.asarray(uv, float)
    if image.ndim != 2 or min(image.shape) < 1 or uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("bilinear sampling needs a nonempty image and K,2 coordinates")
    height, width = image.shape
    inside = np.isfinite(uv).all(axis=1)
    inside &= (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
    inside &= (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
    result = np.full(len(uv), np.nan)
    pixels = uv[inside]
    lower = np.floor(pixels).astype(np.int64)
    x0, y0 = lower.T
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = (pixels - lower).T
    result[inside] = (
        image[y0, x0] * (1 - dx) * (1 - dy)
        + image[y0, x1] * dx * (1 - dy)
        + image[y1, x0] * (1 - dx) * dy
        + image[y1, x1] * dx * dy
    )
    return result


def nearest_segment_distance(xyz, segments):
    xyz = np.asarray(xyz, float)
    segments = np.asarray(segments, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("distance queries must be finite K,3 coordinates")
    if segments.ndim != 3 or segments.shape[1:] != (2, 3) or not np.isfinite(segments).all():
        raise ValueError("healthy segments must be finite M,2,3 coordinates")
    distance = np.full(len(xyz), np.inf)
    for first, second in segments:
        direction = second - first
        length_squared = float(np.dot(direction, direction))
        delta = xyz - first
        if length_squared > 0:
            fraction = np.clip(np.sum(delta * direction, axis=1) / length_squared, 0, 1)
            delta -= fraction[:, None] * direction
        distance = np.minimum(distance, np.linalg.norm(delta, axis=1))
    return distance


class MaskGrid:
    """Project each cell centre once, then reuse those projections for all frames."""

    def __init__(self, rig, config=None):
        self.config = config or MaskGridConfig()
        self.rig = rig
        if tuple(rig.views) != VIEWS or rig.world_units != "mm":
            raise ValueError("grid requires strict top/side1/side2 view order and millimetres")
        bounds = np.asarray(self.config.bounds_mm, float)
        counts = np.rint(np.diff(bounds, axis=1).ravel() / self.config.spacing_mm).astype(int)
        self.grid_shape = tuple(int(v) for v in counts)
        axes = [
            bounds[a, 0] + (np.arange(counts[a]) + 0.5) * self.config.spacing_mm for a in range(3)
        ]
        self.xyz = readonly(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3))
        # One full vectorised camera call per view; no per-frame geometry recomputation.
        self.projected_uv = readonly(
            np.stack([rig.camera(v).project(self.xyz) for v in VIEWS], axis=1)
        )
        if self.projected_uv.shape != (len(self.xyz), 3, 2):
            raise ValueError("camera projection returned inconsistent dimensions")

    @classmethod
    def from_rig(cls, rig, config=None):
        return cls(rig, config)

    def _unions(self, masks):
        if set(masks) != set(VIEWS):
            raise ValueError("all three views must be explicit, including missing views")
        available = np.asarray([masks[v] is not None for v in VIEWS], bool)
        unions, informative = [], []
        for view in VIEWS:
            value = masks[view]
            if value is None:
                unions.append(None)
                informative.append(False)
                continue
            value = np.asarray(value)
            if (
                value.dtype != bool
                or value.ndim != 3
                or value.shape[1:] != self.config.image_shape_hw
            ):
                raise ValueError("instance masks must be bool N,H,W in the declared native pixels")
            union = np.any(value, axis=0)
            informative.append(bool(union.any()))
            unions.append(union if informative[-1] else None)
        return unions, available, np.asarray(informative, bool)

    def _sample(self, projected_uv, masks):
        unions, available, informative = self._unions(masks)
        values = np.full(projected_uv.shape[:2], np.nan)
        for v, union in enumerate(unions):
            if union is not None:
                values[:, v] = bilinear_sample(union, projected_uv[:, v])
        known = np.isfinite(values).sum(axis=1).astype(np.uint8)
        count = np.sum(values >= self.config.support_threshold, axis=1).astype(np.uint8)
        mean = np.full(len(values), np.nan)
        np.divide(np.nansum(values, axis=1), known, out=mean, where=known > 0)
        return MaskSamples(
            *map(
                readonly,
                (
                    values,
                    available,
                    informative,
                    known,
                    count,
                    count >= 2,
                    count == 3,
                    (count == 2) & (known == 3),
                    mean,
                ),
            )
        )

    def sample_points(self, xyz, masks):
        """Independent projected-mask diagnostics at arbitrary finite spatial points."""
        xyz = np.asarray(xyz, float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("query points must be finite K,3 coordinates")
        uv = np.stack([self.rig.camera(v).project(xyz) for v in VIEWS], axis=1)
        return self._sample(uv, masks)

    def evaluate(self, masks, healthy_segments_xyz=None):
        samples = self._sample(self.projected_uv, masks)
        segments = np.empty((0, 2, 3)) if healthy_segments_xyz is None else healthy_segments_xyz
        distance = nearest_segment_distance(self.xyz, segments)
        novelty = 1 - self.config.healthy_downweight * np.exp(
            -0.5 * (distance / self.config.healthy_sigma_mm) ** 2
        )
        score = samples.mean_known_support * novelty
        return GridSupport(samples, readonly(distance), readonly(novelty), readonly(score))

    def candidate_centers(self, support):
        """Deterministic global image pool, soft suppression and Euclidean 2-mm NMS.

        Only mask-supported (>=2 cameras) grid centres are considered. The score
        is mean known-view support times soft novelty. Ties favour more supporting
        views then the fixed cell index. Radius-inclusive NMS suppresses immediate
        grid neighbours. Neither a previous target position nor an ID can gate it.
        """
        samples = support.samples
        if support.ranking_score.shape != (len(self.xyz),):
            raise ValueError("support does not match this grid")
        available = np.flatnonzero(samples.at_least_two)
        order = np.lexsort(
            (
                available,
                -samples.count_supported[available].astype(int),
                -support.ranking_score[available],
            )
        )
        remaining, selected = available[order], []
        while len(remaining) and len(selected) < self.config.max_candidates:
            chosen = remaining[0]
            selected.append(chosen)
            distance_squared = np.sum((self.xyz[remaining] - self.xyz[chosen]) ** 2, axis=1)
            remaining = remaining[distance_squared > self.config.nms_radius_mm**2]
        indices = np.asarray(selected, np.int64)
        return CandidatePool(
            *map(
                readonly,
                (
                    indices,
                    self.xyz[indices],
                    samples.view_support[indices],
                    samples.known_view_count[indices],
                    samples.count_supported[indices],
                    samples.third_view_contradiction[indices],
                    support.healthy_distance_mm[indices],
                    support.novelty_weight[indices],
                    support.ranking_score[indices],
                ),
            )
        )

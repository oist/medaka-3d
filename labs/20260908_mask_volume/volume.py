"""Independent union masks sampled through the calibrated refractive projection.

This is a per-frame support diagnostic, not a three-fish model or an identity
tracker. Components can be fragments, overlapping animals, reflections or ghosts.
Their centroids are volume summaries, not anatomical keypoints.

Image coordinates must match ``RefractiveCamera.project``: pixel (0, 0) is the
centre of the first native-resolution pixel. Images are never silently resized.
CPU uses NumPy bilinear interpolation; CUDA uses torch ``grid_sample``. Geometry
is always projected once on the CPU through the same existing calibrated model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

from medaka3d.types import Detection2D, Instance3D

if TYPE_CHECKING:
    from medaka3d.calib import Rig


@dataclass(frozen=True)
class VolumeConfig:
    """Cell-centred, axis-aligned grid in the rig's world coordinates (mm).

    Each extent must be an integer multiple of ``spacing_mm``. This avoids
    silently overshooting the specified domain or misreporting boundary volumes.
    The default floor tolerance is an engineering setting, not a surveyed floor.
    ``sampling_dtype`` controls mask/confidence arithmetic, CUDA's normalized
    sampling grid, and returned floating arrays. Float64 is the default because
    float32 image-coordinate rounding can change a threshold decision at an edge.
    """

    bounds_mm: tuple[tuple[float, float], ...] = ((0, 192), (0, 192), (-6, 186))
    spacing_mm: float = 2.0
    support_threshold: float = 0.5
    sampling_dtype: str = "float64"

    def __post_init__(self) -> None:
        bounds = np.asarray(self.bounds_mm, dtype=float)
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
            raise ValueError("bounds_mm must be three finite (lower, upper) pairs")
        if not np.isfinite(self.spacing_mm) or self.spacing_mm <= 0:
            raise ValueError("spacing_mm must be finite and positive")
        lengths = bounds[:, 1] - bounds[:, 0]
        counts = lengths / self.spacing_mm
        if np.any(lengths <= 0) or not np.allclose(counts, np.round(counts), atol=1e-9,
                                                 rtol=0):
            raise ValueError("each positive grid extent must be a multiple of spacing_mm")
        if not np.isfinite(self.support_threshold) or not 0 < self.support_threshold <= 1:
            raise ValueError("support_threshold must be in (0, 1]")
        if self.sampling_dtype not in ("float32", "float64"):
            raise ValueError("sampling_dtype must be 'float32' or 'float64'")


@dataclass(frozen=True)
class VolumeResult:
    """Flat arrays follow ``MaskVolume.xyz``; view axis follows ``rig.views``.

    ``sampled`` is foreground support conditional on available image pixels,
    with NaN for unknown; ``confidence`` is the available bilinear weight.
    ``soft_score`` is a confidence-weighted arithmetic mean across known views,
    with NaN for entirely unknown points. It is not a calibrated probability.
    ``two_of_three`` includes points contradicted by a known third-view mask;
    it is deliberately a separate diagnostic from ``strict3``.

    Components use face (6-neighbour) connectivity with no size/count filter.
    Component volumes count entire grid cells; centroids weight their centres
    by soft_score. Neither is an estimate of anatomical body volume or position.
    """

    sampled: np.ndarray  # (3, N), config.sampling_dtype
    confidence: np.ndarray  # (3, N), config.sampling_dtype
    known: np.ndarray  # (3, N), bool
    known_count: np.ndarray  # (N,), uint8
    support_count: np.ndarray  # (N,), uint8
    soft_score: np.ndarray  # (N,), config.sampling_dtype
    strict3: np.ndarray  # (N,), bool
    two_of_three: np.ndarray  # (N,), bool
    components: dict[str, list[dict]]
    labels: dict[str, np.ndarray]  # flattened int32 labels; 0 is background


def _sample_numpy(channels: np.ndarray, uv: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Independent CPU bilinear sampler; outside points have zero available weight."""
    out = np.zeros((channels.shape[0], len(uv)), dtype=channels.dtype)
    pixels = uv[inside]
    if not len(pixels):
        return out
    x0, y0 = np.floor(pixels).astype(np.int64).T
    x1 = np.minimum(x0 + 1, channels.shape[2] - 1)
    y1 = np.minimum(y0 + 1, channels.shape[1] - 1)
    dx, dy = (pixels - np.column_stack((x0, y0))).T
    out[:, inside] = (
        channels[:, y0, x0] * ((1 - dx) * (1 - dy))
        + channels[:, y0, x1] * (dx * (1 - dy))
        + channels[:, y1, x0] * ((1 - dx) * dy)
        + channels[:, y1, x1] * (dx * dy)
    )
    return out


class MaskVolume:
    """Precompute a fixed refractive grid; evaluate independent frames.

    ``device='cpu'`` does not import torch. An explicit unavailable CUDA device
    raises, so provenance never silently describes a different backend. All
    geometry and connected-component summaries use NumPy/SciPy on the CPU.
    """

    def __init__(self, rig: Rig, config: VolumeConfig | None = None, device: str = "cpu"):
        self.rig = rig
        self.config = config or VolumeConfig()
        self.numpy_dtype = np.dtype(self.config.sampling_dtype)
        self.views = rig.views
        if len(self.views) != 3 or len(set(self.views)) != 3:
            raise ValueError("MaskVolume requires three distinct calibrated camera views")
        if rig.world_units != "mm":
            raise ValueError("MaskVolume requires a rig in mm")
        self.device = str(device)
        self._torch = None
        if self.device != "cpu":
            if self.device != "cuda" and not self.device.startswith("cuda:"):
                raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable; choose device='cpu'")
            # Validate the index immediately rather than during a later frame.
            torch.empty(0, device=self.device)
            self._torch = torch
        bounds = np.asarray(self.config.bounds_mm, dtype=float)
        counts = np.round((bounds[:, 1] - bounds[:, 0]) / self.config.spacing_mm).astype(int)
        self.grid_shape = tuple(int(value) for value in counts)
        axes = [bounds[j, 0] + (np.arange(counts[j]) + 0.5) * self.config.spacing_mm
                for j in range(3)]
        self.xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        self.projected_by_view: dict[str, np.ndarray] = {}
        for view in self.views:
            projected = np.empty((len(self.xyz), 2), dtype=float)
            for start in range(0, len(self.xyz), 100_000):
                projected[start:start + 100_000] = rig.camera(view).project(
                    self.xyz[start:start + 100_000]
                )
            self.projected_by_view[view] = projected
        self._inside: dict[tuple, np.ndarray] = {}
        self._torch_grids: dict[tuple, object] = {}

    def _coordinates(self, view: str, shape: tuple[int, int]) -> tuple[np.ndarray, object]:
        key = (view, *shape)
        if key not in self._inside:
            height, width = shape
            uv = self.projected_by_view[view]
            inside = np.isfinite(uv).all(axis=1)
            inside &= (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
            inside &= (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
            self._inside[key] = inside
            if self._torch is not None:
                # align_corners=False handles even a one-pixel-wide image.
                normal = (2 * (uv + 0.5) / np.array([width, height]) - 1).astype(
                    self.numpy_dtype
                )
                normal[~inside] = 2.0
                self._torch_grids[key] = self._torch.from_numpy(normal[:, None, :]).to(
                    self.device
                )
        return self._inside[key], self._torch_grids.get(key)

    def _channels(self, mask: np.ndarray, confidence: float | np.ndarray | None) -> np.ndarray:
        mask = np.asarray(mask, dtype=self.numpy_dtype)
        if mask.ndim != 2 or min(mask.shape) < 1:
            raise ValueError("each mask must be a nonempty 2D array in native image pixels")
        finite = np.isfinite(mask)
        if np.any((mask[finite] < 0) | (mask[finite] > 1)) or np.isinf(mask).any():
            raise ValueError("mask values must be in [0, 1], or NaN for unknown pixels")
        if confidence is None:
            weights = np.zeros_like(mask)
        else:
            weights = np.asarray(confidence, dtype=self.numpy_dtype)
            if weights.ndim == 0:
                weights = np.full_like(mask, weights)
            elif weights.shape != mask.shape:
                raise ValueError("pixel confidence must have the same shape as its mask")
            if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
                raise ValueError("confidence values must be finite and in [0, 1]")
            weights = np.where(finite, weights, 0)
        # Weight-normalized interpolation excludes unknown pixel contributions.
        return np.stack((np.where(finite, mask, 0) * weights, weights))

    def evaluate(
        self,
        mask_by_view: Mapping[str, np.ndarray | None],
        confidence_by_view: Mapping[str, float | np.ndarray | None] | None = None,
    ) -> VolumeResult:
        """Sample union masks without using keypoints, identity or other frames.

        A missing key, None mask, zero confidence or out-of-image projection is
        unknown, never observed background. A supplied all-zero mask with
        positive confidence is observed background. Confidence defaults to one
        for supplied masks; an explicit None confidence makes that view unknown.
        NaN mask pixels are also unknown. Confidence modifies the soft mean;
        support_count counts known views above the configured mask threshold.
        """
        confidence_by_view = confidence_by_view or {}
        extra = (set(mask_by_view) | set(confidence_by_view)) - set(self.views)
        if extra:
            raise ValueError(f"unrecognised camera views: {sorted(extra)}")
        sampled = np.full((3, len(self.xyz)), np.nan, dtype=self.numpy_dtype)
        confidence = np.zeros_like(sampled)
        groups: dict[tuple[int, int], list[tuple[int, np.ndarray, np.ndarray, object]]] = {}
        for index, view in enumerate(self.views):
            mask = mask_by_view.get(view)
            if mask is None:
                continue
            channels = self._channels(mask, confidence_by_view.get(view, 1.0))
            inside, grid = self._coordinates(view, channels.shape[1:])
            if self._torch is None:
                values = _sample_numpy(channels, self.projected_by_view[view], inside)
                confidence[index] = values[1]
                np.divide(values[0], values[1], out=sampled[index], where=values[1] > 0)
            else:
                groups.setdefault(channels.shape[1:], []).append((index, channels, inside, grid))
        if self._torch is not None:
            torch = self._torch
            with torch.inference_mode():
                for items in groups.values():
                    images = torch.from_numpy(np.stack([item[1] for item in items])).to(self.device)
                    grids = torch.stack([item[3] for item in items])
                    values = torch.nn.functional.grid_sample(
                        images, grids, mode="bilinear", padding_mode="zeros", align_corners=False
                    ).squeeze(-1).cpu().numpy()
                    for row, (index, _, inside, _) in enumerate(items):
                        weights = np.where(inside, values[row, 1], 0)
                        confidence[index] = weights
                        np.divide(values[row, 0], weights, out=sampled[index], where=weights > 0)
        np.clip(sampled, 0, 1, out=sampled)
        np.clip(confidence, 0, 1, out=confidence)
        known = confidence > 0
        known_count = known.sum(axis=0, dtype=np.uint8)
        support_count = (known & (sampled >= self.config.support_threshold)).sum(
            axis=0, dtype=np.uint8
        )
        weight_sum = confidence.sum(axis=0)
        score = np.full(len(self.xyz), np.nan, dtype=self.numpy_dtype)
        np.divide(np.nansum(sampled * confidence, axis=0), weight_sum, out=score,
                  where=weight_sum > 0)
        strict3 = support_count == 3
        two_of_three = support_count >= 2
        components = {}
        labels = {}
        for name, support in (("strict3", strict3), ("two_of_three", two_of_three)):
            labels[name], components[name] = self._components(support, score)
        return VolumeResult(sampled, confidence, known, known_count, support_count, score,
                            strict3, two_of_three, components, labels)

    def _components(self, support: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        labels, count = ndimage.label(support.reshape(self.grid_shape))
        flat = labels.ravel()
        if count == 0:
            return flat, []
        sizes = np.bincount(flat, minlength=count + 1)
        weights = np.where(support, score, 0).astype(float)
        weight_sum = np.bincount(flat, weights=weights, minlength=count + 1)
        centroids = np.stack([
            np.bincount(flat, weights=weights * self.xyz[:, axis], minlength=count + 1)
            for axis in range(3)
        ], axis=1)
        centroids[1:] /= weight_sum[1:, None]
        lower = np.asarray(self.config.bounds_mm, dtype=float)[:, 0]
        spacing = self.config.spacing_mm
        result = []
        for label, box in enumerate(ndimage.find_objects(labels), start=1):
            bounds = [[float(lower[axis] + part.start * spacing),
                       float(lower[axis] + part.stop * spacing)]
                      for axis, part in enumerate(box)]
            result.append({
                "label": label,
                "voxel_count": int(sizes[label]),
                "volume_mm3": float(sizes[label] * spacing ** 3),
                "centroid_xyz": centroids[label].tolist(),
                "bounds_mm": bounds,
                "mean_score": float(weight_sum[label] / sizes[label]),
            })
        return flat, result


class MaskVolumeAssociator:
    """Associator adapter returning all three-view component centroids.

    Slots are frame-local component indices. xyz is a support-volume centroid,
    not the pectoral node. No instance correspondence can be recovered from a
    union mask, so sources stays empty and residual stays NaN (not calibrated mm).
    Detector scores are not calibrated pixel confidences and are not multiplied
    into masks. Input detections, including their scores, should be retained by
    the caller's provenance. Empty detection lists remain unknown: this contract
    does not say whether an image was successfully processed with zero detections.
    """

    name = "refractive-mask-volume"

    def __init__(self, volume: MaskVolume):
        self.volume = volume

    def associate(
        self, detections: dict[str, Sequence[Detection2D]], frame: int
    ) -> list[Instance3D]:
        extra = set(detections) - set(self.volume.views)
        if extra:
            raise ValueError(f"unrecognised camera views: {sorted(extra)}")
        union_by_view = {}
        for view, rows in detections.items():
            masks = []
            for detection in rows:
                if detection.view != view or detection.frame != frame:
                    raise ValueError("detections must match the requested view and frame")
                if detection.mask is not None:
                    # Validate without converting unknown pixels into background.
                    self.volume._channels(detection.mask, 1.0)
                    masks.append(np.asarray(detection.mask, dtype=self.volume.numpy_dtype))
            if masks:
                if any(mask.shape != masks[0].shape for mask in masks):
                    raise ValueError("all masks in one view must have the same native image shape")
                union_by_view[view] = np.maximum.reduce(masks)
        result = self.volume.evaluate(union_by_view)
        return [Instance3D(frame=frame, xyz=np.asarray(component["centroid_xyz"]), slot=slot)
                for slot, component in enumerate(result.components["strict3"])]

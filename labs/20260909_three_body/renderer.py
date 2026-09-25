"""Differentiable three-body silhouettes along fixed calibrated water rays.

The existing calibration supplies each native image pixel's refracted half-ray
in water. Only body parameters are differentiated; neither Snell's law nor the
camera projection is reimplemented in torch.

Each mouth--pectoral--tail polyline carries a fixed number of axisymmetric
Gaussian ellipsoids. Their peak response along a ray supplies a soft silhouette,
not a radiometric density integral. The union is symmetric in fish indices and
therefore cannot establish identity. Reflections, surface waves, scattering and
view-dependent colour are outside this renderer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from medaka3d.calib import Rig


@dataclass(frozen=True)
class RendererConfig:
    gaussians_per_fish: int = 8
    width_ratio: float = 0.06  # Maximum radial sigma / mouth-to-pect-to-tail length.
    opacity: float = 0.85  # Per-Gaussian peak alpha, not measured optical opacity.
    axial_overlap: float = 0.8  # Axial sigma / spacing of centres on each segment.
    minimum_sigma_mm: float = 0.05  # Numerical regularisation, not a body-size claim.
    pixel_chunk_size: int = 4096

    def __post_init__(self) -> None:
        if not isinstance(self.gaussians_per_fish, int) or not 4 <= self.gaussians_per_fish <= 32:
            raise ValueError("gaussians_per_fish must be an integer between 4 and 32")
        if not isinstance(self.pixel_chunk_size, int) or self.pixel_chunk_size < 1:
            raise ValueError("pixel_chunk_size must be a positive integer")
        if not np.isfinite(self.opacity) or not 0 < self.opacity < 1:
            raise ValueError("opacity must be in (0, 1)")
        for name in ("width_ratio", "axial_overlap", "minimum_sigma_mm"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class GaussianBodies:
    centers: torch.Tensor  # (B, 3, K, 3), world mm
    tangents: torch.Tensor  # (B, 3, K, 3), unit directions
    axial_sigma_mm: torch.Tensor  # (B, 3, K)
    radial_sigma_mm: torch.Tensor  # (B, 3, K)
    segment_lengths_mm: torch.Tensor  # (B, 3, 2)
    nondegenerate: torch.Tensor  # (B, 3), both segments longer than minimum_sigma_mm


@dataclass(frozen=True)
class RenderResult:
    per_fish_alpha: torch.Tensor  # (B, V, N, 3), independent amodal silhouettes
    union_alpha: torch.Tensor  # (B, V, N), 1 - product(1 - per_fish_alpha)
    per_fish_depth_mm: torch.Tensor  # (B, V, N, 3), water-ray distance; inf if unsupported
    visible_alpha: torch.Tensor  # (B, V, N, 3), approximate front-to-back contribution
    ownership: torch.Tensor  # (B, V, N, 3), visible_alpha / union; zero for background
    per_fish_mahalanobis_sq: torch.Tensor  # (B, V, N, 3), nearest Gaussian
    ray_valid: torch.Tensor  # (V, N), padding and invalid rays are false
    bodies: GaussianBodies


class RefractiveFishRenderer:
    """Cache rays for arbitrary pixels, then render one or more three-fish poses.

    ``pixels_by_view`` contains native distorted image coordinates, (N_view, 2).
    Different counts are padded to the largest N; missing views and NaN pixels
    remain invalid. The caller must exclude ``ray_valid=False`` from image loss.
    No image resize or coordinate conversion is inferred.

    ``render`` always returns a batch dimension, including an unbatched input.
    Width ratios can be positive scalars, (3,), or (B, 3) differentiable tensors.
    All returned alpha values are differentiable in pose and width. Visibility
    sorts approximate per-fish depths and is only piecewise differentiable when
    that order changes; union-alpha does not depend on the order. Numerically
    equal depths average tied compositing orders instead of privileging a fish ID.
    """

    def __init__(
        self,
        rig: Rig,
        pixels_by_view: Mapping[str, np.ndarray],
        config: RendererConfig | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config or RendererConfig()
        self.rig = rig
        self.views = rig.views
        self.device = torch.device(device)
        self.dtype = dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("renderer dtype must be torch.float32 or torch.float64")
        if rig.world_units != "mm" or len(self.views) != 3 or len(set(self.views)) != 3:
            raise ValueError("renderer requires three distinct calibrated views in mm")
        extra = set(pixels_by_view) - set(self.views)
        if extra:
            raise ValueError(f"unrecognised camera views: {sorted(extra)}")
        supplied = {}
        for view in self.views:
            pixels = np.asarray(pixels_by_view.get(view, np.empty((0, 2))), dtype=float)
            if pixels.ndim != 2 or pixels.shape[1] != 2:
                raise ValueError("pixel arrays must have shape (N, 2)")
            supplied[view] = pixels
        self.pixel_counts = {view: len(pixels) for view, pixels in supplied.items()}
        self.n_pixels = max(self.pixel_counts.values())
        if self.n_pixels == 0:
            raise ValueError("at least one sampled image pixel is required")
        self.pixels = np.full((3, self.n_pixels, 2), np.nan)
        origins = np.zeros((3, self.n_pixels, 3))
        directions = np.zeros_like(origins)
        directions[..., 2] = 1
        valid = np.zeros((3, self.n_pixels), dtype=bool)
        for index, view in enumerate(self.views):
            pixels = supplied[view]
            self.pixels[index, :len(pixels)] = pixels
            finite = np.isfinite(pixels).all(axis=1)
            if finite.any():
                ray_o, ray_d = rig.camera(view).rays(pixels[finite])
                length = np.linalg.norm(ray_d, axis=1)
                good = np.isfinite(ray_o).all(axis=1) & np.isfinite(ray_d).all(axis=1)
                good &= length > 1e-12
                locations = np.flatnonzero(finite)[good]
                origins[index, locations] = ray_o[good]
                directions[index, locations] = ray_d[good] / length[good, None]
                valid[index, locations] = True
        self.ray_origins = torch.as_tensor(origins, device=self.device, dtype=dtype)
        self.ray_directions = torch.as_tensor(directions, device=self.device, dtype=dtype)
        self.ray_valid = torch.as_tensor(valid, device=self.device)
        head_count = max(2, self.config.gaussians_per_fish // 3)
        tail_count = self.config.gaussians_per_fish - head_count
        self._head_count = head_count
        self._tail_count = tail_count
        self._fractions = torch.cat([
            (torch.arange(head_count, device=self.device, dtype=dtype) + 0.5) / head_count,
            (torch.arange(tail_count, device=self.device, dtype=dtype) + 0.5) / tail_count,
        ])
        self._segment = torch.tensor([0] * head_count + [1] * tail_count, device=self.device)
        # Fixed engineering taper. Width is fitted; the taper is not learned here.
        self._taper = torch.cat([
            0.45 + 0.55 * self._fractions[:head_count],
            1.0 - 0.85 * self._fractions[head_count:] ** 1.2,
        ])
        self._counts = torch.tensor([head_count, tail_count], device=self.device, dtype=dtype)

    def gaussians(
        self, pose_xyz: torch.Tensor | np.ndarray, width_ratio: torch.Tensor | float | None = None
    ) -> GaussianBodies:
        """Build ellipsoids along both segments; input node coordinates are never replaced."""
        pose = torch.as_tensor(pose_xyz, dtype=self.dtype, device=self.device)
        if pose.shape == (3, 3, 3):
            pose = pose.unsqueeze(0)
        if pose.ndim != 4 or pose.shape[1:] != (3, 3, 3) or pose.shape[0] < 1:
            raise ValueError("pose_xyz must have shape (3, 3, 3) or (B, 3, 3, 3)")
        if not bool(torch.isfinite(pose).all()):
            raise ValueError("pose_xyz must be finite; missing observations are not missing poses")
        width = torch.as_tensor(self.config.width_ratio if width_ratio is None else width_ratio,
                                dtype=self.dtype, device=self.device)
        if width.ndim == 0:
            width = width.expand(pose.shape[0], 3)
        elif width.shape == (3,):
            width = width.unsqueeze(0).expand(pose.shape[0], 3)
        elif width.shape != (pose.shape[0], 3):
            raise ValueError("width_ratio must be a scalar, (3,), or (B, 3)")
        if not bool(torch.isfinite(width).all()) or not bool((width > 0).all()):
            raise ValueError("width_ratio must be finite and positive")
        vectors = pose[..., 1:, :] - pose[..., :-1, :]
        lengths = torch.linalg.vector_norm(vectors, dim=-1)
        unit = vectors / lengths.clamp_min(1e-12).unsqueeze(-1)
        # A collapsed segment has a finite numerical ellipsoid and is explicitly
        # flagged. Anatomy/admission checks belong to the fitting model, not here.
        fallback = torch.zeros_like(unit)
        fallback[..., 0] = 1
        unit = torch.where((lengths > 1e-12).unsqueeze(-1), unit, fallback)
        centers = pose[..., :-1, :][..., self._segment, :]
        centers = centers + vectors[..., self._segment, :] * self._fractions[:, None]
        axial = (lengths / self._counts * self.config.axial_overlap)[..., self._segment]
        radial = lengths.sum(dim=-1, keepdim=True) * width.unsqueeze(-1) * self._taper
        return GaussianBodies(
            centers=centers,
            tangents=unit[..., self._segment, :],
            axial_sigma_mm=axial.clamp_min(self.config.minimum_sigma_mm),
            radial_sigma_mm=radial.clamp_min(self.config.minimum_sigma_mm),
            segment_lengths_mm=lengths,
            nondegenerate=(lengths > self.config.minimum_sigma_mm).all(dim=-1),
        )

    def _ray_response(self, bodies: GaussianBodies, start: int, stop: int):
        origins = self.ray_origins[None, :, start:stop, None, None, :]
        directions = self.ray_directions[None, :, start:stop, None, None, :]
        centers = bodies.centers[:, None, None]
        tangent = bodies.tangents[:, None, None]
        delta = centers - origins
        delta_parallel = (delta * tangent).sum(dim=-1)
        direction_parallel = (directions * tangent).sum(dim=-1)
        delta_perp = delta - delta_parallel.unsqueeze(-1) * tangent
        direction_perp = directions - direction_parallel.unsqueeze(-1) * tangent
        inverse_axial_sq = bodies.axial_sigma_mm[:, None, None].reciprocal().square()
        inverse_radial_sq = bodies.radial_sigma_mm[:, None, None].reciprocal().square()
        # For Q = covariance^-1 and c-o = delta, the unconstrained minimum is
        # t = d^T Q delta / (d^T Q d). Clamp to the water half-ray t >= 0.
        numerator = (direction_perp * delta_perp).sum(dim=-1) * inverse_radial_sq
        numerator = numerator + direction_parallel * delta_parallel * inverse_axial_sq
        denominator = direction_perp.square().sum(dim=-1) * inverse_radial_sq
        denominator = denominator + direction_parallel.square() * inverse_axial_sq
        depth = (numerator / denominator).clamp_min(0)
        residual = delta - depth.unsqueeze(-1) * directions
        residual_parallel = (residual * tangent).sum(dim=-1)
        residual_perp = residual - residual_parallel.unsqueeze(-1) * tangent
        # Use the residual directly instead of subtracting two large quadratic
        # terms (delta^T Q delta - numerator^2 / denominator), avoiding cancellation.
        mahalanobis = residual_perp.square().sum(dim=-1) * inverse_radial_sq
        mahalanobis = mahalanobis + residual_parallel.square() * inverse_axial_sq
        alpha = self.config.opacity * torch.exp(-0.5 * mahalanobis)
        valid = self.ray_valid[None, :, start:stop, None, None]
        alpha = torch.where(valid, alpha, 0)
        fish_alpha = 1 - (1 - alpha).prod(dim=-1)
        weight_sum = alpha.sum(dim=-1)
        supported = weight_sum > torch.finfo(self.dtype).eps
        fish_depth = (alpha * depth).sum(dim=-1) / weight_sum.clamp_min(torch.finfo(self.dtype).eps)
        fish_depth = torch.where(supported, fish_depth, torch.inf)
        nearest = mahalanobis.amin(dim=-1)
        nearest = torch.where(valid.squeeze(-1), nearest, torch.inf)
        return fish_alpha, fish_depth, nearest

    def render(
        self, pose_xyz: torch.Tensor | np.ndarray, width_ratio: torch.Tensor | float | None = None
    ) -> RenderResult:
        bodies = self.gaussians(pose_xyz, width_ratio)
        chunks = [self._ray_response(bodies, start, min(start + self.config.pixel_chunk_size,
                                                       self.n_pixels))
                  for start in range(0, self.n_pixels, self.config.pixel_chunk_size)]
        fish_alpha = torch.cat([chunk[0] for chunk in chunks], dim=2)
        fish_depth = torch.cat([chunk[1] for chunk in chunks], dim=2)
        nearest = torch.cat([chunk[2] for chunk in chunks], dim=2)
        union = 1 - (1 - fish_alpha).prod(dim=-1)
        order = fish_depth.argsort(dim=-1, stable=True)
        sorted_alpha = torch.gather(fish_alpha, -1, order)
        sorted_depth = torch.gather(fish_depth, -1, order)
        transmission = torch.cat([torch.ones_like(sorted_alpha[..., :1]),
                                  (1 - sorted_alpha[..., :-1]).cumprod(dim=-1)], dim=-1)
        sorted_visible = sorted_alpha * transmission
        a, b, c = sorted_alpha.unbind(dim=-1)
        tie_tolerance = 16 * torch.finfo(self.dtype).eps
        tie_first = torch.isclose(sorted_depth[..., 0], sorted_depth[..., 1],
                                  rtol=tie_tolerance, atol=0)
        tie_last = torch.isclose(sorted_depth[..., 1], sorted_depth[..., 2],
                                 rtol=tie_tolerance, atol=0)
        first_average = torch.stack([a * (1 - b / 2), b * (1 - a / 2),
                                     c * (1 - a) * (1 - b)], dim=-1)
        last_average = torch.stack([a, (1 - a) * b * (1 - c / 2),
                                    (1 - a) * c * (1 - b / 2)], dim=-1)
        pair_products = torch.stack([b * c, a * c, a * b], dim=-1)
        other_sum = sorted_alpha.sum(dim=-1, keepdim=True) - sorted_alpha
        all_average = sorted_alpha * (1 - other_sum / 2 + pair_products / 3)
        sorted_visible = torch.where(tie_first.unsqueeze(-1), first_average, sorted_visible)
        sorted_visible = torch.where(tie_last.unsqueeze(-1), last_average, sorted_visible)
        sorted_visible = torch.where((tie_first & tie_last).unsqueeze(-1), all_average,
                                     sorted_visible)
        visible = torch.zeros_like(sorted_visible).scatter(-1, order, sorted_visible)
        ownership = visible / union.unsqueeze(-1).clamp_min(torch.finfo(self.dtype).eps)
        return RenderResult(fish_alpha, union, fish_depth, visible, ownership, nearest,
                            self.ray_valid, bodies)

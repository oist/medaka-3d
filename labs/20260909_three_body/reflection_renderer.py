"""Differentiable fixed-path reflection responses for a controlled local probe.

Only the three bodies' original nine anatomical coordinates vary. The frozen
eight-Gaussian construction, calibration, nominal wall/floor geometry, indices,
opacity and width configuration are retained. Rays and optical weights are
cached without gradients. No pose fitting, correspondence, visibility ordering
across optical paths, or identity decisions occur here.

These are Gaussian peak responses, not radiometric images or probabilities.
The Fresnel factors assume unpolarized light at the existing water ray; camera
entry losses/polarization are not modeled. Unit-weight union is only an upper
response for these same branches. Unbounded direct plus finite reflections is
an inherited algebraic control, not a physically complete tank model. The
finite direct mode additionally assumes the nominal z=0 floor is appropriate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from reflection_paths import ReflectionPaths, trace_y192_reflections
from renderer import GaussianBodies, RefractiveFishRenderer, RendererConfig

if TYPE_CHECKING:
    from medaka3d.calib import Rig


@dataclass(frozen=True)
class FixedRaySegment:
    """Safe tensor geometry; invalid paths have dummy geometry and zero response."""

    origins: torch.Tensor  # V,N,3
    directions: torch.Tensor  # V,N,3, unit even at invalid locations
    maximum_distance_mm: torch.Tensor  # V,N, finite; zero if invalid
    valid: torch.Tensor  # V,N


@dataclass(frozen=True)
class ReflectionRenderResult:
    direct_alpha: torch.Tensor  # B,V,N,3; selected direct mode
    inner_alpha: torch.Tensor  # B,V,N,3; unweighted finite branch
    outer_alpha: torch.Tensor  # B,V,N,3; unweighted finite branch
    direct_union: torch.Tensor  # B,V,N
    unit_union: torch.Tensor  # B,V,N; all three bodies and all three branches
    fresnel_union: torch.Tensor  # B,V,N; factors applied to per-body branch alpha
    ray_valid: torch.Tensor  # V,N; original calibrated-ray denominator
    direct_valid: torch.Tensor  # V,N; selected mode, NOT the image denominator
    inner_valid: torch.Tensor  # V,N
    outer_valid: torch.Tensor  # V,N
    inner_weight: torch.Tensor  # V,N
    outer_weight: torch.Tensor  # V,N
    bodies: GaussianBodies
    direct_mode: str


class ReflectionFishRenderer:
    """Cache fixed water segments, then differentiate only the original pose.

    ``render`` accepts (3,3,3) or (B,3,3,3), always returns a batch dimension,
    and does not mutate the input or cache. Views and padded pixel positions
    follow the frozen renderer. Missing/invalid calibrated rays are neutral.
    Rejected physical paths also have zero alpha but do NOT change ``ray_valid``
    or an externally fixed image-loss denominator.

    ``direct_mode`` is explicit: ``unbounded`` calls the frozen half-ray response
    in its original operation order; ``finite`` bounds depth at the first
    nominal-box exit. Reflection depths end at the next nominal-box exit.
    Gradients are piecewise smooth, with kinks at endpoint clamps and the old
    body scale/degeneracy clamps. CPU float64 is tested; other devices use the
    same tensor code but require their own numerical validation before use.

    This is deliberately not an adapter for graph_fit/fast_union/CUDA captures:
    those frozen paths bypass ``render`` and would omit the new optical paths.
    A future caller must explicitly select direct_union/unit_union/fresnel_union.
    """

    def __init__(
        self,
        rig: Rig,
        pixels_by_view: Mapping[str, np.ndarray],
        config: RendererConfig | None = None,
        *,
        direct_mode: Literal["unbounded", "finite"],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        if direct_mode not in ("unbounded", "finite"):
            raise ValueError("direct_mode must be 'unbounded' or 'finite'")
        if dtype != torch.float64:
            raise ValueError("this controlled renderer requires torch.float64")
        config = config or RendererConfig()
        if config.gaussians_per_fish != 8:
            raise ValueError("this controlled renderer requires the original eight Gaussians")
        self._base = RefractiveFishRenderer(rig, pixels_by_view, config, device, dtype)
        self.rig, self.views = rig, rig.views
        self.config, self.device, self.dtype = config, self._base.device, dtype
        self.direct_mode = direct_mode
        self.n_pixels, self.pixel_counts = self._base.n_pixels, self._base.pixel_counts.copy()
        self.pixels = self._base.pixels.copy()
        self.ray_valid = self._base.ray_valid
        origins = self._base.ray_origins.detach().cpu().numpy()
        directions = self._base.ray_directions.detach().cpu().numpy()
        valid = self.ray_valid.detach().cpu().numpy()
        traces = []
        for v, view in enumerate(self.views):
            camera = rig.camera(view)
            traces.append(
                trace_y192_reflections(
                    np.where(valid[v, :, None], origins[v], np.nan),
                    np.where(valid[v, :, None], directions[v], np.nan),
                    index_water=camera.index_water,
                    index_glass=camera.index_glass,
                    index_air=camera.index_air,
                )
            )
        self.paths: tuple[ReflectionPaths, ...] = tuple(traces)
        self.finite_direct = self._segment(
            origins,
            directions,
            np.stack([x.first_distance_mm for x in traces]),
            np.stack([x.direct_valid for x in traces]),
        )
        for name in ("inner", "outer"):
            branches = [getattr(x, name) for x in traces]
            segment = self._segment(
                np.stack([x.origins for x in branches]),
                np.stack([x.directions for x in branches]),
                np.stack([x.max_distance_mm for x in branches]),
                np.stack([x.valid for x in branches]),
            )
            weight = np.stack([x.weight for x in branches])
            if not np.isfinite(weight).all() or np.any((weight < 0) | (weight > 1)):
                raise ValueError("fixed branch weights must be finite within [0,1]")
            if np.any(weight[~np.stack([x.valid for x in branches])] != 0):
                raise ValueError("invalid fixed branches must have zero weight")
            setattr(self, name, segment)
            setattr(self, name + "_weight", self._tensor(weight))

    def _tensor(self, value, *, boolean=False):
        return torch.tensor(value, dtype=torch.bool if boolean else self.dtype, device=self.device)

    def _segment(self, origins, directions, maximum_distance, valid):
        if (
            not np.isfinite(origins[valid]).all()
            or not np.isfinite(directions[valid]).all()
            or not np.isfinite(maximum_distance[valid]).all()
            or np.any(maximum_distance[valid] < 0)
            or not np.allclose(np.linalg.norm(directions[valid], axis=-1), 1, rtol=0, atol=1e-10)
        ):
            raise ValueError("valid fixed paths require finite unit geometry and finite lengths")
        # Sanitizing BEFORE arithmetic is essential: a later torch.where alone
        # cannot prevent NaNs in derivatives of an invalid branch's operations.
        return FixedRaySegment(
            self._tensor(np.where(valid[..., None], origins, 0)),
            self._tensor(np.where(valid[..., None], directions, [0.0, 0.0, 1.0])),
            self._tensor(np.where(valid, maximum_distance, 0)),
            self._tensor(valid, boolean=True),
        )

    def gaussians(self, pose_xyz: torch.Tensor | np.ndarray) -> GaussianBodies:
        """The unmodified old body construction with fixed config width/opacity."""
        return self._base.gaussians(pose_xyz)

    def _finite_response(self, bodies, segment, start, stop):
        origins = segment.origins[None, :, start:stop, None, None, :]
        directions = segment.directions[None, :, start:stop, None, None, :]
        tangent = bodies.tangents[:, None, None]
        delta = bodies.centers[:, None, None] - origins
        dp = (delta * tangent).sum(dim=-1)
        vp = (directions * tangent).sum(dim=-1)
        dperp = delta - dp.unsqueeze(-1) * tangent
        vperp = directions - vp.unsqueeze(-1) * tangent
        ia = bodies.axial_sigma_mm[:, None, None].reciprocal().square()
        ir = bodies.radial_sigma_mm[:, None, None].reciprocal().square()
        numerator = (vperp * dperp).sum(dim=-1) * ir
        numerator = numerator + vp * dp * ia
        denominator = vperp.square().sum(dim=-1) * ir
        denominator = denominator + vp.square() * ia
        depth = (numerator / denominator).clamp_min(0)
        depth = torch.minimum(depth, segment.maximum_distance_mm[None, :, start:stop, None, None])
        residual = delta - depth.unsqueeze(-1) * directions
        rp = (residual * tangent).sum(dim=-1)
        rr = residual - rp.unsqueeze(-1) * tangent
        mahalanobis = rr.square().sum(dim=-1) * ir
        mahalanobis = mahalanobis + rp.square() * ia
        alpha = self.config.opacity * torch.exp(-0.5 * mahalanobis)
        alpha = torch.where(segment.valid[None, :, start:stop, None, None], alpha, 0)
        return 1 - (1 - alpha).prod(dim=-1)

    def render(self, pose_xyz: torch.Tensor | np.ndarray) -> ReflectionRenderResult:
        bodies = self.gaussians(pose_xyz)
        if not all(
            bool(torch.isfinite(getattr(bodies, key)).all())
            for key in ("centers", "tangents", "axial_sigma_mm", "radial_sigma_mm")
        ):
            raise FloatingPointError("body construction overflowed")
        chunks = [[], [], []]
        for start in range(0, self.n_pixels, self.config.pixel_chunk_size):
            stop = min(start + self.config.pixel_chunk_size, self.n_pixels)
            if self.direct_mode == "unbounded":
                direct = self._base._ray_response(bodies, start, stop)[0]
            else:
                direct = self._finite_response(bodies, self.finite_direct, start, stop)
            chunks[0].append(direct)
            chunks[1].append(self._finite_response(bodies, self.inner, start, stop))
            chunks[2].append(self._finite_response(bodies, self.outer, start, stop))
        direct, inner, outer = (torch.cat(values, dim=2) for values in chunks)
        if not all(bool(torch.isfinite(x).all()) for x in (direct, inner, outer)):
            raise FloatingPointError("reflection response arithmetic became nonfinite")
        branches = torch.stack((direct, inner, outer), dim=-1)
        weighted = torch.stack(
            (direct, inner * self.inner_weight[None, ..., None],
             outer * self.outer_weight[None, ..., None]), dim=-1
        )
        # Flatten body then branch in C order, matching NumPy's product over the
        # last two axes. No visibility order is mixed across different paths.
        return ReflectionRenderResult(
            direct, inner, outer,
            1 - (1 - direct).prod(dim=-1),
            1 - (1 - branches).flatten(-2).prod(dim=-1),
            1 - (1 - weighted).flatten(-2).prod(dim=-1),
            self.ray_valid,
            self.ray_valid if self.direct_mode == "unbounded" else self.finite_direct.valid,
            self.inner.valid, self.outer.valid, self.inner_weight, self.outer_weight,
            bodies, self.direct_mode,
        )

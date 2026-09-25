"""Union silhouettes with each time slice matched to its own calibrated rays.

The Gaussian construction is reused unchanged. The anisotropic half-ray alpha
below is adapted from ``renderer.RefractiveFishRenderer._ray_response`` only by
adding a matched time axis; it never forms a time-by-time Cartesian product.
This is a renderer, not a motion model, observation admission rule or ID solver.
"""

from __future__ import annotations

import numpy as np
import torch
from renderer import RefractiveFishRenderer, RendererConfig
from torch.utils.checkpoint import checkpoint

ORIGIN = {
    "path": "labs/20260909_three_body/renderer.py",
    "sha256": "dfb585e9d51214a2a3c9447d4bc81da8af80d3451e8fc3ef7a44b1a841732e1c",
    "adapted": "_ray_response half-ray alpha, with matched time axis; diagnostics omitted",
}


class TemporalUnionRenderer:
    """Cache per-frame rays and return union alpha of shape (B,T,V,Nmax).

    ``pixels_by_time`` is a nonempty sequence of view-to-native-(N,2)-pixel
    mappings. Views and entire frames may be empty, but at least one sample is
    required in the window. All frames/views are padded to the common maximum N;
    ``ray_valid[T,V,Nmax]`` distinguishes these unknown samples from background.

    Pose must have shape (B,T,3,3,3). Width can be a scalar, (3,), or (B,3) shared
    through time; a (B,T,3) tensor supplies independent widths. (1,T,3) broadcasts
    across candidate batches. A two-dimensional (T,3) width is not inferred.

    Pixel chunks use RendererConfig.pixel_chunk_size. Time chunks cap temporary
    ray/body products. For gradient-enabled fitting, checkpoint_chunks=True also
    avoids retaining those large products for all times at once: backward
    recomputes each chunk. Output and Gaussian/pose storage still scale with T;
    this is bounded intermediate-memory processing, not a constant-memory claim.
    """

    def __init__(
        self, rig, pixels_by_time, config=None, device="cpu", dtype=torch.float64,
        time_chunk_size=5, checkpoint_chunks=True,
    ):
        if not isinstance(time_chunk_size, int) or time_chunk_size < 1:
            raise ValueError("time_chunk_size must be a positive integer")
        self.config = config or RendererConfig()
        self.rig = rig
        self.views = rig.views
        self.device = torch.device(device)
        self.dtype = dtype
        self.time_chunk_size = time_chunk_size
        self.checkpoint_chunks = bool(checkpoint_chunks)
        frames = list(pixels_by_time)
        if not frames:
            raise ValueError("pixels_by_time must contain at least one frame")
        supplied, counts = [], []
        for frame in frames:
            extra = set(frame) - set(self.views)
            if extra:
                raise ValueError(f"unrecognised camera views: {sorted(extra)}")
            values = {}
            for view in self.views:
                pixels = np.asarray(frame.get(view, np.empty((0, 2))), dtype=float)
                if pixels.ndim != 2 or pixels.shape[1] != 2:
                    raise ValueError("pixel arrays must have shape (N, 2)")
                values[view] = pixels
            supplied.append(values)
            counts.append({view: len(pixels) for view, pixels in values.items()})
        self.n_times = len(frames)
        self.n_pixels = max(count for frame in counts for count in frame.values())
        if self.n_pixels == 0:
            raise ValueError("at least one sampled image pixel is required in the window")
        self.pixel_counts = counts
        cached = []
        for values in supplied:
            padded = {}
            for view, pixels in values.items():
                padded[view] = np.full((self.n_pixels, 2), np.nan)
                padded[view][:len(pixels)] = pixels
            cached.append(RefractiveFishRenderer(
                rig, padded, self.config, device=self.device, dtype=self.dtype))
        self._body_renderer = cached[0]
        self.pixels = np.stack([item.pixels for item in cached])
        self.ray_origins = torch.stack([item.ray_origins for item in cached])
        self.ray_directions = torch.stack([item.ray_directions for item in cached])
        self.ray_valid = torch.stack([item.ray_valid for item in cached])

    def _render_time_chunk(self, pose, width, ray_origins, ray_directions, ray_valid):
        batch, times = pose.shape[:2]
        bodies = self._body_renderer.gaussians(
            pose.reshape(batch * times, 3, 3, 3), width.reshape(batch * times, 3))
        centers = bodies.centers.reshape(batch, times, 3, -1, 3)[:, :, None, None]
        tangent = bodies.tangents.reshape(batch, times, 3, -1, 3)[:, :, None, None]
        inverse_axial_sq = bodies.axial_sigma_mm.reshape(batch, times, 3, -1)
        inverse_axial_sq = inverse_axial_sq[:, :, None, None].reciprocal().square()
        inverse_radial_sq = bodies.radial_sigma_mm.reshape(batch, times, 3, -1)
        inverse_radial_sq = inverse_radial_sq[:, :, None, None].reciprocal().square()
        chunks = []
        for start in range(0, self.n_pixels, self.config.pixel_chunk_size):
            stop = min(start + self.config.pixel_chunk_size, self.n_pixels)
            origins = ray_origins[None, :, :, start:stop, None, None, :]
            directions = ray_directions[None, :, :, start:stop, None, None, :]
            delta = centers - origins
            delta_parallel = (delta * tangent).sum(dim=-1)
            direction_parallel = (directions * tangent).sum(dim=-1)
            delta_perp = delta - delta_parallel.unsqueeze(-1) * tangent
            direction_perp = directions - direction_parallel.unsqueeze(-1) * tangent
            numerator = (direction_perp * delta_perp).sum(dim=-1) * inverse_radial_sq
            numerator = numerator + direction_parallel * delta_parallel * inverse_axial_sq
            denominator = direction_perp.square().sum(dim=-1) * inverse_radial_sq
            denominator = denominator + direction_parallel.square() * inverse_axial_sq
            depth = (numerator / denominator).clamp_min(0)
            residual = delta - depth.unsqueeze(-1) * directions
            residual_parallel = (residual * tangent).sum(dim=-1)
            residual_perp = residual - residual_parallel.unsqueeze(-1) * tangent
            mahalanobis = residual_perp.square().sum(dim=-1) * inverse_radial_sq
            mahalanobis = mahalanobis + residual_parallel.square() * inverse_axial_sq
            alpha = self.config.opacity * torch.exp(-0.5 * mahalanobis)
            valid = ray_valid[None, :, :, start:stop, None, None]
            alpha = torch.where(valid, alpha, 0)
            fish_alpha = 1 - (1 - alpha).prod(dim=-1)
            chunks.append(1 - (1 - fish_alpha).prod(dim=-1))
        return torch.cat(chunks, dim=3)

    def render(self, pose_xyz, width_ratio=None):
        pose = torch.as_tensor(pose_xyz, dtype=self.dtype, device=self.device)
        if (pose.ndim != 5 or pose.shape[1:] != (self.n_times, 3, 3, 3)
                or pose.shape[0] < 1):
            raise ValueError("pose_xyz must have shape (B,T,3,3,3) matching cached times")
        batch = pose.shape[0]
        width = torch.as_tensor(self.config.width_ratio if width_ratio is None else width_ratio,
                                dtype=self.dtype, device=self.device)
        if width.ndim == 0 or width.shape == (3,):
            width = width.expand(batch, self.n_times, 3)
        elif width.shape == (batch, 3):
            width = width[:, None].expand(batch, self.n_times, 3)
        elif width.shape == (1, self.n_times, 3):
            width = width.expand(batch, self.n_times, 3)
        elif width.shape != (batch, self.n_times, 3):
            raise ValueError("width_ratio must be scalar, (3,), (B,3), or (B,T,3)")
        use_checkpoint = (self.checkpoint_chunks and torch.is_grad_enabled()
                          and (pose.requires_grad or width.requires_grad))
        chunks = []
        for start in range(0, self.n_times, self.time_chunk_size):
            stop = min(start + self.time_chunk_size, self.n_times)
            args = (pose[:, start:stop], width[:, start:stop], self.ray_origins[start:stop],
                    self.ray_directions[start:stop], self.ray_valid[start:stop])
            if use_checkpoint:
                chunks.append(checkpoint(self._render_time_chunk, *args, use_reentrant=False,
                                         preserve_rng_state=False))
            else:
                chunks.append(self._render_time_chunk(*args))
        return torch.cat(chunks, dim=1)

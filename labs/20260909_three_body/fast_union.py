"""Union-only evaluation using the frozen renderer's exact ray-response math.

This adapter is for optimization steps that consume only a silhouette. It skips
the full render's depth sorting, tied-order averages, visibility, ownership and
diagnostic concatenation. The existing ``_ray_response`` still computes its local
depth/nearest summaries; they are immediately discarded, with no path into the
returned autograd graph. It deliberately does not duplicate or approximate that
helper's Gaussian, refraction, precision or half-ray calculations.

Call the full renderer when visibility/support diagnostics are actually needed.
This function neither sets inference mode nor detaches parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np
    from renderer import RefractiveFishRenderer


def render_union(
    renderer: RefractiveFishRenderer,
    pose_xyz: torch.Tensor | np.ndarray,
    width_ratio: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Return (B, V, N) union alpha, with the original renderer's pixel padding.

    Invalid/unknown pixels remain zero in the union and false in
    ``renderer.ray_valid``. Losses must still respect that validity mask. The
    batch dimension is present even for an unbatched three-fish pose.
    """
    bodies = renderer.gaussians(pose_xyz, width_ratio)
    alpha_chunks = []
    for start in range(0, renderer.n_pixels, renderer.config.pixel_chunk_size):
        stop = min(start + renderer.config.pixel_chunk_size, renderer.n_pixels)
        alpha_chunks.append(renderer._ray_response(bodies, start, stop)[0])
    fish_alpha = torch.cat(alpha_chunks, dim=2)
    return 1 - (1 - fish_alpha).prod(dim=-1)

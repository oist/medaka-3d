"""Color costs for six identity assignments of one frozen three-body geometry.

P[state, identity] names the raw geometric slot. Shape, visibility, widths and
raw observations stay attached to geometry; only the color-template identity
changes. One rendering is sufficient because its visibility is permutation
equivariant. The costs are engineering scores, not calibrated ID probabilities.
"""

from dataclasses import dataclass
from itertools import permutations

import numpy as np
import torch
from appearance_ranking import sample_appearance_pixels
from pixel_appearance_loss import PixelAppearanceConfig, pixel_appearance_loss
from renderer import RefractiveFishRenderer, RendererConfig

from medaka3d.types import VIEWS

PERMUTATIONS = np.asarray(list(permutations(range(3))), dtype=np.int64)


@dataclass(frozen=True)
class IdentityEmissionConfig:
    maximum_samples: int = 512
    width_ratio: float = 0.06
    active_views: tuple[str, ...] = ("top_cam",)

    def __post_init__(self):
        if type(self.maximum_samples) is not int or self.maximum_samples < 1:
            raise ValueError("maximum_samples must be a positive integer")
        if not np.isfinite(self.width_ratio) or self.width_ratio <= 0:
            raise ValueError("width_ratio must be finite and positive")
        if (not self.active_views or len(set(self.active_views)) != len(self.active_views)
                or any(view not in VIEWS for view in self.active_views)):
            raise ValueError("active_views must contain distinct canonical camera names")


@dataclass(frozen=True)
class IdentityEmission:
    costs: np.ndarray  # state (six)
    per_view_costs: np.ndarray  # state, view
    enabled_views: np.ndarray  # view
    sampled_pixels: np.ndarray  # view, before ray exclusion
    valid_pixels: np.ndarray  # view, after ray exclusion
    available_core_pixels: np.ndarray  # view, before the fixed sampling cap
    visible_mass: np.ndarray  # view, RAW slot; never renamed by the chosen identity
    rendering_performed: bool


def score_identity_permutations(
    rig, frame, raw_pose_xyz, templates, enrollment_enabled, *,
    config=None, pixel_config=None, device="cpu",
):
    """Sample independent D2 foreground and score all six fixed-shape assignments.

    The caller supplies frozen causal templates and is responsible for excluding
    enrollment frames from the temporal objective. This function does not read
    a trajectory, human labels, old confidence flags, or another time point.
    """
    config = config or IdentityEmissionConfig()
    pixel_config = pixel_config or PixelAppearanceConfig()
    pose, colors = np.asarray(raw_pose_xyz), np.asarray(templates)
    enabled = np.asarray(enrollment_enabled)
    if pose.shape != (3, 3, 3) or not np.isfinite(pose).all():
        raise ValueError("one finite raw three-body geometry is required")
    if colors.shape != (3, 3, 6) or np.isinf(colors).any():
        raise ValueError("frozen templates must be view,identity,six features")
    if enabled.shape != (3,) or enabled.dtype != bool:
        raise ValueError("three explicit enrollment view flags are required")
    if tuple(rig.views) != tuple(VIEWS):
        raise ValueError("renderer, pixels and templates require the canonical view order")
    permitted = np.asarray([view in config.active_views for view in VIEWS])
    # Incomplete color information makes the whole view unavailable; it cannot
    # become evidence for the remaining two labels by identity elimination.
    enabled = enabled & permitted & np.isfinite(colors).all(axis=(1, 2))
    evidence = sample_appearance_pixels(frame, enabled, config.maximum_samples)
    if not evidence.enabled.any():
        return IdentityEmission(
            np.zeros(6), np.zeros((6, 3)), np.zeros(3, bool), evidence.counts.copy(),
            np.zeros(3, np.int64), evidence.available_core_counts.copy(), np.zeros((3, 3)), False,
        )
    renderer = RefractiveFishRenderer(
        rig, evidence.pixels, RendererConfig(width_ratio=config.width_ratio),
        device=device, dtype=torch.float64,
    )

    def tensor(value):
        return torch.as_tensor(value, device=device, dtype=torch.float64)

    with torch.no_grad():
        rendered = renderer.render(tensor(pose))
        indices = torch.as_tensor(PERMUTATIONS, device=device)
        visible = torch.stack([
            rendered.visible_alpha[0].index_select(-1, index) for index in indices
        ])
        weights = tensor(evidence.weights) * rendered.ray_valid
        scored = pixel_appearance_loss(
            visible, tensor(evidence.features), tensor(colors), weights,
            torch.as_tensor(evidence.enabled, device=device), pixel_config,
        )
        counts = (weights > 0).sum(dim=1).cpu().numpy()
        used = scored.view_enabled.cpu().numpy() & (counts > 0)
        mass = (rendered.visible_alpha[0] * weights[..., None]).sum(dim=1).cpu().numpy()
    return IdentityEmission(
        scored.loss.cpu().numpy(), scored.per_view_loss.cpu().numpy(), used,
        evidence.counts.copy(), counts, evidence.available_core_counts.copy(), mass, True,
    )

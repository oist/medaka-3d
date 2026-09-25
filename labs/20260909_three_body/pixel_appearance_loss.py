"""Differentiable diagnostic color mixture on fixed observed foreground pixels.

The renderer supplies refractive visible_alpha; this module does not project rays
or convert colors. Pixel features must come from pixel_template_assignment's
pixel_features and templates must use that same six-feature scale. The fixed
defaults are uncalibrated engineering values, not an identity probability model.
This module is not integrated into tracking and makes no identity decisions.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PixelAppearanceConfig:
    sigma: float = 0.1
    background_cost: float = 2.0
    outlier_fraction: float = 0.1

    def __post_init__(self):
        for name in ("sigma", "background_cost", "outlier_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite real scalar")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real scalar")
        if self.sigma <= 0 or self.background_cost < 0 or not 0 < self.outlier_fraction < 1:
            raise ValueError("require sigma > 0, background_cost >= 0, and 0 < rho < 1")


@dataclass(frozen=True)
class PixelAppearanceResult:
    loss: torch.Tensor  # B; one fixed foreground-weight denominator across enabled views
    per_view_loss: torch.Tensor  # B,V; view's fixed foreground denominator; empty -> 0
    view_enabled: torch.Tensor  # V; requested AND all three templates complete
    foreground_mass: torch.Tensor  # V; effective foreground weight, independent of pose


def _bounded_finite(value, name):
    if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
        raise ValueError(f"{name} must be finite and in [0,1]")


def pixel_appearance_loss(
    visible_alpha, pixel_features, templates, fgweights, view_enabled, config=None
):
    """Return weighted negative log mixture scores with gradients through visibility.

    q_f = exp(-||x-T_f||^2 / (2 sigma^2)), q_0 = exp(-background_cost)
    p = rho*q_0 + (1-rho)*((1-U)*q_0 + sum_f visible_alpha_f*q_f).

    U is the sum of the three depth-composited visible alphas. Foreground weights
    and templates are observations fixed across candidate poses. A fully hidden
    scene has loss background_cost on nonempty foreground, not zero. This is only
    a foreground color surrogate, not a complete background or unexplained-image
    likelihood. Similar templates cannot certify an identity.

    Inputs share dtype (float32/64) and device, except boolean view_enabled. NaN
    templates mean incomplete enrollment and disable the whole view. Nonfinite
    pixel features are allowed only at zero effective weight; both features and
    templates are safely replaced BEFORE arithmetic so NaN*0 cannot contaminate
    scores or derivatives. Inf templates are invalid. All visibility and weights
    must remain finite even on disabled views, so malformed renderer output fails.

    Roundoff up to 32 dtype eps is accepted for sum(alpha) > 1; only the resulting
    tiny negative uncovered mass is clamped to zero, and p is bounded above by 1.
    The stated mixture is unchanged for mathematically valid interior inputs.
    """
    cfg = config or PixelAppearanceConfig()
    if not isinstance(cfg, PixelAppearanceConfig):
        raise ValueError("config must be PixelAppearanceConfig")
    values = dict(
        visible_alpha=visible_alpha,
        pixel_features=pixel_features,
        templates=templates,
        fgweights=fgweights,
        view_enabled=view_enabled,
    )
    if any(not isinstance(value, torch.Tensor) for value in values.values()):
        raise ValueError("all inputs must be torch tensors")
    if (
        visible_alpha.ndim != 4
        or visible_alpha.shape[-1] != 3
        or visible_alpha.shape[0] < 1
        or visible_alpha.shape[1] < 1
    ):
        raise ValueError("visible_alpha must have shape B,V,N,3 with positive B,V")
    _, views, pixels, _ = visible_alpha.shape
    if (
        pixel_features.shape != (views, pixels, 6)
        or templates.shape != (views, 3, 6)
        or fgweights.shape != (views, pixels)
        or view_enabled.shape != (views,)
    ):
        raise ValueError("expected pixel features V,N,6, templates V,3,6, weights V,N, views V")
    if visible_alpha.dtype not in (torch.float32, torch.float64):
        raise ValueError("visibility and features must use float32 or float64")
    if view_enabled.dtype != torch.bool:
        raise ValueError("view_enabled must be boolean")
    if any(value.device != visible_alpha.device for value in values.values()):
        raise ValueError("all inputs must share a device")
    if any(
        value.dtype != visible_alpha.dtype
        for name, value in values.items()
        if name != "view_enabled"
    ):
        raise ValueError("visibility, features, templates and weights must share a dtype")
    finfo = torch.finfo(visible_alpha.dtype)
    if math.log(cfg.outlier_fraction) - cfg.background_cost < math.log(finfo.tiny):
        raise ValueError("rho*exp(-background_cost) must remain a positive normal dtype value")
    variance_twice = 2 * cfg.sigma * cfg.sigma
    if not math.isfinite(variance_twice) or variance_twice < finfo.tiny:
        raise ValueError("2*sigma^2 must be finite and representable in the input dtype")
    _bounded_finite(visible_alpha, "visible_alpha")
    union = visible_alpha.sum(dim=-1)
    if bool((union > 1 + 32 * finfo.eps).any()):
        raise ValueError("visible_alpha must sum to at most one per pixel")
    if not bool(torch.isfinite(fgweights).all()) or bool((fgweights < 0).any()):
        raise ValueError("foreground weights must be finite and nonnegative")
    if bool(torch.isinf(templates).any()):
        raise ValueError("template infinities are invalid; use NaN for missing enrollment")
    complete = torch.isfinite(templates).all(dim=(1, 2))
    effective_views = view_enabled & complete
    effective_weights = torch.where(
        effective_views[:, None], fgweights, torch.zeros_like(fgweights)
    )
    used = effective_weights > 0
    safe_features = torch.where(used[..., None], pixel_features, torch.zeros_like(pixel_features))
    safe_templates = torch.where(
        effective_views[:, None, None], templates, torch.zeros_like(templates)
    )
    _bounded_finite(safe_features, "positively weighted pixel features")
    _bounded_finite(safe_templates, "enabled templates")
    distance_sq = ((safe_features[:, :, None] - safe_templates[:, None]) ** 2).sum(dim=-1)
    quality = torch.exp(-distance_sq / variance_twice)
    q0 = math.exp(-cfg.background_cost)
    mixture = cfg.outlier_fraction * q0 + (1 - cfg.outlier_fraction) * (
        torch.clamp(1 - union, min=0) * q0 + (visible_alpha * quality[None]).sum(dim=-1)
    )
    pixel_loss = -torch.log(torch.clamp(mixture, max=1))
    weighted = pixel_loss * effective_weights[None]
    mass = effective_weights.sum(dim=-1)
    if not bool(torch.isfinite(mass.sum())):
        raise ValueError("total foreground weight must remain finite in the input dtype")
    # clamp_min(1) would change scores for valid normalized weights summing below 1.
    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
    per_view = weighted.sum(dim=-1) / safe_mass[None]
    total_mass = mass.sum()
    safe_total = torch.where(total_mass > 0, total_mass, torch.ones_like(total_mass))
    loss = weighted.sum(dim=(1, 2)) / safe_total
    return PixelAppearanceResult(loss, per_view, effective_views, mass)

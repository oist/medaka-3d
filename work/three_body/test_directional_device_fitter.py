"""Analytic correction must match autograd of the registered directional cost."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"))
from directional_chest_fitter import directional_center
from directional_device_fitter import center_correction


def test_mixed_parent_axes_and_ages_analytic_correction_matches_autograd():
    rng = np.random.default_rng(9527)
    delta = torch.tensor(rng.normal(size=(7, 3, 3)) * 20, dtype=torch.float64, requires_grad=True)
    axis = rng.normal(size=(7, 3, 3))
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    axis[0, 0] = 0  # disabled parents have no ray axis
    axes = torch.tensor(axis, dtype=torch.float64)
    sigma = torch.tensor(rng.uniform(0.8, 3, size=(7, 3)), dtype=torch.float64)
    parallel_sigma = torch.maximum(
        sigma, torch.tensor(rng.uniform(0.8, 6.8, size=(7, 3)), dtype=torch.float64)
    )
    enabled = torch.tensor(rng.uniform(size=(7, 3)) > 0.3)
    enabled[0, 0] = False
    original = ((delta / sigma[..., None]) ** 2).sum(-1).mean(-1)
    direct = directional_center(delta, sigma, axes, parallel_sigma, enabled) - original
    reference = torch.autograd.grad(direct.sum(), delta)[0]
    difference, gradient, _ = center_correction(
        delta.detach(), sigma, axes, parallel_sigma, enabled
    )
    torch.testing.assert_close(difference, direct, atol=1e-12, rtol=0)
    torch.testing.assert_close(gradient, reference, atol=1e-12, rtol=0)


def test_disabled_correction_is_exactly_zero():
    delta = torch.tensor([[[2.0, 3.0, -4.0]] * 3], dtype=torch.float64)
    sigma = torch.full((1, 3), 0.8, dtype=torch.float64)
    axes = torch.zeros_like(delta)
    difference, gradient, _ = center_correction(
        delta, sigma, axes, sigma * 8, torch.zeros((1, 3), dtype=torch.bool)
    )
    assert torch.equal(difference, torch.zeros_like(difference))
    assert torch.equal(gradient, torch.zeros_like(gradient))

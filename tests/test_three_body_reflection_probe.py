"""Finite Gaussian interval and fixed-response algebra, without real images."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import minimize_scalar

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, LAB / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = load("reflection_probe_pure_test", "reflection_probe.py")
renderer = load("reflection_probe_original_renderer_test", "renderer.py")


def body(center=(0.0, 0.0, 20.0)):
    return probe.BodyArrays(
        np.tile(center, (3, 1, 1)),
        np.tile([1.0, 0.0, 0.0], (3, 1, 1)),
        np.full((3, 1), 4.0),
        np.full((3, 1), 2.0),
    )


def test_finite_endpoint_against_independent_scalar_minimum():
    b = body()
    o, d = np.array([[2.0, 1.0, 0.0]]), np.array([[0.0, 0.0, 1.0]])
    alpha = probe.finite_response(b, o, d, np.array([12.0]), np.array([True]))

    def q(t):
        return (2 / 4) ** 2 + (1 / 2) ** 2 + ((20 - t) / 2) ** 2

    best = minimize_scalar(q, bounds=(0, 12), method="bounded")
    assert best.x == pytest.approx(12, abs=1e-5)
    np.testing.assert_allclose(alpha, 0.85 * np.exp(-0.5 * q(12)), atol=1e-15)
    control = probe.finite_response(b, o, d, np.array([np.inf]), np.array([True]))
    assert np.all(alpha < control)


def test_half_ray_and_zero_length_do_not_see_behind_origin():
    b = body(center=(0.0, 0.0, -5.0))
    result = probe.finite_response(
        b,
        np.zeros((2, 3)),
        np.tile([0.0, 0.0, 1.0], (2, 1)),
        np.array([0.0, np.inf]),
        np.ones(2, bool),
    )
    np.testing.assert_allclose(result, 0.85 * np.exp(-0.5 * (5 / 2) ** 2), atol=1e-15)


def test_invalid_nan_branch_is_zero_but_valid_nan_rejected():
    o = np.full((1, 3), np.nan)
    np.testing.assert_array_equal(
        probe.finite_response(body(), o, o, np.array([np.nan]), np.array([False])), np.zeros((1, 3))
    )
    with pytest.raises(ValueError, match="unit finite"):
        probe.finite_response(body(), o, o, np.array([np.nan]), np.array([True]))


def test_unbounded_response_matches_frozen_float64_renderer(rig):
    pose = np.array(
        [
            [[50.0, 70.0, 80.0], [53.0, 72.0, 83.0], [65.0, 74.0, 85.0]],
            [[100.0, 120.0, 140.0], [101.0, 119.0, 135.0], [103.0, 110.0, 125.0]],
            [[140.0, 150.0, 40.0], [137.0, 148.0, 41.0], [125.0, 140.0, 45.0]],
        ]
    )
    rng = np.random.default_rng(611)
    pixels = {
        cam.view: np.vstack(
            [cam.project(pose.reshape(-1, 3)), rng.uniform(0, 1200, (41, 2)), [[np.nan, np.nan]]]
        )
        for cam in rig.cameras
    }
    engine = renderer.RefractiveFishRenderer(rig, pixels, dtype=torch.float64)
    old = engine.render(pose)
    new = probe.finite_response(
        probe.BodyArrays.from_torch(old.bodies),
        engine.ray_origins.numpy(),
        engine.ray_directions.numpy(),
        np.full(engine.ray_valid.shape, np.inf),
        engine.ray_valid.numpy(),
    )
    np.testing.assert_allclose(new, old.per_fish_alpha[0].numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        1 - np.prod(1 - new, axis=-1), old.union_alpha[0].numpy(), rtol=0, atol=1e-12
    )


def test_branch_weights_monotonic_and_permutation_symmetric():
    rng = np.random.default_rng(742)
    a, b, c = rng.uniform(0, 1, (3, 17, 3))
    wi, wo = rng.uniform(0, 1, (2, 17))
    result = probe.combine_branches(a, b, c, wi, wo)
    assert np.all(result["finite_direct"] <= result["both_fresnel"] + 1e-15)
    assert np.all(result["both_fresnel"] <= result["both_unit"] + 1e-15)
    zero = probe.combine_branches(a, b, c, np.zeros(17), np.zeros(17))
    np.testing.assert_array_equal(zero["finite_direct"], zero["both_fresnel"])
    shuffled = probe.combine_branches(a[:, [2, 0, 1]], b[:, [2, 0, 1]], c[:, [2, 0, 1]], wi, wo)
    for key in result:
        np.testing.assert_allclose(result[key], shuffled[key], rtol=0, atol=1e-15)


def test_reject_malformed_weights_and_unit_ray():
    with pytest.raises(ValueError, match="bounded"):
        probe.combine_branches(
            np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)), np.ones(2) * 1.1, np.ones(2)
        )
    with pytest.raises(ValueError, match="unit finite"):
        probe.finite_response(
            body(),
            np.zeros((1, 3)),
            np.array([[0.0, 0.0, 2.0]]),
            np.array([20.0]),
            np.array([True]),
        )

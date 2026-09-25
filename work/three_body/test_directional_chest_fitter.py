"""Geometry and gradients of the new directional term, independent of full fitting."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"))
from directional_chest_fitter import directional_center


def evaluate(delta, *, enabled=True, parallel=2.0):
    value = torch.tensor([[delta]], dtype=torch.float64, requires_grad=True)
    loss = directional_center(
        value,
        torch.tensor([[0.8]], dtype=torch.float64),
        torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64),
        torch.tensor([[parallel]], dtype=torch.float64),
        torch.tensor([[enabled]]),
    )
    loss.sum().backward()
    return loss.item(), value.grad.numpy()[0, 0]


def test_transverse_precision_does_not_change():
    loss, gradient = evaluate([3.0, 4.0, 0.0])
    np.testing.assert_allclose(loss, 25 / 0.8**2, atol=1e-12, rtol=0)
    np.testing.assert_allclose(gradient, [6 / 0.8**2, 8 / 0.8**2, 0.0], atol=1e-12, rtol=0)


def test_only_parallel_precision_softens():
    loss, gradient = evaluate([0.0, 0.0, 5.0])
    np.testing.assert_allclose(loss, 25 / 2**2, atol=1e-12, rtol=0)
    np.testing.assert_allclose(gradient, [0.0, 0.0, 10 / 2**2], atol=1e-12, rtol=0)


def test_disabled_or_equal_scale_keeps_original_value_and_gradient():
    expected = evaluate([3.0, -4.0, 5.0], enabled=False)
    equal = evaluate([3.0, -4.0, 5.0], parallel=0.8)
    assert expected[0] == equal[0]
    assert expected[1].tobytes() == equal[1].tobytes()
    np.testing.assert_allclose(expected[0], 50 / 0.8**2, atol=1e-12, rtol=0)


def test_oblique_axis_has_expected_elliptical_quadratic_gradient():
    vector = np.asarray([2.0, -3.0, 5.0])
    axis = np.asarray([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    projection = float(np.dot(vector, axis))
    expected_gradient = 2 * (vector - projection * axis) / 0.8**2 + 2 * projection * axis / 3.0**2
    value = torch.tensor([[vector.tolist()]], dtype=torch.float64, requires_grad=True)
    result = directional_center(
        value,
        torch.tensor([[0.8]], dtype=torch.float64),
        torch.tensor([[axis.tolist()]], dtype=torch.float64),
        torch.tensor([[3.0]], dtype=torch.float64),
        torch.tensor([[True]]),
    )
    result.sum().backward()
    np.testing.assert_allclose(value.grad.numpy()[0, 0], expected_gradient, atol=1e-12, rtol=0)

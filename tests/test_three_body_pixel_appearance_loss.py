"""Pure color mixture invariants and CPU autograd, without tracking or identity labels."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load(name):
    spec = importlib.util.spec_from_file_location(f"pixel_loss_test_{name}", LAB / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load("pixel_appearance_loss")
FEATURES = load("pixel_template_assignment")


def inputs(dtype=torch.float64, views=2, batch=2):
    # Invoke the existing conversion; do not reconstruct the six-feature transform.
    colors = np.array([[25, 70, 220], [75, 190, 65], [215, 85, 40]], np.uint8)
    templates = torch.tensor(FEATURES.pixel_features(colors), dtype=dtype)
    templates = templates[None].expand(views, -1, -1).clone()
    features = templates.clone()
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=dtype)[None].expand(views, -1).clone()
    visible = (0.75 * torch.eye(3, dtype=dtype))[None, None].expand(batch, views, -1, -1).clone()
    enabled = torch.ones(views, dtype=torch.bool)
    return visible, features, templates, weights, enabled


def test_scalar_mixture_matches_independent_formula_and_fixed_weights():
    visible, features, templates, weights, enabled = inputs(batch=1, views=1)
    result = M.pixel_appearance_loss(visible, features, templates, weights, enabled)
    expected = []
    for pixel in range(3):
        q = [
            math.exp(-sum(float(v) ** 2 for v in (features[0, pixel] - t)) / 0.02)
            for t in templates[0]
        ]
        alpha = visible[0, 0, pixel].tolist()
        q0 = math.exp(-2)
        probability = 0.1 * q0 + 0.9 * (
            (1 - sum(alpha)) * q0 + sum(a * f for a, f in zip(alpha, q, strict=True))
        )
        expected.append(-math.log(probability))
    total = sum(float(w) * loss for w, loss in zip(weights[0], expected, strict=True))
    torch.testing.assert_close(result.loss, torch.tensor([total], dtype=torch.float64))
    torch.testing.assert_close(result.per_view_loss[0, 0], result.loss[0])
    torch.testing.assert_close(result.foreground_mass, torch.ones(1, dtype=torch.float64))


def test_matching_colors_cost_less_than_wrong_assignment_and_hidden_scene():
    args = inputs()
    correct = M.pixel_appearance_loss(*args)
    wrong = M.pixel_appearance_loss(args[0], args[1], args[2][:, [1, 2, 0]], *args[3:])
    hidden = M.pixel_appearance_loss(torch.zeros_like(args[0]), *args[1:])
    assert (correct.loss < hidden.loss).all()
    assert (hidden.loss < wrong.loss).all()
    torch.testing.assert_close(hidden.loss, torch.full((2,), 2.0, dtype=torch.float64))


def test_simultaneous_template_and_fish_permutation_preserves_scores_and_gradient():
    visible, *other = inputs()
    visible.requires_grad_()
    result = M.pixel_appearance_loss(visible, *other)
    (gradient,) = torch.autograd.grad(result.loss.sum(), visible)
    permutation = torch.tensor([2, 0, 1])
    permuted = visible.detach()[..., permutation].clone().requires_grad_()
    features, templates, weights, enabled = other
    revised = M.pixel_appearance_loss(
        permuted, features, templates[:, permutation], weights, enabled
    )
    (revised_gradient,) = torch.autograd.grad(revised.loss.sum(), permuted)
    torch.testing.assert_close(revised.loss, result.loss, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(revised_gradient, gradient[..., permutation], atol=1e-14, rtol=1e-14)


def test_identical_templates_cannot_distinguish_fish_ids():
    visible, features, templates, weights, enabled = inputs()
    templates = templates[:, :1].expand(-1, 3, -1).clone()
    original = M.pixel_appearance_loss(visible, features, templates, weights, enabled)
    swapped = M.pixel_appearance_loss(
        visible[..., [1, 0, 2]], features, templates, weights, enabled
    )
    torch.testing.assert_close(swapped.loss, original.loss, atol=1e-14, rtol=1e-14)


def test_disabled_or_incomplete_view_is_neutral_before_nan_arithmetic():
    args = list(inputs(views=3))
    args[0].requires_grad_()
    args[4][1] = False
    args[1][1:] = float("nan")
    args[2][1] = float("nan")
    args[2][2, 1, 4] = float("nan")  # One incomplete template disables all three fish in this view.
    result = M.pixel_appearance_loss(*args)
    reference = M.pixel_appearance_loss(
        args[0][:, :1], args[1][:1], args[2][:1], args[3][:1], args[4][:1]
    )
    torch.testing.assert_close(result.loss, reference.loss)
    assert result.view_enabled.tolist() == [True, False, False]
    assert not result.per_view_loss[:, 1:].any()
    (gradient,) = torch.autograd.grad(result.loss.sum(), args[0])
    assert torch.isfinite(gradient).all() and not gradient[:, 1:].any()


@pytest.mark.parametrize("case", ["disabled", "zero_weight", "empty_pixels"])
def test_empty_evidence_returns_differentiable_zero(case):
    visible, features, templates, weights, enabled = inputs()
    if case == "disabled":
        enabled[:] = False
        templates[:] = float("nan")
        features[:] = float("nan")
    elif case == "zero_weight":
        weights[:] = 0
        features[:] = float("nan")
    else:
        visible, features, weights = visible[:, :, :0], features[:, :0], weights[:, :0]
    visible = visible.clone().requires_grad_()
    result = M.pixel_appearance_loss(visible, features, templates, weights, enabled)
    assert torch.equal(result.loss, torch.zeros(2, dtype=torch.float64))
    assert not result.foreground_mass.any()
    result.loss.sum().backward()
    assert visible.grad is not None and not visible.grad.any()


def test_zero_weight_pixel_nan_does_not_contaminate_value_or_gradient():
    visible, features, templates, weights, enabled = inputs()
    weights[:, 1] = 0
    clean = M.pixel_appearance_loss(visible, features, templates, weights, enabled)
    features[:, 1] = float("nan")
    visible.requires_grad_()
    result = M.pixel_appearance_loss(visible, features, templates, weights, enabled)
    torch.testing.assert_close(result.loss, clean.loss)
    result.loss.sum().backward()
    assert torch.isfinite(visible.grad).all()
    assert not visible.grad[:, :, 1].any()


def test_normalized_small_foreground_weights_keep_fixed_common_denominator():
    visible, features, templates, weights, enabled = inputs()
    visible[:, 1] = 0
    weights[0] *= 0.01
    weights[1] *= 0.03
    result = M.pixel_appearance_loss(visible, features, templates, weights, enabled)
    expected = (result.per_view_loss[:, 0] + 3 * result.per_view_loss[:, 1]) / 4
    torch.testing.assert_close(result.loss, expected)
    scaled = M.pixel_appearance_loss(visible, features, templates, weights * 0.01, enabled)
    torch.testing.assert_close(scaled.loss, result.loss)
    torch.testing.assert_close(
        result.foreground_mass, torch.tensor([0.01, 0.03], dtype=torch.float64)
    )


def test_double_precision_visibility_gradcheck():
    visible, features, templates, weights, enabled = inputs()
    visible[:] = torch.tensor([0.08, 0.17, 0.21], dtype=torch.float64)
    visible.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda alpha: M.pixel_appearance_loss(alpha, features, templates, weights, enabled).loss,
        (visible,),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
    )


def test_gradient_propagates_through_depth_composited_gaussian_body_positions():
    _, features, templates, weights, enabled = inputs(views=1, batch=1)
    pixels = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
    positions = torch.tensor([-0.4, 0.1, 0.4], dtype=torch.float64, requires_grad=True)

    def render_and_score(body_positions):
        raw = 0.8 * torch.exp(-((pixels[:, None] - body_positions[None]) ** 2) / 0.08)
        visible = torch.stack(
            (raw[:, 0], raw[:, 1] * (1 - raw[:, 0]), raw[:, 2] * (1 - raw[:, 0]) * (1 - raw[:, 1])),
            dim=-1,
        )
        return M.pixel_appearance_loss(
            visible[None, None], features, templates, weights, enabled
        ).loss

    assert torch.autograd.gradcheck(render_and_score, (positions,), eps=1e-6, atol=1e-6, rtol=1e-4)
    before = render_and_score(positions)
    (gradient,) = torch.autograd.grad(before.sum(), positions)
    assert torch.isfinite(gradient).all() and (gradient.abs() > 1e-5).all()
    after = render_and_score(positions.detach() - 0.001 * gradient)
    assert (after < before).all()


def test_float32_and_float64_scores_and_gradients_agree():
    outputs = []
    for dtype in (torch.float32, torch.float64):
        visible, *other = inputs(dtype=dtype)
        visible[:] = torch.tensor([0.08, 0.17, 0.21], dtype=dtype)
        visible.requires_grad_()
        result = M.pixel_appearance_loss(visible, *other)
        (gradient,) = torch.autograd.grad(result.loss.sum(), visible)
        outputs.append((result.loss, gradient))
    for a, b in zip(*outputs, strict=True):
        torch.testing.assert_close(a.double(), b, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "mutation",
    [
        "alpha_nan",
        "alpha_negative",
        "alpha_over_one",
        "sum_over_one",
        "weights_nan",
        "weights_negative",
        "used_feature_nan",
        "feature_range",
        "template_inf",
        "template_range",
        "shape",
        "dtype",
        "view_dtype",
    ],
)
def test_malformed_inputs_fail_loudly(mutation):
    args = list(inputs())
    if mutation == "alpha_nan":
        args[0][0, 0, 0, 0] = float("nan")
    elif mutation == "alpha_negative":
        args[0][0, 0, 0, 0] = -0.01
    elif mutation == "alpha_over_one":
        args[0][0, 0, 0, 0] = 1.01
    elif mutation == "sum_over_one":
        args[0][0, 0, 0] = 0.5
    elif mutation == "weights_nan":
        args[3][0, 0] = float("nan")
    elif mutation == "weights_negative":
        args[3][0, 0] = -1
    elif mutation == "used_feature_nan":
        args[1][0, 0, 0] = float("nan")
    elif mutation == "feature_range":
        args[1][0, 0, 0] = 1.1
    elif mutation == "template_inf":
        args[2][0, 0, 0] = float("inf")
    elif mutation == "template_range":
        args[2][0, 0, 0] = -0.1
    elif mutation == "shape":
        args[1] = args[1][..., :5]
    elif mutation == "dtype":
        args[3] = args[3].float()
    else:
        args[4] = args[4].float()
    with pytest.raises(ValueError):
        M.pixel_appearance_loss(*args)


def test_roundoff_only_union_excess_has_nonnegative_finite_loss():
    visible, features, templates, weights, enabled = inputs()
    visible[:] = torch.tensor([0.5, 0.5, 4 * torch.finfo(torch.float64).eps])
    result = M.pixel_appearance_loss(visible, features, templates, weights, enabled)
    assert torch.isfinite(result.loss).all() and (result.loss >= 0).all()


@pytest.mark.parametrize(
    "changes",
    [
        dict(sigma=0),
        dict(sigma=float("nan")),
        dict(background_cost=-1),
        dict(outlier_fraction=0),
        dict(outlier_fraction=1),
        dict(sigma=True),
    ],
)
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        M.PixelAppearanceConfig(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        dict(sigma=1e308),
        dict(sigma=1e-200),
        dict(background_cost=1000),
        dict(outlier_fraction=1e-320),
    ],
)
def test_unrepresentable_config_fails_before_nonfinite_log_or_gradients(changes):
    with pytest.raises(ValueError):
        M.pixel_appearance_loss(*inputs(), config=M.PixelAppearanceConfig(**changes))

"""Independent CPU loss/gradient and frozen-dispatch checks; no fitting or GPU."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import mask_transport_fitter as m  # noqa: E402


def loss_inputs():
    own = torch.zeros((2, 3, 3, 3), dtype=torch.float64)
    own[:, 0, 0, 0] = torch.tensor([0.25, 0.75])
    own[:, 0, 1, 1] = torch.tensor([0.5, 0.25])
    own[:, 1, 0, 2] = torch.tensor([0.75, 0.25])
    target = torch.zeros_like(own)
    target[:, 0, 0, 0] = 0.5
    target[:, 0, 1, 1] = 0.25
    target[:, 1, 0, 2] = 1
    target[:, 1, 1:, 0] = 1
    foreground = torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=torch.float64)
    weight = torch.tensor([[0.6, 0.4, 17.0], [1.0, 1000.0, 1000.0], [0, 0, 0]], dtype=torch.float64)
    valid = torch.tensor([[True, True, True], [True, False, True], [False, False, False]])
    return own.requires_grad_(), target, foreground, weight, valid


def pad_views(own, target, foreground, weight, valid):
    """Unobserved cameras add no evidence, without changing the fixed denominator."""
    v = own.shape[1]
    n = own.shape[2]
    return (
        torch.cat((own, own.new_zeros((len(own), 3 - v, n, 3))), dim=1),
        torch.cat((target, target.new_zeros((len(target), 3 - v, n, 3))), dim=1),
        torch.cat((foreground.to(own.dtype), own.new_zeros((3 - v, n)))),
        torch.cat((weight, weight.new_zeros((3 - v, n)))),
        torch.cat((valid, torch.zeros((3 - v, n), dtype=torch.bool))),
    )


def byte_snapshot(values):
    return [(value.dtype, tuple(value.shape), value.detach().numpy().tobytes()) for value in values]


def test_three_effective_pixels_have_hand_computed_scores_and_gradients():
    values = loss_inputs()
    before = byte_snapshot(values)
    score = m.ownership_loss(*values)
    expected = torch.tensor([41 / 480, 17 / 80], dtype=torch.float64)
    assert score.shape == (2,) and score.dtype == torch.float64
    torch.testing.assert_close(score, expected, rtol=0, atol=1e-15)
    score.sum().backward()
    expected_gradient = torch.zeros_like(values[0])
    expected_gradient[:, 0, 0, 0] = torch.tensor([-0.15, -0.05], dtype=torch.float64)
    expected_gradient[:, 0, 1, 1] = torch.tensor([-1 / 30, -0.05], dtype=torch.float64)
    expected_gradient[:, 1, 0, 2] = torch.tensor([-1 / 6, -0.5], dtype=torch.float64)
    torch.testing.assert_close(values[0].grad, expected_gradient, rtol=0, atol=1e-15)
    assert byte_snapshot(values) == before


def test_same_total_ownership_but_wrong_body_has_positive_assignment_cost():
    own = torch.tensor([[[[1.0, 0.0, 0.0]]], [[[0.0, 1.0, 0.0]]]], dtype=torch.float64)
    target = torch.tensor([[[[1.0, 0.0, 0.0]]]] * 2, dtype=torch.float64)
    flags = torch.ones((1, 1), dtype=torch.bool)
    weight = torch.ones((1, 1), dtype=torch.float64)
    torch.testing.assert_close(own.sum(-1)[0], own.sum(-1)[1], rtol=0, atol=0)
    score = m.ownership_loss(*pad_views(own, target, flags, weight, flags))
    torch.testing.assert_close(
        score, torch.tensor([0.0, 1 / 3], dtype=torch.float64), rtol=0, atol=0
    )


def test_proposal_batches_use_their_own_immutable_target_and_do_not_share_gradient():
    own = torch.tensor([[[[0.75, 0.25, 0.0]]]] * 2, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0, 0.0]]], [[[0.0, 1.0, 0.0]]]], dtype=torch.float64)
    flags = torch.ones((1, 1), dtype=torch.bool)
    score = m.ownership_loss(
        *pad_views(own, target, flags, torch.ones((1, 1), dtype=torch.float64), flags)
    )
    torch.testing.assert_close(
        score, torch.tensor([1 / 48, 3 / 16], dtype=torch.float64), rtol=0, atol=0
    )
    score[0].backward()
    assert torch.count_nonzero(own.grad[1]).item() == 0
    assert own.grad[0, 0, 0, 0] == -1 / 6
    assert torch.count_nonzero(own.grad[0]).item() == 1


def test_target_and_sampling_mass_are_not_renormalized():
    own, target, foreground, weight, valid = loss_inputs()
    baseline = m.ownership_loss(own, target, foreground, weight, valid)
    scaled = m.ownership_loss(own, target * 0.125, foreground, weight, valid)
    torch.testing.assert_close(scaled, baseline * 0.125, rtol=0, atol=1e-15)
    doubled = m.ownership_loss(
        torch.cat((own, own), dim=2),
        torch.cat((target, target), dim=2),
        torch.cat((foreground, foreground), dim=1),
        torch.cat((weight, weight), dim=1),
        torch.cat((valid, valid), dim=1),
    )
    torch.testing.assert_close(doubled, baseline * 2, rtol=0, atol=1e-15)


def test_loss_derivative_matches_finite_differences_through_a_smooth_ownership_map():
    logits = torch.tensor(
        [[[[0.3, -0.2, 0.1], [1.0, -0.5, 0.2]]]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor([[[[0.5, 0.0, 0.0], [0.0, 0.25, 0.0]]]], dtype=torch.float64)
    foreground = torch.ones((1, 2), dtype=torch.bool)
    weights = torch.tensor([[0.4, 0.6]], dtype=torch.float64)
    valid = foreground.clone()

    def loss(argument):
        return m.ownership_loss(
            *pad_views(torch.softmax(argument, dim=-1), target, foreground, weights, valid)
        )

    assert torch.autograd.gradcheck(loss, (logits,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_finite_zero_evidence_has_exact_zero_cost_and_ownership_gradient():
    own, target, foreground, weight, valid = loss_inputs()
    score = m.ownership_loss(own, torch.zeros_like(target), foreground, weight, valid)
    assert torch.equal(score, torch.zeros(2, dtype=torch.float64))
    score.sum().backward()
    assert torch.equal(own.grad, torch.zeros_like(own))


@pytest.mark.parametrize("evidence_index", [1, 2, 3])
def test_evidence_cannot_be_optimized_along_with_ownership(evidence_index):
    values = list(loss_inputs())
    values[evidence_index] = values[evidence_index].requires_grad_()
    with pytest.raises(ValueError, match="detached evidence"):
        m.ownership_loss(*values)
    assert all(value.grad is None for value in values)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_ownership_is_rejected_even_where_ray_has_zero_weight(invalid):
    values = list(loss_inputs())
    values[0] = values[0].detach()
    values[0][:, 1, 1, :] = invalid
    assert not values[-1][1, 1]
    with pytest.raises(ValueError, match="finite floating ownership"):
        m.ownership_loss(*values)


def test_same_pixel_cannot_constrain_two_target_bodies():
    values = list(loss_inputs())
    values[1][:, 0, 0, 1] = 0.1
    with pytest.raises(ValueError, match="multiple bodies"):
        m.ownership_loss(*values)


def old_result():
    return (
        np.arange(54, dtype=np.float64).reshape(2, 3, 3, 3),
        np.array([-0.0, 1.0], np.float64),
        np.arange(10, dtype=np.float64).reshape(2, 5),
        np.array([[True, False, True], [False, True, False]]),
    )


def forbidden(*args, **kwargs):
    raise AssertionError("no extra renderer, ray assignment, graph or fit is permitted")


@pytest.fixture
def forbid_extra_work(monkeypatch):
    for name in ("RefractiveFishRenderer", "GraphFit", "assigned_rays", "objective"):
        monkeypatch.setattr(m, name, forbidden)


@pytest.mark.parametrize("raise_error", [False, True])
def test_coefficient_zero_delegates_before_reading_target_and_preserves_return_or_exception(
    monkeypatch, forbid_extra_work, raise_error
):
    tracker, proposals, states, rays, targets = (SimpleNamespace() for _ in range(5))
    target = object()  # No shape/dtype/array protocol: lambda zero must not inspect it.
    expected = old_result()
    error = RuntimeError("same original failure")
    calls = []

    def frozen(*args):
        assert all(
            a is b for a, b in zip(args, (tracker, proposals, states, rays, targets), strict=True)
        )
        calls.append(True)
        if raise_error:
            raise error
        return expected

    monkeypatch.setattr(m, "frozen_fit", frozen)
    if raise_error:
        with pytest.raises(RuntimeError) as caught:
            m.fit_with_ownership(tracker, proposals, states, rays, targets, target, coefficient=0)
        assert caught.value is error
    else:
        result = m.fit_with_ownership(
            tracker, proposals, states, rays, targets, target, coefficient=0
        )
        assert result is expected
        assert tracker.last_owner_fit["reason"] == "coefficient_zero"
        assert tracker.last_owner_fit["base_score"].tobytes() == expected[1].tobytes()
        assert tracker.last_owner_fit["total_score"].tobytes() == expected[1].tobytes()
        assert not np.shares_memory(tracker.last_owner_fit["total_score"], expected[1])
    assert calls == [True]


@pytest.mark.parametrize("zero", ["target", "foreground", "sampling_weight"])
def test_zero_effective_mass_delegates_before_any_extra_renderer(
    monkeypatch, forbid_extra_work, zero
):
    tracker, states, rays = (SimpleNamespace() for _ in range(3))
    proposals = [object(), object()]
    targets = SimpleNamespace(target=np.ones((3, 2)), weights=np.ones((3, 2)))
    target = np.zeros((2, 3, 2, 3), np.float64)
    target[..., 0] = 0.25
    if zero == "target":
        target[:] = 0
    elif zero == "foreground":
        targets.target[:] = 0
    else:
        targets.weights[:] = 0
    before = tuple(x.tobytes() for x in (target, targets.target, targets.weights))
    expected, calls = old_result(), []

    def frozen(*args):
        assert all(
            a is b for a, b in zip(args, (tracker, proposals, states, rays, targets), strict=True)
        )
        calls.append(True)
        return expected

    monkeypatch.setattr(m, "frozen_fit", frozen)
    assert m.fit_with_ownership(tracker, proposals, states, rays, targets, target) is expected
    assert calls == [True]
    assert tracker.last_owner_fit["reason"] == "zero_effective_target"
    assert not tracker.last_owner_fit["active"]
    assert tuple(x.tobytes() for x in (target, targets.target, targets.weights)) == before


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_frozen_dispatch_uses_exact_original_cpu_or_cuda_method_without_device_work(
    monkeypatch, device
):
    tracker = SimpleNamespace(device=SimpleNamespace(type=device))
    args, result = (object(), object(), object(), object()), old_result()
    calls = []

    def called(self, *received):
        assert self is tracker and all(a is b for a, b in zip(args, received, strict=True))
        calls.append(device)
        return result

    cpu = called if device == "cpu" else forbidden
    cuda = called if device == "cuda" else forbidden
    monkeypatch.setattr(m.ThreeBodyTracker, "_fit", cpu)
    monkeypatch.setattr(m.AcceleratedThreeBodyTracker, "_fit", cuda)
    assert m.frozen_fit(tracker, *args) is result
    assert calls == [device]

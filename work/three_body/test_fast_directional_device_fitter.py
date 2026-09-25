"""CPU tensor semantics only; no objective fit, CUDA execution or timing claims."""

# ruff: noqa: E402

import sys
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"),
    str(Path(__file__).resolve().parents[2] / "src"),
]

import fast_directional_device_fitter as m
import numpy as np
import pytest
import torch


def poses(count, offset=0):
    return torch.arange(count * 27, dtype=torch.float64).reshape(count, 3, 3, 3) + offset


def assert_bytes(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.detach().numpy().tobytes() == expected.detach().numpy().tobytes()


def test_strict_best_selects_whole_candidate_and_retains_exact_tie():
    old, current = poses(4), poses(4, 1000)
    best_score = torch.tensor([4.0, 2.0, 1.0, 0.0], dtype=torch.float64)
    score = torch.tensor([3.0, 2.0, 2.0, -1.0], dtype=torch.float64)
    chosen, minimum = m.select_strict_best(old, best_score, current, score)

    assert_bytes(chosen, torch.stack((current[0], old[1], old[2], current[3])))
    assert_bytes(minimum, torch.tensor([3.0, 2.0, 1.0, -1.0], dtype=torch.float64))


@pytest.mark.parametrize("count", [1, 6, 24])
def test_fifteen_evaluations_match_frozen_boolean_selection_bit_for_bit(count):
    generator = torch.Generator().manual_seed(9197)
    legacy_pose = poses(count)
    legacy_score = torch.full((count,), torch.inf, dtype=torch.float64)
    best_pose, best_score = legacy_pose.clone(), legacy_score.clone()
    for step in range(15):
        current = torch.randn((count, 3, 3, 3), generator=generator, dtype=torch.float64)
        # Discrete scores make exact ties common; no fitter or optimizer is run.
        score = torch.randint(-2, 3, (count,), generator=generator).to(torch.float64)
        if step == 14:
            score = legacy_score.clone()
        improved = score < legacy_score
        legacy_pose[improved] = current[improved]
        legacy_score = torch.minimum(legacy_score, score)
        best_pose, best_score = m.select_strict_best(best_pose, best_score, current, score)
        assert_bytes(best_pose, legacy_pose)
        assert_bytes(best_score, legacy_score)


def test_next_representable_score_improves_and_signed_zero_tie_keeps_prior_pose():
    old, current = poses(3), poses(3, 200)
    previous = torch.tensor([1.0, 0.0, -0.0], dtype=torch.float64)
    score = torch.tensor([np.nextafter(1.0, 0.0), -0.0, 0.0], dtype=torch.float64)
    selected, minimum = m.select_strict_best(old, previous, current, score)

    assert_bytes(selected, torch.stack((current[0], old[1], old[2])))
    # Keep torch.minimum's exact signed-zero behavior from the frozen path.
    assert_bytes(minimum, torch.minimum(previous, score))


def test_selection_owns_output_and_does_not_mutate_or_alias_inputs():
    old, current = poses(2), poses(2, 500)
    previous = torch.tensor([1.0, 1.0], dtype=torch.float64)
    score = torch.tensor([0.0, 2.0], dtype=torch.float64)
    values = (old, previous, current, score)
    before = tuple(value.clone() for value in values)
    selected, minimum = m.select_strict_best(*values)
    selected.fill_(-99)
    minimum.fill_(-99)
    for value, original in zip(values, before, strict=True):
        assert_bytes(value, original)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_any_nonfinite_iteration_stays_invalid_after_later_finite_values(invalid):
    initial = torch.ones((), dtype=torch.bool)
    flag = m.accumulate_finite(initial, torch.ones((2, 3)), torch.empty(0))
    flag = m.accumulate_finite(flag, torch.tensor([1.0, invalid], dtype=torch.float64))
    flag = m.accumulate_finite(flag, torch.zeros((3, 4)))

    assert bool(initial)
    assert not bool(flag)
    with pytest.raises(FloatingPointError, match="score"):
        m.require_finite(flag, initial, initial)


def test_terminal_score_failure_is_checked_even_without_a_terminal_gradient():
    score_ok = gradient_ok = torch.ones((), dtype=torch.bool)
    for step in range(15):
        score = torch.tensor([float("inf") if step == 14 else 1.0])
        score_ok = m.accumulate_finite(score_ok, score)
        if step < 14:
            gradient_ok = m.accumulate_finite(gradient_ok, torch.zeros((2, 3, 3, 3)))
    assert bool(gradient_ok)
    with pytest.raises(FloatingPointError, match="score"):
        m.require_finite(score_ok, gradient_ok, torch.ones((), dtype=torch.bool))


def test_selection_and_finite_accumulation_do_not_read_host_scalars(monkeypatch):
    old, current = poses(2), poses(2, 10)
    previous, score = torch.tensor([2.0, 1.0]), torch.tensor([1.0, 1.0])
    initial = torch.ones((), dtype=torch.bool)

    def forbidden(*args, **kwargs):
        pytest.fail("selection/finite accumulation requested a host read")

    with monkeypatch.context() as guarded:
        for method in ("__bool__", "item", "cpu", "numpy", "tolist"):
            guarded.setattr(torch.Tensor, method, forbidden)
        flag = m.accumulate_finite(initial, current, score)
        selected, minimum = m.select_strict_best(old, previous, current, score)
    assert bool(flag)
    assert_bytes(selected, torch.stack((current[0], old[1])))
    assert_bytes(minimum, torch.tensor([1.0, 1.0]))


@pytest.mark.parametrize(
    "flags,label",
    [
        ((True, True, True), None),
        ((False, True, True), "score"),
        ((True, False, True), "gradient"),
        ((True, True, False), "output"),
        ((False, False, False), "score"),
    ],
)
def test_final_gate_transfers_all_flags_once_and_rejects_every_failure(monkeypatch, flags, label):
    cpu = torch.Tensor.cpu
    transferred = []

    def counted_cpu(value, *args, **kwargs):
        transferred.append((value.shape, value.dtype))
        return cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    values = tuple(torch.tensor(value, dtype=torch.bool) for value in flags)
    if label is None:
        assert m.require_finite(*values) is None
    else:
        with pytest.raises(FloatingPointError, match=label):
            m.require_finite(*values)
    assert transferred == [(torch.Size([3]), torch.bool)]


@pytest.mark.parametrize("directional", [False, True])
def test_initial_diagnostics_are_owned_before_graph_buffer_reuse(directional):
    score = torch.tensor([3.0, 7.0], dtype=torch.float64, requires_grad=True)
    parts = torch.arange(10, dtype=torch.float64).reshape(2, 5).requires_grad_(True)
    center = torch.tensor([0.25, 1.75], dtype=torch.float64)
    expected_score = score.detach().clone()
    expected_parts = parts.detach().clone()
    if directional:
        expected_parts[:, 2] = center
    initial_score, initial_parts = m.snapshot_initial_diagnostics(score, parts, center, directional)
    with torch.no_grad():
        score.fill_(999)
        parts.fill_(999)
        center.fill_(999)

    assert not initial_score.requires_grad and not initial_parts.requires_grad
    assert initial_score.data_ptr() != score.data_ptr()
    assert initial_parts.data_ptr() != parts.data_ptr()
    assert_bytes(initial_score, expected_score)
    assert_bytes(initial_parts, expected_parts)


def test_cpu_path_delegates_to_frozen_fitter_without_any_cuda_setup(monkeypatch):
    tracker = object.__new__(m.FastDirectionalDeviceFitter)
    tracker.device = torch.device("cpu")
    arguments = tuple(object() for _ in range(4))
    sentinel = object()
    calls = []

    def inherited(self, *args):
        calls.append((self, args))
        return sentinel

    def forbidden(*args, **kwargs):
        pytest.fail("CPU delegation entered the new CUDA path")

    monkeypatch.setattr(m.DirectionalDeviceFitter, "_fit", inherited)
    monkeypatch.setattr(m, "parent_policy", forbidden)
    monkeypatch.setattr(m, "GraphFit", forbidden)
    assert tracker._fit(*arguments) is sentinel
    assert len(calls) == 1 and calls[0][0] is tracker
    assert all(a is b for a, b in zip(calls[0][1], arguments, strict=True))

"""Independent CPU oracle for the fixed-step FP64 foreach Adam adapter.

No renderer, model, saved packet or CUDA is executed. Every update is checked
against the installed public torch.optim.Adam, including both moment tensors.
"""

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "labs/20260909_three_body"), str(REPO / "src")]
import full_fit_graph_fitter as m  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU Adam oracle must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    # Torch2.13 otherwise probes CUDA capture state even for a CPU optimizer.
    # This changes only the health check, not the Adam update implementation.
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda **kwargs: None)
    yield
    torch.set_num_threads(previous_threads)


def tensor_bytes(value):
    return value.dtype, tuple(value.shape), value.detach().numpy().tobytes(order="C")


def initial_pose(offset=0.0):
    value = torch.linspace(-5.0, 20.0, 54, dtype=torch.float64).reshape(2, 3, 3, 3)
    return value + offset


def gradients(*, strided=False, reverse=False):
    index = torch.arange(54, dtype=torch.float64).reshape(2, 3, 3, 3)
    scales = torch.tensor([1e-10, 1.0, 100.0], dtype=torch.float64)
    rows = []
    for step in range(14):
        value = torch.sin((index + 1) * (step + 1)) * scales
        value[0, 0, 0] = torch.tensor([0.0, -0.0, 0.0], dtype=torch.float64)
        # Real optimizer steps must retain old moments during zero-gradient steps.
        if step in (3, 10):
            value.zero_()
        if strided:
            storage = torch.empty((*value.shape, 2), dtype=torch.float64)
            storage[..., 0] = value
            storage[..., 1] = 123456.0
            value = storage[..., 0]
            assert not value.is_contiguous()
        rows.append(value)
    return rows[::-1] if reverse else rows


def compare_fourteen_updates(start, sequence, learning_rate, schedule, buffers=None):
    actual = start.clone() if buffers is None else buffers[0]
    mean = torch.zeros_like(actual) if buffers is None else buffers[1]
    square_mean = torch.zeros_like(actual) if buffers is None else buffers[2]
    expected = start.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([expected], lr=learning_rate, foreach=True)
    addresses = tuple(v.data_ptr() for v in (actual, mean, square_mean))
    snapshots = []
    for index, (gradient, constants) in enumerate(zip(sequence, schedule, strict=True), start=1):
        before_gradient = tensor_bytes(gradient)
        # Feed the same strides to both implementations. clone(preserve_format)
        # densifies a sliced tensor and can select different foreach kernels.
        expected.grad = gradient
        optimizer.step()
        with torch.no_grad():
            returned = m.foreach_adam_update_(actual, gradient, mean, square_mean, constants)
        assert returned is None
        state = optimizer.state[expected]
        assert state["step"].item() == index
        for value, reference in zip(
            (actual, mean, square_mean),
            (expected, state["exp_avg"], state["exp_avg_sq"]),
            strict=True,
        ):
            assert tensor_bytes(value) == tensor_bytes(reference), f"step {index}"
            assert torch.isfinite(value).all()
        assert tensor_bytes(gradient) == before_gradient
        assert tuple(v.data_ptr() for v in (actual, mean, square_mean)) == addresses
        snapshots.append(tuple(tensor_bytes(v) for v in (actual, mean, square_mean)))
    return (actual, mean, square_mean), snapshots


@pytest.mark.parametrize("learning_rate,strided", [(0.25, False), (0.03125, True)])
def test_all_fourteen_pose_and_moment_updates_match_public_foreach_adam(learning_rate, strided):
    schedule = m.frozen_adam_schedule(learning_rate)
    assert len(schedule) == 14
    compare_fourteen_updates(initial_pose(), gradients(strided=strided), learning_rate, schedule)


def test_second_fit_reuses_schedule_and_buffers_after_explicit_full_state_reset():
    schedule = m.frozen_adam_schedule(0.25)
    buffers, first = compare_fourteen_updates(initial_pose(), gradients(), 0.25, schedule)
    next_pose = initial_pose(offset=3.5)
    with torch.no_grad():
        buffers[0].copy_(next_pose)
        buffers[1].zero_()
        buffers[2].zero_()
    _, reused = compare_fourteen_updates(
        next_pose, gradients(reverse=True), 0.25, schedule, buffers=buffers
    )
    _, fresh = compare_fourteen_updates(next_pose, gradients(reverse=True), 0.25, schedule)
    assert reused == fresh
    assert first != reused


def test_schedule_is_immutable_host_scalars_with_fixed_order():
    schedule = m.frozen_adam_schedule(0.25)
    assert isinstance(schedule, tuple)
    assert tuple(s.index for s in schedule) == tuple(range(1, 15))
    for step in schedule:
        assert type(step.step_size) is float and step.step_size < 0
        assert type(step.bias_correction2_sqrt) is float
        assert 0 < step.bias_correction2_sqrt < 1
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        schedule[0].index = 99


@pytest.mark.parametrize("learning_rate", [True, 0.0, -0.25, float("nan"), float("inf"), 1, "0.25"])
def test_invalid_learning_rate_is_rejected_before_any_update(learning_rate):
    with pytest.raises((ValueError, TypeError)):
        m.frozen_adam_schedule(learning_rate)


@pytest.mark.parametrize("steps", [True, 0, -1, 13, 15, 14.0, "14"])
def test_invalid_step_count_is_rejected_before_any_update(steps):
    with pytest.raises((ValueError, TypeError)):
        m.frozen_adam_schedule(0.25, steps=steps)

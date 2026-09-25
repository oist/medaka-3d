"""Protect exact initial poses and old FP64 inputs in the new proposal search."""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import directional_device_fitter as old  # noqa: E402
import mixed_precision_fitter as m  # noqa: E402

from test_fixed_body_width_fitter import fit_fixture, snapshot  # noqa: E402
from test_fixed_body_width_renderer import exact  # noqa: E402
from test_fp64_pose_bank_rescore import fixture as bank_fixture  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU verification must never initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("device_alias", ["cpu", "cpu:0"])
def test_original_input_preparation_matches_frozen_fitter_before_first_objective(
    monkeypatch, device_alias
):
    args = fit_fixture()
    fitter, proposals, states, rays, targets = args[:5]
    fitter.directional = True
    fitter.device = torch.device(device_alias)
    fitter.lower, fitter.upper = np.full(3, -100.0), np.full(3, 100.0)
    policy = dict(
        axes=np.arange(18, dtype=np.float64).reshape(2, 3, 3) / 23,
        enabled=np.array([[True, False, True], [False, True, False]]),
        parallel_sigma=np.array([[1.1, 1.3, 1.7], [2.3, 2.9, 3.1]]),
        arbitrary_input=np.array([0, 1], dtype=np.int32),
    )
    monkeypatch.setattr(m, "parent_policy", lambda *args: policy)
    monkeypatch.setattr(old, "parent_policy", lambda *args: policy)
    before = snapshot(args)
    capture = {}

    class FirstObjective(Exception):
        pass

    def intercept(pose, cfg, inputs, denominator, renderer, **kwargs):
        capture.update(pose=pose, inputs=inputs, renderer=renderer, denominator=denominator)
        raise FirstObjective

    monkeypatch.setattr(old, "objective", intercept)
    with pytest.raises(FirstObjective):
        old.DirectionalDeviceFitter._fit(fitter, proposals, states, rays, targets)
    problem = m.prepare_original_problem(fitter, proposals, states, rays, targets)
    assert problem.renderer.device == problem.pose.device == torch.device("cpu")
    assert fitter.device == torch.device(device_alias) and fitter.config.device == "cpu"
    exact(problem.pose, capture["pose"])
    for f in dataclasses.fields(problem.inputs):
        exact(getattr(problem.inputs, f.name), getattr(capture["inputs"], f.name))
    for name in ("ray_origins", "ray_directions", "ray_valid", "_fractions", "_taper"):
        exact(getattr(problem.renderer, name), getattr(capture["renderer"], name))
    assert not torch.equal(
        problem.renderer.ray_origins, problem.renderer.ray_origins.float().double()
    )
    assert problem.parents.tolist() == [1, 0]
    np.testing.assert_array_equal(problem.axes.numpy(), policy["axes"][[1, 0]])
    rescored = m.rescore_pose_bank(
        problem.pose[None].repeat(16, 1, 1, 1, 1),
        config=fitter.config,
        inputs=problem.inputs,
        renderer=problem.renderer,
        axes=problem.axes,
        parallel_sigma=problem.parallel_sigma,
        enabled=problem.enabled,
    )
    assert rescored["selected_bank_index"].tolist() == [0, 0]
    assert snapshot(args) == before


def test_rounding_inputs_is_one_way_owned_and_leaves_fp64_original_untouched():
    _, kwargs = bank_fixture()
    original = kwargs["inputs"]
    before = {f.name: getattr(original, f.name).clone() for f in dataclasses.fields(original)}
    rounded = m.inputs_as_fp32(original)
    for f in dataclasses.fields(original):
        a, b = getattr(original, f.name), getattr(rounded, f.name)
        assert b.dtype == torch.float32 and b.data_ptr() != a.data_ptr()
        exact(b, a.float())
        b.zero_()
        exact(a, before[f.name])


def test_bank_keeps_original_precision_and_all_fifteen_poses_including_rounded_start():
    original_bank, _ = bank_fixture()
    original = original_bank[0].clone().requires_grad_(True)
    path = torch.stack([original.float() + i / 17 for i in range(15)])
    bank = m.assemble_pose_bank(original, path)
    assert not bank.requires_grad
    exact(bank[0], original)
    exact(bank[1:], path.double())
    assert not torch.equal(bank[0], bank[1])
    bank.zero_()
    assert torch.count_nonzero(original) > 0 and torch.count_nonzero(path) > 0


@pytest.mark.parametrize("invalid", ["rows", "dtype", "start", "nan_original", "nan_late"])
def test_bank_rejects_incomplete_wrong_start_or_nonfinite_path(invalid):
    original_bank, _ = bank_fixture()
    original = original_bank[0].clone()
    path = original.float()[None].repeat(15, 1, 1, 1, 1)
    if invalid == "rows":
        path = path[:14]
    elif invalid == "dtype":
        path = path.double()
    elif invalid == "start":
        path[0, 0, 0, 0, 0] += 1
    elif invalid == "nan_original":
        original[0, 0, 0, 0] = float("nan")
    else:
        path[-1, 0, 0, 0, 0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        m.assemble_pose_bank(original, path)


def test_subset_two_detects_fp32_ranking_error_and_preserves_initial_ties():
    scores = torch.full((16, 3), 20.0, dtype=torch.float64)
    scores[0] = torch.tensor([10.0, 10.0, 10.0])
    # FP32 chose iterations 3,4,5. A different FP64-scored path row wins B0.
    scores[4, 0], scores[12, 0] = 8.0, 7.0
    scores[5, 1], scores[6, 2] = 10.0, 9.0
    before = scores.clone()
    result = m.two_candidate_diagnostics(scores, torch.tensor([3, 4, 5]))
    assert result["subset2_bank_index"].tolist() == [4, 0, 6]
    assert result["subset2_missed_strict_improvement"].tolist() == [True, False, False]
    assert result["subset2_cost_regret"].tolist() == [1.0, 0.0, 0.0]
    exact(scores, before)


@pytest.mark.parametrize("index", [-1, 15])
def test_subset_rejects_uninitialized_or_out_of_range_iteration(index):
    with pytest.raises(ValueError):
        m.two_candidate_diagnostics(torch.ones(16, 1, dtype=torch.float64), torch.tensor([index]))


def test_cpu_fitter_does_not_silently_run_frozen_algorithm():
    fitter = object.__new__(m.MixedPrecisionDirectionalDeviceFitter)
    fitter.device = torch.device("cpu")
    with pytest.raises(ValueError, match="requires CUDA"):
        fitter._fit([], [], None, None)

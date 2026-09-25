"""Tiny CPU full-objective recurrence; no saved packets or CUDA construction."""

# ruff: noqa: E402
import dataclasses
import sys
from pathlib import Path

import pytest
import torch

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"),
    str(Path(__file__).resolve().parents[2] / "src"),
]
import full_fit_graph_fitter as m
from qualify_fixed_body_width_cuda import synthetic_case
from renderer import RefractiveFishRenderer


def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.detach().numpy().tobytes() == b.detach().numpy().tobytes()


def fixture(directional=True):
    cfg, pose, inputs, fixed = synthetic_case(0, "cpu")
    cfg.opt_steps, cfg.learning_rate_mm = 14, 0.25
    pixels = {v: fixed.pixels[i, :4].copy() for i, v in enumerate(fixed.views)}
    renderer = RefractiveFishRenderer(fixed.rig, pixels, fixed.config, dtype=torch.float64)
    inputs = dataclasses.replace(inputs, target=inputs.target[:, :4], weight=inputs.weight[:, :4])
    axes = torch.zeros((2, 3, 3), dtype=torch.float64)
    axes[..., 0] = 1
    parallel = inputs.motion_sigma * 1.5
    enabled = torch.tensor([[True, False, True], [False, True, True]])
    lower, upper = torch.tensor([0.0, 0.0, -6.0]), torch.tensor([192.0, 192.0, 186.0])
    # An active clamp boundary checks update -> clamp ordering on a real objective.
    upper[0] = 36.0
    graph = object.__new__(m.FullFitGraph)
    graph._initialize(
        cfg,
        pose,
        inputs,
        renderer,
        axes,
        parallel,
        enabled,
        lower.double(),
        upper.double(),
        directional,
    )
    return graph


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("directional", [False, True])
def test_all15_scores_gradients_and14_adam_clamps_match_original_foreach(directional):
    graph = fixture(directional)
    graph._operation()
    pose = graph._start.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pose], lr=graph.config.learning_rate_mm, foreach=True)
    scratch = fixture(directional)
    best = pose.detach().clone()
    score_min = torch.full((2,), float("inf"), dtype=torch.float64)
    for step in range(15):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            scratch._pose.copy_(pose)
        m.GraphFit._operation(scratch)
        score, parts, gradient = [
            v.detach().clone() for v in (scratch._scores, scratch._parts, scratch._pose.grad)
        ]
        with torch.no_grad():
            difference, correction, center = m.center_correction(
                pose[:, :, 1] - scratch._inputs.predicted[:, :, 1],
                scratch._inputs.motion_sigma,
                scratch._axes,
                scratch._parallel,
                scratch._enabled,
            )
            if directional:
                score = score + scratch.config.motion_weight * difference
            for name, value in (
                ("pose_before", pose),
                ("score", score),
                ("parts", parts),
                ("base_gradient", gradient),
            ):
                exact(graph.trace[name][step], value)
            changed = score < score_min
            best[changed] = pose.detach()[changed]
            score_min = torch.minimum(score_min, score)
            if step == 0:
                expected_parts = parts.clone()
                if directional:
                    expected_parts[:, 2] = center
                exact(graph.initial_score, score)
                exact(graph.initial_parts, expected_parts)
            if step == 14:
                break
            pose.grad = gradient.clone()
            if directional:
                pose.grad[:, :, 1] += scratch.config.motion_weight * correction
            exact(graph.trace["update_gradient"][step], pose.grad)
        optimizer.step()
        with torch.no_grad():
            pose.clamp_(scratch._lower, scratch._upper)
        state = optimizer.state[pose]
        exact(graph.trace["moment"][step], state["exp_avg"])
        exact(graph.trace["variance"][step], state["exp_avg_sq"])
        exact(graph.trace["pose_after"][step], pose)
    exact(graph.best_pose, best)
    exact(graph.best_score, score_min)
    assert bool(graph.pose_ok & graph.score_ok & graph.gradient_ok)


def test_complete_sequence_resets_every_moment_best_and_pose_without_mutating_input():
    graph = fixture()
    original = graph._start.clone()
    graph._operation()
    first = {k: v.clone() for k, v in graph.trace.items()}
    graph._moment.fill_(900)
    graph._variance.fill_(300)
    graph._working.fill_(-800)
    graph.best_pose.fill_(-100)
    graph.best_score.fill_(-torch.inf)
    graph._operation()
    for key in first:
        exact(first[key], graph.trace[key])
    exact(original, graph._start)
    assert graph.trace["score"].shape == (15, 2)
    assert graph.trace["update_gradient"].shape == (14, 2, 3, 3, 3)


def test_reconfigure_new_inputs_and_directions_are_owned_and_equivalent_to_fresh():
    graph, fresh = fixture(), fixture()
    inputs = dataclasses.replace(graph._inputs, predicted=graph._inputs.predicted + 0.05)
    axes = graph._axes.roll(1, -1)
    for value in (graph, fresh):
        value.configure_frame(
            inputs,
            value._renderer,
            axes,
            value._parallel,
            value._enabled,
            value._lower,
            value._upper,
        )
        value._operation()
    for name in graph.trace:
        exact(graph.trace[name], fresh.trace[name])
    inputs.predicted.fill_(123)
    axes.fill_(234)
    assert not bool((graph._inputs.predicted == 123).all())
    assert not bool((graph._axes == 234).all())


def test_extra_validation_rejects_before_mutating_any_static_input():
    graph = fixture()
    before = graph._inputs.predicted.clone()
    inputs = dataclasses.replace(graph._inputs, predicted=graph._inputs.predicted + 2)
    with pytest.raises(ValueError, match="finite"):
        graph.configure_frame(
            inputs,
            graph._renderer,
            graph._axes * float("nan"),
            graph._parallel,
            graph._enabled,
            graph._lower,
            graph._upper,
        )
    exact(before, graph._inputs.predicted)


def test_public_constructor_rejects_cpu_before_cuda_or_capture(monkeypatch):
    graph = fixture()
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: pytest.fail("CUDA called"))
    with pytest.raises(ValueError, match="CUDA float64"):
        m.FullFitGraph(
            graph.config,
            graph._start,
            graph._inputs,
            graph._renderer,
            graph._axes,
            graph._parallel,
            graph._enabled,
            graph._lower,
            graph._upper,
        )

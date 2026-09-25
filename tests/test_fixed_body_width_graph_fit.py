"""Fixed-width capture allocation/configure/operation on CPU, without CUDA APIs."""

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import fixed_body_width_graph_fit as m
import graph_fit as old
from fast_union import render_union
from fixed_body_width_renderer import FixedWidthRenderer

from test_fixed_body_width_renderer import exact, exact_fields, renderers, scene, seed


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture():
    _, renderer = renderers()
    value = np.stack((seed(), (seed() - seed()[:, 1:2]) * 1.4 + seed()[:, 1:2]))
    pose = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    b, n = len(pose), renderer.n_pixels
    cfg = SimpleNamespace(
        width_ratio=renderer.config.width_ratio,
        ray_sigma_mm=0.8,
        missing_cost=4,
        shape_weight=1.3,
        motion_weight=0.7,
        repulsion_weight=0.4,
        mask_weight=2.0,
    )
    directions = torch.zeros((b, 3, 3, 3, 3), dtype=pose.dtype)
    directions[..., 2] = 1
    lengths = torch.linalg.vector_norm(torch.diff(pose[:1].detach(), dim=2), dim=-1)
    inputs = m.FitInputs(
        origins=torch.zeros_like(directions),
        directions=directions,
        held=torch.ones((b, 3, 3, 3), dtype=pose.dtype),
        predicted=pose.detach().clone() + 0.1,
        motion_sigma=torch.full((b, 3), 2.0, dtype=pose.dtype),
        lengths=lengths,
        length_sigma=(0.2 * lengths).clamp_min(1),
        node_weight=torch.tensor([1.0, 1.0, 0.35], dtype=pose.dtype)[None, None, None],
        target=torch.ones((3, n), dtype=pose.dtype),
        weight=torch.full((3, n), 0.2, dtype=pose.dtype),
    )
    return cfg, pose, inputs, renderer


def cpu_static_fixture():
    """Only configure/static-buffer tests; this is deliberately not a CUDA graph."""
    cfg, pose, inputs, renderer = fixture()
    graph = object.__new__(m.FixedWidthGraphFit)
    graph.config, graph.device, graph.dtype = cfg, pose.device, pose.dtype
    graph.batch_size, graph.n_pixels = len(pose), renderer.n_pixels
    graph._renderer_signature = (
        renderer.views,
        renderer.config,
        renderer.device,
        renderer.dtype,
        renderer.n_pixels,
    )
    graph._fixed_width_signature = renderer.fixed_width_signature
    graph._renderer = m.make_capture_renderer(renderer)
    graph._inputs = dataclasses.replace(
        inputs, **{f.name: getattr(inputs, f.name).clone() for f in dataclasses.fields(inputs)}
    )
    graph._pose = pose.detach().clone().requires_grad_(True)
    graph._pose.grad = torch.zeros_like(pose)
    graph.denominator = 9 * float(inputs.node_weight.sum())
    graph._width = torch.full((len(pose), 3), cfg.width_ratio, dtype=pose.dtype)
    graph.uses_positive_product_bound = (
        m._positive_transmission_bound(renderer.config, pose.dtype)
        > 4 * torch.finfo(pose.dtype).eps
    )
    return graph, pose, inputs, renderer


def test_capture_fast_union_and_original_operation_use_same_fixed_radial_cpu():
    graph, pose, inputs, renderer = cpu_static_fixture()
    actual = graph._renderer.gaussians(pose, graph._width)
    exact_fields(actual, renderer.gaussians(pose))
    exact(actual.radial_sigma_mm[0], actual.radial_sigma_mm[1])
    normal = render_union(renderer, pose)
    captured = render_union(graph._renderer, pose, width_ratio=graph._width)
    exact(normal, captured)
    exact(normal, renderer.render(pose).union_alpha)
    expected, parts, _ = old.objective(pose, graph.config, inputs, graph.denominator, renderer)
    gradient = torch.autograd.grad(expected.sum(), pose)[0]
    graph._operation()  # Frozen tensor operation on CPU; no graph construction or replay.
    exact(graph._scores, expected)
    exact(graph._parts, parts)
    exact(graph._pose.grad, gradient)
    assert m.FixedWidthGraphFit._operation is old.GraphFit._operation
    assert m.FixedWidthGraphFit.configure is old.GraphFit.configure
    assert m.FixedWidthGraphFit.evaluate is old.GraphFit.evaluate
    assert m.FixedWidthGraphFit._validate_pose is old.GraphFit._validate_pose


def test_capture_owns_every_tensor_and_array_and_public_gaussian_width_does_not_alias():
    _, renderer = renderers()
    capture = m.make_capture_renderer(renderer)
    for name, value in renderer.__dict__.items():
        copied = getattr(capture, name)
        if isinstance(value, torch.Tensor):
            exact(copied, value)
            assert copied.data_ptr() != value.data_ptr()
        elif isinstance(value, np.ndarray):
            assert copied.tobytes() == value.tobytes()
            assert not np.shares_memory(copied, value)
    initial = capture.radial_sigma_mm
    bodies = capture.gaussians(torch.tensor(seed()[None]), torch.full((1, 3), 0.06))
    bodies.radial_sigma_mm.fill_(100)
    exact(capture.radial_sigma_mm, initial)
    renderer.ray_origins.add_(1)
    assert not torch.equal(capture.ray_origins, renderer.ray_origins)
    assert capture.validate_fixed_width() == renderer.fixed_width_signature


def test_matching_reference_allows_new_rays_and_inputs_but_never_copies_radial_width():
    graph, _, inputs, renderer = cpu_static_fixture()
    rig, pixels = scene()
    changed = FixedWidthRenderer(
        rig, {v: xy + 1 for v, xy in pixels.items()}, reference_pose_xyz=seed(), dtype=torch.float64
    )
    radial_before = graph._renderer.radial_sigma_mm
    reference_before = graph._fixed_width_signature
    new_inputs = dataclasses.replace(
        inputs, predicted=inputs.predicted + 2, target=inputs.target * 0
    )
    graph.configure(new_inputs, changed)
    exact(graph._renderer.ray_origins, changed.ray_origins)
    exact(graph._inputs.predicted, new_inputs.predicted)
    exact(graph._renderer.radial_sigma_mm, radial_before)
    assert graph._fixed_width_signature == reference_before == renderer.fixed_width_signature


@pytest.mark.parametrize(
    "fault",
    [
        "translated_reference",
        "replacement_buffer",
        "capture_buffer",
        "dynamic_renderer",
        "config",
        "node_weight",
    ],
)
def test_configure_mismatch_rejected_before_static_input_mutation(fault):
    graph, _, inputs, renderer = cpu_static_fixture()
    original_inputs = {
        f.name: getattr(graph._inputs, f.name).clone() for f in dataclasses.fields(inputs)
    }
    if fault == "translated_reference":
        rig, pixels = scene()
        renderer = FixedWidthRenderer(
            rig, pixels, reference_pose_xyz=seed() + 2, dtype=torch.float64
        )
    elif fault == "replacement_buffer":
        renderer._fixed_radial_sigma_mm.mul_(1.1)
    elif fault == "capture_buffer":
        graph._renderer._fixed_radial_sigma_mm.mul_(1.1)
    elif fault == "dynamic_renderer":
        renderer = renderers()[0]
    elif fault == "config":
        renderer.config = dataclasses.replace(renderer.config, width_ratio=0.07)
    else:
        inputs = dataclasses.replace(inputs, node_weight=inputs.node_weight * 2)
    inputs = dataclasses.replace(inputs, predicted=inputs.predicted + 999)
    with pytest.raises(ValueError):
        graph.configure(inputs, renderer)
    for name, before in original_inputs.items():
        exact(getattr(graph._inputs, name), before)


def test_cpu_constructor_rejected_without_cuda_use_and_plain_renderer_cannot_be_captured():
    cfg, pose, inputs, renderer = fixture()
    with pytest.raises(ValueError, match="CUDA"):
        m.FixedWidthGraphFit(cfg, pose, inputs, renderer)
    with pytest.raises(ValueError, match="registered FixedWidthRenderer"):
        m.make_capture_renderer(renderers()[0])
    assert not torch.cuda.is_initialized()

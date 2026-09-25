"""Graph replay must honor changed samples and preserve public validation/copies."""

import gc
import importlib.util
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from medaka3d.types import VIEWS

_LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"three_body_graph_union_test_{name}",
                                                 _LAB / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_RENDERER = load_module("renderer")
_FAST = load_module("fast_union")
_OLD = {name: sys.modules.get(name) for name in ("renderer", "fast_union")}
sys.modules.update(renderer=_RENDERER, fast_union=_FAST)
try:
    _GRAPH = load_module("graph_union")
finally:
    for _name, _value in _OLD.items():
        if _value is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _value


@pytest.fixture
def case(rig):
    rig = replace(rig, cameras=tuple(
        replace(camera, index_air=1.0003, index_glass=1.52, index_water=1.3325,
                layer_count=1 if camera.view == VIEWS[0] else 2,
                distortion=(-0.15, 0.07, 0.003, -0.002, -0.01))
        for camera in rig.cameras))
    centers = np.array([[70., 65., 80.], [73., 65., 90.], [135., 120., 135.]])
    pose = centers[:, None] + np.array([[5., 0., 0.], [0., 0., 0.], [-12., 2., -1.]])[None]
    offsets = np.array([[-3., 0.], [0., -2.], [2., 1.]])
    pixels = {camera.view: (camera.project(pose.reshape(-1, 3))[:, None]
                            + offsets[None]).reshape(-1, 2) for camera in rig.cameras}
    return rig, pose, pixels


def reference(renderer, pose, width, target, weights):
    pose = pose.detach().clone().requires_grad_(True)
    alpha = _FAST.render_union(renderer, pose, width)
    loss = ((alpha - target).square() * weights * renderer.ray_valid).sum((1, 2)) / 3
    gradient, = torch.autograd.grad(loss.sum(), pose)
    return _GRAPH.GraphUnionResult(alpha.detach(), loss.detach(), gradient)


def assert_result_exact(actual, expected):
    for name in ("union_alpha", "loss", "pose_gradient"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)


def test_private_capture_geometry_preserves_existing_values_and_gradients_on_cpu(case):
    rig, pose, pixels = case
    renderer = _RENDERER.RefractiveFishRenderer(rig, pixels, dtype=torch.float64)
    capture = object.__new__(_GRAPH._CaptureRenderer)
    capture.__dict__ = renderer.__dict__.copy()
    poses = torch.tensor(np.stack((pose, pose + 0.2)), requires_grad=True)
    width = torch.tensor([[0.06, 0.07, 0.08], [0.05, 0.06, 0.09]], dtype=torch.float64,
                         requires_grad=True)
    actual, expected = capture.gaussians(poses, width), renderer.gaussians(poses, width)
    for field in fields(expected):
        torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name),
                                   rtol=0, atol=0)
    terms = ("centers", "tangents", "axial_sigma_mm", "radial_sigma_mm")
    old = torch.autograd.grad(sum(getattr(expected, name).square().sum() for name in terms),
                              (poses, width), retain_graph=True)
    new = torch.autograd.grad(sum(getattr(actual, name).square().sum() for name in terms),
                              (poses, width))
    for a, b in zip(new, old, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="CUDA renderer"):
        _GRAPH.GraphUnion(renderer, poses)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("count", [3, 8, 32])
@pytest.mark.parametrize("zero_count", [0, 1, 2])
def test_capture_product_preserves_global_zero_branch_floating_order(dtype, count, zero_count):
    generator = torch.Generator().manual_seed(9383)
    value = torch.rand((2, 3, count), generator=generator, dtype=dtype)
    value[0, 0, :zero_count] = 0
    value.requires_grad_()
    weights = torch.linspace(-0.7, 1.1, 6, dtype=dtype).reshape(2, 3)
    expected = value.prod(dim=-1)
    actual = _GRAPH._CaptureLastDimProd.apply(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    original_grad, = torch.autograd.grad((expected * weights).sum(), value)
    capture_grad, = torch.autograd.grad((actual * weights).sum(), value)
    torch.testing.assert_close(capture_grad, original_grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph parity requires CUDA")
@pytest.mark.parametrize("batch", [1, 6, 18])
def test_cuda_replay_values_gradients_changed_rays_targets_and_output_ownership(case, batch):
    rig, pose, pixels = case
    renderer = _RENDERER.RefractiveFishRenderer(rig, pixels, device="cuda", dtype=torch.float64)
    poses = torch.tensor(np.repeat(pose[None], batch, axis=0), device="cuda", dtype=torch.float64)
    width = torch.tensor([0.053, 0.067, 0.081], device="cuda", dtype=torch.float64)
    target = torch.linspace(0, 1, 81, device="cuda", dtype=torch.float64).reshape(3, 27)
    weights = torch.linspace(0.001, 0.1, 81, device="cuda", dtype=torch.float64).reshape(3, 27)
    graph = _GRAPH.GraphUnion(renderer, poses, width, warmup=1)
    assert graph.uses_positive_product_bound
    first = graph.evaluate(poses, target, weights)
    assert_result_exact(first, reference(renderer, poses, width, target, weights))
    assert first.pose_gradient.abs().max() > 1e-8
    kept = tuple(getattr(first, n).clone() for n in ("union_alpha", "loss", "pose_gradient"))
    changed_pixels = {view: xy + np.array([3., -2.]) for view, xy in pixels.items()}
    changed_pixels[VIEWS[1]][3:] = np.nan
    changed_pixels[VIEWS[2]][:] = np.nan
    changed_renderer = _RENDERER.RefractiveFishRenderer(
        rig, changed_pixels, device="cuda", dtype=torch.float64)
    changed_pose = poses + torch.tensor([0.3, -0.2, 0.1], device="cuda")
    changed_width = width + 0.004
    changed_target, changed_weight = 1 - target, weights * 0.8
    second = graph.evaluate(changed_pose, changed_target, changed_weight, changed_width,
                            renderer=changed_renderer)
    assert_result_exact(second, reference(changed_renderer, changed_pose, changed_width,
                                          changed_target, changed_weight))
    assert not second.union_alpha[:, ~changed_renderer.ray_valid].any()
    assert (second.loss - first.loss).abs().max() > 1e-6
    for name, held in zip(("union_alpha", "loss", "pose_gradient"), kept, strict=True):
        torch.testing.assert_close(getattr(first, name), held, rtol=0, atol=0)
    borrowed = graph.evaluate(poses, target, weights, width, renderer=renderer, copy_outputs=False)
    previous_loss = borrowed.loss.clone()
    next_result = graph.evaluate(changed_pose, changed_target, changed_weight, changed_width,
                                 renderer=changed_renderer, copy_outputs=False)
    assert borrowed.loss.data_ptr() == next_result.loss.data_ptr()
    assert (borrowed.loss - previous_loss).abs().max() > 1e-6
    # No replacement renderer means the most recently copied frame rays remain.
    stable = graph.evaluate(changed_pose, changed_target, changed_weight, changed_width)
    assert_result_exact(stable, next_result)
    for value, message in ((torch.full_like(poses, torch.nan), "pose must be finite"),):
        with pytest.raises(ValueError, match=message):
            graph.evaluate(value, target, weights)
    with pytest.raises(ValueError, match="positive"):
        graph.evaluate(poses, target, weights, -0.1)
    with pytest.raises(ValueError, match="nonnegative"):
        graph.evaluate(poses, target, -weights)
    with pytest.raises(ValueError, match="captured"):
        graph.evaluate(poses, target[:, :-1], weights[:, :-1])
    # Failed public validation cannot poison the captured buffers.
    again = graph.evaluate(poses, target, weights, width, renderer=renderer)
    assert_result_exact(again, first)
    # Reset every ray to invalid, with positive target/weights. It remains unknown.
    invalid = _RENDERER.RefractiveFishRenderer(
        rig, {VIEWS[0]: np.full((27, 2), np.nan)}, device="cuda", dtype=torch.float64)
    empty = graph.evaluate(poses, target, weights, renderer=invalid)
    assert not empty.loss.any() and not empty.union_alpha.any() and not empty.pose_gradient.any()
    del graph
    gc.collect()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph parity requires CUDA")
def test_high_opacity_collapsed_body_uses_zero_safe_capture_and_can_replay_nonzero(case):
    rig, pose, _ = case
    collapsed = np.repeat(pose[:, 1:2], 3, axis=1)[None]
    pixels = {view: rig.camera(view).project(collapsed[0, :, 1]) for view in VIEWS}
    config = _RENDERER.RendererConfig(gaussians_per_fish=32, opacity=0.9999999)
    renderer = _RENDERER.RefractiveFishRenderer(rig, pixels, config, device="cuda",
                                              dtype=torch.float64)
    graph = _GRAPH.GraphUnion(renderer, collapsed, warmup=1)
    assert not graph.uses_positive_product_bound
    target = torch.zeros((3, 3), device="cuda", dtype=torch.float64)
    weights = torch.ones_like(target)
    poses = torch.tensor(collapsed, device="cuda", dtype=torch.float64)
    actual = graph.evaluate(poses, target, weights)
    assert (actual.union_alpha == 1).any()  # Exact zero transmission really occurs.
    expected = reference(renderer, poses, 0.06, target, weights)
    assert_result_exact(actual, expected)
    moved = poses + torch.tensor([0.1, 0.1, 0.1], device="cuda")
    assert_result_exact(graph.evaluate(moved, target, weights),
                        reference(renderer, moved, 0.06, target, weights))

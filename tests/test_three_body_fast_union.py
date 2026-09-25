"""Union optimization must preserve all pose/width derivatives of full rendering."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from medaka3d.types import VIEWS

_LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"three_body_fast_union_test_{name}",
                                                 _LAB / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_RENDERER = load_module("renderer")
render_union = load_module("fast_union").render_union


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def case(rig):
    rig = replace(rig, cameras=tuple(
        replace(camera, index_air=1.0003, index_glass=1.52, index_water=1.3325,
                layer_count=1 if camera.view == VIEWS[0] else 2,
                distortion=(-0.15, 0.07, 0.003, -0.002, -0.01))
        for camera in rig.cameras
    ))
    centers = np.array([[70., 65., 80.], [73., 65., 90.], [135., 120., 135.]])
    pose = centers[:, None] + np.array([[5., 0., 0.], [0., 0., 0.], [-12., 2., -1.]])[None]
    offsets = np.array([[-5., 0.], [-1., 2.], [0., -3.], [2., 1.], [4., -2.]])
    pixels = {camera.view: (camera.project(pose[:, 1])[:, None] + offsets[None]).reshape(-1, 2)
              for camera in rig.cameras}
    return rig, pose, pixels


def compare_values_and_gradients(renderer, pose_array, width_array):
    pose = torch.tensor(pose_array, dtype=renderer.dtype, requires_grad=True)
    width = torch.tensor(width_array, dtype=renderer.dtype, requires_grad=True)
    initial_pose, initial_width = pose.detach().clone(), width.detach().clone()
    full = renderer.render(pose, width).union_alpha
    fast = render_union(renderer, pose, width)
    torch.testing.assert_close(fast, full, rtol=0, atol=0)
    # Nonuniform signed coefficients avoid merely checking a symmetric sum.
    coefficients = torch.linspace(-0.6, 1.1, full.numel(), dtype=renderer.dtype).reshape(full.shape)
    full_loss = (full.square() * coefficients).sum()
    fast_loss = (fast.square() * coefficients).sum()
    expected = torch.autograd.grad(full_loss, (pose, width))
    actual = torch.autograd.grad(fast_loss, (pose, width))
    for candidate, reference in zip(actual, expected, strict=True):
        assert torch.isfinite(candidate).all()
        torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
    torch.testing.assert_close(pose, initial_pose, rtol=0, atol=0)
    torch.testing.assert_close(width, initial_width, rtol=0, atol=0)
    return fast, actual


@pytest.mark.parametrize("pixel_chunk_size", [3, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_union_matches_all_pose_and_per_fish_width_gradients(case, pixel_chunk_size, dtype):
    rig, pose, pixels = case
    renderer = _RENDERER.RefractiveFishRenderer(
        rig, pixels, _RENDERER.RendererConfig(pixel_chunk_size=pixel_chunk_size), dtype=dtype)
    _, gradients = compare_values_and_gradients(renderer, pose, [0.051, 0.068, 0.074])
    assert (torch.linalg.vector_norm(gradients[0], dim=-1) > 1e-8).all()
    assert (gradients[1].abs() > 1e-8).all()


def test_batch_and_scalar_width_gradient_match_without_changing_shapes(case):
    rig, pose, pixels = case
    renderer = _RENDERER.RefractiveFishRenderer(rig, pixels, dtype=torch.float64)
    output, gradients = compare_values_and_gradients(renderer,
                                                   np.stack([pose, pose + 0.2]), 0.063)
    assert output.shape == (2, 3, 15)
    assert gradients[0].shape == (2, 3, 3, 3)
    assert gradients[1].ndim == 0


def test_missing_views_and_nan_pixel_padding_have_identical_zero_support(case):
    rig, pose, pixels = case
    pixels[VIEWS[1]] = np.vstack([pixels[VIEWS[1]][:2], [np.nan, np.nan]])
    del pixels[VIEWS[2]]
    renderer = _RENDERER.RefractiveFishRenderer(
        rig, pixels, _RENDERER.RendererConfig(pixel_chunk_size=4), dtype=torch.float64)
    result, _ = compare_values_and_gradients(renderer, pose, [0.06, 0.07, 0.08])
    assert (result[0][~renderer.ray_valid] == 0).all()
    assert int(renderer.ray_valid[1].sum()) == 2
    assert not renderer.ray_valid[2].any()


def test_entirely_invalid_pixels_yield_zero_value_and_zero_parameter_gradients(case):
    rig, pose, _ = case
    renderer = _RENDERER.RefractiveFishRenderer(
        rig, {VIEWS[0]: np.full((5, 2), np.nan)}, dtype=torch.float64)
    result, gradients = compare_values_and_gradients(renderer, pose, 0.06)
    assert not result.any()
    assert all(not gradient.any() for gradient in gradients)


def test_optional_width_uses_the_identical_renderer_default(case):
    rig, pose, pixels = case
    renderer = _RENDERER.RefractiveFishRenderer(
        rig, pixels, _RENDERER.RendererConfig(width_ratio=0.072), dtype=torch.float64)
    expected = renderer.render(pose).union_alpha
    actual = render_union(renderer, pose)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fast_path_does_not_invoke_full_visibility_renderer(case, monkeypatch):
    rig, pose, pixels = case
    renderer = _RENDERER.RefractiveFishRenderer(rig, pixels, dtype=torch.float64)
    expected = renderer.render(pose).union_alpha

    def forbidden(*args, **kwargs):
        raise AssertionError("union-only evaluation must not invoke full rendering")

    monkeypatch.setattr(renderer, "render", forbidden)
    torch.testing.assert_close(render_union(renderer, pose), expected, rtol=0, atol=0)

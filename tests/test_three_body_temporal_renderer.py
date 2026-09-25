"""Matched-time ray rendering preserves the frozen per-frame values/derivatives."""

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
    spec = importlib.util.spec_from_file_location(f"three_body_temporal_test_{name}",
                                                 _LAB / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_RENDERER = load_module("renderer")
render_union = load_module("fast_union").render_union
_PREVIOUS = sys.modules.get("renderer")
sys.modules["renderer"] = _RENDERER
try:
    TemporalUnionRenderer = load_module("temporal_renderer").TemporalUnionRenderer
finally:
    if _PREVIOUS is None:
        sys.modules.pop("renderer", None)
    else:
        sys.modules["renderer"] = _PREVIOUS


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
    times = np.stack([pose, pose + np.array([3., -2., 1.]),
                      pose + np.array([-1., 5., -2.])])
    poses = np.stack([times, times + np.array([0.15, -0.1, 0.2])])
    offsets = np.array([[-3., 0.], [0., -2.], [2., 1.]])
    pixels = [{camera.view: (camera.project(p.reshape(-1, 3))[:, None]
                             + offsets[None]).reshape(-1, 2)
               for camera in rig.cameras} for p in times]
    return rig, poses, pixels


def per_frame_union(renderer, poses, expanded_width):
    outputs = []
    for t in range(renderer.n_times):
        one = _RENDERER.RefractiveFishRenderer(
            renderer.rig, {v: renderer.pixels[t, vi] for vi, v in enumerate(VIEWS)},
            renderer.config, dtype=renderer.dtype)
        outputs.append(render_union(one, poses[:, t], expanded_width[:, t]))
        torch.testing.assert_close(renderer.ray_valid[t], one.ray_valid)
    return torch.stack(outputs, dim=1)


def width_case(kind, batch, times):
    values = {
        "scalar": torch.tensor(0.061, dtype=torch.float64),
        "fish": torch.tensor([0.051, 0.068, 0.074], dtype=torch.float64),
        "batch_fish": torch.linspace(0.051, 0.074, batch * 3).reshape(batch, 3).double(),
        "time_fish": torch.linspace(0.051, 0.074, times * 3).reshape(1, times, 3).double(),
        "batch_time_fish": torch.linspace(0.051, 0.074, batch * times * 3)
        .reshape(batch, times, 3).double(),
    }
    width = values[kind].requires_grad_()
    expanded = width[:, None] if kind == "batch_fish" else width
    return width, expanded.expand(batch, times, 3)


@pytest.mark.parametrize("checkpoint_chunks", [False, True])
@pytest.mark.parametrize("width_kind", ["scalar", "fish", "batch_fish", "time_fish",
                                       "batch_time_fish"])
def test_values_and_every_pose_width_gradient_match_loop(case, checkpoint_chunks, width_kind):
    rig, poses, pixels = case
    renderer = TemporalUnionRenderer(
        rig, pixels, _RENDERER.RendererConfig(pixel_chunk_size=7), time_chunk_size=2,
        checkpoint_chunks=checkpoint_chunks)
    pose = torch.tensor(poses, dtype=torch.float64, requires_grad=True)
    width, expanded = width_case(width_kind, *pose.shape[:2])
    expected = per_frame_union(renderer, pose, expanded)
    actual = renderer.render(pose, width)
    assert actual.shape == (2, 3, 3, 27)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Time/view/pixel-specific signed weights expose accidental time mixing.
    weights = torch.linspace(-0.7, 1.3, actual.numel(), dtype=torch.float64).reshape(actual.shape)
    reference = torch.autograd.grad((expected.square() * weights).sum(), (pose, width),
                                    retain_graph=True)
    gradients = torch.autograd.grad((actual.square() * weights).sum(), (pose, width))
    for got, want in zip(gradients, reference, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-11)
    assert (torch.linalg.vector_norm(gradients[0], dim=-1) > 1e-8).all()
    assert (gradients[1].abs() > 1e-8).all()


@pytest.mark.parametrize("time_chunk_size", [1, 2, 50])
def test_time_pixel_chunking_and_default_width_have_identical_values(case, time_chunk_size):
    rig, poses, pixels = case
    renderer = TemporalUnionRenderer(
        rig, pixels, _RENDERER.RendererConfig(pixel_chunk_size=4),
        time_chunk_size=time_chunk_size)
    expected = per_frame_union(renderer, torch.tensor(poses),
                               torch.full((2, 3, 3), 0.06, dtype=torch.float64))
    torch.testing.assert_close(renderer.render(poses), expected, rtol=0, atol=0)


def test_different_frame_counts_missing_views_and_empty_time_have_zero_gradients(case):
    rig, poses, pixels = case
    pixels[0][VIEWS[1]] = pixels[0][VIEWS[1]][:3]
    pixels[0][VIEWS[2]][4] = np.nan
    pixels[1] = {}
    pixels[2] = {VIEWS[0]: pixels[2][VIEWS[0]][:7]}
    renderer = TemporalUnionRenderer(rig, pixels, time_chunk_size=2)
    pose = torch.tensor(poses, requires_grad=True)
    alpha = renderer.render(pose)
    assert renderer.ray_valid.shape == (3, 3, 27)
    assert renderer.pixel_counts[0][VIEWS[1]] == 3
    assert not renderer.ray_valid[1].any()
    assert renderer.ray_valid[2, 0].sum() == 7
    assert not renderer.ray_valid[2, 1:].any()
    assert not alpha[:, ~renderer.ray_valid].any()
    gradient, = torch.autograd.grad(alpha.sum(), pose)
    assert not gradient[:, 1].any()
    assert torch.linalg.vector_norm(gradient[:, 0]) > 1e-8
    expected = per_frame_union(renderer, pose, torch.full((2, 3, 3), 0.06, dtype=torch.float64))
    torch.testing.assert_close(alpha, expected, rtol=0, atol=0)


def test_output_time_depends_only_on_its_own_pose_and_corresponding_rays(case):
    rig, poses, pixels = case
    renderer = TemporalUnionRenderer(rig, pixels)
    pose = torch.tensor(poses, requires_grad=True)
    alpha = renderer.render(pose)
    gradient, = torch.autograd.grad(alpha[:, 1].sum(), pose)
    assert not gradient[:, [0, 2]].any()
    assert torch.linalg.vector_norm(gradient[:, 1]) > 1e-8
    changed = pose.detach().clone()
    changed[:, 1] += torch.tensor([12., -8., 5.])
    other = renderer.render(changed)
    torch.testing.assert_close(other[:, [0, 2]], alpha[:, [0, 2]], rtol=0, atol=0)
    assert (other[:, 1] - alpha[:, 1]).abs().max() > 0.1


def test_reordering_frames_reorders_outputs_without_pairing_other_times_rays(case):
    rig, poses, pixels = case
    original = TemporalUnionRenderer(rig, pixels, time_chunk_size=2)
    order = [2, 0, 1]
    permuted = TemporalUnionRenderer(rig, [pixels[t] for t in order], time_chunk_size=1)
    expected = original.render(poses)[:, order]
    actual = permuted.render(poses[:, order])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(permuted.ray_valid, original.ray_valid[order])
    # Deliberately mismatching pixels alone must fail that equality.
    wrong = permuted.render(poses)
    assert (wrong - expected).abs().max() > 0.1


def test_one_time_and_half_ray_clamp_match_original(case):
    rig, poses, pixels = case
    renderer = TemporalUnionRenderer(rig, pixels[:1], time_chunk_size=5)
    pose = torch.tensor(poses[:, :1], requires_grad=True)
    # Above the top water-ray origin: this checks half-ray t>=0 rather than an
    # infinite-line renderer; other cameras still supply their own water rays.
    changed = pose.detach().clone()
    changed[:, :, 0, :, 2] = 200.
    changed.requires_grad_()
    actual = renderer.render(changed)
    expected = per_frame_union(renderer, changed,
                               torch.full((2, 1, 3), 0.06, dtype=torch.float64))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad, = torch.autograd.grad(actual.sum(), changed, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected.sum(), changed)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-11)


def test_checkpoint_avoids_saving_all_large_time_ray_products(case):
    rig, poses, pixels = case
    sizes = []
    outputs, gradients = [], []
    for enabled in (False, True):
        renderer = TemporalUnionRenderer(rig, pixels, time_chunk_size=1,
                                         checkpoint_chunks=enabled)
        pose = torch.tensor(poses, requires_grad=True)
        saved = []

        def pack(tensor, saved=saved):
            saved.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            alpha = renderer.render(pose)
        sizes.append(sum(saved))
        outputs.append(alpha.detach())
        gradients.append(torch.autograd.grad(alpha.sum(), pose)[0])
    assert sizes[1] < 0.1 * sizes[0]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)


def test_invalid_shapes_and_empty_windows_raise(case):
    rig, poses, pixels = case
    with pytest.raises(ValueError, match="at least one frame"):
        TemporalUnionRenderer(rig, [])
    with pytest.raises(ValueError, match="at least one sampled"):
        TemporalUnionRenderer(rig, [{}, {}])
    with pytest.raises(ValueError, match="time_chunk_size"):
        TemporalUnionRenderer(rig, pixels, time_chunk_size=0)
    renderer = TemporalUnionRenderer(rig, pixels)
    with pytest.raises(ValueError, match="matching cached times"):
        renderer.render(poses[:, :1])
    with pytest.raises(ValueError, match="width_ratio"):
        renderer.render(poses, np.ones((3, 3)))
    with pytest.raises(ValueError, match="finite and positive"):
        renderer.render(poses, -0.1)
    bad = poses.copy()
    bad[0, 1, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="pose_xyz must be finite"):
        renderer.render(bad)

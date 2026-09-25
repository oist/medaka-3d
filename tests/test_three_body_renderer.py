"""Physical ray response and useful gradients, independently of tracking labels."""

import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import minimize_scalar

from medaka3d.calib import Rig
from medaka3d.types import VIEWS

_PATH = Path(__file__).resolve().parents[1] / "labs/20260909_three_body/renderer.py"
_SPEC = importlib.util.spec_from_file_location("three_body_renderer_test_module", _PATH)
_RENDERER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RENDERER
_SPEC.loader.exec_module(_RENDERER)
RendererConfig = _RENDERER.RendererConfig
RefractiveFishRenderer = _RENDERER.RefractiveFishRenderer
GaussianBodies = _RENDERER.GaussianBodies


@dataclass
class ParallelCamera:
    view: str

    def rays(self, uv):
        origins = np.column_stack([uv, np.zeros(len(uv))])
        return origins, np.broadcast_to([0, 0, 1], origins.shape).copy()


@pytest.fixture
def parallel_rig():
    return Rig(tuple(ParallelCamera(view) for view in VIEWS), Path("synthetic-parallel-rays"))


@pytest.fixture
def poses():
    centers = np.array([[55., 60., 65.], [115., 105., 90.], [145., 140., 130.]])
    offsets = np.array([[5., 0., 0.], [0., 0., 0.], [-12., 2., 1.]])
    return centers[:, None, :] + offsets[None]


@pytest.fixture
def refractive_rig(rig):
    return replace(rig, cameras=tuple(
        replace(cam, index_air=1.0003, index_glass=1.52, index_water=1.3325,
                layer_count=1 if cam.view == VIEWS[0] else 2,
                distortion=(-0.15, 0.07, 0.003, -0.002, -0.01))
        for cam in rig.cameras
    ))


def sample_pixels(rig, poses, offsets=None):
    if offsets is None:
        offsets = np.array([[-3., 1.], [1., 2.], [4., -2.], [-9., 3.]])
    return {cam.view: (cam.project(poses[:, 1])[:, None] + offsets[None]).reshape(-1, 2)
            for cam in rig.cameras}


def one_gaussian_bodies(center, tangent, axial, radial):
    # Direct ellipsoid test bypasses the body approximation: compare its ray
    # response with known geometry instead of fitting this renderer to itself.
    return GaussianBodies(
        centers=torch.tensor(center, dtype=torch.float64).expand(1, 3, 1, 3),
        tangents=torch.tensor(tangent, dtype=torch.float64).expand(1, 3, 1, 3),
        axial_sigma_mm=torch.full((1, 3, 1), axial, dtype=torch.float64),
        radial_sigma_mm=torch.full((1, 3, 1), radial, dtype=torch.float64),
        segment_lengths_mm=torch.ones((1, 3, 2), dtype=torch.float64),
        nondegenerate=torch.ones((1, 3), dtype=torch.bool),
    )


def test_axis_aligned_gaussian_has_closed_form_radial_response(parallel_rig):
    renderer = RefractiveFishRenderer(parallel_rig, {view: np.array([[2., 1.]]) for view in VIEWS},
                                     dtype=torch.float64)
    bodies = one_gaussian_bodies([0, 0, 20], [1, 0, 0], axial=4, radial=2)
    alpha, depth, q = renderer._ray_response(bodies, 0, 1)
    expected_q = (2 / 4) ** 2 + (1 / 2) ** 2
    torch.testing.assert_close(q, torch.full_like(q, expected_q))
    torch.testing.assert_close(alpha, torch.full_like(alpha, 0.85 * np.exp(-expected_q / 2)))
    torch.testing.assert_close(depth, torch.full_like(depth, 20))


def test_tilted_gaussian_matches_independent_numerical_ray_minimum(parallel_rig):
    renderer = RefractiveFishRenderer(parallel_rig, {view: np.array([[0., 0.]]) for view in VIEWS},
                                     dtype=torch.float64)
    tangent = np.array([1., 0., 1.]) / np.sqrt(2)
    center = np.array([2., 3., 20.])
    covariance = 2 ** 2 * np.eye(3) + (5 ** 2 - 2 ** 2) * np.outer(tangent, tangent)
    inverse = np.linalg.inv(covariance)

    def ray_distance(depth):
        residual = np.array([0., 0., depth]) - center
        return residual @ inverse @ residual

    optimum = minimize_scalar(ray_distance, bounds=(0, 60), method="bounded")
    alpha, depth, q = renderer._ray_response(one_gaussian_bodies(center, tangent, 5, 2), 0, 1)
    np.testing.assert_allclose(depth.numpy(), optimum.x, atol=1e-7)
    np.testing.assert_allclose(q.numpy(), optimum.fun, atol=1e-10)
    np.testing.assert_allclose(alpha.numpy(), 0.85 * np.exp(-optimum.fun / 2), atol=1e-10)


def test_gaussian_behind_water_origin_cannot_use_the_infinite_line(parallel_rig):
    renderer = RefractiveFishRenderer(parallel_rig, {view: np.array([[0., 0.]]) for view in VIEWS},
                                     dtype=torch.float64)
    alpha, _, q = renderer._ray_response(one_gaussian_bodies([0, 0, -20], [1, 0, 0], 4, 2),
                                        0, 1)
    np.testing.assert_allclose(q.numpy(), 100)
    assert float(alpha.max()) < 1e-20


def test_body_centers_follow_both_node_segments_and_width_is_sigma(refractive_rig, poses):
    renderer = RefractiveFishRenderer(refractive_rig, sample_pixels(refractive_rig, poses),
                                     dtype=torch.float64)
    original = poses.copy()
    body = renderer.gaussians(poses)
    assert body.centers.shape == (1, 3, 8, 3)
    assert body.nondegenerate.all()
    for fish in range(3):
        for index, center in enumerate(body.centers[0, fish].numpy()):
            start, stop = poses[fish, :2] if index < 2 else poses[fish, 1:]
            direction = stop - start
            fraction = np.dot(center - start, direction) / np.dot(direction, direction)
            assert 0 < fraction < 1
            np.testing.assert_allclose(center, start + fraction * direction, atol=1e-12)
    doubled_width = renderer.gaussians(poses, width_ratio=0.12)
    torch.testing.assert_close(doubled_width.radial_sigma_mm, 2 * body.radial_sigma_mm)
    torch.testing.assert_close(doubled_width.axial_sigma_mm, body.axial_sigma_mm)
    np.testing.assert_array_equal(poses, original)


def test_refraction_ray_cache_and_padding_stay_explicit(refractive_rig, poses):
    pixels = sample_pixels(refractive_rig, poses)
    pixels[VIEWS[1]] = pixels[VIEWS[1]][:2]
    pixels[VIEWS[2]] = np.array([[np.nan, np.nan]])
    renderer = RefractiveFishRenderer(refractive_rig, pixels, dtype=torch.float64)
    result = renderer.render(poses)
    origins, directions = refractive_rig.camera(VIEWS[0]).rays(pixels[VIEWS[0]])
    np.testing.assert_allclose(renderer.ray_origins[0], origins)
    np.testing.assert_allclose(renderer.ray_directions[0], directions)
    assert result.per_fish_alpha.shape == (1, 3, 12, 3)
    assert result.ray_valid[0].all() and int(result.ray_valid[1].sum()) == 2
    assert not result.ray_valid[2].any()
    assert (result.union_alpha[0][~result.ray_valid] == 0).all()
    assert torch.isinf(result.per_fish_depth_mm[0][~result.ray_valid]).all()
    assert (result.ownership[0][~result.ray_valid] == 0).all()


def test_pose_and_width_autograd_agree_with_finite_differences(refractive_rig, poses):
    renderer = RefractiveFishRenderer(refractive_rig, sample_pixels(refractive_rig, poses),
                                     dtype=torch.float64)
    pose = torch.tensor(poses, dtype=torch.float64, requires_grad=True)
    width = torch.tensor([0.055, 0.067, 0.071], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda p, w: renderer.render(p, w).union_alpha,
                                   (pose, width), eps=1e-5, atol=2e-5, rtol=2e-4)
    gradients = torch.autograd.grad(renderer.render(pose, width).union_alpha.sum(), (pose, width))
    assert all(torch.isfinite(value).all() for value in gradients)
    assert (torch.linalg.vector_norm(gradients[0], dim=-1) > 1e-8).all()
    assert (gradients[1].abs() > 1e-8).all()


def test_image_loss_gradient_moves_translation_towards_supported_pose(refractive_rig, poses):
    yy, xx = np.meshgrid(np.linspace(-12, 12, 7), np.linspace(-18, 18, 7), indexing="ij")
    pixels = sample_pixels(refractive_rig, poses, np.column_stack([xx.ravel(), yy.ravel()]))
    renderer = RefractiveFishRenderer(refractive_rig, pixels, dtype=torch.float64)
    target = renderer.render(poses).union_alpha.detach()
    shift = torch.tensor([1.3, -0.9, 0.7], dtype=torch.float64, requires_grad=True)
    initial = (renderer.render(torch.tensor(poses) + shift).union_alpha - target).square().mean()
    gradient, = torch.autograd.grad(initial, shift)
    updated = shift.detach() - 0.1 * gradient / gradient.norm()
    after = (renderer.render(torch.tensor(poses) + updated).union_alpha - target).square().mean()
    assert after < initial.detach()
    assert updated.norm() < shift.detach().norm()


def depth_poses():
    centers = np.array([[0., 0., 20.], [0., 0., 40.], [0., 0., 60.]])
    body = np.array([[5., 0., 0.], [0., 0., 0.], [-12., 0., 0.]])
    return centers[:, None] + body[None]


def test_front_body_occludes_rear_ownership_without_erasing_amodal_support(parallel_rig):
    renderer = RefractiveFishRenderer(parallel_rig, {view: np.array([[0., 0.]]) for view in VIEWS},
                                     dtype=torch.float64)
    result = renderer.render(depth_poses())
    alpha = result.per_fish_alpha[0, 0, 0]
    torch.testing.assert_close(alpha, alpha[:1].expand(3))
    np.testing.assert_allclose(result.per_fish_depth_mm[0, 0, 0], [20, 40, 60])
    assert result.ownership[0, 0, 0, 0] > 0.8
    assert result.visible_alpha[0, 0, 0, 0] > result.visible_alpha[0, 0, 0, 1]
    torch.testing.assert_close(result.visible_alpha.sum(dim=-1), result.union_alpha)
    torch.testing.assert_close(result.ownership.sum(dim=-1), torch.ones_like(result.union_alpha))


def test_fish_permutation_changes_only_fish_axis_not_union_or_physical_visibility(parallel_rig):
    renderer = RefractiveFishRenderer(parallel_rig,
                                     {view: np.array([[0., 0.], [1., 2.]]) for view in VIEWS},
                                     dtype=torch.float64)
    pose = depth_poses()
    permutation = [2, 0, 1]
    original = renderer.render(pose)
    permuted = renderer.render(pose[permutation])
    torch.testing.assert_close(original.union_alpha, permuted.union_alpha)
    for field in ("per_fish_alpha", "per_fish_depth_mm", "visible_alpha", "ownership"):
        torch.testing.assert_close(getattr(original, field)[..., permutation],
                                   getattr(permuted, field))


def test_exact_depth_tie_does_not_invent_identity_ownership(parallel_rig):
    renderer = RefractiveFishRenderer(parallel_rig, {view: np.array([[0., 0.]]) for view in VIEWS},
                                     dtype=torch.float64)
    pose = np.repeat(depth_poses()[:1], 3, axis=0)
    result = renderer.render(pose)
    torch.testing.assert_close(result.ownership, torch.full_like(result.ownership, 1 / 3))
    torch.testing.assert_close(result.visible_alpha.sum(dim=-1), result.union_alpha)


def test_equal_depth_unequal_alpha_ownership_is_permutation_equivariant(parallel_rig):
    renderer = RefractiveFishRenderer(parallel_rig, {view: np.array([[0., 1.]]) for view in VIEWS},
                                     dtype=torch.float64)
    pose = np.repeat(depth_poses()[:1], 3, axis=0)
    widths = torch.tensor([0.045, 0.065, 0.09], dtype=torch.float64)
    permutation = [2, 0, 1]
    result = renderer.render(pose, widths)
    permuted = renderer.render(pose[permutation], widths[permutation])
    torch.testing.assert_close(result.ownership[..., permutation], permuted.ownership)
    torch.testing.assert_close(result.visible_alpha.sum(dim=-1), result.union_alpha)


def test_pixel_chunking_and_batching_preserve_values_and_gradients(refractive_rig, poses):
    pixels = sample_pixels(refractive_rig, poses)
    renderer = RefractiveFishRenderer(refractive_rig, pixels, dtype=torch.float64)
    chunked = RefractiveFishRenderer(refractive_rig, pixels,
                                    RendererConfig(pixel_chunk_size=3), dtype=torch.float64)
    batches = torch.tensor(np.stack([poses, poses + 0.3]), requires_grad=True)
    full = renderer.render(batches).union_alpha
    split = chunked.render(batches).union_alpha
    torch.testing.assert_close(full, split)
    grad_full, = torch.autograd.grad(full.sum(), batches)
    grad_split, = torch.autograd.grad(split.sum(), batches)
    torch.testing.assert_close(grad_full, grad_split)
    torch.testing.assert_close(full[:1], renderer.render(batches[0]).union_alpha)


def test_cuda_cpu_parity_includes_pose_gradient(refractive_rig, poses):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; CPU geometry and derivatives are tested separately")
    pixels = sample_pixels(refractive_rig, poses)
    cpu = RefractiveFishRenderer(refractive_rig, pixels, dtype=torch.float64)
    gpu = RefractiveFishRenderer(refractive_rig, pixels, device="cuda", dtype=torch.float64)
    cpu_pose = torch.tensor(poses, dtype=torch.float64, requires_grad=True)
    gpu_pose = torch.tensor(poses, dtype=torch.float64, device="cuda", requires_grad=True)
    expected = cpu.render(cpu_pose)
    actual = gpu.render(gpu_pose)
    torch.testing.assert_close(actual.union_alpha.cpu(), expected.union_alpha,
                               atol=1e-10, rtol=1e-9)
    cpu_gradient, = torch.autograd.grad(expected.union_alpha.sum(), cpu_pose)
    gpu_gradient, = torch.autograd.grad(actual.union_alpha.sum(), gpu_pose)
    torch.testing.assert_close(gpu_gradient.cpu(), cpu_gradient, atol=1e-9, rtol=1e-8)


def test_degenerate_body_is_finite_but_flagged_and_bad_inputs_fail(parallel_rig):
    renderer = RefractiveFishRenderer(parallel_rig, {view: np.array([[0., 0.]]) for view in VIEWS})
    pose = np.repeat(depth_poses()[:, 1:2], 3, axis=1)
    result = renderer.render(pose)
    assert not result.bodies.nondegenerate.any()
    assert torch.isfinite(result.union_alpha).all()
    with pytest.raises(ValueError, match="must be finite"):
        renderer.render(np.full((3, 3, 3), np.nan))
    with pytest.raises(ValueError, match="width_ratio"):
        renderer.render(depth_poses(), width_ratio=-1)
    with pytest.raises(ValueError, match="shape"):
        renderer.render(np.zeros((2, 3, 3)))
    with pytest.raises(ValueError, match="sampled image pixel"):
        RefractiveFishRenderer(parallel_rig, {})

"""Representation-only checks: no real images, fit, observation or identity labels."""

import importlib
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from medaka3d.calib import Rig
from medaka3d.types import VIEWS


def load_module():
    lab = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
    names = ("renderer", "curved_renderer")
    saved, old_path = {name: sys.modules.get(name) for name in names}, sys.path.copy()
    try:
        sys.path.insert(0, str(lab))
        for name in names:
            sys.modules.pop(name, None)
        return importlib.import_module("curved_renderer")
    finally:
        sys.path[:] = old_path
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


m = load_module()


@dataclass
class ParallelCamera:
    view: str

    def rays(self, pixels):
        origins = np.column_stack((pixels, np.zeros(len(pixels))))
        return origins, np.broadcast_to([0., 0., 1.], origins.shape).copy()


@pytest.fixture
def parallel_rig():
    return Rig(tuple(ParallelCamera(v) for v in VIEWS), Path("synthetic-curvature-rays"))


@pytest.fixture
def pose():
    pectorals = np.array([[30., 30., 25.], [52., 49., 40.], [73., 26., 70.]])
    offsets = np.array([[4., 1., .3], [0., 0., 0.], [-10., 2., 1.]])
    return pectorals[:, None] + offsets[None]


def renderer(rig, pose, *, cls=m.CurvedFishRenderer, pixels=None):
    if pixels is None:
        uv = pose[:, :, :2].reshape(-1, 2)
        uv = np.concatenate((uv, uv + np.array([0., 2.5])))
        pixels = {v: uv for v in VIEWS}
    return cls(rig, pixels, dtype=torch.float64)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("width", [None, .08, [0.03, 0.07, 0.05]])
def test_zero_offset_exactly_matches_all_old_gaussians_and_visibility(
    parallel_rig, pose, batch, width,
):
    old = renderer(parallel_rig, pose, cls=m.RefractiveFishRenderer)
    new = renderer(parallel_rig, pose)
    value = np.stack((pose, pose + np.array([1., -1., 2.]))) if batch else pose
    a = old.render(value, width)
    b = new.render(value, width, tail_control_offset_mm=np.zeros((3, 3)))
    for field in fields(a.bodies):
        assert torch.equal(getattr(a.bodies, field.name), getattr(b.bodies, field.name))
    for field in fields(a):
        if field.name != "bodies":
            assert torch.equal(getattr(a, field.name), getattr(b, field.name))


def test_curved_centers_follow_independent_bezier_and_tangents(parallel_rig, pose):
    r = renderer(parallel_rig, pose)
    offset = np.array([[0., 8., 1.], [2., -5., 3.], [1., 0., -4.]])
    body = r.gaussians(pose, tail_control_offset_mm=offset)
    old = m.RefractiveFishRenderer.gaussians(r, pose)
    t = r._fractions[r._head_count:].numpy()
    start, stop = pose[:, 1, None], pose[:, 2, None]
    control = (start + stop) / 2 + offset[:, None]
    expected = (1-t)[None, :, None]**2 * start + 2*(1-t)[None, :, None]*t[None, :, None]*control
    expected += t[None, :, None]**2 * stop
    derivative = 2*(1-t)[None, :, None]*(control-start) + 2*t[None, :, None]*(stop-control)
    np.testing.assert_allclose(body.centers[0, :, r._head_count:], expected, atol=1e-13)
    np.testing.assert_allclose(body.tangents[0, :, r._head_count:],
                               derivative / np.linalg.norm(derivative, axis=-1, keepdims=True))
    for key in ("centers", "tangents"):
        assert torch.equal(getattr(body, key)[:, :, :r._head_count],
                           getattr(old, key)[:, :, :r._head_count])
    for key in ("axial_sigma_mm", "radial_sigma_mm", "segment_lengths_mm"):
        assert torch.equal(getattr(body, key), getattr(old, key))


def test_fixed_reference_scales_remain_chord_based_and_support_batched_width(parallel_rig, pose):
    r = renderer(parallel_rig, pose)
    current = np.stack((pose.copy(), pose.copy()))
    current[:, :, 2] += [-4., 2., 0.]
    widths = torch.tensor([[.04, .05, .06], [.07, .08, .09]], dtype=torch.float64)
    reference = np.broadcast_to(pose, current.shape).copy()
    old = m.RefractiveFishRenderer.gaussians(r, reference, widths)
    body = r.gaussians(current, widths, tail_control_offset_mm=np.full((2, 3, 3), 5.),
                       sigma_reference_pose_xyz=pose)
    assert torch.equal(body.axial_sigma_mm, old.axial_sigma_mm)
    assert torch.equal(body.radial_sigma_mm, old.radial_sigma_mm)
    assert not torch.equal(body.segment_lengths_mm, old.segment_lengths_mm)
    actual = r.render(current, widths, tail_control_offset_mm=np.full((2, 3, 3), 5.),
                      sigma_reference_pose_xyz=pose)
    assert torch.equal(actual.bodies.radial_sigma_mm, old.radial_sigma_mm)


def test_bending_one_body_changes_only_its_geometry_and_leaves_inputs_and_renderer_unbound(
    parallel_rig, pose,
):
    r = renderer(parallel_rig, pose)
    value = torch.tensor(pose, dtype=torch.float64, requires_grad=True)
    offset = torch.zeros((3, 3), dtype=torch.float64)
    offset[0, 1] = 8
    before, saved_offset = value.detach().clone(), offset.clone()
    straight = r.render(value)
    bent = r.render(value, tail_control_offset_mm=offset, sigma_reference_pose_xyz=value)
    for name in ("centers", "tangents", "axial_sigma_mm", "radial_sigma_mm"):
        assert torch.equal(getattr(straight.bodies, name)[:, 1:], getattr(bent.bodies, name)[:, 1:])
    assert torch.equal(straight.per_fish_alpha[..., 1:], bent.per_fish_alpha[..., 1:])
    assert torch.equal(value, before) and torch.equal(offset, saved_offset)
    assert "_curve_arguments" not in vars(r)
    assert torch.equal(straight.union_alpha, r.render(value).union_alpha)


def test_nonselfinverse_permutation_keeps_curve_width_and_reference_attached_to_raw_body(
    parallel_rig, pose,
):
    r = renderer(parallel_rig, pose)
    offset = np.array([[0., 4., 0.], [1., -3., 1.], [2., 1., -2.]])
    width, p = np.array([.04, .06, .08]), [2, 0, 1]
    a = r.render(pose, width, tail_control_offset_mm=offset, sigma_reference_pose_xyz=pose)
    b = r.render(pose[p], width[p], tail_control_offset_mm=offset[p],
                 sigma_reference_pose_xyz=pose[p])
    torch.testing.assert_close(a.union_alpha, b.union_alpha, rtol=2e-14, atol=2e-14)
    for key in ("per_fish_alpha", "per_fish_depth_mm", "visible_alpha", "ownership"):
        torch.testing.assert_close(getattr(b, key), getattr(a, key)[..., p], rtol=2e-14, atol=2e-14)


def test_coincident_curved_depth_ties_are_symmetric(parallel_rig, pose):
    same = np.repeat(pose[:1], 3, axis=0)
    r = renderer(parallel_rig, same)
    result = r.render(same, tail_control_offset_mm=np.tile([0., 7., 0.], (3, 1)))
    torch.testing.assert_close(result.visible_alpha[..., 0], result.union_alpha / 3)
    torch.testing.assert_close(result.visible_alpha[..., 0], result.visible_alpha[..., 2])


def test_zero_offset_still_has_a_useful_gradient_and_float64_render_gradcheck(parallel_rig, pose):
    r = renderer(parallel_rig, pose)
    offset = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    centers = r.gaussians(pose, tail_control_offset_mm=offset).centers
    derivative = torch.autograd.grad(centers.sum(), offset)[0]
    assert torch.isfinite(derivative).all() and torch.all(derivative > 0)
    value = torch.tensor(pose, dtype=torch.float64, requires_grad=True)
    offset = torch.tensor([[.2, 2., .4], [.3, -1., .2], [.4, .8, -.2]],
                          dtype=torch.float64, requires_grad=True)
    width = torch.tensor([.05, .06, .07], dtype=torch.float64, requires_grad=True)

    def output(x, o, w):
        rendered = r.render(x, w, tail_control_offset_mm=o)
        return torch.cat((rendered.union_alpha.flatten(), rendered.visible_alpha.flatten()))

    assert torch.autograd.gradcheck(output, (value, offset, width), eps=1e-6, atol=2e-5, rtol=2e-4,
                                    fast_mode=True)


def test_arc_length_matches_independent_quadrature_and_zero_chord_limit(pose):
    offsets = np.array([[0., 8., 1.], [2., -5., 3.], [1., 0., -4.]])
    result = m.tail_curve_diagnostics(pose, offsets)
    chord = pose[:, 2] - pose[:, 1]
    for fish in range(3):
        expected = quad(
            lambda t, fish=fish: np.linalg.norm(chord[fish]+(2-4*t)*offsets[fish]), 0., 1.,
            epsabs=1e-11, epsrel=1e-11,
        )[0]
        assert result.arc_length_mm[0, fish].item() == pytest.approx(expected, abs=2e-10)
    assert torch.all(result.arc_length_mm > result.chord_length_mm)
    straight = m.tail_curve_diagnostics(pose)
    assert torch.equal(straight.arc_length_mm, straight.chord_length_mm)
    collapsed = pose.copy()
    collapsed[:, 2] = collapsed[:, 1]
    result = m.tail_curve_diagnostics(collapsed, offsets)
    np.testing.assert_allclose(result.arc_length_mm[0], np.linalg.norm(offsets, axis=1), atol=1e-12)
    assert result.degenerate.all()


def test_cusp_between_samples_and_sampled_zero_derivative_remain_finite_but_flagged(
    parallel_rig, pose,
):
    r = renderer(parallel_rig, pose)
    chord = pose[:, 2] - pose[:, 1]
    # Place B'(t)=0 exactly at the first tail Gaussian.
    t = r._fractions[r._head_count].item()
    offset = -chord / (2-4*t)
    result = r.render(pose, tail_control_offset_mm=offset)
    info = m.tail_curve_diagnostics(pose, offset)
    assert not result.bodies.nondegenerate.any() and info.degenerate.all()
    assert torch.isfinite(result.bodies.tangents).all()
    assert torch.isfinite(result.union_alpha).all()
    for fish in range(3):
        expected = quad(lambda u, fish=fish: np.linalg.norm(chord[fish]+(2-4*u)*offset[fish]),
                        0., 1., points=[t], epsabs=1e-10)[0]
        assert info.arc_length_mm[0, fish].item() == pytest.approx(expected, abs=2e-10)
    offset = chord * 2  # cusp at t=.625, between this renderer's sample centers
    assert m.tail_curve_diagnostics(pose, offset).degenerate.all()
    assert not r.gaussians(pose, tail_control_offset_mm=offset).nondegenerate.any()


def test_independent_curved_line_probe_is_explained_without_moving_any_anatomical_node(
    parallel_rig, pose,
):
    value = pose.copy()
    value[0] = [[34., 30., 25.], [30., 30., 25.], [20., 30., 25.]]
    # Analytic Bezier midpoint from an independently specified control (25,40,25).
    probe = np.array([[25., 35.]])
    r = renderer(parallel_rig, value, pixels={v: probe for v in VIEWS})
    offset = np.zeros((3, 3))
    offset[0] = [0., 10., 0.]
    before = value.copy()
    straight = r.render(value)
    bent = r.render(value, tail_control_offset_mm=offset)
    assert straight.per_fish_alpha[0, 0, 0, 0] < .01
    assert bent.per_fish_alpha[0, 0, 0, 0] > .5
    np.testing.assert_array_equal(value, before)
    assert torch.equal(straight.bodies.segment_lengths_mm, bent.bodies.segment_lengths_mm)


def test_real_refractive_rays_and_invalid_pixels_keep_old_zero_offset_parity(rig, pose):
    refractive = replace(rig, cameras=tuple(replace(cam, index_air=1.0003, index_glass=1.52,
                                                   index_water=1.333) for cam in rig.cameras))
    pixels = {cam.view: np.vstack((cam.project(pose[:, 1]), [np.nan, np.nan]))
              for cam in refractive.cameras}
    old = renderer(refractive, pose, cls=m.RefractiveFishRenderer, pixels=pixels).render(pose)
    new = renderer(refractive, pose, pixels=pixels).render(pose)
    assert torch.equal(old.union_alpha, new.union_alpha)
    assert torch.equal(old.visible_alpha, new.visible_alpha)
    assert not new.ray_valid[:, -1].any() and not new.union_alpha[..., -1].any()


@pytest.mark.parametrize("offset", [np.zeros(3), np.zeros((2, 3, 3)), np.full((3, 3), np.nan),
                                    np.full((3, 3), np.inf)])
def test_invalid_offsets_fail_loudly(parallel_rig, pose, offset):
    r = renderer(parallel_rig, pose)
    with pytest.raises(ValueError, match="offset"):
        r.render(pose, tail_control_offset_mm=offset)
    with pytest.raises(ValueError, match="offset"):
        m.tail_curve_diagnostics(pose, offset)


def test_invalid_pose_reference_width_and_diagnostic_dtype_fail(parallel_rig, pose):
    r = renderer(parallel_rig, pose)
    for value in (np.zeros((0, 3, 3, 3)), np.full_like(pose, np.nan)):
        with pytest.raises(ValueError, match="pose"):
            r.render(value)
    with pytest.raises(ValueError, match="reference batch"):
        r.render(pose, sigma_reference_pose_xyz=np.stack((pose, pose)))
    with pytest.raises(ValueError, match="finite"):
        r.render(pose, sigma_reference_pose_xyz=np.full_like(pose, np.inf))
    with pytest.raises(ValueError, match="width"):
        r.render(pose, -0.1)
    with pytest.raises(ValueError, match="float"):
        m.tail_curve_diagnostics(pose, dtype=torch.int64)


def test_nonzero_batch_matches_separate_calls_without_curve_state_leaking(parallel_rig, pose):
    r = renderer(parallel_rig, pose)
    batch = np.stack((pose, pose + np.array([1., 2., 3.])))
    offsets = np.arange(18).reshape(2, 3, 3) / 3 - 3
    widths = np.array([[.04, .05, .06], [.08, .03, .05]])
    combined = r.render(batch, widths, tail_control_offset_mm=offsets,
                        sigma_reference_pose_xyz=pose)
    for index in range(2):
        separate = r.render(batch[index], widths[index], tail_control_offset_mm=offsets[index],
                            sigma_reference_pose_xyz=pose)
        assert torch.equal(combined.union_alpha[index], separate.union_alpha[0])
        assert torch.equal(combined.visible_alpha[index], separate.visible_alpha[0])
    assert "_curve_arguments" not in vars(r)


def test_arc_diagnostic_has_finite_correct_pose_and_offset_gradients(pose):
    value = torch.tensor(pose, dtype=torch.float64, requires_grad=True)
    offset = torch.tensor([[0., 8., 1.], [2., -5., 3.], [1., 0., -4.]],
                          dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, o: m.tail_curve_diagnostics(x, o).arc_length_mm,
        (value, offset), fast_mode=True, eps=1e-6, atol=2e-6, rtol=2e-5,
    )


def test_finite_but_overflowing_curve_is_rejected(parallel_rig, pose):
    r = renderer(parallel_rig, pose)
    offset = np.full((3, 3), 1e200)
    with pytest.raises(ValueError, match="overflow"):
        r.render(pose, tail_control_offset_mm=offset)
    with pytest.raises(ValueError, match="overflow"):
        m.tail_curve_diagnostics(pose, offset)

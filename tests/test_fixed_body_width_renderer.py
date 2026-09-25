"""Independent fixed-radius geometry checks on small synthetic CPU ray scenes."""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "work/three_body"))
import fixed_body_width_renderer as m  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402

from medaka3d.calib import Rig  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


class AxisCamera:
    def __init__(self, view, axis):
        self.view, self.axis = view, axis

    def rays(self, xy):
        origin = np.zeros((len(xy), 3), np.float64)
        other = [axis for axis in range(3) if axis != self.axis]
        origin[:, other] = xy
        direction = np.zeros_like(origin)
        direction[:, self.axis] = 1
        return origin, direction


def seed():
    return np.array(
        [
            [[6.0, 1.0, 10.0], [1.0, 1.5, 9.8], [-11.0, 3.5, 11.0]],
            [[15.0, 12.0, 16.0], [9.0, 10.0, 16.0], [0.0, 9.0, 16.5]],
            [[-6.0, -11.0, 24.0], [-9.0, -13.0, 24.0], [-18.0, -14.0, 25.0]],
        ],
        np.float64,
    )


def scene(*, missing=False):
    rig = Rig(
        tuple(AxisCamera(v, a) for v, a in zip(VIEWS, (2, 0, 1), strict=True)), Path("synthetic")
    )
    pixels = {view: np.array([[0.4, 0.7], [3.2, -2.1], [-5.0, 4.5]], np.float64) for view in VIEWS}
    if missing:
        pixels[VIEWS[0]][1] = np.nan
        pixels[VIEWS[1]] = np.empty((0, 2), np.float64)
    return rig, pixels


def renderers(reference=None, *, config=None, missing=False, dtype=torch.float64):
    rig, pixels = scene(missing=missing)
    config = config or RendererConfig()
    reference = seed() if reference is None else reference
    old = RefractiveFishRenderer(rig, pixels, config, device="cpu", dtype=dtype)
    fixed = m.FixedWidthRenderer(
        rig,
        pixels,
        reference_pose_xyz=reference,
        config=config,
        device="cpu",
        dtype=dtype,
    )
    return old, fixed


def exact(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert left.detach().cpu().numpy().tobytes() == right.detach().cpu().numpy().tobytes()


def exact_fields(left, right, *, omit=()):
    assert type(left) is type(right)
    for field in dataclasses.fields(left):
        if field.name in omit:
            continue
        a, b = getattr(left, field.name), getattr(right, field.name)
        if dataclasses.is_dataclass(a):
            exact_fields(a, b)
        else:
            exact(a, b)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_registered_seed_matches_all_old_forward_fields_with_padding_and_invalid_rays(dtype):
    old, fixed = renderers(missing=True, dtype=dtype)
    xyz = seed()
    before = xyz.tobytes()
    exact_fields(fixed.gaussians(xyz), old.gaussians(xyz))
    exact_fields(fixed.render(xyz), old.render(xyz))
    assert xyz.tobytes() == before
    assert not fixed.ray_valid[1].any() and not fixed.ray_valid[0, 1]
    q = fixed.render(xyz).per_fish_mahalanobis_sq
    assert torch.isinf(q[:, 1]).all() and torch.isinf(q[:, 0, 1]).all()


@pytest.mark.parametrize("motion", ["translation", "rotation", "stretch", "bend"])
def test_only_radial_field_is_replaced_for_moving_and_deforming_poses(motion):
    old, fixed = renderers()
    initial = seed()
    moved = initial.copy()
    if motion == "translation":
        moved += [2.5, -1.25, 3.0]
    elif motion == "rotation":
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        moved = (moved - moved[:, 1:2]) @ rotation.T + moved[:, 1:2]
    elif motion == "stretch":
        moved = (moved - moved[:, 1:2]) * 1.5 + moved[:, 1:2]
    else:
        moved[:, 1] += [0.5, 2.0, -0.75]
    batch = np.stack((initial, moved))
    before = batch.tobytes()
    actual, dynamic = fixed.gaussians(batch), old.gaussians(batch)
    exact_fields(actual, dynamic, omit=("radial_sigma_mm",))
    initial_radial = old.gaussians(initial).radial_sigma_mm
    exact(actual.radial_sigma_mm, initial_radial.expand(2, -1, -1))
    assert batch.tobytes() == before
    if motion == "stretch":
        torch.testing.assert_close(dynamic.radial_sigma_mm[1], initial_radial[0] * 1.5)
        assert not torch.equal(dynamic.radial_sigma_mm[1], actual.radial_sigma_mm[1])


def test_seed_forward_equality_does_not_claim_equal_radial_pose_gradient():
    old, fixed = renderers()
    xyz = torch.tensor(seed(), dtype=torch.float64, requires_grad=True)
    dynamic, static = old.gaussians(xyz), fixed.gaussians(xyz)
    exact(dynamic.radial_sigma_mm, static.radial_sigma_mm)
    assert dynamic.radial_sigma_mm.requires_grad and not static.radial_sigma_mm.requires_grad
    original_grad = torch.autograd.grad(dynamic.radial_sigma_mm.sum(), xyz)[0]
    fixed_grad = torch.autograd.grad(static.radial_sigma_mm.sum() + xyz.sum() * 0, xyz)[0]
    assert torch.count_nonzero(original_grad) > 0
    assert torch.count_nonzero(fixed_grad) == 0
    assert static.centers.requires_grad and static.axial_sigma_mm.requires_grad


@pytest.mark.parametrize("term", ["gap_distance", "union"])
def test_finite_difference_retains_center_tangent_axial_gradients(term):
    _, fixed = renderers()
    xyz = torch.tensor(seed(), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        xyz[0, 1] += torch.tensor([0.17, -0.31, 0.23], dtype=torch.float64)
    direction = torch.zeros_like(xyz)
    direction[0] = torch.tensor(
        [[0.13, -0.21, 0.07], [0.19, 0.11, -0.17], [-0.08, 0.16, 0.23]], dtype=torch.float64
    )

    def score(value):
        result = fixed.render(value)
        if term == "gap_distance":
            q = result.per_fish_mahalanobis_sq[..., 0]
            weights = torch.arange(1, 10, dtype=torch.float64).reshape(1, 3, 3) / 9
            return ((q.clamp_min(1).sqrt() - 1) * weights).sum()
        return (result.union_alpha.square() * 0.25).sum()

    value = score(xyz)
    gradient = torch.autograd.grad(value, xyz)[0]
    directional = (gradient * direction).sum().item()
    h = 1e-5
    fd = (
        (score(xyz.detach() + h * direction) - score(xyz.detach() - h * direction)) / (2 * h)
    ).item()
    assert np.isfinite(directional) and abs(directional) > 1e-8
    assert directional == pytest.approx(fd, rel=2e-5, abs=2e-8)


def test_reference_input_is_owned_and_output_batch_rows_cannot_change_later_reference():
    reference = seed()
    _, fixed = renderers(reference)
    expected = fixed.gaussians(seed()).radial_sigma_mm.detach().clone()
    reference[:] *= 3
    exact(fixed.gaussians(seed()).radial_sigma_mm, expected)
    output = fixed.gaussians(np.stack((seed(), seed()))).radial_sigma_mm
    exact(output, expected.expand(2, -1, -1))
    output[0] = 99
    exact(output[1], expected[0])
    exact(fixed.gaussians(seed()).radial_sigma_mm, expected)
    fixed.validate_fixed_width()


def test_strided_reference_and_public_reference_properties_are_independent_owned_copies():
    backing = np.zeros((3, 3, 6), np.float64)
    reference = backing[..., ::2]
    reference[:] = seed()
    assert not reference.flags.c_contiguous
    before = reference.tobytes()
    _, fixed = renderers(reference)
    signature = fixed.fixed_width_signature
    assert isinstance(signature, tuple) and isinstance(fixed.reference_signature, tuple)
    assert fixed.validate_fixed_width() == signature
    fetched_pose, fetched_radius = fixed.reference_pose_xyz, fixed.radial_sigma_mm
    np.testing.assert_array_equal(fetched_pose, reference)
    radius = fetched_radius.clone()
    fetched_pose[:] = -100
    fetched_radius[:] = 100
    assert reference.tobytes() == before
    assert fixed.reference_pose_xyz.tobytes() == before
    exact(fixed.radial_sigma_mm, radius)
    assert fixed.validate_fixed_width() == signature


def test_collapsed_current_segment_keeps_original_axial_floor_and_nondegenerate_flag():
    old, fixed = renderers()
    current = seed()
    current[0, 0] = current[0, 1]
    current[1] = current[1, 1]
    actual, dynamic = fixed.gaussians(current), old.gaussians(current)
    exact_fields(actual, dynamic, omit=("radial_sigma_mm",))
    exact(actual.radial_sigma_mm, fixed.gaussians(seed()).radial_sigma_mm)
    assert not actual.nondegenerate[0, 0] and not actual.nondegenerate[0, 1]
    assert torch.isfinite(fixed.render(current).union_alpha).all()


@pytest.mark.parametrize("method", ["gaussians", "render"])
@pytest.mark.parametrize("override", [0.06, 0.12])
def test_fixed_shape_cannot_receive_dynamic_width_override(method, override):
    _, fixed = renderers()
    with pytest.raises(ValueError):
        getattr(fixed, method)(seed(), width_ratio=override)


@pytest.mark.parametrize(
    "bad", ["nan", "inf", "shape", "float32", "collapsed_mouth", "collapsed_tail"]
)
def test_invalid_registered_reference_is_rejected(bad):
    reference = seed()
    if bad in ("nan", "inf"):
        reference[0, 0, 0] = np.nan if bad == "nan" else np.inf
    elif bad == "shape":
        reference = reference[:2]
    elif bad == "float32":
        reference = reference.astype(np.float32)
    elif bad == "collapsed_mouth":
        reference[0, 0] = reference[0, 1]
    else:
        reference[1, 2] = reference[1, 1]
    with pytest.raises(ValueError):
        renderers(reference)

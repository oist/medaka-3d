"""CPU numerical controls for fixed-path differentiable reflection responses."""

import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from reflection_renderer import ReflectionFishRenderer  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "reflection_renderer_numpy_test", LAB / "reflection_probe.py"
)
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


@dataclass
class Camera:
    view: str
    origins: np.ndarray
    directions: np.ndarray
    index_water: float = 1.3325
    index_glass: float = 1.52
    index_air: float = 1.0003

    def rays(self, uv):
        indices = np.asarray(uv[:, 0], int)
        return self.origins[indices], self.directions[indices]


@dataclass
class TestRig:
    __test__ = False
    cameras: tuple
    world_units: str = "mm"

    @property
    def views(self):
        return tuple(x.view for x in self.cameras)

    def camera(self, view):
        return next(x for x in self.cameras if x.view == view)


def scene():
    origins = np.array(
        [[70.7, 170, 100.6], [90.7, 170, 100.6], [110.7, 170, 100.6],
         [20, 20, 50], [193, 170, 100], [np.nan, np.nan, np.nan]], float
    )
    directions = np.tile([0.0, 1.0, 0.0], (6, 1))
    directions[3] = [0, 0, -1]
    cameras = tuple(
        Camera(v, origins + np.array([0, 0, dz]), directions.copy())
        for v, dz in zip(("top_cam", "side1_cam", "side2_cam"), (0, 0.25, -0.3), strict=True)
    )
    rig = TestRig(cameras)
    pixels = {v: np.column_stack((np.arange(6), np.zeros(6))) for v in rig.views}
    pixels[rig.views[0]] = np.vstack((pixels[rig.views[0]], [[np.nan, np.nan]]))
    pose = np.array(
        [[[x - 3, 176, 100], [x, 175, 100], [x + 10, 174, 100]] for x in (70, 90, 110)], float
    )
    return rig, pixels, pose


def numpy_comparison(engine, pose):
    b = probe.BodyArrays.from_torch(engine.gaussians(pose))
    o, d = engine._base.ray_origins.numpy(), engine._base.ray_directions.numpy()
    valid = engine.ray_valid.numpy()
    if engine.direct_mode == "unbounded":
        direct = probe.finite_response(b, o, d, np.full(valid.shape, np.inf), valid)
    else:
        direct = np.stack([
            probe.finite_response(b, o[v], d[v], trace.first_distance_mm, trace.direct_valid)
            for v, trace in enumerate(engine.paths)
        ])
    branches = []
    for name in ("inner", "outer"):
        branches.append(np.stack([
            probe.finite_response(b, x.origins, x.directions, x.max_distance_mm, x.valid)
            for x in (getattr(trace, name) for trace in engine.paths)
        ]))
    inner, outer = branches
    combined = probe.combine_branches(
        direct, inner, outer, engine.inner_weight.numpy(), engine.outer_weight.numpy()
    )
    return dict(direct_alpha=direct, inner_alpha=inner, outer_alpha=outer,
                direct_union=combined["finite_direct"], unit_union=combined["both_unit"],
                fresnel_union=combined["both_fresnel"])


def test_original_unbounded_operation_order_exact(rig):
    _, _, pose = scene()
    rng = np.random.default_rng(820)
    pixels = {v: np.vstack((rig.camera(v).project(pose.reshape(-1, 3)),
                            rng.uniform(0, 1200, (23, 2)), [[np.nan, 0]])) for v in rig.views}
    config = RendererConfig(pixel_chunk_size=7)
    old = RefractiveFishRenderer(rig, pixels, config, dtype=torch.float64).render(pose)
    new = ReflectionFishRenderer(rig, pixels, config, direct_mode="unbounded").render(pose)
    assert torch.equal(old.per_fish_alpha, new.direct_alpha)
    assert torch.equal(old.union_alpha, new.direct_union)
    for field in old.bodies.__dataclass_fields__:
        assert torch.equal(getattr(old.bodies, field), getattr(new.bodies, field))


@pytest.mark.parametrize("mode", ["finite", "unbounded"])
def test_all_branches_match_independent_numpy(mode):
    rig, pixels, pose = scene()
    engine = ReflectionFishRenderer(
        rig, pixels, RendererConfig(pixel_chunk_size=2), direct_mode=mode
    )
    result, expected = engine.render(pose), numpy_comparison(engine, pose)
    for field, value in expected.items():
        np.testing.assert_allclose(getattr(result, field)[0].numpy(), value, rtol=0, atol=1e-12)
    assert float(result.inner_alpha.max()) > 0.1
    assert float(result.outer_alpha.max()) > 0.1
    assert float(result.inner_weight.max()) > 0
    assert float(result.outer_weight.max()) > 0
    assert torch.all(result.direct_union <= result.fresnel_union + 1e-15)
    assert torch.all(result.fresnel_union <= result.unit_union + 1e-15)


@pytest.mark.parametrize("mode", ["finite", "unbounded"])
def test_pose_autograd_matches_central_difference(mode):
    rig, pixels, pose = scene()
    engine = ReflectionFishRenderer(rig, pixels, direct_mode=mode)
    value = torch.tensor(pose, dtype=torch.float64, requires_grad=True)
    weights = torch.arange(21, dtype=torch.float64).reshape(1, 3, 7) / 21 + 0.2

    def loss(x):
        result = engine.render(x)
        return ((result.unit_union + 0.37 * result.fresnel_union) * weights).sum()

    analytic = torch.autograd.grad(loss(value), value)[0].numpy()
    assert np.linalg.norm(analytic) > 0.01
    finite = np.zeros_like(pose)
    h = 1e-5
    for index in np.ndindex(pose.shape):
        plus, minus = pose.copy(), pose.copy()
        plus[index] += h
        minus[index] -= h
        finite[index] = (float(loss(plus)) - float(loss(minus))) / (2 * h)
    np.testing.assert_allclose(analytic, finite, rtol=2e-6, atol=2e-8)


def test_translated_body_gradient_against_closed_form():
    rig, pixels, pose = scene()
    pose[:, :, 1] = 175
    engine = ReflectionFishRenderer(rig, pixels, direct_mode="finite")
    translation = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    result = engine.render(torch.tensor(pose) + translation * direction)
    actual = torch.autograd.grad(result.direct_union[0, 0, 0], translation)[0].item()
    b = probe.BodyArrays.from_torch(engine.gaussians(pose))
    dx = b.centers[..., 0] - rig.cameras[0].origins[0, 0]
    dz = b.centers[..., 2] - rig.cameras[0].origins[0, 2]
    alpha = 0.85 * np.exp(-0.5 * (dx**2 / b.axial_sigma_mm**2 + dz**2 / b.radial_sigma_mm**2))
    derivative = -alpha * dz / b.radial_sigma_mm**2
    expected = np.prod(1 - alpha) * np.sum(derivative / (1 - alpha))
    assert abs(expected) > 0.01
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_finite_endpoint_differs_from_unbounded_and_has_gradient():
    rig, pixels, pose = scene()
    pose[:, :, 1] = 194
    finite = ReflectionFishRenderer(rig, pixels, direct_mode="finite")
    unbounded = ReflectionFishRenderer(rig, pixels, direct_mode="unbounded")
    value = torch.tensor(pose, requires_grad=True)
    a, b = finite.render(value), unbounded.render(value)
    assert torch.all(a.direct_alpha[..., :3, :] < b.direct_alpha[..., :3, :] + 1e-15)
    assert float((b.direct_alpha - a.direct_alpha).max().detach()) > 0.1
    grad = torch.autograd.grad(a.direct_union.sum(), value)[0]
    assert torch.isfinite(grad).all() and float(grad.norm()) > 0.01
    np.testing.assert_allclose(
        a.direct_alpha[0].detach(), numpy_comparison(finite, pose)["direct_alpha"],
        rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("mode", ["finite", "unbounded"])
def test_invalid_paths_and_missing_views_neutral_no_nan_gradient(mode):
    rig, pixels, pose = scene()
    pixels = {rig.views[0]: np.array([[5, 0], [np.nan, 0]])}
    engine = ReflectionFishRenderer(rig, pixels, direct_mode=mode)
    value = torch.tensor(pose, requires_grad=True)
    result = engine.render(value)
    assert not result.ray_valid.any()
    for field in ("direct_alpha", "inner_alpha", "outer_alpha", "unit_union", "fresnel_union"):
        assert torch.equal(getattr(result, field), torch.zeros_like(getattr(result, field)))
    grad = torch.autograd.grad(result.unit_union.sum(), value)[0]
    assert torch.equal(grad, torch.zeros_like(grad))
    for segment in (engine.finite_direct, engine.inner, engine.outer):
        for field in segment.__dataclass_fields__:
            tensor = getattr(segment, field)
            assert not tensor.requires_grad and torch.isfinite(tensor).all()


def test_physical_invalid_keeps_calibrated_denominator():
    rig, pixels, pose = scene()
    engine = ReflectionFishRenderer(rig, pixels, direct_mode="finite")
    result = engine.render(pose)
    assert result.ray_valid[:, 4].all()  # finite source ray outside the nominal physical box
    assert not result.direct_valid[:, 4].any()
    assert torch.equal(result.unit_union[:, :, 4], torch.zeros((1, 3), dtype=torch.float64))


def test_batch_fish_view_order_and_chunk_size():
    rig, pixels, pose = scene()
    engine = ReflectionFishRenderer(rig, pixels, direct_mode="finite")
    other = pose + np.array([0.2, -0.3, 0.1])
    batch = engine.render(np.stack((pose, other)))
    for i, x in enumerate((pose, other)):
        separate = engine.render(x)
        for field in ("direct_alpha", "inner_alpha", "outer_alpha", "unit_union", "fresnel_union"):
            torch.testing.assert_close(getattr(batch, field)[i], getattr(separate, field)[0],
                                       rtol=0, atol=1e-15)
    order = [2, 0, 1]
    shuffled = engine.render(pose[order])
    for field in ("direct_alpha", "inner_alpha", "outer_alpha"):
        torch.testing.assert_close(getattr(shuffled, field), getattr(batch, field)[:1, ..., order],
                                   rtol=0, atol=1e-15)
    torch.testing.assert_close(shuffled.unit_union, batch.unit_union[:1], rtol=0, atol=1e-15)
    reordered = replace(rig, cameras=tuple(rig.cameras[i] for i in order))
    changed = ReflectionFishRenderer(reordered, pixels, RendererConfig(pixel_chunk_size=1),
                                    direct_mode="finite").render(pose)
    torch.testing.assert_close(changed.unit_union, batch.unit_union[:1, order], rtol=0, atol=1e-15)


@pytest.mark.parametrize("kwargs,match", [
    ({"direct_mode": "unknown"}, "direct_mode"),
    ({"direct_mode": "finite", "dtype": torch.float32}, "float64"),
    ({"direct_mode": "finite", "config": RendererConfig(gaussians_per_fish=12)}, "eight"),
])
def test_reject_unregistered_modes(kwargs, match):
    rig, pixels, _ = scene()
    with pytest.raises(ValueError, match=match):
        ReflectionFishRenderer(rig, pixels, **kwargs)

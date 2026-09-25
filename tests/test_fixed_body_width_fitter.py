"""New fixed-width fit wiring on actual tiny CPU tensors, without saved inputs."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "work/three_body"))
import fixed_body_width_fitter as m  # noqa: E402
import mask_transport_fitter as old_owner  # noqa: E402
import mask_transport_gap_fitter as old_gap  # noqa: E402
from fixed_body_width_renderer import FixedWidthRenderer  # noqa: E402
from node_geometry import RayObservations  # noqa: E402
from observations import PixelTargets  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402
from tracker import Proposal, ThreeBodyTracker, TrackerConfig  # noqa: E402

from test_fixed_body_width_renderer import exact, scene, seed  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_only_no_dynamic_fit(monkeypatch):
    before = torch.get_num_threads()
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "post-t0 fixed-width fitter cannot call a dynamic fitter or CUDA graph"
        )

    monkeypatch.setattr(old_owner, "frozen_fit", forbidden)
    monkeypatch.setattr(old_gap, "frozen_fit", forbidden)
    monkeypatch.setattr(ThreeBodyTracker, "_fit", forbidden)
    monkeypatch.setattr(m, "FixedWidthGraphFit", forbidden)
    yield
    torch.set_num_threads(before)


def fit_fixture(mode="active", *, learning_rate=0.02):
    rig, _ = scene()
    reference = seed()
    pixels, point_rows, origins, directions = {}, {}, [], []
    for view, axis in zip(rig.views, (2, 0, 1), strict=True):
        other = [a for a in range(3) if a != axis]
        point_rows[view] = reference[..., other].copy()
        pixels[view] = np.vstack((reference[0, [1, 0, 2]][:, other], [35.0, 35.0]))
        ray_o, ray_d = reference.copy(), np.zeros_like(reference)
        ray_o[..., axis], ray_d[..., axis] = 0, 1
        origins.append(ray_o)
        directions.append(ray_d)
    target_array = np.zeros((2, 3, 4, 3), np.float64)
    target_array[0, :, :3, 0] = [0.5, 0.75, 1.0]
    target_array[1, :, :3, 1] = [0.5, 0.75, 1.0]
    if mode == "zero_effective":
        target_array[:] = 0
    elif mode == "zero_valid":
        for xy in pixels.values():
            xy[:3] = np.nan
    foreground = np.broadcast_to([1.0, 1.0, 1.0, 0.0], (3, 4)).copy()
    targets = PixelTargets(pixels, foreground, np.full((3, 4), 0.25), foreground.astype(bool))
    config = TrackerConfig(device="cpu", learning_rate_mm=learning_rate)
    tracker = SimpleNamespace(
        rig=rig,
        config=config,
        device=torch.device("cpu"),
        dtype=torch.float64,
        lengths=np.linalg.norm(np.diff(reference, axis=1), axis=-1),
        lower=np.full((3, 3, 3), -100.0),
        upper=np.full((3, 3, 3), 100.0),
        _fixed_width_reference_pose=reference.copy(),
        _tensor=lambda value: torch.as_tensor(value, dtype=torch.float64),
    )
    checked = []

    def check_reference(renderer):
        assert type(renderer) is FixedWidthRenderer
        assert renderer.reference_pose_xyz.tobytes() == reference.tobytes()
        checked.append(renderer.validate_fixed_width())

    tracker.validate_fixed_width_renderer = check_reference
    tracker.reference_checks = checked
    tracker._fixed_width_reference_pose.setflags(write=False)
    poses = [
        (reference - reference[:, 1:2]) * 1.35 + reference[:, 1:2],
        (reference - reference[:, 1:2]) * 1.1 + reference[:, 1:2] + [0.2, -0.1, 0.05],
    ]
    plan = np.broadcast_to(np.arange(3, dtype=np.int32)[:, None, None], (3, 3, 3)).copy()
    proposals = [
        Proposal(pose.copy(), reference.copy(), plan.copy(), parent, "synthetic")
        for pose, parent in zip(poses, (1, 0), strict=True)
    ]
    states = [SimpleNamespace(missing=np.zeros(3)), SimpleNamespace(missing=np.array([1, 2, 3]))]
    rays = RayObservations(point_rows, tuple(origins), tuple(directions))
    coefficient = 0.0 if mode == "zero_coefficient" else 3.0
    target = object() if coefficient == 0 else target_array
    return tracker, proposals, states, rays, targets, target, coefficient


def snapshot(args):
    tracker, proposals, states, rays, targets, target, _ = args
    arrays = [tracker._fixed_width_reference_pose, tracker.lengths, targets.target, targets.weights]
    arrays.extend(targets.pixels.values())
    arrays.extend(rays.points.values())
    arrays.extend(rays.origins)
    arrays.extend(rays.directions)
    arrays.extend(s.missing for s in states)
    arrays.extend(a for p in proposals for a in (p.pose, p.predicted, p.plan))
    if isinstance(target, np.ndarray):
        arrays.append(target)
    return [(a.dtype.str, a.shape, a.tobytes()) for a in arrays]


def fit(args):
    return m.fit_with_fixed_width(*args[:6], coefficient=args[6])


def spy_objective(monkeypatch):
    original = m.objective
    calls = []

    def call(pose, cfg, inputs, denominator, renderer=None, **kwargs):
        assert type(renderer) is FixedWidthRenderer
        result = original(pose, cfg, inputs, denominator, renderer, **kwargs)
        calls.append(
            dict(
                pose=pose.detach().clone(),
                inputs=inputs,
                renderer=renderer,
                denominator=denominator,
                result=result,
                diagnostics=kwargs.get("diagnostics", False),
            )
        )
        return result

    monkeypatch.setattr(m, "objective", call)
    return calls, original


@pytest.mark.parametrize("mode", ["zero_coefficient", "zero_effective", "zero_valid", "active"])
def test_actual_cpu_all_modes_fit_fixed_base_keep_full_iteration_history_and_final_support(
    monkeypatch, mode
):
    args = fit_fixture(mode)
    before = snapshot(args)
    calls, objective = spy_objective(monkeypatch)
    result = fit(args)
    tracker = args[0]
    d = tracker.last_gap_fit
    assert len(tracker.reference_checks) == 1
    assert snapshot(args) == before
    assert len(calls) == 16 and sum(c["diagnostics"] for c in calls) == 1
    assert len({id(c["renderer"]) for c in calls}) == 1
    renderer = calls[0]["renderer"]
    for c in calls:
        radius = renderer.gaussians(c["pose"]).radial_sigma_mm
        exact(radius[0], renderer.radial_sigma_mm)
        exact(radius[1], renderer.radial_sigma_mm)
    assert all(
        v.shape == (15, 2) for k, v in d.items() if k.startswith("iteration_") and "norm" not in k
    )
    assert (
        d["iteration_owner_gradient_norm"].shape
        == d["iteration_gap_gradient_norm"].shape
        == (14, 2)
    )
    np.testing.assert_array_equal(
        d["best_iteration"], np.argmin(d["iteration_total_score"], axis=0)
    )
    assert (d["best_iteration"] >= 0).all()
    np.testing.assert_array_equal(
        d["iteration_total_score"],
        d["iteration_base_score"] + args[6] * (d["iteration_owner_loss"] + d["iteration_gap_loss"]),
    )
    np.testing.assert_array_equal(
        result[1], d["base_score"] + args[6] * (d["owner_loss"] + d["gap_loss"])
    )
    np.testing.assert_array_equal(result[2], d["original_loss_parts"])
    np.testing.assert_allclose(
        result[1], d["iteration_total_score"][d["best_iteration"], np.arange(2)], rtol=0, atol=1e-10
    )
    expected_pose = np.stack(
        [calls[int(d["best_iteration"][b])]["pose"][b].numpy() for b in range(2)]
    )
    assert result[0].tobytes() == expected_pose.tobytes()
    assert d["active"] is (mode == "active")
    reasons = dict(
        active="fixed_radial_owner_plus_gap",
        zero_coefficient="fixed_radial_coefficient_zero",
        zero_effective="fixed_radial_zero_effective_target",
        zero_valid="fixed_radial_zero_valid_ray_target",
    )
    assert d["reason"] == reasons[mode]
    if mode == "active":
        assert d["target_diagnostics_computed"] is True
        assert d["target_mass"].shape == (2,) and np.all(d["target_mass"] > 0)
        np.testing.assert_allclose(
            d["final_gap_by_view_body"].sum((1, 2)), d["gap_loss"], rtol=0, atol=1e-12
        )
    else:
        assert d["target_diagnostics_computed"] is False
        assert d["target_mass"].shape == (0,)
        assert d["positive_gap_pixel_count"].shape == (0, 3, 3)
        for name in ("owner", "gap"):
            assert not d[f"{name}_loss"].any()
            assert not d[f"iteration_{name}_loss"].any()
            assert not d[f"iteration_{name}_gradient_norm"].any()
    # Reconstruct final support from the SAME fixed renderer's visible result.
    final = calls[-1]
    render = final["result"][2]
    fg, w = final["inputs"].target, final["inputs"].weight
    pos = (render.visible_alpha * (render.ownership >= 0.65) * (w * fg)[None, :, :, None]).sum(2)
    neg = (render.visible_alpha * (w * (1 - fg))[None, :, :, None]).sum(2)
    supported = ((pos >= 0.025) & (pos >= 2 * neg)).sum(1) >= 2
    np.testing.assert_array_equal(result[3], supported.numpy())
    if mode != "zero_valid":
        # Stretched proposals make dynamic and fixed base union numerically distinct.
        dynamic = RefractiveFishRenderer(
            tracker.rig, args[4].pixels, RendererConfig(width_ratio=0.06), dtype=torch.float64
        )
        c = calls[0]
        old_score = objective(c["pose"], tracker.config, c["inputs"], c["denominator"], dynamic)[0]
        assert not torch.allclose(old_score, c["result"][0], atol=1e-12, rtol=0)


def test_every_active_adam_gradient_equals_joint_scalar_autograd_and_reference_stays_owned(
    monkeypatch,
):
    args = fit_fixture()
    calls, objective = spy_objective(monkeypatch)
    original_adam, gradients = torch.optim.Adam, []
    target = torch.tensor(args[5], dtype=torch.float64)

    class CheckedAdam(original_adam):
        def step(self, *step_args, **step_kwargs):
            pose = self.param_groups[0]["params"][0]
            c = calls[-1]
            independent = pose.detach().clone().requires_grad_(True)
            base = objective(
                independent, args[0].config, c["inputs"], c["denominator"], c["renderer"]
            )[0]
            rendering = c["renderer"].render(independent)
            mass = (c["inputs"].weight * c["inputs"].target)[None, :, :, None] * target
            owner = (mass * (1 - rendering.ownership).square()).sum((1, 2, 3)) / 3
            valid = rendering.ray_valid[None, :, :, None]
            q = torch.where(valid, rendering.per_fish_mahalanobis_sq, 0)
            gap = (mass * (q.clamp_min(1).sqrt() - 1)).sum((1, 2, 3)) / 3
            expected = torch.autograd.grad((base + 3 * (owner + gap)).sum(), independent)[0]
            assert torch.isfinite(pose.grad).all()
            torch.testing.assert_close(pose.grad, expected, rtol=0, atol=2e-13)
            gradients.append(pose.grad.detach().clone())
            return super().step(*step_args, **step_kwargs)

    monkeypatch.setattr(m.torch.optim, "Adam", CheckedAdam)
    before = snapshot(args)
    fit(args)
    assert len(gradients) == 14 and snapshot(args) == before
    assert any(torch.count_nonzero(g) for g in gradients)


def test_unchanged_pose_tie_selects_iteration_zero_with_fourteen_real_adam_steps():
    args = fit_fixture("zero_coefficient", learning_rate=1e-300)
    expected = np.stack([p.pose for p in args[1]])
    result = fit(args)
    assert result[0].tobytes() == expected.tobytes()
    np.testing.assert_array_equal(args[0].last_gap_fit["best_iteration"], [0, 0])
    assert args[0].last_gap_fit["iteration_total_score"].shape == (15, 2)


@pytest.mark.parametrize("failure", ["nonfinite_score", "nonfinite_gradient"])
def test_nonfinite_fit_fails_before_new_diagnostic_can_replace_prior_record(monkeypatch, failure):
    args = fit_fixture("zero_coefficient")
    sentinel = {"prior": "unchanged"}
    args[0].last_gap_fit = sentinel
    original = m.objective

    class BadBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, gradient):
            return torch.full_like(gradient, torch.nan)

    def bad(*values, **kwargs):
        score, parts, rendering = original(*values, **kwargs)
        score = score + torch.inf if failure == "nonfinite_score" else BadBackward.apply(score)
        return score, parts, rendering

    monkeypatch.setattr(m, "objective", bad)
    with pytest.raises(FloatingPointError, match=r"objective|gradient"):
        fit(args)
    assert args[0].last_gap_fit is sentinel


@pytest.mark.parametrize("coefficient", [True, 1.0, -3.0])
def test_unregistered_coefficient_rejected_before_renderer(monkeypatch, coefficient):
    args = fit_fixture()

    def forbidden(*values, **kwargs):
        raise AssertionError("invalid coefficient reached renderer")

    monkeypatch.setattr(m, "FixedWidthRenderer", forbidden)
    with pytest.raises(ValueError, match="coefficient"):
        m.fit_with_fixed_width(*args[:6], coefficient=coefficient)


def test_no_pixel_targets_rejected_instead_of_dynamic_fitter_fallback():
    args = list(fit_fixture("zero_coefficient"))
    args[4] = None
    with pytest.raises(ValueError, match="PixelTargets"):
        fit(args)


def test_tracker_reference_rejection_precedes_any_objective_or_diagnostic_write(monkeypatch):
    args = fit_fixture("zero_coefficient")
    error = ValueError("registered seed binding rejected")
    prior = {"old": "diagnostic"}
    args[0].last_gap_fit = prior
    calls = []

    def reject(renderer):
        assert type(renderer) is FixedWidthRenderer
        calls.append(renderer.reference_signature)
        raise error

    def forbidden(*values, **kwargs):
        raise AssertionError("rejected reference reached objective")

    args[0].validate_fixed_width_renderer = reject
    monkeypatch.setattr(m, "objective", forbidden)
    with pytest.raises(ValueError) as caught:
        fit(args)
    assert caught.value is error and len(calls) == 1
    assert args[0].last_gap_fit is prior

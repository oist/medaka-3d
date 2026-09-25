"""New gap math/fit wiring on synthetic CPU inputs only; no actual observations."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "work/three_body"))
import mask_transport_fitter as old  # noqa: E402
import mask_transport_gap_fitter as m  # noqa: E402

from medaka3d.calib import Rig  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def values():
    q = torch.full((2, 3, 2, 3), 9.0, dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(q)
    target[:, 0, 0, 0] = 0.5
    target[:, 1, 1, 2] = 1
    foreground = torch.ones((3, 2), dtype=torch.float64)
    weight = torch.full((3, 2), 0.5, dtype=torch.float64)
    valid = torch.ones((3, 2), dtype=torch.bool)
    valid[1, 1] = False
    with torch.no_grad():
        q[:, 1, 1, :] = torch.inf
    return q, target, foreground, weight, valid


def test_hand_gap_derivative_mass_and_invalid_infinity_are_kept_separate():
    args = values()
    before = [v.detach().numpy().tobytes() for v in args]
    score, detail = m.ray_gap_loss(*args, return_details=True)
    torch.testing.assert_close(score, torch.full((2,), 1 / 6, dtype=torch.float64))
    score.sum().backward()
    expected = torch.zeros_like(args[0])
    expected[:, 0, 0, 0] = 1 / 72
    torch.testing.assert_close(args[0].grad, expected, rtol=0, atol=1e-17)
    assert detail["positive_gap_pixel_count"].sum().item() == 2
    torch.testing.assert_close(detail["target_mass_by_view_body"].sum((1, 2)), score / 2)
    torch.testing.assert_close(detail["final_gap_by_view_body"].sum((1, 2)), score)
    assert [v.detach().numpy().tobytes() for v in args] == before


@pytest.mark.parametrize("invalid", [float("inf"), float("nan"), -0.01])
def test_all_valid_q_checked_even_without_target(invalid):
    args = values()
    with torch.no_grad():
        args[0][:, 2, 1, 1] = invalid
    assert not args[1][:, 2, 1, 1].any()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        m.ray_gap_loss(*args)


@pytest.mark.parametrize("qvalue,expected_derivative", [(0.0, 0.0), (0.5, 0.0), (1.0, 0.5)])
def test_q_one_boundary_uses_original_clamp_min_derivative(qvalue, expected_derivative):
    q, target, fg, weight, valid = values()
    with torch.no_grad():
        q[:, 0, 0, 0] = qvalue
    score = m.ray_gap_loss(q, target, fg, weight, valid)
    assert torch.equal(score, torch.zeros_like(score))
    score.sum().backward()
    torch.testing.assert_close(
        q.grad[:, 0, 0, 0], torch.full((2,), expected_derivative / 12, dtype=torch.float64)
    )


def test_equal_nearest_gaussians_keep_amin_tie_derivative_and_mass_scaling():
    squared = torch.tensor([4.0, 4.0], dtype=torch.float64, requires_grad=True)
    q = squared.amin().expand(1, 3, 1, 3)
    target = torch.zeros_like(q)
    target[0, 0, 0, 0] = 1
    fg = torch.ones((3, 1), dtype=torch.float64)
    valid = torch.ones((3, 1), dtype=torch.bool)
    score = m.ray_gap_loss(q, target, fg, fg, valid)
    score.sum().backward()
    torch.testing.assert_close(squared.grad, torch.tensor([1 / 24, 1 / 24], dtype=torch.float64))
    scaled = m.ray_gap_loss(q, target * 0.125, fg, fg, valid)
    torch.testing.assert_close(scaled, score * 0.125)


class ParallelCamera:
    def __init__(self, view):
        self.view = view

    def rays(self, xy):
        origins = np.column_stack([xy, np.zeros(len(xy))])
        return origins, np.broadcast_to([0, 0, 1], origins.shape).copy()


def test_true_far_ray_pose_gradient_survives_zero_alpha_and_matches_rigid_shift_difference():
    rig = Rig(tuple(ParallelCamera(v) for v in VIEWS), Path("synthetic-only"))
    renderer = m.RefractiveFishRenderer(
        rig, {v: np.array([[0.0, 0.0]]) for v in VIEWS}, dtype=torch.float64
    )
    pose = torch.tensor(
        [
            [[105.0, 30.0, 25.0], [100.0, 30.0, 25.0], [88.0, 32.0, 26.0]],
            [[205.0, 30.0, 35.0], [200.0, 30.0, 35.0], [188.0, 32.0, 36.0]],
            [[305.0, 30.0, 45.0], [300.0, 30.0, 45.0], [288.0, 32.0, 46.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    result = renderer.render(pose)
    assert result.per_fish_alpha.max().item() == 0
    assert result.ownership.max().item() == 0
    target = torch.zeros_like(result.per_fish_mahalanobis_sq)
    target[..., 0] = 1
    foreground = torch.ones((3, 1), dtype=torch.float64)

    def score(value):
        q = renderer.render(value).per_fish_mahalanobis_sq
        return m.ray_gap_loss(q, target, foreground, foreground, renderer.ray_valid).sum()

    loss = score(pose)
    gradient = torch.autograd.grad(loss, pose)[0]
    rigid_x = gradient[0, :, 0].sum().item()
    assert loss.item() > 1 and np.isfinite(rigid_x) and rigid_x > 0.1
    shift = torch.zeros_like(pose)
    shift[0, :, 0] = 1
    h = 1e-4
    fd = ((score(pose.detach() + h * shift) - score(pose.detach() - h * shift)) / (2 * h)).item()
    assert rigid_x == pytest.approx(fd, abs=1e-8, rel=1e-7)
    assert score(pose.detach() - h * shift) < loss
    assert torch.count_nonzero(gradient[1:]) == 0


@pytest.mark.parametrize("bad", ["multiple_bodies", "target_gradient", "dtype"])
def test_target_contract_cannot_change_the_fitted_evidence(bad):
    args = list(values())
    if bad == "multiple_bodies":
        args[1][:, 0, 0, 1] = 0.25
    elif bad == "target_gradient":
        args[1].requires_grad_()
    else:
        args[1] = args[1].float()
    with pytest.raises(ValueError):
        m.ray_gap_loss(*args)


@pytest.mark.parametrize("mode", ["zero_coefficient", "zero_mass", "exception"])
def test_frozen_dispatch_precedes_target_render_and_does_not_overwrite_owner(monkeypatch, mode):
    assert m.ownership_loss is old.ownership_loss and m.frozen_fit is old.frozen_fit
    tracker = SimpleNamespace(last_owner_fit="unchanged")
    proposals, states, rays = [object()], object(), object()
    targets = SimpleNamespace(target=np.ones((3, 1)), weights=np.ones((3, 1)))
    target = np.zeros((1, 3, 1, 3), np.float64) if mode == "zero_mass" else object()
    expected = (np.zeros((1, 3, 3, 3)), np.array([-0.0]), np.ones((1, 5)), np.ones((1, 3), bool))
    error = RuntimeError("frozen failure")
    calls = []

    def frozen(*args):
        assert all(
            a is b for a, b in zip(args, (tracker, proposals, states, rays, targets), strict=True)
        )
        calls.append(True)
        if mode == "exception":
            raise error
        return expected

    def forbidden(*args, **kwargs):
        raise AssertionError("extra renderer/distance work")

    monkeypatch.setattr(m, "frozen_fit", frozen)
    for name in ("assigned_rays", "GraphFit", "RefractiveFishRenderer", "ray_gap_loss"):
        monkeypatch.setattr(m, name, forbidden)
    coeff = 3 if mode == "zero_mass" else 0
    if mode == "exception":
        with pytest.raises(RuntimeError) as caught:
            m.fit_with_gap(tracker, proposals, states, rays, targets, target, coefficient=coeff)
        assert caught.value is error
    else:
        result = m.fit_with_gap(
            tracker, proposals, states, rays, targets, target, coefficient=coeff
        )
        assert result is expected
        detail = tracker.last_gap_fit
        assert detail["total_score"].tobytes() == expected[1].tobytes()
        assert detail["gap_loss"].shape == (1,) and not detail["gap_loss"].any()
        assert not detail["target_diagnostics_computed"]
        assert detail["final_gap_by_view_body"].shape == (0, 3, 3)
        assert detail["target_mass"].shape == (0,)
    assert calls == [True] and tracker.last_owner_fit == "unchanged"


def test_fourteen_updates_combine_distinct_gradients_and_save_final_best_decomposition(monkeypatch):
    """Controlled smooth CPU objective checks new assembly, not frozen optimizer quality."""
    cfg = SimpleNamespace(motion_sigma_mm=1.0, width_ratio=0.06, learning_rate_mm=0.1, opt_steps=14)
    tracker = SimpleNamespace(
        config=cfg,
        device=torch.device("cpu"),
        dtype=torch.float64,
        rig=None,
        lengths=np.ones((3, 2)),
        lower=np.full((3, 3, 3), -20.0),
        upper=np.full((3, 3, 3), 20.0),
        _tensor=lambda v: torch.as_tensor(v, dtype=torch.float64),
    )
    pose = np.full((3, 3, 3), 0.5)
    proposals = [SimpleNamespace(plan=np.full((3, 3, 3), -1), pose=pose, predicted=pose, parent=0)]
    states = [SimpleNamespace(missing=np.zeros(3))]
    pixels = {v: np.zeros((2, 2)) for v in VIEWS}
    targets = SimpleNamespace(target=np.ones((3, 2)), weights=np.full((3, 2), 0.5), pixels=pixels)
    target = np.zeros((1, 3, 2, 3))
    target[..., 0] = 0.5
    monkeypatch.setattr(
        m,
        "assigned_rays",
        lambda *a: (np.zeros((1, 3, 3, 3, 3)), np.ones((1, 3, 3, 3, 3)), np.zeros((1, 3, 3, 3))),
    )

    class Renderer:
        ray_valid = torch.ones((3, 2), dtype=torch.bool)
        n_pixels = 2

        def __init__(self, *args, **kwargs):
            pass

        def render(self, xyz):
            x = xyz[:, :, 0, 0][:, None, None, :].expand(-1, 3, 2, -1)
            own = torch.sigmoid(x)
            return SimpleNamespace(
                ownership=own, visible_alpha=own, per_fish_mahalanobis_sq=(x - 4).square() + 2
            )

    def objective(xyz, cfg, inputs, denominator, renderer, **kwargs):
        score = (xyz + 1).square().sum((1, 2, 3))
        return score, score[:, None].expand(-1, 5).clone(), renderer.render(xyz)

    monkeypatch.setattr(m, "RefractiveFishRenderer", Renderer)
    monkeypatch.setattr(m, "objective", objective)
    snapshots = []
    real_adam = torch.optim.Adam

    class CheckedAdam(real_adam):
        def step(self, *args, **kwargs):
            p = self.param_groups[0]["params"][0]
            expected = 2 * (p.detach() + 1)
            x = p.detach()[0, 0, 0, 0]
            sigmoid = torch.sigmoid(x)
            owner_derivative = -(1 - sigmoid) * sigmoid * (1 - sigmoid)
            gap_derivative = 0.5 * (x - 4) / torch.sqrt((x - 4).square() + 2)
            expected[0, 0, 0, 0] += 3 * (owner_derivative + gap_derivative)
            torch.testing.assert_close(p.grad, expected, rtol=0, atol=1e-14)
            snapshots.append(p.detach().numpy().copy())
            return super().step(*args, **kwargs)

    monkeypatch.setattr(m.torch.optim, "Adam", CheckedAdam)
    result = m.fit_with_gap(tracker, proposals, states, None, targets, target)
    d = tracker.last_gap_fit
    assert len(snapshots) == 14
    assert d["iteration_gap_loss"].shape == d["iteration_owner_loss"].shape == (15, 1)
    assert d["iteration_gap_gradient_norm"].shape == (14, 1)
    np.testing.assert_array_equal(
        d["iteration_total_score"],
        d["iteration_base_score"] + 3 * (d["iteration_owner_loss"] + d["iteration_gap_loss"]),
    )
    np.testing.assert_array_equal(
        result[1], d["base_score"] + 3 * (d["owner_loss"] + d["gap_loss"])
    )
    assert int(d["best_iteration"][0]) == int(np.argmin(d["iteration_total_score"][:, 0]))
    np.testing.assert_allclose(
        result[1], d["iteration_total_score"][d["best_iteration"], [0]], rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        d["final_gap_by_view_body"].sum((1, 2)), d["gap_loss"], rtol=0, atol=1e-14
    )
    assert d["positive_gap_pixel_count"].sum() == 6
    assert not hasattr(tracker, "last_owner_fit")
    assert np.array_equal(proposals[0].pose, pose) and np.all(target[..., 0] == 0.5)

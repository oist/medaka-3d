"""Bounded curve constraints and optimizer selection; fake scalar images, CPU only."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_curved_body_probe as m


def pose():
    p = torch.tensor(
        [[40.0, 40.0, 30.0], [45.0, 40.0, 30.0], [55.0, 40.0, 30.0]], dtype=torch.float64
    )
    return p[None].repeat(3, 1, 1)


def test_perpendicular_chord_limit_and_box_use_one_scalar():
    p = pose()
    p[0] = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [3.0, 3.0, 0.0]])
    proposed = torch.tensor(
        [[10.0, -10.0, 100.0], [4.0, 200.0, 20.0], [4.0, 1.0, 2.0]], dtype=torch.float64
    )
    original = p.clone()
    bounded, action = m.constrain_offsets(p, proposed)
    chord = p[:, 2] - p[:, 1]
    np.testing.assert_allclose((chord * bounded).sum(-1), 0, atol=1e-13)
    assert torch.all(
        torch.linalg.vector_norm(bounded, dim=-1) <= torch.linalg.vector_norm(chord, dim=-1)
    )
    control = (p[:, 1] + p[:, 2]) / 2 + bounded
    assert torch.all(control >= torch.tensor(m.LOWER))
    assert torch.all(control <= torch.tensor(m.UPPER))
    assert torch.equal(p, original)
    direction = chord / torch.linalg.vector_norm(chord, dim=-1)[:, None]
    planar = proposed - (proposed * direction).sum(-1)[:, None] * direction
    assert torch.equal(bounded, planar * action["applied_scalar"][:, None])
    assert bool(action["norm_constraint_active"][1])


def test_box_limit_acts_without_componentwise_clipping():
    p = pose()
    p[:, :, 1] = 1
    proposed = torch.tensor([[0.0, -10.0, 3.0]] * 3, dtype=torch.float64)
    bounded, action = m.constrain_offsets(p, proposed)
    np.testing.assert_allclose(bounded.numpy(), [[0.0, -1.0, 0.3]] * 3, atol=1e-15)
    assert torch.all(action["box_constraint_active"])
    assert torch.all(action["applied_scalar"] == 0.1)


def test_degenerate_chord_permits_only_zero_offset():
    p = pose()
    p[:, 2] = p[:, 1]
    bounded, action = m.constrain_offsets(p, torch.ones((3, 3), dtype=torch.float64))
    assert torch.equal(bounded, torch.zeros_like(bounded))
    assert torch.all(action["collapsed_chord"])


@pytest.mark.parametrize("kind", ("nan", "outside"))
def test_invalid_constraint_input_fails(kind):
    p = pose()
    p[0, 0, 0] = float("nan") if kind == "nan" else -1
    with pytest.raises(ValueError):
        m.constrain_offsets(p, torch.zeros((3, 3), dtype=torch.float64))


class ScalarRenderer:
    device = torch.device("cpu")
    dtype = torch.float64

    def __init__(self, nonfinite_after=None):
        self.calls = 0
        self.nonfinite_after = nonfinite_after

    def render(self, pose_xyz, *, tail_control_offset_mm, sigma_reference_pose_xyz):
        assert torch.equal(pose_xyz, sigma_reference_pose_xyz)
        self.calls += 1
        y = tail_control_offset_mm[:, 1].mean()
        if self.nonfinite_after is not None and self.calls > self.nonfinite_after:
            y = y * torch.tensor(float("nan"))
        # Foreground soft response increases with transverse tail offset.
        return SimpleNamespace(
            union_alpha=torch.sigmoid(y).expand(1, 3, 1),
            ray_valid=torch.ones((3, 1), dtype=torch.bool),
        )


def targets(value=1, weight=1):
    return SimpleNamespace(
        target=np.full((3, 1), value, float), weights=np.full((3, 1), weight, float)
    )


def test_best_finite_iterate_replayed_with_fixed_nodes():
    p = pose()
    saved = p.clone()
    fitted = m.optimize_curve_offsets(ScalarRenderer(), p, targets(), steps=12)
    assert fitted["best_iteration"] == 12
    assert fitted["best_loss"] < fitted["loss_history"][0]
    assert fitted["completed_adam_steps"] == 12
    assert torch.equal(saved, p)
    assert np.array_equal(fitted["offset_mm"], fitted["offset_history_mm"][12])
    assert np.all(fitted["offset_mm"][:, 0] == 0)


def test_zero_candidate_retained_at_optimum_even_after_iterations():
    fitted = m.optimize_curve_offsets(ScalarRenderer(), pose(), targets(0.5), steps=5)
    assert fitted["best_iteration"] == 0
    assert fitted["best_loss"] == 0
    assert not fitted["offset_mm"].any()


def test_nonfinite_later_fit_preserves_earlier_result_with_explicit_status():
    fitted = m.optimize_curve_offsets(ScalarRenderer(nonfinite_after=3), pose(), targets(), steps=7)
    assert fitted["best_iteration"] == 2
    assert np.isfinite(fitted["best_loss"])
    assert fitted["status"] == "nonfinite_fit_stopped_with_best_finite_retained"
    assert fitted["finite_iterates"] == 3


def test_no_foreground_is_explicitly_neutral():
    fitted = m.optimize_curve_offsets(ScalarRenderer(), pose(), targets(weight=0), steps=10)
    assert fitted["status"] == "neutral_no_fitting_foreground"
    assert fitted["best_iteration"] == fitted["completed_adam_steps"] == 0
    assert not fitted["offset_mm"].any()


def test_real_cpu_renderer_optimizer_improves_a_known_curve_without_moving_nodes():
    from medaka3d.calib import Rig

    class Camera:
        def __init__(self, view):
            self.view = view

        def rays(self, xy):
            origins = np.column_stack((xy, np.zeros(len(xy))))
            return origins, np.broadcast_to([0.0, 0.0, 1.0], origins.shape).copy()

    rig = Rig(tuple(Camera(v) for v in m.VIEWS), Path("synthetic-body-curve"))
    p = pose()
    p[:, :, 1] += torch.arange(3, dtype=torch.float64)[:, None] * 25
    points = []
    for fish in range(3):
        yy, xx = np.mgrid[38 + fish * 25 : 45 + fish * 25, 42:56]
        points.append(np.column_stack((xx.ravel(), yy.ravel())))
    pixels = dict.fromkeys(m.VIEWS, np.concatenate(points))
    engine = m.CurvedFishRenderer(rig, pixels, dtype=torch.float64, device="cpu")
    known = torch.tensor([[0.0, 3.0, 0.0]] * 3, dtype=torch.float64)
    with torch.no_grad():
        target = engine.render(p, tail_control_offset_mm=known).union_alpha[0].numpy()
    target_data = SimpleNamespace(target=target, weights=np.full(target.shape, 1 / target.shape[1]))
    original = p.clone()
    fitted = m.optimize_curve_offsets(engine, p, target_data, steps=12)
    assert fitted["best_loss"] < 0.5 * fitted["loss_history"][0]
    assert torch.equal(original, p)
    assert fitted["best_iteration"] > 0

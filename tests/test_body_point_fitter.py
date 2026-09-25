"""Independent body-point loss/fit checks with small synthetic CPU rays only."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
import body_point_fitter as m  # noqa: E402
from renderer import RefractiveFishRenderer  # noqa: E402

from medaka3d.calib import Rig  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402
from test_fixed_body_width_fitter import fit_fixture as base_fixture  # noqa: E402
from test_mask_transport_gap_fitter import ParallelCamera  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic CPU test must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    # Torch2.13's CPU Adam otherwise probes the available GPU's capture state.
    # Hide accelerator availability in this fixture; leave Adam math unchanged.
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda **kwargs: None)
    yield
    torch.set_num_threads(previous)


def loss_fixture():
    ownership = torch.zeros((2, 3, 3, 3), dtype=torch.float64)
    q = torch.zeros_like(ownership)
    weight = torch.zeros_like(ownership)
    valid = torch.ones((3, 3), dtype=torch.bool)
    ownership[0, 0, 0, 0], q[0, 0, 0, 0], weight[0, 0, 0, 0] = 0.25, 9.0, 0.6
    # Two original selected points shared mass1.2/S2. Losing one ray must not
    # double the surviving point's0.6 or turn its missing measurement into data.
    valid[0, 1] = False
    ownership[:, 0, 1] = torch.nan
    q[:, 0, 1] = torch.inf
    weight[0, 0, 1, 0] = 0.6
    ownership[0, 2, 2, 2], q[0, 2, 2, 2], weight[0, 2, 2, 2] = 0.5, 0.25, 0.3
    q[1, 1, 0, 1], weight[1, 1, 0, 1] = 16.0, 0.9
    return ownership.requires_grad_(), q.requires_grad_(), weight, valid


def tensor_bytes(values):
    return [(v.dtype, tuple(v.shape), v.detach().cpu().numpy().tobytes()) for v in values]


def test_hand_calculated_owner_gap_gradients_and_missing_ray_denominator():
    own, q, weight, valid = loss_fixture()
    before = tensor_bytes((own, q, weight, valid))
    owner, gap, detail = m.body_point_loss(own, q, weight, valid, return_details=True)
    torch.testing.assert_close(owner, torch.tensor([0.1375, 0.3], dtype=torch.float64))
    torch.testing.assert_close(gap, torch.tensor([0.4, 0.9], dtype=torch.float64))
    assert detail["final_gap_by_view_body"].shape == (2, 3, 3)
    torch.testing.assert_close(detail["final_gap_by_view_body"].sum((1, 2)), gap)
    torch.testing.assert_close(
        detail["target_mass_by_view_body"].sum((1, 2)),
        torch.tensor([0.3, 0.3], dtype=torch.float64),
    )
    expected_counts = torch.zeros((2, 3, 3), dtype=torch.int64)
    expected_counts[0, 0, 0] = expected_counts[1, 1, 1] = 1
    assert torch.equal(detail["positive_gap_pixel_count"], expected_counts)
    (owner + gap).sum().backward()
    expected_owner = torch.zeros_like(own)
    expected_owner[0, 0, 0, 0] = -0.3
    expected_owner[0, 2, 2, 2] = -0.1
    expected_owner[1, 1, 0, 1] = -0.6
    expected_q = torch.zeros_like(q)
    expected_q[0, 0, 0, 0] = 1 / 30
    expected_q[1, 1, 0, 1] = 0.0375
    torch.testing.assert_close(own.grad, expected_owner, atol=1e-15, rtol=0)
    torch.testing.assert_close(q.grad, expected_q, atol=1e-15, rtol=0)
    assert weight.grad is None and tensor_bytes((own, q, weight, valid)) == before


def test_duplicate_point_with_split_mass_and_extra_zero_rows_do_not_change_loss():
    own, q, weight, valid = loss_fixture()
    expected = m.body_point_loss(own, q, weight, valid)
    duplicated = m.body_point_loss(
        own.repeat_interleave(2, dim=2),
        q.repeat_interleave(2, dim=2),
        weight.repeat_interleave(2, dim=2) / 2,
        valid.repeat_interleave(2, dim=1),
    )
    for actual, wanted in zip(duplicated, expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=1e-15)


@pytest.mark.parametrize("field,bad", [("q", np.nan), ("q", np.inf), ("q", -0.1), ("own", np.nan)])
def test_nonfinite_or_negative_valid_distance_rejected_even_at_zero_weight(field, bad):
    own, q, weight, valid = loss_fixture()
    with torch.no_grad():
        (q if field == "q" else own)[1, 2, 0, 2] = bad
    assert weight[1, 2, 0, 2] == 0
    with pytest.raises(ValueError):
        m.body_point_loss(own, q, weight, valid)


@pytest.mark.parametrize("fault", ["gradient", "negative", "infinite", "wrong_dtype", "two_bodies"])
def test_fixed_point_weights_cannot_be_learned_or_silently_retyped(fault):
    own, q, weight, valid = loss_fixture()
    if fault == "gradient":
        weight.requires_grad_()
    elif fault == "negative":
        weight[0, 0, 0, 0] = -0.1
    elif fault == "infinite":
        weight[0, 0, 0, 0] = torch.inf
    elif fault == "wrong_dtype":
        weight = weight.float()
    else:
        weight[0, 0, 0, 1] = 0.1
    with pytest.raises(ValueError):
        m.body_point_loss(own, q, weight, valid)


def test_point_mass_above_one_is_not_clipped_and_q_one_keeps_clamp_derivative():
    own = torch.ones((1, 3, 1, 3), dtype=torch.float64, requires_grad=True)
    q = torch.ones_like(own, requires_grad=True)
    weight = torch.zeros_like(own)
    weight[0, 0, 0, 0] = 2.4
    valid = torch.ones((3, 1), dtype=torch.bool)
    owner, gap = m.body_point_loss(own, q, weight, valid)
    assert owner.item() == gap.item() == 0
    (owner + gap).sum().backward()
    assert q.grad[0, 0, 0, 0].item() == pytest.approx(0.4)
    assert torch.count_nonzero(q.grad) == 1 and torch.count_nonzero(own.grad) == 0


def test_far_body_point_has_real_pose_gradient_even_when_every_alpha_is_zero():
    rig = Rig(tuple(ParallelCamera(v) for v in VIEWS), Path("synthetic-only"))
    renderer = RefractiveFishRenderer(
        rig, {v: np.array([[0.0, 0.0]]) for v in VIEWS}, dtype=torch.float64
    )
    pose = torch.tensor(
        [
            [[105, 30, 25], [100, 30, 25], [88, 32, 26]],
            [[205, 30, 35], [200, 30, 35], [188, 32, 36]],
            [[305, 30, 45], [300, 30, 45], [288, 32, 46]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    rendered = renderer.render(pose)
    assert rendered.per_fish_alpha.max().item() == rendered.ownership.max().item() == 0
    weight = torch.zeros_like(rendered.ownership)
    weight[..., 0] = 0.6

    def loss(value):
        r = renderer.render(value)
        return m.body_point_loss(r.ownership, r.per_fish_mahalanobis_sq, weight, r.ray_valid)

    owner, gap = loss(pose)
    assert owner.item() == pytest.approx(0.6) and gap.item() > 1
    derivative = torch.autograd.grad((owner + gap).sum(), pose)[0]
    shift = torch.zeros_like(pose)
    shift[0, :, 0] = 1
    rigid_x = derivative[0, :, 0].sum().item()
    assert rigid_x > 0.1 and torch.count_nonzero(derivative[1:]) == 0
    h = 1e-4
    plus = sum(loss(pose.detach() + h * shift)).sum()
    minus = sum(loss(pose.detach() - h * shift)).sum()
    assert rigid_x == pytest.approx(((plus - minus) / (2 * h)).item(), abs=1e-8, rel=1e-7)


@pytest.mark.parametrize("mode", ["coefficient_zero", "zero_mass", "frozen_error"])
def test_inactive_returns_the_exact_frozen_tuple_before_any_extra_render(monkeypatch, mode):
    tracker = SimpleNamespace(last_owner_fit=object(), last_gap_fit=object())
    prior_owner, prior_gap = tracker.last_owner_fit, tracker.last_gap_fit
    proposals, states, rays, targets = [object(), object()], object(), object(), object()
    packet = (
        SimpleNamespace(
            point_weight=np.zeros((1, 3, 0, 3), np.float64),
            point_pixels=np.empty((3, 0, 2), np.float64),
            ray_origins=np.empty((3, 0, 3), np.float64),
            ray_directions=np.empty((3, 0, 3), np.float64),
            ray_valid=np.empty((3, 0), bool),
            query_counts=np.zeros(3, np.int64),
            anchor_ids=np.zeros(1, np.int64),
        )
        if mode == "zero_mass"
        else None
    )
    expected = (
        np.zeros((2, 3, 3, 3)),
        np.array([-0.0, 1.5]),
        np.ones((2, 5)),
        np.ones((2, 3), bool),
    )
    rejection = RuntimeError("original fitter failure")
    calls = []

    def frozen(*args):
        assert all(
            a is b for a, b in zip(args, (tracker, proposals, states, rays, targets), strict=True)
        )
        calls.append(True)
        if mode == "frozen_error":
            raise rejection
        return expected

    def forbidden(*args, **kwargs):
        raise AssertionError("inactive point loss reached renderer/graph/objective")

    monkeypatch.setattr(m, "frozen_fit", frozen)
    for name in ("RefractiveFishRenderer", "GraphFit", "objective", "body_point_loss"):
        monkeypatch.setattr(m, name, forbidden)
    coefficient = 3.0 if mode == "zero_mass" else 0.0
    anchors = np.zeros(2, np.int64)
    if mode == "frozen_error":
        with pytest.raises(RuntimeError) as caught:
            m.fit_with_body_points(
                tracker,
                proposals,
                states,
                rays,
                targets,
                packet,
                candidate_anchor=anchors,
                coefficient=coefficient,
            )
        assert caught.value is rejection and tracker.last_gap_fit is prior_gap
    else:
        result = m.fit_with_body_points(
            tracker,
            proposals,
            states,
            rays,
            targets,
            packet,
            candidate_anchor=anchors,
            coefficient=coefficient,
        )
        assert result is expected
        d = tracker.last_gap_fit
        assert d["total_score"].tobytes() == expected[1].tobytes()
        assert not d["active"] and not d["target_diagnostics_computed"]
        assert d["iteration_total_score"].shape == (0, 2)
        assert np.all(d["best_iteration"] == -1)
    assert calls == [True] and tracker.last_owner_fit is prior_owner


def fitted_fixture(*, learning_rate=0.02, invalid=False):
    args = list(base_fixture(learning_rate=learning_rate)[:5])
    tracker = args[0]
    counts = np.array([3, 2, 1], np.int64)
    pixels = np.full((3, 3, 2), np.nan, np.float64)
    for vi, count in enumerate(counts):
        pixels[vi, :count] = args[4].pixels[VIEWS[vi]][:count] + np.array([2.2, -0.8])
    pixels[0, 1] = np.nan
    if invalid:
        pixels[:] = np.nan
    renderer = RefractiveFishRenderer(
        tracker.rig,
        {v: pixels[i, : counts[i]] for i, v in enumerate(VIEWS)},
        dtype=torch.float64,
    )
    weights = np.zeros((3, 3, 3, 3), np.float64)
    weights[0, :, 0, 1] = [0.3, 0.15, 0.075]
    weights[1, :, 0, 2] = 0.02
    weights[2, :, 0, 0] = [0.15, 0.075, 0.3]
    weights[2, 1, 1, 2] = 0.05
    packet = SimpleNamespace(
        point_pixels=pixels,
        point_weight=weights,
        ray_origins=renderer.ray_origins.numpy().copy(),
        ray_directions=renderer.ray_directions.numpy().copy(),
        ray_valid=renderer.ray_valid.numpy().copy(),
        query_counts=counts,
        anchor_ids=np.arange(3, dtype=np.int64),
    )
    # Parent indices are[1,0], whereas persistent anchor IDs are[2,0].
    return args, packet, np.array([2, 0], np.int64)


def fit_case(fixture):
    args, packet, anchors = fixture
    return m.fit_with_body_points(*args, packet, candidate_anchor=anchors)


def fingerprint(value):
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.tobytes()
    if isinstance(value, dict):
        return {k: fingerprint(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(v) for v in value]
    if isinstance(value, SimpleNamespace):
        return fingerprint(vars(value))
    return value


def observed_base_calls(monkeypatch):
    original = m.objective
    calls = []

    def call(pose, cfg, inputs, denominator, renderer, **kwargs):
        result = original(pose, cfg, inputs, denominator, renderer, **kwargs)
        calls.append(
            dict(
                pose=pose.detach().clone(),
                inputs=inputs,
                denominator=denominator,
                renderer=renderer,
                result=result,
                final=kwargs.get("diagnostics", False),
            )
        )
        return result

    monkeypatch.setattr(m, "objective", call)
    return calls, original


def test_actual_cpu_fit_keeps_old_base_pixels_final_support_and_all15_best_records(monkeypatch):
    fixture = fitted_fixture()
    args, packet, anchors = fixture
    before = fingerprint((vars(packet), anchors))
    original_pose = [p.pose.tobytes() for p in args[1]]
    calls, _ = observed_base_calls(monkeypatch)
    result = fit_case(fixture)
    detail = args[0].last_gap_fit
    assert detail["active"] and detail["reason"] == "fixed_body_point_owner_plus_gap"
    assert len(calls) == 16 and sum(c["final"] for c in calls) == 1
    assert len({id(c["renderer"]) for c in calls}) == 1
    base_renderer = calls[0]["renderer"]
    assert type(base_renderer) is RefractiveFishRenderer and base_renderer.n_pixels == 4
    for vi, view in enumerate(VIEWS):
        np.testing.assert_array_equal(base_renderer.pixels[vi], args[4].pixels[view])
    assert packet.point_pixels.shape[1] == 3
    for name in ("owner_loss", "gap_loss", "base_score", "total_score"):
        assert detail["iteration_" + name].shape == (15, 2)
    for name in ("owner", "gap"):
        assert detail[f"iteration_{name}_gradient_norm"].shape == (14, 2)
    selected = np.argmin(detail["iteration_total_score"], axis=0)
    np.testing.assert_array_equal(detail["best_iteration"], selected)
    best_pose = np.stack([calls[int(selected[b])]["pose"][b].numpy() for b in range(2)])
    assert result[0].tobytes() == best_pose.tobytes()
    np.testing.assert_allclose(
        result[1], detail["iteration_total_score"][selected, np.arange(2)], atol=1e-10, rtol=0
    )
    np.testing.assert_array_equal(
        result[1], detail["base_score"] + 3 * (detail["owner_loss"] + detail["gap_loss"])
    )
    np.testing.assert_array_equal(result[2], detail["original_loss_parts"])
    np.testing.assert_array_equal(
        detail["iteration_total_score"],
        detail["iteration_base_score"]
        + 3 * (detail["iteration_owner_loss"] + detail["iteration_gap_loss"]),
    )
    # Explicit, unequal m/S masses distinguish the candidate anchor order and
    # must not receive another original foreground/weight factor.
    expected_mass = (packet.point_weight[anchors] * packet.ray_valid[None, :, :, None]).sum(
        (1, 2, 3)
    ) / 3
    np.testing.assert_allclose(detail["target_mass"], expected_mass, rtol=0, atol=1e-15)
    final = calls[-1]
    old_image = final["result"][2]
    fg, weight = final["inputs"].target, final["inputs"].weight
    pos = (
        old_image.visible_alpha * (old_image.ownership >= 0.65) * (weight * fg)[None, :, :, None]
    ).sum(2)
    neg = (old_image.visible_alpha * (weight * (1 - fg))[None, :, :, None]).sum(2)
    expected_support = ((pos >= 0.025) & (pos >= 2 * neg)).sum(1) >= 2
    np.testing.assert_array_equal(result[3], expected_support.numpy())
    assert fingerprint((vars(packet), anchors)) == before
    assert [p.pose.tobytes() for p in args[1]] == original_pose


def test_all14_adam_gradients_equal_joint_old_base_plus_independent_point_scalar(monkeypatch):
    fixture = fitted_fixture()
    args, packet, anchors = fixture
    calls, old_objective = observed_base_calls(monkeypatch)
    point_renderer = RefractiveFishRenderer(
        args[0].rig,
        {v: packet.point_pixels[i, : packet.query_counts[i]] for i, v in enumerate(VIEWS)},
        dtype=torch.float64,
    )
    selected_weight = packet.point_weight[anchors]
    adam = torch.optim.Adam
    gradients = []

    class OracleAdam(adam):
        def step(self, *step_args, **step_kwargs):
            pose = self.param_groups[0]["params"][0]
            independent = pose.detach().clone().requires_grad_(True)
            c = calls[-1]
            score = old_objective(
                independent, args[0].config, c["inputs"], c["denominator"], c["renderer"]
            )[0].sum()
            points = point_renderer.render(independent)
            # Independent scalar assembly: coefficient3 cancels fixed /3 here.
            # Query validity and preallocated m/S are the only point weights.
            for b, vi, qi, body in np.argwhere(selected_weight > 0):
                if packet.ray_valid[vi, qi]:
                    q = points.per_fish_mahalanobis_sq[b, vi, qi, body]
                    own = points.ownership[b, vi, qi, body]
                    score = score + float(selected_weight[b, vi, qi, body]) * (
                        (1 - own) ** 2 + torch.sqrt(torch.maximum(q, q.new_tensor(1))) - 1
                    )
            expected = torch.autograd.grad(score, independent)[0]
            assert torch.isfinite(pose.grad).all()
            torch.testing.assert_close(pose.grad, expected, rtol=0, atol=2e-13)
            gradients.append(pose.grad.detach().clone())
            return super().step(*step_args, **step_kwargs)

    monkeypatch.setattr(m.torch.optim, "Adam", OracleAdam)
    fit_case(fixture)
    assert len(gradients) == 14 and any(torch.count_nonzero(g) for g in gradients)


def test_strict_ties_keep_initial_pose_after_fourteen_real_adam_steps():
    fixture = fitted_fixture(learning_rate=1e-300)
    initial = np.stack([p.pose for p in fixture[0][1]])
    result = fit_case(fixture)
    assert result[0].tobytes() == initial.tobytes()
    np.testing.assert_array_equal(fixture[0][0].last_gap_fit["best_iteration"], [0, 0])
    assert fixture[0][0].last_gap_fit["iteration_total_score"].shape == (15, 2)


@pytest.mark.parametrize("field", ["ray_origins", "ray_directions", "ray_valid"])
def test_packet_geometry_drift_fails_before_objective_or_diagnostic_mutation(monkeypatch, field):
    fixture = fitted_fixture()
    args, packet, _ = fixture
    if field == "ray_valid":
        packet.ray_valid[0, 0] = False
    else:
        getattr(packet, field)[0, 0, 0] += 0.1
    prior = args[0].last_gap_fit = {"previous": True}

    def forbidden(*args, **kwargs):
        raise AssertionError("changed point rays reached objective")

    monkeypatch.setattr(m, "objective", forbidden)
    with pytest.raises(ValueError, match="point renderer packet"):
        fit_case(fixture)
    assert args[0].last_gap_fit is prior


def test_nonzero_weights_with_all_invalid_rays_return_frozen_without_optimizer(monkeypatch):
    fixture = fitted_fixture(invalid=True)
    expected = (np.zeros((2, 3, 3, 3)), np.zeros(2), np.zeros((2, 5)), np.zeros((2, 3), bool))
    calls = []

    def frozen(*args):
        assert all(a is b for a, b in zip(args, fixture[0], strict=True))
        calls.append(True)
        return expected

    def forbidden(*args, **kwargs):
        raise AssertionError("all invalid point rays reached objective or optimizer")

    monkeypatch.setattr(m, "frozen_fit", frozen)
    monkeypatch.setattr(m, "objective", forbidden)
    monkeypatch.setattr(m.torch.optim, "Adam", forbidden)
    assert fit_case(fixture) is expected and calls == [True]
    d = fixture[0][0].last_gap_fit
    assert d["reason"] == "zero_valid_ray_target" and not d["active"]


def test_unused_anchor_mass_cannot_activate_selected_zero_mass_proposals(monkeypatch):
    fixture = fitted_fixture()
    args, packet, anchors = fixture
    packet.point_weight[anchors] = 0
    assert packet.point_weight[1].any() and not packet.point_weight[anchors].any()
    expected = (np.zeros((2, 3, 3, 3)), np.zeros(2), np.zeros((2, 5)), np.zeros((2, 3), bool))
    calls = []

    def frozen(*values):
        assert all(a is b for a, b in zip(values, args, strict=True))
        calls.append(True)
        return expected

    def forbidden(*values, **kwargs):
        raise AssertionError("unselected anchor activated point fitting")

    monkeypatch.setattr(m, "frozen_fit", frozen)
    monkeypatch.setattr(m, "RefractiveFishRenderer", forbidden)
    monkeypatch.setattr(m, "objective", forbidden)
    assert fit_case(fixture) is expected and calls == [True]
    assert args[0].last_gap_fit["reason"] == "zero_effective_target"


@pytest.mark.parametrize("fault", ["objective", "gradient", "final"])
def test_nonfinite_fit_rejects_before_publishing_any_new_diagnostics(monkeypatch, fault):
    fixture = fitted_fixture()
    prior = fixture[0][0].last_gap_fit = {"previous": True}
    original = m.objective

    class InvalidBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, value):
            return torch.full_like(value, torch.nan)

    def call(*args, **kwargs):
        score, parts, rendered = original(*args, **kwargs)
        if fault == "objective" or (fault == "final" and kwargs.get("diagnostics")):
            score = score + torch.inf
        elif fault == "gradient":
            score = InvalidBackward.apply(score)
        return score, parts, rendered

    monkeypatch.setattr(m, "objective", call)
    with pytest.raises(FloatingPointError, match=r"objective|gradient|output"):
        fit_case(fixture)
    assert fixture[0][0].last_gap_fit is prior

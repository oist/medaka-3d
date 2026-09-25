"""Fixed bank selection, frozen numerical objective and support on tiny CPU scenes."""

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import fp64_pose_bank_rescore as m  # noqa: E402
from directional_device_fitter import center_correction  # noqa: E402
from graph_fit import FitInputs, objective  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402
from tracker import TrackerConfig  # noqa: E402

from test_fixed_body_width_renderer import exact, scene, seed  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("rescore cannot construct a CUDA graph or initialize GPU")

    monkeypatch.setattr(m.GraphFit, "__init__", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    yield
    torch.set_num_threads(threads)


def fixture(*, missing=False):
    rig, pixels = scene(missing=missing)
    cfg = TrackerConfig(device="cpu")
    renderer = RefractiveFishRenderer(
        rig, pixels, RendererConfig(width_ratio=cfg.width_ratio), device="cpu", dtype=torch.float64
    )
    reference = torch.from_numpy(seed())
    predicted = reference[None].repeat(2, 1, 1, 1)
    original = predicted + torch.tensor([0.4, -0.3, 0.2], dtype=torch.float64)
    path = torch.stack(
        [(predicted + i * 0.025).to(torch.float32).to(torch.float64) for i in range(15)]
    )
    bank = torch.cat((original[None], path))
    origins = predicted[:, :, None].repeat(1, 1, 3, 1, 1)
    directions = torch.zeros_like(origins)
    for view, axis in enumerate((2, 0, 1)):
        origins[:, :, view, :, axis] -= 100
        directions[:, :, view, :, axis] = 1
    lengths = torch.linalg.vector_norm(torch.diff(reference, dim=1), dim=-1)[None]
    target = torch.tensor([[1, 1, 0]] * 3, dtype=torch.float64)
    inputs = FitInputs(
        origins=origins,
        directions=directions,
        held=torch.ones((2, 3, 3, 3), dtype=torch.float64),
        predicted=predicted,
        motion_sigma=torch.full((2, 3), 0.8, dtype=torch.float64),
        lengths=lengths,
        length_sigma=torch.clamp(0.2 * lengths, min=1.0),
        node_weight=torch.tensor([1, 1, 0.35], dtype=torch.float64)[None, None, None],
        target=target,
        weight=torch.full_like(target, 1 / 3) * renderer.ray_valid,
    )
    axes = torch.zeros((2, 3, 3), dtype=torch.float64)
    axes[:, :, 2] = 1
    return bank, dict(
        config=cfg,
        inputs=inputs,
        renderer=renderer,
        axes=axes,
        parallel_sigma=torch.full((2, 3), 2.0, dtype=torch.float64),
        enabled=torch.tensor([[True, False, True], [False, True, False]]),
    )


def snapshot(bank, args):
    tensors = [bank, args["axes"], args["parallel_sigma"], args["enabled"]]
    tensors += [getattr(args["inputs"], f.name) for f in dataclasses.fields(FitInputs)]
    tensors += [
        getattr(args["renderer"], k) for k in ("ray_origins", "ray_directions", "ray_valid")
    ]
    return [(t.dtype, tuple(t.shape), t.detach().numpy().tobytes()) for t in tensors]


def fake_objective(monkeypatch, *, corrupt_at=None, corrupt_field="score", rendering=None):
    calls = []

    def score(pose, cfg, inputs, denominator, renderer, *, diagnostics=False):
        assert diagnostics is True and denominator == 9 * float(inputs.node_weight.sum())
        calls.append(pose.clone())
        values = pose[:, 0, 0, 0].clone()
        parts = torch.stack((values, values * 0, values * 0, values * 0, values * 0), dim=-1)
        shape = (len(pose), 3, renderer.n_pixels, 3)
        rendered = rendering or SimpleNamespace(
            visible_alpha=torch.full(shape, 0.1, dtype=torch.float64),
            ownership=torch.full(shape, 0.7, dtype=torch.float64),
            union_alpha=torch.full(shape[:-1], 0.3, dtype=torch.float64),
        )
        if len(calls) == corrupt_at:
            if corrupt_field == "score":
                values[0] = float("nan")
            elif corrupt_field == "parts":
                parts[0, 1] = float("inf")
            elif corrupt_field == "final_change":
                values[0] += 1
            else:
                rendered.ownership[0, 0, 0, 0] = float("nan")
        return values, parts, rendered

    monkeypatch.setattr(m, "objective", score)
    return calls


@pytest.mark.parametrize("directional", [False, True])
def test_real_frozen_fp64_objective_all16_rows_final_support_and_input_immutability(directional):
    bank, args = fixture(missing=True)
    bank.requires_grad_(True)
    before = snapshot(bank, args)
    result = m.rescore_pose_bank(bank, **args, directional=directional)
    expected_scores, expected_parts = [], []
    with torch.no_grad():
        for value in bank:
            score, parts, _ = objective(
                value,
                args["config"],
                args["inputs"],
                9 * float(args["inputs"].node_weight.sum()),
                args["renderer"],
                diagnostics=True,
            )
            if directional:
                difference, _, center = center_correction(
                    value[:, :, 1] - args["inputs"].predicted[:, :, 1],
                    args["inputs"].motion_sigma,
                    args["axes"],
                    args["parallel_sigma"],
                    args["enabled"],
                )
                score = score + args["config"].motion_weight * difference
                parts = torch.cat((parts[:, :2], center[:, None], parts[:, 3:]), dim=-1)
            expected_scores.append(score)
            expected_parts.append(parts)
    exact(result["bank_scores"], torch.stack(expected_scores))
    exact(result["bank_parts"], torch.stack(expected_parts))
    expected_indices = torch.argmin(torch.stack(expected_scores), dim=0)
    exact(result["selected_bank_index"], expected_indices)
    exact(result["fitted_pose_xyz"], bank[expected_indices, torch.arange(2)])
    exact(result["local_cost"], result["bank_scores"][expected_indices, torch.arange(2)])
    exact(result["final_loss_parts"], result["bank_parts"][expected_indices, torch.arange(2)])
    rendered = args["renderer"].render(result["fitted_pose_xyz"])
    w, t = args["inputs"].weight, args["inputs"].target
    positive = (
        rendered.visible_alpha
        * (rendered.ownership >= 0.65)
        * w[None, :, :, None]
        * t[None, :, :, None]
    ).sum(2)
    negative = (rendered.visible_alpha * w[None, :, :, None] * (1 - t)[None, :, :, None]).sum(2)
    exact(result["positive"], positive)
    exact(result["negative"], negative)
    exact(result["view_supported"], (positive >= 0.025) & (positive >= 2 * negative))
    exact(result["mask_supported"], result["view_supported"].sum(1) >= 2)
    assert snapshot(bank, args) == before and bank.grad is None
    assert all(
        not value.requires_grad for value in result.values() if isinstance(value, torch.Tensor)
    )
    assert (result["local_cost"] <= result["initial_score"]).all()
    assert result["bank_size"] == 16 and result["objective_evaluations"] == 17
    # Owning the initial diagnostic must not alias either the bank result or inputs.
    previous = result["bank_scores"].clone()
    result["initial_score"].add_(100)
    exact(result["bank_scores"], previous)
    assert snapshot(bank, args) == before


def test_all16_are_evaluated_strict_tie_keeps_first_per_outer_proposal(monkeypatch):
    bank, args = fixture()
    bank[:, 0, 0, 0, 0] = torch.tensor([9] * 7 + [1, 1] + [9] * 7, dtype=torch.float64)
    bank[:, 1, 0, 0, 0] = torch.tensor([0] * 16, dtype=torch.float64)
    # Distinct tied poses in other coordinates ensure this tests pose retention.
    bank[:, :, 1, 2, 0] = torch.arange(16, dtype=torch.float64)[:, None]
    calls = fake_objective(monkeypatch)
    result = m.rescore_pose_bank(bank, **args, directional=False)
    exact(result["selected_bank_index"], torch.tensor([7, 0], dtype=torch.int64))
    exact(result["fitted_pose_xyz"], bank[torch.tensor([7, 0]), torch.arange(2)])
    assert len(calls) == 17
    for index in range(16):
        exact(calls[index], bank[index])
    exact(calls[-1], result["fitted_pose_xyz"])
    assert result["bank_scores"].shape == (16, 2) and result["bank_parts"].shape == (16, 2, 5)


def test_support_exact_thresholds_two_views_no_pixel_or_view_renormalization(monkeypatch):
    bank, args = fixture()
    args["inputs"] = dataclasses.replace(
        args["inputs"],
        target=torch.tensor([[1, 0, 0]] * 3, dtype=torch.float64),
        weight=torch.tensor([[1, 1, 0]] * 3, dtype=torch.float64),
    )
    contribution = torch.zeros((2, 3, 3, 3), dtype=torch.float64)
    ownership = torch.full_like(contribution, 0.65)
    contribution[:, :, 0, 0] = 0.025
    contribution[:, :, 1, 0] = 0.0125
    contribution[:, :, 0, 1] = 0.025
    ownership[:, 1:, 0, 1] = np.nextafter(0.65, 0)
    contribution[:, :, 0, 2] = 0.05
    contribution[:, :, 1, 2] = torch.tensor([0.03, 0, 0.03])
    rendered = SimpleNamespace(
        visible_alpha=contribution, ownership=ownership, union_alpha=contribution.sum(-1)
    )
    fake_objective(monkeypatch, rendering=rendered)
    result = m.rescore_pose_bank(bank, **args, directional=False)
    exact(result["mask_supported"], torch.tensor([[True, False, False]] * 2))
    assert torch.equal(result["view_supported"][:, :, 0], torch.ones((2, 3), dtype=torch.bool))
    exact(result["positive"][:, :, 0], torch.full((2, 3), 0.025, dtype=torch.float64))
    exact(result["negative"][:, :, 0], torch.full((2, 3), 0.0125, dtype=torch.float64))


@pytest.mark.parametrize(
    "bad",
    [
        "bank_dtype",
        "bank_count",
        "empty_batch",
        "bank_nonfinite",
        "input_dtype",
        "input_nonfinite",
        "axis_nonfinite",
        "enabled_dtype",
        "sigma_zero",
        "invalid_ray_weight",
        "renderer_dtype",
    ],
)
def test_invalid_input_fails_before_any_objective_call(monkeypatch, bad):
    bank, args = fixture(missing=True)
    calls = fake_objective(monkeypatch)
    if bad == "bank_dtype":
        bank = bank.float()
    elif bad == "bank_count":
        bank = bank[:15]
    elif bad == "empty_batch":
        bank = bank[:, :0]
    elif bad == "bank_nonfinite":
        bank[15, 0, 0, 0, 0] = float("inf")
    elif bad == "input_dtype":
        args["inputs"] = dataclasses.replace(
            args["inputs"], predicted=args["inputs"].predicted.float()
        )
    elif bad == "input_nonfinite":
        args["inputs"].lengths[0, 0, 0] = float("nan")
    elif bad == "axis_nonfinite":
        args["axes"][0, 0, 0] = float("nan")
    elif bad == "enabled_dtype":
        args["enabled"] = args["enabled"].double()
    elif bad == "sigma_zero":
        args["parallel_sigma"][0, 0] = 0
    elif bad == "invalid_ray_weight":
        args["inputs"].weight[1, 0] = 1
    else:
        args["renderer"].dtype = torch.float32
    with pytest.raises(ValueError):
        m.rescore_pose_bank(bank, **args)
    assert not calls


@pytest.mark.parametrize("field", ["score", "parts", "ownership"])
def test_nonfinite_late_unselected_bank_output_is_not_silently_ignored(monkeypatch, field):
    bank, args = fixture()
    fake_objective(monkeypatch, corrupt_at=15, corrupt_field=field)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        m.rescore_pose_bank(bank, **args, directional=False)


def test_same_fp64_final_reevaluation_change_is_rejected(monkeypatch):
    bank, args = fixture()
    fake_objective(monkeypatch, corrupt_at=17, corrupt_field="final_change")
    with pytest.raises(FloatingPointError, match="reevaluation changed"):
        m.rescore_pose_bank(bank, **args, directional=False)

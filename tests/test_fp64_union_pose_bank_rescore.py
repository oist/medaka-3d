"""Full visibility and union-only bank equivalence on tiny original CPU scenes."""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import fp64_pose_bank_rescore as old  # noqa: E402
import fp64_union_pose_bank_rescore as new  # noqa: E402

from test_fixed_body_width_renderer import exact  # noqa: E402
from test_fp64_pose_bank_rescore import fixture, snapshot  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a, **k: pytest.fail("CPU only"))
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("directional", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_all_bank_scores_parts_winners_and_final_tensors_byte_exact(directional, missing):
    bank, args = fixture(missing=missing)
    before = snapshot(bank, args)
    full = old.rescore_pose_bank(bank, **args, directional=directional)
    union = new.rescore_union_pose_bank(bank, **args, directional=directional)
    for key, value in full.items():
        if isinstance(value, torch.Tensor):
            exact(value, union[key])
            assert not union[key].requires_grad
    assert snapshot(bank, args) == before
    assert union["union_objective_evaluations"] == 16
    assert union["full_diagnostic_evaluations"] == 1


def test_only_selected_pose_gets_visibility_and_all_ties_keep_initial(monkeypatch):
    bank, args = fixture()
    bank = bank[:1].repeat(16, 1, 1, 1, 1)
    renderer = args["renderer"]
    original = renderer.render
    calls = []

    def render(pose, *a, **k):
        calls.append(pose.clone())
        return original(pose, *a, **k)

    monkeypatch.setattr(renderer, "render", render)
    result = new.rescore_union_pose_bank(bank, **args)
    assert len(calls) == 1 and result["selected_bank_index"].tolist() == [0, 0]
    exact(calls[0], bank[0])


def test_nonfinite_late_nonwinning_bank_cannot_be_hidden(monkeypatch):
    bank, args = fixture()
    original, count = new.objective, 0

    def broken(*a, **k):
        nonlocal count
        count += 1
        scores, parts, rendering = original(*a, **k)
        if count == 15:
            parts[0, 4] = float("nan")
        return scores, parts, rendering

    monkeypatch.setattr(new, "objective", broken)
    with pytest.raises(FloatingPointError, match="union-bank evaluation"):
        new.rescore_union_pose_bank(bank, **args)


def test_full_diagnostic_mismatch_is_rejected_without_tolerance_relaxation(monkeypatch):
    bank, args = fixture()
    original = new._evaluate

    def changed(*a, **k):
        score, parts, rendering = original(*a, **k)
        score = torch.nextafter(score, torch.full_like(score, float("inf")))
        return score, parts, rendering

    monkeypatch.setattr(new, "_evaluate", changed)
    with pytest.raises(FloatingPointError, match="differs from union-only"):
        new.rescore_union_pose_bank(bank, **args)

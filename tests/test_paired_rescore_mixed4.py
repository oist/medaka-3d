"""New saved-bank caller boundaries, no model, GPU, fitting or historical rerun."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import paired_rescore_mixed4 as m  # noqa: E402


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a, **k: pytest.fail("CPU only"))


def saved_case():
    b, n = 2, 3
    score = np.full((16, b), 10.0)
    score[2, 0], score[3, 1] = 1.0, 2.0
    parts = np.repeat(score[..., None], 5, axis=-1)
    selected = np.array([2, 3], np.int64)
    arrays = dict(
        selected_bank_index=selected,
        bank_scores=score,
        bank_parts=parts,
        initial_score=score[0].copy(),
        initial_parts=parts[0].copy(),
        fitted_pose_xyz=np.zeros((b, 3, 3, 3)),
        local_cost=np.array([1.0, 2.0]),
        final_loss_parts=parts[selected, np.arange(b)].copy(),
        mask_supported=np.ones((b, 3), bool),
        view_supported=np.ones((b, 3, 3), bool),
        positive=np.ones((b, 3, 3)),
        negative=np.zeros((b, 3, 3)),
        visible_alpha=np.ones((b, 3, n, 3)),
        ownership=np.ones((b, 3, n, 3)),
        union_alpha=np.ones((b, 3, n)),
        ray_valid=np.ones((3, n), bool),
    )
    assert arrays.keys() == set(m.TENSOR_FIELDS)
    trace = {"fp64_" + k: v.copy() for k, v in arrays.items()}
    meta = dict(
        schema="fp64_union_pose_bank_rescore_v1",
        bank_size=16,
        objective_evaluations=17,
        union_objective_evaluations=16,
        full_diagnostic_evaluations=1,
        directional=True,
    )
    return arrays, trace, meta


def test_exact_all16_fields_and_strict_minimum():
    arrays, trace, meta = saved_case()
    assert m.comparison(arrays, trace)["passed"]
    m.validate_result(arrays, meta, trace, "union")
    full = {
        k: v
        for k, v in meta.items()
        if k not in ("union_objective_evaluations", "full_diagnostic_evaluations")
    }
    full["schema"] = "fp64_pose_bank_rescore_v1"
    m.validate_result(arrays, full, trace, "full")


@pytest.mark.parametrize("change", ["ulp", "support", "dtype", "missing", "extra", "nan", "zero"])
def test_every_result_difference_is_reported_then_rejected(change):
    arrays, trace, meta = saved_case()
    if change == "ulp":
        arrays["bank_scores"][14, 0] = np.nextafter(10.0, np.inf)
    elif change == "support":
        arrays["view_supported"][1, 2, 0] = False
    elif change == "dtype":
        arrays["selected_bank_index"] = arrays["selected_bank_index"].astype(np.int32)
    elif change == "missing":
        del arrays["ray_valid"]
    elif change == "extra":
        arrays["unexpected"] = np.array([0.0])
    elif change == "nan":
        arrays["ownership"][0, 0, 0, 0] = np.nan
    else:
        arrays["fitted_pose_xyz"][0, 0, 0, 0] = -0.0
    report = m.comparison(arrays, trace)
    assert not report["passed"]
    if change == "zero":
        assert report["fields"]["fitted_pose_xyz"]["changed_indices"] == []
        assert not report["fields"]["fitted_pose_xyz"]["byte_exact"]
    with pytest.raises(ValueError, match="BYTEEXACT"):
        m.validate_result(arrays, meta, trace, "union")


def test_wrong_union_call_counts_rejected():
    arrays, trace, meta = saved_case()
    meta["full_diagnostic_evaluations"] = 17
    with pytest.raises(ValueError, match="call counts"):
        m.validate_result(arrays, meta, trace, "union")


def test_signed_zero_result_and_failure_report_saved_before_raise(tmp_path):
    arrays, trace, meta = saved_case()
    arrays["fitted_pose_xyz"][0, 0, 0, 0] = -0.0
    output = m.proof.Output(tmp_path / "new_result")
    with pytest.raises(ValueError, match="BYTEEXACT"):
        m.save_checked_result(
            output,
            "f66524_union",
            arrays,
            meta,
            0.01,
            trace,
            "union",
            1,
            dict(allocated_bytes=0, reserved_bytes=0),
        )
    assert len(output.records) == 4
    root = tmp_path / "new_result"
    loaded = m.read_arrays(root / "f66524_union_rescore.npz")
    assert loaded["fitted_pose_xyz"].tobytes() == arrays["fitted_pose_xyz"].tobytes()
    report = json.loads((root / "f66524_union_comparison.json").read_text())
    assert not report["passed"]


def test_timer_excludes_cpu_copy_and_includes_final_sync(monkeypatch):
    events = []
    ticks = iter((10.0, 12.0))
    original_numpy = torch.Tensor.numpy

    def numpy(self, *a, **k):
        events.append("copy")
        return original_numpy(self, *a, **k)

    monkeypatch.setattr(torch.Tensor, "numpy", numpy)

    def clock():
        events.append("clock")
        return next(ticks)

    def rescore(bank, **kwargs):
        events.append("rescore")
        assert kwargs["directional"] is True
        return dict(score=bank.clone(), schema="fake")

    problem = SimpleNamespace(
        inputs=None, renderer=None, axes=None, parallel_sigma=None, enabled=None
    )
    arrays, metadata, elapsed = m.invoke(
        rescore,
        torch.ones(2),
        problem,
        None,
        synchronize=lambda: events.append("sync"),
        clock=clock,
    )
    assert events == ["sync", "clock", "rescore", "sync", "clock", "copy"]
    assert elapsed == 2 and arrays["score"].tolist() == [1, 1] and metadata == {"schema": "fake"}


def test_new_manifest_size_and_input_string_are_distinct_from_old(tmp_path):
    root = tmp_path / "mixed"
    root.mkdir()
    path = root / "tiny.json"
    path.write_text("{}", encoding="utf-8")
    evidence = m.proof.Evidence()
    digest = evidence.add(path)
    seal = dict(
        outputs={"tiny.json": dict(path=str(path), size_bytes=2, sha256=digest)},
        inputs={str(path): digest},
    )
    assert m.mixed_member(root, seal, "tiny.json", evidence) == path
    assert m.mixed_input(seal, path, evidence) == path
    bad = copy.deepcopy(seal)
    bad["outputs"]["tiny.json"]["path"] = str(root / "unrelated.json")
    with pytest.raises(ValueError, match="path/size"):
        m.mixed_member(root, bad, "tiny.json", evidence)
    bad = copy.deepcopy(seal)
    bad["inputs"][str(path)] = dict(path=str(path), sha256=digest)
    with pytest.raises(ValueError, match="input missing"):
        m.mixed_input(bad, path, evidence)


@pytest.mark.requires_artifacts(m.MIXED_ROOT / "qualification.provenance.json")
def test_actual_small_summary_plan_manifest_contract_without_numeric_payload(monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("no numeric input"))
    evidence = m.proof.Evidence()
    seal = m.previous.sealed_header(
        m.MIXED_ROOT, "qualification.provenance.json", m.MIXED_SHA, m.prior.STAGE, evidence
    )
    summary = m.proof.read_json(m.mixed_member(m.MIXED_ROOT, seal, "summary.json", evidence))
    plan = m.proof.read_json(m.mixed_member(m.MIXED_ROOT, seal, "plan.json", evidence))
    m.validate_prior_summary(summary, plan)
    changed = copy.deepcopy(summary)
    changed["comparisons"][2]["frame_idx"] += 1
    with pytest.raises(ValueError, match="scope"):
        m.validate_prior_summary(changed, plan)
    assert not torch.cuda.is_initialized()


def test_order_and_summary_do_not_hide_cold_first_pair():
    rows = [
        dict(
            frame_idx=n,
            batch=m.BATCHES[i],
            order=list(m.arm_order(i)),
            seconds=dict(full=2.0, union=x),
        )
        for i, (n, x) in enumerate(zip(m.FRAMES, (10.0, 1.0, 2.0, 3.0), strict=True))
    ]
    out = m.timing_summary(rows)
    assert out["first_pair"] == rows[0] and out["remaining_pairs"] == rows[1:]
    assert out["remaining_paired_median"] == 1 and out["remaining_union_faster_count"] == 1
    rows[1]["order"] = ["full", "union"]
    with pytest.raises(ValueError, match="timing scope"):
        m.timing_summary(rows)


def test_invalid_inputs_fail_before_gpu_and_preserve_failure_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m.previous, "snapshot", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(m.prior.baseline, "gpu_sample", lambda: calls.append("GPU"))

    def reject(evidence):
        raise ValueError("synthetic invalid bank binding")

    monkeypatch.setattr(m, "load_inputs", reject)
    root = tmp_path / "failed_input"
    with pytest.raises(ValueError, match="invalid bank binding"):
        m.run(SimpleNamespace(output=root, execute=True, device="cuda"))
    seal = json.loads((root / "qualification.provenance.json").read_text())
    assert seal["status"] == "failed" and seal["completed_rescore_calls"] == 0
    assert not seal["GPU_used"] and not seal["GPU_initialization_attempted"] and not calls
    assert "plan.json" in seal["outputs"]


def test_problem_reference_rays_are_bound_before_rescore():
    packet = dict(proposal_pose_xyz=np.zeros((2, 3, 3, 3)))
    trace = dict(fp64_pose_bank=np.zeros((16, 2, 3, 3, 3)))
    values = dict(
        pose_bank=trace["fp64_pose_bank"].copy(),
        original_pose_xyz=packet["proposal_pose_xyz"].copy(),
    )
    for k in (
        "image_ray_origins",
        "image_ray_directions",
        "inputs_origins",
        "inputs_directions",
        "inputs_held",
    ):
        values[k] = np.zeros((2, 3))
        trace["fp64_" + k] = values[k].copy()
    values["image_ray_valid"] = np.ones((3, 2), bool)
    trace["fp64_ray_valid"] = values["image_ray_valid"].copy()
    m.validate_problem(values, packet, trace)
    values["inputs_origins"][0, 0] = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="original FP64 problem"):
        m.validate_problem(values, packet, trace)

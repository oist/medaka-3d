"""New bank/schema/finite-control/lifecycle boundaries, synthetic CPU only."""

import copy
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import mixed_precision_fitter as mixed_module  # noqa: E402
import qualify_mixed_precision4 as m  # noqa: E402
from fp64_pose_bank_rescore import rescore_pose_bank  # noqa: E402

from test_fp64_pose_bank_rescore import fixture as real_fixture  # noqa: E402


def fields77():
    return {f"field_{i}": np.array([i, 1.0], np.float64) for i in range(77)}


def statistics(index):
    captures = (1, 2, 3, 3)[index]
    return dict(
        schema="mixed_precision_fitter_v1",
        bank_size=16,
        FP64_objective_evaluations=17,
        FP32_evaluations_per_replay=15,
        FP32_updates_per_replay=14,
        scientific_dtype="float64",
        equivalent_to_FP64_search=False,
        prepare_seconds=0.1,
        generate_seconds=0.2,
        rescore_seconds=0.3,
        diagnostic_copy_seconds=0.1,
        cache=dict(
            capacity=4,
            captures=captures,
            hits=index + 1 - captures,
            misses=captures,
            evictions=0,
            failures=0,
            resident=captures,
            peak_resident=captures,
        ),
    )


@pytest.fixture(scope="module")
def bank_case():
    bank, args = real_fixture()
    original = bank[0].numpy().copy()
    path = np.stack([(original.astype(np.float32) - np.float32(i * 0.025)) for i in range(15)])
    bank = np.concatenate((original[None], path.astype(np.float64)))
    inputs = args["inputs"]
    args["inputs"] = dataclasses.replace(
        inputs,
        origins=inputs.origins.float().double(),
        directions=inputs.directions.float().double(),
    )
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = rescore_pose_bank(torch.from_numpy(bank), **args)
    finally:
        torch.set_num_threads(before)
    trace = {
        "fp64_" + k: v.numpy().copy() for k, v in result.items() if isinstance(v, torch.Tensor)
    }
    trace.update(
        original_pose_xyz=original,
        fp64_pose_bank=bank,
        fp32_pose_before=path,
        fp32_score=np.broadcast_to(np.arange(15, dtype=np.float32)[:, None], (15, 2)).copy(),
        fp32_parts=np.zeros((15, 2, 5), np.float32),
        fp32_best_iteration=np.zeros(2, np.int64),
        fp32_best_score=np.zeros(2, np.float32),
        fp32_finite_flags=np.ones(3, bool),
        fp32_uses_positive_product_bound=np.asarray(False),
    )
    for k, steps in (
        ("base_gradient", 15),
        ("update_gradient", 14),
        ("moment", 14),
        ("variance", 14),
    ):
        trace["fp32_" + k] = np.zeros((steps, 2, 3, 3, 3), np.float32)
    trace["fp32_pose_after"] = path[1:].copy()
    scores = trace["fp64_bank_scores"]
    alternative = np.ones(2, np.int64)
    chosen = np.where(scores[alternative, np.arange(2)] < scores[0], alternative, 0)
    subset = scores[chosen, np.arange(2)]
    trace.update(
        subset2_bank_index=chosen,
        subset2_score=subset,
        subset2_missed_strict_improvement=subset > scores.min(0),
        subset2_cost_regret=subset - scores.min(0),
    )
    arrays = {
        k: trace["fp64_" + k].copy()
        for k in (
            "fitted_pose_xyz",
            "local_cost",
            "final_loss_parts",
            "initial_score",
            "initial_parts",
            "mask_supported",
        )
    }
    packet = dict(
        proposal_pose_xyz=original,
        fit_pixels=args["renderer"].pixels.copy(),
        proposal_plan=np.broadcast_to(
            np.arange(3, dtype=np.int32)[None, :, None, None], (2, 3, 3, 3)
        ).copy(),
    )
    for view_index, view in enumerate(args["renderer"].views):
        packet[view + "_ray_origins"] = inputs.origins[0, :, view_index].numpy().astype(np.float32)
        packet[view + "_ray_directions"] = (
            inputs.directions[0, :, view_index].numpy().astype(np.float32)
        )
    for k in ("origins", "directions"):
        trace["fp64_image_ray_" + k] = getattr(args["renderer"], "ray_" + k).numpy().copy()
    for k in ("origins", "directions", "held"):
        trace["fp64_inputs_" + k] = getattr(args["inputs"], k).numpy().copy()
    return packet, arrays, trace, args


def test_actual_cpu_helper_output_bank_and_original_ray_boundaries(bank_case):
    packet, arrays, trace, args = copy.deepcopy(bank_case)
    result = m.validate_trace(packet, arrays, trace)
    assert result["all16_first_minimum"] and result["final_no_worse_than_initial"]
    assert m.validate_fp64_rays(packet, trace, args["renderer"].rig, args["config"])[
        "original_FP64_image_rays_exact"
    ]
    assert (
        result["subset2_missed_proposals"]
        == np.flatnonzero(trace["subset2_missed_strict_improvement"]).tolist()
    )


@pytest.mark.parametrize(
    "change", ["initial", "promote", "last_gradient", "first_minimum", "subset", "product_mode"]
)
def test_bank_contract_rejects_concrete_corruption(bank_case, change):
    packet, arrays, trace, _ = copy.deepcopy(bank_case)
    if change == "initial":
        trace["fp64_pose_bank"][0, 0, 0, 0, 0] += 1e-6
    elif change == "promote":
        trace["fp64_pose_bank"][15, 0, 0, 0, 0] += 1e-6
    elif change == "last_gradient":
        trace["fp32_base_gradient"][-1, 0, 0, 0, 0] = np.nan
    elif change == "first_minimum":
        trace["fp64_selected_bank_index"][0] = (trace["fp64_selected_bank_index"][0] + 1) % 16
    elif change == "subset":
        trace["subset2_cost_regret"][0] += 1
    else:
        trace["fp32_uses_positive_product_bound"][...] = True
    with pytest.raises(ValueError):
        m.validate_trace(packet, arrays, trace)


def test_original_ray_round_then_promote_is_detected(bank_case):
    packet, _, trace, args = copy.deepcopy(bank_case)
    # A calibrated float64 image ray must retain even its low bits.
    trace["fp64_image_ray_origins"][0, 0, 0] = np.nextafter(
        trace["fp64_image_ray_origins"][0, 0, 0], np.inf
    )
    with pytest.raises(ValueError, match="image rays"):
        m.validate_fp64_rays(packet, trace, args["renderer"].rig, args["config"])


def test_mixed_differences_keep_all_discrete_and_nonfinite_locations_without_equivalence_gate():
    ref = fields77()
    ref["field_0"] = np.array([1, 2, 3], np.int32)
    ref["field_1"] = np.array(["predict", "tail"], dtype="U10")
    ref["field_2"] = np.array([np.nan, np.inf, -np.inf, 1.0])
    actual = copy.deepcopy(ref)
    actual["field_0"][:2] = [3, 4]
    actual["field_1"][1] = "predict"
    actual["field_2"][:] = [0, np.nan, np.inf, 1234]
    report = m.mixed_difference(actual, ref)
    assert report["fields"]["field_0"]["changed_indices"] == [[0], [1]]
    assert report["fields"]["field_0"]["actual_values"] == [3, 4]
    assert report["fields"]["field_1"]["reference_values"] == ["tail"]
    assert report["fields"]["field_2"]["maximum_absolute_difference_common_finite"] == 1233
    assert report["fields"]["field_2"]["nonfinite"]["reference"]["negative_inf"] == [[2]]
    assert report["tolerance_or_equivalence_pass_not_applied"]
    assert m.proof.json_bytes(report)
    # It is still an error to lose the public shape/dtype after saving differences.
    actual["field_0"] = actual["field_0"][:2]
    assert not m.mixed_difference(actual, ref)["fields"]["field_0"]["same_shape"]
    with pytest.raises(ValueError, match="shape/dtype"):
        m.validate_public_schema(actual, ref)


def test_fast_byteexact_gate_remains_stricter_than_cpu_atol():
    ref = fields77()
    actual = copy.deepcopy(ref)
    actual["field_0"][1] = np.nextafter(1.0, 2.0)
    assert not m.previous.compare_all77(
        actual, {k: ref for k in ("CUDA_golden", "Fast301", "CPU4")}
    )["passed"]
    assert m.mixed_difference(actual, ref)["byte_equal_fields"] == 76


def test_semantic_nonfinite_scope_is_explicit_and_computed_values_still_fail():
    a = fields77()
    del a["field_0"], a["field_1"], a["field_2"]
    a.update(
        before_native_pixel_error=np.array([np.nan, 0.1]),
        before_assigned_source=np.array([False, True]),
        control_score_margin=np.array([np.inf]),
    )
    m.validate_numerical_outputs(a)
    a["before_native_pixel_error"][1] = np.nan
    with pytest.raises(ValueError, match="assigned"):
        m.validate_numerical_outputs(a)
    a["before_native_pixel_error"][1] = 0.1
    a["field_3"][0] = np.inf
    with pytest.raises(ValueError, match="scientific"):
        m.validate_numerical_outputs(a)


@pytest.mark.parametrize("key", ["captures", "hits", "failures", "capacity"])
def test_cache_history_and_failures_are_hard_gates(key):
    for i in range(4):
        m.validate_statistics(statistics(i), i)
    bad = statistics(3)
    bad["cache"][key] += 1
    with pytest.raises(ValueError, match="cache transition"):
        m.validate_statistics(bad, 3)


def test_pending_pin_stops_before_output_and_model_import(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "SOURCE_PINS", {"pending": "PENDING"})
    monkeypatch.setattr(m.importlib, "import_module", lambda *a: pytest.fail("import before pins"))
    with pytest.raises(ValueError, match="pending"):
        m.run(SimpleNamespace(execute=True, device="cuda", output=tmp_path / "absent"))
    assert not (tmp_path / "absent").exists()


def mock_run(monkeypatch, tmp_path, *, failure=None):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "SOURCE_PINS", {})
    monkeypatch.setattr(
        m, "_snapshot", lambda e, o, s, i: o.write(f"source_inventory_{i}.json", {"fixture": True})
    )
    packets = {
        n: dict(
            proposal_pose_xyz=np.zeros((b, 3, 3, 3)),
            initial_pose_xyz=np.zeros((3, 3, 3)),
            initial_lengths=np.ones((3, 2)),
        )
        for n, b in zip(m.FRAMES, m.BATCHES, strict=True)
    }
    references = {n: {k: fields77() for k in ("CUDA_golden", "Fast301", "CPU4")} for n in m.FRAMES}
    metadata = dict(
        runtime=dict(
            numpy=np.__version__,
            scipy=m.scipy.__version__,
            opencv=m.cv2.__version__,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
        ),
        opencv_threads=32,
    )
    monkeypatch.setattr(
        m.previous,
        "load_inputs",
        lambda e: (m.frozen.base.TrackerConfig(), None, packets, references, metadata),
    )
    for mod, key in (
        (torch, "set_num_threads"),
        (torch, "set_num_interop_threads"),
        (torch, "manual_seed"),
        (m.cv2, "setNumThreads"),
    ):
        monkeypatch.setattr(mod, key, lambda *a: None)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda *a: (
            (_ for _ in ()).throw(RuntimeError("init failed")) if failure == "init" else None
        ),
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA GeForce RTX 5070 Ti")
    for key in (
        "reset_peak_memory_stats",
        "max_memory_reserved",
        "max_memory_allocated",
        "memory_allocated",
        "memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, key, lambda *a: 0)
    monkeypatch.setattr(
        m.baseline, "gpu_sample", lambda: {"free_MiB": 100 if failure == "headroom" else 9000}
    )
    calls = []

    class Fast:
        arm = "fast"

        def __init__(self, *a):
            self.initialize_count = 0
            self.lengths = np.ones((3, 2))

    class Mixed(Fast):
        arm = "mixed"

    monkeypatch.setattr(m, "FastDirectionalDeviceFitter", Fast)
    monkeypatch.setattr(mixed_module, "MixedPrecisionDirectionalDeviceFitter", Mixed)

    def fitted(fitter, packet):
        calls.append(fitter.arm)
        a = fields77()
        if failure == "fast" and fitter.arm == "fast":
            a["field_0"][1] = np.nextafter(1.0, 2.0)
        if fitter.arm == "mixed":
            fitter.last_mixed_trace = {"fixture": np.zeros(1)}
            fitter.last_mixed_statistics = statistics(calls.count("mixed") - 1)
        return a, 0.1

    monkeypatch.setattr(m.frozen, "fitted_frame", fitted)
    monkeypatch.setattr(
        m,
        "validate_trace",
        lambda *a: (
            (_ for _ in ()).throw(ValueError("mixed bank failed"))
            if failure == "bank"
            else {"fixture": True}
        ),
    )
    monkeypatch.setattr(m, "validate_fp64_rays", lambda *a: {"fixture": True})
    return SimpleNamespace(execute=True, device="cuda", output=tmp_path / "output"), calls


def test_mock_eight_fit_lifecycle_preserves_alternation_and_final_counts(tmp_path, monkeypatch):
    args, calls = mock_run(monkeypatch, tmp_path)
    seal = m.run(args)
    assert calls == ["fast", "mixed", "mixed", "fast", "fast", "mixed", "mixed", "fast"]
    assert seal["status"] == "ok"
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["completed_fit_calls"] == 8 and summary["FP32_warmup_full_fits"] == 9
    assert (
        summary["FP32_graph_constructions"] == 3
        and not summary["quality_or_speed_adoption_claimed"]
    )


@pytest.mark.parametrize(
    "failure,count,attempted",
    [("headroom", 0, False), ("init", 0, True), ("fast", 1, True), ("bank", 2, True)],
)
def test_mock_failure_preserves_evidence_stops_and_never_reports_success(
    tmp_path, monkeypatch, failure, count, attempted
):
    args, calls = mock_run(monkeypatch, tmp_path, failure=failure)
    with pytest.raises((ValueError, RuntimeError)):
        m.run(args)
    seal = json.loads((args.output / "qualification.provenance.json").read_text())
    assert seal["status"] == "failed" and seal["completed_fit_calls"] == len(calls) == count
    assert seal["GPU_initialization_attempted"] is attempted
    if failure in ("headroom", "init"):
        assert not seal["GPU_used"]
    assert "plan.json" in seal["outputs"] and "startup_gpu.json" in seal["outputs"]
    assert not (args.output / "summary.json").exists()

"""Synthetic benchmark contracts; all fitted_frame calls are CPU test doubles."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import benchmark_fast_directional_fitter as m
import numpy as np
import pytest


def golden_arrays():
    arrays = {f"field_{i}": np.array([float(i), i + 0.25]) for i in range(72)}
    arrays.update(
        selected_indices=np.array([0, 1], np.int64),
        original_selected_indices=np.array([1, 0], np.int64),
        mask_supported=np.array([True, False]),
        proposal_kind=np.array(["predict", "tail_recovery"]),
        undefined=np.array([np.nan, np.inf, -np.inf]),
    )
    assert len(arrays) == 77
    return arrays


def packet(index=0):
    return dict(
        frame_idx=np.asarray(m.FRAMES[index], np.int64),
        parent_pose=np.arange(27.0).reshape(3, 3, 3),
    )


def fitters():
    return {arm: SimpleNamespace(arm=arm) for arm in m.ARMS}


def summary(device):
    return dict(
        schema="tail_same_objective_local_fit_v1",
        status="ok",
        device=device,
        frames=list(m.FRAMES if device == "cuda" else m.CPU_FRAMES),
        control_all19_records_reproduced=True,
        independent_frames=True,
        allow_bounded=False,
        target_node=2,
        temporal_tracking_performed=False,
        original_pool_preserved=True,
        CPU_CUDA_all_candidate_frames=list(m.CPU_FRAMES),
    )


@pytest.mark.parametrize("index", [0, 1, 2, 299, 300])
def test_registered_ab_ba_order_compares_all77_and_owns_packet(monkeypatch, index):
    expected, data = golden_arrays(), packet(index)
    before, calls, saved = m.signature(data), [], []

    def fitted(fitter, values):
        assert values is data
        calls.append(fitter.arm)
        arrays = copy.deepcopy(expected)
        if fitter.arm == "fast":
            arrays["field_0"][0] += 5e-11
        return arrays, 1.25 if fitter.arm == "original" else 0.75

    monkeypatch.setattr(m.frozen, "fitted_frame", fitted)
    row = m.paired_frame(
        index,
        fitters(),
        data,
        expected,
        on_result=lambda arm, arrays, detail, seconds: saved.append((arm, len(arrays), seconds)),
    )
    assert calls == list(m.pair_order(index)) == row["order"]
    assert [item[0] for item in saved] == calls
    assert all(item[1] == 77 for item in saved)
    assert row["fit_seconds"] == {"original": 1.25, "fast": 0.75}
    assert len(row["comparisons"]["paired"]) == 77
    for arm in m.ARMS:
        assert len(row["comparisons"][arm]["CUDA_golden"]) == 77
    assert m.signature(data) == before


def test_cpu4_golden_is_checked_for_both_arms_without_running_cpu_fit(monkeypatch):
    expected, calls = golden_arrays(), []

    def fitted(fitter, values):
        calls.append(fitter.arm)
        return copy.deepcopy(expected), 1.0

    monkeypatch.setattr(m.frozen, "fitted_frame", fitted)
    row = m.paired_frame(0, fitters(), packet(), expected, cpu=copy.deepcopy(expected))
    assert calls == ["original", "fast"]
    for arm in m.ARMS:
        assert len(row["comparisons"][arm]["CPU_reference"]) == 77
    with pytest.raises(ValueError, match="CPU4"):
        m.paired_frame(1, fitters(), packet(1), expected, cpu=expected)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "fault", ["float", "decision", "dtype", "shape", "nonfinite", "missing", "extra"]
)
def test_either_arm_cannot_pass_with_any_changed_golden_field(monkeypatch, fault):
    expected = golden_arrays()
    saved, calls = [], []

    def fitted(fitter, values):
        calls.append(fitter.arm)
        result = copy.deepcopy(expected)
        if fitter.arm == "fast":
            if fault == "float":
                result["field_0"][0] = 2e-10
            elif fault == "decision":
                result["selected_indices"] = np.array([1, 0], np.int64)
            elif fault == "dtype":
                result["field_0"] = result["field_0"].astype(np.float32)
            elif fault == "shape":
                result["field_0"] = result["field_0"].reshape(1, 2)
            elif fault == "nonfinite":
                result["undefined"][0] = 0
            elif fault == "missing":
                result.pop("field_0")
            else:
                result["unregistered"] = np.array(1)
        return result, 1.0

    monkeypatch.setattr(m.frozen, "fitted_frame", fitted)
    with pytest.raises(ValueError):
        m.paired_frame(
            0, fitters(), packet(), expected, on_result=lambda arm, *args: saved.append(arm)
        )
    assert calls == ["original", "fast"]
    assert saved == ["original"]


def test_packet_mutation_fails_before_results_publish_or_second_arm(monkeypatch):
    calls, published = [], []

    def fitted(fitter, values):
        calls.append(fitter.arm)
        values["parent_pose"].flat[0] += 1
        return golden_arrays(), 1.0

    monkeypatch.setattr(m.frozen, "fitted_frame", fitted)
    with pytest.raises(ValueError, match="mutated"):
        m.paired_frame(
            0, fitters(), packet(), golden_arrays(), on_result=lambda *args: published.append(True)
        )
    assert calls == ["original"] and not published


def test_budget_failure_stops_without_changing_arm_order_or_retrying(monkeypatch):
    calls, memory_calls = [], []

    def fitted(fitter, values):
        calls.append(fitter.arm)
        return golden_arrays(), 1.0

    def after_fit(arm):
        memory_calls.append(arm)
        m.check_memory(
            dict(
                allocated_bytes=1,
                reserved_bytes=m.MAX_RESERVED_BYTES + 1,
                peak_allocated_bytes=1,
                peak_reserved_bytes=m.MAX_RESERVED_BYTES + 1,
            )
        )

    monkeypatch.setattr(m.frozen, "fitted_frame", fitted)
    with pytest.raises(ValueError, match="4GiB"):
        m.paired_frame(1, fitters(), packet(1), golden_arrays(), after_fit=after_fit)
    assert calls == memory_calls == ["fast"]


def test_timing_keeps_initial_cold_all301_and_remaining300_separate():
    rows = [
        dict(
            frame_idx=native,
            order=list(m.pair_order(i)),
            fit_seconds={"original": 10.0 if i == 0 else 2.0, "fast": 8.0 if i == 0 else 1.0},
        )
        for i, native in enumerate(m.FRAMES)
    ]
    result = m.timing_summary(rows)
    assert result["initial_cold_frame"]["seconds_by_arm"] == {"original": 10.0, "fast": 8.0}
    assert result["all301"]["seconds_by_arm"] == {"original": 610.0, "fast": 308.0}
    assert result["remaining300"]["seconds_by_arm"] == {"original": 600.0, "fast": 300.0}
    assert result["remaining300"]["original_over_fast_ratio"] == 2.0
    assert result["all301"]["median_seconds_by_arm"] == {"original": 2.0, "fast": 1.0}
    assert result["all301"]["paired_median_original_over_fast_ratio"] == 2.0
    assert result["all301"]["fast_faster_frames"] == 301
    assert result["all301"]["fast_slower_frames"] == result["all301"]["equal_frames"] == 0
    assert result["remaining300"]["paired_fast_minus_original_mean_seconds"] == -1.0
    assert any("Concurrent" in line for line in result["limitations"])
    for bad in (rows[:-1], [rows[1], rows[0], *rows[2:]]):
        with pytest.raises(ValueError):
            m.timing_summary(bad)
    rows[0]["order"] = ["fast", "original"]
    with pytest.raises(ValueError):
        m.timing_summary(rows)


def test_paired_medians_and_three_outcome_counts_survive_one_cold_outlier():
    rows = [
        dict(
            frame_idx=native,
            order=list(m.pair_order(i)),
            fit_seconds={
                "original": 10000.0 if i == 0 else 2.0,
                "fast": 8.0 if i == 0 else (1.0, 2.0, 4.0)[(i - 1) % 3],
            },
        )
        for i, native in enumerate(m.FRAMES)
    ]
    result = m.timing_summary(rows)
    assert result["all301"]["median_seconds_by_arm"] == {"original": 2.0, "fast": 2.0}
    assert result["all301"]["paired_median_original_over_fast_ratio"] == 1.0
    assert result["all301"]["fast_faster_frames"] == 101
    assert result["all301"]["fast_slower_frames"] == result["all301"]["equal_frames"] == 100
    assert result["remaining300"]["paired_ratio_defined_frames"] == 300


def test_fit_attempt_is_recorded_when_fitter_raises_without_returning(monkeypatch):
    attempted, returned = [], []

    def failed(*args):
        raise RuntimeError("synthetic fit failure")

    monkeypatch.setattr(m.frozen, "fitted_frame", failed)
    with pytest.raises(RuntimeError, match="synthetic fit failure"):
        m.paired_frame(
            0,
            fitters(),
            packet(),
            golden_arrays(),
            on_attempt=attempted.append,
            after_fit=returned.append,
        )
    assert attempted == ["original"] and returned == []


def test_input_failure_preserves_failed_evidence_without_claiming_cuda(tmp_path, monkeypatch):
    own = str(Path(m.__file__).resolve())
    code = {own: m.checks.sha(own)}
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "SOURCE_PINS", {})
    monkeypatch.setattr(m.base, "local_code", lambda: code.copy())

    def failed_inputs(*args):
        raise ValueError("synthetic input rejection")

    def forbidden(*args, **kwargs):
        pytest.fail("input failure attempted GPU access")

    monkeypatch.setattr(m, "load_evidence", failed_inputs)
    monkeypatch.setattr(m, "gpu_sample", forbidden)
    monkeypatch.setattr(m.torch.cuda, "set_per_process_memory_fraction", forbidden)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="synthetic input rejection"):
        m.main(SimpleNamespace(child_pid=33360, output=output))
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    seal = json.loads((output / "benchmark.provenance.json").read_text(encoding="utf-8"))
    assert result["status"] == seal["status"] == "failed"
    assert result["GPU_used"] is False and result["GPU_initialization_attempted"] is False
    assert result["fit_calls_attempted"] == result["fit_calls_returned"] == 0
    assert result["validated_fit_calls"] == 0
    assert seal["params"]["all_input_source_output_SHA_unchanged"] is True


@pytest.mark.parametrize("startup,boundary", [(True, 6144), (False, 2048)])
def test_global_headroom_thresholds_are_exact(startup, boundary):
    m.check_headroom(dict(total_MiB=16384, used_MiB=16384 - boundary), startup=startup)
    with pytest.raises(ValueError):
        m.check_headroom(dict(total_MiB=16384, used_MiB=16384 - boundary + 1), startup=startup)


def test_memory_checks_cumulative_peak_even_if_current_allocation_shrank():
    valid = dict(
        allocated_bytes=0,
        reserved_bytes=m.MAX_RESERVED_BYTES,
        peak_allocated_bytes=100,
        peak_reserved_bytes=m.MAX_RESERVED_BYTES,
    )
    m.check_memory(valid)
    valid["peak_reserved_bytes"] += 1
    with pytest.raises(ValueError, match="4GiB"):
        m.check_memory(valid)


@pytest.mark.parametrize("ended", [None, "child", "owner"])
def test_startup_owner_check_accepts_preparing_without_wait_or_event_poll(ended):
    calls = []
    guard = SimpleNamespace(
        _attach=lambda: calls.append("attach"),
        child=SimpleNamespace(signalled=lambda: ended == "child"),
        owner=SimpleNamespace(signalled=lambda: ended == "owner"),
    )
    if ended:
        with pytest.raises(ValueError, match="live at startup"):
            m.attach_startup_reference(guard)
    else:
        m.attach_startup_reference(guard)
    assert calls == ["attach"]


@pytest.mark.parametrize("pid", [0, -1, True, "33360", 33361])
def test_unregistered_pid_is_rejected_before_filesystem_or_gpu(pid):
    with pytest.raises(ValueError, match="PID33360"):
        m.main(SimpleNamespace(child_pid=pid))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_saved_gold_requires_complete_same_objective_independent_scope(device):
    value = summary(device)
    m.validate_gold_summary(value, device)
    for key, replacement in (
        ("status", "failed"),
        ("frames", value["frames"][:-1]),
        ("control_all19_records_reproduced", False),
        ("allow_bounded", True),
        ("target_node", 0),
        ("independent_frames", False),
        ("temporal_tracking_performed", True),
    ):
        changed = copy.deepcopy(value)
        changed[key] = replacement
        with pytest.raises(ValueError):
            m.validate_gold_summary(changed, device)

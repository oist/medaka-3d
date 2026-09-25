"""Independent synthetic contracts for the paired full-fit benchmark.

No saved packet, model, fit or CUDA call is used. The frozen array comparator
checks tiny generated arrays; timers and fitters are controlled test doubles.
"""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "labs/20260909_three_body"), str(REPO / "src")]
import benchmark_full_fit_graph301 as m  # noqa: E402


@pytest.fixture(autouse=True)
def no_real_fits_or_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic benchmark tests must not fit or access CUDA")

    monkeypatch.setattr(m.torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(m, "gpu_sample", forbidden)
    monkeypatch.setattr(m, "memory_sample", forbidden)
    monkeypatch.setattr(m.frozen, "fitted_frame", forbidden)


def arrays77():
    result = {f"field_{i}": np.array([float(i), i + 0.25]) for i in range(72)}
    result.update(
        selected_indices=np.array([0, 1], np.int64),
        original_selected_indices=np.array([1, 0], np.int64),
        mask_supported=np.array([True, False]),
        proposal_kind=np.array(["predict", "tail_recovery"]),
        undefined=np.array([np.nan, np.inf, -np.inf]),
    )
    return result


def packet(index=0):
    return {
        "frame_idx": np.array(m.FRAMES[index], np.int64),
        "parent_pose": np.arange(27.0).reshape(3, 3, 3),
    }


def fitters():
    return {arm: SimpleNamespace(arm=arm) for arm in m.ARMS}


@pytest.mark.parametrize("index", [0, 1, 2, 299, 300])
def test_order_saves_each_result_before_its_comparison(monkeypatch, index):
    expected, data = arrays77(), packet(index)
    before, events = m.signature(data), []
    real_compare = m.frozen.compare_fit_arrays
    returned = {}

    def fitted(fitter, values):
        assert values is data
        events.append((fitter.arm, "fit"))
        result = copy.deepcopy(expected)
        returned[fitter.arm] = result
        return result, 1.25 if fitter.arm == "fast" else 0.75

    def saved(arm, values, detail, elapsed):
        assert values is returned[arm]
        assert detail is None
        assert elapsed == (1.25 if arm == "fast" else 0.75)
        events.append((arm, "save"))

    def compared(actual, reference):
        arm = next(key for key, value in returned.items() if value is actual)
        events.append((arm, "compare" if reference is expected else "paired"))
        return real_compare(actual, reference)

    monkeypatch.setattr(m.frozen, "fitted_frame", fitted)
    monkeypatch.setattr(m.frozen, "compare_fit_arrays", compared)
    ticks = iter([10.0, 19.0])
    monkeypatch.setattr(m.time, "perf_counter", lambda: next(ticks))
    row = m.paired_frame(
        index,
        fitters(),
        data,
        expected,
        on_attempt=lambda arm: events.append((arm, "attempt")),
        on_result=saved,
        after_fit=lambda arm: events.append((arm, "resources")),
    )
    order = ("fast", "full") if index % 2 == 0 else ("full", "fast")
    assert events == [
        (arm, action)
        for arm in order
        for action in ("attempt", "fit", "save", "resources", "compare")
    ] + [("full", "paired")]
    assert row["order"] == list(order)
    assert row["frame_idx"] == m.FRAMES[index]
    assert row["fit_seconds"] == {"fast": 1.25, "full": 0.75}
    assert row["pair_wall_seconds"] == 9.0
    assert set(row["comparisons"]) == {"fast", "full", "paired"}
    for detail in (
        row["comparisons"]["fast"]["CUDA_golden"],
        row["comparisons"]["full"]["CUDA_golden"],
        row["comparisons"]["paired"],
    ):
        assert len(detail) == 77
        assert all(item["byte_exact"] for item in detail.values())
    assert m.signature(data) == before


@pytest.mark.parametrize("index", [-1, 301, True, 0.0, np.int64(0)])
def test_order_requires_fixed_python_index(index):
    with pytest.raises(ValueError, match="index"):
        m.pair_order(index)


@pytest.mark.parametrize(
    "fault",
    ["ulp", "signed_zero", "nan_bits", "decision", "missing", "extra", "dtype", "shape"],
)
@pytest.mark.parametrize("index", [0, 1])
def test_first_arm_failure_keeps_its_raw_result_and_resources(monkeypatch, fault, index):
    expected, events, kept = arrays77(), [], []

    def fitted(fitter, values):
        events.append((fitter.arm, "fit"))
        result = copy.deepcopy(expected)
        if fault == "ulp":
            result["field_1"][0] = np.nextafter(1.0, np.inf)
        elif fault == "signed_zero":
            result["field_0"][0] = -0.0
        elif fault == "nan_bits":
            result["undefined"].view(np.uint64)[0] = 0x7FF8000000000001
        elif fault == "decision":
            result["selected_indices"][0] = 1
        elif fault == "missing":
            result.pop("field_71")
        elif fault == "extra":
            result["extra"] = np.array(0)
        elif fault == "dtype":
            result["field_71"] = result["field_71"].astype(np.float32)
        else:
            result["field_71"] = result["field_71"].reshape(1, 2)
        return result, 1.0

    def saved(arm, result, detail, seconds):
        assert detail is None
        kept.append(copy.deepcopy(result))
        events.append((arm, "save"))

    monkeypatch.setattr(m.frozen, "fitted_frame", fitted)
    with pytest.raises(ValueError):
        m.paired_frame(
            index,
            fitters(),
            packet(index),
            expected,
            on_attempt=lambda arm: events.append((arm, "attempt")),
            on_result=saved,
            after_fit=lambda arm: events.append((arm, "resources")),
        )
    first = "fast" if index == 0 else "full"
    assert events == [(first, action) for action in ("attempt", "fit", "save", "resources")]
    assert len(kept) == 1
    assert m.signature(kept[0]) != m.signature(expected)


def test_second_arm_mismatch_preserves_first_success_and_second_result(monkeypatch):
    expected, saved, resources = arrays77(), [], []

    def fitted(fitter, values):
        result = copy.deepcopy(expected)
        if fitter.arm == "full":
            result["field_0"][0] = 1e-11
        return result, 0.5

    monkeypatch.setattr(m.frozen, "fitted_frame", fitted)
    with pytest.raises(ValueError, match="byte level: full"):
        m.paired_frame(
            0,
            fitters(),
            packet(),
            expected,
            on_result=lambda arm, *args: saved.append(arm),
            after_fit=resources.append,
        )
    assert saved == resources == ["fast", "full"]


def test_cpu4_uses_original_tolerance_and_checks_both_arms(monkeypatch):
    expected, reference = arrays77(), arrays77()
    reference["field_0"][0] = 5e-11
    monkeypatch.setattr(m.frozen, "fitted_frame", lambda *args: (copy.deepcopy(expected), 1.0))
    row = m.paired_frame(0, fitters(), packet(), expected, cpu=reference)
    for arm in m.ARMS:
        result = row["comparisons"][arm]["CPU_reference"]
        assert len(result) == 77
        assert result["field_0"] == {
            "byte_exact": False,
            "maximum_absolute_difference": 5e-11,
        }
    reference["field_0"][0] = 2e-10
    with pytest.raises(ValueError, match="numerical mismatch"):
        m.paired_frame(0, fitters(), packet(), expected, cpu=reference)


@pytest.mark.parametrize("fault", ["native", "arms", "golden76", "cpu76", "cpu_nonfixed"])
def test_scope_rejection_precedes_any_fitted_frame(fault):
    index = 1 if fault == "cpu_nonfixed" else 0
    data, engines, golden = packet(index), fitters(), arrays77()
    cpu = arrays77() if fault.startswith("cpu") else None
    if fault == "native":
        data["frame_idx"] += 1
    elif fault == "arms":
        engines.pop("full")
    elif fault == "golden76":
        golden.pop("field_0")
    elif fault == "cpu76":
        cpu.pop("field_0")
    with pytest.raises(ValueError):
        m.paired_frame(index, engines, data, golden, cpu=cpu)


@pytest.mark.parametrize("fault", ["packet", "nan_time", "negative_time", "resource"])
def test_returned_data_saved_before_noncomparison_failures(monkeypatch, fault):
    expected, saved, resource_calls, calls = arrays77(), [], [], []

    def fitted(fitter, values):
        calls.append(fitter.arm)
        if fault == "packet":
            values["parent_pose"][0, 0, 0] = 9
        elapsed = np.nan if fault == "nan_time" else (-1 if fault == "negative_time" else 1)
        return copy.deepcopy(expected), elapsed

    def resource(arm):
        resource_calls.append(arm)
        if fault == "resource":
            raise ValueError("synthetic process cap")

    monkeypatch.setattr(m.frozen, "fitted_frame", fitted)
    with pytest.raises(ValueError):
        m.paired_frame(
            0,
            fitters(),
            packet(),
            expected,
            on_result=lambda arm, *args: saved.append(arm),
            after_fit=resource,
        )
    assert calls == saved == resource_calls == ["fast"]


def test_raised_fit_records_attempt_without_claiming_return(monkeypatch):
    error, events = RuntimeError("synthetic allocation failure"), []

    def fitted(*args):
        raise error

    monkeypatch.setattr(m.frozen, "fitted_frame", fitted)
    with pytest.raises(RuntimeError) as caught:
        m.paired_frame(
            0,
            fitters(),
            packet(),
            arrays77(),
            on_attempt=lambda arm: events.append((arm, "attempt")),
            on_result=lambda arm, *args: events.append((arm, "return")),
            after_fit=lambda arm: events.append((arm, "resources")),
        )
    assert caught.value is error
    assert events == [("fast", "attempt")]


def timing_rows():
    rows = []
    for i, native in enumerate(m.FRAMES):
        fast, full = (1000.0, 2000.0) if i == 0 else (2.0, (1.0, 2.0, 4.0)[(i - 1) % 3])
        rows.append(
            dict(
                frame_idx=native,
                order=list(m.pair_order(i)),
                fit_seconds=dict(fast=fast, full=full),
            )
        )
    return rows


def test_cold_first_does_not_dominate_remaining300_medians_or_pair_counts():
    result = m.timing_summary(timing_rows())
    cold, rest, all_frames = (result[k] for k in ("initial_cold_frame", "remaining300", "all301"))
    assert cold["frames"] == 1 and cold["fast_over_full_ratio"] == 0.5
    assert rest["frames"] == 300 and all_frames["frames"] == 301
    assert rest["seconds_by_arm"] == {"fast": 600.0, "full": 700.0}
    assert all_frames["seconds_by_arm"] == {"fast": 1600.0, "full": 2700.0}
    assert rest["median_seconds_by_arm"] == {"fast": 2.0, "full": 2.0}
    assert rest["mean_seconds_by_arm"] == {"fast": 2.0, "full": 700 / 300}
    assert rest["paired_median_fast_over_full_ratio"] == 1.0
    assert rest["paired_ratio_defined_frames"] == 300
    assert rest["full_faster_frames"] == rest["full_slower_frames"] == rest["equal_frames"] == 100
    assert all_frames["full_slower_frames"] == 101
    assert rest["paired_full_minus_fast_mean_seconds"] == pytest.approx(1 / 3)


def test_zero_duration_keeps_denominators_and_undefined_ratios_explicit():
    rows = timing_rows()
    for row in rows:
        row["fit_seconds"] = {"fast": 0.0, "full": 0.0}
    result = m.timing_summary(rows)["all301"]
    assert result["paired_ratio_defined_frames"] == 0
    assert result["paired_median_fast_over_full_ratio"] is None
    assert result["fast_over_full_ratio"] is None
    assert result["reduction_fraction_vs_fast"] is None
    assert result["equal_frames"] == 301
    rows[-1]["fit_seconds"] = {"fast": 2.0, "full": 1.0}
    result = m.timing_summary(rows)["all301"]
    assert result["paired_ratio_defined_frames"] == 1
    assert result["paired_median_fast_over_full_ratio"] == 2.0
    assert result["equal_frames"] == 300


@pytest.mark.parametrize("fault", ["missing", "duplicate", "order", "negative", "nan"])
def test_timing_rejects_scope_order_and_invalid_durations(fault):
    rows = timing_rows()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = copy.deepcopy(rows[-2])
    elif fault == "order":
        rows[1]["order"] = ["fast", "full"]
    else:
        rows[1]["fit_seconds"]["full"] = -1.0 if fault == "negative" else np.nan
    with pytest.raises(ValueError):
        m.timing_summary(rows)


def synthetic_trace(batch=6):
    # Independent fixed15/14 schema, without calling the implementation's layout.
    def pose(steps):
        return np.zeros((steps, batch, 3, 3, 3), np.float64)

    return dict(
        pose_before=pose(15),
        score=np.zeros((15, batch), np.float64),
        parts=np.zeros((15, batch, 5), np.float64),
        base_gradient=pose(15),
        update_gradient=pose(14),
        moment=pose(14),
        variance=pose(14),
        pose_after=pose(14),
        best_iteration=np.arange(batch, dtype=np.int64) % 15,
        best_score=np.zeros(batch, np.float64),
    )


@pytest.mark.parametrize("batch", [6, 17, 30])
def test_full_trace_capacity_includes_every_gradient_moment_and_best(batch):
    trace = synthetic_trace(batch)
    per_candidate = 8 * (15 * (27 + 1 + 5 + 27) + 14 * (27 * 4) + 2)
    assert per_candidate == 19312
    assert m.trace_bytes(batch) == m.validate_trace(trace, batch) == per_candidate * batch
    assert 4949 * per_candidate == 95_575_088


@pytest.mark.parametrize("batch", [0, 5, 31, True, 6.0, np.int64(6)])
def test_trace_rejects_unregistered_candidate_budget(batch):
    with pytest.raises(ValueError):
        m.trace_bytes(batch)


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "update13", "float32", "nan", "inf", "best_negative", "best15"]
)
def test_trace_rejects_partial_or_invalid_returned_trajectory(fault):
    trace = synthetic_trace()
    if fault == "missing":
        trace.pop("variance")
    elif fault == "extra":
        trace["extra"] = np.zeros(1)
    elif fault == "update13":
        trace["update_gradient"] = trace["update_gradient"][:13]
    elif fault == "float32":
        trace["moment"] = trace["moment"].astype(np.float32)
    elif fault in ("nan", "inf"):
        trace["base_gradient"][2, 3, 1, 0, 2] = np.nan if fault == "nan" else np.inf
    else:
        trace["best_iteration"][0] = -1 if fault == "best_negative" else 15
    with pytest.raises(ValueError):
        m.validate_trace(trace, 6)


@pytest.mark.parametrize("startup,limit", [(True, 6144), (False, 2048)])
def test_actual_free_memory_is_guarded_at_each_scope(startup, limit):
    sample = {"total_MiB": 16000, "used_MiB": 1000, "free_MiB": limit}
    m.check_headroom(sample, startup=startup)
    for free in (limit - 1, np.nan, np.inf, -1):
        sample["free_MiB"] = free
        with pytest.raises(ValueError):
            m.check_headroom(sample, startup=startup)


def test_both_live_caches_and_peaks_share_the_four_gib_bound():
    limit = 4 * 1024**3
    assert m.ALLOCATOR_FRACTION == 0.25
    assert limit == m.MAX_RESERVED_BYTES
    memory = dict(
        allocated_bytes=limit,
        reserved_bytes=limit,
        peak_allocated_bytes=limit,
        peak_reserved_bytes=limit,
    )
    m.check_memory(memory)
    for field in memory:
        changed = {**memory, field: limit + 1}
        with pytest.raises(ValueError):
            m.check_memory(changed)
    with pytest.raises(ValueError):
        m.check_memory({**memory, "allocated_bytes": -1})

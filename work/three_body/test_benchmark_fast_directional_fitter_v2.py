"""Resource revision tests using synthetic files, arrays and GPU telemetry only."""

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import benchmark_fast_directional_fitter as old
import benchmark_fast_directional_fitter_v2 as m
import pytest
from test_benchmark_fast_directional_fitter import fitters, golden_arrays, packet


def test_all_original_nonresource_functions_have_identical_ast():
    previous = ast.parse(Path(old.__file__).read_text(encoding="utf-8"))
    current = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
    functions = {node.name: node for node in current.body if isinstance(node, ast.FunctionDef)}
    checked = set()
    for node in previous.body:
        if isinstance(node, ast.FunctionDef) and node.name not in {
            "check_headroom",
            "check_memory",
            "main",
        }:
            assert ast.dump(node) == ast.dump(functions[node.name]), node.name
            checked.add(node.name)
    assert {"paired_frame", "timing_summary", "load_evidence", "pair_order"} <= checked
    for name in (
        "FRAMES",
        "CPU_FRAMES",
        "ARMS",
        "PACKETS",
        "GOLDEN",
        "CPU_REFERENCE",
        "PACKET_SHA",
        "GOLDEN_SHA",
        "CPU_SHA",
        "REFERENCE_CHILD_PID",
    ):
        assert getattr(m, name) == getattr(old, name)
    assert m.frozen.fitted_frame is old.frozen.fitted_frame
    assert m.frozen.compare_fit_arrays is old.frozen.compare_fit_arrays
    for name in old.SOURCE_PINS:
        assert m.SOURCE_PINS[name] == old.SOURCE_PINS[name]
    assert m.checks.sha(old.__file__) == m.SOURCE_PINS["benchmark_fast_directional_fitter.py"]


def test_main_fitter_construction_and_full_pair_loop_change_only_resource_recording():
    def relevant(function):
        tree = ast.parse(inspect.getsource(function))
        pair_loop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Tuple)
            and ast.unparse(node.target) == "(index, native)"
        )
        # Telemetry is now retained before headroom rejection; all fit/cache/callback
        # statements must remain identical and in the same order.
        pair_loop.body = [
            node
            for node in pair_loop.body
            if not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) in {"check_headroom", "telemetry.append"}
            )
        ]
        construction = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "fitters" for target in node.targets
            )
        )
        return ast.dump(construction), ast.dump(pair_loop)

    assert relevant(m.main) == relevant(old.main)


def test_gpu_sample_uses_measured_free_and_hidden_single_device_query(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="NVIDIA GeForce RTX 5070 Ti, 16303, 7527, 8471, 72, 201.5\n")

    monkeypatch.setattr(m.subprocess, "run", run)
    sample = m.gpu_sample()
    assert sample["total_MiB"] == 16303 and sample["used_MiB"] == 7527
    assert sample["free_MiB"] == 8471
    assert sample["free_MiB"] != sample["total_MiB"] - sample["used_MiB"]
    assert sample["utilization_percent"] == 72 and sample["power_W"] == 201.5
    assert "memory.free" in calls[0][0][1]
    assert calls[0][1] == dict(
        check=True,
        capture_output=True,
        text=True,
        creationflags=getattr(m.subprocess, "CREATE_NO_WINDOW", 0),
    )


@pytest.mark.parametrize(
    "output",
    [
        "Other GPU, 16000, 7000, 8500, 70, 200\n",
        "NVIDIA GeForce RTX 5070 Ti, 16000, 7000, 70, 200\n",
        "NVIDIA GeForce RTX 5070 Ti, 16000, 7000, 8500, 70, 200\n" * 2,
    ],
)
def test_gpu_sample_rejects_wrong_device_or_missing_free_or_multiple_devices(monkeypatch, output):
    monkeypatch.setattr(m.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=output))
    with pytest.raises(ValueError, match="fixed single GPU"):
        m.gpu_sample()


@pytest.mark.parametrize("startup,boundary", [(True, 8192), (False, 2048)])
def test_measured_free_threshold_rejects_total_minus_used_false_pass(startup, boundary):
    sample = dict(total_MiB=16303, used_MiB=16303 - boundary - 305, free_MiB=boundary)
    m.check_headroom(sample, startup=startup)
    sample["free_MiB"] -= 1
    assert sample["total_MiB"] - sample["used_MiB"] > boundary
    with pytest.raises(ValueError, match=f"{boundary}MiB"):
        m.check_headroom(sample, startup=startup)
    for invalid in (float("nan"), float("inf"), -1):
        sample["free_MiB"] = invalid
        with pytest.raises(ValueError):
            m.check_headroom(sample, startup=startup)


def test_six_gib_current_and_peak_caps_remain_hard_bounds():
    assert m.ALLOCATOR_FRACTION == 0.40
    assert m.MAX_RESERVED_BYTES == 6 * 1024**3
    memory = dict(
        allocated_bytes=5918162944,
        reserved_bytes=m.MAX_RESERVED_BYTES,
        peak_allocated_bytes=5918162944,
        peak_reserved_bytes=m.MAX_RESERVED_BYTES,
    )
    m.check_memory(memory)
    for key in ("reserved_bytes", "peak_reserved_bytes"):
        invalid = memory.copy()
        invalid[key] += 1
        with pytest.raises(ValueError, match="6GiB"):
            m.check_memory(invalid)
    memory.update(allocated_bytes=0, reserved_bytes=0, peak_reserved_bytes=m.MAX_RESERVED_BYTES + 1)
    with pytest.raises(ValueError, match="6GiB"):
        m.check_memory(memory)


def test_v2_cap_failure_has_no_retry_or_altered_order(monkeypatch):
    calls, attempted = [], []

    def fitted(fitter, values):
        calls.append(fitter.arm)
        return golden_arrays(), 1.0

    def after_fit(arm):
        m.check_memory(
            dict(
                allocated_bytes=1,
                reserved_bytes=1,
                peak_allocated_bytes=1,
                peak_reserved_bytes=m.MAX_RESERVED_BYTES + 1,
            )
        )

    monkeypatch.setattr(m.frozen, "fitted_frame", fitted)
    with pytest.raises(ValueError, match="6GiB"):
        m.paired_frame(
            1,
            fitters(),
            packet(1),
            golden_arrays(),
            after_fit=after_fit,
            on_attempt=attempted.append,
        )
    assert calls == attempted == ["fast"]


def failure_fixture(tmp_path, monkeypatch):
    root = tmp_path / "failed_v1"
    root.mkdir()
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "FAILED_V1", root)
    summary = dict(
        schema=old.STAGE,
        status="failed",
        passed=False,
        frames=list(m.FRAMES),
        completed_pairs=41,
        completed_arms=82,
        fit_calls_attempted=83,
        fit_calls_returned=82,
        validated_fit_calls=82,
        GPU_used=True,
        error="OutOfMemoryError('CUDA allocator cap')",
    )
    source_name = "benchmark_fast_directional_fitter.py"
    source = tmp_path / source_name
    source.write_text("frozen synthetic driver", encoding="utf-8")
    monkeypatch.setattr(m, "SOURCE_PINS", {source_name: m.checks.sha(source)})
    inputs = [source]
    for constant, filename in (
        ("PACKETS", "preparation.provenance.json"),
        ("GOLDEN", "fit.provenance.json"),
        ("CPU_REFERENCE", "fit.provenance.json"),
    ):
        directory = tmp_path / constant.lower()
        directory.mkdir()
        path = directory / filename
        path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(m, constant, directory)
        inputs.append(path)
    seal = dict(
        stage=old.STAGE,
        status="failed",
        inputs={str(p): dict(sha256=m.checks.sha(p)) for p in inputs},
        outputs={},
        params=dict(allocator_fraction_cap=0.25, maximum_process_reserved_bytes=4 * 1024**3),
    )
    payloads = {"summary.json": summary, **{f"retained_{i}.json": {"row": i} for i in range(172)}}
    for name, value in payloads.items():
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        seal["outputs"][name] = dict(
            path=str(path), size=path.stat().st_size, sha256=m.checks.sha(path)
        )
    seal_path = root / "benchmark.provenance.json"

    def reseal():
        path = root / "summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        seal["outputs"]["summary.json"].update(size=path.stat().st_size, sha256=m.checks.sha(path))
        seal_path.write_text(json.dumps(seal), encoding="utf-8")
        monkeypatch.setattr(m, "FAILED_V1_SHA", m.checks.sha(seal_path))

    reseal()
    return root, source, summary, seal, reseal


def test_failed_seal_binds_all_outputs_and_inputs_without_loading_fit_arrays(tmp_path, monkeypatch):
    root, source, _, _, _ = failure_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(m.base, "load_arrays", lambda *args: pytest.fail("failed arrays reused"))
    hashes = {}
    m.verify_failed_v1(hashes, {str(source): m.checks.sha(source)})
    assert len(hashes) == 173 + 1 + 4
    assert hashes[str(root / "benchmark.provenance.json")] == m.FAILED_V1_SHA
    assert all(m.checks.sha(path) == digest for path, digest in hashes.items())


@pytest.mark.parametrize("fault", ["seal", "output", "source", "summary", "status", "budget"])
def test_failed_v1_drift_cannot_authorize_resource_revision(tmp_path, monkeypatch, fault):
    root, source, summary, seal, reseal = failure_fixture(tmp_path, monkeypatch)
    if fault == "seal":
        (root / "benchmark.provenance.json").write_text("{}", encoding="utf-8")
    elif fault == "output":
        (root / "retained_99.json").write_text("changed", encoding="utf-8")
    elif fault == "source":
        source.write_text("changed source", encoding="utf-8")
    elif fault == "summary":
        summary["validated_fit_calls"] = 81
        reseal()
    elif fault == "status":
        seal["status"] = "ok"
        reseal()
    else:
        seal["params"]["allocator_fraction_cap"] = 0.4
        reseal()
    with pytest.raises(ValueError):
        m.verify_failed_v1({}, {})


def test_failed_v1_rejection_is_preserved_before_cuda_or_golden_loading(tmp_path, monkeypatch):
    own = str(Path(m.__file__).resolve())
    code = {own: m.checks.sha(own)}
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "SOURCE_PINS", {})
    monkeypatch.setattr(m.base, "local_code", lambda: code.copy())

    def reject(*args):
        raise ValueError("synthetic failed-v1 rejection")

    def forbidden(*args, **kwargs):
        pytest.fail("failed-v1 rejection continued to GPU or golden inputs")

    monkeypatch.setattr(m, "verify_failed_v1", reject)
    monkeypatch.setattr(m, "load_evidence", forbidden)
    monkeypatch.setattr(m, "gpu_sample", forbidden)
    monkeypatch.setattr(m.torch.cuda, "set_per_process_memory_fraction", forbidden)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="synthetic failed-v1"):
        m.main(SimpleNamespace(child_pid=33360, output=output))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    seal = json.loads((output / "benchmark.provenance.json").read_text(encoding="utf-8"))
    assert summary["status"] == seal["status"] == "failed"
    assert summary["GPU_used"] is summary["GPU_initialization_attempted"] is False
    assert summary["validated_fit_calls"] == summary["fit_calls_attempted"] == 0
    assert summary["resource_revision"]["previous_fit_results_reused"] == 0
    assert summary["resource_revision"]["all301_pairs_run_afresh"] is True


def test_startup_actual_free_failure_records_measurement_and_never_initializes_cuda(
    tmp_path, monkeypatch
):
    own = str(Path(m.__file__).resolve())
    code = {own: m.checks.sha(own)}
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "SOURCE_PINS", {})
    monkeypatch.setattr(m.base, "local_code", lambda: code.copy())
    monkeypatch.setattr(m, "verify_failed_v1", lambda *args: None)
    monkeypatch.setattr(m, "load_evidence", lambda *args: (None, None, None, None, None, None))
    calls = []
    guard = SimpleNamespace(
        _attach=lambda: calls.append("attach"),
        child=SimpleNamespace(signalled=lambda: False),
        owner=SimpleNamespace(signalled=lambda: False),
        close=lambda: calls.append("close"),
    )
    monkeypatch.setattr(m, "ReplayGPUWindow", lambda *args: guard)
    sample = dict(total_MiB=16303, used_MiB=7830, free_MiB=8168)
    monkeypatch.setattr(m, "gpu_sample", lambda: sample.copy())
    monkeypatch.setattr(
        m.torch.cuda,
        "set_per_process_memory_fraction",
        lambda *args: pytest.fail("insufficient actual free initialized CUDA"),
    )
    output = tmp_path / "headroom_failure"
    with pytest.raises(ValueError, match="8192MiB"):
        m.main(SimpleNamespace(child_pid=33360, output=output, supervisor_root=tmp_path))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    telemetry = json.loads((output / "GPU_telemetry.json").read_text(encoding="utf-8"))
    assert summary["GPU_used"] is summary["GPU_initialization_attempted"] is False
    assert summary["fit_calls_attempted"] == 0 and calls == ["attach", "close"]
    assert telemetry == [dict(stage="startup", **sample)]


def test_resource_summary_keeps_failed_v1_and_fresh_complete_scope():
    assert m.RESOURCE_REVISION["failed_v1_seal_sha256"] == m.FAILED_V1_SHA
    assert m.RESOURCE_REVISION["failed_v1_completed_pairs"] == 41
    assert m.RESOURCE_REVISION["failed_v1_attempted_calls"] == 83
    assert m.RESOURCE_REVISION["allocator_fraction_cap"] == 0.4
    assert m.RESOURCE_REVISION["maximum_process_reserved_bytes"] == 6 * 1024**3
    assert len(m.FRAMES) == 301 and m.STAGE != old.STAGE

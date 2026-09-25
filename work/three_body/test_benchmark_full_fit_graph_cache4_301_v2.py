"""Only the new resource bounds and failed-input/fresh-run boundaries."""

# ruff: noqa: E402
import ast
import sys
from pathlib import Path

import pytest

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"),
    str(Path(__file__).resolve().parents[2] / "src"),
]
import benchmark_full_fit_graph_cache4_301_v2 as m


def test_only_registered_resource_limits_change():
    assert m.ALLOCATOR_FRACTION == 0.375
    assert m.MAX_RESERVED_BYTES == 6 * 1024**3
    assert (m.STARTUP_FREE_MIB, m.ARM_FREE_MIB) == (8192, 2048)
    m.check_headroom({"free_MiB": 8192}, startup=True)
    with pytest.raises(ValueError):
        m.check_headroom({"free_MiB": 8191}, startup=True)
    m.check_headroom({"free_MiB": 2048})
    with pytest.raises(ValueError):
        m.check_headroom({"free_MiB": 2047})
    memory = dict(
        allocated_bytes=5 * 1024**3,
        reserved_bytes=6 * 1024**3,
        peak_allocated_bytes=5 * 1024**3,
        peak_reserved_bytes=6 * 1024**3,
    )
    m.check_memory(memory)
    memory["peak_reserved_bytes"] += 1
    with pytest.raises(ValueError):
        m.check_memory(memory)


@pytest.fixture
def failed_input(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "FAILED_RUN", tmp_path)
    error = (
        "CUDA out of memory. Tried to allocate 34.00 MiB. 10.56 GiB is free. "
        "3.97 GiB allowed; 3.92 GiB allocated, with 3.70 GiB private pools"
    )
    inputs = {str(m.REPO / key): sha for key, sha in m.SOURCE_PINS.items()}
    for root, filename, sha in (
        (m.QUALIFICATION, "qualification.provenance.json", m.QUALIFICATION_SHA),
        (m.CACHE4_QUALIFICATION, "qualification.provenance.json", m.CACHE4_QUALIFICATION_SHA),
        (m.baseline.PACKETS, "preparation.provenance.json", m.baseline.PACKET_SHA),
        (m.baseline.GOLDEN, "fit.provenance.json", m.baseline.GOLDEN_SHA),
        (m.baseline.CPU_REFERENCE, "fit.provenance.json", m.baseline.CPU_SHA),
    ):
        inputs[str(root / filename)] = sha
    seal = dict(
        stage="full_fit_graph_cache4_paired_301_v1",
        status="failed",
        finished_at="done",
        error_type="OutOfMemoryError",
        error=error,
        completed_pairs=41,
        completed_arms=82,
        inputs=inputs,
        outputs={},
    )
    summary = dict(
        status="failed",
        passed=False,
        error_type="OutOfMemoryError",
        error=error,
        completed_pairs=41,
        completed_arms=82,
        fit_calls_attempted=83,
        fit_calls_returned=82,
        validated_fit_calls=82,
        frames=list(m.FRAMES),
        GPU_used=True,
        GPU_initialization_attempted=True,
    )
    attempts = [
        dict(frame_idx=native, arm=arm, fitted_frame_returned=True)
        for i, native in enumerate(m.FRAMES[:41])
        for arm in m.pair_order(i)
    ]
    attempts.append(
        dict(frame_idx=m.FRAMES[41], arm=m.pair_order(41)[0], fitted_frame_returned=False)
    )
    documents = {"summary.json": summary, "fit_attempts.json": attempts}
    for name, doc in documents.items():
        path = tmp_path / name
        path.write_bytes(m.proof.json_bytes(doc))
        seal["outputs"][name] = dict(
            path=str(path), size_bytes=path.stat().st_size, sha256=m.proof.sha(path)
        )
    original_read = m.proof.read_json
    monkeypatch.setattr(
        m.proof,
        "read_json",
        lambda path: (
            seal
            if Path(path).name == "benchmark.provenance.json"
            else documents[Path(path).name]
            if Path(path).parent == tmp_path
            else original_read(path)
        ),
    )
    seen = []

    class Evidence:
        def add(self, path, sha):
            seen.append((path, sha))

    return seal, summary, attempts, Evidence(), seen


def test_failed_gate_keeps_truthful_counts_and_fresh_full301(failed_input):
    _, _, _, evidence, seen = failed_input
    result = m.resource_revision_gate(evidence)
    assert result["completed_pairs"] == 41 and result["completed_fit_returns"] == 82
    assert result["fit_attempts"] == 83 and result["failed_native"] == 66660
    assert result["failed_arm"] == "full"
    assert result["all301_pairs_reexecution_planned"] is True
    assert result["newly_fitted_calls_planned"] == 602 and result["failed_fit_arrays_reused"] == 0
    assert {p.suffix for p, _ in seen} == {".json", ".py"}


@pytest.mark.parametrize("mismatch", ["success", "return_count", "last_attempt", "source", "input"])
def test_failed_gate_rejects_incompatible_provenance(failed_input, mismatch):
    seal, summary, attempts, evidence, _ = failed_input
    if mismatch == "success":
        seal["status"] = "ok"
    elif mismatch == "return_count":
        summary["fit_calls_returned"] = 83
    elif mismatch == "last_attempt":
        attempts[-1]["fitted_frame_returned"] = True
    elif mismatch == "source":
        seal["inputs"][str(m.WORK / "benchmark_full_fit_graph_cache4_301.py")] = "0" * 64
    else:
        seal["inputs"][str(m.baseline.PACKETS / "preparation.provenance.json")] = "0" * 64
    with pytest.raises(ValueError):
        m.resource_revision_gate(evidence)


def test_all_numerical_and_cache_functions_stay_identical():
    def functions(path):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef)
        }

    old_path = Path(m.__file__).with_name("benchmark_full_fit_graph_cache4_301.py")
    old, new = functions(old_path), functions(Path(m.__file__))
    assert {name for name in old.keys() & new.keys() if old[name] != new[name]} == {
        "run",
        "check_memory",
    }
    assert new.keys() - old.keys() == {"resource_revision_gate"}
    text = old_path.read_text().replace("fixed 4GiB process bound", "fixed 6GiB process bound")
    expected = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == "check_memory"
    )
    assert ast.dump(expected, include_attributes=False) == new["check_memory"]

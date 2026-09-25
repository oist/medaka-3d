"""New cache4 caller boundaries only; no actual packets, GPU or old suite rerun."""

# ruff: noqa: E402
import ast
import copy
import sys
from pathlib import Path

import pytest

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"),
    str(Path(__file__).resolve().parents[2] / "src"),
]
import benchmark_full_fit_graph_cache4_301 as m


def test_registered_fill_evict_hit_recapture_and_repeated_shape():
    p = m.cache_prediction([15, 14, 16, 18, 17, 16, 15, 16])
    assert [s["captured"] for s in p["steps"]] == [True] * 5 + [False, True, False]
    assert p["final_statistics"] == dict(
        capacity=4,
        captures=6,
        misses=6,
        hits=2,
        evictions=2,
        failures=0,
        resident=4,
        peak_resident=4,
    )
    repeat = m.cache_prediction([15] * 301)
    assert repeat["final_statistics"] == dict(
        capacity=4,
        captures=1,
        misses=1,
        hits=300,
        evictions=0,
        failures=0,
        resident=1,
        peak_resident=1,
    )
    assert all(s["statistics"]["peak_resident"] == 1 for s in repeat["steps"])


def test_expected_four_residencies_not_single_graph_for_all301():
    p = m.cache_prediction(([14, 15, 16, 17] * 76)[:301])
    assert p["final_statistics"]["captures"] == 4
    assert p["final_statistics"]["hits"] == 297
    assert p["final_statistics"]["evictions"] == 0
    # Every prefix is an owned value; adjusting the final summary cannot alter it.
    p["final_statistics"]["hits"] = -1
    assert p["steps"][-1]["statistics"]["hits"] == 297


@pytest.mark.parametrize("counts", [[], [True], [5], [31], [15.0]])
def test_prediction_rejects_outside_frozen_candidate_budget(counts):
    with pytest.raises(ValueError):
        m.cache_prediction(counts)


@pytest.mark.parametrize("field", ["captures", "hits", "evictions", "resident", "failures"])
def test_actual_statistics_must_match_every_prefix(field):
    step = m.cache_prediction([15, 14, 16, 18, 17])["steps"][-1]
    actual = copy.deepcopy(step["statistics"])
    m.validate_cache_statistics(actual, step)
    actual[field] += 1
    with pytest.raises(ValueError, match="history differs"):
        m.validate_cache_statistics(actual, step)


@pytest.fixture
def qualified_cache4(tmp_path, monkeypatch):
    root = tmp_path / "cache4_qualification"
    root.mkdir()
    monkeypatch.setattr(m, "CACHE4_QUALIFICATION", root)
    pins = {"synthetic_adapter.py": "a" * 64}
    monkeypatch.setattr(m.cache_qualified, "SOURCE_PINS", pins)
    summary = dict(
        passed=True,
        all77_CUDA_byte_exact=True,
        frames=list(m.cache_qualified.FRAMES),
        sequence_batches=list(m.cache_qualified.BATCHES),
        completed_fit_calls=8,
        actual_graph_constructions=6,
        initialize_calls=0,
        advance_calls=0,
        raw_decode=0,
        cache_statistics=m.cache_prediction(list(m.cache_qualified.BATCHES))["final_statistics"],
    )
    path = root / "summary.json"
    path.write_bytes(m.proof.json_bytes(summary))
    caller = "work/three_body/qualify_full_fit_graph_cache4.py"
    inputs = {str(m.REPO / key): digest for key, digest in pins.items()}
    inputs[str(m.REPO / caller)] = m.SOURCE_PINS[caller]
    inputs[str(m.cache_qualified.QUAL4 / "qualification.provenance.json")] = (
        m.cache_qualified.QUAL4_SHA
    )
    inputs[str(m.cache_qualified.QUAL301 / "benchmark.provenance.json")] = (
        m.cache_qualified.QUAL301_SHA
    )
    record = dict(
        inputs=inputs,
        outputs={
            "summary.json": dict(
                path=str(path),
                size_bytes=path.stat().st_size,
                sha256=m.proof.sha(path),
            )
        },
    )
    seen = []

    def header(got_root, name, digest, stage, evidence):
        assert got_root == root and name == "qualification.provenance.json"
        assert digest == m.CACHE4_QUALIFICATION_SHA and stage == m.cache_qualified.STAGE
        return record

    monkeypatch.setattr(m.cache_qualified, "sealed_header", header)

    class Evidence:
        def add(self, path, digest):
            if path == root / "summary.json":
                assert m.proof.sha(path) == digest
            seen.append((path, digest))

    return record, Evidence(), seen


def test_cache4_gate_reads_named_summary_and_binds_actual_adapter(qualified_cache4):
    record, evidence, seen = qualified_cache4
    summary = m.cache4_qualification_gate(evidence)
    assert summary["completed_fit_calls"] == 8
    assert len(seen) == 2  # One small summary, one consumed source; no old payload reopening.
    assert seen[-1] == (m.REPO / "synthetic_adapter.py", "a" * 64)
    assert len(record["outputs"]) == 1


@pytest.mark.parametrize("edge", ["adapter", "caller", "four_packet", "paired301"])
def test_cache4_gate_rejects_unqualified_source_or_prior_success_edge(qualified_cache4, edge):
    record, evidence, _ = qualified_cache4
    paths = dict(
        adapter=m.REPO / "synthetic_adapter.py",
        caller=m.REPO / "work/three_body/qualify_full_fit_graph_cache4.py",
        four_packet=m.cache_qualified.QUAL4 / "qualification.provenance.json",
        paired301=m.cache_qualified.QUAL301 / "benchmark.provenance.json",
    )
    record["inputs"][str(paths[edge])] = "b" * 64
    with pytest.raises(ValueError, match="differs"):
        m.cache4_qualification_gate(evidence)


def test_copied_scientific_functions_remain_identical():
    def functions(path):
        tree = ast.parse(path.read_text())
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }

    old = functions(Path(m.__file__).with_name("benchmark_full_fit_graph301.py"))
    new = functions(Path(m.__file__))
    changed = {name for name in old.keys() & new.keys() if old[name] != new[name]}
    assert changed == {"run", "timing_summary"}
    assert new.keys() - old.keys() == {
        "cache_prediction",
        "validate_cache_statistics",
        "cache4_qualification_gate",
    }
    # The timing function change is only this truthful cache-capacity description.
    old_text = Path(m.__file__).with_name("benchmark_full_fit_graph301.py").read_text()
    old_text = old_text.replace(
        "Remaining300 can contain new captures; single Full cache differs from Fast cache4.",
        "Remaining300 can contain new captures; both arms use capacity4 LRU caches.",
    )
    expected = next(
        n
        for n in ast.parse(old_text).body
        if isinstance(n, ast.FunctionDef) and n.name == "timing_summary"
    )
    assert ast.dump(expected, include_attributes=False) == new["timing_summary"]

"""Cache exercise integration and fixed-input comparison boundaries, no GPU."""

# ruff: noqa: E402
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"),
    str(Path(__file__).resolve().parents[2] / "src"),
]
import qualify_full_fit_graph_cache4 as m


def test_fixed_eight_steps_exercise_hits_eviction_and_different_packet_configure():
    assert m.FRAMES == (66524, 66628, 66703, 66639, 66655, 66703, 66524, 66907)
    assert m.BATCHES == (15, 14, 16, 18, 17, 16, 15, 16)
    assert len(set(m.FRAMES)) == 6
    cache, hits, misses, evictions = [], 0, 0, 0
    for index, shape in enumerate(m.BATCHES):
        if shape in cache:
            hits += 1
            cache.remove(shape)
        else:
            misses += 1
            if len(cache) == 4:
                cache.pop(0)
                evictions += 1
        cache.append(shape)
        stats = dict(
            captures=misses,
            misses=misses,
            hits=hits,
            evictions=evictions,
            failures=0,
            resident=len(cache),
            peak_resident=min(index + 1, 4),
            capacity=4,
        )
        assert m.validate_cache_step(index, stats) == stats
    assert (hits, misses, evictions) == (2, 6, 2)
    assert m.FRAMES[-1] != m.FRAMES[-3] and m.BATCHES[-1] == m.BATCHES[-3]


@pytest.mark.parametrize("field", ["hits", "misses", "evictions", "failures", "resident"])
def test_replay_without_correct_cache_history_rejected(field):
    stats = dict(
        captures=6,
        misses=6,
        hits=2,
        evictions=2,
        failures=0,
        resident=4,
        peak_resident=4,
        capacity=4,
    )
    stats[field] += 1
    with pytest.raises(ValueError, match="cache transition"):
        m.validate_cache_step(7, stats)


def test_extra_packet_uses_only_existing_cuda_controls_and_remains_byte_strict():
    actual = {f"field_{i}": np.array([i, -0.0, np.nan], np.float64) for i in range(77)}
    controls = {name: copy.deepcopy(actual) for name in ("CUDA_golden", "Fast301")}
    result = m.compare_all77(actual, controls)
    assert result["passed"] and "CPU4" not in result["comparisons"]
    actual["field_1"][0] = np.nextafter(1.0, 2.0)
    assert not m.compare_all77(actual, controls)["passed"]


def test_pending_adapter_pin_blocks_before_output_or_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "SOURCE_PINS", {"adapter": "PENDING"})
    monkeypatch.setattr(
        m.torch.cuda, "set_per_process_memory_fraction", lambda *args: pytest.fail("GPU accessed")
    )
    output = tmp_path / "absent"
    with pytest.raises(ValueError, match="pins pending"):
        m.run(SimpleNamespace(execute=True, device="cuda", output=output))
    assert not output.exists()


def test_small_qualified_summary_reads_only_named_member(tmp_path):
    path = tmp_path / "summary.json"
    path.write_bytes(m.proof.json_bytes(dict(passed=True, completed_pairs=301)))
    seal = {
        "outputs": {
            "summary.json": dict(
                path=str(path), size_bytes=path.stat().st_size, sha256=m.proof.sha(path)
            ),
            "unrelated": {},
        }
    }
    seen = []

    class Evidence:
        def add(self, p, sha):
            assert m.proof.sha(p) == sha
            seen.append(p)

    assert m.completed_summary(tmp_path, seal, Evidence())["completed_pairs"] == 301
    assert seen == [path]
    seal["outputs"]["summary.json"]["size_bytes"] += 1
    with pytest.raises(ValueError, match="path/size"):
        m.completed_summary(tmp_path, seal, Evidence())

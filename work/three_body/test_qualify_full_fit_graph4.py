"""Only new saved-control and fail-before-execution boundaries; no saved arrays."""

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
import qualify_full_fit_graph4 as m


def reference():
    return {f"field_{i}": np.array([i, -0.0, np.nan], np.float64) for i in range(77)}


def test_cuda_control_rejects_one_ulp_even_when_frozen_cpu_tolerance_passes():
    actual = reference()
    controls = {k: copy.deepcopy(actual) for k in ("CUDA_golden", "Fast301", "CPU4")}
    assert m.compare_all77(actual, controls)["passed"]
    actual["field_1"][0] = np.nextafter(1.0, 2.0)
    result = m.compare_all77(actual, controls)
    assert result["passed"] is False
    assert result["comparisons"]["CPU4"]["field_1"]["maximum_absolute_difference"] < 1e-10


def test_control_preserves_nanbits_signedzero_and_discrete_record_fields():
    actual = reference()
    actual["field_2"] = np.array([2, 1, 0], np.int64)
    controls = {k: copy.deepcopy(actual) for k in ("CUDA_golden", "Fast301", "CPU4")}
    actual["field_3"][1] = 0.0
    assert not m.compare_all77(actual, controls)["passed"]
    actual["field_2"][0] = 1
    with pytest.raises((ValueError, AssertionError), match="decision mismatch"):
        m.compare_all77(actual, controls)


def test_exact_scope_and_pending_pin_gate_precedes_mkdir_or_cuda(tmp_path, monkeypatch):
    assert m.FRAMES == (66524, 66628, 66703, 66907)
    monkeypatch.setattr(m, "SOURCE_PINS", {"pending": None})
    monkeypatch.setattr(
        m.torch.cuda, "set_per_process_memory_fraction", lambda *a: pytest.fail("GPU")
    )
    destination = tmp_path / "no_output"
    with pytest.raises(ValueError, match="pins pending"):
        m.run(SimpleNamespace(execute=True, device="cuda", output=destination))
    assert not destination.exists()


def test_named_member_requires_path_size_and_byte_sha_without_opening_other_payload(tmp_path):
    path = tmp_path / "four.npz"
    path.write_bytes(b"synthetic-four")
    seen = []

    class Evidence:
        def add(self, value, expected):
            assert m.proof.sha(value) == expected
            seen.append(Path(value))

    row = dict(path=str(path), size=path.stat().st_size, sha256=m.proof.sha(path))
    seal = {"outputs": {"four.npz": row, "unconsumed": {"path": "do-not-read"}}}
    assert m.consumed(tmp_path, seal, "four.npz", Evidence()) == path
    assert seen == [path]
    row["size"] += 1
    with pytest.raises(ValueError, match="size changed"):
        m.consumed(tmp_path, seal, "four.npz", Evidence())


def test_header_rejects_failed_seal_without_verifying_ancillary_outputs(tmp_path):
    path = tmp_path / "seal.json"
    path.write_bytes(m.proof.json_bytes(dict(stage="fixture", status="failed", finished_at="now")))

    class Evidence:
        def add(self, value, expected):
            assert m.proof.sha(value) == expected

    with pytest.raises(ValueError, match="successful"):
        m.sealed_header(tmp_path, path.name, m.proof.sha(path), "fixture", Evidence())

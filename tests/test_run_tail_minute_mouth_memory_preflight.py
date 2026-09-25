"""Finite new MP input edge: sealed raw export, metadata-only, no DP score."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work/three_body"))
import run_tail_minute_mouth_memory_preflight as mod  # noqa: E402


def test_actual_small_JSON_support_export_binding_no_array_read(monkeypatch):
    original = Path.read_bytes

    def read(path):
        assert path.suffix in (".py", ".json"), "payload forbidden in metadata probe"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("NPZ read prohibited"))
    monkeypatch.setattr(
        mod.counter, "reachable_counts", lambda *a, **k: pytest.fail("counter prohibited")
    )
    evidence = mod.previous.Evidence()
    _, plan, _ = mod.load_inputs(evidence)
    assert plan["node_indices"] == [1, 0]
    assert plan["prior_identity_sha256"] == mod.PRIOR_SHA
    assert set(plan["raw_candidate_references"]) == {"original", "tail"}
    assert all(Path(p).suffix != ".npz" for p in evidence.inputs)


@pytest.mark.parametrize("fault", ["changed_binding", "source", "modified_raw"])
def test_reject_wrong_prior_summary_or_input_source(monkeypatch, fault):
    read = mod.previous.named_read
    member = mod.previous.input_member

    def named(seal, root, name, evidence):
        raw = read(seal, root, name, evidence)
        if root == mod.PRIOR and name in ("plan.json", "summary.json"):
            value = json.loads(raw)
            if name == "plan.json" and fault == "changed_binding":
                value["binding"]["dp_config"]["weight"] = 0.21
            if name == "summary.json" and fault == "modified_raw":
                value["raw_candidate21_modified"] = True
            raw = json.dumps(value).encode()
        return raw

    def input_check(seal, path, digest):
        if (
            fault == "source"
            and Path(path).name == "run_tail_minute_identity.py"
            and seal.get("stage") == mod.previous.STAGES["solve"]
        ):
            seal = copy.deepcopy(seal)
            seal["inputs"][str(Path(path).resolve())] = "0" * 64
        return member(seal, path, digest)

    monkeypatch.setattr(mod.previous, "named_read", named)
    monkeypatch.setattr(mod.previous, "input_member", input_check)
    with pytest.raises(ValueError):
        mod.load_inputs(mod.previous.Evidence())


def test_explicit_node_axis_order_and_initial_missing_condition():
    observed = np.zeros((2, 3, 3), bool)
    observed[0] = True
    observed[1, :, 1] = [1, 0, 1]
    observed[1, :, 0] = [0, 1, 1]
    support = observed[:, :, mod.NODE_INDICES]
    assert support.shape == (2, 3, 2)
    np.testing.assert_array_equal(support[:, :, 0], observed[:, :, 1])
    np.testing.assert_array_equal(support[:, :, 1], observed[:, :, 0])
    support[0, 0, 1] = False
    with pytest.raises(ValueError, match="initial"):
        mod.counter.reachable_counts(np.arange(2, dtype=np.int64), support)

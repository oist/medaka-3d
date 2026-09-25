"""Small refusal tests; no long arrays, raw decoding, GPU or actual registration."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import register_long_next_phase as module


class Evidence:
    def __init__(self):
        self.paths = {}

    def add(self, path, expected=None):
        value = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if expected is not None and value != expected:
            raise ValueError("SHA differs")
        self.paths[str(Path(path))] = value
        return value


def test_source_scope_excludes_vendor_environment_but_keeps_local_helpers():
    for relative in ("labs/20260909_three_body/continuous_tracker.py", "src/medaka3d/calib.py",
                     "work/three_body/audit_long_multichunk_minute.py"):
        assert module.local_source(module.ROOT / relative)
    for relative in (".venv/Lib/site-packages/numpy/__init__.py", ".venv/Lib/json/__init__.py",
                     "work/three_body/plan.json", "tests/conftest.py"):
        assert not module.local_source(module.ROOT / relative)


@pytest.mark.parametrize("status,guard", [("failed", True), ("running", True), ("ok", False)])
def test_incomplete_or_unguarded_parent_rejected(tmp_path, status, guard):
    (tmp_path / "seal.json").write_text(json.dumps(dict(
        stage="three_body_long_continuation", status=status,
        params=dict(all_input_source_snapshot_output_sha_unchanged=guard))))
    with pytest.raises((ValueError, AssertionError)):
        module.read_sealed(tmp_path, "seal.json", "three_body_long_continuation", Evidence())


@pytest.fixture
def audit_case(tmp_path, monkeypatch):
    parent, audit = tmp_path / "parent", tmp_path / "audit"
    parent.mkdir()
    audit.mkdir()
    monkeypatch.setattr(module, "PARENT", parent)
    evidence = Evidence()
    candidate, parent_seal = parent / "candidate.npz", parent / "continuation.provenance.json"
    candidate.write_bytes(b"not deserialized")
    parent_seal.write_text("{}")
    digest = evidence.add(candidate)
    result = dict(schema="long_multichunk_minute_independent_audit_v1", status="ok",
                  run=str(parent),
                  frame_range=[49800, 59760], native_frames=9960, logical_initialization_count=1,
                  checkpoint_restore_count=2, external_pose_reset_count=0,
                  all_39_checkpoint_arrays_verified=True, full_sequence_complete=False,
                  independent_ID_truth=False, candidate_sha256=digest,
                  history=dict(active_beam_count=6, all_parent_recurrences_exact=True))
    summary = audit / "summary.json"
    summary.write_text(json.dumps(result))
    source = Path(module.minute.__file__).resolve()
    record = dict(outputs={"summary.json": dict(path=str(summary), sha256=evidence.add(summary))},
                  inputs={str(p): dict(path=str(p), sha256=evidence.add(p))
                          for p in (candidate, parent_seal, source)})
    parent_record = dict(outputs={"candidate": dict(path=str(candidate), sha256=digest)})
    return audit, record, parent_record, evidence


def test_matching_successful_audit_bindings_pass_without_candidate_deserialization(audit_case):
    audit, record, parent, evidence = audit_case
    assert module.verify_audit(audit, record, parent, evidence)["native_frames"] == 9960


@pytest.mark.parametrize("change", ["candidate", "missing_parent", "source"])
def test_foreign_or_missing_audit_evidence_rejected(audit_case, change):
    audit, record, parent, evidence = audit_case
    record, parent = copy.deepcopy(record), copy.deepcopy(parent)
    if change == "candidate":
        parent["outputs"]["candidate"]["sha256"] = "0" * 64
    elif change == "missing_parent":
        record["inputs"].pop(parent["outputs"]["candidate"]["path"])
    else:
        record["inputs"][str(Path(module.minute.__file__).resolve())]["sha256"] = "0" * 64
    with pytest.raises((ValueError, AssertionError)):
        module.verify_audit(audit, record, parent, evidence)

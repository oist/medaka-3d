"""Boundary checks for one new gap arm; no arrays, model, fit or CUDA runs."""

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
import run_mask_transport_gap200 as m  # noqa: E402


def fixture():
    seal = {"inputs": {str(m.PREVIOUS / "tracking.provenance.json"): m.PREVIOUS_SHA}}
    summary = dict(
        status="ok",
        native_interval=[m.FIRST, m.STOP],
        arms=list(m.previous.ARMS),
        strict_candidate_support_rechecks=600,
        unused_pixel_draws=153600,
        independent_ID_truth=False,
        trajectory_or_review_modified=False,
    )
    return seal, summary


def test_matching_previous_evaluation_without_mutation():
    seal, summary = fixture()
    before = copy.deepcopy((seal, summary))
    m.validate_previous_evaluation(seal, summary)
    assert (seal, summary) == before


@pytest.mark.parametrize(
    "fault", ["tracking", "status", "native", "arms", "support", "pixels", "truth", "modified"]
)
def test_rejects_different_or_incomplete_evidence(fault):
    seal, summary = fixture()
    if fault == "tracking":
        seal["inputs"][str(m.PREVIOUS / "tracking.provenance.json")] = "f" * 64
    else:
        key, value = {
            "status": ("status", "failed"),
            "native": ("native_interval", [m.FIRST + 1, m.STOP]),
            "arms": ("arms", ["original", "ct_first"]),
            "support": ("strict_candidate_support_rechecks", 599),
            "pixels": ("unused_pixel_draws", 153599),
            "truth": ("independent_ID_truth", True),
            "modified": ("trajectory_or_review_modified", True),
        }[fault]
        summary[key] = value
    with pytest.raises(ValueError):
        m.validate_previous_evaluation(seal, summary)


def test_actual_small_json_source_edges_before_any_payload():
    path = m.EVALUATION / "evaluation.provenance.json"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == m.EVALUATION_SHA
    seal = json.loads(raw)
    summary_raw = (m.EVALUATION / "summary.json").read_bytes()
    row = seal["outputs"]["summary.json"]
    assert len(summary_raw) == row["size_bytes"]
    assert hashlib.sha256(summary_raw).hexdigest() == row["sha256"]
    m.validate_previous_evaluation(seal, json.loads(summary_raw))
    previous_raw = (m.PREVIOUS / "tracking.provenance.json").read_bytes()
    assert hashlib.sha256(previous_raw).hexdigest() == m.PREVIOUS_SHA
    assert json.loads(previous_raw)["status"] == "ok"


def test_execution_requires_reviewed_sources(monkeypatch):
    args = SimpleNamespace(execute=True, device="cuda", samples_seal_sha256=m.previous.SAMPLES_SHA)
    monkeypatch.setattr(m, "EXECUTION_READY", False)
    with pytest.raises(ValueError, match="draft"):
        m.validate_execution(args)
    monkeypatch.setattr(m, "EXECUTION_READY", True)
    monkeypatch.setattr(m, "SOURCE_PINS", {"fake": "f" * 64})
    m.validate_execution(args)
    args.samples_seal_sha256 = "0" * 64
    with pytest.raises(ValueError, match="samples"):
        m.validate_execution(args)

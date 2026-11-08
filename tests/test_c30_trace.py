"""C30 instrumentation must preserve decisions and measure the actual clock span."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    monkeypatch.syspath_prepend(str(root / "analyses/20261031_p11_identity_evaluation"))
    return import_module("c30_slot_stitch"), import_module("audit_c30_trace")


def test_velocity_audit_matches_uninstrumented_result_and_native_duration(modules):
    c30, audit = modules
    n = 800
    t = np.arange(n, dtype=float)
    xyz = np.empty((n, 3, 3))
    xyz[:, 0] = np.stack([0.8 * (t - 600), np.full(n, 2.5), np.full(n, 50.0)], axis=1)
    xyz[:, 1] = np.stack([0.8 * (t - 600), np.full(n, -2.5), np.full(n, 50.0)], axis=1)
    xyz[:, 2] = [150.0, 150.0, 100.0]
    xyz += np.random.default_rng(5).normal(0, 0.04, xyz.shape)
    xyz[600:] = xyz[600:, [1, 0, 2]]
    times = np.cumsum(np.linspace(0.005, 0.007, n))
    baseline = c30.stitch(xyz, np.arange(n), degraded=np.zeros(n, bool))
    actual = audit.trace_stitch(
        c30, xyz, np.arange(n), np.zeros(n, bool), times, window=8, margin_mm=5.0
    )
    audit.require_identical_replay(
        actual,
        baseline[0],
        {
            "join_list": baseline[1],
            "refusal_list": baseline[2],
        },
    )
    decision = next(r for r in actual[3] if r["after_frame"] == 600)
    assert decision["velocity_adopted"]
    assert decision["velocity_called"]
    assert decision["velocity_history_start_frame"] == 549
    assert decision["velocity_history_end_frame"] == 599
    assert decision["velocity_history_span_seconds"] == pytest.approx(times[599] - times[549])
    assert decision["physical_identity_correctness"] == "not_adjudicated"


def test_audit_rejects_a_changed_saved_permutation(modules):
    _c30, audit = modules
    with pytest.raises(ValueError, match="permutation differs"):
        audit.require_identical_replay(
            (np.array([[0, 1, 2]]), [], [], []),
            np.array([[1, 0, 2]]),
            {"join_list": [], "refusal_list": []},
        )

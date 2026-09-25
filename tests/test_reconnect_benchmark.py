"""Prevent the evaluation wrapper from hiding coordinate changes or review removal."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def benchmark():
    path = (Path(__file__).resolve().parents[1] / "labs/20260908_generative_tracking"
            / "benchmark_reconnect.py")
    spec = importlib.util.spec_from_file_location("reconnect_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example():
    raw = SimpleNamespace(
        frame_idx=np.arange(4), pose_xyz=np.arange(108).reshape(4, 3, 3, 3),
        xyz=np.arange(36).reshape(4, 3, 3).astype(float),
        observed=np.ones((4, 3, 3), bool), node_source=np.arange(108).reshape(4, 3, 3, 3),
        geometry_review=np.zeros((4, 3), bool),
    )
    raw.geometry_review[2, 1] = True
    order = np.tile([0, 2, 1], (4, 1))
    rows = np.arange(4)[:, None]
    connected = SimpleNamespace(frame_idx=raw.frame_idx.copy(), permutation=order)
    for field in ("pose_xyz", "xyz", "observed", "node_source", "geometry_review"):
        setattr(connected, field, getattr(raw, field)[rows, order].copy())
    return raw, connected


def test_wrapper_allows_only_a_recorded_permutation_of_raw_geometry():
    module = benchmark()
    raw, connected = example()
    module.assert_only_relabelled(raw, connected)
    connected.xyz[0, 0, 0] += 1
    with pytest.raises(AssertionError):
        module.assert_only_relabelled(raw, connected)


def test_wrapper_refuses_clearing_an_inherited_geometry_review():
    module = benchmark()
    raw, connected = example()
    connected.geometry_review[:] = False
    with pytest.raises(ValueError, match="geometry review"):
        module.assert_only_relabelled(raw, connected)


def test_infeasible_model_result_is_saved_without_claiming_a_feasible_path():
    module = benchmark()
    raw, _ = example()
    raw.pose_xyz[:] = raw.pose_xyz[0]
    raw.pose_xyz[1] += np.array([500, 0, 0])
    raw.xyz = raw.pose_xyz[:, :, 1].astype(float)
    raw.review = np.zeros((4, 3), bool)
    raw.identity_review = np.zeros((4, 3), bool)
    raw.appearance = np.zeros((4, 3, 3, 24))
    raw.appearance_quality = np.zeros((4, 3, 3))
    raw.diagnostics = {}
    connected = module.reconnect(raw, raw.pose_xyz[0])
    assert connected.review.all()
    assert np.isinf(connected.diagnostics["global_cost"])
    with pytest.raises(ValueError, match="JSON compliant"):
        json.dumps(connected.diagnostics, allow_nan=False)
    safe = module.json_safe_summary({"candidate": {"diagnostics": connected.diagnostics}})
    restored = json.loads(json.dumps(safe, allow_nan=False))
    assert restored["candidate"]["diagnostics"]["global_cost"] is None
    assert not restored["candidate"]["diagnostics"]["physically_feasible_global_path"]
    assert "candidate.diagnostics.global_cost" in restored["serialization"][
        "nonfinite_values_written_as_null"]
    assert np.isnan(connected.accepted_xyz).all()

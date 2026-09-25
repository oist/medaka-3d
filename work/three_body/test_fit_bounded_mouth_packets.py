"""Fail-closed checks for local fit evidence and CPU/CUDA comparison."""

import copy

import numpy as np
import pytest
from fit_bounded_mouth_packets import bind_input, checks, compare_fit_arrays, member


def comparison_fixture():
    return {
        "pose": np.array([1.0, np.nan, np.inf, -np.inf]),
        "selection": np.array([2, 0, 1], np.int64),
        "observed": np.array([True, False]),
        "kind": np.array(["predict", "bounded_mouth_recovery"]),
    }


def test_comparison_accepts_fixed_tolerance_but_records_actual_difference():
    expected = comparison_fixture()
    actual = copy.deepcopy(expected)
    actual["pose"][0] += 5e-11
    result = compare_fit_arrays(actual, expected)
    assert not result["pose"]["byte_exact"]
    assert 0 < result["pose"]["maximum_absolute_difference"] < 1e-10


@pytest.mark.parametrize("index,value", [(0, 1.0 + 2e-10), (1, 0.0), (2, -np.inf), (3, np.inf)])
def test_comparison_rejects_numerical_or_nonfinite_changes(index, value):
    expected = comparison_fixture()
    actual = copy.deepcopy(expected)
    actual["pose"][index] = value
    with pytest.raises(ValueError):
        compare_fit_arrays(actual, expected)


@pytest.mark.parametrize("key", ["selection", "observed", "kind"])
def test_small_pose_tolerance_never_relaxes_discrete_decisions(key):
    expected = comparison_fixture()
    actual = copy.deepcopy(expected)
    actual[key] = actual[key][::-1].copy()
    with pytest.raises(ValueError, match="decision mismatch"):
        compare_fit_arrays(actual, expected)


def test_comparison_rejects_dtype_change_and_missing_field():
    expected = comparison_fixture()
    actual = copy.deepcopy(expected)
    actual["selection"] = actual["selection"].astype(np.int32)
    with pytest.raises(ValueError, match="schema"):
        compare_fit_arrays(actual, expected)
    actual.pop("selection")
    with pytest.raises(ValueError, match="fields"):
        compare_fit_arrays(actual, expected)


def test_existing_unsealed_member_is_rejected(tmp_path):
    path = tmp_path / "packet.npz"
    path.write_bytes(b"not evidence")
    with pytest.raises(ValueError, match="not a sealed member"):
        member(tmp_path, path.name, {})
    hashes = {str(path.resolve()): checks.sha(path)}
    assert member(tmp_path, path.name, hashes) == path.resolve()
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        member(tmp_path, path.name, hashes)


def test_producer_input_membership_and_conflict_checked(tmp_path):
    path = (tmp_path / "calibration.json").resolve()
    path.write_bytes(b"bound")
    digest = checks.sha(path)
    seal = {"inputs": {str(path): {"sha256": digest}}}
    hashes = {}
    assert bind_input(seal, path, hashes) == path
    assert hashes == {str(path): digest}
    with pytest.raises(ValueError, match="did not bind"):
        bind_input({"inputs": {}}, path, {})
    with pytest.raises(ValueError, match="conflicting"):
        bind_input(seal, path, {str(path): "0" * 64})
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed"):
        bind_input(seal, path, {})

"""Behavioral safeguards for experimental geometry-only reassignment proposals."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs" / "20260916_bidirectional_reconnect"


@pytest.fixture(scope="module")
def reconnect():
    path = LAB / "reconnect.py"
    if not path.exists():
        pytest.skip("bidirectional reconnect experiment has been retired")
    spec = importlib.util.spec_from_file_location("_bidirectional_reconnect", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def straight(n=180):
    initial = np.array([[0.0, 0, 0], [0, 20.0, 0], [0, 0, 30.0]])
    velocity = np.array([[0.1, 0.03, 0], [0.2, -0.02, 0], [0.15, 0.01, 0.02]])
    return initial + np.arange(n)[:, None, None] * velocity


def test_persistent_swap_proposal_preserves_input_and_scores_all_six(reconnect):
    xyz = straight()
    xyz[80:] = xyz[80:, [1, 0, 2]]
    original = xyz.copy()
    result = reconnect.audit_window(xyz, np.arange(len(xyz)), 78, 82)
    assert result["status"] == "geometry_proposal_pending_review"
    assert result["suggested_permutation"] == [1, 0, 2]
    assert result["forward_best"] == result["backward_best"]
    assert result["forward_margin_mm"] >= 5
    assert result["backward_margin_mm"] >= 5
    assert len(result["scores"]) == 6
    assert not result["independent_identity_validated"]
    assert "change_frame_not_inferred" in result["application_scope"]
    np.testing.assert_array_equal(xyz, original)
    json.dumps(result, allow_nan=False)


def test_transient_swap_is_local_and_requires_exact_boundaries(reconnect):
    truth = straight()
    xyz = truth.copy()
    xyz[80:84] = xyz[80:84, [1, 0, 2]]
    result = reconnect.audit_transient(xyz, np.arange(len(xyz)), 80, 83)
    assert result["suggested_permutation"] == [1, 0, 2]
    fixed = reconnect.apply_permutation(xyz, 80, 83, result["suggested_permutation"])
    np.testing.assert_array_equal(fixed, truth)
    assert not np.shares_memory(fixed, xyz)
    exterior = reconnect.audit_window(xyz, np.arange(len(xyz)), 78, 85)
    assert exterior["status"] == "identity_preferred"
    padded = reconnect.audit_transient(xyz, np.arange(len(xyz)), 78, 85)
    assert padded["suggested_permutation"] is None


def test_coordinate_jump_is_not_repaired_by_identity_permutation(reconnect):
    xyz = straight()
    xyz[80, 0] += [0, 50, 50]
    frames = np.arange(len(xyz))
    persistent = reconnect.audit_window(xyz, frames, 78, 82)
    transient = reconnect.audit_transient(xyz, frames, 80, 80)
    assert persistent["status"] == "identity_preferred"
    assert transient["suggested_permutation"] is None


def test_transient_scan_uses_clean_flanks_and_cannot_fix_only_part_of_swap(reconnect):
    xyz = straight()
    xyz[80:83] = xyz[80:83, [1, 0, 2]]
    frames = np.arange(len(xyz))
    exact = reconnect.audit_transient(xyz, frames, 80, 82, reference_window=(78, 84))
    assert exact["suggested_permutation"] == [1, 0, 2]
    partial = reconnect.audit_transient(xyz, frames, 81, 81, reference_window=(78, 84))
    assert partial["forward_best"] == [1, 0, 2]
    assert partial["suggested_permutation"] is None
    assert partial["reason"] == "proposed_interval_leaves_other_excluded_rows_inconsistent"


def test_genuine_sharp_turn_does_not_trigger_id_proposal(reconnect):
    xyz = straight()
    xyz[80:, 0] = xyz[79, 0] + np.arange(1, len(xyz) - 79)[:, None] * [0, 0.35, 0]
    result = reconnect.audit_window(xyz, np.arange(len(xyz)), 78, 82)
    assert result["suggested_permutation"] is None


@pytest.mark.parametrize("failure", ["edge", "missing", "gap", "unsupported", "spike"])
def test_invalid_flanks_abstain(reconnect, failure):
    xyz = straight()
    xyz[80:] = xyz[80:, [1, 0, 2]]
    frames = np.arange(len(xyz))
    support = np.ones(xyz.shape[:2], dtype=bool)
    lo, hi = 78, 82
    if failure == "edge":
        lo = 5
    elif failure == "missing":
        xyz[70, 0] = np.nan
    elif failure == "gap":
        frames[70:] += 1
    elif failure == "unsupported":
        support[70, 1] = False
    elif failure == "spike":
        xyz[70, 0, 0] += 10
    result = reconnect.audit_window(xyz, frames, lo, hi, support)
    assert result["status"] == "insufficient_reference"
    assert result["suggested_permutation"] is None
    json.dumps(result, allow_nan=False)


def test_missing_losing_slot_cannot_create_a_zero_cost_path(reconnect):
    xyz = straight()
    xyz[80:] = xyz[80:, [1, 0, 2]]
    support = np.ones(xyz.shape[:2], dtype=bool)
    support[80, 1] = False
    result = reconnect.audit_window(xyz, np.arange(len(xyz)), 78, 82, support)
    assert result["status"] == "rejected_support"
    assert result["forward_best"] == [1, 0, 2]
    assert result["suggested_permutation"] is None


def test_long_window_and_non_native_interval_refused(reconnect):
    xyz = straight(240)
    result = reconnect.audit_window(xyz, np.arange(len(xyz)), 40, 123)
    assert result["status"] == "rejected_long_window"
    frames = np.arange(len(xyz))
    frames[81:] += 1
    result = reconnect.audit_window(xyz, frames, 78, 82)
    assert result["status"] == "rejected_frame_gap"


def test_time_reversal_inverts_persistent_cycle_mapping(reconnect):
    xyz = straight()
    xyz[80:] = xyz[80:, [1, 2, 0]]
    frames = np.arange(len(xyz))
    result = reconnect.audit_window(xyz, frames, 78, 82)
    reversed_result = reconnect.audit_window(xyz[::-1], frames, len(xyz) - 83, len(xyz) - 79)
    assert result["suggested_permutation"] == [2, 0, 1]
    assert reversed_result["suggested_permutation"] == [1, 2, 0]
    assert result["forward_margin_mm"] == pytest.approx(reversed_result["backward_margin_mm"])
    assert result["backward_margin_mm"] == pytest.approx(reversed_result["forward_margin_mm"])


def test_next_best_margin_instead_of_identity_advantage(reconnect):
    xyz = straight()
    # IDs 1 and 2 are indistinguishable, although moving ID 0 fixes a huge cost.
    xyz[:, 2] = xyz[:, 1]
    xyz[80:] = xyz[80:, [1, 0, 2]]
    result = reconnect.audit_window(xyz, np.arange(len(xyz)), 78, 82)
    assert result["suggested_permutation"] is None
    assert result["status"] == "ambiguous"
    assert result["forward_margin_mm"] == pytest.approx(0)


def test_low_total_cost_cannot_hide_bad_single_fish(reconnect):
    xyz = straight()
    xyz[80:] = xyz[80:, [1, 0, 2]]
    xyz[80:, 2] += [0, 0, 3.5]
    result = reconnect.audit_window(xyz, np.arange(len(xyz)), 78, 82)
    assert result["forward_best"] == [1, 0, 2]
    assert result["status"] == "rejected_motion_mismatch"
    assert result["suggested_permutation"] is None


def test_interval_permutation_preserves_all_nodes_and_source_pairs(reconnect):
    xyz = np.arange(12 * 3 * 4 * 3).reshape(12, 3, 4, 3)
    source = np.arange(12 * 3 * 3).reshape(12, 3, 3)
    permutation = [2, 0, 1]
    for original in (xyz, source):
        result = reconnect.apply_permutation(original, 4, 6, permutation)
        np.testing.assert_array_equal(result[:4], original[:4])
        np.testing.assert_array_equal(result[7:], original[7:])
        for target, raw in enumerate(permutation):
            np.testing.assert_array_equal(result[4:7, target], original[4:7, raw])

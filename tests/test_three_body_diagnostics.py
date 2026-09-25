"""Diagnostic disagreement must never become self-generated identity truth."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).parents[1] / "labs/20260909_three_body/diagnostics.py"
SPEC = importlib.util.spec_from_file_location("_three_body_diagnostics", PATH)
assert SPEC and SPEC.loader
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def _result(count=6):
    initial = np.array(
        [
            [[20, 21, 10], [25, 21, 10], [35, 21, 10]],
            [[65, 51, 15], [70, 51, 15], [80, 51, 15]],
            [[100, 81, 20], [105, 81, 20], [115, 81, 20]],
        ],
        dtype=float,
    )
    return {
        "frame_idx": np.arange(100, 100 + count),
        "pose_xyz": np.repeat(initial[None], count, axis=0),
        "observed": np.ones((count, 3, 3), bool),
        "identity_review": np.zeros((count, 3), bool),
        "sources": np.broadcast_to(np.arange(3)[None, :, None, None], (count, 3, 3, 3)).copy(),
        "body_supported": np.ones((count, 3), bool),
    }


def _baseline(result):
    return {"frame_idx": result["frame_idx"].copy(), "xyz": result["pose_xyz"][:, :, 1].copy()}


def test_returning_disagreement_detected_despite_matching_endpoints():
    result = _result()
    baseline = _baseline(result)
    result["pose_xyz"][2:4] = result["pose_xyz"][2:4, [2, 1, 0]]
    report = diagnostics.diagnose(result, baseline_positions={"frozen_arm": baseline})
    comparison = report["baseline_disagreement"]["frozen_arm"]
    assert comparison["disagreement_frames"] == 2
    assert comparison["returning_disagreement_episodes"] == 1
    event = comparison["episodes"][0]
    assert (event["start"], event["stop"], event["affected_fish"]) == (102, 104, [0, 2])
    assert "NOT identity GT" in comparison["interpretation"]
    assert report["identity_evidence"]["status"] == "independent_human_three_animal_truth_required"
    json.dumps(report, allow_nan=False)


def test_absolute_baseline_slot_permutation_does_not_create_disagreement():
    result = _result()
    baseline = _baseline(result)
    baseline["xyz"] = baseline["xyz"][:, [1, 2, 0]]
    compared = diagnostics.diagnose(result, baseline_positions={"arm": baseline})
    arm = compared["baseline_disagreement"]["arm"]
    assert arm["disagreement_frames"] == 0
    assert arm["initial_candidate_to_baseline_slots"] == [2, 0, 1]
    assert arm["reference_normalized_permutation_by_frame"] == [[0, 1, 2]] * 6


def test_missing_or_tied_comparison_does_not_bridge_a_returning_event():
    result = _result()
    baseline = _baseline(result)
    result["pose_xyz"][2] = result["pose_xyz"][2, [2, 1, 0]]
    baseline["xyz"][3] = np.nan
    baseline["xyz"][4, 0] = baseline["xyz"][4, 2]
    arm = diagnostics.diagnose(result, baseline_positions={"arm": baseline})[
        "baseline_disagreement"
    ]["arm"]
    assert arm["returning_disagreement_episodes"] == 0
    assert arm["uncompared_frames"] == 2
    assert arm["reference_normalized_permutation_by_frame"][3] == [-1, -1, -1]


def test_review_grouping_source_switches_and_exclusivity_are_distinct():
    result = _result()
    result["identity_review"][1:4, 0] = True
    result["identity_review"][2:5, 2] = True
    result["sources"][1:, 1] = 9  # ephemeral detection row changes are allowed
    result["sources"][3, 2, 0, 0] = 0  # actual within-frame/node exclusivity violation
    report = diagnostics.diagnose(result)
    assert report["review_workload"]["review_union_frames"] == 4
    assert report["review_workload"]["review_fish_frames"] == 6
    assert report["review_workload"]["co_reviewed_pairs"]["0-2"][0]["frames"] == 2
    assert report["sources"]["within_frame_row_changes"][1][0][0] == 1
    violations = report["sources"]["exclusive_source_violations"]
    assert len(violations) == 1
    assert violations[0]["fish"] == [0, 2]
    assert violations[0]["start"] == 103
    assert report["baseline_disagreement"] == {}


def test_stationarity_anatomy_and_mask_bridge_remain_diagnostics():
    result = _result(30)
    initial = result["pose_xyz"][0].copy()
    result["observed"][:, 0, 1] = False
    result["sources"][:, 0] = -1
    result["sources"][:, 0, 0, 0] = 0
    result["pose_xyz"][-1, 2, 2, 0] += 20
    report = diagnostics.diagnose(result, initial_pose_xyz=initial)
    assert report["body_bridge_frames_per_fish"] == [30, 0, 0]
    assert report["bridge_with_only_one_mouth_or_pectoral_ray_per_fish"] == [30, 0, 0]
    stopped = next(
        row
        for row in report["stationary_pose_runs"]
        if row["fish"] == 0 and row["node"] == "pectoral"
    )
    assert stopped["frames"] == 30
    assert stopped["observed_fraction"] == 0
    assert report["segment_lengths"][-1]["maximum_initial_ratio"] == 3
    assert report["identity_evidence"]["accepted_pectoral"][0]["frames"] == 0


def test_baseline_misalignment_fails_and_diagnosis_does_not_mutate_input():
    result = _result()
    before = copy.deepcopy(result)
    baseline = _baseline(result)
    baseline["frame_idx"] += 1
    with pytest.raises(ValueError, match="exactly match"):
        diagnostics.diagnose(result, baseline_positions={"arm": baseline})
    for name in result:
        np.testing.assert_array_equal(result[name], before[name])


def test_finite_loss_serialization_and_actual_accepted_mask_discrepancy():
    result = _result()
    result["score_margin"] = np.array([np.inf, -1, 0, 2, np.nan, np.inf])
    result["loss_parts"] = np.zeros((6, 5))
    result["accepted_xyz"] = result["pose_xyz"][:, :, 1].copy()
    result["identity_review"][0, 0] = True
    report = diagnostics.diagnose(result)
    assert report["online_margin_summary"]["positive_infinity_count"] == 2
    assert report["online_margin_summary"]["nonfinite_count"] == 3
    assert report["online_margin_summary"]["negative_count"] == 1
    assert (
        report["identity_evidence"]["accepted_mask_inconsistent_with_observed_and_review_count"]
        == 1
    )
    json.dumps(report, allow_nan=False)


def test_unreviewed_counterpart_is_compared_while_another_fish_is_missing():
    result = _result()
    baseline = _baseline(result)
    result["pose_xyz"][2, 0] = result["pose_xyz"][2, 2]
    result["observed"][2, 2] = False
    result["identity_review"][2:, 2] = True
    arm = diagnostics.diagnose(result, baseline_positions={"arm": baseline})[
        "baseline_disagreement"
    ]["arm"]
    assert arm["disagreement_frames"] == 0
    partial = arm["partial_counterpart_comparison"]
    assert partial["unreviewed_disagreement_frames_per_fish"] == [1, 0, 0]
    assert partial["episodes_per_fish"]["0"][0]["start"] == 102
    assert partial["reference_normalized_nearest_position_by_frame"][2] == [2, 1, -1]

"""Reuse the unchanged numeric tests; validate the actual 21-field metadata envelope."""

import diagnose_instance_mouth_candidates_v2 as implementation
import numpy as np
import pytest
import test_diagnose_instance_mouth_candidates as prior_tests


def test_all_previous_pure_numeric_cases_against_v2(monkeypatch):
    monkeypatch.setattr(prior_tests, "diag", implementation)
    for name in (
        "test_best_mouth_and_whole_scene_support_tradeoff_are_distinct",
        "test_mask_cost_worsening_for_another_fish_is_not_hidden",
        "test_reference_mask_is_not_reassigned_to_make_wrong_fish_look_close",
        "test_nonfinite_candidate_stays_invalid_without_making_view_neutral",
        "test_selected_assignment_tie_is_not_arbitrarily_used_as_reference",
        "test_stable_ranks_and_full_node_dedup_match_frozen_selection_rule",
        "test_selected_matching_uses_all_cameras_and_support_and_rejects_duplicates",
    ):
        getattr(prior_tests, name)()
    for fault in ("none", "changed", "missing", "extra", "duplicate"):
        prior_tests.test_companion_descriptors_preserve_nonfinite_bytes_and_reject_tampering(fault)


def test_four_reconstructed_fields_validate_against_larger_signature_envelope():
    rows = [
        dict(
            frame_idx=np.int64(t),
            pose_xyz=np.zeros((3, 3, 3)),
            observed=np.zeros((3, 3), bool),
            sources=np.full((3, 3, 3), -1, np.int32),
        )
        for t in (77709, 77723, 77880)
    ]
    expected = {
        k: implementation.proof.signature(np.asarray([row[k] for row in rows])) for k in rows[0]
    }
    expected.update({f"unused_{k}": {"original_saved_field": True} for k in range(17)})
    assert len(expected) == 21
    assert implementation.verify_selected_rows(rows, expected) == [
        "frame_idx",
        "pose_xyz",
        "observed",
        "sources",
    ]
    rows[0]["pose_xyz"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="pose_xyz"):
        implementation.verify_selected_rows(rows, expected)

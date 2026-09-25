"""Synthetic only: chronology and post-fit source classification, no dataset reads."""

import copy

import diagnose_bounded_tail_first_chunk as d
import numpy as np
import pytest


def history():
    result = {"history_offsets": np.arange(0, 19, 6, dtype=np.int64)}
    floating = {"pose_xyz", "costs", "path_costs", "loss_parts", "score_margin"}
    for key in d.RECORD_FIELDS:
        if key == "proposal_kind":
            value = np.full(18, "predict", dtype="U7")
        elif key == "frame_idx":
            value = np.repeat(np.arange(10, 13), 6)
        else:
            value = np.zeros(18, dtype=np.float64 if key in floating else np.int64)
        result["history_" + key] = value
    return result


def test_chronology_distinguishes_numeric_discrete_and_unicode_storage():
    old = history()
    new = copy.deepcopy(old)
    new["history_proposal_kind"] = new["history_proposal_kind"].astype("U32")
    new["history_costs"][1] = -0.0  # Byte change alone precedes numerical change.
    new["history_pose_xyz"][7] = 0.25
    new["history_proposal_kind"][13] = "bounded_mouth_recovery"
    new["history_hypotheses"][12:] = 2
    result, arrays = d.history_analysis(old, new, start=10, stop=13, first=10)
    assert result["first_any_difference"] == 10
    assert result["first_float_byte_difference"] == 10
    assert result["first_float_numerical_difference"] == 11
    assert result["first_discrete_value_difference"] == 12
    assert result["first_bounded_retained"] == 12
    assert result["fields"]["proposal_kind"]["changed_rows"] == 1
    assert result["first_hypothesis_count_difference"] == 12
    assert arrays["bounded_pose_xyz"][1, 1] == 0.25
    arrays["bounded_pose_xyz"][1, 1] = 99
    assert new["history_pose_xyz"][7] == 0.25


def test_common_parent_and_native_mismatch_rejected():
    old, new = history(), history()
    new["history_costs"][0] = 1
    with pytest.raises(ValueError, match="before common parent"):
        d.history_analysis(old, new, start=11, stop=13, first=10)
    new = history()
    new["history_frame_idx"][7] += 1
    with pytest.raises(ValueError, match="native"):
        d.history_analysis(old, new, start=10, stop=13, first=10)


def test_numeric_dtype_not_silently_coerced():
    with pytest.raises(ValueError, match="dtype"):
        d.row_differences(np.ones(2, np.float64), np.ones(2, np.float32))


def source_fixture():
    old = dict(
        frame_idx=np.arange(5),
        sources=np.full((5, 3, 3, 3), -1, np.int64),
        observed=np.zeros((5, 3, 3), bool),
        identity_review=np.ones((5, 3), bool),
        pose_xyz=np.zeros((5, 3, 3, 3)),
    )
    old["sources"][:, 0, :2, 2] = [4, 7]
    old["observed"][:, 0, 2] = True
    new = copy.deepcopy(old)
    new["observed"][:, 0, 2] = False
    new["sources"][:, 0, :, 2] = -1
    cross = dict(
        valid_source=np.zeros((5, 3, 3, 3), bool),
        ray_distance_mm=np.full((5, 3, 3, 3), np.nan),
        native_pixel_error=np.full((5, 3, 3, 3), np.nan),
    )
    cross["valid_source"][:, 0, :2, 2] = True
    cross["ray_distance_mm"][:, 0, :2, 2] = 1
    cross["native_pixel_error"][:, 0, :2, 2] = 2
    # Frame0: two omitted valid free rows; frame1: collision only;
    # frame2: geometry fails AND collision, each flag must remain visible;
    # frame3: one saved row plus one missing valid row; frame4: two missing valid.
    new["sources"][1:3, 1, 0, 2] = 4
    cross["valid_source"][2, 0, 0, 2] = False
    new["sources"][3, 0, 0, 2] = 4
    return old, new, cross


def test_all_occupancy_geometry_omission_categories_and_nonmutation():
    old, new, cross = source_fixture()
    before = [copy.deepcopy(x) for x in (old, new, cross)]
    summary, cases, rows, flags = d.classify_tail_rows(old, new, cross)
    assert summary["lost_fish_frames"] == 5
    assert summary["lost_case_categories"] == {
        "geometrically_valid_available_rows_omitted_from_saved_sources": 3,
        "support_possible_only_ignoring_other_slot_occupancy": 1,
        "old_and_current_rows_geometrically_insufficient_at_new_pose": 1,
    }
    assert summary["occupied_old_source_rows_in_lost"] == 2
    assert summary["occupied_and_geometrically_valid_old_rows_in_lost"] == 1
    assert summary["lost_with_two_new_unassigned_valid_unoccupied_rows"] == 2
    assert summary["lost_support_recoverable_by_existing_plus_unassigned_old_rows"] == 3
    row = next(r for r in rows if r["frame_idx"] == 2 and r["view"] == d.VIEWS[0])
    assert row["category"] == "geometrically_incompatible"
    assert row["other_slot_occupants"] == [1]
    assert row["geometry_pass"] is False
    assert cases[3]["new_unassigned_valid_unoccupied_views"] == 1
    flags["old_rows"][:] = 0
    for prior, actual in zip(before, (old, new, cross), strict=True):
        for key in prior:
            assert d.checks.equal_bytes(prior[key], actual[key])


def test_other_node_or_same_slot_does_not_create_other_tail_occupancy():
    old, new, cross = source_fixture()
    new["sources"][:, 1, 0, 2] = -1
    new["sources"][:, 1, 0, 0] = 4
    _, _, _, flags = d.classify_tail_rows(old, new, cross)
    assert not flags["other_slot_occupancy"].any()


def test_absent_first_and_nonfinite_json():
    assert d.first_native(np.zeros(3, bool), np.arange(3)) is None
    assert d.finite_json({"a": np.array([np.nan, np.inf, 2.0])}) == {"a": [None, None, 2.0]}

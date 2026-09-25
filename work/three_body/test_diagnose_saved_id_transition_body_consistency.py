"""Small synthetic checks; no saved recording input is read."""

import diagnose_saved_id_transition_body_consistency as diagnostic
import numpy as np


def fixture():
    points = np.array(
        [[[0, 0], [1, 0], [2, 0]], [[0, 20], [1, 20], [2, 20]], [[0, 40], [1, 40], [2, 40]]],
        dtype=float,
    )
    return points.copy(), points, np.tile(np.arange(3)[:, None], (1, 3)), np.ones((3, 3), bool)


def test_same_row_complete_body_and_distinct_chests_zero():
    projected, points, source, observed = fixture()
    result = diagnostic.view_diagnostics(projected, points, source, observed)
    assert result["chest_one_to_one"]["best_distinct_SL_rows"] == [0, 1, 2]
    assert result["chest_one_to_one"]["best_sum_squared_px"] == 0
    assert result["globally_observed_unassigned_view_over_12px"] == []
    for slot, row in enumerate(result["slots"]):
        assert row["best_complete_body_SL_rows"] == [slot]
        assert row["same_SL_row_bodies"][slot]["all_three_nodes_within_12px"] is True


def test_duplicate_nearest_does_not_duplicate_one_to_one_assignment():
    projected, points, source, observed = fixture()
    projected[2] = points[0]
    result = diagnostic.view_diagnostics(projected, points, source, observed)
    chest = result["chest_one_to_one"]
    assert chest["duplicated_unique_nearest_slot_pairs"] == [[0, 2]]
    assert chest["independent_nearest_sum_squared_px"] == 0
    assert len(set(chest["best_distinct_SL_rows"])) == 3
    assert chest["best_sum_squared_px"] > 0


def test_missing_SL_tail_is_not_zero_error_complete_body():
    projected, points, source, observed = fixture()
    points[0, 2] = np.nan
    result = diagnostic.view_diagnostics(projected, points, source, observed)
    group = result["slots"][0]["same_SL_row_bodies"][0]
    assert group["comparable_nodes"] == 2 and group["RMS_of_available_nodes_px"] == 0
    assert group["complete_MCT_RMS_px"] is None and group["all_three_nodes_within_12px"] is False
    assert group["node_distances_px"][2] is None


def test_global_direct_unassigned_third_view_outside_scale_distinguished():
    projected, points, source, observed = fixture()
    projected[0, 1, 0] += 12
    source[0, 1] = -1
    result = diagnostic.view_diagnostics(projected, points, source, observed)
    assert result["globally_observed_unassigned_view_over_12px"] == []
    projected[0, 1, 0] = np.nextafter(projected[0, 1, 0], np.inf)
    result = diagnostic.view_diagnostics(projected, points, source, observed)
    assert [
        (r["slot"], r["node"]) for r in result["globally_observed_unassigned_view_over_12px"]
    ] == [(0, "chest")]
    observed[0, 1] = False
    assert (
        diagnostic.view_diagnostics(projected, points, source, observed)[
            "globally_observed_unassigned_view_over_12px"
        ]
        == []
    )


def test_incomplete_view_cannot_supply_three_distinct_chests():
    projected, points, _, observed = fixture()
    source = np.full((3, 3), -1, dtype=np.int64)
    result = diagnostic.view_diagnostics(projected, points[:2], source, observed)
    assert result["chest_one_to_one"]["feasible"] is False
    assert result["chest_one_to_one"]["best_distinct_SL_rows"] is None
    result = diagnostic.view_diagnostics(projected, [], source, observed)
    assert result["chest_one_to_one"]["independent_nearest_sum_squared_px"] is None
    assert result["slots"][0]["best_complete_body_SL_rows"] == []

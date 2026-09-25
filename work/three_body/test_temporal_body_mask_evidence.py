"""Synthetic body/mask evidence only; no reference arrays, geometry runs or models."""

import copy

import numpy as np
import pytest
import temporal_body_mask_evidence as m


def reference(assignments=(4, 1, -1), eligible=(True, False, False)):
    return dict(
        slots=[
            dict(
                slot=body,
                assigned_mask_row=row,
                geometry_eligible=allowed,
                reasons=[] if allowed else ["ambiguous_or_missing"],
            )
            for body, (row, allowed) in enumerate(zip(assignments, eligible, strict=True))
        ]
    )


def byte_snapshot(arrays):
    return [(x.dtype.str, x.shape, x.tobytes()) for x in arrays]


def vote_inputs():
    return dict(
        coverage=np.array([[1.0, 0.1, 0.2], [0.5, 0.5, 0.1]]),
        complete=np.ones(3, bool),
        membership=np.array([[1, 0], [1, 0], [0, 1], [1, 1], [0, 0]], bool),
        selected=np.ones(5, bool),
        linked=np.array([True, False, False]),
        collision=np.zeros((2, 3), bool),
        available=True,
    )


def test_initial_link_is_only_frozen_eligible_assignment_with_all_mask_rows_retained():
    data = reference()
    before = copy.deepcopy(data)
    actual = m.initial_link_matrix(data, 5)
    expected = np.zeros((5, 3), bool)
    expected[4, 0] = True
    np.testing.assert_array_equal(actual, expected)
    assert data == before
    actual[:] = True
    np.testing.assert_array_equal(m.initial_link_matrix(data, 5), expected)


def test_initial_missing_masks_or_ambiguous_assignments_do_not_force_a_body():
    missing = reference((-1, -1, -1), (False, False, False))
    assert m.initial_link_matrix(missing, 0).shape == (0, 3)
    ambiguous = reference((1, 1, 1), (False, False, False))
    assert not m.initial_link_matrix(ambiguous, 3).any()


@pytest.mark.parametrize("fault", ["collision", "bad_row", "no_row", "reason", "order", "bool"])
def test_inconsistent_initial_geometry_rejected(fault):
    data = reference((0, 1, 2), (True, True, True))
    if fault == "collision":
        data["slots"][1]["assigned_mask_row"] = 0
    elif fault == "bad_row":
        data["slots"][0]["assigned_mask_row"] = 3
    elif fault == "no_row":
        data["slots"][0]["assigned_mask_row"] = -1
    elif fault == "reason":
        data["slots"][0]["reasons"] = ["ambiguous"]
    elif fault == "order":
        data["slots"].reverse()
    else:
        data["slots"][0]["geometry_eligible"] = 1
    with pytest.raises(ValueError):
        m.initial_link_matrix(data, 3)


def test_single_votes_use_all_reported_denominator_without_multi_or_unassigned_renormalizing():
    args = vote_inputs()
    arrays = [value for value in args.values() if isinstance(value, np.ndarray)]
    before = byte_snapshot(arrays)
    result = m.score_votes(**args)
    assert result["selected_count"] == 5 and result["single_count"] == 3
    assert result["multi_count"] == 1 and result["unassigned_count"] == 1
    np.testing.assert_array_equal(result["votes"], [3, 2])
    np.testing.assert_array_equal(result["single_votes"], [2, 1])
    np.testing.assert_allclose(result["weights"], [0.4, 0.2], rtol=0, atol=1e-15)
    np.testing.assert_allclose(result["scores"], [0.5, 0.14, 0.1], rtol=0, atol=1e-15)
    assert result["single_vote_mass"] == 0.6
    np.testing.assert_array_equal(result["available_body"], [True, False, False])
    assert result["reasons"][0] == []
    assert byte_snapshot(arrays) == before


def test_unselected_votes_are_excluded_but_selected_zero_members_remain_in_denominator():
    args = vote_inputs()
    args["selected"] = np.array([True, False, False, False, True])
    result = m.score_votes(**args)
    assert result["selected_count"] == 2 and result["single_count"] == 1
    np.testing.assert_array_equal(result["votes"], [1, 0])
    np.testing.assert_array_equal(result["weights"], [0.5, 0.0])
    np.testing.assert_allclose(result["scores"], [0.5, 0.05, 0.1])
    assert not result["available_body"].any()
    assert "fewer_than_two_single_members" in result["reasons"][0]


def test_one_point_keeps_score_without_becoming_an_available_observation():
    args = vote_inputs()
    args["selected"][:] = False
    args["selected"][0] = True
    result = m.score_votes(**args)
    np.testing.assert_array_equal(result["scores"], [1.0, 0.1, 0.2])
    assert result["selected_count"] == result["single_count"] == 1
    assert result["single_vote_mass"] == 1.0
    assert not result["available_body"].any()
    assert "fewer_than_two_selected_points" in result["reasons"][0]
    assert "fewer_than_two_single_members" in result["reasons"][0]


def test_zero_selected_retains_all_nan_primary_scores_and_all_mask_rows():
    args = vote_inputs()
    args["selected"][:] = False
    result = m.score_votes(**args)
    assert result["selected_count"] == result["single_count"] == 0
    assert np.isnan(result["weights"]).all() and result["weights"].shape == (2,)
    assert np.isnan(result["scores"]).all()
    assert not result["maximum_ties"].any() and not result["available_body"].any()
    assert result["single_vote_mass"] is None


def test_nonempty_mask_set_with_zero_queries_keeps_scheduled_body_reasons():
    args = vote_inputs()
    args["membership"] = np.empty((0, 2), bool)
    args["selected"] = np.empty(0, bool)
    result = m.score_votes(**args)
    assert result["votes"].tolist() == [0, 0]
    assert len(result["reasons"]) == 3 and np.isnan(result["scores"]).all()


@pytest.mark.parametrize("available", [False, True])
def test_missing_observation_and_available_zero_instances_remain_distinct(available):
    args = vote_inputs()
    args.update(
        coverage=np.empty((0, 3)),
        membership=np.empty((5, 0), bool),
        collision=np.empty((0, 3), bool),
        available=available,
    )
    result = m.score_votes(**args)
    assert result["selected_count"] == 5 and result["single_count"] == 0
    assert result["votes"].shape == result["weights"].shape == (0,)
    assert not result["available_body"].any()
    if available:
        np.testing.assert_array_equal(result["scores"], np.zeros(3))
        assert result["unassigned_count"] == 5 and result["multi_count"] == 0
        assert result["single_vote_mass"] == 0.0
        assert "no_current_masks" in result["reasons"][0]
    else:
        assert np.isnan(result["scores"]).all()
        assert result["unassigned_count"] is None and result["multi_count"] is None
        assert result["single_vote_mass"] is None
        assert "masks_unavailable" in result["reasons"][0]


@pytest.mark.parametrize("zero", [False, True])
def test_all_exact_maximum_ties_and_zero_score_ambiguity_are_preserved(zero):
    args = vote_inputs()
    args["coverage"][:] = [0, 0, 0] if zero else [0.7, 0.7, 0.2]
    result = m.score_votes(**args)
    np.testing.assert_array_equal(result["maximum_ties"], [True, True, zero])
    np.testing.assert_array_equal(result["positive_maximum_ties"], [not zero, not zero, False])
    assert not result["available_body"].any()
    assert "not_positive_unique_body_maximum" in result["reasons"][0]


def test_incomplete_competitor_keeps_nan_instead_of_ranking_the_remaining_bodies():
    args = vote_inputs()
    args["complete"][2] = False
    args["coverage"][:, 2] = [np.nan, 0.75]
    result = m.score_votes(**args)
    np.testing.assert_allclose(result["scores"][:2], [0.5, 0.14])
    assert np.isnan(result["scores"][2])
    assert not result["maximum_ties"].any() and not result["available_body"].any()
    assert "incomplete_competitor_projection" in result["reasons"][0]


def test_later_unique_score_never_resolves_missing_strict_initial_link():
    args = vote_inputs()
    args["linked"][:] = False
    result = m.score_votes(**args)
    assert result["scores"][0] > result["scores"][1:].max()
    assert not result["available_body"].any()
    assert "no_strict_initial_link" in result["reasons"][0]


def test_collision_of_a_contributing_single_vote_mask_vetoes_unique_body_score():
    args = vote_inputs()
    args["collision"][1, 0] = True
    result = m.score_votes(**args)
    assert result["maximum_ties"].tolist() == [True, False, False]
    assert result["collision"].tolist() == [True, False, False]
    assert not result["available_body"].any()
    assert "contributing_mask_contains_competing_node" in result["reasons"][0]


def test_unvoted_mask_collision_does_not_erase_other_usable_evidence():
    args = vote_inputs()
    args["membership"] = np.array([[1, 0], [1, 0], [1, 1]], bool)
    args["selected"] = np.ones(3, bool)
    args["collision"][1, 0] = True  # Multi-only membership supplies no score weight.
    result = m.score_votes(**args)
    assert result["single_votes"].tolist() == [2, 0]
    assert not result["collision"].any() and result["available_body"][0]


def test_more_than_three_masks_remain_and_a_late_row_can_supply_evidence():
    args = vote_inputs()
    args.update(
        coverage=np.zeros((5, 3)),
        membership=np.zeros((2, 5), bool),
        selected=np.ones(2, bool),
        linked=np.array([False, False, True]),
        collision=np.zeros((5, 3), bool),
    )
    args["coverage"][4] = [0.1, 0.2, 0.9]
    args["membership"][:, 4] = True
    result = m.score_votes(**args)
    assert result["votes"].tolist() == [0, 0, 0, 0, 2]
    np.testing.assert_allclose(result["scores"], [0.1, 0.2, 0.9])
    assert result["available_body"].tolist() == [False, False, True]


@pytest.mark.parametrize("fault", ["missing_rows", "nonfinite", "range", "complete", "selection"])
def test_vote_schema_and_projection_contract_rejected(fault):
    args = vote_inputs()
    if fault == "missing_rows":
        args["available"] = False
    elif fault == "nonfinite":
        args["coverage"][0, 0] = np.nan
    elif fault == "range":
        args["coverage"][0, 0] = 1.01
    elif fault == "complete":
        args["complete"] = args["complete"].astype(np.int64)
    else:
        args["selected"] = args["selected"][:-1]
    with pytest.raises(ValueError):
        m.score_votes(**args)


def flags():
    return np.zeros((3, 3), bool), np.zeros((3, 3), bool), np.ones((3, 3), bool)


@pytest.mark.parametrize(
    ("temporal_views", "direct_views", "case_a", "case_b"),
    [([0, 1], [], True, False), ([0], [2], False, True), ([0], [0], False, False)],
)
def test_A_two_temporal_views_and_B_different_direct_view_have_distinct_requirements(
    temporal_views, direct_views, case_a, case_b
):
    temporal, direct, observed = flags()
    temporal[temporal_views, 1] = True
    direct[direct_views, 1] = True
    observed[1, 2] = False
    before = byte_snapshot((temporal, direct, observed))
    result = m.availability_cases(temporal, direct, observed)
    assert result[1]["case_A"] is case_a and result[1]["case_B"] is case_b
    assert result[1]["case_A_or_B"] is (case_a or case_b)
    assert result[1]["missing_nodes"] == [2]
    assert not result[0]["case_A_or_B"] and not result[2]["case_A_or_B"]
    assert byte_snapshot((temporal, direct, observed)) == before


def test_all_complementary_view_pairs_and_A_B_overlap_are_retained():
    temporal, direct, observed = flags()
    temporal[[0, 1], 2] = True
    direct[[1, 2], 2] = True
    observed[2, [0, 2]] = False
    body = m.availability_cases(temporal, direct, observed)[2]
    assert body["temporal_views"] == [0, 1] and body["current_direct_views"] == [1, 2]
    assert body["complementary_view_pairs"] == [[0, 1], [0, 2], [1, 2]]
    assert body["missing_nodes"] == [0, 2] and body["case_A_and_B"]


@pytest.mark.parametrize("initial", [False, True])
def test_current_node_deficit_and_later_time_are_both_required(initial):
    temporal, direct, observed = flags()
    temporal[:] = direct[:] = True
    if initial:
        observed[:] = False
    result = m.availability_cases(temporal, direct, observed, initial_frame=initial)
    assert all(not body["case_A_or_B"] for body in result)
    assert all(body["missing_nodes"] == ([0, 1, 2] if initial else []) for body in result)


def test_availability_axes_distinguish_camera_body_and_node():
    temporal, direct, observed = flags()
    temporal[1, 2] = True
    direct[0, 2] = True
    observed[2, 1] = False
    result = m.availability_cases(temporal, direct, observed)
    assert [body["case_B"] for body in result] == [False, False, True]
    assert result[2]["complementary_view_pairs"] == [[1, 0]]
    assert result[2]["missing_nodes"] == [1]


def test_availability_requires_boolean_axes_and_explicit_initial_flag():
    temporal, direct, observed = flags()
    with pytest.raises(ValueError):
        m.availability_cases(temporal, direct, observed.astype(np.int64))
    with pytest.raises(ValueError):
        m.availability_cases(temporal, direct, observed, initial_frame=0)

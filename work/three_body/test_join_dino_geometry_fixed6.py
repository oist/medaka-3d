"""Synthetic-only conditional matrices; no real geometry labels or model."""

import copy

import join_dino_geometry_fixed6 as m
import numpy as np
import pytest


@pytest.mark.parametrize(
    "row,same,count,reason,rank,margin,best",
    [
        ([np.nan, np.nan, np.nan], 0, 0, "no_finite_targets", np.nan, np.nan, []),
        ([0.2, np.nan, np.nan], 0, 1, "same_only_no_cross_comparison", 1, np.nan, [0]),
        ([0.2, np.nan, 0.1], 1, 2, "same_missing_cross_only", np.nan, np.nan, [2]),
        ([0.2, 0.2, 0.3], 0, 3, "same_and_cross_available", 1, 0, [0, 1]),
        ([0.2, 0.1, np.nan], 0, 2, "same_and_cross_available", 2, -0.1, [1]),
    ],
)
def test_row_rank_is_only_among_finite_targets(row, same, count, reason, rank, margin, best):
    values = np.asarray(row)
    before = values.tobytes()
    result = m.row_ranking(values, same_slot=same)
    assert result["finite_target_count"] == count and result["reason"] == reason
    assert result["best_targets"] == best and values.tobytes() == before
    np.testing.assert_equal(result["same_slot_rank_min"], rank)
    np.testing.assert_equal(result["same_slot_margin"], margin)
    if not result["same_and_cross_comparable"]:
        assert result["conditional_same_slot_is_minimum"] is None
    assert result["rank_scope"] == "finite_targets_only_not_identity_accuracy"


def test_exact_tie_rank_interval_and_one_ulp_is_not_a_tie():
    tied = m.row_ranking(np.array([0.2, 0.2, 0.2]), same_slot=1)
    assert tied["same_slot_rank_min"] == 1 and tied["same_slot_rank_max"] == 3
    assert tied["best_targets"] == [0, 1, 2]
    untied = m.row_ranking(np.array([0.2, np.nextafter(0.2, 1), 0.5]), same_slot=1)
    assert untied["best_targets"] == [0] and untied["same_slot_rank_min"] == 2
    assert untied["same_slot_margin"] < 0


def test_six_costs_requires_original_all3_even_with_new_complete_matrix():
    matrix = np.arange(9, dtype=np.float64).reshape(3, 3)
    before = matrix.tobytes()
    assert np.isnan(m.six_costs(matrix, False)).all()
    costs = m.six_costs(matrix, True)
    np.testing.assert_array_equal(costs, [4] * 6)
    assert matrix.tobytes() == before
    matrix[0, 2] = np.nan
    assert np.isnan(m.six_costs(matrix, True)).all()


@pytest.fixture
def partial_case():
    left = dict(
        assigned=[2, 0, -1],
        eligible=[True, True, False],
        reasons=[[], [], ["missing"]],
        ready=False,
    )
    right = dict(
        assigned=[1, 3, -1],
        eligible=[True, True, False],
        reasons=[[], [], ["missing"]],
        ready=False,
    )
    references = {("directional", 1, "top_cam"): left, ("directional", 2, "top_cam"): right}
    request = dict(
        method="directional",
        earlier_frame=1,
        later_frame=2,
        view="top_cam",
        distances=[[0.7, 0.8, "NaN"], [0.9, 0.6, "NaN"], ["NaN"] * 3],
        earlier_ready=False,
        later_ready=False,
        scorable=False,
        earlier_slot_reasons=left["reasons"],
        later_slot_reasons=right["reasons"],
        distance_entry_reasons=[
            [
                []
                if i < 2 and j < 2
                else dict(earlier=left["reasons"][i], later=right["reasons"][j])
                for j in range(3)
            ]
            for i in range(3)
        ],
        request_reason="one_or_both_all3_geometry_core_gates_failed",
    )
    lookup = {}
    for condition, matrix in zip(
        m.CONDITIONS, ([[0.25, 0.5], [0.1, 0.2]], [[0.6, 0.2], [0.3, 0.4]]), strict=True
    ):
        for i in range(2):
            for j in range(2):
                lookup[1, 2, "top_cam", left["assigned"][i], right["assigned"][j], condition] = (
                    dict(
                        pair_index=2 * i + j,
                        condition=condition,
                        status="comparable",
                        comparable=True,
                        median_mutual_cosine_distance=matrix[i][j],
                        mutual_edge_count=1,
                        earlier_eligible=7,
                        later_eligible=9,
                        earlier_valid=7,
                        later_valid=9,
                        earlier_coverage=1 / 7,
                        later_coverage=1 / 9,
                        forward_tie_fraction=0,
                        reverse_tie_fraction=0,
                    )
                )
    return request, references, lookup


def test_conditional_matrix_keeps_all_missing_original_region_and_single_edge(partial_case):
    before = copy.deepcopy(partial_case)
    result = m.conditional_matrix(*partial_case)
    assert partial_case == before
    assert result["distance"].shape == (2, 3, 3)
    np.testing.assert_array_equal(result["distance"][0, :2, :2], [[0.25, 0.5], [0.1, 0.2]])
    assert np.isfinite(result["distance"]).sum() == 8 and np.isnan(result["distance"][:, 2]).all()
    assert len(result["entries"]) == 18 and result["mask_pair_index"][2, 2] == -1
    assert not result["original_all3_gate"]
    assert result["entries"][0]["metric"]["mutual_edge_count"] == 1


@pytest.mark.parametrize(
    "fault",
    ["old_available", "original_reason", "original_gate", "missing_pair", "condition_pair_drift"],
)
def test_no_scope_promotion_or_invented_mask_mapping(partial_case, fault):
    request, refs, lookup = partial_case
    if fault == "old_available":
        request["distances"][2][2] = 0.2
    elif fault == "original_reason":
        request["distance_entry_reasons"][2][2] = []
    elif fault == "original_gate":
        request["scorable"] = True
    elif fault == "missing_pair":
        del lookup[next(iter(lookup))]
    else:
        lookup[1, 2, "top_cam", 2, 1, "masked"]["pair_index"] = 99
    with pytest.raises(ValueError):
        m.conditional_matrix(request, refs, lookup)


def test_missing_one_image_condition_keeps_other_values_and_all_rows(partial_case):
    request, refs, lookup = partial_case
    metric = lookup[1, 2, "top_cam", 2, 1, "masked"]
    metric.update(
        median_mutual_cosine_distance=np.nan,
        status="no_eligible_tokens",
        comparable=False,
        mutual_edge_count=0,
        earlier_coverage=np.nan,
        later_coverage=np.nan,
    )
    arrays, rows, entries, paired = m.join_requests([request], refs, lookup)
    assert arrays["distance"][0, 0, 0, 0] == 0.25
    assert np.isnan(arrays["distance"][0, 1, 0, 0])
    assert len(rows) == 6 and len(entries) == 1 and len(paired) == 3
    assert np.isnan(arrays["permutation_cost"]).all()
    assert paired[0]["raw"]["same_and_cross_comparable"]
    assert not paired[0]["masked"]["same_and_cross_comparable"]
    assert paired[0]["best_target_set_equal"] is None
    assert np.isnan(paired[0]["masked_minus_raw_margin"])
    assert np.isnan(paired[0]["target_background_deltas"][0]["masked_minus_raw_earlier_coverage"])


def test_background_pairing_has_full_auxiliary_counts_without_rescoring(partial_case):
    request, refs, lookup = partial_case
    arrays, rows, _, paired = m.join_requests([request], refs, lookup)
    assert arrays["distance"].shape == (1, 2, 3, 3) and arrays["same_slot_rank_min"].shape == (
        1,
        2,
        3,
    )
    assert paired[0]["both_same_and_cross_comparable"]
    assert paired[0]["best_target_set_equal"] is False
    assert paired[0]["masked_minus_raw_margin"] == pytest.approx(-0.65)
    assert rows[0]["targets"][0]["metric"]["earlier_eligible"] == 7
    assert all(np.isnan(x).all() for x in arrays["permutation_cost"])


def complete_endpoint():
    slots = [
        dict(
            slot=i,
            assigned_mask_row=i,
            assignment_reason="usable",
            observed_mouth_chest=[True, True],
            view_sources_mouth_chest=[i, i],
            direct_mouth_chest=True,
            geometry_eligible=True,
            reasons=[],
            body_coherence=dict(
                requested_points=33, valid_points=33, best_mask_rows=[i], C_single=0.0
            ),
        )
        for i in range(3)
    ]
    reference = dict(
        slots=slots,
        assigned_mask_rows=[0, 1, 2],
        all3_distinct=True,
        all3_geometry_eligible=True,
        slot_geometry_eligible=[True] * 3,
    )
    context = dict(
        slot_reasons=[[], [], []],
        all3_distinct=True,
        scorable=True,
        slot_descriptors=np.zeros((3, 6)).tolist(),
    )
    masks = [dict(descriptor_valid=True)] * 3
    return reference, context, masks


def test_source_coherence_core_gate_retained_without_reprojection():
    values = complete_endpoint()
    before = copy.deepcopy(values)
    result = m.endpoint(*values)
    assert result["ready"] and result["eligible"] == [True] * 3
    assert values == before


@pytest.mark.parametrize(
    "fault", ["direct", "coherence_row", "coherence_points", "source", "core", "context", "row"]
)
def test_inconsistent_original_association_state_rejected(fault):
    reference, context, masks = complete_endpoint()
    if fault == "direct":
        reference["slots"][0]["direct_mouth_chest"] = False
    elif fault == "coherence_row":
        reference["slots"][0]["body_coherence"]["best_mask_rows"] = [1]
    elif fault == "coherence_points":
        reference["slots"][0]["body_coherence"]["valid_points"] = 32
    elif fault == "source":
        reference["slots"][0]["view_sources_mouth_chest"][0] = -1
    elif fault == "core":
        masks[0] = dict(descriptor_valid=False)
    elif fault == "context":
        context["scorable"] = False
    else:
        reference["slots"][0]["assigned_mask_row"] = 3
    with pytest.raises(ValueError):
        m.endpoint(reference, context, masks)


def image_fixture():
    plan, rows = dict(pairs=[]), []
    for i in range(700):
        pair = dict(
            pair_index=i,
            earlier_frame=1,
            later_frame=2,
            view="top_cam",
            earlier_mask_row=i,
            later_mask_row=0,
        )
        plan["pairs"].append(pair)
        for condition in m.CONDITIONS:
            rows.append(
                dict(
                    **pair,
                    condition=condition,
                    comparable=True,
                    status="comparable",
                    metadata=dict(
                        mutual_distance_distribution=dict(median=0.25, count=2),
                        left=dict(
                            eligible_count=7,
                            valid_count=7,
                            matched_fraction=2 / 7,
                            tie_fraction=0.0,
                        ),
                        right=dict(
                            eligible_count=9,
                            valid_count=9,
                            matched_fraction=2 / 9,
                            tie_fraction=0.0,
                        ),
                    ),
                )
            )
    distance = np.full((256, 256), np.nan)
    distance[0, 1], distance[1, 0] = 0.1, 0.4
    mutual = np.isfinite(distance)
    return plan, rows, dict(distances=distance, mutual_nn=mutual)


def test_all1400_medians_recomputed_from_all256_arrays_and_duplicates_refused():
    plan, rows, arrays = image_fixture()
    before = {k: v.tobytes() for k, v in arrays.items()}
    lookup, records = m.image_metrics(plan, rows, lambda _: arrays)
    assert len(lookup) == len(records) == 1400
    assert all(
        r["median_mutual_cosine_distance"] == 0.25 and r["mutual_edge_count"] == 2 for r in records
    )
    assert before == {k: v.tobytes() for k, v in arrays.items()}
    rows[1] = rows[0]
    with pytest.raises(ValueError, match="duplicate"):
        m.image_metrics(plan, rows, lambda _: arrays)


def test_saved_median_drift_rejected_before_geometry_can_join():
    plan, rows, arrays = image_fixture()
    rows[0]["metadata"]["mutual_distance_distribution"]["median"] = 0.1
    with pytest.raises(ValueError, match="median"):
        m.image_metrics(plan, rows, lambda _: arrays)


def test_existing_destination_rejected_without_numeric_input_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m.images, "verified", lambda *_: pytest.fail("must not open input"))
    with pytest.raises(ValueError, match="new fixed"):
        m.run(tmp_path)

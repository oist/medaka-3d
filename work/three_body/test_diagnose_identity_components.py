"""Synthetic geometry/partial-data fixtures; no experimental images or poses."""

from copy import deepcopy

import diagnose_identity_components as diag
import numpy as np
import pytest


def fixture():
    masks = np.zeros((4, 80, 80), bool)
    projected = np.zeros((3, 3, 2), float)
    for slot, y in enumerate((10, 30, 50)):
        masks[slot, y - 4 : y + 8, 6:25] = True
        projected[slot] = ((10, y), (17, y), (23, y))
    masks[3, 72, 72] = True  # An unrelated invalid core must not be used as a substitute.
    observed = np.ones((3, 3), bool)
    sources = np.arange(9).reshape(3, 3)
    metrics = [
        dict(requested_points=33, valid_points=33, best_mask_rows=[s], C_single=0.0)
        for s in range(3)
    ]
    features = np.arange(24, dtype=float).reshape(4, 6) / 24
    features[3] = np.nan
    return masks, projected, observed, sources, metrics, features


def reference(values):
    masks, projected, observed, sources, metrics, _ = values
    return diag.geometry_reference(masks, True, projected, observed, sources, metrics)


def test_fixed_denominators_and_no_tail():
    assert diag.FRAMES == (66524, 66600, 66628, 66703, 66800, 66908)
    assert diag.METHODS == ("directional", "bounded")
    assert len(diag.PAIRS) == 15
    assert len(set(diag.PAIRS)) == 15
    assert all(a < b for a, b in diag.PAIRS)
    assert len(diag.PAIRS) * len(diag.METHODS) * len(diag.VIEWS) == 90
    assert diag.PERMUTATIONS.shape == (6, 3)
    assert len({tuple(p) for p in diag.PERMUTATIONS}) == 6


def test_geometry_nonmutation_and_unreferenced_mask_retention():
    values = fixture()
    before = deepcopy(values)
    ref = reference(values)
    descriptors, reasons, ready = diag.slot_descriptors(ref, values[-1])
    assert ref["assigned_mask_rows"] == [0, 1, 2]
    assert ref["all3_distinct"] and ready
    assert reasons == [[], [], []]
    np.testing.assert_array_equal(descriptors, values[-1][:3])
    assert values[0].shape[0] == values[-1].shape[0] == 4
    for actual, expected in zip(values, before, strict=True):
        if isinstance(actual, np.ndarray):
            assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
        else:
            assert actual == expected


@pytest.mark.parametrize(
    "change", ["observation", "source", "coherence_tie", "coherence_row", "partial"]
)
def test_one_unusable_slot_preserves_partial_distances_but_all_costs_nan(change):
    values = fixture()
    if change == "observation":
        values[2][0, 0] = False
    elif change == "source":
        values[3][0, 1] = -1
    elif change == "coherence_tie":
        values[4][0]["best_mask_rows"] = [0, 3]
    elif change == "coherence_row":
        values[4][0]["best_mask_rows"] = [3]
    else:
        values[4][0]["valid_points"] = 32
    left, why, ready = diag.slot_descriptors(reference(values), values[-1])
    assert not ready and why[0] and not why[1] and not why[2]
    answer = diag.pair_costs(left, fixture()[-1][:3], ready, True)
    assert np.isnan(answer["distances"][0]).all()
    assert np.isfinite(answer["distances"][1:]).all()
    assert np.isnan(answer["permutation_costs"]).all()
    assert not answer["scorable"] and answer["best_permutation_indices"] == []
    assert answer["second_minus_first"] is None


def test_unobserved_latent_competitor_still_blocks_shared_mask():
    values = fixture()
    values[1][1, 0] = values[1][0, 0]
    values[2][1, 0] = False
    ref = reference(values)
    assert ref["slots"][0]["assignment_reason"] == "mask_contains_other_projected_node"
    assert not ref["slot_geometry_eligible"][0]


@pytest.mark.parametrize("available", [False, True])
def test_missing_and_available_empty_have_distinct_reasons(available):
    values = fixture()
    ref = diag.geometry_reference(
        values[0][:0], available, values[1], values[2], values[3], values[4]
    )
    expected = "no_masks" if available else "view_unavailable"
    assert all(r["assignment_reason"] == expected for r in ref["slots"])
    result, _, ready = diag.slot_descriptors(ref, values[-1][:0])
    assert not ready and np.isnan(result).all()


def test_invalid_core_keeps_geometric_reference_and_other_slot_features():
    values = fixture()
    ref = reference(values)
    values[-1][1] = np.nan
    result, reasons, ready = diag.slot_descriptors(ref, values[-1])
    assert ref["all3_geometry_eligible"] and not ready
    assert reasons[1] == ["assigned_core_invalid"]
    assert np.isnan(result[1]).all() and np.isfinite(result[[0, 2]]).all()


def test_same_mask_set_must_be_distinct_for_aggregate_even_if_cells_present():
    values = fixture()
    ref = reference(values)
    ref["all3_geometry_eligible"] = False
    descriptors, _, ready = diag.slot_descriptors(ref, values[-1])
    answer = diag.pair_costs(descriptors, descriptors.copy(), ready, True)
    assert np.isfinite(answer["distances"]).all()
    assert np.isnan(answer["permutation_costs"]).all()


def test_euclidean_permutation_direction_and_inputs_unchanged():
    left = np.zeros((3, 6))
    left[:, 0] = (0, 0.4, 1)
    right = left[[2, 0, 1]].copy()
    before = (left.tobytes(), right.tobytes())
    answer = diag.pair_costs(left, right, True, True)
    expected = np.array([[1, 0, 0.4], [0.6, 0.4, 0], [0, 1, 0.6]])
    np.testing.assert_allclose(answer["distances"], expected, rtol=0, atol=1e-15)
    idx = answer["best_permutation_indices"]
    assert len(idx) == 1
    np.testing.assert_array_equal(diag.PERMUTATIONS[idx[0]], [1, 2, 0])
    assert answer["permutation_costs"][idx[0]] == 0
    assert before == (left.tobytes(), right.tobytes())


def test_all_color_ties_retained():
    answer = diag.pair_costs(np.zeros((3, 6)), np.zeros((3, 6)), True, True)
    assert answer["best_permutation_indices"] == list(range(6))
    assert answer["second_minus_first"] == 0


def test_cross_view_keeps_all_top_ties_and_missing_does_not_agree():
    requests = []
    for method in diag.METHODS:
        for early, late in diag.PAIRS:
            for view in diag.VIEWS:
                ready = view != diag.VIEWS[2]
                result = diag.pair_costs(np.zeros((3, 6)), np.zeros((3, 6)), ready, ready)
                requests.append(
                    dict(method=method, earlier_frame=early, later_frame=late, view=view, **result)
                )
    cross = diag.cross_view_ranks(requests)
    assert len(cross) == 30
    for row in cross:
        scored = row["sides"][diag.VIEWS[1]]
        absent = row["sides"][diag.VIEWS[2]]
        assert len(scored["top_minimum_ranks"]) == 6
        assert all(x["competition_rank"] == 1 for x in scored["top_minimum_ranks"])
        assert not absent["comparison_available"] and absent["top_minimum_ranks"] == []


def test_false_scorable_claim_rejected():
    right = np.zeros((3, 6))
    right[1, 0] = np.nan
    with pytest.raises(ValueError, match="lacks a descriptor"):
        diag.pair_costs(np.zeros((3, 6)), right, True, True)


def test_original_core_helper_uses_all_masks_and_tiny_mask_is_missing():
    values = fixture()
    image = np.zeros((80, 80, 3), np.uint8)
    image[..., 2] = 255
    image_before, masks_before = image.tobytes(), values[0].tobytes()
    table = [diag.appearance.describe_core(image, mask) for mask in values[0]]
    assert len(table) == 4
    assert all(d is not None for d, _ in table[:3])
    assert table[3][0] is None
    # The frozen OpenCV HSV path computes saturation in float32.
    np.testing.assert_allclose(table[0][0], [1, 0, 1, 0.5, 1, 1], rtol=0, atol=2e-7)
    assert image.tobytes() == image_before and values[0].tobytes() == masks_before

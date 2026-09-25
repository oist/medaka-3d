"""Independent synthetic behavior checks; no tracking inputs or filesystem fixtures.

Every observation and pose is constructed below. These tests call only the new
CPU observation/scoring API; no tracking runner or saved-data loader is imported.
"""

from dataclasses import replace

import numpy as np
import partial_body_bundle_score as bundle
import pytest


def project(xyz):
    return xyz[..., :2]


def scene():
    points = np.array(
        [[[x, 30.0], [x + 10.0, 30.0], [x + 20.0, 30.0]] for x in (20.0, 100.0, 180.0)]
    )
    masks = np.zeros((3, 70, 240), dtype=bool)
    for index, x in enumerate((20, 100, 180)):
        masks[index, 20:41, x - 5 : x + 26] = True
    poses = np.concatenate([points, np.zeros((3, 3, 1))], axis=-1)[None]
    return masks, points, poses


def score(masks, points, poses, cfg=None, projector=project):
    observation = bundle.prepare_partial_bundle_observation(
        masks, points, bundle.DEFAULT_CONFIG if cfg is None else cfg
    )
    return bundle.score_partial_body_bundles(poses, projector, observation)


def same_main_result(actual, expected):
    for key in ("best_cost", "best_gain", "assignment_by_body", "assignment_margin_by_body"):
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.mark.parametrize("mask_count", [0, 2, 4])
def test_variable_mask_counts_preserve_semantic_assignments(mask_count):
    masks, points, poses = scene()
    if mask_count == 4:
        extra = np.zeros_like(masks[:1])
        extra[:, 55:60, 220:225] = True
        masks = np.concatenate([masks, extra])
    else:
        masks = masks[:mask_count]
    result = score(masks, points, poses)
    np.testing.assert_array_equal(result["assignment_by_body"], [[0, 1, 2]])
    assert np.isfinite(result["best_cost"]).all()
    assert result["best_gain"][0] >= 12.0


@pytest.mark.parametrize("finite_nodes, expected_gain", [([0], 6.0), ([0, 1], 9.0), ([2], 3.0)])
def test_mouth_only_and_incomplete_rows_remain_usable(finite_nodes, expected_gain):
    _, points, poses = scene()
    partial = np.full_like(points, np.nan)
    partial[:, finite_nodes] = points[:, finite_nodes]
    result = score(None, partial, poses)
    np.testing.assert_array_equal(result["assignment_by_body"], [[0, 1, 2]])
    assert result["best_gain"][0] == expected_gain
    assert result["best_cost"][0] == 15.0 - expected_gain


def test_missing_node_does_not_receive_per_row_average_discount():
    _, points, poses = scene()
    full = score(None, points, poses)
    partial = points.copy()
    partial[0, 2] = np.nan
    missing_tail = score(None, partial, poses)
    assert full["best_gain"][0] == 12.0
    assert missing_tail["best_gain"][0] == 11.0
    assert missing_tail["best_cost"][0] == full["best_cost"][0] + 1.0


def test_one_merged_mask_can_support_three_distinct_semantic_rows():
    masks, points, poses = scene()
    merged = np.zeros_like(masks[:1])
    merged[:, 20:41, 15:206] = True
    result = score(merged, points, poses)
    np.testing.assert_array_equal(result["assignment_by_body"], [[0, 1, 2]])
    assert result["best_gain"][0] == 15.0
    assert result["best_cost"][0] == 0.0


def test_single_row_cannot_be_reused_by_three_duplicate_bodies():
    masks, points, poses = scene()
    duplicate_bodies = np.repeat(poses[:, :1], 3, axis=1)
    result = score(masks[:1], points[:1], duplicate_bodies)
    assigned = result["assignment_by_body"][0]
    assert np.count_nonzero(assigned == 0) == 1
    assert np.count_nonzero(assigned == -1) == 2
    assert result["best_gain"][0] == 5.0
    assert result["best_cost"][0] == 10.0
    assert np.isnan(result["assignment_margin_by_body"][0, assigned == -1]).all()


def test_joint_assignment_avoids_greedy_consumption_of_only_good_row():
    _, points, poses = scene()
    points[1] = points[0] + [12.0, 0.0]
    poses[0, 0, :, :2] = points[0] + [5.0, 0.0]
    poses[0, 1, :, :2] = points[0]
    result = score(None, points, poses)
    # Body0 can use row1; body1 has no gain on row1. Giving row0 to body0 is greedy.
    np.testing.assert_array_equal(result["assignment_by_body"], [[1, 0, 2]])
    assert result["best_gain"][0] > 11.0


def test_unused_empty_and_outlier_masks_do_not_force_assignments():
    masks, points, poses = scene()
    original = score(masks, points, poses)
    empty = np.zeros_like(masks[:1])
    outlier = empty.copy()
    outlier[:, 55:60, 220:225] = True
    augmented = score(np.concatenate([empty, masks, outlier]), points, poses)
    same_main_result(augmented, original)
    evidence = augmented["observation_evidence"]
    assert evidence["nonempty_mask_indices"] == [1, 2, 3, 4]
    assert evidence["compatible_mask_indices_by_row"] == [[1], [2], [3]]
    np.testing.assert_array_equal(
        augmented["best_mask_by_body_row"][0, np.arange(3), np.arange(3)], [1, 2, 3]
    )


def test_split_fragments_are_not_silently_unioned_for_row_compatibility():
    _, points, poses = scene()
    # Each node has its own tiny mask; no mask contains all three finite nodes.
    fragments = np.zeros((9, 70, 240), dtype=bool)
    points = points.copy()
    points[:, 1, 0] += 10
    points[:, 2, 0] += 20
    poses = poses.copy()
    poses[0, :, :, :2] = points
    for index, (x, y) in enumerate(points.reshape(-1, 2).astype(int)):
        fragments[index, y, x] = True
    without_mask = score(None, points, poses)
    fragmented = score(fragments, points, poses)
    same_main_result(fragmented, without_mask)
    assert fragmented["best_gain"][0] == 12.0


def test_far_candidate_uses_dummies_instead_of_zero_gain_real_rows():
    masks, points, poses = scene()
    far = poses + np.array([1000.0, 1000.0, 0.0])
    result = score(masks, points, far)
    np.testing.assert_array_equal(result["assignment_by_body"], [[-1, -1, -1]])
    assert result["best_gain"][0] == 0.0
    assert result["best_cost"][0] == 15.0
    assert np.isnan(result["assignment_margin_by_body"]).all()


@pytest.mark.parametrize("bad_coordinate", [np.nan, np.inf, -np.inf])
def test_nonfinite_projection_invalidates_only_affected_candidate(bad_coordinate):
    masks, points, poses = scene()
    batch = np.repeat(poses, 2, axis=0)

    def nonfinite_projector(xyz):
        xy = project(xyz).copy()
        xy[1, 1, 0, 0] = bad_coordinate
        return xy

    result = score(masks, points, batch, projector=nonfinite_projector)
    assert np.isfinite(result["best_cost"][0])
    assert np.isposinf(result["best_cost"][1])


def test_duplicate_rows_have_zero_assignment_margin_not_certified_identity():
    _, points, poses = scene()
    rows = np.stack([points[0], points[0]])
    result = score(None, rows, poses)
    assert result["assignment_by_body"][0, 0] in (0, 1)
    np.testing.assert_array_equal(result["assignment_by_body"][0, 1:], [-1, -1])
    assert result["assignment_margin_by_body"][0, 0] == 0.0
    assert result["best_gain"][0] == 4.0


def test_dropped_nonfinite_rows_preserve_original_row_numbers():
    _, points, poses = scene()
    rows = np.full((5, 3, 2), np.nan)
    rows[[1, 3, 4]] = points
    rows[2, 0, 0] = 5.0  # A single finite coordinate does not make a node observed.
    result = score(None, rows, poses)
    np.testing.assert_array_equal(result["assignment_by_body"], [[1, 3, 4]])


@pytest.mark.parametrize("points", [None, np.empty((0, 3, 2)), np.full((2, 3, 2), np.nan)])
def test_no_finite_semantic_observation_is_neutral(points):
    masks, _, poses = scene()
    result = score(masks, points, poses)
    assert result["best_cost"][0] == 0.0
    np.testing.assert_array_equal(result["assignment_by_body"], [[-1, -1, -1]])


def test_missing_semantic_tail_still_contributes_to_mask_containment():
    masks, points, poses = scene()
    points = points.copy()
    points[0, 2] = np.nan
    batch = np.repeat(poses, 2, axis=0)
    batch[1, 0, 2, 1] += 80.0
    no_mask = score(None, points, batch)
    assert no_mask["best_cost"][0] == no_mask["best_cost"][1]
    with_mask = score(masks, points, batch)
    assert with_mask["best_cost"][1] > with_mask["best_cost"][0]
    np.testing.assert_array_equal(with_mask["assignment_by_body"], [[0, 1, 2], [0, 1, 2]])


def test_large_mouth_error_does_not_buy_tail_repair_in_this_fixture():
    masks, points, poses = scene()
    batch = np.repeat(poses, 2, axis=0)
    batch[0, 0, 2, 1] += 80.0  # Correct mouth/chest, incorrect tail.
    batch[1, 0, 0, 1] += 20.0  # Repair tail by worsening mouth.
    result = score(masks, points, batch)
    assert result["best_cost"][0] < result["best_cost"][1]
    # This one counterexample guard is not a universal mouth-first guarantee.


def test_configured_mouth_weight_and_inputs_remain_unchanged():
    masks, points, poses = scene()
    original = [array.tobytes() for array in (masks, points, poses)]
    cfg = replace(bundle.DEFAULT_CONFIG, node_weights=(3.0, 1.0, 1.0), mask_weight=0.0)
    partial = np.full_like(points, np.nan)
    partial[:, 0] = points[:, 0]
    partial_before = partial.tobytes()
    result = score(masks, partial, poses, cfg)
    assert result["best_gain"][0] == 9.0
    assert cfg.node_weights == (3.0, 1.0, 1.0)
    assert [array.tobytes() for array in (masks, points, poses)] == original
    assert partial.tobytes() == partial_before

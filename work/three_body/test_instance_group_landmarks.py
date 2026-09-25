"""Synthetic semantic failures that silhouette membership alone cannot distinguish."""

from dataclasses import replace

import instance_body_score as mask
import instance_group_landmarks as m
import numpy as np
import pytest


def scene():
    masks = np.zeros((3, 50, 220), dtype=bool)
    points = np.array(
        [[[10 + 70 * k, 25], [25 + 70 * k, 25], [50 + 70 * k, 25]] for k in range(3)], dtype=float
    )
    for k in range(3):
        masks[k, 15:36, 5 + 70 * k : 56 + 70 * k] = True
    pose = np.concatenate([points, np.ones((3, 3, 1))], axis=-1)
    folded = pose.copy()
    folded[2, 0, 0] = folded[2, 2, 0]
    return masks, points, np.stack([pose, folded])


def scored(bank=None):
    masks, points, poses = scene()
    observation = mask.prepare_instance_observation(masks, points)
    result = mask.score_instance_bodies(
        poses if bank is None else bank, lambda x: x[..., :2], observation
    )
    return result, points


def test_folded_mouth_inside_correct_mask_gets_semantic_penalty():
    source, points = scored()
    assert source["eligible"]
    np.testing.assert_array_equal(source["best_cost"], [0, 0])
    before = source["body_instance_costs"].tobytes()
    result = m.add_group_landmark_cost(source, points)
    np.testing.assert_array_equal(result["best_assignment"], [[0, 1, 2], [0, 1, 2]])
    assert result["best_cost"][0] == 0 < result["best_cost"][1]
    assert result["semantic_body_instance_costs"][1, 2, 2] == 3
    assert source["body_instance_costs"].tobytes() == before
    np.testing.assert_array_equal(source["best_cost"], [0, 0])


def test_body_permutation_preserves_single_group_for_all_three_nodes():
    _, _, bank = scene()
    source, points = scored(bank[:, [2, 0, 1]])
    result = m.add_group_landmark_cost(source, points)
    np.testing.assert_array_equal(result["best_assignment"], [[2, 0, 1], [2, 0, 1]])
    np.testing.assert_array_equal(result["semantic_SL_row_by_instance"], [0, 1, 2])
    for i, assignment in enumerate(mask.PERMUTATIONS):
        np.testing.assert_array_equal(
            result["assignment_costs"][:, i],
            result["body_instance_costs"][:, np.arange(3), assignment].sum(axis=1),
        )


def test_zero_weight_keeps_frozen_scores_exact_and_nonfinite_candidates_invalid():
    _, _, bank = scene()
    bank[1, 0, 0] = np.nan
    source, points = scored(bank)
    result = m.add_group_landmark_cost(source, points, replace(m.DEFAULT_CONFIG, weight=0))
    for key in (
        "best_cost",
        "best_assignment",
        "assignment_costs",
        "optimal_assignment_mask",
        "body_instance_costs",
    ):
        assert result[key] is source[key]
    active = m.add_group_landmark_cost(source, points)
    assert np.isinf(active["best_cost"][1])
    np.testing.assert_array_equal(active["best_assignment"][1], [-1, -1, -1])
    assert not active["optimal_assignment_mask"][1].any()


def test_neutral_view_remains_neutral_without_reading_missing_SLEAP():
    _, _, bank = scene()
    source = mask.score_instance_bodies(
        bank,
        lambda _: pytest.fail("neutral projector called"),
        mask.prepare_instance_observation(None, None),
    )
    result = m.add_group_landmark_cost(source, None)
    assert not result["eligible"] and result["neutral"]
    assert result["semantic_body_instance_costs"] is None
    assert result["best_cost"] is source["best_cost"]


@pytest.mark.parametrize("groups", [[[0], [0], [2]], [[0, 1], [1], [2]], [[0], [1], [99]]])
def test_ambiguous_or_stale_group_indices_fail(groups):
    source, points = scored()
    source["observation_evidence"] = dict(source["observation_evidence"], qualifying_SL_rows=groups)
    with pytest.raises(ValueError):
        m.add_group_landmark_cost(source, points)


@pytest.mark.parametrize(
    "setting",
    [dict(pixel_scale_px=0), dict(weight=-1), dict(per_node_cap=np.inf), dict(weight=True)],
)
def test_invalid_configuration_fails(setting):
    with pytest.raises(ValueError):
        m.GroupLandmarkConfig(**setting)

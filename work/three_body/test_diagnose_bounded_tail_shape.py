"""Synthetic geometry only; no actual fish truth or run inference."""

import diagnose_bounded_tail_shape as d
import numpy as np
import pytest


def poses():
    straight = np.array([[-2.0, 0, 0], [0, 0, 0], [3, 0, 0]])
    folded = np.array([[-2.0, 0, 0], [0, 0, 0], [-3, 0, 0]])
    right = np.array([[-2.0, 0, 0], [0, 0, 0], [0, 3, 0]])
    return np.stack([straight, folded, right])


def test_angle_definition_and_length_preserving_fold():
    p = poses()
    s = d.body_shape(p)
    np.testing.assert_array_equal(s["angle_deg"], [180, 0, 90])
    np.testing.assert_array_equal(s["segment_lengths_mm"], [[2, 3]] * 3)
    np.testing.assert_array_equal(s["mouth_tail_distance_mm"][:2], [5, 1])
    np.testing.assert_array_equal(s["polyline_length_mm"], [5, 5, 5])
    lengths = np.array([[2, 3]] * 3)
    assert d.length_loss(s["segment_lengths_mm"], lengths) == 0


def test_rigid_transform_invariance_no_input_mutation():
    p = poses()
    before = p.copy()
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    original = d.body_shape(p)
    transformed = d.body_shape(p @ rotation.T + [91, -20, 8])
    for key in ("angle_deg", "segment_lengths_mm", "mouth_tail_distance_mm", "polyline_length_mm"):
        np.testing.assert_allclose(original[key], transformed[key], atol=1e-12)
    assert np.array_equal(p, before)


def test_degenerate_angle_is_unknown_not_zero_or_straight():
    p = poses()
    p[0, 0] = p[0, 1]
    angles = d.body_shape(p)["angle_deg"]
    assert np.isnan(angles[0])
    assert d.angle_bins(angles).tolist() == [-1, 0, 2]
    assert d.distribution(angles)["undefined_count"] == 1


def test_bins_exact_boundaries_and_reject_outside():
    assert d.angle_bins(np.array([0.0, 44.9, 45, 90, 135, 180, np.nan])).tolist() == [
        0,
        0,
        1,
        2,
        3,
        3,
        -1,
    ]
    with pytest.raises(ValueError):
        d.angle_bins(np.array([181.0]))


def test_any_all_and_direct_retained_first_times_are_distinct():
    angles = np.full((4, 6, 3), 180.0)
    direct = np.zeros_like(angles, dtype=bool)
    angles[1, 3, 0] = 20
    angles[2:, :, 0] = 10
    direct[3, 1, 0] = True
    result = d.threshold_timeline(angles, direct, np.arange(100, 104))
    row = result[0]
    assert row["first_any_rank"] == 101
    assert row["first_every_rank"] == 102
    assert row["first_any_direct_tail_rank"] == 103
    assert row["below_rank_frames"] == 13
    assert result[3]["first_any_rank"] is None


def test_old_new_tail_direction_comparison_keeps_at_chest_origin():
    p = poses()
    original = d.body_shape(p[:1])
    translated = d.body_shape(p[1:2] + np.array([42, 21, -6]))
    np.testing.assert_array_equal(
        d.vector_angle(original["tail_vector"], translated["tail_vector"]), [180]
    )

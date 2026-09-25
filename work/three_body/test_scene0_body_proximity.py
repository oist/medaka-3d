"""Analytic checks for the independent proximity measurement, not new tracking rules."""

import numpy as np
import pytest
from measure_scene0_body_proximity import (
    covariance,
    gaussian_pair_distances,
    polyline_closest,
    segment_closest,
)


@pytest.mark.parametrize(
    ("points", "distance"),
    [
        (([0, 0, 0], [2, 0, 0], [1, -1, 0], [1, 1, 0]), 0),
        (([0, 0, 0], [2, 0, 0], [1, -1, 3], [1, 1, 3]), 3),
        (([0, 0, 0], [2, 0, 0], [3, 2, 0], [5, 2, 0]), np.sqrt(5)),
        (([0, 0, 0], [2, 0, 0], [1, 2, 0], [3, 2, 0]), 2),
        (([0, 0, 0], [0, 0, 0], [1, -1, 0], [1, 1, 0]), 1),
        (([0, 0, 0], [0, 0, 0], [1, 2, 2], [1, 2, 2]), 3),
    ],
)
def test_closed_segments_cover_crossing_skew_parallel_endpoints_and_zero_length(points, distance):
    result = segment_closest(*points)
    assert result["distance_mm"] == pytest.approx(distance, abs=1e-12)
    assert all(0 <= t <= 1 for t in result["fractions"])
    assert np.linalg.norm(np.diff(result["points_xyz"], axis=0)) == pytest.approx(
        distance, abs=1e-12
    )
    swapped = segment_closest(*points[2:], *points[:2])
    assert swapped["distance_mm"] == pytest.approx(distance, abs=1e-12)


def test_polyline_can_touch_while_pectoral_centers_are_distant():
    a = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0]], float)
    b = np.array([[9, -1, 0], [9, 0, 0], [9, 3, 0]], float)
    assert np.linalg.norm(a[1] - b[1]) == 8
    assert polyline_closest(a, b)["distance_mm"] == 0


def test_isotropic_combined_covariance_scale_and_symmetry():
    centers_a, centers_b = np.zeros((1, 3)), np.array([[3, 0, 0]])
    cov_a, cov_b = np.eye(3)[None] * 4, np.eye(3)[None] * 5
    euclidean, standard = gaussian_pair_distances(centers_a, cov_a, centers_b, cov_b)
    np.testing.assert_array_equal(euclidean, [[3]])
    np.testing.assert_allclose(standard, [[1]], rtol=0, atol=1e-14)
    reversed_values = gaussian_pair_distances(centers_b, cov_b, centers_a, cov_a)
    np.testing.assert_array_equal(standard, reversed_values[1].T)


def test_axis_covariance_and_rigid_rotation_preserve_distances():
    tangent = np.array([[1, 0, 0]], float)
    cov = covariance(tangent, np.array([3.0]), np.array([1.0]))
    np.testing.assert_array_equal(cov[0], np.diag([9, 1, 1]))
    origin, points = np.zeros((1, 3)), np.array([[3, 0, 0], [0, 3, 0]], float)
    e, d = gaussian_pair_distances(origin, cov, points, np.broadcast_to(cov, (2, 3, 3)))
    np.testing.assert_allclose(d, [[1 / np.sqrt(2), 3 / np.sqrt(2)]])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    rotated = rotation @ cov @ rotation.T
    er, dr = gaussian_pair_distances(
        origin @ rotation.T + 14,
        rotated,
        points @ rotation.T + 14,
        np.broadcast_to(rotated, (2, 3, 3)),
    )
    np.testing.assert_allclose(e, er)
    np.testing.assert_allclose(d, dr)


def test_nonfinite_and_nonpositive_covariance_fail_explicitly():
    with pytest.raises(ValueError, match="finite"):
        segment_closest([np.nan, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0])
    with pytest.raises(ValueError, match="positive definite"):
        gaussian_pair_distances(
            np.zeros((1, 3)), -np.eye(3)[None], np.ones((1, 3)), np.eye(3)[None]
        )

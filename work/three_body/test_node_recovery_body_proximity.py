"""A fixed surrogate's analytic scale and label-order invariance."""

import numpy as np
import pytest
from measure_node_recovery_body_proximity import proximity_surrogate


def test_coincident_and_separated_gaussian_pairs_have_known_scale():
    d = np.full((3, 8, 8), 12.0)
    d[0, 3, 4] = 0.0
    d[1, 4, 1] = 2.0
    d[2, 5, 7] = 4.0
    pairs, mean = proximity_surrogate(d)
    np.testing.assert_allclose(pairs, [1.0, np.exp(-2), np.exp(-8)])
    assert mean == pytest.approx((1 + np.exp(-2) + np.exp(-8)) / 3)
    permuted_pairs, permuted_mean = proximity_surrogate(d[[2, 0, 1], ::-1, ::-1])
    np.testing.assert_array_equal(permuted_pairs, pairs[[2, 0, 1]])
    assert mean == pytest.approx(permuted_mean)


def test_invalid_schema_and_nonfinite_distances_fail():
    with pytest.raises(ValueError, match="three fish"):
        proximity_surrogate(np.zeros((2, 8, 8)))
    with pytest.raises(ValueError, match="finite"):
        proximity_surrogate(np.full((3, 8, 8), np.nan))

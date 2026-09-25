import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "work/three_body"), str(ROOT / "labs/20260909_three_body")]
from body_point_identity import compare_mapping, identity_costs  # noqa: E402


def fixture():
    rng = np.random.default_rng(82)
    own = rng.random((3, 3, 4, 3))
    q = rng.random(own.shape) * 16
    weights = np.zeros((3, 2, 3, 4, 3))
    for t, a, v, j in np.ndindex(weights.shape[:-1]):
        weights[t, a, v, j, (t + a + v + j) % 3] = 0.3 + a + j
    valid = np.ones((3, 3, 4), bool)
    valid[1, 2, 0] = False
    return own, q, weights, valid


def test_all_identity_raw_pairs_match_independent_scalar_sum():
    own, q, w, valid = fixture()
    before = [a.copy() for a in (own, q, w, valid)]
    result = identity_costs(own, q, w, valid)
    expected = np.zeros((3, 2, 3, 3, 3))
    for t, a, v, identity, raw in np.ndindex(expected.shape):
        for j in range(4):
            if valid[t, v, j]:
                expected[t, a, v, identity, raw] += w[t, a, v, j, identity] * (
                    (1 - own[t, v, j, raw]) ** 2 + (np.sqrt(max(q[t, v, j, raw], 1)) - 1)
                )
    np.testing.assert_allclose(result["combined_cost_by_view"], expected, rtol=1e-14)
    perms = np.array(list(itertools.permutations(range(3))), dtype=np.int64)
    expected_u = np.zeros((3, 2, 6))
    for t, a, p in np.ndindex(expected_u.shape):
        expected_u[t, a, p] = sum(
            expected[t, a, v, i, perms[p, i]] for v in range(3) for i in range(3)
        )
    np.testing.assert_allclose(result["measured_unary"], expected_u, rtol=1e-14)
    np.testing.assert_array_equal(result["unary"][0], 0)
    np.testing.assert_array_equal(result["unary"][1:], result["measured_unary"][1:])
    for current, original in zip((own, q, w, valid), before, strict=True):
        np.testing.assert_array_equal(current, original)


def test_permutation_changes_only_raw_body_axis_not_initial_identity():
    own = np.zeros((2, 3, 3, 3))
    q = np.full_like(own, 9.0)
    w = np.zeros((2, 2, 3, 3, 3))
    expected_mapping = [2, 0, 1]
    for identity, raw in enumerate(expected_mapping):
        own[:, :, identity, raw] = 1
        q[:, :, identity, raw] = 0
        w[:, 0, :, identity, identity] = 2.0 + identity
        # Different initial-identity evidence in the other anchor must survive.
        w[:, 1, :, identity, raw] = 2.0 + identity
    r = identity_costs(own, q, w, np.ones((2, 3, 3), bool))
    np.testing.assert_array_equal(r["permutations"][r["unary"][1, 0].argmin()], expected_mapping)
    np.testing.assert_array_equal(r["permutations"][r["unary"][1, 1].argmin()], [0, 1, 2])
    assert r["unary"][1, 0].min() == 0
    assert r["unary"][1, 0, 0] > 0


def test_invalid_nan_is_neutral_and_does_not_rescale_remaining_mass():
    own, q, w, valid = fixture()
    own[~valid] = np.nan
    q[~valid] = np.inf
    r = identity_costs(own, q, w, valid)
    expected = (w * valid[:, None, :, :, None]).sum(axis=3)
    np.testing.assert_array_equal(r["mass_by_view_identity"], expected)
    assert np.isfinite(r["unary"]).all()
    assert expected[1, :, 2].sum() < w[1, :, 2].sum()


@pytest.mark.parametrize("which,value", [(0, np.nan), (1, np.inf), (1, -1.0)])
def test_bad_valid_raw_cost_rejected_even_with_no_assigned_identity_mass(which, value):
    data = list(fixture())
    data[2][:] = 0
    data[which][1, 0, 0, 2] = value
    with pytest.raises(ValueError, match="all valid raw-body"):
        identity_costs(*data)


@pytest.mark.parametrize("queries", [0, 3])
def test_zero_evidence_leaves_all_six_candidates_equal(queries):
    r = identity_costs(
        np.zeros((2, 3, queries, 3)),
        np.ones((2, 3, queries, 3)),
        np.zeros((2, 2, 3, queries, 3)),
        np.ones((2, 3, queries), bool),
    )
    np.testing.assert_array_equal(r["unary"], 0)
    assert r["unary"].shape == (2, 2, 6)


def test_competing_identity_mass_rejected():
    own, q, w, valid = fixture()
    w[0, 0, 0, 0] = [1.0, 2.0, 0.0]
    with pytest.raises(ValueError, match="exclusive"):
        identity_costs(own, q, w, valid)


@pytest.mark.parametrize("value", [True, -1.0, np.nan, np.inf])
def test_bad_coefficient(value):
    with pytest.raises(ValueError, match="coefficient"):
        identity_costs(*fixture(), coefficient=value)


def test_mapping_changes_keep_mixed_direct_and_latent_ids_separate():
    frames = np.arange(10, 13, dtype=np.int64)
    before = np.tile(np.arange(3), (3, 1))
    after = np.array([[0, 1, 2], [1, 2, 0], [1, 0, 2]], dtype=np.int64)
    direct = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 1]], bool)
    r = compare_mapping(frames, before, after, direct)
    np.testing.assert_array_equal(r["changed_native"], [11, 12])
    np.testing.assert_array_equal(r["direct_involved_change"][1], [1, 0, 1])
    np.testing.assert_array_equal(r["latent_only_change"][1], [0, 1, 0])
    np.testing.assert_array_equal(r["direct_involved_native"], [11])
    # One native may contain both categories; they describe different IDs.
    np.testing.assert_array_equal(r["latent_only_native"], [11, 12])


def test_incomplete_identity_mapping_is_not_silently_assigned():
    with pytest.raises(ValueError, match="complete"):
        compare_mapping(
            np.arange(2, dtype=np.int64),
            np.tile(np.arange(3), (2, 1)),
            np.array([[0, 1, 2], [0, -1, 2]], dtype=np.int64),
            np.ones((2, 3), bool),
        )

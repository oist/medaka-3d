"""Behavioral checks for reserving one observation-supported hypothesis."""

import numpy as np
from observation_reserve_selection import (
    observation_eligibility,
    select_with_observation_reserve,
)


def candidates(count=8):
    poses = np.zeros((count, 3, 3, 3))
    poses[:, 0, 0, 0] = 2 * np.arange(count)
    return (
        poses,
        np.arange(count, dtype=float),
        np.zeros(count, dtype=int),
        np.tile(np.arange(3), (count, 1)),
    )


def test_reserve_outsider_keeps_best_five_and_actual_costs():
    values = candidates()
    before = [a.copy() for a in values]
    eligible = np.array([False] * 7 + [True])
    selected, record = select_with_observation_reserve(*values, eligible)
    np.testing.assert_array_equal(selected, [0, 1, 2, 3, 4, 7])
    assert record["replacement"] == dict(removed=5, admitted=7, extra_path_cost=2.0)
    for first, second in zip(values, before, strict=True):
        np.testing.assert_array_equal(first, second)


def test_reservation_does_not_replace_an_existing_supported_branch():
    values = candidates()
    for eligible in (np.zeros(8, bool), np.arange(8) == 2):
        selected, record = select_with_observation_reserve(*values, eligible)
        np.testing.assert_array_equal(selected, np.arange(6))
        assert not record["changed"]


def test_duplicate_rule_protects_same_parent_and_identity_but_checks_all_nodes():
    poses, path, parent, perm = candidates(9)
    poses[6] = poses[0]  # Strong but duplicates state0: it must not replace it.
    poses[7] = poses[0]
    poses[7, 2, 2, 1] = 0.6  # Identical chests, distinct tail: not a duplicate.
    eligible = np.zeros(9, bool)
    eligible[6:8] = True
    selected, record = select_with_observation_reserve(poses, path, parent, perm, eligible)
    np.testing.assert_array_equal(selected, [0, 1, 2, 3, 4, 7])
    assert record["duplicate_rejections"] == [dict(candidate=6, duplicate_of=[0])]
    poses[7] = poses[0]
    parent[7] = 1  # A distinct history remains distinct under the existing rule.
    selected, _ = select_with_observation_reserve(poses, path, parent, perm, eligible)
    assert selected[-1] == 7


def test_supported_sources_exclude_mixed_rows_reuse_and_partial_views():
    count = 6
    sources = np.broadcast_to(np.arange(3)[None, :, None, None], (count, 3, 3, 3)).copy()
    observed = np.ones((count, 3, 3), bool)
    parts = np.zeros((count, 5))
    parts[:, (0, 4)] = 1
    parts[0, (0, 4)] = 2
    sources[0, 2] = -1
    observed[0, 2] = False
    sources[2, 0, 0, 2] = 1  # A tail from a different SLEAP body row.
    sources[3, 1] = sources[3, 0]  # Two modeled fish use the same detections.
    sources[4, :, 2] = -1  # Two-view support is insufficient for this reservation.
    parts[5, 4] = 3  # Better SLEAP fit but worse independent image fit.
    eligible, _ = observation_eligibility(sources, observed, parts, baseline_best=0)
    np.testing.assert_array_equal(eligible, [False, True, False, False, False, False])

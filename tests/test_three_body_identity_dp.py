"""Exhaustive small chains test exact global alternatives rather than local winners."""

import importlib.util
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "labs/20260909_three_body/identity_dp.py"
SPEC = importlib.util.spec_from_file_location("three_body_identity_dp_test", PATH)
DP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DP
SPEC.loader.exec_module(DP)


def exhaustive(unary, edges, initial=0):
    n = len(unary)
    choices = ([initial] if initial is not None else range(6), *([range(6)] * (n - 1)))
    paths = np.array(list(product(*choices)), np.int64)
    cost = np.array([
        sum(unary[t, p[t]] for t in range(n))
        + sum(edges[t-1, p[t-1], p[t]] for t in range(1, n)) for p in paths
    ])
    finite = np.isfinite(cost)
    paths, cost = paths[finite], cost[finite]
    marginal = np.full((n, 6), np.inf)
    for t, state in np.ndindex(n, 6):
        selected = cost[paths[:, t] == state]
        if len(selected):
            marginal[t, state] = selected.min()
    return paths, cost, marginal


@pytest.mark.parametrize("n,initial", [(4, 0), (5, 0), (4, None), (5, 4)])
def test_all_min_marginals_and_id_alternatives_match_every_possible_small_path(n, initial):
    rng = np.random.default_rng(n + (initial or 0))
    unary = rng.integers(-2, 15, size=(n, 6)).astype(float)
    edges = rng.integers(0, 20, size=(n-1, 6, 6)).astype(float)
    edges[0, 1, 2] = np.inf
    unary[2, 3] = np.inf
    frames = np.arange(100, 100+n)
    result = DP.solve_identity_dp(frames, unary, edges, initial_state=initial)
    paths, costs, marginal = exhaustive(unary, edges, initial)
    best = costs.min()
    assert result.status == "optimal" and result.path_cost == best
    np.testing.assert_array_equal(result.min_marginal_cost, marginal)
    np.testing.assert_array_equal(result.min_marginal_delta, marginal - best)
    np.testing.assert_array_equal(result.forward_cost + result.backward_cost, marginal)
    selected = np.all(paths == result.path_state, axis=1)
    assert selected.sum() == 1 and costs[selected][0] == best
    assert result.global_alternative_gap == costs[~selected].min() - best
    for t in range(n):
        other = costs[paths[:, t] != result.path_state[t]]
        expected = other.min() - best if len(other) else np.inf
        assert result.state_alternative_gap[t] == expected
        for identity in range(3):
            assignments = DP.PERMUTATIONS[paths[:, t], identity]
            for slot in range(3):
                compatible = costs[assignments == slot]
                expected = compatible.min() - best if len(compatible) else np.inf
                assert result.id_slot_min_marginal_delta[t, identity, slot] == expected
            alternative = costs[assignments != result.permutation_by_id[t, identity]]
            expected = alternative.min() - best if len(alternative) else np.inf
            assert result.identity_alternative_gap[t, identity] == expected


def test_prefix_and_suffix_costs_use_current_unary_exactly_once():
    unary = np.arange(24, dtype=float).reshape(4, 6)
    edges = np.arange(108, dtype=float).reshape(3, 6, 6) % 11
    result = DP.solve_identity_dp(np.arange(4), unary, edges)
    for t in range(4):
        for state in range(6):
            prefix_paths, prefix_cost, _ = exhaustive(unary[:t+1], edges[:t], 0)
            compatible = prefix_cost[prefix_paths[:, -1] == state]
            expected = compatible.min() if len(compatible) else np.inf
            assert result.forward_cost[t, state] == expected
            _, suffix_cost, _ = exhaustive(unary[t:], edges[t:], state)
            assert result.backward_cost[t, state] == suffix_cost.min() - unary[t, state]


def test_additive_constant_in_each_frame_does_not_change_gap_or_physical_path():
    rng = np.random.default_rng(63)
    unary, edges = rng.random((5, 6)), rng.random((4, 6, 6))
    a = DP.solve_identity_dp(np.arange(5), unary, edges)
    constants = np.array([10., -2., .75, -5., 100.])
    b = DP.solve_identity_dp(np.arange(5), unary + constants[:, None], edges)
    np.testing.assert_array_equal(a.path_state, b.path_state)
    assert b.path_cost == pytest.approx(a.path_cost + constants.sum(), abs=1e-12)
    np.testing.assert_allclose(a.min_marginal_delta, b.min_marginal_delta, atol=1e-12)
    np.testing.assert_allclose(a.identity_alternative_gap, b.identity_alternative_gap, atol=1e-12)
    assert a.global_alternative_gap == pytest.approx(b.global_alternative_gap, abs=1e-12)
    assert a.numerical_tie_tolerance == pytest.approx(b.numerical_tie_tolerance, abs=1e-20)
    np.testing.assert_array_equal(a.optimal_state_mask, b.optimal_state_mask)


def test_large_representable_common_offsets_do_not_manufacture_numerical_ties():
    unary = np.zeros((4, 6))
    unary[:, 1:] = np.array([1., 2., 3., 4., 5.]) / 1024
    edges = np.zeros((3, 6, 6))
    baseline = DP.solve_identity_dp(np.arange(4), unary, edges)
    changed = DP.solve_identity_dp(
        np.arange(4), unary + 2.**40, edges + 2.**39
    )
    np.testing.assert_array_equal(baseline.min_marginal_delta, changed.min_marginal_delta)
    np.testing.assert_array_equal(baseline.optimal_state_mask, changed.optimal_state_mask)
    assert baseline.numerical_tie_tolerance == changed.numerical_tie_tolerance
    assert changed.global_alternative_status == "finite_gap"
    assert changed.global_alternative_gap == 1 / 1024


def test_raw_slot_permutation_equivariance_with_unique_optimum_and_physical_geometry():
    rng = np.random.default_rng(122)
    unary, edges = rng.random((5, 6)), rng.random((4, 6, 6))
    baseline = DP.solve_identity_dp(np.arange(5), unary, edges)
    assert baseline.global_alternative_gap > baseline.numerical_tie_tolerance
    rename = np.array([rng.permutation(3) for _ in range(5)])  # new slot -> old slot
    maps = []
    for t in range(5):
        new_permutations = np.argsort(rename[t])[DP.PERMUTATIONS]
        maps.append(np.array([
            np.flatnonzero((row == DP.PERMUTATIONS).all(axis=1))[0] for row in new_permutations
        ]))
    maps = np.array(maps)
    mapped_unary, mapped_edges = np.empty_like(unary), np.empty_like(edges)
    for t in range(5):
        mapped_unary[t, maps[t]] = unary[t]
        if t:
            mapped_edges[t-1][np.ix_(maps[t-1], maps[t])] = edges[t-1]
    relabeled = DP.solve_identity_dp(
        np.arange(5), mapped_unary, mapped_edges, initial_state=int(maps[0, 0])
    )
    np.testing.assert_array_equal(relabeled.path_state, maps[np.arange(5), baseline.path_state])
    for t in range(5):
        np.testing.assert_allclose(
            relabeled.min_marginal_delta[t, maps[t]], baseline.min_marginal_delta[t], atol=1e-12
        )
    points = rng.normal(size=(5, 3, 3))
    moved = points[np.arange(5)[:, None], rename]
    physical_a = points[np.arange(5)[:, None], baseline.permutation_by_id]
    physical_b = moved[np.arange(5)[:, None], relabeled.permutation_by_id]
    np.testing.assert_array_equal(physical_a, physical_b)


def test_missing_or_identical_evidence_retains_all_ties_after_the_fixed_first_state():
    result = DP.solve_identity_dp(np.arange(5), np.zeros((5, 6)), np.zeros((4, 6, 6)))
    assert result.status == "optimal" and result.global_alternative_gap == 0
    assert result.global_alternative_status == "tie_within_tolerance"
    assert result.optimal_state_mask[0].sum() == 1
    assert result.optimal_state_mask[1:].all()
    assert np.isposinf(result.identity_alternative_gap[0]).all()
    assert (result.identity_alternative_gap[1:] == 0).all()
    assert (result.identity_alternative_status[0] == "fixed_initial_constraint").all()


def test_interior_swap_and_return_is_a_distinct_alternative_with_same_endpoints():
    unary = np.full((4, 6), 100.)
    unary[:, 0] = 0
    unary[1, 2] = 1
    result = DP.solve_identity_dp(np.arange(4), unary, np.zeros((3, 6, 6)))
    assert result.path_state.tolist() == [0, 0, 0, 0]
    assert result.global_alternative_gap == 1
    assert result.state_alternative_gap[1] == 1
    assert result.state_alternative_gap[-1] == 100
    assert result.min_marginal_delta[1, 2] == 1


def test_no_feasible_alternative_is_not_a_large_confidence_value():
    unary = np.full((3, 6), np.inf)
    unary[:, 0] = 0
    result = DP.solve_identity_dp(np.arange(3), unary, np.zeros((2, 6, 6)))
    assert np.isposinf(result.global_alternative_gap)
    assert result.global_alternative_status == "no_feasible_alternative"
    assert result.state_alternative_status.tolist() == [
        "fixed_initial_constraint", "no_feasible_alternative", "no_feasible_alternative"
    ]


def test_single_frame_fixed_and_free_initial_cases():
    unary = np.array([[0., 1, 2, 3, 4, 5]])
    free = DP.solve_identity_dp([40], unary, np.zeros((0, 6, 6)), initial_state=None)
    fixed = DP.solve_identity_dp([40], unary, np.zeros((0, 6, 6)), initial_state=3)
    assert free.path_cost == 0 and free.global_alternative_gap == 1
    assert fixed.path_cost == 3 and fixed.path_state.tolist() == [3]
    assert fixed.global_alternative_status == "no_feasible_alternative"


@pytest.mark.parametrize("where", ["initial", "middle", "edges"])
def test_infeasible_chain_has_no_fake_winner_or_infinite_certainty(where):
    unary, edges = np.zeros((4, 6)), np.zeros((3, 6, 6))
    if where == "edges":
        edges[1] = np.inf
    elif where == "middle":
        unary[2] = np.inf
    else:
        unary[0, 0] = np.inf
    result = DP.solve_identity_dp(np.arange(4), unary, edges)
    assert result.status == "infeasible" and np.isposinf(result.path_cost)
    assert (result.path_state == -1).all() and (result.permutation_by_id == -1).all()
    assert not result.optimal_state_mask.any()
    assert np.isnan(result.min_marginal_delta).all()
    assert np.isnan(result.global_alternative_gap)
    assert result.global_alternative_status == "infeasible"


@pytest.mark.parametrize("frames", [[1, 3], [1, 1], [1., 2.], [-1, 0], [], [True, False]])
def test_native_frame_gaps_and_invalid_indices_are_rejected(frames):
    with pytest.raises(ValueError):
        DP.solve_identity_dp(
            frames, np.zeros((len(frames), 6)), np.zeros((max(len(frames)-1, 0), 6, 6))
        )


@pytest.mark.parametrize("value", [np.nan, -np.inf, 1+2j])
def test_invalid_objective_values_are_rejected(value):
    unary = np.zeros((2, 6), dtype=complex if np.iscomplexobj(value) else float)
    unary[1, 2] = value
    with pytest.raises(ValueError):
        DP.solve_identity_dp([1, 2], unary, np.zeros((1, 6, 6)))


@pytest.mark.parametrize("initial", [True, -1, 6, .5])
def test_invalid_initial_states_fail(initial):
    with pytest.raises(ValueError):
        DP.solve_identity_dp([0], np.zeros((1, 6)), np.zeros((0, 6, 6)), initial_state=initial)


def test_float_overflow_is_an_explicit_error_not_silent_infeasibility():
    with pytest.raises(FloatingPointError):
        DP.solve_identity_dp([0, 1], np.full((2, 6), 1e308), np.full((1, 6, 6), 1e308))


def test_numerical_tie_tolerance_keeps_the_actual_positive_gap():
    result = DP.solve_identity_dp(
        [0], np.array([[0., 5e-11, 1, 2, 3, 4]]), np.zeros((0, 6, 6)), initial_state=None
    )
    assert result.global_alternative_gap == 5e-11
    assert result.global_alternative_status == "tie_within_tolerance"
    assert result.optimal_state_mask[0, :2].all()


def test_motion_matches_explicit_fixed_three_fish_sum_and_preserves_support_counts():
    xyz = np.array([[[0, 0, 0], [10, 0, 0], [20, 0, 0]],
                    [[3, 0, 0], [12, 0, 0], [99, 99, 99]]], float)
    observed = np.array([[True, True, True], [True, True, False]])
    result = DP.build_adjacent_motion_cost([0, 1], xyz, observed, scale_mm=2., weight=.5)
    for previous, current in product(range(6), repeat=2):
        p, q = DP.PERMUTATIONS[previous], DP.PERMUTATIONS[current]
        held = observed[0, p] & observed[1, q]
        expected = .5 / 3 * sum(
            np.sum((xyz[1, q[i]] - xyz[0, p[i]])**2) / 4 for i in range(3) if held[i]
        )
        assert result.transitions[0, previous, current] == expected
        assert result.supported_pair_count[0, previous, current] == held.sum()
        np.testing.assert_array_equal(result.supported_pairs[0, previous, current], held)
    assert result.transitions[0, 0, 0] == pytest.approx(.5 / 3 * (9 + 4) / 4)
    assert np.isnan(result.squared_distance_mm2[0, 0, 0, 2])


def test_motion_missing_pairs_can_evade_cost_and_are_explicitly_reported():
    xyz = np.array([[[0, 0, 0], [np.nan]*3, [np.nan]*3],
                    [[10, 0, 0], [np.nan]*3, [np.nan]*3]], float)
    held = np.array([[True, False, False], [True, False, False]])
    result = DP.build_adjacent_motion_cost([0, 1], xyz, held, scale_mm=1., weight=1.)
    assert result.transitions[0, 0, 0] > 0
    assert result.supported_pair_count[0, 0, 0] == 1
    alternative = np.flatnonzero(DP.PERMUTATIONS[:, 0] == 1)[0]
    assert result.transitions[0, 0, alternative] == 0
    assert result.supported_pair_count[0, 0, alternative] == 0
    assert result.missing_connection_cost_evasion_possible


def test_all_missing_motion_stays_neutral_and_never_uses_nan_latent_positions():
    motion = DP.build_adjacent_motion_cost(
        np.arange(3), np.full((3, 3, 3), np.nan), np.zeros((3, 3), bool)
    )
    assert (motion.transitions == 0).all() and not motion.supported_pairs.any()
    result = DP.solve_identity_dp(np.arange(3), np.zeros((3, 6)), motion.transitions)
    assert result.global_alternative_gap == 0
    assert result.optimal_state_mask[1:].all()

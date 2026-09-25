"""Independent path enumeration for last-direct-event identity histories."""

import importlib.util
import sys
from itertools import permutations, product
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load_module():
    saved, old_path = sys.modules.get("identity_dp"), sys.path.copy()
    try:
        sys.path.insert(0, str(LAB))
        for name in ("identity_dp", "identity_memory_dp"):
            spec = importlib.util.spec_from_file_location(
                "_memory_test_" + name, LAB / (name+".py")
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            if name == "identity_dp":
                sys.modules[name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        if saved is None:
            sys.modules.pop("identity_dp", None)
        else:
            sys.modules["identity_dp"] = saved


m = load_module()
P = np.array(list(permutations(range(3))), np.int64)


def brute(unary, xyz, observed, scale=.8, weight=.2):
    """Enumerate every P sequence; no state merging, graph, or DP helper is used."""
    n = len(unary)
    paths = np.array([(0, *suffix) for suffix in product(range(6), repeat=n-1)], np.int64)
    count = len(paths)
    remembered_frames = np.zeros((count, 3), np.int64)
    remembered_slots = np.tile(np.arange(3), (count, 1))
    events = np.zeros((count, n, 3), np.int64)
    components = unary[np.arange(n)[None], paths].copy()
    for t in range(n):
        slots = P[paths[:, t]]
        direct = observed[t, slots]
        if t:
            for identity in range(3):
                indices = np.flatnonzero(direct[:, identity])
                previous_t = remembered_frames[indices, identity]
                previous_slot = remembered_slots[indices, identity]
                elapsed = t-previous_t
                distance = np.square(
                    xyz[t, slots[indices, identity]] - xyz[previous_t, previous_slot]
                ).sum(axis=1)
                components[indices, t] += weight/3 * (
                    distance/(scale*scale*elapsed) + 3*np.log(elapsed)
                )
        remembered_frames = np.where(direct, t, remembered_frames)
        remembered_slots = np.where(direct, slots, remembered_slots)
        events[:, t] = 3*remembered_frames + remembered_slots
    prefix = np.cumsum(components, axis=1)
    suffix = np.zeros_like(components)
    suffix[:, :-1] = np.cumsum(components[:, :0:-1], axis=1)[:, ::-1]
    totals = prefix[:, -1]
    return paths, events, totals, prefix, suffix


def assert_enumerated(result, unary, xyz, observed):
    paths, events, costs, prefix, suffix = brute(unary, xyz, observed)
    best = costs.min()
    assert result.path_cost == pytest.approx(best, abs=2e-10)
    chosen = np.flatnonzero(np.all(paths == result.path_state, axis=1))
    assert len(chosen) == 1 and costs[chosen[0]] == pytest.approx(best, abs=2e-10)
    expected_marginal = np.full_like(unary, np.inf)
    for t, state in np.ndindex(len(unary), 6):
        matched = paths[:, t] == state
        if matched.any():
            expected_marginal[t, state] = costs[matched].min()
    np.testing.assert_allclose(result.min_marginal_cost, expected_marginal, atol=2e-10)
    np.testing.assert_allclose(result.min_marginal_delta, expected_marginal-best, atol=2e-10)
    for t in range(len(unary)):
        for node in range(result.layer_offsets[t], result.layer_offsets[t+1]):
            matched = ((paths[:, t] == result.state_permutation[node])
                       & (events[:, t] == result.state_last_event[node]).all(axis=1))
            assert matched.any()
            assert result.forward_cost[node] == pytest.approx(prefix[matched, t].min(), abs=2e-10)
            assert result.backward_cost[node] == pytest.approx(suffix[matched, t].min(), abs=2e-10)
            assert result.node_min_marginal_cost[node] == pytest.approx(
                costs[matched].min(), abs=2e-10
            )
        for identity, slot in np.ndindex(3, 3):
            allowed = P[paths[:, t], identity] == slot
            expected = costs[allowed].min()-best if allowed.any() else np.inf
            assert result.id_slot_min_marginal_delta[t, identity, slot] == pytest.approx(expected)
        alternatives = paths[:, t] != result.path_state[t]
        expected = costs[alternatives].min()-best if alternatives.any() else np.inf
        assert result.state_alternative_gap[t] == pytest.approx(expected, abs=2e-10)
        for identity in range(3):
            other = P[paths[:, t], identity] != result.permutation_by_id[t, identity]
            expected = costs[other].min()-best if other.any() else np.inf
            assert result.identity_alternative_gap[t, identity] == pytest.approx(
                expected, abs=2e-10
            )
    alternatives = np.any(paths != result.path_state, axis=1)
    expected = costs[alternatives].min()-best if alternatives.any() else np.inf
    assert result.global_alternative_gap == pytest.approx(expected, abs=2e-10)
    assert result.edge_offsets[-1] == 6*np.diff(result.layer_offsets)[:-1].sum()
    assert result.state_alternative_status[0] == "fixed_initial_constraint"


@pytest.mark.parametrize("n,seed", [(4, 19), (5, 27), (7, 43)])
def test_full_path_enumeration_all_memory_nodes_marginals_and_identity_gaps(n, seed):
    rng = np.random.default_rng(seed)
    xyz, unary = rng.normal(size=(n, 3, 3)), rng.uniform(0, 2, size=(n, 6))
    observed = rng.random((n, 3)) > .4
    observed[0] = True
    # A fully observed ending explicitly exercises merging without a new seed.
    observed[-1] = True
    result = m.solve_identity_memory_dp(np.arange(n), unary, xyz, observed)
    assert_enumerated(result, unary, xyz, observed)
    assert np.diff(result.layer_offsets)[-1] == 6
    assert not result.terminal_unsettled_gap.any()


def test_all_direct_reduces_to_old_adjacent_objective_and_six_states():
    rng, n = np.random.default_rng(20), 6
    xyz, unary = rng.normal(size=(n, 3, 3)), rng.normal(size=(n, 6))
    direct = np.ones((n, 3), bool)
    result = m.solve_identity_memory_dp(np.arange(n), unary, xyz, direct)
    old = sys.modules["_memory_test_identity_dp"]
    transitions = old.build_adjacent_motion_cost(np.arange(n), xyz, direct).transitions
    expected = old.solve_identity_dp(np.arange(n), unary, transitions)
    np.testing.assert_array_equal(np.diff(result.layer_offsets), [1, 6, 6, 6, 6, 6])
    np.testing.assert_array_equal(result.path_state, expected.path_state)
    np.testing.assert_allclose(result.min_marginal_cost, expected.min_marginal_cost, atol=2e-12)
    np.testing.assert_allclose(result.min_marginal_delta, expected.min_marginal_delta, atol=2e-12)
    assert result.path_cost == pytest.approx(expected.path_cost, abs=2e-12)


def test_same_current_permutation_can_have_different_memories_and_future_optimal_prefix():
    rng = np.random.default_rng(46)
    n = 5
    xyz, unary = rng.normal(0, 5, (n, 3, 3)), rng.uniform(0, .2, (n, 6))
    direct = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]], bool)
    result = m.solve_identity_memory_dp(np.arange(n), unary, xyz, direct)
    assert_enumerated(result, unary, xyz, direct)
    reversals = []
    for t in range(1, n-1):
        layer = np.arange(result.layer_offsets[t], result.layer_offsets[t+1])
        for p in range(6):
            nodes = layer[result.state_permutation[layer] == p]
            if len(nodes) > 1:
                cheap_prefix = nodes[np.argmin(result.forward_cost[nodes])]
                best_complete = nodes[np.argmin(result.node_min_marginal_cost[nodes])]
                if cheap_prefix != best_complete:
                    reversals.append((cheap_prefix, best_complete))
    assert reversals, "P alone is insufficient: a costlier history wins after the future endpoint"


def test_merged_state_keeps_other_prefixes_for_earlier_alternative_gaps():
    xyz = np.zeros((4, 3, 3))
    direct = np.ones((4, 3), bool)
    unary = np.zeros((4, 6))
    unary[1, 1:] = np.arange(1, 6)
    unary[2:, 1:] = 100
    result = m.solve_identity_memory_dp(np.arange(4), unary, xyz, direct)
    assert_enumerated(result, unary, xyz, direct)
    assert result.path_state.tolist() == [0, 0, 0, 0]
    assert result.global_alternative_gap == 1
    node_t2_p0 = result.layer_offsets[2]
    incoming = np.flatnonzero(result.edge_target == node_t2_p0)
    assert len(incoming) == 6  # Costlier histories were not discarded from the graph.


def test_terminal_gaps_keep_last_direct_events_without_fabricated_coordinates():
    xyz = np.zeros((4, 3, 3))
    xyz[:, :, 0] = [0, 10, 20]
    direct = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0]], bool)
    unary = np.full((4, 6), np.inf)
    unary[:, 0] = 0
    xyz[~direct] = np.nan
    result = m.solve_identity_memory_dp(np.arange(100, 104), unary, xyz, direct)
    np.testing.assert_array_equal(result.terminal_last_event_frame, [103, 101, 100])
    np.testing.assert_array_equal(result.terminal_last_event_slot, [0, 1, 2])
    np.testing.assert_array_equal(result.terminal_age_frames, [0, 2, 3])
    np.testing.assert_array_equal(result.terminal_unsettled_gap, [False, True, True])
    assert result.global_alternative_status == "no_feasible_alternative"
    assert not result.terminal_missing_endpoint_cost_applied
    assert not result.observation_occurrence_model_used
    assert result.path_cost == 0


def test_reobservation_pays_entire_gap_and_log_term_once():
    xyz = np.zeros((4, 3, 3))
    direct = np.ones((4, 3), bool)
    direct[1:3, 0] = False
    xyz[~direct] = np.nan
    xyz[3, 0, 0] = 2.4
    unary = np.full((4, 6), np.inf)
    unary[:, 0] = 0
    result = m.solve_identity_memory_dp(np.arange(4), unary, xyz, direct)
    assert result.path_cost == pytest.approx(.2/3*((2.4**2)/(.8**2*3)+3*np.log(3)))
    edge = np.flatnonzero((result.edge_source == result.path_node[2])
                         & (result.edge_target == result.path_node[3]))
    assert len(edge) == 1
    np.testing.assert_array_equal(result.edge_delta_frames[edge[0]], [3, 1, 1])


def test_raw_slot_shuffle_preserves_unique_optimal_physical_identity_and_marginal_set():
    rng, n = np.random.default_rng(82), 5
    xyz, unary = rng.normal(size=(n, 3, 3)), rng.uniform(size=(n, 6))
    direct = rng.random((n, 3)) > .4
    direct[0] = True
    result = m.solve_identity_memory_dp(np.arange(n), unary, xyz, direct)
    assert result.global_alternative_gap > 1e-6
    shuffle = P[rng.integers(6, size=n)]
    shuffle[0] = P[0]  # Same registered first-frame slot/ID convention.
    changed_unary = np.empty_like(unary)
    original_p = np.empty((n, 6), int)
    for t, p in np.ndindex(n, 6):
        mapping = shuffle[t, P[p]]
        original_p[t, p] = np.flatnonzero((mapping == P).all(axis=1))[0]
        changed_unary[t, p] = unary[t, original_p[t, p]]
    changed = m.solve_identity_memory_dp(
        np.arange(n), changed_unary, xyz[np.arange(n)[:, None], shuffle],
        direct[np.arange(n)[:, None], shuffle],
    )
    mapped = shuffle[np.arange(n)[:, None], changed.permutation_by_id]
    np.testing.assert_array_equal(mapped, result.permutation_by_id)
    np.testing.assert_allclose(
        changed.min_marginal_delta,
        result.min_marginal_delta[np.arange(n)[:, None], original_p], atol=2e-12,
    )


def test_large_common_unary_offsets_preserve_small_gaps_and_tie_policy():
    xyz = np.zeros((5, 3, 3))
    direct = np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]], bool)
    unary = np.tile(np.arange(6)/16, (5, 1))
    a = m.solve_identity_memory_dp(np.arange(5), unary, xyz, direct)
    b = m.solve_identity_memory_dp(np.arange(5), unary+2**40, xyz, direct)
    np.testing.assert_array_equal(a.path_state, b.path_state)
    np.testing.assert_array_equal(a.min_marginal_delta, b.min_marginal_delta)
    np.testing.assert_array_equal(a.optimal_state_mask, b.optimal_state_mask)
    assert a.numerical_tie_tolerance == b.numerical_tie_tolerance
    assert b.common_cost_offset-a.common_cost_offset == 5*2**40


def test_all_missing_after_initial_retains_ties_and_unsettled_terminal_events():
    xyz = np.zeros((4, 3, 3))
    direct = np.zeros((4, 3), bool)
    direct[0] = True
    result = m.solve_identity_memory_dp(np.arange(4), np.zeros((4, 6)), xyz, direct)
    assert_enumerated(result, np.zeros((4, 6)), xyz, direct)
    assert result.optimal_state_mask[1:].all()
    assert result.global_alternative_status == "tie_within_tolerance"
    np.testing.assert_array_equal(result.terminal_age_frames, [3, 3, 3])


def test_single_frame_has_only_fixed_initial_path_and_no_edges():
    result = m.solve_identity_memory_dp([7], np.zeros((1, 6)), np.zeros((1, 3, 3)),
                                      np.ones((1, 3), bool))
    assert result.path_state.tolist() == [0] and result.edge_source.shape == (0,)
    assert result.global_alternative_status == "no_feasible_alternative"
    assert result.terminal_age_frames.tolist() == [0, 0, 0]


def test_infeasible_unary_chain_has_no_lexical_winner_or_confident_terminal_flags():
    unary = np.zeros((4, 6))
    unary[2] = np.inf
    result = m.solve_identity_memory_dp(np.arange(4), unary, np.zeros((4, 3, 3)),
                                      np.ones((4, 3), bool))
    assert result.status == "infeasible" and np.isinf(result.path_cost)
    assert (result.path_node == -1).all() and (result.path_state == -1).all()
    assert np.isnan(result.min_marginal_delta).all()
    assert result.global_alternative_status == "infeasible"
    assert (result.terminal_age_frames == -1).all()


@pytest.mark.parametrize("cap,value,resource", [
    ("max_layer_states", 5, "layer_states"),
    ("max_total_states", 6, "total_states"),
    ("max_total_edges", 5, "total_edges"),
])
def test_capacity_failure_stops_without_pruning_or_partial_solution(cap, value, resource):
    with pytest.raises(m.IdentityMemoryCapacityError) as caught:
        m.solve_identity_memory_dp(np.arange(4), np.zeros((4, 6)), np.zeros((4, 3, 3)),
                                   np.ones((4, 3), bool), **{cap: value})
    assert caught.value.frame_offset == 1 and caught.value.resource == resource
    assert caught.value.attempted > value and "no pruning/partial solution" in str(caught.value)


def test_memory_states_count_against_cap_even_when_unary_forbids_their_prefix():
    observed = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]], bool)
    unary = np.full((4, 6), np.inf)
    unary[:, 0] = 0
    with pytest.raises(m.IdentityMemoryCapacityError) as caught:
        m.solve_identity_memory_dp(np.arange(4), unary, np.zeros((4, 3, 3)), observed,
                                   max_layer_states=6)
    assert caught.value.frame_offset == 2 and caught.value.resource == "layer_states"


def test_graph_edges_preserve_all_successors_and_remembered_direct_event_sources():
    rng, n = np.random.default_rng(21), 5
    xyz = rng.normal(size=(n, 3, 3))
    observed = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1]], bool)
    original = xyz.copy()
    result = m.solve_identity_memory_dp(np.arange(100, 105), np.zeros((n, 6)), xyz, observed)
    np.testing.assert_array_equal(xyz, original)
    for t in range(1, n):
        edges = slice(result.edge_offsets[t-1], result.edge_offsets[t])
        src, dst = result.edge_source[edges], result.edge_target[edges]
        for node in range(result.layer_offsets[t-1], result.layer_offsets[t]):
            chosen = dst[src == node]
            np.testing.assert_array_equal(result.state_permutation[chosen], np.arange(6))
        for edge in range(result.edge_offsets[t-1], result.edge_offsets[t]):
            prior, target = result.edge_source[edge], result.edge_target[edge]
            slots = P[result.state_permutation[target]]
            before = result.state_last_event[prior]
            after = result.state_last_event[target]
            for identity, slot in enumerate(slots):
                if observed[t, slot]:
                    assert after[identity] == 3*t+slot
                    assert result.edge_delta_frames[edge, identity] == t-before[identity]//3
                else:
                    assert after[identity] == before[identity]
                    assert result.edge_delta_frames[edge, identity] == 0
                    assert np.isnan(result.edge_squared_distance_mm2[edge, identity])


@pytest.mark.parametrize("problem", [
    "frame_gap", "negative_frame", "float_frame", "initial_missing", "initial_nonzero",
    "initial_bool", "bad_shape", "nonfinite_direct", "float_support", "nan_unary",
    "negative_inf_unary", "scale_zero", "negative_weight", "bool_cap", "zero_cap",
])
def test_invalid_registration_or_numerics_are_rejected(problem):
    frames, unary, xyz, direct, kwargs = (
        np.arange(4), np.zeros((4, 6)), np.zeros((4, 3, 3)), np.ones((4, 3), bool), {}
    )
    if problem == "frame_gap":
        frames[2:] += 1
    elif problem == "negative_frame":
        frames -= 1
    elif problem == "float_frame":
        frames = frames.astype(float)
    elif problem == "initial_missing":
        direct[0, 0] = False
    elif problem == "initial_nonzero":
        kwargs["initial_state"] = 1
    elif problem == "initial_bool":
        kwargs["initial_state"] = False
    elif problem == "bad_shape":
        xyz = xyz[:, :, :2]
    elif problem == "nonfinite_direct":
        xyz[1, 0, 0] = np.inf
    elif problem == "float_support":
        direct = direct.astype(float)
    elif problem == "nan_unary":
        unary[1, 0] = np.nan
    elif problem == "negative_inf_unary":
        unary[1, 0] = -np.inf
    elif problem == "scale_zero":
        kwargs["scale_mm"] = 0
    elif problem == "negative_weight":
        kwargs["weight"] = -.2
    elif problem == "bool_cap":
        kwargs["max_total_edges"] = True
    else:
        kwargs["max_layer_states"] = 0
    with pytest.raises(ValueError):
        m.solve_identity_memory_dp(frames, unary, xyz, direct, **kwargs)


def test_finite_input_overflow_is_not_silently_treated_as_infeasible():
    xyz = np.zeros((2, 3, 3))
    xyz[1] = 1e308
    with pytest.raises(FloatingPointError):
        m.solve_identity_memory_dp([0, 1], np.zeros((2, 6)), xyz, np.ones((2, 3), bool))

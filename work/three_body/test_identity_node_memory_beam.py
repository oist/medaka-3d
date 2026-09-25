"""Tiny exhaustive path oracles; no actual medaka, GPU or changed exact solver."""

import dataclasses
import itertools

import identity_node_memory_beam as beam
import numpy as np
import pytest


def case(seed=7, n=4, k=2):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n, 3, k, 3)) * 0.4
    xyz[..., 0] += np.arange(3)[None, :, None] * 1.2
    direct = rng.random((n, 3, k)) > 0.4
    direct[0] = True
    xyz[~direct] = np.nan
    return np.arange(100, 100 + n), rng.uniform(-0.4, 0.8, (n, 6)), xyz, direct


def solve(values, **kwargs):
    return beam.solve_identity_node_memory_beam(
        *values, node_indices=(1,) if values[3].shape[2] == 1 else (1, 0), **kwargs
    )


def frozen_result(values):
    return beam.frozen.solve_identity_node_memory_dp(
        *values, node_indices=(1,) if values[3].shape[2] == 1 else (1, 0)
    )


def enumerate_retained(result, values):
    """Independent coordinate/last-observation cost; graph is used ONLY as membership.

    Enumerates whole permutation sequences, so paths that swap and return before
    the end remain distinct. No saved F/B/edge cost computes any oracle score.
    """
    frames, unary, xyz, direct = values
    n, k = len(frames), direct.shape[2]
    lookup = {}
    for t in range(n):
        for node in range(result.layer_offsets[t], result.layer_offsets[t + 1]):
            lookup[
                t, int(result.state_permutation[node]), tuple(result.state_last_event[node].flat)
            ] = node
    edges = set(zip(result.edge_source.tolist(), result.edge_target.tolist(), strict=True))
    paths = []
    marginal = np.full((n, 6), np.inf)
    backward = np.full(len(result.state_permutation), np.inf)
    for suffix in itertools.product(range(6), repeat=n - 1):
        path = (0, *suffix)
        nodes, increments = [], []
        for t, p in enumerate(path):
            memory = []
            charge = float(unary[t, p])
            for identity, node_index in np.ndindex(3, k):
                raw = int(beam.frozen.PERMUTATIONS[p, identity])
                last = next(
                    old
                    for old in range(t, -1, -1)
                    if direct[old, beam.frozen.PERMUTATIONS[path[old], identity], node_index]
                )
                last_slot = int(beam.frozen.PERMUTATIONS[path[last], identity])
                memory.append(3 * last + last_slot)
                if t == 0 or not direct[t, raw, node_index]:
                    continue
                prior = next(
                    old
                    for old in range(t - 1, -1, -1)
                    if direct[old, beam.frozen.PERMUTATIONS[path[old], identity], node_index]
                )
                prior_slot = int(beam.frozen.PERMUTATIONS[path[prior], identity])
                distance = sum(
                    (
                        float(xyz[t, raw, node_index, d])
                        - float(xyz[prior, prior_slot, node_index, d])
                    )
                    ** 2
                    for d in range(3)
                )
                elapsed = t - prior
                charge += (
                    result.weight
                    / 3
                    * (distance / (result.scale_mm**2 * elapsed) + 3 * np.log(elapsed))
                )
            index = lookup.get((t, p, tuple(memory)))
            if index is None or (nodes and (nodes[-1], index) not in edges):
                break
            nodes.append(index)
            increments.append(charge)
        else:
            cost = sum(increments)
            paths.append((cost, path, nodes))
            for t, p in enumerate(path):
                marginal[t, p] = min(marginal[t, p], cost)
                backward[nodes[t]] = min(backward[nodes[t]], sum(increments[t + 1 :]))
    return paths, marginal, backward


def assert_exact_common(actual, expected):
    for field in dataclasses.fields(expected):
        name = field.name
        candidate = "conditional_" + name if hasattr(actual, "conditional_" + name) else name
        if not hasattr(actual, candidate) or name in ("status", "approximation_used"):
            continue
        left, right = getattr(actual, candidate), getattr(expected, name)
        if isinstance(right, np.ndarray):
            assert left.shape == right.shape and left.dtype == right.dtype, name
            assert left.tobytes() == right.tobytes(), name
        elif isinstance(right, float):
            assert np.asarray(left).tobytes() == np.asarray(right).tobytes(), name
        else:
            assert left == right, name


@pytest.mark.parametrize("seed,k", [(0, 1), (19, 1), (0, 2), (7, 2), (19, 2)])
def test_no_pruning_matches_frozen_full_graph_and_independent_oracle(seed, k, monkeypatch):
    values = case(seed, k=k)
    values[1][1, 3] = np.inf
    expected = frozen_result(values)
    monkeypatch.setattr(
        beam.frozen,
        "solve_identity_node_memory_dp",
        lambda *a, **kw: pytest.fail("beam called whole exact solver"),
    )
    actual = solve(values, beam_width=100_000)
    assert type(actual) is beam.IdentityNodeMemoryBeamResult
    assert not isinstance(actual, beam.frozen.IdentityNodeMemoryDPResult)
    assert_exact_common(actual, expected)
    assert not actual.pruning_occurred and not actual.finite_prefixes_discarded
    assert actual.global_optimality_certified is False
    paths, marginal, backward = enumerate_retained(actual, values)
    np.testing.assert_allclose(actual.path_cost, min(p[0] for p in paths), atol=1e-12, rtol=1e-13)
    np.testing.assert_allclose(
        actual.conditional_min_marginal_cost, marginal, atol=1e-12, rtol=1e-13
    )
    np.testing.assert_allclose(actual.backward_cost, backward, atol=1e-12, rtol=1e-13)


@pytest.mark.parametrize("width", [1, 2, 6, 12])
@pytest.mark.parametrize("seed", [0, 7])
def test_truncated_graph_all_alternatives_match_independent_whole_path_oracle(width, seed):
    values = case(seed, n=5)
    actual = solve(values, beam_width=width)
    assert actual.pruning_occurred and actual.finite_prefixes_discarded
    paths, marginal, _ = enumerate_retained(actual, values)
    best = min(p[0] for p in paths)
    np.testing.assert_allclose(actual.path_cost, best, atol=1e-12, rtol=1e-13)
    np.testing.assert_allclose(
        actual.conditional_min_marginal_cost, marginal, atol=1e-12, rtol=1e-13
    )
    np.testing.assert_allclose(
        actual.conditional_min_marginal_delta, marginal - best, atol=1e-12, rtol=1e-13
    )
    changed_paths = [p[0] for p in paths if p[1] != tuple(actual.path_state)]
    expected_gap = min(changed_paths, default=np.inf) - best
    np.testing.assert_allclose(
        actual.conditional_global_alternative_gap, expected_gap, atol=1e-12, rtol=1e-13
    )
    for t, identity in np.ndindex(len(values[0]), 3):
        slot = actual.permutation_by_id[t, identity]
        other = [p[0] for p in paths if beam.frozen.PERMUTATIONS[p[1][t], identity] != slot]
        np.testing.assert_allclose(
            actual.conditional_identity_alternative_gap[t, identity],
            min(other, default=np.inf) - best,
            atol=1e-12,
            rtol=1e-13,
        )
    assert actual.gaps_conditional_on_retained_graph and not actual.global_optimality_certified
    assert not actual.physical_ID_truth and not actual.review_flags_cleared
    assert not actual.intermediate_ID_reset


def test_swap_and_return_alternative_survives_merging_and_all_incoming_edges():
    frames = np.arange(100, 103)
    xyz, direct = np.zeros((3, 3, 2, 3)), np.ones((3, 3, 2), bool)
    unary = np.array([[0, 1, 1, 1, 1, 1], [0, 0.2, 3, 4, 5, 6], [0, 10, 11, 12, 13, 14]])
    values = frames, unary, xyz, direct
    result = solve(values, beam_width=2)
    np.testing.assert_array_equal(result.path_state, [0, 0, 0])
    assert result.retained_edge_counts.tolist() == [0, 2, 4]
    assert result.conditional_global_alternative_gap == pytest.approx(0.2)
    assert result.conditional_state_alternative_gap[1] == pytest.approx(0.2)
    assert result.conditional_state_alternative_gap[2] == pytest.approx(10.0)
    paths, _, _ = enumerate_retained(result, values)
    assert any(p[1] == (0, 1, 0) and p[0] == pytest.approx(0.2) for p in paths)


def test_missing_alternatives_are_unknown_after_pruning():
    result = solve(case(n=3), beam_width=1)
    assert result.conditional_global_alternative_status == "not_retained_unknown"
    assert np.isposinf(result.conditional_global_alternative_gap)
    assert np.all(result.conditional_state_alternative_status[1:] == "not_retained_unknown")
    assert np.all(result.conditional_identity_alternative_status[1:] == "not_retained_unknown")
    assert result.conditional_state_alternative_status[0] == "fixed_initial_constraint"
    assert np.all(result.conditional_identity_alternative_status[0] == "fixed_initial_constraint")


def test_pruned_finite_prefix_can_contain_true_optimum_so_infinite_gap_is_not_certification():
    values = case(7, n=5)
    full = frozen_result(values)
    result = solve(values, beam_width=1)
    assert result.path_cost > full.path_cost + 0.1
    assert not np.array_equal(result.path_state, full.path_state)
    assert result.finite_prefixes_discarded and result.approximation_used
    assert np.isposinf(result.conditional_global_alternative_gap)
    assert result.conditional_global_alternative_status == "not_retained_unknown"
    assert result.global_optimality_certified is False


def test_discarded_infinite_states_and_finite_prefixes_are_reported_separately():
    values = list(case(n=3))
    values[2][:] = 0
    values[3][:] = True
    values[1][:] = np.inf
    values[1][:, 0] = 0
    result = solve(values, beam_width=1)
    assert result.pruning_occurred and result.approximation_used
    assert not result.finite_prefixes_discarded
    assert result.discarded_finite_state_counts.sum() == 0
    assert result.discarded_state_counts.sum() == 10
    assert not result.global_optimality_certified


@pytest.mark.parametrize("fault", ["initial", "middle", "last"])
def test_infeasible_retained_graph_is_not_global_infeasibility(fault):
    values = case(n=4)
    t = {"initial": 0, "middle": 2, "last": 3}[fault]
    values[1][t] = np.inf
    result = solve(values, beam_width=1)
    assert result.status == "infeasible_on_retained_graph"
    assert np.all(result.path_state == -1) and np.all(result.permutation_by_id == -1)
    assert np.isnan(result.conditional_min_marginal_delta).all()
    assert not result.global_optimality_certified
    assert np.all(result.conditional_state_alternative_status == "retained_infeasible_unknown")


@pytest.mark.parametrize(
    "kind", ["ties", "huge_offset", "terminal_missing", "no_observations", "single_frame"]
)
def test_no_pruning_numeric_and_memory_edge_cases_match_exact(kind):
    values = list(case(n=1 if kind == "single_frame" else 4))
    if kind == "ties":
        values[2][:] = 0
        values[3][:] = True
        values[1][:] = 0
    elif kind == "huge_offset":
        values[1] += 1e14
    elif kind == "terminal_missing":
        values[3][-2:, 2, 1] = False
        values[2][-2:, 2, 1] = np.nan
    elif kind == "no_observations":
        values[3][1:] = False
        values[2][1:] = np.nan
    result, expected = solve(values, beam_width=100_000), frozen_result(values)
    assert_exact_common(result, expected)
    if kind == "no_observations":
        assert result.terminal_unsettled_gap.all()
        assert np.all(result.terminal_age_frames == 3)


def test_deterministic_ranking_keeps_lexicographic_full_states_not_cost_order():
    values = list(case(n=4))
    values[2][:] = 0
    values[3][:] = True
    values[1][:] = 0
    first, second = solve(values, beam_width=2), solve(values, beam_width=2)
    for field in dataclasses.fields(first):
        a, b = getattr(first, field.name), getattr(second, field.name)
        if isinstance(a, np.ndarray):
            assert a.tobytes() == b.tobytes()
    assert first.state_permutation.tolist() == [0, 0, 1, 0, 1, 0, 1]
    np.testing.assert_array_equal(first.path_state, [0, 0, 0, 0])
    values[1][1] = [3, 2, 1, 0, 4, 5]
    reordered = solve(values, beam_width=2)
    assert reordered.state_permutation[1:3].tolist() == [2, 3]  # lex order, not winner-first
    assert reordered.path_state[1] == 3


def test_counts_keep_expansion_and_retention_scopes_distinct():
    result = solve(case(n=4), beam_width=1)
    assert result.expanded_state_counts.tolist() == [1, 6, 6, 6]
    assert result.retained_state_counts.tolist() == [1, 1, 1, 1]
    assert result.discarded_state_counts.tolist() == [0, 5, 5, 5]
    assert result.expanded_edge_counts.tolist() == [0, 6, 6, 6]
    assert result.retained_edge_counts.tolist() == [0, 1, 1, 1]
    assert result.discarded_edge_counts.tolist() == [0, 5, 5, 5]
    assert result.discarded_finite_state_counts.tolist() == [0, 5, 5, 5]
    assert result.layer_offsets[-1] == result.retained_state_counts.sum()
    assert result.edge_offsets[-1] == result.retained_edge_counts.sum()
    assert "stored retained" in result.total_states_scope
    assert "all generated" in result.total_edges_scope


@pytest.mark.parametrize(
    "kwargs,resource,offset,attempted",
    [
        ({"max_layer_states": 5}, "layer_states", 1, 6),
        ({"max_total_states": 2}, "total_states", 2, 3),
        ({"max_total_edges": 11}, "total_edges", 2, 12),
    ],
)
def test_caps_stop_with_no_result_including_generated_discarded_edges(
    kwargs, resource, offset, attempted
):
    with pytest.raises(beam.IdentityBeamCapacityError) as error:
        solve(case(n=4), beam_width=1, **kwargs)
    assert error.value.resource == resource
    assert error.value.frame_offset == offset and error.value.attempted == attempted


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_layer_states", 100001),
        ("max_total_states", 1000001),
        ("max_total_edges", 6000001),
        ("beam_width", 0),
        ("beam_width", True),
        ("beam_width", 100001),
    ],
)
def test_caps_cannot_be_silently_raised_or_beam_width_invalid(key, value):
    with pytest.raises(ValueError):
        solve(case(n=2), **{key: value})


def test_input_bytes_flags_and_independent_chronological_path_ledger_preserved():
    values = case(n=5)
    before = [(v.tobytes(), v.flags.writeable) for v in values]
    result = solve(values, beam_width=6)
    assert before == [(v.tobytes(), v.flags.writeable) for v in values]
    for field in dataclasses.fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            assert not any(np.shares_memory(value, original) for original in values)
    ledger = beam.frozen.node_memory_path_links(
        values[0], values[2], values[3], result.permutation_by_id
    )
    unary_cost = sum(values[1][t, p] for t, p in enumerate(result.path_state))
    assert result.path_cost == pytest.approx(
        ledger["motion_cost"] + unary_cost, abs=1e-12, rel=1e-13
    )
    for key in (
        "terminal_last_event_frame",
        "terminal_last_event_slot",
        "terminal_age_frames",
        "terminal_unsettled_gap",
    ):
        np.testing.assert_array_equal(getattr(result, key), ledger[key])


@pytest.mark.parametrize(
    "fault", ["frame_gap", "initial_missing", "direct_nan", "wrong_order", "reset", "nan_unary"]
)
def test_invalid_native_or_node_contract_rejected(fault):
    values = list(case(n=3))
    kwargs = {}
    if fault == "frame_gap":
        values[0][1] += 1
    elif fault == "initial_missing":
        values[3][0, 0, 0] = False
    elif fault == "direct_nan":
        values[2][0, 0, 0] = np.nan
    elif fault == "wrong_order":
        kwargs["node_indices"] = (0, 1)
    elif fault == "reset":
        kwargs["initial_state"] = 1
    else:
        values[1][1, 0] = np.nan
    with pytest.raises(ValueError):
        beam.solve_identity_node_memory_beam(*values, **kwargs)

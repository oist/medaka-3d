"""Exhaustive small-path oracle for shared-ID, independent-node memory."""

import sys
from dataclasses import fields
from itertools import product
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"))
from identity_memory_dp import solve_identity_memory_dp
from identity_node_memory_dp import (
    PERMUTATIONS,
    IdentityMemoryCapacityError,
    node_memory_path_links,
    solve_identity_node_memory_dp,
)


def oracle(xyz, direct, unary):
    """Enumerate P sequences; find past observations by backwards search."""
    n, _, k = direct.shape
    paths = [(0, *tail) for tail in product(range(6), repeat=n - 1)]
    values = []
    for path in paths:
        value = sum(unary[t, p] for t, p in enumerate(path))
        for t, identity, node in product(range(1, n), range(3), range(k)):
            raw = PERMUTATIONS[path[t], identity]
            if not direct[t, raw, node]:
                continue
            previous = next(
                j for j in range(t - 1, -1, -1) if direct[j, PERMUTATIONS[path[j], identity], node]
            )
            displacement = (
                xyz[t, raw, node] - xyz[previous, PERMUTATIONS[path[previous], identity], node]
            )
            value += (
                0.2
                / 3
                * (
                    sum(float(v) ** 2 for v in displacement) / (0.8**2 * (t - previous))
                    + 3 * np.log(t - previous)
                )
            )
        values.append(value)
    paths, values = np.array(paths), np.array(values)
    marginal = np.full((n, 6), np.inf)
    for t, p in np.ndindex(n, 6):
        matching = values[paths[:, t] == p]
        if len(matching):
            marginal[t, p] = min(matching)
    return paths, values, marginal


@pytest.mark.parametrize("seed", range(5))
def test_exhaustive_shared_permutation_and_full_min_marginals(seed):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(4, 3, 2, 3))
    direct = rng.random((4, 3, 2)) > 0.45
    direct[0] = True
    xyz[~direct] = np.nan
    unary = rng.uniform(0, 2, (4, 6))
    if seed == 1:
        unary[2, 3:] = np.inf
    result = solve_identity_node_memory_dp(np.arange(70, 74), unary, xyz, direct)
    paths, costs, marginal = oracle(xyz, direct, unary)
    np.testing.assert_allclose(result.path_cost, costs.min(), rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(result.path_state, paths[costs.argmin()])
    np.testing.assert_allclose(result.min_marginal_cost, marginal, rtol=1e-13, atol=1e-13)
    for identity, raw in np.ndindex(3, 3):
        expect = marginal[:, PERMUTATIONS[:, identity] == raw].min(axis=1) - costs.min()
        np.testing.assert_allclose(
            result.id_slot_min_marginal_delta[:, identity, raw], expect, rtol=1e-12, atol=1e-12
        )
    alternative = marginal - costs.min()
    alternative[np.arange(4), result.path_state] = np.inf
    np.testing.assert_allclose(result.global_alternative_gap, alternative.min(), atol=1e-12)
    links = node_memory_path_links(np.arange(70, 74), xyz, direct, result.permutation_by_id)
    expected_motion = result.path_cost - unary[np.arange(4), result.path_state].sum()
    np.testing.assert_allclose(links["motion_cost"], expected_motion, atol=1e-12)
    for name in (
        "terminal_last_event_frame",
        "terminal_last_event_slot",
        "terminal_age_frames",
        "terminal_unsettled_gap",
    ):
        np.testing.assert_array_equal(getattr(result, name), links[name])
    assert result.state_last_event.shape[1:] == (3, 2)
    assert result.edge_delta_frames.shape[1:] == (3, 2)


@pytest.mark.parametrize("seed", range(3))
def test_chest_only_all_old_outputs_exact(seed):
    rng = np.random.default_rng(seed + 12)
    xyz = rng.normal(size=(5, 3, 3))
    direct = rng.random((5, 3)) > 0.4
    direct[0] = True
    xyz[~direct] = np.nan
    unary = rng.random((5, 6))
    old = solve_identity_memory_dp(np.arange(5), unary, xyz, direct)
    new = solve_identity_node_memory_dp(
        np.arange(5), unary, xyz[:, :, None], direct[:, :, None], node_indices=(1,)
    )
    expanded = {
        "state_last_event",
        "edge_delta_frames",
        "edge_squared_distance_mm2",
        "terminal_last_event_frame",
        "terminal_last_event_slot",
        "terminal_age_frames",
        "terminal_unsettled_gap",
    }
    for field in fields(old):
        before, after = getattr(old, field.name), getattr(new, field.name)
        if field.name in expanded:
            after = after[..., 0]
        if isinstance(before, np.ndarray):
            assert before.dtype == after.dtype and before.shape == after.shape
            assert before.tobytes() == np.ascontiguousarray(after).tobytes(), field.name
        else:
            assert before == after, field.name


def test_mouth_observation_resolves_chest_only_missing_roundtrip():
    xyz = np.zeros((3, 3, 2, 3))
    xyz[:, :, :, 0] = np.array([0, 40, 80])[None, :, None]
    direct = np.ones((3, 3, 2), bool)
    direct[1, [0, 2], 0] = False
    direct[1, 2, 1] = False
    xyz[~direct] = np.nan
    unary = np.zeros((3, 6))
    old = solve_identity_memory_dp(np.arange(3), unary, xyz[:, :, 0], direct[:, :, 0])
    new = solve_identity_node_memory_dp(np.arange(3), unary, xyz, direct)
    assert old.optimal_state_mask[1, 5]
    assert not new.optimal_state_mask[1, 5]
    np.testing.assert_array_equal(new.path_state, [0, 0, 0])
    assert new.global_alternative_gap > new.numerical_tie_tolerance


def test_terminal_node_gaps_and_one_frame_shapes():
    xyz = np.zeros((3, 3, 2, 3))
    direct = np.ones((3, 3, 2), bool)
    direct[1:, 0, 0] = False
    direct[2, 0, 1] = False
    unary = np.full((3, 6), np.inf)
    unary[:, 0] = 0
    result = solve_identity_node_memory_dp(np.arange(90, 93), unary, xyz, direct)
    np.testing.assert_array_equal(result.terminal_age_frames[0], [2, 1])
    assert result.terminal_unsettled_gap[0].all()
    one = solve_identity_node_memory_dp(np.array([90]), unary[:1], xyz[:1], direct[:1])
    assert one.edge_delta_frames.shape == (0, 3, 2)
    assert one.state_last_event.shape == (1, 3, 2)
    assert one.path_cost == 0 and np.isinf(one.global_alternative_gap)


def test_infeasible_paths_not_lexical_winner():
    xyz, direct = np.zeros((2, 3, 2, 3)), np.ones((2, 3, 2), bool)
    unary = np.zeros((2, 6))
    unary[1] = np.inf
    result = solve_identity_node_memory_dp(np.arange(2), unary, xyz, direct)
    assert result.status == "infeasible" and np.all(result.path_state == -1)
    assert np.isnan(result.min_marginal_delta).all()


@pytest.mark.parametrize(
    "kw", [{"max_layer_states": 5}, {"max_total_states": 6}, {"max_total_edges": 5}]
)
def test_exact_caps_stop_without_pruning(kw):
    with pytest.raises(IdentityMemoryCapacityError):
        solve_identity_node_memory_dp(
            np.arange(2), np.zeros((2, 6)), np.zeros((2, 3, 2, 3)), np.ones((2, 3, 2), bool), **kw
        )


@pytest.mark.parametrize("bad", ["direct_nan", "initial_missing", "node_order", "node_bool"])
def test_invalid_node_evidence_rejected(bad):
    xyz, direct = np.zeros((2, 3, 2, 3)), np.ones((2, 3, 2), bool)
    nodes = (1, 0)
    if bad == "direct_nan":
        xyz[1, 0, 1] = np.nan
    elif bad == "initial_missing":
        direct[0, 0, 1] = False
    elif bad == "node_order":
        nodes = (0, 1)
    else:
        nodes = (True, 0)
    with pytest.raises(ValueError):
        solve_identity_node_memory_dp(
            np.arange(2), np.zeros((2, 6)), xyz, direct, node_indices=nodes
        )

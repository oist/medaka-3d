"""Independent short-path enumeration; no real candidates or long solve."""

import dataclasses
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "labs/20260909_three_body"))
import identity_terminal_conditioned as m  # noqa: E402
from identity_node_memory_dp import solve_identity_node_memory_dp  # noqa: E402

PERMS = np.asarray(list(itertools.permutations(range(3))), dtype=np.int64)


def tiny_case(seed=0, k=2, length=4):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(length, 3, k, 3)) * 0.4
    xyz[..., 0] += np.arange(3)[None, :, None] * 1.2
    direct = rng.random((length, 3, k)) > 0.4
    direct[0] = True
    xyz[~direct] = np.nan
    unary = rng.uniform(-0.4, 0.8, (length, 6))
    return xyz, direct, unary


def completed(xyz, direct, unary):
    nodes = (1,) if direct.shape[2] == 1 else (1, 0)
    return solve_identity_node_memory_dp(
        np.arange(100, 100 + len(xyz), dtype=np.int64), unary, xyz, direct, node_indices=nodes
    )


def snapshot(result):
    return {
        field.name: (value.dtype.str, value.shape, value.tobytes(order="C"))
        if isinstance(value := getattr(result, field.name), np.ndarray)
        else repr(value)
        for field in dataclasses.fields(result)
    }


def enumerate_conditions(result, xyz, direct, unary):
    """Enumerate all6^(T-1) paths and find last observations by backward search.

    No DP edge weights, F/B or recurrence are used to obtain costs. State keys
    are looked up only to label the independent suffix oracle for each node.
    Forbidden prefix costs do not erase structurally valid suffix conditions.
    """
    length, _, k = direct.shape
    n = len(result.state_permutation)
    keys = {}
    for time in range(length):
        for index in range(result.layer_offsets[time], result.layer_offsets[time + 1]):
            keys[
                time,
                int(result.state_permutation[index]),
                tuple(result.state_last_event[index].flat),
            ] = index
    transfer, constrained = np.full(6, np.inf), np.full((length, 6, 6), np.inf)
    backward = np.full((n, 6), np.inf)
    for suffix in itertools.product(range(6), repeat=length - 1):
        path = (0, *suffix)
        increments = [float(unary[0, 0])]
        states = []
        for time in range(length):
            memory = []
            charge = float(unary[time, path[time]])
            for identity in range(3):
                raw = int(PERMS[path[time], identity])
                for node in range(k):
                    last = next(
                        old
                        for old in range(time, -1, -1)
                        if direct[old, PERMS[path[old], identity], node]
                    )
                    memory.append(3 * last + int(PERMS[path[last], identity]))
                    if time == 0 or not direct[time, raw, node]:
                        continue
                    old = next(
                        old
                        for old in range(time - 1, -1, -1)
                        if direct[old, PERMS[path[old], identity], node]
                    )
                    old_raw = int(PERMS[path[old], identity])
                    squared = sum(
                        (float(xyz[time, raw, node, axis]) - float(xyz[old, old_raw, node, axis]))
                        ** 2
                        for axis in range(3)
                    )
                    gap = time - old
                    charge += 0.2 / 3 * (squared / (0.8**2 * gap) + 3 * np.log(gap))
            states.append(keys[time, path[time], tuple(memory)])
            if time:
                increments.append(charge)
        total, terminal = sum(increments), path[-1]
        transfer[terminal] = min(transfer[terminal], total)
        for time, permutation in enumerate(path):
            constrained[time, permutation, terminal] = min(
                constrained[time, permutation, terminal], total
            )
            cost = sum(increments[time + 1 :])
            backward[states[time], terminal] = min(backward[states[time], terminal], cost)
    return transfer, constrained, backward


@pytest.mark.parametrize("seed", [0, 7, 19])
@pytest.mark.parametrize("k", [1, 2])
def test_all_terminal_conditions_against_independent_full_enumeration(seed, k):
    xyz, direct, unary = tiny_case(seed, k)
    unary[2, 4] = np.inf
    unary[-1, 3] = np.inf
    result = completed(xyz, direct, unary)
    before, before_unary = snapshot(result), unary.tobytes()
    actual = m.terminal_conditioned(result, unary=unary)
    expected = enumerate_conditions(result, xyz, direct, unary)
    for value, oracle in zip(
        (actual.transfer, actual.constrained_cost, actual.backward_cost), expected, strict=True
    ):
        np.testing.assert_allclose(value, oracle, atol=1e-12, rtol=1e-13)
    assert np.isposinf(actual.transfer[3])
    assert np.isposinf(actual.constrained_cost[:, :, 3]).all()
    assert not np.isnan(actual.backward_cost).any()
    assert snapshot(result) == before and unary.tobytes() == before_unary
    assert actual.centered_forward_bytes == 8 * len(result.state_permutation)
    assert actual.centered_forward_passes == 1 and actual.terminal_backward_sweeps == 6
    assert actual.free_end_consistent and actual.forward_reproduced_byte_exact
    assert actual.free_end_reproduced_byte_exact


def test_default_unary_zero_missing_memory_and_terminal_unsettled_not_charged():
    xyz = np.zeros((4, 3, 2, 3))
    xyz[..., 0] = np.arange(3)[None, :, None] * 3.0
    direct = np.ones((4, 3, 2), dtype=bool)
    direct[1:3, 0, 0] = False
    direct[2:, 1, 1] = False
    xyz[~direct] = np.nan
    unary = np.zeros((4, 6))
    result = completed(xyz, direct, unary)
    actual = m.terminal_conditioned(result)
    transfer, constrained, backward = enumerate_conditions(result, xyz, direct, unary)
    np.testing.assert_allclose(actual.transfer, transfer, atol=1e-12, rtol=1e-13)
    np.testing.assert_allclose(actual.constrained_cost, constrained, atol=1e-12, rtol=1e-13)
    np.testing.assert_allclose(actual.backward_cost, backward, atol=1e-12, rtol=1e-13)
    terminal = slice(result.layer_offsets[-2], None)
    nodes = np.arange(len(result.state_permutation))[terminal]
    assert np.all(actual.backward_cost[nodes, result.state_permutation[terminal]] == 0)
    assert not actual.unary_supplied
    assert result.terminal_unsettled_gap.any()


@pytest.mark.parametrize("initial_unary", [0.0, 2.75, np.inf])
def test_one_frame_only_initial_P_is_structural_and_initial_cost_once(initial_unary):
    xyz, direct = np.zeros((1, 3, 2, 3)), np.ones((1, 3, 2), dtype=bool)
    unary = np.full((1, 6), 7.0)
    unary[0, 0] = initial_unary
    result = completed(xyz, direct, unary)
    actual = m.terminal_conditioned(result, unary=unary)
    expected = np.full(6, np.inf)
    expected[0] = initial_unary
    np.testing.assert_array_equal(actual.transfer, expected)
    assert actual.backward_cost.shape == (1, 6)
    np.testing.assert_array_equal(
        actual.backward_cost[0], [0, np.inf, np.inf, np.inf, np.inf, np.inf]
    )
    wanted = np.full((1, 6, 6), np.inf)
    wanted[0, 0, 0] = initial_unary
    np.testing.assert_array_equal(actual.constrained_cost, wanted)


def test_all_ties_remain_all36_internal_and_all6_terminals():
    xyz, direct = np.zeros((3, 3, 2, 3)), np.ones((3, 3, 2), dtype=bool)
    result = completed(xyz, direct, np.zeros((3, 6)))
    actual = m.terminal_conditioned(result)
    assert np.all(actual.transfer == 0)
    assert np.all(actual.constrained_cost[1] == 0)
    assert np.all(actual.constrained_cost[0, 0] == 0)
    assert np.isposinf(actual.constrained_cost[0, 1:]).all()
    np.testing.assert_array_equal(np.diag(actual.constrained_cost[-1]), np.zeros(6))
    assert np.isposinf(actual.constrained_cost[-1][~np.eye(6, dtype=bool)]).all()


def test_forbidden_middle_layer_keeps_suffix_costs_but_all_complete_costs_infinite():
    xyz, direct, unary = tiny_case(3, length=3)
    unary[1] = np.inf
    result = completed(xyz, direct, unary)
    actual = m.terminal_conditioned(result, unary=unary)
    assert result.status == "infeasible"
    assert np.isposinf(actual.transfer).all()
    assert np.isposinf(actual.constrained_cost).all()
    assert np.isfinite(actual.backward_cost[result.layer_offsets[1] :]).any()
    np.testing.assert_allclose(
        actual.backward_cost, enumerate_conditions(result, xyz, direct, unary)[2], atol=1e-12
    )


def test_large_common_offsets_reproduce_old_centered_arithmetic_exactly():
    xyz, direct, unary = tiny_case(7)
    unary += 1e14
    result = completed(xyz, direct, unary)
    actual = m.terminal_conditioned(result, unary=unary)
    assert actual.constrained_cost.min(axis=2).tobytes() == result.min_marginal_cost.tobytes()
    assert actual.backward_cost.min(axis=1).tobytes() == result.backward_cost.tobytes()
    assert actual.transfer.min() == result.path_cost
    # At least one stored absolute F+B loses the frozen centering grouping.
    assert np.any(result.forward_cost + result.backward_cost != result.node_min_marginal_cost)


def test_default_unary_cannot_silently_rescore_a_nonzero_unary_result():
    xyz, direct, unary = tiny_case()
    result = completed(xyz, direct, unary)
    with pytest.raises(ValueError, match="reproduce"):
        m.terminal_conditioned(result)
    with pytest.raises(ValueError, match="reproduce"):
        m.terminal_conditioned(result, unary=unary + 1)


@pytest.mark.parametrize("fault", ["shape", "nan", "negative_inf"])
def test_invalid_explicit_unary(fault):
    xyz, direct, unary = tiny_case()
    result = completed(xyz, direct, unary)
    if fault == "shape":
        unary = unary[:, :5]
    else:
        unary[1, 2] = np.nan if fault == "nan" else -np.inf
    with pytest.raises(ValueError):
        m.terminal_conditioned(result, unary=unary)


@pytest.mark.parametrize(
    "fault",
    [
        "frames",
        "dtype",
        "layer",
        "edge_count",
        "source_order",
        "target_layer",
        "P_order",
        "memory_future",
        "forward",
        "backward",
        "marginal",
        "offset",
        "cost",
        "cap",
        "initial",
        "approximation",
    ],
)
def test_malformed_or_inconsistent_completed_graph_rejected(fault):
    xyz, direct, _ = tiny_case()
    result = completed(xyz, direct, np.zeros((4, 6)))
    names = {
        "frames": "frame_idx",
        "dtype": "edge_source",
        "layer": "layer_offsets",
        "edge_count": "edge_offsets",
        "source_order": "edge_source",
        "target_layer": "edge_target",
        "P_order": "edge_target",
        "memory_future": "state_last_event",
        "forward": "forward_cost",
        "backward": "backward_cost",
        "marginal": "min_marginal_cost",
        "cost": "edge_cost",
    }
    if fault in names:
        name = names[fault]
        value = getattr(result, name).copy()
        if fault == "frames":
            value[-1] += 1
        elif fault == "dtype":
            value = value.astype(np.int32)
        elif fault == "layer":
            value[1] += 1
        elif fault in {"edge_count", "source_order"}:
            value[-1] -= 1
        elif fault == "target_layer":
            value[0] = 0
        elif fault == "P_order":
            value[[0, 1]] = value[[1, 0]]
        elif fault == "memory_future":
            value[0, 0, 0] = 3
        elif fault == "cost":
            value[0] = -1
        else:
            value.flat[0] = np.nextafter(value.flat[0], np.inf)
        result = dataclasses.replace(result, **{name: value})
    else:
        key, value = {
            "offset": ("common_cost_offset", result.common_cost_offset + 1),
            "cap": ("max_total_states", 1),
            "initial": ("initial_state", 1),
            "approximation": ("approximation_used", True),
        }[fault]
        result = dataclasses.replace(result, **{key: value})
    with pytest.raises(ValueError):
        m.terminal_conditioned(result)


def test_readonly_strided_inputs_and_owned_return_arrays():
    xyz, direct, unary = tiny_case()
    result = completed(xyz, direct, unary)
    storage = np.empty(2 * len(result.forward_cost))
    storage[::2] = result.forward_cost
    result = dataclasses.replace(result, forward_cost=storage[::2])
    for field in dataclasses.fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    before = snapshot(result)
    actual = m.terminal_conditioned(result, unary=unary)
    actual.backward_cost.fill(0)
    actual.transfer.fill(0)
    actual.constrained_cost.fill(0)
    actual.frame_idx[:] += 1
    assert snapshot(result) == before

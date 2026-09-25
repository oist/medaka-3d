"""Six terminal-permutation conditions on an already completed identity DAG.

No graph is generated, pruned, relabelled or solved with different costs. The
frozen result stores absolute forward costs, not the centered intermediates.
One pass over its existing edges reconstructs the original centered forward
arithmetic and requires byte-exact reproduction of the saved forward array.
This avoids recovering centered values by lossy subtraction from absolute F.

The six backwards sweeps retain the frozen addition order and restore offsets
only at the same output boundary. This is a block-mathematics component, not a
block solver, a new resource budget or a long-sequence qualification.
"""

from dataclasses import dataclass

import numpy as np
from identity_node_memory_dp import (
    PERMUTATIONS,
    IdentityNodeMemoryDPResult,
    _center_cost_blocks,
    _costs,
)

SOURCE_DP_SHA = "e5331f706137cc361154cc065caaa58f02686a53019c1971795adb5702e94dd6"
SCHEMA = "identity_terminal_conditioned_v1"


@dataclass(frozen=True)
class TerminalConditionedResult:
    backward_cost: np.ndarray  # N,terminal_P; excludes current-node unary
    transfer: np.ndarray  # terminal_P; includes original initial unary once
    constrained_cost: np.ndarray  # T,current_P,terminal_P; absolute complete cost
    frame_idx: np.ndarray
    initial_state: int
    node_indices: tuple
    unary_supplied: bool
    centered_forward_bytes: int
    schema: str = SCHEMA
    free_end_consistent: bool = True
    forward_reproduced_byte_exact: bool = True
    free_end_reproduced_byte_exact: bool = True
    graph_reenumerated: bool = False
    approximation_used: bool = False
    caps_reinterpreted: bool = False
    initial_cost_included_once: bool = True
    centered_forward_passes: int = 1
    terminal_backward_sweeps: int = 6


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(result, name, shape, dtype, *, costs=False):
    value = getattr(result, name)
    _require(
        isinstance(value, np.ndarray) and value.shape == shape and value.dtype == np.dtype(dtype),
        "completed DAG array schema: " + name,
    )
    if costs:
        _require(
            not np.isnan(value).any() and not np.isneginf(value).any(),
            "DAG costs allow finite/+inf only: " + name,
        )
    return value


def _same(actual, expected, name):
    _require(
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and actual.tobytes(order="C") == expected.tobytes(order="C"),
        "frozen arithmetic does not reproduce " + name,
    )


def _validate(result):
    _require(isinstance(result, IdentityNodeMemoryDPResult), "completed frozen DAG result required")
    _require(result.status in ("optimal", "infeasible"), "incomplete DAG status")
    _require(
        type(result.initial_state) is int and result.initial_state == 0,
        "fixed initial P012 required",
    )
    _require(
        result.node_indices in ((1,), (1, 0))
        and type(result.node_indices) is tuple
        and all(type(x) is int for x in result.node_indices),
        "original node order required",
    )
    _require(result.approximation_used is False, "approximated DAG is not accepted")
    _require(
        result.event_encoding == "3*frame_offset+raw_slot"
        and result.terminal_missing_endpoint_cost_applied is False,
        "original event and terminal-missing convention required",
    )
    frames = result.frame_idx
    _require(
        isinstance(frames, np.ndarray)
        and frames.ndim == 1
        and frames.dtype == np.int64
        and len(frames) > 0
        and np.all(np.diff(frames) == 1),
        "contiguous native int64 frames required",
    )
    t, k = len(frames), len(result.node_indices)
    layers = _array(result, "layer_offsets", (t + 1,), "int64")
    _require(layers[0] == 0 and layers[1] == 1 and np.all(np.diff(layers) > 0), "layer offsets")
    n = int(layers[-1])
    edges = _array(result, "edge_offsets", (t,), "int64")
    _require(edges[0] == 0 and np.all(np.diff(edges) >= 0), "edge offsets")
    e = int(edges[-1])
    permutation = _array(result, "state_permutation", (n,), "int64")
    _require(np.all((permutation >= 0) & (permutation < 6)) and permutation[0] == 0, "state P")
    _same(_array(result, "permutations", (6, 3), "int64"), PERMUTATIONS, "permutation table")
    memory = _array(result, "state_last_event", (n, 3, k), "int64")
    times = np.repeat(np.arange(t), np.diff(layers))
    _require(
        np.all(memory >= 0) and np.all(memory // 3 <= times[:, None, None]),
        "memory event lies outside its prefix",
    )
    _same(memory[0], np.repeat(np.arange(3), k).reshape(3, k), "initial memory")
    source = _array(result, "edge_source", (e,), "int64")
    target = _array(result, "edge_target", (e,), "int64")
    motion = _array(result, "edge_cost", (e,), "float64", costs=True)
    _require(np.isfinite(motion).all() and np.all(motion >= 0), "finite nonnegative motion edges")
    for name, shape in (
        ("forward_cost", (n,)),
        ("backward_cost", (n,)),
        ("node_min_marginal_cost", (n,)),
        ("min_marginal_cost", (t, 6)),
    ):
        _array(result, name, shape, "float64", costs=True)
    for name, actual in (
        ("max_layer_states", int(np.diff(layers).max())),
        ("max_total_states", n),
        ("max_total_edges", e),
    ):
        limit = getattr(result, name)
        _require(type(limit) is int and limit > 0 and actual <= limit, "original cap: " + name)
    _require(
        np.isfinite(result.common_cost_offset) and np.isfinite(result.numerical_cost_span),
        "finite original centering metadata required",
    )
    _require(
        np.isfinite(result.numerical_tie_tolerance) and result.numerical_tie_tolerance >= 0,
        "original tie tolerance required",
    )
    _require(
        np.isfinite(result.path_cost)
        if result.status == "optimal"
        else np.isposinf(result.path_cost),
        "status and saved path cost differ",
    )
    for time in range(1, t):
        lo, hi = int(edges[time - 1]), int(edges[time])
        old_lo, old_hi = int(layers[time - 1]), int(layers[time])
        now_lo, now_hi = old_hi, int(layers[time + 1])
        _require(hi - lo == 6 * (old_hi - old_lo), "all six outgoing edges required")
        _same(source[lo:hi], np.repeat(np.arange(old_lo, old_hi), 6), "source-major edge order")
        _require(np.all((target[lo:hi] >= now_lo) & (target[lo:hi] < now_hi)), "edge layer target")
        _same(permutation[target[lo:hi]], np.tile(np.arange(6), old_hi - old_lo), "edge P order")
        _require(len(np.unique(target[lo:hi])) == now_hi - now_lo, "unreachable structural node")
    return frames, layers, edges, permutation, source, target, motion, times


def terminal_conditioned(result, *, unary=None):
    """Return six exact terminal conditions without changing the supplied DAG.

    unary=None declares the registered all-zero unary. The original nonzero or
    +inf unary must be supplied explicitly because edge_cost stores only motion;
    unary is never inferred from saved forward costs. A mismatch is rejected by
    exact centering/forward/free-end reproduction, not treated as a new objective.

    At the terminal layer B[s,r]=0 for P(s)=r, +inf otherwise. For earlier nodes,
    B[s,r] excludes the current unary. Transfer contains the fixed-start total
    cost, including the original initial unary once. H includes both endpoints.
    All returned arrays are owned; no input array or scalar is modified.
    The additional centered-forward work is one existing-edge scan plus N
    float64 values (8*N bytes), recorded in the result metadata. Returned
    backward/H storage is 6*N and 36*T float64 values respectively.
    """
    frames, layers, edges, permutation, source, target, motion, times = _validate(result)
    t, n = len(frames), len(permutation)
    supplied = unary is not None
    emission = _costs(np.zeros((t, 6)) if unary is None else unary, (t, 6), "original unary")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        emission, unary_offsets, unary_spans = _center_cost_blocks(emission, (1,))
        edge_minima, edge_spans = np.zeros(t), np.zeros(t)
        forward = np.full(n, np.inf)
        forward[0] = emission[0, 0]
        # Exact frozen recurrence; no state enumeration, source geometry or ID
        # assignment is invoked. Retain only one reconstructed centered F array.
        for time in range(1, t):
            lo, hi = int(edges[time - 1]), int(edges[time])
            now_lo, now_hi = int(layers[time]), int(layers[time + 1])
            edge_minima[time] = motion[lo:hi].min()
            edge_spans[time] = motion[lo:hi].max() - edge_minima[time]
            centered = motion[lo:hi] - edge_minima[time]
            candidates = forward[source[lo:hi]] + centered
            best_prefix = np.full(now_hi - now_lo, np.inf)
            np.minimum.at(best_prefix, target[lo:hi] - now_lo, candidates)
            forward[now_lo:now_hi] = best_prefix + emission[time, permutation[now_lo:now_hi]]
        offsets = np.cumsum(unary_offsets + edge_minima)
        total_offset = float(offsets[-1])
        cost_span = float(unary_spans.sum() + edge_spans.sum())
        _same(np.asarray(total_offset), np.asarray(result.common_cost_offset), "common offset")
        _same(np.asarray(cost_span), np.asarray(result.numerical_cost_span), "cost span")
        _same(forward + offsets[times], result.forward_cost, "saved absolute forward")

        backward = np.full((n, 6), np.inf)
        terminal_nodes = np.arange(int(layers[-2]), n)
        backward[terminal_nodes, permutation[terminal_nodes]] = 0
        for terminal in range(6):
            for time in range(t - 1, 0, -1):
                lo, hi = int(edges[time - 1]), int(edges[time])
                centered = motion[lo:hi] - edge_minima[time]
                # Same left-associative centered_edge + current_unary + B as
                # the frozen free-end reverse pass. Do not regroup these terms.
                cost = (
                    centered
                    + emission[time, permutation[target[lo:hi]]]
                    + backward[target[lo:hi], terminal]
                )
                np.minimum.at(backward[:, terminal], source[lo:hi], cost)
        free_backward = backward.min(axis=1)
        remaining_offsets = total_offset - offsets[times]
        _same(free_backward + remaining_offsets, result.backward_cost, "saved free-end backward")
        _same(
            forward + free_backward + total_offset, result.node_min_marginal_cost, "node marginal"
        )
        constrained = np.full((t, 6, 6), np.inf)
        for terminal in range(6):
            node_cost = forward + backward[:, terminal]
            np.minimum.at(constrained[:, :, terminal], (times, permutation), node_cost)
        constrained += total_offset
        _same(constrained.min(axis=2), result.min_marginal_cost, "saved free-end P marginal")
        transfer = np.full(6, np.inf)
        np.minimum.at(transfer, permutation[terminal_nodes], result.forward_cost[terminal_nodes])
        _same(np.asarray(transfer.min()), np.asarray(result.path_cost), "saved optimal total")
        backward += remaining_offsets[:, None]
    return TerminalConditionedResult(
        backward,
        transfer,
        constrained,
        frames.copy(),
        result.initial_state,
        result.node_indices,
        supplied,
        forward.nbytes,
    )

"""Approximate whole-fish identity search with retained last-direct-event memory.

Every retained state expands all six permutations using the frozen chest/mouth
motion objective. Identical full states merge before deterministic prefix-cost
ranking. The retained nodes stay in lexicographic full-state order, and ALL
edges between retained adjacent layers survive for a conditional backward pass.

Discarded finite prefixes can contain the true optimum. Every reported gap is
conditional on this retained graph, never a certified global gap or identity
probability. No raw geometry, source assignments or review flags are modified.

Resource accounting is explicit and differs from a full-graph exact solve:
total_states bounds STORED retained nodes; total_edges bounds ALL generated
predecessor-times-six edges, including those later discarded. layer_states
bounds the merged expanded layer BEFORE pruning. The fixed ceiling numbers
remain 100k / 1m / 6m; lower limits are allowed for bounded qualification tests.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[2] / "labs/20260909_three_body"
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
import identity_node_memory_dp as frozen  # noqa: E402

SCHEMA = "identity_node_memory_approximate_beam_v1"
FROZEN_SHA = "e5331f706137cc361154cc065caaa58f02686a53019c1971795adb5702e94dd6"
CORE_SHA = "2e848b6c4b14eebcd3d4d3791a4f61aae3e8f4e9fd0413588bd6a1548121e354"
CEILINGS = dict(layer_states=100_000, total_states=1_000_000, total_edges=6_000_000)


class IdentityBeamCapacityError(RuntimeError):
    """The distinct approximate stage stopped without returning a partial result."""

    def __init__(self, frame_offset, resource, attempted, limit):
        self.frame_offset, self.resource = int(frame_offset), resource
        self.attempted, self.limit = int(attempted), int(limit)
        super().__init__(
            f"beam memory search stopped at offset {frame_offset}: "
            f"{resource} {attempted} exceeds cap {limit}; no partial result"
        )


@dataclass(frozen=True)
class IdentityNodeMemoryBeamResult:
    schema: str
    status: str  # optimal_on_retained_graph / infeasible_on_retained_graph
    frame_idx: np.ndarray
    permutations: np.ndarray
    path_state: np.ndarray
    permutation_by_id: np.ndarray
    path_node: np.ndarray
    path_cost: float
    layer_offsets: np.ndarray
    state_permutation: np.ndarray
    state_last_event: np.ndarray  # retained N,ID,K; event=3*frame_offset+raw_slot
    edge_offsets: np.ndarray
    edge_source: np.ndarray
    edge_target: np.ndarray
    edge_cost: np.ndarray  # motion excluding current unary; retained edges only
    forward_cost: np.ndarray
    backward_cost: np.ndarray
    conditional_min_marginal_cost: np.ndarray  # T,6 on retained DAG
    conditional_min_marginal_delta: np.ndarray
    conditional_optimal_state_mask: np.ndarray
    conditional_id_slot_min_marginal_delta: np.ndarray
    conditional_state_alternative_gap: np.ndarray
    conditional_identity_alternative_gap: np.ndarray
    conditional_state_alternative_status: np.ndarray
    conditional_identity_alternative_status: np.ndarray
    conditional_global_alternative_gap: (
        float  # distinct retained whole P path, including swap-return
    )
    conditional_global_alternative_status: str
    terminal_last_event_frame: np.ndarray
    terminal_last_event_slot: np.ndarray
    terminal_age_frames: np.ndarray
    terminal_unsettled_gap: np.ndarray
    expanded_state_counts: np.ndarray  # unique merged candidates, including discarded states
    retained_state_counts: np.ndarray
    discarded_state_counts: np.ndarray
    discarded_finite_state_counts: np.ndarray  # finite prefix at removal, not number of paths
    expanded_edge_counts: np.ndarray  # six per retained predecessor; first row zero
    retained_edge_counts: np.ndarray
    discarded_edge_counts: np.ndarray
    numerical_tie_tolerance: float
    roundoff_clamped_deltas: int
    common_cost_offset: float
    numerical_cost_span: float
    beam_width: int
    max_layer_states: int
    max_total_states: int
    max_total_edges: int
    node_indices: tuple
    scale_mm: float
    weight: float
    initial_state: int
    pruning_occurred: bool
    finite_prefixes_discarded: bool
    approximation_used: bool
    global_optimality_certified: bool = False
    gaps_conditional_on_retained_graph: bool = True
    physical_ID_truth: bool = False
    review_flags_cleared: bool = False
    intermediate_ID_reset: bool = False
    terminal_missing_endpoint_cost_applied: bool = False
    total_states_scope: str = "stored retained nodes"
    total_edges_scope: str = "all generated edges including discarded"
    layer_states_scope: str = "merged expanded layer before pruning"
    ranking_scope: str = "centered prefix plus current unary; full-state lexicographic ties"


def verify_sources():
    for path, expected in ((Path(frozen.__file__), FROZEN_SHA), (LAB / "identity_dp.py", CORE_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("frozen motion source changed: " + str(path))


def limit(t, name, value, maximum):
    if value > maximum:
        raise IdentityBeamCapacityError(t, name, value, maximum)


def capacity(value, name):
    value = frozen._cap_value(value, name)
    if value > CEILINGS[name]:
        raise ValueError("registered capacity ceiling cannot be increased: " + name)
    return value


def inputs(frame_idx, unary, node_xyz, direct_nodes, node_indices, initial_state):
    frames = frozen._frames(frame_idx)
    if (
        type(node_indices) is not tuple
        or node_indices not in ((1,), (1, 0))
        or any(type(i) is not int for i in node_indices)
    ):
        raise ValueError("chest then mouth node order must be exactly (1,) or (1,0)")
    if (
        isinstance(initial_state, (bool, np.bool_))
        or not isinstance(initial_state, (int, np.integer))
        or initial_state != 0
    ):
        raise ValueError("one fixed initial_state=0 required")
    n, k = len(frames), len(node_indices)
    xyz, direct = np.asarray(node_xyz), np.asarray(direct_nodes)
    if xyz.shape != (n, 3, k, 3) or xyz.dtype.kind not in "fiu":
        raise ValueError("real T,raw_slot,K,3 node coordinates required")
    if direct.shape != (n, 3, k) or direct.dtype != bool or not direct[0].all():
        raise ValueError("boolean T,raw_slot,K support with all first-frame nodes direct required")
    xyz = xyz.astype(np.float64, copy=True)
    if not np.isfinite(xyz[direct]).all():
        raise ValueError("directly observed nodes must have finite coordinates")
    return frames, frozen._costs(unary, (n, 6), "unary"), xyz, direct.copy()


def solve_identity_node_memory_beam(
    frame_idx,
    unary,
    node_xyz,
    direct_nodes,
    *,
    beam_width=64,
    node_indices=(1, 0),
    scale_mm=0.8,
    weight=0.2,
    initial_state=0,
    max_layer_states=100_000,
    max_total_states=1_000_000,
    max_total_edges=6_000_000,
    tie_atol=1e-10,
    tie_rtol=1e-12,
):
    """Search fixed raw bodies and report conditional alternatives without clearing review.

    The width is an approximation parameter, not a full-state resource limit.
    Nonfinite prefixes participate in deterministic ranking after finite ones;
    removing them is still reported as structural pruning. An infeasible
    retained graph says nothing about feasibility of discarded alternatives.
    """
    verify_sources()
    frames, emission, xyz, direct = inputs(
        frame_idx, unary, node_xyz, direct_nodes, node_indices, initial_state
    )
    width = frozen._cap_value(beam_width, "beam_width")
    if width > CEILINGS["layer_states"]:
        raise ValueError("beam width exceeds registered layer ceiling")
    layer_cap = capacity(max_layer_states, "layer_states")
    state_cap = capacity(max_total_states, "total_states")
    edge_cap = capacity(max_total_edges, "total_edges")
    scale, factor = (
        frozen._scalar(scale_mm, "scale_mm", positive=True),
        frozen._scalar(weight, "weight"),
    )
    atol, rtol = frozen._scalar(tie_atol, "tie_atol"), frozen._scalar(tie_rtol, "tie_rtol")
    n, k = len(frames), len(node_indices)
    with np.errstate(over="raise", invalid="raise"):
        emission, unary_offsets, unary_spans = frozen._center_cost_blocks(emission, (1,))

    layers = [np.r_[0, np.repeat(np.arange(3), k)][None].astype(np.int64)]
    layer_offsets, edge_offsets = [0, 1], [0]
    source_blocks, target_blocks, motion_blocks, centered_blocks = [], [], [], []
    forward_layers, parent_layers = [np.array([emission[0, 0]])], [np.array([-1], np.int64)]
    minima, spans = np.zeros(n), np.zeros(n)
    expanded_states, retained_states, discarded_finite = (np.zeros(n, np.int64) for _ in range(3))
    expanded_edges, retained_edges = (np.zeros(n, np.int64) for _ in range(2))
    expanded_states[0] = retained_states[0] = 1
    generated_edges = 0
    with np.errstate(over="raise", invalid="raise"):
        for t in range(1, n):
            expanded_edges[t] = 6 * len(layers[-1])
            generated_edges += int(expanded_edges[t])
            limit(t, "total_edges", generated_edges, edge_cap)
            current, source, target, motion, _, _ = frozen._next_layer(
                layers[-1], t, direct, xyz, scale, factor
            )
            expanded_states[t] = len(current)
            limit(t, "layer_states", len(current), layer_cap)
            minima[t] = motion.min()
            centered = motion - minima[t]
            spans[t] = centered.max()
            candidates = forward_layers[-1][source] + centered
            best_prefix = np.full(len(current), np.inf)
            np.minimum.at(best_prefix, target, candidates)
            parent_edge = np.full(len(current), len(candidates), np.int64)
            is_best = candidates == best_prefix[target]
            np.minimum.at(parent_edge, target[is_best], np.flatnonzero(is_best))
            complete_prefix = best_prefix + emission[t, current[:, 0]]
            # Stable sorting on the already lexicographic merged states defines all ties.
            keep = np.sort(np.argsort(complete_prefix, kind="stable")[:width])
            retained_states[t] = len(keep)
            limit(t, "total_states", layer_offsets[-1] + len(keep), state_cap)
            selected = np.zeros(len(current), bool)
            selected[keep] = True
            discarded_finite[t] = np.count_nonzero(~selected & np.isfinite(complete_prefix))
            remap = np.full(len(current), -1, np.int64)
            remap[keep] = np.arange(len(keep))
            edge_keep = selected[target]
            # Every edge to a retained target survives, including a nonwinning prefix.
            source_blocks.append(source[edge_keep] + layer_offsets[-2])
            target_blocks.append(remap[target[edge_keep]] + layer_offsets[-1])
            motion_blocks.append(motion[edge_keep])
            centered_blocks.append(centered[edge_keep])
            retained_edges[t] = np.count_nonzero(edge_keep)
            parent_layers.append(source[parent_edge[keep]] + layer_offsets[-2])
            forward_layers.append(complete_prefix[keep])
            layers.append(current[keep])
            layer_offsets.append(layer_offsets[-1] + len(keep))
            edge_offsets.append(edge_offsets[-1] + int(retained_edges[t]))
        offsets = np.cumsum(unary_offsets + minima)
        total_offset = float(offsets[-1])
        span = float(unary_spans.sum() + spans.sum())
        tolerance = float(atol + rtol * span)

    layer_offsets, edge_offsets = np.array(layer_offsets), np.array(edge_offsets)
    nodes = np.concatenate(layers)
    times = np.repeat(np.arange(n), np.diff(layer_offsets))
    forward, parents = np.concatenate(forward_layers), np.concatenate(parent_layers)
    backward = np.full(len(nodes), np.inf)
    backward[layer_offsets[-2] :] = 0
    with np.errstate(over="raise", invalid="raise"):
        for t in range(n - 1, 0, -1):
            source, target = source_blocks[t - 1], target_blocks[t - 1]
            costs = centered_blocks[t - 1] + emission[t, nodes[target, 0]] + backward[target]
            np.minimum.at(backward, source, costs)
        marginal = np.full((n, 6), np.inf)
        np.minimum.at(marginal, (times, nodes[:, 0]), forward + backward)
        best = float(forward[layer_offsets[-2] :].min())
        absolute_forward = forward + offsets[times]
        absolute_backward = backward + (total_offset - offsets[times])
        absolute_marginal = marginal + total_offset
        path_cost = float(np.float64(best) + total_offset)

    path, path_nodes = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    mapping = np.full((n, 3), -1, np.int64)
    terminal_frame, terminal_slot, terminal_age = (np.full((3, k), -1, np.int64) for _ in range(3))
    unsettled, clamped = np.zeros((3, k), bool), 0
    if not np.isfinite(best):
        status = "infeasible_on_retained_graph"
        deltas, id_deltas = np.full((n, 6), np.nan), np.full((n, 3, 3), np.nan)
        optimum = np.zeros((n, 6), bool)
        state_gap, id_gap = np.full(n, np.nan), np.full((n, 3), np.nan)
        state_status, id_status = (
            np.full(n, "infeasible", "U32"),
            np.full((n, 3), "infeasible", "U32"),
        )
        global_gap, global_status = np.nan, "infeasible_on_retained_graph"
    else:
        status = "optimal_on_retained_graph"
        path_nodes[-1] = layer_offsets[-2] + np.argmin(forward[layer_offsets[-2] :])
        for t in range(n - 1, 0, -1):
            path_nodes[t - 1] = parents[path_nodes[t]]
        path[:] = nodes[path_nodes, 0]
        mapping[:] = frozen.PERMUTATIONS[path]
        deltas = np.full((n, 6), np.inf)
        finite = np.isfinite(marginal)
        deltas[finite] = marginal[finite] - best
        if np.any(deltas < -tolerance):
            raise FloatingPointError("retained min-marginal below best beyond tolerance")
        negative = deltas < 0
        clamped = int(negative.sum())
        deltas[negative] = 0.0
        if np.any(deltas[np.arange(n), path] > tolerance):
            raise FloatingPointError("retained path disagrees with forward/backward costs")
        optimum = deltas <= tolerance
        alternative = deltas.copy()
        alternative[np.arange(n), path] = np.inf
        state_gap = alternative.min(axis=1)
        id_deltas = np.empty((n, 3, 3))
        for identity, slot in np.ndindex(3, 3):
            id_deltas[:, identity, slot] = deltas[:, frozen.PERMUTATIONS[:, identity] == slot].min(
                axis=1
            )
        id_alt = id_deltas.copy()
        id_alt[np.arange(n)[:, None], np.arange(3)[None], mapping] = np.inf
        id_gap = id_alt.min(axis=2)
        state_status, id_status = (
            frozen._alternative_status(g, tolerance) for g in (state_gap, id_gap)
        )
        state_status[0], id_status[0] = "fixed_initial_constraint", "fixed_initial_constraint"
        global_gap = float(state_gap.min())
        global_status = str(frozen._alternative_status(np.array(global_gap), tolerance))
        terminal_event = nodes[path_nodes[-1], 1:].reshape(3, k)
        terminal_frame, terminal_slot = frames[terminal_event // 3], terminal_event % 3
        terminal_age = frames[-1] - terminal_frame
        unsettled = terminal_age > 0

    def cat(blocks, dtype):
        return np.concatenate(blocks) if blocks else np.empty(0, dtype=dtype)

    discarded_states = expanded_states - retained_states
    pruned = bool(discarded_states.any())
    if pruned:
        # An absent retained alternative is unknown globally after discarding states.
        if np.isfinite(best):
            state_status[np.isposinf(state_gap)] = "not_retained_unknown"
            id_status[np.isposinf(id_gap)] = "not_retained_unknown"
            state_status[0], id_status[0] = "fixed_initial_constraint", "fixed_initial_constraint"
            if np.isposinf(global_gap):
                global_status = "not_retained_unknown"
        else:
            state_status[:] = "retained_infeasible_unknown"
            id_status[:] = "retained_infeasible_unknown"
    result = IdentityNodeMemoryBeamResult(
        schema=SCHEMA,
        status=status,
        frame_idx=frames,
        permutations=frozen.PERMUTATIONS.copy(),
        path_state=path,
        permutation_by_id=mapping,
        path_node=path_nodes,
        path_cost=path_cost,
        layer_offsets=layer_offsets,
        state_permutation=nodes[:, 0].copy(),
        state_last_event=nodes[:, 1:].reshape(-1, 3, k).copy(),
        edge_offsets=edge_offsets,
        edge_source=cat(source_blocks, np.int64),
        edge_target=cat(target_blocks, np.int64),
        edge_cost=cat(motion_blocks, np.float64),
        forward_cost=absolute_forward,
        backward_cost=absolute_backward,
        conditional_min_marginal_cost=absolute_marginal,
        conditional_min_marginal_delta=deltas,
        conditional_optimal_state_mask=optimum,
        conditional_id_slot_min_marginal_delta=id_deltas,
        conditional_state_alternative_gap=state_gap,
        conditional_identity_alternative_gap=id_gap,
        conditional_state_alternative_status=state_status,
        conditional_identity_alternative_status=id_status,
        conditional_global_alternative_gap=global_gap,
        conditional_global_alternative_status=global_status,
        terminal_last_event_frame=terminal_frame,
        terminal_last_event_slot=terminal_slot,
        terminal_age_frames=terminal_age,
        terminal_unsettled_gap=unsettled,
        expanded_state_counts=expanded_states,
        retained_state_counts=retained_states,
        discarded_state_counts=discarded_states,
        discarded_finite_state_counts=discarded_finite,
        expanded_edge_counts=expanded_edges,
        retained_edge_counts=retained_edges,
        discarded_edge_counts=expanded_edges - retained_edges,
        numerical_tie_tolerance=tolerance,
        roundoff_clamped_deltas=clamped,
        common_cost_offset=total_offset,
        numerical_cost_span=span,
        beam_width=width,
        max_layer_states=layer_cap,
        max_total_states=state_cap,
        max_total_edges=edge_cap,
        node_indices=node_indices,
        scale_mm=scale,
        weight=factor,
        initial_state=0,
        pruning_occurred=pruned,
        finite_prefixes_discarded=bool(discarded_finite.any()),
        approximation_used=pruned,
    )
    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    verify_sources()
    return result

"""Exact identity DP with separate last-direct memory for chest and mouth.

One shared permutation maps whole bodies; each selected node retains its own
direct event. Adding mouth leaves the old chest cost unchanged (denominator 3).

P[id] is a raw geometric slot. Missing assignments retain their previous event;
on reobservation the displacement is charged over the entire native-frame gap.
This is an experimental Brownian-shaped engineering cost, not a validated
observation model or calibrated identity probability. An unclosed terminal gap
has no future endpoint to charge, and is explicitly reported as unsettled.

All reachable full states and edges are retained, including states forbidden by
an infinite unary. Resource caps stop execution; no approximate pruning occurs.
Geometry, support and existing reviews are never changed by this module.
"""

from dataclasses import dataclass

import numpy as np
from identity_dp import (
    PERMUTATIONS,
    _alternative_status,
    _center_cost_blocks,
    _costs,
    _frames,
    _scalar,
)


class IdentityMemoryCapacityError(RuntimeError):
    """Exact graph exceeds a registered resource cap; no solution is returned."""

    def __init__(self, frame_offset, resource, attempted, limit):
        self.frame_offset = int(frame_offset)
        self.resource = resource
        self.attempted = int(attempted)
        self.limit = int(limit)
        super().__init__(
            f"exact memory DP stopped at offset {frame_offset}: "
            f"{resource} {attempted} exceeds cap {limit}; no pruning/partial solution"
        )


@dataclass(frozen=True)
class IdentityNodeMemoryDPResult:
    status: str
    frame_idx: np.ndarray
    permutations: np.ndarray
    path_state: np.ndarray  # T permutation indices; -1 if infeasible
    permutation_by_id: np.ndarray  # T,ID -> raw slot
    path_node: np.ndarray  # T global layered-node indices; -1 if infeasible
    path_cost: float
    layer_offsets: np.ndarray  # T+1; global node range at each native frame
    state_permutation: np.ndarray  # N
    state_last_event: np.ndarray  # N,ID,K; event=3*frame_offset+raw_slot
    edge_offsets: np.ndarray  # T; block t-1 -> t is [offset[t-1],offset[t])
    edge_source: np.ndarray  # E global nodes; every reachable predecessor/P edge
    edge_target: np.ndarray  # E global nodes; multiple prefixes can merge here
    edge_cost: np.ndarray  # E motion only, excluding current unary
    edge_delta_frames: np.ndarray  # E,ID,K; zero means not directly observed now
    edge_squared_distance_mm2: np.ndarray  # E,ID,K; NaN for no current direct event
    forward_cost: np.ndarray  # N; includes current unary and initial constraint
    backward_cost: np.ndarray  # N; excludes current unary
    node_min_marginal_cost: np.ndarray  # N; all prefix/continuation combinations
    min_marginal_cost: np.ndarray  # T,6; minimum over matching memory states
    min_marginal_delta: np.ndarray  # T,6; NaN for an infeasible whole problem
    optimal_state_mask: np.ndarray  # T,6; within explicit numerical tolerance
    id_slot_min_marginal_delta: np.ndarray  # T,ID,slot
    state_alternative_gap: np.ndarray  # T; any different P at this native frame
    identity_alternative_gap: np.ndarray  # T,ID
    state_alternative_status: np.ndarray
    identity_alternative_status: np.ndarray
    global_alternative_gap: float  # different whole P sequence, even if it returns
    global_alternative_status: str
    terminal_last_event_frame: np.ndarray  # ID,K; native source frame, -1 if infeasible
    terminal_last_event_slot: np.ndarray  # ID,K; raw slot at that last event
    terminal_age_frames: np.ndarray  # ID,K; last native frame minus last event frame
    terminal_unsettled_gap: np.ndarray  # ID,K; age>0, not a settled zero-motion interval
    numerical_tie_tolerance: float
    roundoff_clamped_deltas: int
    common_cost_offset: float
    numerical_cost_span: float
    initial_state: int
    scale_mm: float
    weight: float
    max_layer_states: int
    max_total_states: int
    max_total_edges: int
    node_indices: tuple
    approximation_used: bool = False
    event_encoding: str = "3*frame_offset+raw_slot"
    observation_occurrence_model_used: bool = False
    terminal_missing_endpoint_cost_applied: bool = False


def _cap_value(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _check_cap(t, resource, actual, limit):
    if actual > limit:
        raise IdentityMemoryCapacityError(t, resource, actual, limit)


def _next_layer(previous, t, observed, xyz, scale, weight):
    """All six successors per previous node, then equivalent full-state merging."""
    count = len(previous)
    k = observed.shape[2]
    old_events = np.repeat(previous[:, 1:].reshape(-1, 3, k), 6, axis=0)
    permutation = np.tile(np.arange(6), count)
    slots = PERMUTATIONS[permutation, :, None]
    nodes = np.arange(k)[None, None, :]
    direct = observed[t, slots, nodes]
    keys = np.column_stack(
        (permutation, np.where(direct, 3 * t + slots, old_events).reshape(-1, 3 * k))
    )
    current, target = np.unique(keys, axis=0, return_inverse=True)
    delta = t - old_events // 3
    # Every remembered event was directly observed. Missing current coordinates
    # are replaced only for safe arithmetic and their entire contribution is zero.
    old_xyz = xyz[old_events // 3, old_events % 3, nodes]
    now_xyz = np.where(direct[..., None], xyz[t, slots, nodes], 0.0)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        distance = np.square(now_xyz - old_xyz).sum(axis=3)
        per_id = distance / (np.float64(scale) ** 2 * delta) + 3 * np.log(delta)
        motion = weight / 3 * np.where(direct, per_id, 0.0).sum(axis=(1, 2))
    if not np.isfinite(motion).all():
        raise FloatingPointError("nonfinite remembered-event motion cost")
    return (
        current,
        np.repeat(np.arange(count, dtype=np.int64), 6),
        target.astype(np.int64),
        motion,
        np.where(direct, delta, 0),
        np.where(direct, distance, np.nan),
    )


def solve_identity_node_memory_dp(
    frame_idx,
    unary,
    node_xyz,
    direct_nodes,
    *,
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
    """Minimize unary plus last-direct-event motion on the complete layered DAG.

    For each newly direct ID/node pair, add
      weight/3 * (||x_new-x_last||^2/(scale_mm^2*delta_frames) + 3*log(delta_frames)).
    Missing ID/node pairs retain their event and accrue no provisional endpoint charge.
    Initial P=0 and all selected first-frame nodes are mandatory.
    node_indices is (1,) for old-chest equivalence tests or (1,0) for chest+mouth.
    Caller supplies node_xyz/direct_nodes already sliced in this exact axis order.
    Adding mouth does not renormalize or downweight the original chest term.
    Reaching a fully direct frame merges histories with identical P/event into
    six equivalent states without resetting the accumulated cost or identity.

    Merging takes the best prefix for optimization, but retains every edge for
    backward costs and min-marginals. A P sequence deterministically specifies
    all its memories; different full paths therefore differ in P at some time.
    The minimum alternative P min-marginal equals the second distinct path gap.
    Shared per-time unary and edge minima are removed before arithmetic, and
    restored only to absolute outputs. Gap tolerance depends on finite cost
    spans, not an arbitrary added constant. Floating-point ties are not proof of
    identity; large gaps remain conditional on this unvalidated motion model.
    """
    frames = _frames(frame_idx)
    n = len(frames)
    if (
        isinstance(initial_state, (bool, np.bool_))
        or not isinstance(initial_state, (int, np.integer))
        or initial_state != 0
    ):
        raise ValueError("this registered memory DP requires fixed initial_state=0")
    scale, factor = _scalar(scale_mm, "scale_mm", positive=True), _scalar(weight, "weight")
    atol, rtol = _scalar(tie_atol, "tie_atol"), _scalar(tie_rtol, "tie_rtol")
    max_layer_states = _cap_value(max_layer_states, "max_layer_states")
    max_total_states = _cap_value(max_total_states, "max_total_states")
    max_total_edges = _cap_value(max_total_edges, "max_total_edges")
    if (
        type(node_indices) is not tuple
        or node_indices not in ((1,), (1, 0))
        or any(type(i) is not int for i in node_indices)
    ):
        raise ValueError("node_indices must be exactly (1,) or (1,0), chest then mouth")
    k = len(node_indices)
    xyz, observed = np.asarray(node_xyz), np.asarray(direct_nodes)
    if xyz.shape != (n, 3, k, 3) or xyz.dtype.kind not in "fiu":
        raise ValueError("node_xyz must be real T,raw_slot,K,3 coordinates")
    if observed.shape != (n, 3, k) or observed.dtype != bool or not observed[0].all():
        raise ValueError(
            "direct_nodes must be boolean T,3,K with all selected nodes initially direct"
        )
    xyz = xyz.astype(np.float64, copy=True)
    if not np.isfinite(xyz[observed]).all():
        raise ValueError("every directly observed selected-node coordinate must be finite")
    emission = _costs(unary, (n, 6), "unary")
    with np.errstate(over="raise", invalid="raise"):
        emission, unary_offsets, unary_spans = _center_cost_blocks(emission, (1,))

    layers = [np.r_[0, np.repeat(np.arange(3), k)][None].astype(np.int64)]
    layer_offsets, edge_offsets = [0, 1], [0]
    source_blocks, target_blocks, motion_blocks = [], [], []
    delta_blocks, distance_blocks, centered_edge_blocks = [], [], []
    forward_layers = [np.array([emission[0, 0]])]
    parent_layers = [np.array([-1], np.int64)]
    edge_minima, edge_spans = np.zeros(n), np.zeros(n)
    with np.errstate(over="raise", invalid="raise"):
        for t in range(1, n):
            attempted_edges = edge_offsets[-1] + 6 * len(layers[-1])
            _check_cap(t, "total_edges", attempted_edges, max_total_edges)
            current, source, target, motion, delta, distance = _next_layer(
                layers[-1], t, observed, xyz, scale, factor
            )
            _check_cap(t, "layer_states", len(current), max_layer_states)
            _check_cap(t, "total_states", layer_offsets[-1] + len(current), max_total_states)
            if observed[t].all() and len(current) != 6:
                raise AssertionError("fully direct histories must merge into six permutations")
            edge_minima[t] = motion.min()
            centered = motion - edge_minima[t]
            edge_spans[t] = centered.max()
            candidates = forward_layers[-1][source] + centered
            best_prefix = np.full(len(current), np.inf)
            np.minimum.at(best_prefix, target, candidates)
            # Stable edge order is predecessor-node, then current permutation.
            parents = np.full(len(current), len(candidates), np.int64)
            is_best = candidates == best_prefix[target]
            np.minimum.at(parents, target[is_best], np.flatnonzero(is_best))
            parents = source[parents] + layer_offsets[-2]
            forward_layers.append(best_prefix + emission[t, current[:, 0]])
            parent_layers.append(parents)
            source_blocks.append(source + layer_offsets[-2])
            target_blocks.append(target + layer_offsets[-1])
            motion_blocks.append(motion)
            centered_edge_blocks.append(centered)
            delta_blocks.append(delta)
            distance_blocks.append(distance)
            layers.append(current)
            layer_offsets.append(layer_offsets[-1] + len(current))
            edge_offsets.append(attempted_edges)

        offsets = np.cumsum(unary_offsets + edge_minima)
        total_offset = float(offsets[-1])
        cost_span = float(unary_spans.sum() + edge_spans.sum())
        tolerance = float(atol + rtol * cost_span)

    layer_offsets, edge_offsets = np.array(layer_offsets), np.array(edge_offsets)
    nodes = np.concatenate(layers)
    frame_offsets = np.repeat(np.arange(n), np.diff(layer_offsets))
    forward, parents = np.concatenate(forward_layers), np.concatenate(parent_layers)
    backward = np.full(len(nodes), np.inf)
    backward[layer_offsets[-2] :] = 0
    with np.errstate(over="raise", invalid="raise"):
        for t in range(n - 1, 0, -1):
            source, target = source_blocks[t - 1], target_blocks[t - 1]
            cost = centered_edge_blocks[t - 1] + emission[t, nodes[target, 0]] + backward[target]
            np.minimum.at(backward, source, cost)
        node_marginal = forward + backward
        marginal = np.full((n, 6), np.inf)
        np.minimum.at(marginal, (frame_offsets, nodes[:, 0]), node_marginal)
        best = float(forward[layer_offsets[-2] :].min())
        absolute_forward = forward + offsets[frame_offsets]
        absolute_backward = backward + (total_offset - offsets[frame_offsets])
        absolute_node_marginal = node_marginal + total_offset
        absolute_marginal = marginal + total_offset
        path_cost = float(np.float64(best) + total_offset)

    path, path_nodes = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    mapping = np.full((n, 3), -1, np.int64)
    terminal_frame, terminal_slot, terminal_age = (np.full((3, k), -1, np.int64) for _ in range(3))
    unsettled = np.zeros((3, k), bool)
    clamped = 0
    if not np.isfinite(best):
        status = "infeasible"
        deltas = np.full((n, 6), np.nan)
        optimum = np.zeros((n, 6), bool)
        id_deltas = np.full((n, 3, 3), np.nan)
        state_gap, id_gap = np.full(n, np.nan), np.full((n, 3), np.nan)
        state_status = np.full(n, "infeasible", "U32")
        id_status = np.full((n, 3), "infeasible", "U32")
        global_gap, global_status = np.nan, "infeasible"
    else:
        status = "optimal"
        path_nodes[-1] = layer_offsets[-2] + np.argmin(forward[layer_offsets[-2] :])
        for t in range(n - 1, 0, -1):
            path_nodes[t - 1] = parents[path_nodes[t]]
        path[:] = nodes[path_nodes, 0]
        mapping[:] = PERMUTATIONS[path]
        deltas = np.full((n, 6), np.inf)
        finite = np.isfinite(marginal)
        deltas[finite] = marginal[finite] - best
        if np.any(deltas < -tolerance):
            raise FloatingPointError("min-marginal lies below optimal cost beyond tolerance")
        negative = deltas < 0
        clamped = int(negative.sum())
        deltas[negative] = 0.0
        if np.any(deltas[np.arange(n), path] > tolerance):
            raise FloatingPointError("chosen path disagrees with forward/backward costs")
        optimum = deltas <= tolerance
        alternative = deltas.copy()
        alternative[np.arange(n), path] = np.inf
        state_gap = alternative.min(axis=1)
        id_deltas = np.empty((n, 3, 3))
        for identity, slot in np.ndindex(3, 3):
            id_deltas[:, identity, slot] = deltas[:, PERMUTATIONS[:, identity] == slot].min(axis=1)
        id_alt = id_deltas.copy()
        id_alt[np.arange(n)[:, None], np.arange(3)[None], mapping] = np.inf
        id_gap = id_alt.min(axis=2)
        state_status = _alternative_status(state_gap, tolerance)
        id_status = _alternative_status(id_gap, tolerance)
        state_status[0], id_status[0] = "fixed_initial_constraint", "fixed_initial_constraint"
        global_gap = float(state_gap.min())
        global_status = str(_alternative_status(np.array(global_gap), tolerance))
        terminal_event = nodes[path_nodes[-1], 1:].reshape(3, k)
        terminal_frame = frames[terminal_event // 3]
        terminal_slot = terminal_event % 3
        terminal_age = frames[-1] - terminal_frame
        unsettled = terminal_age > 0

    def concatenate(blocks, shape, dtype):
        return np.concatenate(blocks) if blocks else np.empty(shape, dtype=dtype)

    return IdentityNodeMemoryDPResult(
        status,
        frames,
        PERMUTATIONS.copy(),
        path,
        mapping,
        path_nodes,
        path_cost,
        layer_offsets,
        nodes[:, 0].copy(),
        nodes[:, 1:].reshape(-1, 3, k).copy(),
        edge_offsets,
        concatenate(source_blocks, (0,), np.int64),
        concatenate(target_blocks, (0,), np.int64),
        concatenate(motion_blocks, (0,), np.float64),
        concatenate(delta_blocks, (0, 3, k), np.int64),
        concatenate(distance_blocks, (0, 3, k), np.float64),
        absolute_forward,
        absolute_backward,
        absolute_node_marginal,
        absolute_marginal,
        deltas,
        optimum,
        id_deltas,
        state_gap,
        id_gap,
        state_status,
        id_status,
        global_gap,
        global_status,
        terminal_frame,
        terminal_slot,
        terminal_age,
        unsettled,
        tolerance,
        clamped,
        total_offset,
        cost_span,
        0,
        scale,
        factor,
        max_layer_states,
        max_total_states,
        max_total_edges,
        node_indices,
    )


def node_memory_path_links(
    frame_idx,
    node_xyz,
    direct_nodes,
    mapping,
    *,
    node_indices=(1, 0),
    scale_mm=0.8,
    weight=0.2,
):
    """Independent chronological selected-path ledger; no DP graph is accessed.

    event_frames/event_slots identify the previous direct event used for the
    current charge (including missing frames, whose cost and delta stay zero).
    Initial events are the initial observations, charged zero. Terminal arrays
    describe memory AFTER the final frame. Missing coordinates never contribute.
    """
    frames = _frames(frame_idx)
    if (
        type(node_indices) is not tuple
        or node_indices not in ((1,), (1, 0))
        or any(type(i) is not int for i in node_indices)
    ):
        raise ValueError("node_indices must be exactly (1,) or (1,0)")
    n, k = len(frames), len(node_indices)
    xyz, direct, selected = np.asarray(node_xyz), np.asarray(direct_nodes), np.asarray(mapping)
    if xyz.shape != (n, 3, k, 3) or xyz.dtype.kind not in "fiu":
        raise ValueError("node_xyz must be real T,3,K,3")
    if direct.shape != (n, 3, k) or direct.dtype != bool or not direct[0].all():
        raise ValueError("all selected nodes must be initially direct")
    if not np.isfinite(xyz[direct]).all():
        raise ValueError("direct coordinates must be finite")
    if (
        selected.shape != (n, 3)
        or selected.dtype.kind not in "iu"
        or not np.array_equal(np.sort(selected, axis=1), np.broadcast_to(np.arange(3), (n, 3)))
        or not np.array_equal(selected[0], [0, 1, 2])
    ):
        raise ValueError("mapping must be T permutations with fixed initial P012")
    scale = _scalar(scale_mm, "scale_mm", positive=True)
    factor = _scalar(weight, "weight")
    xyz = xyz.astype(np.float64, copy=False)
    previous_time = np.zeros((3, k), np.int64)
    previous_slot = np.broadcast_to(np.arange(3)[:, None], (3, k)).copy()
    event_frames, event_slots, delta = (np.zeros((n, 3, k), np.int64) for _ in range(3))
    distance = np.full((n, 3, k), np.nan)
    costs = np.zeros((n, 3, k), np.float64)
    for t in range(n):
        event_frames[t] = frames[previous_time]
        event_slots[t] = previous_slot
        if t == 0:
            continue
        for identity, node in np.ndindex(3, k):
            raw = int(selected[t, identity])
            if not direct[t, raw, node]:
                continue
            old_t, old_raw = int(previous_time[identity, node]), int(previous_slot[identity, node])
            elapsed = int(frames[t] - frames[old_t])
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                displacement = xyz[t, raw, node] - xyz[old_t, old_raw, node]
                squared = float(np.square(displacement).sum())
                cost = factor / 3 * (squared / (scale**2 * elapsed) + 3 * np.log(elapsed))
            delta[t, identity, node] = elapsed
            distance[t, identity, node] = squared
            costs[t, identity, node] = cost
            previous_time[identity, node], previous_slot[identity, node] = t, raw
    if not np.isfinite(costs).all():
        raise FloatingPointError("nonfinite direct-event ledger cost")
    terminal_frames = frames[previous_time]
    ages = frames[-1] - terminal_frames
    return dict(
        cost_by_event=costs,
        event_frames=event_frames,
        event_slots=event_slots,
        delta_frames=delta,
        distance_squared=distance,
        motion_cost=float(costs.sum()),
        terminal_last_event_frame=terminal_frames,
        terminal_last_event_slot=previous_slot,
        terminal_age_frames=ages,
        terminal_unsettled_gap=ages > 0,
    )

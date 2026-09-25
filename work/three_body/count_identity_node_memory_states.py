"""Support-only exact reachable states for one or two shared-permutation nodes.

No positions, unary costs, pruning or biological identity evidence. The event
tuple is ID-major: (ID0node0, ID0node1, ID1node0, ...). Every node uses the same
current identity->raw-slot permutation. Event=3*relative_frame+raw_slot; the
tuple axis, not the integer event, identifies which anatomical node it stores.
"""

from __future__ import annotations

import numpy as np
from count_identity_memory_states import PERMUTATIONS, Caps, ReachableCount


def reachable_counts(frame_idx, direct_nodes, caps=None, *, retain_layers=False):
    """Count all K=1/2-node states, stopping at the first unchanged resource cap.

    The caller fixes the anatomical node order (production: chest,mouth = 1,0).
    Every initial ID/node must be directly supported. Tiny retained states expose
    exact state tuples for independent exhaustive tests; real runs retain two layers.
    """
    frames, direct = np.asarray(frame_idx), np.asarray(direct_nodes)
    if (
        frames.ndim != 1
        or not len(frames)
        or frames.dtype != np.int64
        or np.any(frames < 0)
        or np.any(np.diff(frames) != 1)
    ):
        raise ValueError("consecutive nonnegative native int64 frame_idx required")
    if (
        direct.ndim != 3
        or direct.shape[:2] != (len(frames), 3)
        or direct.shape[2] not in (1, 2)
        or direct.dtype != bool
    ):
        raise ValueError("boolean support T,3,K with K=1 or 2 required")
    if not direct[0].all():
        raise ValueError("all initial ID/node cells must be directly supported")
    if len(frames) > np.iinfo(np.int64).max // 3:
        raise ValueError("event encoding exceeds int64")
    if retain_layers and len(frames) > 8:
        raise ValueError("retained layers restricted to tiny <=8-frame fixtures")
    caps = Caps() if caps is None else caps
    if not isinstance(caps, Caps):
        raise ValueError("unchanged Caps object required")
    nodes = direct.shape[2]
    initial = tuple(slot for slot in range(3) for _ in range(nodes))
    current = {(0, initial)}
    counts, edges, total_states, total_edges = [1], [], 1, 0
    layers = [frozenset(current)] if retain_layers else None
    stop = None
    for t in range(1, len(frames)):
        full_edges = 6 * len(current)
        if total_edges + full_edges > caps.maximum_total_edges:
            stop = dict(
                reason="maximum_total_edges",
                attempted_frame=int(frames[t]),
                next_layer_edges=full_edges,
                required_total_edges=total_edges + full_edges,
                attempted_layer_states_lower_bound=None,
            )
            break
        limit = min(caps.maximum_layer_states, caps.maximum_total_states - total_states)
        next_states = set()
        # Equivalent memories can share successor construction. Their incoming
        # paths still contribute six edges per complete predecessor state.
        memories = (None,) if direct[t].all() else {s[1] for s in current}
        for memory in memories:
            for p, slots in enumerate(PERMUTATIONS):
                events = tuple(
                    3 * t + slot if direct[t, slot, k] else memory[identity * nodes + k]
                    for identity, slot in enumerate(slots)
                    for k in range(nodes)
                )
                next_states.add((p, events))
                if len(next_states) > limit:
                    stop = dict(
                        reason="maximum_layer_states"
                        if len(next_states) > caps.maximum_layer_states
                        else "maximum_total_states",
                        attempted_frame=int(frames[t]),
                        next_layer_edges=full_edges,
                        required_total_edges=total_edges + full_edges,
                        attempted_layer_states_lower_bound=len(next_states),
                    )
                    break
            if stop is not None:
                break
        if stop is not None:
            break
        current = next_states
        counts.append(len(current))
        edges.append(full_edges)
        total_states += len(current)
        total_edges += full_edges
        if layers is not None:
            layers.append(frozenset(current))
    return ReachableCount(
        stop is None,
        frames[: len(counts)].copy(),
        np.asarray(counts, np.int64),
        np.asarray(edges, np.int64),
        total_states,
        total_edges,
        stop,
        None if layers is None else tuple(layers),
    )

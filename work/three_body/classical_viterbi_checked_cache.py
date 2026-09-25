"""C25's original objective with a checked transition-cache identity.

The solve loop is copied from the pinned C25 source. Only cached transition
reuse changes: each existing key keeps one (packed-state copy, moved matrix),
and reuse requires exact equality with that packed-state list. A mismatch
replaces the entry. No seed constraint, objective, prune bound or cap changes.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import c25_variable_cardinality as original  # noqa: E402
from c25_variable_cardinality import (  # noqa: E402
    BLOCK,
    PRUNE_ABOVE,
    TRANSITION_LIMIT,
    frame_margin,
    prune,
    slot_identities,
    slots_moved,
    state_costs,
    state_table,
    viterbi_step,
)

ORIGINAL_SHA256 = "1b7662bd5c1235165f0aa9665ba899ee45c8a1b9acbfb27d629f5b63b5dfe7ad"


def verify_source():
    if hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("frozen original C25 source differs")


def solve(
    rays,
    nodes,
    label_at,
    k,
    frames,
    lam: float,
    mu: float,
    base: int,
    *,
    diagnostics: dict | None = None,
):
    """Original six-value result; optional empty dict records cache replacements.

    Replacement counters identify exercised cache defects, not differences in
    the final real-data path, which this solve does not compare with an old run.
    """
    verify_source()
    if diagnostics is not None:
        if not isinstance(diagnostics, dict) or diagnostics:
            raise ValueError("diagnostics must be an empty output dict")
        diagnostics.update(cache_replacements=0, numerically_different_cache_replacements=0)
    counts = np.stack([k[view][frames] for view in original.VIEWS], axis=1)
    groups, inverse = np.unique(counts, axis=0, return_inverse=True)
    tables = [state_table(group) for group in groups]
    sizes = np.array([table.shape[0] for table in tables])

    n_states = sizes[inverse]
    offset = np.r_[0, np.cumsum(n_states)]
    back = np.zeros(int(offset[-1]), np.int32)
    margin = np.full(frames.size, np.nan)

    value, previous, shared = None, None, {}
    kept_of: dict[int, np.ndarray] = {}
    pruned: dict[int, tuple] = {}
    for start in range(0, frames.size, BLOCK):
        block = slice(start, min(start + BLOCK, frames.size))
        cost_of: dict[int, np.ndarray] = {}
        packed_of: dict[int, np.ndarray] = {}
        for index, table in enumerate(tables):
            member = np.flatnonzero(inverse[block] == index)
            if member.size == 0:
                continue
            picked = frames[block][member]
            cost = state_costs(rays, nodes, table, picked, mu)
            packed = slot_identities(table, label_at, picked, base)
            margin[np.arange(block.start, block.stop)[member]] = frame_margin(cost, packed)
            if table.shape[0] > PRUNE_ABOVE:
                trimmed, thinned, keeps = prune(cost, packed, lam)
                for position, local in enumerate(member):
                    cost_of[int(local)] = trimmed[position]
                    packed_of[int(local)] = thinned[position]
                    kept_of[block.start + int(local)] = keeps[position]
                    pruned[block.start + int(local)] = (
                        int(table.shape[0]),
                        int(keeps[position].size),
                    )
                continue
            for position, local in enumerate(member):
                cost_of[int(local)] = cost[position]
                packed_of[int(local)] = packed[position]

        for local in range(block.stop - block.start):
            t = block.start + local
            here, cost = packed_of[local], cost_of[local]
            if value is None:
                value = cost.copy()
            else:
                moved = None
                if (
                    inverse[t] == inverse[t - 1]
                    and np.array_equal(previous, here)
                    and here.shape[0] ** 2 <= TRANSITION_LIMIT
                ):
                    key = (int(inverse[t]), int(here.shape[0]))
                    cached = shared.get(key)
                    if cached is not None and np.array_equal(cached[0], here):
                        moved = cached[1]
                    else:
                        moved = slots_moved(previous, here)
                        if cached is not None and diagnostics is not None:
                            diagnostics["cache_replacements"] += 1
                            diagnostics["numerically_different_cache_replacements"] += int(
                                not np.array_equal(cached[1], moved)
                            )
                        shared[key] = (here.copy(), moved)
                value, best = viterbi_step(value, previous, here, cost, lam, moved)
                back[offset[t] : offset[t] + best.size] = best
            previous = here

    chosen = np.zeros(frames.size, np.int32)
    chosen[-1] = int(value.argmin())
    for t in range(frames.size - 1, 0, -1):
        chosen[t - 1] = back[offset[t] + chosen[t]]
    for t, kept in kept_of.items():
        chosen[t] = kept[chosen[t]]
    return chosen, float(value.min()), inverse, tables, margin, pruned

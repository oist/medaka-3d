"""C25 — cross-view correspondence when a camera's detection count is not three.

=============================================================================
WHAT THIS REPLACES
=============================================================================

`c19` solves the correspondence as a sequence: a state is the pair of
permutations ``top label -> side1 label``, ``top label -> side2 label``, 36 of
them, the same 36 at every frame, chosen by an exact Viterbi. `c20` removed the
"exactly three detections per view" rule while keeping those 36 states, because
`c16`'s capped tracker holds exactly three labels per view by construction.

`c24` removes the cap, for the reason `c23` measured: a cap does not refuse a
false positive, it makes one animal go unlabelled while the false positive takes
its label. Once the cap is gone a view may hold four labels, or two, and "three
labels per view" is no longer available as a state space.

=============================================================================
THE STATE, AND WHY IT IS STILL SMALL
=============================================================================

**Alphabet: the labels this frame holds, and nothing else.** Keeping recently
seen labels in the alphabet was measured and rejected before this was written —
on `side1_cam` 100,000-120,000 the mean state count per frame goes 66 (present
only) -> 11,523 (present within 25 frames) -> 27,839 (within one second),
because two states differing only in which *absent* label a slot points at cost
exactly the same and are still counted separately. Identity across a gap is
carried by the **slot**, which is what the sequence model is for, not by
keeping a label alive in the alphabet.

**State: an injective, maximal assignment of the frame's present labels to the
three slots.** Maximal means no present label is left unused while a slot sits
empty; a slot is empty only where the frame is short of labels. Per view that is
``perm(k, 3)`` assignments for ``k >= 3`` and ``perm(3, k)`` below:

    k        0   1   2   3    4    5
    states   1   3   6   6   24   60

``k = 3`` gives 6, which is exactly `c19`'s six permutations of one view. The
frame's state is the product over the three views: 216 on a normal frame against
`c19`'s 36, the factor 6 being the slot naming `c19` fixes to ``top_cam`` and
this cannot, because a slot's identity has to survive ``top_cam`` changing which
labels it holds.

**Cost: every state carries the same number of terms.** `c20` found that its
cost, a mean over the terms a state happened to support, lets a state that
explains less win — "a frame resting on four rays pushes as hard as one resting
on nine" — and its state changes tripled. Here the denominator is fixed at
``n_slots x n_nodes`` and a (slot, node) that no two cameras support contributes
``--mu`` millimetres instead of dropping out. ``mu`` is the price of leaving an
animal unexplained in a frame, in the same unit as the residual, and it is what
decides whether a fourth detection is an animal or a reflection — **by geometry,
in millimetres, never by detector score**, which `c24` measured does not
separate the two (92.7% of non-primary-label instances score above the 5th
percentile of the primary labels').

**Penalty: per slot.** `c19` charges lambda when *the state* changes. A handover
— one label ending while another continues the same animal, 12.6% of frames on
the window `c24` measured — changes one slot, and charging it as a whole-state
change is what makes handovers look like correspondence errors.

    J(s) = SUM_t SUM_a cost_t(s_t,a)  +  lambda * SUM_t #{a : s_t,a != s_t-1,a}

Both terms decompose over slots, so the exact Viterbi is unchanged in kind. The
state list differs from frame to frame, so the transition is computed against
the previous frame's list instead of being read out of a constant matrix.

=============================================================================
WHAT IT MUST REPRODUCE, AND THE TOOL CHECKS IT
=============================================================================

``--check-reduction`` runs on `c16`'s **capped** labels, where every frame holds
at most three labels per view. On every frame that holds three labels in all
three views and supports all six (slot, node) terms, both are a mean over the
same six terms, so this enumeration must reproduce `c20`'s `label_costs` — and
any difference is a bug in the compact layout, which is silent on real data and
would otherwise look like a result.

Two comparisons, because they fail differently. The **cost** is compared to a
tolerance *derived* from `label_costs` storing float32: a few ulp at the
observed magnitude is the storage, not the arithmetic. Chosen tolerances hide
exactly this kind of bug, so it is computed rather than picked. The **winner**
— the set of label triples the correspondence chooses — is compared outright,
because float32 cannot move it except at a tie, and a tie is counted as a tie.

The objective is deliberately *not* identical everywhere: ``mu`` and the
per-slot penalty are the changes under test. The reduction is asserted where the
two agree by construction, and the difference elsewhere is the measurement.

**n_groups = 1, n_experiments = 1.** Swap counts of 1-8 are small samples and no
ranking built on them is stable.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c25_variable_cardinality.py --check-reduction
    .venv\\Scripts\\python.exe tools/c25_variable_cardinality.py --labels c24
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from itertools import permutations, product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from c02_tracking_probe import CALIBRATION
from c19_viterbi_correspondence import KAPPA, coverage, derive_gate

from medaka3d import load_paths
from medaka3d.calib import load_rig, triangulate
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.reconstruct import N_ANIMALS, UNRESOLVED, Reconstruction
from medaka3d.track import METHODS, UNTRACKED, propagate_to_views
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN = "bottomup"
LABEL_SOURCES = {
    "c16": Path("c16_sleap_tracker_2d") / "mouth-pectoral_oks",
    "c24": Path("c24_uncapped_tracklets") / "mouth-pectoral_oks",
    "c24b": Path("c24b_gated_tracklets") / "mouth-pectoral_oks",
}

#: A slot holding no detection in a view.
EMPTY = -1

#: More labels in one frame than any view has been measured to hold. The census
#: over the three cached K days (2026-08-24, `s16` bottom-up): the most any view
#: ever reports in one frame is **10**, on ``top_cam`` in K_1_1; the sides reach
#: 7. Frames past five detections are 4-11 per hundred thousand. The layout
#: raises rather than truncating if a frame exceeds this — a silently dropped
#: detection is exactly the failure this tool exists to remove.
K_MAX = 12

#: Frames per block. Chosen for peak memory, not for speed: the cost tensor is
#: (block, states, slots, views, 3, 3) inside `triangulate`.
BLOCK = 2_000

#: Elements in one transition temporary. A frame can hold 172,800 states — six
#: labels in one view, four in another, five in the third — and the transition
#: against its neighbour is ``n_before x n_after``, so it is chunked rather than
#: allocated whole. Without this the rare frame decides the memory.
TRANSITION_LIMIT = 16_000_000

#: Above this many states a frame is **pruned, provably losslessly**. The
#: penalty is ``lambda`` per slot that changes, at most ``3 lambda`` for one
#: transition, so replacing a state by the frame's cheapest costs at most
#: ``6 lambda`` — three slots entering and three leaving. A state more than
#: ``6 lambda`` above the cheapest can therefore never lie on the optimal path,
#: whatever the rest of the recording does, and dropping it changes no answer.
#: This is a bound, not a heuristic. Only frames past this size are pruned, so
#: the ordinary 216-state frame is untouched and keeps its cached transition.
PRUNE_ABOVE = 2_000


# ---------------------------------------------------------------------------
# the frame's labels, laid out compactly
# ---------------------------------------------------------------------------


def compact_layout(detections, labels: np.ndarray, nodes: tuple[str, ...]):
    """Per frame, this view's present labels in ascending order, packed to 0..k-1.

    Returns ``(uv, label_at, row_at, k)``:

    ``uv``        ``node -> (n_frames, K_MAX, 2)`` pixels; NaN past ``k``.
    ``label_at``  (n_frames, K_MAX) which 2D track label each compact index
                  holds, ``EMPTY`` past ``k``. **The transition term is defined
                  on this**, never on the compact index, which means something
                  different at every frame.
    ``row_at``    (n_frames, K_MAX) the detector's own row offset, because `c09`
                  inverts it to find which slot holds the human's fish; handing
                  it anything but a row pairs one animal's 3D position with
                  another animal's pixels, silently.
    ``k``         (n_frames,) how many labels the frame holds.
    """
    n_frames = detections.n_frames
    live = labels >= 0
    frame = detections.frame_idx[live].astype(np.int64)
    label = labels[live].astype(np.int64)
    row = (np.arange(labels.size) - detections.start[detections.frame_idx])[live]

    order = np.lexsort((label, frame))
    frame, label, row = frame[order], label[order], row[order]
    rank = np.arange(frame.size) - np.searchsorted(frame, frame, side="left")
    if rank.size and rank.max() >= K_MAX:
        raise SystemExit(f"a frame holds {rank.max() + 1} labels; K_MAX is {K_MAX}")

    uv = {}
    for node in nodes:
        values = np.full((n_frames, K_MAX, 2), np.nan)
        values[frame, rank] = detections.node(node)[live][order]
        uv[node] = values
    label_at = np.full((n_frames, K_MAX), EMPTY, np.int32)
    label_at[frame, rank] = label
    row_at = np.full((n_frames, K_MAX), UNRESOLVED, np.int8)
    row_at[frame, rank] = row.astype(np.int8)
    k = np.bincount(frame, minlength=n_frames)[:n_frames].astype(np.int8)
    return uv, label_at, row_at, k


def view_assignments(k: int) -> np.ndarray:
    """(n, 3) compact index per slot for one view holding ``k`` present labels.

    Maximal and injective: every present label is used, and a slot is ``EMPTY``
    only because the frame holds fewer labels than there are slots.
    """
    if k >= N_ANIMALS:
        return np.asarray(list(permutations(range(k), N_ANIMALS)), dtype=np.int8)
    out = []
    for slots in permutations(range(N_ANIMALS), k):
        row = np.full(N_ANIMALS, EMPTY, dtype=np.int8)
        for index, slot in enumerate(slots):
            row[slot] = index
        out.append(row)
    return np.asarray(out, dtype=np.int8)


def state_table(counts) -> np.ndarray:
    """(n_states, n_slots, n_views) compact index, for one combination of counts."""
    per_view = [view_assignments(int(k)) for k in counts]
    return np.stack([np.stack(choice, axis=-1) for choice in product(*per_view)],
                    axis=0).astype(np.int8)


# ---------------------------------------------------------------------------
# what a state costs
# ---------------------------------------------------------------------------


def compact_rays(rig, uv_by_view: dict, nodes: tuple[str, ...]) -> dict:
    """``(view, node) -> (origins, directions, present)`` on the compact layout.

    Presence is read off the rays, not off the pixels, so a pixel that
    back-projects to nothing counts as absent rather than as a NaN that poisons
    a triangulation. Same rule as `correspond.label_rays`.
    """
    out = {}
    for view in rig.views:
        camera = rig.camera(view)
        for node in nodes:
            origins, directions = camera.rays(uv_by_view[view][node])
            present = (np.isfinite(origins).all(axis=-1)
                       & np.isfinite(directions).all(axis=-1))
            out[(view, node)] = (origins, directions, present)
    return out


def state_costs(rays, nodes, table: np.ndarray, frames: np.ndarray,
                mu: float) -> np.ndarray:
    """(n_frames, n_states) mm: the mean over a **fixed** number of terms.

    Every state at a frame divides by ``n_slots * n_nodes``, so states are
    directly comparable however much of the frame they explain. A (slot, node)
    no two cameras support costs ``mu``.
    """
    rows = frames[:, None, None]
    total = np.zeros((frames.size, table.shape[0]))
    for node in nodes:
        origins, directions, held = [], [], []
        for index, view in enumerate(VIEWS):
            ray_o, ray_d, present = rays[(view, node)]
            pick = table[None, :, :, index]                 # (1, n_states, n_slots)
            take = np.where(pick >= 0, pick, 0)
            origins.append(ray_o[rows, take])               # (f, s, slot, 3)
            directions.append(ray_d[rows, take])
            held.append(present[rows, take] & (pick >= 0))
        origins = np.stack(origins, axis=3)                 # (f, s, slot, view, 3)
        directions = np.stack(directions, axis=3)
        held = np.stack(held, axis=3)                       # (f, s, slot, view)

        _, rms = triangulate(origins, directions, valid=held)
        live = (held.sum(axis=3) >= 2) & np.isfinite(rms)
        total += np.where(live, rms, mu).sum(axis=2)
    return (total / (N_ANIMALS * len(nodes))).astype(np.float64)


def state_costs_by_node(rays, nodes, table: np.ndarray, frames: np.ndarray,
                       mu: float) -> dict:
    """``node -> (n_frames, n_states)`` mm, each on the same fixed denominator.

    The combined cost is the mean of these, so a penalty chosen against one node
    transfers to the other — which is what the held-out selection needs.
    """
    rows = frames[:, None, None]
    out, usable = {}, {}
    for node in nodes:
        origins, directions, held = [], [], []
        for index, view in enumerate(VIEWS):
            ray_o, ray_d, present = rays[(view, node)]
            pick = table[None, :, :, index]
            take = np.where(pick >= 0, pick, 0)
            origins.append(ray_o[rows, take])
            directions.append(ray_d[rows, take])
            held.append(present[rows, take] & (pick >= 0))
        seen = np.stack(held, axis=3)
        _, rms = triangulate(np.stack(origins, axis=3), np.stack(directions, axis=3),
                             valid=seen)
        live = (seen.sum(axis=3) >= 2) & np.isfinite(rms)
        out[node] = np.where(live, rms, mu).sum(axis=2) / N_ANIMALS
        # Which frames this node says anything about. Everywhere else every
        # state costs `mu`, the margin is 0 mm by construction, and including
        # those frames in the median would let absence choose the penalty.
        usable[node] = live.any(axis=(1, 2))
    return out, usable


def sweep_kappa(rays, nodes, label_at, k, frames, grid, mu: float, base: int) -> dict:
    """`c19`'s pre-registered choice of lambda, on this state space.

    Fit the sequence on one node, score it on the other, both ways, and report
    in ``kappa = lambda / m`` where ``m`` is the fit node's median runner-up
    margin — so the number transfers from a one-node fit to the two-node run it
    is used in. **No human annotation is read**, which is what makes the choice
    a rule rather than a result.

    Two passes. The first builds every frame's per-node state costs and measures
    the margins; the second runs the grid. The margin has to be a property of
    the whole window rather than of whichever frames came first, and building
    the costs once serves both directions.

    The grid runs in **one pass** over the frames. The penalty decomposes as
    ``lambda x (slots that changed)``, so every lambda shares one count matrix,
    and the held-out cost and the change count accumulate along the DP instead
    of being read back off a stored path — 25 paths over 348,600 frames is
    memory this does not need to spend.
    """
    counts = np.stack([k[view][frames] for view in VIEWS], axis=1)
    groups, inverse = np.unique(counts, axis=0, return_inverse=True)
    tables = [state_table(group) for group in groups]

    # --- pass one: the costs, and the margins that scale the grid ------------
    cost_of: list = [None] * frames.size
    packed_of: list = [None] * frames.size
    seen = {node: [] for node in nodes}
    seen_total: list = []
    for start in range(0, frames.size, BLOCK):
        block = slice(start, min(start + BLOCK, frames.size))
        for index, table in enumerate(tables):
            member = np.flatnonzero(inverse[block] == index)
            if member.size == 0:
                continue
            picked = frames[block][member]
            per_node, usable = state_costs_by_node(rays, nodes, table, picked, mu)
            packed = slot_identities(table, label_at, picked, base)
            both = usable[nodes[0]] & usable[nodes[1]]
            combined = sum(per_node.values()) / len(nodes)
            seen_total.append(frame_margin(combined, packed)[both])
            for node in nodes:
                seen[node].append(frame_margin(per_node[node], packed)[usable[node]])
            for position, local in enumerate(member):
                t = block.start + int(local)
                cost_of[t] = {node: per_node[node][position].astype(np.float32)
                              for node in nodes}
                packed_of[t] = packed[position]

    margins = {}
    for node in nodes:
        pooled = np.concatenate(seen[node])
        margins[node] = float(np.nanmedian(pooled)) if pooled.size else 0.0
        if not margins[node] > 0:
            raise SystemExit(
                f"the {node} node discriminates nothing on these frames; the "
                "penalty cannot be expressed in its margin")
    margins["total"] = float(np.nanmedian(np.concatenate(seen_total)))

    # --- pass two: the grid, both directions ---------------------------------
    curves = {}
    for fit, other in ((nodes[0], nodes[1]), (nodes[1], nodes[0])):
        lam = np.asarray(grid) * margins[fit]
        value = held = changes = previous = None
        shared: dict[tuple, np.ndarray] = {}

        for t in range(frames.size):
            here = packed_of[t]
            fit_cost = cost_of[t][fit].astype(float)
            held_cost = cost_of[t][other].astype(float)
            if t > 0 and here.shape[0] > PRUNE_ABOVE:
                keep = np.flatnonzero(fit_cost <= fit_cost.min() + 6.0 * lam.max())
                here, fit_cost, held_cost = here[keep], fit_cost[keep], held_cost[keep]
            if value is None:
                value = np.tile(fit_cost, (lam.size, 1))
                held = np.tile(held_cost, (lam.size, 1))
                changes = np.zeros_like(value)
                previous = here
                continue

            cached = None
            if (np.array_equal(previous, here)
                    and here.shape[0] ** 2 <= TRANSITION_LIMIT):
                key = (int(inverse[t]), int(here.shape[0]))
                cached = shared.get(key)
                if cached is None:
                    cached = slots_moved(previous, here)
                    shared[key] = cached
            n_here = here.shape[0]
            new_value = np.empty((lam.size, n_here))
            new_held = np.empty((lam.size, n_here))
            new_changes = np.empty((lam.size, n_here))
            width = (n_here if cached is not None
                     else chunk_width(previous.shape[0], lam.size))
            columns = np.arange(lam.size)[:, None]
            for edge in range(0, n_here, width):
                piece = slice(edge, min(edge + width, n_here))
                moved = (cached[:, piece] if cached is not None
                         else slots_moved(previous, here[piece]))
                candidate = value[:, :, None] + lam[:, None, None] * moved[None]
                best = candidate.argmin(axis=1)
                inside = np.arange(best.shape[1])[None]
                new_value[:, piece] = candidate[columns, best, inside] + fit_cost[piece]
                new_held[:, piece] = held[columns, best] + held_cost[piece]
                new_changes[:, piece] = changes[columns, best] + moved[best, inside]
            value, held, changes = new_value, new_held, new_changes
            previous = here

        rows = np.arange(lam.size)
        winner = value.argmin(axis=1)
        held_total = held[rows, winner]
        best_index = int(held_total.argmin())
        curves[fit] = {
            "held_out": other, "margin_mm": margins[fit],
            "kappa": list(map(float, grid)),
            "held_mean_mm": (held_total / frames.size).tolist(),
            "slot_changes": changes[rows, winner].astype(int).tolist(),
            "best_index": best_index, "best_kappa": float(grid[best_index]),
            "interior": bool(0 < best_index < len(grid) - 1),
        }

    kappa = float(np.sqrt(curves[nodes[0]]["best_kappa"] * curves[nodes[1]]["best_kappa"]))
    return {"curves": curves, "kappa": kappa, "margins_mm": margins,
            "lambda_mm": kappa * margins["total"],
            "rule": "held-out node cross-validation, c19's rule on c25's state space",
            "interior_minimum_both_directions": bool(
                curves[nodes[0]]["interior"] and curves[nodes[1]]["interior"])}


def slot_identities(table: np.ndarray, label_at: dict, frames: np.ndarray,
                    base: int) -> np.ndarray:
    """(n_frames, n_states, n_slots): one integer naming each slot's label triple.

    Two slots hold the same thing in consecutive frames exactly when this
    matches, so the transition term compares integers rather than triples.
    """
    packed = np.zeros((frames.size, table.shape[0], N_ANIMALS), np.int64)
    for index, view in enumerate(VIEWS):
        pick = table[None, :, :, index]
        take = np.where(pick >= 0, pick, 0)
        label = label_at[view][frames[:, None, None], take]
        packed = packed * base + np.where(pick >= 0, label + 1, 0)
    return packed


# ---------------------------------------------------------------------------
# the sequence
# ---------------------------------------------------------------------------


def chosen_geometry(rays, nodes, tables, inverse, chosen, label_at, row_at,
                    frames, anchor: str):
    """The reconstruction the sequence chose, in the shape every other stage reads.

    ``source`` records the detector's own **row** within the frame, never the 2D
    track label: `c09` inverts it to find which slot holds the human's fish, and
    handing it a label would pair one animal's 3D position with another animal's
    pixels, silently. ``margin_mm`` is measured against the cheapest state whose
    *set of slot label triples* differs — 216 ordered states cover each physical
    correspondence six times over, and comparing against a renaming of the
    winner would report 0 mm on every frame and read as total ambiguity.
    """
    size = frames.size
    xyz = np.full((size, N_ANIMALS, 3), np.nan)
    residual = np.full((size, N_ANIMALS), np.nan)
    source = np.full((size, N_ANIMALS, len(VIEWS)), UNRESOLVED, np.int8)
    slot_rays = np.zeros((size, N_ANIMALS), np.int8)

    for index, table in enumerate(tables):
        member = np.flatnonzero(inverse == index)
        for start in range(0, member.size, BLOCK):
            take = member[start:start + BLOCK]
            window = frames[take]
            pick = table[chosen[take]]                     # (w, n_slots, n_views)
            total = np.zeros((take.size, N_ANIMALS))
            count = np.zeros((take.size, N_ANIMALS))

            for node in nodes:
                origins, directions, held = [], [], []
                for i, view in enumerate(VIEWS):
                    ray_o, ray_d, present = rays[(view, node)]
                    here = pick[:, :, i]
                    safe = np.where(here >= 0, here, 0)
                    origins.append(ray_o[window[:, None], safe])
                    directions.append(ray_d[window[:, None], safe])
                    held.append(present[window[:, None], safe] & (here >= 0))
                seen = np.stack(held, axis=2)                  # (w, n_slots, n_views)
                point, rms = triangulate(np.stack(origins, axis=2),
                                         np.stack(directions, axis=2), valid=seen)
                live = (seen.sum(axis=2) >= 2) & np.isfinite(rms)
                total += np.where(live, rms, 0.0)
                count += live
                if node == anchor:
                    xyz[take] = point
                    slot_rays[take] = seen.sum(axis=2).astype(np.int8)

            residual[take] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
            for i, view in enumerate(VIEWS):
                here = pick[:, :, i]
                safe = np.where(here >= 0, here, 0)
                source[take, :, i] = np.where(
                    here >= 0, row_at[view][window[:, None], safe], UNRESOLVED)

    return xyz, residual, source, slot_rays


def chosen_identities(pick: np.ndarray, label_at: dict, window: np.ndarray,
                      base: int) -> np.ndarray:
    """(n_frames, n_slots) the label triple each slot holds, as one integer."""
    packed = np.zeros(pick.shape[:2], np.int64)
    for index, view in enumerate(VIEWS):
        here = pick[:, :, index]
        label = label_at[view][window[:, None], np.where(here >= 0, here, 0)]
        packed = packed * base + np.where(here >= 0, label + 1, 0)
    return packed


def frame_margin(cost: np.ndarray, packed: np.ndarray) -> np.ndarray:
    """(n_frames,) cheapest *physically different* state, minus the cheapest.

    Two states differing only in which slot is called which are the same
    correspondence — 216 ordered states cover each one six times over — so the
    runner-up is taken among states whose **set** of slot label triples differs.
    Measured against a renaming of the winner it would read 0 mm on every frame
    and look like total ambiguity. Reported against the per-frame cheapest, as
    `c19` and `c20` report theirs, so the percentiles compare.
    """
    key = np.sort(packed, axis=2)
    best = cost.argmin(axis=1)
    winner = key[np.arange(best.size), best][:, None, :]
    different = (key != winner).any(axis=2)
    other = np.where(different, cost, np.inf).min(axis=1)
    return other - cost[np.arange(best.size), best]


def slots_moved(before: np.ndarray, here: np.ndarray) -> np.ndarray:
    """(n_before, n_here) how many of the three slots changed label triple."""
    return N_ANIMALS - (before[:, None, :] == here[None, :, :]).sum(axis=2)


def chunk_width(n_before: int, layers: int = 1) -> int:
    """How many destination states one temporary may cover."""
    return max(1, TRANSITION_LIMIT // max(n_before * layers, 1))


def prune(cost: np.ndarray, packed: np.ndarray, lam: float):
    """Drop states that provably cannot lie on the optimal path.

    Returns ``(cost, packed, kept)`` as one entry per frame, where ``kept``
    indexes back into the state table. See `PRUNE_ABOVE` for why this is exact.
    """
    limit = cost.min(axis=1) + 6.0 * lam
    out_cost, out_packed, out_kept = [], [], []
    for row in range(cost.shape[0]):
        kept = np.flatnonzero(cost[row] <= limit[row])
        out_cost.append(cost[row][kept])
        out_packed.append(packed[row][kept])
        out_kept.append(kept)
    return out_cost, out_packed, out_kept


def viterbi_step(value: np.ndarray, before: np.ndarray, here: np.ndarray,
                 cost: np.ndarray, lam: float, moved: np.ndarray | None = None):
    """One frame of ``min_prev [ V + lam * #{slots that changed} ] + cost``."""
    n_here = here.shape[0]
    out = np.empty(n_here)
    best = np.empty(n_here, np.int32)
    width = n_here if moved is not None else chunk_width(before.shape[0])
    for start in range(0, n_here, width):
        piece = slice(start, min(start + width, n_here))
        step = moved[:, piece] if moved is not None else slots_moved(before, here[piece])
        candidate = value[:, None] + lam * step
        pick = candidate.argmin(axis=0)
        out[piece] = candidate[pick, np.arange(pick.size)] + cost[piece]
        best[piece] = pick
    return out, best


def solve(rays, nodes, label_at, k, frames, lam: float, mu: float, base: int):
    """Exact Viterbi over per-frame state lists.

    Returns ``(chosen, objective, inverse, tables, margin)``: the chosen state
    index per frame, the value of the objective, the per-frame index into
    ``tables`` — the state list depends only on how many labels each view holds,
    so frames sharing a cardinality share a table — and the per-frame runner-up
    margin in mm.
    """
    counts = np.stack([k[view][frames] for view in VIEWS], axis=1)
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
                    # `cost_of` is consumed inside this block, so it is keyed
                    # block-locally; `kept_of` is read by the backtrack after
                    # every block and must be keyed by the frame's own position.
                    cost_of[int(local)] = trimmed[position]
                    packed_of[int(local)] = thinned[position]
                    kept_of[block.start + int(local)] = keeps[position]
                    pruned[block.start + int(local)] = (
                        int(table.shape[0]), int(keeps[position].size))
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
                # Consecutive frames overwhelmingly hold the same labels in the
                # same numbers — the label set changes on 0.14% of frames — so
                # the penalty matrix is the same object over and over. Building
                # it once per state table turns the transition from the cost of
                # the loop into an addition.
                moved = None
                if (inverse[t] == inverse[t - 1] and np.array_equal(previous, here)
                        and here.shape[0] ** 2 <= TRANSITION_LIMIT):
                    # Keyed by the state count too: a group's list is not fixed
                    # once pruning is in play, and handing a pruned frame the
                    # unpruned matrix is wrong in a way that can be silent.
                    key = (int(inverse[t]), int(here.shape[0]))
                    moved = shared.get(key)
                    if moved is None:
                        moved = slots_moved(previous, here)
                        shared[key] = moved
                value, best = viterbi_step(value, previous, here, cost, lam, moved)
                back[offset[t]:offset[t] + best.size] = best
            previous = here

    chosen = np.zeros(frames.size, np.int32)
    chosen[-1] = int(value.argmin())
    for t in range(frames.size - 1, 0, -1):
        chosen[t - 1] = back[offset[t] + chosen[t]]
    for t, kept in kept_of.items():                 # pruned index -> table row
        chosen[t] = kept[chosen[t]]
    return chosen, float(value.min()), inverse, tables, margin, pruned


# ---------------------------------------------------------------------------
# the reduction: where this and `c20` must agree by construction
# ---------------------------------------------------------------------------


def signature(labels: np.ndarray) -> np.ndarray:
    """(..., n_slots) frame-order-independent encoding of a correspondence.

    Slots are arbitrary within a frame — `c19.slot_signature` says so, and this
    enumeration names them differently from `c20`, which fixes slot *a* to
    ``top_cam`` label *a*. Comparing element-wise would report a disagreement
    that is not one, so each slot's label triple becomes one integer and the
    three are sorted: only the **set** of triples is compared.
    """
    weights = 4 ** np.arange(labels.shape[-1])
    return np.sort(((labels + 1) * weights).sum(axis=-1), axis=-1)


def check_reduction(paths, rig, nodes, frames, rays, k, label_at) -> dict:
    """On fully supported three-label frames, is this `c20`'s `label_costs`?

    Two comparisons, because they fail differently.

    **The cost**, to a tolerance *derived* rather than chosen: `label_costs`
    stores its costs as float32, so a difference of a few ulp at the observed
    magnitude is the storage and not the arithmetic. Anything larger is a wrong
    index in the compact layout.

    **The winner**, which is the substantive claim and is unaffected by float32:
    the correspondence chosen must be the same set of label triples. A frame
    where the two disagree but this enumeration's best two states are within
    tolerance of each other is a tie, not a disagreement, and is counted as one.
    """
    import c20_uncapped_correspondence as c20

    from medaka3d.correspond import label_costs
    from medaka3d.reconstruct import hypotheses

    by_label = {}
    for view in VIEWS:
        detections = c01.load(paths.work_dir, EXPERIMENT, view, RUN)
        labels = np.load(paths.work_dir / LABEL_SOURCES["c16"] / f"track2d__{view}.npy")
        by_label[view], _, _, _ = c20.label_arrays(detections, labels)
    old = label_costs(rig, by_label, c01.N_FRAMES, anchor=nodes[0], nodes=tuple(nodes))
    old_states = hypotheses(len(VIEWS))

    three = np.logical_and.reduce([k[view][frames] == N_ANIMALS for view in VIEWS])
    comparable = three & (old.terms[frames].min(axis=1) == N_ANIMALS * len(nodes))
    picked = frames[comparable]
    if picked.size == 0:
        raise SystemExit("no frame is comparable; the reduction cannot be checked")

    table = state_table((N_ANIMALS,) * len(VIEWS))
    gap, magnitude, seen, disagreed, ties = 0.0, 0.0, 0, 0, 0
    for start in range(0, picked.size, BLOCK):
        window = picked[start:start + BLOCK]
        mine = state_costs(rays, nodes, table, window, 0.0)
        theirs = old.cost[window].astype(float)
        best_mine, best_theirs = mine.argmin(axis=1), theirs.argmin(axis=1)
        gap = max(gap, float(np.abs(mine.min(axis=1) - theirs.min(axis=1)).max()))
        magnitude = max(magnitude, float(np.abs(theirs.min(axis=1)).max()))

        # This enumeration's winner, in labels rather than compact indices.
        pick = table[best_mine]                                   # (f, slot, view)
        mine_labels = np.stack(
            [label_at[view][window[:, None], np.where(pick[:, :, i] >= 0,
                                                      pick[:, :, i], 0)]
             for i, view in enumerate(VIEWS)], axis=2)
        differs = (signature(mine_labels) != signature(old_states[best_theirs])).any(axis=1)

        spread = np.partition(mine, 1, axis=1)[:, 1] - mine.min(axis=1)
        margin = 16 * np.finfo(np.float32).eps * np.maximum(np.abs(theirs).max(axis=1), 1.0)
        ties += int((differs & (spread <= margin)).sum())
        disagreed += int((differs & (spread > margin)).sum())
        seen += window.size

    tolerance = float(16 * np.finfo(np.float32).eps * max(magnitude, 1.0))
    return {"frames_compared": int(seen), "max_abs_gap_mm": gap,
            "cost_magnitude_mm": magnitude, "tolerance": tolerance,
            "winner_disagreements": disagreed, "winner_ties": ties,
            "passed": bool(gap <= tolerance and disagreed == 0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--labels", default="c24", choices=sorted(LABEL_SOURCES))
    parser.add_argument("--nodes", nargs="+", default=["pectoral", "mouth"],
                        help="the first is the anchor, as in c13/c18")
    parser.add_argument("--mu", type=float, default=4.0,
                        help="mm charged for a (slot, node) no two cameras support")
    parser.add_argument("--lam", type=float, default=None,
                        help="override the cross-validated penalty. POST-HOC")
    parser.add_argument("--frames", nargs=2, type=int, default=None,
                        help="frame range; for validation runs. Not a different algorithm")
    parser.add_argument("--select-frames", nargs=2, type=int, default=None,
                        help="frames the lambda selection reads. Defaults to all of "
                             "them; a window is a disclosed deviation and the tool "
                             "records it, so run two disjoint ones and compare")
    parser.add_argument("--gate-mm", type=float, default=None,
                        help="hold the linker's gate instead of deriving it. POST-HOC")
    parser.add_argument("--max-gap-frames", type=int, default=None,
                        help="hold the linker's gap tolerance. POST-HOC")
    parser.add_argument("--check-reduction", action="store_true")
    args = parser.parse_args()

    paths = load_paths()
    nodes = tuple(args.nodes)
    source = "c16" if args.check_reduction else args.labels
    tag = source if args.lam is None else f"{source}_lam{args.lam:g}"
    out_dir = paths.work_dir / "c25_variable_cardinality" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("variable_cardinality", out_dir, name="c25") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        rig = load_rig(paths.asset_root / CALIBRATION)
        record.input("calibration", rig.manifest)

        lo, hi = args.frames if args.frames else (0, c01.N_FRAMES)
        frames = np.arange(lo, hi)
        print(f"{EXPERIMENT}   labels {source}   nodes {nodes}   mu {args.mu} mm"
              + (f"   lambda {args.lam} (post-hoc)" if args.lam is not None else ""))
        print(f"frames {lo:,}-{hi:,}   n_groups = 1, n_experiments = 1\n")

        uv, label_at, row_at, k, n_labels = {}, {}, {}, {}, 0
        starts, counts = {}, {}
        for view in VIEWS:
            detections = c01.load(paths.work_dir, EXPERIMENT, view, RUN)
            labels = np.load(paths.work_dir / LABEL_SOURCES[source]
                             / f"track2d__{view}.npy")
            uv[view], label_at[view], row_at[view], k[view] = compact_layout(
                detections, labels, nodes)
            starts[view], counts[view] = detections.start, int(labels.size)
            n_labels = max(n_labels, int(labels.max()) + 1)
            census = np.bincount(k[view][frames], minlength=6)
            print(f"  {view:<10} labels per frame  "
                  + "  ".join(f"{i}:{c:,}" for i, c in enumerate(census) if c))
        rays = compact_rays(rig, uv, nodes)

        if args.check_reduction:
            started = time.time()
            result = check_reduction(paths, rig, nodes, frames, rays, k, label_at)
            record.param("reduction", result)
            print(f"\nreduction to c20 on {result['frames_compared']:,} fully "
                  f"supported three-label frames, {time.time() - started:.0f}s")
            print(f"  max |cost gap|        {result['max_abs_gap_mm']:.3e} mm  "
                  f"against a derived {result['tolerance']:.3e} mm "
                  f"(float32 storage in label_costs; costs reach "
                  f"{result['cost_magnitude_mm']:.1f} mm)")
            print(f"  winner disagreements  {result['winner_disagreements']:,}"
                  f"   ties counted as ties: {result['winner_ties']:,}")
            print(f"  {'passed' if result['passed'] else 'FAILED'}")
            if not result["passed"]:
                raise SystemExit("the enumeration does not reproduce c20's geometry")
            return

        if args.lam is None:
            started = time.time()
            window = (np.arange(*args.select_frames) if args.select_frames
                      else frames)
            choice = sweep_kappa(rays, nodes, label_at, k, window, KAPPA, args.mu,
                                 n_labels + 2)
            choice["selection_frames"] = [int(window[0]), int(window[-1]) + 1]
            choice["selection_is_a_window"] = args.select_frames is not None
            penalty = choice["lambda_mm"]
            record.param("lambda_choice", choice)
            print(f"\n  lambda chosen in {time.time() - started:.0f}s by "
                  f"held-out node cross-validation, no human annotation read")
            for node, curve in choice["curves"].items():
                print(f"    fit {node:<9} score {curve['held_out']:<9} "
                      f"kappa* {curve['best_kappa']:.4f}  "
                      f"margin {curve['margin_mm']:.3f} mm  "
                      f"{'interior' if curve['interior'] else 'AT THE GRID EDGE'}")
            print(f"    kappa {choice['kappa']:.4f} x total margin "
                  f"{choice['margins_mm']['total']:.3f} mm  ->  lambda "
                  f"{penalty:.3f} mm")
        else:
            penalty = args.lam
            print(f"\n  !! lambda held at {penalty} mm — a POST-HOC choice")
        record.param("lambda_mm_used", float(penalty))
        record.param("lambda_is_post_hoc", args.lam is not None)

        started = time.time()
        chosen, objective, inverse, tables, margin, pruned = solve(
            rays, nodes, label_at, k, frames, penalty, args.mu, n_labels + 2)
        sizes = np.array([table.shape[0] for table in tables])[inverse]
        print(f"\n  states per frame  mean {sizes.mean():.0f}  max {sizes.max():,}"
              f"   (c19's 36, counted the same way, is 216)")
        print(f"  solved in {time.time() - started:.0f}s   objective {objective:,.1f}")
        record.param("states_per_frame_mean", float(sizes.mean()))
        record.param("states_per_frame_max", int(sizes.max()))
        record.param("objective", objective)
        if pruned:
            dropped = [(full, kept) for full, kept in pruned.values()]
            print(f"  {len(pruned):,} frames pruned losslessly (over {PRUNE_ABOVE:,} "
                  f"states): {max(f for f, _ in dropped):,} states at most, "
                  f"{max(k for _, k in dropped):,} kept at most")
            record.param("frames_pruned", len(pruned))
            record.param("pruned_largest_kept", int(max(k for _, k in dropped)))

        # How often the chosen assignment moves a slot onto a different label.
        # A handover costs one slot; a correspondence change costs two or three.
        packed = np.zeros((frames.size, N_ANIMALS), np.int64)
        for index, table in enumerate(tables):
            member = np.flatnonzero(inverse == index)
            for start in range(0, member.size, BLOCK):
                take = member[start:start + BLOCK]
                packed[take] = chosen_identities(
                    table[chosen[take]], label_at, frames[take], n_labels + 2)
        changed = (packed[1:] != packed[:-1]).sum(axis=1)
        print(f"  slot label changes {int(changed.sum()):,} over {frames.size:,} frames"
              f"   (1 slot: {int((changed == 1).sum()):,}, "
              f"2: {int((changed == 2).sum()):,}, 3: {int((changed == 3).sum()):,})")
        record.param("slot_label_changes", int(changed.sum()))
        record.param("frames_with_one_slot_changed", int((changed == 1).sum()))

        # --- the reconstruction, in the shape every other stage reads ---------
        started = time.time()
        xyz, residual, source_rows, slot_rays = chosen_geometry(
            rays, nodes, tables, inverse, chosen, label_at, row_at, frames, nodes[0])
        out = Reconstruction(
            experiment=EXPERIMENT, views=tuple(VIEWS), xyz=xyz, source=source_rows,
            residual_mm=residual, margin_mm=margin * N_ANIMALS,
            n_views_used=slot_rays.min(axis=1).astype(np.int8))
        print(f"  reconstructed in {time.time() - started:.0f}s   "
              f"resolved {out.resolved.mean():.2%}   "
              f"three cameras on every animal "
              f"{(slot_rays[out.resolved] == 3).all(axis=1).mean():.2%}")
        record.param("resolved", float(out.resolved.mean()))

        # A gate derived on *this* reconstruction is a property of it, so
        # comparing two reconstructions through their own gates compares gates
        # as much as methods — the confound `c20` named. Worse here: resolving
        # 99.99% of frames leaves almost no unresolved runs, so the derived
        # `max_gap_frames` collapses to 1 and the linker breaks a track at every
        # missing frame. Holding it at `c19`'s 4.0 mm / 73 frames is the
        # controlled comparison, and it is post-hoc.
        derived = derive_gate(out)
        config = dataclasses.replace(
            derived,
            max_step_mm=derived.max_step_mm if args.gate_mm is None else args.gate_mm,
            max_gap_frames=(derived.max_gap_frames if args.max_gap_frames is None
                            else args.max_gap_frames))
        print(f"  gate derived on this reconstruction: {derived.max_step_mm} mm / "
              f"{derived.max_gap_frames} frames")
        if (args.gate_mm, args.max_gap_frames) != (None, None):
            print(f"  !! held at {config.max_step_mm} mm / {config.max_gap_frames} "
                  f"frames — a POST-HOC choice")
        record.param("gate_is_post_hoc", (args.gate_mm, args.max_gap_frames) != (None, None))
        record.param("gate_derived_mm", derived.max_step_mm)
        record.param("gate_derived_gap_frames", derived.max_gap_frames)
        record.param("max_step_mm", config.max_step_mm)
        record.param("max_gap_frames", config.max_gap_frames)

        label_track = np.where(out.resolved[:, None], np.arange(N_ANIMALS)[None, :],
                               UNTRACKED).astype(np.int32)
        arms = {"C25-gated": METHODS["gated"](out, config), "C25-label": label_track}
        summary = {}
        for name, slot_track in arms.items():
            per_view = propagate_to_views(out, slot_track, starts, counts)
            path = out_dir / f"{name}__{EXPERIMENT}.npz"
            np.savez_compressed(
                path, slot_track=slot_track, xyz=out.xyz.astype(np.float32),
                residual_mm=out.residual_mm.astype(np.float32),
                margin_mm=out.margin_mm.astype(np.float32),
                n_views_used=out.n_views_used, source=out.source,
                slot_rays=slot_rays, state=chosen.astype(np.int32),
                **{f"track__{v}": per_view[v] for v in VIEWS})
            record.output(f"{name}__{EXPERIMENT}", path)
            summary[name] = {"path": str(path),
                             "coverage": coverage(slot_track, frames.size)}
            print(f"  {name:<10} {summary[name]['coverage']}")

        np.save(out_dir / "chosen.npy", chosen)
        record.output("chosen", out_dir / "chosen.npy")
        record.param("arms", summary)
        (out_dir / "c25_summary.json").write_text(
            json.dumps({"experiment": EXPERIMENT, "n_groups": 1, "n_experiments": 1,
                        "groups": [ExperimentId.parse(EXPERIMENT).group],
                        "labels": source, "nodes": list(nodes), "mu": args.mu,
                        "lambda": float(penalty), "frames": [int(lo), int(hi)],
                        "objective": objective, "resolved": float(out.resolved.mean()),
                        "slot_label_changes": int(changed.sum()),
                        "max_step_mm": config.max_step_mm,
                        "states_per_frame_mean": float(sizes.mean()),
                        "arms": summary},
                       indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

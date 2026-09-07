"""Cross-view correspondence solved on more than one node.

`reconstruct` matches detections across cameras using **one point per animal**.
Its input contract, ``uv_by_view: view -> (n_frames, 3, 2)``, has no node axis,
so the whole correspondence rests on a single keypoint — the mouth since `c02`.
Meanwhile the per-camera 2D tracker matches a whole pose. The two halves of the
pipeline were using different amounts of the same detections.

Measured on K_1_2 (2026-08-18), summing the residual over the mouth **and** the
pectoral instead of taking either alone:

| cameras | nodes | runner-up margin p1 | margin < 1 mm |
|---|---|---|---|
| 3 | pectoral | 0.989 mm | 1.01% |
| 3 | mouth | 1.240 mm | 0.67% |
| 3 | **both** | **3.635 mm** | **0.03%** |
| 2 | pectoral | 0.006 mm | 13.04% |
| 2 | **both** | **0.252 mm** | **3.59%** |

The gain is largest exactly where it is needed. On a two-view frame there is no
third camera to contradict a wrong pairing, and one node leaves the correct
correspondence and its runner-up 0.006 mm apart at the first percentile — no
discrimination at all. A second node is a second, independent constraint on the
same frame, and it recovers what the missing camera would have supplied.

The winner changes on 6-11% of frames, so this is not a refinement of an answer
that was already right.

=============================================================================
A MISSING NODE COSTS A NODE, NEVER A CAMERA
=============================================================================

`c03` marks a view complete only if all three **mouths** are finite, and
bottom-up leaves a mouth unplaced in 1.065-1.507% of instances, so one missing
keypoint used to cost a whole camera for that frame — and with it the
three-camera check. `c13` measured what fixing that is worth: two-view frames
fell 20.6% -> 13.9% and the female's swaps halved.

Solving on several nodes must not undo that. A view takes part when its
**anchor** node is present; every other node contributes to the cost only on the
frames where every participating view has it. So a frame missing one mouth is
solved on the pectoral alone — weaker, and honestly so — rather than losing a
camera.

The reported ``xyz`` is the anchor's, because downstream distances, encounters
and gates are all defined on one point and changing which point they mean is a
separate decision. ``residual_mm`` is the mean over the nodes that took part, so
it stays in millimetres and stays comparable with the single-node runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from medaka3d.reconstruct import (
    N_ANIMALS,
    UNRESOLVED,
    Reconstruction,
    hypotheses,
    triangulate,
)


def hypothesis_costs(origins: np.ndarray, directions: np.ndarray):
    """``(cost, combo_xyz, combo_rms, hypothesis, flat)`` for a batch of frames.

    `reconstruct.solve` computes the same cost matrix and then keeps only the
    winner and the runner-up. Summing over nodes needs the whole row, so the
    shared arithmetic lives here rather than being repeated per node.
    """
    n_frames, n_views = origins.shape[0], origins.shape[1]
    if n_views < 2:
        raise ValueError("cross-view correspondence needs at least two views")
    combos = np.asarray(list(product(range(N_ANIMALS), repeat=n_views)), dtype=np.intp)
    frame = np.arange(n_frames)[:, None, None]
    view = np.arange(n_views)[None, None, :]
    combo_xyz, combo_rms = triangulate(
        origins[frame, view, combos[None, :, :]],
        directions[frame, view, combos[None, :, :]],
    )
    powers = N_ANIMALS ** np.arange(n_views - 1, -1, -1)
    hypothesis = hypotheses(n_views)
    flat = (hypothesis * powers).sum(axis=-1)
    cost = combo_rms[:, flat].sum(axis=-1)
    return np.where(np.isfinite(cost), cost, np.inf), combo_xyz, combo_rms, hypothesis, flat


def node_available(
    uv: dict[str, dict[str, np.ndarray]], views: tuple[str, ...], node: str,
) -> np.ndarray:
    """(n_frames,) True where every named view has all three of ``node`` finite."""
    return np.logical_and.reduce(
        [np.isfinite(uv[view][node]).all(axis=(1, 2)) for view in views]
    )


def reconstruct_multinode(
    rig, experiment: str, uv: dict[str, dict[str, np.ndarray]],
    complete: dict[str, np.ndarray], n_frames: int, *,
    anchor: str, nodes: tuple[str, ...], batch: int = 20_000,
) -> tuple[Reconstruction, dict]:
    """Solve the correspondence on ``nodes``, report the ``anchor``'s geometry.

    Args:
        uv: ``view -> node -> (n_frames, 3, 2)`` pixels, laid out in the
            detector's own order within each frame.
        complete: ``view -> (n_frames,)`` True where that view reported exactly
            three detections **with a finite anchor**. Judged on the anchor
            alone, so a missing secondary node never removes a camera.
        nodes: every node that contributes to the cost. Must contain ``anchor``.

    Returns the reconstruction and, per node, how many frames it contributed to
    and how often the multi-node winner differs from the anchor-only winner.
    The second number is what says whether this changed anything.
    """
    if anchor not in nodes:
        raise ValueError(f"anchor {anchor!r} must be among the solved nodes {nodes}")
    views = rig.views
    available = np.stack([complete[v] for v in views], axis=1)
    used = available.sum(axis=1).astype(np.int8)

    xyz = np.full((n_frames, N_ANIMALS, 3), np.nan)
    residual = np.full((n_frames, N_ANIMALS), np.nan)
    margin = np.full(n_frames, np.nan)
    source = np.full((n_frames, N_ANIMALS, len(views)), UNRESOLVED, dtype=np.int8)

    rays = {(v, node): rig.camera(v).rays(uv[v][node]) for v in views for node in nodes}
    tally = {"frames": 0, "changed_vs_anchor_only": 0,
             **{f"node_frames.{node}": 0 for node in nodes}}

    for pattern in {tuple(row) for row in available[used >= 2]}:
        present = [i for i, flag in enumerate(pattern) if flag]
        names = tuple(views[i] for i in present)
        frames = np.flatnonzero((available == np.asarray(pattern)).all(axis=1))
        usable = {node: node_available(uv, names, node) for node in nodes}
        for lo in range(0, frames.size, batch):
            block = frames[lo : lo + batch]
            total = None
            anchor_cost = anchor_xyz = anchor_rms = hypothesis = flat = None
            contributing = np.zeros(block.size)
            for node in nodes:
                origins = np.stack([rays[(views[i], node)][0][block] for i in present], axis=1)
                directions = np.stack(
                    [rays[(views[i], node)][1][block] for i in present], axis=1)
                cost, combo_xyz, combo_rms, hypothesis, flat = hypothesis_costs(
                    origins, directions)
                here = usable[node][block]
                tally[f"node_frames.{node}"] += int(here.sum())
                contributing += here
                # A node that is absent contributes nothing rather than infinity,
                # so one missing keypoint cannot make every hypothesis unusable.
                share = np.where(here[:, None], np.where(np.isfinite(cost), cost, np.inf), 0.0)
                total = share if total is None else total + share
                if node == anchor:
                    anchor_cost, anchor_xyz, anchor_rms = cost, combo_xyz, combo_rms

            pick = np.argmin(total, axis=-1)
            anchor_pick = np.argmin(anchor_cost, axis=-1)
            tally["frames"] += int(block.size)
            tally["changed_vs_anchor_only"] += int((pick != anchor_pick).sum())

            rows = np.arange(block.size)
            ordered = np.sort(total, axis=-1)
            xyz[block] = anchor_xyz[rows[:, None], flat[pick]]
            residual[block] = (
                anchor_rms[rows[:, None], flat[pick]]
                if len(nodes) == 1 else
                np.where(contributing[:, None] > 0,
                         total[rows, pick][:, None] / np.maximum(contributing, 1)[:, None]
                         / N_ANIMALS, np.nan)
            )
            margin[block] = ordered[:, 1] - ordered[:, 0]
            picked = hypothesis[pick]
            for column, view_index in enumerate(present):
                source[block, :, view_index] = picked[:, :, column]

    used = np.where(np.isfinite(xyz).all(axis=(1, 2)), used, 0).astype(np.int8)
    return Reconstruction(
        experiment=experiment, views=views, xyz=xyz, source=source,
        residual_mm=residual, margin_mm=margin, n_views_used=used,
    ), tally


# =============================================================================
# STAGE B — THE CORRESPONDENCE AS A PIECEWISE-CONSTANT SEQUENCE
# =============================================================================
#
# Everything above solves one frame at a time. That is what
# `docs/decisions/2026-08-18-viterbi-correspondence.md` replaces: the
# correspondence over a whole recording is one object, and it is chosen by
# minimising
#
#     J(s) = SUM_t cost_t(s_t)  +  lambda * #{t : s_t != s_{t-1}}
#
# exactly, by dynamic programming. Three earlier attempts to add a temporal term
# used a window and a vote; all three lagged a genuine change by half a window
# and were retired (see the previous decision's Addendum 2). A Viterbi has no
# window and no direction of time, so a change at frame t is paid for by
# evidence on both sides of t at once.
#
# THE STATE MUST MEAN THE SAME THING AT EVERY FRAME, and detection rows do not:
# they are the detector's own ordering within a frame. `c16`'s per-camera 2D
# track labels do — three labels per view, alive for the whole recording, and on
# every frame holding exactly three detections they form a permutation of
# {0,1,2} in all three views (measured on K_1_2: rate 1.000). So the caller
# passes arrays indexed by **label**, and a state is the pair of permutations
# top-label -> side1-label, top-label -> side2-label: 36 states, constant
# meaning. `hypotheses(3)` already enumerates exactly those, view 0 fixed.
#
# The per-frame cost is `reconstruct_multinode`'s, divided by the number of
# contributing nodes. Within a frame that set is the same for all 36 states, so
# the division cannot move the per-frame winner — at lambda = 0 this reproduces
# `reconstruct_multinode` frame for frame, which the tool asserts. What it buys
# is that the cost stays in millimetres whatever a frame supplies, so one lambda
# means the same thing on a one-node frame and a two-node frame.
#
# A frame with only two complete views has an unconstrained third permutation,
# so its 36 costs collapse into 6 blocks of 6 equal values and the DP chooses
# inside the block by persistence. That is the point: 13.9% of K_1_2 is
# two-view, and 43.7% of the frames inside a measured swap interval are. A frame
# with fewer than two complete views costs 0 in every state and carries the
# state across at no charge.


@dataclass(frozen=True, slots=True)
class StateCosts:
    """Per-frame cost of every correspondence state, per node and combined.

    ``cost`` is (n_frames, n_states) in mm — the summed point-to-ray residual of
    the three animals, averaged over the nodes that contributed. ``by_node``
    holds each node's own (n_frames, n_states) before combination and ``usable``
    says where each contributed. ``margin`` is the runner-up gap among the
    states this frame can actually tell apart, which on a two-view frame is 6
    rather than 36 — quoting it over all 36 would report 0 mm on every two-view
    frame and read as total ambiguity.
    """

    cost: np.ndarray
    by_node: dict[str, np.ndarray]
    usable: dict[str, np.ndarray]
    margin: dict[str, np.ndarray]
    contributing: np.ndarray
    n_views_used: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.cost.shape[1])

    def fit_cost(self, node: str | None = None) -> np.ndarray:
        """The objective to minimise: every node, or one node held in isolation.

        A frame the node is missing from costs 0 in every state, which is the
        honest encoding of "no evidence here" and lets the DP carry the state
        across rather than inventing a preference.
        """
        if node is None:
            return self.cost
        return np.where(self.usable[node][:, None], self.by_node[node], 0.0)


def _pattern_blocks(rig, uv, complete, n_frames, *, nodes, batch):
    """Yield one block of frames sharing which views are complete.

    Both passes need the same grouping — pass 1 to build the cost matrix, pass 2
    to rebuild the geometry of the states the sequence chose — and duplicating
    the bookkeeping is how the two would drift apart.
    """
    views = rig.views
    states = hypotheses(len(views))
    available = np.stack([complete[v] for v in views], axis=1)
    n_views = available.sum(axis=1).astype(np.int8)
    rays = {(v, node): rig.camera(v).rays(uv[v][node]) for v in views for node in nodes}

    for pattern in sorted({tuple(row) for row in available[n_views >= 2]}):
        present = [i for i, flag in enumerate(pattern) if flag]
        names = tuple(views[i] for i in present)
        powers = N_ANIMALS ** np.arange(len(present) - 1, -1, -1)
        # Flat index, per state and slot, into the product of one detection per
        # present view. Built from labels rather than from `hypotheses(n_present)`
        # so a pattern missing `top_cam` is still indexed by the same 36 states
        # as every other frame.
        flat = (states[:, :, present] * powers).sum(axis=-1)
        seen: dict[tuple, int] = {}
        for s in range(states.shape[0]):
            seen.setdefault(tuple(sorted(flat[s].tolist())), s)
        distinct = np.asarray(sorted(seen.values()), dtype=np.intp)

        frames = np.flatnonzero((available == np.asarray(pattern)).all(axis=1))
        usable = {node: node_available(uv, names, node) for node in nodes}
        for lo in range(0, frames.size, batch):
            block = frames[lo : lo + batch]
            rms, xyz = {}, {}
            for node in nodes:
                origins = np.stack(
                    [rays[(views[i], node)][0][block] for i in present], axis=1)
                directions = np.stack(
                    [rays[(views[i], node)][1][block] for i in present], axis=1)
                _, combo_xyz, combo_rms, _, _ = hypothesis_costs(origins, directions)
                rms[node], xyz[node] = combo_rms, combo_xyz
            yield block, present, states, flat, distinct, usable, rms, xyz


def _runner_up(cost: np.ndarray) -> np.ndarray:
    """(n_frames,) second-smallest minus smallest, over distinguishable states."""
    if cost.shape[1] < 2:
        return np.full(cost.shape[0], np.inf)
    ordered = np.partition(cost, 1, axis=1)
    with np.errstate(invalid="ignore"):
        # inf - inf where no hypothesis triangulated at all: NaN is the honest
        # answer for "there was nothing to be more confident than".
        return ordered[:, 1] - ordered[:, 0]


def state_costs(
    rig, uv: dict[str, dict[str, np.ndarray]], complete: dict[str, np.ndarray],
    n_frames: int, *, nodes: tuple[str, ...], batch: int = 20_000,
) -> StateCosts:
    """Cost of every correspondence state at every frame.

    Args:
        uv: ``view -> node -> (n_frames, 3, 2)`` pixels **indexed by 2D track
            label**, not by detection row. Passing detector rows here would make
            a state mean something different at every frame, which is the one
            error this design cannot survive.
    """
    n_states = hypotheses(len(rig.views)).shape[0]
    by_node = {node: np.zeros((n_frames, n_states), np.float32) for node in nodes}
    usable = {node: np.zeros(n_frames, bool) for node in nodes}
    margin = {node: np.full(n_frames, np.nan, np.float32) for node in (*nodes, "total")}
    contributing = np.zeros(n_frames, np.int8)
    cost = np.zeros((n_frames, n_states), np.float32)
    used = np.stack([complete[v] for v in rig.views], axis=1).sum(axis=1).astype(np.int8)

    for block, _, _, flat, distinct, node_ok, rms, _ in _pattern_blocks(
        rig, uv, complete, n_frames, nodes=nodes, batch=batch
    ):
        total = np.zeros((block.size, n_states))
        count = np.zeros(block.size)
        for node in nodes:
            per_state = rms[node][:, flat].sum(axis=-1)
            per_state = np.where(np.isfinite(per_state), per_state, np.inf)
            here = node_ok[node][block]
            by_node[node][block] = per_state
            usable[node][block] = here
            margin[node][block] = _runner_up(per_state[:, distinct])
            total += np.where(here[:, None], per_state, 0.0)
            count += here
        contributing[block] = count
        cost[block] = total / np.maximum(count, 1)[:, None]
        margin["total"][block] = _runner_up(cost[block][:, distinct])

    return StateCosts(cost=cost, by_node=by_node, usable=usable, margin=margin,
                      contributing=contributing, n_views_used=used)


def viterbi_sweep(
    cost: np.ndarray, lambdas: np.ndarray, held_out: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Solve the sequence for many penalties at once, without storing the paths.

    Returns ``changes``, ``fit_cost`` and — when ``held_out`` is given —
    ``held_cost``, the sum of a *different* cost along the path this one chose.
    That is the cross-validation criterion in the decision document, and it is
    why the sweep carries accumulators rather than back-pointers: the quantities
    wanted per lambda are scalars, and (n_lambda, n_frames) back-pointer arrays
    for a 348,600-frame recording are not worth materialising to recover three
    numbers.

    The transition penalty is the same for every pair of distinct states, so the
    recursion is O(T x S), not O(T x S^2)::

        V_t(s) = cost_t(s) + min( V_{t-1}(s),  lambda + min_s' V_{t-1}(s') )

    Ties go to **staying**, so at lambda -> 0 this returns the per-frame winner
    with the fewest changes consistent with it.
    """
    lam = np.asarray(lambdas, float)[:, None]
    rows = np.arange(lam.shape[0])

    value = np.broadcast_to(cost[0], (lam.shape[0], cost.shape[1])).astype(float).copy()
    changes = np.zeros(value.shape, np.int64)
    held = (np.broadcast_to(held_out[0], value.shape).astype(float).copy()
            if held_out is not None else None)

    for frame in range(1, cost.shape[0]):
        best = value.argmin(axis=1)
        switch = value[rows, best][:, None] + lam
        take = switch < value
        if held is not None:
            held = np.where(take, held[rows, best][:, None], held) + held_out[frame]
        changes = np.where(take, (changes[rows, best] + 1)[:, None], changes)
        value = np.where(take, switch, value) + cost[frame]

    winner = value.argmin(axis=1)
    out = {"lambda": np.asarray(lambdas, float),
           "changes": changes[rows, winner].astype(np.int64),
           "fit_cost": value[rows, winner]}
    if held is not None:
        out["held_cost"] = held[rows, winner]
    return out


def viterbi_path(cost: np.ndarray, penalty: float) -> np.ndarray:
    """(n_frames,) chosen state per frame, for one penalty. Same recursion, backtracked."""
    n_frames, n_states = cost.shape
    take = np.zeros((n_frames, n_states), bool)
    came_from = np.zeros(n_frames, np.int16)
    value = cost[0].astype(float).copy()

    for frame in range(1, n_frames):
        best = int(value.argmin())
        came_from[frame] = best
        switch = value[best] + penalty
        take[frame] = switch < value
        value = np.where(take[frame], switch, value) + cost[frame]

    path = np.empty(n_frames, np.int16)
    path[-1] = int(value.argmin())
    for frame in range(n_frames - 1, 0, -1):
        path[frame - 1] = came_from[frame] if take[frame, path[frame]] else path[frame]
    return path


def reconstruct_from_states(
    rig, experiment: str, uv: dict[str, dict[str, np.ndarray]],
    complete: dict[str, np.ndarray], row_of_label: dict[str, np.ndarray],
    chosen: np.ndarray, n_frames: int, *,
    anchor: str, nodes: tuple[str, ...], batch: int = 20_000,
) -> Reconstruction:
    """Rebuild the geometry of the correspondence the sequence chose.

    ``row_of_label`` is ``view -> (n_frames, 3)`` giving the detector's own row
    offset within the frame for each 2D track label. It exists so that ``source``
    still records rows: `c09` inverts ``source`` to find which slot holds the
    human's fish, and handing it labels instead would silently pair one animal's
    3D position with another animal's pixels — the trap that cost the previous
    session the most time.

    ``margin_mm`` is left to the caller, because the runner-up gap is a property
    of the frame's geometry and must be measured over the states the frame can
    tell apart rather than over all 36.
    """
    views = rig.views
    xyz = np.full((n_frames, N_ANIMALS, 3), np.nan)
    residual = np.full((n_frames, N_ANIMALS), np.nan)
    source = np.full((n_frames, N_ANIMALS, len(views)), UNRESOLVED, dtype=np.int8)
    used = np.stack([complete[v] for v in views], axis=1).sum(axis=1).astype(np.int8)

    for block, present, states, flat, _, node_ok, rms, combo_xyz in _pattern_blocks(
        rig, uv, complete, n_frames, nodes=nodes, batch=batch
    ):
        rows = np.arange(block.size)
        pick = flat[chosen[block]]  # (block, n_animals) flat combo index per slot
        xyz[block] = combo_xyz[anchor][rows[:, None], pick]
        total = np.zeros((block.size, N_ANIMALS))
        count = np.zeros(block.size)
        for node in nodes:
            here = node_ok[node][block]
            total += np.where(here[:, None], rms[node][rows[:, None], pick], 0.0)
            count += here
        residual[block] = total / np.maximum(count, 1)[:, None]
        labels = states[chosen[block]]  # (block, n_animals, n_views)
        for view_index in present:
            source[block, :, view_index] = row_of_label[views[view_index]][
                block[:, None], labels[:, :, view_index]]

    used = np.where(np.isfinite(xyz).all(axis=(1, 2)), used, 0).astype(np.int8)
    return Reconstruction(
        experiment=experiment, views=views, xyz=xyz, source=source,
        residual_mm=residual, margin_mm=np.full(n_frames, np.nan),
        n_views_used=used,
    )


# =============================================================================
# STAGE C — THE CARDINALITY LIMITER REMOVED
# =============================================================================
#
# Everything above requires a view to report **exactly three** detections before
# it may take part in a frame. That rule is not a fact about the experiment; it
# is a convenience that let the solver index detections by row. Measured on
# K_1_2's bottom-up run, it costs:
#
#     instances per frame   0      1       2         3      4      5+
#     top_cam             115     559  12,841   327,977  6,925    183
#     side1_cam             0   3,770  18,215   321,538  4,972    105
#     side2_cam            71   4,146  14,146   327,423  2,715     99
#
# A frame with four detections is discarded although all three animals are in
# it; a frame with two is discarded although two animals are. Under that rule a
# view is usable on 92.2-94.1% of frames and all three views on 83.7%.
#
# The replacement is already in the pipeline. `c16`'s per-camera 2D tracker
# holds exactly three track labels per view, so **the label, not the row, says
# which detection is which**, and a label is either present in a frame or it is
# not — no counting involved. A fourth detection carries no label and is
# ignored; a frame with two detections supplies two labels and the third is
# absent. Under that rule a view is usable on 93.7-96.1% of frames and all three
# on 87.5%, and the state space does not change, because it was defined on
# labels already.
#
# WHAT A MISSING LABEL COSTS, PRECISELY. Above, a view either takes part in a
# whole frame or not at all. Here participation is per animal: the slot for top
# label *a* is triangulated from whichever views hold its label under the state
# being scored, so one camera losing one animal weakens **that animal's** slot
# by one ray and leaves the other two untouched. This is
# `reconstruct_multinode`'s "a missing node costs a node, never a camera",
# applied along the other axis.
#
# THE COST IS A MEAN OVER THE TERMS THAT EXIST, which is what makes the number
# comparable between a frame with nine rays and a frame with seven, and so lets
# one lambda mean one thing throughout. On a frame where every view holds every
# label it is exactly `state_costs`' cost divided by N_ANIMALS — a constant
# factor, so the per-frame winner, the margin ratio and therefore kappa are all
# unchanged, and `c20` asserts that equality rather than trusting it.
#
# THE ONE BIAS, STATED. When two views are each missing a label, different
# states leave different numbers of slots with fewer than two rays, so the
# divisor is not constant across states within such a frame and a state that
# explains less can be charged less. It happens on 3.44% of K_1_2's frames on the
# two-node cost, every one of which the old rule discarded outright, and the
# alternative — forbidding states that explain fewer detections — is worse: when
# both side cameras lose the *same* animal, the correct state is exactly the one
# that leaves a slot short. Weak evidence entering a sequence model that already
# weighs persistence against it is the lesser error, and `terms` reports how much
# of the recording it applies to.
#
# WHAT THIS BOUGHT AND WHAT IT COST, measured by `c20` on K_1_2 and recorded in
# `docs/decisions/2026-08-19-uncapped-cardinality.md`: the reconstruction is
# strictly better (resolved 97.53% -> 98.55%, three cameras 83.68% -> 87.50%),
# and the *sequence* is not. The admitted frames triple the number of
# correspondence changes the Viterbi keeps, 973 -> 3,090, and that is not an
# artefact of lambda's rescaling — at c19's own effective penalty it is still
# 2,939. Nothing here is promoted on that evidence.


def label_rays(rig, uv: dict[str, dict[str, np.ndarray]], nodes: tuple[str, ...]):
    """``(view, node) -> (origins, directions, present)``, indexed by 2D label.

    ``uv`` is ``view -> node -> (n_frames, N_ANIMALS, 2)`` where row *l* holds
    the detection carrying 2D track label *l*, or NaN if this frame has no such
    detection. Presence is read off the rays rather than off the pixels, so a
    pixel that back-projects to nothing — total internal reflection, a ray
    running away from the interface — counts as absent instead of as a NaN that
    poisons a triangulation.
    """
    out = {}
    for view in rig.views:
        camera = rig.camera(view)
        for node in nodes:
            origins, directions = camera.rays(uv[view][node])
            present = np.isfinite(origins).all(axis=-1) & np.isfinite(directions).all(axis=-1)
            out[(view, node)] = (origins, directions, present)
    return out


def _combo_table(n_views: int):
    """``(combos, states, flat)``: the label triples, and each state's index into them."""
    combos = np.asarray(list(product(range(N_ANIMALS), repeat=n_views)), dtype=np.intp)
    states = hypotheses(n_views)
    powers = N_ANIMALS ** np.arange(n_views - 1, -1, -1)
    return combos, states, (states * powers).sum(axis=-1)


def _view_stack(rays, views, node, block, combos, item):
    """(n_frames, n_combos, n_views, ...) one ray per view for every label triple."""
    return np.stack(
        [rays[(view, node)][item][block][:, combos[:, i]] for i, view in enumerate(views)],
        axis=2,
    )


def _distinguishable(present: np.ndarray, states: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """(n_frames, n_states) key, equal exactly for states this frame cannot separate.

    A state is a set of three slots and a slot is a triple of labels, but a label
    the frame does not hold contributes nothing, so two states differing only in
    where they send an absent label produce identical geometry. Encoding each
    slot by its *present* labels and comparing the sorted set is what stops the
    runner-up margin being reported as 0 mm on every such frame, which would read
    as total ambiguity when the frame is simply short of evidence.
    """
    n_views = states.shape[2]
    code = np.zeros((present.shape[0], states.shape[0], N_ANIMALS), np.int64)
    for view in range(n_views):
        label = states[None, :, :, view]  # (1, n_states, n_animals)
        held = present[:, flat, view]  # (n_frames, n_states, n_animals)
        code += np.where(held, label + 1, 0) * (4**view)
    code = np.sort(code, axis=2)
    return (code * (64 ** np.arange(N_ANIMALS))).sum(axis=2)


def _margin_against_other(cost: np.ndarray, key: np.ndarray) -> np.ndarray:
    """(n_frames,) cheapest state that is not the winner's equal, minus the winner."""
    rows = np.arange(cost.shape[0])
    best = cost.argmin(axis=1)
    floor = cost[rows, best]
    other = np.where(key != key[rows, best][:, None], cost, np.inf).min(axis=1)
    with np.errstate(invalid="ignore"):
        gap = other - floor
    return np.where(np.isfinite(gap), gap, np.nan)


@dataclass(frozen=True, slots=True)
class LabelCosts:
    """`StateCosts` without the cardinality rule: the costs, and what each rests on.

    ``cost`` is (n_frames, n_states) in mm — the mean point-to-ray residual over
    the (node, slot) terms the frame supports. ``terms`` is (n_frames, n_states):
    how many terms that was, which is the honest measure of what a frame's number
    is worth. ``slot_rays`` is (n_frames, n_states, n_animals) for the **anchor**
    node — how many cameras saw each animal under each state, and the only thing
    that decides whether a slot has a 3D position at all.
    """

    cost: np.ndarray
    by_node: dict[str, np.ndarray]
    usable: dict[str, np.ndarray]
    margin: dict[str, np.ndarray]
    terms: np.ndarray
    slot_rays: np.ndarray
    n_views_used: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.cost.shape[1])

    def fit_cost(self, node: str | None = None) -> np.ndarray:
        """The objective to minimise: every node, or one node held in isolation.

        A frame the node is missing from costs 0 in every state, which is the
        honest encoding of "no evidence here" and lets the sequence carry the
        state across rather than inventing a preference.
        """
        if node is None:
            return self.cost
        return np.where(self.usable[node][:, None], self.by_node[node], 0.0)

    def resolvable(self) -> np.ndarray:
        """(n_frames,) True where some state gives all three animals two cameras."""
        return (self.slot_rays >= 2).all(axis=2).any(axis=1)


def label_costs(
    rig, uv: dict[str, dict[str, np.ndarray]], n_frames: int, *,
    anchor: str, nodes: tuple[str, ...], batch: int = 20_000,
) -> LabelCosts:
    """Cost of every correspondence state at every frame, with no cardinality rule.

    Args:
        uv: ``view -> node -> (n_frames, N_ANIMALS, 2)`` pixels indexed by 2D
            track label, NaN where the frame holds no detection with that label.
            Passing detector rows here would make a state mean something
            different at every frame, which is the one error this design cannot
            survive.

    There is no grouping by which views are complete, because completeness is no
    longer a property of a view: every frame is solved over the same label
    triples, and a triple simply rests on fewer rays where a label is absent.
    """
    if anchor not in nodes:
        raise ValueError(f"anchor {anchor!r} must be among the solved nodes {nodes}")
    views = rig.views
    combos, states, flat = _combo_table(len(views))
    n_states = states.shape[0]
    rays = label_rays(rig, uv, nodes)

    cost = np.zeros((n_frames, n_states), np.float32)
    terms = np.zeros((n_frames, n_states), np.int8)
    slot_rays = np.zeros((n_frames, n_states, N_ANIMALS), np.int8)
    by_node = {node: np.zeros((n_frames, n_states), np.float32) for node in nodes}
    usable = {node: np.zeros(n_frames, bool) for node in nodes}
    margin = {node: np.full(n_frames, np.nan, np.float32) for node in (*nodes, "total")}
    n_views_used = np.zeros(n_frames, np.int8)

    for lo in range(0, n_frames, batch):
        block = slice(lo, min(lo + batch, n_frames))
        size = block.stop - block.start
        total = np.zeros((size, n_states))
        count = np.zeros((size, n_states), np.int16)
        anchor_present = None

        for node in nodes:
            origins = _view_stack(rays, views, node, block, combos, 0)
            directions = _view_stack(rays, views, node, block, combos, 1)
            held = _view_stack(rays, views, node, block, combos, 2)
            _, combo_rms = triangulate(origins, directions, valid=held)
            live = (held.sum(axis=2) >= 2) & np.isfinite(combo_rms)  # (size, n_combos)

            per_slot = np.where(live[:, flat], combo_rms[:, flat], 0.0)
            node_terms = live[:, flat].sum(axis=2)
            total += per_slot.sum(axis=2)
            count += node_terms.astype(np.int16)

            by_node[node][block] = per_slot.sum(axis=2) / np.maximum(node_terms, 1)
            usable[node][block] = node_terms.max(axis=1) > 0
            margin[node][block] = _margin_against_other(
                by_node[node][block], _distinguishable(held, states, flat))
            if node == anchor:
                anchor_present = held
                slot_rays[block] = held.sum(axis=2)[:, flat].astype(np.int8)
                n_views_used[block] = np.stack(
                    [rays[(view, node)][2][block].any(axis=1) for view in views], axis=1
                ).sum(axis=1).astype(np.int8)

        cost[block] = total / np.maximum(count, 1)
        terms[block] = np.minimum(count, 127).astype(np.int8)
        margin["total"][block] = _margin_against_other(
            cost[block], _distinguishable(anchor_present, states, flat))

    return LabelCosts(cost=cost, by_node=by_node, usable=usable, margin=margin,
                      terms=terms, slot_rays=slot_rays, n_views_used=n_views_used)


def reconstruct_from_label_states(
    rig, experiment: str, uv: dict[str, dict[str, np.ndarray]],
    row_of_label: dict[str, np.ndarray], chosen: np.ndarray, n_frames: int, *,
    anchor: str, nodes: tuple[str, ...], batch: int = 20_000,
) -> tuple[Reconstruction, np.ndarray]:
    """Rebuild the geometry of the correspondence the sequence chose.

    ``row_of_label`` is ``view -> (n_frames, N_ANIMALS)`` giving the detector's
    own row offset within the frame for each 2D track label, or ``UNRESOLVED``
    where the frame holds no such detection. ``source`` records rows because
    `c09` inverts it to find which slot holds the human's fish, and handing it
    labels instead would silently pair one animal's 3D position with another
    animal's pixels.

    Returns the reconstruction and ``slot_rays`` (n_frames, n_animals): how many
    cameras saw each animal. ``n_views_used`` is the **minimum** over the three
    slots, so ``resolved`` keeps the meaning it has everywhere else — a resolved
    frame is one where every animal has a 3D position — and a frame where one
    animal was seen by a single camera is not resolved, however good the other
    two are.
    """
    views = rig.views
    combos, states, flat = _combo_table(len(views))
    rays = label_rays(rig, uv, nodes)

    xyz = np.full((n_frames, N_ANIMALS, 3), np.nan)
    residual = np.full((n_frames, N_ANIMALS), np.nan)
    source = np.full((n_frames, N_ANIMALS, len(views)), UNRESOLVED, dtype=np.int8)
    slot_rays = np.zeros((n_frames, N_ANIMALS), np.int8)

    for lo in range(0, n_frames, batch):
        block = slice(lo, min(lo + batch, n_frames))
        size = block.stop - block.start
        pick = flat[chosen[block]]  # (size, n_animals) combo index per slot
        rows = np.arange(size)[:, None]
        total = np.zeros((size, N_ANIMALS))
        count = np.zeros((size, N_ANIMALS))

        for node in nodes:
            origins = _view_stack(rays, views, node, block, combos, 0)
            directions = _view_stack(rays, views, node, block, combos, 1)
            held = _view_stack(rays, views, node, block, combos, 2)
            combo_xyz, combo_rms = triangulate(origins, directions, valid=held)
            live = (held.sum(axis=2) >= 2) & np.isfinite(combo_rms)
            total += np.where(live[rows, pick], combo_rms[rows, pick], 0.0)
            count += live[rows, pick]
            if node == anchor:
                xyz[block] = combo_xyz[rows, pick]
                slot_rays[block] = held.sum(axis=2)[rows, pick].astype(np.int8)

        residual[block] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        labels = states[chosen[block]]  # (size, n_animals, n_views)
        for index, view in enumerate(views):
            source[block, :, index] = row_of_label[view][block][rows, labels[:, :, index]]

    used = np.where(np.isfinite(xyz).all(axis=2), slot_rays, 0).min(axis=1).astype(np.int8)
    return Reconstruction(
        experiment=experiment, views=views, xyz=xyz, source=source,
        residual_mm=residual, margin_mm=np.full(n_frames, np.nan),
        n_views_used=used,
    ), slot_rays

"""Exact six-state first-order identity optimization on externally fixed geometries.

P[id] = raw geometry slot. No shape fitting, color extraction, probability,
last-observed memory, acceleration, or review clearing occurs in this module.
Exact refers to the enumerated first-order objective; arithmetic is float64 and
ties use an explicit numerical tolerance. A large conditional gap is not truth.
"""

from dataclasses import dataclass
from itertools import permutations

import numpy as np

PERMUTATIONS = np.array(list(permutations(range(3))), dtype=np.int64)
PERMUTATIONS.flags.writeable = False


@dataclass(frozen=True)
class IdentityDPResult:
    status: str  # optimal / infeasible
    frame_idx: np.ndarray
    permutations: np.ndarray  # state,id -> raw geometry slot
    path_state: np.ndarray  # T; -1 if no feasible path
    permutation_by_id: np.ndarray  # T,id; -1 if no feasible path
    path_cost: float
    forward_cost: np.ndarray  # includes current unary and initial constraint
    backward_cost: np.ndarray  # excludes current unary
    min_marginal_cost: np.ndarray  # T,6 = forward + backward
    min_marginal_delta: np.ndarray  # T,6; NaN if the entire chain is infeasible
    optimal_state_mask: np.ndarray  # T,6; within numerical tie tolerance
    id_slot_min_marginal_delta: np.ndarray  # T,id,slot
    state_alternative_gap: np.ndarray  # cheapest whole path differing at this time
    identity_alternative_gap: np.ndarray  # T,id; cheapest different assignment
    state_alternative_status: np.ndarray
    identity_alternative_status: np.ndarray
    global_alternative_gap: float  # distinct state sequence, not another endpoint only
    global_alternative_status: str
    numerical_tie_tolerance: float
    roundoff_clamped_deltas: int
    initial_state: int | None
    common_cost_offset: float
    numerical_cost_span: float


@dataclass(frozen=True)
class AdjacentMotionCosts:
    frame_idx: np.ndarray
    transitions: np.ndarray  # T-1,previous state,current state
    supported_pairs: np.ndarray  # T-1,previous state,current state,id
    supported_pair_count: np.ndarray
    squared_distance_mm2: np.ndarray  # NaN for unobserved pairs, never latent evidence
    scale_mm: float
    weight: float
    denominator_fish: int = 3
    unsupported_pairs_cost_zero: bool = True
    missing_connection_cost_evasion_possible: bool = True


def _frames(frame_idx):
    values = np.asarray(frame_idx)
    if (
        values.ndim != 1 or not len(values) or values.dtype.kind not in "iu"
        or np.any(values < 0) or np.any(values > np.iinfo(np.int64).max)
    ):
        raise ValueError("native frame_idx must be nonempty nonnegative int64-compatible integers")
    values = values.astype(np.int64, copy=True)
    if np.any(np.diff(values) != 1):
        raise ValueError("every consecutive native frame must be present")
    return values


def _scalar(value, name, *, positive=False):
    if (
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
        or not np.isrealobj(value) or not np.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        constraint = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {constraint} scalar")
    return float(value)


def _costs(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array of shape {shape}")
    value = value.astype(np.float64, copy=True)
    if np.isnan(value).any() or np.isneginf(value).any():
        raise ValueError(f"{name} permits finite values or +inf constraints, not NaN/-inf")
    return value


def _alternative_status(gaps, tolerance):
    result = np.full(np.shape(gaps), "finite_gap", dtype="U32")
    result[np.isposinf(gaps)] = "no_feasible_alternative"
    result[np.isfinite(gaps) & (gaps <= tolerance)] = "tie_within_tolerance"
    return result


def _center_cost_blocks(cost, axes):
    """Remove a shared finite minimum from each time block; +inf stays forbidden."""
    minimum = cost.min(axis=axes)
    minimum = np.where(np.isfinite(minimum), minimum, 0)
    offset_shape = (len(cost),) + (1,) * len(axes)
    centered = cost - minimum.reshape(offset_shape)
    maximum = np.where(np.isfinite(centered), centered, 0).max(axis=axes)
    return centered, minimum, maximum


def solve_identity_dp(
    frame_idx, unary, transitions, *, initial_state=0, tie_atol=1e-10, tie_rtol=1e-12
):
    """Minimize sum_t unary[t,p_t] + sum_t transitions[t-1,p_(t-1),p_t].

    A fixed initial state is the only optional boundary; None leaves it free.
    Permutation changes themselves carry no extra penalty. Input +inf may forbid
    a state/edge, but these constraints must be recorded by the caller.

    The second distinct path's gap is min_(t,p != best_t) min_marginal[t,p] - C*.
    Thus two paths that swap and swap back are competitors even if endpoints
    agree. Infinity means no feasible alternative was compared, not confidence.
    An infeasible chain has -1 path indices and NaN gaps, never a lexical winner.
    Backtracking chooses deterministic argmins; all tied states remain exposed.
    Shared per-time unary/edge offsets are removed internally and restored to
    absolute F/B/cost outputs. Deltas use the centered costs to avoid subtracting
    large irrelevant constants. Tie tolerance scales with the sum of within-block
    finite cost spans, not absolute path cost, so common offsets do not change
    the numerical tie policy. Floating inputs already rounded by their producer
    cannot recover lost distinctions. Raw F+B may differ by roundoff from M.
    """
    frames = _frames(frame_idx)
    n = len(frames)
    if initial_state is not None and (
        isinstance(initial_state, (bool, np.bool_))
        or not isinstance(initial_state, (int, np.integer)) or not 0 <= initial_state < 6
    ):
        raise ValueError("initial_state must be None or an integer state index 0..5")
    initial_state = None if initial_state is None else int(initial_state)
    atol = _scalar(tie_atol, "tie_atol")
    rtol = _scalar(tie_rtol, "tie_rtol")
    emission = _costs(unary, (n, 6), "unary")
    edges = _costs(transitions, (n - 1, 6, 6), "transitions")
    with np.errstate(over="raise", invalid="raise"):
        emission, unary_offset, unary_span = _center_cost_blocks(emission, (1,))
        edges, edge_offset, edge_span = _center_cost_blocks(edges, (1, 2))
        prefix_offset = np.cumsum(unary_offset + np.r_[0., edge_offset])
        total_offset = float(prefix_offset[-1])
        cost_span = float(unary_span.sum() + edge_span.sum())
        tolerance = float(atol + np.float64(rtol) * cost_span)
    forward = np.full((n, 6), np.inf)
    backward = np.full((n, 6), np.inf)
    parent = np.full((n, 6), -1, np.int64)
    forward[0] = emission[0]
    if initial_state is not None:
        forward[0, np.arange(6) != initial_state] = np.inf
    # Overflow cannot silently turn a feasible finite path into an impossible one.
    with np.errstate(over="raise", invalid="raise"):
        for t in range(1, n):
            candidates = forward[t - 1, :, None] + edges[t - 1]
            parent[t] = np.argmin(candidates, axis=0)
            forward[t] = emission[t] + np.min(candidates, axis=0)
        backward[-1] = 0
        for t in range(n - 2, -1, -1):
            backward[t] = np.min(
                edges[t] + emission[t + 1, None] + backward[t + 1, None], axis=1
            )
        marginal = forward + backward
    centered_best = float(forward[-1].min())
    with np.errstate(over="raise", invalid="raise"):
        path_cost = float(np.float64(centered_best) + total_offset)
        absolute_forward = forward + prefix_offset[:, None]
        absolute_backward = backward + (total_offset - prefix_offset[:, None])
        absolute_marginal = marginal + total_offset
    path = np.full(n, -1, np.int64)
    ids = np.full((n, 3), -1, np.int64)
    if not np.isfinite(centered_best):
        return IdentityDPResult(
            "infeasible", frames, PERMUTATIONS.copy(), path, ids, path_cost,
            absolute_forward, absolute_backward, absolute_marginal,
            np.full((n, 6), np.nan), np.zeros((n, 6), bool),
            np.full((n, 3, 3), np.nan), np.full(n, np.nan), np.full((n, 3), np.nan),
            np.full(n, "infeasible", dtype="U32"),
            np.full((n, 3), "infeasible", dtype="U32"), np.nan, "infeasible",
            tolerance, 0, initial_state, total_offset, cost_span,
        )
    path[-1] = np.argmin(forward[-1])
    for t in range(n - 1, 0, -1):
        path[t - 1] = parent[t, path[t]]
    ids[:] = PERMUTATIONS[path]
    delta = np.full((n, 6), np.inf)
    finite = np.isfinite(marginal)
    delta[finite] = marginal[finite] - centered_best
    if np.any(delta < -tolerance):
        raise FloatingPointError("min-marginal below optimal cost beyond numerical tolerance")
    negative = delta < 0
    clamped = int(negative.sum())
    delta[negative] = 0.0
    # Also require the chosen path to pass the same forward/backward consistency check.
    if np.any(delta[np.arange(n), path] > tolerance):
        raise FloatingPointError("forward/backward path costs disagree beyond numerical tolerance")
    alternative = delta.copy()
    alternative[np.arange(n), path] = np.inf
    state_gap = alternative.min(axis=1)
    id_delta = np.empty((n, 3, 3))
    for identity, slot in np.ndindex(3, 3):
        id_delta[:, identity, slot] = delta[:, PERMUTATIONS[:, identity] == slot].min(axis=1)
    id_alternative = id_delta.copy()
    id_alternative[np.arange(n)[:, None], np.arange(3)[None], ids] = np.inf
    id_gap = id_alternative.min(axis=2)
    state_status = _alternative_status(state_gap, tolerance)
    id_status = _alternative_status(id_gap, tolerance)
    if initial_state is not None:
        state_status[0] = "fixed_initial_constraint"
        id_status[0] = "fixed_initial_constraint"
    global_gap = float(state_gap.min())
    global_status = str(_alternative_status(np.array(global_gap), tolerance))
    return IdentityDPResult(
        "optimal", frames, PERMUTATIONS.copy(), path, ids, path_cost,
        absolute_forward, absolute_backward, absolute_marginal,
        delta, delta <= tolerance, id_delta, state_gap, id_gap,
        state_status, id_status, global_gap, global_status, tolerance, clamped,
        initial_state, total_offset, cost_span,
    )


def build_adjacent_motion_cost(
    frame_idx, pectoral_xyz, direct_pectoral, *, scale_mm=0.8, weight=0.2
):
    """Direct adjacent pectoral displacement only, with a fixed three-fish denominator.

    C_t(p,q) = weight/3 * sum_id held_id * ||x_t[q(id)]-x_(t-1)[p(id)]||^2/scale_mm^2.

    A pair is held only if both adjacent endpoints are directly supported. No
    latent or last-observed positions substitute for a missing endpoint. Missing
    pairs cost zero and can let a path avoid a costly supported link. Fixed /3
    avoids a variable-denominator artifact but does NOT remove this cost-evasion
    limitation. Pair flags/counts must accompany the costs, and low-cost missing
    paths must not be interpreted as identity confirmation. These scale/weight
    values are engineering defaults, not calibrated motion likelihoods.
    """
    frames = _frames(frame_idx)
    xyz, observed = np.asarray(pectoral_xyz), np.asarray(direct_pectoral)
    if xyz.shape != (len(frames), 3, 3) or xyz.dtype.kind not in "fiu":
        raise ValueError("pectoral_xyz must be a real T,3,3 array")
    if observed.shape != (len(frames), 3) or observed.dtype != bool:
        raise ValueError("direct_pectoral must be boolean T,3")
    xyz = xyz.astype(np.float64, copy=True)
    if not np.isfinite(xyz[observed]).all():
        raise ValueError("directly observed pectoral points must be finite")
    scale = _scalar(scale_mm, "scale_mm", positive=True)
    factor = _scalar(weight, "weight")
    safe = np.where(observed[..., None], xyz, 0)
    held = observed[:-1, PERMUTATIONS][:, :, None] & observed[1:, PERMUTATIONS][:, None]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        displacement = safe[1:, PERMUTATIONS][:, None] - safe[:-1, PERMUTATIONS][:, :, None]
        distance_sq = np.square(displacement).sum(axis=-1)
        cost = factor / 3 * (np.where(held, distance_sq, 0) / scale**2).sum(axis=-1)
    return AdjacentMotionCosts(
        frames, cost, held, held.sum(axis=-1), np.where(held, distance_sq, np.nan), scale, factor
    )

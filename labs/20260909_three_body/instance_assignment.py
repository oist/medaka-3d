"""Exact local SLEAP row coherence controls, without changing the ray objective.

Rows denote detector instances within one view and one frame, never persistent
animal IDs. The coherent control gives a body at most one row per view and a
row at most one body. Bad or absent nodes within that row may remain missing.
The relaxed control assigns rows separately for each anatomical node. Both
use identical half-ray errors, weights, outlier costs and normalization.
Neither control certifies a detector's grouping or an animal's identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class AssignmentResult:
    plan: np.ndarray  # fish, view, node; detection rows or -1
    cost: float  # same 9 * sum(node_weights) denominator as the frozen fitter
    cost_by_view: np.ndarray  # unnormalized numerator, three views
    optimal_view_counts: tuple[int, int, int]


def half_ray_costs(pose, rays, ray_sigma_mm=.7):
    """Pairwise robust error using the frozen fitter's float32 ray cache.

Return three arrays (fish, detection row, node). Invalid rays have infinite
cost. Forward half-rays, rather than infinite lines, match the fit objective.
No assignment, missing penalty or fitted support flag is included here.
"""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (3, 3, 3) or not np.isfinite(pose).all():
        raise ValueError('pose must be finite three-fish, three-node geometry')
    if not np.isfinite(ray_sigma_mm) or ray_sigma_mm <= 0:
        raise ValueError('ray sigma must be finite and positive')
    if len(rays.origins) != 3 or len(rays.directions) != 3:
        raise ValueError('three explicit views are required')
    output = []
    for raw_o, raw_d in zip(rays.origins, rays.directions, strict=True):
        origin = np.asarray(raw_o, dtype=np.float32).astype(np.float64)
        direction = np.asarray(raw_d, dtype=np.float32).astype(np.float64)
        if (origin.ndim != 3 or origin.shape[1:] != (3, 3)
                or direction.shape != origin.shape):
            raise ValueError('rays must retain detection-row and anatomical-node axes')
        valid = (np.isfinite(origin).all(-1) & np.isfinite(direction).all(-1)
                 & (np.linalg.norm(direction, axis=-1) > 0))
        delta = pose[:, None] - np.where(valid[..., None], origin, 0)[None]
        safe_direction = np.where(valid[..., None], direction, 0)[None]
        depth = np.maximum((delta * safe_direction).sum(-1, keepdims=True), 0)
        error = np.sum((delta - depth * safe_direction) ** 2, axis=-1) / ray_sigma_mm**2
        robust = np.where(error <= 9., error, 6. * np.sqrt(error + 1e-12) - 9.)
        output.append(np.where(valid[None], robust, np.inf))
    return tuple(output)


def _exclusive_minimum(cost, missing):
    """Enumerate exact three-body partial injections; no candidate truncation.

Only rows strictly cheaper than the missing alternative can improve a body.
Ties are counted using exact arithmetic equality. The returned representative
is lexicographic (including -1); it is not a row-permutation-invariant identity
decision when exact ties exist. Missing-equivalent row choices are canonicalized
to null and therefore are not counted as separate optimal matchings.
"""
    choices = [(-1, *np.flatnonzero(row < missing).tolist()) for row in cost]
    best, selected, ties = np.inf, None, 0
    for indexes in product(*choices):
        used = [i for i in indexes if i >= 0]
        if len(set(used)) != len(used):
            continue
        value = sum(missing if j < 0 else cost[f, j] for f, j in enumerate(indexes))
        if value < best:
            best, selected, ties = float(value), indexes, 1
        elif value == best:
            ties += 1
    return best, np.asarray(selected, dtype=np.int32), ties


def assign(costs, *, coherent, missing_cost=9., node_weights=(1., 1., .35)):
    """Minimize one of two comparable objectives over all three views.

The only feasible-set difference is whole-row versus nodewise exclusivity.
Within a coherent row each node is independently retained only when its cost
is below missing_cost. A detector-merged or misgrouped row may thus hurt this
control: SLEAP grouping is a hypothesis here, not anatomical ground truth.
"""
    weights = np.asarray(node_weights, dtype=np.float64)
    if (weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights <= 0)
            or not np.isfinite(missing_cost) or missing_cost <= 0):
        raise ValueError('positive finite three-node weights and missing cost required')
    with np.errstate(over='ignore'):
        normalization = 9. * weights.sum()
        maximum_numerator = missing_cost * normalization
    if not np.isfinite(normalization) or not np.isfinite(maximum_numerator):
        raise ValueError('aggregated node weights and missing costs must remain finite')
    if type(coherent) is not bool or len(costs) != 3:
        raise ValueError('explicit coherent boolean and three cost arrays required')
    plan = np.full((3, 3, 3), -1, dtype=np.int32)
    view_cost, view_ties = [], []
    for view, raw in enumerate(costs):
        cost = np.asarray(raw, dtype=np.float64)
        if (cost.ndim != 3 or cost.shape[0] != 3 or cost.shape[2] != 3
                or np.isnan(cost).any() or np.any(cost < 0)):
            raise ValueError('cost must have fish,row,node axes and nonnegative values or +inf')
        if coherent:
            per_row = (np.minimum(cost, missing_cost) * weights).sum(-1)
            per_row = np.where(np.any(cost < missing_cost, axis=-1), per_row, np.inf)
            _, rows, ties = _exclusive_minimum(per_row, missing_cost * weights.sum())
            for fish, row in enumerate(rows):
                if row >= 0:
                    keep = cost[fish, row] < missing_cost
                    plan[fish, view, keep] = row
        else:
            ties = 1
            for node in range(3):
                _, rows, node_ties = _exclusive_minimum(
                    cost[:, :, node] * weights[node], missing_cost * weights[node])
                plan[:, view, node] = rows
                ties *= node_ties
        # Recompute with one common summation order for both feasible sets.
        selected_cost = np.full((3, 3), missing_cost, dtype=np.float64)
        for fish, node in np.argwhere(plan[:, view] >= 0):
            selected_cost[fish, node] = cost[fish, plan[fish, view, node], node]
        view_cost.append(float((selected_cost * weights).sum()))
        view_ties.append(int(ties))
    numerator = np.asarray(view_cost, dtype=np.float64)
    return AssignmentResult(plan, float(numerator.sum() / normalization),
                            numerator, tuple(view_ties))

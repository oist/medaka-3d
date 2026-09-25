"""A box-constrained geometric seed from two or three current observation rays.

This minimizes the sum of squared distances to the *infinite lines* underlying
the rays. Forward-ray flags are diagnostics, not additional constraints. There
is no anatomy, pixel reprojection, support threshold, identity or temporal prior
in this helper. A successful solve is not an accepted fish observation.

Directions are normalized on owned float64 copies. Numerically rank-deficient
line systems are rejected even when the box could select an arbitrary minimizer.
The convex BVLS solution is checked independently for feasibility and KKT
optimality. No clipped unconstrained solution is used as the returned answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

VERSION = "bounded_ray_seed_v1"
SOLVER_TOLERANCE = 1e-12
MAX_ITERATIONS = 100


@dataclass(frozen=True)
class BoundedRaySeed:
    """All arrays are owned; distances/positions use the caller's world units (mm).

    active_bounds is -1 at a lower bound, +1 at an upper bound, otherwise 0.
    forward_ray includes zero depth. kkt_violation is for the gradient of
    half the squared-error sum; its tolerance includes explicit roundoff slack.
    condition_number describes stacked A, not the normal matrix A.T @ A (whose
    condition number is its square). Rank three alone does not imply accuracy.
    """

    point_xyz: np.ndarray
    ray_distances_mm: np.ndarray
    rms_mm: float
    ray_depth_mm: np.ndarray
    forward_ray: np.ndarray
    active_bounds: np.ndarray
    unit_directions: np.ndarray
    sum_squared_distance_mm2: float
    unconstrained_point_xyz: np.ndarray
    rank: int
    rank_threshold: float
    singular_values: np.ndarray
    condition_number: float
    gradient: np.ndarray
    kkt_violation: float
    kkt_tolerance: float
    solver_optimality: float
    solver_status: int
    solver_iterations: int
    solver_message: str
    success: bool = True


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.dtype.hasobject:
        raise ValueError(f"{name} must contain real numeric values, not bool/object/complex")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite and representable as float64")
    return result


def solve_bounded_ray_seed(origins, directions, lower, upper):
    """Return the unique numerically full-rank box-constrained line-distance seed.

    origins/directions must both be (2,3) or (3,3). Each nonzero direction is
    normalized; positive rescaling therefore leaves the objective unchanged.
    lower/upper must be finite (3,) arrays with lower < upper in every coordinate.
    We solve in coordinates relative to lower to avoid an unnecessary global
    origin offset in the linear system. SciPy BVLS uses dense exact least squares.

    Invalid, degenerate or unrepresentable input raises ValueError. Solver failure
    or a failed independent feasibility/optimality check raises RuntimeError.
    No caller array is mutated, and returned arrays do not alias caller arrays.
    """
    origins = _real_array(origins, "origins")
    directions = _real_array(directions, "directions")
    lower = _real_array(lower, "lower")
    upper = _real_array(upper, "upper")
    if origins.shape not in ((2, 3), (3, 3)) or directions.shape != origins.shape:
        raise ValueError("origins and directions must have matching (2,3) or (3,3) shapes")
    if lower.shape != (3,) or upper.shape != (3,) or not np.all(lower < upper):
        raise ValueError("finite box bounds must have shape (3,) and strictly lower < upper")
    scale = np.max(np.abs(directions), axis=1)
    if np.any(scale == 0):
        raise ValueError("every ray direction must be nonzero")
    unit = directions / scale[:, None]
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            span = upper - lower
            local_origins = origins - lower
            projectors = np.eye(3)[None] - unit[:, :, None] * unit[:, None, :]
            matrix = projectors.reshape(-1, 3)
            target = np.einsum("rij,rj->ri", projectors, local_origins).reshape(-1)
            if not np.isfinite(target).all() or not np.isfinite(span).all():
                raise ValueError("ray system exceeds finite float64 coordinates")
            singular = np.linalg.svd(matrix, compute_uv=False)
            rank_threshold = float(max(matrix.shape) * np.finfo(float).eps * singular[0])
            rank = int(np.count_nonzero(singular > rank_threshold))
            if rank != 3:
                raise ValueError(
                    "ray system is numerically rank deficient; a unique seed is required"
                )
            condition = float(singular[0] / singular[-1])
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("ray system cannot be represented or factorized safely") from error

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = lsq_linear(
                matrix,
                target,
                bounds=(np.zeros(3), span),
                method="bvls",
                lsq_solver="exact",
                tol=SOLVER_TOLERANCE,
                max_iter=MAX_ITERATIONS,
            )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise RuntimeError("bounded ray least-squares solver failed numerically") from error
    if not result.success or result.status not in (1, 2, 3):
        raise RuntimeError(f"bounded ray solver did not converge: {result.message}")
    if (
        np.shape(result.x) != (3,)
        or not np.isfinite(result.x).all()
        or not np.isfinite(result.optimality)
        or not np.isfinite(result.cost)
    ):
        raise RuntimeError("bounded ray solver returned nonfinite or malformed output")
    if np.any(result.x < 0) or np.any(result.x > span):
        raise RuntimeError("bounded ray solver returned an infeasible point")
    if (
        np.shape(result.active_mask) != (3,)
        or not np.isin(result.active_mask, (-1, 0, 1)).all()
        or np.any((result.active_mask == -1) & (result.x != 0))
        or np.any((result.active_mask == 1) & (result.x != span))
    ):
        raise RuntimeError("bounded ray solver returned inconsistent active bounds")

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            point = lower + result.x
            # Restore exact world coordinates at solver-active bounds. This is
            # the bound solution, not coordinate clipping of a free solution.
            point = np.where(result.active_mask == -1, lower, point)
            point = np.where(result.active_mask == 1, upper, point)
            if not np.isfinite(point).all() or np.any(point < lower) or np.any(point > upper):
                raise RuntimeError("reconstructed world point is infeasible")
            active = np.where(point == lower, -1, np.where(point == upper, 1, 0)).astype(np.int8)
            local_point = point - lower
            residual = matrix @ local_point - target
            gradient = matrix.T @ residual
            if not np.isfinite(gradient).all() or not np.isfinite(residual).all():
                raise RuntimeError("bounded ray KKT diagnostics are not finite")
            violation = np.where(active == 0, np.abs(gradient), gradient * active)
            kkt = float(max(0.0, np.max(violation)))
            roundoff_scale = max(
                1.0,
                float(
                    np.linalg.norm(matrix, 1)
                    * (
                        np.linalg.norm(matrix, np.inf) * np.linalg.norm(local_point, np.inf)
                        + np.linalg.norm(target, np.inf)
                    )
                ),
            )
            if not np.isfinite(roundoff_scale):
                raise RuntimeError("bounded ray optimality tolerance is not finite")
            kkt_tolerance = max(SOLVER_TOLERANCE, 64 * np.finfo(float).eps * roundoff_scale)
            if kkt > kkt_tolerance:
                raise RuntimeError("bounded ray solution failed independent KKT optimality")
            delta = point[None] - origins
            depth = np.sum(delta * unit, axis=1)
            perpendicular = delta - depth[:, None] * unit
            distance = np.linalg.norm(perpendicular, axis=1)
            squared = float(np.sum(distance**2))
            rms = float(np.sqrt(squared / len(origins)))
            unconstrained = np.asarray(result.unbounded_sol[0], dtype=np.float64) + lower
            if (
                not np.isfinite(unconstrained).all()
                or not np.isfinite(depth).all()
                or not np.isfinite(distance).all()
                or not np.isfinite(squared)
            ):
                raise RuntimeError("bounded ray geometric diagnostics are not finite")
    except FloatingPointError as error:
        raise RuntimeError("bounded ray output diagnostics overflowed") from error
    return BoundedRaySeed(
        point.copy(),
        distance.copy(),
        rms,
        depth.copy(),
        (depth >= 0).copy(),
        active.copy(),
        unit.copy(),
        squared,
        unconstrained.copy(),
        rank,
        rank_threshold,
        singular.copy(),
        condition,
        gradient.copy(),
        kkt,
        kkt_tolerance,
        float(result.optimality),
        int(result.status),
        int(result.nit),
        str(result.message),
    )

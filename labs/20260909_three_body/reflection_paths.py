"""Finite, single-wall reflection hypotheses for already refracted water rays.

This CPU prototype fixes the NOMINAL water box at [0,192]^2 x [0,186] mm,
the opposite inner wall at y=192, and its outer glass face at y=196. These
planes are hypotheses in the existing calibration gauge, not surveyed floor or
wall positions. There is no fitted plane and no engineering z=-6 extension.
Only an unambiguous first exit through the interior of y=192 can reflect.

The inner branch reflects in water. The outer branch transmits into glass,
reflects once at glass/air, and returns through glass/water. Each water branch
ends at its next nominal box exit. Aperture edges are explicitly rejected for
the reflecting wall; a subsequent corner exit still bounds a valid segment.

Weights are ideal, incoherent, lossless ray-power fractions for incident light
unpolarized IN WATER. Parallel-interface s/p powers are propagated separately,
then averaged. Camera-entry polarization/transmission, absorption, interference,
surface waves, multiple bounces, lighting and fish radiance are not modelled.
Multiplying a heuristic Gaussian silhouette by these weights is not radiometry.
The Fresnel convention is R_s,p = |r_s,p|^2 and T_s,p = 1-R_s,p; see MIT
6.974, section 2.2.3, Fresnel Reflection:
https://ocw.mit.edu/courses/6-974-fundamentals-of-photonics-quantum-electronics-spring-2006/
fafcb9e36b7749f4eeb099d666bdf70a_electr_wav_inter.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Wall(IntEnum):
    NONE = -1
    X0 = 0
    X192 = 1
    Y0 = 2
    Y192 = 3
    Z0 = 4
    Z186 = 5


class Reason(IntEnum):
    OK = 0
    NONFINITE_RAY = 1
    NONUNIT_DIRECTION = 2
    ORIGIN_OUTSIDE_BOX = 3
    NO_POSITIVE_EXIT = 4
    FIRST_HIT_EDGE = 5
    OTHER_FIRST_WALL = 6
    WATER_GLASS_TIR = 7
    GLASS_GRAZING = 8
    OUTER_APERTURE_OUTSIDE = 9
    OUTER_APERTURE_EDGE = 10
    RETURN_APERTURE_OUTSIDE = 11
    RETURN_APERTURE_EDGE = 12
    GLASS_WATER_TIR = 13


BOUNDARY_ATOL_MM = 1e-9
UNIT_ATOL = 1e-6  # Accepted calibrated cache roundoff is normalized, never fitted.
_UPPER = np.array([192.0, 192.0, 186.0])


@dataclass(frozen=True)
class RayBranch:
    origins: np.ndarray  # (...,3), water; NaN for invalid branches
    directions: np.ndarray  # (...,3), unit; NaN for invalid branches
    valid: np.ndarray  # (...,), bool; weight=0 can still be geometrically valid
    reason: np.ndarray  # (...,), uint8 Reason
    weight: np.ndarray  # (...,), ideal ray-power fraction; invalid -> 0
    max_distance_mm: np.ndarray  # (...,), finite next water-box exit, invalid -> NaN
    next_wall: np.ndarray  # (...,), int8 Wall; first numeric wall at a tied exit
    next_wall_mask: np.ndarray  # (...,), uint8 bitset, (1 << Wall)


@dataclass(frozen=True)
class ReflectionPaths:
    direct_valid: np.ndarray  # (...,), valid finite water segment, including corner exits
    first_hit_mm: np.ndarray  # (...,3); invalid direct segment -> NaN
    first_distance_mm: np.ndarray  # (...,); geometric distance, not optical path length
    first_wall: np.ndarray  # (...,), int8 Wall; use mask/reason at edges
    first_wall_mask: np.ndarray  # (...,), uint8, includes parallel-on-boundary faces
    first_reason: np.ndarray  # (...,), uint8; OK/EDGE can both have direct_valid=True
    inner: RayBranch
    outer: RayBranch
    outer_hit_mm: np.ndarray  # (...,3), diagnostic glass-face hit, even if outside aperture
    return_point_mm: np.ndarray  # (...,3), diagnostic water-face hit after outer reflection
    glass_out_directions: np.ndarray  # (...,3), outgoing transmitted unit ray
    glass_return_directions: np.ndarray  # (...,3), after outer reflection, before refraction
    reflectance_water_glass: np.ndarray  # (...,), unpolarized average; not a path product
    reflectance_glass_air: np.ndarray
    reflectance_glass_water: np.ndarray
    outer_tir: np.ndarray  # (...,), glass/air TIR at a valid outer aperture hit


def _indices(*values: float) -> tuple[float, ...]:
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
            raise ValueError("refractive indices must be finite positive real scalars")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("refractive indices must be finite positive real scalars") from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError("refractive indices must be finite positive real scalars")
        result.append(value)
    # Avoid silently overflowing ratios/squares for nonsensical numerical inputs.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ratios = np.array(result)[:, None] / np.array(result)[None, :]
        squared = ratios * ratios
    if not np.isfinite(squared).all() or (squared == 0).any():
        raise ValueError("refractive-index ratios must have representable positive squares")
    return tuple(result)


def _exit_box(origins: np.ndarray, directions: np.ndarray):
    """Select the first forward EXIT, retaining immediate outward exits as invalid.

    Flat (R,3) arrays; inputs can contain NaN padding. Distances are computed using
    unit directions supplied by the caller. Entry surfaces pointing inward do not
    count as a new exit. Tied or grazing-on-edge hits are not reflection candidates.
    """
    count = len(origins)
    finite = np.isfinite(origins).all(axis=1) & np.isfinite(directions).all(axis=1)
    norms = np.hypot.reduce(directions, axis=1)
    unit = np.abs(norms - 1) <= UNIT_ATOL
    # Calibrated side-camera entry y can be -4e-16 after plane intersection.
    # This fixed numerical band changes classification only, never coordinates.
    inside = ((origins >= -BOUNDARY_ATOL_MM)
              & (origins <= _UPPER + BOUNDARY_ATOL_MM)).all(axis=1)
    reason = np.zeros(count, dtype=np.uint8)
    reason[~inside] = Reason.ORIGIN_OUTSIDE_BOX
    reason[~unit] = Reason.NONUNIT_DIRECTION
    reason[~finite] = Reason.NONFINITE_RAY
    admissible = finite & unit & inside
    # Normalize accepted roundoff only, so max_distance is in geometric mm.
    safe_directions = np.zeros_like(directions)
    np.divide(directions, norms[:, None], out=safe_directions,
              where=admissible[:, None])
    destinations = np.where(safe_directions > 0, _UPPER, 0.0)
    distances = np.full_like(origins, np.inf)
    np.divide(destinations - origins, safe_directions, out=distances,
              where=(safe_directions != 0) & admissible[:, None])
    axis = np.argmin(distances, axis=1)
    minimum = distances[np.arange(count), axis]
    positive = np.isfinite(minimum) & (minimum > BOUNDARY_ATOL_MM)
    reason[admissible & ~positive] = Reason.NO_POSITIVE_EXIT
    valid = admissible & positive
    hit = np.full_like(origins, np.nan)
    hit[valid] = origins[valid] + minimum[valid, None] * safe_directions[valid]
    faces = 2 * axis + (safe_directions[np.arange(count), axis] > 0)
    wall = np.where(valid, faces, Wall.NONE).astype(np.int8)
    mask = np.zeros(count, dtype=np.uint8)
    # Include a parallel boundary face too: a wall/floor intersection has no
    # unique physical reflection normal even if its plane-distance was infinite.
    for coordinate in range(3):
        for upper in (0, 1):
            plane = _UPPER[coordinate] if upper else 0.0
            touches = valid & (np.abs(hit[:, coordinate] - plane) <= BOUNDARY_ATOL_MM)
            mask[touches] |= np.uint8(1 << (2 * coordinate + upper))
    edge = valid & ((mask & (mask - np.uint8(1))) != 0)
    reason[edge] = Reason.FIRST_HIT_EDGE
    return (valid, hit, np.where(valid, minimum, np.nan), wall, mask, reason,
            safe_directions)


def _fresnel(cosine: np.ndarray, n_from: float, n_to: float):
    """Return power Rs, Rp, transmitted cosine, and TIR; NaN padding is preserved."""
    # Scaling indices equally leaves Fresnel unchanged and avoids n*cos overflow.
    scale = max(n_from, n_to)
    a, b = n_from / scale, n_to / scale
    ci = np.clip(cosine, 0.0, 1.0)
    sin2_t = (n_from / n_to) ** 2 * np.maximum(1 - ci * ci, 0.0)
    tir = sin2_t > 1.0
    ct = np.sqrt(np.maximum(1 - sin2_t, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = ((a * ci - b * ct) / (a * ci + b * ct)) ** 2
        rp = ((b * ci - a * ct) / (b * ci + a * ct)) ** 2
    if n_from == n_to:
        rs = np.where(np.isfinite(ci), 0.0, np.nan)
        rp = rs.copy()
    return np.where(tir, 1.0, rs), np.where(tir, 1.0, rp), ct, tir


def _aperture(points: np.ndarray):
    xz = points[:, [0, 2]]
    upper = _UPPER[[0, 2]]
    inside = ((xz >= -BOUNDARY_ATOL_MM) & (xz <= upper + BOUNDARY_ATOL_MM)).all(axis=1)
    edge = inside & ((xz <= BOUNDARY_ATOL_MM)
                     | (upper - xz <= BOUNDARY_ATOL_MM)).any(axis=1)
    return inside, edge


def _branch(origins, directions, eligible, reason, weight, shape):
    valid, _, distance, wall, mask, _, unit_directions = _exit_box(origins, directions)
    # Boundary points were solved analytically, not clipped into the box.
    branch_valid = eligible & valid
    reason = reason.copy()
    reason[eligible & ~valid] = Reason.NO_POSITIVE_EXIT
    return RayBranch(
        np.where(branch_valid[:, None], origins, np.nan).reshape((*shape, 3)).copy(),
        np.where(branch_valid[:, None], unit_directions, np.nan).reshape((*shape, 3)).copy(),
        branch_valid.reshape(shape).copy(), reason.reshape(shape),
        np.where(branch_valid, weight, 0.0).reshape(shape),
        np.where(branch_valid, distance, np.nan).reshape(shape),
        np.where(branch_valid, wall, Wall.NONE).astype(np.int8).reshape(shape),
        np.where(branch_valid, mask, 0).astype(np.uint8).reshape(shape),
    )


def trace_y192_reflections(
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    index_water: float,
    index_glass: float,
    index_air: float,
) -> ReflectionPaths:
    """Trace two fixed reflection branches from matching (...,3) water rays.

    Indices must come from the actual camera/rig. NaN/zero/nonunit/outside rays
    get per-ray invalid reasons; malformed shapes/dtypes/indices raise ValueError.
    Unit length within 1e-6 is normalized to remove cache roundoff; no direction,
    origin or plane is fitted. Boundary classification has a fixed 1e-9 mm
    roundoff band, without changing origin coordinates. All output arrays own
    storage independent of inputs.
    A y192 hit must have positive length and interior x/z aperture. Valid direct
    rays hitting another wall remain available through ``first_distance_mm``.
    """
    nw, ng, na = _indices(index_water, index_glass, index_air)
    o, d = np.asarray(origins), np.asarray(directions)
    if (o.shape != d.shape or o.ndim < 1 or o.shape[-1] != 3
            or o.dtype.kind not in "fiu" or d.dtype.kind not in "fiu"):
        raise ValueError("origins and directions must be real arrays of matching (...,3) shape")
    shape = o.shape[:-1]
    o, d = o.astype(np.float64).reshape(-1, 3), d.astype(np.float64).reshape(-1, 3)
    direct, hit, distance, wall, mask, first_reason, d = _exit_box(o, d)
    reason = first_reason.copy()
    reason[(first_reason == Reason.OK) & (wall != Wall.Y192)] = Reason.OTHER_FIRST_WALL
    eligible = reason == Reason.OK
    # Set exact analytical plane coordinate; x/z are neither moved nor clipped.
    hit[eligible, 1] = 192.0
    ci = np.where(eligible, d[:, 1], np.nan)
    rs_wg, rp_wg, ct_glass, inner_tir = _fresnel(ci, nw, ng)
    reflected = d.copy()
    reflected[:, 1] *= -1
    inner = _branch(hit, reflected, eligible, reason, .5 * (rs_wg + rp_wg), shape)

    outer_reason = reason.copy()
    outer_reason[eligible & inner_tir] = Reason.WATER_GLASS_TIR
    transmit = eligible & ~inner_tir
    grazing = transmit & (ct_glass <= 0)
    outer_reason[grazing] = Reason.GLASS_GRAZING
    transmit &= ~grazing
    glass_out = np.full_like(o, np.nan)
    glass_out[transmit] = (nw / ng) * d[transmit]
    glass_out[transmit, 1] = ct_glass[transmit]
    outer_hit = np.full_like(o, np.nan)
    outer_hit[transmit] = hit[transmit] + (4 / ct_glass[transmit, None]) * glass_out[transmit]
    outer_hit[transmit, 1] = 196.0
    outer_inside, outer_edge = _aperture(outer_hit)
    outer_reason[transmit & ~outer_inside] = Reason.OUTER_APERTURE_OUTSIDE
    outer_reason[transmit & outer_edge] = Reason.OUTER_APERTURE_EDGE
    at_outer = transmit & outer_inside & ~outer_edge
    rs_ga, rp_ga, _, outer_tir = _fresnel(np.where(at_outer, ct_glass, np.nan), ng, na)
    glass_return = np.full_like(o, np.nan)
    glass_return[at_outer] = glass_out[at_outer]
    glass_return[at_outer, 1] *= -1
    returned = np.full_like(o, np.nan)
    returned[at_outer] = (outer_hit[at_outer]
                          + (4 / ct_glass[at_outer, None]) * glass_return[at_outer])
    returned[at_outer, 1] = 192.0
    return_inside, return_edge = _aperture(returned)
    outer_reason[at_outer & ~return_inside] = Reason.RETURN_APERTURE_OUTSIDE
    outer_reason[at_outer & return_edge] = Reason.RETURN_APERTURE_EDGE
    at_return = at_outer & return_inside & ~return_edge
    rs_gw, rp_gw, ct_water, return_tir = _fresnel(
        np.where(at_return, ct_glass, np.nan), ng, nw)
    outer_reason[at_return & return_tir] = Reason.GLASS_WATER_TIR
    reenter = at_return & ~return_tir
    water_return = np.full_like(o, np.nan)
    water_return[reenter] = (ng / nw) * glass_return[reenter]
    water_return[reenter, 1] = -ct_water[reenter]
    # Parallel walls preserve incidence-plane s/p bases. Average only at the end.
    outer_weight = .5 * ((1 - rs_wg) * rs_ga * (1 - rs_gw)
                         + (1 - rp_wg) * rp_ga * (1 - rp_gw))
    outer = _branch(returned, water_return, reenter, outer_reason, outer_weight, shape)
    return ReflectionPaths(
        direct.reshape(shape), hit.reshape((*shape, 3)), distance.reshape(shape),
        wall.reshape(shape), mask.reshape(shape), first_reason.reshape(shape), inner, outer,
        outer_hit.reshape((*shape, 3)), returned.reshape((*shape, 3)),
        glass_out.reshape((*shape, 3)), glass_return.reshape((*shape, 3)),
        (.5 * (rs_wg + rp_wg)).reshape(shape), (.5 * (rs_ga + rp_ga)).reshape(shape),
        (.5 * (rs_gw + rp_gw)).reshape(shape), (at_outer & outer_tir).reshape(shape),
    )

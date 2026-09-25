"""Saved point lineage to fixed-geometry identity costs; pure NumPy, no fitting.

The identity axis belongs to initial query groups. Only the evaluated raw body
axis changes under a permutation. Evidence mass, invalid-ray denominator, and
all raw geometry remain fixed. These engineering costs are not probabilities.
"""

from __future__ import annotations

import numpy as np
from identity_dp import PERMUTATIONS

SCHEMA = "body_point_fixed_geometry_identity_cost_v1"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def identity_costs(ownership, ray_distance_sq, point_weight, ray_valid, *, coefficient=3.0):
    """Return all initial-ID to raw-body costs and six unaries for every anchor.

    ownership/q: T,V,Q,raw-body; weights: T,A,V,Q,initial-ID;
    valid: T,V,Q. V and both body axes are exactly three. Q may be zero.
    The caller supplies the saved CT/Original slice and canonical packet weights.
    All inputs are read-only. t0 is evaluated for diagnostics, but its unary is
    zero because the frozen DP uses the explicit P012 initial constraint.
    """
    own, distance, weights, valid = map(
        np.asarray, (ownership, ray_distance_sq, point_weight, ray_valid)
    )
    _require(
        own.ndim == 4
        and own.shape[0] > 0
        and own.shape[1] == own.shape[3] == 3
        and distance.shape == own.shape
        and own.dtype == distance.dtype == np.float64,
        "float64 T,3,Q,3 saved ownership and distance required",
    )
    n, _, query_count, _ = own.shape
    _require(
        weights.ndim == 5
        and weights.shape[0] == n
        and weights.shape[1] > 0
        and weights.shape[2:] == (3, query_count, 3)
        and weights.dtype == np.float64,
        "float64 T,A,3,Q,3 initial-identity weights required",
    )
    _require(
        valid.shape == (n, 3, query_count) and valid.dtype == np.bool_,
        "boolean T,3,Q ray validity required",
    )
    _require(
        np.isfinite(weights).all()
        and (weights >= 0).all()
        and ((weights > 0).sum(axis=-1) <= 1).all(),
        "finite nonnegative exclusive point mass required; no upper-one restriction",
    )
    _require(
        isinstance(coefficient, (int, float, np.integer, np.floating))
        and not isinstance(coefficient, (bool, np.bool_))
        and np.isfinite(coefficient)
        and coefficient >= 0,
        "finite nonnegative scalar coefficient required",
    )
    good = np.broadcast_to(valid[..., None], own.shape)
    _require(
        np.isfinite(own[good]).all()
        and np.isfinite(distance[good]).all()
        and (distance[good] >= 0).all(),
        "all valid raw-body costs must be finite, with nonnegative q, even at zero mass",
    )
    # Neutralize invalid raw values before arithmetic; never renormalize mass.
    safe_own = np.where(good, own, 0.0)
    safe_distance = np.where(good, distance, 0.0)
    effective_mass = weights * valid[:, None, :, :, None]
    with np.errstate(over="raise", invalid="raise"):
        owner = np.einsum(
            "tavqi,tvqr->tavir", effective_mass, (1.0 - safe_own) ** 2, optimize=False
        )
        gap = np.einsum(
            "tavqi,tvqr->tavir",
            effective_mass,
            np.sqrt(np.maximum(safe_distance, 1.0)) - 1.0,
            optimize=False,
        )
        per_view = owner + gap
        cost = per_view.sum(axis=2) / 3.0
        unary = np.stack(
            [cost[:, :, np.arange(3), p].sum(axis=-1) for p in PERMUTATIONS], axis=-1
        ) * np.float64(coefficient)
    _require(np.isfinite(unary).all(), "nonfinite complete permutation cost")
    measured_unary = unary.copy()
    unary[0] = 0.0
    return dict(
        owner_cost_by_view=owner,
        gap_cost_by_view=gap,
        combined_cost_by_view=per_view,
        identity_raw_body_cost=cost,
        measured_unary=measured_unary,
        unary=unary,
        mass_by_view_identity=effective_mass.sum(axis=3),
        permutations=PERMUTATIONS.copy(),
    )


def compare_mapping(frame_idx, baseline_mapping, proposal_mapping, direct_pectoral):
    """Separate ID assignment changes with direct endpoints from latent-only ones.

    An ID whose old or new raw slot is directly supported has a direct-involved
    change. Neither support truth nor identity correctness follows from this flag.
    """
    frames, before, after, direct = map(
        np.asarray, (frame_idx, baseline_mapping, proposal_mapping, direct_pectoral)
    )
    _require(
        frames.ndim == 1
        and frames.dtype == np.int64
        and len(frames) > 0
        and np.all(np.diff(frames) == 1),
        "consecutive int64 native frames required",
    )
    _require(
        before.shape == after.shape == direct.shape == (len(frames), 3)
        and before.dtype == after.dtype == np.int64
        and direct.dtype == np.bool_
        and np.all(np.sort(before, axis=-1) == np.arange(3))
        and np.all(np.sort(after, axis=-1) == np.arange(3)),
        "two complete int64 identity-to-slot permutations and bool support required",
    )
    changed = before != after
    old_direct = np.take_along_axis(direct, before, axis=1)
    new_direct = np.take_along_axis(direct, after, axis=1)
    direct_changed = changed & (old_direct | new_direct)
    latent_changed = changed & ~old_direct & ~new_direct
    return dict(
        changed_by_identity=changed,
        direct_involved_change=direct_changed,
        latent_only_change=latent_changed,
        changed_native=frames[changed.any(axis=1)].copy(),
        direct_involved_native=frames[direct_changed.any(axis=1)].copy(),
        latent_only_native=frames[latent_changed.any(axis=1)].copy(),
    )

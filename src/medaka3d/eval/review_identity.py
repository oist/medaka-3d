"""Conservative identity agreement against a reviewed three-animal reference.

Spatial matching is independent of identity labels. A fixed anchor mapping is
never re-estimated after a crossing. Unknown reference or matching evidence is
kept outside the correctness denominator and inside total expected coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise, permutations

import numpy as np


@dataclass(frozen=True)
class SpatialMatches:
    """Reference-indexed matches, with -1 for an unaccepted correspondence."""

    slots: np.ndarray
    distance_mm: np.ndarray
    margin_mm: np.ndarray
    reason: np.ndarray


def match_positions(
    reference: np.ndarray,
    prediction: np.ndarray,
    reference_valid: np.ndarray,
    prediction_valid: np.ndarray,
    *,
    max_distance_mm: float,
    min_margin_mm: float,
) -> SpatialMatches:
    """Match mutually closest positions with distance and ambiguity gates.

    The margin must clear the threshold against every competing predicted position
    and every other finite reference position. An invalid reference can still make
    a nearby association ambiguous; it cannot receive an accepted match. No
    identity labels influence this calculation, and no match is forced.
    """
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    if ref.ndim != 3 or pred.ndim != 3 or ref.shape != pred.shape or ref.shape[-1] != 3:
        raise ValueError("Expected equal (frames, animals, xyz) arrays")
    if ref.shape[1] < 2:
        raise ValueError("At least two animals are required for an ambiguity margin")
    if max_distance_mm <= 0 or min_margin_mm <= 0:
        raise ValueError("Distance and margin thresholds must be positive")
    ref_mask = np.asarray(reference_valid, bool)
    pred_mask = np.asarray(prediction_valid, bool)
    if ref_mask.shape != ref.shape[:2] or pred_mask.shape != pred.shape[:2]:
        raise ValueError("Validity mask shape mismatch")
    ref_ok = ref_mask & np.isfinite(ref).all(axis=-1)
    pred_ok = pred_mask & np.isfinite(pred).all(axis=-1)
    distance = np.linalg.norm(ref[:, :, None, :] - pred[:, None, :, :], axis=-1)
    distance = np.where(np.isfinite(distance) & pred_ok[:, None, :], distance, np.inf)
    best = np.argmin(distance, axis=2)
    chosen = np.take_along_axis(distance, best[..., None], axis=2)[..., 0]
    second = np.partition(distance, 1, axis=2)[:, :, 1]
    # For each proposed pair, examine competing reference animals at that slot.
    competing = np.full(ref.shape[:2], np.inf)
    for animal in range(ref.shape[1]):
        others = np.arange(ref.shape[1]) != animal
        competing[:, animal] = distance[:, others, :].min(axis=1)[
            np.arange(ref.shape[0]), best[:, animal]
        ]
    with np.errstate(invalid="ignore"):
        margin = np.minimum(second - chosen, competing - chosen)
    accepted = (
        ref_ok & np.isfinite(chosen) & (chosen <= max_distance_mm) & (margin >= min_margin_mm)
    )
    reason = np.full(ref.shape[:2], "reference_unavailable", dtype="U24")
    reason[ref_ok] = "prediction_missing"
    reason[ref_ok & np.isfinite(chosen)] = "distance_mismatch"
    within = ref_ok & np.isfinite(chosen) & (chosen <= max_distance_mm)
    reason[within] = "spatial_ambiguity"
    reason[accepted] = "matched"
    slots = np.where(accepted, best, -1).astype(np.int8)
    # Positive mutual margins imply injectivity; guard that invariant explicitly.
    for a in range(ref.shape[1]):
        for b in range(a):
            if np.any((slots[:, a] >= 0) & (slots[:, a] == slots[:, b])):
                raise AssertionError("A prediction was assigned to two reference animals")
    return SpatialMatches(slots, chosen, margin, reason)


def fixed_anchor_mapping(
    matched_slots: np.ndarray,
    anchor_frames: np.ndarray,
    *,
    minimum_matches: int = 100,
    minimum_agreement: float = 0.95,
) -> dict:
    """Map reference animals to raw slots once, using a predeclared time window."""
    slots = np.asarray(matched_slots)
    selected = slots[np.asarray(anchor_frames, bool)]
    n = slots.shape[1]
    votes = np.array([[np.count_nonzero(selected[:, i] == j) for j in range(n)] for i in range(n)])
    candidates = list(permutations(range(n)))
    totals = [int(votes[np.arange(n), p].sum()) for p in candidates]
    order = np.argsort(totals)[::-1]
    permutation = np.array(candidates[int(order[0])], dtype=int)
    counts = votes.sum(axis=1)
    support = np.divide(
        votes[np.arange(n), permutation], counts, out=np.zeros(n, float), where=counts > 0
    )
    unique = len(totals) < 2 or totals[order[0]] > totals[order[1]]
    valid = (counts >= minimum_matches) & (support >= minimum_agreement) & unique
    return {
        "reference_to_slot": np.where(valid, permutation, -1).tolist(),
        "candidate_permutation": permutation.tolist(),
        "votes": votes.tolist(),
        "matched_observations": counts.tolist(),
        "agreement": support.tolist(),
        "unique_best_permutation": bool(unique),
        "minimum_matches": minimum_matches,
        "minimum_agreement": minimum_agreement,
    }


def identity_status(matches: SpatialMatches, mapping: list[int]) -> np.ndarray:
    """Classify agreement without treating unassessed positions as incorrect."""
    status = matches.reason.copy()
    target = np.asarray(mapping)
    if target.shape != (matches.slots.shape[1],):
        raise ValueError("Expected one fixed anchor slot per reference animal")
    matched = matches.slots >= 0
    status[matched] = "anchor_unresolved"
    supported = matched & (target[None, :] >= 0)
    status[supported] = "wrong_id"
    status[supported & (matches.slots == target[None, :])] = "correct_id"
    return status


def interval_weights(times_s: np.ndarray, start_s: float) -> tuple[np.ndarray, float]:
    """Left-held durations on [start, last acquisition), with no extrapolation.

    Frames before start are excluded even if their hold interval crosses start.
    The resulting leading sub-frame interval stays unprocessed in the denominator.
    The last acquisition receives zero duration because its end is unobserved.
    """
    times = np.asarray(times_s, float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError("Acquisition times must be finite and strictly increasing")
    weights = np.r_[np.diff(times), 0.0]
    weights[times < start_s] = 0.0
    duration = max(0.0, float(times[-1] - max(start_s, times[0])))
    return weights, duration


def contiguous_runs(mask: np.ndarray) -> np.ndarray:
    """Return half-open intervals; false values always break a run."""
    values = np.asarray(mask, bool)
    if values.ndim != 1:
        raise ValueError("Expected one-dimensional mask")
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    return np.column_stack((np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def slot_change_events(slots: np.ndarray, eligible: np.ndarray) -> list[dict]:
    """Report label changes separately for adjacent observations and gaps.

    These are observed correspondence changes, not independently adjudicated
    crossing events. Multiple fish changing at one boundary share an event row.
    """
    slots = np.asarray(slots)
    eligible = np.asarray(eligible, bool)
    events: dict[tuple[int, int], dict] = {}
    for animal in range(slots.shape[1]):
        valid = np.flatnonzero(eligible & (slots[:, animal] >= 0))
        for before, after in pairwise(valid):
            old, new = int(slots[before, animal]), int(slots[after, animal])
            if old == new:
                continue
            key = (int(before), int(after))
            event = events.setdefault(
                key,
                {
                    "before_frame": int(before),
                    "after_frame": int(after),
                    "kind": "adjacent_change" if after == before + 1 else "gap_endpoint_change",
                    "changes": [],
                },
            )
            event["changes"].append(
                {"reference_index": animal, "before_slot": old, "after_slot": new}
            )
    return sorted(events.values(), key=lambda e: (e["after_frame"], e["before_frame"]))

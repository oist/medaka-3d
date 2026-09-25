"""Anonymous, abstention-aware identity checks at three human-selected probes.

Inputs contain projected mouth coordinates, never a tracker or camera model.
The same function can score any method whose permanent slots are expressed in
these arrays. Human continuity assertions are inputs, not inferred from motion.
"""

from __future__ import annotations

from itertools import product

import numpy as np

ANIMALS = ("A", "B", "C")
MATCH_RADIUS_PX = 5.0
CONTINUITY_STATES = ("confirmed", "ambiguous", "unreviewed", "not_visible")


def _unique_matching(projected: np.ndarray, available: np.ndarray, points: np.ndarray) -> list:
    """Keep assignments invariant across all maximum-cardinality matchings.

    Unmatched animals are permitted. One missing mark cannot make an unrelated
    unique match unavailable; two animals competing for one slot both abstain.
    Every marked view must support the *same* slot inside the fixed radius.
    """
    marked = np.isfinite(points).all(axis=-1)
    distance = np.linalg.norm(points[:, None] - projected[None], axis=-1)
    feasible = np.all(
        ~marked[:, None] | (np.isfinite(distance) & (distance <= MATCH_RADIUS_PX)), axis=-1
    ) & marked.any(axis=1)[:, None] & available[None]
    options = [(-1, *map(int, np.flatnonzero(row))) for row in feasible]
    assignments = [
        row for row in product(*options)
        if len([slot for slot in row if slot >= 0]) == len({slot for slot in row if slot >= 0})
    ]
    cardinality = max(sum(slot >= 0 for slot in row) for row in assignments)
    maximum = [row for row in assignments if sum(slot >= 0 for slot in row) == cardinality]
    result = []
    for animal in range(3):
        possible = {row[animal] for row in maximum}
        slot = next(iter(possible)) if len(possible) == 1 and -1 not in possible else None
        if not marked[animal].any():
            reason = "human mouth unavailable"
        elif not feasible[animal].any():
            reason = "no slot within 5 px in every marked view"
        elif slot is None:
            reason = "nonunique one-to-one correspondence"
        else:
            reason = "matched"
        result.append({
            "animal": ANIMALS[animal], "slot": slot, "reason": reason,
            "marked_views": np.flatnonzero(marked[animal]).tolist(),
            "eligible_slots": np.flatnonzero(feasible[animal]).tolist(),
            "error_px_by_view": [
                float(distance[animal, slot, view])
                if slot is not None and marked[animal, view] else None
                for view in range(points.shape[1])
            ],
        })
    return result


def _score_matches(matches: list, anchor_frames: np.ndarray, continuity: np.ndarray) -> dict:
    initial = {animal: matches[0][i]["slot"] for i, animal in enumerate(ANIMALS)}
    intervals, checkpoint_checks, returning = [], [], []
    interval_counts = {"preserved": 0, "changed": 0, "unscored": 0}
    checkpoint_counts = {"same_as_entry": 0, "different_from_entry": 0, "unscored": 0}
    endpoint_counts = {"same_as_entry": 0, "different_from_entry": 0, "unscored": 0}
    for i, animal in enumerate(ANIMALS):
        slots = [row[i]["slot"] for row in matches]
        for edge in range(2):
            reason = "matched confirmed continuity"
            if continuity[i, edge] != "confirmed":
                reason = f"human continuity {continuity[i, edge]}"
            elif np.any(continuity[i, :edge] != "confirmed"):
                reason = "human continuity to entry is unconfirmed"
            elif slots[0] is None:
                reason = "entry correspondence unavailable"
            elif slots[edge] is None or slots[edge + 1] is None:
                reason = "probe correspondence unavailable"
            outcome = "unscored" if reason != "matched confirmed continuity" else (
                "preserved" if slots[edge] == slots[edge + 1] else "changed"
            )
            interval_counts[outcome] += 1
            intervals.append({
                "animal": animal, "edge": ("entry_to_after", "after_to_exit")[edge],
                "start_frame": int(anchor_frames[edge]),
                "stop_frame": int(anchor_frames[edge + 1]),
                "human_status": str(continuity[i, edge]), "outcome": outcome, "reason": reason,
            })
            checkpoint_available = (
                np.all(continuity[i, :edge + 1] == "confirmed")
                and slots[0] is not None and slots[edge + 1] is not None
            )
            checkpoint = "unscored" if not checkpoint_available else (
                "same_as_entry" if slots[edge + 1] == slots[0] else "different_from_entry"
            )
            checkpoint_counts[checkpoint] += 1
            if edge == 1:
                endpoint_counts[checkpoint] += 1
            checkpoint_checks.append({
                "animal": animal, "frame": int(anchor_frames[edge + 1]), "outcome": checkpoint,
            })
        complete = all(slot is not None for slot in slots) and np.all(continuity[i] == "confirmed")
        outcome = "unscored" if not complete else (
            "returning_change_seen" if slots[0] == slots[2] and slots[1] != slots[0]
            else "returning_change_not_seen_at_probes"
        )
        returning.append({"animal": animal, "outcome": outcome})
    return {
        "entry_animal_to_slot": initial,
        "requested_anchors": 9,
        "matched_anchors": sum(row["slot"] is not None for probe in matches for row in probe),
        "anchors": [
            {"frame": int(frame), "animals": row}
            for frame, row in zip(anchor_frames, matches, strict=True)
        ],
        "requested_animal_intervals": 6, "interval_counts": interval_counts,
        "intervals": intervals, "post_entry_checkpoint_counts": checkpoint_counts,
        "post_entry_checkpoint_checks": checkpoint_checks,
        "entry_exit_only_counts": endpoint_counts,
        "returning_change_probes": returning,
        "returning_changes_seen": sum(
            row["outcome"] == "returning_change_seen" for row in returning
        ),
    }


def score_encounter_review(
    frame_idx: np.ndarray,
    projected_mouth: np.ndarray,
    observed_mouth: np.ndarray,
    review_mask: np.ndarray,
    anchor_frames: np.ndarray,
    human_mouth: np.ndarray,
    continuity: np.ndarray,
) -> dict:
    """Score raw and accepted permanent slots against anonymous human marks.

    Shapes: projections (T,3,V,2), observation/review (T,3), human mouths
    (3 probes,3 animals,V,2), continuity (3 animals,2 edges). Missing human
    mouths are [NaN,NaN]. At least one marked view is required per mouth;
    all marked views must agree. Projected mouths must include *all* latent
    slots, including unobserved competitors. Spatial uniqueness is determined
    first; the observation mask then excludes matches lacking measured support.

    Accepted uses the *same raw correspondence*, then withholds reviewed
    slots; deleting a competitor cannot create new identity evidence. Entry
    is the only slot naming reference. Unconfirmed links also break the chain
    back to entry. Every one of the six requested animal edges is retained.
    """
    frames, anchors = np.asarray(frame_idx), np.asarray(anchor_frames)
    projected, points = np.asarray(projected_mouth, float), np.asarray(human_mouth, float)
    observed, flagged = np.asarray(observed_mouth), np.asarray(review_mask)
    states = np.asarray(continuity)
    if (frames.ndim != 1 or not len(frames) or frames.dtype.kind not in "iu"
            or np.any(np.diff(frames) <= 0)):
        raise ValueError("frame_idx must be nonempty, strictly increasing integers")
    if anchors.shape != (3,) or anchors.dtype.kind not in "iu" or np.any(np.diff(anchors) <= 0):
        raise ValueError("anchor_frames must be three strictly increasing integers")
    if projected.ndim != 4 or projected.shape[:2] != (len(frames), 3):
        raise ValueError("projected_mouth must have shape (frames, 3, views, 2)")
    if projected.shape[-1] != 2 or projected.shape[2] < 1:
        raise ValueError("projected_mouth requires at least one view and x/y coordinates")
    if points.shape != (3, 3, projected.shape[2], 2):
        raise ValueError("human_mouth shape must agree with three probes, animals and views")
    for name, mask in (("observed_mouth", observed), ("review_mask", flagged)):
        if mask.shape != (len(frames), 3) or mask.dtype.kind != "b":
            raise ValueError(f"{name} must be a boolean (frames, 3) array")
    if np.isinf(projected).any() or np.isinf(points).any():
        raise ValueError("coordinates may contain NaN for missing points, never infinity")
    if np.any(np.isnan(points).any(axis=-1) != np.isnan(points).all(axis=-1)):
        raise ValueError("missing human points must have both coordinates NaN")
    if states.shape != (3, 2) or not np.isin(states, CONTINUITY_STATES).all():
        raise ValueError("continuity must contain allowed human states for all six edges")
    for animal, edge in zip(*np.where(states == "confirmed"), strict=True):
        if not all(np.isfinite(points[p, animal]).all(axis=-1).any() for p in (edge, edge + 1)):
            raise ValueError("confirmed continuity requires mouth marks at both probes")
    positions = {int(frame): i for i, frame in enumerate(frames)}
    raw, accepted = [], []
    for frame in anchors:
        index = positions.get(int(frame))
        probe = len(raw)
        row = _unique_matching(
            np.full(projected.shape[1:], np.nan) if index is None else projected[index],
            np.ones(3, dtype=bool), points[probe],
        )
        if index is None:
            for match in row:
                match["reason"] = "candidate frame unavailable"
        else:
            for match in row:
                if match["slot"] is not None and not observed[index, match["slot"]]:
                    match.update(slot=None, reason="mouth lacks declared measured support")
        raw.append(row)
        held = []
        for match in row:
            copied = dict(match)
            if match["slot"] is not None and flagged[index, match["slot"]]:
                copied.update(slot=None, reason="identity requires review")
            held.append(copied)
        accepted.append(held)
    return {
        "protocol": "anonymous_three_probe_review_v1", "match_radius_px": MATCH_RADIUS_PX,
        "human_confirmed_edges": int(np.sum(states == "confirmed")),
        "human_edge_states": {state: int(np.sum(states == state)) for state in CONTINUITY_STATES},
        "raw": _score_matches(raw, anchors, states),
        "accepted": _score_matches(accepted, anchors, states),
        "scope": [
            "Anonymous animal continuity within the reviewed clip; no biological naming inference.",
            "Three probes can reveal returning changes; their intervals are not densely validated.",
            "Changed intervals and animals are not a count of discrete physical swap events.",
            "Unscored edges remain in the six-edge denominator; absence is not correctness.",
            "No whole-recording, all-three-ID or zero-swap guarantee follows from these probes.",
        ],
    }

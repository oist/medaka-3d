"""Retrospective segment/link evidence, without promoting automatic identities.

Edges are continuations asserted by an output slot. Time groups are review
candidates, not independently adjudicated encounters or swap-event counts.
"""

from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class JoinLedgerConfig:
    minimum_matched_frames: int = 100
    minimum_matched_seconds: float = 0.5
    minimum_matched_fraction: float = 0.95
    maximum_minority_fraction: float = 0.0
    candidate_group_span_seconds: float = 0.25

    def __post_init__(self):
        if (
            self.minimum_matched_frames < 1
            or not np.isfinite(self.minimum_matched_seconds)
            or self.minimum_matched_seconds <= 0
        ):
            raise ValueError("Positive support thresholds are required")
        if not 0 < self.minimum_matched_fraction <= 1:
            raise ValueError("Matched fraction must be in (0, 1]")
        if not 0 <= self.maximum_minority_fraction < 0.5:
            raise ValueError("Minority fraction must be in [0, 0.5)")
        if not np.isfinite(self.candidate_group_span_seconds) or (
            self.candidate_group_span_seconds < 0
        ):
            raise ValueError("Candidate time span must be finite and nonnegative")


def _evidence(identities, weights, labels, config):
    matched = identities >= 0
    seconds = np.bincount(identities[matched], weights=weights[matched], minlength=len(labels))
    matched_seconds = float(seconds.sum())
    observed_seconds = float(weights.sum())
    winner = int(seconds.argmax())
    minority = matched_seconds - float(seconds[winner])
    purity = float(seconds[winner] / matched_seconds) if matched_seconds else None
    matched_fraction = matched_seconds / observed_seconds if observed_seconds else None
    if not matched.any() or not matched_seconds:
        status = "unassessed"
    elif np.count_nonzero(seconds > 0) > 1 and (
        minority / matched_seconds > config.maximum_minority_fraction
    ):
        status = "mixed"
    elif (
        matched.sum() < config.minimum_matched_frames
        or matched_seconds < config.minimum_matched_seconds
    ):
        status = "insufficient_support"
    elif matched_fraction < config.minimum_matched_fraction:
        status = "reference_incomplete"
    else:
        status = "supported_reference_label"
    return {
        "observed_frames": len(identities),
        "observed_seconds": observed_seconds,
        "matched_frames": int(matched.sum()),
        "matched_seconds": matched_seconds,
        "matched_fraction": matched_fraction,
        "majority_label": labels[winner] if matched_seconds else None,
        "purity": purity,
        "label_status": status,
        "supported_reference_label": (
            labels[winner] if status == "supported_reference_label" else None
        ),
        **{f"reference_{label}_seconds": float(seconds[i]) for i, label in enumerate(labels)},
    }


def build_join_ledger(
    local_track,
    matched_slot,
    times,
    weights,
    eligible,
    position_valid,
    *,
    reference_labels=("F", "M1", "M2"),
    config=None,
):
    """Describe local tracks and same-slot edges on an explicit native grid.

    ``matched_slot[frame, reference_identity]`` is the existing independent
    spatial matching result, with -1 for unavailable matches. A segment ends
    at every missing native observation, slot change, or local-ID change.
    Global track support AND segment support are required to score an edge.
    Equal supported endpoint labels establish only endpoint agreement across
    gaps; they never establish identity or a swap count inside a gap.
    """
    config = config or JoinLedgerConfig()
    tracks, matches = np.asarray(local_track), np.asarray(matched_slot)
    times, weights = np.asarray(times, float), np.asarray(weights, float)
    eligible, valid = np.asarray(eligible, bool), np.asarray(position_valid, bool)
    labels = list(reference_labels)
    if tracks.ndim != 2 or matches.shape != (len(tracks), len(labels)):
        raise ValueError("Track and reference matching grids differ")
    n, slots = tracks.shape
    if n < 2 or times.shape != (n,) or weights.shape != (n,) or eligible.shape != (n,):
        raise ValueError("Expected the complete native time grid")
    if valid.shape != tracks.shape:
        raise ValueError("Position mask shape differs from tracks")
    if not np.issubdtype(tracks.dtype, np.integer) or not np.issubdtype(matches.dtype, np.integer):
        raise ValueError("Track IDs and matched slots must be integers")
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Native timestamps must increase strictly")
    if not np.isfinite(weights).all() or (weights < 0).any() or (weights[~eligible] != 0).any():
        raise ValueError("Weights must be finite, nonnegative, and zero outside the window")
    if (matches < -1).any() or (matches >= slots).any() or (tracks < -1).any():
        raise ValueError("Invalid slot or local-track ID")
    identity_at_slot = np.full(tracks.shape, -1, np.int16)
    for identity in range(len(labels)):
        frames = np.flatnonzero(matches[:, identity] >= 0)
        assigned = matches[frames, identity]
        if (identity_at_slot[frames, assigned] >= 0).any():
            raise ValueError("Spatial matches are not one-to-one")
        identity_at_slot[frames, assigned] = identity
    active = (tracks >= 0) & eligible[:, None] & valid
    # A local ID may move columns, but cannot describe two simultaneous rows.
    for a in range(slots):
        for b in range(a):
            if (active[:, a] & active[:, b] & (tracks[:, a] == tracks[:, b])).any():
                raise ValueError("One local track occupies multiple slots simultaneously")
    track_rows = []
    flat_indices = np.flatnonzero(active.ravel())
    flat_ids = tracks.ravel()[flat_indices]
    ordering = np.argsort(flat_ids, kind="stable")
    ordered_indices = flat_indices[ordering]
    ordered_ids = flat_ids[ordering]
    cuts = np.r_[0, np.flatnonzero(np.diff(ordered_ids)) + 1, len(ordered_ids)]
    for lo, hi in pairwise(cuts):
        if lo == hi:
            continue
        indices = ordered_indices[lo:hi]
        frames = indices // slots
        track_rows.append(
            {
                "local_track_id": int(ordered_ids[lo]),
                "first_frame": int(frames.min()),
                "last_frame": int(frames.max()),
                "slots_seen": ",".join(map(str, np.unique(indices % slots))),
                **_evidence(identity_at_slot.ravel()[indices], weights[frames], labels, config),
            }
        )
    by_track = {r["local_track_id"]: r for r in track_rows}
    segments, edges = [], []
    for slot in range(slots):
        state = np.where(active[:, slot], tracks[:, slot], -1)
        boundaries = np.r_[0, np.flatnonzero(np.diff(state)) + 1, n]
        previous = None
        for first, stop in pairwise(boundaries):
            track = int(state[first])
            if track < 0:
                continue
            segment = {
                "segment_id": len(segments),
                "raw_slot": slot,
                "local_track_id": track,
                "first_frame": int(first),
                "last_frame": int(stop - 1),
                "stop_frame_exclusive": int(stop),
                "start_s": float(times[first]),
                "stop_s": float(times[min(stop, n - 1)]),
                **_evidence(
                    identity_at_slot[first:stop, slot], weights[first:stop], labels, config
                ),
                "track_label_status": by_track[track]["label_status"],
                "track_supported_reference_label": by_track[track]["supported_reference_label"],
            }
            segments.append(segment)
            if previous is not None:
                left, right = (
                    previous["supported_reference_label"],
                    segment["supported_reference_label"],
                )
                reasons = []
                for side, row in (("before", previous), ("after", segment)):
                    for level in ("", "track_"):
                        status = row[f"{level}label_status"]
                        if status != "supported_reference_label":
                            reasons.append(f"{side}_{level}{status}")
                    if row["supported_reference_label"] is not None and (
                        row["track_supported_reference_label"] is not None
                        and row["supported_reference_label"]
                        != row["track_supported_reference_label"]
                    ):
                        reasons.append(f"{side}_segment_conflicts_with_track_support")
                    endpoint = row["last_frame"] if side == "before" else row["first_frame"]
                    endpoint_identity = int(identity_at_slot[endpoint, slot])
                    if endpoint_identity < 0:
                        reasons.append(f"{side}_endpoint_unassessed")
                    elif row["supported_reference_label"] is not None and (
                        labels[endpoint_identity] != row["supported_reference_label"]
                    ):
                        reasons.append(f"{side}_endpoint_conflicts_with_support")
                before, after = previous["last_frame"], segment["first_frame"]
                gap = after - before - 1
                endpoint_outcome = (
                    "undecidable"
                    if reasons
                    else (
                        "same_reference_identity"
                        if left == right
                        else "different_reference_identity"
                    )
                )
                # Supported physical reference labels can assess the endpoint
                # correspondence across a gap. They do not assess its interior
                # or determine how many exchanges happened during missing time.
                outcome = "undecidable" if reasons else ("correct" if left == right else "wrong")
                edges.append(
                    {
                        "edge_id": len(edges),
                        "raw_slot": slot,
                        "before_segment_id": previous["segment_id"],
                        "after_segment_id": segment["segment_id"],
                        "before_local_track_id": previous["local_track_id"],
                        "after_local_track_id": track,
                        "before_frame": before,
                        "after_frame": after,
                        "before_time_s": float(times[before]),
                        "after_time_s": float(times[after]),
                        "gap_frames": gap,
                        "gap_seconds": float(weights[before + 1 : after].sum()),
                        "gap_interior_identity_status": "unassessed" if gap else "no_gap",
                        "before_reference_label": left,
                        "after_reference_label": right,
                        "endpoint_outcome": endpoint_outcome,
                        "outcome": outcome,
                        "undecidable_reasons": ";".join(dict.fromkeys(reasons)),
                        "identity_support": "retrospective_reference_diagnostic",
                    }
                )
            previous = segment
    # Bounded-span grouping prevents a chain of nearby onsets from collapsing
    # an arbitrarily long recording. It does not merge gaps or declare events.
    groups, group = [], None
    for edge in sorted(edges, key=lambda e: (e["after_time_s"], e["edge_id"])):
        if group is None or edge["after_time_s"] - group["first_onset_s"] > (
            config.candidate_group_span_seconds
        ):
            group = {
                "candidate_event_id": len(groups),
                "first_onset_s": edge["after_time_s"],
                "last_onset_s": edge["after_time_s"],
                "edge_ids": [],
                "review_status": "unreviewed_temporal_candidate",
            }
            groups.append(group)
        group["edge_ids"].append(edge["edge_id"])
        group["last_onset_s"] = edge["after_time_s"]
        edge["candidate_event_id"] = group["candidate_event_id"]
    for group in groups:
        group["edge_count"] = len(group["edge_ids"])
        group["edge_ids"] = ",".join(map(str, group["edge_ids"]))
    counts = {
        status: sum(e["outcome"] == status for e in edges)
        for status in ("correct", "wrong", "undecidable")
    }
    assessed = counts["correct"] + counts["wrong"]
    summary = {
        "configuration": asdict(config),
        "edge_count": len(edges),
        "edge_outcomes": counts,
        "assessed_edges": assessed,
        "correct_fraction_of_assessed_edges": counts["correct"] / assessed if assessed else None,
        "assessed_fraction_of_all_edges": assessed / len(edges) if edges else None,
        "candidate_group_count": len(groups),
        "adjudicated_event_count": None,
        "expected_grid_fish_seconds": float(weights.sum() * slots),
        "local_position_fish_seconds": float((active * weights[:, None]).sum()),
        "scope": "Same-slot segment edges; not all possible within-segment identity changes",
    }
    return {
        "tracks": pd.DataFrame(track_rows),
        "segments": pd.DataFrame(segments),
        "edges": pd.DataFrame(edges),
        "event_candidates": pd.DataFrame(groups),
        "summary": summary,
    }

"""Reference-based evaluation of warning intervals and slot-only corrections.

Reference transitions are diagnostic brackets, not adjudicated swap events.
All durations use the supplied native acquisition clock and expected coverage.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd


def reference_change_brackets(
    slots, times, weights, *, minimum_seconds=0.1, minimum_samples=10, split_gap_s=0.5
):
    """Use stable matched runs without imposing whole-track reference purity.

    Missing observations do not vote; a long gap starts a separate support run.
    A change across a gap retains its entire uncertainty bracket, even if long.
    """
    rows = []
    for animal in range(slots.shape[1]):
        idx = np.flatnonzero((slots[:, animal] >= 0) & (weights > 0))
        if not len(idx):
            continue
        values = slots[idx, animal]
        breaks = np.flatnonzero((np.diff(values) != 0) | (np.diff(times[idx]) > split_gap_s)) + 1
        bounds = np.r_[0, breaks, len(idx)]
        supported = []
        for lo, hi in pairwise(bounds):
            frames = idx[lo:hi]
            seconds = float(weights[frames].sum())
            if len(frames) >= minimum_samples and seconds >= minimum_seconds:
                supported.append((frames[0], frames[-1], int(values[lo]), seconds, len(frames)))
        for before, after in pairwise(supported):
            if before[2] == after[2]:
                continue
            rows.append(
                {
                    "reference_index": animal,
                    "before_frame": int(before[1]),
                    "after_frame": int(after[0]),
                    "before_slot": before[2],
                    "after_slot": after[2],
                    "start_s": float(times[before[1]]),
                    "end_s": float(times[after[0]]),
                    "bracket_seconds": float(times[after[0]] - times[before[1]]),
                    "before_support_s": before[3],
                    "after_support_s": after[3],
                }
            )
    columns = [
        "reference_index",
        "before_frame",
        "after_frame",
        "before_slot",
        "after_slot",
        "start_s",
        "end_s",
        "bracket_seconds",
        "before_support_s",
        "after_support_s",
    ]
    table = pd.DataFrame(rows, columns=columns).sort_values(["start_s", "end_s"])
    table.insert(0, "change_id", np.arange(len(table)))
    return table.reset_index(drop=True)


def group_change_brackets(changes):
    """Union overlapping brackets; preserve member IDs instead of counting swaps."""
    groups = []
    for row in changes.itertuples():
        if groups and row.start_s <= groups[-1]["end_s"]:
            group = groups[-1]
            group["end_s"] = max(group["end_s"], row.end_s)
            group["after_frame"] = max(group["after_frame"], row.after_frame)
            group["change_ids"].append(row.change_id)
            group["reference_indices"].add(row.reference_index)
        else:
            groups.append(
                {
                    "start_s": row.start_s,
                    "end_s": row.end_s,
                    "before_frame": row.before_frame,
                    "after_frame": row.after_frame,
                    "change_ids": [row.change_id],
                    "reference_indices": {row.reference_index},
                }
            )
    for i, group in enumerate(groups):
        group["group_id"] = i
        group["change_ids"] = ",".join(map(str, group["change_ids"]))
        group["reference_indices"] = ",".join(map(str, sorted(group["reference_indices"])))
        group["bracket_seconds"] = group["end_s"] - group["start_s"]
    return pd.DataFrame(
        groups,
        columns=[
            "group_id",
            "start_s",
            "end_s",
            "before_frame",
            "after_frame",
            "bracket_seconds",
            "change_ids",
            "reference_indices",
        ],
    )


def capture_brackets(brackets, intervals):
    """Distinguish overlap from complete temporal localization of the bracket."""
    out = brackets.copy()
    overlap, contained, ids = [], [], []
    for row in out.itertuples():
        hits = intervals[(intervals.start_s <= row.end_s) & (intervals.end_s >= row.start_s)]
        overlap.append(bool(len(hits)))
        full = hits[(hits.start_s <= row.start_s) & (hits.end_s >= row.end_s)]
        contained.append(bool(len(full)))
        ids.append(",".join(map(str, hits.interval_id)))
    out["warning_overlap"] = overlap
    out["warning_contains_bracket"] = contained
    out["warning_interval_ids"] = ids
    return out


def corrected_slots(matched_slots, permutation):
    """Transform reference-indexed raw slots by output-to-input permutations."""
    inverse = np.argsort(permutation, axis=1)
    result = np.take_along_axis(inverse, np.maximum(matched_slots, 0), axis=1)
    return np.where(matched_slots >= 0, result, -1).astype(np.int8)


def paired_identity_time(before, after, anchor, weights):
    """Score fixed-anchor harm and benefit on identical matched observations."""
    supported = (before >= 0) & (after >= 0) & (np.asarray(anchor)[None] >= 0)
    old = before == np.asarray(anchor)[None]
    new = after == np.asarray(anchor)[None]
    mask = {
        "correct_seconds": supported & new,
        "wrong_seconds": supported & ~new,
        "corrected_seconds": supported & ~old & new,
        "newly_wrong_seconds": supported & old & ~new,
        "unassessed_seconds": ~supported,
    }
    return {name: float((value * weights[:, None]).sum()) for name, value in mask.items()}


def endpoint_outcomes(decisions, matched_slots):
    """Judge exact observed endpoints, with no imputation across unmatched frames."""
    slot_labels = np.full_like(matched_slots, -1)
    for animal in range(matched_slots.shape[1]):
        idx = np.flatnonzero(matched_slots[:, animal] >= 0)
        slot_labels[idx, matched_slots[idx, animal]] = animal
    rows = []
    for row in decisions:
        before = slot_labels[row["before_frame"]]
        after = slot_labels[row["after_frame"]]
        q = row["permutation"]
        for slot in range(3):
            known_old = before[slot] >= 0 and after[slot] >= 0
            known_new = before[slot] >= 0 and after[q[slot]] >= 0
            old = (
                "unknown"
                if not known_old
                else ("correct" if before[slot] == after[slot] else "wrong")
            )
            new = (
                "unknown"
                if not known_new
                else ("correct" if before[slot] == after[q[slot]] else "wrong")
            )
            rows.append(
                {
                    "interval_id": row["interval_id"],
                    "slot": slot,
                    "decision": row["decision"],
                    "before_outcome": old,
                    "after_outcome": new,
                    "slot_changed": slot != q[slot],
                }
            )
    return pd.DataFrame(rows)


def connection_changes(before, after, times, weights):
    """Paired endpoint continuity on identical reference observations.

    Rows are fish-level endpoint edges, not independent crossing events. Gaps
    remain explicit. Identity renaming by any fixed permutation is invariant.
    Only edges wrong under at least one method are exported; totals retain the
    entire known-endpoint denominator.
    """
    if not np.array_equal(before >= 0, after >= 0):
        raise ValueError("Connection comparison requires identical matched availability")
    rows, totals = [], []
    for animal in range(before.shape[1]):
        idx = np.flatnonzero((before[:, animal] >= 0) & (weights > 0))
        a, b = idx[:-1], idx[1:]
        old = before[a, animal] == before[b, animal]
        new = after[a, animal] == after[b, animal]
        for kind, selected in (("adjacent", b == a + 1), ("gap_endpoints", b > a + 1)):
            totals.append(
                {
                    "reference_index": animal,
                    "kind": kind,
                    "assessed_edges": int(selected.sum()),
                    "corrected_edges": int((selected & ~old & new).sum()),
                    "newly_wrong_edges": int((selected & old & ~new).sum()),
                    "unchanged_wrong_edges": int((selected & ~old & ~new).sum()),
                }
            )
        for j in np.flatnonzero(~old | ~new):
            rows.append(
                {
                    "reference_index": animal,
                    "before_frame": int(a[j]),
                    "after_frame": int(b[j]),
                    "start_s": float(times[a[j]]),
                    "end_s": float(times[b[j]]),
                    "kind": "adjacent" if b[j] == a[j] + 1 else "gap_endpoints",
                    "baseline_correct": bool(old[j]),
                    "candidate_correct": bool(new[j]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(totals)

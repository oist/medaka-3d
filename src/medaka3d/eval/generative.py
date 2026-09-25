"""Shared observations and abstention-aware scoring for local identity recovery.

These metrics evaluate permanent animal slots through detection rows matched to
human mouth clicks. They never equate continuity or small reprojection error
with correct identity, and never bridge over an unavailable human anchor.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from medaka3d.types import VIEWS


@dataclass(frozen=True)
class MouthAnchor:
    frame: int
    detection_rows: dict[str, int]
    pixels: dict[str, tuple[float, float]]
    match_errors_px: dict[str, float]


def match_mouth_anchors(
    records: Iterable[dict],
    mouth_at: Callable[[str, int], np.ndarray],
    start: int,
    stop: int,
    match_px: float = 5.0,
) -> list[MouthAnchor]:
    """Match confirmed female mouth marks to untracked detector rows once.

    A confirmed annotation remains in the denominator even if no usable view
    matches. Matching uses only pixels, never either tracker's chosen geometry.
    """
    anchors = []
    seen = set()
    for record in records:
        frame = int(record["frame"])
        if record["gt_status"] != "confirmed" or not start <= frame < stop:
            continue
        if frame in seen:
            raise ValueError(f"duplicate confirmed annotation at frame {frame}")
        seen.add(frame)
        rows, pixels, errors = {}, {}, {}
        for view, prefix in (("side1_cam", "side1"), ("side2_cam", "side2")):
            if record[f"{prefix}_selection_status"] != "sleap_linked":
                continue
            try:
                target = np.array([record[f"{prefix}_mouth_x"],
                                   record[f"{prefix}_mouth_y"]], dtype=float)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(target).all():
                continue
            pixels[view] = tuple(map(float, target))
            points = np.asarray(mouth_at(view, frame), dtype=float)
            if not points.size:
                continue
            distance = np.linalg.norm(points - target, axis=-1)
            distance = np.where(np.isfinite(distance), distance, np.inf)
            order = np.argsort(distance)
            first = int(order[0])
            # An exact tie provides no observation identity.
            if distance[first] > match_px or (
                len(order) > 1 and np.isclose(distance[first], distance[order[1]])
            ):
                continue
            rows[view], errors[view] = first, float(distance[first])
        anchors.append(MouthAnchor(frame, rows, pixels, errors))
    return sorted(anchors, key=lambda anchor: anchor.frame)


def score_female_identity(
    frame_idx: np.ndarray,
    mouth_source: np.ndarray,
    observed_mouth: np.ndarray,
    anchors: list[MouthAnchor],
    *,
    review: np.ndarray | None = None,
) -> dict:
    """Score all adjacent requested anchors, including abstained intervals.

    ``mouth_source`` is (frames, permanent identities, views), containing local
    detector row numbers, or -1. ``review`` makes that identity unavailable.
    Results are invariant to any single permutation of permanent slot names.
    Endpoint changes are swap *intervals*, never a count of individual events.
    """
    frames = np.asarray(frame_idx)
    source = np.asarray(mouth_source)
    observed = np.asarray(observed_mouth, dtype=bool)
    if frames.ndim != 1 or len(np.unique(frames)) != len(frames):
        raise ValueError("frame_idx must contain unique frames")
    if source.shape != (len(frames), 3, len(VIEWS)) or observed.shape != (len(frames), 3):
        raise ValueError("source or observed shape is inconsistent with three identities")
    flagged = np.zeros(observed.shape, dtype=bool) if review is None else np.asarray(review)
    if flagged.shape != observed.shape:
        raise ValueError("review must have shape (frames, identities)")
    position = {int(frame): i for i, frame in enumerate(frames)}
    assigned = []
    for anchor in anchors:
        index = position.get(anchor.frame)
        candidates = set()
        duplicate = False
        if index is not None:
            for view, row in anchor.detection_rows.items():
                slots = np.flatnonzero(source[index, :, VIEWS.index(view)] == row)
                duplicate |= slots.size > 1
                candidates.update(map(int, slots))
        identity = next(iter(candidates)) if len(candidates) == 1 and not duplicate else None
        reason = "matched"
        if index is None:
            reason = "frame unavailable"
        elif duplicate or len(candidates) > 1:
            reason = "conflicting source identities"
        elif identity is None:
            reason = "mouth observation unmatched"
        elif not observed[index, identity]:
            reason = "mouth not supported by two views"
            identity = None
        elif flagged[index, identity]:
            reason = "identity requires review"
            identity = None
        assigned.append({"frame": anchor.frame, "identity": identity, "reason": reason})
    intervals = []
    counts = {"preserved": 0, "swap": 0, "unscored": 0}
    for first, second in pairwise(assigned):
        outcome = "unscored" if None in (first["identity"], second["identity"]) else (
            "preserved" if first["identity"] == second["identity"] else "swap"
        )
        counts[outcome] += 1
        intervals.append({"start": first["frame"], "stop": second["frame"],
                          "outcome": outcome})
    return {
        "requested_anchors": len(anchors),
        "available_anchors": sum(row["identity"] is not None for row in assigned),
        "requested_intervals": len(intervals),
        "interval_counts": counts,
        "anchors": assigned,
        "intervals": intervals,
        "scope": "female anchor endpoints; no event count or male identity validation",
    }


def observed_motion_summary(xyz: np.ndarray, frame_idx: np.ndarray) -> dict:
    """Continuity diagnostics from observed points only, never interpolated gaps."""
    points = np.asarray(xyz, dtype=float)
    frames = np.asarray(frame_idx)
    if points.shape != (len(frames), 3, 3):
        raise ValueError("xyz must have shape (frames, three identities, xyz)")
    finite = np.isfinite(points).all(axis=-1)
    adjacent = (np.diff(frames) == 1)[:, None] & finite[1:] & finite[:-1]
    step = np.linalg.norm(np.diff(points, axis=0), axis=-1)[adjacent]
    return {
        "frames": len(frames),
        "observed_animal_frames": int(finite.sum()),
        "total_animal_frames": int(finite.size),
        "observed_fraction": float(finite.mean()) if finite.size else None,
        "all_three_observed_frames": int(finite.all(axis=1).sum()),
        "scored_adjacent_steps": len(step),
        "step_median_mm": float(np.median(step)) if step.size else None,
        "step_p99_mm": float(np.percentile(step, 99)) if step.size else None,
        "steps_over_10mm": int((step > 10).sum()),
        "steps_over_30mm": int((step > 30).sum()),
        "scope": "geometric diagnostics; low values do not establish correct identity",
    }


def skeleton_shape_summary(
    pose_xyz: np.ndarray, observed: np.ndarray, initial_xyz: np.ndarray,
) -> dict:
    """Expose lost nodes and implausible latent bodies alongside point continuity.

    The fixed initialization supplies each segment's reference length. Maxima
    include predicted nodes: hiding a stretched prediction because it was not
    observed would conceal exactly the failure this diagnostic should expose.
    Undefined initial lengths or entirely missing segments are reported as null.
    No anatomical acceptance threshold or new identity claim is defined here.
    """
    pose = np.asarray(pose_xyz, dtype=float)
    held = np.asarray(observed, dtype=bool)
    initial = np.asarray(initial_xyz, dtype=float)
    if pose.ndim != 4 or pose.shape[1:] != (3, 3, 3):
        raise ValueError("pose_xyz must have shape (frames, fish, nodes, xyz)")
    if held.shape != pose.shape[:-1] or initial.shape != (3, 3, 3):
        raise ValueError("observed or initial_xyz has an inconsistent shape")
    if np.any(held & ~np.isfinite(pose).all(axis=-1)):
        raise ValueError("an observed node must have finite coordinates")
    lengths = np.linalg.norm(np.diff(pose, axis=2), axis=-1)
    references = np.linalg.norm(np.diff(initial, axis=1), axis=-1)

    def number(value):
        return float(value) if np.isfinite(value) else None

    slots = []
    for fish in range(3):
        segments = {}
        for segment, name in enumerate(("mouth-pectoral", "pectoral-tail")):
            values = lengths[:, fish, segment]
            finite = values[np.isfinite(values)]
            largest = float(finite.max()) if finite.size else np.nan
            reference = references[fish, segment]
            ratio = (largest / reference if np.isfinite(reference) and reference > 0
                     else np.nan)
            segments[name] = {"initial_mm": number(reference), "maximum_mm": number(largest),
                              "maximum_to_initial_ratio": number(ratio),
                              "finite_frames": int(finite.size)}
        counts = held[:, fish].sum(axis=0)
        slots.append({
            "slot": fish, "total_frames": len(pose),
            "observed_nodes": {
                name: {"count": int(counts[node]),
                       "fraction": float(counts[node] / len(pose)) if len(pose) else None}
                for node, name in enumerate(("mouth", "pectoral", "tail"))
            },
            "all_nodes_observed_frames": int(held[:, fish].all(axis=1).sum()),
            "segments": segments,
        })
    return {"slots": slots, "scope": "includes latent predictions; not an identity score"}


def mouth_reprojection_summary(
    pose_xyz: np.ndarray, frame_idx: np.ndarray, anchors: list[MouthAnchor],
    scored: dict, project: Callable[[str, np.ndarray], np.ndarray],
) -> dict:
    """Human mouth errors for source-matched anchors, using the mouth in both methods."""
    indexes = {int(frame): i for i, frame in enumerate(frame_idx)}
    assigned = {row["frame"]: row["identity"] for row in scored["anchors"]}
    errors = []
    for anchor in anchors:
        identity = assigned[anchor.frame]
        if identity is None:
            continue
        mouth = pose_xyz[indexes[anchor.frame], identity, 0]
        for view, target in anchor.pixels.items():
            uv = project(view, mouth)
            if np.isfinite(uv).all():
                errors.append(float(np.linalg.norm(uv - target)))
    return {"view_observations": len(errors),
            "median_px": float(np.median(errors)) if errors else None,
            "max_px": float(np.max(errors)) if errors else None,
            "scope": "mouth geometry at source-matched human anchors, not an ID score"}

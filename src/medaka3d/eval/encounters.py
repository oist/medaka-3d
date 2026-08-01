"""Encounter segmentation by sustained separation.

An *encounter* is a window where two animals came close enough that identity is
at risk, bounded on both sides by a stretch of unambiguous separation.

Why the bounding matters. The retired repo ran a blinded identity audit, then
had to invalidate it (`AUDIT_REVISION_NOTICE.md`, 2026-07-14): clips carried
only 83 frames of context, so many started with the animals already close. A
human cannot say "the fish leaving on the left is the one that arrived on the
right" if they never saw it arrive. Worse, under a proper sustained-separation
rule the audit's "7 crossings" collapsed to **2 encounters** — it had been
counting one continuous ambiguity as several independent tests, inflating the
apparent sample size by 3.5x.

So the rule is: an encounter that does not begin and end in separation is not
an encounter. It is part of a longer one.

Encounters are also the unit of annotation. They are rare — tens per
experiment, not thousands — which is what makes a human-labelled benchmark
affordable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, pairwise

import numpy as np

from medaka3d.types import FPS, Encounter

#: Below this separation, identity is considered at risk.
DEFAULT_CLOSE_MM = 20.0
#: Above this separation, two animals are unambiguously distinct.
DEFAULT_SEPARATION_MM = 30.0
#: Separation must hold this long to bound an encounter. 1 s at 166 fps.
DEFAULT_MIN_SEPARATION_FRAMES = 166
#: Context included on each side of the clip so the annotator sees the approach.
DEFAULT_CONTEXT_FRAMES = 166


@dataclass(frozen=True, slots=True)
class EncounterConfig:
    """Thresholds defining what counts as an encounter.

    ``separation_mm`` must exceed ``close_mm``; the gap between them is
    hysteresis that stops a pair hovering near one threshold from being chopped
    into many spurious encounters.
    """

    close_mm: float = DEFAULT_CLOSE_MM
    separation_mm: float = DEFAULT_SEPARATION_MM
    min_separation_frames: int = DEFAULT_MIN_SEPARATION_FRAMES
    context_frames: int = DEFAULT_CONTEXT_FRAMES

    def __post_init__(self) -> None:
        if self.separation_mm <= self.close_mm:
            raise ValueError(
                f"separation_mm ({self.separation_mm}) must exceed close_mm "
                f"({self.close_mm}); without hysteresis a pair lingering near the "
                f"threshold fragments into spurious encounters"
            )
        if self.min_separation_frames < 1:
            raise ValueError("min_separation_frames must be >= 1")


def _sustained_runs(flags: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of True at least ``min_length`` long."""
    if not flags.any():
        return []
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    runs = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]
    return [(a, b) for a, b in runs if b - a >= min_length]


def find_pair_encounters(
    distance_mm: np.ndarray,
    *,
    experiment: str = "",
    participants: tuple[int, ...] = (0, 1),
    config: EncounterConfig | None = None,
    frame_offset: int = 0,
) -> list[Encounter]:
    """Segment one pair's distance trace into encounters.

    Args:
        distance_mm: Per-frame separation. NaN means unknown (a dropped
            detection) and is treated as *not separated* — an unobserved gap
            cannot serve as evidence that the animals were apart.
        participants: Slot indices of the pair, recorded on each encounter.
        frame_offset: Added to every emitted frame index, for traces that do
            not start at frame 0.

    Returns:
        Encounters ordered by entry frame. Each is bounded by sustained
        separation on both sides, so leading and trailing ambiguity that is not
        properly bounded is deliberately dropped rather than half-adjudicated.
    """
    cfg = config or EncounterConfig()
    distance = np.asarray(distance_mm, dtype=float)
    if distance.ndim != 1:
        raise ValueError(f"distance_mm must be 1-D, got shape {distance.shape}")

    separated = np.nan_to_num(distance, nan=-np.inf) >= cfg.separation_mm
    anchors = _sustained_runs(separated, cfg.min_separation_frames)
    if len(anchors) < 2:
        return []  # no bounded interval exists

    close = np.nan_to_num(distance, nan=np.inf) <= cfg.close_mm

    encounters: list[Encounter] = []
    for (_, entry), (exit_, _) in pairwise(anchors):
        # entry = first frame after separation ends; exit_ = first frame of the
        # next sustained separation.
        if not close[entry:exit_].any():
            continue  # they never actually got close
        window = distance[entry:exit_]
        finite = window[np.isfinite(window)]
        encounters.append(
            Encounter(
                experiment=experiment,
                start_frame=frame_offset + max(0, entry - cfg.context_frames),
                end_frame=frame_offset + min(len(distance), exit_ + cfg.context_frames),
                entry_frame=frame_offset + entry,
                exit_frame=frame_offset + exit_,
                participants=participants,
                min_distance_mm=float(finite.min()) if finite.size else float("nan"),
            )
        )
    return encounters


def find_encounters(
    tracks_xyz: np.ndarray,
    *,
    experiment: str = "",
    config: EncounterConfig | None = None,
    frame_offset: int = 0,
) -> list[Encounter]:
    """Segment all pairs of a multi-animal trajectory array into encounters.

    Args:
        tracks_xyz: (n_frames, n_animals, 3) in mm. NaN for missing.

    Returns:
        Encounters across every pair, ordered by entry frame then participants.
    """
    tracks = np.asarray(tracks_xyz, dtype=float)
    if tracks.ndim != 3 or tracks.shape[2] != 3:
        raise ValueError(f"tracks_xyz must be (n_frames, n_animals, 3), got {tracks.shape}")

    out: list[Encounter] = []
    for a, b in combinations(range(tracks.shape[1]), 2):
        distance = np.linalg.norm(tracks[:, a, :] - tracks[:, b, :], axis=1)
        out.extend(
            find_pair_encounters(
                distance,
                experiment=experiment,
                participants=(a, b),
                config=config,
                frame_offset=frame_offset,
            )
        )
    out.sort(key=lambda e: (e.entry_frame, e.participants))
    return out


def summarise(encounters: list[Encounter]) -> dict[str, float]:
    """Headline counts for an encounter set, for annotation budgeting.

    ``annotation_minutes_estimate`` assumes ~1.5 min per encounter: watch the
    clip, mark three animals at entry and again at exit.
    """
    if not encounters:
        return {"count": 0, "total_seconds": 0.0, "annotation_minutes_estimate": 0.0}
    durations = np.array([e.duration_s for e in encounters])
    return {
        "count": len(encounters),
        "total_seconds": float(durations.sum()),
        "median_seconds": float(np.median(durations)),
        "max_seconds": float(durations.max()),
        "over_half_second": int((durations >= 0.5).sum()),
        "over_one_second": int((durations >= 1.0).sum()),
        "annotation_minutes_estimate": round(len(encounters) * 1.5, 1),
        "fps_assumed": FPS,
    }

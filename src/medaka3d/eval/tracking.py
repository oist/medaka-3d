"""Scoring a tracker with **no ground truth** — only its own output and geometry.

There is no human track for M1 or M2 anywhere in this project. Ground truth
labels the female alone (`CLAUDE.md`), so nothing here may compare a tracker to a
human answer. Every quantity below is therefore a property of the prediction, in
the same spirit as `s10`'s part-mixing metrics: it asks whether a tracker's own
output is *geometrically possible*, not whether it is right.

That is a real limitation and it is stated rather than worked around. These
metrics can prove a tracker wrong; they cannot prove it right. A tracker that
swaps two animals while they are touching, and swaps them back, is invisible to
all of them.

Four families, each answering a different failure:

**switch candidates.** Between adjacent frames, the tracker asserts a
correspondence by re-using track IDs. Geometry asserts one too — at 166 fps a
mouth moves a few pixels per frame, so the minimum-displacement assignment is
close to forced. Where the two disagree, either the tracker swapped two animals
or the animals genuinely crossed within 6 ms. Neither can be told apart without
a human, so this is a rate of *candidates* and is named that way.

**track jumps.** A single track whose mouth moved further in one frame than half
a body length. This one needs no comparison at all: it is the tracker claiming an
animal teleported.

**fragmentation.** The metric that stops a tracker winning by refusing to link.
Three animals are present in every frame of every experiment, so the ideal track
count is exactly 3 and everything above it is a break. Reported beside the switch
rate, never traded against it silently.

**encounter identity preservation.** Encounters come from
`medaka3d.eval.encounters` — windows bounded on both sides by sustained
separation, which is the only unit where "did this survive?" is a well-posed
question. Without human truth we cannot ask whether the *right* animal came out;
we can ask whether the tracker kept three unbroken identities across the window,
which is a necessary condition for it having been right.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations

import numpy as np

#: Three animals, so the assignment problem is six permutations and enumerating
#: them is exhaustive. A Hungarian solver would return the same winner and
#: discard the runner-up, which several of these metrics need.
PERMUTATIONS: tuple[tuple[int, ...], ...] = tuple(permutations(range(3)))

#: Instances the tracker declined to assign.
UNTRACKED = -1


@dataclass(frozen=True, slots=True)
class ViewTracks:
    """One camera's tracked detections, flat and sorted by frame.

    ``frame_idx`` (n,), ``mouth`` (n, d), ``track`` (n,) with ``UNTRACKED`` for
    instances the tracker refused. Refusal is a first-class outcome here for the
    same reason `Identity.UNKNOWN` is in `medaka3d.types`: a tracker that
    declines to link has not swapped anyone.

    ``d`` is 2 for a camera's pixels and 3 for reconstructed millimetres. The
    metrics are dimension-agnostic and every threshold is passed in, so the same
    code scores a 2D tracker per view and a 3D tracker on its slots — which is
    the only way the two routes can be compared on one number. ``view`` names
    the camera, or the reconstruction that replaced it.
    """

    experiment: str
    view: str
    frame_idx: np.ndarray
    mouth: np.ndarray
    track: np.ndarray
    n_frames: int

    def __post_init__(self) -> None:
        n = self.frame_idx.shape[0]
        if self.mouth.ndim != 2 or self.mouth.shape[0] != n or self.track.shape != (n,):
            raise ValueError(
                f"ragged arrays: frame_idx {self.frame_idx.shape}, mouth "
                f"{self.mouth.shape}, track {self.track.shape}"
            )
        if n and not np.all(np.diff(self.frame_idx) >= 0):
            raise ValueError("frame_idx must be sorted")


def triples(frame_idx: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Frames holding exactly three instances, and the (m, 3) row indices into them.

    Everything that enumerates the six assignments needs exactly three on both
    sides. Frames with two or four are excluded and the exclusion is counted by
    the caller — never quietly padded, because a padded frame would contribute a
    fictitious correspondence.
    """
    counts = np.bincount(frame_idx, minlength=n_frames)[:n_frames]
    frames = np.flatnonzero(counts == 3)
    start = np.searchsorted(frame_idx, frames, side="left")
    return frames, start[:, None] + np.arange(3)[None, :]


def assignment_cost(here: np.ndarray, there: np.ndarray) -> np.ndarray:
    """(m, 6) total displacement of every assignment between two sets of three points."""
    per = np.linalg.norm(here[:, :, None, :] - there[:, None, :, :], axis=-1)
    return np.stack([per[:, [0, 1, 2], p].sum(axis=1) for p in PERMUTATIONS], axis=-1)


def switch_candidates(tracks: ViewTracks, gate: float) -> dict:
    """Transitions where the tracker's correspondence contradicts geometry.

    Restricted to adjacent frame pairs that both hold exactly three tracked
    instances carrying three distinct IDs — the only case where "the tracker's
    correspondence" is a permutation at all. ``gate`` excludes transitions
    where even the geometric winner moved impossibly far, since there the
    detections themselves are unreliable and the tracker is not the thing being
    measured. Both exclusions are returned as counts.
    """
    frames, rows = triples(tracks.frame_idx, tracks.n_frames)
    if frames.size == 0:
        return _empty_switch()
    position = {int(f): i for i, f in enumerate(frames)}
    pairs = np.asarray([(i, position[f + 1]) for i, f in enumerate(frames) if f + 1 in position])
    if pairs.size == 0:
        return _empty_switch()

    here_rows, there_rows = rows[pairs[:, 0]], rows[pairs[:, 1]]
    here_track, there_track = tracks.track[here_rows], tracks.track[there_rows]

    # A permutation is only defined when both sides carry three distinct IDs and
    # the two frames name the same three.
    def distinct(ids: np.ndarray) -> np.ndarray:
        return (np.sort(ids, axis=1)[:, :-1] != np.sort(ids, axis=1)[:, 1:]).all(axis=1)

    tracked = (here_track != UNTRACKED).all(axis=1) & (there_track != UNTRACKED).all(axis=1)
    comparable = tracked & distinct(here_track) & distinct(there_track)
    comparable &= (np.sort(here_track, axis=1) == np.sort(there_track, axis=1)).all(axis=1)

    here_xy = tracks.mouth[here_rows]
    there_xy = tracks.mouth[there_rows]
    cost = assignment_cost(here_xy, there_xy)
    geometric = np.argmin(cost, axis=1)
    within_gate = cost[np.arange(cost.shape[0]), geometric] <= 3.0 * gate

    # The tracker's own permutation: where did ID k of `here` reappear in `there`?
    tracker_perm = np.argmax(here_track[:, :, None] == there_track[:, None, :], axis=2)
    geometric_perm = np.asarray(PERMUTATIONS)[geometric]

    scored = comparable & within_gate
    disagree = scored & (tracker_perm != geometric_perm).any(axis=1)
    runner_up = np.sort(cost, axis=1)
    forced = runner_up[:, 1] - runner_up[:, 0]

    return {
        "transitions": int(pairs.shape[0]),
        "scored": int(scored.sum()),
        "excluded_untracked_or_degenerate": int((~comparable).sum()),
        "excluded_over_gate": int((comparable & ~within_gate).sum()),
        "switch_candidates": int(disagree.sum()),
        "switch_candidate_rate": float(disagree.sum() / scored.sum()) if scored.any() else np.nan,
        "median_forced_margin": float(np.median(forced[scored])) if scored.any() else np.nan,
    }


def _empty_switch() -> dict:
    return {
        "transitions": 0,
        "scored": 0,
        "excluded_untracked_or_degenerate": 0,
        "excluded_over_gate": 0,
        "switch_candidates": 0,
        "switch_candidate_rate": float("nan"),
        "median_forced_margin": float("nan"),
    }


def track_jumps(tracks: ViewTracks, gate: float) -> dict:
    """Consecutive appearances of one track whose mouth moved further than ``gate``.

    Only strictly adjacent frames count. Across a gap the animal had time to
    move, so a large displacement is not evidence of anything; the gap itself is
    counted instead, as ``bridged``.
    """
    keep = tracks.track != UNTRACKED
    frame = tracks.frame_idx[keep]
    track = tracks.track[keep]
    mouth = tracks.mouth[keep]
    order = np.lexsort((frame, track))
    frame, track, mouth = frame[order], track[order], mouth[order]

    same = track[1:] == track[:-1]
    step = frame[1:] - frame[:-1]
    moved = np.linalg.norm(mouth[1:] - mouth[:-1], axis=1)
    adjacent = same & (step == 1)
    return {
        "adjacent_pairs": int(adjacent.sum()),
        "bridged_gaps": int((same & (step > 1)).sum()),
        "jumps": int((adjacent & (moved > gate)).sum()),
        "jump_rate": float((moved[adjacent] > gate).mean()) if adjacent.any() else np.nan,
        "median_step": float(np.median(moved[adjacent])) if adjacent.any() else np.nan,
        "p99_step": float(np.percentile(moved[adjacent], 99)) if adjacent.any() else np.nan,
    }


def fragmentation(tracks: ViewTracks, n_animals: int = 3) -> dict:
    """Track-count and track-length statistics against the known cardinality.

    ``tracks_per_animal`` is the headline: three animals are present in every
    frame, so 1.0 is perfect and 40.0 means the tracker produced forty fragments
    where one identity should be. This is the number that stops "never link
    anything" from scoring a perfect switch rate.
    """
    keep = tracks.track != UNTRACKED
    labels = tracks.track[keep]
    if labels.size == 0:
        return {
            "tracks": 0,
            "tracks_per_animal": float("nan"),
            "assigned_instances": 0,
            "coverage": 0.0,
            "mean_track_frames": float("nan"),
            "median_track_frames": float("nan"),
            "longest_track_frames": 0,
            "frames_covered_by_top3": 0.0,
        }
    lengths = np.asarray(sorted(Counter(labels.tolist()).values(), reverse=True))
    return {
        "tracks": int(lengths.size),
        "tracks_per_animal": float(lengths.size / n_animals),
        "assigned_instances": int(labels.size),
        "coverage": float(keep.mean()),
        "mean_track_frames": float(lengths.mean()),
        "median_track_frames": float(np.median(lengths)),
        "longest_track_frames": int(lengths.max()),
        # What fraction of all assigned detections the three longest tracks hold.
        # A tracker can have three enormous tracks and a tail of debris; this
        # separates that from three thousand equal fragments.
        "frames_covered_by_top3": float(lengths[:n_animals].sum() / lengths.sum()),
    }


def identity_preservation(
    encounters, slot_track: np.ndarray, *, participants_only: bool = True
) -> dict:
    """Did the tracker hold unbroken identities across each encounter?

    Args:
        encounters: `Encounter` objects from `medaka3d.eval.encounters`, whose
            frame indices address rows of ``slot_track``.
        slot_track: (n_frames, n_slots) track ID per 3D slot per frame,
            ``UNTRACKED`` where the slot had no detection.
        participants_only: Score only the two slots the encounter names. The
            third animal was elsewhere and its identity was never at risk, so
            including it would dilute the rate toward 1 exactly where the metric
            is supposed to be sharp.

    Returns:
        Counts and the preserved rate. **This is a necessary condition, not
        proof.** A tracker that swaps two animals inside the window and keeps
        both IDs continuous scores as preserved; only a human could see that.
        The metric is still worth having because the common failure — a track
        dying at the crossing and a new ID emerging — is exactly what it catches.
    """
    preserved = broken = skipped = 0
    for encounter in encounters:
        slots = (
            list(encounter.participants)
            if participants_only
            else list(range(slot_track.shape[1]))
        )
        entry = slot_track[encounter.entry_frame - 1, slots]
        exit_ = slot_track[min(encounter.exit_frame, slot_track.shape[0] - 1), slots]
        if UNTRACKED in entry or UNTRACKED in exit_:
            skipped += 1  # nothing to preserve if an animal was not tracked at a boundary
            continue
        if np.array_equal(entry, exit_):
            preserved += 1
        else:
            broken += 1
    scored = preserved + broken
    return {
        "encounters": len(encounters),
        "scored": scored,
        "skipped_untracked_boundary": skipped,
        "preserved": preserved,
        "broken": broken,
        "preserved_rate": float(preserved / scored) if scored else float("nan"),
    }


def modal_agreement(pairings: np.ndarray) -> dict:
    """How often a per-frame cross-view pairing equals the pairing it uses most.

    ``pairings`` is (n_frames, k) of small integers — for three animals, the
    permutation taking view A's tracks to view B's. Rows containing a negative
    value are unresolved and excluded.

    A tracker that assigns IDs jointly across the three cameras scores 1.0 here
    **by construction**, and that must be reported as construction rather than as
    a measurement. The number means something only for a tracker whose views were
    solved independently.
    """
    resolved = pairings[(pairings >= 0).all(axis=1)]
    if resolved.size == 0:
        return {"frames": 0, "resolved": 0, "agree": 0, "agreement_rate": float("nan")}
    keys = [tuple(int(v) for v in row) for row in resolved]
    modal, count = Counter(keys).most_common(1)[0]
    return {
        "frames": int(pairings.shape[0]),
        "resolved": int(resolved.shape[0]),
        "modal_pairing": list(modal),
        "agree": int(count),
        "agreement_rate": float(count / resolved.shape[0]),
    }


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z-test. Returns ``(z, two-sided p)``.

    Same test `s10` used, repeated here so the tracking comparison does not
    depend on importing a script.
    """
    from scipy.stats import norm

    if n1 <= 0 or n2 <= 0:
        return float("nan"), float("nan")
    pooled = (k1 + k2) / (n1 + n2)
    se = float(np.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)))
    if se == 0.0:
        return 0.0, 1.0
    z = (k1 / n1 - k2 / n2) / se
    return float(z), float(2.0 * norm.sf(abs(z)))

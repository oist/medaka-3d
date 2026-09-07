"""Tracklets — identity inside **one camera**, decided before any 3D is involved.

The pipeline used to establish identity only in 3D: `reconstruct` chose a
cross-view correspondence once per frame with no temporal term, `track` linked
the resulting 3D points, and `track.propagate_to_views` pushed those IDs back
onto the 2D detections. Two consequences, both measured on 2026-08-18 and
recorded in `docs/decisions/2026-08-18-2d-first-tracking.md`:

* A correspondence that flips on **one frame** hands an animal's identity to
  another permanently, and nothing downstream can see it — the per-view track
  IDs are derived from the same flip, so they agree with it.
* The three cameras therefore agree by construction, and every per-view metric
  was a function of one 3D assignment rather than independent evidence.

This module supplies the missing half. It links detections **within a view**,
using only that view's pixels and time, so the three cameras produce three
opinions that can disagree — which is what makes cross-view agreement a
measurement instead of a tautology.

=============================================================================
WHAT A TRACKLET IS, AND WHERE IT STOPS
=============================================================================

A tracklet is a run of frames over which this camera can follow the same three
animals **confidently**. It ends the moment it cannot, and that is the point:
a tracker that guesses through ambiguity produces long tracks and wrong science,
while one that stops produces short tracks and an honest handover to the stage
that has more evidence.

Three reasons to stop, all local to the view:

``cardinality``  either frame does not hold exactly three detections. Three
                 animals are always present, so anything else means the detector
                 missed one or invented one, and a link built on it would be
                 fiction.
``gate``         the winning assignment moves a node further in one frame than
                 ``gate_px``. Same threshold `s10` derives — half the median
                 labelled mouth-tail span.
``ambiguity``    the runner-up assignment costs less than ``margin_px`` more
                 than the winner. **This is the rule that makes a tracklet mean
                 something.** Without it the linker always has an answer, and
                 its answer at a crossing is a coin toss recorded as a fact.

The anchor node is the **pectoral** by default: it is the node bottom-up never
fails to place (0.000% of complete frames, 2026-08-13), and a linker that stops
because a node was missing is stopping for the wrong reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from medaka3d.reconstruct import N_ANIMALS

#: An instance that belongs to no tracklet. Same convention as `track.UNTRACKED`.
UNTRACKED = -1

#: Three animals, so the assignment is six permutations and enumerating them is
#: exhaustive. A solver would return the same winner and discard the runner-up,
#: which the ambiguity rule needs.
PERMUTATIONS: tuple[tuple[int, ...], ...] = tuple(permutations(range(N_ANIMALS)))


@dataclass(frozen=True, slots=True)
class Tracklets:
    """Per-instance tracklet IDs for one camera, plus why each link was refused.

    ``tracklet`` is (n_instances,) parallel to the detection array it came from.
    ``cuts`` counts the refusals by reason, so a short tracklet set can always be
    explained rather than merely reported.
    """

    experiment: str
    view: str
    tracklet: np.ndarray
    n_frames: int
    gate_px: float
    margin_px: float
    cuts: dict[str, int]

    @property
    def count(self) -> int:
        live = self.tracklet[self.tracklet != UNTRACKED]
        return int(np.unique(live).size)

    def lengths(self) -> np.ndarray:
        """Frames spanned by each tracklet, longest first."""
        live = self.tracklet[self.tracklet != UNTRACKED]
        if live.size == 0:
            return np.zeros(0, dtype=int)
        _, counts = np.unique(live, return_counts=True)
        # Three instances per frame per tracklet-generation, so a tracklet of
        # `k` frames carries `k` instances of one animal.
        return np.sort(counts)[::-1]


def assignment_costs(here: np.ndarray, there: np.ndarray) -> np.ndarray:
    """(m, 6) total displacement of every assignment between two triples."""
    per = np.linalg.norm(here[:, :, None, :] - there[:, None, :, :], axis=-1)
    return np.stack([per[:, [0, 1, 2], p].sum(axis=1) for p in PERMUTATIONS], axis=-1)


def confident_links(
    points: np.ndarray, complete: np.ndarray, *, gate_px: float, margin_px: float
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Which adjacent frame pairs this view can link, and by which permutation.

    Args:
        points: (n_frames, 3, 2) anchor pixels, NaN where the frame is unusable.
        complete: (n_frames,) True where the frame holds exactly three finite
            anchors.
        gate_px: Per-frame movement a link may not exceed, summed over three
            animals against ``3 * gate_px``.
        margin_px: How much worse the runner-up must be. Zero disables the
            ambiguity rule, which is offered for measurement and is never the
            production setting.

    Returns:
        ``(link, permutation, cuts)``. ``link[i]`` is True where frame ``i`` can
        be linked to ``i + 1``; ``permutation[i]`` maps frame ``i``'s slots to
        frame ``i + 1``'s. ``cuts`` counts refusals by reason and its values sum
        to the number of unlinked pairs.
    """
    if points.ndim != 3 or points.shape[1] != N_ANIMALS or points.shape[2] != 2:
        raise ValueError(f"points must be (n_frames, {N_ANIMALS}, 2), got {points.shape}")
    if margin_px < 0.0:
        raise ValueError("margin_px must not be negative")

    n = points.shape[0]
    permutation = np.tile(np.arange(N_ANIMALS), (max(n - 1, 0), 1))
    if n < 2:
        return np.zeros(0, dtype=bool), permutation, {"cardinality": 0, "gate": 0,
                                                      "ambiguity": 0}

    usable = complete[:-1] & complete[1:]
    cost = np.full((n - 1, len(PERMUTATIONS)), np.inf)
    if usable.any():
        cost[usable] = assignment_costs(points[:-1][usable], points[1:][usable])
    ordered = np.sort(cost, axis=1)
    best, runner_up = ordered[:, 0], ordered[:, 1]
    winner = np.argmin(cost, axis=1)

    within_gate = np.isfinite(best) & (best <= N_ANIMALS * gate_px)
    # inf - inf is nan and warns; an unusable pair is not unambiguous either way.
    finite = np.isfinite(best) & np.isfinite(runner_up)
    unambiguous = np.zeros_like(finite)
    unambiguous[finite] = (runner_up[finite] - best[finite]) >= margin_px
    link = usable & within_gate & unambiguous

    cuts = {
        "cardinality": int((~usable).sum()),
        "gate": int((usable & ~within_gate).sum()),
        "ambiguity": int((usable & within_gate & ~unambiguous).sum()),
    }
    permutation[link] = np.asarray(PERMUTATIONS)[winner[link]]
    return link, permutation, cuts


def build(
    frame_idx: np.ndarray,
    points: np.ndarray,
    n_frames: int,
    *,
    experiment: str,
    view: str,
    gate_px: float,
    margin_px: float,
) -> Tracklets:
    """Assign a tracklet ID to every instance of one camera.

    Args:
        frame_idx: (n_instances,) sorted frame of each detection.
        points: (n_instances, 2) the anchor node in pixels.
        n_frames: Length of the recording, so frames with no detection are
            represented rather than skipped.

    Returns:
        `Tracklets`. Instances in frames that do not hold exactly three finite
        anchors carry ``UNTRACKED`` — they are not assigned to anything, because
        a frame whose cardinality is wrong cannot say which animal is which.
    """
    if frame_idx.shape[0] != points.shape[0]:
        raise ValueError(
            f"ragged: {frame_idx.shape[0]} frames, {points.shape[0]} points"
        )
    if frame_idx.size and not np.all(np.diff(frame_idx) >= 0):
        raise ValueError("frame_idx must be sorted")

    counts = np.bincount(frame_idx, minlength=n_frames)[:n_frames]
    start = np.searchsorted(frame_idx, np.arange(n_frames), side="left")
    triple = counts == N_ANIMALS
    rows = start[triple][:, None] + np.arange(N_ANIMALS)[None, :]

    grid = np.full((n_frames, N_ANIMALS, 2), np.nan)
    grid[triple] = points[rows]
    complete = triple & np.isfinite(grid).all(axis=(1, 2))

    link, permutation, cuts = confident_links(
        grid, complete, gate_px=gate_px, margin_px=margin_px
    )

    # Walk forward once. A linked pair carries the previous frame's IDs through
    # the winning permutation; anything else starts three fresh IDs, so a
    # tracklet never spans a refusal.
    slot_id = np.full((n_frames, N_ANIMALS), UNTRACKED, dtype=np.int64)
    next_id = 0
    for frame in range(n_frames):
        if not complete[frame]:
            continue
        if frame > 0 and complete[frame - 1] and link[frame - 1]:
            slot_id[frame] = slot_id[frame - 1][np.argsort(permutation[frame - 1])]
        else:
            slot_id[frame] = np.arange(next_id, next_id + N_ANIMALS)
            next_id += N_ANIMALS

    tracklet = np.full(frame_idx.shape[0], UNTRACKED, dtype=np.int64)
    tracklet[rows.ravel()] = slot_id[triple].ravel()
    return Tracklets(
        experiment=experiment, view=view, tracklet=tracklet, n_frames=n_frames,
        gate_px=gate_px, margin_px=margin_px, cuts=cuts,
    )

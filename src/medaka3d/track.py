"""Linking 3D triples over time. Three methods, differing in exactly one idea each.

Every method here consumes `medaka3d.reconstruct.Reconstruction` — per-frame
triples of 3D mouths in mm, slots arbitrary — and returns a track ID per slot per
frame. None of them names an animal. Naming is `Identity`, it is a claim about the
world, and nothing in this module has the evidence to make it.

The three differ by a single mechanism apiece, so that a difference in score has
one candidate explanation rather than four:

* **nearest** — Hungarian on 3D mouth distance, every frame, no gate and no
  memory. The baseline. It cannot decline: if a frame resolves, three tracks
  continue, however far anything moved.
* **gated** — the same, plus a distance gate and a gap tolerance. A link further
  than the gate is refused and the track is held open for a while instead of
  being handed to whatever detection was nearest. This is the method that can
  say "I do not know", and abstention is the behaviour the project's metrics
  reward.
* **predictive** — `gated`, plus a constant-velocity prediction. Matching is done
  against where each track was *going*, not where it was. The hypothesis is
  specific: two animals passing close are ambiguous by position and separable by
  direction, so this should differ from `gated` only at crossings, which is
  exactly where identity is decided.

The comparison rule is in `tools/c04_score_tracking.py` and was written before
any of them ran.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from medaka3d.reconstruct import N_ANIMALS, Reconstruction

#: A slot with no track. Never a failure state — see `medaka3d.types`.
UNTRACKED = -1


@dataclass(frozen=True, slots=True)
class TrackConfig:
    """Every knob any method takes. Each is a threshold, never a switch.

    Design rule 5 — no flag may change which algorithm runs — is why the three
    methods are three functions and not one function with a ``method=`` argument.
    A value here changes how strict a method is, never which one it is.

    ``max_step_mm`` is derived from the data by the caller and printed; there is
    no default worth trusting, because it depends on the frame rate and on how
    fast these animals actually move.
    """

    max_step_mm: float
    max_gap_frames: int = 0
    #: A track absent this long is closed rather than bridged. Beyond a few
    #: frames the gate stops meaning anything: an animal free for 100 ms can be
    #: anywhere the gate allows and the link is a guess dressed as a measurement.
    velocity_frames: int = 0
    #: Ceiling for the widened gate, in mm. The per-track ceiling is
    #: ``max_step_mm * frames absent``, which is an honest upper bound on how far
    #: an animal could have travelled and a vacuous one once it exceeds the tank:
    #: at K_1_2 frame 82,717 a track 56 frames old carried a ceiling of 224 mm in
    #: a 192 mm tank, so it could claim any animal in the volume and did. The
    #: default keeps the unbounded behaviour every run before `c21` had.
    max_ceiling_mm: float = float("inf")


def _tracks_array(n_frames: int) -> np.ndarray:
    return np.full((n_frames, N_ANIMALS), UNTRACKED, dtype=np.int32)


def _match(
    cost: np.ndarray, ceiling: np.ndarray | None, *, refuse_first: bool = False
) -> list[tuple[int, int]]:
    """Hungarian on true distance, then drop links over their own ceiling.

    Two things are deliberately separate here and were once conflated, at real
    cost. The **assignment** is made on the raw distance; the **rejection** uses a
    per-track ceiling that widens with how long that track has been missing,
    because an animal absent for ten frames had ten frames in which to move.
    Folding the ceiling into the cost — dividing distance by it before solving —
    inverts the comparison: a 60-frame-old track 30 mm away scores better than a
    1-frame-old track 1 mm away, so stale tracks steal slots from fresh ones and
    the identity set churns every few frames while the *track count* stays low,
    which is a failure both the switch metric and the track count hide.

    ``refuse_first`` reverses the remaining order — ceiling first, assignment
    afterwards — and is `strict`'s single idea. The default was chosen because an
    infinite cell pushes the solver into scrambling the other two links to avoid
    it. K_1_2 frame 82,717 is the case where the default's own reasoning turns
    around: a **finite** cell does the scrambling. One animal had just left both
    side cameras, its track had nowhere within reach, and parking it 11.00 mm
    away was cheaper for the total than leaving it unmatched — which pushed the
    female's track 2.52 mm sideways onto the neighbouring slot. The ceiling then
    killed the 11.00 mm link and kept the 2.52 mm one, so the refusal that was
    supposed to protect her is what completed the swap. Refusing first gives the
    identity assignment and a new track for the slot nothing can claim.
    """
    if ceiling is not None and refuse_first:
        allowed = np.isfinite(cost) & (cost <= ceiling[:, None])
        rows, cols = linear_sum_assignment(np.where(allowed, cost, 1e12))
        return [
            (int(r), int(c))
            for r, c in zip(rows, cols, strict=False)
            if allowed[r, c]
        ]
    rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e12))
    pairs = [(int(r), int(c)) for r, c in zip(rows, cols, strict=False)]
    if ceiling is None:
        return pairs
    return [
        (r, c) for r, c in pairs if np.isfinite(cost[r, c]) and cost[r, c] <= ceiling[r]
    ]


def nearest(reconstruction: Reconstruction, config: TrackConfig) -> np.ndarray:
    """Hungarian on 3D distance between consecutive resolved frames. No gate, no gaps.

    Returns (n_frames, n_animals) track IDs. This is the method every other one
    has to beat, and it is deliberately the one with nothing in it: if the
    frame-to-frame problem is as forced as `c02` measured, a bare Hungarian
    should already be close to right, and any added mechanism has to earn its
    complexity against that.
    """
    return _link(reconstruction, config, gate=False, predict=False)


def gated(reconstruction: Reconstruction, config: TrackConfig) -> np.ndarray:
    """`nearest` plus a distance gate and gap bridging."""
    return _link(reconstruction, config, gate=True, predict=False)


def predictive(reconstruction: Reconstruction, config: TrackConfig) -> np.ndarray:
    """`gated` plus constant-velocity prediction of where each track is going."""
    return _link(reconstruction, config, gate=True, predict=True)


def strict(reconstruction: Reconstruction, config: TrackConfig) -> np.ndarray:
    """`gated`, with the gate applied **before** the assignment instead of after.

    One idea, and it changes who pays for a track that cannot be matched. Under
    `gated` an unmatchable track is still assigned — to whatever is cheapest for
    the total — and only then refused, so it can displace a link that was
    correct. Under `strict` it is refused first and simply goes unmatched.
    """
    return _link(reconstruction, config, gate=True, predict=False, refuse_first=True)


def _link(
    reconstruction: Reconstruction, config: TrackConfig, *,
    gate: bool, predict: bool, refuse_first: bool = False,
) -> np.ndarray:
    xyz = reconstruction.xyz
    resolved = reconstruction.resolved
    n_frames = xyz.shape[0]
    tracks = _tracks_array(n_frames)

    limit = config.max_step_mm if gate else None
    max_gap = config.max_gap_frames if gate else 0

    # Open tracks: id -> (last frame seen, last position, velocity per frame).
    live: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    history: dict[int, list[tuple[int, np.ndarray]]] = {}
    next_id = 0

    def open_track(frame: int, slot: int) -> None:
        nonlocal next_id
        live[next_id] = (int(frame), xyz[frame, slot], np.zeros(3))
        history[next_id] = [(int(frame), xyz[frame, slot])]
        tracks[frame, slot] = next_id
        next_id += 1

    for frame in np.flatnonzero(resolved):
        points = xyz[frame]
        # An animal with no 3D position on this frame offers nothing to match
        # against. Where every animal has one — every reconstruction before
        # stage C, on every frame it called resolved — this is all three slots
        # and nothing below changes.
        slots = np.flatnonzero(np.isfinite(points).all(axis=1))
        if slots.size == 0:
            continue
        if not live:
            for slot in slots:
                open_track(int(frame), int(slot))
            continue

        ids = sorted(live)
        ages = np.asarray([frame - live[i][0] for i in ids])
        if gate:
            keep = ages <= max(max_gap, 1)
            ids = [i for i, ok in zip(ids, keep, strict=True) if ok]
            for dead in set(live) - set(ids):
                del live[dead]
            if not ids:
                for slot in slots:
                    open_track(int(frame), int(slot))
                continue
            ages = np.asarray([frame - live[i][0] for i in ids])

        expected = np.stack(
            [
                live[i][1] + (live[i][2] * age if predict else 0.0)
                for i, age in zip(ids, ages, strict=True)
            ]
        )
        cost = np.linalg.norm(expected[:, None, :] - points[None, slots, :], axis=-1)
        # The ceiling is per frame of absence: an animal gone for three frames had
        # three frames in which to move, and charging it a one-frame budget would
        # break exactly the tracks that gap bridging exists to keep. It bounds the
        # link; it does not score it.
        ceiling = (None if limit is None else
                   np.minimum(limit * np.maximum(ages, 1), config.max_ceiling_mm))
        pairs = _match(cost, ceiling, refuse_first=refuse_first)

        claimed = set()
        for row, column in pairs:
            slot = int(slots[column])
            track_id = ids[row]
            previous_frame, previous_xy, velocity = live[track_id]
            step = int(frame) - previous_frame
            new_velocity = (
                (points[slot] - previous_xy) / step
                if config.velocity_frames and step > 0
                else velocity
            )
            if config.velocity_frames > 1 and len(history[track_id]) >= config.velocity_frames:
                old_frame, old_xy = history[track_id][-config.velocity_frames]
                if int(frame) > old_frame:
                    new_velocity = (points[slot] - old_xy) / (int(frame) - old_frame)
            live[track_id] = (int(frame), points[slot], new_velocity)
            history[track_id].append((int(frame), points[slot]))
            tracks[frame, slot] = track_id
            claimed.add(slot)

        for slot in slots:
            if int(slot) not in claimed:
                open_track(int(frame), int(slot))

    return tracks


#: The methods, by name. Adding one here is how the next gets compared; there is
#: no `method=` string threaded through a single implementation.
METHODS = {
    "nearest": nearest,
    "gated": gated,
    "predictive": predictive,
    "strict": strict,
}


def propagate_to_views(
    reconstruction: Reconstruction,
    slot_track: np.ndarray,
    frame_starts: dict[str, np.ndarray],
    n_instances: dict[str, int],
) -> dict[str, np.ndarray]:
    """Push 3D track IDs back onto the 2D detections that produced them.

    A 3D tracker assigns one ID per animal across all three cameras at once, so
    the per-view track IDs it implies agree between views **by construction**.
    That is worth having anyway: it lets the same per-view metrics score a 3D
    tracker and a 2D one, and it means a wrong cross-view correspondence shows up
    as a per-view track jumping between animals — the only way that failure is
    visible at all.

    Args:
        frame_starts: ``view -> (n_frames,)`` index of each frame's first
            detection in that view's flat instance array.

    Returns:
        ``view -> (n_instances,)`` track ID, ``UNTRACKED`` where the instance
        took no part in a resolved correspondence.
    """
    out = {}
    for column, view in enumerate(reconstruction.views):
        starts = frame_starts[view]
        assigned = np.full(n_instances[view], UNTRACKED, dtype=np.int32)
        # Participation is per slot, not per view. Reading it off slot 0 was
        # equivalent while a view either supplied all three detections or none;
        # once a camera can lose one animal and keep the other two, that test
        # throws away the two it kept.
        frames = np.flatnonzero(reconstruction.resolved)
        for slot in range(N_ANIMALS):
            offsets = reconstruction.source[frames, slot, column]
            valid = offsets >= 0
            rows = starts[frames[valid]] + offsets[valid]
            assigned[rows] = slot_track[frames[valid], slot]
        out[view] = assigned
    return out

"""Per-frame cross-view correspondence: which detection in each camera is the same animal.

Three cameras each report three detections. Which of the 3 x 3 x 3 = 27 possible
triples are the three real animals is not given by the detector — it is solved
here, geometrically, and the answer is what makes a 3D point mean anything.

**Why the cost is a point-to-ray residual and not a reprojection error.** The
refractive model back-projects in closed form and forward-projects only by
iteration (`medaka3d.calib`), so the residual is the natural currency; it is also
the unit the calibration was validated in, at 0.24 mm rms on held-out wet
checkerboards. A correct triple reconstructs to ~1.3 mm on this data; a wrong one
to ~15 mm, because a wrong triple asks three rays aimed at three different
animals to meet. That order-of-magnitude gap is what makes the correspondence
solvable at all, and it is measured in `tools/c02_tracking_probe.py` rather than
assumed here.

**Two cameras are enough.** Triangulation needs two rays, and requiring three
would discard every frame where one view lost an animal — 98.96% of frames have
at least two views reporting exactly three detections against 86.4% that have all
three. The views actually used are recorded per frame, because a two-view
reconstruction is a weaker measurement than a three-view one and downstream code
must be able to tell them apart.

**Enumeration, not Hungarian.** With three animals there are 3! = 6 assignments
per extra view, so 36 hypotheses for three cameras and 6 for two. Enumerating
them is exhaustive, costs nothing at this size, and — unlike a Hungarian solver —
keeps the runner-up, which is the only evidence available about whether the
winner was forced or lucky.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np

from medaka3d.calib import Rig, triangulate

#: Three animals per trial, always. This is a fact about the experiment, not a
#: parameter: one female and two males are placed in the tank and none leave.
N_ANIMALS = 3

PERMUTATIONS: tuple[tuple[int, ...], ...] = tuple(permutations(range(N_ANIMALS)))

#: Slot for a frame that could not be resolved.
UNRESOLVED = -1


@dataclass(frozen=True, slots=True)
class Reconstruction:
    """Per-frame 3D triples and the correspondence that produced them.

    ``xyz`` is (n_frames, 3, 3) in mm — frame, slot, coordinate — NaN where the
    frame was not resolved. ``source`` is (n_frames, 3, n_views) giving, for each
    slot, the index of the contributing detection within that frame's view, or
    ``UNRESOLVED`` where the view did not take part. ``residual_mm`` is the
    per-slot point-to-ray RMS; ``margin_mm`` is the runner-up hypothesis's total
    cost minus the winner's, which is how forced the correspondence was.

    Slots are arbitrary and stable only within a frame. They are not identities
    and not tracks — naming them would be the exact conflation
    `medaka3d.types` exists to prevent.
    """

    experiment: str
    views: tuple[str, ...]
    xyz: np.ndarray
    source: np.ndarray
    residual_mm: np.ndarray
    margin_mm: np.ndarray
    n_views_used: np.ndarray

    @property
    def resolved(self) -> np.ndarray:
        """(n_frames,) True where a correspondence was found."""
        return self.n_views_used >= 2


def hypotheses(n_views: int) -> np.ndarray:
    """(n_hypotheses, n_animals, n_views) detection index per slot per view.

    View 0 is the reference and keeps the identity assignment; the remaining
    views are permuted against it. Fixing one view is what stops the same
    labelling being counted 6 times over.
    """
    out = []
    for perms in product(PERMUTATIONS, repeat=n_views - 1):
        out.append(np.stack([np.arange(N_ANIMALS), *perms], axis=-1))
    return np.asarray(out, dtype=np.intp)


def solve(
    origins: np.ndarray, directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose the correspondence for a batch of frames.

    Args:
        origins: (n_frames, n_views, n_animals, 3) ray origins, view-major.
        directions: same shape, unit directions.

    Returns:
        ``(assignment, xyz, residual, best, second)``. ``assignment`` is
        (n_frames, n_animals, n_views); ``xyz`` (n_frames, n_animals, 3);
        ``residual`` (n_frames, n_animals); ``best``/``second`` (n_frames,) the
        winning and runner-up total costs in mm.
    """
    n_frames, n_views = origins.shape[0], origins.shape[1]
    if n_views < 2:
        raise ValueError("cross-view correspondence needs at least two views")

    # Every combination of one detection per view, triangulated once. The 36
    # hypotheses are sums of three of these, so triangulating per hypothesis
    # would repeat the same arithmetic four times over.
    combos = np.asarray(list(product(range(N_ANIMALS), repeat=n_views)), dtype=np.intp)
    frame = np.arange(n_frames)[:, None, None]
    view = np.arange(n_views)[None, None, :]
    combo_origins = origins[frame, view, combos[None, :, :]]  # (f, n_combos, n_views, 3)
    combo_directions = directions[frame, view, combos[None, :, :]]
    combo_xyz, combo_rms = triangulate(combo_origins, combo_directions)

    # Flat index of a (view -> detection) tuple inside `combos`, base-3.
    powers = N_ANIMALS ** np.arange(n_views - 1, -1, -1)
    hypothesis = hypotheses(n_views)  # (h, n_animals, n_views)
    flat = (hypothesis * powers).sum(axis=-1)  # (h, n_animals)

    cost = combo_rms[:, flat].sum(axis=-1)  # (f, h)
    finite = np.where(np.isfinite(cost), cost, np.inf)
    order = np.argsort(finite, axis=-1)
    winner = order[:, 0]
    best = np.take_along_axis(finite, order[:, :1], axis=-1)[:, 0]
    second = (
        np.take_along_axis(finite, order[:, 1:2], axis=-1)[:, 0]
        if order.shape[1] > 1
        else np.full(n_frames, np.inf)
    )

    chosen = flat[winner]  # (f, n_animals)
    rows = np.arange(n_frames)[:, None]
    return (
        hypothesis[winner],
        combo_xyz[rows, chosen],
        combo_rms[rows, chosen],
        np.where(np.isfinite(best), best, np.nan),
        np.where(np.isfinite(second), second, np.nan),
    )


def reconstruct(
    rig: Rig,
    experiment: str,
    uv_by_view: dict[str, np.ndarray],
    complete: dict[str, np.ndarray],
    n_frames: int,
    *,
    batch: int = 20_000,
) -> Reconstruction:
    """Solve every frame that has at least two views reporting three detections.

    Args:
        uv_by_view: ``view -> (n_frames, 3, 2)`` pixels. Rows for frames where
            that view did not report exactly three detections are ignored and may
            hold anything.
        complete: ``view -> (n_frames,)`` bool, True where that view reported
            exactly three.

    Frames are grouped by *which* views are complete and each group is solved
    together, so a two-view frame is never padded with a fabricated third view.
    """
    views = rig.views
    available = np.stack([complete[v] for v in views], axis=1)  # (n_frames, n_views)

    xyz = np.full((n_frames, N_ANIMALS, 3), np.nan)
    residual = np.full((n_frames, N_ANIMALS), np.nan)
    margin = np.full(n_frames, np.nan)
    source = np.full((n_frames, N_ANIMALS, len(views)), UNRESOLVED, dtype=np.int8)
    used = available.sum(axis=1).astype(np.int8)

    rays = {v: rig.camera(v).rays(uv_by_view[v]) for v in views}

    for pattern in {tuple(row) for row in available[used >= 2]}:
        present = [i for i, flag in enumerate(pattern) if flag]
        frames = np.flatnonzero((available == np.asarray(pattern)).all(axis=1))
        for lo in range(0, frames.size, batch):
            block = frames[lo : lo + batch]
            origins = np.stack([rays[views[i]][0][block] for i in present], axis=1)
            directions = np.stack([rays[views[i]][1][block] for i in present], axis=1)
            assignment, points, rms, best, second = solve(origins, directions)
            xyz[block] = points
            residual[block] = rms
            margin[block] = second - best
            for column, view_index in enumerate(present):
                source[block, :, view_index] = assignment[:, :, column]

    # A frame whose triangulation failed outright is not resolved, whatever the
    # detector reported.
    used = np.where(np.isfinite(xyz).all(axis=(1, 2)), used, 0).astype(np.int8)
    return Reconstruction(
        experiment=experiment,
        views=views,
        xyz=xyz,
        source=source,
        residual_mm=residual,
        margin_mm=margin,
        n_views_used=used,
    )

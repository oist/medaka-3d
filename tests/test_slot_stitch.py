"""`c30`'s join rule on trajectories whose right answer is known by construction.

Three synthetic animals far apart, with the failures the real data shows
injected at known frames: a lasting slot permutation, a later three-cycle, a
single-frame triangulation spike that lands near another animal, and an
unresolved gap. The rule must find exactly the injected permutations, at the
injected frames, and must not fire on the spike - the spike is the case the
median window exists for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

c30 = pytest.importorskip("c30_slot_stitch")

N_FRAMES = 2_000
SWAP_AT, SWAP = 700, (1, 0, 2)
CYCLE_AT, CYCLE = 1_400, (1, 2, 0)
SPIKE_AT = 300
GAP = slice(1_000, 1_020)


def _world() -> np.ndarray:
    """(F, 3, 3) three smooth trajectories tens of mm apart, ~0.3 mm/frame."""
    t = np.arange(N_FRAMES) / 166.0
    centres = np.array([[50.0, 50.0, 50.0], [140.0, 60.0, 120.0], [60.0, 150.0, 90.0]])
    xyz = np.empty((N_FRAMES, 3, 3))
    for s in range(3):
        xyz[:, s, 0] = centres[s, 0] + 12 * np.sin(0.9 * t + s)
        xyz[:, s, 1] = centres[s, 1] + 12 * np.cos(0.7 * t + 2 * s)
        xyz[:, s, 2] = centres[s, 2] + 8 * np.sin(0.5 * t + 3 * s)
    return xyz


def _observed(world: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(7)
    seen = world + rng.normal(0.0, 0.3, world.shape)
    seen[SWAP_AT:] = seen[SWAP_AT:][:, list(SWAP), :]
    seen[CYCLE_AT:] = seen[CYCLE_AT:][:, list(CYCLE), :]
    # One frame where slot 2's triangulation lands beside animal 0 and returns.
    seen[SPIKE_AT, 2] = seen[SPIKE_AT, 0] + [1.5, 0.0, 0.0]
    seen[GAP] = np.nan
    return seen


def test_finds_the_injected_permutations_and_nothing_else():
    world = _world()
    seen = _observed(world)
    ok_idx = np.flatnonzero(np.isfinite(seen).any(axis=(1, 2)))
    _perm, joins, refusals = c30.stitch(seen, ok_idx)

    assert [j["frame"] for j in joins] == [SWAP_AT, CYCLE_AT]
    assert joins[0]["permutation"] == list(SWAP)
    assert joins[1]["permutation"] == list(CYCLE)
    assert not any(r["frame"] == SPIKE_AT for r in refusals)


def test_stitched_output_restores_continuity():
    world = _world()
    seen = _observed(world)
    ok_idx = np.flatnonzero(np.isfinite(seen).any(axis=(1, 2)))
    perm, _joins, _ = c30.stitch(seen, ok_idx)
    fixed = seen[np.arange(N_FRAMES)[:, None], perm]

    # Away from the injected spike, every slot follows one animal throughout.
    keep = np.ones(N_FRAMES, bool)
    keep[SPIKE_AT] = False
    keep[GAP] = False
    error = np.linalg.norm(fixed[keep] - world[keep], axis=-1)
    assert np.nanmax(error) < 2.0

    # And the visible symptom is gone: no large adjacent step remains.
    adjacent = np.diff(ok_idx) == 1
    step = np.linalg.norm(
        fixed[ok_idx[1:][adjacent]] - fixed[ok_idx[:-1][adjacent]], axis=-1)
    step = step[np.isfinite(step)]
    unfixed_step = np.linalg.norm(
        seen[ok_idx[1:][adjacent]] - seen[ok_idx[:-1][adjacent]], axis=-1)
    unfixed_step = unfixed_step[np.isfinite(unfixed_step)]
    assert (unfixed_step > 30.0).sum() >= 4          # the injected switches
    assert (step > 30.0).sum() == 2                  # the spike out and back


def test_close_encounter_is_refused_not_guessed():
    """When all animals are within the margin of each other, no join is made."""
    world = _world()
    # Collapse everyone to within 2 mm around the swap moment.
    tight = world.copy()
    centre = world[SWAP_AT].mean(axis=0)
    window = slice(SWAP_AT - 20, SWAP_AT + 20)
    tight[window] = centre + 0.02 * (world[window] - centre)
    seen = tight + np.random.default_rng(11).normal(0.0, 0.3, world.shape)
    seen[SWAP_AT:] = seen[SWAP_AT:][:, list(SWAP), :]

    ok_idx = np.arange(N_FRAMES)
    _, joins, _ = c30.stitch(seen, ok_idx)
    assert not any(abs(j["frame"] - SWAP_AT) < 20 for j in joins)


# ---------------------------------------------------------------------------
# v2 - the evidence ladder (decisions/2026-09-01-slot-stitching.md, v2 section)
# ---------------------------------------------------------------------------
def test_velocity_joins_where_medians_refuse():
    """Lane-keeping swap: labels flip between two fish 3.5 mm apart moving
    fast in parallel. The median windows straddle the motion, so the swap and
    identity costs both fill with travelled distance and the margin collapses
    below 5 mm - v1 refused exactly this class four times and the female
    annotation showed all four were real swaps. Velocity extrapolation
    removes the motion component and is decisive."""
    rng = np.random.default_rng(5)
    n = 2_000
    t = np.arange(n, dtype=float)
    world = np.empty((n, 3, 3))
    world[:, 0] = np.stack([0.8 * (t - 600), np.full(n, 2.5), np.full(n, 50.0)], 1)
    world[:, 1] = np.stack([0.8 * (t - 600), np.full(n, -2.5), np.full(n, 50.0)], 1)
    world[:, 2] = np.stack([np.full(n, 150.0), np.full(n, 150.0), np.full(n, 100.0)], 1)
    seen = world + rng.normal(0.0, 0.04, world.shape)
    seen[600:] = seen[600:][:, [1, 0, 2]]           # lasting swap of slots 0, 1

    perm, joins, _refusals = c30.stitch(seen, np.arange(n))
    velocity_joins = [j for j in joins if j.get("evidence") == "velocity"]
    assert [j["frame"] for j in velocity_joins] == [600]
    assert velocity_joins[0]["permutation"] == [1, 0, 2]
    assert 2.5 <= velocity_joins[0]["margin_mm"] <= 5.0     # leaning, not deciding
    assert velocity_joins[0]["velocity_margin_mm"] > 5.0
    fixed = seen[np.arange(n)[:, None], perm]
    assert np.linalg.norm(fixed[610:] - world[610:], axis=-1).max() < 1.0


def _ghost_world(n=1_600):
    t = np.arange(n, dtype=float)
    world = np.empty((n, 3, 3))
    world[:, 0] = np.stack([50 + 0.1 * t % 60, np.full(n, 40.0), np.full(n, 60.0)], 1)
    world[:, 1] = np.stack([np.full(n, 140.0), 40 + 0.05 * t, np.full(n, 90.0)], 1)
    world[:, 2] = np.stack([np.full(n, 90.0), np.full(n, 150.0), np.full(n, 30.0)], 1)
    return world


def test_bridge_repairs_a_ghost_zone():
    """31479 in miniature: thirty frames of fictitious geometry inside a degraded
    stretch - one slot lost, the rest teleporting - with a lasting swap on the
    far side. The interior boundaries cannot decide anything (that is the
    point); the post-pass audit compares the clean endpoints and corrects."""
    rng = np.random.default_rng(9)
    world = _ghost_world()
    n = world.shape[0]
    seen = world + rng.normal(0.0, 0.1, world.shape)
    ghost = world[800:830, 1] + rng.normal(0.0, 30.0, (30, 3))
    seen[800:830, 0] = np.nan                               # one slot lost
    seen[800:830, 1] = ghost                                # two slots fused on
    seen[800:830, 2] = ghost + 0.01                         # one wandering ghost
    seen[800:] = seen[800:][:, [1, 0, 2]]                   # lasting swap
    degraded = np.zeros(n, bool)
    degraded[800:830] = True

    perm, joins, _refusals = c30.stitch(seen, np.arange(n), degraded=degraded)
    bridges = [j for j in joins if j.get("evidence") == "bridge"]
    assert len(bridges) == 1
    assert bridges[0]["zone"] == [800, 829]
    assert bridges[0]["permutation"] == [1, 0, 2]
    fixed = seen[np.arange(n)[:, None], perm]
    assert np.nanmax(np.linalg.norm(fixed[830:] - world[830:], axis=-1)) < 1.0


def test_bridge_refuses_when_endpoints_are_ambiguous():
    """Two animals 1.2 mm apart with identical motion on both sides of a ghost
    zone: no evidence can name the connection, so none is invented."""
    rng = np.random.default_rng(11)
    world = _ghost_world()
    n = world.shape[0]
    world[:, 1] = world[:, 0] + [0.0, 1.2, 0.0]             # parallel, same speed
    seen = world + rng.normal(0.0, 0.05, world.shape)
    ghost = world[800:830, 1] + rng.normal(0.0, 30.0, (30, 3))
    seen[800:830, 0] = np.nan
    seen[800:830, 1] = ghost
    seen[800:830, 2] = ghost + 0.01
    degraded = np.zeros(n, bool)
    degraded[800:830] = True

    perm, joins, refusals = c30.stitch(seen, np.arange(n), degraded=degraded)
    assert not any(j.get("evidence") == "bridge" for j in joins)
    assert any(r.get("reason") == "zone unvouched"
               and r.get("zone") == [800, 829] for r in refusals)
    assert (perm[810:] == [0, 1, 2]).all()

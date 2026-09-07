"""`c34`'s out-and-back repair on trajectories whose right answer is constructed.

Three failure modes are planted and exactly one may be touched: a one-frame
flip that reverts (repair it), a lasting exchange (the stitch's jurisdiction -
leave it), and a triangulation spike (identity is its own best assignment -
leave it)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

c34 = pytest.importorskip("c34_microflip")

N = 1_000
FLIP_AT, FLIP = 300, (1, 0, 2)          # one frame, reverts
TWO_AT, TWO = 500, (2, 1, 0)            # two frames, reverts
LASTING_AT, LASTING = 700, (0, 2, 1)    # never reverts
SPIKE_AT = 900


def _world() -> np.ndarray:
    t = np.arange(N) / 166.0
    centres = np.array([[50.0, 50, 50], [140.0, 60, 120], [60.0, 150, 90]])
    xyz = np.empty((N, 3, 3))
    for s in range(3):
        xyz[:, s] = centres[s] + 10 * np.stack(
            [np.sin(t + s), np.cos(t + 2 * s), np.sin(0.5 * t + s)], axis=-1)
    return xyz


def _observed(world: np.ndarray) -> np.ndarray:
    seen = world + np.random.default_rng(3).normal(0.0, 0.25, world.shape)
    seen[FLIP_AT] = seen[FLIP_AT][list(FLIP)]
    seen[TWO_AT:TWO_AT + 2] = seen[TWO_AT:TWO_AT + 2][:, list(TWO)]
    seen[LASTING_AT:] = seen[LASTING_AT:][:, list(LASTING)]
    seen[SPIKE_AT, 2] += [40.0, 0.0, 0.0]
    return seen


def test_windows_are_exactly_the_reverting_flips():
    seen = _observed(_world())
    windows = c34.find_windows(seen, np.arange(N))
    found = {(w["first_frame"], w["last_frame"]): tuple(w["permutation"])
             for w in windows}
    assert found == {(FLIP_AT, FLIP_AT): FLIP, (TWO_AT, TWO_AT + 1): TWO}


def test_repair_restores_continuity_and_touches_nothing_else():
    world = _world()
    seen = _observed(world)
    windows = c34.find_windows(seen, np.arange(N))
    fixed = c34.apply_windows({"xyz": seen}, windows)["xyz"]

    # The reverting flips are gone: those frames match the world again.
    for frame in (FLIP_AT, TWO_AT, TWO_AT + 1):
        assert np.linalg.norm(fixed[frame] - world[frame], axis=-1).max() < 2.0
    # The lasting exchange and the spike are untouched, byte for byte.
    assert np.array_equal(fixed[LASTING_AT:], seen[LASTING_AT:])
    assert np.array_equal(fixed[SPIKE_AT], seen[SPIKE_AT])

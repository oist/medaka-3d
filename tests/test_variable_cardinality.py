"""The correspondence when a camera reports four detections, or two.

`c25`'s claim is that a false positive costs **nothing but itself**: the three
animals keep their slots, and the extra detection is left in no slot at all,
refused by geometry in millimetres. That is the property `c23` measured the old
arrangement fails — with three labels for four detections, a reflection takes a
real animal's label and the animal goes unlabelled.

The fixture rig is a pinhole (see `conftest.py`), so three synthetic fish are
projected exactly and a fourth point can be placed where no pair of rays meets:
a detection that is geometrically impossible, which is what a reflection in the
tank wall is.
"""

from __future__ import annotations

import sys
from math import perm
from pathlib import Path

import numpy as np
import pytest

from conftest import project

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

c25 = pytest.importorskip("c25_variable_cardinality")

TRIO = np.array([[40.0, 60.0, 30.0], [120.0, 90.0, 100.0], [70.0, 150.0, 160.0]])
N_FRAMES = 8
NODES = ("pectoral", "mouth")


def layout(rig, xyz, extra=None):
    """Build `c25`'s compact layout directly, without going through a detector.

    ``xyz`` is (n_frames, 3, 3) world mm. ``extra`` is ``view -> (frames, uv)``
    adding a fourth detection to that view on those frames, carrying its own 2D
    track label — which is what an uncapped tracker does with a reflection.
    """
    k_max = 4
    uv, label_at, k = {}, {}, {}
    for view in rig.views:
        points = project(rig.camera(view), xyz)  # (n_frames, 3, 2)
        values = np.full((N_FRAMES, k_max, 2), np.nan)
        values[:, :3] = points
        labels = np.tile(np.array([0, 1, 2, c25.EMPTY]), (N_FRAMES, 1)).astype(np.int32)
        count = np.full(N_FRAMES, 3, np.int8)
        if extra is not None and view in extra:
            frames, pixel = extra[view]
            values[frames, 3] = pixel
            labels[frames, 3] = 3
            count[frames] = 4
        # Both nodes sit on the same point here: the correspondence is what is
        # under test, not whether two nodes disagree.
        uv[view] = {node: values.copy() for node in NODES}
        label_at[view], k[view] = labels, count
    return uv, label_at, k


def solve_layout(rig, uv, label_at, k, *, mu=4.0, lam=1.0):
    rays = c25.compact_rays(rig, uv, NODES)
    frames = np.arange(N_FRAMES)
    chosen, objective, inverse, tables, _margin, _pruned = c25.solve(
        rays, NODES, label_at, k, frames, lam, mu, base=8)
    labels = np.full((N_FRAMES, 3, len(rig.views)), c25.EMPTY, np.int32)
    for t in range(N_FRAMES):
        table = tables[inverse[t]][chosen[t]]  # (n_slots, n_views)
        for view_index, view in enumerate(rig.views):
            pick = table[:, view_index]
            labels[t, :, view_index] = np.where(
                pick >= 0, label_at[view][t, np.where(pick >= 0, pick, 0)], c25.EMPTY)
    return labels, objective


def test_view_assignments_are_injective_and_maximal():
    for k in range(6):
        rows = c25.view_assignments(k)
        assert rows.shape == (perm(k, 3) if k >= 3 else perm(3, k), 3)
        for row in rows:
            used = row[row >= 0]
            assert used.size == min(k, 3), "a present label was left unused"
            assert np.unique(used).size == used.size, "one label filled two slots"


def test_state_table_is_the_product_over_views():
    assert c25.state_table((3, 3, 3)).shape == (6 ** 3, 3, 3)
    assert c25.state_table((4, 3, 3)).shape == (24 * 6 * 6, 3, 3)
    assert c25.state_table((2, 3, 3)).shape == (6 * 6 * 6, 3, 3)


def test_three_clean_animals_are_recovered(rig):
    """With three detections everywhere, every slot holds one animal, always."""
    xyz = np.repeat(TRIO[None], N_FRAMES, axis=0)
    xyz = xyz + np.linspace(0, 2, N_FRAMES)[:, None, None]
    labels, _ = solve_layout(rig, *layout(rig, xyz))

    assert (labels >= 0).all(), "a slot was left empty although every view had three"
    for slot in range(3):
        triple = labels[:, slot, :]
        assert (triple == triple[0]).all(), "a slot changed label with nothing to explain it"
    assert len({tuple(labels[0, slot]) for slot in range(3)}) == 3


def test_a_fourth_detection_is_refused_and_costs_no_animal_its_slot(rig):
    """The property `c23` found the cap fails: a false positive takes a label.

    The fourth detection is placed where no other camera has anything, so no
    pairing including it can triangulate — a reflection in the tank wall.
    """
    xyz = np.repeat(TRIO[None], N_FRAMES, axis=0)
    xyz = xyz + np.linspace(0, 2, N_FRAMES)[:, None, None]
    view = rig.views[1]
    frames = np.array([3, 4, 5])
    clean, _ = solve_layout(rig, *layout(rig, xyz))

    spurious = project(rig.camera(view), np.array([[300.0, -250.0, 400.0]]))[0]
    uv, label_at, k = layout(rig, xyz, extra={view: (frames, spurious)})
    labels, _ = solve_layout(rig, uv, label_at, k)

    index = rig.views.index(view)
    assert (labels[frames][:, :, index] != 3).all(), "the reflection took an animal's slot"
    assert (labels == clean).all(), "a real animal lost its label to the reflection"


def test_a_missing_detection_costs_one_slot_one_view(rig):
    """A view holding two labels keeps the other animals' rays and its own slots."""
    xyz = np.repeat(TRIO[None], N_FRAMES, axis=0)
    uv, label_at, k = layout(rig, xyz)
    view = rig.views[2]
    frames = np.array([2, 3])
    for node in NODES:
        uv[view][node][frames, 2] = np.nan
    label_at[view][frames, 2] = c25.EMPTY
    k[view][frames] = 2

    labels, _ = solve_layout(rig, uv, label_at, k)
    index = rig.views.index(view)
    empty = labels[frames][:, :, index] == c25.EMPTY
    assert empty.sum(axis=1).tolist() == [1, 1], "exactly one slot should lose this view"
    others = [i for i in range(len(rig.views)) if i != index]
    assert (labels[frames][:, :, others] >= 0).all(), "another view lost a label too"

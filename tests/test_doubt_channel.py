"""`c31`'s detectors and `c33`'s repair on cases whose right answer is constructed.

The pinhole fixture rig projects three known animals exactly (see
`tests/conftest.py`), so a contaminated ray can be planted in one named view
and the leave-one-out repair checked against the true position - not against
another estimate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import project

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

c31 = pytest.importorskip("c31_doubt_channel")
c33 = pytest.importorskip("c33_ray_rejection")
c01 = pytest.importorskip("c01_s16_predictions")

VIEWS = ("top_cam", "side1_cam", "side2_cam")
NODES = ("mouth", "pectoral", "tail")
N_FRAMES = 6

#: Three animals well apart, in the fixture rig's 192 x 192 x 186 volume.
WORLD = np.array([
    [[60.0, 60.0, 60.0], [62.0, 60.0, 60.0], [64.0, 60.0, 60.0]],
    [[140.0, 70.0, 120.0], [142.0, 70.0, 120.0], [144.0, 70.0, 120.0]],
    [[70.0, 150.0, 90.0], [72.0, 150.0, 90.0], [74.0, 150.0, 90.0]],
])  # (animal, node, 3)


def _detections(rig, corrupt: dict) -> dict:
    """Per view: every animal every frame, with named (view, frame, animal)
    pixel rows replaced by another animal's projection."""
    out = {}
    for view in VIEWS:
        camera = rig.camera(view)
        pixels = np.tile(project(camera, WORLD)[None], (N_FRAMES, 1, 1, 1))
        for (v, frame, animal), impostor in corrupt.items():
            if v == view:
                pixels[frame, animal] = project(camera, WORLD[impostor])
        out[view] = c01.Detections(
            experiment="synthetic", view=view,
            frame_idx=np.repeat(np.arange(N_FRAMES, dtype=np.int32), 3),
            points=pixels.reshape(-1, 3, 2).astype(np.float32),
            score=np.full(N_FRAMES * 3, 1.9, np.float32),
            n_frames=N_FRAMES, nodes=NODES)
    return out


def _arm(residual_flags: list[tuple[int, int]]) -> dict:
    xyz = np.tile(WORLD[None, :, 1, :], (N_FRAMES, 1, 1)).astype(np.float64)
    residual = np.full((N_FRAMES, 3), 0.3)
    for frame, slot in residual_flags:
        residual[frame, slot] = 10.0
    return {
        "xyz": xyz, "residual_mm": residual,
        "slot_rays": np.full((N_FRAMES, 3), 3, np.int8),
        "source": np.tile(np.arange(3, dtype=np.int8)[None, :, None],
                          (N_FRAMES, 1, len(VIEWS))),
    }


def test_repair_drops_the_contaminated_ray_and_restores_the_point(rig):
    detections = _detections(rig, {("top_cam", 2, 1): 2})
    fixed, stats = c33.repair(_arm([(2, 1)]), detections, rig)

    assert stats == {"flagged": 1, "adopted": 1, "not_sharp": 0, "not_clean": 0,
                     "no_geometry": 0,
                     "dropped_view": {"top_cam": 1, "side1_cam": 0, "side2_cam": 0}}
    assert np.linalg.norm(fixed["xyz"][2, 1] - WORLD[1, 1]) < 0.05
    assert fixed["slot_rays"][2, 1] == 2
    assert fixed["source"][2, 1, 0] == c33.UNRESOLVED
    # Nothing else moved.
    untouched = np.ones((N_FRAMES, 3), bool)
    untouched[2, 1] = False
    assert (fixed["slot_rays"][untouched] == 3).all()


def test_repair_refuses_when_two_views_disagree(rig):
    """Two contaminated views leave no sharp culprit - the frame stays honest."""
    detections = _detections(rig, {("top_cam", 2, 1): 2, ("side1_cam", 2, 1): 0})
    fixed, stats = c33.repair(_arm([(2, 1)]), detections, rig)
    assert stats["adopted"] == 0
    assert stats["not_sharp"] + stats["not_clean"] == 1
    assert fixed["slot_rays"][2, 1] == 3


def test_spike_detector_fires_on_out_and_back_only():
    xyz = np.tile(WORLD[None, :, 1, :], (20, 1, 1)).astype(float)
    xyz[10, 0] += [30.0, 0.0, 0.0]              # out and back
    xyz[15:, 1] += [30.0, 0.0, 0.0]             # lasting shift
    ok = np.ones(20, bool)
    flagged = c31.spikes(xyz, ok)
    assert flagged[10] and flagged.sum() == 1


def test_exchange_detector_fires_on_a_swap():
    xyz = np.tile(WORLD[None, :, 1, :], (20, 1, 1)).astype(float)
    xyz[12:, [0, 1]] = xyz[12:, [1, 0]]
    flagged = c31.exchange_boundaries(xyz, np.ones(20, bool))
    assert flagged[12] and flagged.sum() == 1

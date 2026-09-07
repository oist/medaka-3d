"""Tracklets must stop where the evidence stops, and must not stop anywhere else."""

from __future__ import annotations

import numpy as np
import pytest

from medaka3d.track2d import UNTRACKED, build, confident_links


def straight(n_frames: int, step: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Three animals swimming apart in parallel — the unambiguous case."""
    frames = np.repeat(np.arange(n_frames), 3)
    base = np.asarray([[100.0, 100.0], [100.0, 400.0], [100.0, 700.0]])
    shift = np.asarray([step, 0.0])
    points = np.concatenate([base + shift * f for f in range(n_frames)])
    return frames, points


def test_parallel_swimmers_make_one_tracklet_generation():
    frames, points = straight(50)
    result = build(frames, points, 50, experiment="E", view="top_cam",
                   gate_px=60.0, margin_px=30.0)
    assert result.count == 3, "no refusal is warranted, so no new IDs should be minted"
    assert (result.tracklet != UNTRACKED).all()
    assert sorted(result.lengths().tolist()) == [50, 50, 50]
    assert result.cuts == {"cardinality": 0, "gate": 0, "ambiguity": 0}


def test_a_frame_without_three_detections_cuts():
    frames, points = straight(20)
    keep = ~((frames == 10) & (np.arange(frames.size) % 3 == 0))
    result = build(frames[keep], points[keep], 20, experiment="E", view="top_cam",
                   gate_px=60.0, margin_px=30.0)
    assert result.cuts["cardinality"] > 0
    assert result.count == 6, "the run before and the run after are different tracklets"
    assert (result.tracklet[frames[keep] == 10] == UNTRACKED).all()


def test_a_teleport_cuts_on_the_gate_not_silently():
    frames, points = straight(20)
    points[frames >= 10] += np.asarray([500.0, 0.0])
    result = build(frames, points, 20, experiment="E", view="top_cam",
                   gate_px=60.0, margin_px=30.0)
    assert result.cuts["gate"] == 1
    assert result.cuts["ambiguity"] == 0
    assert result.count == 6


def test_the_ambiguity_rule_is_what_stops_a_coin_toss():
    """Two animals meeting at one point: the linker must decline, not guess.

    Without the margin the winner and the runner-up are equally good and the
    linker still answers. That answer is the swap this module exists to prevent,
    so the two settings are compared on the same data.
    """
    n = 21
    frames = np.repeat(np.arange(n), 3)
    rows = []
    for f in range(n):
        offset = abs(10 - f) * 1.0  # two animals converge at frame 10 and part
        rows.append([[300.0 - offset, 100.0], [300.0 + offset, 100.0], [100.0, 700.0]])
    points = np.concatenate([np.asarray(r) for r in rows])

    guessing = build(frames, points, n, experiment="E", view="top_cam",
                     gate_px=60.0, margin_px=0.0)
    declining = build(frames, points, n, experiment="E", view="top_cam",
                      gate_px=60.0, margin_px=5.0)

    assert guessing.cuts["ambiguity"] == 0
    assert guessing.count == 3, "with no margin the linker always has an answer"
    assert declining.cuts["ambiguity"] > 0
    assert declining.count > guessing.count, "declining must cost tracklets, visibly"


def test_permutation_is_applied_in_the_right_direction():
    """Animals that exchange places keep their IDs through the permutation.

    A linker that applied the permutation the wrong way round would look correct
    on parallel swimmers and silently relabel here, which is exactly the failure
    mode being engineered against.
    """
    frames = np.repeat(np.arange(2), 3)
    points = np.asarray([
        [0.0, 0.0], [10.0, 0.0], [0.0, 100.0],       # frame 0
        [11.0, 0.0], [1.0, 0.0], [1.0, 100.0],       # frame 1, rows 0 and 1 swapped
    ])
    result = build(frames, points, 2, experiment="E", view="top_cam",
                   gate_px=60.0, margin_px=1.0)
    assert result.count == 3
    first, second = result.tracklet[:3], result.tracklet[3:]
    assert first[0] == second[1], "the animal at x=0 moved to row 1"
    assert first[1] == second[0]
    assert first[2] == second[2]


def test_shapes_and_sorting_are_enforced():
    frames, points = straight(4)
    with pytest.raises(ValueError, match="ragged"):
        build(frames, points[:-1], 4, experiment="E", view="top_cam",
              gate_px=60.0, margin_px=1.0)
    with pytest.raises(ValueError, match="sorted"):
        build(frames[::-1], points, 4, experiment="E", view="top_cam",
              gate_px=60.0, margin_px=1.0)
    with pytest.raises(ValueError, match="must be"):
        confident_links(np.zeros((4, 2, 2)), np.ones(4, bool), gate_px=1.0, margin_px=0.0)
    with pytest.raises(ValueError, match="negative"):
        confident_links(np.zeros((4, 3, 2)), np.ones(4, bool), gate_px=1.0, margin_px=-1.0)


def test_cuts_account_for_every_unlinked_pair():
    frames, points = straight(30)
    points[frames >= 15] += np.asarray([500.0, 0.0])
    result = build(frames, points, 30, experiment="E", view="top_cam",
                   gate_px=60.0, margin_px=30.0)
    link, _, cuts = confident_links(
        np.stack([points[frames == f] for f in range(30)]),
        np.ones(30, bool), gate_px=60.0, margin_px=30.0,
    )
    assert sum(cuts.values()) == int((~link).sum())
    assert sum(result.cuts.values()) == 29 - int(link.sum()) + result.cuts["cardinality"]

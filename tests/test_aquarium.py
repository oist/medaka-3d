"""Diagnostic replays must preserve source identity, missingness and native jumps."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import project
from medaka3d.aquarium import (
    NODES,
    detect_events,
    gather_detections,
    movie_indices,
    reconstruct_pose,
    trail_at,
)


def _cache(path, **changes):
    contents = {
        "experiment": np.array("test"),
        "view": np.array("top_cam"),
        "nodes": np.array(NODES),
        "frame_idx": np.array([0, 0, 2, 2, 2, 4]),
        "n_frames": np.array(5),
        "points": np.arange(36, dtype=float).reshape(6, 3, 2),
    }
    contents.update(changes)
    np.savez(path, **contents)
    return contents["points"]


def _stationary(n_frames):
    return np.tile([[20., 30., 40.], [80., 90., 100.], [140., 150., 160.]],
                   (n_frames, 1, 1))


def test_gather_uses_frame_local_rows_and_preserves_slot_order(tmp_path):
    path = tmp_path / "detections.npz"
    points = _cache(path)
    frames = np.array([2, 0, 3, 4])
    rows = np.array([[2, 0, 1], [1, -1, 0], [-1, -1, -1], [0, -1, -1]])

    result = gather_detections(path, "test", "top_cam", frames, rows)

    np.testing.assert_array_equal(result[0], points[[4, 2, 3]])
    np.testing.assert_array_equal(result[1, [0, 2]], points[[1, 0]])
    np.testing.assert_array_equal(result[3, 0], points[5])
    assert np.isnan(result[rows < 0]).all()
    # Neither spatial position nor detector order may relabel a replay slot.
    assert result.shape == (4, 3, 3, 2)


@pytest.mark.parametrize("rows", [
    np.array([[3, 0, 1]]),  # Frame 2 has only rows 0, 1 and 2.
    np.array([[0., 1., 2.]]),
    np.array([[0, 1]]),
])
def test_gather_rejects_invalid_source_rows(tmp_path, rows):
    path = tmp_path / "detections.npz"
    _cache(path)
    with pytest.raises(ValueError, match="row"):
        gather_detections(path, "test", "top_cam", np.array([2]), rows)


@pytest.mark.parametrize(("change", "message"), [
    ({"experiment": np.array("other")}, "identity mismatch"),
    ({"view": np.array("side1_cam")}, "identity mismatch"),
    ({"nodes": np.array(["tail", "pectoral", "mouth"])}, "skeleton order"),
    ({"frame_idx": np.array([0, 2, 0, 2, 2, 4])}, "Unsorted"),
])
def test_gather_rejects_mismatched_cache(tmp_path, change, message):
    path = tmp_path / "detections.npz"
    _cache(path, **change)
    with pytest.raises(ValueError, match=message):
        gather_detections(path, "test", "top_cam", np.array([2]), np.array([[0, 1, 2]]))


def test_single_native_frame_out_and_back_survives_movie_downsampling():
    anchors = _stationary(166)
    frames = np.arange(830, 996)
    anchors[1, 1] += [12., 0., 0.]

    movie = movie_indices(len(frames), 166, 30)
    assert 1 not in movie
    np.testing.assert_array_equal(anchors[movie], _stationary(30))

    events = detect_events(anchors, frames, 166)
    assert [(event["frame"], event["slot"], event["type"]) for event in events] == [
        (831, 1, "jump"), (832, 1, "jump"),
    ]
    assert [event["distance_mm"] for event in events] == [12., 12.]
    assert [event["previous_frame"] for event in events] == [830, 831]
    assert events[0]["clip_seconds"] == pytest.approx(1 / 166)
    assert events[0]["source_seconds"] == pytest.approx(5 + 1 / 166)
    np.testing.assert_array_equal(events[0]["end_xyz"], anchors[1, 1])
    np.testing.assert_array_equal(events[1]["end_xyz"], anchors[2, 1])


def test_gap_is_reappearance_and_missing_data_are_never_fabricated():
    anchors = _stationary(8)
    anchors[2:4, 0] = np.nan
    anchors[4:, 0] += [50., 0., 0.]
    anchors[2:5, 1, 0] = np.nan  # A partially missing point is also unavailable.
    anchors[5:, 2] = np.nan  # Disappearance alone is not an event.
    original = anchors.copy()
    events = detect_events(anchors, np.arange(100, 108), 166)

    assert [(event["type"], event["slot"], event["frame"], event["missing_frames"])
            for event in events] == [("reappearance", 0, 104, 2),
                                     ("reappearance", 1, 105, 3)]
    assert events[0]["distance_mm"] == 50.
    assert events[1]["distance_mm"] == 0.
    assert not any(event["type"] == "jump" for event in events)
    trails = trail_at(anchors, 7, 8, events, 100)
    for slot, trail in enumerate(trails):
        np.testing.assert_array_equal(
            trail[np.isfinite(trail).all(-1)],
            original[:, slot][np.isfinite(original[:, slot]).all(-1)],
        )
        assert np.isnan(trail).any()
    np.testing.assert_array_equal(anchors, original)


def test_trails_break_teleport_lines_without_dropping_native_points_or_relabeling():
    anchors = _stationary(4)
    anchors[1, 2] += [10., 0., 0.]
    original = anchors.copy()
    events = detect_events(anchors, np.arange(500, 504), 166)
    trails = trail_at(anchors, 3, 4, events, 500)

    np.testing.assert_array_equal(trails[0], anchors[:, 0])
    np.testing.assert_array_equal(trails[1], anchors[:, 1])
    assert trails[2].shape == (6, 3)
    assert np.isnan(trails[2][[1, 3]]).all()
    np.testing.assert_array_equal(trails[2][[0, 2, 4, 5]], anchors[:, 2])
    # A jump before the visible trail must not affect another slot or time window.
    trimmed = trail_at(anchors, 3, 2, events, 500)
    for slot in range(3):
        np.testing.assert_array_equal(trimmed[slot], anchors[2:, slot])
    np.testing.assert_array_equal(anchors, original)


@pytest.mark.parametrize("seconds", [10, 20, 30])
def test_movie_duration_and_native_sample_timestamps(seconds):
    indices = movie_indices(seconds * 166, 166, 30)
    assert len(indices) == seconds * 30
    assert len(indices) / 30 == seconds
    assert np.issubdtype(indices.dtype, np.integer)
    assert indices[0] == 0
    assert indices[-1] < seconds * 166
    assert (np.diff(indices) > 0).all()
    # Every displayed pose belongs to the nearest actual source frame.
    assert np.max(np.abs(indices / 166 - np.arange(len(indices)) / 30)) <= .5 / 166


def test_noncontiguous_native_frames_are_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        detect_events(_stationary(3), np.array([10, 11, 13]), 166)


def test_reconstruction_preserves_recorded_anchor_and_missing_geometry(rig, tmp_path):
    frames = np.arange(2)
    pose = _stationary(2)[:, :, None, :] + np.array([[-3., 0., 0.],
                                                   [0., 0., 0.],
                                                   [9., 0., 0.]])
    anchors = pose[:, :, 1].copy()
    anchors[0, 0] += [.35, 0., 0.]  # Recorded anchor wins over retriangulation.
    anchors[1, 2] = np.nan
    source = np.empty((2, 3, 3), dtype=int)
    for vi, (view, order) in enumerate(zip(rig.views, ([2, 0, 1], [1, 2, 0], [0, 2, 1]),
                                           strict=True)):
        uv = project(rig.camera(view), pose)
        uv[0, 1, 0] = np.nan  # Missing mouth stays missing, although anchor is valid.
        _cache(tmp_path / f"test__{view}.npz", view=np.array(view),
               frame_idx=np.repeat(frames, 3), n_frames=np.array(2),
               points=uv[:, order].reshape(6, 3, 2))
        source[:, :, vi] = np.argsort(order)

    reconstructed, _, chest, _ = reconstruct_pose(source, anchors, frames,
                                                   tmp_path, "test", tmp_path)

    np.testing.assert_array_equal(reconstructed[:, :, 1], anchors)
    np.testing.assert_allclose(chest, pose[:, :, 1], atol=1e-6)
    np.testing.assert_allclose(reconstructed[:, :, 2][np.isfinite(anchors).all(-1)],
                               pose[:, :, 2][np.isfinite(anchors).all(-1)], atol=1e-6)
    assert np.isnan(reconstructed[0, 1, 0]).all()
    assert np.isnan(reconstructed[1, 2]).all()

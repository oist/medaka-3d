"""Behavioral tests for the experimental detection-preserving 2D comparison."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def tracking():
    path = Path(__file__).resolve().parents[1] / "labs/20260917_custom2d_comparison/tracking.py"
    spec = importlib.util.spec_from_file_location("_custom2d_comparison_tracking", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def detections(positions, *, keep=None, first=70000):
    """Input order deliberately changes; truth identifies synthetic fish only."""
    frames, points, truth = [], [], []
    for t, fish in enumerate(np.asarray(positions, dtype=float)):
        for identity in np.roll(np.arange(len(fish)), t % len(fish)):
            if keep is not None and not keep(t, int(identity)):
                continue
            anchor = fish[identity]
            frames.append(first + t)
            points.append(anchor + np.array([[4, 0], [0, 0], [-12, 0]]))
            truth.append(identity)
    return np.asarray(frames, np.int64), np.asarray(points), np.asarray(truth, np.int64)


def partition(labels):
    return labels[:, None] == labels[None, :]


def assert_kept(result, frames):
    assert result.labels.dtype == np.int64
    assert result.labels.shape == frames.shape
    assert np.all(result.labels >= 0)
    for frame in np.unique(frames):
        rows = result.labels[frames == frame]
        assert np.unique(rows).size == rows.size
    assert result.metadata["retained_detections"] == len(frames)
    assert result.metadata["identity_accuracy_claim"] is False


def test_c_crossing_between_frames_uses_motion_and_row_order_is_irrelevant(tracking):
    t = np.arange(20)
    xy = np.stack([np.c_[-47.5 + 5 * t, 0 * t], np.c_[47.5 - 5 * t, 0 * t]], axis=1)
    frames, points, truth = detections(xy)
    result = tracking.track_bidirectional(frames, points, view="side1_cam")
    assert_kept(result, frames)
    np.testing.assert_array_equal(partition(result.labels), partition(truth))
    order = np.concatenate([np.flatnonzero(frames == frame)[::-1] for frame in np.unique(frames)])
    shuffled = tracking.track_bidirectional(frames[order], points[order], view="side1_cam")
    np.testing.assert_array_equal(partition(result.labels[order]), partition(shuffled.labels))


def test_c_exact_overlap_declines_identity_guess_without_dropping_detections(tracking):
    t = np.arange(17)
    xy = np.stack([np.c_[-40 + 5 * t, 0 * t], np.c_[40 - 5 * t, 0 * t]], axis=1)
    frames, points, _ = detections(xy)
    result = tracking.track_bidirectional(frames, points, view="side1_cam")
    assert_kept(result, frames)
    assert any(e["kind"] == "ambiguous_directional_assignment" for e in result.events)
    assert result.metadata["output_tracklets"] > 2
    assert result.metadata["retained_links"] > 0


def test_c_variable_counts_and_three_missing_frames_recover_without_three_fish_cap(tracking):
    t = np.arange(18)
    xy = np.stack([np.c_[2 * t + 110 * i, 0 * t] for i in range(5)], axis=1)

    def keep(frame, fish):
        return not (fish == 0 and 7 <= frame <= 9) and not (fish >= 2 and frame < 4)

    frames, points, truth = detections(xy, keep=keep)
    result = tracking.track_bidirectional(frames, points, view="top_cam")
    assert_kept(result, frames)
    np.testing.assert_array_equal(partition(result.labels), partition(truth))
    counts = np.unique(frames, return_counts=True)[1]
    assert min(counts) == 2 and max(counts) == 5


@pytest.mark.parametrize("arm", ["B", "C"])
def test_long_missing_gap_starts_new_track_in_native_frame_coordinates(tracking, arm):
    frames = np.array([100, 101, 102, 108, 109, 110], dtype=np.int64)
    points = np.tile(np.array([[4.0, 0], [0, 0], [-12, 0]]), (6, 1, 1))
    if arm == "B":
        result = tracking.cut_sleap_links(frames, points, np.zeros(6, np.int64), view="side2_cam")
    else:
        result = tracking.track_bidirectional(frames, points, view="side2_cam")
    assert_kept(result, frames)
    assert result.labels[0] == result.labels[2]
    assert result.labels[3] == result.labels[5]
    assert result.labels[2] != result.labels[3]


@pytest.mark.parametrize("motion", ["stationary", "turn", "u_turn"])
def test_stationary_and_small_real_turns_are_not_cut_just_for_direction(tracking, motion):
    step = np.tile([2.0, 0], (24, 1))
    if motion == "stationary":
        step[:] = 0
    elif motion == "turn":
        step[12:] = [0, 2]
    else:
        step[12:] = [-2, 0]
    frames, points, _ = detections(np.cumsum(step, axis=0)[:, None, :])
    for result in (
        tracking.cut_sleap_links(frames, points, np.zeros(len(frames), np.int64), view="side1_cam"),
        tracking.track_bidirectional(frames, points, view="side1_cam"),
    ):
        assert_kept(result, frames)
        assert len(np.unique(result.labels)) == 1


def test_b_small_lateral_jump_with_stable_flanks_is_not_just_c24b_distance_gate(tracking):
    t = np.arange(20)
    path = np.c_[2.0 * t, np.zeros_like(t)]
    path[10:, 1] += 14
    frames, points, _ = detections(path[:, None, :])
    result = tracking.cut_sleap_links(
        frames, points, np.zeros(len(frames), np.int64), view="side1_cam"
    )
    assert_kept(result, frames)
    event = next(e for e in result.events if e["to_frame"] == frames[10])
    assert "lateral_jump_between_consistent_flanks" in event["reasons"]
    assert "hard_speed_gate" not in event["reasons"]
    assert result.labels[9] != result.labels[10]
    assert result.metadata["cut_links"] < result.metadata["retained_links"]


def test_b_one_frame_out_and_back_splits_both_edges_and_ignores_tail(tracking):
    t = np.arange(25)
    path = np.c_[2.0 * t, 0.0 * t]
    path[12, 1] += 32
    frames, points, _ = detections(path[:, None, :])
    baseline = np.zeros(len(frames), np.int64)
    result = tracking.cut_sleap_links(frames, points, baseline, view="side1_cam")
    assert {e["to_frame"] for e in result.events} >= {frames[12], frames[13]}
    assert len(set(result.labels[[11, 12, 13]])) == 3
    noisy = points.copy()
    noisy[:, 2] = np.random.default_rng(20).normal(0, 10000, (len(frames), 2))
    tail_result = tracking.cut_sleap_links(frames, noisy, baseline, view="side1_cam")
    np.testing.assert_array_equal(result.labels, tail_result.labels)


def test_b_future_flank_can_refuse_wrong_prefix_when_other_detection_fits(tracking):
    t = np.arange(12)
    xy = np.stack([np.c_[2.0 * t, 0.0 * t], np.c_[200 + 2.0 * t, 0.0 * t]], axis=1)
    frames, points, truth = detections(xy)
    # At the first frame ID0 selects a misplaced row; another observed row lies
    # at the position predicted backwards from its later stable observations.
    first_zero = np.flatnonzero((frames == frames.min()) & (truth == 0))[0]
    first_one = np.flatnonzero((frames == frames.min()) & (truth == 1))[0]
    points[first_one] = points[first_zero]
    points[first_zero, :, 1] += 30
    result = tracking.cut_sleap_links(frames, points, truth, view="side1_cam")
    event = next(e for e in result.events if e["from_row"] == first_zero)
    assert event["forward_prediction_error_px"] is None
    assert "reverse_prediction_prefers_other_detection" in event["reasons"]


def test_b_label_value_changes_cannot_change_the_partition_or_merge_original_ids(tracking):
    t = np.arange(20)
    xy = np.stack([np.c_[2.0 * t, 0.0 * t], np.c_[200 + t, 0.0 * t]], axis=1)
    frames, points, truth = detections(xy)
    points[(frames == frames[0] + 10) & (truth == 0), :, 1] += 30
    a = tracking.cut_sleap_links(frames, points, truth, view="top_cam")
    b = tracking.cut_sleap_links(frames, points, np.array([800, 17])[truth], view="top_cam")
    np.testing.assert_array_equal(a.labels, b.labels)
    assert not np.any(partition(a.labels) & ~partition(truth))


def test_c_time_reversal_and_absolute_frame_offset_preserve_links(tracking):
    t = np.arange(15)
    xy = np.stack([np.c_[2 * t, 0 * t], np.c_[150 - 3 * t, 40 + 0 * t]], axis=1)
    frames, points, _ = detections(xy)
    result = tracking.track_bidirectional(frames, points, view="top_cam")
    shifted = tracking.track_bidirectional(frames + 1000, points, view="top_cam")
    np.testing.assert_array_equal(result.labels, shifted.labels)
    order = np.argsort(-frames, kind="stable")
    reversed_result = tracking.track_bidirectional(
        frames.max() - frames[order], points[order], view="top_cam"
    )
    np.testing.assert_array_equal(
        partition(result.labels[order]), partition(reversed_result.labels)
    )


@pytest.mark.parametrize("arm", ["B", "C"])
def test_missing_primary_nodes_keep_every_detection_and_cannot_use_tail_as_id(tracking, arm):
    frames, points, truth = detections(np.zeros((12, 1, 2)))
    points[6, :2] = np.nan
    before = points.copy()
    if arm == "B":
        result = tracking.cut_sleap_links(frames, points, truth, view="top_cam")
    else:
        result = tracking.track_bidirectional(frames, points, view="top_cam")
    assert_kept(result, frames)
    assert np.count_nonzero(result.labels == result.labels[6]) == 1
    np.testing.assert_array_equal(points, before)


def test_c_missing_mouth_and_wild_tail_keep_reliable_pectoral_track(tracking):
    frames, points, _ = detections(np.c_[np.arange(20), np.zeros(20)][:, None])
    points[::2, 0] = np.nan
    points[:, 2] = np.random.default_rng(4).normal(0, 1000, (20, 2))
    result = tracking.track_bidirectional(frames, points, view="side2_cam")
    assert_kept(result, frames)
    assert result.metadata["output_tracklets"] == 1


def test_duplicate_and_negative_baseline_labels_are_retained_without_same_frame_collisions(
    tracking,
):
    frames = np.array([10, 10, 11, 11, 12, 12], np.int64)
    points = np.tile(np.array([[4.0, 0], [0, 0], [-12, 0]]), (6, 1, 1))
    result = tracking.cut_sleap_links(frames, points, np.array([0, 0, 0, -1, 0, 1]), view="top_cam")
    assert_kept(result, frames)
    assert any("duplicate_baseline_label_in_frame" in e["reasons"] for e in result.events)


def test_empty_and_unsorted_input_contract(tracking):
    result = tracking.track_bidirectional(
        np.zeros(0, np.int64), np.empty((0, 3, 2)), view="top_cam"
    )
    assert_kept(result, np.zeros(0, np.int64))
    with pytest.raises(ValueError, match="sorted"):
        tracking.track_bidirectional(np.array([3, 2]), np.zeros((2, 3, 2)), view="top_cam")


@pytest.mark.parametrize("speed", [0.0, 1.0, 2.0])
def test_c_recovers_stable_track_after_one_ambiguous_extra_detection(tracking, speed):
    rng = np.random.default_rng(1729)
    frames, points = [], []
    for frame in range(80):
        anchor = np.array([speed * frame, 0]) + rng.normal(0, 0.08, 2)
        pose = anchor + np.array([[4.0, 0], [0, 0], [-12, 0]])
        frames.append(1000 + frame)
        points.append(pose)
        if frame == 20:
            # An indistinguishable extra detection creates one ambiguous frame.
            # It must not make an isolated, smooth fish ambiguous forever.
            frames.append(1000 + frame)
            points.append(pose.copy())
    frames, points = np.asarray(frames), np.asarray(points)
    result = tracking.track_bidirectional(frames, points, view="side1_cam")
    assert_kept(result, frames)
    assert np.unique(result.labels[frames >= 1040]).size == 1
    assert np.unique(result.labels[frames < 1010]).size == 1
    assert result.metadata["output_tracklets"] <= 5

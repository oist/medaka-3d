"""Contract tests for conservative, detection-preserving clean 2D tracklets."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def tracker():
    path = Path(__file__).resolve().parents[1] / "labs/20260917_clean_tracklets/tracking.py"
    spec = importlib.util.spec_from_file_location("_clean_tracklets_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def detections(positions, keep=None):
    frames, points, truth = [], [], []
    for t, coordinates in enumerate(np.asarray(positions, float)):
        for identity in np.roll(np.arange(len(coordinates)), t % len(coordinates)):
            if keep is not None and not keep(t, identity):
                continue
            frames.append(75270 + t)
            points.append(coordinates[identity] + np.array([[6, 0], [0, 0], [-18, 0]]))
            truth.append(identity)
    return np.array(frames, np.int64), np.asarray(points), np.array(truth)


def straight(n=20, fish=3, speed=1.2):
    t = np.arange(n)
    return np.stack([np.c_[speed * t + 60 * i, 0 * t] for i in range(fish)], axis=1)


def partition(labels):
    return labels[:, None] == labels[None, :]


def assert_contract(result, frames):
    assert result.labels.shape == result.clean_edge.shape == frames.shape
    assert result.labels.dtype == np.int64 and result.clean_edge.dtype == bool
    assert np.all(result.labels >= 0)
    assert result.metadata["retained_detections"] == len(frames)
    assert result.metadata["sleap_tracking_ids_used"] is False
    assert result.metadata["identity_accuracy_claim"] is False
    assert result.metadata["tail_used_for_identity"] is False
    for frame in np.unique(frames):
        held = result.labels[frames == frame]
        assert len(np.unique(held)) == len(held)
    for label in np.unique(result.labels):
        rows = np.flatnonzero(result.labels == label)
        assert not result.clean_edge[rows[0]]
        assert np.all(result.clean_edge[rows[1:]])
        assert np.all(np.diff(frames[rows]) == 1)
    assert result.metadata["clean_links"] == result.clean_edge.sum()


def flow_for(frames, points, truth):
    result = {}
    for name, step in (("forward", 1), ("backward", -1)):
        predicted = np.full_like(points, np.nan)
        valid = np.zeros(points.shape[:2], bool)
        for row, (frame, fish) in enumerate(zip(frames, truth, strict=True)):
            other = np.flatnonzero((frames == frame + step) & (truth == fish))
            if len(other):
                predicted[row] = points[other[0]]
                valid[row] = np.isfinite(predicted[row]).all(-1)
        result[f"predicted_{name}"] = predicted
        result[f"valid_{name}"] = valid
    return result


@pytest.mark.parametrize("mode", ["pectoral", "nodes"])
def test_clear_motion_preserves_all_three_fish_and_exact_edge_meaning(tracker, mode):
    frames, points, truth = detections(straight())
    result = tracker.build_tracklets(frames, points, view="side1_cam", feature_mode=mode)
    assert_contract(result, frames)
    np.testing.assert_array_equal(partition(result.labels), partition(truth))
    assert result.clean_edge.sum() == 3 * 19


def test_crossing_cuts_only_the_nearby_fish_and_never_guesses_overlap_identity(tracker):
    t = np.arange(25)
    xy = np.stack([np.c_[t - 10, 0 * t], np.c_[10 - t, 0 * t], np.c_[80 + t, 0 * t]], 1)
    frames, points, truth = detections(xy)
    result = tracker.build_tracklets(frames, points, view="side2_cam")
    assert_contract(result, frames)
    assert len(np.unique(result.labels[truth == 2])) == 1
    near = (np.abs(frames - 75280) <= 3) & (truth != 2)
    assert not result.clean_edge[near].any()
    for fish in (0, 1):
        assert len(np.unique(result.labels[truth == fish])) > 2
    assert result.metadata["unsafe_reasons"]["nearby_detection"] > 0


def test_variable_counts_missing_fish_and_empty_native_frame_preserve_other_tracks(tracker):
    frames, points, truth = detections(
        straight(fish=4),
        keep=lambda t, fish: not (fish == 0 and 7 <= t <= 8) and not (fish == 3 and t < 4),
    )
    result = tracker.build_tracklets(frames, points, view="top_cam")
    assert_contract(result, frames)
    assert len(np.unique(result.labels[truth == 0])) == 2
    for fish in (1, 2, 3):
        assert len(np.unique(result.labels[truth == fish])) == 1
    # Native 166 fps coordinates, not array positions, define continuity.
    moved = frames.copy()
    moved[moved >= 75280] += 1
    broken = tracker.build_tracklets(moved, points, view="top_cam")
    assert_contract(broken, moved)
    assert not broken.clean_edge[moved == 75281].any()


@pytest.mark.parametrize(
    "node,reason", [(0, "large_primary_step"), (2, "tail_pectoral_motion_disagrees")]
)
def test_transient_node_exchange_is_cut_locally_and_other_fish_continues(tracker, node, reason):
    frames, points, truth = detections(straight(n=14))
    affected = np.flatnonzero((frames == 75276) & (truth < 2))
    points[affected, node] = points[affected[::-1], node]
    result = tracker.build_tracklets(frames, points, view="side1_cam")
    assert_contract(result, frames)
    assert len(np.unique(result.labels[truth == 2])) == 1
    boundary = np.isin(frames, [75276, 75277]) & (truth < 2)
    assert not result.clean_edge[boundary].any()
    assert result.metadata["refusal_reasons"].get(reason, 0) > 0
    pectoral = tracker.build_tracklets(frames, points, view="side1_cam", feature_mode="pectoral")
    np.testing.assert_array_equal(partition(pectoral.labels), partition(truth))


def test_missing_primary_node_never_uses_tail_as_identity_fallback(tracker):
    frames, points, truth = detections(straight(n=12))
    affected = (frames == 75275) & (truth == 0)
    points[affected, 1] = np.nan
    for mode in ("pectoral", "nodes"):
        result = tracker.build_tracklets(frames, points, view="side1_cam", feature_mode=mode)
        assert_contract(result, frames)
        assert not result.clean_edge[affected].any()
        assert len(np.unique(result.labels[truth == 1])) == 1
    frames, points, truth = detections(straight(n=12))
    points[truth == 0, 0] = np.nan
    nodes = tracker.build_tracklets(frames, points, view="side1_cam")
    pectoral = tracker.build_tracklets(frames, points, view="side1_cam", feature_mode="pectoral")
    assert not nodes.clean_edge[truth == 0].any()
    assert len(np.unique(pectoral.labels[truth == 0])) == 1


def test_stationary_and_smooth_turn_do_not_get_a_spurious_heading(tracker):
    t = np.arange(25)
    xy = np.stack([np.c_[0 * t, 0 * t], np.c_[100 + 8 * np.sin(t / 6), 8 * np.cos(t / 6)]], 1)
    frames, points, truth = detections(xy)
    result = tracker.build_tracklets(frames, points, view="side1_cam")
    np.testing.assert_array_equal(partition(result.labels), partition(truth))


def test_invalid_or_one_way_flow_does_not_force_an_alternative_connection(tracker):
    frames, points, _ = detections(straight(n=12))
    baseline = tracker.build_tracklets(frames, points, view="side1_cam")
    flow = {
        f"{kind}_{direction}": (
            np.full_like(points, 1e6) if kind == "predicted" else np.zeros(points.shape[:2], bool)
        )
        for kind in ("predicted", "valid")
        for direction in ("forward", "backward")
    }
    invalid = tracker.build_tracklets(frames, points, view="side1_cam", flow=flow)
    np.testing.assert_array_equal(partition(baseline.labels), partition(invalid.labels))
    flow["valid_forward"][:] = True
    one_way = tracker.build_tracklets(frames, points, view="side1_cam", flow=flow)
    np.testing.assert_array_equal(partition(baseline.labels), partition(one_way.labels))
    assert one_way.metadata["clean_links_with_flow"] == 0


def test_valid_but_disagreeing_flow_cuts_only_affected_fish(tracker):
    frames, points, truth = detections(straight(n=12))
    flow = flow_for(frames, points, truth)
    row = np.flatnonzero((frames == 75275) & (truth == 0))[0]
    flow["predicted_forward"][row, :2, 0] += 15
    result = tracker.build_tracklets(frames, points, view="side1_cam", flow=flow)
    assert_contract(result, frames)
    assert len(np.unique(result.labels[truth == 1])) == 1
    assert not result.clean_edge[(frames == 75276) & (truth == 0)].any()
    assert result.metadata["refusal_reasons"]["bidirectional_flow_disagrees"] > 0


def test_paired_primary_flow_can_resolve_motion_admissible_counterpassing(tracker):
    xy = np.array([[[-6, 0], [6, 0]], [[6, 0], [-6, 0]]], float)
    frames, points, truth = detections(xy)
    flow = flow_for(frames, points, truth)
    result = tracker.build_tracklets(frames, points, view="side1_cam", flow=flow)
    assert_contract(result, frames)
    np.testing.assert_array_equal(partition(result.labels), partition(truth))
    assert result.metadata["clean_links_with_flow"] == 2
    # Accurate tail LK alone supplies no positive identity evidence.
    for direction in ("forward", "backward"):
        flow[f"valid_{direction}"][:, :2] = False
    tail_only = tracker.build_tracklets(frames, points, view="side1_cam", flow=flow)
    plain = tracker.build_tracklets(frames, points, view="side1_cam")
    np.testing.assert_array_equal(partition(tail_only.labels), partition(plain.labels))


def test_row_shuffle_and_time_reversal_preserve_the_partition_with_flow(tracker):
    frames, points, truth = detections(straight(n=18, fish=4))
    flow = flow_for(frames, points, truth)
    baseline = tracker.build_tracklets(frames, points, view="top_cam", flow=flow)
    for reverse in (False, True):
        changed = frames.min() + frames.max() - frames if reverse else frames
        order = np.concatenate([np.flatnonzero(changed == f)[::-1] for f in np.unique(changed)])
        held = {}
        for direction, opposite in (("forward", "backward"), ("backward", "forward")):
            for kind in ("valid", "predicted"):
                held[f"{kind}_{direction}"] = flow[f"{kind}_{opposite if reverse else direction}"][
                    order
                ]
        result = tracker.build_tracklets(changed[order], points[order], view="top_cam", flow=held)
        assert_contract(result, changed[order])
        np.testing.assert_array_equal(partition(result.labels), partition(baseline.labels[order]))


def test_tail_cohesion_cannot_break_a_tie_between_primary_hypotheses(tracker):
    xy = np.array([[[-6, 0], [6, 0]], [[0, -6], [0, 6]]], float)
    frames, points, truth = detections(xy)
    # Only the tail distinguishes the two mappings; it must not create confidence.
    points[truth == 1, 2] = points[truth == 1, 1] + np.array([18, 0])
    result = tracker.build_tracklets(frames, points, view="side1_cam")
    assert_contract(result, frames)
    assert not result.clean_edge.any()
    assert result.metadata["refusal_reasons"]["assignment_margin"] > 0


def test_flow_never_overrides_primary_motion_gate(tracker):
    xy = np.array([[[0, 0]], [[70, 0]]], float)
    frames, points, truth = detections(xy)
    result = tracker.build_tracklets(
        frames, points, view="side1_cam", flow=flow_for(frames, points, truth)
    )
    assert not result.clean_edge.any()
    assert result.metadata["refusal_reasons"]["large_primary_step"] > 0


def test_sparse_singletons_empty_and_invalid_inputs(tracker):
    result = tracker.build_tracklets(np.empty(0, np.int64), np.empty((0, 3, 2)), view="top_cam")
    assert_contract(result, np.empty(0, np.int64))
    frames, points, _ = detections(straight(n=3, fish=1))
    result = tracker.build_tracklets(frames * 2, points, view="side1_cam")
    assert not result.clean_edge.any()
    assert len(np.unique(result.labels)) == 3
    with pytest.raises(ValueError, match="sorted"):
        tracker.build_tracklets(frames[::-1], points, view="top_cam")
    with pytest.raises(ValueError, match="Flow"):
        tracker.build_tracklets(
            frames,
            points,
            view="top_cam",
            flow={"predicted_forward": points[:1], "valid_forward": np.ones((1, 3), bool)},
        )

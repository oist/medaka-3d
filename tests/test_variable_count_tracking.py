"""Structural evidence for independent P1; synthetic IDs are not real-data GT."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def tracker():
    path = (Path(__file__).resolve().parents[1]
            / "labs/20260917_variable_count_tracklets/tracking.py")
    spec = importlib.util.spec_from_file_location("_variable_count_tracking_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture(frames_xy):
    frames, points = [], []
    for frame, coordinates in enumerate(frames_xy):
        for x, y in coordinates:
            frames.append(frame)
            points.append(np.array([[x + 6, y], [x, y], [x - 18, y]], dtype=float))
    return (
        np.asarray(frames, dtype=np.int64),
        np.asarray(points, dtype=float).reshape(-1, 3, 2),
        np.arange(len(frames), dtype=np.int64) * 7 + 123,
    )


def run(tracker, frames_xy, **kwargs):
    data = fixture(frames_xy)
    return data, tracker.track(*data, view="side1_cam", **kwargs)


def edges(result):
    return {(edge["from_global_row"], edge["to_global_row"]) for edge in result.edges}


def contract(data, result):
    frames, points, rows = data
    assert result.labels.shape == frames.shape
    assert len(result.node_quality) == len(frames)
    assert result.metadata["retained_detections"] == len(frames)
    assert result.metadata["gap_links"] == 0
    assert not result.metadata["sleap_tracking_ids_used"]
    assert not result.metadata["identity_accuracy_claim"]
    lookup = {int(row): index for index, row in enumerate(rows)}
    for frame in np.unique(frames):
        ids = result.labels[frames == frame]
        assert len(np.unique(ids)) == len(ids)
    for edge in result.edges:
        a, b = lookup[edge["from_global_row"]], lookup[edge["to_global_row"]]
        assert frames[b] - frames[a] == 1
        assert np.isfinite(points[[a, b], 1]).all()
        assert result.labels[a] == result.labels[b]
        assert edge["forward"]["reason"] == edge["backward"]["reason"] == "directional_agreement"
    json.dumps({
        "edges": result.edges, "events": result.events, "predictions": result.predictions,
        "reconnects": result.reconnect_candidates, "quality": result.node_quality,
        "metadata": result.metadata,
    }, allow_nan=False)


@pytest.mark.parametrize("count", range(6))
def test_all_counts_zero_through_five_preserve_rows_and_clear_continuation(tracker, count):
    data, result = run(tracker, [[(80 * fish + frame, 10) for fish in range(count)]
                                 for frame in range(8)])
    contract(data, result)
    assert len(np.unique(result.labels)) == count
    assert len(result.edges) == count * 7


def test_variable_counts_and_zero_recovery_keep_visible_third_fish(tracker):
    counts = [3, 2, 1, 0, 2, 4, 3]
    data, result = run(tracker, [[(80 * fish + frame, 10) for fish in range(count)]
                                 for frame, count in enumerate(counts)])
    contract(data, result)
    frames, _, _ = data
    assert len(result.labels) == sum(counts)
    assert result.labels[0] == result.labels[3] == result.labels[5]
    assert not set(result.labels[frames < 3]) & set(result.labels[frames > 3])
    assert result.predictions and result.reconnect_candidates
    assert any(prediction["target_frame"] == 3 for prediction in result.predictions)
    assert all(not candidate["accepted"] for candidate in result.reconnect_candidates)


@pytest.mark.parametrize("auxiliary", [False, True])
def test_missing_mouth_tail_never_breaks_clear_pectoral(tracker, auxiliary):
    frames, points, rows = fixture([[(frame, 20)] for frame in range(9)])
    points[2:6, 0] = np.nan
    points[4:8, 2] = np.nan
    original = points.copy()
    result = tracker.track(frames, points, rows, "top_cam", auxiliary=auxiliary)
    contract((frames, points, rows), result)
    assert len(np.unique(result.labels)) == 1
    np.testing.assert_array_equal(points, original)
    assert result.node_quality[4]["missing_nodes"] == ["M", "T"]
    assert result.node_quality[4]["observed_mask"] == [False, True, False]


def test_missing_p_is_an_unresolved_retained_observation(tracker):
    frames, points, rows = fixture([[(frame, 20)] for frame in range(9)])
    points[4, 1] = np.nan
    result = tracker.track(frames, points, rows, "top_cam", auxiliary=True)
    contract((frames, points, rows), result)
    assert result.labels[4] not in result.labels[np.arange(9) != 4]
    assert result.labels[3] != result.labels[5]
    assert any(event["reason"] == "missing_pectoral" for event in result.events)
    assert any(c["from_global_row"] == rows[3] and c["to_global_row"] == rows[5]
               for c in result.reconnect_candidates)


def test_node_mixture_is_quality_evidence_not_fish_identity_veto(tracker):
    frames, points, rows = fixture([[(frame, 20)] for frame in range(9)])
    points[4, 0] += [150, 20]
    points[5, 2] += [35, 0]
    result = tracker.track(frames, points, rows, "side1_cam", auxiliary=True)
    assert len(np.unique(result.labels)) == 1
    assert "mouth_span_suspect" in result.node_quality[4]["geometry_flags"]
    assert "T_relative_motion_suspect" in result.node_quality[5]["temporal_cohesion_flags"]


def test_merge_competition_never_duplicates_or_merges_identity(tracker):
    data, result = run(tracker, [
        [(-5, 10), (5, 10), (100, 10)],
        [(-3, 10), (3, 10), (101, 10)],
        [(0, 10), (102, 10)],
        [(1, 10), (103, 10)],
    ])
    contract(data, result)
    assert len({result.labels[i] for i in (2, 5, 7, 9)}) == 1
    assert result.labels[6] not in {result.labels[3], result.labels[4]}
    assert any(event["reason"] in ("primary_distance_tie", "candidate_margin")
               for event in result.events)


def test_single_detection_has_multiple_open_endpoint_competitors(tracker):
    data, result = run(tracker, [[(-5, 10), (5, 10)], [(-3, 10), (3, 10)],
                                 [(0, 10)], [(1, 10)]])
    contract(data, result)
    _, _, rows = data
    assert (int(rows[4]), int(rows[5])) not in edges(result)
    refused = [event for event in result.events
               if event.get("frame") == 3 and event.get("direction") == "forward"]
    assert any(any(candidate["age_frames"] > 1 for candidate in event["candidates"])
               for event in refused)


def test_stable_extra_image_and_wall_side_fish_are_not_filtered(tracker):
    data, result = run(tracker, [[(1 + frame, 1), (100 + frame, 10), (200 + frame, 10),
                                 (300 + frame, 10)] for frame in range(10)])
    contract(data, result)
    assert len(np.unique(result.labels)) == 4
    assert all(item["reflection_status"] == "unknown" for item in result.node_quality)
    assert all(item["reflection_evidence"] == [] for item in result.node_quality)


def test_nearby_but_uniquely_moving_fish_can_continue(tracker):
    data, result = run(tracker, [[(0.5 * frame, 10), (6 + 0.5 * frame, 10)]
                                 for frame in range(10)])
    contract(data, result)
    assert len(np.unique(result.labels)) == 2


def test_unique_but_large_jump_cannot_connect(tracker):
    data, result = run(tracker, [[(0, 10)], [(1, 10)], [(500, 10)], [(501, 10)]])
    contract(data, result)
    assert result.labels[1] != result.labels[2]
    assert any(event["reason"] == "large_step" for event in result.events)


def test_crossing_and_later_separation_do_not_assign_entrance_identity(tracker):
    data, result = run(tracker, [[(frame - 8, 10), (8 - frame, 10), (200 + frame, 10)]
                                 for frame in range(18)])
    contract(data, result)
    assert len(np.unique(result.labels[2::3])) == 1
    assert not set(result.labels[:6]) & set(result.labels[-6:][[0, 1, 3, 4]])


def test_exact_ties_do_not_resolve_by_row_order(tracker):
    frames, points, rows = fixture([[(-1, 10), (1, 10)], [(0, 9), (0, 11)]])
    result = tracker.track(frames, points, rows, "side1_cam")
    assert len(result.edges) == 0
    assert any(event["reason"] == "primary_distance_tie" for event in result.events)


def test_forward_backward_disagreement_is_held(tracker):
    data, result = run(tracker, [[(x, 10)] for x in (0, 0, 9, 10)])
    contract(data, result)
    assert result.metadata["disagreement_links"] > 0
    assert any(event["kind"] == "directional_disagreement" for event in result.events)
    for event in result.events:
        if event["kind"] == "directional_disagreement":
            assert event["forward"] is not None and event["backward"] is not None
            assert event["forward"]["prediction_error_px"] is not None
            assert event["backward"]["prediction_error_px"] is not None


def test_shuffling_rows_preserves_global_connections_labels_and_masks(tracker):
    frames, points, rows = fixture([[(frame - 8, 10), (8 - frame, 10), (200 + frame, 10)]
                                   for frame in range(18)])
    points[8, 0] = np.nan
    points[17, 1] = np.nan
    original = points.copy()
    baseline = tracker.track(frames, points, rows, "side2_cam", auxiliary=True)
    permutation = np.random.default_rng(100).permutation(len(rows))
    shuffled = tracker.track(frames[permutation], points[permutation], rows[permutation],
                             "side2_cam", auxiliary=True)
    assert edges(baseline) == edges(shuffled)
    np.testing.assert_array_equal(baseline.labels[permutation], shuffled.labels)
    np.testing.assert_array_equal(points, original)
    assert [baseline.node_quality[i] for i in permutation] == shuffled.node_quality
    assert baseline.reconnect_candidates == shuffled.reconnect_candidates


def test_predictions_remain_separate_and_expire(tracker):
    frames, points, rows = fixture([[(0, 10)], [(1, 10)], *([[]] * 30), [(33, 10)]])
    original = points.copy()
    result = tracker.track(frames, points, rows, "side1_cam")
    contract((frames, points, rows), result)
    np.testing.assert_array_equal(original, points)
    assert len(result.labels) == 3
    assert result.reconnect_candidates == []
    assert all(record["age_frames"] <= tracker.P1Config().endpoint_lifetime_frames
               for record in result.predictions)
    assert all(record["observed"] is False for record in result.predictions)
    forward = [record for record in result.predictions if record["direction"] == "forward"]
    assert len(forward) > 2
    assert forward[0]["uncertainty_px"] < forward[-1]["uncertainty_px"]


def test_angle_alone_does_not_cut_real_turn(tracker):
    data, result = run(tracker, [[(0, 0)], [(3, 0)], [(3, 3)], [(3, 6)]])
    contract(data, result)
    assert len(result.edges) == 3
    assert any(edge[direction]["heading_review"] for edge in result.edges
               for direction in ("forward", "backward"))


def test_auxiliary_is_separate_and_missing_support_equivalent(tracker):
    frames, points, rows = fixture([[(frame, 20)] for frame in range(8)])
    points[:, (0, 2)] = np.nan
    p = tracker.track(frames, points, rows, "top_cam")
    aux = tracker.track(frames, points, rows, "top_cam", auxiliary=True)
    assert edges(p) == edges(aux)
    assert all(edge["forward"]["observed_support_nodes"] == [] for edge in aux.edges)
    assert p.metadata["auxiliary"] is False and aux.metadata["auxiliary"] is True


def test_declared_sensitivity_changes_only_margin(tracker):
    base = tracker.P1Config()
    configs = tracker.sensitivity_configs()
    assert configs["P1_margin_loose"] == replace(base, margin_fraction=0.015)
    assert configs["P1_margin_strict"] == replace(base, margin_fraction=0.035)
    assert base.endpoint_lifetime_frames == 17


def test_duplicate_global_rows_fail_loudly(tracker):
    frames, points, rows = fixture([[(0, 10)], [(1, 10)]])
    rows[1] = rows[0]
    with pytest.raises(ValueError, match="unique"):
        tracker.track(frames, points, rows, "side1_cam")

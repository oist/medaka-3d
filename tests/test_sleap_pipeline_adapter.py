"""Recovery, ambiguity, native SLEAP paths and all-row contracts for the adapter."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def adapter():
    path = (Path(__file__).resolve().parents[1]
            / "labs/20260918_sleap_pipeline_adapter/adapter.py")
    spec = importlib.util.spec_from_file_location("_sleap_pipeline_adapter_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def data(positions):
    frames, points = [], []
    for frame, xy in enumerate(positions):
        for x, y in xy:
            frames.append(frame)
            points.append(np.array([[x + 6, y + 2], [x, y], [x - 18, y - 4]], float))
    frames = np.asarray(frames, dtype=np.int64)
    points = np.asarray(points, float).reshape(-1, 3, 2)
    score = np.linspace(0.1, 0.9, len(frames))
    rows = np.arange(len(frames), dtype=np.int64) * 7 + 1000
    return frames, points, score, rows


def straight(n=30, count=3):
    return data([[(80 * fish + frame * 0.2, 100) for fish in range(count)] for frame in range(n)])


def assert_contract(values, result):
    frames, points, _, rows = values
    assert result.labels.shape == result.provisional_labels.shape == frames.shape
    assert len(result.events) == len(frames)
    assert np.all(result.labels >= 0)
    assert result.metadata["retained_detections"] == len(frames)
    assert result.metadata["gap_links"] == 0
    assert not result.metadata["provisional_identity_is_ground_truth"]
    assert not result.metadata["identity_accuracy_claim"]
    lookup = {int(row): index for index, row in enumerate(rows)}
    for frame in np.unique(frames):
        ids = result.labels[frames == frame]
        assert len(np.unique(ids)) == len(ids)
        provisional = result.provisional_labels[frames == frame]
        assigned = provisional[provisional >= 0]
        assert len(np.unique(assigned)) == len(assigned)
    for edge in result.edges:
        a, b = lookup[edge["from_global_row"]], lookup[edge["to_global_row"]]
        assert frames[b] - frames[a] == 1
        assert np.isfinite(points[[a, b], 1]).all()
        assert result.labels[a] == result.labels[b]
        assert edge["score"] >= result.metadata["config"]["match_threshold"]
        assert min(edge["row_margin"], edge["column_margin"]) >= edge["required_margin"]
    assert all(not candidate["accepted"] for candidate in result.reconnect_candidates)
    json.dumps({"events": result.events, "edges": result.edges,
                "reconnects": result.reconnect_candidates, "metadata": result.metadata},
               allow_nan=False)


@pytest.mark.parametrize("count", range(6))
def test_zero_through_five_clear_detections_remain_all_rows(adapter, count):
    values = straight(n=15, count=count)
    result = adapter.track(*values, "side1_cam")
    assert_contract(values, result)
    assert len(np.unique(result.labels)) == count
    assert result.metadata["birth_count"] == count
    assert len(result.edges) == 14 * count


def test_single_fish_nine_pixel_displacement_has_no_self_competition(adapter):
    values = data([[(0.2 * frame + (9 if frame >= 5 else 0), 100)] for frame in range(120)])
    result = adapter.track(*values, "side1_cam")
    assert_contract(values, result)
    assert len(np.unique(result.labels)) == 1
    assert result.metadata["provisional_tracks_created"] == 1
    assert result.metadata["max_active_tracks"] == 1
    assert all(event["confirmed"] for event in result.events[30:])


def test_soft_refusal_updates_same_history_and_recovers(adapter):
    values = data([[(0.2 * frame + (9 if frame >= 5 else 0), 100)] for frame in range(120)])
    config = replace(adapter.AdapterConfig(), margin=0.30)
    result = adapter.track(*values, "side1_cam", config=config)
    assert_contract(values, result)
    assert result.events[5]["reason"] == "insufficient_similarity_margin"
    assert result.metadata["provisional_tracks_created"] == 1
    assert len(np.unique(result.provisional_labels)) == 1
    assert all(event["confirmed"] for event in result.events[12:])
    assert len(np.unique(result.labels[12:])) == 1
    assert len(np.unique(result.labels)) < 8


def test_real_birth_does_not_regenerate_old_forecasts_each_frame(adapter):
    values = data([[(0.2 * frame + (80 if frame >= 5 else 0), 100)] for frame in range(120)])
    result = adapter.track(*values, "side1_cam")
    assert_contract(values, result)
    assert result.events[5]["reason"] == "dummy_unmatched"
    assert result.metadata["provisional_tracks_created"] == 2
    assert result.metadata["max_active_tracks"] == 2
    assert result.metadata["expired_tracks"] == 1
    assert all(event["confirmed"] for event in result.events[6:])


def test_zero_frame_and_reappearance_keep_gap_separate(adapter):
    counts = [3, 2, 1, 0, 2, 4, 3]
    values = data([[(80 * fish + frame, 100) for fish in range(count)]
                   for frame, count in enumerate(counts)])
    result = adapter.track(*values, "side1_cam")
    assert_contract(values, result)
    frames = values[0]
    assert not set(result.labels[frames < 3]) & set(result.labels[frames > 3])
    assert result.provisional_labels[0] == result.provisional_labels[6]
    assert result.reconnect_candidates
    assert result.metadata["zero_frames_between_observations"] == 1


def test_mouth_tail_missing_or_implausible_do_not_change_p_tracking(adapter):
    frames, points, score, rows = straight(n=15)
    baseline = adapter.track(frames, points, score, rows, "top_cam")
    points[::2, 0] = np.nan
    points[::3, 2] = np.nan
    points[1::4, 0] += 1000
    changed = adapter.track(frames, points, score, rows, "top_cam")
    np.testing.assert_array_equal(changed.labels, baseline.labels)
    assert changed.edges == baseline.edges


def test_missing_p_is_singleton_with_no_invented_history_observation(adapter):
    frames, points, score, rows = straight(n=10, count=1)
    points[4, 1] = np.nan
    original = points.copy()
    result = adapter.track(frames, points, score, rows, "side1_cam")
    assert_contract((frames, points, score, rows), result)
    assert result.provisional_labels[4] == -1
    assert result.labels[4] not in result.labels[np.arange(10) != 4]
    assert result.labels[3] != result.labels[5]
    assert result.provisional_labels[3] == result.provisional_labels[5]
    assert result.reconnect_candidates[0]["age_frames"] == 2
    assert all(4 not in candidate["history_frames"] for event in result.events
               for candidate in event["candidates"])
    np.testing.assert_array_equal(points, original)


def test_all_p_missing_keeps_all_observations(adapter):
    frames, points, score, rows = straight(n=6, count=2)
    points[:, 1] = np.nan
    result = adapter.track(frames, points, score, rows, "side1_cam")
    assert_contract((frames, points, score, rows), result)
    assert (result.provisional_labels == -1).all()
    assert len(np.unique(result.labels)) == len(frames)
    assert result.metadata["native_tracker_track_calls"] == 0


def test_expiry_never_reuses_provisional_identity(adapter):
    values = data([[(0, 100)], [(0.2, 100)], *([[]] * 25), [(0.4, 100)]])
    result = adapter.track(*values, "side1_cam")
    assert_contract(values, result)
    np.testing.assert_array_equal(result.provisional_labels, [0, 0, 1])
    assert result.metadata["expired_tracks"] == 1
    assert result.reconnect_candidates == []


def test_noncontiguous_active_ids_map_columns_back_to_actual_ids(adapter):
    values = data([[(0, 100), (100, 100)], [(100, 100)], *([[]] * 16),
                   [(100, 100)], [(100, 100)], [(300, 100)]])
    result = adapter.track(*values, "side1_cam")
    assert_contract(values, result)
    np.testing.assert_array_equal(result.provisional_labels, [0, 1, 1, 1, 1, 2])
    assert result.events[3]["candidates"][0]["provisional_id"] == 1
    assert result.metadata["provisional_tracks_created"] == 3


def test_expiry_filters_each_observation_not_just_track_last_seen(adapter):
    frames, points, score, rows = data([[(0, 100)]] * 5)
    frames[:] = [0, 1, 15, 16, 18]
    result = adapter.track(frames, points, score, rows, "side1_cam")
    assert result.events[-1]["candidates"][0]["history_frames"] == [1, 15, 16]
    assert result.metadata["maximum_scored_history_age_frames"] == 17
    assert result.metadata["expired_history_observations"] == 1


def test_history_is_native_sleap_mean_fixed_p_oks(adapter):
    values = data([[(0, 100)], [(3, 100)], [(4, 100)]])
    result = adapter.track(*values, "side1_cam")
    assert result.events[1]["score"] == pytest.approx(np.exp(-9 / 50))
    assert result.events[2]["score"] == pytest.approx((np.exp(-16 / 50) + np.exp(-1 / 50)) / 2)
    assert result.metadata["native_tracker_track_calls"] == 3
    assert result.metadata["native_get_scores_calls"] == 2
    assert result.metadata["native_hungarian_calls"] == 2


def test_two_histories_competing_for_one_row_do_not_merge_confirmed_ids(adapter):
    values = data([[(-5, 100), (5, 100), (200, 100)],
                   [(-3, 100), (3, 100), (201, 100)],
                   [(0, 100), (202, 100)], [(1, 100), (203, 100)]])
    result = adapter.track(*values, "side1_cam")
    assert_contract(values, result)
    assert not result.events[6]["confirmed"]
    assert result.events[6]["reason"] == "nonunique_or_nonmutual_candidate"
    assert result.labels[6] not in {result.labels[3], result.labels[4]}
    assert len(set(result.labels[[2, 5, 7, 9]])) == 1


def test_crossing_is_held_and_later_clear_fragments_recover(adapter):
    values = data([[(frame - 10, 100), (10 - frame, 100), (200 + frame, 100)]
                   for frame in range(40)])
    result = adapter.track(*values, "side1_cam")
    assert_contract(values, result)
    frames = values[0]
    assert any(not event["confirmed"] for event in result.events
               if event["frame"] == 10)
    assert len(np.unique(result.labels[2::3])) == 1
    assert all(event["confirmed"] for event in result.events if event["frame"] >= 30)
    later_crossing_fish = (frames > 30) & (np.arange(len(frames)) % 3 != 2)
    assert not set(result.labels[frames < 2]) & set(result.labels[later_crossing_fish])


def test_exact_ties_provisional_only_and_no_row_order_certification(adapter):
    values = data([[(-1, 100), (1, 100)], [(0, 99), (0, 101)]])
    result = adapter.track(*values, "side1_cam")
    assert len(result.edges) == 0
    assert (result.provisional_labels >= 0).all()
    assert all(event["reason"] == "nonunique_or_nonmutual_candidate"
               for event in result.events[2:])


def test_margin_dummy_limits_and_hard_step_gate_are_distinct(adapter):
    values = data([[(0, 100)], [(100, 100)]])
    config = replace(adapter.AdapterConfig(), oks_stddev=1.0)
    result = adapter.track(*values, "side1_cam", config=config)
    assert result.events[1]["score"] > config.match_threshold
    assert result.provisional_labels[0] == result.provisional_labels[1]
    assert result.events[1]["reason"] == "large_pectoral_step"
    assert result.labels[0] != result.labels[1]


def test_extra_smooth_image_is_not_removed_or_classified_as_reflection(adapter):
    values = straight(n=20, count=5)
    result = adapter.track(*values, "side1_cam")
    assert_contract(values, result)
    assert len(np.unique(result.labels)) == 5
    assert all(event["reflection_status"] == "unknown" for event in result.events)


def test_shuffling_preserves_labels_evidence_and_original_arrays(adapter):
    frames, points, score, rows = data([[(frame - 10, 100), (10 - frame, 100)]
                                      for frame in range(25)])
    points[8, 0] = np.nan
    points[15, 1] = np.nan
    original = points.copy()
    baseline = adapter.track(frames, points, score, rows, "top_cam")
    permutation = np.random.default_rng(20260918).permutation(len(frames))
    changed = adapter.track(frames[permutation], points[permutation], score[permutation],
                            rows[permutation], "top_cam")
    np.testing.assert_array_equal(changed.labels, baseline.labels[permutation])
    np.testing.assert_array_equal(changed.provisional_labels,
                                  baseline.provisional_labels[permutation])
    np.testing.assert_array_equal(points, original)
    assert changed.edges == baseline.edges
    assert changed.reconnect_candidates == baseline.reconnect_candidates
    assert changed.events == [baseline.events[i] for i in permutation]


def test_native_p_control_only_changes_nodes_and_scale_and_keeps_gap_ids(adapter):
    values = data([[(0, 100)], [], [(0.2, 100)]])
    native = adapter.native_pectoral(*values, "side1_cam")
    gated = adapter.track(*values, "side1_cam")
    assert len(np.unique(native.labels)) == 1
    assert native.metadata["gap_links"] == 1
    assert len(np.unique(gated.labels)) == 2
    assert len(set(gated.provisional_labels)) == 1
    assert native.metadata["dummy_used"] is False


def test_scorer_extension_does_not_change_native_global_registry(adapter):
    import sleap_nn.tracking.tracker as installed

    original = installed.compute_oks
    native = installed.Tracker.from_config()
    registry = dict(native._scoring_functions)
    adapter.track(*straight(n=3, count=1), "side1_cam")
    adapter.native_pectoral(*straight(n=3, count=1), "side1_cam")
    assert installed.compute_oks is original
    assert native._scoring_functions == registry
    assert "fixed_pectoral_oks" not in installed.Tracker.from_config()._scoring_functions


@pytest.mark.parametrize("age", [8, 17, 33])
def test_age_in_seconds_declares_native_frame_lifetime(adapter, age):
    config = replace(adapter.AdapterConfig(), max_age_seconds=age / 166)
    assert config.max_age_frames == age


def test_duplicate_global_rows_are_rejected(adapter):
    frames, points, score, rows = straight(n=2, count=1)
    rows[1] = rows[0]
    with pytest.raises(ValueError, match="unique"):
        adapter.track(frames, points, score, rows, "side1_cam")

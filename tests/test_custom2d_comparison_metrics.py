"""Check that comparison diagnostics cannot substitute fragmentation for accuracy."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def metrics():
    lab = Path(__file__).resolve().parents[1] / "labs/20260917_custom2d_comparison"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "path", sys.path.copy())
        patch.syspath_prepend(str(lab))
        for name in ("tracking", "downstream", "_custom2d_metrics_runner"):
            filename = "run.py" if name == "_custom2d_metrics_runner" else f"{name}.py"
            spec = importlib.util.spec_from_file_location(name, lab / filename)
            module = importlib.util.module_from_spec(spec)
            patch.setitem(sys.modules, name, module)
            spec.loader.exec_module(module)
        yield module


def point_rows(x):
    xy = np.c_[np.asarray(x, float), np.zeros(len(x))]
    return xy[:, None] + np.array([[[4.0, 0], [0, 0], [-12, 0]]])


def stage_at_rest(n=12):
    xyz = np.broadcast_to(np.array([[0.0, 0, 0], [40, 0, 0], [90, 0, 0]]), (n, 3, 3)).copy()
    source = np.broadcast_to(np.arange(9).reshape(3, 3), (n, 3, 3)).copy()
    pose = xyz[:, :, None] + np.array([[[[1.0, 0, 0], [0, 0, 0], [-1, 0, 0]]]])
    return dict(xyz=xyz, source=source, pose=pose, residual_mm=np.zeros((n, 3)))


def test_cutting_every_link_removes_jumps_but_exposes_fragmentation_and_keeps_detections(metrics):
    frames = np.arange(20, 26)
    points = point_rows(np.arange(6) * 80)
    connected = metrics.tracking_metrics(frames, points, np.zeros(6, int), 20, 26, "top_cam")
    cut = metrics.tracking_metrics(frames, points, np.arange(6), 20, 26, "top_cam")
    assert connected["large_link_steps"] == 5 and cut["large_link_steps"] == 0
    assert connected["connected_edges"] == 5 and cut["connected_edges"] == 0
    assert connected["fragments_in_core"] == 1 and cut["fragments_in_core"] == 6
    assert connected["new_tracks_in_core"] == 1 and cut["new_tracks_in_core"] == 6
    assert connected["detection_retention"] == cut["detection_retention"] == 1
    assert cut["motion_px"]["n"] == 0 and cut["motion_px"]["median"] is None
    assert cut["fragment_length_frames"]["maximum"] == 1


def test_gap_speed_gate_uses_native_dt_and_reports_hidden_frames(metrics):
    frames = np.array([100, 101, 104, 108])
    points = point_rows([0, 55, 165, 375])
    result = metrics.tracking_metrics(frames, points, np.zeros(4, int), 100, 109, "side1_cam")
    # 55/1 and 210/4 exceed 50 px/frame; 110/3 does not.
    assert result["large_link_steps"] == 2
    assert result["large_consecutive_steps"] == 1
    assert result["connected_edges"] == 3
    assert result["consecutive_edges"] == 1
    assert result["gap_links"] == 2
    assert result["hidden_frames_in_links"] == 5
    assert result["motion_px"]["maximum"] == 210
    # Length is observed frames, not interpolated span or hidden-frame credit.
    assert result["fragment_length_frames"]["maximum"] == 4


def test_tracking_counts_edges_ending_in_core_including_prefix_entry(metrics):
    frames = np.arange(99, 104)
    result = metrics.tracking_metrics(
        frames, point_rows([0, 1, 2, 3, 500]), np.zeros(5, int), 100, 103, "side1_cam"
    )
    assert result["detection_rows"] == 3
    assert result["connected_edges"] == 3
    assert result["new_tracks_in_core"] == 0
    assert result["large_link_steps"] == 0  # The suffix jump is outside the core.


def test_unassigned_rows_are_visible_and_nan_motion_is_not_stationary_motion(metrics):
    frames = np.arange(3)
    points = point_rows([0, 0, 0])
    dropped = metrics.tracking_metrics(frames, points, np.array([0, -1, 0]), 0, 3, "top_cam")
    assert dropped["detection_rows"] == 3
    assert dropped["assigned_rows"] == 2
    assert dropped["detection_retention"] == pytest.approx(2 / 3)
    assert dropped["gap_links"] == 1
    points[1, :2] = np.nan
    unknown = metrics.tracking_metrics(frames, points, np.zeros(3, int), 0, 3, "top_cam")
    assert unknown["connected_edges"] == 2
    assert unknown["motion_px"]["n"] == 0
    assert unknown["motion_px"]["median"] is None
    assert unknown["motion_px"]["maximum"] is None


def test_empty_core_has_no_invented_retention_or_motion(metrics):
    result = metrics.tracking_metrics(
        np.array([1]), point_rows([0]), np.array([0]), 10, 20, "side2_cam"
    )
    assert result["detection_rows"] == 0
    assert result["detection_retention"] is None
    assert result["motion_px"]["n"] == 0
    assert result["motion_px"]["median"] is None


def test_prefix_alignment_uses_one_permutation_for_all_fields_and_preserves_core_swap(metrics):
    reference = stage_at_rest(18)
    order = [2, 0, 1]
    stage = {k: v[:, order].copy() for k, v in reference.items()}
    # A real later swap affects every field consistently, after the prefix.
    for key in stage:
        stage[key][10:] = stage[key][10:, [1, 0, 2]]
    stage["frames"] = np.arange(200, 218)
    before = {k: v.copy() for k, v in stage.items()}
    result = metrics.fixed_prefix_alignment(stage, reference["xyz"], np.arange(6))
    assert result["valid"]
    np.testing.assert_array_equal(result["permutation"], [1, 2, 0])
    q = result["permutation"]
    for key in ("xyz", "source", "pose", "residual_mm"):
        np.testing.assert_array_equal(stage[key], before[key][:, q])
    np.testing.assert_array_equal(stage["frames"], before["frames"])
    np.testing.assert_array_equal(stage["xyz"][:6], reference["xyz"][:6])
    assert not np.array_equal(stage["xyz"][10:], reference["xyz"][10:])
    diagnostic = metrics.shape_metrics(stage, np.arange(18) >= 6)
    assert diagnostic["steps_over_10mm"] == 2
    assert diagnostic["alternative_permutation_edges"] == 1
    assert diagnostic["alternative_permutation_relative_frames"] == [4]


def test_alignment_ignores_core_even_when_it_strongly_prefers_another_naming(metrics):
    reference = stage_at_rest(105)
    stage = {key: value.copy() for key, value in reference.items()}
    for key in stage:
        stage[key][5:] = stage[key][5:, [1, 0, 2]]
    result = metrics.fixed_prefix_alignment(stage, reference["xyz"], np.arange(5))
    np.testing.assert_array_equal(result["permutation"], [0, 1, 2])
    np.testing.assert_array_equal(stage["xyz"][5:], reference["xyz"][5:, [1, 0, 2]])


def test_alignment_declines_when_either_prefix_side_lacks_five_joint_frames(metrics):
    reference = stage_at_rest(9)["xyz"]
    reference[:5, 0] = np.nan
    stage = stage_at_rest(9)
    before = {key: value.copy() for key, value in stage.items()}
    result = metrics.fixed_prefix_alignment(stage, reference, np.arange(9))
    assert result["valid"] is False
    assert result["frames_used"] == 4
    for key in stage:
        np.testing.assert_array_equal(stage[key], before[key])


def test_alignment_counts_joint_frames_not_missing_coordinates_as_zero(metrics):
    reference = stage_at_rest(10)["xyz"]
    reference[:2, 0] = np.nan
    stage = stage_at_rest(10)
    stage["xyz"][2:4, 1] = np.nan
    result = metrics.fixed_prefix_alignment(stage, reference, np.arange(10))
    assert result["valid"] is True
    assert result["frames_used"] == 6
    assert np.isfinite(result["median_distance_mm"])


def test_3d_missing_rows_remain_in_denominators_and_do_not_create_bridged_steps(metrics):
    stage = stage_at_rest(4)
    stage["xyz"][1, 0] = np.nan
    stage["pose"][1, 0] = np.nan
    stage["residual_mm"][1, 0] = np.nan
    stage["pose"][2, 2, 0] = np.nan
    stage["xyz"][2, 1, 0] += 12
    result = metrics.shape_metrics(stage, np.ones(4, bool))
    assert result["possible_fish_frames"] == 12
    assert result["observed_fish_frames"] == 11
    assert result["missing_fish_frames"] == 1
    assert result["availability"] == pytest.approx(11 / 12)
    assert result["possible_native_steps"] == 9
    assert result["observed_native_steps"] == 7
    assert result["step_mm"]["n"] == 7
    assert result["steps_over_10mm"] == 2
    assert result["residual_mm"]["n"] == 11
    np.testing.assert_array_equal(result["node_observed_counts"], [10, 11, 11])
    assert result["mouth_tail_length_mm"]["n"] == 10


def test_3d_core_edges_exclude_prefix_and_suffix_boundaries(metrics):
    stage = stage_at_rest(5)
    stage["xyz"][0] += 1000
    stage["xyz"][4] += 1000
    result = metrics.shape_metrics(stage, np.array([False, True, True, True, False]))
    assert result["native_frames"] == 3
    assert result["possible_native_steps"] == result["observed_native_steps"] == 6
    assert result["steps_over_4mm"] == 0
    assert result["step_mm"]["maximum"] == 0


def test_all_missing_3d_does_not_report_zero_distance_measurements(metrics):
    stage = stage_at_rest(3)
    for key in ("xyz", "pose", "residual_mm"):
        stage[key][:] = np.nan
    result = metrics.shape_metrics(stage, np.ones(3, bool))
    assert result["possible_fish_frames"] == result["missing_fish_frames"] == 9
    assert result["observed_fish_frames"] == 0
    assert result["possible_native_steps"] == 6
    assert result["observed_native_steps"] == 0
    assert result["step_mm"]["n"] == 0
    assert result["step_mm"]["maximum"] is None
    assert result["residual_mm"]["median"] is None
    assert result["mouth_tail_length_mm"]["median"] is None


def test_alternative_permutation_is_a_diagnostic_and_can_count_a_known_swap(metrics):
    stage = stage_at_rest(2)
    for key in stage:
        stage[key][1] = stage[key][1, [1, 0, 2]]
    result = metrics.shape_metrics(stage, np.ones(2, bool))
    assert result["steps_over_10mm"] == 2
    assert result["alternative_permutation_edges"] == 1
    assert "not confirmed swaps" in result["identity_metric_note"]

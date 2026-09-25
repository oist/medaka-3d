"""Shared identity scoring must expose gaps and ignore arbitrary slot naming."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from medaka3d.eval.generative import (
    MouthAnchor,
    match_mouth_anchors,
    mouth_reprojection_summary,
    observed_motion_summary,
    score_female_identity,
    skeleton_shape_summary,
)


def anchor(frame):
    return MouthAnchor(frame, {"side1_cam": 0, "side2_cam": 0}, {}, {})


def test_permanent_slot_rename_does_not_change_identity_score():
    source = np.tile(np.arange(3)[None, :, None], (3, 1, 3))
    observed = np.ones((3, 3), dtype=bool)
    source[2] = source[2, [1, 0, 2]]
    anchors = [anchor(i) for i in range(3)]
    first = score_female_identity(np.arange(3), source, observed, anchors)
    renamed = score_female_identity(np.arange(3), source[:, [2, 0, 1]], observed, anchors)
    assert first["interval_counts"] == renamed["interval_counts"]
    assert first["interval_counts"] == {"preserved": 1, "swap": 1, "unscored": 0}


def test_missing_anchor_is_not_silently_bridged():
    source = np.tile(np.arange(3)[None, :, None], (3, 1, 3))
    source[1] = -1
    result = score_female_identity(np.arange(3), source, np.ones((3, 3), bool),
                                   [anchor(i) for i in range(3)])
    assert result["requested_anchors"] == 3
    assert result["available_anchors"] == 2
    assert result["interval_counts"] == {"preserved": 0, "swap": 0, "unscored": 2}


def test_conflicting_views_or_review_abstain():
    source = np.tile(np.arange(3)[None, :, None], (2, 1, 3))
    source[0, :, 2] = [1, 0, 2]
    review = np.zeros((2, 3), bool)
    review[1, 0] = True
    result = score_female_identity(np.arange(2), source, np.ones((2, 3), bool),
                                   [anchor(0), anchor(1)], review=review)
    assert result["available_anchors"] == 0
    assert result["anchors"][0]["reason"] == "conflicting source identities"
    assert result["anchors"][1]["reason"] == "identity requires review"


def test_predicted_mouth_is_not_counted_as_an_observation():
    source = np.tile(np.arange(3)[None, :, None], (1, 1, 3))
    observed = np.ones((1, 3), bool)
    observed[0, 0] = False
    result = score_female_identity(np.arange(1), source, observed, [anchor(0)])
    assert result["available_anchors"] == 0


def test_unmatched_confirmed_click_stays_in_denominator():
    record = {"frame": 10, "gt_status": "confirmed",
              "side1_selection_status": "sleap_linked", "side1_mouth_x": 10,
              "side1_mouth_y": 10, "side2_selection_status": "not_selected"}
    anchors = match_mouth_anchors([record], lambda view, frame: np.array([[80, 80]]), 0, 20)
    assert len(anchors) == 1
    assert anchors[0].detection_rows == {}


def test_duplicate_annotations_are_rejected():
    record = {"frame": 10, "gt_status": "confirmed",
              "side1_selection_status": "not_selected", "side2_selection_status": "not_selected"}
    with pytest.raises(ValueError, match="duplicate"):
        match_mouth_anchors([record, record], lambda view, frame: np.empty((0, 2)), 0, 20)


def test_motion_does_not_bridge_missing_or_nonadjacent_frames():
    xyz = np.zeros((4, 3, 3))
    xyz[1, 0] = np.nan
    xyz[2:] = 100
    summary = observed_motion_summary(xyz, np.array([10, 11, 20, 21]))
    assert summary["scored_adjacent_steps"] == 5
    assert summary["steps_over_10mm"] == 0


def test_reprojection_scores_mouth_not_pectoral():
    pose = np.zeros((1, 3, 3, 3))
    pose[0, 0, 0] = [10, 20, 30]
    pose[0, 0, 1] = [90, 90, 90]
    anchored = MouthAnchor(8, {"side1_cam": 0}, {"side1_cam": (10.0, 20.0)}, {})
    summary = mouth_reprojection_summary(
        pose, np.array([8]), [anchored], {"anchors": [{"frame": 8, "identity": 0}]},
        lambda view, xyz: xyz[:2],
    )
    assert summary["view_observations"] == 1
    assert summary["max_px"] == 0


def test_cache_offsets_preserve_empty_frames_and_view_identity(tmp_path):
    path = (Path(__file__).resolve().parents[1]
            / "labs/20260908_generative_tracking/benchmark.py")
    spec = importlib.util.spec_from_file_location("_generative_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = tmp_path / "work/c01_s16_predictions/bottomup"
    cache.mkdir(parents=True)
    for index, view in enumerate(module.VIEWS):
        points = np.arange(18).reshape(3, 3, 2) + 100 * index
        np.savez(cache / f"test__{view}.npz", experiment="test", view=view,
                 frame_idx=np.array([0, 2, 2], dtype=np.int32), points=points,
                 score=np.ones(3), n_frames=4, nodes=np.array(module.NODES))
    detections = module.CachedDetections(tmp_path, "test")
    assert detections.points("top_cam", 1).shape == (0, 3, 2)
    assert detections.points("side2_cam", 3).shape == (0, 3, 2)
    np.testing.assert_array_equal(detections.points("side1_cam", 2),
                                  np.arange(6, 18).reshape(2, 3, 2) + 100)
    with pytest.raises(IndexError):
        detections.points("top_cam", 4)


def test_shape_diagnostic_includes_stretched_unobserved_predictions():
    initial = np.tile(np.array([[0., 0, 0], [6, 0, 0], [24, 0, 0]])[None], (3, 1, 1))
    pose = np.stack([initial, initial.copy()])
    pose[1, 0, 2] = [186, 0, 0]
    observed = np.ones((2, 3, 3), dtype=bool)
    observed[1, 0, 2] = False
    summary = skeleton_shape_summary(pose, observed, initial)
    fish = summary["slots"][0]
    assert fish["segments"]["pectoral-tail"] == {
        "initial_mm": 18.0, "maximum_mm": 180.0, "maximum_to_initial_ratio": 10.0,
        "finite_frames": 2,
    }
    assert fish["observed_nodes"]["tail"] == {"count": 1, "fraction": .5}
    assert fish["all_nodes_observed_frames"] == 1


def test_shape_missing_or_zero_reference_stays_null():
    initial = np.zeros((3, 3, 3))
    initial[1, 2] = np.nan
    pose = initial[None].copy()
    observed = np.isfinite(pose).all(axis=-1)
    summary = skeleton_shape_summary(pose, observed, initial)
    assert summary["slots"][0]["segments"]["mouth-pectoral"]["maximum_to_initial_ratio"] is None
    tail = summary["slots"][1]["segments"]["pectoral-tail"]
    assert tail["initial_mm"] is None
    assert tail["maximum_mm"] is None
    assert tail["finite_frames"] == 0


def test_shape_cannot_count_nonfinite_node_as_observed():
    initial = np.zeros((3, 3, 3))
    pose = initial[None].copy()
    pose[0, 1, 2] = np.nan
    with pytest.raises(ValueError, match="observed node"):
        skeleton_shape_summary(pose, np.ones((1, 3, 3), bool), initial)

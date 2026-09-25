"""Verify comparison denominators and identity-independent link accounting."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260917_clean2d_video_comparison"
sys.path.insert(0, str(LAB))
spec = importlib.util.spec_from_file_location("clean2d_direct_measurements", LAB / "run.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.pop(0)


def test_missing_rows_negative_labels_and_gaps_are_different():
    frames = np.array([8, 9, 10, 11, 11, 13, 14, 14])
    labels = np.array([5, 5, 5, 5, -1, 5, 5, -1])
    points = np.zeros((8, 3, 2))
    points[5, 0] = np.nan
    stats, _ = module.measure(frames, points, labels, 10, 15)
    assert stats["detection_rows"] == stats["retained_rows"] == 6
    assert stats["unassigned_rows"] == 2
    assert stats["unique_ids"] == 1
    assert stats["frames_without_detections"] == 1
    assert stats["contiguous_runs"] == 4
    assert stats["adjacent_links"] == 2
    assert stats["adjacent_link_denominator"] == 5
    assert stats["gap_links"] == 1
    assert stats["nodes"]["mouth"]["missing_rows"] == 1
    previous, gaps = module.predecessors(frames, labels)
    assert previous[4] == previous[7] == -1
    assert gaps[5] == 2


def test_label_renaming_does_not_change_edge_agreement():
    frames = np.array([10, 10, 11, 11, 12, 12])
    points = np.zeros((6, 3, 2))
    baseline = np.array([1, 2, 1, 2, 1, 2])
    previous, _ = module.predecessors(frames, baseline)
    stats, cuts = module.measure(
        frames, points, np.array([70, 90, 70, 90, 70, 90]), 10, 13, previous
    )
    assert stats["sleap_adjacent_reproduced"] == 4
    assert stats["sleap_adjacent_cut"] == 0
    assert stats["sleap_adjacent_different_predecessor"] == 0
    assert not cuts


def test_alternative_predecessor_is_not_counted_as_a_break():
    frames = np.array([10, 10, 11, 11, 12, 12])
    points = np.zeros((6, 3, 2))
    previous, _ = module.predecessors(frames, np.array([1, 2, 1, 2, 1, 2]))
    stats, cuts = module.measure(frames, points, np.array([8, 9, 9, 8, 10, 8]), 10, 13, previous)
    assert stats["sleap_adjacent_different_predecessor"] == 2
    assert stats["sleap_adjacent_cut"] == 1
    assert stats["sleap_adjacent_reproduced"] == 1
    assert len(cuts) == 3
    assert stats["contiguous_runs"] == 3


def test_same_id_twice_in_frame_rejected():
    with pytest.raises(ValueError, match="twice"):
        module.predecessors(np.array([10, 10]), np.array([4, 4]))


def test_metrics_exclude_warmup_link_and_do_not_call_steps_identity_truth():
    frames = np.array([8, 9, 10, 11])
    points = np.zeros((4, 3, 2))
    points[2:, 1, 0] = 100
    labels = np.ones(4, dtype=int)
    stats, _ = module.measure(frames, points, labels, 10, 12)
    assert stats["adjacent_links"] == stats["adjacent_link_denominator"] == 1
    assert stats["nodes"]["pectoral"]["steps_above_30px"] == 0
    assert stats["contiguous_runs"] == 1


def test_sensitivity_preserves_missing_node_and_no_gap_architecture():
    base = module.TrackingConfig()
    variants = module.sensitivity_configs()
    assert len(variants) == 6
    for cfg in variants.values():
        assert cfg.max_primary_span_fraction == base.max_primary_span_fraction
        assert cfg.min_primary_span_fraction == base.min_primary_span_fraction
        assert cfg.history_observations == base.history_observations
        assert cfg.flow_prediction_weight == base.flow_prediction_weight
    assert variants["F_motion_loose"].speed_fraction == 2 * base.speed_fraction
    assert variants["F_ambiguity_loose"].assignment_margin_fraction == (
        base.assignment_margin_fraction / 2
    )

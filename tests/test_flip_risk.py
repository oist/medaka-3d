"""Check temporal coverage, unknown endpoints, and permutation composition."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from medaka3d.eval.flip_risk import (
    capture_brackets,
    connection_changes,
    corrected_slots,
    endpoint_outcomes,
    group_change_brackets,
    paired_identity_time,
    reference_change_brackets,
)
from medaka3d.stages import Linker
from medaka3d.types import Instance3D

spec = importlib.util.spec_from_file_location(
    "test_geometry_motion_linker",
    Path(__file__).parents[1] / "labs/20261104_geometry_flip_risk/linker.py",
)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_sustained_changes_use_mixed_track_sections_and_keep_gap_bracket():
    times = np.arange(10) * 0.1
    slots = np.tile([0, 1, 2], (10, 1))
    slots[3:6, :2] = -1
    slots[6:, :2] = [1, 0]
    changes = reference_change_brackets(slots, times, np.full(10, 0.1), minimum_samples=2)
    assert len(changes) == 2
    groups = group_change_brackets(changes)
    assert len(groups) == 1
    assert groups.iloc[0].before_frame == 2
    assert groups.iloc[0].after_frame == 6
    narrow = pd.DataFrame([{"interval_id": 0, "start_s": 0.3, "end_s": 0.5}])
    capture = capture_brackets(groups, narrow)
    assert capture.warning_overlap.all()
    assert not capture.warning_contains_bracket.any()


def test_unmatched_and_known_wrong_time_are_distinct():
    before = np.array([[0, 1, 2], [1, 0, -1]])
    perm = np.array([[0, 1, 2], [1, 0, 2]])
    after = corrected_slots(before, perm)
    result = paired_identity_time(before, after, [0, 1, 2], np.array([0.1, 0.2]))
    assert np.isclose(result["corrected_seconds"], 0.4)
    assert np.isclose(result["unassessed_seconds"], 0.2)
    assert result["newly_wrong_seconds"] == 0
    assert after[1, 2] == -1


def test_partial_endpoint_is_unknown_not_automatic_wrong():
    decisions = [
        {
            "interval_id": 0,
            "before_frame": 0,
            "after_frame": 1,
            "permutation": [1, 0, 2],
            "decision": "corrected",
        }
    ]
    result = endpoint_outcomes(decisions, np.array([[0, 1, 2], [1, 0, -1]]))
    assert result.after_outcome.tolist() == ["correct", "correct", "unknown"]


def fixture():
    times = np.arange(301) * 0.01
    xyz = np.zeros((301, 3, 3))
    xyz[:, :, 0] = np.array([0, 40, 80])[None] + times[:, None] * 5
    xyz[111:, [0, 1]] = xyz[111:, [1, 0]]
    xyz[211:, [1, 2]] = xyz[211:, [2, 1]]
    intervals = pd.DataFrame(
        [
            {"interval_id": 0, "start_frame": 100, "stop_frame_exclusive": 111},
            {"interval_id": 1, "start_frame": 200, "stop_frame_exclusive": 211},
        ]
    )
    tracks = np.tile(np.arange(3), (301, 1))
    config = {
        "max_interval_s": 0.75,
        "history_s": 0.1,
        "minimum_span_s": 0.08,
        "minimum_samples": 10,
        "assignment_margin_mm": 5,
    }
    return xyz, times, tracks, intervals, config


def test_two_noncommuting_corrections_compose_and_protocol_runs():
    xyz, times, tracks, intervals, config = fixture()
    linker = lab.MotionBridge(times, intervals, tracks, config)
    assert isinstance(linker, Linker)
    linked = linker.link(
        [Instance3D(frame=f, slot=s, xyz=xyz[f, s]) for f in range(len(times)) for s in range(3)]
    )
    np.testing.assert_allclose(
        linked[220:, :, 0], np.array([0, 40, 80])[None] + times[220:, None] * 5
    )
    assert [row["decision"] for row in linker.decisions] == ["corrected", "corrected"]


def test_motion_does_not_cross_local_track_boundary():
    xyz, times, tracks, intervals, config = fixture()
    tracks[95:100, 0] = 8
    _, rows = lab.MotionBridge(times, intervals, tracks, config).permutation(xyz)
    assert rows[0]["decision"] == "refused"


def test_warnings_are_not_truncated_to_maximum_bridge_length():
    times = np.arange(100) * 0.01
    weights = np.r_[np.diff(times), 0]
    flag = np.zeros(100, bool)
    flag[10:90] = True
    intervals, mask = lab.warning_intervals(
        {"missing_position": flag}, np.full(100, 100), times, weights, 10, 0.05
    )
    assert len(intervals) == 1
    assert intervals.iloc[0].duration_s > 0.75
    assert mask[10:90].all()


def test_empty_reference_changes_are_supported():
    slots = np.full((5, 3), -1)
    changes = reference_change_brackets(slots, np.arange(5), np.ones(5))
    assert group_change_brackets(changes).empty


def test_connection_counts_do_not_confuse_persistent_wrong_naming_with_new_edges():
    before = np.array([[0, 1, 2], [1, 0, 2], [1, 0, 2], [1, 0, 2]])
    after = np.tile([0, 1, 2], (4, 1))
    rows, counts = connection_changes(before, after, np.arange(4), np.ones(4))
    assert len(rows) == 2
    assert counts.corrected_edges.sum() == 2
    assert counts.newly_wrong_edges.sum() == 0
    # An arbitrary fixed permutation does not change a connection score.
    renamed = np.array([2, 0, 1])[after]
    _, other = connection_changes(before, renamed, np.arange(4), np.ones(4))
    pd.testing.assert_frame_equal(counts, other)


def test_c30_adapter_keeps_c34_strict_frame_mask_and_restores_parameter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    adapter_spec = importlib.util.spec_from_file_location(
        "test_c30_risk_adapter",
        Path(__file__).parents[1] / "labs/20261104_geometry_flip_risk/c30_ablation.py",
    )
    module = importlib.util.module_from_spec(adapter_spec)
    adapter_spec.loader.exec_module(module)
    seen = {}
    old = module.c30.VELOCITY_NEEDS_MEDIAN_MM

    def fake_stitch(xyz, resolved, degraded):
        seen["threshold"] = module.c30.VELOCITY_NEEDS_MEDIAN_MM
        seen["c30_frames"] = resolved
        return np.tile(np.arange(3), (len(xyz), 1)), [], []

    def fake_windows(xyz, resolved):
        seen["c34_frames"] = resolved
        return []

    monkeypatch.setattr(module.c30, "stitch", fake_stitch)
    monkeypatch.setattr(module.c34, "find_windows", fake_windows)
    xyz = np.zeros((5, 3, 3))
    xyz[2, 0] = np.nan
    module.PermissiveVelocity(5, np.arange(5), None).permutation(xyz)
    assert seen["threshold"] == 0
    np.testing.assert_array_equal(seen["c30_frames"], np.arange(5))
    np.testing.assert_array_equal(seen["c34_frames"], [0, 1, 3, 4])
    assert old == module.c30.VELOCITY_NEEDS_MEDIAN_MM

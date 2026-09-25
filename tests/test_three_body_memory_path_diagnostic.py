"""Check direct-node reassignment and links that persist beyond latent gaps."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
SPEC = importlib.util.spec_from_file_location(
    "memory_path_diagnostic", WORK / "diagnose_identity_memory_paths.py"
)
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


def geometry():
    pose = np.zeros((4, 3, 3, 3), dtype=np.float64)
    pose[:, :, :, 0] = np.arange(3)[None, :, None] * 10
    pose[:, 0, :, 0] += np.arange(4)[:, None]
    return dict(
        frame_idx=np.arange(100, 104, dtype=np.int64), pose_xyz=pose,
        observed=np.ones((4, 3, 3), dtype=bool),
        sources=np.zeros((4, 3, 3, 3), dtype=np.int64),
        identity_review=np.zeros((4, 3), dtype=bool),
    )


def test_missing_frame_keeps_direct_event_until_return():
    raw = geometry()
    raw["observed"][1, 0, 1] = False
    links = diag.sequential_links(raw, np.tile([0, 1, 2], (4, 1)), .8, .2)
    assert links["previous_direct_frame"][2, 0] == 100
    assert links["gap_frames_when_reobserved"][2, 0] == 2
    assert not links["motion_cost_applied"][1, 0]
    assert links["brownian_motion_cost_by_id"][1, 0] == 0
    expected = .2 / 3 * (4 / (.8 ** 2 * 2) + 3 * np.log(2))
    assert links["brownian_motion_cost_by_id"][2, 0] == pytest.approx(expected)
    np.testing.assert_array_equal(links["displacement_mm_when_reobserved"][2, 0], [2, 0, 0])


def test_head_only_support_is_observation_reassignment_even_without_pectoral():
    raw = geometry()
    raw["observed"][1, :2] = False
    raw["observed"][1, 0, 0] = True
    mapping = np.tile([0, 1, 2], (4, 1))
    mapping[1] = [1, 0, 2]
    rows = diag.reassignment_rows(raw, mapping)
    assert rows[0]["raw_slot"] == 0 and rows[0]["proposed_identity"] == 1
    assert rows[0]["latent_mapping_only"] is False
    assert rows[1]["latent_mapping_only"] is True
    result = diag.summarize_reassignment(rows)
    assert result["supported_node_reassignments_by_node"] == [1, 0, 0]
    assert result["frames_reassigning_any_observed_node"] == [101]


def test_all_missing_swap_changes_no_direct_observation_and_no_motion_cost():
    raw = geometry()
    raw["observed"][1, :2] = False
    fixed = np.tile([0, 1, 2], (4, 1))
    changed = fixed.copy()
    changed[1] = [1, 0, 2]
    rows = diag.reassignment_rows(raw, changed)
    assert diag.summarize_reassignment(rows)["supported_node_reassignments"] == 0
    left = diag.sequential_links(raw, fixed, .8, .2)
    right = diag.sequential_links(raw, changed, .8, .2)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_reassignment_moves_current_direct_observation_and_future_memory():
    raw = geometry()
    raw["observed"][1, 1] = False
    fixed = np.tile([0, 1, 2], (4, 1))
    changed = fixed.copy()
    changed[1] = [1, 0, 2]
    baseline = diag.sequential_links(raw, fixed, .8, .2)
    links = diag.sequential_links(raw, changed, .8, .2)
    assert links["motion_cost_applied"][1].tolist() == [False, True, True]
    assert links["previous_direct_raw_slot"][2].tolist() == [0, 0, 2]
    rows = diag.link_rows(raw, changed, links, baseline)
    assert any(row["frame"] == 102 for row in rows)
    assert diag.summarize_reassignment(diag.reassignment_rows(raw, changed))[
        "supported_node_reassignments_by_node"
    ] == [1, 1, 1]


def test_color_scoring_uses_mapping_state_and_preserves_enrollment_zeros():
    raw = geometry()
    mapping = np.tile([0, 1, 2], (4, 1))
    mapping[2] = [1, 0, 2]
    unary = np.arange(24, dtype=float).reshape(4, 6)
    unary[:2] = 0
    _, cost = diag.path_costs(raw, mapping, unary, dict(motion_scale_mm=.8, motion_weight=.2))
    np.testing.assert_array_equal(cost["color_by_frame"], [0, 0, 14, 18])
    np.testing.assert_allclose(
        cost["joint_by_frame"], cost["color_by_frame"] + cost["motion_by_frame"]
    )


def test_nonidentity_first_state_rejected():
    with pytest.raises(ValueError, match="fixed first"):
        diag.path_costs(geometry(), np.tile([1, 0, 2], (4, 1)), np.zeros((4, 6)), {})

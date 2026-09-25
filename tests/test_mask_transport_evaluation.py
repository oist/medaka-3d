"""Cross-anchor comparison cannot choose favorable evidence per pose."""
import sys
from pathlib import Path

import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
import mask_transport_evaluation as m  # noqa: E402


def scene():
    foreground = np.tile([1.0, 0.0], (3, 1))
    weights = np.tile([1.0, 0.25], (3, 1))
    union = np.broadcast_to(foreground, (3, 3, 2)).copy()
    owner = np.zeros((3, 3, 2, 3), np.float64)
    for method in range(3):
        owner[method, :, 0, method] = 1
    target = np.zeros((2, 2, 3, 2, 3), np.float64)
    target[0, 0, :, 0, 0] = 1
    target[0, 1, :, 0, 1] = 0.5
    target[1, 0, :, 0, 2] = 0.25
    target[1, 1, :, 0, 0] = 1
    return union, owner, target, foreground, weights, np.ones((3, 2), bool)


def test_identical_union_distinct_body_assignment_all_evidence_and_anchors_kept():
    result = m.render_scores(*scene())
    assert result["union_loss"].tolist() == [0, 0, 0]
    assert result["owner_loss"].shape == (2, 3, 2)
    np.testing.assert_array_equal(result["owner_loss"],
                                  [[[0, .5], [1, 0], [1, .5]],
                                   [[.25, 0], [.25, 1], [0, 1]]])
    assert result["owner_per_body_view"].shape == (2, 3, 2, 3, 3)
    assert result["target_mass"].shape == (2, 2, 3, 3)


def test_invalid_view_does_not_renormalize_remaining_two_views():
    arrays = list(scene())
    arrays[-1][1] = False
    result = m.render_scores(*arrays)
    assert result["owner_loss"][0, 1, 0] == 2 / 3
    assert not result["target_mass"][:, :, 1].any()


@pytest.mark.parametrize("fault", ["nonfinite", "shared_pixel", "omitted_evidence"])
def test_invalid_cross_score_inputs_are_rejected(fault):
    args = list(scene())
    if fault == "nonfinite":
        args[1][0, 0, 0, 0] = np.nan
    elif fault == "shared_pixel":
        args[2][0, 0, 0, 0, 1] = 1
    else:
        args[2] = args[2][:1]
    with pytest.raises(ValueError):
        m.render_scores(*args)


def test_missing_runs_keep_both_boundaries_and_singleton():
    frames = np.arange(20, 28, dtype=np.int64)
    observed = np.array([False, False, True, False, True, True, False, False])
    assert m.missing_runs(observed, frames) == [dict(start=20, stop=22, frames=2),
                                               dict(start=23, stop=24, frames=1),
                                               dict(start=26, stop=28, frames=2)]
    assert m.missing_runs(np.ones(8, bool), frames) == []
    assert m.missing_runs(np.zeros(8, bool), frames) == [dict(start=20, stop=28, frames=8)]


def test_support_gain_loss_uses_full_native_denominator_and_all_nodes():
    frames = np.arange(333519, 333719, dtype=np.int64)
    original = np.ones((200, 3, 3), bool)
    original[15:20, 2, 2] = False
    pose = np.zeros((200, 3, 3, 3), np.float64)
    candidates = {a: dict(frame_idx=frames.copy(), observed=original.copy(), pose_xyz=pose.copy())
                  for a in m.ARMS}
    candidates["ct_first"]["observed"][16, 2, 2] = True
    candidates["ct_first"]["observed"][199, 0, 0] = False
    result = m.summarize_support(candidates)
    row = result["ct_first"]
    assert row["per_node_denominator"] == 600
    assert row["direct_gain_by_body_node"][2][2] == 1
    assert row["direct_loss_by_body_node"][0][0] == 1
    assert row["gain_rows"] == [dict(native=333535, body=2, node=2)]
    assert row["loss_rows"] == [dict(native=333718, body=0, node=0)]
    assert result["original"]["longest_missing_by_body_node"][2][2] == 5
    assert row["longest_missing_by_body_node"][2][2] == 3

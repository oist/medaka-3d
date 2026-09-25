"""Synthetic CPU checks; no recording input, renderer, solver or GPU."""

import copy

import diagnose_instance_mouth_candidates as diag
import numpy as np
import pytest


def fixture():
    points = np.array(
        [
            [[10, 10], [15, 10], [25, 10]],
            [[100, 10], [105, 10], [115, 10]],
            [[200, 10], [205, 10], [215, 10]],
        ],
        dtype=np.float32,
    )
    pose = np.concatenate([points, np.ones((3, 3, 1))], axis=-1)
    poses = np.repeat(pose[None], 3, axis=0)
    poses[0, 2, 0, 0] += 50
    poses[2, 2, 0, 0] += 5
    plans = np.broadcast_to(np.arange(3)[None, :, None, None], (3, 3, 3, 3)).copy()
    observed = np.ones((3, 3, 3), dtype=bool)
    observed[0, 2, 0] = False
    observed[1, 0, 2] = False
    projected = np.repeat(poses[:, None, :, :, :2], 3, axis=1)
    views = {v: dict(eligible=False, reason="neutral") for v in diag.VIEWS}
    views["top_cam"] = dict(
        eligible=True,
        reason="eligible",
        best_assignment=np.broadcast_to(np.arange(3), (3, 3)).copy(),
        optimal_assignment_mask=np.broadcast_to(
            [True, False, False, False, False, False], (3, 6)
        ).copy(),
        observation_evidence=dict(qualifying_SL_rows=[[0], [1], [2]]),
        body_instance_costs=np.zeros((3, 3, 3)),
    )
    bank = dict(
        candidate_pose_xyz=poses,
        candidate_plan=plans,
        candidate_parent=np.array([2, 0, 1], dtype=np.int64),
        candidate_permutation=np.broadcast_to(np.arange(3), (3, 3)).copy(),
        candidate_kind=["selected", "loses_other_tail", "preserves_support"],
        base_local_cost=np.array([1.0, 3.0, 2.0]),
        adjusted_local_cost=np.array([1.0, 3.0, 2.0]),
        adjusted_path_cost=np.array([10.0, 3.0, 4.0]),
        views=views,
    )
    return bank, projected, plans.copy(), observed, 0, dict.fromkeys(diag.VIEWS, points)


def test_best_mouth_and_whole_scene_support_tradeoff_are_distinct():
    args = fixture()
    before = copy.deepcopy(args)
    result, arrays = diag.analyze_bank(*args)
    row = result["comparisons"][2]
    assert row["selected"]["mouth_error_px"] == 50
    assert row["best_mouth"]["index"] == 1 and row["best_mouth"]["mouth_error_px"] == 0
    assert row["best_mouth"]["direct_losses"] == [[0, 2]]
    assert row["best_mouth"]["adjusted_local_rank"] == 3
    assert row["best_mouth"]["path_rank"] == 1
    assert row["best_no_mask_increase_or_direct_loss"]["index"] == 2
    assert row["best_no_mask_increase_or_direct_loss"]["mouth_error_px"] == 5
    assert arrays["reference_available"].tolist() == [True, False, False]
    assert np.isnan(arrays["group_node_error_px"][:, 1:]).all()
    assert np.array_equal(args[0]["candidate_pose_xyz"], before[0]["candidate_pose_xyz"])
    assert arrays["candidate_pose_xyz"] is not args[0]["candidate_pose_xyz"]


def test_mask_cost_worsening_for_another_fish_is_not_hidden():
    args = fixture()
    args[0]["views"]["top_cam"]["body_instance_costs"][2, 0, 0] = 0.1
    result, _ = diag.analyze_bank(*args)
    assert result["comparisons"][2]["best_no_mask_increase_or_direct_loss"]["index"] == 0
    assert not result["comparisons"][2][
        "any_conservative_candidate_within_existing_12px_reference_gate"
    ]


def test_reference_mask_is_not_reassigned_to_make_wrong_fish_look_close():
    args = fixture()
    args[1][1, 0, 2, 0] = args[-1]["top_cam"][0, 0]
    args[0]["views"]["top_cam"]["best_assignment"][1] = [2, 1, 0]
    _, arrays = diag.analyze_bank(*args)
    assert arrays["group_node_error_px"][1, 0, 2, 0] == 190
    assert arrays["reference_mask_assignment"][0].tolist() == [0, 1, 2]


def test_nonfinite_candidate_stays_invalid_without_making_view_neutral():
    args = fixture()
    args[1][1, 0, 2, 0] = np.nan
    result, arrays = diag.analyze_bank(*args)
    assert arrays["reference_available"][0]
    assert np.isposinf(arrays["worst_reference_view_mouth_error_px"][1, 2])
    assert result["comparisons"][2]["best_mouth"]["index"] == 2


def test_selected_assignment_tie_is_not_arbitrarily_used_as_reference():
    args = fixture()
    args[0]["views"]["top_cam"]["optimal_assignment_mask"][0, 1] = True
    with pytest.raises(ValueError, match="no unambiguous"):
        diag.analyze_bank(*args)


def test_stable_ranks_and_full_node_dedup_match_frozen_selection_rule():
    assert diag.stable_rank(np.array([2.0, 1.0, 1.0])).tolist() == [3, 1, 2]
    poses = np.zeros((4, 3, 3, 3))
    poses[1, 0, 0, 0] = 0.49
    poses[2, 0, 0, 0] = 0.5
    parents = np.array([0, 0, 0, 1])
    permutations = np.broadcast_to([0, 1, 2], (4, 3))
    assert diag.retained_candidates(poses, parents, permutations, np.arange(4.0), 3).tolist() == [
        True,
        False,
        True,
        True,
    ]


def test_selected_matching_uses_all_cameras_and_support_and_rejects_duplicates():
    _bank, projected, sources, observed, selected, _ = fixture()
    reference = dict(
        observed=observed[selected].tolist(),
        sources=sources[selected].tolist(),
        views={
            v: [dict(projected_uv=a.tolist()) for a in projected[selected, vi].reshape(-1, 2)]
            for vi, v in enumerate(diag.VIEWS)
        },
    )
    assert diag.identify_selected(projected, observed, sources, reference) == 0
    projected[1] = projected[0]
    assert diag.identify_selected(projected, observed, sources, reference) == 0
    observed[1] = observed[0]
    with pytest.raises(ValueError, match="non-unique"):
        diag.identify_selected(projected, observed, sources, reference)


@pytest.mark.parametrize("fault", ["none", "changed", "missing", "extra", "duplicate"])
def test_companion_descriptors_preserve_nonfinite_bytes_and_reject_tampering(fault):
    array = np.array([np.nan, np.inf, -0.0])
    document = dict(tree=dict(x={"__array__": "a0", "signature": diag.proof.signature(array)}))
    archive = dict(a0=array.copy())
    if fault == "changed":
        archive["a0"][-1] = 0.0
    if fault == "missing":
        archive.clear()
    if fault == "extra":
        archive["unused"] = array
    if fault == "duplicate":
        document["tree"]["y"] = document["tree"]["x"]
    if fault != "none":
        with pytest.raises(ValueError):
            diag.restore_companion(document, archive)
    else:
        assert diag.restore_companion(document, archive)["x"].tobytes() == array.tobytes()

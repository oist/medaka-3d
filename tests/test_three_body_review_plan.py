"""Review suggestions must retain missing, censored and unresolved scenes."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from review_plan import build_review_plan  # noqa: E402


def case(count=40):
    return {
        "frame_idx": np.arange(100, 100 + count),
        "pose_xyz": np.ones((count, 3, 3, 3)),
        "observed": np.ones((count, 3, 3), bool),
        "body_supported": np.ones((count, 3), bool),
        "mask_supported": np.zeros((count, 3), bool),
        "identity_review": np.zeros((count, 3), bool),
        "missing": np.zeros((count, 3), int),
        "score_margin": np.full(count, np.inf),
    }


def lose(data, lo, hi, fish=0):
    data["observed"][lo:hi, fish, :2] = False
    data["body_supported"][lo:hi, fish] = False
    data["missing"][lo:hi, fish] = np.arange(1, hi - lo + 1)
    if hi - lo >= 4:
        data["identity_review"][lo + 3:, fish] = True


def test_nearest_supported_boundaries_and_no_repair_or_mutation():
    data = case()
    lose(data, 10, 15)
    original = copy.deepcopy(data)
    report = build_review_plan(data)
    scene = report["scenes"][0]
    assert scene["proposed_anchor_frames"] == [109, 115, 139]
    assert scene["anchor_identity_review_by_slot"] == [
        [False, False, False], [True, False, False], [True, False, False]]
    assert scene["entry_contains_unresolved_machine_id"] is False
    assert scene["ids_repaired"] is False and scene["human_confirmed_links"] == 0
    for key in data:
        np.testing.assert_array_equal(data[key], original[key])


def test_all_three_mouths_required_even_when_pectoral_supported():
    data = case()
    lose(data, 10, 15)
    data["observed"][9, 2, 0] = False
    data["observed"][15:18, 1, 0] = False
    data["observed"][39, 0, 0] = False
    assert build_review_plan(data)["scenes"][0]["proposed_anchor_frames"] == [108, 118, 138]


@pytest.mark.parametrize("lo,hi,expected", [
    (0, 4, [None, 104, 139]), (36, 40, [135, None, None]),
    (35, 39, [134, 139, None]), (0, 40, [None, None, None]),
])
def test_censored_probes_stay_missing(lo, hi, expected):
    data = case()
    lose(data, lo, hi)
    report = build_review_plan(data)
    assert report["scenes"][0]["proposed_anchor_frames"] == expected
    assert report["incomplete_scene_count"] == 1
    assert report["proposed_scene_count"] == 0


def test_repeated_risks_group_whole_scene_not_only_first_return():
    data = case()
    lose(data, 5, 9)
    lose(data, 15, 19, fish=1)
    scene = build_review_plan(data)["scenes"][0]
    assert len(scene["event_ids"]) == 2
    assert scene["proposed_anchor_frames"] == [104, 119, 139]


def test_gaps_split_scenes_and_probes_never_cross_them():
    data = case()
    data["frame_idx"][20:] += 10
    lose(data, 15, 19)
    lose(data, 20, 24, fish=1)
    scenes = build_review_plan(data)["scenes"]
    assert [s["proposed_anchor_frames"] for s in scenes] == [
        [114, 119, None], [None, 134, 149]]


def test_later_scene_reports_unresolved_entry_instead_of_resetting():
    data = case(400)
    lose(data, 5, 9)
    lose(data, 250, 254)
    scenes = build_review_plan(data)["scenes"]
    assert len(scenes) == 2
    assert scenes[1]["entry_contains_unresolved_machine_id"] is True
    assert scenes[1]["proposed_anchor_frames"] == [349, 354, 436]


def test_scalar_margin_inspects_all_mouths_and_preserves_unknown_pair():
    data = case()
    data["score_margin"][10:15] = 0.5
    data["observed"][15, 2, 0] = False
    scene = build_review_plan(data)["scenes"][0]
    assert scene["contains_unknown_affected_fish"] is True
    assert scene["proposed_anchor_frames"] == [109, 116, 139]


def test_no_risks_produces_no_truth_claim_or_fake_control_scene():
    report = build_review_plan(case())
    assert report["scenes"] == []
    assert report["proposed_scene_count"] == report["incomplete_scene_count"] == 0
    assert "not identity truth" in report["scope"]


def test_corrupt_observation_counters_fail_in_shared_inventory():
    data = case()
    data["missing"][10, 0] = 1
    with pytest.raises(ValueError, match="counters disagree"):
        build_review_plan(data)

"""New local optimizer must reproduce the real frozen CPU optimizer."""

import sys
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))

from node_geometry import assignment_plans  # noqa: E402
from reflection_local_fit import fit_local  # noqa: E402
from tracker import Proposal, State, ThreeBodyTracker, TrackerConfig  # noqa: E402

from test_three_body_local_fit import make_case  # noqa: E402


def test_direct_fourteen_steps_match_frozen_tracker_and_preserve_input(rig):
    seed, rays, targets = make_case(rig)
    start = seed + np.array([.18, -.07, .03])
    predicted = seed - [.1, 0, .04]
    missing = np.array([0, 5, 27])
    plan = assignment_plans(start, rays, keep=1)[0][1]
    originals = [x.copy() for x in (start, predicted, missing, plan)]
    config = TrackerConfig(device="cpu", opt_steps=14)
    tracker = ThreeBodyTracker(rig, seed, config)
    state = State(seed, np.zeros((3, 3)), missing, np.zeros(3, bool), 0.)
    proposal = Proposal(start, predicted, plan, 0, "test")
    expected, scores, parts, _ = tracker._fit([proposal], [state], rays, targets)
    actual = fit_local(rig, seed, config, start, plan, predicted, missing, rays, targets, "direct")
    np.testing.assert_array_equal(actual["pose_xyz"], expected[0])
    np.testing.assert_array_equal(actual["loss_parts"], parts[0])
    assert actual["score"] == scores[0]
    assert len(actual["iteration_scores"]) == 15
    assert actual["best_step"] == np.argmin(actual["iteration_scores"])
    np.testing.assert_array_equal(actual["iteration_pose_xyz"][actual["best_step"]], expected[0])
    assert actual["score"] <= actual["iteration_scores"][0] + 1e-12
    assert actual["tracking_initializations"] == tracker.initialize_count == 0
    assert not actual["identity_updated"]
    for new, old in zip((start, predicted, missing, plan), originals, strict=True):
        np.testing.assert_array_equal(new, old)


def test_outside_start_and_fractional_gap_are_rejected(rig):
    seed, rays, targets = make_case(rig)
    plan = assignment_plans(seed, rays, keep=1)[0][1]
    config = TrackerConfig(device="cpu", opt_steps=1)
    outside = seed.copy()
    outside[0, 0, 2] = -7
    with pytest.raises(ValueError, match="outside"):
        fit_local(rig, seed, config, outside, plan, seed, np.zeros(3, int),
                  rays, targets, "direct")
    with pytest.raises(ValueError, match="integer"):
        fit_local(rig, seed, config, seed, plan, seed, np.full(3, .1),
                  rays, targets, "direct")


@pytest.mark.parametrize("arm", ["unit", "fresnel"])
def test_reflected_arms_keep_fixed_plan_with_missing_rays_and_invalid_pixels(rig, arm):
    seed, rays, targets = make_case(rig)
    plan = assignment_plans(seed, rays, keep=1)[0][1]
    config = TrackerConfig(device="cpu", opt_steps=2)
    result = fit_local(rig, seed, config, seed, plan, seed, np.array([0, 3, 7]),
                       rays, targets, arm)
    np.testing.assert_array_equal(result["plan"], plan)
    assert np.isfinite(result["pose_xyz"]).all()
    assert np.isfinite(result["iteration_scores"]).all()
    assert result["score"] <= result["iteration_scores"][0] + 1e-12
    assert result["tracking_initializations"] == 0 and not result["identity_updated"]
    assert "identity_review" not in result and "accepted_xyz" not in result
    for view in range(3):
        for node in range(3):
            sources = result["sources"][:, view, node]
            sources = sources[sources >= 0]
            assert len(sources) == len(np.unique(sources))

"""Reflected fitting changes cost only; inherited temporal policy stays intact."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))

from node_geometry import assignment_plans  # noqa: E402
from observations import FrameObservation  # noqa: E402
from persistent_node_tracker import PersistentNodeThreeBodyTracker  # noqa: E402
from reflection_local_fit import fit_local  # noqa: E402
from reflection_tracker import ReflectionPersistentNodeTracker  # noqa: E402
from tracker import Proposal, State, TrackerConfig  # noqa: E402

from test_three_body_local_fit import make_case  # noqa: E402


def cases(rig):
    seed, rays, targets = make_case(rig)
    starts = [seed + np.array([.18, -.07, .03]), seed + np.array([-.12, .04, -.02])]
    predicted = [seed - [.1, 0, .04], seed + np.array([.05, -.03, .02])]
    missing = [np.array([0, 5, 27]), np.array([4, 0, 8])]
    states = [State(seed.copy(), np.zeros((3, 3)), gap, np.zeros(3, bool), 0.)
              for gap in missing]
    proposals = [Proposal(start, pred, assignment_plans(start, rays, keep=1)[0][1], i, "test")
                 for i, (start, pred) in enumerate(zip(starts, predicted, strict=True))]
    return seed, rays, targets, states, proposals


def test_direct_arm_delegates_exactly_and_preserves_temporal_methods(rig):
    seed, rays, targets, states, proposals = cases(rig)
    cfg = TrackerConfig(device="cpu", opt_steps=14)
    old = PersistentNodeThreeBodyTracker(rig, seed, cfg)
    new = ReflectionPersistentNodeTracker(rig, seed, cfg, mask_arm="direct")
    expected = old._fit(proposals, states, rays, targets)
    actual = new._fit(proposals, states, rays, targets)
    for value, ref in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, ref)
    for name in ("track", "_proposals", "_age_eligible", "_different_identity"):
        assert getattr(type(new), name) is getattr(type(old), name)
    assert new.initialize_count == old.initialize_count == 0


@pytest.mark.parametrize("arm", ["unit", "fresnel"])
def test_batched_fourteen_step_fit_matches_single_local_fits(rig, arm, monkeypatch):
    seed, rays, targets, states, proposals = cases(rig)
    cfg = TrackerConfig(device="cpu", opt_steps=14)
    tracker = ReflectionPersistentNodeTracker(rig, seed, cfg, mask_arm=arm)
    originals = [p.pose.copy() for p in proposals]
    poses, scores, parts, mask_support = tracker._fit(proposals, states, rays, targets)
    for i, proposal in enumerate(proposals):
        result = fit_local(rig, seed, cfg, proposal.pose, proposal.plan, proposal.predicted,
                           states[proposal.parent].missing, rays, targets, arm)
        np.testing.assert_allclose(poses[i], result["pose_xyz"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(scores[i], result["score"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(parts[i], result["loss_parts"], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(proposal.pose, originals[i])
    # Suppress just the reference optimizer update so the frozen fitter's own
    # final visibility code evaluates exactly the returned poses.
    frozen = PersistentNodeThreeBodyTracker(rig, seed, replace(cfg, opt_steps=1))
    at_best = [replace(p, pose=poses[i]) for i, p in enumerate(proposals)]
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", lambda self: None)
        expected_support = frozen._fit(at_best, states, rays, targets)[3]
    np.testing.assert_array_equal(mask_support, expected_support)
    assert tracker.initialize_count == 0


def test_nodes_only_fit_cannot_add_reflection_support(rig):
    seed, rays, _, states, proposals = cases(rig)
    cfg = TrackerConfig(device="cpu", opt_steps=2, mode="nodes")
    original = PersistentNodeThreeBodyTracker(rig, seed, cfg)
    reflected = ReflectionPersistentNodeTracker(rig, seed, cfg, mask_arm="fresnel")
    expected = original._fit(proposals, states, rays, None)
    actual = reflected._fit(proposals, states, rays, None)
    for value, ref in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, ref)
    assert not actual[3].any()


def test_direct_temporal_three_frames_preserve_all_twenty_one_arrays(rig):
    seed, _, _ = make_case(rig)
    cfg = TrackerConfig(device="cpu", mode="nodes", opt_steps=1, beam_size=3,
                        max_hypotheses=6)
    frames = []
    for t in range(3):
        points = {v: rig.camera(v).project(seed + np.array([.1 * t, 0., 0.]))
                  for v in rig.views}
        points[rig.views[1]][2, 2] = np.nan
        frames.append(FrameObservation(200 + t, points, {}, {}, {}))
    old = PersistentNodeThreeBodyTracker(rig, seed, cfg)
    new = ReflectionPersistentNodeTracker(rig, seed, cfg, mask_arm="direct")
    expected, actual = old.track(frames), new.track(frames)
    assert set(actual) == set(expected) and len(actual) == 21
    for key in expected:
        assert actual[key].dtype == expected[key].dtype
        assert actual[key].shape == expected[key].shape
        assert actual[key].tobytes() == expected[key].tobytes(), key
    assert old.initialize_count == new.initialize_count == 1
    with pytest.raises(RuntimeError, match="single-use"):
        new.track(frames)


def test_unknown_arm_rejected_before_tracker_setup(rig):
    seed, _, _ = make_case(rig)
    with pytest.raises(ValueError, match="mask_arm"):
        ReflectionPersistentNodeTracker(rig, seed, TrackerConfig(device="cpu"), mask_arm="auto")

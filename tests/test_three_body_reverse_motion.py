"""Real descending time, raw-only motion and explicit unsupported-node propagation."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import test_three_body_motion as forward_tests

motion = forward_tests.motion
textured_translation = forward_tests.textured_translation

PATH = Path(__file__).resolve().parents[1] / "labs/20260909_three_body/reverse_motion.py"
SPEC = importlib.util.spec_from_file_location("_three_body_reverse_motion_test", PATH)
reverse = importlib.util.module_from_spec(SPEC)
saved = sys.modules.get("motion")
try:
    sys.modules["motion"] = motion
    sys.modules[SPEC.name] = reverse
    SPEC.loader.exec_module(reverse)
finally:
    if saved is None:
        sys.modules.pop("motion", None)
    else:
        sys.modules["motion"] = saved


def test_reverse_native_warp_recovers_earlier_pose_without_reading_sleap(textured_translation):
    rig, pose, shift, earlier, later = textured_translation
    result = reverse.ReverseFlowObservations.from_frames(later, earlier, rig, pose + shift)
    assert result.previous_frame_idx == 11 and result.frame_idx == 10
    assert (earlier.frame_idx, later.frame_idx) == (10, 11)
    assert result.node_valid.all()
    np.testing.assert_allclose(result.pose_xyz, pose, atol=0.25)
    assert len(result.feature_audits) == 9
    assert all(a.usable and a.feature_valid.sum() >= 3 for a in result.feature_audits)
    assert not result.pose_xyz.flags.writeable
    assert not hasattr(result, "identity_review")


@pytest.mark.parametrize("numbers", [(10, 11), (11, 11), (12, 10), (11.0, 10)])
def test_only_genuine_integer_minus_one_time_steps_are_accepted(textured_translation, numbers):
    rig, pose, _, first, second = textured_translation
    first.frame_idx, second.frame_idx = numbers
    with pytest.raises(ValueError, match="native"):
        reverse.ReverseFlowObservations.from_frames(first, second, rig, pose)


def test_audited_texture_helper_matches_original_accepted_tracks(textured_translation):
    rig, pose, shift, earlier, later = textured_translation
    config = motion.FlowConfig()
    for view in motion.VIEWS:
        before, after = motion._gray(later.images_bgr[view]), motion._gray(earlier.images_bgr[view])
        projected = rig.camera(view).project(pose + shift)
        capsules = motion._capsules(projected, before.shape, config.body_radius_px)
        union = later.masks[view].any(axis=0)
        for fish in range(3):
            mask = capsules[fish] & union
            expected = motion._tracked_features(before, after, mask, config)
            actual, audit = reverse._tracked_features_audit(before, after, mask, config, view, fish)
            assert expected is not None and actual is not None and audit.usable
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(a, b)


def test_backward_inconsistency_is_recorded_and_never_triangulated(
    textured_translation, monkeypatch
):
    rig, pose, shift, earlier, later = textured_translation
    actual = motion.cv2.calcOpticalFlowPyrLK
    calls = 0

    def corrupt_return(*args, **kwargs):
        nonlocal calls
        calls += 1
        points, status, error = actual(*args, **kwargs)
        if calls % 2 == 0 and points is not None:
            points = points + np.array([8, 0], np.float32)
        return points, status, error

    monkeypatch.setattr(reverse.cv2, "calcOpticalFlowPyrLK", corrupt_return)
    result = reverse.ReverseFlowObservations.from_frames(later, earlier, rig, pose + shift)
    assert not result.valid.any() and not result.node_valid.any()
    assert any(
        a.forward_valid.any() and np.nanmax(a.fb_error_px) > 1 for a in result.feature_audits
    )
    assert all(not a.usable for a in result.feature_audits)


def test_empty_masks_remain_missing(textured_translation):
    rig, pose, shift, earlier, later = textured_translation
    later.masks = {view: np.zeros((0, 200, 200), bool) for view in motion.VIEWS}
    result = reverse.ReverseFlowObservations.from_frames(later, earlier, rig, pose + shift)
    assert not result.valid.any() and not result.node_valid.any()
    assert np.isnan(result.pose_xyz).all()
    assert len(result.feature_audits) == 9
    assert all(a.reason == "empty_or_too_small_feature_mask" for a in result.feature_audits)
    step = reverse.propagate_reverse(result, pose + shift)
    np.testing.assert_array_equal(step.pose_xyz, pose + shift)
    assert not step.node_flow_valid.any()


def test_partial_flow_translates_invalid_nodes_and_all_missing_uses_capped_reverse_velocity():
    previous = np.arange(27, dtype=float).reshape(3, 3, 3)
    older = previous - [10.0, 0.0, 0.0]
    valid = np.zeros((3, 3), bool)
    valid[0, :2], valid[1, 1] = True, True
    xyz = np.full((3, 3, 3), np.nan)
    xyz[0, :2] = previous[0, :2] + [[2, 0, 0], [4, 0, 0]]
    xyz[1, 1] = previous[1, 1] + [0, -1, 0]
    flow = SimpleNamespace(node_valid=valid, pose_xyz=xyz)
    result = reverse.propagate_reverse(flow, previous, older)
    np.testing.assert_array_equal(result.pose_xyz[0, :2], xyz[0, :2])
    np.testing.assert_array_equal(result.pose_xyz[0, 2], previous[0, 2] + [3, 0, 0])
    np.testing.assert_array_equal(result.pose_xyz[1], previous[1] + [0, -1, 0])
    np.testing.assert_array_equal(result.pose_xyz[2], previous[2] + [2, 0, 0])
    np.testing.assert_array_equal(result.node_flow_valid, valid)
    assert result.mode.tolist() == ["flow_with_translation"] * 2 + ["capped_velocity_fallback"]

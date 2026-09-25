"""Proposal revision retains baseline alternatives and never clears sticky review."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from medaka3d.types import VIEWS
from test_three_body_tracker import (
    config,
    frame_with_nodes,
    pose,  # noqa: F401
)

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def modules():
    names = ("prepare_sequence", "observations", "node_geometry", "renderer", "fast_union",
             "tracker", "graph_union", "graph_fit", "accelerated_tracker", "node_recovery",
             "node_recovery_tracker")
    previous, loaded = {n: sys.modules.get(n) for n in names}, {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(f"mouth_tracker_test_{name}",
                                                         LAB / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return loaded


M = modules()
T = M["tracker"]
R = M["node_recovery_tracker"]


def assert_proposal_equal(actual, expected):
    for key in ("pose", "predicted", "plan"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    for key in ("parent", "kind", "permutation"):
        assert getattr(actual, key) == getattr(expected, key)


def test_added_mouth_preserves_every_original_proposal_and_parent_state(rig, pose):  # noqa: F811
    frame, _ = frame_with_nodes(rig, pose, 45)
    rays = M["node_geometry"].RayObservations.from_points(rig, frame.detections)
    predicted = pose.copy()
    predicted[1, 0] = pose[1, 1] + [-5, 0, 0]
    state = T.State(predicted.copy(), np.zeros((3, 3)), np.zeros(3, int),
                    np.array([False, True, False]), 7.5)
    baseline = T.ThreeBodyTracker(rig, pose, config())
    revised = R.NodeRecoveryThreeBodyTracker(rig, pose, config())
    expected = baseline._proposals([state], rays)
    actual = revised._proposals([state], rays)
    assert len(expected) < len(actual) <= len(expected) + 6
    for before, after in zip(expected, actual[:len(expected)], strict=True):
        assert_proposal_equal(before, after)
    for added in actual[len(expected):]:
        assert added.kind == "mouth_recovery" and added.parent == 0
        assert added.permutation == (0, 1, 2)
        np.testing.assert_array_equal(added.predicted, expected[0].predicted)
        delta = np.linalg.norm(added.pose - expected[0].pose, axis=-1) > 1e-8
        assert np.array_equal(np.argwhere(delta), [[1, 0]])
    np.testing.assert_array_equal(state.pose, predicted)
    np.testing.assert_array_equal(state.review, [False, True, False])
    assert state.cost == 7.5 and not state.missing.any()


def test_healthy_scene_has_exactly_the_original_continuous_outputs(rig, pose):  # noqa: F811
    frames = [frame_with_nodes(rig, pose + np.array([0.1 * i, 0, 0]), 50 + i)[0]
              for i in range(3)]
    baseline = T.ThreeBodyTracker(rig, pose, config(opt_steps=2))
    revised = R.NodeRecoveryThreeBodyTracker(rig, pose, config(opt_steps=2))
    expected, actual = baseline.track(frames), revised.track(frames)
    assert set(expected) == set(actual)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    assert revised.node_recovery_diagnostics["calls"] == 3
    assert revised.node_recovery_diagnostics["calls_with_additional_proposals"] == 0


def test_current_mouth_can_recover_without_body_missing_or_reinitialization(rig, pose):  # noqa: F811
    actual_pose = pose.copy()
    actual_pose[1, 0] = pose[1, 1] + [-5, 0, 0]
    frames = [frame_with_nodes(rig, actual_pose, 70 + i)[0] for i in range(3)]
    baseline = T.ThreeBodyTracker(rig, pose, config(opt_steps=1))
    revised = R.NodeRecoveryThreeBodyTracker(rig, pose, config(opt_steps=1))
    old, new = baseline.track(frames), revised.track(frames)
    assert not old["observed"][:, 1, 0].any()
    assert new["observed"][:, 1, :2].all()
    np.testing.assert_allclose(new["pose_xyz"][:, 1, 0],
                               np.broadcast_to(actual_pose[1, 0], (3, 3)), atol=0.5)
    assert new["proposal_kind"][0] == "mouth_recovery"
    assert revised.initialize_count == 1 and not new["missing"].any()
    with pytest.raises(RuntimeError, match="single-use"):
        revised.track(frames)


def test_later_recovery_retains_review_after_unsupported_body(rig, pose):  # noqa: F811
    empty = {v: np.empty((0, 3, 2)) for v in VIEWS}
    frame_type = M["observations"].FrameObservation
    frames = [frame_type(90 + i, empty, {}, {}, {}) for i in range(4)]
    current = pose.copy()
    current[1, 0] = pose[1, 1] + [-5, 0, 0]
    frames.append(frame_with_nodes(rig, current, 94)[0])
    revised = R.NodeRecoveryThreeBodyTracker(rig, pose, config(opt_steps=1))
    result = revised.track(frames)
    assert result["identity_review"][3:].all()
    assert np.isnan(result["accepted_xyz"][3:]).all()
    assert result["observed"][-1, 1, :2].all()


def test_accelerated_variant_changes_fit_only_and_uses_recovery_proposals():
    cls = R.AcceleratedNodeRecoveryTracker
    assert cls._fit is M["accelerated_tracker"].AcceleratedThreeBodyTracker._fit
    assert cls._proposals is R.NodeRecoveryThreeBodyTracker._proposals
    assert cls.track is T.ThreeBodyTracker.track


@pytest.mark.parametrize("changes", [dict(proposals_per_fish=3), dict(upper_mm=(193, 192, 186))])
def test_tracker_rejects_unregistered_extra_budget_or_bounds(rig, pose, changes):  # noqa: F811
    with pytest.raises(ValueError):
        R.NodeRecoveryThreeBodyTracker(
            rig, pose, config(), recovery_config=M["node_recovery"].NodeRecoveryConfig(**changes)
        )

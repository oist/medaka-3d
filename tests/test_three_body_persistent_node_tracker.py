"""Parent-specific direct-node ages and compatibility, using bounded CPU scenes."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

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
             "persistent_node_tracker")
    previous, loaded = {name: sys.modules.get(name) for name in names}, {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(f"persistent_tracker_test_{name}",
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
T, P, G = M["tracker"], M["persistent_node_tracker"], M["node_geometry"]


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def assert_old_arrays_exact(actual, expected):
    assert len(expected) == 19
    assert set(actual) == set(expected) | {"node_missing", "direct_pectoral_age"}
    for name, a in expected.items():
        b = actual[name]
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert a.tobytes() == b.tobytes(), name


def wrong_mouth(scene):
    result = scene.copy()
    result[1, 0] = scene[1, 1] + [-5, 0, 0]
    return result


def state(scene, mouth_age=0, pectoral_age=0):
    missing = np.zeros((3, 3), int)
    missing[1, 0] = mouth_age
    age = np.zeros(3, int)
    age[1] = pectoral_age
    return P.PersistentNodeState(scene.copy(), np.zeros((3, 3)), np.zeros(3, int),
                                 np.array([False, True, False]), 0.0, missing, age)


def fake_fit(proposals, states, rays, targets):
    count = len(proposals)
    return (np.stack([p.pose for p in proposals]), np.zeros(count), np.zeros((count, 5)),
            np.zeros((count, 3), bool))


@pytest.mark.parametrize("enabled", [False, True])
def test_healthy_and_transient_frames_preserve_all_original_arrays(rig, pose, enabled):  # noqa: F811
    frames = [frame_with_nodes(rig, current, 300 + i)[0]
              for i, current in enumerate([pose, wrong_mouth(pose), pose, pose, pose])]
    cfg = config(opt_steps=1)
    old = T.ThreeBodyTracker(rig, pose, cfg).track(frames)
    revised = P.PersistentNodeThreeBodyTracker(
        rig, pose, cfg, persistence_config=P.PersistentNodeConfig(enabled=enabled))
    result = revised.track(frames)
    assert_old_arrays_exact(result, old)
    assert revised.node_recovery_diagnostics["calls_with_additional_proposals"] == 0
    assert not (result["node_missing"][:, 1, 0] >= 4).any()
    np.testing.assert_array_equal(result["direct_pectoral_age"],
                                   np.broadcast_to(np.arange(1, 6)[:, None], (5, 3)))


def test_disable_preserves_original_even_after_long_mouth_loss(rig, pose):  # noqa: F811
    frames = [frame_with_nodes(rig, wrong_mouth(pose), 400 + i)[0] for i in range(7)]
    cfg = config(opt_steps=1)
    expected = T.ThreeBodyTracker(rig, pose, cfg).track(frames)
    tracker = P.PersistentNodeThreeBodyTracker(
        rig, pose, cfg, persistence_config=P.PersistentNodeConfig(enabled=False))
    result = tracker.track(frames)
    assert_old_arrays_exact(result, expected)
    np.testing.assert_array_equal(result["node_missing"][:, 1, 0], np.arange(1, 8))
    assert tracker.node_recovery_diagnostics["calls_with_age_eligible_fish"] == 0


def test_absent_observations_preserve_original_nan_bytes_and_sticky_review(rig, pose):  # noqa: F811
    empty = {v: np.empty((0, 3, 2)) for v in VIEWS}
    frames = [M["observations"].FrameObservation(450 + i, empty, {}, {}, {}) for i in range(5)]
    cfg = config(opt_steps=1)
    expected = T.ThreeBodyTracker(rig, pose, cfg).track(frames)
    tracker = P.PersistentNodeThreeBodyTracker(rig, pose, cfg)
    result = tracker.track(frames)
    assert_old_arrays_exact(result, expected)
    assert result["identity_review"][-1].all()
    assert np.isnan(result["accepted_xyz"]).all()
    assert not result["direct_pectoral_age"].any()
    np.testing.assert_array_equal(result["node_missing"][-1], np.full((3, 3), 5))


def test_flow_continuation_with_recovery_disabled_keeps_original_arrays(rig, pose, monkeypatch):  # noqa: F811
    from types import SimpleNamespace

    class FixedFlow:
        @staticmethod
        def from_frames(previous, current, rig, parent_pose):
            assert current.frame_idx == previous.frame_idx + 1
            return SimpleNamespace(node_valid=np.ones((3, 3), bool),
                                   pose_xyz=parent_pose + np.array([0.3, 0.0, 0.0]))

    monkeypatch.setitem(sys.modules, "motion", SimpleNamespace(FlowObservations=FixedFlow))
    frames = [frame_with_nodes(rig, pose + np.array([0.3 * i, 0, 0]), 470 + i)[0]
              for i in range(3)]
    cfg = config(opt_steps=1, flow_proposals=True)
    expected = T.ThreeBodyTracker(rig, pose, cfg).track(frames)
    tracker = P.PersistentNodeThreeBodyTracker(
        rig, pose, cfg, persistence_config=P.PersistentNodeConfig(enabled=False))
    result = tracker.track(frames)
    assert_old_arrays_exact(result, expected)
    assert result["proposal_flow_nodes"][1:].all()


def test_recovery_waits_four_completed_frames_and_resets_only_after_direct_return(rig, pose):  # noqa: F811
    truth = wrong_mouth(pose)
    frames = [frame_with_nodes(rig, truth, 500 + i)[0] for i in range(6)]
    tracker = P.PersistentNodeThreeBodyTracker(rig, pose, config(opt_steps=1))
    result = tracker.track(frames)
    assert not result["observed"][:4, 1, 0].any()
    assert result["observed"][4:, 1, :2].all()
    np.testing.assert_array_equal(result["node_missing"][:, 1, 0], [1, 2, 3, 4, 0, 0])
    np.testing.assert_array_equal(result["direct_pectoral_age"][:, 1], np.arange(1, 7))
    assert result["proposal_kind"][4] == "mouth_recovery"
    assert "mouth_recovery" not in result["proposal_kind"][:4]
    np.testing.assert_allclose(result["pose_xyz"][4:, 1, 0],
                               np.broadcast_to(truth[1, 0], (2, 3)), atol=0.5)
    assert tracker.initialize_count == 1
    for source in result["sources"]:
        for view in range(3):
            for node in range(3):
                rows = source[:, view, node]
                rows = rows[rows >= 0]
                assert len(rows) == len(set(rows))


def test_current_strict_pectoral_support_still_required_after_history_gate(rig, pose):  # noqa: F811
    frame, sources = frame_with_nodes(rig, pose, 600)
    for vi, view in enumerate(VIEWS):
        frame.detections[view][sources[1, vi, 1], 1] = np.nan
    rays = G.RayObservations.from_points(rig, frame.detections)
    tracker = P.PersistentNodeThreeBodyTracker(rig, pose, config())
    parent = state(wrong_mouth(pose), mouth_age=20, pectoral_age=20)
    actual = tracker._proposals([parent], rays)
    assert tracker._age_eligible(parent)[1]
    assert all(p.kind != "mouth_recovery" for p in actual)
    assert tracker.node_recovery_diagnostics["calls_with_age_eligible_fish"] == 1


def test_history_gate_tracks_selected_parent_not_other_beam_age(rig, pose):  # noqa: F811
    frame, _ = frame_with_nodes(rig, pose, 610)
    rays = G.RayObservations.from_points(rig, frame.detections)
    tracker = P.PersistentNodeThreeBodyTracker(rig, pose, config())
    fresh = state(wrong_mouth(pose))
    eligible = state(wrong_mouth(pose), 4, 4)
    assert not any(p.kind == "mouth_recovery" for p in tracker._proposals([fresh, eligible], rays))
    proposals = tracker._proposals([eligible, fresh], rays)
    extra = [p for p in proposals if p.kind == "mouth_recovery"]
    assert extra and all(p.parent == 0 for p in extra)
    np.testing.assert_array_equal(eligible.node_missing[1], [4, 0, 0])
    np.testing.assert_array_equal(fresh.node_missing, np.zeros((3, 3)))
    for p in extra:
        np.testing.assert_array_equal(p.predicted, proposals[0].predicted)
        assert p.permutation == proposals[0].permutation


def test_direct_pectoral_streak_restarts_and_ages_do_not_alias(rig, pose):  # noqa: F811
    tracker = P.PersistentNodeThreeBodyTracker(rig, pose, config())
    parent = state(pose, 4, 4)
    observed = np.ones((3, 3), bool)
    observed[1, 0] = observed[1, 1] = False
    node_missing, age = P.updated_ages(parent, observed)
    assert node_missing[1, 0] == 5 and age[1] == 0
    assert not np.shares_memory(node_missing, parent.node_missing)
    assert not np.shares_memory(age, parent.direct_pectoral_age)
    observed[1, 1] = True
    for count in range(1, 5):
        parent = P.PersistentNodeState(parent.pose, parent.velocity, parent.missing, parent.review,
                                       0.0, node_missing, age)
        node_missing, age = P.updated_ages(parent, observed)
        parent = P.PersistentNodeState(parent.pose, parent.velocity, parent.missing, parent.review,
                                       0.0, node_missing, age)
        assert tracker._age_eligible(parent)[1] == (count >= 4)


def test_mask_body_bridge_does_not_count_as_direct_pectoral_history(rig, pose, monkeypatch):  # noqa: F811
    frames = []
    for i in range(6):
        frame, sources = frame_with_nodes(rig, pose, 700 + i)
        for vi, view in enumerate(VIEWS):
            frame.detections[view][sources[1, vi, 1], 1] = np.nan
        frames.append(frame)
    tracker = P.PersistentNodeThreeBodyTracker(rig, pose, config(opt_steps=1))

    def bridge_fit(proposals, states, rays, targets):
        fitted, costs, parts, masks = fake_fit(proposals, states, rays, targets)
        masks[:] = True
        return fitted, costs, parts, masks

    monkeypatch.setattr(tracker, "_fit", bridge_fit)
    result = tracker.track(frames)
    assert result["body_supported"][:, 1].all()
    assert not result["observed"][:, 1, 1].any()
    assert not result["direct_pectoral_age"][:, 1].any()
    np.testing.assert_array_equal(result["node_missing"][:, 1, 1], np.arange(1, 7))


def test_offered_but_unselected_mouth_does_not_reset_selected_age(rig, pose, monkeypatch):  # noqa: F811
    frames = [frame_with_nodes(rig, wrong_mouth(pose), 800 + i)[0] for i in range(6)]
    tracker = P.PersistentNodeThreeBodyTracker(rig, pose, config(beam_size=1, opt_steps=1))

    def prefer_base(proposals, states, rays, targets):
        fitted, costs, parts, masks = fake_fit(proposals, states, rays, targets)
        costs[:] = [100 if p.kind == "mouth_recovery" else 0 for p in proposals]
        return fitted, costs, parts, masks

    monkeypatch.setattr(tracker, "_fit", prefer_base)
    result = tracker.track(frames)
    assert tracker.node_recovery_diagnostics["calls_with_additional_proposals"] == 2
    assert "mouth_recovery" not in result["proposal_kind"]
    np.testing.assert_array_equal(result["node_missing"][:, 1, 0], np.arange(1, 7))


def test_branch_ages_follow_temporal_parent_and_backtrace(rig, pose, monkeypatch):  # noqa: F811
    frames = [frame_with_nodes(rig, pose, 900 + i)[0] for i in range(2)]
    tracker = P.PersistentNodeThreeBodyTracker(rig, pose, config(opt_steps=1))
    calls = []

    def branching(states, rays, flow=None):
        calls.append(states)
        good = G.assignment_plans(pose, rays, keep=1)[0][1]
        if len(calls) == 1:
            bad = wrong_mouth(pose)
            bad_plan = G.assignment_plans(bad, rays, keep=1)[0][1]
            return [T.Proposal(bad, pose, bad_plan, 0, "branch_bad"),
                    T.Proposal(pose.copy(), pose, good, 0, "branch_good")]
        assert states[0].node_missing[1, 0] == 1
        assert states[1].node_missing[1, 0] == 0
        return [T.Proposal(pose.copy(), pose, good, parent, "reconverged")
                for parent in range(2)]

    def branch_fit(proposals, states, rays, targets):
        fitted, _, parts, masks = fake_fit(proposals, states, rays, targets)
        costs = np.array([0.0, 1.0] if len(calls) == 1 else [10.0, 0.0])
        return fitted, costs, parts, masks

    monkeypatch.setattr(tracker, "_proposals", branching)
    monkeypatch.setattr(tracker, "_fit", branch_fit)
    result = tracker.track(frames)
    assert result["proposal_kind"][0] == "branch_good"
    assert not result["node_missing"].any()
    np.testing.assert_array_equal(result["direct_pectoral_age"], [[1] * 3, [2] * 3])
    # Both retained first-frame parents have independent counters and remain unchanged.
    assert calls[1][0].node_missing[1, 0] == 1
    assert calls[1][1].node_missing[1, 0] == 0


def test_single_use_and_discontinuity_checks_preserved(rig, pose):  # noqa: F811
    first, _ = frame_with_nodes(rig, pose, 10)
    tracker = P.PersistentNodeThreeBodyTracker(rig, pose, config(opt_steps=1))
    tracker.track([first])
    with pytest.raises(RuntimeError, match="single-use"):
        tracker.track([first])
    later, _ = frame_with_nodes(rig, pose, 12)
    with pytest.raises(ValueError, match="continuous native frames"):
        P.PersistentNodeThreeBodyTracker(rig, pose, config(opt_steps=1)).track([first, later])


def test_accelerated_variant_only_replaces_fit():
    cls = P.AcceleratedPersistentNodeTracker
    assert cls._fit is M["accelerated_tracker"].AcceleratedThreeBodyTracker._fit
    assert cls.track is P.PersistentNodeThreeBodyTracker.track
    assert cls._proposals is P.PersistentNodeThreeBodyTracker._proposals


@pytest.mark.parametrize("changes", [dict(enabled=1), dict(minimum_missing_frames=0),
                                      dict(minimum_pectoral_frames=True),
                                      dict(minimum_missing_frames=2.5)])
def test_invalid_age_config_fails(changes):
    with pytest.raises(ValueError):
        P.PersistentNodeConfig(**changes)

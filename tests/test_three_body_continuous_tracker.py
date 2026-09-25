"""CPU continuation parity with the frozen loop; synthetic inputs are not ID truth."""

from __future__ import annotations

import copy
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from medaka3d.types import VIEWS

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load_modules():
    names = (
        "prepare_sequence",
        "observations",
        "node_geometry",
        "renderer",
        "fast_union",
        "motion",
        "tracker",
        "graph_union",
        "graph_fit",
        "accelerated_tracker",
        "node_recovery",
        "persistent_node_tracker",
        "continuous_tracker",
    )
    old = {name: sys.modules.get(name) for name in names}
    loaded = {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(
                "continuous_test_" + name, LAB / (name + ".py")
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return loaded


M = load_modules()
C, P, T, OBS, G = (
    M[key]
    for key in (
        "continuous_tracker",
        "persistent_node_tracker",
        "tracker",
        "observations",
        "node_geometry",
    )
)


@dataclass
class SmallCamera:
    axes: tuple[int, int]
    offset: np.ndarray

    def project(self, xyz):
        return np.asarray(xyz)[..., self.axes] + self.offset

    def rays(self, uv):
        uv = np.asarray(uv)
        origin = np.zeros((*uv.shape[:-1], 3))
        origin[..., self.axes] = uv - self.offset
        direction = np.zeros_like(origin)
        direction[..., next(iter(set(range(3)) - set(self.axes)))] = 1
        origin[~np.isfinite(uv).all(-1)] = np.nan
        return origin, direction


@dataclass
class SmallRig:
    cameras: tuple[SmallCamera, ...]
    views: tuple[str, ...] = VIEWS
    world_units: str = "mm"

    def camera(self, view):
        return self.cameras[self.views.index(view)]


@pytest.fixture
def rig():
    return SmallRig(tuple(SmallCamera(axes, np.zeros(2)) for axes in ((0, 1), (0, 2), (1, 2))))


@pytest.fixture
def pose():
    centers = np.array([[45.0, 48.0, 50.0], [95.0, 100.0, 92.0], [150.0, 145.0, 148.0]])
    return (
        centers[:, None] + np.array([[8.0, 6.0, 4.0], [0.0, 0.0, 0.0], [-14.0, -10.0, -6.0]])[None]
    )


@pytest.fixture(autouse=True)
def bound_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def config(**values):
    return replace(
        T.TrackerConfig(
            device="cpu",
            mode="nodes",
            opt_steps=2,
            beam_size=2,
            max_hypotheses=4,
            positive_samples=12,
            near_samples=8,
            global_samples=4,
        ),
        **values,
    )


def frame(rig, pose, native, *, missing_mouth=False, empty=False):
    points = {}
    for vi, view in enumerate(VIEWS):
        projected = rig.camera(view).project(pose)
        if missing_mouth:
            projected[1, 0] = np.nan
        if empty:
            projected = np.empty((0, 3, 2))
        else:
            projected = projected[np.random.default_rng(native + vi).permutation(3)]
        points[view] = projected
    return OBS.FrameObservation(native, points, {}, {}, {})


def wrong_mouth(pose):
    other = pose.copy()
    length = np.linalg.norm(pose[1, 0] - pose[1, 1])
    other[1, 0] = other[1, 1] + [-length, 0.0, 0.0]
    return other


def assert_exact(actual, expected):
    assert set(actual) == set(expected) and len(actual) == 21
    for key in expected:
        assert actual[key].shape == expected[key].shape, key
        assert actual[key].dtype == expected[key].dtype, key
        assert actual[key].tobytes() == expected[key].tobytes(), key


def run_split(rig, pose, cfg, frames, split, *, configure=None):
    first = C.ContinuousPersistentNodeTracker(rig, pose, cfg)
    if configure is not None:
        configure(first)
    first.begin()
    for observation in frames[:split]:
        first.advance(observation)
    arrays, metadata = first.export_state()
    resumed = C.ContinuousPersistentNodeTracker(copy.deepcopy(rig), pose.copy(), cfg)
    if configure is not None:
        configure(resumed)
    resumed.restore_state(arrays, metadata)
    assert resumed.initialize_count == 1 and resumed.restore_count == 1
    assert resumed.processed_frames == split
    assert resumed.previous_frame == frames[split - 1].frame_idx
    for observation in frames[split:]:
        resumed.advance(observation)
    return resumed.finish(), resumed, arrays, metadata


@pytest.mark.parametrize("split", [3, 4, 5, 10])
def test_real_optimizer_recovery_and_sticky_loss_match_frozen_loop_after_restore(rig, pose, split):
    cfg = config(opt_steps=1)
    frames = [frame(rig, wrong_mouth(pose), 300 + i) for i in range(6)]
    frames += [frame(rig, pose, 300 + i, empty=True) for i in range(6, 11)]
    frames += [frame(rig, pose, 311)]
    oracle = P.PersistentNodeThreeBodyTracker(rig, pose, cfg)
    expected = oracle.track(frames)
    actual, resumed, _, _ = run_split(rig, pose, cfg, frames, split)
    assert_exact(actual, expected)
    assert resumed.node_recovery_diagnostics == oracle.node_recovery_diagnostics
    assert expected["node_missing"][3, 1, 0] == 4
    assert expected["proposal_kind"][4] == "mouth_recovery"
    assert expected["identity_review"][-1].all()
    assert np.isnan(expected["accepted_xyz"][-1]).all()


def test_multiple_checkpoints_preserve_history_and_finish_output_does_not_alias(rig, pose):
    cfg = config(opt_steps=1)
    frames = [frame(rig, pose + np.array([0.15 * i, 0.0, 0.0]), 400 + i) for i in range(5)]
    expected = P.PersistentNodeThreeBodyTracker(rig, pose, cfg).track(frames)
    current = C.ContinuousPersistentNodeTracker(rig, pose, cfg)
    current.begin()
    for i, observation in enumerate(frames):
        current.advance(observation)
        arrays, meta = current.export_state()
        replacement = C.ContinuousPersistentNodeTracker(rig, pose, cfg)
        replacement.restore_state(arrays, meta)
        current = replacement
        assert current.restore_count == i + 1
    result = current.finish()
    assert_exact(result, expected)
    result["pose_xyz"][:] = -123
    assert (current._history[-1][0]["pose_xyz"] >= 0).all()
    assert not np.shares_memory(result["pose_xyz"], pose)


def configure_branches(tracker, initial):
    def proposals(states, rays, flow=None):
        tracker._recovery_calls += 1
        good = G.assignment_plans(initial, rays, keep=1)[0][1]
        if states[0].direct_pectoral_age[0] == 0:
            bad = wrong_mouth(initial)
            bad_plan = G.assignment_plans(bad, rays, keep=1)[0][1]
            return [
                T.Proposal(bad, initial, bad_plan, 0, "early_best"),
                T.Proposal(initial.copy(), initial, good, 0, "later_winner"),
            ]
        return [
            T.Proposal(initial.copy(), initial, good, i, "reconverged") for i in range(len(states))
        ]

    def fit(proposals, states, rays, targets):
        early = states[0].direct_pectoral_age[0] == 0
        cost = np.array([0.0, 1.0] if early else [10.0, 0.0])
        return (
            np.stack([p.pose for p in proposals]),
            cost,
            np.zeros((2, 5)),
            np.zeros((2, 3), bool),
        )

    tracker._proposals = proposals
    tracker._fit = fit


def test_nonleading_parent_reverses_after_checkpoint_and_keeps_its_own_ages(rig, pose):
    cfg = config(opt_steps=1)
    frames = [frame(rig, pose, 500 + i) for i in range(2)]
    oracle = P.PersistentNodeThreeBodyTracker(rig, pose, cfg)
    configure_branches(oracle, pose)
    expected = oracle.track(frames)
    actual, _, saved, _ = run_split(
        rig, pose, cfg, frames, 1, configure=lambda obj: configure_branches(obj, pose)
    )
    assert saved["history_offsets"].tolist() == [0, 2]
    assert saved["state_node_missing"][0, 1, 0] == 1
    assert saved["state_node_missing"][1, 1, 0] == 0
    assert expected["proposal_kind"][0] == "later_winner"
    assert_exact(actual, expected)


def test_mask_only_body_bridge_keeps_direct_pectoral_age_zero_across_restore(rig, pose):
    cfg = config(opt_steps=1)
    frames = [frame(rig, pose, 550 + index) for index in range(5)]
    for observation in frames:
        for view in VIEWS:
            observation.detections[view][:, 1] = np.nan

    def configure(tracker):
        def fitted(proposals, states, rays, targets):
            count = len(proposals)
            return (
                np.stack([proposal.pose for proposal in proposals]),
                np.zeros(count),
                np.zeros((count, 5)),
                np.ones((count, 3), bool),
            )

        tracker._fit = fitted

    oracle = P.PersistentNodeThreeBodyTracker(rig, pose, cfg)
    configure(oracle)
    expected = oracle.track(frames)
    actual, _, _, _ = run_split(rig, pose, cfg, frames, 3, configure=configure)
    assert_exact(actual, expected)
    assert actual["body_supported"].all()
    assert not actual["observed"][:, :, 1].any()
    assert not actual["direct_pectoral_age"].any()
    np.testing.assert_array_equal(actual["node_missing"][-1, :, 1], [5, 5, 5])


def textured_frames(rig, pose):
    base_images, base_masks = {}, {}
    for vi, view in enumerate(VIEWS):
        masks = np.zeros((3, 200, 200), np.uint8)
        for fish, uv in enumerate(rig.camera(view).project(pose)):
            points = np.rint(uv).astype(np.int32)
            cv2.polylines(masks[fish], [points], False, 1, 16)
            for point in points:
                cv2.circle(masks[fish], tuple(point), 8, 1, -1)
        texture = np.random.default_rng(400 + vi).integers(30, 240, (200, 200), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0.5)
        gray = np.where(masks.any(0), texture, 15).astype(np.uint8)
        base_images[view] = np.repeat(gray[:, :, None], 3, axis=2)
        base_masks[view] = masks
    frames = []
    for index in range(3):
        shift = np.array([0.8, -0.7, 0.5]) * index
        current = frame(rig, pose + shift, 600 + index)
        images, masks = {}, {}
        for view in VIEWS:
            dx, dy = rig.camera(view).project(shift)
            matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])
            images[view] = cv2.warpAffine(
                base_images[view], matrix, (200, 200), borderValue=(15, 15, 15)
            )
            masks[view] = np.stack(
                [
                    cv2.warpAffine(mask, matrix, (200, 200), flags=cv2.INTER_NEAREST).astype(bool)
                    for mask in base_masks[view]
                ]
            )
        frames.append(
            OBS.FrameObservation(
                current.frame_idx,
                current.detections,
                images,
                masks,
                {v: np.full((200, 200, 3), 15, np.uint8) for v in VIEWS},
            )
        )
    return frames


@pytest.mark.parametrize("mode", ["nodes", "detectron2"])
def test_real_lk_and_pixel_sampling_cross_split_with_exact_frozen_arrays(
    rig, pose, mode, monkeypatch
):
    frames = textured_frames(rig, pose)
    cfg = config(opt_steps=1, flow_proposals=True, mode=mode)
    # The old loop imports motion at call time; bind the same frozen numeric implementation.
    monkeypatch.setitem(sys.modules, "motion", M["motion"])
    oracle = P.PersistentNodeThreeBodyTracker(rig, pose, cfg)
    expected = oracle.track(frames)
    assert expected["proposal_flow_nodes"][1:].any()
    actual, _, arrays, meta = run_split(rig, pose, cfg, frames, 2)
    assert_exact(actual, expected)
    assert meta["previous_frame"] == 601
    for view in VIEWS:
        np.testing.assert_array_equal(
            arrays["previous_images_bgr_" + view], frames[1].images_bgr[view]
        )


def test_snapshot_and_consumed_frame_buffers_are_independent_and_mask_availability_survives(
    rig, pose
):
    observation = textured_frames(rig, pose)[0]
    observation.masks[VIEWS[0]] = None
    observation.masks[VIEWS[1]] = np.empty((0, 200, 200), bool)
    cfg = config(opt_steps=1, flow_proposals=True)
    current = C.ContinuousPersistentNodeTracker(rig, pose, cfg)
    current.begin()
    current.advance(observation)
    first, first_meta = current.export_state()
    frozen = {key: value.copy() for key, value in first.items()}
    observation.images_bgr[VIEWS[2]][:] = 0
    observation.masks[VIEWS[2]][:] = False
    observation.detections[VIEWS[2]][:] = np.nan
    second, second_meta = current.export_state()
    for key in first:
        assert second[key].tobytes() == frozen[key].tobytes(), key
        assert not np.shares_memory(first[key], second[key]), key
    assert first_meta == second_meta
    assert first_meta["previous_observation"]["previous_masks_" + VIEWS[0]] is False
    assert first_meta["previous_observation"]["previous_masks_" + VIEWS[1]] is True
    assert first["previous_masks_" + VIEWS[1]].shape == (0, 200, 200)
    resumed = C.ContinuousPersistentNodeTracker(rig, pose, cfg)
    resumed.restore_state(first, first_meta)
    first["state_pose"][:] = -100
    first["previous_images_bgr_" + VIEWS[2]][:] = 2
    first_meta["node_recovery_diagnostics"]["proposed_by_fish"][0] = 999
    restored, _ = resumed.export_state()
    for key in restored:
        assert restored[key].tobytes() == frozen[key].tobytes(), key


@pytest.mark.parametrize("damage", ["missing_image", "mask_shape", "mask_false_presence"])
def test_flow_checkpoint_refuses_incomplete_previous_frame_evidence(rig, pose, damage):
    cfg = config(opt_steps=1, flow_proposals=True)
    current = C.ContinuousPersistentNodeTracker(rig, pose, cfg)
    current.begin()
    current.advance(textured_frames(rig, pose)[0])
    arrays, metadata = current.export_state()
    if damage == "missing_image":
        arrays.pop("previous_images_bgr_" + VIEWS[0])
    elif damage == "mask_shape":
        arrays["previous_masks_" + VIEWS[0]] = arrays["previous_masks_" + VIEWS[0]][:, :-1]
    else:
        metadata["previous_observation"]["previous_masks_" + VIEWS[0]] = False
    receiver = C.ContinuousPersistentNodeTracker(rig, pose, cfg)
    with pytest.raises(ValueError):
        receiver.restore_state(arrays, metadata)
    assert receiver.processed_frames == 0 and receiver.initialize_count == 0


def checkpoint(rig, pose):
    cfg = config(opt_steps=1)
    tracker = C.ContinuousPersistentNodeTracker(rig, pose, cfg)
    tracker.begin()
    tracker.advance(frame(rig, pose, 700))
    arrays, meta = tracker.export_state()
    return cfg, tracker, arrays, meta


def test_lifecycle_is_single_begin_terminal_finish_and_fresh_only_restore(rig, pose):
    cfg, started, arrays, meta = checkpoint(rig, pose)
    fresh = C.ContinuousPersistentNodeTracker(rig, pose, cfg)
    with pytest.raises(RuntimeError):
        fresh.advance(frame(rig, pose, 700))
    fresh.begin()
    with pytest.raises(ValueError, match="completed frame"):
        fresh.export_state()
    with pytest.raises(ValueError, match="empty"):
        fresh.finish()
    for current in (fresh, started):
        with pytest.raises(RuntimeError):
            current.begin()
        with pytest.raises(RuntimeError):
            current.restore_state(arrays, meta)
    started.finish()
    for action in (
        started.finish,
        started.export_state,
        lambda: started.advance(frame(rig, pose, 701)),
    ):
        with pytest.raises(RuntimeError):
            action()


@pytest.mark.parametrize("native", [700, 702, -1, True, 701.0, np.uint64(2**63)])
def test_invalid_native_input_does_not_mutate_completed_state(rig, pose, native):
    _, current, before, metadata = checkpoint(rig, pose)
    original = frame(rig, pose, 701)
    invalid = OBS.FrameObservation(native, original.detections, {}, {}, {})
    with pytest.raises(ValueError):
        current.advance(invalid)
    after, after_metadata = current.export_state()
    assert after_metadata == metadata
    for key in before:
        assert after[key].tobytes() == before[key].tobytes()
    current.advance(original)
    assert current.processed_frames == 2


def test_internal_fit_failure_poisoned_state_cannot_continue_or_export(rig, pose):
    _, current, _, _ = checkpoint(rig, pose)

    def failed(*args):
        raise FloatingPointError("synthetic failed optimizer")

    current._fit = failed
    with pytest.raises(FloatingPointError):
        current.advance(frame(rig, pose, 701))
    assert current.processed_frames == 1
    for action in (
        current.export_state,
        current.finish,
        current.begin,
        lambda: current.advance(frame(rig, pose, 701)),
    ):
        with pytest.raises(RuntimeError):
            action()


@pytest.mark.parametrize("change", ["config", "recovery", "persistence", "initial", "rig", "class"])
def test_checkpoint_binds_exact_algorithm_configuration_initial_pose_and_rig(rig, pose, change):
    cfg, _, arrays, metadata = checkpoint(rig, pose)
    changed_rig, changed_pose, extra = copy.deepcopy(rig), pose.copy(), {}
    cls = C.ContinuousPersistentNodeTracker
    if change == "config":
        cfg = replace(cfg, motion_weight=cfg.motion_weight + 0.01)
    elif change == "recovery":
        extra["recovery_config"] = replace(
            M["node_recovery"].NodeRecoveryConfig(), proposals_per_fish=1
        )
    elif change == "persistence":
        extra["persistence_config"] = P.PersistentNodeConfig(minimum_missing_frames=5)
    elif change == "initial":
        changed_pose[0, 0, 0] += 0.01
    elif change == "rig":
        changed_rig.cameras[0].offset[0] += 0.01
    else:
        cls = C.AcceleratedContinuousPersistentNodeTracker
    receiver = cls(changed_rig, changed_pose, cfg, **extra)
    with pytest.raises(ValueError, match="context"):
        receiver.restore_state(arrays, metadata)
    assert receiver.initialize_count == 0 and receiver.processed_frames == 0


@pytest.mark.parametrize(
    "corruption",
    [
        "parent",
        "history_age",
        "beam_age",
        "velocity",
        "frame",
        "path_cost",
        "review",
        "body",
        "source",
        "offsets",
        "field",
        "schema",
        "counter",
    ],
)
def test_corrupted_history_or_active_beam_is_rejected_atomically(rig, pose, corruption):
    cfg, _, arrays, metadata = checkpoint(rig, pose)
    if corruption == "parent":
        arrays["history_parent"][0] = 1
    elif corruption == "history_age":
        arrays["history_node_missing"][0, 0, 0] += 1
    elif corruption == "beam_age":
        arrays["state_direct_pectoral_age"][0, 0] += 1
    elif corruption == "velocity":
        arrays["state_velocity"][0, 0, 0] += 0.5
    elif corruption == "frame":
        metadata["previous_frame"] += 1
    elif corruption == "path_cost":
        arrays["history_path_costs"][0] += 0.01
    elif corruption == "review":
        arrays["state_review"][0, 0] = ~arrays["state_review"][0, 0]
    elif corruption == "body":
        arrays["history_body_supported"][0, 0] = ~arrays["history_body_supported"][0, 0]
    elif corruption == "source":
        arrays["history_sources"][0, 0, 0, 0] = arrays["history_sources"][0, 1, 0, 0]
    elif corruption == "offsets":
        arrays["history_offsets"][-1] += cfg.beam_size
    elif corruption == "field":
        arrays.pop("state_review")
    elif corruption == "schema":
        metadata["schema"] = "not_this_tracker"
    else:
        metadata["node_recovery_diagnostics"]["calls"] += 1
    receiver = C.ContinuousPersistentNodeTracker(rig, pose, cfg)
    with pytest.raises(ValueError):
        receiver.restore_state(arrays, metadata)
    assert receiver.initialize_count == 0 and receiver.processed_frames == 0


def test_accelerated_class_preserves_state_methods_and_only_replaces_fitter():
    cls = C.AcceleratedContinuousPersistentNodeTracker
    assert cls._fit is M["accelerated_tracker"].AcceleratedThreeBodyTracker._fit
    for method in ("begin", "advance", "finish", "export_state", "restore_state"):
        assert getattr(cls, method) is getattr(C.ContinuousPersistentNodeTracker, method)
    assert cls._proposals is P.PersistentNodeThreeBodyTracker._proposals

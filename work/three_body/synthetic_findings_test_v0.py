"""Joint observations and temporal branches, without real-video identity claims."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import project
from medaka3d.types import VIEWS

_LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def isolated_lab_modules():
    """Resolve this lab's absolute imports without overwriting other lab aliases."""
    names = ("prepare_sequence", "observations", "node_geometry", "renderer", "tracker")
    original = {name: sys.modules.get(name) for name in names}
    loaded = {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(f"three_body_tracker_test_{name}",
                                                          _LAB / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        for name, prior in original.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
    return loaded


_MODULES = isolated_lab_modules()
_TRACKER = _MODULES["tracker"]
_OBSERVATIONS = _MODULES["observations"]
_GEOMETRY = _MODULES["node_geometry"]
ThreeBodyTracker = _TRACKER.ThreeBodyTracker
TrackerConfig = _TRACKER.TrackerConfig
FrameObservation = _OBSERVATIONS.FrameObservation


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def config(**changes):
    return replace(TrackerConfig(device="cpu", mode="nodes", beam_size=2,
                                 max_hypotheses=4, opt_steps=6,
                                 positive_samples=48, near_samples=32, global_samples=16), **changes)


@pytest.fixture
def pose():
    centers = np.array([[60., 65., 70.], [115., 105., 100.], [145., 65., 140.]])
    return centers[:, None] + np.array([[5., 0., 0.], [0., 0., 0.], [-12., 1., -1.]])[None]


def shuffled_detections(rig, pose, seed):
    rng = np.random.default_rng(seed)
    points, sources = {}, np.empty((3, 3, 3), np.int32)
    for vi, view in enumerate(VIEWS):
        truth = project(rig.camera(view), pose)
        points[view] = np.empty_like(truth)
        for node in range(3):
            order = rng.permutation(3)
            points[view][:, node] = truth[order, node]
            sources[:, vi, node] = np.argsort(order)
    return points, sources


def frame_with_nodes(rig, pose, frame_idx):
    points, sources = shuffled_detections(rig, pose, frame_idx)
    return FrameObservation(frame_idx, points, {}, {}, {}), sources


def test_crossing_and_node_row_shuffle_keep_the_initial_fish_correspondence(rig, pose):
    initial = pose.copy()
    initial[:2] += np.array([[18., 25., 15.], [-21., -15., -5.]])[:, None]
    truth, frames, sources = [], [], []
    for index in range(8):
        current = initial.copy()
        current[0] += [2 * index, 0, 0]
        current[1] -= [2 * index, 0, 0]
        frame, expected_sources = frame_with_nodes(rig, current, 100 + index)
        truth.append(current)
        frames.append(frame)
        sources.append(expected_sources)
    tracker = ThreeBodyTracker(rig, initial, config(opt_steps=8, learning_rate_mm=0.3))
    result = tracker.track(frames)
    np.testing.assert_allclose(result["pose_xyz"][:, :, 1], np.asarray(truth)[:, :, 1], atol=1.0)
    np.testing.assert_array_equal(result["sources"], sources)
    assert result["observed"].all()
    assert np.isfinite(result["xyz"]).all()
    assert tracker.initialize_count == 1
    for source in result["sources"]:
        for view in range(3):
            for node in range(3):
                assert len(set(source[:, view, node])) == 3


def test_single_use_and_discontinuous_input_fail_without_hidden_reset(rig, pose):
    first, _ = frame_with_nodes(rig, pose, 10)
    tracker = ThreeBodyTracker(rig, pose, config(opt_steps=1))
    tracker.track([first])
    with pytest.raises(RuntimeError, match="single-use"):
        tracker.track([first])
    later, _ = frame_with_nodes(rig, pose, 12)
    with pytest.raises(ValueError, match="continuous native frames"):
        ThreeBodyTracker(rig, pose, config(opt_steps=1)).track([first, later])


def test_missing_node_and_view_do_not_become_observed_from_the_latent_pose(rig, pose):
    points, sources = shuffled_detections(rig, pose, 20)
    for vi in (1, 2):
        points[VIEWS[vi]][sources[2, vi, 2], 2] = np.nan
    points[VIEWS[2]] = np.empty((0, 3, 2))
    frame = FrameObservation(20, points, {}, {}, {})
    result = ThreeBodyTracker(rig, pose, config(opt_steps=2)).track([frame])
    assert not result["observed"][0, 2, 2]
    assert result["observed"][0, :2].all()
    assert (result["sources"][0, :, 2] == -1).all()
    assert np.isfinite(result["pose_xyz"]).all()


def test_completely_missing_evidence_has_finite_latent_bodies_but_no_measured_positions(rig, pose):
    empty = {view: np.empty((0, 3, 2)) for view in VIEWS}
    frames = [FrameObservation(index, empty, {}, {}, {}) for index in range(3)]
    result = ThreeBodyTracker(rig, pose, config(opt_steps=1, review_gap_frames=2)).track(frames)
    assert np.isfinite(result["pose_xyz"]).all()
    assert not result["observed"].any() and not result["body_supported"].any()
    assert (result["sources"] == -1).all()
    assert np.isnan(result["xyz"]).all()
    assert np.isnan(result["accepted_xyz"]).all()
    assert np.isnan(result["body_xyz"]).all()
    assert result["identity_review"][-1].all()


def ellipsoid_masks(rig, pose, shape=(320, 320)):
    """Independent binary silhouettes from analytic ray/ellipsoid intersections."""
    yy, xx = np.indices(shape)
    pixels = np.stack([xx, yy], axis=-1)
    masks = {}
    for camera in rig.cameras:
        origin, direction = camera.rays(pixels)
        rendered = []
        for body in pose:
            center = (body[0] + body[2]) / 2
            axis = body[0] - body[2]
            length = np.linalg.norm(axis)
            tangent = axis / length
            axial_radius, radial_radius = length / 2, 0.095 * length
            inverse = (np.eye(3) / radial_radius ** 2
                       + (1 / axial_radius ** 2 - 1 / radial_radius ** 2)
                       * np.outer(tangent, tangent))
            offset = origin - center
            a = np.einsum("...i,ij,...j->...", direction, inverse, direction)
            b = 2 * np.einsum("...i,ij,...j->...", direction, inverse, offset)
            c = np.einsum("...i,ij,...j->...", offset, inverse, offset) - 1
            discriminant = b * b - 4 * a * c
            furthest = (-b + np.sqrt(np.maximum(discriminant, 0))) / (2 * a)
            rendered.append((discriminant >= 0) & (furthest >= 0))
        masks[camera.view] = np.stack(rendered)
    return masks


@pytest.fixture
def mask_case(rig, pose):
    rig = replace(rig, cameras=tuple(replace(cam, fx=700., fy=700., cx=160., cy=160.)
                                     for cam in rig.cameras))
    masks = ellipsoid_masks(rig, pose)
    black = np.zeros((320, 320, 3), dtype=np.uint8)
    images = {view: black for view in VIEWS}
    detections = {view: np.empty((0, 3, 2)) for view in VIEWS}
    return rig, FrameObservation(30, detections, images, masks, images)


def test_joint_union_penalizes_duplicate_bodies_and_unexplained_third_foreground(mask_case, pose):
    rig, frame = mask_case
    cfg = config(mode="detectron2", opt_steps=1, learning_rate_mm=1e-7,
                 motion_weight=0, repulsion_weight=0)
    tracker = ThreeBodyTracker(rig, pose, cfg)
    duplicate = pose.copy()
    duplicate[1] = pose[0]
    missing_plan = np.full((3, 3, 3), -1, np.int32)
    state = _TRACKER.State(pose, np.zeros((3, 3)), np.zeros(3), np.zeros(3, bool), 0)
    proposals = [_TRACKER.Proposal(value, value, missing_plan, 0, "test")
                 for value in (pose, duplicate)]
    rays = _GEOMETRY.RayObservations.from_points(rig, frame.detections)
    targets = _OBSERVATIONS.sample_pixels(frame, n_positive=96, n_near=64, n_global=32)
    _, scores, parts, _ = tracker._fit(proposals, [state], rays, targets)
    assert parts[1, 4] > parts[0, 4] + 0.08
    assert scores[1] > scores[0]
    np.testing.assert_allclose(parts[0, :3], parts[1, :3], atol=1e-8)


def test_masks_move_nearby_latent_bodies_without_fabricating_direct_node_support(mask_case, pose):
    rig, frame = mask_case
    initial = pose + np.array([1.5, -1., 0.5])
    base = config(opt_steps=14, learning_rate_mm=0.2, motion_weight=0,
                  repulsion_weight=0, review_gap_frames=1)
    nodes = ThreeBodyTracker(rig, initial, base).track([frame])
    masks = ThreeBodyTracker(rig, initial, replace(base, mode="detectron2", mask_weight=10)).track(
        [frame]
    )
    node_error = np.linalg.norm(nodes["pose_xyz"][0, :, 1] - pose[:, 1], axis=-1).mean()
    mask_error = np.linalg.norm(masks["pose_xyz"][0, :, 1] - pose[:, 1], axis=-1).mean()
    assert mask_error < node_error - 0.1
    assert not masks["observed"].any() and not masks["body_supported"].any()
    assert np.isnan(masks["xyz"]).all() and np.isnan(masks["accepted_xyz"]).all()
    assert masks["identity_review"].all()


def test_converged_current_poses_keep_distinct_parents_for_future_backtrace(rig, pose, monkeypatch):
    tracker = ThreeBodyTracker(rig, pose, config(opt_steps=1, ambiguity_margin=0))
    frames = [frame_with_nodes(rig, pose, 40 + index)[0] for index in range(3)]
    swap = np.array([1, 0, 2])
    generation = {"index": 0}
    beam_counts = []

    def proposals(states, rays):
        index = generation["index"]
        beam_counts.append(len(states))
        base_plan = _GEOMETRY.assignment_plans(pose, rays, keep=1)[0][1]
        if index == 0:
            return [_TRACKER.Proposal(pose, pose, base_plan, 0, "first"),
                    _TRACKER.Proposal(pose[swap], pose, base_plan[swap], 0, "alternative")]
        return [_TRACKER.Proposal(pose, pose, base_plan, parent, "converged")
                for parent in range(len(states))]

    def fitted(proposals, states, rays, targets):
        index = generation["index"]
        generation["index"] += 1
        count = len(proposals)
        poses = np.stack([proposal.pose for proposal in proposals])
        costs = ([0., 0.2] if index == 0 else [0., 0.] if index == 1 else [5., 0.])[:count]
        return poses, np.asarray(costs), np.zeros((count, 5)), np.zeros((count, 3), bool)

    monkeypatch.setattr(tracker, "_proposals", proposals)
    monkeypatch.setattr(tracker, "_fit", fitted)
    result = tracker.track(frames)
    assert beam_counts == [1, 2, 2]
    np.testing.assert_array_equal(result["pose_xyz"][0], pose[swap])
    assert result["path_costs"][-1] == pytest.approx(0.2)

"""Joint observations and temporal branches, without real-video identity claims."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from conftest import project
from medaka3d.stages import Linker
from medaka3d.types import VIEWS, Instance3D

_LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def isolated_lab_modules():
    """Resolve this lab's absolute imports without overwriting other lab aliases."""
    names = ("prepare_sequence", "observations", "node_geometry", "renderer", "fast_union",
             "motion", "tracker", "linker")
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
                                 positive_samples=48, near_samples=32, global_samples=16),
                   **changes)


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
    correct = result["sources"] == np.asarray(sources)
    assert (correct.sum(axis=2) >= 2).all()
    assert (result["sources"] >= 0).all()
    # Coincident/nearby projected nodes need not have an identifiable row label
    # in one view. The other two views and the 3D trajectories disambiguate them.
    # Use the stated 0.7-mm ray sigma, converted to this pinhole camera's pixels,
    # rather than choosing a pixel threshold from the observed mismatches.
    for index, fish, view_index, node in np.argwhere(~correct):
        view = VIEWS[view_index]
        camera = rig.camera(view)
        depth = ((truth[index][fish, node] - camera.center) @ camera.rotation.T)[2]
        precision_px = tracker.config.ray_sigma_mm * max(camera.fx, camera.fy) / depth
        actual_row = result["sources"][index, fish, view_index, node]
        actual = frames[index].detections[view][actual_row, node]
        expected = frames[index].detections[view][sources[index][fish, view_index, node], node]
        assert np.linalg.norm(actual - expected) <= precision_px
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


def disjoint_pixel_targets(frame, fit_pixels):
    """Within-image mask check on pixels not used by the optimizer, not new-animal validation."""
    pixels, target, weight = {}, [], []
    for vi, view in enumerate(VIEWS):
        union = np.any(frame.masks[view], axis=0)
        height, width = union.shape
        used = np.zeros(height * width, bool)
        xy = fit_pixels[view].astype(int)
        used[xy[:, 1] * width + xy[:, 0]] = True
        band = cv2.dilate(union.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & ~union
        pools = [np.flatnonzero(union.ravel() & ~used), np.flatnonzero(band.ravel() & ~used)]
        rng = np.random.default_rng(629 + vi)
        flat = np.concatenate([rng.choice(pool, 96, replace=len(pool) < 96) for pool in pools])
        assert not used[flat].any()
        yy, xx = np.unravel_index(flat, (height, width))
        pixels[view] = np.column_stack([xx, yy])
        target.append(np.r_[np.ones(96), np.zeros(96)])
        weight.append(np.full(192, 1 / 192))
    return pixels, torch.tensor(np.array(target)), torch.tensor(np.array(weight))


def test_masks_move_nearby_latent_bodies_without_fabricating_direct_node_support(mask_case, pose):
    rig, frame = mask_case
    initial = pose + np.array([1.5, -1., 0.5])
    base = config(opt_steps=14, learning_rate_mm=0.2, motion_weight=0,
                  repulsion_weight=0, review_gap_frames=1)
    nodes = ThreeBodyTracker(rig, initial, base).track([frame])
    masks = ThreeBodyTracker(rig, initial, replace(base, mode="detectron2", mask_weight=10)).track(
        [frame]
    )
    fit_targets = _OBSERVATIONS.sample_pixels(frame, n_positive=48, n_near=32, n_global=16)
    pixels, target, weight = disjoint_pixel_targets(frame, fit_targets.pixels)
    renderer = _MODULES["renderer"].RefractiveFishRenderer(rig, pixels, dtype=torch.float64)
    node_prediction = renderer.render(nodes["pose_xyz"][0]).union_alpha
    mask_prediction = renderer.render(masks["pose_xyz"][0]).union_alpha
    node_loss = ((node_prediction - target) ** 2 * weight).sum() / 3
    mask_loss = ((mask_prediction - target) ** 2 * weight).sum() / 3
    assert mask_loss < node_loss - 0.01
    assert np.linalg.norm(masks["pose_xyz"] - nodes["pose_xyz"]) > 0.1
    # Anatomical pectoral error worsened in this different-body-model example;
    # that failure and both losses are preserved in synthetic_findings.json.
    # Pixel agreement must never be silently promoted to observed SLEAP nodes.
    assert not masks["observed"].any() and not masks["body_supported"].any()
    assert np.isnan(masks["xyz"]).all() and np.isnan(masks["accepted_xyz"]).all()
    assert masks["identity_review"].all()


@pytest.mark.parametrize("known_empty", [False, True])
def test_empty_or_missing_masks_do_not_create_body_support(mask_case, pose, known_empty):
    rig, frame = mask_case
    masks = {view: np.zeros((0, 320, 320), bool) if known_empty else None for view in VIEWS}
    frame = replace(frame, masks=masks)
    result = ThreeBodyTracker(rig, pose, config(mode="detectron2", opt_steps=1)).track([frame])
    assert not result["mask_supported"].any()
    assert not result["body_supported"].any()
    assert not result["observed"].any()
    assert np.isnan(result["accepted_xyz"]).all()


def test_one_mask_view_and_one_head_ray_cannot_create_multiview_body_support(mask_case, pose):
    rig, frame = mask_case
    detections = {view: np.empty((0, 3, 2)) for view in VIEWS}
    detections[VIEWS[0]] = project(rig.camera(VIEWS[0]), pose)
    masks = {view: frame.masks[view] if index == 0 else None
             for index, view in enumerate(VIEWS)}
    frame = replace(frame, detections=detections, masks=masks)
    result = ThreeBodyTracker(rig, pose, config(mode="detectron2", opt_steps=2)).track([frame])
    assert (result["sources"][0, :, 0] >= 0).all()
    assert (result["sources"][0, :, 1:] == -1).all()
    assert not result["mask_supported"].any()
    assert not result["body_supported"].any()
    assert not result["observed"].any()
    assert np.isnan(result["accepted_xyz"]).all()


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


def test_linker_protocol_uses_only_first_frame_instances_and_cannot_reset(rig, pose):
    calls = []
    frames = [frame_with_nodes(rig, pose, 70 + index)[0] for index in range(2)]

    def factory():
        calls.append(1)
        return iter(frames)

    adapter = _MODULES["linker"].ThreeBodyLinker(
        rig, 70, factory, config(opt_steps=1, beam_size=1, max_hypotheses=1))
    assert isinstance(adapter, Linker)
    seeds = [Instance3D(frame=70, xyz=body, slot=slot)
             for slot, body in zip([7, 19, 42], pose, strict=True)]
    # Later pipeline instances may be incomplete or wrong; they must not become
    # external resets, constraints, or a replacement for the observation stream.
    poison = [Instance3D(frame=71, xyz=np.full((3, 3), np.nan), slot=7),
              Instance3D(frame=72, xyz=np.full((3, 3), 1e8), slot=999)]
    output = adapter.link([*reversed(seeds), *poison])
    assert output.shape == (2, 3, 3)
    np.testing.assert_allclose(output, np.broadcast_to(pose[:, 1], output.shape), atol=1e-4)
    assert adapter.tracker.initialize_count == 1
    assert calls == [1]
    with pytest.raises(RuntimeError, match="new linker"):
        adapter.link(seeds)
    assert calls == [1]


@pytest.mark.parametrize("mismatch", ["instances", "observations"])
def test_linker_rejects_a_seed_or_observation_frame_mismatch(rig, pose, mismatch):
    first = 81 if mismatch == "observations" else 80
    frame = frame_with_nodes(rig, pose, first)[0]
    adapter = _MODULES["linker"].ThreeBodyLinker(
        rig, 80, lambda: iter([frame]), config(opt_steps=1))
    seed_frame = 79 if mismatch == "instances" else 80
    seeds = [Instance3D(frame=seed_frame, xyz=body, slot=fish) for fish, body in enumerate(pose)]
    with pytest.raises(ValueError, match=r"first-frame|schedule"):
        adapter.link(seeds)


@pytest.mark.parametrize("partial", [False, True])
def test_flow_candidates_preserve_parent_predictions_and_use_current_node_rows(rig, pose, partial):
    tracker = ThreeBodyTracker(rig, pose, config(flow_proposals=True, max_hypotheses=8))
    velocity = np.array([[0.3, 0., 0.], [0., -0.2, 0.], [0., 0., 0.15]])
    states = [_TRACKER.State(pose.copy(), velocity, np.zeros(3, int), np.zeros(3, bool), 0),
              _TRACKER.State(pose + 0.1, -velocity, np.zeros(3, int), np.zeros(3, bool), 0.2)]
    flow_pose = pose + np.array([[2.5, 0.4, 0.1], [-2., 0.6, -0.2], [0., 2.8, 0.3]])[:, None]
    flow_pose[1, 0] += [0.4, 0., 0.]
    flow_pose[1, 2] -= [0.4, 0., 0.]
    valid = (np.array([[False, True, False], [True, False, True], [False, False, False]])
             if partial else np.ones((3, 3), bool))
    supplied = np.where(valid[..., None], flow_pose, np.nan)
    rigid, warped = pose + velocity[:, None], pose + velocity[:, None]
    for fish in range(3):
        available = valid[fish]
        if available.any():
            shift = np.median(flow_pose[fish, available] - pose[fish, available], axis=0)
            rigid[fish] = pose[fish] + shift
            warped[fish] = rigid[fish]
            warped[fish, available] = flow_pose[fish, available]
    points, expected_sources = shuffled_detections(rig, warped, 922)
    _, previous_sources = shuffled_detections(rig, pose, 114)
    rays = _GEOMETRY.RayObservations.from_points(rig, points)
    flow = SimpleNamespace(node_valid=valid, pose_xyz=supplied)
    proposals = tracker._proposals(states, rays, flow)
    predictions = [proposal for proposal in proposals if proposal.kind == "predict"]
    assert len(predictions) == len(states)
    for proposal in predictions:
        expected = states[proposal.parent].pose + states[proposal.parent].velocity[:, None]
        np.testing.assert_array_equal(proposal.pose, expected)
        np.testing.assert_array_equal(proposal.predicted, expected)
    candidates = {proposal.kind: proposal for proposal in proposals
                  if proposal.kind.startswith("image_flow")}
    assert set(candidates) == {"image_flow_nodes", "image_flow_translation"}
    np.testing.assert_allclose(candidates["image_flow_nodes"].pose, warped)
    np.testing.assert_allclose(candidates["image_flow_translation"].pose, rigid)
    for proposal in candidates.values():
        assert proposal.parent == 0
        np.testing.assert_array_equal(proposal.predicted, pose + velocity[:, None])
        np.testing.assert_array_equal(proposal.plan, expected_sources)
        assert np.any(proposal.plan != previous_sources)
    np.testing.assert_array_equal(states[0].pose, pose)
    np.testing.assert_array_equal(flow.pose_xyz, supplied)


def test_invalid_flow_is_neutral_and_never_replaces_constant_velocity(rig, pose):
    tracker = ThreeBodyTracker(rig, pose, config(flow_proposals=True))
    state = _TRACKER.State(pose, np.full((3, 3), 0.2), np.zeros(3, int), np.zeros(3, bool), 0)
    points, _ = shuffled_detections(rig, pose, 73)
    rays = _GEOMETRY.RayObservations.from_points(rig, points)
    baseline = tracker._proposals([state], rays)
    flow = SimpleNamespace(node_valid=np.zeros((3, 3), bool), pose_xyz=np.full((3, 3, 3), np.nan))
    proposals = tracker._proposals([state], rays, flow)
    assert [proposal.kind for proposal in proposals] == [proposal.kind for proposal in baseline]
    for actual, expected in zip(proposals, baseline, strict=True):
        np.testing.assert_array_equal(actual.pose, expected.pose)
        np.testing.assert_array_equal(actual.predicted, expected.predicted)
        np.testing.assert_array_equal(actual.plan, expected.plan)


def test_selected_image_flow_cannot_promote_missing_sleap_nodes_to_observed(
    mask_case, pose, monkeypatch
):
    rig, frame = mask_case
    initial_points, _ = shuffled_detections(rig, pose, frame.frame_idx)
    first = replace(frame, detections=initial_points)
    current_pose = pose + np.array([1.8, -0.6, 0.4])
    current = replace(frame, frame_idx=frame.frame_idx + 1,
                      masks=ellipsoid_masks(rig, current_pose))
    calls = []

    class ControlledFlow:
        @classmethod
        def from_frames(cls, previous, current, rig, previous_pose_xyz):
            calls.append((previous.frame_idx, current.frame_idx, previous_pose_xyz.copy()))
            return SimpleNamespace(node_valid=np.ones((3, 3), bool), pose_xyz=current_pose.copy())

    motion = _MODULES["motion"]
    monkeypatch.setattr(motion, "FlowObservations", ControlledFlow)
    monkeypatch.setitem(sys.modules, "motion", motion)  # The tracker imports flow lazily.
    cfg = config(mode="detectron2", flow_proposals=True, max_hypotheses=8, opt_steps=2,
                 learning_rate_mm=0.1, mask_weight=10, motion_weight=0, repulsion_weight=0)
    result = ThreeBodyTracker(rig, pose, cfg).track([first, current])
    assert len(calls) == 1 and calls[0][:2] == (frame.frame_idx, frame.frame_idx + 1)
    assert result["proposal_kind"][1].startswith("image_flow")
    assert result["proposal_flow_nodes"][1].all()
    assert result["mask_supported"][1].any()
    assert not result["observed"][1].any()
    assert not result["body_supported"][1].any()
    assert (result["sources"][1] == -1).all()
    assert np.isnan(result["xyz"][1]).all()
    assert np.isnan(result["accepted_xyz"][1]).all()
    assert np.isfinite(result["pose_xyz"][1]).all()


def test_coincident_center_permutations_are_reviewed_despite_argmin_tie(rig, pose, monkeypatch):
    tied = pose.copy()
    tied[1] = tied[0]
    tracker = ThreeBodyTracker(rig, tied, config(opt_steps=1))
    frame, _ = frame_with_nodes(rig, tied, 93)

    def proposals(states, rays):
        plan = _GEOMETRY.assignment_plans(tied, rays, keep=1)[0][1]
        return [_TRACKER.Proposal(tied, tied, plan, 0, "predict"),
                _TRACKER.Proposal(tied, tied, plan[[1, 0, 2]], 0, "body_permutation", (1, 0, 2))]

    def fitted(proposals, states, rays, targets):
        return np.stack([tied, tied]), np.zeros(2), np.zeros((2, 5)), np.zeros((2, 3), bool)

    monkeypatch.setattr(tracker, "_proposals", proposals)
    monkeypatch.setattr(tracker, "_fit", fitted)
    result = tracker.track([frame])
    assert result["identity_review"][0, :2].all()
    assert not result["identity_review"][0, 2]
    assert result["score_margin"][0] == 0
    assert np.isnan(result["accepted_xyz"][0, :2]).all()
    assert np.isfinite(result["accepted_xyz"][0, 2]).all()

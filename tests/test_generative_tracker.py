"""A 3D hypothesis must explain observations without inheriting detector IDs.

Synthetic scenes exercise known failure mechanisms, not biological identity
accuracy on the experiment. Real-video comparison is a separate benchmark.
"""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

_LAB = Path(__file__).resolve().parents[1] / "labs/20260908_generative_tracking"


def load_lab_module(filename):
    spec = importlib.util.spec_from_file_location(
        f"generative_{filename}_test", _LAB / f"{filename}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TRACKER = load_lab_module("tracker")
_OBSERVATIONS = load_lab_module("observations")
FrameInput = _TRACKER.FrameInput
GenerativeTracker = _TRACKER.GenerativeTracker
TrackerConfig = _TRACKER.TrackerConfig


@pytest.fixture
def refractive_rig(rig):
    return replace(rig, cameras=tuple(
        replace(cam, index_air=1.0003, index_glass=1.52, index_water=1.3325,
                layer_count=1 if cam.view == "top_cam" else 2,
                distortion=(-0.222, 0.204, 0.0044, 0.0019, -0.112))
        for cam in rig.cameras
    ))


@pytest.fixture
def trio():
    centers = np.array([[45.0, 50.0, 40.0], [130.0, 90.0, 90.0], [75.0, 150.0, 150.0]])
    # Node order is mouth, pectoral, tail. The pectoral is the state anchor.
    body = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-12.0, 0.0, 0.0]])
    return centers[:, None] + body[None]


def detections_for(rig, pose):
    return {cam.view: cam.project(pose) for cam in rig.cameras}


def run_tracker(rig, initial, detections, frames=None):
    frames = range(len(detections)) if frames is None else frames
    config = TrackerConfig(beam_width=2, proposals_per_parent=2, max_nfev=8)
    inputs = [FrameInput(frame, observations)
              for frame, observations in zip(frames, detections, strict=True)]
    return GenerativeTracker(rig, initial, config).track(inputs)


def test_observations_change_the_initial_3d_hypothesis(refractive_rig, trio):
    target = trio + np.array([1.5, -0.5, 0.7])
    observations = detections_for(refractive_rig, target)
    result = run_tracker(refractive_rig, trio, [observations])
    initial_error = np.linalg.norm(trio - target, axis=-1)
    fitted_error = np.linalg.norm(result.pose_xyz[0] - target, axis=-1)
    # Tail observations have lower weight, so they can remain closer to the
    # quadratic motion prior. Each node must move toward the evidence; the
    # whole fitted skeleton must substantially improve over the seed.
    assert np.all(fitted_error < initial_error)
    assert fitted_error.mean() < initial_error.mean() * 0.5
    assert result.observed[0].all()


def test_refraction_noise_and_detector_row_permutation(refractive_rig, trio):
    rng = np.random.default_rng(885)
    orders = [[2, 0, 1], [1, 2, 0], [2, 1, 0]]
    observations = {}
    for cam, order in zip(refractive_rig.cameras, orders, strict=True):
        uv = cam.project(trio)
        observations[cam.view] = uv[order] + rng.normal(0, 0.2, uv.shape)
    result = run_tracker(refractive_rig, trio, [observations])
    assert np.max(np.linalg.norm(result.pose_xyz[0] - trio, axis=-1)) < 0.3
    assert result.observed[0].all()
    for view, order in enumerate(orders):
        expected = np.broadcast_to(np.argsort(order)[:, None], (3, 3))
        np.testing.assert_array_equal(result.node_source[0, :, view], expected)


def test_exactly_three_detections_can_include_a_false_positive(refractive_rig, trio):
    observations = detections_for(refractive_rig, trio)
    observations["top_cam"][2] += [450.0, -350.0]
    result = run_tracker(refractive_rig, trio, [observations])
    assert (result.node_source[0, 2, 0] == -1).all()
    assert result.observed[0, 2].all()  # two clean sides still observe this fish
    np.testing.assert_allclose(result.pose_xyz[0], trio, atol=0.1, rtol=0)


def test_exchanged_tail_does_not_discard_correct_mouth_and_pectoral(refractive_rig, trio):
    observations = detections_for(refractive_rig, trio)
    observations["top_cam"][[0, 1], 2] = observations["top_cam"][[1, 0], 2]
    result = run_tracker(refractive_rig, trio, [observations])
    np.testing.assert_array_equal(result.node_source[0, 0, 0], [0, 0, 1])
    np.testing.assert_array_equal(result.node_source[0, 1, 0], [1, 1, 0])
    assert result.observed[0].all()
    np.testing.assert_allclose(result.pose_xyz[0], trio, atol=0.1, rtol=0)


@pytest.mark.parametrize("visible_views", [0, 1])
def test_hidden_fish_has_latent_state_but_no_measured_xyz(refractive_rig, trio, visible_views):
    observations = detections_for(refractive_rig, trio)
    for cam in refractive_rig.cameras[visible_views:]:
        observations[cam.view][1] = np.nan
    result = run_tracker(refractive_rig, trio, [observations])
    assert not result.observed[0, 1].any()
    assert np.isnan(result.xyz[0, 1]).all()
    assert np.isfinite(result.pose_xyz[0, 1]).all()
    assert result.observed[0, [0, 2]].all()


def test_inputs_are_not_mutated_and_no_future_reference_poses_are_supplied(refractive_rig, trio):
    initial = trio.copy()
    future = [trio + step * np.array([0.4, 0.1, 0.0]) for step in range(4)]
    observations = [detections_for(refractive_rig, pose) for pose in future]
    before = [{view: uv.copy() for view, uv in frame.items()} for frame in observations]
    result = run_tracker(refractive_rig, initial, observations)
    np.testing.assert_array_equal(initial, trio)
    for previous, current in zip(before, observations, strict=True):
        for view in previous:
            np.testing.assert_array_equal(previous[view], current[view])
    # Only the first-frame seed was passed as 3D. Motion after it must come
    # from the 2D images, not an old tracker's future positions or identities.
    assert np.mean(result.pose_xyz[-1, :, :, 0] - trio[:, :, 0]) > 0.6
    assert np.max(np.linalg.norm(result.pose_xyz[-1] - future[-1], axis=-1)) < 0.5


def test_non_increasing_frame_sequence_is_rejected(refractive_rig, trio):
    observations = [detections_for(refractive_rig, trio)] * 2
    with pytest.raises(ValueError, match=r"frame|increasing|order"):
        run_tracker(refractive_rig, trio, observations, frames=[10, 10])


@pytest.mark.parametrize("separation_mm", [0.0, 2.0])
def test_coherent_identity_alternative_survives_cheap_single_node_omissions(
    refractive_rig, trio, separation_mm,
):
    nearby = trio.copy()
    nearby[1] = nearby[0] + [separation_mm, 0.0, 0.0]
    cfg = TrackerConfig(beam_width=3, proposals_per_parent=3)
    tracker = GenerativeTracker(refractive_rig, nearby, cfg)
    frame = FrameInput(0, detections_for(refractive_rig, nearby))
    plans = [plan for _, plan in tracker._plans(nearby, frame)]
    identity = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3))
    exchanged = identity[[1, 0, 2]]
    assert any(np.array_equal(plan, identity) for plan in plans)
    assert any(np.array_equal(plan, exchanged) for plan in plans)


def test_real_image_masks_affect_3d_fit_without_inventing_measured_points(refractive_rig, trio):
    images, backgrounds = {}, {}
    for cam in refractive_rig.cameras:
        image = np.zeros((1200, 1200, 3), np.uint8)
        for fish in cam.project(trio):
            cv2.polylines(image, [np.rint(fish).astype(np.int32)], False, (230, 230, 230), 3)
        images[cam.view] = image
        backgrounds[cam.view] = np.zeros_like(image)
    evidence = _OBSERVATIONS.SceneEvidence.from_frames(images, backgrounds)
    missing = {view: np.empty((0, 3, 2)) for view in refractive_rig.views}
    initial = trio + np.array([0.0, 2.0, 0.0])
    cfg = TrackerConfig(beam_width=2, proposals_per_parent=2, max_nfev=20,
                        image_weight=3.0, motion_sigma_mm=3.0)
    with_images = GenerativeTracker(refractive_rig, initial, cfg).track([
        FrameInput(0, missing, evidence)
    ])
    without_images = GenerativeTracker(refractive_rig, initial, cfg).track([
        FrameInput(0, missing)
    ])
    image_error = np.mean(np.linalg.norm(with_images.pose_xyz[0] - trio, axis=-1))
    prior_error = np.mean(np.linalg.norm(without_images.pose_xyz[0] - trio, axis=-1))
    assert image_error < 0.75 * prior_error
    assert not with_images.observed.any()
    assert np.isnan(with_images.xyz).all()


def test_linker_cannot_read_a_future_baseline_path(refractive_rig, trio):
    from medaka3d.types import Instance3D

    frames = [FrameInput(frame, detections_for(refractive_rig, trio + frame * 0.2))
              for frame in range(3)]
    seeds = [Instance3D(0, pose, slot) for slot, pose in enumerate(trio)]
    # A later baseline item deliberately contains no usable skeleton. Any use
    # after initialization must either fail or measurably contaminate the fit.
    poisoned = [*seeds, Instance3D(1, np.full((3, 3), np.nan), 0),
                Instance3D(2, np.full((3, 3), 1e6), 1)]
    cfg = TrackerConfig(beam_width=2, proposals_per_parent=2, max_nfev=8)
    clean = _TRACKER.GenerativeLinker(refractive_rig, frames, cfg).link(seeds)
    with_future = _TRACKER.GenerativeLinker(refractive_rig, frames, cfg).link(poisoned)
    np.testing.assert_array_equal(clean, with_future)


def test_linker_withholds_reviewed_but_measured_positions(refractive_rig, trio):
    from medaka3d.types import Instance3D

    detections = detections_for(refractive_rig, trio)
    # A missing mouth leaves measured pectorals, but the fish requires review.
    for points in detections.values():
        points[0, 0] = np.nan
    frames = [FrameInput(0, detections)]
    seeds = [Instance3D(0, pose, slot) for slot, pose in enumerate(trio)]
    linker = _TRACKER.GenerativeLinker(refractive_rig, frames)
    accepted = linker.link(seeds)
    assert linker.result.review[0, 0]
    assert np.isfinite(linker.result.xyz[0, 0]).all()
    assert np.isnan(accepted[0, 0]).all()
    assert np.isfinite(accepted[0, 1:]).all()


def test_later_keypoints_change_an_earlier_retained_identity_branch(refractive_rig, trio):
    pose = trio.copy()
    pose[0] = np.array([50.0, 80.0, 90.0]) + np.array([[0, 5, 0], [0, 0, 0], [0, -12, 0]])
    pose[1] = pose[0] + [8.0, 0.0, 0.0]
    crossing, separated = pose.copy(), pose.copy()
    crossing[0, :, 0], crossing[1, :, 0] = 55.0, 53.0
    separated[0, :, 0], separated[1, :, 0] = 60.0, 48.0
    frames = [FrameInput(i, detections_for(refractive_rig, current))
              for i, current in enumerate([pose, crossing, separated])]
    cfg = TrackerConfig(beam_width=2, proposals_per_parent=2, max_nfev=15,
                        missing_cost=100.0, pixel_sigma=1.0, motion_sigma_mm=2.0,
                        velocity_update=0.8)
    prefix = GenerativeTracker(refractive_rig, pose, cfg).track(frames[:2])
    full = GenerativeTracker(refractive_rig, pose, cfg).track(frames)
    # At the close approach, nearest continuation favours the other fish.
    # The next image constrains velocity and changes the backtracked parent.
    assert prefix.pose_xyz[1, 0, 1, 0] < 54.0
    assert full.pose_xyz[1, 0, 1, 0] > 54.0
    assert full.pose_xyz[2, 0, 1, 0] > 59.0


def test_tail_omission_competitors_do_not_create_identity_review(refractive_rig, trio):
    config = TrackerConfig(beam_width=3, proposals_per_parent=3, max_nfev=8,
                           review_margin=1000.0)
    result = GenerativeTracker(refractive_rig, trio, config).track([
        FrameInput(0, detections_for(refractive_rig, trio))
    ])
    # This beam really has a close-cost alternative, so the old filtering-gap
    # review rule would flag all animals. Its difference is only node support.
    assert np.isfinite(result.margin[0])
    assert result.margin[0] < config.review_margin
    assert np.isinf(result.identity_margin).all()
    assert not result.review.any()
    assert result.diagnostics["review_competing_identity_animal_frames"] == 0


def test_close_fish_are_reviewed_when_the_beam_keeps_no_identity_alternative(refractive_rig, trio):
    nearby = trio.copy()
    nearby[1] = nearby[0] + [2.0, 0.0, 0.0]
    config = TrackerConfig(beam_width=1, proposals_per_parent=1, max_nfev=8)
    result = GenerativeTracker(refractive_rig, nearby, config).track([
        FrameInput(0, detections_for(refractive_rig, nearby))
    ])
    assert result.observed.all()
    assert np.isinf(result.identity_margin).all()  # absence of alternatives, not certainty
    np.testing.assert_array_equal(result.review[0], [True, True, False])
    assert result.diagnostics["review_close_without_alternative_animal_frames"] == 2


def test_review_threshold_changes_abstention_but_never_inferred_pose_or_sources(
    refractive_rig, trio,
):
    nearby = trio.copy()
    nearby[1] = nearby[0] + [2.0, 0.0, 0.0]
    frame = FrameInput(0, detections_for(refractive_rig, nearby))
    config = TrackerConfig(beam_width=2, proposals_per_parent=2, max_nfev=15,
                           motion_sigma_mm=4.0, review_margin=1e-6)
    low = GenerativeTracker(refractive_rig, nearby, config).track([frame])
    high = GenerativeTracker(
        refractive_rig, nearby, replace(config, review_margin=1e6)
    ).track([frame])
    np.testing.assert_array_equal(low.pose_xyz, high.pose_xyz)
    np.testing.assert_array_equal(low.node_source, high.node_source)
    np.testing.assert_array_equal(low.xyz, high.xyz)
    np.testing.assert_array_equal(low.identity_margin, high.identity_margin)
    assert np.isfinite(low.identity_margin[0, :2]).all()
    assert not low.review.any()
    np.testing.assert_array_equal(high.review[0], [True, True, False])


@pytest.mark.parametrize("corrupted_node", [1, 2], ids=["pectoral", "tail"])
def test_consistent_multiview_misdetections_cannot_become_a_grossly_stretched_fish(
    refractive_rig, trio, corrupted_node,
):
    initial = trio.copy()
    initial[0] += np.array([80.0, -10.0, -5.0])
    frames = []
    for frame in range(49):
        false_pose = initial.copy()
        # Smooth and geometrically consistent across cameras: reprojection
        # alone cannot reject this wrong node, and a single-frame motion gate
        # does not prevent gradual drift into an impossible animal skeleton.
        false_pose[0, corrupted_node, 0] -= frame
        frames.append(FrameInput(frame, detections_for(refractive_rig, false_pose)))
    config = TrackerConfig(beam_width=1, proposals_per_parent=1, max_nfev=15)
    result = GenerativeTracker(refractive_rig, initial, config).track(frames)
    seed_lengths = np.linalg.norm(np.diff(initial[0], axis=0), axis=-1)
    lengths = np.linalg.norm(np.diff(result.pose_xyz[:, 0], axis=1), axis=-1)
    grossly_stretched = (lengths > 3.0 * seed_lengths).any(axis=-1)
    fully_observed = result.observed[:, 0].all(axis=-1)
    assert not (grossly_stretched & fully_observed).any(), (
        f"a fully observed skeleton reached {np.max(lengths / seed_lengths):.2f} times "
        "its seed segment length under consistent false observations"
    )
    assert not grossly_stretched.any()  # even an abstained latent fish stays physically bounded
    # Geometric consistency cannot identify WHICH node is wrong. The model can
    # follow the false pectoral while refusing its contradictory head/tail;
    # that whole-fish review must prevent acceptance of the wrong center.
    accepted_corrupted_node = (result.observed[-10:, 0, corrupted_node]
                               & ~result.review[-10:, 0])
    assert not accepted_corrupted_node.any()
    assert result.observed[:, 1:].all()  # a bad node must not discard the other fish
    if corrupted_node == 1:
        assert np.isnan(result.xyz[~result.observed[:, 0, 1], 0]).all()

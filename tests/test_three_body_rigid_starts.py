"""Rigid image-derived starts preserve anatomy and explicitly retain exclusions."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from renderer import RefractiveFishRenderer  # noqa: E402
from rigid_body_starts import (  # noqa: E402
    RigidStartConfig,
    body_frame,
    gaussian_volume_center,
    oriented_bodies,
    random_scene_starts,
    volume_scene_starts,
)


def scene():
    return np.array([[40, 50, 60], [80, 90, 100], [130, 130, 140]])[:, None] + np.array(
        [[6., 0., 0.], [0., 0., 0.], [-16., 2., 1.]])[None]


def component():
    return dict(centroid=np.array([90., 100., 110.]), axes=np.eye(3))


def distances(pose):
    return np.linalg.norm(pose[..., :, None, :] - pose[..., None, :, :], axis=-1)


def test_volume_center_matches_actual_frozen_gaussian_parameters(rig):
    renderer = RefractiveFishRenderer(
        rig, {rig.views[0]: np.array([[500., 500.]])}, dtype=torch.float64
    )
    body = renderer.gaussians(scene())
    weight = (body.axial_sigma_mm * body.radial_sigma_mm.square()).numpy()[0]
    expected = np.sum(body.centers.numpy()[0] * weight[:, :, None], axis=1) / weight.sum(1)[:, None]
    np.testing.assert_allclose([gaussian_volume_center(p) for p in scene()], expected,
                               rtol=1e-14, atol=1e-14)


def test_eight_orientations_align_volume_centers_preserve_distances_and_chirality():
    body = scene()[0]
    target = component()
    poses, rotations, labels = oriented_bodies(body, **target)
    assert len(poses) == 8
    assert set(map(tuple, labels)) == {(s, r) for s in (-1, 1) for r in range(4)}
    for pose, rotation, (sign, _) in zip(poses, rotations, labels, strict=True):
        np.testing.assert_allclose(distances(pose), distances(body), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(gaussian_volume_center(pose), target['centroid'], atol=3e-14)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
        assert np.linalg.det(rotation) == pytest.approx(1.)
        axis = (pose[2] - pose[0]) / np.linalg.norm(pose[2] - pose[0])
        np.testing.assert_allclose(axis, sign * target['axes'][0], atol=1e-14)


def test_pca_axis_sign_ambiguity_does_not_change_eight_pose_set():
    body, target = scene()[0], component()
    expected = oriented_bodies(body, **target)[0]
    target['axes'] = np.diag([-1., -1., -1.])
    other = oriented_bodies(body, **target)[0]
    # PCA signs carry no anatomical identity or mouth-tail information.
    error = np.max(np.abs(expected[:, None] - other[None]), axis=(2, 3))
    assert np.max(np.min(error, axis=1)) < 1e-12
    assert np.max(np.min(error, axis=0)) < 1e-12


def test_scene_starts_move_every_raw_slot_not_only_missing_slots_and_keep_other_bodies():
    original = scene()
    saved = original.copy()
    result = volume_scene_starts(original, [component()])
    assert result['pose_xyz'].shape == (25, 3, 3, 3)
    assert result['admissible'].all()
    assert np.array_equal(result['pose_xyz'][0], original)
    for pose, (_, slot, _, _) in zip(
            result['pose_xyz'][1:], result['component_slot_sign_roll'][1:], strict=True):
        np.testing.assert_array_equal(pose[np.arange(3) != slot], original[np.arange(3) != slot])
        np.testing.assert_allclose(distances(pose[slot]), distances(original[slot]), atol=1e-13)
    np.testing.assert_array_equal(original, saved)


def test_boundary_trials_are_recorded_without_clipping_or_hidden_replacement():
    target = component()
    target['centroid'] = np.array([0., 0., -6.])
    result = volume_scene_starts(scene(), [target])
    assert result['admissible'][0]
    assert result['outside_coordinates'][1:].any()
    assert not result['admissible'][1:].all()
    assert result['pose_xyz'][1:].min() < -6
    assert len(result['pose_xyz']) == 25


def test_world_rigid_transform_commutes_for_nondegenerate_templates():
    body, target = scene()[0], component()
    rotation = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    shift = np.array([4., -2., 7.])
    original = oriented_bodies(body, **target)[0]
    moved = oriented_bodies(body @ rotation.T + shift,
                            target['centroid'] @ rotation.T + shift,
                            target['axes'] @ rotation.T)[0]
    np.testing.assert_allclose(moved, original @ rotation.T + shift, atol=6e-14)


def test_straight_body_has_finite_right_handed_frame():
    body = np.array([[6., 0, 0], [0, 0, 0], [-16., 0, 0]])
    axes = body_frame(body)
    np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(axes) == pytest.approx(1.)


def test_empty_components_preserve_only_original_scene():
    result = volume_scene_starts(scene(), [])
    np.testing.assert_array_equal(result['pose_xyz'], scene()[None])
    np.testing.assert_array_equal(result['admissible'], [True])
    random = random_scene_starts(scene(), [], seed=0)
    np.testing.assert_array_equal(random['pose_xyz'], scene()[None])


def test_matched_random_control_is_deterministic_and_preserves_trial_slot_and_anatomy():
    slots = np.repeat([0, 1, 2], 8)
    first = random_scene_starts(scene(), slots, seed=913)
    second = random_scene_starts(scene(), slots, seed=913)
    for key in ('pose_xyz', 'raw_slot', 'outside_coordinates', 'admissible'):
        np.testing.assert_array_equal(first[key], second[key])
    assert len(first['pose_xyz']) == 25
    for pose, slot in zip(first['pose_xyz'][1:], slots, strict=True):
        np.testing.assert_array_equal(pose[np.arange(3) != slot], scene()[np.arange(3) != slot])
        np.testing.assert_allclose(distances(pose[slot]), distances(scene()[slot]), atol=1e-13)
    changed_seed = random_scene_starts(scene(), slots, seed=914)
    assert not np.array_equal(first['pose_xyz'], changed_seed['pose_xyz'])


@pytest.mark.parametrize('bad', [np.zeros((3, 3)), np.ones((3, 2)), np.full((3, 3), np.nan),
                                [[0, 0, 0], [1, 0, 0], [0, 0, 0]]])
def test_degenerate_templates_are_rejected(bad):
    with pytest.raises(ValueError):
        oriented_bodies(bad, **component())


@pytest.mark.parametrize('axes', [np.ones((3, 3)), np.eye(3) * 2, np.full((3, 3), np.nan)])
def test_invalid_pca_axes_rejected(axes):
    with pytest.raises(ValueError):
        oriented_bodies(scene()[0], np.zeros(3), axes)


def test_finite_pool_and_fixed_roll_budget_are_explicit():
    with pytest.raises(ValueError):
        volume_scene_starts(scene(), [component()] * 13)
    with pytest.raises(ValueError):
        RigidStartConfig(rolls=5)
    with pytest.raises(ValueError):
        random_scene_starts(scene(), [3], seed=0)

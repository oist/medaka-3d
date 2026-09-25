"""Synthetic gate ablation and its deliberately unresolved reflection counterexample."""

import dataclasses
import hashlib
from pathlib import Path

import instance_body_score as body
import mask_only_instance_observation as new
import numpy as np
import pytest
import sparse_instance_body_score as old


def scene():
    masks = np.zeros((3, 100, 180), dtype=bool)
    points = []
    for k, x in enumerate((10, 70, 130)):
        masks[k, 30:46, x : x + 30] = True
        points.append([[x + 4.0, 35.0], [x + 15.0, 37.0], [x + 25.0, 40.0]])
    points = np.array(points)
    poses = np.concatenate([points, np.ones((3, 3, 1))], axis=-1)[None]
    return masks, points, poses


def project(xyz):
    return xyz[..., :2]


def assert_array_exact(a, b):
    assert a.shape == b.shape and a.dtype.str == b.dtype.str
    assert a.tobytes() == b.tobytes()


def test_frozen_sparse_source_and_default_scoring_config_are_reused():
    assert (
        hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest() == new.REFERENCE_SPARSE_SHA256
    )
    masks, _, _ = scene()
    observation = new.prepare_mask_only_instance_observation(masks)
    assert observation.config is body.DEFAULT_CONFIG
    assert isinstance(observation.sparse_fields, old.SparseDistanceFields)
    assert isinstance(observation, old.SparseInstanceObservation)


@pytest.mark.parametrize("zero_weights", [False, True])
def test_formerly_eligible_every_numerical_score_array_is_byte_exact(zero_weights):
    masks, points, pose = scene()
    config = body.DEFAULT_CONFIG
    if zero_weights:
        config = dataclasses.replace(config, worst_node_weight=0.0, segment_leakage_weight=0.0)
    before = old.prepare_sparse_instance_observation(masks, points, config)
    after = new.prepare_mask_only_instance_observation(masks, config)
    bank = np.repeat(pose, 5, axis=0)
    bank[1, 2] = bank[1, 0]  # Whole body duplicates another silhouette.
    bank[2, 0, 0] = [-8.25, 80.75, 2.0]  # Noninteger and outside samples.
    bank[3, 0, 0, 2] = np.nan
    bank[4, 1, 2, 0] = np.inf
    original_bytes = masks.tobytes(), points.tobytes(), bank.tobytes()
    expected = old.score_sparse_instance_bodies(bank, project, before)
    actual = old.score_sparse_instance_bodies(bank, project, after)
    assert before.eligible and after.eligible
    assert set(actual) == set(expected)
    arrays = [key for key in expected if isinstance(expected[key], np.ndarray)]
    assert len(arrays) == 10
    for key in arrays:
        assert_array_exact(actual[key], expected[key])
    assert actual["permanent_IDs_assigned"] is False
    assert actual["observed_flags_used"] is False
    assert actual["instance_assignments_are_frame_local"] is True
    assert actual["candidate_projection_finite"].tolist() == [True, True, True, False, False]
    assert np.all(actual["best_assignment"][3:] == -1)
    assert np.isinf(actual["best_cost"][3:]).all()
    assert original_bytes == (masks.tobytes(), points.tobytes(), bank.tobytes())
    for key in (
        "mask_area_pixels",
        "pair_intersection_pixels_01_02_12",
        "pair_mask_distance_px_01_02_12",
    ):
        assert before.evidence[key] == after.evidence[key]


@pytest.mark.parametrize("sl_fault", ["absent", "missing_tail", "duplicate", "wrong_group"])
def test_only_sleap_group_failures_can_become_eligible_without_inventing_correspondence(sl_fault):
    masks, points, _ = scene()
    if sl_fault == "absent":
        points = None
    elif sl_fault == "missing_tail":
        points[2, 2] = np.nan
    elif sl_fault == "duplicate":
        points = np.concatenate([points, points[:1]])
    else:
        points[2] = points[0]
    assert not old.prepare_sparse_instance_observation(masks, points).eligible
    after = new.prepare_mask_only_instance_observation(masks)
    assert after.eligible and after.sleap_row_by_instance is None
    assert after.evidence["SLEAP_groups_used_for_eligibility"] is False
    assert after.evidence["SLEAP_instance_correspondence_available"] is False
    assert after.evidence["candidate_pose_used_for_eligibility"] is False
    assert after.evidence["detection_confidence_used"] is False
    for key in ("complete_SL_rows", "qualifying_SL_rows", "unmatched_complete_SL_rows"):
        assert after.evidence[key] is None


@pytest.mark.parametrize("count", [0, 2, 4, 5])
def test_other_instance_counts_are_unavailable_and_never_project(count):
    masks, _, pose = scene()
    masks = np.concatenate([masks, masks[:2]])[:count]
    observation = new.prepare_mask_only_instance_observation(masks)
    assert observation.reason == "exactly_three_mask_instances_required"

    def forbidden(_):
        pytest.fail("neutral observations must not project candidate poses")

    score = old.score_sparse_instance_bodies(pose, forbidden, observation)
    assert score["neutral"] and not score["eligible"]
    assert_array_exact(score["best_cost"], np.zeros(1))
    assert score["assignment_costs"] is None and observation.sparse_fields is None


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("none", "mask_view_unavailable"),
        ("empty", "empty_mask_instance"),
        ("overlap", "overlapping_mask_instances"),
    ],
)
def test_remaining_mask_gates_retain_original_reasons_and_evidence(fault, reason):
    masks, points, _ = scene()
    if fault == "none":
        masks = None
    elif fault == "empty":
        masks[2] = False
    else:
        masks[0] |= masks[1]
    before = old.prepare_sparse_instance_observation(masks, points)
    after = new.prepare_mask_only_instance_observation(masks)
    assert not after.eligible and after.reason == before.reason == reason
    assert after.image_shape_hw == before.image_shape_hw
    assert after.sparse_fields is None and after.sleap_row_by_instance is None
    for key, value in before.evidence.items():
        assert after.evidence[key] == value


@pytest.mark.parametrize("gap", [1, 2, 3, 4])
def test_three_pixel_separation_boundary_is_unchanged(gap):
    masks, points, _ = scene()
    masks[0, 30:46, 39 : 70 - gap + 1] = True
    before = old.prepare_sparse_instance_observation(masks, points)
    after = new.prepare_mask_only_instance_observation(masks)
    assert after.evidence["pair_mask_distance_px_01_02_12"][0] == gap
    assert after.eligible == before.eligible == (gap > 3)


def test_clone_is_penalized_by_whole_body_bijection_and_latent_mouth_is_included():
    masks, _, pose = scene()
    bank = np.repeat(pose, 3, axis=0)
    bank[1, 2] = bank[1, 0]
    bank[2, 2, 0] = bank[2, 0, 0]  # Mouth remains scored even without an observed flag.
    score = old.score_sparse_instance_bodies(
        bank, project, new.prepare_mask_only_instance_observation(masks)
    )
    assert score["best_cost"][0] == 0
    assert np.all(score["best_cost"][1:] > 0)
    assert score["assignment_costs"].shape == (3, 6)
    assert np.all(score["assignment_costs"][1:] > 0)
    assert score["observed_flags_used"] is False


@pytest.mark.parametrize("representation", ["overlapping_three", "merged_two"])
def test_genuine_projected_occlusion_is_neutral_when_observation_reports_it(representation):
    masks, _, pose = scene()
    pose[0, 1] = pose[0, 0]
    pose[0, 1, :, 2] += 20  # Distinct synthetic 3D fish with identical orthographic projection.
    masks[1] = masks[0]
    if representation == "merged_two":
        masks = masks[[0, 2]]
    observation = new.prepare_mask_only_instance_observation(masks)
    assert not observation.eligible

    def forbidden(_):
        pytest.fail("overlap must not create an artificial separation penalty")

    score = old.score_sparse_instance_bodies(pose, forbidden, observation)
    assert score["neutral"] and score["best_cost"][0] == 0
    assert score["assignment_costs"] is None


def test_reflection_counterexample_can_prefer_a_false_body_and_is_not_resolved():
    # Synthetic semantics: masks0/1 are real silhouettes; mask2 is a reflection.
    # Real fish0/2 overlap at different depth inside mask0. A false candidate
    # moves fish2 onto reflection2. The observation alone cannot distinguish it.
    masks, _, pose = scene()
    real = pose.copy()
    real[0, 2] = real[0, 0]
    real[0, 2, :, 2] += 20
    bank = np.concatenate([real, pose])
    observation = new.prepare_mask_only_instance_observation(masks)
    score = old.score_sparse_instance_bodies(bank, project, observation)
    assert observation.eligible
    assert score["best_cost"][1] == 0 < score["best_cost"][0]
    assert observation.evidence["mask_instances_are_verified_fish"] is False
    assert observation.evidence["occlusion_or_reflection_resolved"] is False
    assert score["permanent_IDs_assigned"] is False


def test_projection_still_receives_three_dimensional_interior_samples():
    masks, _, pose = scene()
    pose[0, 0, :, 2] = [1.0, 3.0, 5.0]
    seen = []

    def nonlinear_project(xyz):
        seen.append(xyz.copy())
        return np.stack([xyz[..., 0] + xyz[..., 2] ** 2, xyz[..., 1]], axis=-1)

    score = old.score_sparse_instance_bodies(
        pose, nonlinear_project, new.prepare_mask_only_instance_observation(masks)
    )
    assert len(seen) == 1
    assert_array_exact(seen[0], body.sample_body_xyz(pose, body.DEFAULT_CONFIG.interior_fractions))
    projected = score["projected_samples"]
    assert projected[0, 0, 4, 0] != (projected[0, 0, 0, 0] + projected[0, 0, 1, 0]) / 2


@pytest.mark.parametrize("fault", ["list", "uint8", "two_dimensions", "empty_image", "config"])
def test_malformed_inputs_are_errors_not_neutral_observations(fault):
    masks, _, _ = scene()
    cfg = body.DEFAULT_CONFIG
    if fault == "list":
        masks = masks.tolist()
    elif fault == "uint8":
        masks = masks.astype(np.uint8)
    elif fault == "two_dimensions":
        masks = masks[0]
    elif fault == "empty_image":
        masks = masks[:, :0]
    else:
        cfg = {}
    with pytest.raises(ValueError):
        new.prepare_mask_only_instance_observation(masks, cfg)

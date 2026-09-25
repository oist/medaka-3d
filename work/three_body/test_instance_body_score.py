"""Synthetic geometry/observation tests, not medaka results or physical ID labels."""

import dataclasses

import instance_body_score as m
import numpy as np
import pytest


def scene():
    masks = np.zeros((3, 100, 180), bool)
    for k, x in enumerate((10, 70, 130)):
        masks[k, 30:46, x : x + 30] = True
    points = np.array(
        [[[x + 4.0, 35.0], [x + 15.0, 37.0], [x + 25.0, 40.0]] for x in (10, 70, 130)]
    )
    pose = np.concatenate([points, np.ones((3, 3, 1))], axis=-1)[None]
    return masks, points, pose


def project(xyz):
    return xyz[..., :2]


def signature(value):
    return value.dtype.str, value.shape, value.tobytes()


def test_six_whole_body_assignments_and_exact_observation_grouping():
    masks, points, pose = scene()
    before = [signature(x) for x in (masks, points, pose)]
    observation = m.prepare_instance_observation(masks, points)
    assert observation.eligible and observation.sleap_row_by_instance.tolist() == [0, 1, 2]
    assert observation.evidence["candidate_pose_used_for_eligibility"] is False
    assert observation.evidence["detection_confidence_used"] is False
    result = m.score_instance_bodies(pose, project, observation)
    assert result["body_instance_costs"].shape == (1, 3, 3)
    assert result["assignment_costs"].shape == (1, 6)
    assert result["best_cost"].tolist() == [0.0]
    assert result["best_assignment"].tolist() == [[0, 1, 2]]
    assert result["optimal_assignment_mask"].sum() == 1
    assert result["permanent_IDs_assigned"] is False
    assert result["instance_assignments_are_frame_local"] is True
    assert before == [signature(x) for x in (masks, points, pose)]


def test_split_body_is_penalized_even_when_all_endpoints_have_zero_union_distance():
    masks, points, pose = scene()
    bad = pose.copy()
    bad[0, 2, :2] = pose[0, 0, :2]
    bank = np.concatenate([pose, bad])
    union = m.distance_transform_edt(~masks.any(axis=0))[None]
    assert np.all(m.sample_distance_fields(union, project(bank)) == 0.0)
    observation = m.prepare_instance_observation(masks, points)
    result = m.score_instance_bodies(bank, project, observation)
    assert result["best_cost"][0] == 0.0
    assert result["best_cost"][1] > 0.0
    assert (result["body_instance_costs"][1, 2] > 0).all()
    assert len(set(result["best_assignment"][1])) == 3
    assert result["observed_flags_used"] is False  # No flag can hide the displaced latent endpoint.


def test_duplicate_bodies_cannot_each_select_the_same_instance_for_free():
    masks, points, pose = scene()
    pose[0, 2] = pose[0, 0]
    result = m.score_instance_bodies(pose, project, m.prepare_instance_observation(masks, points))
    assert np.min(result["body_instance_costs"], axis=-1).sum() == 0.0
    assert result["best_cost"][0] > 0.0
    assert all(len(set(p)) == 3 for p in m.PERMUTATIONS)


def test_interiors_are_interpolated_in_3d_before_perspective_projection():
    masks, points, pose = scene()
    pose[0, 0] = [[15.0, 35.0, 1.0], [50.0, 70.0, 2.0], [120.0, 140.0, 4.0]]
    calls = []

    def perspective(samples):
        calls.append(samples.copy())
        return samples[..., :2] / samples[..., 2:]

    result = m.score_instance_bodies(
        pose, perspective, m.prepare_instance_observation(masks, points)
    )
    assert len(calls) == 1 and calls[0].shape == (1, 3, 9, 3)
    np.testing.assert_array_equal(calls[0][:, :, :3], pose)
    np.testing.assert_array_equal(calls[0][0, 0, 4], 0.5 * pose[0, 0, 0] + 0.5 * pose[0, 0, 1])
    actual = result["projected_samples"][0, 0, 4]
    assert np.allclose(actual, [32.5 / 1.5, 35.0])
    endpoints_uv = pose[0, 0, :2, :2] / pose[0, 0, :2, 2:]
    assert not np.allclose(actual, endpoints_uv.mean(axis=0))


def test_rigid_world_transform_with_corresponding_camera_preserves_scores():
    masks, points, pose = scene()
    observation = m.prepare_instance_observation(masks, points)
    original = m.score_instance_bodies(pose, project, observation)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([5.0, -17.0, 23.0])
    changed = pose @ rotation.T + translation
    transformed = m.score_instance_bodies(
        changed, lambda x: ((x - translation) @ rotation)[..., :2], observation
    )
    for key in (
        "projected_samples",
        "body_instance_costs",
        "assignment_costs",
        "best_cost",
        "best_assignment",
    ):
        np.testing.assert_array_equal(original[key], transformed[key])


def test_mask_instance_permutation_changes_only_frame_local_assignment_indices():
    masks, points, pose = scene()
    bad = pose.copy()
    bad[0, 2, :2] = pose[0, 0, :2]
    bank = np.concatenate([pose, bad])
    original = m.score_instance_bodies(bank, project, m.prepare_instance_observation(masks, points))
    order = np.array([2, 0, 1])
    changed_observation = m.prepare_instance_observation(masks[order], points)
    changed = m.score_instance_bodies(bank, project, changed_observation)
    assert changed_observation.sleap_row_by_instance.tolist() == [2, 0, 1]
    np.testing.assert_array_equal(changed["best_cost"], original["best_cost"])
    np.testing.assert_array_equal(
        np.sort(changed["assignment_costs"], axis=1), np.sort(original["assignment_costs"], axis=1)
    )
    np.testing.assert_array_equal(order[changed["best_assignment"]], original["best_assignment"])


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("missing_view", "mask_view_unavailable"),
        ("two_instances", "exactly_three_mask_instances_required"),
        ("empty", "empty_mask_instance"),
        ("overlap", "overlapping_mask_instances"),
        ("touching", "mask_instances_not_clearly_separated"),
        ("missing_SL", "SLEAP_view_unavailable"),
        ("incomplete_SL", "missing_complete_SLEAP_group_for_instance"),
        ("merged_SL", "missing_complete_SLEAP_group_for_instance"),
        ("duplicate_SL", "duplicate_SLEAP_groups_for_instance"),
    ],
)
def test_ambiguous_observations_are_neutral_without_projecting_or_forcing_three(fault, reason):
    masks, points, pose = scene()
    if fault == "missing_view":
        masks = None
    elif fault == "two_instances":
        masks = masks[:2]
    elif fault == "empty":
        masks[2] = False
    elif fault == "overlap":
        masks[0] |= masks[1]
    elif fault == "touching":
        masks[0, 30:46, 39:70] = True
    elif fault == "missing_SL":
        points = None
    elif fault == "incomplete_SL":
        points[2, 2] = np.nan
    elif fault == "merged_SL":
        points[2] = points[0]
    else:
        points = np.concatenate([points, points[:1]])
    observation = m.prepare_instance_observation(masks, points)
    assert not observation.eligible and observation.reason == reason

    def forbidden(_):
        pytest.fail("unavailable observation called candidate projector")

    result = m.score_instance_bodies(np.repeat(pose, 4, axis=0), forbidden, observation)
    np.testing.assert_array_equal(result["best_cost"], np.zeros(4))
    assert result["neutral"] and result["best_assignment"] is None
    assert result["body_instance_costs"] is None and result["assignment_costs"] is None


def test_extra_unmatched_SL_row_is_recorded_without_inventing_confidence():
    masks, points, _ = scene()
    points = np.concatenate([points, np.full((1, 3, 2), [0.0, 90.0])])
    observation = m.prepare_instance_observation(masks, points)
    assert observation.eligible
    assert observation.evidence["unmatched_complete_SL_rows"] == [3]
    assert observation.evidence["detection_confidence_used"] is False


def test_SL_row_compatible_with_multiple_instances_makes_observation_neutral():
    masks, points, _ = scene()
    wide = dataclasses.replace(m.InstanceBodyConfig(), sleap_mask_tolerance_px=200.0)
    observation = m.prepare_instance_observation(masks, points, wide)
    assert not observation.eligible and observation.reason == "SLEAP_row_matches_multiple_instances"


def test_segment_leakage_detects_unsupported_straight_interiors_with_endpoints_inside():
    masks, points, pose = scene()
    masks[0] = False
    masks[0, 20:46, 10:16] = True
    masks[0, 20:46, 34:40] = True
    masks[0, 20:26, 10:40] = True
    points[0] = [[12.0, 42.0], [37.0, 42.0], [37.0, 23.0]]
    pose[0, 0, :, :2] = points[0]
    observation = m.prepare_instance_observation(masks, points)
    assert observation.eligible
    result = m.score_instance_bodies(pose, project, observation)
    assert result["worst_endpoint_costs"][0, 0, 0] == 0.0
    assert result["segment_leakage_costs"][0, 0, 0] > 0.0


def test_endpoint_tolerance_and_pixel_scale_are_explicit_not_observation_gates():
    masks, points, pose = scene()
    pose[0, 0, 0, 0] = 6.0  # Four pixels outside the first silhouette.
    default = m.prepare_instance_observation(masks, points)
    strict = m.prepare_instance_observation(
        masks, points, dataclasses.replace(default.config, endpoint_tolerance_px=0.0)
    )
    assert default.eligible == strict.eligible and default.evidence == strict.evidence
    a = m.score_instance_bodies(pose, project, default)
    b = m.score_instance_bodies(pose, project, strict)
    assert a["worst_endpoint_costs"][0, 0, 0] == 0.0
    assert b["worst_endpoint_costs"][0, 0, 0] == (4.0 / 6.0) ** 2


@pytest.mark.parametrize("fault", ["nonfinite_pose", "nonfinite_projection", "outside_image"])
def test_bad_candidate_never_disables_an_eligible_observation(fault):
    masks, points, pose = scene()
    bank = np.repeat(pose, 2, axis=0)
    observation = m.prepare_instance_observation(masks, points)

    def selected(x):
        uv = project(x).copy()
        if fault == "nonfinite_projection":
            uv[1, 0, 0] = np.nan
        return uv

    if fault == "nonfinite_pose":
        bank[1, 0, 0, 2] = np.nan
    elif fault == "outside_image":
        bank[1, 0, :, 0] = -40.0
    result = m.score_instance_bodies(bank, selected, observation)
    assert result["eligible"] and not result["neutral"]
    assert result["best_cost"][0] == 0.0
    if fault == "outside_image":
        assert result["best_cost"][1] > 0.0
    else:
        assert np.isposinf(result["best_cost"][1])
        assert result["best_assignment"][1].tolist() == [-1, -1, -1]
        assert not result["optimal_assignment_mask"][1].any()


def test_zero_weight_cannot_turn_invalid_latent_projection_into_nan_or_neutral():
    masks, points, pose = scene()
    cfg = dataclasses.replace(m.DEFAULT_CONFIG, worst_node_weight=0.0, segment_leakage_weight=0.0)
    observation = m.prepare_instance_observation(masks, points, cfg)

    def invalid(samples):
        result = project(samples).copy()
        result[0, 0, 0] = np.nan
        return result

    with np.errstate(invalid="raise"):
        result = m.score_instance_bodies(pose, invalid, observation)
    assert result["eligible"] and not result["neutral"]
    assert np.isposinf(result["best_cost"][0])


@pytest.mark.parametrize(
    "change",
    [
        dict(endpoint_tolerance_px=-1.0),
        dict(pixel_scale_px=0.0),
        dict(sleap_mask_tolerance_px=np.nan),
        dict(interior_fractions=(0.0, 0.5)),
        dict(interior_fractions=(0.5, 0.5)),
    ],
)
def test_invalid_explicit_scoring_settings_rejected(change):
    with pytest.raises(ValueError):
        m.InstanceBodyConfig(**change)

"""Independent synthetic endpoint classification and invalid-evidence contracts."""

import copy

import endpoint_mask_conflict_score as m
import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt


@pytest.fixture(autouse=True)
def no_array_archive_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("endpoint tests may not read saved arrays")

    monkeypatch.setattr(np, "load", forbidden)


def project(xyz):
    return xyz[..., :2]


def forbidden_project(*args, **kwargs):
    pytest.fail("unmeasured bodies must not be projected")


def scene(body=None):
    body = np.array([[3, 6, 1], [7, 6, 1], [11, 6, 1]], float) if body is None else body
    return np.broadcast_to(body, (1, 3, 3, 3)).copy()


def masks_and_cases():
    masks = np.zeros((5, 70, 80), bool)
    masks[1, 4:11, 2:13] = True
    masks[2, 4:11, 32:43] = True
    masks[3:, 4:11, 52:63] = True
    bodies = [
        [[3, 6, 1], [7, 6, 1], [11, 6, 1]],
        [[3, 6, 1], [7, 6, 1], [34, 6, 1]],
        [[3, 6, 1], [7, 6, 1], [78, 65, 1]],
        [[53, 6, 1], [55, 6, 1], [57, 6, 1]],
        [[53, 6, 1], [55, 6, 1], [78, 65, 1]],
        [[33, 6, 1], [37, 6, 1], [41, 6, 1]],
    ]
    return masks, np.asarray(bodies, float).reshape(2, 3, 3, 3)


def dense_bilinear_oracle(masks, xy):
    """Full integer EDT; hand bilinear corners with explicit border extension."""
    h, w = masks.shape[1:]
    edt = [distance_transform_edt(~mask) for mask in masks]
    result = np.empty((*xy.shape[:-1], len(masks)), float)
    for index in np.ndindex(xy.shape[:-1]):
        x, y = xy[index]
        cx, cy = min(max(x, 0), w - 1), min(max(y, 0), h - 1)
        x0, y0 = int(np.floor(cx)), int(np.floor(cy))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        wx, wy = cx - x0, cy - y0
        extension = np.hypot(x - cx, y - cy)
        for mask_index, image in enumerate(edt):
            result[(*index, mask_index)] = (
                image[y0, x0] * (1 - wx) * (1 - wy)
                + image[y0, x1] * wx * (1 - wy)
                + image[y1, x0] * (1 - wx) * wy
                + image[y1, x1] * wx * wy
                + extension
            )
    return result


def oracle_label(distances, tolerance):
    choices = [np.flatnonzero(row <= tolerance).tolist() for row in distances]
    if any(not ids for ids in choices):
        return "uncovered_endpoint"
    if any(len(ids) > 1 for ids in choices):
        return "ambiguous_endpoint"
    return "same_instance" if len({ids[0] for ids in choices}) == 1 else "split_instances"


def test_normal_classes_original_indices_and_uncovered_precedence():
    masks, poses = masks_and_cases()
    observation = m.prepare_endpoint_mask_observation(masks)
    result = m.score_endpoint_mask_conflicts(poses, project, observation)
    assert result["classification_by_body"].tolist() == [
        ["same_instance", "split_instances", "uncovered_endpoint"],
        ["ambiguous_endpoint", "uncovered_endpoint", "same_instance"],
    ]
    assert result["split_by_body"].tolist() == [[False, True, False], [False, False, False]]
    assert result["unknown_by_body"].tolist() == [[False, False, True], [True, True, False]]
    assert result["mask_indices"].tolist() == [1, 2, 3, 4]
    assert result["unique_mask_by_node"][0, 1].tolist() == [1, 1, 2]
    assert result["unique_mask_by_node"][1, 0].tolist() == [-1, -1, -1]
    assert result["ambiguous_by_node"][1, 1].tolist() == [True, True, False]
    assert result["uncovered_by_node"][1, 1].tolist() == [False, False, True]
    expected = dense_bilinear_oracle(masks[1:], poses[..., :2])
    np.testing.assert_allclose(
        result["distances_by_candidate_body_node_mask"], expected, atol=1e-14
    )
    for index in np.ndindex(poses.shape[:2]):
        assert result["classification_by_body"][index] == oracle_label(expected[index], 6.0)


def test_subpixel_outside_distances_and_fixed_tolerance_equal_boundary():
    masks = np.zeros((2, 11, 18), bool)
    masks[0, 2:9, 3] = True
    masks[1, 2:9, 16] = True
    bodies = np.array(
        [
            [[9.0, 4, 1], [9.0, 5, 1], [9.0, 6, 1]],
            [[9.0001, 4, 1], [9.0001, 5, 1], [9.0001, 6, 1]],
            [[-1.125, -2.25, 1], [8.375, 5.625, 1], [18.25, 11.125, 1]],
        ]
    )[None]
    result = m.score_endpoint_mask_conflicts(
        bodies, project, m.prepare_endpoint_mask_observation(masks)
    )
    expected = dense_bilinear_oracle(masks, bodies[..., :2])
    np.testing.assert_allclose(
        result["distances_by_candidate_body_node_mask"], expected, rtol=0, atol=1e-14
    )
    assert result["classification_by_body"][0, :2].tolist() == [
        "same_instance",
        "uncovered_endpoint",
    ]
    assert result["unique_mask_by_node"][0, 0].tolist() == [0, 0, 0]
    assert result["classification_by_body"][0, 2] == oracle_label(expected[0, 2], 6.0)


@pytest.mark.parametrize("mask_kind", ["none", "empty_count", "all_zero"])
def test_missing_masks_are_unknown_skip_projection_and_keep_invalid_geometry(mask_kind):
    masks = {
        "none": None,
        "empty_count": np.zeros((0, 7, 9), bool),
        "all_zero": np.zeros((3, 7, 9), bool),
    }[mask_kind]
    poses = scene()
    poses[0, 1, 2, 0] = np.nan
    observation = m.prepare_endpoint_mask_observation(masks)
    result = m.score_endpoint_mask_conflicts(poses, forbidden_project, observation)
    assert result["classification_by_body"].tolist() == [
        ["unavailable", "invalid_geometry", "unavailable"]
    ]
    assert not result["available"] and result["unknown_by_body"].all()
    assert not result["split_by_body"].any()
    assert (result["nearby_mask_count_by_node"] == -1).all()
    assert (result["unique_mask_by_node"] == -1).all()
    assert not result["uncovered_by_node"].any() and not result["ambiguous_by_node"].any()
    for key in (
        "projected_xy",
        "projection_finite_by_body",
        "distances_by_candidate_body_node_mask",
    ):
        assert result[key] is None
    assert observation.evidence["reason"] == (
        "mask_view_unavailable" if masks is None else "no_nonempty_masks"
    )


def test_geometry_and_projection_invalid_bodies_do_not_become_split_or_measured(monkeypatch):
    masks, poses = masks_and_cases()
    poses[0, 0, 0, 0] = np.nan
    poses[1, 2, 0, 0] = np.inf
    observation = m.prepare_endpoint_mask_observation(masks)
    calls = []
    original_sample = m.SparseDistanceFields.sample

    def checked_sample(self, xy):
        assert np.isfinite(xy).all()
        calls.append(xy.copy())
        return original_sample(self, xy)

    monkeypatch.setattr(m.SparseDistanceFields, "sample", checked_sample)

    def invalid_project(xyz):
        assert xyz.shape == (4, 3, 3) and np.isfinite(xyz).all()
        out = xyz[..., :2].copy()
        out[0, 0, 0] = np.nan
        out[1, 1, 1] = np.inf
        return out

    result = m.score_endpoint_mask_conflicts(poses, invalid_project, observation)
    assert result["classification_by_body"].tolist() == [
        ["invalid_geometry", "invalid_projection", "invalid_projection"],
        ["ambiguous_endpoint", "uncovered_endpoint", "invalid_geometry"],
    ]
    invalid = np.array([[True, True, True], [False, False, True]])
    assert result["unknown_by_body"][invalid].all() and not result["split_by_body"].any()
    assert (result["nearby_mask_count_by_node"][invalid] == -1).all()
    assert np.isinf(result["distances_by_candidate_body_node_mask"][invalid]).all()
    assert len(calls) == 1 and calls[0].shape == (2, 3, 2)


def test_all_invalid_geometry_skips_project_even_with_masks():
    poses = np.full((2, 3, 3, 3), np.nan)
    obs = m.prepare_endpoint_mask_observation(np.ones((1, 3, 3), bool))
    result = m.score_endpoint_mask_conflicts(poses, forbidden_project, obs)
    assert (result["classification_by_body"] == "invalid_geometry").all()
    assert np.isinf(result["distances_by_candidate_body_node_mask"]).all()


@pytest.mark.parametrize("available", [False, True])
def test_zero_candidates_preserve_empty_axes_and_skip_project(available):
    obs = m.prepare_endpoint_mask_observation(np.ones((2, 3, 3), bool) if available else None)
    result = m.score_endpoint_mask_conflicts(np.empty((0, 3, 3, 3)), forbidden_project, obs)
    for key in (
        "classification_by_body",
        "split_by_body",
        "unknown_by_body",
        "geometry_finite_by_body",
    ):
        assert result[key].shape == (0, 3)
    assert result["nearby_mask_count_by_node"].shape == (0, 3, 3)
    if available:
        assert result["distances_by_candidate_body_node_mask"].shape == (0, 3, 3, 2)
        assert result["projected_xy"].shape == (0, 3, 3, 2)
    else:
        assert result["distances_by_candidate_body_node_mask"] is None


@pytest.mark.parametrize(
    "tolerance", [True, np.bool_(False), np.nan, np.inf, -np.inf, -1, "6", None]
)
def test_invalid_tolerance_rejected_even_without_evidence(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        m.score_endpoint_mask_conflicts(
            scene(), forbidden_project, m.prepare_endpoint_mask_observation(None), tolerance
        )


@pytest.mark.parametrize("tolerance", [0, 6, np.float32(6), np.float64(6)])
def test_numeric_zero_and_numpy_real_tolerances_are_supported(tolerance):
    obs = m.prepare_endpoint_mask_observation(np.ones((1, 12, 20), bool))
    result = m.score_endpoint_mask_conflicts(scene(), project, obs, tolerance)
    assert (result["classification_by_body"] == "same_instance").all()


@pytest.mark.parametrize(
    "bad", [[], np.ones((1, 3, 3), np.uint8), np.ones((3, 3), bool), np.ones((1, 0, 3), bool)]
)
def test_invalid_mask_inputs_rejected(bad):
    with pytest.raises(ValueError, match="boolean masks"):
        m.prepare_endpoint_mask_observation(bad)


def test_shape_and_observation_contract_fail_before_projection():
    obs = m.prepare_endpoint_mask_observation(None)
    for bad in (np.zeros((3, 3, 3)), np.zeros((1, 2, 3, 3)), np.zeros((1, 3, 2, 3))):
        with pytest.raises(ValueError, match="candidate body poses"):
            m.score_endpoint_mask_conflicts(bad, forbidden_project, obs)
    with pytest.raises(ValueError, match="EndpointMaskObservation"):
        m.score_endpoint_mask_conflicts(scene(), forbidden_project, {})
    obs = m.prepare_endpoint_mask_observation(np.ones((1, 12, 20), bool))
    with pytest.raises(ValueError, match="preserve body/MCT"):
        m.score_endpoint_mask_conflicts(scene(), lambda xyz: np.zeros((3, 2)), obs)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0, "shape"])
def test_distance_backend_bad_output_rejected_not_reclassified(monkeypatch, bad):
    obs = m.prepare_endpoint_mask_observation(np.ones((1, 12, 20), bool))
    monkeypatch.setattr(
        m.SparseDistanceFields,
        "sample",
        lambda self, xy: np.zeros((1, 3, 1)) if bad == "shape" else np.full((3, 3, 1), bad),
    )
    with pytest.raises(ValueError, match="finite nonnegative mask distances"):
        m.score_endpoint_mask_conflicts(scene(), project, obs)


def test_masks_pose_and_observation_evidence_are_owned_and_not_mutated():
    masks, poses = masks_and_cases()
    original_masks, original_poses = masks.copy(), poses.copy()
    obs = m.prepare_endpoint_mask_observation(masks)
    evidence = copy.deepcopy(obs.evidence)
    expected = m.score_endpoint_mask_conflicts(poses, project, obs)
    np.testing.assert_array_equal(masks, original_masks)
    np.testing.assert_array_equal(poses, original_poses)
    masks[:] = False
    actual = m.score_endpoint_mask_conflicts(poses, project, obs)
    np.testing.assert_array_equal(
        actual["distances_by_candidate_body_node_mask"],
        expected["distances_by_candidate_body_node_mask"],
    )
    actual["mask_indices"][:] = 999
    actual["observation_evidence"]["reason"] = "changed"
    assert obs.evidence == evidence

    def mutating_project(xyz):
        xy = xyz[..., :2].copy()
        xyz[:] = 12345
        return xy

    m.score_endpoint_mask_conflicts(poses, mutating_project, obs)
    np.testing.assert_array_equal(poses, original_poses)
    assert not obs.evidence["SLEAP_compatibility_used"]
    assert not obs.evidence["physical_IDs_assigned"]


def test_reflection_or_two_bodies_on_same_mask_is_not_certified_identity():
    # Two plausible detector regions can be reflections or the same animal;
    # this single-body diagnostic intentionally imposes no inter-body capacity.
    obs = m.prepare_endpoint_mask_observation(np.ones((1, 12, 20), bool))
    result = m.score_endpoint_mask_conflicts(scene(), project, obs)
    assert (result["classification_by_body"] == "same_instance").all()
    assert not result["split_by_body"].any()
    assert result["unique_mask_by_node"].tolist() == [[[0, 0, 0]] * 3]
    assert not result["observation_evidence"]["physical_IDs_assigned"]

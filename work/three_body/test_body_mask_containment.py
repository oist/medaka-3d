"""Small synthetic image-distance contracts; no stored data or tracking calls."""

import body_mask_containment as m
import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt


@pytest.fixture(autouse=True)
def no_array_archives(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic containment tests may not read array archives")

    monkeypatch.setattr(np, "load", forbidden)


def pose():
    return np.array([[[2.0, 4.0, 1.0], [6.0, 4.0, 1.0], [14.0, 4.0, 1.0]]])


def project(xyz):
    return xyz[..., :2]


def fields(masks):
    return m.SparseDistanceFields.from_masks(masks)


def dense_oracle(masks, xy):
    """Independent full EDT plus explicit bilinear corners and border extension."""
    height, width = masks.shape[1:]
    clipped = np.clip(xy, [0, 0], [width - 1, height - 1])
    outside = np.linalg.norm(xy - clipped, axis=-1)
    x0, y0 = np.floor(clipped).astype(int).reshape(-1, 2).T
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = (clipped.reshape(-1, 2) - np.stack((x0, y0), axis=-1)).T
    distances = []
    for mask in masks:
        edt = distance_transform_edt(~mask)
        value = (
            edt[y0, x0] * (1 - wx) * (1 - wy)
            + edt[y0, x1] * wx * (1 - wy)
            + edt[y1, x0] * (1 - wx) * wy
            + edt[y1, x1] * wx * wy
        )
        distances.append(value.reshape(xy.shape[:-1]) + outside)
    return np.stack(distances, axis=-1)


@pytest.mark.parametrize("count", [1, 2, 4, 5])
def test_one_mask_contains_every_sample_regardless_of_other_instance_count(count):
    masks = np.zeros((count, 12, 20), bool)
    masks[0, 3:6, 1:17] = True
    masks[1:, 10, 19] = True
    result = m.measure_body_mask_containment(pose(), project, fields(masks))
    assert result["available"] and result["best_single_mask"].tolist() == [0]
    assert result["sample_labels"] == [
        "M",
        "C",
        "T",
        "MC_025",
        "MC_050",
        "MC_075",
        "CT_025",
        "CT_050",
        "CT_075",
    ]
    np.testing.assert_array_equal(result["single_mask_max_distance_px"], [0.0])
    np.testing.assert_array_equal(result["union_max_distance_px"], [0.0])
    np.testing.assert_array_equal(result["best_single_mask_distances_px"], np.zeros((1, 9)))
    assert not result["SLEAP_compatibility_used"] and not result["support_flags_used"]
    assert not result["physical_IDs_assigned"] and not result["rejection_threshold_applied"]


def test_split_body_has_zero_union_but_positive_whole_body_distance():
    masks = np.zeros((2, 12, 20), bool)
    masks[0, 3:6, 1:9] = True
    masks[1, 3:6, 9:16] = True
    result = m.measure_body_mask_containment(pose(), project, fields(masks))
    # M/C and MC interiors belong to mask0; T and two CT interiors to mask1.
    np.testing.assert_array_equal(result["union_distances_by_sample_px"], np.zeros((1, 9)))
    np.testing.assert_array_equal(result["single_mask_max_distance_px"], [6.0])
    np.testing.assert_array_equal(result["mask_switch_advantage_px"], [6.0])
    np.testing.assert_array_equal(result["maximum_distance_by_body_mask_px"], [[6.0, 7.0]])
    assert result["best_single_mask"].tolist() == [0]


def test_interior_leakage_remains_visible_when_all_three_endpoints_are_inside():
    masks = np.zeros((1, 12, 20), bool)
    masks[0, 4, [2, 6, 14]] = True
    result = m.measure_body_mask_containment(pose(), project, fields(masks))
    np.testing.assert_array_equal(result["endpoint_single_mask_max_distance_px"], [0.0])
    np.testing.assert_array_equal(result["single_mask_max_distance_px"], [4.0])
    np.testing.assert_array_equal(result["union_max_distance_px"], [4.0])


def test_subpixel_and_outside_distances_match_independent_dense_edt_oracle():
    masks = np.zeros((4, 13, 17), bool)
    masks[0, 3:8, 2:5] = True
    masks[1, 8:11, 10:15] = True
    masks[2, 0, 0] = True
    masks[3, 6, 12] = True
    bodies = np.array(
        [
            [[-2.5, -3.25, 1], [2.125, 6.875, 1], [18.5, 15.25, 1]],
            [[0.5, 0.5, 1], [7.75, 5.375, 1], [16.25, -1.125, 1]],
        ]
    )
    result = m.measure_body_mask_containment(bodies, project, fields(masks))
    oracle = dense_oracle(masks, result["projected_xy"])
    np.testing.assert_allclose(result["distances_by_body_sample_mask"], oracle, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        result["single_mask_max_distance_px"], oracle.max(1).min(1), rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        result["union_max_distance_px"], oracle.min(2).max(1), rtol=0, atol=1e-14
    )
    # Pixel (0.5,0.5) samples bilinear integer EDT, not continuous nearest center.
    assert oracle[1, 0, 2] == pytest.approx((2 + np.sqrt(2)) / 4)
    assert oracle[1, 0, 2] != pytest.approx(np.sqrt(0.5))


def test_interiors_are_interpolated_in_three_dimensions_before_projection():
    bodies = np.array([[[2.0, 4.0, 1.0], [12.0, 8.0, 4.0], [20.0, 6.0, 2.0]]])

    def perspective(xyz):
        return xyz[..., :2] / xyz[..., 2:3]

    result = m.measure_body_mask_containment(bodies, perspective, None)
    expected_xyz = np.array(
        [
            [2, 4, 1],
            [12, 8, 4],
            [20, 6, 2],
            [4.5, 5, 1.75],
            [7, 6, 2.5],
            [9.5, 7, 3.25],
            [14, 7.5, 3.5],
            [16, 7, 3],
            [18, 6.5, 2.5],
        ],
        dtype=float,
    )[None]
    np.testing.assert_array_equal(result["sample_xyz"], expected_xyz)
    np.testing.assert_allclose(result["projected_xy"], perspective(expected_xyz), rtol=0, atol=0)
    assert not np.allclose(result["projected_xy"][0, 4], perspective(bodies)[0, :2].mean(0))


def test_original_mask_indices_and_permutation_preserve_distances_and_tie_count():
    masks = np.zeros((3, 12, 20), bool)
    masks[0, 3:6, 1:17] = masks[2, 3:6, 1:17] = True
    masks[1, 11, 19] = True
    indices = np.array([8, 2, 14], np.int32)
    result = m.measure_body_mask_containment(pose(), project, fields(masks), indices)
    permutation = np.array([1, 2, 0])
    permuted = m.measure_body_mask_containment(
        pose(), project, fields(masks[permutation]), indices[permutation]
    )
    assert result["best_single_mask"].tolist() == [8]
    assert permuted["best_single_mask"].tolist() == [14]  # deterministic first tied mask
    assert result["best_single_mask_tie_count"].tolist() == [2]
    assert permuted["best_single_mask_tie_count"].tolist() == [2]
    np.testing.assert_array_equal(
        permuted["distances_by_body_sample_mask"],
        result["distances_by_body_sample_mask"][..., permutation],
    )
    np.testing.assert_array_equal(
        permuted["single_mask_max_distance_px"], result["single_mask_max_distance_px"]
    )


def test_unavailable_fields_keep_geometry_without_manufacturing_a_zero_distance():
    result = m.measure_body_mask_containment(pose(), project, None)
    assert result["available"] is False and result["sample_xyz"].shape == (1, 9, 3)
    for key in (
        "single_mask_max_distance_px",
        "union_max_distance_px",
        "best_single_mask",
        "distances_by_body_sample_mask",
        "mask_switch_advantage_px",
    ):
        assert key not in result
    with pytest.raises(ValueError, match="labels without fields"):
        m.measure_body_mask_containment(pose(), project, None, [0])


def test_inputs_and_sparse_indices_are_unchanged_and_returned_labels_do_not_alias():
    masks = np.ones((1, 12, 20), bool)
    body = pose()
    indices = np.array([7], np.int64)
    backend = fields(masks)
    before = [x.tobytes() for x in (masks, body, indices, *backend.centers_yx)]
    result = m.measure_body_mask_containment(body, project, backend, indices)
    assert [x.tobytes() for x in (masks, body, indices, *backend.centers_yx)] == before
    result["mask_indices"][:] = 999
    np.testing.assert_array_equal(indices, [7])
    result["sample_xyz"][:] = 999
    np.testing.assert_array_equal(body, pose())


@pytest.mark.parametrize("damage", ["missing_batch", "two_nodes", "nan", "inf"])
def test_incomplete_or_nonfinite_body_is_rejected(damage):
    body = pose()
    if damage == "missing_batch":
        body = body[0]
    elif damage == "two_nodes":
        body = body[:, :2]
    else:
        body[0, 0, 0] = np.nan if damage == "nan" else np.inf
    with pytest.raises(ValueError, match="finite complete body"):
        m.measure_body_mask_containment(body, project, None)


@pytest.mark.parametrize("damage", ["shape", "nan"])
def test_invalid_projection_is_rejected(damage):
    def callback(xyz):
        return np.zeros((1, 3, 2)) if damage == "shape" else xyz[..., :2] * np.nan

    with pytest.raises(ValueError, match="finite projections"):
        m.measure_body_mask_containment(pose(), callback, None)


@pytest.mark.parametrize("labels", [[2, 2], [-1, 4], [1.0, 2.0], [1]])
def test_invalid_mask_labels_are_not_silently_reassigned(labels):
    with pytest.raises(ValueError, match="unique nonnegative"):
        m.measure_body_mask_containment(pose(), project, fields(np.ones((2, 12, 20), bool)), labels)


def test_invalid_field_object_or_zero_fields_is_not_a_successful_match():
    for value in (object(), m.SparseDistanceFields((12, 20), (), ())):
        with pytest.raises(ValueError, match="nonempty sparse"):
            m.measure_body_mask_containment(pose(), project, value)

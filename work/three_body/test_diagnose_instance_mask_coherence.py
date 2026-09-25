"""Synthetic CPU checks for instance-mask coherence; no real tracking inputs."""

import json

import diagnose_instance_mask_coherence as d
import numpy as np
import pytest


def _metrics(fields, indices=None):
    if indices is None:
        indices = np.arange(len(fields["valid"]), dtype=np.int64)
    return d.coherence_metrics(fields, np.asarray(indices, dtype=np.int64))


def _assert_nan_costs(result):
    for key in ("C_single", "C_union", "delta"):
        assert np.isnan(result[key]), key


def test_project_body_samples_in_3d_before_nonlinear_projection():
    class NonlinearCamera:
        def __init__(self):
            self.calls = []

        def project(self, xyz):
            self.calls.append(xyz.copy())
            return np.column_stack((xyz[:, 0] + xyz[:, 2] ** 2, xyz[:, 1]))

    pose = np.array([[-2.0, 0, 0], [2, 4, 2], [8, 0, 4]])
    before = pose.copy()
    camera = NonlinearCamera()
    result = d.project_body(pose, camera)

    assert result["points_xyz"].shape == (33, 3)
    assert result["projected_uv"].shape == (33, 2)
    np.testing.assert_array_equal(result["points_xyz"][[0, 16, 32]], pose)
    np.testing.assert_array_equal(result["points_xyz"][[8, 24]], [[0, 2, 1], [5, 2, 3]])
    np.testing.assert_allclose(
        np.diff(result["points_xyz"][:17], axis=0), np.tile([0.25, 0.25, 0.125], (16, 1))
    )
    np.testing.assert_allclose(
        np.diff(result["points_xyz"][16:], axis=0), np.tile([0.375, -0.25, 0.125], (16, 1))
    )
    assert np.count_nonzero(np.all(result["points_xyz"] == pose[1], axis=1)) == 1
    np.testing.assert_array_equal(result["projected_uv"][[8, 24]], [[1, 2], [14, 2]])
    assert result["projected_uv"][8, 0] != np.mean(result["projected_uv"][[0, 16], 0])
    assert len(camera.calls) == 1
    np.testing.assert_array_equal(camera.calls[0], result["points_xyz"])
    np.testing.assert_array_equal(pose, before)


def test_exact_pixel_center_distances_and_signed_inside_margin():
    masks = np.zeros((1, 5, 5), dtype=bool)
    masks[0, 2, 2] = True
    fields = d.sample_mask_fields(masks, True, np.array([[2.0, 2], [3, 2], [0, 0]]))

    assert fields["membership"].dtype == np.int8
    assert fields["mask_status"].dtype.kind == "U"
    assert fields["mask_status"].tolist() == ["mixed", "mixed"]
    for row in (0, 1):
        np.testing.assert_array_equal(fields["membership"][row], [1, 0, 0])
        np.testing.assert_allclose(fields["outside_distance"][row], [0, 1, np.sqrt(8)])
        np.testing.assert_allclose(fields["signed_distance"][row], [-1, 1, np.sqrt(8)])
        np.testing.assert_allclose(fields["inside_margin"][row], [1, 0, 0])


def test_inside_margin_uses_native_background_without_image_padding():
    masks = np.ones((1, 5, 5), dtype=bool)
    masks[0, 4, 4] = False
    fields = d.sample_mask_fields(masks, True, np.array([[0.0, 0], [4, 0], [4, 4]]))

    np.testing.assert_allclose(fields["inside_margin"][0], [np.sqrt(32), 4, 0])
    np.testing.assert_allclose(fields["signed_distance"][0], [-np.sqrt(32), -4, 1])
    np.testing.assert_allclose(fields["outside_distance"][0], [0, 0, 1])


def test_rounding_is_ties_to_even_and_invalid_pixels_remain_unknown():
    masks = np.ones((1, 3, 4), dtype=bool)
    uv = np.array(
        [
            [-0.5, 0.5],
            [0.5, 1.5],
            [2.5, 1],
            [3.5, 1],
            [np.nan, 1],
            [np.inf, 0],
            [-0.51, 0],
            [0, -0.5],
            [1e100, 0],
        ]
    )
    fields = d.sample_mask_fields(masks, True, uv)

    assert fields["valid"].dtype == np.bool_
    assert fields["finite"].dtype == np.bool_
    assert fields["pixel_xy"].dtype == np.int64
    np.testing.assert_array_equal(
        fields["finite"], [True, True, True, True, False, False, True, True, True]
    )
    np.testing.assert_array_equal(
        fields["valid"], [True, True, True, False, False, False, False, True, False]
    )
    np.testing.assert_array_equal(
        fields["pixel_xy"][[0, 1, 2, 7]], [[0, 0], [0, 2], [2, 1], [0, 0]]
    )
    invalid = ~fields["valid"]
    assert np.all(fields["pixel_xy"][invalid] == -1)
    assert np.all(fields["membership"][:, invalid] == -1)
    for key in ("outside_distance", "signed_distance", "inside_margin"):
        assert np.isnan(fields[key][:, invalid]).all()


def test_split_instances_with_union_gap_have_distinct_coherence_costs():
    masks = np.zeros((2, 3, 5), dtype=bool)
    masks[0, 1, 1] = True
    masks[1, 1, 3] = True
    fields = d.sample_mask_fields(masks, True, np.array([[1.0, 1], [2, 1], [3, 1]]))
    result = _metrics(fields)

    np.testing.assert_array_equal(fields["outside_distance"], [[0, 1, 2], [2, 1, 0], [0, 1, 0]])
    assert result["C_single"] == 1
    assert result["C_union"] == pytest.approx(1 / 3)
    assert result["delta"] == pytest.approx(2 / 3)
    assert result["best_mask_rows"] == [0, 1]
    assert result["coherence_status"] == "finite"
    assert result["projection_status"] == "complete"
    assert result["observation_status"] == "available"
    assert result["requested_points"] == result["valid_points"] == 3
    for row in result["per_mask"]:
        assert row["inside_count"] == 1
        assert row["inside_fraction"] == pytest.approx(1 / 3)
        assert row["mean_outside_distance"] == 1
        assert row["max_outside_distance"] == 2
    assert result["union"]["inside_count"] == 2
    assert result["union"]["inside_fraction"] == pytest.approx(2 / 3)
    assert result["union"]["max_outside_distance"] == 1


def test_duplicate_instances_preserve_every_exact_best_row_tie():
    masks = np.zeros((3, 3, 3), dtype=bool)
    masks[:, 1, 1] = True
    result = _metrics(d.sample_mask_fields(masks, True, np.array([[0.0, 0], [2, 2]])))

    assert len(result["per_mask"]) == 3
    assert result["best_mask_rows"] == [0, 1, 2]
    assert result["C_single"] == pytest.approx(np.sqrt(2))
    assert result["C_union"] == pytest.approx(np.sqrt(2))
    assert result["delta"] == 0


def test_empty_and_full_masks_have_explicit_infinite_distance_semantics():
    masks = np.zeros((2, 3, 4), dtype=bool)
    masks[1] = True
    fields = d.sample_mask_fields(masks, True, np.array([[0.0, 0], [3, 2]]))
    result = _metrics(fields)

    assert fields["mask_status"].tolist() == ["empty", "full", "full"]
    np.testing.assert_array_equal(fields["membership"], [[0, 0], [1, 1], [1, 1]])
    assert np.isposinf(fields["outside_distance"][0]).all()
    assert np.isposinf(fields["signed_distance"][0]).all()
    assert np.all(fields["inside_margin"][0] == 0)
    assert np.all(fields["outside_distance"][1:] == 0)
    assert np.isneginf(fields["signed_distance"][1:]).all()
    assert np.isposinf(fields["inside_margin"][1:]).all()
    assert result["best_mask_rows"] == [1]
    assert result["coherence_status"] == "finite"
    assert result["C_single"] == result["C_union"] == result["delta"] == 0
    assert np.isposinf(result["per_mask"][0]["mean_outside_distance"])


def test_all_empty_instances_keep_infinite_costs_and_all_ties():
    masks = np.zeros((3, 2, 4), dtype=bool)
    fields = d.sample_mask_fields(masks, True, np.array([[0.0, 0], [3, 1]]))
    result = _metrics(fields)

    assert fields["mask_status"].tolist() == ["empty"] * 4
    assert result["coherence_status"] == "empty_union"
    assert result["observation_status"] == "available"
    assert result["best_mask_rows"] == [0, 1, 2]
    assert np.isposinf(result["C_single"])
    assert np.isposinf(result["C_union"])
    assert np.isnan(result["delta"])
    assert result["union"]["inside_count"] == 0
    assert result["union"]["inside_fraction"] == 0


def test_available_zero_instances_is_an_empty_union_not_a_missing_view():
    fields = d.sample_mask_fields(
        np.zeros((0, 2, 4), dtype=bool), True, np.array([[0.0, 0], [3, 1]])
    )
    result = _metrics(fields)

    assert fields["membership"].shape == (1, 2)
    assert fields["mask_status"].tolist() == ["empty"]
    assert np.all(fields["membership"] == 0)
    assert np.isposinf(fields["outside_distance"]).all()
    assert result["coherence_status"] == "no_instances"
    assert result["observation_status"] == "available_no_instances"
    assert result["projection_status"] == "complete"
    assert result["per_mask"] == []
    assert result["best_mask_rows"] == []
    assert np.isnan(result["C_single"])
    assert np.isposinf(result["C_union"])
    assert np.isnan(result["delta"])


def test_missing_view_keeps_unknown_observations_and_requires_no_masks():
    masks = np.zeros((0, 2, 4), dtype=bool)
    uv = np.array([[0.0, 0], [3, 1]])
    fields = d.sample_mask_fields(masks, False, uv)
    result = _metrics(fields)

    assert fields["mask_status"].tolist() == ["missing"]
    assert np.all(fields["valid"])
    assert np.all(fields["membership"] == -1)
    for key in ("outside_distance", "signed_distance", "inside_margin"):
        assert np.isnan(fields[key]).all()
    assert result["coherence_status"] == "missing"
    assert result["observation_status"] == "missing"
    assert result["projection_status"] == "complete"
    assert result["per_mask"] == []
    assert result["best_mask_rows"] == []
    _assert_nan_costs(result)
    with pytest.raises(ValueError):
        d.sample_mask_fields(np.zeros((1, 2, 4), dtype=bool), False, uv)


def test_partial_projection_uses_one_selected_valid_set_for_every_mask():
    masks = np.zeros((2, 3, 5), dtype=bool)
    masks[0, 1, 1] = True
    masks[1, 1, 3] = True
    uv = np.array([[1.0, 1], [3, 1], [99, 0], [np.nan, 0], [0, 0]])
    fields = d.sample_mask_fields(masks, True, uv)
    result = _metrics(fields, [0, 1, 2, 3])
    valid_only = _metrics(fields, [0, 1])

    assert result["requested_points"] == 4
    assert result["valid_points"] == 2
    assert result["projection_status"] == "partial"
    assert result["coherence_status"] == "finite"
    assert result["C_single"] == 1
    assert result["C_union"] == 0
    assert result["delta"] == 1
    assert result["per_mask"] == valid_only["per_mask"]
    assert result["union"] == valid_only["union"]
    assert [row["inside_fraction"] for row in result["per_mask"]] == [0.5, 0.5]
    assert result["union"]["inside_fraction"] == 1


def test_all_invalid_points_are_unknown_and_empty_selection_is_rejected():
    masks = np.zeros((2, 3, 5), dtype=bool)
    masks[0, 1, 1] = True
    masks[1] = True
    fields = d.sample_mask_fields(masks, True, np.array([[99.0, 0], [np.nan, 1]]))

    result = _metrics(fields, [0, 1])
    assert result["requested_points"] == 2
    assert result["valid_points"] == 0
    assert result["projection_status"] == "none"
    assert result["observation_status"] == "available"
    assert result["coherence_status"] == "no_valid_points"
    assert result["best_mask_rows"] == []
    _assert_nan_costs(result)
    for row in result["per_mask"] + [result["union"]]:
        assert row["inside_count"] is None
        assert np.isnan(row["inside_fraction"])
        assert np.isnan(row["mean_outside_distance"])
        assert np.isnan(row["max_outside_distance"])
    with pytest.raises(ValueError, match="unique in-range point indices"):
        _metrics(fields, [])


def test_sampling_and_aggregation_accept_readonly_inputs_without_mutation():
    masks = np.zeros((2, 3, 5), dtype=bool)
    masks[0, 1, 1] = True
    masks[1, 1, 3] = True
    uv = np.array([[1.2, 1], [2.2, 1], [np.nan, 0]])
    indices = np.array([2, 0, 1], dtype=np.int64)
    snapshots = [value.copy() for value in (masks, uv, indices)]
    for value in (masks, uv, indices):
        value.flags.writeable = False

    fields = d.sample_mask_fields(masks, True, uv)
    field_snapshots = {}
    for key, value in fields.items():
        if isinstance(value, np.ndarray):
            field_snapshots[key] = value.copy()
            value.flags.writeable = False
    result = d.coherence_metrics(fields, indices)

    assert result["projection_status"] == "partial"
    for actual, before in zip((masks, uv, indices), snapshots, strict=True):
        np.testing.assert_array_equal(actual, before)
    for key, before in field_snapshots.items():
        np.testing.assert_array_equal(fields[key], before)


def test_json_encoding_preserves_distinctions_between_unknown_and_infinities():
    values = {
        "distance": np.array([np.nan, np.inf, -np.inf, 1.25]),
        "summary": {"missing_count": None, "empty_union_cost": np.float64(np.inf)},
        "count": np.int64(2),
    }
    encoded = d.encode_nonfinite(values)
    restored = json.loads(json.dumps(encoded, allow_nan=False))

    assert restored == {
        "distance": ["NaN", "+inf", "-inf", 1.25],
        "summary": {"missing_count": None, "empty_union_cost": "+inf"},
        "count": 2,
    }
    assert np.isnan(values["distance"][0])
    assert np.isposinf(values["distance"][1])
    assert np.isneginf(values["distance"][2])


def test_group_means_expose_partial_exclusion_without_dropping_case_count():
    import diagnose_instance_mask_coherence as d

    full = np.ones((1, 3, 3), dtype=bool)
    complete = d.coherence_metrics(
        d.sample_mask_fields(full, True, np.array([[1.0, 1.0]])), np.array([0])
    )
    partial = d.coherence_metrics(
        d.sample_mask_fields(full, True, np.array([[1.0, 1.0], [10.0, 1.0]])), np.array([0, 1])
    )
    rows = [
        dict(method=d.METHODS[0], view=d.VIEWS[0], metrics={k: v for k in d.PARTS})
        for v in (complete, partial)
    ]
    group = d.aggregate_records(rows)[0]
    assert group["requested_body_cases"] == 2
    assert group["finite_complete_cases"] == 1 and group["other_cases"] == 1
    assert group["projection_status"] == {"complete": 1, "partial": 1}
    assert group["mean_over_finite_complete"] == {"C_single": 0.0, "C_union": 0.0, "delta": 0.0}

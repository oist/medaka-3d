"""Pure fixed200 reporting checks; no images, model, real geometry or saved arrays."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import diagnose_semantic_node200_geometry as m


def snapshot(arrays):
    return [(value.dtype.str, value.shape, value.tobytes()) for value in arrays]


def point_fixture():
    queries, payloads, inputs = {}, {}, []
    for vi, view in enumerate(m.VIEWS):
        query = dict(
            body_slot=np.repeat(np.arange(3, dtype=np.int32), 3),
            node_index=np.tile(np.arange(3, dtype=np.int32), 3),
        )
        payload = {"lk": {}, "emissions": {}}
        for method, (group, xy_key, flag_key) in enumerate(
            (
                ("lk", "xy", "valid"),
                ("emissions", "first_xy", "first_visibility"),
                ("emissions", "final_xy", "final_visibility"),
            )
        ):
            t, q = np.indices((200, 9))
            code = (method * 100000 + vi * 10000 + q * 1000 + t).astype(np.float32)
            xy = np.stack((code, -code - np.float32(0.25)), axis=-1)
            flags = (t + q + method + vi) % 4 != 0
            xy[5, 2] = np.nan
            flags[5, 2] = False
            # Duplicate positions remain separate semantic queries and flags.
            xy[8, 1] = xy[8, 0]
            flags[8, :2] = [True, False]
            payload[group][xy_key], payload[group][flag_key] = xy, flags
            inputs.extend((xy, flags))
        queries[view], payloads[view] = query, payload
        inputs.extend(query.values())
    return queries, payloads, inputs


def test_first_horizons_are_release_times_not_track_times_or_call_indices():
    actual = m.first_horizons()
    assert actual.dtype == np.int64 and actual.shape == (200,)
    assert actual[:16].tolist() == [15] * 16
    for call in range(1, 24):
        new_start = 8 + 8 * call
        assert actual[new_start : new_start + 8].tolist() == [new_start + 7] * 8
    assert actual[[0, 15, 16, 23, 24, 191, 192, 199]].tolist() == [
        15,
        15,
        23,
        23,
        31,
        191,
        199,
        199,
    ]
    assert (m.FIRST + actual)[[0, -1]].tolist() == [333534, 333718]
    lookahead = actual - np.arange(200)
    assert lookahead.max() == 15 and lookahead[16:].max() == 7
    actual[:] = -1
    assert (m.first_horizons() >= 15).all()


def test_scatter_preserves_all_method_body_view_node_axes_and_missing_values():
    queries, payloads, inputs = point_fixture()
    before = snapshot(inputs)
    coordinates, reported, initial = m.scatter_points(queries, payloads)
    assert initial.shape == (3, 3, 3) and initial.dtype == bool and initial.all()
    for method in m.METHODS:
        assert coordinates[method].shape == (200, 3, 3, 3, 2)
        assert coordinates[method].dtype == np.float32
        assert reported[method].shape == (200, 3, 3, 3)
        assert reported[method].dtype == bool
        for vi, view in enumerate(m.VIEWS):
            if method == "lk":
                xy, flags = payloads[view]["lk"]["xy"], payloads[view]["lk"]["valid"]
            else:
                prefix = "first" if method == "ct_first" else "final"
                xy = payloads[view]["emissions"][prefix + "_xy"]
                flags = payloads[view]["emissions"][prefix + "_visibility"]
            for body in range(3):
                for node in range(3):
                    q = 3 * body + node
                    assert coordinates[method][:, body, vi, node].tobytes() == xy[:, q].tobytes()
                    np.testing.assert_array_equal(reported[method][:, body, vi, node], flags[:, q])
        # A false flag must not delete a finite coordinate or merge a duplicate.
        assert np.isfinite(coordinates[method][8, 0, :, 1]).all()
        assert not reported[method][8, 0, :, 1].any()
        assert np.isnan(coordinates[method][5, 0, :, 2]).all()
        for out in (coordinates[method], reported[method]):
            assert all(not np.shares_memory(out, value) for value in inputs)
    assert snapshot(inputs) == before


@pytest.mark.parametrize("kind", ["reordered", "duplicated"])
def test_scatter_rejects_query_labels_that_break_fixed_semantic_order(kind):
    queries, payloads, _ = point_fixture()
    nodes = queries[m.VIEWS[1]]["node_index"]
    if kind == "reordered":
        nodes[[0, 1]] = nodes[[1, 0]]
    else:
        nodes[1] = nodes[0]
    with pytest.raises(ValueError, match="order"):
        m.scatter_points(queries, payloads)


def reference_pose():
    pose = np.zeros((3, 3, 3), np.float64)
    pose[:, :, 0] = [0, 1, 3]
    pose[:, :, 1] = np.arange(3)[:, None] * 10
    return pose


def test_measures_are_relative_to_reference_with_translation_scale_and_direction():
    reference = reference_pose()
    pose = reference.copy()
    pose[0] += [0, 0, 4]
    pose[1] = [[0, 10, 0], [0, 12, 0], [0, 16, 0]]
    pose[2, :, 0] *= -1
    valid = np.ones((3, 3), bool)
    before = snapshot([reference, pose, valid])
    result = m.measures(dict(pose_xyz=pose, node_valid=valid), reference)
    np.testing.assert_allclose(
        result["node_reference_distance_mm"], [[4, 4, 4], [0, np.sqrt(5), np.sqrt(45)], [0, 2, 6]]
    )
    np.testing.assert_allclose(result["segment_length_mm"], [[1, 2], [2, 4], [1, 2]])
    np.testing.assert_allclose(result["segment_length_delta_mm"], [[0, 0], [1, 2], [0, 0]])
    np.testing.assert_allclose(
        result["segment_reference_angle_degrees"], [[0, 0], [90, 90], [180, 180]]
    )
    assert result["segment_valid"].all() and result["segment_direction_valid"].all()
    # Names explicitly describe reference-relative diagnostics, not error to truth.
    assert "node_reference_distance_mm" in result and "ground_truth_error_mm" not in result
    assert snapshot([reference, pose, valid]) == before


def test_invalid_middle_node_invalidates_both_adjacent_segments_without_bridging():
    reference = reference_pose()
    pose, valid = reference.copy(), np.ones((3, 3), bool)
    pose[0, 1] = 1e12
    valid[0, 1] = False
    pose[1, 2] = np.nan
    valid[1, 2] = False
    before = snapshot([reference, pose, valid])
    result = m.measures(dict(pose_xyz=pose, node_valid=valid), reference)
    np.testing.assert_array_equal(
        result["segment_valid"], [[False, False], [True, False], [True, True]]
    )
    for name in ("segment_length_mm", "segment_length_delta_mm", "segment_reference_angle_degrees"):
        assert np.isnan(result[name][0]).all()
        assert np.isnan(result[name][1, 1])
    assert np.isnan(result["node_reference_distance_mm"][0, 1])
    assert result["segment_length_mm"].shape == (3, 2)
    assert snapshot([reference, pose, valid]) == before


def test_zero_length_preserves_length_but_has_no_direction():
    reference = reference_pose()
    pose = reference.copy()
    pose[0, 1] = pose[0, 0]
    reference[1, 1] = reference[1, 0]
    result = m.measures(dict(pose_xyz=pose, node_valid=np.ones((3, 3), bool)), reference)
    assert result["segment_valid"].all()
    assert result["segment_length_mm"][0, 0] == 0
    assert result["segment_length_delta_mm"][0, 0] == -1
    assert result["reference_segment_length_mm"][1, 0] == 0
    assert not result["segment_direction_valid"][0, 0]
    assert not result["segment_direction_valid"][1, 0]
    assert np.isnan(result["segment_reference_angle_degrees"][[0, 1], 0]).all()


def test_entirely_unavailable_geometry_remains_nan_with_all_six_segments_retained():
    result = m.measures(
        dict(pose_xyz=np.full((3, 3, 3), np.nan), node_valid=np.zeros((3, 3), bool)),
        reference_pose(),
    )
    assert not result["segment_valid"].any()
    assert not result["segment_direction_valid"].any()
    assert np.isnan(result["node_reference_distance_mm"]).all()
    assert np.isnan(result["segment_length_mm"]).all()
    assert np.isnan(result["segment_reference_angle_degrees"]).all()
    np.testing.assert_array_equal(result["reference_segment_length_mm"], [[1, 2]] * 3)


def test_availability_gain_loss_orientations_and_reference_deficit_keep_all199():
    first, second = np.zeros((199, 3, 3), bool), np.zeros((199, 3, 3), bool)
    observed = np.ones_like(first)
    first[0, 0, 0] = second[0, 0, 0] = True
    first[198, 2, 2] = True
    second[1, 1, 2] = True
    observed[1, 1, 2] = observed[100, 2, 1] = False
    inputs = [first, second, observed]
    before = snapshot(inputs)
    result = m.availability_comparison(*inputs)
    assert result["common"].sum() == result["gained"].sum() == result["lost"].sum() == 1
    assert result["gained"][1, 1, 2] and result["lost"][198, 2, 2]
    assert result["neither"].sum() == 1788
    assert result["reference_deficit"].sum() == 2
    assert result["deficit_gained"].sum() == result["deficit_second_available"].sum() == 1
    assert not result["deficit_first_available"].any()
    assert not result["deficit_lost"].any() and not result["deficit_common"].any()
    partition = sum(result[key].astype(np.int8) for key in ("common", "gained", "lost", "neither"))
    np.testing.assert_array_equal(partition, np.ones((199, 3, 3), np.int8))
    for value in result.values():
        assert value.dtype == bool and value.shape == (199, 3, 3)
        assert all(not np.shares_memory(value, source) for source in inputs)
    assert snapshot(inputs) == before


@pytest.mark.parametrize("kind", ["includes_anchor", "flattened", "integer_flags"])
def test_availability_rejects_changed_denominator_or_boolean_schema(kind):
    a = np.ones((199, 3, 3), bool)
    if kind == "includes_anchor":
        a = np.ones((200, 3, 3), bool)
    elif kind == "flattened":
        a = a.reshape(199, 9)
    else:
        a = a.astype(np.uint8)
    with pytest.raises(ValueError, match="availability schema"):
        m.availability_comparison(a, a.copy(), a.copy())

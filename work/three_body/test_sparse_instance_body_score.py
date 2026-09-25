"""Synthetic exact-parity tests against the frozen dense EDT implementation."""

import dataclasses
import hashlib
from pathlib import Path

import instance_body_score as dense
import numpy as np
import pytest
import sparse_instance_body_score as sparse
from scipy.ndimage import distance_transform_edt


def assert_same_array(a, b):
    assert a.shape == b.shape and a.dtype.str == b.dtype.str
    assert a.tobytes() == b.tobytes()


def assert_same_result(a, b):
    assert set(a) == set(b)
    for key in a:
        if isinstance(a[key], np.ndarray):
            assert_same_array(a[key], b[key])
        else:
            assert a[key] == b[key], key


def scene():
    masks = np.zeros((3, 100, 180), bool)
    points = []
    for k, x in enumerate((10, 70, 130)):
        masks[k, 30:46, x : x + 30] = True
        points.append([[x + 4.0, 35.0], [x + 15.0, 37.0], [x + 25.0, 40.0]])
    points = np.array(points)
    pose = np.concatenate([points, np.ones((3, 3, 1))], axis=-1)[None]
    return masks, points, pose


def project(xyz):
    return xyz[..., :2]


def observations(masks, points, cfg=dense.DEFAULT_CONFIG):
    a = dense.prepare_instance_observation(masks, points, cfg)
    b = sparse.prepare_sparse_instance_observation(masks, points, cfg)
    for name in ("eligible", "reason", "image_shape_hw", "evidence", "config"):
        assert getattr(a, name) == getattr(b, name), name
    if a.sleap_row_by_instance is None:
        assert b.sleap_row_by_instance is None and b.sparse_fields is None
    else:
        assert_same_array(a.sleap_row_by_instance, b.sleap_row_by_instance)
    return a, b


def test_reference_source_is_the_frozen_core():
    assert (
        hashlib.sha256(Path(dense.__file__).read_bytes()).hexdigest()
        == sparse.REFERENCE_SOURCE_SHA256
    )


@pytest.mark.parametrize("shape", [(1, 1), (1, 43), (41, 1), (49, 53), (111, 127)])
@pytest.mark.parametrize("density", [0.015, 0.2, 1.0])
def test_random_masks_integer_lattice_subpixels_borders_and_nonfinite_bit_exact(shape, density):
    rng = np.random.default_rng(4501 + shape[0] + shape[1])
    masks = rng.random((3, *shape)) < density
    masks[:, 0, 0] = True
    fields = sparse.SparseDistanceFields.from_masks(masks)
    edt = np.stack([distance_transform_edt(~mask) for mask in masks])
    yy, xx = np.indices(shape)
    integer = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    subpixel = rng.uniform([-20, -20], [shape[1] + 20, shape[0] + 20], size=(1000, 2))
    edges = np.array(
        [
            [0.0, 0.0],
            [shape[1] - 1.0, shape[0] - 1.0],
            [-0.25, 1.25],
            [shape[1] - 0.25, shape[0] - 0.75],
            [np.nan, 0],
            [0, np.inf],
            [-np.inf, np.inf],
            [np.nan, np.nan],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )
    xy = np.concatenate([integer, subpixel, edges])
    before = (masks.tobytes(), xy.tobytes())
    actual, expected = fields.sample(xy), dense.sample_distance_fields(edt, xy)
    assert_same_array(actual, expected)
    assert before == (masks.tobytes(), xy.tobytes())
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert fields.minimum_separation(a, b) == edt[a][masks[b]].min()
    assert_same_array(fields.sample(xy[:24].reshape(2, 3, 4, 2)), expected[:24].reshape(2, 3, 4, 3))
    assert_same_array(fields.sample(xy[0]), expected[0])
    assert_same_array(fields.sample(np.empty((0, 2))), np.empty((0, 3)))


def test_bilinear_integer_edt_is_not_direct_subpixel_nearest_neighbor_distance():
    masks = np.zeros((1, 4, 4), bool)
    masks[0, 0, 0] = True
    fields = sparse.SparseDistanceFields.from_masks(masks)
    xy = np.array([[0.5, 0.5], [-2.5, -3.5]])
    result = fields.sample(xy)
    assert result[0, 0] == (2 + np.sqrt(2)) / 4
    assert result[0, 0] != np.sqrt(0.5)
    assert result[1, 0] == np.linalg.norm(xy[1])


def test_equidistant_centers_and_boundary_extension_are_bit_exact():
    masks = np.zeros((2, 11, 11), bool)
    masks[0, [3, 3, 7, 7], [3, 7, 3, 7]] = True
    masks[1, 5, 5] = True
    fields = sparse.SparseDistanceFields.from_masks(masks)
    xy = np.array([[5, 5], [5.25, 5.25], [-1.5, 6.25], [12.75, -3.125], [5, 5]], float)
    dense_fields = np.stack([distance_transform_edt(~mask) for mask in masks])
    assert_same_array(fields.sample(xy), dense.sample_distance_fields(dense_fields, xy))
    # The tree owns foreground coordinates and retains no input masks/full-image fields.
    before = fields.sample(xy)
    masks[:] = False
    assert_same_array(fields.sample(xy), before)
    for centers, tree in zip(fields.centers_yx, fields.trees, strict=True):
        assert centers.shape[1] == 2 and tree.data.shape == centers.shape
        assert not centers.flags.writeable and not tree.data.flags.writeable


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "zero",
        "two",
        "empty",
        "overlap",
        "touch",
        "no_sl",
        "incomplete",
        "merged",
        "duplicate",
        "unmatched",
        "wide_tolerance",
    ],
)
def test_observation_gate_reasons_and_full_evidence_match(fault):
    masks, points, pose = scene()
    cfg = dense.DEFAULT_CONFIG
    if fault == "none":
        masks = None
    elif fault == "zero":
        masks = masks[:0]
    elif fault == "two":
        masks = masks[:2]
    elif fault == "empty":
        masks[2] = False
    elif fault == "overlap":
        masks[0] |= masks[1]
    elif fault == "touch":
        masks[0, 30:46, 39:70] = True
    elif fault == "no_sl":
        points = None
    elif fault == "incomplete":
        points[2, 2] = np.nan
    elif fault == "merged":
        points[2] = points[0]
    elif fault == "duplicate":
        points = np.concatenate([points, points[:1]])
    elif fault == "unmatched":
        points = np.concatenate([points, np.full((1, 3, 2), [0.0, 90.0])])
    elif fault == "wide_tolerance":
        cfg = dataclasses.replace(cfg, sleap_mask_tolerance_px=200)
    a, b = observations(masks, points, cfg)

    def guarded_project(x):
        assert a.eligible, "an unavailable observation must not project a candidate"
        return project(x)

    assert_same_result(
        dense.score_instance_bodies(pose, guarded_project, a),
        sparse.score_sparse_instance_bodies(pose, guarded_project, b),
    )


@pytest.mark.parametrize("gap", [2, 3, 4])
def test_mask_separation_gate_exact_boundary(gap):
    masks, points, _ = scene()
    masks[0, 30:46, 39 : 70 - gap + 1] = True
    a, _ = observations(masks, points)
    assert a.evidence["pair_mask_distance_px_01_02_12"][0] == gap
    assert a.eligible == (gap > 3)


@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_nextafter_mask_gap_and_sleap_tolerance_thresholds(direction):
    masks, points, _ = scene()
    masks[0, 30:46, 39:68] = True  # gap to mask1 is exactly 3 lattice pixels.
    threshold = 3.0 if direction == 0 else np.nextafter(3.0, -np.inf if direction < 0 else np.inf)
    a, _ = observations(
        masks, points, dataclasses.replace(dense.DEFAULT_CONFIG, minimum_instance_gap_px=threshold)
    )
    assert a.eligible == (direction < 0)
    masks, points, _ = scene()
    points[0, :, 0] = 2.0  # Exactly 8 pixels from the first instance.
    threshold = 8.0 if direction == 0 else np.nextafter(8.0, -np.inf if direction < 0 else np.inf)
    a, _ = observations(
        masks, points, dataclasses.replace(dense.DEFAULT_CONFIG, sleap_mask_tolerance_px=threshold)
    )
    assert a.eligible == (direction >= 0)
    points[0, :, 0] = (
        2.0 if direction == 0 else np.nextafter(2.0, -np.inf if direction < 0 else np.inf)
    )
    observations(masks, points)  # Coordinate roundoff, independently of tolerance roundoff.


@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_nextafter_assignment_tie_tolerance_has_the_same_decision(direction):
    masks, points, pose = scene()
    a, _ = observations(masks, points)
    result = dense.score_instance_bodies(pose, project, a)
    gap = float(np.min(result["assignment_costs"][result["assignment_costs"] > 0]))
    tolerance = gap if direction == 0 else np.nextafter(gap, -np.inf if direction < 0 else np.inf)
    a, b = observations(
        masks, points, dataclasses.replace(dense.DEFAULT_CONFIG, tie_atol=tolerance)
    )
    expected = dense.score_instance_bodies(pose, project, a)
    actual = sparse.score_sparse_instance_bodies(pose, project, b)
    assert_same_result(actual, expected)
    assert (actual["optimal_assignment_mask"].sum() > 1) == (direction >= 0)


@pytest.mark.parametrize("configuration", ["default", "zero_weights", "fractional", "strict"])
def test_all_scoring_arrays_and_assignments_bit_exact_with_mixed_invalid_random_candidates(
    configuration,
):
    masks, points, pose = scene()
    cfg = dense.DEFAULT_CONFIG
    if configuration == "zero_weights":
        cfg = dataclasses.replace(cfg, worst_node_weight=0.0, segment_leakage_weight=0.0)
    elif configuration == "fractional":
        cfg = dataclasses.replace(
            cfg, interior_fractions=(0.13, 0.29, 0.61, 0.89), pixel_scale_px=3.7
        )
    elif configuration == "strict":
        cfg = dataclasses.replace(cfg, endpoint_tolerance_px=0.0, interior_tolerance_px=0.0)
    a, b = observations(masks, points, cfg)
    rng = np.random.default_rng(1452)
    bank = np.repeat(pose, 35, axis=0)
    bank[1, 2, :2] = pose[0, 0, :2]
    bank[2, 2] = pose[0, 0]
    bank[3, 0, 0, 2] = np.nan
    bank[5:] += rng.normal(0, 42, size=bank[5:].shape)

    def invalid_projection(x):
        result = project(x).copy()
        result[4, 0, 0] = np.nan
        return result

    before = [x.tobytes() for x in (masks, points, bank)]
    actual = sparse.score_sparse_instance_bodies(bank, invalid_projection, b)
    expected = dense.score_instance_bodies(bank, invalid_projection, a)
    assert_same_result(actual, expected)
    assert before == [x.tobytes() for x in (masks, points, bank)]
    assert np.all(actual["best_assignment"][3:5] == -1)
    assert not actual["optimal_assignment_mask"][3:5].any()
    if configuration == "zero_weights":
        assert actual["optimal_assignment_mask"][0].sum() == 6


@pytest.mark.parametrize("seed", range(5))
def test_irregular_separated_masks_and_subpixel_detection_gates(seed):
    masks, points, pose = scene()
    rng = np.random.default_rng(seed)
    masks &= rng.random(masks.shape) > 0.3
    points += rng.uniform(-0.5, 0.5, size=points.shape)
    a, b = observations(masks, points)
    bank = np.repeat(pose, 24, axis=0) + rng.normal(0, 12, size=(24, 3, 3, 3))
    assert_same_result(
        dense.score_instance_bodies(bank, project, a),
        sparse.score_sparse_instance_bodies(bank, project, b),
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((3, 2, 2), float),
        np.zeros((0, 2, 2), bool),
        np.zeros((1, 0, 2), bool),
        np.zeros((1, 3, 3), bool),
    ],
)
def test_sparse_fields_reject_undefined_or_non_boolean_inputs(bad):
    with pytest.raises(ValueError):
        sparse.SparseDistanceFields.from_masks(bad)


@pytest.mark.parametrize(
    "bad", [np.zeros((3, 2, 2), float), np.zeros((3, 0, 2), bool), np.zeros((3, 2), bool)]
)
def test_bad_observation_inputs_raise_like_the_frozen_gate(bad):
    for prepare in (dense.prepare_instance_observation, sparse.prepare_sparse_instance_observation):
        with pytest.raises(ValueError):
            prepare(bad, None)

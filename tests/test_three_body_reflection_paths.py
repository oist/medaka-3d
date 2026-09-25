"""Fixed-plane optical paths only: no image fitting, fish labels, or GPU."""

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from medaka3d.calib import refract


def _load_module():
    path = (Path(__file__).resolve().parents[1]
            / "labs/20260909_three_body/reflection_paths.py")
    spec = importlib.util.spec_from_file_location("_test_reflection_paths", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m = _load_module()
INDICES = {"index_water": 1.3325, "index_glass": 1.52, "index_air": 1.0003}


def trace(origins, directions, **indices):
    return m.trace_y192_reflections(np.array(origins), np.array(directions),
                                   **(INDICES | indices))


def unit(values):
    array = np.array(values, dtype=float)
    return array / np.linalg.norm(array, axis=-1, keepdims=True)


def assert_no_alias(value, inputs):
    for field in fields(value):
        array = getattr(value, field.name)
        if isinstance(array, np.ndarray):
            assert all(not np.shares_memory(array, source) for source in inputs)
        else:
            assert_no_alias(array, inputs)


def test_normal_incidence_finite_segments_and_power_weights():
    result = trace([96., 0., 93.], [0., 1., 0.])
    assert result.direct_valid
    assert result.first_wall == m.Wall.Y192
    assert result.first_wall_mask == 1 << m.Wall.Y192
    assert result.first_reason == m.Reason.OK
    assert result.first_distance_mm == 192
    np.testing.assert_array_equal(result.first_hit_mm, [96, 192, 93])
    np.testing.assert_array_equal(result.outer_hit_mm, [96, 196, 93])
    np.testing.assert_array_equal(result.return_point_mm, result.first_hit_mm)
    r_wg = ((1.3325 - 1.52) / (1.3325 + 1.52)) ** 2
    r_ga = ((1.52 - 1.0003) / (1.52 + 1.0003)) ** 2
    for branch in (result.inner, result.outer):
        assert branch.valid and branch.reason == m.Reason.OK
        np.testing.assert_array_equal(branch.origins, [96, 192, 93])
        np.testing.assert_array_equal(branch.directions, [0, -1, 0])
        assert branch.next_wall == m.Wall.Y0
        assert branch.max_distance_mm == 192
    assert result.inner.weight == pytest.approx(r_wg, abs=1e-16)
    assert result.outer.weight == pytest.approx((1-r_wg)**2 * r_ga, abs=1e-16)


def test_reflection_law_and_virtual_point_collinearity():
    origin = np.array([70., 160., 90.])
    incoming = unit([.2, .9, -.1])
    result = trace(origin, incoming)
    branch = result.inner
    assert branch.valid
    normal = np.array([0., 1., 0.])
    expected = incoming - 2 * (incoming @ normal) * normal
    np.testing.assert_allclose(branch.directions, expected, atol=1e-15, rtol=0)
    assert np.dot(incoming, normal) == pytest.approx(-branch.directions @ normal)
    real_point = branch.origins + 27 * branch.directions
    mirror_point = real_point.copy()
    mirror_point[1] = 384 - mirror_point[1]
    np.testing.assert_allclose(mirror_point, origin + (result.first_distance_mm + 27)
                              * incoming, atol=5e-14, rtol=0)
    assert branch.max_distance_mm > 27


def test_oblique_snell_roundtrip_and_glass_leg_aperture_geometry():
    origin = np.array([80., 140., 100.])
    direction = unit([.25, 1., -.2])
    result = trace(origin, direction)
    assert result.inner.valid and result.outer.valid
    glass = refract(direction, [0, 1, 0], 1.3325 / 1.52)
    np.testing.assert_allclose(result.glass_out_directions, glass, atol=2e-15, rtol=0)
    reflected_glass = glass * [1, -1, 1]
    water = refract(reflected_glass, [0, -1, 0], 1.52 / 1.3325)
    np.testing.assert_allclose(result.outer.directions, water, atol=2e-15, rtol=0)
    np.testing.assert_allclose(result.outer.directions, direction * [1, -1, 1],
                              atol=2e-15, rtol=0)
    np.testing.assert_allclose(result.outer_hit_mm - result.first_hit_mm,
                              glass * (4 / glass[1]), atol=2e-14, rtol=0)
    np.testing.assert_allclose(result.return_point_mm - result.outer_hit_mm,
                              reflected_glass * (4 / glass[1]), atol=2e-14, rtol=0)


def test_outer_fresnel_preserves_s_p_polarization_and_outer_tir():
    # Two angles: one ordinary reflected beam; one glass/air TIR beam.
    theta = np.deg2rad([35., 60.])
    directions = np.column_stack((np.sin(theta), np.cos(theta), np.zeros(2)))
    result = trace([[80., 185., 90.]] * 2, directions)
    assert result.outer.valid.all()
    np.testing.assert_array_equal(result.outer_tir, [False, True])
    ci = np.cos(theta)
    cg = np.sqrt(1 - (1.3325 / 1.52)**2 * np.sin(theta)**2)

    def fresnel_pair(n1, n2, cosine1, cosine2):
        rs = ((n1*cosine1 - n2*cosine2) / (n1*cosine1 + n2*cosine2))**2
        rp = ((n2*cosine1 - n1*cosine2) / (n2*cosine1 + n1*cosine2))**2
        return rs, rp

    rs, rp = fresnel_pair(1.3325, 1.52, ci, cg)
    ca0 = np.sqrt(1 - (1.52 / 1.0003)**2 * (1 - cg[0]**2))
    rs_outer0, rp_outer0 = fresnel_pair(1.52, 1.0003, cg[0], ca0)
    expected = [.5*((1-rs[0])**2*rs_outer0 + (1-rp[0])**2*rp_outer0),
                .5*((1-rs[1])**2 + (1-rp[1])**2)]
    np.testing.assert_allclose(result.outer.weight, expected, atol=1e-15, rtol=0)
    wrong = ((1-result.reflectance_water_glass)**2 * result.reflectance_glass_air)
    assert np.all(np.abs(result.outer.weight - wrong) > 1e-6)
    assert np.all(result.inner.weight + result.outer.weight <= 1)


def test_first_wall_floor_and_behind_wall_are_not_skipped():
    directions = [[0, 1, 0], [1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 1]]
    result = trace([[96, 96, 93]] * 5, directions)
    np.testing.assert_array_equal(result.first_wall,
                                  [m.Wall.Y192, m.Wall.X192, m.Wall.Y0,
                                   m.Wall.Z0, m.Wall.Z186])
    np.testing.assert_array_equal(result.inner.valid, [True, False, False, False, False])
    assert np.all(result.inner.reason[1:] == m.Reason.OTHER_FIRST_WALL)
    # It would cross y192 later, but hits floor first: no wall branch.
    floor = trace([96, 190, 1], unit([0, 1, -2]))
    assert floor.first_wall == m.Wall.Z0 and not floor.inner.valid


def test_corner_and_boundary_parallel_hits_have_explicit_edge_masks():
    corner = trace([96, 96, 96], unit([1, 1, -1]))
    assert corner.direct_valid and not corner.inner.valid
    assert corner.first_reason == m.Reason.FIRST_HIT_EDGE
    assert corner.first_wall_mask == (1 << m.Wall.X192 | 1 << m.Wall.Y192 | 1 << m.Wall.Z0)
    parallel = trace([0, 160, 80], [0, 1, 0])
    assert parallel.first_wall == m.Wall.Y192
    assert parallel.first_wall_mask == (1 << m.Wall.X0 | 1 << m.Wall.Y192)
    assert parallel.direct_valid and not parallel.outer.valid
    # Starting at a water-entry interface inward is allowed, outward is not.
    at_wall = trace([[96, 192, 90]] * 2, [[0, -1, 0], [0, 1, 0]])
    np.testing.assert_array_equal(at_wall.direct_valid, [True, False])
    assert at_wall.first_reason[1] == m.Reason.NO_POSITIVE_EXIT


def test_outer_and_return_aperture_failures_are_distinct():
    # With equal water/glass n the lateral shift on each glass leg is exactly4.
    result = trace([[190, 191, 90], [186, 191, 90], [187, 191, 90], [183, 191, 90]],
                   [unit([1, 1, 0])] * 4, index_glass=1.3325)
    np.testing.assert_array_equal(result.inner.valid, [True]*4)
    np.testing.assert_array_equal(result.outer.reason,
                                  [m.Reason.OUTER_APERTURE_OUTSIDE,
                                   m.Reason.RETURN_APERTURE_OUTSIDE,
                                   m.Reason.OUTER_APERTURE_EDGE,
                                   m.Reason.RETURN_APERTURE_EDGE])
    assert not result.outer.valid.any()
    assert np.isnan(result.outer.origins).all() and np.all(result.outer.weight == 0)
    assert np.isfinite(result.outer_hit_mm).all()
    assert np.isnan(result.return_point_mm[[0, 2]]).all()
    assert np.isfinite(result.return_point_mm[[1, 3]]).all()


def test_inner_tir_remains_reflected_but_cannot_reach_outer_face():
    result = trace([90, 191, 90], unit([1, .2, 0]),
                   index_water=1.5, index_glass=1.)
    assert result.inner.valid and result.inner.weight == 1
    assert not result.outer.valid and result.outer.reason == m.Reason.WATER_GLASS_TIR
    assert np.isnan(result.outer_hit_mm).all()
    assert not result.outer_tir


def test_invalid_rays_masked_individually_no_engineering_floor_extension():
    origins = [[96, 100, 90], [96, 100, 90], [96, 100, -0.01],
               [96, 100, np.nan], [96, 100, 90]]
    result = trace(origins, [[0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 2, 0]])
    np.testing.assert_array_equal(result.direct_valid, [True, False, False, False, False])
    np.testing.assert_array_equal(result.first_reason,
                                  [m.Reason.OK, m.Reason.NONUNIT_DIRECTION,
                                   m.Reason.ORIGIN_OUTSIDE_BOX, m.Reason.NONFINITE_RAY,
                                   m.Reason.NONUNIT_DIRECTION])
    assert np.isnan(result.first_hit_mm[1:]).all()
    assert np.all(result.outer.weight[1:] == 0)


def test_batch_shapes_permutation_empty_and_no_input_aliases():
    origins = np.array([[[80, 150, 90], [90, 160, 80], [100, 170, 100]]], dtype=float)
    directions = unit([[[.1, 1, .2], [-.2, 1, .1], [.3, 1, -.1]]])
    before = origins.copy(), directions.copy()
    result = trace(origins, directions)
    assert result.inner.origins.shape == (1, 3, 3)
    assert result.inner.valid.shape == (1, 3) and result.inner.valid.all()
    permuted = trace(origins[:, [2, 0, 1]], directions[:, [2, 0, 1]])
    np.testing.assert_array_equal(permuted.outer.origins, result.outer.origins[:, [2, 0, 1]])
    np.testing.assert_array_equal(permuted.outer.weight, result.outer.weight[:, [2, 0, 1]])
    assert_no_alias(result, [origins, directions])
    np.testing.assert_array_equal(origins, before[0])
    np.testing.assert_array_equal(directions, before[1])
    empty = trace(np.empty((2, 0, 3)), np.empty((2, 0, 3)))
    assert empty.outer.valid.shape == (2, 0)
    result.outer.origins[...] = -999
    np.testing.assert_array_equal(origins, before[0])


def test_malformed_shape_dtype_and_indices_rejected():
    for bad_origins, bad_directions in [([1, 2], [1, 2]), ([[1, 2, 3]], [1, 2, 3]),
                                        ([True]*3, [0, 1, 0]), ([1j]*3, [0, 1, 0])]:
        with pytest.raises(ValueError, match="matching"):
            trace(bad_origins, bad_directions)
    for value in [0, -1, np.nan, np.inf, True, [1.3], 1e-300]:
        with pytest.raises(ValueError, match="refractive"):
            trace([90, 90, 90], [0, 1, 0], index_water=value)


def test_next_corner_exit_is_a_finite_bound_not_another_reflection():
    # Inner hit(100,192,100), outgoing [0,-1,-1]/sqrt2 reaches z0 before y0.
    result = trace([100, 182, 110], unit([0, 1, -1]))
    assert result.inner.valid
    assert result.inner.next_wall == m.Wall.Z0
    assert result.inner.max_distance_mm == pytest.approx(100*np.sqrt(2))
    endpoint = result.inner.origins + result.inner.max_distance_mm * result.inner.directions
    np.testing.assert_allclose(endpoint, [100, 92, 0], atol=1e-13, rtol=0)
    farther = result.inner.origins + (result.inner.max_distance_mm + 1) * result.inner.directions
    assert farther[2] < 0  # downstream renderer must exclude this continuation


def test_entry_plane_roundoff_is_classification_tolerance_not_floor_extension():
    origins = np.array([[96, -4.44e-16, 90], [96, -1e-7, 90], [96, 96, -6.]])
    result = trace(origins, [[0, 1, 0]] * 3)
    np.testing.assert_array_equal(result.direct_valid, [True, False, False])
    assert result.inner.valid[0]
    assert np.all(result.first_reason[1:] == m.Reason.ORIGIN_OUTSIDE_BOX)
    np.testing.assert_array_equal(origins[0], [96, -4.44e-16, 90])
    # Accepted float32 unit-cache roundoff keeps the geometric distance in mm.
    rounded = trace([96, 0, 90], np.array([0, 1 + 5e-8, 0]))
    assert rounded.first_distance_mm == 192
    np.testing.assert_array_equal(rounded.inner.directions, [0, -1, 0])

"""Synthetic slab mathematics and native batching only; no real image/rig reads."""

import numpy as np
import prepare_top_ray_domain_feasibility as m
import pytest


@pytest.fixture(autouse=True)
def no_real_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path)

    def fail(*args, **kwargs):
        pytest.fail("real input entry forbidden")

    for owner, name in (
        (m, "load_context"),
        (m, "run"),
        (m.calib, "load_rig"),
        (m.previous, "decode"),
        (m.cv2, "VideoCapture"),
        (m.cv2, "imdecode"),
        (m.np, "load"),
    ):
        monkeypatch.setattr(owner, name, fail)


@pytest.mark.parametrize(
    "origin,direction,expected,interval",
    [
        ((1, 1, 1), (1, 0, 0), True, (0, 1)),  # starts inside
        ((-1, 1, 1), (1, 0, 0), True, (1, 3)),  # starts outside, enters later
        ((-1, 1, 1), (-1, 0, 0), False, None),  # all intersections behind ray
        ((0, 1, 1), (-1, 0, 0), False, None),  # immediate outward exit has zero length
        ((0, 1, 1), (0, 1, 0), True, (0, 1)),  # lies on boundary plane, positive segment
        ((-1, 1, 0), (1, 1, 0), False, None),  # touches corner only, entry==exit
        ((-1, 3, 1), (1, 0, 0), False, None),  # parallel outside slab
        ((2, 2, 2), (-1, -1, -1), True, (0, 2)),  # corner inward
        ((1, 1, 1), (0, 0, 0), False, None),  # zero direction is invalid
        ((np.nan, 1, 1), (1, 0, 0), False, None),
        ((1, 1, 1), (np.inf, 0, 0), False, None),
    ],
)
def test_closed_box_forward_positive_length(origin, direction, expected, interval):
    o, d = np.array([origin], float), np.array([direction], float)
    before_o, before_d = o.copy(), d.copy()
    hit, finite, enter, leave = m.ray_box_intersection(o, d, [0, 0, 0], [2, 2, 2])
    assert bool(hit[0]) is expected
    assert bool(finite[0]) == bool(np.isfinite(o).all() and np.isfinite(d).all() and np.any(d))
    if interval is None:
        assert np.isnan(enter[0]) and np.isnan(leave[0])
    else:
        np.testing.assert_array_equal([enter[0], leave[0]], interval)
    np.testing.assert_array_equal(o, before_o)
    np.testing.assert_array_equal(d, before_d)


def test_negative_z_domain_not_nominal_floor_and_direction_scale():
    o = np.array([[1, 1, -3]], float)
    d = np.array([[1, 0, 0]], float)
    full = m.ray_box_intersection(o, d, m.LOWER, m.UPPER)
    nominal = m.ray_box_intersection(o, d, [0, 0, 0], m.UPPER)
    assert full[0][0] and not nominal[0][0]
    scaled = m.ray_box_intersection(o, d * 2, m.LOWER, m.UPPER)
    assert scaled[0][0] and scaled[3][0] == full[3][0] / 2


@pytest.mark.parametrize(
    "origins,directions,lower,upper",
    [
        (np.zeros((2, 2)), np.zeros((2, 2)), [0, 0, 0], [1, 1, 1]),
        (np.zeros((2, 3)), np.zeros((1, 3)), [0, 0, 0], [1, 1, 1]),
        (np.zeros((1, 3), bool), np.ones((1, 3)), [0, 0, 0], [1, 1, 1]),
        (np.zeros((1, 3)), np.ones((1, 3)), [0, 0, 0], [0, 1, 1]),
        (np.zeros((1, 3)), np.ones((1, 3)), [0, 0, np.nan], [1, 1, 1]),
    ],
)
def test_malformed_ray_or_bounds(origins, directions, lower, upper):
    with pytest.raises(ValueError):
        m.ray_box_intersection(origins, directions, lower, upper)


def test_batched_native_pixel_order_and_status():
    calls = []

    class Camera:
        def rays(self, uv):
            calls.append(uv.copy())
            o = np.column_stack((uv, np.full(len(uv), 3.0)))
            d = np.tile([0.0, 0.0, -1.0], (len(uv), 1))
            d[(uv[:, 0] == 3) & (uv[:, 1] == 2)] = np.nan
            return o, d

    valid, status, stats = m.domain_map(Camera(), [0, 0, 0], [2, 2, 2], shape=(3, 4), batch_size=5)
    expected = np.array([[True, True, True, False]] * 3)
    np.testing.assert_array_equal(valid, expected)
    assert status[2, 3] == 2 and (status == 0).sum() == 2
    assert stats == dict(
        ray_calls=3,
        maximum_ray_batch=5,
        model_intersection_pixels=9,
        finite_nonintersection_pixels=2,
        invalid_ray_pixels=1,
        unknown_pixels=3,
        total_native_pixels=12,
    )
    np.testing.assert_array_equal(
        np.concatenate(calls), [[x, y] for y in range(3) for x in range(4)]
    )
    assert [len(c) for c in calls] == [5, 5, 2]

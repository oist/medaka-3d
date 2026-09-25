"""Synthetic point identity, loss, timing, and observation-boundary tests."""

import cv2
import numpy as np
import pytest
import temporal_point_core as c


def texture():
    image = np.random.default_rng(411).integers(0, 256, (96, 96), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0.6)


def test_queries_keep_duplicate_mask_references_without_mutating_input():
    image = texture()
    mask = np.zeros(image.shape, bool)
    mask[20:75, 20:75] = True
    masks = np.stack([mask, mask])
    before = image.copy(), masks.copy()
    q = c.select_queries(image, masks)
    n = q["rows"][0]["query_count"]
    assert n > 2 and n <= 64
    assert np.array_equal(q["xy"][:n], q["xy"][n:])
    assert np.array_equal(q["mask_index"], np.repeat([0, 1], n))
    assert q["rows"][1]["query_start"] == n
    assert np.array_equal(image, before[0]) and np.array_equal(masks, before[1])


@pytest.mark.parametrize("count", [1, 2])
def test_one_and_two_queries_survive_point_selection(monkeypatch, count):
    query = np.array([[20.0, 20.0], [25.0, 25.0]], np.float32)[:count]
    monkeypatch.setattr(cv2, "goodFeaturesToTrack", lambda *a, **kw: query[:, None].copy())
    q = c.select_queries(texture(), np.ones((1, 96, 96), bool))
    assert q["rows"][0]["status"] == "ready"
    assert np.array_equal(q["xy"], query)


def test_missing_empty_coreless_and_cornerless_are_separate():
    image = np.zeros((30, 30), np.uint8)
    missing = c.select_queries(image, None)
    empty = c.select_queries(image, np.empty((0, 30, 30), bool))
    masks = np.zeros((3, 30, 30), bool)
    masks[1, 15, 15] = True
    masks[2, 5:25, 5:25] = True
    result = c.select_queries(image, masks)
    assert missing["mask_count"] is None and not missing["available"]
    assert empty["mask_count"] == 0 and empty["available"]
    assert [row["status"] for row in result["rows"]] == ["empty_mask", "empty_core", "no_corners"]
    assert result["xy"].shape == (0, 2)


def test_query_core_uses_zero_outside_image_boundary():
    image = np.zeros((30, 30), np.uint8)
    result = c.select_queries(image, np.ones((1, 30, 30), bool))
    assert result["rows"][0]["mask_pixels"] == 900
    assert result["rows"][0]["core_pixels"] == 26 * 26


def test_lk_follows_known_translation_on_full_frames():
    first = texture()
    frames = [np.roll(first, i, axis=1) for i in range(4)]
    query = np.array([[30, 30], [50, 50], [65, 35]], np.float32)
    result = c.lk_chain(frames, query)
    assert result["valid"].all()
    expected = query[None] + np.stack([np.arange(4), np.zeros(4)], axis=-1)[:, None]
    np.testing.assert_allclose(result["xy"], expected, atol=0.1, rtol=0)
    assert not result["attempted"][0].any()
    assert result["attempted"][1:].all()


def test_lost_query_is_never_reseeded_or_passed_to_later_edge():
    calls = []

    def edge(a, b, xy):
        calls.append(xy.copy())
        result = c.lk_edge(a, b, xy)
        if len(calls) == 1:
            result["valid"][0] = False
            result["failure_bits"][0] = c.Failure.FORWARD
        return result

    first = texture()
    query = np.array([[30, 30], [50, 50]], np.float32)
    result = c.lk_chain([first] * 4, query, edge=edge)
    assert [len(x) for x in calls] == [2, 1, 1]
    assert np.isnan(result["xy"][1:, 0]).all()
    assert not result["attempted"][2:, 0].any()
    assert np.all(result["failure_bits"][2:, 0] == c.Failure.ALREADY_LOST)
    assert result["valid"][:, 1].all()


def test_undefined_failed_forward_is_never_given_to_backward(monkeypatch):
    calls = []

    def fake_lk(a, b, points, initial, **options):
        calls.append(points.copy())
        if len(calls) == 1:
            target = np.array([[[np.nan, np.nan]], [[21, 20]]], np.float32)
            return target, np.array([[0], [1]], np.uint8), np.array([[np.nan], [1]], np.float32)
        assert points.shape == (1, 1, 2) and np.isfinite(points).all()
        return (
            np.array([[[20, 20]]], np.float32),
            np.ones((1, 1), np.uint8),
            np.zeros((1, 1), np.float32),
        )

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)
    result = c.lk_edge(texture(), texture(), np.array([[10, 10], [20, 20]], np.float32))
    assert len(calls) == 2
    assert result["valid"].tolist() == [False, True]
    assert np.isnan(result["proposed_xy"][0]).all()
    assert result["failure_bits"][0] & c.Failure.FORWARD


def test_zero_query_keeps_time_axis_and_does_not_call_lk(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("zero-query LK invoked")

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", forbidden)
    result = c.lk_chain([texture()] * 64, np.empty((0, 2), np.float32))
    assert result["xy"].shape == (64, 0, 2)
    assert result["valid"].shape == (64, 0)


def test_mask_membership_does_not_clip_or_merge_missing_and_empty():
    xy = np.array([[-0.1, 0], [3.5, 2], [1.5, 1.5], [np.nan, 1], [1e30, 0]])
    masks = np.ones((2, 4, 4), bool)
    masks[1, 2, 2] = False
    result = c.mask_memberships(xy, masks, (4, 4))
    assert result["in_image"].tolist() == [False, True, True, False, False]
    assert result["sampled"].tolist() == [True, False, True, False, False]
    assert result["membership"].tolist() == [
        [True, True],
        [False, False],
        [True, False],
        [False, False],
        [False, False],
    ]
    assert not c.mask_memberships(xy, None, (4, 4))["available"]
    empty = c.mask_memberships(xy, np.empty((0, 4, 4), bool), (4, 4))
    assert empty["available"] and empty["membership"].shape == (5, 0)


def test_seven_windows_preserve_first_emission_and_final_revision():
    ledger = c.EmissionLedger(64, 2)
    for i, start in enumerate(range(0, 49, 8)):
        stop = start + 16
        xy = np.full((stop, 2, 2), i, np.float32)
        visible = np.full((stop, 2), i % 2 == 0, bool)
        ledger.add(xy, visible, input_start=start, input_stop=stop)
        xy[:] = 999
    arrays = ledger.arrays()
    assert arrays["first_horizon"].tolist() == [15] * 16 + [
        s for s in [23, 31, 39, 47, 55, 63] for _ in range(8)
    ]
    assert np.all(arrays["first_xy"][:16] == 0)
    assert np.all(arrays["final_xy"] == 6)
    assert np.all(arrays["final_horizon"] == 63)
    arrays["first_xy"][:] = -10
    assert np.all(ledger.first_xy[:16] == 0)


@pytest.mark.parametrize("kind", ["future", "repeat", "shrink"])
def test_emission_rejects_wrong_schedule(kind):
    ledger = c.EmissionLedger(64, 1)
    ledger.add(
        np.zeros((16, 1, 2), np.float32), np.zeros((16, 1), bool), input_start=0, input_stop=16
    )
    stop, length = {"future": (24, 25), "repeat": (16, 16), "shrink": (24, 8)}[kind]
    with pytest.raises(ValueError):
        ledger.add(
            np.zeros((length, 1, 2), np.float32),
            np.zeros((length, 1), bool),
            input_start=8,
            input_stop=stop,
        )


def test_bilinear_patch_matches_linear_ramp_without_padding():
    xx, yy = np.meshgrid(np.arange(32), np.arange(32))
    image = (xx + 2 * yy).astype(np.uint8)
    result = c.sample_patches(image, np.array([[10.25, 12.5], [4.999, 10], [np.nan, 12]]))
    dx, dy = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6))
    expected = (10.25 + dx + 2 * (12.5 + dy)).ravel() / 255
    np.testing.assert_allclose(result["values"][0], expected, atol=1e-15, rtol=0)
    assert result["inside"].tolist() == [True, False, False]
    assert np.isnan(result["values"][1:]).all()


def test_constant_patch_is_not_zncc_success_and_missing_is_not_zero_error():
    patches = np.array([[73 / 255] * 121, [np.nan] * 121], np.float64)
    result = c.patch_comparison(patches, patches)
    assert result["mad"][0] == 0 and np.isnan(result["mad"][1])
    assert np.isnan(result["zncc"]).all()
    assert result["pair_available"].tolist() == [True, False]
    assert not result["variance_ok"].any()


def test_patch_diagnostic_distinguishes_known_motion_from_stationary():
    image = texture()
    frames = [np.roll(image, i, axis=1) for i in range(4)]
    moving = np.array([[[30 + i, 30]] for i in range(4)], np.float32)
    stationary = np.repeat(moving[:1], 4, axis=0)
    tracked = c.patch_diagnostics(frames, moving)
    static = c.patch_diagnostics(frames, stationary)
    assert np.all(tracked["initial_mad"] == 0)
    assert np.all(static["initial_mad"][1:] > 0)
    assert np.isnan(tracked["adjacent_mad"][0]).all()
    np.testing.assert_allclose(tracked["adjacent_zncc"][1:], 1, atol=1e-14, rtol=0)


def test_paired_metrics_retain_one_sided_failure_and_common_query_domain():
    a = {"mad": np.array([[1.0, np.nan, 3.0, 4.0]])}
    b = {"mad": np.array([[2.0, 2.0, np.nan, 5.0]])}
    aa = np.array([[True, True, True, False]])
    bb = np.ones((1, 4), bool)
    result = c.paired_metrics(a, b, aa, bb)["mad"]
    assert result["common_reported"].tolist() == [[True, False, False, False]]
    assert result["a_only"].tolist() == [[False, False, True, False]]
    assert result["b_only"].tolist() == [[False, True, False, True]]
    assert result["b_minus_a"][0, 0] == 1
    assert np.isnan(result["b_minus_a"][0, 1:]).all()

"""Independent pool/weight/no-training-pixel checks for the spatial diagnostic."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
import body_pixel_evaluation as m  # noqa: E402
from observations import FrameObservation  # noqa: E402


def fixture(masks=None, shape=(80, 80), frame_idx=120):
    if masks is None:
        masks = np.zeros((2, *shape), bool)
        masks[0, 20:45, 20:45] = True
        masks[1, 30:50, 30:60] = True
    frame = FrameObservation(
        frame_idx,
        {},
        dict.fromkeys(m.VIEWS, np.zeros((*shape, 3), np.uint8)),
        dict.fromkeys(m.VIEWS, masks),
        {},
    )
    fit = dict.fromkeys(m.VIEWS, np.array([[0, 0], [21, 21], [21, 21], [50, 50]], float))
    return frame, fit


def test_exact_pools_rng_draws_and_train_exclusion():
    frame, fit = fixture()
    result = m.sample_body_evaluation(frame, fit)
    union = frame.masks[m.VIEWS[0]].any(axis=0)
    dilated = cv2.dilate(union.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    eligible = np.ones(union.size, bool)
    eligible[[0, 21 * 80 + 21, 50 * 80 + 50]] = False
    for v, view in enumerate(m.VIEWS):
        rng = np.random.default_rng(120 * 7 + v + 6137)
        flat = (result.pixels[view][:, 1] * 80 + result.pixels[view][:, 0]).astype(int)
        assert not np.intersect1d(flat, [0, 1701, 4050]).size
        for k, selector in enumerate((union, dilated & ~union, ~union)):
            pool = np.flatnonzero(selector.ravel() & eligible)
            size = min(len(pool), (768, 768, 512)[k])
            chosen = rng.choice(pool, size, replace=False)
            actual = flat[result.strata[v, : len(flat)] == k]
            assert np.array_equal(actual, chosen)
            assert len(np.unique(actual)) == size
            assert result.pool_counts[v, k] == len(pool)
            assert result.weights[v, result.strata[v] == k].sum() == pytest.approx(
                (1, 0.75, 0.25)[k]
            )
    assert np.array_equal(result.excluded_fit_unique_count, [3, 3, 3])


def test_ring_and_global_may_overlap_but_not_fit():
    frame, fit = fixture(
        shape=(20, 20), masks=np.pad(np.ones((1, 3, 3), bool), ((0, 0), (8, 9), (8, 9)))
    )
    fit = dict.fromkeys(m.VIEWS, np.array([[0.0, 0.0]]))
    result = m.sample_body_evaluation(frame, fit)
    for v, view in enumerate(m.VIEWS):
        xy = result.pixels[view]
        assert np.array_equal(
            np.sort(xy[result.strata[v] == 1], axis=0), np.sort(xy[result.strata[v] == 2], axis=0)
        )


def test_missing_and_empty_neutral_distinction():
    frame, fit = fixture(masks=np.empty((0, 80, 80), bool))
    frame.masks[m.VIEWS[0]] = None
    result = m.sample_body_evaluation(frame, fit)
    assert result.target.shape == (3, 0)
    assert not result.foreground_nonempty.any()
    assert np.array_equal(result.view_available, [False, True, True])
    assert result.pool_counts[0].sum() == 0
    assert result.pool_counts[1, 2] == 6397
    losses = m.summarize_pixel_error(np.empty((2, 3, 0)), np.empty((3, 0), bool), result)
    assert np.array_equal(losses["mean_view_loss"], [0, 0])
    assert not losses["valid_counts"].any()


def test_rgb_and_instance_order_do_not_change_eval():
    frame, fit = fixture()
    a = m.sample_body_evaluation(frame, fit)
    swapped = FrameObservation(
        frame.frame_idx,
        {},
        {v: 255 - frame.images_bgr[v] for v in m.VIEWS},
        {v: frame.masks[v][::-1] for v in m.VIEWS},
        {},
    )
    b = m.sample_body_evaluation(swapped, fit)
    for view in m.VIEWS:
        assert np.array_equal(a.pixels[view], b.pixels[view])
    for key in ("target", "weights", "strata", "sampled_counts", "pool_counts"):
        assert np.array_equal(getattr(a, key), getattr(b, key))


def test_invalid_rays_excluded_without_new_denominator():
    frame, fit = fixture()
    targets = m.sample_body_evaluation(frame, fit)
    valid = targets.strata >= 0
    union = np.full(targets.target.shape, 0.25)
    lost = np.flatnonzero(targets.strata[0] == 0)[0]
    valid[0, lost] = False
    result = m.summarize_pixel_error(union, valid, targets)
    expected = np.sum((union - targets.target) ** 2 * targets.weights * valid, axis=1)
    np.testing.assert_allclose(result["loss_by_view"][0], expected, atol=1e-15)
    assert result["valid_counts"][0, 0] == targets.sampled_counts[0, 0] - 1
    assert result["loss_by_stratum"].shape == (1, 3, 3)


@pytest.mark.parametrize("xy", [[[0.5, 3]], [[80, 2]], [[2, -1]], [[np.nan, 2]]])
def test_reject_bad_native_fit_coordinates(xy):
    frame, fit = fixture()
    fit[m.VIEWS[0]] = np.array(xy)
    with pytest.raises(ValueError, match="fitting pixels"):
        m.sample_body_evaluation(frame, fit)


def test_all_foreground_has_empty_background_strata():
    frame, _ = fixture(masks=np.ones((1, 80, 80), bool))
    fit = dict.fromkeys(m.VIEWS, np.array([[0, 0]], float))
    targets = m.sample_body_evaluation(frame, fit)
    assert np.array_equal(targets.sampled_counts, np.tile([768, 0, 0], (3, 1)))
    assert np.array_equal(targets.weights.sum(axis=1), np.ones(3))


def test_missing_view_padding_does_not_add_pixels_or_weights():
    frame, fit = fixture()
    frame.masks[m.VIEWS[1]] = None
    targets = m.sample_body_evaluation(frame, fit)
    assert targets.pixels[m.VIEWS[1]].shape == (0, 2)
    assert not targets.weights[1].any()
    assert np.all(targets.strata[1] == -1)
    assert targets.target.shape[1] == len(targets.pixels[m.VIEWS[0]])

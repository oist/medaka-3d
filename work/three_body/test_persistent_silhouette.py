"""Focused added-core and native decoding tests; no real inputs or GPU."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import measure_persistent_silhouette as subject


def test_core_border_empty_and_thin_detection():
    masks = np.zeros((3, 12, 12), bool)
    masks[0] = True
    masks[1, 4, 2:9] = True
    core = subject.cores(masks)
    assert np.count_nonzero(core[0]) == 8 * 8
    assert not core[1:].any()
    assert not core[:, :2].any()
    assert subject.cores(masks[:0]).shape == (0, 12, 12)


def test_overlapping_cores_report_nonadditive_union_and_invalid_rays():
    masks = np.ones((2, 12, 12), bool)
    flat = np.arange(144)
    alpha = np.tile([0.2, 0.4, 0.0], (144, 1))
    visible = np.tile([0.2, 0.32, 0.0], (144, 1))
    valid = np.ones(144, bool)
    valid[5 * 12 + 5] = False
    core, rows, union = subject.core_metrics(masks, flat, alpha, visible, valid)
    assert core.sum() == 128
    assert [r["valid_ray_area_px"] for r in rows] == [63, 63]
    assert union["valid_ray_area_px"] == 63
    assert union["unexplained_mass_px"] == pytest.approx(63 * 0.48)
    assert union["pair_shared_mean"][0] == pytest.approx(0.08)
    assert sum(union["unique_mass_px"]) != pytest.approx(union["union_mean"] * 63)


def test_empty_core_is_unavailable_not_zero_error():
    masks = np.zeros((1, 12, 12), bool)
    masks[0, 4, 4] = True
    _, rows, union = subject.core_metrics(
        masks, np.array([52]), np.zeros((1, 3)), np.zeros((1, 3)), np.ones(1, bool)
    )
    assert rows[0]["native_area_px"] == 0
    assert np.isnan(rows[0]["unexplained_mean"])
    assert np.isnan(union["union_mean"])


def test_core_instance_permutation_only_permutes_rows():
    masks = np.zeros((2, 16, 16), bool)
    masks[0, :10, :10] = True
    masks[1, 6:, 6:] = True
    rng = np.random.default_rng(42)
    alpha = rng.random((256, 3))
    visible = alpha * 0
    args = (np.arange(256), alpha, visible, np.ones(256, bool))
    _, rows, union = subject.core_metrics(masks, *args)
    _, permuted, perm_union = subject.core_metrics(masks[::-1], *args)
    for i in range(2):
        assert rows[i]["unexplained_mass_px"] == permuted[1 - i]["unexplained_mass_px"]
    assert union["unexplained_mass_px"] == perm_union["unexplained_mass_px"]


class Capture:
    def __init__(self, *, wrong_after=False, wrong_before=False):
        self.position = 0
        self.wrong_after = wrong_after
        self.wrong_before = wrong_before

    def set(self, key, value):
        assert key == cv2.CAP_PROP_POS_FRAMES
        self.position = value + self.wrong_before
        return True

    def get(self, key):
        assert key == cv2.CAP_PROP_POS_FRAMES
        return self.position

    def read(self):
        self.position += 1 + self.wrong_after
        return True, np.zeros((1200, 1200, 3), np.uint8)


def test_native_read_exact_index_and_hash():
    expected = np.zeros((1200, 1200, 3), np.uint8)
    actual = subject.native_read(Capture(), 2752, subject.array_sha256(expected))
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "kwargs,sha,error",
    [
        ({"wrong_before": True}, "0" * 64, "before"),
        ({"wrong_after": True}, "0" * 64, "after"),
        ({}, "0" * 64, "SHA"),
    ],
)
def test_native_read_rejects_wrong_frame_or_bytes(kwargs, sha, error):
    with pytest.raises(ValueError, match=error):
        subject.native_read(Capture(**kwargs), 2752, sha)


def test_d2_sampler_reads_shape_only():
    masks = np.zeros((1, 20, 20), bool)
    masks[0, 5:14, 5:14] = True
    targets = []
    for value in (0, 255):
        frame = subject.FrameObservation(
            79372,
            {},
            dict.fromkeys(subject.VIEWS, np.full((20, 20, 3), value, np.uint8)),
            dict.fromkeys(subject.VIEWS, masks),
            {},
        )
        targets.append(subject.sample_pixels(frame, "detectron2", 12, 12, 8))
    for view in subject.VIEWS:
        assert np.array_equal(targets[0].pixels[view], targets[1].pixels[view])
    for key in ("target", "weights", "foreground"):
        assert np.array_equal(getattr(targets[0], key), getattr(targets[1], key))

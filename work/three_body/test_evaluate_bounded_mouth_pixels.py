"""Unused evaluation pixels must exclude fitting coordinates in each native view."""

import numpy as np
import pytest
from evaluate_bounded_mouth_pixels import disjoint_pixels


def samples():
    fitted = np.tile(np.column_stack((np.arange(1024), np.zeros(1024))), (3, 1, 1))
    unused = np.tile(np.column_stack((np.arange(256), np.ones(256))), (3, 1, 1))
    return fitted, unused


def test_disjoint_pixels_accepts_separate_native_coordinates():
    disjoint_pixels(*samples())


@pytest.mark.parametrize("view", range(3))
def test_one_reused_pixel_in_any_view_rejected(view):
    fitted, unused = samples()
    unused[view, -1] = fitted[view, -1]
    with pytest.raises(ValueError, match="reused"):
        disjoint_pixels(fitted, unused)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1, 1200, 0.5])
def test_invalid_or_noninteger_coordinate_rejected(value):
    fitted, unused = samples()
    unused[0, 0, 0] = value
    with pytest.raises(ValueError, match="invalid"):
        disjoint_pixels(fitted, unused)


def test_cross_view_reuse_is_not_same_observation():
    fitted, unused = samples()
    fitted[1:, :, 1] = 2
    unused[1:, :, 1] = 0
    disjoint_pixels(fitted, unused)


def test_missing_view_rejected():
    fitted, unused = samples()
    with pytest.raises(ValueError, match="shape"):
        disjoint_pixels(fitted[:2], unused[:2])

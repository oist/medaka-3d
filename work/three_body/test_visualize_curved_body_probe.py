"""Small image-layout/ROI geometry checks without real data or rendering."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import visualize_curved_body_probe as m


def test_display_bounds_include_all_masks_and_curved_points():
    masks = np.zeros((2, 1200, 1200), bool)
    masks[0, 50:60, 100:110] = True
    masks[1, 500:600, 700:750] = True
    nodes = np.array([[[110.0, 100.0]] * 3] * 3)
    curve = np.array([[[850.0, 200.0], [300.0, 950.0]]] * 3)
    bounds = m.crop_bounds(masks, nodes, curve)
    assert bounds == (68, 18, 883, 983)


def test_bezier_nodes_and_zero_control_keep_original_endpoints():
    pose = np.arange(27, dtype=float).reshape(3, 3, 3)
    offsets = np.tile([1.0, -2.0, 4.0], (3, 1))
    lines = m.body_lines(pose, offsets)
    assert np.array_equal(lines[:, 0], pose[:, 0])
    assert np.array_equal(lines[:, 1], pose[:, 1])
    assert np.array_equal(lines[:, -1], pose[:, 2])
    np.testing.assert_allclose(lines[:, 17], (pose[:, 1] + pose[:, 2]) / 2 + offsets / 2)


def test_raw_cell_preserves_aspect_and_places_headers_outside_image():
    raw = np.full((100, 200, 3), 201, np.uint8)
    shown, transform = m.cell(raw, "raw", "native source", width=200)
    assert shown.shape == (264, 200, 3)
    assert transform["display_scale"] == 1
    x, y = transform["image_origin_xy"]
    assert y >= 64
    assert np.array_equal(shown[y : y + 100, x : x + 200], raw)

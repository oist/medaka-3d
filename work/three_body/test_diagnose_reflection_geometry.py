"""Small CPU invariants for post-hoc geometry and mask membership."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_reflection_geometry as diagnostic


def test_bilinear_native_centers_missing_boundary_and_empty_instances():
    masks = np.zeros((1, 3, 3), bool)
    masks[0, 1, 1] = True
    pixels = np.zeros((3, 3, 2), float)
    pixels[0] = [[1, 1], [1.5, 1], [2, 2]]
    pixels[1] = [[-0.01, 1], [3, 1], [np.nan, 1]]
    values, known = diagnostic.bilinear_membership(masks, pixels)
    np.testing.assert_array_equal(values[0, 0], [1, 0.5, 0])
    assert known[0].all() and not known[1].any() and np.isnan(values[0, 1]).all()
    empty, same = diagnostic.bilinear_membership(masks[:0], pixels)
    assert empty.shape == (0, 3, 3)
    np.testing.assert_array_equal(same, known)


def fixture():
    one = np.array([[0, 0, 1], [2, 0, 1], [4, 0, 1]], float)
    pose = np.broadcast_to(one, (2, 3, 3, 3)).copy()
    pose[:, 2, :, 1] += 20
    sources = np.broadcast_to(np.arange(3)[None, :, None, None], (2, 3, 3, 3)).copy()
    return dict(pose_xyz=pose, sources=sources)


def test_coincident_geometry_does_not_imply_reused_source_rows():
    data = fixture()
    result = diagnostic.geometry(data)
    assert not result["same_view_node_row_shared"].any()
    assert not result["cross_node_shared_instance_rows"].any()
    np.testing.assert_array_equal(result["pectoral_pair_distance_mm"], [[0, 20, 20]] * 2)
    np.testing.assert_array_equal(result["polyline_pair_distance_mm"], [[0, 20, 20]] * 2)
    np.testing.assert_array_equal(result["body_polyline_length_mm"], np.full((2, 3), 4.0))


def test_cross_node_row_sharing_is_separate_from_forbidden_same_node_reuse():
    data = fixture()
    data["sources"][:, 0, 0, 2] = -1
    data["sources"][:, 1, 0, 2] = 0
    result = diagnostic.geometry(data)
    assert result["cross_node_shared_instance_rows"][:, 0, 0].tolist() == [1, 1]
    assert not result["same_view_node_row_shared"].any()
    data["sources"][:, 0, 0, 2] = 0
    with pytest.raises(ValueError, match="used by two fish"):
        diagnostic.geometry(data)

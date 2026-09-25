"""Image volume votes are spatial evidence, including real multi-view ghosts."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from medaka3d.types import VIEWS

PATH = Path(__file__).resolve().parents[1] / "labs/20260909_three_body/mask_grid.py"
SPEC = importlib.util.spec_from_file_location("_three_body_mask_grid_test", PATH)
grid_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grid_module
SPEC.loader.exec_module(grid_module)


class Camera:
    def __init__(self, axes):
        self.axes = axes
        self.calls = 0

    def project(self, xyz):
        self.calls += 1
        return np.asarray(xyz)[..., self.axes]


class Rig:
    views = VIEWS
    world_units = "mm"

    def __init__(self):
        self.cameras = dict(
            zip(VIEWS, [Camera((0, 1)), Camera((0, 2)), Camera((1, 2))], strict=True)
        )

    def camera(self, view):
        return self.cameras[view]


def grid(**options):
    config = grid_module.MaskGridConfig(
        bounds_mm=((0, 8), (0, 8), (0, 8)), image_shape_hw=(16, 16), **options
    )
    return grid_module.MaskGrid.from_rig(Rig(), config)


def point_masks(instance_xyz):
    masks = {}
    for view, axes in zip(VIEWS, ((0, 1), (0, 2), (1, 2)), strict=True):
        array = np.zeros((len(instance_xyz), 16, 16), bool)
        for index, xyz in enumerate(instance_xyz):
            x, y = np.asarray(xyz)[list(axes)]
            array[index, y, x] = True
        masks[view] = array
    return masks


def test_fractional_native_sampler_and_unknown_image_boundary():
    image = np.array([[0.0, 1.0], [1.0, 0.0]])
    uv = np.array([[0.25, 0.25], [1, 0], [1, 1], [-0.01, 0], [np.nan, 0], [2, 1]])
    result = grid_module.bilinear_sample(image, uv)
    np.testing.assert_allclose(result[:3], [0.375, 1, 0])
    assert np.isnan(result[3:]).all()


def test_cell_centres_and_once_per_view_projection():
    value = grid()
    assert value.grid_shape == (4, 4, 4)
    np.testing.assert_array_equal(value.xyz[[0, -1]], [[1, 1, 1], [7, 7, 7]])
    masks = point_masks([[1, 1, 1]])
    for _ in range(3):
        value.evaluate(masks)
    assert [value.rig.camera(v).calls for v in VIEWS] == [1, 1, 1]
    assert not value.xyz.flags.writeable


def test_empty_available_and_missing_views_are_distinct_but_both_neutral():
    value = grid()
    masks = {
        VIEWS[0]: np.zeros((0, 16, 16), bool),
        VIEWS[1]: None,
        VIEWS[2]: np.zeros((1, 16, 16), bool),
    }
    result = value.evaluate(masks)
    np.testing.assert_array_equal(result.samples.view_available, [True, False, True])
    assert not result.samples.view_informative.any()
    assert np.isnan(result.samples.view_support).all()
    assert not result.samples.at_least_two.any()
    assert len(value.candidate_centers(result).xyz) == 0


def test_detector_instance_permutation_and_mapping_order_have_no_identity_effect():
    value = grid()
    masks = point_masks([[1, 1, 1], [3, 3, 3]])
    altered = {view: masks[view][::-1] for view in reversed(VIEWS)}
    left, right = value.evaluate(masks), value.evaluate(altered)
    np.testing.assert_array_equal(left.samples.view_support, right.samples.view_support)
    np.testing.assert_array_equal(
        value.candidate_centers(left).xyz, value.candidate_centers(right).xyz
    )


def test_known_third_view_contradiction_remains_in_two_view_pool():
    value = grid()
    masks = point_masks([[1, 1, 1], [1, 3, 3]])
    ghost = np.flatnonzero(np.all(value.xyz == [1, 1, 3], axis=1))[0]
    result = value.evaluate(masks)
    assert result.samples.at_least_two[ghost]
    assert result.samples.third_view_contradiction[ghost]
    assert not result.samples.strict_three[ghost]
    np.testing.assert_array_equal(result.samples.view_support[ghost], [1, 1, 0])
    masks[VIEWS[2]] = None
    unknown = value.evaluate(masks).samples
    assert unknown.at_least_two[ghost]
    assert not unknown.third_view_contradiction[ghost]
    assert unknown.known_view_count[ghost] == 2


def test_even_strict_three_silhouettes_can_create_a_nonanimal_ghost():
    value = grid()
    physical = np.array([[1, 1, 3], [1, 3, 1], [3, 1, 1]])
    ghost = np.array([1, 1, 1])
    assert not np.any(np.all(physical == ghost, axis=1))
    result = value.sample_points(ghost[None], point_masks(physical))
    assert result.strict_three[0]
    # A strict-three vote is deliberately not exposed as a correct-animal flag.
    assert not hasattr(result, "identity")


def test_healthy_soft_suppression_preserves_mask_support_and_nms_is_geometric():
    value = grid(max_candidates=12)
    masks = {view: np.ones((1, 16, 16), bool) for view in VIEWS}
    healthy = np.array([[[1, 1, 1], [1, 1, 7]]], float)
    original, result = value.evaluate(masks), value.evaluate(masks, healthy)
    np.testing.assert_array_equal(original.samples.view_support, result.samples.view_support)
    assert result.novelty_weight[0] == pytest.approx(0.2)
    assert (result.novelty_weight > 0).all()
    pool = value.candidate_centers(result)
    assert len(pool.xyz) == 12
    distance = np.linalg.norm(pool.xyz[:, None] - pool.xyz[None], axis=-1)
    assert np.all(distance[~np.eye(len(pool.xyz), dtype=bool)] > 2)
    assert pool.healthy_distance_mm[0] > 4


def test_wrong_view_order_dtype_and_hard_exclusion_configuration_are_rejected():
    rig = Rig()
    rig.views = VIEWS[::-1]
    with pytest.raises(ValueError, match="view order"):
        grid_module.MaskGrid(rig)
    with pytest.raises(ValueError, match="hard exclusion"):
        grid_module.MaskGridConfig(healthy_downweight=1)
    value = grid()
    masks = point_masks([[1, 1, 1]])
    masks[VIEWS[0]] = masks[VIEWS[0]].astype(float)
    with pytest.raises(ValueError, match="bool"):
        value.evaluate(masks)

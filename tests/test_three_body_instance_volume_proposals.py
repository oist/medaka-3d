"""Native mask evidence produces finite regions, not fish identities or landmarks."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from medaka3d.types import VIEWS

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
SPEC = importlib.util.spec_from_file_location(
    "_instance_volume_proposals_test", LAB / "instance_volume_proposals.py"
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def proposer(shape=(6, 6, 6), image_shape=(16, 16), shift=None, **options):
    axes = [1 + np.arange(n) * 2 for n in shape]
    xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(float)
    uv = np.stack([xyz[:, [0, 1]], xyz[:, [0, 2]], xyz[:, [1, 2]]], axis=1)
    if shift is not None:
        xyz = xyz + np.asarray(shift)
    return module.InstanceVolumeProposer(
        xyz, uv, shape, image_shape_hw=image_shape, config=module.InstanceVolumeConfig(**options)
    )


def masks_for_points(points, image_shape=(16, 16), top_rows=None):
    points = np.asarray(points, dtype=int).reshape(-1, 3)
    top_rows = list(range(len(points))) if top_rows is None else top_rows
    top_count = max(top_rows, default=-1) + 1
    result = {view: np.zeros((top_count if v == 0 else len(points), *image_shape), bool)
              for v, view in enumerate(VIEWS)}
    for i, point in enumerate(points):
        for v, axes in enumerate(((0, 1), (0, 2), (1, 2))):
            x, y = point[list(axes)]
            result[VIEWS[v]][top_rows[i] if v == 0 else i, y, x] = True
    return result


def signature(pool):
    return sorted((tuple(c.centroid_xyz), tuple(c.voxel_indices), c.voxel_count,
                   c.two_view_voxels, c.three_view_voxels)
                  for c in pool.components)


def test_line_volume_pca_axes_are_rows_and_full_cell_extents():
    grid = proposer()
    masks = masks_for_points([[1, 3, 3], [3, 3, 3], [5, 3, 3]], top_rows=[0, 0, 0])
    result = grid.evaluate(masks)
    assert len(result.components) == 1
    part = result.components[0]
    np.testing.assert_allclose(part.centroid_xyz, [3, 3, 3])
    np.testing.assert_allclose(part.pca_axes[0], [1, 0, 0])
    np.testing.assert_allclose(part.pca_axes @ part.pca_axes.T, np.eye(3))
    assert np.linalg.det(part.pca_axes) == pytest.approx(1)
    np.testing.assert_allclose(part.pca_eigenvalues_mm2, [8 / 3, 0, 0])
    np.testing.assert_allclose(part.pca_extent_mm, [6, 2, 2])
    assert part.volume_mm3 == 24 and part.voxel_count == 3
    assert part.pca_rank == 1 and part.pca_degenerate and part.principal_axis_defined
    assert part.three_view_voxels == 3 and part.two_view_voxels == 0
    assert result.diagnostics["centroids_are_pectoral_nodes"] is False


def test_pca_axes_diagonal_row_actually_diagonalizes_covariance():
    grid = proposer()
    # A thick staircase is six-connected and has a non-world-axis principal direction.
    points = [[1, 1, 1], [3, 1, 1], [3, 3, 1], [5, 3, 1], [5, 5, 1]]
    part = grid.evaluate(masks_for_points(points, top_rows=[0] * len(points))).components[0]
    selected = grid.xyz[part.voxel_indices] - part.centroid_xyz
    covariance = selected.T @ selected / len(selected)
    np.testing.assert_allclose(part.pca_axes @ covariance @ part.pca_axes.T,
                               np.diag(part.pca_eigenvalues_mm2), atol=1e-12)
    assert abs(part.pca_axes[0, 0]) > .5 and abs(part.pca_axes[0, 1]) > .5


def test_native_bilinear_threshold_uses_fractional_pixel_centres():
    grid = proposer(shape=(2, 1, 1), image_shape=(2, 2))
    uv = np.zeros((2, 3, 2))
    uv[:, 0] = [[.25, .25], [.5, .5]]
    value = module.InstanceVolumeProposer(grid.xyz, uv, grid.grid_shape, image_shape_hw=(2, 2))
    masks = {view: np.ones((1, 2, 2), bool) for view in VIEWS}
    masks[VIEWS[0]][0] = [[False, True], [True, False]]
    result = value.evaluate(masks)
    assert len(result.components) == 1
    np.testing.assert_array_equal(result.components[0].voxel_indices, [1])
    np.testing.assert_allclose(result.components[0].mean_view_support, [.5, 1, 1])


def test_integer_native_pixel_has_no_half_pixel_offset():
    grid = proposer(shape=(1, 1, 1), image_shape=(2, 2))
    uv = np.array([[[1, 0], [1, 0], [1, 0]]], float)
    value = module.InstanceVolumeProposer(grid.xyz, uv, grid.grid_shape, image_shape_hw=(2, 2))
    masks = {view: np.array([[[False, True], [False, False]]]) for view in VIEWS}
    result = value.evaluate(masks)
    assert result.components[0].three_view_voxels == 1


def test_known_empty_third_view_is_contradiction_missing_is_unknown():
    grid = proposer()
    masks = masks_for_points([[1, 1, 1]])
    masks[VIEWS[2]] = np.zeros((0, 16, 16), bool)
    known = grid.evaluate(masks)
    assert known.view_available.tolist() == [True, True, True]
    part = known.components[0]
    assert part.two_view_voxels == part.known_third_contradiction_voxels == 1
    assert part.unknown_third_view_voxels == 0
    np.testing.assert_array_equal(part.view_known_voxels, [1, 1, 1])
    masks[VIEWS[2]] = None
    missing = grid.evaluate(masks).components[0]
    assert missing.two_view_voxels == missing.unknown_third_view_voxels == 1
    assert missing.known_third_contradiction_voxels == 0
    np.testing.assert_array_equal(missing.view_known_voxels, [1, 1, 0])
    assert np.isnan(missing.mean_view_support[2])


@pytest.mark.parametrize("coordinate", [[-0.001, 1], [16, 1], [np.nan, 1], [np.inf, 1]])
def test_outside_third_projection_is_unknown_not_contradiction(coordinate):
    grid = proposer(shape=(1, 1, 1))
    uv = grid.projected_uv.copy()
    uv[:, 2] = coordinate
    value = module.InstanceVolumeProposer(grid.xyz, uv, grid.grid_shape, image_shape_hw=(16, 16))
    part = value.evaluate(masks_for_points([[1, 1, 1]])).components[0]
    assert part.unknown_third_view_voxels == 1 and part.known_third_contradiction_voxels == 0


def test_no_top_means_no_candidate_even_when_both_sides_support():
    grid = proposer()
    masks = masks_for_points([[1, 1, 1]])
    masks[VIEWS[0]] = None
    result = grid.evaluate(masks)
    assert not result.components and result.diagnostics["top_view_missing"]
    masks[VIEWS[0]] = np.zeros((0, 16, 16), bool)
    result = grid.evaluate(masks)
    assert not result.components and not result.diagnostics["top_view_missing"]
    assert result.diagnostics["top_instance_count"] == 0
    masks[VIEWS[0]] = np.zeros((1, 16, 16), bool)
    result = grid.evaluate(masks)
    assert not result.components and result.top_diagnostics[0]["status"] == "empty_top_instance"


def test_top_alone_never_makes_a_supported_volume():
    masks = masks_for_points([[1, 1, 1]])
    masks[VIEWS[1]], masks[VIEWS[2]] = None, np.zeros((0, 16, 16), bool)
    result = proposer().evaluate(masks)
    assert not result.components
    assert result.top_diagnostics[0]["status"] == "no_top_plus_side_support"


def test_components_are_six_connected_and_diagonal_thin_body_can_fragment():
    grid = proposer()
    result = grid.evaluate(masks_for_points([[1, 1, 1], [3, 3, 3]], top_rows=[0, 0]))
    assert len(result.components) == 2
    assert all(c.voxel_count == 1 for c in result.components)
    assert result.diagnostics["potential_thin_body_fragmentation"]
    face = grid.evaluate(masks_for_points([[1, 1, 1], [3, 1, 1]], top_rows=[0, 0]))
    assert len(face.components) == 1 and face.components[0].voxel_count == 2


def test_one_top_instance_can_return_multiple_volume_components():
    result = proposer().evaluate(masks_for_points([[1, 1, 1], [9, 9, 9]], top_rows=[0, 0]))
    assert len(result.components) == 2
    assert [c.top_row for c in result.components] == [0, 0]
    assert result.top_diagnostics[0]["component_count"] == 2
    assert result.diagnostics["components_are_fish"] is False


def test_single_cell_is_retained_with_degenerate_pca_not_invented_orientation():
    part = proposer().evaluate(masks_for_points([[1, 1, 1]])).components[0]
    assert part.pca_rank == 0 and part.pca_degenerate and not part.principal_axis_defined
    np.testing.assert_array_equal(part.pca_eigenvalues_mm2, [0, 0, 0])
    np.testing.assert_array_equal(part.pca_extent_mm, [2, 2, 2])


def test_repeated_full_rank_eigenvalues_are_also_marked_degenerate():
    grid = proposer(shape=(2, 2, 2))
    masks = {view: np.ones((1, 16, 16), bool) for view in VIEWS}
    part = grid.evaluate(masks).components[0]
    assert part.pca_rank == 3 and part.pca_degenerate and not part.principal_axis_defined
    np.testing.assert_array_equal(part.pca_eigenvalues_mm2, [1, 1, 1])


def test_translation_moves_centroid_preserving_shape_and_candidate_indices():
    masks = masks_for_points([[1, 3, 3], [3, 3, 3], [5, 3, 3]], top_rows=[0, 0, 0])
    first = proposer().evaluate(masks).components[0]
    shift = [10, -4, 18]
    second = proposer(shift=shift).evaluate(masks).components[0]
    np.testing.assert_allclose(second.centroid_xyz, first.centroid_xyz + shift)
    for key in ("voxel_indices", "pca_axes", "pca_eigenvalues_mm2", "pca_extent_mm"):
        np.testing.assert_array_equal(getattr(first, key), getattr(second, key))


def test_pool_round_robin_includes_each_top_before_second_and_is_row_invariant():
    grid = proposer(shape=(15, 1, 1), image_shape=(32, 32), max_total_components=4)
    points = [[x, 1, 1] for x in (1, 5, 9, 13, 17, 21, 25, 29)]
    masks = masks_for_points(points, image_shape=(32, 32), top_rows=[0, 0, 0, 0, 1, 2, 2, 2])
    first = grid.evaluate(masks)
    assert len(first.components) == 4 and {c.top_row for c in first.components} == {0, 1, 2}
    assert first.diagnostics["omitted_by_reason"] == {
        "min_component_voxels": 0, "per_top_component_limit": 1, "global_component_limit": 3,
    }
    reorder = [2, 0, 1]
    altered = {view: masks[view][reorder] if v == 0 else masks[view][::-1]
               for v, view in enumerate(VIEWS)}
    second = grid.evaluate(altered)
    assert signature(first) == signature(second)
    by_geometry = {tuple(c.centroid_xyz): c.top_row for c in first.components}
    assert all(reorder[c.top_row] == by_geometry[tuple(c.centroid_xyz)] for c in second.components)
    assert sum(x["omitted_components"] for x in first.top_diagnostics) == 4


def test_minimum_size_rejections_are_counted_and_reasoned():
    result = proposer(min_component_voxels=2).evaluate(masks_for_points([[1, 1, 1], [9, 9, 9]]))
    assert not result.components
    assert result.diagnostics["total_components"] == 2
    assert result.diagnostics["omitted_by_reason"]["min_component_voxels"] == 2
    assert all(x["status"] == "all_components_below_minimum" for x in result.top_diagnostics)


def test_duplicate_top_rows_remain_duplicate_hypotheses_and_input_unchanged():
    masks = masks_for_points([[1, 1, 1], [1, 1, 1]])
    original = {key: value.copy() for key, value in masks.items()}
    value = proposer()
    result = value.evaluate(masks)
    assert len(result.components) == 2
    np.testing.assert_array_equal(
        result.components[0].centroid_xyz, result.components[1].centroid_xyz
    )
    for view in VIEWS:
        np.testing.assert_array_equal(masks[view], original[view])
        assert masks[view].flags.writeable
    assert not result.components[0].voxel_indices.flags.writeable


def test_three_view_ghost_is_not_declared_to_be_an_animal():
    value = proposer()
    physical = np.array([[1, 1, 3], [1, 3, 1], [3, 1, 1]])
    result = value.evaluate(masks_for_points(physical))
    ghost = np.flatnonzero(np.all(value.xyz == [1, 1, 1], axis=1))[0]
    containing = [c for c in result.components if c.top_row == 0 and ghost in c.voxel_indices]
    assert len(containing) == 1 and containing[0].three_view_voxels >= 1
    assert not result.diagnostics["components_are_fish"]
    assert not hasattr(containing[0], "identity")


@pytest.mark.parametrize("options", [
    {"spacing_mm": 1.0}, {"support_threshold": 0}, {"support_threshold": np.nan},
    {"max_total_components": 13}, {"max_components_per_top": 4},
    {"min_component_voxels": 0}, {"pca_degeneracy_rtol": 0},
])
def test_invalid_configuration_rejected(options):
    with pytest.raises(ValueError):
        module.InstanceVolumeConfig(**options)


def test_native_mask_shape_dtype_and_view_keys_are_required():
    grid = proposer()
    masks = masks_for_points([[1, 1, 1]])
    wrong = dict(masks)
    wrong[VIEWS[0]] = masks[VIEWS[0]].astype(np.uint8)
    with pytest.raises(ValueError, match="bool"):
        grid.evaluate(wrong)
    wrong[VIEWS[0]] = np.zeros((1, 8, 8), bool)
    with pytest.raises(ValueError, match="native"):
        grid.evaluate(wrong)
    with pytest.raises(ValueError, match="all three"):
        grid.evaluate({VIEWS[0]: masks[VIEWS[0]]})


def test_xyz_order_and_fixed_two_mm_spacing_are_validated():
    grid = proposer()
    shuffled = grid.xyz.copy()
    shuffled[[0, 1]] = shuffled[[1, 0]]
    with pytest.raises(ValueError, match="regular 2-mm"):
        module.InstanceVolumeProposer(shuffled, grid.projected_uv, grid.grid_shape)
    with pytest.raises(ValueError, match="regular 2-mm"):
        module.InstanceVolumeProposer(grid.xyz / 2, grid.projected_uv, grid.grid_shape)
    with pytest.raises(ValueError, match="N,3,2"):
        module.InstanceVolumeProposer(
            grid.xyz, grid.projected_uv.transpose(1, 0, 2), grid.grid_shape
        )

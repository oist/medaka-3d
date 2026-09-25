"""Independent adversarial checks; spatial pools are not complete fish hypotheses."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from medaka3d.types import VIEWS

PATH = Path(__file__).resolve().parents[1] / "labs/20260909_three_body/mask_grid.py"
SPEC = importlib.util.spec_from_file_location("_mask_grid_independent_audit", PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class Camera:
    def __init__(self, axes, unavailable=False):
        self.axes, self.unavailable = axes, unavailable

    def project(self, xyz):
        uv = np.asarray(xyz)[..., self.axes].copy()
        return np.full_like(uv, np.nan) if self.unavailable else uv


class Rig:
    views = VIEWS
    world_units = "mm"

    def __init__(self):
        self.cameras = dict(zip(VIEWS, map(Camera, ((0, 1), (0, 2), (1, 2))), strict=True))

    def camera(self, view):
        return self.cameras[view]


def make_grid(**kwargs):
    return module.MaskGrid(
        Rig(),
        module.MaskGridConfig(
            bounds_mm=((0, 24), (0, 24), (0, 24)),
            image_shape_hw=(24, 24),
            **kwargs,
        ),
    )


def masks_at_points(points):
    masks = {}
    for view, axes in zip(VIEWS, ((0, 1), (0, 2), (1, 2)), strict=True):
        image = np.zeros((1, 24, 24), bool)
        for point in points:
            u, v = np.asarray(point)[list(axes)]
            image[0, v, u] = True
        masks[view] = image
    return masks


def test_global_48_pool_can_omit_an_isolated_equally_supported_region():
    """A passing limitation test: NMS provides spacing, not component coverage."""
    value = make_grid()
    isolated = np.array([21, 21, 21])
    masks = masks_at_points([isolated])
    for image in masks.values():
        image[:, 1:12, 1:12] = True
    support = value.evaluate(masks)
    isolated_index = np.flatnonzero(np.all(value.xyz == isolated, axis=1))[0]
    assert support.samples.strict_three[isolated_index]
    assert support.ranking_score[isolated_index] == 1
    assert np.count_nonzero(support.samples.at_least_two) == 6**3 + 1
    pool = value.candidate_centers(support)
    assert len(pool.xyz) == 48
    assert isolated_index not in pool.grid_indices
    assert np.linalg.norm(pool.xyz - isolated, axis=1).min() > value.config.nms_radius_mm
    assert np.all(pool.xyz <= 11)
    # Removing only the large plateau restores the isolated candidate.
    isolated_pool = value.candidate_centers(value.evaluate(masks_at_points([isolated])))
    np.testing.assert_array_equal(isolated_pool.xyz, isolated[None])


def test_low_rank_isolated_region_is_not_rescued_by_nms_spacing():
    value = make_grid()
    isolated = np.array([21, 21, 21])
    masks = masks_at_points([isolated])
    for image in masks.values():
        image[:, 1:12, 1:12] = True
    masks[VIEWS[2]][0, 21, 21] = False
    support = value.evaluate(masks)
    index = np.flatnonzero(np.all(value.xyz == isolated, axis=1))[0]
    assert support.samples.third_view_contradiction[index]
    assert support.ranking_score[index] == pytest.approx(2 / 3)
    assert index not in value.candidate_centers(support).grid_indices


def test_soft_healthy_weight_can_become_effective_exclusion_in_a_finite_pool():
    value = make_grid(max_candidates=1)
    first, second = np.array([1, 1, 1]), np.array([21, 21, 21])
    masks = masks_at_points([first, second])
    original = value.evaluate(masks)
    suppressed = value.evaluate(masks, np.array([[first, first]], float))
    np.testing.assert_array_equal(original.samples.view_support, suppressed.samples.view_support)
    assert suppressed.novelty_weight[0] == pytest.approx(0.2)
    np.testing.assert_array_equal(value.candidate_centers(original).xyz, first[None])
    np.testing.assert_array_equal(value.candidate_centers(suppressed).xyz, second[None])
    # Neither point is ground truth: the supplied "healthy" path can be wrong.


def test_two_mm_grid_can_miss_a_point_supported_by_all_three_native_masks():
    value = make_grid()
    point = np.array([[2, 2, 2]])
    masks = masks_at_points(point)
    assert value.sample_points(point, masks).strict_three[0]
    support = value.evaluate(masks)
    assert not support.samples.at_least_two.any()
    assert len(value.candidate_centers(support).xyz) == 0


def test_empty_view_policy_differs_from_informative_view_with_local_background():
    value = make_grid()
    point = np.array([[1, 1, 1]])
    masks = masks_at_points(point)
    masks[VIEWS[2]][:] = False
    empty = value.sample_points(point, masks)
    assert empty.view_available.all()
    assert empty.known_view_count[0] == 2
    assert not empty.third_view_contradiction[0]
    assert empty.mean_known_support[0] == 1
    masks[VIEWS[2]][0, 22, 22] = True
    unrelated_detection = value.sample_points(point, masks)
    assert unrelated_detection.known_view_count[0] == 3
    assert unrelated_detection.third_view_contradiction[0]
    assert unrelated_detection.mean_known_support[0] == pytest.approx(2 / 3)


def test_nonfinite_camera_projection_is_unknown_and_cannot_supply_a_third_vote():
    value = make_grid()
    value.rig.camera(VIEWS[1]).unavailable = True
    point = np.array([[1, 3, 5]])
    samples = value.sample_points(point, masks_at_points(point))
    np.testing.assert_array_equal(samples.view_support[0, [0, 2]], [1, 1])
    assert np.isnan(samples.view_support[0, 1])
    assert samples.at_least_two[0]
    assert not samples.strict_three[0]
    assert not samples.third_view_contradiction[0]
    # Available images and geometrically known samples are separate properties.
    assert samples.view_available.all()

"""Focused CPU checks for the new observation-only comparison."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import measure_top_contrast as d


def test_ray_forward_parallel_inside_outside_tangent_invalid():
    origin = np.array(
        [
            [0.5, 0.5, 0.5],
            [-1, 0, 0.5],
            [-1, 2, 0.5],
            [2, 0.5, 0.5],
            [1, 0.5, 0.5],
            [np.nan, 0, 0],
            [0.5, 0.5, 0.5],
        ]
    )
    direction = np.array(
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]]
    )
    hit, valid = d.ray_box_intersection(origin, direction, [0, 0, 0], [1, 1, 1])
    np.testing.assert_array_equal(hit, [True, True, False, False, True, False, False])
    np.testing.assert_array_equal(valid, [True] * 5 + [False, False])


def test_small_nonzero_direction_is_not_made_parallel():
    hit, valid = d.ray_box_intersection([[-1, 0.5, 0.5]], [[1e-14, 0, 0]], [0, 0, 0], [1, 1, 1])
    assert hit[0] and valid[0]


@pytest.mark.parametrize(
    "lower,upper", [([0, 0], [1, 1, 1]), ([0, 0, 0], [0, 1, 1]), ([np.nan, 0, 0], [1, 1, 1])]
)
def test_invalid_box_rejected(lower, upper):
    with pytest.raises(ValueError):
        d.ray_box_intersection([[0, 0, 0]], [[1, 0, 0]], lower, upper)


def test_frozen_contrast_both_signs_global_shift_and_six_pixel_gate():
    background = np.full((60, 60, 3), 80, np.uint8)
    image = background + 7
    image[5:7, 5:8, 0] += 12  # six pixels exactly
    image[20:22, 20:23, 1] -= 12
    image[30, 30:35, 2] += 12  # five pixels removed
    image[40:43, 40:43, 0] += 11  # under threshold
    masks = np.zeros((2, 60, 60), bool)
    masks[0, 5:7, 5:8] = True
    masks[1, 5:7, 5:8] = True  # duplicate D2 masks must not double union mass
    box = {key: np.ones((60, 60), bool) for key in ("engineering", "physical", "valid")}
    stats, arrays = d.measure(image, background, masks, box)
    assert stats["median_channel_offset_bgr"] == [7, 7, 7]
    assert stats["contrast"]["total"] == 12
    assert stats["area_gate_removed_pixels"] == 5
    assert stats["retained_contrast_component_count"] == 2
    assert stats["d2_union"]["total"] == 6
    assert stats["intersection"]["total"] == 6
    assert stats["contrast_only"]["total"] == 6
    assert stats["positive_difference_pixels"] == stats["negative_difference_pixels"] == 6
    np.testing.assert_array_equal(arrays["component_instance_overlap_pixels"], [[6, 6], [0, 0]])


def test_empty_detector_and_box_partition_remain_separate():
    image = np.zeros((20, 20, 3), np.uint8)
    image[5:7, 5:8] = 30
    box = {key: np.zeros((20, 20), bool) for key in ("engineering", "physical", "valid")}
    box["valid"][5:7, 5:7] = True
    box["engineering"][5:7, 5] = True
    stats, arrays = d.measure(image, np.zeros_like(image), np.zeros((0, 20, 20), bool), box)
    assert stats["contrast"] == dict(
        total=6, engineering_box=2, physical_box=0, outside_engineering_box=2, invalid_ray=2
    )
    assert stats["d2_instance_count"] == 0
    assert stats["contrast_only"]["total"] == 6
    assert arrays["component_instance_overlap_pixels"].shape == (1, 0)


def test_instance_permutation_only_reorders_instance_columns():
    image = np.zeros((30, 30, 3), np.uint8)
    image[4:7, 4:7] = 40
    image[15:18, 15:18] = 40
    masks = np.zeros((2, 30, 30), bool)
    masks[0, 4:7, 4:7] = True
    masks[1, 15:18, 15:18] = True
    box = {key: np.ones((30, 30), bool) for key in ("engineering", "physical", "valid")}
    first, a = d.measure(image, np.zeros_like(image), masks, box)
    second, b = d.measure(image, np.zeros_like(image), masks[::-1], box)
    assert first["contrast"] == second["contrast"]
    assert first["intersection"] == second["intersection"]
    np.testing.assert_array_equal(
        a["component_instance_overlap_pixels"][:, ::-1], b["component_instance_overlap_pixels"]
    )

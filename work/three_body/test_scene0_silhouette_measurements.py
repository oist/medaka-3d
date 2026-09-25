"""Analytic checks for interpretation and accounting in the diagnostic metrics."""

import numpy as np
from measure_scene0_silhouette import (
    VIEWS,
    FrameObservation,
    fixed_sample_metrics,
    instance_rows,
    node_membership,
    sample_pixels,
    union_and_unique,
)


def test_unique_is_exact_marginal_union_and_removal_delta_sign():
    alpha = np.array([[[0.8, 0.7, 0.0], [0.0, 0.0, 0.6]]])
    union, unique = union_and_unique(alpha)
    np.testing.assert_allclose(union, [[0.94, 0.6]])
    np.testing.assert_allclose(unique, [[[0.24, 0.14, 0.0], [0.0, 0.0, 0.6]]])
    for fish in range(3):
        without = 1 - np.prod(1 - alpha[..., [i for i in range(3) if i != fish]], axis=-1)
        np.testing.assert_allclose(union - without, unique[..., fish], atol=2e-16)
    visible = np.stack(
        [
            alpha[..., 0],
            (1 - alpha[..., 0]) * alpha[..., 1],
            (1 - alpha[..., 0]) * (1 - alpha[..., 1]) * alpha[..., 2],
        ],
        axis=-1,
    )
    ownership = visible / union[..., None]
    result = fixed_sample_metrics(
        alpha, visible, ownership, np.ones((1, 2), bool), np.array([[1.0, 0.0]]), np.ones((1, 2))
    )
    np.testing.assert_allclose(result["remove_body_delta_loss_by_view"][0], [0.0864, 0.0364, -0.36])


def test_pair_overlap_and_unexplained_other_instance_can_coexist():
    masks = np.zeros((2, 1, 2), bool)
    masks[0, 0, 0] = masks[1, 0, 1] = True
    alpha = np.array([[0.8, 0.7, 0.0], [0.0, 0.0, 0.0]])
    visible = np.array([[0.8, 0.14, 0.0], [0.0, 0.0, 0.0]])
    rows, iou = instance_rows(masks, np.array([0, 1]), alpha, visible, np.ones(2, bool))
    np.testing.assert_allclose(rows[0]["pair_shared_mass_px"], [0.56, 0.0, 0.0])
    assert rows[1]["unexplained_mass_px"] == 1
    np.testing.assert_array_equal(iou, np.eye(2))
    # Duplicate detector masks are measured separately; no 2-fish interpretation.
    duplicated = masks[[0, 0]]
    rows, iou = instance_rows(duplicated, np.array([0]), alpha[:1], visible[:1], np.ones(1, bool))
    np.testing.assert_array_equal(iou, np.ones((2, 2)))
    assert rows[0]["union_mean"] == rows[1]["union_mean"]


def test_invalid_ray_is_excluded_from_unexplained_mass():
    masks = np.ones((1, 1, 2), bool)
    rows, _ = instance_rows(
        masks, np.array([0, 1]), np.zeros((2, 3)), np.zeros((2, 3)), np.array([True, False])
    )
    assert rows[0]["native_area_px"] == 2 and rows[0]["valid_ray_area_px"] == 1
    assert rows[0]["invalid_ray_area_px"] == 1 and rows[0]["unexplained_mass_px"] == 1


def test_bilinear_membership_distinguishes_outside_image():
    masks = np.zeros((1, 2, 2), bool)
    masks[0, 0, 0] = True
    uv = np.zeros((3, 3, 2))
    uv[0] = [[0.5, 0.5], [0.0, 0.0], [-0.01, 0.0]]
    membership = node_membership(masks, uv)
    np.testing.assert_allclose(membership["bilinear"][0, 0, :2], [0.25, 1.0])
    assert np.isnan(membership["bilinear"][0, 0, 2])
    assert not membership["node_inside_image"][0, 2]


def test_detectron2_sampling_ignores_raw_content_but_empty_remains_neutral():
    masks = np.zeros((1, 8, 8), bool)
    masks[0, 3:5, 3:5] = True
    base = FrameObservation(
        79372,
        {},
        dict.fromkeys(VIEWS, np.zeros((8, 8, 3), np.uint8)),
        dict.fromkeys(VIEWS, masks),
        {},
    )
    changed = FrameObservation(
        79372, {}, dict.fromkeys(VIEWS, np.full((8, 8, 3), 255, np.uint8)), base.masks, {}
    )
    a, b = sample_pixels(base, "detectron2", 3, 3, 3), sample_pixels(changed, "detectron2", 3, 3, 3)
    for view in VIEWS:
        np.testing.assert_array_equal(a.pixels[view], b.pixels[view])
    np.testing.assert_array_equal(a.target, b.target)
    np.testing.assert_array_equal(a.weights, b.weights)
    empty = FrameObservation(
        79372, {}, base.images_bgr, dict.fromkeys(VIEWS, np.empty((0, 8, 8), bool)), {}
    )
    assert not sample_pixels(empty, "detectron2", 3, 3, 3).weights.any()

"""Synthetic native images and vectors only; no model, real data or GPU."""

import copy

import dino_mask_partial_correspondence as d
import numpy as np
import pytest


def geometry(count, eligible=None):
    crop = np.column_stack((np.arange(count) + 0.25, np.arange(count) + 0.5))
    return dict(
        token_eligible=np.ones(count, bool) if eligible is None else np.asarray(eligible, bool),
        token_center_crop_xy=crop,
        token_center_native_xy=crop * 3 + [100, 200],
    )


def correspondence(left, right, left_geometry=None, right_geometry=None):
    left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
    return d.token_correspondence(
        left,
        right,
        geometry(len(left)) if left_geometry is None else left_geometry,
        geometry(len(right)) if right_geometry is None else right_geometry,
    )


def test_square_odd_spare_pixel_padding_and_coordinate_round_trip():
    image = np.full((10, 10, 3), [30, 60, 90], np.uint8)
    mask = np.zeros((10, 10), bool)
    mask[2:8, :3] = True
    result = d.prepare_mask_crop(image, mask)
    np.testing.assert_array_equal(result["bbox_xyxy"], [0, 2, 3, 8])
    np.testing.assert_array_equal(result["square_xyxy"], [-1, 2, 5, 8])
    np.testing.assert_array_equal(result["source_xyxy"], [0, 2, 5, 8])
    np.testing.assert_array_equal(result["padding_ltrb"], [1, 0, 0, 0])
    assert result["square_side"] == 6
    np.testing.assert_allclose(result["raw_rgb"][0, 0], d.IMAGENET_MEAN, atol=3e-8)
    assert not result["resized_native_region"][0, 0]
    assert result["resized_native_region"][0, -1]
    points = np.array([[-1.0, 2, 1], [4, 7, 1], [1.25, 5.75, 1]])
    mapped = points @ result["native_to_crop"].T
    np.testing.assert_allclose(mapped[0, :2], [224 / 12 - 0.5] * 2)
    np.testing.assert_allclose(mapped @ result["crop_to_native"].T, points, atol=1e-14)
    centers = np.column_stack((result["token_center_crop_xy"], np.ones(256)))
    np.testing.assert_allclose(
        (centers @ result["crop_to_native"].T)[:, :2], result["token_center_native_xy"]
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_background_is_replaced_before_resize_with_native_foreground_identical(
    monkeypatch, transpose
):
    image = np.full((8, 8, 3), [255, 100, 0], np.uint8)
    mask = np.zeros((8, 8), bool)
    mask[1:7, 1:4] = True
    if transpose:
        mask, image = mask.T.copy(), image.transpose(1, 0, 2).copy()
    before_image, before_mask = image.copy(), mask.copy()
    native_inputs = []
    original_resize = d.cv2.resize

    def record_resize(source, size, interpolation):
        native_inputs.append(source.copy())
        assert source.dtype == np.float32
        assert size == (224, 224)
        assert interpolation == d.cv2.INTER_LINEAR
        return original_resize(source, size, interpolation=interpolation)

    monkeypatch.setattr(d.cv2, "resize", record_resize)
    result = d.prepare_mask_crop(image, mask)
    assert len(native_inputs) == 2
    raw, masked = native_inputs
    sx, sy, ex, ey = result["square_xyxy"]
    native_mask = mask[sy:ey, sx:ex]
    np.testing.assert_array_equal(raw[native_mask], masked[native_mask])
    np.testing.assert_array_equal(
        raw[native_mask], image[mask].astype(np.float32) / np.float32(255)
    )
    expected_background = np.broadcast_to(
        np.asarray(d.IMAGENET_MEAN, np.float32), masked[~native_mask].shape
    )
    np.testing.assert_array_equal(masked[~native_mask], expected_background)
    assert np.any(raw[~native_mask] != masked[~native_mask])
    # Upsampled pixels whose nearest mask sample is inside can still mix exterior RGB.
    differences = np.any(result["raw_rgb"] != result["masked_rgb"], axis=2)
    assert np.any(differences & result["resized_mask"])
    assert np.any(~differences & result["resized_mask"])
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(mask, before_mask)


def test_model_input_is_float32_rgb_chw_and_channel_normalization_is_explicit():
    image = np.full((32, 32, 3), [255, 0, 128], np.uint8)
    mask = np.ones((32, 32), bool)
    result = d.prepare_mask_crop(image, mask)
    expected = (
        np.array([1, 0, np.float32(128) / 255], np.float32) - d.IMAGENET_MEAN
    ) / d.IMAGENET_STD
    for key in ("raw_normalized_chw", "masked_normalized_chw"):
        assert result[key].shape == (3, 224, 224)
        assert result[key].dtype == np.float32
        assert result[key].flags.c_contiguous
        np.testing.assert_allclose(result[key][:, 100, 100], expected, atol=3e-7)
    np.testing.assert_array_equal(result["raw_rgb"], result["masked_rgb"])
    result["raw_rgb"][0, 0] = 0
    np.testing.assert_array_equal(image[0, 0], [255, 0, 128])


def test_empty_and_thin_masks_keep_distinct_crop_and_missing_token_states():
    image = np.zeros((9, 9, 3), np.uint8)
    empty = np.zeros((9, 9), bool)
    thin = empty.copy()
    thin[4, 2:7] = True
    absent, visible = d.prepare_mask_crop(image, empty), d.prepare_mask_crop(image, thin)
    assert absent["crop_status"] == "empty_mask"
    assert visible["crop_status"] == "ready"
    assert absent["mask_pixels"] == 0 and visible["mask_pixels"] == 5
    for result in (absent, visible):
        assert result["token_status"] == "no_core_tokens"
        assert result["eligible_token_count"] == result["core_pixels"] == 0
        assert result["token_eligible"].shape == (256,)
        assert not result["token_eligible"].any()
    assert np.isnan(absent["raw_normalized_chw"]).all()
    assert np.isnan(absent["token_center_native_xy"]).all()
    np.testing.assert_array_equal(absent["bbox_xyxy"], [-1] * 4)
    assert np.isfinite(visible["raw_normalized_chw"]).all()
    assert np.isfinite(visible["token_center_native_xy"]).all()


def test_native_ellipse_erosion_keeps_small_cores_and_zero_border():
    mask = np.ones((7, 7), bool)
    expected = np.zeros((7, 7), bool)
    expected[2:5, 2:5] = True
    np.testing.assert_array_equal(d.eroded_core(mask), expected)
    # Nine pixels are retained despite the old color descriptor's minimum of twelve.
    result = d.prepare_mask_crop(np.zeros((7, 7, 3), np.uint8), mask)
    assert result["core_pixels"] == 9
    assert result["eligible_token_count"] > 0
    assert result["eligible_token_count"] < 256


def test_patch_geometry_is_row_major_and_covers_exact_cells():
    result = d.token_geometry(np.ones((300, 300), bool), np.array([10, 20, 234, 244]))
    np.testing.assert_array_equal(result["token_index"], np.arange(256))
    np.testing.assert_array_equal(
        result["token_center_crop_xy"][[0, 1, 16, 255]],
        [[6.5, 6.5], [20.5, 6.5], [6.5, 20.5], [216.5, 216.5]],
    )
    np.testing.assert_array_equal(
        result["token_center_native_xy"][[0, 255]], [[16.5, 26.5], [226.5, 236.5]]
    )
    np.testing.assert_array_equal(
        result["token_native_xyxy"][[0, 255]], [[10, 20, 24, 34], [220, 230, 234, 244]]
    )
    assert np.all(result["token_footprint_pixels"] == 196)
    assert result["token_eligible"].all()


def test_full_native_footprint_detects_a_hole_skipped_by_center_sampling():
    core = np.ones((512, 512), bool)
    core[1, 1] = False
    result = d.token_geometry(core, [0, 0, 512, 512])
    center = result["token_center_native_xy"][0]
    assert core[int(np.rint(center[1])), int(np.rint(center[0]))]
    assert result["token_footprint_pixels"][0] == 1024
    assert result["token_core_pixels"][0] == 1023
    assert not result["token_eligible"][0]
    assert result["token_eligible"].sum() == 255


def test_fractional_footprint_edges_and_padding_do_not_drop_partial_pixels():
    core = np.ones((20, 20), bool)
    result = d.token_geometry(core, [-1, 0, 16, 17])
    np.testing.assert_array_equal(
        result["token_native_xyxy"][:3], [[-1, 0, 1, 2], [0, 0, 2, 2], [1, 0, 3, 2]]
    )
    np.testing.assert_array_equal(result["token_native_xyxy"][-1], [14, 15, 16, 17])
    assert not result["token_eligible"].reshape(16, 16)[:, 0].any()
    assert result["token_eligible"].reshape(16, 16)[:, 1:].all()
    assert np.all(result["token_footprint_pixels"] == 4)


@pytest.mark.parametrize(
    "image, mask, message",
    [
        (np.zeros((4, 4, 3), np.float32), np.ones((4, 4), bool), "uint8 RGB"),
        (np.zeros((4, 4, 4), np.uint8), np.ones((4, 4), bool), "uint8 RGB"),
        (np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.uint8), "bool mask"),
        (np.zeros((4, 4, 3), np.uint8), np.ones((3, 4), bool), "bool mask"),
    ],
)
def test_invalid_native_input_is_rejected(image, mask, message):
    with pytest.raises(ValueError, match=message):
        d.prepare_mask_crop(image, mask)


def test_l2_normalization_has_no_tiny_threshold_and_does_not_overflow():
    values = np.array([[3, 4], [1e308, 1e308], [1e-300, -1e-300], [0, 0], [np.nan, 1], [1, np.inf]])
    before = values.copy()
    result = d.normalize_tokens(values)
    np.testing.assert_array_equal(result["input_finite"], [True] * 4 + [False] * 2)
    np.testing.assert_array_equal(result["valid"], [True] * 3 + [False] * 3)
    np.testing.assert_allclose(
        result["normalized"][:3], [[0.6, 0.8], [2**-0.5, 2**-0.5], [2**-0.5, -(2**-0.5)]]
    )
    assert result["norm"][0] == 5
    assert result["norm"][2] > 0 and result["norm"][3] == 0
    assert np.isnan(result["normalized"][3:]).all()
    np.testing.assert_array_equal(values, before)
    huge = d.normalize_tokens(np.full((1, 4), 1e308))
    assert huge["valid"][0] and np.isinf(huge["norm"][0])
    np.testing.assert_array_equal(huge["normalized"], [[0.5] * 4])


def test_all_cosine_pairs_and_every_bidirectional_tie_are_preserved():
    result = correspondence([[1, 0], [1, 0], [0, 1]], [[1, 0], [0, 1], [0, 1]])
    np.testing.assert_array_equal(result["distances"], [[0, 1, 1], [0, 1, 1], [1, 0, 0]])
    np.testing.assert_array_equal(
        result["correspondences"]["pairs"], [[0, 0], [1, 0], [2, 1], [2, 2]]
    )
    assert result["correspondences"]["mutual"].all()
    assert result["finite_distance_distribution"]["count"] == 9
    assert result["finite_distance_distribution"]["median"] == 1
    assert result["mutual_distance_distribution"]["count"] == 4
    for side in ("left", "right"):
        assert result[side]["matched_count"] == 3
        assert result[side]["matched_fraction"] == 1
        assert result[side]["tie_fraction"] == pytest.approx(1 / 3)
    pairs = result["correspondences"]["pairs"]
    np.testing.assert_array_equal(
        result["correspondences"]["left_native_xy"],
        geometry(3)["token_center_native_xy"][pairs[:, 0]],
    )
    np.testing.assert_array_equal(
        result["correspondences"]["right_crop_xy"], geometry(3)["token_center_crop_xy"][pairs[:, 1]]
    )


def test_directed_union_retains_unreciprocated_matches_and_asymmetric_coverage():
    result = correspondence([[1, 0], [2, 1]], [[1, 0]])
    np.testing.assert_array_equal(result["correspondences"]["pairs"], [[0, 0], [1, 0]])
    np.testing.assert_array_equal(result["correspondences"]["forward"], [True, True])
    np.testing.assert_array_equal(result["correspondences"]["reverse"], [True, False])
    np.testing.assert_array_equal(result["correspondences"]["mutual"], [True, False])
    assert result["left"]["matched_fraction"] == 0.5
    assert result["right"]["matched_fraction"] == 1
    assert result["finite_distance_distribution"]["count"] == 2


def test_exact_ties_are_not_expanded_with_a_numerical_tolerance():
    result = correspondence([[1, 0]], [[1, 0], [1, 1e-6]])
    assert 0 < result["distances"][0, 1] < 1e-12
    np.testing.assert_array_equal(result["forward_nn"], [[True, False]])
    np.testing.assert_array_equal(result["reverse_nn"], [[True, True]])
    assert result["left"]["tie_fraction"] == 0


def test_large_distance_nearest_neighbors_are_kept_without_acceptance_threshold():
    result = correspondence([[1, 0]], [[-1, 0]])
    np.testing.assert_array_equal(result["distances"], [[2]])
    assert result["mutual_nn"][0, 0]
    assert result["left"]["matched_fraction"] == 1


def test_invalid_selected_tokens_preserve_nan_rows_and_coverage_denominator():
    left_geometry = geometry(4, [True, True, True, False])
    left_geometry["token_center_native_xy"][2, 0] = np.nan
    result = correspondence([[1, 0], [0, 0], [0, 1], [1, 0]], [[1, 0]], left_geometry)
    assert result["status"] == "partial_invalid_tokens"
    np.testing.assert_array_equal(result["finite_pair"], [[True], [False], [False], [False]])
    assert np.isnan(result["distances"][1:]).all()
    assert result["left"]["eligible_count"] == 3
    assert result["left"]["valid_count"] == result["left"]["matched_count"] == 1
    assert result["left"]["invalid_eligible_count"] == 2
    assert result["left"]["matched_fraction"] == pytest.approx(1 / 3)
    assert result["left"]["matched_valid_fraction"] == 1
    np.testing.assert_array_equal(result["left"]["coordinate_finite"], [True, True, False, True])


@pytest.mark.parametrize("mode", ["no_eligible", "zero", "nonfinite", "empty_array"])
def test_no_comparable_tokens_are_missing_rates_not_zero_scores(mode):
    left = [[1, 0]]
    left_geometry = geometry(1)
    expected_status = "no_valid_tokens"
    if mode == "no_eligible":
        left_geometry["token_eligible"][:] = False
        expected_status = "no_eligible_tokens"
    elif mode == "zero":
        left = [[0, 0]]
    elif mode == "nonfinite":
        left = [[1, np.inf]]
    else:
        left, left_geometry = np.empty((0, 2)), geometry(0)
        expected_status = "no_eligible_tokens"
    result = correspondence(left, [[1, 0]], left_geometry)
    assert result["status"] == expected_status
    assert not result["comparable"]
    assert not result["finite_pair"].any()
    assert result["distances"].shape == (len(left), 1)
    assert np.isnan(result["distances"]).all()
    assert result["correspondences"]["pairs"].shape == (0, 2)
    for side in ("left", "right"):
        assert result[side]["matched_count"] == 0
        for key in ("matched_fraction", "matched_valid_fraction", "tie_fraction"):
            assert np.isnan(result[side][key])
    for name in ("finite_distance_distribution", "mutual_distance_distribution"):
        assert result[name]["count"] == 0
        assert all(np.isnan(value) for key, value in result[name].items() if key != "count")


def test_correspondence_does_not_mutate_tokens_or_geometry_and_swaps_symmetrically():
    left = np.array([[2, 1], [1, 0]], np.float32)
    right = np.array([[1, 0], [0, 1]], np.float64)
    lg, rg = geometry(2), geometry(2)
    before = copy.deepcopy((left, right, lg, rg))
    result = d.token_correspondence(left, right, lg, rg)
    reverse = d.token_correspondence(right, left, rg, lg)
    np.testing.assert_array_equal(result["distances"], reverse["distances"].T)
    np.testing.assert_array_equal(result["forward_nn"], reverse["reverse_nn"].T)
    np.testing.assert_array_equal(result["mutual_nn"], reverse["mutual_nn"].T)
    for current, previous in zip((left, right), before[:2], strict=True):
        np.testing.assert_array_equal(current, previous)
    for current, previous in zip((lg, rg), before[2:], strict=True):
        for key in current:
            np.testing.assert_array_equal(current[key], previous[key])
    result["left"]["eligible"][:] = False
    result["left"]["normalization"]["normalized"][:] = 0
    np.testing.assert_array_equal(lg["token_eligible"], before[2]["token_eligible"])
    np.testing.assert_array_equal(left, before[0])


def test_all_45_mask_requests_and_more_than_three_instances_are_preserved():
    counts = {(frame, view): 1 for frame in d.FRAMES for view in d.VIEWS}
    counts[d.FRAMES[0], d.VIEWS[0]] = 5
    result = d.enumerate_mask_pairs(counts)
    assert len(result) == 45
    assert len({(row["earlier_frame"], row["later_frame"], row["view"]) for row in result}) == 45
    assert sum(len(row["mask_pairs"]) for row in result) == 65
    np.testing.assert_array_equal(result[0]["mask_pairs"], [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
    assert all(row["earlier_frame"] < row["later_frame"] for row in result)
    assert all(row["status"] == "available" for row in result)
    assert counts[d.FRAMES[0], d.VIEWS[0]] == 5


def test_missing_observation_and_zero_instances_still_have_request_rows():
    counts = {(frame, view): 1 for frame in d.FRAMES for view in d.VIEWS}
    counts[d.FRAMES[0], d.VIEWS[0]] = None
    counts[d.FRAMES[0], d.VIEWS[1]] = 0
    rows = d.enumerate_mask_pairs(counts)
    assert len(rows) == 45
    assert sum(row["status"] == "missing_observation" for row in rows) == 5
    assert sum(row["status"] == "no_mask_pairs" for row in rows) == 5
    assert rows[0]["earlier_mask_count"] is None
    assert rows[1]["earlier_mask_count"] == 0
    assert rows[0]["mask_pairs"].shape == rows[1]["mask_pairs"].shape == (0, 2)


@pytest.mark.parametrize("change", ["missing", "extra", "bool", "negative", "float"])
def test_mask_request_input_cannot_silently_change_the_fixed_universe(change):
    counts = {(frame, view): 1 for frame in d.FRAMES for view in d.VIEWS}
    first = next(iter(counts))
    if change == "missing":
        del counts[first]
    elif change == "extra":
        counts[1, "other"] = 1
    else:
        counts[first] = {"bool": True, "negative": -1, "float": 1.0}[change]
    with pytest.raises(ValueError):
        d.enumerate_mask_pairs(counts)


def test_token_feature_and_geometry_contracts_reject_dimension_drift():
    with pytest.raises(ValueError, match="floating"):
        d.normalize_tokens(np.array([[1, 2]], np.int64))
    with pytest.raises(ValueError, match="feature dimensions"):
        correspondence([[1, 2]], [[1, 2, 3]])
    with pytest.raises(ValueError, match="geometry shapes"):
        correspondence([[1, 2]], [[1, 2]], geometry(2))
    with pytest.raises(ValueError, match="positive square"):
        d.token_geometry(np.ones((4, 4), bool), [0, 0, 2, 3])
    with pytest.raises(ValueError, match="integer square"):
        d.token_geometry(np.ones((4, 4), bool), [0.0, 0, 2, 2])

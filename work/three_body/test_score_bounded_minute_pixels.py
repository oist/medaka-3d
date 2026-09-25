"""Protect fixed samples, reused renders and three-way aggregation from substitutions."""

import copy

import numpy as np
import pytest
import score_bounded_minute_pixels as score


def fixture():
    poses = np.arange(54, dtype=np.float64).reshape(2, 3, 3, 3)
    sample = dict(
        pixels=np.ones((3, 256, 2), dtype=np.float64),
        target=np.zeros((3, 256)),
        weights=np.full((3, 256), 1 / 256),
        fit_pixels=np.zeros((3, 1024, 2)),
        fit_target=np.zeros((3, 1024)),
        fit_weights=np.full((3, 1024), 1 / 1024),
        fit_foreground=np.zeros((3, 1024), dtype=bool),
        reused_fitted_coordinate=np.zeros((3, 256), dtype=bool),
    )
    alpha = np.zeros((2, 3, 256))
    alpha[0] = 0.25
    alpha[1] = 0.5
    valid = np.ones((3, 256), dtype=bool)
    saved = dict(
        **copy.deepcopy(sample),
        poses=poses.copy(),
        union_alpha=alpha,
        ray_valid=valid,
        per_view_error=score.squared_error(alpha, sample["target"], sample["weights"], valid),
    )
    return sample, saved, poses


def test_exact_reuse_preserves_method_order_and_loss():
    sample, saved, poses = fixture()
    actual = score.validate_saved_frame(saved, sample, poses)
    np.testing.assert_array_equal(actual, [[0.0625] * 3, [0.25] * 3])


@pytest.mark.parametrize("fault", ["pose", "order", "pixels", "target", "weights", "loss", "alpha"])
def test_substituted_saved_assay_is_rejected(fault):
    sample, saved, poses = fixture()
    if fault == "pose":
        saved["poses"].flat[0] += 1
    elif fault == "order":
        saved["poses"] = saved["poses"][::-1].copy()
    elif fault in ("pixels", "target", "weights"):
        saved[fault].flat[0] += 1
    elif fault == "loss":
        saved["per_view_error"].flat[0] += 1
    else:
        saved["union_alpha"].flat[0] += 1
    with pytest.raises(ValueError):
        score.validate_saved_frame(saved, sample, poses)


def test_actual_overlap_is_rejected_even_if_saved_overlap_flag_is_false():
    sample, saved, poses = fixture()
    sample["pixels"][0, 0] = sample["fit_pixels"][0, 0]
    saved["pixels"] = sample["pixels"].copy()
    with pytest.raises(ValueError, match="fit pixel reused"):
        score.validate_saved_frame(saved, sample, poses)


@pytest.mark.parametrize("fault", ["nan", "negative_weight", "valid_dtype", "shape"])
def test_invalid_render_arithmetic_cannot_produce_a_summary(fault):
    sample, saved, _ = fixture()
    if fault == "nan":
        saved["union_alpha"].flat[0] = np.nan
    elif fault == "negative_weight":
        sample["weights"].flat[0] = -1
    elif fault == "valid_dtype":
        saved["ray_valid"] = saved["ray_valid"].astype(float)
    else:
        saved["union_alpha"] = saved["union_alpha"][:, :, :-1]
    with pytest.raises(ValueError):
        score.squared_error(
            saved["union_alpha"], sample["target"], sample["weights"], saved["ray_valid"]
        )


def test_three_way_summary_uses_all_sixty_frames_and_both_references():
    errors = np.empty((60, 3, 3))
    errors[:, 0] = 2
    errors[:, 1] = 1
    errors[:20, 2] = 0
    errors[20:40, 2] = 1
    errors[40:, 2] = 3
    result = score.summary_errors(errors)
    assert result["mean_error_by_method"] == dict(original=2, directional=1, bounded=4 / 3)
    first, second = (result["bounded_compared_with"][key] for key in ("original", "directional"))
    assert (first["frames_better"], first["frames_worse"], first["frames_equal"]) == (40, 20, 0)
    assert (second["frames_better"], second["frames_worse"], second["frames_equal"]) == (20, 20, 20)
    assert first["relative_reduction_percent"] == pytest.approx(100 / 3)
    assert second["relative_reduction_percent"] == pytest.approx(-100 / 3)
    with pytest.raises(ValueError):
        score.summary_errors(errors[:59])
    assert (
        score.summary_errors(np.zeros((60, 3, 3)))["bounded_compared_with"]["original"][
            "relative_reduction_percent"
        ]
        is None
    )


def test_a_seal_reference_requires_an_independently_verified_identical_input(tmp_path):
    path = str((tmp_path / "candidate.npz").resolve())
    seal = dict(inputs={path: dict(sha256="a" * 64)})
    score.bound_input(seal, path, {path: "a" * 64})
    for hashes in ({}, {path: "b" * 64}):
        with pytest.raises(ValueError):
            score.bound_input(seal, path, hashes)
    with pytest.raises(ValueError):
        score.bound_input(dict(inputs={}), path, {path: "a" * 64})

"""Four-way evidence must preserve all three old renders and fixed sample draws."""

import ast
import copy
from pathlib import Path

import numpy as np
import pytest
import score_tail_minute_pixels as score


def fixture():
    poses = np.arange(81, dtype=np.float64).reshape(3, 3, 3, 3)
    sample = dict(
        pixels=np.ones((3, 256, 2)),
        target=np.zeros((3, 256)),
        weights=np.full((3, 256), 1 / 256),
        fit_pixels=np.zeros((3, 1024, 2)),
        fit_target=np.zeros((3, 1024)),
        fit_weights=np.full((3, 1024), 1 / 1024),
        fit_foreground=np.zeros((3, 1024), dtype=bool),
        reused_fitted_coordinate=np.zeros((3, 256), dtype=bool),
    )
    alpha = np.broadcast_to(np.array([0.25, 0.5, 0.75])[:, None, None], (3, 3, 256)).copy()
    valid = np.ones((3, 256), dtype=bool)
    saved = dict(
        **copy.deepcopy(sample),
        poses=poses.copy(),
        union_alpha=alpha,
        ray_valid=valid,
        per_view_error=score.squared_error(alpha, sample["target"], sample["weights"], valid),
    )
    return sample, saved, poses


def test_reuse_checks_all_three_existing_methods():
    sample, saved, poses = fixture()
    np.testing.assert_array_equal(
        score.validate_saved_frame(saved, sample, poses), [[0.0625] * 3, [0.25] * 3, [0.5625] * 3]
    )


@pytest.mark.parametrize("method", range(3))
@pytest.mark.parametrize("field", ["poses", "union_alpha", "per_view_error"])
def test_changed_old_method_is_rejected(method, field):
    sample, saved, poses = fixture()
    saved[field][method].flat[0] += 1
    with pytest.raises(ValueError):
        score.validate_saved_frame(saved, sample, poses)


def test_old_method_order_and_missing_fourth_comparison_are_not_accepted():
    sample, saved, poses = fixture()
    saved["poses"] = saved["poses"][[0, 2, 1]]
    with pytest.raises(ValueError, match="pose differs"):
        score.validate_saved_frame(saved, sample, poses)
    with pytest.raises(ValueError, match="60x4x3"):
        score.summary_errors(np.ones((60, 3, 3)))


def test_actual_pixel_overlap_is_rejected_despite_a_false_saved_flag():
    sample, saved, poses = fixture()
    sample["pixels"][2, 13] = 0
    saved["pixels"] = sample["pixels"].copy()
    with pytest.raises(ValueError, match="fit pixel reused"):
        score.validate_saved_frame(saved, sample, poses)


def test_four_way_aggregation_compares_tail_to_each_reference_on_all_sixty_frames():
    values = np.empty((60, 4, 3))
    values[:, 0] = 4
    values[:, 1] = 2
    values[:, 2] = 1
    values[:20, 3] = 0
    values[20:40, 3] = 1
    values[40:, 3] = 3
    result = score.summary_errors(values)
    assert result["mean_error_by_method"] == dict(original=4, directional=2, bounded=1, tail=4 / 3)
    expected = dict(original=(60, 0, 0), directional=(40, 20, 0), bounded=(20, 20, 20))
    for method, counts in expected.items():
        row = result["tail_compared_with"][method]
        assert tuple(row[k] for k in ("frames_better", "frames_worse", "frames_equal")) == counts
    with pytest.raises(ValueError):
        score.summary_errors(values[:59])


def test_binding_and_pixel_arithmetic_match_the_successful_previous_wrapper():
    trees = []
    for filename in ("score_bounded_minute_pixels_v2.py", "score_tail_minute_pixels.py"):
        tree = ast.parse(Path(score.__file__).with_name(filename).read_text(encoding="utf-8"))
        trees.append(
            {
                f.name: ast.dump(f, include_attributes=False)
                for f in tree.body
                if isinstance(f, ast.FunctionDef)
            }
        )
    for name in ("squared_error", "bound_input", "shared_source_bindings"):
        assert trees[0][name] == trees[1][name]
    assert score.prior.SCORED.tolist() == [66483 + 166 * k for k in range(60)]

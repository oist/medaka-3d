"""Fixed-shape color costs cannot manufacture information for invisible slots."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load(name):
    path = Path(__file__).resolve().parents[1] / "labs/20260909_three_body" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load("pixel_template_assignment")
a = load("appearance_diagnostic")


def test_constant_color_agrees_with_old_aggregate_feature_scale():
    for color in ([0, 0, 0], [120, 120, 120], [20, 40, 100], [255, 0, 0]):
        image = np.full((20, 20, 3), color, np.uint8)
        descriptor, core = a.describe_core(image, np.ones((20, 20), bool))
        values = m.pixel_features(image[core])
        np.testing.assert_allclose(values, np.broadcast_to(descriptor, values.shape), atol=1e-7)


def test_union_is_eroded_once_and_duplicate_masks_do_not_double_pixel_mass():
    masks = np.zeros((2, 20, 20), bool)
    masks[0, 3:17, 3:10] = True
    masks[1, 3:17, 10:17] = True
    union, core = m.union_core(masks)
    assert core[10, 9] and core[10, 10]  # Joined seam survives union-then-erosion.
    _, duplicated = m.union_core(np.concatenate([masks, masks]))
    np.testing.assert_array_equal(core, duplicated)
    assert core.sum() < union.sum()


def test_weighted_cost_exact_common_denominator_and_all_six_assignments():
    features = np.zeros((2, 6))
    features[1, 0] = 1
    templates = np.zeros((3, 6))
    templates[:, 0] = [0, 0.5, 1]
    alpha = np.array([[0.8, 0.1, 0], [0, 0.2, 0.7]])
    result = m.assignment_costs(features, templates, alpha)
    expected = np.array([[0, 0.2, 0.8], [0.2, 0.075, 0.1], [0.7, 0.175, 0]])
    np.testing.assert_allclose(result["fish_template_raw_cost"], expected)
    assert result["common_denominator"] == pytest.approx(1.8)
    for index, permutation in enumerate(m.PERMUTATIONS):
        assert result["scores"][index] == pytest.approx(
            sum(expected[fish, label] for fish, label in enumerate(permutation)) / 1.8
        )


def test_zero_mass_returns_no_winner_and_invisible_pair_stays_tied():
    features, templates = np.zeros((1, 6)), np.zeros((3, 6))
    result = m.assignment_costs(features, templates, np.zeros((1, 3)))
    assert not result["scored"] and np.isnan(result["scores"]).all()
    assert np.all(result["rank_order"] == -1)
    templates[:, 0] = [0, 0.5, 1]
    result = m.assignment_costs(features, templates, np.array([[0.7, 0, 0]]))
    assert result["scores"][0] == result["scores"][1] == 0
    assert np.isnan(result["fish_template_normalized_cost"][1:]).all()


def test_identical_templates_remain_tied_and_slot_shuffle_only_reindexes_costs():
    rng = np.random.default_rng(73)
    features, alpha = rng.random((9, 6)), rng.random((9, 3)) / 3
    tied = m.assignment_costs(features, np.full((3, 6), 0.4), alpha)
    np.testing.assert_allclose(tied["scores"], tied["scores"][0], atol=1e-15)
    templates = rng.random((3, 6))
    original = m.assignment_costs(features, templates, alpha)
    shuffle = np.array([2, 0, 1])
    altered = m.assignment_costs(features, templates, alpha[:, shuffle])
    for index, permutation in enumerate(m.PERMUTATIONS):
        match = np.flatnonzero(np.all(permutation[shuffle] == m.PERMUTATIONS, axis=1))[0]
        assert altered["scores"][match] == pytest.approx(original["scores"][index])


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_render_weights_rejected(bad):
    alpha = np.zeros((1, 3))
    alpha[0, 1] = bad
    with pytest.raises(ValueError, match="alpha"):
        m.assignment_costs(np.zeros((1, 6)), np.zeros((3, 6)), alpha)


def test_empty_union_retains_empty_denominator():
    union, core = m.union_core(np.zeros((0, 20, 20), bool))
    assert not union.any() and not core.any()
    values = m.pixel_features(np.zeros((0, 3), np.uint8))
    result = m.assignment_costs(values, np.zeros((3, 6)), np.zeros((0, 3)))
    assert not result["scored"]

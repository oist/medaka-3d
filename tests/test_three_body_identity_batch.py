"""Exact floating/tie parity against the current frozen tracker method itself."""

import ast
import importlib.util
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

_LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
_SPEC = importlib.util.spec_from_file_location("three_body_identity_batch_test",
                                             _LAB / "identity_batch.py")
_BATCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BATCH)

# Compile only the frozen static method, not tracker imports or GPU machinery.
_TREE = ast.parse((_LAB / "tracker.py").read_text(encoding="utf-8"))
_CLASS = next(node for node in _TREE.body if isinstance(node, ast.ClassDef)
              and node.name == "ThreeBodyTracker")
_METHOD = next(node for node in _CLASS.body if isinstance(node, ast.FunctionDef)
               and node.name == "_different_identity")
_METHOD.decorator_list = []
_SCOPE = {"np": np, "permutations": permutations}
exec(compile(ast.Module(body=[_METHOD], type_ignores=[]), str(_LAB / "tracker.py"), "exec"),
     _SCOPE)
original = _SCOPE["_different_identity"]


def compare(first, second):
    expected = np.asarray([[original(a, b) for b in second] for a in first])
    actual = _BATCH.different_identity_batch(first, second)
    np.testing.assert_array_equal(actual, expected)
    costs = np.asarray([[
        [sum(np.linalg.norm(a[f, 1] - b[p[f], 1]) ** 2 for f in range(3))
         for p in permutations(range(3))] for b in second] for a in first])
    np.testing.assert_array_equal(_BATCH.identity_costs_batch(first, second), costs)
    return actual


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", [29, 731, 933544, 77107])
def test_random_six_by_eighteen_all_costs_and_affected_exact(dtype, seed):
    rng = np.random.default_rng(seed)
    first = rng.uniform(-6, 192, (6, 3, 3, 3)).astype(dtype)
    second = rng.uniform(-6, 192, (18, 3, 3, 3)).astype(dtype)
    compare(first, second)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_and_ulp_close_ties_preserve_original_lexical_choice(dtype):
    first = np.zeros((1, 3, 3, 3), dtype=dtype)
    first[0, :, 1] = [[-1, 0, 0], [1, 0, 0], [0, 10, 0]]
    epsilon = np.finfo(dtype).eps
    delta = np.array([0, epsilon / 4, -epsilon / 4, epsilon, -epsilon,
                      4 * epsilon, -4 * epsilon], dtype=dtype)
    second = np.zeros((len(delta), 3, 3, 3), dtype=dtype)
    second[:, :, 1] = [[0, -1, 0], [0, 1, 0], [0, 10, 0]]
    second[:, 0, 1, 0] = delta
    second[:, 1, 1, 0] = -delta
    affected = compare(first, second)
    assert not affected[0, 0].any()  # Both matchings tie; identity is lexically first.
    assert affected[0, -2, :2].all()
    assert not affected[0, -1].any()


def test_full_coincidence_and_symmetry_have_no_added_tie_breaker():
    first = np.zeros((2, 3, 3, 3), dtype=np.float64)
    second = first.copy()
    first[:, :, 0] = np.arange(18).reshape(2, 3, 3)
    second[:, :, 2] = np.arange(18).reshape(2, 3, 3)[::-1]
    assert not compare(first, second).any()


def test_common_fish_relabeling_preserves_affected_set_when_matching_is_unique():
    rng = np.random.default_rng(303)
    base = rng.uniform(0, 150, (3, 3, 3))
    all_permutations = list(permutations(range(3)))
    first = base[None]
    second = np.stack([base[list(p)] for p in all_permutations])
    expected = compare(first, second)
    for order in all_permutations:
        actual = compare(first[:, order], second[:, order])
        np.testing.assert_array_equal(actual, expected[:, :, order])


def test_only_pectoral_coordinates_enter_frozen_comparison_and_inputs_are_unchanged():
    rng = np.random.default_rng(212)
    first, second = rng.normal(size=(2, 3, 3, 3)), rng.normal(size=(4, 3, 3, 3))
    first[:, :, [0, 2]] = np.nan
    second[:, :, [0, 2]] = 1e50
    old_first, old_second = first.copy(), second.copy()
    compare(first, second)
    np.testing.assert_array_equal(first, old_first)
    np.testing.assert_array_equal(second, old_second)


def test_norm_then_scalar_power_is_preserved_at_a_known_rounding_counterexample():
    # Scalar power and ndarray multiplication differ by one ULP for this value.
    value = np.float64(1.7505074288781703)
    assert value**2 != (np.array([value]) ** 2)[0]
    first = np.zeros((1, 3, 3, 3), dtype=np.float64)
    second = first.copy()
    second[0, 0, 1, 0] = value
    compare(first, second)


def test_empty_batches_and_invalid_shapes():
    empty = np.empty((0, 3, 3, 3), dtype=np.float64)
    one = np.zeros((1, 3, 3, 3), dtype=np.float64)
    assert _BATCH.different_identity_batch(empty, one).shape == (0, 1, 3)
    assert _BATCH.different_identity_batch(one, empty).shape == (1, 0, 3)
    with pytest.raises(ValueError, match="first must have shape"):
        _BATCH.different_identity_batch(one[0], one)
    with pytest.raises(ValueError, match="float32 or float64"):
        _BATCH.different_identity_batch(one.astype(int), one)
    one[0, 0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="pectoral coordinates must be finite"):
        _BATCH.different_identity_batch(one, empty)

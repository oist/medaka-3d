"""Independent feasible-set and arithmetic checks for row-coherent controls."""

import importlib.util
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / 'labs/20260909_three_body'
SPEC = importlib.util.spec_from_file_location('instance_assignment', LAB / 'instance_assignment.py')
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def exhaustive_coherent(cost, weights, missing):
    """Independent enumerate row choices including costly rows and every node subset."""
    best = np.inf
    for rows in product(range(-1, cost.shape[1]), repeat=3):
        used = [r for r in rows if r >= 0]
        if len(used) != len(set(used)):
            continue
        values = []
        for fish, row in enumerate(rows):
            if row < 0:
                values.append(missing * weights.sum())
            else:
                values.append(min(sum(weights[k] * (cost[fish, row, k] if keep[k]
                                                    else missing) for k in range(3))
                                  for keep in product((False, True), repeat=3)))
        best = min(best, sum(values))
    return best


def test_empty_views_and_unavailable_nodes():
    result = module.assign([np.empty((3, 0, 3))] * 3, coherent=True)
    assert np.all(result.plan == -1)
    assert result.cost == pytest.approx(9.)
    inf = np.full((3, 4, 3), np.inf)
    assert module.assign([inf] * 3, coherent=False).cost == pytest.approx(9.)


def test_coherent_prevents_chimeric_body_and_cross_body_node_reuse():
    cost = np.full((3, 2, 3), 20.)
    cost[0, 0, 0] = 0.
    cost[0, 1, 1] = 0.
    cost[1, 0, 2] = 0.
    relaxed = module.assign([cost] * 3, coherent=False)
    grouped = module.assign([cost] * 3, coherent=True)
    assert relaxed.plan[0, 0].tolist() == [0, 1, -1]
    assert relaxed.plan[1, 0, 2] == 0
    assert grouped.cost > relaxed.cost
    for view in range(3):
        used = []
        for fish in range(3):
            rows = set(grouped.plan[fish, view]) - {-1}
            assert len(rows) <= 1
            used.extend(rows)
        assert len(set(used)) == len(used)


def test_outlier_node_does_not_force_bad_assignment():
    cost = np.full((3, 1, 3), np.inf)
    cost[0, 0] = [1., 2., 100.]
    result = module.assign([cost] * 3, coherent=True)
    assert result.plan[0, 0].tolist() == [0, 0, -1]


def test_bruteforce_60_random_cases_and_relaxation_lower_bound():
    rng = np.random.default_rng(9109)
    weights = np.array([1., 1., .35])
    for _ in range(60):
        costs = [rng.uniform(0, 15, (3, 2, 3)) for _ in range(3)]
        before = [c.copy() for c in costs]
        grouped = module.assign(costs, coherent=True)
        relaxed = module.assign(costs, coherent=False)
        expected = sum(exhaustive_coherent(c, weights, 9.) for c in costs) / (9 * weights.sum())
        assert grouped.cost == pytest.approx(expected, abs=1e-13)
        assert grouped.cost >= relaxed.cost - 1e-13
        for c, old in zip(costs, before, strict=True):
            np.testing.assert_array_equal(c, old)


def test_row_permutation_preserves_unique_solution_not_row_numbers():
    rng = np.random.default_rng(112)
    costs = [rng.uniform(0, 15, (3, 4, 3)) for _ in range(3)]
    permutation = np.array([2, 0, 3, 1])
    for coherent in (False, True):
        original = module.assign(costs, coherent=coherent)
        permuted = module.assign([c[:, permutation] for c in costs], coherent=coherent)
        assert original.cost == permuted.cost
        valid = permuted.plan >= 0
        mapped = permuted.plan.copy()
        mapped[valid] = permutation[mapped[valid]]
        np.testing.assert_array_equal(original.plan, mapped)


def test_row_labels_are_local_to_each_camera():
    costs = []
    for view in range(3):
        cost = np.full((3, 3, 3), 20.)
        for fish in range(3):
            cost[fish, (fish + view) % 3] = 0.
        costs.append(cost)
    result = module.assign(costs, coherent=True)
    assert result.cost == 0
    assert result.plan[0, :, 0].tolist() == [0, 1, 2]


def test_exact_tie_is_explicit_and_deterministic():
    costs = [np.zeros((3, 3, 3))] * 3
    result = module.assign(costs, coherent=True)
    assert result.optimal_view_counts == (6, 6, 6)
    np.testing.assert_array_equal(result.plan, module.assign(costs, coherent=True).plan)


def test_half_ray_clamping_and_float32_cache_match_analytic_error():
    origin = np.zeros((1, 3, 3))
    direction = np.broadcast_to([1., 0., 0.], origin.shape).copy()
    rays = SimpleNamespace(origins=(origin,) * 3, directions=(direction,) * 3)
    pose = np.broadcast_to([-2., 1., 0.], (3, 3, 3)).copy()
    costs = module.half_ray_costs(pose, rays, ray_sigma_mm=1.)
    np.testing.assert_array_equal(costs[0], np.full((3, 1, 3), 5.))
    origin[0, 2] = np.nan
    assert np.isinf(module.half_ray_costs(pose, rays)[0][:, :, 2]).all()


@pytest.mark.parametrize('bad', [np.nan, -1., -np.inf])
def test_invalid_costs_rejected(bad):
    costs = np.zeros((3, 1, 3))
    costs[0, 0, 0] = bad
    with pytest.raises(ValueError):
        module.assign([costs] * 3, coherent=True)


def test_aggregate_overflow_is_rejected_explicitly():
    with pytest.raises(ValueError, match='aggregated'):
        module.assign([np.zeros((3, 1, 3))] * 3, coherent=True,
                      node_weights=(1e308, 1e308, 1e308))

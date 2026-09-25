"""Synthetic geometry only: no saved fish pose, human labels, images or GPU."""

from itertools import product

import bounded_ray_seed as module
import numpy as np
import pytest


def interior(count=3):
    point = np.array([0.25, 0.5, 0.75])
    directions = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 2.0], [1.0, -1.0, 1.0]])[:count]
    origins = point - directions * np.arange(1, count + 1)[:, None]
    return origins, directions, np.zeros(3), np.ones(3)


def clipping_case():
    free = np.array([-1.0, 1.0, 0.0])
    directions = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return free - 2 * directions, directions, np.array([0.0, 0.0, -1.0]), np.array([2.0, 2.0, 1.0])


def squared_error(point, origins, directions):
    terms = []
    for origin, direction in zip(origins, directions, strict=True):
        unit = direction / np.linalg.norm(direction)
        delta = point - origin
        terms.append(np.dot(delta - np.dot(delta, unit) * unit, delta - np.dot(delta, unit) * unit))
    return float(sum(terms))


def exhaustive_box_oracle(origins, directions, lower, upper):
    """Enumerate all 27 face/interior states; solve free normal equations.

    Independent small oracle: no scipy optimizer and no production helper calls.
    The objective is strictly convex for these synthetic nonparallel systems.
    """
    hessian = np.zeros((3, 3))
    linear = np.zeros(3)
    for origin, direction in zip(origins, directions, strict=True):
        unit = direction / np.linalg.norm(direction)
        projector = np.eye(3) - np.outer(unit, unit)
        hessian += projector
        linear += projector @ origin
    feasible = []
    for state in product((-1, 0, 1), repeat=3):
        state = np.array(state)
        free = state == 0
        point = np.where(state == -1, lower, upper)
        if free.any():
            point[free] = np.linalg.solve(
                hessian[np.ix_(free, free)],
                linear[free] - hessian[np.ix_(free, ~free)] @ point[~free],
            )
        if np.all(point >= lower) and np.all(point <= upper):
            feasible.append((squared_error(point, origins, directions), point))
    return min(feasible, key=lambda pair: pair[0])


@pytest.mark.parametrize("count", [2, 3])
def test_exact_internal_intersection(count):
    result = module.solve_bounded_ray_seed(*interior(count))
    np.testing.assert_allclose(result.point_xyz, [0.25, 0.5, 0.75], atol=2e-14, rtol=0)
    np.testing.assert_allclose(result.ray_distances_mm, 0, atol=2e-14, rtol=0)
    np.testing.assert_array_equal(result.active_bounds, [0, 0, 0])
    assert result.success and result.rank == 3 and result.solver_status == 3
    assert result.rms_mm < 2e-14 and result.forward_ray.all()
    assert result.kkt_violation <= result.kkt_tolerance


def test_boundary_solution_strictly_improves_on_clipping_with_analytic_oracle():
    inputs = clipping_case()
    result = module.solve_bounded_ray_seed(*inputs)
    np.testing.assert_allclose(result.unconstrained_point_xyz, [-1, 1, 0], atol=1e-14)
    np.testing.assert_allclose(result.point_xyz, [0, 4 / 3, 0], atol=1e-14)
    np.testing.assert_array_equal(result.active_bounds, [-1, 0, 0])
    assert squared_error(
        np.clip(result.unconstrained_point_xyz, inputs[2], inputs[3]), *inputs[:2]
    ) == pytest.approx(1.5)
    assert result.sum_squared_distance_mm2 == pytest.approx(4 / 3)
    assert result.rms_mm == pytest.approx(np.sqrt(2 / 3))
    assert result.forward_ray.all()
    assert result.gradient[0] > 0
    np.testing.assert_allclose(result.gradient[1:], 0, atol=1e-14)


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("seed", [11, 19, 31, 43])
def test_matches_all_27_active_sets(count, seed):
    rng = np.random.default_rng(seed)
    origins = rng.uniform(-4, 4, (count, 3))
    directions = rng.normal(size=(count, 3))
    lower, upper = np.array([-1.0, -0.5, -2.0]), np.array([0.5, 1.0, 0.2])
    value, point = exhaustive_box_oracle(origins, directions, lower, upper)
    result = module.solve_bounded_ray_seed(origins, directions, lower, upper)
    np.testing.assert_allclose(result.point_xyz, point, rtol=0, atol=2e-13)
    assert result.sum_squared_distance_mm2 == pytest.approx(value, rel=1e-13, abs=1e-13)
    assert result.kkt_violation <= result.kkt_tolerance


def test_ray_order_positive_scale_translation_and_sign():
    origins, directions, lower, upper = clipping_case()
    result = module.solve_bounded_ray_seed(origins, directions, lower, upper)
    translated = module.solve_bounded_ray_seed(
        origins[::-1] + 128,
        directions[::-1] * np.array([[1e-200], [1e200]]),
        lower + 128,
        upper + 128,
    )
    np.testing.assert_allclose(translated.point_xyz - 128, result.point_xyz, atol=3e-14, rtol=0)
    np.testing.assert_allclose(
        translated.ray_distances_mm[::-1], result.ray_distances_mm, atol=3e-14, rtol=0
    )
    reversed_rays = module.solve_bounded_ray_seed(origins, -directions, lower, upper)
    np.testing.assert_array_equal(reversed_rays.point_xyz, result.point_xyz)
    np.testing.assert_array_equal(reversed_rays.forward_ray, ~result.forward_ray)
    np.testing.assert_array_equal(reversed_rays.ray_depth_mm, -result.ray_depth_mm)


def test_behind_ray_is_reported_not_a_changed_objective():
    origins, directions, lower, upper = interior(2)
    point = np.array([0.25, 0.5, 0.75])
    origins = 2 * point - origins
    result = module.solve_bounded_ray_seed(origins, directions, lower, upper)
    np.testing.assert_allclose(result.point_xyz, point, atol=1e-14)
    assert not result.forward_ray.any()
    assert np.all(result.ray_depth_mm < 0)


@pytest.mark.parametrize(
    "directions",
    [
        [[1, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [-2, 0, 0]],
        [[1, 0, 0], [1, 1e-18, 0]],
        [[0, 0, 0], [1, 0, 0]],
    ],
)
def test_degenerate_or_zero_directions_rejected(directions):
    with pytest.raises(ValueError, match=r"rank deficient|nonzero"):
        module.solve_bounded_ray_seed(np.zeros((2, 3)), directions, np.zeros(3), np.ones(3))


@pytest.mark.parametrize(
    ("index", "bad"),
    [
        (0, np.zeros((1, 3))),
        (1, np.ones((3, 3))),
        (2, 0),
        (3, [1, 1]),
        (2, [1, 0, 0]),
        (3, [-1, 1, 1]),
        (0, [[np.nan, 0, 0], [0, 0, 0]]),
        (1, [[np.inf, 0, 0], [0, 0, 1]]),
        (2, [0, -np.inf, 0]),
        (0, np.zeros((2, 3), bool)),
        (1, np.ones((2, 3), complex)),
        (0, np.zeros((2, 3), object)),
    ],
)
def test_bad_inputs_rejected(index, bad):
    inputs = list(interior(2))
    inputs[index] = bad
    with pytest.raises(ValueError):
        module.solve_bounded_ray_seed(*inputs)


def test_inputs_are_unchanged_and_all_outputs_owned():
    inputs = clipping_case()
    before = [array.tobytes() for array in inputs]
    for array in inputs:
        array.flags.writeable = False
    result = module.solve_bounded_ray_seed(*inputs)
    assert [array.tobytes() for array in inputs] == before
    for value in vars(result).values():
        if isinstance(value, np.ndarray):
            assert all(not np.shares_memory(value, source) for source in inputs)
            value.flat[0] = 0
    assert [array.tobytes() for array in inputs] == before


@pytest.mark.parametrize("corruption", ["failure", "nonfinite", "infeasible", "nonoptimal"])
def test_failed_or_corrupt_solver_does_not_become_success(monkeypatch, corruption):
    original = module.lsq_linear

    def fake(*args, **kwargs):
        result = original(*args, **kwargs)
        if corruption == "failure":
            result.success, result.status = False, 0
        elif corruption == "nonfinite":
            result.x[1] = np.nan
        elif corruption == "infeasible":
            result.x[0] = -0.01
        else:
            # Feasible clipping point with fake success: fails independent KKT.
            result.x[1] = 1
        return result

    monkeypatch.setattr(module, "lsq_linear", fake)
    with pytest.raises(RuntimeError):
        module.solve_bounded_ray_seed(*clipping_case())


def test_unrepresentable_finite_box_rejected():
    origins, directions, _, _ = interior(2)
    with pytest.raises(ValueError):
        module.solve_bounded_ray_seed(origins, directions, [-1e308] * 3, [1e308] * 3)

"""Small independent-oracle checks; the real audit replays every sealed pose."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
SPEC = importlib.util.spec_from_file_location(
    "instance_rescore_audit_test_module", WORK / "audit_instance_association_rescore.py"
)
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


def test_complete_domain_keeps_fourth_row_and_counts_equal_solutions():
    cost = np.array([[0, 1, 2, 2], [0, 100, 100, 100], [100, 0, 100, 100.0]])
    rows, ties = m.complete_injection(cost, 9.0)
    np.testing.assert_array_equal(rows, [2, 0, 1])
    assert ties == 2
    rows, ties = m.complete_injection(np.full((3, 4), 9.0), 9.0)
    np.testing.assert_array_equal(rows, [-1, -1, -1])
    assert ties == 1  # Null-equivalent rows do not invent assignment alternatives.
    rows, ties = m.complete_injection(np.empty((3, 0)), 9.0)
    np.testing.assert_array_equal(rows, [-1, -1, -1])
    assert ties == 1


def test_whole_row_constraint_exposes_cross_node_chimera_cost():
    cost = np.full((3, 3, 3), 20.0)
    cost[0, 0, :2] = 0
    cost[0, 1, 2] = 0
    cost[1, 1, :2] = 0
    cost[1, 0, 2] = 0
    cost[2, 2] = 0
    costs = [cost.copy() for _ in range(3)]
    relaxed, _ = m.complete_assignment(costs, False)
    whole, _ = m.complete_assignment(costs, True)
    assert m.validate_plan(relaxed, costs) == (0.0, 6, 3)
    whole_cost, mixed, shared = m.validate_plan(whole, costs, coherent=True)
    assert whole_cost > 0 and mixed == shared == 0
    np.testing.assert_array_equal(whole[0, :, 2], -1)
    with pytest.raises(ValueError, match="mixes or shares"):
        m.validate_plan(relaxed, costs, coherent=True)
    collision = relaxed.copy()
    collision[1, 0, 0] = collision[0, 0, 0]
    with pytest.raises(ValueError, match="same view/node"):
        m.validate_plan(collision, costs)


def test_row_relabel_preserves_unique_matching_and_objective():
    rng = np.random.default_rng(9021)
    costs = [rng.uniform(0, 20, (3, 4, 3)) for _ in range(3)]
    order = np.array([2, 0, 3, 1])
    shuffled = [cost[:, order] for cost in costs]
    for coherent in (False, True):
        plan, _ = m.complete_assignment(costs, coherent)
        changed, _ = m.complete_assignment(shuffled, coherent)
        restored = changed.copy()
        restored[changed >= 0] = order[changed[changed >= 0]]
        np.testing.assert_array_equal(restored, plan)
        assert m.validate_plan(plan, costs)[0] == m.validate_plan(changed, shuffled)[0]


def test_half_ray_cost_and_strict_support_distinguish_depth_and_pixel_gate():
    origin = np.zeros((1, 3, 3))
    direction = np.broadcast_to([0.0, 0.0, 1.0], origin.shape).copy()
    points = {view: np.zeros((1, 3, 2)) for view in m.p.base.VIEWS}
    rays = SimpleNamespace(origins=[origin] * 3, directions=[direction] * 3, points=points)
    pose = np.zeros((3, 3, 3))
    pose[0, 0] = [0, 0, -7]  # Behind the half-ray: robust node cost remains positive.
    pose[0, 1] = [0.7, 0, 7]
    costs = m.raw_costs(pose, rays)
    np.testing.assert_allclose(costs[0][0, 0, :2], [51.0, 1.0], rtol=1e-12)
    plan = np.full((3, 3, 3), -1, np.int32)
    plan[0, :, 0] = 0
    uv = [np.zeros((3, 3, 2)) for _ in range(3)]
    uv[0][0, 0] = [12.0001, 0]
    source, observed = m.strict_support(pose, plan, rays, uv)
    np.testing.assert_array_equal(source[0, :, 0], [-1, 0, 0])
    assert observed[0, 0]  # Original strict support uses infinite line, then 2+ views.
    uv[1][0, 0] = [12.0001, 0]
    _, observed = m.strict_support(pose, plan, rays, uv)
    assert not observed[0, 0]

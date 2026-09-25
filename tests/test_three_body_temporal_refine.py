"""Small CPU checks of temporal proposals, never biological identity evidence."""

import importlib.util
import sys
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import project
from medaka3d.types import VIEWS

LAB = Path(__file__).parents[1] / "labs/20260909_three_body"


def load_modules():
    names = ("node_geometry", "renderer", "support_batch", "temporal_renderer", "temporal_refine")
    previous, loaded = {name: sys.modules.get(name) for name in names}, {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(f"temporal_refine_test_{name}",
                                                         LAB / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return loaded


MODULES = load_modules()
refine = MODULES["temporal_refine"]
geometry = MODULES["node_geometry"]


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def poses():
    centers = np.array([[55., 65., 70.], [105., 100., 95.], [145., 130., 135.]])
    initial = centers[:, None] + np.array([[5., 0., 0.], [0., 0., 0.], [-12., 1., -1.]])
    return np.stack([initial + np.array([t * 0.25, np.sin(t) * 0.1, 0]) for t in range(5)])


def config(**changes):
    return replace(refine.RefinementConfig(mode="nodes", device="cpu", steps=3,
                                          reassign_every=2, learning_rate_mm=0.15), **changes)


def ray_sequence(rig, poses, *, empty_times=(), hidden_tails=False, shuffle=False):
    rays = []
    for t, pose in enumerate(poses):
        points = {view: project(rig.camera(view), pose) for view in VIEWS}
        if hidden_tails and t in (0, 1, len(poses) - 2, len(poses) - 1):
            for pixels in points.values():
                pixels[:, 2] = np.nan
        if t in empty_times:
            points = {view: np.empty((0, 3, 2)) for view in VIEWS}
        if shuffle:
            rng = np.random.default_rng(t + 107)
            points = {view: pixels.copy() for view, pixels in points.items()}
            for pixels in points.values():
                for node in range(3):
                    pixels[:, node] = pixels[rng.permutation(len(pixels)), node]
        rays.append(geometry.RayObservations.from_points(rig, points))
    return rays


def objective(rig, poses, cfg=None, rays=None, initial=None):
    return refine.TemporalObjective(
        rig, ray_sequence(rig, poses) if rays is None else rays, [None] * len(poses),
        poses[0] if initial is None else initial, cfg or config(),
    )


def test_all_twelve_starts_have_correct_boundary_permutation_and_supported_nodes_fixed(poses):
    observed = np.ones(poses.shape[:-1], bool)
    observed[[0, 1, 3, 4], :, 2] = False
    starts, perms, kinds, times, values, masks = refine.boundary_starts(poses, observed)
    assert starts.shape == (12, 5, 3, 3, 3)
    assert kinds.tolist() == ["linear", "scene"] * 6
    np.testing.assert_array_equal(perms[::2], list(permutations(range(3))))
    np.testing.assert_array_equal(perms[::2], perms[1::2])
    assert not masks[..., 2].any() and masks[..., :2].all()
    for index, permutation in enumerate(perms):
        np.testing.assert_array_equal(values[index, :2], poses[:2])
        np.testing.assert_array_equal(values[index, 2:], poses[-2:, permutation])
        np.testing.assert_array_equal(
            starts[index, times][masks[index]], values[index][masks[index]]
        )


def test_optimizer_keeps_all_fixed_boundaries_and_allows_unobserved_tail_motion(rig, poses):
    forward, observed = poses.copy(), np.ones(poses.shape[:-1], bool)
    forward[[0, 1, 3, 4], :, 2, 0] += 3.0
    observed[[0, 1, 3, 4], :, 2] = False
    rays = ray_sequence(rig, poses, hidden_tails=True)
    before, held_before = forward.copy(), observed.copy()
    starts, _, _, times, values, masks = refine.boundary_starts(forward, observed)
    output = refine.refine_window(rig, forward, observed, rays, [None] * 5, poses[0], config())
    fixed = np.broadcast_to(masks[..., None], values.shape)
    np.testing.assert_array_equal(output["pose_xyz"][:, times][fixed], values[fixed])
    assert (output["boundary_max_error_mm"] == 0).all()
    assert np.max(np.abs(output["pose_xyz"][:, times, :, 2] - starts[:, times, :, 2])) > 0.05
    assert not output["observed"][:, times, :, 2].any()
    assert "accepted_xyz" not in output and "identity_review" not in output
    np.testing.assert_array_equal(forward, before)
    np.testing.assert_array_equal(observed, held_before)


def test_saved_final_cost_and_support_are_recomputed_at_actual_retained_pose(rig, poses):
    rays, cfg = ray_sequence(rig, poses), config(steps=3, reassign_every=2)
    observed = np.ones(poses.shape[:-1], bool)
    output = refine.refine_window(rig, poses, observed, rays, [None] * 5, poses[0], cfg)
    obj = objective(rig, poses, cfg, rays)
    stored = obj.tensor(output["pose_xyz"])
    fresh_plans = obj.assignments(stored)
    cost, parts = obj(stored, obj.ray_tensors(fresh_plans))
    np.testing.assert_array_equal(output["costs"], cost.numpy())
    np.testing.assert_array_equal(output["loss_parts"], parts.numpy())
    np.testing.assert_array_equal(output["checkpoint_costs"][-1], output["costs"])
    assert (np.diff(output["checkpoint_costs"], axis=0) <= 1e-12).all()
    for t, rays_t in enumerate(rays):
        source, support, _ = refine.supported_nodes_batch(
            output["pose_xyz"][:, t], fresh_plans[:, t], rays_t, rig
        )
        np.testing.assert_array_equal(output["sources"][:, t], source)
        np.testing.assert_array_equal(output["observed"][:, t], support)


def test_unobserved_middle_can_have_finite_proposals_but_never_manufactures_sources(rig, poses):
    observed = np.ones(poses.shape[:-1], bool)
    observed[2] = False
    rays = ray_sequence(rig, poses, empty_times=(2,))
    output = refine.refine_window(rig, poses, observed, rays, [None] * 5, poses[0], config(steps=2))
    assert np.isfinite(output["pose_xyz"]).all() and np.isfinite(output["costs"]).all()
    assert not output["observed"][:, 2].any()
    assert (output["sources"][:, 2] == -1).all()
    assert not {"accepted_xyz", "identity_review", "identity_probability"}.intersection(output)


def test_time_reversing_pose_and_observations_preserves_objective(rig, poses):
    candidate = poses.copy()
    candidate[2, 0] += [3, -2, 1]
    rays = ray_sequence(rig, poses)
    forward = objective(rig, poses, rays=rays)
    backward = objective(rig, poses[::-1], rays=rays[::-1], initial=poses[0])
    p, q = forward.tensor(candidate[None]), backward.tensor(candidate[::-1].copy()[None])
    cost, parts = forward(p, forward.ray_tensors(forward.assignments(p)))
    reverse_cost, reverse_parts = backward(q, backward.ray_tensors(backward.assignments(q)))
    torch.testing.assert_close(cost, reverse_cost, atol=1e-12, rtol=0)
    torch.testing.assert_close(parts, reverse_parts, atol=1e-12, rtol=0)


def test_node_row_shuffling_changes_only_ephemeral_sources_and_preserves_exclusivity(rig, poses):
    first = objective(rig, poses)
    second = objective(rig, poses, rays=ray_sequence(rig, poses, shuffle=True))
    batch = first.tensor(poses[None])
    plans, shuffled = first.assignments(batch), second.assignments(batch)
    assert not np.array_equal(plans, shuffled)
    original, _ = first(batch, first.ray_tensors(plans))
    reordered, _ = second(batch, second.ray_tensors(shuffled))
    torch.testing.assert_close(original, reordered, rtol=0, atol=0)
    for t in range(5):
        for view in range(3):
            for node in range(3):
                assigned = shuffled[0, t, :, view, node]
                used = assigned[assigned >= 0]
                assert len(set(used)) == len(used)


def test_coincident_identical_fish_have_equal_six_costs_without_an_id_verdict(rig, poses):
    same = np.broadcast_to(poses[0, :1], poses.shape).copy()
    output = refine.refine_window(rig, same, np.ones(same.shape[:-1], bool),
                                  ray_sequence(rig, same), [None] * 5, same[0], config(steps=2))
    np.testing.assert_allclose(output["costs"], output["costs"][0], rtol=0, atol=1e-12)
    assert not {"winner_identity", "accepted_xyz", "identity_review", "identity_probability",
                "certified_identity"}.intersection(output)


def test_global_identity_renaming_preserves_physical_objective(rig, poses):
    candidate = poses.copy()
    candidate[2, 0] += [1, 2, 0]
    order = [2, 0, 1]
    first = objective(rig, poses)
    second = objective(rig, poses, initial=poses[0, order])
    p, q = first.tensor(candidate[None]), second.tensor(candidate[:, order][None])
    a, ap = first(p, first.ray_tensors(first.assignments(p)))
    b, bp = second(q, second.ray_tensors(second.assignments(q)))
    torch.testing.assert_close(a, b, atol=1e-12, rtol=0)
    torch.testing.assert_close(ap, bp, atol=1e-12, rtol=0)


def test_velocity_and_acceleration_penalize_a_return_jump_without_turning_it_into_truth(rig, poses):
    obj = objective(rig, poses)
    smooth = obj.tensor(poses[None])
    jumped = poses.copy()
    jumped[2, 0] += [30, 0, 0]
    discontinuous = obj.tensor(jumped[None])
    _, a = obj(smooth, obj.ray_tensors(obj.assignments(smooth)))
    _, b = obj(discontinuous, obj.ray_tensors(obj.assignments(discontinuous)))
    assert b[0, 2] > a[0, 2] + 10  # dimensionless acceleration loss, not identity error
    assert b[0, 3] > a[0, 3] + 10


@pytest.mark.parametrize("frames", [4, 51])
def test_window_frame_limits_fail_before_proposal_allocation(poses, frames):
    data = np.repeat(poses[:1], frames, axis=0)
    with pytest.raises(ValueError, match="five and fifty"):
        refine.boundary_starts(data, np.ones(data.shape[:-1], bool))


def test_nonboolean_support_and_hidden_out_of_bounds_pose_are_rejected(poses):
    with pytest.raises(ValueError, match="boolean"):
        refine.boundary_starts(poses, np.ones(poses.shape[:-1], float))
    changed, observed = poses.copy(), np.ones(poses.shape[:-1], bool)
    changed[2, 0, :, 2] = -100
    observed[2, 0] = False
    with pytest.raises(ValueError, match="engineering bounds"):
        refine.boundary_starts(changed, observed)
    observed[0, 0, 0] = False
    with pytest.raises(ValueError, match="supported mouth/pectoral"):
        refine.boundary_starts(poses, observed)


def test_infinite_anatomy_and_wrong_target_count_fail_before_optimization(rig, poses):
    anatomy = poses[0].copy()
    anatomy[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        objective(rig, poses, initial=anatomy)
    with pytest.raises(ValueError, match="one independent image target"):
        refine.TemporalObjective(rig, ray_sequence(rig, poses), [None] * 4, poses[0], config())

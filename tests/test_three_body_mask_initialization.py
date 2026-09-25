"""Bounded occupancy starts preserve uncertainty, temporal optimality and boundaries."""

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
SPEC = importlib.util.spec_from_file_location(
    "_three_body_mask_initialization_test", LAB / "mask_initialization.py"
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def evidence(frame, support, novelty=None):
    support = np.asarray(support, float).reshape(-1, 3)
    return module.FrameMaskVoxels(
        frame,
        np.arange(len(support)),
        support,
        np.ones(len(support)) if novelty is None else np.asarray(novelty),
    )


def base_pose():
    centres = np.array([[20, 30, 40], [60, 30, 40], [100, 30, 40]], float)
    return centres[:, None] + np.array([[2, 0, 0], [0, 0, 0], [-4, 0, 0]])[None]


def test_diverse_protected_phase_keeps_small_isolated_region_despite_healthy_weight():
    dense = np.stack(
        np.meshgrid(np.arange(40, 63, 2), np.arange(40, 63, 2), [40]), axis=-1
    ).reshape(-1, 3)
    points = np.vstack((dense, [150, 150, 100]))
    values = np.ones((len(points), 3))
    values[-1] = [0.6, 0.6, 0]
    novelty = np.ones(len(points))
    novelty[-1] = 0.001
    result = module.diverse_pool(points, evidence(1, values, novelty))
    assert len(result.xyz) == 48
    selected = np.flatnonzero(result.grid_indices == len(points) - 1)
    assert len(selected) == 1
    assert result.selection_phase[selected[0]] == 0
    assert result.known_third_contradiction[selected[0]]
    assert np.sum(result.selection_phase == 0) == 24
    distance = np.linalg.norm(result.xyz[:, None] - result.xyz[None], axis=-1)
    assert (distance[~np.eye(len(result.xyz), dtype=bool)] > 2).all()


def test_empty_image_evidence_retains_only_neutral_base_fallback():
    pool = module.diverse_pool(np.empty((0, 3)), evidence(1, []))
    base = base_pose()
    frame = module.whole_pose_candidates(base, 2, pool)
    np.testing.assert_array_equal(frame["poses"], base[None])
    np.testing.assert_array_equal(frame["emissions"], [3])
    np.testing.assert_array_equal(frame["grid_indices"], [-1])
    assert frame["kinds"].tolist() == ["neutral_base_fallback"]
    observed = module.whole_pose_candidates(base, None, pool)
    np.testing.assert_array_equal(observed["emissions"], [0])


def test_whole_body_outside_bounds_is_rejected_without_clipping():
    base = base_pose()
    base[0] = [[2, 10, 10], [1, 10, 10], [0, 10, 10]]
    points = np.array([[0, 10, 10], [1, 10, 10], [192, 10, 10]], float)
    cfg = module.MaskInitializationConfig(nms_radius_mm=0.1)
    pool = module.diverse_pool(points, evidence(1, np.ones((3, 3))), cfg)
    frame = module.whole_pose_candidates(base, 0, pool, cfg)
    assert set(frame["rejected_grid_indices"]) == {0, 2}
    assert frame["poses"].shape[0] == 2
    assert frame["rejected_outside"].any(axis=(1, 2, 3)).all()
    assert np.any(frame["rejected_outside"][:, 0, 0, 0])
    assert np.any(frame["rejected_outside"][:, 0, 2, 0])
    np.testing.assert_array_equal(frame["poses"][:, 0, :, 0], [[2, 1, 0], [2, 1, 0]])


def test_first_order_dp_matches_all_bruteforce_three_body_paths():
    rng = np.random.default_rng(47)
    candidates = []
    for _ in range(4):
        poses = np.repeat(base_pose()[None], 2, axis=0)
        poses += rng.normal(0, 8, (2, 3, 1, 3))
        candidates.append({"poses": poses, "emissions": rng.uniform(0, 1, 2)})
    brute = []
    for path in itertools.product(range(2), repeat=4):
        cost = sum(candidates[t]["emissions"][path[t]] for t in range(4))
        for t in range(1, 4):
            delta = (
                candidates[t]["poses"][path[t], :, 1]
                - candidates[t - 1]["poses"][path[t - 1], :, 1]
            )
            cost += 0.05 * np.mean(np.sum(delta**2, axis=-1)) / 3**2
        brute.append((cost, path))
    expected = min(brute)
    result = module.dynamic_path(candidates)
    assert result["path_cost"] == pytest.approx(expected[0], abs=1e-12)
    assert tuple(result["path_indices"]) == expected[1]


def test_all_six_starts_keep_original_boundary_constraints_and_no_identity_verdict():
    forward = np.repeat(base_pose()[None], 7, axis=0)
    forward += np.arange(7)[:, None, None, None] * np.array([0.1, 0, 0])
    held = np.ones((7, 3, 3), bool)
    held[3, 2, 1] = False
    held[1, 0, 2] = False
    original_pose, original_held = forward.copy(), held.copy()
    original_starts = module.boundary_starts(forward, held)
    points = np.array([[99, 30, 40], [105, 31, 40], [70, 30, 40]], float)
    images = [evidence(100 + t, np.ones((3, 3))) for t in range(7)]
    result = module.build_mask_initializations(forward, held, points, images)
    assert result.pose_xyz.shape == (6, 7, 3, 3, 3)
    assert len({tuple(p) for p in result.permutations}) == 6
    scene_rows = np.flatnonzero(original_starts[2] == "scene")
    for b, row in enumerate(scene_rows):
        np.testing.assert_array_equal(
            result.pose_xyz[b, original_starts[3]], original_starts[0][row, original_starts[3]]
        )
        assert result.missing_slot[b, 3] == int(np.flatnonzero(result.scene_mapping[b, 3] == 2)[0])
    for index in (3, 4, 5):
        wanted = original_starts[index] if index == 3 else original_starts[index][scene_rows]
        np.testing.assert_array_equal(result.starts[index], wanted)
    np.testing.assert_array_equal(forward, original_pose)
    np.testing.assert_array_equal(held, original_held)
    assert result.metadata["new_external_resets"] == 0
    assert result.metadata["accepted_identity_output"] is False
    assert "accepted_xyz" not in result.arrays
    assert not any(value.dtype.hasobject for value in result.arrays.values())
    json.dumps(result.metadata, allow_nan=False)
    offsets = result.arrays["candidate_offsets"]
    assert len(offsets) == 6 * 7 + 1
    assert offsets[-1] == len(result.arrays["dp_costs"])


def test_more_than_one_missing_physical_pectoral_is_explicitly_rejected():
    forward = np.repeat(base_pose()[None], 5, axis=0)
    held = np.ones((5, 3, 3), bool)
    held[2, :2, 1] = False
    images = [evidence(100 + t, []) for t in range(5)]
    with pytest.raises(ValueError, match="more than one physically missing"):
        module.build_mask_initializations(forward, held, np.empty((0, 3)), images)


def test_known_third_contradiction_and_unknown_third_have_distinct_emissions():
    points = np.array([[90, 30, 40], [100, 30, 40]], float)
    pool = module.diverse_pool(points, evidence(1, [[1, 1, 0], [1, 1, np.nan]]))
    frame = module.whole_pose_candidates(base_pose(), 2, pool)
    by_grid = dict(zip(frame["grid_indices"], frame["emissions"], strict=True))
    assert by_grid[-1] == 3
    assert by_grid[0] == pytest.approx(1)
    assert by_grid[1] == 0
    assert pool.known_third_contradiction[pool.grid_indices == 0].all()
    assert not pool.known_third_contradiction[pool.grid_indices == 1].any()


def test_missing_native_frame_and_unobserved_boundary_are_not_repaired():
    forward = np.repeat(base_pose()[None], 5, axis=0)
    held = np.ones((5, 3, 3), bool)
    bad_frames = [evidence(t, []) for t in [100, 101, 103, 104, 105]]
    with pytest.raises(ValueError, match="frame gaps"):
        module.build_mask_initializations(forward, held, np.empty((0, 3)), bad_frames)
    held[0, 0, 1] = False
    with pytest.raises(ValueError, match="both ends"):
        module.build_mask_initializations(forward, held, np.empty((0, 3)), bad_frames)

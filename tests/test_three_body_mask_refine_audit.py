"""Additional starts retain source-bound boundaries and never manufacture ID truth."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from test_three_body_temporal_refine import MODULES, config, ray_sequence

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
refine = MODULES["temporal_refine"]


def load_runner():
    names = (
        *MODULES,
        "prepare_sequence",
        "observations",
        "run_experiment",
        "run_temporal_refine",
        "mask_initialization",
        "run_mask_refine",
    )
    previous = {name: sys.modules.get(name) for name in names}
    old_path = sys.path.copy()
    loaded = dict(MODULES)
    try:
        sys.modules.update(MODULES)
        for name in names[len(MODULES):]:
            spec = importlib.util.spec_from_file_location(f"_mask_refine_audit_{name}",
                                                         LAB / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        sys.path[:] = old_path
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return loaded["run_mask_refine"], loaded["run_temporal_refine"]


runner, validation = load_runner()


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def scene():
    centers = np.array([[55., 65., 70.], [105., 100., 95.], [145., 130., 135.]])
    skeleton = centers[:, None] + np.array([[5., 0., 0.], [0., 0., 0.], [-12., 1., -1.]])
    pose = np.stack([skeleton + np.array([t * 0.25, np.sin(t) * 0.1, 0]) for t in range(5)])
    observed = np.ones(pose.shape[:-1], bool)
    observed[2, 2] = False
    return pose, observed


def six_starts(pose, observed):
    arrays = refine.boundary_starts(pose, observed)
    return (
        arrays[0][::2].copy(), arrays[1][::2].copy(), np.full(6, "mask_grid"),
        arrays[3].copy(), arrays[4][::2].copy(), arrays[5][::2].copy(),
    )


def optimize(rig, scene, starts=None, rays=None):
    pose, observed = scene
    return refine.optimize_starts(
        rig, pose, observed, ray_sequence(rig, pose) if rays is None else rays,
        [None] * len(pose), pose[0], six_starts(pose, observed) if starts is None else starts,
        config(steps=2, reassign_every=1),
    )


def test_six_added_starts_keep_boundaries_costs_and_inputs_and_no_admission(rig, scene):
    pose, observed = scene
    starts = six_starts(pose, observed)
    starts[0][:, 2, 2] += [2.0, -1.0, 0.5]
    before_pose, before_observed = pose.copy(), observed.copy()
    before_starts = tuple(array.copy() for array in starts)
    rays = ray_sequence(rig, pose)
    result = optimize(rig, scene, starts, rays)
    assert result["pose_xyz"].shape == (6, 5, 3, 3, 3)
    np.testing.assert_array_equal(result["permutations"], starts[1])
    np.testing.assert_array_equal(result["start_kind"], ["mask_grid"] * 6)
    validation.validate_hypotheses(result, np.arange(5), rays, expected_kinds=("mask_grid",))
    validation.validate_boundaries(result, pose, observed)
    with pytest.raises(ValueError, match="invalid refinement output"):
        validation.validate_hypotheses(result, np.arange(5), rays)
    obj = refine.TemporalObjective(rig, rays, [None] * 5, pose[0], config(steps=2))
    actual = obj.tensor(result["pose_xyz"])
    costs, parts = obj(actual, obj.ray_tensors(obj.assignments(actual)))
    np.testing.assert_array_equal(result["costs"], costs.numpy())
    np.testing.assert_array_equal(result["loss_parts"], parts.numpy())
    np.testing.assert_array_equal(pose, before_pose)
    np.testing.assert_array_equal(observed, before_observed)
    for after, before in zip(starts, before_starts, strict=True):
        np.testing.assert_array_equal(after, before)
    forbidden = {"accepted_xyz", "identity_review", "identity_probability", "winner_identity"}
    assert not forbidden & result.keys()


@pytest.mark.parametrize("corruption", ["remove_support", "matched_value", "unheld_value"])
def test_self_consistent_boundary_tampering_is_rejected_against_forward(rig, scene, corruption):
    pose, observed = scene
    if corruption == "unheld_value":
        observed[0, 0, 2] = False
    starts = six_starts(pose, observed)
    if corruption == "remove_support":
        starts[5][0, 0, 0, 0] = False
    elif corruption == "matched_value":
        starts[4][0, 0, 0, 0, 0] += 1.0
        starts[0][0, 0, 0, 0, 0] += 1.0
    else:
        starts[4][0, 0, 0, 2, 0] += 1.0
    with pytest.raises(ValueError, match="original forward boundary"):
        optimize(rig, scene, starts)


@pytest.mark.parametrize("corruption", ["pose_nan", "duplicate_slot", "float_permutation",
                                        "nonboolean_mask", "float_times", "value_nan"])
def test_invalid_starts_fail_before_objective_creation(rig, scene, monkeypatch, corruption):
    starts = list(six_starts(*scene))
    if corruption == "pose_nan":
        starts[0][0, 2, 0, 0, 0] = np.nan
    elif corruption == "duplicate_slot":
        starts[1][0] = [0, 0, 2]
    elif corruption == "float_permutation":
        starts[1] = starts[1].astype(float)
    elif corruption == "nonboolean_mask":
        starts[5] = starts[5].astype(int)
    elif corruption == "float_times":
        starts[3] = starts[3].astype(float)
    else:
        starts[4][0, 0, 0, 0, 0] = np.nan

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid starts reached the objective")

    monkeypatch.setattr(refine, "TemporalObjective", forbidden)
    with pytest.raises(ValueError):
        optimize(rig, scene, tuple(starts))


def test_six_poses_remain_latent_when_all_middle_observations_are_missing(rig, scene):
    rays = ray_sequence(rig, scene[0], empty_times=(2,))
    result = optimize(rig, scene, rays=rays)
    assert np.isfinite(result["pose_xyz"][:, 2]).all()
    assert not result["observed"][:, 2].any()
    assert (result["sources"][:, 2] == -1).all()


def test_output_validator_rejects_missing_competitor_or_nan_even_with_six_rows(rig, scene):
    rays = ray_sequence(rig, scene[0])
    result = optimize(rig, scene, rays=rays)
    invalid = {key: value.copy() for key, value in result.items()}
    invalid["permutations"][5] = invalid["permutations"][0]
    with pytest.raises(ValueError, match="all six permutations"):
        validation.validate_hypotheses(invalid, np.arange(5), rays, expected_kinds=("mask_grid",))
    invalid = {key: value.copy() for key, value in result.items()}
    invalid["costs"][5] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        validation.validate_hypotheses(invalid, np.arange(5), rays, expected_kinds=("mask_grid",))


def test_nan_objective_fails_the_whole_six_competitor_run(rig, scene, monkeypatch):
    original = refine.TemporalObjective.__call__

    def with_failed_competitor(self, pose, rays):
        cost, parts = original(self, pose, rays)
        cost = cost.clone()
        cost[-1] = torch.nan
        return cost, parts

    monkeypatch.setattr(refine.TemporalObjective, "__call__", with_failed_competitor)
    with pytest.raises(FloatingPointError, match="nonfinite temporal objective"):
        optimize(rig, scene)


def test_sealed_source_cannot_silently_replace_a_previously_recorded_digest(tmp_path):
    source = tmp_path / "input.npz"
    source.write_bytes(b"synthetic sealed input")
    digest = runner.sha256(source)
    record = {
        "status": "ok",
        "inputs": {"source": {"path": str(source), "sha256": digest}},
        "outputs": {},
    }
    (tmp_path / "refinement.provenance.json").write_text(json.dumps(record))
    hashes = {str(source.resolve()): "0" * 64}
    with pytest.raises(ValueError, match="inconsistent sealed source digests"):
        runner.sealed_record(tmp_path, "refinement", hashes)


@pytest.mark.parametrize("unbound", ["refinement", "diagnostic"])
def test_same_frame_range_cannot_substitute_another_sealed_forward(tmp_path, monkeypatch, unbound):
    base = tmp_path / "base"
    base.mkdir()
    (base / "summary.json").write_text(json.dumps({"frames": [10, 15]}))
    required = str((tmp_path / "actual_forward.npz").resolve())
    alternate = str((tmp_path / "other_forward.npz").resolve())
    digest = "a" * 64

    def sealed(root, name, hashes):
        path = alternate if name == unbound else required
        return {"inputs": {"forward": {"path": path, "sha256": digest}}}

    monkeypatch.setattr(runner, "sealed_record", sealed)
    monkeypatch.setattr(
        runner, "load_sealed_forward",
        lambda *args: ({}, None, None, None, None, {required: digest}, {}),
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="not bound to this forward input"):
        runner.main([
            "--forward", str(tmp_path / "forward"), "--sequence", str(tmp_path / "sequence"),
            "--base-refinement", str(base), "--mask-grid", str(tmp_path / "grid"),
            "--output", str(output), "--device", "cpu",
        ])
    assert not output.exists()

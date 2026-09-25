"""CPU integration boundaries for the preregistered local reflection experiment."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
import run_reflection_local_probe as probe  # noqa: E402

from test_three_body_local_fit import make_case  # noqa: E402


def test_three_real_cpu_arms_share_one_plan_and_preserve_inputs(rig, monkeypatch):
    initial, rays, fit = make_case(rig)
    start = initial + np.array([.12, -.08, .04])
    predicted, missing = initial.copy(), np.array([0, 4, 21])
    copies = [x.copy() for x in (initial, start, predicted, missing)]
    original = probe.assignment_plans
    calls = []

    def assigned(*args, **kwargs):
        calls.append((args[2:], kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(probe, "assignment_plans", assigned)
    cfg = probe.TrackerConfig(device="cpu", opt_steps=2)
    plan, results, sources, observed = probe.fit_arms(
        rig, initial, cfg, start, predicted, missing, rays, fit)
    assert calls == [((cfg.ray_sigma_mm, cfg.missing_cost), {"keep": 1})]
    assert [r["arm"] for r in results] == list(probe.ARMS)
    assert sources.shape == (3, 3, 3, 3, 3) and observed.shape == (3, 3, 3, 3)
    for r in results:
        np.testing.assert_array_equal(r["plan"], plan)
        assert r["iteration_pose_xyz"].shape == (3, 3, 3, 3)
    for actual, expected in zip((initial, start, predicted, missing), copies, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_all_fitted_poses_cross_evaluated_without_own_model_only_selection(rig):
    initial, _, fit = make_case(rig)
    target = SimpleNamespace(pixels=fit.pixels, target=fit.target, weights=fit.weights,
                             strata=np.where(fit.target > 0, 0, 2))
    poses = np.stack([initial, initial + np.array([.1, .03, -.05]),
                      initial + np.array([-.08, .1, .02])])
    unions, valid, metrics = probe.evaluate_poses(
        rig, probe.TrackerConfig(device="cpu"), poses, target)
    assert set(unions) == set(metrics) == set(probe.ARMS)
    for name in probe.ARMS:
        assert unions[name].shape == (3, *target.target.shape)
        explicit = ((unions[name] - target.target) ** 2 * target.weights * valid).sum(-1)
        np.testing.assert_allclose(metrics[name]["loss_by_view"], explicit, atol=1e-15, rtol=0)


def test_empty_evaluation_does_not_invent_pixels_or_construct_renderer(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("renderer called on empty evaluation")

    monkeypatch.setattr(probe, "ReflectionFishRenderer", forbidden)
    target = SimpleNamespace(target=np.empty((3, 0)), weights=np.empty((3, 0)),
                             strata=np.empty((3, 0), np.int8))
    _, valid, metrics = probe.evaluate_poses(None, None, np.zeros((3, 3, 3, 3)), target)
    assert valid.shape == (3, 0)
    for values in metrics.values():
        np.testing.assert_array_equal(values["mean_view_loss"], np.zeros(3))


def test_changed_renderer_alone_cannot_pass_and_gate_has_fixed_tolerance():
    frames = np.asarray(probe.FRAMES)
    support = np.ones((13, 3, 3, 3), bool)
    losses = np.ones((13, 3, 3))
    losses[:, :, 1:] = .5  # identical poses, only scoring model improves all rows
    result = probe.screening(frames, support, losses)
    assert result["unit"]["average_chosen_error_lower_than_direct"]
    assert not result["unit"]["same_model_error_lower_than_direct_arm_pose"]
    assert not result["unit"]["gate_pass"]
    losses[:, 1, 1] -= .5e-10
    assert not probe.screening(frames, support, losses)["unit"]["gate_pass"]
    losses[:, 1, 1] -= 2e-10
    assert probe.screening(frames, support, losses)["unit"]["gate_pass"]


@pytest.mark.parametrize("node,frame", [(1, 79316), (0, 82719), (0, 82720)])
def test_a_lost_scene_slot_is_not_cancelled_by_another_slot_gain(node, frame):
    frames = np.asarray(probe.FRAMES)
    support = np.ones((13, 3, 3, 3), bool)
    t = list(frames).index(frame)
    support[t, 0, 2, node] = False  # a gain elsewhere cannot compensate slot 0's loss
    support[t, 1, 0, node] = False
    losses = np.ones((13, 3, 3))
    losses[:, 1, 1] = .5
    result = probe.screening(frames, support, losses)["unit"]
    assert not result["gate_pass"]
    key = "lost_chest_scene_slots" if node == 1 else "lost_recovery_mouth_scene_slots"
    assert result[key] == [dict(frame_idx=frame, slot=0)]


def test_per_view_node_exclusivity_rejects_duplicate_even_with_other_nodes_missing():
    plan = np.full((3, 3, 3), -1, np.int32)
    plan[0, 1, 0] = plan[2, 1, 0] = 0
    with pytest.raises(ValueError, match="consumed twice"):
        probe.assert_exclusive(plan)
    plan[2, 1, 0], plan[2, 1, 1] = -1, 0
    probe.assert_exclusive(plan)  # different semantic nodes remain the frozen policy


@pytest.fixture
def isolated_main(tmp_path, monkeypatch):
    source = tmp_path / "new_source.py"
    evidence = tmp_path / "candidate.npz"
    source.write_text("# synthetic runner source\n")
    evidence.write_bytes(b"sealed but unread candidate bytes")
    code = {str(source): probe.sha256(source)}
    hashes = {str(evidence): probe.sha256(evidence)}
    monkeypatch.setattr(probe, "REPO", tmp_path)
    monkeypatch.setattr(probe, "source_hashes", lambda: dict(code))
    monkeypatch.setattr(probe, "verify_evidence", lambda args, code: (
        SimpleNamespace(root=tmp_path / "run"), probe.TrackerConfig(device="cpu"),
        tmp_path / "pixels", dict(hashes)))
    calls = []

    def fake_fits(verified, cfg, pixels_root, output, record):
        assert cfg.device == "cpu" and torch.get_num_threads() == 2
        assert (output / "plan.json").is_file()
        calls.append("fit after registration")
        path = output / "synthetic_result.json"
        path.write_text('{"synthetic": true}')
        record.output("result", path, sha256=probe.sha256(path))

    monkeypatch.setattr(probe, "run_fits", fake_fits)
    return tmp_path, source, evidence, calls, fake_fits


def test_registration_does_not_fit_then_sealed_execution_restores_threads(isolated_main):
    root, _, _, calls, _ = isolated_main
    threads = torch.get_num_threads()
    registration, output = root / "registration", root / "output"
    probe.main(["--register-only", "--output", str(registration)])
    assert calls == [] and torch.get_num_threads() == threads
    assert probe.read_json(registration / "registration.provenance.json")["status"] == "ok"
    probe.main(["--registration", str(registration), "--output", str(output)])
    assert calls == ["fit after registration"] and torch.get_num_threads() == threads
    seal = probe.read_json(output / "probe.provenance.json")
    assert seal["status"] == "ok"
    for entry in seal["outputs"].values():
        assert probe.sha256(entry["path"]) == entry["sha256"]
    with pytest.raises(ValueError, match="new workspace"):
        probe.main(["--register-only", "--output", str(output)])


@pytest.mark.parametrize("change", ["input", "snapshot", "output", "fit_error"])
def test_mutation_seals_failure_and_restores_threads(isolated_main, monkeypatch, change):
    root, _, evidence, calls, original = isolated_main
    registration, output = root / "registration", root / "output"
    probe.main(["--register-only", "--output", str(registration)])
    threads = torch.get_num_threads()

    def failing(*args):
        original(*args)
        current = args[-2]
        if change == "fit_error":
            raise RuntimeError("synthetic fit failure")
        path = {"input": evidence, "snapshot": current / "code_snapshot/new_source.py",
                "output": current / "synthetic_result.json"}[change]
        path.write_text("changed")

    monkeypatch.setattr(probe, "run_fits", failing)
    with pytest.raises((RuntimeError, ValueError)):
        probe.main(["--registration", str(registration), "--output", str(output)])
    assert calls == ["fit after registration"]
    assert torch.get_num_threads() == threads
    assert probe.read_json(output / "probe.provenance.json")["status"] == "failed"


def test_registration_plan_tamper_rejected_before_fit(isolated_main):
    root, _, _, calls, _ = isolated_main
    registration = root / "registration"
    probe.main(["--register-only", "--output", str(registration)])
    path = registration / "plan.json"
    altered = probe.read_json(path)
    altered["threads"] = 3
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="SHA"):
        probe.main(["--registration", str(registration), "--output", str(root / "output")])
    assert calls == []

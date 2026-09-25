"""CPU-only operational boundaries for the separately registered temporal run."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
import run_reflection_temporal as runner  # noqa: E402


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def entry(path):
    return dict(path=str(path), sha256=runner.sha256(path))


def fake_native(monkeypatch, indices):
    state = dict(read=[], closed=False, start=None)

    def source(reader, start):
        state["start"] = start
        try:
            for index in indices:
                state["read"].append(index)
                yield SimpleNamespace(frame_idx=index)
        finally:
            state["closed"] = True

    monkeypatch.setattr(runner, "iter_from", source)
    return state


def test_subwindow_reads_exact_native_200_not_next_and_closes(monkeypatch):
    reader = SimpleNamespace(frames=np.arange(76620, 86580))
    state = fake_native(monkeypatch, range(runner.START, runner.STOP + 1))
    observed = list(runner.bounded_frames(reader))
    assert state == dict(
        read=list(range(runner.START, runner.STOP)), closed=True, start=runner.START
    )
    assert [v.frame_idx for v in observed] == runner.FRAMES.tolist()
    assert reader.frames[0] == 76620  # Full manifest was never rewritten.


def test_subwindow_consumer_close_releases_without_extra_read(monkeypatch):
    state = fake_native(monkeypatch, range(100, 110))
    iterator = runner.bounded_frames(SimpleNamespace(frames=np.arange(95, 115)), 100, 105)
    assert next(iterator).frame_idx == 100
    iterator.close()
    assert state["closed"] and state["read"] == [100]


@pytest.mark.parametrize(
    "indices,match", [([101], "misordered"), ([100, 102], "misordered"), ([100], "ended before")]
)
def test_wrong_seek_gap_and_early_end_fail_with_close(monkeypatch, indices, match):
    state = fake_native(monkeypatch, indices)
    with pytest.raises(ValueError, match=match):
        list(runner.bounded_frames(SimpleNamespace(frames=np.arange(95, 115)), 100, 103))
    assert state["closed"]


@pytest.mark.parametrize(
    "start,stop", [(94, 100), (100, 116), (100, 100), (101, 100), (100.0, 105), (True, 105)]
)
def test_invalid_subwindow_never_opens_source(monkeypatch, start, stop):
    def forbidden(*args):
        pytest.fail("invalid range opened native source")

    monkeypatch.setattr(runner, "iter_from", forbidden)
    with pytest.raises(ValueError, match="invalid native"):
        list(runner.bounded_frames(SimpleNamespace(frames=np.arange(95, 115)), start, stop))


def qualification_errors():
    result = {
        key: dict(maximum_absolute_difference=0.0, tolerance=1e-9, passed=True)
        for key in (
            "direct_alpha",
            "inner_alpha",
            "outer_alpha",
            "direct_union",
            "unit_union",
            "fresnel_union",
            "initial_lengths_mm",
            "fit_score",
            "fit_parts",
            *("gradient_" + a for a in ("direct", "unit", "fresnel")),
            *("loss_" + a for a in ("direct", "unit", "fresnel")),
        )
    }
    result.update(
        {
            key: dict(maximum_absolute_difference=0.0, tolerance=limit, passed=True)
            for key, limit in (
                ("inner_weight", 1e-12),
                ("outer_weight", 1e-12),
                ("fit_pose_xyz", 1e-8),
            )
        }
    )
    result.update(
        {
            key: dict(exact=True, mismatched_elements=0)
            for key in (
                "ray_valid",
                "direct_valid",
                "inner_valid",
                "outer_valid",
                "fit_direct_mask_support",
            )
        }
    )
    return result


@pytest.fixture
def qualification(tmp_path, monkeypatch):
    lab = tmp_path / "lab"
    lab.mkdir()
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "LAB", lab)
    protocol = tmp_path / "plan.md"
    protocol.write_text("fixed plan")
    producer = tmp_path / "qualify.py"
    producer.write_text("fixed producer")
    monkeypatch.setattr(runner, "PROTOCOL", protocol)
    monkeypatch.setattr(runner, "PROTOCOL_SHA", runner.sha256(protocol))
    monkeypatch.setattr(runner, "QUALIFICATION_SOURCE", producer)
    monkeypatch.setattr(runner, "QUALIFICATION_SHA", runner.sha256(producer))
    codes = {str(p): runner.sha256(p) for p in (protocol, producer)}
    for name in (
        "reflection_renderer.py",
        "reflection_paths.py",
        "reflection_tracker.py",
        "reflection_local_fit.py",
        "renderer.py",
        "graph_fit.py",
        "tracker.py",
        "node_geometry.py",
        "persistent_node_tracker.py",
    ):
        path = lab / name
        path.write_text(name)
        codes[str(path)] = runner.sha256(path)
    root = tmp_path / "qualified"
    root.mkdir()
    (root / "arrays.npz").write_bytes(b"sealed arrays, not read by consumer")
    outputs = {"arrays.npz": entry(root / "arrays.npz")}
    for path in codes:
        target = root / "code_snapshot" / Path(path).relative_to(tmp_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(path).read_bytes())
        outputs[str(target)] = entry(target)
    summary = dict(
        schema="reflection_CPU_CUDA_qualification_v1",
        passed=True,
        frame_idx=79381,
        batch=3,
        start_arms=["direct", "unit", "fresnel"],
        tolerances=runner.QUALIFICATION_TOLERANCES.copy(),
        code_hashes=codes,
        input_hashes=codes.copy(),
        runtime={**runner.software_runtime(), "cuda_device_name": "fake device"},
        tracker_config={**runner.configs()["tracker"], "device": "cpu"},
        direct_mode="unbounded",
        fitted_mask_arm="fresnel",
        opt_steps=14,
        GT=False,
        no_temporal_tracking=True,
        cuda_device_verified=True,
        errors=qualification_errors(),
    )
    seal = dict(
        status="ok",
        stage="reflection_CPU_CUDA_qualification",
        outputs=outputs,
        inputs={p: entry(Path(p)) for p in codes},
        params={
            "plan": copy.deepcopy(summary),
            "all_input_source_snapshot_output_sha_unchanged": True,
        },
    )

    def save():
        dump(root / "summary.json", summary)
        seal["outputs"]["summary.json"] = entry(root / "summary.json")
        dump(root / "qualification.provenance.json", seal)

    save()
    return root, codes, summary, seal, save


def test_qualification_final_role_summary_hash_and_cpu_safe_runtime(qualification, monkeypatch):
    root, codes, _, _, _ = qualification

    def forbidden(*args):
        pytest.fail("qualification binding initialized CUDA")

    monkeypatch.setattr(runner.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(runner.torch.cuda, "get_device_name", forbidden)
    actual = runner.verify_qualification(root, codes, {})
    assert actual["summary_sha256"] == runner.sha256(root / "summary.json")
    assert actual["summary_sha256"] != list(codes.values())[-1]


@pytest.mark.parametrize(
    "key,value",
    [
        ("passed", False),
        ("opt_steps", 2),
        ("direct_mode", "finite"),
        ("fitted_mask_arm", "unit"),
        ("GT", True),
        ("cuda_device_verified", False),
    ],
)
def test_qualification_rejects_failed_or_wrong_conditions(qualification, key, value):
    root, codes, summary, _, save = qualification
    summary[key] = value
    save()
    with pytest.raises(ValueError, match="incomplete or failed"):
        runner.verify_qualification(root, codes, {})


def test_qualification_cannot_relax_tolerance_or_hide_failed_flag(qualification):
    root, codes, summary, _, save = qualification
    summary["errors"]["ray_valid"]["exact"] = False
    save()
    with pytest.raises(ValueError, match="flag comparison"):
        runner.verify_qualification(root, codes, {})
    summary["errors"] = qualification_errors()
    summary["tolerances"]["response"] = 1.0
    save()
    with pytest.raises(ValueError, match="incomplete or failed"):
        runner.verify_qualification(root, codes, {})


def test_qualification_rejects_unbound_snapshot_and_changed_registration(qualification):
    root, codes, _summary, seal, save = qualification
    role = next(k for k in seal["outputs"] if "code_snapshot" in k)
    removed = seal["outputs"].pop(role)
    save()
    with pytest.raises(ValueError, match="snapshot unbound"):
        runner.verify_qualification(root, codes, {})
    seal["outputs"][role] = removed
    seal["params"]["plan"]["opt_steps"] = 15
    save()
    with pytest.raises(ValueError, match="differs from its registration"):
        runner.verify_qualification(root, codes, {})


@pytest.fixture
def registration(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO", tmp_path)
    root = tmp_path / "registered"
    root.mkdir()
    code_path = tmp_path / "source.py"
    code_path.write_text("frozen")
    candidate = tmp_path / "historical_candidate.npz"
    candidate.write_bytes(b"old coordinate archive; must never deserialize")
    code = {str(code_path): runner.sha256(code_path)}
    seed = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    np.save(root / "initial_pose_xyz.npy", seed)
    qualification = dict(root=str(tmp_path), summary_sha256="qualification bound elsewhere")
    monkeypatch.setattr(runner, "verify_qualification", lambda *args: qualification.copy())
    plan = dict(
        preflight_only=False,
        frames=runner.FRAMES.tolist(),
        arms=list(runner.ARMS),
        protocol_sha256=runner.PROTOCOL_SHA,
        config=runner.configs(),
        code_hashes=code,
        software_runtime=runner.software_runtime(),
        qualification=qualification,
        seed_array_sha256=runner.sha256(root / "initial_pose_xyz.npy"),
        evidence_hashes={str(candidate): runner.sha256(candidate)},
    )
    seal = dict(
        status="ok",
        stage="reflection_temporal_registration",
        outputs={},
        inputs={p: entry(Path(p)) for p in [str(candidate), str(code_path)]},
    )

    def save():
        dump(root / "plan.json", plan)
        seal["outputs"] = {
            "plan": entry(root / "plan.json"),
            "initial_pose": entry(root / "initial_pose_xyz.npy"),
        }
        dump(root / "registration.provenance.json", seal)

    save()
    return root, code, seed, candidate, plan, seal, save


def test_execution_verifier_only_deserializes_separate_seed(registration, monkeypatch):
    root, code, seed, candidate, _, _, _ = registration
    original = np.load
    loads = []

    def tracked_load(path, *args, **kwargs):
        loads.append(Path(path))
        assert Path(path) != candidate
        return original(path, *args, **kwargs)

    monkeypatch.setattr(runner.np, "load", tracked_load)
    _, actual, hashes = runner.load_registration(root, code)
    assert loads == [root / "initial_pose_xyz.npy"]
    assert hashes[str(candidate)] == runner.sha256(candidate)
    np.testing.assert_array_equal(actual, seed)


def test_registration_rejects_preflight_or_changed_seed(registration):
    root, code, _, _, plan, _, save = registration
    plan["preflight_only"] = True
    save()
    with pytest.raises(ValueError, match="registered conditions"):
        runner.load_registration(root, code)
    plan["preflight_only"] = False
    save()
    np.save(root / "initial_pose_xyz.npy", np.zeros((3, 3, 3)))
    with pytest.raises(ValueError, match="SHA differs"):
        runner.load_registration(root, code)


def test_registration_rejects_unbound_or_changed_historical_evidence(registration):
    root, code, _, candidate, _, seal, save = registration
    seal["inputs"].pop(str(candidate))
    save()
    with pytest.raises(ValueError, match="input set"):
        runner.load_registration(root, code)


def test_evaluation_refuses_one_saved_arm_before_reading_pixels(tmp_path):
    path = tmp_path / "direct_candidate.npz"
    path.write_bytes(b"first arm saved")
    record = SimpleNamespace(outputs={"direct_candidate": entry(path)})
    with pytest.raises(ValueError, match="both trajectories"):
        runner.evaluate_completed({}, None, None, None, tmp_path, record)
    assert not (tmp_path / "evaluation").exists()


def test_both_arms_run_sequentially_and_save_native_outputs(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "get_device_name", lambda *args: "fake device")
    for name in ("empty_cache", "reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(runner.torch.cuda, name, lambda: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(runner.torch.cuda, name, lambda: 0)
    monkeypatch.setattr(runner, "validate_arrays", lambda *args: None)

    def stream(reader):
        try:
            for frame in runner.FRAMES:
                yield SimpleNamespace(frame_idx=int(frame))
        finally:
            events.append("closed")

    monkeypatch.setattr(runner, "bounded_frames", stream)

    class FakeTracker:
        arm = "direct"

        def __init__(self, rig, seed, **kwargs):
            assert not np.any(seed)
            seed[:] = 12  # Constructor mutation must not contaminate the other arm.
            self.initialize_count = 1
            self.node_recovery_diagnostics = {}

        def track(self, observations):
            events.append(self.arm)
            frames = [v.frame_idx for v in observations]
            assert frames == runner.FRAMES.tolist()
            return {"frame_idx": np.array(frames)}

    class FakeFresnel(FakeTracker):
        arm = "fresnel"

        def __init__(self, *args, **kwargs):
            assert (tmp_path / "direct_candidate.npz").is_file()
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(runner, "PersistentNodeThreeBodyTracker", FakeTracker)
    monkeypatch.setattr(runner, "ReflectionPersistentNodeTracker", FakeFresnel)
    record = SimpleNamespace(outputs={})
    record.output = lambda role, path, sha256: record.outputs.update({role: entry(path)})
    plan = dict(
        config=runner.configs(), qualification={"runtime": {"cuda_device_name": "fake device"}}
    )
    seed = np.zeros((3, 3, 3))
    results, timings = runner.execute_arms(plan, seed, None, None, tmp_path, record)
    assert list(results) == ["direct", "fresnel"]
    assert events == ["direct", "closed", "fresnel", "closed"]
    assert not np.any(seed)
    for arm in runner.ARMS:
        assert timings[arm]["initialization_count"] == 1
        assert timings[arm]["reset_count"] == 0
        assert timings[arm]["native_frames_read"] == 200
        assert arm + "_candidate" in record.outputs


def statistics():
    row = dict(
        direct_chest_per_slot=[100, 100, 100],
        direct_chest_total=300,
        accepted_chest_per_slot=[80, 80, 80],
        accepted_chest_total=240,
        longest_chest_gap_per_slot=[2, 2, 2],
    )
    return {a: copy.deepcopy(row) for a in runner.ARMS}


def test_gate_requires_pose_improvement_not_branch_only():
    stats = statistics()
    assert runner.promotion_gate([[0.5, 0.4], [0.5, 0.3]], stats)["passed"]
    assert not runner.promotion_gate([[0.5, 0.4], [0.5, 0.4]], stats)["passed"]
    assert not runner.promotion_gate([[0.3, 0.5], [0.3, 0.4]], stats)["passed"]
    assert not runner.promotion_gate([[0.5, 0.4], [0.5, 0.4 - 5e-11]], stats)["passed"]


@pytest.mark.parametrize(
    "key,new",
    [
        ("direct_chest_per_slot", [99, 200, 100]),
        ("accepted_chest_per_slot", [79, 90, 90]),
        ("longest_chest_gap_per_slot", [3, 1, 1]),
    ],
)
def test_gate_cannot_cancel_slot_damage_with_total_gain(key, new):
    stats = statistics()
    stats["fresnel"][key] = new
    stats["fresnel"]["direct_chest_total"] = 399
    stats["fresnel"]["accepted_chest_total"] = 259
    assert not runner.promotion_gate([[0.5, 0.4], [0.5, 0.3]], stats)["passed"]


def test_metrics_preserve_native_half_open_gaps_and_terminal_review():
    observed = np.ones((200, 3, 3), bool)
    observed[:2, 0, 1] = False
    observed[-3:, 1, 1] = False
    review = np.zeros((200, 3), bool)
    review[-1, 2] = True
    xyz = np.where(observed[:, :, 1, None], np.ones((200, 3, 3)), np.nan)
    accepted = np.where(review[:, :, None], np.nan, xyz)
    result = runner.metrics(dict(observed=observed, identity_review=review, accepted_xyz=accepted))
    assert result["chest_gap_ranges"] == [[[79236, 79238]], [[79433, 79436]], []]
    assert result["review_ranges"][2] == [[79435, 79436]]
    assert result["direct_chest_total"] == 595
    assert result["accepted_chest_total"] == 594

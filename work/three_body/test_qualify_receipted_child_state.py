"""Small synthetic full-state CPU probe; no registered or experimental inputs."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import qualify_receipted_child_state as q
from test_receipted_long_runner_migration import bindings, synthetic_state


@pytest.fixture(autouse=True)
def no_tracking(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("state probe must not fit/advance/query CUDA")

    monkeypatch.setattr(q.m.original.AcceleratedContinuousPersistentNodeTracker, "_fit", forbidden)
    monkeypatch.setattr(
        q.m.original.AcceleratedContinuousPersistentNodeTracker, "advance", forbidden
    )
    monkeypatch.setattr(q.m.original.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(q.m, "execution_sources", forbidden)
    monkeypatch.setattr(q.runner, "prepare", forbidden)


def fixture():
    template, arrays, metadata = synthetic_state(count=3, restores=10)
    made = []

    def factory():
        tracker = type(template)(template.rig, template.initial.copy(), template.config)
        made.append(tracker)
        return tracker

    return SimpleNamespace(
        arrays=arrays,
        metadata=metadata,
        candidate=q.m.traced_candidate(arrays),
        factory=factory,
        made=made,
        binding={"schema": q.m.BINDING_SCHEMA, "synthetic_codec_binding": True},
    )


def qualify(f, output, **kwargs):
    return q.qualify_state(
        f.arrays, f.metadata, f.candidate, f.binding, output, f.factory, **kwargs
    )


def test_two_checkpoint_stages_preserve39_and_only_second_metadata_restores_once(tmp_path):
    f = fixture()
    before = (
        q.m.array_signatures(f.arrays),
        copy.deepcopy(f.metadata),
        q.m.array_signatures(f.candidate),
    )
    events = []
    result = qualify(
        f,
        tmp_path / "probe",
        on_checkpoint=lambda label, path, manifest: events.append((label, "checkpoint")),
        on_stage=lambda label, value: events.append((label, "checked")),
    )
    assert events == [
        ("transport", "checkpoint"),
        ("transport", "checked"),
        ("restore", "checkpoint"),
        ("restore", "checked"),
    ]
    assert len(f.made) == 2 and f.made[0] is not f.made[1]
    assert f.made[0].initialize_count == f.made[0].restore_count == f.made[0].processed_frames == 0
    assert f.made[1].initialize_count == 1 and f.made[1].restore_count == 11
    assert f.made[1].processed_frames == 3
    first, first_meta, _ = q.m.read_checkpoint(
        tmp_path / "probe/transport_checkpoint", expected_bindings=f.binding
    )
    second, second_meta, _ = q.m.read_checkpoint(
        tmp_path / "probe/restored_checkpoint", expected_bindings=f.binding
    )
    q.m.original._same_arrays(first, f.arrays)
    q.m.original._same_arrays(second, f.arrays)
    assert first_meta == f.metadata
    assert second_meta == {**f.metadata, "restore_count": 11}
    assert len(first) == len(second) == 39
    assert result["transport"]["logical_restore_calls"] == 0
    assert result["restore"]["logical_restore_calls"] == 1
    assert len(result["restore"]["all21_candidate_arrays_exact_backtrace"]) == 21
    assert result["restore"]["history_prefix"] == dict(
        parent_native_frames=3, parent_rows=18, all_history_prefix_exact=True
    )
    for stage in ("transport", "restore"):
        assert result[stage]["CPU"] is True and result[stage]["GPU"] is False
        assert result[stage]["tracking_performed"] is False
        assert result[stage]["ownership_verified"] is False
        assert result[stage]["independent_ID_truth"] is False
    assert before == (q.m.array_signatures(f.arrays), f.metadata, q.m.array_signatures(f.candidate))
    assert any(np.any(array) for key, array in first.items() if key == "state_velocity")
    assert first["state_review"][:, 2].all()
    for key in first:
        assert not np.shares_memory(first[key], second[key])


@pytest.mark.parametrize(
    "fault",
    [
        "missing_array",
        "beam",
        "restore",
        "initializations",
        "context",
        "candidate",
        "candidate_missing",
        "unselected_history",
    ],
)
def test_invalid_parent_or_candidate_stops_before_any_checkpoint(tmp_path, fault):
    f = fixture()
    if fault == "missing_array":
        del f.arrays["previous_images_bgr_top_cam"]
    elif fault == "beam":
        f.arrays["state_pose"] = f.arrays["state_pose"][:5]
    elif fault == "restore":
        f.metadata["restore_count"] = 9
    elif fault == "initializations":
        f.metadata["initialize_count"] = True
    elif fault == "context":
        f.metadata["context"]["tracker_config"]["motion_sigma_mm"] += 1
    elif fault == "candidate":
        f.candidate["pose_xyz"][0, 0, 0, 0] += 1
    elif fault == "candidate_missing":
        del f.candidate["accepted_xyz"]
    else:
        f.arrays["history_node_missing"][5, 0, 0] = 100
    with pytest.raises((ValueError, RuntimeError)):
        qualify(f, tmp_path / "probe")
    assert not (tmp_path / "probe/transport_checkpoint").exists()
    assert not any(tracker.restore_count for tracker in f.made)


@pytest.mark.parametrize("fault", ["wrong_class", "noncpu", "started", "same_object"])
def test_cpu_original_fresh_object_contract(tmp_path, fault):
    f = fixture()
    original_factory = f.factory
    if fault == "wrong_class":
        f.factory = lambda: SimpleNamespace(device=SimpleNamespace(type="cpu"))
    elif fault == "same_object":
        tracker = original_factory()
        f.factory = lambda: tracker
    else:

        def factory():
            tracker = original_factory()
            if fault == "noncpu":
                tracker.device = SimpleNamespace(type="cuda")
            else:
                tracker.initialize_count = 1
            return tracker

        f.factory = factory
    with pytest.raises(ValueError):
        qualify(f, tmp_path / "probe")
    assert not (tmp_path / "probe/restored_checkpoint").exists()


def test_interrupted_probe_retains_completed_transport_without_second_stage(tmp_path):
    f = fixture()
    failure = RuntimeError("synthetic interruption after transport")

    def stop(label, value):
        assert label == "transport" and value["metadata_unchanged"] is True
        raise failure

    with pytest.raises(RuntimeError) as caught:
        qualify(f, tmp_path / "probe", on_stage=stop)
    assert caught.value is failure
    assert (tmp_path / "probe/transport_checkpoint/manifest.json").is_file()
    assert not (tmp_path / "probe/restored_checkpoint").exists()
    assert len(f.made) == 1 and f.made[0].restore_count == 0


def test_transport_readback_metadata_drift_is_rejected_before_restore(monkeypatch, tmp_path):
    f = fixture()
    original_read = q.m.read_checkpoint
    calls = 0

    def changed(*args, **kwargs):
        nonlocal calls
        arrays, metadata, manifest = original_read(*args, **kwargs)
        calls += 1
        if calls == 2:
            metadata["restore_count"] += 1
        return arrays, metadata, manifest

    monkeypatch.setattr(q.m, "read_checkpoint", changed)
    with pytest.raises(ValueError, match="transport changed parent metadata"):
        qualify(f, tmp_path / "probe")
    assert len(f.made) == 1 and f.made[0].restore_count == 0


def test_original_parent_previous_image_mutation_by_callback_cannot_pass_final_guard(tmp_path):
    f = fixture()

    def mutate(label, value):
        if label == "restore":
            f.arrays["previous_images_bgr_top_cam"][0, 0, 0] += 1

    with pytest.raises(ValueError, match="caller parent arrays mutated"):
        qualify(f, tmp_path / "probe", on_stage=mutate)


def plan_fixture(monkeypatch, tmp_path):
    legacy, code = bindings(monkeypatch, tmp_path)
    current = q.m.execution_binding(legacy, code)
    plan = dict(
        schema="receipted_long_child_plan_v1",
        stage=q.m.STAGE,
        binding=current,
        parent_binding=copy.deepcopy(legacy),
        transition=q.m.transition(legacy, current, legacy),
        expected_initializations=1,
        expected_restores=11,
        expected_external_resets=0,
        full_interval=[q.m.FIRST, q.m.FULL_STOP],
        native_subsampling=False,
        serial_flow=True,
        same_original_accelerated_class=True,
        science_branch_change=False,
        processes_started=False,
        exclusive_execution_owner_is_external=True,
        next_native_interval=[136120, 146080],
        expected_previous_frame=136119,
        output_interval=[q.m.FIRST, 146080],
    )
    return legacy, plan


def test_registered_plan_only_accepts_first_finite_migration(monkeypatch, tmp_path):
    legacy, plan = plan_fixture(monkeypatch, tmp_path)
    q.validate_plan(plan, legacy)
    plan["parent_binding"] = copy.deepcopy(plan["binding"])
    with pytest.raises(ValueError, match="legacy parent"):
        q.validate_plan(plan, legacy)


@pytest.mark.parametrize("fault", ["source", "inputs", "restore", "class", "boundary"])
def test_registered_scientific_or_lifecycle_drift_is_rejected(monkeypatch, tmp_path, fault):
    legacy, plan = plan_fixture(monkeypatch, tmp_path)
    if fault == "source":
        plan["binding"]["code_sha256"][next(iter(legacy["code_sha256"]))] = "f" * 64
    elif fault == "inputs":
        plan["binding"]["inputs"]["seed"] = "other"
    elif fault == "restore":
        plan["expected_restores"] = 12
    elif fault == "class":
        plan["same_original_accelerated_class"] = False
    else:
        plan["expected_previous_frame"] += 1
    with pytest.raises(ValueError):
        q.validate_plan(plan, legacy)


def test_reserved_child_output_is_rejected_before_mkdir(monkeypatch, tmp_path):
    output = tmp_path / "planned_tracking"
    monkeypatch.setattr(q, "WORK", tmp_path)
    monkeypatch.setattr(q, "REGISTRATION_ROOT", tmp_path / "registration")
    monkeypatch.setattr(q.m, "REPO", tmp_path)
    monkeypatch.setattr(
        q.runner, "load_registration", lambda *_: ({"expected_output": str(output)}, {})
    )
    args = SimpleNamespace(
        registration=q.REGISTRATION_ROOT, registration_sha256=q.REGISTRATION_SHA, output=output
    )
    with pytest.raises(ValueError, match="occupy registered child"):
        q.run(args)
    assert not output.exists()


@pytest.mark.parametrize("bad", ["", "a" * 63, "A" * 64, "x" * 64, True])
def test_invalid_registration_sha_rejected_before_any_file_access(bad):
    assert not q.valid_sha(bad)
    with pytest.raises(ValueError, match="registration SHA256"):
        q.run(SimpleNamespace(registration_sha256=bad))

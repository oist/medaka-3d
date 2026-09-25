"""Small file-backed registration/execution orchestration, never actual GPU/fit/raw."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import run_receipted_long_continuation as r
from test_receipted_long_runner_migration import synthetic_state


class Store:
    def __init__(self, root):
        self.root, self.uses, self.evidence_hashes = root, [], {}
        self.guards = 0

    def final_guard(self):
        self.guards += 1


def fixture(monkeypatch, tmp_path):
    m = r.m
    monkeypatch.setattr(m, "REPO", tmp_path)
    source = tmp_path / "script.py"
    source.write_text("# synthetic source\n")
    code = {str(source): m.sha(source)}
    tracker, parent_arrays, parent_meta = synthetic_state(3)
    _, child_arrays, child_meta = synthetic_state(5, restores=8)
    store = Store(tmp_path / "cache")
    parent_root = tmp_path / "parent"
    parent_root.mkdir()
    binding = {"schema": m.BINDING_SCHEMA, "code_sha256": code}
    plan = dict(
        schema="receipted_long_child_plan_v1",
        stage=m.STAGE,
        expected_output=str(tmp_path / "run"),
        binding=binding,
        parent=str(parent_root),
        parent_seal_sha256="a" * 64,
        parent_audit={"path": str(tmp_path / "parent_audit.json"), "sha256": "b" * 64},
        next_native_interval=[m.FIRST + 3, m.FIRST + 5],
        receipt_cache_root=str(store.root),
        expected_restores=8,
        selected_chunk_records=[],
        transition={"kind": "synthetic"},
    )
    inputs = SimpleNamespace(
        rig=tracker.rig,
        initial_pose_xyz=tracker.initial,
        input_root=tmp_path,
        protocol={"experiment": "synthetic"},
        frame_idx=np.arange(m.FIRST, m.FIRST + 5),
        chunk_records=[],
    )
    protocol = dict(
        tracker_config=copy.deepcopy(parent_meta["context"]["tracker_config"]),
        node_recovery_config=copy.deepcopy(parent_meta["context"]["recovery_config"]),
        persistence_config=copy.deepcopy(parent_meta["context"]["persistence_config"]),
    )
    context = SimpleNamespace(
        code=code,
        inputs=inputs,
        parent={"arrays": parent_arrays, "metadata": parent_meta},
        store=store,
        protocol=protocol,
        legacy={},
    )
    events = []

    def prepare(args, evidence, *, store_ready=None, existing_output=None):
        events.append("prepare")
        evidence.add(source, code[str(source)])
        if store_ready:
            store_ready(store)
        if existing_output is not None:
            assert existing_output == tmp_path / "run" and not any(existing_output.iterdir())
        return copy.deepcopy(plan), context

    monkeypatch.setattr(r, "prepare", prepare)
    monkeypatch.setattr(m, "execution_sources", lambda *_: code)
    monkeypatch.setattr(r, "configure_runtime", lambda *_: events.append("runtime"))
    monkeypatch.setattr(r.old.torch.cuda, "is_available", lambda: pytest.fail("no actual CUDA"))

    def advance(actual, actual_inputs, start, progress):
        assert actual_inputs is inputs and start == m.FIRST + 3
        assert actual.previous_frame == start - 1 and actual.restore_count == 8
        events.append("advance_stub")
        history, states, obs, _ = actual._decode_state(child_arrays, child_meta)
        actual._history, actual._states, actual._previous_observation = history, states, obs
        actual._previous_frame = m.FIRST + 4
        actual._recovery_calls = 5
        progress.write_text('{"synthetic_only":true}\n')
        return 1.0, 2.0

    monkeypatch.setattr(r.old, "_advance_chunks", advance)
    monkeypatch.setattr(r.old, "CachedNodes", lambda *_: None)
    monkeypatch.setattr(r.old.base, "validate_result", lambda *_: events.append("validate_result"))
    monkeypatch.setattr(r.old, "validate_arm_arrays", lambda *_: events.append("validate_arm"))
    return SimpleNamespace(
        m=m,
        source=source,
        code=code,
        store=store,
        plan=plan,
        context=context,
        events=events,
        prepare=prepare,
    )


def register_fixture(state, tmp_path):
    r.register(SimpleNamespace(output=tmp_path / "registration"))
    path = tmp_path / "registration/registration.provenance.json"
    assert state.m.read(path)["status"] == "ok"
    return SimpleNamespace(
        registration=path.parent, registration_sha256=state.m.sha(path), output=tmp_path / "run"
    )


def test_sealed_registration_then_complete_original_class_codec_and_backtrace(
    monkeypatch, tmp_path
):
    state = fixture(monkeypatch, tmp_path)
    args = register_fixture(state, tmp_path)
    summary = r.execute(args)
    assert summary["logical_initialization_count"] == 1
    assert summary["checkpoint_restore_count"] == 8
    assert summary["restores_during_phase"] == summary["external_pose_reset_count"] == 0
    assert summary["all_39_checkpoint_arrays_verified"]
    assert len(summary["all_21_candidate_arrays_exact_backtrace"]) == 21
    assert summary["history_prefix"]["parent_rows"] == 18
    assert summary["independent_phase_audit_complete"] is False
    seal = state.m.read(args.output / "continuation.provenance.json")
    assert seal["stage"] == state.m.STAGE and seal["status"] == "ok"
    assert seal["params"]["bindings"] == state.plan["binding"]
    assert state.events == [
        "prepare",
        "prepare",
        "runtime",
        "advance_stub",
        "validate_result",
        "validate_arm",
    ]
    assert state.store.guards >= 2
    for item in seal["outputs"].values():
        assert state.m.sha(item["path"]) == item["sha256"]
    with np.load(args.output / "candidate.npz", allow_pickle=False) as data:
        assert np.isnan(data["accepted_xyz"][:, 2]).all()
    with pytest.raises(ValueError, match="exclusive"):
        r.execute(args)


@pytest.mark.parametrize("failure", ["changed_plan", "source_after_fit", "runtime", "advance"])
def test_failure_is_sealed_and_no_unproven_success_or_second_step(monkeypatch, tmp_path, failure):
    state = fixture(monkeypatch, tmp_path)
    args = register_fixture(state, tmp_path)
    if failure == "changed_plan":
        state.plan["expected_restores"] = 9
    elif failure == "runtime":
        monkeypatch.setattr(
            r, "configure_runtime", lambda *_: (_ for _ in ()).throw(ValueError("runtime mismatch"))
        )
    elif failure == "advance":
        monkeypatch.setattr(
            r.old,
            "_advance_chunks",
            lambda *_: (_ for _ in ()).throw(RuntimeError("synthetic advance failure")),
        )
    else:
        monkeypatch.setattr(
            r.old,
            "validate_arm_arrays",
            lambda *_: state.source.write_text("# changed after computation\n"),
        )
    with pytest.raises((ValueError, RuntimeError)):
        r.execute(args)
    seal = state.m.read(args.output / "continuation.provenance.json")
    assert seal["status"] == "failed"
    assert seal["params"].get("all_input_source_snapshot_output_sha_unchanged") is not True
    assert "mask_validation_receipts" in seal["outputs"]
    assert str(state.source) in seal["inputs"]
    if failure in {"changed_plan", "runtime"}:
        assert "advance_stub" not in state.events
        assert not (args.output / "candidate.npz").exists()


def test_registration_failure_retains_receipt_usage_and_source_evidence(monkeypatch, tmp_path):
    state = fixture(monkeypatch, tmp_path)

    def failed(args, evidence, **kwargs):
        state.prepare(args, evidence, **kwargs)
        raise ValueError("synthetic input rejection")

    monkeypatch.setattr(r, "prepare", failed)
    output = tmp_path / "failed_registration"
    with pytest.raises(ValueError, match="input rejection"):
        r.register(SimpleNamespace(output=output))
    seal = state.m.read(output / "registration.provenance.json")
    assert seal["status"] == "failed" and "mask_validation_receipts" in seal["outputs"]
    assert str(state.source) in seal["inputs"]
    assert not (output / "plan.json").exists()
    assert "runtime" not in state.events


@pytest.mark.parametrize(
    "failure", ["digest", "payload", "failed_status", "wrong_stage", "foreign_role"]
)
def test_registration_rejection_happens_before_runtime(monkeypatch, tmp_path, failure):
    state = fixture(monkeypatch, tmp_path)
    args = register_fixture(state, tmp_path)
    seal_path = args.registration / "registration.provenance.json"
    if failure == "digest":
        args.registration_sha256 = "0" * 64
    elif failure == "payload":
        (args.registration / "plan.json").write_text("{}")
    else:
        seal = state.m.read(seal_path)
        if failure == "failed_status":
            seal["status"] = "failed"
        if failure == "wrong_stage":
            seal["stage"] = "other"
        if failure == "foreign_role":
            seal["inputs"]["other"] = seal["inputs"].pop(str(state.source))
        seal_path.write_bytes(state.m._canonical(seal))
        args.registration_sha256 = state.m.sha(seal_path)
    with pytest.raises(ValueError):
        r.execute(args)
    assert state.events == ["prepare"]
    assert not args.output.exists()


def test_cli_does_not_allow_implicit_unregistered_execution(tmp_path):
    with pytest.raises(SystemExit) as exc:
        r.main(["--registration", str(tmp_path), "--output", str(tmp_path / "run")])
    assert exc.value.code == 2


def test_original_frame_loop_retained_without_copying_or_parallel_flow():
    import run_long_continuation as old

    assert r.old._advance_chunks is old._advance_chunks
    assert (
        r.old.AcceleratedContinuousPersistentNodeTracker
        is old.AcceleratedContinuousPersistentNodeTracker
    )


def test_receipt_final_guard_is_enforced_after_consumption(monkeypatch, tmp_path):
    state = fixture(monkeypatch, tmp_path)
    args = register_fixture(state, tmp_path)
    calls = 0

    def guard():
        nonlocal calls
        calls += 1
        raise ValueError("receipt replaced after computation")

    state.store.final_guard = guard
    with pytest.raises(ValueError, match="receipt replaced"):
        r.execute(args)
    seal = state.m.read(args.output / "continuation.provenance.json")
    assert calls == 1 and seal["status"] == "failed"
    assert "candidate" in seal["outputs"] and "mask_validation_receipts" in seal["outputs"]
    assert str(state.source) in seal["inputs"]

"""File-backed operational contracts with a fake child; never launch a process."""

from __future__ import annotations

import copy
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FunctionType, SimpleNamespace

import pytest

WORK = Path(__file__).resolve().parent
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

import receipted_long_phase_operation as module  # noqa: E402
import run_long_remaining_supervisor as original  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def entry(path):
    return {"path": str(path), "size": path.stat().st_size, "sha256": module.sha256(path)}


def fake_tracker(root, stop, binding, *, started=None):
    root.mkdir()
    (root / "candidate.npz").write_bytes(b"file-binding fixture; not trajectory data")
    write(root / "summary.json", dict(
        schema="long_continuation_phase_v1", technical_completion=True,
        processed_interval=[49800, stop], logical_initialization_count=1,
        external_pose_reset_count=0, active_beam_count=6))
    write(root / "continuation.provenance.json", dict(
        status="ok", stage="three_body_long_continuation",
        started_at=started or datetime.now(UTC).isoformat(),
        params=dict(through_frame=stop, bindings=binding,
                    all_input_source_snapshot_output_sha_unchanged=True),
        outputs={"candidate": entry(root / "candidate.npz"),
                 "summary": entry(root / "summary.json")}))


def fake_parent_audit(root, parent, stage):
    root.mkdir()
    seal_name, summary_name, role = module.PARENT_AUDIT_STAGES[stage]
    write(root / summary_name, dict(status="ok", run=str(parent),
                                   candidate_sha256=module.sha256(parent / "candidate.npz")))
    (root / "other_audited_output.bin").write_bytes(b"all outputs must remain bound")
    write(root / seal_name, dict(
        stage=stage, status="ok", params={"all_input_source_snapshot_output_sha_unchanged": True},
        inputs={str(parent / name): entry(parent / name)
                for name in ("candidate.npz", "continuation.provenance.json")},
        outputs={role: entry(root / summary_name),
                 "other": entry(root / "other_audited_output.bin")}))


@pytest.fixture
def fixture(tmp_path):
    parent, phase, cache = (tmp_path / name for name in ("parent", "phase", "cache"))
    phase.mkdir()
    seed, inputs, stage = (tmp_path / name for name in ("seed", "inputs", "stage"))
    for root in (seed, inputs, stage):
        root.mkdir()
    (seed / "registration.json").write_bytes(b"seed path fixture")
    binding = {"inputs": {"full_frame_range": [49800, 348600],
                          "seed": {"registration_path": str(seed / "registration.json"),
                                   "input_root": str(inputs)}, "input_stage_root": str(stage)}}
    fake_tracker(parent, 66400, binding)
    audit_root = tmp_path / "parent_audit"
    fake_parent_audit(audit_root, parent, "long_supervised_phase_operation")
    code_path = tmp_path / "operation_source.py"
    code_path.write_bytes(b"# isolated test source\n")
    code = {str(code_path): module.sha256(code_path)}
    stores, calls, events = [], [], []
    outcome = dict(exit=0, old_child_time=False)
    result = dict(schema="long_continuation_generic_phase_audit_v1", status="ok",
                  frame_range=[49800, 76360], counts={"fixture": 1})

    class Store:
        def __init__(self, root):
            self.root = Path(root)
            self.root.mkdir(exist_ok=True)
            path = self.root / f"fixture_receipt_{len(stores)}.json"
            path.write_bytes(b"isolated receipt fixture")
            self.evidence_hashes = {str(path): module.sha256(path)}
            self.uses = []
            stores.append(self)

        def final_guard(self):
            module.unchanged(self.evidence_hashes)

    def prepare(parent, stop, child, evidence, *, receipt_store):
        calls.append("prepare")
        receipt_store.uses.append({"phase": "prepare", "cache_hit": False})
        for path, digest in receipt_store.evidence_hashes.items():
            evidence.add(path, digest)
        plan = dict(schema="long_continuation_registered_phase_v1", parent_phase=str(parent),
                    expected_output=str(child), output_interval=[49800, stop],
                    next_native_interval=[66400, stop], frozen_input_and_code_binding=binding,
                    expected_initializations=1, expected_external_resets=0, parent_active_beam=6)
        return plan, {"unchanged": "context"}

    def verify(child, plan, evidence, *, receipt_store):
        calls.append("verify")
        receipt_store.uses.append({"phase": "verify", "cache_hit": True})
        for path, digest in receipt_store.evidence_hashes.items():
            evidence.add(path, digest)
        return result

    def child(command, log, info, *, on_started):
        calls.append("child")
        info.update(pid=123456789, started_at=datetime.now(UTC).isoformat(),
                    returncode=outcome["exit"], child_alive_after_return=False)
        log.write_bytes(b"fake child only; no process was launched")
        on_started(info)
        if not outcome["exit"]:
            fake_tracker(phase / "tracking", 76360, binding,
                         started="2000-01-01T00:00:00+00:00" if outcome["old_child_time"] else None)
        return outcome["exit"]

    namespace = dict(vars(module))
    namespace.update(
        ReceiptStore=Store, sources=lambda: code,
        backend=SimpleNamespace(Digests=module.backend.Digests, prepare_phase=prepare,
                                verify_phase=verify), run_child=child,
    )
    # Use the same snapshot code with an isolated fixture root, avoiding a
    # doubled pytest temp path beyond the Windows legacy path limit.
    namespace["snapshot"] = FunctionType(
        original.snapshot.__code__, {**vars(original), "REPO": tmp_path}, "snapshot")
    for name in ("bind_parent_audit", "_save_receipts", "register_phase",
                 "_verify_registration", "execute_phase"):
        f = getattr(module, name)
        namespace[name] = FunctionType(f.__code__, namespace, name, f.__defaults__)
    return SimpleNamespace(parent=parent, phase=phase, cache=cache, audit=audit_root, code=code,
                           binding=binding, stores=stores, calls=calls, events=events,
                           outcome=outcome, result=result, namespace=namespace)


def register(f):
    return f.namespace["register_phase"](
        f.parent, 76360, f.phase, f.code, f.binding, f.audit, receipt_root=f.cache)


def execute(f, registered):
    plan, child, registration, command = registered
    return f.namespace["execute_phase"](
        f.phase, plan, child, registration, command, f.code,
        lambda name, **fields: f.events.append((name, fields)), receipt_root=f.cache)


def test_new_stages_exact_command_plan_result_and_separate_fresh_receipts(fixture):
    f = fixture
    registered = register(f)
    plan, child, registration, command = registered
    saved_plan = copy.deepcopy(plan)
    assert command == original.command_for(f.parent, 76360, child, f.binding)
    assert module.read(registration / "plan.json") == saved_plan
    assert not any("receipt" in key for key in plan)
    assert f.calls == ["prepare"]
    assert execute(f, registered) == f.result
    assert plan == saved_plan and len(f.stores) == 2 and f.stores[0] is not f.stores[1]
    assert f.calls == ["prepare", "child", "verify"]
    for root, seal_name, expected, phase in (
        (registration, "registration.provenance.json", module.REGISTRATION_STAGE, "prepare"),
        (f.phase, "operation.provenance.json", module.OPERATION_STAGE, "verify"),
    ):
        seal = module.read(root / seal_name)
        assert seal["stage"] == expected and seal["status"] == "ok"
        assert seal["params"]["all_input_source_snapshot_output_sha_unchanged"] is True
        assert module.read(root / "mask_validation_receipts.json")["phase"] == phase
        for item in seal["outputs"].values():
            assert module.sha256(item["path"]) == item["sha256"]
    operation = module.read(f.phase / "operation.provenance.json")
    for store in f.stores:
        assert set(store.evidence_hashes).issubset(operation["inputs"])
    assert module.read(f.phase / "audit.json") == f.result
    assert [event[0] for event in f.events] == [
        "phase_start", "owned_child_started", "phase_complete"]


@pytest.mark.parametrize("stage", list(module.PARENT_AUDIT_STAGES))
def test_genuine_parent_stage_allowlist_without_translation(fixture, stage):
    root = fixture.phase / "another_audit"
    fake_parent_audit(root, fixture.parent, stage)
    evidence = module.backend.Digests()
    result = module.bind_parent_audit(fixture.parent, root, evidence)
    assert result["stage"] == stage
    assert str(root / "other_audited_output.bin") in evidence.paths
    evidence.final_guard()


@pytest.mark.parametrize("bad", ["failed", "candidate", "guard"])
def test_new_handoff_bootstrap_keeps_failure_candidate_and_guard_checks(fixture, bad):
    root = fixture.phase / "handoff_audit"
    fake_parent_audit(root, fixture.parent, "long_receipted_handoff_bootstrap_v1")
    path = root / "audit.provenance.json"
    seal = module.read(path)
    if bad == "failed":
        seal["status"] = "failed"
    elif bad == "candidate":
        (fixture.parent / "candidate.npz").write_bytes(b"different parent candidate")
    else:
        seal["params"]["all_input_source_snapshot_output_sha_unchanged"] = False
    write(path, seal)
    with pytest.raises(ValueError):
        module.bind_parent_audit(fixture.parent, root, module.backend.Digests())


@pytest.mark.parametrize("bad", ["stage", "failed", "candidate", "other_output", "input"])
def test_invalid_parent_audit_never_registers(fixture, bad):
    f = fixture
    path = f.audit / "operation.provenance.json"
    seal = module.read(path)
    if bad == "stage":
        seal["stage"] = "a fabricated old success"
    elif bad == "failed":
        seal["status"] = "failed"
    elif bad == "candidate":
        (f.parent / "candidate.npz").write_bytes(b"changed")
    elif bad == "other_output":
        (f.audit / "other_audited_output.bin").write_bytes(b"changed")
    else:
        seal["inputs"] = {}
    write(path, seal)
    with pytest.raises(ValueError):
        register(f)
    assert not f.calls and not (f.phase / "registration").exists()


@pytest.mark.parametrize("bad", ["parent_audit_output", "prepare_receipt", "future_registration",
                                 "old_registration_stage", "command"])
def test_changed_registration_evidence_or_future_time_stops_before_child(fixture, bad):
    f = fixture
    registered = register(f)
    registration = registered[2]
    if bad == "parent_audit_output":
        (f.audit / "other_audited_output.bin").write_bytes(b"changed after registration")
    elif bad == "prepare_receipt":
        Path(next(iter(f.stores[0].evidence_hashes))).write_bytes(b"changed receipt")
    elif bad == "command":
        registered[3].append("--unregistered-flag")
    else:
        path = registration / "registration.provenance.json"
        seal = module.read(path)
        if bad == "future_registration":
            seal["finished_at"] = (datetime.now(UTC) + timedelta(days=1)).isoformat()
        else:
            seal["stage"] = "register_long_supervised_phase"
        write(path, seal)
    with pytest.raises(ValueError):
        execute(f, registered)
    assert f.calls == ["prepare"]
    assert module.read(f.phase / "operation.provenance.json")["status"] == "failed"


def test_nonzero_child_exit_is_fail_fast_and_cannot_retry(fixture):
    f = fixture
    registered = register(f)
    f.outcome["exit"] = 7
    with pytest.raises(ValueError, match="tracking child failed"):
        execute(f, registered)
    assert f.calls == ["prepare", "child"] and len(f.stores) == 1
    assert not (f.phase / "audit.json").exists()
    assert module.read(f.phase / "child_execution.json")["returncode"] == 7
    seal_path = f.phase / "operation.provenance.json"
    old_digest = module.sha256(seal_path)
    assert module.read(seal_path)["status"] == "failed"
    with pytest.raises(ValueError, match="already attempted"):
        execute(f, registered)
    assert module.sha256(seal_path) == old_digest and f.calls == ["prepare", "child"]


def test_child_seal_cannot_predate_registration(fixture):
    f = fixture
    registered = register(f)
    f.outcome["old_child_time"] = True
    with pytest.raises(ValueError, match="registered after the child started"):
        execute(f, registered)
    assert f.calls == ["prepare", "child"]
    assert not (f.phase / "audit.json").exists()

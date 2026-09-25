"""Operational fixtures only: no real witness, NPZ decode, process, GPU or detector."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

import pytest

WORK = Path(__file__).resolve().parent
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

import run_long_receipted_supervisor as module  # noqa: E402
import run_long_remaining_supervisor as old  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def entry(path):
    return dict(path=str(path), size=path.stat().st_size, sha256=module.sha256(path))


def tracker(root, stop, binding, restores):
    root.mkdir(parents=True)
    (root / "candidate.npz").write_bytes(b"not an NPZ; wrapper never decodes a candidate")
    summary = dict(
        schema="long_continuation_phase_v1", technical_completion=True,
        processed_interval=[49800, stop], logical_initialization_count=1,
        checkpoint_restore_count=restores, external_pose_reset_count=0, active_beam_count=6,
        full_sequence_complete=stop == 348600, prefix_backtrace_provisional=stop != 348600)
    write(root / "summary.json", summary)
    write(root / "continuation.provenance.json", dict(
        status="ok", stage="three_body_long_continuation",
        params=dict(through_frame=stop, bindings=binding,
                    all_input_source_snapshot_output_sha_unchanged=True),
        outputs={"candidate": entry(root / "candidate.npz"),
                 "summary": entry(root / "summary.json")}))
    return summary


def audit_summary(parent, stop, restores):
    return dict(
        status="ok", run=str(parent), frame_range=[49800, stop],
        candidate_sha256=module.sha256(parent / "candidate.npz"),
        all_39_checkpoint_arrays_verified=True,
        all_21_candidate_arrays_exact_backtrace={str(i): {"fixture": True} for i in range(21)},
        history={"active_beam_count": 6}, logical_initialization_count=1,
        checkpoint_restore_count=restores, external_pose_reset_count=0,
        full_sequence_complete=stop == 348600, prefix_backtrace_provisional=stop != 348600)


def audit_files(root, parent, stop, restores, *, bootstrap=False):
    root.mkdir(exist_ok=True)
    seal_name, json_name, role, stage = (
        ("audit.provenance.json", "summary.json", "summary", "long_receipted_handoff_bootstrap_v1")
        if bootstrap else ("operation.provenance.json", "audit.json", "audit",
                           module.operation.OPERATION_STAGE))
    result = audit_summary(parent, stop, restores)
    write(root / json_name, result)
    write(root / "mask_validation_receipts.json", {"uses": [{"fixture": True}]})
    write(root / seal_name, dict(
        status="ok", stage=stage, params={"all_input_source_snapshot_output_sha_unchanged": True},
        inputs={str(parent / name): entry(parent / name)
                for name in ("candidate.npz", "continuation.provenance.json")},
        outputs={role: entry(root / json_name),
                 "mask_validation_receipts": entry(root / "mask_validation_receipts.json")}))
    return result


@pytest.fixture
def fixture(tmp_path):
    paths = {name: tmp_path / name for name in ("parent", "output", "cache", "witness", "stage",
                                               "seed", "inputs")}
    for name in ("witness", "stage", "seed", "inputs"):
        paths[name].mkdir()
    witness = paths["witness"] / "completion.json"
    write(witness, {"status": "ok"})
    (paths["seed"] / "registration.json").write_bytes(b"seed path only")
    code_path = tmp_path / "fixture_source.py"
    code_path.write_bytes(b"# test source\n")
    code = {str(code_path): module.sha256(code_path)}
    binding = dict(
        code_sha256={str(module.RUNNER): module.sha256(module.RUNNER)},
        inputs=dict(full_frame_range=[49800, 348600], input_stage_root=str(paths["stage"]),
                    seed=dict(registration_path=str(paths["seed"] / "registration.json"),
                              input_root=str(paths["inputs"]))))
    tracker(paths["parent"], 76360, binding, 4)
    old_lock = tmp_path / "old.lock"
    old_lock.write_bytes(b"old owner evidence remains unchanged")
    control = dict(stop_at=None, child_error=False, bad_restore=False, bad_full=False,
                   stop_bytes=b"planned improvement at a completed boundary\n", ready=True,
                   corrupt_input=False, corrupt_source=False, bad_audit=False,
                   bootstrap_error=False, keyboard_interrupt=False)
    calls, commands = [], []
    namespace = dict(vars(module))
    lock = tmp_path / "new_shared.lock"

    def request():
        (paths["output"] / module.STOP_NAME).write_bytes(control["stop_bytes"])

    def verify(witness_root, evidence):
        calls.append("verify_handoff")
        assert witness_root == paths["witness"]
        evidence.add(witness)
        module.require(module.read(witness)["status"] == "ok", "unfinished handoff")
        assert not lock.exists()  # completed evidence is inspected before ownership
        return dict(parent=paths["parent"], binding=binding,
                    lineage={"fixture": "no process operations"})

    def bootstrap(witness_root, root, actual_code, *, receipt_root):
        calls.append("bootstrap")
        assert lock.exists() and receipt_root == paths["cache"] and actual_code == code
        if control["bootstrap_error"]:
            root.mkdir()
            write(root / "audit.provenance.json", dict(
                stage="long_receipted_handoff_bootstrap_v1", status="failed"))
            raise ValueError("bootstrap semantic validation failed")
        audit_files(root, paths["parent"], 76360, 4, bootstrap=True)
        if control["stop_at"] == "bootstrap":
            request()
        if control["corrupt_input"]:
            witness.write_bytes(b"input changed after verification")
        if control["corrupt_source"]:
            code_path.write_bytes(b"source changed during bootstrap")
        return root

    def register(parent, stop, root, actual_code, actual_binding, parent_audit, *, receipt_root):
        calls.append("register")
        assert actual_code == code and actual_binding == binding and receipt_root == paths["cache"]
        # The fake backend still uses the actual strict parent-audit binding helper.
        evidence = module.Digests()
        lineage = module.operation.bind_parent_audit(parent, parent_audit, evidence)
        evidence.final_guard()
        first = module.read(parent / "summary.json")["processed_interval"][1]
        child, registration = root / "tracking", root / "registration"
        registration.mkdir()
        plan = dict(parent_phase=str(parent), expected_output=str(child),
                    next_native_interval=[first, stop], output_interval=[49800, stop],
                    frozen_input_and_code_binding=binding)
        command = old.command_for(parent, stop, child, binding)
        commands.append(command)
        write(registration / "plan.json", plan)
        write(registration / "execution.json", {"command": command, "parent_audit": lineage})
        write(registration / "mask_validation_receipts.json", {"uses": ["prepare fixture"]})
        write(registration / "registration.provenance.json", dict(
            stage=module.operation.REGISTRATION_STAGE, status="ok"))
        if control["stop_at"] == "registration":
            request()
        return plan, child, registration, command

    def execute(root, plan, child, registration, command, actual_code, emit, *, receipt_root):
        calls.append("execute")
        assert command == old.command_for(Path(plan["parent_phase"]), plan["output_interval"][1],
                                          child, binding)
        assert actual_code == code and receipt_root == paths["cache"]
        emit("owned_child_started", pid=987654321, fixture=True)
        (root / "tracking.log").write_bytes(b"fake child log; no OS process")
        write(root / "child_execution.json", {"fixture": True})
        if control["child_error"]:
            write(root / "operation.provenance.json", dict(
                stage=module.operation.OPERATION_STAGE, status="failed"))
            raise ValueError("fake child nonzero exit")
        if control["keyboard_interrupt"]:
            write(root / "operation.provenance.json", dict(
                stage=module.operation.OPERATION_STAGE, status="failed"))
            raise KeyboardInterrupt("fake operation already performed its owned-child cleanup")
        parent = Path(plan["parent_phase"])
        count = module.read(parent / "summary.json")["checkpoint_restore_count"] + 1
        stop = plan["output_interval"][1]
        actual_count = count - 1 if control["bad_restore"] else count
        tracker(child, stop, binding, actual_count)
        if control["bad_full"]:
            summary = module.read(child / "summary.json")
            summary["full_sequence_complete"] = not summary["full_sequence_complete"]
            write(child / "summary.json", summary)
            seal = module.read(child / "continuation.provenance.json")
            seal["outputs"]["summary"] = entry(child / "summary.json")
            write(child / "continuation.provenance.json", seal)
        result = audit_files(root, child, stop, actual_count)
        if control["bad_audit"]:
            result["history"]["active_beam_count"] = 1
        if control["stop_at"] == "child":
            request()  # It is observed only after this child and audit return.
        return result

    def hints(stage, first, stop):
        calls.append("chunk_hints")
        assert stage == paths["stage"]
        return {(i, i + 1660) for i in range(first, stop, 1660)} if control["ready"] else set()

    namespace.update(
        handoff=SimpleNamespace(verify_handoff=verify, audit_handoff=bootstrap),
        operation=SimpleNamespace(register_phase=register, execute_phase=execute,
                                  bind_parent_audit=module.operation.bind_parent_audit),
        sources=lambda: code, SHARED_LOCK=lock, successful_chunk_hints=hints,
        snapshot=FunctionType(old.snapshot.__code__, {**vars(old), "REPO": tmp_path}, "snapshot"))
    for name in ("phase_ranges", "acquire_lock", "checked_header", "validate_audit",
                 "capture_stop_request", "wait_for_ready", "collect_phase_outputs",
                 "result_summary", "supervise", "main"):
        f = getattr(module, name)
        clone = FunctionType(f.__code__, namespace, name, f.__defaults__)
        clone.__kwdefaults__ = f.__kwdefaults__
        namespace[name] = clone
    args = SimpleNamespace(witness_root=paths["witness"], output=paths["output"],
                           receipt_root=paths["cache"], execute=True)
    return SimpleNamespace(**paths, witness_file=witness, code=code, code_path=code_path,
                           binding=binding, old_lock=old_lock, lock=lock, calls=calls,
                           commands=commands, control=control, namespace=namespace,
                           request=request, args=args)


def run(f, *, sleeper=lambda seconds: pytest.fail("unexpected waiting")):
    return f.namespace["supervise"](f.args, sleeper=sleeper)


def seal(f):
    return module.read(f.output / "supervisor.provenance.json")


def test_full_28_phase_chain_command_binding_receipts_and_final_two_chunks(fixture):
    f = fixture
    old_digest = module.sha256(f.old_lock)
    assert run(f) == f.output
    summary = module.read(f.output / "summary.json")
    assert summary["status"] == "complete" and summary["through"] == 348600
    assert summary["full_sequence_complete"] is True
    assert summary["prefix_backtrace_provisional"] is False
    assert len(summary["phases"]) == 28 and summary["checkpoint_restore_count"] == 32
    assert f.calls[:2] == ["verify_handoff", "bootstrap"]
    assert f.calls.count("execute") == 28 and len(f.commands) == 28
    assert module.phase_ranges()[-1] == [345280, 348600]
    assert all(command[0:3] == [str(old.PYTHON.resolve()), "-B", str(old.RUNNER.resolve())]
               for command in f.commands)
    assert module.sha256(f.old_lock) == old_digest and f.lock.exists()
    record = seal(f)
    assert record["stage"] == module.STAGE and record["status"] == "ok"
    assert record["params"]["all_input_source_snapshot_output_sha_unchanged"] is True
    assert record["params"]["full_sequence_complete"] is True
    assert len([x for x in record["outputs"] if x.endswith("mask_validation_receipts.json")]) == 57
    for item in record["outputs"].values():
        assert module.sha256(item["path"]) == item["sha256"]


@pytest.mark.parametrize("where,executed,registered,stop", [
    ("bootstrap", 0, 0, 76360), ("registration", 0, 1, 76360), ("child", 1, 1, 86320)])
def test_stop_at_safe_boundaries_keeps_partial_result(fixture, where, executed, registered, stop):
    f = fixture
    f.control["stop_at"] = where
    run(f)
    summary = module.read(f.output / "summary.json")
    assert summary["status"] == "stopped_after_completed_phase"
    assert summary["through"] == stop and not summary["full_sequence_complete"]
    assert summary["prefix_backtrace_provisional"]
    assert f.calls.count("execute") == executed and f.calls.count("register") == registered
    assert summary["stop_request"]["child_interrupted"] is False
    assert (f.output / "honored_stop_request.bin").read_bytes() == f.control["stop_bytes"]
    assert seal(f)["status"] == "ok" and f.lock.exists()


def test_stop_while_waiting_no_child_and_no_poll_spam(fixture):
    f = fixture
    f.control["ready"] = False
    sleeps = []
    def sleeper(seconds):
        sleeps.append(seconds)
        f.request()
    run(f, sleeper=sleeper)
    assert sleeps == [30] and "register" not in f.calls and "execute" not in f.calls
    events = [json.loads(x) for x in (f.output / "events.jsonl").read_text().splitlines()]
    assert [x["event"] for x in events] == ["handoff_bootstrap_complete", "stop_request_honored"]


def test_stop_arbitrary_bytes_preserved_without_json_nan(fixture):
    f = fixture
    f.control.update(stop_at="bootstrap", stop_bytes=b"reason \xff\x00 NaN\n")
    run(f)
    assert (f.output / "honored_stop_request.bin").read_bytes() == f.control["stop_bytes"]
    assert module.read(f.output / "stop_request.json")["sha256"] == module.sha256(
        f.output / module.STOP_NAME)


@pytest.mark.parametrize("status", ["running", "failed"])
def test_unfinished_or_failed_handoff_never_lock_or_launch(fixture, status):
    f = fixture
    write(f.witness_file, {"status": status})
    with pytest.raises(ValueError, match="unfinished handoff"):
        run(f)
    assert not f.lock.exists() and not f.output.exists() and "bootstrap" not in f.calls


def test_global_lock_excludes_two_output_roots_and_keeps_old_lock(fixture):
    f = fixture
    f.control["stop_at"] = "bootstrap"
    run(f)
    first = f.lock.read_bytes()
    f.args.output = f.output.parent / "different_output"
    with pytest.raises(ValueError, match="shared receipted supervisor lock"):
        run(f)
    assert f.lock.read_bytes() == first and not f.args.output.exists()
    assert f.old_lock.read_bytes() == b"old owner evidence remains unchanged"


def test_lock_exclusive_creation_handles_race(fixture):
    f = fixture
    f.namespace["acquire_lock"](f.output, f.parent)
    with pytest.raises(FileExistsError):
        f.namespace["acquire_lock"](f.output.parent / "second", f.parent)


@pytest.mark.parametrize("problem,match", [
    ("child_error", "fake child nonzero"), ("bad_restore", "restore did not increase"),
    ("bad_full", "completion flags"), ("bad_audit", "phase audit differs")])
def test_failed_phase_never_advances_or_retries_and_partial_outputs_remain(fixture, problem, match):
    f = fixture
    f.control[problem] = True
    with pytest.raises(ValueError, match=match):
        run(f)
    summary = module.read(f.output / "summary.json")
    assert summary["status"] == "failed" and summary["through"] == 76360
    assert summary["final_run"] == str(f.parent) and not summary["phases"]
    assert not summary["full_sequence_complete"] and summary["incomplete_phase"]
    assert seal(f)["status"] == "failed" and f.calls.count("execute") == 1
    assert any(x.endswith("child_execution.json") for x in seal(f)["outputs"])
    assert f.lock.exists()
    with pytest.raises(ValueError, match="new supervisor root"):
        run(f)


@pytest.mark.parametrize("problem", ["corrupt_input", "corrupt_source"])
def test_final_input_and_source_mutation_guard(fixture, problem):
    f = fixture
    f.control.update(stop_at="bootstrap", **{problem: True})
    with pytest.raises(ValueError, match="changed"):
        run(f)
    assert seal(f)["status"] == "failed" and "execute" not in f.calls


def test_keyboard_interrupt_preserves_operation_failure_and_does_not_start_another(fixture):
    f = fixture
    f.control["keyboard_interrupt"] = True
    with pytest.raises(KeyboardInterrupt, match="owned-child cleanup"):
        run(f)
    assert f.calls.count("execute") == 1 and seal(f)["status"] == "failed"
    summary = module.read(f.output / "summary.json")
    assert summary["through"] == 76360 and not summary["phases"]
    assert any(x.endswith("operation.provenance.json") for x in seal(f)["outputs"])
    assert f.lock.exists()


def test_missing_execute_has_no_io_or_handoff(fixture):
    f = fixture
    f.args.execute = False
    with pytest.raises(ValueError, match="execute"):
        run(f)
    assert not f.calls and not f.output.exists() and not f.lock.exists()


def test_failed_bootstrap_does_not_claim_last_independently_audited_run(fixture):
    f = fixture
    f.control["bootstrap_error"] = True
    with pytest.raises(ValueError, match="bootstrap semantic"):
        run(f)
    summary = module.read(f.output / "summary.json")
    assert summary["status"] == "failed" and summary["bootstrap_completed"] is False
    assert summary["final_run"] is None and summary["through"] is None
    assert summary["checkpoint_restore_count"] is None
    assert summary["intended_initial_parent"] == str(f.parent)
    assert "bootstrap/audit.provenance.json" in seal(f)["outputs"]
    assert seal(f)["status"] == "failed" and "register" not in f.calls


def test_wrong_initial_parent_flags_fail_before_lock(fixture):
    f = fixture
    summary = module.read(f.parent / "summary.json")
    summary["full_sequence_complete"] = True
    write(f.parent / "summary.json", summary)
    record = module.read(f.parent / "continuation.provenance.json")
    record["outputs"]["summary"] = entry(f.parent / "summary.json")
    write(f.parent / "continuation.provenance.json", record)
    with pytest.raises(ValueError, match="completion flags"):
        run(f)
    assert not f.lock.exists() and not f.output.exists()

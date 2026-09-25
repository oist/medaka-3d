"""No real child/GPU: failure, ownership, registration and scheduling boundaries."""

import json
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_long_remaining_supervisor as module


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_all_remaining_170_chunks_once_with_final_two():
    intervals = module.phase_ranges()
    assert len(intervals) == 29
    assert intervals[0] == [66400, 76360]
    assert intervals[-1] == [345280, 348600]
    assert sum(b - a for a, b in intervals) == 170 * 1660
    assert all(b - a == 9960 for a, b in intervals[:-1])
    assert all(left[1] == right[0] for left, right in pairwise(intervals))


@pytest.mark.parametrize("stop", [66399, 66401, 348600, 348601, True, 66400.0])
def test_off_grid_or_finished_end_is_not_a_new_phase(stop):
    with pytest.raises(ValueError, match="endpoint"):
        module.next_stop(stop)


def test_existing_lock_is_never_removed_or_taken_over(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "SHARED_LOCK", tmp_path / "global.lock")
    first_output, second_output = tmp_path / "output_a", tmp_path / "output_b"
    lock = module.acquire_lock(first_output, tmp_path / "parent")
    original = lock.read_bytes()
    with pytest.raises(FileExistsError):
        module.acquire_lock(second_output, tmp_path / "parent")
    assert lock.read_bytes() == original
    assert module.read(lock)["output_root"] == str(first_output)


def test_failed_incomplete_and_other_stage_are_not_readiness(tmp_path):
    parent = tmp_path / "chunks/f066400-068060"
    for i, data in enumerate(
        (
            dict(status="failed", stage="prepare_independent_mask_chunk"),
            dict(status="ok", stage="different"),
            None,
        )
    ):
        attempt = parent / f"attempt_{i + 1:04d}"
        attempt.mkdir(parents=True)
        if data:
            write(attempt / "staging.provenance.json", data)
    assert module.successful_chunk_hints(tmp_path, 66400, 68060) == set()
    write(
        parent / "attempt_0004/staging.provenance.json",
        dict(status="ok", stage="prepare_independent_mask_chunk"),
    )
    assert module.successful_chunk_hints(tmp_path, 66400, 68060) == {(66400, 68060)}


def test_poll_announces_new_chunks_only(monkeypatch, tmp_path):
    first, second = (66400, 68060), (68060, 69720)
    snapshots = iter(({first}, {first}, {first, second}))
    monkeypatch.setattr(module, "successful_chunk_hints", lambda *args: next(snapshots))
    events, waits = [], []
    module.wait_for_chunks(
        tmp_path,
        66400,
        69720,
        30,
        lambda kind, **fields: events.append((kind, fields)),
        sleeper=waits.append,
    )
    assert waits == [30, 30]
    assert [row[1]["start"] for row in events] == [66400, 68060]


class Child:
    pid = 123456

    def __init__(self, *, interrupt=False, stubborn=False, exit_code=0):
        self.interrupt, self.stubborn, self.exit_code = interrupt, stubborn, exit_code
        self.returncode = None
        self.calls = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if timeout is None and self.interrupt:
            raise KeyboardInterrupt
        if timeout is not None and self.stubborn:
            self.stubborn = False
            raise subprocess.TimeoutExpired("owned fake child", timeout)
        self.returncode = (
            -9 if "kill" in self.calls else -15 if "terminate" in self.calls else self.exit_code
        )
        return self.returncode

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")


def test_child_uses_argument_vector_and_hidden_window_no_shell(tmp_path):
    child, calls, info = Child(exit_code=7), [], {}

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return child

    command = ["python with space.exe", "runner.py", "--resume", "parent with space"]
    assert module.run_child(command, tmp_path / "log", info, popen=popen) == 7
    assert calls[0][0] is command and calls[0][1]["shell"] is False
    assert calls[0][1]["stderr"] == subprocess.STDOUT
    assert info["returncode"] == 7 and not info["child_alive_after_return"]
    assert "terminate" not in child.calls


@pytest.mark.parametrize("stubborn", [False, True])
def test_interrupt_terminates_owned_child_and_waits(tmp_path, stubborn, monkeypatch):
    monkeypatch.setattr(module, "WINDOWS", False)
    child, info = Child(interrupt=True, stubborn=stubborn), {}
    with pytest.raises(KeyboardInterrupt):
        module.run_child(["unused"], tmp_path / "log", info, popen=lambda *a, **k: child)
    assert "terminate" in child.calls
    assert ("kill" in child.calls) is stubborn
    assert info["termination_requested"] is True
    assert info["child_alive_after_return"] is False
    assert child.returncode in (-15, -9)
    assert (tmp_path / "log").is_file()


def test_spawn_failure_does_not_attempt_to_terminate_other_process(tmp_path):
    info = {}

    def failed(*args, **kwargs):
        raise OSError("spawn failed")

    with pytest.raises(OSError, match="spawn failed"):
        module.run_child(["unused"], tmp_path / "log", info, popen=failed)
    assert info["child_alive_after_return"] is False
    assert "pid" not in info and "termination_requested" not in info


@pytest.mark.parametrize("denied", [False, True])
def test_windows_cleanup_targets_owned_tree_and_never_falls_back(tmp_path, monkeypatch, denied):
    monkeypatch.setattr(module, "WINDOWS", True)
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path / "Windows"))
    child, info, calls = Child(interrupt=True, exit_code=1), {}, []

    def taskkill(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=int(denied), stdout="owned fixture", stderr="denied" if denied else ""
        )

    monkeypatch.setattr(module.subprocess, "run", taskkill)
    expected = ValueError if denied else KeyboardInterrupt
    with pytest.raises(expected):
        module.run_child(["unused"], tmp_path / "log", info, popen=lambda *a, **k: child)
    assert calls[0][0] == [
        str(tmp_path / "Windows/System32/taskkill.exe"),
        "/PID",
        str(child.pid),
        "/T",
        "/F",
    ]
    assert calls[0][1]["capture_output"] is True
    assert info["tree_termination_stderr"] == ("denied" if denied else "")
    assert info["tree_termination_returncode"] == int(denied)
    assert "terminate" not in child.calls and "kill" not in child.calls
    assert info["child_alive_after_return"] is denied
    assert (("wait", 10) in child.calls) is not denied


@pytest.mark.parametrize("callback_fails", [False, True])
def test_owned_pid_callback_precedes_wait_and_failure_stops_child(
    tmp_path, monkeypatch, callback_fails
):
    monkeypatch.setattr(module, "WINDOWS", False)
    child, info, observed = Child(), {}, []

    def started(state):
        assert child.calls == []
        observed.append(state["pid"])
        if callback_fails:
            raise OSError("event flush failed")

    if callback_fails:
        with pytest.raises(OSError, match="event flush"):
            module.run_child(
                ["unused"], tmp_path / "log", info, popen=lambda *a, **k: child, on_started=started
            )
        assert "terminate" in child.calls
    else:
        assert (
            module.run_child(
                ["unused"], tmp_path / "log", info, popen=lambda *a, **k: child, on_started=started
            )
            == 0
        )
    assert observed == [child.pid] and info["child_alive_after_return"] is False


class Digests:
    def __init__(self):
        self.paths = {}

    def add(self, path, expected=None):
        digest = module.sha256(path)
        if expected is not None and expected != digest:
            raise ValueError("evidence SHA changed")
        self.paths[str(path)] = digest
        return digest

    def final_guard(self):
        module.unchanged(self.paths)


@pytest.fixture
def phase(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "REPO", tmp_path)
    root = tmp_path / "phase"
    root.mkdir()
    parent = tmp_path / "parent"
    parent.mkdir()
    source = tmp_path / "runner.py"
    source.write_text("frozen")
    monkeypatch.setattr(module, "RUNNER", source)
    code = {str(source): module.sha256(source)}
    monkeypatch.setattr(module, "sources", lambda: code.copy())
    seed = tmp_path / "seed/registration.json"
    write(seed, {})
    stage, inputs = tmp_path / "stage", tmp_path / "inputs"
    stage.mkdir()
    inputs.mkdir()
    binding = dict(
        code_sha256=code,
        inputs=dict(
            input_stage_root=str(stage),
            seed=dict(registration_path=str(seed), input_root=str(inputs)),
        ),
    )
    parent_audit = tmp_path / "bootstrap"
    write(parent / "continuation.provenance.json", {"outputs": {"candidate": {"sha256": "a" * 64}}})
    audit_path = parent_audit / "summary.json"
    write(audit_path, dict(status="ok", run=str(parent), candidate_sha256="a" * 64))
    write(
        parent_audit / "audit.provenance.json",
        dict(
            status="ok",
            stage="long_supervised_bootstrap_audit",
            params={"all_input_source_snapshot_output_sha_unchanged": True},
            outputs={"summary": {"path": str(audit_path), "sha256": module.sha256(audit_path)}},
        ),
    )
    prepare_calls, verify_calls = [], []

    def prepare(parent, stop, output_run, evidence):
        prepare_calls.append((parent, stop, output_run))
        return dict(
            parent_phase=str(parent),
            expected_output=str(output_run),
            next_native_interval=[66400, stop],
            output_interval=[49800, stop],
            frozen_input_and_code_binding=binding,
        ), {}

    def verify(run, plan, evidence):
        verify_calls.append((run, plan.copy()))
        return dict(status="ok", full_sequence_complete=False)

    backend = SimpleNamespace(Digests=Digests, prepare_phase=prepare, verify_phase=verify)
    return SimpleNamespace(
        root=root,
        parent=parent,
        parent_audit=parent_audit,
        code=code,
        source=source,
        binding=binding,
        backend=backend,
        prepare=prepare_calls,
        verify=verify_calls,
    )


def test_registration_preserves_backend_plan_and_seals_concrete_command(phase):
    plan, child, registration, command = module.register_phase(
        phase.backend,
        phase.parent,
        76360,
        phase.root,
        phase.code,
        phase.binding,
        phase.parent_audit,
    )
    assert "supervisor_command" not in plan
    assert not child.exists()
    assert command[-2:] == ["--resume", str(phase.parent)]
    assert command[command.index("--through-frame") + 1] == "76360"
    seal = module.read(registration / "registration.provenance.json")
    assert seal["status"] == "ok"
    assert module.read(registration / "execution.json")["command"] == command
    assert phase.verify == []
    for path in (phase.parent_audit / "summary.json", phase.parent_audit / "audit.provenance.json"):
        assert seal["inputs"][str(path)]["sha256"] == module.sha256(path)


@pytest.mark.parametrize("corruption", ["run", "candidate", "unbound", "failed"])
def test_wrong_parent_audit_rejected_before_prepare(phase, corruption):
    path = phase.parent_audit / "summary.json"
    audit = module.read(path)
    seal_path = phase.parent_audit / "audit.provenance.json"
    seal = module.read(seal_path)
    if corruption == "run":
        audit["run"] = str(phase.parent.parent / "foreign")
    elif corruption == "candidate":
        audit["candidate_sha256"] = "b" * 64
    elif corruption == "unbound":
        audit["extra"] = "changed bytes after seal"
    else:
        seal["status"] = "failed"
    write(path, audit)
    if corruption != "unbound":
        seal["outputs"]["summary"]["sha256"] = module.sha256(path)
    write(seal_path, seal)
    with pytest.raises(ValueError):
        module.register_phase(
            phase.backend,
            phase.parent,
            76360,
            phase.root,
            phase.code,
            phase.binding,
            phase.parent_audit,
        )
    assert phase.prepare == [] and not (phase.root / "tracking").exists()


def test_subsequent_operation_audit_uses_exact_parent_candidate(phase):
    root = phase.root / "prior_phase"
    path = root / "audit.json"
    write(path, module.read(phase.parent_audit / "summary.json"))
    write(
        root / "operation.provenance.json",
        dict(
            status="ok",
            stage="long_supervised_phase_operation",
            params={"all_input_source_snapshot_output_sha_unchanged": True},
            outputs={"audit": {"path": str(path), "sha256": module.sha256(path)}},
        ),
    )
    evidence = Digests()
    lineage = module.bind_parent_audit(phase.parent, root, evidence)
    assert lineage["parent_candidate_sha256"] == "a" * 64
    assert str(root / "operation.provenance.json") in evidence.paths


def test_successful_child_is_audited_only_after_registered_native_exit_zero(phase, monkeypatch):
    plan, child, registration, command = module.register_phase(
        phase.backend,
        phase.parent,
        76360,
        phase.root,
        phase.code,
        phase.binding,
        phase.parent_audit,
    )

    def fake_child(cmd, log, info, *, on_started):
        assert module.read(registration / "registration.provenance.json")["status"] == "ok"
        assert cmd == command
        log.write_bytes(b"native tracking log")
        info.update(pid=Child.pid, returncode=0, child_alive_after_return=False)
        on_started(info)
        write(
            child / "continuation.provenance.json",
            {"started_at": module.datetime.now(module.UTC).isoformat()},
        )
        return 0

    monkeypatch.setattr(module, "run_child", fake_child)
    events = []
    module.execute_phase(
        phase.backend,
        phase.root,
        plan,
        child,
        registration,
        command,
        phase.code,
        lambda kind, **kwargs: events.append(kind),
    )
    assert len(phase.verify) == 1
    assert events == ["phase_start", "owned_child_started", "phase_complete"]
    assert module.read(phase.root / "operation.provenance.json")["status"] == "ok"


def test_nonzero_child_stops_before_audit_and_preserves_failed_evidence(phase, monkeypatch):
    plan, child, registration, command = module.register_phase(
        phase.backend,
        phase.parent,
        76360,
        phase.root,
        phase.code,
        phase.binding,
        phase.parent_audit,
    )

    def failed(cmd, log, info, **kwargs):
        log.write_bytes(b"failed child log")
        info.update(returncode=3, child_alive_after_return=False)
        return 3

    monkeypatch.setattr(module, "run_child", failed)
    with pytest.raises(ValueError, match="will not be retried"):
        module.execute_phase(
            phase.backend,
            phase.root,
            plan,
            child,
            registration,
            command,
            phase.code,
            lambda *args, **kwargs: None,
        )
    assert phase.verify == []
    assert module.read(phase.root / "operation.provenance.json")["status"] == "failed"
    assert module.read(phase.root / "child_execution.json")["returncode"] == 3
    assert (phase.root / "tracking.log").read_bytes() == b"failed child log"
    assert not (phase.root / "audit.json").exists()


def test_registered_command_cannot_be_changed_before_launch(phase, monkeypatch):
    plan, child, registration, command = module.register_phase(
        phase.backend,
        phase.parent,
        76360,
        phase.root,
        phase.code,
        phase.binding,
        phase.parent_audit,
    )
    monkeypatch.setattr(module, "run_child", lambda *args: pytest.fail("must not launch"))
    altered = [*command, "--foreign"]
    with pytest.raises(ValueError, match="registered command"):
        module.execute_phase(
            phase.backend,
            phase.root,
            plan,
            child,
            registration,
            altered,
            phase.code,
            lambda *args, **kwargs: None,
        )


def test_backend_verification_failure_prevents_successful_operation(phase, monkeypatch):
    plan, child, registration, command = module.register_phase(
        phase.backend,
        phase.parent,
        76360,
        phase.root,
        phase.code,
        phase.binding,
        phase.parent_audit,
    )

    def fake(cmd, log, info, **kwargs):
        log.write_bytes(b"native exit zero but invalid checkpoint")
        info.update(returncode=0, child_alive_after_return=False)
        write(
            child / "continuation.provenance.json",
            {"started_at": module.datetime.now(module.UTC).isoformat()},
        )
        return 0

    def bad(*args):
        raise ValueError("beam/prefix mismatch")

    phase.backend.verify_phase = bad
    monkeypatch.setattr(module, "run_child", fake)
    with pytest.raises(ValueError, match="beam/prefix"):
        module.execute_phase(
            phase.backend,
            phase.root,
            plan,
            child,
            registration,
            command,
            phase.code,
            lambda *args, **kwargs: None,
        )
    assert module.read(phase.root / "operation.provenance.json")["status"] == "failed"


def test_help_and_missing_execute_flag_never_import_backend(monkeypatch):
    monkeypatch.setattr(
        module.importlib, "import_module", lambda *args: pytest.fail("backend/child must not start")
    )
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    with pytest.raises(ValueError, match="--execute"):
        module.main(["--parent", "unused", "--output", "unused"])


@pytest.fixture
def waiting(phase, monkeypatch):
    parent = phase.parent.parent / "expected_parent"
    parent.mkdir()
    root = phase.root / "wait_plan"
    plan_path, seal_path = root / "plan.json", root / "registration.provenance.json"
    write(plan_path, dict(expected_output=str(parent), next_native_interval=[59760, 66400]))
    write(
        seal_path,
        dict(
            status="ok",
            stage="register_long_next_phase_execution",
            outputs={"plan": {"path": str(plan_path), "sha256": module.sha256(plan_path)}},
        ),
    )
    monkeypatch.setattr(module, "WAIT_PARENT", parent)
    monkeypatch.setattr(module, "WAIT_PLAN", root)
    monkeypatch.setattr(module, "WAIT_PLAN_SHA", module.sha256(plan_path))
    monkeypatch.setattr(module, "WAIT_SEAL_SHA", module.sha256(seal_path))
    return SimpleNamespace(
        parent=parent, record=parent / "continuation.provenance.json", phase=phase
    )


def test_wait_parent_retries_missing_and_partial_json_without_payload_read(waiting, capsys):
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        if len(waits) == 1:
            waiting.record.write_text('{"unfinished":')
        else:
            write(waiting.record, dict(status="ok", stage="three_body_long_continuation"))

    result = module.wait_for_parent(waiting.parent, 30, waiting.phase.code, sleeper=sleep)
    assert waits == [30, 30] and len(result) == 2
    assert len(capsys.readouterr().out.splitlines()) == 2
    assert not (waiting.parent / "candidate.npz").exists()


@pytest.mark.parametrize("problem", ["failed", "foreign", "changed_source", "wrong_stage"])
def test_wait_parent_fails_closed(waiting, problem):
    parent = waiting.parent
    record = dict(status="ok", stage="three_body_long_continuation")
    if problem == "failed":
        record["status"] = "failed"
    elif problem == "foreign":
        parent = waiting.phase.parent
    elif problem == "changed_source":
        waiting.phase.source.write_text("modified while waiting")
    else:
        record["stage"] = "unrelated"
    write(waiting.record, record)
    with pytest.raises(ValueError):
        module.wait_for_parent(
            parent, 30, waiting.phase.code, sleeper=lambda _: pytest.fail("must stop")
        )


@pytest.mark.parametrize("problem", [None, "terminal_flag", "initial_input_changed"])
def test_supervisor_bootstrap_then_one_final_phase_and_final_input_guard(
    phase, monkeypatch, problem
):
    # A tiny operational schedule with mocked scientific audit, no real process.
    monkeypatch.setattr(module, "FINAL_STOP", 68060)
    monkeypatch.setattr(module, "SHARED_LOCK", phase.root / "shared.lock")
    output = phase.root / "supervisor"
    initial_summary = phase.parent / "summary.json"
    write(initial_summary, {})
    sequence = []

    def header(parent, stop):
        sequence.append(("header", stop))
        return phase.binding, dict(
            full_sequence_complete=(stop == 68060 and problem != "terminal_flag"),
            prefix_backtrace_provisional=(stop != 68060),
        )

    def bootstrap(parent, evidence):
        sequence.append(("bootstrap", parent))
        return module.read(phase.parent_audit / "summary.json")

    def verify(run, plan, evidence):
        sequence.append(("verify", run))
        return dict(
            status="ok",
            run=str(run),
            candidate_sha256="a" * 64,
            full_sequence_complete=True,
            prefix_backtrace_provisional=False,
        )

    def child(command, log, info, *, on_started):
        sequence.append(("child", command))
        child_root = Path(command[command.index("--output") + 1])
        write(
            child_root / "continuation.provenance.json",
            dict(
                started_at=module.datetime.now(module.UTC).isoformat(),
                outputs={"candidate": {"sha256": "a" * 64}},
            ),
        )
        log.write_bytes(b"no real process")
        info.update(pid=Child.pid, child_alive_after_return=False, returncode=0)
        on_started(info)
        if problem == "initial_input_changed":
            write(initial_summary, {"changed": True})
        return 0

    monkeypatch.setattr(module, "header", header)
    monkeypatch.setattr(
        module, "wait_for_chunks", lambda *a: sequence.append(("chunks_ready", None))
    )
    monkeypatch.setattr(module, "run_child", child)
    phase.backend.verify_bootstrap = bootstrap
    phase.backend.verify_phase = verify
    args = SimpleNamespace(
        parent=phase.parent, output=output, poll_seconds=30, wait_for_parent=False
    )
    if problem:
        with pytest.raises(
            ValueError, match="terminal completion" if problem == "terminal_flag" else "changed"
        ):
            module.supervise(args, phase.backend)
    else:
        module.supervise(args, phase.backend)
    seal = module.read(output / "supervisor.provenance.json")
    assert seal["status"] == ("failed" if problem else "ok")
    assert module.SHARED_LOCK.exists()
    order = [name for name, _ in sequence]
    assert (
        order.index("bootstrap")
        < order.index("chunks_ready")
        < order.index("child")
        < order.index("verify")
    )
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert any(row["event"] == "owned_child_started" and row["pid"] == Child.pid for row in events)

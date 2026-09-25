"""Synthetic operation plus tiny newly owned CPU-process fixtures, never tracking."""

import copy
import json
import os
import sys
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import receipted_runner_phase_operation as o
from test_receipted_runner_phase_audit import plan_fixture


def now():
    return datetime.now(UTC).isoformat()


def fixture(monkeypatch, tmp_path, *, failure=None):
    plan, legacy = plan_fixture(monkeypatch, tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    audit_path = tmp_path / "old_audit.json"
    audit_path.write_text("{}")
    cache = tmp_path / "cache"
    cache.mkdir()
    args = SimpleNamespace(
        parent=parent,
        parent_seal_sha256="a" * 64,
        parent_audit_seal=audit_path,
        parent_audit_sha256="b" * 64,
        through_frame=146080,
        receipt_root=cache,
        output=tmp_path / "phase",
    )
    plan.update(
        parent=str(parent),
        parent_seal_sha256=args.parent_seal_sha256,
        parent_audit=dict(path=str(audit_path), sha256=args.parent_audit_sha256),
        expected_output=str(args.output / "tracking"),
        receipt_cache_root=str(cache),
    )
    parent_record = dict(
        stage=o.m.original.STAGE,
        status="ok",
        params=dict(bindings=legacy),
        finished_at="2020-01-01T00:00:00+00:00",
    )
    fixed = dict(
        parent=parent,
        parent_record=parent_record,
        parent_audit=audit_path,
        legacy=legacy,
        binding=plan["binding"],
        edge=plan["transition"],
        start=136120,
    )
    monkeypatch.setattr(o, "preflight", lambda *_: fixed)
    source = tmp_path / "operation_source.py"
    source.write_text("# fixed synthetic operational source")
    source_sha = o.m.sha(source)
    code = {str(source): source_sha}
    flag, calls = {"late": False}, []

    def sources(evidence):
        evidence.add(source, source_sha)
        result = dict(code)
        if flag["late"]:
            extra = tmp_path / "late.py"
            if not extra.exists():
                extra.write_text("# late import")
            result[str(extra)] = evidence.add(extra)
        return result

    monkeypatch.setattr(o, "operational_sources", sources)
    monkeypatch.setattr(
        o.m.original.torch.cuda, "is_available", lambda: pytest.fail("no GPU query")
    )
    monkeypatch.setattr(o.child, "register", lambda *_: pytest.fail("fresh CLI required"))
    monkeypatch.setattr(o.child, "prepare", lambda *_: pytest.fail("no in-process child prepare"))

    def run(command, log_path, info, *, on_started):
        registration = "--register-only" in command
        label = "registration" if registration else "tracking"
        calls.append(label)
        info.update(pid=101 if registration else 202, started_at=now())
        log_path.write_text("synthetic " + label + "\n")
        try:
            on_started(info)
            root = args.output / label
            root.mkdir()
            stage = o.m.REGISTRATION_STAGE if registration else o.m.STAGE
            name = "registration" if registration else "continuation"
            with o.m.original.provenance(stage, root, name=name) as record:
                record.input(str(source), source, sha256=source_sha)
                if registration:
                    saved = copy.deepcopy(plan)
                    if failure == "wrong_registration":
                        saved["expected_output"] = str(tmp_path / "foreign")
                    o.child.save_json(root / "plan.json", saved, record, "plan")
                    record.param("tracking_GPU_or_GT_performed", False)
                else:
                    path = root / "candidate.npz"
                    path.write_bytes(b"synthetic candidate; fake auditor never opens this")
                    record.output("candidate", path, sha256=o.m.sha(path))
                if failure == label + "_exit":
                    raise ValueError("synthetic producer failure")
                record.param("all_input_source_snapshot_output_sha_unchanged", True)
            if registration and failure == "source_after_registration":
                source.write_text("# modified operational source")
            if registration and failure == "late_import":
                flag["late"] = True
            info["returncode"] = 0
            return 0
        except ValueError:
            info["returncode"] = 1
            return 1
        finally:
            info.update(finished_at=now(), child_alive_after_return=False)

    monkeypatch.setattr(o.owned, "run_child", run)

    def audit(params):
        calls.append("audit")
        assert o.m.read(params.owned_execution)["returncode"] == 0
        root = params.output
        root.mkdir()
        producer = o.m.read(params.run / "continuation.provenance.json")
        result = dict(
            status="ok",
            independent_phase_audit_complete=True,
            run=str(params.run),
            registration_sha256=params.registration_sha256,
            candidate_sha256=producer["outputs"]["candidate"]["sha256"],
        )
        with o.m.original.provenance(o.auditor.STAGE, root, name="audit") as record:
            record.input(str(source), source, sha256=source_sha)
            if failure == "audit_failure":
                raise ValueError("synthetic independent audit rejection")
            o.child.save_json(root / "summary.json", result, record, "summary")
            record.param("all_input_source_snapshot_output_sha_unchanged", True)
        if failure == "audit_return_changed":
            result["candidate_sha256"] = "e" * 64
        return result

    monkeypatch.setattr(o.auditor, "audit", audit)
    return args, plan, fixed, calls, source, flag


def test_registration_cli_child_cli_owned_exit_then_independent_audit(monkeypatch, tmp_path):
    args, plan, _, calls, _, _ = fixture(monkeypatch, tmp_path)
    result = o.execute(args)
    assert calls == ["registration", "tracking", "audit"]
    assert result["independent_phase_audit_complete"] is True
    record = o.m.read(args.output / "operation.provenance.json")
    assert record["status"] == "ok" and record["stage"] == o.STAGE
    assert record["params"]["tracking_child_exit_code"] == 0
    own = o.m.read(args.output / "child_execution.json")
    reg = args.output / "registration"
    assert own["command"] == o.auditor.expected_command(
        args.output / "tracking", reg, o.m.sha(reg / "registration.provenance.json")
    )
    assert own["schema"] == o.auditor.EXECUTION_SCHEMA
    summary = o.m.read(args.output / "summary.json")
    assert summary["status"] == "ok" and summary["parent_promoted"] is False
    assert summary["operation_success_requires_successful_seal"] is True
    commands = o.m.read(args.output / "registration_command.json")["command"]
    assert commands[3] == "--register-only" and "--through-frame" in commands
    assert commands[commands.index("--through-frame") + 1] == str(plan["next_native_interval"][1])
    for item in record["outputs"].values():
        assert o.m.sha(item["path"]) == item["sha256"]
    events = [
        json.loads(line)["event"]
        for line in (args.output / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1] == "phase_checks_complete"
    assert "phase_complete" not in events  # Return and success seal are the completion authority.
    with pytest.raises(ValueError, match="exclusive"):
        o.execute(args)


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("registration_exit", ["registration"]),
        ("wrong_registration", ["registration"]),
        ("source_after_registration", ["registration"]),
        ("late_import", ["registration"]),
        ("tracking_exit", ["registration", "tracking"]),
        ("audit_failure", ["registration", "tracking", "audit"]),
        ("audit_return_changed", ["registration", "tracking", "audit"]),
    ],
)
def test_failure_never_promotes_or_retries(monkeypatch, tmp_path, failure, expected):
    args, _, _, calls, _, _ = fixture(monkeypatch, tmp_path, failure=failure)
    with pytest.raises(ValueError):
        o.execute(args)
    assert calls == expected
    seal = o.m.read(args.output / "operation.provenance.json")
    assert seal["status"] == "failed"
    assert seal["params"].get("all_input_source_snapshot_output_sha_unchanged") is not True
    summary = o.m.read(args.output / "summary.json")
    assert summary["status"] == "failed" and summary["completed_run"] is None
    assert summary["parent_promoted"] is False
    assert "registration_execution" in seal["outputs"]
    if "tracking" in calls:
        assert "child_execution" in seal["outputs"]
    else:
        assert not (args.output / "tracking").exists()


def test_external_callback_failure_is_not_a_phase_success(monkeypatch, tmp_path):
    args, _, _, calls, _, _ = fixture(monkeypatch, tmp_path)

    def callback(event, **_):
        if event == "phase_checks_complete":
            raise RuntimeError("owner event sink unavailable")

    with pytest.raises(RuntimeError):
        o.execute(args, emit=callback)
    assert calls == ["registration", "tracking", "audit"]
    assert o.m.read(args.output / "operation.provenance.json")["status"] == "failed"
    assert o.m.read(args.output / "summary.json")["status"] == "failed"


def test_final_guard_failure_requires_failed_seal_even_after_completed_checks(
    monkeypatch, tmp_path
):
    args, _, _, _, source, _ = fixture(monkeypatch, tmp_path)

    def callback(event, **_):
        if event == "phase_checks_complete":
            source.write_text("# changed after checks")

    with pytest.raises(ValueError):
        o.execute(args, emit=callback)
    assert o.m.read(args.output / "operation.provenance.json")["status"] == "failed"
    summary = o.m.read(args.output / "summary.json")
    assert summary["operation_success_requires_successful_seal"] is True
    assert summary["parent_promoted"] is False


def test_without_execute_cli_cannot_start_a_process(monkeypatch, tmp_path):
    monkeypatch.setattr(o, "execute", lambda *_: pytest.fail("operation launch forbidden"))
    with pytest.raises(ValueError, match="--execute"):
        o.main(
            [
                "--parent",
                str(tmp_path),
                "--parent-seal-sha256",
                "a" * 64,
                "--parent-audit-seal",
                str(tmp_path),
                "--parent-audit-sha256",
                "b" * 64,
                "--through-frame",
                "146080",
                "--receipt-root",
                str(tmp_path),
                "--output",
                str(tmp_path / "not_created"),
            ]
        )
    assert not (tmp_path / "not_created").exists()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_actual_new_owned_cpu_child_exit_and_log(tmp_path, exit_code):
    """Only a new Python -c print/exit fixture; never the tracking runner."""
    info, started = {}, []
    command = [
        sys.executable,
        "-B",
        "-c",
        f"import sys; print('OWNED_CPU_FIXTURE'); sys.exit({exit_code})",
    ]
    result = o.owned.run_child(
        command,
        tmp_path / "owned_cpu.log",
        info,
        on_started=lambda state: started.append(state["pid"]),
    )
    assert result == info["returncode"] == exit_code
    assert started == [info["pid"]] and info["child_alive_after_return"] is False
    assert "OWNED_CPU_FIXTURE" in (tmp_path / "owned_cpu.log").read_text()
    (tmp_path / "owned_cpu.json").write_text(json.dumps(info, indent=2))


@pytest.mark.skipif(
    os.environ.get("MEDAKA_OWNED_CLEANUP_FIXTURE") != "1",
    reason="explicit separate owned-tree fixture execution",
)
def test_actual_only_new_owned_cpu_tree_cleanup(tmp_path):
    """Opt-in to tested utility terminating ONLY this newly created CPU child tree."""
    marker = tmp_path / "owned_identity.json"
    script = (
        "import os,json,time; from pathlib import Path; "
        f"Path({str(marker)!r}).write_text(json.dumps(dict(pid=os.getpid(),ppid=os.getppid()))); "
        "time.sleep(30)"
    )
    info = {}

    def on_started(state):
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        require = marker.exists()
        assert require, "new fixture did not publish identity"
        identity = json.loads(marker.read_text())
        assert state["pid"] in (identity["pid"], identity["ppid"])
        raise RuntimeError("synthetic owned callback interruption")

    try:
        with pytest.raises(RuntimeError, match="synthetic owned"):
            o.owned.run_child(
                [sys.executable, "-B", "-c", script],
                tmp_path / "cleanup.log",
                info,
                on_started=on_started,
            )
    finally:
        (tmp_path / "cleanup.json").write_text(json.dumps(info, indent=2))
    assert info["termination_requested"] is True
    assert info["child_alive_after_return"] is False
    if os.name == "nt":
        assert info["tree_termination_returncode"] == 0
        assert info["tree_termination_command"][-3:] == [str(info["pid"]), "/T", "/F"]

"""Evidence/handle witnesses; fakes and self-created auto-exit CPU processes only."""

import copy
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_owned_long_exit as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def line(value):
    return (json.dumps(value) + "\n").encode()


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    owner, child = 123, 456
    root = tmp_path / "supervisor"
    phase = root / "phases/f066400-076360"
    output, parent = phase / "tracking", tmp_path / "old_parent"
    output.mkdir(parents=True)
    parent.mkdir()
    old_source = tmp_path / "old_source.py"
    old_source.write_bytes(b"# fixture source\n")
    lock_path = tmp_path / "shared.lock"
    monkeypatch.setattr(module, "EXPECTED_SUPERVISOR", root)
    monkeypatch.setattr(module, "SHARED_LOCK", lock_path)
    monkeypatch.setattr(module, "OLD_SOURCE", old_source)
    monkeypatch.setattr(module, "OLD_SOURCE_SHA", module.sha(old_source))
    for key in ("PYTHON", "RUNNER"):
        path = tmp_path / key
        path.write_bytes(key.encode())
        monkeypatch.setattr(module, key, path)
    t = datetime(2026, 1, 1, tzinfo=UTC)

    def stamp(seconds):
        return (t + timedelta(seconds=seconds)).isoformat()

    lock = dict(
        pid=owner,
        created_at=stamp(1),
        output_root=str(root),
        full_target_interval=[49800, 348600],
        source_sha256=module.OLD_SOURCE_SHA,
    )
    write(lock_path, lock)
    (root / "supervisor.lock").write_bytes(lock_path.read_bytes())
    command = [
        str(module.PYTHON),
        "-B",
        str(module.RUNNER),
        "--seed-root",
        "seed",
        "--input-stage-root",
        "stage",
        "--input-root",
        "input",
        "--through-frame",
        "76360",
        "--output",
        str(output),
        "--resume",
        str(parent),
    ]
    execution = dict(
        command=command,
        output=str(output),
        parent=str(parent),
        through_frame=76360,
        source_hashes={str(old_source): module.OLD_SOURCE_SHA},
    )
    plan = dict(
        expected_output=str(output),
        parent_phase=str(parent),
        output_interval=[49800, 76360],
        next_native_interval=[66400, 76360],
    )
    registration = phase / "registration"
    outputs = {}
    for name, document in (("plan", plan), ("execution", execution)):
        path = registration / (name + ".json")
        write(path, document)
        outputs[name] = dict(path=str(path), sha256=module.sha(path))
    seal = dict(
        stage="register_long_supervised_phase",
        status="ok",
        finished_at=stamp(2),
        params={"all_input_source_snapshot_output_sha_unchanged": True},
        outputs=outputs,
    )
    write(registration / "registration.provenance.json", seal)
    start = dict(
        event="phase_start",
        time=stamp(3),
        output=str(output),
        parent=str(parent),
        interval=[66400, 76360],
    )
    event = dict(
        event="owned_child_started",
        time=stamp(5),
        pid=child,
        command=command,
        log=str(phase / "tracking.log"),
        output=str(output),
    )
    (phase / "tracking.log").write_bytes(b"")
    (root / "events.jsonl").write_bytes(line(start) + line(event))
    return dict(
        root=root,
        owner=owner,
        child=child,
        event=event,
        start=start,
        stamp=stamp,
        registration=registration,
        lock=lock_path,
        source=old_source,
    )


def binding_for(case):
    return module.build_binding(case["root"], case["child"])


def test_binding_exact_ownership_and_command(evidence):
    binding, baseline = binding_for(evidence)
    assert binding["owner_pid"] == 123 and binding["child_pid"] == 456
    assert binding["child_event"] == evidence["event"]
    assert module.digest(baseline) == binding["baseline_events_sha256"]
    module.guard(binding["immutable_hashes"])


@pytest.mark.parametrize("damage", ["pid", "command", "duplicate", "next_start", "lock", "seal"])
def test_foreign_ambiguous_or_unregistered_owner_rejected(evidence, damage):
    events = evidence["root"] / "events.jsonl"
    if damage == "pid":
        evidence["child"] = 999
    elif damage in ("command", "duplicate", "next_start"):
        event = copy.deepcopy(evidence["event"])
        if damage == "command":
            event["command"][10] = "86320"
            events.write_bytes(line(evidence["start"]) + line(event))
        else:
            with events.open("ab") as stream:
                stream.write(line(event if damage == "duplicate" else evidence["start"]))
    elif damage == "lock":
        with evidence["lock"].open("ab") as stream:
            stream.write(b" ")
    else:
        path = evidence["registration"] / "registration.provenance.json"
        value = json.loads(path.read_text())
        value["status"] = "failed"
        write(path, value)
    with pytest.raises(ValueError):
        binding_for(evidence)


class Process:
    def __init__(self, pid, created, image, code=0):
        self.identity = dict(
            pid=pid,
            creation_time_utc=created,
            image_path=str(image),
            live_when_handle_acquired=True,
        )
        self.exited, self.closed, self.code = False, False, code

    def signalled(self):
        return self.exited

    def exit_info(self):
        assert self.exited
        return dict(
            **self.identity, exit_code=self.code, wait_signalled=True, exit_observed_at=module.now()
        )

    def close(self):
        self.closed = True


def fake_processes(evidence):
    return {
        123: Process(123, evidence["stamp"](0), module.PYTHON, code=9),
        456: Process(456, evidence["stamp"](4), module.PYTHON, code=0),
    }


def test_child_receipt_usable_while_owner_alive_then_nonzero_owner(evidence, tmp_path):
    binding, baseline = binding_for(evidence)
    sink = module.WitnessOutput(
        tmp_path / "witness",
        binding,
        baseline,
        {str(evidence["source"]): module.sha(evidence["source"])},
    )
    processes = fake_processes(evidence)
    calls = []

    def tick(_):
        if not calls:
            processes[456].exited = True
        else:
            assert (sink.root / "child_exit.receipt.json").is_file()
            assert not (sink.root / "owner_exit.json").exists()
            payload = json.loads((sink.root / "child_exit.json").read_text())
            assert payload["other_role_signalled_at_observation"] is False
            processes[123].exited = True
        calls.append(True)

    result = module.observe(
        binding, sink, opener=processes.__getitem__, sleeper=tick, base_python=module.PYTHON
    )
    assert result["child"]["exit_code"] == 0 and result["owner"]["exit_code"] == 9
    assert all(process.closed for process in processes.values())
    for role in ("child", "owner"):
        receipt = json.loads((sink.root / (role + "_exit.receipt.json")).read_text())
        assert receipt["status"] == "ok" and not receipt["tracking_success_asserted"]
        module.guard({**receipt["inputs"], **receipt["outputs"]})


@pytest.mark.parametrize(
    "damage", ["reused_child", "reused_owner", "foreign_image", "already_dead"]
)
def test_handle_identity_rejects_stale_pid_or_different_image(evidence, tmp_path, damage):
    binding, baseline = binding_for(evidence)
    processes = fake_processes(evidence)
    if damage == "reused_child":
        processes[456].identity["creation_time_utc"] = evidence["stamp"](6)
    elif damage == "reused_owner":
        processes[123].identity["creation_time_utc"] = evidence["stamp"](2)
    elif damage == "foreign_image":
        processes[456].identity["image_path"] = str(tmp_path / "foreign.exe")
    else:
        processes[123].exited = True
    sink = module.WitnessOutput(tmp_path / "witness", binding, baseline, {})
    with pytest.raises(ValueError):
        module.observe(binding, sink, opener=processes.__getitem__, base_python=module.PYTHON)
    assert all(process.closed for process in processes.values())
    assert not (sink.root / "child_exit.receipt.json").exists()


def test_interruption_preserves_child_receipt_and_closes_handles(evidence, tmp_path):
    binding, baseline = binding_for(evidence)
    sink = module.WitnessOutput(tmp_path / "witness", binding, baseline, {})
    processes = fake_processes(evidence)

    def tick(_):
        if processes[456].exited:
            raise KeyboardInterrupt
        processes[456].exited = True

    with pytest.raises(KeyboardInterrupt):
        module.observe(
            binding, sink, opener=processes.__getitem__, sleeper=tick, base_python=module.PYTHON
        )
    assert (sink.root / "child_exit.receipt.json").is_file()
    assert not (sink.root / "owner_exit.receipt.json").exists()
    assert all(process.closed for process in processes.values())


def test_append_only_events_report_new_launch_and_reject_prefix_change(evidence, tmp_path):
    binding, baseline = binding_for(evidence)
    sink = module.WitnessOutput(tmp_path / "witness", binding, baseline, {})
    path = Path(binding["events_path"])
    next_start = dict(event="phase_start", time=evidence["stamp"](99), output="next")
    path.write_bytes(baseline + line(next_start) + b'{"partial":')
    capture = sink.events("later.jsonl")
    assert capture["subsequent_launch_events"] == [next_start]
    assert capture["partial_trailing_line"] is True
    path.write_bytes(baseline.replace(b"phase_start", b"phase_wrong"))
    with pytest.raises(ValueError, match="prefix"):
        sink.events("wrong.jsonl")


def test_immutable_or_output_change_prevents_usable_receipt(evidence, tmp_path):
    binding, baseline = binding_for(evidence)
    sink = module.WitnessOutput(tmp_path / "witness", binding, baseline, {})
    (sink.root / "binding.json").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        sink.exit_receipt("child", {"exit_code": 0}, False)
    assert not (sink.root / "child_exit.receipt.json").exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_main_seals_observations_or_failure_without_changing_old_files(
    evidence, tmp_path, monkeypatch, interrupt
):
    processes = fake_processes(evidence)
    original = module.observe
    output = tmp_path / "main_witness"
    old_lock = evidence["lock"].read_bytes()
    monkeypatch.setattr(sys, "executable", str(module.PYTHON))

    def tick(_):
        if not processes[456].exited:
            processes[456].exited = True
        elif interrupt:
            raise KeyboardInterrupt
        else:
            processes[123].exited = True

    def observed(binding, sink, **kwargs):
        return original(
            binding,
            sink,
            opener=processes.__getitem__,
            sleeper=tick,
            base_python=module.PYTHON,
            **kwargs,
        )

    monkeypatch.setattr(module, "observe", observed)
    args = [
        "--supervisor-root",
        str(evidence["root"]),
        "--child-pid",
        "456",
        "--output",
        str(output),
    ]
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            module.main(args)
    else:
        module.main(args)
    record = json.loads((output / "witness.provenance.json").read_text())
    assert record["status"] == ("failed" if interrupt else "ok")
    assert record["tracking_or_supervisor_success_asserted"] is False
    module.guard({**record["inputs"], **record["outputs"]})
    assert (output / "child_exit.receipt.json").exists()
    assert evidence["lock"].read_bytes() == old_lock
    assert not (evidence["root"] / "supervisor.provenance.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows query/synchronize handle fixture")
def test_real_new_owned_cpu_processes_exit_without_signals(tmp_path):
    """Only Popen children created here; stdin releases normal exit(0) / exit(7)."""
    code = (
        "import os,sys; print(os.getpid(),flush=True); "
        "sys.stdin.readline(); sys.exit(int(sys.argv[1]))"
    )
    processes = []
    base = Path(sys._base_executable)
    try:
        owner = subprocess.Popen(
            [str(base), "-B", "-c", code, "7"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        processes.append(owner)
        assert int(owner.stdout.readline()) == owner.pid
        lock_time = module.now()
        start_time = module.now()
        child = subprocess.Popen(
            [str(module.PYTHON), "-B", "-c", code, "0"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        processes.append(child)
        assert int(child.stdout.readline()) > 0
        event_time = module.now()
        events = tmp_path / "events.jsonl"
        events.write_bytes(b"")
        binding = dict(
            owner_pid=owner.pid,
            child_pid=child.pid,
            owner_lock={"created_at": lock_time},
            phase_start={"time": start_time},
            child_event={"time": event_time, "command": [str(module.PYTHON)]},
            child_event_index=0,
            events_path=str(events),
            immutable_hashes={},
        )

        class Sink(module.WitnessOutput):
            def json(self, name, value):
                result = super().json(name, value)
                if name == "attachment.json":
                    child.stdin.write("exit\n")
                    child.stdin.flush()
                return result

            def exit_receipt(self, role, observation, other_signalled):
                super().exit_receipt(role, observation, other_signalled)
                if role == "child":
                    assert not other_signalled
                    owner.stdin.write("exit\n")
                    owner.stdin.flush()

        sink = Sink(
            tmp_path / "native_witness",
            binding,
            b"",
            {str(Path(module.__file__)): module.sha(module.__file__)},
        )
        result = module.observe(binding, sink, poll_seconds=0.01, base_python=base)
        assert result["child"]["exit_code"] == 0 and result["owner"]["exit_code"] == 7
        assert result["child"]["creation_filetime_100ns"] > 0
        assert result["owner"]["exit_filetime_100ns"] > 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.stdin.close()  # EOF lets this fixture exit naturally; never terminate.
            process.wait(timeout=15)
            process.stdout.close()
            if not process.stdin.closed:
                process.stdin.close()

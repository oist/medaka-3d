"""Synthetic metadata and fake retained handles only; no real PID query or signal."""

import copy
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_receipted_owner_exit as module

frozen = module.frozen


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def line(value):
    return (json.dumps(value) + "\n").encode()


@pytest.fixture
def case(tmp_path, monkeypatch):
    # Patch only the NEW builder constants; frozen module globals remain intact.
    root = tmp_path / "supervisor"
    phase = root / "phases/f136120-146080"
    output = phase / "tracking"
    parent = tmp_path / "parent"
    output.mkdir(parents=True)
    parent.mkdir()
    source = tmp_path / "receipted_supervisor.py"
    source.write_bytes(b"# synthetic current supervisor\n")
    lock_path = tmp_path / "shared.lock"
    monkeypatch.setattr(module, "EXPECTED_SUPERVISOR", root)
    monkeypatch.setattr(module, "SHARED_LOCK", lock_path)
    monkeypatch.setattr(module, "SUPERVISOR_SOURCE", source)
    monkeypatch.setattr(module, "SUPERVISOR_SHA", frozen.sha(source))
    origin = datetime(2026, 1, 1, tzinfo=UTC)

    def stamp(seconds):
        return (origin + timedelta(seconds=seconds)).isoformat()

    lock = dict(
        schema=module.SUPERVISOR_STAGE,
        pid=123,
        created_at=stamp(1),
        output_root=str(root),
        full_target_interval=[49800, 348600],
        source_path=str(source),
        source_sha256=module.SUPERVISOR_SHA,
    )
    write(lock_path, lock)
    (root / "supervisor.lock").write_bytes(lock_path.read_bytes())
    command = [
        str(module.PYTHON),
        "-B",
        str(module.RUNNER),
        "--seed-root",
        "synthetic_seed",
        "--input-stage-root",
        "synthetic_stage",
        "--input-root",
        "synthetic_input",
        "--through-frame",
        "146080",
        "--output",
        str(output),
        "--resume",
        str(parent),
    ]
    plan = dict(
        schema="long_continuation_registered_phase_v1",
        full_target_interval=[49800, 348600],
        expected_output=str(output),
        parent_phase=str(parent),
        parent_processed_interval=[49800, 136120],
        next_native_interval=[136120, 146080],
        output_interval=[49800, 146080],
    )
    execution = dict(
        command=command,
        output=str(output),
        parent=str(parent),
        through_frame=146080,
        source_hashes={str(source): module.SUPERVISOR_SHA},
    )
    registration = phase / "registration"
    seal = dict(
        stage=module.REGISTRATION_STAGE,
        status="ok",
        finished_at=stamp(2),
        params={"all_input_source_snapshot_output_sha_unchanged": True},
        inputs={str(source): dict(path=str(source), sha256=module.SUPERVISOR_SHA)},
        outputs={},
    )
    for role, document in (("plan", plan), ("execution", execution)):
        path = registration / (role + ".json")
        write(path, document)
        seal["outputs"][role] = dict(
            path=str(path), sha256=frozen.sha(path), size=path.stat().st_size
        )
    write(registration / "registration.provenance.json", seal)
    start = dict(
        event="phase_start",
        time=stamp(3),
        output=str(output),
        parent=str(parent),
        interval=[136120, 146080],
    )
    event = dict(
        event="owned_child_started",
        time=stamp(5),
        pid=456,
        command=command,
        log=str(phase / "tracking.log"),
        output=str(output),
    )
    (phase / "tracking.log").write_bytes(b"")
    (root / "events.jsonl").write_bytes(line(start) + line(event))
    return dict(
        root=root,
        registration=registration,
        start=start,
        event=event,
        lock=lock_path,
        source=source,
        stamp=stamp,
        plan=plan,
        execution=execution,
    )


def bind(case):
    return module.build_binding(case["root"], 456)


def update_document(case, role, value):
    path = case["registration"] / (role + ".json")
    write(path, value)
    seal_path = case["registration"] / "registration.provenance.json"
    seal = frozen.decode(seal_path.read_bytes())
    seal["outputs"][role].update(sha256=frozen.sha(path), size=path.stat().st_size)
    write(seal_path, seal)


def test_binding_uses_current_metadata_and_original_engine(case):
    old_constants = (frozen.EXPECTED_SUPERVISOR, frozen.SHARED_LOCK, frozen.OLD_SOURCE)
    binding, baseline = bind(case)
    assert binding["schema"] == "owned_long_process_exit_witness_v1"
    assert binding["owner_pid"] == 123 and binding["child_pid"] == 456
    assert binding["child_event"] == case["event"]
    assert binding["baseline_events_sha256"] == frozen.digest(baseline)
    assert binding["immutable_hashes"][str(module.FROZEN_SOURCE)] == module.FROZEN_SHA
    for name in ("WindowsProcess", "validate_identities", "WitnessOutput", "observe"):
        assert getattr(module, name) is getattr(frozen, name)
    assert old_constants == (frozen.EXPECTED_SUPERVISOR, frozen.SHARED_LOCK, frozen.OLD_SOURCE)
    frozen.guard(binding["immutable_hashes"])


@pytest.mark.parametrize(
    "damage",
    [
        "lock_schema",
        "lock_source_path",
        "lock_copy",
        "old_registration_stage",
        "failed_registration",
        "registration_source",
        "output_size",
        "duplicate_child",
        "later_launch",
        "no_phase_start",
        "command_difference",
        "reordered_flags",
        "short_command",
        "range",
        "bool_through",
        "plan_schema",
        "plan_sha",
        "chronology",
    ],
)
def test_metadata_failure_refuses_binding_without_handles(case, damage):
    seal_path = case["registration"] / "registration.provenance.json"
    seal = frozen.decode(seal_path.read_bytes())
    event, start = copy.deepcopy(case["event"]), copy.deepcopy(case["start"])
    plan, execution = copy.deepcopy(case["plan"]), copy.deepcopy(case["execution"])
    events = case["root"] / "events.jsonl"
    if damage.startswith("lock_"):
        lock = frozen.decode(case["lock"].read_bytes())
        if damage == "lock_schema":
            lock["schema"] = "old_supervisor"
        elif damage == "lock_source_path":
            lock["source_path"] = str(module.FROZEN_SOURCE)
        write(case["lock"], lock)
        if damage != "lock_copy":
            (case["root"] / "supervisor.lock").write_bytes(case["lock"].read_bytes())
        else:
            case["lock"].write_bytes(case["lock"].read_bytes() + b" ")
    elif damage in (
        "old_registration_stage",
        "failed_registration",
        "registration_source",
        "output_size",
        "chronology",
    ):
        if damage == "old_registration_stage":
            seal["stage"] = "register_long_supervised_phase"
        elif damage == "failed_registration":
            seal["status"] = "failed"
        elif damage == "registration_source":
            seal["inputs"] = {}
        elif damage == "output_size":
            seal["outputs"]["execution"]["size"] += 1
        else:
            seal["finished_at"] = case["stamp"](6)
        write(seal_path, seal)
    elif damage in ("duplicate_child", "later_launch", "no_phase_start"):
        if damage == "no_phase_start":
            events.write_bytes(line(event))
        else:
            events.write_bytes(
                events.read_bytes() + line(event if damage == "duplicate_child" else start)
            )
    elif damage in ("command_difference", "reordered_flags", "short_command"):
        if damage == "command_difference":
            event["command"][10] = "156040"
        elif damage == "reordered_flags":
            event["command"][3], event["command"][5] = event["command"][5], event["command"][3]
        else:
            event["command"] = event["command"][:-2]
        if damage != "command_difference":
            execution["command"] = event["command"]
            update_document(case, "execution", execution)
        events.write_bytes(line(start) + line(event))
    elif damage == "bool_through":
        execution["through_frame"] = True
        update_document(case, "execution", execution)
    elif damage == "plan_sha":
        path = case["registration"] / "plan.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        if damage == "range":
            plan["parent_processed_interval"] = [49800, 126160]
        else:
            plan["schema"] = "other_plan"
        update_document(case, "plan", plan)
    with pytest.raises(ValueError):
        bind(case)


class Process:
    def __init__(self, pid, created, code):
        self.identity = dict(
            pid=pid,
            creation_time_utc=created,
            image_path=str(module.PYTHON),
            live_when_handle_acquired=True,
        )
        self.exited, self.closed, self.code = False, False, code

    def signalled(self):
        return self.exited

    def exit_info(self):
        assert self.exited
        return dict(
            **self.identity, exit_code=self.code, wait_signalled=True, exit_observed_at=frozen.now()
        )

    def close(self):
        self.closed = True


@pytest.mark.parametrize("interrupt", [False, True])
def test_old_observe_child_receipt_then_owner_or_interruption(case, tmp_path, interrupt):
    binding, baseline = bind(case)
    processes = {
        123: Process(123, case["stamp"](0), 9),
        456: Process(456, case["stamp"](4), 0),
    }
    output = tmp_path / "witness"
    old_lock = case["lock"].read_bytes()
    source_hashes = {
        str(Path(module.__file__).resolve()): frozen.sha(module.__file__),
        str(module.FROZEN_SOURCE): module.FROZEN_SHA,
        str(case["source"]): module.SUPERVISOR_SHA,
    }

    def tick(_):
        if not processes[456].exited:
            processes[456].exited = True
        else:
            receipt = frozen.decode((output / "child_exit.receipt.json").read_bytes())
            assert receipt["status"] == "ok"
            frozen.guard({**receipt["inputs"], **receipt["outputs"]})
            assert not (output / "owner_exit.json").exists()
            payload = frozen.decode((output / "child_exit.json").read_bytes())
            assert payload["other_role_signalled_at_observation"] is False
            if interrupt:
                raise KeyboardInterrupt
            processes[123].exited = True

    def observed(binding, sink, **kwargs):
        return frozen.observe(
            binding,
            sink,
            opener=processes.__getitem__,
            sleeper=tick,
            base_python=module.PYTHON,
            **kwargs,
        )

    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            module.capture(binding, baseline, output, source_hashes, observer=observed)
    else:
        result = module.capture(binding, baseline, output, source_hashes, observer=observed)
        assert result["child"]["exit_code"] == 0 and result["owner"]["exit_code"] == 9
    record = frozen.decode((output / "witness.provenance.json").read_bytes())
    assert record["schema"] == frozen.SCHEMA and record["stage"] == module.STAGE
    assert record["status"] == ("failed" if interrupt else "ok")
    assert record["tracking_or_supervisor_success_asserted"] is False
    frozen.guard({**record["inputs"], **record["outputs"]})
    assert all(process.closed for process in processes.values())
    assert case["lock"].read_bytes() == old_lock
    assert not (case["root"] / "supervisor.provenance.json").exists()


def test_partial_events_then_later_launch_are_reported_not_hidden(case, tmp_path):
    events = case["root"] / "events.jsonl"
    events.write_bytes(events.read_bytes() + b'{"event":')
    binding, baseline = bind(case)
    sink = module.WitnessOutput(tmp_path / "witness", binding, baseline, {})
    events.write_bytes(baseline + b'"phase_start","time":"later"}\n')
    capture = sink.events("later.jsonl")
    assert capture["subsequent_launch_events"] == [dict(event="phase_start", time="later")]
    assert capture["partial_trailing_line"] is False
    events.write_bytes(events.read_bytes().replace(b"phase_start", b"phase_wrong", 1))
    with pytest.raises(ValueError, match="prefix"):
        sink.events("changed.jsonl")


def test_source_conflict_refused_before_output_and_guard_failure_sealed(case, tmp_path):
    binding, baseline = bind(case)
    output = tmp_path / "conflict"
    with pytest.raises(ValueError, match="conflicts"):
        module.capture(binding, baseline, output, {str(case["source"]): "0" * 64})
    assert not output.exists()

    def changed(binding, sink, **kwargs):
        case["source"].write_bytes(b"changed")
        return {"owner": {}, "child": {}}

    output = tmp_path / "guard_failure"
    with pytest.raises(ValueError, match="changed"):
        module.capture(binding, baseline, output, {}, observer=changed)
    record = frozen.decode((output / "witness.provenance.json").read_bytes())
    assert record["status"] == "failed" and "changed" in record["error"]


def test_wrong_pid_or_root_rejected(case, tmp_path):
    for pid in (True, 0, -1, 2**32, 999):
        with pytest.raises(ValueError):
            module.build_binding(case["root"], pid)
    with pytest.raises(ValueError, match="different supervisor root"):
        module.build_binding(tmp_path, 456)


def test_main_preflight_failure_does_not_open_handles(case, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("failed metadata must not reach observation")

    monkeypatch.setattr(module, "capture", forbidden)
    (case["root"] / "supervisor.lock").write_bytes(b"changed")
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="copy"):
        module.main(
            ["--supervisor-root", str(case["root"]), "--child-pid", "456", "--output", str(output)]
        )
    assert not output.exists()

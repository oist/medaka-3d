"""Read-only held exits for the existing receipted supervisor and recorded child.

Only the ownership/registration adapter is new. The frozen query/synchronize
handles, identity checks, append-only event capture and per-role receipts are
reused unchanged. A successful witness proves two observed process exits, not
tracking success, descendant exit, a safe handoff, or absence of future launches.
No process enumeration, signals, lock changes, tracking, numerical imports or GPU.
Metadata preflight must succeed before an output or any process handle is opened.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import capture_owned_long_exit as frozen

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work/three_body"
EXPECTED_SUPERVISOR = WORK / "long_remaining_receipted_supervisor_v1"
SHARED_LOCK = WORK / "long_remaining_receipted_supervisor.lock"
SUPERVISOR_SOURCE = WORK / "run_long_receipted_supervisor.py"
SUPERVISOR_SHA = "ddf95225f99e908e1b065915111766d069badf31f8b048414073c7d5855a9ee6"
FROZEN_SOURCE = WORK / "capture_owned_long_exit.py"
FROZEN_SHA = "6a5fbed6209a5baeb56166af3c66e1eeb988027b0128df46422e0a7a97278a3e"
SUPERVISOR_STAGE = "long_remaining_receipted_supervisor_v1"
REGISTRATION_STAGE = "register_long_receipted_phase_v1"
STAGE = "owned_receipted_supervisor_exit_witness_v1"
SCHEMA = frozen.SCHEMA
PYTHON, RUNNER = frozen.PYTHON, frozen.RUNNER

# These are the original objects, not copied implementations or patched globals.
WindowsProcess = frozen.WindowsProcess
validate_identities = frozen.validate_identities
WitnessOutput = frozen.WitnessOutput
observe = frozen.observe
require = frozen.require


def build_binding(supervisor_root, child_pid):
    """Read only immutable small metadata and the append-only event baseline.

    The registration's plan/execution and source binding identify the recorded
    launcher. This does not repeat the registration's heavy scientific audit.
    PID values are supplied by the caller and checked against lock/events.
    """
    require(type(child_pid) is int and 0 < child_pid <= 0xFFFFFFFF, "invalid child PID")
    root = frozen.within(supervisor_root)
    require(root == EXPECTED_SUPERVISOR.resolve(), "different supervisor root")
    hashes = {}

    def read_fixed(path, expected=None):
        path = frozen.within(path)
        data = path.read_bytes()
        actual = frozen.digest(data)
        require(expected is None or actual == expected, "fixed evidence SHA differs")
        require(str(path) not in hashes or hashes[str(path)] == actual, "changed fixed evidence")
        hashes[str(path)] = actual
        return data

    read_fixed(FROZEN_SOURCE, FROZEN_SHA)
    read_fixed(SUPERVISOR_SOURCE, SUPERVISOR_SHA)
    lock_bytes = read_fixed(SHARED_LOCK)
    require(read_fixed(root / "supervisor.lock") == lock_bytes, "owner lock copy differs")
    lock = frozen.decode(lock_bytes)
    owner_pid = lock["pid"]
    require(type(owner_pid) is int and 0 < owner_pid <= 0xFFFFFFFF, "invalid owner PID")
    require(owner_pid != child_pid, "owner and child PID must differ")
    require(
        lock["schema"] == SUPERVISOR_STAGE
        and frozen.within(lock["source_path"]) == SUPERVISOR_SOURCE.resolve()
        and lock["source_sha256"] == SUPERVISOR_SHA
        and frozen.within(lock["output_root"]) == root,
        "owner lock source/stage/root differs",
    )
    require(lock["full_target_interval"] == [49800, 348600], "different registered interval")
    events_path = frozen.within(root / "events.jsonl")
    baseline = events_path.read_bytes()
    rows = frozen.event_rows(baseline)
    matches = [
        (i, row)
        for i, row in enumerate(rows)
        if row.get("event") == "owned_child_started" and row.get("pid") == child_pid
    ]
    require(len(matches) == 1, "child PID is not uniquely recorded by this supervisor")
    index, event = matches[0]
    require(
        index > 0 and rows[index - 1].get("event") == "phase_start",
        "owned child has no immediately preceding phase_start",
    )
    require(
        not any(
            row.get("event") in ("phase_start", "owned_child_started") for row in rows[index + 1 :]
        ),
        "a later tracking launch is already recorded",
    )
    start = rows[index - 1]
    child_root = frozen.within(event["output"])
    require(
        child_root.parent.parent == root / "phases" and child_root.name == "tracking",
        "child is outside the owned phase tree",
    )
    require(
        frozen.within(start["output"]) == child_root
        and frozen.within(event["log"]) == child_root.parent / "tracking.log",
        "event output/log disagree",
    )
    registration = child_root.parent / "registration"
    seal = frozen.decode(read_fixed(registration / "registration.provenance.json"))
    require(
        seal["status"] == "ok"
        and seal["stage"] == REGISTRATION_STAGE
        and seal["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
        "child registration was not successfully guarded",
    )
    documents = {}
    for role, filename in (("plan", "plan.json"), ("execution", "execution.json")):
        item = seal["outputs"][role]
        path = frozen.within(registration / filename)
        require(frozen.within(item["path"]) == path, "registration output role/path differs")
        data = read_fixed(path, item["sha256"])
        require(len(data) == item["size"], "registration output size differs")
        documents[role] = frozen.decode(data)
    plan, execution = documents["plan"], documents["execution"]
    require(
        plan["schema"] == "long_continuation_registered_phase_v1"
        and plan["full_target_interval"] == [49800, 348600],
        "registered plan schema/scope differs",
    )
    command = event["command"]
    require(
        isinstance(command, list) and all(isinstance(value, str) for value in command),
        "invalid registered command",
    )
    require(
        command == execution["command"]
        and len(command) == 15
        and frozen.within(command[0]) == PYTHON.resolve()
        and command[1] == "-B"
        and frozen.within(command[2]) == RUNNER.resolve(),
        "event command differs from frozen registered child",
    )
    require(
        command[3::2]
        == [
            "--seed-root",
            "--input-stage-root",
            "--input-root",
            "--through-frame",
            "--output",
            "--resume",
        ],
        "unexpected child command flags",
    )
    interval = plan["next_native_interval"]
    require(
        isinstance(interval, list)
        and len(interval) == 2
        and all(type(value) is int for value in interval)
        and 49800 <= interval[0] < interval[1] <= 348600
        and type(execution["through_frame"]) is int
        and command[10] == str(interval[1])
        and execution["through_frame"] == interval[1]
        and plan["output_interval"] == [49800, interval[1]]
        and plan["parent_processed_interval"] == [49800, interval[0]],
        "registered native range differs",
    )
    require(
        frozen.within(command[12]) == child_root
        and frozen.within(execution["output"]) == child_root
        and frozen.within(plan["expected_output"]) == child_root
        and start["interval"] == interval
        and frozen.within(command[14])
        == frozen.within(execution["parent"])
        == frozen.within(plan["parent_phase"])
        == frozen.within(start["parent"]),
        "registered native range/parent/output differs",
    )
    source_key = str(SUPERVISOR_SOURCE.resolve())
    source_input = seal["inputs"].get(source_key)
    require(
        execution["source_hashes"].get(source_key) == SUPERVISOR_SHA
        and isinstance(source_input, dict)
        and frozen.within(source_input["path"]) == SUPERVISOR_SOURCE.resolve()
        and source_input["sha256"] == SUPERVISOR_SHA,
        "registration does not bind the pinned supervisor source",
    )
    require(
        frozen.utc(seal["finished_at"]) <= frozen.utc(start["time"]) <= frozen.utc(event["time"]),
        "registration/event chronology differs",
    )
    frozen.guard(hashes)
    return dict(
        schema=SCHEMA,
        supervisor_root=str(root),
        owner_pid=owner_pid,
        child_pid=child_pid,
        owner_lock=lock,
        child_event=event,
        child_event_index=index,
        phase_start=start,
        immutable_hashes=hashes,
        events_path=str(events_path),
        baseline_events_sha256=frozen.digest(baseline),
        baseline_events_bytes=len(baseline),
    ), baseline


def capture(binding, baseline, output, source_hashes, *, poll_seconds=1.0, observer=observe):
    """Seal the unchanged observation routine under the new overall stage.

    Observer injection is only for synthetic tests; the CLI always uses the
    original routine. Partial role receipts remain usable after interruption.
    """
    for path, value in source_hashes.items():
        require(
            binding["immutable_hashes"].get(path, value) == value,
            "source snapshot conflicts with binding",
        )
    sink = WitnessOutput(output, binding, baseline, source_hashes)
    started, status, failure, result = frozen.now(), "failed", None, None
    try:
        result = observer(binding, sink, poll_seconds=poll_seconds)
        require(set(result) == {"child", "owner"}, "both process exit observations required")
        status = "ok"
        return result
    except BaseException:
        failure = traceback.format_exc()
        raise
    finally:
        try:
            sink.final_guard()
        except BaseException:
            status = "failed"
            failure = (failure or "") + traceback.format_exc()
            raise
        finally:
            sink.json(
                "witness.provenance.json",
                dict(
                    schema=SCHEMA,
                    stage=STAGE,
                    status=status,
                    started_at=started,
                    finished_at=frozen.now(),
                    error=failure,
                    observations=result,
                    inputs=sink.inputs,
                    outputs=sink.outputs.copy(),
                    tracking_or_supervisor_success_asserted=False,
                    signals_enumeration_GPU_or_tracking_performed=False,
                ),
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervisor-root", type=Path, required=True)
    parser.add_argument("--child-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    require(
        os.name == "nt" and Path(sys.executable).resolve() == PYTHON.resolve(),
        "use the registered Windows .venv interpreter",
    )
    sources = {
        str(Path(__file__).resolve()): frozen.sha(__file__),
        str(FROZEN_SOURCE.resolve()): FROZEN_SHA,
        str(SUPERVISOR_SOURCE.resolve()): SUPERVISOR_SHA,
        str(REPO / ".venv/pyvenv.cfg"): frozen.sha(REPO / ".venv/pyvenv.cfg"),
    }
    binding, baseline = build_binding(args.supervisor_root, args.child_pid)
    return capture(binding, baseline, args.output, sources, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()

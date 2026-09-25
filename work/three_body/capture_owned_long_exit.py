"""Read-only Windows exit observations for a registered supervisor and its child.

No signals, process enumeration, tracking, detector or GPU calls. Successful
witness receipts mean that held process handles signalled, not that tracking or
the supervisor succeeded. The child receipt is usable before the owner exits.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from ctypes import wintypes
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work/three_body"
EXPECTED_SUPERVISOR = WORK / "long_remaining_supervisor_v1"
SHARED_LOCK = WORK / "long_remaining_supervisor.lock"
OLD_SOURCE = WORK / "run_long_remaining_supervisor.py"
OLD_SOURCE_SHA = "4e4e19a35bb7b11a7c38f87f13c03e66cc987252e055ebfa3d2b163c84f214db"
PYTHON = REPO / ".venv/Scripts/python.exe"
RUNNER = REPO / "labs/20260909_three_body/run_long_continuation.py"
SCHEMA = "owned_long_process_exit_witness_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(UTC).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=unique)


def within(path, *, exists=True):
    path = Path(path)
    require(not path.is_symlink(), "symbolic link is not an owned evidence path")
    result = path.resolve(strict=exists)
    require(result.is_relative_to(REPO), "evidence/output must remain in workspace")
    return result


def utc(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "timestamp requires timezone")
    return result.astimezone(UTC)


def event_rows(data):
    """Only completed lines; retain partial trailing bytes in the actual snapshot."""
    return [decode(line) for line in data.splitlines(keepends=True) if line.endswith(b"\n")]


def guard(hashes):
    for path, expected in hashes.items():
        require(sha(path) == expected, "immutable witness evidence changed: " + str(path))


def build_binding(supervisor_root, child_pid):
    require(type(child_pid) is int and 0 < child_pid <= 0xFFFFFFFF, "invalid child PID")
    root = within(supervisor_root)
    require(root == EXPECTED_SUPERVISOR.resolve(), "different supervisor root")
    hashes = {}

    def read_fixed(path, expected=None):
        path = within(path)
        data = path.read_bytes()
        actual = digest(data)
        require(expected is None or actual == expected, "fixed evidence SHA differs")
        require(str(path) not in hashes or hashes[str(path)] == actual, "changed fixed evidence")
        hashes[str(path)] = actual
        return data

    read_fixed(OLD_SOURCE, OLD_SOURCE_SHA)
    lock_bytes = read_fixed(SHARED_LOCK)
    require(read_fixed(root / "supervisor.lock") == lock_bytes, "owner lock copy differs")
    lock = decode(lock_bytes)
    owner_pid = lock["pid"]
    require(type(owner_pid) is int and 0 < owner_pid <= 0xFFFFFFFF, "invalid owner PID")
    require(owner_pid != child_pid, "owner and child PID must differ")
    require(
        within(lock["output_root"]) == root and lock["source_sha256"] == OLD_SOURCE_SHA,
        "owner lock source/root differs",
    )
    require(lock["full_target_interval"] == [49800, 348600], "different registered interval")
    events_path = within(root / "events.jsonl")
    baseline = events_path.read_bytes()
    rows = event_rows(baseline)
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
            r.get("event") in ("phase_start", "owned_child_started") for r in rows[index + 1 :]
        ),
        "a later tracking launch is already recorded",
    )
    start = rows[index - 1]
    child_root = within(event["output"])
    require(
        child_root.parent.parent == root / "phases" and child_root.name == "tracking",
        "child is outside the owned phase tree",
    )
    require(
        within(start["output"]) == child_root
        and within(event["log"]) == child_root.parent / "tracking.log",
        "event output/log disagree",
    )
    registration = child_root.parent / "registration"
    seal = decode(read_fixed(registration / "registration.provenance.json"))
    require(
        seal["status"] == "ok"
        and seal["stage"] == "register_long_supervised_phase"
        and seal["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
        "child registration was not successfully guarded",
    )
    documents = {}
    for role, filename in (("plan", "plan.json"), ("execution", "execution.json")):
        item = seal["outputs"][role]
        path = within(registration / filename)
        require(within(item["path"]) == path, "registration output role/path differs")
        documents[role] = decode(read_fixed(path, item["sha256"]))
    plan, execution = documents["plan"], documents["execution"]
    command = event["command"]
    require(
        isinstance(command, list) and all(isinstance(x, str) for x in command),
        "invalid registered command",
    )
    require(
        command == execution["command"]
        and len(command) == 15
        and within(command[0]) == PYTHON.resolve()
        and command[1] == "-B"
        and within(command[2]) == RUNNER.resolve(),
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
    require(
        within(command[12]) == child_root
        and within(execution["output"]) == child_root
        and within(plan["expected_output"]) == child_root
        and int(command[10]) == execution["through_frame"] == plan["output_interval"][1]
        and start["interval"] == plan["next_native_interval"]
        and within(command[14]) == within(execution["parent"]) == within(plan["parent_phase"])
        and within(start["parent"]) == within(plan["parent_phase"]),
        "registered native range/parent/output differs",
    )
    require(
        execution["source_hashes"].get(str(OLD_SOURCE.resolve())) == OLD_SOURCE_SHA,
        "registration does not bind the pinned supervisor source",
    )
    require(
        utc(seal["finished_at"]) <= utc(start["time"]) <= utc(event["time"]),
        "registration/event chronology differs",
    )
    guard(hashes)
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
        baseline_events_sha256=digest(baseline),
        baseline_events_bytes=len(baseline),
    ), baseline


def filetime_utc(ticks):
    return (datetime(1601, 1, 1, tzinfo=UTC) + timedelta(microseconds=ticks // 10)).isoformat()


class WindowsProcess:
    """A retained query/synchronize handle. No terminate rights are requested."""

    def __init__(self, pid):
        require(os.name == "nt", "Windows process handles are required")
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            "GetProcessId": ([wintypes.HANDLE], wintypes.DWORD),
            "WaitForSingleObject": ([wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
            "GetExitCodeProcess": (
                [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)],
                wintypes.BOOL,
            ),
            "GetProcessTimes": (
                [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)],
                wintypes.BOOL,
            ),
            "QueryFullProcessImageNameW": (
                [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)],
                wintypes.BOOL,
            ),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = args, result
        self.handle = self.api.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.pid = pid
        try:
            require(self.api.GetProcessId(self.handle) == pid, "opened process PID differs")
            require(not self.signalled(), "process already exited before attachment")
            buffer, size = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
            self.check(
                self.api.QueryFullProcessImageNameW(self.handle, 0, buffer, ctypes.byref(size))
            )
            created, _ = self.times()
            self.identity = dict(
                pid=pid,
                creation_filetime_100ns=created,
                creation_time_utc=filetime_utc(created),
                image_path=buffer.value,
                attached_at=now(),
                access_mask=0x00101000,
                live_when_handle_acquired=True,
            )
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())

    def signalled(self):
        result = self.api.WaitForSingleObject(self.handle, 0)
        if result not in (0, 258):
            raise ctypes.WinError(ctypes.get_last_error())
        return result == 0

    def times(self):
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        self.check(
            self.api.GetProcessTimes(
                self.handle,
                ctypes.byref(created),
                ctypes.byref(exited),
                ctypes.byref(kernel),
                ctypes.byref(user),
            )
        )
        return tuple((x.dwHighDateTime << 32) | x.dwLowDateTime for x in (created, exited))

    def exit_info(self):
        require(self.signalled(), "process handle has not signalled")
        code = wintypes.DWORD()
        self.check(self.api.GetExitCodeProcess(self.handle, ctypes.byref(code)))
        created, exited = self.times()
        require(
            created == self.identity["creation_filetime_100ns"] and exited > 0,
            "held process identity/exit time differs",
        )
        return dict(
            **self.identity,
            exit_code=int(code.value),
            exit_filetime_100ns=exited,
            exit_time_utc=filetime_utc(exited),
            exit_observed_at=now(),
            wait_signalled=True,
            exit_code_success=code.value == 0,
        )

    def close(self):
        if self.handle:
            self.check(self.api.CloseHandle(self.handle))
            self.handle = None


def validate_identities(binding, processes, base_python):
    owner, child = (processes[role].identity for role in ("owner", "child"))
    require(
        owner["pid"] == binding["owner_pid"] and child["pid"] == binding["child_pid"],
        "held PID differs from ownership evidence",
    )
    require(
        Path(owner["image_path"]).resolve() in {PYTHON.resolve(), Path(base_python).resolve()},
        "owner executable differs from this registered Python environment",
    )
    require(
        Path(child["image_path"]).resolve() == Path(binding["child_event"]["command"][0]).resolve(),
        "child launcher image differs from registered command",
    )
    require(
        utc(owner["creation_time_utc"])
        <= utc(binding["owner_lock"]["created_at"])
        <= utc(child["creation_time_utc"]),
        "owner creation time suggests PID reuse",
    )
    require(
        utc(binding["phase_start"]["time"])
        <= utc(child["creation_time_utc"])
        <= utc(binding["child_event"]["time"]),
        "child creation time suggests PID reuse",
    )
    require(
        all(not process.signalled() for process in processes.values()),
        "both bound processes must still be alive at initial attachment",
    )


class WitnessOutput:
    def __init__(self, output, binding, baseline, source_hashes):
        self.root = within(output, exists=False)
        require(not self.root.exists(), "new witness output required")
        self.root.mkdir(parents=True)
        self.inputs = {**binding["immutable_hashes"], **source_hashes}
        self.outputs = {}
        self.binding, self.baseline = binding, baseline
        for index, (path, expected) in enumerate(source_hashes.items()):
            data = Path(path).read_bytes()
            require(digest(data) == expected, "source changed before snapshot")
            self.write(f"source_snapshot/{index:02d}_{Path(path).name}", data)
        self.write("events_baseline.jsonl", baseline)
        self.json("binding.json", binding)

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self.outputs[str(path)] = digest(data)
        return path

    def json(self, name, value):
        return self.write(name, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())

    def final_guard(self):
        guard({**self.inputs, **self.outputs})

    def events(self, name):
        data = Path(self.binding["events_path"]).read_bytes()
        require(data.startswith(self.baseline), "supervisor event prefix changed or shrank")
        rows = event_rows(data)
        subsequent = [
            row
            for row in rows[self.binding["child_event_index"] + 1 :]
            if row.get("event") in ("phase_start", "owned_child_started")
        ]
        path = self.write(name, data)
        return dict(
            path=str(path),
            sha256=digest(data),
            bytes=len(data),
            complete_lines=len(rows),
            partial_trailing_line=bool(data and not data.endswith(b"\n")),
            subsequent_launch_events=subsequent,
            observed_at=now(),
            scope="Events as captured now; not a guarantee about future launches",
        )

    def exit_receipt(self, role, observation, other_signalled):
        capture = self.events(f"events_at_{role}_exit.jsonl")
        payload = dict(
            schema=SCHEMA,
            role=role,
            observation=observation,
            other_role_signalled_at_observation=other_signalled,
            event_capture=capture,
            binding_sha256=self.outputs[str(self.root / "binding.json")],
            semantics="Observed process exit only; no tracking/supervisor success claim",
        )
        self.json(role + "_exit.json", payload)
        self.final_guard()
        self.json(
            role + "_exit.receipt.json",
            dict(
                schema=SCHEMA,
                status="ok",
                role=role,
                finished_at=now(),
                all_evidence_sha_unchanged=True,
                inputs=self.inputs,
                outputs=self.outputs.copy(),
                tracking_success_asserted=False,
                process_signal_or_termination_performed=False,
            ),
        )
        self.final_guard()


def observe(
    binding, sink, *, opener=WindowsProcess, poll_seconds=1.0, sleeper=time.sleep, base_python=None
):
    require(
        isinstance(poll_seconds, (int, float))
        and not isinstance(poll_seconds, bool)
        and math.isfinite(poll_seconds)
        and 0.01 <= poll_seconds <= 5,
        "poll seconds must be finite in [0.01,5]",
    )
    processes = {}
    completed = {}
    try:
        for role in ("owner", "child"):
            processes[role] = opener(binding[role + "_pid"])
        validate_identities(binding, processes, base_python or sys._base_executable)
        sink.json(
            "attachment.json",
            dict(
                schema=SCHEMA,
                acquired_at=now(),
                processes={r: p.identity for r, p in processes.items()},
                runtime=dict(
                    python=sys.version,
                    executable=sys.executable,
                    base_executable=sys._base_executable,
                    platform=platform.platform(),
                ),
                scope=(
                    "Known owner and recorded launcher only; "
                    "no descendant enumeration or exit proof"
                ),
            ),
        )
        sink.final_guard()
        print(
            json.dumps(
                dict(
                    event="attached", owner_pid=binding["owner_pid"], child_pid=binding["child_pid"]
                )
            ),
            flush=True,
        )
        while len(completed) < 2:
            status = {role: process.signalled() for role, process in processes.items()}
            for role in ("child", "owner"):
                if role not in completed and status[role]:
                    observation = processes[role].exit_info()
                    sink.exit_receipt(
                        role, observation, status["owner" if role == "child" else "child"]
                    )
                    completed[role] = observation
                    print(json.dumps(dict(event=role + "_exit", **observation)), flush=True)
            if len(completed) < 2:
                sleeper(poll_seconds)
        sink.final_guard()
        return completed
    finally:
        for process in processes.values():
            process.close()


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
        str(Path(__file__).resolve()): sha(__file__),
        str(OLD_SOURCE): OLD_SOURCE_SHA,
        str(REPO / ".venv/pyvenv.cfg"): sha(REPO / ".venv/pyvenv.cfg"),
    }
    binding, baseline = build_binding(args.supervisor_root, args.child_pid)
    sink = WitnessOutput(args.output, binding, baseline, sources)
    started, status, failure, result = now(), "failed", None, None
    try:
        result = observe(binding, sink, poll_seconds=args.poll_seconds)
        status = "ok"
    except BaseException:
        failure = traceback.format_exc()
        raise
    finally:
        # Never change old records; an interrupted witness preserves completed role receipts.
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
                    stage="capture_owned_long_process_exits",
                    status=status,
                    started_at=started,
                    finished_at=now(),
                    error=failure,
                    observations=result,
                    inputs=sink.inputs,
                    outputs=sink.outputs.copy(),
                    tracking_or_supervisor_success_asserted=False,
                    signals_enumeration_GPU_or_tracking_performed=False,
                ),
            )


if __name__ == "__main__":
    main()

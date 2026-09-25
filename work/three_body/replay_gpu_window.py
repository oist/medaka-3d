"""Read-only scheduling guard for a short replay between long tracking children.

This does not pause, stop, or modify the supervisor. It retains a handle to its
already registered child, waits for that child's successful exit, and refuses
further replay frames once the supervisor records another tracking launch.
It is a scheduling observation, not an exclusive lock on all GPU applications.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from capture_owned_long_exit import WindowsProcess, now

REPO = Path(__file__).resolve().parents[2]
SUPERVISOR = REPO / "work/three_body/run_long_receipted_supervisor.py"
SUPERVISOR_SHA = "ddf95225f99e908e1b065915111766d069badf31f8b048414073c7d5855a9ee6"
PYTHON = REPO / ".venv/Scripts/python.exe"
RUNNER = REPO / "labs/20260909_three_body/run_long_continuation.py"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def workspace(path):
    path = Path(path).resolve()
    require(path.is_relative_to(REPO / "work"), "window path must be in workspace work")
    return path


def complete_events(data):
    """Ignore an unfinished last line; the next check sees the committed line."""
    committed = data[: data.rfind(b"\n") + 1]
    return committed, [json.loads(line) for line in committed.splitlines()]


class ReplayGPUWindow:
    def __init__(self, supervisor_root, child_pid, output_dir, timeout_seconds=7200):
        self.root, self.output = workspace(supervisor_root), workspace(output_dir)
        require(type(child_pid) is int and child_pid > 0, "positive child PID required")
        require(0 < timeout_seconds <= 43200, "invalid scheduling timeout")
        require(self.output.is_dir(), "replay output must already exist")
        self.child_pid, self.timeout_seconds = child_pid, timeout_seconds
        self.child = self.owner = None
        self.baseline = None
        self.opened = False
        self.report = dict(
            schema="read_only_replay_gpu_window_v1", status="created",
            supervisor_root=str(self.root), child_pid=child_pid,
            production_process_signalled_by_guard=False, exclusive_gpu_lock=False,
            limitation="Launch events are checked before each frame; no global GPU lock exists.",
        )

    def _save(self):
        (self.output / "gpu_window.json").write_text(
            json.dumps(self.report, indent=2, allow_nan=False), encoding="utf-8")

    def attach(self):
        try:
            self._attach()
        except BaseException as error:
            self.report.update(status="failed", error=repr(error), failed_at=now())
            self.close()
            raise

    def _attach(self):
        require(self.child is None and self.baseline is None, "window already attached")
        require(sha(SUPERVISOR.read_bytes()) == SUPERVISOR_SHA, "supervisor source changed")
        lock_data = (self.root / "supervisor.lock").read_bytes()
        shared_lock = SUPERVISOR.parent / "long_remaining_receipted_supervisor.lock"
        require(lock_data == shared_lock.read_bytes(),
                "supervisor ownership lock copy differs")
        lock = json.loads(lock_data)
        require(
            lock["schema"] == "long_remaining_receipted_supervisor_v1"
            and Path(lock["output_root"]).resolve() == self.root
            and Path(lock["source_path"]).resolve() == SUPERVISOR.resolve()
            and lock["source_sha256"] == SUPERVISOR_SHA,
            "unexpected supervisor owner binding")
        baseline, rows = complete_events((self.root / "events.jsonl").read_bytes())
        matches = [(i, row) for i, row in enumerate(rows)
                   if row.get("event") == "owned_child_started"
                   and row.get("pid") == self.child_pid]
        require(len(matches) == 1, "child is not uniquely owned in supervisor events")
        index, event = matches[0]
        require(index > 0 and rows[index - 1].get("event") == "phase_start",
                "child lacks registered phase start")
        require(not any(row.get("event") in ("phase_start", "owned_child_started")
                        for row in rows[index + 1:]), "a later tracking launch already exists")
        command = event["command"]
        self.tracking_root = workspace(event["output"])
        require(self.tracking_root.parent.parent == self.root / "phases"
                and self.tracking_root.name == "tracking", "child output has unexpected scope")
        require(len(command) == 15 and command[1] == "-B"
                and Path(command[0]).resolve() == PYTHON.resolve()
                and Path(command[2]).resolve() == RUNNER.resolve()
                and command[3::2] == ["--seed-root", "--input-stage-root", "--input-root",
                                      "--through-frame", "--output", "--resume"]
                and Path(command[12]).resolve() == self.tracking_root,
                "unexpected child program or output")
        registration = self.tracking_root.parent / "registration/registration.provenance.json"
        registration_data = registration.read_bytes()
        seal = json.loads(registration_data)
        require(seal["status"] == "ok" and seal["stage"] == "register_long_receipted_phase_v1",
                "child registration is not successful")
        execution_item = seal["outputs"]["execution"]
        execution_path = self.tracking_root.parent / "registration/execution.json"
        execution_data = execution_path.read_bytes()
        require(Path(execution_item["path"]).resolve() == execution_path
                and sha(execution_data) == execution_item["sha256"]
                and json.loads(execution_data)["command"] == command,
                "registered child command differs")
        try:
            self.owner = WindowsProcess(lock["pid"])
            self.child = WindowsProcess(self.child_pid)
            owner_created = datetime.fromisoformat(self.owner.identity["creation_time_utc"])
            locked_at = datetime.fromisoformat(lock["created_at"])
            require(Path(self.owner.identity["image_path"]).resolve()
                    == Path(sys._base_executable).resolve()
                    and locked_at - timedelta(minutes=5) <= owner_created <= locked_at,
                    "held owner identity does not match registered supervisor")
            require(Path(self.child.identity["image_path"]).resolve() == PYTHON.resolve(),
                    "held child executable differs")
            created = datetime.fromisoformat(self.child.identity["creation_time_utc"])
            start = datetime.fromisoformat(rows[index - 1]["time"])
            recorded = datetime.fromisoformat(event["time"])
            require(start - timedelta(seconds=1) <= created <= recorded + timedelta(seconds=1),
                    "held child creation time does not match owned launch")
            self.baseline = baseline
            (self.output / "gpu_window_events_at_attach.jsonl").write_bytes(baseline)
            self.report.update(
                status="attached_waiting_for_preparation", attached_at=now(),
                owner_identity=self.owner.identity, child_identity=self.child.identity,
                owner_lock_sha256=sha(lock_data), registration_sha256=sha(registration_data),
                registered_execution_sha256=sha(execution_data),
                baseline_events_sha256=sha(baseline), baseline_events_bytes=len(baseline),
                tracking_root=str(self.tracking_root), owned_child_event=event)
            self._check_events()
            self._save()
        except BaseException as error:
            self.report.update(status="failed", error=repr(error), failed_at=now())
            self.close()
            raise

    def _check_events(self):
        require(self.baseline is not None, "window must be attached")
        data = (self.root / "events.jsonl").read_bytes()
        require(data.startswith(self.baseline), "supervisor event prefix changed")
        _, later = complete_events(data[len(self.baseline):])
        require(not any(row.get("event") in ("phase_start", "owned_child_started")
                        for row in later), "next tracking phase has started; replay window ended")
        self.report.update(last_events_check_at=now(), last_events_bytes=len(data),
                           last_events_sha256=sha(data))

    def wait_until_open(self):
        require(self.child is not None and not self.opened, "invalid window wait lifecycle")
        started = time.monotonic()
        self.report.update(status="waiting_for_child_exit", ready_at=now())
        self._save()
        try:
            while not self.child.signalled():
                self._check_events()
                require(not self.owner.signalled(), "supervisor exited before GPU window")
                require(time.monotonic() - started < self.timeout_seconds,
                        "timed out waiting for tracking child exit")
                time.sleep(2)
            info = self.child.exit_info()
            require(info["exit_code_success"], "long tracking child did not exit successfully")
            self._check_events()
            seal_path = self.tracking_root / "continuation.provenance.json"
            seal_data = seal_path.read_bytes()
            seal = json.loads(seal_data)
            require(seal["status"] == "ok" and seal["stage"] == "three_body_long_continuation",
                    "exited child has no successful continuation seal")
            self.opened = True
            self.report.update(status="open", opened_at=now(), child_exit=info,
                               child_continuation_seal_sha256=sha(seal_data),
                               wait_seconds=time.monotonic() - started)
            self.assert_open()
        except BaseException as error:
            self.report.update(status="failed", error=repr(error), failed_at=now())
            self._save()
            raise
        return self.report.copy()

    def assert_open(self):
        require(self.opened and self.child is not None, "GPU replay window is not open")
        try:
            self._check_events()
            require(not self.owner.signalled(), "supervisor exited during replay")
            self.report["last_open_check_at"] = now()
            self._save()
        except BaseException as error:
            self.report.update(status="failed", error=repr(error), failed_at=now())
            self._save()
            raise

    def close(self):
        for name in ("child", "owner"):
            handle = getattr(self, name)
            if handle is not None:
                handle.close()
                setattr(self, name, None)
        self.report["closed_at"] = now()
        if self.report["status"] != "failed":
            self.report["status"] = "closed"
        self._save()

"""Operational continuation of an immutable long tracker; never start detection.

Each ready six-chunk group (last group: two) receives a successful registration
before a fresh child process restores the complete previous checkpoint. A failed
phase stops this supervisor. Locks, partial outputs and previous checkpoints are
never removed or overwritten. Only --execute can launch a tracking process.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work/three_body"
LAB = REPO / "labs/20260909_three_body"
sys.path[:0] = [str(WORK), str(LAB), str(REPO / "src")]
from medaka3d.provenance import provenance  # noqa: E402

FIRST, INITIAL_STOP, FINAL_STOP = 49800, 66400, 348600
CHUNK, PHASE_CHUNKS = 1660, 6
WINDOWS = os.name == "nt"
RUNNER = LAB / "run_long_continuation.py"
PYTHON = REPO / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
SHARED_LOCK = WORK / "long_remaining_supervisor.lock"
WAIT_PARENT = WORK / "long_continuation_through66400_v1"
WAIT_PLAN = WORK / "long_next_phase_plan_v1"
WAIT_PLAN_SHA = "deb79e143eb3473c95d4b0b7ee93e6daf70a5657c15f74b06435d81e34f12aad"
WAIT_SEAL_SHA = "3b2a61745e00b20ea3f16ec37cfe4d9aac7374d042af1aacd1e66df4d18c7425"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def within(path, *, exists=True):
    path = Path(path).resolve(strict=exists)
    require(path.is_relative_to(REPO), "path must remain in workspace")
    return path


def equal(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        b, sort_keys=True, allow_nan=False
    )


def next_stop(stop):
    require(
        type(stop) is int and INITIAL_STOP <= stop < FINAL_STOP and (stop - FIRST) % CHUNK == 0,
        "invalid completed native endpoint",
    )
    return min(stop + PHASE_CHUNKS * CHUNK, FINAL_STOP)


def phase_ranges():
    result, start = [], INITIAL_STOP
    while start < FINAL_STOP:
        stop = next_stop(start)
        result.append([start, stop])
        start = stop
    return result


def acquire_lock(output, parent):
    """Exclusive creation; even a stale existing lock is never taken over."""
    path = SHARED_LOCK
    payload = dict(
        pid=os.getpid(),
        created_at=datetime.now(UTC).isoformat(),
        parent=str(parent),
        output_root=str(output),
        full_target_interval=[FIRST, FINAL_STOP],
        source_sha256=sha256(__file__),
        removal_policy="never removed or automatically taken over",
    )
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def sources():
    paths = {
        Path(__file__).resolve(),
        RUNNER,
        LAB / "long_tracking_protocol.json",
        LAB / "long_input_protocol.json",
    }
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and any(
                path.is_relative_to(scope) for scope in (LAB, REPO / "src", WORK)
            ):
                paths.add(path)
    return {str(path): sha256(path) for path in sorted(paths)}


def unchanged(hashes):
    for path, digest in hashes.items():
        require(sha256(path) == digest, "source/output changed: " + path)


def header(root, expected_stop):
    """Cheap startup/readiness context; full checkpoint validation is mandatory later."""
    root = within(root)
    seal = read(root / "continuation.provenance.json")
    require(
        seal.get("stage") == "three_body_long_continuation"
        and seal.get("status") == "ok"
        and seal["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
        "successful guarded parent required",
    )
    item = seal["outputs"]["summary"]
    path = root / "summary.json"
    require(
        within(item["path"]) == path and sha256(path) == item["sha256"],
        "parent summary is not bound",
    )
    summary = read(path)
    require(
        summary["schema"] == "long_continuation_phase_v1"
        and summary["technical_completion"] is True
        and summary["processed_interval"] == [FIRST, expected_stop]
        and summary["logical_initialization_count"] == 1
        and summary["external_pose_reset_count"] == 0
        and summary["active_beam_count"] == 6
        and seal["params"]["through_frame"] == expected_stop,
        "parent is not the declared full-beam continuation",
    )
    binding = seal["params"]["bindings"]
    require(
        binding["inputs"]["full_frame_range"] == [FIRST, FINAL_STOP],
        "different registered recording",
    )
    return binding, summary


def wait_for_parent(parent, poll_seconds, code, *, sleeper=time.sleep):
    require(parent == WAIT_PARENT.resolve(), "waiting is restricted to the registered 66400 parent")
    plan_path, seal_path = WAIT_PLAN / "plan.json", WAIT_PLAN / "registration.provenance.json"
    hashes = {str(plan_path): WAIT_PLAN_SHA, str(seal_path): WAIT_SEAL_SHA}
    unchanged(hashes)
    plan, seal = read(plan_path), read(seal_path)
    require(
        seal["status"] == "ok"
        and seal["stage"] == "register_long_next_phase_execution"
        and within(seal["outputs"]["plan"]["path"]) == plan_path
        and seal["outputs"]["plan"]["sha256"] == WAIT_PLAN_SHA
        and plan["expected_output"] == str(parent)
        and plan["next_native_interval"] == [59760, INITIAL_STOP],
        "waiting parent differs from its successful registration",
    )
    print(json.dumps(dict(event="waiting_for_registered_parent", parent=str(parent))), flush=True)
    path = parent / "continuation.provenance.json"
    while True:
        try:
            record = read(path)
        except (FileNotFoundError, json.JSONDecodeError):
            record = {}
        require(record.get("status") != "failed", "registered parent failed; supervisor stops")
        if record.get("status") == "ok":
            require(
                record.get("stage") == "three_body_long_continuation",
                "unexpected parent completion stage",
            )
            unchanged({**code, **hashes})
            print(
                json.dumps(dict(event="registered_parent_complete", parent=str(parent))), flush=True
            )
            return hashes
        sleeper(poll_seconds)


def successful_chunk_hints(stage, first, stop):
    """Small completion records only; a hint never substitutes for full_load."""
    ready = set()
    for start in range(first, stop, CHUNK):
        end = start + CHUNK
        parent = stage / "chunks" / f"f{start:06d}-{end:06d}"
        if not parent.is_dir():
            continue
        for attempt in sorted(parent.iterdir()):
            if not re.fullmatch(r"attempt_\d{4,}", attempt.name) or not attempt.is_dir():
                continue
            require(
                not attempt.is_symlink() and attempt.resolve().is_relative_to(stage),
                "foreign detector attempt",
            )
            try:
                seal = read(attempt / "staging.provenance.json")
            except (FileNotFoundError, json.JSONDecodeError):
                continue  # Producer may still be writing its completion record.
            if seal.get("status") == "ok" and seal.get("stage") == "prepare_independent_mask_chunk":
                ready.add((start, end))
                break
    return ready


def wait_for_chunks(stage, first, stop, poll_seconds, emit, *, sleeper=time.sleep):
    expected = {(start, start + CHUNK) for start in range(first, stop, CHUNK)}
    announced = set()
    while True:
        ready = successful_chunk_hints(stage, first, stop)
        for start, end in sorted(ready - announced):
            emit(
                "chunk_completion_hint",
                start=start,
                stop=end,
                scope="Full immutable payload verification still required",
            )
        announced |= ready
        if ready == expected:
            return
        sleeper(poll_seconds)


def command_for(parent, stop, output, binding):
    inputs = binding["inputs"]
    seed = inputs["seed"]
    return [
        str(PYTHON.resolve()),
        "-B",
        str(RUNNER.resolve()),
        "--seed-root",
        str(within(seed["registration_path"]).parent),
        "--input-stage-root",
        str(within(inputs["input_stage_root"])),
        "--input-root",
        str(within(seed["input_root"])),
        "--through-frame",
        str(stop),
        "--output",
        str(output),
        "--resume",
        str(parent),
    ]


def stop_owned_child(process, info):
    """Stop our live owned process tree; a hard-killed supervisor cannot run cleanup."""
    if process.poll() is not None:
        return
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        info["termination_requested"] = True
        if WINDOWS:
            # A Windows venv executable launches a second Python process. Stopping
            # only the Popen launcher is insufficient; never use that fallback.
            command = [
                str(Path(os.environ["SYSTEMROOT"]) / "System32/taskkill.exe"),
                "/PID",
                str(process.pid),
                "/T",
                "/F",
            ]
            outcome = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            info["tree_termination_command"] = command
            info["tree_termination_returncode"] = outcome.returncode
            info["tree_termination_stdout"] = outcome.stdout
            info["tree_termination_stderr"] = outcome.stderr
            require(
                outcome.returncode == 0,
                "owned tree termination denied/failed; no further phase or launcher-only fallback",
            )
            process.wait(timeout=10)
        else:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                info["kill_requested"] = True
                process.kill()
                process.wait(timeout=10)
        info["terminated_returncode"] = process.returncode
    finally:
        signal.signal(signal.SIGINT, previous)


def run_child(command, log_path, info, *, popen=subprocess.Popen, on_started=None):
    process = None
    try:
        with log_path.open("xb") as stream:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            process = popen(
                command,
                cwd=REPO,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                shell=False,
                creationflags=flags,
            )
            info["pid"] = process.pid
            info["started_at"] = datetime.now(UTC).isoformat()
            if on_started is not None:
                on_started(info)
            code = process.wait()
            info["returncode"] = code
            return code
    except BaseException:
        if process is not None:
            stop_owned_child(process, info)
        raise
    finally:
        info["finished_at"] = datetime.now(UTC).isoformat()
        info["child_alive_after_return"] = process is not None and process.poll() is None
        info["alive_flag_scope"] = (
            "Direct Popen handle; hard-killed-supervisor orphan cleanup is not guaranteed"
        )


def save_json(path, value, record, role=None):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    record.output(role or path.name, path, sha256=sha256(path))


def snapshot(code, root, record):
    for path, digest in code.items():
        target = root / "code_snapshot" / Path(path).relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        require(sha256(target) == digest, "source snapshot mismatch")
        record.output(target.relative_to(root).as_posix(), target, sha256=digest)


def final_outputs(record):
    unchanged({item["path"]: item["sha256"] for item in record.outputs.values()})


def audit_bootstrap(backend, parent, output, code):
    evidence = backend.Digests()
    root = output / "bootstrap"
    root.mkdir(exist_ok=False)
    with provenance("long_supervised_bootstrap_audit", root, name="audit") as record:
        for path, digest in code.items():
            evidence.add(path, digest)
        result = backend.verify_bootstrap(parent, evidence)
        require(result["status"] == "ok", "bootstrap audit did not succeed")
        require(equal(sources(), code), "late workspace imports or modified source")
        for path, digest in evidence.paths.items():
            record.input(path, path, sha256=digest)
        snapshot(code, root, record)
        save_json(root / "summary.json", result, record, "summary")
        evidence.final_guard()
        final_outputs(record)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
        record.param("tracking_detector_GPU_or_GT_performed", False)
    return root


def bind_parent_audit(parent, audit_root, evidence):
    """The prior independent audit must bind this exact parent's candidate."""
    if (audit_root / "audit.provenance.json").is_file():
        seal_path, stage, name, role = (
            audit_root / "audit.provenance.json",
            "long_supervised_bootstrap_audit",
            "summary.json",
            "summary",
        )
    else:
        seal_path, stage, name, role = (
            audit_root / "operation.provenance.json",
            "long_supervised_phase_operation",
            "audit.json",
            "audit",
        )
    evidence.add(seal_path)
    seal = read(seal_path)
    require(
        seal["status"] == "ok"
        and seal["stage"] == stage
        and seal["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
        "successful independent parent audit required",
    )
    summary_path = audit_root / name
    item = seal["outputs"][role]
    require(within(item["path"]) == summary_path, "parent audit summary role differs")
    evidence.add(summary_path, item["sha256"])
    audit = read(summary_path)
    parent_seal_path = parent / "continuation.provenance.json"
    evidence.add(parent_seal_path)
    parent_seal = read(parent_seal_path)
    require(
        audit["status"] == "ok"
        and audit["run"] == str(parent)
        and audit["candidate_sha256"] == parent_seal["outputs"]["candidate"]["sha256"],
        "independent audit belongs to a different parent/candidate",
    )
    return dict(
        root=str(audit_root),
        seal_sha256=evidence.paths[str(seal_path)],
        summary_sha256=evidence.paths[str(summary_path)],
        parent_candidate_sha256=audit["candidate_sha256"],
    )


def register_phase(backend, parent, stop, phase_root, code, binding, parent_audit):
    child_root = phase_root / "tracking"
    evidence = backend.Digests()
    for path, digest in code.items():
        evidence.add(path, digest)
    lineage = bind_parent_audit(parent, parent_audit, evidence)
    plan, context = backend.prepare_phase(parent, stop, child_root, evidence)
    del context
    gc.collect()
    require(equal(sources(), code), "late workspace imports or modified source")
    require(equal(plan["frozen_input_and_code_binding"], binding), "fixed binding changed")
    command = command_for(parent, stop, child_root, binding)
    require(
        plan["parent_phase"] == str(parent)
        and plan["expected_output"] == str(child_root)
        and plan["output_interval"] == [FIRST, stop],
        "phase plan command differs",
    )
    registration = phase_root / "registration"
    registration.mkdir(parents=True, exist_ok=False)
    with provenance("register_long_supervised_phase", registration, name="registration") as record:
        for path, digest in evidence.paths.items():
            record.input(path, path, sha256=digest)
        snapshot(code, registration, record)
        save_json(registration / "plan.json", plan, record, "plan")
        save_json(
            registration / "execution.json",
            dict(
                command=command,
                source_hashes=code,
                parent=str(parent),
                output=str(child_root),
                through_frame=stop,
                independent_parent_audit=lineage,
            ),
            record,
            "execution",
        )
        evidence.final_guard()
        final_outputs(record)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
        record.param("tracking_detector_GPU_or_GT_performed", False)
    return plan, child_root, registration, command


def execute_phase(backend, phase_root, plan, child_root, registration, command, code, emit):
    log_path, info = phase_root / "tracking.log", {}
    evidence = backend.Digests()
    with provenance("long_supervised_phase_operation", phase_root, name="operation") as record:
        for path, digest in code.items():
            evidence.add(path, digest)
        seal = read(registration / "registration.provenance.json")
        require(seal["status"] == "ok", "successful phase registration required before launch")
        for entry in seal["outputs"].values():
            evidence.add(entry["path"], entry["sha256"])
        evidence.add(registration / "registration.provenance.json")
        for path, digest in evidence.paths.items():
            record.input(path, path, sha256=digest)
        require(equal(read(registration / "plan.json"), plan), "phase plan changed before launch")
        execution = read(registration / "execution.json")
        require(
            execution["command"] == command and execution["source_hashes"] == code,
            "registered command/source changed before launch",
        )
        evidence.final_guard()
        emit(
            "phase_start",
            parent=plan["parent_phase"],
            output=str(child_root),
            interval=plan["next_native_interval"],
        )
        try:
            native_exit = run_child(
                command,
                log_path,
                info,
                on_started=lambda state: emit(
                    "owned_child_started",
                    pid=state["pid"],
                    command=command,
                    log=str(log_path),
                    output=str(child_root),
                    force_kill_of_supervisor_does_not_guarantee_child_cleanup=True,
                ),
            )
            require(native_exit == 0, "tracking child failed; phase will not be retried")
            require(info["child_alive_after_return"] is False, "owned child is still alive")
            child_seal = read(child_root / "continuation.provenance.json")
            require(
                datetime.fromisoformat(seal["finished_at"])
                <= datetime.fromisoformat(child_seal["started_at"]),
                "phase was registered after the child started",
            )
            audited = backend.verify_phase(child_root, plan, evidence)
            save_json(phase_root / "audit.json", audited, record, "audit")
            for path, digest in evidence.paths.items():
                record.input(path, path, sha256=digest)
            evidence.final_guard()
            unchanged(code)
            record.param("all_input_source_snapshot_output_sha_unchanged", True)
            record.param("tracking_child_exit_code", native_exit)
        finally:
            save_json(phase_root / "child_execution.json", info, record, "child_execution")
            if log_path.is_file():
                record.output("tracking_log", log_path, sha256=sha256(log_path))
            final_outputs(record)
        emit("phase_complete", output=str(child_root), interval=plan["next_native_interval"])
    return audited


def supervise(args, backend):
    parent, output = within(args.parent, exists=False), within(args.output, exists=False)
    require(not output.exists(), "new supervisor root required; no automatic takeover")
    require(not SHARED_LOCK.exists(), "shared supervisor lock exists; no automatic takeover")
    code = sources()
    waiting_hashes = (
        wait_for_parent(parent, args.poll_seconds, code) if args.wait_for_parent else {}
    )
    binding, _ = header(parent, INITIAL_STOP)
    unchanged(code)
    require(str(RUNNER) in binding["code_sha256"], "frozen runner is not registered")
    unchanged(binding["code_sha256"])
    lock = acquire_lock(output, parent)
    output.mkdir(parents=True, exist_ok=False)
    lock_copy = output / "supervisor.lock"
    shutil.copyfile(lock, lock_copy)
    require(sha256(lock_copy) == sha256(lock), "shared lock evidence copy differs")
    events = output / "events.jsonl"
    with events.open("x", encoding="utf-8", buffering=1) as stream:

        def emit(kind, **fields):
            row = dict(event=kind, time=datetime.now(UTC).isoformat(), **fields)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(row, allow_nan=False), flush=True)

        with provenance("long_remaining_supervisor", output, name="supervisor") as record:
            record.params_from(args)
            for path, digest in code.items():
                record.input(path, path, sha256=digest)
            for path, digest in waiting_hashes.items():
                record.input(path, path, sha256=digest)
            for name in ("continuation.provenance.json", "summary.json"):
                path = parent / name
                record.input(str(path), path, sha256=sha256(path))
            record.input("shared_supervisor_lock", lock, sha256=sha256(lock))
            record.output("lock", lock_copy, sha256=sha256(lock_copy))
            snapshot(code, output, record)
            policy = dict(
                schema="long_remaining_operational_supervisor_v1",
                initial_parent=str(parent),
                initial_stop=INITIAL_STOP,
                target_interval=[FIRST, FINAL_STOP],
                phase_intervals=phase_ranges(),
                fixed_bindings=binding,
                source_hashes=code,
                detector_started=False,
                new_external_pose_seed=False,
                failed_phase_retry=False,
                lock_removed=False,
                scheduling_changes_scientific_protocol=False,
                note="Each child rechecks its registered runtime and restores all beams",
            )
            save_json(output / "supervisor_plan.json", policy, record, "plan")
            stop, completed = INITIAL_STOP, []
            try:
                bootstrap = audit_bootstrap(backend, parent, output, code)
                parent_audit = bootstrap
                for name in ("summary.json", "audit.provenance.json"):
                    path = bootstrap / name
                    record.output(path.relative_to(output).as_posix(), path, sha256=sha256(path))
                while stop < FINAL_STOP:
                    through = next_stop(stop)
                    stage = within(binding["inputs"]["input_stage_root"])
                    wait_for_chunks(stage, stop, through, args.poll_seconds, emit)
                    unchanged(code)
                    phase_root = output / "phases" / f"f{stop:06d}-{through:06d}"
                    phase_root.mkdir(parents=True, exist_ok=False)
                    plan, child_root, registration, command = register_phase(
                        backend, parent, through, phase_root, code, binding, parent_audit
                    )
                    record.param("active_phase", str(phase_root))
                    try:
                        execute_phase(
                            backend, phase_root, plan, child_root, registration, command, code, emit
                        )
                    finally:
                        for relative in (
                            "registration/registration.provenance.json",
                            "registration/plan.json",
                            "registration/execution.json",
                            "operation.provenance.json",
                            "audit.json",
                            "child_execution.json",
                        ):
                            path = phase_root / relative
                            if path.is_file():
                                record.output(
                                    path.relative_to(output).as_posix(), path, sha256=sha256(path)
                                )
                    child_binding, summary = header(child_root, through)
                    require(equal(child_binding, binding), "child changed the fixed binding")
                    require(
                        summary["full_sequence_complete"] is (through == FINAL_STOP)
                        and summary["prefix_backtrace_provisional"] is (through != FINAL_STOP),
                        "terminal completion flags differ",
                    )
                    completed.append(dict(parent=str(parent), run=str(child_root), through=through))
                    parent, stop, parent_audit = child_root, through, phase_root
                save_json(
                    output / "summary.json",
                    dict(
                        status="ok",
                        final_run=str(parent),
                        through=stop,
                        full_sequence_complete=True,
                        phases=completed,
                        logical_initialization_count=1,
                        external_pose_reset_count=0,
                        independent_ID_truth=False,
                    ),
                    record,
                    "summary",
                )
                record.param("all_input_source_snapshot_output_sha_unchanged", True)
            finally:
                stream.flush()
                record.output("events", events, sha256=sha256(events))
                unchanged(code)
                require(sha256(lock) == sha256(lock_copy), "shared supervisor lock changed")
                unchanged({item["path"]: item["sha256"] for item in record.inputs.values()})
                final_outputs(record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--wait-for-parent",
        action="store_true",
        help="Wait only for the already-registered through66400 parent",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Launch the already-authorized frozen tracking children",
    )
    args = parser.parse_args(argv)
    require(args.execute, "--execute is required; help/tests never launch tracking")
    require(1 <= args.poll_seconds <= 60, "poll interval must be between 1 and 60 seconds")
    require(Path(sys.executable).resolve() == PYTHON.resolve(), "use the registered main .venv")
    backend = importlib.import_module("long_phase_audit")
    supervise(args, backend)


if __name__ == "__main__":
    main()

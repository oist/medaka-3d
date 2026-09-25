"""Continue only a completed, independently witnessed long-tracker handoff.

This operational wrapper never stops an old process or starts detection. It
uses a new exclusive lock, preserves the frozen tracker command and restores
all saved beams. A stop_after_phase.request is honored only when no tracking
child is owned. Failed phases and locks are retained; there is no retry/takeover.
Requests arriving after the final explicit pre-operation check are honored
after that operation finishes, including its internal prelaunch SHA checks.
Only --execute permits bootstrap verification and subsequent child launches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work/three_body"
sys.path[:0] = [str(WORK), str(REPO / "labs/20260909_three_body"), str(REPO / "src")]

import receipted_long_handoff as handoff  # noqa: E402
import receipted_long_phase_operation as operation  # noqa: E402
from receipted_long_phase_audit import Digests  # noqa: E402
from run_long_remaining_supervisor import (  # noqa: E402
    CHUNK,
    FINAL_STOP,
    FIRST,
    PYTHON,
    RUNNER,
    equal,
    final_outputs,
    header,
    next_stop,
    provenance,
    read,
    require,
    save_json,
    sha256,
    snapshot,
    sources,
    successful_chunk_hints,
    unchanged,
    within,
)

STAGE = "long_remaining_receipted_supervisor_v1"
INITIAL_STOP = 76360
SHARED_LOCK = WORK / "long_remaining_receipted_supervisor.lock"
STOP_NAME = "stop_after_phase.request"
POLL_SECONDS = 30
PHASE_ARTIFACTS = (
    "registration/registration.provenance.json", "registration/plan.json",
    "registration/execution.json", "registration/mask_validation_receipts.json",
    "operation.provenance.json", "audit.json", "mask_validation_receipts.json",
    "child_execution.json", "tracking.log",
)


def phase_ranges():
    result, start = [], INITIAL_STOP
    while start < FINAL_STOP:
        stop = next_stop(start)
        result.append([start, stop])
        start = stop
    return result


def acquire_lock(output, parent):
    """New shared x-lock; neither the old nor this lock is removed or adopted."""
    payload = dict(
        schema=STAGE, pid=os.getpid(), created_at=datetime.now(UTC).isoformat(),
        parent=str(parent), output_root=str(output), full_target_interval=[FIRST, FINAL_STOP],
        source_path=str(Path(__file__).resolve()), source_sha256=sha256(__file__),
        old_lock_removed=False, removal_policy="never removed or automatically taken over",
    )
    with SHARED_LOCK.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return SHARED_LOCK


def checked_header(root, stop, *, binding=None, restores=None):
    actual, summary = header(root, stop)
    require(binding is None or equal(actual, binding), "child changed the fixed binding")
    count = summary["checkpoint_restore_count"]
    require(type(count) is int and count >= 1, "invalid checkpoint restore count")
    require(restores is None or count == restores, "checkpoint restore did not increase by one")
    require(summary["full_sequence_complete"] is (stop == FINAL_STOP)
            and summary["prefix_backtrace_provisional"] is (stop != FINAL_STOP),
            "terminal completion flags differ")
    return actual, summary


def validate_audit(audit, parent, stop, restores):
    require(audit["status"] == "ok" and audit["run"] == str(parent)
            and audit["frame_range"] == [FIRST, stop]
            and audit["all_39_checkpoint_arrays_verified"] is True
            and len(audit["all_21_candidate_arrays_exact_backtrace"]) == 21
            and audit["history"]["active_beam_count"] == 6
            and audit["logical_initialization_count"] == 1
            and audit["checkpoint_restore_count"] == restores
            and audit["external_pose_reset_count"] == 0
            and audit["full_sequence_complete"] is (stop == FINAL_STOP)
            and audit["prefix_backtrace_provisional"] is (stop != FINAL_STOP),
            "phase audit differs from the full-beam continuation contract")


def capture_stop_request(output, record, emit):
    """Preserve arbitrary request bytes; this never signals or terminates a child."""
    path = output / STOP_NAME
    if not path.exists():
        return None
    require(path.is_file() and not path.is_symlink(), "stop request must be a local regular file")
    digest = sha256(path)
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, "stop request changed while read")
    copied = output / "honored_stop_request.bin"
    with copied.open("xb") as stream:
        stream.write(data)
    record.input("graceful_stop_request", path, sha256=digest)
    record.output("honored_stop_request", copied, sha256=digest)
    info = dict(path=str(path), sha256=digest, size=len(data),
                reason=data.decode("utf-8", errors="replace"),
                reason_decoding="UTF-8 with replacement; exact bytes preserved separately",
                child_interrupted=False, boundary="completed, independently audited parent")
    save_json(output / "stop_request.json", info, record, "stop_request")
    emit("stop_request_honored", path=str(path), sha256=digest)
    return info


def wait_for_ready(stage, first, stop, emit, check_stop, *, sleeper=time.sleep):
    expected = {(start, start + CHUNK) for start in range(first, stop, CHUNK)}
    reported = set()
    while True:
        requested = check_stop()
        if requested is not None:
            return requested
        ready = successful_chunk_hints(stage, first, stop)
        require(ready.issubset(expected), "chunk readiness outside requested phase")
        for interval in sorted(ready - reported):
            emit("detector_chunk_complete_hint", interval=list(interval))
        reported |= ready
        if ready == expected:
            return None
        sleeper(POLL_SECONDS)


def collect_phase_outputs(root, output, record):
    for relative in PHASE_ARTIFACTS:
        path = root / relative
        if path.is_file():
            record.output(path.relative_to(output).as_posix(), path, sha256=sha256(path))


def result_summary(parent, stop, restores, completed, *, outcome, requested, active_phase,
                   bootstrap_completed, initial_parent):
    complete = bootstrap_completed and stop == FINAL_STOP and outcome == "complete"
    return dict(
        schema=STAGE, status=outcome, final_run=str(parent) if bootstrap_completed else None,
        through=stop if bootstrap_completed else None,
        bootstrap_completed=bootstrap_completed, intended_initial_parent=str(initial_parent),
        full_sequence_complete=complete, prefix_backtrace_provisional=not complete,
        phases=completed, logical_initialization_count=1 if bootstrap_completed else None,
        checkpoint_restore_count=restores if bootstrap_completed else None,
        external_pose_reset_count=0 if bootstrap_completed else None,
        active_beam_count=6 if bootstrap_completed else None, stop_request=requested,
        incomplete_phase=active_phase, independent_ID_truth=False,
        next_phase_retried=False, detector_started=False, old_process_stopped=False,
        note="final_run is the last independently audited checkpoint, or null before bootstrap; "
             "prefix paths are not joined",
    )


def supervise(args, *, sleeper=time.sleep):
    require(args.execute, "--execute is required; no implicit operational launch")
    output = within(args.output, exists=False)
    witness_root = within(args.witness_root)
    receipt_root = within(args.receipt_root, exists=False)
    require(not output.exists(), "new supervisor root required; no automatic takeover")
    require(not SHARED_LOCK.exists(), "shared receipted supervisor lock exists; no takeover")
    code = sources()
    evidence = Digests()
    transferred = handoff.verify_handoff(witness_root, evidence)
    parent = within(transferred["parent"])
    binding, initial_summary = checked_header(parent, INITIAL_STOP)
    require(equal(transferred["binding"], binding), "handoff scientific binding differs")
    require(str(RUNNER) in binding["code_sha256"], "frozen runner is not registered")
    unchanged(binding["code_sha256"])
    evidence.final_guard()
    require(equal(sources(), code), "late imports or source changes during handoff verification")
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

        with provenance(STAGE, output, name="supervisor") as record:
            record.params_from(args)
            for path, digest in {**code, **evidence.paths}.items():
                record.input(path, path, sha256=digest)
            for name in ("continuation.provenance.json", "summary.json"):
                path = parent / name
                record.input(str(path), path, sha256=sha256(path))
            record.input("shared_supervisor_lock", lock, sha256=sha256(lock))
            record.output("lock", lock_copy, sha256=sha256(lock_copy))
            snapshot(code, output, record)
            save_json(output / "supervisor_plan.json", dict(
                schema=STAGE, initial_parent=str(parent), initial_stop=INITIAL_STOP,
                target_interval=[FIRST, FINAL_STOP], phase_intervals=phase_ranges(),
                fixed_bindings=binding, source_hashes=code, handoff_lineage=transferred["lineage"],
                receipt_cache_root=str(receipt_root), receipt_evidence_outside_child_binding=True,
                detector_started=False, old_process_stopped=False, new_external_pose_seed=False,
                failed_phase_retry=False, old_lock_removed=False, new_lock_removed=False,
                scheduling_changes_scientific_protocol=False,
                stop_request_name=STOP_NAME, stop_request_only_at_completed_boundaries=True,
                stop_arriving_during_operation_prelaunch_checks_waits_until_operation_complete=True,
                forced_supervisor_termination_guarantees_child_cleanup=False,
                note="Owned-child exceptions use frozen run_child cleanup; no old PID operations",
            ), record, "plan")
            stop, restores = INITIAL_STOP, initial_summary["checkpoint_restore_count"]
            completed, requested, active_phase = [], None, None
            initial_parent, bootstrap_completed = parent, False
            outcome = "failed"
            try:
                bootstrap_root = output / "bootstrap"
                try:
                    bootstrap = handoff.audit_handoff(
                        witness_root, bootstrap_root, code, receipt_root=receipt_root)
                finally:
                    for name in ("audit.provenance.json", "summary.json", "handoff.json",
                                 "mask_validation_receipts.json"):
                        path = bootstrap_root / name
                        if path.is_file():
                            record.output(path.relative_to(output).as_posix(), path,
                                          sha256=sha256(path))
                require(within(bootstrap) == bootstrap_root,
                        "handoff returned a different bootstrap")
                bootstrap_evidence = Digests()
                lineage = operation.bind_parent_audit(parent, bootstrap, bootstrap_evidence)
                require(lineage["stage"] == "long_receipted_handoff_bootstrap_v1",
                        "explicit new handoff bootstrap required")
                validate_audit(read(bootstrap / "summary.json"), parent, stop, restores)
                bootstrap_evidence.final_guard()
                for path, digest in bootstrap_evidence.paths.items():
                    record.input(path, path, sha256=digest)
                bootstrap_completed = True
                parent_audit = bootstrap
                emit("handoff_bootstrap_complete", parent=str(parent), through=stop)
                while stop < FINAL_STOP:
                    through = next_stop(stop)
                    stage = within(binding["inputs"]["input_stage_root"])
                    requested = wait_for_ready(
                        stage, stop, through, emit,
                        lambda: capture_stop_request(output, record, emit), sleeper=sleeper)
                    if requested is not None:
                        break
                    require(equal(sources(), code), "source set changed before phase registration")
                    phase_root = output / "phases" / f"f{stop:06d}-{through:06d}"
                    phase_root.mkdir(parents=True, exist_ok=False)
                    active_phase = str(phase_root)
                    record.param("active_phase", active_phase)
                    try:
                        plan, child, registration, command = operation.register_phase(
                            parent, through, phase_root, code, binding, parent_audit,
                            receipt_root=receipt_root)
                        requested = capture_stop_request(output, record, emit)
                        if requested is not None:
                            break  # Successful unused registration remains evidence, not a child.
                        audited = operation.execute_phase(
                            phase_root, plan, child, registration, command, code, emit,
                            receipt_root=receipt_root)
                    finally:
                        collect_phase_outputs(phase_root, output, record)
                    child_binding, child_summary = checked_header(
                        child, through, binding=binding, restores=restores + 1)
                    validate_audit(audited, child, through, restores + 1)
                    require(equal(child_binding, binding), "scientific binding changed")
                    completed.append(dict(parent=str(parent), run=str(child), through=through,
                                          parent_audit=str(parent_audit), operation=str(phase_root),
                                          checkpoint_restore_count=restores + 1))
                    parent, stop, parent_audit = child, through, phase_root
                    restores = child_summary["checkpoint_restore_count"]
                    active_phase = None
                outcome = "complete" if stop == FINAL_STOP else "stopped_after_completed_phase"
            finally:
                # Failed or interrupted operations never promote an unaudited child.
                save_json(output / "summary.json", result_summary(
                    parent, stop, restores, completed, outcome=outcome,
                    requested=requested, active_phase=active_phase,
                    bootstrap_completed=bootstrap_completed, initial_parent=initial_parent),
                    record, "summary")
                stream.flush()
                record.output("events", events, sha256=sha256(events))
                require(equal(sources(), code), "late imports or modified supervisor sources")
                unchanged(code)
                require(sha256(lock) == sha256(lock_copy), "shared supervisor lock changed")
                unchanged({item["path"]: item["sha256"] for item in record.inputs.values()})
                final_outputs(record)
                record.param("all_input_source_snapshot_output_sha_unchanged", True)
                record.param("full_sequence_complete", outcome == "complete" and stop == FINAL_STOP)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    require(args.execute, "--execute is required; help/tests never launch tracking")
    require(Path(sys.executable).resolve() == PYTHON.resolve(), "use the registered main .venv")
    supervise(args)


if __name__ == "__main__":
    main()

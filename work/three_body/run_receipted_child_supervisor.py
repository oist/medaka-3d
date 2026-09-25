"""Continue a sealed, held-exit old prefix with the qualified receipt child.

Fresh registration and tracking interpreters preserve the child's fixed source
closure. No old PID operation, detector, lock takeover, retry or pose seed. A
graceful request waits for the complete registration/tracking/audit operation.
The reviewed final-chain and operation source pins are checked before bootstrap.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work/three_body"
sys.path[:0] = [str(WORK), str(REPO / "labs/20260909_three_body"), str(REPO / "src")]

import long_final_receipted_child_chain as chain  # noqa: E402
import receipted_runner_phase_operation as operation  # noqa: E402

m, child, auditor = operation.m, operation.child, operation.auditor
require, equal = m.require, m.equal
current = chain.current
STAGE = "long_remaining_receipted_child_supervisor_v1"
SHARED_LOCK = WORK / "long_remaining_receipted_child_supervisor.lock"
FINAL_CHAIN_SHA = "fa5a6b13b63ceec32b24abbdaa5ee6ca8ac464c0fc9be297feae15020546cdfb"
OPERATION_SHA = "6cee76cbf18088a5ae7595222f28343db20a1db2bf9089344238c4328a662bdc"


def remaining_phases(stop):
    intervals = [list(x) for x in chain.expected_new_phases()]
    matches = [i for i, (start, _) in enumerate(intervals) if start == stop]
    require(type(stop) is int and len(matches) == 1, "completed old phase boundary required")
    return intervals[matches[0] :]


def sources(evidence):
    require(
        isinstance(FINAL_CHAIN_SHA, str) and len(FINAL_CHAIN_SHA) == 64,
        "final-chain source review pin has not been set",
    )
    evidence.add(chain.__file__, FINAL_CHAIN_SHA)
    evidence.add(operation.__file__, OPERATION_SHA)
    return operation.operational_sources(evidence)


def acquire_lock(output, parent):
    payload = dict(
        schema=STAGE,
        pid=os.getpid(),
        created_at=datetime.now(UTC).isoformat(),
        parent=str(parent),
        output_root=str(output),
        full_target_interval=[chain.FIRST, chain.FINAL_STOP],
        source_path=str(Path(__file__).resolve()),
        source_sha256=m.sha(__file__),
        old_lock_removed=False,
        removal_policy="never removed or automatically taken over",
    )
    with SHARED_LOCK.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return SHARED_LOCK


def handoff_data(prefix, args):
    counts = prefix["prefix_counts"]
    return dict(
        schema=chain.HANDOFF_SCHEMA,
        old_supervisor=str(m.local(args.old_supervisor)),
        old_supervisor_sha256=args.old_supervisor_sha256,
        exit_witness=dict(path=str(m.local(args.exit_witness)), sha256=args.exit_witness_sha256),
        old_completed_phases=counts["completed_phases"],
        initial_parent=str(prefix["final_run"]),
        initial_stop=counts["through"],
        initial_restores=counts["checkpoint_restores"],
        prefix_counts=counts,
        unused_registration=prefix["unused_registration"],
        scientific_context_changed=False,
        owner_exit=prefix["owner_exit"],
        new_external_pose_seed=False,
    )


def collect_operation(root, output, record):
    """The owner binds every operation output, including failed/unused artifacts."""
    path = root / "operation.provenance.json"
    if not path.is_file():
        return None
    item = m.read(path)
    for value in item["outputs"].values():
        target = m.local(value["path"])
        require(target.is_relative_to(root), "operation output escaped its phase")
        require(m.sha(target) == value["sha256"], "operation output changed")
        record.output(target.relative_to(output).as_posix(), target, sha256=value["sha256"])
    record.output(path.relative_to(output).as_posix(), path, sha256=m.sha(path))
    return item


def completed_operation(root, returned, parent, start, stop, restores, binding, record):
    """Promote only the exact sealed result of the separate CPU phase audit."""
    require(
        record is not None
        and record.get("stage") == operation.STAGE
        and record.get("status") == "ok"
        and record["params"].get("all_input_source_snapshot_output_sha_unchanged") is True
        and record["params"].get("tracking_child_exit_code") == 0,
        "operation did not seal successfully",
    )
    run, reg, audit_root = root / "tracking", root / "registration", root / "audit"
    summary = m.read(root / "summary.json")
    for key, expected in dict(
        status="ok",
        completed_run=str(run),
        independent_audit=str(audit_root),
        child_started=True,
        parent_promoted=False,
        operation_success_requires_successful_seal=True,
        automatic_retry=False,
        execution_resource_owner_external=True,
    ).items():
        equal(summary.get(key), expected, "operation summary differs: " + key)
    outputs = record["outputs"]

    def named(role, path):
        item = outputs[role]
        require(m.local(item["path"]) == path, "operation named output differs: " + role)
        equal(m.sha(path), item["sha256"], "operation named output SHA differs: " + role)
        return m.read(path), item["sha256"]

    registration, reg_sha = named("registration_seal", reg / "registration.provenance.json")
    producer, child_sha = named("tracking_seal", run / "continuation.provenance.json")
    audit_record, audit_sha = named("audit_seal", audit_root / "audit.provenance.json")
    for sealed, expected in (
        (registration, m.REGISTRATION_STAGE),
        (producer, m.STAGE),
        (audit_record, auditor.STAGE),
    ):
        require(
            sealed["stage"] == expected
            and sealed["status"] == "ok"
            and sealed["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
            "lower stage is not successfully guarded",
        )
    audited, _ = named("audit/summary.json", audit_root / "summary.json")
    equal(audited, returned, "returned independent audit differs from its sealed summary")
    require(
        audited.get("independent_phase_audit_complete") is True,
        "separate phase audit is incomplete",
    )
    current.validate_audit(audited, run, stop, restores)
    equal(audited["parent"], str(parent), "independent audit parent differs")
    equal(audited["new_interval"], [start, stop], "independent audit interval differs")
    equal(audited["registration_sha256"], reg_sha, "independent audit registration differs")
    equal(producer["params"]["bindings"], binding, "child fixed binding changed")
    equal(producer["params"]["resume_phase"], str(parent), "child parent changed")
    equal(producer["params"]["through_frame"], stop, "child stop changed")
    seals = dict(
        operation_sha256=m.sha(root / "operation.provenance.json"),
        registration_sha256=reg_sha,
        child_sha256=child_sha,
        audit_sha256=audit_sha,
    )
    return chain.phase_row(root, parent, [start, stop], restores, seals)


def result_summary(parent, stop, restores, completed, *, outcome, requested, active_phase):
    complete = outcome == "complete" and stop == chain.FINAL_STOP
    return dict(
        schema=STAGE,
        status=outcome,
        final_run=str(parent),
        through=stop,
        full_sequence_complete=complete,
        prefix_backtrace_provisional=not complete,
        logical_initialization_count=1,
        checkpoint_restore_count=restores,
        external_pose_reset_count=0,
        active_beam_count=6,
        tracking_processes=restores + 1,
        consumed_chunk_count=(stop - chain.FIRST) // chain.native.CHUNK,
        phases=completed,
        stop_request=requested,
        incomplete_phase=active_phase,
        independent_ID_truth=False,
        final_audit_complete=False,
        detector_started=False,
        old_process_stopped=False,
        next_phase_retried=False,
    )


def supervise(args, *, sleeper=time.sleep):
    require(args.execute, "--execute is required; no implicit launch")
    output = m.local(args.output, exists=False)
    require(
        not output.exists() and not SHARED_LOCK.exists(), "new root and exclusive lock required"
    )
    evidence = m.Evidence()
    code = sources(evidence)
    reference = dict(path=str(m.local(args.exit_witness)), sha256=args.exit_witness_sha256)
    prefix = chain.load_completed_prefix(
        m.local(args.old_supervisor), args.old_supervisor_sha256, reference, evidence
    )
    transferred = handoff_data(prefix, args)
    chain.validate_handoff(transferred, prefix)
    parent = prefix["final_run"]
    stop, restores = transferred["initial_stop"], transferred["initial_restores"]
    intervals = remaining_phases(stop)
    legacy = prefix["binding"]
    new_code = {p: evidence.add(p, d) for p, d in legacy["code_sha256"].items()}
    new_code.update({str(m.REPO / p): evidence.add(m.REPO / p) for p in m.ADDED_IO})
    binding = m.execution_binding(legacy, new_code)
    parent_audit = prefix["last_audit_root"] / prefix["last_audit_name"]
    receipt_root = m.local(args.receipt_root, exists=False)
    evidence.final_guard()
    equal(sources(evidence), code, "source closure changed during bootstrap")
    lock = acquire_lock(output, parent)
    output.mkdir(parents=True, exist_ok=False)
    lock_copy = output / "supervisor.lock"
    shutil.copyfile(lock, lock_copy)
    evidence.add(lock)
    equal(m.sha(lock_copy), evidence.paths[str(lock)], "owner lock copy differs")
    completed, requested, active_phase, outcome = [], None, None, "failed"
    with (output / "events.jsonl").open("x", encoding="utf-8", buffering=1) as stream:

        def emit(event, **values):
            row = dict(event=event, time=datetime.now(UTC).isoformat(), **values)
            line = json.dumps(row, allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)

        with m.original.provenance(STAGE, output, name="supervisor") as record:
            try:
                record.params_from(args)
                child.snapshot(code, output, record)
                record.output("lock", lock_copy, sha256=m.sha(lock_copy))
                child.save_json(output / "handoff.json", transferred, record, "handoff")
                policy = dict(
                    schema=STAGE,
                    initial_parent=str(parent),
                    initial_stop=stop,
                    initial_restores=restores,
                    target_interval=[chain.FIRST, chain.FINAL_STOP],
                    phase_intervals=intervals,
                    fixed_bindings=binding,
                    source_hashes=code,
                    receipt_cache_root=str(receipt_root),
                    new_external_pose_seed=False,
                    failed_phase_retry=False,
                    scheduling_changes_scientific_protocol=False,
                    stop_request_name=current.STOP_NAME,
                    stop_request_only_at_completed_boundaries=True,
                    registration_tracking_audit_is_one_indivisible_operation=True,
                    forced_supervisor_termination_guarantees_child_cleanup=False,
                    detector_started=False,
                    old_process_stopped=False,
                )
                child.save_json(output / "supervisor_plan.json", policy, record, "plan")
                emit("handoff_bootstrap_complete", parent=str(parent), through=stop)
                for start, through in intervals:
                    require(start == stop, "remaining schedule has a gap")
                    requested = current.wait_for_ready(
                        m.local(binding["inputs"]["input_stage_root"]),
                        stop,
                        through,
                        emit,
                        lambda: current.capture_stop_request(output, record, emit),
                        sleeper=sleeper,
                    )
                    if requested is not None:
                        break
                    equal(sources(evidence), code, "source closure changed before operation")
                    root = output / "phases" / f"f{stop:06d}-{through:06d}"
                    root.parent.mkdir(parents=True, exist_ok=True)
                    require(not root.exists(), "new operation output required")
                    # Last stop check precedes the indivisible registration/child/audit operation.
                    requested = current.capture_stop_request(output, record, emit)
                    if requested is not None:
                        break
                    active_phase = str(root)
                    emit(
                        "operation_start",
                        parent=str(parent),
                        interval=[stop, through],
                        output=str(root),
                    )
                    sealed = None
                    try:
                        returned = operation.execute(
                            SimpleNamespace(
                                parent=parent,
                                parent_seal_sha256=m.sha(parent / "continuation.provenance.json"),
                                parent_audit_seal=parent_audit,
                                parent_audit_sha256=m.sha(parent_audit),
                                through_frame=through,
                                receipt_root=receipt_root,
                                output=root,
                            ),
                            emit=emit,
                        )
                    finally:
                        sealed = collect_operation(root, output, record)
                    row = completed_operation(
                        root, returned, parent, stop, through, restores + 1, binding, sealed
                    )
                    completed.append(row)
                    parent, stop, restores = root / "tracking", through, restores + 1
                    parent_audit = root / "audit/audit.provenance.json"
                    active_phase = None
                    emit("phase_complete", **row)
                outcome = (
                    "complete" if stop == chain.FINAL_STOP else "stopped_after_completed_phase"
                )
            finally:
                child.save_json(
                    output / "summary.json",
                    result_summary(
                        parent,
                        stop,
                        restores,
                        completed,
                        outcome=outcome,
                        requested=requested,
                        active_phase=active_phase,
                    ),
                    record,
                    "summary",
                )
                stream.flush()
                record.output(
                    "events", output / "events.jsonl", sha256=m.sha(output / "events.jsonl")
                )
                equal(sources(evidence), code, "late supervisor source/import change")
                child.close_evidence(record, evidence, None)
                # Stop requests are captured directly in record.inputs by the frozen helper.
                for item in record.inputs.values():
                    equal(m.sha(item["path"]), item["sha256"], "supervisor input changed")
            record.param("all_input_source_snapshot_output_sha_unchanged", True)
            record.param("full_sequence_complete", outcome == "complete")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-supervisor", type=Path, required=True)
    parser.add_argument("--old-supervisor-sha256", required=True)
    parser.add_argument("--exit-witness", type=Path, required=True)
    parser.add_argument("--exit-witness-sha256", required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    require(args.execute, "--execute is required; help/tests never launch tracking")
    require(
        Path(sys.executable).resolve() == (REPO / ".venv/Scripts/python.exe").resolve(),
        "use the fixed main .venv Python",
    )
    return supervise(args)


if __name__ == "__main__":
    main()

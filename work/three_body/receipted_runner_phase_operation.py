"""One explicit receipt-child operation: CPU registration, tracking child, CPU audit.

No supervisor loop, lock, takeover, stop request, detector or alternative tracker.
The caller owns exclusive execution resources. An external graceful stop arriving
inside this indivisible operation is handled after it returns. The frozen owned
process utility retains its Windows tree-cleanup and hard-kill limitations.
Only --execute launches processes; no live supervisor imports this new file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import receipted_long_phase_audit  # noqa: F401  # Eager independent-auditor closure.
import receipted_runner_phase_audit as auditor
import run_long_remaining_supervisor as owned

m, child = auditor.m, auditor.child
require, equal = m.require, m.equal
STAGE = "long_receipted_runner_phase_operation_v1"
PINNED = {
    "work/three_body/receipted_runner_phase_audit.py": (
        "bf16c2d7acb84d0c24f257ceefe3f6d85fa9c2735688e46f05d37126687b2047"
    ),
    "work/three_body/run_long_remaining_supervisor.py": (
        "4e4e19a35bb7b11a7c38f87f13c03e66cc987252e055ebfa3d2b163c84f214db"
    ),
}


def operational_sources(evidence):
    for relative, digest in PINNED.items():
        evidence.add(m.REPO / relative, digest)
    return auditor.audit_sources(evidence)


def verify_sources(code, evidence):
    equal(operational_sources(evidence), code, "late operation import/source closure changed")
    evidence.final_guard()


def registration_command(args, root):
    """Fresh interpreter is required by the child's exact 40-source closure."""
    return [
        str((m.REPO / ".venv/Scripts/python.exe").resolve()),
        "-B",
        str(Path(child.__file__).resolve()),
        "--register-only",
        "--parent",
        str(m.local(args.parent)),
        "--parent-seal-sha256",
        args.parent_seal_sha256,
        "--parent-audit-seal",
        str(m.local(args.parent_audit_seal)),
        "--parent-audit-sha256",
        args.parent_audit_sha256,
        "--through-frame",
        str(args.through_frame),
        "--receipt-root",
        str(m.local(args.receipt_root, exists=False)),
        "--run-output",
        str(root / "tracking"),
        "--output",
        str(root / "registration"),
    ]


def preflight(args, root, evidence):
    parent = m.local(args.parent)
    parent_path = parent / "continuation.provenance.json"
    evidence.add(parent_path, args.parent_seal_sha256)
    record = m.read(parent_path)
    require(
        record.get("status") == "ok"
        and record["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
        "successful guarded parent required",
    )
    start = record["params"]["through_frame"]
    m.native_phase(start, args.through_frame)
    parent_audit = m.local(args.parent_audit_seal)
    evidence.add(parent_audit, args.parent_audit_sha256)
    audit_record = m.read(parent_audit)
    require(
        audit_record.get("status") == "ok"
        and audit_record["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
        "successful guarded parent audit required",
    )
    legacy = m.legacy_binding(evidence)
    new_code = {p: evidence.add(p, d) for p, d in legacy["code_sha256"].items()}
    new_code.update({str(m.REPO / p): evidence.add(m.REPO / p) for p in m.ADDED_IO})
    binding = m.execution_binding(legacy, new_code)
    edge = m.transition(record["params"]["bindings"], binding, legacy)
    expected_stage = m.original.STAGE if edge["kind"] == "legacy_to_receipted" else m.STAGE
    require(record["stage"] == expected_stage, "parent stage/binding differs")
    require(
        not (root / "registration").exists()
        and not (root / "tracking").exists()
        and not (root / "audit").exists(),
        "all three child output directories must be new",
    )
    return dict(
        parent=parent,
        parent_record=record,
        parent_audit=parent_audit,
        legacy=legacy,
        binding=binding,
        edge=edge,
        start=start,
    )


def verify_registration(args, root, fixed, info, evidence):
    """Bind successful CPU registration without repeating its checkpoint/loader work."""
    require(
        type(info.get("returncode")) is int
        and info["returncode"] == 0
        and info.get("child_alive_after_return") is False,
        "CPU registration process did not complete successfully",
    )
    reg = root / "registration"
    digest = evidence.add(reg / "registration.provenance.json")
    plan, record = child.load_registration(reg, digest, evidence)
    equal(
        record["params"].get("tracking_GPU_or_GT_performed"),
        False,
        "CPU registration declared GPU/GT work",
    )
    require(
        auditor.timestamp(record["started_at"])
        <= auditor.timestamp(record["finished_at"])
        <= auditor.timestamp(info["finished_at"]),
        "registration process/producer chronology differs",
    )
    require(
        auditor.timestamp(info["started_at"]) <= auditor.timestamp(record["finished_at"]),
        "CPU registration timing differs",
    )
    for key, expected in dict(
        parent=str(fixed["parent"]),
        parent_seal_sha256=args.parent_seal_sha256,
        expected_output=str(root / "tracking"),
        next_native_interval=[fixed["start"], args.through_frame],
        receipt_cache_root=str(m.local(args.receipt_root)),
        binding=fixed["binding"],
    ).items():
        equal(plan.get(key), expected, "CPU registration changed requested contract: " + key)
    equal(plan["parent_audit"]["path"], str(fixed["parent_audit"]), "parent audit path differs")
    equal(plan["parent_audit"]["sha256"], args.parent_audit_sha256, "parent audit SHA differs")
    auditor.registered_edge(plan, fixed["legacy"], fixed["parent_record"]["params"]["bindings"])
    require(
        auditor.timestamp(fixed["parent_record"]["finished_at"])
        <= auditor.timestamp(record["started_at"]),
        "parent postdates registration",
    )
    return plan, record, digest


def capture_completed_stage(root, name, stage, evidence):
    path = root / name
    evidence.add(path)
    record = m.read(path)
    require(record.get("stage") == stage, "completed stage name differs")
    m.verified_outputs(root, record, evidence)
    for role, item in record["inputs"].items():
        require(str(m.local(item["path"])) == role, "completed stage input role differs")
        evidence.add(role, item["sha256"])
    return record


def execute(args, *, emit=None):
    """Explicit single operation. Scheduling and graceful-stop policy are external."""
    root, evidence = child.new_output(args.output), m.Evidence()
    registration_info, tracking_info = {}, {}
    registration_digest = None
    outcome, current_step, plan, result = "failed", "preflight", None, None
    events_path = root / "events.jsonl"
    with m.original.provenance(STAGE, root, name="operation") as record:
        record.param("execution_resource_owner_external", True)
        record.param("existing_owner_lock_detector_or_scientific_core_changed", False)
        with events_path.open("x", encoding="utf-8", newline="\n") as events:

            def event(name, **values):
                events.write(
                    json.dumps(
                        dict(event=name, at=datetime.now(UTC).isoformat(), **values),
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                events.flush()
                if emit is not None:
                    emit(name, **values)

            code = None
            try:
                code = operational_sources(evidence)
                child.snapshot(code, root, record)
                fixed = preflight(args, root, evidence)
                command = registration_command(args, root)
                child.save_json(
                    root / "registration_command.json",
                    dict(command=command, tracking_child=False, source_hashes=code),
                    record,
                    "registration_command",
                )
                verify_sources(code, evidence)
                current_step = "CPU registration"
                event(
                    "registration_start",
                    parent=str(fixed["parent"]),
                    interval=[fixed["start"], args.through_frame],
                )
                try:
                    code_exit = owned.run_child(
                        command,
                        root / "registration.log",
                        registration_info,
                        on_started=lambda info: event(
                            "owned_registration_started",
                            pid=info["pid"],
                            command=command,
                            output=str(root / "registration"),
                        ),
                    )
                    require(
                        type(code_exit) is int and code_exit == 0, "CPU registration child failed"
                    )
                finally:
                    child.save_json(
                        root / "registration_execution.json",
                        dict(
                            schema="receipted_long_registration_execution_v1",
                            command=command,
                            **registration_info,
                        ),
                        record,
                        "registration_execution",
                    )
                plan, registration_record, registration_digest = verify_registration(
                    args, root, fixed, registration_info, evidence
                )
                event("registration_complete", registration_sha256=registration_digest)
                tracking_command = auditor.expected_command(
                    root / "tracking", root / "registration", registration_digest
                )
                child.save_json(
                    root / "execution.json",
                    dict(
                        command=tracking_command,
                        registration_sha256=registration_digest,
                        parent=str(fixed["parent"]),
                        output=str(root / "tracking"),
                        source_hashes=code,
                        execution_resource_owner_external=True,
                    ),
                    record,
                    "execution",
                )
                verify_sources(code, evidence)
                require(not (root / "tracking").exists(), "tracking output appeared before launch")
                current_step = "tracking child"
                event(
                    "phase_start",
                    parent=str(fixed["parent"]),
                    output=str(root / "tracking"),
                    interval=plan["next_native_interval"],
                )

                def started(info):
                    require(
                        auditor.timestamp(registration_record["finished_at"])
                        <= auditor.timestamp(info["started_at"]),
                        "registration postdates child start",
                    )
                    event(
                        "owned_child_started",
                        pid=info["pid"],
                        command=tracking_command,
                        output=str(root / "tracking"),
                        log=str(root / "tracking.log"),
                        force_kill_of_supervisor_does_not_guarantee_child_cleanup=True,
                    )

                try:
                    code_exit = owned.run_child(
                        tracking_command, root / "tracking.log", tracking_info, on_started=started
                    )
                    require(type(code_exit) is int and code_exit == 0, "tracking child failed")
                    require(
                        tracking_info.get("child_alive_after_return") is False,
                        "owned tracking child remains alive",
                    )
                finally:
                    child.save_json(
                        root / "child_execution.json",
                        dict(
                            schema=auditor.EXECUTION_SCHEMA,
                            command=tracking_command,
                            output=str(root / "tracking"),
                            registration_sha256=registration_digest,
                            **tracking_info,
                        ),
                        record,
                        "child_execution",
                    )
                owned_path = root / "child_execution.json"
                evidence.add(owned_path, record.outputs["child_execution"]["sha256"])
                producer = capture_completed_stage(
                    root / "tracking", "continuation.provenance.json", m.STAGE, evidence
                )
                auditor.verify_owned_execution(
                    m.read(owned_path),
                    root / "tracking",
                    root / "registration",
                    registration_digest,
                    registration_record,
                    producer,
                )
                current_step = "independent CPU audit"
                verify_sources(code, evidence)
                event("independent_audit_start", output=str(root / "audit"))
                result = auditor.audit(
                    SimpleNamespace(
                        run=root / "tracking",
                        registration=root / "registration",
                        registration_sha256=registration_digest,
                        owned_execution=owned_path,
                        receipt_root=Path(args.receipt_root),
                        output=root / "audit",
                    )
                )
                audit_record = capture_completed_stage(
                    root / "audit", "audit.provenance.json", auditor.STAGE, evidence
                )
                require(
                    result.get("status") == "ok"
                    and result.get("independent_phase_audit_complete") is True,
                    "independent phase audit did not complete",
                )
                summary_path = root / "audit/summary.json"
                require(
                    m.local(audit_record["outputs"]["summary"]["path"]) == summary_path,
                    "independent audit summary role differs",
                )
                equal(m.read(summary_path), result, "returned audit differs from sealed result")
                equal(result["run"], str(root / "tracking"), "audit belongs to another child")
                equal(
                    result["registration_sha256"], registration_digest, "audit registration differs"
                )
                equal(
                    result["candidate_sha256"],
                    producer["outputs"]["candidate"]["sha256"],
                    "auditor candidate differs",
                )
                verify_sources(code, evidence)
                outcome, current_step = "ok", "checks complete; final guard pending"
                event(
                    "phase_checks_complete",
                    output=str(root / "tracking"),
                    interval=plan["next_native_interval"],
                )
            except BaseException:
                outcome = "failed"
                raise
            finally:
                events.flush()
                summary = dict(
                    schema=STAGE,
                    status=outcome,
                    last_step=current_step,
                    registration_sha256=registration_digest,
                    completed_run=str(root / "tracking") if outcome == "ok" else None,
                    independent_audit=str(root / "audit") if outcome == "ok" else None,
                    child_started=bool(tracking_info.get("pid")),
                    parent_promoted=False,
                    operation_success_requires_successful_seal=True,
                    automatic_retry=False,
                    execution_resource_owner_external=True,
                    independent_ID_truth=False,
                )
                child.save_json(root / "summary.json", summary, record, "summary")
                for name in ("registration.log", "tracking.log", "events.jsonl"):
                    p = root / name
                    if p.is_file():
                        record.output(name, p, sha256=m.sha(p))
                for subdir, sealname in (
                    ("registration", "registration.provenance.json"),
                    ("tracking", "continuation.provenance.json"),
                    ("audit", "audit.provenance.json"),
                ):
                    p = root / subdir / sealname
                    if p.is_file():
                        record.output(subdir + "_seal", p, sha256=m.sha(p))
                for name in ("audit/summary.json", "audit/mask_validation_receipts.json"):
                    p = root / name
                    if p.is_file():
                        record.output(name, p, sha256=m.sha(p))
                if code is not None:
                    equal(
                        operational_sources(evidence),
                        code,
                        "late operation import/source closure changed",
                    )
                child.close_evidence(record, evidence, None)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
        record.param("tracking_child_exit_code", tracking_info["returncode"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-seal-sha256", required=True)
    parser.add_argument("--parent-audit-seal", type=Path, required=True)
    parser.add_argument("--parent-audit-sha256", required=True)
    parser.add_argument("--through-frame", type=int, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    require(args.execute, "--execute is required; help/tests never launch tracking")
    require(
        Path(sys.executable).resolve() == (m.REPO / ".venv/Scripts/python.exe").resolve(),
        "use the fixed main .venv Python",
    )
    return execute(args)


if __name__ == "__main__":
    main()

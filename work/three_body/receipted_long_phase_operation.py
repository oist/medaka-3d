"""One explicitly invoked receipted phase; no supervisor loop, lock or handoff.

The frozen v1 tracking command, bindings, registration plan and audit result are
unchanged. Receipts are separate operational evidence under new stage names.
Only execute_phase starts the one child explicitly supplied by its caller; the
caller must own scheduling/locking and any required process-tree permissions.
"""

from __future__ import annotations

import gc
from datetime import UTC, datetime

import receipted_long_phase_audit as backend
from mask_validation_receipts import ReceiptStore
from run_long_remaining_supervisor import (
    FIRST,
    command_for,
    equal,
    final_outputs,
    header,
    provenance,
    read,
    require,
    run_child,
    save_json,
    sha256,
    snapshot,
    sources,
    unchanged,
    within,
)

REGISTRATION_STAGE = "register_long_receipted_phase_v1"
OPERATION_STAGE = "long_receipted_phase_operation_v1"
PARENT_AUDIT_STAGES = {
    "long_supervised_bootstrap_audit": ("audit.provenance.json", "summary.json", "summary"),
    "long_receipted_handoff_bootstrap_v1": (
        "audit.provenance.json", "summary.json", "summary"),
    "long_supervised_phase_operation": ("operation.provenance.json", "audit.json", "audit"),
    OPERATION_STAGE: ("operation.provenance.json", "audit.json", "audit"),
}


def bind_parent_audit(parent, audit_root, evidence):
    """Accept actual named stages, never fabricate an old success sidecar."""
    parent, audit_root = within(parent), within(audit_root)
    names = ("audit.provenance.json", "operation.provenance.json")
    candidates = [audit_root / name for name in names
                  if (audit_root / name).is_file()]
    require(len(candidates) == 1, "one explicit completed parent audit seal required")
    seal_path = candidates[0]
    evidence.add(seal_path)
    seal = read(seal_path)
    stage = seal.get("stage")
    require(stage in PARENT_AUDIT_STAGES, "unrecognized parent audit stage")
    filename, summary_name, role = PARENT_AUDIT_STAGES[stage]
    require(seal_path.name == filename and seal.get("status") == "ok"
            and seal["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
            "successful guarded parent audit required")
    for item in seal["outputs"].values():
        path = within(item["path"])
        require(path.is_relative_to(audit_root) and path.stat().st_size == item["size"],
                "parent audit output escaped scope or changed size")
        evidence.add(path, item["sha256"])
    summary_path = audit_root / summary_name
    require(within(seal["outputs"][role]["path"]) == summary_path,
            "parent audit summary role differs")
    audit = read(summary_path)
    parent_seal_path = parent / "continuation.provenance.json"
    evidence.add(parent_seal_path)
    parent_seal = read(parent_seal_path)
    require(parent_seal.get("stage") == "three_body_long_continuation"
            and parent_seal.get("status") == "ok"
            and parent_seal["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
            "parent continuation did not complete")
    candidate_path = parent / "candidate.npz"
    candidate = parent_seal["outputs"]["candidate"]
    require(within(candidate["path"]) == candidate_path, "parent candidate path differs")
    evidence.add(candidate_path, candidate["sha256"])
    require(audit.get("status") == "ok" and audit.get("run") == str(parent)
            and audit.get("candidate_sha256") == candidate["sha256"],
            "independent audit belongs to another parent/candidate")
    for path in (parent_seal_path, candidate_path):
        item = seal["inputs"].get(str(path))
        require(item is not None and within(item["path"]) == path
                and item["sha256"] == evidence.paths[str(path)],
                "parent audit lacks the exact parent input binding")
    return dict(root=str(audit_root), stage=stage,
                seal_sha256=evidence.paths[str(seal_path)],
                summary_sha256=evidence.paths[str(summary_path)],
                parent_candidate_sha256=audit["candidate_sha256"])


def _save_receipts(store, root, record, phase, returned):
    """Keep cache bookkeeping outside the scientific plan and audit result."""
    for path, digest in store.evidence_hashes.items():
        record.input(path, path, sha256=digest)
    save_json(root / "mask_validation_receipts.json", dict(
        schema="long_phase_mask_validation_usage_v1", phase=phase,
        verification_returned=returned, cache_root=str(store.root),
        uses=store.uses, evidence_hashes=store.evidence_hashes,
        scientific_plan_or_child_binding_changed=False,
    ), record, "mask_validation_receipts")


def register_phase(parent, stop, phase_root, code, binding, parent_audit, *, receipt_root):
    parent, phase_root = within(parent), within(phase_root)
    receipt_root = within(receipt_root, exists=False)
    child_root = phase_root / "tracking"
    evidence = backend.Digests()
    for path, digest in code.items():
        evidence.add(path, digest)
    lineage = bind_parent_audit(parent, parent_audit, evidence)
    store = ReceiptStore(receipt_root)
    plan, context = backend.prepare_phase(parent, stop, child_root, evidence, receipt_store=store)
    del context
    gc.collect()
    require(equal(sources(), code), "late workspace imports or modified source")
    require(equal(plan["frozen_input_and_code_binding"], binding), "fixed binding changed")
    command = command_for(parent, stop, child_root, binding)
    require(plan["parent_phase"] == str(parent) and plan["expected_output"] == str(child_root)
            and plan["output_interval"] == [FIRST, stop], "phase plan command differs")
    registration = phase_root / "registration"
    registration.mkdir(parents=True, exist_ok=False)
    with provenance(REGISTRATION_STAGE, registration, name="registration") as record:
        for path, digest in evidence.paths.items():
            record.input(path, path, sha256=digest)
        snapshot(code, registration, record)
        save_json(registration / "plan.json", plan, record, "plan")
        save_json(registration / "execution.json", dict(
            command=command, source_hashes=code, parent=str(parent), output=str(child_root),
            through_frame=stop, independent_parent_audit=lineage,
            receipt_cache_root=str(receipt_root),
        ), record, "execution")
        _save_receipts(store, registration, record, "prepare", True)
        store.final_guard()
        evidence.final_guard()
        final_outputs(record)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
        record.param("tracking_detector_GPU_or_GT_performed", False)
    return plan, child_root, registration, command


def _verify_registration(registration, plan, child_root, command, code, receipt_root, evidence):
    seal_path = registration / "registration.provenance.json"
    evidence.add(seal_path)
    seal = read(seal_path)
    require(seal.get("stage") == REGISTRATION_STAGE and seal.get("status") == "ok"
            and seal["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
            "successful new-stage phase registration required before launch")
    for role, item in seal["inputs"].items():
        path = within(item["path"])
        require(str(path) == role, "registration input role/path differs")
        evidence.add(path, item["sha256"])
    for item in seal["outputs"].values():
        path = within(item["path"])
        require(path.is_relative_to(registration) and path.stat().st_size == item["size"],
                "registration output escaped scope or changed size")
        evidence.add(path, item["sha256"])
    for role, name in (("plan", "plan.json"), ("execution", "execution.json"),
                       ("mask_validation_receipts", "mask_validation_receipts.json")):
        require(within(seal["outputs"][role]["path"]) == registration / name,
                "registration named output differs")
    require(equal(read(registration / "plan.json"), plan), "phase plan changed before launch")
    execution = read(registration / "execution.json")
    require(execution["command"] == command and execution["source_hashes"] == code
            and execution["parent"] == plan["parent_phase"]
            and execution["output"] == str(child_root)
            and execution["through_frame"] == plan["next_native_interval"][1]
            and execution["receipt_cache_root"] == str(receipt_root),
            "registered command/source/cache changed before launch")
    parent = within(plan["parent_phase"])
    binding, _ = header(parent, plan["next_native_interval"][0])
    require(equal(binding, plan["frozen_input_and_code_binding"]), "parent binding changed")
    require(command == command_for(parent, plan["next_native_interval"][1], child_root, binding),
            "command differs from frozen v1 runner CLI")
    require(datetime.fromisoformat(seal["finished_at"]) <= datetime.now(UTC),
            "registration completion is in the future")
    return seal


def execute_phase(phase_root, plan, child_root, registration, command, code, emit, *, receipt_root):
    phase_root, registration = within(phase_root), within(registration)
    child_root, receipt_root = within(child_root, exists=False), within(receipt_root, exists=False)
    require(registration == phase_root / "registration" and child_root == phase_root / "tracking"
            and not child_root.exists(), "new child output required under this registered phase")
    require(not any((phase_root / name).exists() for name in (
        "operation.provenance.json", "tracking.log", "child_execution.json", "audit.json",
        "mask_validation_receipts.json")), "operation already attempted; no automatic retry")
    log_path, info, store, returned = phase_root / "tracking.log", {}, None, False
    evidence = backend.Digests()
    with provenance(OPERATION_STAGE, phase_root, name="operation") as record:
        for path, digest in code.items():
            evidence.add(path, digest)
        seal = _verify_registration(
            registration, plan, child_root, command, code, receipt_root, evidence)
        for path, digest in evidence.paths.items():
            record.input(path, path, sha256=digest)
        require(equal(sources(), code), "late workspace imports or modified source")
        evidence.final_guard()
        emit("phase_start", parent=plan["parent_phase"], output=str(child_root),
             interval=plan["next_native_interval"])

        def started(state):
            require(datetime.fromisoformat(seal["finished_at"])
                    <= datetime.fromisoformat(state["started_at"]),
                    "registration postdates the owned process start")
            emit("owned_child_started", pid=state["pid"], command=command,
                 log=str(log_path), output=str(child_root),
                 force_kill_of_supervisor_does_not_guarantee_child_cleanup=True)

        try:
            native_exit = run_child(command, log_path, info, on_started=started)
            require(native_exit == 0, "tracking child failed; phase will not be retried")
            require(info["child_alive_after_return"] is False, "owned child is still alive")
            child_seal = read(child_root / "continuation.provenance.json")
            require(datetime.fromisoformat(seal["finished_at"])
                    <= datetime.fromisoformat(child_seal["started_at"]),
                    "phase was registered after the child started")
            child_binding, _ = header(child_root, plan["next_native_interval"][1])
            require(equal(child_binding, plan["frozen_input_and_code_binding"]),
                    "child fixed binding changed")
            store = ReceiptStore(receipt_root)
            audited = backend.verify_phase(child_root, plan, evidence, receipt_store=store)
            require(audited.get("status") == "ok", "phase audit did not succeed")
            returned = True
            save_json(phase_root / "audit.json", audited, record, "audit")
            for path, digest in evidence.paths.items():
                record.input(path, path, sha256=digest)
        finally:
            save_json(phase_root / "child_execution.json", info, record, "child_execution")
            if log_path.is_file():
                record.output("tracking_log", log_path, sha256=sha256(log_path))
            if store is not None:
                _save_receipts(store, phase_root, record, "verify", returned)
                store.final_guard()
            evidence.final_guard()
            unchanged(code)
            unchanged({item["path"]: item["sha256"] for item in record.inputs.values()})
            final_outputs(record)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
        record.param("tracking_child_exit_code", native_exit)
        emit("phase_complete", output=str(child_root), interval=plan["next_native_interval"])
    return audited

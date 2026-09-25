"""Verify one completed legacy child and issue a new, explicit bootstrap audit.

No process query, signal, launch, lock removal, GPU, detector or raw decoding.
The caller owns scheduling and ownership transfer. A sealed exit witness is
required; neither an unfinished witness nor an unfinished child is opened.
"""

from __future__ import annotations

import gc
from pathlib import Path

import capture_owned_long_exit as witness
import receipted_long_phase_audit as backend
import receipted_long_phase_operation as operation
import run_long_remaining_supervisor as old
from mask_validation_receipts import ReceiptStore

REPO, WORK = old.REPO, old.WORK
OLD_ROOT = WORK / "long_remaining_supervisor_v1"
OLD_PARENT = WORK / "long_continuation_through66400_v1"
CHILD = OLD_ROOT / "phases/f066400-076360/tracking"
WITNESS_SOURCE = WORK / "capture_owned_long_exit.py"
WITNESS_SHA = "6a5fbed6209a5baeb56166af3c66e1eeb988027b0128df46422e0a7a97278a3e"
POLICY_SHA = "4d380919ee7c233746428bddcd0e19683ea3d7c9f4966391266c6c35d6c87696"
BOOTSTRAP_SHA = "d9d832f5842af3b5a0304cfb1e3e22bd7a619e03e621858aa3a3701819a3ff6d"
STAGE = "long_receipted_handoff_bootstrap_v1"
require, equal, within = old.require, old.equal, old.within


def read_bound(path, evidence, expected=None):
    path = within(path)
    evidence.add(path, expected)
    return witness.decode(path.read_bytes())


def record_files(record, root, evidence):
    """Recheck every standard provenance input/output, with exact output scope."""
    for kind in ("inputs", "outputs"):
        require(isinstance(record.get(kind), dict), "missing standard file records")
        for item in record[kind].values():
            path = within(item["path"])
            require(
                path.is_file() and path.stat().st_size == item["size"], "sealed file size differs"
            )
            if kind == "outputs":
                require(path.is_relative_to(root), "sealed output escapes its producer")
            evidence.add(path, item["sha256"])


def standard_seal(path, stage, evidence):
    record = read_bound(path, evidence)
    require(
        record.get("stage") == stage
        and record.get("status") == "ok"
        and record["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
        "successful guarded legacy record required",
    )
    record_files(record, Path(path).parent, evidence)
    return record


def output_path(record, root, role, filename):
    path = within(record["outputs"][role]["path"])
    require(path == root / filename, "legacy output role/path differs")
    return path


def _flat_hashes(values, evidence, *, root=None):
    require(isinstance(values, dict), "witness hashes must be a dictionary")
    for name, digest in values.items():
        path = within(name)
        require(
            str(path) == name and isinstance(digest, str) and len(digest) == 64,
            "witness path/hash schema differs",
        )
        require(root is None or path.is_relative_to(root), "witness output escaped scope")
        evidence.add(path, digest)


def _verify_witness(root, evidence):
    """Verify held-handle receipts as data; never instantiate WindowsProcess."""
    root = within(root)
    seal_path = root / "witness.provenance.json"
    seal = read_bound(seal_path, evidence)
    require(
        seal.get("stage") == "capture_owned_long_process_exits"
        and seal.get("schema") == witness.SCHEMA
        and seal.get("status") == "ok"
        and seal.get("error") is None,
        "completed successful exit witness required before payload reads",
    )
    require(
        seal.get("tracking_or_supervisor_success_asserted") is False
        and seal.get("signals_enumeration_GPU_or_tracking_performed") is False,
        "wrong witness semantics",
    )
    require(
        seal["inputs"].get(str(WITNESS_SOURCE)) == WITNESS_SHA,
        "unregistered witness implementation",
    )
    _flat_hashes(seal["inputs"], evidence)
    _flat_hashes(seal["outputs"], evidence, root=root)
    evidence.add(WITNESS_SOURCE, WITNESS_SHA)
    names = {
        "binding.json",
        "events_baseline.jsonl",
        "attachment.json",
        "child_exit.json",
        "child_exit.receipt.json",
        "events_at_child_exit.jsonl",
        "owner_exit.json",
        "owner_exit.receipt.json",
        "events_at_owner_exit.jsonl",
        "source_snapshot/00_capture_owned_long_exit.py",
        "source_snapshot/01_run_long_remaining_supervisor.py",
        "source_snapshot/02_pyvenv.cfg",
    }
    require(
        set(seal["outputs"]) == {str(root / n) for n in names}, "witness output membership differs"
    )
    source_pairs = [
        (WITNESS_SOURCE, "00_capture_owned_long_exit.py"),
        (old.WORK / "run_long_remaining_supervisor.py", "01_run_long_remaining_supervisor.py"),
        (REPO / ".venv/pyvenv.cfg", "02_pyvenv.cfg"),
    ]
    for path, name in source_pairs:
        require(
            seal["outputs"][str(root / "source_snapshot" / name)] == seal["inputs"][str(path)],
            "witness source snapshot differs",
        )
    binding = read_bound(root / "binding.json", evidence)
    require(
        binding["schema"] == witness.SCHEMA
        and within(binding["supervisor_root"]) == OLD_ROOT
        and within(binding["child_event"]["output"]) == CHILD
        and binding["phase_start"]["interval"] == [66400, 76360],
        "only the first legacy supervised child can transfer",
    )
    expected_inputs = {
        **binding["immutable_hashes"],
        **{str(p): seal["inputs"][str(p)] for p, _ in source_pairs},
    }
    require(equal(seal["inputs"], expected_inputs), "unexpected witness input membership")
    for path, digest in binding["immutable_hashes"].items():
        require(seal["inputs"].get(path) == digest, "witness immutable input differs")
    # This legacy helper only reads evidence; it performs no process operation.
    current, final_events = witness.build_binding(OLD_ROOT, binding["child_pid"])
    for key in set(binding) - {"baseline_events_sha256", "baseline_events_bytes"}:
        require(equal(binding[key], current[key]), "current ownership binding changed: " + key)
    baseline = (root / "events_baseline.jsonl").read_bytes()
    require(
        witness.digest(baseline) == binding["baseline_events_sha256"]
        and len(baseline) == binding["baseline_events_bytes"]
        and final_events.startswith(baseline),
        "old event prefix changed",
    )
    event_path = within(binding["events_path"])
    evidence.add(event_path, witness.digest(final_events))
    require(final_events.endswith(b"\n"), "incomplete final event line cannot rule out a launch")
    rows = witness.event_rows(final_events)
    index = binding["child_event_index"]
    require(
        type(index) is int
        and 0 < index < len(rows)
        and equal(rows[index], binding["child_event"])
        and equal(rows[index - 1], binding["phase_start"])
        and not any(
            r.get("event") in ("phase_start", "owned_child_started") for r in rows[index + 1 :]
        ),
        "later or inconsistent legacy launch",
    )
    attachment = read_bound(root / "attachment.json", evidence)
    require(
        attachment["schema"] == witness.SCHEMA
        and set(attachment["processes"]) == {"owner", "child"},
        "wrong process attachment",
    )
    require(set(seal["observations"]) == {"owner", "child"}, "both exits required")
    observations = {}
    for role in ("child", "owner"):
        receipt = read_bound(root / f"{role}_exit.receipt.json", evidence)
        payload = read_bound(root / f"{role}_exit.json", evidence)
        require(
            receipt["schema"] == payload["schema"] == witness.SCHEMA
            and receipt["role"] == payload["role"] == role
            and receipt["status"] == "ok"
            and receipt["all_evidence_sha_unchanged"] is True
            and receipt["tracking_success_asserted"] is False
            and receipt["process_signal_or_termination_performed"] is False,
            "role receipt is not a completed exit observation",
        )
        require(equal(receipt["inputs"], seal["inputs"]), "role receipt inputs differ")
        for path, digest in receipt["outputs"].items():
            require(seal["outputs"].get(path) == digest, "role receipt output differs")
        for name in (
            "binding.json",
            "attachment.json",
            f"{role}_exit.json",
            f"events_at_{role}_exit.jsonl",
        ):
            require(
                receipt["outputs"].get(str(root / name)) == seal["outputs"][str(root / name)],
                "role receipt omits required evidence",
            )
        require(
            payload["binding_sha256"] == seal["outputs"][str(root / "binding.json")],
            "exit receipt belongs to another binding",
        )
        obs, identity = payload["observation"], attachment["processes"][role]
        require(
            equal(obs, seal["observations"][role])
            and all(equal(obs[k], v) for k, v in identity.items()),
            "held process identity changed at exit",
        )
        require(
            type(obs["pid"]) is int
            and obs["pid"] == binding[role + "_pid"]
            and obs["live_when_handle_acquired"] is True
            and obs["wait_signalled"] is True
            and obs["access_mask"] == 0x00101000
            and type(obs["exit_code"]) is int
            and 0 <= obs["exit_code"] <= 0xFFFFFFFF,
            "role lacks a signalled held process handle",
        )
        for prefix in ("creation", "exit"):
            ticks = obs[prefix + "_filetime_100ns"]
            require(
                type(ticks) is int
                and ticks > 0
                and witness.filetime_utc(ticks) == obs[prefix + "_time_utc"],
                "FILETIME and stated time differ",
            )
        require(
            obs["creation_filetime_100ns"] < obs["exit_filetime_100ns"]
            and obs["exit_code_success"] is (obs["exit_code"] == 0)
            and witness.utc(obs["creation_time_utc"])
            <= witness.utc(obs["attached_at"])
            <= witness.utc(obs["exit_time_utc"])
            <= witness.utc(obs["exit_observed_at"])
            <= witness.utc(receipt["finished_at"])
            <= witness.utc(seal["finished_at"]),
            "exit chronology/flags differ",
        )
        capture = payload["event_capture"]
        captured = root / f"events_at_{role}_exit.jsonl"
        data = captured.read_bytes()
        require(
            within(capture["path"]) == captured
            and witness.digest(data) == capture["sha256"] == seal["outputs"][str(captured)]
            and len(data) == capture["bytes"]
            and data.startswith(baseline)
            and final_events.startswith(data)
            and capture["complete_lines"] == len(witness.event_rows(data))
            and capture["subsequent_launch_events"] == [],
            "exit event capture differs",
        )
        if role == "owner":
            require(
                data == final_events and capture["partial_trailing_line"] is False,
                "current final events differ from owner exit snapshot",
            )
        observations[role] = obs
    owner, child = observations["owner"], observations["child"]
    require(child["exit_code"] == 0, "tracking child exit must be zero")
    require(
        child["exit_filetime_100ns"] <= owner["exit_filetime_100ns"],
        "owner exited before the tracked child",
    )
    require(
        witness.utc(owner["creation_time_utc"])
        <= witness.utc(binding["owner_lock"]["created_at"])
        <= witness.utc(child["creation_time_utc"])
        and witness.utc(binding["phase_start"]["time"])
        <= witness.utc(child["creation_time_utc"])
        <= witness.utc(binding["child_event"]["time"]),
        "process ID reuse chronology differs",
    )
    runtime = attachment["runtime"]
    require(
        within(child["image_path"]) == old.PYTHON.resolve()
        and Path(owner["image_path"]).resolve()
        in {old.PYTHON.resolve(), Path(runtime["base_executable"]).resolve()},
        "witness executable identities differ",
    )
    return dict(
        root=root,
        seal=seal,
        binding=binding,
        observations=observations,
        final_events_sha256=witness.digest(final_events),
    )


def verify_handoff(witness_root, evidence):
    """Return a verified legacy edge, without loading candidate/checkpoint arrays."""
    proof = _verify_witness(witness_root, evidence)
    evidence.add(OLD_ROOT / "supervisor_plan.json", POLICY_SHA)
    policy = read_bound(OLD_ROOT / "supervisor_plan.json", evidence)
    require(
        policy["schema"] == "long_remaining_operational_supervisor_v1"
        and within(policy["initial_parent"]) == OLD_PARENT
        and policy["initial_stop"] == 66400
        and policy["target_interval"] == [49800, 348600]
        and policy["phase_intervals"] == old.phase_ranges()
        and all(
            policy[k] is False
            for k in (
                "detector_started",
                "new_external_pose_seed",
                "failed_phase_retry",
                "lock_removed",
                "scheduling_changes_scientific_protocol",
            )
        ),
        "different old supervisor policy",
    )
    bootstrap = OLD_ROOT / "bootstrap"
    evidence.add(bootstrap / "audit.provenance.json", BOOTSTRAP_SHA)
    prior_audit = operation.bind_parent_audit(OLD_PARENT, bootstrap, evidence)
    bootstrap_seal = standard_seal(
        bootstrap / "audit.provenance.json", "long_supervised_bootstrap_audit", evidence
    )
    reg = CHILD.parent / "registration"
    registration = standard_seal(
        reg / "registration.provenance.json", "register_long_supervised_phase", evidence
    )
    plan = read_bound(output_path(registration, reg, "plan", "plan.json"), evidence)
    execution = read_bound(output_path(registration, reg, "execution", "execution.json"), evidence)
    require(
        plan["schema"] == "long_continuation_registered_phase_v1"
        and plan["next_native_interval"] == [66400, 76360]
        and plan["output_interval"] == [49800, 76360]
        and within(plan["parent_phase"]) == OLD_PARENT
        and within(plan["expected_output"]) == CHILD,
        "different old registered phase",
    )
    lineage = execution["independent_parent_audit"]
    require(
        all(
            equal(lineage[k], prior_audit[k])
            for k in ("root", "seal_sha256", "summary_sha256", "parent_candidate_sha256")
        ),
        "old registration was not anchored to the successful 66400 bootstrap",
    )
    require(
        equal(execution["source_hashes"], policy["source_hashes"]),
        "old registered source set differs from supervisor policy",
    )
    for source, digest in execution["source_hashes"].items():
        source_path = within(source)
        evidence.add(source_path, digest)
        for root, record in ((reg, registration), (bootstrap, bootstrap_seal)):
            snapshot = root / "code_snapshot" / source_path.relative_to(REPO)
            role = snapshot.relative_to(root).as_posix()
            require(
                within(record["outputs"][role]["path"]) == snapshot
                and record["outputs"][role]["sha256"] == digest,
                "old registration/bootstrap code snapshot differs",
            )
        evidence.add(OLD_ROOT / "code_snapshot" / source_path.relative_to(REPO), digest)
    child_seal = standard_seal(
        CHILD / "continuation.provenance.json", "three_body_long_continuation", evidence
    )
    binding, _ = old.header(CHILD, 76360)
    parent_binding, _ = old.header(OLD_PARENT, 66400)
    require(
        equal(binding, parent_binding)
        and equal(binding, policy["fixed_bindings"])
        and equal(binding, plan["frozen_input_and_code_binding"]),
        "parent/child/policy/registration scientific binding differs",
    )
    command = old.command_for(OLD_PARENT, 76360, CHILD, binding)
    require(
        execution["command"] == proof["binding"]["child_event"]["command"] == command,
        "executed legacy command differs from fixed CLI",
    )
    require(
        witness.utc(bootstrap_seal["finished_at"])
        <= witness.utc(registration["started_at"])
        <= witness.utc(registration["finished_at"])
        <= witness.utc(child_seal["started_at"])
        <= witness.utc(child_seal["finished_at"])
        <= witness.utc(proof["observations"]["child"]["exit_time_utc"]),
        "successful bootstrap/registration/child/exit chronology differs",
    )
    candidate = output_path(child_seal, CHILD, "candidate", "candidate.npz")
    evidence.add(candidate, child_seal["outputs"]["candidate"]["sha256"])
    # Retain any old partial/failed operational sidecars as immutable evidence;
    # they are neither required to report success nor rewritten by this module.
    preserved = {}
    for path in (
        OLD_ROOT / "supervisor.provenance.json",
        CHILD.parent / "operation.provenance.json",
        CHILD.parent / "audit.json",
        CHILD.parent / "child_execution.json",
    ):
        if path.is_file():
            evidence.add(path)
            preserved[str(path)] = evidence.paths[str(path)]
    result = dict(
        parent=CHILD,
        old_root=OLD_ROOT,
        old_plan=plan,
        binding=binding,
        witness_root=proof["root"],
        lineage=dict(
            schema="long_receipted_first_child_handoff_v1",
            witness_root=str(proof["root"]),
            witness_seal_sha256=evidence.paths[str(proof["root"] / "witness.provenance.json")],
            witness_source_sha256=WITNESS_SHA,
            old_supervisor_root=str(OLD_ROOT),
            old_policy_sha256=POLICY_SHA,
            old_bootstrap=prior_audit,
            old_registration=str(reg),
            old_registration_seal_sha256=evidence.paths[str(reg / "registration.provenance.json")],
            old_plan_sha256=evidence.paths[str(reg / "plan.json")],
            transferred_run=str(CHILD),
            transferred_interval=[49800, 76360],
            owner_exit=proof["observations"]["owner"],
            child_exit=proof["observations"]["child"],
            final_events_sha256=proof["final_events_sha256"],
            old_optional_artifacts_preserved=preserved,
            old_lock_removed=False,
            old_success_claim_created=False,
            new_process_started=False,
            new_tracker_state_initialized=False,
            descendant_exit_independently_enumerated=False,
        ),
    )
    evidence.final_guard()
    return result


def audit_handoff(witness_root, output_root, code, *, receipt_root):
    """Create exactly the requested new bootstrap root; no loop or process launch."""
    output = within(output_root, exists=False)
    require(not output.exists(), "new bootstrap output required")
    output.mkdir(parents=True)
    evidence = backend.Digests()
    with old.provenance(STAGE, output, name="audit") as record:
        for source, digest in code.items():
            evidence.add(source, digest)
        require(equal(old.sources(), code), "actual handoff source set differs")
        old.snapshot(code, output, record)
        handoff = verify_handoff(witness_root, evidence)
        store = ReceiptStore(receipt_root)
        try:
            result = backend.verify_phase(
                handoff["parent"], handoff["old_plan"], evidence, receipt_store=store
            )
            require(
                result["status"] == "ok"
                and result["run"] == str(CHILD)
                and result["frame_range"] == [49800, 76360],
                "wrong handoff audit result",
            )
            result = {**result, "handoff_lineage": handoff["lineage"]}
            old.save_json(output / "summary.json", result, record, "summary")
            old.save_json(output / "handoff.json", handoff["lineage"], record, "handoff")
            for path, digest in evidence.paths.items():
                record.input(path, path, sha256=digest)
            operation._save_receipts(store, output, record, "handoff_verify", True)
            store.final_guard()
            require(equal(old.sources(), code), "late imports or modified handoff source")
            evidence.final_guard()
            old.final_outputs(record)
            record.param("all_input_source_snapshot_output_sha_unchanged", True)
            record.param("tracking_detector_GPU_or_GT_performed", False)
            record.param("old_lock_or_operational_outputs_modified", False)
        finally:
            # Failed semantic validation never yields a successful bootstrap.
            if not (output / "mask_validation_receipts.json").exists():
                operation._save_receipts(store, output, record, "handoff_verify", False)
            store.final_guard()
    gc.collect()
    return output

"""Read-only finite old-prefix/new-child chain adapter; never a final audit itself.

No CLI, process query, tracker, raw decoder, receipt writer or implicit experiment.
The future supervisor and held-exit witness contracts are isolated below. Frozen
old-prefix/native orchestration is copied explicitly; frozen globals stay intact.
Callers must snapshot this source/dependencies and seal evidence.final_guard().
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import capture_receipted_owner_exit as capture
import long_final_mixed_chain as old
import numpy as np
import receipted_runner_phase_audit as auditor

legacy, native, handoff = old.legacy, old.native, old.handoff
current, operation = old.current, old.operation
m, child = auditor.m, auditor.child
REPO, WORK, FIRST, FINAL_STOP = old.REPO, old.WORK, old.FIRST, old.FINAL_STOP
COMMON, phase, require, equal = old.COMMON, old.phase, old.require, old.equal
read_record, bound_sources = old.read_record, old.bound_sources
verify_ancestors, check_phase_edge = old.verify_ancestors, old.check_phase_edge
verify_receipt_usage, expected_new_phases = old.verify_receipt_usage, old.expected_new_phases
NEW_STAGE = old.NEW_STAGE
SUPERVISOR_STAGE = "long_remaining_receipted_child_supervisor_v1"
OPERATION_STAGE = "long_receipted_runner_phase_operation_v1"
HANDOFF_SCHEMA = "receipted_child_handoff_v1"
EXIT_SCHEMA = "owned_receipted_supervisor_exit_witness_v1"
PINS = {
    (WORK / "receipted_runner_phase_operation.py"): (
        "6cee76cbf18088a5ae7595222f28343db20a1db2bf9089344238c4328a662bdc"
    ),
    Path(capture.__file__).resolve(): (
        "ec88eeb3d1534f6437d7e4211a10531bf23e9393b4e8213d1825201fa53634f4"
    ),
    **old.PINS,
    Path(
        old.__file__
    ).resolve(): "b41c30830c61480df7e847d6814756743d115cfaf8cdaead85fd922b01a34215",
    Path(m.__file__).resolve(): "96d8d0e9aa1fb364cd3af06ae5796c252b1844d72942f0cf453d41a30105975f",
    Path(
        child.__file__
    ).resolve(): "776725de736b35715bf2d5c53226a0abc48e8f5ecff94516f9e9fdb8ffaac364",
    Path(
        auditor.__file__
    ).resolve(): "bf16c2d7acb84d0c24f257ceefe3f6d85fa9c2735688e46f05d37126687b2047",
}


def prefix_counts(k):
    require(type(k) is int and 0 <= k < 28, "stopped prefix needs 0..27 completed phases")
    stop = 76360 + 9960 * k
    return dict(
        completed_phases=k,
        through=stop,
        consumed_chunks=16 + 6 * k,
        native_frames=stop - FIRST,
        tracking_processes=5 + k,
        checkpoint_restores=4 + k,
    )


def validate_prefix_header(summary, policy):
    rows = summary.get("phases")
    require(isinstance(rows, list), "completed prefix rows required")
    counts = prefix_counts(len(rows))
    expected = dict(
        schema=NEW_STAGE,
        status="stopped_after_completed_phase",
        through=counts["through"],
        bootstrap_completed=True,
        full_sequence_complete=False,
        prefix_backtrace_provisional=True,
        logical_initialization_count=1,
        checkpoint_restore_count=counts["checkpoint_restores"],
        external_pose_reset_count=0,
        active_beam_count=6,
        independent_ID_truth=False,
        next_phase_retried=False,
        detector_started=False,
        old_process_stopped=False,
        intended_initial_parent=str(handoff.CHILD),
    )
    for key, value in expected.items():
        equal(summary.get(key), value, "stopped prefix differs: " + key)
    require(isinstance(summary.get("stop_request"), dict), "honored stop record required")
    for key, value in dict(
        schema=NEW_STAGE,
        initial_parent=str(handoff.CHILD),
        initial_stop=76360,
        target_interval=[FIRST, FINAL_STOP],
        phase_intervals=[list(p) for p in expected_new_phases()],
        receipt_evidence_outside_child_binding=True,
        detector_started=False,
        old_process_stopped=False,
        new_external_pose_seed=False,
        failed_phase_retry=False,
        old_lock_removed=False,
        new_lock_removed=False,
        scheduling_changes_scientific_protocol=False,
    ).items():
        equal(policy.get(key), value, "old prefix policy differs: " + key)
    return counts


def within(path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(REPO), "witness evidence outside workspace")
    return path


def read_bound(path, evidence):
    path = within(path)
    evidence.add(path)
    return capture.frozen.decode(path.read_bytes())


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


def witness_equal(a, b):
    # Frozen witness helpers use a boolean comparator, unlike this module's raising equal.
    return m._canonical(a) == m._canonical(b)


def verify_exit_witness(reference, supervisor_root, supervisor, evidence):
    """Verify held-handle receipts as data; never instantiate WindowsProcess."""
    witness = capture.frozen
    WITNESS_SOURCE = Path(capture.__file__).resolve()
    WITNESS_SHA = PINS[WITNESS_SOURCE]
    OLD_ROOT = within(supervisor_root)
    summary = legacy.json_output(supervisor, "summary", OLD_ROOT / "summary.json")
    CHILD = within(summary["final_run"])
    interval = summary["phases"][-1]["through"] if summary["phases"] else 76360
    start = interval - 9960
    path = within(reference["path"])
    require(path.name == "witness.provenance.json", "wrong held witness seal name")
    root = path.parent
    evidence.add(path, reference["sha256"])
    seal_path = root / "witness.provenance.json"
    seal = read_bound(seal_path, evidence)
    require(
        seal.get("stage") == EXIT_SCHEMA
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
        "source_snapshot/00_capture_receipted_owner_exit.py",
        "source_snapshot/01_capture_owned_long_exit.py",
        "source_snapshot/02_run_long_receipted_supervisor.py",
        "source_snapshot/03_pyvenv.cfg",
    }
    require(
        set(seal["outputs"]) == {str(root / n) for n in names}, "witness output membership differs"
    )
    source_pairs = [
        (WITNESS_SOURCE, "00_capture_receipted_owner_exit.py"),
        (Path(witness.__file__).resolve(), "01_capture_owned_long_exit.py"),
        (Path(current.__file__).resolve(), "02_run_long_receipted_supervisor.py"),
        (REPO / ".venv/pyvenv.cfg", "03_pyvenv.cfg"),
    ]
    for source, _ in source_pairs[:-1]:
        expected = (
            WITNESS_SHA
            if source == WITNESS_SOURCE
            else (
                capture.FROZEN_SHA
                if source == Path(witness.__file__).resolve()
                else capture.SUPERVISOR_SHA
            )
        )
        require(seal["inputs"].get(str(source)) == expected, "unregistered held witness dependency")
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
        and binding["phase_start"]["interval"] == [start, interval],
        "held child differs from last successfully audited stopped-prefix child",
    )
    expected_inputs = {
        **binding["immutable_hashes"],
        **{str(p): seal["inputs"][str(p)] for p, _ in source_pairs},
    }
    require(witness_equal(seal["inputs"], expected_inputs), "unexpected witness input membership")
    for path, digest in binding["immutable_hashes"].items():
        require(seal["inputs"].get(path) == digest, "witness immutable input differs")
    # This legacy helper only reads evidence; it performs no process operation.
    rebuilt, final_events = capture.build_binding(OLD_ROOT, binding["child_pid"])
    for key in set(binding) - {"baseline_events_sha256", "baseline_events_bytes"}:
        require(
            witness_equal(binding[key], rebuilt[key]), "current ownership binding changed: " + key
        )
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
        and witness_equal(rows[index], binding["child_event"])
        and witness_equal(rows[index - 1], binding["phase_start"])
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
        require(witness_equal(receipt["inputs"], seal["inputs"]), "role receipt inputs differ")
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
            witness_equal(obs, seal["observations"][role])
            and all(witness_equal(obs[k], v) for k, v in identity.items()),
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
        event_capture = payload["event_capture"]
        captured = root / f"events_at_{role}_exit.jsonl"
        data = captured.read_bytes()
        require(
            within(event_capture["path"]) == captured
            and witness.digest(data) == event_capture["sha256"] == seal["outputs"][str(captured)]
            and len(data) == event_capture["bytes"]
            and data.startswith(baseline)
            and final_events.startswith(data)
            and event_capture["complete_lines"] == len(witness.event_rows(data))
            and event_capture["subsequent_launch_events"] == [],
            "exit event capture differs",
        )
        if role == "owner":
            require(
                data == final_events and event_capture["partial_trailing_line"] is False,
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
        within(child["image_path"]) == capture.PYTHON.resolve()
        and Path(owner["image_path"]).resolve()
        in {capture.PYTHON.resolve(), Path(runtime["base_executable"]).resolve()},
        "witness executable identities differ",
    )
    require(owner["exit_code"] == 0, "normally stopped old owner must exit zero")
    require(
        witness.utc(supervisor["finished_at"]) <= witness.utc(owner["exit_time_utc"]),
        "owner exited before stopped supervisor seal finished",
    )
    lock = legacy.json_output(supervisor, "lock", OLD_ROOT / "supervisor.lock")
    require(witness_equal(lock, binding["owner_lock"]), "held owner differs from sealed owned lock")
    return dict(path=str(seal_path), sha256=reference["sha256"], **owner)


def verify_stopped_tail(root, summary, supervisor, counts, binding, parent, evidence):
    """Bind stop bytes and retain a possible successful-but-unused CPU registration."""
    stop = legacy.json_output(supervisor, "stop_request", root / "stop_request.json")
    equal(stop, summary["stop_request"], "honored stop summary differs")
    require(
        stop.get("child_interrupted") is False
        and stop.get("boundary") == "completed, independently audited parent",
        "stop did not occur at audited boundary",
    )
    source, saved = Path(stop["path"]).resolve(strict=True), root / "honored_stop_request.bin"
    evidence.add(source, stop["sha256"])
    evidence.add(saved, stop["sha256"])
    legacy.input_binding(supervisor, source, stop["sha256"])
    legacy.output_binding(supervisor, saved, stop["sha256"])
    require(
        source.stat().st_size == saved.stat().st_size == stop["size"], "stop byte count differs"
    )
    unused = summary.get("incomplete_phase")
    if unused is None:
        return None
    start, end = expected_new_phases()[counts["completed_phases"]]
    op = root / "phases" / f"f{start:06d}-{end:06d}"
    equal(str(Path(unused).resolve(strict=True)), str(op), "unused registration position differs")
    require(
        not any(
            (op / name).exists()
            for name in (
                "tracking",
                "child_execution.json",
                "tracking.log",
                "operation.provenance.json",
                "audit.json",
            )
        ),
        "incomplete phase contains execution evidence",
    )
    reg = op / "registration"
    seal = read_record(reg, "registration.provenance.json", operation.REGISTRATION_STAGE, evidence)
    plan = legacy.json_output(seal, "plan", reg / "plan.json")
    for key, value in dict(
        parent_phase=str(parent),
        expected_output=str(op / "tracking"),
        next_native_interval=[start, end],
        expected_restores=counts["checkpoint_restores"] + 1,
        frozen_input_and_code_binding=binding,
    ).items():
        equal(plan.get(key), value, "unused registration differs: " + key)
    paths = [
        reg / name
        for name in (
            "registration.provenance.json",
            "plan.json",
            "execution.json",
            "mask_validation_receipts.json",
        )
    ]
    for path in paths:
        legacy.output_binding(supervisor, path, evidence.paths[str(path)])
    return dict(
        root=str(reg),
        seal_sha256=evidence.paths[str(paths[0])],
        interval=[start, end],
        tracking_processes=0,
        logical_restores=0,
    )


def load_completed_prefix(root, expected_seal, exit_witness, evidence):
    """Verify the sealed stopped old prefix without promoting it to full success."""
    root = Path(root).resolve(strict=True)
    path = root / "supervisor.provenance.json"
    evidence.add(path, expected_seal)
    header = COMMON.read_json(path)
    require(header.get("status") == "ok", "completed successful supervisor required")
    require(header.get("stage") == NEW_STAGE, "unsupported supervisor version")
    # Check completion before reading heavyweight dependency payloads or any NPZ arrays.
    summary = legacy.json_output(header, "summary", root / "summary.json")
    policy = legacy.json_output(header, "plan", root / "supervisor_plan.json")
    counts = validate_prefix_header(summary, policy)
    supervisor = read_record(root, "supervisor.provenance.json", NEW_STAGE, evidence)
    for path, digest in PINS.items():
        evidence.add(path, digest)
    code, binding = policy["source_hashes"], policy["fixed_bindings"]
    bound_sources(supervisor, root, code, evidence)
    witness_root = Path(policy["handoff_lineage"]["witness_root"]).resolve(strict=True)
    transferred = handoff.verify_handoff(witness_root, evidence)
    equal(transferred["lineage"], policy["handoff_lineage"], "supervisor handoff lineage differs")
    equal(transferred["binding"], binding, "handoff science binding differs")
    ancestors = verify_ancestors(binding, evidence)
    old_root, old_parent = transferred["old_root"], handoff.OLD_PARENT
    prior_record = read_record(
        old_parent, "continuation.provenance.json", "three_body_long_continuation", evidence
    )
    old_bootstrap = old_root / "bootstrap"
    old_audit_seal = read_record(
        old_bootstrap, "audit.provenance.json", "long_supervised_bootstrap_audit", evidence
    )
    old_audit = legacy.json_output(old_audit_seal, "summary", old_bootstrap / "summary.json")
    legacy.check_audit(
        old_audit,
        old_parent,
        prior_record["outputs"]["candidate"]["sha256"],
        stop=66400,
        restores=3,
    )
    equal(
        old_audit["bootstrap_lineage"]["audit_seal_sha256"],
        phase.MINUTE_AUDIT_SHA,
        "missing independent minute anchor",
    )
    evidence.add(phase.MINUTE_AUDIT / "audit.provenance.json", phase.MINUTE_AUDIT_SHA)
    legacy.input_binding(
        old_audit_seal, phase.MINUTE_AUDIT / "audit.provenance.json", phase.MINUTE_AUDIT_SHA
    )
    old_row, previous_record, old_result = check_phase_edge(
        handoff.CHILD.parent,
        old_parent,
        prior_record,
        old_bootstrap,
        "long_supervised_bootstrap_audit",
        "audit.provenance.json",
        binding,
        66400,
        76360,
        4,
        None,
        evidence,
        receipted=False,
    )
    old_operation = COMMON.read_json(handoff.CHILD.parent / "operation.provenance.json")
    require(
        datetime.fromisoformat(old_operation["finished_at"])
        <= datetime.fromisoformat(transferred["lineage"]["owner_exit"]["exit_time_utc"])
        <= datetime.fromisoformat(supervisor["started_at"]),
        "old operation/owner exit/new supervisor chronology differs",
    )
    bootstrap = root / "bootstrap"
    bootstrap_seal = read_record(bootstrap, "audit.provenance.json", handoff.STAGE, evidence)
    audit = legacy.json_output(bootstrap_seal, "summary", bootstrap / "summary.json")
    for key in old_result:
        equal(
            audit[key], old_result[key], "handoff re-audit differs from completed old audit: " + key
        )
    equal(audit["handoff_lineage"], transferred["lineage"], "bootstrap handoff differs")
    equal(
        legacy.json_output(bootstrap_seal, "handoff", bootstrap / "handoff.json"),
        transferred["lineage"],
        "handoff JSON differs",
    )
    require(
        datetime.fromisoformat(transferred["lineage"]["owner_exit"]["exit_time_utc"])
        <= datetime.fromisoformat(bootstrap_seal["started_at"]),
        "bootstrap preceded owner exit",
    )
    bound_sources(bootstrap_seal, bootstrap, code, evidence)
    cache_root = Path(policy["receipt_cache_root"]).resolve(strict=True)
    verify_receipt_usage(
        bootstrap_seal,
        bootstrap,
        label="handoff_verify",
        plan=transferred["old_plan"],
        evidence=evidence,
        cache_root=cache_root,
    )
    for name in (
        "audit.provenance.json",
        "summary.json",
        "handoff.json",
        "mask_validation_receipts.json",
    ):
        legacy.output_binding(supervisor, bootstrap / name, evidence.paths[str(bootstrap / name)])
    chain, parent = [old_row], handoff.CHILD
    prev_root, prev_stage, prev_name = bootstrap, handoff.STAGE, "audit.provenance.json"
    for i, ((start, stop), row) in enumerate(
        zip(expected_new_phases()[: counts["completed_phases"]], summary["phases"], strict=True)
    ):
        op_root = root / "phases" / f"f{start:06d}-{stop:06d}"
        run = op_root / "tracking"
        equal(
            row,
            dict(
                parent=str(parent),
                run=str(run),
                through=stop,
                parent_audit=str(prev_root),
                operation=str(op_root),
                checkpoint_restore_count=i + 5,
            ),
            "new phase row/order differs",
        )
        linked, previous_record, audit = check_phase_edge(
            op_root,
            parent,
            previous_record,
            prev_root,
            prev_stage,
            prev_name,
            binding,
            start,
            stop,
            i + 5,
            supervisor,
            evidence,
            receipted=True,
            code=code,
            cache_root=cache_root,
        )
        chain.append(linked)
        parent, prev_root, prev_stage, prev_name = (
            run,
            op_root,
            operation.OPERATION_STAGE,
            "operation.provenance.json",
        )
    require(
        summary["final_run"] == str(parent)
        and chain[-1]["restores"] == counts["checkpoint_restores"],
        "terminal run/restore differs",
    )
    require(
        len(ancestors) + len(chain) == counts["tracking_processes"], "prefix process count differs"
    )
    unused = verify_stopped_tail(root, summary, supervisor, counts, binding, parent, evidence)
    exited = verify_exit_witness(exit_witness, root, supervisor, evidence)
    return dict(
        supervisor=root,
        supervisor_record=supervisor,
        policy=policy,
        final_run=parent,
        final_record=previous_record,
        phases=chain,
        binding=binding,
        final_audit=audit,
        ancestors=ancestors,
        chain_version="stopped_old_receipted_prefix_v1",
        handoff_lineage=transferred["lineage"],
        tracking_processes=counts["tracking_processes"],
        checkpoint_restores=counts["checkpoint_restores"],
        prefix_counts=counts,
        summary=summary,
        last_audit_root=prev_root,
        last_audit_stage=prev_stage,
        last_audit_name=prev_name,
        unused_registration=unused,
        owner_exit=exited,
        sampled_frames=native.fixed_sample_frames(),
        final_audit_complete=False,
    )


def verify_source_closure(record, root, evidence, required):
    """Executed imports come from this producer's own snapshot tree.

    Inputs also contain earlier producers' .py snapshots; those are evidence bytes,
    not modules imported by this producer and must not acquire another snapshot.
    """
    snapshot_root = root / "code_snapshot"
    code = {
        str(REPO / Path(item["path"]).resolve().relative_to(snapshot_root)): item["sha256"]
        for item in record["outputs"].values()
        if Path(item["path"]).suffix == ".py"
        and Path(item["path"]).resolve().is_relative_to(snapshot_root)
    }
    for source, digest in required.items():
        equal(code.get(str(source)), digest, "missing pinned executed source")
    bound_sources(record, root, code, evidence)
    return code


def validate_new_header(summary, policy, handoff_data):
    """Small future-supervisor adapter; reject incomplete runs before array loading."""
    k = handoff_data["old_completed_phases"]
    counts = prefix_counts(k)
    for key, value in dict(
        schema=SUPERVISOR_STAGE,
        status="complete",
        through=FINAL_STOP,
        full_sequence_complete=True,
        prefix_backtrace_provisional=False,
        logical_initialization_count=1,
        checkpoint_restore_count=32,
        external_pose_reset_count=0,
        active_beam_count=6,
        tracking_processes=33,
        consumed_chunk_count=180,
        stop_request=None,
        incomplete_phase=None,
        independent_ID_truth=False,
        final_audit_complete=False,
        detector_started=False,
        old_process_stopped=False,
        next_phase_retried=False,
    ).items():
        equal(summary.get(key), value, "new supervisor is not complete: " + key)
    require(
        isinstance(summary.get("phases"), list) and len(summary["phases"]) == 28 - k,
        "remaining phase denominator differs",
    )
    for key, value in dict(
        schema=SUPERVISOR_STAGE,
        initial_parent=handoff_data["initial_parent"],
        initial_stop=counts["through"],
        initial_restores=counts["checkpoint_restores"],
        target_interval=[FIRST, FINAL_STOP],
        phase_intervals=[list(p) for p in expected_new_phases()[k:]],
        new_external_pose_seed=False,
        failed_phase_retry=False,
        scheduling_changes_scientific_protocol=False,
    ).items():
        equal(policy.get(key), value, "new supervisor policy differs: " + key)
    return counts


def validate_handoff(data, prefix):
    counts = prefix["prefix_counts"]
    for key, value in dict(
        schema=HANDOFF_SCHEMA,
        old_completed_phases=counts["completed_phases"],
        initial_parent=str(prefix["final_run"]),
        initial_stop=counts["through"],
        initial_restores=counts["checkpoint_restores"],
        unused_registration=prefix["unused_registration"],
        prefix_counts=counts,
        scientific_context_changed=False,
    ).items():
        equal(data.get(key), value, "handoff differs from audited old prefix: " + key)


def phase_row(root, parent, interval, restores, seals):
    """Exact future-supervisor row; lower named roles carry the detailed contract."""
    return dict(
        parent=str(parent),
        run=str(root / "tracking"),
        operation=str(root),
        interval=list(interval),
        checkpoint_restore_count=restores,
        **seals,
    )


def validate_new_edge(
    plan,
    producer,
    summary,
    audit,
    before_meta,
    after_meta,
    proof,
    legacy_binding,
    binding,
    *,
    parent,
    run,
    start,
    stop,
    restores,
    parent_audit_path,
    parent_audit_sha,
):
    """Metadata-only edge checks; sealed independent audit owns full 39/21/recurrences."""
    edge = auditor.registered_edge(plan, legacy_binding, plan["parent_binding"])
    equal(plan["binding"], binding, "registered binding changed")
    for key, value in dict(
        parent=str(parent),
        expected_output=str(run),
        next_native_interval=[start, stop],
        expected_restores=restores,
        parent_context=before_meta["context"],
    ).items():
        equal(plan.get(key), value, "new phase registration differs: " + key)
    equal(plan["parent_audit"]["path"], str(parent_audit_path), "parent audit path differs")
    equal(plan["parent_audit"]["sha256"], parent_audit_sha, "parent audit SHA differs")
    equal(producer["params"]["bindings"], binding, "executed binding changed")
    equal(producer["params"]["resume_phase"], str(parent), "executed parent changed")
    equal(producer["params"]["through_frame"], stop, "executed stop changed")
    auditor.validate_summary(
        summary,
        after_meta,
        start=start,
        stop=stop,
        restores=restores,
        chunks=(stop - FIRST) // native.CHUNK,
    )
    equal(after_meta["context"], before_meta["context"], "scientific context changed")
    require(
        before_meta["initialize_count"] == 1
        and before_meta["restore_count"] + 1 == restores
        and before_meta["previous_frame"] == start - 1
        and before_meta["processed_frames"] == start - FIRST,
        "parent lifecycle differs",
    )
    signatures = plan["parent_state_signatures"]
    require(isinstance(signatures, dict) and len(signatures) == 39, "39 import signatures required")
    equal(
        proof,
        dict(
            all_checkpoint_arrays_exact=sorted(signatures),
            array_signatures=signatures,
            context_unchanged=True,
            initialize_count=1,
            restore_count=restores,
            previous_frame=start - 1,
            external_pose_reset_count=0,
        ),
        "39 logical import differs",
    )
    legacy.check_audit(
        audit, run, producer["outputs"]["candidate"]["sha256"], stop=stop, restores=restores
    )
    for key, value in dict(
        audit_version=auditor.STAGE,
        parent=str(parent),
        new_interval=[start, stop],
        restores_during_phase=0,
        independent_phase_audit_complete=True,
        producer_independent_phase_audit_complete=False,
        transition=edge,
        parent_audit=plan["parent_audit"],
    ).items():
        equal(audit.get(key), value, "separate independent phase audit differs: " + key)
    equal(summary["verifier_transition"], edge, "producer transition differs")
    equal(
        summary["history_prefix"],
        dict(
            parent_native_frames=start - FIRST,
            parent_rows=plan["parent_history_rows"],
            all_history_prefix_exact=True,
        ),
        "producer original history prefix differs",
    )
    equal(summary["active_beam_count"], 6, "producer dropped a retained beam")
    return edge


def receipt_validator_binding(evidence):
    """Read-only construction of the frozen receipt validator identity; no store mkdir."""
    receipts = old.receipts
    paths = [
        Path(x.__file__).resolve()
        for x in (receipts, receipts.chunk_masks, receipts.prepare_sequence)
    ]
    return dict(
        source_sha256={str(p): evidence.add(p) for p in paths},
        python=old.platform.python_version(),
        implementation=old.platform.python_implementation(),
        numpy=np.__version__,
        schema_dtypes={k: v.str for k, v in receipts.chunk_masks.SCHEMA_DTYPES.items()},
        mask_bitorder="little",
    )


def check_receipted_edge(
    root,
    parent,
    previous_record,
    parent_audit_path,
    parent_audit_sha,
    legacy_binding,
    binding,
    start,
    stop,
    restores,
    supervisor,
    evidence,
):
    """Read the new operation's sealed artifacts; never run its execution API."""
    reg, run, audit_root = root / "registration", root / "tracking", root / "audit"
    op = read_record(root, "operation.provenance.json", OPERATION_STAGE, evidence)
    operation_summary = legacy.json_output(op, "summary", root / "summary.json")
    for key, value in dict(
        schema=OPERATION_STAGE,
        status="ok",
        completed_run=str(run),
        independent_audit=str(audit_root),
        child_started=True,
        parent_promoted=False,
        operation_success_requires_successful_seal=True,
        automatic_retry=False,
        execution_resource_owner_external=True,
        independent_ID_truth=False,
    ).items():
        equal(operation_summary.get(key), value, "successful operation required: " + key)
    records = {}
    for name, path, filename, stage in (
        ("registration", reg, "registration.provenance.json", m.REGISTRATION_STAGE),
        ("tracking", run, "continuation.provenance.json", m.STAGE),
        ("audit", audit_root, "audit.provenance.json", auditor.STAGE),
    ):
        records[name] = read_record(path, filename, stage, evidence)
        legacy.output_binding(op, path / filename, evidence.paths[str(path / filename)])
    registration, producer, audit_record = (
        records[x] for x in ("registration", "tracking", "audit")
    )
    reg_sha = evidence.paths[str(reg / "registration.provenance.json")]
    plan, registration = child.load_registration(reg, reg_sha, evidence)
    equal(operation_summary["registration_sha256"], reg_sha, "operation registration differs")
    equal(producer["params"]["registration_sha256"], reg_sha, "producer registration differs")
    equal(producer["params"]["registration_path"], str(reg), "producer registration path differs")
    expected_inputs = auditor.child_input_hashes(registration, reg, reg_sha)
    auditor.exact_child_inputs(producer, expected_inputs, evidence)
    equal(plan["parent_binding"], previous_record["params"]["bindings"], "parent binding differs")
    equal(
        plan["parent_seal_sha256"],
        evidence.paths[str(parent / "continuation.provenance.json")],
        "registered parent seal differs",
    )
    for name, key in (
        ("manifest.json", "parent_checkpoint_sha256"),
        ("metadata.json", "parent_metadata_sha256"),
        ("arrays.npz", "parent_payload_sha256"),
    ):
        equal(
            evidence.paths[str(parent / "checkpoint" / name)],
            plan[key],
            "registered parent checkpoint differs",
        )
    owned = legacy.json_output(op, "child_execution", root / "child_execution.json")
    owned_check = auditor.verify_owned_execution(owned, run, reg, reg_sha, registration, producer)
    execution = legacy.json_output(op, "execution", root / "execution.json")
    equal(
        execution["command"], auditor.expected_command(run, reg, reg_sha), "operation CLI differs"
    )
    for key, value in dict(
        registration_sha256=reg_sha,
        parent=str(parent),
        output=str(run),
        execution_resource_owner_external=True,
    ).items():
        equal(execution.get(key), value, "operation execution differs: " + key)
    registration_info = legacy.json_output(
        op, "registration_execution", root / "registration_execution.json"
    )
    require(
        registration_info.get("schema") == "receipted_long_registration_execution_v1"
        and type(registration_info.get("pid")) is int
        and registration_info["pid"] > 0
        and type(registration_info.get("returncode")) is int
        and registration_info["returncode"] == 0
        and registration_info.get("child_alive_after_return") is False,
        "CPU registration did not return successfully",
    )
    registration_command = legacy.json_output(
        op, "registration_command", root / "registration_command.json"
    )
    equal(registration_info["command"], registration_command["command"], "CPU command differs")
    command = [
        str((REPO / ".venv/Scripts/python.exe").resolve()),
        "-B",
        str(Path(child.__file__).resolve()),
        "--register-only",
        "--parent",
        str(parent),
        "--parent-seal-sha256",
        plan["parent_seal_sha256"],
        "--parent-audit-seal",
        str(parent_audit_path),
        "--parent-audit-sha256",
        parent_audit_sha,
        "--through-frame",
        str(stop),
        "--receipt-root",
        plan["receipt_cache_root"],
        "--run-output",
        str(run),
        "--output",
        str(reg),
    ]
    equal(registration_command["command"], command, "CPU registration CLI differs")
    t = auditor.timestamp
    require(
        t(previous_record["finished_at"])
        <= t(registration["started_at"])
        <= t(registration["finished_at"])
        <= t(registration_info["finished_at"])
        <= t(owned["started_at"])
        <= t(owned["finished_at"])
        <= t(audit_record["started_at"])
        <= t(audit_record["finished_at"])
        <= t(op["finished_at"]),
        "operation/registration/child/audit chronology differs",
    )
    require(
        t(op["started_at"]) <= t(registration_info["started_at"]) <= t(registration["finished_at"]),
        "CPU process start chronology differs",
    )
    audit = legacy.json_output(audit_record, "summary", audit_root / "summary.json")
    equal(audit["registration_sha256"], reg_sha, "audit registration differs")
    equal(
        audit["owned_execution"],
        dict(
            **owned_check,
            path=str(root / "child_execution.json"),
            sha256=evidence.paths[str(root / "child_execution.json")],
        ),
        "audit ownership proof differs",
    )
    before_meta, after_meta = (
        COMMON.read_json(p / "checkpoint/metadata.json") for p in (parent, run)
    )
    summary = legacy.json_output(producer, "summary", run / "summary.json")
    proof = legacy.json_output(producer, "state_import", run / "state_import.json")
    edge = validate_new_edge(
        plan,
        producer,
        summary,
        audit,
        before_meta,
        after_meta,
        proof,
        legacy_binding,
        binding,
        parent=parent,
        run=run,
        start=start,
        stop=stop,
        restores=restores,
        parent_audit_path=parent_audit_path,
        parent_audit_sha=parent_audit_sha,
    )
    validator = receipt_validator_binding(evidence)
    receipt_registration = auditor.verify_receipt_usage(
        registration, reg, plan, evidence, validator
    )
    receipt_execution = auditor.verify_receipt_usage(producer, run, plan, evidence, validator)
    equal(
        receipt_registration["stable_uses"],
        receipt_execution["stable_uses"],
        "child receipt uses changed from registration",
    )
    for key, receipt in (
        ("receipt_registration", receipt_registration),
        ("receipt_execution", receipt_execution),
    ):
        equal(
            audit[key],
            {k: v for k, v in receipt.items() if k != "stable_uses"},
            "audited receipt usage differs",
        )
    # The independent audit must actually bind the producer's complete scientific evidence.
    for item in list(producer["inputs"].values()) + list(producer["outputs"].values()):
        legacy.input_binding(audit_record, Path(item["path"]), item["sha256"])
    for path in (
        run / "continuation.provenance.json",
        parent_audit_path,
        root / "child_execution.json",
    ):
        legacy.input_binding(audit_record, path, evidence.paths[str(path)])
    for record, directory in ((registration, reg), (producer, run)):
        bound_sources(record, directory, binding["code_sha256"], evidence)
    pinned = {
        p: d
        for p, d in PINS.items()
        if p
        in (
            Path(auditor.__file__).resolve(),
            Path(m.__file__).resolve(),
            Path(child.__file__).resolve(),
        )
    }
    verify_source_closure(audit_record, audit_root, evidence, pinned)
    equal(
        verify_source_closure(
            op,
            root,
            evidence,
            {
                **pinned,
                WORK / "receipted_runner_phase_operation.py": PINS[
                    WORK / "receipted_runner_phase_operation.py"
                ],
            },
        ),
        execution["source_hashes"],
        "operation executed source closure differs",
    )
    # All operation artifacts, including logs and every lower seal, belong to the owner seal.
    for item in op["outputs"].values():
        legacy.output_binding(supervisor, Path(item["path"]), item["sha256"])
    legacy.output_binding(
        supervisor,
        root / "operation.provenance.json",
        evidence.paths[str(root / "operation.provenance.json")],
    )
    seals = dict(
        operation_sha256=evidence.paths[str(root / "operation.provenance.json")],
        registration_sha256=reg_sha,
        child_sha256=evidence.paths[str(run / "continuation.provenance.json")],
        audit_sha256=evidence.paths[str(audit_root / "audit.provenance.json")],
    )
    return (
        dict(
            parent=str(parent),
            run=str(run),
            interval=[start, stop],
            restores=restores,
            candidate_sha256=audit["candidate_sha256"],
            transition=edge,
            **seals,
        ),
        producer,
        audit,
    )


def load_supervisor_chain(root, evidence):
    root = Path(root).resolve(strict=True)
    path = root / "supervisor.provenance.json"
    evidence.add(path)
    header = COMMON.read_json(path)
    require(
        header.get("stage") == SUPERVISOR_STAGE and header.get("status") == "ok",
        "completed new supervisor required",
    )
    summary = legacy.json_output(header, "summary", root / "summary.json")
    policy = legacy.json_output(header, "plan", root / "supervisor_plan.json")
    transferred = legacy.json_output(header, "handoff", root / "handoff.json")
    counts = validate_new_header(summary, policy, transferred)
    record = read_record(root, "supervisor.provenance.json", SUPERVISOR_STAGE, evidence)
    for source, digest in PINS.items():
        evidence.add(source, digest)
    evidence.add(Path(__file__))
    bound_sources(record, root, policy["source_hashes"], evidence)
    prefix = load_completed_prefix(
        transferred["old_supervisor"],
        transferred["old_supervisor_sha256"],
        transferred["exit_witness"],
        evidence,
    )
    validate_handoff(transferred, prefix)
    for p in (
        Path(transferred["old_supervisor"]) / "supervisor.provenance.json",
        Path(transferred["exit_witness"]["path"]),
    ):
        legacy.input_binding(record, p, evidence.paths[str(p)])
    require(
        auditor.timestamp(prefix["owner_exit"]["exit_time_utc"])
        <= auditor.timestamp(record["started_at"]),
        "new owner started before old owner exited",
    )
    binding, legacy_binding = policy["fixed_bindings"], prefix["binding"]
    equal(
        binding,
        m.execution_binding(legacy_binding, binding["code_sha256"]),
        "new binding exceeds four I/O changes",
    )
    parent, previous = prefix["final_run"], prefix["final_record"]
    audit_path = prefix["last_audit_root"] / prefix["last_audit_name"]
    audit_sha = evidence.paths[str(audit_path)]
    chain = list(prefix["phases"])
    for i, ((start, stop), row) in enumerate(
        zip(expected_new_phases()[counts["completed_phases"] :], summary["phases"], strict=True)
    ):
        op_root = root / "phases" / f"f{start:06d}-{stop:06d}"
        restores = counts["checkpoint_restores"] + i + 1
        linked, previous, audit = check_receipted_edge(
            op_root,
            parent,
            previous,
            audit_path,
            audit_sha,
            legacy_binding,
            binding,
            start,
            stop,
            restores,
            record,
            evidence,
        )
        expected_kind = "legacy_to_receipted" if i == 0 else "receipted_to_receipted"
        equal(linked["transition"]["kind"], expected_kind, "migration position/count differs")
        seals = {
            key: linked[key]
            for key in ("operation_sha256", "registration_sha256", "child_sha256", "audit_sha256")
        }
        equal(
            row,
            phase_row(op_root, parent, [start, stop], restores, seals),
            "phase row/order differs",
        )
        require(
            auditor.timestamp(record["started_at"])
            <= auditor.timestamp(
                COMMON.read_json(op_root / "operation.provenance.json")["started_at"]
            )
            <= auditor.timestamp(previous["finished_at"])
            <= auditor.timestamp(record["finished_at"]),
            "phase outside owning supervisor lifetime",
        )
        chain.append(linked)
        parent, audit_path, audit_sha = (
            Path(linked["run"]),
            op_root / "audit/audit.provenance.json",
            linked["audit_sha256"],
        )
    require(
        len(prefix["ancestors"]) + len(chain) == 33 and chain[-1]["restores"] == 32,
        "terminal process/restore count differs",
    )
    equal(summary["final_run"], str(parent), "terminal run differs")
    equal(chain[-1]["interval"], [345280, FINAL_STOP], "terminal remainder differs")
    return dict(
        supervisor=root,
        supervisor_record=record,
        policy=policy,
        final_run=parent,
        final_record=previous,
        phases=chain,
        binding=binding,
        final_audit=audit,
        ancestors=prefix["ancestors"],
        prefix=prefix,
        unused_registration=prefix["unused_registration"],
        chain_version="old_prefix_then_receipted_child_v1",
        tracking_processes=33,
        checkpoint_restores=32,
        sampled_frames=native.fixed_sample_frames(),
        final_audit_complete=False,
    )


# Explicit frozen terminal/native orchestration with version-specific entry guards.
def load_completed_final(supervisor_root, evidence):
    """Load final arrays once after sealed chain checks; not a complete final audit."""
    chain = load_supervisor_chain(supervisor_root, evidence)
    root, record = chain["final_run"], chain["final_record"]
    summary = legacy.json_output(record, "summary", root / "summary.json")
    require(
        Path(record["outputs"]["candidate"]["path"]).resolve() == root / "candidate.npz"
        and Path(record["outputs"]["checkpoint_manifest"]["path"]).resolve()
        == root / "checkpoint/manifest.json",
        "terminal output role differs",
    )
    state, metadata, manifest = phase.old.prior.read_checkpoint(
        root / "checkpoint", expected_bindings=chain["binding"]
    )
    require(
        len(state) == 39
        and metadata["processed_frames"] == FINAL_STOP - FIRST
        and metadata["previous_frame"] == FINAL_STOP - 1
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 32
        and state["state_pose"].shape == (6, 3, 3, 3),
        "terminal 39-state/lifecycle differs",
    )
    auditor.validate_summary(
        summary, metadata, start=345280, stop=FINAL_STOP, restores=32, chunks=180
    )
    consumed = legacy.json_output(record, "consumed_chunks", root / "consumed_chunks.json")
    native.validate_consumed_ranges(consumed)
    candidate = phase.old.read_npz(root / "candidate.npz")
    traced = COMMON.traced_candidate(state)
    require(
        set(candidate) == set(traced) and len(candidate) == 21, "terminal candidate schema differs"
    )
    exact = {key: COMMON.same_array(candidate[key], traced[key], key) for key in traced}
    COMMON.same_array(
        candidate["frame_idx"], np.arange(FIRST, FINAL_STOP, dtype=np.int64), "native frames"
    )
    legacy.support_input(candidate)
    require(np.isfinite(candidate["pose_xyz"]).all(), "persistent latent poses must all be finite")
    equal(
        exact,
        chain["final_audit"]["all_21_candidate_arrays_exact_backtrace"],
        "terminal exact-backtrace digests differ from phase audit",
    )
    counts = legacy.SUPPORT.direct_counts(candidate)
    equal(counts, chain["final_audit"]["counts"], "terminal audited availability differs")
    return dict(
        chain=chain,
        candidate=candidate,
        checkpoint=state,
        checkpoint_metadata=metadata,
        checkpoint_manifest=manifest,
        exact_backtrace=exact,
        counts=counts,
        final_audit_complete=False,
        remaining_checks=[
            "all-frame support replay",
            "543 fixed native samples / 1629 images",
            "179 chunk-boundary flow checks",
            "33 checkpoint previous observations",
            "global detector completion/assembly",
            "final provenance/report",
        ],
    )


def load_native_sample_inputs(chain, evidence):
    """Verify completed chain and consumed attempts; no raw decode or candidate load."""
    final = legacy

    evidence.add(native.PLAN, native.PLAN_SHA)
    evidence.add(Path(__file__))
    require(
        chain.get("chain_version") == "old_prefix_then_receipted_child_v1"
        and chain.get("tracking_processes") == 33
        and chain.get("checkpoint_restores") == 32
        and chain.get("final_audit_complete") is False
        and chain["final_record"]["params"]["through_frame"] == FINAL_STOP,
        "verified complete receipt-child chain required",
    )
    evidence.add(Path(final.__file__))
    record, run = chain["final_record"], chain["final_run"]
    records = final.json_output(record, "consumed_chunks", run / "consumed_chunks.json")
    native.validate_consumed_ranges(records)
    fixed = chain["binding"]["inputs"]
    for path, digest in fixed["detector_binding"]["code_sha256"].items():
        evidence.add(REPO / path, digest)
    stage_root = Path(fixed["input_stage_root"]).resolve(strict=True)
    chunks, videos = [], {}
    for row in records:
        attempt = native.checked_relative(stage_root, row["attempt_path"])
        manifest_path, seal_path = attempt / "manifest.json", attempt / "staging.provenance.json"
        evidence.add(manifest_path, row["manifest_sha256"])
        evidence.add(seal_path, row["provenance_sha256"])
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        require(
            seal.get("status") == "ok"
            and seal.get("stage") == "prepare_independent_mask_chunk"
            and seal.get("finished_at"),
            "sample chunk lacks successful detector seal",
        )
        final.equal(
            seal["params"]["binding"], fixed["detector_binding"], "detector binding differs"
        )
        # The fixed binding stores path/SHA only; provenance also has size/mtime.
        declared_inputs = {
            key: dict(path=str(Path(value["path"]).resolve(strict=True)), sha256=value["sha256"])
            for key, value in seal["inputs"].items()
        }
        final.equal(declared_inputs, fixed["detector_inputs"], "detector inputs differ")
        for item in seal["inputs"].values():
            require(
                Path(item["path"]).stat().st_size == item["size"], "detector input size differs"
            )
            evidence.add(item["path"], item["sha256"])
        outputs = {
            key: dict(path=str(Path(v["path"]).resolve(strict=True)), sha256=v["sha256"])
            for key, v in seal["outputs"].items()
        }
        final.equal(outputs, row["output_hashes"], "consumed chunk output proofs differ")
        require(
            Path(outputs["manifest"]["path"]) == manifest_path
            and outputs["manifest"]["sha256"] == row["manifest_sha256"],
            "manifest is not bound to chunk output",
        )
        for item in outputs.values():
            require(Path(item["path"]).is_relative_to(attempt), "chunk output escaped attempt")
            evidence.add(item["path"], item["sha256"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(
            manifest["experiment"] == native.EXPERIMENT
            and manifest["fps"] == 166
            and manifest["image_shape_hw"] == [1200, 1200]
            and manifest["frames"] == list(range(row["start"], row["stop"]))
            and manifest["protocol_sha256"] == fixed["protocol_sha256"]
            and manifest["tracking_performed"] is False
            and manifest["human_truth_created"] is False,
            "chunk native metadata differs",
        )
        final.equal(
            manifest["binding"], fixed["detector_binding"], "manifest detector binding differs"
        )
        final.equal(
            manifest["runtime"],
            fixed["detector_binding"]["runtime"],
            "manifest detector runtime differs",
        )
        native.validate_source_manifest(attempt, manifest)
        for view in native.VIEWS:
            for category, suffix in (
                ("raw", "raw"),
                ("masks", "masks"),
                ("backgrounds", "background"),
            ):
                item = manifest[category][view]
                path = native.checked_relative(attempt, item["path"])
                final.equal(
                    outputs[view + "_" + suffix],
                    dict(path=str(path), sha256=item["sha256"]),
                    "manifest payload/output binding differs",
                )
            raw = manifest["raw"][view]
            require(
                len(raw["frame_bgr_sha256"]) == native.CHUNK
                and all(
                    isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)
                    for x in raw["frame_bgr_sha256"]
                ),
                "invalid native BGR SHA list",
            )
            link = row["raw_hardlinks"][view]
            original, alias = (
                Path(link["source_path"]).resolve(strict=True),
                native.checked_relative(attempt, raw["path"]),
            )
            stat = alias.stat()
            original_input = fixed["detector_inputs"]["video:" + view]
            require(
                Path(link["path"]).resolve() == alias
                and os.path.samefile(original, alias)
                and Path(original_input["path"]).resolve(strict=True) == original
                and original_input["sha256"] == raw["sha256"]
                and (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                == (link["device"], link["inode"], link["bytes"], link["mtime_ns"])
                and link["sha256"] == raw["sha256"],
                "raw source hardlink differs",
            )
            evidence.add(original, raw["sha256"])
            require(view not in videos or videos[view] == original, "chunk changed canonical video")
            videos[view] = original
            require(
                manifest["backgrounds"][view]["sha256"] == fixed["background_sha256"][view],
                "registered median background differs",
            )
        chunks.append(dict(root=attempt, manifest=manifest, record=row))
    seed = fixed["seed"]
    input_root = Path(seed["input_root"]).resolve(strict=True)
    for path, digest in seed["SL_and_calibration_sha256"].items():
        evidence.add(path, digest)
    nodes = native.CachedNodes(
        input_root, native.EXPERIMENT, np.arange(native.FIRST, native.STOP, dtype=np.int64)
    )
    for path in nodes.paths.values():
        require(str(path.resolve()) in seed["SL_and_calibration_sha256"], "unbound SL node input")
    return native.NativeSampleInputs(
        native.fixed_sample_frames(),
        tuple(chunks),
        videos,
        nodes,
        evidence,
        chain,
        dict(
            schema="long_final_native_sample_inputs_v1",
            native_samples=543,
            raw_images=1629,
            independent_ID_truth=False,
            candidate_or_checkpoint_arrays_opened=False,
            flow_boundary_checks=False,
            checkpoint_previous_checks=False,
            global_detector_assembly_verified=False,
        ),
    )

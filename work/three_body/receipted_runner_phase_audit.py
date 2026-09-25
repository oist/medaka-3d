"""Independent CPU audit for a completed receipt-child phase, including full end.

No tracking, restore, process management, GPU query, raw decode, flow or geometric
reprojection. The producer's self-audit flag remains False; this distinct artifact
certifies codec/history/lineage checks, not independent identity truth. Receipt
validation retains the existing trusted-local-writer/runtime contract.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import long_phase_audit as history
import mask_validation_receipts as receipts
import numpy as np
import receipted_long_runner_migration as m
import run_receipted_long_continuation as child

STAGE = "long_receipted_runner_phase_audit_v1"
SUMMARY_SCHEMA = "long_continuation_generic_phase_audit_v1"
EXECUTION_SCHEMA = "receipted_long_child_execution_v1"
PINNED = {
    "work/three_body/receipted_long_runner_migration.py": (
        "96d8d0e9aa1fb364cd3af06ae5796c252b1844d72942f0cf453d41a30105975f"
    ),
    "work/three_body/run_receipted_long_continuation.py": (
        "776725de736b35715bf2d5c53226a0abc48e8f5ecff94516f9e9fdb8ffaac364"
    ),
    "work/three_body/long_phase_audit.py": (
        "a89541031a662b1c6633d6b3cc996dd7d5e88da6627eb3e4ef17261103963ca0"
    ),
    "work/three_body/audit_long_multichunk_minute.py": (
        "3e186c0f06e7bfbe64e4dd8bfdae066533c8d4203731c4f46c1f97a3f475dbbd"
    ),
}
COMMON = history.COMMON
require, equal = m.require, m.equal


def expected_command(run, registration, registration_sha256):
    return [
        str((m.REPO / ".venv/Scripts/python.exe").resolve()),
        "-B",
        str(Path(child.__file__).resolve()),
        "--registration",
        str(registration),
        "--registration-sha256",
        registration_sha256,
        "--output",
        str(run),
    ]


def timestamp(value):
    require(type(value) is str, "timestamp must be text")
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "timestamp must carry timezone")
    return result


def verify_owned_execution(
    info, run, registration, registration_sha256, registration_record, producer
):
    """Validate separately recorded direct-child observations; never attach a PID."""
    equal(info.get("schema"), EXECUTION_SCHEMA, "owned execution schema differs")
    equal(
        info.get("command"),
        expected_command(run, registration, registration_sha256),
        "owned command differs from sealed registration",
    )
    equal(info.get("output"), str(run), "owned output differs")
    equal(info.get("registration_sha256"), registration_sha256, "owned registration differs")
    require(
        type(info.get("pid")) is int and info["pid"] > 0, "positive observed owned PID required"
    )
    require(
        type(info.get("returncode")) is int
        and info["returncode"] == 0
        and info.get("child_alive_after_return") is False,
        "owned child has not successfully exited",
    )
    # info.started_at is recorded after Popen returns; a scheduled child may start
    # its provenance first. Neither value is an OS creation-time attestation.
    require(
        timestamp(registration_record["finished_at"])
        <= timestamp(info["started_at"])
        <= timestamp(producer["finished_at"])
        <= timestamp(info["finished_at"])
        <= datetime.now(UTC),
        "registration/owned child/producer chronology differs",
    )
    require(
        timestamp(registration_record["finished_at"])
        <= timestamp(producer["started_at"])
        <= timestamp(producer["finished_at"]),
        "producer chronology differs",
    )
    return dict(
        pid=info["pid"],
        returncode=0,
        child_alive_after_return=False,
        started_at=info["started_at"],
        finished_at=info["finished_at"],
        scope="Recorded direct Popen handle, not an OS re-query or descendant witness",
    )


def child_input_hashes(registration_record, registration, registration_sha256):
    """Exact input closure of the frozen new child's load_registration/prepare path."""
    result = {}
    for category in ("inputs", "outputs"):
        for role, item in registration_record[category].items():
            path = str(Path(item["path"]).resolve())
            require(category != "inputs" or role == path, "registration input role differs")
            require(
                path not in result or result[path] == item["sha256"],
                "conflicting registration input/output digest",
            )
            result[path] = item["sha256"]
    key = str(registration / "registration.provenance.json")
    require(key not in result, "registration seal cannot be its own dependency")
    result[key] = registration_sha256
    return result


def exact_child_inputs(producer, expected, evidence):
    require(set(producer["inputs"]) == set(expected), "child input membership differs")
    for path, digest in expected.items():
        item = producer["inputs"][path]
        require(
            str(m.local(item["path"])) == path and item["sha256"] == digest,
            "child input path/SHA differs",
        )
        evidence.add(path, digest)


def registered_edge(plan, legacy, parent_binding):
    """Finite I/O migration only; does not inspect this auditor's loaded modules."""
    equal(plan.get("schema"), "receipted_long_child_plan_v1", "wrong child plan schema")
    equal(plan.get("stage"), m.STAGE, "wrong child plan stage")
    edge = m.transition(parent_binding, plan["binding"], legacy)
    equal(plan["parent_binding"], parent_binding, "registered parent binding differs")
    equal(plan["transition"], edge, "registered migration edge differs")
    start, stop = plan["next_native_interval"]
    m.native_phase(start, stop)
    for key, value in dict(
        full_interval=[m.FIRST, m.FULL_STOP],
        output_interval=[m.FIRST, stop],
        expected_initializations=1,
        expected_external_resets=0,
        expected_previous_frame=start - 1,
        native_subsampling=False,
        serial_flow=True,
        same_original_accelerated_class=True,
        science_branch_change=False,
        processes_started=False,
        exclusive_execution_owner_is_external=True,
    ).items():
        equal(plan.get(key), value, "plan contract differs: " + key)
    return edge


def verify_receipt_usage(record, root, plan, evidence, validator_binding):
    """Read existing success receipts only. No new receipt or mask semantic decode."""
    path = root / "mask_validation_receipts.json"
    require(
        m.local(record["outputs"]["mask_validation_receipts"]["path"]) == path,
        "receipt usage role differs",
    )
    usage = m.read(path)
    equal(usage.get("schema"), "receipted_long_child_mask_usage_v1", "child receipt schema differs")
    require(usage.get("scientific_state_changed") is False, "receipt changed scientific state")
    cache = m.local(plan["receipt_cache_root"])
    equal(usage["cache_root"], str(cache), "receipt cache differs")
    files = usage["evidence_hashes"]
    for name, digest in files.items():
        item = record["inputs"].get(name)
        require(
            item is not None and item["path"] == name and item["sha256"] == digest,
            "receipt evidence missing from producer inputs",
        )
        evidence.add(name, digest)
    selected = plan["selected_chunk_records"]
    expected = {(r["start"], r["stop"], v) for r in selected for v in m.original.base.VIEWS}
    require(len(usage["uses"]) == len(expected), "receipt use count differs")
    by_range = {(r["start"], r["stop"]): r for r in selected}
    stage = m.local(plan["binding"]["inputs"]["input_stage_root"])
    seen, stable_uses = set(), []
    for use in usage["uses"]:
        scope = use["scope"]
        frames = scope["frame_range"]
        require(len(frames) == 2 and all(type(x) is int for x in frames), "receipt frames differ")
        key = (*frames, scope["view"])
        require(
            key in expected and key not in seen and type(use["cache_hit"]) is bool,
            "repeated/unselected/invalid receipt use",
        )
        seen.add(key)
        row = by_range[tuple(frames)]
        attempt = m.local(stage / row["attempt_path"])
        mask = row["output_hashes"][key[2] + "_masks"]
        for name, digest in (
            ("manifest", row["manifest_sha256"]),
            ("seal", row["provenance_sha256"]),
        ):
            filename = "manifest.json" if name == "manifest" else "staging.provenance.json"
            require(
                m.local(scope[name + "_path"]) == attempt / filename
                and scope[name + "_sha256"] == digest,
                "receipt attempt binding differs",
            )
        equal(scope["image_shape_hw"], [1200, 1200], "receipt native shape differs")
        equal(
            scope["detector_binding"],
            plan["binding"]["inputs"]["detector_binding"],
            "receipt detector binding differs",
        )
        require(
            use["npz_path"] == mask["path"] and scope["expected_npz_sha256"] == mask["sha256"],
            "receipt mask differs",
        )
        for name in ("receipt", "commit"):
            p = m.local(use[name + "_path"])
            require(
                p.is_relative_to(cache) and files.get(str(p)) == use[name + "_sha256"],
                "unbound receipt/commit path",
            )
        body, commit = m.read(use["receipt_path"]), m.read(use["commit_path"])
        equal(
            commit,
            dict(
                schema=receipts.SCHEMA,
                status="ok",
                receipt=Path(use["receipt_path"]).name,
                sha256=use["receipt_sha256"],
            ),
            "receipt success commit differs",
        )
        require(set(body) == {"request", "result"}, "receipt body keys differ")
        request = body["request"]
        equal(
            request,
            dict(
                schema=receipts.SCHEMA,
                npz_path=mask["path"],
                npz_bytes=Path(mask["path"]).stat().st_size,
                scope=scope,
                validator_binding=validator_binding,
            ),
            "receipt request/runtime differs",
        )
        # ReceiptStore uses ensure_ascii=False, sorted compact JSON, unlike checkpoint JSON.
        require(
            hashlib.sha256(receipts.canonical(request).encode()).hexdigest() == use["key"],
            "receipt request digest differs",
        )
        equal(
            body["result"],
            dict(
                status="ok",
                validator="chunk_masks.validate_packed_masks",
                frame_count=frames[1] - frames[0],
                validated_fields=sorted(receipts.chunk_masks.SCHEMA_DTYPES),
                every_decoded_frame_sha_verified=True,
            ),
            "receipt result differs",
        )
        for source, digest in validator_binding["source_sha256"].items():
            require(files.get(source) == digest, "unbound receipt validator source")
        stable_uses.append(
            {
                k: use[k]
                for k in (
                    "key",
                    "receipt_path",
                    "receipt_sha256",
                    "commit_path",
                    "commit_sha256",
                    "npz_path",
                    "scope",
                )
            }
        )
    require(seen == expected, "receipt coverage differs")
    return dict(
        uses=len(seen),
        hit_count=sum(u["cache_hit"] for u in usage["uses"]),
        usage_sha256=evidence.paths[str(path)],
        stable_uses=stable_uses,
    )


def validate_summary(summary, metadata, *, start, stop, restores, chunks):
    require(
        type(stop) is int and m.FIRST < stop <= m.FULL_STOP and (stop - m.FIRST) % m.CHUNK == 0,
        "invalid completed endpoint",
    )
    expected = dict(
        schema=m.SUMMARY_SCHEMA,
        technical_completion=True,
        full_sequence_complete=stop == m.FULL_STOP,
        prefix_backtrace_provisional=stop != m.FULL_STOP,
        registered_interval=[m.FIRST, m.FULL_STOP],
        processed_interval=[m.FIRST, stop],
        processed_total=stop - m.FIRST,
        phase_frames=[start, stop],
        phase_processed=stop - start,
        consumed_chunk_count=chunks,
        logical_initialization_count=1,
        checkpoint_restore_count=restores,
        external_pose_reset_count=0,
        restores_during_phase=0,
        temporal_subsampling=False,
        human_annotation_input=False,
        post_initialization_baseline_coordinates_input=False,
        all_39_checkpoint_arrays_verified=True,
        all_history_recurrences_checked_by_frozen_decoder=True,
        independent_ID_truth=False,
        independent_phase_audit_complete=False,
    )
    for key, value in expected.items():
        equal(summary.get(key), value, "new child summary differs: " + key)
    for key, value in dict(
        previous_frame=stop - 1,
        processed_frames=stop - m.FIRST,
        initialize_count=1,
        restore_count=restores,
    ).items():
        equal(metadata[key], value, "checkpoint lifecycle differs: " + key)
    equal(
        metadata["node_recovery_diagnostics"],
        summary["node_recovery_diagnostics"],
        "recovery diagnostics differ",
    )


def read_candidate(root, arrays):
    with np.load(root / "candidate.npz", allow_pickle=False) as archive:
        require(len(archive.files) == 21 == len(set(archive.files)), "candidate keys differ")
        candidate = {k: archive[k] for k in archive.files}
    traced = COMMON.traced_candidate(arrays)
    require(set(candidate) == set(traced), "candidate/backtrace schema differs")
    comparisons = {k: COMMON.same_array(candidate[k], traced[k], k) for k in candidate}
    return candidate, comparisons


def load_completed_phase(root, seal, binding, inputs, protocol, evidence, *, parent_metadata):
    """New-stage terminal-capable I/O adapter; no aliases of the old sealed schema."""
    require(seal.get("stage") == m.STAGE, "wrong completed receipt child stage")
    equal(seal["params"]["bindings"], binding, "completed binding differs")
    m.verified_outputs(root, seal, evidence)
    for role, filename in {
        **m.PARENT_ROLES,
        "state_import": "state_import.json",
        "mask_validation_receipts": "mask_validation_receipts.json",
    }.items():
        require(
            m.local(seal["outputs"][role]["path"]) == root / filename,
            "completed output role differs: " + role,
        )
    for name in ("metadata.json", "arrays.npz"):
        require(str(root / "checkpoint" / name) in seal["outputs"], "unsealed checkpoint payload")
    for source, digest in binding["code_sha256"].items():
        target = root / "code_snapshot" / Path(source).relative_to(m.REPO)
        require(
            str(target) in seal["outputs"]
            and m.local(seal["outputs"][str(target)]["path"]) == target,
            "unsealed completed source snapshot",
        )
        evidence.add(source, digest)
        evidence.add(target, digest)
    arrays, metadata, manifest = m.read_checkpoint(root / "checkpoint", expected_bindings=binding)
    summary = m.read(root / "summary.json")
    start, stop = m.FIRST + parent_metadata["processed_frames"], seal["params"]["through_frame"]
    m.native_phase(start, stop)
    validate_summary(
        summary,
        metadata,
        start=start,
        stop=stop,
        restores=parent_metadata["restore_count"] + 1,
        chunks=(stop - m.FIRST) // m.CHUNK,
    )
    require(
        len(arrays) == 39
        and arrays["state_pose"].shape == (6, 3, 3, 3)
        and manifest["payload"]["bytes"] == summary["checkpoint_payload_bytes"],
        "completed 39-state/beam/payload size differs",
    )
    equal(metadata["context"], parent_metadata["context"], "scientific checkpoint context changed")
    context = metadata["context"]
    equal(
        context["initial_pose"],
        COMMON.digest_array(inputs.initial_pose_xyz),
        "initial seed differs",
    )
    equal(
        context["cameras"],
        COMMON.normalize([dataclasses.asdict(c) for c in inputs.rig.cameras]),
        "actual calibrated cameras differ",
    )
    for key, source in (
        ("tracker_config", "tracker_config"),
        ("recovery_config", "node_recovery_config"),
        ("persistence_config", "persistence_config"),
    ):
        equal(context[key], protocol[source], "registered context differs: " + key)
    require(
        context["implementation"] == "AcceleratedContinuousPersistentNodeTracker"
        and context["views"] == list(inputs.rig.views)
        and context["world_units"] == "mm",
        "scientific class or coordinate system changed",
    )
    consumed = m.read(root / "consumed_chunks.json")
    equal(consumed, inputs.chunk_records, "completed consumed prefix differs")
    progress = [
        m._decode(line.encode()) for line in (root / "progress.jsonl").read_text().splitlines()
    ]
    counts = [row["processed"] for row in progress]
    require(
        counts
        and all(type(n) is int for n in counts)
        and counts == sorted(set(counts))
        and counts[-1] == stop - m.FIRST,
        "completed progress order differs",
    )
    for row in progress:
        equal(row["source_frame"], m.FIRST + row["processed"] - 1, "progress native differs")
        equal(
            row["phase_processed"], row["processed"] - (start - m.FIRST), "phase progress differs"
        )
        require(row["phase_processed"] > 0, "non-positive phase progress")
    require(
        {row["stop"] - m.FIRST for row in consumed if row["stop"] > start}.issubset(counts),
        "progress lacks new chunk endpoints",
    )
    candidate, comparisons = read_candidate(root, arrays)
    COMMON.same_array(
        candidate["frame_idx"], np.arange(m.FIRST, stop, dtype=np.int64), "native frames"
    )
    equal(
        sorted(candidate),
        summary["all_21_candidate_arrays_exact_backtrace"],
        "producer backtrace field list differs",
    )
    # Full frozen decode validates all 39 previous-buffer/schema/history fields, without restore.
    decoder = m.validation_tracker(inputs, protocol)
    decoder._decode_state(arrays, metadata)
    require(
        decoder.initialize_count == decoder.restore_count == decoder.processed_frames == 0,
        "CPU semantic audit initialized/restored a tracker",
    )
    return dict(
        arrays=arrays,
        metadata=metadata,
        summary=summary,
        consumed=consumed,
        candidate=candidate,
        comparisons=comparisons,
        manifest=manifest,
    )


def verify_state_import(proof, before, plan):
    arrays, metadata = before["arrays"], before["metadata"]
    equal(
        plan["parent_state_signatures"], m.array_signatures(arrays), "registered parent39 differs"
    )
    equal(plan["parent_context"], metadata["context"], "registered parent context differs")
    equal(
        plan["parent_history_rows"],
        int(arrays["history_offsets"][-1]),
        "parent history length differs",
    )
    equal(
        proof,
        dict(
            all_checkpoint_arrays_exact=sorted(arrays),
            array_signatures=m.array_signatures(arrays),
            context_unchanged=True,
            initialize_count=1,
            restore_count=metadata["restore_count"] + 1,
            previous_frame=metadata["previous_frame"],
            external_pose_reset_count=0,
        ),
        "restored import evidence differs from actual parent39",
    )
    equal(
        plan["expected_restores"],
        metadata["restore_count"] + 1,
        "registered restore increment differs",
    )


def verify_phase(
    run, registration, registration_sha256, owned_execution_path, evidence, *, receipt_store
):
    run, registration = m.local(run), m.local(registration)
    producer_path = run / "continuation.provenance.json"
    evidence.add(producer_path)
    producer = m.read(producer_path)
    require(
        producer.get("stage") == m.STAGE
        and producer.get("status") == "ok"
        and producer["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
        "completed successful receipt child required before array/input loading",
    )
    # Small completion/ownership records first; incomplete children never cause a
    # heavy registration dependency scan or candidate/checkpoint deserialization.
    reg_path = registration / "registration.provenance.json"
    evidence.add(reg_path, registration_sha256)
    reg = m.read(reg_path)
    require(
        reg.get("stage") == m.REGISTRATION_STAGE and reg.get("status") == "ok",
        "completed successful registration required",
    )
    owned_path = m.local(owned_execution_path)
    evidence.add(owned_path)
    owned = verify_owned_execution(
        m.read(owned_path), run, registration, registration_sha256, reg, producer
    )
    plan, reg = child.load_registration(registration, registration_sha256, evidence)
    equal(
        reg["params"].get("tracking_GPU_or_GT_performed"),
        False,
        "registration performed tracking/GPU/GT",
    )
    equal(str(run), plan["expected_output"], "completed run differs from registration")
    equal(
        producer["params"]["registration_sha256"],
        registration_sha256,
        "child registration SHA differs",
    )
    equal(
        producer["params"]["registration_path"],
        str(registration),
        "child registration path differs",
    )
    exact_child_inputs(
        producer, child_input_hashes(reg, registration, registration_sha256), evidence
    )
    legacy = m.legacy_binding(evidence)
    parent = m.local(plan["parent"])
    parent_path = parent / "continuation.provenance.json"
    evidence.add(parent_path, plan["parent_seal_sha256"])
    parent_record = m.read(parent_path)
    edge = registered_edge(plan, legacy, parent_record["params"]["bindings"])
    parent_stage = (
        m.STAGE if plan["parent_binding"]["schema"] == m.BINDING_SCHEMA else m.original.STAGE
    )
    require(
        parent_record.get("stage") == parent_stage
        and parent_record.get("status") == "ok"
        and parent_record["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
        "successful guarded version-specific parent required",
    )
    require(
        timestamp(parent_record["finished_at"]) <= timestamp(reg["started_at"]),
        "parent completion postdates registration",
    )
    equal(producer["params"]["bindings"], plan["binding"], "executed new binding differs")
    equal(producer["params"]["resume_phase"], str(parent), "executed parent differs")
    start, stop = plan["next_native_interval"]
    equal(parent_record["params"]["through_frame"], start, "parent endpoint differs")
    equal(producer["params"]["through_frame"], stop, "child endpoint differs")
    lineage = m.bind_parent_audit(
        plan["parent_audit"]["path"],
        plan["parent_audit"]["sha256"],
        parent,
        parent_record,
        evidence,
    )
    equal(lineage, plan["parent_audit"], "registered independent parent audit differs")
    # Use the existing cached loader once, and keep its receipts in this auditor's evidence.
    require(
        m.local(receipt_store.root) == m.local(plan["receipt_cache_root"]), "audit cache differs"
    )
    inputs, protocol = history_inputs(plan["binding"], stop, evidence, receipt_store)
    equal(inputs.chunk_records, plan["selected_chunk_records"], "registered chunks changed")
    if parent_record["params"]["bindings"]["schema"] == m.BINDING_SCHEMA:
        before = m.load_parent(
            parent, plan["parent_seal_sha256"], plan["parent_binding"], inputs, evidence
        )
        before["candidate"], before["comparisons"] = read_candidate(parent, before["arrays"])
    else:
        before = history.load_phase(parent, parent_record, inputs, evidence, protocol)
    for name, key in (
        ("manifest.json", "parent_checkpoint_sha256"),
        ("metadata.json", "parent_metadata_sha256"),
        ("arrays.npz", "parent_payload_sha256"),
    ):
        equal(
            evidence.paths[str(parent / "checkpoint" / name)],
            plan[key],
            "registered parent payload differs",
        )
    after = load_completed_phase(
        run,
        producer,
        plan["binding"],
        inputs,
        protocol,
        evidence,
        parent_metadata=before["metadata"],
    )
    verify_state_import(m.read(run / "state_import.json"), before, plan)
    equal(after["summary"]["verifier_transition"], edge, "producer migration report differs")
    equal(
        after["summary"]["history_prefix"],
        m.compare_prefix(before["arrays"], after["arrays"]),
        "producer history prefix report differs",
    )
    recurrence = history.suffix_history(before, after)
    counts = history.old.prior.support.direct_counts(after["candidate"])
    for key, reported in (
        ("direct_pectoral_fish_frames", "direct_pectoral_count"),
        ("accepted_pectoral_fish_frames", "accepted_pectoral_count"),
        ("review_fish_frames", "identity_review_count"),
        ("direct_pectoral_per_slot", "per_slot_direct_pectoral"),
        ("accepted_pectoral_per_slot", "per_slot_accepted_pectoral"),
    ):
        equal(counts[key], after["summary"][reported], "producer availability differs")
    receipt_before = verify_receipt_usage(
        reg, registration, plan, evidence, receipt_store.validator_binding
    )
    receipt_after = verify_receipt_usage(
        producer, run, plan, evidence, receipt_store.validator_binding
    )
    equal(
        receipt_before["stable_uses"],
        receipt_after["stable_uses"],
        "executed receipt proofs differ from registration",
    )
    m.bind_receipts(receipt_store, evidence)
    return dict(
        schema=SUMMARY_SCHEMA,
        audit_version=STAGE,
        status="ok",
        run=str(run),
        parent=str(parent),
        frame_range=[m.FIRST, stop],
        new_interval=[start, stop],
        candidate_sha256=evidence.paths[str(run / "candidate.npz")],
        all_39_checkpoint_arrays_verified=True,
        all_21_candidate_arrays_exact_backtrace=after["comparisons"],
        logical_initialization_count=1,
        checkpoint_restore_count=after["metadata"]["restore_count"],
        external_pose_reset_count=0,
        restores_during_phase=0,
        history=recurrence,
        counts=counts,
        full_sequence_complete=stop == m.FULL_STOP,
        prefix_backtrace_provisional=stop != m.FULL_STOP,
        independently_redecoded_raw_images=0,
        independent_geometric_reprojection=False,
        independent_ID_truth=False,
        independent_phase_audit_complete=True,
        producer_independent_phase_audit_complete=False,
        transition=edge,
        registration_sha256=registration_sha256,
        parent_audit=lineage,
        owned_execution={
            **owned,
            "path": str(owned_path),
            "sha256": evidence.paths[str(owned_path)],
        },
        receipt_registration={k: v for k, v in receipt_before.items() if k != "stable_uses"},
        receipt_execution={k: v for k, v in receipt_after.items() if k != "stable_uses"},
        scope="39 codec/semantic fields, 21 exact backtrace, all parent history and new suffix "
        "recurrences/exclusivity, current input/code/output/receipt SHA. No independent "
        "raw decode, flow rerun, ray/projection support or biological identity truth.",
    )


def history_inputs(binding, stop, evidence, receipt_store):
    # Frozen receipt loader wrapper has no tracking import-closure requirement.
    from receipted_long_phase_audit import verified_inputs

    return verified_inputs(binding, stop, evidence, receipt_store=receipt_store)


def audit_sources(evidence):
    for relative, digest in PINNED.items():
        evidence.add(m.REPO / relative, digest)
    result = {}
    for module in tuple(sys.modules.values()):
        origin = getattr(module, "__file__", None)
        if origin:
            path = Path(origin).resolve()
            if path.suffix == ".py" and any(
                path.is_relative_to(m.REPO / p)
                for p in ("labs/20260909_three_body", "src/medaka3d", "work/three_body")
            ):
                result[str(path)] = evidence.add(path)
    return result


def audit(args):
    output, evidence = child.new_output(args.output), m.Evidence()
    store = None
    with m.original.provenance(STAGE, output, name="audit") as record:
        try:
            # Eager import before capturing this independent operational source closure.
            import receipted_long_phase_audit  # noqa: F401

            code = audit_sources(evidence)
            child.snapshot(code, output, record)
            store = m.ReceiptStore(m.local(args.receipt_root))
            summary = verify_phase(
                args.run,
                args.registration,
                args.registration_sha256,
                args.owned_execution,
                evidence,
                receipt_store=store,
            )
            child.save_json(output / "summary.json", summary, record, "summary")
            equal(audit_sources(evidence), code, "late auditor workspace import/source change")
        finally:
            if store is not None:
                child.receipt_usage(output, store, record)
            child.close_evidence(record, evidence, store)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
        record.param("tracking_detector_GPU_or_GT_performed", False)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--owned-execution", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return audit(parser.parse_args(argv))


if __name__ == "__main__":
    main()

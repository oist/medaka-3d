"""Version-aware final-chain adapter; no CLI, process control or implicit decoding.

Use this module's load_supervisor_chain/load_completed_final/
load_native_sample_inputs entrypoints instead of changing frozen module globals.
Then use audit_long_final.verify_support_partitioned and
long_final_native_samples.read_native_samples on the returned objects normally.
The fixed 543 native samples are delegated to the frozen sample selector.
All entrypoints require completed full-range supervisors. Caller snapshots this
module/dependencies before use and seals evidence.final_guard() afterwards.
Support replay, 179 flow boundaries, 33 previous observations, global detector
assembly and a final report are still separate checks, not certified here.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime
from pathlib import Path

import audit_long_final as legacy
import long_final_native_samples as native
import mask_validation_receipts as receipts
import numpy as np
import receipted_long_handoff as handoff
import receipted_long_phase_operation as operation
import run_long_receipted_supervisor as current

REPO, WORK, FIRST, FINAL_STOP = legacy.REPO, current.WORK, legacy.FIRST, legacy.FINAL_STOP
COMMON, phase, require, equal = legacy.COMMON, legacy.phase, legacy.require, legacy.equal
NEW_STAGE = "long_remaining_receipted_supervisor_v1"
NEW_START = 76360
PINS = {
    **legacy.PINS,
    Path(
        legacy.__file__
    ).resolve(): "9140be0d5401228f38fefa04b1d4d4e68ea672ab8a0793130fbd62952c3e61eb",
    Path(
        native.__file__
    ).resolve(): "f9c82e6fc7150edd20f424dd9fa75fda2ef321ad10f41076744ef7a7689e7791",
    Path(
        current.__file__
    ).resolve(): "ddf95225f99e908e1b065915111766d069badf31f8b048414073c7d5855a9ee6",
    Path(
        handoff.__file__
    ).resolve(): "8d029ae65c85cdfc2696c680a6a5ff4637808af389296ca43730c3fde8f053b4",
    Path(
        operation.__file__
    ).resolve(): "fddad2527c4d423a96cf4ccd84367eb22a16e77f23f899f863bd0ed6f8649d97",
    (
        WORK / "receipted_long_phase_audit.py"
    ): "44438d6d36668e96cd311152df6b811a1c0a5fb8ec1277667d8fab9956414774",
    (
        WORK / "cached_long_tracking_inputs.py"
    ): "c25ab121b2ffdf80309ebe18e03821af559e5e3cbb900bd17edb8a4f4e0a62c1",
    Path(
        receipts.__file__
    ).resolve(): "82a284d9874608003529f9cc5ccb158f2f69adcb6da5faeda072daab11938df0",
}
ANCESTORS = (
    (WORK / "long_continuation_first1660_v1", 51460),
    (WORK / "long_continuation_through53120_v1", 53120),
    (WORK / "long_continuation_through59760_v1", 59760),
    (WORK / "long_continuation_through66400_v1", 66400),
)


def expected_new_phases():
    return legacy.expected_phases()[1:]


def read_record(root, name, stage, evidence):
    evidence.add(Path(root) / name)
    return legacy.read_record(root, name, stage, evidence)


def bound_sources(record, root, code, evidence):
    require(isinstance(code, dict) and code, "missing executed source map")
    for name, digest in code.items():
        source = Path(name).resolve(strict=True)
        require(source.is_relative_to(REPO), "source outside workspace")
        evidence.add(source, digest)
        legacy.input_binding(record, source, digest)
        snapshot = root / "code_snapshot" / source.relative_to(REPO)
        legacy.output_binding(record, snapshot, digest)
        evidence.add(snapshot, digest)


def check_new_header(summary, policy):
    expected = dict(
        schema=NEW_STAGE,
        status="complete",
        through=FINAL_STOP,
        bootstrap_completed=True,
        full_sequence_complete=True,
        prefix_backtrace_provisional=False,
        logical_initialization_count=1,
        checkpoint_restore_count=32,
        external_pose_reset_count=0,
        active_beam_count=6,
        stop_request=None,
        incomplete_phase=None,
        independent_ID_truth=False,
        next_phase_retried=False,
        detector_started=False,
        old_process_stopped=False,
    )
    for key, value in expected.items():
        equal(summary[key], value, "incomplete or inconsistent supervisor: " + key)
    require(len(summary["phases"]) == 28, "exactly 28 new completed phases required")
    for key, value in dict(
        schema=NEW_STAGE,
        initial_parent=str(handoff.CHILD),
        initial_stop=NEW_START,
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
        equal(policy[key], value, "new supervisor policy differs: " + key)
    equal(summary["intended_initial_parent"], str(handoff.CHILD), "handoff parent differs")


def verify_receipt_usage(record, root, *, label, plan, evidence, cache_root):
    """Recheck successful recorded uses; never invoke a validator/store or create receipts."""
    path = root / "mask_validation_receipts.json"
    usage = legacy.json_output(record, "mask_validation_receipts", path)
    equal(usage["schema"], "long_phase_mask_validation_usage_v1", "receipt usage schema differs")
    require(
        usage["phase"] == label
        and usage["verification_returned"] is True
        and usage["scientific_plan_or_child_binding_changed"] is False
        and Path(usage["cache_root"]).resolve() == cache_root,
        "wrong successful receipt usage",
    )
    files = usage["evidence_hashes"]
    for name, digest in files.items():
        legacy.input_binding(record, Path(name).resolve(), digest)
        evidence.add(name, digest)
    expected_validator = dict(
        source_sha256={
            str(Path(p).resolve()): evidence.paths[str(Path(p).resolve())]
            for p in (
                receipts.__file__,
                receipts.chunk_masks.__file__,
                receipts.prepare_sequence.__file__,
            )
        },
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        numpy=np.__version__,
        schema_dtypes={k: v.str for k, v in native.SCHEMA_DTYPES.items()},
        mask_bitorder="little",
    )
    selected = plan["full_selected_chunk_records"]
    expected = {(row["start"], row["stop"], view) for row in selected for view in native.VIEWS}
    seen = set()
    require(len(usage["uses"]) == len(expected), "receipt uses differ from selected view/chunk set")
    stage = Path(plan["frozen_input_and_code_binding"]["inputs"]["input_stage_root"])
    for use in usage["uses"]:
        require(type(use["cache_hit"]) is bool, "receipt cache flag must be boolean")
        scope = use["scope"]
        key = (*scope["frame_range"], scope["view"])
        require(key in expected and key not in seen, "missing/repeated/unselected receipt scope")
        seen.add(key)
        row = selected[(key[0] - FIRST) // native.CHUNK]
        attempt = stage / row["attempt_path"]
        require(
            Path(scope["manifest_path"]).resolve() == attempt / "manifest.json"
            and scope["manifest_sha256"] == row["manifest_sha256"]
            and Path(scope["seal_path"]).resolve() == attempt / "staging.provenance.json"
            and scope["seal_sha256"] == row["provenance_sha256"]
            and scope["image_shape_hw"] == [1200, 1200],
            "receipt scope differs from consumed chunk",
        )
        equal(
            scope["detector_binding"],
            plan["frozen_input_and_code_binding"]["inputs"]["detector_binding"],
            "receipt detector binding differs",
        )
        mask = row["output_hashes"][key[2] + "_masks"]
        require(
            use["npz_path"] == mask["path"] and scope["expected_npz_sha256"] == mask["sha256"],
            "receipt packed input differs",
        )
        for k in ("receipt", "commit"):
            p = Path(use[k + "_path"]).resolve(strict=True)
            require(
                p.is_relative_to(cache_root) and files.get(str(p)) == use[k + "_sha256"],
                "receipt file is not separately sealed",
            )
        body = receipts.read(use["receipt_path"])
        commit = receipts.read(use["commit_path"])
        require(
            commit
            == dict(
                schema=receipts.SCHEMA,
                status="ok",
                receipt=Path(use["receipt_path"]).name,
                sha256=use["receipt_sha256"],
            ),
            "invalid receipt success commit",
        )
        request = body["request"]
        equal(
            request["validator_binding"],
            expected_validator,
            "receipt validator/runtime contract differs",
        )
        require(
            request["schema"] == receipts.SCHEMA
            and request["npz_path"] == mask["path"]
            and request["npz_bytes"] == Path(mask["path"]).stat().st_size
            and receipts.canonical(request["scope"]) == receipts.canonical(scope),
            "receipt request differs from declared use",
        )
        # Receipt key authenticates its exact canonical request only as a local hash.
        require(
            hashlib.sha256(receipts.canonical(request).encode()).hexdigest() == use["key"],
            "receipt request key differs",
        )
        equal(
            body["result"],
            dict(
                status="ok",
                validator="chunk_masks.validate_packed_masks",
                frame_count=key[1] - key[0],
                validated_fields=sorted(native.SCHEMA_DTYPES),
                every_decoded_frame_sha_verified=True,
            ),
            "receipt validation result differs",
        )
        for source, digest in request["validator_binding"]["source_sha256"].items():
            require(files.get(source) == digest, "receipt validator source is not bound")
            evidence.add(source, digest)
    require(seen == expected, "receipt coverage differs")
    return dict(
        uses=len(seen),
        hit_count=sum(u["cache_hit"] for u in usage["uses"]),
        usage_sha256=evidence.paths[str(path)],
    )


def check_phase_edge(
    root,
    parent,
    previous_record,
    audit_root,
    audit_stage,
    audit_name,
    binding,
    start,
    stop,
    restores,
    supervisor,
    evidence,
    *,
    receipted,
    code=None,
    cache_root=None,
):
    run, reg = root / "tracking", root / "registration"
    registration = read_record(
        reg,
        "registration.provenance.json",
        operation.REGISTRATION_STAGE if receipted else "register_long_supervised_phase",
        evidence,
    )
    op = read_record(
        root,
        "operation.provenance.json",
        operation.OPERATION_STAGE if receipted else "long_supervised_phase_operation",
        evidence,
    )
    child = read_record(
        run, "continuation.provenance.json", "three_body_long_continuation", evidence
    )
    plan = legacy.json_output(registration, "plan", reg / "plan.json")
    config = legacy.json_output(registration, "execution", reg / "execution.json")
    execution = legacy.json_output(op, "child_execution", root / "child_execution.json")
    require(
        op["params"]["tracking_child_exit_code"] == 0
        and execution["returncode"] == 0
        and execution["child_alive_after_return"] is False,
        "completed zero-exit child required",
    )
    require(
        datetime.fromisoformat(registration["finished_at"])
        <= datetime.fromisoformat(execution["started_at"])
        <= datetime.fromisoformat(child["finished_at"])
        <= datetime.fromisoformat(execution["finished_at"])
        <= datetime.fromisoformat(op["finished_at"])
        and datetime.fromisoformat(registration["finished_at"])
        <= datetime.fromisoformat(child["started_at"]),
        "phase chronology differs",
    )
    for key, value in dict(
        parent_phase=str(parent),
        expected_output=str(run),
        next_native_interval=[start, stop],
        expected_restores=restores,
        expected_initializations=1,
        expected_external_resets=0,
        parent_active_beam=6,
    ).items():
        equal(plan[key], value, "phase plan differs: " + key)
    equal(plan["frozen_input_and_code_binding"], binding, "registered binding changed")
    equal(child["params"]["bindings"], binding, "child binding changed")
    require(
        Path(child["params"]["resume_phase"]).resolve() == parent
        and child["params"]["through_frame"] == stop,
        "child parent/endpoint differs",
    )
    equal(
        plan["parent_seal_sha256"],
        evidence.paths[str(parent / "continuation.provenance.json")],
        "registered parent seal changed",
    )
    lineage = dict(
        root=str(audit_root),
        seal_sha256=evidence.paths[str(audit_root / audit_name)],
        summary_sha256=evidence.paths[
            str(
                audit_root
                / ("summary.json" if audit_name == "audit.provenance.json" else "audit.json")
            )
        ],
        parent_candidate_sha256=previous_record["outputs"]["candidate"]["sha256"],
    )
    if receipted:
        lineage["stage"] = audit_stage
    equal(config["independent_parent_audit"], lineage, "parent audit lineage differs")
    command = current.operation.command_for(parent, stop, run, binding)
    equal(config["command"], command, "frozen child command differs")
    for name in (
        audit_name,
        "summary.json" if audit_name == "audit.provenance.json" else "audit.json",
    ):
        legacy.input_binding(
            registration, audit_root / name, evidence.paths[str(audit_root / name)]
        )
    legacy.input_binding(
        registration,
        parent / "continuation.provenance.json",
        evidence.paths[str(parent / "continuation.provenance.json")],
    )
    legacy.input_binding(
        op,
        run / "continuation.provenance.json",
        evidence.paths[str(run / "continuation.provenance.json")],
    )
    audit = legacy.json_output(op, "audit", root / "audit.json")
    legacy.check_audit(
        audit, run, child["outputs"]["candidate"]["sha256"], stop=stop, restores=restores
    )
    equal(audit["new_interval"], [start, stop], "audited interval differs")
    if receipted:
        equal(config["source_hashes"], code, "phase operational source map differs")
        bound_sources(registration, reg, code, evidence)
        for item in registration["inputs"].values():
            legacy.input_binding(op, Path(item["path"]).resolve(), item["sha256"])
        verify_receipt_usage(
            registration, reg, label="prepare", plan=plan, evidence=evidence, cache_root=cache_root
        )
        verify_receipt_usage(
            op, root, label="verify", plan=plan, evidence=evidence, cache_root=cache_root
        )
        for relative in current.PHASE_ARTIFACTS:
            path = root / relative
            legacy.output_binding(supervisor, path, evidence.paths[str(path)])
    return (
        dict(
            parent=str(parent),
            run=str(run),
            interval=[start, stop],
            restores=restores,
            candidate_sha256=audit["candidate_sha256"],
            operation_sha256=evidence.paths[str(root / "operation.provenance.json")],
            operation_stage=op["stage"],
        ),
        child,
        audit,
    )


def verify_ancestors(binding, evidence):
    rows, previous, start = [], None, FIRST
    for restores, (root, stop) in enumerate(ANCESTORS):
        record = read_record(
            root, "continuation.provenance.json", "three_body_long_continuation", evidence
        )
        equal(record["params"]["bindings"], binding, "seed ancestor binding differs")
        summary = legacy.json_output(record, "summary", root / "summary.json")
        phase.validate_summary(
            summary,
            stop=stop,
            phase_start=start,
            restores=restores,
            chunks=(stop - FIRST) // native.CHUNK,
        )
        require(record["params"]["through_frame"] == stop, "ancestor endpoint differs")
        resume = record["params"]["resume_phase"]
        require(
            resume is None if previous is None else Path(resume).resolve() == previous,
            "ancestor resume path differs",
        )
        if previous is not None:
            for name in phase.old.prior.PARENT_FILES:
                legacy.input_binding(record, previous / name, evidence.paths[str(previous / name)])
        rows.append(
            dict(
                run=str(root),
                interval=[start, stop],
                restores=restores,
                seal_sha256=evidence.paths[str(root / "continuation.provenance.json")],
            )
        )
        previous, start = root, stop
    return rows


def load_supervisor_chain(root, evidence):
    """Accept only the complete legacy schedule or this one registered mixed schedule."""
    root = Path(root).resolve(strict=True)
    path = root / "supervisor.provenance.json"
    evidence.add(path)
    header = COMMON.read_json(path)
    require(header.get("status") == "ok", "completed successful supervisor required")
    if header.get("stage") == "long_remaining_supervisor":
        result = legacy.load_supervisor_chain(root, evidence)
        result["chain_version"] = "legacy_v1"
        return result
    require(header.get("stage") == NEW_STAGE, "unsupported supervisor version")
    # Check completion before reading heavyweight dependency payloads or any NPZ arrays.
    summary = legacy.json_output(header, "summary", root / "summary.json")
    policy = legacy.json_output(header, "plan", root / "supervisor_plan.json")
    check_new_header(summary, policy)
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
        zip(expected_new_phases(), summary["phases"], strict=True)
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
        summary["final_run"] == str(parent) and chain[-1]["restores"] == 32,
        "terminal run/restore differs",
    )
    require(len(ancestors) + len(chain) == 33, "33 tracking processes required")
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
        chain_version="legacy_then_receipted_v1",
        handoff_lineage=transferred["lineage"],
        tracking_processes=33,
        checkpoint_restores=32,
        sampled_frames=native.fixed_sample_frames(),
        final_audit_complete=False,
    )


# Explicit frozen orchestration copies; entrypoints call this version-aware chain.
def load_completed_final(supervisor_root, evidence):
    """Load final arrays once after sealed chain checks; not a complete final audit."""
    chain = load_supervisor_chain(supervisor_root, evidence)
    root, record = chain["final_run"], chain["final_record"]
    summary = legacy.json_output(record, "summary", root / "summary.json")
    phase.validate_summary(summary, stop=FINAL_STOP, phase_start=345280, restores=32, chunks=180)
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
            "fixed native samples/boundary flow",
            "global detector completion/assembly",
            "final provenance/report",
        ],
    )


def load_native_sample_inputs(supervisor_root, evidence):
    """Verify completed chain and consumed attempts; no raw decode or candidate load."""
    final = legacy

    evidence.add(native.PLAN, native.PLAN_SHA)
    evidence.add(Path(__file__))
    chain = load_supervisor_chain(supervisor_root, evidence)
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

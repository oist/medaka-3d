"""Registered receipt child runner, retaining the frozen original tracking loop.

Registration and codec preparation are CPU-only. Execution requires an explicitly
sealed registration; the caller owns process scheduling/exclusion. This entry
point neither controls the live supervisor nor performs automatic handoff.
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import receipted_long_runner_migration as m

old = m.original


def save_json(path, value, record, role):
    with path.open("xb") as stream:
        stream.write(m._canonical(value) + b"\n")
    record.output(role, path, sha256=m.sha(path))


def snapshot(code, output, record):
    for source, expected in code.items():
        target = output / "code_snapshot" / Path(source).relative_to(m.REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        m.require(m.sha(target) == expected, "source changed while snapshotting")
        record.output(str(target), target, sha256=expected)


def close_evidence(record, evidence, store):
    # Preserve already known inputs even if the final receipt guard fails.
    for path, expected in evidence.paths.items():
        record.input(path, path, sha256=expected)
    if store is not None:
        m.bind_receipts(store, evidence)
    for path, expected in evidence.paths.items():
        record.input(path, path, sha256=expected)
    evidence.final_guard()
    for item in record.outputs.values():
        m.require(m.sha(item["path"]) == item["sha256"], "saved output changed before seal")


def receipt_usage(output, store, record):
    save_json(
        output / "mask_validation_receipts.json",
        dict(
            schema="receipted_long_child_mask_usage_v1",
            cache_root=str(store.root),
            uses=store.uses,
            evidence_hashes=store.evidence_hashes,
            scientific_state_changed=False,
        ),
        record,
        "mask_validation_receipts",
    )


def prepare(args, evidence, *, store_ready=None, existing_output=None):
    """Build an exact repeatable plan; no CUDA query, tracker restore or temporal step."""
    legacy = m.legacy_binding(evidence)
    code = m.execution_sources(legacy, evidence)
    binding = m.execution_binding(legacy, code)
    parent_root = m.local(args.parent)
    seal_path = parent_root / "continuation.provenance.json"
    evidence.add(seal_path, args.parent_seal_sha256)
    parent_seal = m.read(seal_path)
    parent_binding = parent_seal["params"]["bindings"]
    edge = m.transition(parent_binding, binding, legacy)
    start = parent_seal["params"]["through_frame"]
    m.native_phase(start, args.through_frame)
    inputs, store = m.load_inputs(
        binding, args.through_frame, args.receipt_root, evidence, store_ready=store_ready
    )
    protocol = m.read(old.PROTOCOL_PATH)
    m.equal(protocol["source_frame_range"], [m.FIRST, m.FULL_STOP], "registered range differs")
    m.require(
        m.sha(old.PROTOCOL_PATH) == binding["tracking_protocol_sha256"], "tracking protocol changed"
    )
    parent = m.load_parent(parent_root, args.parent_seal_sha256, parent_binding, inputs, evidence)
    audit = m.bind_parent_audit(
        args.parent_audit_seal, args.parent_audit_sha256, parent_root, parent["seal"], evidence
    )
    validator = m.validation_tracker(inputs, protocol)
    validator._decode_state(parent["arrays"], parent["metadata"])
    m.require(
        validator.initialize_count == validator.restore_count == validator.processed_frames == 0,
        "registration decoder must not initialize or restore",
    )
    m.require(parent["metadata"]["previous_frame"] == start - 1, "parent native boundary differs")
    output = m.local(args.run_output, exists=False)
    if existing_output is None:
        m.require(not output.exists(), "new child output required")
    else:
        m.require(
            output == existing_output and output.is_dir() and not any(output.iterdir()),
            "only the newly created empty execution directory is allowed",
        )
    plan = dict(
        schema="receipted_long_child_plan_v1",
        stage=m.STAGE,
        parent=str(parent_root),
        parent_seal_sha256=args.parent_seal_sha256,
        parent_audit=audit,
        parent_binding=parent_binding,
        binding=binding,
        transition=edge,
        parent_checkpoint_sha256=evidence.paths[str(parent_root / "checkpoint/manifest.json")],
        parent_metadata_sha256=evidence.paths[str(parent_root / "checkpoint/metadata.json")],
        parent_payload_sha256=evidence.paths[str(parent_root / "checkpoint/arrays.npz")],
        parent_context=parent["metadata"]["context"],
        parent_state_signatures=m.array_signatures(parent["arrays"]),
        parent_history_rows=int(parent["arrays"]["history_offsets"][-1]),
        expected_initializations=1,
        expected_restores=parent["metadata"]["restore_count"] + 1,
        expected_external_resets=0,
        expected_previous_frame=start - 1,
        next_native_interval=[start, args.through_frame],
        full_interval=[m.FIRST, m.FULL_STOP],
        output_interval=[m.FIRST, args.through_frame],
        expected_output=str(output),
        selected_chunk_records=inputs.chunk_records,
        receipt_cache_root=str(store.root),
        native_subsampling=False,
        serial_flow=True,
        same_original_accelerated_class=True,
        science_branch_change=False,
        processes_started=False,
        exclusive_execution_owner_is_external=True,
    )
    m.equal(
        m.execution_sources(legacy, evidence), code, "source closure changed during preparation"
    )
    return plan, SimpleNamespace(
        inputs=inputs, store=store, parent=parent, protocol=protocol, legacy=legacy, code=code
    )


def new_output(path):
    output = m.local(path, exists=False)
    m.require(not output.exists(), "new exclusive output directory required")
    output.mkdir(parents=True, exist_ok=False)
    return output


def register(args):
    output, evidence, store = new_output(args.output), m.Evidence(), None

    def retain(value):
        nonlocal store
        store = value

    with old.provenance(m.REGISTRATION_STAGE, output, name="registration") as record:
        try:
            plan, context = prepare(args, evidence, store_ready=retain)
            snapshot(context.code, output, record)
            save_json(output / "plan.json", plan, record, "plan")
        finally:
            if store is not None:
                receipt_usage(output, store, record)
            close_evidence(record, evidence, store)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
        record.param("tracking_GPU_or_GT_performed", False)
    return plan


def load_registration(root, expected_sha, evidence):
    root = m.local(root)
    path = root / "registration.provenance.json"
    evidence.add(path, expected_sha)
    seal = m.read(path)
    m.require(seal.get("stage") == m.REGISTRATION_STAGE, "wrong registration stage")
    m.verified_outputs(root, seal, evidence)
    m.require(
        m.local(seal["outputs"]["plan"]["path"]) == root / "plan.json",
        "registration plan role differs",
    )
    for role, item in seal["inputs"].items():
        m.require(str(m.local(item["path"])) == role, "registration input role/path differs")
        evidence.add(role, item["sha256"])
    m.require(
        datetime.fromisoformat(seal["finished_at"]).timestamp() <= time.time(),
        "registration has a future completion time",
    )
    return m.read(root / "plan.json"), seal


def configure_runtime(protocol, binding):
    """Execution only: preserve every original runtime flag; no new precision policy."""
    old.torch.set_num_threads(protocol["torch_cpu_threads"])
    m.require(old.torch.cuda.is_available(), "registered continuation requires CUDA")
    actual = dict(
        package_versions={
            "numpy": np.__version__,
            "scipy": old.scipy.__version__,
            "opencv": old.cv2.__version__,
            "torch": str(old.torch.__version__),
            "cuda": old.torch.version.cuda,
            "cuda_device_name": old.torch.cuda.get_device_name(0),
        },
        torch_cpu_threads=old.torch.get_num_threads(),
        opencv_threads=old.cv2.getNumThreads(),
        torch_deterministic_algorithms=old.torch.are_deterministic_algorithms_enabled(),
        cuda_matmul_tf32=old.torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=old.torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=old.torch.backends.cudnn.benchmark,
    )
    for key, value in actual.items():
        m.equal(value, binding[key], "runtime changed: " + key)
    m.equal(actual["package_versions"], protocol["package_versions"], "protocol runtime differs")
    m.equal(
        dataclasses.asdict(old.FlowConfig()), protocol["flow_config"], "serial flow config differs"
    )


def execute(args):
    wall = time.perf_counter()
    evidence, store = m.Evidence(), None
    registered, registration_seal = load_registration(
        args.registration, args.registration_sha256, evidence
    )
    output = m.local(args.output, exists=False)
    m.require(str(output) == registered["expected_output"], "execution output differs from plan")
    output = new_output(output)

    def retain(value):
        nonlocal store
        store = value

    with old.provenance(m.STAGE, output, name="continuation") as record:
        record.param("registration_sha256", args.registration_sha256)
        record.param("registration_path", str(m.local(args.registration)))
        try:
            audit = registered["parent_audit"]
            planned_args = SimpleNamespace(
                parent=Path(registered["parent"]),
                parent_seal_sha256=registered["parent_seal_sha256"],
                parent_audit_seal=Path(audit["path"]),
                parent_audit_sha256=audit["sha256"],
                through_frame=registered["next_native_interval"][1],
                receipt_root=Path(registered["receipt_cache_root"]),
                run_output=output,
            )
            plan, context = prepare(
                planned_args, evidence, store_ready=retain, existing_output=output
            )
            m.equal(plan, registered, "registration no longer describes exact executable inputs")
            m.require(
                datetime.fromisoformat(registration_seal["finished_at"])
                <= datetime.fromisoformat(record.started_at),
                "registration postdates execution",
            )
            binding, parent, inputs = plan["binding"], context.parent, context.inputs
            record.param("bindings", binding)
            record.param("through_frame", plan["next_native_interval"][1])
            record.param("resume_phase", plan["parent"])
            snapshot(context.code, output, record)
            save_json(
                output / "consumed_chunks.json", inputs.chunk_records, record, "consumed_chunks"
            )
            configure_runtime(context.protocol, binding)
            tracker = old.AcceleratedContinuousPersistentNodeTracker(
                inputs.rig,
                inputs.initial_pose_xyz.copy(),
                old.TrackerConfig(**context.protocol["tracker_config"]),
                recovery_config=old.NodeRecoveryConfig(**context.protocol["node_recovery_config"]),
                persistence_config=old.PersistentNodeConfig(
                    **context.protocol["persistence_config"]
                ),
            )
            imported = m.restore_parent(tracker, parent["arrays"], parent["metadata"])
            save_json(output / "state_import.json", imported, record, "state_import")
            start, stop = plan["next_native_interval"]
            m.require(
                tracker.restore_count == plan["expected_restores"]
                and tracker.previous_frame == start - 1
                and tracker.processed_frames == start - m.FIRST,
                "restored native lifecycle differs",
            )
            preparation_seconds = time.perf_counter() - wall
            reader_seconds, tracking_seconds = old._advance_chunks(
                tracker, inputs, start, output / "progress.jsonl"
            )
            m.require(
                tracker.processed_frames == stop - m.FIRST
                and tracker.initialize_count == 1
                and tracker.restore_count == plan["expected_restores"],
                "in-phase reset/restore",
            )
            checkpoint_start = time.perf_counter()
            arrays, metadata = tracker.export_state()  # Frozen full-history semantic decoder.
            prefix = m.compare_prefix(parent["arrays"], arrays)
            m.checkpoint_roundtrip(output / "checkpoint", arrays, metadata, binding)
            checkpoint_seconds = time.perf_counter() - checkpoint_start
            for filename in ("manifest.json", "metadata.json", "arrays.npz"):
                path = output / "checkpoint" / filename
                record.output(str(path), path, sha256=m.sha(path))
            record.output(
                "checkpoint_manifest",
                output / "checkpoint/manifest.json",
                sha256=m.sha(output / "checkpoint/manifest.json"),
            )
            result = tracker.finish()
            exact = m.compare_backtrace(arrays, result)
            nodes = old.CachedNodes(
                inputs.input_root, inputs.protocol["experiment"], inputs.frame_idx
            )
            old.base.validate_result(result, inputs.frame_idx, nodes)
            old.validate_arm_arrays(result, inputs.frame_idx)
            np.savez_compressed(output / "candidate.npz", **result)
            record.output(
                "candidate", output / "candidate.npz", sha256=m.sha(output / "candidate.npz")
            )
            record.output(
                "progress", output / "progress.jsonl", sha256=m.sha(output / "progress.jsonl")
            )
            summary = phase_summary(
                tracker,
                arrays,
                result,
                plan,
                prefix,
                exact,
                preparation_seconds,
                reader_seconds,
                tracking_seconds,
                checkpoint_seconds,
                output,
            )
            save_json(output / "summary.json", summary, record, "summary")
            m.equal(
                m.execution_sources(context.legacy, evidence), context.code, "late source changes"
            )
        finally:
            if store is not None:
                receipt_usage(output, store, record)
            close_evidence(record, evidence, store)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
    return summary


def phase_summary(
    tracker, arrays, result, plan, prefix, exact, preparation, reader, tracking, checkpoint, output
):
    start, stop = plan["next_native_interval"]
    return dict(
        schema=m.SUMMARY_SCHEMA,
        technical_completion=True,
        full_sequence_complete=stop == m.FULL_STOP,
        prefix_backtrace_provisional=stop != m.FULL_STOP,
        registered_interval=[m.FIRST, m.FULL_STOP],
        processed_interval=[m.FIRST, stop],
        processed_total=tracker.processed_frames,
        phase_frames=[start, stop],
        phase_processed=stop - start,
        consumed_chunk_count=len(plan["selected_chunk_records"]),
        logical_initialization_count=1,
        checkpoint_restore_count=tracker.restore_count,
        external_pose_reset_count=0,
        restores_during_phase=0,
        temporal_subsampling=False,
        human_annotation_input=False,
        post_initialization_baseline_coordinates_input=False,
        preparation_seconds=preparation,
        reader_setup_seconds=reader,
        tracking_seconds=tracking,
        checkpoint_seconds=checkpoint,
        checkpoint_payload_bytes=(output / "checkpoint/arrays.npz").stat().st_size,
        retained_history_rows=int(arrays["history_offsets"][-1]),
        active_beam_count=len(arrays["state_pose"]),
        node_recovery_diagnostics=tracker.node_recovery_diagnostics,
        direct_pectoral_count=int(result["observed"][:, :, 1].sum()),
        accepted_pectoral_count=int(np.isfinite(result["accepted_xyz"]).all(-1).sum()),
        identity_review_count=int(result["identity_review"].sum()),
        per_slot_direct_pectoral=result["observed"][:, :, 1].sum(0).tolist(),
        per_slot_accepted_pectoral=np.isfinite(result["accepted_xyz"]).all(-1).sum(0).tolist(),
        all_39_checkpoint_arrays_verified=len(arrays) == 39,
        all_21_candidate_arrays_exact_backtrace=exact,
        history_prefix=prefix,
        all_history_recurrences_checked_by_frozen_decoder=True,
        verifier_transition=plan["transition"],
        independent_ID_truth=False,
        raw_scope="Unchanged SourceSequenceReader consumes only this phase's native suffix.",
        independent_phase_audit_complete=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register-only", action="store_true")
    mode.add_argument("--registration", type=Path)
    parser.add_argument("--registration-sha256")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--parent-seal-sha256")
    parser.add_argument("--parent-audit-seal", type=Path)
    parser.add_argument("--parent-audit-sha256")
    parser.add_argument("--through-frame", type=int)
    parser.add_argument("--receipt-root", type=Path)
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.register_only:
        for name in (
            "parent",
            "parent_seal_sha256",
            "parent_audit_seal",
            "parent_audit_sha256",
            "through_frame",
            "receipt_root",
            "run_output",
        ):
            if getattr(args, name) is None:
                parser.error("registration requires --" + name.replace("_", "-"))
        return register(args)
    if args.registration_sha256 is None:
        parser.error("execution requires --registration-sha256")
    return execute(args)


if __name__ == "__main__":
    main()

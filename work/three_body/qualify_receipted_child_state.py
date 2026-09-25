"""CPU qualification of registered receipt-child checkpoint transport and restore.

The caller pins a completed registration. Two new checkpoint copies demonstrate
unchanged metadata transport and exactly one logical restore of the original
39-array state. No tracking step, raw decode, flow, GPU query or ownership action
is performed. This is CPU state evidence, not a completed tracking phase/audit.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import run_receipted_long_continuation as runner

m = runner.m
WORK = Path(__file__).resolve().parent
STAGE = "qualify_receipted_child_state_v1"
PARENT_RESTORES = 10
REGISTRATION_ROOT = WORK / "receipted_child_cpu520_registration_v1"
REGISTRATION_SHA = "da3330e1c992c07ec0dd05b543038e735c1663259ef49154e0f6d432fdd412fa"
PINS = {
    Path(m.__file__).resolve(): "96d8d0e9aa1fb364cd3af06ae5796c252b1844d72942f0cf453d41a30105975f",
    Path(
        runner.__file__
    ).resolve(): "776725de736b35715bf2d5c53226a0abc48e8f5ecff94516f9e9fdb8ffaac364",
}


def valid_sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_plan(plan, legacy):
    """Only the registered first I/O migration, without invoking runner preparation."""
    m.equal(
        plan["binding"],
        m.execution_binding(legacy, plan["binding"]["code_sha256"]),
        "registration changed the finite I/O binding",
    )
    m.equal(plan["parent_binding"], legacy, "CPU probe requires the unchanged legacy parent")
    m.equal(
        plan["transition"],
        m.transition(legacy, plan["binding"], legacy),
        "registered transition differs",
    )
    expected = dict(
        schema="receipted_long_child_plan_v1",
        stage=m.STAGE,
        expected_initializations=1,
        expected_restores=PARENT_RESTORES + 1,
        expected_external_resets=0,
        full_interval=[m.FIRST, m.FULL_STOP],
        native_subsampling=False,
        serial_flow=True,
        same_original_accelerated_class=True,
        science_branch_change=False,
        processes_started=False,
        exclusive_execution_owner_is_external=True,
    )
    m.equal(
        {key: plan.get(key) for key in expected}, expected, "registered CPU probe scope differs"
    )
    start, stop = plan["next_native_interval"]
    m.native_phase(start, stop)
    m.equal(plan["expected_previous_frame"], start - 1, "registered parent boundary differs")
    m.equal(plan["output_interval"], [m.FIRST, stop], "registered output interval differs")


def bind_probe_sources(binding, evidence):
    """Bind imported workspace sources without the execution-only 40-source scan."""
    own = Path(__file__).resolve()
    allowed = {Path(path).resolve() for path in binding["code_sha256"]} | {own}
    sources = {str(own): evidence.add(own)}
    for module in tuple(sys.modules.values()):
        origin = getattr(module, "__file__", None)
        if origin:
            path = Path(origin).resolve()
            workspace_code = path.suffix == ".py" and (
                path.is_relative_to(m.LAB)
                or path.is_relative_to(m.REPO / "src/medaka3d")
                or path.is_relative_to(WORK)
            )
            if workspace_code:
                m.require(path in allowed, "unregistered imported probe source: " + str(path))
                expected = binding["code_sha256"].get(str(path))
                sources[str(path)] = evidence.add(path, expected)
    for path, expected in PINS.items():
        evidence.add(path, expected)
        m.require(
            binding["code_sha256"].get(str(path)) == expected,
            "registration used another migration helper or runner",
        )
    return sources


def load_inputs(registration, registration_sha256, evidence, *, store_ready=None):
    plan, seal = runner.load_registration(registration, registration_sha256, evidence)
    m.require(
        seal["params"].get("tracking_GPU_or_GT_performed") is False,
        "completed CPU-only registration required",
    )
    legacy = m.legacy_binding(evidence)
    validate_plan(plan, legacy)
    root = m.local(registration)
    for source, expected in plan["binding"]["code_sha256"].items():
        path = m.local(source)
        item = seal["inputs"].get(str(path))
        m.require(
            item is not None and m.local(item["path"]) == path and item["sha256"] == expected,
            "registered source input differs",
        )
        evidence.add(path, expected)
        snapshot = root / "code_snapshot" / path.relative_to(m.REPO)
        item = seal["outputs"].get(str(snapshot))
        m.require(
            item is not None and m.local(item["path"]) == snapshot and item["sha256"] == expected,
            "registered source snapshot differs",
        )
        evidence.add(snapshot, expected)
    code = bind_probe_sources(plan["binding"], evidence)
    inputs, store = m.load_inputs(
        plan["binding"],
        plan["next_native_interval"][1],
        plan["receipt_cache_root"],
        evidence,
        store_ready=store_ready,
    )
    m.equal(inputs.chunk_records, plan["selected_chunk_records"], "registered chunks changed")
    parent = m.load_parent(
        plan["parent"], plan["parent_seal_sha256"], plan["parent_binding"], inputs, evidence
    )
    audit = m.bind_parent_audit(
        plan["parent_audit"]["path"],
        plan["parent_audit"]["sha256"],
        parent["root"],
        parent["seal"],
        evidence,
    )
    m.equal(audit, plan["parent_audit"], "registered independent parent audit changed")
    for filename, key in (
        ("manifest.json", "parent_checkpoint_sha256"),
        ("metadata.json", "parent_metadata_sha256"),
        ("arrays.npz", "parent_payload_sha256"),
    ):
        m.equal(
            evidence.paths[str(parent["root"] / "checkpoint" / filename)],
            plan[key],
            "registered checkpoint differs: " + filename,
        )
    metadata = parent["metadata"]
    m.equal(
        m.array_signatures(parent["arrays"]),
        plan["parent_state_signatures"],
        "registered parent state signatures differ",
    )
    m.equal(metadata["context"], plan["parent_context"], "registered scientific context differs")
    m.equal(metadata["previous_frame"], plan["expected_previous_frame"], "parent native differs")
    m.equal(
        int(parent["arrays"]["history_offsets"][-1]),
        plan["parent_history_rows"],
        "registered full history size differs",
    )
    protocol_path = m.local(m.original.PROTOCOL_PATH)
    evidence.add(protocol_path, plan["binding"]["tracking_protocol_sha256"])
    protocol = m.read(protocol_path)
    m.equal(protocol["source_frame_range"], [m.FIRST, m.FULL_STOP], "scientific range changed")
    candidate_path = parent["root"] / "candidate.npz"
    with np.load(candidate_path, allow_pickle=False) as data:
        m.require(len(data.files) == len(set(data.files)) == 21, "all21 parent fields required")
        candidate = {key: data[key].copy() for key in data.files}
    m.compare_backtrace(parent["arrays"], candidate)
    return dict(
        plan=plan,
        parent=parent,
        candidate=candidate,
        inputs=inputs,
        store=store,
        protocol=protocol,
        code=code,
    )


def cpu_validator(factory):
    tracker = factory()
    m.require(
        type(tracker) is m.original.AcceleratedContinuousPersistentNodeTracker,
        "unchanged original tracker class required",
    )
    m.require(tracker.device.type == "cpu", "CPU-only validation object required")
    m.require(
        tracker.initialize_count == tracker.restore_count == tracker.processed_frames == 0,
        "fresh CPU validation object required",
    )
    return tracker


def qualify_state(
    arrays,
    metadata,
    candidate,
    binding,
    output,
    tracker_factory,
    *,
    on_checkpoint=lambda *args: None,
    on_stage=lambda *args: None,
):
    """Two codec copies, one logical restore; complete synthetic fixtures are allowed."""
    m.require(
        len(arrays) == 39 and arrays["state_pose"].shape == (6, 3, 3, 3),
        "all39 arrays and six parents required",
    )
    m.equal(metadata["initialize_count"], 1, "one original initialization required")
    m.equal(metadata["restore_count"], PARENT_RESTORES, "fixed parent restore10 required")
    m.require(len(candidate) == 21, "all21 original candidate fields required")
    before, meta_before = m.array_signatures(arrays), copy.deepcopy(metadata)
    candidate_before = m.array_signatures(candidate)
    validator = cpu_validator(tracker_factory)
    validator._decode_state(arrays, metadata)
    m.require(
        validator.initialize_count == validator.restore_count == validator.processed_frames == 0,
        "semantic validation performed a logical initialization/restore",
    )
    m.compare_backtrace(arrays, candidate)

    output = Path(output)
    transported_path = output / "transport_checkpoint"
    manifest = m.checkpoint_roundtrip(transported_path, arrays, metadata, binding)
    on_checkpoint("transport", transported_path, manifest)
    transported, transport_meta, _ = m.read_checkpoint(transported_path, expected_bindings=binding)
    m.original._same_arrays(transported, arrays)
    m.equal(transport_meta, metadata, "I/O-only transport changed parent metadata")
    m.compare_backtrace(transported, candidate)
    transport = dict(
        schema=STAGE + "_transport",
        status="ok",
        checkpoint_arrays=before,
        parent_metadata_sha256=m.digest(metadata),
        transported_metadata_sha256=m.digest(transport_meta),
        metadata_unchanged=True,
        all39_bytes_exact=True,
        all21_backtrace_exact=True,
        context_unchanged=True,
        logical_restore_calls=0,
        restore_count=PARENT_RESTORES,
        CPU=True,
        GPU=False,
        tracking_performed=False,
        ownership_verified=False,
        independent_ID_truth=False,
    )
    on_stage("transport", transport)

    restored_tracker = cpu_validator(tracker_factory)
    m.require(restored_tracker is not validator, "restore requires a separate fresh CPU object")
    imported = m.restore_parent(restored_tracker, transported, transport_meta)
    restored, restored_meta = restored_tracker.export_state()
    m.original._same_arrays(restored, arrays)
    expected_metadata = copy.deepcopy(metadata)
    expected_metadata["restore_count"] += 1
    m.equal(restored_meta, expected_metadata, "restore changed more than one counter")
    prefix = m.compare_prefix(arrays, restored)
    exact = m.compare_backtrace(restored, candidate)
    restored_path = output / "restored_checkpoint"
    manifest = m.checkpoint_roundtrip(restored_path, restored, restored_meta, binding)
    on_checkpoint("restore", restored_path, manifest)
    restore = dict(
        schema=STAGE + "_restore",
        status="ok",
        imported=imported,
        checkpoint_arrays=m.array_signatures(restored),
        metadata_sha256=m.digest(restored_meta),
        all39_bytes_exact=True,
        metadata_only_restore_increment=True,
        context_unchanged=True,
        all21_candidate_arrays_exact_backtrace=exact,
        history_prefix=prefix,
        logical_initialize_count=1,
        logical_restore_calls=1,
        parent_restore_count=PARENT_RESTORES,
        restored_restore_count=PARENT_RESTORES + 1,
        CPU=True,
        GPU=False,
        tracking_performed=False,
        ownership_verified=False,
        independent_ID_truth=False,
    )
    on_stage("restore", restore)
    m.equal(m.array_signatures(arrays), before, "caller parent arrays mutated")
    m.equal(metadata, meta_before, "caller parent metadata mutated")
    m.equal(m.array_signatures(candidate), candidate_before, "caller candidate mutated")
    m.equal(transport_meta, meta_before, "CPU restore mutated transported metadata")
    m.original._same_arrays(transported, arrays)
    return dict(
        transport=transport,
        restore=restore,
        candidate_signatures=candidate_before,
        parent_context_sha256=m.digest(metadata["context"]),
    )


def retain_checkpoint(record, label, root, manifest):
    for name in ("manifest.json", "metadata.json", "arrays.npz"):
        path = root / name
        record.output(label + "_" + name, path, sha256=m.sha(path))


def run(args):
    m.require(
        valid_sha(args.registration_sha256), "explicit lowercase registration SHA256 required"
    )
    output = m.local(args.output, exists=False)
    m.require(
        output.is_relative_to(WORK) and not output.exists(), "new workspace probe output required"
    )
    evidence, store = m.Evidence(), None
    m.require(
        args.registration_sha256 == REGISTRATION_SHA
        and Path(args.registration).resolve() == REGISTRATION_ROOT.resolve(),
        "fixed successful CPU520 registration required",
    )
    registered, _ = runner.load_registration(args.registration, args.registration_sha256, evidence)
    m.require(
        output != Path(registered["expected_output"]).resolve(),
        "probe cannot occupy registered child output",
    )

    def retain(value):
        nonlocal store
        store = value

    output.mkdir(parents=True, exist_ok=False)
    with m.original.provenance(STAGE, output, name="qualification") as record:
        record.param("registration_path", str(Path(args.registration).resolve()))
        record.param("registration_sha256", args.registration_sha256)
        record.param("CPU_only", True)
        record.param("tracking_GPU_or_ownership_performed", False)
        try:
            context = load_inputs(
                args.registration, args.registration_sha256, evidence, store_ready=retain
            )
            plan, parent = context["plan"], context["parent"]
            m.require(
                output != Path(plan["expected_output"]),
                "probe cannot occupy registered child output",
            )
            runner.snapshot(context["code"], output, record)
            runner.save_json(
                output / "plan.json",
                dict(
                    schema=STAGE,
                    registration_sha256=args.registration_sha256,
                    parent=plan["parent"],
                    parent_seal_sha256=plan["parent_seal_sha256"],
                    parent_binding=plan["parent_binding"],
                    binding=plan["binding"],
                    transition=plan["transition"],
                    source_hashes=context["code"],
                    CPU_only=True,
                    tracking_performed=False,
                    ownership_verified=False,
                    independent_ID_truth=False,
                ),
                record,
                "plan",
            )
            checks = qualify_state(
                parent["arrays"],
                parent["metadata"],
                context["candidate"],
                plan["binding"],
                output,
                lambda: m.validation_tracker(context["inputs"], context["protocol"]),
                on_checkpoint=lambda label, path, manifest: retain_checkpoint(
                    record, label, path, manifest
                ),
                on_stage=lambda label, value: runner.save_json(
                    output / (label + ".json"), value, record, label
                ),
            )
            candidate_path = output / "candidate.npz"
            with candidate_path.open("xb") as stream:
                np.savez_compressed(stream, **context["candidate"])
            with np.load(candidate_path, allow_pickle=False) as data:
                saved = {key: data[key] for key in data.files}
            m.original._same_arrays(saved, context["candidate"])
            record.output("candidate", candidate_path, sha256=m.sha(candidate_path))
            summary = dict(
                schema=STAGE,
                status="ok",
                checks=checks,
                parent=plan["parent"],
                processed_frames=parent["metadata"]["processed_frames"],
                previous_frame=parent["metadata"]["previous_frame"],
                checkpoint_copies=2,
                parent_restore_count=PARENT_RESTORES,
                restored_restore_count=PARENT_RESTORES + 1,
                logical_initialization_count=1,
                physical_restore_calls=1,
                all39_checkpoint_arrays_byte_exact=True,
                all21_candidate_arrays_byte_exact=True,
                metadata_change="restore_count10->11 only in restored copy; transport unchanged",
                CPU=True,
                GPU=False,
                advance_calls=0,
                fit_calls=0,
                raw_decodes=0,
                flow_calls=0,
                registration_only_CPU_evidence=True,
                tracking_phase_complete=False,
                independent_phase_audit_complete=False,
                ownership_verified=False,
                ID_labels_changed=False,
                independent_ID_truth=False,
            )
            runner.save_json(output / "summary.json", summary, record, "summary")
            m.equal(
                bind_probe_sources(plan["binding"], evidence),
                context["code"],
                "probe imported source closure changed",
            )
        finally:
            if store is not None:
                runner.receipt_usage(output, store, record)
            runner.close_evidence(record, evidence, store)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not valid_sha(args.registration_sha256):
        parser.error("--registration-sha256 must be 64 lowercase hex characters")
    summary = run(args)
    print(f"CPU state qualification {summary['status']}: 39 arrays, restore10->11", flush=True)


if __name__ == "__main__":
    main()

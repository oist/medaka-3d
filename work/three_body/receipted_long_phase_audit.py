"""Receipt-backed input checks for unchanged continuation phase science.

Public prepare/verify return the exact frozen v1 plan/context/result contracts.
Receipt usage and evidence remain operational metadata: the caller owns their
seal and stage, and receives receipt/source hashes in its separate Digests.
No monkeypatch, process, detector, tracker, GPU, raw decode or annotation.
The registered 59760-to-66400 bootstrap remains owned by the old helper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from cached_long_tracking_inputs import load_cached_long_inputs
from long_phase_audit import (
    CHUNK,
    COMMON,
    FIRST,
    FULL_STOP,
    LAB,
    REPO,
    Digests,
    equal,
    load_phase,
    native_endpoint,
    old,
    phase_range,
    require,
    sealed,
    suffix_history,
)
from mask_validation_receipts import ReceiptStore


def _bind_receipt_evidence(evidence: Digests, receipt_store: ReceiptStore):
    """Record a separately guarded receipt set without modifying scientific inputs."""
    receipt_store.final_guard()
    for path, digest in receipt_store.evidence_hashes.items():
        evidence.add(path, digest)


def verified_inputs(binding, stop, evidence, *, receipt_store: ReceiptStore):
    source = LAB / "long_tracking_protocol.json"
    evidence.add(source, binding["tracking_protocol_sha256"])
    protocol = COMMON.read_json(source)
    equal(protocol["source_frame_range"], [FIRST, FULL_STOP], "registered full interval differs")
    for relative, digest in protocol["frozen_tracker_sources_sha256"].items():
        evidence.add(REPO / relative, digest)
    evidence.add(LAB / "long_input_protocol.json", protocol["input_protocol_sha256"])
    for source, digest in binding["code_sha256"].items():
        evidence.add(source, digest)
    equal(binding["package_versions"], protocol["package_versions"], "frozen runtime differs")
    equal(binding["torch_cpu_threads"], protocol["torch_cpu_threads"], "thread count differs")
    registered = binding["inputs"]
    inputs = load_cached_long_inputs(
        Path(registered["seed"]["registration_path"]).parent,
        Path(registered["input_stage_root"]), Path(registered["seed"]["input_root"]),
        through_frame=stop, receipt_store=receipt_store)
    equal(inputs.fixed_bindings, registered, "input context differs")
    require(inputs.full_frame_range == (FIRST, FULL_STOP)
            and np.array_equal(inputs.frame_idx, np.arange(FIRST, stop))
            and len(inputs.chunk_records) == (stop-FIRST)//CHUNK, "native prefix differs")
    for path, digest in inputs.verified_hashes.items():
        evidence.add(path, digest)
    return inputs, protocol


def prepare_phase(parent, through_frame, output_run, evidence, *, receipt_store: ReceiptStore):
    through_frame = native_endpoint(through_frame)
    parent, seal = sealed(parent, evidence)
    start = seal["params"]["through_frame"]
    phase_range(start, through_frame)
    output_run = Path(output_run).resolve()
    require(output_run.is_relative_to(REPO) and not output_run.exists(), "new output required")
    binding = seal["params"]["bindings"]
    inputs, protocol = verified_inputs(
        binding, through_frame, evidence, receipt_store=receipt_store)
    previous = load_phase(parent, seal, inputs, evidence, protocol)
    require(previous["arrays"]["state_pose"].shape == (6, 3, 3, 3), "six active parents required")
    first_chunk = (start-FIRST)//CHUNK
    plan = dict(
        schema="long_continuation_registered_phase_v1", full_target_interval=[FIRST, FULL_STOP],
        parent_phase=str(parent),
        parent_seal_sha256=evidence.paths[str(parent / "continuation.provenance.json")],
        expected_output=str(output_run), parent_processed_interval=[FIRST, start],
        next_native_interval=[start, through_frame], output_interval=[FIRST, through_frame],
        expected_initializations=1, expected_restores=previous["metadata"]["restore_count"]+1,
        expected_external_resets=0, parent_active_beam=6,
        parent_history_rows=int(previous["arrays"]["history_offsets"][-1]),
        frozen_input_and_code_binding=binding,
        parent_checkpoint_context=previous["metadata"]["context"],
        full_selected_chunk_records=inputs.chunk_records,
        next_selected_chunk_records=inputs.chunk_records[first_chunk:],
        full_sequence_complete=through_frame == FULL_STOP,
        prefix_backtrace_provisional=through_frame != FULL_STOP,
        independent_ID_truth=False, processes_started=False)
    _bind_receipt_evidence(evidence, receipt_store)
    return plan, dict(inputs=inputs, parent=previous, protocol=protocol)


def verify_phase(run, plan, evidence, *, receipt_store: ReceiptStore):
    return _verify_phase(run, plan, evidence, bootstrap=False, receipt_store=receipt_store)


def _verify_phase(run, plan, evidence, *, bootstrap, receipt_store: ReceiptStore):
    require(plan["schema"] == "long_continuation_registered_phase_v1", "unknown phase plan")
    run, record = sealed(run, evidence)
    parent, prior_record = sealed(plan["parent_phase"], evidence)
    require(run == Path(plan["expected_output"]).resolve()
            and Path(record["params"]["resume_phase"]).resolve() == parent,
            "executed phase/parent differs from plan")
    equal(evidence.paths[str(parent / "continuation.provenance.json")],
          plan["parent_seal_sha256"], "registered parent seal differs")
    start, stop = plan["next_native_interval"]
    if bootstrap:
        equal([start, stop], [59760, 66400], "bootstrap interval differs")
    else:
        phase_range(start, stop)
    for key, expected in dict(
        full_target_interval=[FIRST, FULL_STOP], parent_processed_interval=[FIRST, start],
        output_interval=[FIRST, stop], expected_initializations=1,
        expected_external_resets=0, parent_active_beam=6,
        full_sequence_complete=stop == FULL_STOP, prefix_backtrace_provisional=stop != FULL_STOP,
    ).items():
        equal(plan[key], expected, "registered lifecycle/range differs: " + key)
    require(record["params"]["through_frame"] == stop
            and prior_record["params"]["through_frame"] == start, "executed endpoint differs")
    binding = record["params"]["bindings"]
    equal(binding, prior_record["params"]["bindings"], "parent/current binding differs")
    equal(binding, plan["frozen_input_and_code_binding"], "registered binding differs")
    inputs, protocol = verified_inputs(binding, stop, evidence, receipt_store=receipt_store)
    equal(inputs.chunk_records, plan["full_selected_chunk_records"], "registered chunks differ")
    equal(inputs.chunk_records[(start-FIRST)//CHUNK:], plan["next_selected_chunk_records"],
          "registered new chunk subset differs")
    before = load_phase(parent, prior_record, inputs, evidence, protocol)
    after = load_phase(run, record, inputs, evidence, protocol)
    equal(after["metadata"]["restore_count"], plan["expected_restores"], "restore count differs")
    equal(before["metadata"]["context"], plan["parent_checkpoint_context"],
          "registered context differs")
    equal(after["metadata"]["context"], before["metadata"]["context"], "restored context differs")
    equal(int(before["arrays"]["history_offsets"][-1]), plan["parent_history_rows"],
          "parent rows differ")
    history = suffix_history(before, after)
    counts = old.prior.support.direct_counts(after["candidate"])
    for key, value in (("direct_pectoral_fish_frames", "direct_pectoral_count"),
                       ("accepted_pectoral_fish_frames", "accepted_pectoral_count"),
                       ("review_fish_frames", "identity_review_count"),
                       ("direct_pectoral_per_slot", "per_slot_direct_pectoral"),
                       ("accepted_pectoral_per_slot", "per_slot_accepted_pectoral")):
        equal(counts[key], after["summary"][value], "summary availability differs")
    result = dict(schema="long_continuation_generic_phase_audit_v1", status="ok", run=str(run),
                parent=str(parent), frame_range=[FIRST, stop], new_interval=[start, stop],
                candidate_sha256=evidence.paths[str(run / "candidate.npz")],
                all_39_checkpoint_arrays_verified=True,
                all_21_candidate_arrays_exact_backtrace=after["comparisons"],
                logical_initialization_count=1,
                checkpoint_restore_count=after["metadata"]["restore_count"],
                external_pose_reset_count=0, history=history, counts=counts,
                full_sequence_complete=stop == FULL_STOP,
                prefix_backtrace_provisional=stop != FULL_STOP,
                independently_redecoded_raw_images=0, independent_geometric_reprojection=False,
                independent_ID_truth=False,
                scope="All saved input/code/output hashes, checkpoint codec, complete backtrace, "
                      "exact parent history, all new beam recurrences and source exclusivity. "
                      "No independent raw decode, flow rerun or 2mm/12px reprojection here; "
                      "those remain producer contracts and separate sampled audits.")
    _bind_receipt_evidence(evidence, receipt_store)
    return result

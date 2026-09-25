"""Resume the committed Nodewise 74700 checkpoint with one observation reserve.

Scientific comparison window [74700,76360), inherited history [49800,74700).
The parent five-minute process may still be running: this caller binds the
committed checkpoint and its fixed plan, not a future full-run success seal.
Actual GPU execution is explicit through this standalone command.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import continue_fast_tail_five_minutes as base
import nodewise_body_row_tracker as nodewise
import numpy as np

# isort: split
# The frozen base bootstrap adds the lab modules to sys.path before this import.
import observation_reserve_tracker as variant
from observation_reserve_diagnostic_batch import ObservationReserveDiagnosticBatch

prior, proof, original, qualified = base.prior, base.proof, base.original, base.qualified
torch, WORK, REPO, require = base.torch, base.WORK, base.REPO, base.require
FIRST, ORIGINAL_SCIENCE_START, START, STOP, CHUNK = 49800, 66400, 74700, 76360, 1660
SCIENCE_START = START
STAGE = "observation_reserve_ten_second_continuation_v1"
PARENT = WORK / "nodewise_body_row_five_minute_continuation_v1"
PARENT_SOURCE_SHA = "0ea3d27b939a804c081d6249c7462498ee0a16504e623e021037b0fc32f50b64"
CHECKPOINT_RELATIVE = "checkpoints/through_074700"
CHECKPOINT_SHA = "7c9672b6fde9f96ee5d23efe7dbbd6e7f88d48299d81e86e00a87fcca5c5a076"
PARENT_HEADERS = {
    "plan.json": "e05e174a291d6d79bf2cf3f74b848e7b125b3304f8ad1f0d168165dac3a8ba31",
    "input_index.json": "f5840d96f166f04516c72326aee157e78fac01a290296653aabdec01dd8ae568",
    "state_import.json": "fb68e6c59ef50452645f69ecd09aa19fbff8c6f5625742fb7335659d2baeb45e",
    CHECKPOINT_RELATIVE + "/manifest.json": CHECKPOINT_SHA,
    CHECKPOINT_RELATIVE
    + "/metadata.json": "46ba3c749f0ec5232b68d0616e928eed0cd4d1910a9facaa81b4c39207e93502",
}
ORIGINAL_TRACKING_SHA = "9915c44204c7acaee0abcada3a3fa2dfa82b7ac017c0b7d4fc9b50b628d48a6e"
ORIGINAL_CHECKPOINT_SHA = "0f8db6356175de3223589576b4539a3a73977eb8151324b94340c5b1b174730f"
BATCH_SIZE = 100

PINS = {
    "observation_reserve_diagnostic_batch.py": (
        "f04cd7ef660b71f77672022a1889600555d80793de74b2225804c25c989ea60d"
    ),
    "nodewise_diagnostic_batch.py": (
        "f494819316ea4b1000378c8672c0491951b065980a43468143d80b016c896bab"
    ),
    "observation_reserve_tracker.py": (
        "7b03698dacda5d18be87ff99c0bccd47474a94db08ae89aff0231f4dcb50c74a"
    ),
    "observation_reserve_selection.py": (
        "b4b69b67e488db00fdff49b278a0c39f3966cc24e55749e2b1ca8251d1196853"
    ),
    "nodewise_body_row_proposals.py": (
        "ce8c3f7716ef36d87f1dc32e10faa4686575d5ed78b2f102064a78a5d9577f06"
    ),
    "nodewise_body_row_tracker.py": (
        "b574ce26481c2c1457ceb1428659aa7bce25783e36957fd2eb9b6399f52b8d16"
    ),
    "continue_fast_tail_five_minutes.py": (
        "d90e9eb49038dac4fd1e18068334ba89eb8df1a813a37c6aec071c20db1a61a6"
    ),
    "run_instance_rescore_10s.py": (
        "91a66646f89a6abb59b6d60bc1b357b383167c21eb9d40bc47491802d05e9544"
    ),
    "sparse_instance_rescore_tracker.py": (
        "d041aff3a0d8cbd71341dbe67dd9b5a7a1e2e7aeb165f55e4b08f8fabdb8fb79"
    ),
    "body_row_proposals.py": ("c4dbca4c44c6b2cee2f68a2175c809987e489b5e131a1fb792f9bbd36de918fb"),
}


def lifecycle(tracker, through):
    require(
        tracker.initialize_count == 1
        and tracker.restore_count == 5
        and tracker.previous_frame == through - 1
        and tracker.processed_frames == through - FIRST,
        "observation-reserve continuation lifecycle changed",
    )


def load_parent_headers(evidence):
    """Committed checkpoint plus fixed small metadata; no parent completion claim."""
    docs = {}
    for relative, digest in PARENT_HEADERS.items():
        path = PARENT / relative
        evidence.add(path, digest)
        docs[relative] = proof.read_json(path)
    manifest = docs[CHECKPOINT_RELATIVE + "/manifest.json"]
    metadata = docs[CHECKPOINT_RELATIVE + "/metadata.json"]
    plan, index, imported = (
        docs[k] for k in ("plan.json", "input_index.json", "state_import.json")
    )
    require(
        manifest["schema"] == "medaka_tracker_checkpoint_v1"
        and manifest["status"] == "ok"
        and len(manifest["arrays"]) == 39
        and manifest["bindings"] == plan["checkpoint_binding"]
        and metadata["context"] == plan["tracker_context"],
        "committed Nodewise checkpoint and plan differ",
    )
    require(
        manifest["metadata"]["file"] == "metadata.json"
        and manifest["metadata"]["sha256"] == PARENT_HEADERS[CHECKPOINT_RELATIVE + "/metadata.json"]
        and manifest["metadata"]["bytes"]
        == (PARENT / CHECKPOINT_RELATIVE / "metadata.json").stat().st_size
        and manifest["payload"]["file"] == "arrays.npz"
        and proof.valid_sha(manifest["payload"]["sha256"])
        and manifest["payload"]["bytes"] > 0,
        "committed checkpoint member descriptors differ",
    )
    require(
        metadata["schema"] == "persistent_tracker_full_state_v1"
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 4
        and metadata["previous_frame"] == START - 1
        and metadata["processed_frames"] == START - FIRST
        and metadata["context"]["implementation"] == "BodyRowContinuousTracker"
        and metadata["context"]["sparse_distance_backend"]["effective_from_native"]
        == ORIGINAL_SCIENCE_START
        and metadata["context"]["body_row_policy"]["effective_from_native"]
        == ORIGINAL_SCIENCE_START
        and metadata["context"]["instance_rescore_policy"]["effective_from_native"] == STOP
        and "observation_reserve_policy" not in metadata["context"]
        and "endpoint_mask_priority_policy" not in metadata["context"],
        "parent native/lifecycle/policy differs",
    )
    source = str((WORK / "run_nodewise_body_row_five_minutes.py").resolve())
    require(
        plan["schema"] == PARENT.name
        and plan["source_sha256"].get(source) == PARENT_SOURCE_SHA
        and plan["process_interval"] == [ORIGINAL_SCIENCE_START, 116200]
        and plan["retained_interval"] == [FIRST, 116200]
        and plan["scientific_interval"] == [ORIGINAL_SCIENCE_START, 116200]
        and plan["parent_tracking_sha256"] == ORIGINAL_TRACKING_SHA
        and plan["parent_checkpoint_sha256"] == ORIGINAL_CHECKPOINT_SHA
        and plan["initialize_count"] == 1
        and plan["restore_count"] == 4
        and plan["new_begin_calls"] == plan["external_pose_resets"] == 0
        and plan["mask_split_priority_applied"] is False,
        "parent original history or scientific plan differs",
    )
    evidence.add(source, PARENT_SOURCE_SHA)
    require(
        imported["schema"] == "nodewise_original_state_import_v1"
        and imported["status"] == "ok"
        and imported["new_context"] == metadata["context"]
        and imported["all_arrays_byte_preserved"] is True
        and imported["original_metadata_unchanged"] is True
        and imported["previous_frame"] == ORIGINAL_SCIENCE_START - 1
        and imported["processed_frames"] == ORIGINAL_SCIENCE_START - FIRST
        and imported["logical_initialize_count"] == 1
        and imported["added_restore_count"] == 1
        and imported["new_begin_calls"] == 0
        and imported["imported_metadata"]["restore_count"] == 4
        and imported["old_history_rewritten"] is False
        and imported["review_cleared_by_migration"] is False,
        "original 5:00 history import differs",
    )
    binding = manifest["bindings"]["original_binding"]
    require(
        index["original_binding"] == plan["original_binding"] == binding
        and len(index["prefix_chunks"]) == 40,
        "parent registered source prefix differs",
    )
    return dict(manifest=manifest, metadata=metadata, binding=binding, input_index=index, plan=plan)


def load_parent_payload(headers, evidence):
    """Execution only: bind and decode all39 once, preserving the committed metadata."""
    path = PARENT / CHECKPOINT_RELATIVE / "arrays.npz"
    descriptor = headers["manifest"]["payload"]
    require(path.stat().st_size == descriptor["bytes"], "parent payload size changed")
    evidence.add(path, descriptor["sha256"])
    arrays, metadata, manifest = prior.read_checkpoint(
        PARENT / CHECKPOINT_RELATIVE, expected_bindings=headers["manifest"]["bindings"]
    )
    require(
        len(arrays) == 39 and metadata == headers["metadata"] and manifest == headers["manifest"],
        "captured parent checkpoint differs from verified headers",
    )
    return arrays, metadata


def source_code(evidence):
    code = prior.local_code()
    prior.source_contract(code)
    require(
        str(Path(__file__).resolve()) in code, "new continuation caller absent from source closure"
    )
    for name, digest in PINS.items():
        require(
            code.get(str((WORK / name).resolve())) == digest, "frozen dependency differs: " + name
        )
    for path, digest in code.items():
        evidence.add(path, digest)
    return code


def drive(tracker, iterator, batch, *, synchronize, guard, first=START, stop=STOP):
    require((first, stop) == (START, STOP), "only the fixed native window is authorized")
    rows = []
    timings = dict(
        advance_seconds=0.0, native_next_seconds=0.0, diagnostic_save_seconds=0.0, guard_seconds=0.0
    )
    for native in range(first, stop):
        lifecycle(tracker, native)
        tick = time.perf_counter()
        observation = next(iterator)
        timings["native_next_seconds"] += time.perf_counter() - tick
        require(observation.frame_idx == native, "native frame skipped or reordered")
        tick = time.perf_counter()
        guard()
        timings["guard_seconds"] += time.perf_counter() - tick
        tick = time.perf_counter()
        tracker.advance(observation)
        synchronize()
        timings["advance_seconds"] += time.perf_counter() - tick
        lifecycle(tracker, native + 1)
        tick = time.perf_counter()
        rows.append(batch.append(tracker, native))
        timings["diagnostic_save_seconds"] += time.perf_counter() - tick
        tick = time.perf_counter()
        guard()
        timings["guard_seconds"] += time.perf_counter() - tick
        if (native + 1 - first) % 100 == 0 or native + 1 == stop:
            print(
                json.dumps(dict(event="observation_reserve_progress", through_frame=native + 1)),
                flush=True,
            )
    tick = time.perf_counter()
    batch.flush()
    timings["diagnostic_save_seconds"] += time.perf_counter() - tick
    require(
        batch.saved_frames == batch.totalframes == STOP - START and batch.pending_frames == 0,
        "complete three-family diagnostic batch coverage required",
    )
    require(
        [r["frame_idx"] for r in rows] == list(range(START, STOP)), "diagnostic frame coverage gap"
    )
    return rows, timings


def proposal_summary(rows, candidate):
    require(
        [r["frame_idx"] for r in rows] == list(range(START, STOP)),
        "diagnostic frame coverage differs",
    )
    admitted = [r["body_row"]["admitted"] for r in rows]
    selected = (candidate["frame_idx"] >= START) & (candidate["frame_idx"] < STOP)
    require(int(selected.sum()) == STOP - START, "candidate process extent differs")
    return dict(
        per_frame_admitted=admitted,
        admitted_min=min(admitted),
        admitted_max=max(admitted),
        total_admitted=sum(admitted),
        frames_with_additions=sum(n > 0 for n in admitted),
        zero_headroom_frames=sum(r["body_row"]["headroom"] == 0 for r in rows),
        final_selected_body_row_frame_count=int(
            np.sum(candidate["proposal_kind"][selected] == "body_row")
        ),
        final_selected_scope=[START, STOP],
        proposal_diagnostic_is_not_selected_path=True,
    )


def reserve_summary(rows):
    require([r["frame_idx"] for r in rows] == list(range(START, STOP)), "reserve coverage differs")
    values = [r["observation_reserve"] for r in rows]
    reasons = {}
    for value in values:
        reason = value["reason"]
        reasons[reason] = reasons.get(reason, 0) + 1
    return dict(
        changed_frames=sum(value["changed"] for value in values),
        reasons=reasons,
        current_beam_selection_only=True,
        final_path_reserved_ancestor_count_not_computed=True,
        original_numeric_costs_preserved=True,
        no_future_retention_guarantee=True,
    )


def main(args):
    destination = args.output.resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(),
        "fresh workspace output required",
    )
    out, evidence = proof.Output(destination), original.phase.Digests()
    started, checkpoint, code = proof.utc(), None, {}
    try:
        code = source_code(evidence)
        for path in sorted(code):
            out.write(
                "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), Path(path).read_bytes()
            )
        headers = load_parent_headers(evidence)
        arrays, metadata = load_parent_payload(headers, evidence)
        binding = headers["binding"]
        store = qualified.ReceiptStore(args.receipt_root.resolve())
        print(
            json.dumps(dict(event="observation_reserve_input_validation", output=str(destination))),
            flush=True,
        )
        inputs, protocol = original.verified_inputs(binding, STOP, evidence, receipt_store=store)
        require(
            len(inputs.chunk_roots) == len(inputs.chunk_records) == 16
            and inputs.chunk_records == headers["input_index"]["prefix_chunks"][:16],
            "same consumed source prefix through76360 required",
        )
        out.write(
            "input_index.json", dict(original_binding=binding, prefix_chunks=inputs.chunk_records)
        )
        original.cv2.setNumThreads(binding["opencv_threads"])
        torch.set_num_threads(binding["torch_cpu_threads"])
        torch.set_num_interop_threads(1)
        prior.runtime(binding, gpu=False)
        qualified.check_headroom(prior.smi_sample(), startup=True)
        torch.cuda.set_per_process_memory_fraction(0.25, 0)
        torch.cuda.reset_peak_memory_stats()
        environment = prior.runtime(binding, gpu=True)
        config = qualified.TrackerConfig(**protocol["tracker_config"])
        require(config.device == "cuda", "registered CUDA execution required")
        kwargs = dict(
            recovery_config=qualified.NodeRecoveryConfig(**protocol["node_recovery_config"]),
            persistence_config=qualified.PersistentNodeConfig(**protocol["persistence_config"]),
            sparse_effective_from_native=ORIGINAL_SCIENCE_START,
            body_row_effective_from_native=ORIGINAL_SCIENCE_START,
        )
        source = nodewise.BodyRowContinuousTracker(
            inputs.rig, inputs.initial_pose_xyz.copy(), config, **kwargs
        )
        tracker = variant.ObservationReserveContinuousTracker(
            inputs.rig,
            inputs.initial_pose_xyz.copy(),
            config,
            observation_reserve_effective_from_native=START,
            **kwargs,
        )
        imported = variant.import_nodewise_state(tracker, source, arrays, metadata)
        lifecycle(tracker, START)
        require(source.initialize_count == 0, "migration source unexpectedly initialized")
        out.write("state_import.json", imported)
        context = tracker._context()
        checkpoint_binding = dict(
            schema=STAGE + "_checkpoint",
            original_binding=binding,
            source_sha256=code,
            parent_checkpoint_producer_source_sha256=PARENT_SOURCE_SHA,
            parent_five_minute_completion_claimed=False,
            original_tracking_sha256=ORIGINAL_TRACKING_SHA,
            original_checkpoint_sha256=ORIGINAL_CHECKPOINT_SHA,
            parent_checkpoint_sha256=CHECKPOINT_SHA,
            retained_interval=[FIRST, STOP],
            process_interval=[START, STOP],
            scientific_interval=[SCIENCE_START, STOP],
            original_scientific_history_start=ORIGINAL_SCIENCE_START,
            tracker_context=context,
        )
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                source_sha256=code,
                original_binding=binding,
                checkpoint_binding=checkpoint_binding,
                tracker_context=context,
                process_interval=[START, STOP],
                retained_interval=[FIRST, STOP],
                scientific_interval=[SCIENCE_START, STOP],
                original_scientific_history_start=ORIGINAL_SCIENCE_START,
                parent_checkpoint_producer_source_sha256=PARENT_SOURCE_SHA,
                parent_five_minute_completion_claimed=False,
                original_tracking_sha256=ORIGINAL_TRACKING_SHA,
                original_checkpoint_sha256=ORIGINAL_CHECKPOINT_SHA,
                parent_checkpoint_sha256=CHECKPOINT_SHA,
                initialize_count=1,
                restore_count=5,
                new_begin_calls=0,
                external_pose_resets=0,
                environment=environment,
                precision="float64",
                allocator_fraction=0.25,
                reserved_limit_bytes=4 * 1024**3,
                existing_five_parts_and_sixth_term_unchanged=True,
                source_assignment_recomputed_for_new_proposals=False,
                original_proposal_pool_retained=True,
                old_candidate_cap_unchanged=True,
                only_pre74700_temporal_state_imported=True,
                observation_reserve_effective_from_native=START,
                body_row_and_sparse_backend_effective_from_native=ORIGINAL_SCIENCE_START,
                sixth_term_effective_from_native=STOP,
                sixth_term_neutral_for_entire_scientific_window=True,
                diagnostic_batch_size=BATCH_SIZE,
                diagnostic_families=["instance", "body_row", "observation_reserve"],
                all_frame_diagnostics_saved=True,
                inherited_seed_provenance_unchanged=True,
                parent_committed_checkpoint_used_without_final_seal=True,
                physical_ID_truth=False,
                automatic_adoption=False,
            ),
        )
        evidence.final_guard()
        print(
            json.dumps(dict(event="observation_reserve_imported", through_frame=START)), flush=True
        )

        def guard():
            qualified.check_headroom(prior.cuda_headroom())
            qualified.check_memory(qualified.qualification.memory_sample())

        tick = time.perf_counter()
        reader = original.SourceSequenceReader(
            inputs.chunk_roots[(START - FIRST) // CHUNK], inputs.input_root
        )
        require(
            reader.frames[0] == START and reader.frames[-1] + 1 == STOP,
            "wrong native source chunk15",
        )
        iterator = original.bounded_frames(reader, first=START, stop=STOP)
        reader_seconds = time.perf_counter() - tick
        try:
            batch = ObservationReserveDiagnosticBatch(out, batch_size=BATCH_SIZE)
            rows, timings = drive(
                tracker, iterator, batch, synchronize=torch.cuda.synchronize, guard=guard
            )
        finally:
            iterator.close()
        out.write(
            "diagnostic_index.json",
            dict(
                rows=rows,
                batches=batch.batches,
                process_interval=[START, STOP],
                batch_size=BATCH_SIZE,
                diagnostic_families=["instance", "body_row", "observation_reserve"],
            ),
        )
        path = destination / "checkpoints" / f"through_{STOP:06d}"
        saved, saved_metadata, checkpoint = base.preserve_checkpoint(
            tracker, dict(arrays=arrays, metadata=metadata), path, checkpoint_binding
        )
        # Validate actual decode eligibility without restoring or changing the tracker.
        tracker._decode_state(saved, saved_metadata)
        lifecycle(tracker, STOP)
        for name in ("manifest.json", "metadata.json", "arrays.npz"):
            payload = path / name
            out.records[payload.relative_to(destination).as_posix()] = dict(
                path=str(payload),
                sha256=prior.checks.sha(payload),
                size_bytes=payload.stat().st_size,
            )
        candidate = tracker.finish()
        require(
            prior.compare_candidate(candidate, original.phase.COMMON.traced_candidate(saved), STOP)[
                "exact"
            ],
            "finish changed saved backtrace",
        )
        candidate_record = out.write("candidate.npz", candidate)
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                process_interval=[START, STOP],
                native_frames=STOP - START,
                process_frames=STOP - START,
                retained_interval=[FIRST, STOP],
                scientific_interval=[SCIENCE_START, STOP],
                original_scientific_history_start=ORIGINAL_SCIENCE_START,
                scientific_native_frames=STOP - SCIENCE_START,
                parent_checkpoint_producer_source_sha256=PARENT_SOURCE_SHA,
                parent_five_minute_completion_claimed=False,
                original_tracking_sha256=ORIGINAL_TRACKING_SHA,
                original_checkpoint_sha256=ORIGINAL_CHECKPOINT_SHA,
                parent_checkpoint_sha256=CHECKPOINT_SHA,
                source_sha256=code,
                tracker_context=context,
                candidate=candidate_record,
                candidate_arrays=base.array_signature(candidate),
                checkpoint_manifest_sha256=checkpoint["manifest_sha256"],
                checkpoint=checkpoint,
                checkpoint_count=1,
                initialize_count=1,
                restore_count=5,
                external_pose_resets=0,
                new_begin_calls=0,
                finish_calls=1,
                all39_roundtrip=True,
                completed_checkpoint_decoder_accepted=True,
                prior_history_prefix_unchanged=True,
                finish_matches_checkpoint_backtrace=True,
                diagnostic_frames=len(rows),
                diagnostic_batch_count=len(batch.batches),
                observation_reserve=reserve_summary(rows),
                body_row=proposal_summary(rows, candidate),
                timings=dict(reader_seconds=reader_seconds, **timings),
                existing_five_parts_and_sixth_term_unchanged=True,
                physical_ID_truth=False,
                independent_evaluation_pending=True,
                automatic_adoption=False,
            ),
        )
        store.final_guard()
        for path, digest in store.evidence_hashes.items():
            evidence.add(path, digest)
        out.write("mask_validation_receipts.json", store.uses)
        require(prior.local_code() == code, "source closure changed")
        evidence.final_guard()
        out.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(error),
            committed_checkpoint=checkpoint,
        )
        with (destination / "tracking.provenance.json").open("xb") as stream:
            stream.write(proof.json_bytes(seal))
        raise
    with (destination / "tracking.provenance.json").open("xb") as stream:
        stream.write(proof.json_bytes(seal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    main(parser.parse_args())

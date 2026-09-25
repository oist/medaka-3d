"""Continue original66400 state for 49,800 native Nodewise frames on CUDA.

The original5:00 initialization and all39 state arrays survive one explicit
context migration. Body proposals and the sparse backend start at66400; the
frozen sixth-term policy still starts at76360. All full candidate/body diagnostics
are saved in100-frame batches. No sampling, split-priority, reset or ID labels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import run_nodewise_body_row_negative_continuation as old

# isort: split
# The frozen caller above establishes the laboratory module import paths.
import nodewise_diagnostic_batch as batching
import nodewise_original_state_import as migration

base, prior, proof = old.base, old.prior, old.proof
original, qualified, torch = old.original, old.qualified, old.torch
WORK, REPO, require, variant = old.WORK, old.REPO, old.require, old.variant
FIRST, START, STOP, CHUNK = 49800, 66400, 116200, 1660
STAGE = "nodewise_body_row_five_minute_continuation_v1"
PARENT = WORK / "long_continuation_through66400_v1"
PARENT_SHA = "9915c44204c7acaee0abcada3a3fa2dfa82b7ac017c0b7d4fc9b50b628d48a6e"
CHECKPOINT_SHA = "0f8db6356175de3223589576b4539a3a73977eb8151324b94340c5b1b174730f"
METADATA_SHA = "6f79853f019fa6ca4f298bb39178dd70e5ca48dae2bd8cef65191a1ea16b0d98"
BATCH_SIZE = 100
MILESTONES = tuple(range(START + 9960, STOP + 1, 9960))
PINS = {
    **old.PINS,
    **migration.SOURCE_PINS,
    "run_nodewise_body_row_negative_continuation.py": (
        "4f2d31a383a8d59320ef0c09f8dcddf645dfb20e0849bd2f1dc27398d719669e"
    ),
    # Filled only after the two new components pass their focused validation.
    "nodewise_original_state_import.py": (
        "25801a84fd848d312dc902852124b2c215b04f0ca75e44293cf77829cf1ab8e7"
    ),
    "nodewise_diagnostic_batch.py": (
        "f494819316ea4b1000378c8672c0491951b065980a43468143d80b016c896bab"
    ),
}


def chunks():
    return tuple((first, first + CHUNK) for first in range(START, STOP, CHUNK))


def lifecycle(tracker, through):
    require(
        tracker.initialize_count == 1
        and tracker.restore_count == 4
        and tracker.previous_frame == through - 1
        and tracker.processed_frames == through - FIRST,
        "original-to-Nodewise continuous lifecycle differs",
    )


def source_code(evidence):
    code = prior.local_code()
    prior.source_contract(code)
    require(str(Path(__file__).resolve()) in code, "new caller missing from source closure")
    for name, digest in PINS.items():
        require(proof.valid_sha(digest), "component not yet frozen: " + name)
        require(code.get(str((WORK / name).resolve())) == digest, "source differs: " + name)
    for path, digest in code.items():
        evidence.add(path, digest)
    return code


def parent_member(seal, role, relative, evidence):
    path = (PARENT / relative).resolve(strict=True)
    require(path.is_relative_to(PARENT), "parent member escaped root")
    row = seal["outputs"][role]
    require(
        Path(row["path"]).resolve() == path
        and path.is_file()
        and path.stat().st_size == row["size"],
        "original parent descriptor differs: " + relative,
    )
    evidence.add(path, row["sha256"])
    return path


def load_parent_headers(evidence):
    """Bound original metadata only; no checkpoint/candidate/image array reads."""
    evidence.add(PARENT / "continuation.provenance.json", PARENT_SHA)
    seal = proof.read_json(PARENT / "continuation.provenance.json")
    require(
        seal["status"] == "ok"
        and seal["stage"] == "three_body_long_continuation"
        and seal["params"]["through_frame"] == START
        and seal["params"]["all_input_source_snapshot_output_sha_unchanged"] is True,
        "successful original66400 parent required",
    )
    manifest_path = parent_member(seal, "checkpoint_manifest", "checkpoint/manifest.json", evidence)
    metadata_path = parent_member(
        seal, str(PARENT / "checkpoint/metadata.json"), "checkpoint/metadata.json", evidence
    )
    require(
        evidence.paths[str(manifest_path)] == CHECKPOINT_SHA
        and evidence.paths[str(metadata_path)] == METADATA_SHA,
        "different original checkpoint",
    )
    manifest, metadata = proof.read_json(manifest_path), proof.read_json(metadata_path)
    summary = proof.read_json(parent_member(seal, "summary", "summary.json", evidence))
    consumed = proof.read_json(
        parent_member(seal, "consumed_chunks", "consumed_chunks.json", evidence)
    )
    binding = seal["params"]["bindings"]
    require(
        manifest["schema"] == "medaka_tracker_checkpoint_v1"
        and manifest["status"] == "ok"
        and manifest["bindings"] == binding
        and len(manifest["arrays"]) == 39,
        "original checkpoint schema/binding differs",
    )
    for role, name in (("metadata", "metadata.json"), ("payload", "arrays.npz")):
        rec = seal["outputs"][str(PARENT / "checkpoint" / name)]
        require(
            manifest[role]["file"] == name
            and manifest[role]["sha256"] == rec["sha256"]
            and manifest[role]["bytes"] == rec["size"],
            "manifest and original success record differ",
        )
    require(
        metadata["schema"] == "persistent_tracker_full_state_v1"
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 3
        and metadata["previous_frame"] == START - 1
        and metadata["processed_frames"] == START - FIRST
        and metadata["context"]["implementation"] == "AcceleratedContinuousPersistentNodeTracker",
        "original lifecycle/context differs",
    )
    require(
        summary["schema"] == "long_continuation_phase_v1"
        and summary["technical_completion"] is True
        and summary["processed_interval"] == [FIRST, START]
        and summary["processed_total"] == START - FIRST
        and summary["logical_initialization_count"] == 1
        and summary["checkpoint_restore_count"] == 3
        and summary["external_pose_reset_count"] == 0
        and summary["temporal_subsampling"] is False
        and summary["human_annotation_input"] is False
        and summary["post_initialization_baseline_coordinates_input"] is False
        and summary["consumed_chunk_count"] == len(consumed) == 10,
        "original successful summary differs",
    )
    return dict(seal=seal, manifest=manifest, metadata=metadata, binding=binding, consumed=consumed)


def load_parent_payload(headers, evidence):
    parent_member(
        headers["seal"], str(PARENT / "checkpoint/arrays.npz"), "checkpoint/arrays.npz", evidence
    )
    arrays, metadata, manifest = prior.read_checkpoint(
        PARENT / "checkpoint", expected_bindings=headers["binding"]
    )
    require(
        len(arrays) == 39 and metadata == headers["metadata"] and manifest == headers["manifest"],
        "original checkpoint differs from bound headers",
    )
    return dict(arrays=arrays, metadata=metadata)


def validate_source_prefix(inputs, headers):
    require(
        len(inputs.chunk_roots) == len(inputs.chunk_records) == 40
        and inputs.chunk_records[:10] == headers["consumed"]
        and np.array_equal(inputs.frame_idx, np.arange(FIRST, STOP)),
        "complete common input prefix required",
    )
    for i, row in enumerate(inputs.chunk_records):
        require(
            (row["index"], row["start"], row["stop"])
            == (i, FIRST + i * CHUNK, FIRST + (i + 1) * CHUNK),
            "source chunk skipped or reordered",
        )
    require(
        [(r["start"], r["stop"]) for r in inputs.chunk_records[10:]] == list(chunks()),
        "exact30 scientific chunks required",
    )


def drive_chunk(tracker, iterator, batch, first, stop, *, synchronize, guard):
    """Serial native frames; no initialization, restoration or finish in this loop."""
    require((first, stop) in chunks(), "only a full registered chunk may be advanced")
    rows = []
    timings = dict(
        advance_seconds=0.0, native_next_seconds=0.0, diagnostic_save_seconds=0.0, guard_seconds=0.0
    )
    for native in range(first, stop):
        lifecycle(tracker, native)
        tick = time.perf_counter()
        observation = next(iterator)
        timings["native_next_seconds"] += time.perf_counter() - tick
        require(observation.frame_idx == native, "native observation skipped/reordered")
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
                json.dumps(
                    dict(
                        event="nodewise_five_minute_progress",
                        through_frame=native + 1,
                        science_processed=native + 1 - START,
                    )
                ),
                flush=True,
            )
    tick = time.perf_counter()
    batch.flush()
    timings["diagnostic_save_seconds"] += time.perf_counter() - tick
    require([r["frame_idx"] for r in rows] == list(range(first, stop)), "diagnostic coverage gap")
    require(
        batch.saved_frames == batch.totalframes and batch.pending_frames == 0,
        "all accepted frame diagnostics must be committed at the chunk boundary",
    )
    return rows, timings


def summarize_proposals(rows, candidate):
    require([r["frame_idx"] for r in rows] == list(range(START, STOP)), "all frame counts required")
    mask = (candidate["frame_idx"] >= START) & (candidate["frame_idx"] < STOP)
    require(int(mask.sum()) == STOP - START, "candidate scientific interval differs")
    added = [r["body_row"]["admitted"] for r in rows]
    return dict(
        total_admitted=sum(added),
        admitted_min=min(added),
        admitted_max=max(added),
        frames_with_additions=sum(v > 0 for v in added),
        zero_headroom_frames=sum(r["body_row"]["headroom"] == 0 for r in rows),
        final_selected_body_row_frame_count=int(
            np.sum(candidate["proposal_kind"][mask] == "body_row")
        ),
        final_selected_scope=[START, STOP],
        proposal_diagnostic_is_not_selected_path=True,
    )


def main(args):
    destination = args.output.resolve()
    require(
        destination.is_relative_to(WORK) and destination != WORK and not destination.exists(),
        "fresh workspace output required",
    )
    out, evidence = proof.Output(destination), original.phase.Digests()
    started, committed, code = proof.utc(), None, {}
    started_clock = time.perf_counter()
    try:
        code = source_code(evidence)
        for path in sorted(code):
            out.write(
                "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), Path(path).read_bytes()
            )
        headers = load_parent_headers(evidence)
        binding = headers["binding"]
        store = qualified.ReceiptStore(args.receipt_root.resolve())
        print(
            json.dumps(
                dict(event="nodewise_five_minute_input_validation", output=str(destination))
            ),
            flush=True,
        )
        inputs, protocol = original.verified_inputs(binding, STOP, evidence, receipt_store=store)
        validate_source_prefix(inputs, headers)
        parent = load_parent_payload(headers, evidence)
        previous = original.verify_previous_observation(parent, inputs)
        out.write("previous_observation_check.json", previous)
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
        )
        source = original.AcceleratedContinuousPersistentNodeTracker(
            inputs.rig, inputs.initial_pose_xyz.copy(), config, **kwargs
        )
        tracker = variant.BodyRowContinuousTracker(
            inputs.rig,
            inputs.initial_pose_xyz.copy(),
            config,
            body_row_effective_from_native=START,
            sparse_effective_from_native=START,
            **kwargs,
        )
        imported = migration.import_original_state(
            tracker, source, parent["arrays"], parent["metadata"]
        )
        lifecycle(tracker, START)
        require(source.initialize_count == 0, "source unexpectedly initialized")
        out.write("state_import.json", imported)
        context = tracker._context()
        binding_new = dict(
            schema=STAGE + "_checkpoint",
            original_binding=binding,
            source_sha256=code,
            parent_tracking_sha256=PARENT_SHA,
            parent_checkpoint_sha256=CHECKPOINT_SHA,
            retained_interval=[FIRST, STOP],
            process_interval=[START, STOP],
            scientific_interval=[START, STOP],
            tracker_context=context,
        )
        plan = dict(
            schema=STAGE,
            source_sha256=code,
            original_binding=binding,
            checkpoint_binding=binding_new,
            tracker_context=context,
            process_interval=[START, STOP],
            retained_interval=[FIRST, STOP],
            scientific_interval=[START, STOP],
            parent_tracking_sha256=PARENT_SHA,
            parent_checkpoint_sha256=CHECKPOINT_SHA,
            initialize_count=1,
            restore_count=4,
            new_begin_calls=0,
            external_pose_resets=0,
            environment=environment,
            precision="float64",
            allocator_fraction=0.25,
            reserved_limit_bytes=4 * 1024**3,
            body_row_and_sparse_backend_effective_from_native=START,
            sixth_term_effective_from_native=76360,
            sixth_term_neutral_before_boundary=True,
            existing_five_parts_and_sixth_term_unchanged=True,
            mask_split_priority_applied=False,
            source_assignment_recomputed_for_new_proposals=False,
            original_proposal_pool_retained=True,
            old_candidate_cap_unchanged=True,
            inherited_seed_provenance_unchanged=True,
            diagnostic_batch_size=BATCH_SIZE,
            all_native_frames_and_complete_diagnostics_saved=True,
            temporal_subsampling=False,
            future_control_coordinates_used=False,
            physical_ID_truth=False,
            automatic_adoption=False,
        )
        require(
            context["instance_rescore_policy"]["effective_from_native"] == 76360,
            "sixth term must keep its registered boundary",
        )
        out.write("plan.json", plan)
        evidence.final_guard()
        print(
            json.dumps(dict(event="nodewise_five_minute_imported", through_frame=START)), flush=True
        )
        timers = dict(
            setup_seconds=time.perf_counter() - started_clock,
            reader_seconds=0.0,
            checkpoint_seconds=0.0,
            milestone_seconds=0.0,
            advance_seconds=0.0,
            native_next_seconds=0.0,
            diagnostic_save_seconds=0.0,
            guard_seconds=0.0,
        )
        batch = batching.NodewiseDiagnosticBatch(out, batch_size=BATCH_SIZE)
        checkpoints, all_rows = [], []

        def guard():
            qualified.check_headroom(prior.cuda_headroom())
            qualified.check_memory(qualified.qualification.memory_sample())

        for first, stop in chunks():
            tick = time.perf_counter()
            reader = original.SourceSequenceReader(
                inputs.chunk_roots[(first - FIRST) // CHUNK], inputs.input_root
            )
            require(
                np.array_equal(reader.frames, np.arange(first, stop)),
                "wrong complete native source chunk",
            )
            iterator = original.bounded_frames(reader, first=first, stop=stop)
            timers["reader_seconds"] += time.perf_counter() - tick
            try:
                rows, measured = drive_chunk(
                    tracker,
                    iterator,
                    batch,
                    first,
                    stop,
                    synchronize=torch.cuda.synchronize,
                    guard=guard,
                )
            finally:
                iterator.close()
            all_rows.extend(rows)
            for key, value in measured.items():
                timers[key] += value
            out.write(
                f"diagnostic_index/through_{stop:06d}.json",
                dict(rows=rows, process_interval=[first, stop]),
            )
            tick = time.perf_counter()
            path = destination / "checkpoints" / f"through_{stop:06d}"
            saved, meta, committed = base.preserve_checkpoint(tracker, parent, path, binding_new)
            tracker._decode_state(saved, meta)
            lifecycle(tracker, stop)
            # Use the explicit output root: the frozen Output implementation is not extended.
            for name in ("manifest.json", "metadata.json", "arrays.npz"):
                f = path / name
                out.records[f.relative_to(destination).as_posix()] = dict(
                    path=str(f), sha256=prior.checks.sha(f), size_bytes=f.stat().st_size
                )
            checkpoints.append(committed)
            timers["checkpoint_seconds"] += time.perf_counter() - tick
            if stop in MILESTONES:
                tick = time.perf_counter()
                interim = original.phase.COMMON.traced_candidate(saved)
                require(
                    np.array_equal(interim["frame_idx"], np.arange(FIRST, stop)),
                    "milestone extent differs",
                )
                member = out.write(f"milestones/through_{stop:06d}/candidate.npz", interim)
                out.write(
                    f"milestones/through_{stop:06d}/summary.json",
                    dict(
                        schema=STAGE + "_milestone",
                        status="ok",
                        horizon_stop=stop,
                        scientific_interval=[START, stop],
                        native_frames=stop - START,
                        candidate=member,
                        candidate_arrays=base.array_signature(interim),
                        checkpoint=committed,
                        initialize_count=1,
                        restore_count=4,
                        finish_calls_so_far=0,
                        new_begin_calls=0,
                        external_pose_resets=0,
                        all39_roundtrip=True,
                        prior_history_prefix_unchanged=True,
                        whole_process_completion_pending=True,
                        physical_ID_truth=False,
                    ),
                )
                timers["milestone_seconds"] += time.perf_counter() - tick
                del interim
            out.write(
                f"checkpoint_records/progress_{stop:06d}.json",
                dict(
                    checkpoints=checkpoints,
                    timings=timers.copy(),
                    through_frame=stop,
                    diagnostic_frames=batch.totalframes,
                ),
            )
            parent = dict(arrays=saved, metadata=meta)
            print(
                json.dumps(
                    dict(
                        event="nodewise_five_minute_checkpoint",
                        through_frame=stop,
                        science_processed=stop - START,
                    )
                ),
                flush=True,
            )
        require(
            batch.totalframes == batch.saved_frames == STOP - START and batch.pending_frames == 0,
            "complete49,800 committed diagnostic frames required",
        )
        candidate = tracker.finish()
        require(
            np.array_equal(candidate["frame_idx"], np.arange(FIRST, STOP)),
            "full66,400 frame backtrace required",
        )
        require(
            prior.compare_candidate(candidate, original.phase.COMMON.traced_candidate(saved), STOP)[
                "exact"
            ],
            "finish changed checkpoint backtrace",
        )
        member = out.write("candidate.npz", candidate)
        out.write(
            "diagnostic_batches.json",
            dict(
                schema=STAGE + "_batches",
                batches=batch.batches,
                native_frames=batch.totalframes,
                all_frames_saved=True,
            ),
        )
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                process_interval=[START, STOP],
                native_frames=STOP - START,
                scientific_interval=[START, STOP],
                scientific_native_frames=STOP - START,
                retained_interval=[FIRST, STOP],
                source_sha256=code,
                tracker_context=context,
                parent_tracking_sha256=PARENT_SHA,
                parent_checkpoint_sha256=CHECKPOINT_SHA,
                candidate=member,
                candidate_arrays=base.array_signature(candidate),
                checkpoints=checkpoints,
                checkpoint_count=len(checkpoints),
                initialize_count=1,
                restore_count=4,
                new_begin_calls=0,
                external_pose_resets=0,
                finish_calls=1,
                all39_roundtrip=True,
                completed_checkpoint_decoder_accepted=True,
                prior_history_prefix_unchanged=True,
                finish_matches_checkpoint_backtrace=True,
                diagnostic_frames=batch.totalframes,
                diagnostic_batch_size=BATCH_SIZE,
                body_row=summarize_proposals(all_rows, candidate),
                timings=timers,
                body_row_and_sparse_backend_effective_from_native=START,
                sixth_term_effective_from_native=76360,
                existing_five_parts_and_sixth_term_unchanged=True,
                mask_split_priority_applied=False,
                physical_ID_truth=False,
                independent_evaluation_pending=True,
                automatic_adoption=False,
            ),
        )
        store.final_guard()
        for path, digest in store.evidence_hashes.items():
            evidence.add(path, digest)
        out.write("mask_validation_receipts.json", store.uses)
        require(prior.local_code() == code, "source closure changed during run")
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
            committed_checkpoint=committed,
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

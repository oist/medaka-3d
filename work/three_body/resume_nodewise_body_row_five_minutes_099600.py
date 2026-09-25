"""Resume the interrupted Nodewise five-minute run at committed frame 99600.

All 39 tracker arrays, all retained history, velocities and review flags are
restored once. Native frames [99600,116200) are then processed serially on CUDA.
This is process recovery, not pose or identity reinitialization.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import run_nodewise_body_row_five_minutes as parent_run

base, prior, proof = parent_run.base, parent_run.prior, parent_run.proof
original, qualified, torch = parent_run.original, parent_run.qualified, parent_run.torch
WORK, REPO, require = parent_run.WORK, parent_run.REPO, parent_run.require
nodewise, batching = parent_run.variant, parent_run.batching

FIRST, SCIENCE_START, RESUME, STOP, CHUNK = 49800, 66400, 99600, 116200, 1660
STAGE = "nodewise_body_row_five_minute_resume_099600_v1"
PARENT = WORK / "nodewise_body_row_five_minute_continuation_v1"
CHECKPOINT_RELATIVE = "checkpoints/through_099600"
PINS = {
    "plan.json": "e05e174a291d6d79bf2cf3f74b848e7b125b3304f8ad1f0d168165dac3a8ba31",
    "input_index.json": "f5840d96f166f04516c72326aee157e78fac01a290296653aabdec01dd8ae568",
    "state_import.json": "fb68e6c59ef50452645f69ecd09aa19fbff8c6f5625742fb7335659d2baeb45e",
    "checkpoint_records/progress_099600.json": (
        "8fbb2331353c7c625c5cd7a6db6d5aa290ddbf59d1f23c26b1b5a0a5ebe068a9"
    ),
    CHECKPOINT_RELATIVE + "/manifest.json": (
        "60170ebec150caa385f911224a25fee77f615269077b47413a926f8fe6fc5cf1"
    ),
    CHECKPOINT_RELATIVE + "/metadata.json": (
        "6e6ab06ce6a23d8e5bcb159a241b30e6c5359a84b1f9b3c2b411cf2d0de38b5e"
    ),
}
BATCH_SIZE = 100


def chunks():
    return tuple((first, first + CHUNK) for first in range(RESUME, STOP, CHUNK))


def lifecycle(tracker, through):
    require(
        tracker.initialize_count == 1
        and tracker.restore_count == 5
        and tracker.previous_frame == through - 1
        and tracker.processed_frames == through - FIRST,
        "resumed full-state lifecycle differs",
    )


def source_code(evidence):
    code = prior.local_code()
    prior.source_contract(code)
    require(str(Path(__file__).resolve()) in code, "resume caller absent from source closure")
    for path, digest in code.items():
        evidence.add(path, digest)
    return code


def load_headers(evidence):
    documents = {}
    for relative, digest in PINS.items():
        path = PARENT / relative
        evidence.add(path, digest)
        documents[relative] = proof.read_json(path)
    plan = documents["plan.json"]
    index = documents["input_index.json"]
    imported = documents["state_import.json"]
    progress = documents["checkpoint_records/progress_099600.json"]
    manifest = documents[CHECKPOINT_RELATIVE + "/manifest.json"]
    metadata = documents[CHECKPOINT_RELATIVE + "/metadata.json"]
    last = progress["checkpoints"][-1]
    require(
        plan["schema"] == PARENT.name
        and plan["process_interval"] == [SCIENCE_START, STOP]
        and plan["retained_interval"] == [FIRST, STOP]
        and plan["initialize_count"] == 1
        and plan["restore_count"] == 4
        and plan["new_begin_calls"] == plan["external_pose_resets"] == 0,
        "interrupted parent plan differs",
    )
    require(
        imported["status"] == "ok"
        and imported["all_arrays_byte_preserved"] is True
        and imported["old_history_rewritten"] is False
        and imported["review_cleared_by_migration"] is False,
        "original 5:00 state import differs",
    )
    require(
        manifest["schema"] == "medaka_tracker_checkpoint_v1"
        and manifest["status"] == "ok"
        and len(manifest["arrays"]) == 39
        and manifest["bindings"] == plan["checkpoint_binding"]
        and manifest["metadata"]["sha256"] == PINS[CHECKPOINT_RELATIVE + "/metadata.json"]
        and manifest["payload"]["sha256"]
        == "36fd169edae49c494e471589a395ae1de3cd000da36a1fc071dafe400b7f8b8a"
        and manifest["payload"]["bytes"] == 84813066,
        "committed checkpoint manifest differs",
    )
    require(
        metadata["context"] == plan["tracker_context"]
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 4
        and metadata["previous_frame"] == RESUME - 1
        and metadata["processed_frames"] == RESUME - FIRST,
        "committed checkpoint lifecycle differs",
    )
    require(
        last["manifest_sha256"] == PINS[CHECKPOINT_RELATIVE + "/manifest.json"]
        and last["previous_frame"] == RESUME - 1
        and last["processed_frames"] == RESUME - FIRST
        and last["initialize_count"] == 1
        and last["restore_count"] == 4,
        "progress record does not end at committed checkpoint",
    )
    require(
        len(index["prefix_chunks"]) == 40
        and index["original_binding"]
        == plan["original_binding"]
        == manifest["bindings"]["original_binding"],
        "registered full source index differs",
    )
    return dict(plan=plan, index=index, manifest=manifest, metadata=metadata)


def load_payload(headers, evidence):
    path = PARENT / CHECKPOINT_RELATIVE / "arrays.npz"
    descriptor = headers["manifest"]["payload"]
    require(path.stat().st_size == descriptor["bytes"], "checkpoint payload size changed")
    evidence.add(path, descriptor["sha256"])
    arrays, metadata, manifest = prior.read_checkpoint(
        PARENT / CHECKPOINT_RELATIVE,
        expected_bindings=headers["manifest"]["bindings"],
    )
    require(
        len(arrays) == 39 and metadata == headers["metadata"] and manifest == headers["manifest"],
        "checkpoint decode differs from verified headers",
    )
    return arrays, metadata


def restore_exact(tracker, arrays, metadata):
    before_meta = copy.deepcopy(metadata)
    before = base.array_signature(arrays)
    tracker.restore_state(arrays, metadata)
    restored, current = tracker.export_state()
    expected = copy.deepcopy(before_meta)
    expected["restore_count"] += 1
    require(
        metadata == before_meta
        and base.array_signature(arrays) == before
        and current == expected
        and set(restored) == set(arrays)
        and all(original.byte_equal(restored[k], arrays[k]) for k in arrays),
        "restore changed checkpoint bytes or metadata beyond restore count",
    )
    lifecycle(tracker, RESUME)
    return dict(
        all39_byte_exact=True,
        metadata_change="restore_count:4->5",
        identity_or_pose_reinitialization=False,
        velocities_history_previous_images_and_review_preserved=True,
    )


def drive_chunk(tracker, iterator, batch, first, stop, *, guard):
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
        torch.cuda.synchronize()
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
                json.dumps(dict(event="nodewise_resume_progress", through_frame=native + 1)),
                flush=True,
            )
    tick = time.perf_counter()
    batch.flush()
    timings["diagnostic_save_seconds"] += time.perf_counter() - tick
    require([r["frame_idx"] for r in rows] == list(range(first, stop)), "diagnostic gap")
    return rows, timings


def main(args):
    destination = args.output.resolve()
    require(destination.is_relative_to(WORK) and not destination.exists(), "fresh output required")
    out, evidence = proof.Output(destination), original.phase.Digests()
    started, committed, code = proof.utc(), None, {}
    started_clock = time.perf_counter()
    try:
        code = source_code(evidence)
        for path in sorted(code):
            out.write(
                "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), Path(path).read_bytes()
            )
        headers = load_headers(evidence)
        arrays, metadata = load_payload(headers, evidence)
        binding = headers["manifest"]["bindings"]["original_binding"]
        store = qualified.ReceiptStore(args.receipt_root.resolve())
        inputs, protocol = original.verified_inputs(binding, STOP, evidence, receipt_store=store)
        require(
            len(inputs.chunk_roots) == len(inputs.chunk_records) == 40
            and inputs.chunk_records == headers["index"]["prefix_chunks"],
            "registered 5:00-11:40 inputs differ",
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
        require(config.device == "cuda", "registered CUDA required")
        tracker = nodewise.BodyRowContinuousTracker(
            inputs.rig,
            inputs.initial_pose_xyz.copy(),
            config,
            recovery_config=qualified.NodeRecoveryConfig(**protocol["node_recovery_config"]),
            persistence_config=qualified.PersistentNodeConfig(**protocol["persistence_config"]),
            sparse_effective_from_native=SCIENCE_START,
            body_row_effective_from_native=SCIENCE_START,
        )
        restored = restore_exact(tracker, arrays, metadata)
        context = tracker._context()
        require(context == metadata["context"], "tracker context changed at process recovery")
        out.write("state_restore.json", restored)
        checkpoint_binding = dict(
            schema=STAGE + "_checkpoint",
            original_binding=binding,
            source_sha256=code,
            interrupted_parent_checkpoint_sha256=PINS[CHECKPOINT_RELATIVE + "/manifest.json"],
            retained_interval=[FIRST, STOP],
            process_interval=[RESUME, STOP],
            scientific_interval=[SCIENCE_START, STOP],
            tracker_context=context,
        )
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                source_sha256=code,
                checkpoint_binding=checkpoint_binding,
                process_interval=[RESUME, STOP],
                retained_interval=[FIRST, STOP],
                scientific_interval=[SCIENCE_START, STOP],
                tracker_context=context,
                process_recovery_from_committed_frame=RESUME,
                initialize_count=1,
                restore_count=5,
                new_begin_calls=0,
                external_pose_resets=0,
                identity_or_pose_reinitialization=False,
                environment=environment,
                precision="float64",
                allocator_fraction=0.25,
                diagnostic_batch_size=BATCH_SIZE,
                sixth_term_effective_from_native=76360,
                physical_ID_truth=False,
                interrupted_uncommitted_frames_discarded=[RESUME, 100500],
            ),
        )
        evidence.final_guard()
        print(json.dumps(dict(event="nodewise_resume_imported", through_frame=RESUME)), flush=True)
        timers = dict(
            setup_seconds=time.perf_counter() - started_clock,
            reader_seconds=0.0,
            checkpoint_seconds=0.0,
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

        parent = dict(arrays=arrays, metadata=metadata)
        saved = arrays
        for first, stop in chunks():
            tick = time.perf_counter()
            reader = original.SourceSequenceReader(
                inputs.chunk_roots[(first - FIRST) // CHUNK], inputs.input_root
            )
            require(np.array_equal(reader.frames, np.arange(first, stop)), "wrong source chunk")
            iterator = original.bounded_frames(reader, first=first, stop=stop)
            timers["reader_seconds"] += time.perf_counter() - tick
            try:
                rows, measured = drive_chunk(tracker, iterator, batch, first, stop, guard=guard)
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
            saved, saved_meta, committed = base.preserve_checkpoint(
                tracker, parent, path, checkpoint_binding
            )
            tracker._decode_state(saved, saved_meta)
            lifecycle(tracker, stop)
            for name in ("manifest.json", "metadata.json", "arrays.npz"):
                item = path / name
                out.records[item.relative_to(destination).as_posix()] = dict(
                    path=str(item), sha256=prior.checks.sha(item), size_bytes=item.stat().st_size
                )
            checkpoints.append(committed)
            timers["checkpoint_seconds"] += time.perf_counter() - tick
            out.write(
                f"checkpoint_records/progress_{stop:06d}.json",
                dict(
                    checkpoints=checkpoints,
                    timings=timers.copy(),
                    through_frame=stop,
                    diagnostic_frames=batch.totalframes,
                ),
            )
            parent = dict(arrays=saved, metadata=saved_meta)
            print(
                json.dumps(dict(event="nodewise_resume_checkpoint", through_frame=stop)), flush=True
            )
        require(
            batch.totalframes == batch.saved_frames == STOP - RESUME and batch.pending_frames == 0,
            "complete resumed diagnostic coverage required",
        )
        candidate = tracker.finish()
        require(
            np.array_equal(candidate["frame_idx"], np.arange(FIRST, STOP)),
            "full retained backtrace required",
        )
        require(
            prior.compare_candidate(candidate, original.phase.COMMON.traced_candidate(saved), STOP)[
                "exact"
            ],
            "finish changed final checkpoint backtrace",
        )
        candidate_record = out.write("candidate.npz", candidate)
        out.write(
            "diagnostic_batches.json",
            dict(
                schema=STAGE + "_batches",
                batches=batch.batches,
                native_frames=batch.totalframes,
                all_frames_saved=True,
            ),
        )
        mask = (candidate["frame_idx"] >= RESUME) & (candidate["frame_idx"] < STOP)
        added = [r["body_row"]["admitted"] for r in all_rows]
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                process_interval=[RESUME, STOP],
                native_frames=STOP - RESUME,
                scientific_interval=[SCIENCE_START, STOP],
                retained_interval=[FIRST, STOP],
                source_sha256=code,
                tracker_context=context,
                candidate=candidate_record,
                candidate_arrays=base.array_signature(candidate),
                checkpoints=checkpoints,
                checkpoint_count=len(checkpoints),
                initialize_count=1,
                restore_count=5,
                new_begin_calls=0,
                external_pose_resets=0,
                finish_calls=1,
                identity_or_pose_reinitialization=False,
                all39_roundtrip=True,
                finish_matches_checkpoint_backtrace=True,
                diagnostic_frames=batch.totalframes,
                body_row=dict(
                    total_admitted=sum(added),
                    frames_with_additions=sum(v > 0 for v in added),
                    final_selected_body_row_frame_count=int(
                        np.sum(candidate["proposal_kind"][mask] == "body_row")
                    ),
                ),
                timings=timers,
                sixth_term_effective_from_native=76360,
                full_five_minute_path_recovered=True,
                physical_ID_truth=False,
                independent_evaluation_pending=True,
                automatic_adoption=False,
            ),
        )
        store.final_guard()
        for path, digest in store.evidence_hashes.items():
            evidence.add(path, digest)
        out.write("mask_validation_receipts.json", store.uses)
        require(prior.local_code() == code, "source closure changed during recovery")
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
        destination.mkdir(parents=True, exist_ok=True)
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

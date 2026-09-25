"""Continue the audited committed minute to five minutes without ID reinitialization.

Restore all39 saved fields once at76360, then keep one tracker through116200.
Pure retained-history backtraces are published at each minute; only the final
endpoint calls finish. Process restoration is reported separately from resets.
No FullGraph/padding/mixed fitter, detector, ID correction or scientific change.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import audit_fast_tail_partial_minute as audit
import numpy as np
from directional_chest_tracker import array_signature

prior, proof = audit.prior, audit.proof
original, qualified, torch = prior.original, prior.qualified, prior.torch
WORK, REPO, require = prior.WORK, prior.REPO, proof.require
FIRST, START, RESUME, STOP, CHUNK = 49800, 66400, 76360, 116200, 1660
STAGE = "fast_tail_five_minute_continuation_v1"
AUDIT_ROOT = WORK / "fast_tail_interrupted_minute_checkpoint_audit_v1"
MILESTONES = (86320, 96280, 106240, 116200)


def lifecycle(tracker, stop):
    require(
        tracker.initialize_count == 1
        and tracker.restore_count == 5
        and tracker.previous_frame == stop - 1
        and tracker.processed_frames == stop - FIRST,
        "full retained tracking state/lifecycle changed",
    )


def restore_exact(tracker, arrays, metadata):
    require(tracker.initialize_count == 0, "fresh never-initialized target required")
    expected = copy.deepcopy(metadata)
    signatures = array_signature(arrays)
    tracker.restore_state(arrays, metadata)
    restored, current = tracker.export_state()
    require(
        metadata == expected and array_signature(arrays) == signatures,
        "restore mutated checkpoint inputs",
    )
    expected["restore_count"] += 1
    require(
        set(restored) == set(arrays)
        and len(arrays) == 39
        and all(original.byte_equal(restored[k], arrays[k]) for k in arrays)
        and current == expected,
        "restore changed state bytes or unexpected metadata",
    )
    lifecycle(tracker, RESUME)
    return dict(
        all39_BYTEEXACT=True,
        metadata_only_change="restore_count:4->5",
        initializes=1,
        new_begin_calls=0,
        external_pose_resets=0,
        restored_previous_native_frame=RESUME - 1,
        previous_images_velocity_all_beams_and_history_preserved=True,
    )


def drive(tracker, frames, first, stop, *, synchronize, guard, progress):
    """Strict serial native loop; no begin, restore or finish calls in this function."""
    advance_seconds = decode_seconds = 0.0
    for native in range(first, stop):
        lifecycle(tracker, native)
        tick = time.perf_counter()
        frame = next(frames)
        decode_seconds += time.perf_counter() - tick
        require(frame.frame_idx == native, "native image skipped or reordered")
        guard()
        tick = time.perf_counter()
        tracker.advance(frame)
        synchronize()
        advance_seconds += time.perf_counter() - tick
        lifecycle(tracker, native + 1)
        guard()
        if (native + 1 - RESUME) % 100 == 0 or native + 1 == stop:
            progress(native + 1)
    return dict(advance_seconds=advance_seconds, native_next_seconds=decode_seconds)


def preserve_checkpoint(tracker, parent, path, binding):
    """Preserve the actual parent history length, including every retained beam."""
    arrays, metadata = tracker.export_state()
    old = parent["arrays"]
    offsets = old["history_offsets"]
    require(
        len(offsets) == parent["metadata"]["processed_frames"] + 1
        and len(arrays["history_offsets"]) == metadata["processed_frames"] + 1
        and len(arrays["history_offsets"]) >= len(offsets),
        "checkpoint retained frame extent differs",
    )
    for key in ("history_offsets", *("history_" + f for f in qualified.RECORD_FIELDS)):
        end = len(offsets) if key == "history_offsets" else int(offsets[-1])
        actual, expected = arrays[key][:end], old[key]
        exact = (
            actual.shape == expected.shape and np.array_equal(actual, expected)
            if actual.dtype.kind == expected.dtype.kind == "U"
            else original.byte_equal(actual, expected)
        )
        require(exact, "retained history prefix changed: " + key)
    require(len(arrays) == 39, "complete39 checkpoint required")
    prior.write_checkpoint(path, arrays, metadata, bindings=binding)
    loaded, loaded_meta, manifest = prior.read_checkpoint(path, expected_bindings=binding)
    require(
        loaded_meta == metadata
        and set(loaded) == set(arrays)
        and all(original.byte_equal(loaded[k], v) for k, v in arrays.items()),
        "checkpoint roundtrip differs",
    )
    return (
        loaded,
        loaded_meta,
        dict(
            path=str(path),
            arrays=39,
            manifest_sha256=prior.checks.sha(path / "manifest.json"),
            payload_bytes=manifest["payload"]["bytes"],
            **{
                k: metadata[k]
                for k in ("previous_frame", "processed_frames", "initialize_count", "restore_count")
            },
        ),
    )


def load_parent(audit_sha, evidence):
    seal_path = AUDIT_ROOT / "audit.provenance.json"
    evidence.add(seal_path, audit_sha)
    seal = proof.read_json(seal_path)
    require(
        seal["stage"] == audit.STAGE and seal["status"] == "ok", "successful parent audit required"
    )
    audit_source = str(Path(audit.__file__).resolve())
    evidence.add(audit_source, seal["inputs"][audit_source])
    row = seal["outputs"]["summary.json"]
    evidence.add(row["path"], row["sha256"])
    summary = proof.read_json(row["path"])
    require(
        summary["eligible_for_exact_state_restore"] is True
        and summary["checkpoint_manifest_sha256"] == audit.CHECKPOINT_SHA
        and summary["committed_interval"] == [START, RESUME],
        "wrong audited checkpoint",
    )
    evidence.add(audit.CHECKPOINT / "manifest.json", audit.CHECKPOINT_SHA)
    manifest = proof.read_json(audit.CHECKPOINT / "manifest.json")
    for field in ("metadata", "payload"):
        path = audit.CHECKPOINT / manifest[field]["file"]
        require(
            seal["inputs"].get(str(path)) == manifest[field]["sha256"],
            "audit payload input differs",
        )
        evidence.add(path, manifest[field]["sha256"])
    for path, digest in manifest["bindings"]["experimental_code_sha256"].items():
        require(seal["inputs"].get(path) == digest, "unaudited source")
        evidence.add(path, digest)
    arrays, metadata, _ = prior.read_checkpoint(
        audit.CHECKPOINT, expected_bindings=manifest["bindings"]
    )
    return arrays, metadata, manifest["bindings"]["original_binding"]


def main(args):
    require(
        args.execute and proof.valid_sha(args.audit_sha), "explicit execution/audit SHA required"
    )
    destination = args.output.resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(), "new work output required"
    )
    out, evidence = proof.Output(destination), original.phase.Digests()
    store = qualified.ReceiptStore(args.receipt_root.resolve())
    code = prior.local_code()
    prior.source_contract(code)
    started, tick = proof.utc(), time.perf_counter()
    tracker, checkpoints = None, []
    timings = dict(
        prepare_seconds=0.0,
        reader_ctor_seconds=0.0,
        advance_seconds=0.0,
        native_next_seconds=0.0,
        checkpoint_seconds=0.0,
    )
    try:
        print(
            json.dumps(dict(event="five_minute_validation_started", output=str(destination))),
            flush=True,
        )
        for path, digest in code.items():
            evidence.add(path, digest)
            out.write(
                "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), Path(path).read_bytes()
            )
        arrays, metadata, binding = load_parent(args.audit_sha, evidence)
        print(
            json.dumps(
                dict(event="audited_parent_loaded", through_frame=RESUME, fields=len(arrays))
            ),
            flush=True,
        )
        inputs, protocol = original.verified_inputs(binding, STOP, evidence, receipt_store=store)
        require(
            len(inputs.chunk_records) == 40 and len(inputs.chunk_roots) == 40,
            "registered400s input prefix required for 300s evaluation interval",
        )
        print(
            json.dumps(dict(event="five_minute_inputs_verified", chunks=len(inputs.chunk_records))),
            flush=True,
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
        tracker = prior.FastTailRecoveryContinuousTracker(
            inputs.rig,
            inputs.initial_pose_xyz.copy(),
            config,
            recovery_config=qualified.NodeRecoveryConfig(**protocol["node_recovery_config"]),
            persistence_config=qualified.PersistentNodeConfig(**protocol["persistence_config"]),
        )
        restored = restore_exact(tracker, arrays, metadata)
        out.write("restore.json", restored)
        parent = dict(arrays=arrays, metadata=metadata)
        checkpoint_binding = dict(
            schema=STAGE + "_checkpoint",
            original_binding=binding,
            source_sha256=code,
            audited_parent_sha256=args.audit_sha,
            parent_checkpoint_sha256=audit.CHECKPOINT_SHA,
            scientific_interval=[START, STOP],
            process_interval=[RESUME, STOP],
        )
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                scientific_interval=[START, STOP],
                process_interval=[RESUME, STOP],
                milestones=list(MILESTONES),
                original_binding=binding,
                checkpoint_binding=checkpoint_binding,
                source_sha256=code,
                environment=environment,
                precision="float64",
                allocator_fraction=0.25,
                reserved_limit_bytes=4 * 1024**3,
                frame_free_MiB=2048,
                initialize_count=1,
                restore_count=5,
                new_begin_calls=0,
                previous_process_incomplete=True,
                external_pose_resets=0,
                process_restore_is_not_ID_reinitialization=True,
                first_minute_audit_sha256=args.audit_sha,
                physical_ID_truth=False,
                automatic_adoption=False,
            ),
        )
        evidence.final_guard()
        timings["prepare_seconds"] = time.perf_counter() - tick
        print(json.dumps(dict(event="restored_for_five_minutes", **restored)), flush=True)

        def guard():
            qualified.check_headroom(prior.cuda_headroom())
            qualified.check_memory(qualified.qualification.memory_sample())

        def progress(stop):
            print(
                json.dumps(
                    dict(
                        event="five_minute_progress",
                        through_frame=stop,
                        completed_this_process=stop - RESUME,
                        total_this_process=STOP - RESUME,
                        total_scientific_frames=stop - START,
                        target_scientific_frames=STOP - START,
                    )
                ),
                flush=True,
            )

        for first in range(RESUME, STOP, CHUNK):
            stop = first + CHUNK
            tick = time.perf_counter()
            reader = original.SourceSequenceReader(
                inputs.chunk_roots[(first - FIRST) // CHUNK], inputs.input_root
            )
            require(
                reader.frames[0] == first and reader.frames[-1] + 1 == stop, "wrong reader chunk"
            )
            iterator = original.bounded_frames(reader, first=first, stop=stop)
            timings["reader_ctor_seconds"] += time.perf_counter() - tick
            try:
                measured = drive(
                    tracker,
                    iterator,
                    first,
                    stop,
                    synchronize=torch.cuda.synchronize,
                    guard=guard,
                    progress=progress,
                )
                for key, value in measured.items():
                    timings[key] += value
            finally:
                iterator.close()
            tick = time.perf_counter()
            path = destination / "checkpoints" / f"through_{stop:06d}"
            saved, meta, row = preserve_checkpoint(tracker, parent, path, checkpoint_binding)
            lifecycle(tracker, stop)
            checkpoints.append(row)
            for name in ("manifest.json", "metadata.json", "arrays.npz"):
                out.write(
                    f"checkpoint_records/{stop:06d}_{name}.json",
                    dict(
                        path=str(path / name),
                        sha256=prior.checks.sha(path / name),
                        size_bytes=(path / name).stat().st_size,
                    ),
                )
                # Actual checkpoint files are also included in the final output inventory.
                out.records[(path / name).relative_to(destination).as_posix()] = dict(
                    path=str(path / name),
                    sha256=prior.checks.sha(path / name),
                    size_bytes=(path / name).stat().st_size,
                )
            if stop in MILESTONES:
                candidate = original.phase.COMMON.traced_candidate(saved)
                require(
                    np.array_equal(candidate["frame_idx"], np.arange(FIRST, stop)),
                    "backtrace extent",
                )
                member = out.write(f"milestones/through_{stop:06d}/candidate.npz", candidate)
                out.write(
                    f"milestones/through_{stop:06d}/summary.json",
                    dict(
                        schema=STAGE + "_milestone",
                        status="ok",
                        horizon_stop=stop,
                        scientific_interval=[START, stop],
                        native_frames=stop - START,
                        candidate=member,
                        checkpoint=row,
                        initialize_count=1,
                        restore_count=5,
                        source_sha256=code,
                        parent_audit_sha256=args.audit_sha,
                        original_binding=binding,
                        external_pose_resets=0,
                        finish_calls_so_far=0,
                        all39_roundtrip=True,
                        prior_history_prefix_unchanged=True,
                        physical_ID_truth=False,
                        five_minute_candidate=stop == STOP,
                        whole_process_completion_pending=True,
                    ),
                )
            out.write(
                f"checkpoint_records/progress_{stop:06d}.json",
                dict(checkpoints=checkpoints, timings=timings.copy()),
            )
            timings["checkpoint_seconds"] += time.perf_counter() - tick
            parent = dict(arrays=saved, metadata=meta)
            print(json.dumps(dict(event="five_minute_checkpoint", through_frame=stop)), flush=True)
        candidate = tracker.finish()
        require(
            prior.compare_candidate(candidate, original.phase.COMMON.traced_candidate(saved), STOP)[
                "exact"
            ],
            "final finish changed history backtrace",
        )
        out.write("candidate.npz", candidate)
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                scientific_interval=[START, STOP],
                process_interval=[RESUME, STOP],
                native_frames=STOP - START,
                process_frames=STOP - RESUME,
                initialize_count=1,
                restore_count=5,
                external_pose_resets=0,
                new_begin_calls=0,
                finish_calls=1,
                checkpoints=checkpoints,
                timings=timings,
                physical_ID_truth=False,
                independent_evaluation_pending=True,
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
            committed_checkpoints=checkpoints,
        )
        with (destination / "tracking.provenance.json").open("xb") as stream:
            stream.write(proof.json_bytes(seal))
        raise
    with (destination / "tracking.provenance.json").open("xb") as stream:
        stream.write(proof.json_bytes(seal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-sha", required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    main(parser.parse_args())

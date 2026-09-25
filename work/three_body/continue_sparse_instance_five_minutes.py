"""Continue the instance-scored branch to five scientific minutes without resets.

The first two new frames also run a dense control. Require identical39 temporal
arrays before releasing that control and continuing the same sparse tracker.
The existing mask scoring policy is unchanged; no semantic landmark term enters.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import continue_fast_tail_five_minutes as base
import instance_rescore_tracker as dense
import numpy as np
import run_instance_rescore_10s as diagnostic
import sparse_instance_rescore_tracker as sparse

prior, proof, original, qualified = base.prior, base.proof, base.original, base.qualified
torch, WORK, REPO, require = base.torch, base.WORK, base.REPO, base.require
FIRST, START, RESUME, STOP, CHUNK = 49800, 66400, 78020, 116200, 1660
MILESTONES = (86320, 96280, 106240, 116200)
STAGE = "sparse_instance_five_minute_continuation_v1"
PARENT = WORK / "instance_rescore_ten_second_continuation_v1"
PARENT_SHA = "a71ddab8dfef8d97fecf68c1d5b7a5d7300fe21dfe49b48bc6f1df91c840196a"
QUALIFICATION = WORK / "sparse_instance_saved_ten_second_qualification_v1"
QUALIFICATION_SHA = "941ca436a5e59fca5a63794483d315da3950402aafdb4ce13c34630e98014125"
PINS = {
    "instance_body_score.py": (
        "cb9e04a20160f2b681aa1b7cca46ed5b4017c7b05760f9dbbeff691bf058cb87"
    ),
    "continue_fast_tail_five_minutes.py": (
        "d90e9eb49038dac4fd1e18068334ba89eb8df1a813a37c6aec071c20db1a61a6"
    ),
    "instance_rescore_tracker.py": (
        "f625513ed379e17241ad120c8a454c56b398764a1341ab7756790062231863e5"
    ),
    "run_instance_rescore_10s.py": (
        "91a66646f89a6abb59b6d60bc1b357b383167c21eb9d40bc47491802d05e9544"
    ),
    "sparse_instance_body_score.py": (
        "68c41510d64a41098c6c1daf5e98621723f37c2f0f35730a68bd0e896517a634"
    ),
}
EXECUTION_READY = True


def lifecycle(tracker, through):
    require(
        tracker.initialize_count == 1
        and tracker.restore_count == 6
        and tracker.previous_frame == through - 1
        and tracker.processed_frames == through - FIRST,
        "single-initialization continuous lifecycle changed",
    )


def bound_member(root, seal, name, evidence):
    path = (root / name).resolve(strict=True)
    require(path.is_relative_to(root), "member escaped run")
    row = seal["outputs"][name]
    require(
        path == Path(row["path"]).resolve() and path.stat().st_size == row["size_bytes"],
        "member metadata differs",
    )
    evidence.add(path, row["sha256"])
    return path


def load_parent(evidence):
    evidence.add(PARENT / "tracking.provenance.json", PARENT_SHA)
    seal = proof.read_json(PARENT / "tracking.provenance.json")
    require(
        seal["status"] == "ok" and seal["stage"] == PARENT.name,
        "successful instance parent required",
    )
    plan = proof.read_json(bound_member(PARENT, seal, "plan.json", evidence))
    summary = proof.read_json(bound_member(PARENT, seal, "summary.json", evidence))
    require(
        summary["process_interval"] == [76360, RESUME]
        and summary["retained_interval"] == [FIRST, RESUME]
        and summary["initialize_count"] == 1
        and summary["restore_count"] == 5
        and summary["all39_roundtrip"]
        and summary["prior_history_prefix_unchanged"]
        and summary["finish_matches_checkpoint_backtrace"],
        "parent scope/lifecycle differs",
    )
    checkpoint = PARENT / "checkpoints/through_078020"
    for name in ("manifest.json", "metadata.json", "arrays.npz"):
        bound_member(PARENT, seal, "checkpoints/through_078020/" + name, evidence)
    require(
        prior.checks.sha(checkpoint / "manifest.json") == summary["checkpoint_manifest_sha256"],
        "parent checkpoint edge",
    )
    arrays, metadata, manifest = prior.read_checkpoint(
        checkpoint, expected_bindings=plan["checkpoint_binding"]
    )
    require(
        len(arrays) == 39
        and metadata["previous_frame"] == RESUME - 1
        and metadata["processed_frames"] == RESUME - FIRST
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 5,
        "actual parent state differs",
    )
    return (
        arrays,
        metadata,
        manifest["bindings"]["original_binding"],
        summary["checkpoint_manifest_sha256"],
    )


def compare_control(tracker, control):
    """Full state equality; only the explicitly declared backend context differs."""
    arrays, metadata = tracker.export_state()
    expected, original_metadata = control.export_state()
    require(set(arrays) == set(expected) and len(arrays) == 39, "full39 control fields")
    require(
        all(original.byte_equal(arrays[k], expected[k]) for k in arrays),
        "dense/sparse serial state differs",
    )
    adapted = copy.deepcopy(metadata)
    backend = adapted["context"].pop("sparse_distance_backend")
    adapted["context"]["implementation"] = original_metadata["context"]["implementation"]
    require(adapted == original_metadata, "undeclared serial metadata difference")
    return dict(
        status="ok",
        native_interval=[RESUME, RESUME + 2],
        all39_BYTEEXACT=True,
        compared_arrays=base.array_signature(arrays),
        backend=backend,
        metadata_difference="declared backend context and implementation only",
        initialize_count=tracker.initialize_count,
        restore_count=tracker.restore_count,
        full_five_minute_equivalence_proven=False,
        physical_ID_truth=False,
    )


def drive_chunk(tracker, control, iterator, first, stop, out, *, synchronize, guard):
    rows = []
    timings = dict(advance_seconds=0.0, native_next_seconds=0.0, diagnostic_save_seconds=0.0)
    for frame in range(first, stop):
        lifecycle(tracker, frame)
        tick = time.perf_counter()
        observation = next(iterator)
        timings["native_next_seconds"] += time.perf_counter() - tick
        require(observation.frame_idx == frame, "native frame skipped or reordered")
        guard()
        tick = time.perf_counter()
        tracker.advance(observation)
        synchronize()
        timings["advance_seconds"] += time.perf_counter() - tick
        lifecycle(tracker, frame + 1)
        tick = time.perf_counter()
        rows.append(diagnostic.save_diagnostic(out, tracker, frame))
        timings["diagnostic_save_seconds"] += time.perf_counter() - tick
        if control is not None:
            lifecycle(control, frame)
            control.advance(observation)
            synchronize()
            lifecycle(control, frame + 1)
            if frame + 1 == RESUME + 2:
                out.write(
                    "serial_dense_sparse_qualification.json", compare_control(tracker, control)
                )
                control = None
                gc.collect()
                torch.cuda.empty_cache()
        guard()
        if (frame + 1 - RESUME) % 100 == 0 or frame + 1 == stop:
            print(
                json.dumps(dict(event="sparse_instance_progress", through_frame=frame + 1)),
                flush=True,
            )
    return control, rows, timings


def main(args):
    require(
        EXECUTION_READY and args.execute and proof.valid_sha(args.tracker_sha),
        "reviewed execution and tracker SHA required",
    )
    destination = args.output.resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(),
        "fresh workspace output required",
    )
    out, evidence = proof.Output(destination), original.phase.Digests()
    store = qualified.ReceiptStore(args.receipt_root.resolve())
    started, checkpoints, code = proof.utc(), [], {}
    try:
        for name, sha in PINS.items():
            evidence.add(WORK / name, sha)
        evidence.add(Path(sparse.__file__), args.tracker_sha)
        evidence.add(QUALIFICATION / "qualification.provenance.json", QUALIFICATION_SHA)
        qseal = proof.read_json(QUALIFICATION / "qualification.provenance.json")
        require(qseal["status"] == "ok", "full saved10s sparse qualification required")
        code = prior.local_code()
        prior.source_contract(code)
        for path, sha in code.items():
            evidence.add(path, sha)
            out.write(
                "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), Path(path).read_bytes()
            )
        arrays, metadata, binding, parent_sha = load_parent(evidence)
        inputs, protocol = original.verified_inputs(binding, STOP, evidence, receipt_store=store)
        require(
            len(inputs.chunk_roots) == len(inputs.chunk_records) == 40,
            "full400s source prefix required",
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
        kwargs = dict(
            recovery_config=qualified.NodeRecoveryConfig(**protocol["node_recovery_config"]),
            persistence_config=qualified.PersistentNodeConfig(**protocol["persistence_config"]),
        )
        tracker = sparse.SparseInstanceRescoreContinuousTracker(
            inputs.rig, inputs.initial_pose_xyz.copy(), config, **kwargs
        )
        control = dense.InstanceRescoreContinuousTracker(
            inputs.rig, inputs.initial_pose_xyz.copy(), config, **kwargs
        )
        imported = sparse.import_dense_instance_state(tracker, control, arrays, metadata)
        # Migration's source decoder validates without restoring its source object.
        require(control.initialize_count == 0, "control source unexpectedly initialized")
        control.restore_state(arrays, metadata)
        lifecycle(tracker, RESUME)
        lifecycle(control, RESUME)
        out.write("state_import.json", imported)
        checkpoint_binding = dict(
            schema=STAGE + "_checkpoint",
            original_binding=binding,
            source_sha256=code,
            parent_tracking_sha256=PARENT_SHA,
            parent_checkpoint_sha256=parent_sha,
            scientific_interval=[START, STOP],
            process_interval=[RESUME, STOP],
            scoring_policy=tracker._context()["instance_rescore_policy"],
            sparse_distance_backend=tracker._context()["sparse_distance_backend"],
        )
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                scientific_interval=[START, STOP],
                retained_interval=[FIRST, STOP],
                process_interval=[RESUME, STOP],
                milestones=list(MILESTONES),
                original_binding=binding,
                checkpoint_binding=checkpoint_binding,
                source_sha256=code,
                environment=environment,
                precision="float64",
                allocator_fraction=0.25,
                reserved_limit_bytes=4 * 1024**3,
                initialize_count=1,
                restore_count=6,
                new_begin_calls=0,
                external_pose_resets=0,
                dense_serial_control_interval=[RESUME, RESUME + 2],
                sparse_rollout_continues_after_control_without_restore=True,
                mask_policy_effective_from_native=76360,
                sparse_backend_effective_from_native=RESUME,
                semantic_landmark_term_used=False,
                physical_ID_truth=False,
                automatic_adoption=False,
            ),
        )
        evidence.final_guard()
        print(json.dumps(dict(event="sparse_instance_imported", through_frame=RESUME)), flush=True)

        def guard():
            qualified.check_headroom(prior.cuda_headroom())
            qualified.check_memory(qualified.qualification.memory_sample())

        parent = dict(arrays=arrays, metadata=metadata)
        timings = dict(advance_seconds=0.0, native_next_seconds=0.0, diagnostic_save_seconds=0.0)
        for first in range(RESUME, STOP, CHUNK):
            stop = first + CHUNK
            reader = original.SourceSequenceReader(
                inputs.chunk_roots[(first - FIRST) // CHUNK], inputs.input_root
            )
            require(
                reader.frames[0] == first and reader.frames[-1] + 1 == stop, "wrong reader chunk"
            )
            iterator = original.bounded_frames(reader, first=first, stop=stop)
            try:
                segments = (
                    [(first, RESUME + 2), (RESUME + 2, stop)]
                    if first == RESUME
                    else [(first, stop)]
                )
                rows = []
                for left, right in segments:
                    control, part, measured = drive_chunk(
                        tracker,
                        control,
                        iterator,
                        left,
                        right,
                        out,
                        synchronize=torch.cuda.synchronize,
                        guard=guard,
                    )
                    rows.extend(part)
                    for key, value in measured.items():
                        timings[key] += value
                    if left == RESUME:
                        # The caller's final control reference has now been
                        # released too; continue this iterator and sparse state.
                        gc.collect()
                        torch.cuda.empty_cache()
                        guard()
            finally:
                iterator.close()
            out.write(
                f"diagnostic_index/through_{stop:06d}.json",
                dict(rows=rows, process_interval=[first, stop]),
            )
            path = destination / "checkpoints" / f"through_{stop:06d}"
            saved, meta, row = base.preserve_checkpoint(tracker, parent, path, checkpoint_binding)
            lifecycle(tracker, stop)
            checkpoints.append(row)
            for name in ("manifest.json", "metadata.json", "arrays.npz"):
                payload = path / name
                out.records[payload.relative_to(destination).as_posix()] = dict(
                    path=str(payload),
                    sha256=prior.checks.sha(payload),
                    size_bytes=payload.stat().st_size,
                )
            if stop in MILESTONES:
                candidate = original.phase.COMMON.traced_candidate(saved)
                require(
                    np.array_equal(candidate["frame_idx"], np.arange(FIRST, stop)),
                    "milestone backtrace extent",
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
                        restore_count=6,
                        source_sha256=code,
                        parent_tracking_sha256=PARENT_SHA,
                        original_binding=binding,
                        external_pose_resets=0,
                        finish_calls_so_far=0,
                        all39_roundtrip=True,
                        prior_history_prefix_unchanged=True,
                        physical_ID_truth=False,
                        whole_process_completion_pending=True,
                    ),
                )
            out.write(
                f"checkpoint_records/progress_{stop:06d}.json",
                dict(checkpoints=checkpoints, timings=timings.copy()),
            )
            parent = dict(arrays=saved, metadata=meta)
            print(
                json.dumps(dict(event="sparse_instance_checkpoint", through_frame=stop)), flush=True
            )
        candidate = tracker.finish()
        require(
            prior.compare_candidate(candidate, original.phase.COMMON.traced_candidate(saved), STOP)[
                "exact"
            ],
            "finish changed saved backtrace",
        )
        out.write("candidate.npz", candidate)
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                scientific_interval=[START, STOP],
                process_interval=[RESUME, STOP],
                retained_interval=[FIRST, STOP],
                native_frames=STOP - START,
                process_frames=STOP - RESUME,
                initialize_count=1,
                restore_count=6,
                external_pose_resets=0,
                new_begin_calls=0,
                finish_calls=1,
                all39_roundtrip=True,
                prior_history_prefix_unchanged=True,
                finish_matches_checkpoint_backtrace=True,
                checkpoints=checkpoints,
                timings=timings,
                candidate_arrays=base.array_signature(candidate),
                physical_ID_truth=False,
                independent_evaluation_pending=True,
                automatic_adoption=False,
            ),
        )
        store.final_guard()
        for path, sha in store.evidence_hashes.items():
            evidence.add(path, sha)
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
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--tracker-sha", required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())

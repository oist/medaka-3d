"""Disabled continuation of the existing -12 mm policy through five minutes.

Restore the successful 78020 ExpandedDepth checkpoint exactly once. Its mask,
sparse distance and depth policies already began at 76360 and remain unchanged.
No new policy migration or dense control is introduced. Every future native
frame retains companion diagnostics and every 1660 frames retains all39 state.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import run_expanded_depth_instance_rescore_10s as parent_run

base, diagnostic, variant = parent_run.base, parent_run.diagnostic, parent_run.variant
prior, proof, original, qualified = base.prior, base.proof, base.original, base.qualified
torch, WORK, REPO, require = base.torch, base.WORK, base.REPO, base.require
FIRST, START, RESUME, STOP, CHUNK = 49800, 66400, 78020, 116200, 1660
MILESTONES = (86320, 96280, 106240, 116200)
STAGE = "expanded_depth_five_minute_continuation_v1"
EXECUTION_READY = True
TRACKER_SHA = parent_run.TRACKER_SHA
PARENT = WORK / "expanded_depth_instance_rescore_ten_second_v1"
PARENT_SHA = "0ff5f6a31f020a2f3773176e5af60990ffb6bbf5fc7c67cbf505dd208d6e1d8c"
PARENT_RUNNER_SHA = "b1a16cedb9a4c7cacf92d96c6fa496f1d0a143e27b023fdc5b022126923f49aa"
PARENT_CHECKPOINT_SHA = "72af2d3c435a6e644f3ff48cbe3e94eff50d01c16c01588b2c2cc874c79c751b"
QA = WORK / "expanded_depth_ten_second_independent_qa_v1/qa.json"
QA_SHA = "984f9c0efeb688930a42a3c2ea25d058b1754f4d7e3864e27340c542f7198d8c"
QA_SOURCE = WORK / "qa_expanded_depth_saved_10s.py"
QA_SOURCE_SHA = "d06447f72ef972707761c8ef61bf8ba7e8ccad333fd87ad40606d4b2519e38d0"
QA_SELECTED = QA.parent / "selected_candidates.npz"
QA_SELECTED_SHA = "96f6e9164b2f84384207bdf809a52772396813cc0b4b81cdbcdf827c6954472c"
PARENT_COMPARISON = (
    WORK / "expanded_depth_instance_ten_second_comparison_v1/comparison.provenance.json"
)
PARENT_COMPARISON_SHA = "bb6002ab37441a86daac6a05ed19176b512601e8ad8c30a1df620fb064848f59"
PINS = {**parent_run.PINS, "run_expanded_depth_instance_rescore_10s.py": PARENT_RUNNER_SHA}
POLICY_FIELDS = {
    "scoring_policy": "instance_rescore_policy",
    "sparse_distance_backend": "sparse_distance_backend",
    "expanded_depth_policy": "expanded_depth_policy",
}


def source_closure():
    code = prior.local_code()
    prior.source_contract(code)
    require(str(Path(__file__).resolve()) in code, "new caller missing from import closure")
    for name, digest in PINS.items():
        require(code.get(str(WORK / name)) == digest, "source missing or changed: " + name)
    return code


def lifecycle(tracker, through):
    require(
        tracker.initialize_count == 1
        and tracker.restore_count == 6
        and tracker.previous_frame == through - 1
        and tracker.processed_frames == through - FIRST,
        "single-initialization continuous lifecycle changed",
    )


def policies(context):
    result = {name: copy.deepcopy(context[key]) for name, key in POLICY_FIELDS.items()}
    require(
        all(p["effective_from_native"] == 76360 for p in result.values()),
        "policy effective frame changed",
    )
    require(result["expanded_depth_policy"]["expanded_lower_z_mm"] == -12.0, "depth lower changed")
    return result


def validate_parent_metadata(metadata):
    require(
        metadata["previous_frame"] == RESUME - 1
        and metadata["processed_frames"] == RESUME - FIRST
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 5,
        "actual parent lifecycle differs",
    )
    policies(metadata["context"])


def load_qa(evidence):
    for path, digest in (
        (QA, QA_SHA),
        (QA_SOURCE, QA_SOURCE_SHA),
        (QA_SELECTED, QA_SELECTED_SHA),
        (PARENT_COMPARISON, PARENT_COMPARISON_SHA),
    ):
        evidence.add(path, digest)
    qa, comparison = proof.read_json(QA), proof.read_json(PARENT_COMPARISON)
    require(
        qa["status"] == "ok"
        and qa["producer_seal_sha256"] == PARENT_SHA
        and qa["comparison_seal_sha256"] == PARENT_COMPARISON_SHA
        and qa["qa_source_sha256"] == QA_SOURCE_SHA
        and qa["selected_candidates"]["path"] == str(QA_SELECTED)
        and qa["selected_candidates"]["sha256"] == QA_SELECTED_SHA
        and qa["diagnostics"]["frames"] == 1660
        and qa["diagnostics"]["all_finished_rows_uniquely_matched"]
        and qa["all21_final_candidate_arrays_verified"]
        and qa["imported39_signatures_and_metadata_bound"],
        "successful bound parent QA required",
    )
    require(
        comparison["status"] == "ok"
        and comparison["stage"] == PARENT_COMPARISON.parent.name
        and comparison["inputs"].get(str(PARENT / "tracking.provenance.json")) == PARENT_SHA,
        "QA comparison did not consume this successful parent",
    )
    return dict(
        path=str(QA),
        sha256=QA_SHA,
        source_path=str(QA_SOURCE),
        source_sha256=QA_SOURCE_SHA,
        selected_candidates=copy.deepcopy(qa["selected_candidates"]),
        comparison_path=str(PARENT_COMPARISON),
        comparison_sha256=PARENT_COMPARISON_SHA,
        independent_physical_ID_or_depth_truth=False,
    )


def load_parent_headers(evidence):
    """Small proof and metadata only; numeric39 are opened separately at execution."""
    evidence.add(PARENT / "tracking.provenance.json", PARENT_SHA)
    seal = proof.read_json(PARENT / "tracking.provenance.json")
    require(
        seal["status"] == "ok" and seal["stage"] == PARENT.name, "successful depth parent required"
    )
    docs = {
        name: parent_run.bound_member(PARENT, seal, name + ".json", evidence)
        for name in ("plan", "summary", "state_import", "input_index")
    }
    plan, summary = docs["plan"], docs["summary"]
    required = dict(
        schema=PARENT.name,
        status="ok",
        process_interval=[76360, RESUME],
        native_frames=1660,
        retained_interval=[FIRST, RESUME],
        initialize_count=1,
        restore_count=5,
        new_begin_calls=0,
        external_pose_resets=0,
        finish_calls=1,
        all39_roundtrip=True,
        prior_history_prefix_unchanged=True,
        finish_matches_checkpoint_backtrace=True,
        diagnostic_frames=1660,
        checkpoint_manifest_sha256=PARENT_CHECKPOINT_SHA,
    )
    require(
        all(summary[k] == value for k, value in required.items()), "parent scope/completion differs"
    )
    require(
        plan["source_sha256"].get(str(Path(parent_run.__file__).resolve())) == PARENT_RUNNER_SHA,
        "parent runner differs",
    )
    for path, digest in plan["source_sha256"].items():
        require(seal["inputs"].get(path) == digest, "parent consumed source edge")
        evidence.add(path, digest)
        relative = "code_snapshot/" + Path(path).relative_to(REPO).as_posix()
        require(seal["outputs"][relative]["sha256"] == digest, "parent source snapshot edge")
        snapshot = Path(seal["outputs"][relative]["path"])
        require(snapshot == PARENT / relative, "parent snapshot path")
        evidence.add(snapshot, digest)
    checkpoint = PARENT / "checkpoints/through_078020"
    manifest = parent_run.bound_member(
        PARENT, seal, "checkpoints/through_078020/manifest.json", evidence
    )
    metadata = parent_run.bound_member(
        PARENT, seal, "checkpoints/through_078020/metadata.json", evidence
    )
    require(
        prior.checks.sha(checkpoint / "manifest.json") == PARENT_CHECKPOINT_SHA,
        "parent manifest differs",
    )
    require(manifest["bindings"] == plan["checkpoint_binding"], "parent checkpoint binding")
    validate_parent_metadata(metadata)
    parent_policies = policies(metadata["context"])
    for output_key, old_key in (
        ("scoring_policy", "new_scoring_policy"),
        ("sparse_distance_backend", "distance_backend"),
        ("expanded_depth_policy", "expanded_depth_policy"),
    ):
        require(
            all(
                doc[old_key] == parent_policies[output_key]
                for doc in (plan, summary, manifest["bindings"])
            ),
            "parent scientific policy differs",
        )
    require(
        docs["input_index"]["original_binding"]
        == plan["original_binding"]
        == manifest["bindings"]["original_binding"],
        "parent observation binding",
    )
    require(len(docs["input_index"]["prefix_chunks"]) == 17, "parent17 chunks required")
    require(
        docs["state_import"]["new_context"] == metadata["context"], "parent import context differs"
    )
    require(
        docs["state_import"]["all_arrays_byte_preserved"]
        and not docs["state_import"]["review_cleared_by_migration"],
        "parent import proof differs",
    )
    qa = load_qa(evidence)
    return dict(
        seal=seal,
        plan=plan,
        summary=summary,
        manifest=manifest,
        metadata=metadata,
        checkpoint=checkpoint,
        parent_qa=qa,
    )


def load_parent_arrays(headers, evidence):
    checkpoint, seal = headers["checkpoint"], headers["seal"]
    row = seal["outputs"]["checkpoints/through_078020/arrays.npz"]
    path = checkpoint / "arrays.npz"
    require(
        row["path"] == str(path) and row["size_bytes"] == path.stat().st_size,
        "parent payload descriptor",
    )
    evidence.add(path, row["sha256"])
    arrays, metadata, manifest = prior.read_checkpoint(
        checkpoint, expected_bindings=headers["manifest"]["bindings"]
    )
    require(
        metadata == headers["metadata"] and manifest == headers["manifest"] and len(arrays) == 39,
        "parent complete39/header differs",
    )
    validate_parent_metadata(metadata)
    return arrays, metadata


def restore_parent(tracker, arrays, metadata):
    """Same exact class/context, one native restore; no adapter or policy rewrite."""
    require(
        type(tracker) is variant.ExpandedDepthInstanceRescoreContinuousTracker,
        "exact ExpandedDepth class required",
    )
    require(
        tracker.initialize_count == tracker.restore_count == tracker.processed_frames == 0
        and tracker.previous_frame is None,
        "fresh uninitialized target required",
    )
    validate_parent_metadata(metadata)
    require(
        tracker._context() == tracker._initial_context == metadata["context"],
        "unchanged parent context required",
    )
    before, old_metadata = base.array_signature(arrays), copy.deepcopy(metadata)
    require(len(arrays) == 39, "complete39 parent required")
    tracker.restore_state(arrays, metadata)
    lifecycle(tracker, RESUME)
    restored, new_metadata = tracker.export_state()
    expected = copy.deepcopy(old_metadata)
    expected["restore_count"] = 6
    require(
        base.array_signature(restored) == before and new_metadata == expected,
        "all39 or metadata changed during restore",
    )
    require(
        base.array_signature(arrays) == before and metadata == old_metadata, "restore mutated input"
    )
    return dict(
        schema=STAGE + "_same_policy_restore",
        status="ok",
        arrays=before,
        original_metadata=old_metadata,
        restored_metadata=new_metadata,
        all39_byte_preserved=True,
        source_inputs_unchanged=True,
        context_unchanged=True,
        initialize_count=1,
        source_restore_count=5,
        restore_count=6,
        new_begin_calls=0,
        policy_transition=False,
        review_cleared=False,
        resume_native=RESUME,
    )


def drive_chunk(tracker, iterator, first, stop, out, *, synchronize, guard):
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
        guard()
        if (frame + 1 - RESUME) % 100 == 0 or frame + 1 == stop:
            print(
                json.dumps(dict(event="expanded_depth_progress", through_frame=frame + 1)),
                flush=True,
            )
    return rows, timings


def save_milestone(out, saved, metadata, checkpoint, binding, stop):
    require(
        stop in MILESTONES and metadata["previous_frame"] == stop - 1,
        "milestone checkpoint boundary",
    )
    require(
        metadata["initialize_count"] == 1 and metadata["restore_count"] == 6, "milestone lifecycle"
    )
    candidate = original.phase.COMMON.traced_candidate(saved)
    require(
        len(candidate) == 21 and np.array_equal(candidate["frame_idx"], np.arange(FIRST, stop)),
        "milestone all21 backtrace extent",
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
            retained_interval=[FIRST, stop],
            process_interval=[RESUME, stop],
            candidate=member,
            candidate_arrays=base.array_signature(candidate),
            checkpoint=checkpoint,
            initialize_count=1,
            restore_count=6,
            source_sha256=binding["source_sha256"],
            parent_tracking_sha256=PARENT_SHA,
            parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA,
            parent_qa=binding["parent_qa"],
            original_binding=binding["original_binding"],
            external_pose_resets=0,
            new_begin_calls=0,
            finish_calls_so_far=0,
            all39_roundtrip=True,
            prior_history_prefix_unchanged=True,
            policy_transition=False,
            physical_ID_truth=False,
            physical_depth_truth=False,
            whole_process_completion_pending=True,
            **{key: binding[key] for key in POLICY_FIELDS},
        ),
    )


def main(args):
    require(
        EXECUTION_READY and args.execute and args.tracker_sha == TRACKER_SHA,
        "reviewed execution is not enabled",
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
        code = source_closure()
        for path, digest in code.items():
            evidence.add(path, digest)
            out.write(
                "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), Path(path).read_bytes()
            )
        headers = load_parent_headers(evidence)
        arrays, metadata = load_parent_arrays(headers, evidence)
        binding = headers["manifest"]["bindings"]["original_binding"]
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
        tracker = variant.ExpandedDepthInstanceRescoreContinuousTracker(
            inputs.rig,
            inputs.initial_pose_xyz.copy(),
            qualified.TrackerConfig(**protocol["tracker_config"]),
            recovery_config=qualified.NodeRecoveryConfig(**protocol["node_recovery_config"]),
            persistence_config=qualified.PersistentNodeConfig(**protocol["persistence_config"]),
        )
        restored = restore_parent(tracker, arrays, metadata)
        out.write("state_restore.json", restored)
        checkpoint_binding = dict(
            schema=STAGE + "_checkpoint",
            original_binding=binding,
            source_sha256=code,
            parent_tracking_sha256=PARENT_SHA,
            parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA,
            parent_qa=headers["parent_qa"],
            scientific_interval=[START, STOP],
            process_interval=[RESUME, STOP],
            retained_interval=[FIRST, STOP],
            policy_transition=False,
            **policies(tracker._context()),
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
                parent_tracking_sha256=PARENT_SHA,
                parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA,
                parent_qa=headers["parent_qa"],
                environment=environment,
                precision="float64",
                allocator_fraction=0.25,
                reserved_limit_bytes=4 * 1024**3,
                initialize_count=1,
                restore_count=6,
                new_begin_calls=0,
                external_pose_resets=0,
                policy_transition=False,
                semantic_landmark_term_used=False,
                physical_ID_truth=False,
                physical_depth_truth=False,
                automatic_adoption=False,
                **policies(tracker._context()),
            ),
        )
        evidence.final_guard()
        print(json.dumps(dict(event="expanded_depth_restored", through_frame=RESUME)), flush=True)

        def guard():
            qualified.check_headroom(prior.cuda_headroom())
            qualified.check_memory(qualified.qualification.memory_sample())

        parent = dict(arrays=arrays, metadata=metadata)
        timings = dict(advance_seconds=0.0, native_next_seconds=0.0, diagnostic_save_seconds=0.0)
        diagnostic_frames = 0
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
                rows, measured = drive_chunk(
                    tracker,
                    iterator,
                    first,
                    stop,
                    out,
                    synchronize=torch.cuda.synchronize,
                    guard=guard,
                )
            finally:
                iterator.close()
            require(
                [r["frame_idx"] for r in rows] == list(range(first, stop)),
                "complete chunk diagnostics required",
            )
            diagnostic_frames += len(rows)
            for key, value in measured.items():
                timings[key] += value
            out.write(
                f"diagnostic_index/through_{stop:06d}.json",
                dict(rows=rows, process_interval=[first, stop], **policies(tracker._context())),
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
                save_milestone(out, saved, meta, row, checkpoint_binding, stop)
            out.write(
                f"checkpoint_records/progress_{stop:06d}.json",
                dict(
                    checkpoints=checkpoints,
                    timings=timings.copy(),
                    diagnostic_frames=diagnostic_frames,
                ),
            )
            parent = dict(arrays=saved, metadata=meta)
            print(
                json.dumps(dict(event="expanded_depth_checkpoint", through_frame=stop)), flush=True
            )
        require(diagnostic_frames == STOP - RESUME, "full continuation diagnostics required")
        candidate = tracker.finish()
        lifecycle(tracker, STOP)
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
                diagnostic_frames=diagnostic_frames,
                candidate_arrays=base.array_signature(candidate),
                parent_tracking_sha256=PARENT_SHA,
                parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA,
                parent_qa=headers["parent_qa"],
                policy_transition=False,
                physical_ID_truth=False,
                physical_depth_truth=False,
                independent_evaluation_pending=True,
                automatic_adoption=False,
                **policies(tracker._context()),
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
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--tracker-sha", required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())

"""Resume Sparse at 89640 with unchanged nodewise proposals through 91300.

One explicit context migration preserves the complete parent history and state.
The original candidate pool, FP64 fitter, sixth term, support and review remain;
new whole-row proposals use only residual capacity of the existing candidate cap.
No threshold tuning, new ID labels, pose reset or implicit restart is performed.
This is the unmasked negative-window control: no split-priority module is imported.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import continue_fast_tail_five_minutes as base
import numpy as np
import run_instance_rescore_10s as diagnostic
import sparse_instance_rescore_tracker as sparse

# isort: split
# The frozen base bootstrap adds the lab modules to sys.path before this import.
import nodewise_body_row_tracker as variant

prior, proof, original, qualified = base.prior, base.proof, base.original, base.qualified
torch, WORK, REPO, require = base.torch, base.WORK, base.REPO, base.require
FIRST, SCIENCE_START, START, STOP, CHUNK = 49800, 66400, 89640, 91300, 1660
STAGE = "nodewise_body_row_negative_ten_second_continuation_v1"
PARENT = WORK / "sparse_instance_five_minute_continuation_v1"
PARENT_SHA = "596b23afe5fb572a66a72b96b0b1f5ba9b3971bfef4dd43cb8ee6b32459e32ad"
PARENT_SOURCE_SHA = "3f5141a5aeec6fadc464ad242ac19fe13e227065af6f1188c638af4c58129096"
CHECKPOINT_RELATIVE = "checkpoints/through_089640"
CHECKPOINT_SHA = "f76a4031f831bdd5b2e0516237242713e981a305c40f2b57966c90fa625a6e70"
PINS = {
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


def source_window(first=FIRST, start=START, stop=STOP, chunk=CHUNK):
    """One complete source chunk, with a prefix derived from native bounds."""
    require(
        all(type(value) is int for value in (first, start, stop, chunk))
        and chunk > 0
        and first <= start < stop
        and (start - first) % chunk == 0
        and stop - start == chunk,
        "one aligned native source chunk required",
    )
    return dict(chunk_index=(start - first) // chunk, prefix_chunks=(stop - first) // chunk)


def validate_source_prefix(inputs, parent_index):
    window = source_window()
    count = window["prefix_chunks"]
    require(
        len(inputs.chunk_roots) == len(inputs.chunk_records) == count
        and inputs.chunk_records == parent_index["prefix_chunks"][:count],
        "same consumed source prefix required",
    )
    for index, row in enumerate(inputs.chunk_records):
        require(
            row["index"] == index
            and row["start"] == FIRST + index * CHUNK
            and row["stop"] == FIRST + (index + 1) * CHUNK,
            "source prefix native boundaries differ",
        )
    return window


def validate_reader_frames(reader):
    require(
        np.array_equal(reader.frames, np.arange(START, STOP)),
        "wrong or noncontiguous native source chunk24",
    )


def lifecycle(tracker, through):
    require(
        tracker.initialize_count == 1
        and tracker.restore_count == 7
        and tracker.previous_frame == through - 1
        and tracker.processed_frames == through - FIRST,
        "whole-row continuation lifecycle changed",
    )


def bound_member(root, seal, relative, evidence):
    root = Path(root).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), "sealed member escaped its run")
    row = seal["outputs"][relative]
    require(
        path == Path(row["path"]).resolve()
        and path.is_file()
        and path.stat().st_size == row["size_bytes"]
        and proof.valid_sha(row["sha256"]),
        "sealed member descriptor differs",
    )
    evidence.add(path, row["sha256"])
    return path


def load_parent_headers(evidence):
    """Read only completed-run metadata; defer the 39-field payload to execution."""
    evidence.add(PARENT / "tracking.provenance.json", PARENT_SHA)
    seal = proof.read_json(PARENT / "tracking.provenance.json")
    require(
        seal["stage"] == PARENT.name and seal["status"] == "ok", "successful Sparse parent required"
    )
    docs = {
        name: proof.read_json(bound_member(PARENT, seal, name + ".json", evidence))
        for name in ("plan", "summary", "input_index")
    }
    plan, summary, index = (docs[k] for k in ("plan", "summary", "input_index"))
    require(
        summary["schema"] == PARENT.name
        and summary["status"] == "ok"
        and summary["process_interval"] == [78020, 116200]
        and summary["retained_interval"] == [FIRST, 116200]
        and summary["scientific_interval"] == [SCIENCE_START, 116200]
        and summary["initialize_count"] == 1
        and summary["restore_count"] == 6
        and summary["new_begin_calls"] == summary["external_pose_resets"] == 0
        and summary["finish_calls"] == 1
        and summary["all39_roundtrip"] is True
        and summary["prior_history_prefix_unchanged"] is True
        and summary["finish_matches_checkpoint_backtrace"] is True,
        "parent completion or lifecycle differs",
    )
    source = str((WORK / "continue_sparse_instance_five_minutes.py").resolve())
    require(
        seal["inputs"].get(source) == plan["source_sha256"].get(source) == PARENT_SOURCE_SHA,
        "parent producer source edge differs",
    )
    evidence.add(source, PARENT_SOURCE_SHA)
    checkpoint = PARENT / CHECKPOINT_RELATIVE
    manifest = proof.read_json(
        bound_member(PARENT, seal, CHECKPOINT_RELATIVE + "/manifest.json", evidence)
    )
    metadata = proof.read_json(
        bound_member(PARENT, seal, CHECKPOINT_RELATIVE + "/metadata.json", evidence)
    )
    require(
        evidence.paths[str((checkpoint / "manifest.json").resolve())] == CHECKPOINT_SHA
        and manifest["schema"] == "medaka_tracker_checkpoint_v1"
        and manifest["status"] == "ok"
        and manifest["bindings"] == plan["checkpoint_binding"],
        "parent checkpoint binding differs",
    )
    matches = [row for row in summary["checkpoints"] if row["previous_frame"] == START - 1]
    require(
        len(matches) == 1
        and Path(matches[0]["path"]).resolve() == checkpoint.resolve()
        and matches[0]["manifest_sha256"] == CHECKPOINT_SHA,
        "parent checkpoint is absent from successful summary",
    )
    require(
        metadata["schema"] == "persistent_tracker_full_state_v1"
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 6
        and metadata["previous_frame"] == START - 1
        and metadata["processed_frames"] == START - FIRST
        and metadata["context"]["implementation"] == "SparseInstanceRescoreContinuousTracker"
        and metadata["context"]["sparse_distance_backend"]["effective_from_native"] == 78020,
        "parent checkpoint native extent or state differs",
    )
    for field, name in (("metadata", "metadata.json"), ("payload", "arrays.npz")):
        row = seal["outputs"][CHECKPOINT_RELATIVE + "/" + name]
        require(
            manifest[field]["file"] == name
            and manifest[field]["sha256"] == row["sha256"]
            and manifest[field]["bytes"] == row["size_bytes"],
            "checkpoint manifest and success member differ",
        )
    binding = manifest["bindings"]["original_binding"]
    require(
        index["original_binding"] == plan["original_binding"] == binding
        and len(index["prefix_chunks"]) == 40,
        "parent consumed source prefix differs",
    )
    return dict(seal=seal, manifest=manifest, metadata=metadata, binding=binding, input_index=index)


def load_parent_payload(headers, evidence):
    bound_member(PARENT, headers["seal"], CHECKPOINT_RELATIVE + "/arrays.npz", evidence)
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


def save_frame_diagnostics(out, tracker, frame):
    instance_row = diagnostic.save_diagnostic(out, tracker, frame)
    body = tracker.export_body_row_diagnostics()
    require(
        body["schema"] == "body_row_frame_diagnostics_v1"
        and body["frame_idx"] == frame
        and body["frame_completed"] is True,
        "body-row diagnostic is not the completed native frame",
    )
    counts = body_row_counts(body)
    require(
        instance_row["candidate_count"] == counts["original_count"] + counts["admitted"],
        "instance and body-row diagnostics describe different candidate pools",
    )
    tree, arrays = diagnostic.split_diagnostic(body)
    member = out.write(f"body_row_diagnostics/f{frame}.npz", arrays)
    with np.load(member["path"], allow_pickle=False) as archive:
        restored = diagnostic.restore_diagnostic(tree, {key: archive[key] for key in archive.files})
    require(restored["frame_idx"] == frame, "body-row diagnostic roundtrip changed native frame")
    out.write(f"body_row_diagnostics/f{frame}.json", dict(archive=member, tree=tree))
    return dict(**instance_row, body_row=counts)


def body_row_counts(body):
    """Compact explicit proposal admission counts; never final trajectory counts."""
    keys = (
        "incoming_proposal_count",
        "outgoing_proposal_count",
        "decoder_candidate_cap",
        "headroom",
        "generated_scene_count",
        "added_proposal_count",
        "excluded_by_capacity",
    )
    require(
        all(type(body[k]) is int and body[k] >= 0 for k in keys), "integer proposal counts required"
    )
    incoming, outgoing = body["incoming_proposal_count"], body["outgoing_proposal_count"]
    admitted, headroom = body["added_proposal_count"], body["headroom"]
    require(
        incoming > 0
        and outgoing == incoming + admitted
        and headroom == body["decoder_candidate_cap"] - incoming
        and 0 <= admitted <= headroom
        and body["excluded_by_capacity"] == body["generated_scene_count"] - admitted
        and body["zero_headroom"] is (headroom == 0)
        and np.array_equal(body["new_proposal_indices"], np.arange(incoming, outgoing))
        and body["new_proposal_kind"] == "body_row",
        "body-row admission/cap/index contract differs",
    )
    return dict(
        original_count=incoming,
        admitted=admitted,
        headroom=headroom,
        generated_scene_count=body["generated_scene_count"],
        excluded_by_capacity=body["excluded_by_capacity"],
    )


def drive(tracker, iterator, out, *, synchronize, guard, first=START, stop=STOP):
    rows = []
    timings = dict(advance_seconds=0.0, native_next_seconds=0.0, diagnostic_save_seconds=0.0)
    for native in range(first, stop):
        lifecycle(tracker, native)
        tick = time.perf_counter()
        observation = next(iterator)
        timings["native_next_seconds"] += time.perf_counter() - tick
        require(observation.frame_idx == native, "native frame skipped or reordered")
        guard()
        tick = time.perf_counter()
        tracker.advance(observation)
        synchronize()
        timings["advance_seconds"] += time.perf_counter() - tick
        lifecycle(tracker, native + 1)
        tick = time.perf_counter()
        rows.append(save_frame_diagnostics(out, tracker, native))
        timings["diagnostic_save_seconds"] += time.perf_counter() - tick
        guard()
        if (native + 1 - first) % 100 == 0 or native + 1 == stop:
            print(json.dumps(dict(event="body_row_progress", through_frame=native + 1)), flush=True)
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
            json.dumps(dict(event="body_row_input_validation", output=str(destination))), flush=True
        )
        inputs, protocol = original.verified_inputs(binding, STOP, evidence, receipt_store=store)
        window = validate_source_prefix(inputs, headers["input_index"])
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
            sparse_effective_from_native=78020,
        )
        source = sparse.SparseInstanceRescoreContinuousTracker(
            inputs.rig, inputs.initial_pose_xyz.copy(), config, **kwargs
        )
        tracker = variant.BodyRowContinuousTracker(
            inputs.rig,
            inputs.initial_pose_xyz.copy(),
            config,
            body_row_effective_from_native=START,
            **kwargs,
        )
        imported = variant.import_sparse_state(tracker, source, arrays, metadata)
        lifecycle(tracker, START)
        require(source.initialize_count == 0, "migration source unexpectedly initialized")
        out.write("state_import.json", imported)
        context = tracker._context()
        checkpoint_binding = dict(
            schema=STAGE + "_checkpoint",
            original_binding=binding,
            source_sha256=code,
            parent_tracking_sha256=PARENT_SHA,
            parent_checkpoint_sha256=CHECKPOINT_SHA,
            retained_interval=[FIRST, STOP],
            process_interval=[START, STOP],
            scientific_interval=[SCIENCE_START, STOP],
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
                parent_tracking_sha256=PARENT_SHA,
                parent_checkpoint_sha256=CHECKPOINT_SHA,
                initialize_count=1,
                restore_count=7,
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
                only_pre89640_temporal_state_imported=True,
                mask_split_priority_applied=False,
                source_window=window,
                inherited_seed_provenance_unchanged=True,
                parent_final_seal_used_for_provenance_only=True,
                physical_ID_truth=False,
                automatic_adoption=False,
            ),
        )
        evidence.final_guard()
        print(json.dumps(dict(event="body_row_imported", through_frame=START)), flush=True)

        def guard():
            qualified.check_headroom(prior.cuda_headroom())
            qualified.check_memory(qualified.qualification.memory_sample())

        tick = time.perf_counter()
        reader = original.SourceSequenceReader(
            inputs.chunk_roots[window["chunk_index"]], inputs.input_root
        )
        validate_reader_frames(reader)
        iterator = original.bounded_frames(reader, first=START, stop=STOP)
        reader_seconds = time.perf_counter() - tick
        try:
            rows, timings = drive(
                tracker, iterator, out, synchronize=torch.cuda.synchronize, guard=guard
            )
        finally:
            iterator.close()
        out.write("diagnostic_index.json", dict(rows=rows, process_interval=[START, STOP]))
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
                scientific_native_frames=STOP - SCIENCE_START,
                parent_tracking_sha256=PARENT_SHA,
                parent_checkpoint_sha256=CHECKPOINT_SHA,
                source_sha256=code,
                tracker_context=context,
                candidate=candidate_record,
                candidate_arrays=base.array_signature(candidate),
                checkpoint_manifest_sha256=checkpoint["manifest_sha256"],
                checkpoint=checkpoint,
                checkpoint_count=1,
                initialize_count=1,
                restore_count=7,
                external_pose_resets=0,
                new_begin_calls=0,
                finish_calls=1,
                all39_roundtrip=True,
                completed_checkpoint_decoder_accepted=True,
                prior_history_prefix_unchanged=True,
                finish_matches_checkpoint_backtrace=True,
                diagnostic_frames=len(rows),
                body_row=proposal_summary(rows, candidate),
                timings=dict(reader_seconds=reader_seconds, **timings),
                existing_five_parts_and_sixth_term_unchanged=True,
                mask_split_priority_applied=False,
                source_window=window,
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

"""One explicit scoring-policy transition, then 1660 native frames without resets.

This separate experiment adds a finite instance cost after FastTail fitting and
before beam selection. It tests candidate selection, not optimization of a new
differentiable fitting objective. Every frame's full candidate cost diagnostics
are saved apart from the unchanged 39-field temporal checkpoint schema.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import continue_fast_tail_five_minutes as base
import instance_body_score as core
import instance_rescore_tracker as variant
import numpy as np

prior, proof, original, qualified = base.prior, base.proof, base.original, base.qualified
torch, WORK, REPO, require = base.torch, base.WORK, base.REPO, base.require
FIRST, START, STOP = 49800, 76360, 78020
STAGE = "instance_rescore_ten_second_continuation_v1"
AUDIT_SHA = "77bc769867edeb3f87c84abaf6f6c8f438b4f30f9a92c747b5e2a51e02d86188"
BASE_SHA = "d90e9eb49038dac4fd1e18068334ba89eb8df1a813a37c6aec071c20db1a61a6"
CORE_SHA = "cb9e04a20160f2b681aa1b7cca46ed5b4017c7b05760f9dbbeff691bf058cb87"
DIAGNOSIS = WORK / "saved_capacity_island_instance_body_score_v1"
DIAGNOSIS_SHA = "cb0e989408270e2999b232ffbe7693a9f0ad4f97226d4eec68abae3a14576783"


def split_diagnostic(value):
    """Keep nested arrays/nonfinite values exact in NPZ, with a JSON shape tree."""
    arrays = {}

    def split(item):
        if isinstance(item, np.ndarray) or (
            isinstance(item, (float, np.floating)) and not np.isfinite(item)
        ):
            array = np.asarray(item)
            require(not array.dtype.hasobject, "object diagnostic arrays are forbidden")
            key = f"a{len(arrays):05d}"
            arrays[key] = array.copy()
            return {"__array__": key, "signature": proof.signature(array)}
        if isinstance(item, np.generic):
            return split(item.item())
        if isinstance(item, dict):
            require(all(isinstance(k, str) for k in item), "string diagnostic keys required")
            require("__array__" not in item, "reserved diagnostic descriptor key")
            return {key: split(v) for key, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [split(v) for v in item]
        require(
            item is None or isinstance(item, (str, bool, int, float)), "unknown diagnostic type"
        )
        return item

    tree = split(value)
    return tree, arrays


def restore_diagnostic(tree, arrays):
    """Verify the complete saved array membership while reconstructing metadata."""
    used = set()

    def restore(item):
        if isinstance(item, dict) and "__array__" in item:
            key = item["__array__"]
            require(set(item) == {"__array__", "signature"}, "invalid array descriptor")
            require(key in arrays and key not in used, "missing or repeated diagnostic array")
            used.add(key)
            value = arrays[key]
            require(proof.signature(value) == item["signature"], "diagnostic array differs")
            return value
        if isinstance(item, dict):
            return {key: restore(v) for key, v in item.items()}
        if isinstance(item, list):
            return [restore(v) for v in item]
        return item

    result = restore(tree)
    require(set(arrays) == used, "extra diagnostic array")
    return result


def save_diagnostic(out, tracker, frame):
    value = tracker.export_instance_diagnostics()
    require(value["frame_idx"] == frame, "wrong frame diagnostic")
    tree, arrays = split_diagnostic(value)
    member = out.write(f"diagnostics/f{frame}.npz", arrays)
    with np.load(member["path"], allow_pickle=False) as archive:
        restored = restore_diagnostic(tree, {k: archive[k] for k in archive.files})
    require(restored["frame_idx"] == frame, "diagnostic native index changed")
    out.write(f"diagnostics/f{frame}.json", dict(archive=member, tree=tree))
    return dict(
        frame_idx=frame,
        candidate_count=len(value["candidate_index"]),
        gate_preparation_seconds=value["gate_preparation_seconds"],
        candidate_scoring_seconds=value["candidate_scoring_seconds"],
        eligible_views=[view for view, row in value["views"].items() if row["eligible"]],
    )


def drive(tracker, iterator, out, *, synchronize, guard):
    rows = []
    timings = dict(advance_seconds=0.0, native_next_seconds=0.0, diagnostic_save_seconds=0.0)
    for frame in range(START, STOP):
        base.lifecycle(tracker, frame)
        tick = time.perf_counter()
        observation = next(iterator)
        timings["native_next_seconds"] += time.perf_counter() - tick
        require(observation.frame_idx == frame, "native frame skipped or reordered")
        guard()
        tick = time.perf_counter()
        tracker.advance(observation)
        synchronize()
        timings["advance_seconds"] += time.perf_counter() - tick
        base.lifecycle(tracker, frame + 1)
        tick = time.perf_counter()
        rows.append(save_diagnostic(out, tracker, frame))
        timings["diagnostic_save_seconds"] += time.perf_counter() - tick
        guard()
        if (frame + 1 - START) % 100 == 0 or frame + 1 == STOP:
            print(
                json.dumps(dict(event="instance_rescore_progress", through_frame=frame + 1)),
                flush=True,
            )
    return rows, timings


def main(args):
    require(
        args.execute and proof.valid_sha(args.tracker_sha), "execution and tracker SHA required"
    )
    destination = args.output.resolve()
    require(destination.is_relative_to(WORK) and not destination.exists(), "fresh output required")
    out, evidence = proof.Output(destination), original.phase.Digests()
    started = proof.utc()
    code, checkpoint = {}, None
    try:
        evidence.add(Path(base.__file__), BASE_SHA)
        evidence.add(Path(core.__file__), CORE_SHA)
        evidence.add(Path(variant.__file__), args.tracker_sha)
        evidence.add(DIAGNOSIS / "scoring.provenance.json", DIAGNOSIS_SHA)
        diagnosis = proof.read_json(DIAGNOSIS / "scoring.provenance.json")
        require(diagnosis["status"] == "ok", "successful fixed diagnostic required")
        code = prior.local_code()
        prior.source_contract(code)
        for path, digest in code.items():
            evidence.add(path, digest)
            out.write(
                "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), Path(path).read_bytes()
            )
        store = qualified.ReceiptStore(args.receipt_root.resolve())
        print(
            json.dumps(dict(event="instance_rescore_validation_started", output=str(destination))),
            flush=True,
        )
        arrays, metadata, binding = base.load_parent(AUDIT_SHA, evidence)
        require(
            metadata["previous_frame"] == START - 1
            and metadata["processed_frames"] == START - FIRST
            and metadata["initialize_count"] == 1
            and metadata["restore_count"] == 4,
            "wrong full scientific parent lifecycle",
        )
        inputs, protocol = original.verified_inputs(binding, STOP, evidence, receipt_store=store)
        require(
            len(inputs.chunk_records) == len(inputs.chunk_roots) == 17,
            "registered 170s prefix required",
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
        tracker = variant.InstanceRescoreContinuousTracker(
            inputs.rig, inputs.initial_pose_xyz.copy(), config, **kwargs
        )
        source = prior.FastTailRecoveryContinuousTracker(
            inputs.rig, inputs.initial_pose_xyz.copy(), config, **kwargs
        )
        imported = variant.import_fast_tail_state(tracker, source, arrays, metadata)
        base.lifecycle(tracker, START)
        out.write("state_import.json", imported)
        checkpoint_binding = dict(
            schema=STAGE + "_checkpoint",
            original_binding=binding,
            source_sha256=code,
            parent_audit_sha256=AUDIT_SHA,
            parent_checkpoint_sha256=base.audit.CHECKPOINT_SHA,
            retained_interval=[FIRST, STOP],
            process_interval=[START, STOP],
            new_scoring_policy=tracker._context()["instance_rescore_policy"],
        )
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                source_sha256=code,
                original_binding=binding,
                checkpoint_binding=checkpoint_binding,
                environment=environment,
                process_interval=[START, STOP],
                retained_interval=[FIRST, STOP],
                parent_audit_sha256=AUDIT_SHA,
                diagnosis_sha256=DIAGNOSIS_SHA,
                initialize_count=1,
                restore_count=5,
                new_begin_calls=0,
                external_pose_resets=0,
                post_fit_candidate_selection_only=True,
                new_gradient_objective=False,
                new_scoring_policy=tracker._context()["instance_rescore_policy"],
                precision="float64",
                allocator_fraction=0.25,
                reserved_limit_bytes=4 * 1024**3,
                frame_free_MiB=2048,
                automatic_adoption=False,
                physical_ID_truth=False,
            ),
        )
        evidence.final_guard()
        print(json.dumps(dict(event="instance_rescore_imported", through_frame=START)), flush=True)

        def guard():
            qualified.check_headroom(prior.cuda_headroom())
            qualified.check_memory(qualified.qualification.memory_sample())

        tick = time.perf_counter()
        reader = original.SourceSequenceReader(inputs.chunk_roots[16], inputs.input_root)
        require(reader.frames[0] == START and reader.frames[-1] + 1 == STOP, "wrong source chunk")
        iterator = original.bounded_frames(reader, first=START, stop=STOP)
        reader_seconds = time.perf_counter() - tick
        try:
            rows, timings = drive(
                tracker, iterator, out, synchronize=torch.cuda.synchronize, guard=guard
            )
        finally:
            iterator.close()
        out.write("diagnostic_index.json", dict(rows=rows, process_interval=[START, STOP]))
        parent = dict(arrays=arrays, metadata=metadata)
        path = destination / "checkpoints" / f"through_{STOP:06d}"
        saved, _, checkpoint = base.preserve_checkpoint(tracker, parent, path, checkpoint_binding)
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
            "finish changed the saved history backtrace",
        )
        require(
            np.array_equal(candidate["frame_idx"], np.arange(FIRST, STOP)),
            "candidate extent differs",
        )
        out.write("candidate.npz", candidate)
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                process_interval=[START, STOP],
                native_frames=STOP - START,
                retained_interval=[FIRST, STOP],
                parent_audit_sha256=AUDIT_SHA,
                parent_checkpoint_sha256=base.audit.CHECKPOINT_SHA,
                candidate_arrays=base.array_signature(candidate),
                checkpoint_manifest_sha256=checkpoint["manifest_sha256"],
                initialize_count=1,
                restore_count=5,
                new_begin_calls=0,
                external_pose_resets=0,
                finish_calls=1,
                all39_roundtrip=True,
                prior_history_prefix_unchanged=True,
                finish_matches_checkpoint_backtrace=True,
                diagnostic_frames=len(rows),
                checkpoint=checkpoint,
                timings=dict(reader_seconds=reader_seconds, **timings),
                post_fit_candidate_selection_only=True,
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
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--tracker-sha", required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())

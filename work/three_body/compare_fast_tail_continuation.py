"""Compare immutable two/five-minute milestones against the same Original horizon.

The ongoing process need not have finished. An explicit milestone-summary SHA
binds its candidate and checkpoint header; a separate optional successful final
seal is required to claim that the whole five-minute process has completed.
Checkpoint payloads are not reopened. Strict native SL support is recomputed on
CPU; female endpoints cover only minute one and do not establish all-three IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import compare_fast_tail_two_minutes as previous
import continue_fast_tail_five_minutes as runner
import numpy as np

producer, metrics, proof = previous.producer, previous.metrics, runner.proof
WORK, REPO, require = producer.WORK, producer.REPO, proof.require
FIRST, START, HALF = 49800, 66400, 76360
STAGE = "fast_tail_continuation_comparison_v1"
BRANCH = WORK / runner.STAGE
RUNNER_SHA = "d90e9eb49038dac4fd1e18068334ba89eb8df1a813a37c6aec071c20db1a61a6"
PREVIOUS_SHA = "e07b51f364f5d6f85e8e26d34a0051145412df29e34c91d379d2141aec9b649d"
AUDIT_SHA = "77bc769867edeb3f87c84abaf6f6c8f438b4f30f9a92c747b5e2a51e02d86188"
ORIGINALS = {
    86320: dict(
        root=previous.ORIGINAL,
        seal=previous.ORIGINAL_SHA,
        operation=previous.ORIGINAL_OPERATION_SHA,
        audit=previous.ORIGINAL_AUDIT_SHA,
        candidate=previous.ORIGINAL_CANDIDATE_SHA,
        restores=5,
    ),
    116200: dict(
        root=WORK / "long_remaining_receipted_supervisor_v1/phases/f106240-116200/tracking",
        seal="e99be2c7ea160752e93c52df924a77decebcd9d39955594c1dfda6cc0bb53005",
        operation="8d312514dc78f9c38f685d9a090442756872263b4301615b6a600ee2c99aafaa",
        audit="5f52d7859245cb2d72ddd2337a45fdf138ffc6aedc0059f9d3a49b2c4ba58ca7",
        candidate="c4e69b6e0d6403c7b4538f1fda3aa9feaf1bff8af8b3ac92acf548e913432a8c",
        restores=8,
    ),
}


def source_code():
    code = previous.source_code()
    for module, digest in ((runner, RUNNER_SHA), (previous, PREVIOUS_SHA)):
        require(code.get(str(Path(module.__file__).resolve())) == digest, "pinned caller changed")
    return code


def intervals(stop):
    require(type(stop) is int and stop in ORIGINALS, "registered two/five-minute horizon required")
    result = {"full": (START, stop)}
    for number, first in enumerate(range(START, stop, 9960), 1):
        result[f"minute_{number}"] = (first, first + 9960)
    return result


def json_input(path, evidence, digest=None):
    evidence.add(path, digest)
    return proof.read_json(path)


def custom_member(seal, root, name, evidence):
    """Read a consumed member of a custom size_bytes/string-input seal."""
    path = (root / name).resolve(strict=True)
    row = seal["outputs"].get(name)
    require(
        path.is_relative_to(root.resolve())
        and row is not None
        and Path(row["path"]).resolve() == path
        and path.stat().st_size == row["size_bytes"],
        "custom sealed member differs: " + name,
    )
    evidence.add(path, row["sha256"])
    return path


def validate_milestone(summary, stop):
    expected = dict(
        schema=runner.STAGE + "_milestone",
        status="ok",
        horizon_stop=stop,
        scientific_interval=[START, stop],
        native_frames=stop - START,
        initialize_count=1,
        restore_count=5,
        parent_audit_sha256=AUDIT_SHA,
        external_pose_resets=0,
        finish_calls_so_far=0,
        all39_roundtrip=True,
        prior_history_prefix_unchanged=True,
        physical_ID_truth=False,
        five_minute_candidate=stop == runner.STOP,
        whole_process_completion_pending=True,
    )
    require(type(stop) is int and stop in ORIGINALS, "unsupported milestone")
    require(
        all(type(summary.get(k)) is type(v) and summary[k] == v for k, v in expected.items()),
        "milestone extent/lifecycle/proof differs",
    )
    require(
        summary["source_sha256"].get(str(Path(runner.__file__).resolve())) == RUNNER_SHA,
        "milestone did not execute pinned continuation",
    )


def audited_parent(evidence):
    audit = runner.audit
    seal = json_input(runner.AUDIT_ROOT / "audit.provenance.json", evidence, AUDIT_SHA)
    require(seal["stage"] == audit.STAGE and seal["status"] == "ok", "first-minute audit failed")
    summary = proof.read_json(custom_member(seal, runner.AUDIT_ROOT, "summary.json", evidence))
    require(
        summary["committed_interval"] == [START, HALF]
        and summary["eligible_for_exact_state_restore"] is True
        and summary["checkpoint_manifest_sha256"] == audit.CHECKPOINT_SHA
        and summary["checkpoint_all39_comparison"]["exact"] is True
        and len(summary["checkpoint_all39_comparison"]["all39"]) == 39
        and all(
            v["exact"] is True for v in summary["checkpoint_all39_comparison"]["all39"].values()
        )
        and summary["ordered_beam_comparison_frames"] == 9960
        and summary["original_run_completed"] is False,
        "audited full parent state is incomplete",
    )
    previous.proof21(summary["candidate_all21_comparison"])
    manifest = json_input(audit.CHECKPOINT / "manifest.json", evidence, audit.CHECKPOINT_SHA)
    meta_path = audit.CHECKPOINT / "metadata.json"
    require(
        seal["inputs"].get(str(meta_path)) == manifest["metadata"]["sha256"], "parent meta edge"
    )
    metadata = json_input(meta_path, evidence, manifest["metadata"]["sha256"])
    require(
        metadata["previous_frame"] == HALF - 1
        and metadata["processed_frames"] == HALF - FIRST
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 4,
        "wrong parent lifecycle",
    )
    for path, digest in manifest["bindings"]["experimental_code_sha256"].items():
        require(seal["inputs"].get(path) == digest, "unaudited parent source")
        evidence.add(path, digest)
    return dict(seal=seal, summary=summary, manifest=manifest, metadata=metadata)


def validate_checkpoint(summary, manifest, metadata, parent, stop):
    fixed = manifest["bindings"]
    expected = dict(
        schema=runner.STAGE + "_checkpoint",
        original_binding=summary["original_binding"],
        source_sha256=summary["source_sha256"],
        audited_parent_sha256=AUDIT_SHA,
        parent_checkpoint_sha256=runner.audit.CHECKPOINT_SHA,
        scientific_interval=[START, runner.STOP],
        process_interval=[HALF, runner.STOP],
    )
    require(fixed == expected, "milestone checkpoint bindings differ")
    canonical = json.dumps(
        fixed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    require(
        manifest["schema"] == "medaka_tracker_checkpoint_v1"
        and manifest["status"] == "ok"
        and manifest["bindings_sha256"] == hashlib.sha256(canonical).hexdigest()
        and len(manifest["arrays"]) == 39
        and manifest["arrays"]["history_offsets"]["shape"] == [stop - FIRST + 1]
        and manifest["arrays"]["state_pose"]["shape"] == [6, 3, 3, 3],
        "complete retained checkpoint header required",
    )
    require(
        metadata["context"] == parent["metadata"]["context"]
        and metadata["record_fields"] == parent["metadata"]["record_fields"]
        and metadata["state_fields"] == parent["metadata"]["state_fields"]
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 5
        and metadata["previous_frame"] == stop - 1
        and metadata["processed_frames"] == stop - FIRST,
        "milestone state context/lifecycle differs",
    )
    row = summary["checkpoint"]
    require(
        row["arrays"] == 39
        and row["payload_bytes"] == manifest["payload"]["bytes"]
        and all(
            row[k] == metadata[k]
            for k in ("previous_frame", "processed_frames", "initialize_count", "restore_count")
        ),
        "milestone checkpoint row differs",
    )
    require(
        parent["manifest"]["bindings"]["original_binding"] == summary["original_binding"],
        "parent original input binding differs",
    )


def whole_process_completion(branch, summary_path, milestone_sha, evidence, final_sha=None):
    if final_sha is None:
        return False
    require(proof.valid_sha(final_sha), "explicit final successful tracking seal SHA required")
    seal = json_input(branch / "tracking.provenance.json", evidence, final_sha)
    require(
        seal["stage"] == runner.STAGE and seal["status"] == "ok", "whole process not successful"
    )
    path = custom_member(seal, branch, summary_path.relative_to(branch).as_posix(), evidence)
    require(evidence.paths[str(path)] == milestone_sha, "final seal names a different milestone")
    final = proof.read_json(custom_member(seal, branch, "summary.json", evidence))
    expected = dict(
        schema=runner.STAGE,
        status="ok",
        scientific_interval=[START, runner.STOP],
        process_interval=[HALF, runner.STOP],
        native_frames=49800,
        process_frames=39840,
        initialize_count=1,
        restore_count=5,
        external_pose_resets=0,
        new_begin_calls=0,
        finish_calls=1,
        physical_ID_truth=False,
    )
    require(all(final.get(k) == v for k, v in expected.items()), "final process extent/lifecycle")
    require(
        len(final["checkpoints"]) == 24
        and [r["previous_frame"] + 1 for r in final["checkpoints"]]
        == list(range(HALF + 1660, runner.STOP + 1660, 1660)),
        "whole process checkpoint sequence differs",
    )
    custom_member(seal, branch, "candidate.npz", evidence)
    require(
        seal["inputs"].get(str(Path(runner.__file__).resolve())) == RUNNER_SHA, "final code edge"
    )
    return True


def milestone_metadata(stop, milestone_sha, evidence, final_sha=None):
    require(proof.valid_sha(milestone_sha), "explicit immutable milestone summary SHA required")
    branch = BRANCH.resolve(strict=True)
    summary_path = branch / f"milestones/through_{stop:06d}/summary.json"
    summary = json_input(summary_path, evidence, milestone_sha)
    validate_milestone(summary, stop)
    parent = audited_parent(evidence)
    checkpoint = branch / f"checkpoints/through_{stop:06d}"
    require(
        Path(summary["checkpoint"]["path"]).resolve() == checkpoint, "checkpoint escaped milestone"
    )
    manifest = json_input(
        checkpoint / "manifest.json", evidence, summary["checkpoint"]["manifest_sha256"]
    )
    require(
        manifest["metadata"]["file"] == "metadata.json"
        and manifest["payload"]["file"] == "arrays.npz",
        "codec filenames",
    )
    metadata = json_input(checkpoint / "metadata.json", evidence, manifest["metadata"]["sha256"])
    validate_checkpoint(summary, manifest, metadata, parent, stop)
    require(
        (checkpoint / "arrays.npz").stat().st_size == manifest["payload"]["bytes"],
        "checkpoint payload size differs",
    )
    for path, digest in summary["source_sha256"].items():
        source = Path(path).resolve(strict=True)
        require(source.is_relative_to(REPO), "source outside workspace")
        evidence.add(source, digest)
        evidence.add(branch / "code_snapshot" / source.relative_to(REPO), digest)
    complete = whole_process_completion(branch, summary_path, milestone_sha, evidence, final_sha)
    return dict(
        summary=summary,
        metadata=metadata,
        parent=parent,
        manifest=manifest,
        binding=summary["original_binding"],
        whole_five_minute_process_complete=complete,
    )


def original_metadata(new, stop, evidence):
    fixed = ORIGINALS[stop]
    root = fixed["root"]
    bundle = producer.read_seal(
        root,
        "continuation.provenance.json",
        "three_body_long_continuation",
        fixed["seal"],
        evidence,
    )
    operation = producer.read_seal(
        root.parent,
        "operation.provenance.json",
        "long_receipted_phase_operation_v1",
        fixed["operation"],
        evidence,
    )
    previous.input_edge(operation["seal"], root / "continuation.provenance.json", fixed["seal"])
    audit_path = producer.consumed(operation, "audit.json", evidence)
    require(evidence.paths[str(audit_path)] == fixed["audit"], "Original audited result changed")
    audit = proof.read_json(audit_path)
    binding = bundle["seal"]["params"]["bindings"]
    require(
        binding == new["binding"]
        and bundle["seal"]["params"]["through_frame"] == stop
        and bundle["seal"]["params"].get("all_input_source_snapshot_output_sha_unchanged") is True
        and audit["status"] == "ok"
        and audit["frame_range"] == [FIRST, stop]
        and audit["candidate_sha256"] == fixed["candidate"]
        and audit["all_39_checkpoint_arrays_verified"] is True
        and len(audit["all_21_candidate_arrays_exact_backtrace"]) == 21
        and audit["logical_initialization_count"] == 1
        and audit["checkpoint_restore_count"] == fixed["restores"]
        and audit["external_pose_reset_count"] == 0,
        "Original same-horizon binding/audit differs",
    )
    metadata = proof.read_json(producer.consumed(bundle, "checkpoint/metadata.json", evidence))
    require(
        metadata["initialize_count"] == 1
        and metadata["restore_count"] == fixed["restores"]
        and metadata["previous_frame"] == stop - 1
        and metadata["processed_frames"] == stop - FIRST,
        "Original checkpoint lifecycle differs",
    )
    # The audited Fast import records the same original66400 state and context.
    migration_path = runner.audit.PARENT / "state_import.json"
    migration_sha = new["parent"]["manifest"]["bindings"]["state_import_sha256"]
    require(
        new["parent"]["seal"]["inputs"].get(str(migration_path)) == migration_sha,
        "audited migration edge",
    )
    migration = json_input(migration_path, evidence, migration_sha)
    parent_bundle = producer.read_seal(
        producer.original.PARENT,
        "continuation.provenance.json",
        "three_body_long_continuation",
        producer.PARENT_SHA,
        evidence,
    )
    parent_meta = proof.read_json(
        producer.consumed(parent_bundle, "checkpoint/metadata.json", evidence)
    )
    require(
        parent_bundle["seal"]["params"]["bindings"] == binding
        and parent_bundle["seal"]["params"]["through_frame"] == START
        and migration["original_metadata"] == parent_meta
        and migration["old_context"] == metadata["context"] == parent_meta["context"]
        and migration["new_context"] == new["metadata"]["context"],
        "same original66400 parent/context required",
    )
    for path, digest in binding["code_sha256"].items():
        evidence.add(path, digest)
    return dict(bundle=bundle, binding=binding, metadata=metadata, audit=audit)


def load_observations(new, reference, stop, evidence):
    binding = reference["binding"]
    shared = binding["inputs"]["seed"]["SL_and_calibration_sha256"]
    require(len(shared) == 7 and binding == new["binding"], "same SL3/calibration4 required")
    for path, digest in shared.items():
        evidence.add(path, digest)
        previous.input_edge(reference["bundle"]["seal"], path, digest)
    paths = {
        view: [
            Path(p)
            for p in shared
            if Path(p).name == previous.old.prior.EXPERIMENT + "__" + view + ".npz"
        ]
        for view in previous.old.prior.VIEWS
    }
    require(all(len(p) == 1 for p in paths.values()), "one registered raw SL cache per view")
    calibration = [Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"]
    require(len(calibration) == 1, "one registered calibration")
    rig = previous.old.prior.load_rig(calibration[0])
    info = binding["inputs"]["seed"]
    seed = np.asarray(
        json_input(info["registration_path"], evidence, info["registration_sha256"])["pose_xyz"],
        np.float64,
    )
    common, context = producer.original.phase.COMMON, reference["metadata"]["context"]
    require(
        context["initial_pose"] == common.digest_array(seed)
        and context["cameras"] == common.normalize([asdict(c) for c in rig.cameras]),
        "seed or refraction calibration differs",
    )
    nodes = previous.old.prior.RegisteredNodes(
        {v: p[0] for v, p in paths.items()}, np.arange(START, stop, dtype=np.int64)
    )
    return nodes, rig, seed, shared


def load_milestone_candidate(new, stop, evidence):
    row = new["summary"]["candidate"]
    path = (BRANCH / f"milestones/through_{stop:06d}/candidate.npz").resolve(strict=True)
    require(
        Path(row["path"]).resolve() == path and path.stat().st_size == row["size_bytes"],
        "milestone candidate path/size",
    )
    evidence.add(path, row["sha256"])
    with np.load(path, allow_pickle=False) as z:
        candidate = {k: z[k].copy() for k in z.files}
    metrics.validate_candidate(candidate, FIRST, stop)
    return candidate, dict(
        path=str(path), sha256=evidence.paths[str(path)], arrays=previous.signature(candidate)
    )


def run(args):
    windows = intervals(args.horizon_stop)
    require(
        args.execute is True and proof.valid_sha(args.milestone_sha),
        "explicit execution and milestone SHA required",
    )
    destination = args.output.resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(),
        "fresh workspace result required",
    )
    evidence, out = producer.original.phase.Digests(), proof.Output(destination)
    code, started, tick = source_code(), proof.utc(), time.perf_counter()
    stop = args.horizon_stop
    try:
        require(not producer.torch.cuda.is_initialized(), "CPU-only comparison required")
        for path, digest in code.items():
            evidence.add(path, digest)
            out.write(
                "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), Path(path).read_bytes()
            )
        new = milestone_metadata(stop, args.milestone_sha, evidence, args.final_tracking_sha)
        reference = original_metadata(new, stop, evidence)
        binding = reference["binding"]
        producer.original.cv2.setNumThreads(binding["opencv_threads"])
        producer.torch.set_num_threads(binding["torch_cpu_threads"])
        producer.torch.set_num_interop_threads(1)
        environment = producer.runtime(binding, gpu=False)
        original, original_ref = previous.load_candidate(
            reference["bundle"], "candidate.npz", evidence, stop
        )
        require(
            original_ref["sha256"] == ORIGINALS[stop]["candidate"]
            and original_ref["arrays"]
            == reference["audit"]["all_21_candidate_arrays_exact_backtrace"],
            "Original all21 audited arrays differ",
        )
        tail, tail_ref = load_milestone_candidate(new, stop, evidence)
        snapshot_bundle = producer.read_seal(
            producer.TAIL,
            "tracking.provenance.json",
            "tail_one_minute_continuous",
            producer.TAIL_SHA,
            evidence,
        )
        snapshot, snapshot_ref = previous.load_candidate(
            snapshot_bundle, "candidate.npz", evidence, HALF
        )
        require(
            new["parent"]["seal"]["inputs"].get(snapshot_ref["path"]) == snapshot_ref["sha256"],
            "first-minute snapshot differs from audited parent backtrace",
        )
        candidates, refs = (
            dict(original=original, fast_tail=tail),
            dict(original=original_ref, fast_tail=tail_ref),
        )
        before = {k: previous.signature(v) for k, v in candidates.items()}
        snapshot_before = previous.signature(snapshot)
        nodes, rig, seed, shared = load_observations(new, reference, stop, evidence)
        nodes_before = {
            v: previous.signature(dict(points=p, offsets=o)) for v, (p, o) in nodes.data.items()
        }
        anchors = previous.female_anchors(shared, evidence)
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                horizon_stop=stop,
                intervals=windows,
                candidate_references=refs,
                first_minute_snapshot=snapshot_ref,
                milestone_sha256=args.milestone_sha,
                final_tracking_sha256=args.final_tracking_sha,
                source_sha256=code,
                original_source_seal_sha256=ORIGINALS[stop]["seal"],
                original_audit_sha256=ORIGINALS[stop]["audit"],
                first_minute_audit_sha256=AUDIT_SHA,
                same_original66400_parent_sha256=producer.PARENT_SHA,
                shared_SL_calibration_sha256=shared,
                environment=environment,
                support_gates=dict(ray_mm=2.0, native_pixel=12.0, block_frames=1660),
                checkpoint_payloads_read_here=0,
                complete_output_audit_repeated=False,
                fast_mask_payload_lineage_verified_here=False,
                same_registered_input_binding=True,
                numeric_checkpoint_proof_scope=(
                    "Producer readback plus audited first minute; header verified here only"
                ),
                physical_ID_truth=False,
            ),
        )
        methods = {}
        for arm, candidate in candidates.items():
            part = {
                k: candidate[k][START - FIRST :]
                for k in ("frame_idx", "pose_xyz", "observed", "sources")
            }

            def save_support(meta, arrays, arm=arm):
                first, end = meta["frame_range"]
                out.write(f"support/{arm}_{first:06d}-{end:06d}.npz", arrays)
                print(
                    json.dumps(dict(event="continuation_strict_support", arm=arm, **meta)),
                    flush=True,
                )

            support = previous.old.prior.support_audit.verify_support_partitioned(
                part, nodes, rig, block_frames=1660, on_block=save_support
            )
            values = {}
            for name, (first, end) in windows.items():
                value, motion, shape = metrics.interval_metrics(candidate, seed, first, end)
                values[name] = value
                out.write(f"motion/{arm}_{name}.json", motion)
                if name == "full":
                    out.write(f"geometry/{arm}.npz", shape)
            methods[arm] = dict(
                intervals=values,
                strict_support=support,
                female_first_minute=previous.old.prior.female_scores(candidate, anchors),
            )
        differences = {}
        for name, (first, end) in windows.items():
            value, rows, arrays = metrics.differences(original, tail, first, end)
            differences[name] = value
            out.write(f"differences/{name}_gain_loss.json", rows)
            if name == "full":
                out.write("differences/all_native_transitions.npz", arrays)
        horizon, arrays = metrics.horizon_changes(snapshot, tail, START, HALF)
        out.write("horizon/first_minute_changes.json", horizon)
        out.write("horizon/first_minute_changes.npz", arrays)
        require(
            all(previous.signature(v) == before[k] for k, v in candidates.items())
            and previous.signature(snapshot) == snapshot_before,
            "candidate input mutated",
        )
        require(
            all(
                previous.signature(dict(points=p, offsets=o)) == nodes_before[v]
                for v, (p, o) in nodes.data.items()
            ),
            "raw SL input mutated",
        )
        require(not producer.torch.cuda.is_initialized(), "comparison initialized CUDA")
        summary = dict(
            schema=STAGE,
            status="ok",
            horizon_stop=stop,
            native_interval=[START, stop],
            native_frames=stop - START,
            intervals=windows,
            methods=methods,
            support_delta=differences,
            candidate_references=refs,
            first_minute_horizon_changes=horizon,
            whole_five_minute_process_complete=new["whole_five_minute_process_complete"],
            milestone_is_immutable_completed_checkpoint=True,
            final_tracking_seal_verified=args.final_tracking_sha is not None,
            fast_mask_payload_lineage_verified_here=False,
            same_registered_input_binding=True,
            same_parent_and_horizon=True,
            logical_initialization_count=1,
            restore_count_by_arm=dict(original=ORIGINALS[stop]["restores"], fast_tail=5),
            external_pose_resets=0,
            strict_support_recomputed_arms=2,
            strict_support_native_frames=2 * (stop - START),
            checkpoint_numeric_payloads_opened=0,
            candidate_all21_and_source_arrays_unchanged=True,
            physical_ID_truth=False,
            all_three_ID_truth=False,
            swap_zero_verified=False,
            adopted=False,
            review_modified=False,
            raw_decoded_frames=0,
            CUDA_used=False,
            image_evaluation_included=False,
            identity_solver_included=False,
            female_endpoint_frames=list(previous.old.old.ANCHOR_FRAMES),
            female_truth_interval=[START, HALF],
            requested_female_anchors=11,
            elapsed_seconds_before_final_guards=time.perf_counter() - tick,
            limitations=[
                "Same numeric slots are not independently verified physical fish identities.",
                "Strict2mm/12px SL support is an engineering gate, not human identity truth.",
                "Female eleven endpoints cover minute one only; "
                "male and later IDs remain unverified.",
                "A longer backtrace may revise minute one; "
                "this is separate from same76360 qualification.",
                "Only checkpoint headers and size are checked here; "
                "all39 numeric proof is the producer's.",
                "Individual mask-byte consumption edges are not checked here; "
                "the pixel evaluator requires the successful final tracking seal.",
            ],
        )
        out.write("summary.json", summary)
        require(source_code() == code, "source closure changed")
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
        )
        with (destination / "comparison.provenance.json").open("xb") as stream:
            stream.write(proof.json_bytes(seal))
        raise
    with (destination / "comparison.provenance.json").open("xb") as stream:
        stream.write(proof.json_bytes(seal))
    print(json.dumps(dict(status="ok", horizon_stop=stop, output=str(destination))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-stop", type=int, choices=sorted(ORIGINALS), required=True)
    parser.add_argument("--milestone-sha", required=True)
    parser.add_argument("--final-tracking-sha")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())

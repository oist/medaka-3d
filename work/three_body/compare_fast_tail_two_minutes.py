"""Saved-only two-arm same86320 comparison; source review precedes execution.

Candidate21 and raw SL are consumed, not tracking checkpoint payloads. Reuse the
producer's sealed39/history/backtrace controls and the existing Original phase
audit without repeating complete output audits. Recompute strict2mm/12px source
support for both arms. No video, masks, model, fit, CUDA, ID solver or new labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import fast_tail_two_minute_metrics as metrics
import numpy as np
import run_fast_tail_two_minutes as producer

from medaka3d.eval.generative import MouthAnchor
from medaka3d.provenance import provenance

old, checks = metrics.old, producer.checks
WORK, REPO = producer.WORK, producer.REPO
FIRST, START, HALF, STOP = 49800, 66400, 76360, 86320
STAGE = "fast_tail_two_minute_comparison_v1"
ARMS = ("original", "fast_tail")
INTERVALS = {"full": (START, STOP), "first_minute": (START, HALF), "second_minute": (HALF, STOP)}
ORIGINAL = WORK / "long_remaining_receipted_supervisor_v1/phases/f076360-086320/tracking"
ORIGINAL_SHA = "f605619d94c59e2bd3313c00fdff32d8fa8a8e029319637d2a4d71961c80e763"
ORIGINAL_CANDIDATE_SHA = "972ddbe7e7f5bcb437016f9676133281af22494c869214722826efd991f87eaf"
ORIGINAL_OPERATION_SHA = "593e56d20e59069aad4a660e1de7128be90ee4d7b92d5b85dd84e96c4cd9dbb9"
ORIGINAL_AUDIT_SHA = "717fd6c03a3c191b4c4e424197bf94c1baf3a8598a7805903a36197a019f63ed"
ORIGINAL_PREVIOUS = WORK / "long_remaining_supervisor_v1/phases/f066400-076360/tracking"
ORIGINAL_PREVIOUS_SHA = "95098e39bbd4d1e09d7ef98e3a7c244fe89ab2caf36a3650f360d35fc8289382"
PRODUCER_SHA = "8efbba7abb6cdedd0b077b8dc953a7a1d10c268b8aab7844b17f25cc337862af"
PINS = {
    "fast_tail_two_minute_metrics.py": (
        "a3ebd24da07308938899695723090fbde20c4bbe27f5dd8fb9d441bef9ee9ddb"
    ),
    "run_fast_tail_two_minutes.py": PRODUCER_SHA,
    "compare_tail_one_minute.py": (
        "da5f50c1dcc1399464c9ca8edfd65287a9f037f78119eee17cef7c27be388737"
    ),
    "compare_bounded_one_minute.py": (
        "5e1a4f570afadf204f96f3c6e83d55a9089f6fa3a14cf1b4c54cc84e4fe76cd0"
    ),
    "compare_directional_one_minute.py": (
        "066f702d0c64ddf048fe5d9c65e6add64f92daf40025e90b50a698355e6707c3"
    ),
    "mask_transport_diagnostics.py": (
        "6567710637b4a3b69dd03106cc525a1f7fac1659a7ab54acadef3c98517f3e56"
    ),
    "audit_long_final.py": "9140be0d5401228f38fefa04b1d4d4e68ea672ab8a0793130fbd62952c3e61eb",
    "long_baseline_support.py": "7e7ed92ee8c2879c1a09a3b504c9f5190d8f2b270aa78168ac184cfffc4cb5e4",
}


def signature(arrays):
    return {
        k: dict(
            dtype=v.dtype.str,
            shape=list(v.shape),
            sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(),
        )
        for k, v in arrays.items()
    }


def source_code():
    code = {**producer.local_code(), str(Path(__file__).resolve()): checks.sha(__file__)}
    for name, digest in {**producer.SOURCE_PINS, **PINS}.items():
        checks.require(code.get(str(WORK / name)) == digest, "frozen source changed: " + name)
    return code


def input_edge(seal, path, digest):
    path = str(Path(path).resolve())
    item = seal["inputs"].get(path)
    checks.require(
        item is not None
        and Path(item["path"]).resolve() == Path(path)
        and item["sha256"] == digest,
        "missing/conflicting consumed input edge: " + path,
    )


def completion(summary):
    required = dict(
        schema=producer.STAGE,
        status="ok",
        two_minutes_completed=True,
        native_interval=[START, STOP],
        processed_full_history_interval=[FIRST, STOP],
        native_frames=19920,
        native_BGR_view_frames=59760,
        checkpoints=12,
        first_minute_exact_checkpoints=6,
        first_minute_all6_history19_exact_frames=9960,
        first_minute_same76360_candidate21_exact=True,
        final_candidate21_exact_backtrace=True,
        final_horizon_stop=STOP,
        first_minute_snapshot_horizon=HALF,
        logical_initialization_count=1,
        checkpoint_restore_count=4,
        processed_frames=36520,
        previous_frame=86319,
        restores_during_interval=0,
        new_begin_calls=0,
        external_pose_resets=0,
        finish_calls=1,
    )
    checks.require(
        all(summary.get(k) == v for k, v in required.items()),
        "completed fixed two-minute tracking required",
    )


def proof21(value):
    checks.require(
        value.get("exact") is True
        and value.get("all21_bytes_required") is True
        and len(value.get("fields", {})) == 21
        and all(v is True for v in value["fields"].values()),
        "all21 producer proof required",
    )


def tracking_metadata(branch, expected_sha, evidence):
    """Completion first: this function opens small metadata/source files only."""
    new = producer.read_seal(
        branch, "tracking.provenance.json", producer.STAGE, expected_sha, evidence
    )
    checks.require(
        new["seal"]["params"].get("all_source_input_snapshot_output_SHA_unchanged") is True,
        "successful producer final guards required",
    )
    summary = checks.read(producer.consumed(new, "summary.json", evidence))
    completion(summary)
    plan = checks.read(producer.consumed(new, "plan.json", evidence))
    checks.require(
        plan["schema"] == producer.STAGE
        and plan["native_interval"] == [START, STOP]
        and plan["contract_sha256"] == producer.PLAN_SHA
        and plan["source_sha256"].get(str(Path(producer.__file__).resolve())) == PRODUCER_SHA,
        "producer plan/source contract changed",
    )
    fixed = new["seal"]["params"]["bindings"]
    checks.require(
        fixed["schema"] == "fast_tail_two_minute_checkpoint_binding_v1"
        and fixed["native_interval"] == [START, STOP]
        and fixed["experimental_code_sha256"] == plan["source_sha256"],
        "producer checkpoint binding changed",
    )
    for path, digest in fixed["experimental_code_sha256"].items():
        evidence.add(path, digest)
        input_edge(new["seal"], path, digest)
        snap = producer.consumed(
            new, "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), evidence
        )
        checks.require(evidence.paths[str(snap)] == digest, "producer source snapshot changed")
    migration_path = producer.consumed(new, "state_import.json", evidence)
    checks.require(
        evidence.paths[str(migration_path)] == fixed["state_import_sha256"], "migration SHA changed"
    )
    migration = checks.read(migration_path)
    checks.require(
        migration["schema"] == "fast_tail_original_state_import_v1"
        and migration["status"] == "ok"
        and migration["all_arrays_byte_preserved"] is True
        and migration["new_begin_calls"] == 0
        and migration["added_restore_count"] == 1
        and migration["review_cleared_by_migration"] is False,
        "explicit original Fast migration required",
    )
    for stop in range(START + 1660, HALF + 1660, 1660):
        proof = checks.read(producer.consumed(new, f"controls/through_{stop:06d}.json", evidence))
        checks.require(
            proof["through_frame"] == stop
            and proof["exact"] is True
            and len(proof["all39"]) == 39
            and all(v["exact"] is True for v in proof["all39"].values())
            and proof["all19_history_and_offsets_included"] is True
            and proof["metadata_exact_after_declared_fast_context"] is True,
            "sealed first-minute state control incomplete",
        )
    for name, stop in (("candidate_through_076360", HALF), ("final_backtrace", STOP)):
        proof = checks.read(producer.consumed(new, "controls/" + name + ".json", evidence))
        proof21(proof)
        checks.require(proof["horizon_stop"] == stop, "producer candidate proof horizon differs")
    manifest = checks.read(
        producer.consumed(new, "checkpoints/through_086320/manifest.json", evidence)
    )
    meta = checks.read(producer.consumed(new, "checkpoints/through_086320/metadata.json", evidence))
    checks.require(
        manifest["bindings"] == fixed
        and meta["context"] == migration["new_context"]
        and meta["initialize_count"] == 1
        and meta["restore_count"] == 4
        and meta["processed_frames"] == 36520
        and meta["previous_frame"] == 86319,
        "final checkpoint metadata differs",
    )
    return dict(bundle=new, summary=summary, fixed=fixed, migration=migration, metadata=meta)


def original_metadata(new, evidence):
    old_bundle = producer.read_seal(
        ORIGINAL,
        "continuation.provenance.json",
        "three_body_long_continuation",
        ORIGINAL_SHA,
        evidence,
    )
    previous = producer.read_seal(
        ORIGINAL_PREVIOUS,
        "continuation.provenance.json",
        "three_body_long_continuation",
        ORIGINAL_PREVIOUS_SHA,
        evidence,
    )
    parent = producer.read_seal(
        producer.original.PARENT,
        "continuation.provenance.json",
        "three_body_long_continuation",
        producer.PARENT_SHA,
        evidence,
    )
    binding = old_bundle["seal"]["params"]["bindings"]
    checks.require(
        binding
        == previous["seal"]["params"]["bindings"]
        == parent["seal"]["params"]["bindings"]
        == new["fixed"]["original_binding"],
        "same original input/parent binding required",
    )
    checks.require(
        old_bundle["seal"]["params"]["through_frame"] == STOP
        and Path(old_bundle["seal"]["params"]["resume_phase"]).resolve() == ORIGINAL_PREVIOUS
        and previous["seal"]["params"]["through_frame"] == HALF
        and Path(previous["seal"]["params"]["resume_phase"]).resolve() == producer.original.PARENT
        and parent["seal"]["params"]["through_frame"] == START,
        "Original same-parent continuation edges differ",
    )
    op = producer.read_seal(
        ORIGINAL.parent,
        "operation.provenance.json",
        "long_receipted_phase_operation_v1",
        ORIGINAL_OPERATION_SHA,
        evidence,
    )
    for root, digest in (
        (ORIGINAL, ORIGINAL_SHA),
        (ORIGINAL_PREVIOUS, ORIGINAL_PREVIOUS_SHA),
        (producer.original.PARENT, producer.PARENT_SHA),
    ):
        input_edge(op["seal"], root / "continuation.provenance.json", digest)
    audit_path = producer.consumed(op, "audit.json", evidence)
    checks.require(evidence.paths[str(audit_path)] == ORIGINAL_AUDIT_SHA, "Original audit changed")
    audit = checks.read(audit_path)
    checks.require(
        audit["status"] == "ok"
        and audit["frame_range"] == [FIRST, STOP]
        and audit["all_39_checkpoint_arrays_verified"] is True
        and len(audit["all_21_candidate_arrays_exact_backtrace"]) == 21
        and audit["candidate_sha256"] == ORIGINAL_CANDIDATE_SHA
        and audit["logical_initialization_count"] == 1
        and audit["checkpoint_restore_count"] == 5
        and audit["external_pose_reset_count"] == 0,
        "Original terminal audit scope differs",
    )
    parent_meta = checks.read(producer.consumed(parent, "checkpoint/metadata.json", evidence))
    meta = checks.read(producer.consumed(old_bundle, "checkpoint/metadata.json", evidence))
    checks.require(
        meta["context"] == parent_meta["context"] == new["migration"]["old_context"]
        and new["migration"]["original_metadata"] == parent_meta,
        "same original initial state context required",
    )
    input_edge(
        new["bundle"]["seal"],
        producer.original.PARENT / "continuation.provenance.json",
        producer.PARENT_SHA,
    )
    for path, digest in binding["code_sha256"].items():
        evidence.add(path, digest)
    return dict(bundle=old_bundle, binding=binding, audit=audit, metadata=meta)


def load_candidate(bundle, name, evidence, stop):
    path = producer.consumed(bundle, name, evidence)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    metrics.validate_candidate(arrays, FIRST, stop)
    return arrays, dict(path=str(path), sha256=evidence.paths[str(path)], arrays=signature(arrays))


def load_observations(new, reference, evidence):
    binding = reference["binding"]
    shared = binding["inputs"]["seed"]["SL_and_calibration_sha256"]
    checks.require(len(shared) == 7, "same registered SL3/calibration4 required")
    for path, digest in shared.items():
        evidence.add(path, digest)
        input_edge(new["bundle"]["seal"], path, digest)
        input_edge(reference["bundle"]["seal"], path, digest)
    paths = {
        v: [Path(p) for p in shared if Path(p).name == old.prior.EXPERIMENT + "__" + v + ".npz"]
        for v in old.prior.VIEWS
    }
    checks.require(
        all(len(p) == 1 for p in paths.values()), "unique raw SL cache per view required"
    )
    paths = {v: p[0] for v, p in paths.items()}
    calibration = [Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"]
    checks.require(len(calibration) == 1, "unique registered calibration required")
    rig = old.prior.load_rig(calibration[0])
    info = binding["inputs"]["seed"]
    evidence.add(info["registration_path"], info["registration_sha256"])
    seed = np.asarray(checks.read(info["registration_path"])["pose_xyz"], np.float64)
    ctx = reference["metadata"]["context"]
    common = producer.original.phase.COMMON
    checks.require(
        ctx["initial_pose"] == common.digest_array(seed)
        and ctx["cameras"] == common.normalize([asdict(c) for c in rig.cameras]),
        "registered seed/calibration context changed",
    )
    nodes = old.prior.RegisteredNodes(paths, np.arange(START, STOP, dtype=np.int64))
    return nodes, rig, seed, shared


def female_anchors(shared, evidence):
    bundle = producer.read_seal(
        old.old.FEMALE,
        "evaluation.provenance.json",
        "long_female_endpoint_evaluation_160s",
        old.old.FEMALE_SHA,
        evidence,
    )
    rows = checks.read(producer.consumed(bundle, "anchor_matches.json", evidence))
    gt = producer.consumed(bundle, "original_female_sex_gt.csv", evidence)
    checks.require(
        evidence.paths[str(gt)] == old.prior.GT_SHA, "fixed female annotation version changed"
    )
    for path, digest in shared.items():
        input_edge(bundle["seal"], path, digest)
    selected = [r for r in rows if START <= r["frame"] < HALF]
    checks.require(
        tuple(r["frame"] for r in selected) == old.old.ANCHOR_FRAMES,
        "fixed eleven first-minute endpoints required",
    )
    return [MouthAnchor(**r) for r in selected]


def main(args):
    checks.require(
        len(args.expected_seal_sha256) == 64
        and all(c in "0123456789abcdef" for c in args.expected_seal_sha256),
        "explicit successful tracking SHA required",
    )
    branch, output = args.branch.resolve(strict=True), args.output.resolve()
    checks.require(
        branch == WORK / producer.STAGE and output.is_relative_to(WORK) and not output.exists(),
        "fixed new tracking branch and fresh comparison output required",
    )
    code, evidence = source_code(), producer.original.phase.Digests()
    output.mkdir(parents=True)
    outputs, before = {}, {}
    summary = dict(
        schema=STAGE,
        status="incomplete",
        native_interval=[START, STOP],
        horizon_stop=STOP,
        independent_ID_truth=False,
        all_three_ID_truth=False,
        adopted=False,
        CUDA_used=False,
        raw_decoded_frames=0,
        tracking_performed=False,
        review_modified=False,
        image_evaluation_included=False,
        identity_solver_included=False,
    )
    started = time.perf_counter()
    with provenance(STAGE, output, name="comparison") as record:

        def emit(name, value=None, arrays=None):
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if value is not None:
                producer.original.save_json(path, value)
            if arrays is not None:
                np.savez_compressed(path, **arrays)
            digest = checks.sha(path)
            outputs[str(path)] = digest
            record.output(name, path, sha256=digest)

        try:
            checks.require(not producer.torch.cuda.is_initialized(), "CPU-only comparison required")
            for path, digest in code.items():
                evidence.add(path, digest)
                snap = output / "code_snapshot" / Path(path).relative_to(REPO)
                snap.parent.mkdir(parents=True, exist_ok=True)
                snap.write_bytes(Path(path).read_bytes())
                checks.require(checks.sha(snap) == digest, "source snapshot differs")
                emit(snap.relative_to(output).as_posix())
            new = tracking_metadata(branch, args.expected_seal_sha256, evidence)
            reference = original_metadata(new, evidence)
            binding = reference["binding"]
            producer.original.cv2.setNumThreads(binding["opencv_threads"])
            producer.torch.set_num_threads(binding["torch_cpu_threads"])
            producer.torch.set_num_interop_threads(1)
            environment = producer.runtime(binding, gpu=False)
            candidates, candidate_refs = {}, {}
            for arm, bundle in (("original", reference["bundle"]), ("fast_tail", new["bundle"])):
                candidates[arm], candidate_refs[arm] = load_candidate(
                    bundle, "candidate.npz", evidence, STOP
                )
            checks.require(
                candidate_refs["original"]["sha256"] == ORIGINAL_CANDIDATE_SHA
                and candidate_refs["original"]["arrays"]
                == reference["audit"]["all_21_candidate_arrays_exact_backtrace"],
                "consumed Original all21 differ from fixed audit",
            )
            snapshot, snapshot_ref = load_candidate(
                new["bundle"], "candidate_through_076360.npz", evidence, HALF
            )
            before = {arm: signature(a) for arm, a in candidates.items()}
            before["snapshot"] = signature(snapshot)
            nodes, rig, seed, shared = load_observations(new, reference, evidence)
            node_before = {
                v: signature(dict(points=p, offsets=o)) for v, (p, o) in nodes.data.items()
            }
            anchors = female_anchors(shared, evidence)
            emit(
                "plan.json",
                dict(
                    schema=STAGE,
                    arms=list(ARMS),
                    horizon_stop=STOP,
                    intervals={k: list(v) for k, v in INTERVALS.items()},
                    candidate_references=candidate_refs,
                    snapshot_reference=snapshot_ref,
                    source_tracking_sha256=args.expected_seal_sha256,
                    source_tracking_code_sha256=PRODUCER_SHA,
                    old_original_seal_sha256=ORIGINAL_SHA,
                    old_original_audit_sha256=ORIGINAL_AUDIT_SHA,
                    same_original66400_parent_sha256=producer.PARENT_SHA,
                    shared_SL_calibration_sha256=shared,
                    environment=environment,
                    support_reprojection=dict(
                        ray_gate_mm=2.0,
                        native_pixel_gate=12.0,
                        block_frames=1660,
                        arms=2,
                        frames_per_arm=19920,
                        body_node_slots_per_arm=179280,
                        view_node_slots_per_arm=537840,
                    ),
                    checkpoint_payloads_read_here=0,
                    complete_output_audit_repeated=False,
                    checkpoint_proofs_scope=(
                        "Reuse sealed producer controls and Original audit; not a fresh39 decode"
                    ),
                    first_minute_qualification_scope=(
                        "same76360 implementation control; distinct from new86320 backtrace"
                    ),
                    later_image_schedule=[66483 + 166 * k for k in range(120)],
                    image_schedule_only_no_render_or_pixel_scoring=True,
                    female_endpoint_frames=list(old.old.ANCHOR_FRAMES),
                    female_evaluation_interval=[START, HALF],
                    no_new_female_matching=True,
                    no_second_minute_ID_truth=True,
                ),
            )
            all_methods = {}
            for arm in ARMS:
                candidate = candidates[arm]
                part = {
                    k: candidate[k][START - FIRST :]
                    for k in ("frame_idx", "pose_xyz", "observed", "sources")
                }

                def save_support(meta, arrays, arm=arm):
                    start, stop = meta["frame_range"]
                    emit(f"support/{arm}_{start:06d}-{stop:06d}.npz", arrays=arrays)
                    print(
                        json.dumps(dict(event="strict_support_block", arm=arm, **meta)), flush=True
                    )

                geometry = old.prior.support_audit.verify_support_partitioned(
                    part, nodes, rig, block_frames=1660, on_block=save_support
                )
                methods = {}
                for name, (start, stop) in INTERVALS.items():
                    value, motion, shape = metrics.interval_metrics(candidate, seed, start, stop)
                    methods[name] = value
                    emit(f"motion/{arm}_{name}.json", motion)
                    if name == "full":
                        emit(f"geometry/{arm}.npz", arrays=shape)
                all_methods[arm] = dict(
                    intervals=methods,
                    all_native_strict_support=geometry,
                    female_first_minute=old.prior.female_scores(candidate, anchors),
                )
            deltas = {}
            for name, (start, stop) in INTERVALS.items():
                value, changed, arrays = metrics.differences(
                    candidates["original"], candidates["fast_tail"], start, stop
                )
                deltas[name] = value
                emit(f"differences/{name}_gain_loss.json", changed)
                if name == "full":
                    emit("differences/all_native_transitions.npz", arrays=arrays)
            horizon, arrays = metrics.horizon_changes(
                snapshot, candidates["fast_tail"], START, HALF
            )
            emit("horizon/first_minute_changes.json", horizon)
            emit("horizon/first_minute_changes.npz", arrays=arrays)
            checks.require(
                all(signature(candidates[arm]) == before[arm] for arm in ARMS)
                and signature(snapshot) == before["snapshot"],
                "candidate input arrays were mutated",
            )
            checks.require(
                all(
                    signature(dict(points=p, offsets=o)) == node_before[v]
                    for v, (p, o) in nodes.data.items()
                ),
                "raw SL observations were mutated",
            )
            checks.require(not producer.torch.cuda.is_initialized(), "comparison initialized CUDA")
            summary.update(
                status="ok",
                native_frames=19920,
                intervals={k: list(v) for k, v in INTERVALS.items()},
                candidate_references=candidate_refs,
                methods=all_methods,
                support_delta=deltas,
                first_minute_horizon_changes=horizon,
                candidate_all21_and_source_arrays_unchanged=True,
                strict_support_recomputed_arms=2,
                strict_support_native_frames=39840,
                checkpoint_numeric_payloads_opened=0,
                complete_output_audit_repeated=False,
                producer_control_proofs_reused=True,
                original_audit_proofs_reused=True,
                logical_initialization_count=1,
                restore_count_by_arm=dict(original=5, fast_tail=4),
                same_parent_and_horizon=True,
                requested_female_anchors=11,
                requested_female_intervals=10,
                elapsed_seconds_before_final_guards=time.perf_counter() - started,
                limitations=[
                    "Same numeric slots, not independently aligned physical fish IDs.",
                    "SL source support is an engineering 2mm/12px gate, not human ground truth.",
                    "Female eleven endpoints apply only to the first minute; "
                    "no dense male/second-minute ID truth.",
                    "Different same-slot source rows alone do not prove a biological-ID switch.",
                    "Final backtrace may revise the first minute; same76360 exact controls "
                    "remain a separate implementation qualification.",
                    "Complete output/checkpoint QA is not repeated here; "
                    "raw-SL strict support is recomputed.",
                ],
            )
        except BaseException as error:
            summary.update(status="failed", error=repr(error))
            raise
        finally:
            try:
                checks.require(source_code() == code, "source closure changed")
                evidence.final_guard()
                for path, digest in evidence.paths.items():
                    record.input(path, path, sha256=digest)
                for path, digest in outputs.items():
                    checks.require(checks.sha(path) == digest, "saved output changed")
                record.param("all_consumed_source_input_output_SHA_unchanged", True)
            except BaseException as error:
                summary.update(status="failed", final_guard_error=repr(error))
                raise
            finally:
                emit("summary.json", summary)
    print(json.dumps({a: summary["methods"][a]["intervals"]["full"] for a in ARMS}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", type=Path, default=WORK / producer.STAGE)
    parser.add_argument("--expected-seal-sha256", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    main(parser.parse_args())

"""Disabled, separately named ten-second engineering-depth sensitivity caller.

Restore the audited FastTail state once at 76360 and process through 78020.
The -12 mm lower z boundary changes proposal and fitting domains. The sparse
mask loss definition, calibration, support/age/review rules remain inherited.
This is development reuse, not physical depth truth or an ID certification.
The frozen ten-second serial driver and diagnostic codec are reused directly.
"""

# ruff: noqa: I001 -- frozen caller must establish lab/src paths before variant import

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

# The frozen caller first establishes its lab/src import paths.
import run_instance_rescore_10s as diagnostic
import expanded_depth_instance_rescore_tracker as variant
import numpy as np

base = diagnostic.base
prior, proof, original, qualified = base.prior, base.proof, base.original, base.qualified
torch, WORK, REPO, require = base.torch, base.WORK, base.REPO, base.require
FIRST, START, STOP = 49800, 76360, 78020
STAGE = "expanded_depth_instance_rescore_ten_second_v1"
EXECUTION_READY = True
AUDIT_SHA = diagnostic.AUDIT_SHA
TRACKER_SHA = "522f4012fb45b2e1fc89850c5448991fa3979193f736d91fe81b5f6d0344fe57"
PINS = {
    "expanded_depth_instance_rescore_tracker.py": TRACKER_SHA,
    "sparse_instance_rescore_tracker.py": variant.SPARSE_TRACKER_SHA,
    "fast_tail_recovery_tracker.py": variant.FAST_TAIL_SHA,
    "sparse_instance_body_score.py": (
        "68c41510d64a41098c6c1daf5e98621723f37c2f0f35730a68bd0e896517a634"
    ),
    "instance_rescore_tracker.py": (
        "f625513ed379e17241ad120c8a454c56b398764a1341ab7756790062231863e5"
    ),
    "instance_body_score.py": diagnostic.CORE_SHA,
    "continue_fast_tail_five_minutes.py": diagnostic.BASE_SHA,
    "run_instance_rescore_10s.py": (
        "91a66646f89a6abb59b6d60bc1b357b383167c21eb9d40bc47491802d05e9544"
    ),
}
ROOTS = {
    "free": WORK / "grouped_mouth_two_frame_intersection_feasibility_v1",
    "bounded": WORK / "bounded_grouped_mouth_two_frame_feasibility_v1",
    "sparse": WORK / "sparse_instance_saved_ten_second_qualification_v1",
    "dense_control": WORK / "instance_rescore_ten_second_continuation_v1",
}
SEALS = {
    "free": (
        "feasibility.provenance.json",
        "08c5204744c671acde28cfc25ee9f3ff1cc04c95674fa67be21b2ca6e1399ef7",
    ),
    "bounded": (
        "feasibility.provenance.json",
        "9263422b90fa300c1fa857884fb3fc3cfc0ba0e38ce687bbece8cd3e37d367ee",
    ),
    "sparse": (
        "qualification.provenance.json",
        "941ca436a5e59fca5a63794483d315da3950402aafdb4ce13c34630e98014125",
    ),
    "dense_control": (
        "tracking.provenance.json",
        "a71ddab8dfef8d97fecf68c1d5b7a5d7300fe21dfe49b48bc6f1df91c840196a",
    ),
}


def source_closure():
    """Require the actual loaded new implementation and every inherited source pin."""
    code = prior.local_code()
    prior.source_contract(code)
    require(str(Path(__file__).resolve()) in code, "new caller missing from actual import closure")
    for name, digest in PINS.items():
        require(
            code.get(str((WORK / name).resolve())) == digest, "source missing or changed: " + name
        )
    require(
        (diagnostic.FIRST, diagnostic.START, diagnostic.STOP) == (FIRST, START, STOP),
        "frozen driver interval changed",
    )
    return code


def bound_member(root, seal, name, evidence):
    root = root.resolve(strict=True)
    path = (root / name).resolve(strict=True)
    require(path.is_relative_to(root), "evidence member escaped its run")
    row = seal["outputs"][name]
    require(
        path == Path(row["path"]).resolve() and path.stat().st_size == row["size_bytes"],
        "evidence output descriptor differs",
    )
    evidence.add(path, row["sha256"])
    return proof.read_json(path)


def validate_rationale(summaries):
    """Check saved diagnostic conclusions only; no rerun of the eight intersections."""
    free, bounded, sparse, control = (
        summaries[k] for k in ("free", "bounded", "sparse", "dense_control")
    )
    for summary in (free, bounded):
        require(
            [r["frame_idx"] for r in summary["records"]] == [77709, 77723],
            "wrong development frames",
        )
        require(
            all(len(r["trials"]) == 4 for r in summary["records"]), "all eight subsets required"
        )
    for fr, br in zip(free["records"], bounded["records"], strict=True):
        for f, b in zip(fr["trials"], br["trials"], strict=True):
            gate_names = {
                "finite",
                "forward",
                "ray_rms",
                "box",
                "anatomy",
                "source_exclusive",
                "mouth_direct",
                "chest_direct",
                "no_other_direct_support_loss",
            }
            require(
                set(f["gates"]) == set(b["gates"]) == gate_names, "missing spatial gate evidence"
            )
            require(
                (f["index"], f["views"], f["rows"]) == (b["index"], b["views"], b["rows"]),
                "free/bounded subsets differ",
            )
            require(
                f["mouth_xyz"] == b["free_point_xyz"] and f["gates"] == b["free_gates"],
                "free evidence binding differs",
            )
            xyz = np.asarray(f["mouth_xyz"])
            require(
                xyz.shape == (3,)
                and np.isfinite(xyz).all()
                and np.all(xyz >= [0, 0, -12])
                and np.all(xyz <= [192, 192, 186]),
                "free point differs beyond lower z",
            )
            require(
                -12.0 <= f["mouth_xyz"][2] < -6.0
                and f["gates"]["box"] is False
                and all(v is True for k, v in f["gates"].items() if k != "box"),
                "free rejection is not solely lower depth",
            )
            require(
                b["gates"]["box"] is True
                and b["gates"]["ray_rms"] is False
                and np.isfinite(b["ray_rms_mm"])
                and b["ray_rms_mm"] > 1.5
                and b["passes_existing_spatial_gates"] is False,
                "bounded diagnostic conclusion differs",
            )
    require(
        sparse["native_frames"] == 1660
        and sparse["views_compared"] == 4980
        and sparse["all_saved_dense_fields_BYTEEXACT"] is True
        and sparse["all_normalized_and_adjusted_costs_BYTEEXACT"] is True,
        "full saved sparse parity required",
    )
    require(
        sparse["saved_workload_equivalence_only"] is True
        and sparse["continuous_rollout_equivalence"] is False,
        "sparse qualification scope differs",
    )
    require(
        control["process_interval"] == [START, STOP]
        and control["retained_interval"] == [FIRST, STOP]
        and control["parent_audit_sha256"] == AUDIT_SHA
        and control["parent_checkpoint_sha256"] == base.audit.CHECKPOINT_SHA
        and control["initialize_count"] == 1
        and control["restore_count"] == 5
        and control["all39_roundtrip"] is True,
        "dense control source/extent differs",
    )


def load_rationale(evidence):
    summaries, seals, references = {}, {}, {}
    for role, root in ROOTS.items():
        filename, sha = SEALS[role]
        path = root / filename
        evidence.add(path, sha)
        seal = proof.read_json(path)
        require(
            seal["stage"] == root.name and seal["status"] == "ok",
            "successful rationale stage required",
        )
        summary = bound_member(root, seal, "summary.json", evidence)
        require(
            summary["schema"] == root.name and summary["status"] == "ok",
            "rationale summary differs",
        )
        summaries[role], seals[role] = summary, seal
        references[role] = dict(
            path=str(path), sha256=sha, summary=copy.deepcopy(seal["outputs"]["summary.json"])
        )
    require(
        seals["bounded"]["inputs"].get(str((ROOTS["free"] / SEALS["free"][0]).resolve()))
        == SEALS["free"][1],
        "bounded/free seal edge differs",
    )
    require(
        seals["sparse"]["inputs"].get(
            str((ROOTS["dense_control"] / SEALS["dense_control"][0]).resolve())
        )
        == SEALS["dense_control"][1],
        "saved sparse/dense seal edge differs",
    )
    validate_rationale(summaries)
    control_plan = bound_member(
        ROOTS["dense_control"], seals["dense_control"], "plan.json", evidence
    )
    return references, control_plan


def policy_binding(tracker):
    context = tracker._context()
    require(
        context["expanded_depth_policy"]["effective_from_native"] == START
        and context["sparse_distance_backend"]["effective_from_native"] == START,
        "wrong policy boundary",
    )
    return dict(
        new_scoring_policy=copy.deepcopy(context["instance_rescore_policy"]),
        distance_backend=copy.deepcopy(context["sparse_distance_backend"]),
        expanded_depth_policy=copy.deepcopy(context["expanded_depth_policy"]),
        scientific_scope=dict(
            post_fit_candidate_selection_only=False,
            lower_z_changed_mm=[-6.0, -12.0],
            proposal_and_fitting_domains_changed=True,
            mask_loss_definition_and_normalization_unchanged=True,
            new_differentiated_instance_loss=False,
            x_y_upper_bounds_calibration_RMS_anatomy_support_age_review_unchanged=True,
            development_reuse=True,
            physical_depth_or_ID_truth=False,
            automatic_adoption=False,
        ),
    )


def main(args):
    require(
        EXECUTION_READY and args.execute and args.tracker_sha == TRACKER_SHA,
        "source-only preparation; reviewed execution is not enabled",
    )
    destination = args.output.resolve()
    require(destination.is_relative_to(WORK) and not destination.exists(), "fresh output required")
    out, evidence = proof.Output(destination), original.phase.Digests()
    started, code, checkpoint = proof.utc(), {}, None
    try:
        code = source_closure()
        for path, digest in code.items():
            evidence.add(path, digest)
            out.write(
                "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), Path(path).read_bytes()
            )
        references, control_plan = load_rationale(evidence)
        store = qualified.ReceiptStore(args.receipt_root.resolve())
        print(
            json.dumps(dict(event="expanded_depth_validation_started", output=str(destination))),
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
        require(
            control_plan["original_binding"] == binding
            and control_plan["parent_audit_sha256"] == AUDIT_SHA,
            "dense control uses a different original input binding",
        )
        inputs, protocol = original.verified_inputs(binding, STOP, evidence, receipt_store=store)
        require(
            len(inputs.chunk_records) == len(inputs.chunk_roots) == 17,
            "registered 170s prefix required",
        )
        out.write(
            "input_index.json",
            dict(
                original_binding=binding,
                prefix_chunks=inputs.chunk_records,
                rationale_references=references,
            ),
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
        tracker = variant.ExpandedDepthInstanceRescoreContinuousTracker(
            inputs.rig,
            inputs.initial_pose_xyz.copy(),
            config,
            sparse_effective_from_native=START,
            **kwargs,
        )
        source = prior.FastTailRecoveryContinuousTracker(
            inputs.rig, inputs.initial_pose_xyz.copy(), config, **kwargs
        )
        imported = variant.import_fast_tail_state(tracker, source, arrays, metadata)
        base.lifecycle(tracker, START)
        out.write("state_import.json", imported)
        policies = policy_binding(tracker)
        checkpoint_binding = dict(
            schema=STAGE + "_checkpoint",
            original_binding=binding,
            source_sha256=code,
            parent_audit_sha256=AUDIT_SHA,
            parent_checkpoint_sha256=base.audit.CHECKPOINT_SHA,
            retained_interval=[FIRST, STOP],
            process_interval=[START, STOP],
            rationale_references=references,
            **policies,
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
                rationale_references=references,
                initialize_count=1,
                restore_count=5,
                new_begin_calls=0,
                external_pose_resets=0,
                precision="float64",
                allocator_fraction=0.25,
                reserved_limit_bytes=4 * 1024**3,
                frame_free_MiB=2048,
                automatic_adoption=False,
                physical_ID_truth=False,
                **policies,
            ),
        )
        evidence.final_guard()
        print(json.dumps(dict(event="expanded_depth_imported", through_frame=START)), flush=True)

        def guard():
            qualified.check_headroom(prior.cuda_headroom())
            qualified.check_memory(qualified.qualification.memory_sample())

        tick = time.perf_counter()
        reader = original.SourceSequenceReader(inputs.chunk_roots[16], inputs.input_root)
        require(reader.frames[0] == START and reader.frames[-1] + 1 == STOP, "wrong source chunk")
        iterator = original.bounded_frames(reader, first=START, stop=STOP)
        reader_seconds = time.perf_counter() - tick
        try:
            # The actual inherited exporter includes sparse + depth policies in
            # every companion. No wrapper, runtime patch or changed loop is used.
            rows, timings = diagnostic.drive(
                tracker, iterator, out, synchronize=torch.cuda.synchronize, guard=guard
            )
        finally:
            iterator.close()
        require(
            len(rows) == STOP - START
            and [r["frame_idx"] for r in rows] == list(range(START, STOP)),
            "diagnostic coverage differs",
        )
        out.write(
            "diagnostic_index.json", dict(rows=rows, process_interval=[START, STOP], **policies)
        )
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
            "finish changed saved history backtrace",
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
                rationale_references=references,
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
                physical_ID_truth=False,
                independent_evaluation_pending=True,
                automatic_adoption=False,
                **policies,
            ),
        )
        store.final_guard()
        for path, digest in store.evidence_hashes.items():
            evidence.add(path, digest)
        out.write("mask_validation_receipts.json", store.uses)
        require(source_closure() == code, "source closure changed")
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

"""One fresh admitted-body-point200 arm, with sealed prior controls kept read-only.

The prior 200-frame comparison is evidence for this new experiment, never a
per-frame pose input. Only its causal t0 records qualify the fresh initialization.
No detector, point model, query refresh, checkpoint restore or ID reset runs here.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import evaluate_mask_transport200 as evaluation
import numpy as np
import run_mask_transport200_v2 as previous
import torch

base, prepared, inputs, tracking = (
    previous.base,
    previous.prepared,
    previous.inputs,
    previous.tracking,
)
require = previous.require
WORK, REPO, FIRST, STOP, COUNT = (
    previous.WORK,
    previous.REPO,
    previous.FIRST,
    previous.STOP,
    previous.COUNT,
)
STAGE = "body_point200_tracking_v1"
ARM = "ct_body_points"
PREVIOUS = WORK / "mask_transport200_three_arm_tracking_v2"
PREVIOUS_SHA = "49c8771c175252ccc0cb6fdca565cc7834e4d1c8ff6496846a2ea94126d0feda"
EVALUATION = WORK / "mask_transport200_independent_evaluation_v1"
EVALUATION_SHA = "dbb38b01317d4171aa7109803e279dfab4c6913086b2ac3321f79f4bb1d4abf6"
PLAN = WORK / "NEWBODY_POINT_EVIDENCE_FOLLOWUP.md"
PLAN_SHA = "1409a0214ac2c8c2c3c17e13e6497b004290b9b844c8963f74fb258bcbbc15cb"
SIX_EVALUATION = WORK / "fixed_body_width200_independent_evaluation_v1"
SIX_EVALUATION_SHA = "a5b95006cf407e70e03a4368eba9576ea2df1bfebb9fe0c64c6b3bfb01b670db"
PACKETS = WORK / "body_point200_packet_preparation_v1"
PACKETS_SHA = "8e2f31d5ac9364b72ea899bbc06a48e6af9bccf15fc6de5760179677095d3862"
QUALIFICATION = WORK / "body_point_cuda_qualification_v1"
QUALIFICATION_SHA = "471dcdc35ae956dae3950d77093bdd2a68795711038db837d17fdd60a4a2c509"
EXECUTION_READY = True
SOURCE_PINS = {
    **previous.SOURCE_PINS,
    "work/three_body/body_point_transport.py": (
        "a532a395d12f8ee7019bfed27d1b2850bed86e5f6f38dad002c52c646f6c2406"
    ),
    "work/three_body/body_point_fitter.py": (
        "e7196c49f081a0e86c476ad4d49a34ad6cd638d0c0a8149d46c004ce275a68ac"
    ),
    "work/three_body/body_point_tracker.py": (
        "ca4ccc28bfb529c198e5079e78ff5897af41b878fe98fcfc65d7cf4333742877"
    ),
    "work/three_body/prepare_body_point200_packets.py": (
        "73bbaaf11fe4186e72e2fea310c7e7f4ff6cf604e27641c68c0e2024592089ca"
    ),
    "work/three_body/qualify_body_point_cuda.py": (
        "1b674a062d2f5d7c8ba9999644741ef6bb44aaae3d24b079b22cae92bd36a4a4"
    ),
    "work/three_body/evaluate_fixed_body_width200.py": (
        "ab1659274444a077d9b12376717bf1c3563c7433854f1d16196bc46599fa59cf"
    ),
    "work/three_body/run_mask_transport200_v2.py": (
        "71bdefe599465476faac0fc495ec820abbaf4a49c9fce9c6b226188b8fa760bc"
    ),
    "work/three_body/evaluate_mask_transport200.py": (
        "61c94354a8f0c1618234cedfdb2a0640b08a62fc66297bc60070a4021bbfdf0e"
    ),
    "work/three_body/mask_transport_gap_fitter.py": (
        "930ca3c88feb9cc709a62001d14c9515cf196d0d6a1917cec656beacec9c5c75"
    ),
    "work/three_body/mask_transport_gap_tracker.py": (
        "42875f30ede026d2e315044b30c4625b75b2f45c98610f6581d607d93aaa9789"
    ),
}


def validate_packet_preparation(plan, summary):
    require(
        plan["schema"] == summary["schema"] == "body_point200_packet_preparation_v1"
        and summary["status"] == "ok"
        and plan["native_interval"] == summary["native_interval"] == [FIRST, STOP]
        and summary["packet_count"] == 400
        and summary["evidence"] == ["lk", "ct_first"]
        and plan["implementation_plan_sha256"] == PLAN_SHA
        and plan["helper_sha256"] == SOURCE_PINS["work/three_body/body_point_transport.py"]
        and plan["samples_sha256"] == previous.SAMPLES_SHA
        and plan["point_sha256"] == previous.POINTS_SHA
        and plan["preparation_sha256"] == inputs.PREPARATION_SHA
        and summary["all_old_T_exact"] is True
        and summary["all_old_mass_exact"] is True
        and summary["all400_packet_readbacks_exact"] is True
        and summary["all400_query_ray_readbacks_exact"] is True
        and summary["new_model"] is False
        and summary["new_fit"] is False
        and summary["CUDA_initialized"] is False,
        "complete same-input packet preparation required",
    )


def load_prepared_packets(evidence):
    seal, members = base.verified(
        PACKETS,
        "preparation.provenance.json",
        "body_point200_packet_preparation_v1",
        PACKETS_SHA,
        evidence,
    )
    source = "work/three_body/prepare_body_point200_packets.py"
    require(
        seal["inputs"].get(str(REPO / source)) == SOURCE_PINS[source],
        "packet preparation producer differs",
    )
    plan, summary = (base.bound_json(PACKETS, members, n) for n in ("plan.json", "summary.json"))
    validate_packet_preparation(plan, summary)
    return members, plan, evaluation.load_tree(PACKETS, members, "initial_bank")


def validate_qualification(plan, summary):
    require(
        plan["schema"] == summary["schema"] == "body_point_cuda_qualification_v1"
        and summary["status"] == "ok"
        and summary["passed"] is True
        and summary["fitter_sha256"] == SOURCE_PINS["work/three_body/body_point_fitter.py"]
        and summary["dtype"] == "float64"
        and summary["batch"] == 2
        and summary["query_counts"] == [3, 2, 1]
        and summary["candidate_anchor"] == [2, 0]
        and summary["tolerance"] == dict(atol=1e-10, rtol=1e-10)
        and summary["graph_constructions"] == 1
        and summary["evaluated_poses"] == 15
        and summary["compared_adam_gradients"] == summary["optimizer_steps"] == 14
        and summary["strict_first_best_checked"] is True
        and summary["final_old_pixel_support_checked"] is True
        and summary["all_base_graph_comparisons_passed"] is True
        and summary["all_combined_gradient_comparisons_passed"] is True
        and summary["completed_fit"]["passed"] is True
        and summary["actual_saved_data"] is False,
        "completed finite CUDA numerical qualification required",
    )


def load_qualification(evidence):
    seal, members = base.verified(
        QUALIFICATION,
        "qualification.provenance.json",
        "body_point_cuda_qualification_v1",
        QUALIFICATION_SHA,
        evidence,
    )
    for source in (
        "work/three_body/qualify_body_point_cuda.py",
        "work/three_body/body_point_fitter.py",
    ):
        require(
            seal["inputs"].get(str(REPO / source)) == SOURCE_PINS[source],
            "CUDA qualification source differs: " + source,
        )
    plan, summary = (
        base.bound_json(QUALIFICATION, members, n) for n in ("plan.json", "summary.json")
    )
    validate_qualification(plan, summary)


def bind_previous_six(evidence):
    """Bind the completed decision evidence without loading any future poses."""
    seal, members = base.verified(
        SIX_EVALUATION,
        "evaluation.provenance.json",
        "fixed_body_width200_independent_evaluation_v1",
        SIX_EVALUATION_SHA,
        evidence,
    )
    summary = base.bound_json(SIX_EVALUATION, members, "summary.json")
    require(
        summary["status"] == "ok"
        and summary["native_interval"] == [FIRST, STOP]
        and summary["arms"] == ["original", "lk", "ct_first", "ct_gap", "static_base", "static_gap"]
        and summary["strict_candidate_support_rechecks"] == 1200
        and summary["original_four_CPU_image_control_exact_frames"] == 200
        and summary["unused_pixel_draws"] == 153600
        and summary["independent_ID_truth"] is False
        and summary["trajectory_or_review_modified"] is False,
        "completed six-arm comparison scope differs",
    )
    source = "work/three_body/evaluate_fixed_body_width200.py"
    require(
        seal["inputs"].get(str(REPO / source)) == SOURCE_PINS[source],
        "six-arm evaluation source differs",
    )


def validate_previous_evaluation(seal, summary):
    require(
        seal["inputs"].get(str(PREVIOUS / "tracking.provenance.json")) == PREVIOUS_SHA,
        "independent evaluation does not bind the prior successful comparison",
    )
    require(
        summary["status"] == "ok"
        and summary["native_interval"] == [FIRST, STOP]
        and summary["arms"] == list(previous.ARMS)
        and summary["strict_candidate_support_rechecks"] == 600
        and summary["unused_pixel_draws"] == 153600
        and summary["independent_ID_truth"] is False
        and summary["trajectory_or_review_modified"] is False,
        "prior independent comparison scope differs",
    )


def load_controls(evidence):
    seal, members = base.verified(
        PREVIOUS, "tracking.provenance.json", previous.STAGE, PREVIOUS_SHA, evidence
    )
    plan, summary, control, proof = (
        base.bound_json(PREVIOUS, members, name)
        for name in (
            "plan.json",
            "summary.json",
            "original_control.json",
            "original_reuse_qualification.json",
        )
    )
    evaluation.validate_run(plan, summary, control)
    evaluation.validate_reuse(plan, summary, proof)
    require(
        seal["inputs"].get(str(Path(previous.__file__).resolve()))
        == SOURCE_PINS["work/three_body/run_mask_transport200_v2.py"],
        "prior producer source differs",
    )
    evaluation_seal, evaluation_members = base.verified(
        EVALUATION, "evaluation.provenance.json", evaluation.STAGE, EVALUATION_SHA, evidence
    )
    independent = base.bound_json(EVALUATION, evaluation_members, "summary.json")
    validate_previous_evaluation(evaluation_seal, independent)
    # No prior candidate trajectory or future state is supplied to the new arm.
    initial = evaluation.load_tree(PREVIOUS, members, f"original/frames/native_{FIRST}")
    bank = evaluation.load_tree(PREVIOUS, members, "original/initial_anchors")
    anchors = tuple(SimpleNamespace(**row) for row in bank["anchors"])
    require(initial["fit"]["native"] == FIRST, "control must be causal t0")
    return plan, initial["records"], anchors, bank


def validate_execution(args):
    require(args.execute is True and args.device == "cuda", "explicit CUDA execution required")
    require(
        EXECUTION_READY
        and all(base.valid_sha(x) for x in (*SOURCE_PINS.values(), PACKETS_SHA, QUALIFICATION_SHA)),
        "body-point source/qualification is still in draft",
    )
    require(args.samples_seal_sha256 == previous.SAMPLES_SHA, "fixed samples SHA differs")


def run(args):
    validate_execution(args)
    destination = Path(args.output).resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(),
        "new workspace output required",
    )
    output, evidence, captured = base.Output(destination), base.Evidence(), {}
    started, start, last_saved = base.utc(), time.perf_counter(), None
    old_cv = cv2.getNumThreads()
    try:
        evidence.add(Path(__file__).resolve())
        evidence.add(PLAN, PLAN_SHA)
        for relative, digest in SOURCE_PINS.items():
            evidence.add(REPO / relative, digest)
        point_tracker = importlib.import_module("body_point_tracker")
        point_transport = importlib.import_module("body_point_transport")
        bind_previous_six(evidence)
        packet_members, packet_plan, packet_bank = load_prepared_packets(evidence)
        load_qualification(evidence)
        prior_plan, initial_records, initial_anchors, initial_bank = load_controls(evidence)
        previous.same_tree(packet_bank, initial_bank, "prepared/current causal bank")
        common, sample_members, points, reference = previous.load_inputs(
            evidence, previous.SAMPLES_SHA
        )
        body_points = point_transport.SavedBodyPointEvidence(points)
        cfg = common["original_context"]
        require(cfg == prior_plan["original_context"], "prior/new scientific context differs")
        require(cfg == packet_plan["original_context"], "prepared packet science context differs")
        binding = dict(
            schema=STAGE,
            arm=ARM,
            point_evidence_arm="ct_first",
            preparation_sha256=inputs.PREPARATION_SHA,
            point_sha256=previous.POINTS_SHA,
            samples_sha256=previous.SAMPLES_SHA,
            prior_tracking_sha256=PREVIOUS_SHA,
            prior_evaluation_sha256=EVALUATION_SHA,
            prior_six_evaluation_sha256=SIX_EVALUATION_SHA,
            body_point_preparation_sha256=PACKETS_SHA,
            cuda_qualification_sha256=QUALIFICATION_SHA,
            body_point_plan_sha256=PLAN_SHA,
            source_pins=SOURCE_PINS,
        )
        prepared.snapshot_sources(evidence, output, captured)
        environment = previous.runtime(reference["binding"])
        memory = [previous.gpu_memory("body_point_startup", minimum_gib=6)]
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                native_interval=[FIRST, STOP],
                arm=ARM,
                input_binding=binding,
                original_context=cfg,
                environment=environment,
                original_control_passed=True,
                prior_controls_read_only=True,
                prior_numeric_input="causal t0 records and anchors only; no prior trajectory/state",
                objective="base + 3*(body_point_owner + body_point_ray_gap)",
                auxiliary_domain=(
                    "admitted initial-group point rays; old body/view mass divided by S"
                ),
                renderer_width="original dynamic pose-derived radial and axial sigma",
                auxiliary_mass_policy="old fit W*valid*y*T mass, no invalid-ray renormalization",
                point_to_node_correspondence=False,
                D2_missing_fallback=False,
                gap_relative_coefficient=1.0,
                gap="sqrt(max(nearest_gaussian_mahalanobis_sq,1))-1",
                mask_coefficient=3.0,
                extra_proposals=0,
                maximum_pool=24,
                opt_steps=14,
                target_lineage="causal_t0_retained_parent",
                point_evidence_arm="ct_first",
                first_release_horizons=[previous.horizon(n) for n in range(FIRST, STOP)],
                no_external_pose_reset=True,
                no_ID_review_clearing=True,
                independent_ID_truth=False,
                temporal_holdout=False,
            ),
        )
        output.write("original_seed.json", common["seed"])
        output.write("query_rows.json", points.query_rows)
        tracker = point_tracker.BodyPointTracker(
            common["rig"],
            common["initial"].copy(),
            config=previous.TrackerConfig(**cfg["tracker_config"]),
            recovery_config=previous.NodeRecoveryConfig(**cfg["recovery_config"]),
            persistence_config=previous.PersistentNodeConfig(**cfg["persistence_config"]),
            first_native=FIRST,
            group_counts=points.group_counts,
            query_sha256=points.query_sha256,
            vote_provider=body_points.provider("ct_first"),
            input_binding=binding,
            coefficient=3,
        )
        previous.context_match(tracker, cfg)
        tracker.begin()
        samples, times = {}, []

        def prepare_frame(frame):
            samples.clear()
            samples.update(
                previous.read_arrays(
                    previous.SAMPLES, sample_members, f"samples/native_{frame.frame_idx}.npz"
                )
            )
            previous.validate_samples(samples, frame.frame_idx)
            body_points.prepare(frame)

        def on_frame(frame, elapsed):
            nonlocal last_saved
            capture = tracker._fit_history[-1]
            if frame.frame_idx > FIRST:
                saved_packet = evaluation.load_tree(
                    PACKETS, packet_members, f"packets/ct_first/native_{frame.frame_idx}"
                )
                previous.same_tree(capture["packet"], saved_packet, "prepared/actual fit packet")
            previous.save_tree(
                output,
                f"{ARM}/frames/native_{frame.frame_idx}",
                dict(
                    records=[
                        {k: np.asarray(r[k]) for k in previous.RECORD_FIELDS}
                        for r in tracker._history[-1]
                    ],
                    fit=capture,
                    anchor_ids=tracker._anchor_ids,
                    release_native=previous.horizon(frame.frame_idx),
                    advance_seconds=elapsed,
                ),
            )
            last_saved = int(frame.frame_idx)
            previous.fit_sample_match(capture, samples)
            if frame.frame_idx == FIRST:
                bank = dict(anchors=tracker._anchors, facts=tracker._anchor_facts)
                previous.save_tree(output, f"{ARM}/initial_anchors", bank)
                previous.initial_match(
                    tracker._history[0], tracker._anchors, initial_records, initial_anchors
                )
                previous.same_tree(bank, initial_bank, "new body-point causal initial bank")
                output.write(
                    "initial_control.json",
                    dict(
                        passed=True,
                        native=FIRST,
                        initial_anchors=len(initial_anchors),
                        all19_retained_records_exact=True,
                        all_initial_anchors_facts_exact=True,
                        prior_trajectory_used=False,
                        initialized_once=True,
                    ),
                )
            times.append(elapsed)
            if (frame.frame_idx - FIRST + 1) % 20 == 0:
                print(
                    json.dumps(
                        dict(arm=ARM, completed=frame.frame_idx - FIRST + 1, native=frame.frame_idx)
                    ),
                    flush=True,
                )

        previous.run_frames(
            tracker,
            common["reader"],
            prepare_frame=prepare_frame,
            before_frame=lambda n: memory.append(previous.gpu_memory(f"{ARM}_{n}", minimum_gib=2)),
            after_frame=lambda n: None,
            on_frame=on_frame,
            synchronize=lambda: torch.cuda.synchronize(0),
        )
        core, meta, side_arrays, side_meta = tracker.export_transport_state()
        for name, value in (
            ("arrays.npz", core),
            ("metadata.json", meta),
            ("side_arrays.npz", side_arrays),
            ("side_metadata.json", side_meta),
        ):
            output.write(f"{ARM}/checkpoint/{name}", value)
        candidate = tracker.finish()
        output.write(f"{ARM}/candidate.npz", candidate)
        arm_summary = previous.arm_summary(ARM, candidate, core, tracker, times)
        arm_summary.update(
            gap_relative_coefficient=1.0,
            auxiliary_domain="admitted initial-group point rays",
            computed_in_this_process=True,
        )
        output.write(f"{ARM}/summary.json", arm_summary)
        output.write("memory_samples.json", memory)
        summary = dict(
            schema=STAGE,
            status="ok",
            native_interval=[FIRST, STOP],
            arm=arm_summary,
            initialize_calls=1,
            restore_calls=0,
            advance_calls=COUNT,
            prior_tracking_sha256=PREVIOUS_SHA,
            prior_evaluation_sha256=EVALUATION_SHA,
            prior_six_evaluation_sha256=SIX_EVALUATION_SHA,
            body_point_preparation_sha256=PACKETS_SHA,
            cuda_qualification_sha256=QUALIFICATION_SHA,
            all199_fit_packets_exact=True,
            original_control_passed=True,
            causal_initial_control_passed=True,
            maximum_pool=24,
            steps=14,
            model_forwards=0,
            new_detector_or_point_model=False,
            query_refreshes=0,
            no_external_pose_reset=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            environment=environment,
            elapsed_before_guards_seconds=time.perf_counter() - start,
            independent_ID_truth=False,
            independent_evaluation_complete=False,
        )
        output.write("summary.json", summary)
        prepared.snapshot_sources(evidence, output, captured)
        evidence.final_guard()
        output.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            params=dict(all_guards_passed=True),
        )
    except BaseException as error:
        output.write(
            "failure.json",
            dict(
                last_saved_native=last_saved,
                error_type=type(error).__name__,
                error=str(error),
                partial_results_are_not_complete=True,
            ),
        )
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            error_type=type(error).__name__,
            error=str(error),
        )
        with (destination / "tracking.provenance.json").open("xb") as stream:
            stream.write(base.json_bytes(seal))
        raise
    finally:
        cv2.setNumThreads(old_cv)
    with (destination / "tracking.provenance.json").open("xb") as stream:
        stream.write(base.json_bytes(seal))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--device", choices=["cuda"], required=True)
    parser.add_argument("--samples-seal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args())), flush=True)


if __name__ == "__main__":
    main()

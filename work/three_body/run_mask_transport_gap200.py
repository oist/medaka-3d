"""One fresh CT mask-gap200 arm, with sealed prior controls kept read-only.

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
STAGE = "mask_transport_gap200_tracking_v1"
ARM = "ct_gap"
PREVIOUS = WORK / "mask_transport200_three_arm_tracking_v2"
PREVIOUS_SHA = "49c8771c175252ccc0cb6fdca565cc7834e4d1c8ff6496846a2ea94126d0feda"
EVALUATION = WORK / "mask_transport200_independent_evaluation_v1"
EVALUATION_SHA = "dbb38b01317d4171aa7109803e279dfab4c6913086b2ac3321f79f4bb1d4abf6"
PLAN = WORK / "MASK_TRANSPORT_RAY_GAP_PLAN.md"
PLAN_SHA = "d37227f70bf793995f90fc6d88f74be86e0c0c1e7370f1b7e64b397f28817076"
EXECUTION_READY = True
SOURCE_PINS = {
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
        EXECUTION_READY and all(base.valid_sha(x) for x in SOURCE_PINS.values()),
        "gap source/qualification is still in draft",
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
        gap = importlib.import_module("mask_transport_gap_tracker")
        prior_plan, initial_records, initial_anchors, initial_bank = load_controls(evidence)
        common, sample_members, points, reference = previous.load_inputs(
            evidence, previous.SAMPLES_SHA
        )
        cfg = common["original_context"]
        require(cfg == prior_plan["original_context"], "prior/new scientific context differs")
        binding = dict(
            schema=STAGE,
            arm=ARM,
            point_evidence_arm="ct_first",
            preparation_sha256=inputs.PREPARATION_SHA,
            point_sha256=previous.POINTS_SHA,
            samples_sha256=previous.SAMPLES_SHA,
            prior_tracking_sha256=PREVIOUS_SHA,
            prior_evaluation_sha256=EVALUATION_SHA,
            gap_plan_sha256=PLAN_SHA,
            source_pins=SOURCE_PINS,
        )
        prepared.snapshot_sources(evidence, output, captured)
        environment = previous.runtime(reference["binding"])
        memory = [previous.gpu_memory("gap_startup", minimum_gib=6)]
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
                objective="base + 3*(owner + ray_gap)",
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
        tracker = gap.MaskTransportGapTracker(
            common["rig"],
            common["initial"].copy(),
            config=previous.TrackerConfig(**cfg["tracker_config"]),
            recovery_config=previous.NodeRecoveryConfig(**cfg["recovery_config"]),
            persistence_config=previous.PersistentNodeConfig(**cfg["persistence_config"]),
            first_native=FIRST,
            group_counts=points.group_counts,
            query_sha256=points.query_sha256,
            vote_provider=points.provider("ct_first"),
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
            points.prepare(frame)

        def on_frame(frame, elapsed):
            nonlocal last_saved
            capture = tracker._fit_history[-1]
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
                previous.same_tree(bank, initial_bank, "new gap causal initial bank")
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
        arm_summary.update(gap_relative_coefficient=1.0, computed_in_this_process=True)
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

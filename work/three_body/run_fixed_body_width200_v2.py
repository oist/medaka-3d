"""Two fresh fixed-radial200 arms against four sealed read-only controls.

Static-base and static-gap each use causal t0 once, then fixed registered width
in every image term, with 14 original updates. No reset, model or query refresh.
CUDA qualification and all source/input signatures must be bound before run.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import run_mask_transport200_v2 as previous
import run_mask_transport_gap200 as prior_gap
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
STAGE = "fixed_body_width200_tracking_v2"
ARMS = ("static_base", "static_gap")
COEFFICIENTS = (0.0, 3.0)
PLAN = WORK / "FIXED_RADIAL_SHAPE_FOLLOWUP.md"
PLAN_SHA = "399d810288ccf2186d1b453682a623066005e458910a8969258beb6c7c8bd8ff"
QUALIFICATION = WORK / "fixed_body_width_cuda_qualification_v1"
QUALIFICATION_SHA = "475b55a860d268dff8411c9c4e847108f839edf594f27c9964a71f40a62e7f1e"
EXECUTION_READY = True
FAILED = WORK / "fixed_body_width200_tracking_v1"
FAILED_SHA = "15433ff8dfce50422c6bb758d682f0005045261abd18a86c1116b343485ce2b1"
SOURCE_PINS = {
    "work/three_body/fixed_body_width_tracker_v2.py": (
        "7876e6f0ffb62f802aeb077b0f4e52f77e53ad4003e9bb2f22e43c49987999da"
    ),
    "work/three_body/run_mask_transport_gap200.py": (
        "af36a32adc4be70acda870c6037d7fb6a3d7f8a53796ba9fb0c1e4e7bc1991b4"
    ),
    "work/three_body/fixed_body_width_renderer.py": (
        "aca0c16647d28aca957eed2ae24092845219b65c68c118a5a246028ef0cf6563"
    ),
    "work/three_body/fixed_body_width_graph_fit.py": (
        "d5e7c1273cbffd32beab2c579b56f11ee3d47835b10c6c8edfefa6eb0def29f8"
    ),
    "work/three_body/fixed_body_width_fitter.py": (
        "2069c2b9b22f4573f9359238a91fa70ae2ce2fff1158446d9ebbab982ae8574a"
    ),
    "work/three_body/fixed_body_width_tracker.py": (
        "16294aa7264f3aa8634304e53c658623b621402a6ccb957b65d7c750d2a994fd"
    ),
    "work/three_body/qualify_fixed_body_width_cuda.py": (
        "141df5b5123aa5dda60f72307541a09185117d76da205a81396d4d17289f19a4"
    ),
}


def load_controls(evidence):
    return prior_gap.load_controls(evidence)


def qualify_cuda(evidence):
    seal, members = base.verified(
        QUALIFICATION,
        "qualification.provenance.json",
        "fixed_body_width_cuda_qualification_v1",
        QUALIFICATION_SHA,
        evidence,
    )
    summary = base.bound_json(QUALIFICATION, members, "summary.json")
    require(
        summary["passed"] is True
        and summary["status"] == "ok"
        and summary["dtype"] == "float64"
        and summary["batch"] == 2
        and summary["views"] == 3
        and summary["pixels_per_view"] == 64
        and summary["graph_constructions"] == 1
        and summary["graph_configurations"] == 2
        and summary["graph_replay_comparisons"] == 3
        and summary["actual_saved_data"] is False
        and summary["optimizers"] == 0
        and len(summary["cases"]) == 3
        and all(
            row["passed"]
            and set(row["comparisons"]) == {"scores", "parts", "gradient"}
            and all(c["passed"] for c in row["comparisons"].values())
            for row in summary["cases"]
        ),
        "bounded fixed-width CUDA numerical qualification differs",
    )
    for relative in (
        "work/three_body/fixed_body_width_renderer.py",
        "work/three_body/fixed_body_width_graph_fit.py",
        "work/three_body/qualify_fixed_body_width_cuda.py",
    ):
        require(
            seal["inputs"].get(str(REPO / relative)) == SOURCE_PINS[relative],
            "CUDA-qualified source differs: " + relative,
        )
    return summary


def validate_execution(args):
    require(args.execute is True and args.device == "cuda", "explicit CUDA execution required")
    require(
        EXECUTION_READY
        and base.valid_sha(QUALIFICATION_SHA)
        and all(base.valid_sha(x) for x in SOURCE_PINS.values()),
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
        evidence.add(FAILED / "tracking.provenance.json", FAILED_SHA)
        failed = base.read_json(FAILED / "tracking.provenance.json")
        failure_row = failed["outputs"]["failure.json"]
        evidence.add(FAILED / "failure.json", failure_row["sha256"])
        require(
            failed["status"] == "failed"
            and failed["stage"] == "fixed_body_width200_tracking_v1"
            and not any("/frames/" in n or "/checkpoint/" in n for n in failed["outputs"])
            and (FAILED / "failure.json").stat().st_size == failure_row["size_bytes"]
            and base.read_json(FAILED / "failure.json")["last_saved_native"] is None,
            "technical retry requires a pre-begin failure without fitted frames",
        )
        fixed = importlib.import_module("fixed_body_width_tracker_v2")
        qualify_cuda(evidence)
        prior_plan, initial_records, initial_anchors, initial_bank = load_controls(evidence)
        common, sample_members, points, reference = previous.load_inputs(
            evidence, previous.SAMPLES_SHA
        )
        cfg = common["original_context"]
        require(cfg == prior_plan["original_context"], "prior/new scientific context differs")
        binding = dict(
            schema=STAGE,
            arms=list(ARMS),
            point_evidence_arm="ct_first",
            preparation_sha256=inputs.PREPARATION_SHA,
            point_sha256=previous.POINTS_SHA,
            samples_sha256=previous.SAMPLES_SHA,
            prior_tracking_sha256=prior_gap.PREVIOUS_SHA,
            prior_evaluation_sha256=prior_gap.EVALUATION_SHA,
            shape_plan_sha256=PLAN_SHA,
            cuda_qualification_sha256=QUALIFICATION_SHA,
            source_pins=SOURCE_PINS,
            technical_retry_failed_sha256=FAILED_SHA,
            runtime_device_alias="cuda resolves to registered cuda:0; config remains unchanged",
        )
        prepared.snapshot_sources(evidence, output, captured)
        environment = previous.runtime(reference["binding"])
        memory = [previous.gpu_memory("fixed_width_startup", minimum_gib=6)]
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                native_interval=[FIRST, STOP],
                arms=list(ARMS),
                input_binding=binding,
                original_context=cfg,
                environment=environment,
                original_control_passed=True,
                prior_controls_read_only=True,
                prior_numeric_input="causal t0 records and anchors only; no prior trajectory/state",
                objective=(
                    "fixed-width base + coefficient*(fixed-width owner + fixed-width ray_gap)"
                ),
                gap_relative_coefficient=1.0,
                gap="sqrt(max(nearest_gaussian_mahalanobis_sq,1))-1",
                mask_coefficients=dict(zip(ARMS, COEFFICIENTS, strict=True)),
                fixed_width_applied_from_native=FIRST + 1,
                t0_original_all19_required=True,
                width_reference="registered seed only; current centers/tangents/axial unchanged",
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
        summaries = {}

        def run_arm(arm, coefficient):
            tracker = fixed.FixedBodyWidthTracker(
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
                coefficient=coefficient,
            )
            previous.context_match(tracker, cfg)
            output.write(f"{arm}/width_policy.json", tracker._context()["mask_transport"])
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
                    f"{arm}/frames/native_{frame.frame_idx}",
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
                    previous.save_tree(output, f"{arm}/initial_anchors", bank)
                    previous.initial_match(
                        tracker._history[0], tracker._anchors, initial_records, initial_anchors
                    )
                    previous.same_tree(bank, initial_bank, "new gap causal initial bank")
                    output.write(
                        f"{arm}/initial_control.json",
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
                            dict(
                                arm=arm,
                                completed=frame.frame_idx - FIRST + 1,
                                native=frame.frame_idx,
                            )
                        ),
                        flush=True,
                    )

            previous.run_frames(
                tracker,
                common["reader"],
                prepare_frame=prepare_frame,
                before_frame=lambda n: memory.append(
                    previous.gpu_memory(f"{arm}_{n}", minimum_gib=2)
                ),
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
                output.write(f"{arm}/checkpoint/{name}", value)
            candidate = tracker.finish()
            output.write(f"{arm}/candidate.npz", candidate)
            arm_summary = previous.arm_summary(arm, candidate, core, tracker, times)
            arm_summary.update(
                coefficient=coefficient,
                gap_relative_coefficient=1.0,
                computed_in_this_process=True,
                fixed_width_fits=COUNT - 1,
            )
            summaries[arm] = arm_summary
            output.write(f"{arm}/summary.json", arm_summary)

        for arm, coefficient in zip(ARMS, COEFFICIENTS, strict=True):
            run_arm(arm, coefficient)
            gc.collect()
            torch.cuda.empty_cache()
        output.write("memory_samples.json", memory)
        summary = dict(
            schema=STAGE,
            status="ok",
            native_interval=[FIRST, STOP],
            arms=summaries,
            initialize_calls=2,
            restore_calls=0,
            advance_calls=COUNT * 2,
            prior_tracking_sha256=prior_gap.PREVIOUS_SHA,
            prior_evaluation_sha256=prior_gap.EVALUATION_SHA,
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

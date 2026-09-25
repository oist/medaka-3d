"""Four-column raw/projection review after completed body-point200 evaluation.

Raw / Original / CT-gap / CT-body-points share the full native1200 image and
the frozen projection/display functions. Anonymous slots, current-view support,
global two-view support and sticky review flags retain their original meanings.
The eight fixed times are posthoc diagnostics, not identity truth or a holdout.
Optional videos show all200 frames at20fps. No fit, model or GPU runs here.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import cv2
import numpy as np
import render_mask_transport200_v2 as drawing
import render_mask_transport_gap200 as prior_visual
import run_body_point200 as producer

base, prepared, inputs = producer.base, producer.prepared, producer.inputs
previous = producer.previous
WORK, REPO, FIRST, STOP = producer.WORK, producer.REPO, producer.FIRST, producer.STOP
VIEWS, FRAMES = drawing.VIEWS, drawing.FRAMES
ARMS = ("original", "ct_gap", "ct_body_points")
QUANTITATIVE_ARMS = (
    "original",
    "lk",
    "ct_first",
    "ct_gap",
    "static_base",
    "static_gap",
    "ct_body_points",
)
TITLES = ("Original", "CT-gap transport", "CT-body-points")
STAGE = "body_point200_visual_review_v1"
EVALUATION_STAGE = "body_point200_independent_evaluation_v1"
OLD_GAP = WORK / "mask_transport_gap200_tracking_v1"
OLD_GAP_SHA = "a3222e383ce5e08b63a2f4bba8a2404da95cf27f1e6513be44f2f90e4aaf5aa9"
PRODUCER_SHA = "9c8d9a24ebfb98e2346d4b028d1963a2f41a17afb7f41cd35cbab0b412e483e6"
EVALUATOR_SHA = "faacc694e1d4cfb0c4a33304a8da3883bc16ce32768f52e2f67e2aa71b83c02e"
EXECUTION_READY = True
PANEL_W, PANEL_H, IMAGE_Y = drawing.PANEL_W, drawing.PANEL_H, drawing.IMAGE_Y
projection_record, panel, save_png = drawing.projection_record, drawing.panel, drawing.save_png
verify_video = drawing.verify_video  # Same four-column dimensions and all200 extent.
require = base.require
SOURCE_PINS = {
    **prior_visual.SOURCE_PINS,
    **producer.SOURCE_PINS,
    "work/three_body/run_body_point200.py": PRODUCER_SHA,
    "work/three_body/evaluate_body_point200.py": EVALUATOR_SHA,
    "work/three_body/render_mask_transport_gap200.py": (
        "cc4d20f859492a8417aab9a11bea802e7e82502ca92e3d575ff182fdf541be27"
    ),
}


def input_member(seal, path, digest, label):
    require(seal["inputs"].get(str(path)) == digest, label + " input membership differs")


def ready(tracking_digest, evaluation_digest):
    require(
        EXECUTION_READY
        and base.valid_sha(EVALUATOR_SHA)
        and base.valid_sha(tracking_digest)
        and base.valid_sha(evaluation_digest),
        "reviewed final evaluator pin and both success SHA arguments required",
    )


def validate_evaluation(plan, summary, seal, root, digest):
    """Both successful stages precede any candidate or native RGB decoding."""
    require(
        plan["schema"] == summary["schema"] == EVALUATION_STAGE
        and summary["status"] == "ok"
        and plan["native_interval"] == summary["native_interval"] == [FIRST, STOP]
        and plan["arms"] == summary["arms"] == list(QUANTITATIVE_ARMS)
        and plan["source_tracking_sha256"] == digest,
        "complete same-native seven-arm evaluation required",
    )
    require(
        plan["evidence"] == summary["evidence"] == ["lk", "ct_first"]
        and plan["renderer"] == "original_dynamic_all7"
        and plan["samples_sha256"] == previous.SAMPLES_SHA
        and plan["rendering_device"] == "cpu"
        and plan["rendering_dtype"] == "float64"
        and plan["packet_preparation_sha256"] == producer.PACKETS_SHA
        and plan["previous_six_evaluation_sha256"] == producer.SIX_EVALUATION_SHA
        and plan["point_mass_domain"] == "saved FIT1024 m/S; IN-SAMPLE"
        and plan["point_t0_saved_not_fitted"] is True
        and summary["previous_six_common19_exact_frames"] == 200
        and summary["strict_candidate_support_rechecks"] == 1400
        and summary["tracking_CT_packet_exact_frames"] == 199
        and summary["point_evaluation_packets"] == 400
        and summary["point_metrics_are_IN_SAMPLE"] is True
        and summary["t0_packet_used_in_fit"] is False
        and summary["unused_pixel_draws"] == 153600
        and summary["GPU_initialized"] is False
        and summary["trajectory_or_review_modified"] is False
        and summary["independent_ID_truth"] is False,
        "seven-arm independent image/point/state scope differs",
    )
    audit = summary["new_state_audit"]
    require(
        audit["schema"] == "body_point_read_only_state_audit_v1"
        and audit["passed"] is True
        and audit["native_interval"] == [FIRST, STOP]
        and audit["core_arrays"] == 39
        and audit["retained_fields"] == 19
        and audit["history_frames"] == 200
        and audit["semantic_core_and_sidecar_verified"] is True
        and audit["initial_geometry_links_verified"] is True
        and audit["metadata_rewritten"] is False
        and audit["inputs_unchanged"] is True
        and audit["GPU_used"] is False,
        "new independent causal state audit differs",
    )
    for path, expected in (
        (root / "tracking.provenance.json", digest),
        (producer.PREVIOUS / "tracking.provenance.json", producer.PREVIOUS_SHA),
        (OLD_GAP / "tracking.provenance.json", OLD_GAP_SHA),
        (producer.PACKETS / "preparation.provenance.json", producer.PACKETS_SHA),
        (producer.SIX_EVALUATION / "evaluation.provenance.json", producer.SIX_EVALUATION_SHA),
        (WORK / "evaluate_body_point200.py", EVALUATOR_SHA),
        (WORK / "run_body_point200.py", PRODUCER_SHA),
    ):
        input_member(seal, path, expected, "seven-arm evaluation")


def candidate_references(root, members, plan, evaluation_seal, evidence):
    """Bind the displayed three to the exact candidates consumed by all-seven evaluation."""
    rows = plan["candidate_references"]
    require(
        len(rows) == 7 and [r["arm"] for r in rows] == list(QUANTITATIVE_ARMS),
        "seven ordered candidate references required",
    )
    result = []
    for row in rows:
        require(set(row) == {"arm", "path", "sha256"}, "candidate reference fields differ")
        require(base.valid_sha(row["sha256"]), "candidate reference SHA invalid")
        path = Path(row["path"])
        require(path.is_absolute() and path.is_relative_to(WORK), "workspace candidate required")
        input_member(evaluation_seal, path, row["sha256"], "evaluated candidate " + row["arm"])
        if row["arm"] in ARMS:
            source_root = (
                producer.PREVIOUS
                if row["arm"] == "original"
                else OLD_GAP
                if row["arm"] == "ct_gap"
                else root
            )
            require(path == source_root / row["arm"] / "candidate.npz", "candidate role differs")
            if row["arm"] == producer.ARM:
                bound = prepared.bound_path(root, members, f"{producer.ARM}/candidate.npz")
                require(
                    path == bound and members[str(bound)]["sha256"] == row["sha256"],
                    "new evaluated candidate differs from successful tracking",
                )
            evidence.add(path, row["sha256"])
        result.append(dict(**row, displayed=row["arm"] in ARMS))
    return result


def load_inputs(root, digest, evaluation_root, evaluation_digest, evidence):
    ready(digest, evaluation_digest)
    evidence.add(WORK / "evaluate_body_point200.py", EVALUATOR_SHA)
    evaluator = importlib.import_module("evaluate_body_point200")
    seal, members = base.verified(
        root, "tracking.provenance.json", producer.STAGE, digest, evidence
    )
    plan, summary, initial = (
        base.bound_json(root, members, name)
        for name in ("plan.json", "summary.json", "initial_control.json")
    )
    evaluator.validate_run(plan, summary, initial)
    for relative, expected in {
        **producer.SOURCE_PINS,
        "work/three_body/run_body_point200.py": PRODUCER_SHA,
    }.items():
        input_member(seal, REPO / relative, expected, "body-point producer source")
        evidence.add(REPO / relative, expected)
    for path, expected in (
        (producer.PREVIOUS / "tracking.provenance.json", producer.PREVIOUS_SHA),
        (producer.PLAN, producer.PLAN_SHA),
        (producer.PACKETS / "preparation.provenance.json", producer.PACKETS_SHA),
        (producer.QUALIFICATION / "qualification.provenance.json", producer.QUALIFICATION_SHA),
    ):
        input_member(seal, path, expected, "body-point producer")
        evidence.add(path, expected)
    evaluation, eval_members = base.verified(
        evaluation_root, "evaluation.provenance.json", EVALUATION_STAGE, evaluation_digest, evidence
    )
    eval_plan, eval_summary = (
        base.bound_json(evaluation_root, eval_members, name)
        for name in ("plan.json", "summary.json")
    )
    validate_evaluation(eval_plan, eval_summary, evaluation, root, digest)
    for relative, expected in evaluator.SOURCE_PINS.items():
        input_member(evaluation, REPO / relative, expected, "independent evaluator source")
        evidence.add(REPO / relative, expected)
    references = candidate_references(root, members, eval_plan, evaluation, evidence)

    # Reuse only the prior visual input path, never its run or failed-visual flow.
    # This verifies old candidate21 and Original reuse controls. SL3 is consumed
    # through prepared copied detections; no direct tracking->SL3 edge is invented.
    common, old_candidates = prior_visual.load_inputs(OLD_GAP, OLD_GAP_SHA, evidence)
    require(common["original_context"] == plan["original_context"], "scientific context differs")
    for stage in (seal, evaluation):
        require(
            drawing.bind_geometry_lineage(stage, common["context"], evidence)
            == common["visual_geometry_lineage"],
            "all displayed arms/evaluation must share prepared SL3 and calibrated rig",
        )
    candidates = {a: old_candidates[a] for a in ARMS[:2]}
    candidates[producer.ARM] = previous.read_arrays(root, members, f"{producer.ARM}/candidate.npz")
    for candidate in candidates.values():
        drawing.validate_candidate(candidate)
    require(tuple(candidates) == ARMS, "three ordered display candidate21 records required")
    return common, candidates, references


def composite(raw, view, native, projections):
    require(len(projections) == 3, "three arm projections required")
    vi = VIEWS.index(view)
    return np.hstack(
        [panel(raw, f"{view} | native {native}")]
        + [
            panel(raw, title + " | terminal200 backtrace", p, vi)
            for title, p in zip(TITLES, projections, strict=True)
        ]
    )


def run(root, tracking_sha256, evaluation_root, evaluation_sha256, destination, *, videos=False):
    ready(tracking_sha256, evaluation_sha256)
    root, evaluation_root, destination = map(
        lambda p: Path(p).resolve(), (root, evaluation_root, destination)
    )
    require(
        all(p.is_relative_to(WORK) for p in (root, evaluation_root, destination))
        and not destination.exists()
        and base.valid_sha(tracking_sha256)
        and base.valid_sha(evaluation_sha256),
        "new workspace review",
    )
    require(type(videos) is bool, "explicit video flag required")
    output, evidence, captured = base.Output(destination), base.Evidence(), {}
    started, completed, writers, video_rows = base.utc(), [], {}, []
    try:
        evidence.add(Path(__file__).resolve())
        for relative, expected in SOURCE_PINS.items():
            evidence.add(REPO / relative, expected)
        common, candidates, references = load_inputs(
            root, tracking_sha256, evaluation_root, evaluation_sha256, evidence
        )
        prepared.snapshot_sources(evidence, output, captured)
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                tracking_sha256=tracking_sha256,
                evaluation_sha256=evaluation_sha256,
                old_gap_tracking_sha256=OLD_GAP_SHA,
                candidate_references=references,
                quantitative_arms=list(QUANTITATIVE_ARMS),
                prior_three_arm_tracking_sha256=producer.PREVIOUS_SHA,
                body_point_plan_sha256=producer.PLAN_SHA,
                packet_preparation_sha256=producer.PACKETS_SHA,
                geometry_lineage=common["visual_geometry_lineage"],
                frames=list(FRAMES),
                views=list(VIEWS),
                arms=list(ARMS),
                native_interval=[FIRST, STOP],
                terminal_horizon_exclusive=STOP,
                raw_native_shape=[1200, 1200],
                displayed_image_shape=[600, 600],
                columns=["Raw", *TITLES],
                videos=videos,
                video_fps=20 if videos else None,
                full_field_crop=[0, 0, 1200, 1200],
                projection_array_axes=dict(
                    projected_xy="arm,view,anonymous_body_slot,node,xy",
                    sources="arm,anonymous_body_slot,view,node",
                    observed="arm,anonymous_body_slot,node",
                    marker_kind="arm,view,anonymous_body_slot,node",
                    frame_idx="arm; all three values equal the filename native",
                ),
                marker_codes={
                    "open_no_current_view_source": 0,
                    "dot_current_view_only": 1,
                    "filled_current_view_and_global_observed": 2,
                },
                display_edge_policy=(
                    "native-inside XY rounded at 50% and clipped to last display pixel"
                ),
                dashed_segments="geometric pose links, not extra support",
                selection="fixed posthoc diagnostic, not holdout",
                original_reused_from_failed_stage_sha256=drawing.FAILED_SHA,
                sticky_review_flags_visible=True,
                biological_ID_truth=False,
            ),
        )
        try:
            if videos:
                for view in VIEWS:
                    path = destination / (view + ".mp4")
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (4 * PANEL_W, PANEL_H)
                    )
                    writers[view] = writer
                    require(writer.isOpened(), "review video writer unavailable")
            for native in range(FIRST, STOP) if videos else FRAMES:
                frame = common["reader"].at(native)
                require(frame.frame_idx == native, "review native differs")
                projections = [
                    projection_record(candidates[a], native - FIRST, common["rig"]) for a in ARMS
                ]
                stacked = {k: np.stack([p[k] for p in projections]) for k in projections[0]}
                output.write(f"projections/native_{native}.npz", stacked)
                contacts = []
                for view in VIEWS:
                    canvas = composite(frame.images_bgr[view], view, native, projections)
                    if videos:
                        writers[view].write(canvas)
                    if native in FRAMES:
                        save_png(output, f"frames/{view}_{native}.png", canvas)
                        contacts.append(canvas)
                if native in FRAMES:
                    save_png(output, f"contacts/native_{native}.png", np.vstack(contacts))
                completed.append(native)
        finally:
            for view, writer in writers.items():
                writer.release()
                path = destination / (view + ".mp4")
                if path.exists():
                    output.records[path.name] = dict(
                        path=str(path), sha256=base.sha(path), size_bytes=path.stat().st_size
                    )
        if videos:
            video_rows = [dict(view=v, **verify_video(destination / (v + ".mp4"))) for v in VIEWS]
        require(
            completed == (list(range(FIRST, STOP)) if videos else list(FRAMES)),
            "fixed review frames incomplete",
        )
        result = dict(
            schema=STAGE,
            status="ok",
            frames=list(FRAMES),
            PNG_views=24,
            PNG_contacts=8,
            videos=video_rows,
            projection_native_frames=len(completed),
            projected_node_cells=len(completed) * 3 * 3 * 9,
            same_raw_per_arm=True,
            arms=list(ARMS),
            new_fit_or_model_execution=False,
            candidate_support_is_saved_not_requalified=True,
            independent_evaluation_sha256=evaluation_sha256,
            quantitative_arms=list(QUANTITATIVE_ARMS),
            support_markers_are_saved_SLEAP_not_GFTT=True,
            decoded_raw_PNGs=len(completed) * 3,
            scientific_renderer=False,
            candidate_or_review_modified=False,
            manual_review=False,
            verified_biological_IDs=False,
            original_rate_fps=166,
            native_interval=[FIRST, STOP],
        )
        output.write("summary.json", result)
        prepared.snapshot_sources(evidence, output, captured)
        evidence.final_guard()
        output.final_guard()
        status = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
        )
    except BaseException as error:
        status = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            completed_natives=completed,
            error_type=type(error).__name__,
            error=str(error),
        )
        with (destination / "review.provenance.json").open("xb") as stream:
            stream.write(base.json_bytes(status))
        raise
    with (destination / "review.provenance.json").open("xb") as stream:
        stream.write(base.json_bytes(status))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--tracking-sha256", required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--evaluation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.tracking_root,
                args.tracking_sha256,
                args.evaluation_root,
                args.evaluation_sha256,
                args.output,
                videos=args.videos,
            )
        )
    )


if __name__ == "__main__":
    main()

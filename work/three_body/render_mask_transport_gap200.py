"""Five-column raw/projection review for the fixed mask-gap200 comparison.

Frozen camera projection and drawing helpers are reused without modification.
The fixed eight times were chosen after earlier results: this is diagnostic
visualization, not a holdout or proof of fish identity. Optional videos include
all200 native frames at20fps. No model, scientific render, fit, or GPU runs here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import render_mask_transport200_v2 as drawing
import run_mask_transport_gap200 as producer

base, prepared, inputs = producer.base, producer.prepared, producer.inputs
previous = producer.previous
WORK, REPO, FIRST, STOP = producer.WORK, producer.REPO, producer.FIRST, producer.STOP
VIEWS, FRAMES = drawing.VIEWS, drawing.FRAMES
ARMS = (*previous.ARMS, producer.ARM)
TITLES = ("Original", "LK transport", "CT-owner transport", "CT-gap transport")
STAGE = "mask_transport_gap200_visual_review_v1"
PANEL_W, PANEL_H, IMAGE_Y = drawing.PANEL_W, drawing.PANEL_H, drawing.IMAGE_Y
projection_record, panel, save_png = drawing.projection_record, drawing.panel, drawing.save_png
require = base.require
PRODUCER_SHA = "af36a32adc4be70acda870c6037d7fb6a3d7f8a53796ba9fb0c1e4e7bc1991b4"
SOURCE_PINS = {
    **producer.SOURCE_PINS,
    "work/three_body/run_mask_transport_gap200.py": PRODUCER_SHA,
    "work/three_body/render_mask_transport200_v2.py": (
        "5903b976f6f809d77d684eceeefb8064a76f0e3e92d52a73998d5cc1f221822d"
    ),
    "work/three_body/render_semantic_node200.py": (
        "8773ef01eae48fb8e4cdc9110c7e7f62c487ec1f8cde1b66700625a5618ae856"
    ),
    "work/three_body/mask_transport_inputs.py": (
        "eb579dfda7db4f1ab9cc175b72050f332663c3206ab5f77a9cec56cedb58a1fc"
    ),
}


def validate_gap(plan, summary, initial):
    require(
        plan["schema"] == summary["schema"] == producer.STAGE
        and summary["status"] == "ok"
        and plan["native_interval"] == summary["native_interval"] == [FIRST, STOP]
        and plan["arm"] == summary["arm"]["arm"] == producer.ARM,
        "complete fixed gap200 arm required",
    )
    binding = plan["input_binding"]
    require(
        binding["prior_tracking_sha256"]
        == summary["prior_tracking_sha256"]
        == producer.PREVIOUS_SHA
        and binding["prior_evaluation_sha256"]
        == summary["prior_evaluation_sha256"]
        == producer.EVALUATION_SHA
        and binding["preparation_sha256"] == inputs.PREPARATION_SHA
        and binding["point_sha256"] == previous.POINTS_SHA
        and binding["samples_sha256"] == previous.SAMPLES_SHA
        and binding["gap_plan_sha256"] == producer.PLAN_SHA
        and binding["source_pins"] == producer.SOURCE_PINS,
        "fixed gap inputs or source binding differs",
    )
    require(
        plan["objective"] == "base + 3*(owner + ray_gap)"
        and plan["gap_relative_coefficient"] == 1.0
        and plan["mask_coefficient"] == 3.0
        and plan["point_evidence_arm"] == "ct_first"
        and plan["extra_proposals"] == 0
        and plan["maximum_pool"] == summary["maximum_pool"] == 24
        and plan["opt_steps"] == summary["steps"] == 14,
        "fixed gap objective or budget differs",
    )
    require(
        summary["initialize_calls"] == summary["arm"]["initialize_count"] == 1
        and summary["restore_calls"] == summary["arm"]["restore_count"] == 0
        and summary["advance_calls"] == summary["arm"]["frames"] == 200
        and summary["model_forwards"] == summary["query_refreshes"] == 0
        and summary["no_external_pose_reset"] is True
        and summary["original_control_passed"] is True
        and summary["causal_initial_control_passed"] is True
        and initial["passed"] is True
        and initial["native"] == FIRST
        and initial["initialized_once"] is True
        and initial["prior_trajectory_used"] is False
        and initial["all19_retained_records_exact"] is True
        and initial["all_initial_anchors_facts_exact"] is True,
        "complete causal gap initialization and all200 advances required",
    )


def load_inputs(root, digest, evidence):
    seal, members = base.verified(
        root, "tracking.provenance.json", producer.STAGE, digest, evidence
    )
    plan, summary, initial = (
        base.bound_json(root, members, name)
        for name in ("plan.json", "summary.json", "initial_control.json")
    )
    validate_gap(plan, summary, initial)
    for relative, expected in {
        **producer.SOURCE_PINS,
        "work/three_body/run_mask_transport_gap200.py": PRODUCER_SHA,
    }.items():
        path = REPO / relative
        require(seal["inputs"].get(str(path)) == expected, "gap source origin differs: " + relative)
        evidence.add(path, expected)
    for path, expected in (
        (producer.PREVIOUS / "tracking.provenance.json", producer.PREVIOUS_SHA),
        (producer.PLAN, producer.PLAN_SHA),
    ):
        require(seal["inputs"].get(str(path)) == expected, "gap prior/plan origin differs")
        evidence.add(path, expected)
    # This frozen loader validates successful3arm, all21 fields, and the prepared
    # raw/SL/rig lineage. It does not import a new evaluation stage or run a fit.
    common, candidates = drawing.load_inputs(producer.PREVIOUS, producer.PREVIOUS_SHA, evidence)
    require(
        plan["original_context"] == common["original_context"], "same calibrated context required"
    )
    gap_lineage = drawing.bind_geometry_lineage(seal, common["context"], evidence)
    require(
        gap_lineage == common["visual_geometry_lineage"], "four-arm geometry provenance differs"
    )
    candidates[producer.ARM] = previous.read_arrays(root, members, f"{producer.ARM}/candidate.npz")
    drawing.validate_candidate(candidates[producer.ARM])
    require(tuple(candidates) == ARMS, "all four ordered candidate21 records required")
    return common, candidates


def composite(raw, view, native, projections):
    require(len(projections) == 4, "four arm projections required")
    vi = VIEWS.index(view)
    return np.hstack(
        [panel(raw, f"{view} | native {native}")]
        + [
            panel(raw, title + " | terminal200 backtrace", p, vi)
            for title, p in zip(TITLES, projections, strict=True)
        ]
    )


def verify_video(path):
    capture, count = cv2.VideoCapture(str(path)), 0
    try:
        require(
            capture.isOpened() and abs(capture.get(cv2.CAP_PROP_FPS) - 20) < 1e-6,
            "review video FPS/open differs",
        )
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            require(frame.shape == (PANEL_H, 5 * PANEL_W, 3), "review video dimensions differ")
            count += 1
    finally:
        capture.release()
    require(count == 200, "review video must decode all200 frames")
    return dict(frames=count, fps=20, width=5 * PANEL_W, height=PANEL_H)


def run(root, tracking_sha256, destination, *, videos=False):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    require(
        root.is_relative_to(WORK)
        and destination.is_relative_to(WORK)
        and not destination.exists()
        and base.valid_sha(tracking_sha256),
        "new workspace review",
    )
    require(type(videos) is bool, "explicit video flag required")
    output, evidence, captured = base.Output(destination), base.Evidence(), {}
    started, completed, writers, video_rows = base.utc(), [], {}, []
    try:
        evidence.add(Path(__file__).resolve())
        for relative, expected in SOURCE_PINS.items():
            evidence.add(REPO / relative, expected)
        common, candidates = load_inputs(root, tracking_sha256, evidence)
        prepared.snapshot_sources(evidence, output, captured)
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                tracking_sha256=tracking_sha256,
                prior_three_arm_tracking_sha256=producer.PREVIOUS_SHA,
                gap_plan_sha256=producer.PLAN_SHA,
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
                    frame_idx="arm; all four values equal the filename native",
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
                        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (5 * PANEL_W, PANEL_H)
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
            projected_node_cells=len(completed) * 4 * 3 * 9,
            same_raw_per_arm=True,
            arms=list(ARMS),
            new_fit_or_model_execution=False,
            gap_candidate_support_is_saved_not_requalified=True,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    print(
        json.dumps(run(args.tracking_root, args.tracking_sha256, args.output, videos=args.videos))
    )


if __name__ == "__main__":
    main()

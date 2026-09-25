"""Five-column raw/projection review after successful fixed-width six-arm evaluation.

Frozen camera projection and drawing helpers are reused without modification.
The fixed eight times were chosen after earlier results: this is diagnostic
visualization, not a holdout or proof of fish identity. Optional videos include
all200 native frames at20fps. No model, scientific render, fit, or GPU runs here.
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
import run_fixed_body_width200_v2 as producer

base, prepared, inputs = producer.base, producer.prepared, producer.inputs
previous = producer.previous
WORK, REPO, FIRST, STOP = producer.WORK, producer.REPO, producer.FIRST, producer.STOP
VIEWS, FRAMES = drawing.VIEWS, drawing.FRAMES
ARMS = ("original", "ct_gap", "static_base", "static_gap")
QUANTITATIVE_ARMS = ("original", "lk", "ct_first", "ct_gap", "static_base", "static_gap")
TITLES = ("Original", "CT-gap transport", "Fixed width base", "Fixed width CT-gap")
STAGE = "fixed_body_width200_visual_review_v1"
EVALUATION_STAGE = "fixed_body_width200_independent_evaluation_v1"
OLD_GAP = WORK / "mask_transport_gap200_tracking_v1"
OLD_GAP_SHA = "a3222e383ce5e08b63a2f4bba8a2404da95cf27f1e6513be44f2f90e4aaf5aa9"
OLD_EVALUATION = WORK / "mask_transport_gap200_independent_evaluation_v1"
OLD_EVALUATION_SHA = "4a68497916105c4d205101542c79223bddac072bfcf037781a39447c5bc73ef9"
EVALUATOR_SHA = "ab1659274444a077d9b12376717bf1c3563c7433854f1d16196bc46599fa59cf"
TRACKING_SHA = "f0381888f90e9f56ae41d76fb287148a44746b94f1222ca337bef0e25d0b9dda"
EVALUATION_SHA = "a5b95006cf407e70e03a4368eba9576ea2df1bfebb9fe0c64c6b3bfb01b670db"
PANEL_W, PANEL_H, IMAGE_Y = drawing.PANEL_W, drawing.PANEL_H, drawing.IMAGE_Y
projection_record, panel, save_png = drawing.projection_record, drawing.panel, drawing.save_png
require = base.require
PRODUCER_SHA = "d3083a061fc599c1e066f58f3144593dd48d979c915de3e66a390a354d811dcf"
SOURCE_PINS = {
    **prior_visual.SOURCE_PINS,
    **producer.SOURCE_PINS,
    "work/three_body/run_fixed_body_width200_v2.py": PRODUCER_SHA,
    "work/three_body/evaluate_fixed_body_width200.py": EVALUATOR_SHA,
    "work/three_body/render_mask_transport_gap200.py": (
        "cc4d20f859492a8417aab9a11bea802e7e82502ca92e3d575ff182fdf541be27"
    ),
}


def input_member(seal, path, digest, label):
    require(seal["inputs"].get(str(path)) == digest, label + " input membership differs")


def validate_evaluation(plan, summary, seal, root, digest):
    """Require all-six independent completion before candidate/raw deserialization."""
    require(
        plan["schema"] == summary["schema"] == EVALUATION_STAGE
        and summary["status"] == "ok"
        and plan["native_interval"] == summary["native_interval"] == [FIRST, STOP]
        and plan["arms"] == summary["arms"] == list(QUANTITATIVE_ARMS)
        and plan["source_tracking_sha256"] == digest,
        "complete same-native six-arm evaluation required",
    )
    require(
        plan["evidence"] == summary["evidence"] == ["lk", "ct_first"]
        and plan["primary_renderer"] == "original_dynamic_all6"
        and plan["secondary_renderer"] == "registered_fixed_radial_all6"
        and plan["samples_sha256"] == previous.SAMPLES_SHA
        and plan["rendering_device"] == "cpu"
        and plan["rendering_dtype"] == "float64"
        and summary["original_four_CPU_image_control_exact_frames"] == 200
        and summary["strict_candidate_support_rechecks"] == 1200
        and summary["unused_pixel_draws"] == 153600
        and summary["GPU_initialized"] is False
        and summary["trajectory_or_review_modified"] is False,
        "six-arm evaluation scope/old-control evidence differs",
    )
    require(
        plan["previous_tracking_sha256"] == producer.prior_gap.PREVIOUS_SHA
        and plan["previous_evaluation_sha256"] == OLD_EVALUATION_SHA,
        "six-arm previous result lineage differs",
    )
    audits = summary["width_reference_audits"]
    require(set(audits) == set(producer.ARMS), "both fixed-width state audits required")
    for arm in producer.ARMS:
        require(
            audits[arm]["passed"] is True
            and audits[arm]["native_interval"] == [FIRST, STOP]
            and audits[arm]["core_arrays"] == 39
            and audits[arm]["history_frames"] == 200
            and audits[arm]["semantic_core_and_sidecar_verified"] is True
            and audits[arm]["initial_geometry_links_verified"] is True
            and audits[arm]["metadata_rewritten"] is False
            and audits[arm]["GPU_used"] is False,
            "fixed-width independent state audit differs: " + arm,
        )
    for path, expected in (
        (root / "tracking.provenance.json", digest),
        (producer.prior_gap.PREVIOUS / "tracking.provenance.json", producer.prior_gap.PREVIOUS_SHA),
        (OLD_GAP / "tracking.provenance.json", OLD_GAP_SHA),
        (OLD_EVALUATION / "evaluation.provenance.json", OLD_EVALUATION_SHA),
        (WORK / "evaluate_fixed_body_width200.py", EVALUATOR_SHA),
        (Path(producer.__file__).resolve(), PRODUCER_SHA),
    ):
        input_member(seal, path, expected, "six-arm evaluation")


def candidate_references(root, members, old_members, gap_members, evaluation_seal):
    """All six path/SHA edges; only the four display methods are opened later."""
    result = []
    for arm in QUANTITATIVE_ARMS:
        source_root, source_members = (
            (producer.prior_gap.PREVIOUS, old_members)
            if arm in previous.ARMS
            else (OLD_GAP, gap_members)
            if arm == "ct_gap"
            else (root, members)
        )
        path = prepared.bound_path(source_root, source_members, f"{arm}/candidate.npz")
        digest = source_members[str(path)]["sha256"]
        input_member(evaluation_seal, path, digest, "evaluated candidate " + arm)
        result.append(dict(arm=arm, path=str(path), sha256=digest, displayed=arm in ARMS))
    return result


def load_inputs(root, digest, evaluation_root, evaluation_digest, evidence):
    # No eval fitting/rendering entry is called. Reuse only its reviewed metadata
    # predicate, after explicitly binding the exact current source.
    require(digest == TRACKING_SHA, "fixed completed two-arm tracking SHA required")
    require(evaluation_digest == EVALUATION_SHA, "fixed completed six-arm evaluation SHA required")
    evidence.add(WORK / "evaluate_fixed_body_width200.py", EVALUATOR_SHA)
    evaluator = importlib.import_module("evaluate_fixed_body_width200")
    seal, members = base.verified(
        root, "tracking.provenance.json", producer.STAGE, digest, evidence
    )
    plan, summary = (base.bound_json(root, members, n) for n in ("plan.json", "summary.json"))
    initial = {
        arm: base.bound_json(root, members, f"{arm}/initial_control.json") for arm in producer.ARMS
    }
    evaluator.validate_run(plan, summary, initial)
    require(
        plan["input_binding"]["technical_retry_failed_sha256"] == producer.FAILED_SHA,
        "fixed-width technical retry lineage differs",
    )
    for relative, expected in {
        **producer.prior_gap.SOURCE_PINS,
        **producer.SOURCE_PINS,
        "work/three_body/run_fixed_body_width200_v2.py": PRODUCER_SHA,
    }.items():
        input_member(seal, REPO / relative, expected, "fixed-width producer source")
        evidence.add(REPO / relative, expected)
    for path, expected in (
        (producer.prior_gap.PREVIOUS / "tracking.provenance.json", producer.prior_gap.PREVIOUS_SHA),
        (producer.PLAN, producer.PLAN_SHA),
        (producer.FAILED / "tracking.provenance.json", producer.FAILED_SHA),
        (producer.QUALIFICATION / "qualification.provenance.json", producer.QUALIFICATION_SHA),
    ):
        input_member(seal, path, expected, "fixed-width producer")
        evidence.add(path, expected)

    evaluation, eval_members = base.verified(
        evaluation_root, "evaluation.provenance.json", EVALUATION_STAGE, evaluation_digest, evidence
    )
    eval_plan, eval_summary = (
        base.bound_json(evaluation_root, eval_members, n) for n in ("plan.json", "summary.json")
    )
    validate_evaluation(eval_plan, eval_summary, evaluation, root, digest)

    old, old_members = base.verified(
        producer.prior_gap.PREVIOUS,
        "tracking.provenance.json",
        previous.STAGE,
        producer.prior_gap.PREVIOUS_SHA,
        evidence,
    )
    old_plan, old_summary, control, reuse = (
        base.bound_json(producer.prior_gap.PREVIOUS, old_members, n)
        for n in (
            "plan.json",
            "summary.json",
            "original_control.json",
            "original_reuse_qualification.json",
        )
    )
    drawing.validate_completed(old_plan, old_summary, control)
    drawing.validate_reuse(old_plan, old_summary, reuse)
    input_member(
        old,
        Path(previous.__file__).resolve(),
        producer.prior_gap.SOURCE_PINS["work/three_body/run_mask_transport200_v2.py"],
        "original producer",
    )
    gap, gap_members = base.verified(
        OLD_GAP, "tracking.provenance.json", producer.prior_gap.STAGE, OLD_GAP_SHA, evidence
    )
    gap_plan, gap_summary, gap_initial = (
        base.bound_json(OLD_GAP, gap_members, n)
        for n in ("plan.json", "summary.json", "initial_control.json")
    )
    prior_visual.validate_gap(gap_plan, gap_summary, gap_initial)
    for relative, expected in {
        **producer.prior_gap.SOURCE_PINS,
        "work/three_body/run_mask_transport_gap200.py": prior_visual.PRODUCER_SHA,
    }.items():
        input_member(gap, REPO / relative, expected, "CT-gap producer source")
    references = candidate_references(root, members, old_members, gap_members, evaluation)

    common = inputs.load_common(evidence)
    require(
        common["original_context"]
        == plan["original_context"]
        == old_plan["original_context"]
        == gap_plan["original_context"],
        "same calibrated scientific context required",
    )
    lineage = drawing.bind_geometry_lineage(old, common["context"], evidence)
    for stage in (gap, seal, evaluation):
        require(
            drawing.bind_geometry_lineage(stage, common["context"], evidence) == lineage,
            "all displayed arms/evaluation must share prepared SL3 and direct calibration4",
        )
    common["visual_geometry_lineage"] = lineage
    candidates = {}
    for arm in ARMS:
        source_root, source_members = (
            (producer.prior_gap.PREVIOUS, old_members)
            if arm == "original"
            else (OLD_GAP, gap_members)
            if arm == "ct_gap"
            else (root, members)
        )
        candidates[arm] = previous.read_arrays(source_root, source_members, f"{arm}/candidate.npz")
        drawing.validate_candidate(candidates[arm])
    require(tuple(candidates) == ARMS, "four ordered display candidate21 records required")
    return common, candidates, references


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


def run(root, tracking_sha256, evaluation_root, evaluation_sha256, destination, *, videos=False):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    require(
        root.is_relative_to(WORK)
        and evaluation_root.is_relative_to(WORK)
        and destination.is_relative_to(WORK)
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
                evaluation_root=str(evaluation_root),
                quantitative_arms=list(QUANTITATIVE_ARMS),
                candidate_references=references,
                omitted_visual_arms=["lk", "ct_first"],
                omitted_reason="already reviewed; all six retained in independent evaluation",
                prior_three_arm_tracking_sha256=producer.prior_gap.PREVIOUS_SHA,
                prior_CT_gap_tracking_sha256=OLD_GAP_SHA,
                shape_plan_sha256=producer.PLAN_SHA,
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
            candidate_support_is_saved_not_requalified_by_visualizer=True,
            independent_evaluation_sha256=evaluation_sha256,
            independent_evaluation_strict_rechecks=1200,
            quantitative_arms=list(QUANTITATIVE_ARMS),
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

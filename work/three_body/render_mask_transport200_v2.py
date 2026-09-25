"""Fixed raw/Original/LK/CT-first projection review after completed V2 tracking.

The eight native times are fixed posthoc diagnostics, not a holdout. Every panel
uses the complete same 1200x1200 raw image, displayed at 600x600. Colors denote
anonymous slots in the terminal200 backtrace; they do not establish fish IDs.
Optional videos retain all200 frames at20fps and are fully decoded for validation.
No scientific renderer, image synthesis, fitter or point model is executed.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import cv2
import mask_transport_inputs as inputs
import numpy as np
import render_semantic_node200 as drawing

base, prepared = inputs.base, inputs.prepared
WORK, REPO, FIRST, STOP = inputs.WORK, inputs.REPO, inputs.FIRST, inputs.STOP
VIEWS = tuple(inputs.VIEWS)
ARMS = ("original", "lk", "ct_first")
STAGE = "mask_transport200_visual_review_v2"
FAILED_VISUAL_SHA = "14d4adbe4a956a452bb76faec7672841773f54942ac40c472473c6f4f02174fd"
FAILED_VISUAL_SOURCE_SHA = "d19309485411afbb37372c535b8f696d0015484aab54efa8757b889ab6e89d0c"
PRODUCER_STAGE = "mask_transport200_three_arm_tracking_v2"
FAILED_ROOT = WORK / "mask_transport200_three_arm_tracking_v1"
FAILED_SHA = "c22ce3de117a08b3e9e3634791a98dbb4d0c466d37a596674f1dd96519ae563a"
FAILED_SOURCE_SHA = "a605e1594dbae4ee77f8025ca7ff28a358e32a4063d007cb41d2bc42aba49863"
FRAMES = tuple(FIRST + t for t in (0, 16, 18, 40, 80, 120, 160, 199))
COLORS, NODES = drawing.COLORS, drawing.NODES
PANEL_W, PANEL_H, IMAGE_Y = 600, 744, 64
require = base.require


def validate_completed(plan, summary, control):
    require(
        plan["schema"] == summary["schema"] == PRODUCER_STAGE
        and summary["status"] == "ok"
        and plan["native_interval"] == summary["native_interval"] == [FIRST, STOP]
        and plan["arms"] == list(ARMS),
        "successful complete V2 required",
    )
    require(
        summary["original_control_passed"] is True
        and control["passed"] is True
        and all(control[k]["byte_exact"] is True for k in ("candidate", "history", "state")),
        "completed Original exact replay qualification required",
    )
    require(
        [r["arm"] for r in summary["arms"]] == list(ARMS)
        and all(
            r["frames"] == 200 and r["initialize_count"] == 1 and r["restore_count"] == 0
            for r in summary["arms"]
        ),
        "all three terminal200 arms required",
    )
    require(
        plan["input_binding"]["preparation_sha256"] == inputs.PREPARATION_SHA
        and plan["input_binding"]["implementation_plan_sha256"] == inputs.PLAN_SHA,
        "same prepared raw/plan required",
    )


def validate_reuse(plan, summary, proof):
    """Small completed-V2 metadata gate; no old fit/candidate payload is opened."""
    require(
        plan["new_arms_this_process"] == ["lk", "ct_first"]
        and plan["reused_original_source_seal_sha256"] == FAILED_SHA
        and summary["reused_original_source_seal_sha256"] == FAILED_SHA
        and summary["this_process_initialize_calls"] == 2
        and summary["this_process_restore_calls"] == 0
        and summary["new_advance_calls"] == 400
        and summary["reused_original_frames"] == 200,
        "V2 reused versus newly computed scope differs",
    )
    require(
        summary["arms"][0]["computed_in_this_process"] is False
        and summary["arms"][0]["reused_from_seal_sha256"] == FAILED_SHA
        and all(r["computed_in_this_process"] is True for r in summary["arms"][1:]),
        "V2 per-arm origin differs",
    )
    require(
        proof["schema"] == "mask_transport_original_reuse_qualification_v1"
        and proof["status"] == "ok"
        and proof["passed"] is True
        and proof["original_source_seal_sha256"] == FAILED_SHA
        and proof["original_source_sha256"] == FAILED_SOURCE_SHA
        and proof["frames"]
        == proof["actual_fit4_byte_exact_frames"]
        == proof["saved_frame_fit_and_records_validated"]
        == 200
        and proof["original_files_copied_byte_exact"] == 408
        and proof["initial_preparation_SL_masks_byte_exact_views"] == 3,
        "completed saved Original reuse proof required",
    )
    for field in (
        "candidate_21_byte_exact",
        "history_19_plus_offsets_byte_exact",
        "core_39_byte_exact",
        "core_metadata_unchanged",
        "semantic_state_decode_passed",
        "sidecar_validation_passed",
        "original_context_and_sidecar_bytes_preserved",
    ):
        require(proof[field] is True, "incomplete Original reuse proof: " + field)


def validate_candidate(value):
    """All21 fields remain in native order; no biological identity alignment."""
    specs = dict(
        frame_idx=((), "i"),
        pose_xyz=((3, 3, 3), "f"),
        observed=((3, 3), "b"),
        sources=((3, 3, 3), "i"),
        identity_review=((3,), "b"),
        mask_supported=((3,), "b"),
        body_supported=((3,), "b"),
        missing=((3,), "i"),
        costs=((), "f"),
        path_costs=((), "f"),
        loss_parts=((5,), "f"),
        score_margin=((), "f"),
        proposal_kind=((), "U"),
        hypotheses=((), "i"),
        proposal_flow_nodes=((3, 3), "b"),
        proposal_flow_parent=((), "i"),
        node_missing=((3, 3), "i"),
        direct_pectoral_age=((3,), "i"),
        xyz=((3, 3), "f"),
        accepted_xyz=((3, 3), "f"),
        body_xyz=((3, 3), "f"),
    )
    require(set(value) == set(specs), "complete21 candidate field membership differs")
    for name, (shape, kind) in specs.items():
        a = value[name]
        require(
            a.shape == (200, *shape) and a.dtype.kind == kind,
            "candidate field shape/dtype differs: " + name,
        )
        if kind == "f" and name not in ("xyz", "accepted_xyz", "body_xyz"):
            require(
                not np.isnan(a).any() and (name == "score_margin" or np.isfinite(a).all()),
                "nonfinite candidate field: " + name,
            )
    require(
        value["frame_idx"].dtype == np.int64
        and np.array_equal(value["frame_idx"], np.arange(FIRST, STOP, dtype=np.int64)),
        "fixed200 native candidate required",
    )
    require(
        value["pose_xyz"].dtype == np.float64 and (value["sources"] >= -1).all(),
        "native pose/source schema differs",
    )
    expected = dict(
        xyz=np.where(value["observed"][:, :, 1, None], value["pose_xyz"][:, :, 1], np.nan),
        body_xyz=np.where(value["body_supported"][:, :, None], value["pose_xyz"][:, :, 1], np.nan),
    )
    expected["accepted_xyz"] = np.where(
        value["identity_review"][:, :, None], np.nan, expected["xyz"]
    )
    for name, a in expected.items():
        require(
            base.signature(a) == base.signature(value[name]), "derived candidate differs: " + name
        )


def projection_record(candidate, offset, rig):
    """Native projection and display categories preserve all three slots/nodes.

    marker0=open: no current-view source. marker1=dot: a current-view source,
    global two-view observed false. marker2=filled: source and observed true.
    Nonfinite/outside nodes retain their category and coordinates but are not drawn.
    """
    require(type(offset) is int and 0 <= offset < 200, "native offset outside fixed200")
    require(int(candidate["frame_idx"][offset]) == FIRST + offset, "projection native mismatch")
    pose = candidate["pose_xyz"][offset]
    sources, observed = candidate["sources"][offset], candidate["observed"][offset]
    require(
        pose.shape == (3, 3, 3) and pose.dtype == np.float64 and np.isfinite(pose).all(),
        "finite all-slot pose required",
    )
    require(
        sources.shape == (3, 3, 3)
        and sources.dtype.kind == "i"
        and observed.shape == (3, 3)
        and observed.dtype == bool,
        "support axes differ",
    )
    before = [base.signature(a) for a in (pose, sources, observed)]
    projected = np.stack([np.asarray(rig.camera(v).project(pose.copy())) for v in VIEWS])
    require(
        projected.shape == (3, 3, 3, 2) and projected.dtype.kind == "f", "camera projection axes"
    )
    finite = np.isfinite(projected).all(-1)
    inside = finite & (projected >= 0).all(-1) & (projected < 1200).all(-1)
    view_source = sources.transpose(1, 0, 2) >= 0
    marker = np.where(view_source, np.where(observed[None], 2, 1), 0).astype(np.int8)
    pixel = np.full(projected.shape, -1, np.int64)
    # Avoid integer overflow for failed projections; raw XY stays in the output.
    pixel[inside] = np.rint(projected[inside]).astype(np.int64)
    rounded_inside = inside & (pixel >= 0).all(-1) & (pixel < 1200).all(-1)
    require(
        before == [base.signature(a) for a in (pose, sources, observed)],
        "projection input mutation",
    )
    return dict(
        pose_xyz=pose.copy(),
        sources=sources.copy(),
        observed=observed.copy(),
        identity_review=candidate["identity_review"][offset].copy(),
        projected_xy=projected,
        finite=finite,
        inside=inside,
        rounded_pixel_xy=pixel,
        rounded_inside=rounded_inside,
        current_view_source=view_source,
        marker_kind=marker,
        frame_idx=np.asarray(candidate["frame_idx"][offset], np.int64),
    )


def dashed(canvas, first, second, color):
    length = float(np.linalg.norm(np.asarray(second) - first))
    for start in np.arange(0, length, 10):
        a = np.rint(first + (second - first) * (start / length)).astype(int)
        b = np.rint(first + (second - first) * (min(start + 5, length) / length)).astype(int)
        cv2.line(canvas, tuple(a), tuple(b), color, 1, cv2.LINE_AA)


def panel(raw, title, projection=None, view_index=None):
    require(raw.dtype == np.uint8 and raw.shape == (1200, 1200, 3), "native1200 BGR required")
    canvas = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
    canvas[IMAGE_Y : IMAGE_Y + 600] = cv2.resize(raw, (600, 600), interpolation=cv2.INTER_AREA)
    drawing.text(canvas, title, (12, 25), size=0.61)
    if projection is None:
        drawing.text(canvas, "Raw: full native1200 field, display 50%", (12, 49), size=0.46)
        drawing.text(canvas, "Unmarked pixels; fixed diagnostic times", (12, 685), size=0.46)
        drawing.text(
            canvas, "Anonymous slots are not verified fish identities", (12, 708), size=0.43
        )
        drawing.text(canvas, "Flags are unresolved identity-review state", (12, 731), size=0.43)
        return canvas
    require(type(view_index) is int and 0 <= view_index < 3, "ordered camera index required")
    uv, inside = projection["projected_xy"][view_index], projection["inside"][view_index]
    marker = projection["marker_kind"][view_index]
    points = np.full((3, 3, 2), -1, np.int32)
    points[inside] = np.clip(np.rint(uv[inside] * 0.5), 0, 599).astype(np.int32)
    points[..., 1] += IMAGE_Y
    for slot, color in enumerate(COLORS):
        for n in range(2):
            if inside[slot, n] and inside[slot, n + 1]:
                dashed(canvas, points[slot, n], points[slot, n + 1], color)
        for node in range(3):
            if not inside[slot, node]:
                continue
            x, y = (int(v) for v in points[slot, node])
            kind = int(marker[slot, node])
            cv2.circle(
                canvas, (x, y), 1 if kind == 1 else 4, color, 1 if kind == 0 else -1, cv2.LINE_AA
            )
            drawing.text(
                canvas,
                f"{slot}{NODES[node]}",
                (min(x + 6, 567), max(y - 6, IMAGE_Y + 12)),
                color,
                0.42,
            )
    drawing.text(
        canvas,
        f"Native-inside {int(inside.sum())}/9; M mouth / P chest / T tail",
        (12, 49),
        size=0.44,
    )
    drawing.text(canvas, "Filled: view source + global2view support", (12, 685), size=0.44)
    drawing.text(canvas, "Dot: view source only; open: no view source", (12, 708), size=0.44)
    review = projection["identity_review"]
    require(review.shape == (3,) and review.dtype == bool, "three sticky review flags required")
    flags = " ".join(f"S{b}={int(value)}" for b, value in enumerate(review))
    drawing.text(canvas, f"Sticky review: {flags} (not swap labels)", (12, 731), size=0.43)
    return canvas


def composite(raw, view, native, projections):
    vi = VIEWS.index(view)
    names = ("Original", "LK transport", "CT-first transport")
    return np.hstack(
        [panel(raw, f"{view} | native {native}")]
        + [
            panel(raw, name + " | terminal200 backtrace", p, vi)
            for name, p in zip(names, projections, strict=True)
        ]
    )


def bind_geometry_lineage(tracking_seal, context, evidence):
    """SL rows are copied prepared observations; calibration is read directly.

    Both methods consumed the same sealed preparation. The producer did not
    reopen the three original SL NPZs; their recorded SHA comes from that
    successful preparation's direct inputs and plan.
    """
    shared, preparation = context["plan"]["shared_SL_calibration_sha256"], context["seal"]
    require(len(shared) == 7, "same SL3/calibration4 required")
    require(
        tracking_seal["inputs"].get(str(inputs.PREPARATION / "preparation.provenance.json"))
        == inputs.PREPARATION_SHA,
        "tracking preparation success seal differs",
    )
    sl = [p for p in shared if Path(p).suffix == ".npz"]
    calibration = [p for p in shared if Path(p).suffix == ".json"]
    require(len(sl) == 3 and len(calibration) == 4, "SL/calibration scope differs")
    for path, expected in shared.items():
        require(
            preparation["inputs"].get(path) == expected, "prepared SL/calibration origin differs"
        )
        if path in calibration:
            require(
                tracking_seal["inputs"].get(path) == expected, "direct calibration origin differs"
            )
            evidence.add(path, expected)
    return dict(
        SL3_access="copied prepared detections; original SL NPZ not reopened",
        SL3_sha256={p: shared[p] for p in sl},
        calibration4_access="direct current calibrated rig, same prepared source SHA",
        calibration4_sha256={p: shared[p] for p in calibration},
        preparation_success_sha256=inputs.PREPARATION_SHA,
    )


def load_inputs(root, digest, evidence):
    # Late import refuses the unpublished adapter instead of silently accepting V1.
    producer = importlib.import_module("run_mask_transport200_v2")
    seal, members = base.verified(
        root, "tracking.provenance.json", PRODUCER_STAGE, digest, evidence
    )
    source = Path(producer.__file__).resolve()
    require(str(source) in seal["inputs"], "V2 producer source is not bound")
    evidence.add(source, seal["inputs"][str(source)])
    plan, summary, control = (
        base.bound_json(root, members, name)
        for name in ("plan.json", "summary.json", "original_control.json")
    )
    validate_completed(plan, summary, control)
    reuse = base.bound_json(root, members, "original_reuse_qualification.json")
    validate_reuse(plan, summary, reuse)
    common = inputs.load_common(evidence)
    require(common["original_context"] == plan["original_context"], "original context changed")
    common["visual_geometry_lineage"] = bind_geometry_lineage(seal, common["context"], evidence)
    for path, expected in {
        str(inputs.PREPARATION / "preparation.provenance.json"): inputs.PREPARATION_SHA,
        str(FAILED_ROOT / "tracking.provenance.json"): FAILED_SHA,
    }.items():
        require(seal["inputs"].get(path) == expected, "tracking/raw/rig binding differs: " + path)
        evidence.add(path, expected)
    candidates = {a: producer.read_arrays(root, members, f"{a}/candidate.npz") for a in ARMS}
    for c in candidates.values():
        validate_candidate(c)
    return common, candidates


def save_png(output, name, image):
    ok, encoded = cv2.imencode(".png", image)
    require(ok, "PNG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    require(decoded is not None and np.array_equal(decoded, image), "PNG byte roundtrip differs")
    output.write(name, encoded.tobytes())


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
            require(frame.shape == (PANEL_H, 4 * PANEL_W, 3), "review video dimensions differ")
            count += 1
    finally:
        capture.release()
    require(count == 200, "review video must decode all200 frames")
    return dict(frames=count, fps=20, width=4 * PANEL_W, height=PANEL_H)


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
        failed_visual = WORK / "mask_transport200_visual_review_v1/review.provenance.json"
        evidence.add(failed_visual, FAILED_VISUAL_SHA)
        evidence.add(WORK / "render_mask_transport200.py", FAILED_VISUAL_SOURCE_SHA)
        failure = base.read_json(failed_visual)
        require(
            failure["stage"] == "mask_transport200_visual_review_v1"
            and failure["status"] == "failed"
            and failure["outputs"] == {},
            "preserved technical visual failure differs",
        )
        common, candidates = load_inputs(root, tracking_sha256, evidence)
        prepared.snapshot_sources(evidence, output, captured)
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                tracking_sha256=tracking_sha256,
                technical_retry_of_visual_seal_sha256=FAILED_VISUAL_SHA,
                geometry_lineage=common["visual_geometry_lineage"],
                frames=list(FRAMES),
                views=list(VIEWS),
                arms=list(ARMS),
                native_interval=[FIRST, STOP],
                terminal_horizon_exclusive=STOP,
                raw_native_shape=[1200, 1200],
                displayed_image_shape=[600, 600],
                columns=["Raw", *ARMS],
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
                original_reused_from_failed_stage_sha256=FAILED_SHA,
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

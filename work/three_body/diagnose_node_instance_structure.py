"""Read-only SLEAP instance-structure diagnosis of fixed body-pool experiments.

Rows are frame-local 2D SLEAP instances, not biological identity or D2 instances.
Mixed rows establish loss of the input grouping, not a proven biological swap.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import audit_body_volume_probe as audit
import cv2
import numpy as np

p = audit.p
NODES = ("mouth", "pectoral", "tail")


def source_structure(source, points, scores, observed):
    source = np.asarray(source)
    observed = np.asarray(observed)
    if source.shape != (3, 3, 3) or source.dtype.kind not in "iu":
        raise ValueError("source must be fish,view,node integer rows")
    if observed.shape != (3, 3) or observed.dtype != bool:
        raise ValueError("direct observed must be fish,node bool")
    if np.any(source < -1):
        raise ValueError("invalid missing row sentinel")
    cells, sharing = [], []
    for vi, view in enumerate(p.base.VIEWS):
        xy, score = points[view], scores[view]
        if xy.ndim != 3 or xy.shape[1:] != (3, 2) or score.shape != (len(xy),):
            raise ValueError("2D instance points and scalar scores have inconsistent shapes")
        if np.any(source[:, vi] >= len(xy)):
            raise ValueError("frame-local row is out of range")
        for node in range(3):
            rows = source[:, vi, node]
            active = rows[rows >= 0]
            if len(np.unique(active)) != len(active):
                raise ValueError("same view/node row is assigned to two bodies")
        for fish in range(3):
            rows = source[fish, vi]
            active = rows[rows >= 0]
            direct_rows = rows[(rows >= 0) & observed[fish]]
            count = len(active)
            cell = dict(
                raw_slot=fish,
                view=view,
                rows=rows.copy(),
                supported_view_nodes=count,
                distinct_2d_rows=len(np.unique(active)),
                grouping_testable=count >= 2,
                mixed_rows=count >= 2 and len(np.unique(active)) > 1,
                all_three_from_one_row=count == 3 and len(np.unique(active)) == 1,
                grouping_testable_direct_nodes=len(direct_rows) >= 2,
                mixed_rows_direct_nodes=(len(direct_rows) >= 2 and len(np.unique(direct_rows)) > 1),
                observed_nodes=observed[fish].copy(),
                nodes=[],
            )
            for node, row in enumerate(rows):
                if row >= 0 and not np.isfinite(xy[row, node]).all():
                    raise ValueError("a supported source points to a missing SLEAP node")
                cell["nodes"].append(
                    dict(
                        node=NODES[node],
                        row=int(row),
                        point_xy=xy[row, node] if row >= 0 else [None, None],
                        instance_score=float(score[row]) if row >= 0 else None,
                        direct_3d_observed=bool(observed[fish, node]),
                    )
                )
            cells.append(cell)
        for row in np.unique(source[:, vi][source[:, vi] >= 0]):
            uses = np.argwhere(source[:, vi] == row)
            if len(np.unique(uses[:, 0])) > 1:
                sharing.append(
                    dict(
                        view=view,
                        row=int(row),
                        uses=[
                            dict(
                                raw_slot=int(f),
                                node=NODES[n],
                                direct_3d_observed=bool(observed[f, n]),
                            )
                            for f, n in uses
                        ],
                        different_node_types_only=True,
                    )
                )
    return dict(cells=cells, cross_body_shared_rows=sharing)


def geometry_errors(rig, pose, source, points, rays, observed):
    replay_source, replay_observed, ray_error = p.supported_nodes(pose, source, rays, rig)
    audit.equal(replay_source, source)
    audit.equal(replay_observed, observed)
    pixel = np.full((3, 3, 3), np.nan)
    for vi, view in enumerate(p.base.VIEWS):
        projected = rig.camera(view).project(pose)
        for fish, node in np.argwhere(source[:, vi] >= 0):
            row = source[fish, vi, node]
            pixel[fish, vi, node] = np.linalg.norm(projected[fish, node] - points[view][row, node])
    return dict(
        ray_distance_mm=np.where(source >= 0, ray_error, np.nan), reprojection_error_px=pixel
    )


def aggregate(cases):
    groups = {}
    for case in cases:
        key = case["variant"]
        result = groups.setdefault(key, Counter())
        cells = case["structure"]["cells"]
        result["frames"] += 1
        result["body_view_cells"] += len(cells)
        result["grouping_testable_cells"] += sum(c["grouping_testable"] for c in cells)
        result["mixed_row_cells"] += sum(c["mixed_rows"] for c in cells)
        result["all_three_from_one_row_cells"] += sum(c["all_three_from_one_row"] for c in cells)
        result["grouping_testable_direct_node_cells"] += sum(
            c["grouping_testable_direct_nodes"] for c in cells
        )
        result["mixed_row_direct_node_cells"] += sum(c["mixed_rows_direct_nodes"] for c in cells)
        result["frames_with_mixed_rows"] += any(c["mixed_rows"] for c in cells)
        result["cross_body_shared_row_events"] += len(case["structure"]["cross_body_shared_rows"])
        result["frames_with_cross_body_row_sharing"] += bool(
            case["structure"]["cross_body_shared_rows"]
        )
    return {key: dict(value) for key, value in groups.items()}


def row_overlay(frame, points, scores):
    """Frame-local row colours are intentionally independent across cameras."""
    images = {}
    colors = [(120, 160, 255), (130, 255, 150), (255, 200, 110), (210, 120, 240)]
    for view in p.base.VIEWS:
        image = frame.images_bgr[view].copy()
        for row, xy in enumerate(points[view]):
            color = colors[row % len(colors)]
            for a, b in [(0, 1), (1, 2)]:
                if np.isfinite(xy[[a, b]]).all():
                    cv2.line(
                        image,
                        tuple(np.rint(xy[a]).astype(int)),
                        tuple(np.rint(xy[b]).astype(int)),
                        color,
                        1,
                    )
            for node, point in enumerate(xy):
                if np.isfinite(point).all():
                    coordinate = tuple(np.rint(point).astype(int))
                    cv2.circle(image, coordinate, 3, color, 1)
                    label = f"r{row}{NODES[node][0].upper()}({scores[view][row]:.2f})"
                    cv2.putText(
                        image,
                        label,
                        (coordinate[0] + 4, coordinate[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.28,
                        color,
                        1,
                    )
        images[view] = image
    return p.FrameObservation(
        frame.frame_idx, frame.detections, images, frame.masks, frame.backgrounds_bgr
    )


def report_text(summary):
    lines = [
        "# Frame-local SLEAP row structure",
        "",
        "A different row means a different SLEAP 2D instance in one view and frame. "
        "It is not a D2 instance number, a temporal track ID or human animal identity.",
        "",
        "| Variant | Testable body-view cells | Mixed rows | Direct-node testable | "
        "Direct-node mixed | Shared 2D rows across 3D bodies |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, data in summary["aggregate"].items():
        lines.append(
            f"| {name} | {data['grouping_testable_cells']} | "
            f"{data['mixed_row_cells']} | {data['grouping_testable_direct_node_cells']} | "
            f"{data['mixed_row_direct_node_cells']} | {data['cross_body_shared_row_events']} |"
        )
    lines += [
        "",
        "There are 117 body-view cells per thirteen-frame variant; cells with fewer "
        "than two assigned nodes cannot establish row agreement.",
        "",
        "At 79372, tuples below are (mouth, pectoral, tail); -1 means no supported source.",
        "",
        "| Variant | Raw body | Top | Side1 | Side2 | Direct 3D nodes M/P/T |",
        "|---|---|---|---|---|---|",
    ]
    for case in summary["cases"]:
        if case["frame"] != 79372:
            continue
        for fish in range(3):
            cells = {c["view"]: c for c in case["structure"]["cells"] if c["raw_slot"] == fish}
            rows = [tuple(int(v) for v in cells[view]["rows"]) for view in p.base.VIEWS]
            observed = tuple(bool(v) for v in cells[p.base.VIEWS[0]]["observed_nodes"])
            lines.append(
                f"| {case['variant']} | S{fish} | {rows[0]} | {rows[1]} | {rows[2]} | {observed} |"
            )
    lines += [
        "",
        "What the current input pipeline retains:",
        "",
        "- The C01 cache contains grouped points[N,3,2] and one scalar score[N]. "
        "Its stable frame sort preserves within-frame instance rows. Per-node scores "
        "and pair-affinity information are absent from this cache schema.",
        "- CachedNodes reads only frame_idx and points. FrameObservation and "
        "RayObservations preserve the row/node coordinate array but carry no confidence "
        "score. The scalar score still exists in the original NPZ and is recorded here.",
        "- assignment_plans solves an independent injective assignment for each camera "
        "and node. It prevents two 3D bodies sharing the SAME view/node observation. "
        "It does not penalize one body taking different nodes from different 2D rows, "
        "or two bodies using different node types from the same 2D row.",
        "- supported_nodes checks ray distance and reprojection, then requires at least "
        "two views per observed 3D node. Neither this check nor the fixed image-union "
        "loss verifies that mouth, pectoral and tail came from one coherent 2D fish.",
        "",
        "This diagnosis proves loss of the supplied 2D grouping in those assignments. "
        "SLEAP grouping itself can be wrong around overlaps, so enforcing a whole instance "
        "as an indivisible truth could discard good partial nodes. A future observation "
        "model would need to retain grouping/score evidence while allowing explicit "
        "missing or misgrouped-node alternatives. No such model is implemented or tested here.",
        "",
        "All assigned sources were rechecked with the existing 2 mm / 12 px gates; "
        "that geometric consistency does not prove correct anatomical or biological identity. "
        "No fit, new ID verdict, review clearing, or trajectory modification occurred.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(p.REPO) or output.exists():
        raise FileExistsError("new workspace diagnostic output required")
    output.mkdir(parents=True)
    with p.provenance(
        "three_body_sleap_instance_structure_diagnostic", output, name="diagnostic"
    ) as rec:
        code_paths = [
            *p.source_paths(),
            Path(__file__).resolve(),
            Path(audit.__file__).resolve(),
            p.REPO / "tools/c01_s16_predictions.py",
        ]
        hashes = {str(path): p.base.sha256(path) for path in code_paths}
        seal, bound = audit.completed(args.run.resolve(), "three_body_local_volume_start_probe")
        audit.merge_hashes(hashes, bound)
        a_summary = json.loads(audit.require_output(args.run / "summary.json", seal).read_text())
        run = p._verify_run(Path(a_summary["original_geometry_run"]), appearance=False)
        audit.merge_hashes(hashes, run.hashes)
        for path, digest in hashes.items():
            rec.input(path, path, sha256=digest)
        geometry = p._npz(run.root / "candidate.npz")
        p.persistent.validate_arm_arrays(geometry, run.frames)
        nodes = p.CachedNodes(run.input_root, p.base.EXPERIMENT, run.frames)
        p.base.validate_result(geometry, run.frames, nodes)
        rig = p.load_rig(run.root / "calibration_snapshot")
        cache, schemas = {}, {}
        for view in p.base.VIEWS:
            path = nodes.paths[view]
            with np.load(path, allow_pickle=False) as arrays:
                frame_idx, points, scores = arrays["frame_idx"], arrays["points"], arrays["score"]
                if scores.shape != (len(frame_idx),) or points.shape != (len(frame_idx), 3, 2):
                    raise ValueError("unexpected cache points/score schema")
                schemas[view] = dict(
                    path=str(path),
                    sha256=p.base.sha256(path),
                    keys=arrays.files,
                    points_dtype=str(points.dtype),
                    score_dtype=str(scores.dtype),
                    score_scope="one scalar per frame-local SLEAP instance",
                )
                for frame in p.FRAMES:
                    lo = np.searchsorted(frame_idx, frame, side="left")
                    hi = np.searchsorted(frame_idx, frame, side="right")
                    cache[frame, view] = dict(
                        points=points[lo:hi].copy(),
                        scores=scores[lo:hi].copy(),
                        global_cache_rows=np.arange(lo, hi, dtype=np.int64),
                    )
        cases, payload, csvrows, frame79372 = [], {}, [], []
        for saved in a_summary["frames"]:
            frame = saved["frame"]
            t = int(np.searchsorted(run.frames, frame))
            root = args.run / f"frame_{frame}"
            arrays = p._npz(audit.require_output(root / "candidates.npz", seal))
            observations = p._npz(audit.require_output(root / "observations.npz", seal))
            points = {view: cache[frame, view]["points"] for view in p.base.VIEWS}
            scores = {view: cache[frame, view]["scores"] for view in p.base.VIEWS}
            for view in p.base.VIEWS:
                audit.equal(points[view], observations["sleap_" + view])
                audit.equal(points[view], nodes.at(view, frame))
                for key, value in cache[frame, view].items():
                    payload[f"frame{frame}_{view}_{key}"] = value
            rays = p.RayObservations.from_points(rig, points)
            chosen = saved["selected"]["fit_indices"]["mask"]
            variants = [
                (
                    "saved_original",
                    geometry["pose_xyz"][t],
                    geometry["sources"][t],
                    geometry["observed"][t],
                    None,
                ),
                (
                    "fresh_original_before_fit",
                    arrays["fit_input_pose_xyz"][0],
                    arrays["fit_initial_sources"][0],
                    arrays["fit_initial_observed"][0],
                    arrays["fit_plans"][0],
                ),
                (
                    "refitted_original",
                    arrays["fit_pose_xyz"][0],
                    arrays["fit_sources"][0],
                    arrays["fit_observed"][0],
                    arrays["fit_plans"][0],
                ),
                (
                    "selected_mask",
                    arrays["fit_pose_xyz"][chosen],
                    arrays["fit_sources"][chosen],
                    arrays["fit_observed"][chosen],
                    arrays["fit_plans"][chosen],
                ),
            ]
            if frame == 79372:
                for trial in (6, 15):
                    fit = int(arrays["mask_fit_index"][trial])
                    if fit < 0:
                        raise ValueError("the registered detailed nominal trial was rejected")
                    for stage, pose_key, source_key, observed_key in [
                        (
                            "start",
                            "fit_input_pose_xyz",
                            "fit_initial_sources",
                            "fit_initial_observed",
                        ),
                        ("fit", "fit_pose_xyz", "fit_sources", "fit_observed"),
                    ]:
                        variants.append(
                            (
                                f"mask_nominal_trial_{trial}_{stage}",
                                arrays[pose_key][fit],
                                arrays[source_key][fit],
                                arrays[observed_key][fit],
                                arrays["fit_plans"][fit],
                            )
                        )
            for name, pose, source, observed, plan in variants:
                structure = source_structure(source, points, scores, observed)
                errors = geometry_errors(rig, pose, source, points, rays, observed)
                case = dict(
                    frame=frame,
                    variant=name,
                    structure=structure,
                    planned_sources=plan,
                    pose_xyz=pose,
                    old_identity_review=geometry["identity_review"][t],
                    **errors,
                )
                cases.append(case)
                prefix = f"frame{frame}_{name}_"
                payload.update(
                    {
                        prefix + key: value
                        for key, value in dict(
                            pose_xyz=pose, sources=source, observed=observed, **errors
                        ).items()
                    }
                )
                if plan is not None:
                    payload[prefix + "planned_sources"] = plan
                for cell in structure["cells"]:
                    for node, info in enumerate(cell["nodes"]):
                        fish, vi = cell["raw_slot"], p.base.VIEWS.index(cell["view"])
                        csvrows.append(
                            dict(
                                frame=frame,
                                variant=name,
                                raw_slot=fish,
                                view=cell["view"],
                                **info,
                                row_group_mixed=cell["mixed_rows"],
                                ray_error_mm=errors["ray_distance_mm"][fish, vi, node],
                                pixel_error=errors["reprojection_error_px"][fish, vi, node],
                            )
                        )
                if frame == 79372 and name in (
                    "refitted_original",
                    "mask_nominal_trial_6_fit",
                    "mask_nominal_trial_15_fit",
                ):
                    frame79372.append((name, pose.copy(), observed.copy()))
        if [case["frame"] for case in cases if case["variant"] == "saved_original"] != list(
            p.FRAMES
        ):
            raise ValueError("not all registered frames were diagnosed")
        iterator = p.decoded_probes(run, nodes)
        try:
            for _, frame, _raw_evidence in iterator:
                if frame.frame_idx == 79372:
                    break
            else:
                raise ValueError("registered native overlay frame absent")
        finally:
            iterator.close()
        row_frame = row_overlay(
            frame,
            {v: cache[79372, v]["points"] for v in p.base.VIEWS},
            {v: cache[79372, v]["scores"] for v in p.base.VIEWS},
        )
        image, native = p.preview(
            rig,
            row_frame,
            np.stack([row[1] for row in frame79372]),
            np.stack([row[2] for row in frame79372]),
            [row[0] for row in frame79372],
        )
        image_path = output / "sleap_rows_and_3d_79372.jpg"
        if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError("native overlay write failed")
        rec.output("overlay", image_path, sha256=p.base.sha256(image_path))
        p.save_arrays(rec, output, "arrays", "sources_and_scores.npz", payload)
        summary = dict(
            status="completed_sleap_row_structure_diagnostic",
            frames=list(p.FRAMES),
            aggregate=aggregate(cases),
            cache_schema=schemas,
            cases=cases,
            native_overlay_coordinates=native,
            native_overlay_raw_evidence=_raw_evidence,
            mask_trial_indexing="zero-based nominal mask_pose_xyz: trial6->fit3, trial15->fit6",
            instance_rows_are_not_biological_ids=True,
            same_node_source_collisions=0,
            strict_geometry_gate_recheck_passed=True,
            gpu_used=False,
            refit_performed=False,
            ground_truth_loaded=False,
        )
        p.save_json(rec, output, "summary", "summary.json", summary)
        table = output / "node_sources.csv"
        with table.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csvrows[0]))
            writer.writeheader()
            for row in csvrows:
                writer.writerow(p.clean(row))
        rec.output("source_table", table, sha256=p.base.sha256(table))
        report = output / "RESULT.md"
        report.write_text(report_text(summary), encoding="utf-8")
        rec.output("report", report, sha256=p.base.sha256(report))
        p.assert_hashes({**hashes, **{v["path"]: v["sha256"] for v in rec.outputs.values()}})
        rec.param("all_input_source_output_sha_unchanged", True)
    return summary


if __name__ == "__main__":
    main()

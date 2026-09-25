"""Review anonymous whole-row geometry on saved photographs; no tracking or fitting.

Complete scenes use saved projections. For cases without a complete scene, only
saved pool poses are projected using the four registered calibration JSON files.
Partial nodes remain NaN; all displayed body markers are open proposals.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import numpy as np
import render_partial_body_bank_comparison as drawing

proof, require, prior = drawing.proof, drawing.require, drawing.prior
WORK, VIEWS = drawing.WORK, drawing.VIEWS
ROOT = WORK / "whole_row_body_geometry_diagnostic_v1"
GEOMETRY_SHA = "39208ffcd2314098fcf89d944ef9bedbe949fd6c6dbc4c19bc88238acf4a27fb"
BASE_ROOT = WORK / "partial_body_saved_bank_comparison_v1"
BASE_SHA = "10d29aad72ca98674abbcbaf386ce48244087cb4abae552b51e592b5dbb9910f"
DRAWING_SHA = "7658faac53fa0cb19ccfb6b1323144e26ef4c4ef8d4030fb4182d988af3d684e"
STAGE = "whole_row_body_geometry_visual_review_v1"
PER_PAGE = 4


def read_context(evidence):
    seal = drawing.read(ROOT / "geometry.provenance.json", GEOMETRY_SHA, evidence)
    require(seal["stage"] == ROOT.name and seal["status"] == "ok", "successful geometry required")
    plan, summary, payload = [
        drawing.document(ROOT, seal, name + ".json", evidence)
        for name in ("plan", "summary", "render_payload")
    ]
    require(
        plan["input_sha256"] == BASE_SHA and summary["status"] == "ok", "geometry input differs"
    )
    require(
        payload["schema"] == ROOT.name + "_render_payload"
        and payload["body_order_is_anonymous"]
        and payload["physical_ID_truth"] is False
        and payload["new_trajectory_created"] is False,
        "anonymous geometry scope differs",
    )
    context = drawing.load_context(BASE_ROOT, BASE_SHA, evidence)
    evidence.add(WORK / "render_partial_body_bank_comparison.py", DRAWING_SHA)
    context["source_code"][str(WORK / "render_partial_body_bank_comparison.py")] = DRAWING_SHA
    context["source_code"][str(Path(__file__).resolve())] = evidence.add(Path(__file__).resolve())
    expected = {(r["arm"], r["frame"]) for r in context["cases"]}
    require(
        len(summary["cases"]) == len(payload["cases"]) == len(expected) == 11,
        "eleven geometry cases required",
    )
    require(
        {(r["arm"], r["frame_idx"]) for r in summary["cases"]} == expected
        and {(r["arm"], r["frame_idx"]) for r in payload["cases"]} == expected,
        "geometry case membership differs",
    )
    context.update(
        geometry_seal=seal, geometry_plan=plan, geometry_summary=summary, payload=payload
    )
    return context


def load_rig(context, evidence):
    seal = context["geometry_seal"]
    manifests = [Path(p) for p in seal["inputs"] if Path(p).name == "calibration_manifest.json"]
    require(len(manifests) == 1, "one consumed rig manifest required")
    manifest_path = manifests[0]
    manifest = drawing.read(manifest_path, seal["inputs"][str(manifest_path)], evidence)
    models = [
        prior.native.checked_relative(manifest_path.parent, r["path"])
        for r in manifest["model_files"]
    ]
    require(len(models) == len(set(models)) == 3, "three camera models required")
    references = {str(manifest_path): seal["inputs"][str(manifest_path)]}
    for path in models:
        digest = seal["inputs"].get(str(path))
        require(proof.valid_sha(digest), "camera model consumed edge missing")
        evidence.add(path, digest)
        references[str(path)] = digest
    calib = drawing.frozen.observations.calib
    path = str(Path(calib.__file__).resolve())
    expected = context["geometry_plan"]["source_code"][path]
    drawing.edge(seal, path, expected)
    evidence.add(Path(path), expected)
    context["source_code"][path] = expected
    rig = calib.load_rig(manifest_path.parent)
    require(tuple(rig.views) == tuple(VIEWS) and rig.world_units == "mm", "rig views/units differ")
    return rig, references


def project_partial(rig, pose, measured):
    require(
        pose.shape == (3, 3) and measured.shape == (3,) and measured.dtype == np.bool_,
        "pool body schema differs",
    )
    require(
        np.array_equal(np.isfinite(pose).all(-1), measured) and np.isnan(pose[~measured]).all(),
        "partial pose must remain paired NaN",
    )
    projected = {}
    for view in VIEWS:
        uv = np.full((1, 3, 2), np.nan)
        uv[0, measured] = rig.camera(view).project(pose[measured])
        require(np.isfinite(uv[0, measured]).all(), "finite pool projection required")
        projected[view] = uv
    return projected


def pool_rows(case, summary, context, rig, evidence):
    stem = f"cases/{case['arm']}_{case['frame_idx']}"
    seal = context["geometry_seal"]
    doc = drawing.document(ROOT, seal, stem + ".json", evidence)
    path = drawing.previous.member(ROOT, seal, stem + ".npz", evidence)
    require(
        doc["archive"] == summary["archive"] == seal["outputs"][stem + ".npz"],
        "pool archive binding differs",
    )
    tree = doc["tree"]
    require(
        tree["arm"] == case["arm"]
        and tree["frame_idx"] == case["frame_idx"]
        and tree["no_temporal_identity_assignment"]
        and tree["direct_support_not_recomputed"],
        "pool scope differs",
    )
    signatures, rows = {}, []
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:

        def array(desc):
            value = archive[desc["__array__"]]
            require(
                proof.signature(value) == desc["signature"] and not value.dtype.hasobject,
                "pool array signature differs",
            )
            signatures[desc["__array__"]] = desc["signature"]
            return value.copy()

        for i, body in enumerate(tree["pool"]):
            require(
                body["pool_index"] == i and body["same_row_within_each_view"], "pool row differs"
            )
            pose, measured, source_rows, rms = [
                array(body[key])
                for key in ("pose_xyz", "triangulated_nodes", "view_rows", "ray_rms_mm")
            ]
            require(source_rows.shape == (3,) and source_rows.dtype.kind in "iu", "view row schema")
            rows.append(
                dict(
                    kind="individual_pool_body",
                    index=i,
                    pool_indices=[i],
                    pose_xyz=pose[None],
                    triangulated_nodes=measured[None],
                    source_rows=source_rows[None],
                    ray_rms_mm=rms[None],
                    projections=project_partial(rig, pose, measured),
                    projection_origin="new projection of saved pool pose; registered rig4",
                )
            )
    require(len(rows) == summary["pool"]["geometry_candidates"], "pool count differs")
    return rows, dict(archive=doc["archive"], selected_array_signatures=signatures)


def scene_rows(case, summary):
    rows = []
    require(
        len(case["scenes"]) == summary["scenes"]["scenes_returned"] <= PER_PAGE,
        "saved scene count differs",
    )
    for i, scene in enumerate(case["scenes"]):
        pose = np.asarray(scene["pose_xyz"], np.float64)
        source_rows = np.asarray(scene["source_rows"], np.int32)
        projections = {v: np.asarray(scene["projections"][v], np.float64) for v in VIEWS}
        require(
            scene["rank"] == i
            and pose.shape == (3, 3, 3)
            and np.isfinite(pose).all()
            and source_rows.shape == (3, 3)
            and len(set(scene["pool_indices"])) == 3
            and all(p.shape == (3, 3, 2) and np.isfinite(p).all() for p in projections.values()),
            "complete scene schema differs",
        )
        for vi in range(3):
            used = source_rows[:, vi]
            used = used[used >= 0]
            require(len(set(used)) == len(used), "scene reuses a detector row")
        rows.append(
            dict(
                kind="complete_anonymous_scene",
                index=i,
                pool_indices=scene["pool_indices"],
                pose_xyz=pose,
                source_rows=source_rows,
                triangulated_nodes=np.ones((3, 3), bool),
                projections=projections,
                projection_origin="saved geometry scene projections",
                total_used_rays=scene["total_used_rays"],
                sum_squared_ray_rms_mm=scene["sum_squared_ray_rms_mm"],
            )
        )
    return rows


def draw_panel(ax, image, points, row, view):
    ax.imshow(prior.cv2.cvtColor(image, prior.cv2.COLOR_BGR2RGB))
    for i, nodes in enumerate(points):
        finite = nodes[np.isfinite(nodes).all(1)]
        if len(finite):
            ax.plot(finite[:, 0], finite[:, 1], ls="none", marker="+", color="#ff9632", ms=4)
        if np.isfinite(nodes[0]).all():
            ax.annotate(f"SL{i}M", nodes[0], color="#ff9632", fontsize=7, clip_on=True)
    for b, nodes in enumerate(row["projections"][view]):
        label = f"P{row['pool_indices'][b]}" if row["kind"] == "individual_pool_body" else f"B{b}"
        color = drawing.COLORS[b]
        for left, right in ((0, 1), (1, 2)):
            if np.isfinite(nodes[[left, right]]).all():
                ax.plot(
                    nodes[[left, right], 0], nodes[[left, right], 1], color=color, ls="--", lw=1.5
                )
        for xy, node, marker in zip(nodes, drawing.LABELS, drawing.MARKERS, strict=True):
            if np.isfinite(xy).all():
                ax.plot(*xy, marker=marker, markerfacecolor="none", color=color, ms=7)
                ax.annotate(
                    label + node,
                    xy,
                    xytext=(4, 4),
                    textcoords="offset points",
                    color=color,
                    fontsize=8,
                    clip_on=True,
                )


def render_page(case, rows, photos, points, crops, page, total):
    fig, axes = prior.plt.subplots(
        len(rows), 3, squeeze=False, figsize=(18, 1.6 + 4.8 * len(rows)), dpi=140
    )
    try:
        bound = -12 if case["arm"].startswith("expanded") else -6
        photo_type = "lossless native PNG" if case["frame_idx"] in drawing.PRIMARY else "LOSSY JPEG"
        fig.suptitle(
            f"Native {case['frame_idx']} | {case['arm']} | z lower={bound} mm engineering bound | "
            f"page {page}/{total} | {photo_type}\n"
            "GEOMETRY ONLY: same SLEAP row within each view. Anonymous B0/B1/B2 or pool P index.\n"
            "Open M circle / C square / T triangle = proposed geometry; "
            "NaN nodes are NOT completed. "
            "No fitting, temporal initialization, direct support or physical ID certification.\n"
            "Scenes: saved projections. "
            "Single pool bodies: saved poses projected with registered rig.",
            fontsize=10,
        )
        for ri, row in enumerate(rows):
            flags = [
                "/".join(n if yes else n + "=NaN" for n, yes in zip(drawing.LABELS, a, strict=True))
                for a in row["triangulated_nodes"]
            ]
            for vi, view in enumerate(VIEWS):
                ax = axes[ri, vi]
                draw_panel(ax, photos[view], points[view], row, view)
                ax.set_title(
                    f"{view} | {row['kind']} {row['index']} | pool {row['pool_indices']}\n"
                    f"same-row choices top/side1/side2={row['source_rows'].tolist()}\n"
                    f"Triangulated nodes: {flags}",
                    fontsize=8,
                )
                lo, hi = crops[view]
                ax.set_xlim(*[lo[0], hi[0]])
                ax.set_ylim(hi[1], lo[1])
                ax.tick_params(labelsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.91 if len(rows) > 1 else 0.8))
        stream = io.BytesIO()
        fig.savefig(stream, format="png")
        return stream.getvalue()
    finally:
        prior.plt.close(fig)


def geometry_json(value):
    """Display JSON: numeric NaNs are explicit null, with triangulated flags retained."""
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, "object geometry array forbidden")
        return prior.finite_json(value)
    if isinstance(value, dict):
        return {key: geometry_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [geometry_json(item) for item in value]
    if isinstance(value, np.generic):
        return geometry_json(value.item())
    require(value is None or isinstance(value, (str, bool, int, float)), "unknown display value")
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    require(
        "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(), "CPU only"
    )
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        context = read_context(evidence)
        rig, rig_refs = load_rig(context, evidence)
        photos, points, references = {}, {}, []
        for frame in sorted({c["frame_idx"] for c in context["payload"]["cases"]}):
            photos[frame], points[frame] = {}, {}
            for view in VIEWS:
                image, sl, ref = drawing.photograph_and_points(context, frame, view, evidence)
                photos[frame][view], points[frame][view] = image, sl
                references.append(ref)
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                geometry_sha256=GEOMETRY_SHA,
                source_code=context["source_code"],
                registered_rig=rig_refs,
                photographs=references,
                scene_projection_origin="saved",
                pool_projection_origin="new project of saved pose",
                original_video_mask_SL_cache_reads=0,
                geometry_rerun=False,
                GPU_used=False,
                support_recomputed=False,
                physical_ID_truth=False,
                deduplication_used=False,
            ),
        )
        by_key = {(c["arm"], c["frame_idx"]): c for c in context["geometry_summary"]["cases"]}
        records, pool_count = [], 0
        for case in context["payload"]["cases"]:
            summary = by_key[case["arm"], case["frame_idx"]]
            if case["scenes"]:
                rows, pool_ref = scene_rows(case, summary), None
            else:
                require(summary["scenes"]["row_disjoint_complete_triples"] == 0, "missing scenes")
                rows, pool_ref = pool_rows(case, summary, context, rig, evidence)
                pool_count += 1
            require(rows, "no geometry to display")
            frame = case["frame_idx"]
            crops = {
                v: drawing.visual.display_crop(
                    photos[frame][v], None, points[frame][v], [r["projections"][v] for r in rows]
                )
                for v in VIEWS
            }
            pages = []
            total = math.ceil(len(rows) / PER_PAGE)
            for page, start in enumerate(range(0, len(rows), PER_PAGE), 1):
                png = render_page(
                    case,
                    rows[start : start + PER_PAGE],
                    photos[frame],
                    points[frame],
                    crops,
                    page,
                    total,
                )
                name = f"{case['arm']}_f{frame}_p{page:02d}.png"
                pages.append(
                    dict(
                        indices=[r["index"] for r in rows[start : start + PER_PAGE]],
                        image=out.write(name, png),
                    )
                )
            records.append(
                dict(
                    arm=case["arm"],
                    frame_idx=frame,
                    kind=rows[0]["kind"],
                    pages=pages,
                    geometry_rows=geometry_json(rows),
                    pool_input=pool_ref,
                    crops={v: [lo.tolist(), hi.tolist()] for v, (lo, hi) in crops.items()},
                )
            )
            print(f"Rendered {case['arm']} {frame}: {len(rows)} rows / {total} pages", flush=True)
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                cases=records,
                contact_sheets=sum(len(c["pages"]) for c in records),
                photographs_decoded=len(references),
                pool_archives_decoded=pool_count,
                calibration_JSON_files_read=4,
                anonymous_geometry_only=True,
                missing_nodes_completed=False,
                physical_ID_truth=False,
                new_trajectory_created=False,
            ),
        )
        evidence.final_guard()
        out.final_guard()
        require(
            "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
            "CUDA initialized",
        )
        receipt = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        receipt = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(error),
        )
        (output / "render.provenance.json").write_bytes(proof.json_bytes(receipt))
        raise
    (output / "render.provenance.json").write_bytes(proof.json_bytes(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

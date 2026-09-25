"""Compare saved whole-row/nodewise geometry on seven registered development frames.

At most eleven sheets: the first legacy and new complete scene, the first added
body, and the added body with the largest omitted-view residual if different.
At82720, both row(2,2,2) alternatives replace those individual-body slots.
Residual-based selection is a stress case, not a best-body choice or accuracy.
No triangulation, template completion, tracking or support reassessment is run.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import render_whole_row_body_geometry as visual

drawing = visual.drawing
proof, require, WORK, VIEWS = visual.proof, visual.require, visual.WORK, visual.VIEWS
STAGE = "nodewise_body_row_geometry_visual_v1"
PRODUCER = "nodewise_body_row_geometry_v1"
ARMS = ("sparse_instance_rescore", "expanded_depth_instance_rescore")
FRAMES = (*drawing.PRIMARY, *drawing.NEGATIVE)
PINS = {
    "render_whole_row_body_geometry.py": (
        "4466ac00d1ef34bc9b846cc4f43c620ebcca19ffa5748d214a57f4ad7a9cfb87"
    ),
    "render_partial_body_bank_comparison.py": (
        "7658faac53fa0cb19ccfb6b1323144e26ef4c4ef8d4030fb4182d988af3d684e"
    ),
}


def select_bodies(bodies, preferred_rows=None):
    """Deterministic limited display, including the largest excluded-view conflict."""
    if not bodies:
        return []
    if preferred_rows is not None:
        selected = [
            i for i, body in enumerate(bodies) if body["observation_rows"] == list(preferred_rows)
        ]
        require(
            len(selected) <= 2, "more than two target-row alternatives requires explicit layout"
        )
        if selected:
            return selected
    scores = []
    for body in bodies:
        excluded = np.asarray(body["excluded_available_node_views"], bool).T
        errors = np.asarray(body["reprojection_error_px"], float)
        require(excluded.shape == errors.shape == (3, 3), "view/node residual axes")
        scores.append(float(errors[excluded].max()) if excluded.any() else -1.0)
    maximum = int(np.argmax(scores))
    return [0] if maximum == 0 else [0, maximum]


def observations(seal, evidence):
    primary_seal, primary_doc = drawing.observation_document(
        drawing.PRIMARY_ROOT,
        drawing.PRIMARY_SEAL_SHA,
        drawing.PRIMARY_EVIDENCE_SHA,
        seal,
        evidence,
    )
    _, old_doc = drawing.observation_document(
        drawing.OLD_ROOT,
        drawing.OLD_SEAL_SHA,
        drawing.OLD_EVIDENCE_SHA,
        seal,
        evidence,
    )
    jpeg = drawing.read(
        drawing.JPEG_ROOT / "review.provenance.json", drawing.JPEG_SEAL_SHA, evidence
    )
    require(jpeg["status"] == "ok" and jpeg["stage"] == drawing.JPEG_ROOT.name, "JPEG pack")
    drawing.edge(jpeg, drawing.OLD_ROOT / "render.provenance.json", drawing.OLD_SEAL_SHA)
    summary = drawing.document(drawing.JPEG_ROOT, jpeg, "summary.json", evidence)
    require(
        summary["review_clip_range"] == [90040, 90400]
        and summary["consumed_record"] == old_doc["source_chunk"]
        and summary["selection_render_sha256"] == drawing.OLD_SEAL_SHA,
        "JPEG lineage",
    )
    records = {}
    for doc, frames in ((primary_doc, drawing.PRIMARY), (old_doc, drawing.NEGATIVE)):
        selected = [r for r in doc["records"] if r["frame"] in frames]
        require(len(selected) == len(frames), "saved native membership")
        records.update({r["frame"]: r for r in selected})
    return dict(records=records, primary_seal=primary_seal, jpeg_seal=jpeg, jpeg_summary=summary)


def scene_row(scene, pool, label):
    if scene is None:
        return dict(kind="unavailable", label=label + ": no complete row-disjoint triple")
    ids = scene["pool_indices"]
    require(scene["rank"] == 0 and len(ids) == len(set(ids)) == 3, "top anonymous scene")
    bodies = [pool[i] for i in ids]
    pose = np.asarray(scene["pose_xyz"], float)
    require(
        np.array_equal(pose, np.asarray([b["pose_xyz"] for b in bodies], float)),
        "scene/pool pose differs",
    )
    return dict(
        kind="complete_anonymous_scene",
        label=label,
        pool_indices=ids,
        source=np.asarray(scene["proposed_sources"], int),
        projections={v: np.asarray(scene["projections"][v], float) for v in VIEWS},
        bodies=bodies,
    )


def body_row(body, label):
    require(
        body["all_nodes_triangulated_without_template"] is True and all(body["triangulated_nodes"]),
        "added body has no template completion",
    )
    return dict(
        kind="individual_pool_body",
        label=label,
        pool_indices=[body["pool_index"]],
        source=np.asarray(body["proposed_sources"], int)[None],
        bodies=[body],
        projections={v: np.asarray(body["projections"][v], float)[None] for v in VIEWS},
    )


def view_caption(row, view, points):
    if row["kind"] == "unavailable":
        return "No complete scene is substituted."
    vi = VIEWS.index(view)
    parts = []
    for body, uv, source in zip(
        row["bodies"], row["projections"][view], row["source"], strict=True
    ):
        used = "/".join(f"{n}:{int(s)}" for n, s in zip(drawing.LABELS, source[vi], strict=True))
        text = f"P{body['pool_index']} assigned SL rows {used}"
        if body.get("supplemental_nodewise_body"):
            excluded = np.asarray(body["excluded_available_node_views"], bool)[:, vi]
            residual = np.asarray(body["reprojection_error_px"], float)[vi]
            row_index = body["observation_rows"][vi]
            distances = np.linalg.norm(uv - points[row_index], axis=-1)
            available = np.asarray(body["available_node_views"], bool)[:, vi]
            require(
                np.allclose(distances[available], residual[available], rtol=0, atol=1e-7),
                "saved omitted-view residual differs from displayed points",
            )
            omitted = [f"{drawing.LABELS[n]} {residual[n]:.1f}px" for n in np.flatnonzero(excluded)]
            if omitted:
                text += "; UNUSED " + ", ".join(omitted)
        parts.append(text)
    return "\n".join(parts)


def render(case, rows, context, evidence):
    photographs, points, references, crops = {}, {}, [], {}
    shown = [r for r in rows if r["kind"] != "unavailable"]
    for view in VIEWS:
        image, sl, reference = drawing.photograph_and_points(
            context, case["frame_idx"], view, evidence
        )
        photographs[view], points[view] = image, sl
        references.append(reference)
        projected = (
            np.concatenate([r["projections"][view] for r in shown])
            if shown
            else np.empty((0, 3, 2))
        )
        crops[view] = drawing.visual.display_crop(image, None, sl, projected)
    fig, axes = visual.prior.plt.subplots(
        len(rows), 3, figsize=(18, 1.8 + 4.5 * len(rows)), squeeze=False
    )
    try:
        photo_type = (
            "lossless PNG" if case["frame_idx"] in drawing.PRIMARY else "lossy JPEG context"
        )
        fig.suptitle(
            f"Native {case['frame_idx']} | {case['arm']} | {photo_type}\n"
            "Open M/C/T = proposed geometry, NOT direct support. "
            "B0/B1/B2 and pool P are NOT physical IDs.\n"
            "All cameras shown, including UNUSED views. No template filling in added bodies. "
            "Top scene uses geometric rank, not body/image accuracy.",
            fontsize=11,
        )
        for ri, row in enumerate(rows):
            for vi, view in enumerate(VIEWS):
                ax = axes[ri, vi]
                if row["kind"] == "unavailable":
                    ax.imshow(
                        visual.prior.cv2.cvtColor(photographs[view], visual.prior.cv2.COLOR_BGR2RGB)
                    )
                    ax.text(
                        0.5,
                        0.5,
                        "NO COMPLETE SCENE",
                        transform=ax.transAxes,
                        ha="center",
                        color="white",
                        bbox=dict(facecolor="black", alpha=0.7),
                    )
                else:
                    visual.draw_panel(ax, photographs[view], points[view], row, view)
                ax.set_title(
                    row["label"] + " | " + view + "\n" + view_caption(row, view, points[view]),
                    fontsize=8,
                )
                lo, hi = crops[view]
                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(hi[1], lo[1])
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=140)
        return stream.getvalue(), references
    finally:
        visual.prior.plt.close(fig)


def run(root, digest, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(
        root.is_relative_to(WORK) and output.is_relative_to(WORK) and not output.exists(),
        "workspace input and fresh output",
    )
    out, evidence = proof.Output(output), proof.Evidence()
    for name, pin in PINS.items():
        evidence.add(WORK / name, pin)
    evidence.add(Path(__file__))
    seal = drawing.read(root / "geometry.provenance.json", digest, evidence)
    require(seal["stage"] == PRODUCER and seal["status"] == "ok", "successful new geometry")
    plan, summary, payload = [
        drawing.document(root, seal, n + ".json", evidence)
        for n in ("plan", "summary", "render_payload")
    ]
    require(
        plan["schema"] == summary["schema"] == PRODUCER
        and summary["status"] == "ok"
        and payload["schema"] == PRODUCER + "_render_payload"
        and payload["body_order_is_anonymous"]
        and not payload["physical_ID_truth"],
        "geometry scope",
    )
    producer_path = WORK / "run_nodewise_body_row_geometry.py"
    producer_sha = plan["source_code"][str(producer_path)]
    drawing.edge(seal, producer_path, producer_sha)
    evidence.add(producer_path, producer_sha)
    expected = {(a, f) for f in drawing.PRIMARY for a in ARMS} | {
        (ARMS[0], f) for f in drawing.NEGATIVE
    }
    require(
        len(payload["cases"]) == 11
        and {(c["arm"], c["frame_idx"]) for c in payload["cases"]} == expected,
        "eleven fixed engineering conditions",
    )
    context = observations(seal, evidence)
    displayed = []
    for case in payload["cases"]:
        name = f"cases/{case['arm']}_{case['frame_idx']}.json"
        require(case["case_json"] == seal["outputs"][name], "case descriptor differs")
        doc = drawing.document(root, seal, name, evidence)
        require(doc["frame_idx"] == case["frame_idx"] and doc["arm"] == case["arm"], "case native")
        pool = doc["pool"]
        require([b["pool_index"] for b in pool] == list(range(len(pool))), "pool numbering")
        for body in case["new_bodies"]:
            require(
                {k: v for k, v in body.items() if k != "projections"} == pool[body["pool_index"]],
                "new body/pool metadata differs",
            )
        rows = [
            scene_row(case[k][0] if case[k] else None, pool, label)
            for k, label in (
                ("legacy_scenes", "LEGACY top complete scene"),
                ("nodewise_scenes", "NODEWISE top complete scene"),
            )
        ]
        preferred = (2, 2, 2) if case["frame_idx"] == 82720 else None
        selected = select_bodies(case["new_bodies"], preferred)
        for i in selected:
            rows.append(
                body_row(
                    case["new_bodies"][i],
                    "TARGET row(2,2,2): complementary node views"
                    if preferred and case["new_bodies"][i]["observation_rows"] == list(preferred)
                    else "FIRST added body"
                    if i == 0
                    else "LARGEST unused-view residual: stress case",
                )
            )
        image, references = render(case, rows, context, evidence)
        path = output / f"frame_{case['frame_idx']}_{case['arm']}.png"
        path.write_bytes(image)
        out.records[path.name] = dict(
            path=str(path), sha256=proof.sha(path), size_bytes=path.stat().st_size
        )
        displayed.append(
            dict(
                frame_idx=case["frame_idx"],
                arm=case["arm"],
                photographs=references,
                rows=[
                    dict(kind=r["kind"], label=r["label"], pool_indices=r.get("pool_indices", []))
                    for r in rows
                ],
                additional_bodies_not_displayed=[
                    b["pool_index"] for i, b in enumerate(case["new_bodies"]) if i not in selected
                ],
            )
        )
    out.write(
        "summary.json",
        dict(
            schema=STAGE,
            status="ok",
            displayed=displayed,
            geometry_sha256=digest,
            all_camera_views_displayed=True,
            exhaustive_candidate_visual_review=False,
            physical_ID_truth=False,
            support_recomputed=False,
            original_video_or_SL_cache_read=False,
        ),
    )
    evidence.final_guard()
    out.final_guard()
    (output / "render.provenance.json").write_bytes(
        proof.json_bytes(dict(stage=STAGE, status="ok", inputs=evidence.paths, outputs=out.records))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.geometry, args.geometry_sha, args.output)

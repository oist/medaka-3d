"""Two saved mask-containment overlays; no distance computation or video decoding."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import render_nodewise_body_row_geometry as registered

drawing, proof, require, WORK = (
    registered.drawing,
    registered.proof,
    registered.require,
    registered.WORK,
)
VIEWS = registered.VIEWS
STAGE = "nodewise_body_mask_containment_visual_v1"
PRODUCER = "nodewise_body_mask_containment_v1"
GEOMETRY_SHA = "c8eab931e843ce15c7646104b6c8346b33f7430edca81411872a9e9ab5c86cff"
REGISTERED_SHA = "605178b98ed796ecaf5916fc34f5ab47a8f6d1be933900e910d4313568a82f58"
FOCUS = {82720: (18, 19), 90258: (9,)}
SAMPLE_LABELS = ("M", "C", "T", "MC_025", "MC_050", "MC_075", "CT_025", "CT_050", "CT_075")


def samples_of(pose):
    fractions = np.array([0.25, 0.5, 0.75])[:, None]
    return np.concatenate(
        (pose, pose[0] + fractions * (pose[1] - pose[0]), pose[1] + fractions * (pose[2] - pose[1]))
    )


def draw(ax, picture, points, mask, uv):
    plt, cv2 = registered.visual.prior.plt, registered.visual.prior.cv2
    ax.imshow(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))
    for index, instance in enumerate(mask):
        contours, _ = cv2.findContours(
            instance.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        color = plt.get_cmap("tab10")(index % 10)
        for contour in contours:
            xy = contour[:, 0]
            if len(xy) > 1:
                closed = np.concatenate((xy, xy[:1]))
                ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.05, alpha=0.95)
        yy, xx = np.nonzero(instance)
        if len(xx):
            ax.text(
                float(xx.mean()),
                float(yy.mean()),
                f"D2[{index}]",
                color=color,
                fontsize=9,
                bbox=dict(facecolor="black", alpha=0.6, pad=1),
            )
    for row, nodes in enumerate(points):
        finite = np.isfinite(nodes).all(-1)
        ax.plot(nodes[finite, 0], nodes[finite, 1], "+", color="#ff9632", ms=6, ls="none")
        if finite[0]:
            ax.annotate(f"SL{row}M", nodes[0], color="#ff9632", fontsize=7, clip_on=True)
    ax.plot(uv[:3, 0], uv[:3, 1], "--", color="#00ffff", lw=1.4)
    for point, label, marker in zip(uv[:3], ("M", "C", "T"), ("o", "s", "^"), strict=True):
        ax.plot(*point, marker=marker, markerfacecolor="none", markeredgecolor="#00ffff", ms=9)
        ax.annotate(
            label, point, xytext=(4, 4), textcoords="offset points", color="#00ffff", fontsize=9
        )
    ax.plot(
        uv[3:, 0], uv[3:, 1], "o", ms=4, markerfacecolor="none", markeredgecolor="white", ls="none"
    )


def run(root, digest, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(
        root.is_relative_to(WORK) and output == (WORK / STAGE).resolve() and not output.exists(),
        "registered measurement and fresh dedicated output",
    )
    out, evidence = proof.Output(output), proof.Evidence()
    evidence.add(Path(__file__))
    evidence.add(Path(registered.__file__), REGISTERED_SHA)
    seal = drawing.read(root / "measurement.provenance.json", digest, evidence)
    require(seal["stage"] == PRODUCER and seal["status"] == "ok", "successful measurement")
    plan, summary, signs = [
        drawing.document(root, seal, name + ".json", evidence)
        for name in ("plan", "summary", "frame_masks_signatures")
    ]
    require(
        plan["geometry_sha256"] == GEOMETRY_SHA
        and plan["frames"] == list(FOCUS)
        and plan["sample_labels"] == list(SAMPLE_LABELS),
        "fixed geometry/measurement schema",
    )
    for name in ("run_nodewise_body_mask_containment.py", "body_mask_containment.py"):
        source = WORK / name
        source_sha = plan["source_code"][str(source)]
        drawing.edge(seal, source, source_sha)
        evidence.add(source, source_sha)
    geometry_path = WORK / "nodewise_body_row_geometry_v1" / "geometry.provenance.json"
    drawing.edge(seal, geometry_path, GEOMETRY_SHA)
    geometry = drawing.read(geometry_path, GEOMETRY_SHA, evidence)
    context = registered.observations(geometry, evidence)
    mask_path = drawing.previous.member(root, seal, "frame_masks.npz", evidence)
    expected = {f"f{frame}_{view}" for frame in FOCUS for view in VIEWS}
    with np.load(mask_path, allow_pickle=False) as archive:
        require(
            len(archive.files) == 6 and set(archive.files) == set(signs) == expected,
            "six mask arrays",
        )
        masks = {key: archive[key] for key in archive.files}
    for key, mask in masks.items():
        ref = signs[key]
        require(
            proof.signature(mask) == ref["signature"]
            and mask.dtype == np.bool_
            and mask.shape == (ref["original_mask_count"], 1200, 1200),
            "native mask shape/signature",
        )
        frame, view = ref["frame_idx"], ref["view"]
        require(key == f"f{frame}_{view}" and frame in FOCUS and view in VIEWS, "mask native axes")
        ob = context["records"][frame]["observations"][view]
        require(
            drawing.prior.prepare_sequence.array_sha256(mask)
            == ref["original_mask_sha256"]
            == ob["mask_sha256"]
            and len(mask) == ob["mask_count"]
            and ref["view_available"] == ob["mask_available"],
            "saved native mask/photograph linkage",
        )
        captured = summary["captured_mask_records"][key]
        require(
            captured["path"] == ref["source_archive"]
            and seal["inputs"].get(ref["source_archive"]) == captured["sha256"],
            "captured mask edge",
        )
        mask.flags.writeable = False
    displayed = []
    for frame, focus in FOCUS.items():
        case = drawing.document(root, seal, f"cases/f{frame}.json", evidence)
        require(
            case["frame_idx"] == frame and case["focus_pool_indices"] == list(focus), "focus native"
        )
        geometry_name = f"cases/sparse_instance_rescore_{frame}.json"
        require(
            case["source_geometry_case"] == geometry["outputs"][geometry_name],
            "geometry case binding",
        )
        positions = [case["pool_indices"].index(index) for index in focus]
        require(
            [case["bodies"][p]["pool_index"] for p in positions] == list(focus),
            "body position/index annotation differs",
        )
        fig, axes = registered.visual.prior.plt.subplots(
            len(focus), 3, figsize=(18, 1.7 + 5 * len(focus)), squeeze=False
        )
        rows, photos = [], []
        try:
            for vi, view in enumerate(VIEWS):
                picture, points, reference = drawing.photograph_and_points(
                    context, frame, view, evidence
                )
                photos.append(reference)
                measurement = case["views"][view]
                require(
                    proof.signature(np.asarray(measurement["SLEAP_points"], np.float32))
                    == proof.signature(points),
                    "displayed SL/measurement points differ",
                )
                metrics = measurement["containment"]
                require(
                    metrics["sample_labels"] == list(SAMPLE_LABELS), "MCT and six interior order"
                )
                uv = np.asarray(metrics["projected_xy"], np.float64)[positions]
                require(
                    uv.shape == (len(focus), 9, 2) and np.isfinite(uv).all(),
                    "finite saved projections",
                )
                mask = masks[f"f{frame}_{view}"]
                lo, hi = drawing.visual.display_crop(picture, mask, points, uv)
                nonempty = (
                    np.flatnonzero(mask.reshape(len(mask), -1).any(-1))
                    if len(mask)
                    else np.empty(0, int)
                )
                if metrics["available"]:
                    require(
                        np.array_equal(metrics["mask_indices"], nonempty),
                        "all nonempty original masks",
                    )
                for row, (index, position) in enumerate(zip(focus, positions, strict=True)):
                    pose = np.asarray(case["bodies"][position]["pose_xyz"], float)
                    require(
                        np.array_equal(
                            samples_of(pose), np.asarray(metrics["sample_xyz"][position], float)
                        ),
                        "MCT/interior XYZ differ from saved complete body",
                    )
                    ax = axes[row, vi]
                    draw(ax, picture, points, mask, uv[row])
                    if metrics["available"]:
                        single = metrics["single_mask_max_distance_px"][position]
                        union = metrics["union_max_distance_px"][position]
                        best = metrics["best_single_mask"][position]
                        ties = metrics["best_single_mask_tie_count"][position]
                        text = (
                            f"one mask max={single:.2f}px | union max={union:.2f}px\n"
                            f"best D2[{best}], ties={ties} | all {len(mask)} D2 instances"
                        )
                    else:
                        single = union = best = ties = None
                        text = "mask evidence unavailable; no zero-distance success claimed"
                    ax.set_title(f"P{index} | {view}\n{text}", fontsize=10)
                    ax.set_xlim(lo[0], hi[0])
                    ax.set_ylim(hi[1], lo[1])
                    rows.append(
                        dict(
                            pool_index=index,
                            view=view,
                            single_mask_max_distance_px=single,
                            union_max_distance_px=union,
                            best_single_mask=best,
                            ties=ties,
                        )
                    )
            image_type = "lossless native PNG" if frame == 82720 else "lossy JPEG review context"
            fig.suptitle(
                f"Native{frame} | {image_type} | independent mask containment\n"
                "D2[index] is the original detector index, NOT a tracking or physical ID.\n"
                "Cyan open M/C/T + white six interior samples; orange + SLEAP. "
                "No support/identity certification or acceptance threshold.",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.86 if len(focus) == 1 else 0.91))
            stream = io.BytesIO()
            fig.savefig(stream, format="png", dpi=160)
        finally:
            registered.visual.prior.plt.close(fig)
        target = output / f"frame_{frame}.png"
        target.write_bytes(stream.getvalue())
        out.records[target.name] = dict(
            path=str(target), sha256=proof.sha(target), size_bytes=target.stat().st_size
        )
        displayed.append(dict(frame_idx=frame, photographs=photos, rows=rows))
    out.write(
        "summary.json",
        dict(
            schema=STAGE,
            status="ok",
            measurement_sha256=digest,
            displayed=displayed,
            all_original_mask_indices_preserved=True,
            distances_recomputed=False,
            physical_ID_truth=False,
            original_video_or_mask_archives_read=False,
        ),
    )
    evidence.final_guard()
    out.final_guard()
    (output / "render.provenance.json").write_bytes(
        proof.json_bytes(dict(stage=STAGE, status="ok", inputs=evidence.paths, outputs=out.records))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--measurement-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.measurement, args.measurement_sha, args.output)

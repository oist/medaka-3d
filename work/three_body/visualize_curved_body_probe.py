"""CPU-only native-raw/straight-body/saved-curve figures; no optimization or truth."""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import measure_persistent_silhouette as m
from curved_renderer import CurvedFishRenderer
from renderer import RendererConfig

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance

FRAMES = (79372, 79381, 82720)
LEVELS = (0.2, 0.5)


def crop_bounds(masks, projected_nodes, projected_curve, shape=(1200, 1200), margin=32):
    """Cover all D2 foreground and finite projected nodes/control-centreline samples."""
    points = np.concatenate((projected_nodes.reshape(-1, 2), projected_curve.reshape(-1, 2)))
    points = points[np.isfinite(points).all(axis=-1)]
    if len(masks):
        y, x = np.nonzero(masks.any(axis=0))
        if len(x):
            points = np.vstack((points, [[x.min(), y.min()], [x.max(), y.max()]]))
    if not len(points):
        raise ValueError("no finite display extent")
    lower = np.maximum(0, np.floor(points.min(axis=0)).astype(int) - margin)
    upper = np.minimum(np.array(shape[::-1]), np.ceil(points.max(axis=0)).astype(int) + margin + 1)
    if np.any(upper <= lower):
        raise ValueError("display crop does not intersect the source image")
    return int(lower[0]), int(lower[1]), int(upper[0]), int(upper[1])


def body_lines(pose, offset):
    t = np.linspace(0, 1, 33)
    control = (pose[:, 1] + pose[:, 2]) / 2 + offset
    tail = (
        (1 - t)[None, :, None] ** 2 * pose[:, 1, None]
        + 2 * t[None, :, None] * (1 - t)[None, :, None] * control[:, None]
        + t[None, :, None] ** 2 * pose[:, 2, None]
    )
    return np.concatenate((pose[:, :1], tail), axis=1)


def decorated(raw, masks, alpha, projected_line, projected_nodes, observed, review, rows, origin):
    image = raw.copy()
    xy0 = np.asarray(origin)
    for fish, color in enumerate(m.COLORS):
        for level, thickness in zip(LEVELS, (1, 2), strict=True):
            contours, _ = cv2.findContours(
                (alpha[..., fish] >= level).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)
        line = projected_line[fish] - xy0
        if np.isfinite(line).all():
            cv2.polylines(image, [np.rint(line).astype(np.int32)], False, color, 1, cv2.LINE_AA)
        nodes = projected_nodes[fish] - xy0
        for n, point in enumerate(nodes):
            if np.isfinite(point).all():
                cv2.circle(
                    image,
                    tuple(np.rint(point).astype(int)),
                    4 if n == 1 else 3,
                    color,
                    -1 if observed[fish, n] else 1,
                    cv2.LINE_AA,
                )
        if np.isfinite(nodes[1]).all():
            cv2.putText(
                image,
                f"S{fish}" + (" ?" if review[fish] else ""),
                tuple(np.rint(nodes[1] + [6, -7]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
    for i, mask in enumerate(masks):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, (255, 255, 0), 1)
        yy, xx = np.nonzero(mask)
        if len(xx):
            label = f"m{i} U={rows[i]['union_mean']:.2f}"
            anchor = (
                min(max(2, int(xx.min())), max(2, image.shape[1] - 115)),
                max(13, int(yy.min()) - 3),
            )
            # A dark outline leaves cyan small text legible over both water backgrounds.
            cv2.putText(
                image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (15, 15, 15), 3, cv2.LINE_AA
            )
            cv2.putText(
                image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA
            )
    return image


def cell(image, title, subtitle, width=560):
    canvas = np.full((width + 64, width, 3), 22, np.uint8)
    scale = min(width / image.shape[1], width / image.shape[0])
    shape = tuple(np.rint(np.array(image.shape[1::-1]) * scale).astype(int))
    scaled = cv2.resize(
        image, shape, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )
    x, y = (width - scaled.shape[1]) // 2, 64 + (width - scaled.shape[0]) // 2
    canvas[y : y + scaled.shape[0], x : x + scaled.shape[1]] = scaled
    cv2.putText(
        canvas, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 255), 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas, subtitle, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA
    )
    return canvas, dict(
        display_scale=scale, image_origin_xy=[x, y], native_shape_hw=image.shape[:2]
    )


def source_paths():
    return sorted(set(m.code_paths()) | {Path(__file__).resolve()})


def main():
    root = m.REPO / "work/three_body/curve_body_spatial_13frames_v1"
    output = root / "visual_report_v1"
    if output.exists():
        raise FileExistsError("visual report exists; no overwrite")
    codes = {str(path): m.sha256(path) for path in source_paths()}
    hashes = dict(codes)

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        m.require(path.is_relative_to(m.REPO), "input path escaped workspace")
        actual = m.sha256(path)
        m.require(expected is None or actual == expected, "visual input SHA mismatch")
        hashes[str(path)] = actual
        return path

    seal = json.loads(bind(root / "curve.provenance.json").read_text())
    m.require(
        seal["status"] == "ok" and seal["stage"] == "fixed_node_curve_spatial_diagnostic",
        "B run not completed",
    )
    m.require(len(seal["outputs"]) == 59, "unexpected B output contract")
    for item in seal["outputs"].values():
        bind(item["path"], item["sha256"])
    summary = json.loads((root / "summary.json").read_text())
    source = m.identity_inputs._verify_run(Path(summary["source_run"]), appearance=False)
    hashes.update(source.hashes)
    m.require(
        source.hashes[str(source.root / "candidate.npz")] == summary["candidate_sha256"],
        "wrong original candidate",
    )
    m.require(
        source.hashes[str(source.sequence / "manifest.json")] == summary["source_manifest_sha256"],
        "wrong source sequence",
    )
    for path in (
        m.LAB / "curved_renderer.py",
        m.LAB / "renderer.py",
        m.REPO / "src/medaka3d/calib.py",
    ):
        m.require(
            m.sha256(path) == summary["input_sha256_before"][str(path)],
            "render/calibration source differs from B",
        )
    rig = load_rig(source.root / "calibration_snapshot")
    packed = {
        v: m.identity_inputs._npz(
            m.base.resolve_input(source.sequence, source.manifest["masks"][v]["path"])
        )
        for v in m.VIEWS
    }
    raw_paths = {
        v: m.base.resolve_input(source.sequence, source.manifest["raw"][v]["path"]) for v in m.VIEWS
    }
    config = RendererConfig(**summary["renderer_config"])
    output.mkdir(parents=True, exist_ok=False)
    runtime_start, previous_threads = time.perf_counter(), torch.get_num_threads()
    torch.set_num_threads(2)
    results = []
    try:
        with provenance("saved_curve_body_native_visualization", output, name="visual") as rec:
            for path, digest in hashes.items():
                rec.input(path, path, sha256=digest)
            for path, digest in codes.items():
                target = output / "code_snapshot" / Path(path).relative_to(m.REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                m.require(m.sha256(target) == digest, "visual snapshot changed")
                rec.output(str(target.relative_to(output)), target, sha256=digest)
            for frame in FRAMES:
                t = frame - int(source.frames[0])
                info = next(f for f in summary["measurements"] if f["frame_idx"] == frame)
                saved = m.identity_inputs._npz(root / f"f{frame}_curve.npz")
                pose, offset = saved["pose_xyz"], saved["best_offset_mm"]
                lines = dict(
                    original=body_lines(pose, np.zeros((3, 3))), curve=body_lines(pose, offset)
                )
                captures = m.open_sources(raw_paths, len(source.frames), t)
                try:
                    raw = {
                        v: m.native_read(
                            captures[v], t, source.manifest["raw"][v]["frame_bgr_sha256"][t]
                        )
                        for v in m.VIEWS
                    }
                finally:
                    for cap in captures.values():
                        cap.release()
                masks, bounds, pixels, projected = {}, {}, {}, {}
                for view in m.VIEWS:
                    masks[view] = m.decode_frame_masks(packed[view], t)
                    m.require(
                        m.array_sha256(masks[view])
                        == packed[view]["frame_mask_sha256"][t].decode(),
                        "native D2 mask SHA differs",
                    )
                    camera = rig.camera(view)
                    projected[view] = dict(
                        nodes=camera.project(pose),
                        **{key: camera.project(value) for key, value in lines.items()},
                    )
                    bounds[view] = crop_bounds(
                        masks[view], projected[view]["nodes"], projected[view]["curve"]
                    )
                    x0, y0, x1, y1 = bounds[view]
                    yy, xx = np.mgrid[y0:y1, x0:x1]
                    pixels[view] = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
                engine = CurvedFishRenderer(rig, pixels, config, dtype=torch.float64, device="cpu")
                with torch.no_grad():
                    before = engine.render(pose, sigma_reference_pose_xyz=pose)
                    after = engine.render(
                        pose, tail_control_offset_mm=offset, sigma_reference_pose_xyz=pose
                    )
                for key in ("axial_sigma_mm", "radial_sigma_mm", "segment_lengths_mm"):
                    m.require(
                        torch.equal(getattr(before.bodies, key), getattr(after.bodies, key)),
                        "visualization changed scales",
                    )
                rows, records, arrays = (
                    [],
                    {},
                    dict(
                        pose_xyz=pose,
                        best_offset_mm=offset,
                        original_line_xyz=lines["original"],
                        curved_line_xyz=lines["curve"],
                    ),
                )
                for v, view in enumerate(m.VIEWS):
                    x0, y0, x1, y1 = bounds[view]
                    height, width = y1 - y0, x1 - x0
                    count = height * width
                    images = dict(raw=raw[view][y0:y1, x0:x1])
                    max_errors, contour_boundary = {}, {}
                    fg_uv = saved[view + "_foreground_uv"]
                    local = fg_uv.astype(int) - [x0, y0]
                    m.require(
                        np.all(local >= 0)
                        and np.all(local[:, 0] < width)
                        and np.all(local[:, 1] < height),
                        "foreground cropped from quantitative visualization",
                    )
                    for kind, rendered in (("original", before), ("curve", after)):
                        a = (
                            rendered.per_fish_alpha[0, v, :count]
                            .cpu()
                            .numpy()
                            .reshape(height, width, 3)
                        )
                        original_values = saved[view + "_" + kind + "_alpha"]
                        error = float(
                            np.max(np.abs(a[local[:, 1], local[:, 0]] - original_values), initial=0)
                        )
                        m.require(
                            error < 1e-12,
                            "CPU visualization differs from saved GPU foreground alpha",
                        )
                        max_errors[kind] = error
                        boundary = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
                        contour_boundary[kind] = np.any(boundary >= min(LEVELS), axis=0)
                        images[kind] = decorated(
                            images["raw"],
                            masks[view][:, y0:y1, x0:x1],
                            a,
                            projected[view][kind],
                            projected[view]["nodes"],
                            saved["observed"],
                            saved["prior_identity_review"],
                            info["per_instance"][view][kind]["instances"],
                            (x0, y0),
                        )
                        arrays[f"{view}_{kind}_alpha"] = a
                    panel_images, transforms = [], {}
                    for kind in ("raw", "original", "curve"):
                        if kind == "raw":
                            subtitle = f"source {frame}; crop [{x0}:{x1}, {y0}:{y1}]"
                        else:
                            score = info["spatial_evaluation_" + kind]["loss_by_view"][0][v]
                            subtitle = f"spatial eval error={score:.5f}; cyan m#: D2 mean U"
                        shown, transforms[kind] = cell(images[kind], f"{view} | {kind}", subtitle)
                        panel_images.append(shown)
                    rows.append(np.hstack(panel_images))
                    records[view] = dict(
                        crop_xyxy=bounds[view],
                        panel_transforms=transforms,
                        native_bgr_sha256=m.array_sha256(raw[view]),
                        original_source_offset=t,
                        raw_pos_before=t,
                        raw_pos_after=t + 1,
                        saved_gpu_alpha_max_abs_difference=max_errors,
                        display_contour_reaches_crop_boundary=contour_boundary,
                        gaussian_scale_equal=True,
                    )
                    arrays[view + "_crop_xyxy"] = np.asarray(bounds[view])
                    arrays[view + "_ray_valid"] = (
                        before.ray_valid[v, :count].cpu().numpy().reshape(height, width)
                    )
                body = np.vstack(rows)
                canvas = np.full((body.shape[0] + 77, body.shape[1], 3), 20, np.uint8)
                canvas[77:] = body
                cv2.putText(
                    canvas,
                    f"Frame {frame}: unchanged raw / original Gaussian body / saved curve-only fit",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    "Colored body-alpha contours: thin 0.2 / thick 0.5 (display levels only). "
                    "Filled nodes: direct; hollow: unsupported; ?: prior ID review.",
                    (12, 49),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    "Cyan: each D2 instance (not an animal ID). S0/S1/S2 colors arbitrary. "
                    "Same nodes and Gaussian scales; no refit, GT or review changes.",
                    (12, 69),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
                m.save_png(output / f"f{frame}_raw_original_curve.png", canvas, rec)
                path = output / f"f{frame}_visual_arrays.npz"
                np.savez_compressed(path, **arrays)
                rec.output(path.name, path, sha256=m.sha256(path))
                results.append(dict(frame_idx=frame, views=records, canvas_shape_hwc=canvas.shape))
                print(
                    json.dumps(
                        m.safe(
                            dict(
                                frame=frame,
                                crops=bounds,
                                alpha_max_error=max(
                                    max(r["saved_gpu_alpha_max_abs_difference"].values())
                                    for r in records.values()
                                ),
                                elapsed_s=time.perf_counter() - runtime_start,
                            )
                        )
                    ),
                    flush=True,
                )
            data = dict(
                status="completed_saved_curve_visualization",
                source_curve_run=str(root),
                source_curve_summary_sha256=m.sha256(root / "summary.json"),
                original_outputs_verified=59,
                native_raw_images_decoded=9,
                frames=FRAMES,
                views=m.VIEWS,
                measurements=results,
                contour_alpha_levels=LEVELS,
                contour_levels_are_display_only=True,
                renderer_config=dataclasses.asdict(config),
                device="cpu",
                dtype="float64",
                torch_threads=2,
                fitting_reperformed=False,
                gt_or_baseline_bytes_opened=False,
                manual_review_page_changed=False,
                limitation=(
                    "Amodal soft response contours are not measured physical boundaries "
                    "or ID truth; core/instance coverage is still imperfect."
                ),
                input_sha256_before=hashes,
                input_sha256_after=hashes,
                elapsed_s=time.perf_counter() - runtime_start,
            )
            path = output / "summary.json"
            path.write_text(json.dumps(m.safe(data), indent=2, allow_nan=False), encoding="utf-8")
            rec.output("summary", path, sha256=m.sha256(path))
            m.require(
                all(m.sha256(path) == digest for path, digest in hashes.items()),
                "visual source/input changed",
            )
            m.require(
                all(m.sha256(item["path"]) == item["sha256"] for item in rec.outputs.values()),
                "visual output/snapshot changed",
            )
            rec.param("inputs_and_code_unchanged", True)
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    main()

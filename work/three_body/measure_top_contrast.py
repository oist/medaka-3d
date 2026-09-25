"""Frozen top-contrast versus D2 observation diagnostic; no poses or ID truth."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import measure_persistent_silhouette as m
from observations import foreground_contrast

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance

FRAMES = m.FRAMES
VIEW = "top_cam"
LOWER = np.array([0.0, 0.0, -6.0])
UPPER = np.array([192.0, 192.0, 186.0])


def ray_box_intersection(origin, direction, lower, upper):
    """Inclusive forward-ray slab intersection; invalid is distinct from outside.

    Exact zero directions are parallel. Small nonzero directions are not silently
    made parallel. A zero vector is not a valid ray. This is not a visibility test.
    """
    origin, direction = np.asarray(origin, float), np.asarray(direction, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (
        origin.shape != direction.shape
        or origin.shape[-1:] != (3,)
        or lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite([lower, upper]).all()
        or np.any(lower >= upper)
    ):
        raise ValueError("invalid ray/box shape or bounds")
    valid = (
        np.isfinite(origin).all(axis=-1)
        & np.isfinite(direction).all(axis=-1)
        & np.any(direction != 0, axis=-1)
    )
    parallel = direction == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (lower - origin) / direction
        b = (upper - origin) / direction
    entry = np.where(parallel, -np.inf, np.minimum(a, b))
    leave = np.where(parallel, np.inf, np.maximum(a, b))
    parallel_outside = np.any(parallel & ((origin < lower) | (origin > upper)), axis=-1)
    hit = valid & ~parallel_outside & (leave.min(axis=-1) >= np.maximum(entry.max(axis=-1), 0))
    return hit, valid


def footprint(camera, shape=(1200, 1200), chunk=30000):
    flat = np.arange(np.prod(shape))
    physical, engineering, valid = (np.zeros(len(flat), bool) for _ in range(3))
    for start in range(0, len(flat), chunk):
        end = min(start + chunk, len(flat))
        y, x = np.unravel_index(flat[start:end], shape)
        origin, direction = camera.rays(np.column_stack((x, y)))
        engineering[start:end], valid[start:end] = ray_box_intersection(
            origin, direction, LOWER, UPPER
        )
        physical[start:end], _ = ray_box_intersection(origin, direction, [0, 0, 0], UPPER)
    return {
        "engineering": engineering.reshape(shape),
        "physical": physical.reshape(shape),
        "valid": valid.reshape(shape),
    }


def components(mask):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    rows = []
    height, width = mask.shape
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        rows.append(
            dict(
                component=index,
                area=area,
                bbox_xyxy=[x, y, x + w, y + h],
                centroid_xy=centroids[index].tolist(),
                image_boundary=(x == 0 or y == 0 or x + w == width or y + h == height),
            )
        )
    return labels, rows


def partition_counts(mask, box):
    return dict(
        total=int(mask.sum()),
        engineering_box=int((mask & box["engineering"]).sum()),
        physical_box=int((mask & box["physical"]).sum()),
        outside_engineering_box=int((mask & box["valid"] & ~box["engineering"]).sum()),
        invalid_ray=int((mask & ~box["valid"]).sum()),
    )


def measure(image, background, masks, box):
    contrast = foreground_contrast(image, background)
    diff = image.astype(np.float32) - background.astype(np.float32)
    median = np.median(diff, axis=(0, 1))
    diff -= median
    thresholded = np.max(np.abs(diff), axis=-1) >= 12
    union = masks.any(axis=0) if len(masks) else np.zeros(image.shape[:2], bool)
    labels, rows = components(contrast)
    # Reconstruct the frozen area gate for diagnostic bookkeeping only.
    raw_labels, raw_rows = components(thresholded)
    keep = np.array([False] + [row["area"] >= 6 for row in raw_rows])
    m.require(np.array_equal(contrast, keep[raw_labels]), "frozen contrast implementation differs")
    matrix = np.zeros((len(rows), len(masks)), np.int64)
    for i, mask in enumerate(masks):
        matrix[:, i] = np.bincount(labels[mask], minlength=len(rows) + 1)[1:]
    for i, row in enumerate(rows):
        mask = labels == i + 1
        row.update(partition_counts(mask, box))
        row["d2_union_overlap_pixels"] = int((mask & union).sum())
        row["d2_overlap_fraction"] = row["d2_union_overlap_pixels"] / row["area"]
        row["d2_instance_overlap_pixels"] = matrix[i].tolist()
    core = m.cores(masks)
    instances = []
    for i, mask in enumerate(masks):
        area = int(mask.sum())
        overlap = int((mask & contrast).sum())
        core_area = int(core[i].sum())
        instances.append(
            dict(
                instance=i,
                area=area,
                contrast_overlap_pixels=overlap,
                contrast_fraction=overlap / area if area else None,
                core_area=core_area,
                core_contrast_fraction=float(contrast[core[i]].mean()) if core_area else None,
                **{"partition": partition_counts(mask, box)},
            )
        )
    only_labels, only_rows = components(contrast & ~union)
    for row in only_rows:
        row.update(partition_counts(only_labels == row["component"], box))
    measurements = dict(
        median_channel_offset_bgr=median.tolist(),
        threshold_raw_pixels=int(thresholded.sum()),
        area_gate_removed_pixels=int((thresholded & ~contrast).sum()),
        raw_threshold_component_count=len(raw_rows),
        retained_contrast_component_count=len(rows),
        d2_instance_count=len(masks),
        contrast=partition_counts(contrast, box),
        d2_union=partition_counts(union, box),
        intersection=partition_counts(contrast & union, box),
        contrast_only=partition_counts(contrast & ~union, box),
        d2_only=partition_counts(union & ~contrast, box),
        positive_difference_pixels=int((contrast & (diff.max(axis=-1) >= 12)).sum()),
        negative_difference_pixels=int((contrast & (diff.min(axis=-1) <= -12)).sum()),
        instances=instances,
        contrast_components=rows,
        contrast_only_components=only_rows,
    )
    arrays = dict(
        contrast=contrast,
        d2_union=union,
        contrast_components=labels,
        contrast_only_components=only_labels,
        component_instance_overlap_pixels=matrix,
        median_channel_offset_bgr=median,
    )
    return measurements, arrays


def contours(image, mask, color, thickness=1):
    curves, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, curves, -1, color, thickness, cv2.LINE_AA)


def decorate(image, masks, contrast, box, kind):
    out = image.copy()
    if kind in ("d2", "combined"):
        for index, mask in enumerate(masks):
            contours(out, mask, (255, 255, 0), 1)
            yy, xx = np.nonzero(mask)
            if len(xx):
                cv2.putText(
                    out,
                    f"m{index}",
                    (max(0, int(xx.min())), max(15, int(yy.min()) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
    if kind in ("contrast", "combined"):
        contours(out, contrast, (255, 0, 255), 1)
    if kind != "raw":
        contours(out, box["engineering"], (0, 230, 255), 2)
    return out


def panel(image, title, detail, width=600):
    height = round(image.shape[0] * width / image.shape[1])
    out = np.full((height + 64, width, 3), 18, np.uint8)
    out[64:] = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA if width < image.shape[1] else cv2.INTER_NEAREST,
    )
    for y, text in ((23, title), (49, detail)):
        cv2.putText(
            out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA
        )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=m.REPO / "work/three_body/top_contrast_13frames_v1"
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    m.require(output.is_relative_to(m.REPO), "output outside workspace")
    if output.exists():
        raise FileExistsError("new output required")
    code_paths = sorted(set(m.code_paths()) | {Path(__file__).resolve()})
    codes = {str(path): m.sha256(path) for path in code_paths}
    source = m.identity_inputs._verify_run(
        m.REPO / "work/three_body/persistent_mouth_continuous_one_minute_v1", appearance=False
    )
    hashes = dict(source.hashes) | codes
    manifest = source.manifest
    m.require(source.frames.tolist() == list(range(76620, 86580)), "wrong native source range")
    m.require(
        manifest["fps"] == 166 and manifest["image_shape_hw"] == [1200, 1200],
        "wrong native metadata",
    )
    m.require(tuple(manifest["raw"]) == tuple(m.VIEWS), "source view order mismatch")
    m.require(
        m.sha256(m.LAB / "observations.py")
        == source.code["labs/20260909_three_body/observations.py"],
        "contrast code differs from persistent source",
    )
    staging_path = source.sequence / "staging.provenance.json"
    hashes[str(staging_path)] = m.sha256(staging_path)
    staging = json.loads(staging_path.read_text())
    m.require(staging["status"] == "ok", "incomplete raw/mask staging")
    m.require(
        staging["outputs"]["manifest"]["sha256"] == hashes[str(source.sequence / "manifest.json")],
        "unsealed manifest",
    )
    for view in m.VIEWS:
        for key, role in (("raw", "raw"), ("masks", "masks"), ("backgrounds", "background")):
            info = manifest[key][view]
            path = m.base.resolve_input(source.sequence, info["path"])
            item = staging["outputs"][f"{view}_{role}"]
            m.require(
                Path(item["path"]).resolve() == path.resolve()
                and item["sha256"] == info["sha256"] == hashes[str(path)],
                "staging asset binding differs",
            )
    rig = load_rig(source.root / "calibration_snapshot")
    m.require(tuple(rig.views) == tuple(m.VIEWS), "calibration view order differs")
    packed = m.identity_inputs._npz(
        m.base.resolve_input(source.sequence, manifest["masks"][VIEW]["path"])
    )
    m.require(np.array_equal(packed["frame_idx"], source.frames), "D2 and raw indices differ")
    background_path = m.base.resolve_input(source.sequence, manifest["backgrounds"][VIEW]["path"])
    background = cv2.imread(str(background_path))
    m.require(
        background is not None and background.shape == (1200, 1200, 3), "bad native background"
    )
    background_frames = manifest["backgrounds"][VIEW]["frames"]
    expected_background = [
        f for f in manifest["background_candidates"] if not 76620 - 332 <= f < 86580 + 332
    ]
    m.require(background_frames == expected_background, "registered background exclusion differs")
    paths = {
        view: m.base.resolve_input(source.sequence, manifest["raw"][view]["path"])
        for view in m.VIEWS
    }
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(2)
    try:
        with provenance(
            "frozen_top_contrast_d2_observation_diagnostic", output, name="contrast"
        ) as record:
            for path, digest in hashes.items():
                record.input(path, path, sha256=digest)
            for path, digest in codes.items():
                target = output / "code_snapshot" / Path(path).relative_to(m.REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                m.require(m.sha256(target) == digest, "snapshot mismatch")
                record.output(str(target.relative_to(output)), target, sha256=digest)
            box = footprint(rig.camera(VIEW))
            box_path = output / "ray_box_footprint.npz"
            np.savez_compressed(box_path, **box, lower=LOWER, upper=UPPER)
            record.output(box_path.name, box_path, sha256=m.sha256(box_path))
            m.save_png(output / "registered_background_top.png", background, record)
            rows, summary_rows = [], []
            captures = m.open_sources(paths, len(source.frames), 0)
            try:
                for frame in FRAMES:
                    t = frame - int(source.frames[0])
                    image = m.native_read(
                        captures[VIEW], t, manifest["raw"][VIEW]["frame_bgr_sha256"][t]
                    )
                    masks = m.decode_frame_masks(packed, t)
                    m.require(
                        m.array_sha256(masks) == packed["frame_mask_sha256"][t].decode(),
                        "D2 decoded mask SHA mismatch",
                    )
                    m.require(bool(packed["view_available"][t]), "selected D2 view unavailable")
                    stats, arrays = measure(image, background, masks, box)
                    stats.update(
                        frame_idx=frame,
                        native_offset=t,
                        pos_before=t,
                        pos_after=t + 1,
                        raw_bgr_sha256=m.array_sha256(image),
                        mask_available=True,
                        full_masks_sha256=m.array_sha256(masks),
                    )
                    a, b = (int(packed["frame_offsets"][t + k]) for k in (0, 1))
                    stats["d2_scores"] = packed["scores"][a:b].tolist()
                    # Display-only crop: native D2 bbox plus 100px. The full image
                    # above it and all reported counts retain every contrast pixel.
                    yy, xx = np.nonzero(arrays["d2_union"])
                    if len(xx):
                        x0, y0 = np.maximum(0, [xx.min() - 100, yy.min() - 100])
                        x1, y1 = np.minimum(1200, [xx.max() + 101, yy.max() + 101])
                    else:
                        x0, y0, x1, y1 = 0, 0, 1200, 1200
                    stats["detail_crop_xyxy"] = [int(v) for v in (x0, y0, x1, y1)]
                    shown, details = [], []
                    for kind, title in (
                        ("raw", "Raw; no model labels"),
                        ("d2", "D2 cyan / box yellow"),
                        ("contrast", "Contrast magenta / box yellow"),
                    ):
                        display = decorate(image, masks, arrays["contrast"], box, kind)
                        text = (
                            f"frame {frame}; D2 {len(masks)} masks / "
                            f"contrast {stats['retained_contrast_component_count']} components"
                        )
                        shown.append(panel(display, title, text))
                        details.append(
                            panel(
                                display[y0:y1, x0:x1],
                                title + " | detail",
                                f"native crop [{x0}:{x1}, {y0}:{y1}]",
                            )
                        )
                    m.save_png(output / f"f{frame}_full.png", np.hstack(shown), record)
                    m.save_png(output / f"f{frame}_detail.png", np.hstack(details), record)
                    array_path = output / f"f{frame}_observations.npz"
                    np.savez_compressed(array_path, frame_idx=np.array([frame], np.int64), **arrays)
                    record.output(array_path.name, array_path, sha256=m.sha256(array_path))
                    rows.append(stats)
                    row = dict(
                        frame=frame,
                        d2_instances=len(masks),
                        contrast_components=stats["retained_contrast_component_count"],
                    )
                    for key in ("contrast", "d2_union", "intersection", "contrast_only", "d2_only"):
                        row.update(
                            {key + "_" + category: value for category, value in stats[key].items()}
                        )
                    summary_rows.append(row)
                    print(
                        json.dumps(
                            dict(
                                frame=frame,
                                contrast=stats["contrast"],
                                d2=stats["d2_union"],
                                components=stats["retained_contrast_component_count"],
                                elapsed_s=time.perf_counter() - started,
                            )
                        ),
                        flush=True,
                    )
            finally:
                for capture in captures.values():
                    capture.release()
            csv_path = output / "counts.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
                writer.writeheader()
                writer.writerows(summary_rows)
            record.output(csv_path.name, csv_path, sha256=m.sha256(csv_path))
            summary = dict(
                status="completed_observation_only_diagnostic",
                frames=FRAMES,
                view=VIEW,
                source_sequence=str(source.sequence),
                source_manifest_sha256=hashes[str(source.sequence / "manifest.json")],
                background_frames=background_frames,
                background_file_sha256=m.sha256(background_path),
                background_bgr_sha256=m.array_sha256(background),
                original_threshold_and_component_gate_unchanged=True,
                threshold=12,
                threshold_polarity=(
                    "absolute difference, either sign, any BGR channel after "
                    "per-channel full-image median subtraction"
                ),
                connectivity=8,
                minimum_component_pixels=6,
                components_are_not_fish=True,
                box_lower=LOWER,
                box_upper=UPPER,
                physical_box_lower=[0, 0, 0],
                footprint_counts={key: int(mask.sum()) for key, mask in box.items()},
                box_is_diagnostic_only_not_mask_filter=True,
                box_limitation=(
                    "Infinite planar refraction plus ray/axis-box intersection does not model "
                    "reflection, occlusion or visibility; an intersecting ray is not proof "
                    "of a direct fish image."
                ),
                synchronization_scope=(
                    "Native frame indices, raw BGR SHA, packed-mask SHA and frame sequence "
                    "agree. This is not an independent hardware trigger or cross-camera "
                    "temporal-offset validation."
                ),
                background_rule=manifest["background_rule"],
                live_prediction_gt_or_id_arrays_read=False,
                native_top_images_decoded=13,
                raw_only_left_panels=True,
                detail_crop_rule=(
                    "D2 union bounding box plus100px, display only; full images "
                    "and complete quantitative domain retained"
                ),
                measurements=rows,
                code_sha256=codes,
                input_sha256_before=hashes,
                input_sha256_after=hashes,
                elapsed_s_excluding_source_hash_prepost=time.perf_counter() - started,
            )
            summary_path = output / "summary.json"
            summary_path.write_text(
                json.dumps(m.safe(summary), indent=2, allow_nan=False), encoding="utf-8"
            )
            record.output("summary", summary_path, sha256=m.sha256(summary_path))
            m.require(
                all(m.sha256(path) == digest for path, digest in hashes.items()),
                "source/input changed during diagnostic",
            )
            m.require(
                all(m.sha256(item["path"]) == item["sha256"] for item in record.outputs.values()),
                "output/snapshot changed during diagnostic",
            )
            record.param("input_code_output_sha_reverified", True)
    finally:
        cv2.setNumThreads(previous_threads)


if __name__ == "__main__":
    main()

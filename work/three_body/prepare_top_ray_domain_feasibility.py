"""Calibrated top half-ray/full-model-box intersection feasibility, CPU only.

The map is fixed before reading photographs or foreground masks. It is a
conservative model-domain intersection set, not physical visibility or fish
truth. Exterior pixels, invalid rays and point-only tangencies remain unknown.
No erosion, hand ROI, calibration-error padding, sampling change or fit occurs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import compare_top_background_foregrounds as previous
import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from medaka3d import calib

prep, require = previous.prep, previous.require
REPO, WORK = prep.REPO, prep.WORK
STAGE = "top_ray_domain_feasibility_v1"
SHAPE, BATCH = (1200, 1200), 4096
LOWER, UPPER = (0.0, 0.0, -6.0), (192.0, 192.0, 186.0)
LIVE_PLAN = WORK / "nodewise_body_row_five_minute_continuation_v1/plan.json"
LIVE_PLAN_SHA = "e05e174a291d6d79bf2cf3f74b848e7b125b3304f8ad1f0d168165dac3a8ba31"
FOREGROUND = WORK / "top_background_foreground_comparison_v1"
FOREGROUND_SHA = "38a9e1790c30ce906d8d0506a6aff4082e40ef1c70fa555b66651944bb331421"
PINS = {
    **previous.PINS,
    "work/three_body/compare_top_background_foregrounds.py": (
        "804766edb5b62294eb3a3fec55c01e997006c116ce9c65e9b93c32f1a0b9b51e"
    ),
    "src/medaka3d/calib.py": ("60849004cb53febe69f437e7afa8b18cb2b65213ba76ce7c921f7022d2be4274"),
}


def ray_box_intersection(origins, directions, lower, upper):
    """Closed box and t>=0 half-ray; positive interval length is required.

    Exact parallel axes use membership instead of division. A ray can start
    outside and enter later. No tolerance expands/contracts the physical box.
    Returns intersection bool, finite nonzero-ray bool, entry t and exit t.
    Parameters are in mm for unit input directions; validity is scale invariant.
    """
    o, d = np.asarray(origins), np.asarray(directions)
    lo, hi = np.asarray(lower), np.asarray(upper)
    require(
        o.ndim == 2
        and o.shape[1] == 3
        and d.shape == o.shape
        and o.dtype.kind in "fiu"
        and d.dtype.kind in "fiu",
        "matching real Nx3 rays required",
    )
    require(
        lo.shape == hi.shape == (3,)
        and lo.dtype.kind in "fiu"
        and hi.dtype.kind in "fiu"
        and np.isfinite(lo).all()
        and np.isfinite(hi).all()
        and (lo < hi).all(),
        "finite ordered box bounds required",
    )
    o, d, lo, hi = (np.asarray(x, np.float64) for x in (o, d, lo, hi))
    finite = np.isfinite(o).all(1) & np.isfinite(d).all(1) & np.any(d != 0, axis=1)
    safe_o, safe_d = np.where(finite[:, None], o, 0), np.where(finite[:, None], d, 0)
    parallel = safe_d == 0
    parallel_outside = (parallel & ((safe_o < lo) | (safe_o > hi))).any(1)
    a, b = np.full_like(o, -np.inf), np.full_like(o, np.inf)
    np.divide(lo - safe_o, safe_d, out=a, where=~parallel)
    np.divide(hi - safe_o, safe_d, out=b, where=~parallel)
    enter = np.maximum(0.0, np.minimum(a, b).max(1))
    leave = np.maximum(a, b).min(1)
    intersects = (
        finite & ~parallel_outside & np.isfinite(enter) & np.isfinite(leave) & (leave > enter)
    )
    return (
        intersects,
        finite,
        np.where(intersects, enter, np.nan),
        np.where(intersects, leave, np.nan),
    )


def domain_map(camera, lower=LOWER, upper=UPPER, shape=SHAPE, batch_size=BATCH):
    require(
        type(batch_size) is int
        and batch_size > 0
        and len(shape) == 2
        and all(type(x) is int and x > 0 for x in shape),
        "positive dimensions/batch required",
    )
    status = np.empty(int(np.prod(shape)), np.uint8)
    max_batch, calls = 0, 0
    for start in range(0, len(status), batch_size):
        flat = np.arange(start, min(start + batch_size, len(status)))
        y, x = np.unravel_index(flat, shape)
        # Native distorted pixel centres; no resize or +0.5 coordinate shift.
        uv = np.column_stack((x, y)).astype(np.float64)
        origins, directions = camera.rays(uv)
        require(
            np.shape(origins) == np.shape(directions) == (len(flat), 3), "camera ray shape differs"
        )
        hit, finite, _, _ = ray_box_intersection(origins, directions, lower, upper)
        status[flat] = np.where(hit, 1, np.where(finite, 0, 2))
        calls += 1
        max_batch = max(max_batch, len(flat))
    status = status.reshape(shape)
    return (
        status == 1,
        status,
        dict(
            ray_calls=calls,
            maximum_ray_batch=max_batch,
            model_intersection_pixels=int((status == 1).sum()),
            finite_nonintersection_pixels=int((status == 0).sum()),
            invalid_ray_pixels=int((status == 2).sum()),
            unknown_pixels=int((status != 1).sum()),
            total_native_pixels=int(status.size),
        ),
    )


def load_context(evidence):
    live = evidence.read(LIVE_PLAN, LIVE_PLAN_SHA)
    require(
        live["schema"] == "nodewise_body_row_five_minute_continuation_v1", "live plan stage differs"
    )
    ctx = live["tracker_context"]
    require(
        ctx["recovery_config"]["lower_mm"] == list(LOWER)
        and ctx["recovery_config"]["upper_mm"] == list(UPPER)
        and ctx["lower"] == prep.signature(np.asarray(LOWER, np.float64))
        and ctx["upper"] == prep.signature(np.asarray(UPPER, np.float64)),
        "existing full domain differs",
    )
    fg = evidence.read(FOREGROUND / "comparison.provenance.json", FOREGROUND_SHA)
    bg = evidence.read(previous.NEW / "preparation.provenance.json", previous.NEW_SHA)
    require(
        fg["status"] == bg["status"] == "ok"
        and fg["stage"] == FOREGROUND.name
        and bg["stage"] == previous.NEW.name,
        "successful saved image stages required",
    )
    fgplan = previous.document(FOREGROUND, fg, "plan.json", evidence)
    fgsummary = previous.document(FOREGROUND, fg, "summary.json", evidence)
    bgphotos = previous.document(previous.NEW, bg, "frame_evidence.json", evidence)["frames"]
    oldplan_path = previous.OLD / "plan.json"
    oldplan_digest = fg["inputs"][str(oldplan_path.resolve())]["sha256"]
    oldplan = evidence.read(oldplan_path, oldplan_digest)
    rig_refs = oldplan["registered_calibration"]
    shared = live["original_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"]
    require(
        len(rig_refs) == 4 and all(shared.get(p) == h for p, h in rig_refs.items()),
        "same registered rig required",
    )
    manifests = [Path(p) for p in rig_refs if Path(p).name == "calibration_manifest.json"]
    require(len(manifests) == 1, "unique calibration manifest required")
    docs = {p: evidence.read(p, h) for p, h in rig_refs.items()}
    manifest = docs[str(manifests[0])]
    require(
        manifest["backend"] == "refractive"
        and manifest["world_units"] == "mm"
        and manifest["view_order"] == ["top_cam", "side1_cam", "side2_cam"],
        "rig declaration differs",
    )
    for row in manifest["model_files"]:
        require(
            rig_refs.get(str(manifests[0].parent / row["path"])) == row["sha256"],
            "calibration bundle edge differs",
        )
    require(
        [r["frame_idx"] for r in bgphotos] == list(prep.FRAMES),
        "fixed nine pre-evaluation photos required",
    )
    for row in bgphotos:
        require(
            row["image"]
            == previous.member(previous.NEW, bg, f"native/frame_{row['frame_idx']}_top_cam.png"),
            "pre-evaluation photo edge differs",
        )
    photos = fgplan["inputs"]["photos"]
    require(
        fgplan["frames"] == fgsummary["frames"] == list(previous.FRAMES)
        and [r["frame_idx"] for r in photos] == list(previous.FRAMES),
        "fixed three scene frames required",
    )
    for row in photos:
        require(
            fg["inputs"].get(row["image"]["path"], {}).get("sha256") == row["image"]["sha256"],
            "comparison photo input edge differs",
        )
    rows = fgsummary["results"]
    require(
        [r["frame_idx"] for r in rows] == list(previous.FRAMES),
        "fixed foreground result rows required",
    )
    for row in rows:
        for mode in ("old", "new"):
            ref = previous.member(FOREGROUND, fg, f"masks/f{row['frame_idx']}_{mode}.png")
            require(row[mode + "_mask_png"] == ref, "saved binary foreground output differs")
    return dict(
        rig_directory=str(manifests[0].parent),
        rig_references=rig_refs,
        pre_evaluation_photos=bgphotos,
        scene_photos=photos,
        foregrounds=rows,
        nominal_geometry_reference=manifest["physical_geometry"],
        calibration_guardrail=manifest["guardrail"],
    )


def boundary_overlay(image, contours, label):
    panel = image.copy()
    cv2.drawContours(panel, contours, -1, (0, 255, 255), 2)
    out = np.zeros((image.shape[0] + 72, image.shape[1], 3), np.uint8)
    out[72:] = panel
    cv2.putText(out, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        out,
        "Yellow: model intersection boundary; outside unknown, not fish absence",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 255, 255),
        1,
    )
    return out


def run(output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and output != WORK and not output.exists(),
        "fresh workspace output required",
    )
    output.mkdir(parents=True)
    started, evidence, outputs = prep.utc(), prep.Evidence(), {}

    def write(name, value):
        data = (
            value
            if isinstance(value, bytes)
            else (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        )
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        outputs[name] = dict(path=str(path), sha256=prep.sha(data), size_bytes=len(data))
        return outputs[name]

    def png(name, array):
        okay, encoded = cv2.imencode(".png", array)
        require(okay, "PNG encoding failed")
        return write(name, encoded.tobytes())

    try:
        for relative, digest in PINS.items():
            evidence.capture(REPO / relative, digest)
        require(
            Path(calib.__file__).resolve() == REPO / "src/medaka3d/calib.py",
            "calibration import differs",
        )
        source = evidence.capture(Path(__file__))
        write("source_snapshot.py", source)
        context = load_context(evidence)
        cv2.setNumThreads(1)
        with threadpool_limits(limits=1):
            # Existing loader rereads only the four captured registered small JSONs.
            rig = calib.load_rig(context["rig_directory"])
            evidence.final_guard()
            valid, status, stats = domain_map(rig.camera("top_cam"))
        # Publish the fixed geometry before any image or saved FG is decoded.
        map_ref = png("model_domain_mask.png", valid.astype(np.uint8) * 255)
        status_ref = png("ray_status.png", status)
        write(
            "plan.json",
            dict(
                schema=STAGE,
                lower_mm=list(LOWER),
                upper_mm=list(UPPER),
                map_from_calibration_and_existing_domain_only=True,
                map_computed_before_image_decode=True,
                live_plan_sha256=LIVE_PLAN_SHA,
                live_plan_does_not_assert_final_tracking_success=True,
                foreground_comparison_sha256=FOREGROUND_SHA,
                background_photo_stage_sha256=previous.NEW_SHA,
                source_pins=PINS,
                input_bindings=context,
                map_signature=prep.signature(valid),
                status_signature=prep.signature(status),
                status_codes={
                    0: "finite ray without positive box interval",
                    1: "positive box interval",
                    2: "invalid/nonfinite/zero ray",
                },
                boundary_inclusive=True,
                tangency_with_zero_interval_is_unknown=True,
                outside_origin_forward_entries_allowed=True,
                ray_box_is_physical_visibility_proof=False,
                unknown_pixels_are_negative_evidence=False,
                boundary_padding_mm=0,
                source_pixels="native integer pixel centres; no half-pixel offset",
                CPU_threads=1,
                ray_batch_size=BATCH,
                map_output=map_ref,
                ray_status_output=status_ref,
            ),
        )
        contours, _ = cv2.findContours(
            valid.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        thumbnails = []
        for row in context["pre_evaluation_photos"]:
            image = previous.decode(row["image"], evidence, row["bgr_signature"])
            panel = boundary_overlay(
                image,
                contours,
                f"Native {row['frame_idx']} | {row['frame_idx'] / 166:.1f}s | fixed geometry",
            )
            thumbnails.append(cv2.resize(panel, (600, 636), interpolation=cv2.INTER_AREA))
        contact = np.concatenate(
            [np.concatenate(thumbnails[i : i + 3], axis=1) for i in (0, 3, 6)], axis=0
        )
        png("pre_evaluation_nine_photo_contact.png", contact)
        results = []
        for photo, row in zip(context["scene_photos"], context["foregrounds"], strict=True):
            frame = photo["frame_idx"]
            image = previous.decode(photo["image"], evidence, photo["bgr_signature"])
            png(
                f"frame_{frame}_boundary.png",
                boundary_overlay(image, contours, f"Native {frame} | model domain [-6,186] mm z"),
            )
            item = dict(frame_idx=frame, recording_seconds=frame / 166)
            for mode in ("old", "new"):
                ref = row[mode + "_mask_png"]
                encoded = evidence.capture(ref["path"], ref["sha256"])
                value = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
                require(
                    value is not None
                    and value.shape == SHAPE
                    and value.dtype == np.uint8
                    and np.isin(value, [0, 255]).all(),
                    "saved native binary PNG required",
                )
                foreground = value != 0
                require(
                    prep.signature(foreground) == row[mode]["mask_signature"],
                    "saved foreground byte signature differs",
                )
                inside, outside = foreground & valid, foreground & ~valid
                require(
                    int(foreground.sum()) == row[mode]["pixel_count"]
                    and int(inside.sum()) + int(outside.sum()) == int(foreground.sum()),
                    "foreground partition differs",
                )
                item[mode] = dict(
                    original_foreground_pixels=int(foreground.sum()),
                    inside_model_pixels=int(inside.sum()),
                    outside_unknown_pixels=int(outside.sum()),
                    invalid_ray_foreground_pixels=int((foreground & (status == 2)).sum()),
                    inside_signature=prep.signature(inside),
                    outside_signature=prep.signature(outside),
                    inside_png=png(
                        f"partitions/f{frame}_{mode}_inside.png", inside.astype(np.uint8) * 255
                    ),
                    outside_unknown_png=png(
                        f"partitions/f{frame}_{mode}_outside_unknown.png",
                        outside.astype(np.uint8) * 255,
                    ),
                )
            results.append(item)
        summary = dict(
            schema=STAGE,
            status="ok",
            domain_statistics=stats,
            results=results,
            pre_evaluation_photo_count=9,
            scene_photo_count=3,
            saved_foreground_png_count=6,
            model_domain_is_conservative_intersection_set=True,
            physical_visibility_proof=False,
            calibrated_boundaries_are_not_surveyed_pixel_truth=True,
            unknown_is_background=False,
            sampling_or_fit_changed=False,
            tracking_adoption=False,
            original_video_SL_D2_candidate_checkpoint_reads=0,
            GPU_calls=0,
        )
        write("summary.json", summary)
        evidence.final_guard()
        for ref in outputs.values():
            data = Path(ref["path"]).read_bytes()
            require(
                len(data) == ref["size_bytes"] and prep.sha(data) == ref["sha256"],
                "written output changed",
            )
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=prep.utc(),
            inputs=evidence.records,
            outputs=dict(outputs),
        )
    except BaseException as error:
        write(
            "preparation.provenance.json",
            dict(
                stage=STAGE,
                status="failed",
                started_at=started,
                finished_at=prep.utc(),
                inputs=evidence.records,
                outputs=dict(outputs),
                error_type=type(error).__name__,
                error=str(error),
            ),
        )
        raise
    write("preparation.provenance.json", seal)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

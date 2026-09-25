"""CPU diagnostic of sealed persistent geometry, without GT or identity changes."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path[:0] = [str(LAB), str(REPO / "src"), str(Path(__file__).parent)]
import identity_inputs  # noqa: E402
import measure_scene0_silhouette as old  # noqa: E402
import run_experiment as base  # noqa: E402
import run_persistent_node_recovery as persistent  # noqa: E402
from node_geometry import RayObservations, supported_nodes  # noqa: E402
from observations import CachedNodes, FrameObservation, sample_pixels  # noqa: E402
from prepare_sequence import VIEWS, array_sha256, decode_frame_masks, sha256  # noqa: E402
from render_results import COLORS  # noqa: E402
from renderer import RendererConfig  # noqa: E402
from review_scene import open_sources  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

FRAMES = (79316, 79318, 79322, 79355, 79370, 79372, 79381, 79394, 79396, 79405, 79547, 82719, 82720)
VISUAL_FRAMES = (79372, 82719)
require, safe = old.require, old.safe


def cores(masks):
    """Independent 5x5 ellipse erosion; empty cores remain explicitly empty."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return (
        np.stack(
            [
                cv2.erode(
                    m.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
                ).astype(bool)
                for m in masks
            ]
        )
        if len(masks)
        else masks.copy()
    )


def core_metrics(masks, flat, alpha, visible, valid):
    core = cores(masks)
    rows, _ = old.instance_rows(core, flat, alpha, visible, valid)
    union_core = core.any(axis=0) if len(core) else np.zeros(masks.shape[1:], bool)
    summary, _ = old.instance_rows(union_core[None], flat, alpha, visible, valid)
    return core, rows, summary[0]


def verify_support(pose, sources, observed, points, rig):
    for v, view in enumerate(VIEWS):
        for node in range(3):
            rows = sources[:, v, node]
            present = rows[rows >= 0]
            require(
                np.all(rows >= -1) and np.all(present < len(points[view])), "invalid source row"
            )
            require(len(np.unique(present)) == len(present), "shared SLEAP node source")
            require(np.isfinite(points[view][present, node]).all(), "nonfinite assigned SLEAP node")
    rays = RayObservations.from_points(rig, points)
    replay_sources, replay_observed, distance = supported_nodes(pose, sources, rays, rig)
    require(np.array_equal(sources, replay_sources), "saved sources fail original direct gates")
    require(np.array_equal(observed, replay_observed), "saved observed flags fail direct gates")
    return np.where(sources >= 0, distance, np.nan)


def native_read(capture, offset, expected_sha):
    require(capture.set(cv2.CAP_PROP_POS_FRAMES, offset), "raw seek failed")
    require(capture.get(cv2.CAP_PROP_POS_FRAMES) == offset, "raw position before read differs")
    ok, bgr = capture.read()
    require(
        ok and bgr is not None and bgr.shape == (1200, 1200, 3) and bgr.dtype == np.uint8,
        "native raw image invalid",
    )
    require(capture.get(cv2.CAP_PROP_POS_FRAMES) == offset + 1, "raw position after read differs")
    require(array_sha256(bgr) == expected_sha, "native BGR SHA differs")
    return bgr


def overlay(image, masks, uv, observed, review):
    image = image.copy()
    for index, mask in enumerate(masks):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, (255, 255, 0), 1)
        yy, xx = np.nonzero(mask)
        if len(xx):
            cv2.putText(
                image,
                f"m{index}",
                (int(xx.min()), max(15, int(yy.min()) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
    for fish in range(3):
        if not np.isfinite(uv[fish]).all():
            continue
        xy = np.rint(uv[fish]).astype(np.int32)
        cv2.polylines(image, [xy], False, COLORS[fish], 2, cv2.LINE_AA)
        for node in range(3):
            cv2.circle(
                image,
                tuple(xy[node]),
                5 if node == 1 else 3,
                COLORS[fish],
                -1 if observed[fish, node] else 1,
                cv2.LINE_AA,
            )
        cv2.putText(
            image,
            f"S{fish}" + (" ?" if review[fish] else ""),
            tuple(xy[1] + [8, -8]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLORS[fish],
            2,
            cv2.LINE_AA,
        )
    return image


def save_png(path, pixels, record):
    require(cv2.imwrite(str(path), pixels), "PNG write failed")
    require(np.array_equal(cv2.imread(str(path)), pixels), "PNG pixel round-trip differs")
    record.output(path.name, path, sha256=sha256(path))


def visualize(frame, images, masks, pose, observed, review, rig, config, output, record):
    # Explicit native 320-square regions: chosen fish pair only selects the view crop,
    # never the mathematical attribution domain (which remains all D2 foreground).
    pair = (0, 1) if frame == 79372 else (1, 2)
    pixels, offsets, projections = {}, {}, {}
    for view in VIEWS:
        uv = rig.camera(view).project(pose)
        projections[view] = uv
        center = uv[list(pair), 1].mean(axis=0)
        require(np.isfinite(center).all(), "invalid display center")
        x, y = np.clip(np.floor(center - 160).astype(int), 0, 880)
        offsets[view] = [int(x), int(y)]
        yy, xx = np.mgrid[y : y + 320, x : x + 320]
        pixels[view] = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
    rendered = old.render(rig, pixels, pose, config)
    context_raw, context_overlay, crops_raw, crops_overlay = [], [], [], []
    display_arrays = {}
    for v, view in enumerate(VIEWS):
        x, y = offsets[view]
        a = rendered["alpha"][v].reshape(320, 320, 3)
        valid = rendered["valid"][v].reshape(320, 320)
        raw = images[view]
        tinted = raw.copy()
        weight = 0.45 * a * valid[..., None]
        total = weight.sum(axis=-1)
        # Normalized illustrative color blend, not a physical color renderer.
        norm = np.maximum(1, total)[..., None]
        base_weight = 1 - np.minimum(1, total)
        region = raw[y : y + 320, x : x + 320].astype(float)
        region = region * base_weight[..., None] + np.einsum(
            "hwf,fc->hwc", weight / norm, np.asarray(COLORS, float)
        )
        tinted[y : y + 320, x : x + 320] = np.rint(np.clip(region, 0, 255)).astype(np.uint8)
        shown = overlay(tinted, masks[view], projections[view], observed, review)
        cv2.rectangle(shown, (x, y), (x + 319, y + 319), (190, 190, 190), 1)
        context_raw.append(cv2.resize(raw, (600, 600), interpolation=cv2.INTER_AREA))
        context_overlay.append(cv2.resize(shown, (600, 600), interpolation=cv2.INTER_AREA))
        crops_raw.append(raw[y : y + 320, x : x + 320])
        crops_overlay.append(shown[y : y + 320, x : x + 320])
        display_arrays[view + "_alpha"] = a
        display_arrays[view + "_ray_valid"] = valid
        display_arrays[view + "_origin_xy"] = np.array([x, y])
    for name, top, bottom, width in (
        ("context", context_raw, context_overlay, 600),
        ("native_crops", crops_raw, crops_overlay, 320),
    ):
        canvas = np.full((2 * width + 94, 3 * width, 3), 24, np.uint8)
        canvas[54 : 54 + width] = np.hstack(top)
        canvas[94 + width :] = np.hstack(bottom)
        cv2.putText(
            canvas,
            f"Persistent fixed geometry | frame {frame} | top / side1 / side2",
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "RAW above; cyan D2 below; filled=direct, hollow=latent, ?=prior ID review",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Body-alpha color only inside crop. S0/S1/S2 colors are arbitrary; not ID truth.",
            (10, width + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        save_png(output / f"f{frame}_{name}.png", canvas, record)
    path = output / f"f{frame}_display_alpha.npz"
    np.savez_compressed(path, **display_arrays)
    record.output(path.name, path, sha256=sha256(path))
    return dict(
        frame_idx=frame,
        native_crop_size=320,
        origins_xy=offsets,
        display_center_raw_slots=pair,
        per_view_pixel_count=320 * 320,
        alpha_color_blend_strength=0.45,
        native_raw_sha256={v: array_sha256(images[v]) for v in VIEWS},
    )


def code_paths():
    paths = {Path(__file__).resolve(), Path(old.__file__).resolve()}
    for module in tuple(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if value:
            path = Path(value).resolve()
            if (
                path.is_relative_to(LAB) or path.is_relative_to(REPO / "src")
            ) and path.suffix == ".py":
                paths.add(path)
    return sorted(paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=REPO / "work/three_body/persistent_mouth_continuous_one_minute_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(output.is_relative_to(REPO), "output must remain in workspace")
    if output.exists():
        raise FileExistsError("output exists; no overwrite")
    codes = {str(p): sha256(p) for p in code_paths()}
    print("Verifying sealed no-GT geometry inputs, raw/model code hashes...", flush=True)
    verified = identity_inputs._verify_run(args.run, appearance=False)
    hashes = {**verified.hashes, **codes}
    result = identity_inputs._npz(verified.root / "candidate.npz")
    require(set(result) == identity_inputs.GEOMETRY_ARRAYS, "expected persistent 21-array schema")
    persistent.validate_arm_arrays(result, verified.frames)
    require(np.array_equal(result["frame_idx"], np.arange(76620, 86580)), "wrong full minute")
    cfg = verified.params["tracker_config"]
    require(cfg["mode"] == "detectron2", "dummy shape-only pixels require D2 mode")
    for name in ("renderer.py", "observations.py", "prepare_sequence.py", "node_geometry.py"):
        key = (LAB / name).relative_to(REPO).as_posix()
        require(sha256(LAB / name) == verified.code[key], "live measurement differs from producer")
    rig = load_rig(verified.root / "calibration_snapshot")
    require(tuple(rig.views) == tuple(VIEWS), "unexpected camera order")
    nodes = CachedNodes(verified.input_root, verified.manifest["experiment"], verified.frames)
    packed = {
        v: identity_inputs._npz(
            base.resolve_input(verified.sequence, verified.manifest["masks"][v]["path"])
        )
        for v in VIEWS
    }
    for view in VIEWS:
        require(np.array_equal(packed[view]["frame_idx"], verified.frames), "mask frame mismatch")
    renderer_cfg = RendererConfig(width_ratio=cfg["width_ratio"])
    output.mkdir(parents=True, exist_ok=False)
    frames, csv_rows, visuals = [], [], []
    start = time.perf_counter()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with provenance(
            "persistent_fixed_geometry_instance_attribution", output, name="attribution"
        ) as record:
            for path, digest in hashes.items():
                record.input(path, path, sha256=digest)
            for path, digest in codes.items():
                destination = output / "code_snapshot" / Path(path).relative_to(REPO)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                require(sha256(destination) == digest, "snapshot copy differs")
                record.output(str(destination.relative_to(output)), destination, sha256=digest)
            for frame in FRAMES:
                t = frame - int(verified.frames[0])
                pose, observed, sources = (
                    result["pose_xyz"][t],
                    result["observed"][t],
                    result["sources"][t],
                )
                points = {v: nodes.at(v, frame) for v in VIEWS}
                distance = verify_support(pose, sources, observed, points, rig)
                masks, flat, pixels = {}, {}, {}
                for view in VIEWS:
                    data = packed[view]
                    masks[view] = decode_frame_masks(data, t)
                    require(
                        array_sha256(masks[view]) == data["frame_mask_sha256"][t].decode(),
                        "decoded mask SHA differs",
                    )
                    require(
                        data["view_available"][t] or not len(masks[view]),
                        "unavailable detector has instances",
                    )
                    union = (
                        masks[view].any(axis=0)
                        if len(masks[view])
                        else np.zeros((1200, 1200), bool)
                    )
                    flat[view] = np.flatnonzero(union)
                    yy, xx = np.unravel_index(flat[view], (1200, 1200))
                    pixels[view] = np.column_stack((xx, yy)).astype(float)
                dummy = np.zeros((1200, 1200, 3), np.uint8)
                obs = FrameObservation(
                    frame,
                    points,
                    dict.fromkeys(VIEWS, dummy),
                    {v: masks[v] if packed[v]["view_available"][t] else None for v in VIEWS},
                    {},
                )
                targets = sample_pixels(
                    obs,
                    "detectron2",
                    cfg["positive_samples"],
                    cfg["near_samples"],
                    cfg["global_samples"],
                )
                sample = old.render(rig, targets.pixels, pose, renderer_cfg)
                metrics = old.fixed_sample_metrics(
                    sample["alpha"],
                    sample["visible"],
                    sample["ownership"],
                    sample["valid"],
                    targets.target,
                    targets.weights,
                )
                metrics["saved_silhouette_loss"] = float(result["loss_parts"][t, 4])
                metrics["difference_from_saved_loss"] = (
                    metrics["mean_view_loss"] - metrics["saved_silhouette_loss"]
                )
                metrics["saved_mask_supported"] = result["mask_supported"][t]
                metrics["mask_supported_matches_saved"] = bool(
                    np.array_equal(metrics["original_mask_supported"], result["mask_supported"][t])
                )
                require(
                    abs(metrics["difference_from_saved_loss"]) < 1e-10, "fixed sampled loss differs"
                )
                require(metrics["mask_supported_matches_saved"], "original mask bridge differs")
                if any(len(p) for p in pixels.values()):
                    dense = old.render(rig, pixels, pose, renderer_cfg)
                else:
                    dense = dict(
                        alpha=np.empty((3, 0, 3)),
                        visible=np.empty((3, 0, 3)),
                        valid=np.empty((3, 0), bool),
                        conservation_error=0.0,
                    )
                arrays = dict(
                    frame_idx=np.asarray(frame),
                    pose_xyz=pose,
                    sources=sources,
                    observed=observed,
                    identity_review=result["identity_review"][t],
                    ray_distance_mm=distance,
                    sampled_uv=np.stack([targets.pixels[v] for v in VIEWS]),
                    target=targets.target,
                    weights=targets.weights,
                    foreground=targets.foreground,
                    sampled_alpha=sample["alpha"],
                    sampled_visible=sample["visible"],
                    sampled_ownership=sample["ownership"],
                    sampled_ray_valid=sample["valid"],
                )
                views = {}
                for v, view in enumerate(VIEWS):
                    n = len(flat[view])
                    a, vis, valid = (
                        dense["alpha"][v, :n],
                        dense["visible"][v, :n],
                        dense["valid"][v, :n],
                    )
                    rows, iou = old.instance_rows(masks[view], flat[view], a, vis, valid)
                    core, core_rows, core_union = core_metrics(
                        masks[view], flat[view], a, vis, valid
                    )
                    membership = old.node_membership(masks[view], rig.camera(view).project(pose))
                    core_membership = old.node_membership(core, rig.camera(view).project(pose))
                    lo, hi = packed[view]["frame_offsets"][t : t + 2]
                    require(hi - lo == len(rows), "instance offsets differ")
                    for i, row in enumerate(rows):
                        row.update(
                            detector_score=float(packed[view]["scores"][lo + i]),
                            node_mask_bilinear=membership["bilinear"][i],
                            node_mask_nearest_inside=membership["nearest_inside"][i],
                            core_node_mask_bilinear=core_membership["bilinear"][i],
                            core=core_rows[i],
                        )
                        item = dict(
                            frame_idx=frame,
                            view=view,
                            instance_index=i,
                            detector_score=row["detector_score"],
                        )
                        for prefix, value in (("", row), ("core_", core_rows[i])):
                            for key in (
                                "native_area_px",
                                "valid_ray_area_px",
                                "union_mean",
                                "unexplained_mean",
                                "unexplained_mass_px",
                            ):
                                item[prefix + key] = value[key]
                            for fish in range(3):
                                for key in (
                                    "mean_alpha",
                                    "mean_visible",
                                    "unique_mass_px",
                                    "visible_fraction_of_predicted_union",
                                ):
                                    item[f"{prefix}{key}_S{fish}"] = value[key][fish]
                            for k, pair in enumerate(old.PAIRS):
                                item[f"{prefix}shared_mean_S{pair[0]}_S{pair[1]}"] = value[
                                    "pair_shared_mean"
                                ][k]
                        for fish in range(3):
                            for node, name in enumerate(("mouth", "pectoral", "tail")):
                                item[f"S{fish}_{name}_bilinear"] = membership["bilinear"][
                                    i, fish, node
                                ]
                                item[f"S{fish}_{name}_direct"] = observed[fish, node]
                        csv_rows.append(item)
                    union, unique = old.union_and_unique(a)
                    views[view] = dict(
                        view_available=bool(packed[view]["view_available"][t]),
                        detector_instances=len(rows),
                        distinct_foreground_pixels=n,
                        valid_foreground_rays=int(valid.sum()),
                        foreground_likelihood_available=bool(
                            n and packed[view]["view_available"][t]
                        ),
                        instances=rows,
                        instance_iou=iou,
                        core_union_summary=core_union,
                        foreground_union_summary=dict(
                            union_mass_px=float(union[valid].sum()),
                            unexplained_mass_px=float((1 - union[valid]).sum()),
                            pair_shared_mass_px=[
                                float((a[valid, f] * a[valid, g]).sum()) for f, g in old.PAIRS
                            ],
                            unique_mass_px=unique[valid].sum(axis=0),
                        ),
                    )
                    arrays.update(
                        {
                            view + "_" + key: value
                            for key, value in dict(
                                foreground_flat=flat[view],
                                foreground_uv=pixels[view],
                                alpha=a,
                                visible=vis,
                                ray_valid=valid,
                                union=union,
                                node_membership=membership["bilinear"],
                                core_node_membership=core_membership["bilinear"],
                            ).items()
                        }
                    )
                path = output / f"f{frame}_responses.npz"
                np.savez_compressed(path, **arrays)
                record.output(path.name, path, sha256=sha256(path))
                path = output / f"f{frame}_attribution.png"
                old.heatmap(views, frame, path)
                require(cv2.imread(str(path)) is not None, "heatmap PNG decode failed")
                record.output(path.name, path, sha256=sha256(path))
                if frame in VISUAL_FRAMES:
                    raw_paths = {
                        v: base.resolve_input(
                            verified.sequence, verified.manifest["raw"][v]["path"]
                        )
                        for v in VIEWS
                    }
                    captures = open_sources(raw_paths, len(verified.frames), t)
                    try:
                        images = {
                            v: native_read(
                                captures[v], t, verified.manifest["raw"][v]["frame_bgr_sha256"][t]
                            )
                            for v in VIEWS
                        }
                    finally:
                        for cap in captures.values():
                            cap.release()
                    visuals.append(
                        visualize(
                            frame,
                            images,
                            masks,
                            pose,
                            observed,
                            result["identity_review"][t],
                            rig,
                            renderer_cfg,
                            output,
                            record,
                        )
                    )
                frames.append(
                    dict(
                        frame_idx=frame,
                        pose_xyz=pose,
                        observed=observed,
                        sources=sources,
                        prior_identity_review=result["identity_review"][t],
                        sampled_metrics=metrics,
                        all_saved_direct_gates_reproduced=True,
                        max_assigned_ray_distance_mm=float(np.nanmax(distance)),
                        full_foreground=views,
                        conservation_max_abs_error=max(
                            sample["conservation_error"], dense["conservation_error"]
                        ),
                    )
                )
                print(
                    json.dumps(
                        dict(
                            frame=frame,
                            loss_difference=metrics["difference_from_saved_loss"],
                            instances=sum(len(v["instances"]) for v in views.values()),
                            elapsed_s=time.perf_counter() - start,
                        )
                    ),
                    flush=True,
                )
            path = output / "instances.csv"
            with path.open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(safe(csv_rows))
            record.output("instances_csv", path, sha256=sha256(path))
            summary = dict(
                status="completed_diagnostic_not_identity_truth",
                frames=FRAMES,
                measurements=frames,
                visualizations=visuals,
                candidate_sha256=hashes[str(verified.root / "candidate.npz")],
                source_manifest_sha256=hashes[str(verified.sequence / "manifest.json")],
                algorithm_arm=verified.params["algorithm_arm"],
                renderer_config=dataclasses.asdict(renderer_cfg),
                sampling_config={
                    k: cfg[k] for k in ("positive_samples", "near_samples", "global_samples")
                },
                dummy_images=(
                    "Shape-only 1200-square zeros; D2 mode never inspects "
                    "image content or background."
                ),
                native_raw_video_decodes=6,
                gt_or_baseline_bytes_opened=False,
                identity_assignments_added=False,
                old_geometry_results_used=False,
                candidate_or_blind_review_changed=False,
                eroded_core=dict(
                    kernel="ellipse",
                    size=[5, 5],
                    border="constant zero",
                    purpose="diagnostic only, no loss/gate changes",
                ),
                formulas=dict(
                    union="1-prod(1-alpha)",
                    unexplained="sum_valid_mask(1-U)",
                    pair_shared=(
                        "sum_valid_mask(alpha_f*alpha_g): amodal overlap, "
                        "not visible overlap or animal count"
                    ),
                    unique="alpha_f*prod_g!=f(1-alpha_g)=U-U_without_f",
                    remove_one_body="L_without_f-L_all, existing fixed sampled image loss only",
                ),
                limitations=[
                    "Detector instance is not an animal identity; merged masks, "
                    "reflections and missed fish remain possible.",
                    "Overlapping instance masses are not additive; "
                    "foreground/core union is reported once separately.",
                    "Core erosion is a declared measurement, "
                    "not a validated true-fish core or new training target.",
                    "Empty core or invalid rays are unavailable measurements, "
                    "not proof of no fish.",
                    "Current union silhouette is symmetric in slots; "
                    "these diagnostics cannot establish IDs.",
                    "No optimization, new penalty, hard gate, support change or review clearance.",
                ],
                max_abs_saved_loss_difference=max(
                    abs(f["sampled_metrics"]["difference_from_saved_loss"]) for f in frames
                ),
                all_saved_mask_support_reproduced=True,
                all_saved_direct_support_reproduced=True,
                input_sha256_before=hashes,
                input_sha256_after=hashes,
                runtime=dict(
                    device="cpu",
                    dtype="float64",
                    torch_threads=2,
                    python=sys.version,
                    torch=str(torch.__version__),
                    numpy=np.__version__,
                    opencv=cv2.__version__,
                    diagnostic_s=time.perf_counter() - start,
                ),
            )
            path = output / "summary.json"
            path.write_text(json.dumps(safe(summary), indent=2, allow_nan=False), encoding="utf-8")
            record.output("summary", path, sha256=sha256(path))
            print("Checking final input/source/snapshot/output SHA seals...", flush=True)
            require(
                all(sha256(path) == digest for path, digest in hashes.items()),
                "input or live code changed",
            )
            require(
                all(sha256(item["path"]) == item["sha256"] for item in record.outputs.values()),
                "output or snapshot changed",
            )
            record.param("inputs_and_code_unchanged", True)
            record.param("gt_or_baseline_bytes_opened", False)
    finally:
        torch.set_num_threads(old_threads)


if __name__ == "__main__":
    main()

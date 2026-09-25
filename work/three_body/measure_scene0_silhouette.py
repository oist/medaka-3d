"""CPU-only fixed-pose attribution of existing union-silhouette image evidence."""

from __future__ import annotations

import csv
import dataclasses
import itertools
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
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(LAB))
from observations import FrameObservation, sample_pixels  # noqa: E402
from prepare_sequence import VIEWS, array_sha256, decode_frame_masks, sha256  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

RUN = REPO / "work/three_body/v1_continuous_one_minute_detectron2"
DIAGNOSTIC = REPO / "work/three_body/minute_scene0_model_diagnostic"
SEQUENCE = REPO / "work/three_body/inputs/continuous_one_minute_76620-86580"
OUT = REPO / "work/three_body/minute_scene0_silhouette_attribution"
FRAMES = [79316, 79318, 79322, 79355, 79370, 79372, 79381, 79405, 79547]
PAIRS = tuple(itertools.combinations(range(3), 2))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe(value):
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def union_and_unique(alpha):
    union = 1 - np.prod(1 - alpha, axis=-1)
    unique = np.stack(
        [
            alpha[..., fish] * np.prod(1 - alpha[..., [f for f in range(3) if f != fish]], axis=-1)
            for fish in range(3)
        ],
        axis=-1,
    )
    return union, unique


def fixed_sample_metrics(alpha, visible, ownership, valid, target, weights):
    union, unique = union_and_unique(alpha)
    weight = weights * valid
    error = (union - target) ** 2 * weight
    loss_by_view = error.sum(axis=1)
    positive = (visible * (ownership >= 0.65) * weight[..., None] * target[..., None]).sum(axis=1)
    negative = (visible * weight[..., None] * (1 - target)[..., None]).sum(axis=1)
    bridge_view = (positive >= 0.025) & (positive >= 2 * negative)
    delta = np.stack(
        [
            (((union - unique[..., fish] - target) ** 2) * weight).sum(axis=1) - loss_by_view
            for fish in range(3)
        ],
        axis=-1,
    )
    return dict(
        loss_by_view=loss_by_view,
        foreground_loss_by_view=(error * target).sum(axis=1),
        background_loss_by_view=(error * (1 - target)).sum(axis=1),
        mean_view_loss=float(loss_by_view.mean()),
        remove_body_delta_loss_by_view=delta,
        mean_view_remove_body_delta_loss=delta.mean(axis=0),
        owned_positive_by_view_fish=positive,
        negative_by_view_fish=negative,
        original_bridge_by_view_fish=bridge_view,
        original_mask_supported=(bridge_view.sum(axis=0) >= 2),
    )


def node_membership(masks, uv):
    count, height, width = masks.shape
    points = np.asarray(uv).reshape(-1, 2)
    valid = np.isfinite(points).all(axis=-1)
    valid &= (points[:, 0] >= 0) & (points[:, 0] <= width - 1)
    valid &= (points[:, 1] >= 0) & (points[:, 1] <= height - 1)
    values = np.full((count, len(points)), np.nan)
    nearest = np.zeros((count, len(points)), bool)
    if valid.any():
        x, y = points[valid].T
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        fx, fy = x - x0, y - y0
        values[:, valid] = (
            masks[:, y0, x0] * (1 - fx) * (1 - fy)
            + masks[:, y0, x1] * fx * (1 - fy)
            + masks[:, y1, x0] * (1 - fx) * fy
            + masks[:, y1, x1] * fx * fy
        )
        xi, yi = np.floor(x + 0.5).astype(int), np.floor(y + 0.5).astype(int)
        nearest[:, valid] = masks[:, yi, xi]
    return dict(
        bilinear=values.reshape(count, 3, 3),
        nearest_inside=nearest.reshape(count, 3, 3),
        node_inside_image=valid.reshape(3, 3),
    )


def instance_rows(masks, flat, alpha, visible, valid):
    union, unique = union_and_unique(alpha)
    rows = []
    areas = masks.sum(axis=(1, 2), dtype=np.int64)
    iou = np.zeros((len(masks), len(masks)))
    for i, j in itertools.product(range(len(masks)), repeat=2):
        intersection = np.count_nonzero(masks[i] & masks[j])
        total = int(areas[i] + areas[j] - intersection)
        iou[i, j] = intersection / total if total else np.nan
    for index, mask in enumerate(masks):
        selected = mask.reshape(-1)[flat] & valid
        count = int(selected.sum())
        row = dict(
            instance_index=index,
            native_area_px=int(areas[index]),
            valid_ray_area_px=count,
            invalid_ray_area_px=int(areas[index]) - count,
        )
        if not count:
            row.update(
                mean_alpha=[np.nan] * 3,
                mean_visible=[np.nan] * 3,
                union_mean=np.nan,
                unexplained_mass_px=np.nan,
                unexplained_mean=np.nan,
                pair_shared_mass_px=[np.nan] * 3,
                pair_shared_mean=[np.nan] * 3,
                unique_mass_px=[np.nan] * 3,
                unique_mean=[np.nan] * 3,
                visible_mass_px=[np.nan] * 3,
                visible_fraction_of_predicted_union=[np.nan] * 3,
            )
        else:
            a, v, u, marginal = (
                alpha[selected],
                visible[selected],
                union[selected],
                unique[selected],
            )
            shared = np.asarray([(a[:, i] * a[:, j]).sum() for i, j in PAIRS])
            union_mass = float(u.sum())
            row.update(
                mean_alpha=a.mean(axis=0),
                mean_visible=v.mean(axis=0),
                union_mean=float(u.mean()),
                unexplained_mass_px=float((1 - u).sum()),
                unexplained_mean=float((1 - u).mean()),
                pair_shared_mass_px=shared,
                pair_shared_mean=shared / count,
                unique_mass_px=marginal.sum(axis=0),
                unique_mean=marginal.mean(axis=0),
                visible_mass_px=v.sum(axis=0),
                visible_fraction_of_predicted_union=v.sum(axis=0) / union_mass
                if union_mass > 1e-12
                else np.full(3, np.nan),
            )
        rows.append(row)
    return rows, iou


def render(rig, pixels, pose, config):
    engine = RefractiveFishRenderer(rig, pixels, config, device="cpu", dtype=torch.float64)
    with torch.inference_mode():
        result = engine.render(pose)
    alpha = result.per_fish_alpha[0].numpy()
    visible = result.visible_alpha[0].numpy()
    union = result.union_alpha[0].numpy()
    ownership = result.ownership[0].numpy()
    valid = result.ray_valid.numpy()
    require(np.isfinite(alpha).all() and np.isfinite(visible).all(), "nonfinite renderer alpha")
    mass_error = float(np.max(np.abs(visible.sum(axis=-1) - union), initial=0))
    require(mass_error < 1e-12, "visible contributions do not sum to union")
    return dict(
        alpha=alpha,
        visible=visible,
        union=union,
        ownership=ownership,
        valid=valid,
        conservation_error=mass_error,
    )


def heatmap(rows, frame, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    image = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap="viridis")
    labels = ["a S0", "a S1", "a S2", "U", "1-U", "a0*a1", "a0*a2", "a1*a2"]
    for axis, view in zip(axes, VIEWS, strict=True):
        instances = rows[view]["instances"]
        values = np.asarray(
            [
                [
                    *item["mean_alpha"],
                    item["union_mean"],
                    item["unexplained_mean"],
                    *item["pair_shared_mean"],
                ]
                for item in instances
            ],
            float,
        )
        if not len(instances):
            axis.text(0.5, 0.5, "No detector instances", ha="center", va="center")
            axis.set_axis_off()
            continue
        image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
        axis.set_yticks(
            range(len(instances)),
            [f"mask {i['instance_index']} ({i['native_area_px']} px)" for i in instances],
        )
        axis.set_title(view)
        for i, j in itertools.product(range(len(instances)), range(len(labels))):
            value = values[i, j]
            axis.text(
                j,
                i,
                f"{value:.2f}" if np.isfinite(value) else "NA",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value < 0.6 else "black",
            )
    fig.colorbar(image, ax=axes, shrink=0.75, label="mean on valid detector-mask pixels")
    fig.suptitle(
        f"Source {frame}: fixed-pose mask attribution (detector instance is not a fish ID)"
    )
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    if OUT.exists():
        raise FileExistsError("attribution output exists; no overwrite")
    inputs = {}

    def verified(path, expected=None):
        path = Path(path).resolve(strict=True)
        require(path.is_relative_to(REPO), "input escaped workspace")
        digest = sha256(path)
        require(expected is None or digest == expected, f"changed input {path}")
        inputs[str(path)] = digest
        return load(path) if path.suffix == ".json" else path

    old_summary = verified(DIAGNOSTIC / "summary.json")
    old_seal = verified(DIAGNOSTIC / "diagnostic.provenance.json")
    require(
        old_seal["status"] == "ok"
        and old_seal["outputs"]["summary"]["sha256"] == inputs[str(DIAGNOSTIC / "summary.json")],
        "raw/candidate diagnostic is not sealed",
    )
    require(old_summary["selected_frames"] == FRAMES, "selected diagnostic frames differ")
    run_seal = verified(RUN / "experiment.provenance.json")
    require(run_seal["status"] == "ok", "candidate run is not complete")
    run_summary = verified(RUN / "summary.json", run_seal["outputs"]["summary"]["sha256"])
    candidate = verified(RUN / "candidate.npz", run_seal["outputs"]["candidate"]["sha256"])
    require(
        inputs[str(candidate)] == old_summary["candidate_sha256"],
        "candidate differs from previous visual diagnostic",
    )
    with np.load(candidate, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    require(
        len(result) == 19 and np.array_equal(result["frame_idx"], np.arange(76620, 86580)),
        "unexpected candidate minute/schema",
    )
    manifest = verified(SEQUENCE / "manifest.json", old_summary["source_manifest_sha256"])
    staging = verified(SEQUENCE / "staging.provenance.json")
    require(
        staging["status"] == "ok"
        and staging["outputs"]["manifest"]["sha256"] == old_summary["source_manifest_sha256"],
        "mask sequence not sealed",
    )
    packed = {}
    for view in VIEWS:
        info = manifest["masks"][view]
        require(
            info["sha256"]
            == run_seal["inputs"]["masks_" + view]["sha256"]
            == staging["outputs"][view + "_masks"]["sha256"],
            "D2 evidence differs across source seals",
        )
        path = verified(SEQUENCE / info["path"], info["sha256"])
        with np.load(path, allow_pickle=False) as arrays:
            packed[view] = {key: arrays[key] for key in arrays.files}
        require(
            np.array_equal(packed[view]["frame_idx"], result["frame_idx"]),
            "mask native frames differ",
        )
    calibration = RUN / "calibration_snapshot"
    for path in calibration.iterdir():
        matches = [
            item
            for key, item in run_seal["inputs"].items()
            if key.startswith("calibration/") and Path(item["path"]).name == path.name
        ]
        require(len(matches) == 1, "calibration input binding missing")
        verified(path, matches[0]["sha256"])
    rig = load_rig(calibration)
    require(tuple(rig.views) == tuple(VIEWS), "renderer view order differs")
    code = [
        Path(__file__),
        LAB / "renderer.py",
        LAB / "observations.py",
        LAB / "prepare_sequence.py",
        REPO / "src/medaka3d/calib.py",
        REPO / "src/medaka3d/types.py",
        REPO / "src/medaka3d/provenance.py",
    ]
    registered = {
        name.replace("\\", "/"): digest
        for name, digest in run_summary["code_sha256_before"].items()
    }
    for path in code:
        verified(
            path, None if path == Path(__file__) else registered[path.relative_to(REPO).as_posix()]
        )
    cfg = run_summary["tracker_config"]
    require(
        cfg["mode"] == "detectron2",
        "dummy shape images are allowed only for content-independent D2 sampling",
    )
    renderer_config = RendererConfig(width_ratio=cfg["width_ratio"])
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=False)
    frame_results, csv_rows = [], []
    started = time.perf_counter()
    with provenance("fixed_pose_scene0_silhouette_attribution", OUT, name="attribution") as record:
        for path, digest in inputs.items():
            record.input(path, path, sha256=digest)
        for path in code:
            target = OUT / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            record.output("code:" + path.name, target, sha256=sha256(target))
        for frame in FRAMES:
            t = frame - 76620
            pose = result["pose_xyz"][t]
            masks, fg_flat, full_pixels = {}, {}, {}
            for view in VIEWS:
                data = packed[view]
                decoded = decode_frame_masks(data, t)
                require(
                    array_sha256(decoded) == data["frame_mask_sha256"][t].decode(),
                    "decoded instance mask checksum differs",
                )
                masks[view] = decoded if data["view_available"][t] else None
                require(
                    data["view_available"][t] or not len(decoded),
                    "unavailable view contains detections",
                )
                union = np.any(decoded, axis=0) if len(decoded) else np.zeros((1200, 1200), bool)
                fg_flat[view] = np.flatnonzero(union)
                y, x = np.unravel_index(fg_flat[view], (1200, 1200))
                full_pixels[view] = np.column_stack((x, y)).astype(float)
            dummy = np.zeros((1200, 1200, 3), np.uint8)
            observation = FrameObservation(frame, {}, dict.fromkeys(VIEWS, dummy), masks, {})
            targets = sample_pixels(
                observation,
                "detectron2",
                cfg["positive_samples"],
                cfg["near_samples"],
                cfg["global_samples"],
            )
            sampled = render(rig, targets.pixels, pose, renderer_config)
            metrics = fixed_sample_metrics(
                sampled["alpha"],
                sampled["visible"],
                sampled["ownership"],
                sampled["valid"],
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
            if any(len(values) for values in full_pixels.values()):
                dense = render(rig, full_pixels, pose, renderer_config)
            else:
                dense = dict(
                    alpha=np.empty((3, 0, 3)),
                    visible=np.empty((3, 0, 3)),
                    union=np.empty((3, 0)),
                    ownership=np.empty((3, 0, 3)),
                    valid=np.empty((3, 0), bool),
                    conservation_error=0.0,
                )
            arrays = dict(
                frame_idx=np.asarray(frame),
                sampled_uv=np.stack([targets.pixels[v] for v in VIEWS]),
                target=targets.target,
                weights=targets.weights,
                foreground=targets.foreground,
                sampled_alpha=sampled["alpha"],
                sampled_visible=sampled["visible"],
                sampled_ownership=sampled["ownership"],
                sampled_ray_valid=sampled["valid"],
                pose_xyz=pose,
            )
            by_view = {}
            for v, view in enumerate(VIEWS):
                count = len(full_pixels[view])
                mask = masks[view] if masks[view] is not None else np.empty((0, 1200, 1200), bool)
                alpha, visible = dense["alpha"][v, :count], dense["visible"][v, :count]
                valid = dense["valid"][v, :count]
                rows, iou = instance_rows(mask, fg_flat[view], alpha, visible, valid)
                membership = node_membership(mask, rig.camera(view).project(pose))
                data = packed[view]
                first, stop = data["frame_offsets"][t : t + 2]
                for i, row in enumerate(rows):
                    row["detector_score"] = float(data["scores"][first + i])
                    row["node_mask_bilinear"] = membership["bilinear"][i]
                    row["node_mask_nearest_inside"] = membership["nearest_inside"][i]
                    flat_row = dict(
                        frame_idx=frame,
                        view=view,
                        instance_index=i,
                        detector_score=row["detector_score"],
                        native_area_px=row["native_area_px"],
                        valid_ray_area_px=row["valid_ray_area_px"],
                        unexplained_mass_px=row["unexplained_mass_px"],
                        unexplained_mean=row["unexplained_mean"],
                        union_mean=row["union_mean"],
                    )
                    for f in range(3):
                        for key in (
                            "mean_alpha",
                            "mean_visible",
                            "unique_mass_px",
                            "unique_mean",
                            "visible_fraction_of_predicted_union",
                        ):
                            flat_row[f"{key}_S{f}"] = row[key][f]
                    for pair, value in zip(PAIRS, row["pair_shared_mean"], strict=True):
                        flat_row[f"pair_shared_mean_S{pair[0]}_S{pair[1]}"] = value
                    csv_rows.append(flat_row)
                union, unique = union_and_unique(alpha)
                by_view[view] = dict(
                    view_available=bool(data["view_available"][t]),
                    detector_instances=int(stop - first),
                    distinct_foreground_pixels=count,
                    valid_foreground_rays=int(valid.sum()),
                    foreground_likelihood_available=bool(count and data["view_available"][t]),
                    instance_iou=iou,
                    node_inside_image=membership["node_inside_image"],
                    instances=rows,
                    foreground_union_summary=dict(
                        unexplained_mass_px=float((1 - union[valid]).sum()),
                        union_mass_px=float(union[valid].sum()),
                        pair_shared_mass_px=[
                            float((alpha[valid, a] * alpha[valid, b]).sum()) for a, b in PAIRS
                        ],
                        unique_mass_px=unique[valid].sum(axis=0),
                    ),
                )
                arrays.update(
                    {
                        f"{view}_{key}": value
                        for key, value in dict(
                            foreground_uv=full_pixels[view],
                            foreground_flat=fg_flat[view],
                            alpha=alpha,
                            union=union,
                            visible=visible,
                            ownership=dense["ownership"][v, :count],
                            ray_valid=valid,
                            node_membership=membership["bilinear"],
                        ).items()
                    }
                )
            path = OUT / f"f{frame}_responses.npz"
            np.savez_compressed(path, **arrays)
            record.output(path.name, path, sha256=sha256(path))
            path = OUT / f"f{frame}_attribution.png"
            heatmap(by_view, frame, path)
            require(cv2.imread(str(path)) is not None, "heatmap PNG decode failed")
            record.output(path.name, path, sha256=sha256(path))
            frame_results.append(
                dict(
                    frame_idx=frame,
                    sampled_metrics=metrics,
                    full_foreground=by_view,
                    conservation_max_abs_error=max(
                        sampled["conservation_error"], dense["conservation_error"]
                    ),
                )
            )
            print(
                json.dumps(
                    dict(
                        frame=frame,
                        loss_difference=metrics["difference_from_saved_loss"],
                        foreground_pixels=sum(len(p) for p in full_pixels.values()),
                        elapsed_s=time.perf_counter() - started,
                    )
                ),
                flush=True,
            )
        path = OUT / "instances.csv"
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(safe(csv_rows))
        record.output("instances_csv", path, sha256=sha256(path))
        require(
            all(sha256(path) == digest for path, digest in inputs.items()),
            "inputs or frozen code changed during measurement",
        )
        summary = dict(
            status="complete_diagnostic_not_identity_ground_truth",
            frames=FRAMES,
            measurements=frame_results,
            candidate_sha256=inputs[str(candidate)],
            renderer_config=dataclasses.asdict(renderer_config),
            device="cpu",
            dtype="float64",
            torch_threads=2,
            dummy_images=(
                "native 1200-square zero images supply shape only; "
                "verified D2-mode sample_pixels reads no RGB content/background"
            ),
            native_raw_video_decodes=0,
            mask_instances_are_not_assumed_to_be_animals=True,
            formulas=dict(
                union="1-product_f(1-alpha_f)",
                uncovered="sum_mask_valid(1-union)",
                pair_shared=(
                    "sum_mask_valid(alpha_f*alpha_g); amodal overlap, not proof of duplicate fish"
                ),
                unique="sum_mask_valid(alpha_f*product_g!=f(1-alpha_g)) = sum(U-U_without_f)",
                removal_loss_delta=(
                    "original fixed-sample L_without_f-L_all; "
                    "negative means better image-only fit without that body"
                ),
                visible="existing depth-ordered visible_alpha, ties averaged; sum_f visible=union",
                node_membership=(
                    "native bilinear mask value plus nearest pixel inclusion; "
                    "invalid/out-of-image nodes explicitly separate"
                ),
            ),
            original_bridge_thresholds=dict(
                ownership=0.65, owned_positive=0.025, positive_to_negative=2, views=2
            ),
            thresholds_added=False,
            identity_assignments_added=False,
            candidate_or_blind_review_changed=False,
            overlapping_detector_instance_metrics_are_not_additive=True,
            max_abs_saved_loss_difference=max(
                abs(item["sampled_metrics"]["difference_from_saved_loss"]) for item in frame_results
            ),
            all_original_bridge_flags_reproduced=all(
                item["sampled_metrics"]["mask_supported_matches_saved"] for item in frame_results
            ),
            input_sha256_before=inputs,
            input_sha256_after=inputs,
            runtime=dict(
                python=sys.version,
                torch=str(torch.__version__),
                numpy=np.__version__,
                opencv=cv2.__version__,
                elapsed_s=time.perf_counter() - started,
            ),
            limitations=[
                "Soft mask mass is not a count of animals or probability of identity.",
                "Unexplained detector pixels can be real body parts, reflections or detector "
                "errors; this measurement does not decide which.",
                "Per-fish overlap can be physical occlusion; ownership reflects the existing "
                "approximate renderer.",
                "No optimization, tracking, new hard gate, or identity relabeling was performed.",
                "Full-foreground metrics exclude invalid rays; sampled image metrics also "
                "include the existing weighted background samples.",
            ],
        )
        path = OUT / "summary.json"
        path.write_text(json.dumps(safe(summary), indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=sha256(path))


if __name__ == "__main__":
    main()

"""Frozen-pose CPU diagnosis of two opened development support regressions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))

import evaluate_appearance_minute as evaluation  # noqa: E402
import evaluate_variant_minute as persistent_evaluation  # noqa: E402
import run_experiment as base  # noqa: E402
from appearance_ranking import sample_appearance_pixels  # noqa: E402
from node_geometry import RayObservations, supported_nodes  # noqa: E402
from observations import FrameObservation, SequenceReader, sample_pixels  # noqa: E402
from pixel_appearance_loss import PixelAppearanceConfig, pixel_appearance_loss  # noqa: E402
from prepare_sequence import decode_frame_masks  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

FRAMES = np.arange(76620, 77286, dtype=np.int64)
PROBES = np.r_[np.arange(77066, 77071), np.arange(77106, 77111)]
EVENTS = ((77068, 1), (77108, 2))


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_run(root, appearance=False):
    root = evaluation.local_path(root)
    record = evaluation.load_json(root / "experiment.provenance.json")
    if record.get("status") != "ok":
        raise ValueError("read completed and sealed runs only")
    summary = evaluation.load_json(root / "summary.json")
    verifier = (
        evaluation.verify_appearance_evidence
        if appearance
        else persistent_evaluation.verify_variant_evidence
    )
    hashes = verifier(root, summary, expected_frames=FRAMES)
    with np.load(root / "candidate.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    validate = (
        evaluation.validate_appearance_arrays
        if appearance
        else persistent_evaluation.validate_variant_arrays
    )
    return validate(arrays, FRAMES, summary), summary, hashes


def selected_frame(reader, frame_idx, captures):
    index = int(frame_idx - reader.frames[0])
    images, masks = {}, {}
    for view, capture in captures.items():
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise ValueError("lossless clip cannot seek to registered native frame")
        ok, image = capture.read()
        if not ok or capture.get(cv2.CAP_PROP_POS_FRAMES) != index + 1:
            raise ValueError("native frame seek/decode alignment mismatch")
        digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        if digest != reader.pixel_hashes[view][index]:
            raise ValueError("native pixel SHA differs from inference data")
        images[view] = image
        masks[view] = (
            decode_frame_masks(reader.masks[view], index)
            if reader.masks[view]["view_available"][index]
            else None
        )
    return FrameObservation(
        int(frame_idx),
        {v: reader.nodes.at(v, frame_idx) for v in base.VIEWS},
        images,
        masks,
        reader.backgrounds,
    )


def ray_tables(rig, frame, poses, payload):
    tables = {}
    rays = RayObservations.from_points(rig, frame.detections)
    for vi, view in enumerate(base.VIEWS):
        points = frame.detections[view]
        origin = rays.origins[vi].astype(np.float32)
        direction = rays.directions[vi].astype(np.float32)
        # batch,fish,node,row,xyz. Preserve the original assigned_rays float32 storage.
        delta = poses[:, :, :, None] - origin.transpose(1, 0, 2)[None, None]
        d = direction.transpose(1, 0, 2)[None, None]
        ray_mm = np.linalg.norm(delta - np.sum(delta * d, axis=-1, keepdims=True) * d, axis=-1)
        projected = rig.camera(view).project(poses)
        pixel_error = np.linalg.norm(
            projected[:, :, :, None] - points.transpose(1, 0, 2)[None, None], axis=-1
        )
        valid = (
            np.isfinite(points).all(-1).T[None, None]
            & (ray_mm <= 2.0)
            & np.isfinite(pixel_error)
            & (pixel_error <= 12.0)
        )
        tables[view] = (ray_mm, pixel_error, valid)
        prefix = f"f{frame.frame_idx}_{view}"
        payload.update(
            {
                prefix + "_sleap_xy": points,
                prefix + "_projected_xy": projected,
                prefix + "_ray_mm_all_rows": ray_mm,
                prefix + "_pixel_error_all_rows": pixel_error,
                prefix + "_supports_all_rows": valid,
            }
        )
    return rays, tables


def image_scores(rig, frame, poses, config, enrollment, payload):
    pose = torch.as_tensor(poses, dtype=torch.float64)
    targets = sample_pixels(
        frame,
        config["mode"],
        config["positive_samples"],
        config["near_samples"],
        config["global_samples"],
    )
    renderer = RefractiveFishRenderer(
        rig,
        targets.pixels,
        RendererConfig(width_ratio=config["width_ratio"]),
        device="cpu",
        dtype=torch.float64,
    )
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)
    with torch.no_grad():
        rendering = renderer.render(pose)
        losses = (
            ((rendering.union_alpha - tensor(targets.target)) ** 2)
            * tensor(targets.weights)
            * rendering.ray_valid
        ).sum(-1)
        silhouette = losses.sum(-1) / 3
    enabled = enrollment["enabled_views"] & np.array([True, False, False])
    color_pixels = sample_appearance_pixels(frame, enabled, 512)
    color_renderer = RefractiveFishRenderer(
        rig,
        color_pixels.pixels,
        RendererConfig(width_ratio=config["width_ratio"]),
        device="cpu",
        dtype=torch.float64,
    )
    with torch.no_grad():
        colored = color_renderer.render(pose)
        weights = tensor(color_pixels.weights) * colored.ray_valid
        result = pixel_appearance_loss(
            colored.visible_alpha,
            tensor(color_pixels.features),
            tensor(enrollment["templates"]),
            weights,
            torch.as_tensor(color_pixels.enabled),
            PixelAppearanceConfig(),
        )
        visible_mass = (colored.visible_alpha * weights[None, :, :, None]).sum(2)
    prefix = f"f{frame.frame_idx}"
    payload.update(
        {
            prefix + "_silhouette_pixels": renderer.pixels,
            prefix + "_silhouette_targets": targets.target,
            prefix + "_silhouette_weights": targets.weights,
            prefix + "_color_pixels": color_renderer.pixels,
            prefix + "_color_features": color_pixels.features,
            prefix + "_color_weights": weights.numpy(),
            prefix + "_visible_alpha": colored.visible_alpha.numpy(),
        }
    )
    return dict(
        silhouette_loss_by_view=losses.numpy(),
        silhouette_loss=silhouette.numpy(),
        appearance_loss=result.loss.numpy(),
        appearance_loss_by_view=result.per_view_loss.numpy(),
        visible_color_mass_by_view_and_slot=visible_mass.numpy(),
        color_scored_pixels_by_view=(weights > 0).sum(1).numpy(),
    )


def code_paths():
    return [
        Path(__file__),
        *(
            LAB / name
            for name in (
                "evaluate_appearance_minute.py",
                "evaluate_variant_minute.py",
                "evaluate_minute.py",
                "run_experiment.py",
                "run_node_recovery.py",
                "run_persistent_node_recovery.py",
                "run_appearance_tracking.py",
                "appearance_diagnostic.py",
                "appearance_ranking.py",
                "pixel_template_assignment.py",
                "pixel_appearance_loss.py",
                "observations.py",
                "prepare_sequence.py",
                "renderer.py",
                "node_geometry.py",
                "diagnostics.py",
                "review_events.py",
            )
        ),
        *(
            REPO / "src/medaka3d" / name
            for name in (
                "calib.py",
                "types.py",
                "provenance.py",
                "eval/generative.py",
                "eval/reacquisition.py",
            )
        ),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--persistent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    codes = {str(path): base.sha256(path) for path in code_paths()}
    current, summary, hashes = load_run(args.run, appearance=True)
    reference, ref_summary, ref_hashes = load_run(args.persistent)
    parity = evaluation.same_v1_inputs(
        args.run, summary, args.persistent, ref_summary, hashes, ref_hashes
    )
    evaluation.validate_registered_arm(summary)
    for name in ("node_recovery_config", "persistence_config"):
        if not evaluation.exact_json(summary[name], ref_summary[name]):
            raise ValueError("recovery/persistence configuration differs")
    record = evaluation.load_json(args.run / "experiment.provenance.json")
    reader = SequenceReader(
        evaluation.local_path(record["params"]["sequence"]),
        evaluation.local_path(record["params"]["input_root"]),
    )
    enrollment_audit, enrollment_hashes = evaluation.verify_enrollment(
        args.run, summary, current, FRAMES, nodes=reader.nodes
    )
    hashes.update({**ref_hashes, **enrollment_hashes})
    runtime_code = evaluation.code_hashes(summary["code_sha256_before"])
    for name in (
        "renderer.py",
        "node_geometry.py",
        "observations.py",
        "prepare_sequence.py",
        "appearance_ranking.py",
        "pixel_template_assignment.py",
        "pixel_appearance_loss.py",
    ):
        relative = (LAB / name).relative_to(REPO).as_posix()
        if codes[str(LAB / name)] != runtime_code[relative]:
            raise ValueError("CPU diagnostic uses changed rendering/observation source")
    with np.load(args.run / "appearance_enrollment.npz", allow_pickle=False) as saved:
        enrollment = {key: saved[key] for key in saved.files}
    initial, _, _ = base.registered_seed(int(FRAMES[0]))
    seed_lengths = np.linalg.norm(np.diff(initial, axis=1), axis=-1)
    rig = base.load_rig(
        evaluation.local_path(record["params"]["input_root"]) / base.CALIBRATION_DIR
    )
    captures = {view: cv2.VideoCapture(str(path)) for view, path in reader.videos.items()}
    payload, frames = {}, []
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for frame_idx in PROBES:
            index = int(frame_idx - FRAMES[0])
            frame = selected_frame(reader, frame_idx, captures)
            poses = np.stack((reference["pose_xyz"][index], current["pose_xyz"][index]))
            sources = np.stack((reference["sources"][index], current["sources"][index]))
            rays, tables = ray_tables(rig, frame, poses, payload)
            for arm, result in enumerate((reference, current)):
                supported, observed, _ = supported_nodes(poses[arm], sources[arm], rays, rig)
                if not np.array_equal(supported, sources[arm]) or not np.array_equal(
                    observed, result["observed"][index]
                ):
                    raise ValueError("saved supported source fails original gates")
            table = []
            for vi, view in enumerate(base.VIEWS):
                distances, errors, good = tables[view]
                for fish, node in np.ndindex(3, 3):
                    rows = sorted(set(sources[:, fish, vi, node]) - {-1})
                    for row in rows:
                        table.append(
                            dict(
                                view=view,
                                slot=fish,
                                node=node,
                                row=int(row),
                                retained_in_each_arm=sources[:, fish, vi, node] == row,
                                ray_mm=distances[:, fish, node, row],
                                pixel_error=errors[:, fish, node, row],
                                passes_support_gate=good[:, fish, node, row],
                                owning_slot_each_arm=[
                                    np.flatnonzero(sources[a, :, vi, node] == row).tolist()
                                    for a in range(2)
                                ],
                            )
                        )
            image = image_scores(rig, frame, poses, summary["tracker_config"], enrollment, payload)
            lengths = np.linalg.norm(np.diff(poses, axis=2), axis=-1)
            entry = dict(
                frame_idx=int(frame_idx),
                pose_xyz=poses,
                sources=sources,
                observed=np.stack((reference["observed"][index], current["observed"][index])),
                identity_review=np.stack(
                    (reference["identity_review"][index], current["identity_review"][index])
                ),
                pose_difference_norm_mm=np.linalg.norm(poses[1] - poses[0], axis=-1),
                segment_lengths_mm=lengths,
                segment_to_seed_ratio=lengths / seed_lengths,
                saved_loss_parts=np.stack(
                    (reference["loss_parts"][index], current["loss_parts"][index])
                ),
                saved_appearance_cost=float(current["appearance_cost"][index]),
                all_retained_row_comparisons=table,
                images=image,
                silhouette_replay_absolute_error=np.abs(
                    image["silhouette_loss"]
                    - np.array([reference["loss_parts"][index, 4], current["loss_parts"][index, 4]])
                ),
                appearance_replay_absolute_error=abs(
                    image["appearance_loss"][1] - current["appearance_cost"][index]
                ),
            )
            frames.append(entry)
            print(json.dumps({"diagnosed_frame": int(frame_idx)}), flush=True)
    finally:
        for capture in captures.values():
            capture.release()
        torch.set_num_threads(previous_threads)
    losses = []
    for frame_idx, fish in EVENTS:
        item = next(row for row in frames if row["frame_idx"] == frame_idx)
        losses.append(
            dict(
                frame_idx=frame_idx,
                slot=fish,
                observed_before_after=item["observed"][:, fish, 1],
                pectoral_difference_mm=item["pose_difference_norm_mm"][fish, 1],
                old_current_pectoral_sources=item["sources"][:, fish, :, 1],
                pectoral_row_comparisons=[
                    row
                    for row in item["all_retained_row_comparisons"]
                    if row["slot"] == fish and row["node"] == 1
                ],
            )
        )
    report = clean(
        dict(
            status="sealed_saved_pose_diagnostic",
            arms=["persistent", "appearance"],
            probes=PROBES,
            events=losses,
            frames=frames,
            input_parity=parity,
            enrollment_record_audit=enrollment_audit,
            source_ray_storage="float32 before float64 point-line distance",
            native_pixel_gate=12.0,
            ray_gate_mm=2.0,
            cpu_float64_rendering=True,
            torch_threads_during_rendering=1,
            gpu_used=False,
            inference_or_refitting_performed=False,
            identity_truth_used=False,
            fixed_shape_color_comparison_uses_current_frozen_templates=True,
            unsupported_assignment_plans_not_saved=True,
            limitation=(
                "Evaluating old retained rows against new poses is a diagnostic counterfactual; "
                "it does not establish which discarded plans the tracker considered."
            ),
        )
    )
    frozen = {**codes, **hashes}
    if any(base.sha256(path) != value for path, value in frozen.items()):
        raise ValueError("source/input changed during diagnostic")
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance(
        "appearance_interval_support_diagnostic", args.output, name="diagnostic"
    ) as rec:
        rec.params_from(args)
        rec.param("torch_version", str(torch.__version__))
        rec.param("opencv_version", cv2.__version__)
        rec.param("device", "cpu")
        for path, value in frozen.items():
            rec.input(path, path, sha256=value)
        for source, value in codes.items():
            path = args.output / "code_snapshot" / Path(source).relative_to(REPO)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, path)
            if base.sha256(path) != value:
                raise ValueError("source changed during copy")
            rec.output(str(path.relative_to(args.output)), path, sha256=value)
        path = args.output / "summary.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        rec.output("summary", path, sha256=base.sha256(path))
        path = args.output / "native_evidence.npz"
        np.savez_compressed(path, **payload)
        rec.output("native_evidence", path, sha256=base.sha256(path))
        if any(base.sha256(path) != value for path, value in frozen.items()):
            raise ValueError("source/input changed while saving diagnostic")
    print(json.dumps(report["events"], indent=2))
    return report


if __name__ == "__main__":
    main()

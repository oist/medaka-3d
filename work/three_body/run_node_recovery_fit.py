"""Four saved-scene local mouth fits; no temporal continuation or identity inference."""

import dataclasses
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from accelerated_tracker import AcceleratedThreeBodyTracker  # noqa: E402
from graph_fit import FitInputs, objective  # noqa: E402
from node_geometry import (  # noqa: E402
    RayObservations,
    assigned_rays,
    assignment_plans,
    supported_nodes,
)
from node_recovery import NodeRecoveryConfig, node_recovery_proposals  # noqa: E402
from observations import CachedNodes, FrameObservation, sample_pixels  # noqa: E402
from prepare_sequence import decode_frame_masks  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402
from tracker import Proposal, State, ThreeBodyTracker, TrackerConfig  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

RUN = ROOT / "work/three_body/v1_continuous_one_minute_detectron2"
SEQUENCE = ROOT / "work/three_body/inputs/continuous_one_minute_76620-86580"
RAW = ROOT / "work/three_body/minute_scene1_mouth_diagnosis_v2"
OUT = ROOT / "work/three_body/node_recovery_local_fit_v1"
FRAMES = (80261, 80431, 80344, 80370)
PARTS = ("node", "shape", "pectoral_motion", "repulsion", "silhouette")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def score_initial(tracker, proposals, state, rays, targets):
    """Use the frozen public objective with exactly the fitter's input preparation."""
    tensor, cfg = tracker._tensor, tracker.config
    origin, direction, held = assigned_rays(rays, np.stack([p.plan for p in proposals]))
    weight_node = tensor([1.0, 1.0, 0.35])[None, None, None]
    missing = tensor(np.stack([state.missing for _ in proposals]))
    lengths = tensor(tracker.lengths)[None]
    renderer = RefractiveFishRenderer(
        tracker.rig,
        targets.pixels,
        RendererConfig(width_ratio=cfg.width_ratio),
        device=tracker.device,
        dtype=tracker.dtype,
    )
    inputs = FitInputs(
        tensor(origin),
        tensor(direction),
        tensor(held),
        tensor(np.stack([p.predicted for p in proposals])),
        cfg.motion_sigma_mm + 0.3 * torch.clamp(missing, max=20),
        lengths,
        torch.clamp(0.2 * lengths, min=1.0),
        weight_node,
        tensor(targets.target),
        tensor(targets.weights) * renderer.ray_valid,
    )
    with torch.no_grad():
        score, parts, _ = objective(
            tensor(np.stack([p.pose for p in proposals])),
            cfg,
            inputs,
            9 * float(weight_node.sum()),
            renderer,
        )
    return score.cpu().numpy(), parts.cpu().numpy()


def support_record(pose, plan, rays, rig):
    sources, observed, distances = supported_nodes(pose, plan, rays, rig)
    for view in range(3):
        for node in range(3):
            rows = sources[:, view, node]
            rows = rows[rows >= 0]
            assert len(rows) == len(np.unique(rows))
    return dict(
        observed=observed,
        filtered_sources=sources,
        ray_distance_mm=distances,
        per_node_supported_views=(sources >= 0).sum(axis=1),
    )


def draw_comparison(rig, images, base, selected, base_support, selected_support, frame, label):
    """Two rows, identical per-view native crops, with all fish and direct node status."""
    colors = ((255, 110, 0), (0, 180, 255), (255, 40, 210))
    native = {}
    panels = [[], []]
    for view in rig.views:
        uv = [rig.camera(view).project(pose) for pose in (base, selected)]
        finite = np.concatenate([p.reshape(-1, 2) for p in uv])
        finite = finite[np.isfinite(finite).all(axis=-1)]
        low = np.maximum(np.floor(finite.min(0) - 65).astype(int), 0)
        high = np.minimum(np.ceil(finite.max(0) + 65).astype(int), [1200, 1200])
        x0, y0 = low
        x1, y1 = high
        native[view] = dict(crop_xyxy=[x0, y0, x1, y1], base_uv=uv[0], selected_uv=uv[1])
        for row, support in enumerate((base_support, selected_support)):
            canvas = images[view][y0:y1, x0:x1].copy()
            for fish in range(3):
                pts = uv[row][fish] - low
                for a, b in ((0, 1), (1, 2)):
                    if np.isfinite(pts[[a, b]]).all():
                        cv2.line(
                            canvas,
                            tuple(np.rint(pts[a]).astype(int)),
                            tuple(np.rint(pts[b]).astype(int)),
                            colors[fish],
                            1,
                        )
                for node, point in enumerate(pts):
                    if not np.isfinite(point).all():
                        continue
                    point = tuple(np.rint(point).astype(int))
                    if support[fish, node]:
                        cv2.circle(canvas, point, 4, colors[fish], 1)
                    else:
                        cv2.drawMarker(canvas, point, colors[fish], cv2.MARKER_TILTED_CROSS, 9, 1)
                if np.isfinite(pts[1]).all():
                    cv2.putText(
                        canvas,
                        f"S{fish}",
                        tuple(np.rint(pts[1] + 5).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        colors[fish],
                        1,
                    )
            scale = min(590 / canvas.shape[1], 460 / canvas.shape[0])
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            panel = np.full((500, 600, 3), 32, np.uint8)
            panel[40 : 40 + canvas.shape[0], : canvas.shape[1]] = canvas
            text = f"{frame} {view} {'base fit' if row == 0 else label}"
            cv2.putText(panel, text, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            panels[row].append(panel)
    return np.vstack([np.hstack(row) for row in panels]), native


def main():
    OUT.mkdir(exist_ok=False)
    run_seal = json.loads((RUN / "experiment.provenance.json").read_text())
    raw_seal = json.loads((RAW / "diagnosis.provenance.json").read_text())
    assert run_seal["status"] == raw_seal["status"] == "ok"
    candidate, summary_path = RUN / "candidate.npz", RUN / "summary.json"
    assert sha256(candidate) == run_seal["outputs"]["candidate"]["sha256"]
    assert sha256(summary_path) == run_seal["outputs"]["summary"]["sha256"]
    run_summary = json.loads(summary_path.read_text())
    cfg = TrackerConfig(**run_summary["tracker_config"])
    assert cfg.opt_steps == 14 and cfg.mode == "detectron2"
    manifest_path = SEQUENCE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert sha256(manifest_path) == sha256(RUN / "sequence_manifest.json")
    with np.load(candidate, allow_pickle=False) as arrays:
        frames = arrays["frame_idx"]
        # Only the four explicitly registered CURRENT saved scenes enter these fits.
        indices = np.array([np.flatnonzero(frames == f).item() for f in FRAMES])
        scenes = arrays["pose_xyz"][indices].copy()
        missing = arrays["missing"][indices].copy()
        review = arrays["identity_review"][indices].copy()
        saved_support = arrays["observed"][indices].copy()
    initial_path = RUN / "initial_pose_xyz.npy"
    initial = np.load(initial_path, allow_pickle=False)
    assert np.array_equal(initial, np.asarray(run_summary["seed"]["pose_xyz"]))
    rig = load_rig(RUN / "calibration_snapshot")
    nodes = CachedNodes(
        ROOT / "work/generative_tracking_inputs",
        manifest["experiment"],
        np.arange(min(FRAMES), max(FRAMES) + 1),
    )
    sources = [
        Path(__file__),
        *(
            LAB / name
            for name in (
                "node_recovery.py",
                "node_geometry.py",
                "tracker.py",
                "accelerated_tracker.py",
                "graph_fit.py",
                "graph_union.py",
                "fast_union.py",
                "renderer.py",
                "observations.py",
                "prepare_sequence.py",
            )
        ),
        ROOT / "src/medaka3d/calib.py",
    ]
    paths = [
        candidate,
        summary_path,
        initial_path,
        RUN / "experiment.provenance.json",
        manifest_path,
        RUN / "sequence_manifest.json",
        RAW / "diagnosis.provenance.json",
        *(RUN / "calibration_snapshot").glob("*.json"),
        *nodes.paths.values(),
        *sources,
    ]
    masks, backgrounds = {}, {}
    for view in rig.views:
        for kind in ("masks", "backgrounds"):
            entry = manifest[kind][view]
            path = SEQUENCE / entry["path"]
            assert sha256(path) == entry["sha256"]
            paths.append(path)
            if kind == "masks":
                with np.load(path, allow_pickle=False) as arrays:
                    masks[view] = {key: arrays[key] for key in arrays.files}
            else:
                backgrounds[view] = cv2.imread(str(path))
    for frame in FRAMES:
        for view in rig.views:
            path = RAW / f"raw_f{frame}_{view}.png"
            assert sha256(path) == raw_seal["outputs"][path.name]["sha256"]
            paths.append(path)
    hashes = {str(path): sha256(path) for path in paths}
    helper_cfg = NodeRecoveryConfig()
    saved_arrays = {}
    with provenance("four_saved_scene_mouth_local_fit", OUT, name="fit") as record:
        record.param("tracker_config", dataclasses.asdict(cfg))
        record.param("node_recovery_config", dataclasses.asdict(helper_cfg))
        record.param("frames", FRAMES)
        record.param("local_comparator", "base scene re-fitted using the same current evidence")
        record.param("state", "current saved scene; zero velocity; saved missing and review")
        record.param("no_tracking_or_identity_inference", True)
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        tracker = AcceleratedThreeBodyTracker(rig, initial, cfg)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        results, parity = [], None
        for index, frame in enumerate(FRAMES):
            native_index = manifest["frames"].index(frame)
            images, frame_masks, raw_checks = {}, {}, {}
            for view in rig.views:
                image = cv2.imread(str(RAW / f"raw_f{frame}_{view}.png"))
                assert image is not None and image.shape == (1200, 1200, 3)
                image_sha = array_sha(image)
                assert image_sha == manifest["raw"][view]["frame_bgr_sha256"][native_index]
                images[view] = image
                assert masks[view]["frame_idx"][native_index] == frame
                decoded = decode_frame_masks(masks[view], native_index)
                assert array_sha(decoded) == masks[view]["frame_mask_sha256"][native_index].decode()
                frame_masks[view] = decoded if masks[view]["view_available"][native_index] else None
                raw_checks[view] = dict(
                    native_bgr_sha256=image_sha, decoded_mask_sha256=array_sha(decoded)
                )
            observation = FrameObservation(
                frame, {v: nodes.at(v, frame) for v in rig.views}, images, frame_masks, backgrounds
            )
            targets = sample_pixels(
                observation, cfg.mode, cfg.positive_samples, cfg.near_samples, cfg.global_samples
            )
            rays = RayObservations.from_points(rig, observation.detections)
            scene = scenes[index].copy()
            plan = assignment_plans(scene, rays, keep=1)[0][1]
            state = State(
                scene.copy(), np.zeros((3, 3)), missing[index].copy(), review[index].copy(), 0.0
            )
            state_copies = {
                name: getattr(state, name).copy()
                for name in ("pose", "velocity", "missing", "review")
            }
            helper = node_recovery_proposals(scene, plan, rays, rig, tracker.lengths, helper_cfg)
            proposals = [Proposal(scene.copy(), scene.copy(), plan.copy(), 0, "predict")]
            proposals += [
                Proposal(p.pose_xyz.copy(), scene.copy(), p.plan.copy(), 0, "node_mouth_recovery")
                for p in helper.proposals
            ]
            proposal_copies = [
                (p.pose.copy(), p.predicted.copy(), p.plan.copy()) for p in proposals
            ]
            start_score, start_parts = score_initial(tracker, proposals, state, rays, targets)
            torch.cuda.synchronize()
            started = time.perf_counter()
            fitted, scores, parts, mask_support = tracker._fit(proposals, [state], rays, targets)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if index == 0:
                batch24 = [proposals[i % len(proposals)] for i in range(24)]
                reference_start = time.perf_counter()
                reference = ThreeBodyTracker._fit(tracker, batch24, [state], rays, targets)
                torch.cuda.synchronize()
                reference_seconds = time.perf_counter() - reference_start
                accelerated_start = time.perf_counter()
                accelerated = tracker._fit(batch24, [state], rays, targets)
                torch.cuda.synchronize()
                accelerated_seconds = time.perf_counter() - accelerated_start
                comparisons = {}
                for name, a, b in zip(
                    ("poses", "scores", "parts", "mask_support"),
                    reference,
                    accelerated,
                    strict=True,
                ):
                    comparisons[name] = dict(
                        shape=a.shape,
                        dtype=str(a.dtype),
                        bytes_equal=a.dtype == b.dtype
                        and a.shape == b.shape
                        and a.tobytes() == b.tobytes(),
                        max_abs_difference=0 if a.dtype == bool else np.max(np.abs(a - b)),
                    )
                    saved_arrays["parity24_original_" + name] = a
                    saved_arrays["parity24_graph_" + name] = b
                parity = dict(
                    frame=frame,
                    batch=24,
                    repeats_of_current_local_candidates=True,
                    original_seconds=reference_seconds,
                    graph_seconds=accelerated_seconds,
                    arrays=comparisons,
                    all_exact=all(item["bytes_equal"] for item in comparisons.values()),
                )
                print(json.dumps(clean(dict(batch24_parity=parity))), flush=True)
            assert np.isfinite(fitted).all() and np.isfinite(scores).all()
            assert np.all(scores <= start_score + 1e-12)
            assert all(
                np.array_equal(getattr(state, name), value) for name, value in state_copies.items()
            )
            for p, copies in zip(proposals, proposal_copies, strict=True):
                assert all(
                    np.array_equal(current, before)
                    for current, before in zip((p.pose, p.predicted, p.plan), copies, strict=True)
                )
            case_results = []
            for choice, proposal in enumerate(proposals):
                before = support_record(proposal.pose, proposal.plan, rays, rig)
                after = support_record(fitted[choice], proposal.plan, rays, rig)
                fish = None if choice == 0 else helper.proposals[choice - 1].fish
                motion = np.linalg.norm(fitted[choice] - scene, axis=-1)
                compared = np.linalg.norm(fitted[choice] - fitted[0], axis=-1)
                relative = fitted[choice] - fitted[choice, :, 1:2]
                original_relative = scene - scene[:, 1:2]
                orientation = ((relative - original_relative) ** 2).sum(-1).mean() / 25
                case_results.append(
                    dict(
                        index=choice,
                        kind=proposal.kind,
                        target_fish=fish,
                        initial_score=start_score[choice],
                        initial_loss_parts=start_parts[choice],
                        fitted_score=scores[choice],
                        fitted_loss_parts=parts[choice],
                        fitted_orientation_diagnostic=orientation,
                        score_delta_vs_fitted_base=scores[choice] - scores[0],
                        initial_direct=before,
                        fitted_direct=after,
                        final_mask_supported=mask_support[choice],
                        trial_plan=proposal.plan,
                        initial_pose_xyz=proposal.pose,
                        fitted_pose_xyz=fitted[choice],
                        movement_from_saved_scene_mm=motion,
                        movement_from_fitted_base_mm=compared,
                        initial_candidate_mouth_shift_mm=0
                        if fish is None
                        else helper.proposals[choice - 1].displacement_mm,
                        max_other_fish_node_movement_from_saved_mm=None
                        if fish is None
                        else motion[np.arange(3) != fish].max(),
                        max_other_fish_node_difference_from_fitted_base_mm=None
                        if fish is None
                        else compared[np.arange(3) != fish].max(),
                        fitted_segment_lengths_mm=np.linalg.norm(
                            np.diff(fitted[choice], axis=1), axis=-1
                        ),
                        fitted_segment_length_ratios=np.linalg.norm(
                            np.diff(fitted[choice], axis=1), axis=-1
                        )
                        / tracker.lengths,
                    )
                )
            best = int(np.argmin(scores))
            selected = best if best else (int(np.argmin(scores[1:])) + 1 if len(scores) > 1 else 0)
            overlay, coordinates = draw_comparison(
                rig,
                images,
                fitted[0],
                fitted[selected],
                case_results[0]["fitted_direct"]["observed"],
                case_results[selected]["fitted_direct"]["observed"],
                frame,
                f"candidate {selected} fit",
            )
            overlay_path = OUT / f"fit_f{frame}.png"
            assert cv2.imwrite(str(overlay_path), overlay)
            record.output(overlay_path.name, overlay_path, sha256=sha256(overlay_path))
            prefix = f"f{frame}_"
            saved_arrays.update(
                {
                    prefix + "pixels": np.stack([targets.pixels[v] for v in rig.views]),
                    prefix + "target": targets.target,
                    prefix + "weights": targets.weights,
                    prefix + "foreground": targets.foreground,
                    prefix + "initial_pose": np.stack([p.pose for p in proposals]),
                    prefix + "predicted": np.stack([p.predicted for p in proposals]),
                    prefix + "plan": np.stack([p.plan for p in proposals]),
                    prefix + "fitted_pose": fitted,
                    prefix + "scores": scores,
                    prefix + "loss_parts": parts,
                    prefix + "review_unchanged": state.review,
                }
            )
            results.append(
                dict(
                    frame=frame,
                    original_saved_direct=saved_support[index],
                    fixed_missing=state.missing,
                    identity_review_unchanged=state.review,
                    helper=helper.diagnostics,
                    native_checks=raw_checks,
                    best_index=best,
                    fit_seconds_including_graph_setup=elapsed,
                    choices=case_results,
                    overlay_selected_index=selected,
                    overlay_native_coordinates=coordinates,
                )
            )
            print(
                json.dumps(
                    dict(
                        frame=frame,
                        best=best,
                        scores=scores.tolist(),
                        mouth_supported=[
                            c["fitted_direct"]["observed"][:, 0].tolist() for c in case_results
                        ],
                        seconds=elapsed,
                    )
                ),
                flush=True,
            )
        assert {path: sha256(path) for path in hashes} == hashes
        runtime = dict(
            python=sys.version,
            platform=platform.platform(),
            numpy=np.__version__,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            dtype=str(tracker.dtype),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            opencv_threads=cv2.getNumThreads(),
        )
        output = dict(
            status="completed",
            frames=FRAMES,
            config=dataclasses.asdict(cfg),
            helper_config=dataclasses.asdict(helper_cfg),
            loss_part_order=PARTS,
            runtime=runtime,
            inputs=hashes,
            cases=results,
            batch24_original_graph_parity=parity,
            limitations=[
                "Each frame starts separately from its saved inferred scene; not a continuous run.",
                "No human ID labels or later baseline poses enter the fits.",
                "Fresh base assignment can differ from the unsaved original pre-fit plan.",
                "All alternatives retain identical predicted scene, missing counters and review.",
                "Plans are trials; filtered sources and observed encode direct support.",
                "Lowest local cost is neither identity truth nor proof of mouth anatomy.",
                "Five original loss parts omit orientation; it is separately recorded.",
                "Fitting may move all fish; proposals change only one mouth before fitting.",
                "14-step local fitting cannot show online survival or temporal ID continuity.",
                "Timings include graph setup or reuse, and are not throughput tests.",
            ],
        )
        arrays_path = OUT / "local_fit_arrays.npz"
        np.savez_compressed(arrays_path, **saved_arrays)
        record.output("arrays", arrays_path, sha256=sha256(arrays_path))
        result_path = OUT / "summary.json"
        result_path.write_text(json.dumps(clean(output), indent=2, allow_nan=False))
        record.output("summary", result_path, sha256=sha256(result_path))
        snapshot = OUT / "source_snapshot"
        snapshot.mkdir()
        for source in sources:
            target = snapshot / source.name
            target.write_bytes(source.read_bytes())
            record.output("source_" + source.name, target, sha256=sha256(target))


if __name__ == "__main__":
    main()

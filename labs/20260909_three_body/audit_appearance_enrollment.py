"""CPU replay of causal appearance enrollment from sealed online states and raw pixels.

This reuses only the registered instance/core descriptor primitives, independently
repeats enrollment selection/aggregation, and checks SLEAP support. It neither
reruns the tracker nor treats saved slots or their descriptors as identity truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import run_experiment as base
from appearance_diagnostic import (
    EROSION_RADIUS,
    FEATURES,
    MIN_CORE_PIXELS,
    assign_instances,
    describe_core,
)
from node_geometry import RayObservations, supported_nodes
from observations import SequenceReader

from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

ONLINE = {
    "online_pose_xyz": ((3, 3, 3), "float64"),
    "online_sources": ((3, 3, 3), "int64"),
    "online_observed": ((3, 3), "bool"),
    "online_identity_review": ((3,), "bool"),
}
PER_FRAME = {
    "frame_idx": ((), "int64"),
    "features": ((3, 3, 6), "float64"),
    "reasons": ((3, 3), "U64"),
    "instance_rows": ((3, 3), "int64"),
    "core_pixels": ((3, 3), "int64"),
}
FINAL = {
    "counts": ((3, 3), "int64"),
    "templates": ((3, 3, 6), "float64"),
    "enabled_views": ((3,), "bool"),
}
REPLAY_SOURCES = (
    "labs/20260909_three_body/appearance_diagnostic.py",
    "labs/20260909_three_body/online_appearance.py",
    "labs/20260909_three_body/observations.py",
    "labs/20260909_three_body/prepare_sequence.py",
    "labs/20260909_three_body/node_geometry.py",
    "src/medaka3d/calib.py",
    "src/medaka3d/types.py",
)


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def exact_array(actual, expected, name):
    """Require shape, dtype and byte equality, including signed zeros/NaN payloads."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (
        actual.dtype != expected.dtype or actual.shape != expected.shape
        or actual.tobytes(order="C") != expected.tobytes(order="C")
    ):
        raise ValueError(f"enrollment array is not bitwise equal: {name}")
    return {"shape": list(actual.shape), "dtype": str(actual.dtype),
            "sha256": array_sha256(actual), "bitwise_equal": True}


def validate_snapshot(arrays, frames, training_frames):
    """Enforce a complete initial native prefix, with online rather than backtrace states."""
    if type(training_frames) is not int or not 1 <= training_frames <= 166:
        raise ValueError("training_frames must be a registered integer prefix of at most 166")
    frames = np.asarray(frames)
    if (
        frames.ndim != 1 or frames.dtype != np.int64 or len(frames) < training_frames
        or np.any(np.diff(frames) != 1)
    ):
        raise ValueError("source frames must be a complete consecutive native sequence")
    if set(arrays) != set(ONLINE) | set(PER_FRAME) | set(FINAL):
        raise ValueError("enrollment snapshot has missing or unknown array keys")
    for name, (tail, dtype) in {**ONLINE, **PER_FRAME, **FINAL}.items():
        value = arrays[name]
        shape = tail if name in FINAL else (training_frames, *tail)
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"enrollment snapshot shape/dtype mismatch: {name}")
    exact_array(arrays["frame_idx"], frames[:training_frames], "first native frames")
    if not np.isfinite(arrays["online_pose_xyz"]).all():
        raise ValueError("online poses must be finite")
    sources = arrays["online_sources"]
    if np.any(sources < -1):
        raise ValueError("online sources must be -1 or actual nonnegative SLEAP rows")
    if not np.array_equal(arrays["online_observed"], (sources >= 0).sum(axis=2) >= 2):
        raise ValueError("online observed flags must equal strict two-view source support")


def validate_online_support(frame, pose, sources, observed, rig):
    """Check source bounds/exclusivity, then repeat the original native geometry gates.

    The saved sources are already filtered by these gates. They must be a fixed
    point of the same support calculation; absent assignments are not invented.
    """
    if set(frame.detections) != set(VIEWS):
        raise ValueError("all three current SLEAP views are required")
    for vi, view in enumerate(VIEWS):
        points = np.asarray(frame.detections[view])
        if points.ndim != 3 or points.shape[1:] != (3, 2):
            raise ValueError("invalid current SLEAP point shape")
        for node in range(3):
            rows = sources[:, vi, node]
            rows = rows[rows >= 0]
            if np.any(rows >= len(points)):
                raise ValueError("online source row is outside current SLEAP detections")
            if len(np.unique(rows)) != len(rows):
                raise ValueError("online source row is shared by different slots")
            if not np.isfinite(points[rows, node]).all():
                raise ValueError("online source references a nonfinite current SLEAP node")
    rays = RayObservations.from_points(rig, frame.detections)
    supported, held, _ = supported_nodes(pose, sources, rays, rig)
    if not np.array_equal(supported, sources) or not np.array_equal(held, observed):
        raise ValueError("online state fails original ray/pixel support gates")
    return int((sources >= 0).sum()), int(observed.sum())


def describe_online_frame(frame, pose, sources, observed, review, rig):
    """Replay eligibility and descriptor extraction without invoking the accumulator."""
    if tuple(rig.views) != VIEWS:
        raise ValueError("rig view order differs from enrollment order")
    feature = np.full((3, 3, len(FEATURES)), np.nan, np.float64)
    reason = np.full((3, 3), "view_unavailable", dtype="U64")
    rows = np.full((3, 3), -1, np.int64)
    sizes = np.zeros((3, 3), np.int64)
    for vi, view in enumerate(VIEWS):
        image, masks = frame.images_bgr.get(view), frame.masks.get(view)
        if image is None:
            reason[vi] = "image_unavailable"
        elif masks is not None:
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("enrollment replay requires native uint8 BGR")
            if masks.dtype != bool or masks.ndim != 3 or masks.shape[1:] != image.shape[:2]:
                raise ValueError("enrollment replay requires native full-image boolean masks")
            projected = np.asarray(rig.camera(view).project(pose[:, :2]))
            if projected.shape != (3, 2, 2):
                raise ValueError("camera projected an invalid mouth/pectoral shape")
            eligible = observed[:, :2].all(axis=1)
            eligible &= (sources[:, vi, :2] >= 0).all(axis=1)
            eligible &= ~review
            rows[vi], reason[vi] = assign_instances(masks, projected, eligible)
            for fish in np.flatnonzero(rows[vi] >= 0):
                descriptor, core = describe_core(image, masks[rows[vi, fish]])
                sizes[vi, fish] = np.count_nonzero(core)
                if descriptor is None:
                    reason[vi, fish] = "core_too_small"
                else:
                    feature[vi, fish] = descriptor
        reason[vi, review] = "identity_review"
    return {"features": feature, "reasons": reason, "instance_rows": rows, "core_pixels": sizes}


def aggregate_descriptors(features, minimum_template_samples):
    """Compute every count and view gate anew; a disabled view has only NaN templates."""
    if type(minimum_template_samples) is not int or minimum_template_samples < 1:
        raise ValueError("minimum sample count must be a positive integer")
    features = np.asarray(features)
    if features.ndim != 4 or features.shape[1:] != (3, 3, 6):
        raise ValueError("features must be frame,view,fish,6")
    valid = np.isfinite(features).all(axis=3)
    if np.any(~valid & ~np.isnan(features).all(axis=3)):
        raise ValueError("each descriptor must be wholly finite or wholly NaN")
    if np.any((features[valid] < 0) | (features[valid] > 1)):
        raise ValueError("finite descriptor components must remain within [0,1]")
    counts = np.count_nonzero(valid, axis=0).astype(np.int64)
    enabled = np.array([min(counts[vi]) >= minimum_template_samples for vi in range(3)])
    templates = np.full((3, 3, 6), np.nan, np.float64)
    for vi, fish in np.ndindex(3, 3):
        if enabled[vi]:
            templates[vi, fish] = np.median(features[valid[:, vi, fish], vi, fish], axis=0)
    return {"counts": counts, "templates": templates, "enabled_views": enabled}


def replay_prefix(observations, arrays, rig, minimum_template_samples):
    """Consume exactly N frames and release captures even on mismatch or early failure."""
    frames = arrays["frame_idx"]
    results, audit = [], []
    iterator = iter(observations)
    try:
        for index, frame_idx in enumerate(frames):
            try:
                frame = next(iterator)
            except StopIteration as exc:
                raise ValueError("source ended before the complete enrollment prefix") from exc
            if frame.frame_idx != frame_idx:
                raise ValueError("source frame differs from saved online enrollment frame")
            pose, sources, observed, review = (
                arrays[name][index] for name in ONLINE
            )
            source_count, observed_count = validate_online_support(
                frame, pose, sources, observed, rig
            )
            result = describe_online_frame(frame, pose, sources, observed, review, rig)
            for name, value in result.items():
                exact_array(value, arrays[name][index], f"{frame_idx}/{name}")
            results.append(result)
            audit.append({
                "frame_idx": int(frame_idx), "source_rows_verified": source_count,
                "observed_nodes_verified": observed_count,
                "views": {
                    view: {
                        "bgr_sha256": array_sha256(frame.images_bgr[view]),
                        "sleap_points_sha256": array_sha256(frame.detections[view]),
                        "mask_available": frame.masks[view] is not None,
                        "mask_count": None if frame.masks[view] is None else len(frame.masks[view]),
                        "decoded_masks_sha256": None if frame.masks[view] is None
                        else array_sha256(frame.masks[view]),
                    } for view in VIEWS
                },
            })
            if (index + 1) % 25 == 0 or index + 1 == len(frames):
                print(json.dumps({"enrollment_frames_audited": index + 1,
                                  "requested": len(frames)}), flush=True)
    finally:
        if hasattr(iterator, "close"):
            iterator.close()
    recomputed = {"frame_idx": frames.copy()}
    for key in ("features", "reasons", "instance_rows", "core_pixels"):
        recomputed[key] = np.stack([result[key] for result in results])
    recomputed.update(aggregate_descriptors(recomputed["features"], minimum_template_samples))
    comparisons = {name: exact_array(value, arrays[name], name)
                   for name, value in recomputed.items()}
    return recomputed, audit, comparisons


def check_replay_sources(summary, run):
    """Do not replay historical data with an unrecorded replacement implementation."""
    before = {base.safe_relative(k).as_posix(): v
              for k, v in summary["code_sha256_before"].items()}
    hashes = {}
    for relative in REPLAY_SOURCES:
        current = base.REPO / relative
        snapshot = base.resolve_input(run / "code_snapshot", relative)
        digest = base.sha256(current)
        if before.get(relative) != digest or base.sha256(snapshot) != digest:
            raise ValueError(f"replay dependency differs from the recorded run: {relative}")
        hashes[str(current)] = digest
    return hashes


def audit_code_paths():
    return list(dict.fromkeys([
        Path(__file__).resolve(),
        *(base.REPO / p for p in REPLAY_SOURCES),
        *(base.LAB / name for name in (
            "evaluate_appearance_minute.py", "evaluate_variant_minute.py", "evaluate_minute.py",
            "run_appearance_tracking.py", "run_experiment.py", "run_node_recovery.py",
            "run_persistent_node_recovery.py",
        )),
        base.REPO / "src/medaka3d/provenance.py",
    ]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed causal appearance run")
    parser.add_argument("--output", type=Path, required=True, help="New CPU audit directory")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite enrollment audit output")
    # Imported lazily so primitive CPU tests do not require the comparison evaluator.
    from evaluate_appearance_minute import (
        validate_appearance_arrays,
        verify_appearance_evidence,
        verify_enrollment,
    )
    from evaluate_variant_minute import load_json, local_path

    run = local_path(args.run)
    summary = load_json(run / "summary.json")
    config = summary["enrollment_config"]
    if config != {"training_frames": 166, "minimum_template_samples": 5}:
        raise ValueError("the actual enrollment audit requires the fixed 166/5 configuration")
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance("three_body_raw_appearance_enrollment_audit", args.output, name="audit") as rec:
        rec.params_from(args)
        started = time.perf_counter()
        codes = {str(p): base.sha256(p) for p in audit_code_paths()}
        # The verifier binds inference data, candidate/enrollment outputs and copied sources.
        # Evaluation-only old candidates and human annotation assets are not opened here.
        inputs = verify_appearance_evidence(run, summary, include_post_inference=False)
        record_path = run / "experiment.provenance.json"
        inputs[str(record_path)] = base.sha256(record_path)
        record = load_json(record_path)
        source = local_path(record["params"]["sequence"])
        input_root = local_path(record["params"]["input_root"])
        inputs.update(check_replay_sources(summary, run))
        versions = {"numpy": np.__version__, "opencv": cv2.__version__}
        recorded_versions = record["params"]["package_versions"]
        if any(recorded_versions.get(k) != v for k, v in versions.items()):
            raise ValueError("descriptor replay requires the recorded NumPy and OpenCV versions")
        rec.param("package_versions", versions)
        rec.param("device", "cpu")
        rec.param("ground_truth_and_completed_baseline_arrays_opened", False)
        rec.param("reexecuted_beam_selection_or_tracking", False)
        with np.load(run / "appearance_enrollment.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        reader = SequenceReader(source, input_root)
        validate_snapshot(arrays, reader.frames, config["training_frames"])
        # Final-backtrace arrays are checked only for saved scoring totals/schema.
        # None of their pose/source/observed fields enters descriptor replay.
        with np.load(run / "candidate.npz", allow_pickle=False) as saved:
            candidate = {key: saved[key] for key in saved.files}
        candidate = validate_appearance_arrays(candidate, reader.frames, summary)
        enrollment_binding, enrollment_hashes = verify_enrollment(
            run, summary, candidate, reader.frames, nodes=reader.nodes
        )
        inputs.update(enrollment_hashes)
        for path, digest in {**inputs, **codes}.items():
            rec.input(path, path, sha256=digest)
        for path, digest in codes.items():
            target = args.output / "code_snapshot" / Path(path).relative_to(base.REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if base.sha256(target) != digest:
                raise RuntimeError("audit source snapshot changed during copy")
            rec.output("code:" + str(Path(path).relative_to(base.REPO)), target, sha256=digest)
        rig = base.load_rig(input_root / base.CALIBRATION_DIR)
        recomputed, frames_audit, comparisons = replay_prefix(
            reader, arrays, rig, config["minimum_template_samples"]
        )
        payload_path = args.output / "recomputed_enrollment.npz"
        np.savez_compressed(payload_path, **recomputed)
        rec.output("recomputed_enrollment", payload_path, sha256=base.sha256(payload_path))
        qa = {
            "status": "complete_exact_cpu_replay", "algorithm_arm": summary["algorithm_arm"],
            "run": str(run), "enrollment_sha256": base.sha256(run / "appearance_enrollment.npz"),
            "sequence_manifest_sha256": base.sha256(source / "manifest.json"),
            "frame_range": [int(arrays["frame_idx"][0]), int(arrays["frame_idx"][-1] + 1)],
            "native_frames": len(arrays["frame_idx"]), "native_fps": 166,
            "decoded_images": 3 * len(arrays["frame_idx"]),
            "all_recomputed_arrays_bitwise_equal": True,
            "raw_feature_and_mask_recomputation_performed": True,
            "online_native_support_verified": True,
            "online_states_compared_to_final_traceback": False,
            "identity_labels_verified": False,
            "counts": recomputed["counts"].tolist(),
            "enabled_views": recomputed["enabled_views"].tolist(),
            "erosion_radius_px": EROSION_RADIUS, "minimum_core_pixels": MIN_CORE_PIXELS,
            "feature_names": list(FEATURES), "views": list(VIEWS),
            "array_comparisons": comparisons, "frames": frames_audit,
            "enrollment_record_binding": enrollment_binding,
            "elapsed_seconds_including_input_hashes": time.perf_counter() - started,
            "limitations": [
                "Saved current online slots are model hypotheses, not biological identity truth.",
                "Enrollment is replayed from recorded online states; beam selection is not rerun.",
                "The final trajectory backtrace is not substituted for the recorded online prefix.",
                "The same registered assignment and core descriptor primitives are reused.",
                "No post-prefix images, GT or completed baseline arrays are used for replay.",
                "File SHA covers full source videos; native decoding covers only the prefix.",
                "Matching enrollment establishes neither color discriminability nor swap-free IDs.",
            ],
        }
        for path, digest in {**inputs, **codes}.items():
            if base.sha256(path) != digest:
                raise RuntimeError("an audit input or source changed during CPU replay")
        rec.param("inputs_and_sources_unchanged", True)
        qa_path = args.output / "summary.json"
        qa_path.write_text(json.dumps(qa, indent=2, allow_nan=False), encoding="utf-8")
        rec.output("summary", qa_path, sha256=base.sha256(qa_path))
    print(json.dumps({"output": str(args.output), "native_frames": len(arrays["frame_idx"]),
                      "all_recomputed_arrays_bitwise_equal": True}), flush=True)
    return qa


if __name__ == "__main__":
    main()

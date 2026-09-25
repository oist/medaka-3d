"""Read-only CPU audit of a completed registered reflection temporal comparison.

No renderer, fitter, CUDA call, GT or historical trajectory deserialization is
used. Saved unions are arithmetically checked; this is not a new image rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path[:0] = [str(LAB), str(REPO / "src")]

import run_reflection_temporal as producer  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

PRODUCER_SHA = "89e10c2acc60d2e8ede57906bc13e6a3dce2f5c9b7db571e3c5bf54861a1aadc"
RAW_PROBES = (79236, 79316, 79372, 79405, 79435)
ARMS = ("direct", "fresnel")
TOLERANCE = 1e-12
require = producer.require
sha256 = producer.sha256
read_json = producer.read_json


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    require(all(not value.dtype.hasobject for value in result.values()), "object array forbidden")
    return result


def close(actual, expected, description):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, description + " shape differs")
    require(
        np.allclose(actual, expected, atol=TOLERANCE, rtol=0, equal_nan=False),
        description + " numerical value differs",
    )
    return float(np.max(np.abs(actual - expected), initial=0))


def exact(actual, expected, description):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and actual.tobytes(order="C") == expected.tobytes(order="C"),
        description + " dtype/shape/bytes differ",
    )


def verify_completed(root, code):
    """First require completion. Only the separately registered seed is loaded."""
    hashes = {}
    seal = producer.verify_outputs(
        root, "experiment.provenance.json", "reflection_temporal_comparison", hashes
    )
    roles = {
        "plan": "plan.json",
        "summary": "summary.json",
        "comparison": "comparison.json",
        "initial_pose": "initial_pose_xyz.npy",
        "evaluation_metrics": "evaluation_metrics.npz",
        **{
            arm + suffix: arm + filename
            for arm in ARMS
            for suffix, filename in (
                ("_candidate", "_candidate.npz"),
                ("_progress", "_progress.jsonl"),
            )
        },
    }
    for role, filename in roles.items():
        producer.output_path(seal, role, root / filename)
    plan = read_json(root / "plan.json")
    summary = read_json(root / "summary.json")
    require(plan["code_hashes"] == code, "producer code set differs")
    registrations = [
        Path(entry["path"]).resolve()
        for entry in seal["inputs"].values()
        if Path(entry["path"]).name == "registration.provenance.json"
        and read_json(entry["path"]).get("stage") == "reflection_temporal_registration"
    ]
    require(len(registrations) == 1, "unique temporal registration required")
    registered, seed, inputs = producer.load_registration(registrations[0].parent, code)
    require(producer.equal(registered, plan), "executed plan differs from registration")
    expected = {**inputs, **code}
    require(
        {key: item["sha256"] for key, item in seal["inputs"].items()} == expected,
        "producer input membership differs",
    )
    for key, digest in expected.items():
        require(
            producer.workspace(seal["inputs"][key]["path"]) == producer.workspace(key),
            "producer input key/path differs",
        )
        producer.bind(key, digest, hashes)
    for path, digest in code.items():
        target = root / "code_snapshot" / Path(path).relative_to(REPO)
        role = target.relative_to(root).as_posix()
        producer.output_path(seal, role, target)
        require(seal["outputs"][role]["sha256"] == digest, "producer snapshot differs")
    require(
        sha256(root / "initial_pose_xyz.npy") == plan["seed_array_sha256"], "executed seed differs"
    )
    exact(np.load(root / "initial_pose_xyz.npy", allow_pickle=False), seed, "copied seed")
    for key, value in {
        "schema": "reflection_temporal_comparison_v1",
        "frame_range": [79236, 79436],
        "native_fps": 166,
        "temporal_subsampling": False,
        "arms": list(ARMS),
        "technical_completion": True,
        "initialized_once_each": True,
        "external_resets_each": 0,
        "inference_old_candidate_deserialized": False,
        "gt_read": False,
        "evaluation_after_both_saved": True,
        "direct_support_renderer": "original direct only",
        "code_unchanged": True,
        "inputs_unchanged": True,
        "no_global_identity_lineage": True,
        "biological_ID_validation": False,
    }.items():
        require(producer.equal(summary[key], value), "producer summary context differs: " + key)
    for key in ("qualification", "seed", "config"):
        require(producer.equal(summary[key], plan[key]), "summary/plan differs: " + key)
    comparison = read_json(root / "comparison.json")
    require(producer.equal(summary["comparison"], comparison), "embedded comparison differs")
    frames = np.asarray(plan["frames"], np.int64)
    require(np.array_equal(frames, np.arange(79236, 79436)), "registered native range differs")
    for frame in frames:
        filename = f"evaluation/f{frame}.npz"
        producer.output_path(seal, filename, root / filename)
    for arm in ARMS:
        timing = summary["timings"][arm]
        require(
            timing["initialization_count"] == 1
            and timing["reset_count"] == 0
            and timing["track_invocation_count"] == 1
            and timing["native_frames_read"] == len(frames),
            "arm native/init counters differ",
        )
    return plan, summary, comparison, frames, hashes


def runs(flags, frames):
    """Independent half-open native intervals, including first and final gaps."""
    result, first = [], None
    for index, value in enumerate(flags):
        if value and first is None:
            first = int(frames[index])
        if not value and first is not None:
            result.append([first, int(frames[index])])
            first = None
    if first is not None:
        result.append([first, int(frames[-1]) + 1])
    return result


def statistics(data, frames):
    observed, review = data["observed"], data["identity_review"]
    accepted = np.isfinite(data["accepted_xyz"]).all(-1)
    require(np.array_equal(accepted, observed[:, :, 1] & ~review), "acceptance mask differs")
    gaps = [runs(~observed[:, fish, 1], frames) for fish in range(3)]
    return dict(
        direct_nodes_per_slot=observed.sum(0).tolist(),
        direct_nodes_total=observed.sum((0, 1)).tolist(),
        direct_chest_per_slot=observed[:, :, 1].sum(0).tolist(),
        direct_chest_total=int(observed[:, :, 1].sum()),
        accepted_chest_per_slot=accepted.sum(0).tolist(),
        accepted_chest_total=int(accepted.sum()),
        longest_chest_gap_per_slot=[max((b - a for a, b in row), default=0) for row in gaps],
        chest_gap_ranges=gaps,
        review_ranges=[runs(review[:, fish], frames) for fish in range(3)],
    )


def gate(matrix, counts):
    old, new = counts["direct"], counts["fresnel"]
    checks = {
        "image_vs_direct": bool(matrix[1, 1] < matrix[0, 0] - 1e-10),
        "image_same_model": bool(matrix[1, 1] < matrix[0, 1] - 1e-10),
        "direct_chest_total": new["direct_chest_total"] >= old["direct_chest_total"],
        "accepted_chest_total": new["accepted_chest_total"] >= old["accepted_chest_total"],
        "direct_chest_every_slot": all(
            a >= b
            for a, b in zip(new["direct_chest_per_slot"], old["direct_chest_per_slot"], strict=True)
        ),
        "accepted_chest_every_slot": all(
            a >= b
            for a, b in zip(
                new["accepted_chest_per_slot"], old["accepted_chest_per_slot"], strict=True
            )
        ),
        "longest_gap_every_slot": all(
            a <= b
            for a, b in zip(
                new["longest_chest_gap_per_slot"], old["longest_chest_gap_per_slot"], strict=True
            )
        ),
    }
    return dict(checks=checks, passed=all(checks.values()), tolerance=1e-10)


def strict_support(data, reader, rig):
    """All declared sources; independent batch projection and float32 stored rays."""
    pose, source = data["pose_xyz"], data["sources"]
    distance, error = np.full(source.shape, np.nan), np.full(source.shape, np.nan)
    supported = np.zeros(source.shape, bool)
    for vi, view in enumerate(producer.VIEWS):
        pixels = np.full((*pose.shape[:-1], 2), np.nan)
        for t, frame in enumerate(data["frame_idx"]):
            points = reader.nodes.at(view, int(frame))
            for node in range(3):
                rows = source[t, :, vi, node]
                assigned = rows >= 0
                require(np.all(rows[assigned] < len(points)), "source row out of range")
                require(
                    len(set(rows[assigned].tolist())) == int(assigned.sum()),
                    "two bodies claim one current view/node row",
                )
                pixels[t, assigned, node] = points[rows[assigned], node]
        present = source[:, :, vi] >= 0
        require(np.isfinite(pixels[present]).all(), "source point is not finite")
        camera = rig.camera(view)
        origins, directions = camera.rays(pixels)
        origins, directions = origins.astype(np.float32), directions.astype(np.float32)
        offset = pose - origins
        along = np.sum(offset * directions, axis=-1, keepdims=True)
        ray_distance = np.linalg.norm(offset - along * directions, axis=-1)
        pixel_error = np.linalg.norm(camera.project(pose) - pixels, axis=-1)
        valid = present & (ray_distance <= 2.0) & np.isfinite(pixel_error) & (pixel_error <= 12.0)
        require(np.array_equal(valid, present), "source failed strict 2mm/12px support")
        distance[:, :, vi], error[:, :, vi], supported[:, :, vi] = ray_distance, pixel_error, valid
    require(np.array_equal(supported.sum(2) >= 2, data["observed"]), "observed flags differ")
    present = source >= 0
    return dict(
        source_count=int(present.sum()),
        observed_node_count=int(data["observed"].sum()),
        source_exclusivity=True,
        strict_support_matches=True,
        ray_gate_mm=2.0,
        pixel_gate=12.0,
        maximum_ray_mm=float(distance[present].max(initial=0)),
        maximum_pixel_error=float(error[present].max(initial=0)),
    ), dict(ray_distance_mm=distance, native_pixel_error=error, valid_source=supported)


def recompute_error(saved, scorer):
    target, weight = saved["target"], saved["weights"]
    valid, strata, union = saved["calibrated_valid"], saved["strata"], saved[scorer + "_union"]
    require(
        target.ndim == 2
        and target.shape[0] == 3
        and weight.shape == target.shape
        and valid.shape == target.shape
        and valid.dtype == bool
        and strata.shape == target.shape
        and union.shape == (2, *target.shape),
        "saved evaluation shape differs",
    )
    require(
        np.isfinite(union).all()
        and np.all((union >= 0) & (union <= 1))
        and np.isfinite(target).all()
        and np.all((target == 0) | (target == 1))
        and np.isfinite(weight).all()
        and np.all(weight >= 0),
        "invalid saved evaluation values",
    )
    parts, counts = np.zeros((2, 3, 3)), np.zeros((3, 3), np.int64)
    for view in range(3):
        for stratum in range(3):
            membership = strata[view] == stratum
            counts[view, stratum] = np.sum(membership & valid[view])
            # Keep the saved fixed denominator; invalid pixels contribute zero.
            for fitted in range(2):
                error = np.square(union[fitted, view] - target[view]) * weight[view] * valid[view]
                parts[fitted, view, stratum] = np.sum(error * membership)
    by_view = parts.sum(-1)
    result = dict(
        loss_by_stratum=parts,
        loss_by_view=by_view,
        mean_view_loss=by_view.mean(1),
        foreground_loss_by_view=parts[..., 0],
        background_loss_by_view=parts[..., 1:].sum(-1),
        valid_counts=counts,
    )
    maximum = 0.0
    for key, value in result.items():
        if key == "valid_counts":
            exact(saved[scorer + "_" + key], value, "valid evaluation counts")
        else:
            maximum = max(maximum, close(saved[scorer + "_" + key], value, scorer + " " + key))
    return parts, maximum


def validate_samples(saved, frame, reader, config):
    """Reconstruct detector-only samples, without decoding images or rendering."""
    masks, index = {}, int(frame - reader.frames[0])
    for view in producer.VIEWS:
        data = reader.masks[view]
        decoded = producer.decode_frame_masks(data, index)
        require(
            producer.array_sha256(decoded) == data["frame_mask_sha256"][index].decode(),
            "decoded detector mask SHA differs",
        )
        masks[view] = decoded if data["view_available"][index] else None
    dummy = np.zeros((*reader.image_shape, 3), np.uint8)
    observation = producer.FrameObservation(
        int(frame), {}, dict.fromkeys(producer.VIEWS, dummy), masks, {}
    )
    fitting = producer.sample_pixels(
        observation,
        "detectron2",
        config.positive_samples,
        config.near_samples,
        config.global_samples,
    )
    target = producer.sample_body_evaluation(observation, fitting.pixels)
    expected = dict(
        frame_idx=np.array(frame),
        fit_pixels=np.stack(list(fitting.pixels.values())),
        fit_target=fitting.target,
        fit_weights=fitting.weights,
        **{
            key: getattr(target, key)
            for key in (
                "target",
                "weights",
                "foreground",
                "strata",
                "sampled_counts",
                "pool_counts",
                "excluded_fit_unique_count",
                "view_available",
                "foreground_nonempty",
            )
        },
        **{view + "_pixels": target.pixels[view] for view in producer.VIEWS},
    )
    for name, value in expected.items():
        exact(saved[name], value, "fixed detector sample " + name)
    for view in producer.VIEWS:
        fitting_set = set(map(tuple, fitting.pixels[view]))
        require(
            not any(tuple(pixel) in fitting_set for pixel in target.pixels[view]),
            "fitting/evaluation pixel overlap",
        )


def raw_probes(reader, frames):
    require(all(frame in frames for frame in RAW_PROBES), "fixed raw probes outside range")
    is_source = reader.manifest.get("raw_storage") == "original_source_video"
    rows = []
    for view in producer.VIEWS:
        capture = cv2.VideoCapture(str(reader.videos[view]))
        try:
            require(capture.isOpened(), "raw probe video cannot open")
            require(abs(capture.get(cv2.CAP_PROP_FPS) - 166) <= 1e-6, "raw probe fps differs")
            for frame in RAW_PROBES:
                index = int(frame - reader.frames[0])
                offset = frame if is_source else index
                require(capture.set(cv2.CAP_PROP_POS_FRAMES, offset), "raw probe seek failed")
                require(
                    abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - offset) <= 1e-6,
                    "raw probe seek position differs",
                )
                ok, image = capture.read()
                require(
                    ok
                    and image.shape == (*reader.image_shape, 3)
                    and image.dtype == np.uint8
                    and abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - offset - 1) <= 1e-6,
                    "raw probe frame decode differs",
                )
                digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
                require(digest == reader.pixel_hashes[view][index], "native BGR hash differs")
                rows.append(dict(frame_idx=frame, view=view, video_offset=offset, sha256=digest))
        finally:
            capture.release()
    return rows


def save_npz(path, record, **data):
    np.savez_compressed(path, **data)
    record.output(path.name, path, sha256=sha256(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run, output = args.run.resolve(), args.output.resolve()
    require(
        run.is_relative_to(REPO) and output.is_relative_to(REPO) and not output.exists(),
        "existing run/new workspace audit required",
    )
    # Establish live-code digests before invoking a verifier or opening result arrays.
    code = producer.code_hashes()
    require(
        code[str(LAB / "run_reflection_temporal.py")] == PRODUCER_SHA, "producer source changed"
    )
    audit_code = {str(Path(__file__).resolve()): sha256(__file__), **code}
    seal_path = run / "experiment.provenance.json"
    require(
        seal_path.is_file() and read_json(seal_path).get("status") == "ok",
        "producer must be complete before audit",
    )
    output.mkdir(parents=True, exist_ok=False)
    with provenance("independent_reflection_temporal_output_audit", output, name="audit") as record:
        record.params_from(args)
        started = time.perf_counter()
        plan, summary, comparison, frames, bindings = verify_completed(run, code)
        for path, digest in {**bindings, **audit_code}.items():
            record.input(path, path, sha256=digest)
        producer.snapshot(audit_code, output, record)
        manifest = read_json(Path(plan["sequence"]) / "manifest.json")
        reader_class = (
            producer.SourceSequenceReader
            if manifest.get("raw_storage") == "original_source_video"
            else producer.SequenceReader
        )
        reader = reader_class(plan["sequence"], plan["input_root"])
        require(
            np.array_equal(reader.frames, np.arange(*plan["source_frame_range"])),
            "native source range differs",
        )
        rig = producer.load_rig(plan["calibration_root"])
        require(tuple(rig.views) == producer.VIEWS, "calibration view order differs")
        candidates, counts, support = {}, {}, {}
        for arm in ARMS:
            data = arrays(run / f"{arm}_candidate.npz")
            producer.validate_arrays(data, reader)
            exact(data["frame_idx"], frames, "candidate native frames")
            candidates[arm] = data
            counts[arm] = statistics(data, frames)
            require(
                producer.equal(counts[arm], comparison["statistics"][arm]),
                "coverage summary differs",
            )
            support[arm], support_arrays = strict_support(data, reader, rig)
            save_npz(output / f"{arm}_support.npz", record, frame_idx=frames, **support_arrays)
        config = producer.TrackerConfig(**plan["config"]["tracker"])
        all_parts, error_maximum = [], 0.0
        for frame in frames:
            saved = arrays(run / "evaluation" / f"f{frame}.npz")
            validate_samples(saved, frame, reader, config)
            parts = []
            for scorer in ARMS:
                value, error = recompute_error(saved, scorer)
                parts.append(value)
                error_maximum = max(error_maximum, error)
            all_parts.append(np.stack(parts, axis=1))
        parts = np.stack(all_parts)
        means, matrix = parts.sum(-1).mean(-1), parts.sum(-1).mean(-1).mean(0)
        packed = arrays(run / "evaluation_metrics.npz")
        require(
            set(packed) == {"frame_idx", "per_view_stratum_errors", "mean_view_errors"},
            "evaluation aggregate schema differs",
        )
        exact(packed["frame_idx"], frames, "evaluation native frames")
        for name, actual in (("per_view_stratum_errors", parts), ("mean_view_errors", means)):
            error_maximum = max(error_maximum, close(packed[name], actual, name))
        for name, actual in (
            ("matrix", matrix),
            ("per_frame_mean_errors", means),
            ("per_view_stratum_means", parts.mean(0)),
        ):
            error_maximum = max(error_maximum, close(comparison[name], actual, name))
        for name, selection in (
            ("same_model_worse_frames", means[:, 1, 1] > means[:, 0, 1]),
            ("common_direct_worse_frames", means[:, 1, 0] > means[:, 0, 0]),
        ):
            require(comparison[name] == frames[selection].tolist(), "worse-frame summary differs")
        independent_gate = gate(matrix, counts)
        require(
            all(
                producer.equal(comparison["gate"][key], value)
                for key, value in independent_gate.items()
            ),
            "fixed promotion gate differs",
        )
        native = raw_probes(reader, frames)
        producer.save_json(output / "native_raw_probes.json", native, record)
        save_npz(
            output / "recomputed_evaluation.npz",
            record,
            frame_idx=frames,
            per_view_stratum_errors=parts,
            mean_view_errors=means,
            matrix=matrix,
        )
        result = dict(
            schema="independent_reflection_temporal_output_audit_v1",
            passed=True,
            producer=str(run),
            producer_seal_sha256=sha256(seal_path),
            frame_range=[int(frames[0]), int(frames[-1]) + 1],
            native_frames=len(frames),
            candidate_sha256={arm: sha256(run / f"{arm}_candidate.npz") for arm in ARMS},
            source_and_input_guards=True,
            registered_seed_sha256=plan["seed_array_sha256"],
            qualification=plan["qualification"],
            schema_21_each=True,
            node_age_audit=True,
            strict_support=support,
            statistics=counts,
            matrix=matrix.tolist(),
            gate=independent_gate,
            maximum_saved_error_difference=error_maximum,
            error_tolerance=TOLERANCE,
            all_200_fixed_detector_samples_exact=True,
            fitting_evaluation_pixels_disjoint=True,
            raw_probe_native_frames=list(RAW_PROBES),
            raw_probe_images=len(native),
            producer_declared_native_decodes=400,
            producer_timings=summary["timings"],
            rerendered=False,
            refitted=False,
            historical_coordinates_loaded=False,
            GT_read=False,
            biological_ID_truth=False,
            inherited_global_identity_lineage=False,
            optimistic_retrospective_seed=plan["seed"],
            elapsed_seconds=time.perf_counter() - started,
            limitations=[
                "Saved image unions were arithmetically audited, not rerendered.",
                "D2 evaluation pixels are excluded from fitting, but are not mask ground truth.",
                "Strict support and local reset review do not establish biological IDs.",
                "The local experiment does not carry the earlier minute's identity lineage.",
            ],
        )
        producer.save_json(output / "summary.json", result, record)
        report = (
            "# Independent short reflection temporal audit\n\n"
            f"Completed both native 200-frame outputs: all 21 arrays, source/seed/code seals, "
            "strict support, node ages and review masking pass. "
            f"Fixed gate: **{independent_gate['passed']}**.\n\n"
            f"Direct / Fresnel accepted chest counts: {counts['direct']['accepted_chest_total']} / "
            f"{counts['fresnel']['accepted_chest_total']} out of 600.\n\n"
            f"Fitted-arm by scorer matrix: `{matrix.tolist()}`. Maximum arithmetic discrepancy: "
            f"{error_maximum:.3g}. All 200 fixed detector samples agree; "
            "15 raw BGR probes agree.\n\n"
            "This audit performs no new fitting or rendering. A passed software audit is distinct "
            "from passing the fixed development gate. Neither certifies biological identities. "
            "The optimistic saved seed starts a new local lineage with review false; it does not "
            "preserve or repair the original minute's ID history.\n"
        )
        (output / "RESULT.md").write_text(report, encoding="utf-8")
        record.output("RESULT.md", output / "RESULT.md", sha256=sha256(output / "RESULT.md"))
        producer.guard(bindings, audit_code, record)
    print(json.dumps(dict(status="ok", output=str(output)), allow_nan=False))


if __name__ == "__main__":
    main()

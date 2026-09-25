"""Independent CPU checks of a completed persistent minute and its completed comparison."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))

import evaluate_minute as old  # noqa: E402
import evaluate_variant_minute as variant  # noqa: E402
import run_experiment as base  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def complete_gate(run, comparison):
    # This is the first operation. Incomplete candidate or result arrays are never opened.
    paths = [run / "experiment.provenance.json", comparison / "evaluation.provenance.json"]
    for path in paths:
        if not path.is_file() or json_read(path).get("status") != "ok":
            raise ValueError(f"both run and comparison must be sealed ok before audit: {path}")


def verify_comparison(comparison, known):
    path = comparison / "evaluation.provenance.json"
    record = json_read(path)
    hashes = {str(path.resolve()): base.sha256(path)}
    for key, filename in (
        ("comparison", "comparison.json"), ("diagnostics", "diagnostics.json"),
        ("review_events", "review_events.json"),
    ):
        expected = (comparison / filename).resolve(strict=True)
        item = record["outputs"][key]
        if variant.local_path(item["path"]) != expected:
            raise ValueError("comparison required output has an unbound path")
    for category in ("inputs", "outputs"):
        for item in record[category].values():
            value = variant.local_path(item["path"])
            digest = known.get(str(value)) or hashes.get(str(value)) or base.sha256(value)
            if digest != item.get("sha256"):
                raise ValueError("completed comparison file differs from its seal")
            hashes[str(value)] = digest
    return hashes


def first_frame(mask, frames):
    indexes = np.flatnonzero(mask)
    return None if not len(indexes) else int(frames[indexes[0]])


def direct_counts(arrays):
    observed, review, frames = arrays["observed"], arrays["identity_review"], arrays["frame_idx"]
    direct = np.isfinite(arrays["xyz"]).all(axis=-1)
    accepted = np.isfinite(arrays["accepted_xyz"]).all(axis=-1)
    if not np.array_equal(direct, observed[:, :, 1]):
        raise ValueError("direct coordinates differ from observed pectoral support")
    if not np.array_equal(accepted, direct & ~review):
        raise ValueError("accepted coordinates differ from direct support and review exclusion")
    return {
        "node_names": ["mouth", "pectoral", "tail"],
        "direct_node_counts_fish_node": observed.sum(axis=0).tolist(),
        "direct_pectoral_fish_frames": int(direct.sum()),
        "accepted_pectoral_fish_frames": int(accepted.sum()),
        "direct_pectoral_per_slot": direct.sum(axis=0).tolist(),
        "accepted_pectoral_per_slot": accepted.sum(axis=0).tolist(),
        "all_three_direct_frames": int(direct.all(axis=1).sum()),
        "all_three_accepted_frames": int(accepted.all(axis=1).sum()),
        "review_fish_frames": int(review.sum()),
        "review_per_slot": review.sum(axis=0).tolist(),
        "first_review_frame_per_slot": [first_frame(review[:, f], frames) for f in range(3)],
    }


def verify_all_support(arrays, nodes, rig):
    """Reproject every declared source; preserve the original float32 ray storage step."""
    frames, pose, source = arrays["frame_idx"], arrays["pose_xyz"], arrays["sources"]
    old.validate_sources(source, frames, nodes)
    distances = np.full(source.shape, np.nan)
    errors = np.full(source.shape, np.nan)
    validated = np.zeros(source.shape, bool)
    for vi, view in enumerate(base.VIEWS):
        pixels = np.full((*pose.shape[:-1], 2), np.nan)
        for t, frame in enumerate(frames):
            points = nodes.at(view, int(frame))
            for node in range(3):
                present = source[t, :, vi, node] >= 0
                pixels[t, present, node] = points[source[t, present, vi, node], node]
        present = source[:, :, vi] >= 0
        if not np.isfinite(pixels[present]).all():
            raise ValueError("saved supported node refers to a nonfinite SLEAP point")
        camera = rig.camera(view)
        origin, direction = camera.rays(pixels)
        # assigned_rays in the original tracker stores these values as float32.
        origin, direction = origin.astype(np.float32), direction.astype(np.float32)
        delta = pose - origin
        distance = np.linalg.norm(
            delta - np.sum(delta * direction, axis=-1, keepdims=True) * direction, axis=-1
        )
        error = np.linalg.norm(camera.project(pose) - pixels, axis=-1)
        valid = present & (distance <= 2.0) & np.isfinite(error) & (error <= 12.0)
        distances[:, :, vi], errors[:, :, vi], validated[:, :, vi] = distance, error, valid
        if not np.array_equal(valid, present):
            raise ValueError(f"declared sources fail original ray/pixel gates: {view}")
    if not np.array_equal(validated.sum(axis=2) >= 2, arrays["observed"]):
        raise ValueError("observed node mask differs from native source geometry")
    present = source >= 0
    return {
        "source_rows_checked": int(present.sum()),
        "observed_nodes_checked": int(arrays["observed"].sum()),
        "maximum_supported_ray_distance_mm": float(distances[present].max()),
        "maximum_supported_pixel_error": float(errors[present].max()),
        "ray_gate_mm": 2.0, "native_pixel_gate": 12.0,
        "all_sources_and_observed_flags_pass": True,
        "scope": "declared finite sources only; absent assignments are not manufactured",
    }, {"frame_idx": frames, "ray_distance_mm": distances,
        "native_pixel_error": errors, "valid_source": validated}


def compare_trajectory(current, reference):
    frames = current["frame_idx"]
    if not np.array_equal(frames, reference["frame_idx"]):
        raise ValueError("trajectory frame alignment differs")
    first = {}
    for name in ("pose_xyz", "sources", "observed", "identity_review", "proposal_kind"):
        a, b = current[name], reference[name]
        different = a != b
        if a.ndim > 1:
            different = different.any(axis=tuple(range(1, a.ndim)))
        first[name] = first_frame(different, frames)
    displacement = np.linalg.norm(current["pose_xyz"] - reference["pose_xyz"], axis=-1)
    return {
        "first_exact_value_difference_by_array": first,
        "first_any_node_difference_over_0_001_mm": first_frame(
            (displacement > .001).any(axis=(1, 2)), frames
        ),
        "maximum_same_slot_node_difference_mm": float(displacement.max()),
        "scope": "fixed saved slot names; method disagreement is not identity truth",
    }


def code_paths():
    return [Path(__file__), *(LAB / name for name in (
        "evaluate_variant_minute.py", "evaluate_minute.py", "run_experiment.py",
        "run_node_recovery.py", "run_persistent_node_recovery.py", "diagnostics.py",
        "review_events.py", "observations.py", "prepare_sequence.py",
    )), *(REPO / "src/medaka3d" / name for name in (
        "calib.py", "types.py", "provenance.py", "eval/generative.py", "eval/reacquisition.py",
    ))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    complete_gate(args.run, args.comparison)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite independent minute audit")
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance(
        "persistent_mouth_minute_independent_cpu_audit", args.output, name="audit"
    ) as rec:
        rec.params_from(args)
        started = time.perf_counter()
        code = code_paths()
        source_hashes = {str(p.resolve()): base.sha256(p) for p in code}
        run = variant.local_path(args.run)
        comparison = variant.local_path(args.comparison)
        report = json_read(comparison / "comparison.json")
        original_root = variant.local_path(report["methods"]["original_three_body_v1"]["run_path"])
        current, summary, hashes = old.load_completed(run)
        reference, ref_summary, ref_hashes = old.load_completed(original_root)
        bound = variant.verify_variant_evidence(run, summary, expected_frames=old.FRAMES)
        ref_bound = old.verify_run_evidence(original_root, ref_summary)
        parity = variant.same_v1_inputs(run, summary, original_root, ref_summary, bound, ref_bound)
        hashes.update({**ref_hashes, **bound, **ref_bound})
        hashes.update(verify_comparison(comparison, hashes))
        if not variant.exact_json(report["variant_original_input_parity"], parity):
            raise ValueError("comparison input parity differs from independent verification")
        if (report["current_candidate_sha256"] != base.sha256(run / "candidate.npz")
                or variant.local_path(report["current_run_path"]) != run):
            raise ValueError("comparison is bound to a different current candidate")
        current = variant.validate_variant_arrays(current, old.FRAMES, summary)
        record = json_read(run / "experiment.provenance.json")
        input_root = variant.local_path(record["params"]["input_root"])
        nodes = old.CachedNodes(input_root, base.EXPERIMENT, old.FRAMES)
        rig = old.load_rig(input_root / base.CALIBRATION_DIR)
        initial, _, evaluation = base.registered_seed(int(old.FRAMES[0]))
        catalog, _ = base.staging_catalog(input_root)
        gt_path = base.verified_input(
            input_root, evaluation["female_annotations_relative"], catalog
        )
        old.require_binding(gt_path, base.sha256(gt_path), bound)
        with gt_path.open(encoding="utf-8-sig", newline="") as stream:
            anchors = old.match_mouth_anchors(
                csv.DictReader(stream), lambda view, frame: nodes.at(view, frame)[:, 0],
                int(old.FRAMES[0]), int(old.FRAMES[-1] + 1),
            )
        measurements, female = {}, {}
        for name, arrays in (("current_variant", current), ("original_three_body_v1", reference)):
            source, review, observed, xyz, accepted = old.normalized_points(arrays)
            metrics = old.common_metrics(
                old.FRAMES, source, review, observed, xyz, accepted, anchors
            )
            for key, value in metrics.items():
                if not variant.exact_json(value, report["methods"][name][key]):
                    raise ValueError(f"shared metric differs from comparison: {name}/{key}")
            measurements[name] = direct_counts(arrays)
            for field, section in (("direct", "direct_motion"), ("accepted", "accepted_motion")):
                actual = measurements[name][f"{field}_pectoral_fish_frames"] / (len(old.FRAMES) * 3)
                if actual != metrics[section]["observed_fraction"]:
                    raise ValueError("independent count and common fraction differ")
            female[name] = {key: metrics[key] for key in ("female_raw", "female_accepted")}
        geometry, geometry_arrays = verify_all_support(current, nodes, rig)
        diagnostics = old.diagnose(current, initial_pose_xyz=initial)
        saved_diagnostics = json_read(comparison / "diagnostics.json")
        for key, value in diagnostics.items():
            if key != "baseline_disagreement" and not variant.exact_json(
                value, saved_diagnostics[key]
            ):
                raise ValueError(f"current diagnostic differs from comparison: {key}")
        events = old.extract_review_events(
            current, review_gap_frames=summary["tracker_config"]["review_gap_frames"]
        )
        if not variant.exact_json(events, json_read(comparison / "review_events.json")):
            raise ValueError("review-event table differs from comparison")
        selected = current["proposal_kind"] == "mouth_recovery"
        result = {
            "status": "complete_independent_cpu_audit", "algorithm_arm": summary["algorithm_arm"],
            "frame_range": [76620, 86580], "native_frames": 9960, "native_fps": 166,
            "candidate_sha256": base.sha256(run / "candidate.npz"),
            "comparison_sha256": base.sha256(comparison / "comparison.json"),
            "input_parity": parity, "counts": measurements,
            "all_node_source_geometry": geometry, "female_endpoint_tables": female,
            "shared_metric_and_review_event_tables_match": True,
            "selected_mouth_proposal_count": int(selected.sum()),
            "selected_mouth_proposal_frames": old.FRAMES[selected].tolist(),
            "trajectory_difference_from_original_v1": compare_trajectory(current, reference),
            "independent_all_three_identity_truth": False,
            "limitations": [
                "Female endpoint agreement does not verify male IDs or returning swaps.",
                "Support, internal proposal selection and method disagreement are not ID truth.",
                "No raw video decoding, model inference, tracking or new human annotation occurs.",
            ],
            "elapsed_seconds_before_final_rehash": time.perf_counter() - started,
        }
        hashes.update(source_hashes)
        for path, digest in hashes.items():
            rec.input(path, path, sha256=digest)
        for path in code:
            target = args.output / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if base.sha256(target) != source_hashes[str(path.resolve())]:
                raise RuntimeError("audit code changed before its snapshot")
            rec.output("code:" + str(path.relative_to(REPO)), target, sha256=base.sha256(target))
        output = args.output / "native_support.npz"
        np.savez_compressed(output, **geometry_arrays)
        rec.output("native_support", output, sha256=base.sha256(output))
        for path, digest in hashes.items():
            if base.sha256(path) != digest:
                raise RuntimeError("input or source changed during independent audit")
        rec.param("inputs_and_sources_unchanged", True)
        output = args.output / "summary.json"
        output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        rec.output("summary", output, sha256=base.sha256(output))
    print(json.dumps({"output": str(args.output), "counts": measurements}), flush=True)
    return result


if __name__ == "__main__":
    main()

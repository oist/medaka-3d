"""Post-inference evaluation of proposed IDs on immutable, already supported geometry.

Female annotations are first opened here, after producer/input/output verification.
No renderer or tracker is run. Motion, exact DP and geometry export are replayed
on CPU; image emissions are checked as sealed observations, not re-rendered.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import shutil
import sys
from pathlib import Path

import evaluate_minute as shared
import numpy as np
import run_experiment as base
from audit_appearance_enrollment import exact_array
from identity_dp import build_adjacent_motion_cost, solve_identity_dp
from identity_inputs import load_identity_inputs
from identity_output import build_identity_output

from medaka3d.provenance import provenance

STAGE = "three_body_fixed_geometry_identity_dp"
OUTPUTS = {
    "summary": "summary.json", "identity": "identity.npz", "dp": "dp.npz",
    "scores": "scores.npz", "prefix_audit": "prefix_audit.json", "progress": "progress.jsonl",
    "source_manifest": "source_manifest.json",
}
FIXED_CONFIG = {
    "emission": {"maximum_samples": 512, "width_ratio": .06, "active_views": ["top_cam"]},
    "pixel": {"sigma": .1, "background_cost": 2., "outlier_fraction": .1},
    "training_frames": 166, "appearance_weight": 1., "motion_scale_mm": .8,
    "motion_weight": .2, "initial_state": 0, "tie_atol": 1e-10, "tie_rtol": 1e-12,
}
REQUIRED_SOURCE = {
    f"labs/20260909_three_body/{name}" for name in (
        "run_identity_dp.py", "identity_dp.py", "identity_output.py", "identity_inputs.py",
        "identity_emissions.py", "appearance_ranking.py", "pixel_appearance_loss.py",
        "pixel_template_assignment.py", "renderer.py", "observations.py", "prepare_sequence.py",
        "run_experiment.py", "IDENTITY_DP_PROTOCOL.md",
        "audit_appearance_enrollment.py", "run_appearance_tracking.py",
        "run_persistent_node_recovery.py", "appearance_diagnostic.py", "chunk_masks.py",
    )
} | {"src/medaka3d/calib.py", "src/medaka3d/types.py", "src/medaka3d/provenance.py"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def equal_json(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        b, sort_keys=True, allow_nan=False
    )


def local_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else base.REPO / path).resolve(strict=True)
    if not path.is_relative_to(base.REPO.resolve()):
        raise ValueError("identity evaluation requires preserved workspace evidence")
    return path


def npz(path):
    with np.load(path, allow_pickle=False) as saved:
        if len(saved.files) != len(set(saved.files)):
            raise ValueError("duplicate NPZ keys")
        values = {key: saved[key] for key in saved.files}
    if any(value.dtype.hasobject for value in values.values()):
        raise ValueError("object arrays are forbidden")
    return values


def scalar_json(value):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def output_binding(record, role, path, hashes):
    path = local_path(path)
    item = record.get("outputs", {}).get(role)
    if item is None or local_path(item["path"]) != path:
        raise ValueError(f"missing or wrong sealed output path: {role}")
    digest = base.sha256(path)
    if digest != item.get("sha256"):
        raise ValueError(f"sealed output SHA mismatch: {role}")
    hashes[str(path)] = digest


def verify_sources(run, record, summary, manifest):
    """Use the producer's declared source set, never its dynamic code_paths function."""
    code = {
        base.safe_relative(key).as_posix(): value
        for key, value in record["params"]["code_hashes"].items()
    }
    if (len(code) != len(record["params"]["code_hashes"])
            or not REQUIRED_SOURCE.issubset(code)):
        raise ValueError("producer source set is duplicated or lacks a required dependency")
    for name in ("code_sha256_before", "code_sha256_after"):
        normalized = {base.safe_relative(k).as_posix(): v for k, v in summary[name].items()}
        if normalized != code or len(normalized) != len(summary[name]):
            raise ValueError("producer source hashes changed or are not bound to its summary")
    normalized_manifest = {base.safe_relative(k).as_posix(): v for k, v in manifest.items()}
    if len(normalized_manifest) != len(manifest) or set(normalized_manifest) != set(code):
        raise ValueError("source manifest set differs from the recorded producer sources")
    hashes = {}
    for name, digest in code.items():
        entry = normalized_manifest[name]
        expected = (Path("code_snapshot") / name).as_posix()
        if (entry.get("sha256") != digest
                or base.safe_relative(entry["snapshot"]).as_posix() != expected):
            raise ValueError("source snapshot path/digest is not consistently recorded")
        snapshot = base.resolve_input(run, expected)
        current = base.resolve_input(base.REPO, name)
        # The digest-bearing manifest is not a substitute for the sealed output binding.
        roles = [role for role in record.get("outputs", {})
                 if base.safe_relative(role).as_posix() == expected]
        if len(roles) != 1:
            raise ValueError("source snapshot lacks a unique sealed output role")
        output_binding(record, roles[0], snapshot, hashes)
        if base.sha256(snapshot) != digest or base.sha256(current) != digest:
            raise ValueError("source snapshot or live replay dependency changed")
        hashes[str(snapshot)], hashes[str(current)] = digest, digest
    return hashes


def verify_completed(run):
    run = local_path(run)
    record_path = run / "experiment.provenance.json"
    record = read_json(record_path)
    if record.get("status") != "ok" or record.get("stage") != STAGE:
        raise ValueError("identity inference is not a completed registered stage")
    params = record["params"]
    if local_path(params["output"]) != run:
        raise ValueError("producer output directory is not its recorded directory")
    for key, value in (("evaluation_data_opened_during_inference", False),
                       ("code_unchanged_during_run", True), ("inputs_unchanged_during_run", True)):
        if params.get(key) is not value:
            raise ValueError(f"producer lacks required completion declaration: {key}")
    hashes = {str(record_path): base.sha256(record_path)}
    for role, name in OUTPUTS.items():
        output_binding(record, role, run / name, hashes)
    summary = read_json(run / "summary.json")
    if (not equal_json(params["config"], FIXED_CONFIG)
            or not equal_json(summary["config"], FIXED_CONFIG)):
        raise ValueError("evaluation requires the registered fixed identity configuration")
    if params["package_versions"].get("numpy") != np.__version__:
        raise ValueError("exact CPU DP replay requires the recorded NumPy version")
    for key, value in (
        ("protocol", STAGE), ("technical_completion", True), ("native_fps", 166),
        ("geometry_initialization_count", 1), ("external_reset_count", 0),
        ("identity_initial_state", 0), ("geometry_fits_in_this_stage", 0),
        ("temporal_subsampling", False), ("raw_geometry_and_review_preserved", True),
        ("evaluation_data_opened_during_inference", False),
        ("code_unchanged_during_run", True), ("inputs_unchanged_during_run", True),
    ):
        if not equal_json(summary.get(key), value):
            raise ValueError(f"identity summary violates registered completion: {key}")
    hashes.update(verify_sources(run, record, summary, read_json(run / "source_manifest.json")))
    context = load_identity_inputs(
        local_path(params["geometry_run"]), local_path(params["enrollment_run"]),
        local_path(params["enrollment_audit"]),
    )
    frames = context.frame_idx
    bounds = [int(frames[0]), int(frames[-1] + 1)]
    if bounds not in ([333519, 333719], [76620, 77286], [76620, 86580]):
        raise ValueError("only the full registered 200/666/9960 native windows may be evaluated")
    if (summary["frames"] != bounds or summary["requested_frames"] != len(frames)
            or summary["returned_frames"] != len(frames)):
        raise ValueError("identity output does not cover its entire registered geometry interval")
    recorded = {}
    for key, item in record["inputs"].items():
        path = local_path(item["path"])
        if not Path(key).is_absolute() or local_path(key) != path or str(path) in recorded:
            raise ValueError("producer inputs must be unique absolute-path records")
        recorded[str(path)] = item["sha256"]
    if recorded != context.verified_hashes:
        raise ValueError("identity producer inputs differ from independently verified sources")
    for field in ("input_sha256_before", "input_sha256_after"):
        mapping = {str(local_path(key)): value for key, value in summary[field].items()}
        if mapping != recorded or len(mapping) != len(summary[field]):
            raise ValueError("identity input hashes are not bound before and after inference")
    if not equal_json(read_json(run / "prefix_audit.json"), context.prefix_audit):
        raise ValueError("recorded prefix audit differs from independently verified transfer")
    hashes.update(context.verified_hashes)
    return summary, context, hashes


def validate_saved_solution(geometry, enrollment, identity, dp, scores, summary):
    """Replay the mathematical objective/export; no images, labels, or GPU are used."""
    config, frames = summary["config"], geometry["frame_idx"]
    n = len(frames)
    shapes = {
        "frame_idx": (n,), "unary": (n, 6), "per_view_costs": (n, 6, 3),
        "enabled_views": (n, 3), "sampled_pixels": (n, 3), "valid_pixels": (n, 3),
        "available_core_pixels": (n, 3), "visible_mass": (n, 3, 3),
        "rendering_performed": (n,), "transitions": (n-1, 6, 6),
        "supported_pairs": (n-1, 6, 6, 3), "supported_pair_count": (n-1, 6, 6),
        "squared_distance_mm2": (n-1, 6, 6, 3),
    }
    if set(scores) != set(shapes):
        raise ValueError("saved score NPZ has unknown or missing arrays")
    for key, shape in shapes.items():
        if scores[key].shape != shape or scores[key].dtype.hasobject:
            raise ValueError(f"saved score array shape/type mismatch: {key}")
    exact_array(scores["frame_idx"], frames, "scores native frames")
    for key in ("enabled_views", "rendering_performed", "supported_pairs"):
        if scores[key].dtype != bool:
            raise ValueError(f"score field must be boolean: {key}")
    for key in ("sampled_pixels", "valid_pixels", "available_core_pixels", "supported_pair_count"):
        if scores[key].dtype != np.int64 or np.any(scores[key] < 0):
            raise ValueError(f"score field must contain nonnegative integer counts: {key}")
    for key in ("unary", "per_view_costs", "visible_mass", "transitions"):
        if scores[key].dtype != np.float64 or not np.isfinite(scores[key]).all():
            raise ValueError(f"registered scores must be finite float64: {key}")
        if np.any(scores[key] < 0):
            raise ValueError(f"registered identity cost/mass cannot be negative: {key}")
    samples, valid, available = (scores[k] for k in (
        "sampled_pixels", "valid_pixels", "available_core_pixels"
    ))
    if (np.any(valid > samples) or np.any(samples > available)
            or np.any(samples > config["emission"]["maximum_samples"])):
        raise ValueError("saved image sampling counts are inconsistent")
    permitted = np.array([view in config["emission"]["active_views"] for view in base.VIEWS])
    permitted &= enrollment["enabled_views"]
    if (np.any(samples[:, ~permitted]) or np.any(scores["enabled_views"][:, ~permitted])
            or not np.array_equal(scores["enabled_views"], valid > 0)
            or not np.array_equal(scores["rendering_performed"], samples.any(axis=1))):
        raise ValueError("image scores used an unavailable view or unrecorded rendering")
    prefix = config["training_frames"]
    for key in shapes.keys() - {
        "frame_idx", "transitions", "supported_pairs", "supported_pair_count",
        "squared_distance_mm2",
    }:
        if np.any(scores[key][:prefix] != 0):
            raise ValueError("the enrollment prefix cannot provide identity color emissions")
    if np.any(scores["visible_mass"].sum(axis=2) > valid + 1e-9):
        raise ValueError("visible body mass exceeds valid observed pixel mass")
    if np.any(scores["per_view_costs"] * (~scores["enabled_views"])[:, None] != 0):
        raise ValueError("disabled views must have neutral color costs")
    mass = valid.sum(axis=1)
    expected_unary = config["appearance_weight"] * (
        scores["per_view_costs"] * valid[:, None]
    ).sum(axis=2) / np.maximum(mass, 1)[:, None]
    if not np.allclose(scores["unary"], expected_unary, rtol=1e-12, atol=1e-12):
        raise ValueError("unary is not the shared valid-pixel mean of its view costs")
    motion = build_adjacent_motion_cost(
        frames, geometry["xyz"], geometry["observed"][:, :, 1],
        scale_mm=config["motion_scale_mm"], weight=config["motion_weight"],
    )
    for key in ("transitions", "supported_pairs", "supported_pair_count", "squared_distance_mm2"):
        exact_array(scores[key], getattr(motion, key), key)
    solved = solve_identity_dp(
        frames, scores["unary"], motion.transitions, initial_state=config["initial_state"],
        tie_atol=config["tie_atol"], tie_rtol=config["tie_rtol"],
    )
    if solved.status != "optimal":
        raise ValueError("a completed registered identity result must have a feasible path")
    values = {field.name: getattr(solved, field.name) for field in dataclasses.fields(solved)}
    expected_dp = {key: value for key, value in values.items() if isinstance(value, np.ndarray)}
    if set(dp) != set(expected_dp):
        raise ValueError("DP NPZ does not match the result dataclass array schema")
    for key, value in expected_dp.items():
        exact_array(dp[key], value, "DP/" + key)
    scalars = {key: scalar_json(value) for key, value in values.items() if key not in expected_dp}
    if not equal_json(summary["dp"], scalars):
        raise ValueError("DP scalar metadata differs from the replayed objective")
    for key, value in solution_totals(solved, scores).items():
        if not equal_json(summary.get(key), value):
            raise ValueError(f"producer summary total differs from saved arrays: {key}")
    expected_identity = build_identity_output(geometry, solved.permutation_by_id)
    if set(identity) != set(expected_identity):
        raise ValueError("identity export has unknown or missing arrays")
    for key, value in expected_identity.items():
        exact_array(identity[key], value, "identity/" + key)
    count = conservation_counts(geometry, identity)
    return solved, count


def solution_totals(solved, scores):
    pairs = scores["supported_pair_count"][
        np.arange(len(solved.frame_idx) - 1), solved.path_state[:-1], solved.path_state[1:]
    ]
    return {
        "scored_frames": int(scores["enabled_views"].any(axis=1).sum()),
        "sampled_pixels": scores["sampled_pixels"].sum(axis=0).tolist(),
        "mapping_changes": int(np.count_nonzero(np.diff(solved.path_state))),
        "nonidentity_mapping_frames": int(np.count_nonzero(solved.path_state)),
        "selected_motion_supported_pair_histogram": np.bincount(pairs, minlength=4).tolist(),
    }


def motion_only_solution(geometry, scores, config):
    """Registered color ablation on exactly the same frozen adjacent transitions."""
    solved = solve_identity_dp(
        geometry["frame_idx"], np.zeros_like(scores["unary"]), scores["transitions"],
        initial_state=config["initial_state"], tie_atol=config["tie_atol"],
        tie_rtol=config["tie_rtol"],
    )
    if solved.status != "optimal":
        raise ValueError("the registered motion-only reference must be feasible")
    output = build_identity_output(geometry, solved.permutation_by_id)
    conservation_counts(geometry, output)
    return solved, output


def split_solution(solved):
    values = {field.name: getattr(solved, field.name) for field in dataclasses.fields(solved)}
    arrays = {key: value for key, value in values.items() if isinstance(value, np.ndarray)}
    scalars = {key: scalar_json(value) for key, value in values.items() if key not in arrays}
    return arrays, scalars


def proposed_geometry(frames, identity):
    return dict(
        frame_idx=frames, sources=identity["proposed_sources"],
        observed=identity["proposed_observed"], pose_xyz=identity["proposed_pose_xyz"],
        identity_review=identity["prior_review_for_proposed_identity"],
        xyz=identity["proposed_xyz"], accepted_xyz=identity["conservative_accepted_xyz"],
    )


def conservation_counts(geometry, identity):
    raw, proposed = geometry["observed"], identity["proposed_observed"]
    raw_accepted = np.isfinite(geometry["accepted_xyz"]).all(-1)
    proposed_accepted = np.isfinite(identity["conservative_accepted_xyz"]).all(-1)
    for a, b, label in (
        (raw.sum(axis=1), proposed.sum(axis=1), "per-frame node support"),
        (raw_accepted.sum(axis=1), proposed_accepted.sum(axis=1),
         "per-frame old-review acceptance"),
        (geometry["identity_review"].sum(axis=1),
         identity["prior_review_for_proposed_identity"].sum(axis=1), "per-frame old review"),
    ):
        if not np.array_equal(a, b):
            raise ValueError(f"fixed-geometry identity output violated {label} conservation")
    return {
        "direct_node_counts_mouth_pectoral_tail": raw.sum(axis=(0, 1)).tolist(),
        "direct_pectoral_fish_frames": int(raw[:, :, 1].sum()),
        "conservative_accepted_fish_frames": int(raw_accepted.sum()),
        "old_review_fish_frames": int(geometry["identity_review"].sum()),
        "per_native_frame_counts_conserved": True,
        "geometry_unchanged_by_construction_is_not_accuracy_improvement": True,
    }


def female_regression_gate(raw, proposed):
    """A relative endpoint guard, not certification that either trajectory is correct."""
    if len(raw["anchors"]) != len(proposed["anchors"]):
        raise ValueError("the same requested female anchors must be evaluated")
    for a, b in zip(raw["anchors"], proposed["anchors"], strict=True):
        if (a["frame"] != b["frame"] or a["reason"] != b["reason"]
                or (a["identity"] is None) != (b["identity"] is None)):
            raise ValueError("pure relabeling changed female anchor support/availability")
    if len(raw["intervals"]) != len(proposed["intervals"]):
        raise ValueError("female interval tables must have identical requested endpoints")
    regressions, improvements = [], []
    for a, b in zip(raw["intervals"], proposed["intervals"], strict=True):
        if (a["start"], a["stop"]) != (b["start"], b["stop"]):
            raise ValueError("female interval endpoints differ")
        if (a["outcome"] == "unscored") != (b["outcome"] == "unscored"):
            raise ValueError("pure relabeling changed female interval scorable status")
        if a["outcome"] == "preserved" and b["outcome"] == "swap":
            regressions.append({"start": a["start"], "stop": a["stop"]})
        if a["outcome"] == "swap" and b["outcome"] == "preserved":
            improvements.append({"start": a["start"], "stop": a["stop"]})
    raw_swaps = sum(row["outcome"] == "swap" for row in raw["intervals"])
    proposed_swaps = sum(row["outcome"] == "swap" for row in proposed["intervals"])
    counts = {
        "raw_swap_intervals": raw_swaps, "proposed_swap_intervals": proposed_swaps,
        "preserved_to_swap_intervals": regressions, "swap_to_preserved_intervals": improvements,
    }
    evaluable = any(row["outcome"] != "unscored" for row in raw["intervals"])
    passed = None if not evaluable else not regressions and proposed_swaps <= raw_swaps
    return {
        **counts, "passed": passed,
        "status": "unscored" if passed is None else "pass" if passed else "fail",
        "no_previously_preserved_interval_became_swap": not regressions,
        "swap_interval_count_did_not_increase": proposed_swaps <= raw_swaps,
        "scope": "relative female endpoint guard; no dense, male, or returning-swap validation",
    }


def code_paths():
    paths = {Path(__file__).resolve(), base.LAB / "IDENTITY_DP_PROTOCOL.md"}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(base.LAB) or path.is_relative_to(base.REPO / "src")
            ):
                paths.add(path)
    return sorted(paths)


def write_report(path, report):
    lines = [
        "Fixed-geometry identity comparison\n",
        "All methods retain the same physical geometry, direct support and prior reviews. "
        "Only the proposed identity mapping changes.\n",
        "| Method | Female raw preserved / swap / unscored | "
        "Conservative accepted preserved / swap / unscored |",
        "|---|---|---|",
    ]
    for label, method in report["methods"].items():
        columns = []
        for kind in ("female_raw", "female_accepted"):
            counts = method[kind]["interval_counts"]
            columns.append(" / ".join(str(counts.get(k, 0))
                                      for k in ("preserved", "swap", "unscored")))
        lines.append(f"| {label} | {columns[0]} | {columns[1]} |")
    lines += [
        "", "Joint raw-female endpoint regression gate: "
        + report["raw_female_endpoint_regression_gate"]["status"] + ".",
        "Motion-only is the preregistered zero-color-cost reference with the same "
        "transitions and initial state. No geometry, weights or templates were refitted.",
        "Female endpoints do not establish male identity, dense identity, or the absence "
        "of exchanges followed by a return between anchors. Deterministic tied paths and "
        "DP gaps are not identity certification. Original review flags remain in force.",
        "Color costs were structurally checked; this evaluation did not re-render them. "
        "Full arrays, endpoint tables and provenance are saved with this report.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if (not args.output.is_relative_to(base.REPO.resolve())
            or args.output.exists() or args.output.is_symlink()):
        raise FileExistsError("refusing to overwrite identity evaluation")
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance("three_body_identity_dp_post_inference_evaluation", args.output,
                    name="evaluation") as record:
        record.params_from(args)
        code = {str(path): base.sha256(path) for path in code_paths()}
        summary, context, hashes = verify_completed(args.run)
        run = local_path(args.run)
        identity, dp, scores = (npz(run / f"{role}.npz") for role in ("identity", "dp", "scores"))
        solved, counts = validate_saved_solution(
            context.geometry, context.enrollment, identity, dp, scores, summary
        )
        # Fixed zero-unary ablation is also solved before any evaluation annotations are opened.
        motion_dp, motion_identity = motion_only_solution(
            context.geometry, scores, summary["config"]
        )
        motion_arrays, motion_scalars = split_solution(motion_dp)
        for role, arrays in (("motion_only_dp", motion_arrays),
                             ("motion_only_identity", motion_identity)):
            path = args.output / f"{role}.npz"
            np.savez_compressed(path, **arrays)
            record.output(role, path, sha256=base.sha256(path))
        # Evaluation-only boundary: independent female annotations are first opened below.
        catalog, _ = base.staging_catalog(context.input_root)
        _, _, registered = base.registered_seed(int(context.frame_idx[0]))
        gt_path = base.verified_input(
            context.input_root, registered["female_annotations_relative"], catalog
        )
        if base.sha256(gt_path) != registered["female_annotations_sha256"]:
            raise ValueError("female annotations differ from the registered evaluator input")
        hashes[str(gt_path)] = base.sha256(gt_path)
        nodes = shared.CachedNodes(context.input_root, base.EXPERIMENT, context.frame_idx)
        with gt_path.open(encoding="utf-8-sig", newline="") as stream:
            anchors = shared.match_mouth_anchors(
                csv.DictReader(stream), lambda view, frame: nodes.at(view, frame)[:, 0],
                int(context.frame_idx[0]), int(context.frame_idx[-1] + 1),
            )
        geometry = context.geometry
        methods = {}
        for label, arrays in (
            ("frozen_geometry_ids", geometry),
            ("proposed_ids", proposed_geometry(context.frame_idx, identity)),
            ("motion_only_ids", proposed_geometry(context.frame_idx, motion_identity)),
        ):
            source, review, observed, xyz, accepted = shared.normalized_points(arrays)
            shared.validate_sources(source, context.frame_idx, nodes)
            methods[label] = shared.common_metrics(
                context.frame_idx, source, review, observed, xyz, accepted, anchors
            )
        raw, changed = methods["frozen_geometry_ids"], methods["proposed_ids"]
        gate = female_regression_gate(raw["female_raw"], changed["female_raw"])
        accepted_gate = female_regression_gate(raw["female_accepted"], changed["female_accepted"])
        motion_gate = female_regression_gate(
            raw["female_raw"], methods["motion_only_ids"]["female_raw"]
        )
        report = {
            "status": "technical_post_inference_comparison", "frames": summary["frames"],
            "native_fps": 166, "native_frames": len(context.frame_idx), "methods": methods,
            "raw_female_endpoint_regression_gate": gate,
            "conservative_female_endpoint_diagnostic": accepted_gate,
            "motion_only_raw_female_endpoint_regression_gate": motion_gate,
            "motion_only_reference": {
                "dp": motion_scalars, **solution_totals(motion_dp, scores),
                "unary": "all zero", "appearance_scored_frames_in_reference": 0,
                "same_transitions": True, "same_initial_state": True,
                "computed_before_reading_evaluation_labels": True,
                "note": "sampling totals describe the joint input, not a new motion-only rendering",
            },
            "conservation": counts, "geometry_run": str(context.geometry_root),
            "identity_run": str(run), "identity_npz_sha256": hashes[str(run / "identity.npz")],
            "geometry_npz_sha256": hashes[str(context.geometry_root / "candidate.npz")],
            "motion_dp_export_replayed_exactly": True, "color_emissions_rerendered": False,
            "original_review_cleared": False, "independent_all_three_truth_loaded": False,
            "dp_global_alternative_gap": scalar_json(solved.global_alternative_gap),
            "dp_global_alternative_status": solved.global_alternative_status,
            "limitations": [
                "Equal geometry/support counts are a construction invariant, not better accuracy.",
                "Old reviews accompany the chosen geometry; proposed IDs are not newly certified.",
                "A gate failure prevents promotion; passing cannot establish swap-free IDs.",
                "Female endpoints cannot verify males, dense IDs, or exchanges that later return.",
                "Image costs were sealed and structurally checked, not re-rendered here.",
                "Missing transitions can avoid motion cost; DP gaps are not probabilities.",
                "Frozen duplicated or missing physical bodies cannot be repaired by a permutation.",
            ],
        }
        for path, digest in {**hashes, **code}.items():
            record.input(path, path, sha256=digest)
        for path, digest in code.items():
            target = args.output / "code_snapshot" / Path(path).relative_to(base.REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if base.sha256(target) != digest:
                raise RuntimeError("evaluation source changed during snapshot")
            record.output("code:" + str(Path(path).relative_to(base.REPO)), target, sha256=digest)
        if any(base.sha256(path) != digest for path, digest in {**hashes, **code}.items()):
            raise RuntimeError("identity evaluation input or source changed")
        record.param("inputs_and_code_unchanged", True)
        output = args.output / "comparison.json"
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        record.output("comparison", output, sha256=base.sha256(output))
        markdown = args.output / "RESULT.md"
        write_report(markdown, report)
        record.output("report", markdown, sha256=base.sha256(markdown))
        all_outputs = {entry["path"]: entry["sha256"] for entry in record.outputs.values()}
        if any(base.sha256(path) != digest
               for path, digest in {**hashes, **code, **all_outputs}.items()):
            raise RuntimeError("input, source, snapshot or output changed before final seal")
        record.param("final_inputs_sources_snapshots_outputs_unchanged", True)
    print(json.dumps({"output": str(args.output), "female_endpoint_gate": gate["status"],
                      "identity_verdict": "unverified"}), flush=True)
    return report


if __name__ == "__main__":
    main()

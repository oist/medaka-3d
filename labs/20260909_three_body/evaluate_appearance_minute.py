"""Compare a sealed appearance minute with persistent/v1/older tracking and seven C34 arms.

Metrics and review-event definitions are reused unchanged from evaluate_minute.
The explicit appearance verifier extends the copied absolute-path evidence
contract with frozen online enrollment records. It does not mutate an existing
evaluator's arm table or confuse the online registration with final traceback.
Recorded-feature arithmetic is audited here; raw image recomputation is separate.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import evaluate_minute as original
import evaluate_variant_minute as persistent_evaluation
import numpy as np
import run_appearance_tracking as appearance_runner
import run_experiment as base
import run_persistent_node_recovery as persistent_runner
from appearance_diagnostic import EROSION_RADIUS, FEATURES, MIN_CORE_PIXELS

from medaka3d.provenance import provenance

REPO = base.REPO
LAB = base.LAB
VARIANTS = {
    appearance_runner.ALGORITHM_ARM: appearance_runner,
}
VERIFIER_ORIGIN_SHA256 = "1ec2c14fd7f7d93525c00b84deee1b42008af741692c03ff1cbc1c30a40eaa88"
APPEARANCE_ARRAYS = set(appearance_runner.ORIGINAL_ARRAYS) | {
    "node_missing",
    "direct_pectoral_age",
    "geometric_cost",
    "appearance_cost",
    "appearance_enabled",
    "appearance_pixels",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exact_json(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        b, sort_keys=True, allow_nan=False
    )


def local_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else REPO / path).resolve(strict=True)
    if not path.is_relative_to(REPO.resolve()):
        raise ValueError("variant evaluation requires preserved local workspace inputs")
    return path


def normalized_hashes(values):
    result = {}
    for name, digest in values.items():
        path = local_path(name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("invalid recorded SHA256")
        key = str(path)
        if key in result and result[key] != digest:
            raise ValueError("conflicting resolved SHA256 records")
        result[key] = digest
    return result


def code_hashes(values):
    result = {}
    for name, digest in values.items():
        relative = base.safe_relative(name).as_posix()
        if relative in result:
            raise ValueError("duplicate normalized snapshot path")
        result[relative] = digest
    return result


def verify_appearance_evidence(root, summary, *, expected_frames=None, include_post_inference=True):
    """Verify explicit appearance evidence; synthetic short windows exercise the contract.

    The public comparison CLI still requires the registered full minute through the
    unchanged original.load_completed gate. This helper accepts an explicit native
    frame array for tests. This function does not load enrollment; verify_enrollment
    additionally binds and validates those two outputs against the actual arrays.
    Image-only enrollment audits may explicitly omit post-inference GT/reference
    file reads. The inference input set remains complete in either mode.
    """
    root = local_path(root)
    record_path = root / "experiment.provenance.json"
    record = load_json(record_path)
    if record.get("status") != "ok":
        raise ValueError("variant provenance is not complete")
    params = record["params"]
    arm = summary.get("algorithm_arm")
    if arm not in VARIANTS or params.get("algorithm_arm") != arm:
        raise ValueError("unknown or inconsistently bound algorithm arm")
    if (
        summary.get("technical_completion") is not True
        or summary.get("inputs_unchanged_during_run") is not True
        or summary.get("code_unchanged_during_run") is not True
        or params.get("inputs_unchanged_during_run") is not True
        or params.get("code_unchanged_during_run") is not True
    ):
        raise ValueError("variant lacks successful unchanged-input/code evidence")
    before = normalized_hashes(summary["input_sha256_before"])
    after = normalized_hashes(summary["input_sha256_after"])
    if before != after:
        raise ValueError("variant inputs changed during inference")
    hashes, provenance_inference = {}, {}
    for key, item in record["inputs"].items():
        if key.startswith("post_inference:") and not include_post_inference:
            continue
        path = local_path(item["path"])
        digest = base.sha256(path)
        if digest != item.get("sha256"):
            raise ValueError("variant run input digest mismatch")
        hashes[str(path)] = digest
        if key != "seed_registration" and not key.startswith("post_inference:"):
            # Do not accept a role alias or post-inference record as an inference record.
            if not Path(key).is_absolute() or local_path(key) != path:
                raise ValueError("variant inference key is not its actual absolute input path")
            provenance_inference[str(path)] = digest
    if provenance_inference != before:
        raise ValueError("summary inference inputs are not exactly bound to provenance inputs")
    for name, filename in (
        ("summary", "summary.json"),
        ("candidate", "candidate.npz"),
        ("seed_registration", "seed_registration.json"),
    ):
        path = (root / filename).resolve(strict=True)
        item = record["outputs"][name]
        if local_path(item["path"]) != path or base.sha256(path) != item.get("sha256"):
            raise ValueError("variant output path or digest is not bound to its seal")
        hashes[str(path)] = item["sha256"]
    if not exact_json(load_json(root / "summary.json"), summary):
        raise ValueError("provided summary is not the sealed summary")
    source_root = local_path(params["sequence"])
    manifest_path = (source_root / "manifest.json").resolve(strict=True)
    manifest_digest = base.sha256(manifest_path)
    if before.get(str(manifest_path)) != manifest_digest:
        raise ValueError("actual sequence manifest was not an inference input")
    copied_manifest = root / "sequence_manifest.json"
    if base.sha256(copied_manifest) != manifest_digest:
        raise ValueError("saved sequence snapshot differs from actual inference manifest")
    hashes[str(copied_manifest)] = manifest_digest
    manifest = load_json(manifest_path)
    frames = base.sequence_frames(manifest)
    expected = frames if expected_frames is None else np.asarray(expected_frames)
    if not np.array_equal(frames, expected) or summary["frames"] != [
        int(frames[0]),
        int(frames[-1] + 1),
    ]:
        raise ValueError("manifest native frame range differs from evaluated output")
    if (
        summary.get("source_sequence_frames") != summary["frames"]
        or summary.get("requested_frames") != len(frames)
        or summary.get("returned_frames") != len(frames)
        or summary.get("native_fps") != 166
        or summary.get("temporal_subsampling") is not False
        or summary.get("development_prefix_smoke") is not False
        or summary.get("initialization_count") != 1
        or summary.get("track_invocation_count") != 1
        or summary.get("reset_count") != 0
        or manifest.get("image_shape_hw") != [1200, 1200]
    ):
        raise ValueError("variant is not a full native single-initialization window")
    required_inference = {str(manifest_path)}
    for category in ("raw", "masks", "backgrounds"):
        if set(manifest[category]) != set(base.VIEWS):
            raise ValueError("sequence does not contain exactly the three registered views")
        for view in base.VIEWS:
            entry = manifest[category][view]
            path = base.resolve_input(source_root, entry["path"])
            digest = entry["sha256"]
            if before.get(str(path)) != digest or hashes.get(str(path)) != digest:
                raise ValueError("manifest asset is not bound to the actual inference input")
            required_inference.add(str(path))
            if category == "raw" and len(entry.get("frame_bgr_sha256", [])) != len(frames):
                raise ValueError("raw input lacks every native frame's recorded pixel SHA")
    input_root = local_path(params["input_root"])
    staging_path = (input_root / "staging_manifest.json").resolve(strict=True)
    required_inference.add(str(staging_path))
    if before.get(str(staging_path)) != base.sha256(root / "staging_manifest.json"):
        raise ValueError("saved staging catalog is not the inference catalog")
    hashes[str(root / "staging_manifest.json")] = before[str(staging_path)]
    for filename in (
        "calibration_manifest.json",
        *(f"cam{i}_bundle_refractive.json" for i in range(3)),
    ):
        source = (input_root / base.CALIBRATION_DIR / filename).resolve(strict=True)
        required_inference.add(str(source))
        copied = root / "calibration_snapshot" / filename
        if before.get(str(source)) != base.sha256(copied):
            raise ValueError("saved calibration is not the actual inference calibration")
        hashes[str(copied)] = before[str(source)]
    for view in base.VIEWS:
        path = (
            input_root / f"work/c01_s16_predictions/bottomup/{base.EXPERIMENT}__{view}.npz"
        ).resolve(strict=True)
        required_inference.add(str(path))
    if set(before) != required_inference:
        raise ValueError("variant inference input set differs from the fixed v1 input contract")
    for name in (
        "tracker_config",
        "node_recovery_config",
        "persistence_config",
        "appearance_config",
        "enrollment_config",
        "pixel_config",
        "seed",
    ):
        if not exact_json(params.get(name), summary.get(name)):
            raise ValueError(f"variant {name} is not bound to the sealed runtime parameters")
    initial, seed, _evaluation = base.registered_seed(int(frames[0]))
    if not exact_json(summary["seed"], seed) or not exact_json(
        load_json(root / "seed_registration.json"), seed
    ):
        raise ValueError("variant does not use the registered first seed")
    registration = local_path(seed["registration_path"])
    if hashes.get(str(registration)) != seed["registration_sha256"]:
        raise ValueError("registered seed source is not bound")
    initial_path = root / "initial_pose_xyz.npy"
    if not np.array_equal(np.load(initial_path, allow_pickle=False), initial):
        raise ValueError("saved initial pose differs from registered seed")
    hashes[str(initial_path)] = base.sha256(initial_path)
    saved_code = code_hashes(summary["code_sha256_before"])
    if not saved_code or saved_code != code_hashes(summary["code_sha256_after"]):
        raise ValueError("variant source changed during inference")
    if saved_code != code_hashes(params["code_hashes"]):
        raise ValueError("snapshot hashes are not bound to runtime provenance parameters")
    required_code = {path.relative_to(base.REPO).as_posix() for path in VARIANTS[arm].code_paths()}
    if set(saved_code) != required_code:
        raise ValueError("variant snapshot source set is incomplete or unexpected")
    for relative, digest in saved_code.items():
        path = base.resolve_input(root / "code_snapshot", relative)
        if base.sha256(path) != digest:
            raise ValueError("variant snapshot digest mismatch")
        hashes[str(path)] = digest
    hashes[str(record_path)] = base.sha256(record_path)
    return hashes


def validate_appearance_arrays(arrays, frames, summary):
    arm = summary.get("algorithm_arm")
    if arm not in VARIANTS:
        raise ValueError("unknown variant algorithm")
    if set(arrays) != APPEARANCE_ARRAYS:
        raise ValueError("appearance must preserve exactly its declared 25 arrays")
    validated = VARIANTS[arm].validate_arm_arrays(arrays, frames)
    if not np.array_equal(validated["frame_idx"], frames):
        raise ValueError("variant does not contain every native frame")
    original.normalized_points(validated)
    ranking = SimpleNamespace(**summary["appearance_config"])
    enrollment = SimpleNamespace(**summary["enrollment_config"])
    appearance_runner.validate_appearance_arrays(validated, ranking, enrollment)
    if np.any(validated["appearance_cost"] < 0):
        raise ValueError("appearance loss cannot be negative")
    return validated


def validate_enrollment_records(snapshot, metadata, frames, result, summary, *, nodes=None):
    """Audit recorded online registration, never substitute final traceback states.

    Median/count arithmetic and selected-node support are independently checked.
    This does not re-extract features, recompute masks or certify the slot identities.
    """
    config = summary["enrollment_config"]
    if set(config) != {"training_frames", "minimum_template_samples"} or any(
        type(v) is not int or v < 1 for v in config.values()
    ):
        raise ValueError("invalid enrollment configuration")
    count, minimum = config["training_frames"], config["minimum_template_samples"]
    if count > 166 or minimum > count or len(frames) < count:
        raise ValueError("invalid or incomplete enrollment prefix")
    shapes = {
        "frame_idx": ((count,), "int64"),
        "features": ((count, 3, 3, 6), "float64"),
        "instance_rows": ((count, 3, 3), "int64"),
        "core_pixels": ((count, 3, 3), "int64"),
        "online_pose_xyz": ((count, 3, 3, 3), "float64"),
        "online_sources": ((count, 3, 3, 3), "int64"),
        "online_observed": ((count, 3, 3), "bool"),
        "online_identity_review": ((count, 3), "bool"),
        "counts": ((3, 3), "int64"),
        "templates": ((3, 3, 6), "float64"),
        "enabled_views": ((3,), "bool"),
    }
    if set(snapshot) != set(shapes) | {"reasons"}:
        raise ValueError("enrollment NPZ must contain exactly twelve declared arrays")
    for name, (shape, dtype) in shapes.items():
        value = snapshot[name]
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
            raise ValueError(f"invalid enrollment shape or dtype: {name}")
    reasons = snapshot["reasons"]
    if reasons.shape != (count, 3, 3) or reasons.dtype.kind != "U":
        raise ValueError("enrollment reasons must be a non-object Unicode array")
    if not np.isin(
        reasons,
        [
            "usable",
            "view_unavailable",
            "image_unavailable",
            "identity_review",
            "no_direct_mouth_pect",
            "no_masks",
            "projection_invalid",
            "no_joint_instance",
            "multiple_joint_instances",
            "unknown_latent_competitor_projection",
            "mask_contains_other_projected_node",
            "core_too_small",
        ],
    ).all():
        raise ValueError("unknown or incomplete enrollment reason")
    if not exact_json(metadata.get("config"), config):
        raise ValueError("enrollment metadata configuration differs from the runtime")
    for name, expected in (
        ("views", list(base.VIEWS)),
        ("feature_names", list(FEATURES)),
        ("erosion_radius_px", EROSION_RADIUS),
        ("minimum_core_pixels", MIN_CORE_PIXELS),
        ("frozen", True),
        ("training_frames_seen", count),
        ("first_frame_idx", int(frames[0])),
        ("last_training_frame_idx", int(frames[count - 1])),
        ("external_templates_or_saved_poses_used", False),
        ("reviewed_slots_registered", False),
        ("retroactive_beam_relabeling", False),
    ):
        if not exact_json(metadata.get(name), expected):
            raise ValueError(f"enrollment metadata inconsistent: {name}")
    appearance_runner.validate_enrollment_audit({**metadata, **snapshot}, frames, count)
    features = snapshot["features"]
    if np.isinf(features).any():
        raise ValueError("enrollment features contain infinity")
    finite = np.isfinite(features)
    usable = finite.all(axis=-1)
    if np.any(finite.any(axis=-1) != usable) or np.any(
        (features[finite] < 0) | (features[finite] > 1)
    ):
        raise ValueError("features must be whole finite descriptors in [0,1] or whole NaN")
    if not np.array_equal(usable, reasons == "usable"):
        raise ValueError("usable reasons and finite registered features disagree")
    rows, core = snapshot["instance_rows"], snapshot["core_pixels"]
    if (
        np.any(rows < -1)
        or np.any(core < 0)
        or np.any(usable & ((rows < 0) | (core < MIN_CORE_PIXELS)))
    ):
        raise ValueError("registered descriptors lack a valid instance and sufficient core")
    for time_rows in rows:
        for view_rows in time_rows:
            assigned = view_rows[view_rows >= 0]
            if len(assigned) != len(np.unique(assigned)):
                raise ValueError("one D2 instance was registered for multiple slots")
    support = snapshot["online_observed"]
    sources = snapshot["online_sources"]
    review = snapshot["online_identity_review"]
    if np.any(support & ((sources >= 0).sum(axis=2) < 2)):
        raise ValueError("online directly supported nodes lack two source views")
    direct_both = support[:, :, :2].all(axis=-1)
    per_view_sources = (sources[..., :2] >= 0).all(axis=-1).transpose(0, 2, 1)
    allowed = direct_both[:, None] & per_view_sources & ~review[:, None]
    if np.any(usable & ~allowed):
        raise ValueError("registered features violate online review/direct support exclusion")
    for time_sources in sources:
        for view in range(3):
            for node in range(3):
                held = time_sources[:, view, node]
                held = held[held >= 0]
                if len(held) != len(np.unique(held)):
                    raise ValueError("online registration source rows are not exclusive")
    if nodes is not None:
        original.validate_sources(sources, frames[:count], nodes)
    counts = usable.sum(axis=0)
    if not np.array_equal(snapshot["counts"], counts) or not exact_json(
        metadata.get("counts"), counts.tolist()
    ):
        raise ValueError("enrollment counts differ from the recorded feature samples")
    enabled = (counts >= minimum).all(axis=1)
    if not np.array_equal(snapshot["enabled_views"], enabled) or not exact_json(
        metadata.get("enabled_views"), enabled.tolist()
    ):
        raise ValueError("enrollment views require all three sufficiently sampled slots")
    templates = np.full((3, 3, 6), np.nan)
    for view in np.flatnonzero(enabled):
        for fish in range(3):
            templates[view, fish] = np.median(features[usable[:, view, fish], view, fish], axis=0)
    if not np.array_equal(snapshot["templates"], templates, equal_nan=True):
        raise ValueError("frozen templates are not the recorded prefix medians")
    pixels, scored = metadata.get("sampled_pixels"), metadata.get("scored_frames")
    if (
        not isinstance(pixels, list)
        or len(pixels) != 3
        or any(type(value) is not int or value < 0 for value in pixels)
        or type(scored) is not int
        or scored < 0
    ):
        raise ValueError("appearance totals must be nonnegative integer metadata")
    appearance_runner.validate_appearance_totals(result, pixels, metadata.get("scored_frames"))
    if np.any(result["appearance_enabled"][:, ~enabled]):
        raise ValueError("appearance used a view without complete enrollment")
    return dict(
        training_frames=count,
        online_frame_start=int(frames[0]),
        online_frame_stop=int(frames[count - 1] + 1),
        counts=counts.tolist(),
        enabled_views=enabled.tolist(),
        scored_frames=int(metadata["scored_frames"]),
        sampled_pixels=result["appearance_pixels"].sum(axis=0).tolist(),
        recorded_medians_counts_and_support_consistent=True,
        online_source_row_bounds_checked=nodes is not None,
        online_states_compared_to_final_traceback=False,
        raw_feature_and_mask_recomputation_performed=False,
        identity_labels_verified=False,
    )


def verify_enrollment(root, summary, result, frames, *, nodes=None):
    root = local_path(root)
    record = load_json(root / "experiment.provenance.json")
    hashes = {}
    for key, filename in (
        ("appearance_enrollment", "appearance_enrollment.npz"),
        ("appearance_enrollment_metadata", "appearance_enrollment.json"),
    ):
        path = (root / filename).resolve(strict=True)
        item = record["outputs"].get(key)
        if (
            item is None
            or local_path(item["path"]) != path
            or base.sha256(path) != item.get("sha256")
        ):
            raise ValueError("enrollment output is not bound by path and SHA to the run seal")
        hashes[str(path)] = item["sha256"]
    metadata = load_json(root / "appearance_enrollment.json")
    if (
        not exact_json(metadata, summary.get("appearance_enrollment"))
        or metadata.get("output_sha256") != hashes[str(root / "appearance_enrollment.npz")]
    ):
        raise ValueError("enrollment metadata, summary and NPZ output binding disagree")
    with np.load(root / "appearance_enrollment.npz", allow_pickle=False) as arrays:
        snapshot = {key: arrays[key] for key in arrays.files}
    audit = validate_enrollment_records(snapshot, metadata, frames, result, summary, nodes=nodes)
    return audit, hashes


def same_v1_inputs(variant_root, variant, v1_root, v1, variant_hashes, v1_hashes):
    """Compare only the inference input map; post-inference GT/reference reads stay separate."""
    if not exact_json(variant["seed"], v1["seed"]):
        raise ValueError("variant and original v1 first seed differ")
    if not exact_json(variant["tracker_config"], v1["tracker_config"]):
        raise ValueError("variant and original v1 base tracker configuration differ")
    a = normalized_hashes(variant["input_sha256_before"])
    b = normalized_hashes(v1["input_sha256_before"])
    if a != b or b != normalized_hashes(v1["input_sha256_after"]):
        raise ValueError("variant and original v1 inference paths/SHA differ")
    for path, digest in a.items():
        original.require_binding(path, digest, variant_hashes)
        original.require_binding(path, digest, v1_hashes)
    variant_initial = (Path(variant_root) / "initial_pose_xyz.npy").resolve(strict=True)
    v1_initial = (Path(v1_root) / "initial_pose_xyz.npy").resolve(strict=True)
    initial_hash = base.sha256(variant_initial)
    if initial_hash != base.sha256(v1_initial):
        raise ValueError("initial pose file differs between variant and v1")
    variant_hashes[str(variant_initial)] = initial_hash
    v1_hashes[str(v1_initial)] = initial_hash
    variant_params = load_json(Path(variant_root) / "experiment.provenance.json")["params"]
    v1_params = load_json(Path(v1_root) / "experiment.provenance.json")["params"]
    if not exact_json(variant_params.get("flow_config"), v1_params.get("flow_config")):
        raise ValueError("variant and original v1 image-motion configuration differs")
    return dict(
        same_seed=True,
        same_initial_pose_file_sha256=initial_hash,
        same_base_tracker_config=True,
        same_flow_config=True,
        same_inference_paths_and_sha=True,
    )


def validate_registered_arm(summary):
    """The actual minute comparison cannot silently accept a differently tuned arm."""
    for name, expected in (
        ("appearance_config", dict(weight=1.0, maximum_samples=512, active_views=["top_cam"])),
        ("enrollment_config", dict(training_frames=166, minimum_template_samples=5)),
        ("pixel_config", dict(sigma=0.1, background_cost=2.0, outlier_fraction=0.1)),
    ):
        if not exact_json(summary.get(name), expected):
            raise ValueError(f"appearance minute differs from the registered {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--persistent",
        type=Path,
        required=True,
        help="Completed persistent-mouth minute before appearance reranking",
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=REPO / "work/three_body/v1_continuous_one_minute_detectron2",
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=REPO / "work/generative_tracking/recovery_v2_floor6_continuous_one_minute",
    )
    parser.add_argument("--input-root", type=Path, default=REPO / "work/generative_tracking_inputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    code = [
        Path(__file__),
        LAB / "evaluate_minute.py",
        LAB / "evaluate_variant_minute.py",
        LAB / "diagnostics.py",
        LAB / "review_events.py",
        LAB / "run_experiment.py",
        LAB / "observations.py",
        LAB / "prepare_sequence.py",
        LAB / "run_node_recovery.py",
        LAB / "run_persistent_node_recovery.py",
        LAB / "run_appearance_tracking.py",
        LAB / "appearance_diagnostic.py",
        REPO / "src/medaka3d/calib.py",
        REPO / "src/medaka3d/types.py",
        REPO / "src/medaka3d/provenance.py",
        REPO / "src/medaka3d/eval/generative.py",
        REPO / "src/medaka3d/eval/reacquisition.py",
    ]
    sources = {str(p): base.sha256(p) for p in code}
    # Full-minute gate is deliberately unchanged; no prefix-evaluation option exists.
    current, summary, hashes = original.load_completed(args.run)
    persistent, persistent_summary, persistent_hashes = original.load_completed(args.persistent)
    v1, v1_summary, v1_hashes = original.load_completed(args.original)
    older, older_summary, older_hashes = original.load_completed(args.previous, previous=True)
    if persistent_summary.get("algorithm_arm") != persistent_runner.ALGORITHM_ARM:
        raise ValueError("--persistent must identify the persistent-mouth algorithm")
    validate_registered_arm(summary)
    evidence = verify_appearance_evidence(args.run, summary, expected_frames=original.FRAMES)
    persistent_evidence = persistent_evaluation.verify_variant_evidence(
        args.persistent, persistent_summary, expected_frames=original.FRAMES
    )
    v1_evidence = original.verify_run_evidence(args.original, v1_summary)
    older_evidence = original.verify_run_evidence(args.previous, older_summary, previous=True)
    parity = same_v1_inputs(args.run, summary, args.original, v1_summary, evidence, v1_evidence)
    persistent_parity = same_v1_inputs(
        args.persistent,
        persistent_summary,
        args.original,
        v1_summary,
        persistent_evidence,
        v1_evidence,
    )
    for name in ("node_recovery_config", "persistence_config"):
        if not exact_json(summary[name], persistent_summary[name]):
            raise ValueError(f"appearance and persistent {name} differ")
    hashes.update(
        {
            **persistent_hashes,
            **v1_hashes,
            **older_hashes,
            **evidence,
            **persistent_evidence,
            **v1_evidence,
            **older_evidence,
        }
    )
    current = validate_appearance_arrays(current, original.FRAMES, summary)
    persistent = persistent_evaluation.validate_variant_arrays(
        persistent, original.FRAMES, persistent_summary
    )
    catalog, catalog_path = base.staging_catalog(args.input_root)
    hashes[str(catalog_path)] = base.sha256(catalog_path)
    for bound in (evidence, persistent_evidence, v1_evidence, older_evidence):
        original.require_binding(catalog_path, hashes[str(catalog_path)], bound)
    initial, _seed, evaluation = base.registered_seed(int(original.FRAMES[0]))
    for key in catalog:
        if key.startswith(base.CALIBRATION_DIR) or key.startswith("work/c01_s16_predictions/"):
            path = base.verified_input(args.input_root, key, catalog)
            hashes[str(path)] = base.sha256(path)
            for bound in (evidence, persistent_evidence, v1_evidence):
                original.require_binding(path, hashes[str(path)], bound)
    gt_path = base.verified_input(
        args.input_root, evaluation["female_annotations_relative"], catalog
    )
    hashes[str(gt_path)] = base.sha256(gt_path)
    for bound in (evidence, persistent_evidence, v1_evidence):
        original.require_binding(gt_path, hashes[str(gt_path)], bound)
    if hashes[str(gt_path)] != evaluation["female_annotations_sha256"]:
        raise ValueError("female annotations differ from registered evaluation data")
    nodes = original.CachedNodes(args.input_root, base.EXPERIMENT, original.FRAMES)
    rig = original.load_rig(args.input_root / base.CALIBRATION_DIR)
    enrollment_audit, enrollment_hashes = verify_enrollment(
        args.run, summary, current, original.FRAMES, nodes=nodes
    )
    hashes.update(enrollment_hashes)
    with gt_path.open(encoding="utf-8-sig", newline="") as stream:
        anchors = original.match_mouth_anchors(
            csv.DictReader(stream),
            lambda view, frame: nodes.at(view, frame)[:, 0],
            int(original.FRAMES[0]),
            int(original.FRAMES[-1] + 1),
        )
    methods, positions = {}, {}
    for name, arrays, saved, previous, root in (
        ("current_appearance", current, summary, False, args.run),
        ("persistent_mouth", persistent, persistent_summary, False, args.persistent),
        ("original_three_body_v1", v1, v1_summary, False, args.original),
        ("previous_generative", older, older_summary, True, args.previous),
    ):
        source, review, observed, xyz, accepted = original.normalized_points(
            arrays, previous=previous
        )
        original.validate_sources(source, original.FRAMES, nodes)
        methods[name] = original.common_metrics(
            original.FRAMES, source, review, observed, xyz, accepted, anchors
        )
        methods[name].update(
            tracking_seconds=saved["tracking_seconds"],
            review_scope="combined geometry/identity review" if previous else "identity review",
            run_path=str(Path(root).resolve()),
            candidate_sha256=base.sha256(Path(root) / "candidate.npz"),
            algorithm_arm=saved.get("algorithm_arm", name),
        )
        positions[name] = {"frame_idx": original.FRAMES, "xyz": xyz}
    base.validate_result(current, original.FRAMES, nodes)
    base.validate_result(persistent, original.FRAMES, nodes)
    base.validate_result(v1, original.FRAMES, nodes)
    for value in base.BASELINE_LAMBDAS:
        relative = f"{base.BASELINE_DIR}/C34-label__{base.EXPERIMENT}__lam{value}.npz"
        path = base.verified_input(args.input_root, relative, catalog)
        hashes[str(path)] = base.sha256(path)
        with np.load(path, allow_pickle=False) as arrays:
            source_rows = arrays["source"][original.FRAMES]
            accepted_mask = arrays["slot_track"][original.FRAMES] >= 0
            xyz = np.where(accepted_mask[..., None], arrays["xyz"][original.FRAMES], np.nan)
        _pose, observed = base.skeleton_from_sources(rig, nodes, original.FRAMES, source_rows)
        observed &= accepted_mask[..., None]
        source = np.repeat(source_rows[..., None], 3, axis=-1)
        original.validate_sources(source, original.FRAMES, nodes)
        review = np.zeros(accepted_mask.shape, bool)
        name = f"C34_lambda_{value}"
        methods[name] = original.common_metrics(
            original.FRAMES, source, review, observed, xyz, xyz, anchors
        )
        methods[name].update(
            review_scope="C34 slot_track availability; no equivalent review flag",
            tracking_seconds=None,
            input_path=str(path.resolve()),
            candidate_sha256=hashes[str(path)],
        )
        positions[name] = {"frame_idx": original.FRAMES, "xyz": xyz}
    del positions["current_appearance"]
    diagnostics = original.diagnose(current, initial_pose_xyz=initial, baseline_positions=positions)
    events = original.extract_review_events(
        current, review_gap_frames=summary["tracker_config"]["review_gap_frames"]
    )
    report = dict(
        status="technical_comparison_only",
        frames=[76620, 86580],
        fps=166,
        methods=methods,
        current_algorithm_arm=summary["algorithm_arm"],
        current_run_path=str(args.run.resolve()),
        current_candidate_sha256=hashes[str((args.run / "candidate.npz").resolve())],
        appearance_original_input_parity=parity,
        persistent_original_input_parity=persistent_parity,
        appearance_persistent_recovery_and_persistence_config_equal=True,
        current_array_count=len(current),
        persistent_age_arrays_validated=True,
        enrollment_audit=enrollment_audit,
        copied_absolute_path_verifier_origin_sha256=VERIFIER_ORIGIN_SHA256,
        new_method_review_counts=events["counts"],
        new_method_staging_seconds=summary.get("precomputed_sequence_staging_seconds"),
        independent_all_three_truth_loaded=False,
        inference_performed=False,
        old_method_input_provenance_scope=(
            "historical sealed record; original remote videos not rehashed here"
        ),
    )
    frozen = {**hashes, **sources}
    if any(base.sha256(path) != digest for path, digest in frozen.items()):
        raise ValueError("evaluation input changed during comparison")
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance(
        "three_body_appearance_minute_comparison", args.output, name="evaluation"
    ) as record:
        record.params_from(args)
        record.param("algorithm_arm", summary["algorithm_arm"])
        for path, digest in frozen.items():
            record.input(path, path, sha256=digest)
        for path in code:
            target = args.output / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            record.output(str(target.relative_to(args.output)), target, sha256=base.sha256(target))
        for name, data in (
            ("comparison", report),
            ("diagnostics", diagnostics),
            ("review_events", events),
            ("enrollment_audit", enrollment_audit),
        ):
            path = args.output / f"{name}.json"
            path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
            record.output(name, path, sha256=base.sha256(path))
        path = args.output / "RESULT.md"
        text = original.markdown_report(report)
        text += (
            f"\ncurrent_appearance: `{summary['algorithm_arm']}`\n\nRun: `{args.run.resolve()}`\n\n"
            f"Candidate SHA256: `{report['current_candidate_sha256']}`\n"
            "\n最初166時点の保存済みonline状態と登録特徴の集計を検査しました。"
            "最終backtraceの過去姿勢とは同一を要求していません。"
            "原画像からの特徴再計算と暫定slotのIDの正しさは、この評価器では検証していません。\n"
        )
        path.write_text(text, encoding="utf-8")
        record.output("result", path, sha256=base.sha256(path))
        if any(base.sha256(path) != digest for path, digest in frozen.items()):
            raise ValueError("comparison input or code changed while writing outputs")
    print(
        json.dumps(
            dict(
                output=str(args.output),
                algorithm_arm=summary["algorithm_arm"],
                identity_verdict="unverified",
                events=events["counts"],
            )
        )
    )
    return report


if __name__ == "__main__":
    main()

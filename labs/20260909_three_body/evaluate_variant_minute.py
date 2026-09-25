"""Compare a sealed mouth-recovery minute with v1, older tracking and seven C34 arms.

Metrics and review-event definitions are reused unchanged from evaluate_minute.
The new verifier binds the variant runner's absolute-path input records to its
actual sequence manifest. It never fabricates the original runner's role names.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import evaluate_minute as original
import numpy as np
import run_experiment as base
import run_node_recovery as current_frame_runner
import run_persistent_node_recovery as persistent_runner

from medaka3d.provenance import provenance

REPO = base.REPO
LAB = base.LAB
VARIANTS = {
    current_frame_runner.ALGORITHM_ARM: current_frame_runner,
    persistent_runner.ALGORITHM_ARM: persistent_runner,
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


def verify_variant_evidence(root, summary, *, expected_frames=None):
    """Verify complete variant evidence; optionally exercise a sealed short test window.

    The public comparison CLI still requires the registered full minute through the
    unchanged original.load_completed gate. This helper accepts an explicit native
    frame array so the real completed 200-frame runner can test its evidence schema.
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
    for name in ("tracker_config", "node_recovery_config", "seed"):
        if not exact_json(params.get(name), summary.get(name)):
            raise ValueError(f"variant {name} is not bound to the sealed runtime parameters")
    if arm == persistent_runner.ALGORITHM_ARM and not exact_json(
        params.get("persistence_config"), summary.get("persistence_config")
    ):
        raise ValueError("persistence configuration is not bound")
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


def validate_variant_arrays(arrays, frames, summary):
    arm = summary.get("algorithm_arm")
    if arm not in VARIANTS:
        raise ValueError("unknown variant algorithm")
    expected = set(persistent_runner.ORIGINAL_ARRAYS)
    if arm == persistent_runner.ALGORITHM_ARM:
        expected |= {"node_missing", "direct_pectoral_age"}
    if set(arrays) != expected:
        raise ValueError("variant must preserve its exact declared 19 or 21 arrays")
    validated = VARIANTS[arm].validate_arm_arrays(arrays, frames)
    if not np.array_equal(validated["frame_idx"], frames):
        raise ValueError("variant does not contain every native frame")
    original.normalized_points(validated)
    return validated


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
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
        LAB / "diagnostics.py",
        LAB / "review_events.py",
        LAB / "run_experiment.py",
        LAB / "observations.py",
        LAB / "prepare_sequence.py",
        LAB / "run_node_recovery.py",
        LAB / "run_persistent_node_recovery.py",
        REPO / "src/medaka3d/calib.py",
        REPO / "src/medaka3d/types.py",
        REPO / "src/medaka3d/provenance.py",
        REPO / "src/medaka3d/eval/generative.py",
        REPO / "src/medaka3d/eval/reacquisition.py",
    ]
    sources = {str(p): base.sha256(p) for p in code}
    # Full-minute gate is deliberately unchanged; no prefix-evaluation option exists.
    current, summary, hashes = original.load_completed(args.run)
    v1, v1_summary, v1_hashes = original.load_completed(args.original)
    older, older_summary, older_hashes = original.load_completed(args.previous, previous=True)
    evidence = verify_variant_evidence(args.run, summary, expected_frames=original.FRAMES)
    v1_evidence = original.verify_run_evidence(args.original, v1_summary)
    older_evidence = original.verify_run_evidence(args.previous, older_summary, previous=True)
    parity = same_v1_inputs(args.run, summary, args.original, v1_summary, evidence, v1_evidence)
    hashes.update({**v1_hashes, **older_hashes, **evidence, **v1_evidence, **older_evidence})
    current = validate_variant_arrays(current, original.FRAMES, summary)
    catalog, catalog_path = base.staging_catalog(args.input_root)
    hashes[str(catalog_path)] = base.sha256(catalog_path)
    for bound in (evidence, v1_evidence, older_evidence):
        original.require_binding(catalog_path, hashes[str(catalog_path)], bound)
    initial, _seed, evaluation = base.registered_seed(int(original.FRAMES[0]))
    for key in catalog:
        if key.startswith(base.CALIBRATION_DIR) or key.startswith("work/c01_s16_predictions/"):
            path = base.verified_input(args.input_root, key, catalog)
            hashes[str(path)] = base.sha256(path)
            for bound in (evidence, v1_evidence):
                original.require_binding(path, hashes[str(path)], bound)
    gt_path = base.verified_input(
        args.input_root, evaluation["female_annotations_relative"], catalog
    )
    hashes[str(gt_path)] = base.sha256(gt_path)
    for bound in (evidence, v1_evidence):
        original.require_binding(gt_path, hashes[str(gt_path)], bound)
    if hashes[str(gt_path)] != evaluation["female_annotations_sha256"]:
        raise ValueError("female annotations differ from registered evaluation data")
    nodes = original.CachedNodes(args.input_root, base.EXPERIMENT, original.FRAMES)
    rig = original.load_rig(args.input_root / base.CALIBRATION_DIR)
    with gt_path.open(encoding="utf-8-sig", newline="") as stream:
        anchors = original.match_mouth_anchors(
            csv.DictReader(stream),
            lambda view, frame: nodes.at(view, frame)[:, 0],
            int(original.FRAMES[0]),
            int(original.FRAMES[-1] + 1),
        )
    methods, positions = {}, {}
    for name, arrays, saved, previous, root in (
        ("current_variant", current, summary, False, args.run),
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
    del positions["current_variant"]
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
        variant_original_input_parity=parity,
        current_array_count=len(current),
        persistent_age_arrays_validated=summary["algorithm_arm"] == persistent_runner.ALGORITHM_ARM,
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
        "three_body_variant_minute_comparison", args.output, name="evaluation"
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
        ):
            path = args.output / f"{name}.json"
            path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
            record.output(name, path, sha256=base.sha256(path))
        path = args.output / "RESULT.md"
        text = original.markdown_report(report)
        text += (
            f"\ncurrent_variant: `{summary['algorithm_arm']}`\n\nRun: `{args.run.resolve()}`\n\n"
            f"Candidate SHA256: `{report['current_candidate_sha256']}`\n"
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

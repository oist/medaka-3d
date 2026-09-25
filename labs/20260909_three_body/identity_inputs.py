"""Verified inputs for identity inference on frozen persistent geometry.

Producer summaries and final appearance trajectories are never deserialized.
Their bytes remain sealed evidence only. Enrollment uses the recorded causal
166-frame states and a successful raw-pixel replay, not the final beam prefix.
Cross-window transfer is narrowly supported for detectron2 union masks: exact
raw pixels, packed mask rows, SLEAP, calibration, seed and common code/config
are required. Registered background images may differ because this mode never
uses them. This is prefix equivalence, not equality of all inference inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import audit_appearance_enrollment as enrollment_audit
import cv2
import numpy as np
import run_appearance_tracking as appearance_runner
import run_experiment as base
import run_persistent_node_recovery as persistent_runner
from appearance_diagnostic import EROSION_RADIUS, FEATURES, MIN_CORE_PIXELS
from chunk_masks import SCHEMA_DTYPES, validate_packed_masks
from observations import CachedNodes
from prepare_sequence import array_sha256

REPO = base.REPO
PREFIX = 166
AUDIT_SOURCE_SHA256 = "0debc71438f6adcec62ad927fe522e29cce63d0b9b39ad1938dc33c4bc20a972"
GEOMETRY_ARRAYS = persistent_runner.ORIGINAL_ARRAYS | {"node_missing", "direct_pectoral_age"}
COMMON_CONFIGS = ("tracker_config", "flow_config", "node_recovery_config", "persistence_config")


@dataclass(frozen=True)
class IdentityInputs:
    geometry: dict[str, np.ndarray]
    enrollment: dict[str, np.ndarray]
    enrollment_metadata: dict
    frame_idx: np.ndarray
    initial_pose_xyz: np.ndarray
    geometry_root: Path
    enrollment_root: Path
    audit_root: Path
    geometry_sequence_root: Path
    enrollment_sequence_root: Path
    input_root: Path
    params: dict
    verified_hashes: dict[str, str]
    prefix_audit: dict


@dataclass(frozen=True)
class _Run:
    root: Path
    params: dict
    sequence: Path
    input_root: Path
    manifest: dict
    frames: np.ndarray
    initial: np.ndarray
    inference: dict[str, str]
    code: dict[str, str]
    hashes: dict[str, str]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _equal(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        b, sort_keys=True, allow_nan=False
    )


def _path(value):
    path = Path(value)
    path = (path if path.is_absolute() else REPO / path).resolve(strict=True)
    if not path.is_relative_to(REPO.resolve()):
        raise ValueError("identity inputs must be preserved workspace paths")
    return path


def _digest(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("invalid lowercase SHA256")
    return value


def _bind(path, digest, hashes):
    path, digest = _path(path), _digest(digest)
    if base.sha256(path) != digest or hashes.get(str(path), digest) != digest:
        raise ValueError(f"input/output SHA256 mismatch: {path}")
    hashes[str(path)] = digest
    return path


def _output(record, key, path, hashes):
    item = record.get("outputs", {}).get(key)
    if item is None or _path(item["path"]) != _path(path):
        raise ValueError(f"unbound output path: {key}")
    return _bind(path, item["sha256"], hashes)


def _npz(path):
    with np.load(path, allow_pickle=False) as saved:
        if len(saved.files) != len(set(saved.files)):
            raise ValueError("duplicate NPZ array names")
        result = {key: saved[key] for key in saved.files}
    if any(value.dtype.hasobject for value in result.values()):
        raise ValueError("object arrays are forbidden")
    return result


def _verify_run(root, *, appearance):
    root = _path(root)
    path = root / "experiment.provenance.json"
    hashes = {str(path): base.sha256(path)}
    record = _json(path)
    runner = appearance_runner if appearance else persistent_runner
    stage = (
        "three_body_causal_appearance_algorithm_arm"
        if appearance
        else "three_body_persistent_node_recovery_algorithm_arm"
    )
    params = record.get("params", {})
    if record.get("status") != "ok" or record.get("stage") != stage:
        raise ValueError("producer must have the completed registered provenance stage")
    if params.get("algorithm_arm") != runner.ALGORITHM_ARM or _path(params["output"]) != root:
        raise ValueError("producer arm or output path mismatch")
    for key in (
        "inputs_unchanged_during_run",
        "code_unchanged_during_run",
        "old_results_and_gt_opened_only_after_tracking",
    ):
        if params.get(key) is not True:
            raise ValueError(f"producer lacks successful inference evidence: {key}")
    # Establish allowed inference paths before opening any entry. A mislabeled
    # post-inference asset must not become readable merely by dropping its prefix.
    sequence, input_root = _path(params["sequence"]), _path(params["input_root"])
    manifest_path = sequence / "manifest.json"
    manifest_item = record["inputs"].get(str(manifest_path))
    if manifest_item is None or _path(manifest_item["path"]) != manifest_path:
        raise ValueError("sequence manifest must be an actual inference input")
    _bind(manifest_path, manifest_item["sha256"], hashes)
    manifest = _json(manifest_path)
    allowed_inputs = {str(manifest_path), str(input_root / "staging_manifest.json")}
    for filename in (
        "calibration_manifest.json",
        *(f"cam{i}_bundle_refractive.json" for i in range(3)),
    ):
        allowed_inputs.add(str(input_root / base.CALIBRATION_DIR / filename))
    for category in ("raw", "masks", "backgrounds"):
        for entry in manifest[category].values():
            allowed_inputs.add(str(base.resolve_input(sequence, entry["path"])))
    for view in base.VIEWS:
        allowed_inputs.add(
            str(input_root / "work/c01_s16_predictions/bottomup" / f"{base.EXPERIMENT}__{view}.npz")
        )
    inference = {}
    for key, item in record["inputs"].items():
        # Skip before path resolution/hash/open; no GT/reference bytes enter this loader.
        if key.startswith("post_inference:"):
            continue
        path = _path(item["path"])
        if key == "seed_registration":
            if path != _path(base.REGISTRATION):
                raise ValueError("unexpected seed registration path")
        elif str(path) not in allowed_inputs:
            raise ValueError("unexpected inference asset; refusing to open it")
        _bind(path, item["sha256"], hashes)
        if key != "seed_registration":
            if not Path(key).is_absolute() or _path(key) != path or str(path) in inference:
                raise ValueError("inference keys must be unique actual absolute paths")
            inference[str(path)] = item["sha256"]
    for key, filename in (
        ("seed_registration", "seed_registration.json"),
        ("candidate", "candidate.npz"),
        ("summary", "summary.json"),
    ):
        _output(record, key, root / filename, hashes)
    if appearance:
        _output(record, "appearance_enrollment", root / "appearance_enrollment.npz", hashes)
        _output(
            record, "appearance_enrollment_metadata", root / "appearance_enrollment.json", hashes
        )
    sequence = _path(params["sequence"])
    manifest_path = sequence / "manifest.json"
    manifest = _json(manifest_path)
    frames = base.sequence_frames(manifest)
    allowed = ((333519, 333719), (76620, 77286))
    if not appearance:
        allowed += ((76620, 86580),)
    if (int(frames[0]), int(frames[-1] + 1)) not in allowed:
        raise ValueError("only registered complete 200/666 geometry/enrollment or minute geometry")
    if manifest.get("image_shape_hw") != [1200, 1200]:
        raise ValueError("registered image shape differs")
    required = {str(manifest_path)}
    for category in ("raw", "masks", "backgrounds"):
        if set(manifest[category]) != set(base.VIEWS):
            raise ValueError("exactly three registered views are required")
        for view in base.VIEWS:
            item = manifest[category][view]
            asset = base.resolve_input(sequence, item["path"])
            if inference.get(str(asset)) != _digest(item["sha256"]):
                raise ValueError("manifest asset is not bound as an inference input")
            required.add(str(asset))
            if category == "raw":
                pixels = item.get("frame_bgr_sha256", [])
                if len(pixels) != len(frames):
                    raise ValueError("every native raw frame must have a pixel SHA")
                for digest in pixels:
                    _digest(digest)
    input_root = _path(params["input_root"])
    copies = [
        (manifest_path, root / "sequence_manifest.json"),
        (input_root / "staging_manifest.json", root / "staging_manifest.json"),
    ]
    for filename in (
        "calibration_manifest.json",
        *(f"cam{i}_bundle_refractive.json" for i in range(3)),
    ):
        copies.append(
            (input_root / base.CALIBRATION_DIR / filename, root / "calibration_snapshot" / filename)
        )
    for source, copied in copies:
        source = source.resolve(strict=True)
        if str(source) not in inference:
            raise ValueError("manifest/catalog/calibration is missing from inference inputs")
        required.add(str(source))
        _bind(copied, inference[str(source)], hashes)
    for view in base.VIEWS:
        required.add(
            str(
                (
                    input_root
                    / "work/c01_s16_predictions/bottomup"
                    / f"{base.EXPERIMENT}__{view}.npz"
                ).resolve(strict=True)
            )
        )
    if set(inference) != required:
        raise ValueError("unexpected or missing inference inputs")
    initial, seed, _unused_evaluation_metadata = base.registered_seed(int(frames[0]))
    if not _equal(params.get("seed"), seed) or not _equal(
        _json(root / "seed_registration.json"), seed
    ):
        raise ValueError("registered first seed mismatch")
    if hashes.get(str(_path(seed["registration_path"]))) != seed["registration_sha256"]:
        raise ValueError("seed registration is not a sealed input")
    initial_path = root / "initial_pose_xyz.npy"
    enrollment_audit.exact_array(np.load(initial_path, allow_pickle=False), initial, "initial pose")
    hashes[str(initial_path)] = base.sha256(initial_path)
    code = {}
    for relative, digest in params["code_hashes"].items():
        relative = base.safe_relative(relative).as_posix()
        if relative in code:
            raise ValueError("duplicate normalized source snapshot")
        code[relative] = _digest(digest)
    if set(code) != {p.relative_to(REPO).as_posix() for p in runner.code_paths()}:
        raise ValueError("producer code snapshot set is incomplete or unexpected")
    for relative, digest in code.items():
        _bind(base.resolve_input(root / "code_snapshot", relative), digest, hashes)
    return _Run(
        root, params, sequence, input_root, manifest, frames, initial, inference, code, hashes
    )


def validate_enrollment(snapshot, metadata, frames, nodes):
    """Validate causal prefix arithmetic without loading any final appearance arrays."""
    config = {"training_frames": PREFIX, "minimum_template_samples": 5}
    enrollment_audit.validate_snapshot(snapshot, frames, PREFIX)
    for name, expected in (
        ("config", config),
        ("views", list(base.VIEWS)),
        ("feature_names", list(FEATURES)),
        ("erosion_radius_px", EROSION_RADIUS),
        ("minimum_core_pixels", MIN_CORE_PIXELS),
        ("frozen", True),
        ("training_frames_seen", PREFIX),
        ("first_frame_idx", int(frames[0])),
        ("last_training_frame_idx", int(frames[PREFIX - 1])),
        ("external_templates_or_saved_poses_used", False),
        ("reviewed_slots_registered", False),
        ("retroactive_beam_relabeling", False),
    ):
        if not _equal(metadata.get(name), expected):
            raise ValueError(f"invalid causal enrollment metadata: {name}")
    appearance_runner.validate_enrollment_audit({**metadata, **snapshot}, frames, PREFIX)
    features = snapshot["features"]
    finite = np.isfinite(features)
    usable = finite.all(axis=-1)
    if (
        np.isinf(features).any()
        or np.any(finite.any(axis=-1) != usable)
        or np.any((features[finite] < 0) | (features[finite] > 1))
    ):
        raise ValueError("features must be whole [0,1] descriptors or whole NaN")
    if not np.array_equal(usable, snapshot["reasons"] == "usable"):
        raise ValueError("feature availability and usable reasons disagree")
    reasons = (
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
    )
    if not np.isin(snapshot["reasons"], reasons).all():
        raise ValueError("unknown enrollment descriptor reason")
    rows, core = snapshot["instance_rows"], snapshot["core_pixels"]
    if (
        np.any(rows < -1)
        or np.any(core < 0)
        or np.any(usable & ((rows < 0) | (core < MIN_CORE_PIXELS)))
    ):
        raise ValueError("usable descriptor lacks a valid row/core")
    observed, sources = snapshot["online_observed"], snapshot["online_sources"]
    if not np.array_equal(observed, (sources >= 0).sum(axis=2) >= 2):
        raise ValueError("online observed nodes must match direct source counts")
    allowed = (
        observed[:, :, :2].all(-1)[:, None]
        & (sources[..., :2] >= 0).all(-1).transpose(0, 2, 1)
        & ~snapshot["online_identity_review"][:, None]
    )
    if np.any(usable & ~allowed):
        raise ValueError("enrollment uses reviewed or unsupported online nodes")
    for t, frame in enumerate(frames[:PREFIX]):
        for v, view in enumerate(base.VIEWS):
            assigned = rows[t, v][rows[t, v] >= 0]
            if len(np.unique(assigned)) != len(assigned):
                raise ValueError("one instance assigned to multiple enrollment slots")
            held = sources[t, :, v]
            if np.any(held >= len(nodes.at(view, int(frame)))):
                raise ValueError("online SLEAP source row out of range")
            for node in range(3):
                taken = held[:, node][held[:, node] >= 0]
                if len(np.unique(taken)) != len(taken):
                    raise ValueError("online SLEAP sources are not exclusive")
    aggregate = enrollment_audit.aggregate_descriptors(features, 5)
    for key, value in aggregate.items():
        enrollment_audit.exact_array(snapshot[key], value, key)
    for name in ("counts", "enabled_views"):
        if not _equal(metadata.get(name), aggregate[name].tolist()):
            raise ValueError("enrollment aggregate metadata mismatch")


def packed_prefix(data, frames, count=PREFIX):
    """Validate outer schema/offsets, then decode-check only the requested native prefix."""
    if set(data) != set(SCHEMA_DTYPES) or any(data[k].dtype != d for k, d in SCHEMA_DTYPES.items()):
        raise ValueError("packed schema/dtype mismatch")
    if not np.array_equal(data["frame_idx"], frames) or not 1 <= count <= len(frames):
        raise ValueError("packed native frames mismatch or prefix incomplete")
    boxes = data["mask_boxes"]
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError("packed crop shape mismatch")
    instances = len(boxes)
    for key, size, terminal in (
        ("frame_offsets", len(frames) + 1, instances),
        ("mask_byte_offsets", instances + 1, len(data["mask_bits"])),
    ):
        offsets = data[key]
        if (
            offsets.shape != (size,)
            or offsets[0] != 0
            or offsets[-1] != terminal
            or np.any(np.diff(offsets) < 0)
            or np.any(offsets < 0)
            or np.any(offsets > terminal)
        ):
            raise ValueError("packed offset origin/terminal/continuity mismatch")
    for key in ("scores", "classes", "instance_id", "source_track_id"):
        if data[key].shape != (instances,):
            raise ValueError("packed instance metadata length mismatch")
    if data["pred_boxes"].shape != (instances, 4) or data["mask_bits"].ndim != 1:
        raise ValueError("packed payload shape mismatch")
    if any(data[k].shape != (len(frames),) for k in ("view_available", "frame_mask_sha256")):
        raise ValueError("packed frame metadata length mismatch")
    end = int(data["frame_offsets"][count])
    byte_end = int(data["mask_byte_offsets"][end])
    prefix = {}
    for key, value in data.items():
        if key == "image_shape_hw":
            prefix[key] = value.copy()
        elif key == "frame_offsets":
            prefix[key] = value[: count + 1].copy()
        elif key == "mask_byte_offsets":
            prefix[key] = value[: end + 1].copy()
        elif key == "mask_bits":
            prefix[key] = value[:byte_end].copy()
        elif key in ("frame_idx", "view_available", "frame_mask_sha256"):
            prefix[key] = value[:count].copy()
        else:
            prefix[key] = value[:end].copy()
    validate_packed_masks(prefix)
    return prefix


def _prefix_masks(run):
    result = {}
    for view in base.VIEWS:
        path = base.resolve_input(run.sequence, run.manifest["masks"][view]["path"])
        result[view] = packed_prefix(_npz(path), run.frames)
        if not np.array_equal(result[view]["image_shape_hw"], run.manifest["image_shape_hw"]):
            raise ValueError("actual mask image shape differs from manifest")
    return result


def _verify_audit(root, run, enrollment, masks, nodes):
    root = _path(root)
    path = root / "audit.provenance.json"
    hashes = {str(path): base.sha256(path)}
    record = _json(path)
    if (
        record.get("status") != "ok"
        or record.get("stage") != "three_body_raw_appearance_enrollment_audit"
    ):
        raise ValueError("a completed raw enrollment audit is required")
    params = record["params"]
    if _path(params["run"]) != run.root or _path(params["output"]) != root:
        raise ValueError("raw audit run/output binding differs")
    for key, value in (
        ("device", "cpu"),
        ("ground_truth_and_completed_baseline_arrays_opened", False),
        ("reexecuted_beam_selection_or_tracking", False),
        ("inputs_and_sources_unchanged", True),
    ):
        if not _equal(params.get(key), value):
            raise ValueError(f"invalid raw audit scope: {key}")
    expected_inputs = dict(run.hashes)
    code = {p.relative_to(REPO).as_posix(): p for p in enrollment_audit.audit_code_paths()}
    output_code = {
        base.safe_relative(k[5:]).as_posix(): v
        for k, v in record["outputs"].items()
        if k.startswith("code:")
    }
    if set(output_code) != set(code):
        raise ValueError("raw audit source snapshot set differs")
    for relative, source in code.items():
        item = output_code[relative]
        saved = root / "code_snapshot" / relative
        if _path(item["path"]) != _path(saved):
            raise ValueError("raw audit source snapshot path mismatch")
        _bind(saved, item["sha256"], hashes)
        if relative in enrollment_audit.REPLAY_SOURCES and run.code.get(relative) != item["sha256"]:
            raise ValueError("raw audit descriptor/support source differs from enrollment run")
        if (
            relative.endswith("/audit_appearance_enrollment.py")
            and item["sha256"] != AUDIT_SOURCE_SHA256
        ):
            raise ValueError("raw audit implementation is not the reviewed frozen version")
        # Source paths are bound to the audit as well as the immutable snapshots.
        expected_inputs[str(source)] = item["sha256"]
    actual_inputs = {}
    for key, item in record["inputs"].items():
        source = _path(item["path"])
        if not Path(key).is_absolute() or _path(key) != source or str(source) in actual_inputs:
            raise ValueError("raw audit input role/path mismatch")
        # Do not follow an unexpected entry into labels or reference trajectories.
        if expected_inputs.get(str(source)) != item.get("sha256"):
            raise ValueError("raw audit input is unexpected or not bound to producer evidence")
        _bind(source, item["sha256"], hashes)
        actual_inputs[str(source)] = item["sha256"]
    if actual_inputs != expected_inputs:
        raise ValueError("raw audit is missing required producer/code evidence")
    _output(record, "summary", root / "summary.json", hashes)
    _output(record, "recomputed_enrollment", root / "recomputed_enrollment.npz", hashes)
    summary = _json(root / "summary.json")  # Audit-only schema: no tracking evaluation metrics.
    expected = dict(
        status="complete_exact_cpu_replay",
        algorithm_arm=appearance_runner.ALGORITHM_ARM,
        enrollment_sha256=run.hashes[str(run.root / "appearance_enrollment.npz")],
        sequence_manifest_sha256=run.inference[str(run.sequence / "manifest.json")],
        frame_range=[int(run.frames[0]), int(run.frames[PREFIX - 1] + 1)],
        native_frames=PREFIX,
        native_fps=166,
        decoded_images=3 * PREFIX,
        all_recomputed_arrays_bitwise_equal=True,
        raw_feature_and_mask_recomputation_performed=True,
        online_native_support_verified=True,
        online_states_compared_to_final_traceback=False,
        identity_labels_verified=False,
        counts=enrollment["counts"].tolist(),
        enabled_views=enrollment["enabled_views"].tolist(),
        views=list(base.VIEWS),
        feature_names=list(FEATURES),
        erosion_radius_px=EROSION_RADIUS,
        minimum_core_pixels=MIN_CORE_PIXELS,
    )
    if _path(summary["run"]) != run.root or any(
        not _equal(summary.get(k), v) for k, v in expected.items()
    ):
        raise ValueError("raw audit summary does not certify this enrollment prefix")
    replay = _npz(root / "recomputed_enrollment.npz")
    if set(replay) != set(enrollment_audit.PER_FRAME) | set(enrollment_audit.FINAL):
        raise ValueError("raw replay must contain exactly eight declared arrays")
    comparisons = {
        key: enrollment_audit.exact_array(value, enrollment[key], key)
        for key, value in replay.items()
    }
    if not _equal(comparisons, summary.get("array_comparisons")):
        raise ValueError("raw audit array comparisons disagree with actual bytes")
    records = summary.get("frames", [])
    if len(records) != PREFIX:
        raise ValueError("raw audit must retain all 166 native frame records")
    for t, item in enumerate(records):
        if item["frame_idx"] != int(run.frames[t]) or set(item["views"]) != set(base.VIEWS):
            raise ValueError("raw audit native frame/view order mismatch")
        for v, view in enumerate(base.VIEWS):
            data = masks[view]
            lo, hi = map(int, data["frame_offsets"][t : t + 2])
            fields = item["views"][view]
            expected_view = dict(
                bgr_sha256=run.manifest["raw"][view]["frame_bgr_sha256"][t],
                sleap_points_sha256=array_sha256(nodes.at(view, int(run.frames[t]))),
                mask_available=bool(data["view_available"][t]),
                mask_count=hi - lo if data["view_available"][t] else None,
                decoded_masks_sha256=(
                    bytes(data["frame_mask_sha256"][t]).decode("ascii")
                    if data["view_available"][t]
                    else None
                ),
            )
            if any(not _equal(fields.get(k), value) for k, value in expected_view.items()):
                raise ValueError("raw audit pixels/SLEAP/mask prefix binding mismatch")
            if np.any(enrollment["instance_rows"][t, v] >= hi - lo):
                raise ValueError("enrollment instance row outside actual prefix frame")
    return hashes


def compare_runs(geometry, appearance):
    """Check the narrow allowed transfer before expensive raw-prefix decoding."""
    for name in (*COMMON_CONFIGS, "seed"):
        if name not in geometry.params or name not in appearance.params:
            raise ValueError(f"missing common inference configuration: {name}")
        if not _equal(geometry.params.get(name), appearance.params.get(name)):
            raise ValueError(f"geometry/enrollment differ in {name}")
    if geometry.input_root != appearance.input_root:
        raise ValueError("geometry/enrollment calibration and SLEAP roots differ")
    enrollment_audit.exact_array(geometry.initial, appearance.initial, "shared seed")
    shared = set(geometry.code) & set(appearance.code)
    if any(geometry.code[name] != appearance.code[name] for name in shared):
        raise ValueError("geometry/enrollment shared inference code differs")
    same_window = np.array_equal(geometry.frames, appearance.frames)
    if same_window:
        if geometry.inference != appearance.inference:
            raise ValueError("same-window inference paths and SHA must be exactly equal")
    elif (
        int(geometry.frames[0]),
        int(geometry.frames[-1] + 1),
        int(appearance.frames[0]),
        int(appearance.frames[-1] + 1),
    ) != (76620, 86580, 76620, 77286):
        raise ValueError("unsupported cross-window enrollment transfer")
    elif geometry.params["tracker_config"].get("mode") != "detectron2":
        raise ValueError("cross-window backgrounds may differ only in detectron2 mode")
    if not np.array_equal(geometry.frames[:PREFIX], appearance.frames[:PREFIX]):
        raise ValueError("causal enrollment native prefix differs")
    background_equal = {}
    for view in base.VIEWS:
        a, b = geometry.manifest, appearance.manifest
        if (
            a["raw"][view]["frame_bgr_sha256"][:PREFIX]
            != b["raw"][view]["frame_bgr_sha256"][:PREFIX]
        ):
            raise ValueError("recorded raw prefix pixel hashes differ")
        background_equal[view] = (
            a["backgrounds"][view]["sha256"] == b["backgrounds"][view]["sha256"]
        )
    if not same_window:
        # Both complete inference sets were independently verified; common non-image assets
        # must also agree, not merely the root directory strings.
        fixed = {
            p: h
            for p, h in geometry.inference.items()
            if Path(p).is_relative_to(geometry.input_root)
        }
        if fixed != {
            p: h
            for p, h in appearance.inference.items()
            if Path(p).is_relative_to(appearance.input_root)
        }:
            raise ValueError("cross-window SLEAP/calibration/catalog evidence differs")
    return dict(
        same_window=same_window,
        same_all_inference_paths_and_sha=same_window,
        background_sha_equal=all(background_equal.values()),
        background_by_view=background_equal,
        background_policy=(
            "Registered staging backgrounds may differ across windows; detectron2 union masks, "
            "optical flow and enrollment do not use these images."
        ),
        shared_code_files=len(shared),
        native_prefix_frames=PREFIX,
    )


def verify_raw_prefix(path, expected, total_frames, image_shape=(1200, 1200)):
    """Decode exactly the common prefix; never substitute whole-video hash for pixel equality."""
    capture = cv2.VideoCapture(str(path))
    try:
        if (
            not capture.isOpened()
            or capture.get(cv2.CAP_PROP_FRAME_COUNT) != total_frames
            or not np.isclose(capture.get(cv2.CAP_PROP_FPS), 166, rtol=0, atol=1e-6)
        ):
            raise ValueError("raw geometry clip count/fps mismatch")
        for index, digest in enumerate(expected):
            ok, image = capture.read()
            if (
                not ok
                or image.shape != (*image_shape, 3)
                or image.dtype != np.uint8
                or array_sha256(image) != digest
            ):
                raise ValueError(f"actual geometry raw prefix pixel mismatch at offset {index}")
    finally:
        capture.release()


def load_identity_inputs(geometry_run, enrollment_run, enrollment_audit_root):
    """Load sealed geometry and audited causal templates, with no automatic relabeling.

    Returned review fields retain their original meaning. No identity truth, final
    appearance pose, producer evaluation summary or post-inference asset is read.
    Runtime frame lists come from the sealed sequence; only the first registered
    pose initializes identity. Callers must seal this loader and all their own
    inference code and keep the returned input hashes bound until their output.
    """
    geometry = _verify_run(geometry_run, appearance=False)
    appearance = _verify_run(enrollment_run, appearance=True)
    prefix_audit = compare_runs(geometry, appearance)
    if not _equal(
        appearance.params.get("enrollment_config"),
        dict(training_frames=PREFIX, minimum_template_samples=5),
    ):
        raise ValueError("only the fixed causal 166/5 enrollment is supported")
    arrays = _npz(geometry.root / "candidate.npz")
    if set(arrays) != GEOMETRY_ARRAYS:
        raise ValueError("persistent geometry must retain exactly its 21 arrays")
    persistent_runner.validate_arm_arrays(arrays, geometry.frames)
    nodes = CachedNodes(geometry.input_root, base.EXPERIMENT, geometry.frames)
    base.validate_result(arrays, geometry.frames, nodes)
    enrollment = _npz(appearance.root / "appearance_enrollment.npz")
    metadata = _json(appearance.root / "appearance_enrollment.json")
    if (
        metadata.get("output_sha256")
        != appearance.hashes[str(appearance.root / "appearance_enrollment.npz")]
    ):
        raise ValueError("enrollment metadata is not bound to its actual NPZ")
    validate_enrollment(enrollment, metadata, appearance.frames, nodes)
    masks = _prefix_masks(appearance)
    audit_hashes = _verify_audit(enrollment_audit_root, appearance, enrollment, masks, nodes)
    geometry_masks = masks if prefix_audit["same_window"] else _prefix_masks(geometry)
    mask_array_hashes = {}
    for view in base.VIEWS:
        mask_array_hashes[view] = {
            key: enrollment_audit.exact_array(geometry_masks[view][key], value, f"{view}/{key}")[
                "sha256"
            ]
            for key, value in masks[view].items()
        }
    decoded = 0
    for view in base.VIEWS:
        current, enrolled = geometry.manifest["raw"][view], appearance.manifest["raw"][view]
        if current["sha256"] != enrolled["sha256"]:
            verify_raw_prefix(
                base.resolve_input(geometry.sequence, current["path"]),
                current["frame_bgr_sha256"][:PREFIX],
                len(geometry.frames),
            )
            decoded += PREFIX
    hashes = {}
    for source in (geometry.hashes, appearance.hashes, audit_hashes):
        for path, digest in source.items():
            if hashes.get(path, digest) != digest:
                raise ValueError("conflicting evidence hashes")
            hashes[path] = digest
    for path, digest in hashes.items():
        if base.sha256(path) != digest:
            raise RuntimeError("identity inference input changed while loading")
    prefix_audit.update(
        actual_packed_prefix_arrays_equal=True,
        packed_prefix_array_sha256=mask_array_hashes,
        prefix_decoded_masks_verified=True,
        full_geometry_mask_suffix_decoded=False,
        geometry_prefix_images_decoded_now=decoded,
        unchanged_clip_prefix_supported_by_raw_enrollment_audit=True,
        producer_summaries_deserialized=False,
        final_appearance_candidate_deserialized=False,
        post_inference_assets_opened=False,
        online_prefix_compared_to_final_traceback=False,
        identity_labels_verified=False,
        automatic_relabeling=False,
        limitation=(
            "Templates describe provisional causal slots; successful replay does not establish "
            "biological identity or color discriminability."
        ),
    )
    return IdentityInputs(
        arrays,
        enrollment,
        metadata,
        geometry.frames,
        geometry.initial,
        geometry.root,
        appearance.root,
        _path(enrollment_audit_root),
        geometry.sequence,
        appearance.sequence,
        geometry.input_root,
        {name: geometry.params[name] for name in COMMON_CONFIGS},
        hashes,
        prefix_audit,
    )

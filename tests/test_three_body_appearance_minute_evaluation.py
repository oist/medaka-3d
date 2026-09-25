"""Appearance evaluation binds online enrollment without rewriting final traceback."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load_evaluator():
    names = (
        "prepare_sequence",
        "observations",
        "run_experiment",
        "diagnostics",
        "review_events",
        "evaluate_minute",
        "run_node_recovery",
        "run_persistent_node_recovery",
        "evaluate_variant_minute",
        "run_appearance_tracking",
        "appearance_diagnostic",
        "evaluate_appearance_minute",
    )
    saved, old_path = {name: sys.modules.get(name) for name in names}, sys.path.copy()
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(
                f"_appearance_minute_test_{name}", LAB / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sys.modules[name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


evaluation = load_evaluator()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def appearance_arrays(frames, training=3):
    count = len(frames)
    pose = np.arange(count * 27, dtype=np.float64).reshape(count, 3, 3, 3) / 100
    source = np.broadcast_to(np.arange(3)[None, :, None, None], (count, 3, 3, 3)).copy()
    result = {name: np.zeros(count) for name in evaluation.appearance_runner.ORIGINAL_ARRAYS}
    enabled = np.zeros((count, 3), bool)
    enabled[training:, 0] = True
    appearance = enabled.any(axis=1).astype(np.float64) * 0.75
    result.update(
        frame_idx=frames.copy(),
        pose_xyz=pose,
        observed=np.ones((count, 3, 3), bool),
        sources=source,
        identity_review=np.zeros((count, 3), bool),
        xyz=pose[:, :, 1].copy(),
        accepted_xyz=pose[:, :, 1].copy(),
        hypotheses=np.full(count, 6, np.int64),
        node_missing=np.zeros((count, 3, 3), np.int64),
        direct_pectoral_age=np.broadcast_to(np.arange(1, count + 1)[:, None], (count, 3)).copy(),
        geometric_cost=np.ones(count),
        appearance_cost=appearance,
        costs=np.ones(count) + appearance,
        appearance_enabled=enabled,
        appearance_pixels=enabled.astype(np.int64) * 5,
    )
    return result


def enrollment_records(frames, result, training=3):
    features = np.arange(training * 54, dtype=np.float64).reshape(training, 3, 3, 6) / 200
    # One side camera is unavailable, which must not be replaced by zero descriptors.
    features[:, 2] = np.nan
    reasons = np.full((training, 3, 3), "usable", dtype="U64")
    reasons[:, 2] = "view_unavailable"
    rows = np.broadcast_to(np.arange(3), (training, 3, 3)).copy()
    rows[:, 2] = -1
    core = np.full((training, 3, 3), 15, np.int64)
    core[:, 2] = 0
    counts = np.isfinite(features).all(-1).sum(0)
    templates = np.full((3, 3, 6), np.nan)
    templates[:2] = np.median(features[:, :2], axis=0)
    snapshot = dict(
        frame_idx=frames[:training].copy(),
        features=features,
        reasons=reasons,
        instance_rows=rows,
        core_pixels=core,
        online_pose_xyz=result["pose_xyz"][:training].copy(),
        online_sources=result["sources"][:training].copy(),
        online_observed=result["observed"][:training].copy(),
        online_identity_review=result["identity_review"][:training].copy(),
        counts=counts,
        templates=templates,
        enabled_views=np.array([True, True, False]),
    )
    config = dict(training_frames=training, minimum_template_samples=2)
    metadata = dict(
        config=config,
        views=list(evaluation.base.VIEWS),
        feature_names=list(evaluation.FEATURES),
        erosion_radius_px=2,
        minimum_core_pixels=12,
        frozen=True,
        training_frames_seen=training,
        first_frame_idx=int(frames[0]),
        last_training_frame_idx=int(frames[training - 1]),
        external_templates_or_saved_poses_used=False,
        reviewed_slots_registered=False,
        retroactive_beam_relabeling=False,
        scope="Current selected model slots in the initial native prefix; not identity GT",
        counts=counts.tolist(),
        enabled_views=[True, True, False],
        sampled_pixels=result["appearance_pixels"].sum(0).tolist(),
        scored_frames=int(result["appearance_enabled"].any(1).sum()),
    )
    summary = dict(
        algorithm_arm=evaluation.appearance_runner.ALGORITHM_ARM,
        enrollment_config=config,
        appearance_config=dict(weight=1.0, maximum_samples=512, active_views=["top_cam"]),
        pixel_config=dict(sigma=0.1, background_cost=2.0, outlier_fraction=0.1),
    )
    return snapshot, metadata, summary


@pytest.fixture
def enrollment():
    frames = np.arange(76620, 76625, dtype=np.int64)
    result = appearance_arrays(frames)
    snapshot, metadata, summary = enrollment_records(frames, result)
    return frames, result, snapshot, metadata, summary


def test_recorded_online_prefix_and_medians_are_audited_independently(enrollment):
    frames, result, snapshot, metadata, summary = enrollment
    assert len(evaluation.validate_appearance_arrays(result, frames, summary)) == 25
    nodes = SimpleNamespace(at=lambda _view, _frame: np.zeros((3, 3, 2)))
    audit = evaluation.validate_enrollment_records(
        snapshot, metadata, frames, result, summary, nodes=nodes
    )
    assert audit["online_source_row_bounds_checked"]
    assert audit["counts"] == [[3, 3, 3], [3, 3, 3], [0, 0, 0]]
    assert audit["online_states_compared_to_final_traceback"] is False
    assert audit["raw_feature_and_mask_recomputation_performed"] is False
    assert audit["identity_labels_verified"] is False


def test_future_traceback_can_differ_without_relabeling_registration(enrollment):
    frames, result, snapshot, metadata, summary = enrollment
    before = {name: value.copy() for name, value in snapshot.items()}
    # Future evidence may choose a different temporal parent; online registration stays fixed.
    result["pose_xyz"][:3] += 11
    result["sources"][:3] = result["sources"][:3, [1, 0, 2]]
    result["identity_review"][:3, 0] = True
    result["xyz"] = result["pose_xyz"][:, :, 1].copy()
    result["accepted_xyz"] = np.where(result["identity_review"][..., None], np.nan, result["xyz"])
    evaluation.validate_appearance_arrays(result, frames, summary)
    evaluation.validate_enrollment_records(snapshot, metadata, frames, result, summary)
    for name in before:
        np.testing.assert_array_equal(snapshot[name], before[name])


@pytest.mark.parametrize(
    "bad",
    [
        "prefix_gap",
        "missing_online",
        "online_dtype",
        "online_nonfinite",
        "reviewed_feature",
        "single_view_support",
        "missing_registering_view",
        "source_duplicate",
        "source_upper_bound",
        "source_negative",
        "partial_descriptor",
        "descriptor_infinite",
        "descriptor_range",
        "wrong_reason",
        "pending_reason",
        "no_instance",
        "instance_duplicate",
        "tiny_core",
        "wrong_counts",
        "wrong_metadata_counts",
        "one_template_changed",
        "disabled_template_filled",
        "partial_templates",
        "enabled_incomplete_view",
        "scored_unenrolled_view",
        "boolean_config",
        "wrong_views",
        "future_frame_metadata",
        "not_frozen",
        "retroactive_relabel",
        "pixel_total",
        "scored_total",
        "boolean_total",
        "unexpected_npz_array",
    ],
)
def test_invalid_or_relabelled_enrollment_is_rejected(enrollment, bad):
    frames, result, snap, meta, summary = enrollment
    nodes = SimpleNamespace(at=lambda _view, _frame: np.zeros((3, 3, 2)))
    if bad == "prefix_gap":
        snap["frame_idx"][1] += 1
    elif bad == "missing_online":
        del snap["online_sources"]
    elif bad == "online_dtype":
        snap["online_sources"] = snap["online_sources"].astype(np.int32)
    elif bad == "online_nonfinite":
        snap["online_pose_xyz"][0, 0, 0, 0] = np.nan
    elif bad == "reviewed_feature":
        snap["online_identity_review"][0, 0] = True
    elif bad == "single_view_support":
        snap["online_sources"][0, 0, 1:, 0] = -1
    elif bad == "missing_registering_view":
        snap["online_sources"][0, 0, 0, 0] = -1
    elif bad == "source_duplicate":
        snap["online_sources"][0, 1, 0, 0] = 0
    elif bad == "source_upper_bound":
        snap["online_sources"][0, 0, 0, 0] = 3
    elif bad == "source_negative":
        snap["online_sources"][0, 0, 0, 0] = -2
    elif bad in ("partial_descriptor", "descriptor_infinite", "descriptor_range"):
        snap["features"][0, 0, 0, 0] = {
            "partial_descriptor": np.nan,
            "descriptor_infinite": np.inf,
            "descriptor_range": 1.1,
        }[bad]
    elif bad in ("wrong_reason", "pending_reason"):
        snap["reasons"][0, 0, 0] = "view_unavailable" if bad == "wrong_reason" else "pending"
    elif bad == "no_instance":
        snap["instance_rows"][0, 0, 0] = -1
    elif bad == "instance_duplicate":
        snap["instance_rows"][0, 0, 1] = 0
    elif bad == "tiny_core":
        snap["core_pixels"][0, 0, 0] = 11
    elif bad == "wrong_counts":
        snap["counts"][0, 0] -= 1
    elif bad == "wrong_metadata_counts":
        meta["counts"][0][0] -= 1
    elif bad == "one_template_changed":
        snap["templates"][0, 0, 0] += 0.1
    elif bad == "disabled_template_filled":
        snap["templates"][2] = 0
    elif bad == "partial_templates":
        snap["templates"][0, 1, 0] = np.nan
    elif bad == "enabled_incomplete_view":
        snap["enabled_views"][2] = meta["enabled_views"][2] = True
    elif bad == "scored_unenrolled_view":
        result["appearance_enabled"][4, 2] = True
        result["appearance_pixels"][4, 2] = 1
        meta["sampled_pixels"][2] = 1
    elif bad == "boolean_config":
        summary["enrollment_config"]["minimum_template_samples"] = True
    elif bad == "wrong_views":
        meta["views"].reverse()
    elif bad == "future_frame_metadata":
        meta["last_training_frame_idx"] += 1
    elif bad == "not_frozen":
        meta["frozen"] = False
    elif bad == "retroactive_relabel":
        meta["retroactive_beam_relabeling"] = True
    elif bad == "pixel_total":
        meta["sampled_pixels"][0] += 1
    elif bad == "scored_total":
        meta["scored_frames"] += 1
    elif bad == "boolean_total":
        meta["sampled_pixels"][1] = False
    else:
        snap["substituted_final_pose"] = result["pose_xyz"]
    with pytest.raises((ValueError, AssertionError)):
        evaluation.validate_enrollment_records(snap, meta, frames, result, summary, nodes=nodes)


@pytest.mark.parametrize(
    "bad",
    [
        "missing_array",
        "unexpected_array",
        "cost_sum",
        "negative_loss",
        "prefix_loss",
        "prefix_enabled",
        "disabled_cost",
        "pixels_disagree",
        "wrong_view",
        "age",
        "subsampled",
    ],
)
def test_appearance_output_does_not_invent_use_or_mutate_saved_support(enrollment, bad):
    frames, result, _snap, _meta, summary = enrollment
    if bad == "missing_array":
        del result["geometric_cost"]
    elif bad == "unexpected_array":
        result["undocumented"] = np.zeros(len(frames))
    elif bad == "cost_sum":
        result["costs"][-1] += 0.5
    elif bad == "negative_loss":
        result["appearance_cost"][-1] = -0.1
        result["costs"][-1] = 0.9
    elif bad in ("prefix_loss", "disabled_cost"):
        result["appearance_cost"][0] = 0.1
        result["costs"][0] += 0.1
    elif bad == "prefix_enabled":
        result["appearance_enabled"][0, 0] = True
        result["appearance_pixels"][0, 0] = 1
    elif bad == "pixels_disagree":
        result["appearance_pixels"][-1, 0] = 0
    elif bad == "wrong_view":
        result["appearance_enabled"][-1, 1] = True
        result["appearance_pixels"][-1, 1] = 1
    elif bad == "age":
        result["direct_pectoral_age"][-1, 1] = 1
    else:
        result = {name: value[::2] for name, value in result.items()}
    with pytest.raises((ValueError, AssertionError)):
        evaluation.validate_appearance_arrays(result, frames, summary)


@pytest.fixture
def evidence(tmp_path, monkeypatch, enrollment):
    repo = tmp_path / "repo"
    run, sequence, input_root = repo / "run", repo / "sequence", repo / "input"
    for path in (run, sequence, input_root):
        path.mkdir(parents=True)
    monkeypatch.setattr(evaluation, "REPO", repo)
    monkeypatch.setattr(evaluation.base, "REPO", repo)
    monkeypatch.setattr(evaluation.base, "CALIBRATION_DIR", "calibration")
    monkeypatch.setattr(
        evaluation.appearance_runner, "code_paths", lambda: [repo / "labs/tracker.py"]
    )
    frames, arrays, snap, metadata, summary = enrollment
    initial = np.zeros((3, 3, 3))
    registration = repo / "registration.json"
    write_json(registration, {"registered": True})
    seed = dict(
        frame=int(frames[0]),
        pose_xyz=initial.tolist(),
        registration_path=str(registration),
        registration_sha256=evaluation.base.sha256(registration),
    )
    monkeypatch.setattr(evaluation.base, "registered_seed", lambda _frame: (initial, seed, {}))
    inputs = []
    manifest = dict(
        experiment=evaluation.base.EXPERIMENT,
        frames=frames.tolist(),
        fps=166,
        image_shape_hw=[1200, 1200],
    )
    for category in ("raw", "masks", "backgrounds"):
        manifest[category] = {}
        for view in evaluation.base.VIEWS:
            path = sequence / category / f"{view}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{category}:{view}".encode())
            entry = dict(
                path=path.relative_to(sequence).as_posix(), sha256=evaluation.base.sha256(path)
            )
            if category == "raw":
                entry["frame_bgr_sha256"] = ["a" * 64] * len(frames)
            manifest[category][view] = entry
            inputs.append(path)
    manifest_path = sequence / "manifest.json"
    write_json(manifest_path, manifest)
    (run / "sequence_manifest.json").write_bytes(manifest_path.read_bytes())
    inputs.append(manifest_path)
    catalog = input_root / "staging_manifest.json"
    write_json(catalog, {"files": []})
    (run / "staging_manifest.json").write_bytes(catalog.read_bytes())
    inputs.append(catalog)
    for name in (
        "calibration_manifest.json",
        *(f"cam{i}_bundle_refractive.json" for i in range(3)),
    ):
        path = input_root / "calibration" / name
        write_json(path, {"name": name})
        target = run / "calibration_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        inputs.append(path)
    for view in evaluation.base.VIEWS:
        path = (
            input_root
            / f"work/c01_s16_predictions/bottomup/{evaluation.base.EXPERIMENT}__{view}.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(view.encode())
        inputs.append(path)
    source = run / "code_snapshot/labs/tracker.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"sealed inference implementation")
    code = {"labs/tracker.py": evaluation.base.sha256(source)}
    hashes = {str(path): evaluation.base.sha256(path) for path in inputs}
    summary.update(
        technical_completion=True,
        inputs_unchanged_during_run=True,
        code_unchanged_during_run=True,
        input_sha256_before=hashes,
        input_sha256_after=hashes.copy(),
        code_sha256_before=code,
        code_sha256_after=code.copy(),
        frames=[int(frames[0]), int(frames[-1] + 1)],
        source_sequence_frames=[int(frames[0]), int(frames[-1] + 1)],
        requested_frames=len(frames),
        returned_frames=len(frames),
        native_fps=166,
        temporal_subsampling=False,
        development_prefix_smoke=False,
        initialization_count=1,
        track_invocation_count=1,
        reset_count=0,
        tracker_config={"value": 1},
        node_recovery_config={"enabled": True},
        persistence_config={"minimum_missing_frames": 4},
        seed=seed,
    )
    np.save(run / "initial_pose_xyz.npy", initial)
    np.savez(run / "candidate.npz", **arrays)
    np.savez(run / "appearance_enrollment.npz", **snap)
    metadata["output_sha256"] = evaluation.base.sha256(run / "appearance_enrollment.npz")
    summary["appearance_enrollment"] = metadata
    write_json(run / "appearance_enrollment.json", metadata)
    write_json(run / "seed_registration.json", seed)
    record = dict(
        status="ok",
        inputs={str(p): dict(path=str(p), sha256=hashes[str(p)]) for p in inputs},
        params=dict(
            sequence=str(sequence),
            input_root=str(input_root),
            algorithm_arm=summary["algorithm_arm"],
            code_hashes=code.copy(),
            inputs_unchanged_during_run=True,
            code_unchanged_during_run=True,
        ),
        outputs={},
    )
    for name in (
        "seed",
        "tracker_config",
        "node_recovery_config",
        "persistence_config",
        "appearance_config",
        "enrollment_config",
        "pixel_config",
    ):
        record["params"][name] = copy.deepcopy(summary[name])
    record["inputs"]["seed_registration"] = dict(
        path=str(registration), sha256=seed["registration_sha256"]
    )

    def seal_again():
        write_json(run / "summary.json", summary)
        for key, name in (
            ("summary", "summary.json"),
            ("candidate", "candidate.npz"),
            ("seed_registration", "seed_registration.json"),
            ("appearance_enrollment", "appearance_enrollment.npz"),
            ("appearance_enrollment_metadata", "appearance_enrollment.json"),
        ):
            path = run / name
            record["outputs"][key] = dict(path=str(path), sha256=evaluation.base.sha256(path))
        write_json(run / "experiment.provenance.json", record)

    seal_again()
    return dict(
        repo=repo,
        run=run,
        sequence=sequence,
        input_root=input_root,
        frames=frames,
        arrays=arrays,
        summary=summary,
        record=record,
        snapshot=snap,
        metadata=metadata,
        source=source,
        manifest=manifest,
        seal_again=seal_again,
    )


def test_complete_appearance_binding_includes_enrollment_outputs(evidence):
    e = evidence
    hashes = evaluation.verify_appearance_evidence(
        e["run"], e["summary"], expected_frames=e["frames"]
    )
    audit, enrollment_hashes = evaluation.verify_enrollment(
        e["run"], e["summary"], e["arrays"], e["frames"]
    )
    assert hashes[str(e["source"])] == evaluation.base.sha256(e["source"])
    assert set(enrollment_hashes) == {
        str(e["run"] / f"appearance_enrollment.{suffix}") for suffix in ("npz", "json")
    }
    assert audit["training_frames"] == 3


@pytest.mark.parametrize(
    "bad",
    [
        "npz_tamper",
        "metadata_tamper",
        "wrong_output_path",
        "missing_output",
        "metadata_npz_sha",
        "summary_metadata",
    ],
)
def test_enrollment_seal_cannot_be_rebound(evidence, bad):
    e = evidence
    path = e["run"] / "appearance_enrollment.npz"
    if bad == "npz_tamper":
        path.write_bytes(path.read_bytes() + b"changed")
    elif bad == "metadata_tamper":
        write_json(e["run"] / "appearance_enrollment.json", {"other": True})
    elif bad == "wrong_output_path":
        copy_path = e["repo"] / "copied.npz"
        copy_path.write_bytes(path.read_bytes())
        e["record"]["outputs"]["appearance_enrollment"]["path"] = str(copy_path)
        write_json(e["run"] / "experiment.provenance.json", e["record"])
    elif bad == "missing_output":
        del e["record"]["outputs"]["appearance_enrollment"]
        write_json(e["run"] / "experiment.provenance.json", e["record"])
    else:
        e["metadata"]["output_sha256"] = "0" * 64
        if bad == "metadata_npz_sha":
            write_json(e["run"] / "appearance_enrollment.json", e["metadata"])
        e["seal_again"]()
    with pytest.raises(ValueError):
        evaluation.verify_enrollment(e["run"], e["summary"], e["arrays"], e["frames"])


@pytest.mark.parametrize(
    "bad",
    [
        "inference_removed",
        "post_inference_alias",
        "input_tamper",
        "sequence_rebound",
        "unbound_config",
        "snapshot_tamper",
        "unbound_snapshot",
        "initial_pose",
        "output_path",
        "pending",
    ],
)
def test_appearance_evidence_preserves_absolute_manifest_seed_and_code_binding(evidence, bad):
    e = evidence
    raw = e["sequence"] / e["manifest"]["raw"][evaluation.base.VIEWS[0]]["path"]
    if bad in ("inference_removed", "post_inference_alias"):
        item = e["record"]["inputs"].pop(str(raw))
        if bad == "post_inference_alias":
            e["record"]["inputs"]["post_inference:" + str(raw)] = item
    elif bad == "input_tamper":
        raw.write_bytes(b"different observation")
    elif bad == "sequence_rebound":
        other = e["repo"] / "other"
        other.mkdir()
        (other / "manifest.json").write_bytes((e["sequence"] / "manifest.json").read_bytes())
        e["record"]["params"]["sequence"] = str(other)
    elif bad == "unbound_config":
        e["summary"]["pixel_config"]["sigma"] = 0.2
    elif bad in ("snapshot_tamper", "unbound_snapshot"):
        e["source"].write_bytes(b"other source")
        if bad == "unbound_snapshot":
            for key in ("code_sha256_before", "code_sha256_after"):
                e["summary"][key]["labs/tracker.py"] = evaluation.base.sha256(e["source"])
    elif bad == "initial_pose":
        np.save(e["run"] / "initial_pose_xyz.npy", np.ones((3, 3, 3)))
    elif bad == "pending":
        e["record"]["status"] = "running"
    e["seal_again"]()
    if bad == "output_path":
        other = e["repo"] / "candidate.npz"
        other.write_bytes((e["run"] / "candidate.npz").read_bytes())
        e["record"]["outputs"]["candidate"]["path"] = str(other)
        write_json(e["run"] / "experiment.provenance.json", e["record"])
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluation.verify_appearance_evidence(e["run"], e["summary"], expected_frames=e["frames"])


def test_image_audit_can_omit_post_inference_files_but_cannot_omit_inference(evidence):
    e = evidence
    absent = e["repo"] / "unopened_gt.npz"
    e["record"]["inputs"]["post_inference:gt"] = dict(path=str(absent), sha256="f" * 64)
    e["seal_again"]()
    hashes = evaluation.verify_appearance_evidence(
        e["run"], e["summary"], include_post_inference=False
    )
    assert str(absent) not in hashes
    with pytest.raises(FileNotFoundError):
        evaluation.verify_appearance_evidence(e["run"], e["summary"])
    raw = e["sequence"] / e["manifest"]["raw"][evaluation.base.VIEWS[0]]["path"]
    e["record"]["inputs"]["post_inference:raw"] = e["record"]["inputs"].pop(str(raw))
    e["seal_again"]()
    with pytest.raises(ValueError, match="exactly bound"):
        evaluation.verify_appearance_evidence(e["run"], e["summary"], include_post_inference=False)


@pytest.mark.parametrize("key", [None, "appearance_config", "enrollment_config", "pixel_config"])
def test_registered_minute_configuration_is_not_implicitly_retuned(enrollment, key):
    summary = enrollment[-1]
    summary["enrollment_config"] = dict(training_frames=166, minimum_template_samples=5)
    if key is None:
        evaluation.validate_registered_arm(summary)
    else:
        field = next(iter(summary[key]))
        summary[key][field] = 9
        with pytest.raises(ValueError, match="registered"):
            evaluation.validate_registered_arm(summary)


def test_short_development_evidence_cannot_pass_full_minute_gate(evidence):
    with pytest.raises(ValueError, match="complete single-initialization registered minute"):
        evaluation.original.load_completed(evidence["run"])

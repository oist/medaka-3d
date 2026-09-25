"""Absolute-path provenance must preserve the original minute's evidence contract."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

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
    )
    previous, old_path = {name: sys.modules.get(name) for name in names}, sys.path.copy()
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(
                f"_variant_minute_test_{name}", LAB / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sys.modules[name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        for name, saved in previous.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


evaluation = load_evaluator()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    run, sequence, input_root = repo / "run", repo / "sequence", repo / "input"
    for path in (run, sequence, input_root):
        path.mkdir(parents=True)
    monkeypatch.setattr(evaluation, "REPO", repo)
    monkeypatch.setattr(evaluation.base, "REPO", repo)
    monkeypatch.setattr(evaluation.base, "CALIBRATION_DIR", "calibration")
    monkeypatch.setattr(
        evaluation.persistent_runner, "code_paths", lambda: [repo / "labs/tracker.py"]
    )
    frames = np.arange(76620, 76623, dtype=np.int64)
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
    infer_paths = []
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
            path.write_bytes(f"{category} {view}".encode())
            entry = dict(
                path=path.relative_to(sequence).as_posix(), sha256=evaluation.base.sha256(path)
            )
            if category == "raw":
                entry["frame_bgr_sha256"] = ["a" * 64] * len(frames)
            manifest[category][view] = entry
            infer_paths.append(path)
    manifest_path = sequence / "manifest.json"
    write_json(manifest_path, manifest)
    (run / "sequence_manifest.json").write_bytes(manifest_path.read_bytes())
    infer_paths.append(manifest_path)
    catalog = input_root / "staging_manifest.json"
    write_json(catalog, {"files": []})
    (run / "staging_manifest.json").write_bytes(catalog.read_bytes())
    infer_paths.append(catalog)
    for name in (
        "calibration_manifest.json",
        *(f"cam{i}_bundle_refractive.json" for i in range(3)),
    ):
        path = input_root / "calibration" / name
        write_json(path, {"name": name})
        copy_path = run / "calibration_snapshot" / name
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        copy_path.write_bytes(path.read_bytes())
        infer_paths.append(path)
    for view in evaluation.base.VIEWS:
        path = (
            input_root
            / f"work/c01_s16_predictions/bottomup/{evaluation.base.EXPERIMENT}__{view}.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(view.encode())
        infer_paths.append(path)
    snapshot = run / "code_snapshot/labs/tracker.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"original frozen code")
    code = {"labs/tracker.py": evaluation.base.sha256(snapshot)}
    hashes = {str(path): evaluation.base.sha256(path) for path in infer_paths}
    summary = dict(
        algorithm_arm=evaluation.persistent_runner.ALGORITHM_ARM,
        technical_completion=True,
        inputs_unchanged_during_run=True,
        code_unchanged_during_run=True,
        input_sha256_before=hashes,
        input_sha256_after=hashes.copy(),
        code_sha256_before=code,
        code_sha256_after=code.copy(),
        frames=[76620, 76623],
        source_sequence_frames=[76620, 76623],
        requested_frames=3,
        returned_frames=3,
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
    write_json(run / "seed_registration.json", seed)
    np.save(run / "initial_pose_xyz.npy", initial)
    np.savez(run / "candidate.npz", frame_idx=frames)
    record = dict(
        status="ok",
        inputs={
            str(path): dict(path=str(path), sha256=digest)
            for path, digest in ((Path(k), v) for k, v in hashes.items())
        },
        params=dict(
            sequence=str(sequence),
            input_root=str(input_root),
            algorithm_arm=summary["algorithm_arm"],
            seed=copy.deepcopy(seed),
            tracker_config=copy.deepcopy(summary["tracker_config"]),
            node_recovery_config=copy.deepcopy(summary["node_recovery_config"]),
            persistence_config=copy.deepcopy(summary["persistence_config"]),
            code_hashes=code.copy(),
            inputs_unchanged_during_run=True,
            code_unchanged_during_run=True,
        ),
        outputs={},
    )
    record["inputs"]["seed_registration"] = dict(
        path=str(registration), sha256=seed["registration_sha256"]
    )

    def seal_again():
        write_json(run / "summary.json", summary)
        for key, name in (
            ("summary", "summary.json"),
            ("candidate", "candidate.npz"),
            ("seed_registration", "seed_registration.json"),
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
        summary=summary,
        record=record,
        snapshot=snapshot,
        manifest=manifest,
        seal_again=seal_again,
    )


def test_actual_path_binding_does_not_need_fabricated_role_keys(evidence):
    result = evaluation.verify_variant_evidence(
        evidence["run"], evidence["summary"], expected_frames=evidence["frames"]
    )
    assert not any(key.startswith("raw_") for key in evidence["record"]["inputs"])
    asset = evidence["sequence"] / evidence["manifest"]["raw"][evaluation.base.VIEWS[0]]["path"]
    assert result[str(asset)] == evaluation.base.sha256(asset)
    assert str(evidence["snapshot"]) in result


@pytest.mark.parametrize(
    "bad",
    [
        "raw_tamper",
        "manifest_copy",
        "wrong_sequence_same_bytes",
        "manifest_view_swap",
        "inference_removed",
        "post_inference_only",
        "fake_role_key",
        "before_after",
        "snapshot_tamper",
        "unbound_snapshot_hash",
        "missing_snapshot_set",
        "path_escape",
        "seed_pose",
        "config_type",
        "unbound_arm",
        "raw_pixel_count",
        "wrong_frames",
        "calibration_copy",
        "staging_copy",
        "pending",
    ],
)
def test_incomplete_or_rebound_variant_is_rejected(evidence, bad):
    e = evidence
    summary, record = e["summary"], e["record"]
    raw = e["sequence"] / e["manifest"]["raw"][evaluation.base.VIEWS[0]]["path"]
    if bad == "raw_tamper":
        raw.write_bytes(b"different raw")
    elif bad == "manifest_copy":
        write_json(e["run"] / "sequence_manifest.json", {"unrelated": True})
    elif bad == "wrong_sequence_same_bytes":
        other = e["repo"] / "other_sequence"
        other.mkdir()
        (other / "manifest.json").write_bytes((e["sequence"] / "manifest.json").read_bytes())
        record["params"]["sequence"] = str(other)
    elif bad in ("manifest_view_swap", "raw_pixel_count"):
        manifest = e["manifest"]
        if bad == "manifest_view_swap":
            manifest["raw"][evaluation.base.VIEWS[0]]["path"] = manifest["raw"][
                evaluation.base.VIEWS[1]
            ]["path"]
        else:
            manifest["raw"][evaluation.base.VIEWS[0]]["frame_bgr_sha256"].pop()
        write_json(e["sequence"] / "manifest.json", manifest)
        (e["run"] / "sequence_manifest.json").write_bytes(
            (e["sequence"] / "manifest.json").read_bytes()
        )
        name = str(e["sequence"] / "manifest.json")
        digest = evaluation.base.sha256(name)
        record["inputs"][name]["sha256"] = digest
        for key in ("input_sha256_before", "input_sha256_after"):
            summary[key][name] = digest
    elif bad in ("inference_removed", "post_inference_only", "fake_role_key"):
        item = record["inputs"].pop(str(raw))
        if bad == "post_inference_only":
            record["inputs"]["post_inference:" + str(raw)] = item
        elif bad == "fake_role_key":
            record["inputs"]["raw_top_cam"] = item
        else:
            for key in ("input_sha256_before", "input_sha256_after"):
                del summary[key][str(raw)]
    elif bad == "before_after":
        summary["input_sha256_after"][str(raw)] = "0" * 64
    elif bad in ("snapshot_tamper", "unbound_snapshot_hash"):
        e["snapshot"].write_bytes(b"replacement code")
        if bad == "unbound_snapshot_hash":
            digest = evaluation.base.sha256(e["snapshot"])
            for key in ("code_sha256_before", "code_sha256_after"):
                summary[key]["labs/tracker.py"] = digest
    elif bad == "missing_snapshot_set":
        summary["code_sha256_before"] = summary["code_sha256_after"] = {}
        record["params"]["code_hashes"] = {}
    elif bad == "path_escape":
        for key in ("code_sha256_before", "code_sha256_after"):
            summary[key] = {"../tracker.py": "0" * 64}
    elif bad == "seed_pose":
        np.save(e["run"] / "initial_pose_xyz.npy", np.ones((3, 3, 3)))
    elif bad == "config_type":
        summary["tracker_config"]["value"] = True
    elif bad == "unbound_arm":
        summary["algorithm_arm"] = evaluation.current_frame_runner.ALGORITHM_ARM
    elif bad == "wrong_frames":
        summary["frames"][1] += 1
    elif bad == "calibration_copy":
        (e["run"] / "calibration_snapshot/calibration_manifest.json").write_bytes(b"other")
    elif bad == "staging_copy":
        (e["run"] / "staging_manifest.json").write_bytes(b"other")
    else:
        record["status"] = "running"
    e["seal_again"]()
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluation.verify_variant_evidence(e["run"], summary, expected_frames=e["frames"])


def age_arrays():
    count = 3
    observed = np.ones((count, 3, 3), bool)
    observed[1:, 0, 0] = False
    pose = np.zeros((count, 3, 3, 3))
    ages = np.zeros_like(observed, dtype=np.int64)
    ages[1:, 0, 0] = [1, 2]
    arrays = {name: np.zeros(count) for name in evaluation.persistent_runner.ORIGINAL_ARRAYS}
    arrays.update(
        frame_idx=np.arange(count),
        pose_xyz=pose,
        observed=observed,
        sources=np.zeros((count, 3, 3, 3), np.int32),
        identity_review=np.zeros((count, 3), bool),
        xyz=pose[:, :, 1].copy(),
        accepted_xyz=pose[:, :, 1].copy(),
        hypotheses=np.ones(count, np.int64),
        node_missing=ages,
        direct_pectoral_age=np.broadcast_to(np.arange(1, 4)[:, None], (count, 3)).copy(),
    )
    return arrays


def test_persistent_ages_follow_saved_native_backtrace():
    arrays = age_arrays()
    summary = {"algorithm_arm": evaluation.persistent_runner.ALGORITHM_ARM}
    assert len(evaluation.validate_variant_arrays(arrays, np.arange(3), summary)) == 21
    arrays["node_missing"][2, 0, 0] = 1
    with pytest.raises(ValueError, match="backtrace"):
        evaluation.validate_variant_arrays(arrays, np.arange(3), summary)


@pytest.mark.parametrize("change", [None, "seed", "config", "input_path", "initial", "flow"])
def test_v1_comparison_preserves_seed_config_and_exact_inference_path(evidence, change):
    e = evidence
    v1_root = e["repo"] / "v1"
    v1_root.mkdir()
    (v1_root / "initial_pose_xyz.npy").write_bytes((e["run"] / "initial_pose_xyz.npy").read_bytes())
    old = copy.deepcopy(e["summary"])
    old_record = copy.deepcopy(e["record"])
    before = evaluation.normalized_hashes(e["summary"]["input_sha256_before"])
    a, b = before.copy(), before.copy()
    if change == "seed":
        old["seed"]["pose_xyz"][0][0][0] = 1
    elif change == "config":
        old["tracker_config"]["value"] = 2
    elif change == "input_path":
        original_path = next(iter(old["input_sha256_before"]))
        copied = e["repo"] / "identical_bytes_different_path.bin"
        copied.write_bytes(Path(original_path).read_bytes())
        for key in ("input_sha256_before", "input_sha256_after"):
            digest = old[key].pop(original_path)
            old[key][str(copied)] = digest
        b[str(copied)] = b.pop(original_path)
    elif change == "initial":
        np.save(v1_root / "initial_pose_xyz.npy", np.ones((3, 3, 3)))
    elif change == "flow":
        old_record["params"]["flow_config"] = {"different": True}
    write_json(v1_root / "experiment.provenance.json", old_record)
    if change is not None:
        with pytest.raises(ValueError):
            evaluation.same_v1_inputs(e["run"], e["summary"], v1_root, old, a, b)
    else:
        result = evaluation.same_v1_inputs(e["run"], e["summary"], v1_root, old, a, b)
        assert result["same_inference_paths_and_sha"] is True
        assert str(v1_root / "initial_pose_xyz.npy") in b


def test_original_minute_gate_rejects_real_complete_200_frame_window():
    root = Path(__file__).resolve().parents[1] / "work/three_body/persistent_mouth_transient_200_v1"
    if not root.exists():
        pytest.skip("local completed development run is not installed")
    with pytest.raises(ValueError, match="complete single-initialization registered minute"):
        evaluation.original.load_completed(root)


def test_completed_persistent_200_has_real_manifest_and_age_binding():
    root = Path(__file__).resolve().parents[1] / "work/three_body/persistent_mouth_transient_200_v1"
    if not root.exists():
        pytest.skip("local completed development run is not installed")
    summary = evaluation.load_json(root / "summary.json")
    frames = np.arange(333519, 333719)
    hashes = evaluation.verify_variant_evidence(root, summary, expected_frames=frames)
    with np.load(root / "candidate.npz", allow_pickle=False) as stored:
        arrays = {k: stored[k] for k in stored.files}
    assert len(evaluation.validate_variant_arrays(arrays, frames, summary)) == 21
    assert hashes[str(root / "sequence_manifest.json")] == evaluation.base.sha256(
        root / "sequence_manifest.json"
    )

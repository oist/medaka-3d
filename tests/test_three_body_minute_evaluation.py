"""A completed minute is evidence-bound and abstentions cannot disappear."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from medaka3d.eval.generative import MouthAnchor
from medaka3d.types import VIEWS

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load_evaluator():
    names = (
        "prepare_sequence", "observations", "run_experiment", "diagnostics",
        "review_events", "evaluate_minute",
    )
    previous = {name: sys.modules.get(name) for name in names}
    old_path = sys.path.copy()
    loaded = {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(
                f"_minute_evaluation_audit_{name}", LAB / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        sys.path[:] = old_path
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return loaded["evaluate_minute"]


evaluation = load_evaluator()


def candidate(count=14, *, previous=False):
    pose = np.arange(count * 27, dtype=float).reshape(count, 3, 3, 3)
    source = np.broadcast_to(np.arange(3)[None, :, None, None], (count, 3, 3, 3)).copy()
    review = np.zeros((count, 3), bool)
    arrays = {
        "frame_idx": np.arange(count, dtype=np.int64),
        "pose_xyz": pose,
        "observed": np.ones((count, 3, 3), bool),
        "node_source" if previous else "sources": source,
        "identity_review": review.copy(),
        "xyz": pose[:, :, 1].copy(),
        "accepted_xyz": pose[:, :, 1].copy(),
    }
    if previous:
        arrays["review"] = review.copy()
        arrays["geometry_review"] = review.copy()
    return arrays


def metrics(arrays, *, previous=False, anchors=None):
    normalized = evaluation.normalized_points(arrays, previous=previous)
    if anchors is None:
        anchors = [MouthAnchor(int(f), {VIEWS[1]: 1}, {}, {}) for f in arrays["frame_idx"]]
    return evaluation.common_metrics(arrays["frame_idx"], *normalized, anchors)


def test_combined_old_review_preserves_geometry_abstention():
    arrays = candidate(previous=True)
    arrays["geometry_review"][5, 1] = arrays["review"][5, 1] = True
    arrays["accepted_xyz"][5, 1] = np.nan
    result = metrics(arrays, previous=True)
    assert result["review_fish_frames"] == 1
    assert result["female_raw"]["interval_counts"] == {
        "preserved": 13, "swap": 0, "unscored": 0,
    }
    assert result["female_accepted"]["interval_counts"] == {
        "preserved": 11, "swap": 0, "unscored": 2,
    }
    arrays["accepted_xyz"][5, 1] = arrays["xyz"][5, 1]
    with pytest.raises(AssertionError):
        evaluation.normalized_points(arrays, previous=True)


def test_all_reviewed_keeps_all_thirteen_requested_intervals():
    arrays = candidate()
    arrays["identity_review"][:, 1] = True
    arrays["accepted_xyz"][:, 1] = np.nan
    result = metrics(arrays)
    assert result["female_raw"]["available_anchors"] == 14
    assert result["female_accepted"]["requested_anchors"] == 14
    assert result["female_accepted"]["requested_intervals"] == 13
    assert result["female_accepted"]["interval_counts"] == {
        "preserved": 0, "swap": 0, "unscored": 13,
    }


def test_missing_anchor_does_not_connect_across_it_and_arm_math_is_identical():
    new, old = candidate(), candidate(previous=True)
    anchors = [MouthAnchor(f, {} if f == 5 else {VIEWS[1]: 1}, {}, {}) for f in range(14)]
    a, b = metrics(new, anchors=anchors), metrics(old, previous=True, anchors=anchors)
    assert a == b
    assert a["female_raw"]["interval_counts"] == {
        "preserved": 11, "swap": 0, "unscored": 2,
    }
    assert len(a["female_raw"]["intervals"]) == 13


@pytest.mark.parametrize("bad", ["partial_nan", "inf", "one_view", "review_int", "source_float"])
def test_saved_support_and_coordinate_schema_fail_loudly(bad):
    arrays = candidate()
    if bad == "partial_nan":
        arrays["xyz"][0, 0, 0] = np.nan
    elif bad == "inf":
        arrays["accepted_xyz"][0, 0] = np.inf
    elif bad == "one_view":
        arrays["sources"][0, 0, :2, 0] = -1
    elif bad == "review_int":
        arrays["identity_review"] = arrays["identity_review"].astype(int)
    else:
        arrays["sources"] = arrays["sources"].astype(float)
    with pytest.raises(ValueError):
        evaluation.normalized_points(arrays)


class ThreeRows:
    def at(self, view, frame):
        return np.zeros((3, 3, 2))


def test_source_exclusivity_includes_an_unobserved_competitor():
    arrays = candidate()
    arrays["observed"][0, 2] = False
    arrays["sources"][0, 2, 1, 0] = arrays["sources"][0, 1, 1, 0]
    with pytest.raises(ValueError, match="multiple fish"):
        evaluation.validate_sources(arrays["sources"], arrays["frame_idx"], ThreeRows())
    arrays["sources"][0, 2, 1, 0] = 3
    with pytest.raises(ValueError, match="original detector rows"):
        evaluation.validate_sources(arrays["sources"], arrays["frame_idx"], ThreeRows())


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def sealed_run(root, *, frames=None, summary_updates=None, status="ok"):
    root.mkdir(parents=True)
    frames = evaluation.FRAMES if frames is None else frames
    np.savez(root / "candidate.npz", frame_idx=frames)
    summary = {
        "technical_completion": True, "experiment": evaluation.EXPERIMENT,
        "initialization_count": 1, "requested_frames": 9960, "returned_frames": 9960,
        "frames": [76620, 86580], "code_unchanged_during_run": True,
        "native_fps": 166, "temporal_subsampling": False,
        "development_prefix_smoke": False, "track_invocation_count": 1,
        "reset_count": 0, "inputs_unchanged_during_run": True,
    }
    summary.update(summary_updates or {})
    write_json(root / "summary.json", summary)
    record = {"status": status, "outputs": {
        key: {"sha256": evaluation.sha256(root / name)}
        for key, name in (("candidate", "candidate.npz"), ("summary", "summary.json"))
    }}
    write_json(root / "experiment.provenance.json", record)
    return summary, record


def test_requires_successful_sealed_minute_not_candidate_alone(tmp_path):
    root = tmp_path / "run"
    sealed_run(root)
    arrays, _summary, hashes = evaluation.load_completed(root)
    np.testing.assert_array_equal(arrays["frame_idx"], evaluation.FRAMES)
    assert len(hashes) == 3
    (root / "experiment.provenance.json").unlink()
    with pytest.raises(FileNotFoundError):
        evaluation.load_completed(root)


@pytest.mark.parametrize("case", ["pending", "prefix", "lying_prefix", "reset", "subsampled"])
def test_pending_prefix_reset_and_subsampled_minutes_are_rejected(tmp_path, case):
    updates = {}
    frames = evaluation.FRAMES
    status = "running" if case == "pending" else "ok"
    if case in ("prefix", "lying_prefix"):
        frames = frames[:20]
    if case == "prefix":
        updates = {"returned_frames": 20, "development_prefix_smoke": True}
    if case == "reset":
        updates["reset_count"] = 1
    if case == "subsampled":
        updates["temporal_subsampling"] = True
    root = tmp_path / "run"
    sealed_run(root, frames=frames, summary_updates=updates, status=status)
    with pytest.raises(ValueError):
        evaluation.load_completed(root)


@pytest.mark.parametrize("name", ["summary.json", "candidate.npz"])
def test_changed_output_cannot_enter_evaluation(tmp_path, name):
    root = tmp_path / "run"
    sealed_run(root)
    with (root / name).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        evaluation.load_completed(root)


def evidence_fixture(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    root = repo / "run"
    root.mkdir(parents=True)
    monkeypatch.setattr(evaluation, "REPO", repo)
    inputs = {}
    for category in ("raw", "masks", "backgrounds"):
        for view in VIEWS:
            name = f"{category}_{view}"
            path = repo / f"{name}.bin"
            path.write_bytes(name.encode())
            inputs[name] = {"path": str(path), "sha256": evaluation.sha256(path)}
    snapshot = root / "code_snapshot" / "labs" / "tracker.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"frozen source")
    code_hashes = {"labs/tracker.py": evaluation.sha256(snapshot)}
    input_hashes = {item["path"]: item["sha256"] for item in inputs.values()}
    summary = {
        "input_sha256_before": input_hashes.copy(), "input_sha256_after": input_hashes.copy(),
        "code_sha256_before": code_hashes.copy(), "code_sha256_after": code_hashes.copy(),
    }
    record = {"inputs": inputs}
    write_json(root / "experiment.provenance.json", record)
    return repo, root, summary, record, snapshot


def test_run_evidence_binds_all_actual_inputs_and_saved_code(tmp_path, monkeypatch):
    _repo, root, summary, record, snapshot = evidence_fixture(tmp_path, monkeypatch)
    hashes = evaluation.verify_run_evidence(root, summary)
    assert len(hashes) == 10
    item = record["inputs"]["raw_top_cam"]
    evaluation.require_binding(item["path"], item["sha256"], hashes)
    assert hashes[str(snapshot)] == evaluation.sha256(snapshot)
    replacement = root / "same_bytes_elsewhere.bin"
    replacement.write_bytes(Path(item["path"]).read_bytes())
    with pytest.raises(ValueError, match="not the input used"):
        evaluation.require_binding(replacement, item["sha256"], hashes)


@pytest.mark.parametrize("bad", ["input_tamper", "summary_after", "unbound_summary", "code_tamper",
                                 "code_changed", "no_snapshot", "missing_view", "escaped_snapshot"])
def test_run_evidence_rejects_broken_source_binding(tmp_path, monkeypatch, bad):
    _repo, root, summary, record, snapshot = evidence_fixture(tmp_path, monkeypatch)
    item = record["inputs"]["raw_top_cam"]
    if bad == "input_tamper":
        Path(item["path"]).write_bytes(b"different raw")
    elif bad == "summary_after":
        summary["input_sha256_after"][item["path"]] = "0" * 64
    elif bad == "unbound_summary":
        other = root / "another.bin"
        other.write_bytes(b"unregistered input")
        for key in ("input_sha256_before", "input_sha256_after"):
            summary[key][str(other)] = evaluation.sha256(other)
    elif bad == "code_tamper":
        snapshot.write_bytes(b"different source")
    elif bad == "code_changed":
        summary["code_sha256_after"]["labs/tracker.py"] = "0" * 64
    elif bad == "no_snapshot":
        summary["code_sha256_before"] = summary["code_sha256_after"] = {}
    elif bad == "missing_view":
        del record["inputs"]["raw_top_cam"]
        for key in ("input_sha256_before", "input_sha256_after"):
            del summary[key][item["path"]]
        write_json(root / "experiment.provenance.json", record)
    else:
        for key in ("code_sha256_before", "code_sha256_after"):
            summary[key] = {"../tracker.py": "0" * 64}
    with pytest.raises(ValueError):
        evaluation.verify_run_evidence(root, summary)


@pytest.mark.parametrize("bad", [None, "seed", "window", "registration"])
def test_previous_evidence_requires_same_registered_first_pose_and_window(
    tmp_path, monkeypatch, bad,
):
    repo, root, _summary, record, snapshot = evidence_fixture(tmp_path, monkeypatch)
    pose = np.arange(27, dtype=float).reshape(3, 3, 3).tolist()
    monkeypatch.setattr(evaluation, "registered_seed", lambda _frame: (
        np.asarray(pose), {"pose_xyz": pose}, {},
    ))
    for name in ("staged_manifest", "registration"):
        path = repo / f"{name}.json"
        write_json(path, {"registered": True})
        record["inputs"][name] = {"path": str(path), "sha256": evaluation.sha256(path)}
    record["params"] = {
        "registered_window": {"seed_pose_xyz": pose, "start": 76620, "stop": 86580},
        "code_hashes": {str(repo / "labs/tracker.py"): evaluation.sha256(snapshot)},
    }
    if bad == "seed":
        record["params"]["registered_window"]["seed_pose_xyz"] = np.zeros((3, 3, 3)).tolist()
    elif bad == "window":
        record["params"]["registered_window"]["stop"] = 77286
    elif bad == "registration":
        del record["inputs"]["registration"]
    write_json(root / "benchmark.provenance.json", record)
    if bad is None:
        assert len(evaluation.verify_run_evidence(root, {}, previous=True)) == 12
    else:
        with pytest.raises(ValueError):
            evaluation.verify_run_evidence(root, {}, previous=True)

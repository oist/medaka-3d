"""Tiny CPU boundary fixtures for milestone comparison; no recorded candidates."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import compare_fast_tail_continuation as m
import numpy as np
import pytest
from test_compare_fast_tail_two_minutes import candidate


@pytest.fixture(autouse=True)
def no_raw_gpu_or_checkpoint_decode(monkeypatch):
    assert not m.producer.torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU, raw image and numeric checkpoint decode forbidden")

    monkeypatch.setattr(m.producer.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(m.producer.original.cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(m.producer.original.cv2, "imread", forbidden)
    monkeypatch.setattr(m.producer, "read_checkpoint", forbidden)
    yield
    assert not m.producer.torch.cuda.is_initialized()


def fixture(stop=86320):
    source = {str(Path(m.runner.__file__).resolve()): m.RUNNER_SHA}
    binding = {"synthetic": "same_original_binding"}
    metadata = dict(
        context={"synthetic": "same_fast_context"},
        record_fields=list(m.runner.qualified.RECORD_FIELDS),
        state_fields=list(m.producer.original.STATE_FIELDS),
        initialize_count=1,
        restore_count=5,
        previous_frame=stop - 1,
        processed_frames=stop - m.FIRST,
    )
    row = dict(
        arrays=39,
        payload_bytes=1234,
        **{
            k: metadata[k]
            for k in ("initialize_count", "restore_count", "previous_frame", "processed_frames")
        },
    )
    summary = dict(
        schema=m.runner.STAGE + "_milestone",
        status="ok",
        horizon_stop=stop,
        scientific_interval=[m.START, stop],
        native_frames=stop - m.START,
        initialize_count=1,
        restore_count=5,
        parent_audit_sha256=m.AUDIT_SHA,
        external_pose_resets=0,
        finish_calls_so_far=0,
        all39_roundtrip=True,
        prior_history_prefix_unchanged=True,
        physical_ID_truth=False,
        five_minute_candidate=stop == m.runner.STOP,
        whole_process_completion_pending=True,
        source_sha256=source,
        original_binding=binding,
        checkpoint=row,
    )
    fixed = dict(
        schema=m.runner.STAGE + "_checkpoint",
        original_binding=binding,
        source_sha256=source,
        audited_parent_sha256=m.AUDIT_SHA,
        parent_checkpoint_sha256=m.runner.audit.CHECKPOINT_SHA,
        scientific_interval=[m.START, m.runner.STOP],
        process_interval=[m.HALF, m.runner.STOP],
    )
    arrays = {
        "history_offsets": {"shape": [stop - m.FIRST + 1]},
        "state_pose": {"shape": [6, 3, 3, 3]},
    }
    arrays.update({f"other_{i}": {} for i in range(37)})
    manifest = dict(
        schema="medaka_tracker_checkpoint_v1",
        status="ok",
        bindings=fixed,
        bindings_sha256=hashlib.sha256(
            json.dumps(fixed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest(),
        arrays=arrays,
        payload={"bytes": 1234},
    )
    parent = dict(
        metadata=copy.deepcopy(metadata), manifest={"bindings": {"original_binding": binding}}
    )
    parent["metadata"].update(
        restore_count=4, previous_frame=m.HALF - 1, processed_frames=m.HALF - m.FIRST
    )
    return summary, manifest, metadata, parent


@pytest.mark.parametrize("stop,minutes", ((86320, 2), (116200, 5)))
def test_only_same_horizon_complete_native_minute_partitions(stop, minutes):
    windows = m.intervals(stop)
    assert windows["full"] == (m.START, stop)
    assert len(windows) == minutes + 1
    assert [windows[f"minute_{i}"] for i in range(1, minutes + 1)] == [
        (m.START + (i - 1) * 9960, m.START + i * 9960) for i in range(1, minutes + 1)
    ]
    summary, manifest, metadata, parent = fixture(stop)
    m.validate_milestone(summary, stop)
    m.validate_checkpoint(summary, manifest, metadata, parent, stop)


@pytest.mark.parametrize("stop", (76360, 96280, 116199, True, 86320.0))
def test_unregistered_horizons_rejected(stop):
    with pytest.raises(ValueError, match="registered"):
        m.intervals(stop)


@pytest.mark.parametrize(
    "field,value",
    (
        ("restore_count", 4),
        ("initialize_count", 2),
        ("external_pose_resets", 1),
        ("finish_calls_so_far", 1),
        ("status", "incomplete"),
        ("physical_ID_truth", True),
        ("all39_roundtrip", False),
        ("prior_history_prefix_unchanged", False),
        ("horizon_stop", 116200),
        ("native_frames", 19919),
        ("parent_audit_sha256", "0" * 64),
        ("whole_process_completion_pending", False),
        ("five_minute_candidate", True),
        ("restore_count", 5.0),
    ),
)
def test_milestone_must_match_exact_extent_and_lifecycle(field, value):
    summary, _, _, _ = fixture()
    summary[field] = value
    with pytest.raises(ValueError, match="milestone extent/lifecycle/proof"):
        m.validate_milestone(summary, 86320)


def test_milestone_rejects_unqualified_runner_sha():
    summary, _, _, _ = fixture()
    summary["source_sha256"] = {str(Path(m.runner.__file__).resolve()): "0" * 64}
    with pytest.raises(ValueError, match="pinned continuation"):
        m.validate_milestone(summary, 86320)


@pytest.mark.parametrize(
    "change",
    (
        "binding",
        "binding_sha",
        "shape",
        "state_shape",
        "missing_array",
        "context",
        "restore",
        "initialize",
        "previous",
        "processed",
        "record_fields",
        "payload_size",
        "row_restore",
        "parent_binding",
    ),
)
def test_checkpoint_header_disagreement_rejected(change):
    summary, manifest, metadata, parent = fixture()
    if change == "binding":
        manifest["bindings"]["process_interval"] = [66400, 116200]
    elif change == "binding_sha":
        manifest["bindings_sha256"] = "0" * 64
    elif change == "shape":
        manifest["arrays"]["history_offsets"]["shape"][0] -= 1
    elif change == "state_shape":
        manifest["arrays"]["state_pose"]["shape"][0] = 5
    elif change == "missing_array":
        manifest["arrays"].pop("other_0")
    elif change == "context":
        metadata["context"] = {}
    elif change in ("restore", "initialize", "previous", "processed"):
        key = dict(
            restore="restore_count",
            initialize="initialize_count",
            previous="previous_frame",
            processed="processed_frames",
        )[change]
        metadata[key] += 1
    elif change == "record_fields":
        metadata["record_fields"] = metadata["record_fields"][:-1]
    elif change == "payload_size":
        manifest["payload"]["bytes"] += 1
    elif change == "row_restore":
        summary["checkpoint"]["restore_count"] = 4
    else:
        parent["manifest"]["bindings"]["original_binding"] = {"different": True}
    with pytest.raises(ValueError):
        m.validate_checkpoint(summary, manifest, metadata, parent, 86320)


def test_custom_size_bytes_member_is_bound_and_replacement_rejected(tmp_path):
    out = m.proof.Output(tmp_path / "custom")
    row = out.write("nested/value.json", {"value": 3})
    seal = dict(outputs=out.records)
    evidence = m.proof.Evidence()
    path = m.custom_member(seal, out.root, "nested/value.json", evidence)
    assert evidence.paths[str(path)] == row["sha256"]
    path.write_text('{"value":4}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        m.custom_member(seal, out.root, "nested/value.json", evidence)


def test_missing_explicit_final_seal_never_infers_completion(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no final seal should be opened or inferred")

    monkeypatch.setattr(m, "json_input", forbidden)
    assert m.whole_process_completion(tmp_path, tmp_path / "summary.json", "0" * 64, None) is False


def final_fixture(tmp_path):
    branch = tmp_path / "run"
    out = m.proof.Output(branch)
    milestone_path = branch / "milestones/through_116200/summary.json"
    row = out.write("milestones/through_116200/summary.json", {"milestone": "immutable"})
    out.write("candidate.npz", {"synthetic": np.arange(3)})
    final = dict(
        schema=m.runner.STAGE,
        status="ok",
        scientific_interval=[m.START, m.runner.STOP],
        process_interval=[m.HALF, m.runner.STOP],
        native_frames=49800,
        process_frames=39840,
        initialize_count=1,
        restore_count=5,
        external_pose_resets=0,
        new_begin_calls=0,
        finish_calls=1,
        physical_ID_truth=False,
        checkpoints=[
            dict(previous_frame=n - 1) for n in range(m.HALF + 1660, m.runner.STOP + 1660, 1660)
        ],
    )
    return branch, out, milestone_path, row["sha256"], final


@pytest.mark.parametrize(
    "change",
    (None, "failed", "short", "reset", "missing_checkpoint", "different_milestone", "wrong_source"),
)
def test_whole_process_completion_requires_separate_explicit_valid_seal(tmp_path, change):
    branch, out, path, milestone_sha, final = final_fixture(tmp_path)
    if change == "short":
        final["native_frames"] -= 1
    elif change == "reset":
        final["new_begin_calls"] = 1
    elif change == "missing_checkpoint":
        final["checkpoints"].pop(10)
    out.write("summary.json", final)
    seal = dict(
        stage=m.runner.STAGE,
        status="failed" if change == "failed" else "ok",
        outputs=out.records,
        inputs={
            str(Path(m.runner.__file__).resolve()): "0" * 64
            if change == "wrong_source"
            else m.RUNNER_SHA
        },
    )
    seal_path = branch / "tracking.provenance.json"
    seal_path.write_bytes(m.proof.json_bytes(seal))
    sha = m.proof.sha(seal_path)
    if change == "different_milestone":
        milestone_sha = "0" * 64
    evidence = m.proof.Evidence()
    if change is None:
        assert m.whole_process_completion(branch, path, milestone_sha, evidence, sha) is True
    else:
        with pytest.raises(ValueError):
            m.whole_process_completion(branch, path, milestone_sha, evidence, sha)


def test_future_backtrace_change_is_reported_separately_from_snapshot():
    a, _ = candidate(100, 108)
    b = copy.deepcopy(a)
    b["pose_xyz"][3, 2, 0, 0] += 0.5
    before = m.previous.signature(a)
    value, arrays = m.metrics.horizon_changes(a, b, 100, 108)
    assert value["fields"]["pose_xyz"]["changed_rows"] == 1
    assert arrays["pose_xyz_row_changed"].nonzero()[0].tolist() == [3]
    assert m.previous.signature(a) == before

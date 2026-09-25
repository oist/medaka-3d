"""New partial-checkpoint and fixed-window caller contracts; no real payload/GPU."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import run_observation_reserve_continuation as m


def fail(*args, **kwargs):
    pytest.fail("actual data or GPU entry was called")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "PARENT", tmp_path / "parent")
    monkeypatch.setattr(m.np, "load", fail)
    monkeypatch.setattr(m.original, "verified_inputs", fail)
    monkeypatch.setattr(m.original, "SourceSequenceReader", fail)
    monkeypatch.setattr(m.original.cv2, "VideoCapture", fail)
    monkeypatch.setattr(m.prior, "read_checkpoint", fail)
    monkeypatch.setattr(m.torch.cuda, "synchronize", fail)
    original_open = Path.open

    def guarded(path, *args, **kwargs):
        assert path.resolve().is_relative_to(tmp_path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)


@pytest.fixture
def headers(monkeypatch):
    m.PARENT.mkdir()
    (m.PARENT / m.CHECKPOINT_RELATIVE).mkdir(parents=True)
    source = m.WORK / "run_nodewise_body_row_five_minutes.py"
    source.write_bytes(b"synthetic producer only\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(m, "PARENT_SOURCE_SHA", source_sha)
    context = dict(
        implementation="BodyRowContinuousTracker",
        body_row_policy=dict(effective_from_native=66400),
        sparse_distance_backend=dict(effective_from_native=66400),
        instance_rescore_policy=dict(effective_from_native=76360),
    )
    binding = dict(synthetic_original_input=True)
    checkpoint_binding = dict(original_binding=binding, tracker_context=context)
    metadata = dict(
        schema="persistent_tracker_full_state_v1",
        context=context,
        initialize_count=1,
        restore_count=4,
        previous_frame=74699,
        processed_frames=24900,
    )
    plan = dict(
        schema=m.PARENT.name,
        source_sha256={str(source): source_sha},
        tracker_context=context,
        checkpoint_binding=checkpoint_binding,
        original_binding=binding,
        process_interval=[66400, 116200],
        retained_interval=[49800, 116200],
        scientific_interval=[66400, 116200],
        parent_tracking_sha256=m.ORIGINAL_TRACKING_SHA,
        parent_checkpoint_sha256=m.ORIGINAL_CHECKPOINT_SHA,
        initialize_count=1,
        restore_count=4,
        new_begin_calls=0,
        external_pose_resets=0,
        mask_split_priority_applied=False,
    )
    imported = dict(
        schema="nodewise_original_state_import_v1",
        status="ok",
        new_context=context,
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        previous_frame=66399,
        processed_frames=16600,
        logical_initialize_count=1,
        added_restore_count=1,
        new_begin_calls=0,
        imported_metadata=dict(restore_count=4),
        old_history_rewritten=False,
        review_cleared_by_migration=False,
    )
    manifest = dict(
        schema="medaka_tracker_checkpoint_v1",
        status="ok",
        arrays={str(i): {} for i in range(39)},
        bindings=checkpoint_binding,
        metadata={},
        payload=dict(file="arrays.npz", bytes=123, sha256="a" * 64),
    )
    docs = {
        "plan.json": plan,
        "input_index.json": dict(original_binding=binding, prefix_chunks=list(range(40))),
        "state_import.json": imported,
        m.CHECKPOINT_RELATIVE + "/metadata.json": metadata,
        m.CHECKPOINT_RELATIVE + "/manifest.json": manifest,
    }

    def save():
        pins = {}
        # metadata must be recorded before the manifest descriptor is finalized.
        for relative, doc in docs.items():
            if relative.endswith("/manifest.json"):
                continue
            raw = json.dumps(doc, sort_keys=True).encode()
            (m.PARENT / relative).write_bytes(raw)
            pins[relative] = hashlib.sha256(raw).hexdigest()
        meta = m.CHECKPOINT_RELATIVE + "/metadata.json"
        manifest["metadata"] = dict(
            file="metadata.json", sha256=pins[meta], bytes=(m.PARENT / meta).stat().st_size
        )
        raw = json.dumps(manifest, sort_keys=True).encode()
        relative = m.CHECKPOINT_RELATIVE + "/manifest.json"
        (m.PARENT / relative).write_bytes(raw)
        pins[relative] = hashlib.sha256(raw).hexdigest()
        monkeypatch.setattr(m, "PARENT_HEADERS", pins)
        monkeypatch.setattr(m, "CHECKPOINT_SHA", pins[relative])

    save()
    return docs, save


class Evidence:
    def __init__(self):
        self.paths = {}

    def add(self, path, digest):
        path = Path(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        self.paths[str(path)] = digest


def test_committed_partial_parent_needs_no_final_seal_or_payload(headers):
    evidence = Evidence()
    result = m.load_parent_headers(evidence)
    assert len(evidence.paths) == 6
    assert len(result["manifest"]["arrays"]) == 39
    assert result["metadata"]["restore_count"] == 4
    assert not (m.PARENT / "tracking.provenance.json").exists()
    assert not (m.PARENT / m.CHECKPOINT_RELATIVE / "arrays.npz").exists()


@pytest.mark.parametrize(
    "failure", ["restore", "native", "array_count", "sixth_boundary", "history", "prefix"]
)
def test_parent_semantic_mismatch_rejected_even_with_matching_file_hash(headers, failure):
    docs, save = headers
    metadata = docs[m.CHECKPOINT_RELATIVE + "/metadata.json"]
    if failure == "restore":
        metadata["restore_count"] = 5
    elif failure == "native":
        metadata["previous_frame"] = 74700
    elif failure == "array_count":
        docs[m.CHECKPOINT_RELATIVE + "/manifest.json"]["arrays"].pop("38")
    elif failure == "sixth_boundary":
        metadata["context"]["instance_rescore_policy"]["effective_from_native"] = 74700
    elif failure == "history":
        docs["state_import.json"]["old_history_rewritten"] = True
    else:
        docs["input_index.json"]["prefix_chunks"].pop()
    save()
    with pytest.raises(ValueError):
        m.load_parent_headers(Evidence())


class FakeTracker:
    initialize_count = 1
    restore_count = 5

    def __init__(self):
        self.previous_frame = m.START - 1
        self.processed_frames = m.START - m.FIRST
        self.calls = []

    def advance(self, observation):
        assert observation.frame_idx == self.previous_frame + 1
        self.calls.append(observation.frame_idx)
        self.previous_frame = observation.frame_idx
        self.processed_frames += 1


class FakeBatch:
    def __init__(self):
        self.totalframes = self.saved_frames = self.pending_frames = self.flush_calls = 0
        self.rows = []

    def append(self, tracker, native):
        assert tracker.previous_frame == native
        self.totalframes += 1
        self.pending_frames += 1
        value = dict(
            frame_idx=native,
            observation_reserve=dict(
                changed=native == 75053, reason="reserved" if native == 75053 else "no_op"
            ),
        )
        self.rows.append(copy.deepcopy(value))
        return value

    def flush(self):
        self.flush_calls += 1
        self.saved_frames = self.totalframes
        self.pending_frames = 0


def test_all1660_serial_frames_single_flush_and_current_beam_summary():
    tracker, batch = FakeTracker(), FakeBatch()
    counts = dict(guard=0, sync=0)

    def guard():
        counts["guard"] += 1

    def sync():
        counts["sync"] += 1

    iterator = iter(SimpleNamespace(frame_idx=f) for f in range(m.START, m.STOP))
    rows, timing = m.drive(tracker, iterator, batch, synchronize=sync, guard=guard)
    assert tracker.calls == list(range(74700, 76360))
    assert counts == dict(guard=3320, sync=1660)
    assert batch.flush_calls == 1 and batch.saved_frames == 1660
    assert tracker.previous_frame == 76359 and tracker.processed_frames == 26560
    assert all(v >= 0 for v in timing.values()) and "guard_seconds" in timing
    summary = m.reserve_summary(rows)
    assert summary["changed_frames"] == 1 and sum(summary["reasons"].values()) == 1660
    assert summary["current_beam_selection_only"] is True
    assert summary["final_path_reserved_ancestor_count_not_computed"] is True


def test_foreign_native_never_advances_or_saves():
    tracker, batch = FakeTracker(), FakeBatch()
    with pytest.raises(ValueError, match="native frame"):
        m.drive(
            tracker,
            iter([SimpleNamespace(frame_idx=m.START + 1)]),
            batch,
            synchronize=fail,
            guard=fail,
        )
    assert tracker.calls == [] and batch.totalframes == 0


def test_changed_lifecycle_rejected_before_consuming_native():
    tracker, batch = FakeTracker(), FakeBatch()
    tracker.restore_count = 6
    with pytest.raises(ValueError, match="lifecycle"):
        m.drive(tracker, iter(()), batch, synchronize=fail, guard=fail)
    assert tracker.calls == []


def test_fixed_window_and_summary_coverage_reject_partial():
    with pytest.raises(ValueError, match="fixed native"):
        m.drive(
            FakeTracker(), iter(()), FakeBatch(), synchronize=fail, guard=fail, first=m.START + 1
        )
    with pytest.raises(ValueError, match="coverage"):
        m.reserve_summary([])
    with pytest.raises(ValueError, match="coverage"):
        m.proposal_summary([], dict(frame_idx=np.arange(m.START, m.STOP)))

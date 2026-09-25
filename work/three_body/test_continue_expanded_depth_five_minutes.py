"""Synthetic CPU restore/continuation contracts; no scientific payload or GPU."""

import copy
from pathlib import Path
from types import SimpleNamespace

import continue_expanded_depth_five_minutes as m
import numpy as np
import pytest
import test_instance_rescore_tracker as toy


@pytest.fixture
def checkpoint(monkeypatch):
    inputs = toy.inputs.__wrapped__()
    arrays, metadata = copy.deepcopy(toy.checkpoint.__wrapped__(inputs))
    target = m.variant.ExpandedDepthInstanceRescoreContinuousTracker(*inputs)
    shift = m.RESUME - 1 - metadata["previous_frame"]
    arrays["history_frame_idx"] += shift
    metadata["previous_frame"] += shift
    metadata["restore_count"] = 5
    metadata["context"] = target._context()
    monkeypatch.setattr(m, "FIRST", m.RESUME - metadata["processed_frames"])
    # Real frozen decoder/exporter, synthetic five-frame data only.
    target._decode_state(arrays, metadata)
    return target, arrays, metadata


def test_disabled_before_outputs_inputs_and_GPU(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled stage attempted IO or GPU")

    monkeypatch.setattr(m.proof, "Output", forbidden)
    monkeypatch.setattr(m, "load_parent_headers", forbidden)
    monkeypatch.setattr(m, "load_parent_arrays", forbidden)
    monkeypatch.setattr(m.torch.cuda, "set_per_process_memory_fraction", forbidden)
    with pytest.raises(ValueError, match="not enabled"):
        m.main(
            SimpleNamespace(
                execute=True,
                tracker_sha=m.TRACKER_SHA,
                output=tmp_path / "absent",
                receipt_root=tmp_path,
            )
        )
    assert not m.EXECUTION_READY and not (tmp_path / "absent").exists()


def test_actual_source_closure_and_fixed_4milestone_scope():
    code = m.source_closure()
    assert code[str(Path(m.__file__).resolve())] == m.proof.sha(m.__file__)
    assert len(m.PINS) == 9
    assert (m.FIRST, m.START, m.RESUME, m.STOP, m.CHUNK) == (49800, 66400, 78020, 116200, 1660)
    assert m.MILESTONES == (86320, 96280, 106240, 116200)
    assert all((end - m.RESUME) % m.CHUNK == 0 for end in m.MILESTONES)
    assert not m.torch.cuda.is_initialized()


def test_one_real_restore_all39_byte_preserved_and_context_unchanged(
    checkpoint, monkeypatch, tmp_path
):
    tracker, arrays, metadata = checkpoint
    before, original_metadata = m.base.array_signature(arrays), copy.deepcopy(metadata)
    restore = type(tracker).restore_state
    calls = []

    def counted(self, a, meta):
        calls.append(self)
        return restore(self, a, meta)

    monkeypatch.setattr(type(tracker), "restore_state", counted)
    report = m.restore_parent(tracker, arrays, metadata)
    actual, actual_metadata = tracker.export_state()
    assert calls == [tracker]
    assert m.base.array_signature(actual) == before == m.base.array_signature(arrays)
    assert metadata == original_metadata
    assert actual_metadata == {**metadata, "restore_count": 6}
    assert len(actual) == 39 and report["context_unchanged"] and not report["policy_transition"]
    assert tracker.initialize_count == 1 and tracker.restore_count == 6
    assert np.array_equal(actual["state_review"], arrays["state_review"])
    assert all(
        p["effective_from_native"] == 76360 for p in m.policies(actual_metadata["context"]).values()
    )
    saved, saved_meta, row = m.base.preserve_checkpoint(
        tracker,
        dict(arrays=arrays, metadata=metadata),
        tmp_path / "checkpoint",
        dict(synthetic=True),
    )
    assert m.base.array_signature(saved) == before and saved_meta == actual_metadata
    assert row["restore_count"] == 6 and row["processed_frames"] == 5


@pytest.mark.parametrize(
    "damage",
    [
        "restore",
        "initialize",
        "processed",
        "previous",
        "context",
        "depth",
        "sparse",
        "mask",
        "missing_array",
        "review",
        "source",
        "framegap",
    ],
)
def test_parent_damage_rejected_without_successful_restore(checkpoint, damage):
    tracker, arrays, metadata = checkpoint
    if damage == "restore":
        metadata["restore_count"] = 4
    elif damage == "initialize":
        metadata["initialize_count"] = 2
    elif damage == "processed":
        metadata["processed_frames"] += 1
    elif damage == "previous":
        metadata["previous_frame"] -= 1
    elif damage == "context":
        metadata["context"]["implementation"] = "Other"
    elif damage in ("depth", "sparse", "mask"):
        key = {
            "depth": "expanded_depth_policy",
            "sparse": "sparse_distance_backend",
            "mask": "instance_rescore_policy",
        }[damage]
        metadata["context"][key]["effective_from_native"] = m.RESUME
    elif damage == "missing_array":
        arrays.pop("state_pose")
    elif damage == "review":
        arrays["state_review"][0, 2] = False
    elif damage == "source":
        arrays["history_sources"][0, 1, :, 0] = arrays["history_sources"][0, 0, :, 0]
    else:
        arrays["history_frame_idx"][0] -= 1
    with pytest.raises(ValueError):
        m.restore_parent(tracker, arrays, metadata)
    assert tracker.initialize_count == tracker.restore_count == 0


def test_no_second_restore(checkpoint):
    tracker, arrays, metadata = checkpoint
    m.restore_parent(tracker, arrays, metadata)
    with pytest.raises(ValueError, match="fresh"):
        m.restore_parent(tracker, arrays, metadata)
    assert tracker.restore_count == 6


def test_restore_input_mutation_is_detected(checkpoint, monkeypatch):
    tracker, arrays, metadata = checkpoint
    restore = type(tracker).restore_state

    def corrupt(self, a, meta):
        restore(self, a, meta)
        a["previous_images_bgr_top_cam"][0, 0, 0] ^= 1

    monkeypatch.setattr(type(tracker), "restore_state", corrupt)
    with pytest.raises(ValueError, match="input"):
        m.restore_parent(tracker, arrays, metadata)


class FakeTracker:
    def __init__(self):
        self.initialize_count = 1
        self.restore_count = 6
        self.previous_frame = m.RESUME - 1
        self.processed_frames = m.RESUME - m.FIRST
        self.visited = []

    def advance(self, frame):
        assert frame.frame_idx == self.previous_frame + 1
        self.previous_frame = frame.frame_idx
        self.processed_frames += 1
        self.visited.append(frame.frame_idx)


def test_serial_chunks_use_one_tracker_without_restore_or_reset(monkeypatch):
    tracker, saved, guards, sync = FakeTracker(), [], [], []
    monkeypatch.setattr(
        m.diagnostic, "save_diagnostic", lambda out, target, f: saved.append(f) or dict(frame_idx=f)
    )
    iterator = iter(SimpleNamespace(frame_idx=f) for f in range(m.RESUME, m.RESUME + 5))
    all_rows = []
    for first, stop in ((m.RESUME, m.RESUME + 2), (m.RESUME + 2, m.RESUME + 5)):
        rows, _ = m.drive_chunk(
            tracker,
            iterator,
            first,
            stop,
            None,
            synchronize=lambda: sync.append(True),
            guard=lambda: guards.append(True),
        )
        all_rows.extend(rows)
    assert tracker.visited == saved == list(range(m.RESUME, m.RESUME + 5))
    assert len(all_rows) == len(sync) == 5 and len(guards) == 10
    assert tracker.initialize_count == 1 and tracker.restore_count == 6


def test_skipped_native_rejected_before_advance():
    tracker = FakeTracker()
    with pytest.raises(ValueError, match="skipped or reordered"):
        m.drive_chunk(
            tracker,
            iter([SimpleNamespace(frame_idx=m.RESUME + 1)]),
            m.RESUME,
            m.RESUME + 1,
            None,
            synchronize=lambda: None,
            guard=lambda: None,
        )
    assert not tracker.visited


@pytest.mark.parametrize(
    "field,value", [("initialize_count", 2), ("restore_count", 7), ("processed_frames", 1)]
)
def test_lifecycle_drift_rejected(field, value):
    tracker = FakeTracker()
    setattr(tracker, field, value)
    with pytest.raises(ValueError, match="lifecycle"):
        m.lifecycle(tracker, m.RESUME)


def test_pure_saved_milestone_without_tracker_or_finish(checkpoint, monkeypatch, tmp_path):
    tracker, arrays, metadata = checkpoint
    m.restore_parent(tracker, arrays, metadata)
    arrays, metadata = tracker.export_state()
    before = m.base.array_signature(arrays)
    monkeypatch.setattr(m, "MILESTONES", (m.RESUME,))
    binding = dict(
        source_sha256={"synthetic": "a" * 64},
        parent_qa={"synthetic": True},
        original_binding={"synthetic": True},
        **m.policies(metadata["context"]),
    )
    out = m.proof.Output(tmp_path / "out")
    m.save_milestone(out, arrays, metadata, dict(previous_frame=m.RESUME - 1), binding, m.RESUME)
    report = m.proof.read_json(
        Path(out.records[f"milestones/through_{m.RESUME:06d}/summary.json"]["path"])
    )
    assert report["finish_calls_so_far"] == 0 and report["whole_process_completion_pending"]
    assert report["restore_count"] == 6 and not report["policy_transition"]
    assert len(report["candidate_arrays"]) == 21
    assert m.base.array_signature(arrays) == before
    with pytest.raises(ValueError, match="boundary"):
        m.save_milestone(out, arrays, metadata, {}, binding, m.RESUME + 1)


@pytest.mark.parametrize(
    "field", ["producer_seal_sha256", "qa_source_sha256", "comparison_seal_sha256", "status"]
)
def test_wrong_qa_rejected_without_checkpoint_read(monkeypatch, field):
    qa = dict(
        status="ok",
        producer_seal_sha256=m.PARENT_SHA,
        qa_source_sha256=m.QA_SOURCE_SHA,
        comparison_seal_sha256=m.PARENT_COMPARISON_SHA,
        selected_candidates=dict(path=str(m.QA_SELECTED), sha256=m.QA_SELECTED_SHA),
        diagnostics=dict(frames=1660, all_finished_rows_uniquely_matched=True),
        all21_final_candidate_arrays_verified=True,
        imported39_signatures_and_metadata_bound=True,
    )
    qa[field] = "incorrect"
    comparison = dict(
        status="ok",
        stage=m.PARENT_COMPARISON.parent.name,
        inputs={str(m.PARENT / "tracking.provenance.json"): m.PARENT_SHA},
    )
    monkeypatch.setattr(m.proof, "read_json", lambda path: qa if path == m.QA else comparison)
    with pytest.raises(ValueError, match="bound parent QA"):
        m.load_qa(SimpleNamespace(add=lambda *args: None))


@pytest.mark.parametrize("bad_finish", [False, True])
def test_full_small_main_checkpoints_milestones_one_finish_and_failure_receipt(
    tmp_path, monkeypatch, bad_finish
):
    for name, value in dict(
        FIRST=0, START=2, RESUME=4, STOP=10, CHUNK=2, MILESTONES=(6, 10), EXECUTION_READY=True
    ).items():
        monkeypatch.setattr(m, name, value)
    context = {
        "instance_rescore_policy": {"effective_from_native": 76360},
        "sparse_distance_backend": {"effective_from_native": 76360},
        "expanded_depth_policy": {"effective_from_native": 76360, "expanded_lower_z_mm": -12.0},
    }
    initial_metadata = dict(
        context=context, initialize_count=1, restore_count=5, previous_frame=3, processed_frames=4
    )
    binding = dict(opencv_threads=1, torch_cpu_threads=1)
    code = {str(Path(m.__file__).resolve()): m.proof.sha(m.__file__)}
    tracker = FakeTracker()
    restores, finishes, checkpoints, closes = [], [], [], []

    def candidate(stop):
        return {
            "frame_idx": np.arange(stop, dtype=np.int64),
            **{f"v{k}": np.zeros(stop, np.float64) for k in range(20)},
        }

    def finish():
        assert tracker.previous_frame == 9
        assert checkpoints == [6, 8, 10]
        finishes.append(True)
        result = candidate(10)
        if bad_finish:
            result["v0"][0] = 1
        return result

    def restore(target, arrays, metadata):
        assert target is tracker and metadata == initial_metadata
        restores.append(True)
        return dict(status="ok", synthetic=True)

    def preserve(target, parent, path, checkpoint_binding):
        stop = target.previous_frame + 1
        assert parent["metadata"]["previous_frame"] == stop - 3
        assert checkpoint_binding["policy_transition"] is False and not finishes
        checkpoints.append(stop)
        path.mkdir(parents=True)
        for name in ("manifest.json", "metadata.json", "arrays.npz"):
            (path / name).write_bytes(b"synthetic checkpoint placeholder")
        metadata = dict(
            context=context,
            initialize_count=1,
            restore_count=6,
            previous_frame=stop - 1,
            processed_frames=stop,
        )
        return (
            dict(stop=stop),
            metadata,
            dict(previous_frame=stop - 1, initialize_count=1, restore_count=6),
        )

    def frames(reader, *, first, stop):
        try:
            for f in range(first, stop):
                yield SimpleNamespace(frame_idx=f)
        finally:
            closes.append(stop)

    def exact(actual, expected, stop):
        assert stop == 10 and len(actual) == len(expected) == 21 and len(finishes) == 1
        return dict(exact=m.base.array_signature(actual) == m.base.array_signature(expected))

    tracker.finish = finish
    tracker._context = lambda: context
    monkeypatch.setattr(m, "source_closure", lambda: code)
    monkeypatch.setattr(m.prior, "local_code", lambda: code)
    monkeypatch.setattr(
        m,
        "load_parent_headers",
        lambda evidence: dict(
            manifest=dict(bindings=dict(original_binding=binding)), parent_qa=dict(synthetic=True)
        ),
    )
    monkeypatch.setattr(
        m, "load_parent_arrays", lambda headers, evidence: ({"synthetic": True}, initial_metadata)
    )
    monkeypatch.setattr(m, "restore_parent", restore)
    monkeypatch.setattr(
        m.variant, "ExpandedDepthInstanceRescoreContinuousTracker", lambda *args, **kwargs: tracker
    )
    monkeypatch.setattr(
        m.qualified,
        "ReceiptStore",
        lambda root: SimpleNamespace(final_guard=lambda: None, evidence_hashes={}, uses=[]),
    )
    roots = list(range(0, 80, 2))
    inputs = SimpleNamespace(
        chunk_roots=roots,
        chunk_records=[{} for _ in roots],
        input_root=tmp_path,
        rig=None,
        initial_pose_xyz=np.zeros((3, 3, 3)),
    )
    protocol = dict(tracker_config={}, node_recovery_config={}, persistence_config={})
    monkeypatch.setattr(m.original, "verified_inputs", lambda *args, **kwargs: (inputs, protocol))
    monkeypatch.setattr(
        m.original,
        "SourceSequenceReader",
        lambda first, root: SimpleNamespace(frames=[first, first + 1]),
    )
    monkeypatch.setattr(m.original, "bounded_frames", frames)
    monkeypatch.setattr(m.base, "preserve_checkpoint", preserve)
    monkeypatch.setattr(
        m.original.phase.COMMON, "traced_candidate", lambda arrays: candidate(arrays["stop"])
    )
    monkeypatch.setattr(m.prior, "compare_candidate", exact)
    monkeypatch.setattr(m.diagnostic, "save_diagnostic", lambda out, target, f: dict(frame_idx=f))
    for obj, names in (
        (m.original.cv2, ("setNumThreads",)),
        (m.torch, ("set_num_threads", "set_num_interop_threads")),
        (
            m.torch.cuda,
            ("set_per_process_memory_fraction", "reset_peak_memory_stats", "synchronize"),
        ),
        (m.qualified, ("check_headroom", "check_memory")),
        (m.prior, ("runtime", "smi_sample", "cuda_headroom")),
        (m.qualified.qualification, ("memory_sample",)),
    ):
        for name in names:
            monkeypatch.setattr(obj, name, lambda *args, **kwargs: {})
    destination = tmp_path / "small_run"
    args = SimpleNamespace(
        execute=True, tracker_sha=m.TRACKER_SHA, output=destination, receipt_root=tmp_path
    )
    if bad_finish:
        with pytest.raises(ValueError, match="finish changed"):
            m.main(args)
    else:
        m.main(args)
    seal = m.proof.read_json(destination / "tracking.provenance.json")
    assert seal["status"] == ("failed" if bad_finish else "ok")
    assert restores == finishes == [True] and closes == checkpoints == [6, 8, 10]
    assert tracker.visited == list(range(4, 10))
    for stop in (6, 10):
        saved = m.proof.read_json(destination / f"milestones/through_{stop:06d}/summary.json")
        assert saved["finish_calls_so_far"] == 0 and saved["whole_process_completion_pending"]
    if bad_finish:
        assert len(seal["committed_checkpoints"]) == 3
        assert not (destination / "candidate.npz").exists()
        assert not (destination / "summary.json").exists()
    else:
        saved = m.proof.read_json(destination / "summary.json")
        assert saved["finish_calls"] == 1 and saved["diagnostic_frames"] == 6
        assert saved["restore_count"] == 6 and not saved["policy_transition"]

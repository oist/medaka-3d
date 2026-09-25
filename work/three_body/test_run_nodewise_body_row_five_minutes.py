"""CPU contracts with synthetic frames and isolated, fail-spied I/O boundaries."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import run_nodewise_body_row_five_minutes as m


@pytest.fixture(autouse=True)
def isolation(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("actual input, checkpoint, or GPU execution is forbidden")

    for target in (m, m.old, m.base, m.proof, m.original.phase):
        for key in ("WORK", "REPO"):
            if hasattr(target, key):
                monkeypatch.setattr(target, key, tmp_path)
    monkeypatch.setattr(m, "PARENT", tmp_path / "synthetic_parent")
    for target, name in (
        (m, "source_code"),
        (m, "load_parent_headers"),
        (m, "load_parent_payload"),
        (m.original, "verified_inputs"),
        (m.original, "verify_previous_observation"),
        (m.original, "SourceSequenceReader"),
        (m.original, "bounded_frames"),
        (m.original, "AcceleratedContinuousPersistentNodeTracker"),
        (m.variant, "BodyRowContinuousTracker"),
        (m.migration, "import_original_state"),
        (m.prior, "read_checkpoint"),
        (m.prior, "runtime"),
        (m.prior, "smi_sample"),
        (m.prior, "cuda_headroom"),
        (m.prior, "local_code"),
        (m.base, "preserve_checkpoint"),
        (m.torch.cuda, "synchronize"),
        (m.torch.cuda, "set_per_process_memory_fraction"),
        (m.torch.cuda, "reset_peak_memory_stats"),
        (np, "load"),
    ):
        monkeypatch.setattr(target, name, forbidden)


def chunk_records():
    return [
        {"index": i, "start": 49800 + i * 1660, "stop": 49800 + (i + 1) * 1660} for i in range(40)
    ]


def test_exact_window_prefix_and_milestones():
    assert len(m.chunks()) == 30
    assert m.chunks()[0] == (66400, 68060) and m.chunks()[-1] == (114540, 116200)
    assert sum(b - a for a, b in m.chunks()) == 49800
    assert m.STOP - m.FIRST == 66400
    assert m.MILESTONES == (76360, 86320, 96280, 106240, 116200)
    records = chunk_records()
    inputs = SimpleNamespace(
        chunk_roots=list(range(40)),
        chunk_records=copy.deepcopy(records),
        frame_idx=np.arange(m.FIRST, m.STOP),
    )
    headers = {"consumed": copy.deepcopy(records[:10])}
    m.validate_source_prefix(inputs, headers)
    inputs.chunk_records[20]["start"] += 1
    with pytest.raises(ValueError, match=r"skipped|reordered"):
        m.validate_source_prefix(inputs, headers)
    inputs.chunk_records = copy.deepcopy(records)
    inputs.frame_idx[100] += 1
    with pytest.raises(ValueError, match="common input prefix"):
        m.validate_source_prefix(inputs, headers)


class SerialTracker:
    def __init__(self, first=66400):
        self.initialize_count = 1
        self.restore_count = 4
        self.previous_frame = first - 1
        self.processed_frames = first - 49800
        self.seen = []

    def advance(self, frame):
        self.seen.append(frame.frame_idx)
        self.previous_frame = frame.frame_idx
        self.processed_frames += 1


class CountBatch:
    def __init__(self, out=None, batch_size=100):
        self.out = out
        self.batch_size = batch_size
        self.totalframes = self.saved_frames = self.pending_frames = 0
        self.records = []
        self.batches = []
        self.flushes = 0

    def append(self, tracker, native):
        assert tracker.previous_frame == native
        row = {"frame_idx": native, "body_row": {"admitted": 1 if native % 2 else 0, "headroom": 6}}
        self.totalframes += 1
        self.pending_frames += 1
        self.records.append(row)
        if self.pending_frames == self.batch_size:
            self.flush()
        return row

    def flush(self):
        self.flushes += 1
        if self.pending_frames:
            self.batches.append(
                {
                    "frame_count": self.pending_frames,
                    "first_frame": self.records[-self.pending_frames]["frame_idx"],
                    "stop_frame_exclusive": self.records[-1]["frame_idx"] + 1,
                }
            )
            self.saved_frames += self.pending_frames
            self.pending_frames = 0


def test_real_serial_chunk_lifecycle_and_full_diagnostic_flush():
    tracker = SerialTracker()
    batch = CountBatch()
    events = []
    frames = iter(SimpleNamespace(frame_idx=i) for i in range(66400, 68060))
    rows, times = m.drive_chunk(
        tracker,
        frames,
        batch,
        66400,
        68060,
        synchronize=lambda: events.append("sync"),
        guard=lambda: events.append("guard"),
    )
    assert tracker.seen == list(range(66400, 68060)) == [r["frame_idx"] for r in rows]
    assert events == ["guard", "sync", "guard"] * 1660
    assert (
        tracker.initialize_count,
        tracker.restore_count,
        tracker.previous_frame,
        tracker.processed_frames,
    ) == (1, 4, 68059, 18260)
    assert batch.totalframes == batch.saved_frames == 1660 and batch.pending_frames == 0
    assert [b["frame_count"] for b in batch.batches] == [100] * 16 + [60]
    assert set(times) == {
        "advance_seconds",
        "native_next_seconds",
        "diagnostic_save_seconds",
        "guard_seconds",
    }


@pytest.mark.parametrize("damage", ["order", "restore", "unsaved"])
def test_chunk_rejects_bad_observation_lifecycle_or_uncommitted_batch(damage):
    tracker = SerialTracker()
    batch = CountBatch()
    if damage == "restore":
        tracker.restore_count = 5
    if damage == "unsaved":
        batch.flush = lambda: None
    frames = [SimpleNamespace(frame_idx=i) for i in range(66400, 68060)]
    if damage == "order":
        frames[4].frame_idx += 1
    with pytest.raises(ValueError):
        m.drive_chunk(
            tracker, iter(frames), batch, 66400, 68060, synchronize=lambda: None, guard=lambda: None
        )
    if damage == "order":
        assert tracker.seen == list(range(66400, 66404))
    if damage == "restore":
        assert tracker.seen == []


def setup_main(tmp_path, monkeypatch, *, fail_chunk=None):
    """Explicit trust/compute stubs; actual main writes its complete output contract."""
    events = []
    output = tmp_path / "synthetic_run"
    records = chunk_records()
    inputs = SimpleNamespace(
        chunk_roots=list(range(40)),
        chunk_records=records,
        frame_idx=np.arange(m.FIRST, m.STOP),
        rig=object(),
        initial_pose_xyz=np.zeros((3, 3, 3)),
        input_root=tmp_path,
    )
    binding = {"opencv_threads": 1, "torch_cpu_threads": 1}
    headers = {"binding": binding, "consumed": copy.deepcopy(records[:10])}
    protocol = {k: {} for k in ("tracker_config", "node_recovery_config", "persistence_config")}
    monkeypatch.setattr(m, "source_code", lambda e: {})
    monkeypatch.setattr(m, "load_parent_headers", lambda e: headers)
    monkeypatch.setattr(
        m,
        "load_parent_payload",
        lambda h, e: {"arrays": {"synthetic": np.array([1])}, "metadata": {"restore_count": 3}},
    )
    monkeypatch.setattr(m.original, "verified_inputs", lambda *a, **k: (inputs, protocol))
    monkeypatch.setattr(m.original, "verify_previous_observation", lambda *a: {"synthetic": True})
    monkeypatch.setattr(
        m.qualified,
        "ReceiptStore",
        lambda p: SimpleNamespace(
            final_guard=lambda: events.append("receipt_guard"), evidence_hashes={}, uses=[]
        ),
    )
    for target, key in (
        (m.original.cv2, "setNumThreads"),
        (m.torch, "set_num_threads"),
        (m.torch, "set_num_interop_threads"),
        (m.torch.cuda, "reset_peak_memory_stats"),
        (m.torch.cuda, "synchronize"),
    ):
        monkeypatch.setattr(target, key, lambda *a: None)
    monkeypatch.setattr(
        m.torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: events.append(("fraction", fraction, device)),
    )
    monkeypatch.setattr(m.prior, "runtime", lambda *a, **k: {"synthetic_gpu": k["gpu"]})
    monkeypatch.setattr(m.prior, "smi_sample", lambda: {})
    monkeypatch.setattr(m.prior, "cuda_headroom", lambda: {})
    monkeypatch.setattr(m.qualified, "check_headroom", lambda *a, **k: None)
    monkeypatch.setattr(m.qualified.qualification, "memory_sample", lambda: {})
    monkeypatch.setattr(m.qualified, "check_memory", lambda *a: None)
    monkeypatch.setattr(m.qualified, "TrackerConfig", lambda **k: SimpleNamespace(device="cuda"))
    monkeypatch.setattr(m.qualified, "NodeRecoveryConfig", lambda **k: object())
    monkeypatch.setattr(m.qualified, "PersistentNodeConfig", lambda **k: object())
    monkeypatch.setattr(
        m.original,
        "AcceleratedContinuousPersistentNodeTracker",
        lambda *a, **k: SimpleNamespace(initialize_count=0),
    )
    tracker = SerialTracker()
    tracker.initialize_count = 0
    tracker.restore_count = 0
    tracker._context = lambda: {"instance_rescore_policy": {"effective_from_native": 76360}}
    tracker._decode_state = lambda *a: events.append("decode_saved")

    def make_tracker(*a, **k):
        assert k["body_row_effective_from_native"] == k["sparse_effective_from_native"] == 66400
        return tracker

    monkeypatch.setattr(m.variant, "BodyRowContinuousTracker", make_tracker)

    def migrate(target, source, arrays, metadata):
        assert target is tracker and source.initialize_count == 0 and metadata["restore_count"] == 3
        target.initialize_count = 1
        target.restore_count = 4
        events.append("migrate_once")
        return {"synthetic": True, "restore_count": 4}

    monkeypatch.setattr(m.migration, "import_original_state", migrate)

    def reader(index, root):
        assert root == tmp_path
        events.append(("reader", index))
        return SimpleNamespace(
            frames=np.arange(m.FIRST + index * 1660, m.FIRST + (index + 1) * 1660)
        )

    monkeypatch.setattr(m.original, "SourceSequenceReader", reader)

    class Frames:
        def __init__(self, first, stop):
            self.first = first
            self.stop = stop

        def close(self):
            events.append(("close", self.stop))

    monkeypatch.setattr(m.original, "bounded_frames", lambda r, first, stop: Frames(first, stop))
    monkeypatch.setattr(m.batching, "NodewiseDiagnosticBatch", CountBatch)

    def drive(target, iterator, batch, first, stop, **kwargs):
        assert (iterator.first, iterator.stop) == (first, stop)
        m.lifecycle(target, first)
        if first == fail_chunk:
            raise RuntimeError("synthetic chunk failure")
        rows = []
        for native in range(first, stop):
            target.advance(SimpleNamespace(frame_idx=native))
            rows.append(batch.append(target, native))
        batch.flush()
        m.lifecycle(target, stop)
        return rows, {
            k: 0.0
            for k in (
                "advance_seconds",
                "native_next_seconds",
                "diagnostic_save_seconds",
                "guard_seconds",
            )
        }

    monkeypatch.setattr(m, "drive_chunk", drive)

    def preserve(target, parent, path, new_binding):
        through = target.previous_frame + 1
        assert new_binding["retained_interval"] == [49800, 116200]
        assert new_binding["process_interval"] == [66400, 116200]
        assert parent["metadata"]["restore_count"] in (3, 4)
        path.mkdir(parents=True)
        for name in ("manifest.json", "metadata.json", "arrays.npz"):
            (path / name).write_bytes(b"SYNTHETIC CHECKPOINT NEVER DECODED")
        events.append(("checkpoint", through))
        return (
            {"through": through},
            {"restore_count": 4},
            {"path": str(path), "previous_frame": through - 1},
        )

    monkeypatch.setattr(m.base, "preserve_checkpoint", preserve)

    def traced(saved):
        frames = np.arange(m.FIRST, saved["through"])
        kind = np.full(len(frames), "carry", dtype="U8")
        kind[(frames >= m.START) & (frames % 3 == 0)] = "body_row"
        return {"frame_idx": frames, "proposal_kind": kind}

    monkeypatch.setattr(m.original.phase.COMMON, "traced_candidate", traced)

    def finish():
        events.append("finish")
        assert tracker.previous_frame == 116199
        return traced({"through": 116200})

    tracker.finish = finish

    def compare(a, b, stop):
        assert stop == 116200 and all(np.array_equal(a[k], b[k]) for k in a)
        return {"exact": True}

    monkeypatch.setattr(m.prior, "compare_candidate", compare)
    monkeypatch.setattr(m.prior, "local_code", lambda: {})
    return (
        SimpleNamespace(output=output, receipt_root=tmp_path / "synthetic_receipts"),
        events,
        tracker,
    )


def test_all_thirty_chunks_same_state_full_diagnostics_and_one_finish(tmp_path, monkeypatch):
    args, events, tracker = setup_main(tmp_path, monkeypatch)
    m.main(args)
    seal = json.loads((args.output / "tracking.provenance.json").read_bytes())
    summary = json.loads((args.output / "summary.json").read_bytes())
    batches = json.loads((args.output / "diagnostic_batches.json").read_bytes())
    assert seal["status"] == "ok"
    assert [event for event in events if isinstance(event, tuple) and event[0] == "reader"] == [
        ("reader", i) for i in range(10, 40)
    ]
    assert events.count("migrate_once") == events.count("finish") == 1
    assert ("fraction", 0.25, 0) in events
    assert tracker.seen == list(range(66400, 116200))
    assert summary["checkpoint_count"] == 30 and summary["diagnostic_frames"] == 49800
    assert summary["initialize_count"] == 1 and summary["restore_count"] == 4
    assert summary["new_begin_calls"] == summary["external_pose_resets"] == 0
    assert summary["finish_calls"] == 1 and summary["retained_interval"] == [49800, 116200]
    assert batches["native_frames"] == sum(b["frame_count"] for b in batches["batches"]) == 49800
    assert len(batches["batches"]) == 30 * 17
    assert all("frame_count" in b and "body_row" not in b for b in batches["batches"])
    assert [b["frame_count"] for b in batches["batches"]] == ([100] * 16 + [60]) * 30
    assert (
        len(
            [
                name
                for name in seal["outputs"]
                if name.startswith("milestones/") and name.endswith("summary.json")
            ]
        )
        == 5
    )
    assert summary["body_row"]["frames_with_additions"] == 24900
    assert summary["body_row"]["final_selected_body_row_frame_count"] == 16600


def test_failure_preserves_last_committed_checkpoint_without_finish(tmp_path, monkeypatch):
    args, events, _ = setup_main(tmp_path, monkeypatch, fail_chunk=68060)
    with pytest.raises(RuntimeError, match="synthetic chunk failure"):
        m.main(args)
    seal = json.loads((args.output / "tracking.provenance.json").read_bytes())
    assert seal["status"] == "failed"
    assert seal["committed_checkpoint"]["previous_frame"] == 68059
    assert ("close", 69720) in events and "finish" not in events
    assert len([e for e in events if isinstance(e, tuple) and e[0] == "checkpoint"]) == 1
    assert "summary.json" not in seal["outputs"] and "candidate.npz" not in seal["outputs"]
    assert "checkpoints/through_068060/manifest.json" in seal["outputs"]


def test_fresh_output_guard_precedes_source_and_input_reads(tmp_path):
    for path in (tmp_path, tmp_path.parent / "outside_forbidden"):
        with pytest.raises(ValueError, match="fresh workspace"):
            m.main(SimpleNamespace(output=path, receipt_root=tmp_path / "receipts"))

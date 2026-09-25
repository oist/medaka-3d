"""CPU lifecycle tests for full four-minute continuation; never run a fit."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import continue_fast_tail_five_minutes as m
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    assert not m.torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU-only tests must not initialize CUDA")

    for name in ("is_available", "synchronize", "set_per_process_memory_fraction"):
        monkeypatch.setattr(m.torch.cuda, name, forbidden)
    yield
    assert not m.torch.cuda.is_initialized()


class SerialTracker:
    initialize_count = 1
    restore_count = 5

    def __init__(self):
        self.previous_frame = m.RESUME - 1
        self.processed_frames = m.RESUME - m.FIRST
        self.visited = []

    def advance(self, frame):
        assert frame.frame_idx == self.previous_frame + 1
        self.visited.append(frame.frame_idx)
        self.previous_frame = frame.frame_idx
        self.processed_frames += 1

    def begin(self, *args):
        raise AssertionError("ID reinitialization forbidden")

    def restore_state(self, *args):
        raise AssertionError("mid-process restore forbidden")

    def finish(self):
        raise AssertionError("mid-process finish forbidden")


def drive(tracker, frames, first, stop, **overrides):
    opts = dict(synchronize=lambda: None, guard=lambda: None, progress=lambda stop: None)
    opts.update(overrides)
    return m.drive(tracker, iter(frames), first, stop, **opts)


def test_all39840_native_frames_across24_chunks_keep_single_tracker():
    tracker = SerialTracker()
    guards, syncs, reported, endpoints = [], [], [], []
    totals = dict(advance_seconds=0.0, native_next_seconds=0.0)
    for first in range(m.RESUME, m.STOP, m.CHUNK):
        stop = first + m.CHUNK
        frames = (SimpleNamespace(frame_idx=n) for n in range(first, stop))
        measured = drive(
            tracker,
            frames,
            first,
            stop,
            guard=lambda: guards.append(tracker.previous_frame),
            synchronize=lambda: syncs.append(tracker.previous_frame),
            progress=reported.append,
        )
        for key in totals:
            totals[key] += measured[key]
        endpoints.append(stop)
    assert len(endpoints) == 24 and endpoints[-1] == 116200
    assert tracker.visited == list(range(76360, 116200))
    assert tracker.processed_frames == 66400
    assert tracker.initialize_count == 1 and tracker.restore_count == 5
    assert guards == [x for n in range(76360, 116200) for x in (n - 1, n)]
    assert syncs == tracker.visited
    assert set(m.MILESTONES) <= set(endpoints)
    assert reported[-1] == 116200 and all(v >= 0 for v in totals.values())


@pytest.mark.parametrize("native", [m.RESUME - 1, m.RESUME + 1])
def test_wrong_native_is_rejected_before_advancing(native):
    tracker = SerialTracker()
    with pytest.raises(ValueError, match="skipped or reordered"):
        drive(tracker, [SimpleNamespace(frame_idx=native)], m.RESUME, m.RESUME + 1)
    assert tracker.visited == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("initialize_count", 2),
        ("restore_count", 6),
        ("previous_frame", m.RESUME - 2),
        ("processed_frames", 0),
    ],
)
def test_invalid_lifecycle_stops_before_read(field, value):
    tracker = SerialTracker()
    setattr(tracker, field, value)

    def unreadable():
        raise AssertionError("must reject before decoding")
        yield

    with pytest.raises(ValueError, match="lifecycle"):
        drive(tracker, unreadable(), m.RESUME, m.RESUME + 1)


def test_early_eof_is_not_success():
    tracker = SerialTracker()
    with pytest.raises(StopIteration):
        drive(tracker, [], m.RESUME, m.RESUME + 1)
    assert tracker.visited == []


def test_resource_failure_does_not_decode_next_frame():
    tracker, reads = SerialTracker(), []

    def frames():
        for n in range(m.RESUME, m.RESUME + 2):
            reads.append(n)
            yield SimpleNamespace(frame_idx=n)

    def guard():
        if tracker.previous_frame == m.RESUME:
            raise ValueError("resource limit")

    with pytest.raises(ValueError, match="resource limit"):
        drive(tracker, frames(), m.RESUME, m.RESUME + 2, guard=guard)
    assert reads == tracker.visited == [m.RESUME]


def restore_fixture():
    arrays = {f"field_{n}": np.array([0.0, float(n)], dtype=np.float64) for n in range(39)}
    metadata = dict(
        initialize_count=1,
        restore_count=4,
        previous_frame=m.RESUME - 1,
        processed_frames=m.RESUME - m.FIRST,
        context={"review": True},
    )
    return arrays, metadata


class RestoreTracker:
    initialize_count = 0

    def __init__(self, mutation=None):
        self.mutation = mutation

    def restore_state(self, arrays, metadata):
        self.arrays = copy.deepcopy(arrays)
        self.metadata = copy.deepcopy(metadata)
        self.metadata["restore_count"] += 1
        if self.mutation == "input_array":
            arrays["field_0"][0] = -0.0
        elif self.mutation == "input_metadata":
            metadata["context"]["review"] = False
        elif self.mutation == "output_array":
            self.arrays["field_38"][0] = -0.0
        elif self.mutation == "output_metadata":
            self.metadata["context"]["review"] = False
        elif self.mutation == "missing":
            del self.arrays["field_38"]
        for key in ("initialize_count", "restore_count", "previous_frame", "processed_frames"):
            setattr(self, key, self.metadata[key])

    def export_state(self):
        return self.arrays, self.metadata


def test_restore_preserves_all39_inputs_and_changes_only_restore_count():
    arrays, metadata = restore_fixture()
    before = m.array_signature(arrays), copy.deepcopy(metadata)
    tracker = RestoreTracker()
    result = m.restore_exact(tracker, arrays, metadata)
    assert result["all39_BYTEEXACT"] and result["new_begin_calls"] == 0
    assert tracker.restore_count == 5 and tracker.initialize_count == 1
    assert (m.array_signature(arrays), metadata) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "input_array",
        "input_metadata",
        "output_array",
        "output_metadata",
        "missing",
    ],
)
def test_restore_rejects_input_and_output_mutations(mutation):
    arrays, metadata = restore_fixture()
    with pytest.raises(ValueError, match="restore"):
        m.restore_exact(RestoreTracker(mutation), arrays, metadata)


def test_nonfresh_target_is_rejected():
    arrays, metadata = restore_fixture()
    tracker = RestoreTracker()
    tracker.initialize_count = 1
    with pytest.raises(ValueError, match="fresh"):
        m.restore_exact(tracker, arrays, metadata)

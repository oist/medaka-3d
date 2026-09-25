"""Continuation/qualification failure tests without GPU, fitting, or real checkpoint IO."""

from types import SimpleNamespace

import continue_sparse_instance_five_minutes as m
import numpy as np
import pytest


class FakeTracker:
    def __init__(self, sparse, drift=0):
        self.sparse = sparse
        self.drift = drift
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

    def export_state(self):
        arrays = {f"field{i}": np.array([self.previous_frame], dtype=np.int64) for i in range(39)}
        arrays["field38"][0] += self.drift
        context = dict(implementation="Sparse" if self.sparse else "Dense", config="same")
        if self.sparse:
            context["sparse_distance_backend"] = dict(effective_from_native=m.RESUME)
        return arrays, dict(
            context=context,
            initialize_count=self.initialize_count,
            restore_count=self.restore_count,
            previous_frame=self.previous_frame,
            processed_frames=self.processed_frames,
        )


@pytest.fixture
def setup(monkeypatch):
    saved, outputs, cleared = [], {}, []
    monkeypatch.setattr(
        m.diagnostic,
        "save_diagnostic",
        lambda out, tracker, frame: saved.append(frame) or dict(frame_idx=frame),
    )
    monkeypatch.setattr(m.torch.cuda, "empty_cache", lambda: cleared.append(True))
    out = SimpleNamespace(write=lambda name, value: outputs.setdefault(name, value))
    return saved, outputs, cleared, out


def test_two_frame_control_then_same_iterator_and_sparse_tracker_continue(setup):
    saved, outputs, cleared, out = setup
    tracker, control = FakeTracker(True), FakeTracker(False)
    original_control = control
    frames = iter(SimpleNamespace(frame_idx=f) for f in range(m.RESUME, m.RESUME + 5))
    control, first_rows, _ = m.drive_chunk(
        tracker,
        control,
        frames,
        m.RESUME,
        m.RESUME + 2,
        out,
        synchronize=lambda: None,
        guard=lambda: None,
    )
    assert control is None and original_control.visited == [m.RESUME, m.RESUME + 1]
    assert outputs["serial_dense_sparse_qualification.json"]["all39_BYTEEXACT"]
    control, remaining, _ = m.drive_chunk(
        tracker,
        control,
        frames,
        m.RESUME + 2,
        m.RESUME + 5,
        out,
        synchronize=lambda: None,
        guard=lambda: None,
    )
    assert tracker.visited == saved == list(range(m.RESUME, m.RESUME + 5))
    assert len(first_rows) + len(remaining) == 5
    assert tracker.initialize_count == 1 and tracker.restore_count == 6
    assert original_control.visited == [m.RESUME, m.RESUME + 1] and cleared == [True]


def test_any_of_the39_state_fields_differing_stops_before_long_rollout(setup):
    _, outputs, _, out = setup
    tracker, control = FakeTracker(True), FakeTracker(False, drift=1)
    frames = iter(SimpleNamespace(frame_idx=f) for f in range(m.RESUME, m.RESUME + 3))
    with pytest.raises(Exception, match="serial state differs"):
        m.drive_chunk(
            tracker,
            control,
            frames,
            m.RESUME,
            m.RESUME + 2,
            out,
            synchronize=lambda: None,
            guard=lambda: None,
        )
    assert tracker.previous_frame == m.RESUME + 1
    assert "serial_dense_sparse_qualification.json" not in outputs


def test_skipped_frame_is_rejected_before_advance(setup):
    _, _, _, out = setup
    tracker = FakeTracker(True)
    with pytest.raises(Exception, match="skipped or reordered"):
        m.drive_chunk(
            tracker,
            None,
            iter([SimpleNamespace(frame_idx=m.RESUME + 1)]),
            m.RESUME,
            m.RESUME + 1,
            out,
            synchronize=lambda: None,
            guard=lambda: None,
        )
    assert tracker.visited == []


def test_undeclared_metadata_change_is_not_hidden_by_backend_adaptation():
    tracker, control = FakeTracker(True), FakeTracker(False)
    control.restore_count += 1
    with pytest.raises(Exception, match="metadata difference"):
        m.compare_control(tracker, control)


def test_lifecycle_rejects_extra_restore_or_reinitialization():
    tracker = FakeTracker(True)
    tracker.initialize_count = 2
    with pytest.raises(Exception, match="lifecycle"):
        m.lifecycle(tracker, m.RESUME)

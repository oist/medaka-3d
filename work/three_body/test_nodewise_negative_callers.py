"""Synthetic interval contracts only; every scientific entry is fail-spied."""

import copy
from types import SimpleNamespace

import numpy as np
import prepare_nodewise_negative_control as control
import pytest
import run_nodewise_body_row_negative_continuation as runner


@pytest.fixture(autouse=True)
def isolated_inputs(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("scientific input/execution is forbidden in this synthetic suite")

    for module in (runner, control):
        monkeypatch.setattr(module, "WORK", tmp_path)
        monkeypatch.setattr(module, "PARENT", tmp_path / "parent")
    monkeypatch.setattr(runner, "REPO", tmp_path)
    for module, name in (
        (runner, "load_parent_headers"),
        (runner, "load_parent_payload"),
        (runner.original, "verified_inputs"),
        (runner.original, "SourceSequenceReader"),
        (runner.torch.cuda, "synchronize"),
        (control.common.codec, "read_checkpoint"),
        (control.common, "bound_member"),
        (control.proof, "read_json"),
        (np, "load"),
    ):
        monkeypatch.setattr(module, name, forbidden)


def test_aligned_source_window_and_invalid_boundaries():
    assert runner.source_window() == {"chunk_index": 24, "prefix_chunks": 25}
    assert runner.STOP - runner.START == 1660
    assert runner.STOP - runner.FIRST == 250 * 166
    for first, start, stop, chunk in (
        (49800, 89641, 91301, 1660),
        (49800, 89640, 92960, 1660),
        (49800, 89640, 91300, 0),
        (49800, 89640.0, 91300, 1660),
    ):
        with pytest.raises((ValueError, AssertionError)):
            runner.source_window(first, start, stop, chunk)


def test_prefix_includes_exact_chunk24_without_gaps():
    records = [
        {"index": n, "start": 49800 + n * 1660, "stop": 49800 + (n + 1) * 1660} for n in range(40)
    ]
    inputs = SimpleNamespace(chunk_roots=list(range(25)), chunk_records=copy.deepcopy(records[:25]))
    original = copy.deepcopy(inputs.chunk_records)
    parent = {"prefix_chunks": records}
    assert runner.validate_source_prefix(inputs, parent)["chunk_index"] == 24
    assert inputs.chunk_records == original
    inputs.chunk_records[-1]["index"] = 23
    with pytest.raises((ValueError, AssertionError)):
        runner.validate_source_prefix(inputs, parent)
    inputs.chunk_records = copy.deepcopy(records[:24])
    inputs.chunk_roots = list(range(24))
    with pytest.raises((ValueError, AssertionError)):
        runner.validate_source_prefix(inputs, parent)


def test_reader_rejects_internal_gap_even_when_endpoints_match():
    frames = np.arange(89640, 91300)
    runner.validate_reader_frames(SimpleNamespace(frames=frames))
    broken = frames.copy()
    broken[7] += 1
    with pytest.raises((ValueError, AssertionError)):
        runner.validate_reader_frames(SimpleNamespace(frames=broken))
    with pytest.raises((ValueError, AssertionError)):
        runner.validate_reader_frames(SimpleNamespace(frames=np.arange(81340, 83000)))
    assert np.array_equal(frames, np.arange(89640, 91300))


def test_actual_serial_drive_defaults_use_new_native_boundary(monkeypatch):
    seen = []
    tracker = SimpleNamespace(
        initialize_count=1, restore_count=7, previous_frame=89639, processed_frames=39840
    )

    def advance(observation):
        seen.append(observation.frame_idx)
        tracker.previous_frame = observation.frame_idx
        tracker.processed_frames += 1

    tracker.advance = advance
    monkeypatch.setattr(
        runner, "save_frame_diagnostics", lambda out, tracker, frame: {"frame_idx": frame}
    )
    frames = iter(SimpleNamespace(frame_idx=t) for t in range(89640, 91300))
    rows, _ = runner.drive(tracker, frames, None, synchronize=lambda: None, guard=lambda: None)
    assert seen == list(range(89640, 91300))
    assert [row["frame_idx"] for row in rows] == seen
    assert (tracker.previous_frame, tracker.processed_frames) == (91299, 41500)
    assert (tracker.initialize_count, tracker.restore_count) == (1, 7)


def test_control_horizon_is_91300_not_later_backtrace():
    metadata = {
        "schema": "persistent_tracker_full_state_v1",
        "context": {"implementation": "SparseInstanceRescoreContinuousTracker"},
        "previous_frame": 91299,
        "processed_frames": 41500,
        "initialize_count": 1,
        "restore_count": 6,
    }
    original = copy.deepcopy(metadata)
    control.validate_control_metadata(metadata)
    assert metadata == original
    assert control.CHECKPOINT_RELATIVE == "checkpoints/through_091300"
    assert runner.CHECKPOINT_RELATIVE == "checkpoints/through_089640"
    for key, value in (("previous_frame", 116199), ("restore_count", 7), ("initialize_count", 2)):
        wrong = copy.deepcopy(metadata)
        wrong[key] = value
        with pytest.raises((ValueError, AssertionError)):
            control.validate_control_metadata(wrong)


def test_control_rejects_outside_workspace_before_read(tmp_path):
    with pytest.raises((ValueError, AssertionError)):
        control.run(tmp_path.parent / "outside_negative_control")

"""Replay comparison must catch branch, numerical, and native-reader changes."""

from types import SimpleNamespace

import numpy as np
import pytest
import replay_chest_deficit as replay


@pytest.fixture
def sample(monkeypatch):
    monkeypatch.setattr(replay, "FIRST", 100)
    monkeypatch.setattr(replay, "RECORD_FIELDS", ("parent", "pose_xyz", "proposal_kind"))
    records = [dict(parent=i, pose_xyz=np.asarray([float(i)]), proposal_kind="predict")
               for i in range(2)]
    saved = dict(history_offsets=np.asarray([0, 2]), history_parent=np.asarray([0, 1]),
                 history_pose_xyz=np.asarray([[0.0], [1.0]]),
                 history_proposal_kind=np.asarray(["predict", "predict"], dtype="U32"))
    return records, saved


def test_label_capacity_is_not_a_different_label(sample):
    records, saved = sample
    assert replay.compare_records(records, saved, 100)["exact"]
    records[1]["proposal_kind"] = "recovery"
    assert replay.compare_records(records, saved, 100)["differing_fields"] == ["proposal_kind"]


def test_branch_order_and_signed_zero_are_not_accepted(sample):
    records, saved = sample
    assert not replay.compare_records(list(reversed(records)), saved, 100)["exact"]
    records[0]["pose_xyz"][0] = -0.0
    assert replay.compare_records(records, saved, 100)["differing_fields"] == ["pose_xyz"]


def test_missing_beam_is_not_close_enough(sample):
    records, saved = sample
    assert replay.compare_records(records[:1], saved, 100)["differing_fields"] == ["beam_count"]


def test_native_reader_stops_before_extra_frame_and_closes(monkeypatch):
    visited, closed = [], []

    def frames(reader, start):
        try:
            for native in range(start, 110):
                visited.append(native)
                yield SimpleNamespace(frame_idx=native)
        finally:
            closed.append(True)

    monkeypatch.setattr(replay, "iter_from", frames)
    reader = SimpleNamespace(frames=np.arange(100, 110))
    result = [v.frame_idx for v in replay.bounded_frames(reader, first=101, stop=104)]
    assert result == [101, 102, 103]
    assert visited == [101, 102, 103] and closed == [True]

"""Scientific diagnostic preservation and strict serial-driver boundary checks."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import run_instance_rescore_10s as run


def test_nested_diagnostic_disk_roundtrip_keeps_infinite_costs_and_array_bytes(tmp_path):
    values = np.array([0.0, np.inf, -np.inf, np.nan], np.float64)
    payload = dict(
        frame_idx=76360,
        views={"top_cam": dict(costs=values, neutral=False, assignment=np.array([[2, 0, 1]]))},
        invalid_scalar=float("inf"),
        labels=np.array(["tail recovery", "口"], dtype="U20"),
        unavailable=None,
    )
    tree, arrays = run.split_diagnostic(payload)
    copied = {key: run.proof.signature(value) for key, value in arrays.items()}
    values[:] = -5  # The exported diagnostic cannot alias future reused fitter buffers.
    output = run.proof.Output(tmp_path / "output")
    row = output.write("diagnostic.npz", arrays)
    output.write("diagnostic.json", tree)
    with np.load(row["path"], allow_pickle=False) as archive:
        loaded = {key: archive[key] for key in archive.files}
    saved_tree = json.loads((output.root / "diagnostic.json").read_text(encoding="utf8"))
    restored = run.restore_diagnostic(saved_tree, loaded)
    assert {k: run.proof.signature(v) for k, v in loaded.items()} == copied
    assert np.isposinf(restored["invalid_scalar"])
    assert np.isposinf(restored["views"]["top_cam"]["costs"][1])
    assert np.isneginf(restored["views"]["top_cam"]["costs"][2])
    assert np.isnan(restored["views"]["top_cam"]["costs"][3])
    assert restored["labels"][1] == "口"
    assert restored["unavailable"] is None


def test_diagnostic_rejects_changed_missing_extra_and_object_arrays():
    tree, arrays = run.split_diagnostic({"cost": np.array([1.0])})
    with pytest.raises(ValueError, match="missing"):
        run.restore_diagnostic(tree, {})
    with pytest.raises(ValueError, match="extra"):
        run.restore_diagnostic(tree, {**arrays, "extra": np.array([0])})
    with pytest.raises(ValueError, match="differs"):
        run.restore_diagnostic(tree, {"a00000": np.array([2.0])})
    with pytest.raises(ValueError, match="object"):
        run.split_diagnostic({"cost": np.array([None], dtype=object)})


class SerialTracker:
    def __init__(self):
        self.initialize_count = 1
        self.restore_count = 5
        self.previous_frame = run.START - 1
        self.processed_frames = run.START - run.FIRST
        self.advanced = []

    def advance(self, observation):
        self.advanced.append(observation.frame_idx)
        self.previous_frame = observation.frame_idx
        self.processed_frames += 1


def test_driver_advances_once_and_saves_current_diagnostic_without_reset(monkeypatch):
    monkeypatch.setattr(run, "STOP", run.START + 4)
    tracker, saved, guards, syncs = SerialTracker(), [], [], []

    def capture(out, current, frame):
        assert current.previous_frame == frame
        saved.append(frame)
        return {"frame_idx": frame}

    monkeypatch.setattr(run, "save_diagnostic", capture)
    rows, timing = run.drive(
        tracker,
        iter(SimpleNamespace(frame_idx=frame) for frame in range(run.START, run.STOP)),
        None,
        synchronize=lambda: syncs.append(True),
        guard=lambda: guards.append(True),
    )
    assert saved == tracker.advanced == list(range(run.START, run.STOP))
    assert [row["frame_idx"] for row in rows] == saved
    assert len(syncs) == 4 and len(guards) == 8
    assert tracker.initialize_count == 1 and tracker.restore_count == 5
    assert all(v >= 0 for v in timing.values())


def test_driver_rejects_skipped_native_frame_before_advance(monkeypatch):
    monkeypatch.setattr(run, "STOP", run.START + 1)
    tracker = SerialTracker()
    with pytest.raises(ValueError, match="skipped"):
        run.drive(
            tracker,
            iter([SimpleNamespace(frame_idx=run.START + 1)]),
            None,
            synchronize=lambda: None,
            guard=lambda: None,
        )
    assert tracker.advanced == []

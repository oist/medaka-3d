"""C24/C27 labels must follow input instances, including reordered returns."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return import_module("c24_uncapped_tracklets")


def detections(adapter):
    # Six distinct source rows can have identical coordinates. Object identity,
    # rather than coordinate equality, must determine where a label is written.
    return adapter.c01.Detections(
        experiment="synthetic",
        view="top_cam",
        frame_idx=np.repeat([10, 20], 6),
        points=np.zeros((12, 2, 2)),
        score=np.ones(12),
        n_frames=21,
        nodes=("mouth", "pectoral"),
    )


def fake_tracker(monkeypatch, transform):
    from sleap_nn.tracking.tracker import Tracker

    class FakeTracker:
        candidate = SimpleNamespace(max_tracks=None)
        tracking_target_instance_count = None
        tracking_pre_cull_to_target = False

        def track(self, batch, *, frame_idx):
            for i, instance in enumerate(batch):
                instance.track = SimpleNamespace(name=f"animal_{i}")
            return transform(batch) if frame_idx == 20 else batch

    monkeypatch.setattr(Tracker, "from_config", lambda **kwargs: FakeTracker())


def run(adapter, data, frames=range(21)):
    return adapter.run_tracker(
        data,
        ["mouth", "pectoral"],
        window_size=5,
        scoring="oks",
        min_match_points=0,
        frames=frames,
    )


@pytest.mark.parametrize("in_place", [False, True])
def test_frame20_permutation_keeps_labels_on_original_rows(adapter, monkeypatch, in_place):
    def transform(batch):
        permuted = [batch[i] for i in [0, 1, 2, 4, 3, 5]]
        if in_place:
            batch[:] = permuted
            return batch
        return permuted

    fake_tracker(monkeypatch, transform)
    data = detections(adapter)
    original = data.points.copy()
    labels, effective = run(adapter, data)
    np.testing.assert_array_equal(labels, np.tile(np.arange(6), 2))
    np.testing.assert_array_equal(data.points, original)
    assert effective["row_assignment"] == "input_instance_identity_v1"


def test_untracked_return_does_not_shift_other_rows(adapter, monkeypatch):
    def transform(batch):
        batch[3].track = None
        return [batch[i] for i in [0, 1, 2, 4, 3, 5]]

    fake_tracker(monkeypatch, transform)
    labels, _ = run(adapter, detections(adapter))
    np.testing.assert_array_equal(labels[6:], [0, 1, 2, -1, 4, 5])


def test_nonzero_start_and_empty_frames_preserve_global_row_offsets(adapter, monkeypatch):
    fake_tracker(monkeypatch, lambda batch: list(reversed(batch)))
    labels, _ = run(adapter, detections(adapter), range(11, 21))
    np.testing.assert_array_equal(labels[:6], np.full(6, -1))
    # Dense names are still allocated in return order, preserving that contract.
    np.testing.assert_array_equal(labels[6:], np.arange(5, -1, -1))


@pytest.mark.parametrize("invalid", ["duplicate", "missing", "foreign"])
def test_invalid_instance_correspondence_fails(adapter, monkeypatch, invalid):
    def transform(batch):
        if invalid == "duplicate":
            return [*batch[:-1], batch[0]]
        if invalid == "missing":
            return batch[:-1]
        return [*batch[:-1], SimpleNamespace(track=batch[-1].track)]

    fake_tracker(monkeypatch, transform)
    with pytest.raises(ValueError, match="exactly once at frame 20"):
        run(adapter, detections(adapter))


def test_actual_pinned_tracker_returns_input_objects(adapter, monkeypatch):
    from sleap_nn.tracking.tracker import Tracker

    data = detections(adapter)
    # A synthetic, well-separated scene for the real library, not a fish result.
    data.points[:] = np.tile(
        np.array([[[i * 100.0, 0], [i * 100.0 + 10, 0]] for i in range(6)]),
        (2, 1, 1),
    )
    normal, _ = run(adapter, data)
    original = Tracker.track

    def reordered(self, batch, **kwargs):
        result = original(self, batch, **kwargs)
        return result[::-1] if kwargs["frame_idx"] == 20 else result

    monkeypatch.setattr(Tracker, "track", reordered)
    permuted, _ = run(adapter, data)
    np.testing.assert_array_equal(permuted, normal)

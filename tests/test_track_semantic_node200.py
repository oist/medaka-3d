"""Synthetic native binding and one complete streamed view, without model/GPU."""

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
spec = importlib.util.spec_from_file_location(
    "track_semantic_node200", WORK / "track_semantic_node200.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def query():
    return dict(
        frame_idx=np.asarray(m.FIRST, np.int64),
        xy=np.arange(18, dtype=np.float32).reshape(9, 2) + 100,
        body_slot=np.repeat(np.arange(3, dtype=np.int32), 3),
        node_index=np.tile(np.arange(3, dtype=np.int32), 3),
        source_row=np.repeat(np.arange(3, dtype=np.int32), 3),
        query_index=np.arange(9, dtype=np.int32),
        duplicate_pixel=np.zeros(9, bool),
    )


@pytest.mark.parametrize("kind", ["order", "source", "bounds", "dtype", "duplicates"])
def test_semantic_query_corruption_rejected(kind):
    q = query()
    m.validate_queries(q, dict(view="top_cam"), "top_cam")
    if kind == "order":
        q["node_index"][0] = 2
    elif kind == "source":
        q["source_row"][3] = q["source_row"][0]
    elif kind == "bounds":
        q["xy"][1, 0] = 1200
    elif kind == "dtype":
        q["xy"] = q["xy"].astype(np.float64)
    else:
        q["duplicate_pixel"][2] = True
    with pytest.raises(ValueError):
        m.validate_queries(q, dict(view="top_cam"), "top_cam")


def native_rows():
    return [
        dict(
            view=view,
            frame_idx=native,
            native_offset=native - m.FIRST,
            shape_hw=[1200, 1200],
            bgr_sha256="0" * 64,
        )
        for view in m.VIEWS
        for native in range(m.FIRST, m.STOP)
    ]


def test_native_extent_rejects_duplicate_omission_and_offset():
    rows = native_rows()
    m.validate_native_rows(rows)
    with pytest.raises(ValueError):
        m.validate_native_rows(rows[:-1])
    rows[-1] = rows[0].copy()
    with pytest.raises(ValueError):
        m.validate_native_rows(rows)
    rows = native_rows()
    rows[16]["native_offset"] = 15
    with pytest.raises(ValueError):
        m.validate_native_rows(rows)


def test_checked_bytes_rejects_same_size_change(tmp_path):
    p = tmp_path / "raw.bin"
    p.write_bytes(b"original")
    members = {str(p): dict(size_bytes=8, sha256=hashlib.sha256(b"original").hexdigest())}
    assert m.checked_bytes(p, members) == b"original"
    p.write_bytes(b"modified")
    with pytest.raises(ValueError, match="consumption"):
        m.checked_bytes(p, members)


class FakeBackend:
    instances: ClassVar[list] = []

    def __init__(self, device, closure):
        assert device == "cpu"
        self.initializations = self.forwards = self.closed = 0
        self.load_seconds = 0.0
        self.instances.append(self)

    def initialize(self, images, xy):
        self.initializations += 1
        self.xy = xy.copy()
        return None, None

    def predict(self, images):
        assert len(images) == 16 and all(x.shape == (1200, 1200, 3) for x in images)
        hi = 16 + self.forwards * 8
        xy = np.broadcast_to(self.xy, (hi, 9, 2)).copy()
        scores = np.full((hi, 9), 0.8, np.float32)
        self.forwards += 1
        return dict(
            xy=xy,
            model_xy=xy.copy(),
            reported=scores > np.float32(0.6),
            sigmoid_visibility=scores.copy(),
            sigmoid_confidence=np.ones_like(scores),
            product=scores,
        ), dict(forward_seconds=0.0)

    def close(self):
        self.closed += 1
        return dict(max_allocated_bytes=None, max_reserved_bytes=None)


def fixture_context(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PREPARATION", tmp_path)
    image = np.zeros((1200, 1200, 3), np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    p = tmp_path / "image.png"
    p.write_bytes(encoded.tobytes())
    members = {str(p): dict(size_bytes=p.stat().st_size, sha256=m.base.sha(p))}
    rows = [
        dict(
            view="top_cam",
            frame_idx=native,
            png_path=p.name,
            bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        )
        for native in range(m.FIRST, m.STOP)
    ]
    context = dict(natives=rows, queries={"top_cam": query()}, members=members)
    monkeypatch.setattr(m, "load_masks", lambda *_: np.zeros((0, 1200, 1200), bool))
    monkeypatch.setattr(m.base, "TorchOnline", FakeBackend)
    return context


def test_complete_view_stream_persistence_and_close(tmp_path, monkeypatch):
    context = fixture_context(tmp_path, monkeypatch)
    output = m.base.Output(tmp_path / "output")
    result = m.run_view(context, "top_cam", {}, output, "cpu", [])
    backend = FakeBackend.instances[-1]
    assert backend.initializations == 1 and backend.forwards == 24 and backend.closed == 1
    assert result["tracking199_query_denominator"] == 1791
    assert result["ct_first_reported"] == 1791
    assert result["first_release_native"] == 333534
    assert result["final_release_native"] == 333718
    assert len([name for name in output.records if "/memberships/" in name]) == 200
    output.final_guard()
    saved = m.arrays_from_bytes((output.root / "top_cam/emissions.npz").read_bytes())
    assert saved["frame_idx"].tolist() == list(range(m.FIRST, m.STOP))
    assert np.array_equal(saved["first_release_native"], m.FIRST + saved["first_horizon"])


def test_view_failure_closes_backend(tmp_path, monkeypatch):
    context = fixture_context(tmp_path, monkeypatch)

    def fail(_self, _images):
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(FakeBackend, "predict", fail)
    output = m.base.Output(tmp_path / "failed")
    memory = []
    with pytest.raises(RuntimeError, match="synthetic forward"):
        m.run_view(context, "top_cam", {}, output, "cpu", memory)
    assert FakeBackend.instances[-1].closed == 1
    assert memory[-1]["event"] == "backend_closed"


@pytest.mark.parametrize("available", [False, True])
def test_allN_empty_availability_kept(tmp_path, monkeypatch, available):
    monkeypatch.setattr(m, "PREPARATION", tmp_path)
    p = tmp_path / "obs.npz"
    np.savez_compressed(
        p,
        frame_idx=np.asarray(m.FIRST, np.int64),
        mask_available=np.asarray(available, bool),
        masks=np.zeros((0, 1200, 1200), bool),
    )
    context = dict(members={str(p): dict(size_bytes=p.stat().st_size, sha256=m.base.sha(p))})
    row = dict(observation_path=p.name, frame_idx=m.FIRST, mask_count=0, mask_available=available)
    masks = m.load_masks(context, row)
    assert (masks is not None) == available

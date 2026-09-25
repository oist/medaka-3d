"""Synthetic fixed200 schedule, persistence and LK streaming integration checks."""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
spec = importlib.util.spec_from_file_location(
    "semantic_node_stream", WORK / "semantic_node_stream.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Backend:
    def __init__(self):
        self.inits = self.calls = 0

    def initialize(self, images, xy):
        assert len(images) == 1
        self.inits += 1
        self.xy = xy.copy()
        return None, None

    def predict(self, images):
        i = self.calls
        assert len(images) == 16
        assert [int(x[0, 0, 0]) for x in images] == list(range(i * 8, i * 8 + 16))
        hi, n = i * 8 + 16, len(self.xy)
        xy = np.broadcast_to(self.xy, (hi, n, 2)).copy()
        xy[:, :, 0] += np.arange(hi, dtype=np.float32)[:, None] + i + 1
        p = np.full((hi, n), np.float32(0.8))
        if n:
            p[::3, 0] = np.float32(0.5)
        self.calls += 1
        return dict(
            xy=xy, model_xy=xy * np.float32(0.5), reported=p > np.float32(0.6),
            sigmoid_visibility=p.copy(), sigmoid_confidence=np.ones((hi, n), np.float32),
            product=p,
        ), dict(forward_seconds=0.0)


def records(first=333519):
    for i in range(200):
        yield first + i, np.full((12, 12, 3), i, np.uint8)


def test_all24_saved_deltas_first_final_and_releases(tmp_path):
    previous, calls, first_rows = None, [], []

    def save(header, payload):
        nonlocal previous
        path = tmp_path / f'{header["call_metadata"]["call_index"]:02}.npz'
        np.savez_compressed(path, **payload)
        with np.load(path, allow_pickle=False) as z:
            restored = {key: z[key] for key in z.files}
        previous = m.delta.decode_delta(previous, header, restored)
        calls.append(header)

    backend = Backend()
    queries = np.array([[3, 4], [5, 6]], np.float32)
    original = queries.tobytes()
    arrays, meta = m.run_online200(
        records(), queries, backend, save, first=333519,
        on_first=lambda native, image, row, horizon: first_rows.append(
            (native, int(image[0, 0, 0]), row, horizon)
        ),
    )
    assert queries.tobytes() == original
    assert backend.inits == 1 and backend.calls == len(calls) == len(meta) == 24
    assert len(first_rows) == 200
    expected = np.r_[np.full(16, 15), np.repeat(np.arange(23, 200, 8), 8)]
    assert np.array_equal(arrays["first_horizon"], expected)
    for i, (native, image_index, row, horizon) in enumerate(first_rows):
        ci = 0 if i < 16 else (i - 8) // 8
        assert native == 333519 + i and image_index == i and horizon == 333519 + expected[i]
        assert row["xy"][0, 0] == queries[0, 0] + i + ci + 1
        assert row["xy"].tobytes() == arrays["first_xy"][i].tobytes()
        assert row["model_xy"].tobytes() == arrays["first_model_xy"][i].tobytes()
    assert arrays["first_xy"][0, 0, 0] != queries[0, 0]
    assert arrays["final_xy"].tobytes() == previous["xy"].tobytes()
    assert arrays["final_product"].tobytes() == previous["product"].tobytes()
    assert calls[-1]["call_metadata"]["input_native"] == [333703, 333719]
    assert calls[-1]["fields"]["xy"]["outside_changed_rows"] == 184


def test_zero_query_has_horizons_without_backend():
    result, calls = m.run_online200(
        records(), np.empty((0, 2), np.float32), None, lambda *_: None, first=333519,
    )
    assert len(calls) == 24
    assert all(row["status"] == "no_queries" and not row["forward_called"] for row in calls)
    assert result["first_xy"].shape == (200, 0, 2)
    assert result["first_release_native"][-1] == 333718


def test_failure_closes_source_and_does_not_overread():
    consumed, closed = [], []

    def source():
        try:
            for i in range(201):
                consumed.append(i)
                yield 333519 + i, np.full((12, 12, 3), i, np.uint8)
        finally:
            closed.append(True)

    def fail(*_):
        raise RuntimeError("persistence failed")

    with pytest.raises(RuntimeError, match="persistence"):
        m.run_online200(source(), np.array([[2, 3]], np.float32), Backend(), fail, first=333519)
    assert consumed == list(range(16)) and closed == [True]
    consumed.clear()
    closed.clear()
    m.run_online200(
        source(), np.array([[2, 3]], np.float32), Backend(), lambda *_: None, first=333519,
    )
    assert consumed == list(range(200)) and closed == [True]


@pytest.mark.parametrize("kind", ["mutated", "short", "wrongflag", "float64", "count"])
def test_backend_contract_failures(kind):
    class Bad(Backend):
        def initialize(self, images, xy):
            result = super().initialize(images, xy)
            if kind == "mutated":
                xy[0, 0] += 1
            return result

        def predict(self, images):
            result, timing = super().predict(images)
            if kind == "short":
                result = {key: value[:-1] for key, value in result.items()}
            if kind == "wrongflag":
                result["reported"] = ~result["reported"]
            if kind == "float64":
                result["xy"] = result["xy"].astype(np.float64)
            return result, timing

    query = np.ones((10 if kind == "count" else 1, 2), np.float32)
    with pytest.raises(ValueError):
        m.run_online200(records(), query, Bad(), lambda *_: None, first=333519)


def test_lk_stream_matches_existing_chain_all_fields():
    rng = np.random.default_rng(719)
    initial = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    frames = [initial.copy() for _ in range(200)]
    frames[7:] = [np.zeros_like(initial) for _ in range(193)]
    xy = np.array([[22, 20], [41, 30]], np.float32)
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        expected = m.core.lk_chain(frames, xy)
        stream = m.StreamingLK(xy, first=333519)
        for i, image in enumerate(frames):
            stream.push(333519 + i, image)
        actual = stream.finish()
    finally:
        cv2.setNumThreads(old_threads)
    assert set(actual) == set(expected)
    for key in expected:
        assert m.delta.signature(actual[key]) == m.delta.signature(expected[key]), key
    assert not actual["valid"][-1].any()


def test_lk_stream_rejects_native_discontinuity_and_incomplete():
    stream = m.StreamingLK(np.empty((0, 2), np.float32), first=100)
    stream.push(100, np.zeros((12, 12, 3), np.uint8))
    with pytest.raises(ValueError, match="index"):
        stream.push(102, np.zeros((12, 12, 3), np.uint8))
    with pytest.raises(ValueError, match="full200"):
        stream.finish()

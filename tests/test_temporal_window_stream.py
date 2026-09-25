"""Synthetic iterator order, bounded retention, and resource lifetime tests."""

import gc
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work/three_body"))
import temporal_window_stream as s  # noqa: E402


class Payload:
    def __init__(self, n):
        self.pixels = np.asarray([n, n + 1], dtype=np.int64)


class Source:
    def __init__(self, *, first=333519, count=201, bad=None):
        self.first, self.count, self.bad = first, count, bad
        self.reads, self.closed, self.peak = 0, 0, 0
        self.references = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.reads >= self.count:
            raise StopIteration
        n = self.first + self.reads
        if self.bad == "exception" and self.reads == 17:
            raise RuntimeError("source failed")
        if self.bad == "order" and self.reads == 17:
            n -= 1
        if self.bad == "relative":
            n -= self.first
        self.reads += 1
        value = Payload(n)
        self.references.append(weakref.ref(value))
        self.peak = max(self.peak, sum(r() is not None for r in self.references))
        return n, value

    def close(self):
        self.closed += 1


@pytest.mark.parametrize("count,calls", [(16, 1), (24, 2), (64, 7), (200, 24)])
def test_fixed_windows_native_overlap_exactly_once_without_extra_pull(count, calls):
    source = Source()
    overlap = []
    with s.window_stream(source, first=333519, count=count) as stream:
        for index in range(calls):
            window = next(stream)
            lo, hi = 8 * index, 16 + 8 * index
            assert window.call_index == index and window.input_relative == (lo, hi)
            assert window.input_native == (333519 + lo, 333519 + hi)
            assert window.maximum_seen_native == 333519 + hi - 1
            assert [r[0] for r in window.records] == list(range(333519 + lo, 333519 + hi))
            if index:
                assert [id(r[1]) for r in window.records[:8]] == overlap
            for n, value in window.records:
                assert np.array_equal(value.pixels, [n, n + 1])
            # Only keep integer identities. Release borrowed window before next.
            overlap = [id(r[1]) for r in window.records[8:]]
            del window, n, value
            gc.collect()
            assert source.reads == hi and source.closed == 0
        with pytest.raises(StopIteration):
            next(stream)
    assert source.reads == count and source.closed == 1
    assert source.peak <= 16


def test_early_break_closes_without_next_window():
    source = Source()
    with s.window_stream(source, first=333519) as stream:
        first = next(stream)
        assert len(first.records) == 16
    assert source.reads == 16 and source.closed == 1
    with pytest.raises(StopIteration):
        next(stream)


@pytest.mark.parametrize("bad", ["order", "relative", "exception"])
def test_invalid_source_is_closed(bad):
    source = Source(bad=bad)
    with pytest.raises((ValueError, RuntimeError)), s.window_stream(source, first=333519) as stream:
        for _window in stream:
            pass
    assert source.closed == 1


def test_premature_source_end_is_not_silently_shortened():
    source = Source(count=63)
    with (
        pytest.raises(ValueError, match="before fixed stop"),
        s.window_stream(source, first=333519, count=64) as stream,
    ):
        for _window in stream:
            pass
    assert source.reads == 63 and source.closed == 1


@pytest.mark.parametrize("count", [True, 0, 15, 17, 63, 199, 201, 666])
def test_unqualified_short_tail_or_large_window_rejected_without_reads(count):
    source = Source()
    with pytest.raises(ValueError), s.window_stream(source, first=333519, count=count):
        pytest.fail("entered")
    assert source.reads == 0 and source.closed == 0


@pytest.mark.parametrize("first", [True, -1, 1.5, 2**63 - 1])
def test_invalid_origin_before_iterator_ownership(first):
    source = Source()
    with pytest.raises(ValueError), s.window_stream(source, first=first):
        pytest.fail("entered")
    assert source.reads == 0


def test_consumer_exception_closes_iterator():
    source = Source()
    with (
        pytest.raises(RuntimeError, match="consumer"),
        s.window_stream(source, first=333519) as stream,
    ):
        next(stream)
        raise RuntimeError("consumer failed")
    assert source.closed == 1 and source.reads == 16


def test_payload_none_empty_masks_and_all_rows_remain_uninterpreted():
    records = [
        (
            333519 + i,
            dict(masks=None if i % 2 else np.empty((0, 2, 2), bool), rows=list(range(i % 5))),
        )
        for i in range(24)
    ]
    with s.window_stream(iter(records), first=333519, count=24) as stream:
        for window in stream:
            for n, payload in window.records:
                assert payload is records[n - 333519][1]
                assert payload["rows"] == list(range((n - 333519) % 5))
    assert records[0][1]["masks"].shape == (0, 2, 2) and records[1][1]["masks"] is None

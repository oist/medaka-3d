"""Small benchmark schedule, byte-parity and reader-boundary regressions."""

from types import SimpleNamespace

import benchmark_parallel_flow_views as bench
import numpy as np
import pytest


@pytest.mark.parametrize("field", bench.FIELDS)
def test_each_of_all15_fields_must_match_bytes(field):
    golden = {k: np.asarray([1.0, np.nan]) for k in bench.FIELDS}
    actual = {k: v.copy() for k, v in golden.items()}
    bench.equal_flow(actual, golden)
    actual[field][0] = 1.0000000000000002
    with pytest.raises(ValueError, match=field):
        bench.equal_flow(actual, golden)


def test_same_nan_value_different_payload_is_not_byte_parity():
    golden = {k: np.asarray([np.nan]) for k in bench.FIELDS}
    actual = {k: v.copy() for k, v in golden.items()}
    actual[bench.FIELDS[0]].view(np.uint64)[0] ^= np.uint64(1)
    assert np.isnan(actual[bench.FIELDS[0]]).all()
    with pytest.raises(ValueError):
        bench.equal_flow(actual, golden)


def trials():
    return [
        dict(cycle=c, frame_idx=n, method=m, seconds=(2.0 if m == "serial" else 1.0) * (i + 1))
        for c in range(3)
        for i, n in enumerate((10, 11))
        for m in bench.call_order(c, i)
    ]


def test_pairing_is_per_frame_cycle_and_order_alternates():
    rows = trials()
    result = bench.summarize(rows, (10, 11))
    assert result["paired_ratios"] == [2.0] * 6
    assert result["methods"]["serial"]["median_seconds"] == 3
    assert result["methods"]["parallel"]["median_seconds"] == 1.5
    assert rows[0]["method"] != rows[2]["method"] != rows[6]["method"]
    with pytest.raises(ValueError, match="schedule"):
        bench.summarize(rows[:-1], (10, 11))
    with pytest.raises(ValueError, match="schedule"):
        bench.summarize(rows[::-1], (10, 11))
    rows[0]["seconds"] = np.nan
    with pytest.raises(ValueError, match="timing"):
        bench.summarize(rows, (10, 11))


@pytest.mark.parametrize("bad_native", [False, True])
def test_reader_exact_stop_no_extra_decode_and_close(monkeypatch, bad_native):
    delivered, closed = [], []

    def iterator(_reader, first):
        try:
            for n in range(first, 15):
                delivered.append(n)
                yield SimpleNamespace(frame_idx=n + (1 if bad_native and n == 11 else 0))
        finally:
            closed.append(True)

    monkeypatch.setattr(bench.packets, "iter_from", iterator)
    monkeypatch.setattr(bench.packets, "observation_evidence", lambda f, r: [f.frame_idx])
    monkeypatch.setattr(bench, "own_flow_frame", lambda f: f)
    if bad_native:
        with pytest.raises(ValueError, match="native order"):
            bench.read34(None, start=10, stop=13)
        assert delivered == [10, 11]
    else:
        frames, rows = bench.read34(None, start=10, stop=13)
        assert list(frames) == rows == delivered == [10, 11, 12]
    assert closed == [True]


def test_flow_frame_owns_readonly_copy_and_none_masks():
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    mask = np.zeros((1, 2, 2), bool)
    original = SimpleNamespace(frame_idx=9, images_bgr={"a": image}, masks={"a": mask, "b": None})
    frame = bench.own_flow_frame(original)
    signature = bench.frame_signature(frame)
    image[:] = 0
    mask[:] = True
    assert bench.frame_signature(frame) == signature
    assert frame.masks["b"] is None
    assert not frame.images_bgr["a"].flags.writeable and not frame.masks["a"].flags.writeable

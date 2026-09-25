"""Fixed200 anonymous-mask query stream using the qualified16/8 protocol.

The online loop is the frozen semantic stream loop with only its query bound
changed to157. StreamingLK is reused unchanged. No anatomical labels or3D rays.
"""

from __future__ import annotations

import numpy as np
import temporal_point_core as core
import temporal_prefix_deltas as delta
from semantic_node_stream import StreamingLK
from temporal_window_stream import window_stream

COUNT = 200
require = core.require
__all__ = ["StreamingLK", "run_online200"]


def run_online200(
    records, queries, backend, save_call, *, first, before_call=lambda _: None,
    on_first=lambda *_: None,
):
    """Consume exactly200 indexed images and retain all six logical prefix fields.

    save_call(header,payload) must persist the lossless delta. on_first receives
    native, BGR, and owned six-field query rows only at their first release.
    The callback also receives the native release horizon. Its caller must not
    treat that delayed observation as available before that horizon.
    Backend lifecycle (constructor/close, including exceptions) belongs to caller.
    """
    require(
        isinstance(queries, np.ndarray) and queries.dtype == np.float32
        and queries.ndim == 2 and queries.shape[1] == 2
        and len(queries) <= 157 and np.isfinite(queries).all(),
        "at most157 finite float32 auxiliary queries per view required",
    )
    query_signature = delta.signature(queries)
    n = len(queries)
    ledger = core.EmissionLedger(COUNT, n)
    first_raw = {
        key: np.full(delta.layout(key, COUNT, n)[0], np.nan, np.float32)
        for key in ("model_xy", "sigmoid_visibility", "sigmoid_confidence", "product")
    }
    last_changed = np.full((COUNT, n), -1, np.int64)
    seen = np.zeros(COUNT, bool)
    previous, calls = None, []
    with window_stream(records, first=first, count=COUNT) as windows:
        for window in windows:
            lo, hi = window.input_relative
            images = [record[1] for record in window.records]
            if window.call_index == 0 and n:
                before_call(-1)
                require(
                    backend.initialize(images[:1], queries) == (None, None),
                    "initialization returned predictions",
                )
                require(delta.signature(queries) == query_signature, "query mutated at init")
            before_call(window.call_index)
            if n:
                current, timing = backend.predict(images)
            else:
                current = {
                    key: np.empty(*delta.layout(key, hi, 0)) for key in delta.FIELDS
                }
                timing = dict(
                    forward_seconds=0.0, input_transfer_seconds=0.0,
                    capture_transfer_seconds=0.0,
                )
            delta.validate_arrays(current, hi, n)
            require(
                np.array_equal(current["reported"], current["product"] > np.float32(0.6)),
                "official product flag differs",
            )
            require(delta.signature(queries) == query_signature, "backend mutated queries")
            meta = dict(
                call_index=window.call_index, input_relative=list(window.input_relative),
                input_native=list(window.input_native),
                maximum_seen_native=window.maximum_seen_native, output_prefix=hi,
                query_count=n, query_signature=query_signature,
                native_origin=first, forward_called=bool(n),
                status="ok" if n else "no_queries", timing=timing,
            )
            header, payload = delta.encode_delta(previous, current, meta)
            save_call(header, payload)
            if previous is not None:
                p = len(previous["xy"])
                changed = np.any(
                    current["xy"][:p].view(np.uint32) != previous["xy"].view(np.uint32),
                    axis=-1,
                ) | (current["reported"][:p] != previous["reported"])
                last_changed[:p][changed] = window.call_index
            fresh = np.flatnonzero(~seen[:hi])
            require(
                np.array_equal(fresh, np.arange(0 if lo == 0 else hi - 8, hi)),
                "first emission fresh range differs",
            )
            last_changed[fresh] = window.call_index
            for key in first_raw:
                first_raw[key][fresh] = current[key][fresh]
            ledger.add(current["xy"], current["reported"], input_start=lo, input_stop=hi)
            seen[fresh] = True
            for t in fresh:
                on_first(
                    first + int(t), images[int(t) - lo],
                    {key: current[key][t].copy() for key in delta.FIELDS},
                    window.maximum_seen_native,
                )
            previous = delta.decode_delta(previous, header, payload)
            calls.append(meta)
            del current, header, payload, images, window
    require(len(calls) == 24 and seen.all(), "24 calls and full200 required")
    arrays = ledger.arrays()
    expected_horizon = np.r_[np.full(16, 15), np.repeat(np.arange(23, COUNT, 8), 8)]
    require(np.array_equal(arrays["first_horizon"], expected_horizon), "release horizon differs")
    for key in first_raw:
        arrays["first_" + key] = first_raw[key]
        arrays["final_" + key] = previous[key].copy()
    arrays["last_value_changed_call"] = last_changed
    arrays["first_lookahead_frames"] = arrays["first_horizon"] - np.arange(COUNT, dtype=np.int64)
    arrays["first_release_native"] = first + arrays["first_horizon"]
    arrays["frame_idx"] = np.arange(first, first + COUNT, dtype=np.int64)
    return arrays, calls

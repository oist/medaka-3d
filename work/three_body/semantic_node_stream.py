"""Fixed200 semantic-query orchestration; no model implementation or file I/O.

The caller owns source/asset seals, query labels, image decoding and persistence.
The frozen online backend receives only one initial query and24 full16 windows.
First emitted coordinates remain unchanged by later prefix revisions.
"""

from __future__ import annotations

import numpy as np
import temporal_point_core as core
import temporal_prefix_deltas as delta
from temporal_window_stream import window_stream

COUNT = 200
require = core.require


class StreamingLK:
    """Same absorbing LK chain fields as core.lk_chain, with two gray frames."""

    def __init__(self, queries, *, first, edge=core.lk_edge):
        require(type(first) is int and first >= 0, "nonnegative native origin")
        require(
            isinstance(queries, np.ndarray)
            and queries.dtype == np.float32
            and queries.ndim == 2
            and queries.shape[1] == 2
            and np.isfinite(queries).all(),
            "finite float32 query N,2 required",
        )
        self.queries = queries.copy()
        self.first, self.edge, self.consumed = first, edge, 0
        self.previous, self.arrays, self.empty = None, None, None

    def push(self, native, frame):
        require(
            type(native) is int and native == self.first + self.consumed
            and self.consumed < COUNT,
            "LK native index/count differs",
        )
        gray = core.grayscale(frame)
        t, n = self.consumed, len(self.queries)
        if t == 0:
            self.empty = core.lk_edge(gray, gray, np.empty((0, 2), np.float32))
            self.arrays = {
                key: np.full((COUNT, n, *value.shape[1:]), np.nan, value.dtype)
                if np.issubdtype(value.dtype, np.floating)
                else np.zeros((COUNT, n, *value.shape[1:]), value.dtype)
                for key, value in self.empty.items()
            }
            self.arrays["xy"] = np.full((COUNT, n, 2), np.nan, np.float32)
            self.arrays["xy"][0] = self.queries
            self.arrays["valid"][0] = True
            self.arrays["attempted"] = np.zeros((COUNT, n), bool)
        else:
            require(gray.shape == self.previous.shape, "LK native shape differs")
            active = np.flatnonzero(self.arrays["valid"][t - 1])
            self.arrays["failure_bits"][t] = core.Failure.ALREADY_LOST
            if len(active):
                outcome = self.edge(
                    self.previous, gray, self.arrays["xy"][t - 1, active].copy()
                )
                require(set(outcome) == set(self.empty), "LK edge fields differ")
                for key, value in outcome.items():
                    require(
                        value.shape == (len(active), *self.empty[key].shape[1:])
                        and value.dtype == self.empty[key].dtype,
                        "LK edge schema differs: " + key,
                    )
                    self.arrays[key][t, active] = value
                self.arrays["attempted"][t, active] = True
                accepted = active[outcome["valid"]]
                self.arrays["xy"][t, accepted] = outcome["proposed_xy"][outcome["valid"]]
        self.previous = gray
        self.consumed += 1

    def finish(self):
        require(self.consumed == COUNT, "LK full200 required")
        self.previous = None
        return self.arrays


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
        and len(queries) <= 9 and np.isfinite(queries).all(),
        "at most9 finite float32 semantic queries per view required",
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

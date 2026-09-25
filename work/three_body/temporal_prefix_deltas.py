"""Lossless CPU codec for the six saved CoTracker online public-prefix fields.

No model, tensor operations, thresholding, coordinate conversion, or file I/O.
Returns a save-ready JSON header and 18 NumPy arrays. Caller must bind input,
source and output hashes and save a separate success seal. Signed zero and NaN
payload bits are compared as bytes; floating-point values are never recomputed.
The active16 rows are always retained. Every changed earlier row is also kept,
so worst-case storage remains quadratic; no assumed outside-window invariance.
This bounded component supports at most200 prefix frames and640 fixed queries.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

SCHEMA = "temporal_full_prefix_delta_v1"
FIELDS = ("xy", "reported", "model_xy", "sigmoid_visibility", "sigmoid_confidence", "product")
MAX_PREFIX, MAX_QUERIES, WINDOW, STRIDE = 200, 640, 16, 8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(array):
    return dict(
        dtype=array.dtype.str,
        shape=list(array.shape),
        sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    )


def signatures(arrays):
    return {key: signature(array) for key, array in arrays.items()}


def layout(key, prefix, queries):
    return (
        ((prefix, queries, 2), np.dtype("float32"))
        if key in ("xy", "model_xy")
        else ((prefix, queries), np.dtype("bool") if key == "reported" else np.dtype("float32"))
    )


def validate_arrays(arrays, prefix, queries):
    require(
        isinstance(arrays, dict) and set(arrays) == set(FIELDS),
        "exactly six prefix fields required",
    )
    for key in FIELDS:
        shape, dtype = layout(key, prefix, queries)
        value = arrays[key]
        require(
            isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype,
            "prefix shape/dtype differs: " + key,
        )


def validate_metadata(meta):
    require(isinstance(meta, dict), "call metadata dictionary required")
    # JSON-native scalar metadata only. Preserve every field; do not drop timing/context.
    try:
        encoded = json.dumps(meta, ensure_ascii=False, allow_nan=False, sort_keys=True)
        restored = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError("finite JSON call metadata required") from error
    require(restored == meta, "metadata changed by JSON conversion")
    for key in ("call_index", "output_prefix", "query_count", "maximum_seen_native"):
        require(type(meta.get(key)) is int, "integer metadata required: " + key)
    index, prefix, queries = meta["call_index"], meta["output_prefix"], meta["query_count"]
    lo, hi = STRIDE * index, STRIDE * index + WINDOW
    require(
        0 <= index < 24 and prefix == hi <= MAX_PREFIX and 0 <= queries <= MAX_QUERIES,
        "bounded16/8 call index/prefix/query count differs",
    )
    require(
        type(meta.get("input_relative")) is list
        and all(type(x) is int for x in meta["input_relative"])
        and meta["input_relative"] == [lo, hi],
        "active input window differs",
    )
    native = meta.get("input_native")
    require(
        type(native) is list
        and len(native) == 2
        and all(type(x) is int for x in native)
        and native[0] >= lo
        and native[1] == native[0] + WINDOW
        and native[1] <= np.iinfo(np.int64).max
        and meta["maximum_seen_native"] == native[1] - 1,
        "native window/horizon differs",
    )
    require(
        type(meta.get("forward_called")) is bool
        and meta["forward_called"] == bool(queries)
        and meta.get("status") == ("ok" if queries else "no_queries"),
        "query availability metadata differs",
    )
    return lo, hi, queries


def changed_rows(left, right):
    """Return chronological changed row indices, including differing NaN payloads."""
    require(
        left.shape == right.shape and left.dtype == right.dtype, "row comparison schema differs"
    )
    row_bytes = math.prod(left.shape[1:]) * left.dtype.itemsize
    a = np.ascontiguousarray(left).view(np.uint8).reshape(len(left), row_bytes)
    b = np.ascontiguousarray(right).view(np.uint8).reshape(len(right), row_bytes)
    return np.flatnonzero(np.any(a != b, axis=1)).astype(np.int64)


def encode_delta(previous, current, meta):
    """Copy a full active window plus all changed earlier rows; never mutate inputs."""
    lo, hi, queries = validate_metadata(meta)
    validate_arrays(current, hi, queries)
    if meta["call_index"] == 0:
        require(previous is None, "initial call must not have previous prefix")
        before = None
    else:
        require(previous is not None, "subsequent call requires previous full prefix")
        validate_arrays(previous, hi - STRIDE, queries)
        before = signatures(previous)
    original = signatures(current)
    fields, payload = {}, {}
    for key in FIELDS:
        indices = (
            np.empty(0, np.int64)
            if previous is None
            else changed_rows(previous[key][:lo], current[key][:lo])
        )
        names = dict(
            active_key=key + "__active",
            outside_index_key=key + "__outside_indices",
            outside_value_key=key + "__outside_values",
            outside_changed_rows=len(indices),
        )
        payload[names["active_key"]] = current[key][lo:hi].copy(order="C")
        payload[names["outside_index_key"]] = indices.copy()
        payload[names["outside_value_key"]] = current[key][indices].copy(order="C")
        fields[key] = names
    header = dict(
        schema=SCHEMA,
        call_metadata=copy.deepcopy(meta),
        active_range=[lo, hi],
        previous_signatures=before,
        current_signatures=original,
        payload_signatures=signatures(payload),
        fields=fields,
        active_window_values_are_unconditional=True,
        unchanged_outside_window_assumed=False,
        logical_bytes=sum(a.nbytes for a in current.values()),
        delta_array_bytes=sum(a.nbytes for a in payload.values()),
        original_prefix_strides_preserved=False,
    )
    require(
        signatures(current) == original and (previous is None or signatures(previous) == before),
        "prefix inputs changed during encoding",
    )
    # A local roundtrip verifies the emitted patch before the caller saves it.
    restored = decode_delta(previous, header, payload)
    require(signatures(restored) == original, "encoded delta does not reconstruct full prefix")
    return header, payload


def decode_delta(previous, header, payload):
    """Return owned arrays only after every full-prefix and payload signature passes."""
    require(isinstance(header, dict) and header.get("schema") == SCHEMA, "delta schema differs")
    lo, hi, queries = validate_metadata(header["call_metadata"])
    require(
        header["active_range"] == [lo, hi]
        and header.get("active_window_values_are_unconditional") is True
        and header.get("unchanged_outside_window_assumed") is False
        and header.get("original_prefix_strides_preserved") is False,
        "delta representation contract differs",
    )
    expected_fields = {}
    for key in FIELDS:
        expected_fields[key] = dict(
            active_key=key + "__active",
            outside_index_key=key + "__outside_indices",
            outside_value_key=key + "__outside_values",
        )
    require(set(header["fields"]) == set(FIELDS), "delta field membership differs")
    expected_payload = {name for row in expected_fields.values() for name in row.values()}
    require(
        isinstance(payload, dict)
        and set(payload) == expected_payload
        and all(isinstance(v, np.ndarray) for v in payload.values()),
        "delta payload membership differs",
    )
    require(signatures(payload) == header["payload_signatures"], "delta payload bytes changed")
    if header["call_metadata"]["call_index"] == 0:
        require(previous is None and header["previous_signatures"] is None, "initial base differs")
    else:
        require(previous is not None, "previous prefix absent")
        validate_arrays(previous, hi - STRIDE, queries)
        require(
            signatures(previous) == header["previous_signatures"], "previous prefix bytes differ"
        )
    result = {}
    for key in FIELDS:
        desc = header["fields"][key]
        require(
            set(desc) == {*expected_fields[key], "outside_changed_rows"}
            and all(desc[k] == v for k, v in expected_fields[key].items()),
            "delta field storage key differs",
        )
        active, indices, outside = (
            payload[desc[k]] for k in ("active_key", "outside_index_key", "outside_value_key")
        )
        shape, dtype = layout(key, hi, queries)
        require(
            active.dtype == dtype
            and active.shape == (WINDOW, *shape[1:])
            and indices.dtype == np.int64
            and indices.ndim == 1
            and type(desc["outside_changed_rows"]) is int
            and len(indices) == desc["outside_changed_rows"]
            and np.all((indices >= 0) & (indices < lo))
            and np.all(np.diff(indices) > 0)
            and outside.dtype == dtype
            and outside.shape == (len(indices), *shape[1:]),
            "delta active/outside schema differs: " + key,
        )
        full = np.empty(shape, dtype)
        if previous is not None:
            full[: hi - STRIDE] = previous[key]
        full[lo:hi] = active
        full[indices] = outside
        result[key] = full
    require(
        signatures(result) == header["current_signatures"], "reconstructed full prefix bytes differ"
    )
    require(
        header["logical_bytes"] == sum(v.nbytes for v in result.values())
        and header["delta_array_bytes"] == sum(v.nbytes for v in payload.values()),
        "delta byte accounting differs",
    )
    return result

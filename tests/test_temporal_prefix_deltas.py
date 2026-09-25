"""Byte-exact synthetic full-prefix archives; no official model or real NPZ input."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import temporal_prefix_deltas as m

FIELDS = (
    "xy",
    "model_xy",
    "reported",
    "sigmoid_visibility",
    "sigmoid_confidence",
    "product",
)


def metadata(index, queries=2, first=333519):
    lo, hi = 8 * index, 16 + 8 * index
    return dict(
        call_index=index,
        input_relative=[lo, hi],
        input_native=[first + lo, first + hi],
        maximum_seen_native=first + hi - 1,
        output_prefix=hi,
        query_count=queries,
        forward_called=bool(queries),
        status="ok" if queries else "no_queries",
        forward_seconds=0.125,
        transfer_seconds=0.25,
        timing_detail={"scope": "synthetic", "parts": [1, 2]},
    )


def prefix(length, queries=2):
    result = {}
    for index, name in enumerate(FIELDS):
        shape = (length, queries, 2) if name in ("xy", "model_xy") else (length, queries)
        if name == "reported":
            result[name] = np.zeros(shape, bool)
        else:
            result[name] = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + index
    return result


def next_prefix(previous):
    length, queries = previous["reported"].shape
    result = prefix(length + 8, queries)
    for name in FIELDS:
        result[name][:length] = previous[name]
    return result


def snapshot(arrays):
    return {name: (value.dtype.str, value.shape, value.tobytes()) for name, value in arrays.items()}


def assert_exact(actual, expected):
    assert set(actual) == set(expected)
    assert snapshot(actual) == snapshot(expected)


def alter_row(arrays, field, row):
    if field == "reported":
        arrays[field][row, 0] = not arrays[field][row, 0]
    else:
        arrays[field][row].flat[0] += np.float32(123.0)


@pytest.mark.parametrize(("calls", "queries"), [(7, 3), (24, 2), (7, 0)])
def test_saved_json_npz_chain_reconstructs_every_full_prefix(tmp_path, calls, queries):
    previous, reconstructed = None, None
    for index in range(calls):
        current = prefix(16, queries) if previous is None else next_prefix(previous)
        lo, hi = index * 8, index * 8 + 16
        if queries:
            for field in FIELDS:
                alter_row(current, field, lo)
                if index and index % 3 == 0:
                    alter_row(current, field, 0)
        meta = metadata(index, queries)
        expected = snapshot(current)
        header, payload = m.encode_delta(previous, current, meta)
        assert header["schema"] == "temporal_full_prefix_delta_v1"
        assert header["active_range"] == [lo, hi]
        assert header["call_metadata"] == meta
        assert len(payload) == 18
        for field in FIELDS:
            field_meta = header["fields"][field]
            active = payload[field_meta["active_key"]]
            assert active.shape[0] == 16
            assert active.tobytes() == current[field][lo:hi].tobytes()
        header_path, archive_path = tmp_path / f"{index}.json", tmp_path / f"{index}.npz"
        header_path.write_text(json.dumps(header, allow_nan=False), encoding="utf-8")
        np.savez_compressed(archive_path, **payload)
        with np.load(archive_path, allow_pickle=False) as archive:
            restored_payload = {key: archive[key] for key in archive.files}
        reconstructed = m.decode_delta(
            reconstructed, json.loads(header_path.read_text(encoding="utf-8")), restored_payload
        )
        assert snapshot(reconstructed) == expected
        previous = current


@pytest.mark.parametrize("field", FIELDS)
def test_each_field_independently_saves_changes_before_active_window(field):
    previous = prefix(16)
    current = next_prefix(previous)
    alter_row(current, field, 2)
    header, payload = m.encode_delta(previous, current, metadata(1))
    for name in FIELDS:
        row = header["fields"][name]
        indices = payload[row["outside_index_key"]]
        assert indices.dtype == np.int64
        assert indices.tolist() == ([2] if name == field else [])
        assert row["outside_changed_rows"] == (1 if name == field else 0)
        if name == field:
            assert payload[row["outside_value_key"]].tobytes() == current[name][[2]].tobytes()
    assert_exact(m.decode_delta(previous, header, payload), current)


def test_nan_payload_and_signed_zero_changes_are_not_lost_as_equal_values():
    previous = prefix(16)
    previous["xy"].view(np.uint32)[0, 0] = [0x7FC00001, 0x80000000]
    previous["product"].view(np.uint32)[1, 0] = 0x7FC00003
    current = next_prefix(previous)
    current["xy"].view(np.uint32)[0, 0] = [0x7FC00002, 0x00000000]
    current["product"].view(np.uint32)[1, 0] = 0x7FC00004
    header, payload = m.encode_delta(previous, current, metadata(1))
    for field, expected_rows in (("xy", [0]), ("product", [1])):
        assert payload[header["fields"][field]["outside_index_key"]].tolist() == expected_rows
    assert_exact(m.decode_delta(previous, header, payload), current)


def test_product_and_reported_are_copied_without_recalculation_or_model_policy():
    current = prefix(16)
    current["sigmoid_visibility"][:] = np.float32(0.1)
    current["sigmoid_confidence"][:] = np.float32(0.2)
    current["product"][:] = np.float32(0.875)
    current["reported"][:] = False
    header, payload = m.encode_delta(None, current, metadata(0))
    assert_exact(m.decode_delta(None, header, payload), current)


def test_strided_inputs_deepcopied_metadata_and_owned_outputs_are_immutable():
    previous = prefix(16, 3)
    current = next_prefix(previous)
    for arrays in (previous, current):
        for field, original in list(arrays.items()):
            backing = np.empty((original.shape[0], 6, *original.shape[2:]), original.dtype)
            backing[:, ::2] = original
            arrays[field] = backing[:, ::2]
            assert not arrays[field].flags.c_contiguous
    alter_row(current, "model_xy", 1)
    meta = metadata(1, 3)
    before = snapshot(previous), snapshot(current), copy.deepcopy(meta)
    header, payload = m.encode_delta(previous, current, meta)
    payload_before = snapshot(payload)
    restored = m.decode_delta(previous, header, payload)
    assert_exact(restored, current)
    assert before == (snapshot(previous), snapshot(current), meta)
    assert snapshot(payload) == payload_before
    meta["timing_detail"]["parts"][0] = 999
    assert header["call_metadata"]["timing_detail"]["parts"] == [1, 2]
    for value in restored.values():
        for source in (*previous.values(), *current.values(), *payload.values()):
            assert not np.shares_memory(value, source)
    for field in FIELDS:
        alter_row(restored, field, 0)
    assert snapshot(payload) == payload_before
    assert snapshot(previous) == before[0] and snapshot(current) == before[1]


def test_unchanged_outside_rows_are_omitted_without_omitting_active_rows():
    previous = prefix(24)
    current = next_prefix(previous)
    header, payload = m.encode_delta(previous, current, metadata(2))
    for field in FIELDS:
        row = header["fields"][field]
        assert row["outside_changed_rows"] == 0
        assert payload[row["outside_index_key"]].shape == (0,)
        assert payload[row["outside_value_key"]].shape[0] == 0
        assert payload[row["active_key"]].shape[0] == 16
    assert_exact(m.decode_delta(previous, header, payload), current)


def test_all_prior_outside_rows_can_change_without_a_truncation_limit():
    previous = prefix(192, 2)
    current = next_prefix(previous)
    for field in FIELDS:
        for row in range(184):
            alter_row(current, field, row)
    header, payload = m.encode_delta(previous, current, metadata(23))
    for field in FIELDS:
        desc = header["fields"][field]
        assert desc["outside_changed_rows"] == 184
        np.testing.assert_array_equal(payload[desc["outside_index_key"]], np.arange(184))
    assert_exact(m.decode_delta(previous, header, payload), current)


@pytest.mark.parametrize(
    ("indices", "dtype"),
    [([2, 2], np.int64), ([3, 2], np.int64), ([-1], np.int64), ([8], np.int64), ([2], np.int32)],
)
def test_noncanonical_outside_indices_rejected_even_with_valid_payload_signatures(indices, dtype):
    previous = prefix(16)
    current = next_prefix(previous)
    header, payload = m.encode_delta(previous, current, metadata(1))
    desc = header["fields"]["xy"]
    # These are redundant writes of the correct existing rows. Their values
    # could reconstruct exactly, but duplicate/order/range/dtype rules must hold.
    payload[desc["outside_index_key"]] = np.array(indices, dtype=dtype)
    payload[desc["outside_value_key"]] = current["xy"][indices].copy()
    desc["outside_changed_rows"] = len(indices)
    header["payload_signatures"] = {
        key: dict(
            dtype=value.dtype.str,
            shape=list(value.shape),
            sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        )
        for key, value in payload.items()
    }
    header["delta_array_bytes"] = sum(a.nbytes for a in payload.values())
    with pytest.raises(ValueError, match="active/outside schema"):
        m.decode_delta(previous, header, payload)


@pytest.mark.parametrize("field", FIELDS)
def test_wrong_previous_bytes_rejected_even_in_overwritten_active_rows(field):
    previous = prefix(16)
    current = next_prefix(previous)
    header, payload = m.encode_delta(previous, current, metadata(1))
    alter_row(previous, field, 10)
    with pytest.raises(ValueError):
        m.decode_delta(previous, header, payload)


@pytest.mark.parametrize("component", ["active", "outside_indices", "outside_values"])
def test_changed_saved_payload_is_rejected(component):
    previous = prefix(16)
    current = next_prefix(previous)
    alter_row(current, "xy", 2)
    header, payload = m.encode_delta(previous, current, metadata(1))
    key = f"xy__{component}"
    payload[key].flat[0] += 1
    with pytest.raises(ValueError):
        m.decode_delta(previous, header, payload)


@pytest.mark.parametrize(
    "drift",
    [
        lambda h: h.update(schema="other"),
        lambda h: h.update(active_range=[0, 16]),
        lambda h: h["call_metadata"].update(query_count=3),
        lambda h: h["call_metadata"].update(call_index=2),
        lambda h: h["fields"]["xy"].update(outside_changed_rows=99),
        lambda h: h["fields"]["xy"].update(active_key="model_xy__active"),
        lambda h: h["current_signatures"].update(xy={"changed": True}),
        lambda h: h.update(previous_signatures=None),
    ],
)
def test_changed_header_cannot_return_a_reconstruction(drift):
    previous = prefix(16)
    current = next_prefix(previous)
    header, payload = m.encode_delta(previous, current, metadata(1))
    drift(header)
    with pytest.raises(ValueError):
        m.decode_delta(previous, header, payload)


@pytest.mark.parametrize(
    "drift",
    [
        lambda x: x.update(call_index=True),
        lambda x: x.update(input_relative=[0, 24]),
        lambda x: x.update(input_native=[333527, 333542]),
        lambda x: x.update(maximum_seen_native=333543),
        lambda x: x.update(output_prefix=23),
        lambda x: x.update(query_count=3),
        lambda x: x.update(forward_called=False),
        lambda x: x.update(status="no_queries"),
    ],
)
def test_inconsistent_call_metadata_rejected(drift):
    previous = prefix(16)
    current = next_prefix(previous)
    meta = metadata(1)
    drift(meta)
    with pytest.raises(ValueError):
        m.encode_delta(previous, current, meta)


@pytest.mark.parametrize("fault", ["skipped", "missing_previous", "query_count", "dtype", "extra"])
def test_prefix_order_query_schema_and_dtype_fail_closed(fault):
    previous, current, meta = prefix(16), prefix(24), metadata(1)
    if fault == "skipped":
        current, meta = prefix(32), metadata(2)
    elif fault == "missing_previous":
        previous = None
    elif fault == "query_count":
        previous = prefix(16, 3)
    elif fault == "dtype":
        current["xy"] = current["xy"].astype(np.float64)
    else:
        current["extra"] = np.zeros(1)
    with pytest.raises(ValueError):
        m.encode_delta(previous, current, meta)

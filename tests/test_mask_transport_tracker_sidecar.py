"""Synthetic anchor lineage and sidecar codec; no tracker advance, fit or assets."""

import copy
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import mask_transport_tracker as m  # noqa: E402


def test_anchor_lineage_follows_repeated_reordered_parents_without_relabelling():
    previous = np.array([3, 0, 5], np.int64)
    parents = np.array([2, 0, 2, 1, 0], np.int64)
    before = previous.tobytes(), parents.tobytes()
    result = m.propagate_anchor_ids(previous, parents)
    np.testing.assert_array_equal(result, [5, 3, 5, 0, 3])
    next_parents = np.array([4, 1, 3, 4], np.int64)
    np.testing.assert_array_equal(m.propagate_anchor_ids(result, next_parents), [3, 3, 0, 3])
    assert tuple(inspect.signature(m.propagate_anchor_ids).parameters) == ("previous", "parents")
    assert not np.shares_memory(result, previous) and not np.shares_memory(result, parents)
    result[:] = -99
    assert (previous.tobytes(), parents.tobytes()) == before


@pytest.mark.parametrize(
    "kind", ["negative", "outside", "empty_parent", "empty_beam", "int32", "matrix"]
)
def test_anchor_parent_schema_or_range_drift_is_rejected(kind):
    previous, parents = np.array([0, 1, 2], np.int64), np.array([2, 1], np.int64)
    if kind == "negative":
        parents[0] = -1
    elif kind == "outside":
        parents[0] = 3
    elif kind == "empty_parent":
        parents = np.empty(0, np.int64)
    elif kind == "empty_beam":
        previous = np.empty(0, np.int64)
    elif kind == "int32":
        parents = parents.astype(np.int32)
    else:
        parents = parents[:, None]
    with pytest.raises(ValueError):
        m.propagate_anchor_ids(previous, parents)


@dataclass(frozen=True)
class SyntheticAnchor:
    anchor_id: int
    links: tuple


def test_nested_sidecar_npz_json_roundtrip_preserves_exact_array_bits_and_owned_copies(tmp_path):
    bits = np.array([0x7FC00011, 0x7FC01234, 0x80000000, 0], np.uint32)
    floats = bits.view(np.float32)
    strided = np.arange(24, dtype=np.int16).reshape(4, 6)[:, ::2]
    text = np.array(["尾", "unknown", "胸"], dtype="<U8")
    scalar = np.int64(17)
    source = dict(
        anchor=SyntheticAnchor(1, (np.array([[True, False, False]]),)),
        values=[floats, strided, text, scalar],
        labels={"native": 100, "conditional": True, "reason": None, "weight": 0.25},
    )
    original = [floats, strided, text, np.asarray(scalar), source["anchor"].links[0]]
    before = [m.signature(value) for value in original]
    arrays = {}
    tree = m._pack(source, arrays)
    assert all(
        not any(np.shares_memory(value, item) for item in original) for value in arrays.values()
    )
    header_path, payload_path = tmp_path / "side.json", tmp_path / "side.npz"
    header_path.write_text(json.dumps(tree, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    np.savez_compressed(payload_path, **arrays)
    with np.load(payload_path, allow_pickle=False) as stored:
        loaded = {name: stored[name] for name in stored.files}
    used = set()
    result = m._unpack(json.loads(header_path.read_text(encoding="utf-8")), loaded, used)
    assert used == set(loaded) == set(arrays)
    assert result["anchor"]["anchor_id"] == 1
    assert result["labels"] == source["labels"]
    recovered = [*result["values"], result["anchor"]["links"][0]]
    assert [m.signature(value) for value in recovered] == before
    assert recovered[0].view(np.uint32).tolist() == bits.tolist()
    assert recovered[1].dtype == np.int16 and recovered[2].dtype.str == "<U8"
    assert recovered[3].shape == () and recovered[3].dtype == np.int64
    for value in recovered:
        assert not any(np.shares_memory(value, saved) for saved in loaded.values())
        assert not any(np.shares_memory(value, item) for item in original)
    recovered[0][0] = 42
    loaded["root/values/1"][:] = -20
    assert [m.signature(value) for value in original] == before
    assert m.signature(recovered[1]) == before[1]


@pytest.mark.parametrize(
    "value", [np.array([object()], dtype=object), float("nan"), float("inf"), {"__array__": "x"}]
)
def test_pack_rejects_pickle_objects_nonfinite_json_scalars_and_reserved_markers(value):
    with pytest.raises(ValueError):
        m._pack(value, {})


def test_pack_rejects_colliding_nested_array_names():
    source = {"a/b": np.array([1]), "a": {"b": np.array([2])}}
    with pytest.raises(ValueError, match="sidecar array"):
        m._pack(source, {})


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_unpack_rejects_unbound_or_reused_array_references(kind):
    arrays = {"saved": np.array([0, 1], np.int64)}
    marker = {"__array__": "missing" if kind == "missing" else "saved"}
    tree = marker if kind == "missing" else [marker, marker.copy()]
    with pytest.raises(ValueError, match="array reference"):
        m._unpack(tree, arrays, set())


def envelope():
    core = {"history_offsets": np.array([0, 1], np.int64)}
    metadata = {"processed_frames": 1, "context": {"coefficient": 3.0}}
    arrays = {}
    tree = m._pack({"active_anchor_ids": np.array([0], np.int64)}, arrays)
    header = dict(
        schema=m.SCHEMA,
        core_array_signatures={k: m.signature(v) for k, v in core.items()},
        core_metadata_sha256=m.json_hash(metadata),
        array_signatures={k: m.signature(v) for k, v in arrays.items()},
        tree=tree,
    )
    header["content_sha256"] = m.json_hash(header)
    return core, metadata, arrays, header


@pytest.mark.parametrize(
    "kind", ["core_array", "core_context", "side_array", "header", "extra_array"]
)
def test_sidecar_envelope_rejects_state_or_metadata_drift_before_any_tracker_work(kind):
    core, metadata, arrays, header = envelope()
    if kind == "core_array":
        core["history_offsets"][1] = 2
    elif kind == "core_context":
        metadata["context"]["coefficient"] = 0.0
    elif kind == "side_array":
        arrays["root/active_anchor_ids"][0] = 5
    elif kind == "header":
        header["tree"]["unsealed_metadata"] = True
    else:
        # Even internally rehashed extra payload cannot become an unused field.
        arrays["unreferenced"] = np.array([3], np.int64)
        header["array_signatures"] = {k: m.signature(v) for k, v in arrays.items()}
        del header["content_sha256"]
        header["content_sha256"] = m.json_hash(header)
    with pytest.raises(ValueError, match="sidecar"):
        m.MaskTransportTracker._validated_sidecar(object(), core, metadata, arrays, header)


def test_pack_and_unpack_do_not_mutate_caller_tree_or_reuse_array_identity():
    tree = {"arr": {"__array__": "x"}, "labels": [None, True, "conditional", 7]}
    before = copy.deepcopy(tree)
    arrays = {"x": np.arange(4, dtype=np.float64)}
    first = m._unpack(tree, arrays, set())
    second = m._unpack(tree, arrays, set())
    first["arr"][:] = -1
    first["labels"][2] = "changed"
    assert tree == before
    np.testing.assert_array_equal(second["arr"], np.arange(4))
    assert second["labels"][2] == "conditional"

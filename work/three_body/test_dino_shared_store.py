"""Synthetic lossless codec tests; no model, actual image or geometry evidence."""

import copy

import diagnose_dino_fixed_masks as logical
import dino_shared_store as m
import numpy as np
import pytest


def geometry(n=4):
    return dict(
        token_index=np.arange(n, dtype=np.int64),
        token_center_crop_xy=np.arange(2 * n, dtype=np.float64).reshape(n, 2),
        token_center_native_xy=np.arange(2 * n, dtype=np.float64).reshape(n, 2) + 100,
        token_native_xyxy=np.zeros((n, 4), np.int64),
        token_footprint_pixels=np.ones(n, np.int64),
        token_core_pixels=np.ones(n, np.int64),
        token_eligible=np.ones(n, bool),
    )


def fixture(mode="ties"):
    left_tokens = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    right_tokens = left_tokens.copy()
    lg, rg = geometry(), geometry()
    if mode == "partial":
        left_tokens[1] = np.nan
        lg["token_eligible"][2] = False
        rg["token_center_native_xy"][3] = np.nan
    elif mode == "empty":
        lg["token_eligible"][:] = False
    elif mode == "nonfinite":
        left_tokens[:] = np.nan
        right_tokens[1] = 0
    full = m.partial.token_correspondence(left_tokens, right_tokens, lg, rg)
    arrays, meta = logical.pack_correspondence(full)
    return left_tokens, right_tokens, lg, rg, arrays, meta


@pytest.mark.parametrize("mode", ["ties", "partial", "empty", "nonfinite"])
def test_all32_logic_and_metadata_roundtrip_preserve_all_missing_and_ties(mode):
    left_tokens, right_tokens, lg, rg, arrays, meta = fixture(mode)
    before = m.signatures(arrays), m.canonical(meta), m.signature(left_tokens), m.signatures(lg)
    left, right = m.make_features(left_tokens, lg), m.make_features(right_tokens, rg)
    payload, spec = m.split_pair(arrays, meta, left, right, lg, rg)
    restored, metadata = m.expand_pair(payload, spec, left, right, lg, rg)
    assert set(restored) == m.LOGICAL_KEYS and m.signatures(restored) == m.signatures(arrays)
    assert m.canonical(metadata) == m.canonical(meta)
    assert before == (
        m.signatures(arrays),
        m.canonical(meta),
        m.signature(left_tokens),
        m.signatures(lg),
    )
    for key in restored:
        assert not np.shares_memory(restored[key], arrays[key])
    if mode == "ties":
        assert restored["forward_nn"][0].sum() == 2


def test_minimum_nan_payload_bits_use_tiny_explicit_fallback():
    left_tokens, right_tokens, lg, rg, arrays, meta = fixture("empty")
    arrays["left_minimum"].view(np.uint64)[0] = 0x7FF8000000000123
    meta["left_minimum"].update(m.signature(arrays["left_minimum"]))
    left, right = m.make_features(left_tokens, lg), m.make_features(right_tokens, rg)
    payload, spec = m.split_pair(arrays, meta, left, right, lg, rg)
    assert "left_minimum" in payload
    actual, _ = m.expand_pair(payload, spec, left, right, lg, rg)
    assert actual["left_minimum"].view(np.uint64)[0] == 0x7FF8000000000123


@pytest.mark.parametrize(
    "fault", ["metadata_sha", "normalization", "geometry", "missing_key", "nn_tie"]
)
def test_modified_original_logic_is_rejected(fault):
    left_tokens, right_tokens, lg, rg, arrays, meta = fixture()
    left, right = m.make_features(left_tokens, lg), m.make_features(right_tokens, rg)
    if fault == "metadata_sha":
        meta["distances"]["sha256"] = "0" * 64
    elif fault == "normalization":
        left["normalization__normalized"][0, 0] = 0.5
    elif fault == "geometry":
        lg["token_center_native_xy"][0, 0] += 1
    elif fault == "missing_key":
        del arrays["finite_pair"]
    else:
        arrays["forward_nn"][0, 1] = False
        meta["forward_nn"].update(m.signature(arrays["forward_nn"]))
    with pytest.raises(ValueError):
        m.split_pair(arrays, meta, left, right, lg, rg)


def write_store(path):
    left_tokens, right_tokens, lg, rg, arrays, meta = fixture("partial")
    binding = dict(source="synthetic", version=1)
    writer = m.StoreWriter(path, bindings=binding)
    writer.add_crop("left", lg)
    writer.add_crop("right", rg)
    writer.add_feature("left_raw", "left", left_tokens)
    writer.add_feature("right_raw", "right", right_tokens)
    writer.add_pair("p0000_raw", "left_raw", "right_raw", arrays, meta)
    index = writer.finish()
    return index, binding, arrays, meta


def test_disk_shared_once_and_fresh_reader_copy(tmp_path):
    root = tmp_path / "store"
    index, binding, arrays, meta = write_store(root)
    reader = m.StoreReader(
        root,
        expected_bindings=binding,
        expected_manifest_sha256=m.sha(root / "store.manifest.json"),
    )
    actual, metadata = reader.read_pair("p0000_raw")
    assert m.signatures(actual) == m.signatures(arrays) and m.canonical(metadata) == m.canonical(
        meta
    )
    assert len(index["crops"]) == 2 and len(index["features"]) == 2 and len(index["pairs"]) == 1
    assert set(index["pairs"]["p0000_raw"]["arrays"]) == set(m.PAIR_KEYS)
    actual["distances"][:] = 999
    again, _ = reader.read_pair("p0000_raw")
    assert m.signatures(again) == m.signatures(arrays)
    assert reader.verify_all()["files"] == 6


@pytest.mark.parametrize("fault", ["bindings", "manifest", "payload", "extra_file", "missing_file"])
def test_file_binding_membership_tamper_rejected(tmp_path, fault):
    root = tmp_path / "store"
    index, binding, _, _ = write_store(root)
    digest = m.sha(root / "store.manifest.json")
    if fault == "bindings":
        binding["version"] = True
    elif fault == "manifest":
        (root / "store.manifest.json").write_bytes(b"{}")
    elif fault == "payload":
        (root / index["pairs"]["p0000_raw"]["path"]).write_bytes(b"bad")
    elif fault == "missing_file":
        (root / index["features"]["left_raw"]["path"]).unlink()
    else:
        (root / "extra").write_bytes(b"x")
    with pytest.raises((ValueError, FileNotFoundError)):
        reader = m.StoreReader(root, expected_bindings=binding, expected_manifest_sha256=digest)
        reader.verify_all()


def test_unfinished_store_has_no_success_manifest_and_cannot_overwrite(tmp_path):
    path = tmp_path / "store"
    writer = m.StoreWriter(path, bindings={})
    with pytest.raises(ValueError):
        writer.add_crop("../bad", geometry())
    assert not (path / "store.manifest.json").exists()
    with pytest.raises(ValueError):
        m.StoreWriter(path, bindings={})


def test_duplicate_features_and_closed_writer_refused(tmp_path):
    path = tmp_path / "store"
    writer = m.StoreWriter(path, bindings={})
    writer.add_crop("a", geometry())
    tokens = fixture()[0]
    writer.add_feature("a_raw", "a", tokens)
    with pytest.raises(ValueError):
        writer.add_feature("a_raw", "a", tokens)
    writer.finish()
    with pytest.raises(ValueError):
        writer.add_feature("a_masked", "a", tokens)


def test_json_payload_and_python_type_preserved_without_alias():
    original = {"flag": True, "count": 1, "missing": {"nonfinite": "NaN"}, "bounds": [None, 1.0]}
    cloned = m.clone_json(original)
    assert m.canonical(cloned) == m.canonical(original)
    cloned["bounds"][0] = 0
    assert original["bounds"][0] is None
    assert m.canonical({"value": True}) != m.canonical({"value": 1})


def test_input_float32_signed_zero_and_nan_bits_preserved_in_feature_store():
    tokens = fixture()[0]
    tokens[0, 1] = -0.0
    tokens.view(np.uint32)[1, 0] = 0x7FC00123
    before = copy.deepcopy(tokens)
    output = m.make_features(tokens, geometry())
    assert m.same(output["layernorm_tokens"], tokens) and m.same(before, tokens)
    assert not np.shares_memory(output["layernorm_tokens"], tokens)

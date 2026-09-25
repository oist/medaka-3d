"""Storage integrity and aliasing checks, independent of the tracker state model."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
SPEC = importlib.util.spec_from_file_location("three_body_checkpoint_tests",
                                             LAB / "tracker_checkpoint.py")
CHECKPOINT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKPOINT)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def assert_arrays_equal(first, second):
    assert set(first) == set(second)
    for name, expected in first.items():
        actual = second[name]
        assert actual.dtype == expected.dtype, name
        assert actual.shape == expected.shape, name
        assert actual.tobytes(order="C") == expected.tobytes(order="C"), name


def bindings():
    return {"source": {"a.mp4": "ab" * 32}, "views": ["top", "side1", "side2"],
            "version": 1, "flow": True, "box": [-0.0, 192.0], "optional": None}


def create(tmp_path):
    root = tmp_path / "checkpoint"
    arrays = {"pose": np.arange(27, dtype=np.float64).reshape(3, 3, 3),
              "parent": np.array([0, 2, 1], np.int64)}
    metadata = {"native_frame": 333537, "counts": {"proposed": 6}, "kinds": ["flow", "base"]}
    manifest = CHECKPOINT.write_checkpoint(root, arrays, metadata, bindings=bindings())
    return root, arrays, metadata, manifest


def edit_manifest(root, mutate):
    path = root / CHECKPOINT.MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_bytes(canonical(manifest))


def update_payload_seal(root):
    path = root / CHECKPOINT.ARRAYS_NAME
    edit_manifest(root, lambda manifest: manifest.update(
        payload={"file": path.name, "sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}
    ))


def test_roundtrip_preserves_logical_bytes_shapes_and_dtypes(tmp_path):
    noncanonical_nan = np.array([0x7ff8000000000012, 0x8000000000000000,
                                0x7ff0000000000000], np.uint64).view(np.float64)
    structured = np.ndarray((2,), dtype=[("x", "u1"), ("y", "<i4")],
                            buffer=bytearray(range(10)))
    arrays = {
        "poses": np.arange(48, dtype=np.float64).reshape(4, 4, 3)[::-1, ::2],
        "scalar": np.array(333534, np.int64), "zero_sized": np.empty((3, 0, 2), np.float32),
        "endian": np.array([3.25, -0.0, -np.inf], ">f8"), "float_bits": noncanonical_nan,
        "bool": np.array([[False, True]], bool), "kinds": np.array(["基準", "flow"], "U8"),
        "complex": np.array([1 + 2j, complex(-0.0, np.inf)], np.complex128),
        "time": np.array([0, 1], dtype="datetime64[ns]"), "structured": structured,
        "file": np.array(1, np.uint32), "../logical/name": np.array([7], np.int16),
    }
    metadata = {"cost": None, "native_frame": 17, "names": ["G0", "G1", "G2"]}
    root = tmp_path / "new"
    manifest = CHECKPOINT.write_checkpoint(root, arrays, metadata, bindings=bindings())
    actual, loaded_metadata, loaded_manifest = CHECKPOINT.read_checkpoint(
        root, expected_bindings=bindings())
    assert_arrays_equal(arrays, actual)
    assert loaded_metadata == metadata
    assert loaded_manifest == manifest
    assert all(value.flags.c_contiguous and value.flags.writeable for value in actual.values())
    assert manifest["payload"]["sha256"] == sha((root / "arrays.npz").read_bytes())
    assert manifest["metadata_canonical_sha256"] == sha(canonical(metadata))
    for name, descriptor in manifest["arrays"].items():
        assert descriptor["bytes_sha256"] == sha(arrays[name].tobytes(order="C"))
        assert descriptor["shape"] == list(arrays[name].shape)
        assert descriptor["nbytes"] == arrays[name].nbytes
    with zipfile.ZipFile(root / "arrays.npz") as archive:
        assert all(name.startswith("array_") and name.endswith(".npy")
                   for name in archive.namelist())
    assert not (tmp_path / "logical").exists()


def test_save_and_each_load_are_detached_from_mutable_inputs(tmp_path):
    root, arrays, metadata, manifest = create(tmp_path)
    expected = {key: value.copy() for key, value in arrays.items()}
    arrays["pose"][:] = -500
    metadata["counts"]["proposed"] = 999
    manifest["bindings"]["source"]["a.mp4"] = "mutated"
    first, first_meta, first_manifest = CHECKPOINT.read_checkpoint(
        root, expected_bindings=bindings())
    second, second_meta, second_manifest = CHECKPOINT.read_checkpoint(root,
                                                                     expected_bindings=bindings())
    assert_arrays_equal(first, expected)
    first["pose"][:] = 1000
    first_meta["counts"]["proposed"] = 500
    first_manifest["bindings"]["views"].reverse()
    assert_arrays_equal(second, expected)
    assert second_meta["counts"]["proposed"] == 6
    assert second_manifest["bindings"] == bindings()
    assert not np.shares_memory(first["pose"], second["pose"])


def test_all_inputs_are_snapshotted_before_first_filesystem_write(tmp_path, monkeypatch):
    arrays = {"first": np.arange(9).reshape(3, 3), "second": np.array([-0.0, 2.0])}
    expected = {key: value.copy() for key, value in arrays.items()}
    metadata = {"beam": ["first", "second"]}
    binding = bindings()
    original = CHECKPOINT._write_exclusive

    def mutate_caller_inputs(path, data):
        if path.name == "metadata.json":
            arrays["first"][:] = -1
            arrays["second"][:] = 7
            metadata["beam"].reverse()
            binding["views"].reverse()
        original(path, data)

    monkeypatch.setattr(CHECKPOINT, "_write_exclusive", mutate_caller_inputs)
    root = tmp_path / "snapshotted"
    CHECKPOINT.write_checkpoint(root, arrays, metadata, bindings=binding)
    saved, saved_metadata, _ = CHECKPOINT.read_checkpoint(root, expected_bindings=bindings())
    assert_arrays_equal(saved, expected)
    assert saved_metadata == {"beam": ["first", "second"]}


def test_empty_arrays_and_dicts_are_valid(tmp_path):
    root = tmp_path / "empty"
    CHECKPOINT.write_checkpoint(root, {}, {}, bindings={})
    arrays, metadata, manifest = CHECKPOINT.read_checkpoint(root, expected_bindings={})
    assert arrays == metadata == manifest["arrays"] == {}


def test_dictionary_binding_order_is_irrelevant(tmp_path):
    root, *_ = create(tmp_path)
    expected = dict(reversed(list(bindings().items())))
    CHECKPOINT.read_checkpoint(root, expected_bindings=expected)


@pytest.mark.parametrize("key,value", [("flow", 1), ("version", 1.0), ("box", [0.0, 192.0]),
                                       ("views", ["side1", "top", "side2"]),
                                       ("source", {"a.mp4": "changed"})])
def test_bindings_compare_types_nested_values_and_sequence_order(tmp_path, key, value):
    root, *_ = create(tmp_path)
    changed = bindings()
    changed[key] = value
    with pytest.raises(ValueError, match="bindings differ"):
        CHECKPOINT.read_checkpoint(root, expected_bindings=changed)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), np.int64(3), Path("x"),
                                  (1, 2), {1: "integer key"}, {"nested": [float("nan")]}])
@pytest.mark.parametrize("field", ["metadata", "bindings"])
def test_noncanonical_json_inputs_are_rejected_before_writing(tmp_path, bad, field):
    root = tmp_path / "invalid"
    arguments = {"metadata": {}, "bindings": {}}
    arguments[field] = {"value": bad}
    with pytest.raises(ValueError, match="JSON values"):
        CHECKPOINT.write_checkpoint(root, {}, **arguments)
    assert not root.exists()


@pytest.mark.parametrize("bad", [np.array([object()], object),
                                  np.array([(object(),)], dtype=[("nested", object)]),
                                  np.ma.array([1, 2], mask=[False, True]),
                                  np.array([1], dtype=np.dtype("i8", metadata={"unit": "mm"})),
                                  [1, 2]])
def test_object_metadata_or_nonarray_semantics_are_never_silently_saved(tmp_path, bad):
    root = tmp_path / "invalid"
    with pytest.raises(ValueError):
        CHECKPOINT.write_checkpoint(root, {"bad": bad}, {}, bindings={})
    assert not root.exists()


@pytest.mark.parametrize("aligned", [False, True])
def test_npy_dtype_roundtrip_must_be_lossless_before_success_seal(tmp_path, aligned):
    # NPY drops the aligned-struct flag; fail explicitly rather than returning a
    # nominally similar but observably different dtype to the caller.
    root = tmp_path / "unsupported"
    dtype = np.dtype({"names": ["a", "b"], "formats": ["u1", "<i4"],
                      "offsets": [0, 4], "itemsize": 8}, align=aligned)
    # Non-field padding bytes are deliberately nonzero: NumPy can discard them.
    array = np.ndarray((2,), dtype=dtype, buffer=bytearray(range(16)))
    with pytest.raises(ValueError, match="dtype/shape/bytes"):
        CHECKPOINT.write_checkpoint(root, {"value": array}, {}, bindings={})
    assert root.exists()
    assert not (root / "manifest.json").exists()


def test_existing_success_or_failed_directory_is_never_overwritten(tmp_path, monkeypatch):
    root, arrays, metadata, _ = create(tmp_path)
    original = {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(FileExistsError):
        CHECKPOINT.write_checkpoint(root, arrays, metadata, bindings=bindings())
    assert {path.name: path.read_bytes() for path in root.iterdir()} == original
    failed = tmp_path / "failed"

    def fail_save(stream, **arrays):
        stream.write(b"partial archive")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(CHECKPOINT.np, "savez_compressed", fail_save)
    with pytest.raises(OSError, match="interrupted"):
        CHECKPOINT.write_checkpoint(failed, arrays, metadata, bindings=bindings())
    assert (failed / "arrays.npz").read_bytes() == b"partial archive"
    assert not (failed / "manifest.json").exists()
    with pytest.raises(ValueError, match=r"manifest\.json"):
        CHECKPOINT.read_checkpoint(failed, expected_bindings=bindings())
    with pytest.raises(FileExistsError):
        CHECKPOINT.write_checkpoint(failed, arrays, metadata, bindings=bindings())


def test_seal_is_last_after_complete_payload_readback(tmp_path, monkeypatch):
    original = CHECKPOINT._write_exclusive
    events = []

    def inspect_write(path, data):
        events.append(path.name)
        if path.name == "manifest.json":
            assert (path.parent / "metadata.json").is_file()
            with np.load(path.parent / "arrays.npz", allow_pickle=False) as archive:
                assert archive.files
            assert not path.exists()
        original(path, data)

    monkeypatch.setattr(CHECKPOINT, "_write_exclusive", inspect_write)
    create(tmp_path)
    assert events == ["metadata.json", "manifest.json"]


@pytest.mark.parametrize("filename", ["arrays.npz", "metadata.json"])
def test_payload_corruption_is_rejected(tmp_path, filename):
    root, *_ = create(tmp_path)
    path = root / filename
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum/length"):
        CHECKPOINT.read_checkpoint(root, expected_bindings=bindings())


@pytest.mark.parametrize("change", [lambda m: m.update(status="running"),
                                    lambda m: m.update(schema="future"),
                                    lambda m: m.update(extra=1),
                                    lambda m: m.update(bindings_sha256="0" * 64),
                                    lambda m: m.update(metadata_canonical_sha256="0" * 64),
                                    lambda m: m["payload"].update(file="../arrays.npz"),
                                    lambda m: m["arrays"]["pose"].update(shape=[27]),
                                    lambda m: m["arrays"]["pose"].update(nbytes=1),
                                    lambda m: m["arrays"]["pose"].update(bytes_sha256="0" * 64),
                                    lambda m: m["arrays"]["pose"]["dtype"].update(str="<f4")])
def test_manifest_inconsistencies_are_rejected(tmp_path, change):
    root, *_ = create(tmp_path)
    edit_manifest(root, change)
    with pytest.raises(ValueError):
        CHECKPOINT.read_checkpoint(root, expected_bindings=bindings())


def test_changed_array_rejected_even_if_npz_file_seal_was_updated(tmp_path):
    root, *_ = create(tmp_path)
    path = root / "arrays.npz"
    with np.load(path, allow_pickle=False) as archive:
        storage = {key: archive[key] for key in archive.files}
    storage["array_000001"][0, 0, 0] += 1
    np.savez_compressed(path, **storage)
    update_payload_seal(root)
    with pytest.raises(ValueError, match="array dtype/shape/bytes"):
        CHECKPOINT.read_checkpoint(root, expected_bindings=bindings())


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "object"])
def test_npz_membership_and_no_pickle_are_checked_beyond_file_sha(tmp_path, mutation):
    root, *_ = create(tmp_path)
    path = root / "arrays.npz"
    with np.load(path, allow_pickle=False) as archive:
        storage = {key: archive[key] for key in archive.files}
    if mutation == "missing":
        storage.pop("array_000001")
    elif mutation == "extra":
        storage["unregistered"] = np.array([1])
    elif mutation == "object":
        storage["array_000001"] = np.array([object()], object)
    np.savez_compressed(path, **storage)
    if mutation == "duplicate":
        with zipfile.ZipFile(path, "a") as archive, pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("array_000000.npy", archive.read("array_000000.npy"))
    update_payload_seal(root)
    with pytest.raises(ValueError, match=r"membership|Object arrays"):
        CHECKPOINT.read_checkpoint(root, expected_bindings=bindings())


@pytest.mark.parametrize("filename", ["manifest.json", "metadata.json", "arrays.npz"])
def test_missing_files_are_rejected(tmp_path, filename):
    root, *_ = create(tmp_path)
    (root / filename).rename(root / (filename + ".withheld"))
    with pytest.raises(ValueError, match="missing"):
        CHECKPOINT.read_checkpoint(root, expected_bindings=bindings())


@pytest.mark.parametrize("raw", [b'{"schema":', b'{"status":"ok","status":"ok"}',
                                  b'{"bad":NaN}'])
def test_partial_or_ambiguous_json_is_not_a_success_seal(tmp_path, raw):
    root, *_ = create(tmp_path)
    (root / "manifest.json").write_bytes(raw)
    with pytest.raises(ValueError):
        CHECKPOINT.read_checkpoint(root, expected_bindings=bindings())


def test_manifest_replacement_during_load_is_detected(tmp_path, monkeypatch):
    root, *_ = create(tmp_path)
    original = CHECKPOINT._payload

    def replace_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        path = root / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(CHECKPOINT, "_payload", replace_after_read)
    with pytest.raises(ValueError, match="changed during"):
        CHECKPOINT.read_checkpoint(root, expected_bindings=bindings())

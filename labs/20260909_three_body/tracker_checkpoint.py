"""Lossless, non-pickle storage for a caller-defined tracker state snapshot.

The codec validates storage integrity, not tracking semantics or the files named in
``bindings``. The runner must validate those files, snapshot the executing code, and
bind the final manifest SHA in its own provenance. These hashes alone do not establish
authenticity against coordinated replacement of a checkpoint and all of its hashes.

Each destination is a new directory. Payloads are written and read back before the
exclusive final ``manifest.json`` success seal. Failed attempts are retained and may
not be overwritten. Readers reject missing, truncated, or inconsistent seals. This
is a completed-frame checkpoint; publication during an in-flight tracker step is the
caller's error. No CUDA objects, RNG state, or tracker-specific interpretation occurs.

Metadata/bindings use JSON-native dict/list/scalar values with finite floats. Numeric
arrays may contain NaN/Inf and retain signed zero and NaN payload bits. Array logical
names never become paths or ZIP members. Array layout/strides are normalized to C
order, while dtype, shape and logical element bytes are preserved. Object dtypes,
dtype metadata, and dtypes not represented losslessly by NPY are rejected explicitly.
"""

import hashlib
import json
import math
import os
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

SCHEMA = "medaka_tracker_checkpoint_v1"
MANIFEST_NAME = "manifest.json"
ARRAYS_NAME = "arrays.npz"
METADATA_NAME = "metadata.json"


def _json_value(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise ValueError("JSON values must be native dict/list/scalars; floats must be finite")


def _canonical(value):
    _json_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode(data):
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs)
    _json_value(value)
    return value


def _json_dict(value, label):
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON-native dict")
    return _canonical(value)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file_record(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"file": path.name, "sha256": digest.hexdigest(), "bytes": size}


def _dtype_record(dtype):
    if dtype.hasobject:
        raise ValueError("object dtype is forbidden; checkpoint payloads never use pickle")
    if dtype.metadata is not None:
        raise ValueError("dtype metadata is not preserved by NPY")
    if dtype.fields:
        for name in dtype.names:
            _dtype_record(dtype.fields[name][0])
    if dtype.subdtype:
        _dtype_record(dtype.subdtype[0])
    # The NPY descriptor handles structured fields, subarrays, titles, and padding.
    descriptor = json.loads(json.dumps(np.lib.format.dtype_to_descr(dtype)))
    return {"str": dtype.str, "descr": descriptor, "aligned": dtype.isalignedstruct}


def _copy_array(value):
    if not isinstance(value, np.ndarray) or isinstance(value, np.ma.MaskedArray):
        raise ValueError("array values must be ndarrays without masked-array semantics")
    _dtype_record(value.dtype)
    # A field-wise structured-array copy can discard padding bytes. A byte buffer
    # copy preserves these as well as noncanonical NaN payloads and signed zero.
    buffer = bytearray(value.tobytes(order="C"))
    return np.ndarray(value.shape, dtype=value.dtype, buffer=buffer, order="C")


def _array_record(value, storage_key):
    record = {"storage_key": storage_key, "dtype": _dtype_record(value.dtype),
              "shape": list(value.shape), "nbytes": int(value.nbytes)}
    data = memoryview(value.reshape(-1).view(np.uint8))
    record["bytes_sha256"] = _sha(data)
    digest = hashlib.sha256(_canonical(record) + b"\x00")
    digest.update(data)
    record["sha256"] = digest.hexdigest()
    return record


def _write_exclusive(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _regular_path(root, filename):
    path = root / filename
    if not path.is_file() or path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError(f"checkpoint file missing or not a direct regular file: {filename}")
    return path


def _check_record(path, record):
    if _canonical(_file_record(path)) != _canonical(record):
        raise ValueError(f"file checksum/length mismatch: {path.name}")


def _payload(root, manifest, *, retain_arrays=True):
    metadata_path = _regular_path(root, METADATA_NAME)
    array_path = _regular_path(root, ARRAYS_NAME)
    _check_record(metadata_path, manifest["metadata"])
    _check_record(array_path, manifest["payload"])
    metadata_data = metadata_path.read_bytes()
    metadata = _decode(metadata_data)
    if _json_dict(metadata, "metadata") != metadata_data:
        raise ValueError("metadata is not canonical JSON")
    if _sha(metadata_data) != manifest["metadata_canonical_sha256"]:
        raise ValueError("metadata canonical checksum mismatch")
    descriptions = manifest["arrays"]
    if type(descriptions) is not dict or any(type(key) is not str or not key
                                           for key in descriptions):
        raise ValueError("array descriptions must have nonempty string names")
    expected_members = []
    for index, (name, record) in enumerate(sorted(descriptions.items())):
        if type(record) is not dict or record.get("storage_key") != f"array_{index:06d}":
            raise ValueError(f"invalid array storage key: {name}")
        expected_members.append(f"array_{index:06d}.npy")
    with zipfile.ZipFile(array_path) as archive:
        members = archive.namelist()
        if sorted(members) != expected_members:
            raise ValueError("NPZ array membership is missing, extra, or duplicated")
    result = {}
    with np.load(array_path, allow_pickle=False) as archive:
        for name, record in sorted(descriptions.items()):
            value = archive[record["storage_key"]]
            actual = _array_record(value, record["storage_key"])
            if _canonical(actual) != _canonical(record):
                raise ValueError(f"array dtype/shape/bytes checksum mismatch: {name}")
            if retain_arrays:
                result[name] = _copy_array(value)
    # Catch concurrent payload replacement across the read operation.
    _check_record(metadata_path, manifest["metadata"])
    _check_record(array_path, manifest["payload"])
    return result, metadata


def write_checkpoint(output, arrays, metadata, *, bindings):
    """Write a new completed snapshot, returning a detached success manifest.

    ``arrays`` maps nonempty logical names to NumPy arrays. JSON values must be
    normalized by the caller (e.g. tuples/Path/NumPy scalars are not silently cast).
    Existing directories, including failed attempts, are never reused. The codec
    does not dereference any path a caller places in ``bindings``.
    """
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"checkpoint destination already exists: {root}")
    if not isinstance(arrays, Mapping) or any(type(key) is not str or not key for key in arrays):
        raise ValueError("arrays must map nonempty string names to ndarrays")
    metadata_data = _json_dict(metadata, "metadata")
    bindings_data = _json_dict(bindings, "bindings")
    copied = {name: _copy_array(value) for name, value in sorted(arrays.items())}
    descriptions = {name: _array_record(value, f"array_{index:06d}")
                    for index, (name, value) in enumerate(copied.items())}
    manifest = {"schema": SCHEMA, "status": "ok", "bindings": _decode(bindings_data),
                "bindings_sha256": _sha(bindings_data), "arrays": descriptions,
                "metadata_canonical_sha256": _sha(metadata_data)}
    root.mkdir(parents=True, exist_ok=False)
    _write_exclusive(root / METADATA_NAME, metadata_data)
    with (root / ARRAYS_NAME).open("xb") as stream:
        np.savez_compressed(stream, **{descriptions[name]["storage_key"]: value
                                      for name, value in copied.items()})
        stream.flush()
        os.fsync(stream.fileno())
    manifest["metadata"] = _file_record(root / METADATA_NAME)
    manifest["payload"] = _file_record(root / ARRAYS_NAME)
    # Refuse a success seal if this dtype or platform fails a lossless NPY roundtrip.
    _payload(root, manifest, retain_arrays=False)
    _write_exclusive(root / MANIFEST_NAME, _canonical(manifest))
    return _decode(_canonical(manifest))


def read_checkpoint(root, *, expected_bindings):
    """Verify a completed snapshot and return detached arrays, metadata, manifest.

    Dictionary key order is immaterial for bindings; value types and list order
    are exact (``True`` does not match ``1``). Returning writable copies prevents a
    later mutation from changing another load or the caller's saved input arrays.
    """
    root = Path(root)
    expected = _json_dict(expected_bindings, "expected_bindings")
    manifest_path = _regular_path(root, MANIFEST_NAME)
    data = manifest_path.read_bytes()
    manifest = _decode(data)
    expected_fields = {"schema", "status", "bindings", "bindings_sha256", "arrays",
                       "metadata_canonical_sha256", "metadata", "payload"}
    if type(manifest) is not dict or set(manifest) != expected_fields:
        raise ValueError("invalid checkpoint manifest fields")
    if manifest["schema"] != SCHEMA or manifest["status"] != "ok":
        raise ValueError("checkpoint has no recognized success seal")
    if _canonical(manifest) != data:
        raise ValueError("manifest is not canonical JSON")
    actual = _json_dict(manifest["bindings"], "bindings")
    if _sha(actual) != manifest["bindings_sha256"]:
        raise ValueError("bindings checksum mismatch")
    if actual != expected:
        raise ValueError("checkpoint bindings differ from expected bindings")
    arrays, metadata = _payload(root, manifest)
    if manifest_path.read_bytes() != data:
        raise ValueError("checkpoint manifest changed during reading")
    return arrays, metadata, manifest

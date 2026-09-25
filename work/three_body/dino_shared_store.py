"""Lossless NPZ/JSON storage for DINO correspondences with shared crop features.

No inference or geometry calculation. Pair resolution uses saved distances/NN
and indexed saved coordinates; it never repeats cosine matmul or projection.
All old logical arrays and scalar metadata are authenticated, including NaN bits.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path

import dino_mask_partial_correspondence as partial
import numpy as np

VERSION = "dino_shared_store_v1"
GEOMETRY_KEYS = (
    "token_index",
    "token_center_crop_xy",
    "token_center_native_xy",
    "token_native_xyxy",
    "token_footprint_pixels",
    "token_core_pixels",
    "token_eligible",
)
SIDE_KEYS = (
    "eligible",
    "coordinate_finite",
    "usable",
    "normalization__normalized",
    "normalization__input_finite",
    "normalization__nonzero_finite",
    "normalization__valid",
    "normalization__norm",
)
PAIR_KEYS = ("distances", "forward_nn", "reverse_nn")
DERIVED_KEYS = (
    "finite_pair",
    "mutual_nn",
    "left_minimum",
    "right_minimum",
    "correspondences__pairs",
    "correspondences__distances",
    "correspondences__left_crop_xy",
    "correspondences__right_crop_xy",
    "correspondences__left_native_xy",
    "correspondences__right_native_xy",
    "correspondences__forward",
    "correspondences__reverse",
    "correspondences__mutual",
)
LOGICAL_KEYS = set(PAIR_KEYS + DERIVED_KEYS) | {
    s + "__" + k for s in ("left", "right") for k in SIDE_KEYS
}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def clone_json(value):
    return json.loads(canonical(value))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(array):
    require(
        isinstance(array, np.ndarray) and not array.dtype.hasobject,
        "plain nonobject array required",
    )
    return dict(
        dtype=array.dtype.str,
        shape=list(array.shape),
        sha256=hashlib.sha256(array.tobytes()).hexdigest(),
    )


def signatures(arrays):
    return {k: signature(v) for k, v in arrays.items()}


def same(left, right):
    return signature(left) == signature(right)


def copy_arrays(arrays):
    return {
        k: np.frombuffer(v.tobytes(), dtype=v.dtype).reshape(v.shape).copy()
        for k, v in arrays.items()
    }


def metadata_arrays(metadata):
    refs = {}

    def walk(value):
        if isinstance(value, dict):
            if "array" in value:
                require(
                    set(value) == {"array", "dtype", "shape", "sha256"},
                    "array reference schema changed",
                )
                key = value["array"]
                require(isinstance(key, str) and key not in refs, "duplicate logical reference")
                refs[key] = {k: value[k] for k in ("dtype", "shape", "sha256")}
            else:
                for x in value.values():
                    walk(x)
        elif isinstance(value, list):
            for x in value:
                walk(x)

    walk(metadata)
    return refs


def make_features(tokens, geometry):
    require(set(geometry) == set(GEOMETRY_KEYS), "complete shared geometry required")
    require(
        tokens.dtype == np.float32 and tokens.ndim == 2,
        "original LayerNorm float32 token rows required",
    )
    before = signature(tokens), signatures(geometry)
    eligible, _, _, finite = partial._geometry(geometry, len(tokens))
    normalization = partial.normalize_tokens(tokens)
    result = dict(
        layernorm_tokens=tokens.copy(),
        eligible=eligible.copy(),
        coordinate_finite=finite.copy(),
        usable=eligible & finite & normalization["valid"],
    )
    result.update({"normalization__" + k: v.copy() for k, v in normalization.items()})
    require(before == (signature(tokens), signatures(geometry)), "normalization mutated source")
    return result


def derived_arrays(payload, left, right, left_geometry, right_geometry):
    distances, forward, reverse = (payload[k] for k in PAIR_KEYS)
    shape = (len(left["eligible"]), len(right["eligible"]))
    require(
        distances.dtype == np.float64
        and distances.shape == shape
        and forward.dtype == reverse.dtype == bool
        and forward.shape == reverse.shape == shape,
        "full pair distance/NN shape changed",
    )
    finite = np.isfinite(distances)
    require(not ((forward | reverse) & ~finite).any(), "NN cannot name invalid distance")
    mutual = forward & reverse
    pairs = np.argwhere(forward | reverse)
    i, j = pairs[:, 0], pairs[:, 1]
    left_min, right_min = np.full(shape[0], np.nan), np.full(shape[1], np.nan)
    if shape[1]:
        good = finite.any(1)
        left_min[good] = np.where(finite[good], distances[good], np.inf).min(1)
    if shape[0]:
        good = finite.any(0)
        right_min[good] = np.where(finite[:, good], distances[:, good], np.inf).min(0)
    for name, value in (("left_minimum", left_min), ("right_minimum", right_min)):
        if name in payload:
            require(
                payload[name].dtype == value.dtype
                and payload[name].shape == value.shape
                and np.array_equal(payload[name], value, equal_nan=True),
                "minimum fallback not same values",
            )
    left_min = payload.get("left_minimum", left_min).copy()
    right_min = payload.get("right_minimum", right_min).copy()
    require(
        np.array_equal(forward, finite & (distances == left_min[:, None]))
        and np.array_equal(reverse, finite & (distances == right_min[None, :])),
        "all exact NN ties required",
    )
    result = dict(
        distances=distances.copy(),
        forward_nn=forward.copy(),
        reverse_nn=reverse.copy(),
        finite_pair=finite,
        mutual_nn=mutual,
        left_minimum=left_min,
        right_minimum=right_min,
    )
    result.update(
        {
            side + "__" + key: values[key].copy()
            for side, values in (("left", left), ("right", right))
            for key in SIDE_KEYS
        }
    )
    correspondence = dict(
        pairs=pairs,
        distances=distances[i, j],
        left_crop_xy=left_geometry["token_center_crop_xy"][i],
        right_crop_xy=right_geometry["token_center_crop_xy"][j],
        left_native_xy=left_geometry["token_center_native_xy"][i],
        right_native_xy=right_geometry["token_center_native_xy"][j],
        forward=forward[i, j],
        reverse=reverse[i, j],
        mutual=mutual[i, j],
    )
    result.update({"correspondences__" + k: v for k, v in correspondence.items()})
    return result


def split_pair(arrays, metadata, left, right, left_geometry, right_geometry):
    require(set(arrays) == LOGICAL_KEYS, "full original32 logical arrays required")
    expected = signatures(arrays)
    require(metadata_arrays(metadata) == expected, "logical arrays/metadata SHA differs")
    payload = {k: arrays[k].copy() for k in PAIR_KEYS}
    derived = derived_arrays(payload, left, right, left_geometry, right_geometry)
    # Tiny minima are the only fallback: IEEE signed-zero/NaN bits are retained.
    for name in ("left_minimum", "right_minimum"):
        if not same(derived[name], arrays[name]):
            payload[name] = arrays[name].copy()
    derived = derived_arrays(payload, left, right, left_geometry, right_geometry)
    require(signatures(derived) == expected, "logical reconstruction differs from original bytes")
    return payload, dict(logical_arrays=expected, metadata=clone_json(metadata))


def expand_pair(payload, spec, left, right, left_geometry, right_geometry):
    require(
        set(payload) <= set(PAIR_KEYS) | {"left_minimum", "right_minimum"}
        and set(PAIR_KEYS) <= set(payload),
        "unexpected pair payload member",
    )
    arrays = derived_arrays(payload, left, right, left_geometry, right_geometry)
    require(
        signatures(arrays) == spec["logical_arrays"] == metadata_arrays(spec["metadata"]),
        "resolved logical array bytes differ",
    )
    return copy_arrays(arrays), clone_json(spec["metadata"])


def identifier(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_]{1,180}", value),
        "safe unique store identifier required",
    )
    return value


def read_npz(root, ref):
    path = (root / ref["path"]).resolve(strict=True)
    require(
        path.is_relative_to(root)
        and path.stat().st_size == ref["size_bytes"]
        and sha(path) == ref["sha256"],
        "shared payload path/size/SHA differs",
    )
    with np.load(path, allow_pickle=False) as z:
        require(len(z.files) == len(set(z.files)), "duplicate stored NPZ member")
        arrays = {k: z[k].copy() for k in z.files}
    require(
        signatures(arrays) == ref["arrays"] and sha(path) == ref["sha256"],
        "stored arrays changed while reading",
    )
    return arrays


class StoreWriter:
    def __init__(self, root, *, bindings):
        self.root = Path(root).resolve()
        require(not self.root.exists(), "new store root required")
        self.bindings = clone_json(bindings)
        self.root.mkdir(parents=True)
        self.index = dict(schema=VERSION, bindings=self.bindings, crops={}, features={}, pairs={})
        self.outputs = {}
        self.closed = False
        for name in ("geometry", "features", "pairs"):
            (self.root / name).mkdir()
        self.cache = OrderedDict()

    def _write(self, name, arrays):
        require(not self.closed, "store already sealed")
        path = self.root / name
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        ref = dict(
            path=name, sha256=sha(path), size_bytes=path.stat().st_size, arrays=signatures(arrays)
        )
        self.outputs[name] = ref
        return ref

    def _read(self, ref):
        key = ref["path"]
        if key not in self.cache:
            self.cache[key] = read_npz(self.root, ref)
        self.cache.move_to_end(key)
        if len(self.cache) > 12:
            self.cache.popitem(last=False)
        return self.cache[key]

    def add_crop(self, crop_id, geometry):
        identifier(crop_id)
        require(crop_id not in self.index["crops"], "duplicate crop")
        require(set(geometry) == set(GEOMETRY_KEYS), "full shared geometry required")
        self.index["crops"][crop_id] = self._write(f"geometry/{crop_id}.npz", copy_arrays(geometry))

    def add_feature(self, feature_id, crop_id, tokens):
        identifier(feature_id)
        require(feature_id not in self.index["features"], "duplicate feature")
        require(crop_id in self.index["crops"], "unknown crop reference")
        geometry = self._read(self.index["crops"][crop_id])
        features = make_features(tokens, geometry)
        self.index["features"][feature_id] = dict(
            crop_id=crop_id, **self._write(f"features/{feature_id}.npz", features)
        )

    def add_pair(self, pair_id, left_id, right_id, arrays, metadata):
        identifier(pair_id)
        require(pair_id not in self.index["pairs"], "duplicate pair")
        require(
            left_id in self.index["features"] and right_id in self.index["features"],
            "missing feature reference",
        )
        lr, rr = (self.index["features"][k] for k in (left_id, right_id))
        lg, rg = (self._read(self.index["crops"][r["crop_id"]]) for r in (lr, rr))
        payload, spec = split_pair(arrays, metadata, self._read(lr), self._read(rr), lg, rg)
        ref = self._write(f"pairs/{pair_id}.npz", payload)
        self.index["pairs"][pair_id] = dict(left_id=left_id, right_id=right_id, **ref, **spec)

    def finish(self):
        require(not self.closed, "store already sealed")
        for ref in self.outputs.values():
            require(sha(self.root / ref["path"]) == ref["sha256"], "output changed before commit")
        self.index["status"] = "ok"
        self.index["outputs"] = self.outputs
        path = self.root / "store.manifest.json"
        with path.open("xb") as stream:
            stream.write(canonical(self.index))
        self.closed = True
        self.cache.clear()
        return clone_json(self.index)


class StoreReader:
    def __init__(self, root, *, expected_bindings, expected_manifest_sha256):
        self.root = Path(root).resolve(strict=True)
        path = self.root / "store.manifest.json"
        require(sha(path) == expected_manifest_sha256, "store manifest SHA differs")
        self.index = json.loads(path.read_bytes())
        require(
            sha(path) == expected_manifest_sha256
            and self.index.get("schema") == VERSION
            and self.index.get("status") == "ok"
            and canonical(self.index["bindings"]) == canonical(expected_bindings),
            "store schema/bindings differ",
        )
        require(
            set(self.index["outputs"])
            == {
                r["path"]
                for group in ("crops", "features", "pairs")
                for r in self.index[group].values()
            },
            "store payload membership differs",
        )
        self.manifest_sha256 = expected_manifest_sha256

    def read_pair(self, pair_id):
        identifier(pair_id)
        ref = self.index["pairs"][pair_id]
        lr, rr = (self.index["features"][ref[k]] for k in ("left_id", "right_id"))
        left, right = (read_npz(self.root, r) for r in (lr, rr))
        lg, rg = (read_npz(self.root, self.index["crops"][r["crop_id"]]) for r in (lr, rr))
        result = expand_pair(read_npz(self.root, ref), ref, left, right, lg, rg)
        return result

    def read_feature(self, feature_id):
        identifier(feature_id)
        return read_npz(self.root, self.index["features"][feature_id])

    def verify_all(self):
        require(
            sha(self.root / "store.manifest.json") == self.manifest_sha256, "store manifest changed"
        )
        expected = {self.root / "store.manifest.json"}
        for ref in self.index["outputs"].values():
            path = (self.root / ref["path"]).resolve()
            expected.add(path)
            require(
                path.is_relative_to(self.root)
                and path.stat().st_size == ref["size_bytes"]
                and sha(path) == ref["sha256"],
                "stored output SHA/size differs",
            )
        require(
            {p.resolve() for p in self.root.rglob("*") if p.is_file()} == expected,
            "unregistered/missing store files",
        )
        return dict(files=len(expected), bytes=sum(p.stat().st_size for p in expected))

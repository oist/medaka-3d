"""Reuse successful semantic validation of identical packed-mask bytes.

An explicit local performance cache, not an authentication or ID-truth system.
The caller still validates each detector attempt and all original input guards.
Each lookup hashes the complete NPZ, manifest and successful detector seal. Only
an exact scope, validator/decode/receipt source and runtime match can be reused.
Malformed or incomplete entries are retained and cause full validation again.
No frozen module globals or tracker bindings are substituted.
Receipts require a trusted local writer and installed runtime; they must not be
accepted as external attestation. Native wheel/DLL bytes are not authenticated.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import uuid
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
import chunk_masks  # noqa: E402
import prepare_sequence  # noqa: E402

SCHEMA = "three_body_packed_mask_validation_receipt_v1"
SCOPE_KEYS = {
    "view", "frame_range", "image_shape_hw", "manifest_path", "manifest_sha256",
    "seal_path", "seal_sha256", "detector_binding", "expected_npz_sha256",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def checked_file(path):
    path = Path(path)
    require(not path.is_symlink(), "receipt input cannot be a symlink")
    path = path.resolve(strict=True)
    require(path.is_file() and path.is_relative_to(REPO), "foreign receipt input")
    return path


class ReceiptStore:
    """One explicit verification context; use a fresh object for each operation."""

    def __init__(self, root):
        root = Path(root)
        require(not root.is_symlink(), "receipt root cannot be a symlink")
        self.root = root.resolve()
        require(self.root.is_relative_to(REPO), "foreign receipt root")
        self.root.mkdir(parents=True, exist_ok=True)
        source_paths = [Path(__file__), Path(chunk_masks.__file__), Path(prepare_sequence.__file__)]
        self.code_hashes = {str(checked_file(p)): sha(p) for p in source_paths}
        self.validator_binding = dict(
            source_sha256=self.code_hashes.copy(),
            python=platform.python_version(), implementation=platform.python_implementation(),
            numpy=np.__version__,
            schema_dtypes={k: v.str for k, v in chunk_masks.SCHEMA_DTYPES.items()},
            mask_bitorder="little",
        )
        self._hashes = self.code_hashes.copy()
        self.uses = []

    def _guard(self, hashes):
        for path, expected in hashes.items():
            require(sha(checked_file(path)) == expected, f"receipt evidence changed: {path}")

    def _request(self, path, scope):
        require(isinstance(scope, dict) and set(scope) == SCOPE_KEYS, "wrong receipt scope keys")
        scope = json.loads(canonical(scope))
        require(scope["view"] in ("top_cam", "side1_cam", "side2_cam"), "wrong receipt view")
        frames = scope["frame_range"]
        shape = scope["image_shape_hw"]
        require(
            isinstance(frames, list) and len(frames) == 2
            and all(type(x) is int for x in frames) and 0 <= frames[0] < frames[1],
            "wrong receipt native range",
        )
        require(
            isinstance(shape, list) and len(shape) == 2
            and all(type(x) is int and x > 0 for x in shape), "wrong receipt image shape",
        )
        path = checked_file(path)
        files = {str(path): scope["expected_npz_sha256"]}
        for key in ("manifest", "seal"):
            scope[key + "_path"] = str(checked_file(scope[key + "_path"]))
            files[scope[key + "_path"]] = scope[key + "_sha256"]
        self._guard({**self.code_hashes, **files})
        manifest, seal = read(scope["manifest_path"]), read(scope["seal_path"])
        require(
            seal.get("status") == "ok" and seal.get("stage") == "prepare_independent_mask_chunk",
            "receipt requires successful detector seal",
        )
        require(
            canonical(manifest.get("binding")) == canonical(scope["detector_binding"])
            == canonical(seal["params"].get("binding")), "receipt detector binding differs",
        )
        require(
            manifest["source_interval_half_open"] == frames
            and manifest["image_shape_hw"] == shape, "receipt manifest range/shape differs",
        )
        view = scope["view"]
        output = seal["outputs"][view + "_masks"]
        require(
            checked_file(output["path"]) == path
            and output["sha256"] == scope["expected_npz_sha256"]
            and checked_file(Path(scope["manifest_path"]).parent / manifest["masks"][view]["path"])
            == path, "receipt mask path/SHA differs from detector evidence",
        )
        request = dict(
            schema=SCHEMA, npz_path=str(path), npz_bytes=path.stat().st_size,
            scope=scope, validator_binding=self.validator_binding,
        )
        return path, request, files

    def validate(self, path, *, scope):
        started = time.perf_counter()
        path, request, inputs = self._request(path, scope)
        scope = request["scope"]
        key = hashlib.sha256(canonical(request).encode()).hexdigest()
        folder = self.root / key
        require(not folder.is_symlink(), "receipt key directory cannot be a symlink")
        rejected = []
        receipt_path = None
        for commit_path in sorted(folder.glob("*.success.json")):
            try:
                commit_path = checked_file(commit_path)
                commit_digest = sha(commit_path)
                commit = read(commit_path)
                require(set(commit) == {"schema", "status", "receipt", "sha256"}, "bad commit keys")
                require(
                    commit["schema"] == SCHEMA and commit["status"] == "ok", "bad commit status",
                )
                name = commit["receipt"]
                require(isinstance(name, str) and Path(name).name == name, "foreign receipt name")
                candidate = checked_file(folder / name)
                require(sha(candidate) == commit["sha256"], "receipt body SHA differs")
                receipt = read(candidate)
                require(set(receipt) == {"request", "result"}, "bad receipt body keys")
                require(
                    canonical(receipt["request"]) == canonical(request),
                    "different receipt request",
                )
                expected = dict(
                    status="ok", validator="chunk_masks.validate_packed_masks",
                    frame_count=scope["frame_range"][1] - scope["frame_range"][0],
                    validated_fields=sorted(chunk_masks.SCHEMA_DTYPES),
                    every_decoded_frame_sha_verified=True,
                )
                require(canonical(receipt["result"]) == canonical(expected), "bad receipt result")
                receipt_path = candidate
                break
            except (OSError, ValueError, KeyError, TypeError) as exc:
                rejected.append(dict(path=str(commit_path), reason=f"{type(exc).__name__}: {exc}"))
        hit = receipt_path is not None
        if not hit:
            with np.load(path, allow_pickle=False) as archive:
                require(len(archive.files) == len(set(archive.files)), "duplicate mask arrays")
                packed = {name: archive[name] for name in archive.files}
            chunk_masks.validate_packed_masks(packed)
            start, stop = scope["frame_range"]
            require(
                np.array_equal(packed["frame_idx"], np.arange(start, stop, dtype=np.int64))
                and np.array_equal(packed["image_shape_hw"], scope["image_shape_hw"]),
                "validated mask native range/shape differs",
            )
            self._guard({**self.code_hashes, **inputs})
            receipt = dict(request=request, result=dict(
                status="ok", validator="chunk_masks.validate_packed_masks",
                frame_count=stop - start,
                validated_fields=sorted(chunk_masks.SCHEMA_DTYPES),
                every_decoded_frame_sha_verified=True,
            ))
            folder.mkdir(exist_ok=True)
            name = uuid.uuid4().hex
            receipt_path = folder / (name + ".receipt.json")
            with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(canonical(receipt) + "\n")
            commit_path = folder / (name + ".success.json")
            # Success is written last. A killed/incomplete write is never a cache hit.
            with commit_path.open("x", encoding="utf-8", newline="\n") as stream:
                commit = dict(
                    schema=SCHEMA, status="ok", receipt=receipt_path.name,
                    sha256=sha(receipt_path),
                )
                serialized_commit = canonical(commit) + "\n"
                stream.write(serialized_commit)
            commit_digest = hashlib.sha256(serialized_commit.encode("utf-8")).hexdigest()
        evidence = {
            **inputs, str(receipt_path): commit["sha256"], str(commit_path): commit_digest,
        }
        self._guard({**self.code_hashes, **evidence})
        for name, digest in evidence.items():
            require(
                name not in self._hashes or self._hashes[name] == digest,
                "evidence changed within run",
            )
        self._hashes.update(evidence)
        usage = dict(
            key=key, cache_hit=hit, receipt_path=str(receipt_path),
            receipt_sha256=evidence[str(receipt_path)], commit_path=str(commit_path),
            commit_sha256=evidence[str(commit_path)], npz_path=str(path),
            scope=json.loads(canonical(scope)), rejected_entries=rejected,
            elapsed_seconds=time.perf_counter() - started,
        )
        self.uses.append(json.loads(canonical(usage)))
        return usage

    def final_guard(self):
        self._guard(self._hashes)

    @property
    def evidence_hashes(self):
        return self._hashes.copy()

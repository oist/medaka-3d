"""Owned bounded lossless spool for the fixed final native-sample callback.

No video/chain reader is invoked here. A success manifest certifies saved samples,
not the final audit. Failed output directories are retained and never reused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import long_final_native_samples as native
import numpy as np

VERSION = "long_final_native_spool_v1"
VIEWS, FIRST, STOP, CHUNK = tuple(native.VIEWS), native.FIRST, native.STOP, native.CHUNK
require = native.require


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def signature(a):
    require(isinstance(a, np.ndarray) and not a.dtype.hasobject, "non-object ndarray required")
    return dict(dtype=str(a.dtype), shape=list(a.shape), sha256=native.array_sha(a))


def signatures(arrays):
    return {k: signature(v) for k, v in arrays.items()}


def boundaries():
    return tuple(range(FIRST + CHUNK, STOP, CHUNK))


def retained_frames():
    return tuple(sorted({n for b in boundaries() for n in (b - 1, b)} | {STOP - 1}))


def validate_sample(row, data):
    require(
        set(data) == {"image_bgr", "masks", "detections", "packed"}, "sample payload fields differ"
    )
    image, detections, packed = (data[k] for k in ("image_bgr", "detections", "packed"))
    require(
        image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3,
        "native uint8 BGR required",
    )
    require(set(packed) == set(native.SCHEMA_DTYPES), "all13 packed arrays required")
    require(signatures(packed) == row["selected_packed_arrays"], "packed callback SHA differs")
    n = row["frame_idx"]
    require(
        type(n) is int and FIRST <= n < STOP and row["video_offset"] == n, "native offset differs"
    )
    require(
        row["chunk_index"] == (n - FIRST) // CHUNK
        and row["chunk_frame_offset"] == (n - FIRST) % CHUNK,
        "chunk offset differs",
    )
    require(np.array_equal(packed["image_shape_hw"], image.shape[:2]), "image/packed shape differs")
    require(packed["frame_idx"].shape == (1,), "packed spool frame must be singleton")
    masks = data["masks"]
    available = masks is not None
    require(
        row["mask_available"] is available
        and row["mask_count"] == (len(masks) if available else None),
        "None/empty mask availability differs",
    )
    selected, decoded = native.select_packed_frame(packed, 0, n, len(masks) if available else 0)
    require(
        signatures(selected) == signatures(packed), "packed rebase was not already native-singleton"
    )
    require((decoded is None) == (masks is None), "packed/decoded presence differs")
    if available:
        require(
            masks.dtype == bool
            and masks.shape == decoded.shape
            and masks.tobytes() == decoded.tobytes(),
            "callback decoded mask bytes differ",
        )
    require(
        row["mask_decoded_sha256"] == (native.array_sha(decoded) if available else None),
        "decoded mask SHA differs",
    )
    require(
        row["packed_frame_mask_sha256"] == bytes(packed["frame_mask_sha256"][0]).decode(),
        "packed mask hash string differs",
    )
    require(native.array_sha(image) == row["raw_bgr_sha256"], "raw BGR SHA differs")
    require(
        detections.ndim == 3
        and detections.shape[1:] == (3, 2)
        and detections.dtype.kind == "f"
        and not np.isinf(detections).any(),
        "SL shape/dtype differs",
    )
    require(
        signature(detections)
        == {
            "shape": row["detection_shape"],
            "dtype": row["detection_dtype"],
            "sha256": row["detection_sha256"],
        },
        "SL signature differs",
    )


class _SpoolWriter:
    """Generic small-fixture engine; NativeSpool fixes the actual sampling scope."""

    def __init__(self, root, bindings, samples, retained, views):
        self.root = Path(root).resolve()
        require(not self.root.exists(), "new spool directory required")
        self.samples, self.retained, self.views = tuple(samples), tuple(retained), tuple(views)
        require(
            self.samples == tuple(sorted(set(self.samples)))
            and set(self.retained) <= set(self.samples),
            "sample/retained order differs",
        )
        require(
            self.retained == tuple(sorted(set(self.retained))) and self.views == VIEWS,
            "retained/view order differs",
        )
        self.bindings = json.loads(canonical(bindings))
        self.expected = [(n, v) for v in self.views for n in self.samples]
        self.rows, self.outputs, self.closed = [], {}, False
        self.root.mkdir(parents=True)
        (self.root / "samples").mkdir()
        self.registration = dict(
            schema=VERSION,
            bindings=self.bindings,
            samples=list(self.samples),
            retained_frames=list(self.retained),
            views=list(self.views),
            failure_policy="retain_partial_directory_no_retry_no_success_manifest",
            packed13_preserved=True,
            decoded_masks_stored=False,
            final_audit_complete=False,
        )
        self._json("registration.json", self.registration)
        with (self.root / "records.jsonl").open("xb"):
            pass

    def _json(self, name, value):
        with (self.root / name).open("xb") as f:
            f.write(canonical(value))

    def _record(self, path):
        return dict(
            path=path.relative_to(self.root).as_posix(),
            sha256=sha(path),
            size_bytes=path.stat().st_size,
        )

    def abort(self, error):
        if self.closed:
            return
        self.closed = True
        path = self.root / "failure.json"
        if not path.exists():
            self._json(
                "failure.json",
                dict(
                    schema=VERSION,
                    status="failed",
                    accepted_callbacks=len(self.rows),
                    error_type=type(error).__name__,
                    error=str(error),
                    final_audit_complete=False,
                ),
            )

    def __call__(self, row, data):
        require(not self.closed, "closed spool cannot accept callbacks")
        try:
            require(len(self.rows) < len(self.expected), "extra callback")
            require(
                (row["frame_idx"], row["view"]) == self.expected[len(self.rows)],
                "callback native/view order differs",
            )
            validate_sample(row, data)
            flat = {
                "image_bgr": data["image_bgr"],
                "detections": data["detections"],
                **{"packed__" + k: v for k, v in data["packed"].items()},
            }
            before = signatures(flat)
            saved = json.loads(canonical(row))
            saved["spooled"] = row["frame_idx"] in self.retained
            if saved["spooled"]:
                path = self.root / "samples" / f"f{row['frame_idx']:06d}_{row['view']}.npz"
                with path.open("xb") as f:
                    np.savez_compressed(f, **flat)
                with np.load(path, allow_pickle=False) as a:
                    require(
                        set(a.files) == set(flat) and len(a.files) == 15,
                        "spool archive membership differs",
                    )
                    require(
                        {k: signature(a[k]) for k in a.files} == before,
                        "spool write/read bytes differ",
                    )
                saved["payload"] = dict(**self._record(path), arrays=before)
                self.outputs[saved["payload"]["path"]] = saved["payload"]
            require(signatures(flat) == before, "spool mutated callback arrays")
            validate_sample(row, data)
            with (self.root / "records.jsonl").open("ab") as f:
                f.write(canonical(saved) + b"\n")
                f.flush()
            self.rows.append(saved)
        except BaseException as exc:
            self.abort(exc)
            raise

    def finish(self):
        require(not self.closed, "closed spool cannot be finished again")
        try:
            require(len(self.rows) == len(self.expected), "incomplete spool callbacks")
            require(
                sum(r["spooled"] for r in self.rows) == len(self.retained) * len(self.views),
                "spool retained count differs",
            )
            require(
                (self.root / "registration.json").read_bytes() == canonical(self.registration),
                "spool registration changed before commit",
            )
            expected_log = hashlib.sha256()
            for row in self.rows:
                expected_log.update(canonical(row) + b"\n")
            require(
                sha(self.root / "records.jsonl") == expected_log.hexdigest(),
                "spool callback log changed before commit",
            )
            for name in ("registration.json", "records.jsonl"):
                self.outputs[name] = self._record(self.root / name)
            for ref in self.outputs.values():
                require(sha(self.root / ref["path"]) == ref["sha256"], "spool file changed")
            require(
                {p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()}
                == set(self.outputs),
                "unregistered spool files",
            )
            result = dict(
                **self.registration,
                status="ok",
                callback_records=len(self.rows),
                saved_views=len(self.retained) * 3,
                outputs=self.outputs,
                saved_bytes=sum(r["size_bytes"] for r in self.outputs.values()),
            )
            self._json("spool.manifest.json", result)
            self.closed = True
            return result
        except BaseException as exc:
            self.abort(exc)
            raise


class NativeSpool(_SpoolWriter):
    def __init__(self, root, bindings):
        super().__init__(
            root, bindings, native.fixed_sample_frames().tolist(), retained_frames(), VIEWS
        )


class SpoolReader:
    def __init__(self, root, *, expected_manifest_sha256, expected_bindings):
        self.root = Path(root).resolve(strict=True)
        path = self.root / "spool.manifest.json"
        require(sha(path) == expected_manifest_sha256, "spool manifest SHA differs")
        self.manifest = json.loads(path.read_bytes())
        m = self.manifest
        require(
            m["schema"] == VERSION
            and m["status"] == "ok"
            and m["final_audit_complete"] is False
            and canonical(m["bindings"]) == canonical(expected_bindings),
            "spool schema/binding differs",
        )
        self.manifest_sha256 = expected_manifest_sha256
        for name in ("registration.json", "records.jsonl"):
            self._check(m["outputs"][name])
        registration = json.loads((self.root / "registration.json").read_bytes())
        require(all(m[k] == v for k, v in registration.items()), "spool registration differs")
        self.rows = [
            json.loads(line) for line in (self.root / "records.jsonl").read_bytes().splitlines()
        ]
        require(
            [(r["frame_idx"], r["view"]) for r in self.rows]
            == [(n, v) for v in m["views"] for n in m["samples"]],
            "spool record order differs",
        )
        require(len(self.rows) == m["callback_records"], "spool callback count differs")
        self.index = {(r["frame_idx"], r["view"]): r for r in self.rows if r["spooled"]}
        require(
            set(self.index) == {(n, v) for n in m["retained_frames"] for v in m["views"]}
            and len(self.index) == m["saved_views"],
            "spool retained membership differs",
        )
        for r in self.rows:
            if r["spooled"]:
                require(
                    m["outputs"][r["payload"]["path"]] == r["payload"],
                    "spool row/payload binding differs",
                )
        require(
            set(m["outputs"])
            == {"registration.json", "records.jsonl"}
            | {r["payload"]["path"] for r in self.index.values()},
            "spool output membership differs",
        )
        self.verify_all()

    def _check(self, ref):
        path = native.checked_relative(self.root, ref["path"])
        require(
            path.stat().st_size == ref["size_bytes"] and sha(path) == ref["sha256"],
            "spool payload SHA/size differs",
        )
        return path

    def read_view(self, n, view):
        row = self.index[n, view]
        ref = row["payload"]
        path = self._check(ref)
        with np.load(path, allow_pickle=False) as a:
            require(
                len(a.files) == 15 and set(a.files) == set(ref["arrays"]),
                "spool array membership differs",
            )
            arrays = {k: a[k].copy() for k in a.files}
        require(signatures(arrays) == ref["arrays"], "spool array dtype/shape/bytes differ")
        packed = {
            k.removeprefix("packed__"): v for k, v in arrays.items() if k.startswith("packed__")
        }
        _, masks = native.select_packed_frame(packed, 0, n, row["mask_count"] or 0)
        data = dict(
            image_bgr=arrays["image_bgr"],
            detections=arrays["detections"],
            packed=packed,
            masks=masks,
        )
        validate_sample(row, data)
        self._check(ref)
        return data

    def verify_all(self):
        require(
            sha(self.root / "spool.manifest.json") == self.manifest_sha256, "spool manifest changed"
        )
        for ref in self.manifest["outputs"].values():
            self._check(ref)
        expected = set(self.manifest["outputs"]) | {"spool.manifest.json"}
        require(
            {p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()}
            == expected,
            "spool file membership changed",
        )

    def require_final_scope(self):
        require(
            self.manifest["samples"] == native.fixed_sample_frames().tolist()
            and self.manifest["retained_frames"] == list(retained_frames())
            and self.manifest["views"] == list(VIEWS)
            and self.manifest["callback_records"] == 1629
            and self.manifest["saved_views"] == 1077,
            "fixed543/359/1629/1077 final spool required",
        )

"""Completed D2 assembly evidence and bounded packed-array equality, CPU only.

The caller supplies a verified completed tracking chain, its consumed records and
an evidence object with add(path, expected=None)/final_guard(). No tracker,
detector, video decoder or full-mask decoder is called. Completion here proves
the D2 assembly only; the caller still owns final guards, output seals, support,
flow, previous observations and the final technical audit.

NPZ comparison streams the thirteen NPY members in C order. Selected chunks and
member-sized comparison buffers are needed, never another global
concatenation. Only the two offset fields are rebased, exactly as the frozen
concatenate_packed_masks. Decoded-mask correctness is inherited from the bound
producer/consumed evidence; this is an independent byte-level assembly check.
"""

from __future__ import annotations

import copy
import json
import math
import os
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import audit_chunked_source_smoke as frozen
import numpy as np
from chunk_masks import SCHEMA_DTYPES
from prepare_chunked_source_masks import exact_contract, validate_protocol
from source_sequence import validate_source_manifest

REPO = Path(__file__).resolve().parents[2]
FIRST, STOP, CHUNK, COUNT = 49800, 348600, 1660, 180
VIEWS = tuple(frozen.VIEWS)
ENTRY_KEYS = ("start", "stop", "attempt_path", "manifest_sha256", "provenance_sha256")
PINS = {
    "work/three_body/audit_chunked_source_smoke.py": (
        "c78a407f7dab1ce18294d483f1a3dee75f4b7ee6edce667ec3a3d57d467fff24"
    ),
    "labs/20260909_three_body/chunk_masks.py": (
        "9e08248ea593b4ccb105a6ff351fd85c6db24aa6754413b2df766404cdfe4b81"
    ),
    "labs/20260909_three_body/prepare_chunked_source_masks.py": (
        "a081c47e6ed67b4b61c02635e00c97c967803b3528fe107f016ab23b5ae34f16"
    ),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def equal(actual, expected, message):
    require(exact_contract(actual, expected), message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(root, text):
    require(isinstance(text, str), "relative path must be text")
    item = PurePosixPath(text.replace("\\", "/"))
    require(
        item.parts and not item.is_absolute() and ".." not in item.parts and ":" not in text,
        "unsafe D2 relative path",
    )
    root = Path(root).resolve(strict=True)
    path = (root / item).resolve(strict=True)
    require(path.is_relative_to(root), "D2 path escaped root")
    return path


class _Hashes:
    """Adapt the frozen seal verifier to the caller's inode-aware evidence."""

    def __init__(self, evidence):
        self.evidence = evidence

    def digest(self, path):
        return self.evidence.add(path)

    def check(self, path, expected):
        return self.evidence.add(path, expected)


def verified_seal(path, stage, evidence, *, required=()):
    record = frozen.verify_seal(path, _Hashes(evidence), check_inputs=True, required=required)
    require(
        record.get("stage") == stage and record.get("finished_at"),
        "D2 successful stage/finish differs",
    )
    for group in ("inputs", "outputs"):
        for item in record[group].values():
            require(
                type(item.get("size")) is int and Path(item["path"]).stat().st_size == item["size"],
                "D2 sealed file size differs",
            )
    return record


def named(record, name, path):
    item = record["outputs"].get(name, {})
    require(
        Path(item.get("path", "")).resolve() == Path(path).resolve(),
        "D2 named output path differs: " + name,
    )
    return item["sha256"]


def normalized(items):
    return {
        key: dict(path=str(Path(value["path"]).resolve(strict=True)), sha256=value["sha256"])
        for key, value in items.items()
    }


def validate_index(summary, index, consumed, protocol_sha):
    """Pure exact 180-entry identity/range check; history decorations stay separate."""
    require(
        summary.get("status") == "complete"
        and summary.get("tracking_performed") is False
        and type(summary.get("completed_native_frames")) is int
        and summary["completed_native_frames"] == STOP - FIRST
        and type(summary.get("video_payload_bytes_copied")) is int
        and summary["video_payload_bytes_copied"] == 0,
        "D2 invocation is not globally complete without tracking/copying",
    )
    equal(index.get("source_frame_range"), [FIRST, STOP], "D2 global native range differs")
    equal(index.get("chunk_frames"), CHUNK, "D2 chunk size differs")
    require(
        index.get("schema") == "medaka_mask_chunk_index_v1"
        and index.get("protocol_sha256") == summary.get("protocol_sha256") == protocol_sha,
        "D2 index/protocol schema differs",
    )
    history, entries = summary.get("chunks"), index.get("chunks")
    require(
        all(isinstance(rows, list) and len(rows) == COUNT for rows in (history, entries, consumed)),
        "exactly 180 D2/consumed entries required",
    )
    for number, (past, entry, used) in enumerate(zip(history, entries, consumed, strict=True)):
        equal(used.get("index"), number, "consumed D2 index differs")
        require(set(entry) == set(ENTRY_KEYS), "D2 index entry schema differs")
        equal(entry["start"], FIRST + number * CHUNK, "D2 start/order differs")
        equal(entry["stop"], FIRST + (number + 1) * CHUNK, "D2 stop/order differs")
        for row in (past, used):
            equal(
                {key: row[key] for key in ENTRY_KEYS},
                entry,
                "D2 assembly differs from selected/consumed attempt",
            )
        require(
            type(past.get("reused")) is bool and isinstance(past.get("rejected_attempts"), list),
            "D2 invocation attempt history missing",
        )
    require(
        isinstance(summary.get("rejected_assemblies"), list),
        "D2 assembly rejection history missing",
    )


def verify_manifest(root, manifest, seal, fixed, protocol, start, stop, kind, evidence):
    """Verify producer-specific metadata and every payload/model/source binding."""
    binding = fixed["detector_binding"]
    equal(manifest.get("binding"), binding, "D2 manifest binding differs")
    equal(seal["params"].get("binding"), binding, "D2 producer binding differs")
    equal(manifest.get("runtime"), binding["runtime"], "D2 runtime differs")
    equal(manifest.get("frames"), list(range(start, stop)), "D2 manifest frames differ")
    equal(manifest.get("source_interval_half_open"), [start, stop], "D2 manifest interval differs")
    for key in ("experiment", "fps", "image_shape_hw", "background_frames", "background_rule"):
        equal(manifest.get(key), protocol[key], "D2 native/background contract differs: " + key)
    require(
        manifest.get("kind") == kind
        and manifest.get("protocol_sha256") == fixed["protocol_sha256"]
        and manifest.get("tracking_performed") is False
        and manifest.get("human_truth_created") is False
        and manifest.get("mask_schema") == "bbox_packbits_instances_v1"
        and type(manifest.get("video_payload_bytes_copied")) is int
        and manifest["video_payload_bytes_copied"] == 0,
        "D2 producer science/schema flags differ",
    )
    validate_source_manifest(root, manifest)
    for key, expected in fixed["detector_inputs"].items():
        require(key in seal["inputs"], "D2 registered input missing")
        equal(normalized({key: seal["inputs"][key]})[key], expected, "D2 registered input differs")
    evidence.add(root / "protocol.json", fixed["protocol_sha256"])
    require(
        named(seal, "protocol", root / "protocol.json") == fixed["protocol_sha256"],
        "D2 protocol snapshot differs",
    )
    for text, digest in binding["code_sha256"].items():
        source = relative(REPO, text)
        evidence.add(source, digest)
        snapshot = relative(root, "code_snapshot/" + text)
        require(named(seal, "code:" + text, snapshot) == digest, "D2 code snapshot differs")
    for view in VIEWS:
        for category, suffix in (("raw", "raw"), ("masks", "masks"), ("backgrounds", "background")):
            info = manifest[category][view]
            path = relative(root, info["path"])
            require(
                named(seal, view + "_" + suffix, path) == info["sha256"],
                "D2 payload is not the sealed output",
            )
            evidence.add(path, info["sha256"])
        raw = manifest["raw"][view]
        source = fixed["detector_inputs"]["video:" + view]
        require(
            raw.get("storage_link") == "hardlink"
            and raw["sha256"] == source["sha256"]
            and os.path.samefile(relative(root, raw["path"]), source["path"]),
            "D2 raw hardlink/source differs",
        )
        hashes = raw["frame_bgr_sha256"]
        require(
            len(hashes) == stop - start
            and all(
                isinstance(value, str)
                and len(value) == 64
                and all(c in "0123456789abcdef" for c in value)
                for value in hashes
            ),
            "D2 raw native SHA denominator differs",
        )
        equal(
            manifest["backgrounds"][view]["frames"],
            protocol["background_frames"],
            "D2 background sample list differs",
        )
        require(
            manifest["backgrounds"][view]["sha256"] == fixed["background_sha256"][view],
            "D2 background bytes differ",
        )
        counts = manifest["masks"][view]["instances_per_frame"]
        require(
            len(counts) == stop - start and all(type(n) is int and n >= 0 for n in counts),
            "D2 instance-count denominator differs",
        )
    require(set(manifest["models"]) == {"top", "side"}, "D2 model pair differs")
    for kind, model in manifest["models"].items():
        for label in ("weights", "config"):
            actual = model[label]
            equal(
                normalized({label: actual})[label],
                fixed["detector_inputs"][f"asset:{kind}_{label}"],
                "D2 model asset differs",
            )
        effective = relative(root, model["effective_config"])
        require(
            named(seal, kind + "_effective_config", effective) == model["effective_config_sha256"],
            "D2 effective config differs",
        )
        require(
            named(seal, kind + "_source_config", root / f"{kind}_source_config.yaml")
            == fixed["detector_inputs"][f"asset:{kind}_config"]["sha256"],
            "D2 source config differs",
        )


def load_completed_assembly(invocation_seal, expected_sha, chain, consumed, evidence):
    """Read only sealed metadata/hash payloads; no NPZ arrays or video are decoded.

    invocation_seal is an explicit chosen successful runs/run_N/run.provenance.json.
    Failed/paused sibling invocations are never promoted or inferred by directory
    enumeration. The chosen run's rejected attempts/assemblies remain in summary.
    """
    require(
        isinstance(expected_sha, str)
        and len(expected_sha) == 64
        and all(c in "0123456789abcdef" for c in expected_sha),
        "explicit completed D2 invocation SHA256 required",
    )
    require(
        chain.get("chain_version") == "old_prefix_then_receipted_child_v1"
        and chain.get("tracking_processes") == 33
        and chain.get("checkpoint_restores") == 32
        and chain.get("final_audit_complete") is False
        and chain["final_record"]["params"]["through_frame"] == STOP,
        "verified completed tracking chain required",
    )
    evidence.add(Path(__file__))
    for path, digest in PINS.items():
        evidence.add(REPO / path, digest)
    fixed = chain["binding"]["inputs"]
    root = Path(fixed["input_stage_root"]).resolve(strict=True)
    path = Path(invocation_seal).resolve(strict=True)
    require(
        path.parent.parent == root / "runs" and path.name == "run.provenance.json",
        "D2 completed invocation is outside registered stage",
    )
    evidence.add(path, expected_sha)
    run = verified_seal(
        path, "restartable_independent_mask_preparation", evidence, required=("summary.json",)
    )
    equal(run["params"].get("binding"), fixed["detector_binding"], "D2 run binding differs")
    require(
        run["params"].get("global_preparation_complete") is not False,
        "D2 producer explicitly recorded incomplete",
    )
    named(run, "summary", path.parent / "summary.json")
    summary = read_json(path.parent / "summary.json")
    final_record, final_run = chain["final_record"], Path(chain["final_run"])
    used_path = final_run / "consumed_chunks.json"
    evidence.add(used_path, named(final_record, "consumed_chunks", used_path))
    equal(consumed, read_json(used_path), "D2 consumed records differ from final tracking seal")
    assembly = relative(root, summary["final_sequence_path"])
    require(assembly.parent == root / "assemblies", "D2 assembly is outside assembly directory")
    seal_path = assembly / "staging.provenance.json"
    evidence.add(seal_path, summary["final_provenance_sha256"])
    record = verified_seal(
        seal_path,
        "assemble_independent_mask_chunks",
        evidence,
        required=("manifest.json", "protocol.json", "chunks.json"),
    )
    evidence.add(assembly / "manifest.json", summary["final_manifest_sha256"])
    require(
        named(record, "manifest", assembly / "manifest.json") == summary["final_manifest_sha256"],
        "D2 global manifest output differs",
    )
    manifest, index = read_json(assembly / "manifest.json"), read_json(assembly / "chunks.json")
    require(
        manifest.get("chunks") == COUNT
        and manifest.get("chunk_index") == "chunks.json"
        and named(record, "chunk_index", assembly / "chunks.json")
        == manifest["chunk_index_sha256"],
        "D2 global chunk index binding differs",
    )
    require(
        Path(index["attempt_paths_relative_to"]).resolve(strict=True) == root,
        "D2 index attempt root differs",
    )
    validate_index(summary, index, consumed, fixed["protocol_sha256"])
    protocol_path = Path(fixed["detector_inputs"]["protocol"]["path"])
    evidence.add(protocol_path, fixed["protocol_sha256"])
    protocol = validate_protocol(read_json(protocol_path))
    equal(protocol["source_frame_range"], [FIRST, STOP], "D2 protocol range differs")
    equal(protocol["chunk_frames"], CHUNK, "D2 protocol chunk size differs")
    equal(
        fixed["detector_binding"]["input_sha256"],
        {key: value["sha256"] for key, value in fixed["detector_inputs"].items()},
        "D2 fixed input SHA map differs",
    )
    verify_manifest(
        assembly,
        manifest,
        record,
        fixed,
        protocol,
        FIRST,
        STOP,
        "assembled_independent_detector_chunks",
        evidence,
    )
    expected_inputs = copy.deepcopy(fixed["detector_inputs"])
    chunks = []
    for number, row in enumerate(consumed):
        folder = relative(root, row["attempt_path"])
        manifest_path, chunk_seal = folder / "manifest.json", folder / "staging.provenance.json"
        evidence.add(manifest_path, row["manifest_sha256"])
        evidence.add(chunk_seal, row["provenance_sha256"])
        child = verified_seal(
            chunk_seal,
            "prepare_independent_mask_chunk",
            evidence,
            required=("manifest.json", "protocol.json"),
        )
        equal(normalized(child["inputs"]), fixed["detector_inputs"], "D2 chunk input set differs")
        equal(
            normalized(child["outputs"]), row["output_hashes"], "D2 consumed output proofs differ"
        )
        require(
            named(child, "manifest", manifest_path) == row["manifest_sha256"],
            "D2 chunk manifest output differs",
        )
        chunk = read_json(manifest_path)
        verify_manifest(
            folder,
            chunk,
            child,
            fixed,
            protocol,
            row["start"],
            row["stop"],
            "independent_detector_chunk",
            evidence,
        )
        for kind in ("top", "side"):
            equal(
                chunk["models"][kind],
                manifest["models"][kind],
                "D2 assembly model metadata differs",
            )
        inputs = [manifest_path, chunk_seal]
        inputs.extend(
            relative(folder, chunk[category][view]["path"])
            for category in ("masks", "backgrounds")
            for view in VIEWS
        )
        for item in inputs:
            expected_inputs[f"chunk:{number}:{item.relative_to(folder).as_posix()}"] = dict(
                path=str(item), sha256=evidence.add(item)
            )
        lo, hi = row["start"] - FIRST, row["stop"] - FIRST
        for view in VIEWS:
            for category, field in (("raw", "frame_bgr_sha256"), ("masks", "instances_per_frame")):
                equal(
                    manifest[category][view][field][lo:hi],
                    chunk[category][view][field],
                    "D2 assembly per-frame metadata slice differs",
                )
        chunks.append(dict(root=folder, manifest=chunk, record=copy.deepcopy(row)))
    equal(normalized(record["inputs"]), expected_inputs, "D2 assembly input membership differs")
    return dict(
        root=assembly,
        manifest=manifest,
        chunks=tuple(chunks),
        evidence=evidence,
        summary=dict(
            schema="long_final_d2_assembly_inputs_v1",
            chunks=COUNT,
            native_frames=STOP - FIRST,
            invocation=str(path),
            invocation_sha256=expected_sha,
            producer_summary=copy.deepcopy(summary),
            metadata_verified=True,
            packed_arrays_verified=False,
            final_audit_complete=False,
        ),
    )


def packed_layout(data):
    """Structural checks only; no decode and no inferred identity."""
    require(set(data) == set(SCHEMA_DTYPES), "packed schema must retain exactly thirteen fields")
    for key, dtype in SCHEMA_DTYPES.items():
        require(
            isinstance(data[key], np.ndarray) and data[key].dtype == dtype,
            "packed dtype differs: " + key,
        )
    frames, boxes, bits = data["frame_idx"], data["mask_boxes"], data["mask_bits"]
    require(
        frames.ndim == 1 and len(frames) > 0 and frames[0] >= 0 and np.all(np.diff(frames) == 1),
        "packed native frames differ",
    )
    require(
        boxes.ndim == 2 and boxes.shape[1:] == (4,) and bits.ndim == 1,
        "packed instance/byte shapes differ",
    )
    n, m, b = len(frames), len(boxes), len(bits)
    shapes = {
        "image_shape_hw": (2,),
        "frame_offsets": (n + 1,),
        "mask_byte_offsets": (m + 1,),
        "scores": (m,),
        "pred_boxes": (m, 4),
        "classes": (m,),
        "instance_id": (m,),
        "source_track_id": (m,),
        "view_available": (n,),
        "frame_mask_sha256": (n,),
    }
    for key, shape in shapes.items():
        require(data[key].shape == shape, "packed shape differs: " + key)
    require(np.all(data["image_shape_hw"] > 0), "packed image shape differs")
    for key, end in (("frame_offsets", m), ("mask_byte_offsets", b)):
        value = data[key]
        require(
            value[0] == 0
            and value[-1] == end
            and np.all(np.diff(value) >= 0)
            and np.all((value >= 0) & (value <= end)),
            "packed offsets differ: " + key,
        )
    return n, m, b


class PackedStream:
    """Sequential NPZ reader; reads each global array payload exactly once."""

    def __init__(self, path):
        self.path = Path(path)
        self.stack = ExitStack()
        self.streams, self.headers, self.counts = {}, {}, {}

    def __enter__(self):
        try:
            archive = self.stack.enter_context(ZipFile(self.path))
            names = archive.namelist()
            require(
                len(names) == 13 and set(names) == {key + ".npy" for key in SCHEMA_DTYPES},
                "assembled NPZ requires thirteen unique members",
            )
            for key, dtype in SCHEMA_DTYPES.items():
                stream = self.stack.enter_context(archive.open(key + ".npy"))
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "unsupported assembled NPY version")
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, fortran, actual = reader(stream)
                require(actual == dtype and not fortran, "assembled dtype/C-order differs: " + key)
                require(all(type(n) is int and n >= 0 for n in shape), "invalid assembled shape")
                self.streams[key], self.headers[key], self.counts[key] = stream, shape, 0
            return self
        except BaseException:
            self.stack.close()
            raise

    def compare(self, key, expected):
        shape = self.headers[key]
        require(
            len(shape) == expected.ndim and shape[1:] == expected.shape[1:],
            "assembled trailing shape differs: " + key,
        )
        payload = expected.tobytes(order="C")
        actual = self.streams[key].read(len(payload))
        require(actual == payload, "assembled chunk bytes differ: " + key)
        self.counts[key] += expected.size
        require(self.counts[key] <= math.prod(shape), "assembled shape overflow: " + key)

    def finish(self):
        for key, shape in self.headers.items():
            require(
                self.counts[key] == math.prod(shape) and self.streams[key].read(1) == b"",
                "assembled terminal shape/payload differs: " + key,
            )

    def __exit__(self, *args):
        self.stack.close()


def verify_packed_stream(assembled_path, chunks, *, on_chunk=None):
    """Compare an iterable of chunk mappings to one assembled NPZ, without mutation.

    General native start/chunk lengths here support small qualification fixtures;
    the completed-assembly entrypoint separately enforces the exact 180 schedule.
    Returns counts only; it never claims decoded-mask validation or final audit.
    """
    frames = instances = byte_count = count = 0
    previous = shape = None
    with PackedStream(assembled_path) as stream:
        for count, data in enumerate(chunks, 1):
            n, m, b = packed_layout(data)
            require(
                previous is None or int(data["frame_idx"][0]) == previous + 1,
                "packed chunk boundary is not consecutive",
            )
            if shape is None:
                shape = data["image_shape_hw"].copy()
                stream.compare("image_shape_hw", shape)
                for key in ("frame_offsets", "mask_byte_offsets"):
                    stream.compare(key, np.zeros(1, dtype=np.int64))
            else:
                require(
                    np.array_equal(shape, data["image_shape_hw"]),
                    "packed chunk image shapes differ",
                )
            for key in SCHEMA_DTYPES:
                if key == "image_shape_hw":
                    continue
                if key == "frame_offsets":
                    expected = data[key][1:] + instances
                elif key == "mask_byte_offsets":
                    expected = data[key][1:] + byte_count
                else:
                    expected = data[key]
                stream.compare(key, expected)
            previous = int(data["frame_idx"][-1])
            frames, instances, byte_count = frames + n, instances + m, byte_count + b
            if on_chunk:
                on_chunk(
                    dict(
                        index=count - 1,
                        start=int(data["frame_idx"][0]),
                        stop=previous + 1,
                        frames=n,
                        instances=m,
                        mask_bytes=b,
                        exact_fields=13,
                    )
                )
        require(count > 0, "packed assembly requires at least one chunk")
        stream.finish()
    return dict(
        chunks=count,
        frames=frames,
        instances=instances,
        mask_bytes=byte_count,
        exact_fields=13,
        decoded_masks_rechecked=False,
    )


def verify_packed_assembly(context, *, on_chunk=None):
    """Open one chunk NPZ at a time; every view/chunk must compare before success."""
    require(
        context["summary"]["metadata_verified"] is True and len(context["chunks"]) == COUNT,
        "verified D2 metadata context required",
    )
    evidence, results = context["evidence"], {}
    for view in VIEWS:
        item = context["manifest"]["masks"][view]
        assembled = relative(context["root"], item["path"])
        evidence.add(assembled, item["sha256"])

        def chunks(view=view):
            for selected in context["chunks"]:
                row, manifest = selected["record"], selected["manifest"]
                info = manifest["masks"][view]
                path = relative(selected["root"], info["path"])
                evidence.add(path, info["sha256"])
                with np.load(path, allow_pickle=False) as archive:
                    require(
                        len(archive.files) == 13 and set(archive.files) == set(SCHEMA_DTYPES),
                        "chunk NPZ schema/duplicate fields differ",
                    )
                    data = {key: archive[key] for key in archive.files}
                packed_layout(data)
                require(
                    np.array_equal(
                        data["frame_idx"], np.arange(row["start"], row["stop"], dtype=np.int64)
                    )
                    and np.array_equal(data["image_shape_hw"], manifest["image_shape_hw"])
                    and data["view_available"].all()
                    and np.all(data["source_track_id"] == -1),
                    "selected D2 native/availability/independence arrays differ",
                )
                equal(
                    np.diff(data["frame_offsets"]).tolist(),
                    info["instances_per_frame"],
                    "selected D2 instance counts differ",
                )
                yield data

        def saved(row, view=view):
            if on_chunk:
                on_chunk(dict(view=view, **row))

        results[view] = verify_packed_stream(assembled, chunks(), on_chunk=saved)
        require(
            results[view]["chunks"] == COUNT and results[view]["frames"] == STOP - FIRST,
            "D2 packed full denominator differs",
        )
    return dict(
        schema="long_final_d2_assembly_comparison_v1",
        views=results,
        chunk_view_comparisons=COUNT * len(VIEWS),
        logical_field_comparisons=COUNT * len(VIEWS) * 13,
        packed_arrays_verified=True,
        decoded_masks_rechecked=False,
        final_audit_complete=False,
        caller_final_guard_required=True,
    )

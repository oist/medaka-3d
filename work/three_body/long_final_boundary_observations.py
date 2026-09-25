"""179 boundary-flow flags and 33 checkpoint previous-observation comparisons.

Caller supplies a completed final state/chain, bound calibration/backgrounds and
shared evidence. Historical history NPZ members are never deserialized. No raw
video, tracking, fit, process, final-chain loader or final success seal is run.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import audit_long_multichunk_minute as old
import cv2
import long_final_native_spool as spool
import numpy as np
import tracker_checkpoint as codec
from motion import FlowConfig, FlowObservations
from observations import FrameObservation

COMMON, VIEWS = old.COMMON, spool.VIEWS
require = spool.require
CATEGORIES = ("detections", "images_bgr", "masks", "backgrounds_bgr")
PREVIOUS = tuple("previous_" + c + "_" + v for c in CATEGORIES for v in VIEWS)


def endpoint_stops():
    return (51460, 53120, 59760, 66400, 76360, *range(86320, 345281, 9960), 348600)


def background_binding(backgrounds, encoded_records):
    """External verified file SHA and decoded-array signature are different proofs."""
    require(set(backgrounds) == set(encoded_records) == set(VIEWS), "all3 backgrounds required")
    result = {}
    for v in VIEWS:
        a, record = backgrounds[v], encoded_records[v]
        require(
            a.dtype == np.uint8 and a.ndim == 3 and a.shape[-1] == 3,
            "uint8 BGR background required",
        )
        require(
            isinstance(record["path"], str)
            and isinstance(record["sha256"], str)
            and len(record["sha256"]) == 64,
            "encoded background file record required",
        )
        result[v] = dict(
            encoded_file=dict(record),
            decoded_array=spool.signature(a),
            decoded_from_encoded_verified_here=False,
        )
    return result


def frame_from_spool(reader, n, backgrounds):
    values = {c: {} for c in CATEGORIES}
    for view in VIEWS:
        data = reader.read_view(n, view)
        require(
            backgrounds[view].dtype == np.uint8
            and backgrounds[view].shape == data["image_bgr"].shape,
            "registered background shape differs",
        )
        values["detections"][view] = data["detections"]
        values["images_bgr"][view] = data["image_bgr"]
        values["masks"][view] = data["masks"]
        values["backgrounds_bgr"][view] = backgrounds[view].copy()
    return FrameObservation(n, **values)


def frame_signatures(frame):
    return {
        c + "/" + v: None if (value := getattr(frame, c)[v]) is None else spool.signature(value)
        for c in CATEGORIES
        for v in VIEWS
    }


def _flow_checks(
    reader,
    state,
    rig,
    backgrounds,
    flow_config,
    threads,
    first,
    checks,
    restore_starts,
    on_flow=None,
):
    """Small synthetic engine; public wrapper fixes179 boundaries/32 restores."""
    config = FlowConfig()
    require(
        spool.canonical(dataclasses.asdict(config)) == spool.canonical(flow_config),
        "original FlowConfig differs",
    )
    require(
        type(threads) is int and threads > 0, "registered positive OpenCV thread count required"
    )
    offsets = state["history_offsets"]
    require(
        offsets.dtype == np.int64 and offsets.ndim == 1 and offsets[0] == 0,
        "history offsets dtype/start differs",
    )
    results, previous_threads = [], cv2.getNumThreads()
    cv2.setNumThreads(threads)
    try:
        require(cv2.getNumThreads() == threads, "OpenCV did not accept registered threads")
        for n in checks:
            t = n - first
            require(1 <= t < len(offsets) - 1, "boundary outside completed history")
            before, lo, hi = map(int, offsets[t - 1 : t + 2])
            require(
                0 <= before < lo < hi and hi - lo == 6 and lo - before == 6,
                "all6 previous/current rows required",
            )
            require(hi <= len(state["history_pose_xyz"]), "history offset exceeds rows")
            COMMON.same_array(
                state["history_frame_idx"][before:lo],
                np.full(6, n - 1, np.int64),
                "previous native rows",
            )
            COMMON.same_array(
                state["history_frame_idx"][lo:hi], np.full(6, n, np.int64), "current native rows"
            )
            parent = state["history_pose_xyz"][before].copy()
            require(
                parent.shape == (3, 3, 3) and np.isfinite(parent).all(),
                "finite first-retained parent pose required",
            )
            parent_before = spool.signature(parent)
            a, b = (
                frame_from_spool(reader, n - 1, backgrounds),
                frame_from_spool(reader, n, backgrounds),
            )
            inputs = frame_signatures(a), frame_signatures(b)
            flow = FlowObservations.from_frames(a, b, rig, parent, config)
            require(
                inputs == (frame_signatures(a), frame_signatures(b))
                and spool.signature(parent) == parent_before,
                "flow mutated observation/parent inputs",
            )
            require(
                flow.previous_frame_idx == n - 1 and flow.frame_idx == n, "flow native pair differs"
            )
            matches = []
            for r in range(lo, hi):
                matches.append(
                    COMMON.same_array(
                        state["history_proposal_flow_nodes"][r],
                        flow.node_valid,
                        "boundary flow flags",
                    )
                )
                require(state["history_proposal_flow_parent"][r] == 0, "saved flow parent must be0")
            arrays = {
                f.name: np.asarray(getattr(flow, f.name)).copy() for f in dataclasses.fields(flow)
            }
            row = dict(
                previous_frame=n - 1,
                frame_idx=n,
                boundary_kind="restore" if n in restore_starts else "same_process_reader",
                previous_first_history_row=before,
                current_history_rows=[lo, hi],
                first_parent_pose_signature=parent_before,
                six_flow_flags_exact=matches,
                flow_parent=0,
                valid_nodes=int(flow.node_valid.sum()),
                flow_array_signatures=spool.signatures(arrays),
                coordinate_arrays_are_recomputed_diagnostics=True,
            )
            if on_flow is not None:
                on_flow(copy.deepcopy(row), arrays)
            results.append(row)
    finally:
        cv2.setNumThreads(previous_threads)
    return results


def verify_flow_boundaries(
    reader,
    state,
    metadata,
    rig,
    *,
    backgrounds,
    flow_config,
    opencv_threads,
    restore_starts,
    on_flow=None,
):
    reader.require_final_scope()
    require(
        metadata["processed_frames"] == spool.STOP - spool.FIRST
        and metadata["previous_frame"] == spool.STOP - 1
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 32
        and metadata["context"]["tracker_config"]["flow_proposals"] is True,
        "completed final flow-enabled state required",
    )
    require(
        state["history_offsets"].shape == (spool.STOP - spool.FIRST + 1,),
        "full final offsets required",
    )
    require(
        tuple(restore_starts) == endpoint_stops()[:-1],
        "exact33-process restore boundary scope required",
    )
    rows = _flow_checks(
        reader,
        state,
        rig,
        backgrounds,
        flow_config,
        opencv_threads,
        spool.FIRST,
        spool.boundaries(),
        set(restore_starts),
        on_flow,
    )
    require(
        len(rows) == 179 and sum(r["boundary_kind"] == "restore" for r in rows) == 32,
        "179/32/147 flow scope differs",
    )
    return dict(
        boundaries=rows,
        restored_boundaries=32,
        same_process_reader_boundaries=147,
        historical_histories_loaded=False,
        final_audit_complete=False,
    )


def load_previous_members(ref, evidence, expected_context, *, archive_factory=np.load):
    """Bind codec files, deserialize metadata and present previous_* members only.

    evidence.add memoizes prior chain file hashes; outer final_guard is mandatory.
    ZIP member names are checked without reading historical array payloads.
    """
    root = Path(ref["checkpoint_root"]).resolve(strict=True)
    manifest_path = root / "manifest.json"
    evidence.add(manifest_path, ref["manifest_sha256"])
    manifest_bytes = manifest_path.read_bytes()
    manifest = codec._decode(manifest_bytes)
    require(
        set(manifest)
        == {
            "schema",
            "status",
            "bindings",
            "bindings_sha256",
            "arrays",
            "metadata_canonical_sha256",
            "metadata",
            "payload",
        }
        and codec._canonical(manifest) == manifest_bytes
        and codec._sha(codec._canonical(manifest["bindings"])) == manifest["bindings_sha256"],
        "checkpoint manifest canonical/header checksum differs",
    )
    require(
        manifest["schema"] == codec.SCHEMA
        and manifest["status"] == "ok"
        and codec._canonical(manifest["bindings"]) == codec._canonical(ref["bindings"]),
        "checkpoint schema/binding differs",
    )
    paths = {}
    for role, filename in (("metadata", "metadata.json"), ("payload", "arrays.npz")):
        record = manifest[role]
        require(record["file"] == filename, "checkpoint file role differs")
        path = codec._regular_path(root, filename)
        require(path.stat().st_size == record["bytes"], "checkpoint file size differs")
        evidence.add(path, record["sha256"])
        paths[role] = path
    raw = paths["metadata"].read_bytes()
    metadata = codec._decode(raw)
    require(
        codec._canonical(metadata) == raw
        and codec._sha(raw) == manifest["metadata_canonical_sha256"],
        "canonical checkpoint metadata differs",
    )
    require(
        metadata["previous_frame"] == ref["stop"] - 1
        and metadata["processed_frames"] == ref["stop"] - spool.FIRST
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == ref["restore_count"]
        and codec._canonical(metadata["context"]) == codec._canonical(expected_context),
        "previous lifecycle/context differs",
    )
    presence = metadata["previous_observation"]
    require(
        set(presence) == set(PREVIOUS) and all(type(v) is bool for v in presence.values()),
        "full12 previous presence entries required",
    )
    require(
        all(v or k.startswith("previous_masks_") for k, v in presence.items()),
        "non-mask previous evidence absent",
    )
    descriptions = manifest["arrays"]
    require(
        {k for k in descriptions if k.startswith("previous_")}
        == {k for k, v in presence.items() if v},
        "previous member presence differs",
    )
    keys = []
    for i, (_name, spec) in enumerate(sorted(descriptions.items())):
        require(spec["storage_key"] == f"array_{i:06d}", "codec storage-key order differs")
        keys.append(spec["storage_key"])
    result = {}
    with archive_factory(paths["payload"], allow_pickle=False) as archive:
        require(
            len(archive.files) == len(keys) and set(archive.files) == set(keys),
            "codec archive member names differ",
        )
        for name in PREVIOUS:
            if presence[name]:
                spec = descriptions[name]
                value = archive[spec["storage_key"]]
                require(
                    codec._canonical(codec._array_record(value, spec["storage_key"]))
                    == codec._canonical(spec),
                    "previous member dtype/shape/bytes differs",
                )
                result[name] = codec._copy_array(value)
    require(
        manifest_path.read_bytes() == manifest_bytes and paths["metadata"].read_bytes() == raw,
        "checkpoint metadata changed during previous-member read",
    )
    return (
        result,
        metadata,
        dict(
            manifest_sha256=ref["manifest_sha256"],
            metadata_file_sha256=manifest["metadata"]["sha256"],
            payload_file_sha256=manifest["payload"]["sha256"],
            deserialized_members=list(result),
            historical_histories_loaded=False,
        ),
    )


def compare_previous(reader, ref, arrays, metadata, backgrounds):
    frame = frame_from_spool(reader, ref["stop"] - 1, backgrounds)
    checks = {}
    for category in CATEGORIES:
        for view in VIEWS:
            key, value = "previous_" + category + "_" + view, getattr(frame, category)[view]
            require(
                metadata["previous_observation"][key] is (value is not None),
                "previous native availability differs",
            )
            if value is None:
                require(key not in arrays, "invented unavailable previous member")
                checks[key] = dict(present=False)
            else:
                checks[key] = dict(present=True, exact=COMMON.same_array(arrays[key], value, key))
    return dict(
        stop=ref["stop"],
        previous_frame=frame.frame_idx,
        restore_count=ref["restore_count"],
        checks=checks,
    )


def verify_previous_observations(
    reader, endpoint_refs, final_context, *, backgrounds, encoded_background_records, evidence
):
    reader.require_final_scope()
    require(
        [r["stop"] for r in endpoint_refs] == list(endpoint_stops())
        and [r["restore_count"] for r in endpoint_refs] == list(range(33)),
        "fixed33 chain endpoints required",
    )
    binding = background_binding(backgrounds, encoded_background_records)
    for r in encoded_background_records.values():
        evidence.add(Path(r["path"]), r["sha256"])
    results = []
    for ref in endpoint_refs:
        arrays, metadata, proof = load_previous_members(ref, evidence, final_context)
        result = compare_previous(reader, ref, arrays, metadata, backgrounds)
        result["checkpoint_proof"] = proof
        results.append(result)
    return dict(
        checkpoints=results,
        count=33,
        background_bindings=binding,
        historical_histories_loaded=False,
        final_audit_complete=False,
    )

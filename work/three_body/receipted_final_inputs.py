"""Finite metadata/rig/background adapters for the completed receipt-child audit.

No terminal array loader, tracker, detector, process or raw video is invoked.
Caller must first verify the final owner, then the frozen complete chain.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
from pathlib import Path

import cv2
import long_final_boundary_observations as boundary
import long_final_receipted_child_chain as chain_api
import numpy as np

from medaka3d.calib import load_rig

REPO, WORK = chain_api.REPO, chain_api.WORK
FIRST, STOP, VIEWS = 49800, 348600, tuple(boundary.VIEWS)
require, equal, COMMON = chain_api.require, chain_api.equal, chain_api.COMMON


def read_json(path, digest, evidence):
    path = Path(path).resolve(strict=True)
    evidence.add(path, digest)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, "JSON changed after binding")
    return json.loads(raw)


def output_record(record, path, *, role=None, expected=None):
    """Explicit named role where present; metadata/payload may use absolute-path roles."""
    path = Path(path).resolve(strict=True)
    matches = [v for v in record["outputs"].values() if Path(v["path"]).resolve() == path]
    require(bool(matches), "producer output missing: " + str(path))
    digest = matches[0]["sha256"]
    require(all(v["sha256"] == digest for v in matches), "conflicting producer output aliases")
    if role is not None:
        require(
            role in record["outputs"]
            and Path(record["outputs"][role]["path"]).resolve() == path
            and record["outputs"][role]["sha256"] == digest,
            "producer named output role differs",
        )
    require(expected is None or digest == expected, "producer output SHA differs")
    for row in matches:
        sizes = [row[k] for k in ("size", "size_bytes", "bytes") if k in row]
        require(sizes and all(v == path.stat().st_size for v in sizes), "output size differs")
    return digest


def named_json(record, role, path, evidence):
    return read_json(path, output_record(record, path, role=role), evidence)


def completed_header(root, expected_sha, evidence):
    """Small JSON only; full owner proof must still precede the chain/array loader."""
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(REPO), "supervisor outside workspace")
    header = read_json(root / "supervisor.provenance.json", expected_sha, evidence)
    require(
        header.get("stage") == chain_api.SUPERVISOR_STAGE
        and header.get("status") == "ok"
        and bool(header.get("finished_at")),
        "successful final supervisor required",
    )
    summary = named_json(header, "summary", root / "summary.json", evidence)
    policy = named_json(header, "plan", root / "supervisor_plan.json", evidence)
    handoff = named_json(header, "handoff", root / "handoff.json", evidence)
    chain_api.validate_new_header(summary, policy, handoff)
    return header


def make_endpoint_refs(chain, evidence):
    """Only JSON+bound file references; no historical NPZ member is opened here."""
    require(
        chain["chain_version"] == "old_prefix_then_receipted_child_v1"
        and chain["final_audit_complete"] is False
        and len(chain["ancestors"]) == 4
        and len(chain["phases"]) == 29,
        "verified complete mixed chain required",
    )
    rows = [*chain["ancestors"], *chain["phases"]]
    stops = boundary.endpoint_stops()
    old_binding, new_binding = chain["prefix"]["binding"], chain["binding"]
    require(
        len(old_binding["code_sha256"]) == 36 and len(new_binding["code_sha256"]) == 40,
        "fixed 36-to-40 source edge required",
    )
    old_last = chain["prefix"]["prefix_counts"]["checkpoint_restores"]
    require(type(old_last) is int and 4 <= old_last < 32, "invalid migration position")
    refs, previous, roots = [], FIRST, set()
    for index, (row, stop) in enumerate(zip(rows, stops, strict=True)):
        require(
            row["interval"] == [previous, stop]
            and type(row["restores"]) is int
            and row["restores"] == index,
            "endpoint order/range/restore differs",
        )
        root = Path(row["run"]).resolve(strict=True)
        require(root not in roots and root.is_relative_to(REPO), "duplicate/foreign endpoint")
        roots.add(root)
        path = root / "continuation.provenance.json"
        digest = evidence.paths.get(str(path))
        require(digest is not None, "endpoint producer was not verified by chain")
        record = read_json(path, digest, evidence)
        binding = old_binding if index <= old_last else new_binding
        stage = "three_body_long_continuation" if index <= old_last else chain_api.m.STAGE
        require(
            record.get("stage") == stage
            and record.get("status") == "ok"
            and record["params"]["through_frame"] == stop,
            "endpoint producer differs",
        )
        equal(record["params"]["bindings"], binding, "endpoint version binding differs")
        checkpoint = root / "checkpoint"
        manifest_path = checkpoint / "manifest.json"
        manifest_sha = output_record(record, manifest_path, role="checkpoint_manifest")
        require(
            evidence.paths.get(str(manifest_path)) == manifest_sha,
            "endpoint manifest missing from chain evidence",
        )
        manifest = read_json(manifest_path, manifest_sha, evidence)
        require(
            manifest.get("schema") == boundary.codec.SCHEMA and manifest.get("status") == "ok",
            "checkpoint manifest incomplete",
        )
        equal(manifest["bindings"], binding, "manifest version binding differs")
        require(
            boundary.codec._sha(boundary.codec._canonical(binding)) == manifest["bindings_sha256"],
            "manifest binding digest differs",
        )
        for role, filename in (("metadata", "metadata.json"), ("payload", "arrays.npz")):
            spec, target = manifest[role], checkpoint / filename
            require(
                spec["file"] == filename and spec["bytes"] == target.stat().st_size,
                "checkpoint member path/size differs",
            )
            output_record(record, target, expected=spec["sha256"])
            require(
                evidence.paths.get(str(target)) == spec["sha256"],
                "checkpoint member missing from chain evidence",
            )
        metadata = read_json(checkpoint / "metadata.json", manifest["metadata"]["sha256"], evidence)
        require(
            metadata["processed_frames"] == stop - FIRST
            and metadata["previous_frame"] == stop - 1
            and metadata["initialize_count"] == 1
            and metadata["restore_count"] == index,
            "endpoint metadata lifecycle differs",
        )
        refs.append(
            dict(
                checkpoint_root=str(checkpoint),
                manifest_sha256=manifest_sha,
                bindings=binding,
                stop=stop,
                restore_count=index,
            )
        )
        previous = stop
    require(
        Path(rows[-1]["run"]).resolve() == Path(chain["final_run"]).resolve(),
        "terminal endpoint differs",
    )
    return refs


def bind_rig(shared, context, node_paths, evidence):
    shared = {str(Path(p).resolve(strict=True)): d for p, d in shared.items()}
    require(len(shared) == 7, "exact SL3/calibration4 required")
    for path, digest in shared.items():
        evidence.add(path, digest)
    manifests = [Path(p) for p in shared if Path(p).name == "calibration_manifest.json"]
    require(len(manifests) == 1, "one calibration manifest required")
    path = manifests[0]
    manifest = read_json(path, shared[str(path)], evidence)
    models = [
        chain_api.native.checked_relative(path.parent, row["path"])
        for row in manifest["model_files"]
    ]
    require(len(models) == 3 and len(set(models)) == 3, "three distinct model files required")
    cameras = {str(p) for p in (path, *models)}
    sl = {v: str(Path(p).resolve(strict=True)) for v, p in node_paths.items()}
    require(
        set(sl) == set(VIEWS)
        and set(shared) == cameras | set(sl.values())
        and len(cameras) == 4
        and all(Path(sl[v]).name == f"{chain_api.native.EXPERIMENT}__{v}.npz" for v in VIEWS),
        "exact registered SL/calibration membership differs",
    )
    rig = load_rig(path.parent)
    equal(
        COMMON.normalize([dataclasses.asdict(c) for c in rig.cameras]),
        context["cameras"],
        "actual camera context differs",
    )
    require(
        list(rig.views) == context["views"] == list(VIEWS)
        and rig.world_units == context["world_units"] == "mm",
        "view order/units differ",
    )
    return rig, dict(shared_sha256=shared, cameras=context["cameras"], SL_reloaded=False)


def decode_backgrounds(chunks, expected, evidence):
    require(
        len(chunks) == 180 and set(expected) == set(VIEWS), "all180 registered backgrounds required"
    )
    arrays, records, proofs = {}, {}, {}
    for view in VIEWS:
        for chunk in chunks:
            require(
                chunk["manifest"]["backgrounds"][view]["sha256"] == expected[view],
                "chunk background differs",
            )
        chunk = chunks[0]
        item = chunk["manifest"]["backgrounds"][view]
        path = chain_api.native.checked_relative(chunk["root"], item["path"])
        evidence.add(path, expected[view])
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected[view], "encoded background changed")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        require(
            isinstance(image, np.ndarray)
            and image.dtype == np.uint8
            and image.shape == (1200, 1200, 3),
            "background decode shape/dtype differs",
        )
        arrays[view] = image
        records[view] = dict(path=str(path), sha256=expected[view])
        proofs[view] = dict(
            encoded_file=records[view],
            encoded_bytes=len(raw),
            decoded_array=boundary.spool.signature(image),
            decoded_from_encoded_verified_here=True,
            decode_mode="cv2.IMREAD_COLOR/BGR",
            opencv=cv2.__version__,
            original_median_recomputed=False,
        )
    return arrays, records, proofs


def bind_geometry_and_backgrounds(completed, native_context, evidence):
    chain = completed["chain"]
    binding, context = chain["binding"], completed["checkpoint_metadata"]["context"]
    path = REPO / "labs/20260909_three_body/long_tracking_protocol.json"
    protocol = read_json(path, binding["tracking_protocol_sha256"], evidence)
    require(
        protocol["source_frame_range"] == [FIRST, STOP]
        and protocol["fps"] == 166
        and protocol["experiment"] == chain_api.native.EXPERIMENT,
        "tracking protocol scope differs",
    )
    for key, context_key in (
        ("tracker_config", "tracker_config"),
        ("node_recovery_config", "recovery_config"),
        ("persistence_config", "persistence_config"),
    ):
        equal(protocol[key], context[context_key], "scientific context differs: " + key)
    require(
        context["implementation"] == "AcceleratedContinuousPersistentNodeTracker",
        "scientific implementation differs",
    )
    equal(
        protocol["flow_config"], dataclasses.asdict(boundary.FlowConfig()), "flow settings differ"
    )
    equal(protocol["package_versions"], binding["package_versions"], "registered versions differ")
    require(
        protocol["torch_cpu_threads"] == binding["torch_cpu_threads"]
        and type(binding["opencv_threads"]) is int
        and binding["opencv_threads"] > 0,
        "registered thread context differs",
    )
    runtime = dict(
        numpy=np.__version__,
        scipy=importlib.metadata.version("scipy"),
        opencv=cv2.__version__,
        torch=importlib.metadata.version("torch"),
    )
    for key, version in runtime.items():
        require(
            version == binding["package_versions"][key], "installed CPU package differs: " + key
        )
    for name, digest in protocol["frozen_tracker_sources_sha256"].items():
        require(
            binding["code_sha256"].get(str((REPO / name).resolve())) == digest,
            "scientific source not in binding",
        )
        evidence.add(REPO / name, digest)
    rig, proof = bind_rig(
        binding["inputs"]["seed"]["SL_and_calibration_sha256"],
        context,
        native_context.nodes.paths,
        evidence,
    )
    backgrounds, records, decoded = decode_backgrounds(
        native_context.chunks, binding["inputs"]["background_sha256"], evidence
    )
    return dict(
        rig=rig,
        protocol=protocol,
        backgrounds=backgrounds,
        encoded_background_records=records,
        proof=dict(
            geometry=proof,
            background_decode=decoded,
            cpu_runtime=runtime,
            CUDA_queried=False,
            median_recomputed=False,
        ),
    )

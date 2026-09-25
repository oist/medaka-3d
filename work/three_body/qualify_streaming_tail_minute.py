"""One fixed saved medaka minute: all streaming DAG fields versus sealed MP output.

CPU only. The original solver is not rerun, geometry is immutable, and a constant
permutation is not evidence of physical identity correctness. No cap is raised.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import identity_streaming_node_memory as streaming
import numpy as np
import run_tail_minute_identity as proof

WORK, REPO = proof.WORK, proof.REPO
START, STOP = 66400, 76360
STAGE = "streaming_tail_minute_equivalence_v1"
CONTROL = WORK / "tail_minute_mouth_memory_motion_v1"
CONTROL_SHA = "128d49b5ea83ce3143d2151ee09ec62803fef2aa8db95e9cdc2d972c0030b252"
STREAM_SHA = "9a7792a346ec0d45a40c30a0e26d2191fa4ff0f348e93e33dd4c13d4f6890125"
PROOF_SHA = "738622b1fba155f3bb92ccf9c42425c7d685da8830310efd8332864a76c0d7d4"
CONTROL_SOURCE_SHA = "853fbb0e2cfefeb94a5994de1956798fdda59eb6b9a5f08121bf0997bd4ed947"
TARGET_BLOCK_FRAMES = 256
NODES = (1, 0)
require = proof.require


def read_arrays(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
        require(len(set(saved.files)) == len(saved.files), "duplicate array key")
        values = {name: saved[name].copy() for name in saved.files}
    require(all(not value.dtype.hasobject for value in values.values()), "object array")
    return values


def array_signature(value):
    """Hash logical C bytes in bounded slabs, including exact NaN payloads."""
    require(isinstance(value, np.ndarray), "array required")
    digest = hashlib.sha256()
    # All real result arrays are C-contiguous; require it rather than hiding a
    # whole-array allocation in a ravel copy for an unexpected layout.
    require(value.flags.c_contiguous, "C-contiguous result required")
    flat = value.reshape(-1)
    step = max(1, (16 * 1024**2) // value.dtype.itemsize)
    for start in range(0, len(flat), step):
        digest.update(flat[start : start + step].tobytes(order="C"))
    return dict(shape=list(value.shape), dtype=value.dtype.str, sha256=digest.hexdigest())


def scalar_record(value, expected, name):
    if isinstance(value, tuple):
        require(type(expected) is list and list(value) == expected, name + " tuple differs")
        require(all(type(v) is int for v in value + tuple(expected)), name + " tuple types")
        return dict(kind="tuple_encoded_as_JSON_array", value=expected, exact=True)
    require(type(value) is type(expected), name + " scalar type differs")
    if type(value) is float:
        left = np.asarray(value, np.float64).tobytes().hex()
        right = np.asarray(expected, np.float64).tobytes().hex()
        require(left == right, name + " float64 bytes differ")
        return dict(kind="float64", float64_bytes=left, exact=True)
    require(value == expected, name + " scalar differs")
    return dict(kind=type(value).__name__, value=value, exact=True)


def compare_result(result, expected_arrays, expected_scalars):
    require(type(result) is streaming.frozen.IdentityNodeMemoryDPResult, "result schema differs")
    actual = {field.name: getattr(result, field.name) for field in dataclasses.fields(result)}
    array_names = {name for name, value in actual.items() if isinstance(value, np.ndarray)}
    scalar_names = set(actual) - array_names
    require(
        len(array_names) == 29
        and len(scalar_names) == 19
        and array_names == set(expected_arrays)
        and scalar_names == set(expected_scalars),
        "all 29 arrays and 19 scalars required",
    )
    arrays, scalars = {}, {}
    for name in sorted(array_names):
        value, control = actual[name], expected_arrays[name]
        require(not value.flags.writeable, name + " result must be read-only")
        signature = array_signature(value)
        require(signature == array_signature(control), name + " logical bytes differ")
        arrays[name] = dict(**signature, exact=True, read_only=True)
    for name in sorted(scalar_names):
        scalars[name] = scalar_record(actual[name], expected_scalars[name], name)
    return dict(array_count=29, scalar_count=19, all_exact=True, arrays=arrays, scalars=scalars)


def validate_metadata(plan, summary):
    require(
        plan["schema"] == summary["schema"] == "tail_minute_mouth_memory_motion_v1"
        and summary["status"] == "ok"
        and plan["native_interval"] == summary["native_interval"] == [START, STOP]
        and summary["native_frames"] == STOP - START
        and plan["node_indices"] == summary["node_indices"] == [1, 0]
        and plan["geometry"] == summary["geometry"] == "tail",
        "same successful saved Tail minute required",
    )
    require(plan["dp_config"] == proof.DP_CONFIG, "unchanged full-DAG caps and objective required")
    require(
        plan["initial_P"] == 0
        and plan["unary"] == "float64 exact zero T,6"
        and summary["raw_geometry_sources_review_modified"] is False
        and summary["RGB_fit_GPU_executed"] is False
        and summary["report"]["all_three_ID_truth"] is False,
        "saved control scope differs",
    )


def load_inputs(evidence):
    evidence.read(WORK / "identity_streaming_node_memory.py", STREAM_SHA)
    evidence.read(WORK / "run_tail_minute_identity.py", PROOF_SHA)
    evidence.read(WORK / "run_tail_minute_mouth_memory_solve.py", CONTROL_SOURCE_SHA)
    evidence.read(WORK / "identity_node_memory_dp.py", streaming.FROZEN_SHA)
    evidence.read(streaming.LAB / "identity_dp.py", streaming.CORE_SHA)
    seal = proof.seal_read(
        CONTROL,
        "identity.provenance.json",
        "tail_minute_mouth_memory_motion_v1",
        CONTROL_SHA,
        evidence,
    )
    for name, digest in (
        ("run_tail_minute_identity.py", PROOF_SHA),
        ("run_tail_minute_mouth_memory_solve.py", CONTROL_SOURCE_SHA),
        ("identity_node_memory_dp.py", streaming.FROZEN_SHA),
    ):
        proof.input_member(seal, WORK / name, digest)
    plan = json.loads(proof.named_read(seal, CONTROL, "plan.json", evidence))
    summary = json.loads(proof.named_read(seal, CONTROL, "summary.json", evidence))
    validate_metadata(plan, summary)
    # Capture every actually imported repository module, excluding installed
    # packages. Prior historical pins above remain fixed, not learned here.
    code_roots = (WORK, streaming.LAB, REPO / "src")
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and any(path.is_relative_to(root) for root in code_roots):
                paths.add(path)
    for path in sorted(paths):
        evidence.read(path, proof.file_sha(path))
    return seal, plan


def verify_spool(spool, result):
    """Verify saved membership, every file, and every logical layout before sealing."""
    root, manifest_path = Path(spool).resolve(), result.manifest_path.resolve()
    require(manifest_path == root / "manifest.json", "wrong streaming manifest path")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    require(
        metadata["execution_status"] == "complete"
        and metadata["mathematical_status"] == "optimal"
        and metadata["initial_states"] == 1
        and metadata["intermediate_ID_reset"] is False
        and metadata["global_caps_reinterpreted"] is False
        and metadata["resources"] == dataclasses.asdict(streaming.DEFAULT_RESOURCES)
        and metadata["max_layer_states"] == 100000
        and metadata["target_block_frames"] == TARGET_BLOCK_FRAMES,
        "streaming execution contract differs",
    )
    require(metadata["block_count"] > 1, "real multiple-block qualification required")
    require(
        set(manifest["logical_arrays"])
        == {
            field.name
            for field in dataclasses.fields(result.result)
            if isinstance(getattr(result.result, field.name), np.ndarray)
        },
        "all logical arrays must be saved",
    )
    for name, layout in manifest["logical_arrays"].items():
        value = getattr(result.result, name)
        path = (root / layout["path"]).resolve()
        require(
            path.is_relative_to(root)
            and isinstance(value, np.memmap)
            and Path(value.filename).resolve() == path
            and layout["shape"] == list(value.shape)
            and layout["dtype"] == value.dtype.str
            and layout["size_bytes"] == value.nbytes
            and manifest["files"][layout["path"]]["size_bytes"] == value.nbytes,
            "logical output layout differs: " + name,
        )
    records = {}
    for relative, item in manifest["files"].items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "spool path escaped")
        require(
            path.stat().st_size == item["size_bytes"] and proof.file_sha(path) == item["sha256"],
            "spool member changed: " + relative,
        )
        records[path] = dict(path=str(path), **item)
    actual = {path.resolve() for path in root.rglob("*") if path.is_file()}
    require(actual == set(records) | {manifest_path}, "spool membership differs")
    records[manifest_path] = dict(
        path=str(manifest_path),
        size_bytes=manifest_path.stat().st_size,
        sha256=proof.file_sha(manifest_path),
    )
    return metadata, records


def run(output_path):
    evidence, output = proof.Evidence(), proof.Output(output_path)
    began = datetime.now(UTC).isoformat()
    try:
        seal, prior_plan = load_inputs(evidence)
        plan = dict(
            schema=STAGE,
            native_interval=[START, STOP],
            control_sha256=CONTROL_SHA,
            target_block_frames=TARGET_BLOCK_FRAMES,
            node_indices=list(NODES),
            dp_config=prior_plan["dp_config"],
            resources=dataclasses.asdict(streaming.DEFAULT_RESOURCES),
            all_fields=[
                field.name
                for field in dataclasses.fields(streaming.frozen.IdentityNodeMemoryDPResult)
            ],
            original_solver_rerun=False,
            GPU_execution=False,
            tracking_executed=False,
            all_three_ID_truth=False,
        )
        output.save("plan.json", plan)
        for path, digest in tuple(evidence.inputs.items()):
            source = Path(path)
            if source.suffix == ".py":
                relative = "code_snapshot/" + source.relative_to(REPO).as_posix()
                target = output.root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(evidence.read(source, digest))
                output.outputs[relative] = dict(
                    path=str(target), size_bytes=target.stat().st_size, sha256=digest
                )
        geometry, _ = proof.read_candidate(
            proof.named_read(seal, CONTROL, "raw_candidate21.npz", evidence),
            support_only=False,
            first=START,
            start=START,
            stop=STOP,
        )
        raw_signatures = json.loads(
            proof.named_read(seal, CONTROL, "raw_array_signatures.json", evidence)
        )
        require(
            {name: proof.signature(value) for name, value in geometry.items()} == raw_signatures,
            "raw21 signatures differ",
        )
        expected_arrays = read_arrays(proof.named_read(seal, CONTROL, "MP/dp_arrays.npz", evidence))
        expected_scalars = json.loads(
            proof.named_read(seal, CONTROL, "MP/dp_scalars.json", evidence)
        )
        require(
            np.array_equal(expected_arrays["frame_idx"], geometry["frame_idx"]),
            "control frame extent differs",
        )
        config = {
            name: value for name, value in proof.DP_CONFIG.items() if not name.startswith("max_")
        }
        started = time.perf_counter()
        result = streaming.solve_streaming_identity_node_memory_dp(
            geometry["frame_idx"],
            np.zeros((STOP - START, 6), np.float64),
            geometry["pose_xyz"][:, :, NODES, :],
            geometry["observed"][:, :, NODES],
            spool_root=output.root / "spool",
            resources=streaming.DEFAULT_RESOURCES,
            target_block_frames=TARGET_BLOCK_FRAMES,
            node_indices=NODES,
            **config,
        )
        solve_seconds = time.perf_counter() - started
        comparison = compare_result(result.result, expected_arrays, expected_scalars)
        require(
            {name: proof.signature(value) for name, value in geometry.items()} == raw_signatures,
            "raw21 mutated",
        )
        metadata, records = verify_spool(output.root / "spool", result)
        for path, record in records.items():
            output.outputs[path.relative_to(output.root).as_posix()] = record
        output.save("field_comparison.json", comparison)
        summary = dict(
            schema=STAGE,
            status="ok",
            native_interval=[START, STOP],
            native_frames=STOP - START,
            all_29_arrays_exact=True,
            all_19_scalars_exact=True,
            raw21_unchanged=True,
            blocks=metadata["block_count"],
            global_states=metadata["global_states"],
            global_edges=metadata["global_edges"],
            maximum_layer_states=metadata["maximum_layer_states"],
            peak_block_states=metadata["peak_block_states"],
            peak_block_array_bytes=metadata["peak_block_array_bytes"],
            storage_bytes_including_manifest=sum(item["size_bytes"] for item in records.values()),
            solve_seconds=solve_seconds,
            original_solver_rerun=False,
            GPU_execution=False,
            intermediate_ID_reset=False,
            resources=metadata["resources"],
            finite_saved_medaka_minute_qualified=True,
            five_minute_qualified=False,
            all_three_ID_truth=False,
            limitations=(
                "Equivalence on one saved minute; does not establish physical ID correctness, "
                "five-minute capacity, online final identity, or OS RSS bounds."
            ),
        )
        output.save("summary.json", summary)
        evidence.guard()
        output.guard()
        final = dict(
            stage=STAGE,
            status="ok",
            started_at=began,
            finished_at=datetime.now(UTC).isoformat(),
            inputs=evidence.inputs,
            outputs=output.outputs,
        )
    except BaseException as error:
        final = dict(
            stage=STAGE,
            status="failed",
            started_at=began,
            finished_at=datetime.now(UTC).isoformat(),
            inputs=evidence.inputs,
            outputs=output.outputs,
            error=repr(error),
        )
        (output.root / "qualification.provenance.json").write_text(
            json.dumps(final, indent=2, allow_nan=False), encoding="utf-8"
        )
        raise
    seal_path = output.root / "qualification.provenance.json"
    seal_path.write_text(json.dumps(final, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            dict(**summary, seal_sha256=proof.file_sha(seal_path), output_files=len(output.outputs))
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.metadata_only:
        evidence = proof.Evidence()
        _, plan = load_inputs(evidence)
        evidence.guard()
        print(
            json.dumps(
                dict(
                    metadata_only=True,
                    inputs=len(evidence.inputs),
                    native_interval=plan["native_interval"],
                )
            ),
            flush=True,
        )
    elif args.output is None:
        parser.error("--output required")
    else:
        run(args.output)


if __name__ == "__main__":
    main()

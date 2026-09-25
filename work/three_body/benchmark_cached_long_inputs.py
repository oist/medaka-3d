"""Actual first-chunk CPU loader equality; no tracking, detector or CUDA context."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import shutil
import struct
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work/three_body"
LAB = REPO / "labs/20260909_three_body"
PINS = {
    WORK / "mask_validation_receipts.py":
        "82a284d9874608003529f9cc5ccb158f2f69adcb6da5faeda072daab11938df0",
    WORK / "cached_long_tracking_inputs.py":
        "c25ab121b2ffdf80309ebe18e03821af559e5e3cbb900bd17edb8a4f4e0a62c1",
    LAB / "long_tracking_inputs.py":
        "bea59f4e65235df8bce49bed0f1db2e92815f998b3bd7e018f456748f9e173bb",
}
SEAL = WORK / "long_continuation_first1660_v1/continuation.provenance.json"
SEAL_SHA = "9994b9193606f8910eb4ddf39f49084676003f9ba93cc50126aec82bacff15e7"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# Capture possible local dependency bytes before their first import, then retain
# just actual loaded modules. No complete tracker output arrays are opened.
_possible = {
    p.resolve() for folder in (LAB, REPO / "src/medaka3d") for p in folder.rglob("*.py")
} | {Path(__file__).resolve(), *PINS}
_before_import = {str(p): sha(p) for p in sorted(_possible)}
for _path, _digest in PINS.items():
    if _before_import[str(_path)] != _digest:
        raise ValueError(f"frozen source changed: {_path}")
for _folder in (WORK, LAB, REPO / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402
from cached_long_tracking_inputs import load_cached_long_inputs  # noqa: E402
from long_tracking_inputs import INPUT_ROOT, load_long_inputs  # noqa: E402
from mask_validation_receipts import ReceiptStore  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402


def require(value, message):
    if not value:
        raise ValueError(message)


def loaded_sources():
    paths = {Path(__file__).resolve(), *PINS}
    for module in tuple(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and str(Path(file).resolve()) in _before_import:
            paths.add(Path(file).resolve())
    return {str(p): _before_import[str(p)] for p in sorted(paths)}


def guard(hashes):
    for path, expected in hashes.items():
        require(sha(path) == expected, f"file changed: {path}")


def exact(a, b, name="value"):
    """Type-sensitive recursive equality, including signed zero/NaN array bytes."""
    require(type(a) is type(b), f"{name}: type changed")
    if isinstance(a, np.ndarray):
        require(not a.dtype.hasobject and a.dtype == b.dtype and a.shape == b.shape
                and a.tobytes(order="C") == b.tobytes(order="C"),
                f"{name}: array dtype/shape/bytes differ")
        require(not np.shares_memory(a, b), f"{name}: returns share array storage")
    elif dataclasses.is_dataclass(a):
        for field in dataclasses.fields(a):
            exact(getattr(a, field.name), getattr(b, field.name), name + "." + field.name)
    elif isinstance(a, dict):
        require(set(a) == set(b), f"{name}: dictionary keys differ")
        for key in a:
            exact(a[key], b[key], name + "." + str(key))
    elif isinstance(a, (tuple, list)):
        require(len(a) == len(b), f"{name}: length differs")
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            exact(left, right, f"{name}[{index}]")
    elif isinstance(a, float):
        require(struct.pack("!d", a) == struct.pack("!d", b), f"{name}: float bytes differ")
    elif isinstance(a, np.generic):
        require(a.dtype == b.dtype and a.tobytes() == b.tobytes(), f"{name}: scalar differs")
    else:
        require(isinstance(a, (str, int, bool, Path, type(None))) and a == b,
                f"{name}: scalar differs or unsupported type {type(a)}")


def serialize(value, arrays, key="inputs"):
    """Save every array separately and every scalar/container with explicit type."""
    if isinstance(value, np.ndarray):
        arrays[key] = value.copy()
        return dict(type="ndarray", storage_key=key, dtype=value.dtype.str,
                    shape=list(value.shape), sha256=hashlib.sha256(value.tobytes("C")).hexdigest())
    if dataclasses.is_dataclass(value):
        return dict(type=type(value).__module__ + "." + type(value).__qualname__, fields={
            f.name: serialize(getattr(value, f.name), arrays, key + "." + f.name)
            for f in dataclasses.fields(value)})
    if isinstance(value, dict):
        return dict(type="dict", entries={
            k: serialize(v, arrays, key + "." + str(k)) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return dict(type=type(value).__name__, values=[
            serialize(v, arrays, key + "." + str(i)) for i, v in enumerate(value)])
    if isinstance(value, Path):
        return dict(type=type(value).__name__, value=str(value))
    if isinstance(value, float):
        return dict(type="float", value=value, ieee754_hex=struct.pack("!d", value).hex())
    require(value is None or type(value) in (str, int, bool), "unsupported serialization type")
    return dict(type=type(value).__name__, value=value)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def input_guard_from_completed_seal():
    require(sha(SEAL) == SEAL_SHA, "wrong completed first-chunk seal")
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    require(seal["status"] == "ok" and seal["stage"] == "three_body_long_continuation"
            and seal["params"]["through_frame"] == 51460, "wrong completed input reference")
    hashes = {str(SEAL): SEAL_SHA}
    for role, item in seal["inputs"].items():
        path = Path(item["path"]).resolve(strict=True)
        require(path.is_relative_to(REPO) and role == str(path)
                and path.stat().st_size == item["size"], "input membership/path/size differs")
        hashes[str(path)] = item["sha256"]
    guard(hashes)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "new workspace output required")
    output.mkdir(parents=True)
    require(not torch.cuda.is_initialized(), "CPU benchmark cannot own a CUDA context")
    sources = loaded_sources()
    with provenance("actual_cached_long_inputs_equivalence", output, name="benchmark") as record:
        guard(sources)
        for path, digest in sources.items():
            record.input("code:" + path, path, sha256=digest)
            snapshot = output / "code_snapshot" / Path(path).relative_to(REPO)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, snapshot)
            require(sha(snapshot) == digest, "source copy changed")
            record.output(str(snapshot), snapshot, sha256=digest)
        record.param("fixed_range", [49800, 51460])
        record.param("order", ["original", "receipt_cold", "receipt_fresh_store_warm"])
        runtime = dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                       scipy=scipy.__version__, opencv=cv2.__version__,
                       torch=str(torch.__version__),
                       torch_cpu_threads=torch.get_num_threads(), cv2_threads=cv2.getNumThreads(),
                       platform=platform.platform(), cuda_context_initialized=False)
        record.param("actual_runtime", runtime)
        started = time.perf_counter()
        inputs = input_guard_from_completed_seal()
        guard_seconds = time.perf_counter() - started
        for path, digest in inputs.items():
            record.input("sealed_input:" + path, path, sha256=digest)
        seed = WORK / "long_seed_registration_49800_v1"
        stage = WORK / "inputs/long_source_masks_49800-348600_v1"
        root = output / "receipts"
        require(not root.exists(), "cold receipt directory already exists")
        results, timing, receipt_evidence = {}, {}, {}
        for label in ("original", "receipt_cold", "receipt_fresh_store_warm"):
            print(json.dumps(dict(event="start", method=label)), flush=True)
            guard(sources)
            store = None if label == "original" else ReceiptStore(root)
            begin = time.perf_counter()
            if store is None:
                value = load_long_inputs(seed, stage, INPUT_ROOT, through_frame=51460)
            else:
                value = load_cached_long_inputs(
                    seed, stage, INPUT_ROOT, through_frame=51460, receipt_store=store)
            elapsed = time.perf_counter() - begin
            guard(sources)
            require(loaded_sources() == sources, "local imported source membership changed")
            require(not torch.cuda.is_initialized(), "loader initialized CUDA")
            require(value.full_frame_range == (49800, 348600)
                    and len(value.frame_idx) == 1660 and len(value.chunk_records) == 1
                    and not value.full_interval_covered and not value.global_assembly_verified,
                    "wrong bounded input result")
            require(all(inputs.get(p) == d for p, d in value.verified_hashes.items()),
                    "loader introduced an input not bound before execution")
            checks = []
            if results:
                for field in dataclasses.fields(value):
                    exact(getattr(results["original"], field.name), getattr(value, field.name),
                          field.name)
                    checks.append(field.name)
            arrays = {}
            description = serialize(value, arrays)
            for view in value.rig.views:
                arrays["projected_seed_" + view] = value.rig.camera(view).project(
                    value.initial_pose_xyz.reshape(-1, 3))
                if results:
                    reference = results["original"].rig.camera(view).project(
                        results["original"].initial_pose_xyz.reshape(-1, 3))
                    exact(reference, arrays["projected_seed_" + view], "projected_seed_" + view)
            np.savez_compressed(output / (label + ".npz"), **arrays)
            write_json(output / (label + ".json"), description)
            uses = [] if store is None else store.uses
            if store is not None:
                require(len(uses) == 3 and {u["scope"]["view"] for u in uses}
                        == set(value.rig.views), "not exactly three view receipts")
                want_hit = label == "receipt_fresh_store_warm"
                require(all(u["cache_hit"] is want_hit and not u["rejected_entries"] for u in uses),
                        "cold must miss three; fresh warm must hit three clean receipts")
                store.final_guard()
                receipt_evidence.update(store.evidence_hashes)
            timing[label] = dict(full_loader_seconds=elapsed,
                                 receipt_validate_seconds=sum(u["elapsed_seconds"] for u in uses),
                                 receipts=uses, fields_compared_exact=checks)
            results[label] = value
            write_json(output / (label + "_measurement.json"), timing[label])
            print(json.dumps(dict(event="complete", method=label, seconds=elapsed,
                                  fields_equal=checks, hits=sum(u["cache_hit"] for u in uses))),
                  flush=True)
        cold = timing["receipt_cold"]["receipts"]
        warm = timing["receipt_fresh_store_warm"]["receipts"]
        for a, b in zip(cold, warm, strict=True):
            for key in ("key", "receipt_path", "receipt_sha256", "commit_path", "commit_sha256",
                        "npz_path", "scope"):
                exact(a[key], b[key], "cold/warm receipt " + key)
        limitations = [
            "One fixed native chunk, sequential original/cold/warm measurements; no repeats.",
            "Other existing jobs and OS disk caches are uncontrolled; no concurrency causal claim.",
            "Cold means no semantic receipts, not a cold operating-system file cache.",
            "Receipt.validate timing includes request hashes and validation or lookup; it is not "
            "an isolated semantic decode timer. Full loader includes its original raw SHA guards.",
            "No raw image decoding, GPU context, detector, tracking, GT or old candidate arrays.",
            "Registered detector runtime metadata is unchanged; no fake runtime was injected by "
            "this benchmark and native detector execution/DLL equivalence is not measured.",
            "Only a 1660-frame input prefix is checked; no full 30-minute performance forecast.",
            "Receipt evidence is separate from exact LongInputs, fixed binding and chunk records.",
        ]
        summary = dict(schema="actual_cached_long_inputs_equivalence_v1", passed=True,
                       frame_range=[49800, 51460], native_frames=1660,
                       views=list(results["original"].rig.views),
                       runtime=runtime, before_input_guard_seconds=guard_seconds, timings=timing,
                       all_dataclass_fields_exact=True,
                       dataclass_field_count=len(dataclasses.fields(value)),
                       all_rig_fields_and_seed_projections_byte_exact=True,
                       independent_return_arrays=True, cold_misses=3, fresh_store_warm_hits=3,
                       code_hashes=sources, original_verified_hash_count=len(value.verified_hashes),
                       receipt_evidence=receipt_evidence, limitations=limitations)
        write_json(output / "summary.json", summary)
        rows = "\n".join(
            f"| {k} | {v['full_loader_seconds']:.6f} | {v['receipt_validate_seconds']:.6f} |"
            for k, v in timing.items())
        text = (
            "# Actual first-chunk receipt-loader equality\n\n"
            "All 12 LongInputs fields, all Rig/camera fields and 9-point seed projections "
            "match exactly (array dtype/shape/C-order bytes); returned arrays are independent. "
            "Cold creates 3 receipts; a fresh store reuses those same 3 receipts.\n\n"
            "| Invocation | Full loader seconds | Receipt.validate seconds (3 views) |\n"
            "|---|---:|---:|\n" + rows + "\n\n"
            + "\n".join("- " + item for item in limitations) + "\n")
        (output / "RESULT.md").write_text(text, encoding="utf-8")
        for path, digest in receipt_evidence.items():
            if not Path(path).is_relative_to(output):
                record.input("receipt_evidence:" + path, path, sha256=digest)
        for path in sorted(output.rglob("*")):
            if path.is_file():
                record.output(str(path), path, sha256=sha(path))
        guard(inputs)
        guard(sources)
        guard(receipt_evidence)
        guard({item["path"]: item["sha256"] for item in record.outputs.values()})
        record.param("all_input_source_receipt_snapshot_output_sha_unchanged", True)
        record.param("passed", True)
    print(json.dumps(dict(event="success", output=str(output),
                          summary_sha256=sha(output / "summary.json"))),
          flush=True)


if __name__ == "__main__":
    main()

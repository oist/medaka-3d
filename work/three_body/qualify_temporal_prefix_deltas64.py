"""Saved-only exact archive qualification for both fixed successful 64-frame runs.

Read only selected named outputs, producer/core snapshots and official source.
No image, mask, video, SL, model weights, torch, fitting or GPU operations.
All six public-prefix arrays are replayed; seven EmissionLedger fields are an
additional check. Other saved emission diagnostics are not rederived here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import temporal_point_core as core
import temporal_prefix_deltas as codec

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
STAGE = "temporal_prefix_delta_saved64_qualification_v1"
VIEWS = ("top_cam", "side1_cam", "side2_cam")
WINDOWS = [[8 * i, 16 + 8 * i] for i in range(7)]
PLAN = WORK / "TEMPORAL_PREFIX_DELTA_QUALIFICATION_PLAN.md"
CORE_SHA = "3ec308ca7877338f8677c0703d6f180a9d5658c4858b9868451e1152b7d4e25b"
V2_SHA = "2461f383d7c69223c98cb9371058373361a7fb30a409a37d617b60fc2db53790"
KNOWN_SHA = "9db089624f62f2d55bd9f883f9ec6ca8408604dd3d61f1a084d38884837272af"
OFFICIAL = WORK / "assets/cotracker3_online_82e02e8_v2/source/cotracker"
PINS = {
    PLAN: "e27431c7b2ea6914b79beee5928bcad8e64d2728cdce37acc1040c58d8f067f3",
    WORK / "temporal_prefix_deltas.py": (
        "e164678a6e8900492d5a146a4b1c58f8edb360dd7455d178d0a25a591b6fef36"
    ),
    WORK / "temporal_window_stream.py": (
        "e46859d868be4b79d1ffc1698a24bbc6eb693b1e8f191570978bbde663a04048"
    ),
    WORK / "temporal_point_core.py": CORE_SHA,
    WORK / "compare_temporal_point64_v2.py": V2_SHA,
    WORK / "compare_temporal_known_gap64.py": KNOWN_SHA,
    OFFICIAL / "predictor.py": "783536d6c77790fa6c8e005f2df1d6bd4f0d8955c2c67d464bfb4e64d366375f",
    OFFICIAL / "models/core/cotracker/cotracker3_online.py": (
        "4c8bfce5546d85ba22c6b5292b870ad2aa0af1e0b6ba5a22e644c39bb496475e"
    ),
}
DATASETS = (
    dict(
        name="temporal_point64_comparison_v2",
        first=66483,
        seal="90b35bf8183367f432be4333f16816e58b34c526b387ea1799cce57683913e95",
        producer="compare_temporal_point64_v2.py",
        producer_sha=V2_SHA,
        plan_sha="f72e37547e0f9078446927aa85ea15bdb825188b76d62eedeb8f7db4ace4990c",
        queries=[180, 179, 111],
    ),
    dict(
        name="temporal_known_gap64_comparison_v1",
        first=333519,
        seal="bd0bae4df3da0d8f2f4a31f0e754a93e89460c6b72803d2054327655662e732e",
        producer="compare_temporal_known_gap64.py",
        producer_sha=KNOWN_SHA,
        plan_sha="3cd3c7ee8bd262daca3366a9e5d325bcb8bbec11f191e159f1557acbc726f3b2",
        queries=[157, 80, 97],
    ),
)
require = codec.require


def utc():
    return datetime.now(UTC).isoformat()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


class Evidence:
    def __init__(self):
        self.inputs = {}

    def read(self, path, expected):
        path = Path(path).resolve()
        raw = path.read_bytes()
        require(sha_bytes(raw) == expected, "input SHA differs: " + str(path))
        prior = self.inputs.setdefault(str(path), expected)
        require(prior == expected, "conflicting input SHA")
        return raw

    def guard(self):
        for path, digest in self.inputs.items():
            require(file_sha(path) == digest, "input changed: " + path)


def named_bytes(root, seal, role, evidence):
    """Read exactly one successful producer role; never traverse other outputs."""
    root = Path(root).resolve()
    path = (root / role).resolve()
    require(path.is_relative_to(root), "output role escaped producer")
    item = seal["outputs"][role]
    require(Path(item["path"]).resolve() == path, "named output path differs")
    require(type(item["size_bytes"]) is int and item["size_bytes"] >= 0, "output size schema")
    raw = evidence.read(path, item["sha256"])
    require(len(raw) == item["size_bytes"], "named output size differs")
    return raw


def arrays_from_bytes(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
        require(len(saved.files) == len(set(saved.files)), "duplicate NPZ array name")
        return {key: saved[key].copy() for key in saved.files}


def validate_scope(spec, plan, summary):
    first = spec["first"]
    frames = list(range(first, first + 64))
    require(plan["schema"] == summary["schema"] == spec["name"], "producer schema differs")
    require(summary["status"] == "ok", "producer summary not successful")
    require(plan["plan_sha256"] == spec["plan_sha"], "original plan binding differs")
    require(plan["frames"] == summary["frames"] == frames, "native origin/range differs")
    require(
        plan["views"] == list(VIEWS) and plan["windows"] == WINDOWS, "all views/windows required"
    )
    require(
        plan["init_relative_frame"] == 0 and plan["no_reseed"] is True,
        "initial query contract differs",
    )
    require(
        summary["native_frames"] == 64
        and summary["logical_online_requests"] == 21
        and summary["model_initializations"] == 3,
        "complete64 accounting differs",
    )
    require(
        all(
            summary[key] is True
            for key in ("no_query_replacement", "raw_scores_preserved", "all_prefixes_preserved")
        ),
        "prefix/query preservation contract differs",
    )
    require(
        [(v["view"], v["query_count"]) for v in summary["views"]]
        == list(zip(VIEWS, spec["queries"], strict=True)),
        "view query counts differ",
    )
    require(summary["query_count"] == sum(spec["queries"]), "total queries differ")
    require(
        [r["view"] for r in plan["all_query_rows"]] == list(VIEWS), "ordered query groups differ"
    )


def validate_call(meta, first, index, queries):
    codec.validate_metadata(meta)
    lo, hi = WINDOWS[index]
    require(
        meta["call_index"] == index
        and meta["input_relative"] == [lo, hi]
        and meta["input_native"] == [first + lo, first + hi]
        and meta["maximum_seen_native"] == first + hi - 1
        and meta["query_count"] == queries,
        "cross-call origin/order/query binding differs",
    )


def validate_queries(arrays, row, first, queries):
    layouts = dict(
        xy=((queries, 2), np.dtype("float32")),
        mask_index=((queries,), np.dtype("int64")),
        corner_index=((queries,), np.dtype("int64")),
        frame_idx=((), np.dtype("int64")),
        available=((), np.dtype("bool")),
    )
    require(set(arrays) == set(layouts), "five original query arrays required")
    for name, (shape, dtype) in layouts.items():
        require(
            arrays[name].shape == shape and arrays[name].dtype == dtype,
            "query schema differs: " + name,
        )
    require(
        int(arrays["frame_idx"]) == first == row["frame_idx"] and queries == row["query_count"],
        "initial query origin/count differs",
    )
    require(
        codec.signatures(arrays) == row["array_signatures"], "original query logical bytes differ"
    )


def load_dataset(spec, evidence):
    root = WORK / spec["name"]
    seal = parse_json(evidence.read(root / "comparison.provenance.json", spec["seal"]))
    require(
        seal["stage"] == spec["name"] and seal["status"] == "ok",
        "successful fixed producer required",
    )
    code = {
        WORK / "temporal_point_core.py": CORE_SHA,
        WORK / "compare_temporal_point64_v2.py": V2_SHA,
        WORK / spec["producer"]: spec["producer_sha"],
        **{p: s for p, s in PINS.items() if p.is_relative_to(OFFICIAL)},
    }
    for path, digest in code.items():
        require(seal["inputs"].get(str(path)) == digest, "producer source input differs")
        evidence.read(path, digest)
        role = "code_snapshot/" + path.relative_to(REPO).as_posix()
        raw = named_bytes(root, seal, role, evidence)
        require(sha_bytes(raw) == digest, "producer source snapshot differs")
    plan = parse_json(named_bytes(root, seal, "plan.json", evidence))
    summary = parse_json(named_bytes(root, seal, "summary.json", evidence))
    validate_scope(spec, plan, summary)
    return root, seal, plan


class Output:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.outputs = {}

    def register(self, role):
        path = self.root / role
        require(role not in self.outputs, "output already registered")
        self.outputs[role] = dict(
            path=str(path), sha256=file_sha(path), size_bytes=path.stat().st_size
        )
        return path

    def json(self, role, value):
        path = self.root / role
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        return self.register(role)

    def npz(self, role, arrays):
        path = self.root / role
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        return self.register(role)

    def guard(self):
        files = {p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()}
        require(files == set(self.outputs), "new output membership differs")
        for item in self.outputs.values():
            path = Path(item["path"])
            require(
                path.stat().st_size == item["size_bytes"] and file_sha(path) == item["sha256"],
                "new output changed",
            )


def exact_arrays(actual, expected, keys):
    for key in keys:
        require(
            key in actual
            and key in expected
            and codec.signature(actual[key]) == codec.signature(expected[key]),
            "reconstructed array differs: " + key,
        )
        require(
            actual[key].tobytes(order="C") == expected[key].tobytes(order="C"),
            "array bytes differ: " + key,
        )


def replay_view(spec, view, row, read, output):
    """Seven complete saved calls; next base is always the disk-reconstructed prefix."""
    first, queries = spec["first"], row["query_count"]
    query_raw = read(view + "/queries.npz")
    require(sha_bytes(query_raw) == row["sha256"], "copied original query file differs")
    query = arrays_from_bytes(query_raw)
    validate_queries(query, row, first, queries)
    ledger, previous, records = core.EmissionLedger(64, queries), None, []
    base = spec["name"] + "/" + view
    for index, (lo, hi) in enumerate(WINDOWS):
        stem = view + "/online/call_" + str(index)
        meta_raw, current_raw = read(stem + ".json"), read(stem + ".npz")
        meta, current = parse_json(meta_raw), arrays_from_bytes(current_raw)
        validate_call(meta, first, index, queries)
        header, payload = codec.encode_delta(previous, current, meta)
        role = base + "/call_" + str(index)
        hp = output.json(role + ".json", header)
        pp = output.npz(role + ".npz", payload)
        saved_header, saved_payload = (
            parse_json(hp.read_bytes()),
            arrays_from_bytes(pp.read_bytes()),
        )
        require(saved_header["call_metadata"] == meta, "all original call metadata differs")
        restored = codec.decode_delta(previous, saved_header, saved_payload)
        exact_arrays(restored, current, codec.FIELDS)
        ledger.add(restored["xy"], restored["reported"], input_start=lo, input_stop=hi)
        outside = {
            key: dict(
                relative=saved_payload[header["fields"][key]["outside_index_key"]].tolist(),
                native=(saved_payload[header["fields"][key]["outside_index_key"]] + first).tolist(),
            )
            for key in codec.FIELDS
        }
        records.append(
            dict(
                call_index=index,
                input_native=[first + lo, first + hi],
                source_metadata_sha256=sha_bytes(meta_raw),
                source_arrays_sha256=sha_bytes(current_raw),
                source_npz_bytes=len(current_raw),
                query_sha256=sha_bytes(query_raw),
                exact_six_fields=True,
                metadata_exact=True,
                full_prefix_logical_bytes=header["logical_bytes"],
                delta_logical_bytes=header["delta_array_bytes"],
                active_values_bytes=sum(
                    a.nbytes for k, a in payload.items() if k.endswith("__active")
                ),
                outside_indices_bytes=sum(
                    a.nbytes for k, a in payload.items() if k.endswith("__outside_indices")
                ),
                outside_values_bytes=sum(
                    a.nbytes for k, a in payload.items() if k.endswith("__outside_values")
                ),
                new_header_bytes=hp.stat().st_size,
                new_npz_bytes=pp.stat().st_size,
                outside_changed_rows=outside,
            )
        )
        previous = restored
        del current, current_raw, payload, saved_payload
    emissions = arrays_from_bytes(read(view + "/online_emissions.npz"))
    replayed = ledger.arrays()
    exact_arrays(replayed, emissions, tuple(replayed))
    output.npz(base + "/reconstructed_emissions.npz", replayed)
    output.json(
        base + "/replay.json",
        dict(
            view=view,
            first=first,
            query_count=queries,
            query_signatures=codec.signatures(query),
            original_query_rows=row,
            calls=records,
            ledger_fields=list(replayed),
            ledger_fields_exact=True,
            saved_emission_keys_not_rederived=sorted(set(emissions) - set(replayed)),
        ),
    )
    return dict(
        dataset=spec["name"],
        view=view,
        query_count=queries,
        ledger_fields_exact=True,
        calls=records,
    )


def run(output_root):
    start, tick = utc(), time.perf_counter()
    output = Output(output_root)
    evidence, completed = Evidence(), []
    location = "source preflight"
    try:
        for path, expected in {**PINS, Path(__file__).resolve(): file_sha(__file__)}.items():
            evidence.read(path, expected)
            role = "code_snapshot/" + path.relative_to(REPO).as_posix()
            dest = output.root / role
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            require(file_sha(dest) == expected, "source snapshot changed")
            output.register(role)
        require("torch" not in sys.modules, "model runtime unexpectedly imported")
        output.json(
            "plan.json",
            dict(
                stage=STAGE,
                datasets=DATASETS,
                calls=42,
                fields=252,
                windows=WINDOWS,
                selected_output_scope_only=True,
                source_pins={str(p): s for p, s in PINS.items()},
            ),
        )
        for spec in DATASETS:
            location = spec["name"] + " binding"
            root, seal, plan = load_dataset(spec, evidence)

            def read(role, root=root, seal=seal):
                return named_bytes(root, seal, role, evidence)

            for view, row in zip(VIEWS, plan["all_query_rows"], strict=True):
                location = spec["name"] + "/" + view
                completed.append(replay_view(spec, view, row, read, output))
                print(json.dumps(dict(completed_view=location, calls=7)), flush=True)
        calls = [c for v in completed for c in v["calls"]]
        require(len(completed) == 6 and len(calls) == 42, "fixed complete denominator differs")
        totals = {
            key: sum(c[key] for c in calls)
            for key in (
                "full_prefix_logical_bytes",
                "source_npz_bytes",
                "delta_logical_bytes",
                "active_values_bytes",
                "outside_indices_bytes",
                "outside_values_bytes",
                "new_header_bytes",
                "new_npz_bytes",
            )
        }
        output.json(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                datasets=2,
                views=6,
                calls=42,
                logical_prefix_fields=252,
                all_six_fields_byte_exact=True,
                all_call_metadata_exact=True,
                ledger_fields_checked=42,
                all_ledger_fields_byte_exact=True,
                storage=totals,
                view_results=completed,
                raw_decodes=0,
                model_forwards=0,
                GPU_used=False,
                actual_200_run=False,
                ID_truth=False,
                tracking_or_speed_improvement_claim=False,
                scope="Selected saved outputs only; original model success inherited, not rerun. "
                "Worst-case outside corrections and official full-prefix work remain quadratic.",
                python=platform.python_version(),
                numpy=np.__version__,
                seconds_before_final_guard=time.perf_counter() - tick,
            ),
        )
        evidence.guard()
        output.guard()
        status, error = "ok", None
    except BaseException as exc:
        status, error = "failed", dict(type=type(exc).__name__, message=str(exc), location=location)
        output.json("failure.json", dict(error=error, completed_views=len(completed)))
        output.json(
            "qualification.provenance.json",
            dict(
                stage=STAGE,
                status=status,
                started_at=start,
                finished_at=utc(),
                inputs=evidence.inputs,
                outputs=dict(output.outputs),
                error=error,
                final_guards_passed=False,
            ),
        )
        raise
    output.json(
        "qualification.provenance.json",
        dict(
            stage=STAGE,
            status=status,
            started_at=start,
            finished_at=utc(),
            inputs=evidence.inputs,
            outputs=dict(output.outputs),
            error=error,
            final_guards_passed=True,
        ),
    )
    print(
        json.dumps(
            dict(
                status=status,
                output=str(output.root),
                seal_sha256=file_sha(output.root / "qualification.provenance.json"),
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.output.resolve()
    require(path.is_relative_to(WORK) and path != WORK, "new output must be under work/three_body")
    run(path)


if __name__ == "__main__":
    main()

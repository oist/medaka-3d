"""Fixed64 anonymous point comparison using explicitly qualified frozen inputs.

Prepared PNG/observation/query inputs only; no MP4, SL loader, 3D, ID or GT.
Official online calls retain every prefix and read-only sigmoid visibility/confidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import temporal_point_core as core

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
STAGE = "temporal_point64_comparison_v1"
FIRST, STOP = 66483, 66547
FRAMES = tuple(range(FIRST, STOP))
VIEWS = ("top_cam", "side1_cam", "side2_cam")
WINDOWS = tuple((n, n + 16) for n in range(0, 49, 8))
PLAN = WORK / "TEMPORAL_POINT_64_PLAN.md"
PLAN_SHA = "f72e37547e0f9078446927aa85ea15bdb825188b76d62eedeb8f7db4ace4990c"
ASSETS = WORK / "assets/cotracker3_online_82e02e8_v2"
ASSET_SEAL_SHA = "2c516b4d508f4de588f3ca23f83580194439d43e741463627e32da3a472985f8"
ASSET_MANIFEST_SHA = "befdf1b5ba9a376abfddd5b62afd89b7064661c39f28bdb73b55932c0e253ae8"
WEIGHT_SHA = "205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218"
# Fixed successful preparation, source review and CPU load/init qualification.
CORE_SHA = "3ec308ca7877338f8677c0703d6f180a9d5658c4858b9868451e1152b7d4e25b"
PREPARATION_SOURCE_SHA = "42a5d6ca4e877eff0c61b7de7788136dbcba7008e23f13901db0fbb8eb8a27ba"
PREPARATION_SHA = "9771ee33cf36ba320cb2cf5b055c3c44b59f906d835ffac346fccb65241769f3"
MODEL_QUALIFICATION_SHA = "f42a9b9b73bfc479a400632b189dfe524087029bfeae79ab509e3f5ec1dc73ad"
DEPENDENCY_BINDING = {"numpy": "2.5.1", "opencv-python": "5.0.0.93", "torch": "2.13.0+cu130"}
QUALIFIER_SOURCE_SHA = "feb8b13adc59208bef6b56223c0bbdebd06f70d7ca7f2c13293d7c8e61eb84e0"
PREPARATION = WORK / "temporal_point64_preparation_v1"
QUALIFICATION = WORK / "cotracker3_cpu_load_qualification_v1"
SOURCE_REVIEW = WORK / "cotracker3_online_source_review_v1.json"
SOURCE_REVIEW_SHA = "5f68c1b668ff6786bb5665afe6ae6f8418401a303a1a81690f4aed2641acea11"
EXECUTION_READY = True
require = core.require


def utc():
    return datetime.now(UTC).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(value):
    a = np.asarray(value)
    return dict(
        dtype=a.dtype.str, shape=list(a.shape), sha256=hashlib.sha256(a.tobytes()).hexdigest()
    )


def same(a, b, name):
    require(signature(a) == signature(b), "array bytes changed: " + name)


def valid_sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def json_bytes(value):
    return (
        json.dumps(safe_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()


class Evidence:
    def __init__(self):
        self.paths = {}

    def add(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        require(path.is_relative_to(REPO) and path.is_file(), "input outside workspace")
        current = sha(path)
        require(expected is None or current == expected, "input SHA differs: " + str(path))
        require(str(path) not in self.paths or self.paths[str(path)] == current, "input changed")
        self.paths[str(path)] = current
        return current

    def final_guard(self):
        for path, digest in self.paths.items():
            require(sha(path) == digest, "final input guard failed: " + path)


class Output:
    def __init__(self, root):
        self.root, self.records = Path(root), {}
        self.root.mkdir(parents=True, exist_ok=False)

    def write(self, name, value):
        path = self.root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe output")
        require(name not in self.records and not path.exists(), "output already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            if name.endswith(".npz"):
                require(
                    all(np.asarray(a).dtype.kind != "O" for a in value.values()), "object array"
                )
                np.savez_compressed(f, **value)
            else:
                f.write(value if isinstance(value, bytes) else json_bytes(value))
        self.records[name] = dict(
            path=str(path.resolve()), sha256=sha(path), size_bytes=path.stat().st_size
        )
        return self.records[name]

    def final_guard(self):
        actual = {
            str(p.relative_to(self.root)).replace("\\", "/")
            for p in self.root.rglob("*")
            if p.is_file()
        }
        require(actual == set(self.records), "output membership changed")
        for row in self.records.values():
            require(sha(row["path"]) == row["sha256"], "output bytes changed")


class CoreForwardRecorder:
    """A PyTorch forward hook: clone values and return None, preserving the output."""

    def __init__(self):
        self.records = []

    def __call__(self, module, arguments, output):
        require(isinstance(output, tuple) and len(output) == 4, "official core tuple changed")
        require(output[3] is None, "training-mode core output is not permitted")
        coordinates, visibility, confidence = output[:3]
        require(coordinates.ndim == 4 and coordinates.shape[-1] == 2, "core coordinate shape")
        require(visibility.shape == confidence.shape == coordinates.shape[:-1], "core score shape")
        self.records.append(tuple(value.detach().clone() for value in output[:3]))
        return None


def validate_prefix(result, prefix, queries):
    expected = {
        "xy": ((prefix, queries, 2), np.float32),
        "reported": ((prefix, queries), np.bool_),
        "model_xy": ((prefix, queries, 2), np.float32),
        "sigmoid_visibility": ((prefix, queries), np.float32),
        "sigmoid_confidence": ((prefix, queries), np.float32),
        "product": ((prefix, queries), np.float32),
    }
    require(set(result) == set(expected), "online prefix fields changed")
    for name, (shape, dtype) in expected.items():
        a = result[name]
        require(
            isinstance(a, np.ndarray) and a.shape == shape and a.dtype == dtype,
            "online schema: " + name,
        )
    same(result["reported"], result["product"] > np.float32(0.6), "official product flag")
    # Product was computed and compared to public flags on its execution device.
    # CPU re-multiplication need not share CUDA subnormal handling.


def run_online(frames, queries, backend, save_call, *, before_call=lambda _: None):
    """One init at frame0, seven ordered full prefixes; backend owns tensor transfer."""
    require(len(frames) == 64, "fixed64 frames required")
    require(
        queries.dtype == np.float32 and queries.ndim == 2 and queries.shape[1] == 2, "query f32 N,2"
    )
    original = signature(queries)
    n = len(queries)
    ledger = core.EmissionLedger(64, n)
    raw_first = {
        k: np.full((64, n), np.nan, np.float32)
        for k in ("sigmoid_visibility", "sigmoid_confidence", "product")
    }
    last_change = np.full((64, n), -1, np.int64)
    seen = np.zeros(64, bool)
    previous_xy = np.full((64, n, 2), np.nan, np.float32)
    previous_flag = np.zeros((64, n), bool)
    calls = []
    if n:
        before_call(-1)
        require(
            backend.initialize(frames[0:1], queries) == (None, None),
            "initialization returned predictions",
        )
    for index, (lo, hi) in enumerate(WINDOWS):
        before_call(index)
        if n:
            result, timing = backend.predict(frames[lo:hi])
        else:
            result = {
                "xy": np.empty((hi, 0, 2), np.float32),
                "model_xy": np.empty((hi, 0, 2), np.float32),
                "reported": np.empty((hi, 0), bool),
                **{k: np.empty((hi, 0), np.float32) for k in raw_first},
            }
            timing = dict(
                forward_seconds=0.0, input_transfer_seconds=0.0, capture_transfer_seconds=0.0
            )
        validate_prefix(result, hi, n)
        changed_xy = np.any(
            result["xy"].view(np.uint32) != previous_xy[:hi].view(np.uint32), axis=-1
        )
        changed = changed_xy | (result["reported"] != previous_flag[:hi]) | ~seen[:hi, None]
        last_change[:hi][changed] = index
        fresh = np.flatnonzero(~seen[:hi])
        for k in raw_first:
            raw_first[k][fresh] = result[k][fresh]
        ledger.add(result["xy"], result["reported"], input_start=lo, input_stop=hi)
        previous_xy[:hi], previous_flag[:hi], seen[:hi] = result["xy"], result["reported"], True
        meta = dict(
            call_index=index,
            input_relative=[lo, hi],
            input_native=[FIRST + lo, FIRST + hi],
            maximum_seen_native=FIRST + hi - 1,
            output_prefix=hi,
            query_count=n,
            forward_called=bool(n),
            status="ok" if n else "no_queries",
            **timing,
        )
        save_call(meta, {k: v.copy() for k, v in result.items()})
        calls.append(meta)
    require(seen.all() and len(calls) == 7, "missing full online prefix")
    require(signature(queries) == original, "backend mutated initial queries")
    arrays = ledger.arrays()
    for name in raw_first:
        arrays["first_" + name] = raw_first[name]
        arrays["final_" + name] = result[name].copy()
    arrays["last_value_changed_call"] = last_change
    arrays["first_lookahead_frames"] = arrays["first_horizon"] - np.arange(64, dtype=np.int64)
    require(
        np.array_equal(
            arrays["first_horizon"], np.r_[np.full(16, 15), np.repeat(np.arange(23, 64, 8), 8)]
        ),
        "first emission horizon changed",
    )
    return arrays, calls


def finite_distribution(values, eligible=None):
    values = np.asarray(values)
    selected = np.isfinite(values)
    if eligible is not None:
        selected &= eligible
    data = values[selected]
    return dict(
        denominator=int(values.size),
        finite_count=len(data),
        minimum=None if not len(data) else float(data.min()),
        median=None if not len(data) else float(np.median(data)),
        mean=None if not len(data) else float(data.mean()),
        maximum=None if not len(data) else float(data.max()),
    )


def missing_runs(flags):
    require(flags.ndim == 2 and flags.dtype == bool, "T,N flags required")
    first = np.full(flags.shape[1], -1, np.int64)
    longest = np.zeros(flags.shape[1], np.int64)
    returns = np.zeros(flags.shape[1], np.int64)
    for q in range(flags.shape[1]):
        missing = np.flatnonzero(~flags[1:, q]) + 1
        if len(missing):
            first[q] = FIRST + int(missing[0])
        current = 0
        for good in flags[1:, q]:
            current = 0 if good else current + 1
            longest[q] = max(longest[q], current)
        returns[q] = np.count_nonzero(~flags[:-1, q] & flags[1:, q])
    return dict(first_missing_native=first, longest_missing_frames=longest, false_to_true=returns)


def diagnose_method(frames, masks, queries, mask_index, groups, xy, reported, save_frame):
    """All native/query fields and all current N mask columns, without tracking veto."""
    n = len(queries)
    require(xy.shape == (64, n, 2) and reported.shape == (64, n), "diagnostic full64 scope")
    patches = core.patch_diagnostics(frames, xy)
    # Shared immutable anchor is the prepared query, not the method's frame0 prediction.
    reference = core.sample_patches(frames[0], queries)["values"]
    for t, frame in enumerate(frames):
        current = core.sample_patches(frame, xy[t])["values"]
        for key, value in core.patch_comparison(reference, current).items():
            patches["initial_" + key][t] = value
    finite = np.isfinite(xy).all(-1)
    steps = np.full((64, n), np.nan, np.float64)
    acceleration = steps.copy()
    steps[1:] = np.linalg.norm(xy[1:].astype(np.float64) - xy[:-1], axis=-1)
    acceleration[2:] = np.linalg.norm(
        xy[2:].astype(np.float64) - 2 * xy[1:-1].astype(np.float64) + xy[:-2], axis=-1
    )
    inside, sampled = np.zeros((64, n), bool), np.zeros((64, n), bool)
    union = np.zeros((64, n), bool)
    mask_available = np.zeros(64, bool)
    mask_counts = np.full(64, -1, np.int64)
    for t in range(64):
        membership = core.mask_memberships(xy[t], masks[t], frames[t].shape[:2])
        inside[t], sampled[t] = membership["in_image"], membership["sampled"]
        mask_available[t] = membership["available"]
        if membership["available"]:
            mask_counts[t] = membership["membership"].shape[1]
            union[t] = membership["membership"].any(axis=1)
        counts = (
            np.stack(
                [membership["membership"][mask_index == g].sum(axis=0) for g in range(len(groups))],
                axis=0,
            )
            if groups
            else np.zeros((0, membership["membership"].shape[1]), np.int64)
        )
        reported_counts = (
            np.stack(
                [
                    membership["membership"][(mask_index == g) & reported[t]].sum(axis=0)
                    for g in range(len(groups))
                ],
                axis=0,
            )
            if groups
            else counts.copy()
        )
        # Ties at zero remain explicit; a group with no members is not a match.
        for key, value in (
            ("all_query_group_mask_count", counts),
            ("reported_group_mask_count", reported_counts),
        ):
            membership[key] = value.astype(np.int64)
            membership[key + "_argmax_ties"] = (
                value == value.max(axis=1, keepdims=True)
                if value.shape[1]
                else np.zeros(value.shape, bool)
            )
            membership[key + "_positive"] = value.any(axis=1)
        save_frame(t, membership)
    result = dict(
        xy=xy.copy(),
        reported=reported.copy(),
        finite=finite,
        continuous_inside=inside,
        rounded_sampled=sampled,
        mask_available=mask_available,
        mask_count=mask_counts,
        union_membership=union,
        step_px=steps,
        second_difference_px=acceleration,
        **missing_runs(reported),
        **patches,
    )
    return result


def pair_geometry(xy, initial_xy, mask_index):
    pairs = np.asarray(
        [
            (i, j)
            for i in range(len(initial_xy))
            for j in range(i + 1, len(initial_xy))
            if mask_index[i] == mask_index[j]
        ],
        np.int64,
    ).reshape(-1, 2)
    initial = np.linalg.norm(
        initial_xy[pairs[:, 0]].astype(np.float64) - initial_xy[pairs[:, 1]], axis=-1
    )
    distance = np.linalg.norm(xy[:, pairs[:, 0]].astype(np.float64) - xy[:, pairs[:, 1]], axis=-1)
    return dict(
        query_pair=pairs,
        initial_distance_px=initial,
        distance_px=distance,
        distance_change_px=distance - initial[None],
        initial_distance_zero=initial == 0,
    )


def group_summary(result, mask_index, group_rows):
    rows = []
    for group in [None, *range(len(group_rows))]:
        selected = np.ones(len(mask_index), bool) if group is None else mask_index == group
        flags = result["reported"][:, selected]
        row = dict(
            initial_mask_row=group,
            query_count=int(selected.sum()),
            all64_denominator=int(flags.size),
            tracking63_denominator=int(flags[1:].size),
            reported63=int(flags[1:].sum()),
            finite63=int(result["finite"][1:, selected].sum()),
            input_group=None if group is None else group_rows[group],
        )
        row["metrics"] = {
            name: dict(
                all_finite=finite_distribution(result[name][1:, selected]),
                reported=finite_distribution(result[name][1:, selected], flags[1:]),
            )
            for name in (
                "initial_mad",
                "initial_zncc",
                "adjacent_mad",
                "adjacent_zncc",
                "step_px",
                "second_difference_px",
            )
        }
        rows.append(row)
    return rows


def execution_gate():
    require(
        EXECUTION_READY is True
        and all(
            valid_sha(x)
            for x in (
                CORE_SHA,
                PREPARATION_SOURCE_SHA,
                PREPARATION_SHA,
                MODEL_QUALIFICATION_SHA,
                QUALIFIER_SOURCE_SHA,
            )
        )
        and isinstance(DEPENDENCY_BINDING, dict),
        "draft: frozen preparation/core/model qualification required",
    )


def relative(root, name):
    item = Path(name)
    require(
        type(name) is str and not item.is_absolute() and ".." not in item.parts,
        "unsafe relative input",
    )
    result = (root / item).resolve(strict=True)
    require(result.is_relative_to(root.resolve()), "input escaped sealed root")
    return result


def read_json(path):
    return json.loads(Path(path).read_bytes())


def verified(root, seal_name, stage, expected_sha, evidence):
    root = Path(root).resolve(strict=True)
    seal = root / seal_name
    evidence.add(seal, expected_sha)
    record = read_json(seal)
    require(
        record.get("stage") == stage and record.get("status") == "ok" and record.get("finished_at"),
        "successful completed input required",
    )
    members = {}
    for row in record["outputs"].values():
        path = Path(row["path"]).resolve(strict=True)
        require(
            path.is_relative_to(root) and path != seal and str(path) not in members,
            "sealed output membership conflict",
        )
        evidence.add(path, row["sha256"])
        require(path.stat().st_size == row["size_bytes"], "sealed output size differs")
        members[str(path)] = row
    require(
        {str(p.resolve()) for p in root.rglob("*") if p.is_file()} == set(members) | {str(seal)},
        "sealed file set differs",
    )
    return record, members


def bound_json(root, members, name):
    path = relative(root, name)
    require(str(path) in members, "required JSON absent from successful seal")
    return read_json(path)


def validate_preparation(plan, summary, natives, queries):
    require(
        plan["schema"] == "temporal_point64_preparation_v1"
        and plan["plan_sha256"] == PLAN_SHA
        and plan["frames"] == list(FRAMES)
        and plan["views"] == list(VIEWS)
        and plan["native_interval"] == [FIRST, STOP]
        and plan["query_core_sha256"] == CORE_SHA,
        "fixed prepared scope/core differs",
    )
    require(
        [(r["frame_idx"], r["view"]) for r in natives] == [(n, v) for n in FRAMES for v in VIEWS],
        "all192 ordered source rows required",
    )
    require(
        [r["view"] for r in queries] == list(VIEWS)
        and all(r["frame_idx"] == FIRST for r in queries),
        "all3 initial query rows required",
    )
    require(
        summary["status"] == "ok"
        and summary["native_frames"] == 64
        and summary["raw_video_delivered_images"] == 192
        and summary["decoded_mask_stacks"] == 192
        and summary["initial_mask_rows"] == 13
        and summary["query_files"] == 3
        and summary["model_loaded"] is False
        and summary["model_forward_calls"] == 0
        and summary["SL_arrays_decoded"] is False
        and summary["candidate_arrays_read"] is False
        and summary["ID_or_review_changed"] is False,
        "prepared source scope differs",
    )
    require(
        sum(r["mask_count"] or 0 for r in queries) == 13
        and sum(r["query_count"] for r in queries) == summary["query_count"] <= 832,
        "full initial mask/query denominator differs",
    )
    for row in queries:
        require(row["config"] == dataclasses.asdict(core.CONFIG), "prepared corner settings differ")


def validate_model_qualification(summary):
    require(
        summary.get("schema") == "cotracker3_cpu_load_qualification_v1"
        and summary.get("status") == "ok"
        and summary.get("passed") is True
        and summary.get("asset_seal_sha256") == ASSET_SEAL_SHA
        and summary.get("asset_manifest_sha256") == ASSET_MANIFEST_SHA
        and summary.get("weights_only") is True
        and summary.get("strict_load") is True
        and summary.get("model_forward_calls") == 0
        and summary.get("GPU_used") is False
        and summary.get("initialization_checked") is True
        and summary.get("missing_keys") == []
        and summary.get("unexpected_keys") == [],
        "qualified local CPU loading/initialization contract required",
    )
    require(
        summary["parameter_count"] == 25385700
        and summary["buffer_count"] == 17760
        and summary["cpu_threads"] == 2
        and summary["cpu_interop_threads"] == 1
        and summary["torch_version"] == DEPENDENCY_BINDING["torch"]
        and summary["numpy_version"] == DEPENDENCY_BINDING["numpy"],
        "CPU qualification count/runtime differs",
    )


def load_inputs(evidence):
    evidence.add(PLAN, PLAN_SHA)
    evidence.add(core.__file__, CORE_SHA)
    evidence.add(WORK / "prepare_temporal_point64.py", PREPARATION_SOURCE_SHA)
    evidence.add(WORK / "qualify_cotracker3_cpu.py", QUALIFIER_SOURCE_SHA)
    record, members = verified(
        PREPARATION,
        "preparation.provenance.json",
        "temporal_point64_preparation_v1",
        PREPARATION_SHA,
        evidence,
    )
    plan, summary, natives, queries = (
        bound_json(PREPARATION, members, name)
        for name in ("plan.json", "summary.json", "native_evidence.json", "query_rows.json")
    )
    validate_preparation(plan, summary, natives, queries)
    for p, digest in (
        (PLAN, PLAN_SHA),
        (Path(core.__file__).resolve(), CORE_SHA),
        (WORK / "prepare_temporal_point64.py", PREPARATION_SOURCE_SHA),
    ):
        require(
            plan["source_hashes"].get(str(p)) == digest and record["inputs"].get(str(p)) == digest,
            "prepared direct source binding differs",
        )
    for row in natives:
        for name, digest in (
            ("png_path", "png_sha256"),
            ("observation_path", "observation_sha256"),
        ):
            path = relative(PREPARATION, row[name])
            require(
                str(path) in members and members[str(path)]["sha256"] == row[digest],
                "native row not sealed",
            )
    for row in queries:
        path = relative(PREPARATION, row["path"])
        require(
            str(path) in members and members[str(path)]["sha256"] == row["sha256"],
            "query row not sealed",
        )
    asset_record, asset_members = verified(
        ASSETS,
        "acquisition.provenance.json",
        "official_cotracker3_online_asset_acquisition_v2",
        ASSET_SEAL_SHA,
        evidence,
    )
    manifest = bound_json(ASSETS, asset_members, "assets.manifest.json")
    require(
        sha(ASSETS / "assets.manifest.json") == ASSET_MANIFEST_SHA
        and manifest["upstream_weight_sha256"] == WEIGHT_SHA
        and manifest["upstream_weight_size_bytes"] == 101695610,
        "fixed online model asset differs",
    )
    evidence.add(SOURCE_REVIEW, SOURCE_REVIEW_SHA)
    closure = read_json(SOURCE_REVIEW)
    require(
        closure["asset_seal_sha256"] == ASSET_SEAL_SHA
        and closure["asset_manifest_sha256"] == ASSET_MANIFEST_SHA,
        "official import closure binding differs",
    )
    for item in closure["closure"].values():
        path = relative(ASSETS, item["path"])
        require(
            str(path) in asset_members and asset_members[str(path)]["sha256"] == item["sha256"],
            "unsealed official module",
        )
    qualification, qmembers = verified(
        QUALIFICATION,
        "qualification.provenance.json",
        "cotracker3_cpu_load_qualification_v1",
        MODEL_QUALIFICATION_SHA,
        evidence,
    )
    qualification_summary = bound_json(QUALIFICATION, qmembers, "summary.json")
    validate_model_qualification(qualification_summary)
    require(
        qualification["inputs"].get(str(WORK / "qualify_cotracker3_cpu.py"))
        == QUALIFIER_SOURCE_SHA,
        "CPU loader qualification source differs",
    )
    return dict(
        plan=plan,
        summary=summary,
        natives=natives,
        queries=queries,
        source_closure=closure,
        preparation=record,
        asset_record=asset_record,
        qualification=qualification_summary,
        opencv_threads=record["environment"]["opencv_threads"],
    )


def load_view(context, view):
    query_row = context["queries"][VIEWS.index(view)]
    with np.load(relative(PREPARATION, query_row["path"]), allow_pickle=False) as z:
        require(
            len(z.files) == len(set(z.files))
            and set(z.files) == {"xy", "mask_index", "corner_index", "frame_idx", "available"},
            "query archive fields differ",
        )
        query = {k: z[k].copy() for k in z.files}
    require(
        {k: signature(v) for k, v in query.items()} == query_row["array_signatures"],
        "initial query bytes differ",
    )
    xy, origin, corner = query["xy"], query["mask_index"], query["corner_index"]
    require(
        xy.dtype == np.float32
        and xy.shape == (query_row["query_count"], 2)
        and np.isfinite(xy).all()
        and origin.dtype == corner.dtype == np.int64
        and origin.shape == corner.shape == (len(xy),)
        and query["frame_idx"].dtype == np.int64
        and query["frame_idx"].shape == ()
        and int(query["frame_idx"]) == FIRST
        and query["available"].dtype == bool
        and query["available"].shape == (),
        "query numerical schema differs",
    )
    cursor = 0
    for i, group in enumerate(query_row["rows"]):
        lo, hi = group["query_start"], group["query_stop"]
        require(
            group["mask_index"] == i
            and lo == cursor
            and hi - lo == group["query_count"] <= 64
            and np.all(origin[lo:hi] == i)
            and np.array_equal(corner[lo:hi], np.arange(hi - lo, dtype=np.int64)),
            "query group order changed",
        )
        cursor = hi
    require(cursor == len(xy), "omitted initial query")
    frames, masks, before = [], [], []
    rows = [r for r in context["natives"] if r["view"] == view]
    for row in rows:
        image = cv2.imread(str(relative(PREPARATION, row["png_path"])), cv2.IMREAD_COLOR)
        require(
            isinstance(image, np.ndarray)
            and image.dtype == np.uint8
            and image.shape == (1200, 1200, 3)
            and signature(image)["sha256"] == row["bgr_sha256"],
            "prepared PNG native bytes differ",
        )
        with np.load(relative(PREPARATION, row["observation_path"]), allow_pickle=False) as z:
            require(
                len(z.files) == len(set(z.files))
                and set(z.files) == set(row["observation_array_signatures"]),
                "observation member set differs",
            )
            obs = {k: z[k].copy() for k in ("frame_idx", "masks", "mask_available")}
        require(
            all(signature(v) == row["observation_array_signatures"][k] for k, v in obs.items()),
            "prepared observed bytes differ",
        )
        require(
            obs["frame_idx"].shape == ()
            and obs["frame_idx"].dtype == np.int64
            and int(obs["frame_idx"]) == row["frame_idx"]
            and obs["mask_available"].shape == ()
            and obs["mask_available"].dtype == bool
            and bool(obs["mask_available"]) is row["mask_available"]
            and obs["masks"].dtype == bool
            and obs["masks"].shape == (row["mask_count"] or 0, 1200, 1200)
            and signature(obs["masks"])["sha256"] == row["mask_sha256"],
            "native mask schema differs",
        )
        frames.append(image)
        masks.append(obs["masks"] if row["mask_available"] else None)
        before.append(
            dict(image=signature(image), masks=None if masks[-1] is None else signature(masks[-1]))
        )
    require(len(frames) == 64, "view lost native frames")
    return frames, masks, query, query_row, before


class TorchOnline:
    """Use the qualified official loader; only one16-frame RGB tensor lives on device."""

    def __init__(self, device, closure):
        import qualify_cotracker3_cpu as qualification
        import torch

        self.torch, self.device = torch, device
        require(
            sha(qualification.__file__) == QUALIFIER_SOURCE_SHA, "qualified loader source changed"
        )
        started = time.perf_counter()
        self.predictor, self.load_report = qualification.load_cpu_predictor(
            ASSETS, torch_module=torch
        )
        for name, row in closure["closure"].items():
            module = sys.modules.get(name)
            require(
                module is not None
                and Path(module.__file__).resolve() == relative(ASSETS, row["path"]),
                "official module origin changed",
            )
            require(sha(module.__file__) == row["sha256"], "official module bytes changed")
        actual = {
            name for name in sys.modules if name == "cotracker" or name.startswith("cotracker.")
        }
        require(actual == set(closure["closure"]), "unexpected official import closure")
        torch.manual_seed(0)
        self.predictor.eval()
        require(
            self.predictor.step == 8
            and tuple(self.predictor.interp_shape) == (384, 512)
            and self.predictor.model.window_len == 16
            and self.predictor.v2 is False,
            "official online configuration differs",
        )
        require(
            all(
                p.device.type == "cpu" and p.dtype == torch.float32
                for p in self.predictor.parameters()
            ),
            "qualified CPU float32 model required",
        )
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.25, 0)
            torch.cuda.reset_peak_memory_stats(0)
        self.predictor.to(device)
        self.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.recorder = CoreForwardRecorder()
        self.handle = self.predictor.model.register_forward_hook(self.recorder)
        self.initializations, self.forwards = 0, 0

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize(0)

    def tensor(self, images):
        started = time.perf_counter()
        array = np.stack([image[..., ::-1] for image in images]).astype(np.float32)
        result = self.torch.from_numpy(np.ascontiguousarray(array.transpose(0, 3, 1, 2)))[None].to(
            self.device
        )
        self.synchronize()
        return result, time.perf_counter() - started

    def initialize(self, frames, queries):
        require(
            self.initializations == 0 and len(frames) == 1,
            "oneframe single initialization required",
        )
        tensor, transfer = self.tensor(frames)
        q = np.concatenate([np.zeros((len(queries), 1), np.float32), queries], axis=1)[None]
        initial = self.torch.from_numpy(q.copy()).to(self.device)
        with self.torch.inference_mode():
            result = self.predictor(
                tensor, is_first_step=True, queries=initial, grid_size=0, add_support_grid=False
            )
        self.synchronize()
        require(not self.recorder.records, "initialization unexpectedly ran core forward")
        same(initial.detach().cpu().numpy(), q, "official query input")
        self.initializations += 1
        self.initialization_transfer_seconds = transfer
        return result

    def predict(self, frames):
        require(
            self.initializations == 1 and len(frames) == 16 and not self.recorder.records,
            "online schedule/state differs",
        )
        tensor, transfer = self.tensor(frames)
        self.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            require(
                not self.torch.is_autocast_enabled("cuda")
                and not self.torch.is_autocast_enabled("cpu"),
                "autocast is not permitted",
            )
            xy, reported = self.predictor(
                tensor, is_first_step=False, grid_size=0, add_support_grid=False
            )
        self.synchronize()
        forward = time.perf_counter() - started
        require(len(self.recorder.records) == 1, "exactly one recorded core forward required")
        model_xy, visibility, confidence = self.recorder.records.pop()
        for a in (model_xy, visibility, confidence, xy):
            require(a.dtype == self.torch.float32, "official output is not float32")
        require(
            reported.dtype == self.torch.bool and model_xy.shape[0] == 1,
            "official batch/public flag schema differs",
        )
        started = time.perf_counter()
        product = visibility * confidence
        native_xy = model_xy * model_xy.new_tensor([(1200 - 1) / (512 - 1), (1200 - 1) / (384 - 1)])
        require(self.torch.equal(reported, product > 0.6), "public/raw score flags differ")
        arrays = dict(
            xy=xy[0].detach().cpu().numpy().copy(),
            reported=reported[0].detach().cpu().numpy().copy(),
            model_xy=model_xy[0].cpu().numpy().copy(),
            sigmoid_visibility=visibility[0].cpu().numpy().copy(),
            sigmoid_confidence=confidence[0].cpu().numpy().copy(),
            product=product[0].cpu().numpy().copy(),
        )
        same(arrays["xy"], native_xy[0].cpu().numpy(), "official native reverse scale")
        self.synchronize()
        capture = time.perf_counter() - started
        self.forwards += 1
        return arrays, dict(
            forward_seconds=forward,
            input_transfer_seconds=transfer,
            capture_transfer_seconds=capture,
        )

    def close(self):
        self.handle.remove()
        self.recorder.records.clear()
        memory = dict(max_allocated_bytes=None, max_reserved_bytes=None)
        if self.device == "cuda":
            self.synchronize()
            memory.update(
                max_allocated_bytes=self.torch.cuda.max_memory_allocated(0),
                max_reserved_bytes=self.torch.cuda.max_memory_reserved(0),
            )
        del self.predictor
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
        return memory


def comparison_arrays(a, b):
    result = {}
    paired = core.paired_metrics(
        {k: a[k] for k in ("initial_mad", "initial_zncc", "adjacent_mad", "adjacent_zncc")},
        {k: b[k] for k in ("initial_mad", "initial_zncc", "adjacent_mad", "adjacent_zncc")},
        a["reported"],
        b["reported"],
    )
    for name, fields in paired.items():
        for key, value in fields.items():
            result[name + "__" + key] = value
    finite = a["finite"] & b["finite"]
    distance = np.full(a["finite"].shape, np.nan, np.float64)
    distance[finite] = np.linalg.norm(a["xy"][finite].astype(np.float64) - b["xy"][finite], axis=-1)
    result.update(
        position_distance_px=distance,
        common_finite=finite,
        common_reported=finite & a["reported"] & b["reported"],
        a_reported_only=a["reported"] & ~b["reported"],
        b_reported_only=b["reported"] & ~a["reported"],
    )
    return result


def run_view(context, view, output, device, memory_rows, *, backend_factory=TorchOnline):
    begin = time.perf_counter()
    frames, masks, query, row, before = load_view(context, view)
    input_seconds = time.perf_counter() - begin
    query_before = {k: signature(v) for k, v in query.items()}
    xy, origin, groups = query["xy"], query["mask_index"], row["rows"]
    backend = None
    prefix = view + "/"
    output.write(prefix + "queries.npz", query)
    started = time.perf_counter()
    lk = core.lk_chain(frames, xy)
    lk_seconds = time.perf_counter() - started
    output.write(prefix + "lk_raw.npz", lk)

    def memory_guard(call):
        if device == "cuda" and len(xy):
            sample = gpu_memory()
            memory_rows.append(dict(view=view, call_index=call, **sample))
            check_memory(sample)

    def save_call(meta, arrays):
        output.write(prefix + f"online/call_{meta['call_index']}.npz", arrays)
        output.write(prefix + f"online/call_{meta['call_index']}.json", meta)

    try:
        if len(xy):
            memory_guard(-2)
            backend = backend_factory(device, context["source_closure"])
        online, calls = run_online(frames, xy, backend, save_call, before_call=memory_guard)
        if backend is not None:
            require(
                backend.initializations == 1 and backend.forwards == 7,
                "model invocation denominator differs",
            )
            output.write(
                prefix + "model_load.json",
                dict(
                    report=backend.load_report,
                    load_seconds=backend.load_seconds,
                    initialization_input_transfer_seconds=backend.initialization_transfer_seconds,
                    seed=0,
                    seed_set_after_official_import=True,
                ),
            )
    finally:
        if backend is not None:
            output.write(prefix + "memory_peak.json", backend.close())
    output.write(prefix + "online_emissions.npz", online)
    output.write(
        prefix + "release.npz",
        dict(
            native_frame=np.asarray(FRAMES, np.int64),
            ct_first_horizon_native=online["first_horizon"] + FIRST,
            lk_unbuffered_horizon_native=np.asarray(FRAMES, np.int64),
            lk_buffered_release_native=online["first_horizon"] + FIRST,
            ct_first_delay_frames=online["first_lookahead_frames"],
            ct_first_delay_seconds=online["first_lookahead_frames"].astype(np.float64) / 166,
            initial_query_frame=np.arange(64) == 0,
        ),
    )
    arms = {
        "lk": (lk["xy"], lk["valid"]),
        "ct_first": (online["first_xy"], online["first_visibility"]),
        "ct_final": (online["final_xy"], online["final_visibility"]),
        "stationary": (np.broadcast_to(xy, (64, len(xy), 2)).copy(), np.ones((64, len(xy)), bool)),
    }
    diagnostics, summaries = {}, {}
    started = time.perf_counter()
    for name, (poses, reported) in arms.items():

        def membership_save(t, values, name=name):
            arrays = {k: np.asarray(v) for k, v in values.items()}
            arrays["native_frame"] = np.asarray(FIRST + t, np.int64)
            output.write(prefix + f"{name}/membership_{FIRST + t}.npz", arrays)

        metrics = diagnose_method(
            frames, masks, xy, origin, groups, poses, reported, membership_save
        )
        diagnostics[name] = metrics
        output.write(prefix + name + "/metrics.npz", metrics)
        output.write(
            prefix + name + "/within_initial_mask_pairs.npz", pair_geometry(poses, xy, origin)
        )
        summaries[name] = group_summary(metrics, origin, groups)
    pair_summaries = {}
    for a, b in (
        ("lk", "ct_first"),
        ("lk", "ct_final"),
        ("stationary", "ct_first"),
        ("stationary", "ct_final"),
        ("ct_first", "ct_final"),
    ):
        values = comparison_arrays(diagnostics[a], diagnostics[b])
        output.write(prefix + f"paired_{a}_{b}.npz", values)
        pair_summaries[a + "__" + b] = []
        for group in [None, *range(len(groups))]:
            chosen = np.ones(len(origin), bool) if group is None else origin == group
            pair_summaries[a + "__" + b].append(
                dict(
                    initial_mask_row=group,
                    tracking63_denominator=63 * int(chosen.sum()),
                    flag_counts={
                        k: int(v[1:, chosen].sum()) for k, v in values.items() if v.dtype == bool
                    },
                    metrics={
                        k: finite_distribution(v[1:, chosen])
                        for k, v in values.items()
                        if v.dtype.kind == "f"
                    },
                )
            )
    for i in range(64):
        require(
            signature(frames[i]) == before[i]["image"]
            and (None if masks[i] is None else signature(masks[i])) == before[i]["masks"],
            "input frame/mask mutated",
        )
    require(
        {k: signature(v) for k, v in query.items()} == query_before,
        "initial query artifact mutated",
    )
    summary = dict(
        view=view,
        query_count=len(xy),
        initial_mask_rows=groups,
        all64_query_denominator=64 * len(xy),
        tracking63_query_denominator=63 * len(xy),
        model_initializations=int(bool(len(xy))),
        model_forward_calls=7 * int(bool(len(xy))),
        online_requests=7,
        lk_edges=63,
        first_emission_primary=True,
        unique_xy_count=row["unique_xy_count"],
        repeated_xy_count=row["repeated_xy_count"],
        methods=summaries,
        paired=pair_summaries,
        timings=dict(
            input_decode_seconds=input_seconds,
            lk_seconds=lk_seconds,
            diagnostics_with_save_sha_seconds=time.perf_counter() - started,
            total_view_seconds=time.perf_counter() - begin,
        ),
        calls=calls,
    )
    output.write(prefix + "summary.json", summary)
    return summary


def loaded_sources(evidence, output, captured):
    paths = {
        Path(__file__).resolve(),
        Path(core.__file__).resolve(),
        PLAN,
        WORK / "qualify_cotracker3_cpu.py",
        WORK / "prepare_temporal_point64.py",
        SOURCE_REVIEW,
    }
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.is_relative_to(REPO) and ".venv" not in path.parts and path.suffix == ".py":
                paths.add(path)
    for path in sorted(paths):
        digest = evidence.add(path)
        if str(path) in captured:
            require(captured[str(path)] == digest, "loaded source changed")
            continue
        captured[str(path)] = digest
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == digest, "source changed while snapshotting")
        output.write("code_snapshot/" + path.relative_to(REPO).as_posix(), data)


def run(args):
    execution_gate()  # Deliberately before mkdir, artifacts, model import or GPU query.
    require(args.execute and args.device in ("cpu", "cuda"), "reviewed explicit execution required")
    destination = args.output.resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(),
        "new workspace output required",
    )
    require(args.preparation.resolve() == PREPARATION.resolve(), "fixed preparation required")
    require(
        not any(k == "cotracker" or k.startswith("cotracker.") for k in sys.modules),
        "fresh model interpreter required",
    )
    start, started = time.perf_counter(), utc()
    evidence, captured, timings, memory, views = Evidence(), {}, {}, [], []
    output = Output(destination)
    old_cv_threads, torch, old_torch_threads = cv2.getNumThreads(), None, None
    try:
        t = time.perf_counter()
        context = load_inputs(evidence)
        loaded_sources(evidence, output, captured)
        timings["input_verification_with_source_snapshots_seconds"] = time.perf_counter() - t
        require(
            type(context["opencv_threads"]) is int and context["opencv_threads"] > 0,
            "prepared OpenCV thread context missing",
        )
        cv2.setNumThreads(context["opencv_threads"])
        require(cv2.getNumThreads() == context["opencv_threads"], "OpenCV thread context differs")
        if args.device == "cuda":
            sample = gpu_memory()
            memory.append(dict(view=None, call_index=None, startup=True, **sample))
            check_memory(sample, startup=True)
        import torch as torch_library

        torch = torch_library
        old_torch_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        require(
            torch.get_num_threads() == 2 and torch.get_num_interop_threads() == 1,
            "CPU thread policy differs",
        )
        environment = dict(
            python=sys.version,
            packages={
                name: importlib.metadata.version(name)
                for name in ("numpy", "opencv-python", "torch")
            },
            torch_cpu_threads=2,
            torch_cpu_interop_threads=1,
            opencv_threads=cv2.getNumThreads(),
            device=args.device,
            matmul_precision=torch.get_float32_matmul_precision(),
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            float32_tensor_scope="official internal backend math policy unchanged",
            cuda_build=torch.version.cuda,
            allocator_fraction=0.25 if args.device == "cuda" else None,
        )
        require(environment["packages"] == DEPENDENCY_BINDING, "qualified runtime changed")
        plan = dict(
            schema=STAGE,
            plan_sha256=PLAN_SHA,
            frames=list(FRAMES),
            views=list(VIEWS),
            preparation=str(PREPARATION),
            preparation_sha256=PREPARATION_SHA,
            model_qualification_sha256=MODEL_QUALIFICATION_SHA,
            asset_seal_sha256=ASSET_SEAL_SHA,
            asset_manifest_sha256=ASSET_MANIFEST_SHA,
            windows=[list(w) for w in WINDOWS],
            init_relative_frame=0,
            first_emission_primary=True,
            all_query_rows=context["queries"],
            no_reseed=True,
            no_AMP=True,
            no_compile=True,
            input_model_rgb_float32_range=[0, 255],
            internal_model_shape=[384, 512],
            native_image_shape=[1200, 1200],
            align_corners=True,
            support_grid=False,
            raw_scores="sigmoid visibility/confidence, not logits; public product >0.6",
            stationary_scope="initial coordinate repeated; reported means finite arithmetic input",
            GPU_startup_free_MiB=6144,
            GPU_per_call_free_MiB=2048,
            initial_query_frame_excluded_from_primary_scores=True,
            preparation_SL_calibration_claims=context["plan"]["shared_SL_calibration_sha256"],
            inherited_source_scope=(
                "successful preparation byte-bound; MP4, SL, D2 inference not reopened here"
            ),
            initial_patch_anchor="saved initial query for every method",
            timer_scopes=dict(
                forward_seconds=(
                    "official predictor and read-only hook device clones plus synchronize"
                ),
                input_transfer_seconds="CPU RGB/stack/float conversion and device transfer",
                capture_transfer_seconds="same-device parity arithmetic and CPU output copies",
                diagnostics_with_save_sha_seconds="image arithmetic and output writes/hashes",
            ),
            model_forward_limit=21,
            maximum_query_count=832,
            environment=environment,
            human_truth=False,
            ID_or_review_changed=False,
            geometry_used=False,
            temporal_holdout=False,
        )
        output.write("plan.json", plan)
        evidence.final_guard()
        for view in VIEWS:
            views.append(run_view(context, view, output, args.device, memory))
            loaded_sources(evidence, output, captured)
            print(
                json.dumps(
                    dict(
                        event="view_complete",
                        view=view,
                        queries=views[-1]["query_count"],
                        online_calls=views[-1]["model_forward_calls"],
                    )
                ),
                flush=True,
            )
        q = sum(v["query_count"] for v in views)
        require(
            q == context["summary"]["query_count"]
            and len(views) == 3
            and sum(v["online_requests"] for v in views) == 21,
            "global denominator differs",
        )
        output.write("memory_samples.json", memory)
        timings["total_before_final_report_seconds"] = time.perf_counter() - start
        summary = dict(
            schema=STAGE,
            status="ok",
            frames=list(FRAMES),
            native_frames=64,
            views=views,
            query_count=q,
            all64_query_denominator=64 * q,
            tracking63_query_denominator=63 * q,
            logical_online_requests=21,
            model_forward_calls=sum(v["model_forward_calls"] for v in views),
            model_initializations=sum(v["model_initializations"] for v in views),
            lk_view_edges=189,
            source_queries_regenerated=0,
            raw_video_decodes=0,
            saved_PNG_decodes=192,
            no_ID_truth=True,
            ID_or_review_changed=False,
            new_3D=False,
            new_fit=False,
            temporal_holdout=False,
            surrogate_only=True,
            first_frame_not_tracking_score=True,
            raw_scores_preserved=True,
            all_prefixes_preserved=True,
            no_query_replacement=True,
            timing_limits=(
                "single short clip with concurrent external load; no 166fps or causal speed claim"
            ),
            source_hashes=captured,
            environment=environment,
            timings=timings,
        )
        output.write("summary.json", summary)
        report = (
            "# Fixed64 anonymous point comparison\n\n"
            f"All {q} prepared query requests remain: {64 * q} full frame/query positions, "
            f"{63 * q} tracking positions. Three independent views and "
            "all21 logical online windows are retained.\n\n"
            "Primary CoTracker fields are first emission; final-prefix revisions are separate. "
            "LK accepted, CoTracker reported-visible and stationary finite-input flags "
            "have different meanings. "
            "All partial/one-sided/common fields and all mask memberships remain in NPZ outputs. "
            "Image consistency is a surrogate; no point accuracy, visibility accuracy, "
            "animal identity "
            "or production adoption is established. "
            "No raw MP4, SL arrays, 3D or GT was read here.\n"
        )
        output.write("RESULT.md", report.encode())
        loaded_sources(evidence, output, captured)
        t = time.perf_counter()
        evidence.final_guard()
        output.final_guard()
        guard_seconds = time.perf_counter() - t
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=utc(),
            inputs=evidence.paths,
            outputs=output.records,
            params=dict(
                all_source_input_output_guards_passed=True,
                no_ID_truth=True,
                final_guard_seconds=guard_seconds,
                full_elapsed_seconds=time.perf_counter() - start,
            ),
        )
        with (destination / "comparison.provenance.json").open("xb") as stream:
            stream.write(json_bytes(seal))
        return summary
    except BaseException as error:
        partial = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=utc(),
            inputs=evidence.paths,
            outputs=output.records,
            completed_views=[v["view"] for v in views],
            error_type=type(error).__name__,
            error=str(error),
            memory_samples=memory,
            final_audit_complete=False,
            timings=timings,
        )
        seal = destination / "comparison.provenance.json"
        if not seal.exists():
            with seal.open("xb") as stream:
                stream.write(json_bytes(partial))
        raise
    finally:
        cv2.setNumThreads(old_cv_threads)
        if torch is not None and old_torch_threads is not None:
            torch.set_num_threads(old_torch_threads)


def gpu_memory():
    started = time.perf_counter()
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.split(",") for line in completed.stdout.strip().splitlines()]
    require(len(rows) == 1 and int(rows[0][0]) == 0, "single registered GPU0 required")
    free, total = map(float, rows[0][1:])
    require(
        np.isfinite(free) and np.isfinite(total) and 0 <= free <= total, "GPU memory query invalid"
    )
    return dict(
        timestamp=utc(), free_MiB=free, total_MiB=total, query_seconds=time.perf_counter() - started
    )


def check_memory(sample, startup=False):
    require(sample["free_MiB"] >= (6144 if startup else 2048), "GPU free-memory guard failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preparation", type=Path, default=PREPARATION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            dict(
                status=result["status"],
                query_count=result["query_count"],
                model_forward_calls=result["model_forward_calls"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Read-only source/interface audit and independent checks of new adapter outputs.

Calls the existing C25 compact-layout implementation without running calibration,
3D reconstruction, label export, or production entrypoints. Output row identities
remain (view, native frame, C01 global row), never biological fish identities.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "work/sleap_pipeline_adapter_v1"
RAW = ROOT / "work/generative_tracking_inputs/work/c01_s16_predictions/bottomup"
EXPERIMENT = "20251225_K_1_2_FCMC"
VIEWS = ("top_cam", "side1_cam", "side2_cam")
METHODS = (
    "SP",
    "G0",
    "G_history3",
    "G_history9",
    "G_match025",
    "G_match10",
    "G_margin02",
    "G_margin10",
    "G_age8",
    "G_age33",
)
PIPELINE_PATHS = (
    "tools/c01_s16_predictions.py",
    "tools/c24_uncapped_tracklets.py",
    "tools/c25_variable_cardinality.py",
    "tools/c27_uncapped_tracklets_any.py",
    "tools/c28_lambda_grid_transfer.py",
    "src/medaka3d/track.py",
    "src/medaka3d/reconstruct.py",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same(actual, expected, name, *, dtype=True):
    require(np.array_equal(actual, expected, equal_nan=True), f"Changed source array: {name}")
    if dtype:
        require(actual.dtype == expected.dtype, f"Changed source dtype: {name}")


def pipeline_modules():
    """Import callable production tools, never their command-line main functions."""
    tools_path = str(ROOT / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    return (
        importlib.import_module("c01_s16_predictions"),
        importlib.import_module("c25_variable_cardinality"),
    )


def source_location(relative, marker):
    lines = (ROOT / relative).read_text(encoding="utf-8").splitlines()
    return dict(path=relative, line=next(i + 1 for i, line in enumerate(lines) if marker in line))


def interface_contract():
    return dict(
        implementation_inspected=list(PIPELINE_PATHS),
        source_sha256={path: sha256(ROOT / path) for path in PIPELINE_PATHS},
        biological_id_claim=False,
        production_modified=False,
        geometry_executed=False,
        findings=[
            dict(
                code="negative_labels_excluded",
                evidence=source_location("tools/c25_variable_cardinality.py", "live = labels >= 0"),
                observation=(
                    "C25 compact_layout silently excludes rows with labels < 0 before geometry."
                ),
                adapter_contract=(
                    "Confirmed output gives every detection a nonnegative fragment label, "
                    "including singleton uncertain/P-missing observations. Provisional -1 is "
                    "diagnostic only."
                ),
            ),
            dict(
                code="within_frame_source_offset",
                evidence=source_location("tools/c25_variable_cardinality.py", "row = (np.arange"),
                observation=(
                    "row_at holds the original within-frame detection offset, not a tracking "
                    "label or C01 global_row."
                ),
                adapter_contract=(
                    "native_frame = source_frame_offset + local_frame; C01 global_row = "
                    "raw.start[native_frame] + row_at."
                ),
            ),
            dict(
                code="c24_return_order_mapping",
                evidence=source_location(
                    "tools/c24_uncapped_tracklets.py", "assigned[lo + offset]"
                ),
                observation=(
                    "C24 writes Tracker.track return enumeration into input batch row order; "
                    "returned instance order need not match input order. C27 invokes "
                    "C24.run_tracker."
                ),
                adapter_contract=(
                    "Adapter must associate returned instances with stable input "
                    "objects/global_row, never their return-list position."
                ),
            ),
            dict(
                code="partial_c28_propagation_offset",
                evidence=source_location(
                    "tools/c28_lambda_grid_transfer.py", "per_view = propagate_to_views"
                ),
                supporting_evidence=source_location(
                    "src/medaka3d/track.py", "rows = starts[frames[valid]]"
                ),
                observation=(
                    "C28 --frames produces a window-length Reconstruction but calls "
                    "propagate_to_views with full-recording starts. That function uses local "
                    "Reconstruction indices without a native offset parameter."
                ),
                adapter_contract=(
                    "Do not insert short-window outputs directly into the existing C28 command. "
                    "Match starts to the same local frame window, and preserve an explicit "
                    "source_frame_offset."
                ),
            ),
            dict(
                code="frame_numbering_of_c24_slice",
                evidence=source_location("tools/c24_uncapped_tracklets.py", "def slice_detections"),
                observation=(
                    "C24 slice_detections retains native frame_idx and original n_frames; rows "
                    "become a window subset."
                ),
                adapter_contract=(
                    "Never confuse a subset row index with the C01 global row. This audit "
                    "explicitly retains the global-row mapping and reindexes only the temporary "
                    "compact layout."
                ),
            ),
            dict(
                code="legacy_candidate_capacity",
                evidence=source_location("tools/c25_variable_cardinality.py", "K_MAX = 12"),
                observation=(
                    "Legacy C25 accepts 0..12 detections per frame; 13 or more raises rather "
                    "than truncates. A 3-slot candidate state uses at most 3 detections, while "
                    "the union of candidates includes every observed row."
                ),
                adapter_contract=(
                    "No 2D top-3 cull. All rows must survive into the candidate alphabet; "
                    "downstream state search remains a separate 3D decision."
                ),
            ),
            dict(
                code="arbitrary_local_labels",
                evidence=source_location("tools/c25_variable_cardinality.py", "order = np.lexsort"),
                observation=(
                    "Compact indices are sorted by per-view local label and can differ frame to "
                    "frame. The labels are not S0/S1/S2 biological IDs."
                ),
                adapter_contract=(
                    "The downstream inverse mapping must use row_at, independently of label "
                    "values, label renaming, and compact rank."
                ),
            ),
            dict(
                code="legacy_maximal_slot_assignment",
                evidence=source_location(
                    "tools/c25_variable_cardinality.py", "def view_assignments"
                ),
                observation=(
                    "C25 candidate states fill exactly min(k, 3) slots per view. With k >= 3, "
                    "the legacy enumeration has no empty-slot option, although excess rows "
                    "can remain unused by an individual state."
                ),
                adapter_contract=(
                    "Candidate-alphabet preservation does not establish that a slot is a real "
                    "visible fish or that downstream null-assignment handling is sufficient. "
                    "No geometry or 3D selection is approved by this interface audit."
                ),
            ),
        ],
        integration_status=(
            "Dry-run candidate-input compatibility only; no production injection or 3D accuracy "
            "claim."
        ),
    )


def compact_roundtrip(c01, c25, arrays, raw, case, view):
    """Feed every core observation to actual C25 and invert it to original C01 rows."""
    core = (arrays["frame_idx"] >= case["start"]) & (arrays["frame_idx"] < case["stop"])
    native = arrays["frame_idx"][core]
    labels = arrays["labels"][core]
    global_rows = arrays["global_row"][core]
    points, scores = arrays["points"][core], arrays["score"][core]
    offset = int(case["start"])
    detections = c01.Detections(
        experiment=EXPERIMENT,
        view=view,
        frame_idx=native - offset,
        points=points,
        score=scores,
        n_frames=case["stop"] - offset,
        nodes=tuple(str(node) for node in raw["nodes"]),
    )
    require(np.all(labels >= 0), "A negative confirmed label would disappear in C25")
    require(np.all(labels <= np.iinfo(np.int32).max), "C25 int32 label overflow")
    uv, label_at, row_at, count = c25.compact_layout(detections, labels, detections.nodes)
    local_frame, compact_rank = np.nonzero(label_at >= 0)
    native_frame = local_frame + offset
    within_frame = row_at[local_frame, compact_rank].astype(np.int64)
    raw_starts = np.searchsorted(raw["frame_idx"], native_frame, side="left")
    recovered_global = raw_starts + within_frame
    subset_rows = detections.start[local_frame] + within_frame
    same(recovered_global, global_rows[subset_rows], "C25 source offset -> global row", dtype=False)
    same(native_frame, native[subset_rows], "native frame offset", dtype=False)
    same(label_at[local_frame, compact_rank], labels[subset_rows], "compact label", dtype=False)
    same(
        np.sort(recovered_global),
        np.sort(global_rows),
        "all rows in candidate alphabet",
        dtype=False,
    )
    require(len(np.unique(recovered_global)) == len(global_rows), "Duplicated source row")
    for index, node in enumerate(detections.nodes):
        same(
            uv[node][local_frame, compact_rank],
            points[subset_rows, index],
            f"C25 original {node} pixels/NaNs",
            dtype=False,
        )
    expected_count = np.bincount(native - offset, minlength=detections.n_frames)
    same(count, expected_count, "C25 detection counts including 0/4+", dtype=False)
    capacities = []
    for k in np.unique(count):
        candidates = c25.view_assignments(int(k))
        used = np.unique(candidates[candidates >= 0])
        same(
            used, np.arange(k), "all compact rows appear in downstream candidate union", dtype=False
        )
        require(
            all(len(set(row[row >= 0].tolist())) == (row >= 0).sum() for row in candidates),
            "A downstream candidate duplicates one detection",
        )
        capacities.append(
            dict(
                detection_count=int(k),
                candidate_states=len(candidates),
                candidate_union_size=len(used),
                legacy_slots_filled_per_state=min(int(k), 3),
                unused_detection_options=int(k) > 3,
            )
        )
    # Label renaming must affect compact ordering only, never which source pixels survive.
    renamed = labels.max(initial=-1) - labels + 1000
    _, renamed_labels, renamed_rows, renamed_count = c25.compact_layout(
        detections, renamed, detections.nodes
    )
    rf, rk = np.nonzero(renamed_labels >= 0)
    restored = np.searchsorted(raw["frame_idx"], rf + offset) + renamed_rows[rf, rk]
    same(
        np.sort(restored),
        np.sort(global_rows),
        "arbitrary label rename row preservation",
        dtype=False,
    )
    same(renamed_count, count, "arbitrary label rename cardinality")
    return dict(
        core_rows=len(global_rows),
        candidate_rows=len(recovered_global),
        source_frame_offset=offset,
        local_frame_range=[0, detections.n_frames],
        native_frame_range=[case["start"], case["stop"]],
        all_rows_retained=True,
        original_node_nan_masks_preserved=True,
        source_offset_global_row_roundtrip=True,
        arbitrary_label_rename_preserves_rows=True,
        downstream_candidate_capacities=capacities,
        maximum_detections=int(count.max(initial=0)),
        frames_four_or_more=int((count >= 4).sum()),
    )


def synthetic_interface_check(c01, c25):
    counts = np.array([0, 1, 2, 3, 4, 5, 3, 2, 1, 0, 2, 4, 3])
    frame_idx = np.repeat(np.arange(len(counts)), counts)
    points = np.arange(len(frame_idx) * 6, dtype=np.float32).reshape(-1, 3, 2)
    points[3, 0] = np.nan
    score = np.ones(len(frame_idx), np.float32)
    labels = np.arange(len(frame_idx), dtype=np.int64)
    detections = c01.Detections(
        "synthetic_interface_only", "top_cam", frame_idx, points, score, len(counts)
    )
    uv, label_at, row_at, actual = c25.compact_layout(detections, labels, detections.nodes)
    same(actual, counts, "synthetic 0..5 cardinalities", dtype=False)
    frames, ranks = np.nonzero(label_at >= 0)
    rows = detections.start[frames] + row_at[frames, ranks]
    same(np.sort(rows), np.arange(len(labels)), "synthetic singleton preservation", dtype=False)
    same(uv["mouth"][frames, ranks], points[rows, 0], "synthetic node missingness", dtype=False)
    negative = labels.copy()
    negative[3] = -1
    _, _, _, negative_count = c25.compact_layout(detections, negative, detections.nodes)
    require(
        int(negative_count.sum()) == len(labels) - 1,
        "Negative-label exclusion diagnostic did not match actual C25",
    )
    over = c01.Detections(
        "synthetic_interface_only",
        "top_cam",
        np.zeros(13, dtype=int),
        np.zeros((13, 3, 2)),
        np.ones(13),
        1,
    )
    try:
        c25.compact_layout(over, np.arange(13), over.nodes)
    except SystemExit:
        refused = True
    else:
        refused = False
    require(refused, "C25 capacity overflow was not refused")
    return dict(
        count_sequence=counts.tolist(),
        rows=len(labels),
        all_rows_preserved=True,
        negative_label_control_removed_exactly_one=True,
        thirteen_detection_frame_refused_without_truncation=True,
        annotation_status="synthetic implementation check, no biological ground truth",
    )


def predecessors(frames, labels):
    previous = np.full(len(labels), -1, np.int64)
    last = {}
    for row, label in enumerate(labels):
        if label < 0:
            continue
        previous[row] = last.get(int(label), -1)
        last[int(label)] = row
    gap = np.zeros(len(labels), np.int64)
    connected = previous >= 0
    gap[connected] = frames[connected] - frames[previous[connected]]
    return previous, gap


def verify_arrays(arrays, raw, case, context, method):
    required = {
        "frame_idx",
        "native_frame",
        "points",
        "score",
        "global_row",
        "global_rows",
        "core",
        "core_mask",
        "observed_mask",
        "labels",
        "local_id",
        "provisional_labels",
        "previous_row",
        "link_gap",
        "clean_edge",
    }
    require(required <= arrays.keys(), f"Missing NPZ fields: {sorted(required - arrays.keys())}")
    frames, rows = arrays["frame_idx"], arrays["global_row"]
    for alias, primary in (
        ("native_frame", "frame_idx"),
        ("global_rows", "global_row"),
        ("local_id", "labels"),
    ):
        same(arrays[alias], arrays[primary], f"NPZ alias {alias}")
    start, stop = max(0, case["start"] - context), min(int(raw["n_frames"]), case["stop"] + context)
    expected_rows = np.flatnonzero((raw["frame_idx"] >= start) & (raw["frame_idx"] < stop))
    same(rows, expected_rows, "complete context global rows", dtype=False)
    for key in ("frame_idx", "points", "score"):
        same(arrays[key], raw[key][rows], key)
    core = (frames >= case["start"]) & (frames < case["stop"])
    for key in ("core", "core_mask"):
        same(arrays[key], core, key)
    same(arrays["observed_mask"], np.isfinite(arrays["points"]).all(-1), "observed mask")
    labels = arrays["labels"]
    require(labels.shape == frames.shape and labels.dtype.kind in "iu", "Invalid label array")
    require(np.all(labels >= 0), "Confirmed labels must preserve every row as a fragment")
    require(
        len(np.unique(np.c_[frames, labels], axis=0)) == len(rows),
        "Same local fragment consumes multiple detections in one frame",
    )
    previous, gap = predecessors(frames, labels)
    if method.startswith("G"):
        require(not np.any(gap > 1), "A confirmed G fragment crosses missing native frames")
    require(not np.any(gap < 0), "Frames run backwards")
    for key, expected in (("previous_row", previous), ("link_gap", gap), ("clean_edge", gap == 1)):
        same(arrays[key], expected, key, dtype=False)
    provisional = arrays["provisional_labels"]
    require(
        provisional.shape == frames.shape and provisional.dtype.kind in "iu",
        "Provisional row alignment or label dtype",
    )
    assigned = provisional >= 0
    require(
        len(np.unique(np.c_[frames[assigned], provisional[assigned]], axis=0))
        == int(assigned.sum()),
        "Same provisional track consumes multiple detections in one frame",
    )
    connected_rows = np.flatnonzero(gap == 1)
    require(
        np.all(provisional[connected_rows] >= 0)
        and np.array_equal(provisional[connected_rows], provisional[previous[connected_rows]]),
        "Confirmed adjacent edge changes its provisional track",
    )
    eligible = core & (frames > case["start"])
    accepted = (gap == 1) & eligible
    return dict(
        context_rows=len(rows),
        core_rows=int(core.sum()),
        retained_rows=int(core.sum()),
        eligible_observations=int(eligible.sum()),
        adjacent_connections=int(accepted.sum()),
        gap_connections=int(((gap > 1) & eligible).sum()),
        contiguous_runs=int(core.sum() - accepted.sum()),
        unconnected_observations=int((eligible & ~accepted).sum()),
        provisional_unassigned_rows=int(((arrays["provisional_labels"] < 0) & core).sum()),
        raw_arrays_and_masks_exact=True,
        one_to_one_per_native_frame=True,
        provisional_one_to_one_per_native_frame=True,
        confirmed_edges_within_provisional_tracks=True,
        confirmed_scope="adjacent-only"
        if method.startswith("G")
        else "native SLEAP baseline permits gaps",
    )


def shuffled_g0(arrays, view, config_dict):
    from adapter import AdapterConfig, track

    permutation = np.random.default_rng(20260918).permutation(len(arrays["frame_idx"]))
    config = AdapterConfig(**config_dict) if config_dict else AdapterConfig()
    inputs = {
        key: arrays[key][permutation] for key in ("frame_idx", "points", "score", "global_row")
    }
    original_inputs = {key: values.copy() for key, values in inputs.items()}
    result = track(
        inputs["frame_idx"],
        inputs["points"],
        inputs["score"],
        inputs["global_row"],
        view,
        config,
    )
    for key in inputs:
        same(inputs[key], original_inputs[key], f"shuffled tracker input {key}")
    for key in ("labels", "provisional_labels"):
        restored = np.empty_like(getattr(result, key))
        restored[permutation] = getattr(result, key)
        same(restored, arrays[key], "shuffled " + key, dtype=False)
    previous, gap = predecessors(arrays["frame_idx"], arrays["labels"])
    expected = {
        (int(arrays["global_row"][previous[row]]), int(arrays["global_row"][row]))
        for row in np.flatnonzero(gap == 1)
    }
    actual = {(int(edge["from_global_row"]), int(edge["to_global_row"])) for edge in result.edges}
    require(actual == expected, "Row shuffle changed confirmed global-row edges")
    require(len(result.edges) == len(expected), "Row shuffle returned duplicate confirmed edges")
    return dict(
        view=view,
        case="cg20s",
        method="G0",
        rows=len(permutation),
        confirmed_labels_equal=True,
        provisional_labels_equal=True,
        confirmed_global_row_edges_equal=True,
        config=asdict(config),
        score_input_preserved=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--interface-only", action="store_true")
    parser.add_argument("--skip-shuffle", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    c01, c25 = pipeline_modules()
    contract = interface_contract()
    contract["synthetic_compact_layout"] = synthetic_interface_check(c01, c25)
    name = "interface_audit" if args.interface_only else "audit"
    with provenance("sleap_adapter_independent_interface_audit", output, name=name) as record:
        record.params_from(args)
        record.input("audit_code", Path(__file__), sha256=sha256(Path(__file__)))
        for path in PIPELINE_PATHS:
            record.input(path, ROOT / path, sha256=contract["source_sha256"][path])
        write(output / "interface_audit.json", contract)
        record.output(
            "interface_contract",
            output / "interface_audit.json",
            sha256=sha256(output / "interface_audit.json"),
        )
        if args.interface_only:
            print(output / "interface_audit.json", flush=True)
            return
        protocol = read(output / "protocol.json")
        summary = read(output / "summary.json")
        for filename, expected_digest in protocol["frozen_file_sha256"].items():
            path = HERE / filename
            require(sha256(path) == expected_digest, f"Frozen numerical source changed: {filename}")
            record.input("frozen_" + filename, path, sha256=expected_digest)
        import sleap_nn.tracking.tracker as installed

        package = Path(installed.__file__).parent
        for relative in (
            "tracker.py",
            "candidates/local_queues.py",
            "utils.py",
            "../evaluation.py",
        ):
            path = (package / relative).resolve()
            record.input("installed_" + relative, path, sha256=sha256(path))
        records_by_key = {
            (r["case"], r["view"], r.get("method", r.get("condition"))): r
            for r in summary["records"]
        }
        require(
            len(records_by_key) == len(summary["records"]), "Duplicate method-case-view summary"
        )
        for filename in ("protocol.json", "summary.json"):
            record.input(filename, output / filename, sha256=sha256(output / filename))
        configs = protocol.get("configurations", protocol.get("configs", {}))
        g0_config = configs.get("G0", {})
        if not isinstance(g0_config, dict):
            g0_config = protocol.get("adapter_configs", {}).get("G0", {})
        verified, shuffled = [], []
        for view in VIEWS:
            raw_path = RAW / f"{EXPERIMENT}__{view}.npz"
            with np.load(raw_path, allow_pickle=False) as values:
                raw = {key: values[key] for key in values.files}
            raw_digest = sha256(raw_path)
            require(
                raw_digest == protocol["raw_inputs"][view]["sha256"], "Raw C01 changed after freeze"
            )
            record.input(view + "_raw", raw_path, sha256=raw_digest)
            for case in protocol["cases"]:
                for method in METHODS:
                    path = output / case["name"] / f"tracking_{method}_{view}.npz"
                    with np.load(path, allow_pickle=False) as values:
                        arrays = {key: values[key] for key in values.files}
                    actual = verify_arrays(
                        arrays, raw, case, int(protocol.get("context", 332)), method
                    )
                    stats = records_by_key[case["name"], view, method]
                    for key in (
                        "core_rows",
                        "retained_rows",
                        "eligible_observations",
                        "adjacent_connections",
                        "gap_connections",
                        "contiguous_runs",
                        "unconnected_observations",
                    ):
                        if key in stats:
                            require(
                                stats[key] == actual[key], f"Summary mismatch {method}/{view}/{key}"
                            )
                    layout = compact_roundtrip(c01, c25, arrays, raw, case, view)
                    input_digest = sha256(path)
                    require(
                        stats["source_artifact"]["sha256"] == input_digest,
                        f"Summary artifact hash mismatch: {case['name']}/{view}/{method}",
                    )
                    record.input(f"{case['name']}_{view}_{method}", path, sha256=input_digest)
                    verified.append(
                        dict(
                            case=case["name"],
                            view=view,
                            method=method,
                            input_npz_sha256=input_digest,
                            **actual,
                            compact_layout=layout,
                        )
                    )
                    if case["name"] == "cg20s" and method == "G0" and not args.skip_shuffle:
                        shuffled.append(shuffled_g0(arrays, view, g0_config))
                print(
                    f"audit {case['name']} {view}: 10 methods and C25 source roundtrip OK",
                    flush=True,
                )
        expected = len(protocol["cases"]) * len(VIEWS) * len(METHODS)
        require(len(verified) == expected, "Missing adapter condition audit")
        source_after = {path: sha256(ROOT / path) for path in PIPELINE_PATHS}
        require(
            source_after == contract["source_sha256"],
            "Existing production source changed during audit",
        )
        report = dict(
            complete=not args.skip_shuffle,
            records=verified,
            shuffled=shuffled,
            interface_contract=contract,
            geometry_executed=False,
            production_modified=False,
            biological_identity_accuracy=None,
            verified_identity_denominator=0,
            totals=dict(
                condition_case_views=len(verified),
                case_views=len(protocol["cases"]) * 3,
                input_row_contract_failures=0,
                confirmed_G_gap_connections=0,
                compact_layout_rows=sum(r["compact_layout"]["candidate_rows"] for r in verified),
                maximum_native_detections=max(
                    r["compact_layout"]["maximum_detections"] for r in verified
                ),
                per_method_records=dict(Counter(r["method"] for r in verified)),
            ),
        )
        write(output / "audit.json", report)
        record.output("audit", output / "audit.json", sha256=sha256(output / "audit.json"))
        print(
            f"Complete interface audit: {len(verified)} records, {len(shuffled)} main G0 shuffles",
            flush=True,
        )


if __name__ == "__main__":
    main()

"""Prespecified SLEAP adapter experiments on twelve previously inspected windows."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from frozen_metrics import measure, predecessors

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "work/variable_count_tracklets_v1"
OUTPUT = ROOT / "work/sleap_pipeline_adapter_v1"
RAW = ROOT / "work/generative_tracking_inputs/work/c01_s16_predictions/bottomup"
EXPERIMENT = "20251225_K_1_2_FCMC"
VIEWS = ("top_cam", "side1_cam", "side2_cam")
REFERENCES = ("C27", "S1", "S2", "P0", "P1")
CONDITIONS = (
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
FROZEN_FILES = ("README.md", "adapter.py", "run.py", "frozen_metrics.py")
CONTRACT_KEYS = ("frame_idx", "points", "score", "global_row", "observed_mask", "core_mask")


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path, records):
    fields = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(clean(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else clean(value)
                    for key, value in record.items()
                }
            )


def write_evidence(path, value):
    """One compressed evidence copy, deterministic gzip metadata, finite JSON."""
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=6) as zipped,
    ):
        zipped.write(
            json.dumps(
                clean(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        )


def configurations():
    from adapter import AdapterConfig

    base = AdapterConfig()
    return {
        "G0": base,
        "G_history3": replace(base, history=3),
        "G_history9": replace(base, history=9),
        "G_match025": replace(base, match_threshold=0.025),
        "G_match10": replace(base, match_threshold=0.10),
        "G_margin02": replace(base, margin=0.02),
        "G_margin10": replace(base, margin=0.10),
        "G_age8": replace(base, max_age_seconds=8 / 166),
        "G_age33": replace(base, max_age_seconds=33 / 166),
    }


def array_digest(value):
    data = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(data.dtype.str.encode("ascii"))
    digest.update(json.dumps(data.shape).encode("ascii"))
    digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


def contract_fingerprints(data):
    return {key: array_digest(data[key]) for key in CONTRACT_KEYS}


def extract_input(raw, case):
    start, stop = case["start"], case["stop"]
    lo, hi = np.searchsorted(raw["frame_idx"], [start - 332, stop + 332])
    frames, points, score = (raw[key][lo:hi] for key in ("frame_idx", "points", "score"))
    rows = np.arange(lo, hi, dtype=np.int64)
    core = (frames >= start) & (frames < stop)
    return dict(
        frame_idx=frames,
        native_frame=frames,
        points=points,
        score=score,
        global_row=rows,
        global_rows=rows,
        core=core,
        core_mask=core,
        observed_mask=np.isfinite(points).all(axis=-1),
    )


def verify_reference_input(data, path):
    with np.load(path, allow_pickle=False) as prior:
        for key in CONTRACT_KEYS:
            if prior[key].dtype != data[key].dtype:
                raise AssertionError(f"Input dtype mismatch: {path} {key}")
            if not np.array_equal(prior[key], data[key], equal_nan=True):
                raise AssertionError(f"Input values mismatch: {path} {key}")


def initialize(output):
    if (output / "protocol.json").exists():
        raise FileExistsError("Protocol already exists; do not overwrite a frozen experiment")
    prior_protocol = json.loads((OLD / "protocol.json").read_text(encoding="utf-8"))
    configs = configurations()
    source_hashes = {name: sha256(HERE / name) for name in FROZEN_FILES}
    cases = []
    for source in prior_protocol["cases"]:
        case = dict(source)
        case["prior_category"] = source["category"]
        case["category"] = "previously_inspected_regression_window"
        cases.append(case)
    protocol = dict(
        experiment=EXPERIMENT,
        fps=166,
        context=332,
        cases=cases,
        frozen_at_utc=datetime.now(UTC).isoformat(),
        new_conditions=list(CONDITIONS),
        reference_conditions=list(REFERENCES),
        conditions=list((*REFERENCES, *CONDITIONS)),
        configurations={name: asdict(config) for name, config in configs.items()},
        native_pectoral_control=dict(
            condition="SP",
            history=5,
            nodes=["pectoral"],
            candidates_method="local_queues",
            scoring="fixed_area_oks",
            scoring_reduction="mean",
            oks_stddev=0.025,
            top_area_px2=14400,
            side_area_px2=10000,
            max_tracks=None,
            match_threshold=None,
            margin=None,
            max_native_age=None,
            gap_links_allowed=True,
            correct_instance_to_global_row_readout=True,
        ),
        parameter_policy="Nine fixed G settings; one factor from G0; no best-setting selection",
        comparison_axes={
            "S2_to_SP": "Remove M from fixed-area MP OKS; same history and native SLEAP policy",
            "SP_to_G0": "Adapter design bundle: explicit unmatched, gates, margins, finite age",
            "G0_to_sensitivity": "Exactly one declared configuration field changes",
        },
        no_fresh_holdout=True,
        validation_scope="All twelve windows were inspected previously; no new unseen validation",
        no_identity_accuracy_claim=True,
        n_independent_groups=1,
        no_3d_execution=True,
        labels_contract="All rows remain; G labels are adjacent confirmed components/singletons",
        provisional_contract="Separate SLEAP working IDs; gaps never merge G confirmed labels",
        core_edge_scope="Both endpoints in core; first core frame excluded from denominator",
        score_scope="Events selected by exact C01 global_row/native frame, not offset guesses",
        score_definition="event.score is association OKS; NPZ score is original detector score",
        node_score_policy="Original instance scores only; no measured node scores invented",
        all_input_rows_retained=True,
        input_protocol=dict(
            path=str((OLD / "protocol.json").relative_to(ROOT)),
            sha256=sha256(OLD / "protocol.json"),
        ),
        input_verified_manifest=dict(
            path=str((OLD / "numerical_manifest_verified.json").relative_to(ROOT)),
            sha256=sha256(OLD / "numerical_manifest_verified.json"),
        ),
        old_numerical_code_sha256=prior_protocol["code_sha256"],
        frozen_file_sha256=source_hashes,
        raw_inputs={
            view: dict(
                path=str((RAW / f"{EXPERIMENT}__{view}.npz").relative_to(ROOT)),
                sha256=sha256(RAW / f"{EXPERIMENT}__{view}.npz"),
            )
            for view in VIEWS
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    with provenance("sleap_adapter_protocol_freeze", output) as rec:
        for name, digest in source_hashes.items():
            rec.input("code_" + name, HERE / name, sha256=digest)
        rec.input(
            "previous_protocol", OLD / "protocol.json", sha256=protocol["input_protocol"]["sha256"]
        )
        rec.input(
            "previous_verified_manifest",
            OLD / "numerical_manifest_verified.json",
            sha256=protocol["input_verified_manifest"]["sha256"],
        )
        for view, entry in protocol["raw_inputs"].items():
            rec.input("raw_" + view, ROOT / entry["path"], sha256=entry["sha256"])
        import sleap_nn.tracking.tracker as installed

        package = Path(installed.__file__).parent
        for name in ("tracker.py", "candidates/local_queues.py", "utils.py", "../evaluation.py"):
            path = (package / name).resolve()
            rec.input("installed_" + name, path, sha256=sha256(path))
        write_json(output / "protocol.json", protocol)
        rec.output("protocol", output / "protocol.json", sha256=sha256(output / "protocol.json"))
    print(f"Frozen {len(cases)} known cases and {len(CONDITIONS)} new conditions", flush=True)


def check_freeze(output):
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    for name, expected in protocol["frozen_file_sha256"].items():
        if sha256(HERE / name) != expected:
            raise AssertionError(f"Frozen input changed: {name}")
    for name, config in configurations().items():
        if asdict(config) != protocol["configurations"][name]:
            raise AssertionError(f"Configuration differs from frozen protocol: {name}")
    return protocol


def event_summary(events, data, case):
    """Associate diagnostics with observations solely through original global rows."""
    rows = {int(value): i for i, value in enumerate(data["global_row"])}
    core_events, eligible_events = [], []
    for event in events:
        key = event.get("global_row")
        if key is None:
            continue
        if int(key) not in rows:
            raise AssertionError("Evidence refers to a row outside the exact input context")
        row = rows[int(key)]
        frame = int(data["frame_idx"][row])
        stamped = event.get("native_frame", event.get("frame"))
        if stamped is not None and int(stamped) != frame:
            raise AssertionError("Evidence global_row/native-frame correspondence is inconsistent")
        if data["core_mask"][row]:
            core_events.append(event)
            if frame > case["start"]:
                eligible_events.append(event)
    values = [
        float(event["score"])
        for event in eligible_events
        if event.get("score") is not None and np.isfinite(event["score"])
    ]
    return dict(
        evidence_core_events=len(core_events),
        evidence_eligible_events=len(eligible_events),
        reason_counts_core=dict(
            Counter(event.get("reason", "unspecified") for event in core_events)
        ),
        refusal_reasons=dict(
            Counter(
                event.get("reason", "unspecified")
                for event in eligible_events
                if not event.get("confirmed", False)
            )
        ),
        refusal_event_scope="Forward row events selected by exact global row inside eligible core",
        selected_score_observations=len(values),
        selected_score_median=float(np.median(values)) if values else None,
        selected_score_scope="Finite event.score only; rejected candidate scores are excluded",
    )


def validate_result(result, data, method):
    labels = np.asarray(result.labels)
    provisional = np.asarray(result.provisional_labels)
    if labels.shape != data["frame_idx"].shape or provisional.shape != labels.shape:
        raise AssertionError("Output labels do not map one-to-one to all C01 rows")
    if labels.dtype.kind not in "iu" or provisional.dtype.kind not in "iu":
        raise AssertionError("Labels must be integer arrays")
    previous, gaps = predecessors(data["frame_idx"], labels)
    predecessors(data["frame_idx"], provisional)
    if method.startswith("G"):
        if (labels < 0).any():
            raise AssertionError("Every G observation requires a confirmed component or singleton")
        if (gaps > 1).any():
            raise AssertionError("Provisional gap relations leaked into confirmed G labels")
    return previous, gaps


def measured(data, labels, case, view):
    return measure(
        data["frame_idx"],
        data["points"],
        labels,
        data["core_mask"],
        case["start"],
        case["stop"],
        120 if view == "top_cam" else 100,
        [],
    )


def reference_records(data, case, view, rec):
    records = []
    for method in REFERENCES:
        path = OLD / case["name"] / f"tracking_{method}_{view}.npz"
        verify_reference_input(data, path)
        with np.load(path, allow_pickle=False) as z:
            stats, _ = measured(data, z["labels"], case, view)
        old_metrics = path.with_name(f"metrics_{method}_{view}.json")
        prior = json.loads(old_metrics.read_text(encoding="utf-8"))
        for key in (
            "core_rows",
            "contiguous_runs",
            "adjacent_connections",
            "eligible_observations",
            "gap_connections",
            "unassigned_rows",
            "missing_m",
            "missing_p",
            "missing_t",
        ):
            if stats[key] != prior[key]:
                raise AssertionError(
                    f"Frozen metrics parity failed: {case['name']} {view} {method} {key}"
                )
        digest = sha256(path)
        stats.update(
            case=case["name"],
            view=view,
            method=method,
            reference=True,
            source_artifact=dict(path=path.relative_to(ROOT).as_posix(), sha256=digest),
            source_metrics=dict(
                path=old_metrics.relative_to(ROOT).as_posix(), sha256=sha256(old_metrics)
            ),
            metadata=dict(reference_only=True, original_artifact_unmodified=True),
        )
        stats["refusal_reasons"] = prior.get("refusal_reasons", {})
        stats["refusal_event_scope"] = prior.get("refusal_event_scope", "original reference scope")
        records.append(stats)
        rec.input(case["name"] + "_" + method, path, sha256=digest)
        rec.input(case["name"] + "_" + method + "_metrics", old_metrics, sha256=sha256(old_metrics))
    return records


def run_view(output, view, selected=None):
    from adapter import native_pectoral, track

    protocol = check_freeze(output)
    source = RAW / f"{EXPERIMENT}__{view}.npz"
    if sha256(source) != protocol["raw_inputs"][view]["sha256"]:
        raise AssertionError("C01 input changed after freeze")
    with np.load(source, allow_pickle=False) as z:
        if list(z["nodes"]) != ["mouth", "pectoral", "tail"]:
            raise AssertionError("C01 node order is not M/P/T")
        raw = {key: z[key] for key in ("frame_idx", "points", "score")}
    configs = configurations()
    with provenance("sleap_adapter_numerical_runs", output, name="run_" + view) as rec:
        rec.input("protocol", output / "protocol.json", sha256=sha256(output / "protocol.json"))
        rec.input("raw", source, sha256=protocol["raw_inputs"][view]["sha256"])
        for name in FROZEN_FILES:
            rec.input("source_" + name, HERE / name, sha256=sha256(HERE / name))
        for case in protocol["cases"]:
            if selected and case["name"] not in selected:
                continue
            dest = output / case["name"]
            dest.mkdir(parents=True, exist_ok=True)
            done = dest / f"metrics_{view}.json"
            if done.exists():
                raise FileExistsError(f"Do not overwrite a completed case-view: {done}")
            data = extract_input(raw, case)
            prior_input = OLD / case["name"] / f"detections_{view}.npz"
            verify_reference_input(data, prior_input)
            rec.input(case["name"] + "_prior_input", prior_input, sha256=sha256(prior_input))
            fingerprints = contract_fingerprints(data)
            contract = dict(
                case=case["name"],
                view=view,
                input_source_sha256=protocol["raw_inputs"][view]["sha256"],
                core_half_open=[case["start"], case["stop"]],
                context_half_open=[case["start"] - 332, case["stop"] + 332],
                array_sha256=fingerprints,
                exact_prior_npz_match=True,
                global_row_definition="Index in complete C01 view detection arrays",
            )
            write_json(dest / f"input_contract_{view}.json", contract)
            np.savez_compressed(dest / f"detections_{view}.npz", **data)
            records = reference_records(data, case, view, rec)
            for method in CONDITIONS:
                began = time.perf_counter()
                if method == "SP":
                    result = native_pectoral(
                        data["frame_idx"],
                        data["points"],
                        data["score"],
                        data["global_row"],
                        view,
                        history=5,
                    )
                else:
                    result = track(
                        data["frame_idx"],
                        data["points"],
                        data["score"],
                        data["global_row"],
                        view,
                        config=configs[method],
                    )
                if contract_fingerprints(data) != fingerprints:
                    raise AssertionError(
                        "Tracker modified original detections or observation masks"
                    )
                previous, gaps = validate_result(result, data, method)
                target = dest / f"tracking_{method}_{view}.npz"
                np.savez_compressed(
                    target,
                    **data,
                    labels=result.labels,
                    provisional_labels=result.provisional_labels,
                    local_id=result.labels,
                    previous_row=previous,
                    link_gap=gaps,
                    clean_edge=gaps == 1,
                )
                stats, timeline = measured(data, result.labels, case, view)
                provisional_stats, _ = measured(data, result.provisional_labels, case, view)
                stats.update(event_summary(result.events, data, case))
                if method == "SP":
                    stats["selected_score_scope"] = "Native SP does not export per-row score events"
                    stats["refusal_event_scope"] = "Native SP event decisions are not recorded"
                stats.update(
                    case=case["name"],
                    view=view,
                    method=method,
                    reference=False,
                    provisional_metrics=provisional_stats,
                    metadata=result.metadata,
                    elapsed_seconds=time.perf_counter() - began,
                    input_array_sha256=fingerprints,
                    source_artifact=dict(
                        path=target.relative_to(ROOT).as_posix(), sha256=sha256(target)
                    ),
                )
                evidence = dict(
                    case=case["name"],
                    view=view,
                    method=method,
                    global_row_definition="Original full C01 detection index",
                    core_half_open=[case["start"], case["stop"]],
                    context_half_open=[case["start"] - 332, case["stop"] + 332],
                    metadata=result.metadata,
                    edges=result.edges,
                    events=result.events,
                    reconnect_candidates=result.reconnect_candidates,
                )
                write_evidence(dest / f"evidence_{method}_{view}.json.gz", evidence)
                write_json(dest / f"metrics_{method}_{view}.json", stats)
                np.savez_compressed(dest / f"timeline_{method}_{view}.npz", **timeline)
                records.append(stats)
                for path in (
                    target,
                    dest / f"evidence_{method}_{view}.json.gz",
                    dest / f"metrics_{method}_{view}.json",
                    dest / f"timeline_{method}_{view}.npz",
                ):
                    rec.output(case["name"] + "_" + path.name, path, sha256=sha256(path))
                print(
                    f"{case['name']} {view} {method}: {stats['contiguous_runs']} runs; "
                    f"{stats['adjacent_connections']}/{stats['eligible_observations']} links; "
                    f"{stats['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            write_json(
                done,
                dict(
                    case=case["name"], view=view, records=records, all_original_rows_preserved=True
                ),
            )
            for path in (
                done,
                dest / f"input_contract_{view}.json",
                dest / f"detections_{view}.npz",
            ):
                rec.output(case["name"] + "_" + path.name, path, sha256=sha256(path))


def summarize(output):
    protocol = check_freeze(output)
    records = []
    for case in protocol["cases"]:
        for view in VIEWS:
            path = output / case["name"] / f"metrics_{view}.json"
            records.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    if len(records) != len(protocol["cases"]) * len(VIEWS) * (len(REFERENCES) + len(CONDITIONS)):
        raise AssertionError("Incomplete numerical comparison")
    totals = []
    for method in (*REFERENCES, *CONDITIONS):
        selected = [record for record in records if record["method"] == method]
        entry = dict(method=method, case_views=len(selected), reference=method in REFERENCES)
        for key in (
            "core_frames",
            "core_rows",
            "retained_rows",
            "eligible_observations",
            "adjacent_connections",
            "unconnected_observations",
            "contiguous_runs",
            "gap_connections",
            "unassigned_rows",
            "large_adjacent_steps",
        ):
            entry[key] = sum(record[key] for record in selected)
        entry["connection_rate"] = entry["adjacent_connections"] / entry["eligible_observations"]
        entry["abstention_rate"] = 1 - entry["connection_rate"]
        totals.append(entry)
    provisional = [
        dict(case=r["case"], view=r["view"], method=r["method"], **r["provisional_metrics"])
        for r in records
        if not r["reference"]
    ]
    strata = [
        dict(case=r["case"], view=r["view"], method=r["method"], **row)
        for r in records
        for row in r["strata"]
    ]
    with provenance("sleap_adapter_summary", output) as rec:
        rec.input("protocol", output / "protocol.json", sha256=sha256(output / "protocol.json"))
        result = dict(
            complete=True,
            protocol=protocol,
            records=records,
            totals=totals,
            provisional_records=provisional,
            identity_accuracy_claim=False,
            verified_identity_denominator=0,
            parameter_selection_performed=False,
            all_cases_previously_inspected=True,
        )
        for name in ("summary.json", "metrics.json"):
            write_json(output / name, result)
        excluded = ("metadata", "strata", "provisional_metrics", "input_array_sha256")
        write_csv(
            output / "metrics.csv",
            [{k: v for k, v in r.items() if k not in excluded} for r in records],
        )
        write_csv(output / "provisional_metrics.csv", provisional)
        write_json(output / "provisional_metrics.json", dict(records=provisional))
        write_csv(output / "strata.csv", strata)
        write_json(output / "strata.json", dict(records=strata))
        snapshot = output / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        for name in FROZEN_FILES:
            shutil.copy2(HERE / name, snapshot / name)
        names = (
            "summary.json",
            "metrics.json",
            "metrics.csv",
            "provisional_metrics.csv",
            "provisional_metrics.json",
            "strata.csv",
            "strata.json",
        )
        for name in names:
            rec.output(name, output / name, sha256=sha256(output / name))
        # Explicit numerical filenames avoid capturing concurrently changing HTML/media/audits.
        files = [output / "protocol.json", *(output / name for name in names)]
        files.extend(snapshot / name for name in FROZEN_FILES)
        for case in protocol["cases"]:
            dest = output / case["name"]
            for view in VIEWS:
                files.extend(
                    dest / name
                    for name in (
                        f"metrics_{view}.json",
                        f"input_contract_{view}.json",
                        f"detections_{view}.npz",
                    )
                )
                for method in CONDITIONS:
                    files.extend(
                        dest / name
                        for name in (
                            f"tracking_{method}_{view}.npz",
                            f"evidence_{method}_{view}.json.gz",
                            f"metrics_{method}_{view}.json",
                            f"timeline_{method}_{view}.npz",
                        )
                    )
        manifest = {
            str(path.relative_to(output)).replace("\\", "/"): sha256(path) for path in files
        }
        write_json(
            output / "numerical_manifest.json",
            dict(scope="explicit immutable numerical files", sha256=manifest),
        )
        rec.output(
            "numerical_manifest",
            output / "numerical_manifest.json",
            sha256=sha256(output / "numerical_manifest.json"),
        )
    print(
        f"Complete: {len(records)} confirmed/reference and {len(provisional)} provisional records",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--case", action="append")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.init:
        initialize(args.output)
    elif args.view:
        run_view(args.output, args.view, args.case)
    elif args.summarize:
        summarize(args.output)
    else:
        parser.error("Choose --init, --view, or --summarize")


if __name__ == "__main__":
    main()

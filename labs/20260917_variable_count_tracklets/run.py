"""Freeze protocol and execute identical-context variable-count 2D comparisons."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
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
from baseline import predecessors, replay
from frozen_p0 import TrackingConfig, build_tracklets

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = ROOT / "work/variable_count_tracklets_v1"
EXPERIMENT = "20251225_K_1_2_FCMC"
VIEWS = ("top_cam", "side1_cam", "side2_cam")
OLD = ROOT / "work/clean2d_video_comparison_v1"
RAW = ROOT / "work/generative_tracking_inputs/work/c01_s16_predictions/bottomup"
C27 = ROOT / "work/custom2d_comparison_v1/inputs/c27_uncapped_tracklets_any" / EXPERIMENT
C27 = C27 / "mouth-pectoral_oks_serial"
METHODS = ("C27", "S1", "S2", "P0", "P1", "P1aux", "P1_margin_low", "P1_margin_high")
NUMERICAL_CODE = ("baseline.py", "frozen_p0.py", "tracking.py", "run.py")


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path, rows):
    rows = [jsonable(row) for row in rows]
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                }
            )


def sensitivity():
    from tracking import P1Config

    return {
        "P1_margin_low": replace(P1Config(), margin_fraction=0.015),
        "P1_margin_high": replace(P1Config(), margin_fraction=0.035),
    }


def initialize(output, unseen_path):
    from tracking import P1Config

    if (output / "protocol.json").exists():
        raise ValueError("Protocol exists; freeze only once, then use --view")
    original = json.loads((OLD / "protocol.json").read_text(encoding="utf-8"))
    unseen = json.loads(unseen_path.read_text(encoding="utf-8"))
    cases = original["cases"] + [unseen["selected_case"]]
    original_p0 = ROOT / "labs/20260917_clean2d_video_comparison/tracking.py"
    assert sha256(original_p0) == sha256(HERE / "frozen_p0.py")
    protocol = dict(
        experiment=EXPERIMENT,
        fps=166,
        context=332,
        cases=cases,
        frozen_at_utc=datetime.now(UTC).isoformat(),
        stage="A-C",
        geometry_used=False,
        input_rows_removed=False,
        identity_accuracy_claim=False,
        n_independent_groups=1,
        conditions={
            "C27": "saved historical full-session SLEAP C27 (reference only)",
            "S1": "current local SLEAP, correct object-to-input-row readout",
            "S2": "S1 with only fixed OKS area top 14400 / side 10000 px2",
            "P0": "byte-frozen existing pectoral-only forward/backward tracker",
            "P1": "P-primary relative competition, variable count, expiring open endpoints",
            "P1aux": "P1 plus bounded coherent observed M/T support, no missing-node veto",
            "P1_margin_low": "P1 margin_fraction .015 (only change)",
            "P1_margin_high": "P1 margin_fraction .035 (only change)",
        },
        p0_config=asdict(TrackingConfig()),
        p1_config=asdict(P1Config()),
        sensitivity_configs={k: asdict(v) for k, v in sensitivity().items()},
        sensitivity_policy=(
            "All cases and settings reported; no scene-dependent selection or tuning"
        ),
        s2_scale=dict(
            top_body_px=120,
            side_body_px=100,
            area="body_px squared",
            sigma=0.025,
            denominator="2*(2*sigma)^2*(area+eps)",
            top_denominator_px2=72,
            side_denominator_px2=50,
            direction_independent=True,
            other_tracker_settings_changed=False,
        ),
        same_input_context="[core_start-332,core_stop+332); SLEAP uses only past history",
        clean_edge_definition="same-label adjacent-native-frame predecessor; no gap links",
        metric_edge_scope="both endpoints inside core; first core frame excluded from denominator",
        row_definition="view + native frame + C01 global_row; all raw rows retained",
        node_score_policy=(
            "input only provides instance score; no node score synthesized in outputs"
        ),
        reflection_policy="all detections unclassified; continuity is not real-fish probability",
        unseen_selection=unseen,
        source_protocol_sha256=sha256(OLD / "protocol.json"),
        frozen_p0_source=str(original_p0.relative_to(ROOT)),
        frozen_p0_sha256=sha256(original_p0),
        s1_adaptation_source="labs/20260917_clean2d_video_comparison/c27_replay.py",
        s1_adaptation_source_sha256=sha256(
            ROOT / "labs/20260917_clean2d_video_comparison/c27_replay.py"
        ),
        code_sha256={name: sha256(HERE / name) for name in NUMERICAL_CODE},
        package_versions={
            name: importlib.metadata.version(name)
            for name in ("sleap-nn", "sleap-io", "numpy", "scipy")
        },
        annotation_scope="Historical notes are not complete per-detection 2D identity truth",
    )
    output.mkdir(parents=True, exist_ok=True)
    with provenance("variable_count_protocol_freeze", output) as rec:
        for key, path in [
            ("old_protocol", OLD / "protocol.json"),
            ("old_summary", OLD / "summary.json"),
            ("unseen_selection", unseen_path),
            ("frozen_p0_source", original_p0),
        ]:
            rec.input(key, path, sha256=sha256(path))
        for path in HERE.glob("*.py"):
            rec.input("code_" + path.name, path, sha256=sha256(path))
        write_json(output / "protocol.json", protocol)
        rec.output("protocol", output / "protocol.json", sha256=sha256(output / "protocol.json"))
    return protocol


def frame_statistics(frames, points, start, stop, span):
    native = np.arange(start, stop)
    counts = np.diff(np.searchsorted(frames, np.arange(start, stop + 1)))
    missing = np.zeros((len(native), 3), dtype=int)
    nearest = np.full(len(frames), np.inf)
    for frame in native:
        lo, hi = np.searchsorted(frames, [frame, frame + 1])
        missing[frame - start] = (~np.isfinite(points[lo:hi]).all(axis=-1)).sum(axis=0)
        xy = points[lo:hi, 1]
        distance = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        np.fill_diagonal(distance, np.inf)
        distance[~np.isfinite(distance)] = np.inf
        if hi > lo:
            nearest[lo:hi] = distance.min(axis=1)
    proximity = nearest < 0.08 * span
    return native, counts, missing, nearest, proximity


def measure(frames, points, labels, core, start, stop, span, events):
    previous, gaps = predecessors(frames, labels)
    eligible = core & (frames > start)
    adjacent = gaps == 1
    # No edge into core from context is counted in interval-only metrics.
    links = adjacent & eligible
    native, counts, missing, nearest, proximity = frame_statistics(
        frames, points, start, stop, span
    )
    observed = np.isfinite(points).all(axis=-1)
    node_missing = ~observed.all(axis=1)
    rows_per_frame = dict(zip(native.tolist(), counts.tolist(), strict=True))
    row_count = np.array([rows_per_frame.get(int(f), 0) for f in frames])
    ids, id_counts = np.unique(labels[core & (labels >= 0)], return_counts=True)
    steps = np.full(len(frames), np.nan)
    good = previous >= 0
    steps[good] = np.linalg.norm(points[good, 1] - points[previous[good], 1], axis=-1)
    strata = []
    for count in ("0", "1", "2", "3", "4+"):
        count_mask = row_count >= 4 if count == "4+" else row_count == int(count)
        frame_count = (
            int((counts >= 4).sum()) if count == "4+" else int((counts == int(count)).sum())
        )
        for near in (False, True):
            for absent in (False, True):
                mask = eligible & count_mask & (proximity == near) & (node_missing == absent)
                n = int(mask.sum())
                accepted = int((links & mask).sum())
                strata.append(
                    dict(
                        detection_count=count,
                        proximity="near" if near else "separated",
                        any_node_missing=absent,
                        frame_count_count_stratum=frame_count,
                        eligible_observations=n,
                        adjacent_connections=accepted,
                        unconnected=n - accepted,
                        connection_rate=accepted / n if n else None,
                    )
                )
    denominator = int(eligible.sum())
    nlinks = int(links.sum())
    return dict(
        core_frames=stop - start,
        core_rows=int(core.sum()),
        retained_rows=int(core.sum()),
        retention_rate=1.0,
        context_rows=len(frames),
        frame_start=start,
        frame_stop=stop,
        eligible_observations=denominator,
        adjacent_connections=nlinks,
        adjacent_links=nlinks,
        unconnected_observations=denominator - nlinks,
        connection_rate=nlinks / denominator if denominator else None,
        abstention_rate=(denominator - nlinks) / denominator if denominator else None,
        contiguous_runs=int(core.sum()) - nlinks,
        unique_local_ids=len(ids),
        isolated_core_ids=int((id_counts == 1).sum()),
        gap_connections=int((eligible & (gaps > 1)).sum()),
        unassigned_rows=int((core & (labels < 0)).sum()),
        large_adjacent_steps=int((links & (steps > 0.3 * span)).sum()),
        maximum_adjacent_step_px=float(np.nanmax(steps[links])) if links.any() else None,
        missing_m=int(missing[:, 0].sum()),
        missing_p=int(missing[:, 1].sum()),
        missing_t=int(missing[:, 2].sum()),
        detection_count_frames={str(k): int((counts == k).sum()) for k in (0, 1, 2, 3)}
        | {"4+": int((counts >= 4).sum())},
        refusal_reasons=dict(
            Counter(str(e.get("reason")) for e in events if e.get("reason") is not None)
        ),
        refusal_event_scope=("both directional passes over full context; may repeat one endpoint"),
        strata=strata,
        identity_accuracy=None,
        verified_correct_connections=None,
        verified_wrong_connections=None,
        verified_identity_denominator=0,
        truth_status="unknown; no new human labels inferred from smoothness or old 3D slots",
    ), dict(
        native_frame=native,
        native_seconds=native / 166,
        detection_count=counts,
        missing_m=missing[:, 0],
        missing_p=missing[:, 1],
        missing_t=missing[:, 2],
        nearest_other_pectoral_px=nearest,
        close_rows=proximity,
        adjacent_connections_per_frame=np.bincount(frames[links] - start, minlength=stop - start),
        unconnected_per_frame=np.bincount(
            frames[eligible & ~adjacent] - start, minlength=stop - start
        ),
    )


def save_tables(dest, stem, tables):
    for name, rows in tables.items():
        write_json(dest / f"{name}_{stem}.json", rows)
        write_csv(dest / f"{name}_{stem}.csv", rows)


def run_view(output, view, selected=None):
    from tracking import P1Config, track

    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    for name, digest in protocol["code_sha256"].items():
        if sha256(HERE / name) != digest:
            raise ValueError(f"Code changed after protocol freeze: {name}; use a fresh version")
    source = RAW / f"{EXPERIMENT}__{view}.npz"
    with np.load(source, allow_pickle=False) as z:
        raw = {k: z[k] for k in ("frame_idx", "points", "score")}
    saved = np.load(C27 / f"track2d__{view}.npy", allow_pickle=False)
    if saved.shape != raw["frame_idx"].shape:
        raise AssertionError("Saved C27 is not aligned to raw detection rows")
    with provenance("variable_count_2d_runs", output, name="run_" + view) as rec:
        rec.input("raw", source, sha256=sha256(source))
        rec.input(
            "saved_c27", C27 / f"track2d__{view}.npy", sha256=sha256(C27 / f"track2d__{view}.npy")
        )
        rec.input("protocol", output / "protocol.json", sha256=sha256(output / "protocol.json"))
        rec.param("view", view)
        for path in HERE.glob("*.py"):
            rec.input("code_" + path.name, path, sha256=sha256(path))
        import sleap_nn.tracking.tracker as installed

        package = Path(installed.__file__).parent
        for relative in (
            "tracker.py",
            "candidates/local_queues.py",
            "utils.py",
            "../evaluation.py",
        ):
            path = (package / relative).resolve()
            rec.input("installed_" + relative, path, sha256=sha256(path))
        for case in protocol["cases"]:
            if selected and case["name"] not in selected:
                continue
            dest = output / case["name"]
            done = dest / f"metrics_{view}.json"
            if done.exists():
                print(f"Already complete {case['name']} {view}", flush=True)
                continue
            dest.mkdir(parents=True, exist_ok=True)
            start, stop = case["start"], case["stop"]
            lo, hi = np.searchsorted(raw["frame_idx"], [start - 332, stop + 332])
            frames, points, score = (raw[k][lo:hi] for k in ("frame_idx", "points", "score"))
            rows = np.arange(lo, hi, dtype=np.int64)
            core = (frames >= start) & (frames < stop)
            data = dict(
                frame_idx=frames,
                native_frame=frames,
                global_row=rows,
                global_rows=rows,
                points=points,
                score=score,
                observed_mask=np.isfinite(points).all(axis=-1),
                core_mask=core,
                interpretation=np.full(len(frames), "unclassified", dtype="U12"),
            )
            np.savez_compressed(dest / f"detections_{view}.npz", **data)
            metrics = []
            for method in METHODS:
                began = time.perf_counter()
                target = dest / f"tracking_{method}_{view}.npz"
                events, tables, metadata = [], {}, {}
                if method == "C27":
                    labels = saved[lo:hi]
                    metadata = dict(saved_full_session=True, historical_runtime_reproduced=False)
                elif method in ("S1", "S2"):
                    labels, legacy, metadata = replay(
                        frames,
                        points,
                        score,
                        start - 332,
                        stop + 332,
                        view=view,
                        fixed_scale=method == "S2",
                    )
                    np.savez_compressed(
                        dest / f"readout_audit_{method}_{view}.npz",
                        frame_idx=frames,
                        global_row=rows,
                        legacy_labels=legacy,
                        faithful_labels=labels,
                    )
                    metadata["legacy_readout_different_rows_core"] = int(
                        ((labels != legacy) & core).sum()
                    )
                    prior = OLD / "c27_replay" / case["name"] / f"{view}.npz"
                    if method == "S1" and prior.exists():
                        with np.load(prior, allow_pickle=False) as z:
                            checks = {
                                key: bool(np.array_equal(z[key], value))
                                for key, value in (
                                    ("frame_idx", frames),
                                    ("global_rows", rows),
                                    ("row_faithful_labels", labels),
                                )
                            }
                        if not all(checks.values()):
                            raise AssertionError(f"S1 local replay regression failed: {checks}")
                        metadata["prior_local_faithful_reproduction"] = checks
                        rec.input(case["name"] + "_prior_S1", prior, sha256=sha256(prior))
                elif method == "P0":
                    result = build_tracklets(frames, points, view=view, feature_mode="pectoral")
                    labels, metadata, events = result.labels, result.metadata, result.events
                    old = OLD / case["name"] / f"tracking_P_{view}.npz"
                    if old.exists():
                        with np.load(old, allow_pickle=False) as z:
                            checks = {
                                k: bool(np.array_equal(z[k], value, equal_nan=True))
                                for k, value in [
                                    ("frame_idx", frames),
                                    ("points", points),
                                    ("global_rows", rows),
                                    ("labels", labels),
                                    ("clean_edge", result.clean_edge),
                                ]
                            }
                        if not all(checks.values()):
                            raise AssertionError(f"P0 failed exact frozen reproduction: {checks}")
                        metadata["prior_P_exact_reproduction"] = checks
                        metadata["prior_P_sha256"] = sha256(old)
                        rec.input(case["name"] + "_prior_P", old, sha256=sha256(old))
                else:
                    result = track(
                        frames,
                        points,
                        rows,
                        view,
                        config=sensitivity().get(method, P1Config()),
                        auxiliary=method == "P1aux",
                    )
                    labels, metadata, events = result.labels, result.metadata, result.events
                    tables = dict(
                        edges=result.edges,
                        predictions=result.predictions,
                        reconnections=result.reconnect_candidates,
                        node_quality=result.node_quality,
                    )
                previous, gaps = predecessors(frames, labels)
                if not tables:
                    tables["edges"] = [
                        dict(
                            from_global_row=int(rows[previous[r]]),
                            to_global_row=int(rows[r]),
                            from_frame=int(frames[previous[r]]),
                            to_frame=int(frames[r]),
                            gap_frames=int(gaps[r]),
                            accepted=True,
                            reason="saved_label_continuity"
                            if method == "C27"
                            else "method_connection",
                        )
                        for r in np.flatnonzero(previous >= 0)
                    ]
                np.savez_compressed(
                    target,
                    **data,
                    labels=labels,
                    local_id=labels,
                    clean_edge=gaps == 1,
                    previous_row=previous,
                    link_gap=gaps,
                )
                write_json(target.with_suffix(".json"), dict(metadata=metadata, events=events))
                tables["events"] = events
                save_tables(dest, f"{method}_{view}", tables)
                stats, timeline = measure(
                    frames,
                    points,
                    labels,
                    core,
                    start,
                    stop,
                    120 if view == "top_cam" else 100,
                    events,
                )
                stats.update(
                    case=case["name"],
                    view=view,
                    method=method,
                    condition=method,
                    elapsed_seconds=time.perf_counter() - began,
                    context_start=start - 332,
                    context_stop=stop + 332,
                    prior_P_exact_reproduction=metadata.get("prior_P_exact_reproduction"),
                    readout_changed_core=metadata.get("legacy_readout_different_rows_core", 0),
                    metadata=metadata,
                )
                np.savez_compressed(dest / f"timeline_{method}_{view}.npz", **timeline)
                write_json(dest / f"metrics_{method}_{view}.json", stats)
                metrics.append(stats)
                print(
                    f"{case['name']} {view} {method}: {stats['contiguous_runs']} runs, "
                    f"{stats['core_rows']} rows, {stats['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            write_json(
                done,
                dict(
                    case=case["name"],
                    view=view,
                    records=metrics,
                    same_detection_rows_all_conditions=True,
                ),
            )
            for path in dest.glob(f"*{view}.*"):
                rec.output(case["name"] + "_" + path.name, path, sha256=sha256(path))


def summarize(output):
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    records = []
    for case in protocol["cases"]:
        for view in VIEWS:
            path = output / case["name"] / f"metrics_{view}.json"
            records.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    with provenance("variable_count_summary", output) as rec:
        rec.input("protocol", output / "protocol.json", sha256=sha256(output / "protocol.json"))
        totals = []
        for method in METHODS:
            selected = [r for r in records if r["method"] == method]
            fields = (
                "core_rows",
                "retained_rows",
                "eligible_observations",
                "adjacent_connections",
                "unconnected_observations",
                "contiguous_runs",
                "gap_connections",
            )
            entry = dict(method=method, case_views=len(selected))
            entry.update({k: sum(r[k] for r in selected) for k in fields})
            entry["connection_rate"] = (
                entry["adjacent_connections"] / entry["eligible_observations"]
            )
            entry["abstention_rate"] = 1 - entry["connection_rate"]
            totals.append(entry)
        result = dict(
            complete=True,
            protocol=protocol,
            records=records,
            totals=totals,
            verified_identity_denominator=0,
            identity_accuracy_claim=False,
        )
        write_json(output / "metrics.json", result)
        write_json(output / "summary.json", result)
        write_csv(
            output / "metrics.csv",
            [{k: v for k, v in r.items() if k not in ("metadata", "strata")} for r in records],
        )
        strata = [
            dict(case=r["case"], view=r["view"], method=r["method"], **s)
            for r in records
            for s in r["strata"]
        ]
        write_csv(output / "strata.csv", strata)
        snapshot = output / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        for path in HERE.glob("*.py"):
            shutil.copy2(path, snapshot / path.name)
        for name in ("summary.json", "metrics.json", "metrics.csv", "strata.csv"):
            rec.output(name, output / name, sha256=sha256(output / name))
        numerical_files = [
            output / name
            for name in (
                "protocol.json",
                "metrics.json",
                "summary.json",
                "metrics.csv",
                "strata.csv",
            )
        ]
        for case in protocol["cases"]:
            numerical_files.extend(
                p
                for p in (output / case["name"]).iterdir()
                if p.is_file() and p.suffix in (".json", ".csv", ".npz")
            )
        numerical_files.extend(snapshot / name for name in NUMERICAL_CODE)
        manifest = {str(p.relative_to(output)): sha256(p) for p in numerical_files}
        write_json(output / "numerical_manifest.json", manifest)
        rec.output(
            "manifest",
            output / "numerical_manifest.json",
            sha256=sha256(output / "numerical_manifest.json"),
        )
    print(f"Complete: {len(records)} method-case-view records", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--unseen-selection", type=Path)
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--case", action="append")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.init:
        initialize(args.output, args.unseen_selection)
    elif args.summarize:
        summarize(args.output)
    elif args.view:
        run_view(args.output, args.view, args.case)
    else:
        parser.error("Choose --init, --view, or --summarize")


if __name__ == "__main__":
    main()

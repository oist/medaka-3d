"""Measure direct 2D associations on identical detections; never read 3D output."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
from flow import extract_flow
from tracking import TrackingConfig, build_tracklets

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = ROOT / "work/clean2d_video_comparison_v1"
EXPERIMENT = "20251225_K_1_2_FCMC"
VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIDEOS = ("cam0_top.mp4", "cam1_side.mp4", "cam2_side.mp4")
CASE_NAMES = (
    "cg20s",
    "known_21232",
    "known_30735",
    "known_31479",
    "known_105541",
    "known_260434",
    "known_302762",
    "known_323819",
    "known_333619",
    "control_50000",
    "holdout20s",
)
PREVIOUS = ROOT / "work/clean_tracklets_v2"
C27 = ROOT / (
    "work/custom2d_comparison_v1/inputs/c27_uncapped_tracklets_any/"
    + EXPERIMENT
    + "/mouth-pectoral_oks_serial"
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sensitivity_configs():
    """Frozen before results; one factor family at a time, plus joint changes."""
    base = TrackingConfig()
    motion = (
        "speed_fraction",
        "prediction_fraction",
        "heading_step_fraction",
        "heading_error_fraction",
    )
    nodes = (
        "mouth_relative_motion_fraction",
        "tail_relative_motion_fraction",
        "length_change_fraction",
        "axis_change_deg",
    )
    ambiguity = ("assignment_margin_fraction", "proximity_fraction")
    flow = ("flow_error_fraction",)
    families = dict(motion=motion, nodes=nodes, ambiguity=ambiguity, flow=flow)

    def adjusted(selected, loose):
        changes = {}
        for family in selected:
            factor = (
                (0.5 if family == "ambiguity" else 2.0)
                if loose
                else (2.0 if family == "ambiguity" else 0.5)
            )
            for key in families[family]:
                changes[key] = getattr(base, key) * factor
        return replace(base, **changes)

    return {
        "F_motion_loose": adjusted(["motion"], True),
        "F_nodes_loose": adjusted(["nodes"], True),
        "F_ambiguity_loose": adjusted(["ambiguity"], True),
        "F_flow_loose": adjusted(["flow"], True),
        "F_all_loose": adjusted(families, True),
        "F_all_strict": adjusted(families, False),
    }


def initialize(output):
    old_protocol = ROOT / "work/custom2d_comparison_v1/analysis_v3/protocol.json"
    original = json.loads(old_protocol.read_text(encoding="utf-8"))["cases"]
    cases = [dict(next(c for c in original if c["name"] == name)) for name in CASE_NAMES]
    for case in cases:
        case["render"] = case["name"] not in ("control_50000", "holdout20s")
        case["centers"] = (
            [76235, 76791, 76930, 76988, 77415]
            if case["name"] == "cg20s"
            else [int(case["note"]["frame_start"])]
            if "note" in case
            else []
        )
    protocol = dict(
        experiment=EXPERIMENT,
        fps=166,
        context=332,
        cases=cases,
        conditions={
            "SLEAP": "saved full-session C27 mouth-pectoral OKS tracking",
            "P": "independent pectoral",
            "N": "independent nodes",
            "F": "independent nodes plus bidirectional optical flow",
        },
        tracking_config=asdict(TrackingConfig()),
        sensitivity_configs={k: asdict(v) for k, v in sensitivity_configs().items()},
        sensitivity_policy=(
            "All settings reported; no winner selected; retrospective scenes, n_groups=1"
        ),
        tracking_sha256=sha256(HERE / "tracking.py"),
        flow_sha256=sha256(HERE / "flow.py"),
        algorithm_source="byte-exact copies of labs/20260917_clean_tracklets/{tracking,flow}.py",
        source_protocol_sha256=sha256(old_protocol),
        baseline_source=str(C27.relative_to(ROOT)),
        clean_edge_definition=(
            "same-label predecessor exists in adjacent native frame; excludes gap links"
        ),
        previous_row_definition=(
            "previous observation with same nonnegative label within context; -1 if none"
        ),
        geometry_used=False,
        input_rows_removed=False,
        identity_accuracy_claim=False,
    )
    for file in ("tracking.py", "flow.py"):
        assert sha256(HERE / file) == sha256(ROOT / "labs/20260917_clean_tracklets" / file)
    target = output / "protocol.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Frozen protocol changed; use a fresh output directory")
    write_json(target, protocol)
    return protocol


def predecessors(frames, labels):
    """Unassigned rows have no identity; repeated frame/identity is invalid."""
    previous = np.full(len(frames), -1, dtype=np.int64)
    last = {}
    for row, (frame, label) in enumerate(zip(frames, labels, strict=True)):
        if label < 0:
            continue
        if int(label) in last:
            previous[row] = last[int(label)]
            if frames[previous[row]] >= frame:
                raise ValueError("Same identity used twice in one frame")
        last[int(label)] = row
    gaps = np.zeros(len(frames), dtype=np.int64)
    ok = previous >= 0
    gaps[ok] = frames[ok] - frames[previous[ok]]
    return previous, gaps


def measure(frames, points, labels, start, stop, baseline_previous=None, events=()):
    previous, gaps = predecessors(frames, labels)
    core = (frames >= start) & (frames < stop)
    # Boundary links from warm-up are not included in interval-only denominators.
    eligible = core & (frames > start)
    adjacent = gaps == 1
    unique, counts = np.unique(frames[core], return_counts=True)
    node_valid = np.isfinite(points).all(axis=-1)
    ids, id_counts = np.unique(labels[core & (labels >= 0)], return_counts=True)
    lengths = []
    live = {}
    for row in np.flatnonzero(core):
        label = int(labels[row])
        if label < 0:
            lengths.append(1)
            continue
        if label in live and frames[row] == live[label][0] + 1:
            live[label] = (int(frames[row]), live[label][1] + 1)
        else:
            if label in live:
                lengths.append(live[label][1])
            live[label] = (int(frames[row]), 1)
    lengths.extend(value[1] for value in live.values())
    links = np.flatnonzero(eligible & adjacent)
    step = np.linalg.norm(points[links] - points[previous[links]], axis=-1)
    large = []
    for j, row in enumerate(links):
        if np.any(step[j, :2] > 30):
            large.append(
                dict(
                    from_row=int(previous[row]),
                    to_row=int(row),
                    from_frame=int(frames[previous[row]]),
                    to_frame=int(frames[row]),
                    mouth_px=float(step[j, 0]) if np.isfinite(step[j, 0]) else None,
                    pectoral_px=float(step[j, 1]) if np.isfinite(step[j, 1]) else None,
                )
            )
    node_stats = {}
    for i, node in enumerate(("mouth", "pectoral", "tail")):
        values = step[:, i][np.isfinite(step[:, i])]
        node_stats[node] = dict(
            missing_rows=int(np.count_nonzero(core & ~node_valid[:, i])),
            max_adjacent_step_px=float(values.max()) if len(values) else None,
            steps_above_30px=int(np.count_nonzero(values > 30)),
            finite_adjacent_steps=len(values),
        )
    n = int(core.sum())
    stats = dict(
        frame_start=start,
        frame_stop=stop,
        native_frames=stop - start,
        detection_rows=n,
        retained_rows=n,
        unassigned_rows=int(np.count_nonzero(core & (labels < 0))),
        frames_without_detections=stop - start - len(unique),
        detections_per_frame={
            str(i): int(np.count_nonzero(counts == i)) for i in np.unique(counts)
        },
        frames_below_three=int(np.count_nonzero(counts < 3)) + stop - start - len(unique),
        missing_primary_rows=int(np.count_nonzero(core & ~node_valid[:, :2].all(axis=-1))),
        unique_ids=len(ids),
        ids_with_single_core_observation=int(np.count_nonzero(id_counts == 1)),
        contiguous_runs=len(lengths),
        median_run_frames=float(np.median(lengths)) if lengths else None,
        singleton_runs=sum(length == 1 for length in lengths),
        adjacent_links=len(links),
        adjacent_link_denominator=int(eligible.sum()),
        adjacent_link_fraction=float(len(links) / eligible.sum()) if eligible.any() else None,
        gap_links=int(np.count_nonzero(eligible & (gaps > 1))),
        interval_fragment_starts=n - len(links),
        nodes=node_stats,
        large_primary_steps=large,
    )
    assert stats["contiguous_runs"] == stats["interval_fragment_starts"]
    cuts = []
    if baseline_previous is not None:
        baseline_previous = np.asarray(baseline_previous)
        baseline_adjacent = np.zeros(len(frames), dtype=bool)
        good = baseline_previous >= 0
        baseline_adjacent[good] = frames[good] - frames[baseline_previous[good]] == 1
        base_edges = eligible & baseline_adjacent
        same = base_edges & (previous == baseline_previous)
        changes = base_edges & adjacent & (previous != baseline_previous)
        cut = base_edges & ~adjacent
        stats["sleap_adjacent_links"] = int(base_edges.sum())
        stats["sleap_adjacent_reproduced"] = int(same.sum())
        stats["sleap_adjacent_cut"] = int(cut.sum())
        stats["sleap_adjacent_different_predecessor"] = int(changes.sum())
        stats["new_adjacent_without_sleap_adjacent"] = int(
            np.count_nonzero(eligible & adjacent & ~baseline_adjacent)
        )
        by_edge = defaultdict(set)
        by_row = defaultdict(set)
        for event in events:
            if event["kind"] == "unsafe_detection":
                by_row[int(event["row"])].update(event["reasons"])
            elif event["kind"] in ("directional_link_refused", "directional_disagreement"):
                a, b = event.get("from_row"), event.get("to_row")
                if a is not None and b is not None:
                    by_edge[tuple(sorted((int(a), int(b))))].add(event.get("reason", event["kind"]))
        reasons = Counter()
        for row in np.flatnonzero(cut | changes):
            origin = int(baseline_previous[row])
            why = sorted(by_edge[(origin, int(row))] | by_row[origin] | by_row[int(row)])
            if changes[row]:
                why.append("different_predecessor")
            if not why:
                why = ["different_history_or_unrecorded_candidate_rejection"]
            reasons.update(why)
            cuts.append(
                dict(
                    from_row=origin,
                    to_row=int(row),
                    from_frame=int(frames[origin]),
                    to_frame=int(frames[row]),
                    reasons=why,
                    new_previous_row=int(previous[row]),
                    pectoral_step_px=float(np.linalg.norm(points[row, 1] - points[origin, 1]))
                    if np.isfinite(points[[origin, row], 1]).all()
                    else None,
                )
            )
        stats["cut_reason_counts_nonexclusive"] = dict(reasons)
    return stats, cuts


def save_tracking(path, frames, points, labels, global_rows, metadata, events=(), clean_edge=None):
    previous, gaps = predecessors(frames, labels)
    adjacent = gaps == 1
    if clean_edge is not None and not np.array_equal(clean_edge, adjacent):
        raise AssertionError("Frozen tracker clean_edge differs from consecutive-label adjacency")
    np.savez_compressed(
        path,
        frame_idx=frames,
        points=points,
        labels=labels,
        clean_edge=adjacent,
        previous_row=previous,
        link_gap=gaps,
        global_rows=global_rows,
    )
    write_json(path.with_suffix(".json"), dict(metadata=metadata, events=events))


def run_view(output, view, selected=None):
    cv2.setNumThreads(1)
    protocol = initialize(output)
    source = (
        ROOT
        / "work/generative_tracking_inputs/work/c01_s16_predictions/bottomup"
        / f"{EXPERIMENT}__{view}.npz"
    )
    c27_path = C27 / f"track2d__{view}.npy"
    with np.load(source, allow_pickle=False) as data:
        all_frames, all_points = data["frame_idx"], data["points"]
    sleap = np.load(c27_path, allow_pickle=False)
    if sleap.shape != all_frames.shape:
        raise AssertionError("Full C27 row correspondence differs")
    video = (
        ROOT
        / "work/three_body/source_cache/medaka_202512"
        / EXPERIMENT
        / view
        / VIDEOS[VIEWS.index(view)]
    )
    with provenance("direct_clean2d_comparison", output, name="run_" + view) as rec:
        rec.input("untracked_detections", source, sha256=sha256(source))
        rec.input("saved_sleap", c27_path, sha256=sha256(c27_path))
        rec.input("protocol", output / "protocol.json", sha256=sha256(output / "protocol.json"))
        rec.param("view", view)
        for case in protocol["cases"]:
            name, start, stop = case["name"], case["start"], case["stop"]
            if selected and name not in selected:
                continue
            dest = output / name
            dest.mkdir(parents=True, exist_ok=True)
            summary_path = dest / f"metrics_{view}.json"
            if summary_path.exists():
                print(f"Reuse {name} {view}", flush=True)
                continue
            lo, hi = np.searchsorted(all_frames, [start - 332, stop + 332])
            frames, points = all_frames[lo:hi], all_points[lo:hi]
            rows = np.arange(lo, hi, dtype=np.int64)
            labels = sleap[lo:hi]
            save_tracking(
                dest / f"tracking_SLEAP_{view}.npz",
                frames,
                points,
                labels,
                rows,
                dict(source=str(c27_path), temporal_ids_used=True, saved_full_session=True),
            )
            baseline_previous, _ = predecessors(frames, labels)
            measurements = {}
            measurements["SLEAP"], _ = measure(frames, points, labels, start, stop)
            old_cache = ROOT / "work/clean_tracklets_v1" / name / f"flow_{view}.npz"
            cache = dest / f"flow_{view}.npz"
            if old_cache.exists() and not cache.exists():
                shutil.copy2(old_cache, cache)
                rec.input(name + "_flow_reuse", old_cache, sha256=sha256(old_cache))
            print(f"Flow {name} {view}", flush=True)
            flow = extract_flow(video, frames, points, cache_path=cache)
            for arm in ("P", "N", "F", *sensitivity_configs()):
                target = dest / f"tracking_{arm}_{view}.npz"
                old = PREVIOUS / name / target.name
                metadata = {}
                if arm in ("P", "N", "F") and old.exists():
                    with np.load(old, allow_pickle=False) as z:
                        for key, expected in (
                            ("frame_idx", frames),
                            ("points", points),
                            ("global_rows", rows),
                        ):
                            if not np.array_equal(z[key], expected, equal_nan=True):
                                raise AssertionError("Existing tracking interval or rows mismatch")
                        labels, clean = z["labels"], z["clean_edge"]
                    held = json.loads(old.with_suffix(".json").read_text(encoding="utf-8"))
                    metadata = {
                        **held["metadata"],
                        "reuse_source": str(old),
                        "reuse_source_sha256": sha256(old),
                    }
                    events = held["events"]
                else:
                    cfg = sensitivity_configs().get(arm, TrackingConfig())
                    result = build_tracklets(
                        frames,
                        points,
                        view=view,
                        feature_mode="pectoral" if arm == "P" else "nodes",
                        flow=None if arm in ("P", "N") else flow,
                        config=cfg,
                    )
                    labels, clean, events, metadata = (
                        result.labels,
                        result.clean_edge,
                        result.events,
                        result.metadata,
                    )
                save_tracking(target, frames, points, labels, rows, metadata, events, clean)
                stats, cuts = measure(
                    frames, points, labels, start, stop, baseline_previous, events
                )
                measurements[arm] = stats
                write_json(dest / f"cuts_{arm}_{view}.json", cuts)
                rec.output(name + "_" + arm, target, sha256=sha256(target))
                print(
                    f"{name} {view} {arm}: {stats['contiguous_runs']} runs, "
                    f"{stats['sleap_adjacent_cut']} cuts, "
                    f"{stats['sleap_adjacent_different_predecessor']} changed",
                    flush=True,
                )
            write_json(
                summary_path,
                dict(
                    case=name,
                    view=view,
                    frame_start=start,
                    frame_stop=stop,
                    same_detection_rows_all_arms=True,
                    arms=measurements,
                ),
            )
            rec.output(name + "_metrics", summary_path, sha256=sha256(summary_path))


def summarize(output):
    protocol = initialize(output)
    records = []
    for case in protocol["cases"]:
        for view in VIEWS:
            path = output / case["name"] / f"metrics_{view}.json"
            records.append(json.loads(path.read_text(encoding="utf-8")))
    write_json(output / "summary.json", dict(protocol=protocol, complete=True, records=records))
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in ("run.py", "tracking.py", "flow.py", "README.md"):
        shutil.copy2(HERE / name, snapshot / name)
    print(f"Complete: {len(records)} case-view measurements", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    if args.init:
        initialize(args.output)
    elif args.summarize:
        summarize(args.output)
    elif args.view:
        run_view(args.output, args.view, args.case)
    else:
        parser.error("Choose --init, --view or --summarize")


if __name__ == "__main__":
    main()

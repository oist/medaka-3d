"""Reproducible source/parameter audit of saved C27 versus frozen clean P/N/F."""

# Report prose stays readable as complete strings; typographic signs are intentional.
# ruff: noqa: E501, RUF001

from __future__ import annotations

import ast
import base64
import csv
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/clean2d_video_comparison_v1"
EXP = "20251225_K_1_2_FCMC"
C27 = (
    ROOT
    / "work/custom2d_comparison_v1/inputs/c27_uncapped_tracklets_any"
    / EXP
    / "mouth-pectoral_oks_serial"
)
GT = ROOT / "work/generative_tracking_inputs/assets/groundtruth/swap_event_gt" / EXP
OKS_STDDEVS = (0.0125, 0.025, 0.05, 0.1)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def evidence(path, needle):
    text = path.read_text(encoding="utf-8")
    line = next(i for i, value in enumerate(text.splitlines(), 1) if needle in value)
    return dict(path=path.relative_to(ROOT).as_posix(), line=line, sha256=sha256(path))


def installation_records():
    distribution = importlib.metadata.distribution("sleap-nn")
    chosen = (
        "tracking/tracker.py",
        "tracking/candidates/local_queues.py",
        "tracking/utils.py",
        "evaluation.py",
    )
    records = []
    for entry in distribution.files:
        if any(str(entry).replace("\\", "/").endswith("sleap_nn/" + name) for name in chosen):
            path = Path(distribution.locate_file(entry))
            actual = base64.urlsafe_b64encode(bytes.fromhex(sha256(path))).decode().rstrip("=")
            records.append(
                dict(
                    path=str(path),
                    sha256=sha256(path),
                    distribution_record_matches=entry.hash.value == actual,
                )
            )
    return records


def installed_probes():
    import sleap_io as sio
    from sleap_nn.tracking.tracker import Tracker
    from sleap_nn.tracking.utils import hungarian_matching

    skeleton = sio.Skeleton(["mouth", "pectoral"])

    def batch(xs):
        return [
            sio.PredictedInstance.from_numpy(
                points_data=np.array([[x, 1.0], [x + 4.0, 6.0]]),
                skeleton=skeleton,
                point_scores=np.ones(2),
                score=1.0,
            )
            for x in xs
        ]

    def tracker(**kwargs):
        return Tracker.from_config(candidates_method="local_queues", max_tracks=None, **kwargs)

    t = tracker()
    t.track(batch([0.0, 100.0]), frame_idx=0)
    b = batch([500.0, 0.1, 100.1])
    result = t.track(b, frame_idx=1)
    rows = [next(i for i, instance in enumerate(b) if instance is r) for r in result]
    reorder = dict(
        returned_input_rows=rows,
        returned_labels=[r.track.name for r in result],
        correct_input_labels=[r.track.name for r in b],
    )
    assert rows == [1, 2, 0]
    t = tracker()
    t.track(batch([0.0]), frame_idx=0)
    result = t.track(batch([10000.0]), frame_idx=1000)
    zero = dict(
        label=result[0].track.name,
        score=float(result[0].tracking_score),
        queue_frame_indices=[x.frame_idx for x in t.candidate.tracker_queue[0]],
    )
    assert zero["score"] == 0.0 and zero["label"] == "track_0"
    invalid = hungarian_matching(np.array([[np.inf]]))
    t = tracker()
    partial = batch([0.0])
    xy = partial[0].numpy()
    xy[0] = np.nan
    partial = [
        sio.PredictedInstance.from_numpy(
            points_data=xy, skeleton=skeleton, point_scores=np.ones(2), score=1.0
        )
    ]
    partial_result = t.track(partial, frame_idx=0)
    empty = [
        sio.PredictedInstance.from_numpy(
            points_data=np.full((2, 2), np.nan),
            skeleton=skeleton,
            point_scores=np.ones(2),
            score=1.0,
        )
    ]
    with np.errstate(all="ignore"):
        empty_result = t.track(empty, frame_idx=1)
    strict_spawn = tracker(min_new_track_points=1)
    partial_for_spawn = sio.PredictedInstance.from_numpy(
        points_data=xy, skeleton=skeleton, point_scores=np.ones(2), score=1.0
    )
    strict_spawn_result = strict_spawn.track([partial_for_spawn], frame_idx=0)
    strict_match = tracker(min_match_points=1)
    strict_match.track(
        [
            sio.PredictedInstance.from_numpy(
                points_data=xy, skeleton=skeleton, point_scores=np.ones(2), score=1.0
            )
        ],
        frame_idx=0,
    )
    strict_match_result = strict_match.track(batch([0.0]), frame_idx=1)
    return dict(
        returned_order=reorder,
        zero_similarity_after_long_gap=zero,
        infinite_cost_assignment=[x.tolist() for x in invalid],
        one_visible_node_can_spawn=len(partial_result) == 1,
        all_missing_current_row_can_match_existing_track=bool(empty_result),
        all_missing_current_tracking_score=str(empty_result[0].tracking_score)
        if empty_result
        else None,
        default_min_match_points=t.min_match_points,
        default_min_new_track_points=t.candidate.min_new_track_points,
        default_oks_stddev=t.oks_stddev,
        default_use_flow=t.use_flow,
        strict_min_new_1_single_visible_assigned=any(
            x.track is not None for x in strict_spawn_result
        ),
        strict_min_match_1_filtered_history_still_assigned=bool(strict_match_result[0].track),
        strict_min_match_1_filtered_history_tracking_score=str(
            strict_match_result[0].tracking_score
        ),
        scope="Current installed package behavior, not proof of the historical runtime.",
    )


def focused_replay_audit():
    """Inspect exact source-row links and reconstruct cropped per-track OKS scores."""
    from sleap_nn.evaluation import compute_instance_area, compute_oks

    replay = OUT / "c27_replay"
    definitions = [
        ("known_30735", "side1_cam", 30823),
        ("known_31479", "side1_cam", 31489),
        ("cg20s", "top_cam", 75315),
        ("known_333619", "top_cam", 333619),
        ("known_105541", "top_cam", 105541),
    ]
    records, inputs, raw = [], {}, {}
    for case, view, anchor in definitions:
        raw_path = ROOT / "work/generative_tracking_inputs/work/c01_s16_predictions/bottomup"
        raw_path /= f"{EXP}__{view}.npz"
        if view not in raw:
            with np.load(raw_path, allow_pickle=False) as z:
                raw[view] = {k: z[k] for k in ("frame_idx", "points")}
            inputs[str(raw_path)] = sha256(raw_path)
        replay_path = replay / case / f"{view}.npz"
        report_path = replay_path.with_suffix(".json")
        inputs[str(replay_path)] = sha256(replay_path)
        inputs[str(report_path)] = sha256(report_path)
        with np.load(replay_path, allow_pickle=False) as z:
            arrays = dict(z)
        frames, rows = arrays["frame_idx"], arrays["global_rows"]
        points = raw[view]["points"][rows]
        assert np.array_equal(frames, raw[view]["frame_idx"][rows])
        links = {}
        for key in ("saved_full_serial_labels", "legacy_labels", "row_faithful_labels"):
            previous, edges = {}, []
            for row, (frame, label) in enumerate(zip(frames, arrays[key], strict=True)):
                if label < 0:
                    continue
                before = previous.get(int(label))
                if (
                    before is not None
                    and frames[before] + 1 == frame
                    and anchor - 8 <= frame <= anchor + 8
                ):
                    distance = float(np.linalg.norm(points[row, 1] - points[before, 1]))
                    if distance > 10:
                        edges.append(
                            dict(
                                from_frame=int(frames[before]),
                                to_frame=int(frame),
                                from_global_row=int(rows[before]),
                                to_global_row=int(rows[row]),
                                label=int(label),
                                pectoral_px=distance,
                                mouth_finite_at_endpoints=np.isfinite(points[[before, row], 0])
                                .all(axis=-1)
                                .tolist(),
                            )
                        )
                previous[int(label)] = row
            links[key] = edges
        frame_evidence = []
        for frame in (anchor - 1, anchor, anchor + 1):
            present = np.flatnonzero(frames == frame)
            labels = arrays["row_faithful_labels"]
            historical_ids = sorted(set(int(x) for x in labels[frames < frame] if x >= 0))
            history = [
                np.flatnonzero((frames < frame) & (labels == label))[-5:]
                for label in historical_ids
            ]
            sensitivity = []
            areas = compute_instance_area(points[present, :2].astype(float))
            for sigma in OKS_STDDEVS:
                scores = np.full((len(present), len(history)), np.nan)
                for j, row in enumerate(present):
                    for k, previous in enumerate(history):
                        values = [
                            float(
                                compute_oks(
                                    points[row, :2].astype(float),
                                    points[p, :2].astype(float),
                                    stddev=sigma,
                                )[0, 0]
                            )
                            for p in previous
                            if np.isfinite(points[p, :2]).all(axis=-1).any()
                        ]
                        if values:
                            scores[j, k] = np.nanmean(values)
                sensitivity.append(
                    dict(
                        stddev=sigma,
                        score_matrix=[
                            [float(x) if np.isfinite(x) else None for x in row] for row in scores
                        ],
                        normalization_per_current_instance=(
                            2 * (areas + np.spacing(1)) * (2 * sigma) ** 2
                        ).tolist(),
                        all_candidate_scores_zero_current_rows=np.flatnonzero(
                            (scores == 0).all(axis=1)
                        ).tolist(),
                    )
                )
            baseline_scores = next(x["score_matrix"] for x in sensitivity if x["stddev"] == 0.025)
            frame_evidence.append(
                dict(
                    frame=frame,
                    global_rows=rows[present].tolist(),
                    primary_node_finite=np.isfinite(points[present, :2]).all(axis=-1).tolist(),
                    labels={key: arrays[key][present].tolist() for key in links},
                    cropped_candidate_labels=historical_ids,
                    cropped_candidate_last5_global_rows=[rows[h].tolist() for h in history],
                    current_primary_bbox_area_px2=areas.tolist(),
                    current_installed_oks_score_matrix=baseline_scores,
                    frozen_history_oks_stddev_sensitivity=sensitivity,
                    scale_formula="bbox area of current M/P points, nanmin/nanmax ignoring missing points; one visible point gives area0",
                    normalization_formula="2*(area+np.spacing(1))*(2*stddev)^2; epsilon makes area0 denominator strictly positive, not zero",
                    missing_node_rule="Historical missing node -> squared distance inf; current missing node -> KS0 and excluded from visible-node divisor",
                    score_scope="Recomputed from cropped row-faithful history; not historical full-serial score matrix",
                )
            )
        reference = {
            (e["from_global_row"], e["to_global_row"]) for e in links["saved_full_serial_labels"]
        }
        faithful = {
            (e["from_global_row"], e["to_global_row"]) for e in links["row_faithful_labels"]
        }
        records.append(
            dict(
                case=case,
                view=view,
                anchor_frame=anchor,
                inspected_half_open=[anchor - 8, anchor + 9],
                pectoral_links_above_10px=links,
                saved_large_links_reproduced_by_faithful=len(reference & faithful),
                saved_large_links_total=len(reference),
                reordered_returns_within_12=[
                    x
                    for x in read(report_path)["reordered_returns"]
                    if abs(x["frame"] - anchor) <= 12
                ],
                frame_evidence=frame_evidence,
                interpretation=(
                    "All saved >10px links reproduced by current cropped association; no local output reorder."
                    if reference == faithful
                    else "Saved links differ from cropped association; no local reorder establishes serialization as cause. Full-history and historical-runtime effects remain confounded."
                ),
            )
        )
    return records, inputs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, original_run = read(C27 / "c27_summary.json"), read(C27 / "c27.provenance.json")
    protocol = read(ROOT / "work/clean_tracklets_v2/protocol.json")
    old_protocol = read(ROOT / "work/custom2d_comparison_v1/analysis_v3/protocol.json")
    config = protocol["tracking_config"]
    flow_file = ROOT / "labs/20260917_clean_tracklets/flow.py"
    tree = ast.parse(flow_file.read_text(encoding="utf-8"))
    flow_node = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PARAMETERS" for t in node.targets)
    )
    flow = {}
    for key, value in zip(flow_node.keys, flow_node.values, strict=True):
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult):
            resolved = ast.literal_eval(value.left) * ast.literal_eval(value.right)
        else:
            resolved = ast.literal_eval(value)
        flow[ast.literal_eval(key)] = resolved
    with (GT / "swap_events.csv").open(encoding="utf-8-sig", newline="") as f:
        notes = list(csv.DictReader(f))
    metadata = read(GT / "swap_events.json")
    anchors = [21232, 30735, 31479, 105541, 260434, 302762, 333619]
    clean_views = {
        21232: ["top_cam", "side2_cam"],
        30735: ["top_cam", "side1_cam"],
        31479: ["side1_cam", "side2_cam"],
        105541: [],
        260434: ["top_cam"],
        302762: ["side1_cam", "side2_cam"],
        333619: ["top_cam", "side1_cam", "side2_cam"],
    }
    scenes = []
    for anchor in anchors:
        note = next(r for r in notes if int(r["frame_start"]) == anchor)
        case = next(c for c in old_protocol["cases"] if c["name"] == f"known_{anchor}")
        scenes.append(
            dict(
                anchor=anchor,
                original_event_inclusive=[int(note["frame_start"]), int(note["frame_end"])],
                comparison_core_half_open=[case["start"], case["stop"]],
                context_half_open=[case["start"] - 332, case["stop"] + 332],
                clear_views_in_human_note=clean_views[anchor],
                original_note=note["annotator_note"],
                kind=note["kind"],
                row_number=notes.index(note) + 2,
                caution="105541 does not identify a view; 302762 rates side1 strongest. "
                "These are old rendered-chain visual judgements, not current 2D ID truth.",
            )
        )
    source_paths = dict(
        c27=ROOT / "tools/c27_uncapped_tracklets_any.py",
        c24=ROOT / "tools/c24_uncapped_tracklets.py",
        tracker=ROOT / ".venv/Lib/site-packages/sleap_nn/tracking/tracker.py",
        queues=ROOT / ".venv/Lib/site-packages/sleap_nn/tracking/candidates/local_queues.py",
        utils=ROOT / ".venv/Lib/site-packages/sleap_nn/tracking/utils.py",
        oks=ROOT / ".venv/Lib/site-packages/sleap_nn/evaluation.py",
        clean=ROOT / "labs/20260917_clean_tracklets/tracking.py",
        flow=flow_file,
    )
    citations = {
        "c27_serial": evidence(source_paths["c27"], "lo, hi = args.frames"),
        "c27_tracker_config": evidence(source_paths["c24"], "tracker = Tracker.from_config("),
        "c27_readout": evidence(source_paths["c24"], "for offset, instance in enumerate"),
        "local_queue": evidence(source_paths["queues"], "deque(maxlen=self.window_size)"),
        "spawn_support": evidence(
            source_paths["queues"], "num_visible_keypoints > self.min_new_track_points"
        ),
        "candidate_support": evidence(source_paths["tracker"], "> self.min_match_points"),
        "negative_similarity_cost": evidence(source_paths["tracker"], "cost_matrix = -scores"),
        "hungarian_invalid_cost": evidence(
            source_paths["utils"], "invalid = ~np.isfinite(cost_matrix)"
        ),
        "oks_normalization": evidence(
            source_paths["oks"], "scale = compute_instance_area(points_gt)"
        ),
        "clean_fixed_config": evidence(source_paths["clean"], "class TrackingConfig:"),
        "clean_gates": evidence(source_paths["clean"], "def _pair("),
        "clean_bidirectional": evidence(source_paths["clean"], "def build_tracklets("),
        "flow_fixed_config": evidence(source_paths["flow"], "PARAMETERS = {"),
    }
    axes = [
        dict(
            axis="scoring",
            c27="M/P OKS, mean over each track's last 5 observations; current default sigma=.025",
            clean="P: pectoral; N/F: mouth+pectoral weighted prediction error (1:2), tail veto only",
        ),
        dict(
            axis="history",
            c27="Full serial [0,348600); per-ID 5 observations, no native-age expiry",
            clean="4 observations inside adjacent-frame fragments; missing frame resets history",
        ),
        dict(
            axis="assignment",
            c27="Forward Hungarian, uncapped tracks, no explicit dummy refusal/margin",
            clean="One-to-one with unmatched options, competitor margin, mutual forward/backward agreement",
        ),
        dict(
            axis="support",
            c27="min_match_points=0 filters PAST candidates by count>0; new-track default count>0",
            clean="P needs finite pectoral; N/F need finite mouth AND pectoral",
        ),
        dict(
            axis="motion_gate_px_top_side",
            c27="No explicit displacement/prediction gate in configured base Tracker",
            clean="native step <=36/30; prediction error <=14.4/12; heading veto at85deg under motion gates",
        ),
        dict(
            axis="abstention_px_top_side",
            c27="No configured next-best margin or proximity cut",
            clean="assignment margin3/2.5; proximity9.6/8",
        ),
        dict(
            axis="node_cohesion_px_top_side",
            c27="Tail excluded; no independent cohesion veto",
            clean="M-P relative displacement14.4/12; length change14.4/12; tail relative displacement30/25; axis70deg",
        ),
        dict(
            axis="image_flow",
            c27="use_flow=False in this configuration",
            clean="F only; half-resolution LK, 1.5native-px roundtrip, patch L1<=20; endpoint error7.2/6px; weight.5",
        ),
        dict(
            axis="count",
            c27="No3-fish cap; saved whole recording unique IDs8/7/7 with long reuse gaps",
            clean="Every detection retained, disconnected/singleton components get new nonnegative labels",
        ),
    ]
    test_axes = [
        "First compare frozen saved C27 to P/N/F on identical detections and native pixels; no tuning claim.",
        "Cropped C27 with fixed +/-332 history: report edge agreement and whole-core label-invariant partition; never call it exact full-serial replay.",
        "Separate legacy returned-list readout from row-faithful readout; changing readout does not change internal Tracker state.",
        "Executed fixed C27 sensitivity: windows1/5/10 on cg20s/21232/302762/333619, all3views, same cropped history and row-faithful readout. Compare against cropped5, not saved full serial; no best setting selected.",
        "OKS sigma.0125/.025/.05/.1 was evaluated only on frozen cropped histories at five anchors, not by full tracking replay. Full-replay sigma sensitivity and mean vs max reduction remain unexecuted.",
        "Candidate native-age expiry or fixed_window vs local_queues is an algorithm/history-policy change, not merely changing the number5.",
        "min_match_points/new_track_points0 vs1 is strict count>threshold; inf-cost assignment behavior means this alone is not reliable abstention.",
        "P-only OKS has zero bbox area; do not claim fair P-versus-MP feature comparison without defining a nonzero shared scale or different score.",
        "Root's preregistered clean F axes: motion bounds x2; node cohesion bounds x2; margin+proximity x.5; flow gate x2; combined; stricter opposite. Include unchanged controls, coverage and per-view continuous runs.",
        "No full-recording replay launched here: saved historical C27 used4361s wall across3 cameras. Local replay is for boundary/software diagnostics only.",
    ]
    with provenance("c27_vs_clean2d_parameter_audit", OUT) as rec:
        for name, path in source_paths.items():
            rec.input(name, path, sha256=sha256(path))
        for name, path in [
            ("code", Path(__file__)),
            ("saved_summary", C27 / "c27_summary.json"),
            ("saved_provenance", C27 / "c27.provenance.json"),
            ("human_csv", GT / "swap_events.csv"),
            ("human_metadata", GT / "swap_events.json"),
            ("clean_protocol", ROOT / "work/clean_tracklets_v2/protocol.json"),
            ("case_protocol", ROOT / "work/custom2d_comparison_v1/analysis_v3/protocol.json"),
        ]:
            rec.input(name, path, sha256=sha256(path))
        result = dict(
            status="complete",
            saved_c27_effective_config=summary["effective_tracker_config"],
            saved_c27_requested_parameters={
                k: v
                for k, v in original_run["params"].items()
                if k not in ["census", "effective_tracker_config"]
            },
            saved_c27_environment=original_run["environment"],
            clean_config=config,
            flow_config=flow,
            parameter_mapping=axes,
            human_clear_scenes=scenes,
            human_annotation_chain=metadata["chain"],
            human_slot_semantics=metadata["slot_semantics"],
            original_human_note_scope="No exact phrase 'one track' found in these registered notes; clean/easy is a visual judgement, not certified component identity.",
            current_package_versions={
                p: importlib.metadata.version(p) for p in ["sleap-nn", "sleap-io", "numpy", "scipy"]
            },
            current_installation_record_checks=installation_records(),
            installed_synthetic_probes=installed_probes(),
            evidence=citations,
            replay_and_test_axes=test_axes,
            limitations=[
                "Saved C27 provenance records a dirty git tree and no installed package-file hashes; current package behavior is not certified historical runtime.",
                "Long serial history changes candidate sets; equal named parameters do not guarantee cropped replay equivalence.",
                "Historical notes target c27->c28->c33->pre-v4 c30 rendered slots, not current C34 or new P/N/F ground truth.",
                "Later written analysis disputes the literal clean-side-track interpretation at31479; preserve both user observation and measured counterevidence.",
            ],
            verdict="Differences involve both thresholds and architecture. Present data cannot attribute differences solely to tuning or establish identity superiority.",
        )
        replay_path = OUT / "c27_replay/summary.json"
        if replay_path.exists():
            replay_result = read(replay_path)
            records = replay_result["records"]
            result["cropped_replay_summary"] = {
                "path": str(replay_path.relative_to(ROOT)),
                "sha256": sha256(replay_path),
                "records": len(records),
                "legacy_exact_core_partition_cases": sum(
                    x["legacy_vs_saved"]["exact_core_partition_up_to_one_global_mapping"]
                    for x in records
                ),
                "faithful_exact_core_partition_cases": sum(
                    x["row_faithful_vs_saved"]["exact_core_partition_up_to_one_global_mapping"]
                    for x in records
                ),
                "legacy_adjacent_edge_jaccard_range": [
                    min(x["legacy_vs_saved"]["adjacent_edge_jaccard"] for x in records),
                    max(x["legacy_vs_saved"]["adjacent_edge_jaccard"] for x in records),
                ],
                "readout_different_rows_core": sum(
                    x["readout_different_rows_core"] for x in records
                ),
                "readout_different_rows_context": sum(
                    x["readout_different_rows_context"] for x in records
                ),
                "readout_affected_case_views": [
                    dict(
                        case=x["case"],
                        view=x["view"],
                        core_different_rows=x["readout_different_rows_core"],
                        core_reordered_frames=[
                            e
                            for e in x["reordered_returns"]
                            if x["core_half_open"][0] <= e["frame"] < x["core_half_open"][1]
                        ],
                    )
                    for x in records
                    if x["readout_different_rows_core"]
                ],
                "readout_count_scope": "Current cropped replay only; not certified counts of errors in historical full C27",
            }
            rec.input("cropped_replay", replay_path, sha256=sha256(replay_path))
            score_protocol = OUT / "oks_scoring_protocol.json"
            score_protocol.write_text(
                json.dumps(
                    dict(
                        stddevs=OKS_STDDEVS,
                        history="Frozen cropped window5 row-faithful per-ID last5 observations; no tracker replay",
                        anchor_frames=[30823, 31489, 75315, 333619, 105541],
                        score_frames="anchor-1, anchor, anchor+1",
                        selection="All settings reported; no best chosen; scores only, not alternative identities",
                        code_sha256=sha256(Path(__file__)),
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            rec.input("oks_scoring_protocol", score_protocol, sha256=sha256(score_protocol))
            focused, inputs = focused_replay_audit()
            result["focused_saved_jump_audit"] = focused
            for i, (path, digest) in enumerate(inputs.items()):
                rec.input(f"focused_input_{i}", Path(path), sha256=digest)
        sensitivity_path = OUT / "c27_sensitivity/summary.json"
        if sensitivity_path.exists():
            result["c27_window_sensitivity"] = dict(
                path=str(sensitivity_path.relative_to(ROOT)),
                sha256=sha256(sensitivity_path),
                records=len(read(sensitivity_path)["records"]),
                scope="Window1/5/10, four cases x three views, cropped row-faithful comparison",
            )
            rec.input("c27_sensitivity", sensitivity_path, sha256=sha256(sensitivity_path))
        output = OUT / "parameter_audit.json"
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        rows = [
            "# Settings and history audit: C27 versus clean P/N/F",
            "",
            "The difference is not only thresholds. C27 is forward assignment that keeps long-term IDs; P/N/F are forward/backward agreement between adjacent frames that cuts ambiguous links. From this comparison it cannot be concluded either that 'the quality of parameter tuning alone is the cause' or that 'the new method has higher ID accuracy'.",
            "",
            "|Item|Saved C27 / current implementation check|clean P/N/F|",
            "|---|---|---|",
        ]
        rows += [f"|{a['axis']}|{a['c27']}|{a['clean']}|" for a in axes]
        rows += [
            "",
            "## Key facts about reproduction",
            "",
            "- Saved C27 is whole-recording serial over [0,348600), mouth/pectoral, OKS, 5-observation history, max_tracks=None. window=5 is not expiry after 5 native frames.",
            "- In a synthetic example with the currently installed implementation, the same ID was assigned with OKS=0 even to a 10000px move 1000 frames apart. NaN/inf costs also become large finite values before the Hungarian step and are not refused after assignment.",
            "- min_match_points=0 is the condition that a past candidate has >0 non-NaN points, and min_new_track_points=0 likewise starts from a detection with >0. It is neither a similarity threshold nor an ambiguity test.",
            "- It was reproduced synthetically that feeding [new, existing0, existing1] to 2 existing tracks returns the order [1,2,0]. The C24/C27 wrapper treats the returned order as the original row order, so the local replay saves both legacy and row-faithful. This is not an assertion that every row of the old saved C27 is wrong.",
            "- The currently installed files match the distribution RECORD, but the old environment of saved C27 left no package hashes and has git_dirty=true. Exact reproduction of the whole recording is not certified.",
            "- In the cropped rerun of 9 intervals × 3 views, the legacy readout and the readout mapped back to original rows differed on 24 rows, 7 frames and 6 case-views within core. Over the full context it is 34 rows. This is a difference demonstrated in the current rerun and does not imply the same number of errors in the old saved whole recording.",
            "- Agreement of the whole core ID partition with saved full (allowing only a single ID-name substitution) was legacy 7/27 and original-row readout 10/27. Even with a high adjacent-edge agreement rate, this cannot be called an exact reproduction of the saved whole recording.",
            "- side1 f30823 235.15px, f31489 276.23px and top f105541±8 max 37.93px reproduced the same inter-detection links in both readouts. None has a reordering of returned rows within centre ±12; they are jumps reproduced by the current tracker's association.",
            "- top f75315 91/93px and f333619/620 max 233.01px were not reproduced by either cropped readout. There is no reordering near these either; the old full history versus local history and old-runtime differences are not separated. The old saved jump at 333619 is not asserted to be caused by serialization.",
            "- For M/P pairs with mouth missing, the current-side bbox area becomes 0, and OKS for a finite move can collapse to 0. The measured frames and the recomputed scores of each cropped candidate are kept in the JSON. This is not 'mouth missing in every detection'.",
            "- This missingness is not a case where scale becomes NaN. Through nanmin/nanmax the area of 1 visible point becomes 0, and the normalization denominator with epsilon added is a positive 1.110223×10^-18 at sigma=.025. A missing current point has KS=0 and is also excluded from the visible-point mean denominator.",
            "- Comparing sigma=.0125/.025/.05/.1 on the fixed cropped histories alone, the mouth-missing rows at f30823, 31489 and 333619 stayed at score=0 for every candidate. For the M/P-finite row0 at 333619, very small positive values returned at sigma=.05/.1. This is a score diagnostic, not an ID improvement from re-running tracking.",
            "",
            "## The past observation that '2D is clear'",
            "",
            "The annotations are on the old C27→C28→C33→C30 video of 2026-09-03. S0/S1/S2 are that video's slot names, not biological IDs or ground-truth names for the current method. The comments from the original CSV are kept below.",
            "",
            "|Original annotation interval (both ends inclusive)|Comparison core here [start,stop)|Views called clear|Original comment|",
            "|---|---|---|---|",
        ]
        rows += [
            f"|{s['original_event_inclusive']}|{s['comparison_core_half_open']}|{', '.join(s['clear_views_in_human_note']) or 'view not specified'}|{s['original_note']}|"
            for s in scenes
        ]
        rows += [
            "",
            "For 31479, a later analysis records counterevidence that there was a 276px tracklet jump in side1 as well. Whether the detection looks clear and whether the saved temporal ID is continuously correct are checked separately on video.",
            "",
            "## Separating the tests",
            "",
        ] + ["- " + x for x in test_axes]
        rows += ["", "## Evidence", ""]
        rows += [
            f"- {key}: [{value['path']}:{value['line']}](../../{value['path']}:{value['line']})"
            for key, value in citations.items()
        ]
        rows += [
            "- Original comments: [swap_events.csv](../../work/generative_tracking_inputs/assets/groundtruth/swap_event_gt/20251225_K_1_2_FCMC/swap_events.csv)",
            "- Later analysis: [doubt-channel](../../docs/decisions/2026-09-01-doubt-channel.md:342)",
            "",
            "The saved source data, past results and frozen P/N/F were not modified. Details and input hashes are recorded in parameter_audit.json and the provenance.",
        ]
        markdown = OUT / "parameter_audit.md"
        markdown.write_text("\n".join(rows) + "\n", encoding="utf-8")
        rec.output("json", output, sha256=sha256(output))
        rec.output("markdown", markdown, sha256=sha256(markdown))
    print(
        json.dumps(
            {"json": str(output), "markdown": str(markdown), "scenes": len(scenes)},
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

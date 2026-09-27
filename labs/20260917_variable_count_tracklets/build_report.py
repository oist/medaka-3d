"""Build an inspectable report from saved same-row 2D results and media."""

# ruff: noqa: E501, RUF001

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np
from render import METHODS, ROOT, VIDEO_NAMES, VIEWS, load_case

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def safe_json(value):
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    return value


def dumps(value):
    return json.dumps(
        safe_json(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")


def json_items(path):
    """Stream large JSON arrays without materializing hundreds of MB of candidates."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        buffer, eof = "", False
        while True:
            buffer = buffer.lstrip(" \r\n\t,[")
            if buffer.startswith("]"):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    if buffer.strip():
                        raise ValueError(f"Incomplete JSON array {path}") from None
                    return
                chunk = stream.read(1024 * 1024)
                eof = not chunk
                buffer += chunk
                continue
            yield value
            buffer = buffer[end:]


def compact_candidate(candidate):
    keys = (
        "from_global_row",
        "to_global_row",
        "prediction_error_px",
        "step_px",
        "score_px",
        "origin_margin_px",
        "target_margin_px",
        "required_margin_px",
        "origin_rank",
        "target_rank",
        "heading_change_deg",
        "heading_review",
        "uncertainty_px",
        "age_frames",
        "auxiliary_support",
        "observed_support_nodes",
        "node_cohesion_flags",
        "reason",
    )
    value = {key: candidate[key] for key in keys if key in candidate}
    competitors = candidate.get("competing_endpoints", [])
    value["competing_endpoint_count"] = len(competitors)
    value["competing_endpoints_first_4"] = competitors[:4]
    return value


def selected_diagnostics(directory, case, view, start, stop):
    """Keep inspectable core evidence; full JSON/CSV tables remain linked separately."""
    by_row = {}
    links, events = [], []
    for edge in json_items(directory / case / f"edges_P1_{view}.json"):
        frame = edge.get("to_frame", -1)
        if start <= frame < stop:
            row = edge.get("to_global_row", edge.get("to_row"))
            edge = {
                key: edge[key]
                for key in (
                    "from_global_row",
                    "to_global_row",
                    "from_frame",
                    "to_frame",
                    "reason",
                    "accepted",
                )
                if key in edge
            } | {key: compact_candidate(edge.get(key, {})) for key in ("forward", "backward")}
            by_row.setdefault(str(row), {})["accepted_edge"] = edge
            links.append(edge)
    for event in json_items(directory / case / f"events_P1_{view}.json"):
        frame = event.get("frame", event.get("native_frame", -1))
        if start <= frame < stop:
            row = event.get("global_row", event.get("row", event.get("to_global_row")))
            kind = event.get("kind", "")
            candidates = event.get("candidates", [])
            adjacent_candidates = [c for c in candidates if c.get("age_frames") == 1]
            adjacent_errors = [
                c["prediction_error_px"]
                for c in adjacent_candidates
                if c.get("prediction_error_px") is not None
            ]
            events.append(
                dict(
                    frame=frame,
                    kind=kind,
                    reason=event.get("reason"),
                    candidate_count=len(candidates),
                    adjacent_error=min(adjacent_errors) if adjacent_errors else None,
                    old_endpoint_count=sum(c.get("age_frames", 0) > 1 for c in candidates),
                )
            )
            if kind == "open_endpoint":
                continue
            item = {
                key: event[key]
                for key in (
                    "kind",
                    "direction",
                    "global_row",
                    "frame",
                    "reason",
                    "forward_from_global_row",
                    "backward_to_global_row",
                    "from_global_row",
                    "to_global_row",
                )
                if key in event
            }
            item["candidate_count"] = len(candidates)
            item["candidates_first_2_in_saved_rank_order"] = [
                compact_candidate(c) for c in candidates[:2]
            ]
            item["adjacent_candidates_first_2"] = [
                compact_candidate(c) for c in adjacent_candidates[:2]
            ]
            for direction in ("forward", "backward"):
                if isinstance(event.get(direction), dict):
                    item[direction] = compact_candidate(event[direction])
            value = by_row.setdefault(str(row), {})
            value["event_count"] = value.get("event_count", 0) + 1
            if len(value.setdefault("events_first_4", [])) < 4:
                value["events_first_4"].append(item)
    return by_row, links, events


def series_data(data, case, edges, events):
    start, stop = case["start"], case["stop"]
    frames = np.arange(start, stop)
    reference = data["arms"]["C27"]
    detected = np.diff(np.searchsorted(reference["frame_idx"], np.arange(start, stop + 1)))
    missing = np.zeros((len(frames), 3), int)
    connections = {}
    for method, arrays in data["arms"].items():
        core = (arrays["frame_idx"] >= start) & (arrays["frame_idx"] < stop)
        eligible = core & (arrays["frame_idx"] > start)
        connections[method] = np.bincount(
            arrays["frame_idx"][eligible & arrays["clean_edge"]] - start, minlength=stop - start
        )
    for i, frame in enumerate(frames):
        lo, hi = np.searchsorted(reference["frame_idx"], [frame, frame + 1])
        missing[i] = (~np.isfinite(reference["points"][lo:hi]).all(-1)).sum(0)
    prediction = np.full(len(frames), np.nan)
    step = np.full(len(frames), np.nan)
    disagreement = np.zeros(len(frames), int)
    refusal = np.zeros(len(frames), int)
    refused_adjacent_error = np.full(len(frames), np.nan)
    old_endpoints = np.zeros(len(frames), int)
    for edge in edges:
        frame = edge.get("to_frame", -1)
        if start <= frame < stop:
            i = frame - start
            for direction in ("forward", "backward"):
                values = edge.get(direction, {})
                for key, target in (("prediction_error_px", prediction), ("step_px", step)):
                    value = values.get(key)
                    if value is not None and np.isfinite(value):
                        target[i] = (
                            max(float(value), target[i]) if np.isfinite(target[i]) else float(value)
                        )
    for event in events:
        frame = event.get("frame", -1)
        if start <= frame < stop:
            if event.get("kind") == "directional_link_refused":
                refusal[frame - start] += 1
                value = event.get("adjacent_error")
                i = frame - start
                if value is not None:
                    refused_adjacent_error[i] = (
                        max(value, refused_adjacent_error[i])
                        if np.isfinite(refused_adjacent_error[i])
                        else value
                    )
                old_endpoints[i] = max(old_endpoints[i], event.get("old_endpoint_count", 0))
            if "disagree" in str(event.get("reason", "")) or "disagree" in str(
                event.get("kind", "")
            ):
                disagreement[frame - start] += 1
    return dict(
        frames=frames,
        detections=detected,
        missing=missing,
        connections=connections,
        prediction=prediction,
        step=step,
        disagreement=disagreement,
        refusal=refusal,
        refused_adjacent_error=refused_adjacent_error,
        old_endpoints=old_endpoints,
    )


def plot_svg(path, series, title):
    """Native-frame plots retain all samples; missing prediction evidence stays unplotted."""
    width, left, right, panel, top = 1120, 65, 25, 155, 60
    frames = series["frames"]
    start, stop = int(frames[0]), int(frames[-1])
    colors = ["#63d8b3", "#f8c56a", "#b596fa", "#6db8ff", "#ff909b", "#d3d7df"]
    groups = [
        ("Detection rows / native frame", [("all detections", series["detections"], colors[0])]),
        (
            "Missing observed nodes / frame",
            [
                (name, series["missing"][:, i], colors[i + 1])
                for i, name in enumerate(("M missing", "P missing", "T missing"))
            ],
        ),
        (
            "Accepted adjacent edges / frame (not identity accuracy)",
            [
                (name, series["connections"][name], color)
                for name, color in zip(METHODS, colors, strict=True)
            ],
        ),
        (
            "P1 accepted-edge max distance / native frame (px)",
            [
                ("prediction error", series["prediction"], colors[1]),
                ("observed step", series["step"], colors[3]),
            ],
        ),
        (
            "P1 refusal candidates / frame: max of best adjacent errors (px); no acceptance implied",
            [("rejected adjacent prediction error", series["refused_adjacent_error"], colors[4])],
        ),
        (
            "P1 refusal candidate sets: maximum number of older endpoint hypotheses / native frame",
            [("age > 1 candidate endpoints", series["old_endpoints"], colors[2])],
        ),
        (
            "P1 diagnostic events / native frame (directions may both count)",
            [
                ("refusal events", series["refusal"], colors[4]),
                ("forward/reverse disagreement", series["disagreement"], colors[2]),
            ],
        ),
    ]
    result = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{top + panel * len(groups) + 20}" viewBox="0 0 {width} {top + panel * len(groups) + 20}">',
        '<rect width="100%" height="100%" fill="#111c2a"/>',
        f'<text x="20" y="29" fill="#eef4fc" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
    ]
    plot_width = width - left - right
    for group_index, (label, entries) in enumerate(groups):
        origin = top + group_index * panel
        finite = [float(v) for _, arr, _ in entries for v in arr if np.isfinite(v)]
        maximum = max([1, *finite])
        result.append(
            f'<text x="{left}" y="{origin}" fill="#eef4fc" font-family="sans-serif" font-size="13">{html.escape(label)}</text>'
        )
        legend = left
        for name, _, color in entries:
            result.append(
                f'<text x="{legend}" y="{origin + 17}" fill="{color}" font-family="sans-serif" font-size="10">{html.escape(name)}</text>'
            )
            legend += len(name) * 6 + 20
        y0, height = origin + 30, 87
        for tick in range(4):
            y = y0 + tick * height / 3
            value = maximum * (1 - tick / 3)
            result.append(
                f'<path d="M{left},{y:.1f} H{width - right}" stroke="#2a3849"/><text x="5" y="{y + 4:.1f}" fill="#96a6b9" font-family="sans-serif" font-size="10">{value:.1f}</text>'
            )
        for _, arr, color in entries:
            segments, segment = [], []
            for frame, value in zip(frames, arr, strict=True):
                if not np.isfinite(value):
                    if segment:
                        segments.append(segment)
                    segment = []
                    continue
                x = left + (frame - start) / max(1, stop - start) * plot_width
                y = y0 + height * (1 - float(value) / maximum)
                segment.append(f"{x:.1f},{y:.1f}")
            if segment:
                segments.append(segment)
            for segment in segments:
                if len(segment) == 1:
                    x, y = segment[0].split(",")
                    result.append(f'<circle cx="{x}" cy="{y}" r="1.4" fill="{color}"/>')
                else:
                    result.append(
                        f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="1.15" opacity="0.86"/>'
                    )
        for tick in range(5):
            frame = round(start + (stop - start) * tick / 4)
            x = left + tick * plot_width / 4
            result.append(
                f'<text x="{x}" y="{y0 + height + 17}" text-anchor="middle" fill="#96a6b9" font-family="sans-serif" font-size="10">f{frame} / {frame / 166:.3f}s</text>'
            )
    result.append("</svg>")
    path.write_text("\n".join(result), encoding="utf-8")


def build_view_payload(directory, case, view, destination):
    data = load_case(directory, case["name"], view)
    reference = data["arms"]["C27"]
    core = (reference["frame_idx"] >= case["start"]) & (reference["frame_idx"] < case["stop"])
    rows = np.flatnonzero(core)
    diagnostics, edges, events = selected_diagnostics(
        directory, case["name"], view, case["start"], case["stop"]
    )
    payload = dict(
        case=case["name"],
        view=view,
        start=case["start"],
        stop=case["stop"],
        frame=reference["frame_idx"][rows],
        global_row=reference["global_rows"][rows],
        points=reference["points"][rows],
        score=reference.get("score", np.full(len(core), np.nan))[rows],
        methods={
            method: dict(
                labels=arrays["labels"][rows],
                adjacent=arrays["clean_edge"][rows],
                previous_global=np.array(
                    [
                        arrays["global_rows"][p] if p >= 0 else -1
                        for p in arrays["previous_row"][rows]
                    ]
                ),
            )
            for method, arrays in data["arms"].items()
        },
        diagnostics=diagnostics,
        source_hashes=data["hashes"],
    )
    key = case["name"] + "__" + view
    script = destination / (key + ".js")
    script.write_text(
        f"window.TRACKING_DATA=window.TRACKING_DATA||{{}};window.TRACKING_DATA[{dumps(key)}]={dumps(payload)};",
        encoding="utf-8",
    )
    svg = destination / (key + ".svg")
    plot_svg(
        svg,
        series_data(data, case, edges, events),
        f"{case['name']} / {view} / [{case['start']}, {case['stop']})",
    )
    return dict(
        script=script.name,
        plot=svg.name,
        invariants=data["invariants"],
        script_sha256=sha256(script),
        plot_sha256=sha256(svg),
    )


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Variable-count 2D tracklet comparison | 2026-09-17</title>
<style>
:root{color-scheme:dark;--bg:#09121e;--card:#111e2e;--muted:#a2b4c9;--line:#2b4058;--accent:#65dfbb}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf5ff;font:15px/1.75 system-ui,"Yu Gothic",sans-serif}main{max-width:1460px;margin:auto;padding:30px}h1{font-size:31px;line-height:1.35;margin:14px 0}h2{font-size:22px;margin:8px 0 16px}h3{font-size:17px}p{max-width:1150px}a{color:#8bc5ff}code,pre{font-family:Consolas,monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:480px;overflow:auto}button,select,input{font:inherit;color:inherit;background:#1c2d42;border:1px solid #49637f;border-radius:5px;padding:7px 10px}button{cursor:pointer}button:hover{border-color:var(--accent)}label{display:inline-flex;gap:8px;align-items:center}.eyebrow{color:var(--accent);font-size:13px;letter-spacing:2px}.muted,small{color:var(--muted)}.notice{border-left:4px solid #efbc62;background:#2a251e;padding:14px 18px}.section{background:var(--card);padding:24px;margin:24px 0;border:1px solid var(--line);border-radius:10px}.controls{display:flex;gap:13px;flex-wrap:wrap;align-items:center;margin:14px 0}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.kpi{padding:17px;background:#13273a;border-radius:8px}.kpi b{font-size:25px;display:block;color:var(--accent)}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}th{color:#a5c0de;background:#16273a;position:sticky;top:0}td.num{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow:auto;max-height:560px}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:20px}.methods{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}canvas{width:100%;background:#050a10;border:1px solid var(--line)}video{width:100%;max-height:760px;background:#000}#native-source{width:220px;height:145px;object-fit:contain}.stills{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.stills img{width:100%;border:1px solid var(--line)}.graph{width:100%;min-height:300px}details{margin:15px 0}summary{cursor:pointer;color:#9dcaff}.pill{display:inline-block;padding:3px 9px;background:#223c49;border-radius:15px;font-size:12px;margin:4px}#frame-slider{flex:1;min-width:250px}.evidence{color:#d6e9ff}#state{color:var(--accent);min-height:24px}.anchor{scroll-margin-top:20px}.source-list{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0}figcaption{font-size:12px;color:var(--muted)}@media(max-width:900px){main{padding:15px}.kpis{grid-template-columns:1fr 1fr}.twocol,.methods{grid-template-columns:1fr}.section{padding:15px}.stills{grid-template-columns:1fr}h1{font-size:25px}}
</style></head><body><main>
<div class="eyebrow">INDEPENDENT 2D OBSERVATIONS · STAGES A—C</div>
<h1>Comparing scenes where the detection count changes, from the same observations</h1>
<p>Experiment <code>20251225_K_1_2_FCMC</code> · 166 fps · 3 views · one group of the same 3 fish. Compares saved C27, current-environment S1, fixed-scale S2, frozen P0, P1, and P1 + auxiliary nodes. Every input row and dropout is retained; reflection versus physical fish remains undetermined.</p>
<div class="notice"><b>Result: the fixed setting P1 in this run does not meet the acceptance conditions.</b> __RESULT_SUMMARY__ The records confirm excessive holding due to endpoint competition as the main cause. Correct-continuation and wrong-link rates are not computed (2D ground-truth denominator for all individuals: 0). This result is not treated as a successful improvement, and 3D reconnection is left to stage D.</div>
<nav class="controls"><a href="#viewer">Frame check</a><a href="#movies">Comparison videos and before/after images</a><a href="#numbers">Totals and strata</a><a href="#interpretation">Per-scene evaluation</a><a href="#configuration">All settings and sources</a></nav>
<div class="kpis" id="kpis"></div>
<section class="section anchor" id="viewer"><h2>All observations and link evidence per view</h2>
<p class="muted">Shows any 3 of the 6 conditions on the same native frame. M = mouth, P = pectoral fin, T = tail. Points and solid lines are observed values; local IDs are identifiers only within a method and view. The same colour is not evidence of the same individual across methods. Predicted points are not drawn onto the observed image. The evidence panel is summarised to the top 2 candidates + 2 adjacent candidates / the first 4 events per row, alongside the original total count and a link to the full JSON.</p>
<div class="controls"><label>Case <select id="case-select"></select></label><label>View <select id="view-select"></select></label><button id="load-source">Load source video</button><span id="state"></span></div>
<div class="controls"><button id="previous-frame">-1 frame</button><input id="frame-slider" type="range" step="1"><button id="next-frame">+1 frame</button><label>native <input id="frame-number" type="number" style="width:110px"></label><button id="apply-frame">Go to frame</button></div>
<div class="controls"><span id="frame-info"></span><label>Playback speed <select id="speed"><option value="0.18072289156626506">30 native fps / 5.53× slower</option><option value="1">Real time</option></select></label><button id="native-play">Play / pause interval</button></div>
<div class="methods" id="panels"></div>
<details><summary>Source video player and how sync works</summary><video id="native-source" muted playsinline preload="none" controls></video><p class="muted">Seeks the source recording directly to native frame / 166 s. Frame selection while stopped uses a time inside the frame. During playback the observations matching the browser mediaTime are shown, and screen updates depend on the browser draw rate. The exported videos below were checked against OpenCV native frame numbers, with every frame re-decoded.</p></details>
<div class="scroll"><table><thead><tr><th>C01 global_row</th><th>Observed M/P/T</th><th>score</th><th id="label-head">Local ID / adjacent link</th><th>Show P1 evidence</th></tr></thead><tbody id="observation-rows"></tbody></table></div>
<details open><summary>P1 accept/hold reasons, candidate distances, forward/backward predictions, node support</summary><pre id="evidence" class="evidence">Select a row to show the saved evidence.</pre></details>
<img id="timeline" class="graph" alt="Time series of native frame and time, detection count, M/P/T dropouts, adjacent links, candidate prediction error, forward/backward disagreement">
<div id="downloads" class="source-list"></div></section>
<section class="section anchor" id="movies"><h2>20 s overview video and zoomed slow-motion videos of known scenes</h2>
<p class="muted">The 3 fixed columns are C27 / P0 / P1. The same source frames and all detections are shown. The main 20 s is decimated to 30 fps for 20 s of real time; events play the 166 frames at centre ±83 native frames at 30 fps (5.53 s). For events the top row is the whole frame and the bottom row a zoom around the two closest P points at the centre time. The zoom does not automatically identify the annotation target or the error location.</p>
<div class="controls"><label>Video <select id="asset-select"></select></label><button id="asset-to-frame">Open centre in frame check</button></div>
<video id="comparison-video" controls preload="metadata" playsinline></video><p id="asset-caption" class="muted"></p><div class="stills" id="stills"></div>
<details><summary>All videos and verified frame counts</summary><div class="scroll"><table id="media-table"></table></div></details></section>
<section class="section anchor" id="numbers"><h2>Totals, stratified by detection count, proximity and dropout</h2><p class="muted">The link-rate denominator is the observation rows in core excluding its first frame. Link counts are adjacent-frame links with both ends inside core only. Hold rate = share of observations in that denominator with no adjacent link; it is not a miss rate. Fragment count = core rows - adjacent links. Same label across a gap is a separate column and includes cases where a core arrival row reaches back to a previous row in the context. Proximity strata are a diagnostic based on observed inter-P distance; a missing P classified as "separated" does not mean separation was confirmed. refusal_reasons in the JSON are the various event reasons over the whole context (including endpoints and boundaries), not a unique count of refused links.</p>
<div class="controls"><label>Scope <select id="metric-case"></select></label><label>View <select id="metric-view"><option value="all">All views combined</option></select></label></div><div class="scroll"><table id="metric-table"></table></div>
<h3>Source detection counts and node dropouts (common to all methods)</h3><div class="scroll"><table id="count-table"></table></div>
<details><summary>All cases × all views × all conditions (the sensitivity comparison also lists every setting)</summary><div class="scroll"><table id="all-metrics"></table></div></details>
<details><summary>Current case and view, stratified by detection count / proximity / node dropout (P0 and P1)</summary><div class="scroll"><table id="strata-table"></table></div></details>
<p><a href="metrics.csv">All totals CSV</a> · <a href="strata.csv">All strata CSV</a> · <a href="metrics.json">Machine-readable totals JSON</a> · <a href="review/manifest.json">All media, SHA and decode records</a></p></section>
<section class="section anchor" id="interpretation"><h2>Open judgements per scene, and what this run reached</h2><div id="findings"></div><div class="scroll"><table id="case-findings"></table></div><details open><summary>Local findings and remaining 3D correspondence for 13 anchors × 3 views</summary><div class="scroll"><table id="event-findings"></table></div><p><a href="case_findings.json">All local metrics and findings JSON</a> · <a href="scientific_scope_audit.json">Audit of diagnostic evidence and metric scope JSON</a></p></details>
<details><summary>Existing human review (not reused as 2D ground truth here)</summary><div id="human-review"></div></details>
<p>Additional annotation is needed to distinguish which fragments at entry and exit are the same individual, whether a detection is a physical fish or a reflection, and whether an event is a node mix-up or an individual-ID exchange. Before, centre and after images and slow-motion videos were prepared for each anchor. The known no-swap scenes f76791 and f323819 and the node-exchange / no-ID-swap scene f77415 are treated as controls against miscorrection. Old C30/C34 slot names were not converted into 2D ground-truth labels.</p></section>
<section class="section anchor" id="configuration"><h2>Comparison conditions, all settings and sources</h2><div id="conditions"></div><p>Each condition receives the same C01 rows and 332 frames before and after, and is evaluated on the central core. S1/S2 use past history only; P0/P1 use forward and backward context. Saved C27 is an earlier whole-recording result, not a reproduction of the same run history and environment as the short-interval S1.</p>
<details><summary>Fixed protocol and all parameters</summary><pre id="protocol"></pre></details><details><summary>SHA of code, settings and inputs, and output verification</summary><pre id="audit"></pre></details>
<p class="source-list"><a href="protocol.json">protocol.json</a><a href="numerical_manifest.json">Snapshot at the first numerical tally (saved record)</a><a href="numerical_manifest_verified.json">Final verified numerical manifest (canonical copy)</a><a href="verified_numerical_manifest.provenance.json">Numerical manifest verification provenance</a><a href="report_manifest.json">report manifest</a><a href="../../docs/decisions/2026-09-17-variable-count-2d-tracklet-plan.md">Implementation plan</a><a href="../../docs/reviews/2026-09-16_multiview_identity_review.json">Existing human review</a><a href="../../labs/20260917_variable_count_tracklets/RESULT.md">RESULT.md</a></p>
<p class="muted">Figures, videos and stills were generated from this run's saved results. The mapping between estimation IDs and display colours was not carried over to other methods. Whole-recording re-tracking, detector retraining and 3D integration are out of scope for this run.</p></section>
</main><script>const REPORT=__PAYLOAD__;</script><script>
const $=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const num=n=>Number.isFinite(n)?n.toLocaleString('en-US'):'undetermined';const pct=n=>Number.isFinite(n)?(100*n).toFixed(2)+'%':'undetermined';let active=null,frame=0,selectedRow=null,loadedKey='',sourceReady=false;const choices=['C27','P0','P1'];
function table(id,heads,rows){$(id).innerHTML='<thead><tr>'+heads.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody>'}
function options(el,values){el.innerHTML=values.map(v=>'<option value="'+esc(v[0])+'">'+esc(v[1])+'</option>').join('')}
options($('case-select'),REPORT.cases.map(c=>[c.name,c.name+' ['+c.start+','+c.stop+')']));options($('view-select'),REPORT.views.map(v=>[v,v]));options($('metric-case'),[['all','All cases'],...REPORT.cases.map(c=>[c.name,c.name])]);REPORT.views.forEach(v=>$('metric-view').add(new Option(v,v)));$('metric-case').value='cg20s';
$('panels').innerHTML=choices.map((m,i)=>'<div><label>Condition <select id="method-'+i+'">'+REPORT.methods.map(x=>'<option '+(x===m?'selected':'')+'>'+x+'</option>').join('')+'</select></label><canvas id="canvas-'+i+'" width="600" height="600"></canvas></div>').join('');choices.forEach((_,i)=>$('method-'+i).onchange=()=>draw());
async function selectView(target){$('native-source').pause();sourceReady=false;const c=REPORT.cases.find(x=>x.name===$('case-select').value),v=$('view-select').value,key=c.name+'__'+v;loadedKey=key;$('state').textContent='Loading saved observations and evidence';if(!window.TRACKING_DATA?.[key])await new Promise((resolve,reject)=>{const s=document.createElement('script');s.src='report_data/'+REPORT.payloads[key].script;s.onload=resolve;s.onerror=reject;document.head.appendChild(s)});if(key!==loadedKey)return;active=window.TRACKING_DATA[key];frame=target??c.centers?.[0]??c.start;frame=Math.min(c.stop-1,Math.max(c.start,frame));$('frame-slider').min=$('frame-number').min=c.start;$('frame-slider').max=$('frame-number').max=c.stop-1;$('timeline').src='report_data/'+REPORT.payloads[key].plot;$('native-source').removeAttribute('src');$('native-source').load();$('state').textContent='Points and labels cover every native frame. Use the source-video button to show the same frame';$('downloads').innerHTML=REPORT.methods.map(m=>'<a href="'+c.name+'/tracking_'+m+'_'+v+'.npz">'+m+' NPZ</a>').join(' ')+' <a href="'+c.name+'/edges_P1_'+v+'.json">P1 edges</a> <a href="'+c.name+'/events_P1_'+v+'.json">P1 hold reasons</a> <a href="'+c.name+'/reconnections_P1_'+v+'.json">Reconnection candidates</a>';setFrame(frame,false)}
function color(id,method){if(id<0)return'#c5d0dd';let h=(id*94+REPORT.methods.indexOf(method)*62+REPORT.views.indexOf(active.view)*38+34)%360;return'hsl('+h+' 78% 73%)'}
function setFrame(value,seek=true){if(!active||value===''||!Number.isFinite(Number(value)))return;frame=Math.min(active.stop-1,Math.max(active.start,Math.round(Number(value))));$('frame-slider').value=$('frame-number').value=frame;const count=active.frame.reduce((n,f)=>n+(f===frame),0);$('frame-info').textContent='f'+frame+' / '+(frame/166).toFixed(6)+' s · '+count+' detections'+(seek&&sourceReady?' · seeking video':'');if(seek&&sourceReady){$('native-source').pause();$('state').textContent='Syncing source video to f'+frame;$('native-source').currentTime=(frame+.25)/166}else draw()}
function draw(){if(!active)return;const rows=[];active.frame.forEach((f,i)=>{if(f===frame)rows.push(i)});const methods=choices.map((_,i)=>$('method-'+i).value);$('frame-info').textContent='f'+frame+' / '+(frame/166).toFixed(6)+' s · '+rows.length+' detections · physical fish/reflection undetermined';methods.forEach((m,k)=>{const canvas=$('canvas-'+k),ctx=canvas.getContext('2d');ctx.fillStyle='#050a10';ctx.fillRect(0,0,600,600);if(sourceReady&&$('native-source').readyState>=2)ctx.drawImage($('native-source'),0,0,600,600);else{ctx.fillStyle='#8298af';ctx.font='17px sans-serif';ctx.fillText('Load the source video to show it behind the observed points',18,570)};const a=active.methods[m];rows.forEach((r,slot)=>{const xy=active.points[r],c=color(a.labels[r],m);ctx.strokeStyle=c;ctx.fillStyle=c;ctx.lineWidth=1.4;[[0,1],[1,2]].forEach(([i,j])=>{if(xy[i].every(Number.isFinite)&&xy[j].every(Number.isFinite)){ctx.beginPath();ctx.moveTo(xy[i][0]/2,xy[i][1]/2);ctx.lineTo(xy[j][0]/2,xy[j][1]/2);ctx.stroke()}});xy.forEach((p,i)=>{if(p.every(Number.isFinite)){ctx.beginPath();ctx.arc(p[0]/2,p[1]/2,i===1?4:3,0,Math.PI*2);ctx.stroke();ctx.font='10px sans-serif';ctx.fillText('MPT'[i],p[0]/2+5,p[1]/2-3)}});let p=xy.find(p=>p.every(Number.isFinite));if(p){ctx.font='13px monospace';ctx.fillText('L'+a.labels[r]+(a.adjacent[r]?'':' / new'),Math.max(4,Math.min(460,p[0]/2+9)),Math.max(18,Math.min(550,p[1]/2-12+slot*5)))}});ctx.fillStyle='#fff';ctx.font='15px sans-serif';ctx.fillText(m+' · f'+frame+' · '+rows.length+' rows',12,22)});$('label-head').textContent=methods.join(' / ')+' : local ID (adjacent link)';$('observation-rows').innerHTML=rows.map(r=>'<tr><td>'+active.global_row[r]+'</td><td>'+active.points[r].map((p,i)=>'MPT'[i]+':'+(p.every(Number.isFinite)?'○':'missing')).join(' ')+'</td><td>'+Number(active.score[r]).toFixed(4)+'</td><td>'+methods.map(m=>'L'+active.methods[m].labels[r]+' ('+(active.methods[m].adjacent[r]?'accepted':'new/held')+')').join(' / ')+'</td><td><button data-row="'+r+'">Show evidence</button></td></tr>').join('');document.querySelectorAll('[data-row]').forEach(b=>b.onclick=()=>showEvidence(+b.dataset.row));if(rows.length)showEvidence(rows.find(r=>active.global_row[r]===selectedRow)??rows[0]);else $('evidence').textContent='This native frame has 0 source detection rows. No observation points are added.'}
function showEvidence(r){selectedRow=active.global_row[r];const d=active.diagnostics[String(selectedRow)]??{note:'No P1 event/edge is saved for this core row. Initial row or no link candidate. See also the full-record links.'};$('evidence').textContent=JSON.stringify({native_frame:frame,global_row:selectedRow,observed_nodes:active.points[r],instance_score:active.score[r],reflection_status:'unknown',...d},null,2)}
$('case-select').onchange=()=>selectView();$('view-select').onchange=()=>selectView();$('frame-slider').oninput=e=>setFrame(e.target.value);$('frame-number').onchange=e=>setFrame(e.target.value);$('frame-number').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();setFrame(e.target.value)}};$('apply-frame').onclick=()=>setFrame($('frame-number').value);$('previous-frame').onclick=()=>setFrame(frame-1);$('next-frame').onclick=()=>setFrame(frame+1);$('load-source').onclick=()=>{const video=$('native-source');video.pause();sourceReady=false;setFrame($('frame-number').value,false);$('state').textContent='Loading source video';video.src=REPORT.sources[active.view];video.load();video.onloadedmetadata=()=>{sourceReady=true;video.currentTime=(frame+.25)/166;video.playbackRate=+$('speed').value;$('state').textContent='Source video synced to native frame'};video.onerror=()=>{$('state').textContent='Cannot open the source video. The exported comparison videos and stills are below.'}};$('native-source').onseeked=()=>{$('state').textContent='Source video synced to native f'+frame;draw()};$('speed').onchange=()=>{$('native-source').playbackRate=+$('speed').value};$('native-play').onclick=()=>{const v=$('native-source');if(!sourceReady){$('load-source').click();return}v.paused?v.play():v.pause()};
const nativeVideo=$('native-source');function onVideoFrame(now,meta){if(sourceReady&&!nativeVideo.paused&&active){const f=Math.floor(meta.mediaTime*166+1e-4);if(f>=active.stop){nativeVideo.pause();setFrame(active.stop-1)}else{frame=Math.max(active.start,f);$('frame-slider').value=$('frame-number').value=frame;draw()}}nativeVideo.requestVideoFrameCallback(onVideoFrame)}if(nativeVideo.requestVideoFrameCallback)nativeVideo.requestVideoFrameCallback(onVideoFrame);else nativeVideo.ontimeupdate=()=>{if(!nativeVideo.paused)setFrame(Math.floor(nativeVideo.currentTime*166),false)};
options($('asset-select'),REPORT.assets.map((a,i)=>[i,a.view+' · '+(a.center_frame===null?'main 20 s / whole':'f'+a.center_frame+' / zoomed, slow')+' · '+a.case]));function selectAsset(){const a=REPORT.assets[+$('asset-select').value];if(!a)return;$('comparison-video').src='review/'+a.video;$('comparison-video').poster='review/'+a.center;$('asset-caption').textContent=a.output_frames+' frames / '+a.output_duration_seconds.toFixed(3)+' s · native ['+a.native_frame_range_half_open.join(',')+') · all '+a.full_decode.decoded_frames+' frames decode-verified';$('stills').innerHTML=['before','center','after'].map((k,i)=>'<figure><a target="_blank" href="review/'+a[k]+'"><img loading="lazy" src="review/'+a[k]+'" alt="'+a.view+' '+k+' f'+a.still_native_frames[i]+'"></a><figcaption>'+['before','centre','after'][i]+' · f'+a.still_native_frames[i]+' · click for full size</figcaption></figure>').join('')} $('asset-select').onchange=selectAsset;$('asset-to-frame').onclick=()=>{const a=REPORT.assets[+$('asset-select').value];$('case-select').value=a.case;$('view-select').value=a.view;selectView(a.center_frame??a.still_native_frames[1]);$('viewer').scrollIntoView({behavior:'smooth'})};table('media-table',['View / native','Video','Frames','Seconds','Verification'],REPORT.assets.map(a=>[esc(a.view+' / '+(a.center_frame??'20s')),'<a href="review/'+a.video+'">'+esc(a.video)+'</a>',num(a.output_frames),a.output_duration_seconds.toFixed(3),a.full_decode.verified?'all frames OK':'not confirmed']));
const heads=['Condition','core observation rows','Retention rate','Fragments','Adjacent links','Denominator','Link rate','Hold rate','Across gap'];function metricCells(r){return[esc(r.method),num(r.core_rows),pct(r.retention_rate),num(r.contiguous_runs),num(r.adjacent_connections),num(r.eligible_observations),pct(r.connection_rate),pct(r.abstention_rate),num(r.gap_connections)]}function updateMetrics(){const cs=$('metric-case').value,v=$('metric-view').value,rs=REPORT.records.filter(r=>(cs==='all'||r.case===cs)&&(v==='all'||r.view===v));const sums=Object.keys(REPORT.protocol.conditions).map(m=>{const a=rs.filter(r=>r.method===m),o={method:m};['core_rows','retained_rows','contiguous_runs','adjacent_connections','eligible_observations','unconnected_observations','gap_connections'].forEach(k=>o[k]=a.reduce((n,r)=>n+(r[k]??0),0));o.retention_rate=o.retained_rows/o.core_rows;o.connection_rate=o.adjacent_connections/o.eligible_observations;o.abstention_rate=o.unconnected_observations/o.eligible_observations;return o});table('metric-table',heads,sums.map(metricCells));table('count-table',['Case','View','core frames','0 detections','1 detection','2 detections','3 detections','4+ detections','M missing rows','P missing rows','T missing rows'],rs.filter(r=>r.method==='C27').map(r=>[esc(r.case),esc(r.view),num(r.core_frames),...['0','1','2','3','4+'].map(k=>num(r.detection_count_frames[k])),num(r.missing_m),num(r.missing_p),num(r.missing_t)]));const ss=rs.filter(r=>['P0','P1'].includes(r.method)).flatMap(r=>(r.strata??[]).filter(s=>s.eligible_observations>0).map(s=>[esc(r.case),esc(r.view),r.method,s.detection_count,s.proximity,s.any_node_missing?'yes':'no',num(s.eligible_observations),num(s.adjacent_connections),pct(s.connection_rate)]));table('strata-table',['Case','View','Condition','Detections','Proximity','M/T etc. missing','Denominator','Links','Rate'],ss)}$('metric-case').onchange=updateMetrics;$('metric-view').onchange=updateMetrics;table('all-metrics',['Case','View',...heads],REPORT.records.map(r=>[esc(r.case),esc(r.view),...metricCells(r)]));
const main=REPORT.records.filter(r=>r.case==='cg20s'&&r.method==='P1'),rows=main.reduce((n,r)=>n+r.core_rows,0);$('kpis').innerHTML=[['3 views × 20 s',num(rows)+' observation rows compared unchanged'],[REPORT.cases.length+' cases','11 known + validation at another time, fixed in advance'],[REPORT.assets.length+' comparison videos',REPORT.assets.length*3+' before, centre and after images'],['2D ground-truth denominator 0','correct-continuation rate / wrong-link rate: undetermined']].map(x=>'<div class="kpi"><b>'+x[0]+'</b><span>'+x[1]+'</span></div>').join('');
$('protocol').textContent=JSON.stringify(REPORT.protocol,null,2);$('audit').textContent=JSON.stringify(REPORT.audit,null,2);$('conditions').innerHTML='<table>'+Object.entries(REPORT.protocol.conditions).map(([k,v])=>'<tr><th>'+esc(k)+'</th><td>'+esc(v)+'</td></tr>').join('')+'</table>';$('human-review').innerHTML=REPORT.human.events.map(e=>'<p><b>f'+e.frame+'</b> '+esc(e.verbatim_note)+' <span class="muted">['+esc(e.kind)+', view='+esc(e.view??'unspecified')+']</span></p>').join('');
$('findings').innerHTML=REPORT.findings.map(x=>'<p>'+esc(x)+'</p>').join('');table('case-findings',['Case','core range','P0 → P1 fragments (all views)','Evaluation caveats / open judgements'],REPORT.cases.map(c=>{const get=m=>REPORT.records.filter(r=>r.case===c.name&&r.method===m).reduce((n,r)=>n+r.contiguous_runs,0);return[esc(c.name),'['+c.start+', '+c.stop+')',num(get('P0'))+' → '+num(get('P1')),esc(REPORT.case_notes[c.name]??'Compares full row retention and structural metrics. Physical-fish correspondence and ID correctness await additional annotation.')]}));
table('event-findings',['Case / view / native','Local finding','Remaining 3D correspondence','Additional review item'],(REPORT.event_findings.records??[]).map(r=>[esc(r.case+' / '+r.view+' / f'+r.center),esc(r.finding),esc(r.remaining_3d),esc(r.review_question)]));selectAsset();updateMetrics();selectView().catch(e=>{$('state').textContent='Load error: '+e.message;console.error(e)});
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "work/variable_count_tracklets_v1")
    parser.add_argument(
        "--reuse-payloads",
        action="store_true",
        help="Verify saved payload hashes and rebuild HTML only",
    )
    args = parser.parse_args()
    base = args.data.resolve()
    destination = base / "report_data"
    destination.mkdir(exist_ok=True)
    protocol_path = base / "protocol.json"
    protocol = read(protocol_path)
    metrics_path = base / "metrics.json"
    metrics = read(metrics_path)
    media_path = base / "review/manifest.json"
    media = read(media_path)
    if not metrics or not metrics.get("complete") or not media or media["status"] != "complete":
        raise ValueError("Complete numeric results and every verified media asset are required")
    cases = protocol["cases"]
    records = metrics["records"]
    main_p0 = sum(
        r["contiguous_runs"] for r in records if r["case"] == "cg20s" and r["method"] == "P0"
    )
    main_p1 = sum(
        r["contiguous_runs"] for r in records if r["case"] == "cg20s" and r["method"] == "P1"
    )
    main_aux = sum(
        r["contiguous_runs"] for r in records if r["case"] == "cg20s" and r["method"] == "P1aux"
    )
    result_summary = f"While every observation is retained, fragments in the main 20 s increased from {main_p0:,} for P0 to {main_p1:,} for P1, and are {main_aux:,} even with auxiliary nodes."
    findings = [
        f"Contiguous fragments in the main 20 s, summed over 3 views: P0={main_p0:,}, P1={main_p1:,}. This compares structural fragmentation; it is not a count confirmed against same-individual ground truth.",
        "S1 and S2 share the same fix to returned-row correspondence and compare only the OKS scale. S2's top=14400 / side=10000 px² was fixed before the results were checked.",
        "P1 commits only adjacent links that meet its internal conditions, and keeps vanished endpoints and post-reappearance reconnection candidates in a separate table. M/T dropouts are not equated with individual-ID switches.",
        "Links across gaps in saved C27 and S1/S2 are tallied separately from P0/P1 fragments made only of adjacent links.",
        "The 11 known cases are development and regression checks. The additional interval at another time was selected from metadata before seeing results, and is a validation interval of the same 3 fish run after the settings were fixed. It is not a generalisation to an independent group.",
        "In the diagnostic records, at every hold the old predicted endpoint and the new fragment endpoint keep competing along the same path. Because uncertainty is subtracted from the cost, there are cases where the old hypothesis is favoured. At side1 f76304 the other current detections are 109.11 px away, yet against the adjacent candidate's prediction error of 1.653 px, the endpoint from 2 frames earlier scores 2.580-1.5=1.080 px, so the candidate margin of 0.573 px falls below the required margin of 2.662 px. At top f75498, 17 candidates derive from the same frozen P0 label. P0 labels are not ground truth either, but these are internal diagnostics showing hold chains caused by duplicated endpoint hypotheses.",
        "In the local 17 frames of the no-swap control f76791/top, fragments worsened from P0=3 to P1=35 (32 extra cuts, P1aux=5). f21232/side2, which the old note calls clear, also went 3→19, and f31479/side1 went 5→21. The no-swap control f323819/side1 went 6→16 (P1aux=12). These are increases in fragmentation that can be confirmed without new identity ground truth.",
        "Conversely, the local 17 frames of f30735/side2 are 1 fragment under all 6 methods, and f302762/side1 is 3 fragments under all methods. At f333619/top, against saved C27's maximum adjacent P move of 233 px, S1/S2/P0/P1 share 7.78 px and 4 fragments. The reduction in large jumps is not called an improvement specific to P1.",
    ]
    notes = {
        "cg20s": "f76235 / f76930 / f76988 are swaps in the existing 3D review. f76791 is no-swap; f77415 is node exchange / no-ID-swap. Per-view 2D entry/exit correspondence is not confirmed.",
        "known_21232": "The old note calls top/side2 clear. 2D fragments are separated from old 3D slot switches; correctness here awaits new annotation.",
        "known_30735": "side2 changes from 1 detection to 2. A 3D assignment change on the same 2D fragment is separated from a 2D link switch.",
        "known_31479": "The 4-5 detections in side1 are retained. The surplus is not asserted to be reflections. The cause of the old 3D large jump is not tied directly to 2D correctness.",
        "known_105541": "The old note calls 2D clear. Whether P1's continuation here is the same individual needs checking with before/after videos.",
        "known_260434": "2 detections in side1 with an old fusion note, and 4 detections in side2. There are no per-row labels for the cause of occlusion or surplus.",
        "known_302762": "The old note calls side1 the clearest, with node contamination in top. M/P/T support is evaluated separately from individual-ID exchange.",
        "known_323819": 'The later human review says 「スワップない、パーフェクト」 (English: "no swap, perfect"). Even with side1 going 3→2→3→1 detections, new fragments are not confused with ID miscorrections.',
        "known_333619": "The old note says 2D is clear, yet there is a transient swap in 3D. The 3D problem is not deemed solved on the basis of this 2D judgement alone.",
        "control_50000": "A regression control already examined. It is unannotated and is not assumed to be an error-free ground-truth control.",
        "holdout20s": "A 20 s interval already examined. It is a regression check at another time on the same 3 fish, not called a new unseen test.",
        "validation20s_00332": "Before seeing results, overlap of the before/after context with known intervals and annotation metadata was excluded, and the first 20 s on the prescribed grid was selected. Validation at another time on the same 3 fish, run after the settings were fixed. It is not a guarantee of never having been seen before, nor validation on an independent group.",
    }
    sources = {
        v: os.path.relpath(
            ROOT / "work/three_body/source_cache/medaka_202512" / protocol["experiment"] / v / name,
            base,
        ).replace("\\", "/")
        for v, name in zip(VIEWS, VIDEO_NAMES, strict=True)
    }
    review_path = ROOT / "docs/reviews/2026-09-16_multiview_identity_review.json"
    previous_manifest = read(base / "report_manifest.json") if args.reuse_payloads else None
    if args.reuse_payloads and (
        not previous_manifest
        or previous_manifest["numerical_manifest_sha256"]
        != sha256(base / "numerical_manifest.json")
    ):
        raise ValueError("Cannot reuse payloads after numerical inputs changed")
    with provenance("variable_count_html_report", base) as record:
        for name, path in (
            ("report_builder", Path(__file__)),
            ("renderer", Path(__file__).with_name("render.py")),
            ("protocol", protocol_path),
            ("metrics", metrics_path),
            ("media_manifest", media_path),
            ("human_review", review_path),
            ("numeric_source_hashes", base / "numerical_manifest.json"),
            ("verified_numeric_source_hashes", base / "numerical_manifest_verified.json"),
            ("anchor_findings", base / "case_findings.json"),
            ("scientific_scope_audit", base / "scientific_scope_audit.json"),
            ("validation_selection", base / "selection_audit.json"),
        ):
            record.input(name, path, sha256=sha256(path))
        payloads = {}
        for case in cases:
            for view in VIEWS:
                key = case["name"] + "__" + view
                payloads[key] = (
                    previous_manifest["payloads"][key]
                    if previous_manifest
                    else build_view_payload(base, case, view, destination)
                )
                if previous_manifest:
                    for kind in ("script", "plot"):
                        if (
                            sha256(destination / payloads[key][kind])
                            != payloads[key][kind + "_sha256"]
                        ):
                            raise ValueError("Saved native diagnostics payload changed")
                print("Report data and graph " + key, flush=True)
                for kind in ("script", "plot"):
                    record.output(
                        key + "_" + kind,
                        destination / payloads[key][kind],
                        sha256=payloads[key][kind + "_sha256"],
                    )
        audit = dict(
            report_code_sha256=sha256(Path(__file__)),
            renderer_code_sha256=sha256(Path(__file__).with_name("render.py")),
            protocol_sha256=sha256(protocol_path),
            metrics_sha256=sha256(metrics_path),
            media_manifest_sha256=sha256(media_path),
            numerical_manifest_sha256=sha256(base / "numerical_manifest.json"),
            verified_numerical_manifest_sha256=sha256(base / "numerical_manifest_verified.json"),
            case_findings_sha256=sha256(base / "case_findings.json"),
            scientific_scope_audit_sha256=sha256(base / "scientific_scope_audit.json"),
            all_videos_fully_decoded=media["all_videos_fully_decoded"],
            all_method_view_payloads=len(payloads),
            media_count=len(media["assets"]),
            still_count=3 * len(media["assets"]),
            sources=sources,
        )
        payload = dict(
            cases=cases,
            views=VIEWS,
            methods=METHODS,
            records=records,
            protocol=protocol,
            assets=media["assets"],
            payloads=payloads,
            sources=sources,
            human=read(review_path),
            findings=findings,
            case_notes=notes,
            audit=audit,
            event_findings=read(base / "case_findings.json", {"records": []}),
        )
        target = base / "report.html"
        target.write_text(
            TEMPLATE.replace("__PAYLOAD__", dumps(payload)).replace(
                "__RESULT_SUMMARY__", result_summary
            ),
            encoding="utf-8",
        )
        manifest = dict(
            status="complete",
            report="report.html",
            report_sha256=sha256(target),
            **audit,
            payloads=payloads,
            identity_accuracy_claim=False,
            correct_continuation_rate=None,
            incorrect_connection_rate=None,
            identity_truth_denominator=0,
            media_temporal_sampling="20-second overview at 30 fps; every native anchor frame at 30 fps slow motion",
        )
        (base / "report_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        record.output("html", target, sha256=sha256(target))
        record.output(
            "manifest", base / "report_manifest.json", sha256=sha256(base / "report_manifest.json")
        )
    print(target, flush=True)


if __name__ == "__main__":
    main()

"""Build an inspectable Japanese report from saved same-row 2D results and media."""

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
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>可変検出数 2D トラックレット比較 | 2026-09-17</title>
<style>
:root{color-scheme:dark;--bg:#09121e;--card:#111e2e;--muted:#a2b4c9;--line:#2b4058;--accent:#65dfbb}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf5ff;font:15px/1.75 system-ui,"Yu Gothic",sans-serif}main{max-width:1460px;margin:auto;padding:30px}h1{font-size:31px;line-height:1.35;margin:14px 0}h2{font-size:22px;margin:8px 0 16px}h3{font-size:17px}p{max-width:1150px}a{color:#8bc5ff}code,pre{font-family:Consolas,monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:480px;overflow:auto}button,select,input{font:inherit;color:inherit;background:#1c2d42;border:1px solid #49637f;border-radius:5px;padding:7px 10px}button{cursor:pointer}button:hover{border-color:var(--accent)}label{display:inline-flex;gap:8px;align-items:center}.eyebrow{color:var(--accent);font-size:13px;letter-spacing:2px}.muted,small{color:var(--muted)}.notice{border-left:4px solid #efbc62;background:#2a251e;padding:14px 18px}.section{background:var(--card);padding:24px;margin:24px 0;border:1px solid var(--line);border-radius:10px}.controls{display:flex;gap:13px;flex-wrap:wrap;align-items:center;margin:14px 0}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.kpi{padding:17px;background:#13273a;border-radius:8px}.kpi b{font-size:25px;display:block;color:var(--accent)}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}th{color:#a5c0de;background:#16273a;position:sticky;top:0}td.num{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow:auto;max-height:560px}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:20px}.methods{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}canvas{width:100%;background:#050a10;border:1px solid var(--line)}video{width:100%;max-height:760px;background:#000}#native-source{width:220px;height:145px;object-fit:contain}.stills{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.stills img{width:100%;border:1px solid var(--line)}.graph{width:100%;min-height:300px}details{margin:15px 0}summary{cursor:pointer;color:#9dcaff}.pill{display:inline-block;padding:3px 9px;background:#223c49;border-radius:15px;font-size:12px;margin:4px}#frame-slider{flex:1;min-width:250px}.evidence{color:#d6e9ff}#state{color:var(--accent);min-height:24px}.anchor{scroll-margin-top:20px}.source-list{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0}figcaption{font-size:12px;color:var(--muted)}@media(max-width:900px){main{padding:15px}.kpis{grid-template-columns:1fr 1fr}.twocol,.methods{grid-template-columns:1fr}.section{padding:15px}.stills{grid-template-columns:1fr}h1{font-size:25px}}
</style></head><body><main>
<div class="eyebrow">INDEPENDENT 2D OBSERVATIONS · STAGES A—C</div>
<h1>検出数が変わる場面を、同じ観測から比較する</h1>
<p>実験 <code>20251225_K_1_2_FCMC</code> · 166 fps · 3 視点 · 同じ 3 匹の 1 群。保存 C27、現環境 S1、固定尺度 S2、凍結 P0、P1、P1＋補助ノードを比較します。すべての入力行・欠損を保持し、反射／実魚は未判定のままです。</p>
<div class="notice"><b>結果：今回の固定設定 P1 は採用条件を満たしません。</b> __RESULT_SUMMARY__ 端点競合による過剰な保留が主因として記録から確認されます。正しい継続率・誤接続率は未算出（全個体の2D正解分母 0）。今回の結果を改良成功とは扱わず、3D 再接続は工程 D に残します。</div>
<nav class="controls"><a href="#viewer">フレーム照合</a><a href="#movies">比較動画・前後画像</a><a href="#numbers">集計・層別</a><a href="#interpretation">場面別の評価</a><a href="#configuration">全設定・出典</a></nav>
<div class="kpis" id="kpis"></div>
<section class="section anchor" id="viewer"><h2>各ビューの全観測と接続根拠</h2>
<p class="muted">6 条件から任意の 3 条件を同じ native frame で表示。M＝口、P＝胸びれ、T＝尾。点と実線は観測値、ローカル ID は方式・ビュー内だけの識別子です。同色は方式をまたぐ同一個体の証拠ではありません。予測点は観測画像へ描き足しません。根拠パネルは候補上位2件＋隣接候補2件／行の先頭4イベントに要約し、元の全件数と完全なJSONリンクを併記します。</p>
<div class="controls"><label>ケース <select id="case-select"></select></label><label>ビュー <select id="view-select"></select></label><button id="load-source">元映像を読み込む</button><span id="state"></span></div>
<div class="controls"><button id="previous-frame">−1 frame</button><input id="frame-slider" type="range" step="1"><button id="next-frame">＋1 frame</button><label>native <input id="frame-number" type="number" style="width:110px"></label><button id="apply-frame">指定フレームへ移動</button></div>
<div class="controls"><span id="frame-info"></span><label>再生速度 <select id="speed"><option value="0.18072289156626506">30 native fps / 5.53 倍低速</option><option value="1">実時間</option></select></label><button id="native-play">区間を再生 / 一時停止</button></div>
<div class="methods" id="panels"></div>
<details><summary>元映像プレイヤーと同期の説明</summary><video id="native-source" muted playsinline preload="none" controls></video><p class="muted">native frame / 166 秒へ元録画を直接シークします。停止後のフレーム指定はフレーム内の時刻を使用。再生時はブラウザーの mediaTime に対応する観測を表示し、画面更新はブラウザーの描画速度に依存します。下の書き出し動画は OpenCV の native frame 番号を確認し、全フレームを再デコード済みです。</p></details>
<div class="scroll"><table><thead><tr><th>C01 global_row</th><th>観測 M/P/T</th><th>score</th><th id="label-head">ローカル ID / 隣接接続</th><th>P1 の根拠を表示</th></tr></thead><tbody id="observation-rows"></tbody></table></div>
<details open><summary>P1 の採用・保留理由、候補距離、順逆予測、ノード支持</summary><pre id="evidence" class="evidence">行を選ぶと保存した根拠を表示します。</pre></details>
<img id="timeline" class="graph" alt="native frame・時間、検出数、M/P/T 欠損、隣接接続、候補予測誤差、順逆不一致の時系列">
<div id="downloads" class="source-list"></div></section>
<section class="section anchor" id="movies"><h2>20 秒の全体動画と、既知場面の拡大・低速動画</h2>
<p class="muted">固定 3 列は C27 / P0 / P1。同じ元フレーム・全検出を表示します。主 20 秒は 30 fps に間引いて実時間 20 秒、イベントは中心 ±83 native frame の 166 枚を 30 fps で再生（5.53 秒）。イベント上段は全体、下段は中心時点で最も近い P 2 点付近の拡大です。拡大は注釈対象や誤り位置を自動認定するものではありません。</p>
<div class="controls"><label>動画 <select id="asset-select"></select></label><button id="asset-to-frame">中心をフレーム照合で開く</button></div>
<video id="comparison-video" controls preload="metadata" playsinline></video><p id="asset-caption" class="muted"></p><div class="stills" id="stills"></div>
<details><summary>全動画の一覧・検証済みフレーム数</summary><div class="scroll"><table id="media-table"></table></div></details></section>
<section class="section anchor" id="numbers"><h2>集計と検出数・近接・欠損による層別</h2><p class="muted">接続率の分母は core の先頭フレームを除く観測行。接続数は両端が core 内にある隣接フレーム接続のみです。保留率＝その分母に対する隣接接続のない観測の割合で、見逃し率ではありません。断片数＝core 行数−隣接接続数。欠測越し同一ラベルは別列で、core の到着行から文脈内の前行に遡る場合も含みます。近接層別は観測 P 間距離による診断で、P 欠損は「separated」に分類されても離隔が確認された意味ではありません。JSON の refusal_reasons は全 context の各種 event reason（端点・境界を含む）で、唯一の拒否接続数ではありません。</p>
<div class="controls"><label>集計対象 <select id="metric-case"></select></label><label>ビュー <select id="metric-view"><option value="all">全ビュー合計</option></select></label></div><div class="scroll"><table id="metric-table"></table></div>
<h3>元の検出数とノード欠損（方式共通）</h3><div class="scroll"><table id="count-table"></table></div>
<details><summary>全ケース × 全ビュー × 全条件（感度比較も全設定を掲載）</summary><div class="scroll"><table id="all-metrics"></table></div></details>
<details><summary>現在のケース・ビューの検出数 / 近接 / ノード欠損による層別（P0 と P1）</summary><div class="scroll"><table id="strata-table"></table></div></details>
<p><a href="metrics.csv">全集計 CSV</a> · <a href="strata.csv">全層別 CSV</a> · <a href="metrics.json">機械可読集計 JSON</a> · <a href="review/manifest.json">全メディア・SHA・デコード記録</a></p></section>
<section class="section anchor" id="interpretation"><h2>場面別に残る判断と、今回の到達点</h2><div id="findings"></div><div class="scroll"><table id="case-findings"></table></div><details open><summary>13 アンカー × 3 ビューの局所所見・残る3D対応</summary><div class="scroll"><table id="event-findings"></table></div><p><a href="case_findings.json">全局所指標と所見 JSON</a> · <a href="scientific_scope_audit.json">診断根拠と指標範囲の監査 JSON</a></p></details>
<details><summary>既存の人手レビュー（今回の 2D 正解へ転用しない）</summary><div id="human-review"></div></details>
<p>入口・出口でどの断片が同一個体か、実魚／反射のどちらか、ノードの取り違えか個体 ID 交換かを区別する追加注釈が必要です。各アンカーの前・中心・後画像と低速動画を準備しました。既知の no-swap 場面 f76791・f323819 と node-exchange / no-ID-swap 場面 f77415 は誤修正を避ける対照として扱います。旧 C30/C34 のスロット名は 2D の正解ラベルに変換していません。</p></section>
<section class="section anchor" id="configuration"><h2>比較条件・全設定・出典</h2><div id="conditions"></div><p>各条件へ同じ C01 行と前後 332 frame を提供し、中心の core で評価します。S1/S2 は過去の履歴のみ、P0/P1 は順逆の文脈を利用します。保存 C27 は以前の全録画結果であり、短区間 S1 と同じ実行履歴・環境の再現ではありません。</p>
<details><summary>固定プロトコルと全パラメーター</summary><pre id="protocol"></pre></details><details><summary>コード・設定・入力の SHA と出力検証</summary><pre id="audit"></pre></details>
<p class="source-list"><a href="protocol.json">protocol.json</a><a href="numerical_manifest.json">初回数値集計時のスナップショット（保存記録）</a><a href="numerical_manifest_verified.json">最終検証済み数値 manifest（正本）</a><a href="verified_numerical_manifest.provenance.json">数値 manifest 検証来歴</a><a href="report_manifest.json">report manifest</a><a href="../../docs/decisions/2026-09-17-variable-count-2d-tracklet-plan.md">実装計画</a><a href="../../docs/reviews/2026-09-16_multiview_identity_review.json">既存人手レビュー</a><a href="../../labs/20260917_variable_count_tracklets/RESULT.md">RESULT.md</a></p>
<p class="muted">図・動画・静止画はこの実行の保存結果から生成。推定用 ID と表示色の対応付けを別方式へ転用していません。全録画再追跡、検出器再学習、3D の統合は今回の範囲外です。</p></section>
</main><script>const REPORT=__PAYLOAD__;</script><script>
const $=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const num=n=>Number.isFinite(n)?n.toLocaleString('ja-JP'):'未判定';const pct=n=>Number.isFinite(n)?(100*n).toFixed(2)+'%':'未判定';let active=null,frame=0,selectedRow=null,loadedKey='',sourceReady=false;const choices=['C27','P0','P1'];
function table(id,heads,rows){$(id).innerHTML='<thead><tr>'+heads.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody>'}
function options(el,values){el.innerHTML=values.map(v=>'<option value="'+esc(v[0])+'">'+esc(v[1])+'</option>').join('')}
options($('case-select'),REPORT.cases.map(c=>[c.name,c.name+' ['+c.start+','+c.stop+')']));options($('view-select'),REPORT.views.map(v=>[v,v]));options($('metric-case'),[['all','全ケース'],...REPORT.cases.map(c=>[c.name,c.name])]);REPORT.views.forEach(v=>$('metric-view').add(new Option(v,v)));$('metric-case').value='cg20s';
$('panels').innerHTML=choices.map((m,i)=>'<div><label>条件 <select id="method-'+i+'">'+REPORT.methods.map(x=>'<option '+(x===m?'selected':'')+'>'+x+'</option>').join('')+'</select></label><canvas id="canvas-'+i+'" width="600" height="600"></canvas></div>').join('');choices.forEach((_,i)=>$('method-'+i).onchange=()=>draw());
async function selectView(target){$('native-source').pause();sourceReady=false;const c=REPORT.cases.find(x=>x.name===$('case-select').value),v=$('view-select').value,key=c.name+'__'+v;loadedKey=key;$('state').textContent='保存した観測・根拠を読み込み中';if(!window.TRACKING_DATA?.[key])await new Promise((resolve,reject)=>{const s=document.createElement('script');s.src='report_data/'+REPORT.payloads[key].script;s.onload=resolve;s.onerror=reject;document.head.appendChild(s)});if(key!==loadedKey)return;active=window.TRACKING_DATA[key];frame=target??c.centers?.[0]??c.start;frame=Math.min(c.stop-1,Math.max(c.start,frame));$('frame-slider').min=$('frame-number').min=c.start;$('frame-slider').max=$('frame-number').max=c.stop-1;$('timeline').src='report_data/'+REPORT.payloads[key].plot;$('native-source').removeAttribute('src');$('native-source').load();$('state').textContent='点・ラベルは全 native frame。元映像ボタンで同じフレームを表示';$('downloads').innerHTML=REPORT.methods.map(m=>'<a href="'+c.name+'/tracking_'+m+'_'+v+'.npz">'+m+' NPZ</a>').join(' ')+' <a href="'+c.name+'/edges_P1_'+v+'.json">P1 edges</a> <a href="'+c.name+'/events_P1_'+v+'.json">P1 保留理由</a> <a href="'+c.name+'/reconnections_P1_'+v+'.json">再接続候補</a>';setFrame(frame,false)}
function color(id,method){if(id<0)return'#c5d0dd';let h=(id*94+REPORT.methods.indexOf(method)*62+REPORT.views.indexOf(active.view)*38+34)%360;return'hsl('+h+' 78% 73%)'}
function setFrame(value,seek=true){if(!active||value===''||!Number.isFinite(Number(value)))return;frame=Math.min(active.stop-1,Math.max(active.start,Math.round(Number(value))));$('frame-slider').value=$('frame-number').value=frame;const count=active.frame.reduce((n,f)=>n+(f===frame),0);$('frame-info').textContent='f'+frame+' / '+(frame/166).toFixed(6)+' 秒 · '+count+' 検出'+(seek&&sourceReady?' · 映像シーク中':'');if(seek&&sourceReady){$('native-source').pause();$('state').textContent='f'+frame+' の元映像を同期中';$('native-source').currentTime=(frame+.25)/166}else draw()}
function draw(){if(!active)return;const rows=[];active.frame.forEach((f,i)=>{if(f===frame)rows.push(i)});const methods=choices.map((_,i)=>$('method-'+i).value);$('frame-info').textContent='f'+frame+' / '+(frame/166).toFixed(6)+' 秒 · '+rows.length+' 検出 · 実魚/反射は未判定';methods.forEach((m,k)=>{const canvas=$('canvas-'+k),ctx=canvas.getContext('2d');ctx.fillStyle='#050a10';ctx.fillRect(0,0,600,600);if(sourceReady&&$('native-source').readyState>=2)ctx.drawImage($('native-source'),0,0,600,600);else{ctx.fillStyle='#8298af';ctx.font='17px sans-serif';ctx.fillText('元映像を読み込むと観測点の背景を表示',18,570)};const a=active.methods[m];rows.forEach((r,slot)=>{const xy=active.points[r],c=color(a.labels[r],m);ctx.strokeStyle=c;ctx.fillStyle=c;ctx.lineWidth=1.4;[[0,1],[1,2]].forEach(([i,j])=>{if(xy[i].every(Number.isFinite)&&xy[j].every(Number.isFinite)){ctx.beginPath();ctx.moveTo(xy[i][0]/2,xy[i][1]/2);ctx.lineTo(xy[j][0]/2,xy[j][1]/2);ctx.stroke()}});xy.forEach((p,i)=>{if(p.every(Number.isFinite)){ctx.beginPath();ctx.arc(p[0]/2,p[1]/2,i===1?4:3,0,Math.PI*2);ctx.stroke();ctx.font='10px sans-serif';ctx.fillText('MPT'[i],p[0]/2+5,p[1]/2-3)}});let p=xy.find(p=>p.every(Number.isFinite));if(p){ctx.font='13px monospace';ctx.fillText('L'+a.labels[r]+(a.adjacent[r]?'':' / new'),Math.max(4,Math.min(460,p[0]/2+9)),Math.max(18,Math.min(550,p[1]/2-12+slot*5)))}});ctx.fillStyle='#fff';ctx.font='15px sans-serif';ctx.fillText(m+' · f'+frame+' · '+rows.length+' rows',12,22)});$('label-head').textContent=methods.join(' / ')+' : local ID (隣接接続)';$('observation-rows').innerHTML=rows.map(r=>'<tr><td>'+active.global_row[r]+'</td><td>'+active.points[r].map((p,i)=>'MPT'[i]+':'+(p.every(Number.isFinite)?'○':'欠')).join(' ')+'</td><td>'+Number(active.score[r]).toFixed(4)+'</td><td>'+methods.map(m=>'L'+active.methods[m].labels[r]+' ('+(active.methods[m].adjacent[r]?'採用':'新規/保留')+')').join(' / ')+'</td><td><button data-row="'+r+'">根拠を見る</button></td></tr>').join('');document.querySelectorAll('[data-row]').forEach(b=>b.onclick=()=>showEvidence(+b.dataset.row));if(rows.length)showEvidence(rows.find(r=>active.global_row[r]===selectedRow)??rows[0]);else $('evidence').textContent='この native frame の元検出は 0 行。観測点は追加しません。'}
function showEvidence(r){selectedRow=active.global_row[r];const d=active.diagnostics[String(selectedRow)]??{note:'この core 行に保存された P1 event/edge がありません。初期行または接続候補なし。全記録リンクも参照。'};$('evidence').textContent=JSON.stringify({native_frame:frame,global_row:selectedRow,observed_nodes:active.points[r],instance_score:active.score[r],reflection_status:'unknown',...d},null,2)}
$('case-select').onchange=()=>selectView();$('view-select').onchange=()=>selectView();$('frame-slider').oninput=e=>setFrame(e.target.value);$('frame-number').onchange=e=>setFrame(e.target.value);$('frame-number').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();setFrame(e.target.value)}};$('apply-frame').onclick=()=>setFrame($('frame-number').value);$('previous-frame').onclick=()=>setFrame(frame-1);$('next-frame').onclick=()=>setFrame(frame+1);$('load-source').onclick=()=>{const video=$('native-source');video.pause();sourceReady=false;setFrame($('frame-number').value,false);$('state').textContent='元映像を読み込み中';video.src=REPORT.sources[active.view];video.load();video.onloadedmetadata=()=>{sourceReady=true;video.currentTime=(frame+.25)/166;video.playbackRate=+$('speed').value;$('state').textContent='元映像と native frame を同期'};video.onerror=()=>{$('state').textContent='元映像を開けません。書き出し比較動画と静止画は下にあります。'}};$('native-source').onseeked=()=>{$('state').textContent='元映像と native f'+frame+' を同期';draw()};$('speed').onchange=()=>{$('native-source').playbackRate=+$('speed').value};$('native-play').onclick=()=>{const v=$('native-source');if(!sourceReady){$('load-source').click();return}v.paused?v.play():v.pause()};
const nativeVideo=$('native-source');function onVideoFrame(now,meta){if(sourceReady&&!nativeVideo.paused&&active){const f=Math.floor(meta.mediaTime*166+1e-4);if(f>=active.stop){nativeVideo.pause();setFrame(active.stop-1)}else{frame=Math.max(active.start,f);$('frame-slider').value=$('frame-number').value=frame;draw()}}nativeVideo.requestVideoFrameCallback(onVideoFrame)}if(nativeVideo.requestVideoFrameCallback)nativeVideo.requestVideoFrameCallback(onVideoFrame);else nativeVideo.ontimeupdate=()=>{if(!nativeVideo.paused)setFrame(Math.floor(nativeVideo.currentTime*166),false)};
options($('asset-select'),REPORT.assets.map((a,i)=>[i,a.view+' · '+(a.center_frame===null?'主20秒 / 全体':'f'+a.center_frame+' / 拡大・低速')+' · '+a.case]));function selectAsset(){const a=REPORT.assets[+$('asset-select').value];if(!a)return;$('comparison-video').src='review/'+a.video;$('comparison-video').poster='review/'+a.center;$('asset-caption').textContent=a.output_frames+' 枚 / '+a.output_duration_seconds.toFixed(3)+' 秒 · native ['+a.native_frame_range_half_open.join(',')+') · 全 '+a.full_decode.decoded_frames+' 枚のデコード検証済み';$('stills').innerHTML=['before','center','after'].map((k,i)=>'<figure><a target="_blank" href="review/'+a[k]+'"><img loading="lazy" src="review/'+a[k]+'" alt="'+a.view+' '+k+' f'+a.still_native_frames[i]+'"></a><figcaption>'+['前','中心','後'][i]+' · f'+a.still_native_frames[i]+' · クリックで原寸</figcaption></figure>').join('')} $('asset-select').onchange=selectAsset;$('asset-to-frame').onclick=()=>{const a=REPORT.assets[+$('asset-select').value];$('case-select').value=a.case;$('view-select').value=a.view;selectView(a.center_frame??a.still_native_frames[1]);$('viewer').scrollIntoView({behavior:'smooth'})};table('media-table',['ビュー / native','動画','フレーム数','秒','検証'],REPORT.assets.map(a=>[esc(a.view+' / '+(a.center_frame??'20s')),'<a href="review/'+a.video+'">'+esc(a.video)+'</a>',num(a.output_frames),a.output_duration_seconds.toFixed(3),a.full_decode.verified?'全フレーム OK':'未確認']));
const heads=['条件','core 観測行','保持率','断片数','隣接接続','分母','接続率','保留率','欠測越し'];function metricCells(r){return[esc(r.method),num(r.core_rows),pct(r.retention_rate),num(r.contiguous_runs),num(r.adjacent_connections),num(r.eligible_observations),pct(r.connection_rate),pct(r.abstention_rate),num(r.gap_connections)]}function updateMetrics(){const cs=$('metric-case').value,v=$('metric-view').value,rs=REPORT.records.filter(r=>(cs==='all'||r.case===cs)&&(v==='all'||r.view===v));const sums=Object.keys(REPORT.protocol.conditions).map(m=>{const a=rs.filter(r=>r.method===m),o={method:m};['core_rows','retained_rows','contiguous_runs','adjacent_connections','eligible_observations','unconnected_observations','gap_connections'].forEach(k=>o[k]=a.reduce((n,r)=>n+(r[k]??0),0));o.retention_rate=o.retained_rows/o.core_rows;o.connection_rate=o.adjacent_connections/o.eligible_observations;o.abstention_rate=o.unconnected_observations/o.eligible_observations;return o});table('metric-table',heads,sums.map(metricCells));table('count-table',['ケース','ビュー','core frame','0検出','1検出','2検出','3検出','4+検出','M欠損行','P欠損行','T欠損行'],rs.filter(r=>r.method==='C27').map(r=>[esc(r.case),esc(r.view),num(r.core_frames),...['0','1','2','3','4+'].map(k=>num(r.detection_count_frames[k])),num(r.missing_m),num(r.missing_p),num(r.missing_t)]));const ss=rs.filter(r=>['P0','P1'].includes(r.method)).flatMap(r=>(r.strata??[]).filter(s=>s.eligible_observations>0).map(s=>[esc(r.case),esc(r.view),r.method,s.detection_count,s.proximity,s.any_node_missing?'あり':'なし',num(s.eligible_observations),num(s.adjacent_connections),pct(s.connection_rate)]));table('strata-table',['ケース','ビュー','条件','検出数','近接','M/T 等欠損','分母','接続','率'],ss)}$('metric-case').onchange=updateMetrics;$('metric-view').onchange=updateMetrics;table('all-metrics',['ケース','ビュー',...heads],REPORT.records.map(r=>[esc(r.case),esc(r.view),...metricCells(r)]));
const main=REPORT.records.filter(r=>r.case==='cg20s'&&r.method==='P1'),rows=main.reduce((n,r)=>n+r.core_rows,0);$('kpis').innerHTML=[['3 視点 × 20 秒',num(rows)+' 観測行を同じまま比較'],[REPORT.cases.length+' ケース','既知11＋事前固定の別時刻検証'],[REPORT.assets.length+' 比較動画',REPORT.assets.length*3+' 枚の前・中心・後画像'],['2D 正解分母 0','正しい継続率 / 誤接続率：未判定']].map(x=>'<div class="kpi"><b>'+x[0]+'</b><span>'+x[1]+'</span></div>').join('');
$('protocol').textContent=JSON.stringify(REPORT.protocol,null,2);$('audit').textContent=JSON.stringify(REPORT.audit,null,2);$('conditions').innerHTML='<table>'+Object.entries(REPORT.protocol.conditions).map(([k,v])=>'<tr><th>'+esc(k)+'</th><td>'+esc(v)+'</td></tr>').join('')+'</table>';$('human-review').innerHTML=REPORT.human.events.map(e=>'<p><b>f'+e.frame+'</b> '+esc(e.verbatim_note)+' <span class="muted">['+esc(e.kind)+', view='+esc(e.view??'未指定')+']</span></p>').join('');
$('findings').innerHTML=REPORT.findings.map(x=>'<p>'+esc(x)+'</p>').join('');table('case-findings',['ケース','core 範囲','P0 → P1 断片数（全ビュー）','評価上の注意 / 残る判断'],REPORT.cases.map(c=>{const get=m=>REPORT.records.filter(r=>r.case===c.name&&r.method===m).reduce((n,r)=>n+r.contiguous_runs,0);return[esc(c.name),'['+c.start+', '+c.stop+')',num(get('P0'))+' → '+num(get('P1')),esc(REPORT.case_notes[c.name]??'全行保持と構造指標を比較。実魚対応・ID正誤は追加注釈待ち。')]}));
table('event-findings',['ケース / ビュー / native','局所所見','残る3D対応','追加レビュー項目'],(REPORT.event_findings.records??[]).map(r=>[esc(r.case+' / '+r.view+' / f'+r.center),esc(r.finding),esc(r.remaining_3d),esc(r.review_question)]));selectAsset();updateMetrics();selectView().catch(e=>{$('state').textContent='読込エラー: '+e.message;console.error(e)});
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
    result_summary = f"全観測を保持する一方、主20秒の断片数は P0 の {main_p0:,} から P1 の {main_p1:,} に増加し、補助ノード付きでも {main_aux:,} です。"
    findings = [
        f"主20秒の3ビュー合計の連続断片数は P0={main_p0:,}、P1={main_p1:,}。これは構造的な分断の比較であり、同一個体の正解を確認した件数ではありません。",
        "S1 と S2 は戻り行対応を共通に修正し、OKS の尺度だけを比較。S2 の top=14400 / side=10000 px² は結果を確認する前に固定しました。",
        "P1 は内部条件を満たした隣接接続だけを確定し、消えた端点・再出現後の再接続候補を別表に残します。M/T 欠損と個体 ID の切替を同一視しません。",
        "保存 C27・S1/S2 の欠測越し接続と、P0/P1 の隣接接続だけからなる断片を区別して集計しています。",
        "既知11ケースは開発・回帰確認です。追加の別時刻区間は結果を見る前にメタデータから選定し、設定固定後に実行した同じ3匹の検証区間です。独立群への一般化ではありません。",
        "診断記録では、保留のたびに古い予測端点と新しい断片端点が同じ経路で競合し続けます。不確かさをコストから差し引くため古い仮説が有利になる例があります。side1 f76304 は他の現検出まで109.11 pxと離れる一方、隣接候補の予測誤差1.653 pxに対し、2 frame前の端点は2.580−1.5=1.080 pxとなり、候補差0.573 pxが必要差2.662 pxを下回ります。top f75498 では17の候補が同じ凍結P0ラベルに由来します。P0ラベルも正解ではありませんが、これらは端点仮説の重複による保留連鎖を示す内部診断です。",
        "no-swap 対照 f76791/top の局所17フレームは P0=3断片から P1=35（追加切断32、P1aux=5）へ悪化。旧注記が明瞭とする f21232/side2 も3→19、f31479/side1 は5→21。f323819/side1 のno-swap対照では6→16（P1aux=12）です。これらは新しい同一性正解を付けずに確認できる分断の増加です。",
        "一方、f30735/side2 の局所17フレームは全6方式で1断片、f302762/side1 は全方式3断片。f333619/top では保存C27の最大隣接P移動233 pxに対し S1/S2/P0/P1 は7.78 px・4断片で共通します。大ジャンプ減少をP1固有の改善とは呼びません。",
    ]
    notes = {
        "cg20s": "f76235 / f76930 / f76988 は既存の3Dレビューでswap。f76791 はno-swap、f77415 はnode exchange / no-ID-swap。各ビュー2Dの入口/出口対応は未確定。",
        "known_21232": "旧注記では top/side2 が明瞭。2D断片と古い3D slot切替を分離し、今回の正誤は新規注釈待ち。",
        "known_30735": "side2 の1検出から2検出への変化。同じ2D断片の3D割当変更と、2D接続切替を分離。",
        "known_31479": "side1 の4〜5検出を保持。余剰は反射と断定しない。旧3D大ジャンプの原因を2Dの正誤へ直結させない。",
        "known_105541": "旧注記は2Dが明瞭とする。今回P1の継続が同一個体か、前後動画による確認が必要。",
        "known_260434": "side1 の2検出と旧融合注記、side2 の4検出。遮蔽・余剰の原因は行ごとのラベルなし。",
        "known_302762": "旧注記ではside1が最も明瞭、topでノード混入。M/P/Tの支持と個体ID交換を分けて評価。",
        "known_323819": "後の人手レビューは「スワップない、パーフェクト」。side1 3→2→3→1検出でも新規断片とID誤修正を混同しない。",
        "known_333619": "旧注記では2Dが明瞭なのに3Dで一過性swap。今回の2D判定のみで3D問題を解決済みとはしない。",
        "control_50000": "既に調査した回帰対照。未注釈であり、無誤りの正解対照とは仮定しない。",
        "holdout20s": "既に調査済みの20秒。同じ3匹の別時刻の回帰確認で、新規の未見テストとは呼ばない。",
        "validation20s_00332": "結果を見る前に既知区間・注釈のメタデータと前後文脈の重なりを除外し、規定の格子で最初の20秒を選定。設定固定後に実行した同じ3匹の別時刻検証。過去に一度も見ていない保証や独立群の検証ではない。",
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

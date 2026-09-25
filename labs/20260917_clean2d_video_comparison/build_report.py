"""Source-backed, per-view 2D comparison report; never infer identity ground truth."""

# ruff: noqa: RUF001 -- Japanese report punctuation is intentional.

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "work/clean2d_video_comparison_v1"
PLUGIN = Path("C:/Users/machi/.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.8")
NODE = Path(
    "C:/Users/machi/AppData/Local/OpenAI/Codex/runtimes/cua_node/1b23c930bdf84ed6/bin/node.exe"
)
REVIEWS = ROOT / "docs/reviews/2026-09-16_multiview_identity_review.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def source(label, files, definition, components):
    files = [Path(p) for p in files if Path(p).is_file()]
    return dict(
        label=label,
        classification="observed",
        files=[rel(p) for p in files],
        hashes={rel(p): sha256(p) for p in files},
        metricDefinitions=[dict(label=label, definition=definition, componentIds=components)],
        evidenceFlow=[
            dict(
                title="Saved local evidence",
                detail=(
                    "Read saved JSON and NPZ metrics at their recorded native-frame scope; "
                    "no SQL or live service."
                ),
            ),
            dict(title="Adapter", detail=rel(__file__) + " SHA256 " + sha256(Path(__file__))),
        ],
    )


def normalize(summary, protocol):
    """Preserve exact core scopes, with input absence separated from association."""
    cases = [
        dict(id=c["name"], label=("主20秒" if c["name"] == "cg20s" else c["name"]), **c)
        for c in protocol.get("cases", [])
    ]
    metrics, observations, sensitivity, reasons = [], [], [], []
    for record in summary.get("records", []):
        common = {k: record[k] for k in ("case", "view", "frame_start", "frame_stop")}
        baseline = record["arms"]["SLEAP"]
        observations.append(
            dict(
                **common,
                same_detection_rows_all_arms=record["same_detection_rows_all_arms"],
                **{
                    k: baseline[k]
                    for k in (
                        "native_frames",
                        "detection_rows",
                        "retained_rows",
                        "frames_without_detections",
                        "frames_below_three",
                        "missing_primary_rows",
                    )
                },
                **{
                    node + "_missing_rows": values["missing_rows"]
                    for node, values in baseline["nodes"].items()
                },
            )
        )
        for arm, values in record["arms"].items():
            flat = {
                **common,
                "condition": arm,
                **values,
                "large_primary_step_count": len(values["large_primary_steps"]),
            }
            for node, counts in values["nodes"].items():
                flat.update({node + "_" + key: value for key, value in counts.items()})
            (metrics if arm in ("SLEAP", "P", "N", "F") else sensitivity).append(flat)
            if arm == "F":
                sensitivity.append(flat)
            if arm == "F":
                reasons.extend(
                    dict(**common, reason=k, count=v)
                    for k, v in values.get("cut_reason_counts_nonexclusive", {}).items()
                )
    parameters = [
        dict(condition="P/N/F", parameter=k, value=v, basis="保存protocolの実行設定")
        for k, v in protocol.get("tracking_config", {}).items()
    ]
    settings = [
        dict(condition=arm, parameter=k, baseline=protocol["tracking_config"][k], value=v)
        for arm, config in protocol.get("sensitivity_configs", {}).items()
        for k, v in config.items()
        if v != protocol["tracking_config"][k]
    ]
    historical = [
        dict(
            case=c["name"],
            frame=int(c["note"]["frame_start"]),
            verbatim=c["note"]["annotator_note"],
            chain=c.get("note_chain"),
            start=c["start"],
            stop=c["stop"],
        )
        for c in protocol.get("cases", [])
        if "note" in c
    ]
    return dict(
        cases=cases,
        metrics=metrics,
        observations=observations,
        parameters=parameters,
        sensitivity=sensitivity,
        reasons=reasons,
        sensitivity_settings=settings,
        historical=historical,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BASE)
    parser.add_argument("--media-manifest", type=Path)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    base = args.root.resolve()
    app, inputs = base / "report_app", base / "report_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    summary_path = base / "summary.json"
    protocol_path = base / "protocol.json"
    protocol = read(protocol_path) if protocol_path.exists() else {}
    measured_paths = sorted(base.glob("*/metrics_*.json"))
    summary = (
        read(summary_path)
        if summary_path.exists()
        else dict(records=[read(p) for p in measured_paths])
    )
    normalized = normalize(summary, protocol)
    annotation = read(REVIEWS)
    notes = [
        dict(
            frame=e["frame"],
            verbatim=e["verbatim_note"],
            kind=e["kind"],
            original_video=e["video"],
            original_video_sha256=e["video_sha256"],
            panel=annotation.get("reviewed_panel"),
            exact_interval=e.get("human_exact_interval"),
        )
        for e in annotation["events"]
    ]
    data_source = source(
        "同じ検出点に対する2D追跡の保存指標",
        [summary_path, protocol_path, *measured_paths],
        "各case・view・conditionのnative範囲のみ。検出row、欠けたnode、負の追跡ID、"
        "接続と断片を分離する。個体ID精度を測る正解列ではない。",
        [
            "summary",
            "metrics-table",
            "observations",
            "metric-fragments",
            "metric-links",
            "sensitivity",
            "reasons",
            "historical",
        ],
    )
    queries = {
        key: dict(rows=normalized[key], source=data_source)
        for key in (
            "metrics",
            "observations",
            "parameters",
            "sensitivity",
            "sensitivity_settings",
            "reasons",
            "historical",
        )
    }
    queries["human_reviews"] = dict(
        rows=notes,
        source=source(
            "過去の人手レビュー原文",
            [REVIEWS],
            "以前の3D比較動画へのレビュー。3D slot名とSLEAPの視点内IDは異なり、"
            "2D追跡のswap正解には読み替えない。",
            ["review-note", "review-history"],
        ),
    )
    windows_path = base / "review_windows.json"
    window_rows = []
    if windows_path.exists():
        for record in read(windows_path)["records"]:
            for arm, values in record["arms"].items():
                window_rows.append(
                    {
                        "case": record["case"],
                        "view": record["view"],
                        "center_frame": record["anchor"],
                        "condition": arm,
                        "scope": (
                            "中心±8（17 frames）"
                            if record["window_kind"] == "anchor_17_native_frames"
                            else "動画範囲（166 frames）"
                        ),
                        "window_kind": record["window_kind"],
                        **values,
                        "large_primary_step_count": len(values["large_primary_steps"]),
                    }
                )
    queries["local_metrics"] = dict(
        rows=window_rows,
        source=source(
            "中心17 framesと動画166 framesの再集計",
            [windows_path],
            "各frame_start以上frame_stop未満。広い既知区間の切断を注釈中心の切断と混同しない。"
            "同じ検出に対するSLEAP/P/N/F/F_all_looseの直接2D指標。",
            ["local-metrics"],
        ),
    )
    audit_path = base / "parameter_audit.json"
    parameter_audit = read(audit_path) if audit_path.exists() else {}
    if parameter_audit:
        queries["parameters"]["rows"].extend(
            dict(
                condition=f"保存C27 {view}",
                parameter=k,
                value=str(v),
                basis="保存済み effective_tracker_config",
            )
            for view, config in parameter_audit["saved_c27_effective_config"].items()
            for k, v in config.items()
        )
        requested = parameter_audit["saved_c27_requested_parameters"]
        queries["parameters"]["rows"].extend(
            dict(
                condition="保存C27 全視点",
                parameter=k,
                value=json.dumps(requested[k], ensure_ascii=False),
                basis="保存済み requested parameters",
            )
            for k in ("nodes", "scoring", "window_size", "min_match_points", "whole_recording")
        )
        queries["parameters"]["source"] = source(
            "保存C27と固定P/N/Fの実行設定",
            [protocol_path, audit_path],
            "C27は保存effective config、P/N/Fは実行protocol。現在のinstalled実装の"
            "合成監査と、当時の実行環境の完全再現は区別する。",
            ["parameters", "limitations"],
        )
    queries["parameter_mapping"] = dict(
        rows=parameter_audit.get("parameter_mapping", []),
        source=source(
            "追跡方式とパラメータの対応監査",
            [audit_path],
            "同じ名前の閾値でも候補集合・履歴・割当規則が違えば同じ実験にならない。",
            ["parameter-mapping", "replay-notes"],
        ),
    )
    queries["parameter_audit"] = dict(
        rows=[parameter_audit] if parameter_audit else [],
        source=source(
            "現在の実装・保存パラメータ・実データ局所履歴の監査",
            [audit_path, base / "c27_replay/summary.json"],
            "過去の全編実行の内部scoreを再現したものではない。"
            "sigma比較は固定した局所履歴の再採点で、全追跡の感度試験ではない。",
            ["scoring-notes", "software-notes"],
        ),
    )
    c27_path = base / "c27_sensitivity/summary.json"
    c27_rows = []
    if c27_path.exists():
        for record in read(c27_path)["records"]:
            c27_rows.append(
                {
                    "case": record["case"],
                    "view": record["view"],
                    "window": record["window"],
                    **record["metrics"],
                    "large_primary_step_count": len(record["metrics"]["large_primary_steps"]),
                    "npz_sha256": record["npz_sha256"],
                    "readout_different_rows_core": record["readout_different_rows_core"],
                    "maximum_pectoral_step_px": record["metrics"]["nodes"]["pectoral"][
                        "max_adjacent_step_px"
                    ],
                }
            )
    queries["c27_sensitivity"] = dict(
        rows=c27_rows,
        source=source(
            "同一局所区間と環境でのSLEAP履歴長比較",
            [c27_path, base / "c27_sensitivity/protocol.json"],
            "row-faithful出力でwindow=1/5/10のみを変更。全セッション保存C27とは履歴が異なり、"
            "その完全再現として扱わない。",
            ["c27-sensitivity"],
        ),
    )
    queries["media"] = dict(
        rows=[],
        source=source(
            "各視点の直接比較動画",
            [],
            "SLEAP/P/N/Fの4条件を同じraw frameと同じ検出点で比較する。NEW、UNASSIGNEDは"
            "検出が消えたことを意味しない。",
            ["comparison-video"],
        ),
    )
    existing = read(app / "src/data.json") if (app / "src/data.json").exists() else {}
    if args.complete and not normalized["metrics"]:
        raise ValueError("Cannot complete a report without measured data")
    snapshot = dict(
        surface="report",
        title="同じ2D検出で、SLEAPと独自追跡を視点ごとに比較する",
        generatedAt=datetime.now(UTC).isoformat(),
        buildStatus="complete" if args.complete else "updating",
        status="reviewed" if args.complete else "partial",
        filters=[],
        queries=queries,
        cases=normalized["cases"],
        report={"asOf": "2026-09-17"},
        summaryMarkdown="## 2Dの直接比較を準備中\n\n各視点でSLEAPとP/N/Fを同じ検出に重ねます。"
        "検出不足、ノード欠損、未割当、断片の切断を分けて確認します。",
        interpretationMarkdown="このレポートは2Dの比較です。前回3Dで観測を採用しなかった"
        "ことを、2Dの検出が消えたこととして数えません。",
    )
    snapshot["protocol"] = protocol
    if parameter_audit:
        snapshot["parameterNotes"] = (
            "保存C27はmouth/pectoral・OKS・Hungarian・履歴5観測・max_tracksなし。"
            "window=5は5 native framesでの失効ではありません。P/N/Fは隣接frameの"
            "双方向合意と明示的な拒否条件を使います。数値の厳しさだけでは比較できません。"
        )
    if normalized["metrics"]:
        snapshot["summaryMarkdown"] = (
            "## 検出は残り、時間方向の接続が切れています\n\n"
            "主20秒の全29,386検出はSLEAP・P・N・Fの全条件に残っています。"
            "連続した2D断片はSLEAP 224に対しF 1,324。Fは点を消さず、"
            "不確かな接続を切って新しい断片番号を付けています。\n\n"
            "mouthまたはpectoralが欠ける検出は743行あります。"
            "FがSLEAPの接続を切断・変更した1,129件のうち、720件では"
            "端点の主要ノード欠損が拒否理由に含まれます（他理由との重複あり）。"
            "閾値をまとめて緩めてもFの断片は1,028。必要ノードの条件は固定した実験です。\n\n"
            "断片が少ないこと、動きが滑らかなことだけでは、個体IDが正しいと判定できません。"
        )
    snapshot["limitationsMarkdown"] = (
        "## 比較条件と限界\n\nSLEAPは保存済みの全セッションC27追跡、P/N/Fは同じC01検出点から"
        "前後332 native framesの文脈を使った処理です。Pはpectoral、Nはmouth＋pectoralで"
        "候補を選びtailを整合性検査に使い、FはさらにLK flowを使います。"
        "P/N/Fは隣接frameの双方向一致だけを接続し、2Dの空白期間は跨ぎません。"
        "入力の検出点とノードの組み合わせは修正しません。\n\n"
        "同じ3匹の既知区間を再評価しています。全個体の2D ID正解がないため、"
        "swap精度や旧方式への優越は未確認です。過去のS0/S1/S2は3D側のslot名で、"
        "視点ごとのSLEAP番号とは別です。"
    )
    if "id" in existing:
        snapshot["id"] = existing["id"]
    findings_path = base / "case_findings.json"
    if findings_path.exists():
        snapshot["queries"]["case_findings"] = dict(
            rows=read(findings_path)["records"],
            source=source(
                "既知8場面の再評価",
                [findings_path, windows_path, base / "inspection/manifest.json", audit_path],
                "数値は中心±8の17native frames、目視は中心±1の3frame。"
                "アシスタントによる確認で、新しい人手ID正解ではない。",
                ["case-findings"],
            ),
        )
    narrative_path = base / "final_narrative.json"
    if not narrative_path.exists():
        narrative_path = inputs / "final_narrative.json"
    if narrative_path.exists():
        narrative = read(narrative_path)
        snapshot.update(
            {
                k: v
                for k, v in narrative.items()
                if k
                in {
                    "title",
                    "summaryMarkdown",
                    "interpretationMarkdown",
                    "parameterNotes",
                    "limitationsMarkdown",
                    "resultNotes",
                    "sensitivityNotes",
                    "replayNotes",
                    "historyNotes",
                    "scoringNotes",
                    "softwareNotes",
                }
            }
        )
        snapshot["queries"]["narrative"] = dict(
            rows=[dict(key=k, text=v) for k, v in narrative.items() if isinstance(v, str)],
            source=source(
                "実測・原画・設定監査を照合した解釈",
                [
                    narrative_path,
                    summary_path,
                    windows_path,
                    audit_path,
                    c27_path,
                    base / "audit.json",
                    ROOT / "work/clean_tracklets_v2/summary.json",
                ],
                "元の数値・定義・注釈と照合したレポート文面。ID正解の追加ではない。",
                [
                    "summary",
                    "result-notes",
                    "sensitivity-notes",
                    "replay-notes",
                    "limitations",
                    "scoring-notes",
                    "software-notes",
                ],
            ),
        )
    snapshot_path = inputs / "reviewed_snapshot.json"
    write(snapshot_path, snapshot)
    if not (app / "AGENTS.md").exists():
        subprocess.run(
            [
                str(NODE),
                str(PLUGIN / "scripts/prepare-data-app.mjs"),
                "--surface",
                "report",
                "--output",
                str(app),
                "--snapshot",
                str(snapshot_path),
            ],
            check=True,
        )
        snapshot = read(app / "src/data.json")
    manifest_path = args.media_manifest or base / "review/manifest.json"
    imports, mappings, media_rows = [], [], []
    assets = app / "src/content/assets"
    assets.mkdir(exist_ok=True)
    if manifest_path.exists():
        manifest = read(manifest_path)
        for entry in manifest["assets"]:
            for name, kind in (("video", "video"), ("still", "image")):
                original = Path(entry[name])
                if not original.is_absolute():
                    original = manifest_path.parent / original
                digest = sha256(original)
                if digest != entry[name + "_sha256"]:
                    raise ValueError(f"Media hash mismatch: {original}")
                asset_name = original.stem + "_" + digest[:12] + original.suffix
                shutil.copy2(original, assets / asset_name)
                key = f"m{len(imports)}"
                imports.append(f'import {key} from "../assets/{asset_name}";')
                mappings.append(f"  {json.dumps(rel(original))}: {key}")
                media_rows.append(
                    {
                        **entry,
                        "kind": kind,
                        "path": rel(original),
                        "asset_key": rel(original),
                        "sha256": digest,
                    }
                )
        snapshot["queries"]["media"] = dict(
            rows=media_rows,
            source=source(
                "各視点の直接比較動画",
                [manifest_path],
                "同じ検出を全て表示し、条件ごとの時間IDだけを比較。局所は166 native framesを"
                "30fpsで再生し補間しない。原版とreport用がある場合は双方のSHAを記録。",
                ["comparison-video"],
            ),
        )
    inspection_path = base / "inspection/manifest.json"
    inspection_rows = []
    if inspection_path.exists():
        for entry in read(inspection_path)["assets"]:
            if (entry["anchor"], entry["view"]) not in {(21232, "side2_cam"), (333619, "top_cam")}:
                continue
            original = ROOT / entry["path"]
            digest = sha256(original)
            assert digest == entry["sha256"]
            asset_name = original.stem + "_" + digest[:12] + original.suffix
            shutil.copy2(original, assets / asset_name)
            key = f"m{len(imports)}"
            imports.append(f'import {key} from "../assets/{asset_name}";')
            mappings.append(f"  {json.dumps(rel(original))}: {key}")
            inspection_rows.append(
                {**entry, "kind": "image", "path": rel(original), "asset_key": rel(original)}
            )
    snapshot["queries"]["inspection"] = dict(
        rows=inspection_rows,
        source=source(
            "native3frameの原画cropによる確認",
            [inspection_path, base / "c27_replay/summary.json"],
            "同じcropを3つのnative frameに適用。表示は測定2D点と追跡番号で、"
            "新しい人手ID正解は与えていない。",
            ["inspection"],
        ),
    )
    if args.complete:
        assert summary.get("complete") and len(summary["records"]) == 33
        assert read(audit_path)["status"] == "complete"
        assert read(c27_path)["status"] == "complete"
        assert read(base / "audit.json")["status"] == "passed"
        assert read(manifest_path)["status"] in {"complete", "verified"}
        assert len(media_rows) == 84 and len(inspection_rows) == 2
        assert narrative_path.is_file()
    with provenance("clean2d_report_snapshot", inputs, name="snapshot") as rec:
        for path in [
            Path(__file__),
            REVIEWS,
            summary_path,
            protocol_path,
            windows_path,
            audit_path,
            c27_path,
            *measured_paths,
            manifest_path,
            inspection_path,
            findings_path,
            narrative_path,
        ]:
            if path.is_file():
                rec.input(rel(path), path, sha256=sha256(path))
        write(snapshot_path, snapshot)
        write(app / "src/data.json", snapshot)
        (app / "src/content/report/media.js").write_text(
            "\n".join(imports) + "\nexport const media = {\n" + ",\n".join(mappings) + "\n};\n",
            encoding="utf-8",
            newline="\n",
        )
        rec.output("snapshot", snapshot_path, sha256=sha256(snapshot_path))
    print(json.dumps(dict(app=rel(app), metrics=len(normalized["metrics"]), media=len(media_rows))))


if __name__ == "__main__":
    main()

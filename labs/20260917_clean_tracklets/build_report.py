"""Make a source-backed snapshot for the separate clean-fragment report app."""
# Japanese report copy intentionally uses Japanese punctuation and kana.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "work/clean_tracklets_v1"
PLUGIN = Path("C:/Users/machi/.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.8")
NODE = Path(
    "C:/Users/machi/AppData/Local/OpenAI/Codex/runtimes/cua_node/1b23c930bdf84ed6/bin/node.exe"
)
REVIEWS = ROOT / "docs/reviews/2026-09-16_multiview_identity_review.json"
CONDITIONS = {
    "saved_C34": "保存C34",
    "previous_custom_C_v3": "前回C v3",
    "initial_clean_v1": "初期F→3D v1",
    "new": "新F→3D v2",
}
CASES = ("cg20s", "known_30735", "known_31479", "known_323819", "control_50000", "holdout20s")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def source(label, paths, definition, components):
    paths = [Path(p) for p in paths if Path(p).is_file()]
    return dict(
        label=label,
        classification="observed",
        files=[rel(p) for p in paths],
        hashes={rel(p): sha256(p) for p in paths},
        metricDefinitions=[dict(label=label, definition=definition, componentIds=components)],
        evidenceFlow=[
            dict(
                title="Saved local evidence",
                detail="Read saved JSON and NPZ outputs; no remote queries or manual reassignment.",
            ),
            dict(
                title="Report adapter", detail=rel(__file__) + " SHA256 " + sha256(Path(__file__))
            ),
        ],
    )


def metric_rows(case):
    tracking, geometry, intervals, bridges, flows = [], [], [], [], []
    for arm, views in case["tracking"].items():
        for view, values in views.items():
            tracking.append(
                dict(
                    case=case["name"],
                    condition=arm,
                    view=view,
                    **{k: v for k, v in values.items() if not isinstance(v, dict)},
                    median_fragment_frames=values["fragment_length_frames"]["median"],
                    max_fragment_frames=values["fragment_length_frames"]["maximum"],
                    maximum_step_px=values["step_px"]["maximum"],
                )
            )
    for key, label in CONDITIONS.items():
        if key not in case["comparison"]:
            continue
        result = case["comparison"][key]
        g, f = result["geometry"], result["female"]
        fc = f.get("interval_counts", {})
        geometry.append(
            dict(
                case=case["name"],
                condition=label,
                condition_id=key,
                **{k: v for k, v in g.items() if not isinstance(v, dict)},
                maximum_step_mm=g["step_mm"]["maximum"],
                female_requested_intervals=f.get("requested_intervals"),
                female_preserved=fc.get("preserved"),
                female_swap=fc.get("swap"),
                female_unscored=fc.get("unscored"),
            )
        )
        for interval in f.get("intervals", []):
            intervals.append(dict(case=case["name"], condition=label, **interval))
    for i, item in enumerate(case.get("bridges", [])):
        bridges.append(dict(case=case["name"], proposal=i, **item))
    for view, values in case.get("flow", {}).items():
        flows.append(
            dict(case=case["name"], view=view, metadata=json.dumps(values, ensure_ascii=False))
        )
    return tracking, geometry, intervals, bridges, flows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BASE)
    parser.add_argument(
        "--analysis", type=Path, help="Versioned inputs; report app stays under --root"
    )
    parser.add_argument("--complete", action="store_true")
    parser.add_argument(
        "--media-manifest", type=Path, help="Verified report-size media derivatives"
    )
    args = parser.parse_args()
    base, app = args.root.resolve(), args.root.resolve() / "report_app"
    analysis = args.analysis.resolve() if args.analysis else base
    paths = [
        analysis / name / "summary.json"
        for name in CASES
        if (analysis / name / "summary.json").exists()
    ]
    cases = [read(p) for p in paths]
    evidence_paths = [*paths, *[analysis / c["name"] / "reconstruction.npz" for c in cases]]
    evidence_paths.extend(
        BASE / c["name"] / "reconstruction.npz"
        for c in cases
        if "initial_clean_v1" in c["comparison"]
    )
    protocol_path = analysis / "protocol.json"
    annotation = read(REVIEWS)
    review_rows = [
        dict(
            frame=e["frame"],
            verbatim=e["verbatim_note"],
            kind=e["kind"],
            reviewed_video=e["video"],
            reviewed_video_sha256=e["video_sha256"],
            reviewed_panel=annotation.get("reviewed_panel", "unspecified"),
            exact_interval=e.get("human_exact_interval"),
        )
        for e in annotation["events"]
    ]
    methods = [
        dict(
            condition="P",
            name="胸びれ点のみ",
            description="視点ごとに胸びれの移動と前後の対応一致を使い、曖昧な箇所で短い断片に分ける。",
        ),
        dict(
            condition="N",
            name="3ノードの動き",
            description="口・胸びれ・尾の動きの整合性を加える。検出行内のノードを別の魚に再分配しない。",
        ),
        dict(
            condition="F",
            name="3ノード＋画像フロー",
            description=(
                "前後1 native frameの画像フローで各ノードの移動を検証する。"
                "F断片から3Dを再構成する。"
            ),
        ),
    ]
    tracking, geometry, intervals, bridges, flows = [], [], [], [], []
    for c in cases:
        rows = metric_rows(c)
        with np.load(analysis / c["name"] / "reconstruction.npz", allow_pickle=False) as z:
            core = (z["frames"] >= c["start"]) & (z["frames"] < c["stop"])
            present = np.isfinite(z["xyz"][core]).all(-1)
            held = z["identity_review"][core]
            measured_held = int((present & held).sum())
        for row in rows[1]:
            row["measured_and_review_fish_frames"] = (
                measured_held if row["condition_id"] == "new" else None
            )
            if row["condition_id"] == "initial_clean_v1":
                with np.load(BASE / c["name"] / "reconstruction.npz", allow_pickle=False) as z:
                    old_core = (z["frames"] >= c["start"]) & (z["frames"] < c["stop"])
                    old_present = np.isfinite(z["xyz"][old_core]).all(-1)
                    row["measured_and_review_fish_frames"] = int(
                        (old_present & z["identity_review"][old_core]).sum()
                    )
            elif row["condition_id"] != "new":
                row["identity_review_fish_frames"] = None
            elif "initial_clean_v1" not in c["comparison"]:
                row["condition"] = "初期F→3D v1"
        for destination, values in zip(
            (tracking, geometry, intervals, bridges, flows), rows, strict=True
        ):
            destination.extend(values)
    data_source = source(
        "凍結した実験の保存指標",
        evidence_paths,
        (
            "評価範囲内のみ。2D incoming edgeは終点が範囲内で前方文脈からの入口を含む。"
            "3D stepは両端が範囲内の隣接native frameのみ。魚×frame、観測step、検出rowを"
            "別の分母として保持する。観測ありID保留はfinite(xyz)かつidentity_reviewの数。"
            "identity_review全体は欠測も含む。過去方式には同じ保留ラベルがなくnull。"
            "小さい移動や保留はID正解の証明ではない。"
        ),
        [
            "report-summary",
            "tracking-table",
            "geometry-table",
            "all-cases",
            "female-table",
            "metric-fragments",
            "metric-edges",
            "metric-jumps",
            "metric-missing",
            "metric-held",
            "metric-coverage",
        ],
    )

    def query(rows, src):
        return dict(rows=rows, source=src)

    queries = {
        "human_reviews": query(
            review_rows,
            source(
                "過去動画の人手レビュー原文8件",
                [REVIEWS],
                "原文を保存。frameは目印、左右パネルと正確な境界は未指定。新方式の正解判定へ転用しない。",
                ["review-cases", "review-notes"],
            ),
        ),
        "methods": query(
            methods,
            source(
                "実験仕様と凍結設定",
                [Path(__file__).with_name("README.md"), protocol_path],
                "P/N/Fは2Dの診断比較。新3Dの入力はFのみ。過去の時間ID・3D軌跡・人手注釈を推定に使わず、比較にのみ使う。",
                ["methods", "limitations"],
            ),
        ),
        "tracking": query(tracking, data_source),
        "geometry": query(geometry, data_source),
        "female_intervals": query(intervals, data_source),
        "bridges": query(
            bridges,
            source(
                "2D断片の別層の接続提案",
                paths,
                (
                    "前後332framesの文脈を含む候補ログ。acceptedは別層の2D補完で、"
                    "初期clean edgeと3D観測は変更しない。重複場面を独立事象として合計しない。"
                ),
                ["join-table", "join-summary"],
            ),
        ),
        "flow": {
            **query(
                flows,
                source(
                    "画像フローの入力署名・妥当性診断",
                    paths,
                    (
                        "半解像度Gaussian pyrDownのLK。"
                        "前後roundtripとpatch error等を固定条件で検証。"
                        "IDを使わず native ±1frameを処理。動画全体SHAではなく"
                        "size/mtime/先頭末尾サンプルSHA。"
                    ),
                    ["flow-provenance"],
                ),
            ),
            "payloadColumns": ["metadata"],
        },
    }
    assets = app / "src/content/assets"
    imported, mappings = [], []
    media_paths = []
    for query_id, manifest_path in (
        (
            "media",
            args.media_manifest.resolve()
            if args.media_manifest
            else analysis / "review/manifest.json",
        ),
        ("event_figures", analysis / "event_figures/manifest.json"),
    ):
        rows = []
        if manifest_path.exists():
            manifest = read(manifest_path)
            entries = list(manifest.get("entries", []))
            for item in manifest.get("assets", []):
                for name, kind in (("video", "video"), ("still", "image")):
                    if item.get(name):
                        entries.append(
                            {
                                **item,
                                "kind": kind,
                                "path": item[name],
                                "sha256": item[name + "_sha256"],
                            }
                        )
            for entry in entries:
                p = Path(entry["path"])
                if not p.is_absolute():
                    p = manifest_path.parent / p
                digest = sha256(p)
                if digest != entry["sha256"]:
                    raise ValueError(f"Media SHA mismatch: {p}")
                assets.mkdir(parents=True, exist_ok=True)
                name = p.stem + "_" + digest[:12] + p.suffix
                shutil.copy2(p, assets / name)
                key = f"m{len(imported)}"
                imported.append(f'import {key} from "../assets/{name}";')
                mappings.append(f"  {json.dumps(rel(p))}: {key}")
                rows.append({**entry, "path": rel(p), "asset_key": rel(p)})
                media_paths.append(p)
            if query_id == "event_figures":
                queries["review_windows"] = query(
                    manifest.get("window_metrics", []),
                    source(
                        "8場面・166 native frame内の診断",
                        [manifest_path],
                        "clip範囲はcenter−83:center+83。移動は範囲内両端有限の隣接胸びれ3D。保留と欠測を別々に数える。旧出力に新方式の保留ラベルはない。",
                        ["window-table"],
                    ),
                )
        queries[query_id] = query(
            rows,
            source(
                "原画像・保存座標から作成した可視化",
                [manifest_path],
                (
                    "動画の全3視点・全3匹を表示。局所は166 native framesを30fpsで再生、"
                    "主20秒overviewは元時刻に従う間引き。補間なし。"
                    "図中Xは3Dの投影予測、観測点と区別。"
                ),
                ["review-media", "overview-video", "event-plot"],
            ),
        )
    stable_id = read(app / "src/data.json").get("id") if app.exists() else None
    complete = args.complete and len(cases) == len(CASES)
    if args.complete and not complete:
        raise ValueError("Cannot complete a report with missing experiment cases")
    snapshot = dict(
        surface="report",
        title="短い2D断片から3Dをつなぐ：観測と保留を分けて比較",
        generatedAt=datetime.now(UTC).isoformat(),
        status="reviewed" if complete else "partial",
        buildStatus="complete" if complete else "updating",
        filters=[],
        report={
            "asOf": max(
                datetime.fromtimestamp(p.stat().st_mtime, UTC).date().isoformat()
                for p in [REVIEWS, *paths]
            )
        },
        experiment=dict(
            id="20251225_K_1_2_FCMC",
            fps=166,
            status="complete" if complete else "running",
            analysis_directory=rel(analysis),
        ),
        cases=[
            dict(
                id=c["name"],
                label=f"{c['name']} · f{c['start']}–{c['stop'] - 1}",
                start=c["start"],
                stop=c["stop"],
            )
            for c in cases
        ],
        queries=queries,
        summaryMarkdown=(
            "## 比較実験を実行中\n\n保存済みの結果から順次更新します。未測定を0として扱いません。"
        ),
        methodNotes=(
            "P/N/Fは2Dの比較です。3DにはF断片のみを入力し、屈折カメラ幾何・3D移動・"
            "独立視点のclean断片一致から観測を選びます。別層の補完Jは初期断片Fを"
            "保持したまま記録し、欠測画素の観測化や3D座標の修正には使いません。"
        ),
        limitationsMarkdown=(
            "## 結果の読み方\n\n"
            "- 3Dのジャンプ減少と同時に、欠測・保留・観測stepの分母を確認します。"
            "保留の増加による滑らかさをID精度の改善とは呼びません。\n"
            "- 人手レビュー8件は過去の動画に対する原文です。新方式のID修正は未検証で、"
            "オス同士を含む全3匹の正解列はありません。\n"
            "- 同じ3匹の1群です。別20秒も前回調べた区間で、独立した新しいholdoutではありません。"
            "female端点注釈の分母は区間ごとに保持します。\n"
            "- 検出行は全て2Dに残します。検出内のノード混在は再訓練・再分配せず、"
            "矛盾する束を棄却する設計です。"
        ),
    )
    if stable_id:
        snapshot["id"] = stable_id
    primary = next((c for c in cases if c["name"] == "cg20s"), None)
    if primary:
        comparisons = primary["comparison"]
        old, new = comparisons["saved_C34"]["geometry"], comparisons["new"]["geometry"]
        observed_held = next(
            r["measured_and_review_fish_frames"]
            for r in geometry
            if r["case"] == "cg20s" and r["condition_id"] == "new"
        )
        female = comparisons["new"]["female"].get("interval_counts", {})
        snapshot["summaryMarkdown"] = (
            "## 主20秒の結果\n\n"
            "保存C34 → 新F→3Dで、10 mm超の移動は "
            f"**{old['steps_over_10mm']} → {new['steps_over_10mm']}**、欠測は "
            f"**{old['missing_fish_frames']} → {new['missing_fish_frames']} / "
            f"{new['possible_fish_frames']} fish-frames**。新方式の観測ありID保留は "
            f"**{observed_held} fish-frames** です（欠測と分けて表示）。\n\n"
            f"新方式のfemale端点は、維持 **{female.get('preserved', '未測定')}**・"
            f"変化 **{female.get('swap', '未測定')}**・"
            f"判定不能 **{female.get('unscored', '未測定')}**。"
            "少ない移動だけでは、IDスワップ減少を確認できません。"
        )
    narrative_path = base / "report_inputs/final_narrative.json"
    if narrative_path.exists():
        narrative = read(narrative_path)
        for key in (
            "title",
            "summaryMarkdown",
            "methodNotes",
            "limitationsMarkdown",
            "resultNotes",
        ):
            if key in narrative:
                snapshot[key] = narrative[key]
        snapshot["queries"]["narrative"] = query(
            [narrative],
            source(
                "定量監査後の解釈",
                [narrative_path],
                "別途保存した最終解釈。数値は結果queriesを参照する。",
                ["report-summary"],
            ),
        )
    inputs = base / "report_inputs"
    inputs.mkdir(exist_ok=True)
    snapshot_path = inputs / "reviewed_snapshot.json"
    with provenance("clean_tracklets_report_snapshot", inputs, name="snapshot") as rec:
        for p in [REVIEWS, *evidence_paths, *([narrative_path] if narrative_path.exists() else [])]:
            rec.input(rel(p), p, sha256=sha256(p))
        rec.input("adapter", Path(__file__), sha256=sha256(Path(__file__)))
        write(snapshot_path, snapshot)
        if not (app / "AGENTS.md").exists():
            if app.exists():
                # Only media was staged: defer staging until after the preparer.
                staged = inputs / "staged_assets"
                staged.mkdir(exist_ok=True)
                for p in assets.glob("*"):
                    shutil.move(str(p), staged / p.name)
                assets.rmdir()
                assets.parent.rmdir()
                assets.parent.parent.rmdir()
                app.rmdir()
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
            assets.mkdir(exist_ok=True)
            staged = inputs / "staged_assets"
            if staged.exists():
                for p in staged.glob("*"):
                    shutil.copy2(p, assets / p.name)
            snapshot = read(app / "src/data.json")
            write(snapshot_path, snapshot)
        else:
            write(app / "src/data.json", snapshot)
        (app / "src/content/report/media.js").write_text(
            "\n".join(imported) + "\nexport const media = {\n" + ",\n".join(mappings) + "\n};\n",
            encoding="utf-8",
            newline="\n",
        )
        rec.output("snapshot", snapshot_path, sha256=sha256(snapshot_path))
        rec.output("app_data", app / "src/data.json", sha256=sha256(app / "src/data.json"))
    print(
        json.dumps(
            dict(
                cases=len(cases), assets=len(media_paths), snapshot=rel(snapshot_path), app=rel(app)
            )
        )
    )


if __name__ == "__main__":
    main()

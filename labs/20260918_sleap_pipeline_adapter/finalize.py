"""Seal a completed adapter comparison and write its result, without promotion."""
# ruff: noqa: RUF001

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "work/sleap_pipeline_adapter_v1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    metrics, protocol, audit = read("metrics.json"), read("protocol.json"), read("audit.json")
    assert metrics["complete"] and len(metrics["records"]) == 540
    assert audit["complete"] and len(audit["records"]) == 360 and len(audit["shuffled"]) == 3
    for filename in (
        "verification.json",
        "source_audit.json",
        "video_source_audit.json",
        "recovery_check.json",
        "browser_receipt.json",
    ):
        assert read(filename)["status"] == "passed", filename
    assert read("browser_receipt.json")["html_sha256"] == sha(OUT / "report.html")
    for name, digest in protocol["frozen_file_sha256"].items():
        assert sha(HERE / name) == digest, name
    numerical = read("numerical_manifest.json")["sha256"]
    for name, digest in numerical.items():
        assert sha(OUT / name) == digest, name
    for source in read("source_audit.json")["files"]:
        assert sha(Path(source["installed_path"])) == source["installed_sha256"]
    report_manifest = read("report_manifest.json")
    for name, digest in report_manifest["inputs"].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in report_manifest["outputs"].items():
        assert sha(OUT / name) == digest, name
    main_records = [r for r in metrics["records"] if r["case"] == "cg20s"]
    main_totals = {}
    for method in protocol["conditions"]:
        selected = [r for r in main_records if r["method"] == method]
        main_totals[method] = {
            key: sum(r[key] for r in selected)
            for key in (
                "core_rows",
                "contiguous_runs",
                "adjacent_connections",
                "eligible_observations",
                "gap_connections",
            )
        }
    methods = (
        "S1",
        "S2",
        "SP",
        "P0",
        "P1",
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
    lines = [
        "# SLEAPを土台にした2D追跡の設計・パラメータ比較",
        "",
        "**Verdict: parked (not promoted).** "
        "改良版の実装、全設定実行、互換性監査、比較HTMLは完了。",
        "P1の自己競合による持続分断を解消した。実魚の個体対応の正解が不足するため、",
        "誤接続を増やしていないという採用条件は未判定。期限は2026-10-18。",
        "",
        "[HTMLレポート](../../work/sleap_pipeline_adapter_v1/report.html)",
        "",
        "## 主20秒の結果（三視点合算）",
        "",
        "全条件で29,386行を保持。隣接接続率の分母は29,377観測であり、ID正解率ではない。",
        "",
        "| 条件 | 連続区間数 | 隣接接続率 |",
        "|---|---:|---:|",
    ]
    for method in methods:
        values = main_totals[method]
        rate = values["adjacent_connections"] / values["eligible_observations"]
        lines.append(f"| {method} | {values['contiguous_runs']:,} | {rate:.2%} |")
    lines.extend(
        [
            "",
            "SPはS2から口を外し胸びれだけで固定面積OKSを計算する通常SLEAP。",
            "G0はSPに履歴期限・未接続の選択肢・確定判定を加えた設計の比較で、単一変更ではない。",
            "9つのG設定は事前指定の一項目変更。最良設定の選抜・場面別調整は行っていない。",
            "G/SLEAPは過去履歴、P0/P1は前後履歴を使う。入力の前後332フレームは共通。",
            "",
            "## 設計と調節で確認できたこと",
            "",
            "- SLEAPの実Tracker、トラックごとの履歴、平均OKS、Hungarian割当を実行する。",
            "- 暫定IDと確定断片を分離。候補差不足で断片を切っても暫定履歴を増殖させない。",
            "- P1で105断片になった単独魚の120フレーム・一度の9px変化は、G0で1断片。",
            "  厳しい候補差で一時保留した例と、80px変化で新規履歴を作る例も継続へ復帰。",
            "- 主20秒のG0非接続584件は候補差284、dummy162、gap候補133、非相互5。",
            "  候補差不足273件は、dummy0.05＋margin0.05の実効確定下限0.10を下回る。",
            "- side1・f75286は直前変位4.64px・他検出317pxでも、履歴5→9で平均OKSが",
            "  0.166→0.092に低下。過去の位置との距離から独立再計算して一致を確認した。",
            "- 履歴3と候補差0.02は連続性が高い比較候補。ただし追加接続の正しさは未評価。",
            "  両者を同時に適用した条件は実行していない。",
            "",
            "## パイプラインとの互換性と限界",
            "",
            "全検出は非負の確定断片ID（単独断片を含む）を持つ。P欠測の暫定IDは-1でもよい。",
            "元points、欠損mask、検出score、global_rowは変更しない。確定G断片は隣接のみで、",
            "欠測越しの暫定対応は再接続候補に分離する。反射は未分類、実魚のID名は付けない。",
            "",
            "既存C25 compact_layoutへ実際に渡して全行保持・source offsetの往復を検証した。",
            "これは3D候補の入力互換性までで、3D幾何や実魚IDの改善を検証したものではない。",
            "現C25は最大12検出で超過を明示エラーとし、各3D状態には最大3枠という別の制約がある。",
            "C24の戻り行順対応、C28の非ゼロ開始部分区間のoffsetにも既存の注意点が残る。",
            "今回のNPZを全録画C27へ無条件に代入してはならず、明示的なglobal_row対応を用いる。",
            "",
            "## 検証と再現",
            "",
            "- 12既知区間×3視点×10新条件=360配列、既存5条件を加え540比較記録。",
            "- 54テスト、Ruff、全入力配列・一対一・G欠測越し確定0・3視点の実行行順変更を確認。",
            "- SLEAP公式v0.3.1関連4ファイルと実環境のSHA-256一致を確認。",
            "- 元映像3本のSHA-256、36フレームビュー用データ、36グラフとHTMLを検証。",
            "- ブラウザーで元映像の再生・フレーム移動・方式切替・0/5検出の表示を確認。",
            "- 全区間は既に調査済みで新規holdoutではない。同一の3匹、独立群数1。",
            "",
            "```powershell",
            ".venv/Scripts/python.exe labs/20260918_sleap_pipeline_adapter/run.py "
            "--init --output <new-output>",
            ".venv/Scripts/python.exe labs/20260918_sleap_pipeline_adapter/run.py "
            "--view top_cam --output <new-output>",
            "# side1_cam / side2_camも同様に実行し、最後に --summarize",
            "```",
            "",
            "既存outputは凍結済み。run.pyはコードSHA変化を拒否する。",
            "audit.py/build_report.py/diagnose.py/recovery_check.pyは今回の固定outputを検証・描画する。",
            "設定・検証・数値・根拠JSONと各provenanceはHTMLから参照できる。",
        ]
    )
    with provenance("sleap_adapter_final_delivery", OUT) as record:
        result_path = HERE / "RESULT.md"
        result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        links = re.findall(
            r'(?:href|src)="([^"]+)"', (OUT / "report.html").read_text(encoding="utf-8")
        )
        static_links = [link for link in links if not any(x in link for x in (":", "${", "#"))]
        for link in static_links:
            assert (OUT / link).resolve().exists(), link
        receipt = {
            "status": "passed",
            "numerical_files_verified": len(numerical),
            "new_arrays_verified": 360,
            "comparison_records": 540,
            "row_shuffle_views": 3,
            "viewer_cases": 36,
            "plots": 36,
            "main20s": main_totals,
            "decision": "parked_not_promoted",
            "identity_accuracy_claim": False,
            "html_static_links_checked": static_links,
        }
        write(OUT / "final_receipt.json", receipt)
        files = {}
        for path in OUT.rglob("*"):
            if not path.is_file() or any(part.startswith("pytest_") for part in path.parts):
                continue
            if path.name in (
                "delivery_manifest.json",
                "sleap_adapter_final_delivery.provenance.json",
            ):
                continue
            files[str(path.relative_to(ROOT)).replace("\\", "/")] = sha(path)
        for path in HERE.iterdir():
            if path.is_file():
                files[str(path.relative_to(ROOT)).replace("\\", "/")] = sha(path)
        write(OUT / "delivery_manifest.json", {"sha256": files})
        record.input("protocol", OUT / "protocol.json", sha256=sha(OUT / "protocol.json"))
        record.output("result", result_path, sha256=sha(result_path))
        record.output("receipt", OUT / "final_receipt.json", sha256=sha(OUT / "final_receipt.json"))
        record.output(
            "manifest", OUT / "delivery_manifest.json", sha256=sha(OUT / "delivery_manifest.json")
        )
    print("Verified and sealed the completed SLEAP adapter comparison")


if __name__ == "__main__":
    main()

"""Seal verified stage A-C artifacts and write the scientific verdict."""

# ruff: noqa: RUF001

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "work/variable_count_tracklets_v1"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src") and value:
                self.references.append(value)


def main():
    summary, audit = read("summary.json"), read("audit.json")
    media, report = read("review/manifest.json"), read("report_manifest.json")
    tests, browser = read("test_receipt.json"), read("browser_receipt.json")
    scope = read("scientific_scope_audit.json")
    assert scope["complete"] and not scope["parameters_retuned"]
    assert audit["status"] == "passed" and audit["method_case_view_arrays"] == 288
    assert tests["status"] == "passed" and browser["status"] == "passed"
    assert len(media["assets"]) == 42 and media["all_videos_fully_decoded"]
    assert report["all_method_view_payloads"] == 36
    assert report["report_sha256"] == sha256(OUT / "report.html")
    assert browser["html_sha256"] == sha256(OUT / "report.html")
    for asset in media["assets"]:
        for kind in ("video", "before", "center", "after"):
            assert sha256(OUT / "review" / asset[kind]) == asset[kind + "_sha256"]
    for payload in report["payloads"].values():
        for kind in ("script", "plot"):
            assert sha256(OUT / "report_data" / payload[kind]) == payload[kind + "_sha256"]
    records = summary["records"]
    main_rows = [r for r in records if r["case"] == "cg20s"]
    table, totals = [], {}
    for method in summary["protocol"]["conditions"]:
        rows = [r for r in main_rows if r["method"] == method]
        total = {
            k: sum(r[k] for r in rows)
            for k in (
                "core_rows",
                "contiguous_runs",
                "adjacent_connections",
                "eligible_observations",
                "unconnected_observations",
                "gap_connections",
            )
        }
        totals[method] = total
        assert total["core_rows"] == 29386 and total["eligible_observations"] == 29377
        connection = total["adjacent_connections"] / total["eligible_observations"]
        table.append(
            f"| {method} | {total['contiguous_runs']:,} | {connection:.2%} | "
            f"{1 - connection:.2%} | {total['gap_connections']} |"
        )
    difference = totals["P1"]["contiguous_runs"] - totals["P0"]["contiguous_runs"]
    decoded = sum(a["full_decode"]["decoded_frames"] for a in media["assets"])
    result = f"""# 工程A-Cの実行結果: 可変検出数2Dトラックレット

**Verdict: parked (not promoted).** 工程A-Cの実装・全条件実行・HTMLと比較素材・検証は完了。
P1は主20秒でP0より{difference:,}区間多く分断し、採用条件を満たさなかった。
設定は結果に合わせて変更していない。実魚IDの正解率改善は主張しない。

## 主20秒 [75270,78590)、166 fps、同じ3匹

全条件で29,386行を保持。元points・欠損・検出score・global_rowがC01と一致。
接続率の分母は29,377観測（core先頭フレームを除く）。
これは隣接接続の割合で、正しい個体継続の率ではない。

| 条件 | 連続区間数 | 隣接接続率 | 非接続率 | 欠測越し同一ラベル |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

隣接接続は両端がcore内。欠測越し同一ラベルは終点がcore内で、起点が前方contextに
あるものも含む。protocolの一般的なmetric_edge_scope表記は隣接接続についてのみ正しい。
厳密に両端がcore内のgap数は`scientific_scope_audit.json`を参照。
P0/P1は隣接のみを確定し、P1の欠測越し関係は候補表のまま統合しない。

S1はオブジェクト対応で戻り行の順序問題を分離。S2はS1からOKSの尺度だけを
top=14400 / side=10000 px²へ変更。両者とも局所履歴の現在環境での再実行であり、
保存C27の以前の全録画処理を同一再現した結果ではない。
P0は元ソースのバイト一致と全33既知case-viewでの結果一致を確認。

P1の開いた端点による競合評価は大量の保留・順逆不一致を伴った。
M/Tの補助支持は分断を減らしたが、P0よりも多い。近接判定、ノード欠損、検出数の
層別とイベント根拠をHTML/CSVに保存。原因を魚の真のID誤りや反射と断定しない。
実際の候補競合の機序は科学的監査の記録を参照。

## 対象と未判定項目

計画の11ケース×3視点に、事前にmetadataだけで選定した[332,3652)の20秒を追加し、
設定固定後に実行。既知の注釈位置・レビュー区間・長区間実験・背景画像位置をcontext
ごと除外した選定根拠とSHAは`selection_audit.json`にある。
同一録画・同一群であり、独立群への一般化を示す試験ではない。
全録画の検出・保存C27はもともと存在し、「過去に誰も見ていない」という証明でもない。

13アンカー×3視点の局所17フレーム集計とレビュー質問を`case_findings.json`に保存。
f76791/f323819/f77415のno-swapレビューはカメラ・パネル・各2D行の対応が限定され、
全個体2D正解へ転用していない。新規の人手ラベルは作成していない。
確認済み個体正解の分母は0で、正しい継続率・誤接続率は未判定。
反射は全行未判定。3Dの再接続・割当修正は工程Dとして残る。

## 成果物と検証

- [HTMLレポート](../../work/variable_count_tracklets_v1/report.html)
- 42 MP4（全体20秒3本、イベント低速39本）、126 PNG、36 native-frame SVG。
- 全動画{decoded:,}出力フレームをデコード。全体動画は30 fps標本化、
  イベント動画は全166 native frameを30 fpsで再生。HTMLでは6条件を切替可能。
- 288条件/case/view配列、21の実検出数回帰、6回の実入力全行シャッフル、
  416既存ファイル不変を独立監査。
- 合成追跡・SLEAP制御・描画・lab構造テストとRuffを通過。詳細は`test_receipt.json`。
- ブラウザーの切替・再生・参照リンクを確認。詳細は`browser_receipt.json`。
- C01・確定接続・保留根拠・別配列予測・再接続候補・node品質・全設定・
  全層別集計・provenanceとSHAを保存。

採用を保留する理由は、実データでの過剰分断と、完全な個体正解注釈がないこと。
次の設計変更は開いた端点の競合表現を検討し、今回の固定条件とは別版で比較する。
このlabの期限は2026-10-17。本番・以前の結果は変更していない。
"""
    (HERE / "RESULT.md").write_text(result, encoding="utf-8")
    parser = References()
    parser.feed((OUT / "report.html").read_text(encoding="utf-8"))
    checked = []
    for reference in parser.references:
        parsed = urlsplit(reference)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        path = (OUT / unquote(parsed.path)).resolve()
        assert path.exists(), f"Broken report reference: {reference}"
        checked.append(reference)
    receipt = dict(
        status="passed",
        html_static_references_checked=sorted(set(checked)),
        numerical_arrays=288,
        video_count=42,
        still_count=126,
        plot_count=36,
        decoded_video_frames=decoded,
        main20s_totals=totals,
        audit_status=audit["status"],
        tests=tests,
        browser=browser,
        scientific_scope_audit_present=bool(scope),
        decision="parked_not_promoted",
        stages_completed=["A", "B", "C"],
    )
    with provenance("variable_count_final_delivery", OUT) as rec:
        for name in (
            "protocol.json",
            "summary.json",
            "audit.json",
            "report_manifest.json",
            "scientific_scope_audit.json",
            "test_receipt.json",
            "browser_receipt.json",
        ):
            rec.input(name, OUT / name, sha256=sha256(OUT / name))
        rec.output("result", HERE / "RESULT.md", sha256=sha256(HERE / "RESULT.md"))
        target = OUT / "final_receipt.json"
        target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        rec.output("receipt", target, sha256=sha256(target))
        manifest = {
            p.relative_to(OUT).as_posix(): sha256(p)
            for p in OUT.rglob("*")
            if p.is_file()
            and p.name
            not in ("delivery_manifest.json", "variable_count_final_delivery.provenance.json")
        }
        manifest_path = OUT / "delivery_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        rec.output("manifest", manifest_path, sha256=sha256(manifest_path))
    print("Stage A-C delivery verified and sealed")


if __name__ == "__main__":
    main()

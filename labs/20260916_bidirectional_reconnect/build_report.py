"""Build an offline, source-linked review of the frozen development experiment."""

# Japanese prose and embedded HTML/CSS retain their natural punctuation and lines.
# ruff: noqa: E501, RUF001

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance


def esc(value):
    return html.escape(str(value), quote=True)


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{esc(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def link(path, label):
    return f'<a href="{esc(path)}">{esc(label)}</a>'


def angle(value):
    return "方向不定" if value is None else f"{value:.1f}°"


def validate_render_inputs(root, cases):
    """Retained v1 renders are usable only if their core inputs are unchanged."""
    checked = []
    for case in cases:
        name = case["case"]["name"]
        old = root / "analysis" / name / "poses.npz"
        new = root / "analysis_v2" / name / "poses.npz"
        with np.load(old, allow_pickle=False) as a, np.load(new, allow_pickle=False) as b:
            for key in ("frames", "pose", "xyz", "source", "observed", "uv", "slot_track"):
                if not np.array_equal(a[key][a["core"]], b[key][b["core"]], equal_nan=True):
                    raise ValueError(f"Core render input changed: {name}/{key}")
            first, stop = case["case"]["start"], case["case"]["stop"]
            held_a = (a["diag_frames"] >= first) & (a["diag_frames"] < stop)
            held_b = (b["diag_frames"] >= first) & (b["diag_frames"] < stop)
            for key in (
                "frames",
                "step_mm",
                "past_angle_deg",
                "future_angle_deg",
                "distance",
                "forward",
                "bidirectional",
            ):
                if not np.array_equal(
                    a[f"diag_{key}"][held_a], b[f"diag_{key}"][held_b], equal_nan=True
                ):
                    raise ValueError(f"Primary detection changed: {name}/{key}")
        checked.append(dict(case=name, old_sha256=sha256(old), new_sha256=sha256(new)))
    return dict(core_inputs_and_primary_diagnostics_exactly_equal=True, cases=checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("work/bidirectional_reconnect_v1"))
    args = parser.parse_args()
    root = args.root
    target = root / "index.html"
    if target.exists():
        raise FileExistsError(target)
    analysis = root / "analysis_v2"
    summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8"))
    revision_validation = validate_render_inputs(root, summary["cases"])
    manifest = json.loads((root / "review/manifest.json").read_text(encoding="utf-8"))
    main_case = json.loads((analysis / "cg20s/analysis.json").read_text(encoding="utf-8"))
    parts = []
    parts.append("""<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>メダカ追跡：前後方向のjump検出テスト</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#111b24;color:#e7edf3;font:16px/1.75 system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:28px}h1{font-size:28px}h2{margin-top:40px;font-size:22px}h3{font-size:18px}
a{color:#84d5ee}video{display:block;width:100%;max-height:72vh;background:#000;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #3a4854;vertical-align:top}th{background:#1c2e3e}.scroll{overflow-x:auto}.lead{padding:20px;background:#203445;border-left:4px solid #71d2e4}.muted{color:#b9c7d3;font-size:14px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}details{margin:12px 0;padding:12px;background:#192732;border-radius:6px}summary{cursor:pointer}img{display:block;width:100%;height:auto}button,select{font:inherit;border:1px solid #7895aa;border-radius:4px;background:#203445;color:#eef8ff;padding:5px 10px;cursor:pointer}code{overflow-wrap:anywhere}.pills a{display:inline-block;margin:5px 15px 5px 0}button:focus-visible,a:focus-visible{outline:3px solid #ffd46a}.warn{color:#ffd482}
</style><main><h1>前後方向のjump検出：20秒と既知の失敗例</h1>
<p class="muted">2026-09-16 · C34 λ0.767 · 166 fps · 現在の追跡データを固定した開発テスト</p>
<div class="lead"><strong>前後方向で追加の異常候補を検出。ただし、自動修正の採用は0件。</strong>
<p>20秒区間の候補は距離のみ21件、順方向96件、前後方向110件でした。追加14件はすべて元画像の比較シートを作成しました。
候補数は「個体×隣接フレーム」の件数であり、スワップ数や正解数ではありません。既知の12場面、過去の誤修正3区間、対照3区間も評価しています。</p></div>
<nav class="pills"><a href="#movie">20秒動画</a><a href="../aquarium_cg/c34_75270_78590/aquarium.mp4">CG単独</a><a href="#slow">スロー</a><a href="#added">追加14件</a><a href="#known">既知12場面</a><a href="#scores">全区間の数値</a><a href="#sensitivity">閾値比較</a></nav>
<h2 id="movie">3視点の元動画と3D CG</h2>
<p>各時点の追跡座標を簡易魚に反映しています。骨格の丸は口、十字は胸びれ、末端は尾です。色と番号は現在の追跡slotです。
20秒の一覧動画は30 fpsへ間引いています。1フレームの異常の確認には下のスローと静止画を使ってください。</p>
<video id="mainvideo" controls preload="metadata" src="review/cg20s_raw_and_measured_cg.mp4"></video>
<p class="muted">f75270–78589（20秒）。distance=距離、forward=順方向、bidirectional-only=逆方向でのみ追加。ラベルは見つけやすいよう候補の前後17 nativeフレームにも表示します。CGの座標補間・平滑化はありません。</p>
<h2 id="slow">全166 fpsフレームを保持したスロー</h2>
<p>元の2秒332フレームを30 fpsで再生するため、各動画は約11.07秒、約5.53倍のスローです。</p><div class="grid">
<section><h3>f76235付近：大きな位置の飛び</h3><video controls preload="metadata" src="review/native_slow_f76235.mp4"></video>
<p>密接した魚の間で口・胸びれと尾の対応が崩れる可能性があります。魚全体のID交換だけで説明できるかを確認する場面です。</p></section>
<section><h3>f76930付近：両方向とも交換を選好するが保留</h3><video controls preload="metadata" src="review/native_slow_f76930.mp4"></video>
<p>S0↔S1の対応は両方向で最小誤差になりますが、次点との差と個体ごとの誤差が事前条件を満たしません。修正を採用していません。</p></section></div>""")

    group = next(
        g for g in main_case["episodes"] if g["start_frame"] <= 76930 < g["stop_frame_exclusive"]
    )
    audit = group["persistent_audit"]
    comparison = [r for r in audit["scores"] if r["permutation"] in ([0, 1, 2], [1, 0, 2])]
    parts.append(
        table(
            [
                "f76930の対応",
                "順方向・3匹の誤差合計",
                "逆方向・3匹の誤差合計",
                "個体の最大誤差（順／逆）",
            ],
            [
                [
                    esc(r["permutation"]),
                    f"{r['forward_sum_mm']:.2f} mm",
                    f"{r['backward_sum_mm']:.2f} mm",
                    f"{r['forward_max_fish_mm']:.2f} / {r['backward_max_fish_mm']:.2f} mm",
                ]
                for r in comparison
            ],
        )
    )
    parts.append(
        f'<p class="muted">採用の必要条件：次点との差が両方向とも5 mm以上、各個体の予測誤差が両方向とも3 mm以下。'
        f"実際の差は{audit['forward_margin_mm']:.2f} / {audit['backward_margin_mm']:.2f} mm。"
        "この検査は前後のtracklet対応の検査であり、区間内の正確な交換時刻を決定するものではありません。</p>"
    )

    parts.append(
        '<h2 id="added">逆方向で追加した14件</h2><p>各画像は元画像と3Dノードの投影を対にしています。クリックで元サイズを開けます。'
        "アシスタントによる画像所見は独立した個体ID正解ラベルには扱っていません。"
        "14件中6件は過去の参照速度が低い等の理由で順方向の角度が未定義で、未来側からのみ判定しています。</p>"
    )
    for item in manifest["contact_sheets"]:
        if item["role"] != "bidirectional_only":
            continue
        frame = item["center_frame"]
        label = f"f{frame} · S{item['diagnostic_slot']} · {item['step_mm']:.2f} mm · 角度 {angle(item['past_angle_deg'])} / {angle(item['future_angle_deg'])}（過去／未来）"
        path = "review/" + item["path"]
        parts.append(
            f'<details><summary>{esc(label)}</summary><p><button data-seek="{max(0, (frame - 75270) / 166 - 0.5):.4f}">20秒動画のこの付近へ</button> '
            f'{link(path, "比較画像を開く")}</p><a href="{esc(path)}"><img loading="lazy" alt="{esc(label)} の3視点比較" src="{esc(path)}"></a></details>'
        )
    note_files = sorted((root / "review").glob("*visual_notes*.json"))
    parts.append(
        "<p>画像所見："
        + " · ".join(link(p.relative_to(root).as_posix(), p.stem) for p in note_files)
        + "</p>"
    )

    parts.append(
        '<h2 id="known">既知の12場面</h2><p>元の注記は以前のC30の結果に対するものです。C34ですでに直っている場面も含みます。'
        "そのため「注記近傍の検出」は再現率として扱いません。過去のslot名もそのまま現在の個体正解には使いません。</p>"
    )
    for item in manifest["contact_sheets"]:
        if item["role"] != "historical_failure_neighborhood":
            continue
        path = "review/" + item["path"]
        label = f"f{item['center_frame']} · 過去の注記: {item['historical_note']['kind']}"
        flags = item["current_c34_detection"]
        parts.append(
            f"<details><summary>{esc(label)}</summary><p>現在の注記近傍に候補："
            f"距離 {'あり' if flags['distance'] else 'なし'} / 順方向 {'あり' if flags['forward'] else 'なし'} / 前後方向 {'あり' if flags['bidirectional'] else 'なし'}。</p>"
            f'<p>{esc(item["historical_note"]["annotator_note"])}</p><a href="{esc(path)}"><img loading="lazy" alt="{esc(label)} の3視点比較" src="{esc(path)}"></a></details>'
        )

    parts.append(
        '<h2 id="scores">全19区間の結果</h2><p>各手法のセルは「個体×隣接フレームの候補数 / 近接候補をまとめた区間数」です。'
        "f76930の既知例は20秒区間と重なるため、行を単純合算した値を独立な検出数とは解釈しません。</p>"
    )
    rows = []
    for case in summary["cases"]:
        info = case["case"]
        row = [
            link(f"analysis_v2/{info['name']}/analysis.json", info["name"]),
            f"{info['start']}–{info['stop'] - 1}",
        ]
        row += [
            f"{case['methods'][m]['slot_steps']} / {case['methods'][m]['episodes']}"
            for m in ("distance", "forward", "bidirectional")
        ]
        row += [
            str(case["bidirectional_extra_slot_steps"]),
            str(len(case["candidate_geometry_proposals"])),
            str(case["accepted_identity_corrections"]),
        ]
        rows.append(row)
    parts.append(
        table(
            [
                "区間（詳細JSON）",
                "nativeフレーム",
                "距離",
                "順方向",
                "前後方向",
                "逆方向の追加",
                "幾何条件通過",
                "修正採用",
            ],
            rows,
        )
    )
    parts.append("""<p>20秒の55候補区間の接続判定は、元の対応が最小誤差37、参照区間が不十分15、曖昧3でした。
これらは「正しいIDである」と証明する分類ではありません。1–3フレームの一時的交換も検査しましたが、事前の全条件を通過する提案はありませんでした。</p>
<h3>独立した雌の注釈で確認したID維持</h3>
<p><strong>修正前も採用後も、295維持 / 2スワップ / 2判定不能。</strong>
300注釈の隣接299区間を固定し、欠測による区間の削除・橋渡しをしていません。採用修正が0件のため値は同じです。
スワップがある注釈間はf76698–77285とf130760–138892。これは正確な交換フレームの注釈ではありません。
注釈で捉えられない往復交換などもあり、この数値だけで全個体・全時点の同一性は保証できません。</p>""")
    parts.append(link("analysis_v2/female_baseline.json", "固定注釈の全評価結果"))

    parts.append(
        '<h2 id="sensitivity">20秒区間での方向窓・角度の比較</h2><p>結果を見て最良の閾値を選び直していません。主結果は17フレーム、80°です。距離条件は全設定で21件です。</p>'
        "<p>主判定：1フレームの移動が4 mm超、または1 mm超かつ参照方向との角度が80°以上。"
        "参照方向は前後それぞれ17フレームの移動の中央値で、対象の前後2移動を除外します。"
        "参照速度0.05 mm/frame未満では角度を使いません。</p>"
    )
    parts.append(
        table(
            ["方向窓（166 fps）", "角度", "順方向の候補", "前後方向の候補", "前後方向の区間"],
            [
                [
                    str(s["window"]),
                    f"{s['angle_deg']}°",
                    str(s["forward"]["slot_steps"]),
                    str(s["bidirectional"]["slot_steps"]),
                    str(s["bidirectional"]["episodes"]),
                ]
                for s in main_case["sensitivity"]
            ],
        )
    )
    parts.append("""<h2>今回の判断と次の検証</h2>
<p>前後方向は、距離だけでは見逃す横方向の揺れを抽出する入口として使えます。ただし追加検出をスワップ検出の改善とはまだ呼べません。
元画像には近接時のノードの位置揺れ・対応混在が疑われる場面があり、個体丸ごとの6通りの交換だけでは対応できない可能性があります。</p>
<p>次の実装候補は、今回抽出した局所区間で3視点の検出対応を再検討し、口・胸びれ・尾のまとまりと再投影誤差を同時に評価するものです。
前後の安定区間を境界条件にし、ノード・視点の不整合と個体IDの交換を分けて評価します。独立した短区間のID注釈を用いて、既知例と未調整区間で改善と誤修正の両方を確認する必要があります。</p>
<h2>再現用の記録</h2>""")
    records = [
        ("analysis_v2/protocol.json", "事前条件と区間"),
        ("analysis_v2/summary.json", "数値の一覧"),
        ("analysis_v2/bidirectional_development.provenance.json", "解析の入力・コードの記録"),
        ("review/manifest.json", "動画・比較シートの記録"),
        ("review/bidirectional_raw_review.provenance.json", "動画生成の記録"),
    ]
    parts.append("<p>" + " · ".join(link(path, label) for path, label in records) + "</p>")
    parts.append(f'<p class="muted">C34 SHA256: <code>{esc(summary["baseline_sha256"])}</code></p>')
    parts.append("""</main><script>
document.querySelectorAll('[data-seek]').forEach(button=>button.addEventListener('click',()=>{
const video=document.getElementById('mainvideo');video.currentTime=Number(button.dataset.seek);video.pause();video.scrollIntoView({behavior:'smooth',block:'center'});
}));</script></html>""")
    document = "\n".join(parts)
    # Offline links must resolve; links inside this report never require fetch/CDNs.
    for path in re.findall(r'(?:href|src)="([^"]+)"', document):
        if path.startswith("#"):
            continue
        if not (root / html.unescape(path)).is_file():
            raise FileNotFoundError(path)
    with provenance("bidirectional_report", root) as rec:
        for path in [
            Path(__file__),
            analysis / "summary.json",
            analysis / "cg20s/analysis.json",
            root / "review/manifest.json",
            *note_files,
        ]:
            rec.input(path.name, path, sha256=sha256(path))
        target.write_text(document, encoding="utf-8")
        rec.output("offline_review", target, sha256=sha256(target))
        validation_path = root / "revision_validation.json"
        validation_path.write_text(json.dumps(revision_validation, indent=2), encoding="utf-8")
        rec.output("revision_validation", validation_path, sha256=sha256(validation_path))
        rec.param("local_links_checked", True)
        rec.param("browser_interaction_verified", False)
    print(target)


if __name__ == "__main__":
    main()

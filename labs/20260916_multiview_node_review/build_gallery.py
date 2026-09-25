"""Index the exported scientific figures and raw-pixel correspondence comparisons."""

# Japanese artifact text and embedded HTML keep natural punctuation/line lengths.
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


def esc(x):
    return html.escape(str(x), quote=True)


def number(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}"


def link(path, label):
    return f'<a href="{esc(path)}">{esc(label)}</a>'


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{esc(x)}</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("work/multiview_node_review_v1"))
    args = parser.parse_args()
    root = args.root
    target = root / "index.html"
    if target.exists():
        raise FileExistsError(target)
    sources = [root / "analysis/summary.json", root / "review/manifest.json"]
    report = json.loads(sources[0].read_text(encoding="utf-8"))
    manifest = json.loads(sources[1].read_text(encoding="utf-8"))
    entries = list(manifest["entries"])
    for filename in ("supplemental_manifest_v2.json", "supplemental_manifest_v3.json"):
        path = root / "review" / filename
        if path.exists():
            extra = json.loads(path.read_text(encoding="utf-8"))
            entries.extend(extra["entries"])
            sources.append(path)
    entries = list({entry["path"]: entry for entry in entries}.values())
    arrays, details, loss = {}, {}, {}
    for row in report["cases"]:
        name = row["case"]["name"]
        a_path = root / "analysis" / name / "evaluation.npz"
        j_path = a_path.with_suffix(".json")
        sources.extend([a_path, j_path])
        with np.load(a_path, allow_pickle=False) as a:
            arrays[name] = {k: a[k] for k in a.files}
        details[name] = json.loads(j_path.read_text(encoding="utf-8"))
        a = arrays[name]
        old = np.isfinite(a["original_uv"]).all(-1).sum(axis=2)
        new = np.isfinite(a["candidate_uv"]).all(-1).sum(axis=2)
        loss[name] = (new < old).any(axis=(1, 2))
    primary = arrays["cg20s"]
    priority = primary["supported"] & primary["core"] & ~loss["cg20s"]
    priority_frames = primary["frames"][priority].tolist()
    if len(priority_frames) != 6:
        raise ValueError("Primary no-view-loss audit differs from reviewed6frame result")
    parts = [
        """<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>メダカ：3視点と部位対応の再評価</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;color:#24333f;background:#f4f6f8;font:16px/1.8 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:28px}h1{font-size:28px}h2{font-size:23px;margin-top:38px}h3{font-size:18px}a{color:#145d8b}video{display:block;background:#111;width:100%;max-height:78vh}img{width:100%;height:auto;display:block}.lead{padding:20px;background:#e8f0f5;border-left:5px solid #28678c}.warn{padding:14px;background:#fff3d7;border-left:4px solid #b77400}.muted{color:#586c79;font-size:14px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{border-bottom:1px solid #cbd7de;padding:8px 12px;vertical-align:top;text-align:left}th{background:#e3eaf0}details{padding:12px;margin:12px 0;background:white;border:1px solid #d7e0e7;border-radius:5px}summary{cursor:pointer;font-weight:600}nav a{display:inline-block;margin:6px 18px 6px 0}select,input{font:inherit;padding:6px;border:1px solid #7892a2;border-radius:4px}video,img{border-radius:5px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}section{scroll-margin-top:20px}.pill{font-size:13px;padding:3px 8px;background:#e8edf0;border-radius:4px}.supportloss{font-weight:700;color:#895600}[hidden]{display:none!important}:focus-visible{outline:3px solid #bd7100}code{overflow-wrap:anywhere}
</style><main><h1>3視点の検出対応と、口・胸びれ・尾のまとまりを再評価</h1>
<p class="muted">2026-09-16 · C34 λ0.767 · native 166 fps · 元の追跡を保存した比較</p>
<div class="lead"><strong>20秒の55候補区間、計348フレームを再評価しました。</strong>
<p>129フレームで元とは異なる対応候補が選ばれ、事前の幾何・体節長などの条件を24フレームが通過しました。
さらに各部位の支持視点数を確認すると、<strong>視点数を減らさずに条件を満たすのは6フレーム</strong>でした。これらを優先して目視確認できるようにしています。</p>
<p>既知の12場面と対照6区間も同じ方法で評価しました。候補はすべて未採用で、個体IDの修正やスワップの減少を確定した結果ではありません。</p></div>
<nav><a href="#priority">優先確認6フレーム</a><a href="#videos">比較動画</a><a href="#support">視点数の注意点</a><a href="#images">全比較画像</a><a href="#graphs">グラフ</a><a href="#numbers">数値・詳細</a></nav>
<h2>見方</h2><p>画像は各カメラについて元の対応と再評価候補を上下に、動画では左右に並べています。
色はS0・S1・S2の追跡slotです。<strong>丸＝選んだ2D検出点、×と線＝3Dから画像に戻した位置、白枠＝変更した検出点</strong>です。
口→胸びれ→尾の順に線で結んでいます。投影位置に×があっても丸がなければ、その視点からの実際の観測はありません。
動画は1秒166フレームを約5.53秒に伸ばしたスローで、途中フレームを省略していません。</p>"""
    ]

    def detail_at(name, frame):
        return next((d for d in details[name]["frames"] if d["frame"] == frame), None)

    def status_at(name, frame):
        a = arrays[name]
        t = int(np.searchsorted(a["frames"], frame))
        d = detail_at(name, frame)
        if not d or not a["changed"][t]:
            return "対応変更なし"
        if loss[name][t]:
            return "部位の支持視点が減少：残差低下に注意"
        return "支持視点数を維持・優先目視" if a["supported"][t] else "条件未通過・未採用"

    def source_table(name, frame):
        a = arrays[name]
        t = int(np.searchsorted(a["frames"], frame))
        rows = []
        counts = [
            np.isfinite(a[key][t]).all(-1).sum(axis=1) for key in ("original_uv", "candidate_uv")
        ]
        for slot in range(3):
            for node, label in enumerate(("口", "胸びれ", "尾")):
                old, new = int(counts[0][slot, node]), int(counts[1][slot, node])
                support = f"{old} → {new}"
                if new < old:
                    support = f'<span class="supportloss">{support} 減少</span>'
                own = a["original_source"][t, slot, :, node].tolist()
                cand = a["candidate_source"][t, slot, :, node].tolist()
                rows.append(
                    [
                        f"S{slot} {label}",
                        support,
                        f"{number(a['baseline_ray_mm'][t, slot, node])} → {number(a['candidate_ray_mm'][t, slot, node])}",
                        esc(own) + " → " + esc(cand),
                    ]
                )
        return table(
            ["部位", "支持視点数", "光線残差 mm", "検出行［上面,側面1,側面2］（−1=なし）"], rows
        )

    parts.append(
        '<h2 id="priority">支持視点数を減らさない6フレーム</h2><p>24件の元の確認条件に対する追加の診断です。事前条件や解析結果を書き換えていません。'
        "支持が減らないことはID正解の保証ではなく、同じ観測量で比べるための確認です。</p>"
    )
    rows = []
    for frame in priority_frames:
        d = detail_at("cg20s", frame)
        changes = " / ".join(
            f"S{x['slot']} {x['view']} {x['node']} {x['original_row']}→{x['candidate_row']}"
            for x in d["source_changes"]
        )
        rows.append(
            [
                str(frame),
                esc(changes),
                f"{number(d['original_cost'])} → {number(d['candidate_cost'])}",
                esc(str(d["forward_valid"])) + " / " + esc(str(d["backward_valid"])),
            ]
        )
    parts.append(
        table(
            ["frame", "変えた対応", "合成コスト（低いほど整合）", "過去／未来の参照が有効なslot"],
            rows,
        )
    )
    parts.append(
        "<p>f76791は両方向の参照が使える胸びれ対応の候補です。f75398–75399とf76989–76992では一方向の参照が不十分なため、前後のID維持は別途確認が必要です。</p>"
    )

    parts.append('<h2 id="videos">元の対応／再評価候補のスロー比較</h2>')
    captions = {
        75398: "上面の部位対応を見直す候補。支持視点数は減りませんが、過去側の参照は不十分です。",
        76791: "側面2のS1・S2の胸びれ対応を交換する候補。支持視点数を維持し、前後の参照が使えます。",
        76988: "尾の残差は低下しますが、S1の尾が3視点から2視点になります。3視点一致の回復とは解釈できません。隣の76989以降には未使用の検出を使う別候補もあります。",
        76235: "側面1の口・尾の対応を変える候補。胸びれの飛びや水槽外のノードが残り、条件を満たしません。",
        76930: "欠けた尾を再構成する候補が選ばれますが、胸びれ–尾が約37 mmとなり、体節長の条件を満たしません。",
        30735: "尾の観測を別のslotへ移す候補で、元のslotの3D部位が欠けます。確認条件を満たしません。",
        31479: "対応変更で合成コストは下がりますが、一部の口・胸びれの光線残差は悪化します。前後参照も不足し、条件を満たしません。",
        323819: "大きな3視点不整合が残り、中心フレームでは対応変更が選ばれません。2D検出や反射の影響を再確認する場面です。",
        77415: "中心フレームでは元の対応が残ります。近接中の部位対応の正しさを、幾何条件だけで確定できません。",
    }
    videos = [e for e in entries if e["kind"] == "video"]
    order = {
        76791: 0,
        75398: 1,
        76988: 2,
        76235: 3,
        76930: 4,
        30735: 5,
        31479: 6,
        323819: 7,
        77415: 8,
    }
    for e in sorted(videos, key=lambda x: order.get(x["center_frame"], 99)):
        frame, name = e["center_frame"], e["case"]
        path = "review/" + e["path"]
        parts.append(
            f'<section><h3>f{frame} · {esc(status_at(name, frame))}</h3><p>{esc(captions.get(frame, "未採用の再評価候補を比較しています。"))}</p><video controls preload="metadata" src="{esc(path)}"></video>'
        )
        plots = [
            p
            for p in entries
            if p["kind"] == "plot"
            and p["case"] == name
            and p["center_frame"] == frame
            and (p.get("scope") == "local_video" or p["name"].endswith("_local_diagnostics"))
        ]
        if plots:
            p = "review/" + plots[0]["path"]
            parts.append(
                f'<details><summary>この1秒の誤差・体節長グラフ</summary><a href="{esc(p)}"><img loading="lazy" src="{esc(p)}" alt="f{frame}の局所診断グラフ"></a></details>'
            )
        parts.append(
            f"<details><summary>中心フレームの部位ごとの観測と残差</summary>{source_table(name, frame)}</details></section>"
        )

    parts.append("""<h2 id="support">残差の低下と、支持視点数の減少を区別する</h2>
<div class="warn">f76988のS1の尾では、光線残差8.29→0.69 mmと同時に支持視点数が3→2に減っています。
S2は2→3に増え、尾の残差は0.32→1.21 mmに悪化します。全体の観測数だけを数えると、この入れ替わりを見落とします。</div>
<p>2視点で再構成した点は、その2視点に合いやすくなります。3つ目の視点を除いて予測する検査も、この候補のS1の尾では実施不能になります。
したがって、この残差低下を3視点の整合性改善として数えていません。</p>""")
    support_plot = root / "review/tail_support_76988.png"
    if not support_plot.exists():
        raise FileNotFoundError(support_plot)
    sources.extend([support_plot, support_plot.with_suffix(".json")])
    parts.append(
        '<a href="review/tail_support_76988.png"><img loading="lazy" src="review/tail_support_76988.png" alt="尾の残差と支持視点数の推移"></a>'
    )

    parts.append(
        '<h2 id="images">全比較画像</h2><p>主区間の55候補区間から代表フレーム、逆方向で追加された14件、既知12例を収録しています。代表フレームは合成コストの改善量で選んだため、成功例の無作為標本ではありません。</p>'
        '<label>表示 <select id="filter"><option value="all">すべて</option><option value="priority">条件通過・支持視点減少なし</option><option value="loss">支持視点が減る候補</option><option value="known">既知の場面</option></select></label>'
    )
    for e in [x for x in entries if x["kind"] == "image"]:
        name, frame = e["case"], e["center_frame"]
        a = arrays[name]
        t = int(np.searchsorted(a["frames"], frame))
        path = "review/" + e["path"]
        flags = []
        if a["supported"][t] and not loss[name][t]:
            flags.append("priority")
        if a["changed"][t] and loss[name][t]:
            flags.append("loss")
        if name.startswith("known_"):
            flags.append("known")
        parts.append(
            f'<details class="sheet" data-types="{esc(" ".join(flags))}"><summary>{esc(name)} · f{frame} · {esc(status_at(name, frame))}</summary><p>{link(path, "元サイズの画像を開く")}</p><a href="{esc(path)}"><img loading="lazy" src="{esc(path)}" alt="{esc(name)} f{frame}の元対応と再評価候補"></a>{source_table(name, frame)}</details>'
        )

    parts.append(
        '<h2 id="graphs">各区間の全体グラフ</h2><p>破線が元の対応、実線が候補です。灰色帯が再評価したフレーム。体節長は各slot別、その他の誤差は表示slot内の最大値です。'
        "体節長の薄い帯は候補外で得た参照中央値±スケールで、信頼区間ではありません。欠測は線で埋めていません。</p>"
    )
    for e in [x for x in entries if x["kind"] == "plot" and x.get("scope") == "full_case"]:
        path = "review/" + e["path"]
        parts.append(
            f'<details><summary>{esc(e["case"])} の推移</summary><a href="{esc(path)}"><img loading="lazy" src="{esc(path)}" alt="{esc(e["case"])}の全体診断グラフ"></a></details>'
        )
    parts.append(
        '<h2 id="numbers">全区間の集計</h2><p>件数はnativeフレーム数です。同じf76930付近が主区間と既知例に重なるため、合算を独立な観測数とは解釈しません。</p>'
    )
    rows = []
    for c in report["cases"]:
        name = c["case"]["name"]
        a = arrays[name]
        no_loss = int((a["supported"] & a["core"] & ~loss[name]).sum())
        rows.append(
            [
                link(f"analysis/{name}/evaluation.json", name),
                str(c["tested_frames"]),
                str(c["changed_frames"]),
                str(c["review_worthy_frames"]),
                str(no_loss),
            ]
        )
    parts.append(
        table(
            ["区間・詳細JSON", "再評価", "異なる対応", "事前条件通過", "うち部位の支持減なし"], rows
        )
    )
    parts.append("""<p>20秒全体の10 mm超の胸びれ移動は元も候補も3件です。局所の当てはまりを改善する候補はありますが、テレポートやIDスワップを解消した結果ではありません。</p>
<h2>評価の範囲</h2><p>各視点で魚全体を割り当て直す216通りと未使用検出への置換を調べ、元対応と最良の全体対応から、1視点の口・胸びれ・尾・口＋胸びれを交換する候補を追加しました。
これは限定した局所探索で、あらゆる同時交換を網羅したものではありません。体節長と利用可能な前後の動きも評価しましたが、生物学的IDの正解ラベルは候補選択に使っていません。</p>
<p>残差は同じ検出から計算した整合性の指標です。視点を1つ除く検査もID正解を独立に保証しません。提案をそのまま連結するとフレームごとに対応が切り替わる場合があるため、継続時間・境界・IDの確認が必要です。</p>
<p>画像の投影は保存C34と候補の再構成位置を比較しています。別途、両方を同じ方法で再三角測量した残差も確認しました。変更していない部位の再計算差は最大0.0000105 mmで、支持点数の変化はありません。</p>""")
    audit_paths = [root / "support_audit.json", root / "refit_audit.json"]
    note_paths = sorted((root / "review").glob("*visual*notes*.json"))
    sources.extend([p for p in audit_paths + note_paths if p.exists()])
    parts.append(
        "<p>追加の確認記録："
        + " · ".join(
            link(p.relative_to(root).as_posix(), p.stem)
            for p in audit_paths + note_paths
            if p.exists()
        )
        + "</p>"
    )
    parts.append(
        "<p>再現用："
        + link("analysis/protocol.json", "事前条件")
        + " · "
        + link("analysis/summary.json", "全数値")
        + " · "
        + link("review/manifest.json", "画像・動画・グラフの記録")
        + " · "
        + link("analysis/multiview_node_reassessment.provenance.json", "解析の入力・コード")
        + "</p>"
    )
    parts.append(
        """</main><script>document.getElementById('filter').addEventListener('change',function(){for(const sheet of document.querySelectorAll('.sheet')){sheet.hidden=this.value!=='all'&&!sheet.dataset.types.split(' ').includes(this.value);}});</script></html>"""
    )
    document = "\n".join(parts)
    for path in re.findall(r'(?:href|src)="([^"]+)"', document):
        if not path.startswith("#") and not (root / html.unescape(path)).is_file():
            raise FileNotFoundError(path)
    with provenance("multiview_review_gallery", root) as rec:
        rec.input("builder", Path(__file__), sha256=sha256(Path(__file__)))
        for i, p in enumerate(sources):
            rec.input(f"source_{i}:{p.name}", p, sha256=sha256(p))
        rec.param("local_links_checked", True)
        rec.param("browser_interaction_verified", False)
        rec.param("primary_priority_frames_posthoc", priority_frames)
        target.write_text(document, encoding="utf-8")
        rec.output("gallery", target, sha256=sha256(target))
    print(target)


if __name__ == "__main__":
    main()

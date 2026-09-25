"""Render readable axes and a Japanese entry page for the development diagnosis."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parent / "appearance_diagnostic"

with provenance("appearance_diagnostic_rendering", ROOT, name="render") as record:
    record.input("arrays", ROOT / "completed/diagnostic_arrays.npz")
    record.input("summary", ROOT / "completed/summary.json")
    record.input("renderer", Path(__file__))
    with np.load(ROOT / "completed/diagnostic_arrays.npz", allow_pickle=False) as arrays:
        frames, distances = arrays["frame_idx"], arrays["distances"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True, sharey=True,
                             constrained_layout=True)
    colors = ("#e6a72b", "#11a9c2", "#d65da7")
    views = ("top_cam", "side1_cam", "side2_cam")
    for slot in range(3):
        for view in range(3):
            ax = axes[slot, view]
            for template in range(3):
                ax.plot(frames, distances[:, slot, view, template], color=colors[template],
                        lw=1.2, label=f"prefix S{template}")
            ax.axvspan(frames[0], frames[19], color="gray", alpha=.15)
            ax.axvline(333547, color="black", ls=":", lw=1)
            ax.set(title=f"raw S{slot} / {views[view]}", ylim=(0, 1),
                   ylabel="Hellinger distance", xticks=[333519, 333580, 333650, 333718])
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Source frame (dotted line: previously observed event)")
    output = ROOT / "distance_over_time_clear.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    record.output("readable_distance_figure", output)
    html = """<!doctype html><html lang="ja"><meta charset="utf-8">
<title>24 bin 外見特徴の開発区間診断</title>
<style>body{font:17px/1.7 sans-serif;max-width:1250px;margin:35px auto;padding:0 20px}
img{max-width:100%;border:1px solid #ddd}h2{margin-top:40px}code{background:#eee}</style>
<h1>24 bin の色ヒストグラムは、この区間のID確定には不十分</h1>
<p>対象は v1 開発 transient の 333519–333718（200フレーム）のみ。
最初の20フレームから、カメラ別・slot別に品質で重み付けした固定テンプレートを作成した。
baseline .767 は作成後の診断参照に限り使用し、テンプレート学習・閾値調整には使用していない。
GTラベル、validation、1分間の実行結果は読んでいない。coreも変更していない。</p>
<p><b>見えていることと、個体を区別できることが一致していない。</b>
quality は前景の支持量・連結成分の一貫性などを表し、IDが正しい確率ではない。
3台が同時に別個体のテンプレートを選ぶ場面もあるため、カメラの多数決だけでは解決しない。</p>
<ul><li>prefix S0/S2 のHellinger距離は top 0.125、side2 0.232。
この2個体の粗い色分布は近い。</li>
<li>raw S1 は全200フレーム・全3カメラで正のqualityを持つが、333547以降に
自身のprefix S1が最近傍になる割合は top 14.0%、side1 24.4%、side2 40.1%。
baselineとの観測source対応はS1のままなのに、後半は3カメラともprefix S2を選ぶ。</li>
<li>333547以降の単純累積コストでは raw [S0,S1,S2] → prefix [S1,S2,S0] が133.76で最小。
baselineの参照対応 [S2,S1,S0] は159.96。
別の誤った対応への差が26.20まで積み上がる。これは確率でも独立サンプルの証拠量でもない。</li></ul>
<p>コードは各RGBチャネル8 binずつの周辺分布を連結しており、模様の位置やRGB間の組合せを
保存していない。体軸周辺の前景画素から得る値なので、体の向き・撮影位置・反射・混入する画素の
変化が外見に影響し得る。既存プレビューで見え方と周辺反射の変化は確認できるが、
今回の診断はこれらの原因を個別に分離する実験ではない。</p>
<p>baselineの対応もモデル由来で、独立したオスIDの正解ではない。
「同じraw slot」はスワップ後には同じ動物を意味しない。ここでの割合をID正解率と呼ばない。</p>
<h2>prefixテンプレート間の距離</h2><img src="completed/01_template_separation.png">
<h2>自身・他slotテンプレートへの距離の時間変化</h2><img src="distance_over_time_clear.png">
<h2>qualityと3カメラの一致</h2><img src="completed/03_quality_and_view_agreement.png">
<h2>カメラ別の選択と、診断用baseline対応</h2><img src="completed/04_view_votes_and_reference.png">
<h2>累積が誤った対応を強める例</h2><img src="completed/05_cumulative_cost_and_drift.png">
<h2>使用済みの実画像プレビュー（変更なし）</h2>
<p>333519 と333678を見比べる。各画像の上段がbaseline、下段がv1候補。</p>
<img src="../recovery_v1_nominal_dev_transient/preview_000333519.jpg">
<img src="../recovery_v1_nominal_dev_transient/preview_000333678.jpg">
<p><a href="completed/summary.json">数値JSON</a> / 
<a href="completed/diagnostic_arrays.npz">診断用配列</a> / 
<a href="completed/diagnose_appearance.py">再現スクリプト</a></p>
<p>初回描画は軸書式の例外で途中終了した。その来歴と途中図はこのディレクトリ直下に保持し、
完了版の数値・来歴は <code>completed/</code> に保存した。上の時間軸図は文字の重なりを修正した。
入力・解析式は同じで、追跡結果を変更していない。</p></html>"""
    (ROOT / "index.html").write_text(html, encoding="utf-8")
    record.output("entry_page", ROOT / "index.html")

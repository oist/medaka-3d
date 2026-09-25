"""Post hoc report only, from sealed CPU body-pool diagnostic arrays."""
# ruff: noqa: RUF001

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from diagnose_body_pool_79372 import assert_hashes, sha

from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "body_pool_region_diagnostic_79372_v1"


def main():
    seal_path = SOURCE / "diagnostic.provenance.json"
    seal = json.loads(seal_path.read_text())
    if seal["status"] != "ok":
        raise ValueError("completed diagnostic required")
    hashes = {str(seal_path): sha(seal_path), str(Path(__file__)): sha(__file__)}
    hashes.update({v["path"]: v["sha256"] for v in seal["outputs"].values()})
    assert_hashes(hashes)
    summary = json.loads((SOURCE / "summary.json").read_text())
    with np.load(SOURCE / "all_candidates.npz", allow_pickle=False) as loaded:
        arrays = {k: loaded[k].copy() for k in loaded.files}
    output = SOURCE / "report_v1"
    output.mkdir()
    with provenance("body_pool_region_report_79372", output, name="report") as rec:
        for path, digest in hashes.items():
            rec.input(path, path, sha256=digest)
        source_copy = output / Path(__file__).name
        source_copy.write_bytes(Path(__file__).read_bytes())
        rec.output("report_code", source_copy, sha256=sha(source_copy))
        rows = summary["candidate_rows"]
        selected = summary["selected_refit_index"]
        u = arrays["unexplained_fraction"]
        score = arrays["score"]
        parts = arrays["weighted_loss_parts"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
        for arm, color in (("mask", "#0072b2"), ("random", "#d55e00")):
            ids = np.array(
                [
                    i
                    for i, r in enumerate(rows)
                    if r["arm"] == arm and r["stage"] == "after_14_step_fit"
                ]
            )
            axes[0].scatter(100 * u[ids, 0], score[ids], s=22, color=color, alpha=0.65, label=arm)
            axes[1].scatter(100 * u[ids, 2], score[ids], s=22, color=color, alpha=0.65)
            links = arrays["start_to_fit_indices"]
            links = np.array([p for p in links if rows[int(p[0])]["arm"] == arm])
            axes[2].scatter(
                100 * u[links[:, 0]][:, [0, 2]].mean(1),
                100 * u[links[:, 1]][:, [0, 2]].mean(1),
                s=22,
                color=color,
                alpha=0.65,
            )
        for ax, region in zip(axes[:2], (0, 2), strict=True):
            ax.scatter(
                100 * u[selected, region],
                score[selected],
                marker="*",
                s=160,
                color="black",
                label="unchanged objective winner",
                zorder=5,
            )
            ax.set_xlabel(
                ("Top D2 m0" if region == 0 else "Side1 D2 m2") + " unexplained mask mass (%)"
            )
            ax.set_ylabel("Original full fitting objective (lower is better)")
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        axes[2].plot([55, 100], [55, 100], "--", color="gray")
        axes[2].set(
            xlabel="Before fit: mean two-region unexplained (%)",
            ylabel="After fit: mean two-region unexplained (%)",
        )
        axes[2].grid(alpha=0.2)
        fig.suptitle("79372: all admitted starts; retrospective region diagnostics, no ID verdict")
        fig.tight_layout()
        figure = output / "candidate_tradeoffs.png"
        fig.savefig(figure, dpi=160)
        plt.close(fig)
        rec.output("plot", figure, sha256=sha(figure))
        lines = [
            "# 79372: 身体候補の到達性と選択理由",
            "",
            "未説明領域を部分的に描画する候補は生成され、14ステップ後にも残っていました。"
            "この時点の失敗を『候補が一つもない』とは説明できません。"
            "候補の一部は画像項を改善しても、固定SLEAP対応での欠測ペナルティと動きの項が増え、"
            "既存の総目的関数では選ばれません。領域の大半を説明できたわけではなく、"
            "新しい候補が実魚を正しく表すとの判断もしていません。",
            "",
            "成功記録のあるA実験の79372だけを対象に、241種類の開始形状と144種類の最適化後形状を"
            "CPU float64で再描画しました。開始形状には境界外として棄却されたものも含め、"
            "採用可能な群とは分けています。追加の最適化、追跡、候補選択変更、GT読込はありません。",
            "",
            "両群とも開始候補121（共通の元形状を含む）です。境界内はマスク93、ランダム52、"
            "境界外の棄却はそれぞれ28、69です。共通の元形状の再fitは一回だけ数えます。",
            "",
            "| 形状 | top m0 未説明 | side1 m2 未説明 | top m2 未説明 | 総fit目的関数 |",
            "|---|---:|---:|---:|---:|",
        ]
        highlights = [
            (0, "元形状（新しいSL対応で採点）"),
            (selected, "選択された共通元形状の再fit"),
            (15, "マスクtrial15、開始"),
            (247, "同trial15、fit後"),
            (244, "マスクtrial6、fit後"),
            (245, "マスクtrial7、fit後"),
            (248, "マスクtrial16、fit後"),
            (351, "ランダムtrial51、fit後"),
        ]
        for i, label in highlights:
            lines.append(
                f"| {label} | {u[i, 0]:.2%} | {u[i, 2]:.2%} | {u[i, 4]:.2%} | {score[i]:.6f} |"
            )
        lines += [
            "",
            "例示行は結果確認後に、開始からの移動・各領域での改善・目的関数との衝突を"
            "説明するために選んだものです。新たな採用候補の決定ではありません。"
            "全385行と棄却理由・対応元は元診断のCSV/NPZ/JSONに保存しています。",
            "",
            "マスク群では境界内93候補中、対象二領域の未説明率を元形状より両方減らすものが"
            "開始時31、fit後31でした。fit後の個別最小値はtop m0で71.32%、side1 m2で66.66%です"
            "（それぞれ異なる候補）。ランダム群は開始時8、fit後13候補でした。"
            "極小の数値差も含む比較数であり、成功数やID正解数ではありません。",
            "",
            "## 目的関数の内訳",
            "",
            "下表は選択された共通再fitとの差です。すべて実際の重みを掛けた値で、"
            "保存5項には入っていない姿勢方向の動きの項も旧式から再計算しました。"
            "六項の合計と保存総fit目的関数は1e-12の許容差で一致しています。",
            "",
            "| 候補 | node | shape | 胸部motion | 方向motion | repulsion | silhouette | 総差 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for i, label in highlights[2:]:
            delta = parts[i] - parts[selected]
            lines.append(
                "| "
                + label
                + " | "
                + " | ".join(f"{v:+.5f}" for v in delta)
                + f" | {score[i] - score[selected]:+.5f} |"
            )
        lines += [
            "",
            "マスクtrial6/7はraw S0を移した候補です。開始時に決めて固定したS0の口・胸・尾対応は"
            "全3視点とも-1で、fit後の直接支持もS0の3ノードすべてありません。"
            "これは全原画像に魚がない意味ではなく、この候補へSL行が割り当てられなかったことを表します。"
            "trial6の胸部は共通予測から24.80 mm離れ、選択候補の17.77 mmより遠い状態でした。"
            "直前のS0 body欠測ageは16で、旧motion sigmaは5.6 mmです。"
            "画像項は0.05544改善し、shape項も改善していますが、node項1.09312と"
            "胸部motion項0.63557の増加がそれを上回ります。"
            "この具体例ではshape priorを主要な棄却原因とは言えません。",
            "",
            "一方trial15はS1を移す候補で、S1の固定SL対応もすべて-1です。"
            "開始時はS1胸部が予測から11.81 mm離れ、fit後は6.83 mmへ近づきます。"
            "その間に対象領域の未説明率は61.99/68.00%から73.80/79.42%へ増加しました。"
            "S1は欠測age0・motion sigma0.8 mmのまま採点されるため、"
            "開始時には選択候補より胸部motion項が14.53大きくなっています。"
            "最適化された最終差分はmotionへの強い回帰と整合しますが、"
            "一項だけ無効にする反実仮想実験はしておらず、勾配の因果寄与を分離したとは主張しません。",
            "",
            "trial15/16はtop m2から形状を移すため、その領域の未説明率も増えます。"
            "対象二領域が改善しても画像全体のunion項が必ず改善するわけではありません。"
            "元top m2のS0/S1共有alpha平均は18.64%で、選択された再fit後は21.82%へ増えました。"
            "共有はD2 instanceが二匹を含む可能性もあるため、これだけで誤対応とは断定しません。",
            "",
            "## 検証範囲",
            "",
            "領域は独立D2のtop m0、side1 m2、top m2です。full maskに加え5×5楕円erosionの"
            "coreも全画素を集計し、invalid rayは分母から除外、空coreは未評価にしています。"
            "別候補で未説明率を小さくすること自体はIDの正しさや3匹すべての存在証明ではありません。"
            "三視点の輪郭整合性、身体形状、反射、個体識別の問題は残ります。",
            "",
            "元Aの選択四場面のdense未説明量と共有量はCPU再描画で1e-12以内に再現しました。"
            "元Aの全出力・使用校正・再利用コードと新出力のSHAを照合、入力は不変です。"
            "原動画はこの診断では再読込せず、成功記録に束縛されたmaskとposeを用いました。"
            "新テスト5件とRuff通過、385場面のCPU処理は46.13秒です。",
            "",
            "[全候補の対応表](../all_candidates.csv) / [数値NPZ](../all_candidates.npz) / "
            "[診断summary](../summary.json) / [診断来歴](../diagnostic.provenance.json)",
            "",
            "![全候補の画像説明と目的関数の関係](candidate_tradeoffs.png)",
        ]
        report = output / "RESULT.md"
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rec.output("report", report, sha256=sha(report))
        assert_hashes({**hashes, **{v["path"]: v["sha256"] for v in rec.outputs.values()}})
        rec.param("new_fit_or_selection", False)


if __name__ == "__main__":
    main()

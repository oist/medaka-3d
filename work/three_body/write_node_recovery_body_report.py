"""Preserve the negative result of the requested fixed proximity surrogate."""
# ruff: noqa: E501, RUF001

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "labs/20260909_three_body"))
from prepare_sequence import sha256  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

ROOT = REPO / "work/three_body/node_recovery_body_proximity_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    names = ("RESULT.md", "findings.json", "report.provenance.json")
    if any((ROOT / name).exists() for name in names):
        raise FileExistsError("Interpretation already exists; no overwrite")
    seal_path = ROOT / "diagnostic.provenance.json"
    seal = read(seal_path)
    if seal["status"] != "ok":
        raise ValueError("Geometry diagnostic incomplete")
    inputs = {Path(item["path"]): item["sha256"] for item in seal["outputs"].values()}
    inputs.update({Path(__file__): sha256(__file__), seal_path: sha256(seal_path)})
    if not all(sha256(path) == digest for path, digest in inputs.items()):
        raise ValueError("Sealed diagnostic output differs")
    summary = read(ROOT / "summary.json")
    pose_by_key = {(p["frame_idx"], p["choice_index"]): p for p in summary["poses"]}
    pairs = {
        (p["frame_idx"], p["choice_index"], tuple(p["fish_pair"])): p for p in summary["pair_rows"]
    }
    findings = dict(
        status="negative_selectivity_result_diagnostic_only",
        conclusion="The requested geometry surrogate does not preferentially oppose the suspected 80344 mouth-recovery candidate relative to its fitted base; it decreases there and increases for the visually improved 80431 candidate.",
        comparison_80344=[pose_by_key[80344, j] for j in range(3)],
        comparison_80431=[pose_by_key[80431, j] for j in range(3)],
        comparison_80261=[pose_by_key[80261, j] for j in range(3)],
        case_80370_has_no_recovery_candidate=True,
        dominant_gaussian_pair_80344={
            str(j): pairs[80344, j, (1, 2)]["minimum_standardized_pair"] for j in range(3)
        },
        mechanism_note="All three 80344 candidates attain minimum standardized distance at S1 Gaussian7 / S2 Gaussian1: tail-region versus mouth-pectoral region. The max-pair surrogate remains dominated by that pair, not the altered S1 mouth Gaussians.",
        limitations=[
            "The source visual assessments are diagnostic impressions, not independent biological identity labels.",
            "Only four fixed instants and ten final poses were measured; no threshold or new penalty was tried.",
            "A lower local-fit cost or a lower proximity surrogate does not validate the candidate identity.",
            "The dimensionless scalar is not an overlap probability or physical contact criterion.",
            "The 80431 surrogate increases but stays around 1e-6; this is a direction-of-change counterexample, not evidence of a practically large penalty.",
            "Absolute proximity is larger at 80344 than the other sampled frames, but using that cross-frame distinction as a rejection threshold was not validated.",
        ],
    )
    rows = []
    geometry_rows = []
    for frame in sorted({p["frame_idx"] for p in summary["poses"]}):
        for j in sorted(p["choice_index"] for p in summary["poses"] if p["frame_idx"] == frame):
            pose = pose_by_key[frame, j]
            label = "再fit base" if j == 0 else f"口候補{j} / S{pose['target_fish']}"
            rows.append(
                f"| {frame} | {label} | {pose['fitted_score_delta_vs_base']:+.6f} | {pose['proximity_surrogate']:.6g} | {pose['surrogate_delta_vs_fitted_base']:+.6g} |"
            )
            for pair in ((0, 1), (0, 2), (1, 2)):
                row = pairs[frame, j, pair]
                geometry_rows.append(
                    f"| {frame}/{j} | S{pair[0]}/S{pair[1]} | {row['pectoral_distance_mm']:.3f} | {row['polyline_closest']['distance_mm']:.3f} | {row['minimum_standardized_pair']['standardized_distance']:.3f} |"
                )
    report = f"""# 口再捕捉候補に対する3D近接指標の診断

指定された近接surrogateは、80344の疑わしい候補だけを基準姿勢より不利にする性質を示さなかった。
80344の候補1は元の局所fit損失もsurrogateも下がり、逆に口の改善が見られた80431の候補1では
surrogateが小さく増えた。この固定の集約量を、そのまま誤接続防止の新しい項として採用する根拠にはならない。

| frame | 保存済みfinal pose | 元fit損失 Δ vs base | 近接surrogate | 近接surrogate Δ vs base |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(rows)}

80344 の S1/S2 は、base→候補1で胸ヒレ間21.112→21.093 mm、
中心線間最短2.858→3.063 mm、最小標準化距離2.247→2.342。
候補2でも中心線距離3.303 mm、最小標準化距離2.493となり、surrogateはさらに下がる。
3姿勢すべてで最小標準化距離は S1 のGaussian7と S2のGaussian1で決まる。
これは S1の尾側と S2の口→胸ヒレ側で、変更したS1の口側Gaussianではない。
全64組中の最大近接だけを見る集約量が同じ局所的な近接に支配されることも、今回の測定に含まれる。

80431 の候補1では S1/S2 の中心線距離13.991→10.201 mm、最小標準化距離5.918→5.035となる。
surrogateは8.27e-9→1.04e-6へ増えるが、絶対値は小さい。
よい候補に対して大きな実害が出ると主張する結果ではなく、変化の向きだけでは選別できない反例である。
80261 の候補は近接を増やさず、80370には比較可能な口候補がなくbaseのみ。
80344の絶対値は他の時刻より大きいが、その事後差からしきい値を選んだりID誤りと判定したりはしていない。

固定定義は `mean_3魚pair(max_64Gaussianpair(exp(-0.5*d^2)))`。
`d = sqrt(delta^T (Sigma_f+Sigma_g)^-1 delta)` とし、元の8Gaussians×3fish、幅0.06、
軸・半径sigmaをそのまま用いた。無次元の近接の要約であり、重なり体積、衝突確率、ID尤度ではない。
各時刻のchoice0は元シーンを同条件で再fitしたbase。
同一frame内でのみ候補差を比較し、既存のスコア・姿勢・reviewフラグは変更していない。

| frame/choice | 魚pair | 胸ヒレ距離 mm | 中心線最短 mm | 最小標準化距離 |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(geometry_rows)}

全10姿勢・30魚pair・各64Gaussian組の値を [summary.json](summary.json)、
[候補比較CSV](candidate_comparison.csv)、[配列](bodies_and_distances.npz) に保存した。
CPU float64の幾何再計算のみで、新loss追加、重み調整、最適化、追跡再実行はない。
元の胸ヒレrepulsion再現差は最大 {summary["max_saved_repulsion_error"]:.3g}。
前の幾何計算10解析テストに加え、固定surrogateの尺度・順序不変性・不正入力の2テストが通り、Ruffも通過した。
測定は {summary["runtime"]["elapsed_s"]:.2f} 秒。既存の入力seal、全final poseのJSON/NPZ対応、コードと出力SHAを検査した。

元画像の良否は局所診断で、独立した生物学的ID正解ではない。
接触自体は誤りでなく、改善効果の確認には別の固定条件の追跡比較と人手の対応確認が必要。
この測定は [diagnostic.provenance.json](diagnostic.provenance.json)、解釈は別の
[report.provenance.json](report.provenance.json) に記録した。
"""
    with provenance("report_node_recovery_body_proximity", ROOT, name="report") as record:
        for path, digest in inputs.items():
            record.input(str(path), path, sha256=digest)
        path = ROOT / "findings.json"
        path.write_text(json.dumps(findings, indent=2, allow_nan=False), encoding="utf-8")
        record.output("findings", path, sha256=sha256(path))
        path = ROOT / "RESULT.md"
        path.write_text(report, encoding="utf-8")
        record.output("result", path, sha256=sha256(path))
        if not all(sha256(path) == digest for path, digest in inputs.items()):
            raise ValueError("An input changed during reporting")
    print(
        json.dumps(
            {
                name: sha256(ROOT / name)
                for name in (*names, "summary.json", "diagnostic.provenance.json")
            }
        )
    )


if __name__ == "__main__":
    main()

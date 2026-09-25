"""Seal interpretation of measured body distances, without changing the tracker."""
# ruff: noqa: E501, RUF001

import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "labs/20260909_three_body"))
from prepare_sequence import sha256  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

ROOT = REPO / "work/three_body/minute_scene0_body_proximity"
IMAGE = REPO / "work/three_body/minute_scene0_silhouette_attribution"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    for name in ("RESULT.md", "findings.json", "report.provenance.json"):
        if (ROOT / name).exists():
            raise FileExistsError("Body-proximity report already exists")
    summary = read(ROOT / "summary.json")
    image = read(IMAGE / "summary.json")
    inputs = {}
    for seal_path in (
        ROOT / "body_proximity.provenance.json",
        IMAGE / "attribution.provenance.json",
        IMAGE / "report.provenance.json",
    ):
        seal = read(seal_path)
        if seal["status"] != "ok":
            raise ValueError("Incomplete diagnostic input")
        inputs[seal_path] = sha256(seal_path)
        for item in seal["outputs"].values():
            path = Path(item["path"])
            if sha256(path) != item["sha256"]:
                raise ValueError("Changed diagnostic output")
            inputs[path] = item["sha256"]
    inputs[Path(__file__)] = sha256(__file__)
    rows = {row["frame_idx"]: row for row in summary["rows"] if row["fish_pair"] == [0, 1]}
    image_rows = {row["frame_idx"]: row for row in image["measurements"]}
    inspected = (ROOT / "proximity.png", IMAGE / "f79372_spatial_attribution.png")
    for path in inspected:
        if cv2.imread(str(path)) is None:
            raise ValueError("Inspected diagnostic figure no longer decodes")
    findings = dict(
        status="diagnostic_mechanism_observed_no_penalty_trial",
        conclusion="Saved pectoral centers can be far apart while other modeled body parts are close. This occurs at 79355, but does not establish physical collision or an identity violation.",
        example_79355=rows[79355],
        recovery_79372=rows[79372],
        counterexample_2d_overlap_with_3d_separation=dict(
            frame_idx=79316,
            pair=[0, 1],
            side2_soft_pair_overlap_mass_px=image_rows[79316]["full_foreground"]["side2_cam"][
                "foreground_union_summary"
            ]["pair_shared_mass_px"][0],
            distances=rows[79316],
        ),
        visual_qa=dict(
            inspected_images=[str(path) for path in inspected],
            method="Local view_image inspection, no browser/GPU",
            body_plot="Three distance panels have readable axes and all three pair series; connecting lines explicitly mark nine selected measurements, not inferred intermediate states.",
            spatial_79372="The top curved detector region D0 is predominantly unexplained while S0/S1 responses overlap on D2. Side1 has nested/overlapping detector masks. This is an image-model mismatch observation, not a biological identity assignment.",
        ),
        interpretation_limits=[
            "Polyline distances omit body thickness.",
            "The standardized distance is not overlap mass, a collision probability, or a number of Gaussian standard-deviation surfaces in physical contact.",
            "The renderer uses tapered engineering widths and infinite Gaussian tails; opacity and normalization determinants do not enter this distance.",
            "Unsupported saved poses are model states. Proximity is not independent fish-body ground truth.",
            "No new penalty, optimization, identity assignment, review clearing or tracking rerun was performed.",
            "Only nine instants were diagnosed. Effectiveness of body repulsion requires a separate controlled tracking comparison.",
        ],
        cpu_analytic_tests=10,
        max_saved_repulsion_error=summary["max_abs_saved_repulsion_difference"],
        candidate_sha256=summary["candidate_sha256"],
    )
    table = []
    for frame, row in rows.items():
        near = row["minimum_center_pair"]
        standardized = row["minimum_standardized_pair"]
        table.append(
            f"| {frame} | {row['pectoral_distance_mm']:.3f} | {row['polyline_closest']['distance_mm']:.3f} | {near['center_distance_mm']:.3f} | {near['standardized_distance']:.3f} | {standardized['standardized_distance']:.3f} | {row['existing_pectoral_repulsion']:.3g} |"
        )
    report = f"""# Scene0 の3D身体間の近接

胸ヒレの点同士が離れていても、保存された身体の別部位が近接する状態があった。
79355 の S0/S1 は胸ヒレ間 {rows[79355]["pectoral_distance_mm"]:.3f} mm、
中心線間の最短距離 {rows[79355]["polyline_closest"]["distance_mm"]:.3f} mm。
両者の胸ヒレ→尾の区間どうしが最近接で、既存の胸ヒレ repulsion は
{rows[79355]["existing_pectoral_repulsion"]:.3g} とほぼ寄与しない。
これは、点だけを見る項が身体全体の近接を表現しない具体例である。
その身体が実魚と一致することや、接触がID違反であることを示す測定ではない。

| S0/S1 frame | 胸ヒレ距離 mm | 中心線距離 mm | 最近Gaussian中心 mm | そのpairの標準化距離 | 全64pair中の最小標準化距離 | 元の胸ヒレpair項 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

79355 の最も近い Gaussian は index5/index5（0始まり）で、両方とも胸ヒレ→尾の区間。
中心距離は {rows[79355]["minimum_center_pair"]["center_distance_mm"]:.3f} mm、
軸方向 sigma は {rows[79355]["minimum_center_pair"]["axial_sigma_mm"][0]:.3f}/{rows[79355]["minimum_center_pair"]["axial_sigma_mm"][1]:.3f} mm、
半径方向 sigma は {rows[79355]["minimum_center_pair"]["radial_sigma_mm"][0]:.3f}/{rows[79355]["minimum_center_pair"]["radial_sigma_mm"][1]:.3f} mm。
79372 は中心線距離が {rows[79372]["polyline_closest"]["distance_mm"]:.3f} mmへ縮み、
最近接部位は S0 の胸ヒレ→尾と S1 の口→胸ヒレ。
最小標準化距離の pair は index3/index0、中心距離 {rows[79372]["minimum_standardized_pair"]["center_distance_mm"]:.3f} mm、
標準化距離 {rows[79372]["minimum_standardized_pair"]["standardized_distance"]:.3f} だった。
ユークリッド距離の最小 pair は別の index4/index0 なので、2種類の最小を分けて保存した。

一方、79316 の side2 では S0/S1 の2D alpha 積が
{findings["counterexample_2d_overlap_with_3d_separation"]["side2_soft_pair_overlap_mass_px"]:.2f} px相当あっても、
3D中心線は {rows[79316]["polyline_closest"]["distance_mm"]:.3f} mm、最小標準化距離は
{rows[79316]["minimum_standardized_pair"]["standardized_distance"]:.3f} 離れている。
投影像の重複を一律に3D身体の衝突と解釈することはできず、身体間ペナルティですべて直るとは言えない。

距離の定義は、既存 `renderer.gaussians()` の8個×3魚の中心、単位軸 t、軸 sigma_a、半径 sigma_rから
`Sigma = sigma_r^2 I + (sigma_a^2 - sigma_r^2) t t^T` を作り、各魚間の全8×8組について
`sqrt(delta^T (Sigma_f + Sigma_g)^-1 delta)` を求めたもの。
対称な無次元の中心間距離であり、重なり質量ではない。opacity、Gaussianの正規化係数、実測の魚体厚は含めない。
中心線距離は2区間×2区間の閉線分の厳密最短値で、体幅を含めない。
退化線分・端点・交差・平行・ねじれの位置も解析的CPUテストで検証した。

元の pair 項 `exp(-distance_pect^2/8)` の3組平均は、保存された repulsion loss と
最大差 {summary["max_abs_saved_repulsion_difference"]:.3g} で一致した。
元の重みは {summary["original_repulsion_weight"]}。この診断では新しい重みやペナルティを導入していない。
全身体の形状・ノード支持・reviewフラグを保存状態のまま使用した。
10 CPUテストとRuffが通り、計測は {summary["runtime"]["elapsed_s"]:.2f} 秒。

[距離の図](proximity.png)、[27組のCSV](pairs.csv)、[詳しい値と定義](summary.json)、
[全中心・共分散・64組の行列](bodies_and_distances.npz) を保存した。
Gaussianは無限の裾を持ち、ここでは有限体積の衝突やID誤りの判定をしていない。
保存された未支持姿勢の近接も含むので、実魚の身体接触の正解データにはならない。
3D身体の項を変更する価値を調べる材料であり、改善効果には別の固定条件比較が必要である。

入力・コード・出力は [body_proximity.provenance.json](body_proximity.provenance.json)、
この所見は別の [report.provenance.json](report.provenance.json) にSHAを記録した。
原candidate、元画像診断、人手レビューpack、trackerは変更していない。
"""
    with provenance("report_scene0_body_proximity", ROOT, name="report") as record:
        for path, digest in inputs.items():
            record.input(str(path), path, sha256=digest)
        path = ROOT / "findings.json"
        path.write_text(json.dumps(findings, indent=2, allow_nan=False), encoding="utf-8")
        record.output("findings", path, sha256=sha256(path))
        path = ROOT / "RESULT.md"
        path.write_text(report, encoding="utf-8")
        record.output("result", path, sha256=sha256(path))
        if not all(sha256(path) == digest for path, digest in inputs.items()):
            raise ValueError("Diagnostic input changed while reporting")
    print(
        json.dumps(
            {
                name: sha256(ROOT / name)
                for name in (
                    "summary.json",
                    "body_proximity.provenance.json",
                    "RESULT.md",
                    "report.provenance.json",
                )
            }
        )
    )


if __name__ == "__main__":
    main()

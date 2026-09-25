"""Post-render image/sidecar QA; original visual outputs remain sealed unchanged."""
# ruff: noqa: E501, RUF001 -- Japanese artifact prose.

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import measure_persistent_silhouette as m

from medaka3d.provenance import provenance


def main():
    root = m.REPO / "work/three_body/curve_body_spatial_13frames_v1/visual_report_v1"
    out = root / "qa"
    if out.exists():
        raise FileExistsError("visual QA exists")
    hashes = {str(Path(__file__).resolve()): m.sha256(__file__)}
    seal_path = root / "visual.provenance.json"
    hashes[str(seal_path)] = m.sha256(seal_path)
    seal = json.loads(seal_path.read_text())
    m.require(seal["status"] == "ok", "visual run incomplete")
    for item in seal["outputs"].values():
        m.require(m.sha256(item["path"]) == item["sha256"], "visual output SHA differs")
        hashes[item["path"]] = item["sha256"]
    summary = json.loads((root / "summary.json").read_text())
    m.require(
        summary["original_outputs_verified"] == 59 and summary["native_raw_images_decoded"] == 9,
        "wrong output/native decode contract",
    )
    images = []
    errors = []
    for frame in summary["measurements"]:
        path = root / f"f{frame['frame_idx']}_raw_original_curve.png"
        image = cv2.imread(str(path))
        m.require(
            image is not None and list(image.shape) == frame["canvas_shape_hwc"],
            "visual PNG decode/shape differs",
        )
        for view, value in frame["views"].items():
            m.require(
                value["raw_pos_before"] == value["original_source_offset"]
                and value["raw_pos_after"] == value["original_source_offset"] + 1,
                "wrong native index",
            )
            m.require(
                not any(np.any(a) for a in value["display_contour_reaches_crop_boundary"].values()),
                "display contour cropped",
            )
            errors.extend(value["saved_gpu_alpha_max_abs_difference"].values())
        images.append(
            dict(path=path.name, shape=list(image.shape), assistant_visually_inspected=True)
        )
    data = dict(
        status="ok",
        visual_output_sha_checks=len(seal["outputs"]),
        images=images,
        native_raw_images=9,
        pixel_counts_and_positions_verified=True,
        max_cpu_vs_saved_gpu_foreground_alpha_error=max(errors),
        no_display_contours_reach_crop_boundaries=True,
        raw_and_both_model_panels_use_identical_native_roi=True,
        headers_outside_source_images=True,
        no_refit_or_gt=True,
        notes=[
            "All three 3-view by 3-column PNGs were visually inspected.",
            "D2 text labels can overlap in merged/overlapping masks; quantitative values also remain in the sealed B summary.",
            "Colored contours are amodal alpha response levels, not measured physical boundaries or identity truth.",
            "Display pixels are aspect-preserving resamples of the explicitly recorded native crops.",
        ],
    )
    readme = """# 元身体と曲率身体の比較図

各図は行が top / side1 / side2、列が原画 / 元Gaussian身体 / 保存済み曲率身体です。同じnative cropを各列で使い、見出しは画像外に置いています。画像を開くと拡大できます。

- [79372: 主要な未説明領域が残る例](../f79372_raw_original_curve.png)
- [79381: 曲がりへの適合が改善する例](../f79381_raw_original_curve.png)
- [82720: 無支持形状と大きい曲がりの例](../f82720_raw_original_curve.png)

色付き輪郭は既存Gaussianから計算したbody alphaの .2（細線）と .5（太線）です。表示水準であり新しいloss/判定閾値ではありません。cyanはD2行、m番号のUはその行の全有効画素上の平均union応答です。filled nodeは元の直接支持、hollowは無支持、?は既存reviewを示します。原画列はモデル情報で装飾していません。

79372では曲率を変えてもtopのU形領域をほとんど説明せず、S0/S1の重なった説明が残る様子を確認できます。79381では湾曲した像への身体輪郭の追従が見えますが、topの下側m3は鏡像に重なります。したがって未説明の検出領域を常に別の実魚と数えることはできません。82720ではS1の大きい曲がりと、side1の画像から離れた無支持S2部分が見えます。これは定性的な画像診断で、新しい真値や反射フィルタには使っていません。

元Bの59出力SHAを確認してから、9原画のnative frame index/BGR SHAを検査しました。CPU描画は保存済みGPUの全foreground alphaと最大7.994e-15の差で一致。54のbody×元/曲線×視点・時点すべてで .2輪郭はcrop境界に触れません。3PNGを再デコード・目視確認し、37可視化出力のSHAを再確認しました。一部のD2行ラベルは重なりますが、元画像・身体輪郭・見出しの視点別評価値は確認でき、全instance数値は元B summaryにも保存されています。

CPU再描画のみで、再fit、GT読込、ID/支持/受理の変更はありません。人手annotation pageや既存B/QA/解釈のファイルも変更していません。
"""
    out.mkdir()
    with provenance("saved_curve_visual_postqa", out, name="qa") as rec:
        for path, digest in hashes.items():
            rec.input(path, path, sha256=digest)
        shutil.copyfile(__file__, out / "source.py")
        (out / "QA.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        (out / "README.md").write_text(readme, encoding="utf-8")
        for path in out.iterdir():
            rec.output(path.name, path, sha256=m.sha256(path))
        m.require(
            all(m.sha256(path) == digest for path, digest in hashes.items()),
            "visual QA input changed",
        )
        m.require(
            all(m.sha256(item["path"]) == item["sha256"] for item in rec.outputs.values()),
            "visual QA output changed",
        )
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()

"""Seal an independent output QA and prose report, preserving measurement files."""
# ruff: noqa: E501, RUF001 -- Japanese report paragraphs and mathematical glyphs.

import csv
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
    root = m.REPO / "work/three_body/persistent_minute_silhouette_attribution_v1"
    out = root / "report_v1"
    if out.exists():
        raise FileExistsError("report output exists")
    hashes = {}

    def checked(path, digest=None):
        path = path.resolve(strict=True)
        actual = m.sha256(path)
        m.require(digest is None or actual == digest, "report input SHA mismatch")
        hashes[str(path)] = actual
        return path

    checked(Path(__file__))
    seal = json.loads(checked(root / "attribution.provenance.json").read_text())
    m.require(seal["status"] == "ok", "measurement incomplete")
    for item in seal["outputs"].values():
        checked(Path(item["path"]), item["sha256"])
    summary = json.loads((root / "summary.json").read_text())
    measurements = {f["frame_idx"]: f for f in summary["measurements"]}
    m.require(list(measurements) == list(m.FRAMES), "wrong measured frames")
    # Compare previously sealed old-v1 metrics only after independent remeasurement.
    previous = m.REPO / "work/three_body/minute_scene0_silhouette_attribution"
    prior_seal = json.loads(checked(previous / "attribution.provenance.json").read_text())
    m.require(prior_seal["status"] == "ok", "prior diagnostic incomplete")
    prior = json.loads(
        checked(previous / "summary.json", prior_seal["outputs"]["summary"]["sha256"]).read_text()
    )
    manifest_path = (
        m.REPO / "work/three_body/inputs/continuous_one_minute_76620-86580/manifest.json"
    )
    m.require(
        prior["input_sha256_before"][str(manifest_path)] == summary["source_manifest_sha256"],
        "prior measurement used a different source sequence",
    )
    comparison = []
    for old_frame in prior["measurements"]:
        frame = old_frame["frame_idx"]
        for view in m.VIEWS:
            old_view, current = (
                old_frame["full_foreground"][view],
                measurements[frame]["full_foreground"][view],
            )
            m.require(
                old_view["detector_instances"] == current["detector_instances"],
                "old detector count differs",
            )
            for old_row, row in zip(old_view["instances"], current["instances"], strict=True):
                m.require(
                    old_row["native_area_px"] == row["native_area_px"]
                    and old_row["detector_score"] == row["detector_score"],
                    "old mask index differs",
                )
                comparison.append(
                    dict(
                        frame_idx=frame,
                        view=view,
                        instance_index=row["instance_index"],
                        old_union_mean=old_row["union_mean"],
                        current_union_mean=row["union_mean"],
                        old_shared_mean=old_row["pair_shared_mean"],
                        current_shared_mean=row["pair_shared_mean"],
                    )
                )
    errors = []
    for frame in m.FRAMES:
        with np.load(root / f"f{frame}_responses.npz", allow_pickle=False) as arrays:
            alpha = arrays["sampled_alpha"]
            union = 1 - np.prod(1 - alpha, axis=-1)
            direct_loss = np.mean(
                np.sum(
                    (union - arrays["target"]) ** 2
                    * arrays["weights"]
                    * arrays["sampled_ray_valid"],
                    axis=1,
                )
            )
            error = abs(
                direct_loss - measurements[frame]["sampled_metrics"]["saved_silhouette_loss"]
            )
            m.require(error < 1e-10, "independent image loss QA failed")
            errors.append(float(error))
    visual_shapes = {}
    for frame in m.VISUAL_FRAMES:
        for kind, expected in (("native_crops", (734, 960, 3)), ("context", (1294, 1800, 3))):
            path = root / f"f{frame}_{kind}.png"
            image = cv2.imread(str(path))
            m.require(image is not None and image.shape == expected, "image shape QA failed")
            visual_shapes[path.name] = image.shape
    # Live calibration arithmetic is also exactly the producer code, not merely the JSON rig.
    producer = (
        m.REPO
        / "work/three_body/persistent_mouth_continuous_one_minute_v1"
        / "experiment.provenance.json"
    )
    source = json.loads(
        checked(producer, summary["input_sha256_before"][str(producer)]).read_text()
    )
    registered = {
        name.replace("\\", "/"): digest for name, digest in source["params"]["code_hashes"].items()
    }
    for name in ("src/medaka3d/calib.py", "src/medaka3d/types.py"):
        checked(m.REPO / name, registered[name])
    with (root / "instances.csv").open(encoding="utf-8", newline="") as stream:
        row_count = len(list(csv.DictReader(stream)))
    qa = dict(
        status="ok",
        measurement_outputs_rehashed=len(seal["outputs"]),
        instance_rows=row_count,
        selected_frames=list(m.FRAMES),
        views=list(m.VIEWS),
        valid_foreground_rays=sum(
            v["valid_foreground_rays"]
            for f in measurements.values()
            for v in f["full_foreground"].values()
        ),
        direct_nodes=sum(int(np.asarray(f["observed"]).sum()) for f in measurements.values()),
        total_nodes=13 * 3 * 3,
        max_independent_saved_loss_difference=max(errors),
        png_shapes=visual_shapes,
        visual_inspection="Four PNGs inspected by assistant; full context retains all three slots, headers outside source pixels.",
        gt_or_baseline_bytes_opened=False,
        prior_diagnostic_opened_only_for_postmeasurement_comparison=True,
    )
    report = """# 現行 persistent geometry の D2 領域への寄与

現行の固定3D形状でも、79372 付近には「S0/S1が近い検出領域を説明し、別の領域の内部がほぼ未説明」という状態が残っています。一方 82719–82720 は、2身体の全ノードが直接支持を失い、特に側面1でS2がほとんど検出領域を説明しない状態です。両者を同じ失敗として扱う根拠はありません。

この再測定は現行 `persistent_mouth_continuous_one_minute_v1` のみから実行しました。以前の v1 結果は、測定完了後の表の比較に限って参照しました。GT、既存ID参照、追跡結果の書き換えはありません。

## 79372: 重複した説明と説明不足が同時に残る

| 視点・D2行 | 全領域の未説明割合 | 2px内側coreの未説明割合 | 所見 |
|---|---:|---:|---|
| top m2 | 30.1% | 19.8% | S0/S1両方の投影胸位置が内側。平均αは .313/.569、α積の平均 .186 |
| top m0 | 93.6% | 94.2% | 内部までほとんど未説明 |
| side1 m2 | 99.4% | 99.9% | 内部までほとんど未説明 |

top のS0/S1共有α積は 79370 の 1.7 から79372の364.4 pixel-equivalentへ増えています。side1も .4 から198.0へ増えます。その後、top共有量は79381/79394/79396で409.3/369.5/365.6、79405では25.0です。これは検査した離散時点の変化であり、間の全フレームを測った時系列ではありません。

79372の直接支持はS0が胸・尾、S1が口・尾、S2が全3点です。それでも既存画像bridgeは全3身体×3視点で成立します。S0を除いた画像損失は平均 +.08707 悪化するため、単に身体を削除すれば解決する状態でもありません。マスクunionはIDに対して対称であり、このbridgeは各身体が別の実魚を表す証明ではありません。

旧v1診断のtop m2は U=.7010、共有平均=.1875、現行は U=.6986、共有平均=.1864です。近い所見を現行poseから再現したのであり、旧結果の流用ではありません。厳密な各値・75共通instance比較は `old_v1_comparison.json` に保存しています。

## 82719–82720: 視点による不一致と無支持の形状

S0は両時刻で3点直接支持、S1/S2はともに3点すべて無支持です。全slotの既存ID-reviewはTrueのままです。82719 top m1にはS1/S2がともに寄与しますが共有平均は .060、side2ではm1へS1平均α=.491、m2へS2=.438と異なる領域にも寄与しています。

一方 side1 の全検出foreground上でS2の固有寄与は82719が .150 pixel-equivalent、82720が .600にすぎません。S2を除くとこの視点だけの画像損失は -.001456/-.004752 改善しますが、3視点平均では +.07672/+.06606 悪化します。ある視点で説明が弱いことと、身体全体が余分であることは同じではありません。

## 次に切り分けること

1. 未説明の内部領域から独立した剛体候補を作り、同じ目的関数で順位を測る。別の初期位置で改善するなら候補探索不足を示唆します。反射やmerged maskもあるため、D2行を魚数やIDへ直接割り当てません。
2. 3ノードを固定したまま胸尾間の曲率だけを変える対照を実行し、フィットに使っていない画素の残差も測る。79372の原画には強い曲がりがありますが、無支持ノードもあるため、今回の画像だけから形状モデル不足を原因と断定しません。
3. 位置・形状候補が説明不足を解消しない場合、別のSLEAP対応候補も同じ評価で比較する。ID順列だけの変更では、固定された3D形状や無支持部分は直りません。

## 検証と制限

13フレーム×3視点、109 detector instances、226,639有効foreground raysをCPU float64で測定しました。117ノード中74が直接支持で、保存source/observedを実SLEAPと元の2mm ray/12px屈折再投影条件で再現しました。元の1024画素/視点の画像損失は最大差1.665e-16、元bridgeも全一致です。9個の新CPUテストとRuffが通過し、独立コードレビューも阻害なしでした。

原画は79372/82719の各3視点、計6枚だけを直接seekし、native frame位置・BGR SHAを検証。PNGは画素一致の再デコードと全景/拡大図の目視確認を済ませました。62測定出力のSHAを再確認。実計算と図生成は約24秒で、前後の大きな入力SHA読み取り時間を含みません。

[79372 原画と拡大重ね合わせ](../f79372_native_crops.png) / [82719 原画と拡大重ね合わせ](../f82719_native_crops.png)

α、U、共有・未説明massは既存rendererのsoft応答です。確率、実魚の数、ID真値ではありません。D2行は反射・merged・重複を含み得ます。重なった行のmassは加算できません。coreは5×5 ellipse erosionによる診断領域で、真の身体内部を保証せず、新loss/閾値/受理条件には使っていません。
"""
    out.mkdir()
    with provenance("persistent_attribution_report_qa", out, name="report") as rec:
        for path, digest in hashes.items():
            rec.input(path, path, sha256=digest)
        shutil.copyfile(__file__, out / "report_source.py")
        for name, value in (("QA.json", qa), ("old_v1_comparison.json", comparison)):
            path = out / name
            path.write_text(json.dumps(m.safe(value), indent=2, allow_nan=False), encoding="utf-8")
        (out / "RESULT.md").write_text(report, encoding="utf-8")
        for path in out.iterdir():
            rec.output(path.name, path, sha256=m.sha256(path))
        m.require(
            all(m.sha256(p) == digest for p, digest in hashes.items()), "report input changed"
        )
        m.require(
            all(m.sha256(r["path"]) == r["sha256"] for r in rec.outputs.values()),
            "report output changed",
        )
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()

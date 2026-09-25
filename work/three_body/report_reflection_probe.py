"""Read-only report of the registered fixed reflection response experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from medaka3d.provenance import provenance  # noqa: E402

ROOT = REPO / "work/three_body/reflection_probe_13frames_v1"
OUT = ROOT / "report_v1"
MODES = (
    "unbounded_direct",
    "finite_direct",
    "inner_unit",
    "outer_unit",
    "both_unit",
    "inner_fresnel",
    "outer_fresnel",
    "both_fresnel",
)
VIEWS = ("top_cam", "side1_cam", "side2_cam")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError("report output exists")
    seal_path = ROOT / "probe.provenance.json"
    seal = json.loads(seal_path.read_text())
    assert seal["status"] == "ok"
    hashes = {str(seal_path): sha(seal_path)}
    for entry in seal["outputs"].values():
        assert sha(entry["path"]) == entry["sha256"]
        hashes[entry["path"]] = entry["sha256"]
    source_paths = [Path(__file__), REPO / "src/medaka3d/provenance.py"]
    hashes.update({str(p): sha(p) for p in source_paths})
    result = json.loads((ROOT / "summary.json").read_text())
    rows = result["frame_results"]
    primary = next(r for r in rows if r["frame_idx"] == 79381)
    m3 = primary["foreground"]["top_cam"]["instances"][3]
    core = primary["core"]["top_cam"][3]
    error = max(
        [r[k]["unbounded_parity_max_abs"] for r in rows for k in ("fitting", "evaluation")]
        + [r["foreground_parity_max_abs"] for r in rows]
        + [primary["visual"]["unbounded_parity_max_abs"]]
    )
    raw = cv2.imread(str(ROOT / "f79381_top_raw.png"))
    assert raw.shape == (1200, 1200, 3)
    assert hashlib.sha256(raw.tobytes()).hexdigest() == primary["visual"]["native_bgr_sha256"]
    png_qa = {}
    for name in ("f79381_top_reflection.png", "f79381_top_primary_detail.png"):
        image = cv2.imread(str(ROOT / name))
        assert image is not None and image.dtype == np.uint8 and image.shape == (1400, 2100, 3)
        png_qa[name] = dict(
            shape=list(image.shape),
            visual_inspection="Viewed: raw, D2, response panels legible; "
            "partial m3 overlap and outside-D2 lobes visible; no ID claim",
        )
    summaries = {}
    flat_rows = []
    for mode in MODES:
        value = np.array([r["evaluation"]["scores"][mode]["loss_by_stratum"][0] for r in rows])
        summaries[mode] = dict(
            mean=float(value.sum(-1).mean()),
            mean_by_view=value.sum(-1).mean(0).tolist(),
            mean_by_stratum=value.mean((0, 1)).tolist(),
        )
        for r, by_view in zip(rows, value, strict=True):
            for view, parts in zip(VIEWS, by_view, strict=True):
                flat_rows.append(
                    dict(
                        frame_idx=r["frame_idx"],
                        view=view,
                        mode=mode,
                        foreground_loss=parts[0],
                        ring_loss=parts[1],
                        global_background_loss=parts[2],
                        total=parts.sum(),
                    )
                )
    table = "\n".join(
        f"| {r['frame_idx']} | {r['physical_box_outside_nodes']} / "
        f"{r['physical_box_outside_gaussian_centers']} | "
        + " | ".join(
            f"{r['evaluation']['scores'][m]['mean_view_loss'][0]:.6f}"
            for m in ("unbounded_direct", "finite_direct", "both_unit", "both_fresnel")
        )
        + " |"
        for r in rows
    )
    modes_table = "\n".join(
        f"| {m} | {s['mean']:.6f} | " + " / ".join(f"{x:.6f}" for x in s["mean_by_stratum"]) + " |"
        for m, s in summaries.items()
    )
    text = f"""# 反射経路は注目領域を部分的に説明するが、採用判定は保留

固定 geometry と名目上の y=192/196 mm の面を動かさずに、79381 top の D2 m3 へ
反射応答が生じました。m3 全1855画素で最初の壁は Y192、内外の枝とも有効です。
直接経路の平均応答は0、両反射を加えた unit 応答は {m3["mean"]["both_unit"]:.6f}、
Fresnel 重み付きは {m3["mean"]["both_fresnel"]:.6f} でした。core 1409画素でも
{core["mean"]["both_unit"]:.6f} / {core["mean"]["both_fresnel"]:.6f} です。
これは確率や説明できた面積比ではなく、固定 alpha の画素平均です。

主な寄与は固定 raw slot S1 からです。unit の枝別 amodal mass は内面
{m3["per_body_branch_mass"]["inner"][1]:.3f}、外面
{m3["per_body_branch_mass"]["outer"][1]:.3f}（それぞれ他 body との union 前）。
内面 weight の平均は {m3["paths"]["inner"]["weight_mean"]:.6f}、外面は
{m3["paths"]["outer"]["weight_mean"]:.6f}。これを生物学的 ID の同定には使いません。

![主領域の固定対照](../f79381_top_primary_detail.png)

図では m3 の細長い領域への部分的一致に加え、D2 外へも応答が増えています。
その一部の位置には原画で像が見えますが、D2 外を真の背景と決めることも、
それらを真の反射と確定することもしません。魚体曲率、輪郭、可視性、光量の
モデル不足が残っています。表示は4 native pixel間隔、定量は登録済み画素です。

## 13時点すべての同一 spatial 評価

全視点平均の誤差です。小さいほど同じ D2 target に近い値です。fit除外の固定
foreground/ring/global 各画素を使い、元の有効ray分母を保ちました。

| frame | 名目box外のnode / Gaussian center | 元unbounded | finite直接 | 両反射unit | 両反射Fresnel |
|---|---:|---:|---:|---:|---:|
{table}

| モード | 13時点×3view平均 | foreground / ring / global の重み付き誤差 |
|---|---:|---|
{modes_table}

反射の追加は finite 対照に対する平均を改善しますが、unit では79372・79405など
悪化する例もあります。Fresnel でも ring/global の平均誤差は finite より増えます。
追加応答が増えれば常に良いという結論ではありません。全frame/view/stratum の
数値は evaluation.csv、各mask/core/枝/body の寄与は元 summary.json に残しています。

## 別の大きな差：名目底面と固定 geometry

元unbounded平均 {summaries["unbounded_direct"]["mean"]:.6f} に対し、有限直接だけで
{summaries["finite_direct"]["mean"]:.6f} へ悪化します。scene0側の11時点は、固定nodeの
2〜6点とGaussian centerの5〜16点が名目box外にあり、最低zは約−6 mmです。
82719/82720は外側node/centerが0で、finite対照は元と表示精度内で一致します。
side2の一部rayは水側entry自体が名目box外です。無支持rayを評価分母から消して
この差を隠していません。

したがってこの対照では、反射追加の差と、名目 z=0 面で従来応答を打ち切る差が
同時に現れます。底面位置の実測と校正 gauge の対応は未確認です。ここから
底面の正しさ、反射モデル全般の失敗、または新しいfilterの採用を結論しません。
登録geometryの z=−6 許容を反射光路へ流用する変更も行っていません。

## 検証と範囲

独立 NumPy の unbounded 応答と元 torch float64 renderer の最大差は {error:.3g}。
13時点すべてで元fit loss、保存済みfit/eval座標・target・weights、SLEAP支持を
再照合しました。入力/source前後不変、全{len(seal["outputs"])} outputs SHA と成功sealを確認。
計測部分 {result["elapsed_seconds"]:.2f} 秒、前後の全ファイル検証時間は別です。
原画は79381 topの1枚だけをnative seek/POS/保存BGR SHAで照合し、PNG可逆性と
2図のdecode/目視を確認しました。事前登録は reflection_probe_registration_v1。

姿勢、反射面、width、opacity、屈折率のfitはありません。旧結果、支持/review、
IDは不変、GT未読、CPUのみ。Fresnel ray power × engineering alpha は物理画像
likelihoodではありません。今回の13時点は既に開封された開発場面で、独立検証や
一般化の証拠ではありません。現時点は**反射候補を保持する設計材料、追跡への採用は保留**です。
"""
    OUT.mkdir()
    with provenance("report_fixed_reflection_probe", OUT, name="report") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for source in source_paths:
            target = OUT / "code_snapshot" / source.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            record.output(target.relative_to(OUT).as_posix(), target, sha256=sha(target))
        report = OUT / "RESULT.md"
        report.write_text(text, encoding="utf-8")
        path = OUT / "evaluation.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
        qa = dict(
            successful_probe_outputs_checked=len(seal["outputs"]),
            maximum_unbounded_parity_absolute_error=error,
            png_qa=png_qa,
            original_native_raw_sha256=primary["visual"]["native_bgr_sha256"],
            summary=summaries,
            no_fit=True,
            no_gt=True,
            adopted_into_tracking=False,
        )
        qa_path = OUT / "QA.json"
        qa_path.write_text(json.dumps(qa, indent=2, allow_nan=False), encoding="utf-8")
        for item in (report, path, qa_path):
            record.output(item.name, item, sha256=sha(item))
        assert all(sha(p) == value for p, value in hashes.items())
        assert all(sha(e["path"]) == e["sha256"] for e in record.outputs.values())
        print(json.dumps(qa), flush=True)


if __name__ == "__main__":
    main()

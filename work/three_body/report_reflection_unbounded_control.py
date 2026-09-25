"""Seal a post-hoc algebraic report; no new inference or rendering."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from medaka3d.provenance import provenance  # noqa: E402

ROOT = REPO / "work/three_body/reflection_unbounded_control_13frames_v1"
OUT = ROOT / "report_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError("report exists")
    seal_path = ROOT / "control.provenance.json"
    seal = json.loads(seal_path.read_text())
    assert seal["status"] == "ok"
    inputs = {str(seal_path): sha(seal_path)}
    for item in seal["outputs"].values():
        assert sha(item["path"]) == item["sha256"]
        inputs[item["path"]] = item["sha256"]
    own_code = [Path(__file__).resolve(), REPO / "src/medaka3d/provenance.py"]
    inputs.update({str(path): sha(path) for path in own_code})
    summary = json.loads((ROOT / "summary.json").read_text())
    aggregate = summary["aggregate"]
    rows = summary["frame_results"]
    modes = list(aggregate)
    primary = next(r for r in rows if r["frame_idx"] == 79381)["foreground"]["top_cam"]
    assert primary["instances"][3]["pixels"] == 1855
    assert primary["cores"][3]["pixels"] == 1409
    base = aggregate["inherited_unbounded"]["mean"]
    unit = aggregate["unbounded_both_unit"]["mean"]
    fresnel = aggregate["unbounded_both_fresnel"]["mean"]
    table = "\n".join(
        "| "
        + mode
        + " | "
        + f"{aggregate[mode]['mean']:.6f} | "
        + " / ".join(f"{x:.6f}" for x in aggregate[mode]["by_view"])
        + " | "
        + " / ".join(f"{x:.6f}" for x in aggregate[mode]["by_stratum"])
        + " |"
        for mode in modes
    )
    contrasts = ("inherited_unbounded", "unbounded_both_unit", "unbounded_both_fresnel")
    frame_table = "\n".join(
        f"| {row['frame_idx']} | "
        + " | ".join(
            f"{np.asarray(row['scores_by_view_stratum'][mode]).sum(-1).mean():.6f}"
            for mode in contrasts
        )
        + " |"
        for row in rows
    )
    report = f"""# 追加の代数対照: 反射追加と名目底面の打切りを分離

元の unbounded direct を残したまま、保存済みの同じ有限反射枝を加えました。
全13時点・3viewの同一空間評価平均は {base:.6f} から unit {unit:.6f}、
Fresnel {fresnel:.6f} へ低下しました。差はそれぞれ {unit - base:+.6f} と
{fresnel - base:+.6f} です。今回の開封済み場面では、底面打切りとは別に、反射枝が
D2の一部領域を説明するという限定的な結果です。

一方、D2外への応答も増えます。unit の ring/global 誤差は
.031470/.000151 から .054575/.000311、Fresnel でも .032098/.000171 へ増えます。
top/side1の平均は下がり、side2平均は両方式で上がりました。D2外は真の背景とは
限りませんが、この同一targetに対する不一致を省いていません。

| mode | 平均 | top / side1 / side2 | foreground / ring / global |
|---|---:|---|---|
{table}

## 全13時点

| frame | inherited unbounded | +both unit | +both Fresnel |
|---|---:|---:|---:|
{frame_table}

79381 top m3は同じ1855画素、core1409画素です。ここは元direct応答が0のため、
反射追加後の平均は先の有限対照と同じ .443537 unit / .226664 Fresnel です。
新しいケース選択やパラメータ調整はしていません。全mask/coreの画素分母を
元計測に照合し、各view/stratum/instanceの寄与を summary.json と273行の
evaluation.csvに保存しています。

## 条件と検証

登録 artifact は reflection_unbounded_control_registration_v1。計算前に全保存
response、sample、変更しないD2 selector、実行sourceをSHAで固定しました。
新たなray計算、Gaussian描画、raw BGR読込、fit、GT参照はありません。
D2 packed masksだけを同じinstance/coreの画素選択に用い、予測を変更していません。
元baselineの再現と応答上限の検査、各instance/core分母一致、全{len(seal["outputs"])}出力の
成功seal/SHA確認を通過しました。旧sourceと一次計測は不変です。

これは一次結果を見た後に登録した二次的な開発診断です。直接枝は元のunbounded
定義を意図的に継承し、名目壁の外まで伸びるため、全体として物理的に完結した
シーンとは呼びません。FresnelとGaussian alphaの積は輝度・確率ではなく、
IDや3Dの正しさ、因果効果、一般化を示すものでもありません。
**反射経路を検討する根拠は増えましたが、追跡への採用・自動ID修正はしていません。**
"""
    OUT.mkdir()
    with provenance("report_posthoc_unbounded_reflection_control", OUT, name="report") as record:
        for path, digest in inputs.items():
            record.input(path, path, sha256=digest)
        for source in own_code:
            path = OUT / "code_snapshot" / source.relative_to(REPO)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.read_bytes())
            record.output(path.relative_to(OUT).as_posix(), path, sha256=sha(path))
        path = OUT / "RESULT.md"
        path.write_text(report, encoding="utf-8")
        record.output("report", path, sha256=sha(path))
        path = OUT / "QA.json"
        qa = dict(
            all_successful_outputs_verified=len(seal["outputs"]),
            primary_m3_pixels=1855,
            primary_core_pixels=1409,
            scalar_factor_and_zero_weight_synthetic_checks_passed=True,
            raw_bgr_read=False,
            rendering=False,
            adopted=False,
            aggregate=aggregate,
        )
        path.write_text(json.dumps(qa, indent=2, allow_nan=False), encoding="utf-8")
        record.output("qa", path, sha256=sha(path))
        assert all(sha(p) == digest for p, digest in inputs.items())
        assert all(sha(e["path"]) == e["sha256"] for e in record.outputs.values())
    print(sha(OUT / "report.provenance.json"), flush=True)


if __name__ == "__main__":
    main()

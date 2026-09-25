"""Seal a limited scientific interpretation of the completed geometry diagnosis."""
# ruff: noqa: E501, RUF001
# The multiline Japanese report preserves paragraphs and native punctuation.

import csv
import json
from pathlib import Path

import audit_reflection_temporal as audit

from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent
ROOT = HERE / "reflection_temporal_200_geometry_diagnosis_v1"


def main():
    bindings = {str(Path(__file__).resolve()): audit.sha256(__file__)}
    seal = audit.producer.verify_outputs(
        ROOT,
        "diagnosis.provenance.json",
        "posthoc_reflection_temporal_geometry_diagnosis",
        bindings,
    )
    for item in seal["inputs"].values():
        if Path(item["path"]).suffix == ".py":
            audit.producer.bind(item["path"], item["sha256"], bindings)
    for name in (
        "summary.json",
        "fixed_frame_details.json",
        "all_fish_frames.csv",
        "all_pair_frames.csv",
    ):
        audit.producer.output_path(seal, name, ROOT / name)
    summary = audit.read_json(ROOT / "summary.json")
    with (ROOT / "all_fish_frames.csv").open(encoding="utf-8", newline="") as stream:
        fish = {
            (row["arm"], int(row["frame_idx"]), int(row["slot"])): row
            for row in csv.DictReader(stream)
        }
    with (ROOT / "all_pair_frames.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    close_frames = [
        int(row["frame_idx"])
        for row in rows
        if row["arm"] == "fresnel" and row["pair"] == "S0_S1" and float(row["pectoral_mm"]) < 4
    ]
    supported = [
        sum(fish["fresnel", frame, slot]["observed_pectoral"] == "True" for slot in (0, 1))
        for frame in close_frames
    ]
    support_counts = {str(count): supported.count(count) for count in (0, 1, 2)}
    stats = summary["statistics"]
    result = dict(
        schema="reflection_temporal_geometry_interpretation_v1",
        parent_seal_sha256=audit.sha256(ROOT / "diagnosis.provenance.json"),
        fresnel_S0_S1_chest_below4mm_frames=close_frames,
        number_of_direct_chests_within_close_pair_counts=support_counts,
        S0_S1_minimum_mm={
            a: stats[a]["pairs"]["S0_S1"]["pectoral_mm"]["minimum"] for a in audit.ARMS
        },
        interpretation="Late proximity/observation competition is a duplicated-explanation candidate; "
        "it does not establish biological duplication or the first causal divergence.",
        data_unchanged=True,
        new_fit=False,
        new_render=False,
        GT=False,
    )
    text = """# 保存200フレームの解釈

**S0/S1の後半の接近と観測の取り合いは、二重説明の候補である。ただし最初の支持喪失を説明する単一原因とは言えない。** 全200保存系列を調べた結果であり、生物学的に同じ魚を追っているという確定ではない。

- Fresnelの胸距離4mm未満は16フレーム（79376:79384、79394:79402）。Directは0。最小1.602850mmは79401で、平均折れ線体長の6.79%。この時刻の両胸は未観測である。16フレーム全体では両胸直接支持6、片胸4、両胸未支持6なので、接近はlatentだけの現象にも限定できない。
- 最初の支持喪失79318ではS0/S1距離はDirect 10.604mm、Fresnel 10.570mmとほぼ同じ。S0胸のside1 SLEAP row1支持がFresnelにはなく、side2 row1だけが残る。これは後半の密着とは別の最初の差であり、今回の保存配列だけではそのオンライン因果を確定できない。
- 79372では距離が11.567→4.620mm。Directのtop胸投影はS0→D2 row0、S1→row2だが、Fresnelは両方row2へ入る。FresnelのS0胸はside1 SLEAP row2・side2 row1を使い、S1は同じ行の別nodeを保持する。排他規則は同一view・同一nodeの再利用を禁止するため、この別node間の分割共有は規則違反として排除されない。D2 rowとSLEAP rowは別の検出器の番号であり相互に同一視しない。
- 79405ではFresnel S0の全nodeがtop SLEAP row2・side2 row1を使い、これは同時刻Direct S1の支持行と一致する。胸距離は7.271mmへ戻っており、全区間をずっと二重説明しているとは言えない。後半の匿名slot割当の変化もあり得るが、人手なしに生物ID交換とはしない。

全200で同一view・同一nodeのSLEAP行再利用は両方式0。一方、S0/S1の**別nodeが共有するSLEAP instance行**は72→158件（frame/view/body-pair/共有row単位）。各body内の複数row混在はS0 22/392→41/352、S1 4/474→14/424 body-view（分母は2node以上にsourceあり）。幾何の接近と、この観測モデルの分割共有を分けて扱う必要がある。

反証的な数値も残す。S0/S1の胸距離中央値は9.642→9.699mmで、全期間の一様な接近ではない。体軸2線分の最小距離はDirect 0.141mm、Fresnel 0.447mmで、Directにも体軸の接近がある。同じD2領域に両胸が入る件数は420→418で増えていない。各bodyの2node以上が同じD2領域に入る件数は290→323だが、merged/split/反射を含む1maskを1魚と見なせない。これらを二重説明の確定検出器にしない。

折れ線体長はDirect全slotで20.00～30.03mm、Fresnelで20.05～28.92mm。今回の差を巨大な体長伸長だけでは説明できない。S0/S1には両方式とも名目z=0未満が多く、Direct計946、Fresnel940 node-framesである。工学下限-6mm未満は0、S1は両方式で-6mmに達する。これは既存座標・床の不確実性が共通して残ることを示し、今回の反射モデルだけを原因としたり、床位置を新しく確定したりする根拠ではない。

支持低下について現在支持できる仮説は、画像目的の変更後にS0/S1の別nodeへの対応競合と一部の姿勢接近が生じた、という範囲である。損失の重み、反射面、レビュー条件を後付け変更して今回の不合格を救済しない。最終backtraceは未来の選択を含み、最初の保存差からオンラインの因果を逆算しない。

根拠: 上位フォルダの `summary.json`、`all_pair_frames.csv`、`all_fish_frames.csv`、`fixed_frame_details.json`、各armの `*_geometry.npz`。再fit・再描画・原画decode・旧系列座標・人手GTは使用していない。
"""
    output = ROOT / "interpretation_v1"
    output.mkdir(exist_ok=False)
    with provenance(
        "interpret_reflection_temporal_geometry", output, name="interpretation"
    ) as record:
        for path, digest in bindings.items():
            record.input(path, path, sha256=digest)
        audit.producer.save_json(output / "summary.json", result, record)
        path = output / "RESULT.md"
        path.write_text(text, encoding="utf-8")
        record.output(path.name, path, sha256=audit.sha256(path))
        audit.producer.snapshot(
            {str(Path(__file__).resolve()): audit.sha256(__file__)}, output, record
        )
        audit.producer.guard(bindings, {}, record)
    print(json.dumps(dict(status="ok", output=str(output))))


if __name__ == "__main__":
    main()

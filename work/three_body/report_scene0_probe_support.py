"""Small sealed report for geometric probe readiness; no human-label generation."""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import audit_scene0_probe_support as a

from medaka3d.provenance import provenance


def main():
    root = a.m.REPO / "work/three_body/scene0_probe_support_audit_v1"
    out = root / "report_v1"
    if out.exists():
        raise FileExistsError("report exists")
    seal_path = root / "audit.provenance.json"
    seal = json.loads(seal_path.read_text())
    a.m.require(seal["status"] == "ok", "audit incomplete")
    hashes = {
        str(seal_path): a.m.sha256(seal_path),
        str(Path(__file__).resolve()): a.m.sha256(__file__),
    }
    for item in seal["outputs"].values():
        a.m.require(a.m.sha256(item["path"]) == item["sha256"], "changed audit output")
        hashes[item["path"]] = item["sha256"]
    s = json.loads((root / "summary.json").read_text())
    lines = [
        "# scene0の3提案フレーム: 口の幾何支持",
        "",
        "79316 / 79405 / 79480は、persistentとmemoryの双方で3slotすべての口が2視点以上の直接支持を満たします。",
        "memoryはこのclip全体でP=[0,1,2]のため、同じ幾何・支持です。人手の個体対応や連続性の確認は未実施です。",
        "",
        "| frame | raw slot | 支持view | 最大ray距離 mm | 最大再投影誤差 px | 元review |",
        "|---:|---:|---|---:|---:|---|",
    ]
    for probe in s["probes"]:
        for row in probe["persistent_raw_slots"]:
            names = [
                view
                for view, valid in zip(a.m.VIEWS, row["direct_view_gate"], strict=True)
                if valid
            ]
            ray = max(x for x in row["assigned_ray_distance_mm_by_view"] if x is not None)
            error = max(
                x for x in row["assigned_reprojection_error_pixels_by_view"] if x is not None
            )
            lines.append(
                f"| {probe['frame']} | S{row['slot']} | {', '.join(names)} | "
                f"{ray:.3f} | {error:.3f} | {row['original_identity_review']} |"
            )
    lines += [
        "",
        "全9口について3viewへの投影は有限かつ画像内ですが、支持なしviewをdirectとして扱っていません。",
        "特に79480のS0/S2はtop未支持で、side1+side2だけが支持を持ちます。元のreviewはそのままです。",
        "",
        "同じclip [79236,79488) の252frameを元の支持関数で再計算し、全3口directは137frameでした。",
        "該当する半開区間は [79237,79291), [79305,79306), [79316,79317), [79403,79481), [79485,79488) です。",
        "これは幾何条件の列挙であり、新しい推奨probeや人手確認済みframeではありません。",
        "",
        "この検査で満たすのは、保存SLEAP行の排他・有限性、元float32割当rayの距離2 mm以内、",
        "屈折再投影12 px以内、少なくとも2viewという既存条件です。",
        "人がクリックした口への5 px内の一意対応、実際の同じ動物への対応、イベントを跨ぐ連続性は判定していません。",
        "そのため3提案は支持不足で直ちに除外する必要はありませんが、クリックなしにhuman correctionを確定する材料もありません。",
        "人手が選ぶframe・口座標・各動物の対応と連続性が必要です。",
        "",
        "完成済み2runとreview pageの来歴を照合し、memory export全13配列を原geometryの並替えからbytes一致で再現しました。",
        "テンプレートはSHA保護だけで、正解としてdeserializeしていません。CPUのみ、追加原画decode・GT・ページ変更なしです。",
        "4 CPU tests/Ruffと独立read-onlyレビューを通過。詳細は[summary](../summary.json)と[view別CSV](../probe_view_support.csv)です。",
    ]
    out.mkdir(parents=True)
    with provenance("probe_readiness_report", out, name="report") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        path = out / "source.py"
        shutil.copyfile(__file__, path)
        record.output("source", path, sha256=a.m.sha256(path))
        path = out / "RESULT.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        record.output("result", path, sha256=a.m.sha256(path))
        a.m.require(
            all(a.m.sha256(path) == digest for path, digest in hashes.items()), "input changed"
        )
        a.m.require(
            all(a.m.sha256(item["path"]) == item["sha256"] for item in record.outputs.values()),
            "report changed",
        )
        record.param("all_input_output_sha_reverified", True)


if __name__ == "__main__":
    main()

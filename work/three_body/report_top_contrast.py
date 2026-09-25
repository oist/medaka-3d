"""Post-hoc arithmetic and image-artifact QA for a completed observation diagnostic."""

# Long literals below are Japanese report paragraphs and Markdown table rows.
# ruff: noqa: E501

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import measure_top_contrast as d

from medaka3d.provenance import provenance


def main():
    root = d.m.REPO / "work/three_body/top_contrast_13frames_v1"
    output = root / "report_v1"
    if output.exists():
        raise FileExistsError("report exists")
    hashes = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        d.m.require(path.is_relative_to(d.m.REPO), "input escaped workspace")
        digest = d.m.sha256(path)
        d.m.require(expected is None or digest == expected, "input checksum mismatch")
        hashes[str(path)] = digest
        return path

    for path in set(d.m.code_paths()) | {Path(d.__file__), Path(__file__)}:
        bind(path)
    seal = json.loads(bind(root / "contrast.provenance.json").read_text())
    d.m.require(seal["status"] == "ok", "measurement not complete")
    for item in seal["outputs"].values():
        bind(item["path"], item["sha256"])
    summary = json.loads((root / "summary.json").read_text())
    sequence = Path(summary["source_sequence"])
    manifest = json.loads(
        bind(sequence / "manifest.json", summary["source_manifest_sha256"]).read_text()
    )
    mask_info = manifest["masks"][d.VIEW]
    packed = d.m.identity_inputs._npz(bind(sequence / mask_info["path"], mask_info["sha256"]))
    box = d.m.identity_inputs._npz(root / "ray_box_footprint.npz")
    png_checks, frame_checks = [], []
    for row in summary["measurements"]:
        frame = row["frame_idx"]
        t = frame - manifest["frames"][0]
        a = d.m.identity_inputs._npz(root / f"f{frame}_observations.npz")
        masks = d.m.decode_frame_masks(packed, t)
        contrast, union = a["contrast"], masks.any(axis=0)
        d.m.require(np.array_equal(a["d2_union"], union), "D2 union mismatch")
        for key, mask in (
            ("contrast", contrast),
            ("d2_union", union),
            ("intersection", contrast & union),
            ("contrast_only", contrast & ~union),
            ("d2_only", union & ~contrast),
        ):
            expected = dict(
                total=int(mask.sum()),
                engineering_box=int(mask[box["engineering"]].sum()),
                physical_box=int(mask[box["physical"]].sum()),
                outside_engineering_box=int(mask[box["valid"] & ~box["engineering"]].sum()),
                invalid_ray=int(mask[~box["valid"]].sum()),
            )
            d.m.require(row[key] == expected, f"{frame}/{key} counts mismatch")
            d.m.require(
                expected["total"]
                == expected["engineering_box"]
                + expected["outside_engineering_box"]
                + expected["invalid_ray"],
                "partition does not conserve mass",
            )
        for i, instance in enumerate(row["instances"]):
            overlap = int(contrast[masks[i]].sum())
            d.m.require(
                instance["contrast_overlap_pixels"] == overlap, "instance intersection differs"
            )
        labels = a["contrast_components"]
        for i, component in enumerate(row["contrast_components"]):
            mask = labels == i + 1
            expected = [int(other[mask].sum()) for other in masks]
            d.m.require(component["area"] == int(mask.sum()), "component area differs")
            d.m.require(
                component["d2_instance_overlap_pixels"] == expected,
                "component instance overlaps differ",
            )
            d.m.require(
                np.array_equal(a["component_instance_overlap_pixels"][i], expected),
                "saved overlap matrix differs",
            )
        d.m.require(
            row["native_offset"] == t and row["pos_before"] == t and row["pos_after"] == t + 1,
            "native position mismatch",
        )
        d.m.require(
            row["raw_bgr_sha256"] == manifest["raw"][d.VIEW]["frame_bgr_sha256"][t],
            "native SHA record mismatch",
        )
        for kind in ("full", "detail"):
            path = root / f"f{frame}_{kind}.png"
            image = cv2.imread(str(path))
            d.m.require(
                image is not None and image.dtype == np.uint8 and image.shape[1] == 1800,
                "invalid PNG",
            )
            expected_height = (
                664
                if kind == "full"
                else 64
                + round(
                    (row["detail_crop_xyxy"][3] - row["detail_crop_xyxy"][1])
                    * 600
                    / (row["detail_crop_xyxy"][2] - row["detail_crop_xyxy"][0])
                )
            )
            d.m.require(image.shape == (expected_height, 1800, 3), "wrong panel dimensions")
            png_checks.append(dict(file=path.name, shape=image.shape))
        frame_checks.append(
            dict(
                frame=frame,
                native_sha_and_position=True,
                full_counts_and_partitions_exact=True,
                per_instance_and_component_counts_exact=True,
            )
        )
    output.mkdir(parents=True)
    with provenance("top_contrast_post_measurement_qa", output, name="report") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        snapshot = output / "source.py"
        shutil.copyfile(__file__, snapshot)
        record.output("source", snapshot, sha256=d.m.sha256(snapshot))
        qa = dict(
            status="ok",
            parent_outputs_checked=len(seal["outputs"]),
            native_top_images_redecoded_in_this_report=0,
            producer_native_top_images_verified=13,
            frame_checks=frame_checks,
            png_decode_checks=png_checks,
            visual_inspection=[
                "f79322_detail.png",
                "f79381_detail.png",
                "f79372_full.png",
                "f82720_detail.png",
            ],
            background_source_frames=summary["background_frames"],
            gt_read=False,
            detector_or_tracker_changed=False,
        )
        path = output / "QA.json"
        path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
        record.output("QA", path, sha256=d.m.sha256(path))
        lines = [
            "# 固定コントラスト抽出とD2の上面画像比較",
            "",
            "コントラスト抽出はD2が捉えていない魚体らしい領域を拾いますが、反射像も拾い、複数の像が連結します。今回の固定設定をそのまま3匹の検出器へ置き換える根拠にはなりません。",
            "",
            "- [79322 原画・D2・コントラスト](../f79322_detail.png): 中央の曲がった像にはD2マスクがありません。コントラストでは見えますが、下側の像や左側の反射らしい像と14,070画素の同じ連結成分になります。D2の2マスクの94.76% / 98.77%はコントラストと重なります。画像からの診断であり、個体数・IDの正解ラベルではありません。",
            "- [79381 原画・D2・コントラスト](../f79381_detail.png): 下側の反射らしいD2 m3も98.98%、その侵食coreは100%コントラスト内です。4つのD2マスクはすべて同じ16,022画素のコントラスト連結成分に重なります。コントラストとの一致を反射棄却や魚の存在確率に使えません。",
            "- [79372 全画面](../f79372_full.png)には反射のほか、壁付近や小さい変化も抽出されています。[82720 拡大](../f82720_detail.png)でも、重なる像はコントラストで連結します。",
            "",
            "全13フレームとも、D2の全画素から出る屈折rayが旧工学boxと交差します。79381の反射候補m3も100%内側です。今回の画素ではz下限0と-6 mmのfootprintは同じでした。無限平面屈折モデルにおける前向きray/box交差は、反射・遮蔽・実際の可視性を扱わず、今回の反射を除く検査にはなりません。box外のコントラスト画素も除去せず別に保存しています。",
            "",
            "| frame | D2 masks | contrast components | D2 union px | contrast px | overlap px | contrast-only px | contrast outside box px |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary["measurements"]:
            lines.append(
                f"| {row['frame_idx']} | {row['d2_instance_count']} | {row['retained_contrast_component_count']} | {row['d2_union']['total']} | {row['contrast']['total']} | {row['intersection']['total']} | {row['contrast_only']['total']} | {row['contrast']['outside_engineering_box']} |"
            )
        lines += [
            "",
            "既存observations.foreground_contrastを直接実行しました。各BGRチャネルの全画面中央値で背景差を補正し、正負両方向の絶対差12以上、8連結・6画素以上という元の規則は不変です。背景は固定候補9枚から窓内80000を除いた既存8枚median PNGで、今回再学習・閾値選択・背景の再作成はしていません。全componentとD2 instance別の重なり、各チャネルの中央値補正、box内外/invalid別の面積は[元summary](../summary.json)と[CSV](../counts.csv)へ保存しています。",
            "",
            "13原画はnative166fpsの同じframe番号をseekし、前後POSとBGR SHAを照合、D2のframe列と復号マスクSHAも照合済みです。これは保存データの対応確認であり、カメラのハードウェア同期や未知の時刻ずれを独立に測定したものではありません。元の72出力SHA、26PNGの復号/寸法、全画素・instance/component集計を本QAで再確認しました。原画の再デコードは主計測だけで行い、QAでは記録を照合しています。CPUのみで、GT・追跡pose/ID配列は読み込んでいません。既存モデル・推論・レビュー出力は不変です。",
            "",
            "次の候補としては、D2にないコントラスト領域からも空間仮説を出し、反射やmerged領域を別の説明として残して3視点で比べる方法が考えられます。これは設計案に留まります。コントラスト領域のunionを正解マスクにしたり、D2との一致率でIDを確定したりする変更は今回行っていません。",
        ]
        path = output / "RESULT.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        record.output("result", path, sha256=d.m.sha256(path))
        d.m.require(
            all(d.m.sha256(path) == digest for path, digest in hashes.items()),
            "report input/source changed",
        )
        d.m.require(
            all(d.m.sha256(item["path"]) == item["sha256"] for item in record.outputs.values()),
            "report output changed",
        )
        record.param("all_input_output_sha_reverified", True)
        print(
            json.dumps(
                dict(
                    status="ok",
                    parent_outputs=len(seal["outputs"]),
                    frames=len(frame_checks),
                    pngs=len(png_checks),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

"""Record read-only visual findings from the separately sealed diagnostic stills."""
# ruff: noqa: E501, RUF001

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"))
from prepare_sequence import sha256

from medaka3d.provenance import provenance


def main():
    root = Path(__file__).resolve().parent / "minute_scene0_model_diagnostic"
    summary_path, seal_path = root / "summary.json", root / "diagnostic.provenance.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal["status"] != "ok":
        raise ValueError("diagnostic image generation did not complete")
    for item in seal["outputs"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("diagnostic output differs from its completed seal")
    frames = {item["frame_idx"]: item for item in summary["frame_diagnostics"]}
    transition = frames[79372]
    inspected = [
        *(
            f"f{frame}_{view}_pair.png"
            for frame in (79370, 79372)
            for view in ("top_cam", "side1_cam", "side2_cam")
        ),
        "f79316_top_cam_pair.png",
        "f79318_top_cam_pair.png",
        "f79322_top_cam_pair.png",
        "f79355_side1_cam_pair.png",
        "f79547_top_cam_pair.png",
        "f79547_side1_cam_pair.png",
    ]
    findings = dict(
        status="diagnostic_observations_recorded_not_identity_truth",
        inspected_pair_images=inspected,
        visual_method="view_image of local PNG raw/overlay pairs; no browser or GPU",
        transition=dict(
            previous_candidate_frame=79371,
            current_frame=79372,
            s0_pectoral_latent_step_mm=transition["latent_node_step_mm"][0][1],
            s0_mouth_latent_step_mm=transition["latent_node_step_mm"][0][0],
            s0_previous_observed=transition["previous_observed"][0],
            s0_current_observed=transition["observed"][0],
            s1_current_observed=transition["observed"][1],
            current_review=transition["identity_review"],
            proposal_kind=transition["proposal_kind"],
            interpretation="repositioning of an unsupported latent body on recovery; not a directly measured fish displacement",
        ),
        observations=[
            dict(
                frames=[79318, 79322, 79370],
                view="top_cam",
                finding="The red S0 centerline progressively crosses space outside the visible fish body; at 79370 it bridges the gap between visible bodies.",
            ),
            dict(
                frames=[79370, 79372],
                view="three views",
                finding="After recovery the red and green hypotheses remain close in overlapping body regions, especially in the side views; these stills do not establish identity continuity.",
            ),
            dict(
                frames=[79547],
                view="top_cam",
                finding="A cyan detector contour also covers a lower reflected-image region; detector count three is not evidence of three isolated physical fish.",
            ),
        ],
        d2_counts_79316={
            view: data["d2_instances"] for view, data in frames[79316]["views"].items()
        },
        observed_79547=frames[79547]["observed"],
        review_79547=frames[79547]["identity_review"],
        identity_assignment_made=False,
        human_truth_created=False,
        candidate_modified=False,
        caveats=[
            "Saved observed flags are multi-view node support, not per-camera ground truth.",
            "D2 masks may miss fish, combine visible bodies, or cover reflections.",
            "Native crops have moving model-derived centers; use full context for spatial comparisons.",
            "The raw-only human review pack is separate and unchanged.",
        ],
    )
    report = f"""# Scene0 の原画と追跡候補の診断

79372 の S0 の大きな変化は、**観測支持を失った推定姿勢の再配置**だった。
79371→79372 の保存済み胸ヒレ位置の差は {transition["latent_node_step_mm"][0][1]:.3f} mm、
口位置の差は {transition["latent_node_step_mm"][0][0]:.3f} mm。
直前の S0 は 3 ノードとも直接支持なし、79372 は胸ヒレ・尾だけが支持され、
口は未支持のまま。`observation_recovery` が選択され、S0 と S1 の ID review は維持されている。
この値を実個体が 1 フレームで泳いだ距離とは解釈しない。

| 原画との比較 | 所見 |
| --- | --- |
| [79318 top](f79318_top_cam_pair.png)、[79322 top](f79322_top_cam_pair.png) | 赤 S0 の線が曲がった魚体から外れ、未支持ノードが増える |
| [79370 top](f79370_top_cam_pair.png) | 赤 S0 の線が、見えている魚体間の隙間を横切る |
| [79372 top](f79372_top_cam_pair.png)、[side1](f79372_side1_cam_pair.png)、[side2](f79372_side2_cam_pair.png) | 再配置後も赤・緑の仮説は重なり領域で近接し、ID 対応の確定には足りない |
| [79547 top](f79547_top_cam_pair.png) | 下側の反射像の領域にも cyan の検出輪郭がある。3 検出を 3 実個体とは数えられない |

79316 のマスク数は top=1、side1=1、side2=4 だった。
同フレームの保存された 3 匹×3 ノードの支持は揃っていたため、D2 の個体分離と
ノード支持は同じ品質指標ではない。79547 では再び全ノードに支持があるが、
S0/S1 の ID review は解除されていない。正しい生物学的対応は、この診断では付与しない。

指定された 9 フレーム×3 視点だけを FFV1 から直接 seek し、native 位置と BGR SHA を
確認した。全景 9 枚と、各視点の raw・overlay・対比 crop を合わせて PNG 90 枚を保存。
各 crop は S0/S1 の投影胸ヒレ平均を中心とした native 320×320 で、画像外は明示した
padding。中心はフレームごとに動くため、位置比較には各 `fNNNNN_context.png` も使う。

色は任意の保存 slot。cyan は D2、実線は latent pose、塗りつぶしノードは保存された
複数視点での支持、白抜きは未支持、`?` は ID review。
塗りつぶしは、その表示カメラ単独での正しさを意味しない。

画像生成の [summary.json](summary.json) と [diagnostic.provenance.json](diagnostic.provenance.json)、
目視所見の [visual_diagnosis.json](visual_diagnosis.json) を参照。
原画だけの人手レビュー用パックと候補 NPZ は変更していない。GPU やブラウザは使用していない。
"""
    for name in ("visual_diagnosis.json", "RESULT.md", "visual.provenance.json"):
        if (root / name).exists():
            raise FileExistsError("visual diagnosis already exists; no overwrite")
    with provenance("visual_diagnosis_minute_scene0", root, name="visual") as record:
        for path in (Path(__file__), summary_path, seal_path, *(root / name for name in inspected)):
            record.input(str(path), path, sha256=sha256(path))
        path = root / "visual_diagnosis.json"
        path.write_text(json.dumps(findings, indent=2, allow_nan=False), encoding="utf-8")
        record.output("visual_diagnosis", path, sha256=sha256(path))
        path = root / "RESULT.md"
        path.write_text(report, encoding="utf-8")
        record.output("result", path, sha256=sha256(path))
    print(
        json.dumps(
            dict(
                result_sha256=sha256(root / "RESULT.md"),
                visual_provenance_sha256=sha256(root / "visual.provenance.json"),
            )
        )
    )


if __name__ == "__main__":
    main()

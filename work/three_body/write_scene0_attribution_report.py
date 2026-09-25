"""Seal a read-only interpretation and spatial views of measured scene0 evidence."""
# ruff: noqa: E501, RUF001

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "labs/20260909_three_body"))
from prepare_sequence import decode_frame_masks, sha256  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

ROOT = REPO / "work/three_body/minute_scene0_silhouette_attribution"
RAW = REPO / "work/three_body/minute_scene0_model_diagnostic"
SEQUENCE = REPO / "work/three_body/inputs/continuous_one_minute_76620-86580"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_seal(path):
    data = read(path)
    if data["status"] != "ok":
        raise ValueError(f"Incomplete input: {path}")
    for item in data["outputs"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Changed sealed output: {item['path']}")
    return data


def spatial_figure(frame, raw_diagnostics, arrays, masks_by_view, output):
    views = ("top_cam", "side1_cam")
    palette = [(255, 100, 220), (250, 210, 30), (50, 220, 250), (180, 250, 50)]
    fig, axes = plt.subplots(2, 6, figsize=(19.2, 7.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.55")
    for row, view in enumerate(views):
        raw = cv2.imread(str(RAW / f"f{frame}_{view}_raw_crop.png"))
        if raw is None or raw.shape != (320, 320, 3):
            raise ValueError("Raw crop is missing or not native 320-square")
        xyxy = raw_diagnostics["views"][view]["crop"]["requested_box_xyxy"]
        x0, y0, x1, y1 = xyxy
        if min(x0, y0) < 0 or max(x1, y1) > 1200:
            raise ValueError("This small report only plots fully in-image crops")
        instances = masks_by_view[view][:, y0:y1, x0:x1]
        annotated = raw.copy()
        for index, mask in enumerate(instances):
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            color = palette[index % len(palette)]
            cv2.drawContours(annotated, contours, -1, color, 1)
            yy, xx = np.nonzero(mask)
            if len(xx):
                location = (min(275, max(2, int(xx.mean()))), min(305, max(16, int(yy.mean()))))
                cv2.putText(
                    annotated,
                    f"D{index}",
                    location,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"D{index}",
                    location,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        alpha = arrays[f"{view}_alpha"]
        values = (alpha[:, 0], alpha[:, 1], alpha[:, 0] * alpha[:, 1], 1 - arrays[f"{view}_union"])
        axes[row, 0].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        axes[row, 1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        valid = arrays[f"{view}_ray_valid"]
        flat = arrays[f"{view}_foreground_flat"]
        for col, value in enumerate(values, start=2):
            dense = np.full(1200 * 1200, np.nan)
            dense[flat[valid]] = value[valid]
            axes[row, col].imshow(
                dense.reshape(1200, 1200)[y0:y1, x0:x1], cmap=cmap, vmin=0, vmax=1
            )
        for col, title in enumerate(
            (
                "Raw crop",
                "D2 instance indices",
                "S0 alpha",
                "S1 alpha",
                "S0 alpha * S1 alpha",
                "1 - union",
            )
        ):
            axes[row, col].set_title(f"{view}: {title}", fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.colorbar(
        ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap),
        ax=axes[:, 2:].ravel().tolist(),
        shrink=0.9,
        label="Soft response on valid D2 foreground pixels; grey = outside measured foreground",
    )
    fig.suptitle(
        f"Source {frame}: saved pose, no refit. D0... are detector indices, not animal IDs. Same native 320 x 320 crops as prior diagnostic.",
        fontsize=12,
    )
    fig.savefig(output, dpi=120)
    plt.close(fig)
    if cv2.imread(str(output)) is None:
        raise ValueError("Spatial figure decode failed")


def main():
    names = (
        "RESULT.md",
        "findings.json",
        "f79372_spatial_attribution.png",
        "report.provenance.json",
    )
    if any((ROOT / name).exists() for name in names):
        raise FileExistsError("Report already exists; no overwrite")
    seal = verify_seal(ROOT / "attribution.provenance.json")
    summary = read(ROOT / "summary.json")
    raw_seal = verify_seal(RAW / "diagnostic.provenance.json")
    frames = {item["frame_idx"]: item for item in summary["measurements"]}
    if summary["status"] != "complete_diagnostic_not_identity_ground_truth":
        raise ValueError("Measurement did not complete")
    if (
        not summary["all_original_bridge_flags_reproduced"]
        or summary["max_abs_saved_loss_difference"] > 1e-12
    ):
        raise ValueError("Original sampled objective/support flags were not reproduced")
    inspected = [ROOT / f"f{f}_attribution.png" for f in (79370, 79372, 79547)]
    inputs = {Path(item["path"]): item["sha256"] for item in seal["outputs"].values()}
    inputs.update({Path(item["path"]): item["sha256"] for item in seal["inputs"].values()})
    inputs.update(
        {
            p: sha256(p)
            for p in (
                Path(__file__),
                ROOT / "attribution.provenance.json",
                RAW / "summary.json",
                RAW / "diagnostic.provenance.json",
            )
        }
    )
    raw_output_hashes = {
        Path(item["path"]).resolve(): item["sha256"] for item in raw_seal["outputs"].values()
    }
    for view in ("top_cam", "side1_cam"):
        path = RAW / f"f79372_{view}_raw_crop.png"
        inputs[path] = raw_output_hashes[path.resolve()]
    if not all(sha256(path) == digest for path, digest in inputs.items()):
        raise ValueError("A report input or frozen source changed")
    with provenance("report_scene0_silhouette_attribution", ROOT, name="report") as record:
        for path, digest in inputs.items():
            record.input(str(path), path, sha256=digest)
        raw_frame = next(
            item
            for item in read(RAW / "summary.json")["frame_diagnostics"]
            if item["frame_idx"] == 79372
        )
        masks = {}
        for view in ("top_cam", "side1_cam"):
            with np.load(SEQUENCE / f"masks/{view}.npz", allow_pickle=False) as data:
                index = int(np.flatnonzero(data["frame_idx"] == 79372)[0])
                masks[view] = decode_frame_masks(data, index)
        path = ROOT / "f79372_spatial_attribution.png"
        with np.load(ROOT / "f79372_responses.npz", allow_pickle=False) as data:
            spatial_figure(79372, raw_frame, data, masks, path)
        record.output("spatial_attribution", path, sha256=sha256(path))
        m70, m72, m81, m547 = (frames[f] for f in (79370, 79372, 79381, 79547))
        findings = dict(
            status="measured_local_diagnostic_not_identity_truth",
            conclusion="Overlapping support of S0/S1 and substantially unexplained detector foreground coexist with passing original image-support gates; this does not establish which detector regions are real fish.",
            source_candidate_sha256=summary["candidate_sha256"],
            top_79372_same_region=m72["full_foreground"]["top_cam"]["instances"][2],
            top_79372_other_region=m72["full_foreground"]["top_cam"]["instances"][0],
            side1_79372_other_region=m72["full_foreground"]["side1_cam"]["instances"][2],
            side1_79372_instance_iou=m72["full_foreground"]["side1_cam"]["instance_iou"],
            whole_foreground_pair01_mass={
                str(f): {
                    v: frames[f]["full_foreground"][v]["foreground_union_summary"][
                        "pair_shared_mass_px"
                    ][0]
                    for v in ("top_cam", "side1_cam", "side2_cam")
                }
                for f in (79370, 79372)
            },
            removal_image_only={
                str(f): frames[f]["sampled_metrics"]["remove_body_delta_loss_by_view"]
                for f in (79370, 79372)
            },
            all_fish_all_views_bridge_at_79372=bool(
                np.asarray(m72["sampled_metrics"]["original_bridge_by_view_fish"]).all()
            ),
            unexplained_top_79381=m81["full_foreground"]["top_cam"]["instances"][3],
            unexplained_top_79547=m547["full_foreground"]["top_cam"]["instances"][2],
            unexplained_side1_79547=m547["full_foreground"]["side1_cam"]["instances"][3],
            inspected_heatmaps=[str(path) for path in inspected],
            spatial_figure="Native crops from separately verified raw diagnostic; continuous fields only at full foreground rays, no new render or optimization. Figure decode checked; visual inspection is recorded separately after generation.",
            max_saved_loss_error=summary["max_abs_saved_loss_difference"],
            max_visibility_conservation_error=max(
                m["conservation_max_abs_error"] for m in frames.values()
            ),
            instance_records=sum(
                len(v["instances"]) for m in frames.values() for v in m["full_foreground"].values()
            ),
            foreground_rays=sum(
                v["valid_foreground_rays"]
                for m in frames.values()
                for v in m["full_foreground"].values()
            ),
            interpretation_limits=[
                "A detector instance may merge animals, duplicate another instance, or cover reflection.",
                "Per-frame detector indices are not temporal identities and are not matched across frames.",
                "Overlap is amodal alpha product, not visible-alpha sharing or proof of duplicate animals.",
                "Unexplained is continuous soft mass, not a thresholded count of missing fish.",
                "Removing a body changes only this fixed-pixel image loss; full tracker node, motion and shape objectives are not evaluated.",
                "No alternative shape model was fitted, so a shape-model failure is not causally isolated.",
                "All fish review/accepted flags, candidate files and separate human-review pack remain unchanged.",
            ],
        )
        top2 = findings["top_79372_same_region"]
        top0 = findings["top_79372_other_region"]
        side2 = findings["side1_79372_other_region"]
        rows = []
        for f in summary["frames"]:
            m = frames[f]
            counts = "/".join(
                str(m["full_foreground"][v]["detector_instances"])
                for v in ("top_cam", "side1_cam", "side2_cam")
            )
            delta = m["sampled_metrics"]["mean_view_remove_body_delta_loss"]
            rows.append(
                f"| {f} | {counts} | {m['sampled_metrics']['mean_view_loss']:.4f} | {delta[0]:+.4f} | {delta[1]:+.4f} | {delta[2]:+.4f} |"
            )
        report = f"""# Scene0 の画像支持の内訳

79372 では、S0/S1 が同じ前景領域を重ねて説明し、別の検出領域がほぼ未説明のまま残っていた。
それでも既存の画像支持 bridge は 3 匹×3 視点すべてで通った。
これは画像支持が個体の排他的な対応を保証しない例であり、ID の誤りや検出領域の実個体性を確定する結果ではない。

| 79372 の領域 | 面積 | S0 平均 alpha | S1 平均 alpha | 平均 union | 未説明の soft mass / 面積 |
| --- | ---: | ---: | ---: | ---: | ---: |
| top 検出2 | {top2["native_area_px"]} px | {top2["mean_alpha"][0]:.4f} | {top2["mean_alpha"][1]:.4f} | {top2["union_mean"]:.4f} | {top2["unexplained_mean"]:.2%} |
| top 検出0 | {top0["native_area_px"]} px | {top0["mean_alpha"][0]:.4f} | {top0["mean_alpha"][1]:.4f} | {top0["union_mean"]:.4f} | {top0["unexplained_mean"]:.2%} |
| side1 検出2 | {side2["native_area_px"]} px | {side2["mean_alpha"][0]:.4f} | {side2["mean_alpha"][1]:.4f} | {side2["union_mean"]:.4f} | {side2["unexplained_mean"]:.2%} |

top 検出2には S0 と S1 の両胸ヒレが入り、S0 の口と S1 の尾も入る。
S0/S1 の alpha 積の平均は {top2["pair_shared_mean"][0]:.4f}。
両者の unique union 寄与も {top2["unique_mean"][0]:.4f} / {top2["unique_mean"][1]:.4f} あり、完全に同一の説明ではない。
top 検出0には S0 の尾だけが入る。side1 検出1には S0/S1 の全6ノードが入るが、
この検出は未説明の検出2と IoU {m72["full_foreground"]["side1_cam"]["instance_iou"][1][2]:.3f}、
検出3と IoU {m72["full_foreground"]["side1_cam"]["instance_iou"][1][3]:.3f} で重なる。
検出をそのまま個体に一対一対応させる解釈はできない。包含は座標とマスクの関係であり、正解ラベルではない。

[79372 の原画・検出境界・空間応答](f79372_spatial_attribution.png) と
[全検出の比較図](f79372_attribution.png) を参照。
空間応答の灰色は測定対象の D2 前景外、数値は連続値で、新しい二値しきい値は加えていない。

79370→79372 の全前景上の S0/S1 alpha 積の総和は、top で
{m70["full_foreground"]["top_cam"]["foreground_union_summary"]["pair_shared_mass_px"][0]:.2f}→{m72["full_foreground"]["top_cam"]["foreground_union_summary"]["pair_shared_mass_px"][0]:.2f} px相当、
side1 で {m70["full_foreground"]["side1_cam"]["foreground_union_summary"]["pair_shared_mass_px"][0]:.2f}→{m72["full_foreground"]["side1_cam"]["foreground_union_summary"]["pair_shared_mass_px"][0]:.2f} px相当へ増えた。
これは別々のフレームの各前景を測った値で、同一画素集合や検出IDを時間対応させた差ではない。

一方、S0 を除去すると、79370 の top / side1 の画像損失は
{m70["sampled_metrics"]["remove_body_delta_loss_by_view"][0][0]:+.5f} / {m70["sampled_metrics"]["remove_body_delta_loss_by_view"][1][0]:+.5f} と改善するが、
side2 の支持を失うため3視点平均では {m70["sampled_metrics"]["mean_view_remove_body_delta_loss"][0]:+.5f} と悪化する。
79372 は3視点すべてで除去により悪化し、平均差は {m72["sampled_metrics"]["mean_view_remove_body_delta_loss"][0]:+.5f}。
単に S0 が余分という結果ではない。除去差には形状・運動・SLEAP の項を含めていない。

| frame | D2 数 top/side1/side2 | 元画像損失 | 除去 Δ S0 | 除去 Δ S1 | 除去 Δ S2 |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

79381 top 検出3（1855 px）、79547 top 検出2（3607 px）と side1 検出3（2344 px）は、
計算された union が領域全体で0だった。79547 では全9ノードが直接支持されていてもこの状態が残る。
原画には反射像もあるため、未説明面積を未追跡の魚の数とは換算しない。
同様に、soft alpha の重なりは実際の遮蔽でも起きる。body 形状の変更や別の割当を試していないため、
原因を体形モデルの不足だけに限定することもできない。

測定は保存姿勢に対する CPU float64 の再投影だけで、最適化・追跡・ID変更を行っていない。
元の384前景+384近傍+256全画面の固定sampleを同じframe seedで再構成した。
D2 mode の sampler は RGB の値を参照しないため、native shapeを持つdummy画像を使用した。
これは同 mode の黒/白画像で配列が完全一致する単体検証を伴い、原画像の省略を一般化しない。
全前景は {findings["foreground_rays"]:,} 有効ray、検出ごとの記録は {findings["instance_records"]}。
検出同士の重なりは二重計上しうるため、全前景集計には重複を除いた union 画素を用いた。

保存済みの画像損失との差は最大 {summary["max_abs_saved_loss_difference"]:.3g}、元 bridge 判定は全一致、
sum(visible alpha)=union の誤差は最大 {findings["max_visibility_conservation_error"]:.3g}。
5つの解析的CPUテストと Ruff が通過した。全入力・コード・出力SHAを確認し、
測定実行は {summary["runtime"]["elapsed_s"]:.2f} 秒だった（追跡全体の速度ではない）。

定義・全数値は [summary.json](summary.json)、[instances.csv](instances.csv)、
固定sampleと全前景の応答配列は各 `fNNNNN_responses.npz` に保存。
生成の [attribution.provenance.json](attribution.provenance.json) は不変のまま、
この解釈と空間図は別の [report.provenance.json](report.provenance.json) で封印した。
既存 candidate と原画のみの人手レビューpackは変更していない。
"""
        path = ROOT / "findings.json"
        path.write_text(json.dumps(findings, indent=2, allow_nan=False), encoding="utf-8")
        record.output("findings", path, sha256=sha256(path))
        path = ROOT / "RESULT.md"
        path.write_text(report, encoding="utf-8")
        record.output("result", path, sha256=sha256(path))
        if not all(sha256(path) == digest for path, digest in inputs.items()):
            raise ValueError("A report input or frozen source changed during reporting")
    print(json.dumps({name: sha256(ROOT / name) for name in names}))


if __name__ == "__main__":
    main()

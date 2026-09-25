"""Separate post-measurement interpretation seal; no renderer or tracking calls."""
# ruff: noqa: E501, RUF001 -- Report prose and mathematical symbols.

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import measure_persistent_silhouette as m

from medaka3d.provenance import provenance


def main():
    run = m.REPO / "work/three_body/curve_body_spatial_13frames_v1"
    report = run / "report_v1"
    out = report / "interpretation"
    if out.exists():
        raise FileExistsError("interpretation exists")
    inputs = {}

    def checked(path, expected=None):
        digest = m.sha256(path)
        m.require(expected is None or expected == digest, "changed interpretation source")
        inputs[str(path)] = digest
        return path

    checked(Path(__file__).resolve())
    seal = json.loads(checked(run / "curve.provenance.json").read_text())
    qa_seal = json.loads(checked(report / "report.provenance.json").read_text())
    m.require(seal["status"] == qa_seal["status"] == "ok", "incomplete measurement/QA")
    summary = json.loads(
        checked(run / "summary.json", seal["outputs"]["summary"]["sha256"]).read_text()
    )
    qa = json.loads(
        checked(report / "QA.json", qa_seal["outputs"]["QA.json"]["sha256"]).read_text()
    )
    frames = summary["measurements"]
    changes = np.stack(
        [
            np.asarray(f["spatial_evaluation_curve"]["loss_by_stratum"])[0]
            - np.asarray(f["spatial_evaluation_original"]["loss_by_stratum"])[0]
            for f in frames
        ]
    )
    view_changes = changes.sum(axis=-1)
    focused = {}
    for frame in (79372, 79381, 82719, 82720):
        row = next(f for f in frames if f["frame_idx"] == frame)
        focused[frame] = dict(
            offset_mm=row["offset_mm"],
            arc_chord_ratio=row["curve_diagnostics"]["arc_chord_ratio"],
            per_instance=row["per_instance"],
            evaluation_delta_by_view=view_changes[list(m.FRAMES).index(frame)],
        )
    data = dict(
        status="diagnostic_only",
        sampled_frame_view_pairs=39,
        improved_view_pairs=int(np.sum(view_changes < 0)),
        worsened_view_pairs=int(np.sum(view_changes > 0)),
        unchanged_view_pairs=int(np.sum(view_changes == 0)),
        mean_evaluation_relative_reduction=1
        - qa["mean_evaluation_after"] / qa["mean_evaluation_before"],
        foreground_worsened_view_pairs=int(np.sum(changes[..., 0] > 0)),
        background_spill_worsened_view_pairs=int(np.sum(changes[..., 1:].sum(-1) > 0)),
        total_fit_seconds=sum(f["fit_seconds"] for f in frames),
        total_frame_seconds=sum(f["frame_seconds"] for f in frames),
        all_curve_degenerate_false=all(
            not np.any(f["curve_diagnostics"]["degenerate"]) for f in frames
        ),
        all_gaussian_nondegenerate_true=all(np.all(f["gaussian_nondegenerate"]) for f in frames),
        maximum_arc_chord_ratio=max(
            np.max(f["curve_diagnostics"]["arc_chord_ratio"]) for f in frames
        ),
        max_arc_quadrature_difference_mm=max(
            np.max(f["curve_diagnostics"]["arc_quadrature_difference_mm"]) for f in frames
        ),
        eval_delta_by_frame_view=view_changes,
        focused=focused,
        model_or_threshold_changed_after_results=False,
        original_identity_reviews_cleared=False,
    )
    prose = f"""# 曲率対照の解釈

3視点平均では全13時点で空間評価誤差が下がりましたが、個別には **{data["worsened_view_pairs"]}/39 視点・時点で悪化**しています。平均の相対低下は {100 * data["mean_evaluation_relative_reduction"]:.2f}% です。同じD2領域から抽出した別画素での結果であり、汎化精度やID維持率ではありません。

- **79372の主要な説明不足は解消しませんでした。** top評価誤差は+.00307、side1は+.01233悪化、side2の-.02204改善で平均が下がっています。top m0の未説明割合は93.61→93.45%、side1 m2は99.39→99.53%。top m2のS0/S1共有α積の平均は .186376→.186345 とほぼ不変です。別マスク内部の説明不足が、今回の曲率だけで解決したとは言えません。
- **79381は大きく改善する例です。** 空間評価平均は .50812→.40342。top m1のcore説明Uは .250→.555、m2は .281→.583まで増えます。一方m3は元も曲線もほぼ0のままです。S2のarc/chordは1.319なので、曲がりだけでなく弧長の増加を許して得た結果として扱います。
- **82720は制約に到達する例です。** S1のoffset normは17.508mmでchord上限に達し、arc/chordは1.479。side1空間評価は+.00645悪化しています。side2 m1のcore Uは .424→.368へ低下し、m3は .415→.488へ増加しています。別の領域への説明の移動も含むため、平均改善を元の無支持3D形状の回復と呼びません。
- box制約は79394のS0で10更新、79396のS0で77更新に作用。norm制約は82720のS1で17更新に作用しました。保存最良結果に退化はなく、Gaussianの軸/半径σは全て元と厳密一致です。arc積分64/32点差の最大は {data["max_arc_quadrature_difference_mm"]:.3g}mmで、誤差上界ではありません。

このB対照は、柔らかい身体表現が一部の画像残差を減らすことを示す一方、79372の説明不足には十分ではありません。次は事前登録した独立mask-volume初期候補Aと同じ画素・同じ損失で比較し、候補探索不足、固定ノードの不適切さ、採点の問題を切り分ける材料にします。Bの失敗は曲線モデル全般の棄却ではなく、弧長を固定するモデルも未検証です。元の3D座標・支持・review・受理IDは変えていません。

13時点のfit計は {data["total_fit_seconds"]:.2f}秒、前処理/評価/保存を含むframe計は {data["total_frame_seconds"]:.2f}秒でした。前後の全入力SHA I/Oは含まず、連続追跡の速度でもありません。39視点行・全13図・算術QAは親フォルダに、詳細数値は `interpretation.json` に保存しています。図は再デコード後に目視確認し、全13のラベルと元/曲線の系列が見えることを確認しました。
"""
    out.mkdir()
    with provenance("curve_body_postmeasurement_interpretation", out, name="interpretation") as rec:
        for path, digest in inputs.items():
            rec.input(path, path, sha256=digest)
        shutil.copyfile(__file__, out / "source.py")
        (out / "INTERPRETATION.md").write_text(prose, encoding="utf-8")
        (out / "interpretation.json").write_text(
            json.dumps(m.safe(data), indent=2, allow_nan=False), encoding="utf-8"
        )
        for path in out.iterdir():
            rec.output(path.name, path, sha256=m.sha256(path))
        m.require(
            all(m.sha256(path) == digest for path, digest in inputs.items()),
            "interpretation input changed",
        )
        m.require(
            all(m.sha256(item["path"]) == item["sha256"] for item in rec.outputs.values()),
            "interpretation output changed",
        )
    print(
        json.dumps(
            {k: v for k, v in data.items() if k not in ("focused", "eval_delta_by_frame_view")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

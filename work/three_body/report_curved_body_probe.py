"""Independent arithmetic QA of the completed fixed-node spatial curve diagnostic."""
# ruff: noqa: RUF001, E501 -- Japanese report prose, not source identifiers.

import argparse
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run, out = args.run.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError("report output exists")
    m.require(run.is_relative_to(m.REPO) and out.is_relative_to(m.REPO), "workspace paths required")
    hashes = {}

    def bind(path, expected=None):
        digest = m.sha256(path)
        m.require(expected is None or expected == digest, f"changed report input: {path}")
        hashes[str(path)] = digest
        return path

    bind(Path(__file__).resolve())
    seal = json.loads(bind(run / "curve.provenance.json").read_text())
    m.require(
        seal["status"] == "ok" and seal["stage"] == "fixed_node_curve_spatial_diagnostic",
        "curve diagnostic incomplete",
    )
    for item in seal["outputs"].values():
        bind(Path(item["path"]), item["sha256"])
    summary = json.loads((run / "summary.json").read_text())
    m.require(summary["frames"] == list(m.FRAMES), "wrong fixed frame set")
    candidate_path = Path(summary["source_run"]) / "candidate.npz"
    bind(candidate_path, summary["candidate_sha256"])
    with np.load(candidate_path, allow_pickle=False) as arrays:
        original = {
            key: arrays[key]
            for key in ("frame_idx", "pose_xyz", "sources", "observed", "identity_review")
        }
    rows = []
    max_fit_error, max_eval_error = 0.0, 0.0
    for frame_info in summary["measurements"]:
        frame = frame_info["frame_idx"]
        t = int(np.flatnonzero(original["frame_idx"] == frame)[0])
        with np.load(run / f"f{frame}_pixels.npz", allow_pickle=False) as value:
            pixels = {k: value[k] for k in value.files}
        with np.load(run / f"f{frame}_curve.npz", allow_pickle=False) as value:
            result = {k: value[k] for k in value.files}
        for key, original_key in (
            ("pose_xyz", "pose_xyz"),
            ("sources", "sources"),
            ("observed", "observed"),
            ("prior_identity_review", "identity_review"),
        ):
            a, b = result[key], original[original_key][t]
            m.require(
                a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
                "saved fixed geometry/support/review changed",
            )
        iteration = frame_info["best_iteration"]
        m.require(
            result["loss_history"][iteration] == np.min(result["loss_history"]),
            "best finite fit iterate not selected",
        )
        m.require(
            np.array_equal(result["best_offset_mm"], result["offset_history_mm"][iteration]),
            "selected offset/history mismatch",
        )
        pose, offset = result["pose_xyz"], result["best_offset_mm"]
        chord = pose[:, 2] - pose[:, 1]
        length = np.linalg.norm(chord, axis=-1)
        controls = (pose[:, 1] + pose[:, 2]) / 2 + offset
        m.require(
            np.max(np.abs(np.sum(chord * offset, axis=-1))) < 1e-9, "offset is not perpendicular"
        )
        m.require(
            np.all(np.linalg.norm(offset, axis=-1) <= length + 1e-10),
            "offset norm violates chord limit",
        )
        m.require(
            np.all(controls >= np.array([0, 0, -6]) - 1e-10)
            and np.all(controls <= np.array([192, 192, 186]) + 1e-10),
            "control outside box",
        )
        for v, view in enumerate(m.VIEWS):
            fitting = pixels["fit_uv"][v]
            evaluation = pixels[view + "_eval_uv"]
            fit_flat = fitting[:, 1].astype(int) * 1200 + fitting[:, 0].astype(int)
            eval_flat = evaluation[:, 1].astype(int) * 1200 + evaluation[:, 0].astype(int)
            m.require(
                not np.intersect1d(fit_flat, eval_flat).size, "evaluation overlaps fitting pixels"
            )
            labels = pixels["eval_strata"][v, : len(evaluation)]
            for k in range(3):
                indices = eval_flat[labels == k]
                m.require(
                    len(np.unique(indices)) == len(indices), "replacement within evaluation stratum"
                )
        parts = {}
        for kind in ("original", "curve"):
            loss = (
                (result[kind + "_fit_union"] - pixels["fit_target"]) ** 2
                * pixels["fit_weights"]
                * result["fit_ray_valid"]
            )
            fit_parts = loss.sum(axis=1)
            fit_error = abs(fit_parts.mean() - frame_info["fit_" + kind]["mean_view_loss"])
            max_fit_error = max(max_fit_error, float(fit_error))
            error = (
                (result[kind + "_eval_union"] - pixels["eval_target"]) ** 2
                * pixels["eval_weights"]
                * result["eval_ray_valid"]
            )
            strata_parts = np.stack(
                [(error * (pixels["eval_strata"] == k)).sum(axis=1) for k in range(3)], axis=-1
            )
            saved = np.asarray(frame_info["spatial_evaluation_" + kind]["loss_by_stratum"])[0]
            max_eval_error = max(max_eval_error, float(np.max(np.abs(strata_parts - saved))))
            parts[kind] = (fit_parts, strata_parts)
        for v, view in enumerate(m.VIEWS):
            before, after = parts["original"][1][v], parts["curve"][1][v]
            rows.append(
                dict(
                    frame_idx=frame,
                    view=view,
                    fit_original=parts["original"][0][v],
                    fit_curve=parts["curve"][0][v],
                    eval_original=before.sum(),
                    eval_curve=after.sum(),
                    eval_delta=after.sum() - before.sum(),
                    foreground_delta=after[0] - before[0],
                    background_delta=after[1:].sum() - before[1:].sum(),
                    best_iteration=iteration,
                    status=frame_info["status"],
                )
            )
    m.require(max_fit_error < 1e-12 and max_eval_error < 1e-12, "saved score arithmetic differs")
    before_fit = np.array([f["fit_original"]["mean_view_loss"] for f in summary["measurements"]])
    after_fit = np.array([f["fit_curve"]["mean_view_loss"] for f in summary["measurements"]])
    before_eval = np.array(
        [f["spatial_evaluation_original"]["mean_view_loss"][0] for f in summary["measurements"]]
    )
    after_eval = np.array(
        [f["spatial_evaluation_curve"]["mean_view_loss"][0] for f in summary["measurements"]]
    )
    qa = dict(
        status="ok",
        curve_outputs_rehashed=len(seal["outputs"]),
        all_fixed_geometry_support_review_byte_equal=True,
        all_evaluation_pixels_exclude_all_fit_pixels=True,
        best_finite_iterates_match=True,
        max_fit_arithmetic_error=max_fit_error,
        max_evaluation_arithmetic_error=max_eval_error,
        fit_improved_frames=int(np.sum(after_fit < before_fit)),
        evaluation_improved_frames=int(np.sum(after_eval < before_eval)),
        evaluation_worsened_frames=int(np.sum(after_eval > before_eval)),
        evaluation_unchanged_frames=int(np.sum(after_eval == before_eval)),
        mean_fit_before=float(before_fit.mean()),
        mean_fit_after=float(after_fit.mean()),
        mean_evaluation_before=float(before_eval.mean()),
        mean_evaluation_after=float(after_eval.mean()),
        complete_optimizer_frames=sum(f["status"] == "completed" for f in summary["measurements"]),
        gt_or_baseline_bytes_opened=False,
    )
    out.mkdir(parents=True)
    with provenance("fixed_node_curve_report_qa", out, name="report") as rec:
        for path, digest in hashes.items():
            rec.input(path, path, sha256=digest)
        shutil.copyfile(__file__, out / "report_source.py")
        (out / "QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
        with (out / "per_view.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
        x = np.arange(13)
        for ax, first, second, title in (
            (axes[0], before_fit, after_fit, "Original fitting pixels"),
            (axes[1], before_eval, after_eval, "Spatial evaluation pixels (same D2 frame)"),
        ):
            ax.plot(x, first, "o-", label="Fixed two-segment body")
            ax.plot(x, second, "o-", label="Best curve by fitting loss")
            ax.set_xticks(x, m.FRAMES, rotation=70)
            ax.set_title(title)
            ax.set_ylabel("Weighted image error, mean over three cameras")
            ax.legend(fontsize=8)
        fig.savefig(out / "all_frames.png", dpi=150)
        plt.close(fig)
        m.require(cv2.imread(str(out / "all_frames.png")) is not None, "plot decode failed")
        table = "\n".join(
            f"| {frame} | {a:.5f} | {b:.5f} | {c:.5f} | {d:.5f} |"
            for frame, a, b, c, d in zip(
                m.FRAMES, before_fit, after_fit, before_eval, after_eval, strict=True
            )
        )
        prose = f"""# 固定3ノード・曲率のみの対照

最適化画素では {qa["fit_improved_frames"]}/13 フレーム、別の空間評価画素では {qa["evaluation_improved_frames"]}/13 フレームの画像誤差が下がりました。空間評価で悪化したのは {qa["evaluation_worsened_frames"]}/13 です。これは同じD2マスクの画像説明の比較で、IDや3D位置の正しさを示しません。

| frame | fit元 | fit曲率 | 空間評価元 | 空間評価曲率 |
|---|---:|---:|---:|---:|
{table}

各時刻の3ノード位置、幅、Gaussianの軸/半径スケール、source/observed/旧reviewは固定です。胸尾間の二次Bezier制御offsetだけをゼロから100 Adam steps、lr .25mmで最適化し、ゼロを含む最良有限fit iterateを選びました。offsetは胸尾chordに垂直で、normはchord以下、controlは旧box内に単一scalarで縮小しています。曲線の弧長は保存していません。

空間評価は最適化前に保存し、全fit座標を除いたforeground/ring/backgroundから固定seedで768/768/512以下を非復元抽出しました。層ごとの重みは1/.75/.25です。評価値を候補選択や重み調整に使っていません。反射・merged mask・未知の検出誤りが残り、真値による評価ではありません。

全39視点行のforeground/background別の変化は [per_view.csv](per_view.csv)、全13フレームの図は [all_frames.png](all_frames.png) に保存しました。元/曲線の全instance・core寄与、offset履歴、arc/chord、退化と境界制約は親runのsummary/各NPZにあります。

独立再計算で保存fit/eval算術の最大差は {max_fit_error:.3g}/{max_eval_error:.3g}、全fit/eval座標は非重複、source/observed/review/元poseの保存配列は元候補とbyte一致しました。zero curveと元renderer、固定Gaussian scalesの厳密一致は実測driver内で確認しました。{qa["complete_optimizer_frames"]}/13 が100step完了しています。曲率モデル全般を採用/棄却する実験ではなく、固定された誤った可能性のある3Dノードも修復しません。新しい受理IDや追跡結果は出していません。
"""
        (out / "RESULT.md").write_text(prose, encoding="utf-8")
        for path in out.iterdir():
            rec.output(path.name, path, sha256=m.sha256(path))
        m.require(
            all(m.sha256(path) == digest for path, digest in hashes.items()),
            "report inputs changed",
        )
        m.require(
            all(m.sha256(item["path"]) == item["sha256"] for item in rec.outputs.values()),
            "report outputs changed",
        )
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()

"""Saved-output diagnosis after the independent 200-frame audit; no new fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import audit_reflection_temporal as audit
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work/three_body"
RUN = WORK / "reflection_temporal_200_v1"
AUDIT = WORK / "reflection_temporal_200_independent_audit_v1"
DRAFT = WORK / "REFLECTION_AFTER_200_DECISION_DRAFT.md"
DRAFT_SHA = "7e72c4d7c9f1e372acaf7b55f62074131f4cc66d275b11c8de29307f26ef1ff4"
NODES = ("mouth", "pectoral", "tail")


def first_frame(flags, frames):
    where = np.flatnonzero(flags)
    return int(frames[where[0]]) if len(where) else None


def frame_details(data, t):
    result = {
        key: data[key][t].tolist()
        for key in (
            "pose_xyz",
            "sources",
            "observed",
            "identity_review",
            "missing",
            "node_missing",
            "mask_supported",
            "body_supported",
            "loss_parts",
            "costs",
            "score_margin",
            "proposal_kind",
            "hypotheses",
            "proposal_flow_nodes",
            "proposal_flow_parent",
        )
    }
    margin = float(data["score_margin"][t])
    result["score_margin_status"] = "finite" if np.isfinite(margin) else str(margin)
    if not np.isfinite(margin):
        result["score_margin"] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    audit.require(
        output.is_relative_to(ROOT) and not output.exists(), "new workspace output required"
    )
    bindings = {}
    producer = audit.producer
    producer.bind(DRAFT, DRAFT_SHA, bindings)
    producer.bind(Path(__file__), None, bindings)
    seal = producer.verify_outputs(
        AUDIT, "audit.provenance.json", "independent_reflection_temporal_output_audit", bindings
    )
    producer.output_path(seal, "summary.json", AUDIT / "summary.json")
    for item in seal["inputs"].values():
        if Path(item["path"]).suffix == ".py":
            producer.bind(item["path"], item["sha256"], bindings)
    checked = audit.read_json(AUDIT / "summary.json")
    audit.require(
        checked["passed"] is True and checked["frame_range"] == [79236, 79436],
        "completed matching independent audit required",
    )
    producer.bind(RUN / "experiment.provenance.json", checked["producer_seal_sha256"], bindings)
    for arm, digest in checked["candidate_sha256"].items():
        producer.bind(RUN / f"{arm}_candidate.npz", digest, bindings)
    result_seal = audit.read_json(RUN / "experiment.provenance.json")
    for role in ("comparison", "evaluation_metrics", "summary"):
        item = result_seal["outputs"][role]
        producer.bind(item["path"], item["sha256"], bindings)
    data = {arm: audit.arrays(RUN / f"{arm}_candidate.npz") for arm in audit.ARMS}
    direct, fresnel = data["direct"], data["fresnel"]
    frames = direct["frame_idx"]
    metrics = audit.arrays(RUN / "evaluation_metrics.npz")
    comparison = audit.read_json(RUN / "comparison.json")
    matrix = metrics["mean_view_errors"].mean(0)
    audit.close(matrix, checked["matrix"], "audited matrix")
    audit.require(
        comparison["gate"]["passed"] is False, "this diagnosis is for the failed fixed gate"
    )
    gained = fresnel["observed"] & ~direct["observed"]
    lost = direct["observed"] & ~fresnel["observed"]
    node_distance = np.linalg.norm(fresnel["pose_xyz"] - direct["pose_xyz"], axis=-1)
    same_scorer_delta = metrics["mean_view_errors"][:, 1, 1] - metrics["mean_view_errors"][:, 0, 1]
    parts = metrics["per_view_stratum_errors"]
    loss_first = first_frame(lost[:, :, 1].any(1), frames)
    selected = np.abs(frames - 79372) <= 16
    if loss_first is not None:
        selected |= np.abs(frames - loss_first) <= 16
    node_changes = {}
    for slot in range(3):
        node_changes[f"slot_{slot}"] = {
            name: dict(
                gained_count=int(gained[:, slot, node].sum()),
                lost_count=int(lost[:, slot, node].sum()),
                gained_ranges=audit.runs(gained[:, slot, node], frames),
                lost_ranges=audit.runs(lost[:, slot, node], frames),
            )
            for node, name in enumerate(NODES)
        }
    rows = []
    for t in np.flatnonzero(selected):
        rows.append(
            dict(
                frame=int(frames[t]),
                maximum_node_distance_mm=float(node_distance[t].max()),
                same_scorer_loss_delta=float(same_scorer_delta[t]),
                arms={arm: frame_details(data[arm], t) for arm in audit.ARMS},
            )
        )
    summary = dict(
        schema="reflection_temporal_failed_gate_diagnosis_v1",
        independent_audit_seal_sha256=audit.sha256(AUDIT / "audit.provenance.json"),
        draft_sha256=DRAFT_SHA,
        fixed_gate=comparison["gate"],
        continue_reflection_to_666=False,
        reference_long_tracking_unchanged=True,
        node_changes=node_changes,
        matrix=matrix.tolist(),
        same_scorer_mean_delta=float(same_scorer_delta.mean()),
        same_scorer_per_view_delta=(parts[:, 1, 1] - parts[:, 0, 1]).sum(-1).mean(0).tolist(),
        same_scorer_per_view_stratum_delta=(parts[:, 1, 1] - parts[:, 0, 1]).mean(0).tolist(),
        first_direct_supported_fresnel_unsupported_chest_frame=loss_first,
        first_saved_pose_difference_frame=first_frame((node_distance > 1e-12).any((1, 2)), frames),
        first_saved_source_difference_frame=first_frame(
            (direct["sources"] != fresnel["sources"]).any((1, 2, 3)), frames
        ),
        diagnostic_frames=frames[selected].tolist(),
        all_200_scored=True,
        longest_gap_per_slot={
            arm: comparison["statistics"][arm]["longest_chest_gap_per_slot"] for arm in audit.ARMS
        },
        direct_chest={
            arm: comparison["statistics"][arm]["direct_chest_total"] for arm in audit.ARMS
        },
        accepted_chest={
            arm: comparison["statistics"][arm]["accepted_chest_total"] for arm in audit.ARMS
        },
        saved_parts_order=["node", "shape", "center", "repulsion", "silhouette"],
        biological_ID_truth=False,
        new_raw_decode=False,
        refit=False,
        rerender=False,
        limitations=[
            "These are final backtraces. Earliest differences need not be the first "
            "causal online divergence.",
            "Direct support gains can attach a different biological fish; "
            "no independent ID truth is present.",
            "Training silhouette losses across models are not an equivalent "
            "image-scoring comparison.",
            "Higher accepted totals can follow delayed sticky review; "
            "they do not outweigh missing direct support.",
        ],
    )
    output.mkdir(parents=True, exist_ok=False)
    with provenance("diagnose_failed_reflection_temporal_gate", output, name="diagnosis") as record:
        for path, digest in bindings.items():
            record.input(path, path, sha256=digest)
        producer.save_json(output / "summary.json", summary, record)
        producer.save_json(output / "diagnostic_frames.json", rows, record)
        path = output / "all_frame_changes.npz"
        np.savez_compressed(
            path,
            frame_idx=frames,
            gained=gained,
            lost=lost,
            node_distance_mm=node_distance,
            same_scorer_delta=same_scorer_delta,
            direct_score_margin=direct["score_margin"],
            fresnel_score_margin=fresnel["score_margin"],
        )
        record.output(path.name, path, sha256=audit.sha256(path))
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        raster = np.concatenate([data[a]["observed"][:, :, 1].T for a in audit.ARMS])
        axes[0].imshow(
            raster,
            aspect="auto",
            interpolation="nearest",
            cmap="Greys",
            extent=[frames[0] - 0.5, frames[-1] + 0.5, 5.5, -0.5],
            vmin=0,
            vmax=1,
        )
        axes[0].set_yticks(range(6), [f"{a} S{s}" for a in audit.ARMS for s in range(3)])
        axes[0].set_title("Directly supported pectoral node (black = supported)")
        for slot in range(3):
            axes[1].plot(frames, node_distance[:, slot].max(1), label=f"S{slot}")
        axes[1].set_ylabel("Max node difference (mm)")
        axes[1].legend(ncol=3)
        axes[2].plot(frames, same_scorer_delta, color="#a33")
        axes[2].axhline(0, color="black", linewidth=0.7)
        axes[2].set_ylabel("Fresnel-fit minus direct-fit\nFresnel evaluation loss")
        axes[2].set_xlabel("Native frame (166 fps)")
        for ax in axes:
            ax.axvline(79372, color="#168", linestyle="--", alpha=0.6)
        path = output / "comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        record.output(path.name, path, sha256=audit.sha256(path))
        text = (
            "# 200フレーム反射モデルの判定\n\n"
            "**反射モデルの延長・採用は見送る。** 独立監査は通過したが、"
            "事前の性能条件を満たさない。"
            "途中の再初期化は両方式とも0。全200フレームを比較した。\n\n"
            f"胸ヒレ直接支持は{summary['direct_chest']['direct']}/600から"
            f"{summary['direct_chest']['fresnel']}/600へ減った。"
            f"同じFresnel評価式での画像誤差差は{same_scorer_delta.mean():+.8f} (正は悪化)。"
            f"最大欠測長は{summary['longest_gap_per_slot']['direct']}から"
            f"{summary['longest_gap_per_slot']['fresnel']}フレームになった。\n\n"
            "受理数の合計は375から390へ増えたが、slot0は1減り、slot1のreview開始が"
            "79328から79371へ遅れた。直接支持の減少と長い欠測を改善と扱わない。"
            "これは全3個体の正解IDとの比較ではない。\n\n"
            f"DirectにあってFresnelにない胸支持が最初に現れる保存時点は{loss_first}。"
            "その前後16フレームと79372の前後16フレームの姿勢・source・支持・損失・"
            "proposalを保存した。最終backtraceの最初の差は、オンラインの因果的な"
            "分岐点を確定するものではない。支持の増加と喪失は全slot・全node・全区間を列挙した。\n\n"
            "![全200フレームの比較](comparison.png)\n\n"
            "別の反射パラメータを試して今回の判定を救済しない。30分の参照追跡は元の"
            "凍結方式で続ける。次の改善仮説は、独立検出の重複・競合とIDレビューの"
            "対応を調べた上で別に固定する。\n"
        )
        path = output / "RESULT.md"
        path.write_text(text, encoding="utf-8")
        record.output(path.name, path, sha256=audit.sha256(path))
        producer.guard(bindings, {}, record)
    print(json.dumps(dict(output=str(output), gate_passed=False)))


if __name__ == "__main__":
    main()

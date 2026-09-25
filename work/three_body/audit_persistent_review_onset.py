"""Bounded, post-completion explanation of the first persistent S0 review at 78331."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))

from run_experiment import sha256  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    folder = REPO / "work/three_body"
    out = folder / "persistent_mouth_minute_independent_audit_v1"
    run = folder / "persistent_mouth_continuous_one_minute_v1"
    old = folder / "v1_continuous_one_minute_detectron2"
    comparison = folder / "persistent_mouth_continuous_one_minute_comparison_v1"
    inputs = {}
    for root, name in ((run, "experiment"), (old, "experiment"),
                       (comparison, "evaluation"), (out, "audit")):
        record_path = root / f"{name}.provenance.json"
        record = read(record_path)
        if record["status"] != "ok":
            raise ValueError("all source runs and the independent audit must be complete")
        inputs[str(record_path)] = sha256(record_path)
        for key in (("candidate", "summary") if root in (run, old)
                    else ("comparison",) if root == comparison else ("summary",)):
            entry = record["outputs"][key]
            path = Path(entry["path"])
            path = path if path.is_absolute() else REPO / path
            digest = sha256(path)
            if digest != entry["sha256"]:
                raise ValueError("sealed source output changed")
            inputs[str(path)] = digest
    script = Path(__file__).resolve()
    inputs[str(script)] = sha256(script)
    code = run / "code_snapshot/labs/20260909_three_body/persistent_node_tracker.py"
    expected = {k.replace("\\", "/"): v for k, v in read(run / "summary.json")[
        "code_sha256_before"
    ].items()}["labs/20260909_three_body/persistent_node_tracker.py"]
    if sha256(code) != expected:
        raise ValueError("run's review rule snapshot is not bound to its recorded SHA")
    inputs[str(code)] = expected
    names = ["review_onset.json", "REVIEW_ONSET.md", "review_onset.provenance.json",
             "review_onset_code_snapshot"]
    if any((out / name).exists() for name in names):
        raise FileExistsError("refusing to overwrite review-onset report")
    with provenance("persistent_first_review_onset_diagnostic", out, name="review_onset") as rec:
        for path, digest in inputs.items():
            rec.input(path, path, sha256=digest)
        timeline = {}
        for label, root in (("persistent", run), ("original_v1", old)):
            with np.load(root / "candidate.npz", allow_pickle=False) as saved:
                frames = saved["frame_idx"]
                selection = (frames >= 78325) & (frames < 78336)
                arrays = {k: saved[k][selection] for k in (
                    "frame_idx", "sources", "observed", "body_supported", "mask_supported",
                    "missing", "identity_review", "score_margin", "proposal_kind",
                )}
            if not np.array_equal(arrays["frame_idx"], np.arange(78325, 78336)):
                raise ValueError("onset interval is not native and continuous")
            rows = []
            for t, frame in enumerate(arrays["frame_idx"]):
                margin = float(arrays["score_margin"][t])
                rows.append({
                    "frame_idx": int(frame),
                    "observed_s0_mouth_pectoral_tail": arrays["observed"][t, 0].tolist(),
                    "source_s0_view_node": arrays["sources"][t, 0].tolist(),
                    "head_ray_count_s0": int((arrays["sources"][t, 0, :, :2] >= 0).sum()),
                    "body_supported_s0": bool(arrays["body_supported"][t, 0]),
                    "mask_supported_s0": bool(arrays["mask_supported"][t, 0]),
                    "missing_s0": int(arrays["missing"][t, 0]),
                    "review_slots": arrays["identity_review"][t].tolist(),
                    "score_margin": margin if np.isfinite(margin) else None,
                    "score_margin_positive_infinity": bool(np.isposinf(margin)),
                    "proposal_kind": str(arrays["proposal_kind"][t]),
                })
            timeline[label] = rows
        config = read(run / "summary.json")["tracker_config"]
        gap, ambiguity = config["review_gap_frames"], config["ambiguity_margin"]
        rows = {item["frame_idx"]: item for item in timeline["persistent"]}
        if (rows[78330]["review_slots"][0] or not rows[78331]["review_slots"][0]
                or [rows[f]["missing_s0"] for f in range(78328, 78332)] != [1, 2, 3, 4]
                or gap != 4 or any(rows[f]["head_ray_count_s0"] for f in range(78328, 78332))
                or rows[78331]["score_margin"] <= ambiguity):
            raise ValueError("the stated body-loss review explanation is contradicted")
        result = {
            "status": "complete_bounded_trigger_diagnostic", "frames": [78325, 78336],
            "timeline": timeline, "review_gap_frames": gap, "ambiguity_margin": ambiguity,
            "onset_frame": 78331, "onset_slot": 0,
            "mechanism": "four consecutive body-unsupported frames reached the missing threshold",
            "onset_low_identity_margin": False,
            "mask_support_alone_is_insufficient_without_any_mouth_or_pectoral_ray": True,
            "limitations": [
                "This identifies a recorded review trigger, not an identity error.",
                "Lower accepted coverage alone is not an identity regression.",
                "No raw images, detector inference, new labels or threshold changes are used.",
            ],
        }
        path = out / "review_onset.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        rec.output("onset", path, sha256=sha256(path))
        text = (
            "# S0の最初の確認保留 - フレーム78331\n\n"
            "直接の発火条件は、78328〜78331の4フレーム連続の身体支持欠落でした。"
            "missingが1、2、3、4と増え、登録済み閾値4に到達しました。"
            "78331の候補差は29.2287406571で、曖昧さ閾値1による発火ではありません。\n\n"
            "この4フレームは輪郭支持ありですが、S0の口・胸ヒレの観測線が全視点で0本です。"
            "身体支持には直接支持された胸ヒレ、または輪郭支持と少なくとも1本の頭部観測線が"
            "必要なので、輪郭だけでは欠落を補えません。78332で3つの推定点が直接支持へ戻っても"
            "IDの確認保留は維持されます。\n\n"
            "旧v1は78329で観測回復して欠落カウンタが0に戻り、この時点では保留しません。"
            "これだけで旧v1のIDが正しい、新方式のIDが悪化したとは判断できません。"
            "詳細な11フレームの観測元と状態はreview_onset.jsonに保存しました。\n"
        )
        path = out / "REVIEW_ONSET.md"
        path.write_text(text, encoding="utf-8")
        rec.output("report", path, sha256=sha256(path))
        snapshot = out / "review_onset_code_snapshot"
        snapshot.mkdir()
        for source in (script, code):
            target = snapshot / source.name
            shutil.copyfile(source, target)
            rec.output("code:" + source.name, target, sha256=sha256(target))
        if any(sha256(path) != digest for path, digest in inputs.items()):
            raise RuntimeError("review-onset input or code changed")
    print(json.dumps({"output": str(out), "onset": 78331, "trigger": "missing>=4"}))


if __name__ == "__main__":
    main()

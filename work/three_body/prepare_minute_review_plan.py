"""Prepare a review-time proposal only after the frozen minute comparison seals."""

import json
import math
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from evaluate_minute import load_completed  # noqa: E402
from review_plan import build_review_plan  # noqa: E402
from run_experiment import sha256  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402


def main():
    root = REPO / "work/three_body"
    comparison = root / "v1_continuous_one_minute_comparison"
    output = root / "v1_continuous_one_minute_review_plan"
    if output.exists():
        raise FileExistsError(output)
    record_path = comparison / "evaluation.provenance.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "ok":
        raise ValueError("complete frozen comparison required")
    hashes = {str(record_path): sha256(record_path)}
    for name in ("comparison", "review_events"):
        path = comparison / f"{name}.json"
        hashes[str(path)] = sha256(path)
        if record["outputs"][name]["sha256"] != hashes[str(path)]:
            raise ValueError("comparison output differs from its seal")
    code = [Path(__file__), LAB / "review_plan.py", LAB / "review_events.py",
            LAB / "evaluate_minute.py", LAB / "run_experiment.py",
            REPO / "src/medaka3d/provenance.py"]
    hashes.update({str(path): sha256(path) for path in code})
    arrays, summary, run_hashes = load_completed(
        root / "v1_continuous_one_minute_detectron2")
    hashes.update(run_hashes)
    gap = summary["tracker_config"]["review_gap_frames"]
    margin = summary["tracker_config"]["ambiguity_margin"]
    if (type(gap) is not int or gap != 4 or type(margin) not in (int, float)
            or not math.isfinite(margin) or margin != 1.0):
        raise ValueError("fixed v1 review settings required")
    run_path = root / "v1_continuous_one_minute_detectron2/candidate.npz"
    if not any(Path(item["path"]).resolve() == run_path.resolve()
               and item["sha256"] == sha256(run_path)
               for item in record["inputs"].values()):
        raise ValueError("comparison does not reference this exact candidate")
    plan = build_review_plan(arrays)
    compared = json.loads((comparison / "comparison.json").read_text(encoding="utf-8"))
    if plan["risk_counts"] != compared["new_method_review_counts"]:
        raise ValueError("review counts differ from the frozen comparison")
    plan["source_candidate_sha256"] = sha256(run_path)
    plan["comparison_sha256"] = hashes[str(comparison / "comparison.json")]
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError("source changed before review proposal output")
    output.mkdir(parents=True, exist_ok=False)
    with provenance("minute_review_time_proposals", output, name="review_plan") as saved:
        for path, digest in hashes.items():
            saved.input(path, path, sha256=digest)
        for path in code:
            target = output / "source_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        path = output / "plan.json"
        path.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
        saved.output("plan", path, sha256=sha256(path))
        lines = ["# 1分結果の原画像確認候補", "",
                 "モデルが記録した不確実な場面を全て残した一覧です。人のID確認・修正は未実施です。",
                 "候補時点の口には直接観測の支持がありますが、対応の正しさは原画像で確認が必要です。",
                 "", "| scene | 原画像範囲 (終了は含まない) | 提案3時点 | 入口にID保留 | 状態 |",
                 "|---|---|---|---|---|"]
        for scene in plan["scenes"]:
            probes = ", ".join("未選択" if f is None else str(f)
                               for f in scene["proposed_anchor_frames"])
            lines.append(
                f"| {scene['scene_id']} | {scene['context_start_frame']}:"
                f"{scene['context_stop_frame_exclusive']} | {probes} | "
                f"{scene['entry_contains_unresolved_machine_id']} | {scene['status']} |")
        lines += ["", "未選択の時点や未解決な入口も分母から除いていません。",
                  "場面数は必要な人手や修正可能性の実測値ではなく、未検出のスワップも否定しません。",
                  "原のcandidateとID保留は変更していません。"]
        path = output / "RESULT.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        saved.output("result", path, sha256=sha256(path))
        if any(sha256(path) != digest for path, digest in hashes.items()):
            raise ValueError("source changed while saving review proposals")
    print(json.dumps({key: plan[key] for key in (
        "proposed_scene_count", "incomplete_scene_count", "risk_counts")}))


if __name__ == "__main__":
    main()

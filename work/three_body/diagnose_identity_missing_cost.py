"""Explain the observed short-run objective reduction without asserting ID truth."""
import json
import sys
from pathlib import Path

import numpy as np

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "labs/20260909_three_body"))
import run_experiment as base

from medaka3d.provenance import provenance

run = repo / "work/three_body/identity_dp_transient_200_v1"
output = repo / "work/three_body/identity_dp_transient_200_missing_cost_v1"
if output.exists():
    raise FileExistsError(output)
record_path = run / "experiment.provenance.json"
sealed = json.loads(record_path.read_text())
if sealed["status"] != "ok":
    raise ValueError("a completed inference stage is required")
hashes = {str(record_path): base.sha256(record_path), str(Path(__file__)): base.sha256(__file__)}
for item in sealed["outputs"].values():
    if base.sha256(item["path"]) != item["sha256"]:
        raise ValueError("changed completed output")
    hashes[item["path"]] = item["sha256"]
data = {}
for role in ("scores", "dp", "identity"):
    with np.load(run / f"{role}.npz", allow_pickle=False) as saved:
        data[role] = {key: saved[key] for key in saved.files}
s, d, g = (data[key] for key in ("scores", "dp", "identity"))
frames, state = d["frame_idx"], d["path_state"]
rows = []
for t in np.flatnonzero(state):
    row = {
        "frame": int(frames[t]), "mapping": d["permutation_by_id"][t].tolist(),
        "raw_direct_pectoral": g["raw_observed"][t, :, 1].tolist(),
        "all_six_image_costs": s["unary"][t].tolist(),
    }
    for name, edge in (("incoming", t - 1), ("outgoing", t)):
        if not 0 <= edge < len(frames) - 1:
            raise ValueError("diagnostic requires internal changes with both neighboring edges")
        a, b = state[edge:edge + 2]
        row[name] = {
            "fixed_identity_cost": float(s["transitions"][edge, 0, 0]),
            "chosen_cost": float(s["transitions"][edge, a, b]),
            "fixed_identity_supported_pairs": int(s["supported_pair_count"][edge, 0, 0]),
            "chosen_supported_pairs": int(s["supported_pair_count"][edge, a, b]),
            "chosen_supported_ids": s["supported_pairs"][edge, a, b].tolist(),
        }
    rows.append(row)
fixed = float(s["unary"][:, 0].sum() + s["transitions"][:, 0, 0].sum())
chosen = float(s["unary"][np.arange(len(frames)), state].sum()
               + s["transitions"][np.arange(len(frames)-1), state[:-1], state[1:]].sum())
result = {
    "status": "diagnosed_missing_connection_cost_evasion", "changed_frames": rows,
    "fixed_identity_objective": fixed, "chosen_objective": chosen,
    "objective_reduction": fixed - chosen,
    "all_changed_frames_excluded_from_color_scoring": all(
        not np.any(row["all_six_image_costs"]) for row in rows
    ),
    "independent_identity_truth_used": False,
    "decision": "Do not promote the adjacent-only arm to 666/minute before fixing lost memory.",
    "limitations": [
        "This explains a defective engineering objective, not biological identity correctness.",
        "A positive best-path gap does not exclude this cost-evasion failure.",
        "Future memory models must be separately registered and tested, not fitted to GT.",
    ],
}
with provenance("three_body_identity_missing_cost_diagnostic", output, name="diagnostic") as rec:
    for path, digest in hashes.items():
        rec.input(path, path, sha256=digest)
    target = output / "summary.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    rec.output("summary", target, sha256=base.sha256(target))
    text = output / "RESULT.md"
    text.write_text(
        "# 欠測接続の移動コストを避ける経路\n\n"
        "全200フレームを約5.94秒で処理した。333538/540/542/544/546/548だけで"
        "S0/S2の対応を交換し、次のフレームには元に戻った。全て色の登録期間で採点は0。"
        "胸ヒレの支持がある2匹のうちS0を欠測側に対応させ、前後の移動評価を1匹分へ減らす。\n\n"
        f"ID固定の目的関数{fixed:.6f}に対し、選択経路は{chosen:.6f}。"
        "この低下は適切なIDの証拠にならない。最良経路と競合経路の差が正でも同じ問題が起きる。"
        "元の幾何・支持・reviewは保存されている。\n\n"
        "この隣接観測だけの方式は666フレームへ進めず、各ID候補の最後の直接観測を保持する"
        "別の時間モデルを検討する。生物学的IDの正解注釈は用いていない。\n",
        encoding="utf-8",
    )
    rec.output("report", text, sha256=base.sha256(text))
    if any(base.sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("diagnostic evidence changed")
    rec.param("inputs_and_code_unchanged", True)
print(json.dumps({"output": str(output), "objective_reduction": fixed - chosen}))

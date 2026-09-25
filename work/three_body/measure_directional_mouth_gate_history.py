"""Completed first-checkpoint six-beam mouth age gates; saved-array CPU arithmetic.

No velocity reconstruction, current predicted-plan reconstruction, SL search,
projection, fit, raw decode, GPU or biological identity decision.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import diagnose_directional_mouth_first_chunk as prior
import numpy as np
import persistent_node_tracker as frozen

from medaka3d.provenance import provenance

checks, comparison = prior.checks, prior.comparison
ROOT = prior.REPO / "work/three_body/directional_mouth_first_chunk_diagnosis_v1"
SEAL_SHA = "e87d4c2c137ca09ced694efd8a96381b8e7b1d95555b539546575cd9dd3ee609"
START, STOP, HORIZON = 67636, 67779, 68060
STAGE = "directional_mouth_saved_six_beam_gate_v1"
REASONS = {
    0: "age_eligible_only_current_geometry_not_tested",
    1: "persistence_disabled",
    2: "mouth_missing_age_below_minimum",
    3: "direct_pectoral_age_below_minimum",
    4: "both_ages_below_minimum",
}


def age_gates(node_missing, pectoral_age, cfg):
    mouth_ok = node_missing[..., 0] >= cfg.minimum_missing_frames
    chest_ok = pectoral_age >= cfg.minimum_pectoral_frames
    reason = np.where(mouth_ok, np.where(chest_ok, 0, 3), np.where(chest_ok, 2, 4)).astype(np.int8)
    if not cfg.enabled:
        reason.fill(1)
    return reason == 0, reason


def counts(values):
    return dict(sorted(Counter(str(v) for v in np.asarray(values).ravel()).items()))


def analyze(state, metadata):
    cfg = frozen.PersistentNodeConfig(**metadata["context"]["persistence_config"])
    offset = state["history_offsets"]
    checks.require(len(offset) == HORIZON - prior.interim.FIRST + 1, "wrong checkpoint horizon")
    t = np.arange(START, STOP) - prior.interim.FIRST
    checks.require(
        np.all(offset[t] - offset[t - 1] == 6) and np.all(offset[t + 1] - offset[t] == 6),
        "six retained parents/current rows required",
    )
    previous = offset[t - 1, None] + np.arange(6)
    current = offset[t, None] + np.arange(6)
    checks.require(
        np.array_equal(
            state["history_frame_idx"][previous],
            np.broadcast_to((np.arange(START, STOP) - 1)[:, None], previous.shape),
        ),
        "previous native rows differ",
    )
    checks.require(
        np.array_equal(
            state["history_frame_idx"][current],
            np.broadcast_to(np.arange(START, STOP)[:, None], current.shape),
        ),
        "current native rows differ",
    )
    values = dict(
        frame_idx=np.arange(START, STOP, dtype=np.int64),
        previous_global_history_row=previous,
        current_global_history_row=current,
    )
    for name in (
        "node_missing",
        "direct_pectoral_age",
        "missing",
        "observed",
        "body_supported",
        "identity_review",
    ):
        values["previous_" + name] = state["history_" + name][previous].copy()
        values["current_" + name] = state["history_" + name][current].copy()
    for name in ("parent", "proposal_kind", "hypotheses"):
        values["current_" + name] = state["history_" + name][current].copy()
    age, reason = age_gates(
        values["previous_node_missing"], values["previous_direct_pectoral_age"], cfg
    )
    # Execute the frozen pure method as a formula oracle, without constructing a tracker.
    holder = SimpleNamespace(persistence_config=cfg)
    for ti in range(STOP - START):
        for beam in range(6):
            parent = SimpleNamespace(
                node_missing=values["previous_node_missing"][ti, beam],
                direct_pectoral_age=values["previous_direct_pectoral_age"][ti, beam],
            )
            exact = (
                frozen.PersistentNodeThreeBodyTracker._age_eligible(holder, parent) & cfg.enabled
            )
            checks.require(np.array_equal(exact, age[ti, beam]), "frozen age formula differs")
    parent = values["current_parent"]
    checks.require(np.all((parent >= 0) & (parent < 6)), "invalid retained parent")
    pi = np.arange(STOP - START)[:, None]
    obs = values["current_observed"]
    expected = {
        "node_missing": np.where(obs, 0, values["previous_node_missing"][pi, parent] + 1),
        "direct_pectoral_age": np.where(
            obs[..., 1], values["previous_direct_pectoral_age"][pi, parent] + 1, 0
        ),
        "missing": np.where(
            values["current_body_supported"], 0, values["previous_missing"][pi, parent] + 1
        ),
    }
    for key, value in expected.items():
        checks.require(
            np.array_equal(value, values["current_" + key]),
            "saved parent age recurrence differs: " + key,
        )
    winner, index = np.zeros(len(offset) - 1, np.int64), 0
    for k in range(len(winner) - 1, -1, -1):
        checks.require(0 <= index < offset[k + 1] - offset[k], "invalid same-horizon traceback")
        winner[k] = offset[k] + index
        index = int(state["history_parent"][winner[k]])
    winner_parent = state["history_parent"][winner[t]]
    checks.require(
        np.array_equal(previous[np.arange(STOP - START), winner_parent], winner[t - 1]),
        "winner prior parent differs",
    )
    values.update(
        per_parent_per_slot_age_eligible=age,
        per_parent_per_slot_reason_code=reason,
        state0_helper_call_age_gate_open=age[:, 0].any(axis=1),
        final_winner_current_global_row=winner[t],
        final_winner_parent_index=winner_parent,
        final_winner_current_kind=state["history_proposal_kind"][winner[t]].copy(),
        final_winner_previous_age_eligible=age[np.arange(STOP - START), winner_parent],
        final_winner_previous_reason_code=reason[np.arange(STOP - START), winner_parent],
    )
    checks.require(
        not state["history_observed"][winner[t], 0, 0].any(),
        "winner S0 mouth is not missing for all143",
    )
    retained_mouth = values["current_proposal_kind"] == "mouth_recovery"
    checks.require(
        np.all(~retained_mouth | (parent == 0)), "mouth recovery retained with non-main parent"
    )
    checks.require(
        np.all(~retained_mouth.any(1) | values["state0_helper_call_age_gate_open"]),
        "mouth recovery retained when age gate closed",
    )
    return values, cfg


def run(output):
    output = Path(output).resolve()
    checks.require(
        output.is_relative_to(prior.REPO / "work") and not output.exists(), "new output required"
    )
    codes = comparison.source_registry.local_code()
    hashes = dict(codes)
    prior.bind(ROOT / "diagnosis.provenance.json", SEAL_SHA, hashes)
    seal = checks.verify(ROOT, "diagnosis.provenance.json", prior.STAGE, hashes)
    checkpoint = prior.BRANCH / "checkpoints/through_068060"
    for name in ("manifest.json", "metadata.json", "arrays.npz"):
        prior.audit_input(seal, checkpoint / name, hashes)
    binding = checks.read(checkpoint / "manifest.json")["bindings"]
    # Bind the actual immutable source and producer snapshots for both implementations.
    for path, digest in binding["experimental_code_sha256"].items():
        source = prior.bind(path, digest, hashes)
        prior.bind(prior.BRANCH / "code_snapshot" / source.relative_to(prior.REPO), digest, hashes)
    for path, digest in binding["original_binding"]["code_sha256"].items():
        prior.bind(path, digest, hashes)
    state, metadata, _ = comparison.load_bound_checkpoint(checkpoint, binding, hashes)
    comparison.validate_lifecycle(metadata, HORIZON, 4)
    values, cfg = analyze(state, metadata)
    del state
    with np.load(ROOT / "original_same_horizon.npz", allow_pickle=False) as old:
        take = (old["frame_idx"] >= START) & (old["frame_idx"] < STOP)
        checks.require(take.sum() == STOP - START, "old same-horizon reference interval differs")
        values["old_same_horizon_mouth_sources"] = old["sources"][take, :, :, 0].copy()
        values["old_same_horizon_mouth_observed"] = old["observed"][take, :, 0].copy()
    gate, reason = (
        values["per_parent_per_slot_age_eligible"],
        values["per_parent_per_slot_reason_code"],
    )
    kinds = values["current_proposal_kind"]
    eligible_s0 = gate[:, 0, 0]
    old_s0 = values["old_same_horizon_mouth_observed"][:, 0]
    summary = dict(
        schema=STAGE,
        status="ok",
        native_interval=[START, STOP],
        native_frames=STOP - START,
        completed_checkpoint_horizon=HORIZON,
        parent_minute_completion_claim=False,
        persistence_config=dict(
            enabled=cfg.enabled,
            minimum_missing_frames=cfg.minimum_missing_frames,
            minimum_pectoral_frames=cfg.minimum_pectoral_frames,
        ),
        exact_age_expression=(
            "enabled & mouth_node_missing>=minimum_missing_frames "
            "& direct_pectoral_age>=minimum_pectoral_frames"
        ),
        actual_parent=0,
        body_missing_in_mouth_age_gate=False,
        per_parent_age_statistics_are_counterfactual_for_parents_1_to_5=True,
        reason_codes=REASONS,
        all_six_saved_age_recurrences_exact=True,
        state0_helper_call_age_gate_open_frames=int(gate[:, 0].any(1).sum()),
        state0_S0_age_eligible_frames=int(eligible_s0.sum()),
        state0_S0_age_eligible_runs=prior.runs(
            np.column_stack((eligible_s0, np.zeros((len(gate), 2), bool)))
        ),
        state0_S0_reason_counts={
            REASONS[int(key)]: value for key, value in counts(reason[:, 0, 0]).items()
        },
        state0_age_eligible_by_slot=gate[:, 0].sum(0).tolist(),
        age_eligible_by_parent_and_slot=gate.sum(0).tolist(),
        state0_S0_body_missing_positive_frames=int((values["previous_missing"][:, 0, 0] > 0).sum()),
        state0_S0_mouth_age_range=[
            int(values["previous_node_missing"][:, 0, 0, 0].min()),
            int(values["previous_node_missing"][:, 0, 0, 0].max()),
        ],
        state0_S0_direct_pectoral_age_range=[
            int(values["previous_direct_pectoral_age"][:, 0, 0].min()),
            int(values["previous_direct_pectoral_age"][:, 0, 0].max()),
        ],
        retained_current_kind_counts=counts(kinds),
        retained_kind_counts_by_rank=[counts(kinds[:, i]) for i in range(6)],
        retained_mouth_recovery_rows=int((kinds == "mouth_recovery").sum()),
        native_frames_with_retained_mouth_recovery=values["frame_idx"][
            (kinds == "mouth_recovery").any(1)
        ].tolist(),
        final_same_horizon_winner_kind_counts=counts(values["final_winner_current_kind"]),
        final_winner_parent_nonzero_frames=int((values["final_winner_parent_index"] != 0).sum()),
        old_same_horizon_S0_two_view_mouth_reference_frames=int(old_s0.sum()),
        old_reference_and_state0_S0_age_open_frames=int((old_s0 & eligible_s0).sum()),
        all_unretained_proposals_or_fit_plans_saved=False,
        current_predicted_mouth_chest_support_not_reconstructed=True,
        helper_post_age_filters_not_recomputed=True,
        no_velocity_reconstruction=True,
        no_raw_decode_or_projection_or_fit=True,
        no_GPU=True,
        no_ID_truth=True,
    )
    # prior.runs is registered for native66400. Translate this local143 mask explicitly.
    for item in summary["state0_S0_age_eligible_runs"]:
        item["start"] += START - prior.START
        item["stop_exclusive"] += START - prior.START
    rows = []
    for i, frame in enumerate(values["frame_idx"]):
        rows.append(
            dict(
                frame_idx=int(frame),
                previous_native_frame=int(frame) - 1,
                previous_all6_node_missing=values["previous_node_missing"][i].tolist(),
                previous_all6_direct_pectoral_age=values["previous_direct_pectoral_age"][
                    i
                ].tolist(),
                previous_all6_body_missing=values["previous_missing"][i].tolist(),
                per_parent_slot_age_eligible=gate[i].tolist(),
                per_parent_slot_reason_code=reason[i].tolist(),
                state0_helper_call_age_gate_open=bool(gate[i, 0].any()),
                state0_S0_reason=REASONS[int(reason[i, 0, 0])],
                current_retained_kinds=kinds[i].tolist(),
                current_retained_parents=values["current_parent"][i].tolist(),
                final_winner_kind=str(values["final_winner_current_kind"][i]),
                final_winner_previous_parent=int(values["final_winner_parent_index"][i]),
                old_same_horizon_mouth_sources=values["old_same_horizon_mouth_sources"][i].tolist(),
                old_same_horizon_mouth_observed=values["old_same_horizon_mouth_observed"][
                    i
                ].tolist(),
                current_predicted_geometry_gate="not_reconstructed",
            )
        )
    lines = [
        "# 新S0口欠測143フレームの回復age gate",
        "",
        "完成済み68060 checkpointの全6 retained historyを用いた [67636,67779) のCPU算術。",
        "実際に検査される直前state0のS0 age gateは "
        f"{summary['state0_S0_age_eligible_frames']}/143 フレームで開く。"
        "state0内のいずれかの魚がage eligibleなhelper呼出条件は "
        f"{summary['state0_helper_call_age_gate_open_frames']}/143。",
        "",
        "実条件は enabled かつ mouth node_missing>=4 かつ direct_pectoral_age>=4。"
        "body missingはこのmouth gateには使われない。parent1-5は同式を当てた参考値であり、"
        "mouth回復候補をそのparentから生成する実装ではない。",
        "",
        "S0 age gateの閉鎖理由:",
        "",
    ]
    lines += [f"- {key}: {value}" for key, value in summary["state0_S0_reason_counts"].items()]
    lines += [
        "",
        f"retained6候補のmouth_recoveryは {summary['retained_mouth_recovery_rows']} 行。"
        f"同68060最終勝者kindは {summary['final_same_horizon_winner_kind_counts']}。"
        "最終勝者の前段parentが0以外のフレームは "
        f"{summary['final_winner_parent_nonzero_frames']}。",
        "",
        f"旧同horizonでS0の口に2視点支持が保存されているのは {int(old_s0.sum())} フレーム、"
        f"うち新state0 S0のage gateも開いているのは {int((old_s0 & eligible_s0).sum())}。"
        "これは既存の旧口sourceだけの参考値であり、新予測姿勢・未占有観測・候補生成の成功を意味しない。",
        "",
        "後段node_recoveryは、現在の予測口が未支持、胸部が直接支持、未占有mouth rowが2視点以上、"
        "交差数上限・ray RMS・box・解剖長・2mm/12pxを満たすことを別途要求する。"
        "今回は予測pose/planや落とされた全候補がなく、速度復元・再投影も行わないため、それらを未判定として残す。"
        "age gateが開いても生成やfit成功は推定しない。"
        "retained mouth_recoveryがあれば当該scene候補の存在はわかるが、"
        "それがS0用かはkindだけでは決めない。",
        "",
        "143 nativeの全6parent x 3slotのages/reasons、"
        "retained kinds/parentsと同horizon winnerをNPZ/JSONLに保存。"
        "支持は既存診断参照のみ。閾値・追跡・ID・旧reviewは変更していない。1分全体の完走・最終backtraceを主張しない。",
        "",
    ]
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="gate") as record:
        for source, digest in codes.items():
            dest = output / "code_snapshot" / Path(source).relative_to(prior.REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(source).read_bytes())
            checks.require(checks.sha(dest) == digest, "source snapshot changed")
            record.output(str(dest.relative_to(output)), dest, sha256=digest)
        np.savez_compressed(output / "six_beam_gates.npz", **values)
        (output / "frames.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8"
        )
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        (output / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")
        for path in output.iterdir():
            if path.is_file():
                record.output(path.name, path, sha256=checks.sha(path))
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "gate input changed")
        for item in record.outputs.values():
            checks.require(checks.sha(item["path"]) == item["sha256"], "gate output changed")
        record.param("all_opened_input_source_snapshot_output_sha_unchanged", True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

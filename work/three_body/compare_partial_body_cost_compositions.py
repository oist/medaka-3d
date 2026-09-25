"""Compare fixed cost compositions on eleven successful saved candidate banks.

No fitting, tracking, observation decoding or geometry projection is performed.
Alpha 9 matches a maximum cost range only; it is not a recommended weight.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import compare_partial_body_saved_banks as saved
import numpy as np

proof, require, codec = saved.proof, saved.require, saved.codec
WORK, REPO, VIEWS = saved.WORK, saved.REPO, saved.VIEWS
STAGE = "partial_body_cost_compositions_v1"
INPUT_ROOT = WORK / "partial_body_saved_bank_comparison_v1"
INPUT_SHA = "10d29aad72ca98674abbcbaf386ce48244087cb4abae552b51e592b5dbb9910f"
PRODUCER_SHA = "7e069ea6b727c05c951f1e40d70ea8ad54276db2a0a344fa8b4dca534f7d2dd8"
CODEC_SHA = "91a66646f89a6abb59b6d60bc1b357b383167c21eb9d40bc47491802d05e9544"
NEW_MODES = ("row_only", "row_plus_mask")
ALPHAS = (1, 9)
SCOPES = ("all_views", "old_ineligible_views")
RANK_FIELDS = (
    "local_cost",
    "path_cost",
    "best_local_cost",
    "best_path_cost",
    "local_order",
    "path_order",
    "local_minimum_indices",
    "path_minimum_indices",
    "minimum_ties_are_exact_float64",
)
BANK_FIELDS = (
    "frame_idx",
    "policy",
    "candidate_index",
    "candidate_pose_xyz",
    "candidate_parent",
    "candidate_plan",
    "candidate_kind",
    "base_local_cost",
    "sixth_instance_term",
    "adjusted_local_cost",
    "parent_path_cost",
    "adjusted_path_cost",
    "capped_contributions_by_view",
)
VIEW_FIELDS = (
    "eligible",
    "reason",
    "neutral",
    "raw_uncapped_scene_cost",
    "uncapped_mean_body_cost",
    "capped_view_contribution",
    "nonfinite_score",
    "cap_applied",
)


def mode_names():
    return ["baseline", *NEW_MODES] + [
        f"preserve_old__{mode}__{scope}__alpha{alpha}"
        for mode in NEW_MODES
        for alpha in ALPHAS
        for scope in SCOPES
    ]


def same(left, right, message):
    require(proof.signature(np.asarray(left)) == proof.signature(np.asarray(right)), message)


def document(seal, name, evidence):
    return proof.read_json(saved.member(INPUT_ROOT, seal, name, evidence))


def source_code(plan, seal, evidence):
    code = dict(plan["source_code"])
    for name, digest in (
        ("compare_partial_body_saved_banks.py", PRODUCER_SHA),
        ("run_instance_rescore_10s.py", CODEC_SHA),
    ):
        require(code.get(str(WORK / name)) == digest, "frozen producer/helper differs")
    for path, digest in code.items():
        require(Path(path).resolve().is_relative_to(REPO), "source outside repository")
        saved.edge(seal, path, digest)
        evidence.add(Path(path), digest)
    own = str(Path(__file__).resolve())
    code[own] = evidence.add(Path(own))
    return code


def load_headers(evidence):
    seal = saved.read(INPUT_ROOT / "comparison.provenance.json", INPUT_SHA, evidence)
    require(seal["stage"] == saved.STAGE and seal["status"] == "ok", "successful input required")
    plan, summary, payload = [
        document(seal, n + ".json", evidence) for n in ("plan", "summary", "render_payload")
    ]
    expected = saved.case_keys()
    require(
        plan["schema"] == summary["schema"] == saved.STAGE
        and summary["status"] == "ok"
        and plan["cases"] == [list(k) for k in expected]
        and plan["modes"] == list(saved.MODES)
        and plan["terminal_horizon"] == 116200
        and plan["normalizers"] == dict(row_only=15.0, row_plus_mask=15.0)
        and plan["scorer_ceilings"] == dict(row_only=12.0, row_plus_mask=15.0)
        and plan["fixed_parent_history"] is True
        and plan["old_sixth_replaced_not_added"] is True
        and summary["saved_candidate_banks"] == 11
        and summary["baseline_bank_unchanged"] is True
        and summary["physical_ID_truth"] is False
        and summary["new_trajectory_created"] is False,
        "saved comparison scope differs",
    )
    require(
        [(c["arm"], c["frame_idx"]) for c in summary["cases"]] == expected
        and [(c["arm"], c["frame_idx"]) for c in payload["cases"]] == expected
        and payload["schema"] == saved.STAGE + "_render_payload_v1"
        and payload["modes"] == list(saved.MODES)
        and payload["physical_ID_truth"] is False
        and payload["raw_slot_labels_only"] is True
        and payload["new_trajectory_created"] is False,
        "all eleven fixed cases/render scope required",
    )
    expected_outputs = {"plan.json", "summary.json", "render_payload.json"} | {
        f"cases/{arm}_{frame}.{suffix}" for arm, frame in expected for suffix in ("json", "npz")
    }
    require(set(seal["outputs"]) == expected_outputs, "exact completed input membership required")
    return dict(
        seal=seal,
        plan=plan,
        summary=summary,
        payload=payload,
        code=source_code(plan, seal, evidence),
    )


def array_keys(tree):
    keys = []
    if isinstance(tree, dict):
        if "__array__" in tree:
            require(set(tree) == {"__array__", "signature"}, "invalid array descriptor")
            return [tree["__array__"]]
        for value in tree.values():
            keys.extend(array_keys(value))
    elif isinstance(tree, list):
        for value in tree:
            keys.extend(array_keys(value))
    return keys


def selected_tree(tree):
    bank = tree["baseline_bank"]
    selected = {
        k: tree[k]
        for k in (
            "schema",
            "arm",
            "frame_idx",
            "projections",
            "final_selected_match",
            "parent_history_fixed",
            "applies_only_to_saved_fit_candidates",
            "reoptimized_fit",
            "new_temporal_backtrace",
            "physical_ID_truth",
        )
    }
    selected["baseline_bank"] = {k: bank[k] for k in BANK_FIELDS}
    selected["baseline_bank"]["views"] = {
        view: {k: bank["views"][view][k] for k in VIEW_FIELDS} for view in VIEWS
    }
    selected["modes"] = {}
    for mode in saved.MODES:
        keys = (*RANK_FIELDS, "sixth_term")
        if mode != "baseline":
            keys += (
                "raw_costs_by_view",
                "normalized_costs_by_view",
                "normalizer",
                "scorer_ceiling",
                "invalid_projection_candidates",
            )
        selected["modes"][mode] = {k: tree["modes"][mode][k] for k in keys}
    return selected


def load_case(context, summary, evidence):
    arm, frame = summary["arm"], summary["frame_idx"]
    stem = f"cases/{arm}_{frame}"
    record = document(context["seal"], stem + ".json", evidence)
    require(
        record["archive"] == summary["archive"] == context["seal"]["outputs"][stem + ".npz"],
        "case archive references differ",
    )
    path = saved.member(INPUT_ROOT, context["seal"], stem + ".npz", evidence)
    tree = selected_tree(record["tree"])
    all_keys, needed = array_keys(record["tree"]), array_keys(tree)
    require(
        len(all_keys) == len(set(all_keys)) and len(needed) == len(set(needed)),
        "duplicate diagnostic arrays",
    )
    with np.load(path, allow_pickle=False) as archive:
        require(
            len(archive.files) == len(all_keys) and set(archive.files) == set(all_keys),
            "input archive membership differs",
        )
        arrays = {k: archive[k] for k in needed}
    result = codec.restore_diagnostic(tree, arrays)
    for array in arrays.values():
        array.flags.writeable = False
    require(
        result["schema"] == saved.STAGE + "_case"
        and (result["arm"], result["frame_idx"]) == (arm, frame)
        and result["baseline_bank"]["frame_idx"] == frame
        and result["parent_history_fixed"] is True
        and result["applies_only_to_saved_fit_candidates"] is True
        and result["reoptimized_fit"] is False
        and result["new_temporal_backtrace"] is False
        and result["physical_ID_truth"] is False,
        "saved case scope differs",
    )
    require(
        len(result["baseline_bank"]["candidate_index"]) == summary["candidates"],
        "candidate count differs",
    )
    for mode in saved.MODES:
        for key, value in summary["rankings"][mode].items():
            require(
                saved.small_json(result["modes"][mode][key]) == value,
                "input summary ranking differs",
            )
    return result, len(needed), len(all_keys)


def old_components(bank):
    policy = bank["policy"]
    require(
        policy["weight"] == 3.0
        and policy["bodies_per_view"] == 3
        and policy["fixed_camera_denominator"] == 3
        and policy["cap_per_view"] == 9.0
        and policy["camera_order"] == list(VIEWS),
        "old scientific weights differ",
    )
    n = len(bank["candidate_index"])
    same(bank["candidate_index"], np.arange(n, dtype=np.int64), "candidate order differs")
    eligible, columns = [], []
    for view in VIEWS:
        detail = bank["views"][view]
        require(
            type(detail["eligible"]) is bool and detail["neutral"] is not detail["eligible"],
            "observation-only old gate invalid",
        )
        eligible.append(detail["eligible"])
        raw = detail["raw_uncapped_scene_cost"]
        require(
            raw.dtype == np.float64
            and raw.shape == (n,)
            and not np.any(np.isfinite(raw) & (raw < 0)),
            "old raw view costs invalid",
        )
        normalized = raw / 3
        invalid = ~np.isfinite(raw)
        capped = np.clip(normalized, 0.0, 9.0)
        capped[invalid] = 9.0
        if not detail["eligible"]:
            capped[:] = 0.0
        same(normalized, detail["uncapped_mean_body_cost"], "old body mean differs")
        same(invalid, detail["nonfinite_score"], "old invalid mask differs")
        same((normalized > 9.0) | invalid, detail["cap_applied"], "old cap flags differ")
        same(capped, detail["capped_view_contribution"], "old view contribution differs")
        columns.append(capped)
    contributions = np.stack(columns, axis=1)
    same(contributions, bank["capped_contributions_by_view"], "old camera order differs")
    # Preserve the actual multiply-then-divide order, even though the weight is 3.
    penalty = 3.0 * contributions.sum(axis=1) / 3
    same(penalty, bank["sixth_instance_term"], "old sixth term reconstruction differs")
    for key in ("base_local_cost", "parent_path_cost", "adjusted_local_cost", "adjusted_path_cost"):
        require(
            bank[key].dtype == np.float64
            and bank[key].shape == (n,)
            and np.isfinite(bank[key]).all(),
            "finite saved cost vector required",
        )
    same(bank["base_local_cost"] + penalty, bank["adjusted_local_cost"], "old local ledger differs")
    same(
        bank["parent_path_cost"] + bank["adjusted_local_cost"],
        bank["adjusted_path_cost"],
        "old path ledger differs",
    )
    return np.asarray(eligible, dtype=bool), contributions, penalty


def compose(base, parent, old, partial, eligible, alpha, scope):
    require(
        type(alpha) is int and alpha in ALPHAS and scope in SCOPES, "fixed composition required"
    )
    require(
        partial.shape == (len(base), 3)
        and partial.dtype == np.float64
        and not np.isnan(partial).any()
        and np.all(partial >= 0)
        and np.all(partial[np.isfinite(partial)] <= 1)
        and eligible.dtype == bool
        and eligible.shape == (3,),
        "partial/gate shape or range differs",
    )
    # Do not multiply a zero eligibility indicator by +inf.
    addition = partial.copy() if scope == "all_views" else np.where(eligible[None, :], 0.0, partial)
    invalid = np.isposinf(partial).any(axis=1)
    extra = alpha * addition.sum(axis=1)
    # A nonfinite candidate projection remains invalid even in an excluded view.
    extra[invalid] = np.inf
    sixth = old + extra
    ranked = saved.rank_costs(base + sixth, parent)
    return dict(
        ranked,
        sixth_term=sixth,
        new_partial_term=extra,
        selected_partial_contributions_by_view=addition,
        invalid_projection_candidates=invalid,
        alpha=alpha,
        scope=scope,
    )


def compare_case(case):
    bank = case["baseline_bank"]
    eligible, old_views, old = old_components(bank)
    n = len(old)
    require(
        bank["candidate_pose_xyz"].shape == (n, 3, 3, 3)
        and np.isfinite(bank["candidate_pose_xyz"]).all()
        and bank["candidate_plan"].shape == (n, 3, 3, 3)
        and bank["candidate_plan"].dtype.kind in "iu"
        and len(bank["candidate_kind"]) == n
        and bank["candidate_parent"].shape == (n,)
        and set(case["projections"]) == set(VIEWS)
        and all(p.shape == (n, 3, 3, 2) for p in case["projections"].values()),
        "saved proposed geometry differs",
    )
    modes = copy.deepcopy(case["modes"])
    for mode in saved.MODES:
        data = modes[mode]
        expected = saved.rank_costs(data["local_cost"], bank["parent_path_cost"])
        for key in RANK_FIELDS:
            if isinstance(expected[key], np.ndarray):
                same(expected[key], data[key], "saved rank differs")
            else:
                require(expected[key] == data[key], "saved rank scalar differs")
        same(
            bank["base_local_cost"] + data["sixth_term"],
            data["local_cost"],
            "saved replacement ledger differs",
        )
    same(modes["baseline"]["sixth_term"], old, "baseline sixth differs")
    for mode in NEW_MODES:
        data = modes[mode]
        require(
            data["normalizer"] == 15.0
            and data["scorer_ceiling"] == (12.0 if mode == "row_only" else 15.0),
            "new normalization differs",
        )
        raw = data["raw_costs_by_view"]
        require(
            raw.dtype == np.float64
            and raw.shape == (n, 3)
            and np.all(raw[np.isfinite(raw)] <= data["scorer_ceiling"] + 1e-12),
            "new raw cost range differs",
        )
        partial, term = saved.bounded_term(raw, 15.0)
        same(partial, data["normalized_costs_by_view"], "new per-view cost differs")
        same(term, data["sixth_term"], "new sixth term differs")
        same(
            np.isposinf(raw).any(axis=1),
            data["invalid_projection_candidates"],
            "new invalid candidate mask differs",
        )
        for alpha in ALPHAS:
            for scope in SCOPES:
                modes[f"preserve_old__{mode}__{scope}__alpha{alpha}"] = compose(
                    bank["base_local_cost"],
                    bank["parent_path_cost"],
                    old,
                    partial,
                    eligible,
                    alpha,
                    scope,
                )
    require(list(modes) == mode_names(), "composition ordering differs")
    return dict(
        schema=STAGE + "_case",
        arm=case["arm"],
        frame_idx=case["frame_idx"],
        baseline_bank=bank,
        projections=case["projections"],
        modes=modes,
        old_eligible_by_view=eligible,
        old_capped_contributions_by_view=old_views,
        old_sixth_reconstructed=old,
        final_selected_match=case["final_selected_match"],
        fixed_parent_candidate_pool=True,
        support_recomputed=False,
        physical_ID_truth=False,
    )


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    require(
        "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(), "CPU only"
    )
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        context = load_headers(evidence)
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                input_root=str(INPUT_ROOT),
                input_seal_sha256=INPUT_SHA,
                source_code=context["code"],
                cases=[list(k) for k in saved.case_keys()],
                modes=mode_names(),
                alphas=list(ALPHAS),
                alpha9_is_scale_control_not_recommendation=True,
                camera_order=list(VIEWS),
                old_formula="3 * sum_views(capped_scene_cost_div3) / 3",
                added_formula="parent + (base + (old_sixth + alpha * selected_partial_sum))",
                invalid_projection_global_rejection=True,
                excluded_view_uses_selection_not_zero_times_inf=True,
                original_observation_evidence=context["payload"]["original_observation_evidence"],
                fixed_parent_history=True,
                new_trajectory_created=False,
                physical_ID_truth=False,
                original_SL_mask_video_rig_reads=0,
                fit_tracker_solver_calls=0,
                GPU_used=False,
            ),
        )
        summaries, render_rows = [], []
        for summary in context["summary"]["cases"]:
            case, decoded, total = load_case(context, summary, evidence)
            before = codec.split_diagnostic(case)[0]
            result = compare_case(case)
            require(codec.split_diagnostic(case)[0] == before, "input case changed")
            archive = saved.write_case(out, f"cases/{case['arm']}_{case['frame_idx']}", result)
            render_rows.append(saved.render_selection(result))
            summaries.append(
                dict(
                    arm=case["arm"],
                    frame_idx=case["frame_idx"],
                    candidates=summary["candidates"],
                    input_archive=summary["archive"],
                    archive=archive,
                    decoded_input_arrays=decoded,
                    total_input_arrays=total,
                    old_eligible_by_view=result["old_eligible_by_view"],
                    old_sixth_exact_reconstruction=True,
                    original_three_rankings_unchanged=True,
                    final_selected_match=result["final_selected_match"],
                    rankings={
                        mode: {k: d[k] for k in RANK_FIELDS if k not in ("local_cost", "path_cost")}
                        for mode, d in result["modes"].items()
                    },
                )
            )
        out.write(
            "render_payload.json",
            saved.small_json(
                dict(
                    schema=STAGE + "_render_payload_v1",
                    modes=mode_names(),
                    cases=render_rows,
                    original_observation_evidence=context["payload"][
                        "original_observation_evidence"
                    ],
                    raw_slot_labels_only=True,
                    all_markers_are_proposals=True,
                    negative_OldFast_images_are_observation_context_only=True,
                    physical_ID_truth=False,
                    new_trajectory_created=False,
                )
            ),
        )
        out.write(
            "summary.json",
            saved.small_json(
                dict(
                    schema=STAGE,
                    status="ok",
                    cases=summaries,
                    saved_candidate_banks=11,
                    compositions_per_bank=11,
                    alphas=list(ALPHAS),
                    original_three_rankings_unchanged=True,
                    observation_gates_candidate_independent=True,
                    old_sixth_exact_reconstruction=True,
                    fixed_parent_candidate_ranking_only=True,
                    physical_ID_truth=False,
                    new_trajectory_created=False,
                    source_arrays_unchanged=True,
                    limitations=[
                        "These development frames are not held out.",
                        "Alpha 9 aligns maximum range only, not evidential strength.",
                        "Old and new evidence are correlated; a soft sum permits tradeoffs.",
                        "No new projection, support evaluation or continuous trajectory.",
                    ],
                )
            ),
        )
        evidence.final_guard()
        out.final_guard()
        require(
            "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
            "CUDA initialized",
        )
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(error),
        )
        (output / "comparison.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (output / "comparison.provenance.json").write_bytes(proof.json_bytes(seal))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()

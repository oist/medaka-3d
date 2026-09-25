"""One-frame saved pre/post-fit body-row and final-path diagnosis; no tracking."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK / "whole_row_body_ten_second_continuation_v1"
COMPARISON = WORK / "body_row_ten_second_comparison_v1"
TRACKING_SHA = "562acdab952d46cd915ac15ffc16d0e26ed14b2aeba1096cd022f98ae04c9b96"
COMPARISON_SHA = "8dc3f94118b519647a5f4ac604bdd62fec4980318b904310e3a004f1e54d2fda"
FRAME = 82720


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(value):
    value = np.ascontiguousarray(value)
    return dict(
        dtype=value.dtype.str,
        shape=list(value.shape),
        sha256=hashlib.sha256(value.tobytes()).hexdigest(),
    )


def safe(value):
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def restore(tree, arrays):
    """Independent small companion decoder, including complete byte signatures."""
    used = set()

    def visit(value):
        if isinstance(value, dict) and "__array__" in value:
            require(set(value) == {"__array__", "signature"}, "invalid array descriptor")
            key = value["__array__"]
            require(key in arrays and key not in used, "missing or repeated array")
            used.add(key)
            require(signature(arrays[key]) == value["signature"], "array signature differs")
            return arrays[key]
        if isinstance(value, dict):
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        return value

    result = visit(tree)
    require(used == set(arrays), "extra companion array")
    return result


def retained_candidates(poses, parents, permutations, path_cost, limit):
    """Reconstruct only this frozen local sort/dedup rule, not future backtrace."""
    selected, rejected = [], []
    for candidate in np.argsort(path_cost, kind="stable"):
        duplicate = next(
            (
                j
                for j in selected
                if parents[candidate] == parents[j]
                and np.array_equal(permutations[candidate], permutations[j])
                and np.max(np.linalg.norm(poses[candidate] - poses[j], axis=-1)) < 0.5
            ),
            None,
        )
        if duplicate is not None:
            rejected.append(dict(candidate_index=int(candidate), duplicate_of=int(duplicate)))
            continue
        selected.append(int(candidate))
        if len(selected) == limit:
            break
    return selected, rejected


def run():
    output = WORK / "BODY_ROW_82720_FIT_DIAGNOSIS_V1.json"
    require(not output.exists(), "fresh diagnostic output required")
    inputs, cached, archives = {}, {}, []

    def capture(path, digest=None, size=None):
        path = Path(path).resolve()
        if str(path) in cached:
            data = cached[str(path)]
        else:
            data = path.read_bytes()
            cached[str(path)] = data
        actual = hashlib.sha256(data).hexdigest()
        require(digest is None or actual == digest, "input SHA differs: " + str(path))
        require(size is None or len(data) == size, "input size differs")
        inputs[str(path)] = dict(sha256=actual, size_bytes=len(data))
        return data

    def read(path, digest=None):
        return json.loads(capture(path, digest))

    def member(root, seal, role, json_value=True):
        row = seal["outputs"][role]
        path = (root / role).resolve()
        require(str(path) == row["path"], "member path differs")
        data = capture(path, row["sha256"], row["size_bytes"])
        return json.loads(data) if json_value else data

    def diagnostic(role, seal):
        meta = member(ROOT, seal, role + ".json")
        require(meta["archive"] == seal["outputs"][role + ".npz"], "archive edge differs")
        data = member(ROOT, seal, role + ".npz", False)
        with np.load(io.BytesIO(data), allow_pickle=False) as a:
            arrays = {k: a[k].copy() for k in a.files}
        archives.append(dict(path=meta["archive"]["path"], fields=list(arrays)))
        return restore(meta["tree"], arrays)

    tracking = read(ROOT / "tracking.provenance.json", TRACKING_SHA)
    require(tracking["stage"] == ROOT.name and tracking["status"] == "ok", "tracking unsuccessful")
    summary = member(ROOT, tracking, "summary.json")
    require(
        summary["process_interval"] == [81340, 83000] and summary["finish_calls"] == 1,
        "wrong continuous interval",
    )
    config = summary["tracker_context"]["tracker_config"]
    for source in (
        WORK / "body_row_tracker.py",
        WORK / "body_row_proposals.py",
        WORK / "instance_rescore_tracker.py",
        WORK.parents[1] / "labs/20260909_three_body/continuous_tracker.py",
        WORK / "fast_directional_device_fitter.py",
    ):
        capture(source, tracking["inputs"][str(source)])
    capture(Path(__file__))
    before = diagnostic("body_row_diagnostics/f82720", tracking)
    after = diagnostic("diagnostics/f82720", tracking)
    require(
        before["frame_idx"] == after["frame_idx"] == FRAME and before["frame_completed"],
        "wrong/incomplete diagnostic frame",
    )
    count = before["outgoing_proposal_count"]
    added = before["new_proposal_indices"]
    require(
        np.array_equal(after["candidate_index"], np.arange(count)), "candidate ordering differs"
    )
    require(
        np.array_equal(added, np.arange(before["incoming_proposal_count"], count)),
        "additional indices differ",
    )
    poses = after["candidate_pose_xyz"]
    parents = after["candidate_parent"]
    permutation = after["candidate_permutation"]
    local, path_cost = after["adjusted_local_cost"], after["adjusted_path_cost"]
    require(
        np.isfinite(poses).all() and np.isfinite(local).all() and np.isfinite(path_cost).all(),
        "nonfinite fitted bank",
    )
    require(
        np.array_equal(local, after["base_local_cost"] + after["sixth_instance_term"]),
        "local arithmetic differs",
    )
    require(np.array_equal(path_cost, local + after["parent_path_cost"]), "path arithmetic differs")
    require(
        np.array_equal(after["candidate_plan"][added], before["new_proposal_plans"]),
        "body-row plans changed during fitting",
    )
    require(
        np.all(parents[added] == 0)
        and all(after["candidate_kind"][i] == "body_row" for i in added),
        "body-row parent/kind changed",
    )
    local_order, path_order = np.argsort(local, kind="stable"), np.argsort(path_cost, kind="stable")
    local_rank, path_rank = np.argsort(local_order) + 1, np.argsort(path_order) + 1
    retained, duplicate_skip = retained_candidates(
        poses, parents, permutation, path_cost, config["beam_size"]
    )
    # Only the already exported four-row final sample is read, never full candidate/state.
    comparison = read(COMPARISON / "comparison.provenance.json", COMPARISON_SHA)
    require(comparison["status"] == "ok", "successful comparison required")
    csummary = member(COMPARISON, comparison, "summary.json")
    require(
        csummary["variant_tracking_sha256"] == TRACKING_SHA
        and csummary["terminal_horizon"] == 83000,
        "comparison producer/horizon differs",
    )
    sample_signatures = member(COMPARISON, comparison, "review_samples_signatures.json")
    data = member(COMPARISON, comparison, "review_samples.npz", False)
    names = [
        "body_row_" + k for k in ("frame_idx", "pose_xyz", "observed", "sources", "proposal_kind")
    ]
    with np.load(io.BytesIO(data), allow_pickle=False) as a:
        samples = {k: a[k].copy() for k in names}
    for k, value in samples.items():
        require(signature(value) == sample_signatures[k], "final selected sample differs")
    archives.append(dict(path=str(COMPARISON / "review_samples.npz"), fields=names))
    hit = np.flatnonzero(samples["body_row_frame_idx"] == FRAME)
    require(len(hit) == 1, "one selected sample required")
    n = int(hit[0])
    final_pose = samples["body_row_pose_xyz"][n]
    final_kind = str(samples["body_row_proposal_kind"][n])
    final_match = [
        i
        for i in range(count)
        if np.array_equal(poses[i], final_pose) and after["candidate_kind"][i] == final_kind
    ]
    require(
        len(final_match) == 1 and final_match[0] in retained, "final path match not unique/retained"
    )
    selected = final_match[0]
    final_sources = samples["body_row_sources"][n]
    require(
        np.all((final_sources < 0) | (final_sources == after["candidate_plan"][selected])),
        "final sample sources conflict with fitted plan",
    )
    parts = after["original_five_loss_parts"]
    weights = np.array(
        [
            1,
            config["shape_weight"],
            config["motion_weight"],
            config["repulsion_weight"],
            config["mask_weight"],
        ]
    )
    weighted_parts = parts * weights
    scene_details = []
    for s, scene in enumerate(before["scenes"]):
        admitted = s < len(added)
        value = dict(
            scene_index=s,
            admitted=admitted,
            candidate_index=int(added[s]) if admitted else None,
            proposal_heuristic_score=scene["proposal_score"],
            pool_indices=scene["pool_indices"],
            source_rows=scene["source_rows"],
            pre_fit_pose_xyz=scene["pose_xyz"],
            proposed_sources=scene["proposed_sources"],
            triangulated_nodes=scene["triangulated_nodes"],
            template_filled_nodes=~scene["triangulated_nodes"],
            pre_fit_segment_lengths_mm=np.linalg.norm(np.diff(scene["pose_xyz"], axis=1), axis=-1),
            initial_fit_objective_saved=False,
        )
        if admitted:
            i = int(added[s])
            require(
                np.array_equal(scene["pose_xyz"], before["new_proposal_poses"][s]),
                "scene pose differs",
            )
            delta = np.linalg.norm(poses[i] - scene["pose_xyz"], axis=-1)
            value.update(
                post_fit_pose_xyz=poses[i],
                pre_to_post_node_displacement_mm=delta,
                maximum_measured_node_displacement_mm=float(
                    delta[scene["triangulated_nodes"]].max()
                ),
                maximum_filled_node_displacement_mm=(
                    float(delta[~scene["triangulated_nodes"]].max())
                    if (~scene["triangulated_nodes"]).any()
                    else None
                ),
                pre_to_parent_prediction_mm=np.linalg.norm(
                    scene["pose_xyz"] - before["parent_prediction"], axis=-1
                ),
                post_to_parent_prediction_mm=np.linalg.norm(
                    poses[i] - before["parent_prediction"], axis=-1
                ),
                post_fit_segment_lengths_mm=np.linalg.norm(np.diff(poses[i], axis=1), axis=-1),
                local_rank=int(local_rank[i]),
                path_rank=int(path_rank[i]),
                retained_in_reconstructed_local_beam=i in retained,
                local_cost=float(local[i]),
                parent_path_cost=float(after["parent_path_cost"][i]),
                path_cost=float(path_cost[i]),
                original_five_loss_parts=parts[i],
                weighted_five_stored_parts=weighted_parts[i],
                sixth_instance_term=float(after["sixth_instance_term"][i]),
                physical_or_image_regression_independently_evaluated=False,
            )
        scene_details.append(value)
    table = []
    for i in range(count):
        table.append(
            dict(
                candidate_index=i,
                kind=after["candidate_kind"][i],
                parent=int(parents[i]),
                permutation=permutation[i],
                base_local_cost=after["base_local_cost"][i],
                sixth_instance_term=after["sixth_instance_term"][i],
                local_cost=local[i],
                parent_path_cost=after["parent_path_cost"][i],
                path_cost=path_cost[i],
                local_rank=local_rank[i],
                path_rank=path_rank[i],
                retained_in_reconstructed_local_beam=i in retained,
                final_83000_backtrace_selected=i == selected,
                original_five_loss_parts=parts[i],
                weighted_five_stored_parts=weighted_parts[i],
                base_minus_weighted_stored_parts=after["base_local_cost"][i]
                - weighted_parts[i].sum(),
                pose_xyz=poses[i],
                plan=after["candidate_plan"][i],
            )
        )
    result = dict(
        schema="body_row_82720_fit_diagnosis_v1",
        status="ok",
        frame_idx=FRAME,
        process_interval=[81340, 83000],
        final_horizon=83000,
        counts={
            k: before[k]
            for k in (
                "incoming_proposal_count",
                "outgoing_proposal_count",
                "decoder_candidate_cap",
                "headroom",
                "generated_scene_count",
                "added_proposal_count",
                "excluded_by_capacity",
                "excluded_by_scene_limit",
            )
        },
        scene_stats=before["scene_stats"],
        pool_stats=before["pool_stats"],
        parent_prediction=before["parent_prediction"],
        pool=before["pool"],
        scenes=scene_details,
        candidates=table,
        local_order=local_order,
        path_order=path_order,
        reconstructed_retained_indices=retained,
        reconstructed_duplicate_skips=duplicate_skip,
        final_selected_candidate_index=selected,
        final_selected_observed=samples["body_row_observed"][n],
        final_selected_sources=final_sources,
        final_selected_kind=final_kind,
        view_gates={
            v: dict(eligible=r["eligible"], reason=r["reason"], evidence=r["observation_evidence"])
            for v, r in after["views"].items()
        },
        cost_part_order=[
            "node",
            "shape",
            "directional_center",
            "repulsion",
            "foreground_union_silhouette",
        ],
        cost_part_weights=weights,
        orientation_term_not_separately_saved=True,
        physical_ID_truth=False,
        visual_regression_not_evaluated_here=True,
        missing_excluded_candidate_fit_counterfactual=True,
        initial_fit_score_missing_so_no_pre_post_objective_comparison=True,
        support_recomputed=False,
        current_beam_rule_reconstruction_is_not_future_solver=True,
        captured_archives=archives,
        original_candidate_or_39_reads=0,
        original_SL_mask_video_image_reads=0,
        tracker_or_GPU_calls=0,
        inputs=inputs,
    )
    output.write_text(
        json.dumps(safe(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            dict(
                output=str(output),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                counts=result["counts"],
                selected=selected,
                retained=retained,
                body_row=[
                    {
                        k: r[k]
                        for k in (
                            "candidate_index",
                            "kind",
                            "parent",
                            "local_cost",
                            "path_cost",
                            "local_rank",
                            "path_rank",
                        )
                    }
                    for r in safe(table)
                    if r["kind"] == "body_row"
                ],
                best=[
                    {
                        k: safe(table)[i][k]
                        for k in (
                            "candidate_index",
                            "kind",
                            "parent",
                            "local_cost",
                            "path_cost",
                            "local_rank",
                            "path_rank",
                        )
                    }
                    for i in path_order[:6]
                ],
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    run()

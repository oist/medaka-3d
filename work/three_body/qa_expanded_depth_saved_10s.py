"""Independent saved-output arithmetic/codec QA; never opens a 39-state payload."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import compare_expanded_depth_instance_10s as comparison
import numpy as np
import run_instance_rescore_10s as diagnostic

proof, common = comparison.proof, comparison.common
WORK, REPO = comparison.WORK, comparison.WORK.parents[1]
TRACK = WORK / "expanded_depth_instance_rescore_ten_second_v1"
COMPARE = WORK / "expanded_depth_instance_ten_second_comparison_v1"
TRACK_SHA = "0ff5f6a31f020a2f3773176e5af60990ffb6bbf5fc7c67cbf505dd208d6e1d8c"
COMPARE_SHA = "bb6002ab37441a86daac6a05ed19176b512601e8ad8c30a1df620fb064848f59"
COMPARE_SOURCE_SHA = "628178b5083308df12e1a290111e5c97c0bb1dd11f0000d704be4f9ff8f35c72"
OUTPUT = WORK / "expanded_depth_ten_second_independent_qa_v1"
FIRST, START, STOP = 49800, 76360, 78020


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text("utf8"))


def signatures(arrays):
    return {key: proof.signature(value) for key, value in arrays.items()}


def same(left, right, name):
    require(proof.signature(left) == proof.signature(right), name)


def same_arrays(left, right, name):
    require(signatures(left) == signatures(right), name)


def archive(path):
    require("checkpoints" not in Path(path).parts, "39 payload access forbidden")
    with np.load(path, allow_pickle=False) as saved:
        result = {key: saved[key] for key in saved.files}
    require(all(not value.dtype.hasobject for value in result.values()), "object array")
    return result


def outputs(root, filename, expected):
    seal = read(root / filename)
    require(proof.sha(root / filename) == expected, "seal SHA")
    require(seal["stage"] == root.name and seal["status"] == "ok", "successful stage")
    expected_paths = {root / filename}
    skipped = []
    checked_bytes = 0
    for name, entry in seal["outputs"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), "output outside stage")
        require(Path(entry["path"]) == path, "output descriptor path")
        require(path.stat().st_size == entry["size_bytes"], "output size")
        expected_paths.add(path)
        if "checkpoints" in path.parts and path.name == "arrays.npz":
            skipped.append(dict(member=name, **entry))
            continue
        require(proof.sha(path) == entry["sha256"], "output SHA: " + name)
        checked_bytes += entry["size_bytes"]
    actual = {path.resolve() for path in root.rglob("*") if path.is_file()}
    require(actual == expected_paths, "complete stage file membership")
    plan = read(root / "plan.json")
    for path, digest in plan["source_sha256"].items():
        source = Path(path)
        require(seal["inputs"].get(path) == digest, "consumed source edge")
        require(proof.sha(source) == digest, "current source SHA")
        if root == TRACK:
            snapshot = "code_snapshot/" + source.relative_to(REPO).as_posix()
            require(seal["outputs"][snapshot]["sha256"] == digest, "source snapshot edge")
    return seal, dict(
        output_members=len(seal["outputs"]),
        hashed_members=len(seal["outputs"]) - len(skipped),
        hashed_bytes=checked_bytes,
        checkpoint_payload_not_read_or_hashed=skipped,
        source_members_verified=len(plan["source_sha256"]),
        exact_file_membership=True,
    )


def companions(candidate, summary, out):
    index = read(TRACK / "diagnostic_index.json")
    require([r["frame_idx"] for r in index["rows"]] == list(range(START, STOP)), "1660 index")
    for key in (
        "new_scoring_policy",
        "distance_backend",
        "expanded_depth_policy",
        "scientific_scope",
    ):
        require(index[key] == summary[key], "index policy")
    counts, gates = Counter(), Counter()
    arrays_verified = elements = penalty_rows = rank_changes = 0
    selected = []
    permutations = np.asarray(list(itertools.permutations(range(3))))
    for row in index["rows"]:
        frame = row["frame_idx"]
        meta = read(TRACK / f"diagnostics/f{frame}.json")
        name = f"diagnostics/f{frame}.npz"
        require(meta["archive"] == out["outputs"][name], "companion archive edge")
        saved = archive(TRACK / name)
        tree = diagnostic.restore_diagnostic(meta["tree"], saved)
        arrays_verified += len(saved)
        elements += sum(a.size for a in saved.values())
        n = row["candidate_count"]
        same(tree["candidate_index"], np.arange(n, dtype=np.int64), "candidate indices")
        require(tree["frame_idx"] == frame, "companion native")
        require(tree["candidate_pose_xyz"].shape == (n, 3, 3, 3), "candidate pose shape")
        require(tree["candidate_plan"].shape == (n, 3, 3, 3), "candidate plan shape")
        require(tree["policy"] == summary["new_scoring_policy"], "companion mask policy")
        require(tree["distance_backend"] == summary["distance_backend"], "companion sparse policy")
        require(
            tree["expanded_depth_policy"] == summary["expanded_depth_policy"],
            "companion depth policy",
        )
        counts[n] += 1
        policy = tree["policy"]
        contributions = np.zeros((n, 3), dtype=np.float64)
        eligible = []
        for vi, view in enumerate(policy["camera_order"]):
            data = tree["views"][view]
            gates[(view, data["reason"])] += 1
            raw = data["raw_uncapped_scene_cost"]
            normalized = raw / policy["bodies_per_view"]
            invalid = ~np.isfinite(raw)
            capped = np.clip(normalized, 0.0, policy["cap_per_view"])
            capped[invalid] = policy["cap_per_view"]
            if not data["eligible"]:
                capped[:] = 0
                require(data["neutral"] and data["assignment_costs"] is None, "neutral view")
            else:
                eligible.append(view)
                pair = data["body_instance_costs"]
                require(pair.shape == (n, 3, 3), "body-instance shape")
                joint = np.stack([pair[:, np.arange(3), p].sum(1) for p in permutations], -1)
                same(joint, data["assignment_costs"], "six assignment sums")
                choice = np.argmin(joint, axis=1)
                best = joint[np.arange(n), choice]
                same(best, raw, "best assignment cost")
                assignment = permutations[choice].copy()
                assignment[~np.isfinite(best)] = -1
                same(assignment, data["best_assignment"], "best assignment")
                ties = np.isfinite(joint) & np.isclose(
                    joint, best[:, None], rtol=0, atol=policy["observation_config"]["tie_atol"]
                )
                same(ties, data["optimal_assignment_mask"], "assignment ties")
            same(normalized, data["uncapped_mean_body_cost"], "body normalization")
            same(capped, data["capped_view_contribution"], "capped contribution")
            same(invalid, data["nonfinite_score"], "nonfinite flag")
            same((normalized > policy["cap_per_view"]) | invalid, data["cap_applied"], "cap flag")
            contributions[:, vi] = capped
        require(eligible == row["eligible_views"], "index eligible views")
        same(contributions, tree["capped_contributions_by_view"], "all view contributions")
        penalty = policy["weight"] * contributions.sum(1) / policy["fixed_camera_denominator"]
        same(penalty, tree["sixth_instance_term"], "sixth term")
        same(tree["base_local_cost"] + penalty, tree["adjusted_local_cost"], "local cost")
        same(
            tree["parent_path_cost"] + tree["adjusted_local_cost"],
            tree["adjusted_path_cost"],
            "path cost",
        )
        i = frame - FIRST
        matches = []
        fields = (
            ("candidate_pose_xyz", "pose_xyz"),
            ("adjusted_local_cost", "costs"),
            ("adjusted_path_cost", "path_costs"),
            ("original_five_loss_parts", "loss_parts"),
        )
        for j in range(n):
            if all(
                proof.signature(tree[a][j]) == proof.signature(candidate[b][i]) for a, b in fields
            ) and tree["candidate_kind"][j] == candidate["proposal_kind"][i]:
                matches.append(j)
        require(len(matches) == 1, f"unique finished candidate match {frame}: {matches}")
        j = matches[0]
        source = candidate["sources"][i]
        require(
            np.all((source == -1) | (source == tree["candidate_plan"][j])), "source vs proposal"
        )
        same(
            tree["parent_path_cost"][j],
            candidate["path_costs"][i - 1],
            "cumulative selected parent cost",
        )
        selected.append(j)
        penalty_rows += int((penalty > 0).sum())
        rank_changes += int(
            np.argmin(tree["base_local_cost"]) != np.argmin(tree["adjusted_local_cost"])
        )
        if (frame + 1 - START) % 400 == 0:
            print(json.dumps(dict(qa_through_native=frame + 1)), flush=True)
    return dict(
        frames=STOP - START,
        arrays_verified=arrays_verified,
        elements_verified=elements,
        candidate_counts=dict(counts),
        positive_penalty_candidate_rows=penalty_rows,
        local_best_changed_frames=rank_changes,
        gate_reason_counts={f"{v}: {reason}": count for (v, reason), count in gates.items()},
        all_finished_rows_uniquely_matched=True,
        selected_parent_costs_exact=True,
        all_cost_assignment_normalization_arithmetic_byte_exact=True,
    ), dict(
        frame_idx=np.arange(START, STOP, dtype=np.int64),
        selected_candidate_index=np.asarray(selected, np.int64),
    )


def comparison_numbers(candidates, summary):
    checked = {}
    for arm, candidate in candidates.items():
        method = summary["methods"][arm]
        part = {k: v[START - FIRST :] for k, v in candidate.items()}
        observed = part["observed"]
        interval = method["interval"]
        require(observed.sum(0).tolist() == interval["direct_by_slot_node"], "direct counts")
        support = archive(COMPARE / f"support/{arm}.npz")
        same(support["frame_idx"], part["frame_idx"], "support frames")
        present = part["sources"] >= 0
        same(present, support["valid_source"], "source validity")
        require(np.all(~observed | (present.sum(2) >= 2)), "observed requires two source views")
        strict = method["strict_support"]
        require(int(present.sum()) == strict["source_rows_checked"], "support source count")
        require(int(observed.sum()) == strict["observed_nodes_checked"], "support observed count")
        for key, maximum, threshold in (
            ("ray_distance_mm", "maximum_supported_ray_distance_mm", 2.0),
            ("native_pixel_error", "maximum_supported_pixel_error", 12.0),
        ):
            values = support[key][present]
            require(
                np.isfinite(values).all() and (values <= threshold).all(), "saved support threshold"
            )
            require(float(values.max()) == strict[maximum], "saved support maximum")
        geo = archive(COMPARE / f"geometry/{arm}.npz")
        pose = part["pose_xyz"]
        vector = np.diff(pose, axis=2)
        length = np.linalg.norm(vector, axis=-1)
        unit = vector / length[..., None]
        angle = np.degrees(np.arccos(np.clip((unit[..., 0, :] * unit[..., 1, :]).sum(-1), -1, 1)))
        same(length, geo["segment_length_mm"], "segment lengths")
        same(angle, geo["fold_angle_deg"], "fold angle")
        same(length / geo["seed_segment_length_mm"], geo["seed_length_ratio"], "seed ratios")
        same(observed[..., :-1] & observed[..., 1:], geo["direct_segment"], "direct segments")
        same(observed.all(-1), geo["direct_fold"], "direct fold")
        require(
            comparison.metrics.geometry._geometry_summary(geo) == interval["body_geometry"],
            "geometry summary",
        )
        z_summary, z_arrays = comparison.z_diagnostics(candidate)
        require(z_summary == method["z_distribution"], "z summary")
        same_arrays(z_arrays, archive(COMPARE / f"z/{arm}.npz"), "z arrays")
        motion = comparison.metrics.old.old.node_metrics(candidate, start=START, stop=STOP)
        require(motion == read(COMPARE / f"motion/{arm}.json"), "motion and missing runs")
        require(motion["by_slot_node"] == interval["node_metrics"], "node metrics summary")
        same_arrays(
            {k: v[np.asarray(comparison.DEVELOPMENT_FRAMES) - FIRST] for k, v in candidate.items()},
            archive(COMPARE / f"development_frames/{arm}.npz"),
            "development all21",
        )
        checked[arm] = dict(
            direct_nodes=int(observed.sum()),
            source_rows=int(present.sum()),
            z_counts={k: v["counts"] for k, v in z_summary["subsets"].items()},
        )
    a, b = (candidates[arm] for arm in comparison.ARMS)
    delta, rows, arrays = comparison.metrics.differences(a, b, START, STOP)
    require(delta == summary["support_delta"], "support delta summary")
    require(rows == read(COMPARE / "differences/direct_gain_loss.json"), "gain/loss rows")
    same_arrays(
        arrays,
        archive(COMPARE / "differences/all_native_transitions.npz"),
        "all native transitions",
    )
    runs = comparison.support_loss_runs(arrays["frame_idx"], arrays["support_loss"])
    require(
        runs
        == summary["support_loss_runs"]
        == read(COMPARE / "differences/contiguous_support_losses.json"),
        "support loss runs",
    )
    geometry, arrays = common.geometry_deltas(a, b)
    geometry["array_axis0_arm_order"] = list(comparison.ARMS)
    require(geometry == summary["geometry_deltas"], "raw slot geometry summary")
    same_arrays(arrays, archive(COMPARE / "differences/raw_slot_geometry.npz"), "raw slot geometry")
    prefix, arrays = common.same_horizon_prefix(a, b)
    require(prefix == summary["selected_prefix_path_changes"], "same-horizon prefix summary")
    same_arrays(
        arrays, archive(COMPARE / "differences/selected_prefix_paths.npz"), "same-horizon prefix"
    )
    return checked


def main():
    require(not OUTPUT.exists(), "fresh QA output required")
    OUTPUT.mkdir()
    result = dict(
        schema="expanded_depth_ten_second_independent_saved_output_qa_v1", status="failed"
    )
    try:
        require(proof.sha(comparison.__file__) == COMPARE_SOURCE_SHA, "reviewed comparison source")
        track_seal, tracking_outputs = outputs(TRACK, "tracking.provenance.json", TRACK_SHA)
        _, comparison_outputs = outputs(COMPARE, "comparison.provenance.json", COMPARE_SHA)
        evidence = proof.Evidence()
        dense = comparison.load_headers(
            comparison.CONTROL_STAGE,
            comparison.CONTROL_SHA,
            comparison.CONTROL_RUNNER_SHA,
            evidence,
        )
        variant = comparison.load_headers(
            comparison.VARIANT_STAGE, TRACK_SHA, comparison.VARIANT_RUNNER_SHA, evidence
        )
        lineage = comparison.validate_domains(dense, variant)
        comparison.bind_rationale_edges(variant, evidence)
        require(lineage == read(COMPARE / "producer_lineage.json"), "producer lineage")
        summary = read(COMPARE / "summary.json")
        track_summary = read(TRACK / "summary.json")
        candidates = {}
        for arm in comparison.ARMS:
            arrays = archive(COMPARE / f"candidates/{arm}.npz")
            comparison.metrics.validate_candidate(arrays, FIRST, STOP)
            require(
                signatures(arrays) == summary["candidate_references"][arm]["arrays"],
                "all21 reference signatures",
            )
            candidates[arm] = arrays
        actual_final = archive(TRACK / "candidate.npz")
        same_arrays(
            actual_final, candidates[comparison.ARMS[1]], "producer and comparison all21 final"
        )
        require(
            signatures(actual_final) == track_summary["candidate_arrays"],
            "producer candidate all21",
        )
        diagnostics, selected = companions(actual_final, track_summary, track_seal)
        numbers = comparison_numbers(candidates, summary)
        evidence.final_guard()
        require(not comparison.sys.modules["torch"].cuda.is_initialized(), "CUDA initialized")
        selection_path = OUTPUT / "selected_candidates.npz"
        np.savez_compressed(selection_path, **selected)
        result.update(
            status="ok",
            producer_seal_sha256=TRACK_SHA,
            comparison_seal_sha256=COMPARE_SHA,
            qa_source_sha256=proof.sha(__file__),
            tracking_outputs=tracking_outputs,
            comparison_outputs=comparison_outputs,
            diagnostics=diagnostics,
            numerical_checks=numbers,
            all21_final_candidate_arrays_verified=True,
            imported39_signatures_and_metadata_bound=True,
            producer39_roundtrip_prefix_finish_proofs_reused_not_recomputed=True,
            checkpoint39_payload_read_or_hashed=False,
            raw_video_mask_or_SL_payload_read=False,
            GPU_used=False,
            strict_support_saved_residuals_checked=True,
            SL_projections_independently_recomputed=False,
            physical_ID_or_depth_truth=False,
            automatic_adoption=False,
            selected_candidates=dict(
                path=str(selection_path),
                sha256=proof.sha(selection_path),
                arrays=signatures(selected),
            ),
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        (OUTPUT / "qa.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf8"
        )
    print(
        json.dumps(
            dict(status=result["status"], diagnostics=result["diagnostics"], output=str(OUTPUT)),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

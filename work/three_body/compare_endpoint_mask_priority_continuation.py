"""Same91300-terminal Sparse, Nodewise and endpoint-priority saved-path comparison.

Read nine fields from each sealed21-field candidate and three thin priority
companions. Stored support is not rescored; numeric body slots are not fish IDs.
The six chest permutations measure geometric correspondence, not ID continuity.
"""

from __future__ import annotations

import argparse
from itertools import combinations, permutations
from pathlib import Path

import compare_body_row_continuation as old
import numpy as np
import run_instance_rescore_10s as codec

common, proof, require, WORK = old.common, old.proof, old.require, old.WORK
FIRST, START, STOP, CHUNK = 49800, 89640, 91300, 1660
FRAMES = (90141, 90258, 90259)
VIEWS = ("top_cam", "side1_cam", "side2_cam")
ARMS = ("sparse_control", "nodewise", "endpoint_priority")
STAGE = "endpoint_mask_priority_ten_second_comparison_v1"
CONTROL = WORK / "nodewise_negative_same_horizon_control_v1"
CONTROL_SHA = "3c871a046fbd5fe6db52dfd10e8f0b907e7c34ecbc4a6cde858a9d0aeb9c3f26"
PARENT_SHA = "596b23afe5fb572a66a72b96b0b1f5ba9b3971bfef4dd43cb8ee6b32459e32ad"
INITIAL_SHA = "f76a4031f831bdd5b2e0516237242713e981a305c40f2b57966c90fa625a6e70"
CONTROL_CHECKPOINT_SHA = "d1fb9abca83edd772841ac90740454d88370b121f6efc0524bb9187925437bf4"
SPECS = {
    "nodewise": (
        "nodewise_body_row_negative_ten_second_continuation_v1",
        "run_nodewise_body_row_negative_continuation.py",
        "4f2d31a383a8d59320ef0c09f8dcddf645dfb20e0849bd2f1dc27398d719669e",
    ),
    "endpoint_priority": (
        "endpoint_mask_priority_ten_second_continuation_v1",
        "run_endpoint_mask_priority_continuation.py",
        "6d972c0f34c435f1036eca6cae3fc0563d2853b1a63f3273e89eb3d40b06d819",
    ),
}
PINS = {
    "compare_body_row_continuation.py": (
        "474640d350944dd7b5969a76526b2a183d1573dd7a8fdf2ec24940a6449b7cfa"
    ),
    "compare_instance_rescore_10s.py": (
        "bb7653e5f7d71f2b4c078418dee93433fb4a1fc3fdefd7f38ef9c4694ae9040a"
    ),
    "run_instance_rescore_10s.py": (
        "91a66646f89a6abb59b6d60bc1b357b383167c21eb9d40bc47491802d05e9544"
    ),
    "compare_temporal_point64_v2.py": (
        "2461f383d7c69223c98cb9371058373361a7fb30a409a37d617b60fc2db53790"
    ),
}
FIELDS = (
    "frame_idx",
    "pose_xyz",
    "observed",
    "sources",
    "identity_review",
    "costs",
    "path_costs",
    "loss_parts",
    "proposal_kind",
)
ALL21 = set(FIELDS) | {
    "mask_supported",
    "body_supported",
    "missing",
    "score_margin",
    "hypotheses",
    "proposal_flow_nodes",
    "proposal_flow_parent",
    "node_missing",
    "direct_pectoral_age",
    "xyz",
    "accepted_xyz",
    "body_xyz",
}
PERMUTATIONS = np.asarray(list(permutations(range(3))), dtype=np.int64)


def read_stage(root, digest, stage, seal_name, evidence):
    root = Path(root).resolve()
    require(root.is_relative_to(WORK), "workspace producer required")
    evidence.add(root / seal_name, digest)
    seal = proof.read_json(root / seal_name)
    require(seal["status"] == "ok" and seal["stage"] == stage, "successful exact producer required")

    def document(name):
        return proof.read_json(common.bound_member(root, seal, name + ".json", evidence))

    result = dict(root=root, seal=seal, digest=digest, summary=document("summary"))
    require(
        result["summary"]["schema"] == stage and result["summary"]["status"] == "ok",
        "successful summary differs",
    )
    if seal_name == "control.provenance.json":
        result["metadata"] = document("checkpoint_metadata")
    else:
        result.update(
            plan=document("plan"), imported=document("state_import"), index=document("input_index")
        )
    return result


def validate_headers(loaded):
    require(tuple(loaded) == ARMS, "fixed three method axes")
    control = loaded[ARMS[0]]["summary"]
    meta = loaded[ARMS[0]]["metadata"]
    require(
        control["retained_interval"] == [FIRST, STOP]
        and control["comparison_interval"] == [START, STOP]
        and control["terminal_horizon"] == STOP
        and control["parent_tracking_sha256"] == PARENT_SHA
        and control["control_checkpoint_manifest_sha256"] == CONTROL_CHECKPOINT_SHA
        and control["tracking_performed"] is False
        and control["original_candidate_116200_used"] is False,
        "same-terminal Sparse control required",
    )
    require(
        [
            meta[k]
            for k in ("previous_frame", "processed_frames", "initialize_count", "restore_count")
        ]
        == [STOP - 1, STOP - FIRST, 1, 6],
        "control lifecycle differs",
    )
    for arm in ARMS[1:]:
        item = loaded[arm]
        summary, plan, imported, index = (item[k] for k in ("summary", "plan", "imported", "index"))
        producer = str((WORK / SPECS[arm][1]).resolve())
        require(
            item["seal"]["inputs"].get(producer)
            == plan["source_sha256"].get(producer)
            == summary["source_sha256"].get(producer)
            == SPECS[arm][2],
            "producer source differs",
        )
        for doc in (summary, plan):
            require(
                doc["process_interval"] == [START, STOP]
                and doc["retained_interval"] == [FIRST, STOP]
                and doc["scientific_interval"] == [66400, STOP]
                and doc["parent_tracking_sha256"] == PARENT_SHA
                and doc["parent_checkpoint_sha256"] == INITIAL_SHA
                and doc["initialize_count"] == 1
                and doc["restore_count"] == 7
                and doc["new_begin_calls"] == doc["external_pose_resets"] == 0
                and doc["mask_split_priority_applied"] is (arm == "endpoint_priority"),
                "native extent, inherited state or treatment differs",
            )
        require(
            summary["finish_calls"] == 1
            and summary["all39_roundtrip"] is True
            and summary["prior_history_prefix_unchanged"] is True
            and summary["finish_matches_checkpoint_backtrace"] is True,
            "completed continuous backtrace required",
        )
        require(
            plan["original_binding"] == index["original_binding"] == control["original_binding"],
            "same seed/calibration/SL/source binding required",
        )
        require(
            plan["tracker_context"] == summary["tracker_context"] == imported["new_context"],
            "tracker context differs",
        )
        require(
            imported["status"] == "ok"
            and imported["previous_frame"] == START - 1
            and imported["processed_frames"] == START - FIRST
            and imported["effective_from_native"] == START
            and imported["added_restore_count"] == 1
            and imported["new_begin_calls"] == 0
            and imported["all_arrays_byte_preserved"] is True
            and imported["original_metadata_unchanged"] is True
            and len(imported["arrays"]) == 39,
            "verified all39 migration required",
        )
        original_meta = imported["original_metadata"]
        require(
            [
                original_meta[k]
                for k in ("previous_frame", "processed_frames", "initialize_count", "restore_count")
            ]
            == [START - 1, START - FIRST, 1, 6],
            "different initial Sparse lifecycle",
        )
        require(
            original_meta["context"] == meta["context"], "control and initial Sparse config differ"
        )
        count = (STOP - FIRST) // CHUNK
        require(len(index["prefix_chunks"]) == count, "complete25 source chunks required")
        for n, row in enumerate(index["prefix_chunks"]):
            require(
                (row["index"], row["start"], row["stop"])
                == (n, FIRST + n * CHUNK, FIRST + (n + 1) * CHUNK),
                "source native gap",
            )
    left, right = loaded["nodewise"], loaded["endpoint_priority"]
    require(left["index"] == right["index"], "different consumed source25 prefix")
    require(
        left["imported"]["arrays"] == right["imported"]["arrays"]
        and left["imported"]["original_metadata"] == right["imported"]["original_metadata"],
        "different original39 signatures or metadata",
    )
    context = dict(right["plan"]["tracker_context"])
    policy = context.pop("endpoint_mask_priority_policy")
    context["implementation"] = left["plan"]["tracker_context"]["implementation"]
    require(
        context == left["plan"]["tracker_context"]
        and policy["effective_from_native"] == START
        and policy["original_fit_local_path_costs_and_five_plus_sixth_terms_unchanged"] is True,
        "only endpoint beam priority/review policy may differ",
    )


def validate_candidate_fields(candidate):
    count = STOP - FIRST
    shapes = {
        "frame_idx": ((), "i"),
        "pose_xyz": ((3, 3, 3), "f"),
        "observed": ((3, 3), "b"),
        "sources": ((3, 3, 3), "i"),
        "identity_review": ((3,), "b"),
        "costs": ((), "f"),
        "path_costs": ((), "f"),
        "loss_parts": ((5,), "f"),
        "proposal_kind": ((), "U"),
    }
    require(set(candidate) == set(FIELDS), "nine decoded candidate fields required")
    for key, (suffix, kind) in shapes.items():
        a = candidate[key]
        require(a.shape == (count, *suffix) and a.dtype.kind == kind, "candidate axes: " + key)
        if kind == "f":
            require(a.dtype == np.float64 and np.isfinite(a).all(), "finite FP64 candidate: " + key)
    require(
        candidate["frame_idx"].dtype == np.int64
        and np.array_equal(candidate["frame_idx"], np.arange(FIRST, STOP)),
        "retained native frames",
    )
    sources = candidate["sources"]
    require(
        np.all(sources >= -1) and not np.any(candidate["observed"] & ((sources >= 0).sum(2) < 2)),
        "saved direct support source count differs",
    )


def load_candidate(item, evidence):
    signatures = item["summary"]["candidate_arrays"]
    require(set(signatures) == ALL21 and len(signatures) == 21, "producer21 signatures required")
    path = common.bound_member(item["root"], item["seal"], "candidate.npz", evidence)
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == 21 and set(archive.files) == ALL21, "archive21 membership")
        result = {key: archive[key] for key in FIELDS}
    require(
        common.signature(result) == {key: signatures[key] for key in FIELDS}, "decoded signatures"
    )
    validate_candidate_fields(result)
    return result


def summarize(candidate):
    keep = (candidate["frame_idx"] >= START) & (candidate["frame_idx"] < STOP)
    frames = candidate["frame_idx"][keep]
    require(np.array_equal(frames, np.arange(START, STOP)), "full comparison interval")
    pose, observed, sources = (candidate[k][keep] for k in ("pose_xyz", "observed", "sources"))
    steps = np.linalg.norm(np.diff(pose, axis=0), axis=-1)
    both = observed[1:] & observed[:-1]
    lo = np.min(np.where(sources >= 0, sources, np.iinfo(np.int32).max), axis=-1)
    mixed = (sources.max(-1) >= 0) & (lo != sources.max(-1))
    kinds = candidate["proposal_kind"][keep]
    require(not mixed[kinds == "body_row"].any(), "body-row proposal lost same-row property")
    events = [
        dict(
            previous_frame=int(frames[t]),
            frame_idx=int(frames[t + 1]),
            raw_slot=int(b),
            node=int(n),
            distance_mm=float(steps[t, b, n]),
        )
        for t, b, n in np.argwhere(both & (steps > 10.0))
    ]
    direct_max = np.max(np.where(both, steps, -np.inf), axis=0)
    return dict(
        frames=len(frames),
        fish_frames=len(frames) * 3,
        node_order=["mouth", "chest", "tail"],
        directly_observed_by_slot_node=observed.sum(0).tolist(),
        missing_by_slot_node=(~observed).sum(0).tolist(),
        longest_missing_run_by_slot_node=old.longest_true(~observed).tolist(),
        all_three_chests_direct_frames=int(observed[:, :, 1].all(-1).sum()),
        directly_supported_adjacent_pairs_by_slot_node=both.sum(0).tolist(),
        direct_adjacent_steps_over_10mm=events,
        direct_max_step_by_slot_node_mm=[
            [None if not np.isfinite(x) else float(x) for x in row] for row in direct_max
        ],
        all_estimated_max_step_by_slot_node_mm=steps.max(0).tolist(),
        mixed_source_rows_by_slot_camera=mixed.sum(0).tolist(),
        mixed_source_rows_by_camera=mixed.sum((0, 1)).tolist(),
        body_frames_with_any_mixed_source_rows=int(mixed.any(-1).sum()),
        identity_review_by_slot=candidate["identity_review"][keep].sum(0).tolist(),
        final_backtrace_kind_counts={str(k): int(np.sum(kinds == k)) for k in np.unique(kinds)},
        support_recalculated=False,
        raw_slots_are_physical_IDs=False,
    )


def compare_arrays(candidates):
    require(tuple(candidates) == ARMS, "fixed method axes")
    samples, metrics, pairwise, permutation_arrays = (
        {},
        {},
        {},
        {"permutations": PERMUTATIONS.copy()},
    )
    keep = candidates[ARMS[0]]["frame_idx"] >= START
    permutation_arrays["frame_idx"] = candidates[ARMS[0]]["frame_idx"][keep].copy()
    for arm, c in candidates.items():
        require(
            np.array_equal(c["frame_idx"], candidates[ARMS[0]]["frame_idx"]), "same native axes"
        )
        metrics[arm] = summarize(c)
        probes = np.isin(c["frame_idx"], FRAMES)
        require(np.array_equal(c["frame_idx"][probes], FRAMES), "three fixed probes")
        samples.update({arm + "_" + k: c[k][probes].copy() for k in FIELDS})
    for a, b in combinations(ARMS, 2):
        ca, cb = candidates[a], candidates[b]
        pa, pb = ca["pose_xyz"][keep], cb["pose_xyz"][keep]
        delta = np.linalg.norm(pa - pb, axis=-1)
        chest_rms = np.stack(
            [
                np.sqrt(np.mean(np.sum((pa[:, :, 1] - pb[:, p, 1]) ** 2, axis=-1), axis=-1))
                for p in PERMUTATIONS
            ],
            axis=-1,
        )
        best = chest_rms.min(-1)
        ties = np.isclose(chest_rms, best[:, None], rtol=0, atol=1e-12)
        key = a + "_vs_" + b
        permutation_arrays[key + "_chest_rms_mm"] = chest_rms
        permutation_arrays[key + "_optimal_permutation_mask"] = ties
        direct = ca["observed"][keep, :, 1].all(-1) & cb["observed"][keep, :, 1].all(-1)
        pairwise[key] = dict(
            raw_slot_pose_delta_mm=dict(median=float(np.median(delta)), maximum=float(delta.max())),
            geometric_chest_min_rms_mm=dict(
                median=float(np.median(best)), maximum=float(best.max())
            ),
            all_three_chests_direct_in_both_frames=int(direct.sum()),
            unique_nonidentity_chest_permutation_frames=int(
                np.sum((ties.sum(-1) == 1) & ~ties[:, 0])
            ),
            permutation_tie_frames=int(np.sum(ties.sum(-1) > 1)),
            permutation_maps_A_raw_slot_to_B_raw_slot=True,
            framewise_geometry_is_not_temporal_ID_alignment=True,
            chosen_prefix_pose_differences=int(
                np.any(ca["pose_xyz"][~keep] != cb["pose_xyz"][~keep], axis=(1, 2, 3)).sum()
            ),
        )
    return metrics, samples, pairwise, permutation_arrays


def array_keys(tree):
    if isinstance(tree, dict):
        if "__array__" in tree:
            require(set(tree) == {"__array__", "signature"}, "invalid codec descriptor")
            return [tree["__array__"]]
        return [k for v in tree.values() for k in array_keys(v)]
    if isinstance(tree, list):
        return [k for v in tree for k in array_keys(v)]
    return []


def thin_companion(item, prefix, frame, fields, evidence, *, endpoint=False):
    root, seal = item["root"], item["seal"]
    doc = proof.read_json(common.bound_member(root, seal, f"{prefix}/f{frame}.json", evidence))
    tree = doc["tree"]
    schema = (
        "endpoint_mask_frame_diagnostics_v1"
        if endpoint
        else "instance_rescore_frame_diagnostics_v1"
    )
    require(tree["schema"] == schema and tree["frame_idx"] == frame, "companion frame/schema")
    if endpoint:
        require(
            tree["frame_completed"] is True and tree["view_order"] == list(VIEWS),
            "completed endpoint frame",
        )
    selected = {key: tree[key] for key in fields}
    if endpoint:
        selected["views"] = {
            v: {"classification_by_body": tree["views"][v]["classification_by_body"]} for v in VIEWS
        }
    relative = f"{prefix}/f{frame}.npz"
    require(doc["archive"] == seal["outputs"][relative], "companion archive binding")
    path = common.bound_member(root, seal, relative, evidence)
    all_keys, needed = array_keys(tree), array_keys(selected)
    require(
        len(set(all_keys)) == len(all_keys) and len(set(needed)) == len(needed),
        "repeated codec keys",
    )
    with np.load(path, allow_pickle=False) as archive:
        require(
            len(archive.files) == len(all_keys) and set(archive.files) == set(all_keys),
            "codec archive membership",
        )
        arrays = {key: archive[key] for key in needed}
    return codec.restore_diagnostic(selected, arrays)


def bind_final_probe(candidate, frame, fitted, endpoint):
    rows = np.flatnonzero(candidate["frame_idx"] == frame)
    require(len(rows) == 1, "one final native row required")
    row = int(rows[0])
    poses = fitted["candidate_pose_xyz"]
    n = len(poses)
    require(
        np.array_equal(fitted["candidate_index"], np.arange(n))
        and np.array_equal(endpoint["candidate_index"], np.arange(n)),
        "same candidate indexing",
    )
    require(
        np.array_equal(endpoint["candidate_local_cost"], fitted["adjusted_local_cost"])
        and np.array_equal(endpoint["candidate_path_cost"], fitted["adjusted_path_cost"])
        and np.array_equal(endpoint["candidate_parent"], fitted["candidate_parent"])
        and endpoint["candidate_kind"] == fitted["candidate_kind"],
        "two companion banks differ",
    )
    match = np.all(poses == candidate["pose_xyz"][row], axis=(1, 2, 3))
    for target, source in (("costs", "adjusted_local_cost"), ("path_costs", "adjusted_path_cost")):
        match &= fitted[source] == candidate[target][row]
    match &= np.all(fitted["original_five_loss_parts"] == candidate["loss_parts"][row], axis=1)
    match &= np.asarray(fitted["candidate_kind"]) == candidate["proposal_kind"][row]
    sources = candidate["sources"][row]
    match &= np.all(
        (sources < 0)[None] | (fitted["candidate_plan"] == sources[None]), axis=(1, 2, 3)
    )
    matched = np.flatnonzero(match)
    require(len(matched) > 0, "final selected pose/cost/plan was not in saved bank")
    retained = endpoint["selected_candidate_indices"]
    require(np.isin(matched, retained).any(), "no matching final candidate was retained")
    identical_pose = np.flatnonzero(np.all(poses == candidate["pose_xyz"][row], axis=(1, 2, 3)))
    split = endpoint["candidate_split_by_body"]
    unknown = endpoint["unknown_by_candidate_body_view"]
    require(
        split.shape == (n, 3)
        and split.dtype == bool
        and unknown.shape == (n, 3, 3)
        and unknown.dtype == bool,
        "three-body/view endpoint shapes",
    )
    labels = np.stack([endpoint["views"][v]["classification_by_body"] for v in VIEWS], axis=-1)
    known_labels = (
        "same_instance",
        "split_instances",
        "uncovered_endpoint",
        "ambiguous_endpoint",
        "unavailable",
        "invalid_geometry",
        "invalid_projection",
    )
    require(
        labels.shape == (n, 3, 3)
        and labels.dtype.kind == "U"
        and np.isin(labels, known_labels).all()
        and np.array_equal(split, (labels == "split_instances").any(-1))
        and np.array_equal(unknown, ~np.isin(labels, ("same_instance", "split_instances"))),
        "classification, split and unknown flags disagree",
    )
    for values in (
        split,
        unknown,
        *[endpoint["views"][v]["classification_by_body"] for v in VIEWS],
    ):
        require(
            all(np.array_equal(values[i], values[matched[0]]) for i in identical_pose),
            "identical pose produced conflicting endpoint classifications",
        )
    first = int(matched[0])
    require(
        not np.any(split[first] & ~candidate["identity_review"][row]),
        "retained split review was cleared",
    )
    return dict(
        frame_idx=frame,
        matching_candidate_indices=matched.tolist(),
        matching_retained_candidate_indices=[int(i) for i in matched if i in retained],
        identical_pose_candidate_indices=identical_pose.tolist(),
        split_by_raw_body=split[first].tolist(),
        unknown_by_raw_body_view=unknown[first].tolist(),
        classification_by_view={
            v: endpoint["views"][v]["classification_by_body"][first].tolist() for v in VIEWS
        },
        saved_observed=candidate["observed"][row].tolist(),
        saved_identity_review=candidate["identity_review"][row].tolist(),
        source_plan_subset_verified=True,
        parent_chain_independently_decoded=False,
        scope=(
            "three probes only; matched final geometry, "
            "not current state0 or full-interval split counts"
        ),
    )


def load_priority_probes(item, candidate, evidence):
    fit_fields = (
        "candidate_index",
        "candidate_pose_xyz",
        "candidate_parent",
        "candidate_plan",
        "candidate_kind",
        "adjusted_local_cost",
        "adjusted_path_cost",
        "original_five_loss_parts",
    )
    end_fields = (
        "candidate_index",
        "candidate_parent",
        "candidate_kind",
        "candidate_local_cost",
        "candidate_path_cost",
        "candidate_split_by_body",
        "unknown_by_candidate_body_view",
        "selected_candidate_indices",
    )
    return [
        bind_final_probe(
            candidate,
            f,
            thin_companion(item, "diagnostics", f, fit_fields, evidence),
            thin_companion(
                item, "endpoint_mask_diagnostics", f, end_fields, evidence, endpoint=True
            ),
        )
        for f in FRAMES
    ]


def run(nodewise_root, nodewise_sha, priority_root, priority_sha, output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and not output.exists(), "fresh workspace comparison output"
    )
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        for name, digest in PINS.items():
            evidence.add(WORK / name, digest)
        evidence.add(Path(__file__))
        priority = read_stage(
            priority_root,
            priority_sha,
            SPECS["endpoint_priority"][0],
            "tracking.provenance.json",
            evidence,
        )
        nodewise = read_stage(
            nodewise_root, nodewise_sha, SPECS["nodewise"][0], "tracking.provenance.json", evidence
        )
        control = read_stage(
            CONTROL, CONTROL_SHA, CONTROL.name, "control.provenance.json", evidence
        )
        loaded = dict(zip(ARMS, (control, nodewise, priority), strict=True))
        validate_headers(loaded)
        candidates = {arm: load_candidate(item, evidence) for arm, item in loaded.items()}
        metrics, samples, pairwise, permutation_arrays = compare_arrays(candidates)
        probes = load_priority_probes(priority, candidates["endpoint_priority"], evidence)
        out.write("review_samples.npz", samples)
        out.write("review_samples_signatures.json", common.signature(samples))
        out.write("chest_permutations.npz", permutation_arrays)
        out.write("chest_permutations_signatures.json", common.signature(permutation_arrays))
        out.write("priority_final_probe_classification.json", probes)
        refs = {
            arm: dict(
                root=str(item["root"]),
                seal_sha256=item["digest"],
                candidate=item["seal"]["outputs"]["candidate.npz"],
                all21_producer_signatures=item["summary"]["candidate_arrays"],
                decoded_fields=list(FIELDS),
            )
            for arm, item in loaded.items()
        }
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                arms=list(ARMS),
                process_interval=[START, STOP],
                terminal_horizon=STOP,
                frames=list(FRAMES),
                methods=metrics,
                pairwise=pairwise,
                candidate_references=refs,
                priority_current_beam_reported=priority["summary"]["endpoint_mask"],
                priority_final_classification_scope=list(FRAMES),
                full_final_path_split_count_computed=False,
                physical_ID_truth=False,
                automatic_adoption=False,
                chosen_prefix_is_not_the_retained_beam_history=True,
                support_recomputed=False,
                old_input_or_video_or_mask_or_39_payload_read=False,
            ),
        )
        source_paths = [WORK / name for name in PINS] + [Path(__file__).resolve()]
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                source_sha256={
                    str(p.resolve()): evidence.paths[str(p.resolve())] for p in source_paths
                },
                process_interval=[START, STOP],
                retained_interval=[FIRST, STOP],
                terminal_horizon=STOP,
                parent_tracking_sha256=PARENT_SHA,
                parent_checkpoint_sha256=INITIAL_SHA,
                original_binding=control["summary"]["original_binding"],
                same_input_prefix_chunks=25,
                original39_metadata_and_signatures_equal=True,
                candidate_references=refs,
                decoded_candidate_fields=list(FIELDS),
                numeric_verified_undecoded_fields=False,
                chest_permutations=PERMUTATIONS.tolist(),
                chest_tie_atol_mm=1e-12,
                original_3D_support_thresholds_reapplied=False,
                frames=list(FRAMES),
            ),
        )
        evidence.final_guard()
        out.final_guard()
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
    return seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodewise-root", type=Path, default=WORK / SPECS["nodewise"][0])
    parser.add_argument("--nodewise-sha", required=True)
    parser.add_argument("--priority-root", type=Path, default=WORK / SPECS["endpoint_priority"][0])
    parser.add_argument("--priority-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.nodewise_root, args.nodewise_sha, args.priority_root, args.priority_sha, args.output)

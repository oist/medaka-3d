"""Saved final five-minute Sparse versus unmasked Nodewise geometry/support.

Six selected candidate fields are decoded once per arm. Stored direct support
is not rescored. Every framewise body permutation is geometric correspondence,
not a temporal identity solution or biological ground truth. Nine development
probes were selected from known failures; they are not held-out validation.
"""

from __future__ import annotations

import argparse
import copy
from itertools import permutations
from pathlib import Path

import compare_body_row_continuation as old
import numpy as np

common, proof, require, WORK = old.common, old.proof, old.require, old.WORK
FIRST, START, STOP, CHUNK = 49800, 66400, 116200, 1660
SPARSE_START = 78020
STAGE = "nodewise_body_row_five_minute_comparison_v1"
ARMS = ("sparse", "nodewise")
FRAMES = (80425, 82720, 82727, 82728, 88373, 90141, 90153, 90258, 90259)
FIELDS = ("frame_idx", "pose_xyz", "observed", "sources", "identity_review", "proposal_kind")
ALL21 = set(FIELDS) | {
    "mask_supported",
    "body_supported",
    "missing",
    "costs",
    "path_costs",
    "loss_parts",
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
EXECUTION_ENABLED = True
SPARSE_STAGE = "sparse_instance_five_minute_continuation_v1"
NODEWISE_STAGE = "nodewise_body_row_five_minute_continuation_v1"
SPARSE_SHA = "596b23afe5fb572a66a72b96b0b1f5ba9b3971bfef4dd43cb8ee6b32459e32ad"
SPARSE_RUNNER_SHA = "3f5141a5aeec6fadc464ad242ac19fe13e227065af6f1188c638af4c58129096"
NODEWISE_RUNNER_SHA = "0ea3d27b939a804c081d6249c7462498ee0a16504e623e021037b0fc32f50b64"
ORIGINAL_STAGE = "long_continuation_through66400_v1"
ORIGINAL_SHA = "9915c44204c7acaee0abcada3a3fa2dfa82b7ac017c0b7d4fc9b50b628d48a6e"
ORIGINAL_CHECKPOINT_SHA = "0f8db6356175de3223589576b4539a3a73977eb8151324b94340c5b1b174730f"
ORIGINAL_METADATA_SHA = "6f79853f019fa6ca4f298bb39178dd70e5ca48dae2bd8cef65191a1ea16b0d98"
SPARSE_COMPARISON = "sparse_instance_five_minute_comparison_v1"
SPARSE_COMPARISON_SHA = "0cf955b7a2d15457d958c72ff647b1a2bc843e51039e63263e8788081454e909"
PINS = {
    "temporal_point_core.py": ("3ec308ca7877338f8677c0703d6f180a9d5658c4858b9868451e1152b7d4e25b"),
    "compare_body_row_continuation.py": (
        "474640d350944dd7b5969a76526b2a183d1573dd7a8fdf2ec24940a6449b7cfa"
    ),
    "compare_instance_rescore_10s.py": (
        "bb7653e5f7d71f2b4c078418dee93433fb4a1fc3fdefd7f38ef9c4694ae9040a"
    ),
    "compare_temporal_point64_v2.py": (
        "2461f383d7c69223c98cb9371058373361a7fb30a409a37d617b60fc2db53790"
    ),
}
NODEWISE_COMPONENTS = {
    "nodewise_original_state_import.py": (
        "25801a84fd848d312dc902852124b2c215b04f0ca75e44293cf77829cf1ab8e7"
    ),
    "nodewise_body_row_tracker.py": (
        "b574ce26481c2c1457ceb1428659aa7bce25783e36957fd2eb9b6399f52b8d16"
    ),
}


def document(item, name, evidence):
    return proof.read_json(common.bound_member(item["root"], item["seal"], name, evidence))


def read_producer(root, digest, stage, evidence):
    root = Path(root).resolve()
    require(root.is_relative_to(WORK) and proof.valid_sha(digest), "explicit workspace producer")
    evidence.add(root / "tracking.provenance.json", digest)
    seal = proof.read_json(root / "tracking.provenance.json")
    common.exact_fields(seal, dict(stage=stage, status="ok"), "successful final producer required")
    item = dict(root=root, seal=seal, sha256=digest)
    for name in ("summary", "plan", "input_index"):
        item[name] = document(item, name + ".json", evidence)
    return item


def validate_completion(item, arm, runner_sha, evidence):
    summary, plan, index = (item[k] for k in ("summary", "plan", "input_index"))
    is_new = arm == "nodewise"
    stage, resume, restores = (
        (NODEWISE_STAGE, START, 4) if is_new else (SPARSE_STAGE, SPARSE_START, 6)
    )
    expected = dict(
        schema=stage,
        scientific_interval=[START, STOP],
        retained_interval=[FIRST, STOP],
        process_interval=[resume, STOP],
        initialize_count=1,
        restore_count=restores,
        new_begin_calls=0,
        external_pose_resets=0,
    )
    for doc in (summary, plan):
        common.exact_fields(doc, expected, "same final native/lifecycle contract required")
    common.exact_fields(
        summary,
        dict(
            status="ok",
            native_frames=STOP - START,
            finish_calls=1,
            all39_roundtrip=True,
            prior_history_prefix_unchanged=True,
            finish_matches_checkpoint_backtrace=True,
        ),
        "completed final backtrace required",
    )
    runner = (
        "run_nodewise_body_row_five_minutes.py"
        if is_new
        else "continue_sparse_instance_five_minutes.py"
    )
    source = str((WORK / runner).resolve())
    require(proof.valid_sha(runner_sha), "explicit completed runner SHA required")
    require(
        runner_sha == (NODEWISE_RUNNER_SHA if is_new else SPARSE_RUNNER_SHA),
        "unreviewed producer source",
    )
    require(
        plan["source_sha256"].get(source) == item["seal"]["inputs"].get(source) == runner_sha,
        "actual runner source edge differs",
    )
    evidence.add(source, runner_sha)
    if is_new:
        require(
            summary["candidate"] == item["seal"]["outputs"]["candidate.npz"],
            "final candidate descriptor differs",
        )
        require(summary["source_sha256"] == plan["source_sha256"], "producer source maps differ")
        for name, digest in NODEWISE_COMPONENTS.items():
            path = str((WORK / name).resolve())
            require(
                plan["source_sha256"].get(path) == item["seal"]["inputs"].get(path) == digest,
                "reviewed Nodewise component differs",
            )
            evidence.add(path, digest)
        common.exact_fields(
            summary,
            dict(
                scientific_native_frames=STOP - START,
                diagnostic_frames=STOP - START,
                body_row_and_sparse_backend_effective_from_native=START,
                sixth_term_effective_from_native=76360,
                mask_split_priority_applied=False,
            ),
            "full5 unmasked policy coverage required",
        )
    binding = plan["checkpoint_binding"]
    require(
        binding["original_binding"] == plan["original_binding"] == index["original_binding"]
        and binding["source_sha256"] == plan["source_sha256"],
        "input/source binding differs",
    )
    require(
        set(index) == {"original_binding", "prefix_chunks"}
        and len(index["prefix_chunks"]) == (STOP - FIRST) // CHUNK,
        "complete40 source prefix",
    )
    for i, chunk in enumerate(index["prefix_chunks"]):
        require(
            (chunk["index"], chunk["start"], chunk["stop"])
            == (i, FIRST + i * CHUNK, FIRST + (i + 1) * CHUNK),
            "source frame gap/order",
        )
    require(set(summary["candidate_arrays"]) == ALL21, "all21 producer signatures required")
    checkpoints = summary["checkpoints"]
    require(len(checkpoints) == (STOP - resume) // CHUNK, "complete checkpoint sequence")
    for through, row in zip(range(resume + CHUNK, STOP + 1, CHUNK), checkpoints, strict=True):
        common.exact_fields(
            row,
            dict(
                arrays=39,
                previous_frame=through - 1,
                processed_frames=through - FIRST,
                initialize_count=1,
                restore_count=restores,
            ),
            "checkpoint lifecycle differs",
        )
        require(
            Path(row["path"]).resolve() == item["root"] / f"checkpoints/through_{through:06d}",
            "checkpoint native path differs",
        )


def validate_original_import(imported, original_metadata, original_signatures, context):
    common.exact_fields(
        imported,
        dict(
            schema="nodewise_original_state_import_v1",
            status="ok",
            effective_from_native=START,
            previous_frame=START - 1,
            processed_frames=START - FIRST,
            logical_initialize_count=1,
            added_restore_count=1,
            new_begin_calls=0,
            all_arrays_byte_preserved=True,
            original_metadata_unchanged=True,
            instance_policy_effective_from_native=76360,
            review_cleared_by_migration=False,
            mask_split_priority_applied=False,
        ),
        "actual original66400 migration proof differs",
    )
    require(
        imported["original_metadata"] == original_metadata
        and imported["arrays"] == original_signatures
        and len(original_signatures) == 39,
        "different original39 metadata/signatures",
    )
    expected = copy.deepcopy(original_metadata)
    expected["context"], expected["restore_count"] = context, 4
    require(
        imported["new_context"] == context and imported["imported_metadata"] == expected,
        "undeclared original metadata migration",
    )
    require(
        context["sparse_distance_backend"]["effective_from_native"] == START
        and context["body_row_policy"]["effective_from_native"] == START
        and context["instance_rescore_policy"]["effective_from_native"] == 76360,
        "policy boundaries differ",
    )


def sparse_original_chain(sparse, comparison_seal, original_seal, signatures, metadata, evidence):
    """Three small saved documents connect Sparse's audited parent to original66400."""
    dense_root = WORK / "instance_rescore_ten_second_continuation_v1"
    dense_sha = "a71ddab8dfef8d97fecf68c1d5b7a5d7300fe21dfe49b48bc6f1df91c840196a"
    dense_path = dense_root / "tracking.provenance.json"
    require(
        comparison_seal["inputs"].get(str(dense_path)) == dense_sha
        and sparse["plan"]["checkpoint_binding"]["parent_tracking_sha256"] == dense_sha,
        "Sparse did not inherit the completed Dense parent",
    )
    evidence.add(dense_path, dense_sha)
    dense = proof.read_json(dense_path)
    common.exact_fields(dense, dict(status="ok", stage=dense_root.name), "Dense parent failed")
    fast_root = WORK / "fast_tail_two_minute_continuous_v1"
    manifest_path = fast_root / "checkpoints/through_076360/manifest.json"
    manifest_sha = "cef133c3e6192e373df095a3ec8b3b1cd546363a3aed8a6bf8ead670ff529cf6"
    require(
        dense["inputs"].get(str(manifest_path)) == manifest_sha,
        "Dense original Fast checkpoint edge differs",
    )
    evidence.add(manifest_path, manifest_sha)
    manifest = proof.read_json(manifest_path)
    binding = manifest["bindings"]
    require(
        binding["parent_seal_sha256"] == ORIGINAL_SHA
        and binding["original_binding"] == original_seal["params"]["bindings"],
        "Fast ancestor used a different original66400",
    )
    migration_path = fast_root / "state_import.json"
    evidence.add(migration_path, binding["state_import_sha256"])
    migration = proof.read_json(migration_path)
    require(
        migration["status"] == "ok"
        and migration["all_arrays_byte_preserved"] is True
        and migration["original_metadata_unchanged"] is True
        and migration["original_metadata"] == metadata
        and migration["arrays"] == signatures,
        "Sparse ancestor does not preserve the same original39 state",
    )
    return dict(
        dense_tracking_sha256=dense_sha,
        fast76360_manifest_sha256=manifest_sha,
        fast_original_state_import_sha256=binding["state_import_sha256"],
        original39_metadata_and_signature_equality=True,
        successful_Dense_and_Sparse_continuation_checks_inherited_not_reexecuted=True,
        interrupted_Fast_whole_run_success_claimed=False,
    )


def load_lineage(loaded, evidence):
    """Inherit completed Sparse audits; newly compare original39 small signatures only."""
    sparse, nodewise = (loaded[a] for a in ARMS)
    original_root = WORK / ORIGINAL_STAGE
    evidence.add(original_root / "continuation.provenance.json", ORIGINAL_SHA)
    original_seal = proof.read_json(original_root / "continuation.provenance.json")
    require(
        original_seal["status"] == "ok"
        and original_seal["stage"] == "three_body_long_continuation"
        and original_seal["params"]["through_frame"] == START,
        "successful original66400 required",
    )
    parent = {}
    for name, digest, role in (
        ("checkpoint/manifest.json", ORIGINAL_CHECKPOINT_SHA, "checkpoint_manifest"),
        (
            "checkpoint/metadata.json",
            ORIGINAL_METADATA_SHA,
            str(original_root / "checkpoint/metadata.json"),
        ),
    ):
        path = original_root / name
        row = original_seal["outputs"][role]
        require(
            Path(row["path"]).resolve() == path
            and row["sha256"] == digest
            and path.stat().st_size == row["size"],
            "original sealed header descriptor",
        )
        evidence.add(path, digest)
        parent[Path(name).stem] = proof.read_json(path)
    metadata, manifest = parent["metadata"], parent["manifest"]
    require(
        manifest["bindings"]
        == original_seal["params"]["bindings"]
        == nodewise["plan"]["original_binding"],
        "same original seed/source binding required",
    )
    common.exact_fields(
        metadata,
        dict(
            previous_frame=START - 1,
            processed_frames=START - FIRST,
            initialize_count=1,
            restore_count=3,
        ),
        "original checkpoint lifecycle",
    )
    signatures = {
        k: dict(dtype=v["dtype"]["str"], shape=v["shape"], sha256=v["bytes_sha256"])
        for k, v in manifest["arrays"].items()
    }
    imported = document(nodewise, "state_import.json", evidence)
    context = nodewise["plan"]["tracker_context"]
    require(context == nodewise["summary"]["tracker_context"], "new final context differs")
    validate_original_import(imported, metadata, signatures, context)
    for doc in (nodewise["plan"], nodewise["summary"], nodewise["plan"]["checkpoint_binding"]):
        require(
            doc["parent_tracking_sha256"] == ORIGINAL_SHA
            and doc["parent_checkpoint_sha256"] == ORIGINAL_CHECKPOINT_SHA,
            "new trajectory used different original checkpoint",
        )
    for path, digest in (
        (original_root / "continuation.provenance.json", ORIGINAL_SHA),
        (original_root / "checkpoint/manifest.json", ORIGINAL_CHECKPOINT_SHA),
        (original_root / "checkpoint/metadata.json", ORIGINAL_METADATA_SHA),
    ):
        require(
            nodewise["seal"]["inputs"].get(str(path)) == digest, "new sealed original input edge"
        )

    reference_root = WORK / SPARSE_COMPARISON
    evidence.add(reference_root / "comparison.provenance.json", SPARSE_COMPARISON_SHA)
    reference_seal = proof.read_json(reference_root / "comparison.provenance.json")
    common.exact_fields(
        reference_seal,
        dict(stage=SPARSE_COMPARISON, status="ok"),
        "completed Sparse comparison reference required",
    )
    reference = dict(root=reference_root, seal=reference_seal)
    plan = document(reference, "plan.json", evidence)
    lineage = document(reference, "producer_lineage.json", evidence)
    ref = plan["candidate_references"]["sparse_instance_rescore"]
    candidate = sparse["seal"]["outputs"]["candidate.npz"]
    require(
        plan["tracking_seals"]["sparse_instance_rescore"] == SPARSE_SHA
        and Path(ref["path"]).resolve() == Path(candidate["path"]).resolve()
        and ref["sha256"] == candidate["sha256"]
        and ref["terminal_horizon"] == STOP
        and ref["arrays"] == sparse["summary"]["candidate_arrays"]
        and reference_seal["inputs"].get(ref["path"]) == ref["sha256"],
        "different completed Sparse candidate",
    )
    previous_context = lineage["sparse_import"]["new_context"]
    expected = copy.deepcopy(context)
    expected.pop("body_row_policy")
    expected["implementation"] = previous_context["implementation"]
    expected["sparse_distance_backend"]["effective_from_native"] = SPARSE_START
    require(expected == previous_context, "base geometry/support/scoring configuration differs")
    origin_chain = sparse_original_chain(
        sparse, reference_seal, original_seal, signatures, metadata, evidence
    )
    return dict(
        original_tracking_sha256=ORIGINAL_SHA,
        original_checkpoint_sha256=ORIGINAL_CHECKPOINT_SHA,
        original_metadata_sha256=ORIGINAL_METADATA_SHA,
        new_import_all39_signatures_equal=True,
        original39_payload_redecoded=False,
        same_complete40_source_prefix=True,
        sparse_completed_comparison_sha256=SPARSE_COMPARISON_SHA,
        sparse_history_proof_is_attributed_to_sealed_producers=True,
        sparse_origin_chain=origin_chain,
        sparse_backend_start=SPARSE_START,
        nodewise_backend_and_body_start=START,
        unchanged_sixth_term_start=76360,
        temporal_path_and_support_are_not_accuracy_truth=True,
    )


def load_headers(nodewise_root, nodewise_sha, runner_sha, evidence):
    """Only small metadata/source reads; both successes precede candidate decode."""
    sparse = read_producer(WORK / SPARSE_STAGE, SPARSE_SHA, SPARSE_STAGE, evidence)
    nodewise = read_producer(nodewise_root, nodewise_sha, NODEWISE_STAGE, evidence)
    loaded = dict(zip(ARMS, (sparse, nodewise), strict=True))
    for arm, item in loaded.items():
        validate_completion(
            item, arm, runner_sha if arm == "nodewise" else SPARSE_RUNNER_SHA, evidence
        )
    require(sparse["input_index"] == nodewise["input_index"], "different common40 source inputs")
    lineage = load_lineage(loaded, evidence)
    return loaded, lineage


def validate_candidate(candidate):
    expected = {
        "frame_idx": ((STOP - FIRST,), np.int64),
        "pose_xyz": ((STOP - FIRST, 3, 3, 3), np.float64),
        "observed": ((STOP - FIRST, 3, 3), np.bool_),
        "sources": ((STOP - FIRST, 3, 3, 3), np.int32),
        "identity_review": ((STOP - FIRST, 3), np.bool_),
    }
    require(set(candidate) == set(FIELDS), "six selected fields required")
    for key, (shape, dtype) in expected.items():
        require(
            candidate[key].shape == shape and candidate[key].dtype == dtype,
            "candidate axes/dtype: " + key,
        )
    require(
        candidate["proposal_kind"].shape == (STOP - FIRST,)
        and candidate["proposal_kind"].dtype.kind == "U",
        "proposal kind axis",
    )
    require(
        np.array_equal(candidate["frame_idx"], np.arange(FIRST, STOP)), "retained native coverage"
    )
    require(np.isfinite(candidate["pose_xyz"]).all(), "nonfinite latent pose: evaluation refused")
    sources = candidate["sources"]
    require(
        np.all(sources >= -1) and not np.any(candidate["observed"] & ((sources >= 0).sum(2) < 2)),
        "saved direct flag lacks two source views",
    )


def load_candidate(item, evidence):
    path = common.bound_member(item["root"], item["seal"], "candidate.npz", evidence)
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == 21 and set(archive.files) == ALL21, "candidate21 membership")
        candidate = {key: archive[key] for key in FIELDS}
    require(
        common.signature(candidate) == {k: item["summary"]["candidate_arrays"][k] for k in FIELDS},
        "selected candidate signature differs",
    )
    validate_candidate(candidate)
    for array in candidate.values():
        array.flags.writeable = False
    return candidate


def summarize(candidate, start=None, stop=None):
    start, stop = START if start is None else start, STOP if stop is None else stop
    keep = (candidate["frame_idx"] >= start) & (candidate["frame_idx"] < stop)
    frames = candidate["frame_idx"][keep]
    require(
        np.array_equal(frames, np.arange(start, stop)) and len(frames) > 1,
        "complete metric interval required",
    )
    pose, observed, sources = (candidate[k][keep] for k in ("pose_xyz", "observed", "sources"))
    steps = np.linalg.norm(np.diff(pose, axis=0), axis=-1)
    both = observed[1:] & observed[:-1]
    hi = sources.max(-1)
    lo = np.min(np.where(sources >= 0, sources, np.iinfo(np.int32).max), axis=-1)
    mixed = (hi >= 0) & (lo != hi)
    kinds = candidate["proposal_kind"][keep]
    require(not mixed[kinds == "body_row"].any(), "selected body-row proposal changed source rows")
    direct_max = np.max(np.where(both, steps, -np.inf), axis=0)
    return dict(
        interval=[start, stop],
        frames=len(frames),
        adjacent_pairs_per_slot_node=len(frames) - 1,
        directly_observed_by_slot_node=observed.sum(0).tolist(),
        missing_by_slot_node=(~observed).sum(0).tolist(),
        longest_missing_run_by_slot_node=old.longest_true(~observed).tolist(),
        all_three_chests_direct_frames=int(observed[:, :, 1].all(-1).sum()),
        directly_supported_adjacent_pairs_by_slot_node=both.sum(0).tolist(),
        direct_max_step_by_slot_node_mm=[
            [None if not np.isfinite(v) else float(v) for v in row] for row in direct_max
        ],
        all_estimated_max_step_by_slot_node_mm=steps.max(0).tolist(),
        direct_adjacent_steps_over_10mm=[
            dict(
                previous_frame=int(frames[t]),
                frame_idx=int(frames[t + 1]),
                raw_slot=int(b),
                node=int(n),
                distance_mm=float(steps[t, b, n]),
            )
            for t, b, n in np.argwhere(both & (steps > 10.0))
        ],
        mixed_source_rows_by_slot_camera=mixed.sum(0).tolist(),
        body_frames_with_any_mixed_source_rows=int(mixed.any(-1).sum()),
        identity_review_by_slot=candidate["identity_review"][keep].sum(0).tolist(),
        final_backtrace_kind_counts={str(k): int((kinds == k).sum()) for k in np.unique(kinds)},
        nonfinite_pose_count=0,
        support_recalculated=False,
    )


def geometric_comparison(a, b):
    """P[A raw slot] = B raw slot; use one whole-body P for all node/support axes."""
    require(np.array_equal(a["frame_idx"], b["frame_idx"]), "same native axes required")
    pa, pb = a["pose_xyz"], b["pose_xyz"]
    require(np.isfinite(pa).all() and np.isfinite(pb).all(), "finite geometry required")
    squared = np.stack([np.sum((pa - pb[:, p]) ** 2, axis=-1) for p in PERMUTATIONS], axis=1)
    require(np.isfinite(squared).all(), "nonfinite squared geometry: evaluation refused")
    support = np.stack([a["observed"] & b["observed"][:, p] for p in PERMUTATIONS], axis=1)
    chest = np.sqrt(squared[..., 1].mean(-1))
    body = np.sqrt(squared.mean((2, 3)))
    arrays = dict(
        frame_idx=a["frame_idx"].copy(),
        permutations=PERMUTATIONS.copy(),
        chest_rms_mm=chest,
        all_node_rms_mm=body,
        both_direct_node_counts_by_permutation=support.sum((2, 3)),
    )
    report = dict(
        permutation_maps_A_raw_slot_to_B_raw_slot=True,
        framewise_geometry_is_not_temporal_ID_alignment=True,
    )
    for name, costs in (("chest", chest), ("all_node", body)):
        best = costs.min(-1)
        ties = np.isclose(costs, best[:, None], rtol=0, atol=1e-12)
        unique = ties.sum(-1) == 1
        indices = np.argmin(costs, axis=-1)
        arrays[name + "_optimal_permutation_mask"] = ties
        report[name] = dict(
            raw_identity_rms_median_mm=float(np.median(costs[:, 0])),
            raw_identity_rms_maximum_mm=float(costs[:, 0].max()),
            minimum_rms_median_mm=float(np.median(best)),
            minimum_rms_maximum_mm=float(best.max()),
            unique_nonidentity_frames=int((unique & ~ties[:, 0]).sum()),
            tied_frames=int((~unique).sum()),
            unique_winner_counts=[int(np.sum(unique & (indices == i))) for i in range(6)],
            both_direct_node_pairs_on_unique_frames=int(
                support[np.arange(len(costs)), indices][unique].sum()
            ),
        )
    return report, arrays


def compare_arrays(candidates):
    require(tuple(candidates) == ARMS, "two fixed method axes")
    methods, samples, science = {}, {}, {}
    for arm, candidate in candidates.items():
        validate_candidate(candidate)
        methods[arm] = summarize(candidate)
        indices = np.searchsorted(candidate["frame_idx"], FRAMES)
        require(np.array_equal(candidate["frame_idx"][indices], FRAMES), "fixed development probes")
        samples.update({arm + "_" + k: candidate[k][indices].copy() for k in FIELDS})
        science[arm] = {k: v[START - FIRST :] for k, v in candidate.items()}
    pairwise, arrays = geometric_comparison(*(science[a] for a in ARMS))
    pairwise["selected_pre_science_backtrace_pose_difference_frames"] = int(
        np.any(
            candidates["sparse"]["pose_xyz"][: START - FIRST]
            != candidates["nodewise"]["pose_xyz"][: START - FIRST],
            axis=(1, 2, 3),
        ).sum()
    )
    return methods, pairwise, samples, arrays


def run(nodewise_root, nodewise_sha, runner_sha, output):
    require(EXECUTION_ENABLED, "source qualification only; execution is disabled")
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    evidence, out, started = proof.Evidence(), proof.Output(output), proof.utc()
    try:
        for name, digest in PINS.items():
            evidence.add(WORK / name, digest)
        evidence.add(Path(__file__))
        loaded, lineage = load_headers(nodewise_root, nodewise_sha, runner_sha, evidence)
        candidates = {arm: load_candidate(item, evidence) for arm, item in loaded.items()}
        methods, pairwise, samples, permutation_arrays = compare_arrays(candidates)
        out.write("review_samples.npz", samples)
        out.write("review_samples_signatures.json", common.signature(samples))
        out.write("geometric_permutations.npz", permutation_arrays)
        out.write("geometric_permutations_signatures.json", common.signature(permutation_arrays))
        refs = {
            arm: dict(
                root=str(item["root"]),
                tracking_seal_sha256=item["sha256"],
                candidate=item["seal"]["outputs"]["candidate.npz"],
                all21_declared_signatures=item["summary"]["candidate_arrays"],
                decoded_field_signatures=common.signature(candidates[arm]),
            )
            for arm, item in loaded.items()
        }
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                source_sha256={k: v for k, v in evidence.paths.items() if k.endswith(".py")},
                scientific_interval=[START, STOP],
                retained_interval=[FIRST, STOP],
                terminal_horizon=STOP,
                candidate_references=refs,
                lineage=lineage,
                decoded_fields=list(FIELDS),
                undecoded15_numerically_verified=False,
                node_order=["mouth", "chest", "tail"],
                view_order=["top_cam", "side1_cam", "side2_cam"],
                permutations=PERMUTATIONS.tolist(),
                tie_atol_mm=1e-12,
                development_frames=list(FRAMES),
                both_adjacent_endpoints_inside_scientific_interval=True,
            ),
        )
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                arms=list(ARMS),
                scientific_interval=[START, STOP],
                native_frames=STOP - START,
                terminal_horizon=STOP,
                methods=methods,
                pairwise=pairwise,
                candidate_references=refs,
                development_frames=list(FRAMES),
                development_is_holdout=False,
                physical_ID_truth=False,
                temporal_ID_mapping_computed=False,
                automatic_adoption=False,
                candidate_modified=False,
                support_recomputed=False,
                review_cleared=False,
                selected_prefix_is_not_full_retained_history=True,
                classical_or_female_evaluation_rerun=False,
                checkpoint_video_mask_SL_GPU_reads=False,
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
    parser.add_argument("--nodewise-root", type=Path, default=WORK / NODEWISE_STAGE)
    parser.add_argument("--nodewise-sha", required=True)
    parser.add_argument("--nodewise-source-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.nodewise_root, args.nodewise_sha, args.nodewise_source_sha, args.output)

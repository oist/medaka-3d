"""Completed76360 Nodewise milestone versus the successful same-horizon Tail minute.

Six candidate fields are decoded once per arm. The new full-five-minute process
may still be running: only its completed, explicitly SHA-bound milestone is used.
Stored support/review and raw slots are not physical identity ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import compare_nodewise_body_row_five_minutes as metrics
import numpy as np

proof, common, require, WORK = metrics.proof, metrics.common, metrics.require, metrics.WORK
FIRST, START, STOP, CHUNK = 49800, 66400, 76360, 1660
STAGE = "nodewise_body_row_first_minute_comparison_v1"
ARMS, FIELDS = ("tail", "nodewise"), metrics.FIELDS
SCIENCE_FIELDS = ("frame_idx", "pose_xyz", "observed")
TAIL = WORK / "tail_one_minute_66400-76360_v1"
TAIL_SHA = "50ccc8d2a4bbcc7e09aa2b512b70374fb9b37a4d4ece0d0939322c66bfa59416"
TAIL_CANDIDATE_SHA = "4a86511a1bb01a28de7f4497e848ca0e9c71badefe61fe23722039a2e993f7f7"
TAIL_COMPARISON = WORK / "tail_one_minute_comparison_v1"
TAIL_COMPARISON_SHA = "8a3a77326f34e77801f834b529fef79626116fba63a1f99155ec33f2cea3e557"
NODEWISE = WORK / metrics.NODEWISE_STAGE
MILESTONE = NODEWISE / "milestones/through_076360"
MILESTONE_SHA = "bed9dd1f0d7192261f4e985a3021097f444a941e180bd1982bad4dbcb01525de"
METRICS_SHA = "512c46a8aee7b5ce7182fc86c282033ad34408509c2a32315447620c930f921e"
INPUT_ROOT = WORK / "inputs/long_source_masks_49800-348600_v1"


def read(path, evidence, digest=None):
    evidence.add(path, digest)
    return proof.read_json(path)


def old_digest(seal, path):
    row = seal["inputs"].get(str(Path(path).resolve()))
    return row.get("sha256") if isinstance(row, dict) else row


def old_member(root, seal, name, evidence, *, decode=True):
    path = (root / name).resolve()
    refs = [v for v in seal["outputs"].values() if Path(v["path"]).resolve() == path]
    require(len(refs) == 1, "one exact legacy output descriptor required")
    row = refs[0]
    require(
        path.is_relative_to(root) and path.stat().st_size == row["size"], "legacy output path/size"
    )
    if decode:
        return read(path, evidence, row["sha256"])
    return dict(path=str(path), sha256=row["sha256"], size_bytes=row["size"])


def source_code(evidence):
    pins = {
        **metrics.PINS,
        "compare_nodewise_body_row_five_minutes.py": METRICS_SHA,
        "run_nodewise_body_row_five_minutes.py": metrics.NODEWISE_RUNNER_SHA,
        **metrics.NODEWISE_COMPONENTS,
    }
    code = {str((WORK / p).resolve()): evidence.add(WORK / p, h) for p, h in pins.items()}
    code[str(Path(__file__).resolve())] = evidence.add(Path(__file__))
    return code


def verify_prefix(index, tail_seal, original_binding):
    require(
        set(index) == {"original_binding", "prefix_chunks"}
        and index["original_binding"] == original_binding
        and len(index["prefix_chunks"]) == 40,
        "registered full source index",
    )
    prefix = index["prefix_chunks"][: (STOP - FIRST) // CHUNK]
    for i, chunk in enumerate(prefix):
        require(
            (chunk["index"], chunk["start"], chunk["stop"])
            == (i, FIRST + i * CHUNK, FIRST + (i + 1) * CHUNK),
            "source prefix gap/order",
        )
        root = (INPUT_ROOT / chunk["attempt_path"]).resolve()
        require(root.is_relative_to(INPUT_ROOT.resolve()), "source chunk escaped input root")
        for name, digest in (
            ("manifest.json", chunk["manifest_sha256"]),
            ("staging.provenance.json", chunk["provenance_sha256"]),
        ):
            require(
                old_digest(tail_seal, root / name) == digest, "Tail/source manifest lineage differs"
            )
        for role, ref in chunk["output_hashes"].items():
            if role.endswith("_raw"):
                view = role[:-4]
                link = chunk["raw_hardlinks"][view]
                video_sha = original_binding["inputs"]["detector_binding"]["input_sha256"][
                    "video:" + view
                ]
                require(
                    ref["path"] == link["path"] and ref["sha256"] == link["sha256"] == video_sha,
                    "registered raw source differs",
                )
            else:
                require(
                    old_digest(tail_seal, ref["path"]) == ref["sha256"],
                    "Tail/source output lineage differs",
                )
    return prefix


def load_headers(milestone_sha, evidence):
    require(proof.valid_sha(milestone_sha), "explicit completed milestone summary SHA required")
    tail = read(TAIL / "tracking.provenance.json", evidence, TAIL_SHA)
    common.exact_fields(
        tail, dict(stage="tail_one_minute_continuous", status="ok"), "successful Tail minute"
    )
    ts = old_member(TAIL, tail, "summary.json", evidence)
    common.exact_fields(
        ts,
        dict(
            schema="tail_one_minute_v1",
            status="ok",
            one_minute_completed=True,
            native_interval=[START, STOP],
            native_frames=STOP - START,
            logical_initialization_count=1,
            restore_count=4,
            external_pose_resets=0,
            periodic_checkpoints_are_not_reinitializations=True,
        ),
        "same completed Tail lifecycle",
    )
    comparison = read(TAIL_COMPARISON / "comparison.provenance.json", evidence, TAIL_COMPARISON_SHA)
    common.exact_fields(
        comparison,
        dict(stage="tail_one_minute_comparison_v1", status="ok"),
        "qualified Tail comparison",
    )
    previous = old_member(TAIL_COMPARISON, comparison, "summary.json", evidence)
    common.exact_fields(
        previous,
        dict(
            status="ok",
            native_interval=[START, STOP],
            native_frames=STOP - START,
            horizon_stop=STOP,
            same_parent_and_horizon=True,
            all21_arrays_exact_backtrace=True,
            original66400_parent_and_migration_verified=True,
            tail_all_six_checkpoint_prefixes_exact=True,
            restores_during_one_minute=0,
            external_pose_resets=0,
        ),
        "same-horizon Tail proof required",
    )
    tail_ref = old_member(TAIL, tail, "candidate.npz", evidence, decode=False)
    require(
        tail_ref["sha256"] == previous["candidate_sha256"]["tail"] == TAIL_CANDIDATE_SHA
        and Path(previous["candidate_paths"]["tail"]).resolve() == Path(tail_ref["path"])
        and old_digest(comparison, TAIL / "candidate.npz") == TAIL_CANDIDATE_SHA,
        "qualified Tail candidate differs",
    )
    parent_path = WORK / metrics.ORIGINAL_STAGE / "continuation.provenance.json"
    require(old_digest(tail, parent_path) == metrics.ORIGINAL_SHA, "same original66400 parent")
    old_import = old_member(TAIL, tail, "state_import.json", evidence)
    old_manifest = old_member(TAIL, tail, "checkpoints/through_076360/manifest.json", evidence)

    milestone = read(MILESTONE / "summary.json", evidence, milestone_sha)
    common.exact_fields(
        milestone,
        dict(
            schema=metrics.NODEWISE_STAGE + "_milestone",
            status="ok",
            horizon_stop=STOP,
            scientific_interval=[START, STOP],
            native_frames=STOP - START,
            initialize_count=1,
            restore_count=4,
            finish_calls_so_far=0,
            new_begin_calls=0,
            external_pose_resets=0,
            all39_roundtrip=True,
            prior_history_prefix_unchanged=True,
            whole_process_completion_pending=True,
            physical_ID_truth=False,
        ),
        "completed first-minute milestone only",
    )
    plan = read(NODEWISE / "plan.json", evidence)
    index = read(NODEWISE / "input_index.json", evidence)
    imported = read(NODEWISE / "state_import.json", evidence)
    progress = read(NODEWISE / "checkpoint_records/progress_076360.json", evidence)
    common.exact_fields(
        plan,
        dict(
            schema=metrics.NODEWISE_STAGE,
            process_interval=[START, metrics.STOP],
            scientific_interval=[START, metrics.STOP],
            retained_interval=[FIRST, metrics.STOP],
            parent_tracking_sha256=metrics.ORIGINAL_SHA,
            parent_checkpoint_sha256=metrics.ORIGINAL_CHECKPOINT_SHA,
            initialize_count=1,
            restore_count=4,
            new_begin_calls=0,
            external_pose_resets=0,
            body_row_and_sparse_backend_effective_from_native=START,
            sixth_term_effective_from_native=STOP,
            sixth_term_neutral_before_boundary=True,
            mask_split_priority_applied=False,
        ),
        "reviewed Nodewise full5 plan",
    )
    for name, digest in {
        "run_nodewise_body_row_five_minutes.py": metrics.NODEWISE_RUNNER_SHA,
        **metrics.NODEWISE_COMPONENTS,
    }.items():
        require(
            plan["source_sha256"].get(str((WORK / name).resolve())) == digest,
            "Nodewise producer source differs",
        )
    for imp in (old_import, imported):
        require(
            imp["status"] == "ok"
            and imp["all_arrays_byte_preserved"]
            and imp["original_metadata_unchanged"]
            and imp["new_begin_calls"] == 0
            and imp["added_restore_count"] == 1
            and not imp["review_cleared_by_migration"],
            "history-preserving import required",
        )
    common.exact_fields(
        imported,
        dict(
            schema="nodewise_original_state_import_v1",
            previous_frame=START - 1,
            processed_frames=START - FIRST,
            effective_from_native=START,
            old_history_rewritten=False,
        ),
        "exact new import boundary",
    )
    require(
        imported["original_metadata"] == old_import["original_metadata"]
        and imported["arrays"] == old_import["arrays"]
        and len(imported["arrays"]) == 39
        and imported["new_context"] == plan["tracker_context"],
        "same original39 metadata/signatures",
    )
    original_binding = old_manifest["bindings"]["original_binding"]
    require(
        plan["original_binding"]
        == plan["checkpoint_binding"]["original_binding"]
        == original_binding,
        "same original observation binding",
    )
    prefix = verify_prefix(index, tail, original_binding)
    require(
        progress["through_frame"] == STOP
        and progress["diagnostic_frames"] == STOP - START
        and len(progress["checkpoints"]) == (STOP - START) // CHUNK
        and progress["checkpoints"][-1] == milestone["checkpoint"],
        "checkpoint emit not committed",
    )
    for stop, cp in zip(
        range(START + CHUNK, STOP + 1, CHUNK), progress["checkpoints"], strict=True
    ):
        common.exact_fields(
            cp,
            dict(
                arrays=39,
                previous_frame=stop - 1,
                processed_frames=stop - FIRST,
                initialize_count=1,
                restore_count=4,
            ),
            "one continuous checkpoint sequence",
        )
        require(
            Path(cp["path"]).resolve() == (NODEWISE / f"checkpoints/through_{stop:06d}").resolve(),
            "checkpoint path",
        )
    cp = milestone["checkpoint"]
    cp_root = Path(cp["path"]).resolve()
    manifest = read(cp_root / "manifest.json", evidence, cp["manifest_sha256"])
    require(
        manifest["bindings"] == plan["checkpoint_binding"]
        and len(manifest["arrays"]) == 39
        and manifest["payload"]["bytes"] == cp["payload_bytes"],
        "committed checkpoint header differs",
    )
    meta_ref = manifest["metadata"]
    require(meta_ref["file"] == "metadata.json", "checkpoint metadata filename")
    metadata = read(cp_root / "metadata.json", evidence, meta_ref["sha256"])
    common.exact_fields(
        metadata,
        dict(
            previous_frame=STOP - 1,
            processed_frames=STOP - FIRST,
            initialize_count=1,
            restore_count=4,
        ),
        "checkpoint metadata lifecycle",
    )
    require(metadata["context"] == plan["tracker_context"], "checkpoint context differs")
    node_ref = milestone["candidate"]
    require(
        Path(node_ref["path"]).resolve() == (MILESTONE / "candidate.npz").resolve()
        and set(milestone["candidate_arrays"]) == metrics.ALL21,
        "milestone candidate21 declaration",
    )
    binding = dict(
        native_interval=[START, STOP],
        retained_interval=[FIRST, STOP],
        terminal_horizon=STOP,
        original_tracking_sha256=metrics.ORIGINAL_SHA,
        original_checkpoint_sha256=metrics.ORIGINAL_CHECKPOINT_SHA,
        same_original_metadata_and39_signatures=True,
        common_source_prefix_chunks=len(prefix),
        baseline_tracking_sha256=TAIL_SHA,
        baseline_comparison_sha256=TAIL_COMPARISON_SHA,
        baseline_candidate=tail_ref,
        nodewise_source_sha256=metrics.NODEWISE_RUNNER_SHA,
        nodewise_milestone_summary_sha256=milestone_sha,
        nodewise_milestone_candidate=node_ref,
        nodewise_all21_declared_signatures=milestone["candidate_arrays"],
        nodewise_checkpoint_manifest_sha256=cp["manifest_sha256"],
        whole_five_minute_completion_claimed=False,
        checkpoint_payloads_redecoded=False,
        old_all21_backtrace_validation_inherited=True,
        raw_video_bytes_reverified=False,
        raw_registration_inherited_from_same16_manifests=True,
    )
    return dict(
        tail=dict(candidate=tail_ref, signatures=None),
        nodewise=dict(candidate=node_ref, signatures=milestone["candidate_arrays"]),
    ), binding


def validate_candidate(candidate):
    n = STOP - FIRST
    require(set(candidate) == set(FIELDS), "six selected fields")
    for key, shape, dtype in (
        ("frame_idx", (n,), np.int64),
        ("pose_xyz", (n, 3, 3, 3), np.float64),
        ("sources", (n, 3, 3, 3), np.int32),
        ("observed", (n, 3, 3), np.bool_),
        ("identity_review", (n, 3), np.bool_),
    ):
        require(
            candidate[key].shape == shape and candidate[key].dtype == dtype,
            "candidate shape/dtype: " + key,
        )
    require(
        candidate["proposal_kind"].shape == (n,) and candidate["proposal_kind"].dtype.kind == "U",
        "proposal kind axis/dtype",
    )
    require(
        np.array_equal(candidate["frame_idx"], np.arange(FIRST, STOP)),
        "exact retained native range",
    )
    require(np.isfinite(candidate["pose_xyz"]).all(), "finite latent poses required")
    sources = candidate["sources"]
    require(
        (sources >= -1).all() and not np.any(candidate["observed"] & ((sources >= 0).sum(2) < 2)),
        "saved direct support lacks two assigned views",
    )


def load_candidate(ref, evidence):
    desc = ref["candidate"]
    path = Path(desc["path"]).resolve()
    require(
        path.is_relative_to(WORK) and path.stat().st_size == desc["size_bytes"],
        "candidate path/size",
    )
    evidence.add(path, desc["sha256"])
    with np.load(path, allow_pickle=False) as z:
        require(len(z.files) == 21 and set(z.files) == metrics.ALL21, "candidate21 membership")
        candidate = {k: z[k] for k in FIELDS}
    validate_candidate(candidate)
    if ref["signatures"] is not None:
        require(
            common.signature(candidate) == {k: ref["signatures"][k] for k in FIELDS},
            "selected signatures differ",
        )
    for value in candidate.values():
        value.flags.writeable = False
    return candidate


def compare_arrays(candidates):
    require(tuple(candidates) == ARMS, "fixed two-arm axis")
    methods, science = {}, {}
    for arm, candidate in candidates.items():
        validate_candidate(candidate)
        methods[arm] = metrics.summarize(candidate, start=START, stop=STOP)
        science[arm] = {k: v[START - FIRST :] for k, v in candidate.items()}
    geometric, arrays = metrics.geometric_comparison(science["tail"], science["nodewise"])
    pre = {}
    for key in FIELDS:
        a, b = (candidates[arm][key][: START - FIRST] for arm in ARMS)
        changed = (a != b).reshape(len(a), -1).any(1)
        pre[key] = int(changed.sum())
    geometric["selected_pre_science_backtrace_difference_frames"] = pre
    geometric["different_selected_prefix_is_not_historical_payload_modification"] = True
    return (
        methods,
        geometric,
        arrays,
        {arm: {k: row[k].copy() for k in SCIENCE_FIELDS} for arm, row in science.items()},
    )


def run(milestone_sha, output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and output != WORK and not output.exists(),
        "fresh workspace output",
    )
    evidence, out, started = proof.Evidence(), proof.Output(output), proof.utc()
    try:
        code = source_code(evidence)
        refs, binding = load_headers(milestone_sha, evidence)
        candidates = {arm: load_candidate(refs[arm], evidence) for arm in ARMS}
        before = {arm: common.signature(value) for arm, value in candidates.items()}
        methods, geometry, arrays, science = compare_arrays(candidates)
        signatures = {arm: common.signature(values) for arm, values in science.items()}
        for arm, values in science.items():
            out.write(f"science/{arm}.npz", values)
        out.write("science_signatures.json", signatures)
        out.write("geometric_permutations.npz", arrays)
        out.write("geometric_permutation_signatures.json", common.signature(arrays))
        scope = dict(
            schema=STAGE,
            **binding,
            source_sha256=code,
            selected_field_signatures=before,
            science_signatures=signatures,
            decoded_fields=list(FIELDS),
            undecoded15_per_arm_numerically_verified=False,
            all21_membership_verified=True,
            source_candidate_npz_decodes_per_arm=1,
            support_recomputed=False,
            raw_slot_correspondence_is_not_physical_ID=True,
            review_cleared=False,
            checkpoint39_image_mask_SL_GPU_reads=0,
        )
        out.write("plan.json", scope)
        out.write(
            "summary.json",
            dict(**scope, status="ok", methods=methods, geometric_comparison=geometry),
        )
        require(
            before == {arm: common.signature(v) for arm, v in candidates.items()},
            "candidate fields modified",
        )
        evidence.final_guard()
        out.final_guard()
        status = dict(status="ok")
    except BaseException as error:
        status = dict(status="failed", error=repr(error))
        raise
    finally:
        (output / "comparison.provenance.json").write_bytes(
            proof.json_bytes(
                dict(
                    stage=STAGE,
                    started_at=started,
                    finished_at=proof.utc(),
                    inputs=evidence.paths,
                    outputs=out.records,
                    **status,
                )
            )
        )
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milestone-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.milestone_sha, args.output)

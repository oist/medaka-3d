"""Same76360 horizon, six-field Nodewise versus observation-reserve comparison.

No fitting, threshold rescoring or temporal ID relabeling. The earlier completed
minute comparison supplies original initialization/registration qualification;
this adapter checks the shared74700 checkpoint and chunk15 explicitly.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import compare_nodewise_body_row_first_minute as minute
import numpy as np

metrics, proof, common, require = minute.metrics, minute.proof, minute.common, minute.require
WORK = minute.WORK
FIRST, START, STOP, CHUNK = 49800, 74700, 76360, 1660
STAGE = "observation_reserve_ten_second_comparison_v1"
TRACKING_STAGE = "observation_reserve_ten_second_continuation_v1"
TRACKING = WORK / TRACKING_STAGE
PARENT, MILESTONE = minute.NODEWISE, minute.MILESTONE
PARENT_CHECKPOINT = PARENT / "checkpoints/through_074700"
QUALIFIED = WORK / minute.STAGE
QUALIFIED_SHA = "7d3a205eecf1a83b19d39418a2a7e7e286df323eb4b82854655f479251217604"
PARENT_CHECKPOINT_SHA = "7c9672b6fde9f96ee5d23efe7dbbd6e7f88d48299d81e86e00a87fcca5c5a076"
PARENT_METADATA_SHA = "46ba3c749f0ec5232b68d0616e928eed0cd4d1910a9facaa81b4c39207e93502"
PARENT_PLAN_SHA = "e05e174a291d6d79bf2cf3f74b848e7b125b3304f8ad1f0d168165dac3a8ba31"
PARENT_INDEX_SHA = "f5840d96f166f04516c72326aee157e78fac01a290296653aabdec01dd8ae568"
MINUTE_SOURCE_SHA = "705ede6ebeee394b7dfa7fc43697c62d0dd9fa23f4208173737ac4a75f170d98"
RUNNER_SHA = "f2ad6b4039da5d14244cb2f0052003d3003903146a9ae573f362fc14aee437bc"
VARIANT_SHA = "7b03698dacda5d18be87ff99c0bccd47474a94db08ae89aff0231f4dcb50c74a"
SELECTOR_SHA = "b4b69b67e488db00fdff49b278a0c39f3966cc24e55749e2b1ca8251d1196853"
ARMS, FIELDS = ("nodewise", "observation_reserve"), minute.FIELDS
PROBES = (75052, 75053, 75054, 75055, 75056)


def source_code(evidence):
    pins = {
        **metrics.PINS,
        "compare_nodewise_body_row_first_minute.py": MINUTE_SOURCE_SHA,
        "compare_nodewise_body_row_five_minutes.py": minute.METRICS_SHA,
        "run_observation_reserve_continuation.py": RUNNER_SHA,
        "observation_reserve_tracker.py": VARIANT_SHA,
        "observation_reserve_selection.py": SELECTOR_SHA,
    }
    require(proof.valid_sha(RUNNER_SHA), "final reviewed producer source pin required")
    code = {str((WORK / p).resolve()): evidence.add(WORK / p, h) for p, h in pins.items()}
    code[str(Path(__file__).resolve())] = evidence.add(Path(__file__))
    return code


def member(item, name, evidence):
    return proof.read_json(common.bound_member(item["root"], item["seal"], name, evidence))


def validate_import(imported, parent_metadata, parent_manifest, new_context):
    expected_context = copy.deepcopy(new_context)
    policy = expected_context.pop("observation_reserve_policy")
    expected_context["implementation"] = parent_metadata["context"]["implementation"]
    require(expected_context == parent_metadata["context"], "only reserve context may be added")
    require(
        policy["effective_from_native"] == START
        and policy["selection_source_sha256"] == SELECTOR_SHA
        and policy["protected_prefix_count"] == 5,
        "fixed reserve policy differs",
    )
    common.exact_fields(
        imported,
        dict(
            schema="observation_reserve_nodewise_state_import_v1",
            status="ok",
            previous_frame=START - 1,
            processed_frames=START - FIRST,
            effective_from_native=START,
            all_arrays_byte_preserved=True,
            original_metadata_unchanged=True,
            logical_initialize_count=1,
            new_begin_calls=0,
            added_restore_count=1,
            old_history_rewritten=False,
            review_cleared_by_migration=False,
        ),
        "context-only74700 import required",
    )
    signatures = {
        k: dict(dtype=v["dtype"]["str"], shape=v["shape"], sha256=v["bytes_sha256"])
        for k, v in parent_manifest["arrays"].items()
    }
    expected = copy.deepcopy(parent_metadata)
    expected["context"] = new_context
    expected["restore_count"] += 1
    require(
        len(signatures) == 39
        and imported["arrays"] == signatures
        and imported["original_metadata"] == parent_metadata
        and imported["old_context"] == parent_metadata["context"]
        and imported["new_context"] == new_context
        and imported["imported_metadata"] == expected,
        "original39 signatures, metadata or single restore differ",
    )


def summarize_reserve(index, reported):
    require(
        index["process_interval"] == [START, STOP]
        and index["diagnostic_families"] == ["instance", "body_row", "observation_reserve"]
        and index["batch_size"] == 100,
        "three-family diagnostic index scope",
    )
    rows = index["rows"]
    require([r["frame_idx"] for r in rows] == list(range(START, STOP)), "reserve index coverage")
    reasons, changed, eligible, reservations = {}, [], [], 0
    for row in rows:
        value, count = row["observation_reserve"], row["candidate_count"]
        baseline, selected = value["baseline_indices"], value["selected_indices"]
        require(
            1 <= len(baseline) == len(selected) <= 6
            and len(set(baseline)) == len(baseline)
            and len(set(selected)) == len(selected)
            and all(type(i) is int and 0 <= i < count for i in baseline + selected)
            and selected[:-1] == baseline[:-1]
            and selected[0] == baseline[0] == value["baseline_best_index"]
            and type(value["changed"]) is bool
            and value["changed"] == (selected != baseline)
            and value["reservation_count"] == int(value["changed"])
            and value["support_evaluation_count"] == count
            and type(value["eligible_candidate_count"]) is int
            and 0 <= value["eligible_candidate_count"] <= count,
            "compact reserve selection invariants",
        )
        if value["changed"]:
            replacement = value["replacement"]
            require(
                len(selected) == 6
                and replacement["removed"] == baseline[-1]
                and replacement["admitted"] == selected[-1]
                and np.isfinite(replacement["extra_path_cost"])
                and replacement["extra_path_cost"] >= 0,
                "last-slot replacement differs",
            )
            changed.append(row["frame_idx"])
        else:
            require(value["replacement"] is None, "no-change replacement must be absent")
        reasons[value["reason"]] = reasons.get(value["reason"], 0) + 1
        reservations += value["reservation_count"]
        eligible.append(value["eligible_candidate_count"])
    require(
        reported["changed_frames"] == len(changed) == reservations
        and reported["reasons"] == reasons
        and reported["current_beam_selection_only"] is True
        and reported["final_path_reserved_ancestor_count_not_computed"] is True
        and reported["original_numeric_costs_preserved"] is True,
        "producer reserve counters differ from saved index",
    )
    return dict(
        changed_frames=len(changed),
        changed_native_frames=changed,
        eligible_candidate_counts=eligible,
        reasons=reasons,
        current_beam_metadata_recomputed=True,
        final_path_reserved_ancestor_count=None,
        final_path_attribution_requires_candidate_to_saved_parent_backtrace_binding=True,
        final_candidate_kind_alone_does_not_identify_reserved_branch=True,
        batch_payloads_decoded=0,
        development_probes=[r for r in rows if r["frame_idx"] in PROBES],
    )


def load_headers(tracking_sha, evidence):
    require(proof.valid_sha(tracking_sha), "explicit successful tracking seal SHA required")
    seal = minute.read(TRACKING / "tracking.provenance.json", evidence, tracking_sha)
    common.exact_fields(seal, dict(stage=TRACKING_STAGE, status="ok"), "successful new producer")
    item = dict(root=TRACKING, seal=seal)
    plan, summary, index, imported, diagnostics = (
        member(item, name + ".json", evidence)
        for name in ("plan", "summary", "input_index", "state_import", "diagnostic_index")
    )
    for doc in (plan, summary):
        common.exact_fields(
            doc,
            dict(
                schema=TRACKING_STAGE,
                process_interval=[START, STOP],
                scientific_interval=[START, STOP],
                retained_interval=[FIRST, STOP],
                original_scientific_history_start=66400,
                initialize_count=1,
                restore_count=5,
                new_begin_calls=0,
                external_pose_resets=0,
                parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA,
                parent_checkpoint_producer_source_sha256=metrics.NODEWISE_RUNNER_SHA,
                parent_five_minute_completion_claimed=False,
            ),
            "same endpoint/process/parent/lifecycle",
        )
        for name, digest in (
            ("run_observation_reserve_continuation.py", RUNNER_SHA),
            ("observation_reserve_tracker.py", VARIANT_SHA),
            ("observation_reserve_selection.py", SELECTOR_SHA),
        ):
            path = str((WORK / name).resolve())
            require(
                doc["source_sha256"].get(path) == seal["inputs"].get(path) == digest,
                "actual producer source binding differs",
            )
    common.exact_fields(
        summary,
        dict(
            status="ok",
            native_frames=STOP - START,
            process_frames=STOP - START,
            finish_calls=1,
            all39_roundtrip=True,
            completed_checkpoint_decoder_accepted=True,
            prior_history_prefix_unchanged=True,
            finish_matches_checkpoint_backtrace=True,
            diagnostic_frames=STOP - START,
            physical_ID_truth=False,
        ),
        "completed1660 and preserved full history required",
    )
    require(plan["tracker_context"] == summary["tracker_context"], "final tracker context differs")

    # Inherit the earlier successful first-minute qualification without invoking
    # its Tail/initial66400/full-prefix loader or opening any earlier payload.
    qualified_seal = minute.read(QUALIFIED / "comparison.provenance.json", evidence, QUALIFIED_SHA)
    common.exact_fields(qualified_seal, dict(stage=minute.STAGE, status="ok"), "qualified minute")
    qualification = member(dict(root=QUALIFIED, seal=qualified_seal), "plan.json", evidence)
    milestone = minute.read(MILESTONE / "summary.json", evidence, minute.MILESTONE_SHA)
    require(
        qualification["nodewise_milestone_summary_sha256"] == minute.MILESTONE_SHA
        and minute.old_digest(qualified_seal, MILESTONE / "summary.json") == minute.MILESTONE_SHA
        and qualification["terminal_horizon"] == milestone["horizon_stop"] == STOP
        and qualification["same_original_metadata_and39_signatures"] is True
        and qualification["common_source_prefix_chunks"] == 16
        and milestone["status"] == "ok"
        and milestone["candidate"] == qualification["nodewise_milestone_candidate"]
        and milestone["candidate_arrays"] == qualification["nodewise_all21_declared_signatures"],
        "completed same-horizon baseline qualification differs",
    )
    parent_plan = minute.read(PARENT / "plan.json", evidence, PARENT_PLAN_SHA)
    parent_index = minute.read(PARENT / "input_index.json", evidence, PARENT_INDEX_SHA)
    manifest = minute.read(PARENT_CHECKPOINT / "manifest.json", evidence, PARENT_CHECKPOINT_SHA)
    metadata = minute.read(PARENT_CHECKPOINT / "metadata.json", evidence, PARENT_METADATA_SHA)
    for path, digest in (
        (PARENT / "plan.json", PARENT_PLAN_SHA),
        (PARENT / "input_index.json", PARENT_INDEX_SHA),
        (PARENT_CHECKPOINT / "manifest.json", PARENT_CHECKPOINT_SHA),
        (PARENT_CHECKPOINT / "metadata.json", PARENT_METADATA_SHA),
    ):
        require(minute.old_digest(seal, path) == digest, "new producer parent edge differs")
    for path, digest in (
        (PARENT / "plan.json", PARENT_PLAN_SHA),
        (PARENT / "input_index.json", PARENT_INDEX_SHA),
    ):
        require(minute.old_digest(qualified_seal, path) == digest, "baseline parent edge differs")
    common.exact_fields(
        metadata,
        dict(
            previous_frame=START - 1,
            processed_frames=START - FIRST,
            initialize_count=1,
            restore_count=4,
        ),
        "parent74700",
    )
    require(
        manifest["status"] == "ok"
        and manifest["metadata"]["sha256"] == PARENT_METADATA_SHA
        and manifest["bindings"] == parent_plan["checkpoint_binding"]
        and metadata["context"] == parent_plan["tracker_context"]
        and index["original_binding"]
        == parent_index["original_binding"]
        == plan["original_binding"]
        == manifest["bindings"]["original_binding"]
        and len(parent_index["prefix_chunks"]) == 40
        and index["prefix_chunks"] == parent_index["prefix_chunks"][:16],
        "same seed, checkpoint and consumed16 source chunks required",
    )
    chunk = index["prefix_chunks"][15]
    require((chunk["index"], chunk["start"], chunk["stop"]) == (15, START, STOP), "chunk15 scope")
    chunk_root = (minute.INPUT_ROOT / chunk["attempt_path"]).resolve()
    require(chunk_root.is_relative_to(minute.INPUT_ROOT.resolve()), "registered chunk path")
    for name, digest in (
        ("manifest.json", chunk["manifest_sha256"]),
        ("staging.provenance.json", chunk["provenance_sha256"]),
    ):
        require(minute.old_digest(seal, chunk_root / name) == digest, "new chunk15 input differs")
    validate_import(imported, metadata, manifest, plan["tracker_context"])
    cp = summary["checkpoint"]
    common.exact_fields(
        cp,
        dict(
            arrays=39,
            previous_frame=STOP - 1,
            processed_frames=STOP - FIRST,
            initialize_count=1,
            restore_count=5,
        ),
        "completed endpoint",
    )
    require(
        Path(cp["path"]).resolve() == (TRACKING / "checkpoints/through_076360").resolve()
        and cp["manifest_sha256"]
        == summary["checkpoint_manifest_sha256"]
        == seal["outputs"]["checkpoints/through_076360/manifest.json"]["sha256"],
        "final checkpoint output edge differs",
    )
    new_ref = seal["outputs"]["candidate.npz"]
    require(
        new_ref == summary["candidate"]
        and set(summary["candidate_arrays"]) == metrics.ALL21
        and set(milestone["candidate_arrays"]) == metrics.ALL21
        and Path(new_ref["path"]).resolve() == (TRACKING / "candidate.npz").resolve(),
        "same final candidate21 reference required",
    )
    return (
        dict(
            nodewise=dict(
                candidate=milestone["candidate"], signatures=milestone["candidate_arrays"]
            ),
            observation_reserve=dict(candidate=new_ref, signatures=summary["candidate_arrays"]),
        ),
        dict(
            baseline_milestone_summary_sha256=minute.MILESTONE_SHA,
            baseline_qualification_sha256=QUALIFIED_SHA,
            tracking_sha256=tracking_sha,
            same_parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA,
            same_parent39_signatures_and_metadata=True,
            unchanged_history_prefix_attributed_to_successful_producer=True,
            same_chunk15_and_registered16_prefix=True,
            parent_full_five_minute_completion_claimed=False,
        ),
        summarize_reserve(diagnostics, summary["observation_reserve"]),
    )


def compare_arrays(candidates):
    require(tuple(candidates) == ARMS, "two fixed arms required")
    methods, science, samples = {}, {}, {}
    for arm, candidate in candidates.items():
        minute.validate_candidate(candidate)
        methods[arm] = metrics.summarize(candidate, start=START, stop=STOP)
        science[arm] = {k: v[START - FIRST :] for k, v in candidate.items()}
        ids = np.searchsorted(candidate["frame_idx"], PROBES)
        require(
            np.array_equal(candidate["frame_idx"][ids], PROBES), "development probe native axes"
        )
        samples.update({arm + "_" + k: v[ids].copy() for k, v in candidate.items()})
    geometric, arrays = metrics.geometric_comparison(*(science[arm] for arm in ARMS))
    prefix = {}
    for key in FIELDS:
        a, b = (candidates[arm][key][: START - FIRST] for arm in ARMS)
        prefix[key] = int((a != b).reshape(len(a), -1).any(1).sum())
    geometric["selected_pre74700_backtrace_difference_frames"] = prefix
    geometric["different_selected_prefix_is_not_historical_payload_modification"] = True
    return methods, geometric, arrays, samples


def run(tracking_sha, output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and output != WORK and not output.exists(),
        "fresh workspace output required",
    )
    evidence, out, started = proof.Evidence(), proof.Output(output), proof.utc()
    try:
        code = source_code(evidence)
        refs, binding, reserve = load_headers(tracking_sha, evidence)
        candidates = {arm: minute.load_candidate(refs[arm], evidence) for arm in ARMS}
        before = {arm: common.signature(value) for arm, value in candidates.items()}
        methods, geometric, arrays, samples = compare_arrays(candidates)
        out.write("geometric_permutations.npz", arrays)
        out.write("geometric_permutation_signatures.json", common.signature(arrays))
        out.write("review_samples.npz", samples)
        out.write("review_sample_signatures.json", common.signature(samples))
        scope = dict(
            schema=STAGE,
            process_interval=[START, STOP],
            retained_interval=[FIRST, STOP],
            terminal_horizon=STOP,
            native_frames=STOP - START,
            **binding,
            candidate_references=refs,
            selected_field_signatures=before,
            source_sha256=code,
            decoded_fields=list(FIELDS),
            candidate_archive_decodes_per_arm=1,
            all21_declared_other15_numerically_verified=False,
            development_probes=list(PROBES),
            held_out_validation=False,
            support_recomputed=False,
            raw_slot_correspondence_is_not_physical_ID=True,
            costs_are_not_the_primary_quality_metric=True,
            review_cleared=False,
            checkpoint39_image_mask_SL_GPU_reads=0,
            diagnostic_batch_payload_reads=0,
        )
        out.write("plan.json", scope)
        out.write(
            "summary.json",
            dict(
                **scope,
                status="ok",
                methods=methods,
                geometric_comparison=geometric,
                observation_reserve=reserve,
            ),
        )
        require(
            before == {arm: common.signature(v) for arm, v in candidates.items()},
            "candidate fields were modified",
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
    parser.add_argument("--tracking-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.tracking_sha, args.output)

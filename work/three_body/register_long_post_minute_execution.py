"""Register the fixed post-minute schedule; never launch a process or GPU job.

Successful artifacts are independently hashed. The completed tool session and
exit code are a caller's execution observation, not inferred from a stage seal.
Reflection experiment performance is deliberately not an execution gate.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import register_long_next_phase as next_phase

from medaka3d.provenance import provenance

ROOT, LAB, COMMON = next_phase.ROOT, next_phase.LAB, next_phase.COMMON
WORK = ROOT / "work/three_body"
PLAN = WORK / "LONG_POST_MINUTE_EXECUTION_PLAN.md"
NEXT = WORK / "long_next_phase_plan_v1"
AUDIT = next_phase.AUDIT
SCIENCE = WORK / "reflection_temporal_200_v1"
SCIENCE_REGISTRATION = WORK / "reflection_temporal_registration_v1"
QUALIFICATION = WORK / "reflection_gpu_qualification_v1"
PINS = {
    PLAN: "45bbbfc33f739b95cc0afe3db25dc0e5093733bdf4bc1e956b83683e5fb4f0a9",
    NEXT / "plan.json": "deb79e143eb3473c95d4b0b7ee93e6daf70a5657c15f74b06435d81e34f12aad",
    NEXT / "registration.provenance.json":
        "3b2a61745e00b20ea3f16ec37cfe4d9aac7374d042af1aacd1e66df4d18c7425",
    AUDIT / "audit.provenance.json":
        "a1463bb0cb663c1a38cc7e44e0cd8323b3eb2c60910907178d7b9928ae300512",
    SCIENCE_REGISTRATION / "registration.provenance.json":
        "de6bbfc35edcbe80324efbfc0afd8e8852ae15f1f15d13dbbd254e14d0aae687",
    SCIENCE_REGISTRATION / "plan.json":
        "4fc5cf00a1cc57bd01dbc9497c05e44be46686eda88921e6fdea481ae403b567",
    QUALIFICATION / "qualification.provenance.json":
        "9046cb4cbc61c4d86ae9c5325f3ea71c9a719c466386c743cdcc1bd6d7795404",
    Path(next_phase.__file__).resolve():
        "3ce6ffc6daad72bd9c49585426f0540a654977a5a44a021a8ca13de23bbae275",
    LAB / "run_reflection_temporal.py":
        "89e10c2acc60d2e8ede57906bc13e6a3dce2f5c9b7db571e3c5bf54861a1aadc",
}
require, equal = next_phase.require, next_phase.equal


def completion(root, filename, stage):
    """Inspect success before reading any output payload of a prerequisite."""
    record = COMMON.read_json(root / filename)
    require(record.get("status") == "ok" and record.get("stage") == stage
            and bool(record.get("finished_at")), "completed successful stage required")
    return record


def bind_record(root, filename, record, evidence):
    evidence.add(root / filename, PINS.get(root / filename))
    for category in ("inputs", "outputs"):
        for entry in record[category].values():
            path = Path(entry["path"]).resolve(strict=True)
            require(path.is_relative_to(ROOT), "evidence outside workspace")
            require(category != "outputs" or path.is_relative_to(root), "foreign stage output")
            evidence.add(path, entry["sha256"])


def role(record, name, path):
    require(name in record["outputs"]
            and Path(record["outputs"][name]["path"]).resolve() == path,
            "required output role differs: " + name)
    return COMMON.read_json(path)


def bound_input(record, path, digest):
    matches = [item for item in record["inputs"].values()
               if Path(item["path"]).resolve() == path]
    require(len(matches) == 1 and matches[0]["sha256"] == digest,
            "required prior evidence missing or conflicting: " + str(path))


def validate_science(summary, registered):
    for key, expected in dict(
        schema="reflection_temporal_comparison_v1", frame_range=[79236, 79436],
        native_fps=166, temporal_subsampling=False, arms=["direct", "fresnel"],
        technical_completion=True, initialized_once_each=True, external_resets_each=0,
        inference_old_candidate_deserialized=False, gt_read=False,
        evaluation_after_both_saved=True, code_unchanged=True, inputs_unchanged=True,
    ).items():
        equal(summary[key], expected, "scientific stage completion differs: " + key)
    for arm in ("direct", "fresnel"):
        for key, expected in dict(initialization_count=1, reset_count=0,
                                  track_invocation_count=1, native_frames_read=200).items():
            equal(summary["timings"][arm][key], expected, "incomplete scientific arm")
    require(registered["preflight_only"] is False, "preflight cannot satisfy GPU completion")
    equal(summary["qualification"], registered["qualification"], "qualification binding differs")
    equal(summary["config"], registered["config"], "scientific configuration differs")
    # No comparison, image-loss, support-coverage or biological-ID result is gated.


def schedule(plan, scientific_session, scientific_exit_code):
    require(type(scientific_session) is int and scientific_session > 0
            and type(scientific_exit_code) is int and scientific_exit_code == 0,
            "caller must record a completed scientific session with exit zero")
    for key, expected in dict(
        schema="three_body_long_next_phase_execution_plan_v1",
        full_target_interval=[49800, 348600], next_native_interval=[59760, 66400],
        output_interval=[49800, 66400], consumed_prefix_length=6,
        logical_initializations_expected=1, checkpoint_restores_expected=3,
        external_pose_resets_expected=0,
    ).items():
        equal(plan[key], expected, "fixed next phase differs: " + key)
    require(len(plan["full_selected_chunk_records"]) == 10, "ten ready chunks required")
    binding = plan["frozen_input_and_code_binding"]["inputs"]
    cache = Path(binding["detector_inputs"]["source_cache_manifest"]["path"]).parent
    stage_root = Path(binding["input_stage_root"])
    result = dict(
        schema="long_post_minute_execution_registration_v1",
        full_target_interval=[49800, 348600],
        caller_execution_observation=dict(session_id=scientific_session,
                                          terminal_exit_code=scientific_exit_code,
                                          independently_observed_by_this_script=False),
        tracking=dict(
            executable=str(ROOT / ".venv/Scripts/python.exe"),
            argv=[str(LAB / "run_long_continuation.py"),
                  "--seed-root", str(Path(binding["seed"]["registration_path"]).parent),
                  "--input-stage-root", str(stage_root),
                  "--input-root", binding["seed"]["input_root"],
                  "--through-frame", "66400", "--resume", plan["parent_phase"],
                  "--output", plan["expected_output"]],
            new_native_interval=[59760, 66400], expected_initializations=1,
            expected_restores=3, expected_external_resets=0,
            same_tracker_across_four_ready_chunks=True),
        detector=dict(
            executable=binding["detector_binding"]["runtime"]["executable"],
            argv=[str(LAB / "prepare_chunked_source_masks.py"),
                  "--protocol", binding["protocol_path"], "--source-cache", str(cache),
                  "--out", str(stage_root)],
            ready_chunks=10, remaining_chunks=170, new_native_interval=[66400, 348600],
            max_new_chunks=None, concurrent_writer_count=1,
            reuse_existing_successful_chunks=True, final_global_assembly_required=True),
        later_tracking=dict(approximate_chunks_per_phase=6,
                            concrete_parent_and_endpoint_registration_required=True,
                            final_partial_group_allowed=True,
                            never_skip_unsealed_input=True),
        process_policy=dict(start_after_scientific_session_exit=True,
                            permit_one_tracker_and_one_detector=True,
                            no_additional_detector_writer=True,
                            preserve_failed_attempts=True,
                            never_advance_unsealed_checkpoint=True,
                            source_changes_while_bound_process_alive=False),
        scientific_performance_is_restart_condition=False,
        original_geometry_inputs_and_review_policy_unchanged=True,
        processes_started=False, GPU_queried=False, biological_identity_truth=False,
        full_thirty_minutes_prepared=False, full_thirty_minutes_tracked=False,
        caveats=["The caller must still check actual process completion and available resources.",
                 "Stage seals cannot independently establish OS process exit or kernel overlap.",
                 "Initial correspondence is optimistic; floor and all-three truth are unknown.",
                 "Later backtraces can change prefix winners; this is not an ID accuracy claim."],
    )
    return result


def prepare(evidence, session, exit_code):
    # Refuse incomplete GPU evidence before expensive full-file verification.
    science = completion(SCIENCE, "experiment.provenance.json", "reflection_temporal_comparison")
    prerequisites = [
        (NEXT, "registration.provenance.json", "register_long_next_phase_execution"),
        (AUDIT, "audit.provenance.json", "long_multichunk_minute_independent_audit"),
        (QUALIFICATION, "qualification.provenance.json", "reflection_CPU_CUDA_qualification"),
        (SCIENCE_REGISTRATION, "registration.provenance.json", "reflection_temporal_registration"),
    ]
    records = {root: completion(root, filename, stage) for root, filename, stage in prerequisites}
    for path, digest in PINS.items():
        evidence.add(path, digest)
    for root, filename, _stage in prerequisites:
        bind_record(root, filename, records[root], evidence)
    bind_record(SCIENCE, "experiment.provenance.json", science, evidence)
    plan = role(records[NEXT], "plan", NEXT / "plan.json")
    registered = role(records[SCIENCE_REGISTRATION], "plan", SCIENCE_REGISTRATION / "plan.json")
    equal(role(science, "plan", SCIENCE / "plan.json"), registered, "scientific plan differs")
    summary = role(science, "summary", SCIENCE / "summary.json")
    validate_science(summary, registered)
    for arm in ("direct", "fresnel"):
        path = SCIENCE / f"{arm}_candidate.npz"
        require(Path(science["outputs"][arm + "_candidate"]["path"]).resolve() == path,
                "scientific candidate role differs")
    for filename in ("plan.json", "registration.provenance.json"):
        path = SCIENCE_REGISTRATION / filename
        bound_input(science, path, evidence.paths[str(path)])
    for source, digest in registered["code_hashes"].items():
        path = Path(source).resolve()
        evidence.add(path, digest)
        bound_input(science, path, digest)
    bound_input(records[NEXT], AUDIT / "audit.provenance.json",
                PINS[AUDIT / "audit.provenance.json"])
    equal(plan["parent_audit_seal_sha256"], PINS[AUDIT / "audit.provenance.json"],
          "next plan checked another audit")
    q = role(records[QUALIFICATION], "summary.json", QUALIFICATION / "summary.json")
    require(q["passed"] is True and q["cuda_device_verified"] is True,
            "GPU numerical qualification failed")
    equal(registered["qualification"]["root"], str(QUALIFICATION), "qualification root differs")
    equal(registered["qualification"]["summary_sha256"],
          evidence.paths[str(QUALIFICATION / "summary.json")], "qualification SHA differs")
    bound_input(science, QUALIFICATION / "qualification.provenance.json",
                evidence.paths[str(QUALIFICATION / "qualification.provenance.json")])
    result = schedule(plan, session, exit_code)
    result["prerequisites"] = {str(root): dict(stage=records[root]["stage"],
                                             seal_sha256=evidence.paths[str(root / filename)])
                               for root, filename, _stage in prerequisites}
    result["scientific_completion"] = dict(
        root=str(SCIENCE), seal_sha256=evidence.paths[str(SCIENCE / "experiment.provenance.json")],
        summary_sha256=evidence.paths[str(SCIENCE / "summary.json")],
        performance_used_as_gate=False)
    result["execution_plan_sha256"] = PINS[PLAN]
    require(not Path(plan["expected_output"]).exists(), "next tracking output already exists")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scientific-session", type=int, required=True)
    parser.add_argument("--scientific-exit-code", type=int, choices=[0], required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "new workspace output required")
    evidence = next_phase.minute.prior.support_helpers.Digests()
    sources = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name and next_phase.local_source(name):
            sources.add(Path(name).resolve())
    for path in sorted(sources):
        evidence.add(path, PINS.get(path))
    plan = prepare(evidence, args.scientific_session, args.scientific_exit_code)
    output.mkdir(parents=True, exist_ok=False)
    with provenance("register_long_post_minute_execution", output, name="registration") as record:
        for path in sorted(sources):
            target = output / "code_snapshot" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            evidence.add(target, evidence.paths[str(path)])
            record.output(target.relative_to(output).as_posix(), target,
                          sha256=evidence.paths[str(target)])
        for path, digest in evidence.paths.items():
            if not Path(path).is_relative_to(output):
                record.input(path, path, sha256=digest)
        target = output / "plan.json"
        target.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
        record.output("plan", target, sha256=next_phase.minute.BASE.sha256(target))
        evidence.final_guard()
        COMMON.unchanged({entry["path"]: entry["sha256"] for entry in record.outputs.values()})
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
        record.param("processes_started", False)
    print(json.dumps(dict(status="ok", plan=str(target),
                         sha256=next_phase.minute.BASE.sha256(target))))


if __name__ == "__main__":
    main()

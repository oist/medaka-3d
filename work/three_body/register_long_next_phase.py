"""Register only the audited 59760 -> 66400 continuation; never run a tracker.

The existing immutable loader chooses ten valid chunk attempts and checks all
mask payloads. The immutable phase auditor verifies the parent's 39-array
checkpoint and 21-array backtrace. The completed independent minute audit is a
required input, not a statement that its checks were rerun here. This command
does not decode raw video or initialize detector/tracker/GPU state.
"""

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

import audit_long_multichunk_minute as minute
import numpy as np

from medaka3d.provenance import provenance

ROOT, LAB, COMMON = minute.REPO, minute.LAB, minute.COMMON
FIRST, PARENT_STOP, STOP, FULL_STOP = 49800, 59760, 66400, 348600
PARENT = ROOT / "work/three_body/long_continuation_through59760_v1"
AUDIT = ROOT / "work/three_body/long_continuation_through59760_independent_audit_v1"
MINUTE_AUDITOR_SHA = "3e186c0f06e7bfbe64e4dd8bfdae066533c8d4203731c4f46c1f97a3f475dbbd"
require, equal = minute.require, minute.equal


def local_source(path):
    path = Path(path).resolve()
    return path.suffix == ".py" and any(path.is_relative_to(scope) for scope in (
        LAB, ROOT / "src", ROOT / "work/three_body"))


def read_sealed(root, filename, stage, evidence):
    path = Path(root) / filename
    evidence.add(path)
    record = COMMON.read_json(path)
    require(record.get("status") == "ok" and record.get("stage") == stage,
            "successful sealed prerequisite required: " + str(path))
    require(record["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
            "prerequisite lacks completed integrity guards")
    return record


def verify_audit(root, record, parent_record, evidence):
    """Require exact parent output binding before opening its state arrays."""
    for entry in record["outputs"].values():
        path = Path(entry["path"]).resolve(strict=True)
        require(path.is_relative_to(root), "audit output escaped its root")
        evidence.add(path, entry["sha256"])
    path = root / "summary.json"
    require(Path(record["outputs"]["summary.json"]["path"]).resolve() == path,
            "audit summary role mismatch")
    result = COMMON.read_json(path)
    for key, expected in dict(
        schema="long_multichunk_minute_independent_audit_v1", status="ok", run=str(PARENT),
        frame_range=[FIRST, PARENT_STOP], native_frames=PARENT_STOP-FIRST,
        logical_initialization_count=1, checkpoint_restore_count=2, external_pose_reset_count=0,
        all_39_checkpoint_arrays_verified=True, full_sequence_complete=False,
        independent_ID_truth=False,
    ).items():
        equal(result[key], expected, "audit prerequisite differs: " + key)
    require(result["candidate_sha256"] == parent_record["outputs"]["candidate"]["sha256"],
            "audit checked a different candidate")
    equal(result["history"]["active_beam_count"], 6, "audit active beam differs")
    require(result["history"]["all_parent_recurrences_exact"] is True,
            "audit did not verify parent recurrence")
    for entry in parent_record["outputs"].values():
        path = str(Path(entry["path"]).resolve())
        audited = record["inputs"].get(path)
        require(audited is not None and Path(audited["path"]).resolve() == Path(path)
                and audited["sha256"] == entry["sha256"], "parent output absent from audit")
    parent_seal = PARENT / "continuation.provenance.json"
    require(record["inputs"][str(parent_seal)]["sha256"] == evidence.add(parent_seal),
            "audit parent success seal differs")
    source = Path(minute.__file__).resolve()
    require(record["inputs"][str(source)]["sha256"] == MINUTE_AUDITOR_SHA,
            "audit source is not the frozen independent verifier")
    # Preserve the full audit evidence, using one full SHA per canonical file
    # identity in this registry. This avoids rehashing every MP4 hardlink alias.
    for name, entry in record["inputs"].items():
        require(Path(name).resolve() == Path(entry["path"]).resolve(), "audit input role differs")
        evidence.add(entry["path"], entry["sha256"])
    return result


def prepare(audit_root, evidence):
    parent_seal = read_sealed(PARENT, "continuation.provenance.json",
                              "three_body_long_continuation", evidence)
    audit_seal = read_sealed(audit_root, "audit.provenance.json",
                             "long_multichunk_minute_independent_audit", evidence)
    audit = verify_audit(audit_root, audit_seal, parent_seal, evidence)
    binding = parent_seal["params"]["bindings"]
    require(parent_seal["params"]["through_frame"] == PARENT_STOP,
            "parent endpoint differs")
    for source, expected in binding["code_sha256"].items():
        evidence.add(source, expected)
    protocol_path = LAB / "long_tracking_protocol.json"
    evidence.add(protocol_path, binding["tracking_protocol_sha256"])
    protocol = COMMON.read_json(protocol_path)
    for relative, digest in protocol["frozen_tracker_sources_sha256"].items():
        evidence.add(ROOT / relative, digest)
    evidence.add(LAB / "long_input_protocol.json", protocol["input_protocol_sha256"])
    equal(protocol["source_frame_range"], [FIRST, FULL_STOP], "full target changed")
    equal(binding["package_versions"], protocol["package_versions"], "registered runtime changed")
    equal(binding["torch_cpu_threads"], protocol["torch_cpu_threads"], "registered threads differ")
    inputs = minute.prior.first.verified.load_long_inputs(
        Path(binding["inputs"]["seed"]["registration_path"]).parent,
        Path(binding["inputs"]["input_stage_root"]),
        Path(binding["inputs"]["seed"]["input_root"]), through_frame=STOP)
    equal(inputs.fixed_bindings, binding["inputs"], "immutable long input binding differs")
    require(inputs.full_frame_range == (FIRST, FULL_STOP) and len(inputs.chunk_roots) == 10
            and np.array_equal(inputs.frame_idx, np.arange(FIRST, STOP)),
            "ten complete native chunks required")
    for path, digest in inputs.verified_hashes.items():
        evidence.add(path, digest)
    parent = minute.load_phase(PARENT, parent_seal, inputs, evidence, stop=PARENT_STOP,
                               phase_start=53120, restores=2, chunks=6)
    minute.prior.first.verified.verify_prefix_records(parent["consumed"], inputs.chunk_records)
    arrays, metadata = parent["arrays"], parent["metadata"]
    require(len(arrays) == 39 and arrays["state_pose"].shape == (6, 3, 3, 3),
            "full 39-array checkpoint and six active bodies required")
    require(metadata["initialize_count"] == 1 and metadata["restore_count"] == 2
            and metadata["previous_frame"] == PARENT_STOP-1,
            "parent lifecycle is not ready to resume")
    context = metadata["context"]
    equal(context["initial_pose"], COMMON.digest_array(inputs.initial_pose_xyz),
          "initial pose differs")
    equal(context["cameras"], COMMON.normalize([dataclasses.asdict(c) for c in inputs.rig.cameras]),
          "calibration context differs")
    for a, b in (("tracker_config", "tracker_config"), ("recovery_config", "node_recovery_config"),
                 ("persistence_config", "persistence_config")):
        equal(context[a], protocol[b], "checkpoint configuration differs")
    require(context["implementation"] == "AcceleratedContinuousPersistentNodeTracker"
            and context["views"] == list(inputs.rig.views) and context["world_units"] == "mm",
            "checkpoint implementation/coordinates differ")
    rows = int(arrays["history_offsets"][-1])
    equal(rows, audit["history"]["retained_history_rows"], "audited history size differs")
    expected_next = [[PARENT_STOP + 1660*i, PARENT_STOP + 1660*(i+1)] for i in range(4)]
    equal([[c["start"], c["stop"]] for c in inputs.chunk_records[6:]], expected_next,
          "new chunk order differs")
    return dict(
        schema="three_body_long_next_phase_execution_plan_v1",
        full_target_interval=[FIRST, FULL_STOP], parent_phase=str(PARENT),
        parent_processed_interval=[FIRST, PARENT_STOP], next_native_interval=[PARENT_STOP, STOP],
        output_interval=[FIRST, STOP], new_input_intervals=expected_next,
        expected_output=str(ROOT / "work/three_body/long_continuation_through66400_v1"),
        frozen_input_and_code_binding=binding, parent_checkpoint_context=context,
        parent_history_rows_expected=rows, parent_active_beam_expected=6,
        logical_initializations_expected=1, checkpoint_restores_expected=3,
        external_pose_resets_expected=0, independent_parent_audit=str(audit_root),
        parent_candidate_sha256=parent_seal["outputs"]["candidate"]["sha256"],
        parent_audit_summary_sha256=evidence.paths[str(audit_root / "summary.json")],
        parent_audit_seal_sha256=evidence.paths[str(audit_root / "audit.provenance.json")],
        full_selected_chunk_records=inputs.chunk_records, consumed_prefix_length=6,
        next_selected_chunk_records=inputs.chunk_records[6:],
        execution="Use the unchanged run_long_continuation.py with --resume parent, "
        "--through-frame 66400. One process, one restore, all four readers on the same tracker; "
        "no begin/reset between chunks. The script creates no process or tracker.",
        runtime_scope="Recorded frozen context verified; GPU runtime must be checked again "
        "by the existing runner when execution starts. No GPU queried by registration.",
        limits=["Full thirty-minute preparation/tracking is not complete.",
                "The checkpoint retains all histories; its final backtrace remains provisional.",
                "No new uninterrupted reference run or independent three-animal truth is claimed.",
                "Original seed, inputs, geometry policy and review policy remain unchanged."],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output, audit_root = args.output.resolve(), args.audit.resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "new workspace output required")
    require(audit_root.is_relative_to(ROOT), "workspace audit required")
    evidence = minute.prior.support_helpers.Digests()
    sources = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if local_source(path):
                sources.add(path)
    for path in sorted(sources):
        expected = MINUTE_AUDITOR_SHA if path == Path(minute.__file__).resolve() else None
        evidence.add(path, expected)
    plan = prepare(audit_root, evidence)
    expected_output = Path(plan["expected_output"])
    require(not expected_output.exists(), "next tracking output already exists")
    output.mkdir(parents=True, exist_ok=False)
    with provenance("register_long_next_phase_execution", output, name="registration") as record:
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
        record.output("plan", target, sha256=minute.BASE.sha256(target))
        evidence.final_guard()
        COMMON.unchanged({e["path"]: e["sha256"] for e in record.outputs.values()})
        record.param("tracking_detector_GPU_or_GT_performed", False)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
    print(json.dumps(dict(plan=str(target), sha256=minute.BASE.sha256(target), status="ok")))


if __name__ == "__main__":
    main()

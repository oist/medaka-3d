"""CPU continuation registration/audit helpers for the frozen long runner.

No output writes, process management, GPU initialization, video decoding or GT.
The supervisor owns registration timing, previous audit lineage, output seals,
source snapshots and final guards. Prepare returns (JSON plan, loaded context);
the context can be discarded before launching the immutable child process.
The suffix recurrence follows the frozen minute auditor, with native intervals
and parent row counts read from the bound checkpoint instead of fixed constants.
"""

import dataclasses
import json
from datetime import datetime
from pathlib import Path

import audit_long_multichunk_minute as old
import numpy as np

REPO, LAB, COMMON, BASE = old.REPO, old.LAB, old.COMMON, old.BASE
FIRST, FULL_STOP, CHUNK = 49800, 348600, 1660
Digests = old.prior.support_helpers.Digests
require, equal = old.require, old.equal
BOOTSTRAP_ROOT = REPO / "work/three_body/long_next_phase_plan_v1"
BOOTSTRAP_PLAN_SHA = "deb79e143eb3473c95d4b0b7ee93e6daf70a5657c15f74b06435d81e34f12aad"
BOOTSTRAP_SEAL_SHA = "3b2a61745e00b20ea3f16ec37cfe4d9aac7374d042af1aacd1e66df4d18c7425"
MINUTE_AUDIT = REPO / "work/three_body/long_continuation_through59760_independent_audit_v1"
MINUTE_AUDIT_SHA = "a1463bb0cb663c1a38cc7e44e0cd8323b3eb2c60910907178d7b9928ae300512"


def native_endpoint(value):
    require(type(value) is int and FIRST < value <= FULL_STOP
            and (value-FIRST) % CHUNK == 0, "registered integer chunk endpoint required")
    return value


def phase_range(start, stop):
    native_endpoint(start)
    native_endpoint(stop)
    require(start >= 66400 and stop == min(start+6*CHUNK, FULL_STOP) and stop > start,
            "next six ready chunks or final partial group required")


def sealed(root, evidence):
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(REPO), "phase outside workspace")
    path = root / "continuation.provenance.json"
    evidence.add(path)
    result = COMMON.read_json(path)
    require(result.get("status") == "ok" and result.get("stage") == "three_body_long_continuation"
            and result["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
            "completed successful continuation required")
    native_endpoint(result["params"]["through_frame"])
    return root, result


def verified_inputs(binding, stop, evidence):
    source = LAB / "long_tracking_protocol.json"
    evidence.add(source, binding["tracking_protocol_sha256"])
    protocol = COMMON.read_json(source)
    equal(protocol["source_frame_range"], [FIRST, FULL_STOP], "registered full interval differs")
    for relative, digest in protocol["frozen_tracker_sources_sha256"].items():
        evidence.add(REPO / relative, digest)
    evidence.add(LAB / "long_input_protocol.json", protocol["input_protocol_sha256"])
    for source, digest in binding["code_sha256"].items():
        evidence.add(source, digest)
    equal(binding["package_versions"], protocol["package_versions"], "frozen runtime differs")
    equal(binding["torch_cpu_threads"], protocol["torch_cpu_threads"], "thread count differs")
    registered = binding["inputs"]
    inputs = old.prior.first.verified.load_long_inputs(
        Path(registered["seed"]["registration_path"]).parent,
        Path(registered["input_stage_root"]), Path(registered["seed"]["input_root"]),
        through_frame=stop)
    equal(inputs.fixed_bindings, registered, "input context differs")
    require(inputs.full_frame_range == (FIRST, FULL_STOP)
            and np.array_equal(inputs.frame_idx, np.arange(FIRST, stop))
            and len(inputs.chunk_records) == (stop-FIRST)//CHUNK, "native prefix differs")
    for path, digest in inputs.verified_hashes.items():
        evidence.add(path, digest)
    return inputs, protocol


def validate_summary(summary, *, stop, phase_start, restores, chunks):
    for key, expected in dict(
        schema="long_continuation_phase_v1", technical_completion=True,
        full_sequence_complete=stop == FULL_STOP, prefix_backtrace_provisional=stop != FULL_STOP,
        registered_interval=[FIRST, FULL_STOP], processed_interval=[FIRST, stop],
        processed_total=stop-FIRST, phase_frames=[phase_start, stop],
        phase_processed=stop-phase_start, consumed_chunk_count=chunks,
        logical_initialization_count=1, checkpoint_restore_count=restores,
        external_pose_reset_count=0, temporal_subsampling=False, human_annotation_input=False,
        post_initialization_baseline_coordinates_input=False,
    ).items():
        equal(summary.get(key), expected, "phase summary differs: " + key)


def load_phase(root, seal, inputs, evidence, protocol):
    """Adapt the immutable minute load_phase checks, including the full endpoint."""
    binding = seal["params"]["bindings"]
    stop = native_endpoint(seal["params"]["through_frame"])
    chunks = (stop-FIRST)//CHUNK
    later_roots = inputs.chunk_roots[chunks:]
    expected = {p: d for p, d in inputs.verified_hashes.items()
                if not any(Path(p).is_relative_to(r) for r in later_roots)}
    expected.update(binding["code_sha256"])
    resume = Path(seal["params"]["resume_phase"]).resolve(strict=True)
    expected.update({str(resume / f): BASE.sha256(resume / f) for f in old.prior.PARENT_FILES})
    require(set(expected) == set(seal["inputs"]), "phase input membership differs")
    for name, digest in expected.items():
        item = seal["inputs"][name]
        require(Path(item["path"]).resolve() == Path(name) and item["sha256"] == digest,
                "phase input role/path/SHA differs")
        evidence.add(name, digest)
    for item in seal["outputs"].values():
        path = Path(item["path"]).resolve(strict=True)
        require(path.is_relative_to(root) and path.stat().st_size == item["size"],
                "output escaped run or changed size")
        evidence.add(path, item["sha256"])
    for role, filename in {"candidate": "candidate.npz", "summary": "summary.json",
                           "progress": "progress.jsonl", "consumed_chunks": "consumed_chunks.json",
                           "checkpoint_manifest": "checkpoint/manifest.json"}.items():
        require(Path(seal["outputs"][role]["path"]).resolve() == root / filename,
                "named output binding differs")
    for filename in ("checkpoint/arrays.npz", "checkpoint/metadata.json"):
        require(str(root / filename) in seal["outputs"], "unsealed checkpoint payload")
    for source, digest in binding["code_sha256"].items():
        path = root / "code_snapshot" / Path(source).relative_to(REPO)
        require(str(path) in seal["outputs"], "unsealed producer source snapshot")
        evidence.add(path, digest)
    arrays, metadata, manifest = old.prior.read_checkpoint(
        root / "checkpoint", expected_bindings=binding)
    summary = COMMON.read_json(root / "summary.json")
    parent_metadata = COMMON.read_json(resume / "checkpoint/metadata.json")
    phase_start = FIRST+parent_metadata["processed_frames"]
    restores = parent_metadata["restore_count"]+1
    validate_summary(summary, stop=stop, phase_start=phase_start, restores=restores, chunks=chunks)
    require(len(arrays) == 39 and metadata["processed_frames"] == stop-FIRST
            and metadata["previous_frame"] == stop-1 and metadata["initialize_count"] == 1
            and metadata["restore_count"] == restores
            and metadata["node_recovery_diagnostics"] == summary["node_recovery_diagnostics"]
            and manifest["payload"]["bytes"] == summary["checkpoint_payload_bytes"],
            "checkpoint lifecycle or array count differs")
    context = metadata["context"]
    equal(context, parent_metadata["context"], "checkpoint parent context differs")
    equal(context["initial_pose"], COMMON.digest_array(inputs.initial_pose_xyz), "seed differs")
    equal(context["cameras"], COMMON.normalize([dataclasses.asdict(c) for c in inputs.rig.cameras]),
          "calibration context differs")
    for key, source in (("tracker_config", "tracker_config"),
                        ("recovery_config", "node_recovery_config"),
                        ("persistence_config", "persistence_config")):
        equal(context[key], protocol[source], "checkpoint settings differ")
    require(context["implementation"] == "AcceleratedContinuousPersistentNodeTracker"
            and context["views"] == list(inputs.rig.views) and context["world_units"] == "mm",
            "checkpoint implementation/coordinates differ")
    consumed = COMMON.read_json(root / "consumed_chunks.json")
    equal(consumed, inputs.chunk_records[:chunks], "consumed chunk evidence changed")
    progress = [json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()]
    counts = [r["processed"] for r in progress]
    require(counts and counts == sorted(set(counts)) and counts[-1] == stop-FIRST
            and all(r["source_frame"] == FIRST+r["processed"]-1
                    and r["phase_processed"] == r["processed"]-(phase_start-FIRST)
                    and r["phase_processed"] > 0 for r in progress), "native progress differs")
    require({c["stop"]-FIRST for c in consumed if c["stop"] > phase_start}.issubset(counts),
            "progress lacks completed chunk endpoints")
    candidate = old.read_npz(root / "candidate.npz")
    traced = COMMON.traced_candidate(arrays)
    require(set(candidate) == set(traced) and len(candidate) == 21, "candidate schema differs")
    comparisons = {k: COMMON.same_array(candidate[k], traced[k], k) for k in candidate}
    COMMON.same_array(candidate["frame_idx"], np.arange(FIRST, stop, dtype=np.int64), "frames")
    return dict(arrays=arrays, metadata=metadata, summary=summary, consumed=consumed,
                candidate=candidate, comparisons=comparisons)


def suffix_history(parent, current):
    """Retain all parent rows; independently replay every new selected beam row."""
    a, old_a = current["arrays"], parent["arrays"]
    metadata = current["metadata"]
    split, total = parent["metadata"]["processed_frames"], metadata["processed_frames"]
    offsets, old_offsets = a["history_offsets"], old_a["history_offsets"]
    require(offsets.dtype == np.int64 and offsets.shape == (total+1,) and offsets[0] == 0
            and np.all((np.diff(offsets) >= 1) & (np.diff(offsets) <= 6))
            and 0 < split < total and len(old_a["state_pose"]) == 6,
            "history offsets/parent beam differ")
    rows = int(old_offsets[-1])
    COMMON.same_array(offsets[:split+1], old_offsets, "parent history offsets")
    for key in COMMON.HISTORY_FIELDS:
        require(len(a["history_"+key]) == offsets[-1], "unaccounted history rows")
        COMMON.same_array(a["history_"+key][:rows], old_a["history_"+key], "parent " + key)
    require(metadata["record_fields"] == list(COMMON.HISTORY_FIELDS)
            and metadata["state_fields"] == list(COMMON.STATE_FIELDS), "history fields differ")
    expected_keys = ({"history_offsets"} | {"history_"+k for k in COMMON.HISTORY_FIELDS}
                     | {"state_"+k for k in COMMON.STATE_FIELDS}
                     | {k for k, present in metadata["previous_observation"].items() if present})
    require(set(a) == expected_keys, "unexpected checkpoint fields")
    previous = [{k: old_a["state_"+k][i] for k in COMMON.STATE_FIELDS} for i in range(6)]
    eligible = parent["metadata"]["node_recovery_diagnostics"]["calls_with_age_eligible_fish"]
    nonzero, boundaries = 0, []
    for t in range(split, total):
        native = FIRST+t
        if t % CHUNK == 0:
            require(len(previous) == 6, "reader boundary lost an active parent")
            boundaries.append(native)
        eligible += bool(np.any((previous[0]["node_missing"][:, 0] >= 4)
                                & (previous[0]["direct_pectoral_age"] >= 4)))
        following = []
        for row in range(int(offsets[t]), int(offsets[t+1])):
            r = {k: a["history_"+k][row] for k in COMMON.HISTORY_FIELDS}
            pi = int(r["parent"])
            require(0 <= pi < len(previous) and r["frame_idx"] == native, "parent/frame differs")
            nonzero += pi != 0
            p, pose, sources, held = previous[pi], r["pose_xyz"], r["sources"], r["observed"]
            require(np.isfinite(pose).all() and np.all(pose >= [0, 0, -6])
                    and np.all(pose <= [192, 192, 186]) and np.all(sources >= -1),
                    "history engineering pose/source contract differs")
            COMMON.same_array(held, (sources >= 0).sum(1) >= 2, "observed counts")
            for view in range(3):
                for node in range(3):
                    used = sources[:, view, node]
                    used = used[used >= 0]
                    require(len(used) == len(np.unique(used)), "source exclusivity violation")
            body = held[:, 1] | (r["mask_supported"] & (sources[:, :, :2] >= 0).any((1, 2)))
            missing = np.where(body, 0, p["missing"]+1)
            age = np.where(held, 0, p["node_missing"]+1)
            direct_age = np.where(held[:, 1], p["direct_pectoral_age"]+1, 0)
            for key, value in (("body_supported", body), ("missing", missing),
                               ("node_missing", age), ("direct_pectoral_age", direct_age)):
                COMMON.same_array(r[key], value, "parent recurrence " + key)
            require(not np.any(p["review"] & ~r["identity_review"])
                    and not np.any((missing >= 4) & ~r["identity_review"]), "sticky review erased")
            cost = p["cost"]+r["costs"]
            require(np.isfinite(cost), "nonfinite path cost")
            COMMON.same_array(np.asarray(cost), np.asarray(r["path_costs"]), "path cost")
            velocity = .5*p["velocity"] + .5*(pose[:, 1]-p["pose"][:, 1])
            norm = np.linalg.norm(velocity, axis=-1, keepdims=True)
            velocity *= np.minimum(1., 2./np.maximum(norm, 1e-12))
            velocity = np.where(body[:, None], velocity, .8*p["velocity"])
            following.append(dict(pose=pose, velocity=velocity, missing=missing,
                                  review=r["identity_review"], cost=float(cost), node_missing=age,
                                  direct_pectoral_age=direct_age))
        previous = following
    require(len(previous) == 6, "final checkpoint lost a beam")
    for key in COMMON.STATE_FIELDS:
        COMMON.same_array(a["state_"+key], np.asarray([p[key] for p in previous]), "final " + key)
    require(eligible == metadata["node_recovery_diagnostics"]["calls_with_age_eligible_fish"]
            and current["summary"]["retained_history_rows"] == offsets[-1]
            and current["summary"]["active_beam_count"] == 6, "history counters differ")
    return dict(parent_history_rows_exact=rows, parent_all_history_fields_exact=True,
                new_history_rows_verified=int(offsets[-1])-rows,
                new_nonzero_parent_rows=nonzero, active_beam_count=6,
                retained_history_rows=int(offsets[-1]),
                all_suffix_parent_recurrences_exact=True,
                reader_boundary_frames=boundaries, independent_age_eligible_calls=eligible)


def prepare_phase(parent, through_frame, output_run, evidence):
    through_frame = native_endpoint(through_frame)
    parent, seal = sealed(parent, evidence)
    start = seal["params"]["through_frame"]
    phase_range(start, through_frame)
    output_run = Path(output_run).resolve()
    require(output_run.is_relative_to(REPO) and not output_run.exists(), "new output required")
    binding = seal["params"]["bindings"]
    inputs, protocol = verified_inputs(binding, through_frame, evidence)
    previous = load_phase(parent, seal, inputs, evidence, protocol)
    require(previous["arrays"]["state_pose"].shape == (6, 3, 3, 3), "six active parents required")
    first_chunk = (start-FIRST)//CHUNK
    plan = dict(
        schema="long_continuation_registered_phase_v1", full_target_interval=[FIRST, FULL_STOP],
        parent_phase=str(parent),
        parent_seal_sha256=evidence.paths[str(parent / "continuation.provenance.json")],
        expected_output=str(output_run), parent_processed_interval=[FIRST, start],
        next_native_interval=[start, through_frame], output_interval=[FIRST, through_frame],
        expected_initializations=1, expected_restores=previous["metadata"]["restore_count"]+1,
        expected_external_resets=0, parent_active_beam=6,
        parent_history_rows=int(previous["arrays"]["history_offsets"][-1]),
        frozen_input_and_code_binding=binding,
        parent_checkpoint_context=previous["metadata"]["context"],
        full_selected_chunk_records=inputs.chunk_records,
        next_selected_chunk_records=inputs.chunk_records[first_chunk:],
        full_sequence_complete=through_frame == FULL_STOP,
        prefix_backtrace_provisional=through_frame != FULL_STOP,
        independent_ID_truth=False, processes_started=False)
    return plan, dict(inputs=inputs, parent=previous, protocol=protocol)


def verify_phase(run, plan, evidence):
    return _verify_phase(run, plan, evidence, bootstrap=False)


def _verify_phase(run, plan, evidence, *, bootstrap):
    require(plan["schema"] == "long_continuation_registered_phase_v1", "unknown phase plan")
    run, record = sealed(run, evidence)
    parent, prior_record = sealed(plan["parent_phase"], evidence)
    require(run == Path(plan["expected_output"]).resolve()
            and Path(record["params"]["resume_phase"]).resolve() == parent,
            "executed phase/parent differs from plan")
    equal(evidence.paths[str(parent / "continuation.provenance.json")],
          plan["parent_seal_sha256"], "registered parent seal differs")
    start, stop = plan["next_native_interval"]
    if bootstrap:
        equal([start, stop], [59760, 66400], "bootstrap interval differs")
    else:
        phase_range(start, stop)
    for key, expected in dict(
        full_target_interval=[FIRST, FULL_STOP], parent_processed_interval=[FIRST, start],
        output_interval=[FIRST, stop], expected_initializations=1,
        expected_external_resets=0, parent_active_beam=6,
        full_sequence_complete=stop == FULL_STOP, prefix_backtrace_provisional=stop != FULL_STOP,
    ).items():
        equal(plan[key], expected, "registered lifecycle/range differs: " + key)
    require(record["params"]["through_frame"] == stop
            and prior_record["params"]["through_frame"] == start, "executed endpoint differs")
    binding = record["params"]["bindings"]
    equal(binding, prior_record["params"]["bindings"], "parent/current binding differs")
    equal(binding, plan["frozen_input_and_code_binding"], "registered binding differs")
    inputs, protocol = verified_inputs(binding, stop, evidence)
    equal(inputs.chunk_records, plan["full_selected_chunk_records"], "registered chunks differ")
    equal(inputs.chunk_records[(start-FIRST)//CHUNK:], plan["next_selected_chunk_records"],
          "registered new chunk subset differs")
    before = load_phase(parent, prior_record, inputs, evidence, protocol)
    after = load_phase(run, record, inputs, evidence, protocol)
    equal(after["metadata"]["restore_count"], plan["expected_restores"], "restore count differs")
    equal(before["metadata"]["context"], plan["parent_checkpoint_context"],
          "registered context differs")
    equal(after["metadata"]["context"], before["metadata"]["context"], "restored context differs")
    equal(int(before["arrays"]["history_offsets"][-1]), plan["parent_history_rows"],
          "parent rows differ")
    history = suffix_history(before, after)
    counts = old.prior.support.direct_counts(after["candidate"])
    for key, value in (("direct_pectoral_fish_frames", "direct_pectoral_count"),
                       ("accepted_pectoral_fish_frames", "accepted_pectoral_count"),
                       ("review_fish_frames", "identity_review_count"),
                       ("direct_pectoral_per_slot", "per_slot_direct_pectoral"),
                       ("accepted_pectoral_per_slot", "per_slot_accepted_pectoral")):
        equal(counts[key], after["summary"][value], "summary availability differs")
    return dict(schema="long_continuation_generic_phase_audit_v1", status="ok", run=str(run),
                parent=str(parent), frame_range=[FIRST, stop], new_interval=[start, stop],
                candidate_sha256=evidence.paths[str(run / "candidate.npz")],
                all_39_checkpoint_arrays_verified=True,
                all_21_candidate_arrays_exact_backtrace=after["comparisons"],
                logical_initialization_count=1,
                checkpoint_restore_count=after["metadata"]["restore_count"],
                external_pose_reset_count=0, history=history, counts=counts,
                full_sequence_complete=stop == FULL_STOP,
                prefix_backtrace_provisional=stop != FULL_STOP,
                independently_redecoded_raw_images=0, independent_geometric_reprojection=False,
                independent_ID_truth=False,
                scope="All saved input/code/output hashes, checkpoint codec, complete backtrace, "
                      "exact parent history, all new beam recurrences and source exclusivity. "
                      "No independent raw decode, flow rerun or 2mm/12px reprojection here; "
                      "those remain producer contracts and separate sampled audits.")


def adapt_bootstrap_plan(plan, parent_seal_sha256):
    """Only translate the already hash-verified four-chunk plan's field names."""
    for key, value in dict(
        schema="three_body_long_next_phase_execution_plan_v1",
        full_target_interval=[FIRST, FULL_STOP], parent_processed_interval=[FIRST, 59760],
        next_native_interval=[59760, 66400], output_interval=[FIRST, 66400],
        logical_initializations_expected=1, checkpoint_restores_expected=3,
        external_pose_resets_expected=0, parent_active_beam_expected=6,
        parent_audit_seal_sha256=MINUTE_AUDIT_SHA,
    ).items():
        equal(plan[key], value, "bootstrap registration differs: " + key)
    return dict(
        schema="long_continuation_registered_phase_v1", full_target_interval=[FIRST, FULL_STOP],
        parent_phase=plan["parent_phase"], parent_seal_sha256=parent_seal_sha256,
        expected_output=plan["expected_output"], parent_processed_interval=[FIRST, 59760],
        next_native_interval=[59760, 66400], output_interval=[FIRST, 66400],
        expected_initializations=1, expected_restores=3, expected_external_resets=0,
        parent_active_beam=6, parent_history_rows=plan["parent_history_rows_expected"],
        frozen_input_and_code_binding=plan["frozen_input_and_code_binding"],
        parent_checkpoint_context=plan["parent_checkpoint_context"],
        full_selected_chunk_records=plan["full_selected_chunk_records"],
        next_selected_chunk_records=plan["next_selected_chunk_records"],
        full_sequence_complete=False, prefix_backtrace_provisional=True)


def verify_bootstrap(run, evidence):
    """Audit 59760→66400 after success, anchored in the fixed independent minute audit."""
    run, current_record = sealed(run, evidence)  # Never read an unfinished candidate.
    plan_path = BOOTSTRAP_ROOT / "plan.json"
    seal_path = BOOTSTRAP_ROOT / "registration.provenance.json"
    audit_path = MINUTE_AUDIT / "audit.provenance.json"
    evidence.add(plan_path, BOOTSTRAP_PLAN_SHA)
    evidence.add(seal_path, BOOTSTRAP_SEAL_SHA)
    evidence.add(audit_path, MINUTE_AUDIT_SHA)
    plan = COMMON.read_json(plan_path)
    registration, audit = COMMON.read_json(seal_path), COMMON.read_json(audit_path)
    require(datetime.fromisoformat(registration["finished_at"])
            <= datetime.fromisoformat(current_record["started_at"]),
            "bootstrap registration postdates tracking")
    for record, stage in ((registration, "register_long_next_phase_execution"),
                          (audit, "long_multichunk_minute_independent_audit")):
        require(record["status"] == "ok" and record["stage"] == stage
                and record["params"]["all_input_source_snapshot_output_sha_unchanged"] is True,
                "bootstrap requires completed registration and independent minute audit")
        for entry in (*record["inputs"].values(), *record["outputs"].values()):
            evidence.add(entry["path"], entry["sha256"])
    parent = Path(plan["parent_phase"]).resolve()
    require(parent == REPO / "work/three_body/long_continuation_through59760_v1"
            and run == REPO / "work/three_body/long_continuation_through66400_v1"
            and Path(plan["independent_parent_audit"]).resolve() == MINUTE_AUDIT,
            "bootstrap parent/run/audit path differs")
    parent_path = parent / "continuation.provenance.json"
    require(audit["inputs"][str(parent_path)]["sha256"] == evidence.paths[str(parent_path)],
            "bootstrap audit checked another parent")
    translated = adapt_bootstrap_plan(plan, evidence.paths[str(parent_path)])
    result = _verify_phase(run, translated, evidence, bootstrap=True)
    result["bootstrap_lineage"] = dict(
        previous_independent_audit=str(MINUTE_AUDIT), audit_seal_sha256=MINUTE_AUDIT_SHA,
        registered_plan=str(plan_path), registered_plan_sha256=BOOTSTRAP_PLAN_SHA,
        registration_seal_sha256=BOOTSTRAP_SEAL_SHA,
        new_preregistration_performed=False)
    return result

"""Post-run CPU primitives; no final-audit CLI or unfinished-run fallback.

This first implementation supplies partitioned support and the sealed final
chain/candidate loader only. Fixed native samples, boundary-flow replay, global
detector assembly verification and final report/seal remain separate work.
The caller snapshots code and performs evidence.final_guard before sealing.
No GT, detector, tracker, CUDA query or raw video decode is performed here.
"""

from datetime import datetime
from pathlib import Path

import long_phase_audit as phase
import numpy as np

REPO, LAB, COMMON = phase.REPO, phase.LAB, phase.COMMON
FIRST, INITIAL_STOP, FINAL_STOP, BLOCK = 49800, 66400, 348600, 1660
SUPPORT = phase.old.prior.support
Digests = phase.Digests
require, equal = phase.require, phase.equal
PLAN = REPO / "work/three_body/LONG_FINAL_VALIDATION_PLAN.md"
SUPERVISOR = REPO / "work/three_body/run_long_remaining_supervisor.py"
PINS = {
    PLAN: "a3cc8deffdf9b18db3a34ed3ee4c1112f406b5aa69ebe616e9b3fe38010a221d",
    Path(
        phase.__file__
    ).resolve(): "a89541031a662b1c6633d6b3cc996dd7d5e88da6627eb3e4ef17261103963ca0",
    Path(
        SUPPORT.__file__
    ).resolve(): "31d5d9796c5ba10e7f211381c8b37a5fe2dbb3367ff2e829b18b3ebd4f94d0bc",
    SUPERVISOR: "4e4e19a35bb7b11a7c38f87f13c03e66cc987252e055ebfa3d2b163c84f214db",
}


def support_input(arrays):
    frames = np.asarray(arrays["frame_idx"])
    require(
        frames.ndim == 1
        and len(frames) > 0
        and frames.dtype.kind in "iu"
        and frames.min() >= 0
        and frames.max() <= np.iinfo(np.int64).max,
        "nonnegative int64-compatible native frames required",
    )
    frames = frames.astype(np.int64)
    require(np.all(np.diff(frames) == 1), "all native frames must be contiguous")
    pose, sources, observed = (np.asarray(arrays[k]) for k in ("pose_xyz", "sources", "observed"))
    require(
        pose.shape == (len(frames), 3, 3, 3) and pose.dtype.kind == "f",
        "floating three-node poses required",
    )
    require(
        sources.shape == pose.shape
        and sources.dtype.kind in "iu"
        and sources.min() >= -1
        and sources.max() <= np.iinfo(np.int64).max,
        "integer source rows >= -1 required",
    )
    require(
        observed.shape == (len(frames), 3, 3) and observed.dtype == np.bool_,
        "boolean observed node mask required",
    )
    finite, absent = np.isfinite(pose).all(axis=-1), np.isnan(pose).all(axis=-1)
    require(
        np.all(finite | (absent & ~observed & (sources < 0).all(axis=2))),
        "nonfinite node must be whole-NaN with no declared sources or observation",
    )
    return dict(
        frame_idx=frames, pose_xyz=pose, sources=sources.astype(np.int64), observed=observed
    )


def verify_support_partitioned(arrays, nodes, rig, *, block_frames=BLOCK, on_block=None):
    """Every frame exactly once; callback receives owned arrays, not input views."""
    require(type(block_frames) is int and block_frames > 0, "positive integer block size required")
    require(tuple(rig.views) == tuple(SUPPORT.base.VIEWS), "registered view order required")
    data = support_input(arrays)
    count = len(data["frame_idx"])
    totals = dict(
        native_frames=count,
        node_denominator=count * 9,
        pectoral_denominator=count * 3,
        maximum_view_node_pairs=count * 27,
        source_rows_checked=0,
        observed_nodes_checked=0,
        maximum_supported_ray_distance_mm=None,
        maximum_supported_pixel_error=None,
        block_frames=block_frames,
        blocks=0,
        zero_support_blocks=0,
        frame_range=[int(data["frame_idx"][0]), int(data["frame_idx"][-1]) + 1],
        ray_gate_mm=2.0,
        native_pixel_gate=12.0,
        all_sources_and_observed_flags_pass=True,
        temporal_subsampling=False,
    )
    for start in range(0, count, block_frames):
        stop = min(start + block_frames, count)
        part = {key: value[start:stop] for key, value in data.items()}
        if np.any(part["sources"] >= 0):
            try:
                checked, output = SUPPORT.verify_all_support(part, nodes, rig)
            except ValueError as error:
                raise ValueError(
                    f"support block [{part['frame_idx'][0]},{part['frame_idx'][-1] + 1}): {error}"
                ) from error
        else:
            # The old pure helper reduces max over declared sources and cannot
            # represent an empty set. No optical result exists in this branch.
            require(not part["observed"].any(), "empty source block has observed nodes")
            checked = dict(
                source_rows_checked=0,
                observed_nodes_checked=0,
                maximum_supported_ray_distance_mm=None,
                maximum_supported_pixel_error=None,
            )
            output = dict(
                frame_idx=part["frame_idx"],
                ray_distance_mm=np.full(part["sources"].shape, np.nan),
                native_pixel_error=np.full(part["sources"].shape, np.nan),
                valid_source=np.zeros(part["sources"].shape, bool),
            )
            totals["zero_support_blocks"] += 1
        totals["blocks"] += 1
        for key in ("source_rows_checked", "observed_nodes_checked"):
            totals[key] += checked[key]
        for key in ("maximum_supported_ray_distance_mm", "maximum_supported_pixel_error"):
            value = checked[key]
            if value is not None:
                totals[key] = value if totals[key] is None else max(value, totals[key])
        if on_block is not None:
            on_block(
                dict(
                    start_index=start,
                    stop_index=stop,
                    frame_range=[int(part["frame_idx"][0]), int(part["frame_idx"][-1]) + 1],
                    **checked,
                ),
                {key: value.copy() for key, value in output.items()},
            )
    return totals


def expected_phases():
    first = INITIAL_STOP
    result = []
    while first < FINAL_STOP:
        stop = min(first + 6 * BLOCK, FINAL_STOP)
        result.append((first, stop))
        first = stop
    return tuple(result)


def read_record(root, name, stage, evidence):
    root = Path(root).resolve(strict=True)
    require(root.is_relative_to(REPO), "artifact outside workspace")
    path = root / name
    record = COMMON.read_json(path)
    require(
        record.get("status") == "ok"
        and record.get("stage") == stage
        and record.get("finished_at")
        and record["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
        "completed successful guarded stage required: " + str(path),
    )
    evidence.add(path)
    for category in ("inputs", "outputs"):
        for item in record[category].values():
            target = Path(item["path"]).resolve(strict=True)
            require(
                target.is_relative_to(REPO)
                and (category != "outputs" or target.is_relative_to(root)),
                "foreign stage evidence/output",
            )
            evidence.add(target, item["sha256"])
    return record


def json_output(record, role, path):
    require(
        role in record["outputs"] and Path(record["outputs"][role]["path"]).resolve() == path,
        "required sealed output role differs: " + role,
    )
    return COMMON.read_json(path)


def input_binding(record, path, digest):
    items = [item for item in record["inputs"].values() if Path(item["path"]).resolve() == path]
    require(
        items and all(item["sha256"] == digest for item in items),
        "required lineage input is absent or changed",
    )


def output_binding(record, path, digest):
    items = [item for item in record["outputs"].values() if Path(item["path"]).resolve() == path]
    require(
        items and all(item["sha256"] == digest for item in items),
        "required nested output is absent or changed",
    )


def check_audit(audit, run, candidate_sha, *, stop, restores):
    for key, value in dict(
        schema="long_continuation_generic_phase_audit_v1",
        status="ok",
        run=str(run),
        frame_range=[FIRST, stop],
        candidate_sha256=candidate_sha,
        all_39_checkpoint_arrays_verified=True,
        logical_initialization_count=1,
        checkpoint_restore_count=restores,
        external_pose_reset_count=0,
        full_sequence_complete=stop == FINAL_STOP,
        prefix_backtrace_provisional=stop != FINAL_STOP,
        independently_redecoded_raw_images=0,
        independent_geometric_reprojection=False,
        independent_ID_truth=False,
    ).items():
        equal(audit[key], value, "phase audit differs: " + key)
    require(
        len(audit["all_21_candidate_arrays_exact_backtrace"]) == 21
        and audit["history"]["active_beam_count"] == 6
        and audit["history"]["parent_all_history_fields_exact"] is True
        and audit["history"]["all_suffix_parent_recurrences_exact"] is True,
        "phase lacks complete history/backtrace audit",
    )
    for item in audit["all_21_candidate_arrays_exact_backtrace"].values():
        require(
            set(item) == {"dtype", "shape", "sha256"}
            and isinstance(item["shape"], list)
            and isinstance(item["sha256"], str)
            and len(item["sha256"]) == 64,
            "missing exact-backtrace array digest",
        )


def supervisor_header(root, evidence):
    record = read_record(root, "supervisor.provenance.json", "long_remaining_supervisor", evidence)
    summary = json_output(record, "summary", root / "summary.json")
    policy = json_output(record, "plan", root / "supervisor_plan.json")
    require(
        summary["status"] == "ok"
        and summary["through"] == FINAL_STOP
        and summary["full_sequence_complete"] is True
        and summary["logical_initialization_count"] == 1
        and summary["external_pose_reset_count"] == 0
        and summary["independent_ID_truth"] is False,
        "full supervisor endpoint required",
    )
    equal(
        policy["phase_intervals"], [list(row) for row in expected_phases()], "phase ranges differ"
    )
    require(
        policy["initial_stop"] == INITIAL_STOP
        and policy["schema"] == "long_remaining_operational_supervisor_v1"
        and policy["target_interval"] == [FIRST, FINAL_STOP]
        and policy["detector_started"] is False
        and policy["new_external_pose_seed"] is False
        and len(summary["phases"]) == len(expected_phases()),
        "incomplete supervisor schedule",
    )
    for path, digest in policy["source_hashes"].items():
        evidence.add(path, digest)
        input_binding(record, Path(path).resolve(), digest)
    for path, digest in PINS.items():
        evidence.add(path, digest)
    return record, summary, policy


def load_supervisor_chain(root, evidence):
    """No candidate array is deserialized until every supervisor phase is verified."""
    root = Path(root).resolve(strict=True)
    supervisor, summary, policy = supervisor_header(root, evidence)
    binding = policy["fixed_bindings"]
    parent = Path(policy["initial_parent"]).resolve(strict=True)
    require(
        parent == REPO / "work/three_body/long_continuation_through66400_v1",
        "unexpected bootstrap parent",
    )
    previous_record = read_record(
        parent, "continuation.provenance.json", "three_body_long_continuation", evidence
    )
    equal(previous_record["params"]["bindings"], binding, "bootstrap binding differs")
    audit_root = root / "bootstrap"
    audit_seal = read_record(
        audit_root, "audit.provenance.json", "long_supervised_bootstrap_audit", evidence
    )
    audit = json_output(audit_seal, "summary", audit_root / "summary.json")
    for path in (audit_root / "audit.provenance.json", audit_root / "summary.json"):
        output_binding(supervisor, path, evidence.paths[str(path)])
    check_audit(
        audit,
        parent,
        previous_record["outputs"]["candidate"]["sha256"],
        stop=INITIAL_STOP,
        restores=3,
    )
    equal(
        audit["bootstrap_lineage"]["audit_seal_sha256"],
        phase.MINUTE_AUDIT_SHA,
        "bootstrap lost independent minute lineage",
    )
    evidence.add(phase.MINUTE_AUDIT / "audit.provenance.json", phase.MINUTE_AUDIT_SHA)
    input_binding(audit_seal, phase.MINUTE_AUDIT / "audit.provenance.json", phase.MINUTE_AUDIT_SHA)
    previous_audit_paths = (audit_root / "audit.provenance.json", audit_root / "summary.json")
    chain = []
    for index, ((start, stop), row) in enumerate(
        zip(expected_phases(), summary["phases"], strict=True)
    ):
        operation_root = root / "phases" / f"f{start:06d}-{stop:06d}"
        run = operation_root / "tracking"
        require(
            row == dict(parent=str(parent), run=str(run), through=stop), "phase order/gap differs"
        )
        registration_root = operation_root / "registration"
        registration = read_record(
            registration_root,
            "registration.provenance.json",
            "register_long_supervised_phase",
            evidence,
        )
        operation = read_record(
            operation_root, "operation.provenance.json", "long_supervised_phase_operation", evidence
        )
        child = read_record(
            run, "continuation.provenance.json", "three_body_long_continuation", evidence
        )
        require(
            operation["params"]["tracking_child_exit_code"] == 0,
            "nonzero child cannot enter completed chain",
        )
        execution = json_output(
            operation, "child_execution", operation_root / "child_execution.json"
        )
        require(
            execution["returncode"] == 0 and execution["child_alive_after_return"] is False,
            "child termination evidence incomplete",
        )
        require(
            datetime.fromisoformat(registration["finished_at"])
            <= datetime.fromisoformat(child["started_at"])
            <= datetime.fromisoformat(child["finished_at"])
            <= datetime.fromisoformat(operation["finished_at"]),
            "phase time order differs",
        )
        plan = json_output(registration, "plan", registration_root / "plan.json")
        require(
            plan["parent_phase"] == str(parent)
            and plan["expected_output"] == str(run)
            and plan["next_native_interval"] == [start, stop]
            and plan["expected_restores"] == index + 4,
            "registered phase linkage differs",
        )
        equal(plan["frozen_input_and_code_binding"], binding, "registered binding changed")
        equal(child["params"]["bindings"], binding, "child binding changed")
        require(
            Path(child["params"]["resume_phase"]).resolve() == parent
            and child["params"]["through_frame"] == stop,
            "child changed parent/endpoint",
        )
        equal(
            plan["parent_seal_sha256"],
            evidence.paths[str(parent / "continuation.provenance.json")],
            "registered parent seal changed",
        )
        config = json_output(registration, "execution", registration_root / "execution.json")
        equal(
            config["independent_parent_audit"],
            dict(
                root=str(previous_audit_paths[0].parent),
                seal_sha256=evidence.paths[str(previous_audit_paths[0])],
                summary_sha256=evidence.paths[str(previous_audit_paths[1])],
                parent_candidate_sha256=previous_record["outputs"]["candidate"]["sha256"],
            ),
            "registered independent parent audit changed",
        )
        for path in previous_audit_paths:
            input_binding(registration, path, evidence.paths[str(path)])
        input_binding(
            registration,
            parent / "continuation.provenance.json",
            evidence.paths[str(parent / "continuation.provenance.json")],
        )
        audit = json_output(operation, "audit", operation_root / "audit.json")
        input_binding(
            operation,
            run / "continuation.provenance.json",
            evidence.paths[str(run / "continuation.provenance.json")],
        )
        for relative in (
            "registration/registration.provenance.json",
            "registration/plan.json",
            "registration/execution.json",
            "operation.provenance.json",
            "audit.json",
            "child_execution.json",
        ):
            path = operation_root / relative
            output_binding(supervisor, path, evidence.paths[str(path)])
        check_audit(
            audit, run, child["outputs"]["candidate"]["sha256"], stop=stop, restores=index + 4
        )
        equal(audit["new_interval"], [start, stop], "audited interval differs")
        chain.append(
            dict(
                parent=str(parent),
                run=str(run),
                interval=[start, stop],
                restores=index + 4,
                candidate_sha256=audit["candidate_sha256"],
                operation_sha256=evidence.paths[str(operation_root / "operation.provenance.json")],
            )
        )
        parent, previous_record = run, child
        previous_audit_paths = (
            operation_root / "operation.provenance.json",
            operation_root / "audit.json",
        )
    require(summary["final_run"] == str(parent), "supervisor final run differs")
    return dict(
        supervisor=root,
        supervisor_record=supervisor,
        policy=policy,
        final_run=parent,
        final_record=previous_record,
        phases=chain,
        binding=binding,
        final_audit=audit,
    )


def load_completed_final(supervisor_root, evidence):
    """Load final arrays once after sealed chain checks; not a complete final audit."""
    chain = load_supervisor_chain(supervisor_root, evidence)
    root, record = chain["final_run"], chain["final_record"]
    summary = json_output(record, "summary", root / "summary.json")
    phase.validate_summary(summary, stop=FINAL_STOP, phase_start=345280, restores=32, chunks=180)
    require(
        Path(record["outputs"]["candidate"]["path"]).resolve() == root / "candidate.npz"
        and Path(record["outputs"]["checkpoint_manifest"]["path"]).resolve()
        == root / "checkpoint/manifest.json",
        "terminal output role differs",
    )
    state, metadata, manifest = phase.old.prior.read_checkpoint(
        root / "checkpoint", expected_bindings=chain["binding"]
    )
    require(
        len(state) == 39
        and metadata["processed_frames"] == FINAL_STOP - FIRST
        and metadata["previous_frame"] == FINAL_STOP - 1
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 32
        and state["state_pose"].shape == (6, 3, 3, 3),
        "terminal 39-state/lifecycle differs",
    )
    candidate = phase.old.read_npz(root / "candidate.npz")
    traced = COMMON.traced_candidate(state)
    require(
        set(candidate) == set(traced) and len(candidate) == 21, "terminal candidate schema differs"
    )
    exact = {key: COMMON.same_array(candidate[key], traced[key], key) for key in traced}
    COMMON.same_array(
        candidate["frame_idx"], np.arange(FIRST, FINAL_STOP, dtype=np.int64), "native frames"
    )
    support_input(candidate)
    require(np.isfinite(candidate["pose_xyz"]).all(), "persistent latent poses must all be finite")
    equal(
        exact,
        chain["final_audit"]["all_21_candidate_arrays_exact_backtrace"],
        "terminal exact-backtrace digests differ from phase audit",
    )
    counts = SUPPORT.direct_counts(candidate)
    equal(counts, chain["final_audit"]["counts"], "terminal audited availability differs")
    return dict(
        chain=chain,
        candidate=candidate,
        checkpoint=state,
        checkpoint_metadata=metadata,
        checkpoint_manifest=manifest,
        exact_backtrace=exact,
        counts=counts,
        final_audit_complete=False,
        remaining_checks=[
            "all-frame support replay",
            "fixed native samples/boundary flow",
            "global detector completion/assembly",
            "final provenance/report",
        ],
    )

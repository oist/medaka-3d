"""Bounded version/lineage fixtures; no real terminal payload, tracker or process."""

import copy
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import itertools

import long_final_mixed_chain as m
from test_long_final import Evidence, audit_data, entry, seal, write_json


def header():
    summary = dict(
        schema=m.NEW_STAGE,
        status="complete",
        through=m.FINAL_STOP,
        bootstrap_completed=True,
        full_sequence_complete=True,
        prefix_backtrace_provisional=False,
        logical_initialization_count=1,
        checkpoint_restore_count=32,
        external_pose_reset_count=0,
        active_beam_count=6,
        stop_request=None,
        incomplete_phase=None,
        independent_ID_truth=False,
        next_phase_retried=False,
        detector_started=False,
        old_process_stopped=False,
        phases=[{} for _ in range(28)],
        intended_initial_parent=str(m.handoff.CHILD),
    )
    policy = dict(
        schema=m.NEW_STAGE,
        initial_parent=str(m.handoff.CHILD),
        initial_stop=76360,
        target_interval=[49800, 348600],
        phase_intervals=[list(p) for p in m.expected_new_phases()],
        receipt_evidence_outside_child_binding=True,
        detector_started=False,
        old_process_stopped=False,
        new_external_pose_seed=False,
        failed_phase_retry=False,
        old_lock_removed=False,
        new_lock_removed=False,
        scheduling_changes_scientific_protocol=False,
    )
    return summary, policy


def test_fixed_schedule_samples_and_restore_indices():
    summary, policy = header()
    m.check_new_header(summary, policy)
    intervals = [
        (49800, 51460),
        (51460, 53120),
        (53120, 59760),
        (59760, 66400),
        (66400, 76360),
        *m.expected_new_phases(),
    ]
    assert len(intervals) == 33 and intervals[-1] == (345280, 348600)
    assert all(a[1] == b[0] for a, b in itertools.pairwise(intervals))
    assert sum(b - a for a, b in intervals) == 298800
    assert list(range(5, 5 + len(m.expected_new_phases()))) == list(range(5, 33))
    sample = m.native.fixed_sample_frames()
    assert sample.dtype == np.dtype("int64") and sample.size == 543
    expected = {49800, 49801, 49802, 348597, 348598, 348599}
    expected.update(x + d for x in range(51460, 348600, 1660) for d in (-1, 0, 1))
    assert sample.tolist() == sorted(expected) and sample.size * 3 == 1629


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "stopped_after_completed_phase"),
        ("full_sequence_complete", False),
        ("checkpoint_restore_count", 31),
        ("logical_initialization_count", 2),
        ("external_pose_reset_count", 1),
        ("active_beam_count", 5),
        ("stop_request", {}),
        ("independent_ID_truth", True),
        ("phases", [{}] * 27),
        ("intended_initial_parent", "foreign"),
    ],
)
def test_incomplete_or_different_header_rejected(key, value):
    summary, policy = header()
    summary[key] = value
    with pytest.raises(ValueError):
        m.check_new_header(summary, policy)


@pytest.mark.parametrize(
    "key,value",
    [
        ("initial_stop", 66400),
        ("failed_phase_retry", True),
        ("old_lock_removed", True),
        ("receipt_evidence_outside_child_binding", False),
        ("phase_intervals", [[76360, 348600]]),
    ],
)
def test_changed_policy_rejected(key, value):
    summary, policy = header()
    policy[key] = value
    with pytest.raises(ValueError):
        m.check_new_header(summary, policy)


@pytest.mark.parametrize(
    "status,stage", [("running", m.NEW_STAGE), ("failed", m.NEW_STAGE), ("ok", "invented_stage")]
)
def test_reject_before_terminal_arrays(tmp_path, monkeypatch, status, stage):
    write_json(tmp_path / "supervisor.provenance.json", dict(status=status, stage=stage))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("NPZ opened"))
    with pytest.raises(ValueError):
        m.load_completed_final(tmp_path, Evidence())


def test_legacy_dispatch_calls_frozen_entry_without_globals_substitution(tmp_path, monkeypatch):
    write_json(
        tmp_path / "supervisor.provenance.json",
        dict(status="ok", stage="long_remaining_supervisor"),
    )
    calls = []
    monkeypatch.setattr(
        m.legacy,
        "load_supervisor_chain",
        lambda root, evidence: calls.append(root) or {"phases": ["legacy"]},
    )
    result = m.load_supervisor_chain(tmp_path, Evidence())
    assert result == dict(phases=["legacy"], chain_version="legacy_v1")
    assert calls == [tmp_path]


def set_times(path, start, stop):
    data = json.loads(path.read_text())
    data.update(
        started_at=f"2026-09-09T00:{start:02d}:00+00:00",
        finished_at=f"2026-09-09T00:{stop:02d}:00+00:00",
    )
    write_json(path, data)
    return path


@pytest.fixture
def edge(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m.legacy, "REPO", tmp_path)
    root, parent, prior = tmp_path / "op", tmp_path / "parent", tmp_path / "prior"
    binding = {"unchanged": "scientific core"}
    source = write_json(tmp_path / "wrapper.py", {"tiny": "executed code"})
    code = {str(source): entry(source)["sha256"]}
    candidate = write_json(parent / "candidate.npz", {"never": "deserialize"})
    parent_seal = seal(
        parent,
        "continuation.provenance.json",
        "three_body_long_continuation",
        dict(candidate=candidate),
        params=dict(bindings=binding),
    )
    previous = write_json(prior / "summary.json", {"old": "audit"})
    prior_seal = seal(prior, "audit.provenance.json", m.handoff.STAGE, dict(summary=previous))
    reg, run = root / "registration", root / "tracking"
    plan = dict(
        parent_phase=str(parent),
        expected_output=str(run),
        next_native_interval=[76360, 86320],
        expected_restores=5,
        expected_initializations=1,
        expected_external_resets=0,
        parent_active_beam=6,
        frozen_input_and_code_binding=binding,
        parent_seal_sha256=entry(parent_seal)["sha256"],
    )
    planpath = write_json(reg / "plan.json", plan)
    lineage = dict(
        root=str(prior),
        stage=m.handoff.STAGE,
        seal_sha256=entry(prior_seal)["sha256"],
        summary_sha256=entry(previous)["sha256"],
        parent_candidate_sha256=entry(candidate)["sha256"],
    )
    command = ["frozen runner", "--resume", str(parent), "--through-frame", "86320"]
    monkeypatch.setattr(m.current.operation, "command_for", lambda *args: command.copy())
    config = write_json(
        reg / "execution.json",
        dict(independent_parent_audit=lineage, command=command, source_hashes=code),
    )
    snapshot = write_json(reg / "code_snapshot/wrapper.py", {"tiny": "executed code"})
    prep = write_json(reg / "mask_validation_receipts.json", {"tested": "separately"})
    reg_seal = seal(
        reg,
        "registration.provenance.json",
        m.operation.REGISTRATION_STAGE,
        dict(plan=planpath, execution=config, mask_validation_receipts=prep, snapshot=snapshot),
        inputs=[prior_seal, previous, parent_seal, source],
    )
    set_times(reg_seal, 1, 2)
    fitted = write_json(run / "candidate.npz", {"never": "deserialize either"})
    child = seal(
        run,
        "continuation.provenance.json",
        "three_body_long_continuation",
        dict(candidate=fitted),
        params=dict(bindings=binding, resume_phase=str(parent), through_frame=86320),
    )
    set_times(child, 3, 4)
    audit = write_json(
        root / "audit.json", audit_data(run, entry(fitted)["sha256"], 76360, 86320, 5)
    )
    executed = write_json(
        root / "child_execution.json",
        dict(
            returncode=0,
            child_alive_after_return=False,
            started_at="2026-09-09T00:03:00+00:00",
            finished_at="2026-09-09T00:05:00+00:00",
        ),
    )
    verify = write_json(root / "mask_validation_receipts.json", {"tested": "separately"})
    log = write_json(root / "tracking.log", {"log": True})
    op_seal = seal(
        root,
        "operation.provenance.json",
        m.operation.OPERATION_STAGE,
        dict(audit=audit, child_execution=executed, mask_validation_receipts=verify, log=log),
        inputs=[child, prior_seal, previous, parent_seal, source],
        params=dict(tracking_child_exit_code=0),
    )
    set_times(op_seal, 2, 6)
    supervisor = dict(outputs={str(root / p): entry(root / p) for p in m.current.PHASE_ARTIFACTS})
    receipt_calls = []
    monkeypatch.setattr(m, "verify_receipt_usage", lambda *a, **k: receipt_calls.append(k["label"]))

    def run_check():
        evidence = Evidence()
        for p in (parent_seal, prior_seal, previous):
            evidence.add(p)
        return m.check_phase_edge(
            root,
            parent,
            json.loads(parent_seal.read_text()),
            prior,
            m.handoff.STAGE,
            "audit.provenance.json",
            binding,
            76360,
            86320,
            5,
            supervisor,
            evidence,
            receipted=True,
            code=code,
            cache_root=tmp_path / "cache",
        )

    return dict(
        root=root,
        parent=parent,
        check=run_check,
        supervisor=supervisor,
        calls=receipt_calls,
        registration=reg_seal,
        operation=op_seal,
        child=child,
        config=config,
        plan=planpath,
        audit=audit,
        execution=executed,
        source=source,
        snapshot=snapshot,
    )


def refresh_edge(f):
    # Checksums follow intentionally changed fixture content so semantic guards are exercised.
    for name in ("registration", "operation"):
        path = f[name]
        data = json.loads(path.read_text())
        for category in ("inputs", "outputs"):
            for item in data[category].values():
                item["sha256"] = entry(Path(item["path"]))["sha256"]
        write_json(path, data)
    f["supervisor"]["outputs"] = {
        str(f["root"] / p): entry(f["root"] / p) for p in m.current.PHASE_ARTIFACTS
    }


def test_real_sealed_receipted_edge_preserves_stage_binding_and_history_declarations(edge):
    row, child, audit = edge["check"]()
    assert row["restores"] == 5 and row["interval"] == [76360, 86320]
    assert row["operation_stage"] == m.operation.OPERATION_STAGE
    assert child["params"]["bindings"] == {"unchanged": "scientific core"}
    assert len(audit["all_21_candidate_arrays_exact_backtrace"]) == 21
    assert edge["calls"] == ["prepare", "verify"]


@pytest.mark.parametrize(
    "fault",
    [
        "exit",
        "time",
        "parent",
        "restore",
        "binding",
        "lineage_stage",
        "lineage_candidate",
        "command",
        "history",
        "guard",
        "input_membership",
        "snapshot",
    ],
)
def test_real_edge_corruption_is_rejected_after_resealing(edge, fault):
    path = edge[
        dict(
            exit="execution",
            time="registration",
            parent="child",
            restore="plan",
            binding="child",
            lineage_stage="config",
            lineage_candidate="config",
            command="config",
            history="audit",
            guard="operation",
            input_membership="operation",
            snapshot="snapshot",
        )[fault]
    ]
    data = json.loads(path.read_text())
    if fault == "exit":
        data["returncode"] = 1
    elif fault == "time":
        data["finished_at"] = "2026-09-09T00:04:00+00:00"
    elif fault == "parent":
        data["params"]["resume_phase"] = str(edge["root"])
    elif fault == "restore":
        data["expected_restores"] = 4
    elif fault == "binding":
        data["params"]["bindings"] = {"wrapper": "forged as science"}
    elif fault == "lineage_stage":
        data["independent_parent_audit"]["stage"] = "unknown_stage"
    elif fault == "lineage_candidate":
        data["independent_parent_audit"]["parent_candidate_sha256"] = "0" * 64
    elif fault == "command":
        data["command"][-1] = "86321"
    elif fault == "history":
        data["history"]["parent_all_history_fields_exact"] = False
    elif fault == "guard":
        data["params"]["all_input_source_snapshot_output_sha_unchanged"] = False
    elif fault == "input_membership":
        del data["inputs"][str(edge["source"])]
    elif fault == "snapshot":
        data = {"different": "snapshot bytes"}
    write_json(path, data)
    refresh_edge(edge)
    with pytest.raises(ValueError):
        edge["check"]()


@pytest.fixture
def receipt_usage(tmp_path, monkeypatch):
    root, cache = tmp_path / "stage", tmp_path / "cache"
    attempt = tmp_path / "detector/chunks/a"
    manifest = write_json(attempt / "manifest.json", {"tiny": "manifest"})
    sealfile = write_json(attempt / "staging.provenance.json", {"tiny": "seal"})
    row = dict(
        start=49800,
        stop=51460,
        attempt_path="chunks/a",
        manifest_sha256=entry(manifest)["sha256"],
        provenance_sha256=entry(sealfile)["sha256"],
        output_hashes={},
    )
    binding = dict(
        inputs=dict(input_stage_root=str(tmp_path / "detector"), detector_binding={"fixed": "D2"})
    )
    evidence = Evidence()
    sourcefiles = [
        Path(x).resolve()
        for x in (
            m.receipts.__file__,
            m.receipts.chunk_masks.__file__,
            m.receipts.prepare_sequence.__file__,
        )
    ]
    for p in sourcefiles:
        evidence.add(p)
    validator = dict(
        source_sha256={str(p): entry(p)["sha256"] for p in sourcefiles},
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        numpy=np.__version__,
        schema_dtypes={k: v.str for k, v in m.native.SCHEMA_DTYPES.items()},
        mask_bitorder="little",
    )
    files = validator["source_sha256"].copy()
    uses = []
    for view in m.native.VIEWS:
        mask = write_json(attempt / (view + ".npz"), {"unopened": "packed fixture"})
        row["output_hashes"][view + "_masks"] = entry(mask)
        scope = dict(
            frame_range=[49800, 51460],
            view=view,
            image_shape_hw=[1200, 1200],
            manifest_path=str(manifest),
            manifest_sha256=entry(manifest)["sha256"],
            seal_path=str(sealfile),
            seal_sha256=entry(sealfile)["sha256"],
            detector_binding=binding["inputs"]["detector_binding"],
            expected_npz_sha256=entry(mask)["sha256"],
        )
        request = dict(
            schema=m.receipts.SCHEMA,
            npz_path=str(mask),
            npz_bytes=mask.stat().st_size,
            scope=scope,
            validator_binding=validator,
        )
        key = hashlib.sha256(m.receipts.canonical(request).encode()).hexdigest()
        body = write_json(
            cache / (view + ".json"),
            dict(
                request=request,
                result=dict(
                    status="ok",
                    validator="chunk_masks.validate_packed_masks",
                    frame_count=1660,
                    validated_fields=sorted(m.native.SCHEMA_DTYPES),
                    every_decoded_frame_sha_verified=True,
                ),
            ),
        )
        commit = write_json(
            cache / (view + ".commit.json"),
            dict(
                schema=m.receipts.SCHEMA,
                status="ok",
                receipt=body.name,
                sha256=entry(body)["sha256"],
            ),
        )
        for p in (mask, manifest, sealfile, body, commit):
            files[str(p)] = entry(p)["sha256"]
        uses.append(
            dict(
                key=key,
                cache_hit=True,
                receipt_path=str(body),
                receipt_sha256=entry(body)["sha256"],
                commit_path=str(commit),
                commit_sha256=entry(commit)["sha256"],
                npz_path=str(mask),
                scope=scope,
            )
        )
    usage = dict(
        schema="long_phase_mask_validation_usage_v1",
        phase="verify",
        verification_returned=True,
        scientific_plan_or_child_binding_changed=False,
        cache_root=str(cache),
        evidence_hashes=files,
        uses=uses,
    )
    path = write_json(root / "mask_validation_receipts.json", usage)
    record = dict(
        inputs={p: dict(path=p, sha256=h) for p, h in files.items()},
        outputs=dict(mask_validation_receipts=entry(path)),
    )
    plan = dict(full_selected_chunk_records=[row], frozen_input_and_code_binding=binding)

    def run_check():
        e = Evidence()
        for p in sourcefiles:
            e.add(p)
        e.add(path)
        return m.verify_receipt_usage(
            record, root, label="verify", plan=plan, evidence=e, cache_root=cache
        )

    return dict(path=path, record=record, usage=usage, check=run_check, plan=plan)


def test_recorded_receipts_are_checked_without_loading_or_creating_masks(
    receipt_usage, monkeypatch
):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("semantic payload opened"))
    monkeypatch.setattr(m.receipts, "ReceiptStore", lambda *a, **k: pytest.fail("store created"))
    result = receipt_usage["check"]()
    assert result["uses"] == result["hit_count"] == 3


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_view",
        "missing_use",
        "scope",
        "mask_sha",
        "request_key",
        "commit",
        "validator",
        "unsealed_source",
        "result",
    ],
)
def test_receipt_contract_corruption_rejected(receipt_usage, fault):
    f = receipt_usage
    usage = copy.deepcopy(f["usage"])
    use = usage["uses"][0]
    if fault == "duplicate_view":
        usage["uses"][1] = copy.deepcopy(use)
    elif fault == "missing_use":
        usage["uses"].pop()
    elif fault == "scope":
        use["scope"]["image_shape_hw"] = [12, 12]
    elif fault == "mask_sha":
        use["scope"]["expected_npz_sha256"] = "0" * 64
    elif fault == "request_key":
        use["key"] = "0" * 64
    elif fault == "unsealed_source":
        source = next(iter(usage["evidence_hashes"]))
        del f["record"]["inputs"][source]
    else:
        path = Path(use["commit_path"] if fault == "commit" else use["receipt_path"])
        data = json.loads(path.read_text())
        if fault == "commit":
            data["status"] = "failed"
        elif fault == "validator":
            data["request"]["validator_binding"]["numpy"] = "foreign"
        elif fault == "result":
            data["result"]["every_decoded_frame_sha_verified"] = False
        write_json(path, data)
        digest = entry(path)["sha256"]
        use["commit_sha256" if fault == "commit" else "receipt_sha256"] = digest
        usage["evidence_hashes"][str(path)] = digest
        f["record"]["inputs"][str(path)]["sha256"] = digest
        if fault != "commit":
            commit = Path(use["commit_path"])
            c = json.loads(commit.read_text())
            c["sha256"] = digest
            write_json(commit, c)
            use["commit_sha256"] = entry(commit)["sha256"]
            usage["evidence_hashes"][str(commit)] = use["commit_sha256"]
            f["record"]["inputs"][str(commit)]["sha256"] = use["commit_sha256"]
    write_json(f["path"], usage)
    f["record"]["outputs"]["mask_validation_receipts"] = entry(f["path"])
    with pytest.raises(ValueError):
        f["check"]()


def ancestor_summary(start, stop, restores):
    return dict(
        schema="long_continuation_phase_v1",
        technical_completion=True,
        full_sequence_complete=False,
        prefix_backtrace_provisional=True,
        registered_interval=[49800, 348600],
        processed_interval=[49800, stop],
        processed_total=stop - 49800,
        phase_frames=[start, stop],
        phase_processed=stop - start,
        consumed_chunk_count=(stop - 49800) // 1660,
        logical_initialization_count=1,
        checkpoint_restore_count=restores,
        external_pose_reset_count=0,
        temporal_subsampling=False,
        human_annotation_input=False,
        post_initialization_baseline_coordinates_input=False,
    )


@pytest.fixture
def ancestors(tmp_path, monkeypatch):
    monkeypatch.setattr(m.legacy, "REPO", tmp_path)
    roots = tuple((tmp_path / f"a{i}", stop) for i, stop in enumerate((51460, 53120, 59760, 66400)))
    monkeypatch.setattr(m, "ANCESTORS", roots)
    previous = None
    start = 49800
    for i, (root, stop) in enumerate(roots):
        summary = write_json(root / "summary.json", ancestor_summary(start, stop, i))
        outputs = {"summary": summary}
        for name in m.phase.old.prior.PARENT_FILES:
            if name not in ("continuation.provenance.json", "summary.json"):
                outputs[name] = write_json(root / name, {"unopened": name})
        inputs = (
            [] if previous is None else [previous / name for name in m.phase.old.prior.PARENT_FILES]
        )
        seal(
            root,
            "continuation.provenance.json",
            "three_body_long_continuation",
            outputs,
            params=dict(
                bindings={"fixed": True},
                through_frame=stop,
                resume_phase=None if previous is None else str(previous),
            ),
            inputs=inputs,
        )
        previous, start = root, stop
    return roots


def test_all_four_seed_ancestors_have_exact_native_and_checkpoint_parent_membership(ancestors):
    rows = m.verify_ancestors({"fixed": True}, Evidence())
    assert [r["restores"] for r in rows] == [0, 1, 2, 3]
    assert [r["interval"] for r in rows] == [
        [49800, 51460],
        [51460, 53120],
        [53120, 59760],
        [59760, 66400],
    ]


@pytest.mark.parametrize(
    "fault", ["seed_resume", "second_parent", "omitted_payload", "reset", "binding"]
)
def test_ancestor_restart_or_unbound_history_refused(ancestors, fault):
    index = 0 if fault == "seed_resume" else 3
    root = ancestors[index][0]
    path = root / "continuation.provenance.json"
    data = json.loads(path.read_text())
    if fault == "seed_resume":
        data["params"]["resume_phase"] = str(root)
    elif fault == "second_parent":
        data["params"]["resume_phase"] = str(ancestors[1][0])
    elif fault == "omitted_payload":
        data["inputs"].pop(next(iter(data["inputs"])))
    elif fault == "binding":
        data["params"]["bindings"] = {"fixed": False}
    elif fault == "reset":
        summary = root / "summary.json"
        value = json.loads(summary.read_text())
        value["external_pose_reset_count"] = 1
        write_json(summary, value)
        data["outputs"]["summary"] = entry(summary)
    write_json(path, data)
    with pytest.raises(ValueError):
        m.verify_ancestors({"fixed": True}, Evidence())


@pytest.fixture
def mixed_schedule(tmp_path, monkeypatch):
    """All schedule rows are files; separate tests cover sealed edges and receipts."""
    root = tmp_path / "new"
    oldroot = tmp_path / "old"
    child = oldroot / "phases/f066400-076360/tracking"
    parent = tmp_path / "ancestor66400"
    witness = tmp_path / "witness"
    witness.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m.legacy, "REPO", tmp_path)
    monkeypatch.setattr(m, "PINS", {})
    monkeypatch.setattr(m.handoff, "CHILD", child)
    monkeypatch.setattr(m.handoff, "OLD_PARENT", parent)
    binding = {"fixed": "science"}
    candidate = write_json(parent / "candidate.npz", {"not": "decoded"})
    seal(
        parent,
        "continuation.provenance.json",
        "three_body_long_continuation",
        dict(candidate=candidate),
        params=dict(bindings=binding),
    )
    minute = tmp_path / "minute"
    minuteseal = write_json(minute / "audit.provenance.json", {"pinned": "old audit"})
    monkeypatch.setattr(m.phase, "MINUTE_AUDIT", minute)
    monkeypatch.setattr(m.phase, "MINUTE_AUDIT_SHA", entry(minuteseal)["sha256"])
    audit = audit_data(parent, entry(candidate)["sha256"], 59760, 66400, 3)
    audit["bootstrap_lineage"] = {"audit_seal_sha256": entry(minuteseal)["sha256"]}
    previous = write_json(oldroot / "bootstrap/summary.json", audit)
    seal(
        oldroot / "bootstrap",
        "audit.provenance.json",
        "long_supervised_bootstrap_audit",
        dict(summary=previous),
        inputs=[minuteseal],
    )
    write_json(
        child.parent / "operation.provenance.json", {"finished_at": "2026-09-09T00:04:00+00:00"}
    )
    lineage = {
        "witness_root": str(witness),
        "owner_exit": {"exit_time_utc": "2026-09-09T00:05:00+00:00"},
    }
    transferred = dict(
        parent=child, old_root=oldroot, old_plan={}, binding=binding, lineage=lineage
    )
    monkeypatch.setattr(m.handoff, "verify_handoff", lambda *args: copy.deepcopy(transferred))
    monkeypatch.setattr(m, "verify_ancestors", lambda *args: [{"restores": i} for i in range(4)])
    source_calls = []
    receipt_calls = []
    edge_calls = []
    monkeypatch.setattr(m, "bound_sources", lambda *a: source_calls.append(a[1]))
    monkeypatch.setattr(m, "verify_receipt_usage", lambda *a, **k: receipt_calls.append(k["label"]))
    oldresult = audit_data(child, "a" * 64, 66400, 76360, 4)

    def edgecheck(
        op_root,
        p,
        previous_record,
        audit_root,
        audit_stage,
        audit_name,
        b,
        start,
        stop,
        restores,
        s,
        e,
        **kw,
    ):
        edge_calls.append(
            dict(
                parent=p,
                start=start,
                stop=stop,
                restores=restores,
                stage=audit_stage,
                receipted=kw["receipted"],
            )
        )
        current_run = op_root / "tracking"
        result = (
            oldresult if restores == 4 else audit_data(current_run, "b" * 64, start, stop, restores)
        )
        return (
            dict(parent=str(p), run=str(current_run), interval=[start, stop], restores=restores),
            {"outputs": {"candidate": {"sha256": "b" * 64}}},
            copy.deepcopy(result),
        )

    monkeypatch.setattr(m, "check_phase_edge", edgecheck)
    bootstrap = root / "bootstrap"
    summaryfile = write_json(bootstrap / "summary.json", dict(oldresult, handoff_lineage=lineage))
    handofffile = write_json(bootstrap / "handoff.json", lineage)
    receiptfile = write_json(bootstrap / "mask_validation_receipts.json", {})
    bootstrapseal = seal(
        bootstrap,
        "audit.provenance.json",
        m.handoff.STAGE,
        dict(summary=summaryfile, handoff=handofffile, mask_validation_receipts=receiptfile),
    )
    set_times(bootstrapseal, 6, 7)
    summary, policy = header()
    summary["phases"] = []
    p, pa = child, bootstrap
    for i, (start, stop) in enumerate(m.expected_new_phases()):
        op = root / "phases" / f"f{start:06d}-{stop:06d}"
        run = op / "tracking"
        summary["phases"].append(
            dict(
                parent=str(p),
                run=str(run),
                through=stop,
                parent_audit=str(pa),
                operation=str(op),
                checkpoint_restore_count=i + 5,
            )
        )
        p, pa = run, op
    summary["final_run"] = str(p)
    policy.update(
        source_hashes={"fixture": "code"},
        fixed_bindings=binding,
        handoff_lineage=lineage,
        receipt_cache_root=str(cache),
    )
    summarypath = write_json(root / "summary.json", summary)
    policypath = write_json(root / "supervisor_plan.json", policy)
    supervisor = seal(
        root,
        "supervisor.provenance.json",
        m.NEW_STAGE,
        dict(
            summary=summarypath,
            plan=policypath,
            **{str(p): p for p in (summaryfile, handofffile, receiptfile, bootstrapseal)},
        ),
    )
    set_times(supervisor, 6, 59)
    return dict(
        root=root,
        summary=summarypath,
        policy=policypath,
        edge_calls=edge_calls,
        receipt_calls=receipt_calls,
        source_calls=source_calls,
        supervisor=supervisor,
        transferred=transferred,
    )


def test_full_mixed_schedule_delegates_29_edges_in_correct_order_without_arrays(
    mixed_schedule, monkeypatch
):
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("array opened during chain verification")
    )
    f = mixed_schedule
    result = m.load_supervisor_chain(f["root"], Evidence())
    assert result["chain_version"] == "legacy_then_receipted_v1"
    assert result["tracking_processes"] == 33 and result["checkpoint_restores"] == 32
    assert len(result["ancestors"]) == 4 and len(result["phases"]) == 29
    assert [r["restores"] for r in f["edge_calls"]] == list(range(4, 33))
    assert [r["receipted"] for r in f["edge_calls"]] == [False] + [True] * 28
    assert f["edge_calls"][1]["stage"] == m.handoff.STAGE
    assert all(r["stage"] == m.operation.OPERATION_STAGE for r in f["edge_calls"][2:])
    assert f["receipt_calls"] == ["handoff_verify"]
    assert result["sampled_frames"].tobytes() == m.native.fixed_sample_frames().tobytes()
    assert result["final_audit_complete"] is False


@pytest.mark.parametrize(
    "fault", ["parent", "order", "restores", "foreign_audit", "final", "binding", "owner_time"]
)
def test_mixed_orchestration_rejects_wrong_lineage(mixed_schedule, fault):
    f = mixed_schedule
    path = f["policy"] if fault in ("binding", "owner_time") else f["summary"]
    data = json.loads(path.read_text())
    if fault == "parent":
        data["phases"][0]["parent"] = str(f["root"])
    elif fault == "order":
        data["phases"][3], data["phases"][4] = data["phases"][4], data["phases"][3]
    elif fault == "restores":
        data["phases"][27]["checkpoint_restore_count"] = 31
    elif fault == "foreign_audit":
        data["phases"][1]["parent_audit"] = str(f["root"])
    elif fault == "final":
        data["final_run"] = data["phases"][-2]["run"]
    elif fault == "binding":
        data["fixed_bindings"] = {"new": "science"}
    elif fault == "owner_time":
        data["handoff_lineage"]["owner_exit"]["exit_time_utc"] = "2026-09-09T00:07:00+00:00"
        f["transferred"]["lineage"] = copy.deepcopy(data["handoff_lineage"])
    write_json(path, data)
    sealdata = json.loads(f["supervisor"].read_text())
    role = "plan" if path == f["policy"] else "summary"
    sealdata["outputs"][role] = entry(path)
    write_json(f["supervisor"], sealdata)
    with pytest.raises(ValueError):
        m.load_supervisor_chain(f["root"], Evidence())


@pytest.mark.parametrize("entrypoint", ["load_completed_final", "load_native_sample_inputs"])
def test_terminal_entrypoints_use_explicit_adapter_before_reading_payload(
    monkeypatch, tmp_path, entrypoint
):
    calls = []

    def refuse(*args):
        calls.append(args[0])
        raise ValueError("mixed chain refused")

    monkeypatch.setattr(m, "load_supervisor_chain", refuse)
    monkeypatch.setattr(
        m.legacy, "load_supervisor_chain", lambda *a: pytest.fail("legacy-only dispatch")
    )
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("NPZ opened before successful chain")
    )
    with pytest.raises(ValueError, match="mixed chain refused"):
        getattr(m, entrypoint)(tmp_path, Evidence())
    assert calls == [tmp_path]

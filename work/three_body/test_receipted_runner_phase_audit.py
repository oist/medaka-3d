"""Synthetic CPU checks only; no real completed child, GPU, fit or raw payload."""

import copy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import receipted_runner_phase_audit as a
import test_run_receipted_long_continuation as runner_fixture
from test_receipted_long_runner_migration import bindings, synthetic_state


def stamp(seconds):
    return (datetime(2020, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)).isoformat()


def owned_fixture(tmp_path):
    run, reg = tmp_path / "run", tmp_path / "registration"
    digest = "a" * 64
    info = dict(
        schema=a.EXECUTION_SCHEMA,
        pid=123,
        output=str(run),
        command=a.expected_command(run, reg, digest),
        registration_sha256=digest,
        returncode=0,
        child_alive_after_return=False,
        started_at=stamp(2),
        finished_at=stamp(6),
    )
    registration, producer = (
        {"finished_at": stamp(1)},
        {"started_at": stamp(3), "finished_at": stamp(5)},
    )
    return info, run, reg, digest, registration, producer


def test_owned_exit_facts_and_post_popen_recording_delay(tmp_path):
    values = owned_fixture(tmp_path)
    result = a.verify_owned_execution(*values)
    assert result["returncode"] == 0 and "not an OS re-query" in result["scope"]
    values[0]["started_at"] = stamp(4)  # Child may write provenance before owner is scheduled.
    a.verify_owned_execution(*values)


@pytest.mark.parametrize(
    "change",
    [
        "command",
        "output",
        "sha",
        "pid_bool",
        "pid_zero",
        "exit_bool",
        "exit_nonzero",
        "alive",
        "unknown_schema",
        "no_zone",
        "owner_early",
        "owner_after_child",
        "child_early",
        "exit_early",
    ],
)
def test_owned_exit_rejects_unobserved_or_mismatched_success(tmp_path, change):
    values = owned_fixture(tmp_path)
    info, *_, producer = values
    if change == "command":
        info["command"][-1] = str(tmp_path / "another")
    elif change == "output":
        info["output"] += "x"
    elif change == "sha":
        info["registration_sha256"] = "b" * 64
    elif change == "pid_bool":
        info["pid"] = True
    elif change == "pid_zero":
        info["pid"] = 0
    elif change == "exit_bool":
        info["returncode"] = False
    elif change == "exit_nonzero":
        info["returncode"] = 1
    elif change == "alive":
        info["child_alive_after_return"] = True
    elif change == "unknown_schema":
        info["schema"] = "unknown"
    elif change == "no_zone":
        info["finished_at"] = "2020-01-01T00:00:06"
    elif change == "owner_early":
        info["started_at"] = stamp(0)
    elif change == "owner_after_child":
        info["started_at"] = stamp(7)
    elif change == "child_early":
        producer["started_at"] = stamp(0)
    elif change == "exit_early":
        info["finished_at"] = stamp(4)
    with pytest.raises(ValueError):
        a.verify_owned_execution(*values)


def test_registered_input_union_is_exact_and_rejects_conflicts(monkeypatch, tmp_path):
    monkeypatch.setattr(a.m, "REPO", tmp_path)
    reg = tmp_path / "reg"
    reg.mkdir()
    p, q = tmp_path / "input", reg / "plan.json"
    p.write_text("fixed")
    q.write_text("plan")
    seal_path = reg / "registration.provenance.json"
    seal_path.write_text("seal")
    def item(path):
        return dict(path=str(path), sha256=a.m.sha(path))
    record = dict(inputs={str(p): item(p)}, outputs={"plan": item(q)})
    expected = a.child_input_hashes(record, reg, a.m.sha(seal_path))
    assert set(expected) == {str(p), str(q), str(seal_path)}
    producer = dict(inputs={k: dict(path=k, sha256=v) for k, v in expected.items()})
    a.exact_child_inputs(producer, expected, a.m.Evidence())
    producer["inputs"][str(tmp_path / "unknown")] = item(p)
    with pytest.raises(ValueError, match="membership"):
        a.exact_child_inputs(producer, expected, a.m.Evidence())
    del producer["inputs"][str(tmp_path / "unknown")]
    producer["inputs"].pop(str(p))
    with pytest.raises(ValueError, match="membership"):
        a.exact_child_inputs(producer, expected, a.m.Evidence())
    record["outputs"]["conflict"] = dict(path=str(p), sha256="b" * 64)
    with pytest.raises(ValueError, match="conflicting"):
        a.child_input_hashes(record, reg, a.m.sha(seal_path))


def plan_fixture(monkeypatch, tmp_path):
    legacy, code = bindings(monkeypatch, tmp_path)
    current = a.m.execution_binding(legacy, code)
    start, stop = 136120, 146080
    plan = dict(
        schema="receipted_long_child_plan_v1",
        stage=a.m.STAGE,
        binding=current,
        parent_binding=legacy,
        transition=a.m.transition(legacy, current, legacy),
        next_native_interval=[start, stop],
        full_interval=[a.m.FIRST, a.m.FULL_STOP],
        output_interval=[a.m.FIRST, stop],
        expected_initializations=1,
        expected_external_resets=0,
        expected_previous_frame=start - 1,
        native_subsampling=False,
        serial_flow=True,
        same_original_accelerated_class=True,
        science_branch_change=False,
        processes_started=False,
        exclusive_execution_owner_is_external=True,
    )
    return plan, legacy


def test_exact_legacy_edge_followed_by_exact_new_edge(monkeypatch, tmp_path):
    plan, legacy = plan_fixture(monkeypatch, tmp_path)
    before = copy.deepcopy(plan)
    assert a.registered_edge(plan, legacy, legacy)["kind"] == "legacy_to_receipted"
    assert plan == before
    plan["parent_binding"] = plan["binding"]
    plan["transition"] = a.m.transition(plan["binding"], plan["binding"], legacy)
    assert a.registered_edge(plan, legacy, plan["binding"])["kind"] == "receipted_to_receipted"
    changed = copy.deepcopy(plan["binding"])
    changed["torch_cpu_threads"] = 2
    with pytest.raises(ValueError):
        a.registered_edge(plan, legacy, changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("serial_flow", False),
        ("science_branch_change", True),
        ("expected_initializations", True),
        ("next_native_interval", [136120, 146081]),
    ],
)
def test_registered_edge_rejects_science_or_lifecycle_change(monkeypatch, tmp_path, field, value):
    plan, legacy = plan_fixture(monkeypatch, tmp_path)
    plan[field] = value
    with pytest.raises(ValueError):
        a.registered_edge(plan, legacy, legacy)


def completed_fixture(monkeypatch, tmp_path):
    state = runner_fixture.fixture(monkeypatch, tmp_path)
    first = a.m.FIRST
    # Two frames stand in for a phase; all actual state/codec/math uses frozen code.
    monkeypatch.setattr(a.m, "CHUNK", 1)
    monkeypatch.setattr(
        a.m,
        "native_phase",
        lambda start, stop: a.require((start, stop) == (first + 3, first + 5), "synthetic range"),
    )
    records = [dict(start=first + i, stop=first + i + 1) for i in range(5)]
    state.plan["selected_chunk_records"] = records
    state.context.inputs.chunk_records = records
    args = runner_fixture.register_fixture(state, tmp_path)
    state.context.inputs.chunk_records = records
    summary = a.child.execute(args)
    progress = args.output / "progress.jsonl"
    progress.write_bytes(
        b"".join(
            a.m._canonical(dict(processed=i + 1, source_frame=first + i, phase_processed=i - 2))
            + b"\n"
            for i in (3, 4)
        )
    )
    path = args.output / "continuation.provenance.json"
    seal = a.m.read(path)
    seal["outputs"]["progress"].update(sha256=a.m.sha(progress), size=progress.stat().st_size)
    # This file-backed synthetic producer is deliberately rewritten inside tmp_path only.
    path.write_bytes(a.m._canonical(seal))
    return state, args, seal, summary


def load_fixture(state, args, seal):
    return a.load_completed_phase(
        args.output,
        seal,
        state.plan["binding"],
        state.context.inputs,
        state.context.protocol,
        a.m.Evidence(),
        parent_metadata=state.context.parent["metadata"],
    )


def test_completed_new_loader_real39_semantic21_and_independent_suffix(monkeypatch, tmp_path):
    state, args, seal, summary = completed_fixture(monkeypatch, tmp_path)
    after = load_fixture(state, args, seal)
    assert len(after["arrays"]) == 39 and len(after["comparisons"]) == 21
    assert all(set(v) == {"dtype", "shape", "sha256"} for v in after["comparisons"].values())
    before = state.context.parent
    result = a.history.suffix_history(before, after)
    assert result["parent_history_rows_exact"] == 18
    assert result["new_history_rows_verified"] == 12 and result["active_beam_count"] == 6
    assert summary["independent_phase_audit_complete"] is False
    assert np.isnan(after["candidate"]["accepted_xyz"][:, 2]).all()


def refresh(path, role, seal):
    item = seal["outputs"][role]
    item.update(size=path.stat().st_size, sha256=a.m.sha(path))


@pytest.mark.parametrize("change", ["candidate", "progress", "snapshot", "producer_true", "schema"])
def test_completed_loader_rejects_false_consistency(monkeypatch, tmp_path, change):
    state, args, seal, summary = completed_fixture(monkeypatch, tmp_path)
    if change == "candidate":
        p = args.output / "candidate.npz"
        with np.load(p, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["pose_xyz"][0, 0, 0, 0] += 0.1
        np.savez_compressed(p, **arrays)
        refresh(p, "candidate", seal)
    elif change == "progress":
        p = args.output / "progress.jsonl"
        p.write_text('{"processed":5,"source_frame":49804,"phase_processed":2}\n')
        refresh(p, "progress", seal)  # Final count alone does not prove each chunk endpoint.
    elif change == "snapshot":
        p = args.output / "code_snapshot/script.py"
        p.write_text("changed source")
        refresh(p, str(p), seal)
    else:
        p = args.output / "summary.json"
        if change == "producer_true":
            summary["independent_phase_audit_complete"] = True
        else:
            summary["schema"] = "long_continuation_phase_v1"
        p.write_bytes(a.m._canonical(summary))
        refresh(p, "summary", seal)
    with pytest.raises(ValueError):
        load_fixture(state, args, seal)


def test_full_final_endpoint_is_a_valid_completed_child_not_a_parent(monkeypatch, tmp_path):
    _state, args, _seal, summary = completed_fixture(monkeypatch, tmp_path)
    meta = a.m.read(args.output / "checkpoint/metadata.json")
    stop, start = 348600, 345280
    total = stop - a.m.FIRST
    summary.update(
        full_sequence_complete=True,
        prefix_backtrace_provisional=False,
        processed_interval=[a.m.FIRST, stop],
        processed_total=total,
        phase_frames=[start, stop],
        phase_processed=stop - start,
        consumed_chunk_count=180,
        checkpoint_restore_count=32,
    )
    meta.update(previous_frame=stop - 1, processed_frames=total, restore_count=32)
    a.validate_summary(summary, meta, start=start, stop=stop, restores=32, chunks=180)
    with pytest.raises(ValueError):
        a.validate_summary(summary, meta, start=start, stop=stop + 1, restores=32, chunks=180)


def test_import_signature_checks_every_parent_field_and_preserves_input():
    _, arrays, metadata = synthetic_state(3)
    before = dict(arrays=arrays, metadata=metadata)
    plan = dict(
        parent_state_signatures=a.m.array_signatures(arrays),
        parent_context=metadata["context"],
        parent_history_rows=18,
        expected_restores=8,
    )
    proof = dict(
        all_checkpoint_arrays_exact=sorted(arrays),
        array_signatures=a.m.array_signatures(arrays),
        context_unchanged=True,
        initialize_count=1,
        restore_count=8,
        previous_frame=metadata["previous_frame"],
        external_pose_reset_count=0,
    )
    saved = copy.deepcopy(proof)
    a.verify_state_import(proof, before, plan)
    assert proof == saved
    arrays["state_velocity"][5, 0, 0] += 0.01
    with pytest.raises(ValueError, match="parent39"):
        a.verify_state_import(proof, before, plan)


@pytest.mark.parametrize("change", ["unselected_parent", "terminal_velocity", "sticky_review"])
def test_independent_suffix_rejects_unchosen_or_terminal_corruption(change):
    _, before, meta = synthetic_state(3)
    _, after, tail_meta = synthetic_state(5, restores=8)
    current = dict(
        arrays=after,
        metadata=tail_meta,
        summary=dict(retained_history_rows=30, active_beam_count=6),
    )
    if change == "unselected_parent":
        after["history_pose_xyz"][1, 0, 0, 0] += 0.01
    elif change == "terminal_velocity":
        after["state_velocity"][5, 0, 0] += 0.01
    else:
        after["history_identity_review"][-1, 2] = False
    with pytest.raises(ValueError):
        a.history.suffix_history(dict(arrays=before, metadata=meta), current)


def test_incomplete_child_fails_before_registration_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr(a.m, "REPO", tmp_path)
    run, reg = tmp_path / "run", tmp_path / "reg"
    run.mkdir()
    reg.mkdir()
    (run / "continuation.provenance.json").write_bytes(
        a.m._canonical(dict(stage=a.m.STAGE, status="failed"))
    )
    monkeypatch.setattr(
        a.child, "load_registration", lambda *_: pytest.fail("heavy load forbidden")
    )
    with pytest.raises(ValueError, match="completed successful"):
        a.verify_phase(
            run, reg, "a" * 64, tmp_path / "absent_owned.json", a.m.Evidence(), receipt_store=None
        )


def receipt_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(a.m, "REPO", tmp_path)
    root, cache, stage = tmp_path / "producer", tmp_path / "cache", tmp_path / "inputs"
    root.mkdir()
    cache.mkdir()
    attempt = stage / "attempt"
    attempt.mkdir(parents=True)
    files = {}
    for name in ("manifest.json", "staging.provenance.json"):
        p = attempt / name
        p.write_text("{}")
        files[str(p)] = a.m.sha(p)
    code = tmp_path / "validator.py"
    code.write_text("# fixed synthetic validator")
    files[str(code)] = a.m.sha(code)
    validator = dict(
        source_sha256={str(code): files[str(code)]},
        python="synthetic",
        implementation="synthetic",
        numpy="synthetic",
        schema_dtypes={},
        mask_bitorder="little",
    )
    row = dict(
        start=49800,
        stop=51460,
        attempt_path="attempt",
        manifest_sha256=files[str(attempt / "manifest.json")],
        provenance_sha256=files[str(attempt / "staging.provenance.json")],
        output_hashes={},
    )
    uses = []
    for view in a.m.original.base.VIEWS:
        p = attempt / (view + "_masks.npz")
        p.write_bytes(b"synthetic metadata-only fixture; never decompressed")
        files[str(p)] = a.m.sha(p)
        row["output_hashes"][view + "_masks"] = dict(path=str(p), sha256=files[str(p)])
        scope = dict(
            frame_range=[49800, 51460],
            view=view,
            image_shape_hw=[1200, 1200],
            manifest_path=str(attempt / "manifest.json"),
            manifest_sha256=row["manifest_sha256"],
            seal_path=str(attempt / "staging.provenance.json"),
            seal_sha256=row["provenance_sha256"],
            detector_binding={"fixed": True},
            expected_npz_sha256=files[str(p)],
        )
        request = dict(
            schema=a.receipts.SCHEMA,
            npz_path=str(p),
            npz_bytes=p.stat().st_size,
            scope=scope,
            validator_binding=validator,
        )
        key = a.hashlib.sha256(a.receipts.canonical(request).encode()).hexdigest()
        folder = cache / key
        folder.mkdir()
        body, commit = folder / "receipt.json", folder / "receipt.success.json"
        body.write_bytes(
            a.m._canonical(
                dict(
                    request=request,
                    result=dict(
                        status="ok",
                        validator="chunk_masks.validate_packed_masks",
                        frame_count=1660,
                        validated_fields=sorted(a.receipts.chunk_masks.SCHEMA_DTYPES),
                        every_decoded_frame_sha_verified=True,
                    ),
                )
            )
        )
        files[str(body)] = a.m.sha(body)
        commit.write_bytes(
            a.m._canonical(
                dict(
                    schema=a.receipts.SCHEMA,
                    status="ok",
                    receipt=body.name,
                    sha256=files[str(body)],
                )
            )
        )
        files[str(commit)] = a.m.sha(commit)
        uses.append(
            dict(
                key=key,
                cache_hit=True,
                receipt_path=str(body),
                receipt_sha256=files[str(body)],
                commit_path=str(commit),
                commit_sha256=files[str(commit)],
                npz_path=str(p),
                scope=scope,
            )
        )
    usage = dict(
        schema="receipted_long_child_mask_usage_v1",
        cache_root=str(cache),
        scientific_state_changed=False,
        uses=uses,
        evidence_hashes=files,
    )
    p = root / "mask_validation_receipts.json"
    p.write_bytes(a.m._canonical(usage))
    record = dict(
        inputs={k: dict(path=k, sha256=v) for k, v in files.items()},
        outputs={
            "mask_validation_receipts": dict(path=str(p), sha256=a.m.sha(p), size=p.stat().st_size)
        },
    )
    plan = dict(
        receipt_cache_root=str(cache),
        selected_chunk_records=[row],
        binding=dict(inputs=dict(input_stage_root=str(stage), detector_binding={"fixed": True})),
    )
    return root, record, plan, validator, usage


def test_receipt_native_coverage_proofs_ignore_only_hit_and_timing(monkeypatch, tmp_path):
    root, record, plan, validator, _usage = receipt_fixture(monkeypatch, tmp_path)
    evidence = a.m.Evidence()
    a.m.verified_outputs(
        root,
        {
            **record,
            "status": "ok",
            "params": {"all_input_source_snapshot_output_sha_unchanged": True},
        },
        evidence,
    )
    result = a.verify_receipt_usage(record, root, plan, evidence, validator)
    assert result["uses"] == result["hit_count"] == 3
    assert len(result["stable_uses"]) == 3
    evidence.final_guard()


@pytest.mark.parametrize(
    "change", ["repeated_view", "lost_use", "missing_input", "runtime", "commit"]
)
def test_receipt_rejects_wrong_scope_or_unsealed_proof(monkeypatch, tmp_path, change):
    root, record, plan, validator, usage = receipt_fixture(monkeypatch, tmp_path)
    if change == "repeated_view":
        usage["uses"][1] = copy.deepcopy(usage["uses"][0])
    elif change == "lost_use":
        usage["uses"].pop()
    elif change == "missing_input":
        record["inputs"].pop(usage["uses"][0]["commit_path"])
    elif change == "runtime":
        validator = {**validator, "numpy": "different"}
    else:
        p = Path(usage["uses"][0]["commit_path"])
        p.write_text("{}")
    p = root / "mask_validation_receipts.json"
    p.write_bytes(a.m._canonical(usage))
    evidence = a.m.Evidence()
    evidence.add(p)
    with pytest.raises(ValueError):
        a.verify_receipt_usage(record, root, plan, evidence, validator)


@pytest.mark.parametrize("outcome", ["success", "verify_failure", "late_source"])
def test_separate_auditor_success_and_failure_seals(monkeypatch, tmp_path, outcome):
    monkeypatch.setattr(a.m, "REPO", tmp_path)
    source = tmp_path / "audit_source.py"
    source.write_text("# unchanged synthetic source")
    digest = a.m.sha(source)
    store = runner_fixture.Store(tmp_path / "cache")
    monkeypatch.setattr(a.m, "ReceiptStore", lambda *_: store)
    store.root.mkdir()

    def code(evidence):
        return {str(source): evidence.add(source, digest)}

    monkeypatch.setattr(a, "audit_sources", code)

    def verify(*_args, **_kwargs):
        if outcome == "verify_failure":
            raise ValueError("synthetic failed independent check")
        if outcome == "late_source":
            source.write_text("# changed synthetic source")
        return dict(
            status="ok",
            independent_phase_audit_complete=True,
            producer_independent_phase_audit_complete=False,
        )

    monkeypatch.setattr(a, "verify_phase", verify)
    args = SimpleNamespace(
        output=tmp_path / "audit",
        receipt_root=store.root,
        run=tmp_path / "not_executed",
        registration=tmp_path / "not_executed_reg",
        registration_sha256="a" * 64,
        owned_execution=tmp_path / "not_observed.json",
    )
    if outcome == "success":
        result = a.audit(args)
        assert result["independent_phase_audit_complete"] is True
        assert result["producer_independent_phase_audit_complete"] is False
    else:
        with pytest.raises(ValueError):
            a.audit(args)
    seal = a.m.read(args.output / "audit.provenance.json")
    assert seal["status"] == ("ok" if outcome == "success" else "failed")
    assert seal["stage"] == a.STAGE
    assert "mask_validation_receipts" in seal["outputs"]
    assert str(source) in seal["inputs"]
    if outcome != "success":
        assert seal["params"].get("all_input_source_snapshot_output_sha_unchanged") is not True

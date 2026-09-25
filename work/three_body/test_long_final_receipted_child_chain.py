"""Synthetic read-only chain contracts; no actual prefix, process, CUDA or raw data."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import long_final_receipted_child_chain as c
import numpy as np
import pytest
from test_long_final import Evidence as OldEvidence
from test_long_final import audit_data, entry, seal, write_json
from test_long_final_mixed_chain import header
from test_long_final_mixed_chain import mixed_schedule as _mixed_schedule
from test_receipted_runner_phase_audit import plan_fixture

mixed_schedule = _mixed_schedule


class Evidence(OldEvidence):
    def add(self, path, digest=None):
        super().add(path, digest)
        return self.paths[str(Path(path).resolve())]


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("read-only loader attempted computation/process/receipt mutation")

    monkeypatch.setattr(c.child, "execute", refuse)
    monkeypatch.setattr(c.child, "prepare", refuse)
    monkeypatch.setattr(c.m.ReceiptStore, "__init__", refuse)
    monkeypatch.setattr(c.auditor, "audit", refuse)


@pytest.mark.parametrize("k", [0, 1, 6, 26, 27])
def test_prefix_denominators_and_remaining_full_schedule(k):
    result = c.prefix_counts(k)
    assert result["through"] == 76360 + 9960 * k
    assert result["consumed_chunks"] == 16 + 6 * k
    assert result["native_frames"] == (16 + 6 * k) * 1660
    assert result["tracking_processes"] == 5 + k
    assert result["checkpoint_restores"] == 4 + k
    remaining = c.expected_new_phases()[k:]
    assert result["tracking_processes"] + len(remaining) == 33
    assert result["checkpoint_restores"] + len(remaining) == 32
    assert remaining[0][0] == result["through"]
    assert remaining[-1] == (345280, 348600)
    assert len(c.native.fixed_sample_frames()) == 543
    assert len(range(51460, 348600, 1660)) == 179


@pytest.mark.parametrize("k", [-1, 28, True, 6.0, "6"])
def test_non_prefix_or_non_integer_count_rejected(k):
    with pytest.raises(ValueError):
        c.prefix_counts(k)


def stopped_header(k=6):
    summary, policy = header()
    summary.update(
        status="stopped_after_completed_phase",
        through=76360 + 9960 * k,
        full_sequence_complete=False,
        prefix_backtrace_provisional=True,
        checkpoint_restore_count=4 + k,
        phases=[{} for _ in range(k)],
        stop_request={"preserved": "bytes"},
    )
    return summary, policy


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "complete"),
        ("through", 136121),
        ("checkpoint_restore_count", 11),
        ("logical_initialization_count", 2),
        ("full_sequence_complete", True),
        ("stop_request", None),
        ("next_phase_retried", True),
    ],
)
def test_stopped_prefix_is_not_fabricated_full_completion(field, value):
    summary, policy = stopped_header()
    before = copy.deepcopy(summary)
    assert c.validate_prefix_header(summary, policy)["through"] == 136120
    assert summary == before
    summary[field] = value
    with pytest.raises(ValueError):
        c.validate_prefix_header(summary, policy)


@pytest.mark.parametrize("k", [0, 6, 27])
def test_original_ancestor_and_first_phase_are_preserved_in_prefix(mixed_schedule, monkeypatch, k):
    f = mixed_schedule
    for name in ("bound_sources", "verify_ancestors", "verify_receipt_usage", "check_phase_edge"):
        monkeypatch.setattr(c, name, getattr(c.old, name))
    monkeypatch.setattr(c, "PINS", {})
    monkeypatch.setattr(c, "verify_stopped_tail", lambda *args: {"unused": "CPU only"})
    monkeypatch.setattr(c, "verify_exit_witness", lambda *args: {"held": "exit0"})
    monkeypatch.setattr(np, "load", lambda *args, **kw: pytest.fail("NPZ opened"))
    summary = c.COMMON.read_json(f["summary"])
    summary.update(
        status="stopped_after_completed_phase",
        through=76360 + 9960 * k,
        full_sequence_complete=False,
        prefix_backtrace_provisional=True,
        checkpoint_restore_count=4 + k,
        stop_request={"honored": True},
    )
    summary["phases"] = summary["phases"][:k]
    summary["final_run"] = summary["phases"][-1]["run"] if k else str(c.handoff.CHILD)
    write_json(f["summary"], summary)
    record = c.COMMON.read_json(f["supervisor"])
    record["outputs"]["summary"] = entry(f["summary"])
    write_json(f["supervisor"], record)
    result = c.load_completed_prefix(f["root"], entry(f["supervisor"])["sha256"], {}, Evidence())
    assert [x["restores"] for x in f["edge_calls"]] == list(range(4, 5 + k))
    assert [x["receipted"] for x in f["edge_calls"]] == [False] + [True] * k
    assert len(result["ancestors"]) == 4 and len(result["phases"]) == 1 + k
    assert result["tracking_processes"] == 5 + k and result["checkpoint_restores"] == 4 + k
    assert result["summary"] == summary
    assert result["unused_registration"] == {"unused": "CPU only"}
    assert result["final_audit_complete"] is False


@pytest.fixture
def unused_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(c.legacy, "REPO", tmp_path)
    root, parent = tmp_path / "supervisor", tmp_path / "parent"
    root.mkdir()
    parent.mkdir()
    source, saved = root / "stop_after_phase.request", root / "honored_stop_request.bin"
    source.write_bytes(b"stop after complete phase\xff\r\n")
    saved.write_bytes(source.read_bytes())
    stop = dict(
        path=str(source),
        sha256=entry(source)["sha256"],
        size=source.stat().st_size,
        child_interrupted=False,
        boundary="completed, independently audited parent",
    )
    stop_path = write_json(root / "stop_request.json", stop)
    op = root / "phases/f136120-146080"
    reg = op / "registration"
    binding = {"unchanged": "old36"}
    plan = write_json(
        reg / "plan.json",
        dict(
            parent_phase=str(parent),
            expected_output=str(op / "tracking"),
            next_native_interval=[136120, 146080],
            expected_restores=11,
            frozen_input_and_code_binding=binding,
        ),
    )
    execution = write_json(reg / "execution.json", {"CPU": "registration"})
    receipts = write_json(reg / "mask_validation_receipts.json", {"retained": True})
    rs = seal(
        reg,
        "registration.provenance.json",
        c.operation.REGISTRATION_STAGE,
        dict(plan=plan, execution=execution, mask_validation_receipts=receipts),
    )
    ss = seal(
        root,
        "supervisor.provenance.json",
        c.NEW_STAGE,
        dict(
            stop_request=stop_path,
            honored_stop_request=saved,
            **{str(p): p for p in (plan, execution, receipts, rs)},
        ),
        inputs=[source],
    )
    summary = dict(stop_request=stop, incomplete_phase=str(op))
    return root, summary, c.COMMON.read_json(ss), c.prefix_counts(6), binding, parent


def test_unused_successful_registration_retained_without_tracking_or_restore(unused_registration):
    result = c.verify_stopped_tail(*unused_registration, Evidence())
    assert result["interval"] == [136120, 146080]
    assert result["tracking_processes"] == result["logical_restores"] == 0
    assert not (Path(result["root"]).parent / "tracking").exists()


@pytest.mark.parametrize(
    "name",
    ["tracking", "child_execution.json", "tracking.log", "operation.provenance.json", "audit.json"],
)
def test_unused_registration_with_any_execution_evidence_rejected(unused_registration, name):
    op = Path(unused_registration[1]["incomplete_phase"])
    (op / name).write_text("execution happened")
    with pytest.raises(ValueError, match="execution evidence"):
        c.verify_stopped_tail(*unused_registration, Evidence())


def edge_fixture(tmp_path, monkeypatch):
    plan, legacy = plan_fixture(monkeypatch, tmp_path)
    start, stop = plan["next_native_interval"]
    parent, run = tmp_path / "parent", tmp_path / "run"
    before = dict(
        context={"science": "same"},
        previous_frame=start - 1,
        initialize_count=1,
        restore_count=10,
        processed_frames=start - c.FIRST,
    )
    after = dict(
        before,
        previous_frame=stop - 1,
        processed_frames=stop - c.FIRST,
        restore_count=11,
        node_recovery_diagnostics={"same": "registered"},
    )
    signatures = {f"field{i}": dict(dtype="<f8", shape=[1], sha256="c" * 64) for i in range(39)}
    plan.update(
        parent=str(parent),
        expected_output=str(run),
        expected_restores=11,
        parent_context=before["context"],
        parent_state_signatures=signatures,
        parent_history_rows=517913,
        parent_audit=dict(path=str(tmp_path / "old_audit.json"), sha256="d" * 64),
    )
    producer = dict(
        params=dict(bindings=plan["binding"], resume_phase=str(parent), through_frame=stop),
        outputs=dict(candidate=dict(sha256="e" * 64)),
    )
    summary = dict(
        schema=c.m.SUMMARY_SCHEMA,
        technical_completion=True,
        full_sequence_complete=False,
        prefix_backtrace_provisional=True,
        registered_interval=[c.FIRST, c.FINAL_STOP],
        processed_interval=[c.FIRST, stop],
        processed_total=stop - c.FIRST,
        phase_frames=[start, stop],
        phase_processed=stop - start,
        consumed_chunk_count=(stop - c.FIRST) // 1660,
        logical_initialization_count=1,
        checkpoint_restore_count=11,
        external_pose_reset_count=0,
        restores_during_phase=0,
        temporal_subsampling=False,
        human_annotation_input=False,
        post_initialization_baseline_coordinates_input=False,
        all_39_checkpoint_arrays_verified=True,
        all_history_recurrences_checked_by_frozen_decoder=True,
        independent_ID_truth=False,
        independent_phase_audit_complete=False,
        node_recovery_diagnostics=after["node_recovery_diagnostics"],
        verifier_transition=plan["transition"],
        active_beam_count=6,
        history_prefix=dict(
            parent_native_frames=start - c.FIRST, parent_rows=517913, all_history_prefix_exact=True
        ),
    )
    audit = audit_data(run, "e" * 64, start, stop, 11)
    audit.update(
        audit_version=c.auditor.STAGE,
        parent=str(parent),
        restores_during_phase=0,
        independent_phase_audit_complete=True,
        producer_independent_phase_audit_complete=False,
        transition=plan["transition"],
        parent_audit=plan["parent_audit"],
    )
    proof = dict(
        all_checkpoint_arrays_exact=sorted(signatures),
        array_signatures=signatures,
        context_unchanged=True,
        initialize_count=1,
        restore_count=11,
        previous_frame=start - 1,
        external_pose_reset_count=0,
    )
    args = [plan, producer, summary, audit, before, after, proof, legacy, plan["binding"]]
    kw = dict(
        parent=parent,
        run=run,
        start=start,
        stop=stop,
        restores=11,
        parent_audit_path=tmp_path / "old_audit.json",
        parent_audit_sha="d" * 64,
    )
    return args, kw


def test_new_edge_one_migration_then_identical_binding_and_false_producer(tmp_path, monkeypatch):
    args, kw = edge_fixture(tmp_path, monkeypatch)
    before = copy.deepcopy(args)
    assert c.validate_new_edge(*args, **kw)["kind"] == "legacy_to_receipted"
    assert args == before and args[2]["independent_phase_audit_complete"] is False
    args[0]["parent_binding"] = copy.deepcopy(args[0]["binding"])
    edge = c.m.transition(args[0]["binding"], args[0]["binding"], args[7])
    for value, key in (
        (args[0], "transition"),
        (args[2], "verifier_transition"),
        (args[3], "transition"),
    ):
        value[key] = edge
    assert c.validate_new_edge(*args, **kw)["kind"] == "receipted_to_receipted"


@pytest.mark.parametrize(
    "fault",
    [
        "context",
        "restore",
        "signature",
        "producer_true",
        "audit_false",
        "parent_history",
        "suffix",
        "beam",
        "input_binding",
        "source_binding",
        "21",
        "parent_audit",
    ],
)
def test_independent_new_edge_drift_rejected(tmp_path, monkeypatch, fault):
    args, kw = edge_fixture(tmp_path, monkeypatch)
    if fault == "context":
        args[5]["context"] = {"different": True}
    elif fault == "restore":
        args[4]["restore_count"] = 11
    elif fault == "signature":
        args[6] = copy.deepcopy(args[6])
        args[6]["array_signatures"]["field0"]["sha256"] = "f" * 64
    elif fault == "producer_true":
        args[2]["independent_phase_audit_complete"] = True
    elif fault == "audit_false":
        args[3]["independent_phase_audit_complete"] = False
    elif fault == "parent_history":
        args[2]["history_prefix"]["all_history_prefix_exact"] = False
    elif fault == "suffix":
        args[3]["history"]["all_suffix_parent_recurrences_exact"] = False
    elif fault == "beam":
        args[2]["active_beam_count"] = 5
    elif fault == "input_binding":
        args[0]["binding"] = copy.deepcopy(args[0]["binding"])
        args[0]["binding"]["inputs"] = {"seed": "changed"}
    elif fault == "source_binding":
        args[0]["parent_binding"] = copy.deepcopy(args[0]["parent_binding"])
        args[0]["parent_binding"]["opencv_threads"] = 2
    elif fault == "21":
        args[3]["all_21_candidate_arrays_exact_backtrace"].pop("0")
    elif fault == "parent_audit":
        args[0]["parent_audit"] = dict(path=str(tmp_path / "foreign"), sha256="d" * 64)
    with pytest.raises(ValueError):
        c.validate_new_edge(*args, **kw)


def test_terminal_refuses_before_checkpoint_load(tmp_path, monkeypatch):
    write_json(tmp_path / "supervisor.provenance.json", dict(status="ok", stage=c.NEW_STAGE))
    monkeypatch.setattr(
        c.phase.old.prior, "read_checkpoint", lambda *a, **k: pytest.fail("checkpoint opened")
    )
    with pytest.raises(ValueError, match="new supervisor"):
        c.load_completed_final(tmp_path, Evidence())


def test_native_assembly_reuses_frozen_body_without_process_or_binding_substitution():
    original = ast.parse(Path(c.old.__file__).read_text())
    adapted = ast.parse(Path(c.__file__).read_text())
    left = next(
        n
        for n in original.body
        if isinstance(n, ast.FunctionDef) and n.name == "load_native_sample_inputs"
    )
    right = next(
        n
        for n in adapted.body
        if isinstance(n, ast.FunctionDef) and n.name == "load_native_sample_inputs"
    )

    # Everything from consumed-record loading to return is the frozen numerical/I/O body.
    def suffix(node):
        index = next(
            i
            for i, n in enumerate(node.body)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(x, ast.Tuple)
                and any(isinstance(y, ast.Name) and y.id == "run" for y in x.elts)
                for x in n.targets
            )
        )
        return [ast.dump(n, include_attributes=False) for n in node.body[index:]]

    assert suffix(left) == suffix(right)


def new_header(k=6):
    counts = c.prefix_counts(k)
    handoff = dict(
        schema=c.HANDOFF_SCHEMA,
        old_completed_phases=k,
        initial_parent="parent",
        initial_stop=counts["through"],
        initial_restores=4 + k,
        unused_registration=None,
        prefix_counts=counts,
        scientific_context_changed=False,
    )
    summary = dict(
        schema=c.SUPERVISOR_STAGE,
        status="complete",
        through=348600,
        full_sequence_complete=True,
        prefix_backtrace_provisional=False,
        logical_initialization_count=1,
        checkpoint_restore_count=32,
        external_pose_reset_count=0,
        active_beam_count=6,
        tracking_processes=33,
        consumed_chunk_count=180,
        stop_request=None,
        incomplete_phase=None,
        independent_ID_truth=False,
        final_audit_complete=False,
        detector_started=False,
        old_process_stopped=False,
        next_phase_retried=False,
        phases=[{} for _ in range(28 - k)],
    )
    policy = dict(
        schema=c.SUPERVISOR_STAGE,
        initial_parent="parent",
        initial_stop=counts["through"],
        initial_restores=4 + k,
        target_interval=[49800, 348600],
        phase_intervals=[list(p) for p in c.expected_new_phases()[k:]],
        new_external_pose_seed=False,
        failed_phase_retry=False,
        scheduling_changes_scientific_protocol=False,
    )
    return summary, policy, handoff


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "stopped_after_completed_phase"),
        ("full_sequence_complete", False),
        ("tracking_processes", 34),
        ("checkpoint_restore_count", 31),
        ("consumed_chunk_count", 179),
        ("final_audit_complete", True),
        ("stop_request", {}),
        ("incomplete_phase", "registration"),
        ("phases", [{}] * 23),
    ],
)
def test_full_new_header_requires_exact_remaining_denominator_and_separate_audit(field, value):
    summary, policy, handoff = new_header()
    assert c.validate_new_header(summary, policy, handoff) == c.prefix_counts(6)
    summary[field] = value
    with pytest.raises(ValueError):
        c.validate_new_header(summary, policy, handoff)


@pytest.fixture
def new_schedule(tmp_path, monkeypatch):
    from test_receipted_long_runner_migration import bindings

    legacy, code = bindings(monkeypatch, tmp_path)
    binding = c.m.execution_binding(legacy, code)
    monkeypatch.setattr(c.legacy, "REPO", tmp_path)
    monkeypatch.setattr(c, "PINS", {})
    monkeypatch.setattr(c, "bound_sources", lambda *args: None)
    root, oldroot = tmp_path / "newowner", tmp_path / "oldowner"
    parent = oldroot / "tracking"
    parent.mkdir(parents=True)
    os = write_json(oldroot / "supervisor.provenance.json", {"stopped": True})
    witness = write_json(tmp_path / "witness/witness.json", {"held": "exit"})
    pa = write_json(oldroot / "audit/audit.provenance.json", {"audited": True})
    summary, policy, handoff = new_header(6)
    handoff.update(
        initial_parent=str(parent),
        old_supervisor=str(oldroot),
        old_supervisor_sha256=entry(os)["sha256"],
        exit_witness=entry(witness),
    )
    policy.update(initial_parent=str(parent), fixed_bindings=binding, source_hashes={})
    prefix = dict(
        prefix_counts=c.prefix_counts(6),
        final_run=parent,
        final_record={"params": {"bindings": legacy}},
        binding=legacy,
        last_audit_root=pa.parent,
        last_audit_name=pa.name,
        unused_registration=None,
        owner_exit={"exit_time_utc": "2026-09-09T00:00:00+00:00"},
        ancestors=[{}] * 4,
        phases=[dict(restores=r) for r in range(4, 11)],
    )

    def completed_prefix(old_root, sha, witness_reference, evidence):
        assert old_root == str(oldroot) and sha == entry(os)["sha256"]
        assert witness_reference == entry(witness)
        for path in (os, witness, pa):
            evidence.add(path)
        return copy.deepcopy(prefix)

    monkeypatch.setattr(c, "load_completed_prefix", completed_prefix)
    calls = []

    def edge(
        op,
        p,
        previous,
        audit_path,
        audit_sha,
        old_binding,
        new_binding,
        start,
        stop,
        restores,
        owner,
        evidence,
    ):
        calls.append((start, stop, restores, previous["params"]["bindings"]))
        transition = c.m.transition(previous["params"]["bindings"], new_binding, old_binding)
        seals = {
            name: "f" * 64
            for name in ("operation_sha256", "registration_sha256", "child_sha256", "audit_sha256")
        }
        linked = dict(
            parent=str(p),
            run=str(op / "tracking"),
            interval=[start, stop],
            restores=restores,
            transition=transition,
            **seals,
        )
        return (
            linked,
            {"params": {"bindings": new_binding}, "finished_at": "2026-09-09T00:01:30+00:00"},
            {"last": stop},
        )

    monkeypatch.setattr(c, "check_receipted_edge", edge)
    summary["phases"] = []
    p = parent
    for i, (start, stop) in enumerate(c.expected_new_phases()[6:]):
        op = root / "phases" / f"f{start:06d}-{stop:06d}"
        write_json(op / "operation.provenance.json", {"started_at": "2026-09-09T00:01:05+00:00"})
        seals = {
            name: "f" * 64
            for name in ("operation_sha256", "registration_sha256", "child_sha256", "audit_sha256")
        }
        summary["phases"].append(c.phase_row(op, p, [start, stop], 11 + i, seals))
        p = op / "tracking"
    summary["final_run"] = str(p)
    sp, pp, hp = (
        write_json(root / name, data)
        for name, data in (
            ("summary.json", summary),
            ("supervisor_plan.json", policy),
            ("handoff.json", handoff),
        )
    )
    sealfile = seal(
        root,
        "supervisor.provenance.json",
        c.SUPERVISOR_STAGE,
        dict(summary=sp, plan=pp, handoff=hp),
        inputs=[os, witness],
    )
    return dict(root=root, summary=sp, handoff=hp, seal=sealfile, calls=calls, prefix=prefix)


def test_new_full_chain_exact_suffix_and_one_migration(new_schedule, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("array opened"))
    f = new_schedule
    chain = c.load_supervisor_chain(f["root"], Evidence())
    assert [(a, b) for a, b, *_ in f["calls"]] == list(c.expected_new_phases()[6:])
    assert [r for _, _, r, _ in f["calls"]] == list(range(11, 33))
    transitions = [p["transition"]["kind"] for p in chain["phases"][7:]]
    assert transitions == ["legacy_to_receipted"] + ["receipted_to_receipted"] * 21
    assert len(chain["ancestors"]) + len(chain["phases"]) == chain["tracking_processes"] == 33
    assert chain["checkpoint_restores"] == 32 and chain["final_audit_complete"] is False
    assert chain["final_audit"] == {"last": 348600}


@pytest.mark.parametrize(
    "fault", ["row_order", "restore", "unused_registration", "second_migration"]
)
def test_new_full_chain_does_not_promote_wrong_suffix(new_schedule, monkeypatch, fault):
    f = new_schedule
    path = f["handoff"] if fault == "unused_registration" else f["summary"]
    data = c.COMMON.read_json(path)
    if fault == "row_order":
        data["phases"][0], data["phases"][1] = data["phases"][1], data["phases"][0]
    elif fault == "restore":
        data["phases"][-1]["checkpoint_restore_count"] = 31
    elif fault == "unused_registration":
        data["unused_registration"] = {"discarded": True}
    else:
        prior = c.check_receipted_edge

        def edge(*args):
            linked, record, audit = prior(*args)
            linked["transition"]["kind"] = "legacy_to_receipted"
            return linked, record, audit

        monkeypatch.setattr(c, "check_receipted_edge", edge)
    write_json(path, data)
    record = c.COMMON.read_json(f["seal"])
    record["outputs"]["handoff" if path == f["handoff"] else "summary"] = entry(path)
    write_json(f["seal"], record)
    with pytest.raises(ValueError):
        c.load_supervisor_chain(f["root"], Evidence())


def held_fixture(tmp_path, monkeypatch, fault=None):
    from test_receipted_long_handoff import fixture

    f = fixture(tmp_path)
    frozen_source = f.ns["WITNESS_SOURCE"]
    adapter_source = tmp_path / "capture_receipted_owner_exit.py"
    adapter_source.write_text("# synthetic metadata adapter only\n")
    cfg = tmp_path / ".venv/pyvenv.cfg"
    files = [adapter_source, frozen_source, f.source, cfg]
    names = [
        "00_capture_receipted_owner_exit.py",
        "01_capture_owned_long_exit.py",
        "02_run_long_receipted_supervisor.py",
        "03_pyvenv.cfg",
    ]
    # Only this newly created tiny fixture directory is changed.
    for p in (f.root / "source_snapshot").iterdir():
        p.unlink()
    for source, name in zip(files, names, strict=True):
        (f.root / "source_snapshot" / name).write_bytes(source.read_bytes())
    frozen = SimpleNamespace(**{**vars(f.witness), "__file__": str(frozen_source)})
    capture = SimpleNamespace(
        __file__=str(adapter_source),
        frozen=frozen,
        FROZEN_SHA=entry(frozen_source)["sha256"],
        SUPERVISOR_SHA=entry(f.source)["sha256"],
        PYTHON=f.old.PYTHON,
        build_binding=f.witness.build_binding,
    )
    monkeypatch.setattr(c, "REPO", tmp_path)
    monkeypatch.setattr(c, "capture", capture)
    monkeypatch.setattr(c, "current", SimpleNamespace(__file__=str(f.source)))
    monkeypatch.setattr(c, "PINS", {adapter_source: entry(adapter_source)["sha256"]})
    overall = c.COMMON.read_json(f.root / "witness.provenance.json")
    owner = c.COMMON.read_json(f.root / "owner_exit.json")
    owner["observation"].update(exit_code=0, exit_code_success=True)
    overall["observations"]["owner"] = copy.deepcopy(owner["observation"])
    if fault == "nonzero_owner":
        owner["observation"].update(exit_code=1, exit_code_success=False)
        overall["observations"]["owner"] = copy.deepcopy(owner["observation"])
    elif fault == "unheld":
        owner["observation"]["wait_signalled"] = False
        overall["observations"]["owner"] = copy.deepcopy(owner["observation"])
    elif fault == "pid_reuse":
        owner["observation"]["pid"] += 1
        overall["observations"]["owner"] = copy.deepcopy(owner["observation"])
    elif fault == "later_launch":
        owner["event_capture"]["subsequent_launch_events"] = [{"event": "phase_start"}]
    elif fault == "partial_owner_line":
        owner["event_capture"]["partial_trailing_line"] = True
    write_json(f.root / "owner_exit.json", owner)
    inputs = {str(p): entry(p)["sha256"] for p in files}
    outputs = {
        str(f.root / "source_snapshot" / name): entry(f.root / "source_snapshot" / name)["sha256"]
        for name in names
    }
    for name in (
        "binding.json",
        "events_baseline.jsonl",
        "attachment.json",
        "child_exit.json",
        "events_at_child_exit.jsonl",
    ):
        outputs[str(f.root / name)] = entry(f.root / name)["sha256"]
    for role in ("child", "owner"):
        if role == "owner":
            for name in ("owner_exit.json", "events_at_owner_exit.jsonl"):
                outputs[str(f.root / name)] = entry(f.root / name)["sha256"]
        receipt = c.COMMON.read_json(f.root / f"{role}_exit.receipt.json")
        receipt.update(inputs=inputs, outputs=outputs.copy())
        if fault == "unbound_receipt" and role == "owner":
            receipt["outputs"].pop(str(f.root / "attachment.json"))
        receipt_path = write_json(f.root / f"{role}_exit.receipt.json", receipt)
        outputs[str(receipt_path)] = entry(receipt_path)["sha256"]
    overall.update(stage=c.EXIT_SCHEMA, inputs=inputs, outputs=outputs)
    if fault == "old_stage":
        overall["stage"] = "capture_owned_long_process_exits"
    if fault == "failed":
        overall["status"] = "failed"
    path = write_json(f.root / "witness.provenance.json", overall)
    summary = write_json(f.oldroot / "summary.json", dict(final_run=str(f.child), phases=[]))
    binding = c.COMMON.read_json(f.root / "binding.json")
    lock = write_json(f.oldroot / "supervisor.lock", binding["owner_lock"])
    supervisor = dict(
        outputs=dict(summary=entry(summary), lock=entry(lock)), finished_at=f.utc(200)
    )
    if fault == "owner_before_seal":
        supervisor["finished_at"] = f.utc(241)
    return entry(path), f.oldroot, supervisor


def test_held_exit_data_requires_both_successful_exits_without_process_queries(
    tmp_path, monkeypatch
):
    values = held_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        c.capture.frozen, "WindowsProcess", lambda *a: pytest.fail("process queried")
    )
    result = c.verify_exit_witness(*values, Evidence())
    assert result["exit_code"] == 0 and result["wait_signalled"] is True
    assert result["pid"] == 101
    assert result["path"] == values[0]["path"]
    assert result["sha256"] == values[0]["sha256"]


@pytest.mark.parametrize(
    "fault",
    [
        "nonzero_owner",
        "unheld",
        "pid_reuse",
        "later_launch",
        "partial_owner_line",
        "unbound_receipt",
        "old_stage",
        "failed",
        "owner_before_seal",
    ],
)
def test_held_exit_is_not_inferred_from_pid_absence_or_successful_tracking(
    tmp_path, monkeypatch, fault
):
    values = held_fixture(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError):
        c.verify_exit_witness(*values, Evidence())


def test_operational_closure_uses_own_snapshots_not_prior_snapshot_input_modules(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(c, "REPO", tmp_path)
    monkeypatch.setattr(c.old, "REPO", tmp_path)
    source = tmp_path / "work/driver.py"
    source.parent.mkdir()
    source.write_text("# live executed source\n")
    prior = tmp_path / "prior/code_snapshot/work/driver.py"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(source.read_bytes())
    root = tmp_path / "audit"
    snapshot = root / "code_snapshot/work/driver.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(source.read_bytes())
    record = dict(
        inputs={str(p): entry(p) for p in (source, prior)}, outputs={str(snapshot): entry(snapshot)}
    )
    expected = {str(source): entry(source)["sha256"]}
    assert c.verify_source_closure(record, root, Evidence(), expected) == expected
    assert not (root / "code_snapshot/prior").exists()
    snapshot.write_text("changed executed bytes")
    with pytest.raises(ValueError):
        c.verify_source_closure(record, root, Evidence(), expected)

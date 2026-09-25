"""File-backed handoff contracts. No real handles, child process or tracker calls."""

from __future__ import annotations

import copy
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FunctionType, SimpleNamespace

import pytest

WORK = Path(__file__).resolve().parent
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

import receipted_long_handoff as module  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def entry(path):
    return dict(path=str(path), size=path.stat().st_size, sha256=module.old.sha256(path))


def standard(
    root,
    name,
    stage,
    *,
    inputs=None,
    outputs=None,
    params=None,
    started="2026-09-09T00:00:10+00:00",
    finished="2026-09-09T00:00:20+00:00",
):
    record = dict(
        stage=stage,
        status="ok",
        inputs=inputs or {},
        outputs=outputs or {},
        params={"all_input_source_snapshot_output_sha_unchanged": True, **(params or {})},
        started_at=started,
        finished_at=finished,
    )
    write(root / name, record)
    return record


def clone_namespace(overrides):
    namespace = {**vars(module), **overrides}
    for name, function in vars(module).items():
        if isinstance(function, FunctionType) and function.__module__ == module.__name__:
            copied = FunctionType(function.__code__, namespace, name, function.__defaults__)
            copied.__kwdefaults__ = function.__kwdefaults__
            namespace[name] = copied
    return namespace


def fixture(tmp_path, bad=None):
    root, oldroot = tmp_path / "witness", tmp_path / "old"
    oldparent = tmp_path / "old66400"
    child = oldroot / "phases/f066400-076360/tracking"
    for p in (root, oldroot, oldparent, child):
        p.mkdir(parents=True, exist_ok=True)
    witness_source = tmp_path / "capture_owned_long_exit.py"
    witness_source.write_bytes(b"# fixture witness implementation, not real execution\n")
    old_source = tmp_path / "run_long_remaining_supervisor.py"
    old_source.write_bytes(b"# fixture supervisor implementation\n")
    cfg = tmp_path / ".venv/pyvenv.cfg"
    cfg.parent.mkdir()
    cfg.write_bytes(b"fixture runtime binding")
    python = tmp_path / ".venv/python.exe"
    python.write_bytes(b"not executable; no process used")
    base = datetime(2026, 9, 9, tzinfo=UTC)

    def utc(seconds):
        return (base + timedelta(seconds=seconds)).isoformat()

    def tick(seconds):
        delta = base + timedelta(seconds=seconds) - datetime(1601, 1, 1, tzinfo=UTC)
        return (delta.days * 86400 + delta.seconds) * 10_000_000 + delta.microseconds * 10

    command = [str(python), "fixture command, no launch"]
    phase = dict(
        event="phase_start",
        time=utc(60),
        parent=str(oldparent),
        output=str(child),
        interval=[66400, 76360],
    )
    launch = dict(
        event="owned_child_started",
        time=utc(62),
        pid=202,
        command=command,
        output=str(child),
        log=str(child.parent / "tracking.log"),
    )
    baseline = b"".join((json.dumps(r) + "\n").encode() for r in (phase, launch))
    final_events = (
        baseline + (json.dumps(dict(event="phase_complete", time=utc(181))) + "\n").encode()
    )
    if bad == "later_launch":
        final_events += (json.dumps(dict(event="phase_start", time=utc(241))) + "\n").encode()
    if bad == "partial_events":
        final_events += b'{"event": "phase_start"'
    events = oldroot / "events.jsonl"
    events.write_bytes(final_events)
    lock = dict(created_at=utc(5))
    binding = dict(
        schema=module.witness.SCHEMA,
        supervisor_root=str(oldroot),
        owner_pid=101,
        child_pid=202,
        owner_lock=lock,
        child_event=launch,
        child_event_index=1,
        phase_start=phase,
        immutable_hashes={str(old_source): module.old.sha256(old_source)},
        events_path=str(events),
        baseline_events_sha256=module.witness.digest(baseline),
        baseline_events_bytes=len(baseline),
    )
    if bad == "wrong_interval":
        binding["phase_start"]["interval"] = [76360, 86320]
    inputs = {str(p): module.old.sha256(p) for p in (witness_source, old_source, cfg)}
    for index, path in enumerate((witness_source, old_source, cfg)):
        snap = root / "source_snapshot" / f"{index:02d}_{path.name}"
        snap.parent.mkdir(exist_ok=True)
        snap.write_bytes(path.read_bytes())
    write(root / "binding.json", binding)
    (root / "events_baseline.jsonl").write_bytes(baseline)
    identities, observations = {}, {}
    for role, pid, created, ended, code in (("owner", 101, 0, 240, 1), ("child", 202, 61, 180, 0)):
        if bad == "owner_first" and role == "owner":
            ended = 170
        if bad == "child_nonzero" and role == "child":
            code = 7
        if bad == "reused_child_pid" and role == "child":
            created = 70
        identity = dict(
            pid=pid,
            creation_filetime_100ns=tick(created),
            creation_time_utc=utc(created),
            image_path=str(python),
            attached_at=utc(120),
            access_mask=0x00101000,
            live_when_handle_acquired=True,
        )
        observation = dict(
            **identity,
            exit_code=code,
            exit_filetime_100ns=tick(ended),
            exit_time_utc=utc(ended),
            exit_observed_at=utc(ended + 0.1),
            wait_signalled=True,
            exit_code_success=code == 0,
        )
        if bad == "changed_identity" and role == "child":
            observation["pid"] = 999
        if bad == "filetime_mismatch" and role == "child":
            observation["exit_filetime_100ns"] += 10_000_000
        if bad == "not_signalled" and role == "owner":
            observation["wait_signalled"] = False
        identities[role], observations[role] = identity, observation
    write(
        root / "attachment.json",
        dict(
            schema=module.witness.SCHEMA,
            processes=identities,
            runtime=dict(base_executable=str(python)),
        ),
    )
    for role in ("child", "owner"):
        capture = root / f"events_at_{role}_exit.jsonl"
        capture.write_bytes(final_events)
        payload = dict(
            schema=module.witness.SCHEMA,
            role=role,
            observation=observations[role],
            binding_sha256=module.old.sha256(root / "binding.json"),
            event_capture=dict(
                path=str(capture),
                sha256=module.old.sha256(capture),
                bytes=len(final_events),
                complete_lines=len(module.witness.event_rows(final_events)),
                subsequent_launch_events=[],
                partial_trailing_line=not final_events.endswith(b"\n"),
            ),
        )
        write(root / f"{role}_exit.json", payload)
        outputs = {str(p): module.old.sha256(p) for p in root.rglob("*") if p.is_file()}
        write(
            root / f"{role}_exit.receipt.json",
            dict(
                schema=module.witness.SCHEMA,
                role=role,
                status="ok",
                all_evidence_sha_unchanged=True,
                tracking_success_asserted=False,
                process_signal_or_termination_performed=False,
                inputs=inputs,
                outputs=outputs,
                finished_at=utc(241 if role == "owner" else 181),
            ),
        )
    seal = dict(
        schema=module.witness.SCHEMA,
        stage="capture_owned_long_process_exits",
        status="ok",
        error=None,
        inputs=inputs,
        observations=observations,
        outputs={str(p): module.old.sha256(p) for p in root.rglob("*") if p.is_file()},
        tracking_or_supervisor_success_asserted=False,
        signals_enumeration_GPU_or_tracking_performed=False,
        finished_at=utc(242),
    )
    if bad == "failed_witness":
        seal["status"] = "failed"
    write(root / "witness.provenance.json", seal)
    if bad == "changed_payload":
        (root / "attachment.json").write_bytes(b"changed after successful seal")
    if bad == "changed_current_events":
        events.write_bytes(final_events + b"\n")

    def build_binding(*args):
        assert args == (oldroot, 202)
        current = copy.deepcopy(binding)
        current.update(
            baseline_events_sha256=module.witness.digest(events.read_bytes()),
            baseline_events_bytes=events.stat().st_size,
        )
        return current, events.read_bytes()

    fake_old = SimpleNamespace(**{**vars(module.old), "WORK": tmp_path, "PYTHON": python})
    fake_witness = SimpleNamespace(**{**vars(module.witness), "build_binding": build_binding})
    ns = clone_namespace(
        dict(
            REPO=tmp_path,
            WORK=tmp_path,
            OLD_ROOT=oldroot,
            OLD_PARENT=oldparent,
            CHILD=child,
            WITNESS_SOURCE=witness_source,
            WITNESS_SHA=module.old.sha256(witness_source),
            old=fake_old,
            witness=fake_witness,
        )
    )
    return SimpleNamespace(
        root=root,
        oldroot=oldroot,
        oldparent=oldparent,
        child=child,
        ns=ns,
        utc=utc,
        command=command,
        old=fake_old,
        witness=fake_witness,
        source=old_source,
        tmp=tmp_path,
    )


def test_sealed_held_exits_allow_nonzero_owner_without_process_calls(tmp_path):
    f = fixture(tmp_path)
    evidence = module.backend.Digests()
    result = f.ns["_verify_witness"](f.root, evidence)
    assert result["observations"]["child"]["exit_code"] == 0
    assert result["observations"]["owner"]["exit_code"] == 1
    assert str(f.oldroot / "events.jsonl") in evidence.paths
    evidence.final_guard()


@pytest.mark.parametrize(
    "bad",
    [
        "failed_witness",
        "wrong_interval",
        "later_launch",
        "partial_events",
        "owner_first",
        "child_nonzero",
        "reused_child_pid",
        "changed_identity",
        "filetime_mismatch",
        "not_signalled",
        "changed_payload",
        "changed_current_events",
    ],
)
def test_bad_witness_or_handoff_boundary_rejected(tmp_path, bad):
    f = fixture(tmp_path, bad)
    with pytest.raises(ValueError):
        f.ns["_verify_witness"](f.root, module.backend.Digests())


def old_edge(f, bad=None):
    """Small sealed metadata fixtures; candidate files are only hashed, never decoded."""
    binding = {"inputs": {"full_frame_range": [49800, 348600]}}
    for root, stop in ((f.oldparent, 66400), (f.child, 76360)):
        (root / "candidate.npz").write_bytes(b"fixture saved candidate hash, no arrays")
        write(
            root / "summary.json",
            dict(
                schema="long_continuation_phase_v1",
                technical_completion=True,
                processed_interval=[49800, stop],
                logical_initialization_count=1,
                external_pose_reset_count=0,
                active_beam_count=6,
            ),
        )
        standard(
            root,
            "continuation.provenance.json",
            "three_body_long_continuation",
            outputs={
                "candidate": entry(root / "candidate.npz"),
                "summary": entry(root / "summary.json"),
            },
            params=dict(through_frame=stop, bindings=binding),
            started=f.utc(63 if root == f.child else 0),
            finished=f.utc(
                181
                if bad == "late_child_seal" and root == f.child
                else 170
                if root == f.child
                else 10
            ),
        )
    bootstrap = f.oldroot / "bootstrap"
    bootstrap.mkdir()
    write(
        bootstrap / "summary.json",
        dict(
            status="ok",
            run=str(f.oldparent),
            candidate_sha256=module.old.sha256(f.oldparent / "candidate.npz"),
        ),
    )
    code = {str(f.source): module.old.sha256(f.source)}
    bootstrap_outputs = {"summary": entry(bootstrap / "summary.json")}
    reg = f.child.parent / "registration"
    reg.mkdir()
    reg_outputs = {}
    for root, outputs in ((bootstrap, bootstrap_outputs), (reg, reg_outputs)):
        path = root / "code_snapshot" / f.source.relative_to(f.tmp)
        path.parent.mkdir(parents=True)
        path.write_bytes(f.source.read_bytes())
        outputs[path.relative_to(root).as_posix()] = entry(path)
    snap = f.oldroot / "code_snapshot" / f.source.relative_to(f.tmp)
    snap.parent.mkdir()
    snap.write_bytes(f.source.read_bytes())
    standard(
        bootstrap,
        "audit.provenance.json",
        "long_supervised_bootstrap_audit",
        inputs={
            str(f.oldparent / name): entry(f.oldparent / name)
            for name in ("candidate.npz", "continuation.provenance.json")
        },
        outputs=bootstrap_outputs,
    )
    evidence = module.backend.Digests()
    lineage = module.operation.bind_parent_audit(f.oldparent, bootstrap, evidence)
    plan = dict(
        schema="long_continuation_registered_phase_v1",
        next_native_interval=[66400, 76360],
        output_interval=[49800, 76360],
        parent_phase=str(f.oldparent),
        expected_output=str(f.child),
        frozen_input_and_code_binding=binding,
    )
    write(reg / "plan.json", plan)
    write(
        reg / "execution.json",
        dict(
            command=f.command,
            independent_parent_audit={
                k: lineage[k]
                for k in ("root", "seal_sha256", "summary_sha256", "parent_candidate_sha256")
            },
            source_hashes=code,
        ),
    )
    reg_outputs.update(plan=entry(reg / "plan.json"), execution=entry(reg / "execution.json"))
    standard(
        reg,
        "registration.provenance.json",
        "register_long_supervised_phase",
        outputs=reg_outputs,
        started=f.utc(30),
        finished=f.utc(65 if bad == "late_registration" else 50),
    )
    policy = dict(
        schema="long_remaining_operational_supervisor_v1",
        initial_parent=str(f.oldparent),
        initial_stop=66400,
        target_interval=[49800, 348600],
        phase_intervals=module.old.phase_ranges(),
        source_hashes=code,
        fixed_bindings=binding,
        **{
            k: False
            for k in (
                "detector_started",
                "new_external_pose_seed",
                "failed_phase_retry",
                "lock_removed",
                "scheduling_changes_scientific_protocol",
            )
        },
    )
    write(f.oldroot / "supervisor_plan.json", policy)
    f.ns.update(
        POLICY_SHA=module.old.sha256(f.oldroot / "supervisor_plan.json"),
        BOOTSTRAP_SHA=module.old.sha256(bootstrap / "audit.provenance.json"),
    )
    f.old.command_for = lambda *args: f.command
    if bad == "bootstrap_candidate":
        (f.oldparent / "candidate.npz").write_bytes(b"changed old saved candidate")
    if bad == "snapshot":
        snap.write_bytes(b"changed old snapshot")
    return binding


def test_full_metadata_handoff_retains_old_unfinished_record(tmp_path):
    f = fixture(tmp_path)
    binding = old_edge(f)
    path = f.child.parent / "operation.provenance.json"
    path.write_bytes(b'{"status":"failed","reason":"fixture retained, not promoted"}')
    before = path.read_bytes()
    result = f.ns["verify_handoff"](f.root, module.backend.Digests())
    assert result["parent"] == f.child and result["binding"] == binding
    assert path.read_bytes() == before
    assert str(path) in result["lineage"]["old_optional_artifacts_preserved"]
    assert result["lineage"]["old_success_claim_created"] is False


@pytest.mark.parametrize(
    "bad", ["late_child_seal", "late_registration", "bootstrap_candidate", "snapshot"]
)
def test_legacy_registration_bootstrap_child_edge_rejections(tmp_path, bad):
    f = fixture(tmp_path)
    old_edge(f, bad)
    with pytest.raises(ValueError):
        f.ns["verify_handoff"](f.root, module.backend.Digests())


@pytest.mark.parametrize("fails", [False, True])
def test_bootstrap_new_stage_receipts_and_backend_failure(tmp_path, fails):
    f = fixture(tmp_path)
    old_edge(f)
    output, cache = tmp_path / "new_bootstrap", tmp_path / "receipts"
    code = {str(f.source): module.old.sha256(f.source)}
    calls = []

    class Store:
        def __init__(self, root):
            self.root = root
            root.mkdir()
            path = root / "receipt.json"
            path.write_bytes(b"fixture receipt")
            self.evidence_hashes = {str(path): module.old.sha256(path)}
            self.uses = []

        def final_guard(self):
            module.old.unchanged(self.evidence_hashes)

    def verify(parent, plan, evidence, *, receipt_store):
        calls.append((parent, copy.deepcopy(plan)))
        assert parent == f.child and plan["next_native_interval"] == [66400, 76360]
        receipt_store.uses.append({"cache_hit": True})
        if fails:
            raise ValueError("fixture full-state audit failed")
        evidence.add(parent / "candidate.npz")
        evidence.add(parent / "continuation.provenance.json")
        return dict(
            status="ok",
            run=str(parent),
            frame_range=[49800, 76360],
            candidate_sha256=module.old.sha256(parent / "candidate.npz"),
            all_39_checkpoint_arrays_verified=True,
            all_21_candidate_arrays_exact_backtrace={"fixture": True},
        )

    f.ns.update(
        ReceiptStore=Store,
        backend=SimpleNamespace(Digests=module.backend.Digests, verify_phase=verify),
    )
    f.old.sources = lambda: code
    f.old.snapshot = FunctionType(
        module.old.snapshot.__code__, {**vars(module.old), "REPO": tmp_path}
    )
    if fails:
        with pytest.raises(ValueError, match="full-state audit failed"):
            f.ns["audit_handoff"](f.root, output, code, receipt_root=cache)
    else:
        assert f.ns["audit_handoff"](f.root, output, code, receipt_root=cache) == output
        receipt = json.loads((output / "audit.provenance.json").read_text())
        assert receipt["stage"] == "long_receipted_handoff_bootstrap_v1"
        assert receipt["status"] == "ok"
        for name in ("candidate.npz", "continuation.provenance.json"):
            assert str(f.child / name) in receipt["inputs"]
        for item in receipt["outputs"].values():
            assert module.old.sha256(item["path"]) == item["sha256"]
        module.operation.bind_parent_audit(f.child, output, module.backend.Digests())
    assert len(calls) == 1
    if fails:
        assert json.loads((output / "audit.provenance.json").read_text())["status"] == "failed"
        assert not (output / "summary.json").exists()

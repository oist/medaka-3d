"""Scheduling and promotion regressions; no process, tracker, payload or GPU work."""

import sys
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_receipted_child_supervisor as s


def fixture(monkeypatch, tmp_path, *, k=27, failure=None, stop_mode=None):
    parent, old, stage = [tmp_path / x for x in ("parent", "old", "stage")]
    for path in (parent, old, stage):
        path.mkdir()
    source = tmp_path / "fixture_source.py"
    source.write_bytes(b"# synthetic scheduling fixture\n")
    code = {str(source): s.m.sha(source)}
    witness = tmp_path / "witness.json"
    witness.write_bytes(b"synthetic held exit; real verifier replaced\n")
    old_seal = old / "supervisor.provenance.json"
    old_seal.write_bytes(b"synthetic completed prefix\n")
    for name in ("continuation.provenance.json", "operation.provenance.json"):
        (parent / name).write_bytes(b"synthetic old parent authority\n")
    counts = s.chain.prefix_counts(k)
    binding = dict(code_sha256=code, inputs=dict(input_stage_root=str(stage)))
    args = SimpleNamespace(
        old_supervisor=old,
        old_supervisor_sha256=s.m.sha(old_seal),
        exit_witness=witness,
        exit_witness_sha256=s.m.sha(witness),
        receipt_root=tmp_path / "receipts",
        output=tmp_path / "new_owner",
        execute=True,
    )
    lock, calls = tmp_path / "new.lock", []
    monkeypatch.setattr(s, "SHARED_LOCK", lock)
    monkeypatch.setattr(s.m, "ADDED_IO", ())
    monkeypatch.setattr(s.m, "execution_binding", lambda legacy, code: legacy)
    monkeypatch.setattr(s.m.original.torch.cuda, "is_available", lambda: pytest.fail("no GPU"))

    def sources(evidence):
        evidence.add(source, code[str(source)])
        return code

    monkeypatch.setattr(s, "sources", sources)

    def prefix(root, digest, reference, evidence):
        calls.append("prefix")
        assert not lock.exists() and not args.output.exists()
        evidence.add(old_seal, digest)
        evidence.add(witness, reference["sha256"])
        if failure == "prefix":
            raise ValueError("bad held prefix")
        return dict(
            prefix_counts=counts,
            final_run=parent,
            binding=binding,
            unused_registration=None,
            owner_exit=dict(exit_time_utc="2020-01-01T00:00:00+00:00"),
            last_audit_root=parent,
            last_audit_name="operation.provenance.json",
        )

    monkeypatch.setattr(s.chain, "load_completed_prefix", prefix)

    def request():
        (args.output / s.current.STOP_NAME).write_bytes(b"exact stop bytes\x00\xff\n")

    def wait(stage, first, stop, emit, check_stop, **kwargs):
        calls.append("ready")
        if stop_mode == "before" or (stop_mode == "after" and "operation" in calls):
            request()
        return check_stop()

    monkeypatch.setattr(s.current, "wait_for_ready", wait)

    def execute(params, *, emit):
        calls.append("operation")
        assert lock.exists() and not params.output.exists()
        assert params.parent_seal_sha256 == s.m.sha(params.parent / "continuation.provenance.json")
        assert params.parent_audit_sha256 == s.m.sha(params.parent_audit_seal)
        root = params.output
        root.mkdir()
        run, reg, audit_root = [root / x for x in ("tracking", "registration", "audit")]
        for path in (run, reg, audit_root):
            path.mkdir()
        restore = counts["checkpoint_restores"] + calls.count("operation")
        through = params.through_frame
        result = dict(
            status="ok",
            run=str(run),
            parent=str(params.parent),
            frame_range=[49800, through],
            new_interval=[through - (3320 if through == 348600 else 9960), through],
            all_39_checkpoint_arrays_verified=True,
            all_21_candidate_arrays_exact_backtrace={str(i): {} for i in range(21)},
            history={"active_beam_count": 6},
            logical_initialization_count=1,
            checkpoint_restore_count=restore,
            external_pose_reset_count=0,
            full_sequence_complete=through == 348600,
            prefix_backtrace_provisional=through != 348600,
            independent_phase_audit_complete=True,
        )
        with s.m.original.provenance(s.operation.STAGE, root, name="operation") as record:
            if failure == "operation":
                raise ValueError("synthetic operation failure")
            with s.m.original.provenance(s.m.REGISTRATION_STAGE, reg, name="registration") as lower:
                lower.param("all_input_source_snapshot_output_sha_unchanged", True)
            result["registration_sha256"] = s.m.sha(reg / "registration.provenance.json")
            with s.m.original.provenance(s.m.STAGE, run, name="continuation") as lower:
                lower.param("all_input_source_snapshot_output_sha_unchanged", True)
                lower.param("bindings", binding)
                lower.param("resume_phase", str(params.parent))
                lower.param("through_frame", through)
            if failure == "audit":
                result["independent_phase_audit_complete"] = False
            with s.m.original.provenance(s.auditor.STAGE, audit_root, name="audit") as lower:
                s.child.save_json(audit_root / "summary.json", result, lower, "summary")
                lower.param("all_input_source_snapshot_output_sha_unchanged", True)
            for role, path in (
                ("registration_seal", reg / "registration.provenance.json"),
                ("tracking_seal", run / "continuation.provenance.json"),
                ("audit_seal", audit_root / "audit.provenance.json"),
                ("audit/summary.json", audit_root / "summary.json"),
            ):
                record.output(role, path, sha256=s.m.sha(path))
            s.child.save_json(
                root / "summary.json",
                dict(
                    status="ok",
                    completed_run=str(run),
                    independent_audit=str(audit_root),
                    child_started=True,
                    parent_promoted=False,
                    operation_success_requires_successful_seal=True,
                    automatic_retry=False,
                    execution_resource_owner_external=True,
                ),
                record,
                "summary",
            )
            record.param("all_input_source_snapshot_output_sha_unchanged", True)
            record.param("tracking_child_exit_code", 0)
        if failure == "returned":
            result["new_interval"] = [0, through]
        if failure == "source":
            source.write_bytes(b"source changed after operation\n")
        if stop_mode == "during":
            request()
        emit("phase_checks_complete", output=str(run))
        return result

    monkeypatch.setattr(s.operation, "execute", execute)
    return args, parent, lock, calls


def test_all_dynamic_boundaries_cover_the_original_schedule():
    for k in range(28):
        count = s.chain.prefix_counts(k)
        rows = s.remaining_phases(count["through"])
        assert len(rows) == 28 - k and rows[-1] == [345280, 348600]
        assert sum(b - a for a, b in rows) + count["native_frames"] == 298800
        assert count["checkpoint_restores"] + len(rows) == 32
        assert all(a[1] == b[0] for a, b in pairwise(rows))
    for invalid in (76361, 348600, True, 136120.0):
        with pytest.raises(ValueError):
            s.remaining_phases(invalid)


def test_final_operation_seal_then_parent_promotion(monkeypatch, tmp_path):
    args, _, lock, calls = fixture(monkeypatch, tmp_path)
    s.supervise(args)
    summary = s.m.read(args.output / "summary.json")
    assert calls == ["prefix", "ready", "operation"] and lock.exists()
    assert summary["checkpoint_restore_count"] == 32 and summary["tracking_processes"] == 33
    assert summary["consumed_chunk_count"] == 180 and summary["status"] == "complete"
    assert summary["final_audit_complete"] is False and summary["independent_ID_truth"] is False
    assert summary["phases"][0]["interval"] == [345280, 348600]
    record = s.m.read(args.output / "supervisor.provenance.json")
    assert record["status"] == "ok" and record["stage"] == s.STAGE
    for item in record["outputs"].values():
        assert s.m.sha(item["path"]) == item["sha256"]
    with pytest.raises(ValueError, match="exclusive"):
        s.supervise(args)


@pytest.mark.parametrize("when,operations", [("before", 0), ("during", 1), ("after", 1)])
def test_stop_retains_exact_bytes_at_audited_boundary(monkeypatch, tmp_path, when, operations):
    args, parent, _, calls = fixture(monkeypatch, tmp_path, k=6, stop_mode=when)
    s.supervise(args)
    summary = s.m.read(args.output / "summary.json")
    assert calls.count("operation") == operations
    assert summary["through"] == 136120 + operations * 9960
    assert summary["checkpoint_restore_count"] == 10 + operations
    assert summary["status"] == "stopped_after_completed_phase"
    assert summary["stop_request"]["child_interrupted"] is False
    assert (args.output / "honored_stop_request.bin").read_bytes() == b"exact stop bytes\x00\xff\n"
    if not operations:
        assert summary["final_run"] == str(parent)


@pytest.mark.parametrize("failure", ["operation", "audit", "returned"])
def test_failed_operation_or_audit_never_promotes(monkeypatch, tmp_path, failure):
    args, parent, lock, calls = fixture(monkeypatch, tmp_path, failure=failure)
    with pytest.raises(ValueError):
        s.supervise(args)
    summary = s.m.read(args.output / "summary.json")
    assert summary["status"] == "failed" and summary["final_run"] == str(parent)
    assert summary["phases"] == [] and summary["incomplete_phase"] is not None
    assert calls.count("operation") == 1 and lock.exists()
    assert s.m.read(args.output / "supervisor.provenance.json")["status"] == "failed"


def test_invalid_old_exit_never_acquires_lock(monkeypatch, tmp_path):
    args, _, lock, calls = fixture(monkeypatch, tmp_path, failure="prefix")
    with pytest.raises(ValueError, match="held prefix"):
        s.supervise(args)
    assert calls == ["prefix"] and not lock.exists() and not args.output.exists()


def test_source_mutation_prevents_successful_owner_seal(monkeypatch, tmp_path):
    args, _, _, _ = fixture(monkeypatch, tmp_path, failure="source")
    with pytest.raises(ValueError):
        s.supervise(args)
    assert s.m.read(args.output / "supervisor.provenance.json")["status"] == "failed"


def test_unreviewed_source_pin_cannot_execute(monkeypatch):
    monkeypatch.setattr(s, "FINAL_CHAIN_SHA", None)
    with pytest.raises(ValueError, match="review pin"):
        s.sources(s.m.Evidence())

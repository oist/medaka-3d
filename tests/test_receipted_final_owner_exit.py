"""Small owner-exit contracts and fake held-handle lifecycle; no OS attachment."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import receipted_final_owner_exit as p


def fixture_records():
    created = 133000000000000000
    identity = dict(
        pid=11840,
        creation_filetime_100ns=created,
        creation_time_utc=p.frozen.filetime_utc(created),
        image_path=str(Path(sys.executable).resolve()),
        attached_at=p.frozen.filetime_utc(created + 20_000_000),
        access_mask=0x00101000,
        live_when_handle_acquired=True,
    )
    binding = dict(lock=dict(pid=11840, created_at=p.frozen.filetime_utc(created + 10_000_000)))
    header = dict(
        stage=p.OWNER_STAGE,
        status="ok",
        finished_at=p.frozen.filetime_utc(created + 30_000_000),
        params=dict(
            full_sequence_complete=True, all_input_source_snapshot_output_sha_unchanged=True
        ),
    )
    exit_info = dict(
        **identity,
        exit_code=0,
        exit_filetime_100ns=created + 40_000_000,
        exit_time_utc=p.frozen.filetime_utc(created + 40_000_000),
        exit_observed_at=p.frozen.filetime_utc(created + 50_000_000),
        wait_signalled=True,
        exit_code_success=True,
    )
    return identity, binding, header, exit_info, p.frozen.filetime_utc(created + 60_000_000)


def test_real_exit_timestamp_and_later_observation_are_distinct():
    identity, binding, header, ended, audit_time = fixture_records()
    p.validate_identity(identity, binding, allowed_images={Path(sys.executable).resolve()})
    p.validate_exit(ended, identity, header, audit_started_at=audit_time)
    assert ended["exit_time_utc"] != ended["exit_observed_at"]


@pytest.mark.parametrize(
    "fault",
    [
        "bool_exit",
        "nonzero_exit",
        "pid",
        "creation",
        "unsignalled",
        "early_exit",
        "future_observation",
        "unfinished",
        "wrong_stage",
    ],
)
def test_bad_owner_evidence_rejected_without_any_process_query(fault):
    identity, _, header, ended, audit_time = fixture_records()
    if fault == "bool_exit":
        ended["exit_code"] = False
    elif fault == "nonzero_exit":
        ended["exit_code"] = 1
    elif fault == "pid":
        ended["pid"] += 1
    elif fault == "creation":
        ended["creation_filetime_100ns"] += 1
    elif fault == "unsignalled":
        ended["wait_signalled"] = False
    elif fault == "early_exit":
        ended["exit_filetime_100ns"] -= 20_000_000
        ended["exit_time_utc"] = p.frozen.filetime_utc(ended["exit_filetime_100ns"])
    elif fault == "future_observation":
        audit_time = identity["attached_at"]
    elif fault == "unfinished":
        header["params"]["full_sequence_complete"] = False
    else:
        header["stage"] = "old_handoff_owner"
    with pytest.raises(ValueError):
        p.validate_exit(ended, identity, header, audit_started_at=audit_time)


def test_pid_reuse_after_lock_and_terminate_rights_rejected():
    identity, binding, _, _, _ = fixture_records()
    allowed = {Path(sys.executable).resolve()}
    late = copy.deepcopy(binding)
    late["lock"]["created_at"] = p.frozen.filetime_utc(identity["creation_filetime_100ns"] - 1)
    with pytest.raises(ValueError):
        p.validate_identity(identity, late, allowed_images=allowed)
    identity["access_mask"] |= 1
    with pytest.raises(ValueError):
        p.validate_identity(identity, binding, allowed_images=allowed)


def test_partial_or_boolean_summary_cannot_satisfy_full_scope():
    complete = dict(
        schema=p.OWNER_STAGE,
        status="complete",
        through=348600,
        full_sequence_complete=True,
        prefix_backtrace_provisional=False,
        logical_initialization_count=1,
        checkpoint_restore_count=32,
        external_pose_reset_count=0,
        tracking_processes=33,
        consumed_chunk_count=180,
        incomplete_phase=None,
    )
    p.validate_summary(complete)
    for name, wrong in (
        ("through", 156040),
        ("logical_initialization_count", True),
        ("checkpoint_restore_count", 31),
        ("incomplete_phase", "active"),
    ):
        with pytest.raises(ValueError):
            p.validate_summary(dict(complete, **{name: wrong}))


def test_attach_failure_preserves_failed_receipt_and_source_snapshots(tmp_path, monkeypatch):
    inputs = {str(Path(x).resolve()): p.sha(x) for x in (p.__file__, p.frozen.__file__)}
    binding = dict(supervisor_root=str(tmp_path), lock=dict(pid=11840), inputs=inputs)
    monkeypatch.setattr(p, "WORK", tmp_path)
    monkeypatch.setattr(p, "metadata_binding", lambda root, pid: binding)
    calls = []

    def failed_factory(pid):
        calls.append(pid)
        raise OSError("synthetic access failure")

    out = tmp_path / "receipt"
    with pytest.raises(OSError, match="synthetic"):
        p.capture(tmp_path, 11840, out, process_factory=failed_factory)
    record = json.loads((out / "owner_exit.provenance.json").read_bytes())
    assert calls == [11840] and record["status"] == "failed"
    assert record["error"]["type"] == "OSError"
    assert "summary.json" not in record["outputs"]
    assert len(list(out.glob("*.py"))) == 2


def test_invalid_binding_never_opens_handle_or_creates_output(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError("invalid source binding")

    monkeypatch.setattr(p, "metadata_binding", reject)

    def never(*args):
        pytest.fail("process factory reached before ownership qualification")

    out = tmp_path / "receipt"
    with pytest.raises(ValueError, match="invalid source"):
        p.capture(tmp_path, 11840, out, process_factory=never)
    assert not out.exists()


@pytest.mark.parametrize("consumer_fault", [None, "summary", "missing_snapshot"])
def test_fake_held_exit_roundtrip_and_contradictory_receipt_rejection(
    tmp_path, monkeypatch, consumer_fault
):
    identity, binding, header, ended, _ = fixture_records()
    created = identity["creation_filetime_100ns"]
    root = tmp_path / "supervisor"
    root.mkdir()
    real_work = p.WORK

    def fixture_local(value, *, exists=True):
        path = Path(value).resolve(strict=exists)
        assert path.is_relative_to(tmp_path) or path.is_relative_to(real_work)
        return path

    monkeypatch.setattr(p, "local", fixture_local)
    monkeypatch.setattr(p, "ROOT", root)
    clock = iter(
        p.frozen.filetime_utc(created + delta) for delta in (15_000_000, 60_000_000, 70_000_000)
    )
    monkeypatch.setattr(p, "now", lambda: next(clock))

    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")
        return dict(path=str(path), sha256=p.sha(path), size=path.stat().st_size)

    summary = dict(
        schema=p.OWNER_STAGE,
        status="complete",
        through=348600,
        full_sequence_complete=True,
        prefix_backtrace_provisional=False,
        logical_initialization_count=1,
        checkpoint_restore_count=32,
        external_pose_reset_count=0,
        tracking_processes=33,
        consumed_chunk_count=180,
        incomplete_phase=None,
    )
    header["outputs"] = {
        "summary": write(root / "summary.json", summary),
        "lock": write(root / "supervisor.lock", binding["lock"]),
        "plan": write(root / "supervisor_plan.json", {"synthetic": True}),
    }
    write(root / "supervisor.provenance.json", header)
    inputs = {
        str(Path(x).resolve()): p.sha(x)
        for x in (
            p.__file__,
            p.frozen.__file__,
            p.SUPERVISOR,
            root / "supervisor.lock",
            root / "supervisor_plan.json",
        )
    }
    binding.update(supervisor_root=str(root), inputs=inputs)
    monkeypatch.setattr(p, "metadata_binding", lambda selected, pid: binding)
    closed = []

    class FakeProcess:
        def __init__(self, pid):
            assert pid == identity["pid"]
            self.identity = identity

        def signalled(self):
            return True  # Exit immediately after the recorded live acquisition.

        def exit_info(self):
            return ended

        def close(self):
            closed.append(identity["pid"])

    out = tmp_path / "receipt"
    p.capture(root, identity["pid"], out, process_factory=FakeProcess)
    assert closed == [identity["pid"]]
    receipt_path = out / "owner_exit.provenance.json"
    record = json.loads(receipt_path.read_bytes())
    assert record["status"] == "ok" and len(record["outputs"]) == 7
    if consumer_fault == "summary":
        path = out / "summary.json"
        changed = json.loads(path.read_bytes())
        changed["held_exit_code"] = False
        write(path, changed)
        record["outputs"]["summary.json"].update(sha256=p.sha(path), size_bytes=path.stat().st_size)
        write(receipt_path, record)
    elif consumer_fault == "missing_snapshot":
        record["outputs"].pop(Path(p.__file__).name)
        write(receipt_path, record)

    class Evidence:
        def add(self, path, expected):
            assert p.sha(path) == expected

    def consume():
        return p.verify_final_owner_exit(
            receipt_path,
            p.sha(receipt_path),
            root,
            header,
            audit_started_at=p.frozen.filetime_utc(created + 80_000_000),
            evidence=Evidence(),
        )

    if consumer_fault:
        with pytest.raises(ValueError):
            consume()
    else:
        result = consume()
        assert result["exit_code"] == 0 and result["final_audit_complete"] is False

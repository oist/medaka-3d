"""Synthetic process/receipt tests; real process inspection and launch are forbidden."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import wait_and_evaluate_sparse_five_minutes as m


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "EXECUTION_READY", False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(m, "WORK", work)
    monkeypatch.setattr(m, "REPO", tmp_path)
    for name in ("ROOT", "GPU_LAUNCH", "TRACKING", "CONTROL", "BODY", "IDENTITY"):
        monkeypatch.setattr(m, name, work / getattr(m, name).name)
    monkeypatch.setattr(m, "RUNNER", work / m.RUNNER.name)
    source = work / "synthetic_supervisor.py"
    source.write_bytes(b"SYNTHETIC ONLY")
    monkeypatch.setattr(m, "__file__", str(source))

    def forbidden(*args, **kwargs):
        pytest.fail("REAL PROCESS/WAIT MUST NOT BE REACHED")

    monkeypatch.setattr(m.psutil, "Process", forbidden)
    monkeypatch.setattr(m.subprocess, "Popen", forbidden)
    monkeypatch.setattr(m.time, "sleep", forbidden)
    return work


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value).encode()
    path.write_bytes(data)
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def test_disabled_or_repeated_supervisor_never_reads_or_starts_real_inputs(monkeypatch):
    monkeypatch.setattr(m, "source_guard", lambda: pytest.fail("INPUT GUARD REACHED"))
    with pytest.raises(ValueError, match="disabled"):
        m.main(SimpleNamespace(execute=True))
    monkeypatch.setattr(m, "EXECUTION_READY", True)
    m.ROOT.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        m.main(SimpleNamespace(execute=True))


@pytest.mark.parametrize("reused,wrong_command", [(True, False), (False, True), (False, False)])
def test_owned_pid_birth_and_command_checked_without_signal(monkeypatch, reused, wrong_command):
    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return m.OWNED[self.pid] + (1 if reused else 0)

        def cmdline(self):
            return ["python", "wrong.py" if wrong_command else str(m.RUNNER)]

    monkeypatch.setattr(m.psutil, "Process", FakeProcess)
    if reused or wrong_command:
        with pytest.raises(ValueError, match=r"reused|command"):
            m.observe_owned()
    else:
        result = m.observe_owned()
        assert [r["pid"] for r in result] == list(m.OWNED)
        assert all(
            r["state"] == "live" and r["observed_birth"] == m.OWNED[r["pid"]] for r in result
        )


def process_rows(worker, parent):
    return [dict(pid=m.WORKER_PID, state=worker), dict(pid=m.PARENT_PID, state=parent)]


def test_wait_requires_both_processes_gone_and_complete_success_exit(monkeypatch):
    m.ROOT.mkdir()
    started = dict(started_at="synthetic start")
    receipt = dict(
        pid=m.PARENT_PID, exit_code=0, started_at=started["started_at"], automatic_restart=False
    )
    sequence = iter(
        [process_rows("live", "live"), process_rows("gone", "live"), process_rows("gone", "gone")]
    )
    monkeypatch.setattr(m, "observe_owned", lambda: next(sequence))
    sleeps = []
    monkeypatch.setattr(m.time, "sleep", sleeps.append)
    write_json(m.GPU_LAUNCH / "child_exit.json", receipt)
    m.wait_for_gpu_exit(started)
    assert sleeps == [5, 5]
    saved = json.loads((m.ROOT / "tracking_process_exit.json").read_bytes())
    assert saved["launch_exit"]["exit_code"] == 0
    assert saved["worker_exit_code_observed"] is False


@pytest.mark.parametrize("fault", ["nonzero", "wrong_pid", "other_launch", "missing"])
def test_unsuccessful_or_missing_exit_receipt_never_releases_queue(monkeypatch, fault):
    m.ROOT.mkdir()
    monkeypatch.setattr(m, "observe_owned", lambda: process_rows("gone", "gone"))
    tick = iter([0, 31])
    monkeypatch.setattr(m.time, "monotonic", lambda: next(tick))
    monkeypatch.setattr(m.time, "sleep", lambda seconds: None)
    started = dict(started_at="synthetic start")
    receipt = dict(
        pid=m.PARENT_PID, exit_code=0, started_at=started["started_at"], automatic_restart=False
    )
    if fault == "nonzero":
        receipt["exit_code"] = 2
    elif fault == "wrong_pid":
        receipt["pid"] = 999
    elif fault == "other_launch":
        receipt["started_at"] = "other"
    if fault != "missing":
        write_json(m.GPU_LAUNCH / "child_exit.json", receipt)
    with pytest.raises(ValueError):
        m.wait_for_gpu_exit(started)
    assert not (m.ROOT / "tracking_process_exit.json").exists()


def tracking_fixture():
    summary = dict(
        schema=m.TRACKING.name,
        status="ok",
        scientific_interval=[66400, 116200],
        process_interval=[78020, 116200],
        retained_interval=[49800, 116200],
        native_frames=49800,
        process_frames=38180,
        initialize_count=1,
        restore_count=6,
        external_pose_resets=0,
        new_begin_calls=0,
        finish_calls=1,
        all39_roundtrip=True,
        prior_history_prefix_unchanged=True,
        finish_matches_checkpoint_backtrace=True,
        checkpoints=[dict(path=str(m.TRACKING / "checkpoints/through_116200"))],
    )
    return summary


@pytest.mark.parametrize(
    "fault",
    [None, "horizon", "reset", "initializations", "prefix", "failed", "member_sha", "source"],
)
def test_tracking_small_seal_and_lifecycle_gate_without_payloads(fault):
    m.ROOT.mkdir()
    summary = tracking_fixture()
    if fault == "horizon":
        summary["scientific_interval"][-1] -= 1
    elif fault == "reset":
        summary["new_begin_calls"] = 1
    elif fault == "initializations":
        summary["initialize_count"] = True  # bool is not an integer counter.
    elif fault == "prefix":
        summary["prior_history_prefix_unchanged"] = False
    member = write_json(m.TRACKING / "summary.json", summary)
    if fault == "member_sha":
        member["sha256"] = "f" * 64
    seal = dict(
        stage=m.TRACKING.name,
        status="failed" if fault == "failed" else "ok",
        inputs={str(m.RUNNER): "f" * 64 if fault == "source" else m.SOURCES[m.RUNNER.name]},
        outputs={"summary.json": member},
    )
    reference = write_json(m.TRACKING / "tracking.provenance.json", seal)
    if fault is None:
        assert m.tracking_completion() == reference
        assert set(p.name for p in m.TRACKING.iterdir()) == {
            "summary.json",
            "tracking.provenance.json",
        }
    else:
        with pytest.raises(ValueError):
            m.tracking_completion()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_cpu_stage_is_hidden_logged_once_and_records_actual_exit(monkeypatch, exit_code):
    m.ROOT.mkdir()
    monkeypatch.setattr(m, "source_guard", lambda: None)
    calls = []

    class FakeChild:
        pid = 777

        def wait(self, timeout):
            assert timeout <= 30
            return exit_code

    def popen(command, **kwargs):
        assert kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW
        assert kwargs["cwd"] == m.REPO
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert Path(kwargs["stdout"].name).parent == m.ROOT
        assert Path(kwargs["stderr"].name).parent == m.ROOT
        assert "--execute" not in command
        calls.append(command)
        return FakeChild()

    monkeypatch.setattr(m.subprocess, "Popen", popen)
    monkeypatch.setattr(m.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 123.0))
    args = (
        "comparison",
        "compare_sparse_instance_five_minutes.py",
        m.BODY,
        ["--variant-seal-sha256", "a" * 64],
    )
    if exit_code:
        with pytest.raises(ValueError, match="no retry"):
            m.stage(*args)
    else:
        m.stage(*args)
    assert len(calls) == 1
    assert calls[0][-2:] == ["--output", str(m.BODY)]
    assert json.loads((m.ROOT / "comparison_exit.json").read_bytes())["exit_code"] == exit_code


@pytest.mark.parametrize("failure", [None, "tracking", "comparison", "comparison_seal", "identity"])
def test_queue_order_single_calls_and_failure_stop_are_synthetic(monkeypatch, failure):
    monkeypatch.setattr(m, "EXECUTION_READY", True)
    monkeypatch.setattr(m, "source_guard", lambda: None)
    monkeypatch.setattr(m, "launch_binding", lambda: ({}, {"synthetic": True}))
    monkeypatch.setattr(m, "successful_seal", lambda *args: ({}, {}))
    events = []

    def checkpoint(name, result=None):
        events.append(name)
        if failure == name:
            raise ValueError("synthetic failure")
        return {} if result is None else result

    monkeypatch.setattr(m, "wait_for_gpu_exit", lambda _: checkpoint("wait"))
    monkeypatch.setattr(
        m, "tracking_completion", lambda: checkpoint("tracking", {"sha256": "t" * 64})
    )
    monkeypatch.setattr(
        m, "comparison_completion", lambda _: checkpoint("comparison_seal", {"sha256": "c" * 64})
    )
    monkeypatch.setattr(
        m, "identity_completion", lambda _: checkpoint("identity_seal", {"sha256": "i" * 64})
    )

    def stage(name, script, output, arguments):
        if name == "comparison":
            assert arguments == [
                "--control-seal-sha256",
                m.CONTROL_SHA,
                "--variant-seal-sha256",
                "t" * 64,
            ]
            assert output == m.BODY
        else:
            assert arguments == ["--comparison-sha", "c" * 64]
            assert (
                script == "execute_sparse_instance_five_minute_identity_beam.py"
                and output == m.IDENTITY
            )
        checkpoint(name)

    monkeypatch.setattr(m, "stage", stage)
    if failure:
        with pytest.raises(ValueError, match="synthetic failure"):
            m.main(SimpleNamespace(execute=True))
        assert events[-1] == failure
    else:
        assert m.main(SimpleNamespace(execute=True)) == 0
        assert events == [
            "wait",
            "tracking",
            "comparison",
            "comparison_seal",
            "identity",
            "identity_seal",
        ]
    saved = json.loads((m.ROOT / "exit.json").read_bytes())
    assert saved["status"] == ("failed" if failure else "ok")
    assert saved["tracking_mutated"] is False and saved["no_retry"] is True

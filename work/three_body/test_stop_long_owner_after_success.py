"""No real signals: gate and failure sequencing for the owned-owner stop helper."""

import json
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stop_long_owner_after_success as control

FROZEN_STOP = control.old.stop_owned_child


@pytest.fixture
def case(tmp_path, monkeypatch):
    events = []
    source = tmp_path / "source.txt"
    source.write_bytes(b"unchanged evidence")
    output = tmp_path / "stop"
    state = SimpleNamespace(ended=False, closed=False, identity_bad=False, late=False)
    identity = dict(creation_filetime_100ns=1234, image_path=str(source), pid=9876)

    def preflight(path):
        events.append("preflight")
        return {"owner_pid": 9876}, identity.copy(), {str(source): control.witness.sha(source)}

    def launch_gate(binding):
        events.append("launch_gate")
        if state.late and state.ended:
            raise ValueError("later launch")
        return b"event snapshot\n"

    class Process:
        def __init__(self, pid):
            events.append("open")
            assert pid == 9876
            self.pid = pid
            self.identity = identity.copy()
            if state.identity_bad:
                self.identity["creation_filetime_100ns"] += 1

        def signalled(self):
            return state.ended

        def exit_info(self):
            assert state.ended
            return dict(exit_code=1, pid=self.pid)

        def close(self):
            events.append("close")
            state.closed = True

    def stop(owner, info):
        events.append("stop")
        assert not state.closed and owner.poll() is None
        state.ended = True
        assert owner.wait(1) == 1
        info["termination_requested"] = True

    monkeypatch.setattr(control, "preflight", preflight)
    monkeypatch.setattr(control, "no_later_launch", launch_gate)
    monkeypatch.setattr(control, "old", SimpleNamespace(**{
        **vars(control.old), "stop_owned_child": stop,
    }))
    monkeypatch.setattr(control, "witness", SimpleNamespace(**{
        **vars(control.witness), "WindowsProcess": Process,
    }))

    def run(execute=True):
        argv = ["stop", "--adoption-sources", str(source), "--output", str(output)]
        if execute:
            argv.append("--execute")
        monkeypatch.setattr(sys, "argv", argv)
        control.main()

    return SimpleNamespace(run=run, state=state, events=events, output=output, source=source)


def test_without_execute_does_not_open_or_stop_process(case):
    case.run(False)
    assert case.events == ["preflight", "launch_gate"]
    assert not case.output.exists()


def test_owner_identity_is_held_through_stop_and_saved(case):
    case.run()
    assert case.events == ["preflight", "launch_gate", "open", "launch_gate", "stop",
                           "launch_gate", "close"]
    seal = json.loads((case.output / "stop.provenance.json").read_text())
    assert seal["status"] == "ok"
    assert seal["params"]["no_later_tracking_launch_recorded"] is True
    info = json.loads((case.output / "termination.json").read_text())
    assert info["target_role"] == "old_supervisor" and info["pid"] == 9876
    assert info["held_identity"]["creation_filetime_100ns"] == 1234


def test_reused_pid_refuses_stop_and_closes_handle(case):
    case.state.identity_bad = True
    with pytest.raises(ValueError, match="PID was reused"):
        case.run()
    assert "stop" not in case.events and case.state.closed
    assert json.loads((case.output / "stop.provenance.json").read_text())["status"] == "failed"


def test_post_stop_later_launch_is_failed_not_adoption_permission(case):
    case.state.late = True
    with pytest.raises(ValueError, match="later launch"):
        case.run()
    assert case.state.ended and case.state.closed
    assert (case.output / "termination.json").is_file()
    assert json.loads((case.output / "stop.provenance.json").read_text())["status"] == "failed"


def test_preflight_failure_has_no_side_effect(case, monkeypatch):
    def fail(path):
        raise ValueError("child exit not proven")

    monkeypatch.setattr(control, "preflight", fail)
    with pytest.raises(ValueError, match="child exit not proven"):
        case.run()
    assert not case.events and not case.output.exists()


def test_existing_output_refuses_second_action(case):
    case.output.mkdir()
    with pytest.raises(FileExistsError):
        case.run()
    assert "open" not in case.events and "stop" not in case.events


def test_prelaunch_gate_failure_does_not_touch_process(case, monkeypatch):
    def fail(binding):
        raise ValueError("later launch")

    monkeypatch.setattr(control, "no_later_launch", fail)
    with pytest.raises(ValueError, match="later launch"):
        case.run()
    assert case.events == ["preflight"] and not case.output.exists()


def test_frozen_cleanup_reads_real_adapter_returncode(case, monkeypatch):
    def taskkill(command, **kwargs):
        assert command[-4:] == ["/PID", "9876", "/T", "/F"]
        assert kwargs["shell"] is False if "shell" in kwargs else True
        assert kwargs["timeout"] == 10
        assert not case.state.closed
        case.state.ended = True
        return SimpleNamespace(returncode=0, stdout="owned tree stopped", stderr="")

    namespace = FROZEN_STOP.__globals__.copy()
    namespace["WINDOWS"] = True
    namespace["subprocess"] = SimpleNamespace(run=taskkill, CREATE_NO_WINDOW=0x08000000)
    actual_stop = FunctionType(FROZEN_STOP.__code__, namespace, FROZEN_STOP.__name__)
    monkeypatch.setattr(control.old, "stop_owned_child", actual_stop)
    case.run()
    info = json.loads((case.output / "termination.json").read_text())
    assert info["terminated_returncode"] == 1 and case.state.closed
    assert info["tree_termination_returncode"] == 0


@pytest.mark.parametrize("tail", [b'{"event":"phase_start"', b'{"event":"phase_start"}\n'])
def test_launch_gate_rejects_partial_or_complete_later_launch(tmp_path, monkeypatch, tail):
    baseline = b'{"event":"owned_child_started"}\n'
    (tmp_path / "events_baseline.jsonl").write_bytes(baseline)
    events = tmp_path / "events.jsonl"
    events.write_bytes(baseline + tail)
    monkeypatch.setattr(control, "WITNESS_ROOT", tmp_path)
    with pytest.raises(ValueError):
        control.no_later_launch({"events_path": str(events), "child_event_index": 0})

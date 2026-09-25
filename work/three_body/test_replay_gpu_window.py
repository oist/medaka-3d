"""Scheduling checks without process signals, CUDA, or production-file writes."""

import json

import pytest
import replay_gpu_window as window


class FakeProcess:
    def __init__(self, exited=False, code=0):
        self.exited, self.code, self.closed = exited, code, False

    def signalled(self):
        return self.exited

    def exit_info(self):
        return {"exit_code": self.code, "exit_code_success": self.code == 0}

    def close(self):
        self.closed = True


@pytest.fixture
def attached(tmp_path, monkeypatch):
    monkeypatch.setattr(window, "REPO", tmp_path)
    root, output = tmp_path / "work/supervisor", tmp_path / "work/replay"
    root.mkdir(parents=True)
    output.mkdir()
    obj = window.ReplayGPUWindow(root, 123, output)
    obj.baseline = b'{"event":"owned_child_started","pid":123}\n'
    (root / "events.jsonl").write_bytes(obj.baseline)
    obj.owner, obj.child = FakeProcess(), FakeProcess(exited=True)
    obj.tracking_root = root / "phases/current/tracking"
    obj.tracking_root.mkdir(parents=True)
    (obj.tracking_root / "continuation.provenance.json").write_text(json.dumps({
        "status": "ok", "stage": "three_body_long_continuation"}))
    return obj


def test_unfinished_event_is_not_silently_consumed(attached):
    path = attached.root / "events.jsonl"
    path.write_bytes(attached.baseline + b'{"event":"phase_start"')
    attached._check_events()
    assert attached.baseline.endswith(b"\n")
    path.write_bytes(attached.baseline + b'{"event":"phase_start"}\n')
    with pytest.raises(ValueError, match="next tracking phase"):
        attached._check_events()


def test_successful_exit_opens_only_until_new_launch(attached):
    attached.wait_until_open()
    assert attached.opened and attached.report["status"] == "open"
    (attached.root / "events.jsonl").write_bytes(
        attached.baseline + b'{"event":"owned_child_started","pid":456}\n')
    with pytest.raises(ValueError, match="window ended"):
        attached.assert_open()
    assert attached.report["status"] == "failed"
    owner, child = attached.owner, attached.child
    attached.close()
    assert owner.closed and child.closed
    assert attached.report["status"] == "failed"


def test_failure_exit_never_opens(attached):
    attached.child.code = 1
    with pytest.raises(ValueError, match="did not exit successfully"):
        attached.wait_until_open()
    assert not attached.opened


def test_event_rewrite_is_rejected(attached):
    (attached.root / "events.jsonl").write_bytes(b'{"event":"different"}\n')
    with pytest.raises(ValueError, match="prefix changed"):
        attached.wait_until_open()


def test_live_child_timeout_does_not_signal_process(attached, monkeypatch):
    attached.child.exited = False
    times = iter((0.0, 9000.0))
    monkeypatch.setattr(window.time, "monotonic", lambda: next(times))
    with pytest.raises(ValueError, match="timed out"):
        attached.wait_until_open()
    assert not attached.child.closed and not attached.child.signalled()
    assert attached.report["production_process_signalled_by_guard"] is False


def test_success_seal_is_required_after_process_exit(attached):
    (attached.tracking_root / "continuation.provenance.json").write_text(json.dumps({
        "status": "failed", "stage": "three_body_long_continuation"}))
    with pytest.raises(ValueError, match="no successful"):
        attached.wait_until_open()
    assert not attached.opened


def test_attach_preflight_failure_is_saved_without_os_query(attached, monkeypatch):
    obj = window.ReplayGPUWindow(attached.root, 123, attached.output)
    monkeypatch.setattr(window, "SUPERVISOR", attached.root / "absent_source.py")

    def forbidden(pid):
        raise AssertionError("OS query must follow source validation")

    monkeypatch.setattr(window, "WindowsProcess", forbidden)
    with pytest.raises(FileNotFoundError):
        obj.attach()
    saved = json.loads((attached.output / "gpu_window.json").read_text())
    assert saved["status"] == "failed" and "FileNotFoundError" in saved["error"]
    assert obj.owner is None and obj.child is None

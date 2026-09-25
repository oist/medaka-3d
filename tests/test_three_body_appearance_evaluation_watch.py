"""Only hermetic gates run: the real watcher and evaluator must never start in tests."""

import ast
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "work/three_body/continue_appearance_minute_evaluation.py"
SPEC = importlib.util.spec_from_file_location("_appearance_watch_test", SCRIPT)
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_frozen_dependency_set_matches_actual_evaluator_main_exactly():
    tree = ast.parse(watch.EVALUATOR.read_text(encoding="utf-8"))
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assignment = next(
        node
        for node in main.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "code" for target in node.targets)
    )

    def path_value(node):
        if isinstance(node, ast.Name):
            return {"REPO": ROOT, "LAB": ROOT / "labs/20260909_three_body"}[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return path_value(node.left) / path_value(node.right)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Path"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "__file__"
        ):
            return watch.EVALUATOR
        raise AssertionError("evaluator uses a changed source-list expression")

    actual = [path_value(node).resolve() for node in assignment.value.elts]
    assert len(watch.CODE) == len(actual) + 1
    assert set(watch.CODE) == set(actual) | {SCRIPT}
    assert watch.digest(watch.EVALUATOR) == watch.EXPECTED_EVALUATOR_SHA256


@pytest.mark.parametrize("content", [None, b'{"status":', b"{}", b'{"params":{}}', b"\xff"])
def test_missing_or_partially_written_status_is_pending(tmp_path, content):
    path = tmp_path / "status.json"
    if content is not None:
        path.write_bytes(content)
    assert watch.read_status(path) is None


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    run, output, control = tmp_path / "run", tmp_path / "output", tmp_path / "control"
    run.mkdir()
    evaluator, wrapper = tmp_path / "evaluator.py", tmp_path / "wrapper.py"
    evaluator.write_text("frozen evaluator", encoding="utf-8")
    wrapper.write_text("frozen wrapper", encoding="utf-8")
    for name, value in dict(
        REPO=tmp_path,
        RUN=run,
        OUTPUT=output,
        CONTROL=control,
        PERSISTENT=tmp_path / "persistent",
        UPSTREAM=run / "experiment.provenance.json",
        EVALUATOR=evaluator,
        CODE=[evaluator, wrapper],
        EXPECTED_EVALUATOR_SHA256=watch.digest(evaluator),
    ).items():
        monkeypatch.setattr(watch, name, value)
    records = {}
    for key, name in (
        ("summary", "summary.json"),
        ("candidate", "candidate.npz"),
        ("appearance_enrollment", "appearance_enrollment.npz"),
        ("appearance_enrollment_metadata", "appearance_enrollment.json"),
    ):
        path = run / name
        path.write_bytes(name.encode())
        records[key] = dict(path=str(path), sha256=watch.digest(path))
    record = dict(status="ok", outputs=records)
    write_json(watch.UPSTREAM, record)
    calls = []

    def finish_output():
        output.mkdir()
        outputs = {}
        for key, name in (
            ("comparison", "comparison.json"),
            ("diagnostics", "diagnostics.json"),
            ("review_events", "review_events.json"),
            ("enrollment_audit", "enrollment_audit.json"),
            ("result", "RESULT.md"),
        ):
            path = output / name
            write_json(path, {"new_method_review_counts": {}})
            outputs[key] = dict(path=str(path), sha256=watch.digest(path))
        write_json(output / "evaluation.provenance.json", dict(status="ok", outputs=outputs))

    def fake_process(command, **kwargs):
        calls.append((command, kwargs))
        finish_output()
        return SimpleNamespace(
            pid=123, stdout=io.StringIO("synthetic evaluator done\n"), wait=lambda: 0
        )

    monkeypatch.setattr(watch.subprocess, "Popen", fake_process)
    return SimpleNamespace(
        run=run,
        output=output,
        control=control,
        record=record,
        evaluator=evaluator,
        calls=calls,
        finish_output=finish_output,
    )


def test_complete_seal_launches_one_cpu_command_with_required_persistent(prepared):
    watch.main()
    assert len(prepared.calls) == 1
    command = prepared.calls[0][0]
    assert command[command.index("--persistent") + 1] == str(watch.PERSISTENT)
    status = json.loads((prepared.control / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "comparison_complete"
    assert status["automatic_retries"] == 0
    assert status["comparison_outputs_verified"] == 5
    assert len(status["code_sha256"]) == 2
    assert len(status["upstream_sha256"]) == 5
    with pytest.raises(FileExistsError):
        watch.main()
    assert len(prepared.calls) == 1


@pytest.mark.parametrize(
    "bad",
    [
        "upstream_failed",
        "candidate_bytes",
        "wrong_enrollment_path",
        "missing_enrollment_output",
        "wrong_evaluator",
        "wait_code_change",
    ],
)
def test_failed_or_rebound_run_never_starts_evaluator(prepared, monkeypatch, bad):
    p = prepared
    if bad == "upstream_failed":
        p.record["status"] = "failed"
    elif bad == "candidate_bytes":
        (p.run / "candidate.npz").write_bytes(b"changed candidate")
    elif bad == "wrong_enrollment_path":
        copied = p.run / "different.npz"
        copied.write_bytes((p.run / "appearance_enrollment.npz").read_bytes())
        p.record["outputs"]["appearance_enrollment"]["path"] = str(copied)
    elif bad == "missing_enrollment_output":
        del p.record["outputs"]["appearance_enrollment"]
    elif bad == "wrong_evaluator":
        p.evaluator.write_text("changed before waiting", encoding="utf-8")
    elif bad == "wait_code_change":
        p.record["status"] = "running"

        def change_during_wait(_seconds):
            p.evaluator.write_text("changed while waiting", encoding="utf-8")
            p.record["status"] = "ok"
            write_json(watch.UPSTREAM, p.record)

        monkeypatch.setattr(watch.time, "sleep", change_during_wait)
    write_json(watch.UPSTREAM, p.record)
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        watch.main()
    assert not p.calls
    assert json.loads((p.control / "status.json").read_text(encoding="utf-8"))["phase"] == "failed"


def test_partial_status_waits_before_one_shot_launch(prepared, monkeypatch):
    watch.UPSTREAM.write_text('{"status":', encoding="utf-8")
    waits = []

    def complete(_seconds):
        waits.append(_seconds)
        write_json(watch.UPSTREAM, prepared.record)

    monkeypatch.setattr(watch.time, "sleep", complete)
    watch.main()
    assert waits == [5]
    assert len(prepared.calls) == 1


@pytest.mark.parametrize(
    "bad",
    [
        "child_failed",
        "output_tamper",
        "output_missing",
        "output_rebound",
        "upstream_changed",
        "code_changed",
    ],
)
def test_failure_after_launch_is_preserved_without_retry(prepared, monkeypatch, bad):
    p = prepared

    def fake_process(command, **_kwargs):
        p.calls.append(command)
        p.finish_output()
        if bad == "output_tamper":
            (p.output / "comparison.json").write_bytes(b"changed")
        elif bad in ("output_missing", "output_rebound"):
            path = p.output / "evaluation.provenance.json"
            record = watch.read_status(path)
            if bad == "output_missing":
                del record["outputs"]["enrollment_audit"]
            else:
                copied = p.run / "other_comparison.json"
                copied.write_bytes((p.output / "comparison.json").read_bytes())
                record["outputs"]["comparison"]["path"] = str(copied)
            write_json(path, record)
        elif bad == "upstream_changed":
            (p.run / "candidate.npz").write_bytes(b"changed upstream evidence")
        elif bad == "code_changed":
            p.evaluator.write_text("changed during evaluation", encoding="utf-8")
        return SimpleNamespace(
            pid=123, stdout=io.StringIO("done\n"), wait=lambda: 1 if bad == "child_failed" else 0
        )

    monkeypatch.setattr(watch.subprocess, "Popen", fake_process)
    with pytest.raises((ValueError, RuntimeError)):
        watch.main()
    assert len(p.calls) == 1
    status = json.loads((p.control / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "failed" and status["automatic_retries"] == 0
    assert (p.control / "evaluation.log").is_file()


def test_wait_timeout_does_not_launch(prepared, monkeypatch):
    write_json(watch.UPSTREAM, {"status": "running"})
    times = iter([0, 21601])
    monkeypatch.setattr(watch.time, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError):
        watch.main()
    assert not prepared.calls

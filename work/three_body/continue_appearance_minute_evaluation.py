"""One-shot CPU comparison after the appearance minute seals; run only when authorized."""

import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTROL = REPO / "work/three_body/appearance_minute_evaluation_control_v1"
RUN = REPO / "work/three_body/appearance_continuous_one_minute_v1"
PERSISTENT = REPO / "work/three_body/persistent_mouth_continuous_one_minute_v1"
UPSTREAM = RUN / "experiment.provenance.json"
OUTPUT = REPO / "work/three_body/appearance_continuous_one_minute_comparison_v1"
LAB = REPO / "labs/20260909_three_body"
EVALUATOR = LAB / "evaluate_appearance_minute.py"
EXPECTED_EVALUATOR_SHA256 = "a3db47310125b8bc1ac19261cd70c40591f01d2bfff1447785c4d6c81a06b09c"
# Exactly the comparison main's source set, followed by this wrapper.
CODE = [
    LAB / name
    for name in (
        "evaluate_appearance_minute.py",
        "evaluate_minute.py",
        "evaluate_variant_minute.py",
        "diagnostics.py",
        "review_events.py",
        "run_experiment.py",
        "observations.py",
        "prepare_sequence.py",
        "run_node_recovery.py",
        "run_persistent_node_recovery.py",
        "run_appearance_tracking.py",
        "appearance_diagnostic.py",
    )
]
CODE += [
    REPO / "src/medaka3d" / name
    for name in (
        "calib.py",
        "types.py",
        "provenance.py",
        "eval/generative.py",
        "eval/reacquisition.py",
    )
]
CODE.append(Path(__file__).resolve())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_status(path):
    """Missing or partially written JSON is pending; a declared failure is not retried."""
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(result, dict):
        raise ValueError("upstream provenance must be a JSON object")
    return result if result.get("status") is not None else None


def require_unchanged(values, message):
    if any(digest(path) != value for path, value in values.items()):
        raise ValueError(message)


def verify_upstream(record):
    """Bind the completed run and all appearance-specific evidence before launching."""
    if record.get("status") != "ok":
        raise ValueError("upstream has not sealed successfully")
    hashes = {}
    for key, filename in (
        ("summary", "summary.json"),
        ("candidate", "candidate.npz"),
        ("appearance_enrollment", "appearance_enrollment.npz"),
        ("appearance_enrollment_metadata", "appearance_enrollment.json"),
    ):
        item = record["outputs"][key]
        path = (RUN / filename).resolve(strict=True)
        if Path(item["path"]).resolve(strict=True) != path or digest(path) != item["sha256"]:
            raise ValueError("upstream output path/SHA differs from its completed seal")
        hashes[str(path)] = item["sha256"]
    hashes[str(UPSTREAM)] = digest(UPSTREAM)
    return hashes


def verify_comparison():
    record = json.loads((OUTPUT / "evaluation.provenance.json").read_text(encoding="utf-8"))
    if record.get("status") != "ok":
        raise RuntimeError("comparison did not seal successfully")
    outputs = record["outputs"]
    for key, filename in (
        ("comparison", "comparison.json"),
        ("diagnostics", "diagnostics.json"),
        ("review_events", "review_events.json"),
        ("enrollment_audit", "enrollment_audit.json"),
        ("result", "RESULT.md"),
    ):
        if key not in outputs or Path(outputs[key]["path"]).resolve(strict=True) != (
            OUTPUT / filename
        ).resolve(strict=True):
            raise ValueError("comparison is missing or rebinding a required output")
    for name, item in outputs.items():
        path = Path(item["path"]).resolve(strict=True)
        if not path.is_relative_to(OUTPUT.resolve()) or digest(path) != item["sha256"]:
            raise ValueError(f"sealed comparison output path/SHA changed: {name}")
    return len(outputs)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError(OUTPUT)
    CONTROL.mkdir(exist_ok=False)
    frozen = {str(path): digest(path) for path in CODE}
    command = [
        sys.executable,
        "-B",
        str(EVALUATOR),
        "--run",
        str(RUN),
        "--persistent",
        str(PERSISTENT),
        "--output",
        str(OUTPUT),
    ]
    status = {
        "phase": "waiting_for_complete_minute",
        "started_at": datetime.now(UTC).isoformat(),
        "command": command,
        "code_sha256": frozen,
        "automatic_retries": 0,
        "expected_evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
    }

    def save():
        status["updated_at"] = datetime.now(UTC).isoformat()
        (CONTROL / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    save()
    try:
        if frozen[str(EVALUATOR)] != EXPECTED_EVALUATOR_SHA256:
            raise ValueError("evaluator differs from the approved frozen source")
        started = time.monotonic()
        while True:
            upstream = read_status(UPSTREAM)
            if upstream and upstream.get("status") not in ("running", "ok"):
                raise RuntimeError("upstream minute failed; no comparison or restart will run")
            if upstream and upstream.get("status") == "ok":
                upstream_hashes = verify_upstream(upstream)
                require_unchanged(frozen, "evaluation code changed while waiting")
                status["upstream_sha256"] = upstream_hashes
                break
            if time.monotonic() - started > 6 * 3600:
                raise TimeoutError("no complete minute after six hours; evaluation not launched")
            time.sleep(5)
        status["phase"] = "evaluating"
        save()
        print("Appearance minute sealed; starting the fixed CPU comparison.", flush=True)
        with (CONTROL / "evaluation.log").open("x", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            status["evaluation_pid"] = process.pid
            save()
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            status["evaluation_returncode"] = process.wait()
            if status["evaluation_returncode"]:
                raise RuntimeError("evaluation failed; artifacts preserved without automatic retry")
        outputs_verified = verify_comparison()
        require_unchanged(upstream_hashes, "upstream completion evidence changed during evaluation")
        require_unchanged(frozen, "evaluation code changed during execution")
        report = json.loads((OUTPUT / "comparison.json").read_text(encoding="utf-8"))
        status.update(
            phase="comparison_complete",
            comparison_sha256=digest(OUTPUT / "comparison.json"),
            comparison_provenance_sha256=digest(OUTPUT / "evaluation.provenance.json"),
            comparison_outputs_verified=outputs_verified,
            new_method_review_counts=report["new_method_review_counts"],
            independent_identity_verdict="unverified; human three-fish truth still required",
        )
    except BaseException as exc:
        status.update(phase="failed", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

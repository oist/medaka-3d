"""One-shot comparison after the persistent-mouth minute seals successfully."""
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
CONTROL = REPO / "work/three_body/persistent_minute_evaluation_control_v1"
RUN = REPO / "work/three_body/persistent_mouth_continuous_one_minute_v1"
UPSTREAM = RUN / "experiment.provenance.json"
OUTPUT = REPO / "work/three_body/persistent_mouth_continuous_one_minute_comparison_v1"
LAB = REPO / "labs/20260909_three_body"
CODE = [LAB / name for name in (
    "evaluate_variant_minute.py", "evaluate_minute.py", "diagnostics.py", "review_events.py",
    "run_experiment.py", "observations.py", "prepare_sequence.py", "run_node_recovery.py",
    "run_persistent_node_recovery.py")]
CODE += [REPO / "src/medaka3d" / name for name in (
    "calib.py", "types.py", "provenance.py", "eval/generative.py", "eval/reacquisition.py")]
CODE.append(Path(__file__))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_status(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    CONTROL.mkdir(exist_ok=False)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    frozen = {str(path): digest(path) for path in CODE}
    command = [sys.executable, "-B", str(LAB / "evaluate_variant_minute.py"),
               "--run", str(RUN), "--output", str(OUTPUT)]
    status = {"phase": "waiting_for_complete_minute", "started_at": datetime.now(UTC).isoformat(),
              "command": command, "code_sha256": frozen, "automatic_retries": 0}

    def save():
        status["updated_at"] = datetime.now(UTC).isoformat()
        (CONTROL / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    save()
    try:
        started = time.monotonic()
        while True:
            upstream = read_status(UPSTREAM)
            if upstream and upstream.get("status") not in ("running", "ok"):
                raise RuntimeError("upstream minute failed; no comparison or restart will run")
            if upstream and upstream.get("status") == "ok":
                for key, filename in (("summary", "summary.json"), ("candidate", "candidate.npz")):
                    item = upstream["outputs"][key]
                    if Path(item["path"]).resolve() != (RUN / filename).resolve():
                        raise ValueError("upstream output path is not the registered run")
                    if digest(RUN / filename) != item["sha256"]:
                        raise ValueError("upstream output differs from its completion record")
                if any(digest(path) != value for path, value in frozen.items()):
                    raise ValueError("evaluation code changed while waiting")
                status["upstream_provenance_sha256"] = digest(UPSTREAM)
                break
            if time.monotonic() - started > 6 * 3600:
                raise TimeoutError("no complete minute after six hours; evaluation not launched")
            time.sleep(5)
        status["phase"] = "evaluating"
        save()
        print("Persistent-mouth minute sealed; starting the fixed CPU comparison.", flush=True)
        with (CONTROL / "evaluation.log").open("x", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                       errors="replace")
            status["evaluation_pid"] = process.pid
            save()
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            if process.wait():
                raise RuntimeError("evaluation failed; artifacts preserved without automatic retry")
        record = json.loads((OUTPUT / "evaluation.provenance.json").read_text(encoding="utf-8"))
        if record["status"] != "ok":
            raise RuntimeError("comparison did not seal successfully")
        for name, item in record["outputs"].items():
            if digest(item["path"]) != item["sha256"]:
                raise ValueError(f"sealed comparison output changed: {name}")
        if digest(UPSTREAM) != status["upstream_provenance_sha256"]:
            raise ValueError("upstream completion record changed during evaluation")
        if any(digest(path) != value for path, value in frozen.items()):
            raise ValueError("evaluation code changed during execution")
        report = json.loads((OUTPUT / "comparison.json").read_text(encoding="utf-8"))
        status.update(phase="comparison_complete", comparison_sha256=digest(OUTPUT / "comparison.json"),
                      new_method_review_counts=report["new_method_review_counts"],
                      independent_identity_verdict="unverified; human three-fish truth still required")
    except BaseException as exc:
        status.update(phase="failed", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

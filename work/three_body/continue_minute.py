"""One-shot local continuation after the already running staging job seals inputs."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(REPO / "src"))
from run_experiment import frozen_code_paths  # noqa: E402

SEQUENCE = REPO / "work/three_body/inputs/continuous_one_minute_76620-86580"
OUTPUT = REPO / "work/three_body/v1_continuous_one_minute_detectron2"
CONTROL = REPO / "work/three_body/minute_control_v1"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    CONTROL.mkdir(exist_ok=False)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = [*frozen_code_paths(), LAB / "MINUTE_PROTOCOL.md", Path(__file__)]
    frozen = {str(p.relative_to(REPO)): digest(p) for p in paths}
    command = [
        sys.executable, "-B", str(LAB / "run_experiment.py"),
        "--sequence", str(SEQUENCE), "--output", str(OUTPUT),
        "--mode", "detectron2", "--flow-proposals", "--device", "cuda",
    ]
    status = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "phase": "waiting_for_staging", "command": command,
        "code_sha256": frozen, "automatic_retries": 0,
    }

    def write_status():
        (CONTROL / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    write_status()
    try:
        start = time.monotonic()
        sidecar = SEQUENCE / "staging.provenance.json"
        while True:
            if sidecar.exists():
                try:
                    staging = json.loads(sidecar.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    time.sleep(1)
                    continue
                if staging["status"] != "ok":
                    raise RuntimeError("staging failed; preserve its provenance and do not launch")
                manifest = SEQUENCE / "manifest.json"
                if digest(manifest) != staging["outputs"]["manifest"]["sha256"]:
                    raise ValueError("sealed staging manifest checksum mismatch")
                if {str(p.relative_to(REPO)): digest(p) for p in paths} != frozen:
                    raise ValueError("registered code changed while waiting; refuse launch")
                break
            if time.monotonic() - start > 6 * 3600:
                raise TimeoutError("staging has not sealed within six hours; no tracker was launched")
            time.sleep(5)
        status.update(phase="tracking", staging_manifest_sha256=digest(manifest))
        write_status()
        print("Staging sealed; launching the registered native-rate minute.", flush=True)
        with (CONTROL / "tracking.log").open("x", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(
                command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            status["tracking_pid"] = process.pid
            write_status()
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f"tracking subprocess exited with {code}; no automatic restart")
        summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
        provenance = json.loads((OUTPUT / "experiment.provenance.json").read_text(encoding="utf-8"))
        if provenance["status"] != "ok" or not summary["technical_completion"]:
            raise RuntimeError("tracking did not seal a successful complete output")
        if {str(p.relative_to(REPO)): digest(p) for p in paths} != frozen:
            raise ValueError("registered code changed during tracking")
        status.update(phase="complete", summary_sha256=digest(OUTPUT / "summary.json"))
    except BaseException as exc:
        status.update(phase="failed", error=repr(exc))
        raise
    finally:
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_status()


if __name__ == "__main__":
    main()

"""Wait for the owned GPU milestone, then run the two qualified CPU stages once.

This launcher never starts, stops or changes tracking. A failed stage is logged
and stops the queue; no retry, altered solver cap or partial success is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

WORK = Path(__file__).resolve().parent
ROOT = WORK / "fast_tail_two_minute_evaluation_launch_v1"
BODY = WORK / "fast_tail_two_minute_continuation_comparison_v1"
IDENTITY = WORK / "fast_tail_two_minute_continuation_identity_v1"
MILESTONE = WORK / "fast_tail_five_minute_continuation_v1/milestones/through_086320/summary.json"
SOURCES = {
    "compare_fast_tail_continuation.py": (
        "2e4a749d5ea13116f05a622afd47478af311525fbd856b32a116bd9ac765acba"
    ),
    "run_fast_tail_continuation_identity.py": (
        "87154f972479ac71ce44ab3d859fb5fb4e8ee81947f335e81d4f5b49781cf9fa"
    ),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    with (ROOT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def stage(name, script, arguments):
    if sha(WORK / script) != SOURCES[script]:
        raise ValueError("qualified source changed: " + script)
    command = [sys.executable, "-B", "-u", str(WORK / script), "--execute", *arguments]
    process = subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW)
    write(name + "_started.json", dict(pid=process.pid, command=command, started=time.time()))
    print(name, "started", process.pid, flush=True)
    code = process.wait()
    write(name + "_exit.json", dict(pid=process.pid, returncode=code, finished=time.time()))
    if code:
        raise RuntimeError(f"{name} exited {code}; no retry")


def main():
    if any(path.exists() for path in (ROOT, BODY, IDENTITY)):
        raise FileExistsError("launch and both scientific outputs must be fresh")
    ROOT.mkdir()
    write(
        "started.json",
        dict(
            pid=os.getpid(),
            utc=datetime.now(UTC).isoformat(),
            milestone=str(MILESTONE),
            sources=SOURCES,
            owned_gpu_pid=11156,
            owned_gpu_birth=1789035860.1460886,
            scientific_interval=[66400, 86320],
            tracking_mutated=False,
        ),
    )
    try:
        deadline, announced = time.monotonic() + 3600, 0.0
        while True:
            if MILESTONE.exists():
                try:
                    value = json.loads(MILESTONE.read_bytes())
                except json.JSONDecodeError:
                    value = None  # The producer may still be closing this file.
                if value is not None:
                    if value.get("status") != "ok" or value.get("horizon_stop") != 86320:
                        raise ValueError("invalid milestone; scientific stage not launched")
                    break
            if time.monotonic() >= deadline:
                raise TimeoutError("milestone wait expired; tracking left unchanged")
            worker = psutil.Process(11156)
            if worker.create_time() != 1789035860.1460886 or not worker.is_running():
                raise RuntimeError("owned tracking worker no longer live; no restart")
            if time.monotonic() - announced >= 60:
                print("Waiting for owned GPU milestone 86320; tracking remains live", flush=True)
                announced = time.monotonic()
            time.sleep(5)
        milestone_sha = sha(MILESTONE)
        write("milestone.json", dict(path=str(MILESTONE), sha256=milestone_sha))
        stage(
            "comparison",
            "compare_fast_tail_continuation.py",
            [
                "--horizon-stop",
                "86320",
                "--milestone-sha",
                milestone_sha,
                "--output",
                str(BODY),
            ],
        )
        body_seal = BODY / "comparison.provenance.json"
        if json.loads(body_seal.read_bytes()).get("status") != "ok":
            raise ValueError("successful comparison seal required")
        stage(
            "identity",
            "run_fast_tail_continuation_identity.py",
            [
                "--comparison-root",
                str(BODY),
                "--comparison-sha",
                sha(body_seal),
                "--milestone-sha",
                milestone_sha,
                "--horizon-stop",
                "86320",
                "--output",
                str(IDENTITY),
            ],
        )
        write("exit.json", dict(status="ok", returncode=0, finished=time.time()))
        print("Two-minute comparison and identity stages exited successfully", flush=True)
    except BaseException as error:
        write("exit.json", dict(status="failed", error=repr(error), finished=time.time()))
        raise


if __name__ == "__main__":
    main()

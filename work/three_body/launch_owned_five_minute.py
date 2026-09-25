"""Own one continuation child and persist its actual exit code; never restart it."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(__file__).resolve().parent
REPO = WORK.parent.parent
RUNNER = WORK / "continue_fast_tail_five_minutes.py"


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main(args):
    root = args.launch_root.resolve(strict=True)
    output = args.output.resolve()
    if not root.is_relative_to(WORK) or not output.is_relative_to(WORK) or output.exists():
        raise ValueError("fresh workspace tracker output and workspace launch directory required")
    if (root / "child_started.json").exists() or (root / "child_exit.json").exists():
        raise ValueError("launch directory already used")
    actual = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    if actual != args.runner_sha:
        raise ValueError("reviewed continuation source changed")
    command = [
        sys.executable,
        "-B",
        "-u",
        str(RUNNER),
        "--execute",
        "--output",
        str(output),
        "--audit-sha",
        args.audit_sha,
        "--receipt-root",
        str(args.receipt_root.resolve()),
    ]
    started = dt.datetime.now(dt.UTC).isoformat()
    tick = time.monotonic()
    child = subprocess.Popen(command, cwd=REPO, creationflags=subprocess.CREATE_NO_WINDOW)
    write(
        root / "child_started.json",
        dict(
            pid=child.pid,
            command=command,
            started_at=started,
            runner_sha256=actual,
            output=str(output),
            automatic_restart=False,
        ),
    )
    code = child.wait()
    write(
        root / "child_exit.json",
        dict(
            pid=child.pid,
            exit_code=code,
            started_at=started,
            finished_at=dt.datetime.now(dt.UTC).isoformat(),
            elapsed_seconds=time.monotonic() - tick,
            automatic_restart=False,
        ),
    )
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner-sha", required=True)
    parser.add_argument("--audit-sha", required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    sys.exit(main(parser.parse_args()))

"""Own one hidden sparse continuation; retain output and actual exit, never retry."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(__file__).resolve().parent
REPO = WORK.parent.parent
RUNNER = WORK / "continue_sparse_instance_five_minutes.py"


def write(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main(args):
    root, output = args.launch_root.resolve(), args.output.resolve()
    if (
        not root.is_relative_to(WORK)
        or not output.is_relative_to(WORK)
        or root.exists()
        or output.exists()
        or output.is_relative_to(root)
        or root.is_relative_to(output)
    ):
        raise ValueError("distinct fresh workspace output and launch directory required")
    digest = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    if digest != args.runner_sha:
        raise ValueError("reviewed runner source differs")
    command = [
        sys.executable,
        "-B",
        "-u",
        str(RUNNER),
        "--execute",
        "--tracker-sha",
        args.tracker_sha,
        "--output",
        str(output),
        "--receipt-root",
        str(args.receipt_root.resolve()),
    ]
    root.mkdir(parents=True, exist_ok=False)
    started, tick = dt.datetime.now(dt.UTC).isoformat(), time.monotonic()
    write(
        root / "wrapper_started.json",
        dict(
            pid=os.getpid(),
            started_at=started,
            runner_sha256=digest,
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            automatic_restart=False,
        ),
    )
    with (root / "stdout.log").open("xb") as stdout, (root / "stderr.log").open("xb") as stderr:
        child = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        write(
            root / "child_started.json",
            dict(
                pid=child.pid,
                started_at=started,
                command=command,
                runner_sha256=digest,
                tracker_sha256=args.tracker_sha,
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
    parser.add_argument("--tracker-sha", required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    sys.exit(main(parser.parse_args()))

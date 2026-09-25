"""Wait for the existing Sparse GPU process, then own two CPU stages once.

Only small completion documents are interpreted here. The frozen comparison and
beam callers own their scientific validation. No tracker, solver or GPU module is
imported; no process is restarted, terminated or retried by this supervisor.
"""

from __future__ import annotations

import argparse
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
REPO = WORK.parent.parent
EXECUTION_READY = True
ROOT = WORK / "sparse_instance_five_minute_evaluation_launch_v1"
GPU_LAUNCH = WORK / "sparse_instance_five_minute_launch_v1"
TRACKING = WORK / "sparse_instance_five_minute_continuation_v1"
CONTROL = WORK / "fast_tail_five_minute_continuation_v1"
BODY = WORK / "sparse_instance_five_minute_comparison_v1"
IDENTITY = WORK / "sparse_instance_five_minute_identity_beam256_v1"
RUNNER = WORK / "continue_sparse_instance_five_minutes.py"
POLL_SECONDS = 5
EXIT_RECEIPT_GRACE_SECONDS = 30
OWNED = {11840: 1789047237.2205527, 32240: 1789047237.194405}
WORKER_PID, PARENT_PID = 11840, 32240
CONTROL_SHA = "d049a2d9244d9c0c761de7d75035a12f10dee14289af1b58d741fad9ec710469"
LAUNCH_START_SHA = "fa1e53680e65d91886cfcd5010b1585c6af8ece0bb38e258132eca98d8ce85a7"
SOURCES = {
    "continue_sparse_instance_five_minutes.py": (
        "3f5141a5aeec6fadc464ad242ac19fe13e227065af6f1188c638af4c58129096"
    ),
    "sparse_instance_rescore_tracker.py": (
        "d041aff3a0d8cbd71341dbe67dd9b5a7a1e2e7aeb165f55e4b08f8fabdb8fb79"
    ),
    "launch_sparse_instance_five_minutes.py": (
        "eb5a22dfd7af4ae0dbde6e86e8d939b00ffde394494dbad72aa84f7a88624e0b"
    ),
    "compare_sparse_instance_five_minutes.py": (
        "1ec0893d7dd2bbc746004f71d9b81abc60096cc0173bf44871f56be948e1c8dc"
    ),
    "execute_sparse_instance_five_minute_identity_beam.py": (
        "07d9937488057e620f4478a6458e949118d28c8a4ad8e9e75d41eb84e4f12609"
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(UTC).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    with (ROOT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def event(name, **values):
    value = dict(event=name, utc=utc(), **values)
    with (ROOT / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
    print(json.dumps(value, allow_nan=False), flush=True)


def document(path, expected_sha=None):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    require(expected_sha is None or digest == expected_sha, "document SHA differs: " + str(path))
    value = json.loads(data)
    require(isinstance(value, dict), "JSON object required: " + str(path))
    return value, dict(path=str(path), sha256=digest, size_bytes=len(data))


def exact(value, expected, message):
    require(
        all(type(value.get(k)) is type(v) and value.get(k) == v for k, v in expected.items()),
        message,
    )


def source_guard():
    for name, expected in SOURCES.items():
        require(sha(WORK / name) == expected, "frozen source changed: " + name)


def successful_seal(root, filename, expected_sha=None):
    seal, reference = document(root / filename, expected_sha)
    exact(seal, dict(stage=root.name, status="ok"), "successful stage required: " + root.name)
    require(
        isinstance(seal.get("inputs"), dict) and isinstance(seal.get("outputs"), dict),
        "seal schema",
    )
    return seal, reference


def bound_summary(root, seal):
    member = seal["outputs"]["summary.json"]
    require(Path(member["path"]).resolve() == root / "summary.json", "summary output path differs")
    value, reference = document(root / "summary.json", member["sha256"])
    require(
        type(member["size_bytes"]) is int and reference["size_bytes"] == member["size_bytes"],
        "summary size differs",
    )
    return value, reference


def input_edge(seal, path, digest):
    require(seal["inputs"].get(str(path)) == digest, "consumed input edge differs: " + str(path))


def launch_binding():
    value, reference = document(GPU_LAUNCH / "child_started.json", LAUNCH_START_SHA)
    exact(value, dict(pid=PARENT_PID, automatic_restart=False), "owned GPU launch identity differs")
    require(
        value["runner_sha256"] == SOURCES[RUNNER.name]
        and value["tracker_sha256"] == SOURCES["sparse_instance_rescore_tracker.py"]
        and Path(value["output"]).resolve() == TRACKING
        and str(RUNNER) in value["command"],
        "owned GPU launch source/output differs",
    )
    return value, reference


def observe_owned():
    """Only exact known PIDs are inspected; PID reuse fails closed."""
    rows = []
    for pid, birth in OWNED.items():
        try:
            process = psutil.Process(pid)
            actual = process.create_time()
            require(actual == birth, "owned PID was reused: " + str(pid))
            command = process.cmdline()
            require(str(RUNNER) in command, "owned process command changed")
            rows.append(dict(pid=pid, state="live", expected_birth=birth, observed_birth=actual))
        except psutil.NoSuchProcess:
            rows.append(dict(pid=pid, state="gone", expected_birth=birth, observed_birth=None))
    return rows


def wait_for_gpu_exit(started):
    gone_at, last_log, previous = None, -float("inf"), None
    while True:
        rows = observe_owned()
        now = time.monotonic()
        if rows != previous or now - last_log >= 30:
            event("owned_tracking_processes", observations=rows)
            last_log, previous = now, rows
        if all(row["state"] == "gone" for row in rows):
            gone_at = now if gone_at is None else gone_at
            path = GPU_LAUNCH / "child_exit.json"
            try:
                receipt, reference = document(path)
            except (FileNotFoundError, json.JSONDecodeError):
                require(
                    now - gone_at < EXIT_RECEIPT_GRACE_SECONDS,
                    "owned processes exited without a complete exit receipt",
                )
            else:
                exact(
                    receipt,
                    dict(pid=PARENT_PID, exit_code=0, automatic_restart=False),
                    "owned GPU launch failed; queue stopped",
                )
                require(
                    receipt["started_at"] == started["started_at"],
                    "owned exit belongs to another launch",
                )
                write(
                    "tracking_process_exit.json",
                    dict(
                        observations=rows,
                        launch_exit=receipt,
                        launch_exit_reference=reference,
                        worker_exit_code_observed=False,
                        processes_restarted=False,
                    ),
                )
                return
        time.sleep(POLL_SECONDS)


def tracking_completion():
    seal, reference = successful_seal(TRACKING, "tracking.provenance.json")
    input_edge(seal, RUNNER, SOURCES[RUNNER.name])
    summary, summary_reference = bound_summary(TRACKING, seal)
    exact(
        summary,
        dict(
            schema=TRACKING.name,
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
        ),
        "Sparse final horizon/lifecycle differs",
    )
    require(
        Path(summary["checkpoints"][-1]["path"]).resolve()
        == TRACKING / "checkpoints/through_116200",
        "Sparse final checkpoint horizon differs",
    )
    write(
        "tracking_completion.json",
        dict(
            seal=reference,
            summary=summary_reference,
            lifecycle=summary,
            checkpoint_payload_read=False,
            scientific_outputs_independently_rechecked_here=False,
        ),
    )
    return reference


def comparison_completion(tracking_reference):
    seal, reference = successful_seal(BODY, "comparison.provenance.json")
    input_edge(
        seal,
        WORK / "compare_sparse_instance_five_minutes.py",
        SOURCES["compare_sparse_instance_five_minutes.py"],
    )
    input_edge(seal, CONTROL / "tracking.provenance.json", CONTROL_SHA)
    input_edge(seal, TRACKING / "tracking.provenance.json", tracking_reference["sha256"])
    summary, summary_reference = bound_summary(BODY, seal)
    exact(
        summary,
        dict(
            schema=BODY.name,
            status="ok",
            native_interval=[66400, 116200],
            native_frames=49800,
            terminal_horizon_by_arm=dict(fast_tail_control=116200, sparse_instance_rescore=116200),
            candidate_modified=False,
            review_modified=False,
            GPU_used=False,
        ),
        "comparison final scope differs",
    )
    write("comparison_completion.json", dict(seal=reference, summary=summary_reference))
    return reference


def identity_completion(comparison_reference):
    seal, reference = successful_seal(IDENTITY, "identity.provenance.json")
    input_edge(
        seal,
        WORK / "execute_sparse_instance_five_minute_identity_beam.py",
        SOURCES["execute_sparse_instance_five_minute_identity_beam.py"],
    )
    input_edge(seal, BODY / "comparison.provenance.json", comparison_reference["sha256"])
    summary, summary_reference = bound_summary(IDENTITY, seal)
    exact(
        summary,
        dict(
            schema=IDENTITY.name,
            status="ok",
            native_interval=[66400, 116200],
            native_frames=49800,
            terminal_horizon=116200,
            solve_calls=1,
            beam_width=256,
            intermediate_ID_resets=0,
            original_raw21_unchanged=True,
            raw_sources_review_unchanged=True,
            review_flags_cleared=False,
            GPU_used=False,
            global_optimality_certified=False,
            physical_ID_truth=False,
            exact_DP_rerun=False,
            comparison_seal_sha256=comparison_reference["sha256"],
        ),
        "identity final scope differs",
    )
    write("identity_completion.json", dict(seal=reference, summary=summary_reference))
    return reference


def stage(name, script, output, arguments):
    source_guard()
    require(not output.exists(), "scientific output already exists; no retry: " + str(output))
    command = [sys.executable, "-B", "-u", str(WORK / script), *arguments, "--output", str(output)]
    started, tick = utc(), time.monotonic()
    with (
        (ROOT / (name + "_stdout.log")).open("xb") as stdout,
        (ROOT / (name + "_stderr.log")).open("xb") as stderr,
    ):
        child = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        try:
            birth = psutil.Process(child.pid).create_time()
        except psutil.NoSuchProcess:
            birth = None  # An immediate child exit still has its actual wait() code.
        write(
            name + "_started.json",
            dict(
                pid=child.pid,
                process_birth=birth,
                started_at=started,
                command=command,
                source_sha256=SOURCES[script],
                output=str(output),
                CPU_only_environment=True,
                automatic_restart=False,
            ),
        )
        event("CPU_stage_started", stage=name, pid=child.pid)
        while True:
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                event("CPU_stage_running", stage=name, pid=child.pid)
    write(
        name + "_exit.json",
        dict(
            pid=child.pid,
            exit_code=code,
            started_at=started,
            finished_at=utc(),
            elapsed_seconds=time.monotonic() - tick,
            automatic_restart=False,
        ),
    )
    event("CPU_stage_exited", stage=name, pid=child.pid, exit_code=code)
    require(code == 0, name + " failed; no retry or cap increase")


def main(args):
    require(EXECUTION_READY and args.execute, "source-only: supervisor execution disabled")
    require(
        all(path.parent == WORK and not path.exists() for path in (ROOT, BODY, IDENTITY)),
        "supervisor and scientific outputs must all be fresh",
    )
    source_guard()
    started, launch_reference = launch_binding()
    successful_seal(CONTROL, "tracking.provenance.json", CONTROL_SHA)
    own_sha = sha(__file__)
    ROOT.mkdir(exist_ok=False)  # Atomic singleton claim; never remove/reuse this root.
    write(
        "started.json",
        dict(
            pid=os.getpid(),
            started_at=utc(),
            supervisor_sha256=own_sha,
            sources=SOURCES,
            owned_processes={str(pid): birth for pid, birth in OWNED.items()},
            tracking_launch_reference=launch_reference,
            tracking_root=str(TRACKING),
            control_tracking_sha256=CONTROL_SHA,
            comparison_output=str(BODY),
            identity_output=str(IDENTITY),
            no_retry=True,
            no_cap_raise=True,
            tracking_mutated=False,
        ),
    )
    try:
        wait_for_gpu_exit(started)
        source_guard()
        tracking_reference = tracking_completion()
        stage(
            "comparison",
            "compare_sparse_instance_five_minutes.py",
            BODY,
            [
                "--control-seal-sha256",
                CONTROL_SHA,
                "--variant-seal-sha256",
                tracking_reference["sha256"],
            ],
        )
        comparison_reference = comparison_completion(tracking_reference)
        stage(
            "identity",
            "execute_sparse_instance_five_minute_identity_beam.py",
            IDENTITY,
            [
                "--comparison-sha",
                comparison_reference["sha256"],
            ],
        )
        identity_reference = identity_completion(comparison_reference)
        source_guard()
        require(sha(__file__) == own_sha, "supervisor source changed")
        result = dict(
            status="ok",
            exit_code=0,
            completed_CPU_stages=2,
            comparison=comparison_reference,
            identity=identity_reference,
            physical_ID_truth=False,
            dense_swap_zero_certified=False,
        )
    except BaseException as error:
        result = dict(
            status="failed", exit_code=1, error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        write("exit.json", dict(finished_at=utc(), no_retry=True, tracking_mutated=False, **result))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    sys.exit(main(parser.parse_args()))

"""Read-only 10-second GPU samples and artifact lifecycle observation, up to 60 minutes."""

import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from medaka3d.provenance import provenance  # noqa: E402

WORK = REPO / "work/three_body"
PLAN = WORK / "long_pipeline_execution_registration_v1/plan.json"
PLAN_SHA = "87be596d7d78d5ba687ce273c3f82ad67d876e32d9c3aa809881843d8aaafcbe"
TRACK = WORK / "long_continuation_through59760_v1"
STAGE = WORK / "inputs/long_source_masks_49800-348600_v1"
D2 = STAGE / "runs/run_0004"
SAMPLE_SECONDS, MAX_SECONDS = 10.0, 3600.0
COLUMNS = (
    "observed_utc",
    "monotonic_elapsed_seconds",
    "gpu_index",
    "nvidia_local_timestamp",
    "memory_used_MiB",
    "memory_total_MiB",
    "gpu_utilization_percent",
    "power_W",
    "tracker_terminal_status",
    "detector_terminal_status",
)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def utc():
    return datetime.now(UTC).isoformat()


def json_observation(path):
    """Missing or temporarily partial JSON is not a terminal event."""
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return None, None, None
    try:
        return json.loads(content), sha_bytes(content), content
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, sha_bytes(content), content


def terminal(path, stage):
    value, digest, content = json_observation(path)
    if value is None:
        return None
    if value.get("stage") != stage or value.get("status") not in ("ok", "failed"):
        return None
    return dict(
        path=str(path),
        sha256=digest,
        status=value["status"],
        started_at=value["started_at"],
        finished_at=value["finished_at"],
        requested_pause="PreparationPaused" in value.get("error", ""),
        _content=content,
    )


def latest_progress(path):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 32768))
            rows = stream.read().splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(rows):
        try:
            return json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return None


def number(value):
    try:
        return float(value.strip())
    except ValueError:
        return None


def sample_gpu(executable):
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=index,timestamp,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=8,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    rows = []
    for row in csv.reader(io.StringIO(completed.stdout)):
        if len(row) != 6:
            raise ValueError("unexpected nvidia-smi six-field row")
        rows.append(
            dict(
                zip(
                    COLUMNS[2:8],
                    [row[0].strip(), row[1].strip(), *[number(v) for v in row[2:]]],
                    strict=True,
                )
            )
        )
    if not rows:
        raise ValueError("nvidia-smi returned no GPU rows")
    return rows


def public(value):
    return {key: item for key, item in value.items() if not key.startswith("_")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(WORK):
        raise ValueError("require a new work/three_body output")
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is not available")
    source_paths = (Path(__file__).resolve(), REPO / "src/medaka3d/provenance.py")
    inputs = {str(path): sha_bytes(path.read_bytes()) for path in source_paths}
    plan_raw = PLAN.read_bytes()
    if sha_bytes(plan_raw) != PLAN_SHA:
        raise ValueError("registered pipeline plan changed")
    plan = json.loads(plan_raw)
    protocol = WORK / "LONG_PIPELINE_EXECUTION_PLAN.md"
    if sha_bytes(protocol.read_bytes()) != plan["execution_plan_sha256"]:
        raise ValueError("registered execution protocol changed")
    inputs.update({str(PLAN): PLAN_SHA, str(protocol): plan["execution_plan_sha256"]})
    if plan["tracking_output"] != TRACK.name or plan["next_detector_interval"] != [59760, 66400]:
        raise ValueError("fixed observed run/window differs")
    output.mkdir(parents=True, exist_ok=False)
    with provenance(
        "long_pipeline_sampled_resource_observation", output, name="resources"
    ) as record:
        for path, digest in inputs.items():
            record.input(path, path, sha256=digest)
        snapshot_hashes = {}
        for source in source_paths:
            target = output / "code_snapshot" / source.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            snapshot_hashes[str(target)] = inputs[str(source)]
        record.param("sample_interval_seconds", SAMPLE_SECONDS)
        record.param("maximum_seconds", MAX_SECONDS)
        record.param("tracking_run", str(TRACK))
        record.param("detector_run", str(D2))
        record.param("tracking_native_range", [53120, 59760])
        record.param("detector_native_range", [59760, 66400])
        record.param("nvidia_smi", executable)
        record.param("GPU_context_created", False)
        record.param("other_processes_polled_or_modified", False)
        samples, lifecycle, errors, completed_chunks = [], [], [], {}
        seals = {}
        start, deadline = time.monotonic(), time.monotonic()
        stop_reason = None
        with (
            (output / "gpu_samples.csv").open("x", newline="", encoding="utf-8") as csv_file,
            (output / "lifecycle.jsonl").open("x", encoding="utf-8") as event_file,
        ):
            writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
            writer.writeheader()
            while True:
                timestamp = utc()
                elapsed = time.monotonic() - start
                statuses = {}
                for name, path, stage in (
                    (
                        "tracker",
                        TRACK / "continuation.provenance.json",
                        "three_body_long_continuation",
                    ),
                    (
                        "detector",
                        D2 / "run.provenance.json",
                        "restartable_independent_mask_preparation",
                    ),
                ):
                    state = terminal(path, stage)
                    statuses[name] = None if state is None else state["status"]
                    if state is not None:
                        seals[name] = state
                for first in range(59760, 66400, 1660):
                    folder = STAGE / "chunks" / f"f{first:06d}-{first + 1660:06d}"
                    for attempt in sorted(folder.glob("attempt_*")):
                        state = terminal(
                            attempt / "staging.provenance.json", "prepare_independent_mask_chunk"
                        )
                        if state is not None:
                            completed_chunks[str(attempt)] = {
                                **state,
                                "frame_range": [first, first + 1660],
                            }
                try:
                    rows = sample_gpu(executable)
                    for row in rows:
                        row.update(
                            observed_utc=timestamp,
                            monotonic_elapsed_seconds=elapsed,
                            tracker_terminal_status=statuses["tracker"],
                            detector_terminal_status=statuses["detector"],
                        )
                        writer.writerow(row)
                        samples.append(row)
                        used, total = row["memory_used_MiB"], row["memory_total_MiB"]
                        if used is not None and total and used / total >= 0.9:
                            print(
                                json.dumps(
                                    dict(alert="sampled_GPU_memory_at_least_90_percent", **row)
                                ),
                                flush=True,
                            )
                    csv_file.flush()
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    error = dict(observed_utc=timestamp, error=f"{type(exc).__name__}: {exc}")
                    errors.append(error)
                    print(json.dumps(dict(alert="sampling_error", **error)), flush=True)
                event = dict(
                    observed_utc=timestamp,
                    elapsed_seconds=elapsed,
                    terminal_status=statuses,
                    tracker_latest_progress=latest_progress(TRACK / "progress.jsonl"),
                    observed_complete_chunk_seals=[public(v) for v in completed_chunks.values()],
                )
                lifecycle.append(event)
                event_file.write(json.dumps(event, allow_nan=False) + "\n")
                event_file.flush()
                if len(seals) == 2:
                    stop_reason = "both_run_terminal_seals_observed"
                elif (output / "STOP").exists():
                    stop_reason = "explicit_STOP_file"
                elif elapsed >= MAX_SECONDS:
                    stop_reason = "maximum_60_minutes"
                if stop_reason:
                    break
                print(
                    json.dumps(
                        dict(
                            sample=len(lifecycle),
                            elapsed_seconds=round(elapsed, 3),
                            terminal_status=statuses,
                            chunks=len(completed_chunks),
                        )
                    ),
                    flush=True,
                )
                deadline += SAMPLE_SECONDS
                time.sleep(max(0, min(deadline - time.monotonic(), MAX_SECONDS - elapsed)))
        terminal_records = {}
        for name, state in {
            **seals,
            **{f"chunk_{i}": s for i, s in enumerate(completed_chunks.values())},
        }.items():
            target = output / "terminal_seal_snapshots" / f"{name}.json"
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(state["_content"])
            snapshot_hashes[str(target)] = state["sha256"]
            record.input(state["path"], state["path"], sha256=state["sha256"])
            inputs[state["path"]] = state["sha256"]
            terminal_records[name] = public(state)
        overlap = None
        if len(seals) == 2:
            a, b = seals.values()
            first = max(
                datetime.fromisoformat(a["started_at"]), datetime.fromisoformat(b["started_at"])
            )
            last = min(
                datetime.fromisoformat(a["finished_at"]), datetime.fromisoformat(b["finished_at"])
            )
            overlap = dict(
                started_at=first.isoformat(),
                finished_at=last.isoformat(),
                seconds=max(0, (last - first).total_seconds()),
                scope="stage intervals include preflight/final hashes; not GPU kernel overlap",
            )
        maxima = {}
        for gpu in sorted({row["gpu_index"] for row in samples}):
            rows = [row for row in samples if row["gpu_index"] == gpu]
            maxima[gpu] = {
                key: max((row[key] for row in rows if row[key] is not None), default=None)
                for key in ("memory_used_MiB", "gpu_utilization_percent", "power_W")
            }
        summary = dict(
            schema="long_pipeline_sampled_resources_v1",
            stop_reason=stop_reason,
            samples=len(samples),
            sampling_instants=len(lifecycle),
            errors=errors,
            first_observation=lifecycle[0]["observed_utc"],
            final_observation=lifecycle[-1]["observed_utc"],
            monotonic_elapsed_seconds=time.monotonic() - start,
            sampled_maxima_by_GPU=maxima,
            terminal_records=terminal_records,
            artifact_stage_overlap=overlap,
            complete_new_chunks=len(completed_chunks),
            limits=[
                "Ten-second observations are not continuous peaks or kernel traces.",
                "Logging started after producer launch; early load may be missed.",
                "Only stage seal intervals establish process-stage overlap.",
                "No causal speedup is inferred from different clips or historical runs.",
                "Terminal seals are recorded, not a replacement for independent data/ID audits.",
            ],
        )
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        for path in (
            output / "gpu_samples.csv",
            output / "lifecycle.jsonl",
            output / "summary.json",
        ):
            snapshot_hashes[str(path)] = sha_bytes(path.read_bytes())
        for path, digest in snapshot_hashes.items():
            record.output(path, path, sha256=digest)
        for path, digest in {**inputs, **snapshot_hashes}.items():
            if sha_bytes(Path(path).read_bytes()) != digest:
                raise ValueError("logger source/terminal evidence/output changed: " + path)
        record.param("all_input_source_snapshot_output_SHA_unchanged", True)
    print(json.dumps(dict(status="complete", stop_reason=stop_reason, output=str(output))))


if __name__ == "__main__":
    main()

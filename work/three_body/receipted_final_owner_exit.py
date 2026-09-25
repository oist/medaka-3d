"""Hold only the registered supervisor handle and bind its eventual final exit.

Reuses the frozen Windows query/synchronize implementation without terminate
rights. No process discovery, signals, tracking, model or scientific payloads.
The audit consumer reads the sealed receipt; it never attaches to a process.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import capture_owned_long_exit as frozen

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
ROOT = WORK / "long_remaining_receipted_child_supervisor_v1"
SHARED_LOCK = WORK / "long_remaining_receipted_child_supervisor.lock"
SUPERVISOR = WORK / "run_receipted_child_supervisor.py"
SUPERVISOR_SHA = "33abe705b76e40f2b0b399cf5267e5c540a074b021bc91f5ec400cf4db7b59b0"
FROZEN_SHA = "6a5fbed6209a5baeb56166af3c66e1eeb988027b0128df46422e0a7a97278a3e"
OWNER_STAGE = "long_remaining_receipted_child_supervisor_v1"
STAGE = "receipted_final_owner_exit_v1"
require, sha, utc, now = frozen.require, frozen.sha, frozen.utc, frozen.now


def read(path):
    return frozen.decode(Path(path).read_bytes())


def local(path, *, exists=True):
    path = Path(path).resolve(strict=exists)
    require(path.is_relative_to(WORK), "local workspace evidence required")
    return path


def validate_header(header):
    require(
        header["stage"] == OWNER_STAGE
        and header["status"] == "ok"
        and header["params"]["full_sequence_complete"] is True
        and header["params"]["all_input_source_snapshot_output_sha_unchanged"] is True,
        "successful full supervisor required",
    )
    utc(header["finished_at"])


def validate_summary(summary):
    expected = dict(
        schema=OWNER_STAGE,
        status="complete",
        through=348600,
        full_sequence_complete=True,
        prefix_backtrace_provisional=False,
        logical_initialization_count=1,
        checkpoint_restore_count=32,
        external_pose_reset_count=0,
        tracking_processes=33,
        consumed_chunk_count=180,
        incomplete_phase=None,
    )
    for name, value in expected.items():
        require(
            type(summary.get(name)) is type(value) and summary[name] == value,
            "final supervisor summary differs: " + name,
        )


def validate_identity(identity, binding, *, allowed_images):
    lock = binding["lock"]
    require(
        type(identity["pid"]) is int
        and identity["pid"] == lock["pid"]
        and identity["live_when_handle_acquired"] is True
        and identity["access_mask"] == 0x00101000
        and Path(identity["image_path"]).resolve() in allowed_images,
        "held process identity/rights differ",
    )
    require(
        type(identity["creation_filetime_100ns"]) is int
        and identity["creation_filetime_100ns"] > 0
        and frozen.filetime_utc(identity["creation_filetime_100ns"])
        == identity["creation_time_utc"]
        and utc(identity["creation_time_utc"])
        <= utc(lock["created_at"])
        <= utc(identity["attached_at"]),
        "process creation/attachment chronology differs",
    )


def validate_exit(exit_info, identity, header, *, audit_started_at):
    validate_header(header)
    require(
        all(exit_info.get(k) == v for k, v in identity.items())
        and type(exit_info["exit_code"]) is int
        and exit_info["exit_code"] == 0
        and exit_info["wait_signalled"] is True
        and exit_info["exit_code_success"] is True
        and type(exit_info["exit_filetime_100ns"]) is int
        and exit_info["exit_filetime_100ns"] > identity["creation_filetime_100ns"]
        and frozen.filetime_utc(exit_info["exit_filetime_100ns"]) == exit_info["exit_time_utc"],
        "matching held owner exit0 required",
    )
    require(
        utc(identity["attached_at"])
        <= utc(header["finished_at"])
        <= utc(exit_info["exit_time_utc"])
        <= utc(exit_info["exit_observed_at"])
        <= utc(audit_started_at),
        "final owner exit chronology differs",
    )


def metadata_binding(supervisor_root, owner_pid):
    root = local(supervisor_root)
    require(root == ROOT.resolve(), "different registered supervisor")
    require(type(owner_pid) is int and 0 < owner_pid <= 0xFFFFFFFF, "invalid owned PID")
    inputs = {}

    def bind(path, digest=None):
        path = local(path)
        actual = sha(path)
        require(digest is None or digest == actual, "binding SHA differs")
        inputs[str(path)] = actual
        return read(path) if path.suffix == ".json" or path.suffix == ".lock" else None

    bind(SUPERVISOR, SUPERVISOR_SHA)
    bind(frozen.__file__, FROZEN_SHA)
    bind(__file__)
    lock = bind(SHARED_LOCK)
    bind(root / "supervisor.lock", inputs[str(SHARED_LOCK)])
    require(
        lock["schema"] == OWNER_STAGE
        and type(lock["pid"]) is int
        and lock["pid"] == owner_pid
        and local(lock["output_root"]) == root
        and local(lock["source_path"]) == SUPERVISOR.resolve()
        and lock["source_sha256"] == SUPERVISOR_SHA
        and lock["full_target_interval"] == [49800, 348600],
        "registered owner lock differs",
    )
    plan = bind(root / "supervisor_plan.json")
    require(
        plan["schema"] == OWNER_STAGE
        and plan["target_interval"] == [49800, 348600]
        and plan["source_hashes"].get(str(SUPERVISOR)) == SUPERVISOR_SHA,
        "registered owner plan differs",
    )
    return dict(supervisor_root=str(root), lock=lock, inputs=inputs)


def capture(
    supervisor_root, owner_pid, output, *, process_factory=frozen.WindowsProcess, sleeper=time.sleep
):
    binding = metadata_binding(supervisor_root, owner_pid)
    output = local(output, exists=False)
    require(not output.exists(), "new receipt output required")
    output.mkdir(parents=True)
    started, outputs, process = now(), {}, None
    inputs = dict(binding["inputs"])

    def save(name, value):
        path = output / name
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        outputs[name] = dict(path=str(path), sha256=sha(path), size_bytes=path.stat().st_size)

    def bind_named(header, role, filename):
        path = local(Path(binding["supervisor_root"]) / filename)
        row = header["outputs"][role]
        require(
            local(row["path"]) == path and sha(path) == row["sha256"],
            "named owner evidence differs",
        )
        require(path.stat().st_size == row["size"], "named owner evidence size differs")
        require(
            str(path) not in inputs or inputs[str(path)] == row["sha256"],
            "initial owner evidence changed before final seal",
        )
        inputs[str(path)] = row["sha256"]
        return read(path)

    try:
        for path in (__file__, frozen.__file__):
            target = output / Path(path).name
            shutil.copyfile(path, target)
            require(sha(target) == inputs[str(Path(path).resolve())], "observer snapshot differs")
            outputs[target.name] = dict(
                path=str(target), sha256=sha(target), size_bytes=target.stat().st_size
            )
        save("binding.json", binding)
        allowed = {Path(sys.executable).resolve(), Path(sys._base_executable).resolve()}
        save(
            "environment.json",
            dict(python=sys.version, allowed_owner_images=sorted(map(str, allowed))),
        )
        process = process_factory(owner_pid)
        validate_identity(process.identity, binding, allowed_images=allowed)
        save("held_identity.json", process.identity)
        print(
            json.dumps(
                dict(
                    event="final_owner_handle_held",
                    owner_pid=owner_pid,
                    output=str(output),
                    terminate_rights=False,
                )
            ),
            flush=True,
        )
        last_heartbeat = time.monotonic()
        while not process.signalled():
            sleeper(10)
            if time.monotonic() - last_heartbeat >= 60:
                print(
                    json.dumps(dict(event="waiting_for_final_owner_exit", owner_pid=owner_pid)),
                    flush=True,
                )
                last_heartbeat = time.monotonic()
        exit_info = process.exit_info()
        save("held_exit.json", exit_info)
        seal_path = local(Path(binding["supervisor_root"]) / "supervisor.provenance.json")
        inputs[str(seal_path)] = sha(seal_path)
        header = read(seal_path)
        validate_exit(exit_info, process.identity, header, audit_started_at=now())
        summary = bind_named(header, "summary", "summary.json")
        validate_summary(summary)
        require(
            bind_named(header, "lock", "supervisor.lock") == binding["lock"], "final lock differs"
        )
        bind_named(header, "plan", "supervisor_plan.json")
        save(
            "summary.json",
            dict(
                status="ok",
                supervisor_root=binding["supervisor_root"],
                supervisor_seal_sha256=inputs[str(seal_path)],
                owner_pid=owner_pid,
                held_exit_code=0,
                os_exit_time_utc=exit_info["exit_time_utc"],
                observed_at=exit_info["exit_observed_at"],
                final_audit_complete=False,
                identity_accuracy=False,
            ),
        )
        for path, digest in inputs.items():
            require(sha(path) == digest, "receipt input changed")
        for row in outputs.values():
            require(sha(row["path"]) == row["sha256"], "receipt output changed")
        require(
            {p for p in output.rglob("*") if p.is_file()}
            == {Path(r["path"]) for r in outputs.values()},
            "receipt membership differs",
        )
        status, error = "ok", None
    except BaseException as exc:
        status, error = "failed", dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if process is not None:
            process.close()
        # No success seal can be produced by the exception path.
        with (output / "owner_exit.provenance.json").open("x", encoding="utf-8") as stream:
            json.dump(
                dict(
                    stage=STAGE,
                    status=locals().get("status", "failed"),
                    started_at=started,
                    finished_at=now(),
                    inputs=inputs,
                    outputs=outputs,
                    error=locals().get("error"),
                    final_audit_complete=False,
                ),
                stream,
                indent=2,
            )
            stream.write("\n")
    return output


def verify_final_owner_exit(
    receipt_path,
    expected_receipt_sha,
    supervisor_root,
    supervisor_header,
    *,
    audit_started_at,
    evidence,
):
    """Metadata-only completed-owner gate, before any terminal-array loading."""
    path, root = local(receipt_path), local(supervisor_root)
    require(
        isinstance(expected_receipt_sha, str)
        and len(expected_receipt_sha) == 64
        and set(expected_receipt_sha) <= set("0123456789abcdef"),
        "explicit receipt SHA256 required",
    )
    require(
        root == ROOT.resolve() and path.name == "owner_exit.provenance.json",
        "wrong owner receipt scope",
    )
    evidence.add(path, expected_receipt_sha)
    receipt = read(path)
    require(
        receipt["stage"] == STAGE and receipt["status"] == "ok", "completed owner receipt required"
    )
    require(utc(receipt["finished_at"]) <= utc(audit_started_at), "receipt is later than audit")
    records = receipt["outputs"]
    expected_roles = {
        Path(__file__).name,
        Path(frozen.__file__).name,
        "binding.json",
        "environment.json",
        "held_identity.json",
        "held_exit.json",
        "summary.json",
    }
    require(set(records) == expected_roles, "complete owner receipt roles required")
    payloads = {}
    for name, row in records.items():
        item = local(row["path"])
        require(item.parent == path.parent and item.name == name, "receipt output escaped")
        evidence.add(item, row["sha256"])
        require(item.stat().st_size == row["size_bytes"], "receipt output size differs")
        if name.endswith(".json"):
            payloads[name] = read(item)
    require(
        {p for p in path.parent.rglob("*") if p.is_file()}
        == {path, *(Path(r["path"]) for r in records.values())},
        "unexpected receipt output",
    )
    for name, digest in receipt["inputs"].items():
        evidence.add(local(name), digest)
    require(
        receipt["inputs"].get(str(SUPERVISOR)) == SUPERVISOR_SHA
        and receipt["inputs"].get(str(Path(frozen.__file__).resolve())) == FROZEN_SHA
        and receipt["inputs"].get(str(Path(__file__).resolve())) == sha(__file__),
        "receipt producer source differs",
    )
    for source in (__file__, frozen.__file__):
        require(
            records[Path(source).name]["sha256"] == receipt["inputs"][str(Path(source).resolve())],
            "receipt source snapshot differs",
        )
    selected_seal = root / "supervisor.provenance.json"
    require(
        receipt["inputs"].get(str(selected_seal)) == sha(selected_seal)
        and read(selected_seal) == supervisor_header,
        "different selected supervisor seal",
    )
    binding = payloads["binding.json"]
    require(local(binding["supervisor_root"]) == root, "different owner root")
    current = metadata_binding(root, binding["lock"]["pid"])
    require(binding == current, "initial ownership binding changed")
    identity, exit_info = payloads["held_identity.json"], payloads["held_exit.json"]
    require(
        utc(receipt["started_at"])
        <= utc(identity["attached_at"])
        <= utc(exit_info["exit_observed_at"])
        <= utc(receipt["finished_at"]),
        "held observation is outside observer lifetime",
    )
    allowed = {Path(sys.executable).resolve(), Path(sys._base_executable).resolve()}
    require(
        set(payloads["environment.json"]["allowed_owner_images"]) == set(map(str, allowed)),
        "receipt Python environment differs",
    )
    validate_identity(identity, binding, allowed_images=allowed)
    validate_exit(exit_info, identity, supervisor_header, audit_started_at=audit_started_at)
    validate_summary(read(root / "summary.json"))
    expected_summary = dict(
        status="ok",
        supervisor_root=str(root),
        supervisor_seal_sha256=sha(selected_seal),
        owner_pid=identity["pid"],
        held_exit_code=0,
        os_exit_time_utc=exit_info["exit_time_utc"],
        observed_at=exit_info["exit_observed_at"],
        final_audit_complete=False,
        identity_accuracy=False,
    )
    require(
        payloads["summary.json"] == expected_summary
        and all(type(payloads["summary.json"][k]) is type(v) for k, v in expected_summary.items()),
        "receipt summary binding differs",
    )
    return dict(
        status="ok",
        owner_pid=identity["pid"],
        receipt_sha256=expected_receipt_sha,
        supervisor_seal_sha256=sha(selected_seal),
        exit_code=0,
        os_exit_time_utc=exit_info["exit_time_utc"],
        terminal_observed_at=exit_info["exit_observed_at"],
        final_audit_complete=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--owner-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observe", action="store_true")
    args = parser.parse_args()
    require(args.observe, "explicit --observe required")
    print(json.dumps(dict(output=str(capture(args.supervisor, args.owner_pid, args.output)))))

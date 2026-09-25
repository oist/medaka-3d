"""Explicitly stop our old supervisor after its registered child succeeds.

No tracker launch or checkpoint edit. Uses the already tested owned-tree stop
routine while retaining an OS handle, so its PID cannot refer to a reused object.
The independently running exit witness remains responsible for both exit records.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import capture_owned_long_exit as witness
import run_long_remaining_supervisor as old

WORK = old.WORK
WITNESS_ROOT = WORK / "long_old_supervisor_exit_witness_v1"
CHILD = WORK / "long_remaining_supervisor_v1/phases/f066400-076360/tracking"
REQUIRED_CODE = {
    str(WORK / name) for name in (
        "run_long_receipted_supervisor.py", "receipted_long_handoff.py",
        "receipted_long_phase_operation.py", "receipted_long_phase_audit.py",
        "cached_long_tracking_inputs.py", "mask_validation_receipts.py",
        "capture_owned_long_exit.py", "run_long_remaining_supervisor.py",
    )
}
require = old.require


def preflight(adoption_sources):
    hashes = {str(Path(__file__).resolve()): witness.sha(__file__)}

    def bind(path, digest):
        name = str(path)
        require(name not in hashes or hashes[name] == digest, "conflicting evidence SHA")
        hashes[name] = digest

    def read(path):
        path = witness.within(path)
        data = path.read_bytes()
        bind(path, witness.digest(data))
        return witness.decode(data)

    approved = read(adoption_sources)
    require(approved.get("schema") == "reviewed_receipted_handoff_sources_v1"
            and approved.get("parent") == str(CHILD)
            and approved.get("target_interval") == [49800, 348600], "wrong adoption source plan")
    require(set(approved["source_sha256"]) == REQUIRED_CODE, "wrong adoption source membership")
    for path, digest in approved["source_sha256"].items():
        bind(path, digest)
    require(hashes[str(witness.OLD_SOURCE)] == witness.OLD_SOURCE_SHA,
            "old supervisor source is not the witnessed version")
    require(hashes[str(WORK / "capture_owned_long_exit.py")]
            == "6a5fbed6209a5baeb56166af3c66e1eeb988027b0128df46422e0a7a97278a3e",
            "wrong process witness implementation")
    witness.guard(hashes)
    root = witness.within(WITNESS_ROOT)
    require(not (root / "owner_exit.json").exists(),
            "owner already exited; do not issue another stop")
    receipt = read(root / "child_exit.receipt.json")
    require(receipt.get("schema") == witness.SCHEMA and receipt.get("status") == "ok"
            and receipt.get("role") == "child"
            and receipt.get("all_evidence_sha_unchanged") is True,
            "successful child exit observation required")
    require({str(root / name) for name in (
        "binding.json", "attachment.json", "child_exit.json", "events_baseline.jsonl",
    )}.issubset(receipt["outputs"]), "child exit receipt lacks required bound outputs")
    for kind in ("inputs", "outputs"):
        for name, digest in receipt[kind].items():
            path = witness.within(name)
            require(kind != "outputs" or path.is_relative_to(root), "foreign witness output")
            bind(path, digest)
    witness.guard(hashes)
    binding = read(root / "binding.json")
    current, _ = witness.build_binding(binding["supervisor_root"], binding["child_pid"])
    varying = {"baseline_events_sha256", "baseline_events_bytes"}
    require(old.equal({k: v for k, v in current.items() if k not in varying},
                      {k: v for k, v in binding.items() if k not in varying}),
            "current owner lock/registered event differs from witness binding")
    for path, digest in current["immutable_hashes"].items():
        bind(path, digest)
    attachment = read(root / "attachment.json")
    child_exit = read(root / "child_exit.json")
    require(binding["owner_pid"] == binding["owner_lock"]["pid"]
            and witness.within(binding["child_event"]["output"]) == CHILD,
            "witness belongs to another owner/child")
    observation = child_exit["observation"]
    require(child_exit["role"] == "child" and observation["pid"] == binding["child_pid"]
            and observation["exit_code"] == 0 and observation["wait_signalled"] is True
            and observation["creation_filetime_100ns"]
            == attachment["processes"]["child"]["creation_filetime_100ns"],
            "registered child has not been observed exiting successfully")
    require(child_exit["binding_sha256"] == hashes[str(root / "binding.json")],
            "child observation belongs to another binding")
    seal = read(CHILD / "continuation.provenance.json")
    require(seal.get("status") == "ok" and seal.get("stage") == "three_body_long_continuation"
            and seal["params"].get("all_input_source_snapshot_output_sha_unchanged") is True
            and seal["params"]["through_frame"] == 76360,
            "successful complete registered child required")
    for info in seal["outputs"].values():
        path = witness.within(info["path"])
        require(path.is_relative_to(CHILD) and path.stat().st_size == info["size"],
                "child output scope/size differs")
        bind(path, info["sha256"])
    require(witness.utc(seal["finished_at"]) <= witness.utc(observation["exit_time_utc"]),
            "child success was written after its observed exit")
    witness.guard(hashes)
    return binding, attachment["processes"]["owner"], hashes


def no_later_launch(binding):
    data = Path(binding["events_path"]).read_bytes()
    baseline = (WITNESS_ROOT / "events_baseline.jsonl").read_bytes()
    require(data.startswith(baseline), "witness event prefix changed")
    require(not data or data.endswith(b"\n"), "partial final event line; launch state is uncertain")
    rows = witness.event_rows(data)
    require(not any(row.get("event") in ("phase_start", "owned_child_started")
                    for row in rows[binding["child_event_index"] + 1:]),
            "a later phase launch is recorded; do not start a replacement tracker")
    return data


class HeldOwner:
    """Small process-handle interface for the frozen owned-tree cleanup routine."""

    def __init__(self, process):
        self.process, self.pid = process, process.pid

    def poll(self):
        return self.process.exit_info()["exit_code"] if self.process.signalled() else None

    @property
    def returncode(self):
        return self.poll()

    def wait(self, timeout):
        until = time.monotonic() + timeout
        while not self.process.signalled():
            if time.monotonic() >= until:
                raise subprocess.TimeoutExpired("held owner", timeout)
            time.sleep(.05)
        return self.process.exit_info()["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adoption-sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    binding, expected_owner, hashes = preflight(args.adoption_sources)
    no_later_launch(binding)
    if not args.execute:
        print("Preflight passed; no process signal performed.", flush=True)
        return
    output = witness.within(args.output, exists=False)
    output.mkdir(parents=True, exist_ok=False)
    info = dict(target_role="old_supervisor", pid=binding["owner_pid"], child_exit_code=0)
    stage = "stop_owned_long_supervisor_after_child_success"
    with old.provenance(stage, output, name="stop") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        process = witness.WindowsProcess(binding["owner_pid"])
        try:
            require(process.identity["creation_filetime_100ns"]
                    == expected_owner["creation_filetime_100ns"]
                    and Path(process.identity["image_path"]).resolve()
                    == Path(expected_owner["image_path"]).resolve(), "owner PID was reused")
            info["held_identity"] = process.identity
            before = no_later_launch(binding)
            (output / "events_before.jsonl").write_bytes(before)
            record.output("events_before", output / "events_before.jsonl",
                          sha256=witness.digest(before))
            old.stop_owned_child(HeldOwner(process), info)
            require(process.signalled(), "owned supervisor has not exited")
            info["observed_owner_exit"] = process.exit_info()
            after = no_later_launch(binding)
            (output / "events_after.jsonl").write_bytes(after)
            record.output("events_after", output / "events_after.jsonl",
                          sha256=witness.digest(after))
            record.param("no_later_tracking_launch_recorded", True)
            record.param("tracking_checkpoint_or_old_seal_changed_by_helper", False)
        finally:
            process.close()
            old.save_json(output / "termination.json", info, record, "termination")
            witness.guard(hashes)
            old.final_outputs(record)


if __name__ == "__main__":
    main()

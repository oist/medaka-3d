"""Explicit 306-frame continuous branch after ROOT-owned CUDA qualification exit.

The original fixed checkpoint is validated and imported without changing any
state array. A successful qualification seal alone does not start GPU fitting:
ROOT must also record its actual owned exec-session exit in a separate witness.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import fit_chest_recovery_candidates as base
import numpy as np
import replay_chest_deficit as original
import torch
from chest_candidate_diagnostics import CandidateDiagnostics
from continuous_tracker import RECORD_FIELDS, AcceleratedContinuousPersistentNodeTracker
from directional_chest_tracker import DirectionalChestContinuousTracker, import_original_state
from mask_validation_receipts import ReceiptStore
from node_recovery import NodeRecoveryConfig
from persistent_node_tracker import PersistentNodeConfig
from replay_gpu_window import ReplayGPUWindow
from tracker import TrackerConfig
from tracker_checkpoint import read_checkpoint, write_checkpoint

from medaka3d.provenance import provenance

REPO = Path(__file__).resolve().parents[2]
PLAN = Path(__file__).with_name("DIRECTIONAL_CHEST_CONTINUOUS_PLAN.md")
checks = base.evidence_tools


def await_qualification_exit(root, witness, window, evidence):
    started = time.monotonic()
    while not witness.is_file():
        window.assert_open()
        checks.require(time.monotonic() - started < 900, "ROOT qualification exit witness timeout")
        time.sleep(2)
    value = original.read(witness)
    checks.require(
        value.get("schema") == "ROOT_owned_directional_CUDA_exit_v1"
        and value.get("exec_session_id") == 96265
        and value.get("exit_code") == 0
        and Path(value["qualification_root"]).resolve() == root,
        "successful ROOT-owned qualification exit required",
    )
    evidence.add(witness)
    path = root / "qualification.provenance.json"
    evidence.add(path, value["qualification_seal_sha256"])
    hashes = {}
    seal = checks.verify(root, path.name, "directional_CUDA_window_qualification", hashes)
    for p, digest in hashes.items():
        evidence.add(p, digest)
    result = original.read(root / "summary.json")
    checks.require(
        result["status"] == "ok"
        and result["CUDA_qualified_for_fixed_four_frames"]
        and result["frames"] == [66660, 66664, 66667, 66696],
        "fixed CUDA qualification incomplete",
    )
    for name in ("directional_device_fitter.py", "directional_chest_fitter.py"):
        p = Path(__file__).with_name(name)
        checks.require(
            checks.sha(p) == seal["inputs"][str(p)]["sha256"], "qualified fitter changed"
        )
    window.assert_open()
    return value


def main(args):
    output = args.output.resolve()
    checks.require(
        output.is_relative_to(REPO / "work") and not output.exists(),
        "new workspace output required",
    )
    qualification, witness = args.qualification.resolve(), args.qualification_witness.resolve()
    checks.require(
        qualification.is_relative_to(REPO / "work") and witness.is_relative_to(REPO / "work"),
        "local qualification evidence required",
    )
    code = {**base.local_code(), str(PLAN): checks.sha(PLAN)}
    output.mkdir(parents=True)
    evidence = original.phase.Digests()
    for p, digest in code.items():
        evidence.add(p, digest)
    window = ReplayGPUWindow(args.supervisor_root, args.child_pid, output)
    store = ReceiptStore(args.receipt_root.resolve())
    iterator = tracker = diagnostic = None
    rows = []
    result = dict(
        schema="directional_chest_continuous_replay_v1",
        status="incomplete",
        native_interval=[original.START, original.STOP],
        replay_complete=False,
        external_pose_resets=0,
        independent_ID_truth=False,
        adopted=False,
    )
    with provenance("directional_chest_continuous_replay", output, name="replay") as record:
        try:
            window.attach()
            original.snapshot(code, output, record)
            started = time.perf_counter()
            context = original.prepare(evidence, store)
            binding, protocol, parent = context["binding"], context["protocol"], context["parent"]
            torch.set_num_threads(binding["torch_cpu_threads"])
            original.runtime(binding, gpu=False)
            cfg = TrackerConfig(**protocol["tracker_config"])
            kwargs = dict(
                recovery_config=NodeRecoveryConfig(**protocol["node_recovery_config"]),
                persistence_config=PersistentNodeConfig(**protocol["persistence_config"]),
            )
            seed, rig = context["inputs"].initial_pose_xyz.copy(), context["inputs"].rig
            source = AcceleratedContinuousPersistentNodeTracker(rig, seed, cfg, **kwargs)
            tracker = DirectionalChestContinuousTracker(rig, seed, cfg, **kwargs)
            migration = import_original_state(tracker, source, parent["arrays"], parent["metadata"])
            checks.require(
                len(tracker._states) == 6
                and tracker.previous_frame == original.START - 1
                and tracker.processed_frames == original.START - original.FIRST,
                "full source state not retained",
            )
            original.save_json(output / "state_import.json", migration)
            iterator = original.bounded_frames(context["reader"])
            frame = next(iterator)
            original.save_json(
                output / "ready.json",
                dict(
                    native_frame=frame.frame_idx,
                    preparation_seconds=time.perf_counter() - started,
                    logical_initialize_count=tracker.initialize_count,
                    restore_count=tracker.restore_count,
                    all_original_state_arrays_preserved=True,
                    original_previous_observation=context["previous_observation"],
                    GPU_fit_calls=0,
                ),
            )
            evidence.final_guard()
            store.final_guard()
            original.unchanged(code)
            print(
                json.dumps(dict(event="branch_CPU_prepared_waiting", child_pid=args.child_pid)),
                flush=True,
            )
            window.wait_until_open()
            qualification_exit = await_qualification_exit(qualification, witness, window, evidence)
            original.runtime(binding, gpu=True)
            original.unchanged(code)
            fit_start = time.perf_counter()
            with CandidateDiagnostics(tracker, frame_range=original.DIAGNOSTIC_RANGE) as diagnostic:
                for native in range(original.START, original.STOP):
                    if native != original.START:
                        frame = next(iterator)
                    window.assert_open()
                    with diagnostic.frame(native):
                        tracker.advance(frame)
                    torch.cuda.synchronize()
                    window.assert_open()
                    rows.append(
                        dict(
                            frame_idx=native,
                            **tracker.last_chest_proposal_diagnostic,
                            enabled_parent_fish=int(tracker.last_diagnostics["enabled"].sum()),
                        )
                    )
                    if (native - original.START + 1) % 25 == 0 or native == original.STOP - 1:
                        print(
                            json.dumps(
                                dict(
                                    event="directional_branch_progress",
                                    frame=native,
                                    processed=native - original.START + 1,
                                )
                            ),
                            flush=True,
                        )
            fit_seconds = time.perf_counter() - fit_start
            window.assert_open()
            tracker.__dict__.get("_fit_graph_cache", {}).clear()
            torch.cuda.synchronize()
            window.close()  # No further GPU work; save/validate full history on CPU.
            original.save_json(
                output / "fit_completed.json",
                dict(
                    native_interval=[original.START, original.STOP],
                    GPU_fit_seconds=fit_seconds,
                    qualification_exit=qualification_exit,
                ),
            )
            arrays, metadata = tracker.export_state()
            prefix = original.START - original.FIRST
            for key in ("history_offsets", *("history_" + k for k in RECORD_FIELDS)):
                end = (
                    prefix + 1
                    if key == "history_offsets"
                    else parent["arrays"]["history_offsets"][-1]
                )
                prior, now = parent["arrays"][key], arrays[key][:end]
                exact = (
                    np.array_equal(prior, now)
                    if prior.dtype.kind == "U"
                    else original.byte_equal(prior, now)
                )
                checks.require(exact, "original full-beam history prefix changed: " + key)
            checkpoint_binding = dict(
                schema="directional_chest_branch_checkpoint_binding_v1",
                original_binding=binding,
                branch_code_sha256=code,
                state_import_sha256=checks.sha(output / "state_import.json"),
            )
            write_checkpoint(output / "checkpoint", arrays, metadata, bindings=checkpoint_binding)
            loaded, loaded_meta, _ = read_checkpoint(
                output / "checkpoint", expected_bindings=checkpoint_binding
            )
            checks.require(
                loaded_meta == metadata
                and all(original.byte_equal(loaded[k], arrays[k]) for k in arrays),
                "branch checkpoint roundtrip differs",
            )
            candidate = tracker.finish()
            np.savez_compressed(output / "candidate.npz", **candidate)
            scope = candidate["frame_idx"] >= original.START
            target = (candidate["frame_idx"] >= original.DIAGNOSTIC_RANGE[0]) & scope
            checks.require(
                int(scope.sum()) == 306
                and int(target.sum()) == 50
                and tracker.initialize_count == 1,
                "branch interval or lifecycle differs",
            )
            result.update(
                status="ok",
                replay_complete=True,
                GPU_fit_seconds=fit_seconds,
                native_frames=306,
                diagnostic_frames=len(diagnostic.records),
                logical_initialize_count=tracker.initialize_count,
                restore_count=tracker.restore_count,
                original_history_prefix_preserved=True,
                direct_node_support_306=candidate["observed"][scope].sum((0, 1)).tolist(),
                direct_node_support_target50=candidate["observed"][target].sum((0, 1)).tolist(),
                all_three_chest_frames_306=int(candidate["observed"][scope, :, 1].all(1).sum()),
                all_three_chest_frames_target50=int(
                    candidate["observed"][target, :, 1].all(1).sum()
                ),
            )
        finally:
            if iterator is not None:
                iterator.close()
            if diagnostic is not None:
                original.save_diagnostics(output, diagnostic.records)
            if window.child is not None or window.owner is not None:
                window.close()
            original.save_json(output / "summary.json", result)
            original.save_json(output / "chest_proposal_diagnostics.json", rows)
            original.save_json(output / "mask_validation_receipts.json", store.uses)
            for path, digest in evidence.paths.items():
                record.input(path, path, sha256=digest)
            for path in output.rglob("*"):
                if path.is_file() and path != output / "replay.provenance.json":
                    record.output(str(path.relative_to(output)), path, sha256=checks.sha(path))
            original.unchanged(code)
            evidence.final_guard()
            store.final_guard()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output",
        "receipt-root",
        "supervisor-root",
        "qualification",
        "qualification-witness",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--child-pid", type=int, required=True)
    main(p.parse_args())

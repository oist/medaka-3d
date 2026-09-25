"""Bounded extra GPU workload beside an owned active tracker and existing D2.

This measures memory and observed tracker throughput; it is not an exclusive GPU
lock or a controlled causal speed experiment. All 32 fits are compared with the
completed four-frame CPU reference. The new process has a 25% allocator cap.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import time
from itertools import pairwise
from pathlib import Path

import fit_chest_recovery_candidates as base
import numpy as np
import torch
from directional_device_fitter import DirectionalDeviceFitter
from qualify_directional_cuda_window import FRAMES, fit_inputs
from replay_gpu_window import ReplayGPUWindow
from run_directional_chest_local_fit import verify_control

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance

checks = base.evidence_tools
REPO = Path(__file__).resolve().parents[2]


def gpu_sample():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    rows = list(csv.reader(io.StringIO(result.stdout)))
    checks.require(
        len(rows) == 1 and rows[0][0].strip() == "NVIDIA GeForce RTX 5070 Ti",
        "fixed single GPU required",
    )
    name, total, used, utilization, power = [v.strip() for v in rows[0]]
    return dict(
        name=name,
        total_MiB=float(total),
        used_MiB=float(used),
        utilization_percent=float(utilization),
        power_W=float(power),
        sampled_monotonic=time.monotonic(),
    )


def active_progress(guard):
    checks.require(
        not guard.child.signalled() and not guard.owner.signalled(),
        "reference process ended during pilot",
    )
    guard._check_events()
    path = guard.tracking_root.parent / "tracking.log"
    data = path.read_bytes()
    committed = data[: data.rfind(b"\n") + 1]
    rows = []
    for line in committed.splitlines():
        if line.startswith(b"{"):
            value = json.loads(line)
            if "phase_processed" in value and "tracking_seconds" in value:
                rows.append(value)
    checks.require(
        all(
            b["phase_processed"] > a["phase_processed"]
            and b["tracking_seconds"] > a["tracking_seconds"]
            for a, b in pairwise(rows)
        ),
        "nonmonotonic reference progress",
    )
    return rows


def rate(first, last):
    elapsed = last["tracking_seconds"] - first["tracking_seconds"]
    return None if elapsed <= 0 else (last["phase_processed"] - first["phase_processed"]) / elapsed


def fitted_arrays(fitter, inputs, old_count):
    proposals, states, rays, _ = inputs
    pose, cost, parts, masks = fitter._fit(*inputs)
    torch.cuda.synchronize()
    plans = np.stack([p.plan for p in proposals])
    parents = np.array([p.parent for p in proposals], np.int64)
    permutations = np.array([p.permutation for p in proposals], np.int64)
    support = base.support_snapshot(pose, plans, rays, fitter.rig)
    path_cost = cost + np.array([states[p.parent].cost for p in proposals])
    selected, order = base.selected_indices(
        pose, path_cost, parents, permutations, fitter.config.beam_size
    )
    old_selected, _ = base.selected_indices(
        pose[:old_count],
        path_cost[:old_count],
        parents[:old_count],
        permutations[:old_count],
        fitter.config.beam_size,
    )
    return dict(
        fitted_pose_xyz=pose,
        local_cost=cost,
        final_loss_parts=parts,
        mask_supported=masks,
        after_strict_sources=support["strict_sources"],
        after_strict_observed=support["strict_observed"],
        selected_indices=selected,
        original_selected_indices=old_selected,
        stable_path_order=order,
    )


def main(args):
    output, replay, recovery, cpu, qualified = (
        getattr(args, k).resolve() for k in ("output", "replay", "recovery", "cpu", "qualified")
    )
    checks.require(
        output.is_relative_to(REPO / "work") and not output.exists(),
        "new workspace output required",
    )
    code = base.local_code()
    hashes = dict(code)
    output.mkdir(parents=True)
    observed = output / "registered_child"
    observed.mkdir()
    guard = ReplayGPUWindow(args.supervisor_root, args.child_pid, observed)
    records, telemetry = [], []
    before, after = [], []
    with provenance("concurrent_directional_GPU_pilot", output, name="pilot") as record:
        try:
            guard.attach()  # Ownership validation only; never wait for a natural idle window.
            run = checks.verify(
                replay, "replay.provenance.json", "chest_deficit_original_beam_replay_v1", hashes
            )
            checks.verify(
                recovery, "probe.provenance.json", "chest_recovery_counterfactual_proposals", hashes
            )
            cpu_seal = checks.verify(
                cpu, "fit.provenance.json", "directional_chest_local_CPU_fit", hashes
            )
            q = checks.verify(
                qualified,
                "qualification.provenance.json",
                "directional_CUDA_window_qualification",
                hashes,
            )
            for root, name in (
                (replay, "replay.provenance.json"),
                (recovery, "probe.provenance.json"),
            ):
                p = str(root / name)
                checks.require(
                    cpu_seal["inputs"][p]["sha256"] == hashes[p], "different CPU reference inputs"
                )
            for name in ("directional_chest_fitter.py", "directional_device_fitter.py"):
                p = Path(__file__).with_name(name)
                checks.require(
                    q["inputs"][str(p)]["sha256"] == code[str(p)], "qualified fitter changed"
                )
            registration = checks.read(replay / "registration/plan.json")
            binding = registration["original_binding"]
            calibration = Path(binding["inputs"]["seed"]["input_root"]) / base.CALIBRATION_DIR
            seed = (
                Path(binding["inputs"]["seed"]["registration_path"]).parent / "initial_pose_xyz.npy"
            )
            for p in [seed, *[p for p in calibration.iterdir() if p.is_file()]]:
                digest = run["inputs"][str(p)]["sha256"]
                checks.require(checks.sha(p) == digest, "calibration/seed changed")
                hashes[str(p)] = digest
            rig, prepared = load_rig(calibration), []
            for native in FRAMES:
                folder = replay / "diagnostics" / f"frame_{native:06d}"
                a = base.load_arrays(folder / "arrays.npz", hashes)
                new = base.load_arrays(recovery / f"frame_{native:06d}.npz", hashes)
                prepared.append(
                    (
                        native,
                        fit_inputs(
                            a, new, checks.read(folder / "metadata.json")["proposal_kind"], rig
                        ),
                        base.load_arrays(cpu / f"frame_{native:06d}.npz", hashes),
                        len(a["proposal_pose"]),
                    )
                )
            wait_start = time.monotonic()
            while len(active_progress(guard)) < 5:
                checks.require(
                    time.monotonic() - wait_start < 900,
                    "reference has insufficient active progress",
                )
                time.sleep(2)
            before = active_progress(guard)
            telemetry.append(gpu_sample())
            checks.require(
                telemetry[-1]["total_MiB"] - telemetry[-1]["used_MiB"] >= 6144,
                "insufficient initial GPU headroom",
            )
            torch.cuda.set_per_process_memory_fraction(0.25, 0)
            torch.cuda.reset_peak_memory_stats()
            torch.set_num_threads(2)
            fitter = DirectionalDeviceFitter(
                rig,
                np.load(seed, allow_pickle=False),
                base.TrackerConfig(**registration["original_context"]["tracker_config"]),
            )
            record.param("allocator_fraction_cap", 0.25)
            record.param("fixed_four_frames", FRAMES)
            record.param("cycles", 8)
            record.param("comparison_atol", base.ATOL)
            record.param("natural_idle_window_claimed", False)
            start = time.perf_counter()
            for cycle in range(8):
                for native, inputs, saved, old_count in prepared:
                    active_progress(guard)
                    fitted = fitted_arrays(fitter, inputs, old_count)
                    comparisons = verify_control(fitted, saved)
                    records.append(dict(cycle=cycle, frame_idx=native, comparisons=comparisons))
                telemetry.append(gpu_sample())
                checks.require(
                    telemetry[-1]["total_MiB"] - telemetry[-1]["used_MiB"] >= 2048,
                    "remaining GPU headroom fell below 2GiB",
                )
                print(
                    json.dumps(
                        dict(
                            cycle=cycle + 1,
                            compared_fits=len(records),
                            GPU_used_MiB=telemetry[-1]["used_MiB"],
                        )
                    ),
                    flush=True,
                )
            seconds = time.perf_counter() - start
            after = active_progress(guard)
            baseline_rate = rate(before[-5], before[-1])
            during_rate = rate(before[-1], after[-1])
            checks.require(
                after[-1]["phase_processed"] > before[-1]["phase_processed"],
                "reference did not advance during bounded pilot",
            )
            checks.require(
                during_rate is not None and during_rate >= 0.5 * baseline_rate,
                "observed reference throughput fell below half its preceding rate",
            )
            result = dict(
                schema="concurrent_directional_GPU_pilot_v1",
                status="ok",
                fits=len(records),
                seconds=seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                allocator_fraction_cap=0.25,
                reference_frames_advanced=after[-1]["phase_processed"]
                - before[-1]["phase_processed"],
                observed_reference_rate_before=baseline_rate,
                observed_reference_rate_during=during_rate,
                fixed_fit_numerical_and_decision_checks_passed=True,
                resource_pilot_passed=True,
                full_minute_stability_proven=False,
                causal_throughput_effect_proven=False,
                tracking_performed_by_pilot=False,
                production_process_signalled=False,
            )
            for name, value in (
                ("summary.json", result),
                ("comparisons.json", records),
                ("GPU_telemetry.json", telemetry),
                ("reference_progress.json", dict(before=before, after=after)),
            ):
                (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
        finally:
            guard.close()
            for name, value in (
                ("comparisons.json", records),
                ("GPU_telemetry.json", telemetry),
                ("reference_progress.json", dict(before=before, after=after)),
            ):
                (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
            for p, digest in hashes.items():
                checks.require(checks.sha(p) == digest, "bound input changed")
                record.input(p, p, sha256=digest)
            checks.require(code == base.local_code(), "source set changed")
            for p in output.rglob("*"):
                if p.is_file() and p != output / "pilot.provenance.json":
                    record.output(str(p.relative_to(output)), p, sha256=checks.sha(p))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "replay", "recovery", "cpu", "qualified", "supervisor-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--child-pid", type=int, required=True)
    main(p.parse_args())

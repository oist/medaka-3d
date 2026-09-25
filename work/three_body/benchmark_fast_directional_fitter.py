"""Paired fixed-301 CUDA comparison of frozen and fast directional fitters.

Both arms reuse independent four-entry graph caches and the frozen 77-array
fitted_frame contract. No tracking, raw decoding, flow generation or adoption.
Concurrent workloads and cache captures limit causal timing interpretations.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import fit_tail_packets as frozen
import numpy as np
import torch
from directional_device_fitter import DirectionalDeviceFitter
from fast_directional_device_fitter import FastDirectionalDeviceFitter
from qualify_concurrent_directional_gpu import gpu_sample
from replay_gpu_window import ReplayGPUWindow

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance

base, checks, REPO = frozen.base, frozen.checks, frozen.REPO
WORK = REPO / "work/three_body"
STAGE = "fast_directional_fitter_paired_301_v1"
FRAMES, CPU_FRAMES = frozen.FRAMES, frozen.CPU_FRAMES
ARMS = ("original", "fast")
REFERENCE_CHILD_PID = 33360
MAX_RESERVED_BYTES = 4 * 1024**3
PACKETS = WORK / "tail_local_fit_packets_301_v1"
GOLDEN = WORK / "tail_local_cuda301_v1"
CPU_REFERENCE = WORK / "tail_local_cpu4_v1"
PACKET_SHA = "6e0e3a8d7d006a9f3afc673af829f54de77567d1346a7cb613179114f1769cdc"
GOLDEN_SHA = "efc3a4ef51c56cbea058fa67a9091c804a1778f004bc6d66c2b926a3c57b4a22"
CPU_SHA = "88275728678193963972e61eac581871e33f837d90e43370f8e57a97623a5ec8"
SOURCE_PINS = {
    "fit_tail_packets.py": "88308becadd04f1f824d538d93c3047e8add71f0136be62699e2b09360c113bc",
    "directional_device_fitter.py": (
        "3228e3815966a5695fc8647a9b8310b28dea9368e177fd4941d20b35024c45aa"
    ),
    "fast_directional_device_fitter.py": (
        "5804003d537fe5ad144927c9629bf12927fc0ce9dc460615a78125cfe16c4abd"
    ),
}


def signature(arrays):
    return {
        key: (value.dtype.str, value.shape, hashlib.sha256(value.tobytes()).hexdigest())
        for key, value in arrays.items()
    }


def pair_order(index):
    checks.require(type(index) is int and 0 <= index < len(FRAMES), "fixed frame index required")
    return ARMS if index % 2 == 0 else ARMS[::-1]


def validate_reference_pid(value):
    checks.require(
        type(value) is int and value == REFERENCE_CHILD_PID,
        "explicit registered startup child PID33360 required",
    )


def _check_owned_alive(guard):
    checks.require(
        not guard.child.signalled() and not guard.owner.signalled(),
        "owned reference must be live at startup",
    )


def attach_startup_reference(guard):
    """Accept a preparing owned child; never wait for exit or block later phases."""
    guard._attach()
    _check_owned_alive(guard)


def check_headroom(sample, *, startup=False):
    free = sample["total_MiB"] - sample["used_MiB"]
    required = 6144 if startup else 2048
    checks.require(np.isfinite(free) and free >= required, f"less than {required}MiB GPU headroom")


def check_memory(memory):
    checks.require(
        0 <= memory["allocated_bytes"] <= memory["reserved_bytes"] <= MAX_RESERVED_BYTES
        and 0
        <= memory["peak_allocated_bytes"]
        <= memory["peak_reserved_bytes"]
        <= MAX_RESERVED_BYTES,
        "two fitter cache pools exceeded the fixed 4GiB process budget",
    )


def memory_sample():
    sample = dict(
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    return sample


def validate_gold_summary(summary, device):
    expected = FRAMES if device == "cuda" else CPU_FRAMES
    checks.require(
        summary["schema"] == "tail_same_objective_local_fit_v1"
        and summary["status"] == "ok"
        and summary["device"] == device
        and tuple(summary["frames"]) == expected
        and summary["control_all19_records_reproduced"] is True
        and summary["independent_frames"] is True
        and summary["allow_bounded"] is False
        and summary["target_node"] == 2
        and summary["temporal_tracking_performed"] is False
        and summary["original_pool_preserved"] is True,
        "successful fixed golden fit contract required",
    )
    if device == "cuda":
        checks.require(
            tuple(summary["CPU_CUDA_all_candidate_frames"]) == CPU_FRAMES,
            "golden CUDA must include the registered CPU4 parity",
        )


def paired_frame(
    index, fitters, packet, golden, cpu=None, on_result=None, after_fit=None, on_attempt=None
):
    """Fit both arms in registered order; compare every saved array before publishing."""
    order = pair_order(index)
    checks.require(int(packet["frame_idx"]) == FRAMES[index], "packet native frame differs")
    checks.require(
        set(fitters) == set(ARMS) and len(golden) == 77, "two arms and all77 golden fields"
    )
    if cpu is not None:
        checks.require(FRAMES[index] in CPU_FRAMES and len(cpu) == 77, "fixed CPU4/all77 reference")
    before = signature(packet)
    results, seconds, comparisons = {}, {}, {}
    started = time.perf_counter()
    for arm in order:
        if on_attempt is not None:
            on_attempt(arm)
        arrays, elapsed = frozen.fitted_frame(fitters[arm], packet)
        if after_fit is not None:
            after_fit(arm)
        checks.require(np.isfinite(elapsed) and elapsed >= 0, "invalid frozen fit timing")
        checks.require(len(arrays) == 77, "fitted result must retain all77 arrays")
        checks.require(signature(packet) == before, "fitter mutated the shared packet")
        detail = {"CUDA_golden": frozen.compare_fit_arrays(arrays, golden)}
        if cpu is not None:
            detail["CPU_reference"] = frozen.compare_fit_arrays(arrays, cpu)
        results[arm], seconds[arm], comparisons[arm] = arrays, float(elapsed), detail
        if on_result is not None:
            on_result(arm, arrays, detail, float(elapsed))
    comparisons["paired"] = frozen.compare_fit_arrays(results["fast"], results["original"])
    checks.require(signature(packet) == before, "paired fit changed packet")
    return dict(
        frame_idx=FRAMES[index],
        order=list(order),
        fit_seconds=seconds,
        pair_wall_seconds=time.perf_counter() - started,
        comparisons=comparisons,
    )


def timing_summary(rows):
    checks.require(
        [r["frame_idx"] for r in rows] == list(FRAMES)
        and all(r["order"] == list(pair_order(i)) for i, r in enumerate(rows)),
        "complete fixed301 paired timing order required",
    )
    values = np.array([[r["fit_seconds"][arm] for arm in ARMS] for r in rows])
    checks.require(np.isfinite(values).all() and (values >= 0).all(), "invalid paired timing")
    result = {}
    for name, selected in (
        ("initial_cold_frame", values[:1]),
        ("all301", values),
        ("remaining300", values[1:]),
    ):
        total = selected.sum(0)
        delta = selected[:, 1] - selected[:, 0]
        ratio_defined = selected[:, 1] > 0
        result[name] = dict(
            frames=len(selected),
            seconds_by_arm=dict(zip(ARMS, map(float, total), strict=True)),
            mean_seconds_by_arm=dict(zip(ARMS, map(float, selected.mean(0)), strict=True)),
            median_seconds_by_arm=dict(
                zip(ARMS, map(float, np.median(selected, axis=0)), strict=True)
            ),
            original_over_fast_ratio=float(total[0] / total[1]) if total[1] > 0 else None,
            paired_median_original_over_fast_ratio=(
                float(np.median(selected[ratio_defined, 0] / selected[ratio_defined, 1]))
                if ratio_defined.any()
                else None
            ),
            paired_ratio_defined_frames=int(ratio_defined.sum()),
            fast_faster_frames=int((delta < 0).sum()),
            fast_slower_frames=int((delta > 0).sum()),
            equal_frames=int((delta == 0).sum()),
            paired_fast_minus_original_mean_seconds=float(np.mean(delta)),
        )
    result["limitations"] = [
        "Concurrent reference and D2 load varies; alternation does not make this "
        "a controlled causal speed test.",
        "First frame includes each fitter's cold cache; later frames can capture "
        "previously unseen pool sizes.",
        "fitted_frame timers include fit and its CUDA completion, excluding support "
        "reconstruction, comparison, I/O and telemetry.",
        "The two persistent cache pools coexist in one capped process; "
        "no cache/order changes are made to fit the budget.",
    ]
    return result


def verified(root, filename, stage, expected, hashes):
    path = root / filename
    checks.require(checks.sha(path) == expected, "pinned successful seal differs")
    seal = checks.verify(root, filename, stage, hashes)
    checks.require(hashes[str(path)] == expected, "seal changed during verification")
    return seal


def load_evidence(hashes, code):
    producer = verified(
        PACKETS, "preparation.provenance.json", frozen.PACKET_STAGE, PACKET_SHA, hashes
    )
    golden = verified(GOLDEN, "fit.provenance.json", frozen.STAGE, GOLDEN_SHA, hashes)
    cpu = verified(CPU_REFERENCE, "fit.provenance.json", frozen.STAGE, CPU_SHA, hashes)
    for seal in (golden, cpu):
        frozen.bind_input(seal, PACKETS / "preparation.provenance.json", hashes)
        frozen.bind_input(seal, Path(frozen.__file__), hashes)
    frozen.bind_input(golden, CPU_REFERENCE / "fit.provenance.json", hashes)
    for seal in (producer, golden, cpu):
        for name in ("directional_device_fitter.py", "directional_chest_fitter.py"):
            frozen.bind_input(seal, WORK / name, hashes)
        for path in set(code) & set(seal["inputs"]):
            frozen.bind_input(seal, path, hashes)
    metadata = checks.read(frozen.member(PACKETS, "metadata.json", hashes))
    frozen.validate_tail_metadata(metadata)
    packet_summary = checks.read(frozen.member(PACKETS, "summary.json", hashes))
    checks.require(
        packet_summary["status"] == "ok"
        and packet_summary["packets"] == 301
        and packet_summary["originals"] == 4273
        and packet_summary["added"] == 676,
        "fixed packet counts differ",
    )
    for root, device in ((GOLDEN, "cuda"), (CPU_REFERENCE, "cpu")):
        validate_gold_summary(checks.read(frozen.member(root, "summary.json", hashes)), device)
    cfg = base.TrackerConfig(**metadata["tracker_config"])
    checks.require(
        cfg.device == "cuda" and cfg.opt_steps == 14 and base.ATOL == 1e-10,
        "CUDA/14Adam/ATOL contract",
    )
    calibration = Path(metadata["calibration_directory"]).resolve(strict=True)
    files = [p for p in calibration.iterdir() if p.is_file()]
    checks.require(len(files) == 4, "four registered calibration files required")
    for path in files:
        for seal in (producer, golden, cpu):
            frozen.bind_input(seal, path, hashes)
    rig = load_rig(calibration)
    prepared, expected, cpu_expected = {}, {}, {}
    for native in FRAMES:
        prepared[native] = base.load_arrays(
            frozen.member(PACKETS, f"f{native}_packet.npz", hashes), hashes
        )
        expected[native] = base.load_arrays(
            frozen.member(GOLDEN, f"f{native}_fit.npz", hashes), hashes
        )
        checks.require(
            int(prepared[native]["frame_idx"]) == native and len(expected[native]) == 77,
            "native packet/all77 golden differs",
        )
        if native in CPU_FRAMES:
            cpu_expected[native] = base.load_arrays(
                frozen.member(CPU_REFERENCE, f"f{native}_fit.npz", hashes), hashes
            )
            checks.require(len(cpu_expected[native]) == 77, "all77 CPU fields required")
    initial = prepared[FRAMES[0]]["initial_pose_xyz"]
    checks.require(
        all(checks.equal_bytes(p["initial_pose_xyz"], initial) for p in prepared.values()),
        "packet seed changed",
    )
    return rig, cfg, initial, prepared, expected, cpu_expected


def main(args):
    validate_reference_pid(args.child_pid)
    output = args.output.resolve()
    checks.require(output.is_relative_to(WORK) and not output.exists(), "new work output required")
    code = base.local_code()
    checks.require(str(Path(__file__).resolve()) in code, "driver source not registered")
    for name, expected in SOURCE_PINS.items():
        checks.require(
            code.get(str(WORK / name)) == expected, "pinned fitter/helper source differs: " + name
        )
    hashes, outputs = dict(code), {}
    output.mkdir(parents=True)
    rows, arm_rows, telemetry, memories, attempts = [], [], [], [], []
    guard = None
    summary = dict(
        schema=STAGE,
        status="incomplete",
        passed=False,
        frames=list(FRAMES),
        GPU_used=False,
        GPU_initialization_attempted=False,
        adopted=False,
    )
    with provenance(STAGE, output, name="benchmark") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            outputs[str(path)] = checks.sha(path)
            record.output(name, path, sha256=outputs[str(path)])

        try:
            rig, cfg, initial, packets, golden, cpu = load_evidence(hashes, code)
            for path, digest in code.items():
                destination = output / "source_snapshot" / Path(path).relative_to(REPO)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(Path(path).read_bytes())
                checks.require(checks.sha(destination) == digest, "source snapshot changed")
                emit(str(destination.relative_to(output)))
            observed = output / "registered_reference"
            observed.mkdir()
            guard = ReplayGPUWindow(args.supervisor_root, args.child_pid, observed)
            attach_startup_reference(guard)
            record.param("reference_ownership_verified_at_start", True)
            record.param("reference_natural_phase_transitions_allowed", True)
            sample = gpu_sample()
            check_headroom(sample, startup=True)
            telemetry.append(dict(stage="startup", **sample))
            summary["GPU_initialization_attempted"] = True
            torch.cuda.set_per_process_memory_fraction(0.25, 0)
            summary["GPU_used"] = True
            torch.cuda.reset_peak_memory_stats()
            torch.set_num_threads(2)
            fitters = dict(
                original=DirectionalDeviceFitter(rig, initial.copy(), cfg),
                fast=FastDirectionalDeviceFitter(rig, initial.copy(), cfg),
            )
            for fitter in fitters.values():
                checks.require(
                    fitter.device.type == "cuda"
                    and fitter.dtype == torch.float64
                    and fitter.directional is True,
                    "same CUDA/f64/directional fitter required",
                )
                checks.require(
                    all(
                        checks.equal_bytes(fitter.lengths, p["initial_lengths"])
                        for p in packets.values()
                    ),
                    "packet seed lengths differ",
                )
            record.param("tracker_config", dataclasses.asdict(cfg))
            record.param("comparison_atol", base.ATOL)
            record.param("rtol", 0)
            record.param("allocator_fraction_cap", 0.25)
            record.param("maximum_process_reserved_bytes", MAX_RESERVED_BYTES)
            record.param(
                "cache_policy", "one unchanged four-entry cache per arm; both pools coexist"
            )
            record.param("comparison_order", "fixed index even original/fast; odd fast/original")
            for index, native in enumerate(FRAMES):
                sample = gpu_sample()
                check_headroom(sample)
                telemetry.append(dict(frame_idx=native, stage="before_pair", **sample))

                def after_fit(arm, *, native=native):
                    attempts[-1]["fitted_frame_returned"] = True
                    memory = memory_sample()
                    memories.append(dict(frame_idx=native, arm=arm, **memory))
                    check_memory(memory)

                def on_attempt(arm, *, native=native):
                    attempts.append(dict(frame_idx=native, arm=arm, fitted_frame_returned=False))

                def on_result(arm, arrays, detail, seconds, *, native=native):
                    filename = f"f{native}_{arm}_fit.npz"
                    np.savez_compressed(output / filename, **arrays)
                    emit(filename)
                    arm_rows.append(
                        dict(frame_idx=native, arm=arm, fit_seconds=seconds, all77_pass=True)
                    )

                row = paired_frame(
                    index,
                    fitters,
                    packets[native],
                    golden[native],
                    cpu.get(native),
                    on_result,
                    after_fit,
                    on_attempt,
                )
                emit(f"f{native}_parity.json", row["comparisons"])
                rows.append({k: v for k, v in row.items() if k != "comparisons"})
                sample = gpu_sample()
                check_headroom(sample)
                telemetry.append(dict(frame_idx=native, stage="after_pair", **sample))
                emit("frames.json", rows)
                emit("completed_arms.json", arm_rows)
                print(json.dumps(rows[-1]), flush=True)
            checks.require(
                all(f.initialize_count == 0 for f in fitters.values()),
                "unexpected temporal initialization",
            )
            caches = {
                arm: len(f.__dict__.get("_fit_graph_cache", {})) for arm, f in fitters.items()
            }
            checks.require(
                all(count <= 4 for count in caches.values()), "inherited cache capacity changed"
            )
            summary.update(
                status="ok",
                passed=True,
                fitted_calls=len(arm_rows),
                all77_arrays_per_arm_pass=True,
                CPU_reference_frames=list(CPU_FRAMES),
                paired_all77_pass=True,
                timing=timing_summary(rows),
                memory=memory_sample(),
                cache_entries=caches,
                initialize_calls=0,
                restore_calls=0,
                advance_calls=0,
                raw_decodes=0,
                flow_generations=0,
                new_tracking=False,
                original_objective_unchanged=True,
                numerical_equivalence_scope=(
                    "saved whole-pool outputs, discrete exact / float atol1e-10 rtol0"
                ),
                timing_is_concurrent_observation=True,
            )
        except BaseException as error:
            summary.update(
                status="failed",
                error=repr(error),
                completed_pairs=len(rows),
                completed_arms=len(arm_rows),
            )
            raise
        finally:
            summary.update(
                fit_calls_attempted=len(attempts),
                fit_calls_returned=sum(a["fitted_frame_returned"] for a in attempts),
                validated_fit_calls=len(arm_rows),
            )
            if guard is not None:
                guard.close()
                for path in (output / "registered_reference").iterdir():
                    if path.is_file():
                        emit(str(path.relative_to(output)))
            for name, value in (
                ("summary.json", summary),
                ("frames.json", rows),
                ("completed_arms.json", arm_rows),
                ("GPU_telemetry.json", telemetry),
                ("memory.json", memories),
                ("fit_attempts.json", attempts),
            ):
                emit(name, value)
            checks.require(code == base.local_code(), "loaded source set or bytes changed")
            for path, digest in hashes.items():
                checks.require(checks.sha(path) == digest, "input/source changed")
                record.input(path, path, sha256=digest)
            for path, digest in outputs.items():
                checks.require(checks.sha(path) == digest, "output changed after emission")
            record.param("all_input_source_output_SHA_unchanged", True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supervisor-root", type=Path, required=True)
    parser.add_argument("--child-pid", type=int, required=True)
    main(parser.parse_args())

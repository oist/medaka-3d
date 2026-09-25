"""Paired Fast/cache4 versus Full/cache1 over the frozen 301 tail-fit packets.

FP64, 14 Adam updates, 15 scores and all77 CUDA output bytes are fixed. Each
packet is independently fitted twice in alternating order; no temporal state,
flow, raw decoding, packet preparation or old qualification is re-executed.
Both persistent cache pools coexist under 25%; allocator failure stops this run.
Full trace CPU copies remain inside its fit timer; disk writes are outside.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import benchmark_fast_directional_fitter_v2 as baseline
import cv2
import fit_tail_packets as frozen
import full_fit_graph_fitter as full
import numpy as np
import qualify_full_fit_graph4 as qualified
import scipy
import torch
from fast_directional_device_fitter import FastDirectionalDeviceFitter

proof, require = qualified.proof, qualified.require
REPO, WORK = qualified.REPO, qualified.WORK
STAGE = "full_fit_graph_paired_301_v1"
FRAMES, CPU_FRAMES = frozen.FRAMES, frozen.CPU_FRAMES
ARMS = ("fast", "full")
ALLOCATOR_FRACTION = 0.25
MAX_RESERVED_BYTES = 4 * 1024**3
STARTUP_FREE_MIB, ARM_FREE_MIB = 6144, 2048
QUALIFICATION = WORK / "full_fit_graph_four_packet_qualification_v1"
QUALIFICATION_SHA = "d9b2f40b13f62b75918fc34dc4c1a0ec26a45a6da8c2dc657f8091ab9728cf50"
SOURCE_PINS = {
    **qualified.SOURCE_PINS,
    "work/three_body/qualify_full_fit_graph4.py": (
        "40ab249079c11dc10d5e9e3d86770937a60900302df72f3f37dff53380986914"
    ),
}
signature = baseline.signature
gpu_sample = baseline.gpu_sample
memory_sample = baseline.memory_sample


def pair_order(index):
    require(type(index) is int and 0 <= index < len(FRAMES), "fixed frame index required")
    return ARMS if index % 2 == 0 else ARMS[::-1]


def check_headroom(sample, *, startup=False):
    free = sample["free_MiB"]
    required = STARTUP_FREE_MIB if startup else ARM_FREE_MIB
    require(np.isfinite(free) and free >= required, f"less than {required}MiB GPU headroom")


def check_memory(memory):
    require(
        0 <= memory["allocated_bytes"] <= memory["reserved_bytes"] <= MAX_RESERVED_BYTES
        and 0
        <= memory["peak_allocated_bytes"]
        <= memory["peak_reserved_bytes"]
        <= MAX_RESERVED_BYTES,
        "combined unchanged caches exceeded the fixed 4GiB process bound",
    )


def trace_layout(batch):
    require(type(batch) is int and 6 <= batch <= 30, "fixed candidate budget 6..30 required")
    return {
        "pose_before": ((15, batch, 3, 3, 3), np.dtype("float64")),
        "score": ((15, batch), np.dtype("float64")),
        "parts": ((15, batch, 5), np.dtype("float64")),
        "base_gradient": ((15, batch, 3, 3, 3), np.dtype("float64")),
        "update_gradient": ((14, batch, 3, 3, 3), np.dtype("float64")),
        "moment": ((14, batch, 3, 3, 3), np.dtype("float64")),
        "variance": ((14, batch, 3, 3, 3), np.dtype("float64")),
        "pose_after": ((14, batch, 3, 3, 3), np.dtype("float64")),
        "best_iteration": ((batch,), np.dtype("int64")),
        "best_score": ((batch,), np.dtype("float64")),
    }


def trace_bytes(batch):
    return sum(
        int(np.prod(shape)) * dtype.itemsize for shape, dtype in trace_layout(batch).values()
    )


def validate_trace(trace, batch):
    layout = trace_layout(batch)
    require(set(trace) == set(layout), "complete full15/14 trace fields required")
    for name, (shape, dtype) in layout.items():
        value = trace[name]
        require(value.shape == shape and value.dtype == dtype, "trace shape/dtype: " + name)
        require(np.isfinite(value).all(), "nonfinite returned trace: " + name)
    require(
        ((trace["best_iteration"] >= 0) & (trace["best_iteration"] <= 14)).all(),
        "invalid best iteration",
    )
    require(sum(a.nbytes for a in trace.values()) == trace_bytes(batch), "trace capacity mismatch")
    return trace_bytes(batch)


def paired_frame(
    index, fitters, packet, golden, cpu=None, on_result=None, after_fit=None, on_attempt=None
):
    """Old pairing contract with new arms, save-before-check and strict CUDA bytes.

    on_result(arm, arrays, detail, seconds) receives detail=None before comparison
    so even a failed numerical comparison retains its returned arrays. after_fit
    runs after saving and before the comparisons; it records resource/capture data.
    """
    order = pair_order(index)
    require(int(packet["frame_idx"]) == FRAMES[index], "packet native differs")
    require(set(fitters) == set(ARMS) and len(golden) == 77, "two arms and all77 golden required")
    if cpu is not None:
        require(FRAMES[index] in CPU_FRAMES and len(cpu) == 77, "fixed CPU4/all77 required")
    before = signature(packet)
    results, seconds, comparisons = {}, {}, {}
    start = time.perf_counter()
    for arm in order:
        if on_attempt is not None:
            on_attempt(arm)
        arrays, elapsed = frozen.fitted_frame(fitters[arm], packet)
        if on_result is not None:
            on_result(arm, arrays, None, float(elapsed))
        if after_fit is not None:
            after_fit(arm)
        require(np.isfinite(elapsed) and elapsed >= 0, "invalid frozen fit time")
        require(len(arrays) == 77, "fitted result must retain all77 arrays")
        require(signature(packet) == before, "fitter mutated shared packet")
        detail = {"CUDA_golden": frozen.compare_fit_arrays(arrays, golden)}
        if cpu is not None:
            detail["CPU_reference"] = frozen.compare_fit_arrays(arrays, cpu)
        require(
            all(r["byte_exact"] for r in detail["CUDA_golden"].values()),
            "CUDA golden differs at byte level: " + arm,
        )
        results[arm], seconds[arm], comparisons[arm] = arrays, float(elapsed), detail
    comparisons["paired"] = frozen.compare_fit_arrays(results["full"], results["fast"])
    require(
        all(r["byte_exact"] for r in comparisons["paired"].values()), "paired CUDA bytes differ"
    )
    require(signature(packet) == before, "paired fit changed packet")
    return dict(
        frame_idx=FRAMES[index],
        order=list(order),
        fit_seconds=seconds,
        pair_wall_seconds=time.perf_counter() - start,
        comparisons=comparisons,
    )


def timing_summary(rows):
    require(
        [r["frame_idx"] for r in rows] == list(FRAMES)
        and all(r["order"] == list(pair_order(i)) for i, r in enumerate(rows)),
        "complete fixed301 alternating order required",
    )
    values = np.array([[r["fit_seconds"][arm] for arm in ARMS] for r in rows])
    require(np.isfinite(values).all() and (values >= 0).all(), "invalid paired timing")
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
            fast_over_full_ratio=float(total[0] / total[1]) if total[1] > 0 else None,
            reduction_fraction_vs_fast=float(1 - total[1] / total[0]) if total[0] > 0 else None,
            paired_median_fast_over_full_ratio=(
                float(np.median(selected[ratio_defined, 0] / selected[ratio_defined, 1]))
                if ratio_defined.any()
                else None
            ),
            paired_ratio_defined_frames=int(ratio_defined.sum()),
            full_faster_frames=int((delta < 0).sum()),
            full_slower_frames=int((delta > 0).sum()),
            equal_frames=int((delta == 0).sum()),
            paired_full_minus_fast_mean_seconds=float(delta.mean()),
        )
    result["limitations"] = [
        "Concurrent reference/D2 load varies; this is not an isolated causal speed benchmark.",
        "Remaining300 can contain new captures; single Full cache differs from Fast cache4.",
        "Full fit includes all15/14 trace CPU transfers; Fast produces no such trace. "
        "No subtraction is applied.",
        "Fit times include capture/warmup on shape changes and CUDA completion; "
        "disk I/O and support reconstruction are excluded.",
        "Complete-fitter comparison with unequal trace overhead is conservative for Full; "
        "no CUDA-kernel-only speed claim.",
        "First-frame startup burden is unequal; its ratio is not evidence of steady-state speedup.",
    ]
    return result


def qualification_gate(evidence):
    record = qualified.sealed_header(
        QUALIFICATION, "qualification.provenance.json", QUALIFICATION_SHA, qualified.STAGE, evidence
    )
    path = QUALIFICATION / "summary.json"
    row = record["outputs"]["summary.json"]
    require(
        Path(row["path"]) == path and path.stat().st_size == row["size_bytes"],
        "qualified summary path/size changed",
    )
    evidence.add(path, row["sha256"])
    summary = proof.read_json(path)
    require(
        summary["passed"] is True
        and summary["all77_CUDA_byte_exact"] is True
        and summary["frames"] == list(CPU_FRAMES)
        and summary["completed_fit_calls"] == 4
        and summary["initialize_calls"] == summary["advance_calls"] == summary["raw_decode"] == 0,
        "complete four-packet full-fit qualification required",
    )
    for relative, digest in qualified.SOURCE_PINS.items():
        path = REPO / relative
        require(record["inputs"].get(str(path)) == digest, "qualified source edge differs")
        evidence.add(path, digest)
    require(
        record["inputs"].get(str(WORK / "qualify_full_fit_graph4.py"))
        == SOURCE_PINS["work/three_body/qualify_full_fit_graph4.py"],
        "qualification caller differs",
    )
    return summary


def load_inputs(evidence, code):
    """Reuse frozen full301 input verification and all77/CPU4 loading unchanged."""
    hashes = dict(evidence.paths)
    rig, cfg, initial, packets, golden, cpu = baseline.load_evidence(hashes, code)
    for path, digest in hashes.items():
        require(
            path not in evidence.paths or evidence.paths[path] == digest, "input digest conflict"
        )
        evidence.paths[path] = digest
    metadata = proof.read_json(baseline.PACKETS / "metadata.json")
    require(
        tuple(packets) == tuple(golden) == FRAMES and tuple(cpu) == CPU_FRAMES,
        "fixed301 packets/golden and fixedCPU4 membership required",
    )
    counts = [len(packets[n]["proposal_pose_xyz"]) for n in FRAMES]
    require(sum(counts) == 4949, "registered all301 candidate count changed")
    capacity = sum(trace_bytes(count) for count in counts)
    return rig, cfg, initial, packets, golden, cpu, metadata, capacity


def run(args):
    require(args.execute is True and args.device == "cuda", "explicit CUDA run required")
    destination = Path(args.output).resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(), "new work output required"
    )
    output, evidence, saved = proof.Output(destination), proof.Evidence(), set()
    started, begin = proof.utc(), time.perf_counter()
    rows, arms, telemetry, memories, attempts = [], [], [], [], []
    fitters = None
    summary = dict(
        schema=STAGE,
        status="incomplete",
        passed=False,
        frames=list(FRAMES),
        GPU_used=False,
        GPU_initialization_attempted=False,
        adopted=False,
    )
    try:
        evidence.add(Path(__file__).resolve())
        for relative, digest in SOURCE_PINS.items():
            evidence.add(REPO / relative, digest)
        qualified.snapshot(evidence, output, saved, 0)
        qualification_gate(evidence)
        code = {
            path: digest
            for path, digest in evidence.paths.items()
            if path.endswith(".py") and ".venv" not in Path(path).parts
        }
        rig, cfg, initial, packets, golden, cpu, metadata, capacity = load_inputs(evidence, code)
        require(not torch.cuda.is_initialized(), "no CUDA before input qualification")
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                frames=list(FRAMES),
                arms=list(ARMS),
                fitted_calls_planned=602,
                order="index even fast/full; odd full/fast",
                allocator_fraction=ALLOCATOR_FRACTION,
                max_reserved_bytes=MAX_RESERVED_BYTES,
                startup_free_MiB=STARTUP_FREE_MIB,
                before_each_arm_free_MiB=ARM_FREE_MIB,
                cache_entries=dict(fast=4, full=1),
                tracker_config=dataclasses.asdict(cfg),
                qualification_sha256=QUALIFICATION_SHA,
                source_pins=SOURCE_PINS,
                dtype="float64",
                Adam_updates=14,
                objective_evaluations=15,
                all301_full_traces_saved=True,
                trace_raw_array_bytes=capacity,
                trace_per_candidate_bytes=trace_bytes(6) // 6,
                trace_absolute_budget_bytes=301 * trace_bytes(30),
                trace_capacity_note=(
                    "Raw dtype bytes; compressed file sizes/NPZ headers are measured after save."
                ),
                old_Fast_extra_trace=False,
                full_trace_CPU_transfer_in_fitter_timer=True,
                original_packet_preparation_reexecuted=False,
                old306_reexecuted=False,
                fast_warmup_per_capture_objectives=3,
                fast_capture_objectives=1,
                full_warmup_per_capture_full_fits=3,
                full_capture_full_fits=1,
                capture_warmups_excluded_from_602_result_fit_calls=True,
            ),
        )
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(metadata["opencv_threads"])
        torch.manual_seed(0)
        versions = dict(
            numpy=np.__version__,
            scipy=scipy.__version__,
            opencv=cv2.__version__,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
        )
        require(versions == metadata["runtime"], "frozen package versions changed")
        sample = gpu_sample()
        telemetry.append(dict(stage="startup", **sample))
        check_headroom(sample, startup=True)
        summary["GPU_initialization_attempted"] = True
        torch.cuda.set_per_process_memory_fraction(ALLOCATOR_FRACTION, 0)
        torch.cuda.reset_peak_memory_stats(0)
        summary["GPU_used"] = True
        output.write(
            "environment.json",
            dict(
                **versions,
                GPU=torch.cuda.get_device_name(0),
                torch_threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads(),
                opencv_threads=cv2.getNumThreads(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            ),
        )
        fitters = dict(
            fast=FastDirectionalDeviceFitter(rig, initial.copy(), cfg),
            full=full.FullFitDirectionalDeviceFitter(rig, initial.copy(), cfg),
        )
        for fitter in fitters.values():
            require(
                fitter.device.type == "cuda"
                and fitter.dtype == torch.float64
                and fitter.directional is True,
                "same CUDA/f64/directional required",
            )
            require(
                all(
                    frozen.checks.equal_bytes(fitter.lengths, p["initial_lengths"])
                    for p in packets.values()
                ),
                "seed lengths differ",
            )
        for index, native in enumerate(FRAMES):
            old_key = (len(packets[native]["proposal_pose_xyz"]), 1024, cfg)
            fast_capture = old_key not in fitters["fast"].__dict__.get("_fit_graph_cache", {})
            capture_before = fitters["full"].__dict__.get("_full_fit_capture_count", 0)

            def on_attempt(arm, *, native=native):
                sample = gpu_sample()
                telemetry.append(dict(frame_idx=native, arm=arm, stage="before_fit", **sample))
                check_headroom(sample)
                attempts.append(dict(frame_idx=native, arm=arm, fitted_frame_returned=False))

            def on_result(arm, arrays, detail, seconds, *, native=native):
                attempts[-1]["fitted_frame_returned"] = True
                output.write(f"f{native}_{arm}_fit.npz", arrays)
                if arm == "full":
                    trace = fitters[arm].last_full_fit_trace
                    output.write(f"f{native}_full_iterations.npz", trace)
                    validate_trace(trace, len(packets[native]["proposal_pose_xyz"]))
                arms.append(
                    dict(
                        frame_idx=native,
                        arm=arm,
                        fit_seconds=seconds,
                        returned_saved=True,
                        validated=False,
                    )
                )

            def after_fit(arm, *, native=native):
                memory = memory_sample()
                memories.append(dict(frame_idx=native, arm=arm, **memory))
                check_memory(memory)
                require(
                    len(fitters["fast"].__dict__.get("_fit_graph_cache", {})) <= 4,
                    "Fast cache capacity changed",
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
            output.write(f"f{native}_parity.json", row.pop("comparisons"))
            for arm in arms[-2:]:
                arm["validated"] = True
            row.update(
                fast_graph_captured=fast_capture,
                full_graph_captures=fitters["full"]._full_fit_capture_count - capture_before,
                full_trace_array_bytes=trace_bytes(len(packets[native]["proposal_pose_xyz"])),
            )
            rows.append(row)
            output.write(f"f{native}_timing.json", row)
            print(json.dumps(row), flush=True)
        require(
            all(f.initialize_count == 0 for f in fitters.values()), "no temporal initialization"
        )
        require(len(rows) == 301 and len(arms) == len(attempts) == 602, "full301/602 counts")
        fast_captures = sum(r["fast_graph_captured"] for r in rows)
        full_captures = sum(r["full_graph_captures"] for r in rows)
        require(
            full_captures == fitters["full"]._full_fit_capture_count, "capture accounting differs"
        )
        require(
            sum(r["full_trace_array_bytes"] for r in rows) == capacity, "trace raw capacity differs"
        )
        trace_files = [
            r for name, r in output.records.items() if name.endswith("_full_iterations.npz")
        ]
        require(len(trace_files) == 301, "all301 traces required")
        summary.update(
            status="ok",
            passed=True,
            fitted_calls=602,
            all77_CUDA_byte_exact=True,
            CPU_reference_frames=list(CPU_FRAMES),
            paired_all77_byte_exact=True,
            timing=timing_summary(rows),
            memory=memory_sample(),
            capture_counts=dict(fast=fast_captures, full=full_captures),
            fast_extra_objective_backward_evaluations=4 * fast_captures,
            full_extra_warmup_capture_fit_trajectories=4 * full_captures,
            full_extra_objective_backward_evaluations=4 * full_captures * 15,
            full_extra_Adam_updates=4 * full_captures * 14,
            result_fit_objective_backward_evaluations=602 * 15,
            result_fit_Adam_updates=602 * 14,
            trace_files=301,
            trace_raw_array_bytes=capacity,
            trace_encoded_bytes=sum(r["size_bytes"] for r in trace_files),
            initialize_calls=0,
            restore_calls=0,
            advance_calls=0,
            raw_decodes=0,
            flow_generations=0,
            new_tracking=False,
            biological_ID_truth=False,
            numerical_scope=(
                "frozen301 independent fits, all77 CUDA bytes; not 306 continuous tracking"
            ),
            elapsed_before_guards_seconds=time.perf_counter() - begin,
        )
    except BaseException as error:
        summary.update(
            status="failed", passed=False, error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        summary.update(
            completed_pairs=len(rows),
            completed_arms=len(arms),
            fit_calls_attempted=len(attempts),
            fit_calls_returned=sum(a["fitted_frame_returned"] for a in attempts),
            validated_fit_calls=sum(a["validated"] for a in arms),
            validated_fit_call_scope="arms in complete byte-exact pairs only",
        )
        # Save partial scientific and ownership facts even when a fit/compare fails.
        for name, value in (
            ("summary.json", summary),
            ("frames.json", rows),
            ("completed_arms.json", arms),
            ("fit_attempts.json", attempts),
            ("GPU_telemetry.json", telemetry),
            ("memory.json", memories),
        ):
            output.write(name, value)
        try:
            qualified.snapshot(evidence, output, saved, 1)
            evidence.final_guard()
            output.final_guard()
        except BaseException as error:
            seal = dict(
                stage=STAGE,
                status="failed",
                started_at=started,
                finished_at=proof.utc(),
                inputs=evidence.paths,
                outputs=output.records,
                error_type=type(error).__name__,
                error=str(error),
                guard_failure=True,
            )
            with (destination / "benchmark.provenance.json").open("xb") as stream:
                stream.write(proof.json_bytes(seal))
            raise
        seal = dict(
            stage=STAGE,
            status=summary["status"],
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            completed_pairs=len(rows),
            completed_arms=len(arms),
        )
        if summary["status"] != "ok":
            seal.update(error_type=summary.get("error_type"), error=summary.get("error"))
        with (destination / "benchmark.provenance.json").open("xb") as stream:
            stream.write(proof.json_bytes(seal))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--device", choices=["cuda"], required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    print(json.dumps(run(parser.parse_args())), flush=True)

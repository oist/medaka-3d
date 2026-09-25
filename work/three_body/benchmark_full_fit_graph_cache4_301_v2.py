"""Paired Fast/cache4 versus Full/cache4 over the frozen 301 tail-fit packets.

FP64, 14 Adam updates, 15 scores and all77 CUDA output bytes are fixed. Each
packet is independently fitted twice in alternating order; no temporal state,
flow, raw decoding, packet preparation or old qualification is re-executed.
Both four-entry cache pools coexist under 37.5%; allocator failure stops this run.
This is a resource-only revision after the failed25% paired-cache4 run.
All301 pairs are newly measured; no numerical results are adopted from the failed run.
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
import full_fit_graph_cache4 as full
import numpy as np
import qualify_full_fit_graph4 as qualified
import qualify_full_fit_graph_cache4 as cache_qualified
import scipy
import torch
from fast_directional_device_fitter import FastDirectionalDeviceFitter

proof, require = qualified.proof, qualified.require
REPO, WORK = qualified.REPO, qualified.WORK
STAGE = "full_fit_graph_cache4_paired_301_v2"
FRAMES, CPU_FRAMES = frozen.FRAMES, frozen.CPU_FRAMES
ARMS = ("fast", "full")
ALLOCATOR_FRACTION = 0.375
MAX_RESERVED_BYTES = 6 * 1024**3
STARTUP_FREE_MIB, ARM_FREE_MIB = 8192, 2048
QUALIFICATION = WORK / "full_fit_graph_four_packet_qualification_v1"
QUALIFICATION_SHA = "d9b2f40b13f62b75918fc34dc4c1a0ec26a45a6da8c2dc657f8091ab9728cf50"
CACHE4_QUALIFICATION = WORK / "full_fit_graph_cache4_packet_qualification_v1"
CACHE4_QUALIFICATION_SHA = "9bdaf23eb6b8a6adb76052ee75e5d03ceb8fc39a993f8ca886714c2491338101"
FAILED_RUN = WORK / "full_fit_graph_cache4_paired_301_v1"
FAILED_SHA = "01fd34a9e8b81a271e62309421041105c37f0a8e5694d35f72d4a138671b3521"
FAILED_SOURCE_SHA = "19db0bdfbcb148f844a16c5a2652e7ddd8bdfee4489ef9d5d6861bc76f270a09"
SOURCE_PINS = {
    "work/three_body/benchmark_full_fit_graph_cache4_301.py": FAILED_SOURCE_SHA,
    **qualified.SOURCE_PINS,
    **cache_qualified.SOURCE_PINS,
    "work/three_body/qualify_full_fit_graph_cache4.py": (
        "6679eefb76a5d2852f9dc43c4e620d05f020bb1f622d3a98f2f0ee9dd87655d1"
    ),
    "work/three_body/benchmark_full_fit_graph301.py": (
        "2f660bad0bba82c903f86b41b865f489fa89122d4a5bbf1dc70ffa7f0cf22132"
    ),
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
        "combined unchanged caches exceeded the fixed 6GiB process bound",
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
        "Remaining300 can contain new captures; both arms use capacity4 LRU caches.",
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


def cache_prediction(batch_counts):
    """Host-only fixed-shape LRU4 prediction; no tensor, ray or payload access."""
    require(len(batch_counts) > 0, "nonempty candidate-count sequence required")
    lru, steps = [], []
    misses = hits = evictions = 0
    for batch in batch_counts:
        require(type(batch) is int and 6 <= batch <= 30, "fixed candidate budget 6..30 required")
        captured = batch not in lru
        if captured:
            misses += 1
            if len(lru) == 4:
                lru.pop(0)
                evictions += 1
        else:
            hits += 1
            lru.remove(batch)
        lru.append(batch)
        steps.append(
            dict(
                captured=captured,
                statistics=dict(
                    capacity=4,
                    captures=misses,
                    hits=hits,
                    misses=misses,
                    evictions=evictions,
                    failures=0,
                    resident=len(lru),
                    peak_resident=len(lru),
                ),
            )
        )
    return dict(capacity=4, steps=steps, final_statistics=steps[-1]["statistics"].copy())


def validate_cache_statistics(statistics, predicted):
    require(statistics == predicted["statistics"], "Full cache history differs from prediction")


def cache4_qualification_gate(evidence):
    """Require actual eight-fit LRU exercise plus its successful4/301 proof edges."""
    record = cache_qualified.sealed_header(
        CACHE4_QUALIFICATION,
        "qualification.provenance.json",
        CACHE4_QUALIFICATION_SHA,
        cache_qualified.STAGE,
        evidence,
    )
    summary = cache_qualified.completed_summary(CACHE4_QUALIFICATION, record, evidence)
    require(
        summary["passed"] is True
        and summary["all77_CUDA_byte_exact"] is True
        and summary["frames"] == list(cache_qualified.FRAMES)
        and summary["sequence_batches"] == list(cache_qualified.BATCHES)
        and summary["completed_fit_calls"] == 8
        and summary["actual_graph_constructions"] == 6
        and summary["initialize_calls"] == summary["advance_calls"] == summary["raw_decode"] == 0,
        "complete fixed eight-fit cache4 qualification required",
    )
    cache_qualified.validate_cache_step(7, summary["cache_statistics"])
    for relative, digest in cache_qualified.SOURCE_PINS.items():
        path = REPO / relative
        require(record["inputs"].get(str(path)) == digest, "cache4 qualified source edge differs")
        evidence.add(path, digest)
    caller = "work/three_body/qualify_full_fit_graph_cache4.py"
    require(
        record["inputs"].get(str(REPO / caller)) == SOURCE_PINS[caller],
        "cache4 qualification caller differs",
    )
    for root, filename, digest in (
        (cache_qualified.QUAL4, "qualification.provenance.json", cache_qualified.QUAL4_SHA),
        (cache_qualified.QUAL301, "benchmark.provenance.json", cache_qualified.QUAL301_SHA),
    ):
        require(
            record["inputs"].get(str(root / filename)) == digest,
            "cache4 preceding successful qualification edge differs",
        )
    return summary


def resource_revision_gate(evidence):
    """Bind only the failed seal/summary/attempt facts; no failed fit arrays load."""
    seal_path = FAILED_RUN / "benchmark.provenance.json"
    evidence.add(seal_path, FAILED_SHA)
    seal = proof.read_json(seal_path)
    require(
        seal["stage"] == "full_fit_graph_cache4_paired_301_v1"
        and seal["status"] == "failed"
        and bool(seal["finished_at"])
        and seal["error_type"] == "OutOfMemoryError"
        and seal["completed_pairs"] == 41
        and seal["completed_arms"] == 82,
        "fixed failed cache4 comparison required",
    )
    source = WORK / "benchmark_full_fit_graph_cache4_301.py"
    require(
        seal["inputs"].get(str(source)) == FAILED_SOURCE_SHA, "failed comparison source differs"
    )
    evidence.add(source, FAILED_SOURCE_SHA)
    for relative, digest in SOURCE_PINS.items():
        if relative == "work/three_body/benchmark_full_fit_graph_cache4_301.py":
            continue
        require(
            seal["inputs"].get(str(REPO / relative)) == digest,
            "resource revision numerical source edge differs",
        )
    for path, digest in (
        (QUALIFICATION / "qualification.provenance.json", QUALIFICATION_SHA),
        (CACHE4_QUALIFICATION / "qualification.provenance.json", CACHE4_QUALIFICATION_SHA),
        (baseline.PACKETS / "preparation.provenance.json", baseline.PACKET_SHA),
        (baseline.GOLDEN / "fit.provenance.json", baseline.GOLDEN_SHA),
        (baseline.CPU_REFERENCE / "fit.provenance.json", baseline.CPU_SHA),
    ):
        require(
            seal["inputs"].get(str(path)) == digest, "resource revision saved input edge differs"
        )

    def member(name):
        path = FAILED_RUN / name
        row = seal["outputs"][name]
        require(
            Path(row["path"]) == path and path.stat().st_size == row["size_bytes"],
            "failed comparison named metadata path/size differs",
        )
        evidence.add(path, row["sha256"])
        return proof.read_json(path)

    summary, attempts = member("summary.json"), member("fit_attempts.json")
    require(
        summary["status"] == "failed"
        and summary["passed"] is False
        and summary["error_type"] == "OutOfMemoryError"
        and summary["error"] == seal["error"]
        and summary["completed_pairs"] == 41
        and summary["completed_arms"] == 82
        and summary["fit_calls_attempted"] == 83
        and summary["fit_calls_returned"] == 82
        and summary["validated_fit_calls"] == 82
        and summary["frames"] == list(FRAMES)
        and summary["GPU_used"] is True
        and summary["GPU_initialization_attempted"] is True,
        "failed comparison denominators differ",
    )
    expected = [
        dict(frame_idx=native, arm=arm, fitted_frame_returned=True)
        for i, native in enumerate(FRAMES[:41])
        for arm in pair_order(i)
    ]
    expected.append(dict(frame_idx=FRAMES[41], arm=pair_order(41)[0], fitted_frame_returned=False))
    require(attempts == expected, "failed fit sequence differs")
    require(
        all(
            fragment in seal["error"]
            for fragment in (
                "34.00 MiB",
                "10.56 GiB is free",
                "3.97 GiB allowed",
                "3.92 GiB",
                "3.70 GiB",
            )
        ),
        "fixed OOM resource facts differ",
    )
    return dict(
        failed_stage=seal["stage"],
        failed_seal_sha256=FAILED_SHA,
        failed_source_sha256=FAILED_SOURCE_SHA,
        completed_pairs=41,
        completed_fit_returns=82,
        fit_attempts=83,
        failed_native=FRAMES[41],
        failed_arm=pair_order(41)[0],
        OOM_error=seal["error"],
        previous_allocator_fraction=0.25,
        allocator_fraction=ALLOCATOR_FRACTION,
        previous_max_reserved_bytes=4 * 1024**3,
        max_reserved_bytes=MAX_RESERVED_BYTES,
        previous_startup_free_MiB=6144,
        startup_free_MiB=STARTUP_FREE_MIB,
        before_each_arm_free_MiB=ARM_FREE_MIB,
        rationale="two LRU4 fitter pools exceeded the previous per-process allowance; "
        "10.56GiB physical GPU free was reported at OOM",
        only_resource_policy_changed=True,
        all301_pairs_reexecution_planned=True,
        newly_fitted_calls_planned=602,
        failed_fit_arrays_reused=0,
        FP64_and_cache_capacity_unchanged=True,
    )


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
        cache4_qualification_gate(evidence)
        resource_revision = resource_revision_gate(evidence)
        output.write("resource_revision.json", resource_revision)
        code = {
            path: digest
            for path, digest in evidence.paths.items()
            if path.endswith(".py") and ".venv" not in Path(path).parts
        }
        rig, cfg, initial, packets, golden, cpu, metadata, capacity = load_inputs(evidence, code)
        prediction = cache_prediction([len(packets[n]["proposal_pose_xyz"]) for n in FRAMES])
        require(
            prediction["final_statistics"]["captures"] == 13,
            "registered LRU4 capture prediction must be13 for both arms",
        )
        output.write("cache_prediction.json", prediction)
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
                cache_entries=dict(fast=4, full=4),
                expected_capture_counts=dict(fast=13, full=13),
                experiment_scope="resource-only revision; fresh full301 cache4 comparison",
                resource_revision=resource_revision,
                previous_cache1_success_sha256=cache_qualified.QUAL301_SHA,
                cache4_qualification_sha256=CACHE4_QUALIFICATION_SHA,
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
            full=full.Cache4FullFitDirectionalDeviceFitter(rig, initial.copy(), cfg),
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
                    output.write(
                        f"f{native}_full_cache.json", fitters[arm].full_fit_cache_statistics
                    )
                arms.append(
                    dict(
                        frame_idx=native,
                        arm=arm,
                        fit_seconds=seconds,
                        returned_saved=True,
                        validated=False,
                    )
                )

            def after_fit(arm, *, native=native, index=index):
                memory = memory_sample()
                memories.append(dict(frame_idx=native, arm=arm, **memory))
                check_memory(memory)
                require(
                    len(fitters["fast"].__dict__.get("_fit_graph_cache", {})) <= 4,
                    "Fast cache capacity changed",
                )
                if arm == "full":
                    validate_cache_statistics(
                        fitters[arm].full_fit_cache_statistics, prediction["steps"][index]
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
            row["full_cache_statistics"] = fitters["full"].full_fit_cache_statistics
            row["predicted_graph_capture"] = prediction["steps"][index]["captured"]
            output.write(f"f{native}_timing.json", row)
            require(
                fast_capture == row["predicted_graph_capture"]
                and row["full_graph_captures"] == int(row["predicted_graph_capture"]),
                "actual per-arm capture differs from LRU4 prediction",
            )
            rows.append(row)
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
        require(fast_captures == full_captures == 13, "both LRU4 caches must capture13 times")
        validate_cache_statistics(
            fitters["full"].full_fit_cache_statistics, prediction["steps"][-1]
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
            expected_capture_counts=dict(fast=13, full=13),
            full_cache_statistics=fitters["full"].full_fit_cache_statistics,
            adapter_policy=fitters["full"].full_fit_cache_policy,
            cache_entries=dict(fast=4, full=4),
            experiment_scope="resource-only revision; fresh full301 cache4 comparison",
            resource_revision=resource_revision,
            cache4_qualification_sha256=CACHE4_QUALIFICATION_SHA,
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

"""Fixed eight-fit cache4 qualification over six previously saved tail packets.

Fill four shapes, evict one, hit one, recapture the evicted shape, then configure
a different packet of a cached shape. No simultaneous old fitter is run.
All77 CUDA arrays must be BYTEEXACT; old CPU4 has its frozen 1e-10 tolerance.
No raw/video/model/flow/new proposals or continuous state advancement occurs.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
import time
from pathlib import Path

import benchmark_fast_directional_fitter_v2 as baseline
import compare_temporal_point64_v2 as proof
import cv2
import fit_tail_packets as frozen
import full_fit_graph_cache4 as cache4
import full_fit_graph_fitter as new
import numpy as np
import scipy
import torch

from medaka3d.calib import load_rig

REPO, WORK, require = proof.REPO, proof.WORK, proof.require
STAGE = "full_fit_graph_cache4_packet_qualification_v1"
FRAMES = (66524, 66628, 66703, 66639, 66655, 66703, 66524, 66907)
BATCHES = (15, 14, 16, 18, 17, 16, 15, 16)
CACHE4_SHA = "44fb2661d8fc56b258a9616a4b902cd98e90191ab7f2297303f77d4a538320b3"
QUAL4 = Path(__file__).resolve().parent / "full_fit_graph_four_packet_qualification_v1"
QUAL4_SHA = "d9b2f40b13f62b75918fc34dc4c1a0ec26a45a6da8c2dc657f8091ab9728cf50"
QUAL301 = WORK / "full_fit_graph_paired_301_v1"
QUAL301_SHA = "15734b2a74ade056f540de2e52fa6dfcf46a28b9ef490e95e9a598a8b669980c"
FAST = WORK / "fast_directional_fitter_paired_301_v2"
FAST_SHA = "21244298988acd69ac6bc59edc0a833298144af80f5582b6b06de888ac0bb58c"
REPLAY = WORK / "fast_tail_serial_replay_306_v1"
REPLAY_SHA = "c86ded53f52fcf2853745a55377cd01bfdb0392587c318ada7b1772d2a7ced36"
SOURCE_PINS = {
    "work/three_body/full_fit_graph_cache4.py": CACHE4_SHA,
    "work/three_body/full_fit_graph_fitter.py": (
        "3255c53de0e7ed4ca02872229a3bd198160b9651b556bcd0c121caf9c339a376"
    ),
    "work/three_body/benchmark_fast_directional_fitter_v2.py": (
        "684bbe0960545ec5da39f93945f4e558150a9b60948df8f0b3e870e888df8a71"
    ),
    **{"work/three_body/" + k: v for k, v in baseline.SOURCE_PINS.items()},
    "labs/20260909_three_body/graph_fit.py": (
        "807151e664ed8658f291c4d49eaff69ae71feff0f9a1f9cc9fca00292845edc0"
    ),
    "labs/20260909_three_body/graph_union.py": (
        "43fcadf2e684da6fdf9d161800035d7468914217a8ddde10be3dbb810689bc21"
    ),
    "labs/20260909_three_body/fast_union.py": (
        "248e5af87927ab5f4221efe3b2db9ada6fa623466a3b9d99708c226c1f4a7236"
    ),
    "labs/20260909_three_body/renderer.py": (
        "dfb585e9d51214a2a3c9447d4bc81da8af80d3451e8fc3ef7a44b1a841732e1c"
    ),
    "work/three_body/compare_temporal_point64_v2.py": (
        "2461f383d7c69223c98cb9371058373361a7fb30a409a37d617b60fc2db53790"
    ),
}


def validate_cache_step(index, statistics):
    """Registered LRU exercise, including eviction and a changed-input cache hit."""
    require(type(index) is int and 0 <= index < 8, "eight-step cache exercise required")
    captures = (1, 2, 3, 4, 5, 5, 6, 6)[index]
    expected = dict(
        captures=captures,
        misses=captures,
        hits=index + 1 - captures,
        evictions=(0, 0, 0, 0, 1, 1, 2, 2)[index],
        failures=0,
        resident=min(index + 1, 4),
        peak_resident=min(index + 1, 4),
        capacity=4,
    )
    require(statistics == expected, "actual cache transition differs from registered exercise")
    return expected


def completed_summary(root, seal, evidence):
    path = root / "summary.json"
    row = seal["outputs"]["summary.json"]
    require(
        Path(row["path"]) == path and path.stat().st_size == row["size_bytes"],
        "completed summary path/size differs",
    )
    evidence.add(path, row["sha256"])
    return proof.read_json(path)


def sealed_header(root, name, expected, stage, evidence):
    """Bind the already audited seal; only consumed member payloads are reopened."""
    path = root / name
    evidence.add(path, expected)
    seal = proof.read_json(path)
    require(
        seal["stage"] == stage and seal["status"] == "ok" and seal["finished_at"],
        "successful completed fixed input required",
    )
    return seal


def consumed(root, seal, name, evidence):
    path = (root / name).resolve(strict=True)
    require(path.is_relative_to(root) and name in seal["outputs"], "named output missing")
    row = seal["outputs"][name]
    require(
        Path(row["path"]) == path and path.stat().st_size == row["size"],
        "consumed sealed output path/size changed",
    )
    evidence.add(path, row["sha256"])
    return path


def input_edge(seal, path, evidence):
    path = Path(path).resolve(strict=True)
    row = seal["inputs"].get(str(path))
    require(row is not None and Path(row["path"]) == path, "source/input edge absent")
    evidence.add(path, row["sha256"])


def load_inputs(evidence):
    packets = sealed_header(
        baseline.PACKETS,
        "preparation.provenance.json",
        baseline.PACKET_SHA,
        frozen.PACKET_STAGE,
        evidence,
    )
    golden = sealed_header(
        baseline.GOLDEN, "fit.provenance.json", baseline.GOLDEN_SHA, frozen.STAGE, evidence
    )
    cpu = sealed_header(
        baseline.CPU_REFERENCE, "fit.provenance.json", baseline.CPU_SHA, frozen.STAGE, evidence
    )
    fast = sealed_header(FAST, "benchmark.provenance.json", FAST_SHA, baseline.STAGE, evidence)
    # Historical continuous qualification is evidence, not newly exercised here.
    replay = sealed_header(
        REPLAY, "replay.provenance.json", REPLAY_SHA, "fast_tail_serial_replay_306_v1", evidence
    )
    replay_summary = proof.read_json(consumed(REPLAY, replay, "summary.json", evidence))
    require(replay_summary["status"] == "ok", "historical Fast306 proof required")
    input_edge(replay, WORK / "fast_directional_device_fitter.py", evidence)
    for record in (golden, cpu, fast):
        input_edge(record, baseline.PACKETS / "preparation.provenance.json", evidence)
    for path in (
        baseline.GOLDEN / "fit.provenance.json",
        baseline.CPU_REFERENCE / "fit.provenance.json",
    ):
        input_edge(fast, path, evidence)
    for record in (golden, cpu, fast):
        input_edge(record, Path(frozen.__file__), evidence)
    input_edge(fast, WORK / "fast_directional_device_fitter.py", evidence)
    metadata = proof.read_json(consumed(baseline.PACKETS, packets, "metadata.json", evidence))
    frozen.validate_tail_metadata(metadata)
    cfg = frozen.base.TrackerConfig(**metadata["tracker_config"])
    require(cfg.device == "cuda" and cfg.opt_steps == 14, "frozen CUDA/14-step config required")
    for root, record, device in (
        (baseline.GOLDEN, golden, "cuda"),
        (baseline.CPU_REFERENCE, cpu, "cpu"),
    ):
        baseline.validate_gold_summary(
            proof.read_json(consumed(root, record, "summary.json", evidence)), device
        )
    fast_summary = proof.read_json(consumed(FAST, fast, "summary.json", evidence))
    require(
        fast_summary["passed"] is True and fast_summary["frames"] == list(frozen.FRAMES),
        "successful full301 Fast qualification required",
    )
    calibration = Path(metadata["calibration_directory"]).resolve(strict=True)
    files = [p for p in calibration.iterdir() if p.is_file()]
    require(len(files) == 4, "exact registered four calibration files required")
    for path in files:
        for record in (packets, golden, cpu, fast):
            input_edge(record, path, evidence)
    # The four seals bind the numerical source closure they actually consumed.
    # Reuse these exact keys rather than rescanning every old payload or resolving
    # all historical aliases. New-only adapter/caller sources have explicit pins.
    sources, _ = proof.source_inventory(dict(sys.modules), [Path(__file__).resolve()])
    for path in sources:
        for record in (packets, golden, cpu, fast):
            if str(path) in record["inputs"]:
                input_edge(record, path, evidence)
    # No raw image, full SL or old checkpoint is read by load_rig.
    rig = load_rig(calibration)
    prepared, references = {}, {}
    for native in dict.fromkeys(FRAMES):
        with np.load(
            consumed(baseline.PACKETS, packets, f"f{native}_packet.npz", evidence),
            allow_pickle=False,
        ) as z:
            prepared[native] = {k: z[k].copy() for k in z.files}
        require(int(prepared[native]["frame_idx"]) == native, "native mapping changed")
        references[native] = {}
        controls = [
            ("CUDA_golden", baseline.GOLDEN, golden, f"f{native}_fit.npz"),
            ("Fast301", FAST, fast, f"f{native}_fast_fit.npz"),
        ]
        if native in frozen.CPU_FRAMES:
            controls.append(("CPU4", baseline.CPU_REFERENCE, cpu, f"f{native}_fit.npz"))
        for label, root, record, name in controls:
            with np.load(consumed(root, record, name, evidence), allow_pickle=False) as z:
                references[native][label] = {k: z[k].copy() for k in z.files}
            require(len(references[native][label]) == 77, "all77 reference fields required")
    seed = prepared[FRAMES[0]]["initial_pose_xyz"]
    for packet in prepared.values():
        proof.same(seed, packet["initial_pose_xyz"], "same original registered seed")
    require(
        tuple(len(prepared[n]["proposal_pose_xyz"]) for n in FRAMES) == BATCHES,
        "fixed cache exercise shape sequence differs",
    )
    return cfg, rig, prepared, references, metadata


def compare_all77(actual, references):
    require(len(actual) == 77, "all77 output fields required")
    report = {}
    for label, expected in references.items():
        # The frozen comparator preserves nonfinite layout and all discrete fields.
        report[label] = frozen.compare_fit_arrays(actual, expected)
    passed = all(
        row["byte_exact"] for label in ("CUDA_golden", "Fast301") for row in report[label].values()
    )
    return dict(
        passed=passed,
        comparisons=report,
        CUDA_requires_byte_exact=True,
        CPU_atol=frozen.base.ATOL,
        CPU_rtol=0,
    )


def snapshot(evidence, output, saved, index):
    sources, inventory = proof.source_inventory(dict(sys.modules), [Path(__file__).resolve()])
    output.write(f"source_inventory_{index}.json", inventory)
    for path in sorted(sources):
        digest = evidence.add(path)
        if path not in saved:
            raw = path.read_bytes()
            require(
                proof.hashlib.sha256(raw).hexdigest() == digest, "source changed during snapshot"
            )
            output.write("code_snapshot/" + path.relative_to(REPO).as_posix(), raw)
            saved.add(path)
    for module in (
        importlib.import_module("torch.optim.adam"),
        importlib.import_module("torch.optim.optimizer"),
    ):
        path = Path(module.__file__).resolve(strict=True)
        expected = new.ADAM_SOURCE_SHA if path.name == "adam.py" else None
        digest = evidence.add(path, expected)
        name = "installed_torch_source/" + path.name
        if name not in output.records:
            output.write(name, path.read_bytes())
        require(
            output.records[name]["sha256"] == digest, "installed optimizer source snapshot changed"
        )


def run(args):
    require(args.execute and args.device == "cuda", "explicit CUDA qualification required")
    require(all(proof.valid_sha(v) for v in SOURCE_PINS.values()), "final source pins pending")
    destination = Path(args.output).resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(),
        "new workspace output required",
    )
    output, evidence, saved = proof.Output(destination), proof.Evidence(), set()
    started, begin, rows = proof.utc(), time.perf_counter(), []
    try:
        evidence.add(Path(__file__).resolve())
        for relative, digest in SOURCE_PINS.items():
            evidence.add(REPO / relative, digest)
        snapshot(evidence, output, saved, 0)
        qualification = sealed_header(
            QUAL4,
            "qualification.provenance.json",
            QUAL4_SHA,
            "full_fit_graph_four_packet_qualification_v1",
            evidence,
        )
        qsummary = completed_summary(QUAL4, qualification, evidence)
        require(
            qsummary["passed"] is True and qsummary["all77_CUDA_byte_exact"] is True,
            "prior full-fit qualification required",
        )
        require(
            qualification["inputs"][str(REPO / "work/three_body/full_fit_graph_fitter.py")]
            == SOURCE_PINS["work/three_body/full_fit_graph_fitter.py"],
            "same qualified frozen full fitter required",
        )
        q301 = sealed_header(
            QUAL301,
            "benchmark.provenance.json",
            QUAL301_SHA,
            "full_fit_graph_paired_301_v1",
            evidence,
        )
        q301summary = completed_summary(QUAL301, q301, evidence)
        require(
            q301summary["passed"] is True
            and q301summary["all77_CUDA_byte_exact"] is True
            and q301summary["completed_pairs"] == 301
            and q301summary["fitted_calls"] == 602,
            "completed numerical301 qualification required",
        )
        require(
            q301["inputs"][str(REPO / "work/three_body/full_fit_graph_fitter.py")]
            == SOURCE_PINS["work/three_body/full_fit_graph_fitter.py"],
            "same301 frozen full fitter required",
        )
        cfg, rig, packets, references, metadata = load_inputs(evidence)
        require(not torch.cuda.is_initialized(), "no CUDA before input qualification")
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                frames=list(FRAMES),
                new_fit_calls=8,
                dtype="float64",
                evaluations_per_fit=15,
                Adam_updates_per_fit=14,
                source_pins=SOURCE_PINS,
                batches=list(BATCHES),
                expected_cache_hits=2,
                expected_cache_misses=6,
                expected_cache_evictions=2,
                expected_peak_resident_graphs=4,
                prior_full_fit_qualification_sha256=QUAL4_SHA,
                prior301_qualification_sha256=QUAL301_SHA,
                Adam_source_sha256=new.ADAM_SOURCE_SHA,
                Adam_schedule=[
                    dataclasses.asdict(s) for s in new.frozen_adam_schedule(cfg.learning_rate_mm)
                ],
                numerical_policy=(
                    "same default noncapturable foreach; CPU scalar constants are unrolled"
                ),
                nested_CUDA_graphs=0,
                resident_full_fit_graphs_max=4,
                graph_warmup_full_fits=3,
                maximum_graph_constructions=6,
                maximum_warmup_plus_capture_full_fits=24,
                final_diagnostic_objective_outside_graph=True,
                complete_optimizer_loop_in_one_replay=True,
                capture_warmup_not_counted_as_scientific_fit_calls=True,
                inherited_Fast301_seal=FAST_SHA,
                inherited_Fast306_seal=REPLAY_SHA,
                continuous_tracking=False,
                observed_ID_accuracy=False,
                raw_or_video_decode=False,
                new_model=False,
                raw_input_guards_transitive_through_packet_seal=True,
                allocator_fraction=0.25,
                startup_free_GiB=6,
                per_fit_free_GiB=2,
                failure_policy="preserve partial output; no retry/precision change/cache expansion",
                tracker_config=dataclasses.asdict(cfg),
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
        telemetry = baseline.gpu_sample()
        require(telemetry["free_MiB"] >= 6144, "less than 6GiB startup headroom")
        output.write("startup_gpu.json", telemetry)
        torch.cuda.set_per_process_memory_fraction(0.25, 0)
        require(torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5070 Ti", "different GPU")
        torch.cuda.reset_peak_memory_stats(0)
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
        fitter = cache4.Cache4FullFitDirectionalDeviceFitter(
            rig, packets[FRAMES[0]]["initial_pose_xyz"].copy(), cfg
        )
        for index, native in enumerate(FRAMES):
            prefix = f"q{index:02d}_f{native}"
            telemetry = baseline.gpu_sample()
            require(telemetry["free_MiB"] >= 2048, "less than 2GiB per-fit headroom")
            output.write(prefix + "_gpu_before.json", telemetry)
            packet = packets[native]
            proof.same(fitter.lengths, packet["initial_lengths"], "seed lengths")
            original = baseline.signature(packet)
            arrays, seconds = frozen.fitted_frame(fitter, packet)
            output.write(prefix + "_fit.npz", arrays)
            output.write(prefix + "_iterations.npz", fitter.last_full_fit_trace)
            statistics = fitter.full_fit_cache_statistics
            output.write(prefix + "_cache.json", statistics)
            validate_cache_step(index, statistics)
            require(
                torch.cuda.max_memory_reserved(0) <= 4 * 1024**3,
                "fixed 4GiB reserve budget exceeded",
            )
            require(original == baseline.signature(packet), "fit mutated saved packet")
            result = compare_all77(arrays, references[native])
            row = dict(
                frame_idx=native,
                sequence_index=index,
                fit_seconds=seconds,
                cache_statistics=statistics,
                CPU_reference_compared="CPU4" in references[native],
                graph_replays_current_shape=fitter._full_fit_graph.replays,
                full_graph_constructions_so_far=fitter._full_fit_capture_count,
                batch=len(packet["proposal_pose_xyz"]),
                **result,
                allocated_bytes=torch.cuda.memory_allocated(0),
                reserved_bytes=torch.cuda.memory_reserved(0),
            )
            output.write(prefix + "_comparison.json", row)
            rows.append(row)
            print(
                json.dumps(dict(frame_idx=native, passed=row["passed"], seconds=seconds)),
                flush=True,
            )
            require(
                row["passed"], "full-fit CUDA result is not byte-exact; no tolerance relaxation"
            )
        require(
            fitter._full_fit_capture_count == 6, "cache exercise must construct six full graphs"
        )
        require(torch.cuda.max_memory_reserved(0) <= 4 * 1024**3, "fixed 4GiB reserve budget")
        require(fitter.initialize_count == 0, "temporal initialization must remain zero")
        output.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                passed=True,
                frames=list(FRAMES),
                completed_fit_calls=8,
                actual_graph_constructions=fitter._full_fit_capture_count,
                cache_policy="fixed LRU4; arithmetic and trace unchanged",
                cache_statistics=fitter.full_fit_cache_statistics,
                adapter_policy=fitter.full_fit_cache_policy,
                sequence_batches=list(BATCHES),
                graph_warmup_full_fits=3 * fitter._full_fit_capture_count,
                graph_capture_full_fits=fitter._full_fit_capture_count,
                graph_replay_full_fits=8,
                comparisons=rows,
                all77_CUDA_byte_exact=True,
                historical_CPU_control_atol=frozen.base.ATOL,
                raw_decode=0,
                models=0,
                initialize_calls=0,
                advance_calls=0,
                GPU_used=True,
                actual301_or306_requalified=False,
                speed_improvement_claimed=False,
                timing_scope=(
                    "fit + synchronization + iteration diagnostic CPU copies; "
                    "capture/warmup included when new shape"
                ),
                elapsed_before_guards_seconds=time.perf_counter() - begin,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            ),
        )
        snapshot(evidence, output, saved, 1)
        evidence.final_guard()
        output.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=output.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            completed_comparisons=len(rows),
            error_type=type(error).__name__,
            error=str(error),
        )
        with (destination / "qualification.provenance.json").open("xb") as f:
            f.write(proof.json_bytes(seal))
        raise
    with (destination / "qualification.provenance.json").open("xb") as f:
        f.write(proof.json_bytes(seal))
    return seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--device", choices=["cuda"], required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    run(parser.parse_args())

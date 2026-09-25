"""Rescore four immutable saved FP64 banks; no search, optimizer or tracking.

Full17 and union16+full1 share one original FP64 problem per native. Both must
match all16 tensors saved by the successful Mixed4 experiment byte for byte.
Timing excludes input preparation, result CPU copies, comparison and storage.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import time
from functools import partial
from pathlib import Path

import numpy as np
import qualify_mixed_precision4 as prior
import torch

proof, previous = prior.proof, prior.previous
REPO, WORK, require = prior.REPO, prior.WORK, prior.require
STAGE = "mixed_precision_saved_bank_rescore_four_v1"
MIXED_ROOT = WORK / "mixed_precision_four_packet_v1"
MIXED_SHA = "915e97904c852166b17e99658f32ab1638f069d064f8d048be03fb21f3200d22"
PRIOR_SHA = "aa1ffd60dce629e0afd41cdeb4a16ee2a01f2714db3d06cc232d92a3d95b172c"
UNION_SHA = "e338341eb6f868f9f2759038879d14b0289d48af48dec2e8d1ec9a5fdfa415c7"
FRAMES, BATCHES = prior.FRAMES, prior.BATCHES
SOURCE_PINS = {
    **prior.SOURCE_PINS,
    "work/three_body/qualify_mixed_precision4.py": PRIOR_SHA,
    "work/three_body/fp64_union_pose_bank_rescore.py": UNION_SHA,
}
TENSOR_FIELDS = (
    "selected_bank_index",
    "bank_scores",
    "bank_parts",
    "initial_score",
    "initial_parts",
    "fitted_pose_xyz",
    "local_cost",
    "final_loss_parts",
    "mask_supported",
    "view_supported",
    "positive",
    "negative",
    "visible_alpha",
    "ownership",
    "union_alpha",
    "ray_valid",
)


def arm_order(index):
    require(type(index) is int and 0 <= index < 4, "fixed four pairs only")
    return ("full", "union") if index % 2 == 0 else ("union", "full")


def fixed_plan():
    return dict(
        schema=STAGE,
        frames=list(FRAMES),
        batches=list(BATCHES),
        candidates=61,
        source_pins=SOURCE_PINS,
        mixed_root=str(MIXED_ROOT),
        mixed_sha256=MIXED_SHA,
        arm_order=[list(arm_order(i)) for i in range(4)],
        rescore_calls=8,
        full_objective_evaluations=17,
        union_objective_evaluations=16,
        union_final_full_diagnostic_evaluations=1,
        tensor_fields=list(TENSOR_FIELDS),
        equivalence="all16 tensors BYTEEXACT against saved Mixed4 fp64 fields",
        selection="same saved C0 original + C1..15 FP32 promotions; strict first minimum",
        original_problem="frozen.fit_inputs then mixed_precision_fitter.prepare_original_problem",
        timing="rescore call plus GPU synchronization; excludes preparation/CPU copies/storage",
        timing_interpretation="first call cold; remaining three pairs exploratory; no adoption",
        allocator_fraction=0.25,
        peak_reserved_limit_bytes=4 * 1024**3,
        startup_free_MiB=6144,
        per_arm_free_MiB=2048,
        failure_policy="save completed result and comparison then stop; no retry or tolerance",
        FP32_generation=0,
        optimizer_updates=0,
        fit_calls=0,
        proposal_updates=0,
        graph_captures=0,
        initialize_calls=0,
        advance_calls=0,
        raw_decode=0,
        models=0,
    )


def mixed_member(root, seal, name, evidence):
    """New Mixed4 outputs use size_bytes; old packet manifests retain size."""
    path = (root / name).resolve(strict=True)
    require(path.is_relative_to(root) and name in seal["outputs"], "Mixed4 member missing")
    row = seal["outputs"][name]
    require(
        Path(row["path"]) == path and path.stat().st_size == row["size_bytes"],
        "Mixed4 output path/size mismatch",
    )
    evidence.add(path, row["sha256"])
    return path


def mixed_input(seal, path, evidence):
    path = Path(path).resolve(strict=True)
    digest = seal["inputs"].get(str(path))
    require(proof.valid_sha(digest), "Mixed4 direct input missing")
    evidence.add(path, digest)
    return path


def validate_prior_summary(summary, plan):
    require(plan == prior.fixed_plan(), "fixed successful Mixed4 plan changed")
    require(
        summary["schema"] == prior.STAGE
        and summary["status"] == "ok"
        and summary["passed"] is True
        and summary["frames"] == list(FRAMES)
        and summary["completed_pairs"] == 4
        and summary["completed_fit_calls"] == 8
        and summary["Fast_all77_CUDA_byte_exact"] is True
        and summary["Mixed_invariants_passed"] is True
        and summary["Mixed_equivalent_to_FP64_search"] is False,
        "successful fixed Mixed4 qualification required",
    )
    require(len(summary["comparisons"]) == 4, "four completed Mixed4 pairs required")
    for i, row in enumerate(summary["comparisons"]):
        require(
            row["frame_idx"] == FRAMES[i]
            and row["batch"] == BATCHES[i]
            and row["order"] == list(prior.registered_order(i))
            and row["Fast_control_passed"] is True
            and row["Mixed_invariants_passed"] is True,
            "Mixed4 pair scope/qualification changed",
        )


def read_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def load_inputs(evidence):
    """Only four packets + four saved Mixed trace/fit pairs; no old reference fits."""
    seal = previous.sealed_header(
        MIXED_ROOT, "qualification.provenance.json", MIXED_SHA, prior.STAGE, evidence
    )

    def member(name):
        return mixed_member(MIXED_ROOT, seal, name, evidence)

    summary, plan = proof.read_json(member("summary.json")), proof.read_json(member("plan.json"))
    validate_prior_summary(summary, plan)
    mixed_input(seal, Path(prior.__file__), evidence)
    for relative, digest in prior.SOURCE_PINS.items():
        path = mixed_input(seal, REPO / relative, evidence)
        require(evidence.paths[str(path)] == digest, "Mixed4 source pin mismatch")
    packet_root = prior.baseline.PACKETS
    mixed_input(seal, packet_root / "preparation.provenance.json", evidence)
    packet_seal = previous.sealed_header(
        packet_root,
        "preparation.provenance.json",
        prior.baseline.PACKET_SHA,
        prior.frozen.PACKET_STAGE,
        evidence,
    )
    metadata_path = previous.consumed(packet_root, packet_seal, "metadata.json", evidence)
    mixed_input(seal, metadata_path, evidence)
    metadata = proof.read_json(metadata_path)
    prior.frozen.validate_tail_metadata(metadata)
    cfg = prior.frozen.base.TrackerConfig(**metadata["tracker_config"])
    require(cfg.device == "cuda" and cfg.opt_steps == 14, "original CUDA/14-step context")
    require(
        proof.read_json(member("input_contract.json"))
        == dict(
            tracker_config=metadata["tracker_config"],
            runtime=metadata["runtime"],
            opencv_threads=metadata["opencv_threads"],
        ),
        "same Mixed4 original input context",
    )
    calibration = Path(metadata["calibration_directory"]).resolve(strict=True)
    files = [p for p in calibration.iterdir() if p.is_file()]
    require(len(files) == 4, "four original calibration files required")
    for path in files:
        mixed_input(seal, path, evidence)
        previous.input_edge(packet_seal, path, evidence)
    rig = previous.load_rig(calibration)
    packets, traces, checks = {}, {}, []
    for index, native in enumerate(FRAMES):
        path = previous.consumed(packet_root, packet_seal, f"f{native}_packet.npz", evidence)
        mixed_input(seal, path, evidence)
        packet = read_arrays(path)
        require(
            int(packet["frame_idx"]) == native
            and len(packet["proposal_pose_xyz"]) == BATCHES[index],
            "fixed bank/packet native",
        )
        trace = read_arrays(member(f"f{native}_mixed_trace.npz"))
        fitted = read_arrays(member(f"f{native}_mixed_fit.npz"))
        prior.validate_numerical_outputs(fitted)
        check = prior.validate_trace(packet, fitted, trace)
        require(
            check == proof.read_json(member(f"f{native}_mixed_bank_check.json")),
            "saved bank qualification changed",
        )
        ray_check = prior.validate_fp64_rays(packet, trace, rig, cfg)
        require(
            ray_check == proof.read_json(member(f"f{native}_mixed_original_rays_check.json")),
            "original FP64 ray qualification changed",
        )
        require(check == summary["comparisons"][index]["subset2"], "bank/summary mismatch")
        packets[native], traces[native] = packet, trace
        checks.append(dict(frame_idx=native, bank=check, rays=ray_check))
    seed = packets[FRAMES[0]]["initial_pose_xyz"]
    for packet in packets.values():
        proof.same(seed, packet["initial_pose_xyz"], "one original fixed seed")
    require(not torch.cuda.is_initialized(), "no CUDA during input validation")
    return cfg, rig, packets, traces, metadata, checks


def problem_arrays(problem, bank):
    """Snapshot declared numerical inputs; renderer lazy caches are not input state."""
    tensors = {
        "pose_bank": bank,
        "original_pose_xyz": problem.pose,
        "axes": problem.axes,
        "parallel_sigma": problem.parallel_sigma,
        "enabled": problem.enabled,
        "lower": problem.lower,
        "upper": problem.upper,
    }
    tensors.update(
        {
            "inputs_" + f.name: getattr(problem.inputs, f.name)
            for f in dataclasses.fields(problem.inputs)
        }
    )
    tensors.update(
        {
            "image_ray_" + k: getattr(problem.renderer, "ray_" + k)
            for k in ("origins", "directions", "valid")
        }
    )
    require(all(isinstance(v, torch.Tensor) for v in tensors.values()), "complete image problem")
    return {k: v.detach().cpu().numpy().copy() for k, v in tensors.items()}


def validate_problem(arrays, packet, trace):
    proof.same(arrays["pose_bank"], trace["fp64_pose_bank"], "saved bank input")
    proof.same(arrays["original_pose_xyz"], packet["proposal_pose_xyz"], "original initial input")
    for key in (
        "image_ray_origins",
        "image_ray_directions",
        "inputs_origins",
        "inputs_directions",
        "inputs_held",
    ):
        proof.same(arrays[key], trace["fp64_" + key], "original FP64 problem " + key)
    proof.same(arrays["image_ray_valid"], trace["fp64_ray_valid"], "original image validity")
    for key, value in arrays.items():
        require(np.isfinite(value).all(), "nonfinite original problem " + key)


def invoke(function, bank, problem, cfg, *, synchronize, clock=time.perf_counter):
    synchronize()
    started = clock()
    result = function(
        bank,
        config=cfg,
        inputs=problem.inputs,
        renderer=problem.renderer,
        axes=problem.axes,
        parallel_sigma=problem.parallel_sigma,
        enabled=problem.enabled,
        directional=True,
    )
    synchronize()
    elapsed = clock() - started
    arrays = {
        k: v.detach().cpu().numpy().copy() for k, v in result.items() if isinstance(v, torch.Tensor)
    }
    metadata = {k: v for k, v in result.items() if not isinstance(v, torch.Tensor)}
    return arrays, metadata, elapsed


def comparison(actual, trace):
    """Save value-difference locations; byte checks also catch signed-zero changes."""
    expected = {k: trace["fp64_" + k] for k in TENSOR_FIELDS}
    fields = {}
    for k in sorted(actual.keys() | expected.keys()):
        a, b = actual.get(k), expected.get(k)
        shape = a is not None and b is not None and a.shape == b.shape
        dtype = a is not None and b is not None and a.dtype == b.dtype
        row = dict(
            actual_shape=None if a is None else list(a.shape),
            expected_shape=None if b is None else list(b.shape),
            actual_dtype=None if a is None else a.dtype.str,
            expected_dtype=None if b is None else b.dtype.str,
            byte_exact=bool(shape and dtype and a.tobytes() == b.tobytes()),
            changed_indices=np.argwhere(a != b).tolist() if shape else None,
        )
        if a is not None and a.dtype.kind == "f":
            row["nonfinite_indices"] = np.argwhere(~np.isfinite(a)).tolist()
            if shape and b.dtype.kind == "f":
                keep = np.isfinite(a) & np.isfinite(b)
                delta = np.abs(a[keep] - b[keep])
                maximum = float(delta.max()) if len(delta) else 0.0
                row["maximum_absolute_difference"] = maximum if np.isfinite(maximum) else None
        fields[k] = row
    return dict(
        passed=actual.keys() == expected.keys() and all(x["byte_exact"] for x in fields.values()),
        fields=fields,
        tensor_count=len(actual),
    )


def validate_result(arrays, metadata, trace, arm):
    require(arm in ("full", "union"), "known rescore arm")
    require(
        metadata["schema"]
        == ("fp64_pose_bank_rescore_v1" if arm == "full" else "fp64_union_pose_bank_rescore_v1")
        and metadata["bank_size"] == 16
        and metadata["objective_evaluations"] == 17
        and metadata["directional"] is True,
        "rescorer metadata contract",
    )
    if arm == "union":
        require(
            metadata["union_objective_evaluations"] == 16
            and metadata["full_diagnostic_evaluations"] == 1,
            "union/full call counts",
        )
    require(comparison(arrays, trace)["passed"], "saved FP64 result not BYTEEXACT")
    require(all(np.isfinite(v).all() for v in arrays.values()), "nonfinite rescore result")
    scores = arrays["bank_scores"]
    require(
        np.array_equal(arrays["selected_bank_index"], np.argmin(scores, axis=0)),
        "strict first bank minimum changed",
    )
    require(np.all(arrays["local_cost"] <= arrays["initial_score"]), "initial objective regression")


def timing_summary(rows):
    require(len(rows) == 4, "four paired timings required")
    for i, row in enumerate(rows):
        require(
            row["frame_idx"] == FRAMES[i]
            and row["order"] == list(arm_order(i))
            and row["batch"] == BATCHES[i],
            "fixed paired timing scope",
        )
        require(
            set(row["seconds"]) == {"full", "union"}
            and all(np.isfinite(x) and x > 0 for x in row["seconds"].values()),
            "finite positive arm timing",
        )
    ratio = [r["seconds"]["union"] / r["seconds"]["full"] for r in rows]
    return dict(
        first_pair=rows[0],
        remaining_pairs=rows[1:],
        total_seconds={arm: sum(r["seconds"][arm] for r in rows) for arm in ("full", "union")},
        paired_union_over_full=ratio,
        remaining_paired_median=float(np.median(ratio[1:])),
        remaining_union_faster_count=sum(x < 1 for x in ratio[1:]),
        first_call_cold=True,
        steady_state_or_adoption_claimed=False,
    )


def save_checked_result(output, prefix, arrays, info, elapsed, trace, arm, completed, memory):
    output.write(prefix + "_rescore.npz", arrays)
    output.write(prefix + "_metadata.json", info)
    output.write(prefix + "_comparison.json", comparison(arrays, trace))
    output.write(
        prefix + "_timing.json",
        dict(rescore_seconds=elapsed, completed_rescore_calls=completed, **memory),
    )
    validate_result(arrays, info, trace, arm)
    require(np.isfinite(elapsed) and elapsed > 0, "invalid elapsed rescore time")


def run(args):
    require(args.execute and args.device == "cuda", "explicit CUDA execution required")
    require(all(proof.valid_sha(s) for s in SOURCE_PINS.values()), "source pins pending")
    destination = Path(args.output).resolve()
    require(destination.is_relative_to(WORK) and not destination.exists(), "new workspace output")
    output, evidence, saved = proof.Output(destination), proof.Evidence(), set()
    started = proof.utc()
    rows, completed, gpu_used, attempted = [], 0, False, False
    try:
        evidence.add(Path(__file__).resolve())
        for relative, digest in SOURCE_PINS.items():
            evidence.add(REPO / relative, digest)
        mixed = importlib.import_module("mixed_precision_fitter")
        old = importlib.import_module("fp64_pose_bank_rescore")
        union = importlib.import_module("fp64_union_pose_bank_rescore")
        output.write("plan.json", fixed_plan())
        previous.snapshot(evidence, output, saved, 0)
        cfg, rig, packets, traces, metadata, checks = load_inputs(evidence)
        output.write("input_checks.json", checks)
        output.write(
            "input_contract.json",
            dict(
                tracker_config=dataclasses.asdict(cfg),
                runtime=metadata["runtime"],
                opencv_threads=metadata["opencv_threads"],
            ),
        )
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        prior.cv2.setNumThreads(metadata["opencv_threads"])
        torch.manual_seed(0)
        versions = dict(
            numpy=np.__version__,
            scipy=prior.scipy.__version__,
            opencv=prior.cv2.__version__,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
        )
        require(versions == metadata["runtime"], "frozen package versions changed")
        telemetry = prior.baseline.gpu_sample()
        output.write("startup_gpu.json", telemetry)
        require(telemetry["free_MiB"] >= 6144, "less than6GiB startup headroom")
        attempted = True
        torch.cuda.set_per_process_memory_fraction(0.25, 0)
        gpu_used = True
        require(torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5070 Ti", "different GPU")
        torch.cuda.reset_peak_memory_stats(0)
        output.write(
            "environment.json",
            dict(
                **versions,
                GPU=torch.cuda.get_device_name(0),
                torch_threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads(),
                opencv_threads=prior.cv2.getNumThreads(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            ),
        )
        fitter = prior.FastDirectionalDeviceFitter(
            rig, packets[FRAMES[0]]["initial_pose_xyz"].copy(), cfg
        )
        functions = dict(full=old.rescore_pose_bank, union=union.rescore_union_pose_bank)
        for index, native in enumerate(FRAMES):
            packet, trace = packets[native], traces[native]
            signature = prior.baseline.signature(packet)
            trace_signature = prior.baseline.signature(trace)
            proof.same(fitter.lengths, packet["initial_lengths"], "original seed lengths")
            problem = mixed.prepare_original_problem(
                fitter, *prior.frozen.fit_inputs(packet, rig, cfg)
            )
            bank = torch.as_tensor(
                trace["fp64_pose_bank"], dtype=torch.float64, device=problem.pose.device
            ).clone()
            original = problem_arrays(problem, bank)
            validate_problem(original, packet, trace)
            output.write(f"f{native}_problem.npz", original)
            seconds = {}
            for arm in arm_order(index):
                prefix = f"f{native}_{arm}"
                telemetry = prior.baseline.gpu_sample()
                output.write(prefix + "_gpu_before.json", telemetry)
                require(telemetry["free_MiB"] >= 2048, "less than2GiB per-arm headroom")
                arrays, info, elapsed = invoke(
                    functions[arm],
                    bank,
                    problem,
                    cfg,
                    synchronize=partial(torch.cuda.synchronize, problem.pose.device),
                )
                completed += 1
                save_checked_result(
                    output,
                    prefix,
                    arrays,
                    info,
                    elapsed,
                    trace,
                    arm,
                    completed,
                    dict(
                        allocated_bytes=torch.cuda.memory_allocated(0),
                        reserved_bytes=torch.cuda.memory_reserved(0),
                    ),
                )
                after = problem_arrays(problem, bank)
                for key, value in original.items():
                    proof.same(value, after[key], "rescore changed input " + key)
                require(
                    signature == prior.baseline.signature(packet)
                    and trace_signature == prior.baseline.signature(trace),
                    "changed saved bank/packet",
                )
                require(
                    torch.cuda.max_memory_reserved(0) <= 4 * 1024**3, "fixed4GiB reserved limit"
                )
                seconds[arm] = elapsed
                del arrays, after
            row = dict(
                frame_idx=native,
                batch=BATCHES[index],
                order=list(arm_order(index)),
                seconds=seconds,
                all16_tensors_BYTEEXACT=True,
                inputs_unchanged=True,
            )
            output.write(f"f{native}_summary.json", row)
            rows.append(row)
            print(proof.json.dumps(row), flush=True)
            del problem, bank, original
        require(completed == 8 and fitter.initialize_count == 0, "eight rescores/no initialization")
        output.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                passed=True,
                frames=list(FRAMES),
                candidates=61,
                completed_pairs=4,
                completed_rescore_calls=completed,
                comparisons=rows,
                timings=timing_summary(rows),
                all16_tensors_BYTEEXACT=True,
                mixed_qualification_sha256=MIXED_SHA,
                GPU_used=gpu_used,
                GPU_initialization_attempted=attempted,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                FP32_generation=0,
                optimizer_updates=0,
                fit_calls=0,
                proposal_updates=0,
                graph_captures=0,
                initialize_calls=0,
                advance_calls=0,
                raw_decode=0,
                models=0,
                speed_or_tracking_adoption_claimed=False,
            ),
        )
        previous.snapshot(evidence, output, saved, 1)
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
            completed_pairs=len(rows),
            completed_rescore_calls=completed,
            GPU_used=gpu_used,
            GPU_initialization_attempted=attempted,
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())

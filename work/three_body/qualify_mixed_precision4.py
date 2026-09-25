"""Fixed four-packet FP32 search / FP64 rescore exploration, with strict Fast controls.

Eight fit calls in registered alternating order. Mixed is a distinct numerical
trajectory; its differences never weaken or replace the frozen Fast qualification.
No raw/model/flow/proposal generation, temporal begin/advance, or301 adoption.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import qualify_full_fit_graph4 as previous
import scipy
import torch
from fast_directional_device_fitter import FastDirectionalDeviceFitter
from node_geometry import assigned_rays
from renderer import RefractiveFishRenderer, RendererConfig

proof, baseline, frozen = previous.proof, previous.baseline, previous.frozen
REPO, WORK, require = proof.REPO, proof.WORK, proof.require
STAGE = "mixed_precision_four_packet_v1"
FRAMES = (66524, 66628, 66703, 66907)
BATCHES = (15, 14, 16, 16)
MIXED_SHA = "bcd4ba0b75833f327d36083ca74121fb3d21296b4fe34d71bdf52fd331848285"
GENERATOR_SHA = "2967015cb21beadc56545bef03bddd8578a550a6d761bf07b834ce53f26fc79b"
SOURCE_PINS = {
    **previous.SOURCE_PINS,
    "work/three_body/qualify_full_fit_graph4.py": (
        "40ab249079c11dc10d5e9e3d86770937a60900302df72f3f37dff53380986914"
    ),
    "work/three_body/fast_directional_device_fitter.py": (
        "5804003d537fe5ad144927c9629bf12927fc0ce9dc460615a78125cfe16c4abd"
    ),
    "work/three_body/fp64_pose_bank_rescore.py": (
        "80f341d475e298e0b80ea20cb59004d01480a3e944c13b95fff01682ff6125d2"
    ),
    "work/three_body/mixed_precision_fitter.py": MIXED_SHA,
    "work/three_body/fp32_pose_proposals.py": GENERATOR_SHA,
}


def registered_order(index):
    require(type(index) is int and 0 <= index < 4, "four fixed pairs required")
    return ("fast", "mixed") if index % 2 == 0 else ("mixed", "fast")


def fixed_plan():
    return dict(
        schema=STAGE,
        frames=list(FRAMES),
        batches=list(BATCHES),
        arm_order=[list(registered_order(i)) for i in range(4)],
        fit_calls=8,
        source_pins=SOURCE_PINS,
        original_inputs_loader="qualify_full_fit_graph4.load_inputs",
        Fast_control="all77 CUDA golden/Fast301 BYTEEXACT; CPU4 atol1e-10 rtol0",
        Mixed_comparison="descriptive all77 differences; not the old equivalence gate",
        pose_bank="C0 exact original FP64; C1..15 promoted full FP32 path",
        selection="per outer proposal strict first minimum of16 FP64 scores",
        subset2="original initial + FP32 best pose; diagnostic only; all16 retained",
        original_objective_and_ray_inputs="original FP64 before any FP32 rounding",
        FP32_evaluations=15,
        FP32_updates=14,
        FP64_objective_evaluations=17,
        FP32_cache_capacity=4,
        warmup_replays_per_new_graph=3,
        allocator_fraction=0.375,
        peak_reserved_limit_bytes=6 * 1024**3,
        startup_free_MiB=8192,
        per_fit_free_MiB=2048,
        failure_policy="preserve outputs and stop; no automatic retry or parameter change",
        scientific_array_finiteness=(
            "pose/cost/parts/trajectory/bank must be finite; semantic missing-ray or margin "
            "sentinels in77-field evidence are preserved and explicitly reported"
        ),
        raw_decode=0,
        models=0,
        initialize_calls=0,
        advance_calls=0,
        continuous_tracking=False,
        identity_accuracy=False,
        adoption301=False,
        timing_scope="fit+synchronization+trace CPU copies; warmup/capture included on misses",
    )


def mixed_difference(actual, reference):
    """Retain every discrete difference, float nonfinite location and schema difference."""
    require(
        len(actual) == len(reference) == 77 and actual.keys() == reference.keys(),
        "all77 fields required",
    )
    fields = {}
    for key, a in actual.items():
        b = reference[key]
        require(a.dtype.kind in "fbiuUS" and b.dtype.kind in "fbiuUS", "unsupported saved dtype")
        same_shape, same_dtype = a.shape == b.shape, a.dtype == b.dtype
        row = dict(
            actual_shape=list(a.shape),
            reference_shape=list(b.shape),
            actual_dtype=a.dtype.str,
            reference_dtype=b.dtype.str,
            same_shape=same_shape,
            same_dtype=same_dtype,
            byte_exact=same_shape and same_dtype and a.tobytes() == b.tobytes(),
        )
        if a.dtype.kind == "f" or b.dtype.kind == "f":
            row["nonfinite"] = {
                label: {
                    name: np.argwhere(fn(x)).tolist()
                    for name, fn in (
                        ("nan", np.isnan),
                        ("positive_inf", np.isposinf),
                        ("negative_inf", np.isneginf),
                    )
                }
                if x.dtype.kind == "f"
                else {}
                for label, x in (("actual", a), ("reference", b))
            }
            row["maximum_absolute_difference_common_finite"] = None
            row["common_finite_count"] = 0
            if same_shape and a.dtype.kind == b.dtype.kind == "f":
                keep = np.isfinite(a) & np.isfinite(b)
                row["common_finite_count"] = int(keep.sum())
                with np.errstate(over="ignore", invalid="ignore"):
                    delta = np.abs(a[keep].astype(np.float64) - b[keep].astype(np.float64))
                maximum = float(delta.max()) if len(delta) else 0.0
                row["finite_difference_overflow"] = not np.isfinite(maximum)
                row["maximum_absolute_difference_common_finite"] = (
                    maximum if np.isfinite(maximum) else None
                )
        else:
            row["changed_indices"] = None
            if same_shape:
                changed = a != b
                row.update(
                    changed_indices=np.argwhere(changed).tolist(),
                    actual_values=a[changed].tolist(),
                    reference_values=b[changed].tolist(),
                )
        fields[key] = row
    return dict(
        schema="mixed_precision_all77_descriptive_difference_v1",
        fields=fields,
        byte_equal_fields=sum(row["byte_exact"] for row in fields.values()),
        tolerance_or_equivalence_pass_not_applied=True,
    )


def _array(values, key, shape, dtype, *, finite=True):
    a = values[key]
    require(
        isinstance(a, np.ndarray) and a.shape == shape and a.dtype == np.dtype(dtype),
        "saved mixed schema: " + key,
    )
    if finite:
        require(np.isfinite(a).all(), "nonfinite mixed numerical output: " + key)
    return a


def validate_trace(packet, arrays, trace):
    """Saved-array-only bank proof, independent of the producer's selection function."""
    original = packet["proposal_pose_xyz"]
    b = len(original)
    require(
        original.shape == (b, 3, 3, 3) and original.dtype == np.float64,
        "original FP64 proposal schema",
    )
    bank = _array(trace, "fp64_pose_bank", (16, b, 3, 3, 3), "float64")
    path = _array(trace, "fp32_pose_before", (15, b, 3, 3, 3), "float32")
    proof.same(
        _array(trace, "original_pose_xyz", original.shape, "float64"),
        original,
        "saved original proposal",
    )
    proof.same(bank[0], original, "C0 original FP64 exact")
    proof.same(path[0], original.astype(np.float32), "FP32 starts at rounded original")
    proof.same(bank[1:], path.astype(np.float64), "C1..15 exact FP32 promotion")
    f32scores = _array(trace, "fp32_score", (15, b), "float32")
    _array(trace, "fp32_parts", (15, b, 5), "float32")
    for name, steps in (
        ("base_gradient", 15),
        ("update_gradient", 14),
        ("moment", 14),
        ("variance", 14),
        ("pose_after", 14),
    ):
        _array(trace, "fp32_" + name, (steps, b, 3, 3, 3), "float32")
    proof.same(path[1:], trace["fp32_pose_after"], "FP32 update path continuity")
    flags = _array(trace, "fp32_finite_flags", (3,), "bool")
    require(flags.all(), "FP32 score/gradient/pose finite flags failed")
    _array(trace, "fp32_uses_positive_product_bound", (), "bool")
    require(not bool(trace["fp32_uses_positive_product_bound"]), "FP32 product bound must be False")
    f32best = _array(trace, "fp32_best_iteration", (b,), "int64")
    proof.same(f32best, np.argmin(f32scores, axis=0).astype(np.int64), "FP32 strict first best")
    f32bestscore = f32scores[f32best, np.arange(b)]
    proof.same(_array(trace, "fp32_best_score", (b,), "float32"), f32bestscore, "FP32 best score")
    scores = _array(trace, "fp64_bank_scores", (16, b), "float64")
    parts = _array(trace, "fp64_bank_parts", (16, b, 5), "float64")
    selected = _array(trace, "fp64_selected_bank_index", (b,), "int64")
    proof.same(selected, np.argmin(scores, axis=0).astype(np.int64), "FP64 strict first best")
    columns = np.arange(b)
    for name, expected in (
        ("fitted_pose_xyz", bank[selected, columns]),
        ("local_cost", scores[selected, columns]),
        ("final_loss_parts", parts[selected, columns]),
        ("initial_score", scores[0]),
        ("initial_parts", parts[0]),
    ):
        proof.same(
            _array(arrays, name, expected.shape, "float64"),
            expected,
            "returned original-schema " + name,
        )
        proof.same(trace["fp64_" + name], expected, "rescore output " + name)
    require(
        np.all(arrays["local_cost"] <= arrays["initial_score"]),
        "FP64 selected output worsened initial",
    )
    for name in ("positive", "negative"):
        _array(trace, "fp64_" + name, (b, 3, 3), "float64")
    view = (trace["fp64_positive"] >= 0.025) & (
        trace["fp64_positive"] >= 2 * trace["fp64_negative"]
    )
    proof.same(_array(trace, "fp64_view_supported", (b, 3, 3), "bool"), view, "frozen view support")
    support = view.sum(1) >= 2
    proof.same(_array(trace, "fp64_mask_supported", (b, 3), "bool"), support, "frozen mask support")
    proof.same(_array(arrays, "mask_supported", (b, 3), "bool"), support, "returned mask support")
    n = packet["fit_pixels"].shape[1]
    for name in ("visible_alpha", "ownership"):
        _array(trace, "fp64_" + name, (b, 3, n, 3), "float64")
    _array(trace, "fp64_union_alpha", (b, 3, n), "float64")
    _array(trace, "fp64_ray_valid", (3, n), "bool")
    second = f32best + 1
    subset_index = np.where(scores[second, columns] < scores[0], second, 0).astype(np.int64)
    subset_score = scores[subset_index, columns]
    minimum = scores[selected, columns]
    expected_subset = dict(
        subset2_bank_index=subset_index,
        subset2_score=subset_score,
        subset2_missed_strict_improvement=subset_score > minimum,
        subset2_cost_regret=subset_score - minimum,
    )
    for key, value in expected_subset.items():
        proof.same(trace[key], value, "independent subset2 " + key)
    return dict(
        all16_first_minimum=True,
        exact_original_and_promotions=True,
        final_no_worse_than_initial=True,
        candidates=b,
        subset2_missed_proposals=np.flatnonzero(subset_score > minimum).tolist(),
        subset2_maximum_cost_regret=float((subset_score - minimum).max()),
        selected_bank_index=selected.tolist(),
        best16_equals_initial_count=int((selected == 0).sum()),
    )


def validate_numerical_outputs(arrays):
    require(len(arrays) == 77, "all77 Mixed arrays required")
    for key, a in arrays.items():
        if a.dtype.kind != "f":
            continue
        if key.endswith("native_pixel_error"):
            assigned = arrays[key.removesuffix("native_pixel_error") + "assigned_source"]
            require(
                np.array_equal(np.isnan(a), ~assigned) and np.isfinite(a[assigned]).all(),
                "pixel error must be finite for assigned / NaN for absent rows: " + key,
            )
        elif key.endswith("score_margin"):
            require(
                (np.isfinite(a) | np.isposinf(a)).all(), "margin permits only no-competitor +inf"
            )
        else:
            require(np.isfinite(a).all(), "nonfinite Mixed scientific output: " + key)


def validate_public_schema(actual, reference):
    require(actual.keys() == reference.keys() and len(actual) == 77, "same public77 fields")
    for key in actual:
        require(
            actual[key].shape == reference[key].shape and actual[key].dtype == reference[key].dtype,
            "Mixed changed public77 shape/dtype: " + key,
        )


def validate_fp64_rays(packet, trace, rig, config):
    """No render/fit: repeat CPU ray construction from the bound original inputs."""
    renderer = RefractiveFishRenderer(
        rig,
        {v: packet["fit_pixels"][i] for i, v in enumerate(rig.views)},
        RendererConfig(width_ratio=config.width_ratio),
        device="cpu",
        dtype=torch.float64,
    )
    for field in ("origins", "directions"):
        proof.same(
            trace["fp64_image_ray_" + field],
            getattr(renderer, "ray_" + field).numpy(),
            "original FP64 image rays " + field,
        )
    proof.same(trace["fp64_ray_valid"], renderer.ray_valid.numpy(), "original image-ray validity")
    rays = SimpleNamespace(
        origins=tuple(packet[v + "_ray_origins"] for v in rig.views),
        directions=tuple(packet[v + "_ray_directions"] for v in rig.views),
    )
    assigned = assigned_rays(rays, packet["proposal_plan"])
    for name, values in zip(("origins", "directions", "held"), assigned, strict=True):
        proof.same(
            trace["fp64_inputs_" + name],
            values.astype(np.float64),
            "original FP32 observation cache -> FP64 FitInputs " + name,
        )
    return dict(original_FP64_image_rays_exact=True, original_assigned_rays_FP64_exact=True)


def validate_statistics(statistics, index):
    require(statistics["schema"] == "mixed_precision_fitter_v1", "mixed statistics schema")
    require(
        statistics["bank_size"] == 16 and statistics["FP64_objective_evaluations"] == 17,
        "full bank evaluation counts",
    )
    require(
        statistics["FP32_evaluations_per_replay"] == 15
        and statistics["FP32_updates_per_replay"] == 14,
        "FP32 full trajectory counts",
    )
    require(
        statistics["scientific_dtype"] == "float64"
        and statistics["equivalent_to_FP64_search"] is False,
        "separate numerical branch",
    )
    for key in (
        "prepare_seconds",
        "generate_seconds",
        "rescore_seconds",
        "diagnostic_copy_seconds",
    ):
        require(np.isfinite(statistics[key]) and statistics[key] >= 0, "invalid timing " + key)
    require(type(index) is int and 0 <= index < 4, "fixed four Mixed cache transitions")
    captures = (1, 2, 3, 3)[index]
    require(
        statistics["cache"]
        == dict(
            capacity=4,
            captures=captures,
            hits=index + 1 - captures,
            misses=captures,
            evictions=0,
            failures=0,
            resident=captures,
            peak_resident=captures,
        ),
        "fixed FP32 LRU4 cache transition differs",
    )


def _snapshot(evidence, output, saved, index):
    previous.snapshot(evidence, output, saved, index)
    path = Path(__file__).resolve()
    digest = evidence.add(path)
    if path not in saved:
        raw = path.read_bytes()
        require(proof.hashlib.sha256(raw).hexdigest() == digest, "caller changed during snapshot")
        output.write("code_snapshot/" + path.relative_to(REPO).as_posix(), raw)
        saved.add(path)


def run(args):
    require(args.execute and args.device == "cuda", "explicit CUDA execution required")
    require(all(proof.valid_sha(s) for s in SOURCE_PINS.values()), "mixed source pins pending")
    destination = Path(args.output).resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(),
        "new workspace output required",
    )
    output, evidence, saved = proof.Output(destination), proof.Evidence(), set()
    started, begin = proof.utc(), time.perf_counter()
    rows, completed_calls, gpu_used, gpu_attempted = [], 0, False, False
    try:
        evidence.add(Path(__file__).resolve())
        for relative, digest in SOURCE_PINS.items():
            evidence.add(REPO / relative, digest)
        mixed = importlib.import_module("mixed_precision_fitter")
        importlib.import_module("fp32_pose_proposals")
        output.write("plan.json", fixed_plan())
        _snapshot(evidence, output, saved, 0)
        cfg, rig, packets, references, metadata = previous.load_inputs(evidence)
        require(
            tuple(packets) == FRAMES
            and tuple(len(packets[n]["proposal_pose_xyz"]) for n in FRAMES) == BATCHES,
            "fixed four packet order/batches",
        )
        require(not torch.cuda.is_initialized(), "no CUDA before qualified input loading")
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
        cv2.setNumThreads(metadata["opencv_threads"])
        torch.manual_seed(0)
        versions = dict(
            numpy=np.__version__,
            scipy=scipy.__version__,
            opencv=cv2.__version__,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
        )
        require(versions == metadata["runtime"], "qualified package versions changed")
        telemetry = baseline.gpu_sample()
        output.write("startup_gpu.json", telemetry)
        require(telemetry["free_MiB"] >= 8192, "less than8GiB startup headroom")
        gpu_attempted = True
        torch.cuda.set_per_process_memory_fraction(0.375, 0)
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
                opencv_threads=cv2.getNumThreads(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            ),
        )
        fitters = dict(
            fast=FastDirectionalDeviceFitter(
                rig, packets[FRAMES[0]]["initial_pose_xyz"].copy(), cfg
            ),
            mixed=mixed.MixedPrecisionDirectionalDeviceFitter(
                rig, packets[FRAMES[0]]["initial_pose_xyz"].copy(), cfg
            ),
        )
        for index, native in enumerate(FRAMES):
            packet = packets[native]
            initial_signature = baseline.signature(packet)
            pair, seconds = {}, {}
            for arm in registered_order(index):
                prefix = f"f{native}_{arm}"
                telemetry = baseline.gpu_sample()
                output.write(prefix + "_gpu_before.json", telemetry)
                require(telemetry["free_MiB"] >= 2048, "less than2GiB per-fit headroom")
                fitter = fitters[arm]
                proof.same(fitter.lengths, packet["initial_lengths"], "original seed lengths")
                arrays, elapsed = frozen.fitted_frame(fitter, packet)
                completed_calls += 1
                output.write(prefix + "_fit.npz", arrays)
                pair[arm], seconds[arm] = arrays, elapsed
                if arm == "mixed":
                    output.write(prefix + "_trace.npz", fitter.last_mixed_trace)
                    output.write(prefix + "_statistics.json", fitter.last_mixed_statistics)
                    validate_statistics(fitter.last_mixed_statistics, index)
                    validate_numerical_outputs(arrays)
                    bank_check = validate_trace(packet, arrays, fitter.last_mixed_trace)
                    output.write(prefix + "_bank_check.json", bank_check)
                    output.write(
                        prefix + "_original_rays_check.json",
                        validate_fp64_rays(packet, fitter.last_mixed_trace, rig, cfg),
                    )
                else:
                    check = previous.compare_all77(arrays, references[native])
                    output.write(prefix + "_control_comparison.json", check)
                    require(check["passed"], "Fast all77 CUDA byteexact control failed")
                require(
                    initial_signature == baseline.signature(packet), "fit changed original packet"
                )
                require(
                    torch.cuda.max_memory_reserved(0) <= 6 * 1024**3,
                    "fixed6GiB peak reserved limit",
                )
                output.write(
                    prefix + "_timing.json",
                    dict(
                        fit_seconds=elapsed,
                        completed_fit_calls=completed_calls,
                        allocated_bytes=torch.cuda.memory_allocated(0),
                        reserved_bytes=torch.cuda.memory_reserved(0),
                    ),
                )
            comparisons = {"new_Fast": mixed_difference(pair["mixed"], pair["fast"])}
            comparisons.update(
                {
                    label: mixed_difference(pair["mixed"], ref)
                    for label, ref in references[native].items()
                }
            )
            output.write(f"f{native}_mixed_differences.json", comparisons)
            validate_public_schema(pair["mixed"], pair["fast"])
            for reference in references[native].values():
                validate_numerical_outputs(reference)
            row = dict(
                frame_idx=native,
                order=list(registered_order(index)),
                batch=len(packet["proposal_pose_xyz"]),
                fit_seconds=seconds,
                subset2=bank_check,
                Fast_control_passed=True,
                Mixed_invariants_passed=True,
                all77_differences_saved=True,
            )
            output.write(f"f{native}_summary.json", row)
            rows.append(row)
            print(
                proof.json.dumps(
                    dict(frame_idx=native, completed_pairs=len(rows), fit_seconds=seconds)
                ),
                flush=True,
            )
        require(
            completed_calls == 8 and all(f.initialize_count == 0 for f in fitters.values()),
            "exact8 fit calls/no temporal initialize",
        )
        output.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                passed=True,
                frames=list(FRAMES),
                completed_pairs=4,
                completed_fit_calls=8,
                Fast_all77_CUDA_byte_exact=True,
                Mixed_equivalent_to_FP64_search=False,
                Mixed_invariants_passed=True,
                comparisons=rows,
                GPU_used=gpu_used,
                GPU_initialization_attempted=gpu_attempted,
                raw_decode=0,
                models=0,
                initialize_calls=0,
                advance_calls=0,
                quality_or_speed_adoption_claimed=False,
                actual301_requalified=False,
                total_fit_seconds={
                    a: sum(r["fit_seconds"][a] for r in rows) for a in ("fast", "mixed")
                },
                elapsed_before_guards_seconds=time.perf_counter() - begin,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                final_mixed_statistics=fitters["mixed"].last_mixed_statistics,
                FP32_graph_constructions=3,
                FP32_warmup_full_fits=9,
                FP32_capture_full_fits=3,
                FP32_replay_full_fits=4,
            ),
        )
        _snapshot(evidence, output, saved, 1)
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
            completed_fit_calls=completed_calls,
            GPU_used=gpu_used,
            GPU_initialization_attempted=gpu_attempted,
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

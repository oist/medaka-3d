"""Bounded synthetic float64 CUDA eager/capture parity; no experiment data or fit.

One B2/P64 graph serves three deterministic synthetic cases. This qualifies
scores, five objective parts, and pose gradients; it is not a speed benchmark
or a scientific claim about fish tracking. No Adam optimizer is constructed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import compare_temporal_point64_v2 as proof
import fixed_body_width_graph_fit as capture
import fixed_body_width_renderer as fixed
import numpy as np
import torch
from graph_fit import FitInputs, objective
from renderer import RendererConfig

from medaka3d.calib import Rig
from medaka3d.types import VIEWS

REPO, WORK = proof.REPO, proof.WORK
STAGE = "fixed_body_width_cuda_qualification_v1"
ATOL, RTOL = 1e-10, 1e-10
SOURCE_PINS = {
    "work/three_body/fixed_body_width_renderer.py": (
        "aca0c16647d28aca957eed2ae24092845219b65c68c118a5a246028ef0cf6563"
    ),
    "work/three_body/fixed_body_width_graph_fit.py": (
        "d5e7c1273cbffd32beab2c579b56f11ee3d47835b10c6c8edfefa6eb0def29f8"
    ),
    "labs/20260909_three_body/renderer.py": (
        "dfb585e9d51214a2a3c9447d4bc81da8af80d3451e8fc3ef7a44b1a841732e1c"
    ),
    "labs/20260909_three_body/graph_fit.py": (
        "807151e664ed8658f291c4d49eaff69ae71feff0f9a1f9cc9fca00292845edc0"
    ),
    "labs/20260909_three_body/graph_union.py": (
        "43fcadf2e684da6fdf9d161800035d7468914217a8ddde10be3dbb810689bc21"
    ),
    "labs/20260909_three_body/fast_union.py": (
        "248e5af87927ab5f4221efe3b2db9ada6fa623466a3b9d99708c226c1f4a7236"
    ),
    "src/medaka3d/calib.py": ("60849004cb53febe69f437e7afa8b18cb2b65213ba76ce7c921f7022d2be4274"),
    "src/medaka3d/types.py": ("71b002e8dbfe7efb139520bc387c9cc0aa097ce55efb24aadd1a6080da71e33a"),
    "work/three_body/compare_temporal_point64_v2.py": (
        "2461f383d7c69223c98cb9371058373361a7fb30a409a37d617b60fc2db53790"
    ),
}
require = proof.require


class AxisCamera:
    """Explicit orthographic half-rays for a synthetic scene, not calibration."""

    def __init__(self, view, axis):
        self.view, self.axis = view, axis

    def rays(self, xy):
        origins = np.zeros((len(xy), 3), np.float64)
        origins[:, [a for a in range(3) if a != self.axis]] = xy
        directions = np.zeros_like(origins)
        directions[:, self.axis] = 1
        return origins, directions


def synthetic_case(index, device):
    require(type(index) is int and 0 <= index < 3, "three fixed synthetic cases")
    reference = np.array([[20.0, 20.0, 25.0], [35.0, 30.0, 35.0], [25.0, 40.0, 45.0]])[:, None]
    reference = reference + np.array([[5.0, 0.2, 0.4], [0.0, 0.0, 0.0], [-12.0, 2.0, 1.0]])[None]
    pose = np.stack([reference.copy(), reference + np.array([0.2, -0.1, 0.05])])
    if index:
        pose = (pose - pose[:, :, 1:2]) * 1.2 + pose[:, :, 1:2]
        pose[1, 2, 1] += [0.4, -0.3, 0.2]
    if index == 2:
        pose += [-0.15, 0.2, -0.05]
    grid = np.stack(np.meshgrid(np.linspace(8, 43, 8), np.linspace(8, 43, 8)), -1).reshape(64, 2)
    pixels = {v: grid.copy() + ([0.35, -0.2] if index == 2 else [0.0, 0.0]) for v in VIEWS}
    rig = Rig(
        tuple(AxisCamera(v, a) for v, a in zip(VIEWS, (2, 0, 1), strict=True)),
        Path("synthetic-not-a-calibration-directory"),
    )
    renderer = fixed.FixedWidthRenderer(
        rig,
        pixels,
        RendererConfig(width_ratio=0.06),
        device=device,
        dtype=torch.float64,
        reference_pose_xyz=reference,
    )

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64, device=device)

    cfg = SimpleNamespace(
        width_ratio=0.06,
        ray_sigma_mm=0.8,
        missing_cost=4.0,
        shape_weight=1.3,
        motion_weight=0.7,
        repulsion_weight=0.4,
        mask_weight=2.0,
    )
    origins = np.broadcast_to(reference[None, :, None], (2, 3, 3, 3, 3)).copy()
    directions = np.zeros_like(origins)
    for view, axis in enumerate((2, 0, 1)):
        origins[:, :, view, :, axis] = 0
        directions[:, :, view, :, axis] = 1
    if index == 2:
        origins[..., 0] += 0.1
    held = np.ones((2, 3, 3, 3))
    held[:, 1, 0, 2] = 0
    held[:, 2, 1, 0] = 0
    lengths = np.linalg.norm(np.diff(reference, axis=1), axis=-1)[None]
    target = ((np.arange(64)[None] + np.arange(3)[:, None] + index) % 3 != 0).astype(float)
    inputs = FitInputs(
        tensor(origins),
        tensor(directions),
        tensor(held),
        tensor(pose - [0.1, 0.2, -0.1]),
        tensor(np.full((2, 3), 2.0)),
        tensor(lengths),
        tensor(np.maximum(0.2 * lengths, 1)),
        tensor([1.0, 1.0, 0.35])[None, None, None],
        tensor(target),
        tensor(np.full((3, 64), 1 / 64)) * renderer.ray_valid,
    )
    return cfg, tensor(pose).clone().requires_grad_(True), inputs, renderer


def arrays_of(pose, inputs, renderer):
    values = {"pose": pose, **{f.name: getattr(inputs, f.name) for f in dataclasses.fields(inputs)}}
    values.update(
        ray_origins=renderer.ray_origins,
        ray_directions=renderer.ray_directions,
        ray_valid=renderer.ray_valid,
        radial_sigma_mm=renderer.radial_sigma_mm,
    )
    return {name: value.detach().cpu().numpy().copy() for name, value in values.items()}


def compare(expected, actual):
    require(
        expected.dtype == actual.dtype == np.float64 and expected.shape == actual.shape,
        "float64 comparison shape/dtype differs",
    )
    require(np.isfinite(expected).all() and np.isfinite(actual).all(), "nonfinite numerical result")
    delta = actual - expected
    allowance = ATOL + RTOL * np.abs(expected)
    return dict(
        passed=bool((np.abs(delta) <= allowance).all()),
        byte_exact=expected.tobytes() == actual.tobytes(),
        max_absolute_difference=float(np.abs(delta).max(initial=0)),
        max_fraction_of_allowed_error=float((np.abs(delta) / allowance).max(initial=0)),
        atol=ATOL,
        rtol=RTOL,
    ), delta


def snapshot(evidence, output, captured, index):
    sources, inventory = proof.source_inventory(dict(sys.modules), [Path(__file__).resolve()])
    output.write(f"source_inventory_{index}.json", inventory)
    for path in sorted(sources):
        digest = evidence.add(path)
        if path not in captured:
            raw = path.read_bytes()
            require(proof.hashlib.sha256(raw).hexdigest() == digest, "source snapshot drift")
            output.write("code_snapshot/" + path.relative_to(REPO).as_posix(), raw)
            captured.add(path)


def run(args):
    require(args.execute is True and args.device == "cuda", "explicit CUDA qualification required")
    destination = Path(args.output).resolve()
    require(destination.is_relative_to(WORK) and not destination.exists(), "new workspace output")
    output, evidence, captured = proof.Output(destination), proof.Evidence(), set()
    started, start, rows = proof.utc(), time.perf_counter(), []
    try:
        evidence.add(Path(__file__).resolve())
        for relative, digest in SOURCE_PINS.items():
            evidence.add(REPO / relative, digest)
        snapshot(evidence, output, captured, 0)
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                cases=3,
                batch=2,
                pixels_per_view=64,
                views=list(VIEWS),
                dtype="float64",
                atol=ATOL,
                rtol=RTOL,
                actual_saved_data=False,
                case_names=["initial", "deformed", "deformed_changed_rays_and_inputs"],
                graph_constructions=1,
                graph_configurations=2,
                replay_comparisons=3,
                graph_warmup_operations=3,
                graph_capture_operations=1,
                eager_comparisons=3,
                optimizers=0,
                models=0,
                minimum_free_GiB=2,
                allocator_fraction=0.25,
                same_reference_all_cases=True,
                fixed_radial_only=True,
                dynamic_axial_preserved=True,
            ),
        )
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        require(torch.cuda.is_available(), "CUDA unavailable")
        torch.cuda.set_per_process_memory_fraction(0.25, 0)
        free, total = torch.cuda.mem_get_info(0)
        require(free >= 2 * 1024**3, "synthetic qualification requires 2GiB free")
        environment = dict(
            torch=torch.__version__,
            numpy=np.__version__,
            cuda=torch.version.cuda,
            GPU=torch.cuda.get_device_name(0),
            free_bytes=int(free),
            total_bytes=int(total),
            torch_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        )
        output.write("environment.json", environment)
        graph, initial_signature = None, None
        for index in range(3):
            cfg, pose, inputs, renderer = synthetic_case(index, "cuda:0")
            before = arrays_of(pose, inputs, renderer)
            output.write(f"case_{index}/inputs.npz", before)
            reference = renderer.validate_fixed_width()
            if initial_signature is None:
                initial_signature = reference
            require(reference == initial_signature, "qualification changed width reference")
            denominator = 9 * float(inputs.node_weight.sum())
            torch.cuda.synchronize(0)
            eager_start = time.perf_counter()
            scores, parts, _ = objective(pose, cfg, inputs, denominator, renderer)
            gradient = torch.autograd.grad(scores.sum(), pose)[0]
            torch.cuda.synchronize(0)
            eager_seconds = time.perf_counter() - eager_start
            configure_start = time.perf_counter()
            if graph is None:
                graph = capture.FixedWidthGraphFit(cfg, pose, inputs, renderer, warmup=3)
            else:
                graph.configure(inputs, renderer)
            torch.cuda.synchronize(0)
            configure_seconds = time.perf_counter() - configure_start
            replay_start = time.perf_counter()
            result = graph.evaluate(pose)
            torch.cuda.synchronize(0)
            replay_seconds = time.perf_counter() - replay_start
            expected = dict(scores=scores, parts=parts, gradient=gradient)
            actual = dict(scores=result.scores, parts=result.parts, gradient=result.pose_gradient)
            saved, comparisons = {}, {}
            for name in expected:
                a, b = (
                    value.detach().cpu().numpy().copy() for value in (expected[name], actual[name])
                )
                comparisons[name], delta = compare(a, b)
                saved.update({"eager_" + name: a, "capture_" + name: b, "delta_" + name: delta})
            after = arrays_of(pose, inputs, renderer)
            require(
                all(
                    fixed.array_signature(before[k]) == fixed.array_signature(after[k])
                    for k in before
                ),
                "public qualification input mutated",
            )
            require(graph._renderer.validate_fixed_width() == reference, "capture width changed")
            output.write(f"case_{index}/comparisons.npz", saved)
            row = dict(
                case=index,
                passed=all(r["passed"] for r in comparisons.values()),
                comparisons=comparisons,
                fixed_width_signature=reference,
                denominator=denominator,
                config=vars(cfg),
                input_unchanged=True,
                eager_forward_backward_seconds=eager_seconds,
                graph_construct_or_configure_seconds=configure_seconds,
                graph_replay_and_copy_seconds=replay_seconds,
            )
            output.write(f"case_{index}/summary.json", row)
            rows.append(row)
            require(row["passed"], "fixed-width CUDA eager/capture numerical mismatch")
        summary = dict(
            schema=STAGE,
            status="ok",
            passed=True,
            cases=rows,
            dtype="float64",
            batch=2,
            views=3,
            pixels_per_view=64,
            tolerance=dict(atol=ATOL, rtol=RTOL),
            graph_constructions=1,
            graph_configurations=2,
            graph_replay_comparisons=3,
            eager_comparisons=3,
            actual_saved_data=False,
            optimizers=0,
            models=0,
            full_tracking_qualification=False,
            GPU_used=True,
            environment=environment,
            elapsed_before_guards_seconds=time.perf_counter() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
        )
        output.write("summary.json", summary)
        snapshot(evidence, output, captured, 1)
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
            completed_cases=len(rows),
            error_type=type(error).__name__,
            error=str(error),
        )
        with (destination / "qualification.provenance.json").open("xb") as stream:
            stream.write(proof.json_bytes(seal))
        raise
    with (destination / "qualification.provenance.json").open("xb") as stream:
        stream.write(proof.json_bytes(seal))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--device", choices=["cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args())), flush=True)


if __name__ == "__main__":
    main()

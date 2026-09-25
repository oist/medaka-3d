"""One synthetic B2 float64 body-point fit versus eager scalar gradient oracles.

The production fitter runs one ordinary14-step Adam trajectory. Its base alone
uses the frozen GraphFit; every visited pose is independently evaluated without
GraphFit, on the same public old-pixel and query renderers. This checks numerical
assembly, not an independent optimizer trajectory, speed, fish data or identity.
No test module, model, real calibration, video, checkpoint or tracking is loaded.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import body_point_fitter as fitter
import compare_temporal_point64_v2 as proof
import numpy as np
import torch
from graph_fit import FitInputs, objective
from node_geometry import RayObservations, assigned_rays
from observations import PixelTargets
from renderer import RefractiveFishRenderer
from tracker import Proposal, TrackerConfig

from medaka3d.calib import Rig
from medaka3d.types import VIEWS

REPO, WORK = proof.REPO, proof.WORK
STAGE = "body_point_cuda_qualification_v1"
ATOL = RTOL = 1e-10
require = proof.require
SOURCE_PINS = {
    "work/three_body/body_point_fitter.py": (
        "e7196c49f081a0e86c476ad4d49a34ad6cd638d0c0a8149d46c004ce275a68ac"
    ),
    "work/three_body/mask_transport_fitter.py": (
        "51554ea7da2e1d35ed1ca3ade32b76a92894bb4acfe552f47493853f8c7fc232"
    ),
    "work/three_body/mask_transport_gap_fitter.py": (
        "930ca3c88feb9cc709a62001d14c9515cf196d0d6a1917cec656beacec9c5c75"
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
    "labs/20260909_three_body/tracker.py": (
        "f481e2d4105a0223cf24719482e1d057ba4e68094ff9df74af0a826a88d31c20"
    ),
    "labs/20260909_three_body/node_geometry.py": (
        "c7bea7333d5a20704e67c2283e8dc925a70e2f05c4520353d74b18c64b5aff69"
    ),
    "labs/20260909_three_body/observations.py": (
        "f1efcd27b31e63ce693229c9e765152acfc7984b3b8839f1792bcc2efd9cb5df"
    ),
    "src/medaka3d/calib.py": "60849004cb53febe69f437e7afa8b18cb2b65213ba76ce7c921f7022d2be4274",
    "src/medaka3d/types.py": "71b002e8dbfe7efb139520bc387c9cc0aa097ce55efb24aadd1a6080da71e33a",
    "work/three_body/compare_temporal_point64_v2.py": (
        "2461f383d7c69223c98cb9371058373361a7fb30a409a37d617b60fc2db53790"
    ),
}


class AxisCamera:
    def __init__(self, view, axis):
        self.view, self.axis = view, axis

    def rays(self, xy):
        origin = np.zeros((len(xy), 3), np.float64)
        origin[:, [a for a in range(3) if a != self.axis]] = xy
        direction = np.zeros_like(origin)
        direction[:, self.axis] = 1
        return origin, direction


def synthetic_fixture(device="cpu"):
    """Local deterministic geometry matching the independent CPU27 fixture scope."""
    device = torch.device(device)
    require(str(device) in ("cpu", "cuda:0"), "explicit CPU or registered CUDA0 required")
    reference = np.array(
        [
            [[6, 1, 10], [1, 1.5, 9.8], [-11, 3.5, 11]],
            [[15, 12, 16], [9, 10, 16], [0, 9, 16.5]],
            [[-6, -11, 24], [-9, -13, 24], [-18, -14, 25]],
        ],
        np.float64,
    )
    rig = Rig(
        tuple(AxisCamera(v, a) for v, a in zip(VIEWS, (2, 0, 1), strict=True)),
        Path("synthetic-only"),
    )
    pixels, points, origins, directions = {}, {}, [], []
    for view, axis in zip(VIEWS, (2, 0, 1), strict=True):
        other = [a for a in range(3) if a != axis]
        points[view] = reference[..., other].copy()
        pixels[view] = np.vstack((reference[0, [1, 0, 2]][:, other], [35, 35]))
        origin, direction = reference.copy(), np.zeros_like(reference)
        origin[..., axis], direction[..., axis] = 0, 1
        origins.append(origin)
        directions.append(direction)
    foreground = np.broadcast_to([1.0, 1.0, 1.0, 0.0], (3, 4)).copy()
    targets = PixelTargets(pixels, foreground, np.full((3, 4), 0.25), foreground.astype(bool))
    config = TrackerConfig(device=str(device), learning_rate_mm=0.02)
    tracker = SimpleNamespace(
        rig=rig,
        config=config,
        device=device,
        dtype=torch.float64,
        initial=reference.copy(),
        lengths=np.linalg.norm(np.diff(reference, axis=1), axis=-1),
        lower=np.full((3, 3, 3), -100.0),
        upper=np.full((3, 3, 3), 100.0),
        _tensor=lambda v: torch.as_tensor(v, dtype=torch.float64, device=device),
    )
    poses = [
        (reference - reference[:, 1:2]) * 1.35 + reference[:, 1:2],
        (reference - reference[:, 1:2]) * 1.1 + reference[:, 1:2] + np.array([0.2, -0.1, 0.05]),
    ]
    plan = np.broadcast_to(np.arange(3, dtype=np.int32)[:, None, None], (3, 3, 3)).copy()
    proposals = [
        Proposal(p.copy(), reference.copy(), plan.copy(), parent, "synthetic")
        for p, parent in zip(poses, (1, 0), strict=True)
    ]
    states = [SimpleNamespace(missing=np.zeros(3)), SimpleNamespace(missing=np.array([1, 2, 3]))]
    rays = RayObservations(points, tuple(origins), tuple(directions))
    counts = np.array([3, 2, 1], np.int64)
    query = np.full((3, 3, 2), np.nan)
    for vi, count in enumerate(counts):
        query[vi, :count] = pixels[VIEWS[vi]][:count] + np.array([2.2, -0.8])
    query[0, 1] = np.nan
    # The packet's ray evidence is produced by the same CPU calibration path as
    # production. Its float64 bytes are checked against the GPU renderer in fit.
    ray_o, ray_d = np.zeros((3, 3, 3)), np.zeros((3, 3, 3))
    ray_d[..., 2] = 1
    valid = np.zeros((3, 3), bool)
    for vi, view in enumerate(VIEWS):
        selected = np.flatnonzero(np.isfinite(query[vi]).all(-1))
        ray_o[vi, selected], ray_d[vi, selected] = rig.camera(view).rays(query[vi, selected])
        valid[vi, selected] = True
    weights = np.zeros((3, 3, 3, 3))
    weights[0, :, 0, 1] = [0.3, 0.15, 0.075]
    weights[1, :, 0, 2] = 0.02
    weights[2, :, 0, 0] = [0.15, 0.075, 0.3]
    weights[2, 1, 1, 2] = 0.05
    packet = SimpleNamespace(
        point_pixels=query,
        point_weight=weights,
        ray_origins=ray_o,
        ray_directions=ray_d,
        ray_valid=valid,
        query_counts=counts,
        anchor_ids=np.arange(3, dtype=np.int64),
    )
    return (tracker, proposals, states, rays, targets), packet, np.array([2, 0], np.int64)


def fixture_arrays(fixture):
    (tracker, proposals, states, rays, targets), packet, anchors = fixture
    return {
        "initial": tracker.initial.copy(),
        "lengths": tracker.lengths.copy(),
        "lower": tracker.lower.copy(),
        "upper": tracker.upper.copy(),
        "proposal_pose": np.stack([p.pose for p in proposals]),
        "proposal_predicted": np.stack([p.predicted for p in proposals]),
        "proposal_plan": np.stack([p.plan for p in proposals]),
        "proposal_parent": np.asarray([p.parent for p in proposals], np.int64),
        "parent_missing": np.stack([s.missing for s in states]),
        "SL_points": np.stack([rays.points[v] for v in VIEWS]),
        "SL_origins": np.stack(rays.origins),
        "SL_directions": np.stack(rays.directions),
        "base_pixels": np.stack([targets.pixels[v] for v in VIEWS]),
        "base_target": targets.target.copy(),
        "base_weights": targets.weights.copy(),
        "candidate_anchor": anchors.copy(),
        **{k: v.copy() for k, v in vars(packet).items()},
    }


def cpu_inputs(fixture, renderer):
    """CPU specification only; CUDA oracle receives actual GraphFit inputs."""
    (tracker, proposals, states, rays, targets), _, _ = fixture
    t = tracker._tensor
    o, d, held = assigned_rays(rays, np.stack([p.plan for p in proposals]))
    missing = t(np.stack([states[p.parent].missing for p in proposals]))
    lengths = t(tracker.lengths)[None]
    return FitInputs(
        t(o),
        t(d),
        t(held),
        t(np.stack([p.predicted for p in proposals])),
        tracker.config.motion_sigma_mm + 0.3 * missing.clamp(max=20),
        lengths,
        (0.2 * lengths).clamp(min=1),
        t([1.0, 1.0, 0.35])[None, None, None],
        t(targets.target),
        t(targets.weights) * renderer.ray_valid,
    )


def oracle(pose, config, inputs, base_renderer, query_renderer, packet, anchors, *, final=False):
    """Independent scalar body-point assembly; never calls the production loss."""
    value = pose.detach().clone().requires_grad_(True)
    denominator = 9 * float(inputs.node_weight.sum())
    base, parts, rendering = objective(
        value, config, inputs, denominator, base_renderer, diagnostics=final
    )
    point = query_renderer.render(value)
    owner_values, gap_values = [], []
    weights = packet.point_weight[anchors]
    for b in range(len(anchors)):
        owner, gap = value[b].sum() * 0, value[b].sum() * 0
        for vi, qi, body in np.argwhere(weights[b] > 0):
            if packet.ray_valid[vi, qi]:
                w = float(weights[b, vi, qi, body]) / 3
                owner = owner + w * (1 - point.ownership[b, vi, qi, body]) ** 2
                gap = gap + w * (
                    point.per_fish_mahalanobis_sq[b, vi, qi, body].clamp_min(1).sqrt() - 1
                )
        owner_values.append(owner)
        gap_values.append(gap)
    owner, gap = torch.stack(owner_values), torch.stack(gap_values)
    total = base + 3 * (owner + gap)
    gradients = {
        name: torch.autograd.grad(values.sum(), value, retain_graph=True)[0]
        for name, values in (("base", base), ("owner", owner), ("gap", gap), ("total", total))
    }
    result = dict(
        pose=value,
        base=base,
        parts=parts,
        owner=owner,
        gap=gap,
        total=total,
        **{name + "_gradient": gradient for name, gradient in gradients.items()},
    )
    if final:
        w, fg = inputs.weight, inputs.target
        pos = (
            rendering.visible_alpha * (rendering.ownership >= 0.65) * (w * fg)[None, :, :, None]
        ).sum(2)
        neg = (rendering.visible_alpha * (w * (1 - fg))[None, :, :, None]).sum(2)
        result["support"] = ((pos >= 0.025) & (pos >= 2 * neg)).sum(1) >= 2
    return {k: v.detach().cpu().numpy().copy() for k, v in result.items()}


def comparison(expected, actual):
    schema = expected.dtype == actual.dtype and expected.shape == actual.shape
    finite = schema and np.isfinite(expected).all() and np.isfinite(actual).all()
    floating = expected.dtype.kind == "f"
    numeric = bool(finite and (not floating or expected.dtype == np.float64))
    exact = bool(schema and expected.tobytes() == actual.tobytes())
    error = np.abs(actual - expected) if numeric and floating else None
    return dict(
        passed=bool(
            numeric and ((error <= ATOL + RTOL * np.abs(expected)).all() if floating else exact)
        ),
        schema_equal=bool(schema),
        finite=bool(finite),
        byte_exact=exact,
        max_absolute_difference=None if error is None else float(error.max(initial=0)),
        atol=ATOL if floating else 0,
        rtol=RTOL if floating else 0,
    )


def save_comparison(output, prefix, expected, actual):
    require(set(expected) == set(actual), "comparison field membership differs")
    arrays = {
        side + "_" + k: v
        for side, values in (("expected", expected), ("actual", actual))
        for k, v in values.items()
    }
    output.write(prefix + ".npz", arrays)
    records = {k: comparison(expected[k], actual[k]) for k in expected}
    report = dict(passed=all(r["passed"] for r in records.values()), comparisons=records)
    output.write(prefix + ".json", report)
    require(report["passed"], "synthetic numerical mismatch: " + prefix)
    return report


def check_fixture_cpu():
    """One eager CPU forward/backward only; no fit, optimizer, graph or CUDA."""
    fixture = synthetic_fixture()
    args, packet, anchors = fixture
    tracker = args[0]
    base = RefractiveFishRenderer(tracker.rig, args[4].pixels, dtype=torch.float64)
    query = RefractiveFishRenderer(
        tracker.rig,
        {v: packet.point_pixels[i, : packet.query_counts[i]] for i, v in enumerate(VIEWS)},
        dtype=torch.float64,
    )
    for name in ("ray_origins", "ray_directions", "ray_valid"):
        require(
            np.array_equal(getattr(packet, name), getattr(query, name).numpy()),
            "synthetic packet rays differ",
        )
    inputs = cpu_inputs(fixture, base)
    row = oracle(
        tracker._tensor(np.stack([p.pose for p in args[1]])),
        tracker.config,
        inputs,
        base,
        query,
        packet,
        anchors,
    )
    require(all(np.isfinite(a).all() for a in row.values()), "nonfinite synthetic CPU oracle")
    require(
        np.any(row["owner_gradient"]) and np.any(row["gap_gradient"]),
        "both point gradients must be exercised",
    )
    return dict(
        batch=2,
        base_pixels_per_view=4,
        query_padding=3,
        query_counts=packet.query_counts.tolist(),
        query_ray_valid_counts=packet.ray_valid.sum(1).tolist(),
        candidate_anchor=anchors.tolist(),
        proposal_parent=[p.parent for p in args[1]],
        coefficient=3.0,
        opt_steps=tracker.config.opt_steps,
        dtype="float64",
        finite=True,
        nonzero_owner_and_gap_gradients=True,
        GPU_used=False,
        fit_calls=0,
        graph_calls=0,
        optimizer_steps=0,
    )


class Inspection:
    def __init__(self, fixture, output):
        self.fixture, self.output = fixture, output
        self.renderers, self.rows, self.steps, self.graphs = [], [], [], []
        self.graph_attempts = self.graph_returns = self.optimizer_attempts = 0

    def execute(self):
        inspection = self
        args, packet, anchors = self.fixture
        original = fitter.GraphFit, fitter.RefractiveFishRenderer, torch.optim.Adam

        def renderer(*values, **options):
            result = original[1](*values, **options)
            inspection.renderers.append(result)
            return result

        class InspectedGraph(original[0]):
            def __init__(self, cfg, pose, inputs, public_renderer=None, **kwargs):
                super().__init__(cfg, pose, inputs, public_renderer, **kwargs)
                self.public_inputs, self.public_renderer = inputs, public_renderer
                inspection.graphs.append(self)

            def evaluate(self, pose, copy_outputs=True):
                inspection.graph_attempts += 1
                actual = super().evaluate(pose, copy_outputs=copy_outputs)
                inspection.graph_returns += 1
                require(
                    len(inspection.renderers) == 2,
                    "exactly base and query public renderers required",
                )
                expected = oracle(
                    pose,
                    self.config,
                    self.public_inputs,
                    self.public_renderer,
                    inspection.renderers[1],
                    packet,
                    anchors,
                )
                index = len(inspection.rows)
                require(index < 15, "extra model iterate")
                inspection.output.write(f"iterate_{index:02d}/oracle.npz", expected)
                report = save_comparison(
                    inspection.output,
                    f"iterate_{index:02d}/base_graph",
                    {
                        "base": expected["base"],
                        "parts": expected["parts"],
                        "gradient": expected["base_gradient"],
                    },
                    {
                        "base": actual.scores.cpu().numpy().copy(),
                        "parts": actual.parts.cpu().numpy().copy(),
                        "gradient": actual.pose_gradient.cpu().numpy().copy(),
                    },
                )
                inspection.rows.append((expected, report))
                return actual

        class InspectedAdam(original[2]):
            def step(self, *values, **options):
                inspection.optimizer_attempts += 1
                index = len(inspection.steps)
                require(
                    index < 14 and len(inspection.rows) == index + 1, "Adam/iterate order differs"
                )
                pose = self.param_groups[0]["params"][0]
                expected = inspection.rows[index][0]
                report = save_comparison(
                    inspection.output,
                    f"iterate_{index:02d}/adam_input",
                    {"pose": expected["pose"], "gradient": expected["total_gradient"]},
                    {
                        "pose": pose.detach().cpu().numpy().copy(),
                        "gradient": pose.grad.detach().cpu().numpy().copy(),
                    },
                )
                result = super().step(*values, **options)
                inspection.steps.append(report)
                return result

        try:
            fitter.GraphFit, fitter.RefractiveFishRenderer, torch.optim.Adam = (
                InspectedGraph,
                renderer,
                InspectedAdam,
            )
            return fitter.fit_with_body_points(*args, packet, candidate_anchor=anchors)
        finally:
            fitter.GraphFit, fitter.RefractiveFishRenderer, torch.optim.Adam = original

    def finish(self, result):
        require(
            len(self.rows) == 15 and len(self.steps) == 14 and len(self.graphs) == 1,
            "one full synthetic fit required",
        )
        args, packet, anchors = self.fixture
        d = args[0].last_gap_fit
        rows = [r[0] for r in self.rows]
        totals = np.stack([r["total"] for r in rows])
        best = np.argmin(totals, axis=0).astype(np.int64)
        expected_pose = np.stack([rows[int(best[b])]["pose"][b] for b in range(2)])
        graph = self.graphs[0]
        final = oracle(
            args[0]._tensor(expected_pose),
            args[0].config,
            graph.public_inputs,
            self.renderers[0],
            self.renderers[1],
            packet,
            anchors,
            final=True,
        )
        expected = {
            "iteration_" + k: np.stack([r[name] for r in rows])
            for k, name in (
                ("base_score", "base"),
                ("owner_loss", "owner"),
                ("gap_loss", "gap"),
                ("total_score", "total"),
            )
        }
        for name in ("owner", "gap"):
            expected[f"iteration_{name}_gradient_norm"] = np.stack(
                [np.linalg.norm(r[name + "_gradient"].reshape(2, -1), axis=1) for r in rows[:14]]
            )
        expected.update(
            best_iteration=best,
            pose=expected_pose,
            score=final["total"],
            parts=final["parts"],
            support=final["support"],
            base_score=final["base"],
            owner_loss=final["owner"],
            gap_loss=final["gap"],
        )
        actual = {k: d[k] for k in expected if k in d}
        actual.update(pose=result[0], score=result[1], parts=result[2], support=result[3])
        report = save_comparison(self.output, "completed_fit", expected, actual)
        require(
            d["active"] is True and d["reason"] == "fixed_body_point_owner_plus_gap",
            "wrong fit branch",
        )
        return report


def snapshot(evidence, output, captured, index):
    sources, inventory = proof.source_inventory(dict(sys.modules), [Path(__file__).resolve()])
    output.write(f"source_inventory_{index}.json", inventory)
    for path in sorted(sources):
        digest = evidence.add(path)
        if path not in captured:
            raw = path.read_bytes()
            require(proof.hashlib.sha256(raw).hexdigest() == digest, "snapshot changed")
            output.write("code_snapshot/" + path.relative_to(REPO).as_posix(), raw)
            captured.add(path)


def run(args):
    require(args.execute is True and args.device == "cuda", "explicit CUDA qualification required")
    destination = Path(args.output).resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(),
        "new workspace output required",
    )
    output, evidence, captured = proof.Output(destination), proof.Evidence(), set()
    started, start = proof.utc(), time.perf_counter()
    gpu_attempted, inspection = False, None
    try:
        evidence.add(Path(__file__).resolve())
        for relative, digest in SOURCE_PINS.items():
            evidence.add(REPO / relative, digest)
        snapshot(evidence, output, captured, 0)
        fixture = synthetic_fixture()
        inputs = fixture_arrays(fixture)
        execution_config = {**dataclasses.asdict(fixture[0][0].config), "device": "cuda:0"}
        output.write("inputs.npz", inputs)
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                batch=2,
                base_pixels_per_view=4,
                query_counts=[3, 2, 1],
                query_ray_valid_counts=[2, 2, 1],
                candidate_anchor=[2, 0],
                proposal_parent=[1, 0],
                dtype="float64",
                config=execution_config,
                fixture_build_device="cpu",
                execution_device="cuda:0",
                coefficient=3.0,
                evaluations=15,
                optimizer_steps=14,
                graph_constructions=1,
                graph_warmup=3,
                graph_capture=1,
                atol=ATOL,
                rtol=RTOL,
                actual_saved_data=False,
                model_calls=0,
                tracking_calls=0,
                minimum_free_GiB=2,
                allocator_fraction=0.25,
                same_query_public_renderer=True,
                comparison="same-visited-pose eager scalar oracle; not second optimizer trajectory",
                speed_benchmark=False,
            ),
        )
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        require(torch.cuda.is_available(), "CUDA unavailable")
        gpu_attempted = True
        torch.cuda.set_per_process_memory_fraction(0.25, 0)
        free, total = torch.cuda.mem_get_info(0)
        require(free >= 2 * 1024**3, "synthetic qualification requires2GiB free")
        environment = dict(
            torch=torch.__version__,
            numpy=np.__version__,
            cuda=torch.version.cuda,
            GPU=torch.cuda.get_device_name(0),
            free_bytes=int(free),
            total_bytes=int(total),
            torch_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic=torch.are_deterministic_algorithms_enabled(),
        )
        output.write("environment.json", environment)
        fixture = synthetic_fixture("cuda:0")
        require(
            dataclasses.asdict(fixture[0][0].config) == execution_config,
            "executed configuration differs from the saved plan",
        )
        require(
            all(
                proof.signature(v) == proof.signature(fixture_arrays(fixture)[k])
                for k, v in inputs.items()
            ),
            "CPU/GPU fixture arrays differ",
        )
        inspection = Inspection(fixture, output)
        result = inspection.execute()
        completed = inspection.finish(result)
        after = fixture_arrays(fixture)
        require(
            all(proof.signature(v) == proof.signature(after[k]) for k, v in inputs.items()),
            "qualification input mutated",
        )
        summary = dict(
            schema=STAGE,
            status="ok",
            passed=True,
            fitter_sha256=SOURCE_PINS["work/three_body/body_point_fitter.py"],
            batch=2,
            base_pixels_per_view=4,
            query_counts=[3, 2, 1],
            candidate_anchor=[2, 0],
            dtype="float64",
            tolerance=dict(atol=ATOL, rtol=RTOL),
            graph_constructions=1,
            evaluated_poses=15,
            graph_evaluation_attempts=inspection.graph_attempts,
            graph_evaluations_returned=inspection.graph_returns,
            optimizer_step_attempts=inspection.optimizer_attempts,
            compared_adam_gradients=14,
            optimizer_steps=14,
            completed_fit=completed,
            all_base_graph_comparisons_passed=True,
            all_combined_gradient_comparisons_passed=True,
            input_unchanged=True,
            strict_first_best_checked=True,
            final_old_pixel_support_checked=True,
            actual_saved_data=False,
            models=0,
            tracking_calls=0,
            GPU_used=True,
            independent_optimizer_trajectory=False,
            speed_benchmark=False,
            environment=environment,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            elapsed_seconds=time.perf_counter() - start,
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
            GPU_initialization_attempted=gpu_attempted,
            evaluated_poses=0 if inspection is None else inspection.graph_returns,
            checked_poses=0 if inspection is None else len(inspection.rows),
            graph_evaluation_attempts=0 if inspection is None else inspection.graph_attempts,
            optimizer_step_attempts=0 if inspection is None else inspection.optimizer_attempts,
            completed_optimizer_steps=0 if inspection is None else len(inspection.steps),
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

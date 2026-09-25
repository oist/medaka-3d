"""Experimental FP32 pose proposals selected with the original FP64 objective.

This is a new numerical search, not an equivalent execution of the FP64 fitter.
Outer proposals, assignments and temporal state stay in their original schema.
The exact FP64 input pose and all fifteen FP32 poses form a sixteen-row bank;
one pose per original proposal is returned after FP64 rescore and support.
No continuous-tracking or speed qualification is implied by this adapter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields

import numpy as np
import torch
from directional_chest_fitter import parent_policy
from fast_directional_device_fitter import FastDirectionalDeviceFitter, require_finite
from fp64_pose_bank_rescore import rescore_pose_bank
from graph_fit import FitInputs
from node_geometry import assigned_rays
from renderer import RefractiveFishRenderer, RendererConfig


@dataclass(frozen=True)
class OriginalProblem:
    pose: torch.Tensor
    inputs: FitInputs
    renderer: RefractiveFishRenderer
    axes: torch.Tensor
    parallel_sigma: torch.Tensor
    enabled: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    policy: dict
    parents: np.ndarray


def prepare_original_problem(fitter, proposals, states, rays, targets):
    """The frozen Fast input arithmetic, performed entirely in original FP64."""
    cfg, tensor = fitter.config, fitter._tensor
    if fitter.dtype != torch.float64 or cfg.opt_steps != 14 or targets is None:
        raise ValueError("original FP64/14-step/D2 inputs required")
    policy = parent_policy(proposals, states, rays, fitter.rig, cfg)
    parents = np.asarray([p.parent for p in proposals], np.int64)
    axes = tensor(policy["axes"][parents])
    enabled = torch.as_tensor(policy["enabled"][parents] & fitter.directional, device=fitter.device)
    parallel_sigma = tensor(policy["parallel_sigma"][parents])
    origins, directions, valid = assigned_rays(rays, np.stack([p.plan for p in proposals]))
    predicted = tensor(np.stack([p.predicted for p in proposals]))
    missing = tensor(np.stack([states[p.parent].missing for p in proposals]))
    sigma = cfg.motion_sigma_mm + 0.3 * torch.clamp(missing, max=20)
    lengths = tensor(fitter.lengths)[None]
    node_weight = tensor([1.0, 1.0, 0.35])[None, None, None]
    renderer = RefractiveFishRenderer(
        fitter.rig,
        targets.pixels,
        RendererConfig(width_ratio=cfg.width_ratio),
        device=predicted.device,
        dtype=fitter.dtype,
    )
    target, weight = tensor(targets.target), tensor(targets.weights) * renderer.ray_valid
    inputs = FitInputs(
        tensor(origins),
        tensor(directions),
        tensor(valid),
        predicted,
        sigma,
        lengths,
        torch.clamp(0.2 * lengths, min=1.0),
        node_weight,
        target,
        weight,
    )
    pose = tensor(np.stack([p.pose for p in proposals])).clone()
    return OriginalProblem(
        pose,
        inputs,
        renderer,
        axes,
        parallel_sigma,
        enabled,
        tensor(fitter.lower),
        tensor(fitter.upper),
        policy,
        parents,
    )


def inputs_as_fp32(inputs):
    """Round completed FP64 inputs once; never round then re-promote for rescore."""
    return FitInputs(
        **{
            f.name: None
            if getattr(inputs, f.name) is None
            else getattr(inputs, f.name).detach().to(torch.float32).clone()
            for f in fields(inputs)
        }
    )


def assemble_pose_bank(original_pose, fp32_path):
    """Preserve C0 exactly; C1..15 are X32(0..14) promoted without modification."""
    if (
        original_pose.dtype != torch.float64
        or original_pose.ndim != 4
        or original_pose.shape[1:] != (3, 3, 3)
        or len(original_pose) < 1
        or fp32_path.dtype != torch.float32
        or fp32_path.shape != (15, *original_pose.shape)
        or fp32_path.device != original_pose.device
    ):
        raise ValueError("original FP64 pose and all15 same-device FP32 poses required")
    if not bool(torch.isfinite(original_pose).all() & torch.isfinite(fp32_path).all()):
        raise FloatingPointError("nonfinite original pose or FP32 path")
    if not torch.equal(fp32_path[0], original_pose.to(torch.float32)):
        raise ValueError("FP32 path does not start at the rounded original pose")
    return torch.cat((original_pose.detach()[None], fp32_path.detach().to(torch.float64))).clone()


def two_candidate_diagnostics(bank_scores, fp32_best_iteration):
    """Diagnose initial+FP32-best subset using the already completed FP64 bank."""
    if (
        bank_scores.ndim != 2
        or bank_scores.shape[0] != 16
        or bank_scores.dtype != torch.float64
        or fp32_best_iteration.shape != (bank_scores.shape[1],)
        or fp32_best_iteration.dtype != torch.int64
        or fp32_best_iteration.device != bank_scores.device
    ):
        raise ValueError("sixteen FP64 score rows and FP32 best indices required")
    if not bool(torch.isfinite(bank_scores).all()):
        raise FloatingPointError("nonfinite bank scores")
    if not bool(((fp32_best_iteration >= 0) & (fp32_best_iteration < 15)).all()):
        raise ValueError("FP32 best iteration outside 0..14")
    proposal = torch.arange(bank_scores.shape[1], device=bank_scores.device)
    best_index = fp32_best_iteration + 1
    subset_index = torch.where(bank_scores[best_index, proposal] < bank_scores[0], best_index, 0)
    subset_score = bank_scores[subset_index, proposal]
    all_best_score = bank_scores.min(dim=0).values
    return dict(
        subset2_bank_index=subset_index.detach().clone(),
        subset2_score=subset_score.detach().clone(),
        subset2_missed_strict_improvement=(subset_score > all_best_score).detach().clone(),
        subset2_cost_regret=(subset_score - all_best_score).detach().clone(),
    )


def _numpy(value):
    return value.detach().cpu().numpy().copy()


class MixedPrecisionDirectionalDeviceFitter(FastDirectionalDeviceFitter):
    """CUDA-only experiment. Scientific dtype and returned state remain FP64."""

    def _fit(self, proposals, states, rays, targets):
        if self.device.type != "cuda":
            raise ValueError("mixed precision experimental fitter requires CUDA")
        if self.__dict__.get("_mixed_busy", False):
            raise RuntimeError("mixed fitter is serial and nonreentrant")
        self._mixed_busy = True
        self.last_mixed_trace = None
        self.last_mixed_statistics = None
        self.last_diagnostics = None
        try:
            return self._fit_mixed(proposals, states, rays, targets)
        finally:
            self._mixed_busy = False

    def _fit_mixed(self, proposals, states, rays, targets):
        from fp32_pose_proposals import FP32ProposalCache

        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        problem = prepare_original_problem(self, proposals, states, rays, targets)
        # Reconstruct from the same original CPU-calibrated rays. Native FP32
        # renderer constants are computed in FP32; original FP64 rays stay intact.
        renderer32 = RefractiveFishRenderer(
            self.rig,
            targets.pixels,
            problem.renderer.config,
            device=problem.pose.device,
            dtype=torch.float32,
        )
        if not torch.equal(renderer32.ray_valid, problem.renderer.ray_valid):
            raise ValueError("FP32 and original FP64 image-ray validity differs")
        inputs32 = inputs_as_fp32(problem.inputs)
        cache = self.__dict__.get("_fp32_proposal_cache")
        if cache is None:
            cache = FP32ProposalCache()
            self._fp32_proposal_cache = cache
        torch.cuda.synchronize(self.device)
        prepared = time.perf_counter()
        graph = cache.generate(
            self.config,
            problem.pose.to(torch.float32),
            inputs32,
            renderer32,
            problem.axes.to(torch.float32),
            problem.parallel_sigma.to(torch.float32),
            problem.enabled.clone(),
            problem.lower.to(torch.float32),
            problem.upper.to(torch.float32),
            directional=self.directional,
        )
        require_finite(graph.score_ok, graph.gradient_ok, graph.pose_ok)
        torch.cuda.synchronize(self.device)
        generated = time.perf_counter()
        bank = assemble_pose_bank(problem.pose, graph.trace["pose_before"])
        result = rescore_pose_bank(
            bank,
            config=self.config,
            inputs=problem.inputs,
            renderer=problem.renderer,
            axes=problem.axes,
            parallel_sigma=problem.parallel_sigma,
            enabled=problem.enabled,
            directional=self.directional,
        )
        subset = two_candidate_diagnostics(result["bank_scores"], graph.best_iteration)
        torch.cuda.synchronize(self.device)
        rescored = time.perf_counter()
        # Own all diagnostic copies before another generate/configure can reuse
        # a graph buffer. No FP32 score or support is returned as scientific state.
        trace = {"fp32_" + name: _numpy(value) for name, value in graph.trace.items()}
        trace.update(
            {
                "fp64_" + name: _numpy(value)
                for name, value in result.items()
                if isinstance(value, torch.Tensor)
            }
        )
        trace.update({name: _numpy(value) for name, value in subset.items()})
        trace.update(
            original_pose_xyz=_numpy(problem.pose),
            fp64_pose_bank=_numpy(bank),
            fp32_best_iteration=_numpy(graph.best_iteration),
            fp32_best_score=_numpy(graph.best_score),
            fp32_finite_flags=_numpy(
                torch.stack((graph.score_ok, graph.gradient_ok, graph.pose_ok))
            ),
            fp32_uses_positive_product_bound=np.asarray(graph.uses_positive_product_bound),
            fp64_image_ray_origins=_numpy(problem.renderer.ray_origins),
            fp64_image_ray_directions=_numpy(problem.renderer.ray_directions),
            fp64_inputs_origins=_numpy(problem.inputs.origins),
            fp64_inputs_directions=_numpy(problem.inputs.directions),
            fp64_inputs_held=_numpy(problem.inputs.held),
        )
        diagnostics = dict(
            **{k: v.copy() for k, v in problem.policy.items()},
            initial_score=_numpy(result["initial_score"]),
            initial_parts=_numpy(result["initial_parts"]),
            applied=self.directional,
            proposal_parent=problem.parents.copy(),
        )
        outputs = tuple(
            _numpy(result[k])
            for k in (
                "fitted_pose_xyz",
                "local_cost",
                "final_loss_parts",
                "mask_supported",
            )
        )
        self.last_mixed_trace = trace
        self.last_diagnostics = diagnostics
        self.last_mixed_statistics = dict(
            schema="mixed_precision_fitter_v1",
            cache=dict(cache.statistics),
            prepare_seconds=prepared - started,
            generate_seconds=generated - prepared,
            rescore_seconds=rescored - generated,
            diagnostic_copy_seconds=time.perf_counter() - rescored,
            bank_size=16,
            FP64_objective_evaluations=17,
            FP32_evaluations_per_replay=15,
            FP32_updates_per_replay=14,
            float_input_policy="complete original FP64 inputs then round to FP32 once",
            scientific_dtype="float64",
            equivalent_to_FP64_search=False,
        )
        return outputs

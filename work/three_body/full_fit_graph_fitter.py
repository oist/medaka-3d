"""Unqualified FP64 full-fit capture: 14 default-foreach Adam updates, 15 scores.

No nested CUDA graph is constructed. The frozen GraphFit._operation is expanded
15 times within one new capture. Its public host validation is performed before
capture/replay, with intermediate finite flags checked before publishing results.
Final diagnostic objective/parts/support remain the frozen eager operations.
The cache owns at most ONE full-fit graph; memory and CUDA parity are unqualified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
import torch
from directional_chest_fitter import parent_policy
from directional_device_fitter import center_correction
from fast_directional_device_fitter import (
    FastDirectionalDeviceFitter,
    accumulate_finite,
    require_finite,
    select_strict_best,
    snapshot_initial_diagnostics,
)
from graph_fit import FitInputs, GraphFit, objective
from graph_union import _CaptureRenderer, _positive_transmission_bound
from node_geometry import assigned_rays
from renderer import RefractiveFishRenderer, RendererConfig
from torch.optim.optimizer import _default_to_fused_or_foreach

ADAM_SOURCE_SHA = "8ce09c47f3960bb1b5c7d58a820b3b827892bc1d383664e6e2e2cdd3d502a6fc"
STEPS = 14


@dataclass(frozen=True)
class AdamStep:
    index: int
    step_size: float
    bias_correction2_sqrt: float


def frozen_adam_schedule(lr, steps=STEPS):
    """Same Python-float constants as adam.py's noncapturable foreach branch.

    The old CPU float32 step tensor contains exact integers 1..14. Its .item()
    therefore returns these exact Python floats. No mutable optimizer step state
    survives a fit: the schedule always starts at 1, including graph replays.
    """
    if type(lr) is not float or not math.isfinite(lr) or lr <= 0:
        raise ValueError("positive finite Python float learning rate required")
    if type(steps) is not int or steps != STEPS:
        raise ValueError("exactly 14 Adam updates required")
    return tuple(
        AdamStep(i, (lr / (1 - 0.9 ** float(i))) * -1, (1 - 0.999 ** float(i)) ** 0.5)
        for i in range(1, STEPS + 1)
    )


@torch.no_grad()
def foreach_adam_update_(pose, gradient, exp_avg, exp_avg_sq, constants):
    """Installed default single-parameter foreach GPU operation order, unchanged.

    Fixed float betas/eps, weight_decay=0, amsgrad/maximize/differentiable=False.
    Scalar LIST overloads for sqrt division and addcdiv match the old optimizer.
    No capturable Adam branch, compiler, fused optimizer or in-place sqrt.
    """
    torch._foreach_lerp_([exp_avg], [gradient], 1 - 0.9)
    torch._foreach_mul_([exp_avg_sq], 0.999)
    torch._foreach_addcmul_([exp_avg_sq], [gradient], [gradient], 1 - 0.999)
    denominator = torch._foreach_sqrt([exp_avg_sq])
    torch._foreach_div_(denominator, [constants.bias_correction2_sqrt])
    torch._foreach_add_(denominator, 1e-8)
    torch._foreach_addcdiv_([pose], [exp_avg], denominator, [constants.step_size])


class FullFitGraph(GraphFit):
    """Own static inputs, moments, scratch objective and one complete-fit graph.

    Inherits frozen input/pose validation and configure's exact copies. It does
    not invoke GraphFit.__init__/evaluate or create an inner CUDA graph. The
    frozen private _operation executes the unchanged objective/backward.
    """

    def __init__(
        self,
        config,
        initial_pose,
        inputs,
        renderer,
        axes,
        parallel_sigma,
        enabled,
        lower,
        upper,
        *,
        directional=True,
        warmup=3,
    ):
        if initial_pose.device.type != "cuda" or initial_pose.dtype != torch.float64:
            raise ValueError("full-fit capture requires CUDA float64")
        if type(directional) is not bool or type(warmup) is not int or warmup < 1:
            raise ValueError("explicit direction policy and positive integer warmup required")
        if _default_to_fused_or_foreach([initial_pose], False, use_fused=False) != (False, True):
            raise ValueError("installed default Adam does not select unfused foreach")
        self._initialize(
            config,
            initial_pose,
            inputs,
            renderer,
            axes,
            parallel_sigma,
            enabled,
            lower,
            upper,
            directional,
        )
        with torch.cuda.device(self.device):
            self._stream = torch.cuda.Stream(device=self.device)
            current = torch.cuda.current_stream(self.device)
            self._stream.wait_stream(current)
            with torch.cuda.stream(self._stream):
                for _ in range(warmup):
                    self._operation()
            current.wait_stream(self._stream)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, stream=self._stream):
                self._operation()
        self.replays = 0

    def _initialize(
        self,
        config,
        initial_pose,
        inputs,
        renderer,
        axes,
        parallel_sigma,
        enabled,
        lower,
        upper,
        directional,
    ):
        """Also used by tiny synthetic CPU tests, never by the production CPU fitter."""
        if config.opt_steps != STEPS or initial_pose.dtype != torch.float64:
            raise ValueError("fixed float64/14-step configuration required")
        self.schedule = frozen_adam_schedule(config.learning_rate_mm, config.opt_steps)
        self.config, self.device, self.dtype = config, initial_pose.device, initial_pose.dtype
        self.batch_size, self.directional = len(initial_pose), directional
        self.n_pixels = renderer.n_pixels
        self._renderer_signature = (
            renderer.views,
            renderer.config,
            renderer.device,
            renderer.dtype,
            renderer.n_pixels,
        )
        self._validate_pose(initial_pose)
        self._validate_inputs(inputs, renderer)
        self._validate_extra(axes, parallel_sigma, enabled, lower, upper)
        self._start = initial_pose.detach().clone()
        self._working = initial_pose.detach().clone()
        self._pose = torch.empty_like(initial_pose, memory_format=torch.contiguous_format)
        self._pose.copy_(initial_pose.detach()).requires_grad_(True)
        self._pose.grad = torch.zeros_like(self._pose)
        self._moment, self._variance = (
            torch.zeros_like(self._working),
            torch.zeros_like(self._working),
        )
        self._inputs = FitInputs(
            **{
                f.name: None
                if getattr(inputs, f.name) is None
                else getattr(inputs, f.name).detach().clone()
                for f in fields(inputs)
            }
        )
        self.denominator = 9 * float(inputs.node_weight.sum())
        renderer.gaussians(initial_pose)
        self._renderer = object.__new__(_CaptureRenderer)
        self._renderer.__dict__ = renderer.__dict__.copy()
        for name in ("ray_origins", "ray_directions", "ray_valid"):
            setattr(self._renderer, name, getattr(renderer, name).detach().clone())
        self._width = torch.full(
            (self.batch_size, 3), renderer.config.width_ratio, dtype=self.dtype, device=self.device
        )
        bound = _positive_transmission_bound(renderer.config, self.dtype)
        self.uses_positive_product_bound = bound > 4 * torch.finfo(self.dtype).eps
        self._axes, self._parallel, self._enabled = (
            axes.clone(),
            parallel_sigma.clone(),
            enabled.clone(),
        )
        self._lower, self._upper = lower.clone(), upper.clone()

    def _validate_extra(self, axes, parallel_sigma, enabled, lower, upper):
        shapes = (
            (axes, (self.batch_size, 3, 3)),
            (parallel_sigma, (self.batch_size, 3)),
            (lower, (3,)),
            (upper, (3,)),
        )
        for value, shape in shapes:
            if value.shape != shape or value.dtype != self.dtype or value.device != self.device:
                raise ValueError("direction/clamp tensor schema differs")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("direction/clamp values must be finite")
        if (
            enabled.shape != (self.batch_size, 3)
            or enabled.dtype != torch.bool
            or enabled.device != self.device
        ):
            raise ValueError("direction-enabled tensor schema differs")
        if not bool((parallel_sigma > 0).all()) or not bool((lower <= upper).all()):
            raise ValueError("positive parallel scales and ordered clamp bounds required")

    def configure_frame(self, inputs, renderer, axes, parallel_sigma, enabled, lower, upper):
        self._validate_extra(axes, parallel_sigma, enabled, lower, upper)
        GraphFit.configure(self, inputs, renderer)
        with torch.no_grad():
            for dst, src in (
                (self._axes, axes),
                (self._parallel, parallel_sigma),
                (self._enabled, enabled),
                (self._lower, lower),
                (self._upper, upper),
            ):
                dst.copy_(src)

    def _operation(self):
        """One replay includes reset, all updates, strict-best and diagnostic copies."""
        with torch.no_grad():
            self._working.copy_(self._start)
            self._moment.zero_()
            self._variance.zero_()
            best_pose = self._working.detach().clone()
            best_score = torch.full(
                (self.batch_size,), float("inf"), device=self.device, dtype=self.dtype
            )
            score_ok = torch.ones((), device=self.device, dtype=torch.bool)
            gradient_ok = torch.ones_like(score_ok)
            pose_ok = torch.ones_like(score_ok)
            best_iteration = torch.full(
                (self.batch_size,), -1, device=self.device, dtype=torch.int64
            )
        trace = {
            name: []
            for name in (
                "pose_before",
                "score",
                "parts",
                "base_gradient",
                "update_gradient",
                "moment",
                "variance",
                "pose_after",
            )
        }
        for step in range(STEPS + 1):
            with torch.no_grad():
                pose_ok = accumulate_finite(pose_ok, self._working)
                self._pose.copy_(self._working)
            # Frozen exact operation (including grad.zero_ and product backward mode).
            with torch.enable_grad():
                GraphFit._operation(self)
            with torch.no_grad():
                # Same three owned copies as GraphFit.evaluate(copy_outputs=True).
                original_score, parts, original_gradient = (
                    value.detach().clone() for value in (self._scores, self._parts, self._pose.grad)
                )
                difference, gradient, center = center_correction(
                    self._working[:, :, 1] - self._inputs.predicted[:, :, 1],
                    self._inputs.motion_sigma,
                    self._axes,
                    self._parallel,
                    self._enabled,
                )
                score = (
                    original_score + self.config.motion_weight * difference
                    if self.directional
                    else original_score
                )
                if step == 0:
                    self.initial_score, self.initial_parts = snapshot_initial_diagnostics(
                        score, parts, center, self.directional
                    )
                score_ok = accumulate_finite(score_ok, score)
                best_iteration = torch.where(score < best_score, step, best_iteration)
                best_pose, best_score = select_strict_best(
                    best_pose, best_score, self._working, score
                )
                for name, value in (
                    ("pose_before", self._working),
                    ("score", score),
                    ("parts", parts),
                    ("base_gradient", original_gradient),
                ):
                    trace[name].append(value.detach().clone())
                if step == STEPS:
                    break
                update = original_gradient.detach().clone()
                if self.directional:
                    update[:, :, 1] += self.config.motion_weight * gradient
                gradient_ok = accumulate_finite(gradient_ok, update)
                trace["update_gradient"].append(update.clone())
                foreach_adam_update_(
                    self._working, update, self._moment, self._variance, self.schedule[step]
                )
                self._working.clamp_(self._lower, self._upper)
                for name, value in (
                    ("moment", self._moment),
                    ("variance", self._variance),
                    ("pose_after", self._working),
                ):
                    trace[name].append(value.clone())
        with torch.no_grad():
            self.best_pose, self.best_score, self.best_iteration = (
                best_pose,
                best_score,
                best_iteration,
            )
            self.score_ok, self.gradient_ok, self.pose_ok = score_ok, gradient_ok, pose_ok
            self.trace = {name: torch.stack(values) for name, values in trace.items()}

    def replay(self, pose):
        self._validate_pose(pose)
        with torch.no_grad():
            self._start.copy_(pose.detach())
            self._graph.replay()
        self.replays += 1
        # Borrowed graph values remain private until fitter consumes/copies them.
        return self


class FullFitDirectionalDeviceFitter(FastDirectionalDeviceFitter):
    """Experimental CUDA replacement only; CPU delegates to frozen Fast fitter."""

    def _fit(self, proposals, states, rays, targets):
        if self.device.type != "cuda":
            return super()._fit(proposals, states, rays, targets)
        cfg, tensor = self.config, self._tensor
        if self.dtype != torch.float64 or cfg.opt_steps != STEPS or targets is None:
            raise ValueError("full-fit prototype requires FP64/14 steps/original D2 targets")
        policy = parent_policy(proposals, states, rays, self.rig, cfg)
        parents = np.asarray([p.parent for p in proposals], np.int64)
        axes = tensor(policy["axes"][parents])
        enabled = torch.as_tensor(policy["enabled"][parents] & self.directional, device=self.device)
        parallel_sigma = tensor(policy["parallel_sigma"][parents])
        origins, directions, valid = assigned_rays(rays, np.stack([p.plan for p in proposals]))
        predicted = tensor(np.stack([p.predicted for p in proposals]))
        missing = tensor(np.stack([states[p.parent].missing for p in proposals]))
        sigma = cfg.motion_sigma_mm + 0.3 * torch.clamp(missing, max=20)
        lengths = tensor(self.lengths)[None]
        node_weight = tensor([1.0, 1.0, 0.35])[None, None, None]
        renderer = RefractiveFishRenderer(
            self.rig,
            targets.pixels,
            RendererConfig(width_ratio=cfg.width_ratio),
            device=self.device,
            dtype=self.dtype,
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
        denominator = 9 * float(node_weight.sum())
        pose = tensor(np.stack([p.pose for p in proposals])).clone().requires_grad_(True)
        lower, upper = tensor(self.lower), tensor(self.upper)
        key = (len(proposals), renderer.n_pixels, cfg, self.directional)
        if self.__dict__.get("_full_fit_key") != key:
            # At most one capture resident, unlike the old four-entry fitter cache.
            self.__dict__.pop("_full_fit_graph", None)
            self._full_fit_key = key
            self._full_fit_graph = FullFitGraph(
                cfg,
                pose,
                inputs,
                renderer,
                axes,
                parallel_sigma,
                enabled,
                lower,
                upper,
                directional=self.directional,
            )
            self._full_fit_capture_count = self.__dict__.get("_full_fit_capture_count", 0) + 1
        else:
            self._full_fit_graph.configure_frame(
                inputs, renderer, axes, parallel_sigma, enabled, lower, upper
            )
        graph = self._full_fit_graph.replay(pose)
        best_pose, initial_score, initial_parts = (
            graph.best_pose,
            graph.initial_score,
            graph.initial_parts,
        )
        # Frozen Fast final objective and support calculation, outside the new capture.
        with torch.no_grad():
            scores, parts, rendering = objective(
                best_pose, cfg, inputs, denominator, renderer, diagnostics=True
            )
            if self.directional:
                difference, _, center = center_correction(
                    best_pose[:, :, 1] - predicted[:, :, 1], sigma, axes, parallel_sigma, enabled
                )
                scores = scores + cfg.motion_weight * difference
                parts = torch.cat((parts[:, :2], center[:, None], parts[:, 3:]), dim=-1)
            contribution = rendering.visible_alpha
            distinct = rendering.ownership >= 0.65
            positive = (
                contribution * distinct * weight[None, :, :, None] * target[None, :, :, None]
            ).sum(2)
            negative = (
                contribution * weight[None, :, :, None] * (1 - target)[None, :, :, None]
            ).sum(2)
            view_support = (positive >= 0.025) & (positive >= 2 * negative)
            support = view_support.sum(1) >= 2
            output_ok = accumulate_finite(
                graph.pose_ok,
                best_pose,
                scores,
                parts,
                initial_score,
                initial_parts,
                contribution,
                rendering.ownership,
                positive,
                negative,
            )
        require_finite(graph.score_ok, graph.gradient_ok, output_ok)
        self.last_diagnostics = dict(
            **{k: v.copy() for k, v in policy.items()},
            initial_score=initial_score.cpu().numpy().copy(),
            initial_parts=initial_parts.cpu().numpy().copy(),
            applied=self.directional,
            proposal_parent=parents.copy(),
        )
        self.last_full_fit_trace = {
            **{k: v.detach().cpu().numpy().copy() for k, v in graph.trace.items()},
            "best_iteration": graph.best_iteration.cpu().numpy().copy(),
            "best_score": graph.best_score.cpu().numpy().copy(),
        }
        return (
            best_pose.cpu().numpy().copy(),
            scores.cpu().numpy().copy(),
            parts.cpu().numpy().copy(),
            support.cpu().numpy().copy(),
        )

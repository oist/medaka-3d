"""Unqualified CUDA execution variant of the frozen directional objective.

CPU delegates unchanged. CUDA retains the objective, Adam and support arithmetic,
but selects best poses with fixed-shape where and defers fitter finite checks.
GraphFit/renderer validations still synchronize. GPU equivalence and speed have
not been measured; this class is not integrated into any continuous tracker.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from directional_chest_fitter import parent_policy
from directional_device_fitter import DirectionalDeviceFitter, center_correction
from graph_fit import FitInputs, GraphFit, objective
from node_geometry import assigned_rays
from renderer import RefractiveFishRenderer, RendererConfig


def select_strict_best(best_pose, best_score, pose, score):
    """Whole-candidate replacement only for strictly lower scores; ties keep pose."""
    improved = score < best_score
    return (
        torch.where(improved[:, None, None, None], pose.detach(), best_pose),
        torch.minimum(best_score, score),
    )


def accumulate_finite(flag, *values):
    """Accumulate a sticky scalar tensor flag without reading it on the host."""
    for value in values:
        flag = flag & torch.isfinite(value).all()
    return flag


def snapshot_initial_diagnostics(score, parts, center, directional):
    """Own initial device values before subsequent graph evaluations reuse buffers."""
    initial_score, initial_parts = score.detach().clone(), parts.detach().clone()
    if directional:
        initial_parts[:, 2] = center
    return initial_score, initial_parts


def require_finite(score_ok, gradient_ok, output_ok):
    """One flag transfer before publishing diagnostics or returning candidate data."""
    flags = torch.stack((score_ok, gradient_ok, output_ok)).detach().cpu().tolist()
    for finite, label in zip(flags, ("score", "gradient", "output"), strict=True):
        if not finite:
            raise FloatingPointError("nonfinite device directional " + label)


class FastDirectionalDeviceFitter(DirectionalDeviceFitter):
    def _fit(self, proposals, states, rays, targets):
        if self.device.type != "cuda":
            return super()._fit(proposals, states, rays, targets)
        cfg, tensor = self.config, self._tensor
        if targets is None:
            raise ValueError("directional qualification requires the original D2 targets")
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
        cache = self.__dict__.setdefault("_fit_graph_cache", OrderedDict())
        key = (len(proposals), renderer.n_pixels, cfg)
        if key in cache:
            graph = cache.pop(key)
            graph.configure(inputs, renderer)
        else:
            if len(cache) >= 4:
                cache.popitem(last=False)
            graph = GraphFit(cfg, pose, inputs, renderer)
        cache[key] = graph
        optimizer = torch.optim.Adam([pose], lr=cfg.learning_rate_mm)
        best_pose = pose.detach().clone()
        best_score = torch.full(
            (len(proposals),), float("inf"), device=self.device, dtype=self.dtype
        )
        score_ok = torch.ones((), device=self.device, dtype=torch.bool)
        gradient_ok = torch.ones((), device=self.device, dtype=torch.bool)
        initial_score = initial_parts = None
        for step in range(cfg.opt_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            result = graph.evaluate(pose)
            original_score, parts, original_gradient = (
                result.scores,
                result.parts,
                result.pose_gradient,
            )
            with torch.no_grad():
                difference, gradient, center = center_correction(
                    pose[:, :, 1] - predicted[:, :, 1], sigma, axes, parallel_sigma, enabled
                )
                score = (
                    original_score + cfg.motion_weight * difference
                    if self.directional
                    else original_score
                )
                if step == 0:
                    initial_score, initial_parts = snapshot_initial_diagnostics(
                        score, parts, center, self.directional
                    )
                score_ok = accumulate_finite(score_ok, score)
                best_pose, best_score = select_strict_best(best_pose, best_score, pose, score)
                if step == cfg.opt_steps:
                    break
                pose.grad = original_gradient.detach().clone()
                if self.directional:
                    pose.grad[:, :, 1] += cfg.motion_weight * gradient
                gradient_ok = accumulate_finite(gradient_ok, pose.grad)
            optimizer.step()
            with torch.no_grad():
                pose.clamp_(lower, upper)
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
                torch.ones((), device=self.device, dtype=torch.bool),
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
        # Invalid iterations may already have failed GraphFit/renderer validation.
        # If they reached here, fail before replacing diagnostics or returning data.
        require_finite(score_ok, gradient_ok, output_ok)
        self.last_diagnostics = dict(
            **{k: v.copy() for k, v in policy.items()},
            initial_score=initial_score.cpu().numpy().copy(),
            initial_parts=initial_parts.cpu().numpy().copy(),
            applied=self.directional,
            proposal_parent=parents.copy(),
        )
        return (
            best_pose.cpu().numpy().copy(),
            scores.cpu().numpy().copy(),
            parts.cpu().numpy().copy(),
            support.cpu().numpy().copy(),
        )

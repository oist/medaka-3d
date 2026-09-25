"""CPU/CUDA execution of the already fixed directional chest objective.

CUDA uses the frozen isotropic GraphFit and adds the exact directional quadratic
correction and its gradient outside capture. No proposal or temporal rule lives
here. CPU provides a checkable execution path against the completed local fit.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from directional_chest_fitter import directional_center, parent_policy
from graph_fit import FitInputs, GraphFit, objective
from node_geometry import assigned_rays
from renderer import RefractiveFishRenderer, RendererConfig
from tracker import ThreeBodyTracker


def center_correction(delta, sigma, axes, parallel_sigma, enabled):
    """Return center-cost difference and analytic d(difference)/d(chest XYZ)."""
    original = ((delta / sigma[..., None]) ** 2).sum(-1).mean(-1)
    replacement = directional_center(delta, sigma, axes, parallel_sigma, enabled)
    active = enabled & (parallel_sigma > sigma)
    parallel = (delta * axes).sum(-1)
    transverse = delta - parallel[..., None] * axes
    # Keep the derivative of the actual implemented transverse projection. This
    # remains consistent even at the small unit-vector rounding tolerance.
    transverse_grad = transverse - (transverse * axes).sum(-1)[..., None] * axes
    alternative_grad = 2 * (
        transverse_grad / sigma[..., None] ** 2
        + parallel[..., None] * axes / parallel_sigma[..., None] ** 2
    )
    original_grad = 2 * delta / sigma[..., None] ** 2
    gradient = torch.where(active[..., None], alternative_grad - original_grad, 0)
    return replacement - original, gradient / delta.shape[1], replacement


class DirectionalDeviceFitter(ThreeBodyTracker):
    def __init__(self, *args, directional=True, **kwargs):
        super().__init__(*args, **kwargs)
        if type(directional) is not bool or self.device.type not in ("cpu", "cuda"):
            raise ValueError("explicit directional bool and CPU/CUDA required")
        self.directional = directional
        self.last_diagnostics = None

    def _fit(self, proposals, states, rays, targets):
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
        graph = None
        if self.device.type == "cuda":
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
        initial_score = initial_parts = None
        for step in range(cfg.opt_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            if graph is None:
                original_score, parts, _ = objective(pose, cfg, inputs, denominator, renderer)
                original_gradient = torch.autograd.grad(original_score.sum(), pose)[0]
            else:
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
                    initial_score = score.cpu().numpy().copy()
                    initial_parts = parts.detach().cpu().numpy().copy()
                    if self.directional:
                        initial_parts[:, 2] = center.cpu().numpy()
                if not bool(torch.isfinite(score).all()):
                    raise FloatingPointError("nonfinite device directional score")
                improved = score < best_score
                best_pose[improved] = pose.detach()[improved]
                best_score = torch.minimum(best_score, score)
                if step == cfg.opt_steps:
                    break
                pose.grad = original_gradient.detach().clone()
                if self.directional:
                    pose.grad[:, :, 1] += cfg.motion_weight * gradient
                if not bool(torch.isfinite(pose.grad).all()):
                    raise FloatingPointError("nonfinite device directional gradient")
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
        self.last_diagnostics = dict(
            **{k: v.copy() for k, v in policy.items()},
            initial_score=initial_score,
            initial_parts=initial_parts,
            applied=self.directional,
            proposal_parent=parents.copy(),
        )
        return (
            best_pose.cpu().numpy().copy(),
            scores.cpu().numpy().copy(),
            parts.cpu().numpy().copy(),
            support.cpu().numpy().copy(),
        )

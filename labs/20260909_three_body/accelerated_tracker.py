"""Experimental CUDA Graph fitter; every other frozen tracking method is inherited."""

from collections import OrderedDict

import numpy as np
import torch
from graph_fit import FitInputs, GraphFit, objective
from node_geometry import assigned_rays
from renderer import RefractiveFishRenderer, RendererConfig
from tracker import ThreeBodyTracker


class AcceleratedThreeBodyTracker(ThreeBodyTracker):
    """Only _fit changes execution; proposals, support, beam and backtrace are inherited.

    Up to four fixed-shape captures are cached. Eviction only recaptures numerical
    operations; it never clears temporal state or initializes a fish. CPU callers
    must use the original explicit ThreeBodyTracker command instead.
    """

    def _fit(self, proposals, states, rays, targets):
        if self.device.type != "cuda":
            raise ValueError("AcceleratedThreeBodyTracker requires CUDA")
        cfg = self.config
        plans = np.stack([p.plan for p in proposals])
        origin, direction, valid = assigned_rays(rays, plans)
        origins, directions = self._tensor(origin), self._tensor(direction)
        held = self._tensor(valid)
        node_weight = self._tensor([1.0, 1.0, 0.35])[None, None, None]
        denominator = 9 * float(node_weight.sum())
        predicted = self._tensor(np.stack([p.predicted for p in proposals]))
        missing = self._tensor(np.stack([states[p.parent].missing for p in proposals]))
        motion_sigma = cfg.motion_sigma_mm + 0.3 * torch.clamp(missing, max=20)
        lengths = self._tensor(self.lengths)[None]
        length_sigma = torch.clamp(0.2 * lengths, min=1.0)
        lower, upper = self._tensor(self.lower), self._tensor(self.upper)
        pose = self._tensor(np.stack([p.pose for p in proposals])).clone().requires_grad_(True)
        renderer, target, weight = None, None, None
        if targets is not None:
            renderer = RefractiveFishRenderer(
                self.rig, targets.pixels, RendererConfig(width_ratio=cfg.width_ratio),
                device=self.device, dtype=self.dtype)
            target = self._tensor(targets.target)
            weight = self._tensor(targets.weights) * renderer.ray_valid
        inputs = FitInputs(origins, directions, held, predicted, motion_sigma, lengths,
                           length_sigma, node_weight, target, weight)
        cache = self.__dict__.setdefault("_fit_graph_cache", OrderedDict())
        key = (len(proposals), None if renderer is None else renderer.n_pixels, cfg)
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
            (len(proposals),), float("inf"), device=self.device, dtype=self.dtype)
        for step in range(cfg.opt_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            result = graph.evaluate(pose)
            score = result.scores
            if not bool(torch.isfinite(score).all()):
                raise FloatingPointError("nonfinite joint body objective")
            improved = score.detach() < best_score
            best_pose[improved] = pose.detach()[improved]
            best_score = torch.minimum(best_score, score.detach())
            if step == cfg.opt_steps:
                break
            pose.grad = result.pose_gradient
            if pose.grad is None or not bool(torch.isfinite(pose.grad).all()):
                raise FloatingPointError("nonfinite joint body gradient")
            optimizer.step()
            with torch.no_grad():
                pose.clamp_(lower, upper)
        with torch.no_grad():
            scores, parts, rendering = objective(
                best_pose, cfg, inputs, denominator, renderer, diagnostics=True)
            image_support = torch.zeros((len(proposals), 3), dtype=torch.bool, device=self.device)
            if rendering is not None:
                contribution = rendering.visible_alpha
                distinct = rendering.ownership >= 0.65
                positive = (
                    contribution * distinct * weight[None, :, :, None] * target[None, :, :, None]
                ).sum(2)
                negative = (
                    contribution * weight[None, :, :, None] * (1 - target)[None, :, :, None]
                ).sum(2)
                view_support = (positive >= 0.025) & (positive >= 2 * negative)
                image_support = view_support.sum(1) >= 2
        return (best_pose.cpu().numpy(), scores.cpu().numpy(), parts.cpu().numpy(),
                image_support.cpu().numpy())

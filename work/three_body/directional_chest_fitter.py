"""Local CPU fitter with an explicitly experimental directional chest motion cost.

The parent/current-predict observation determines its weak direction. Every
proposal of that parent shares the same policy, including original proposals.
No tracker is started or resumed here. This is not a calibrated probability.
"""

from __future__ import annotations

import numpy as np
import torch
from graph_fit import FitInputs, objective
from node_geometry import assigned_rays, supported_nodes
from renderer import RefractiveFishRenderer, RendererConfig
from tracker import ThreeBodyTracker


def directional_center(delta, sigma, axes, parallel_sigma, enabled):
    """Preserve transverse precision; soften only a registered unit-ray direction."""
    original = ((delta / sigma[..., None]) ** 2).sum(-1)
    parallel = (delta * axes).sum(-1)
    transverse = delta - parallel[..., None] * axes
    alternate = ((transverse / sigma[..., None]) ** 2).sum(-1) + (parallel / parallel_sigma) ** 2
    active = enabled & (parallel_sigma > sigma)
    return torch.where(active, alternate, original).mean(-1)


def parent_policy(proposals, states, rays, rig, cfg):
    count = len(states)
    axes = np.zeros((count, 3, 3), np.float64)
    enabled = np.zeros((count, 3), bool)
    source = np.full((count, 3, 3), -1, np.int32)
    sigma = cfg.motion_sigma_mm + 0.3 * np.minimum(np.stack([s.missing for s in states]), 20)
    age = np.stack([s.node_missing[:, 1] for s in states])
    parallel = np.maximum(sigma, cfg.motion_sigma_mm + 0.3 * np.minimum(age, 20))
    for parent in range(count):
        matches = [
            p
            for p in proposals
            if p.parent == parent and p.kind == "predict" and p.permutation == (0, 1, 2)
        ]
        if len(matches) != 1:
            raise ValueError("each saved parent needs exactly one original predict proposal")
        proposal = matches[0]
        held, _, _ = supported_nodes(proposal.pose, proposal.plan, rays, rig)
        source[parent] = held[:, :, 1]
        for fish in range(3):
            views = np.flatnonzero(source[parent, fish] >= 0)
            if len(views) != 1 or age[parent, fish] < 4:
                continue
            view = int(views[0])
            axis = rays.directions[view][source[parent, fish, view], 1]
            if not np.isclose(np.linalg.norm(axis), 1.0, rtol=0, atol=1e-10):
                raise ValueError("registered weak direction must be a unit ray")
            axes[parent, fish] = axis
            enabled[parent, fish] = True
    return dict(
        axes=axes,
        enabled=enabled,
        sigma=sigma,
        parallel_sigma=parallel,
        sources=source,
        chest_age=age,
    )


class LocalDirectionalChestFitter(ThreeBodyTracker):
    def __init__(self, *args, directional=True, **kwargs):
        super().__init__(*args, **kwargs)
        if self.device.type != "cpu" or type(directional) is not bool:
            raise ValueError("this local experimental fitter requires explicit CPU and bool mode")
        self.directional = directional
        self.last_diagnostics = None

    def _fit(self, proposals, states, rays, targets):
        cfg, tensor = self.config, self._tensor
        if targets is None:
            raise ValueError("this local experiment requires the original D2 pixel targets")
        policy = parent_policy(proposals, states, rays, self.rig, cfg)
        parents = np.asarray([p.parent for p in proposals], np.int64)
        axis = tensor(policy["axes"][parents])
        enabled = torch.as_tensor(policy["enabled"][parents] & self.directional, device=self.device)
        sigma_parallel = tensor(policy["parallel_sigma"][parents])
        plans = np.stack([p.plan for p in proposals])
        origin, direction, valid = assigned_rays(rays, plans)
        weights = tensor([1.0, 1.0, 0.35])[None, None, None]
        predicted = tensor(np.stack([p.predicted for p in proposals]))
        missing = tensor(np.stack([states[p.parent].missing for p in proposals]))
        motion_sigma = cfg.motion_sigma_mm + 0.3 * torch.clamp(missing, max=20)
        lengths = tensor(self.lengths)[None]
        renderer = RefractiveFishRenderer(
            self.rig,
            targets.pixels,
            RendererConfig(width_ratio=cfg.width_ratio),
            device=self.device,
            dtype=self.dtype,
        )
        target, weight = tensor(targets.target), tensor(targets.weights) * renderer.ray_valid
        inputs = FitInputs(
            tensor(origin),
            tensor(direction),
            tensor(valid),
            predicted,
            motion_sigma,
            lengths,
            torch.clamp(0.2 * lengths, min=1.0),
            weights,
            target,
            weight,
        )
        denominator = 9 * float(weights.sum())
        pose = tensor(np.stack([p.pose for p in proposals])).clone().requires_grad_(True)

        def evaluate(value, diagnostics=False):
            score, parts, rendering = objective(
                value, cfg, inputs, denominator, renderer, diagnostics=diagnostics
            )
            if self.directional:
                center = directional_center(
                    value[:, :, 1] - predicted[:, :, 1], motion_sigma, axis, sigma_parallel, enabled
                )
                score = score + cfg.motion_weight * (center - parts[:, 2])
                parts = torch.cat((parts[:, :2], center[:, None], parts[:, 3:]), dim=-1)
            return score, parts, rendering

        optimizer = torch.optim.Adam([pose], lr=cfg.learning_rate_mm)
        best_pose = pose.detach().clone()
        best_score = torch.full(
            (len(proposals),), float("inf"), dtype=self.dtype, device=self.device
        )
        initial_score = initial_parts = None
        for step in range(cfg.opt_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            score, parts, _ = evaluate(pose)
            if not bool(torch.isfinite(score).all()):
                raise FloatingPointError("nonfinite directional objective")
            if step == 0:
                initial_score, initial_parts = (
                    score.detach().numpy().copy(),
                    parts.detach().numpy().copy(),
                )
            improved = score.detach() < best_score
            best_pose[improved] = pose.detach()[improved]
            best_score = torch.minimum(best_score, score.detach())
            if step == cfg.opt_steps:
                break
            score.sum().backward()
            if pose.grad is None or not bool(torch.isfinite(pose.grad).all()):
                raise FloatingPointError("nonfinite directional gradient")
            optimizer.step()
            with torch.no_grad():
                pose.clamp_(tensor(self.lower), tensor(self.upper))
        with torch.no_grad():
            scores, parts, rendering = evaluate(best_pose, diagnostics=True)
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
        self.last_diagnostics = dict(
            **{key: value.copy() for key, value in policy.items()},
            initial_score=initial_score,
            initial_parts=initial_parts,
            applied=self.directional,
            proposal_parent=parents.copy(),
        )
        return (
            best_pose.numpy().copy(),
            scores.numpy().copy(),
            parts.numpy().copy(),
            image_support.numpy().copy(),
        )

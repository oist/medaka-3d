"""Separate temporal arm: fixed reflected mask cost, original direct support.

Only fitting is overridden. The persistent proposal, beam, optical flow, age,
review and backtrace code is inherited without changes. Reflection does not
provide image bridges: those still use the original direct rendering at the
fitted pose. The unbounded direct branch is an inherited approximation and the
nominal reflection planes are not newly surveyed tank geometry.
"""

import numpy as np
import torch
from graph_fit import FitInputs, objective
from node_geometry import assigned_rays
from persistent_node_tracker import PersistentNodeThreeBodyTracker
from reflection_local_fit import UnionAdapter
from reflection_renderer import ReflectionFishRenderer
from renderer import RendererConfig


def direct_image_support(renderer, pose, target, weight):
    """Frozen direct visibility/ownership bridge, never reflected responses."""
    rendering = renderer.render(pose)
    contribution = rendering.visible_alpha
    distinct = rendering.ownership >= 0.65
    positive = (
        contribution * distinct * weight[None, :, :, None] * target[None, :, :, None]
    ).sum(2)
    negative = (
        contribution * weight[None, :, :, None] * (1 - target)[None, :, :, None]
    ).sum(2)
    view_support = (positive >= 0.025) & (positive >= 2 * negative)
    return view_support.sum(1) >= 2


class ReflectionPersistentNodeTracker(PersistentNodeThreeBodyTracker):
    """Explicit CPU/CUDA float64 optimizer; no CUDA graph or extra proposals.

    The direct arm delegates to the original explicit fitter. Reflected arms
    use the same operation order, independent per-proposal best iterate, and
    constant per-frame optical paths. GPU use requires separate qualification.
    """

    def __init__(self, *args, mask_arm="fresnel", **kwargs):
        if mask_arm not in ("direct", "unit", "fresnel"):
            raise ValueError("mask_arm must be direct, unit or fresnel")
        super().__init__(*args, **kwargs)
        self.mask_arm = mask_arm

    def _fit(self, proposals, states, rays, targets):
        if self.mask_arm == "direct" or targets is None:
            return super()._fit(proposals, states, rays, targets)
        cfg = self.config
        plans = np.stack([p.plan for p in proposals])
        origin, direction, valid = assigned_rays(rays, plans)
        predicted = self._tensor(np.stack([p.predicted for p in proposals]))
        missing = self._tensor(np.stack([states[p.parent].missing for p in proposals]))
        node_weight = self._tensor([1., 1., .35])[None, None, None]
        denominator = 9 * float(node_weight.sum())
        lengths = self._tensor(self.lengths)[None]
        reflected = ReflectionFishRenderer(
            self.rig, targets.pixels, RendererConfig(width_ratio=cfg.width_ratio),
            direct_mode="unbounded", device=self.device, dtype=self.dtype,
        )
        renderer = UnionAdapter(reflected, self.mask_arm + "_union")
        target = self._tensor(targets.target)
        weight = self._tensor(targets.weights) * reflected.ray_valid
        inputs = FitInputs(
            self._tensor(origin), self._tensor(direction), self._tensor(valid),
            predicted, cfg.motion_sigma_mm + .3 * torch.clamp(missing, max=20),
            lengths, torch.clamp(.2 * lengths, min=1.), node_weight, target, weight,
        )
        pose = self._tensor(np.stack([p.pose for p in proposals])).clone().requires_grad_(True)
        lower, upper = self._tensor(self.lower), self._tensor(self.upper)
        optimizer = torch.optim.Adam([pose], lr=cfg.learning_rate_mm)
        best_pose = pose.detach().clone()
        best_score = torch.full(
            (len(proposals),), float("inf"), device=self.device, dtype=self.dtype,
        )
        for step in range(cfg.opt_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            score, _, _ = objective(
                pose, cfg, inputs, denominator, renderer, diagnostics=True,
            )
            if not bool(torch.isfinite(score).all()):
                raise FloatingPointError("nonfinite reflected body objective")
            improved = score.detach() < best_score
            best_pose[improved] = pose.detach()[improved]
            best_score = torch.minimum(best_score, score.detach())
            if step == cfg.opt_steps:
                break
            score.sum().backward()
            if pose.grad is None or not bool(torch.isfinite(pose.grad).all()):
                raise FloatingPointError("nonfinite reflected body gradient")
            optimizer.step()
            with torch.no_grad():
                pose.clamp_(lower, upper)
        with torch.no_grad():
            scores, parts, _ = objective(
                best_pose, cfg, inputs, denominator, renderer, diagnostics=True,
            )
            image_support = direct_image_support(reflected._base, best_pose, target, weight)
        return tuple(value.cpu().numpy() for value in (best_pose, scores, parts, image_support))

"""Fixed registered radial widths in base, owner, gap and final image support.

Used only after causal t0. Every call runs the same original Adam loop, even
with coefficient zero or no transport targets. Initial t0 delegation belongs
to the new tracker. Active diagnostics mean auxiliary loss is evaluated;
all post-t0 fits retain their 15 objective and 14 gradient-norm records.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "labs/20260909_three_body")]
from fixed_body_width_graph_fit import FixedWidthGraphFit  # noqa: E402
from fixed_body_width_renderer import FixedWidthRenderer  # noqa: E402
from graph_fit import FitInputs, objective  # noqa: E402
from mask_transport_fitter import ownership_loss  # noqa: E402
from mask_transport_gap_fitter import ray_gap_loss, require  # noqa: E402
from node_geometry import assigned_rays  # noqa: E402
from renderer import RendererConfig  # noqa: E402


def fit_with_fixed_width(tracker, proposals, states, rays, targets, target, *, coefficient=3.0):
    """Same original proposal pool; T[B,V,N,3] is precomputed from parent anchors."""
    require(
        not isinstance(coefficient, (bool, np.bool_)) and coefficient in (0.0, 3.0),
        "coefficient must be fixed3 or the zero qualification control",
    )
    require(targets is not None, "fixed-width experiment requires original D2 PixelTargets")
    active = coefficient != 0
    reason = "fixed_radial_owner_plus_gap" if active else "fixed_radial_coefficient_zero"
    if active:
        require(
            isinstance(target, np.ndarray)
            and target.dtype == np.float64
            and target.shape == (len(proposals), *targets.target.shape, 3)
            and np.isfinite(target).all()
            and ((target >= 0) & (target <= 1)).all()
            and ((target > 0).sum(-1) <= 1).all(),
            "fixed ownership target schema differs",
        )
        mass = target * (targets.weights * targets.target)[None, :, :, None]
        if not np.any(mass):
            active, reason = False, "fixed_radial_zero_effective_target"

    cfg, tensor = tracker.config, tracker._tensor
    plans = np.stack([p.plan for p in proposals])
    origin, direction, valid = assigned_rays(rays, plans)
    origins, directions, held = tensor(origin), tensor(direction), tensor(valid)
    node_weight = tensor([1.0, 1.0, 0.35])[None, None, None]
    denominator = 9 * float(node_weight.sum())
    predicted = tensor(np.stack([p.predicted for p in proposals]))
    missing = tensor(np.stack([states[p.parent].missing for p in proposals]))
    sigma = cfg.motion_sigma_mm + 0.3 * torch.clamp(missing, max=20)
    lengths = tensor(tracker.lengths)[None]
    length_sigma = torch.clamp(0.2 * lengths, min=1.0)
    lower, upper = tensor(tracker.lower), tensor(tracker.upper)
    pose = tensor(np.stack([p.pose for p in proposals])).clone().requires_grad_(True)
    renderer = FixedWidthRenderer(
        tracker.rig,
        targets.pixels,
        RendererConfig(width_ratio=cfg.width_ratio),
        device=tracker.device,
        dtype=tracker.dtype,
        reference_pose_xyz=tracker._fixed_width_reference_pose,
    )
    tracker.validate_fixed_width_renderer(renderer)
    foreground, base_weight = tensor(targets.target), tensor(targets.weights)
    owner_target = tensor(target.copy()).detach() if active else None
    weight = base_weight * renderer.ray_valid
    if active and not bool((owner_target * (weight * foreground)[None, :, :, None]).any()):
        active, reason = False, "fixed_radial_zero_valid_ray_target"
    renderer.validate_fixed_width()
    inputs = FitInputs(
        origins,
        directions,
        held,
        predicted,
        sigma,
        lengths,
        length_sigma,
        node_weight,
        foreground,
        weight,
    )
    graph = None
    if tracker.device.type == "cuda":
        cache = tracker.__dict__.setdefault("_fit_graph_cache", OrderedDict())
        key = (len(proposals), renderer.n_pixels, cfg, renderer.fixed_width_signature)
        if key in cache:
            graph = cache.pop(key)
            graph.configure(inputs, renderer)
        else:
            if len(cache) >= 4:
                cache.popitem(last=False)
            graph = FixedWidthGraphFit(cfg, pose, inputs, renderer)
        cache[key] = graph

    optimizer = torch.optim.Adam([pose], lr=cfg.learning_rate_mm)
    best_pose = pose.detach().clone()
    best_score = torch.full(
        (len(proposals),), float("inf"), device=tracker.device, dtype=tracker.dtype
    )
    owner_values, base_values, total_values, gradient_norms = [], [], [], []
    gap_values, gap_gradient_norms = [], []
    best_iteration = np.full(len(proposals), -1, np.int64)
    for step in range(cfg.opt_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if graph is None:
            base_score, _, _ = objective(pose, cfg, inputs, denominator, renderer)
            base_gradient = (
                torch.autograd.grad(base_score.sum(), pose)[0] if step < cfg.opt_steps else None
            )
        else:
            evaluation = graph.evaluate(pose)
            base_score, base_gradient = evaluation.scores, evaluation.pose_gradient
        if active:
            rendering = renderer.render(pose)
            owner = ownership_loss(
                rendering.ownership, owner_target, foreground, base_weight, renderer.ray_valid
            )
            gap = ray_gap_loss(
                rendering.per_fish_mahalanobis_sq,
                owner_target,
                foreground,
                base_weight,
                renderer.ray_valid,
            )
        else:
            owner, gap = torch.zeros_like(base_score), torch.zeros_like(base_score)
        score = base_score + coefficient * (owner + gap)
        if not bool(torch.isfinite(score).all()):
            raise FloatingPointError("nonfinite owner-plus-gap combined objective")
        owner_values.append(owner.detach().cpu().numpy().copy())
        gap_values.append(gap.detach().cpu().numpy().copy())
        base_values.append(base_score.detach().cpu().numpy().copy())
        total_values.append(score.detach().cpu().numpy().copy())
        improved = score.detach() < best_score
        best_iteration[improved.cpu().numpy()] = step
        best_pose[improved] = pose.detach()[improved]
        best_score = torch.minimum(best_score, score.detach())
        if step == cfg.opt_steps:
            break
        if active:
            owner_gradient = torch.autograd.grad(owner.sum(), pose, retain_graph=True)[0]
            gap_gradient = torch.autograd.grad(gap.sum(), pose)[0]
        else:
            owner_gradient, gap_gradient = torch.zeros_like(pose), torch.zeros_like(pose)
        pose.grad = base_gradient.detach().clone() + coefficient * (
            owner_gradient.detach() + gap_gradient.detach()
        )
        if not bool(torch.isfinite(pose.grad).all()):
            raise FloatingPointError("nonfinite owner-plus-gap combined gradient")
        gradient_norms.append(
            torch.linalg.vector_norm(owner_gradient.flatten(1), dim=1).detach().cpu().numpy().copy()
        )
        gap_gradient_norms.append(
            torch.linalg.vector_norm(gap_gradient.flatten(1), dim=1).detach().cpu().numpy().copy()
        )
        optimizer.step()
        with torch.no_grad():
            pose.clamp_(lower, upper)
    with torch.no_grad():
        base_scores, parts, rendering = objective(
            best_pose,
            cfg,
            inputs,
            denominator,
            renderer,
            diagnostics=True,
        )
        if active:
            owner = ownership_loss(
                rendering.ownership, owner_target, foreground, base_weight, renderer.ray_valid
            )
            gap, details = ray_gap_loss(
                rendering.per_fish_mahalanobis_sq,
                owner_target,
                foreground,
                base_weight,
                renderer.ray_valid,
                return_details=True,
            )
        else:
            owner, gap = torch.zeros_like(base_scores), torch.zeros_like(base_scores)
            details = dict(
                final_gap_by_view_body=torch.empty(
                    (0, 3, 3), device=tracker.device, dtype=tracker.dtype
                ),
                target_mass_by_view_body=torch.empty(
                    (0, 3, 3), device=tracker.device, dtype=tracker.dtype
                ),
                positive_gap_pixel_count=torch.empty(
                    (0, 3, 3), device=tracker.device, dtype=torch.int64
                ),
            )
        scores = base_scores + coefficient * (owner + gap)
        contribution = rendering.visible_alpha
        distinct = rendering.ownership >= 0.65
        positive = (
            contribution * distinct * weight[None, :, :, None] * foreground[None, :, :, None]
        ).sum(2)
        negative = (
            contribution * weight[None, :, :, None] * (1 - foreground)[None, :, :, None]
        ).sum(2)
        view_support = (positive >= 0.025) & (positive >= 2 * negative)
        support = view_support.sum(1) >= 2
        if not all(
            bool(torch.isfinite(a).all())
            for a in (best_pose, base_scores, scores, parts, owner, gap, positive, negative)
        ):
            raise FloatingPointError("nonfinite owner-plus-gap fit output")
    tracker.last_gap_fit = dict(
        active=active,
        reason=reason,
        coefficient=coefficient,
        base_score=base_scores.cpu().numpy().copy(),
        owner_loss=owner.cpu().numpy().copy(),
        gap_loss=gap.cpu().numpy().copy(),
        total_score=scores.cpu().numpy().copy(),
        original_loss_parts=parts.cpu().numpy().copy(),
        iteration_owner_loss=np.stack(owner_values),
        iteration_gap_loss=np.stack(gap_values),
        iteration_owner_gradient_norm=np.stack(gradient_norms),
        iteration_gap_gradient_norm=np.stack(gap_gradient_norms),
        iteration_base_score=np.stack(base_values),
        iteration_total_score=np.stack(total_values),
        best_iteration=best_iteration,
        target_mass=details["target_mass_by_view_body"].sum((1, 2)).cpu().numpy().copy(),
        target_diagnostics_computed=active,
        **{k: v.cpu().numpy().copy() for k, v in details.items()},
    )
    return (
        best_pose.cpu().numpy(),
        scores.cpu().numpy(),
        parts.cpu().numpy(),
        support.cpu().numpy(),
    )

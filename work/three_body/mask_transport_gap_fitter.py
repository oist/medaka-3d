"""Experimental owner plus normalized half-ray gap around the frozen base fitter.

Targets are fixed evidence in proposal-parent anchor order, never selected using
the fitted pose. The old five loss parts and final support rule stay unchanged.
Coefficient zero, or no effective target mass, delegates to the original fitter.
The nonzero path is unqualified until separate CPU/CUDA comparisons have passed.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "labs/20260909_three_body")]
from graph_fit import FitInputs, GraphFit, objective  # noqa: E402
from mask_transport_fitter import frozen_fit, ownership_loss  # noqa: E402
from node_geometry import assigned_rays  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ray_gap_loss(q, target, foreground, weight, ray_valid, *, return_details=False):
    """Fixed /3 target-weighted sqrt(clamp_min(q,1))-1, with no visibility gate.

    q is the frozen renderer's nearest-Gaussian half-ray Mahalanobis squared
    distance, not Euclidean signed distance or an exact union-silhouette border.
    Its radius follows existing pose-derived lengths; no sigma is detached here.
    Invalid-ray inf is replaced before sqrt. Every valid-ray q is finite and
    nonnegative, including unassigned pixels. The renderer's amin/tie gradient
    is retained; clamp_min uses PyTorch's boundary derivative without replacement.
    """
    require(
        isinstance(q, torch.Tensor) and q.ndim == 4 and q.dtype in (torch.float32, torch.float64),
        "floating ray distance B,V,N,3 required",
    )
    b, v, n, bodies = q.shape
    require(b >= 1 and v == 3 and n >= 1 and bodies == 3, "gap axes differ")
    require(
        isinstance(ray_valid, torch.Tensor)
        and ray_valid.dtype == torch.bool
        and ray_valid.shape == (v, n)
        and ray_valid.device == q.device,
        "gap ray validity schema differs",
    )
    for name, value, shape in (
        ("target", target, q.shape),
        ("foreground", foreground, (v, n)),
        ("weight", weight, (v, n)),
    ):
        require(
            isinstance(value, torch.Tensor)
            and value.shape == shape
            and value.dtype == q.dtype
            and value.device == q.device,
            "gap " + name + " tensor schema differs",
        )
        require(
            not value.requires_grad and bool(torch.isfinite(value).all()),
            "gap " + name + " must be finite detached evidence",
        )
    require(
        bool(((target >= 0) & (target <= 1)).all())
        and bool(((target > 0).sum(-1) <= 1).all())
        and bool(((foreground == 0) | (foreground == 1)).all())
        and bool((weight >= 0).all()),
        "gap target/evidence weights differ",
    )
    valid = ray_valid[None, :, :, None].expand_as(q)
    require(
        bool(torch.isfinite(q[valid]).all()) and bool((q[valid] >= 0).all()),
        "valid ray distance must be finite and nonnegative",
    )
    neutral_q = torch.where(valid, q, torch.zeros_like(q))
    gap = torch.sqrt(neutral_q.clamp_min(1)) - 1
    mass = (weight * ray_valid * foreground)[None, :, :, None] * target
    contributions = mass * gap
    score = contributions.sum((1, 2, 3)) / 3
    if not return_details:
        return score
    return score, dict(
        final_gap_by_view_body=contributions.sum(2) / 3,
        target_mass_by_view_body=mass.sum(2) / 3,
        positive_gap_pixel_count=((mass > 0) & (neutral_q > 1)).sum(2),
    )


def _zero_gap_diagnostics(tracker, result, reason, coefficient):
    scores = result[1]
    tracker.last_gap_fit = dict(
        active=False,
        reason=reason,
        coefficient=coefficient,
        base_score=scores.copy(),
        owner_loss=np.zeros_like(scores),
        gap_loss=np.zeros_like(scores),
        total_score=scores.copy(),
        original_loss_parts=result[2].copy(),
        iteration_owner_loss=np.empty((0, len(scores))),
        iteration_gap_loss=np.empty((0, len(scores))),
        iteration_owner_gradient_norm=np.empty((0, len(scores))),
        iteration_gap_gradient_norm=np.empty((0, len(scores))),
        iteration_base_score=np.empty((0, len(scores))),
        iteration_total_score=np.empty((0, len(scores))),
        best_iteration=np.full(len(scores), -1, np.int64),
        target_diagnostics_computed=False,
        target_mass=np.empty(0, np.float64),
        final_gap_by_view_body=np.empty((0, 3, 3)),
        target_mass_by_view_body=np.empty((0, 3, 3)),
        positive_gap_pixel_count=np.empty((0, 3, 3), np.int64),
    )


def fit_with_gap(tracker, proposals, states, rays, targets, target, *, coefficient=3.0):
    """Same original proposal pool; T[B,V,N,3] is precomputed from parent anchors."""
    require(
        not isinstance(coefficient, (bool, np.bool_)) and coefficient in (0.0, 3.0),
        "coefficient must be fixed3 or the zero qualification control",
    )
    if coefficient == 0:
        result = frozen_fit(tracker, proposals, states, rays, targets)
        _zero_gap_diagnostics(tracker, result, "coefficient_zero", coefficient)
        return result
    require(targets is not None, "owner-plus-gap experiment requires original D2 PixelTargets")
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
        result = frozen_fit(tracker, proposals, states, rays, targets)
        _zero_gap_diagnostics(tracker, result, "zero_effective_target", coefficient)
        return result

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
    renderer = RefractiveFishRenderer(
        tracker.rig,
        targets.pixels,
        RendererConfig(width_ratio=cfg.width_ratio),
        device=tracker.device,
        dtype=tracker.dtype,
    )
    foreground, base_weight = tensor(targets.target), tensor(targets.weights)
    owner_target = tensor(target.copy()).detach()
    weight = base_weight * renderer.ray_valid
    if not bool((owner_target * (weight * foreground)[None, :, :, None]).any()):
        result = frozen_fit(tracker, proposals, states, rays, targets)
        _zero_gap_diagnostics(tracker, result, "zero_valid_ray_target", coefficient)
        return result
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
        owner_gradient = torch.autograd.grad(owner.sum(), pose, retain_graph=True)[0]
        gap_gradient = torch.autograd.grad(gap.sum(), pose)[0]
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
        active=True,
        reason="fixed_transport_owner_plus_gap",
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
        target_diagnostics_computed=True,
        **{k: v.cpu().numpy().copy() for k, v in details.items()},
    )
    return (
        best_pose.cpu().numpy(),
        scores.cpu().numpy(),
        parts.cpu().numpy(),
        support.cpu().numpy(),
    )

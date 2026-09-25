"""Move the frozen dynamic owner+gap term to immutable admitted body points.

The five original losses, proposals, 14+initial update budget, strict first best,
CUDA base graph and final old-pixel support are unchanged. Point weights already
contain the old per-view/body mass divided by the original single-point count.
No node correspondence, extra foreground factor or validity renormalization is
introduced. No data/model/device execution is initiated by importing this file.
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
from mask_transport_fitter import frozen_fit  # noqa: E402
from mask_transport_gap_fitter import _zero_gap_diagnostics, require  # noqa: E402
from node_geometry import assigned_rays  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402


def body_point_loss(ownership, q, point_weight, ray_valid, *, return_details=False):
    """Separate owner and gap costs on body rays; all masses use fixed /3.

    ownership/q/point_weight: B,V,Q,3, validity: V,Q. point_weight is detached
    nonnegative evidence, not a probability, so no upper bound 1 is imposed.
    Invalid rays are neutralized before either nonlinear operation. Valid
    nonfinite values are rejected even if their observation has zero mass.
    """
    require(
        isinstance(ownership, torch.Tensor)
        and ownership.ndim == 4
        and ownership.dtype in (torch.float32, torch.float64),
        "floating point ownership B,V,Q,3 required",
    )
    b, v, n, bodies = ownership.shape
    require(b >= 1 and v == 3 and n >= 1 and bodies == 3, "body-point axes differ")
    for name, value in (("q", q), ("point_weight", point_weight)):
        require(
            isinstance(value, torch.Tensor)
            and value.shape == ownership.shape
            and value.dtype == ownership.dtype
            and value.device == ownership.device,
            "body-point " + name + " tensor schema differs",
        )
    require(
        isinstance(ray_valid, torch.Tensor)
        and ray_valid.dtype == torch.bool
        and ray_valid.shape == (v, n)
        and ray_valid.device == ownership.device,
        "body-point ray validity schema differs",
    )
    require(
        not point_weight.requires_grad
        and bool(torch.isfinite(point_weight).all())
        and bool((point_weight >= 0).all())
        and bool(((point_weight > 0).sum(-1) <= 1).all()),
        "body-point weights must be detached finite nonnegative exclusive evidence",
    )
    valid = ray_valid[None, :, :, None].expand_as(q)
    require(
        bool(torch.isfinite(ownership[valid]).all())
        and bool(torch.isfinite(q[valid]).all())
        and bool((q[valid] >= 0).all()),
        "valid body-point ownership/distance must be finite with nonnegative q",
    )
    safe_owner = torch.where(valid, ownership, torch.zeros_like(ownership))
    safe_q = torch.where(valid, q, torch.zeros_like(q))
    mass = point_weight * valid
    owner_contributions = mass * (1 - safe_owner).square()
    gap_contributions = mass * (torch.sqrt(safe_q.clamp_min(1)) - 1)
    owner = owner_contributions.sum((1, 2, 3)) / 3
    gap = gap_contributions.sum((1, 2, 3)) / 3
    if not return_details:
        return owner, gap
    return (
        owner,
        gap,
        dict(
            final_gap_by_view_body=gap_contributions.sum(2) / 3,
            target_mass_by_view_body=mass.sum(2) / 3,
            positive_gap_pixel_count=((mass > 0) & (safe_q > 1)).sum(2),
        ),
    )


def _packet_inputs(packet, candidate_anchor, batch_size):
    """Validate numerical boundary and own the selected immutable evidence."""
    require(packet is not None, "body-point packet required for nonzero coefficient")
    raw_weight = packet.point_weight
    require(
        isinstance(raw_weight, np.ndarray)
        and raw_weight.dtype == np.float64
        and raw_weight.ndim == 4
        and raw_weight.shape[0] >= 1
        and raw_weight.shape[1] == 3
        and raw_weight.shape[-1] == 3
        and np.isfinite(raw_weight).all()
        and (raw_weight >= 0).all()
        and ((raw_weight > 0).sum(-1) <= 1).all(),
        "all-anchor body-point weight schema differs",
    )
    a, _, n, _ = raw_weight.shape
    require(
        isinstance(packet.anchor_ids, np.ndarray)
        and packet.anchor_ids.dtype == np.int64
        and np.array_equal(packet.anchor_ids, np.arange(a, dtype=np.int64)),
        "body-point anchor order differs",
    )
    require(
        isinstance(candidate_anchor, np.ndarray)
        and candidate_anchor.dtype == np.int64
        and candidate_anchor.shape == (batch_size,)
        and ((candidate_anchor >= 0) & (candidate_anchor < a)).all(),
        "candidate parent anchor vector differs",
    )
    pixels, counts = packet.point_pixels, packet.query_counts
    require(
        isinstance(pixels, np.ndarray)
        and pixels.dtype == np.float64
        and pixels.shape == (3, n, 2)
        and isinstance(counts, np.ndarray)
        and counts.dtype == np.int64
        and counts.shape == (3,)
        and ((counts >= 0) & (counts <= n)).all()
        and int(counts.max()) == n,
        "body-point coordinate/count schema differs",
    )
    for name, shape, dtype in (
        ("ray_origins", (3, n, 3), np.float64),
        ("ray_directions", (3, n, 3), np.float64),
        ("ray_valid", (3, n), np.bool_),
    ):
        value = getattr(packet, name)
        require(
            isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype,
            "body-point packet " + name + " schema differs",
        )
        if dtype != np.bool_:
            require(np.isfinite(value).all(), "body-point packet nonfinite " + name)
    for vi, count in enumerate(counts):
        require(
            np.isnan(pixels[vi, count:]).all()
            and not packet.ray_valid[vi, count:].any()
            and not raw_weight[:, vi, count:].any(),
            "body-point padding carries observation",
        )
    return pixels.copy(), raw_weight[candidate_anchor].copy(), counts.copy()


def fit_with_body_points(
    tracker, proposals, states, rays, targets, packet, *, candidate_anchor=None, coefficient=3.0
):
    """Fit the old base on old pixels, and owner+gap on fixed body query rays.

    packet has all causal anchors. candidate_anchor gathers its body axes without
    consulting proposal permutations, positions, losses or fitted winners.
    """
    require(
        not isinstance(coefficient, (bool, np.bool_)) and coefficient in (0.0, 3.0),
        "coefficient must be fixed3 or the zero qualification control",
    )
    if coefficient == 0:
        result = frozen_fit(tracker, proposals, states, rays, targets)
        _zero_gap_diagnostics(tracker, result, "coefficient_zero", coefficient)
        return result
    require(targets is not None, "body-point experiment requires original D2 PixelTargets")
    point_pixels, selected_weight, query_counts = _packet_inputs(
        packet, candidate_anchor, len(proposals)
    )
    if not np.any(selected_weight):
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
    weight = base_weight * renderer.ray_valid
    point_renderer = RefractiveFishRenderer(
        tracker.rig,
        {v: point_pixels[i, : query_counts[i]] for i, v in enumerate(tracker.rig.views)},
        RendererConfig(width_ratio=cfg.width_ratio),
        device=tracker.device,
        dtype=tracker.dtype,
    )
    for name in ("ray_origins", "ray_directions", "ray_valid"):
        expected = getattr(packet, name)
        actual = getattr(point_renderer, name)
        expected_tensor = torch.as_tensor(expected.copy(), device=actual.device, dtype=actual.dtype)
        require(torch.equal(actual, expected_tensor), "point renderer packet " + name + " differs")
    point_weight = tensor(selected_weight.copy()).detach()
    if not bool((point_weight * point_renderer.ray_valid[None, :, :, None]).any()):
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
        point_rendering = point_renderer.render(pose)
        owner, gap = body_point_loss(
            point_rendering.ownership,
            point_rendering.per_fish_mahalanobis_sq,
            point_weight,
            point_renderer.ray_valid,
        )
        score = base_score + coefficient * (owner + gap)
        if not bool(torch.isfinite(score).all()):
            raise FloatingPointError("nonfinite body-point combined objective")
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
            raise FloatingPointError("nonfinite body-point combined gradient")
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
        point_rendering = point_renderer.render(best_pose)
        owner, gap, details = body_point_loss(
            point_rendering.ownership,
            point_rendering.per_fish_mahalanobis_sq,
            point_weight,
            point_renderer.ray_valid,
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
            raise FloatingPointError("nonfinite body-point fit output")
    tracker.last_gap_fit = dict(
        active=True,
        reason="fixed_body_point_owner_plus_gap",
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

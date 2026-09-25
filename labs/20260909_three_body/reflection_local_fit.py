"""CPU-only local pose comparison; no tracking or identity acceptance policy."""

from types import SimpleNamespace

import numpy as np
import torch
from graph_fit import FitInputs, objective
from node_geometry import assigned_rays, supported_nodes
from renderer import RefractiveFishRenderer, RendererConfig
from tracker import ThreeBodyTracker


class UnionAdapter:
    """Select a fixed mask response for the unchanged diagnostic objective."""

    def __init__(self, renderer, field):
        self.renderer, self.field = renderer, field

    def render(self, pose):
        return SimpleNamespace(union_alpha=getattr(self.renderer.render(pose), self.field))


def fit_local(rig, initial_pose, config, start, plan, predicted, missing, rays, targets, arm):
    """Hold a supplied assignment and priors fixed for all fourteen Adam steps.

    The caller supplies the same start/plan to every arm. All outputs are local
    diagnostics; no result is assigned a biological ID or accepted automatically.
    """
    if config.device != "cpu" or arm not in ("direct", "unit", "fresnel"):
        raise ValueError("local comparison requires CPU and an explicit registered arm")
    tracker = ThreeBodyTracker(rig, initial_pose, config)
    start, predicted = np.asarray(start, float), np.asarray(predicted, float)
    missing, plan = np.asarray(missing), np.asarray(plan)
    if (start.shape != (3, 3, 3) or predicted.shape != start.shape
            or not np.isfinite(start).all() or not np.isfinite(predicted).all()
            or missing.shape != (3,) or not np.issubdtype(missing.dtype, np.integer)
            or np.any(missing < 0) or plan.shape != (3, 3, 3)
            or not np.issubdtype(plan.dtype, np.integer) or targets is None):
        raise ValueError("explicit finite poses, integer gaps and source plan required")
    if any(np.any((p < tracker.lower) | (p > tracker.upper)) for p in (start, predicted)):
        raise ValueError("local start/prediction outside inherited engineering bounds")
    tensor = tracker._tensor
    origin, direction, valid = assigned_rays(rays, plan[None])
    render_config = RendererConfig(width_ratio=config.width_ratio)
    if arm == "direct":
        renderer = RefractiveFishRenderer(rig, targets.pixels, render_config,
                                         device="cpu", dtype=tracker.dtype)
        calibrated_valid = renderer.ray_valid
    else:
        from reflection_renderer import ReflectionFishRenderer

        reflected = ReflectionFishRenderer(rig, targets.pixels, render_config,
                                            direct_mode="unbounded", device="cpu",
                                            dtype=tracker.dtype)
        calibrated_valid = reflected.ray_valid
        renderer = UnionAdapter(reflected, arm + "_union")
    weights = tensor(targets.weights) * calibrated_valid
    node_weight = tensor([1., 1., .35])[None, None, None]
    lengths = tensor(tracker.lengths)[None]
    inputs = FitInputs(
        tensor(origin), tensor(direction), tensor(valid), tensor(predicted[None]),
        config.motion_sigma_mm + .3 * torch.clamp(tensor(missing[None]), max=20),
        lengths, torch.clamp(.2 * lengths, min=1.), node_weight,
        tensor(targets.target), weights,
    )
    denominator = 9 * float(node_weight.sum())

    def evaluate(value, final=False):
        return objective(value, config, inputs, denominator, renderer,
                         diagnostics=final or arm != "direct")

    pose = tensor(start[None]).clone().requires_grad_(True)
    lower, upper = tensor(tracker.lower), tensor(tracker.upper)
    optimizer = torch.optim.Adam([pose], lr=config.learning_rate_mm)
    best_pose = pose.detach().clone()
    best_score = torch.full((1,), float("inf"), device="cpu", dtype=tracker.dtype)
    iterations, iteration_parts, iteration_poses, best_step = [], [], [], 0
    for step in range(config.opt_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        score, parts, _ = evaluate(pose)
        if not bool(torch.isfinite(score).all()):
            raise FloatingPointError("nonfinite local reflection objective")
        iterations.append(float(score.detach()[0]))
        iteration_parts.append(parts.detach().cpu().numpy()[0].copy())
        iteration_poses.append(pose.detach().cpu().numpy()[0].copy())
        improved = score.detach() < best_score
        if bool(improved[0]):
            best_step = step
        best_pose[improved] = pose.detach()[improved]
        best_score = torch.minimum(best_score, score.detach())
        if step == config.opt_steps:
            break
        score.sum().backward()
        if pose.grad is None or not bool(torch.isfinite(pose.grad).all()):
            raise FloatingPointError("nonfinite local reflection gradient")
        optimizer.step()
        with torch.no_grad():
            pose.clamp_(lower, upper)
    with torch.no_grad():
        score, parts, _ = evaluate(best_pose, final=True)
    fitted = best_pose.numpy()[0].copy()
    sources, observed, _ = supported_nodes(fitted, plan, rays, rig)
    if tracker.initialize_count != 0:
        raise RuntimeError("local optimization unexpectedly initialized a track")
    if float(score[0]) > iterations[0] + 1e-10:
        raise RuntimeError("local fitter discarded its better initial pose")
    return dict(
        arm=arm, input_pose_xyz=start.copy(), plan=plan.copy(), pose_xyz=fitted,
        score=float(score[0]), loss_parts=parts.numpy()[0].copy(),
        iteration_scores=np.asarray(iterations), iteration_loss_parts=np.stack(iteration_parts),
        iteration_pose_xyz=np.stack(iteration_poses),
        best_step=best_step, sources=sources, observed=observed,
        boundary_coordinates=(fitted <= tracker.lower) | (fitted >= tracker.upper),
        tracking_initializations=0, identity_updated=False,
    )

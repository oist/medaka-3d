"""Local temporal refinement of an uncertain scene, with six boundary matchings.

Boundary coordinates come from the same forward estimate. They are not truth or
new external identity seeds. This module returns proposals and costs; it cannot
clear review flags or certify identity from an optimization margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

import numpy as np
import torch
from node_geometry import assigned_rays, assignment_plans
from renderer import RendererConfig
from support_batch import supported_nodes_batch
from temporal_renderer import TemporalUnionRenderer


@dataclass(frozen=True)
class RefinementConfig:
    mode: str = "detectron2"
    device: str = "cuda"
    steps: int = 100
    reassign_every: int = 10
    learning_rate_mm: float = 0.2
    ray_sigma_mm: float = 0.7
    missing_cost: float = 9.0
    mask_weight: float = 3.0
    shape_weight: float = 0.3
    acceleration_weight: float = 0.2
    acceleration_sigma_mm: float = 0.6
    speed_weight: float = 0.05
    speed_scale_mm: float = 3.0
    repulsion_weight: float = 0.15
    width_ratio: float = 0.06

    def __post_init__(self):
        if self.mode not in ("detectron2", "top_contrast", "nodes"):
            raise ValueError("unknown image observation mode")
        for name in ("steps", "reassign_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "learning_rate_mm",
            "ray_sigma_mm",
            "missing_cost",
            "acceleration_sigma_mm",
            "speed_scale_mm",
            "width_ratio",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "mask_weight",
            "shape_weight",
            "acceleration_weight",
            "speed_weight",
            "repulsion_weight",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def boundary_starts(forward_pose, forward_observed):
    """Every permutation maps left identity -> original right-boundary slot."""
    forward = np.asarray(forward_pose, float)
    observed = np.asarray(forward_observed)
    if forward.ndim != 4 or forward.shape[1:] != (3, 3, 3) or not 5 <= len(forward) <= 50:
        raise ValueError("need between five and fifty native three-body poses")
    if observed.dtype != bool:
        raise ValueError("forward observation support must be boolean")
    if observed.shape != forward.shape[:-1] or not np.isfinite(forward).all():
        raise ValueError("inconsistent finite forward pose/observation arrays")
    if np.any(forward < [0.0, 0.0, -6.0]) or np.any(forward > [192.0, 192.0, 186.0]):
        raise ValueError("forward pose lies outside the registered engineering bounds")
    boundary_times = np.array([0, 1, len(forward) - 2, len(forward) - 1])
    if not observed[boundary_times, :, :2].all():
        raise ValueError("both ends need two supported mouth/pectoral frames for all fish")
    all_permutations = np.asarray(list(permutations(range(3))), int)
    starts, indexes, kinds, boundary_values, boundary_masks = [], [], [], [], []
    fraction = np.linspace(0, 1, len(forward))[:, None, None, None]
    for permutation in all_permutations:
        linear = (1 - fraction) * forward[0] + fraction * forward[-1, permutation]
        scene = np.empty_like(forward)
        for t in range(len(forward)):
            distances = ((linear[t, :, None, 1] - forward[t, None, :, 1]) ** 2).sum(-1)
            costs = distances[np.arange(3)[None], all_permutations].sum(-1)
            physical = all_permutations[np.argmin(costs)]
            scene[t] = forward[t, physical]
            held = observed[t, physical, 1]
            scene[t, ~held] = linear[t, ~held]
        values = np.concatenate((forward[:2], forward[-2:, permutation]), axis=0)
        masks = np.concatenate((observed[:2], observed[-2:, permutation]), axis=0)
        for kind, value in (("linear", linear), ("scene", scene)):
            value[boundary_times] = np.where(masks[..., None], values, value[boundary_times])
            starts.append(value)
            indexes.append(permutation)
            kinds.append(kind)
            boundary_values.append(values)
            boundary_masks.append(masks)
    return (
        np.stack(starts),
        np.stack(indexes),
        np.asarray(kinds),
        boundary_times,
        np.stack(boundary_values),
        np.stack(boundary_masks),
    )


class TemporalObjective:
    """The same fixed image samples and assignment rule evaluate every start."""

    def __init__(self, rig, rays, targets, initial_pose_xyz, config):
        self.rig, self.rays, self.config = rig, rays, config
        self.n_frames = len(rays)
        self.device = torch.device(config.device)
        self.dtype = torch.float64
        if len(targets) != self.n_frames:
            raise ValueError("one independent image target is required per native frame")
        initial = np.asarray(initial_pose_xyz, float)
        if initial.shape != (3, 3, 3) or not np.isfinite(initial).all():
            raise ValueError("initial anatomy must have three finite three-node fish")
        self.lengths = self.tensor(np.linalg.norm(np.diff(initial, axis=1), axis=-1))
        if self.lengths.shape != (3, 2) or not bool((self.lengths > 0.1).all()):
            raise ValueError("initial anatomy must contain six nondegenerate segments")
        self.length_sigma = torch.clamp(self.lengths * 0.2, min=1.0)
        self.node_weights = self.tensor([1.0, 1.0, 0.35])[None, None, None, None]
        self.renderer = None
        if config.mode != "nodes":
            self.renderer = TemporalUnionRenderer(
                rig,
                [t.pixels for t in targets],
                RendererConfig(width_ratio=config.width_ratio),
                device=self.device,
                dtype=self.dtype,
                time_chunk_size=5,
            )
            self.target = self.tensor(np.stack([t.target for t in targets]))
            self.weight = self.tensor(np.stack([t.weights for t in targets]))
            self.weight = self.weight * self.renderer.ray_valid

    def tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def assignments(self, pose):
        arrays = pose.detach().cpu().numpy() if isinstance(pose, torch.Tensor) else pose
        cfg = self.config
        plans = np.stack(
            [
                np.stack(
                    [
                        assignment_plans(p, r, cfg.ray_sigma_mm, cfg.missing_cost, keep=1)[0][1]
                        for p, r in zip(value, self.rays, strict=True)
                    ]
                )
                for value in arrays
            ]
        )
        return plans

    def ray_tensors(self, plans):
        origins, directions, valid = [], [], []
        for t, rays in enumerate(self.rays):
            origin, direction, held = assigned_rays(rays, plans[:, t])
            origins.append(origin)
            directions.append(direction)
            valid.append(held)
        return tuple(
            self.tensor(np.stack(values, axis=1)) for values in (origins, directions, valid)
        )

    def __call__(self, pose, ray_tensors):
        cfg = self.config
        origin, direction, held = ray_tensors
        delta = pose[:, :, :, None] - origin
        depth = torch.clamp((delta * direction).sum(-1, keepdim=True), min=0)
        ray_sq = ((delta - depth * direction) ** 2).sum(-1) / cfg.ray_sigma_mm**2
        robust = torch.where(ray_sq <= 9, ray_sq, 6 * torch.sqrt(ray_sq + 1e-12) - 9)
        node = ((held * robust + (1 - held) * cfg.missing_cost) * self.node_weights).sum(
            (1, 2, 3, 4)
        ) / (self.n_frames * 9 * 2.35)
        lengths = torch.linalg.vector_norm(torch.diff(pose, dim=3), dim=-1)
        shape = (((lengths - self.lengths) / self.length_sigma) ** 2).mean((1, 2, 3))
        velocity = torch.diff(pose[:, :, :, 1], dim=1)
        acceleration_sq = (torch.diff(velocity, dim=1) / cfg.acceleration_sigma_mm).square().sum(-1)
        acceleration = (2 * (torch.sqrt(1 + acceleration_sq) - 1)).mean((1, 2))
        speed = (velocity.square().sum(-1) / cfg.speed_scale_mm**2).mean((1, 2))
        pair_sq = torch.stack(
            [
                ((pose[:, :, a, 1] - pose[:, :, b, 1]) ** 2).sum(-1)
                for a, b in combinations(range(3), 2)
            ],
            dim=-1,
        )
        repulsion = torch.exp(-pair_sq / 8).mean((1, 2))
        image = torch.zeros_like(node)
        if self.renderer is not None:
            union = self.renderer.render(pose)
            image = ((union - self.target).square() * self.weight).sum((1, 2, 3)) / (
                self.n_frames * 3
            )
        components = torch.stack((node, shape, acceleration, speed, repulsion, image), dim=-1)
        cost = (
            node
            + cfg.shape_weight * shape
            + cfg.acceleration_weight * acceleration
            + cfg.speed_weight * speed
            + cfg.repulsion_weight * repulsion
            + cfg.mask_weight * image
        )
        return cost, components


def refine_window(
    rig, forward_pose, forward_observed, rays, targets, initial_pose_xyz, config=None
):
    return optimize_starts(
        rig,
        forward_pose,
        forward_observed,
        rays,
        targets,
        initial_pose_xyz,
        boundary_starts(forward_pose, forward_observed),
        config,
    )


def optimize_starts(
    rig, forward_pose, forward_observed, rays, targets, initial_pose_xyz, starts, config=None
):
    """Optimize explicitly supplied alternatives under the unchanged v2 objective.

    ``starts`` follows boundary_starts' six-array schema. This extension admits
    separately registered image-derived starts; it does not modify the original
    twelve starts, boundary meanings, or observation/identity admission rules.
    """
    cfg = config or RefinementConfig()
    initial, right_permutations, kinds, boundary_times, values, held = starts
    initial = np.asarray(initial, dtype=float)
    right_permutations = np.asarray(right_permutations)
    kinds, boundary_times = np.asarray(kinds), np.asarray(boundary_times)
    values, held = np.asarray(values), np.asarray(held)
    if (
        initial.ndim != 5
        or initial.shape[2:] != (3, 3, 3)
        or not 1 <= len(initial) <= 24
        or not 5 <= initial.shape[1] <= 50
        or not np.isfinite(initial).all()
    ):
        raise ValueError("need 1..24 finite starts with 5..50 native three-body frames")
    count, n_frames = initial.shape[:2]
    if (
        right_permutations.shape != (count, 3)
        or not np.issubdtype(right_permutations.dtype, np.integer)
        or not np.array_equal(
            np.sort(right_permutations, axis=1), np.tile(np.arange(3), (count, 1))
        )
        or kinds.shape != (count,)
        or kinds.dtype.kind not in "US"
        or not np.array_equal(boundary_times, [0, 1, n_frames - 2, n_frames - 1])
        or not np.issubdtype(boundary_times.dtype, np.integer)
        or values.shape != (count, 4, 3, 3, 3)
        or not np.isfinite(values).all()
        or held.shape != (count, 4, 3, 3)
        or held.dtype != bool
        or not np.array_equal(initial[:, boundary_times][held], values[held])
    ):
        raise ValueError("inconsistent start labels or exactly held boundary constraints")
    _, canonical_permutations, _, canonical_times, canonical_values, canonical_masks = (
        boundary_starts(forward_pose, forward_observed)
    )
    indexes = [
        np.flatnonzero(np.all(canonical_permutations == permutation, axis=1))[0]
        for permutation in right_permutations
    ]
    if (
        not np.array_equal(boundary_times, canonical_times)
        or not np.array_equal(values, canonical_values[indexes])
        or not np.array_equal(held, canonical_masks[indexes])
    ):
        raise ValueError("new starts must preserve the original forward boundary constraints")
    if len(rays) != initial.shape[1] or len(rays) > 50:
        raise ValueError("refinement is bounded to at most 50 complete native frames")
    objective = TemporalObjective(rig, rays, targets, initial_pose_xyz, cfg)
    pose = objective.tensor(initial).clone().requires_grad_(True)
    fixed_values = objective.tensor(values)
    fixed_mask = torch.as_tensor(held[..., None], device=objective.device)
    lower, upper = objective.tensor([0.0, 0.0, -6.0]), objective.tensor([192.0, 192.0, 186.0])
    if bool(((pose < lower) | (pose > upper)).any()):
        raise ValueError("a refinement start lies outside the same forward engineering bounds")
    optimizer = torch.optim.Adam([pose], lr=cfg.learning_rate_mm)
    best_pose, best_score = pose.detach().clone(), None
    trace = []
    for step in range(cfg.steps + 1):
        if step % cfg.reassign_every == 0 or step == cfg.steps:
            plans = objective.assignments(pose)
            ray_tensors = objective.ray_tensors(plans)
        optimizer.zero_grad(set_to_none=True)
        cost, _ = objective(pose, ray_tensors)
        if not bool(torch.isfinite(cost).all()):
            raise FloatingPointError("nonfinite temporal objective")
        # Compare checkpoints only after recomputing each one's own assignments.
        if step % cfg.reassign_every == 0 or step == cfg.steps:
            with torch.no_grad():
                if best_score is None:
                    best_score = cost.detach().clone()
                else:
                    old_plans = objective.assignments(best_pose)
                    old_score, _ = objective(best_pose, objective.ray_tensors(old_plans))
                    improved = cost.detach() < old_score
                    best_pose[improved] = pose.detach()[improved]
                    best_score = torch.minimum(old_score, cost.detach())
                trace.append(best_score.cpu().numpy().copy())
            print(
                f"temporal_step={step}/{cfg.steps} min_cost={float(best_score.min()):.6f}",
                flush=True,
            )
        if step == cfg.steps:
            break
        cost.sum().backward()
        if pose.grad is None or not bool(torch.isfinite(pose.grad).all()):
            raise FloatingPointError("nonfinite temporal gradient")
        optimizer.step()
        with torch.no_grad():
            pose.clamp_(lower, upper)
            pose[:, boundary_times] = torch.where(fixed_mask, fixed_values, pose[:, boundary_times])
    # Final values are recomputed at the actual retained poses, with fresh rows.
    final_pose = best_pose.cpu().numpy()
    plans = objective.assignments(final_pose)
    with torch.no_grad():
        final_cost, components = objective(best_pose, objective.ray_tensors(plans))
        original = objective.tensor(np.asarray(forward_pose)[None])
        original_plans = objective.assignments(original)
        original_cost, original_parts = objective(original, objective.ray_tensors(original_plans))
    sources, observed = [], []
    for t, ray in enumerate(rays):
        source, support, _ = supported_nodes_batch(final_pose[:, t], plans[:, t], ray, rig)
        sources.append(source)
        observed.append(support)
    boundary_error = np.abs(final_pose[:, boundary_times] - values)
    boundary_error = np.where(held[..., None], boundary_error, 0.0)
    return dict(
        pose_xyz=final_pose,
        sources=np.stack(sources, axis=1),
        observed=np.stack(observed, axis=1),
        costs=final_cost.cpu().numpy(),
        loss_parts=components.cpu().numpy(),
        permutations=right_permutations,
        start_kind=kinds,
        checkpoint_costs=np.stack(trace),
        boundary_max_error_mm=boundary_error.max(axis=(1, 2, 3, 4)),
        original_cost=original_cost.cpu().numpy(),
        original_loss_parts=original_parts.cpu().numpy(),
    )

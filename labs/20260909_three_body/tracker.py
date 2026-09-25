"""Joint three-body analysis by synthesis with a finite temporal hypothesis beam.

The masks constrain the unordered three-body scene. Identity is carried only by
temporal correspondences and initial shape, never by a mask instance index.
Costs and margins are engineering scores, not calibrated identity probabilities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations, permutations

import numpy as np
import torch
from fast_union import render_union
from node_geometry import (
    RayObservations,
    assigned_rays,
    assignment_plans,
    node_pools,
    recovery_poses,
    supported_nodes,
)
from observations import sample_pixels
from renderer import RefractiveFishRenderer, RendererConfig

from medaka3d.types import VIEWS


@dataclass(frozen=True)
class TrackerConfig:
    mode: str = "detectron2"
    device: str = "cuda"
    beam_size: int = 6
    max_hypotheses: int = 18
    opt_steps: int = 14
    learning_rate_mm: float = 0.25
    ray_sigma_mm: float = 0.7
    missing_cost: float = 9.0
    mask_weight: float = 3.0
    motion_weight: float = 0.2
    shape_weight: float = 0.3
    repulsion_weight: float = 0.15
    width_ratio: float = 0.06
    motion_sigma_mm: float = 0.8
    swap_distance_mm: float = 15.0
    ambiguity_margin: float = 1.0
    review_gap_frames: int = 4
    positive_samples: int = 384
    near_samples: int = 384
    global_samples: int = 256
    flow_proposals: bool = False

    def __post_init__(self):
        if type(self.flow_proposals) is not bool:
            raise ValueError("flow_proposals must be boolean")
        if self.mode not in ("nodes", "detectron2", "top_contrast"):
            raise ValueError("unknown observation mode")
        for name in (
            "beam_size",
            "max_hypotheses",
            "opt_steps",
            "review_gap_frames",
            "positive_samples",
            "near_samples",
            "global_samples",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_hypotheses < self.beam_size:
            raise ValueError("max_hypotheses must accommodate each retained parent")
        for name in (
            "learning_rate_mm",
            "ray_sigma_mm",
            "missing_cost",
            "width_ratio",
            "motion_sigma_mm",
            "swap_distance_mm",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "mask_weight",
            "motion_weight",
            "shape_weight",
            "repulsion_weight",
            "ambiguity_margin",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass
class State:
    pose: np.ndarray
    velocity: np.ndarray
    missing: np.ndarray
    review: np.ndarray
    cost: float


@dataclass
class Proposal:
    pose: np.ndarray
    predicted: np.ndarray
    plan: np.ndarray
    parent: int
    kind: str
    permutation: tuple[int, int, int] = (0, 1, 2)


class ThreeBodyTracker:
    """One initialization, joint fitting, exclusive detections and offline backtrace.

    No callback or argument supplies future baseline positions or identity truth.
    The finite beam is an approximation to global inference, not its exact solution.
    Raw latent poses, image support and direct triangulation support remain distinct.
    Identity review propagates after unresolved evidence loss or a competitive
    whole-body permutation; a later good mask does not silently certify that ID.
    """

    def __init__(self, rig, initial_pose_xyz, config=None):
        self.rig = rig
        if tuple(rig.views) != tuple(VIEWS):
            raise ValueError("rig view order must match the explicit observation view order")
        self.config = config or TrackerConfig()
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; choose CPU explicitly")
        self.dtype = torch.float64
        self.lower = np.array([0.0, 0.0, -6.0])
        self.upper = np.array([192.0, 192.0, 186.0])
        pose = np.asarray(initial_pose_xyz, float)
        if pose.shape != (3, 3, 3) or not np.isfinite(pose).all():
            raise ValueError("initial pose must be three finite three-node fish")
        if np.any(pose < self.lower) or np.any(pose > self.upper):
            raise ValueError("initial pose outside registered engineering bounds")
        self.initial = pose.copy()
        self.lengths = np.linalg.norm(np.diff(pose, axis=1), axis=-1)
        if np.any(self.lengths < 0.1):
            raise ValueError("degenerate initial body segment")
        self.initialize_count = 0

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def _proposals(self, states, rays, flow=None):
        """Reserve each parent, then coherent swaps and observation recovery.

        A one-node runner-up cannot crowd all whole-fish ID alternatives out of
        the proposal set. The same ID's motion prior is retained for a swap.
        """
        cfg = self.config
        proposals, secondary = [], []
        for parent, state in enumerate(states):
            predicted = np.clip(state.pose + state.velocity[:, None], self.lower, self.upper)
            plans = assignment_plans(predicted, rays, cfg.ray_sigma_mm, cfg.missing_cost, keep=2)
            proposals.append(Proposal(predicted, predicted, plans[0][1], parent, "predict"))
            if len(plans) > 1:
                secondary.append(
                    Proposal(predicted, predicted, plans[1][1], parent, "node_alternative")
                )
        main = proposals[0]
        if flow is not None and flow.node_valid.any():
            # Flow is an image-derived candidate, not a replacement for the
            # temporal prior or an independent direct anatomical observation.
            shifted, warped = main.pose.copy(), main.pose.copy()
            for fish in range(3):
                usable = flow.node_valid[fish]
                if not usable.any():
                    continue
                displacement = np.median(
                    flow.pose_xyz[fish, usable] - states[0].pose[fish, usable], axis=0
                )
                rigid = states[0].pose[fish] + displacement
                flexible = rigid.copy()
                flexible[usable] = flow.pose_xyz[fish, usable]
                if np.all(rigid >= self.lower) and np.all(rigid <= self.upper):
                    shifted[fish] = rigid
                if np.all(flexible >= self.lower) and np.all(flexible <= self.upper):
                    warped[fish] = flexible
            for kind, pose in (("image_flow_nodes", warped), ("image_flow_translation", shifted)):
                if np.max(np.abs(pose - main.pose)) < 1e-6:
                    continue
                plan = assignment_plans(pose, rays, cfg.ray_sigma_mm, cfg.missing_cost, keep=1)[0][
                    1
                ]
                proposals.append(Proposal(pose, main.predicted, plan, 0, kind))
        for first, second in combinations(range(3), 2):
            distance = np.linalg.norm(main.pose[first, 1] - main.pose[second, 1])
            if distance > cfg.swap_distance_mm and not np.any(states[0].missing[[first, second]]):
                continue
            permutation = np.arange(3)
            permutation[[first, second]] = [second, first]
            proposals.append(
                Proposal(
                    main.pose[permutation].copy(),
                    main.predicted,
                    main.plan[permutation].copy(),
                    0,
                    "body_permutation",
                    tuple(permutation.tolist()),
                )
            )
        if np.any(states[0].missing):
            pools = node_pools(self.rig, rays, self.lower, self.upper)
            candidates = recovery_poses(
                main.pose,
                states[0].missing,
                self.lengths,
                pools,
                main.plan,
                self.lower,
                self.upper,
                keep=3,
            )
            for _, pose, _ in candidates:
                plan = assignment_plans(pose, rays, cfg.ray_sigma_mm, cfg.missing_cost, keep=1)[0][
                    1
                ]
                proposals.append(Proposal(pose, main.predicted, plan, 0, "observation_recovery"))
        return (proposals + secondary)[: cfg.max_hypotheses]

    def _fit(self, proposals, states, rays, targets):
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
        renderer = None
        if targets is not None:
            renderer = RefractiveFishRenderer(
                self.rig,
                targets.pixels,
                RendererConfig(width_ratio=cfg.width_ratio),
                device=self.device,
                dtype=self.dtype,
            )
            target = self._tensor(targets.target)
            weight = self._tensor(targets.weights) * renderer.ray_valid

        def objective(value, diagnostics=False):
            delta = value[:, :, None] - origins
            depth = torch.clamp((delta * directions).sum(-1, keepdim=True), min=0)
            error_sq = ((delta - depth * directions) ** 2).sum(-1) / cfg.ray_sigma_mm**2
            # Huber squared distance beyond three sigma; absent terms stay explicit.
            robust = torch.where(error_sq <= 9, error_sq, 6 * torch.sqrt(error_sq + 1e-12) - 9)
            node = ((held * robust + (1 - held) * cfg.missing_cost) * node_weight).sum(
                (1, 2, 3)
            ) / denominator
            segment = torch.linalg.vector_norm(torch.diff(value, dim=2), dim=-1)
            shape = (((segment - lengths) / length_sigma) ** 2).mean((1, 2))
            center = (
                (((value[:, :, 1] - predicted[:, :, 1]) / motion_sigma[..., None]) ** 2)
                .sum(-1)
                .mean(-1)
            )
            relative = value - value[:, :, 1:2]
            previous_relative = predicted - predicted[:, :, 1:2]
            orientation = ((relative - previous_relative) ** 2).sum(-1).mean((1, 2)) / 25
            pair_sq = torch.stack(
                [
                    ((value[:, a, 1] - value[:, b, 1]) ** 2).sum(-1)
                    for a, b in combinations(range(3), 2)
                ],
                dim=-1,
            )
            repulsion = torch.exp(-pair_sq / 8).mean(-1)
            silhouette = torch.zeros_like(node)
            rendering = None
            if renderer is not None:
                if diagnostics:
                    rendering = renderer.render(value)
                    union = rendering.union_alpha
                else:
                    union = render_union(renderer, value)
                silhouette = (((union - target) ** 2) * weight).sum((1, 2)) / 3
            total = (
                node
                + cfg.shape_weight * shape
                + cfg.motion_weight * (center + 0.05 * orientation)
                + cfg.repulsion_weight * repulsion
                + cfg.mask_weight * silhouette
            )
            parts = torch.stack((node, shape, center, repulsion, silhouette), dim=-1)
            return total, parts, rendering

        # Retain the best finite iterate per independent batch item, including the
        # initial guess: a fixed Adam schedule must not replace it with a worse fit.
        optimizer = torch.optim.Adam([pose], lr=cfg.learning_rate_mm)
        best_pose = pose.detach().clone()
        best_score = torch.full(
            (len(proposals),), float("inf"), device=self.device, dtype=self.dtype
        )
        for step in range(cfg.opt_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            score, _, _ = objective(pose)
            if not bool(torch.isfinite(score).all()):
                raise FloatingPointError("nonfinite joint body objective")
            improved = score.detach() < best_score
            best_pose[improved] = pose.detach()[improved]
            best_score = torch.minimum(best_score, score.detach())
            if step == cfg.opt_steps:
                break
            score.sum().backward()
            if pose.grad is None or not bool(torch.isfinite(pose.grad).all()):
                raise FloatingPointError("nonfinite joint body gradient")
            optimizer.step()
            with torch.no_grad():
                pose.clamp_(lower, upper)
        with torch.no_grad():
            scores, parts, rendering = objective(best_pose, diagnostics=True)
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
                # A weak engineering bridge, distinct from direct node observation.
                # A bridge requires predominantly owned visible foreground, not
                # an equal-depth split of another body's entire silhouette.
                view_support = (positive >= 0.025) & (positive >= 2 * negative)
                image_support = view_support.sum(1) >= 2
        return (
            best_pose.cpu().numpy(),
            scores.cpu().numpy(),
            parts.cpu().numpy(),
            image_support.cpu().numpy(),
        )

    @staticmethod
    def _different_identity(first, second):
        costs = [
            (sum(np.linalg.norm(first[f, 1] - second[p[f], 1]) ** 2 for f in range(3)), p)
            for p in permutations(range(3))
        ]
        _, best = min(costs)
        return np.asarray(best) != np.arange(3)

    def track(self, frames):
        if self.initialize_count:
            raise RuntimeError("tracker instances are single-use: no hidden reinitialization")
        self.initialize_count = 1
        cfg = self.config
        states = [
            State(self.initial.copy(), np.zeros((3, 3)), np.zeros(3, int), np.zeros(3, bool), 0.0)
        ]
        history, previous_frame = [], None
        previous_observation = None
        if cfg.flow_proposals:
            from motion import FlowObservations
        started = time.perf_counter()
        for index, frame in enumerate(frames):
            if previous_frame is not None and frame.frame_idx != previous_frame + 1:
                raise ValueError("tracking input must be continuous native frames")
            previous_frame = frame.frame_idx
            rays = RayObservations.from_points(self.rig, frame.detections)
            targets = (
                None
                if cfg.mode == "nodes"
                else sample_pixels(
                    frame, cfg.mode, cfg.positive_samples, cfg.near_samples, cfg.global_samples
                )
            )
            flow = None
            if cfg.flow_proposals and previous_observation is not None:
                flow = FlowObservations.from_frames(
                    previous_observation, frame, self.rig, states[0].pose
                )
            proposals = (
                self._proposals(states, rays)
                if flow is None
                else self._proposals(states, rays, flow)
            )
            poses, local_cost, parts, mask_support = self._fit(proposals, states, rays, targets)
            path_cost = local_cost + np.array([states[p.parent].cost for p in proposals])
            order = np.argsort(path_cost, kind="stable")
            selected = []
            for candidate in order:
                if any(
                    proposals[candidate].parent == proposals[j].parent
                    and proposals[candidate].permutation == proposals[j].permutation
                    and np.max(np.linalg.norm(poses[candidate] - poses[j], axis=-1)) < 0.5
                    for j in selected
                ):
                    continue
                selected.append(int(candidate))
                if len(selected) == cfg.beam_size:
                    break
            next_states, records = [], []
            for candidate in selected:
                proposal = proposals[candidate]
                parent = states[proposal.parent]
                pose = poses[candidate]
                sources, observed, _ = supported_nodes(pose, proposal.plan, rays, self.rig)
                has_head_ray = (sources[:, :, :2] >= 0).any(axis=(1, 2))
                body_supported = observed[:, 1] | (mask_support[candidate] & has_head_ray)
                missing = np.where(body_supported, 0, parent.missing + 1)
                review = parent.review | (missing >= cfg.review_gap_frames)
                identity_margin = float("inf")
                for other in order:
                    if other == candidate:
                        continue
                    affected = self._different_identity(pose, poses[other])
                    # At a spatial tie, permutation argmin alone chooses identity
                    # lexically. Preserve explicit same-parent symmetry candidates.
                    if proposals[other].parent == proposal.parent:
                        tied = (
                            np.max(np.linalg.norm(pose[:, 1] - poses[other, :, 1], axis=-1)) < 0.5
                        )
                        if tied:
                            affected |= np.asarray(proposal.permutation) != np.asarray(
                                proposals[other].permutation
                            )
                    if not affected.any():
                        continue
                    gap = float(path_cost[other] - path_cost[candidate])
                    identity_margin = min(identity_margin, gap)
                    if gap <= cfg.ambiguity_margin:
                        review |= affected
                displacement = pose[:, 1] - parent.pose[:, 1]
                velocity = 0.5 * parent.velocity + 0.5 * displacement
                speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
                velocity *= np.minimum(1.0, 2.0 / np.maximum(speed, 1e-12))
                velocity = np.where(body_supported[:, None], velocity, 0.8 * parent.velocity)
                next_states.append(
                    State(pose, velocity, missing, review, float(path_cost[candidate]))
                )
                records.append(
                    dict(
                        frame_idx=frame.frame_idx,
                        pose_xyz=pose,
                        observed=observed,
                        sources=sources,
                        identity_review=review,
                        mask_supported=mask_support[candidate],
                        body_supported=body_supported,
                        missing=missing,
                        costs=local_cost[candidate],
                        path_costs=path_cost[candidate],
                        loss_parts=parts[candidate],
                        score_margin=identity_margin,
                        parent=proposal.parent,
                        proposal_kind=proposal.kind,
                        hypotheses=len(proposals),
                        proposal_flow_nodes=(
                            flow.node_valid if flow is not None else np.zeros((3, 3), bool)
                        ),
                        proposal_flow_parent=(0 if flow is not None else -1),
                    )
                )
            history.append(records)
            states = next_states
            if cfg.flow_proposals:
                previous_observation = frame
            if index % 25 == 0:
                elapsed = time.perf_counter() - started
                direct = records[0]["observed"][:, 1].astype(int).tolist()
                print(
                    f"frame={frame.frame_idx} processed={index + 1} elapsed_s={elapsed:.1f} "
                    f"beam={len(states)} supported={direct} "
                    f"review={records[0]['identity_review'].astype(int).tolist()}",
                    flush=True,
                )
        if not history:
            raise ValueError("empty tracking sequence")
        chosen, state_index = [], 0
        for records in reversed(history):
            record = records[state_index]
            chosen.append(record)
            state_index = record["parent"]
        chosen.reverse()
        result = {
            key: np.asarray([record[key] for record in chosen])
            for key in chosen[0]
            if key != "parent"
        }
        result["xyz"] = np.where(
            result["observed"][:, :, 1, None], result["pose_xyz"][:, :, 1], np.nan
        )
        result["accepted_xyz"] = np.where(
            result["identity_review"][:, :, None], np.nan, result["xyz"]
        )
        result["body_xyz"] = np.where(
            result["body_supported"][:, :, None], result["pose_xyz"][:, :, 1], np.nan
        )
        return result

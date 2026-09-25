"""Causal enrollment and color reranking of the existing 3D fitted candidates.

Geometry fitting is unchanged; the additional image loss ranks its candidates
before beam selection. The first 166 frames enroll colors from current best
states, and templates then freeze. No completed trajectory, old template, future
frame or human annotation supplies an identity or position during tracking.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
from accelerated_tracker import AcceleratedThreeBodyTracker
from appearance_ranking import sample_appearance_pixels
from node_geometry import RayObservations, supported_nodes
from observations import sample_pixels
from online_appearance import OnlineAppearanceAccumulator, OnlineAppearanceConfig
from persistent_node_tracker import (
    PersistentNodeState,
    PersistentNodeThreeBodyTracker,
    updated_ages,
)
from pixel_appearance_loss import PixelAppearanceConfig, pixel_appearance_loss
from renderer import RefractiveFishRenderer, RendererConfig

from medaka3d.types import VIEWS

PERSISTENT_TRACKER_ORIGIN_SHA256 = (
    "841ff2e3a67e174a7ea3205da17368730c47432eff7e8a79cf13000302ea148e"
)


@dataclass(frozen=True)
class AppearanceRankingConfig:
    weight: float = 1.0
    maximum_samples: int = 512
    active_views: tuple[str, ...] = ("top_cam",)

    def __post_init__(self):
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("appearance weight must be finite and nonnegative")
        if type(self.maximum_samples) is not int or self.maximum_samples < 1:
            raise ValueError("maximum_samples must be a positive integer")
        if (not self.active_views or len(set(self.active_views)) != len(self.active_views)
                or any(view not in VIEWS for view in self.active_views)):
            raise ValueError("active_views must be distinct known camera views")


class AppearanceThreeBodyTracker(PersistentNodeThreeBodyTracker):
    """An additional color score, with identical geometric candidate fits.

    Top camera color is the first experimental arm, informed by separate fixed
    appearance diagnostics. All three cameras still supply geometry and masks.
    Body colors are provisional initial-slot templates, not biological ID truth.
    """

    def __init__(self, rig, initial_pose_xyz, config=None, *, recovery_config=None,
                 persistence_config=None, appearance_config=None, enrollment_config=None,
                 pixel_config=None):
        super().__init__(rig, initial_pose_xyz, config, recovery_config=recovery_config,
                         persistence_config=persistence_config)
        self.appearance_config = appearance_config or AppearanceRankingConfig()
        self.enrollment_config = enrollment_config or OnlineAppearanceConfig()
        self.pixel_config = pixel_config or PixelAppearanceConfig()
        self.enrollment = OnlineAppearanceAccumulator(rig, self.enrollment_config)
        self.appearance_scored_frames = 0
        self.appearance_sampled_pixels = np.zeros(3, np.int64)

    def _appearance_scores(self, frame, poses):
        zero = np.zeros(len(poses)), np.zeros(3, bool), np.zeros(3, int)
        if self.appearance_config.weight == 0 or not self.enrollment.frozen:
            return zero
        permitted = np.array([v in self.appearance_config.active_views for v in VIEWS])
        enabled = self.enrollment.enabled_views & permitted
        if not enabled.any():
            return zero
        evidence = sample_appearance_pixels(frame, enabled, self.appearance_config.maximum_samples)
        if not evidence.enabled.any():
            return zero
        renderer = RefractiveFishRenderer(
            self.rig, evidence.pixels, RendererConfig(width_ratio=self.config.width_ratio),
            device=self.device, dtype=self.dtype,
        )
        with torch.no_grad():
            rendering = renderer.render(self._tensor(poses))
            measured = pixel_appearance_loss(
                rendering.visible_alpha, self._tensor(evidence.features),
                self._tensor(self.enrollment.templates),
                self._tensor(evidence.weights) * rendering.ray_valid,
                torch.as_tensor(evidence.enabled, device=self.device), self.pixel_config,
            )
        valid_counts = ((self._tensor(evidence.weights) > 0) & rendering.ray_valid).sum(1)
        counts = valid_counts.cpu().numpy()
        enabled_now = measured.view_enabled.cpu().numpy() & (counts > 0)
        self.appearance_scored_frames += int(enabled_now.any())
        self.appearance_sampled_pixels += counts
        return measured.loss.cpu().numpy(), enabled_now, counts

    def track(self, frames):
        if self.initialize_count:
            raise RuntimeError("tracker instances are single-use: no hidden reinitialization")
        self.initialize_count = 1
        cfg = self.config
        states = [
            PersistentNodeState(self.initial.copy(), np.zeros((3, 3)), np.zeros(3, int),
                                np.zeros(3, bool), 0.0, np.zeros((3, 3), int), np.zeros(3, int))
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
            geometric_cost = local_cost
            appearance_cost, appearance_enabled, appearance_counts = self._appearance_scores(
                frame, poses
            )
            local_cost = geometric_cost + self.appearance_config.weight * appearance_cost
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
                node_missing, direct_pectoral_age = updated_ages(parent, observed)
                next_states.append(
                    PersistentNodeState(
                        pose, velocity, missing, review, float(path_cost[candidate]),
                        node_missing, direct_pectoral_age
                    )
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
                        node_missing=node_missing,
                        direct_pectoral_age=direct_pectoral_age,
                        geometric_cost=geometric_cost[candidate],
                        appearance_cost=appearance_cost[candidate],
                        appearance_enabled=appearance_enabled,
                        appearance_pixels=appearance_counts,
                    )
                )
            history.append(records)
            states = next_states
            enrollment_state = records[0]
            self.enrollment.observe(
                frame, enrollment_state["pose_xyz"], enrollment_state["sources"],
                enrollment_state["observed"], enrollment_state["identity_review"],
            )
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


class AcceleratedAppearanceTracker(AppearanceThreeBodyTracker, AcceleratedThreeBodyTracker):
    """Existing Graph geometry fit plus explicit causal appearance reranking."""

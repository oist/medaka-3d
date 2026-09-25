"""Observation-driven recovery of persistent 3D fish hypotheses.

The first pose is the sole external initialization. Recovery proposals come
from the current camera observations, with a separate uncertain temporal
connection cost. No later baseline paths or identity annotations are inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations, product

import numpy as np
from fast_fit import FastFitTracker
from tracker import PERMUTATIONS, TrackerConfig, TrackResult, _body_samples

from medaka3d.calib import triangulate
from medaka3d.types import VIEWS


@dataclass(frozen=True)
class RecoveryConfig:
    local: TrackerConfig = field(default_factory=TrackerConfig)
    missing_before_recovery: int = 2
    recovery_proposals_per_parent: int = 3
    node_candidate_limit: int = 24
    ray_rms_mm: float = 1.5
    support_px: float = 8.0
    initial_connection_sigma_mm: float = 2.0
    uncertainty_growth_mm_per_frame: float = 1.5
    maximum_connection_sigma_mm: float = 80.0
    recovery_cost: float = 6.0
    recovery_fit_sigma_mm: float = 3.0
    confirmation_frames: int = 3

    def __post_init__(self):
        for name in ("missing_before_recovery", "recovery_proposals_per_parent",
                     "node_candidate_limit", "confirmation_frames"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("ray_rms_mm", "support_px", "initial_connection_sigma_mm",
                     "uncertainty_growth_mm_per_frame", "maximum_connection_sigma_mm",
                     "recovery_cost", "recovery_fit_sigma_mm"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class NodeCandidate:
    xyz: np.ndarray
    source: np.ndarray
    rms: float


@dataclass(frozen=True)
class RecoveryTrackResult(TrackResult):
    appearance: np.ndarray | None = None
    appearance_quality: np.ndarray | None = None
    geometry_review: np.ndarray | None = None
    identity_review: np.ndarray | None = None


def node_candidates(rig, frame, config: RecoveryConfig):
    """Unordered node hypotheses from all two/three-view ray combinations.

    Body-node groupings in SLEAP are not fixed identities. A bad third camera
    cannot remove a good two-camera alternative. Each accepted candidate
    explicitly records which original observations support it.
    """
    result = []
    lower, upper = np.asarray(config.local.lower_mm), np.asarray(config.local.upper_mm)
    for node in range(3):
        points = [np.asarray(frame.detections[v])[:, node] for v in VIEWS]
        rays = [cam.rays(uv) for cam, uv in zip(rig.cameras, points, strict=True)]
        finite = [np.flatnonzero(np.isfinite(p).all(axis=1)) for p in points]
        sources = []
        for count in (3, 2):
            for views in combinations(range(3), count):
                for rows in product(*(finite[v] for v in views)):
                    source = np.full(3, -1, np.int32)
                    source[list(views)] = rows
                    sources.append(source)
        if not sources:
            result.append([])
            continue
        sources = np.asarray(sources)
        origins = np.zeros((len(sources), 3, 3))
        directions = np.zeros_like(origins)
        valid = sources >= 0
        for v in range(3):
            mask = valid[:, v]
            origins[mask, v] = rays[v][0][sources[mask, v]]
            directions[mask, v] = rays[v][1][sources[mask, v]]
        xyz, rms = triangulate(origins, directions, valid)
        keep = (np.isfinite(xyz).all(axis=1) & (rms <= config.ray_rms_mm)
                & (xyz >= lower).all(axis=1) & (xyz <= upper).all(axis=1))
        for v, cam in enumerate(rig.cameras):
            mask = valid[:, v]
            errors = np.linalg.norm(cam.project(xyz[mask])
                                    - points[v][sources[mask, v]], axis=1)
            keep[mask] &= errors <= config.support_px
        indexes = np.flatnonzero(keep)
        indexes = sorted(indexes, key=lambda i: (-int(valid[i].sum()), float(rms[i])))
        chosen = []
        for i in indexes:
            # Keep a good pair if the third view is inconsistent; otherwise a
            # nearby three-view point subsumes its own two-view variants.
            duplicate = any(
                np.linalg.norm(xyz[i] - c.xyz) < 0.75
                and np.all((sources[i] < 0) | (sources[i] == c.source))
                for c in chosen
            )
            if not duplicate:
                chosen.append(NodeCandidate(xyz[i], sources[i], float(rms[i])))
            if len(chosen) >= config.node_candidate_limit:
                break
        result.append(chosen)
    return result


@dataclass
class _State:
    pose: np.ndarray
    velocity: np.ndarray
    sources: np.ndarray
    observed: np.ndarray
    templates: dict
    cost: float
    parent: int
    missing: np.ndarray
    last_position: np.ndarray
    last_frame: np.ndarray
    confirmation: np.ndarray
    recovery: np.ndarray
    identity_unresolved: np.ndarray
    appearance: np.ndarray
    appearance_quality: np.ndarray
    fit_success: bool = True
    image_cost: float = 0.0
    anatomy_refused: int = 0


class ReacquisitionTracker:
    name = "refractive-generative-reacquisition-v2"

    def __init__(self, rig, initial_xyz, config: RecoveryConfig | None = None):
        self.rig = rig
        self.config = config or RecoveryConfig()
        self.local = FastFitTracker(rig, initial_xyz, self.config.local)
        self.recovery_fit = FastFitTracker(
            rig, initial_xyz, replace(self.config.local,
                                     motion_sigma_mm=self.config.recovery_fit_sigma_mm))
        self.initial = self.local.initial
        self.lengths = self.local.lengths

    def _recovery_proposals(self, state, predicted, frame, pools, protected):
        cfg = self.config
        proposals = []
        for fish in np.flatnonzero(state.missing >= cfg.missing_before_recovery):
            others = [f for f in range(3) if f != fish and state.missing[f] == 0]

            def free(candidate, node, others=others):
                return not any(
                    np.any((candidate.source >= 0)
                           & (candidate.source == protected[f, :, node]))
                    for f in others
                )

            centers = [c for c in pools[1] if free(c, 1)]
            centers.sort(key=lambda c: np.linalg.norm(c.xyz - predicted[fish, 1]))
            sigma = min(cfg.maximum_connection_sigma_mm, cfg.initial_connection_sigma_mm
                        + cfg.uncertainty_growth_mm_per_frame * state.missing[fish])
            fish_proposals = []
            for center in centers[:6]:
                choices = []
                for node, length in ((0, self.lengths[fish, 0]), (2, self.lengths[fish, 1])):
                    options = []
                    direction = predicted[fish, node] - predicted[fish, 1]
                    for candidate in pools[node]:
                        distance = np.linalg.norm(candidate.xyz - center.xyz)
                        if free(candidate, node) and 0.5 * length < distance < 1.75 * length:
                            heading = np.linalg.norm((candidate.xyz - center.xyz) - direction)
                            score = ((distance - length) / cfg.local.shape_sigma_mm) ** 2
                            score += 0.02 * heading + candidate.rms
                            options.append((score, candidate.xyz))
                    options.sort(key=lambda p: p[0])
                    if not options:
                        options = [(6.0, center.xyz + direction)]
                    choices.append(options[:2])
                displacement = np.linalg.norm(center.xyz - predicted[fish, 1]) / sigma
                connection = 2 * (np.sqrt(1 + displacement**2) - 1)
                for mouth, tail in product(*choices):
                    pose = predicted.copy()
                    pose[fish] = np.stack((mouth[1], center.xyz, tail[1]))
                    if (np.any(pose < cfg.local.lower_mm)
                            or np.any(pose > cfg.local.upper_mm)):
                        continue
                    rank = connection + mouth[0] + tail[0] + center.rms
                    fish_proposals.append((rank, connection, pose, int(fish)))
            fish_proposals.sort(key=lambda p: p[0])
            # Evaluate view-specific appearance only after cheap geometric
            # ranking. It never learns a new template during proposal ranking.
            for rank, connection, pose, index in fish_proposals[:4]:
                image_cost, _ = self.local._image_score(
                    frame, pose, state.pose, state.templates,
                    np.zeros((3, 3), dtype=bool))
                proposals.append((rank + image_cost, connection, pose, index))
        proposals.sort(key=lambda p: p[0])
        # Retain a proposal for each lost identity before taking alternatives
        # for one fish; otherwise a complete loss could starve one identity.
        selected, represented = [], set()
        for proposal in proposals:
            if proposal[3] not in represented:
                selected.append(proposal)
                represented.add(proposal[3])
        selected.extend(p for p in proposals if not any(p is old for old in selected))
        return selected[:cfg.recovery_proposals_per_parent]

    def _image_score(self, frame, pose, previous, templates, observed):
        """Keep compact per-pose evidence for an independent temporal ID pass."""
        features = np.zeros((3, 3, 24), dtype=np.float32)
        quality = np.zeros((3, 3), dtype=np.float32)
        updated = templates.copy()
        if frame.evidence is None:
            return 0.0, updated, features, quality
        cfg = self.config.local
        total = 0.0
        body_points = _body_samples(pose)
        points = np.concatenate((pose.reshape(-1, 3), body_points.reshape(-1, 3),
                                 previous.reshape(-1, 3)))
        body_end = 9 + body_points.size // 3
        for view, cam in enumerate(self.rig.cameras):
            projected = cam.project(points)
            uv = projected[:9].reshape(3, 3, 2)
            body = projected[9:body_end].reshape(*body_points.shape[:-1], 2)
            old_uv = projected[body_end:].reshape(3, 3, 2)
            for fish in range(3):
                other = np.arange(3) != fish
                distance = np.linalg.norm(body[fish, :, None, None]
                                          - body[other][None], axis=-1)
                overlap = bool(np.nanmin(distance) < cfg.overlap_px)
                key = (fish, cam.view)
                score = frame.evidence.score(
                    cam.view, uv[fish], reference_appearance=templates.get(key),
                    previous_uv=old_uv[fish], occluded=overlap)
                total += cfg.appearance_weight * score.appearance_quality * score.appearance_cost
                total += cfg.flow_weight * score.motion_quality * score.motion_cost
                if score.appearance is not None:
                    if score.appearance.shape != (24,):
                        raise ValueError("the recovery appearance schema requires 8 bins/channel")
                    features[fish, view] = score.appearance
                    # A predicted body must not become strong re-ID evidence.
                    if observed[fish, :2].all() and not overlap:
                        quality[fish, view] = score.appearance_quality
                    if key not in updated and quality[fish, view] >= 0.25:
                        updated[key] = score.appearance.copy()
        return float(total), updated, features, quality

    def track(self, frames):
        cfg, local_cfg = self.config, self.config.local
        beam = [_State(self.initial.copy(), np.zeros((3, 3, 3)),
                       np.full((3, 3, 3), -1, np.int32), np.zeros((3, 3), bool), {},
                       0.0, -1, np.zeros(3, int), self.initial[:, 1].copy(),
                       np.full(3, -1, int), np.zeros(3, int), np.zeros(3, bool),
                       np.zeros(3, bool), np.zeros((3, 3, 24), np.float32),
                       np.zeros((3, 3), np.float32))]
        history, frame_indices, margins = [], [], []
        previous_frame = None
        proposal_frames = recovery_proposals = 0
        for frame in frames:
            if previous_frame is not None and frame.frame_idx <= previous_frame:
                raise ValueError("frame indices must be strictly increasing")
            dt = 1 if previous_frame is None else frame.frame_idx - previous_frame
            if previous_frame is None:
                beam[0].last_frame[:] = frame.frame_idx - 1
            need_recovery = any((s.missing >= cfg.missing_before_recovery).any() for s in beam)
            pools = node_candidates(self.rig, frame, cfg) if need_recovery else None
            proposal_frames += int(need_recovery)
            candidates = []
            for parent, state in enumerate(beam):
                predicted = np.clip(state.pose + state.velocity * dt,
                                    np.asarray(local_cfg.lower_mm) + 1e-4,
                                    np.asarray(local_cfg.upper_mm) - 1e-4)
                local_plans = self.local._plans(predicted, frame)
                proposals = [(predicted, plan, 0.0, -1, self.local)
                             for _, plan in local_plans]
                if pools is not None:
                    for _, connection, pose, fish in self._recovery_proposals(
                        state, predicted, frame, pools, local_plans[0][1]
                    ):
                        # All fish still participate in one exclusive assignment.
                        plan = self.recovery_fit._plans(pose, frame)[0][1]
                        proposals.append((pose, plan, cfg.recovery_cost + connection,
                                          fish, self.recovery_fit))
                        recovery_proposals += 1
                for anchor, plan, penalty, recovering, fitter in proposals:
                    pose, source, observed, cost, success, refused = fitter._fit(
                        anchor, plan, frame, dt)
                    image_cost, templates, features, quality = self._image_score(
                        frame, pose, state.pose, state.templates, observed)
                    supported = observed[:, :2].all(axis=1)
                    missing = np.where(supported, 0, state.missing + dt)
                    confirmation = np.maximum(state.confirmation - supported.astype(int), 0)
                    confirmation = np.where((state.confirmation > 0) & ~supported,
                                            cfg.confirmation_frames, confirmation)
                    recovered = np.zeros(3, bool)
                    # Alternating losses can exchange two identities even if
                    # they are never simultaneously missing for two frames.
                    # Reobserving geometry alone cannot resolve that identity.
                    identity_unresolved = (state.identity_unresolved
                                           | (missing >= cfg.missing_before_recovery))
                    if recovering >= 0 and supported[recovering]:
                        recovered[recovering] = True
                        confirmation[recovering] = cfg.confirmation_frames
                        identity_unresolved[recovering] = True
                    measured = (pose[:, 1] - state.pose[:, 1]) / dt
                    for fish in np.flatnonzero(recovered):
                        elapsed = max(1, frame.frame_idx - state.last_frame[fish])
                        measured[fish] = (pose[fish, 1] - state.last_position[fish]) / elapsed
                    body_velocity = ((1 - local_cfg.velocity_update) * state.velocity[:, 1]
                                     + local_cfg.velocity_update * measured)
                    body_velocity = np.where(observed[:, 1, None], body_velocity,
                                             state.velocity[:, 1] * 0.95)
                    speed = np.linalg.norm(body_velocity, axis=1, keepdims=True)
                    body_velocity *= np.minimum(1.0, local_cfg.max_velocity_mm_per_frame
                                                 / np.maximum(speed, 1e-12))
                    velocity = np.broadcast_to(body_velocity[:, None], (3, 3, 3)).copy()
                    last_position = np.where(supported[:, None], pose[:, 1], state.last_position)
                    last_frame = np.where(supported, frame.frame_idx, state.last_frame)
                    candidates.append(_State(
                        pose, velocity, source, observed, templates,
                        state.cost + cost + image_cost + penalty, parent, missing,
                        last_position, last_frame, confirmation, recovered,
                        identity_unresolved, features, quality,
                        success, image_cost, refused))
            candidates.sort(key=lambda s: s.cost)
            unique = {}
            for state in candidates:
                key = (state.parent, state.sources.tobytes(), state.pose.tobytes(),
                       state.velocity.tobytes(), state.confirmation.tobytes())
                if key not in unique:
                    unique[key] = state
            beam = list(unique.values())[:local_cfg.beam_width]
            if not beam:
                raise RuntimeError(f"no feasible state at frame {frame.frame_idx}")
            margins.append(beam[1].cost - beam[0].cost if len(beam) > 1 else np.nan)
            history.append(beam)
            frame_indices.append(frame.frame_idx)
            previous_frame = frame.frame_idx
        if not history:
            raise ValueError("at least one frame is required")

        def trace(index):
            selected = []
            for layer in reversed(history):
                state = layer[index]
                selected.append(state)
                index = state.parent
            return selected[::-1]

        selected = trace(0)
        poses = np.stack([s.pose for s in selected])
        observed = np.stack([s.observed for s in selected])
        sources = np.stack([s.sources for s in selected])
        xyz = np.where(observed[:, :, 1, None], poses[:, :, 1], np.nan)
        identity_margin = np.full((len(history), 3), np.inf)
        for alternate_index in range(1, len(beam)):
            alternative = trace(alternate_index)
            gap = beam[alternate_index].cost - beam[0].cost
            for t, (best, other) in enumerate(zip(selected, alternative, strict=True)):
                distance = np.linalg.norm(best.pose[:, 1, None] - other.pose[None, :, 1], axis=-1)
                costs = [sum(distance[i, p[i]] for i in range(3)) for p in PERMUTATIONS]
                permutation = np.asarray(PERMUTATIONS[int(np.argmin(costs))])
                exchanged = permutation != np.arange(3)
                identity_margin[t, exchanged] = np.minimum(identity_margin[t, exchanged], gap)
        separation = np.linalg.norm(poses[:, :, None, 1] - poses[:, None, :, 1], axis=-1)
        separation[:, np.arange(3), np.arange(3)] = np.inf
        close = separation.min(axis=2) < local_cfg.unresolved_separation_mm
        no_alternative = close & ~np.isfinite(identity_margin)
        pending = np.stack([s.confirmation > 0 for s in selected])
        geometry_review = (~observed[:, :, :2].all(axis=2) | pending
                           | np.asarray([not s.fit_success for s in selected])[:, None])
        identity_review = (no_alternative | (identity_margin < local_cfg.review_margin)
                           | np.stack([s.identity_unresolved for s in selected]))
        review = geometry_review | identity_review
        recovery = np.stack([s.recovery for s in selected])
        return RecoveryTrackResult(
            np.asarray(frame_indices, dtype=np.int64), poses, xyz, observed, sources,
            review, np.asarray(margins),
            {"algorithm": self.name, "initializations": 1,
             "external_initialization_frames": [int(frame_indices[0])],
             "later_baseline_or_ground_truth_inputs": False,
             "recovery_candidate_frames": proposal_frames,
             "recovery_proposals_fitted": recovery_proposals,
             "selected_recovery_animal_frames": int(recovery.sum()),
             "recovery_events": [{"frame": int(frame_indices[t]), "slot": int(f)}
                                 for t, f in np.argwhere(recovery)],
             "pending_confirmation_animal_frames": int(pending.sum()),
             "unresolved_identity_animal_frames": int(identity_review.sum()),
             "anatomy_refused_animal_frames": sum(s.anatomy_refused for s in selected),
             "fitting_limit_or_failure_frames": sum(not s.fit_success for s in selected),
             "appearance_flow_cost_sum": sum(s.image_cost for s in selected),
             "final_cost": float(beam[0].cost), "window_initialized": True,
             "identity_labels": "anonymous seed slots", "cost_is_probability": False,
             "identity_margin_kind": "final retained branches; infinity is not certainty",
             "reacquisition_identity_policy": "any sustained loss or selected recovery remains "
             "unresolved until a separate identity-evidence pass or human review",
             "search": "local fit and observation-driven recovery; final beam backtracking"},
            identity_margin, np.stack([s.appearance for s in selected]),
            np.stack([s.appearance_quality for s in selected]), geometry_review, identity_review)


class ReacquisitionLinker:
    """Linker protocol: bind observations; use only the first frame's seeds."""

    name = ReacquisitionTracker.name

    def __init__(self, rig, frames, config=None):
        self.rig, self.frames, self.config = rig, frames, config
        self.result = None

    def link(self, instances):
        if not self.frames:
            raise ValueError("bound observations must not be empty")
        first = self.frames[0].frame_idx
        seeds = sorted((s for s in instances if s.frame == first), key=lambda s: s.slot)
        if len(seeds) != 3 or len({s.slot for s in seeds}) != 3:
            raise ValueError("exactly three distinct first-frame skeletons are needed")
        initial = np.stack([s.xyz for s in seeds])
        self.result = ReacquisitionTracker(self.rig, initial, self.config).track(self.frames)
        return self.result.accepted_xyz

"""Experimental 3D prediction -> refractive image fit -> multi-hypothesis update.

All identities are anonymous seed indices. No baseline after initialization and
no human identity annotations are accepted by this module. Beam costs are NOT
posterior probabilities. An observation assignment is per node, so SLEAP's
occasionally mixed instance grouping need not be inherited by the latent fish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations, product
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from medaka3d.calib import Rig
from medaka3d.types import VIEWS

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from observations import SceneEvidence

    from medaka3d.types import Instance3D

PERMUTATIONS = tuple(permutations(range(3)))
NODE_WEIGHTS = np.array([1.0, 1.0, 0.35])


@dataclass(frozen=True)
class TrackerConfig:
    """Fixed pilot settings; millimetres, pixels and native frame increments."""

    beam_width: int = 4
    proposals_per_parent: int = 3
    candidates_per_node: int = 3
    pixel_sigma: float = 4.0
    gate_px: float = 60.0
    missing_cost: float = 6.0
    motion_sigma_mm: float = 1.0
    shape_sigma_mm: float = 2.0
    max_segment_ratio: float = 2.0
    image_weight: float = 0.7
    appearance_weight: float = 4.0
    flow_weight: float = 0.5
    velocity_update: float = 0.5
    max_velocity_mm_per_frame: float = 4.0
    max_nfev: int = 10
    review_margin: float = 2.0
    unresolved_separation_mm: float = 4.0
    overlap_px: float = 14.0
    max_reprojection_px: float = 12.0
    lower_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    upper_mm: tuple[float, float, float] = (192.0, 192.0, 186.0)

    def __post_init__(self):
        for name in ("beam_width", "proposals_per_parent", "candidates_per_node", "max_nfev"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("pixel_sigma", "gate_px", "missing_cost", "motion_sigma_mm",
                     "shape_sigma_mm", "max_reprojection_px", "max_velocity_mm_per_frame"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.velocity_update <= 1:
            raise ValueError("velocity_update must be between zero and one")
        if not np.isfinite(self.max_segment_ratio) or self.max_segment_ratio <= 1:
            raise ValueError("max_segment_ratio must be finite and greater than one")
        if np.any(np.asarray(self.upper_mm) <= self.lower_mm):
            raise ValueError("tank bounds must be ordered")


@dataclass(frozen=True)
class FrameInput:
    frame_idx: int
    detections: dict[str, np.ndarray]
    evidence: SceneEvidence | None = None

    def __post_init__(self):
        if set(self.detections) != set(VIEWS):
            raise ValueError(f"detections must explicitly include {VIEWS}; use empty arrays")
        for view, points in self.detections.items():
            if np.asarray(points).ndim != 3 or np.asarray(points).shape[1:] != (3, 2):
                raise ValueError(f"{view}: expected (n_candidates, 3, 2)")


@dataclass(frozen=True)
class TrackResult:
    frame_idx: np.ndarray
    pose_xyz: np.ndarray
    xyz: np.ndarray
    observed: np.ndarray
    node_source: np.ndarray
    review: np.ndarray
    margin: np.ndarray
    diagnostics: dict = field(default_factory=dict)
    identity_margin: np.ndarray | None = None

    @property
    def accepted_xyz(self) -> np.ndarray:
        """Measured pectorals excluding unresolved identities; no gap filling."""
        return np.where(self.review[..., None], np.nan, self.xyz)


@dataclass
class _State:
    pose: np.ndarray
    velocity: np.ndarray
    sources: np.ndarray
    observed: np.ndarray
    templates: dict
    cost: float
    parent: int
    image_cost: float = 0.0
    fit_success: bool = True
    anatomy_refused: int = 0


def exclusive_assignments(cost: np.ndarray, missing_cost: float,
                          candidates_per_fish: int = 3, keep: int = 2):
    """K best bounded injective assignments; -1 can be chosen at ANY cardinality.

    ``cost`` is (3, n_detections). Candidate truncation makes this approximate
    when n_detections is large; it never forces a reflection into a fish slot.
    """
    cost = np.asarray(cost, float)
    if cost.ndim != 2 or cost.shape[0] != 3:
        raise ValueError("assignment cost must have shape (3, n_detections)")
    options = []
    for row in cost:
        valid = np.flatnonzero(np.isfinite(row) & (row < missing_cost))
        ranked = valid[np.argsort(row[valid], kind="stable")[:candidates_per_fish]]
        options.append([*ranked.tolist(), -1])
    out = []
    for selection in product(*options):
        used = [i for i in selection if i >= 0]
        if len(used) != len(set(used)):
            continue
        value = sum(missing_cost if j < 0 else cost[i, j]
                    for i, j in enumerate(selection))
        out.append((float(value), np.asarray(selection, dtype=np.int32)))
    out.sort(key=lambda item: (item[0], tuple(item[1])))
    return out[:keep]


def _body_samples(pose: np.ndarray) -> np.ndarray:
    alpha = np.linspace(0, 1, 5)[:, None]
    head = pose[..., 0, None, :] * (1 - alpha) + pose[..., 1, None, :] * alpha
    tail = pose[..., 1, None, :] * (1 - alpha) + pose[..., 2, None, :] * alpha
    return np.concatenate([head, tail], axis=-2)


def _robust_observation(residual):
    """Soft-L1 equivalent for observations ONLY; anatomy must not saturate."""
    return residual * np.sqrt(2 / (np.sqrt(1 + residual * residual) + 1))


class GenerativeTracker:
    """Approximate beam filter with final backtracking, driven by 3D projections."""

    name = "refractive-generative-beam"

    def __init__(self, rig: Rig, initial_xyz: np.ndarray,
                 config: TrackerConfig | None = None):
        self.rig = rig
        self.config = config or TrackerConfig()
        self.initial = np.array(initial_xyz, dtype=float, copy=True)
        if self.initial.shape != (3, 3, 3) or not np.isfinite(self.initial).all():
            raise ValueError("initial_xyz must be three finite three-node 3D skeletons")
        if rig.views != VIEWS:
            raise ValueError(f"rig view order must be {VIEWS}")
        if (np.any(self.initial < self.config.lower_mm)
                or np.any(self.initial > self.config.upper_mm)):
            raise ValueError("initial fish must lie inside the explicit tank bounds")
        self.lengths = np.linalg.norm(np.diff(self.initial, axis=1), axis=2)
        if np.any(self.lengths < 0.1):
            raise ValueError("seed body segments must have nonzero length")

    def _project(self, pose):
        return {c.view: c.project(pose) for c in self.rig.cameras}

    def _plans(self, predicted, frame):
        cfg = self.config
        projected = self._project(predicted)
        choices, cost_tables = {}, {}
        base = np.full((3, 3, 3), -1, dtype=np.int32)  # fish, view, node
        for v, view in enumerate(VIEWS):
            points = np.asarray(frame.detections[view], float)
            for node in range(3):
                distance = np.linalg.norm(
                    projected[view][:, node, None] - points[None, :, node], axis=-1)
                distance = np.where(np.isfinite(distance), distance, np.inf)
                # Soft-L1 on pixel error, plus an explicit finite search gate.
                cost = 2 * (np.sqrt(1 + (distance / cfg.pixel_sigma) ** 2) - 1)
                cost = np.where(distance <= cfg.gate_px, cost, np.inf)
                choices[v, node] = exclusive_assignments(
                    cost, cfg.missing_cost, cfg.candidates_per_node, keep=2)
                cost_tables[v, node] = cost
                base[:, v, node] = choices[v, node][0][1]
        plans = [base]
        for (v, node), alternatives in choices.items():
            if len(alternatives) > 1:
                alternate = base.copy()
                alternate[:, v, node] = alternatives[1][1]
                plans.append(alternate)
        # Coherent identity alternatives across ALL nodes/cameras must be present;
        # changing just one camera's assignment cannot represent an ID exchange.
        plans.extend(base[list(p)].copy() for p in PERMUTATIONS[1:])
        coherent_keys = {base[list(p)].tobytes() for p in PERMUTATIONS[1:]}
        unique = {}
        for plan in plans:
            value = 0.0
            feasible = True
            for v in range(3):
                for node in range(3):
                    for fish in range(3):
                        idx = plan[fish, v, node]
                        val = cfg.missing_cost if idx < 0 else cost_tables[v, node][fish, idx]
                        feasible &= bool(np.isfinite(val))
                        value += NODE_WEIGHTS[node] * val
            if feasible:
                unique[plan.tobytes()] = (value, plan)
        ranked = sorted(unique.values(), key=lambda item: item[0])
        selected = ranked[:1]
        # Otherwise cheaper one-node perturbations crowd out every full identity
        # alternative before appearance can evaluate it at an ambiguous crossing.
        coherent = [r for r in ranked[1:] if r[1].tobytes() in coherent_keys]
        if cfg.proposals_per_parent > 1 and coherent:
            selected.append(coherent[0])
        selected_keys = {r[1].tobytes() for r in selected}
        selected.extend(r for r in ranked if r[1].tobytes() not in selected_keys)
        return selected[:cfg.proposals_per_parent]

    def _fit(self, prediction, plan, frame, dt):
        cfg = self.config
        target = np.zeros((3, 3, 3, 2))  # view, fish, node, xy
        used = np.zeros((3, 3, 3), dtype=bool)
        for v, view in enumerate(VIEWS):
            for fish in range(3):
                for node in range(3):
                    row = plan[fish, v, node]
                    if row >= 0:
                        target[v, fish, node] = frame.detections[view][row, node]
                        used[v, fish, node] = np.isfinite(target[v, fish, node]).all()
        motion_scale = cfg.motion_sigma_mm * np.sqrt(max(dt, 1))
        node_weight = np.sqrt(NODE_WEIGHTS)[None, :, None]

        def residual_batch(flat):
            batch = flat.shape[0]
            pose = flat.reshape(batch, 3, 3, 3)
            terms = []
            for v, cam in enumerate(self.rig.cameras):
                uv = cam.project(pose)
                reprojection = (uv - target[v]) * node_weight / cfg.pixel_sigma
                terms.append(_robust_observation(
                    np.nan_to_num(reprojection[:, used[v]], nan=100.0))
                    .reshape(batch, -1))
                if frame.evidence is not None and cfg.image_weight > 0:
                    body_uv = cam.project(_body_samples(pose)).reshape(-1, 2)
                    distance, confidence = frame.evidence.silhouette_residuals(cam.view, body_uv)
                    terms.append(_robust_observation(
                        cfg.image_weight * distance * np.sqrt(confidence))
                                 .reshape(batch, -1))
            terms.append(((pose - prediction) / motion_scale).reshape(batch, -1))
            lengths = np.linalg.norm(np.diff(pose, axis=-2), axis=-1)
            terms.append(((lengths - self.lengths) / cfg.shape_sigma_mm).reshape(batch, -1))
            return np.concatenate(terms, axis=1)

        def residual(flat):
            return residual_batch(flat[None, :])[0]

        def jacobian(flat):
            # Batch finite differences through refraction and smooth image maps.
            # This avoids 27 separate Python/projection calls per Jacobian.
            step = np.where(flat + 1e-3 < upper, 1e-3, -1e-3)
            trial = np.broadcast_to(flat, (28, flat.size)).copy()
            trial[1:] += np.eye(flat.size) * step
            values = residual_batch(trial)
            return ((values[1:] - values[0]) / step[:, None]).T

        lower = np.broadcast_to(np.asarray(cfg.lower_mm) + 1e-5, (3, 3, 3)).ravel()
        upper = np.broadcast_to(np.asarray(cfg.upper_mm) - 1e-5, (3, 3, 3)).ravel()
        initial = np.clip(prediction.ravel(), lower + 1e-7, upper - 1e-7)
        fit = least_squares(residual, initial, jac=jacobian,
                            bounds=(lower, upper), loss="linear",
                            max_nfev=cfg.max_nfev, ftol=1e-3, xtol=1e-3, gtol=1e-3)
        pose = fit.x.reshape(3, 3, 3)
        lengths = np.linalg.norm(np.diff(pose, axis=1), axis=2)
        refused = (lengths > self.lengths * cfg.max_segment_ratio).any(axis=1)
        # Refuse an anatomically impossible measurement; retain the previous
        # physically admissible body's prediction instead of clamping its nodes
        # into a fabricated measured skeleton. The seed-relative cap is an
        # engineering guard, not a calibrated species anatomy distribution.
        pose[refused] = prediction[refused]
        source = plan.copy()
        for v, cam in enumerate(self.rig.cameras):
            error = np.linalg.norm(cam.project(pose) - target[v], axis=2)
            keep = used[v] & (error <= cfg.max_reprojection_px)
            source[:, v, :] = np.where(keep, source[:, v, :], -1)
        source[refused] = -1
        observed = (source >= 0).sum(axis=1) >= 2
        missing = float(((~used).sum(axis=(0, 1)) * NODE_WEIGHTS).sum()) * cfg.missing_cost
        final_cost = float(np.square(residual(pose.ravel())).sum()) + missing
        if refused.any():
            final_cost += float((used[:, refused].sum(axis=(0, 1)) * NODE_WEIGHTS).sum()
                                * cfg.missing_cost)
        return pose, source, observed, final_cost, bool(fit.success), int(refused.sum())

    def _image_score(self, frame, pose, previous, templates, observed):
        if frame.evidence is None:
            return 0.0, templates.copy()
        cfg = self.config
        total = 0.0
        updated = templates.copy()
        for cam in self.rig.cameras:
            uv = cam.project(pose)
            body_uv = cam.project(_body_samples(pose))
            old_uv = cam.project(previous)
            for fish in range(3):
                other = np.arange(3) != fish
                # Conservative projected body proximity, not depth guessed from a
                # merged mask. The silhouette union may still support geometry.
                distances = np.linalg.norm(body_uv[fish, :, None, None, :]
                                           - body_uv[other][None, :, :, :], axis=-1)
                overlap = bool(np.nanmin(distances) < cfg.overlap_px)
                key = (fish, cam.view)
                score = frame.evidence.score(
                    cam.view, uv[fish], reference_appearance=templates.get(key),
                    previous_uv=old_uv[fish], occluded=overlap)
                total += cfg.appearance_weight * score.appearance_quality * score.appearance_cost
                total += cfg.flow_weight * score.motion_quality * score.motion_cost
                # Freeze the first clear per-view reference. Updating on predictions
                # risks teaching the tracker its own mistaken identity.
                if (key not in updated and not overlap and observed[fish, :2].all()
                        and score.appearance is not None and score.appearance_quality >= 0.25):
                    updated[key] = np.array(score.appearance, copy=True)
        return float(total), updated

    def track(self, frames: Iterable[FrameInput]) -> TrackResult:
        cfg = self.config
        root = _State(self.initial.copy(), np.zeros((3, 3, 3)),
                      np.full((3, 3, 3), -1, dtype=np.int32),
                      np.zeros((3, 3), bool), {}, 0.0, -1)
        beam = [root]
        history, frame_indices, margins = [], [], []
        previous_frame = None
        for frame in frames:
            if previous_frame is not None and frame.frame_idx <= previous_frame:
                raise ValueError("frame indices must be strictly increasing")
            dt = 1 if previous_frame is None else frame.frame_idx - previous_frame
            candidates = []
            for parent, state in enumerate(beam):
                predicted = state.pose + state.velocity * dt
                predicted = np.clip(predicted, np.asarray(cfg.lower_mm) + 1e-4,
                                    np.asarray(cfg.upper_mm) - 1e-4)
                for _, plan in self._plans(predicted, frame):
                    pose, source, observed, cost, success, refused = self._fit(
                        predicted, plan, frame, dt)
                    image_cost, templates = self._image_score(
                        frame, pose, state.pose, state.templates, observed)
                    measured = (pose[:, 1] - state.pose[:, 1]) / dt
                    body_velocity = ((1 - cfg.velocity_update) * state.velocity[:, 1]
                                     + cfg.velocity_update * measured)
                    # Forecast a body translation. Independent extrapolations of
                    # missing head/tail nodes previously stretched fish indefinitely.
                    body_velocity = np.where(observed[:, 1, None], body_velocity,
                                             state.velocity[:, 1] * 0.95)
                    velocity = np.broadcast_to(body_velocity[:, None], (3, 3, 3)).copy()
                    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
                    velocity *= np.minimum(1.0, cfg.max_velocity_mm_per_frame
                                           / np.maximum(speed, 1e-12))
                    candidates.append(_State(pose, velocity, source, observed, templates,
                                             state.cost + cost + image_cost, parent,
                                             image_cost, success, refused))
            candidates.sort(key=lambda item: item.cost)
            # Recombine nearly identical present states, keeping the cheapest
            # history. Their earlier histories may differ: this is beam search.
            unique = {}
            for state in candidates:
                key = (state.parent, state.sources.tobytes(), state.pose.tobytes(),
                       state.velocity.tobytes())
                if key not in unique:
                    unique[key] = state
            beam = list(unique.values())[:cfg.beam_width]
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
        margin = np.asarray(margins)
        # Compare identity permutations, not cheaper tail omissions. Retrospective
        # disagreement uses final surviving histories and their accumulated costs.
        # Infinity means no such competitor survived; it is NOT certainty.
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
        close = separation.min(axis=2) < cfg.unresolved_separation_mm
        # A close encounter with no retained identity alternative cannot be
        # certified merely because the beam pruned its alternatives.
        missing_alternative = close & ~np.isfinite(identity_margin)
        review = (~observed[:, :, :2].all(axis=2)
                  | (identity_margin < cfg.review_margin) | missing_alternative
                  | np.asarray([not s.fit_success for s in selected])[:, None])
        return TrackResult(
            np.asarray(frame_indices, dtype=np.int64), poses, xyz, observed, sources,
            review, margin,
            {"algorithm": self.name, "search": "approximate beam + local robust fit",
             "identity_labels": "anonymous seed slots", "cost_is_probability": False,
             "beam_width": cfg.beam_width, "final_cost": beam[0].cost,
             "fitting_limit_or_failure_frames": sum(not s.fit_success for s in selected),
             "anatomy_refused_animal_frames": sum(s.anatomy_refused for s in selected),
             "observation_loss": "soft-L1; motion and anatomy remain quadratic",
             "body_forecast": "shared pectoral translation, no independent node drift",
             "appearance_flow_cost_sum": sum(s.image_cost for s in selected),
             "margin_kind": "filtering retained-state gap, not smoothed identity uncertainty",
             "identity_margin_kind": "final-beam competing permutations; infinity means absent",
             "review_missing_node_animal_frames": int((~observed[:, :, :2].all(axis=2)).sum()),
             "review_competing_identity_animal_frames": int(
                 (identity_margin < cfg.review_margin).sum()),
             "review_close_without_alternative_animal_frames": int(missing_alternative.sum()),
             "window_initialized": True}, identity_margin)


class GenerativeLinker:
    """Existing Linker interface, with video observations bound at construction.

    First-frame instances must carry full (3,3) node skeletons. Later instances
    do not enter inference: otherwise a comparison could leak the reference path.
    """

    name = GenerativeTracker.name

    def __init__(self, rig: Rig, frames: Sequence[FrameInput],
                 config: TrackerConfig | None = None):
        self.rig, self.frames, self.config = rig, frames, config
        self.result: TrackResult | None = None

    def link(self, instances: Sequence[Instance3D]) -> np.ndarray:
        if not self.frames:
            raise ValueError("bound frame observations must not be empty")
        first = self.frames[0].frame_idx
        seeds = sorted((item for item in instances if item.frame == first), key=lambda x: x.slot)
        if len(seeds) != 3 or len({s.slot for s in seeds}) != 3:
            raise ValueError("exactly three distinct first-frame skeletons are needed")
        initial = np.stack([s.xyz for s in seeds])
        self.result = GenerativeTracker(self.rig, initial, self.config).track(self.frames)
        return self.result.accepted_xyz

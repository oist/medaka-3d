"""Experimental exact-structure finite differences for the frozen v3 objective.

Every residual row depends on one fish's nine coordinates. Perturb the same
coordinate within all three fish simultaneously, then scatter each row only
into its own fish's columns. This reduces 28 projected batches to 10 while
retaining the original joint 27-coordinate solver, residuals and tolerances.
When image evidence is present, skeleton nodes and body samples share one
projection call per camera; the projected coordinates retain their row order.
Cross-fish appearance/occlusion scoring remains in the unchanged parent class.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from tracker import NODE_WEIGHTS, GenerativeTracker, _body_samples, _robust_observation

from medaka3d.types import VIEWS


@dataclass(frozen=True)
class FitProblem:
    residual_batch: Callable
    row_fish: np.ndarray
    target: np.ndarray
    used: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    initial: np.ndarray

    def residual(self, flat):
        return self.residual_batch(flat[None, :])[0]

    def jacobian(self, flat):
        step = np.where(flat + 1e-3 < self.upper, 1e-3, -1e-3)
        trial = np.broadcast_to(flat, (10, flat.size)).copy()
        # Coordinate j in each fish can have a different step sign at bounds.
        for fish in range(3):
            columns = fish * 9 + np.arange(9)
            trial[np.arange(1, 10), columns] += step[columns]
        values = self.residual_batch(trial)
        rows = np.arange(len(self.row_fish))[:, None]
        columns = self.row_fish[:, None] * 9 + np.arange(9)
        derivative = (values[1:] - values[0]).T / step[columns]
        jac = np.zeros((len(self.row_fish), flat.size), dtype=float)
        jac[rows, columns] = derivative
        return jac


class FastFitTracker(GenerativeTracker):
    """Same v3 fit objective with its fish-wise Jacobian sparsity used exactly."""

    def fit_problem(self, prediction, plan, frame, dt):
        cfg = self.config
        target = np.zeros((3, 3, 3, 2))
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
        fish_per_node = np.broadcast_to(np.arange(3)[:, None], (3, 3))
        body_count = _body_samples(prediction).shape[-2]
        row_blocks = []
        for v in range(3):
            row_blocks.append(np.repeat(fish_per_node[used[v]], 2))
            if frame.evidence is not None and cfg.image_weight > 0:
                row_blocks.append(np.repeat(np.arange(3), body_count))
        row_blocks.extend((np.repeat(np.arange(3), 9), np.repeat(np.arange(3), 2)))
        row_fish = np.concatenate(row_blocks)

        def residual_batch(flat):
            batch = flat.shape[0]
            pose = flat.reshape(batch, 3, 3, 3)
            has_image = frame.evidence is not None and cfg.image_weight > 0
            if has_image:
                points = np.concatenate((pose.reshape(batch, -1, 3),
                                         _body_samples(pose).reshape(batch, -1, 3)), axis=1)
            terms = []
            for v, cam in enumerate(self.rig.cameras):
                if has_image:
                    projected = cam.project(points)
                    uv = projected[:, :9].reshape(batch, 3, 3, 2)
                else:
                    uv = cam.project(pose)
                reprojection = (uv - target[v]) * node_weight / cfg.pixel_sigma
                terms.append(_robust_observation(
                    np.nan_to_num(reprojection[:, used[v]], nan=100.0))
                    .reshape(batch, -1))
                if has_image:
                    body_uv = projected[:, 9:].reshape(-1, 2)
                    distance, confidence = frame.evidence.silhouette_residuals(cam.view, body_uv)
                    terms.append(_robust_observation(
                        cfg.image_weight * distance * np.sqrt(confidence))
                                 .reshape(batch, -1))
            terms.append(((pose - prediction) / motion_scale).reshape(batch, -1))
            lengths = np.linalg.norm(np.diff(pose, axis=-2), axis=-1)
            terms.append(((lengths - self.lengths) / cfg.shape_sigma_mm).reshape(batch, -1))
            return np.concatenate(terms, axis=1)

        lower = np.broadcast_to(np.asarray(cfg.lower_mm) + 1e-5, (3, 3, 3)).ravel()
        upper = np.broadcast_to(np.asarray(cfg.upper_mm) - 1e-5, (3, 3, 3)).ravel()
        initial = np.clip(prediction.ravel(), lower + 1e-7, upper - 1e-7)
        return FitProblem(residual_batch, row_fish, target, used, lower, upper, initial)

    def _fit(self, prediction, plan, frame, dt):
        cfg = self.config
        problem = self.fit_problem(prediction, plan, frame, dt)
        target, used = problem.target, problem.used
        fit = least_squares(problem.residual, problem.initial, jac=problem.jacobian,
                            bounds=(problem.lower, problem.upper), loss="linear",
                            max_nfev=cfg.max_nfev, ftol=1e-3, xtol=1e-3, gtol=1e-3)
        pose = fit.x.reshape(3, 3, 3)
        lengths = np.linalg.norm(np.diff(pose, axis=1), axis=2)
        refused = (lengths > self.lengths * cfg.max_segment_ratio).any(axis=1)
        pose[refused] = prediction[refused]
        source = plan.copy()
        for v, cam in enumerate(self.rig.cameras):
            error = np.linalg.norm(cam.project(pose) - target[v], axis=2)
            keep = used[v] & (error <= cfg.max_reprojection_px)
            source[:, v, :] = np.where(keep, source[:, v, :], -1)
        source[refused] = -1
        observed = (source >= 0).sum(axis=1) >= 2
        missing = float(((~used).sum(axis=(0, 1)) * NODE_WEIGHTS).sum()) * cfg.missing_cost
        final_cost = float(np.square(problem.residual(pose.ravel())).sum()) + missing
        if refused.any():
            final_cost += float((used[:, refused].sum(axis=(0, 1)) * NODE_WEIGHTS).sum()
                                * cfg.missing_cost)
        return pose, source, observed, final_cost, bool(fit.success), int(refused.sum())

"""Bounded local body comparisons using the unchanged joint fitting objective.

The saved chosen path supplies a common previous-state motion prediction. This
module does not track, reset, relabel, accept coordinates, or read human truth.
Each start gets a fresh nearest exclusive SLEAP plan, fixed during optimization.
The frozen native tracker performs the actual fitting and best-iterate choice.
"""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import torch
from accelerated_tracker import AcceleratedThreeBodyTracker
from graph_fit import FitInputs, objective
from node_geometry import assigned_rays, assignment_plans, supported_nodes
from renderer import RefractiveFishRenderer, RendererConfig
from tracker import Proposal, State, ThreeBodyTracker, TrackerConfig


def replay_chosen_parents(geometry, initial_pose):
    """Reconstruct parents with the original velocity, clipping and gap recurrence.

All native frames from the original initialization must be supplied. Current
body support chooses the same velocity update as the old tracker; direct-node
support alone cannot replace it. Saved missing ages must match exactly.
"""
    frames = np.asarray(geometry['frame_idx'])
    poses = np.asarray(geometry['pose_xyz'])
    supported = np.asarray(geometry['body_supported'])
    missing = np.asarray(geometry['missing'])
    initial = np.asarray(initial_pose, float)
    if (frames.ndim != 1 or not len(frames) or not np.issubdtype(frames.dtype, np.integer)
            or np.any(frames < 0) or np.any(frames > np.iinfo(np.int64).max)):
        raise ValueError('native frames must be nonnegative int64-compatible integers')
    frames = frames.astype(np.int64, copy=False)
    n = len(frames)
    if np.any(np.diff(frames) != 1):
        raise ValueError('parent replay requires a nonempty consecutive native sequence')
    if (poses.shape != (n, 3, 3, 3) or not np.isfinite(poses).all()
            or initial.shape != (3, 3, 3) or not np.isfinite(initial).all()):
        raise ValueError('initial and saved poses must be finite three-body geometry')
    if (supported.shape != (n, 3) or supported.dtype != bool
            or missing.shape != (n, 3) or not np.issubdtype(missing.dtype, np.integer)):
        raise ValueError('saved body support and missing ages are required')
    lower, upper = np.array([0., 0., -6.]), np.array([192., 192., 186.])
    if any(np.any((value < lower) | (value > upper)) for value in (poses, initial)):
        raise ValueError('saved geometry is outside the unchanged engineering bounds')
    predicted = np.empty_like(poses, dtype=float)
    velocities = np.empty((n, 3, 3), float)
    gaps = np.empty((n, 3), np.int64)
    parent_pose = initial.copy()
    parent_velocity = np.zeros((3, 3))
    parent_missing = np.zeros(3, np.int64)
    for t, pose in enumerate(poses):
        predicted[t] = np.clip(parent_pose + parent_velocity[:, None], lower, upper)
        velocities[t] = parent_velocity
        gaps[t] = parent_missing
        actual_missing = np.where(supported[t], 0, parent_missing + 1)
        if not np.array_equal(actual_missing, missing[t]):
            raise ValueError(f'saved missing recurrence differs at native frame {frames[t]}')
        displacement = pose[:, 1] - parent_pose[:, 1]
        velocity = .5 * parent_velocity + .5 * displacement
        speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
        velocity *= np.minimum(1., 2. / np.maximum(speed, 1e-12))
        velocity = np.where(supported[t, :, None], velocity, .8 * parent_velocity)
        parent_pose, parent_velocity, parent_missing = pose, velocity, actual_missing
    return dict(predicted_pose_xyz=predicted, parent_velocity=velocities, parent_missing=gaps)


class LocalBodyFitter:
    """Same scalar score and fourteen-step fitter for image and random starts.

The configuration is explicit and comes from the completed producer. Tests may
use fewer steps; the real experiment's runner must check the registered values.
Only admitted finite scenes inside the old box enter this class. A cap of 24
limits GPU batches; it does not remove trials or reinitialize a trajectory.
"""

    def __init__(self, rig, initial_pose, tracker_config, batch_size=24):
        config = (TrackerConfig(**tracker_config) if isinstance(tracker_config, dict)
                  else tracker_config)
        if not isinstance(config, TrackerConfig):
            raise TypeError('tracker_config must be the frozen tracker configuration')
        if type(batch_size) is not int or not 1 <= batch_size <= 24:
            raise ValueError('batch size must lie in 1..24')
        if config.mode != 'detectron2':
            raise ValueError('the registered body comparison uses independent D2 masks')
        cls = (AcceleratedThreeBodyTracker if torch.device(config.device).type == 'cuda'
               else ThreeBodyTracker)
        self.tracker = cls(rig, initial_pose, config)
        self.batch_size = batch_size
        self.config = asdict(config)

    def _scores(self, poses, plans, predicted, parent_missing, rays, targets):
        tracker, cfg = self.tracker, self.tracker.config
        origin, direction, valid = assigned_rays(rays, plans)
        renderer = RefractiveFishRenderer(
            tracker.rig, targets.pixels, RendererConfig(width_ratio=cfg.width_ratio),
            device=tracker.device, dtype=tracker.dtype,
        )
        tensor = tracker._tensor
        node_weight = tensor([1., 1., .35])[None, None, None]
        lengths = tensor(tracker.lengths)[None]
        inputs = FitInputs(
            tensor(origin), tensor(direction), tensor(valid),
            tensor(np.broadcast_to(predicted, poses.shape).copy()),
            cfg.motion_sigma_mm + .3 * torch.clamp(
                tensor(np.broadcast_to(parent_missing, (len(poses), 3)).copy()), max=20),
            lengths, torch.clamp(.2 * lengths, min=1.), node_weight,
            tensor(targets.target), tensor(targets.weights) * renderer.ray_valid,
        )
        with torch.no_grad():
            scores, parts, _ = objective(
                tensor(poses), cfg, inputs, 9 * float(node_weight.sum()), renderer,
                diagnostics=True,
            )
        return scores.cpu().numpy(), parts.cpu().numpy()

    def run(self, poses, predicted, parent_missing, rays, targets):
        tracker, cfg = self.tracker, self.tracker.config
        poses, predicted = np.asarray(poses, float), np.asarray(predicted, float)
        parent_missing = np.asarray(parent_missing)
        if (poses.ndim != 4 or poses.shape[1:] != (3, 3, 3) or not len(poses)
                or not np.isfinite(poses).all()):
            raise ValueError('local starts must be nonempty finite N,3,3,3 scenes')
        if (predicted.shape != (3, 3, 3) or not np.isfinite(predicted).all()
                or parent_missing.shape != (3,)
                or not np.issubdtype(parent_missing.dtype, np.integer)
                or np.any(parent_missing < 0)):
            raise ValueError('finite common prediction and nonnegative integer gap ages required')
        if any(np.any((p < tracker.lower) | (p > tracker.upper)) for p in (poses, predicted)):
            raise ValueError('outside-box trial: caller must record and exclude it before fitting')
        if targets is None:
            raise ValueError('explicit independent mask samples are required')
        started = time.perf_counter()
        rows = []
        for start in range(0, len(poses), self.batch_size):
            batch = poses[start:start + self.batch_size]
            plans = np.stack([
                assignment_plans(p, rays, cfg.ray_sigma_mm, cfg.missing_cost, keep=1)[0][1]
                for p in batch
            ])
            before, before_parts = self._scores(
                batch, plans, predicted, parent_missing, rays, targets)
            state = State(predicted.copy(), np.zeros((3, 3)), parent_missing.copy(),
                          np.zeros(3, bool), 0.)
            proposals = [Proposal(p.copy(), predicted.copy(), plan.copy(), 0, 'local_body_start')
                         for p, plan in zip(batch, plans, strict=True)]
            fitted, scores, parts, mask = tracker._fit(proposals, [state], rays, targets)
            replay_scores, replay_parts = self._scores(
                fitted, plans, predicted, parent_missing, rays, targets)
            np.testing.assert_allclose(replay_scores, scores, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(replay_parts, parts, rtol=1e-12, atol=1e-12)
            if np.any(scores > before + 1e-10):
                raise RuntimeError('frozen best-iterate fitter discarded a better original start')
            sources, observed = [], []
            for pose, plan in zip(fitted, plans, strict=True):
                source, supported, _ = supported_nodes(pose, plan, rays, tracker.rig)
                sources.append(source)
                observed.append(supported)
            rows.append(dict(
                input_pose_xyz=batch.copy(), initial_scores=before, initial_loss_parts=before_parts,
                pose_xyz=fitted, scores=scores, loss_parts=parts, plans=plans,
                sources=np.stack(sources), observed=np.stack(observed), mask_supported=mask,
                boundary_coordinates=(fitted <= tracker.lower) | (fitted >= tracker.upper),
            ))
        result = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
        if tracker.initialize_count != 0:
            raise RuntimeError('local fitting unexpectedly ran a tracking initialization')
        result.update(elapsed_seconds=time.perf_counter() - started, batch_size=self.batch_size)
        return result

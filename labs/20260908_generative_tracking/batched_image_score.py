"""Separate evaluation of one refractive projection call per camera/image score.

Only concatenation and restoration of existing projected point groups differs
from the frozen reacquisition scorer. No evidence, scoring or threshold changes.
"""

from __future__ import annotations

import numpy as np
from reacquisition import ReacquisitionTracker
from tracker import _body_samples


def image_point_groups(pose, previous):
    body = _body_samples(pose)
    points = np.concatenate((pose.reshape(-1, 3), body.reshape(-1, 3),
                             previous.reshape(-1, 3)))
    sizes = (pose.size // 3, body.size // 3)
    return points, body.shape, sizes


def project_image_groups(camera, points, body_shape, sizes):
    projected = camera.project(points)
    first, second = sizes[0], sum(sizes)
    uv = projected[:first].reshape(3, 3, 2)
    body = projected[first:second].reshape(*body_shape[:-1], 2)
    old_uv = projected[second:].reshape(3, 3, 2)
    return uv, body, old_uv


class BatchedImageTracker(ReacquisitionTracker):
    def _image_score(self, frame, pose, previous, templates, observed):
        features = np.zeros((3, 3, 24), dtype=np.float32)
        quality = np.zeros((3, 3), dtype=np.float32)
        updated = templates.copy()
        if frame.evidence is None:
            return 0.0, updated, features, quality
        cfg = self.config.local
        total = 0.0
        points, body_shape, sizes = image_point_groups(pose, previous)
        for view, cam in enumerate(self.rig.cameras):
            uv, body, old_uv = project_image_groups(cam, points, body_shape, sizes)
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
                    if observed[fish, :2].all() and not overlap:
                        quality[fish, view] = score.appearance_quality
                    if key not in updated and quality[fish, view] >= 0.25:
                        updated[key] = score.appearance.copy()
        return float(total), updated, features, quality

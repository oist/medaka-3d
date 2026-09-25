"""Batch observation admission; the frozen single-pose implementation is the oracle.

Only the camera-call batch shape changes. Ray geometry stays at the precision used
by ``assigned_rays``; the two existing distance thresholds and their inclusivity
stay unchanged. A camera's iterative solver can differ in its last bits between
batch shapes, so projection parity is numerical rather than promised bitwise.
"""

from __future__ import annotations

import numpy as np
from node_geometry import assigned_rays

from medaka3d.types import VIEWS


def supported_nodes_batch(
    poses, plans, rays, rig, maximum_ray_mm=2.0, maximum_pixel_error=12.0
):
    """Return sources[B,F,V,N], observed[B,F,N], and ray distance[B,F,V,N].

    The lab has exactly three fish, views and nodes. Sources index the supplied
    current frame's detections. Missing sources remain -1; latent poses cannot
    manufacture observations. Empty batches are valid and return empty arrays.
    """
    poses = np.asarray(poses)
    plans = np.asarray(plans, np.int32)
    if poses.ndim != 4 or poses.shape[1:] != (3, 3, 3):
        raise ValueError("poses must have shape (batch,fish,node,xyz)")
    if plans.shape != poses.shape:
        raise ValueError("plans must have shape (batch,fish,view,node) matching poses")
    origin, direction, held = assigned_rays(rays, plans)
    delta = poses[:, :, None] - origin
    distance = np.linalg.norm(
        delta - np.sum(delta * direction, axis=-1, keepdims=True) * direction, axis=-1
    )
    valid = held & (distance <= maximum_ray_mm)
    if len(poses):
        for view, name in enumerate(VIEWS):
            projected = rig.camera(name).project(poses)
            # Retain the original scalar two-coordinate norm and comparison.
            for batch, fish, node in np.argwhere(valid[:, :, view]):
                row = plans[batch, fish, view, node]
                error = np.linalg.norm(
                    projected[batch, fish, node] - rays.points[name][row, node]
                )
                valid[batch, fish, view, node] &= (
                    np.isfinite(error) and error <= maximum_pixel_error
                )
    sources = np.where(valid, plans, -1).astype(np.int32)
    return sources, valid.sum(axis=2) >= 2, distance

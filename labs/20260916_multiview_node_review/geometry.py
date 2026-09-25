"""Memoryless geometry diagnostics for existing three-view fish node matches.

These quantities measure geometric consistency, not correctness of animal IDs.
View order is ``medaka3d.types.VIEWS`` and node order is mouth, pectoral, tail.
No temporal smoothing, correspondence search, or coordinate replacement occurs.
"""

from __future__ import annotations

import numpy as np

from medaka3d.calib import triangulate
from medaka3d.types import VIEWS


def _finite_rays(camera, uv):
    """Keep invalid pixels out of calibration solves and preserve missingness."""
    finite = np.isfinite(uv).all(axis=-1)
    origins = np.full((*uv.shape[:-1], 3), np.nan)
    directions = np.full_like(origins, np.nan)
    if finite.any():
        origins[finite], directions[finite] = camera.rays(uv[finite])
    norm = np.linalg.norm(directions, axis=-1)
    valid = finite & np.isfinite(origins).all(axis=-1) & np.isfinite(norm) & (norm > 0)
    directions = np.divide(
        directions,
        norm[..., None],
        out=np.full_like(directions, np.nan),
        where=valid[..., None],
    )
    return origins, directions, valid


def _finite_project(camera, pose):
    finite = np.isfinite(pose).all(axis=-1)
    pixels = np.full((*pose.shape[:-1], 2), np.nan)
    if finite.any():
        pixels[finite] = camera.project(pose[finite])
    # A calibration backend may return infinities for impossible image points.
    return np.where(np.isfinite(pixels).all(axis=-1)[..., None], pixels, np.nan)


def _reprojection(camera, pose, own_uv, valid_source):
    predicted = _finite_project(camera, pose)
    # Replace invalid source inputs before arithmetic; absent evidence must not
    # emit infinities, become zero residual, or contaminate another node.
    delta = predicted - np.where(valid_source[..., None], own_uv, np.nan)
    error = np.linalg.norm(delta, axis=-1)
    return np.where(valid_source & np.isfinite(error), error, np.nan)


def geometry_metrics(rig, uv, frozen_pose=None):
    """Diagnose a fixed set of 2D node correspondences, with optional batches.

    Args:
        rig: Calibrated cameras named and ordered according to ``VIEWS``.
        uv: (..., 3 slots, 3 views, 3 nodes, 2) image coordinates, px.
        frozen_pose: Optional (..., 3 slots, 3 nodes, 3) independent frozen
            coordinates, mm. Only their own-source reprojection is measured.

    Returns:
        ``pose`` (..., slots, nodes, xyz) is the ray-triangulated pose;
        ``ray_rms_mm``, ``view_counts`` and ``observed`` are (..., slots, nodes).
        ``reprojection_px`` and ``leave_one_out_px`` are (..., slots, views,
        nodes). Leave-one-out uses only the other two camera rays to predict a
        held-out measurement and requires all three rays plus a solvable pair.
        Missing/degenerate measurements are NaN, never zero. ``observed`` means
        a finite solution with at least two rays, without an accuracy cutoff.
        ``mp_mm``, ``pt_mm``, ``mt_mm`` and ``bend_deg`` are (..., slots).
        Bend is 0 for a straight mouth-to-pectoral-to-tail chain, 180 for a
        folded-back chain, and NaN for a missing or numerically zero segment.
        If frozen_pose is supplied, ``frozen_reprojection_px`` has the same
        axes as reprojection_px; it never replaces pose or its ray residual.
    """
    uv = np.asarray(uv, dtype=float)
    if uv.ndim < 4 or uv.shape[-4:] != (3, 3, 3, 2):
        raise ValueError("uv must have shape (..., 3 slots, 3 views, 3 nodes, 2)")
    if tuple(rig.views) != VIEWS:
        raise ValueError(f"rig view order must be {VIEWS}")
    if hasattr(rig, "world_units") and rig.world_units != "mm":
        raise ValueError("geometry distances require calibration world_units='mm'")
    pose_shape = (*uv.shape[:-4], 3, 3, 3)
    if frozen_pose is not None:
        frozen_pose = np.asarray(frozen_pose, dtype=float)
        if frozen_pose.shape != pose_shape:
            raise ValueError(f"frozen_pose must have shape {pose_shape}")

    rays = [_finite_rays(rig.camera(view), uv[..., vi, :, :])
            for vi, view in enumerate(VIEWS)]
    origins = np.stack([ray[0] for ray in rays], axis=-2)
    directions = np.stack([ray[1] for ray in rays], axis=-2)
    valid = np.stack([ray[2] for ray in rays], axis=-1)
    # Ray axes are (..., slots, nodes, views, xyz), as triangulate expects.
    pose, ray_rms = triangulate(origins, directions, valid)
    view_counts = valid.sum(axis=-1)
    observed = (view_counts >= 2) & np.isfinite(pose).all(axis=-1)
    ray_rms = np.where(observed & np.isfinite(ray_rms), ray_rms, np.nan)

    reprojection, leave_one_out, frozen_reprojection = [], [], []
    for vi, view in enumerate(VIEWS):
        camera = rig.camera(view)
        own_uv = uv[..., vi, :, :]
        reprojection.append(_reprojection(camera, pose, own_uv, valid[..., vi]))
        pair_valid = valid.copy()
        pair_valid[..., vi] = False
        pair_pose, _ = triangulate(origins, directions, pair_valid)
        held_out_valid = valid.all(axis=-1) & np.isfinite(pair_pose).all(axis=-1)
        leave_one_out.append(_reprojection(camera, pair_pose, own_uv, held_out_valid))
        if frozen_pose is not None:
            frozen_reprojection.append(
                _reprojection(camera, frozen_pose, own_uv, valid[..., vi])
            )

    mouth_to_pectoral = pose[..., 1, :] - pose[..., 0, :]
    pectoral_to_tail = pose[..., 2, :] - pose[..., 1, :]
    mouth_to_tail = pose[..., 2, :] - pose[..., 0, :]
    mp_mm = np.linalg.norm(mouth_to_pectoral, axis=-1)
    pt_mm = np.linalg.norm(pectoral_to_tail, axis=-1)
    mt_mm = np.linalg.norm(mouth_to_tail, axis=-1)
    cosine = np.divide(
        np.sum(mouth_to_pectoral * pectoral_to_tail, axis=-1),
        mp_mm * pt_mm,
        out=np.full(mp_mm.shape, np.nan),
        where=np.isfinite(mp_mm) & np.isfinite(pt_mm) & (mp_mm > 1e-9) & (pt_mm > 1e-9),
    )
    result = {
        "pose": pose,
        "ray_rms_mm": ray_rms,
        "reprojection_px": np.stack(reprojection, axis=-2),
        "leave_one_out_px": np.stack(leave_one_out, axis=-2),
        "mp_mm": mp_mm,
        "pt_mm": pt_mm,
        "mt_mm": mt_mm,
        "bend_deg": np.degrees(np.arccos(np.clip(cosine, -1, 1))),
        "view_counts": view_counts,
        "observed": observed,
    }
    if frozen_pose is not None:
        result["frozen_reprojection_px"] = np.stack(frozen_reprojection, axis=-2)
    return result

"""Measure whole-body agreement with independent, variable-count instance masks.

These are image distances, not a tracking cost or a rejection threshold. Every
nonempty supplied mask participates, independently of SLEAP row compatibility.
The union comparison may switch masks between samples; the single-mask result
cannot. A difference can expose a body assembled across separated instances,
but segmentation errors, occlusion and straight-segment approximation remain.
"""

from __future__ import annotations

import numpy as np
from sparse_instance_body_score import SparseDistanceFields

SAMPLE_LABELS = ("M", "C", "T", "MC_025", "MC_050", "MC_075", "CT_025", "CT_050", "CT_075")


def measure_body_mask_containment(pose_xyz, project, fields, mask_indices=None):
    """Return per-body distances for poses [N, M/C/T, XYZ], with no assignments.

    Fields must contain all nonempty instances for the camera. Their original
    detector indices are optional labels, never eligibility information. No
    fields means unavailable evidence, not a zero-cost successful match.
    """
    pose = np.asarray(pose_xyz, dtype=np.float64)
    if pose.ndim != 3 or pose.shape[1:] != (3, 3) or not np.isfinite(pose).all():
        raise ValueError("finite complete body poses [N,3,3] required")
    fractions = np.array([0.25, 0.5, 0.75])[None, :, None]
    mc = pose[:, 0:1] + fractions * (pose[:, 1:2] - pose[:, 0:1])
    ct = pose[:, 1:2] + fractions * (pose[:, 2:3] - pose[:, 1:2])
    samples = np.concatenate((pose, mc, ct), axis=1)
    projected = np.asarray(project(samples), dtype=np.float64)
    if projected.shape != (len(pose), 9, 2) or not np.isfinite(projected).all():
        raise ValueError("finite projections [N,9,2] required")
    result = dict(
        available=fields is not None,
        sample_labels=list(SAMPLE_LABELS),
        sample_xyz=samples,
        projected_xy=projected,
        SLEAP_compatibility_used=False,
        support_flags_used=False,
        physical_IDs_assigned=False,
        rejection_threshold_applied=False,
    )
    if fields is None:
        if mask_indices is not None and len(mask_indices):
            raise ValueError("mask labels without fields")
        return result
    if not isinstance(fields, SparseDistanceFields) or not fields.trees:
        raise ValueError("nonempty sparse instance fields required")
    indices = np.arange(len(fields.trees)) if mask_indices is None else np.asarray(mask_indices)
    if (
        indices.shape != (len(fields.trees),)
        or indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("unique nonnegative original mask indices required")
    distances = fields.sample(projected)
    if distances.shape != (len(pose), 9, len(indices)) or not np.isfinite(distances).all():
        raise ValueError("finite per-body, sample, mask distances required")
    per_mask_max = distances.max(axis=1)
    choices = np.argmin(per_mask_max, axis=1)
    single = np.take_along_axis(per_mask_max, choices[:, None], axis=1)[:, 0]
    union_by_sample = distances.min(axis=2)
    union = union_by_sample.max(axis=1)
    if np.any(single < union):
        raise RuntimeError("single-mask containment cannot improve on the sample-wise union")
    best_distances = np.take_along_axis(distances, choices[:, None, None], axis=2)[:, :, 0]
    result.update(
        mask_indices=indices.copy(),
        distances_by_body_sample_mask=distances,
        maximum_distance_by_body_mask_px=per_mask_max,
        best_single_mask=indices[choices],
        best_single_mask_tie_count=(per_mask_max == single[:, None]).sum(axis=1),
        best_single_mask_distances_px=best_distances,
        single_mask_max_distance_px=single,
        single_mask_mean_distance_px=best_distances.mean(axis=1),
        endpoint_single_mask_max_distance_px=distances[:, :3].max(axis=1).min(axis=1),
        union_distances_by_sample_px=union_by_sample,
        union_max_distance_px=union,
        mask_switch_advantage_px=single - union,
    )
    return result

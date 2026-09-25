"""Attach proposed identities to frozen geometry without creating observations.

This is an export helper, not a confidence policy or human correction. Original
review flags remain attached to their raw-slot provenance and are also shown in
the proposed ordering. No original candidate array is modified.
"""

import numpy as np


def build_identity_output(geometry, identity_to_raw_slot):
    frames = np.asarray(geometry["frame_idx"])
    mapping = np.asarray(identity_to_raw_slot)
    if (frames.ndim != 1 or frames.dtype.kind not in "iu" or not len(frames)
            or np.any(frames < 0) or np.any(frames > np.iinfo(np.int64).max)):
        raise ValueError("native frame_idx must be nonnegative int64-compatible integers")
    frames = frames.astype(np.int64, copy=True)
    if np.any(np.diff(frames) != 1):
        raise ValueError("a complete consecutive native frame sequence is required")
    n = len(frames)
    if (mapping.shape != (n, 3) or mapping.dtype.kind not in "iu"
            or not np.all(np.sort(mapping, axis=1) == np.arange(3))):
        raise ValueError("each proposed identity mapping must be one permutation of 0,1,2")
    shapes = {
        "pose_xyz": (n, 3, 3, 3), "sources": (n, 3, 3, 3),
        "observed": (n, 3, 3), "identity_review": (n, 3),
        "xyz": (n, 3, 3), "accepted_xyz": (n, 3, 3),
    }
    for key, shape in shapes.items():
        if key not in geometry or np.asarray(geometry[key]).shape != shape:
            raise ValueError(f"missing or invalid frozen geometry array: {key}")
    pose, observed = np.asarray(geometry["pose_xyz"]), np.asarray(geometry["observed"])
    source, review = np.asarray(geometry["sources"]), np.asarray(geometry["identity_review"])
    if (not np.isfinite(pose).all() or observed.dtype != bool or review.dtype != bool
            or source.dtype.kind not in "iu" or np.any(source < -1)):
        raise ValueError("frozen geometry, source, support and review types are inconsistent")
    direct = np.where(observed[:, :, 1, None], pose[:, :, 1], np.nan)
    accepted = np.where(~review[..., None], direct, np.nan)
    if (not np.array_equal(geometry["xyz"], direct, equal_nan=True)
            or not np.array_equal(geometry["accepted_xyz"], accepted, equal_nan=True)):
        raise ValueError("frozen pectoral coordinates must obey their original support and review")

    def reorder(value):
        value = np.asarray(value)
        indices = mapping[(slice(None), slice(None)) + (None,) * (value.ndim - 2)]
        return np.take_along_axis(value, indices, axis=1).copy()

    return {
        "frame_idx": frames.copy(),
        "identity_to_raw_slot": mapping.astype(np.int64, copy=True),
        "raw_pose_xyz": pose.copy(),
        "raw_sources": source.copy(),
        "raw_observed": observed.copy(),
        "raw_identity_review": review.copy(),
        "proposed_pose_xyz": reorder(pose),
        "proposed_sources": reorder(source),
        "proposed_observed": reorder(observed),
        "proposed_xyz": reorder(direct),
        "proposed_node_xyz": reorder(np.where(observed[..., None], pose, np.nan)),
        "prior_review_for_proposed_identity": reorder(review),
        "conservative_accepted_xyz": reorder(accepted),
    }

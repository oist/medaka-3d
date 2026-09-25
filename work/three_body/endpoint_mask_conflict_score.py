"""Post-fit M/C/T conflicts against every independent nonempty instance mask.

A split requires three uniquely matched endpoints and differing original mask
indices within one camera/frame. Missing and ambiguous evidence are unknown,
not split evidence. Detector indices never identify fish across views or time.
This module produces diagnostics only: it does not change poses or costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from sparse_instance_body_score import SparseDistanceFields


@dataclass(frozen=True)
class EndpointMaskObservation:
    fields: SparseDistanceFields | None
    image_shape_hw: tuple[int, int] | None
    mask_indices: tuple[int, ...]
    original_mask_areas: tuple[int, ...] | None
    reason: str

    @property
    def available(self):
        return self.fields is not None

    @property
    def evidence(self):
        # Fresh containers prevent diagnostics consumers mutating the observation.
        return dict(
            available=self.available,
            reason=self.reason,
            image_shape_hw=self.image_shape_hw,
            original_mask_count=(
                None if self.original_mask_areas is None else len(self.original_mask_areas)
            ),
            original_mask_areas=self.original_mask_areas,
            nonempty_mask_indices=self.mask_indices,
            SLEAP_compatibility_used=False,
            detection_confidence_used=False,
            physical_IDs_assigned=False,
        )


def prepare_endpoint_mask_observation(masks):
    """Own sparse fields for all nonempty masks, preserving detector indices."""
    if masks is None:
        return EndpointMaskObservation(None, None, (), None, "mask_view_unavailable")
    if (
        not isinstance(masks, np.ndarray)
        or masks.ndim != 3
        or masks.dtype != np.dtype(bool)
        or min(masks.shape[1:]) < 1
    ):
        raise ValueError("boolean masks [K,H,W] with positive image dimensions required")
    areas = masks.sum(axis=(1, 2), dtype=np.int64)
    indices = np.flatnonzero(areas)
    fields = SparseDistanceFields.from_masks(masks[indices]) if len(indices) else None
    return EndpointMaskObservation(
        fields,
        tuple(map(int, masks.shape[1:])),
        tuple(map(int, indices)),
        tuple(map(int, areas)),
        "all_nonempty_masks" if fields is not None else "no_nonempty_masks",
    )


def score_endpoint_mask_conflicts(poses, project, observation, tolerance_px=6.0):
    """Classify [candidate,3 bodies,3 M/C/T,3 XYZ] for a single camera.

    Invalid geometry takes precedence over unavailable masks. Unavailable masks
    skip projection; other invalid bodies are never sent to the projector. An
    invalid projection is unknown, never a same-instance or split decision.
    Node counts are -1 when unmeasured and unique indices -1 when nonunique.
    Nonfinite body distances use infinity, with explicit invalid classification.
    """
    if not isinstance(observation, EndpointMaskObservation):
        raise ValueError("EndpointMaskObservation required")
    if (
        isinstance(tolerance_px, (bool, np.bool_))
        or not isinstance(tolerance_px, Real)
        or not np.isfinite(tolerance_px)
        or tolerance_px < 0
    ):
        raise ValueError("finite nonnegative numeric endpoint tolerance required")
    pose = np.asarray(poses, dtype=np.float64)
    if pose.ndim != 4 or pose.shape[1:] != (3, 3, 3):
        raise ValueError("candidate body poses [H,3,3,3] required")
    shape = pose.shape[:2]
    geometry_finite = np.isfinite(pose).all(axis=(-2, -1))
    labels = np.full(shape, "unavailable", dtype="<U24")
    labels[~geometry_finite] = "invalid_geometry"
    counts = np.full((*shape, 3), -1, dtype=np.int64)
    unique = np.full((*shape, 3), -1, dtype=np.int64)
    split = np.zeros(shape, dtype=bool)
    unknown = np.ones(shape, dtype=bool)
    projection_finite = projected = distances = None
    indices = np.array(observation.mask_indices, dtype=np.int64)
    if observation.available:
        projected = np.full((*shape, 3, 2), np.nan, dtype=np.float64)
        projection_finite = np.zeros(shape, dtype=bool)
        distances = np.full((*shape, 3, len(indices)), np.inf, dtype=np.float64)
        if geometry_finite.any():
            # Boolean selection owns a copy, even if project mutates its argument.
            xy = np.asarray(project(pose[geometry_finite]), dtype=np.float64)
            if xy.shape != (int(geometry_finite.sum()), 3, 2):
                raise ValueError("project must preserve body/MCT axes as [N,3,2]")
            projected[geometry_finite] = xy
            projection_finite[geometry_finite] = np.isfinite(xy).all(axis=(-2, -1))
        labels[geometry_finite] = "invalid_projection"
        valid = geometry_finite & projection_finite
        if valid.any():
            measured = observation.fields.sample(projected[valid])
            if (
                measured.shape != (int(valid.sum()), 3, len(indices))
                or not np.isfinite(measured).all()
                or np.any(measured < 0)
            ):
                raise ValueError("finite nonnegative mask distances required for valid projections")
            distances[valid] = measured
            near = measured <= float(tolerance_px)
            node_counts = near.sum(axis=-1)
            node_unique = np.where(node_counts == 1, indices[near.argmax(axis=-1)], -1)
            counts[valid] = node_counts
            unique[valid] = node_unique
            uncovered = (node_counts == 0).any(axis=-1)
            ambiguous = (node_counts > 1).any(axis=-1)
            all_unique = ~(uncovered | ambiguous)
            same = all_unique & (node_unique == node_unique[:, :1]).all(axis=-1)
            local_split = all_unique & ~same
            local_labels = np.full(int(valid.sum()), "ambiguous_endpoint", dtype="<U24")
            local_labels[uncovered] = "uncovered_endpoint"
            local_labels[same] = "same_instance"
            local_labels[local_split] = "split_instances"
            labels[valid] = local_labels
            split[valid] = local_split
            unknown[valid] = ~all_unique
    return dict(
        available=observation.available,
        tolerance_px=float(tolerance_px),
        split_by_body=split,
        unknown_by_body=unknown,
        classification_by_body=labels,
        geometry_finite_by_body=geometry_finite,
        projection_finite_by_body=projection_finite,
        distances_by_candidate_body_node_mask=distances,
        nearby_mask_count_by_node=counts,
        unique_mask_by_node=unique,
        uncovered_by_node=counts == 0,
        ambiguous_by_node=counts > 1,
        projected_xy=projected,
        mask_indices=indices,
        observation_evidence=observation.evidence,
    )

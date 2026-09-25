"""CPU frame-local whole-body/instance consistency, with observation-only eligibility.

The observation gate sees masks and untracked SLEAP rows only. When eligible,
every candidate's M/C/T (including latent points) and fixed 3D segment interiors
are projected through the supplied calibrated camera. All six bijections of
three bodies to three masks are scored. Mask indices are never permanent IDs.
This module is a scoring component, not a tracker, fitter, confidence model or
assertion that observation eligibility implies correct segmentation/pose/ID.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

PERMUTATIONS = np.array(list(itertools.permutations(range(3))), dtype=np.int64)
PERMUTATIONS.flags.writeable = False


@dataclass(frozen=True)
class InstanceBodyConfig:
    """Explicit initial diagnostic choices, not fitted or experimentally validated."""

    minimum_instance_gap_px: float = 3.0
    sleap_mask_tolerance_px: float = 8.0
    endpoint_tolerance_px: float = 6.0
    interior_tolerance_px: float = 2.0
    pixel_scale_px: float = 6.0
    worst_node_weight: float = 1.0
    segment_leakage_weight: float = 1.0
    interior_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    tie_atol: float = 1e-12

    def __post_init__(self):
        for name in (
            "minimum_instance_gap_px",
            "sleap_mask_tolerance_px",
            "endpoint_tolerance_px",
            "interior_tolerance_px",
            "worst_node_weight",
            "segment_leakage_weight",
            "tie_atol",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError("finite nonnegative setting required: " + name)
        if (
            isinstance(self.pixel_scale_px, bool)
            or not np.isfinite(self.pixel_scale_px)
            or self.pixel_scale_px <= 0
        ):
            raise ValueError("positive finite pixel scale required")
        fractions = self.interior_fractions
        if (
            not isinstance(fractions, tuple)
            or not fractions
            or any(isinstance(x, bool) or not np.isfinite(x) or not 0 < x < 1 for x in fractions)
            or any(a >= b for a, b in itertools.pairwise(fractions))
        ):
            raise ValueError("ordered unique fixed interior fractions in (0,1) required")


DEFAULT_CONFIG = InstanceBodyConfig()


def sample_distance_fields(fields: np.ndarray, projected_xy: np.ndarray) -> np.ndarray:
    """Bilinear distance field samples [..., instance]; outside-image distance is added.

    Fields are Euclidean distances on pixel centers. Outside the image, the
    sampled border distance plus the distance to that border is a conservative
    extension, not a claim of exact continuous distance to the mask. Nonfinite
    projections get +inf and cannot silently become zero-penalty samples.
    """
    xy = np.asarray(projected_xy, dtype=np.float64)
    if fields.ndim != 3 or xy.ndim < 1 or xy.shape[-1] != 2:
        raise ValueError("distance fields [I,H,W] and coordinates [...,2] required")
    count, height, width = fields.shape
    if count < 1 or height < 1 or width < 1:
        raise ValueError("nonempty distance fields required")
    flat = xy.reshape(-1, 2)
    finite = np.isfinite(flat).all(axis=-1)
    safe = np.where(finite[:, None], flat, 0.0)
    clipped = np.clip(safe, [0, 0], [width - 1, height - 1])
    outside = np.linalg.norm(safe - clipped, axis=-1)
    values = np.stack(
        [
            map_coordinates(field, clipped[:, ::-1].T, order=1, mode="nearest", prefilter=False)
            + outside
            for field in fields
        ],
        axis=-1,
    )
    values[~finite] = np.inf
    return values.reshape(*xy.shape[:-1], count)


@dataclass(frozen=True)
class InstanceObservation:
    """Prepared once per view/frame; includes no candidate or permanent identity state."""

    eligible: bool
    reason: str
    image_shape_hw: tuple[int, int] | None
    distance_fields: np.ndarray | None
    sleap_row_by_instance: np.ndarray | None
    evidence: dict
    config: InstanceBodyConfig


def prepare_instance_observation(masks, sleap_points, config=DEFAULT_CONFIG) -> InstanceObservation:
    """Require exactly three separated masks with unambiguous complete SL grouping.

    Each mask needs exactly one complete M/C/T SL row wholly within the declared
    tolerance. Extra SL rows matching no mask are reported and ignored. Duplicate
    qualifying rows, a row qualifying multiple masks, incomplete support, overlap
    or insufficient separation make the entire view neutral. No detection score
    is invented; FrameObservation provides no confidence values here.
    """
    if not isinstance(config, InstanceBodyConfig):
        raise ValueError("InstanceBodyConfig required")
    evidence = dict(
        candidate_pose_used_for_eligibility=False,
        detection_confidence_used=False,
        permanent_IDs_assigned=False,
        mask_instance_count=None,
        complete_SL_rows=None,
        qualifying_SL_rows=None,
        unmatched_complete_SL_rows=None,
    )
    shape = None

    def unavailable(reason):
        return InstanceObservation(False, reason, shape, None, None, evidence, config)

    if masks is None:
        return unavailable("mask_view_unavailable")
    if not isinstance(masks, np.ndarray) or masks.ndim != 3 or masks.dtype != np.dtype(bool):
        raise ValueError("boolean instance masks [I,H,W] required")
    if min(masks.shape[1:]) < 1:
        raise ValueError("nonempty mask image dimensions required")
    shape = tuple(map(int, masks.shape[1:]))
    evidence["mask_instance_count"] = len(masks)
    if len(masks) != 3:
        return unavailable("exactly_three_mask_instances_required")
    areas = masks.sum(axis=(1, 2), dtype=np.int64)
    evidence["mask_area_pixels"] = areas.tolist()
    if not np.all(areas > 0):
        return unavailable("empty_mask_instance")
    intersections = [
        int(np.count_nonzero(masks[a] & masks[b])) for a, b in ((0, 1), (0, 2), (1, 2))
    ]
    evidence["pair_intersection_pixels_01_02_12"] = intersections
    if any(intersections):
        return unavailable("overlapping_mask_instances")
    fields = np.stack([distance_transform_edt(~mask) for mask in masks])
    gaps = [float(fields[a][masks[b]].min()) for a, b in ((0, 1), (0, 2), (1, 2))]
    evidence["pair_mask_distance_px_01_02_12"] = gaps
    if min(gaps) <= config.minimum_instance_gap_px:
        return unavailable("mask_instances_not_clearly_separated")
    if sleap_points is None:
        return unavailable("SLEAP_view_unavailable")
    points = np.asarray(sleap_points)
    if points.ndim != 3 or points.shape[1:] != (3, 2) or points.dtype.kind != "f":
        raise ValueError("floating untracked SLEAP rows [R,3,2] required")
    complete = np.isfinite(points).all(axis=(1, 2))
    distances = sample_distance_fields(fields, points)
    qualifying = complete[:, None] & (distances <= config.sleap_mask_tolerance_px).all(axis=1)
    evidence["complete_SL_rows"] = np.flatnonzero(complete).tolist()
    evidence["qualifying_SL_rows"] = [np.flatnonzero(qualifying[:, k]).tolist() for k in range(3)]
    evidence["unmatched_complete_SL_rows"] = np.flatnonzero(
        complete & ~qualifying.any(axis=1)
    ).tolist()
    if np.any(qualifying.sum(axis=1) > 1):
        return unavailable("SLEAP_row_matches_multiple_instances")
    per_instance = qualifying.sum(axis=0)
    if np.any(per_instance == 0):
        return unavailable("missing_complete_SLEAP_group_for_instance")
    if np.any(per_instance != 1):
        return unavailable("duplicate_SLEAP_groups_for_instance")
    row_by_instance = np.argmax(qualifying, axis=0).astype(np.int64)
    fields.flags.writeable = False
    row_by_instance.flags.writeable = False
    return InstanceObservation(
        True,
        "three_separated_instances_with_unique_SLEAP_groups",
        shape,
        fields,
        row_by_instance,
        evidence,
        config,
    )


def sample_body_xyz(pose_xyz: np.ndarray, fractions: tuple[float, ...]) -> np.ndarray:
    """M/C/T first, then M-C interiors, then C-T interiors, all interpolated in 3D."""
    pose = np.asarray(pose_xyz)
    if pose.ndim != 4 or pose.shape[1:] != (3, 3, 3) or pose.dtype.kind != "f" or len(pose) == 0:
        raise ValueError("floating candidate bodies [B,3,3,3] required")
    a = np.asarray(fractions, dtype=np.float64)
    segments = [
        pose[:, :, start, None, :] * (1.0 - a[None, None, :, None])
        + pose[:, :, stop, None, :] * a[None, None, :, None]
        for start, stop in ((0, 1), (1, 2))
    ]
    return np.concatenate([pose, *segments], axis=2)


def score_instance_bodies(pose_xyz, project: Callable, observation: InstanceObservation) -> dict:
    """Score a candidate bank using one-to-one, frame-local whole-body assignments.

    Pair cost = worst endpoint excess^2 / scale^2 + mean squared interior
    excess / scale^2, with explicit weights. Endpoints have a larger tolerance
    for tips outside segmentation. All latent endpoints are included; observed
    flags are intentionally not an input. Six joint costs sum over three bodies.

    Unavailable observations produce a zero *neutral term* and no assignments.
    Eligible observations never become unavailable because a candidate is poor:
    nonfinite candidate projections instead get infinite cost. This term has no
    production fit weight yet and has not been validated on a full recording.
    """
    if not isinstance(observation, InstanceObservation):
        raise ValueError("prepared InstanceObservation required")
    samples = sample_body_xyz(pose_xyz, observation.config.interior_fractions)
    count = len(samples)
    result = dict(
        eligible=observation.eligible,
        reason=observation.reason,
        observation_evidence=observation.evidence,
        neutral=not observation.eligible,
        best_cost=np.zeros(count),
        best_assignment=None,
        assignment_costs=None,
        optimal_assignment_mask=None,
        body_instance_costs=None,
        worst_endpoint_costs=None,
        segment_leakage_costs=None,
        projected_samples=None,
        distances_by_body_sample_instance=None,
        permanent_IDs_assigned=False,
        observed_flags_used=False,
        instance_assignments_are_frame_local=True,
    )
    if not observation.eligible:
        return result
    projected = np.asarray(project(samples), dtype=np.float64)
    if projected.shape != (*samples.shape[:-1], 2):
        raise ValueError("projector must preserve batch/body/sample axes and return xy")
    distances = sample_distance_fields(observation.distance_fields, projected)
    cfg = observation.config
    endpoint_excess = np.maximum(distances[:, :, :3] - cfg.endpoint_tolerance_px, 0.0)
    interior_excess = np.maximum(distances[:, :, 3:] - cfg.interior_tolerance_px, 0.0)
    endpoint_cost = (endpoint_excess.max(axis=2) / cfg.pixel_scale_px) ** 2
    segment_cost = np.mean((interior_excess / cfg.pixel_scale_px) ** 2, axis=2)
    pair_cost = np.zeros_like(endpoint_cost)
    if cfg.worst_node_weight:
        pair_cost += cfg.worst_node_weight * endpoint_cost
    if cfg.segment_leakage_weight:
        pair_cost += cfg.segment_leakage_weight * segment_cost
    # A nonfinite 3D input/projection invalidates that candidate regardless of configured weights.
    valid = np.isfinite(samples).all(axis=(1, 2, 3)) & np.isfinite(projected).all(axis=(1, 2, 3))
    pair_cost[~valid] = np.inf
    joint = np.stack(
        [pair_cost[:, np.arange(3), permutation].sum(axis=1) for permutation in PERMUTATIONS],
        axis=-1,
    )
    selected = np.argmin(joint, axis=1)
    best = joint[np.arange(count), selected]
    assignment = PERMUTATIONS[selected].copy()
    assignment[~np.isfinite(best)] = -1
    optimal = np.isfinite(joint) & np.isclose(joint, best[:, None], rtol=0, atol=cfg.tie_atol)
    result.update(
        best_cost=best,
        best_assignment=assignment,
        assignment_costs=joint,
        optimal_assignment_mask=optimal,
        body_instance_costs=pair_cost,
        worst_endpoint_costs=endpoint_cost,
        segment_leakage_costs=segment_cost,
        projected_samples=projected,
        distances_by_body_sample_instance=distances,
        candidate_projection_finite=valid,
    )
    return result

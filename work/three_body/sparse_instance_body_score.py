"""Standalone CPU sparse equivalent of the frozen instance-body distance semantics.

No full-image distance array, lazy ndarray, runtime substitution, tracker, or GPU
is used. Each nonempty mask owns a cKDTree of foreground pixel centers. Samples
query exact nearest centers at four integer lattice corners, then use SciPy's
linear interpolation and the frozen clipped-border-plus-outside extension.
This deliberately does NOT use nearest-mask distance at the subpixel point.

The explicit observation/scoring adapters below retain the frozen formulas from
instance_body_score.py (SHA cb9e04a20160f2b681aa1b7cca46ed5b4017c7b05760f9dbbeff691bf058cb87).
They are separate components and are not integrated into an existing tracker.
Numerical parity must be qualified on the target workload before adoption.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from instance_body_score import DEFAULT_CONFIG, PERMUTATIONS, InstanceBodyConfig, sample_body_xyz
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

REFERENCE_SOURCE_SHA256 = "cb9e04a20160f2b681aa1b7cca46ed5b4017c7b05760f9dbbeff691bf058cb87"


@dataclass(frozen=True)
class SparseDistanceFields:
    """Owned foreground-center indices for nonempty boolean masks; no dense EDT."""

    image_shape_hw: tuple[int, int]
    centers_yx: tuple[np.ndarray, ...]
    trees: tuple[cKDTree, ...]

    @classmethod
    def from_masks(cls, masks: np.ndarray) -> SparseDistanceFields:
        if (
            not isinstance(masks, np.ndarray)
            or masks.ndim != 3
            or masks.dtype != np.dtype(bool)
            or min(masks.shape) < 1
        ):
            raise ValueError("nonempty boolean instance masks [I,H,W] required")
        centers, trees = [], []
        for mask in masks:
            pixels = np.argwhere(mask).astype(np.int64, copy=False)
            if not len(pixels):
                raise ValueError("a sparse distance field requires a nonempty foreground mask")
            pixels.flags.writeable = False
            tree = cKDTree(pixels, copy_data=True)
            tree.data.flags.writeable = False
            centers.append(pixels)
            trees.append(tree)
        return cls(tuple(map(int, masks.shape[1:])), tuple(centers), tuple(trees))

    def _integer_distances(self, mask_index: int, lattice_yx: np.ndarray) -> np.ndarray:
        # eps=0 and workers=1: exact nearest-center queries without taking other
        # workload cores. Recompute the lattice squared distance from the returned
        # center; a different equidistant center must not change the distance.
        _, nearest = self.trees[mask_index].query(lattice_yx, k=1, eps=0, workers=1)
        delta = lattice_yx - self.centers_yx[mask_index][nearest]
        squared = np.sum(delta * delta, axis=-1, dtype=np.int64)
        return np.sqrt(squared.astype(np.float64))

    def minimum_separation(self, first: int, second: int) -> float:
        """Minimum foreground pixel-center distance, identical to EDT[first][second]."""
        return float(self._integer_distances(first, self.centers_yx[second]).min())

    def sample(self, projected_xy: np.ndarray) -> np.ndarray:
        """Return [..., instance] using integer EDT corners, not subpixel NN distance."""
        xy = np.asarray(projected_xy, dtype=np.float64)
        if xy.ndim < 1 or xy.shape[-1] != 2:
            raise ValueError("coordinates [...,2] required")
        count = len(self.trees)
        flat = xy.reshape(-1, 2)
        if not len(flat):
            return np.empty((*xy.shape[:-1], count), dtype=np.float64)
        finite = np.isfinite(flat).all(axis=-1)
        safe = np.where(finite[:, None], flat, 0.0)
        height, width = self.image_shape_hw
        clipped = np.clip(safe, [0, 0], [width - 1, height - 1])
        outside = np.linalg.norm(safe - clipped, axis=-1)
        yx = clipped[:, ::-1]
        lower = np.floor(yx).astype(np.int64)
        upper = np.minimum(lower + 1, [height - 1, width - 1])
        corners = np.stack(
            [
                lower,
                np.stack([lower[:, 0], upper[:, 1]], axis=-1),
                np.stack([upper[:, 0], lower[:, 1]], axis=-1),
                upper,
            ],
            axis=1,
        )
        unique, inverse = np.unique(corners.reshape(-1, 2), axis=0, return_inverse=True)
        # Integer first coordinates select each query's own 2x2 patch. SciPy
        # multiplies that dimension by 1/0, retaining the same y/x linear weights
        # and summation as the frozen 2D map_coordinates call. No cross-query blend.
        patch_coordinates = np.vstack([np.arange(len(flat)), (yx - lower).T])
        result = []
        for mask_index in range(count):
            values = self._integer_distances(mask_index, unique)[inverse].reshape(-1, 2, 2)
            result.append(
                map_coordinates(values, patch_coordinates, order=1, mode="nearest", prefilter=False)
                + outside
            )
        sampled = np.stack(result, axis=-1)
        sampled[~finite] = np.inf
        return sampled.reshape(*xy.shape[:-1], count)


@dataclass(frozen=True)
class SparseInstanceObservation:
    """Same observation-only gate information, with an explicit sparse distance backend."""

    eligible: bool
    reason: str
    image_shape_hw: tuple[int, int] | None
    sparse_fields: SparseDistanceFields | None
    sleap_row_by_instance: np.ndarray | None
    evidence: dict
    config: InstanceBodyConfig


def prepare_sparse_instance_observation(
    masks, sleap_points, config=DEFAULT_CONFIG
) -> SparseInstanceObservation:
    """Frozen gate decisions/evidence, with foreground-tree gaps and sparse samples."""
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
        return SparseInstanceObservation(False, reason, shape, None, None, evidence, config)

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
    fields = SparseDistanceFields.from_masks(masks)
    gaps = [fields.minimum_separation(a, b) for a, b in ((0, 1), (0, 2), (1, 2))]
    evidence["pair_mask_distance_px_01_02_12"] = gaps
    if min(gaps) <= config.minimum_instance_gap_px:
        return unavailable("mask_instances_not_clearly_separated")
    if sleap_points is None:
        return unavailable("SLEAP_view_unavailable")
    points = np.asarray(sleap_points)
    if points.ndim != 3 or points.shape[1:] != (3, 2) or points.dtype.kind != "f":
        raise ValueError("floating untracked SLEAP rows [R,3,2] required")
    complete = np.isfinite(points).all(axis=(1, 2))
    distances = fields.sample(points)
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
    row_by_instance.flags.writeable = False
    return SparseInstanceObservation(
        True,
        "three_separated_instances_with_unique_SLEAP_groups",
        shape,
        fields,
        row_by_instance,
        evidence,
        config,
    )


def score_sparse_instance_bodies(
    pose_xyz, project: Callable, observation: SparseInstanceObservation
) -> dict:
    """Explicit sparse counterpart of the frozen score formulas and result keys."""
    if not isinstance(observation, SparseInstanceObservation):
        raise ValueError("prepared SparseInstanceObservation required")
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
    distances = observation.sparse_fields.sample(projected)
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

"""Partial semantic rows and variable-count masks for three projected bodies.

This is a bounded observation-agreement term, not a confidence or identity model.
Real SLEAP rows are used at most once; unassigned bodies/detections are allowed.
Masks can be shared by different rows (occlusion) and are never arbitrarily
merged. Missing nodes give no reward, with a fixed denominator for all rows.
All latent candidate nodes are scored. A finite soft sum cannot guarantee that
improving one part never sacrifices another; that requires empirical rejection
checks or a separate decision policy. Frozen trackers are not modified here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from instance_body_score import sample_body_xyz
from scipy.optimize import linear_sum_assignment
from sparse_instance_body_score import SparseDistanceFields


@dataclass(frozen=True)
class BundleConfig:
    node_weights: tuple[float, float, float] = (2.0, 1.0, 1.0)
    node_tolerance_px: float = 6.0
    node_scale_px: float = 6.0
    mask_weight: float = 1.0
    mask_row_tolerance_px: float = 8.0
    endpoint_tolerance_px: float = 6.0
    interior_tolerance_px: float = 2.0
    mask_scale_px: float = 6.0
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    tie_atol: float = 1e-12

    def __post_init__(self):
        if (not isinstance(self.node_weights, tuple) or len(self.node_weights) != 3
                or any(isinstance(x, bool) or not np.isfinite(x) or x <= 0
                       for x in self.node_weights)):
            raise ValueError("three positive finite fixed node weights required")
        for key in ("node_scale_px", "mask_scale_px"):
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError("positive finite setting required: " + key)
        for key in ("node_tolerance_px", "mask_weight", "mask_row_tolerance_px",
                    "endpoint_tolerance_px", "interior_tolerance_px", "tie_atol"):
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError("finite nonnegative setting required: " + key)
        if (not isinstance(self.fractions, tuple) or not self.fractions
                or any(isinstance(x, bool) or not np.isfinite(x) or not 0 < x < 1
                       for x in self.fractions)
                or any(a >= b for a, b in zip(self.fractions, self.fractions[1:], strict=False))):
            raise ValueError("ordered unique segment fractions in (0,1) required")

    @property
    def maximum_gain(self) -> float:
        return 3.0 * (sum(self.node_weights) + self.mask_weight)


DEFAULT_CONFIG = BundleConfig()


def _readonly(array):
    out = np.array(array, copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class PartialBundleObservation:
    row_indices: np.ndarray
    points: np.ndarray
    present: np.ndarray
    mask_indices: np.ndarray
    compatible_masks: np.ndarray
    fields: SparseDistanceFields | None
    evidence: dict
    config: BundleConfig


def prepare_partial_bundle_observation(masks, sleap_points, config=DEFAULT_CONFIG):
    """Prepare candidate-independent row/mask compatibility, retaining partial rows.

    One finite semantic point is sufficient to retain a row. Compatibility with
    a single nonempty mask requires *all available* points to fit that mask.
    If no such mask exists, the row still supplies its semantic observations.
    Detector scores, candidate assignments and support flags are not inputs.
    """
    if not isinstance(config, BundleConfig):
        raise ValueError("BundleConfig required")
    if sleap_points is None:
        original = np.empty((0, 3, 2), dtype=np.float64)
    else:
        original = np.asarray(sleap_points)
        if (original.ndim != 3 or original.shape[1:] != (3, 2)
                or original.dtype.kind != "f"):
            raise ValueError("floating untracked SLEAP [R,3,2] required")
    all_present = np.isfinite(original).all(axis=-1)
    rows = np.flatnonzero(all_present.any(axis=-1))
    points = np.array(original[rows], dtype=np.float64, copy=True)
    present = all_present[rows].copy()
    fields = None
    indices = np.empty(0, dtype=np.int64)
    total_masks = None
    if masks is not None:
        if (not isinstance(masks, np.ndarray) or masks.ndim != 3
                or masks.dtype != np.dtype(bool) or min(masks.shape[1:]) < 1):
            raise ValueError("boolean masks [K,H,W], with nonempty image dimensions required")
        total_masks = len(masks)
        indices = np.flatnonzero(masks.any(axis=(1, 2)))
        if len(indices):
            fields = SparseDistanceFields.from_masks(masks[indices])
    compatible = np.zeros((len(rows), len(indices)), dtype=bool)
    if fields is not None and len(rows):
        distances = fields.sample(points)
        compatible = np.all(
            ~present[:, :, None] | (distances <= config.mask_row_tolerance_px), axis=1
        )
    evidence = {
        "mask_instance_count": total_masks,
        "nonempty_mask_indices": indices.tolist(),
        "original_SL_rows": len(original),
        "retained_SL_rows": rows.tolist(),
        "finite_node_count_by_row": present.sum(axis=1).tolist(),
        "compatible_mask_indices_by_row": [indices[k].tolist() for k in compatible],
        "fixed_node_weights": list(config.node_weights),
        "candidate_pose_used_for_eligibility": False,
        "candidate_sources_or_observed_used": False,
        "detection_confidence_used": False,
        "mask_count_equals_three_required": False,
        "complete_SL_triplet_required": False,
        "arbitrary_mask_unions_created": False,
        "mask_sharing_between_distinct_rows_allowed": True,
    }
    return PartialBundleObservation(
        _readonly(rows), _readonly(points), _readonly(present), _readonly(indices),
        _readonly(compatible), fields, evidence, config,
    )


def _affinity(distance, tolerance, scale):
    excess = np.maximum(np.asarray(distance) - tolerance, 0.0) / scale
    # Clip before squaring, so arbitrarily distant/outside points do not overflow.
    return 1.0 - np.square(np.clip(excess, 0.0, 1.0))


def _match(gain):
    """Exact 3-body injective real-row match with three independent dummy columns."""
    matrix = np.concatenate([gain, np.zeros((3, 3), dtype=np.float64)], axis=1)
    rr, cc = linear_sum_assignment(-matrix)
    total = float(matrix[rr, cc].sum())
    assignment = np.full(3, -1, dtype=np.int64)
    margin = np.full(3, np.nan, dtype=np.float64)
    for body, col in zip(rr, cc, strict=True):
        if col >= gain.shape[1] or gain[body, col] <= 0:
            continue
        assignment[body] = col
        alternative = -matrix.copy()
        alternative[body, col] = np.inf
        ar, ac = linear_sum_assignment(alternative)
        other_total = float(matrix[ar, ac].sum())
        margin[body] = max(0.0, total - other_total)
    return total, assignment, margin


def score_partial_body_bundles(pose_xyz, project: Callable, observation):
    """Nonnegative fixed-ceiling cost from semantic and single-mask agreement.

    Missing nodes give zero gain; real rows are injective and unused rows are
    unpenalized. Mask agreement is an additional bounded gain, so a mask may
    support an assignment even when the available semantic point is misplaced.
    Only total-zero matches are reported as dummy. Margins describe this local
    matching problem, not temporal identity certainty or calibrated confidence.
    """
    if not isinstance(observation, PartialBundleObservation):
        raise ValueError("PartialBundleObservation required")
    cfg = observation.config
    samples = sample_body_xyz(pose_xyz, cfg.fractions)
    projected = np.asarray(project(samples), dtype=np.float64)
    if projected.shape != (*samples.shape[:-1], 2):
        raise ValueError("projector must preserve batch/body/sample axes")
    valid = np.isfinite(samples).all(axis=(1, 2, 3)) & np.isfinite(projected).all(axis=(1, 2, 3))
    batch, row_count = len(samples), len(observation.row_indices)
    delta = projected[:, :, None, :3, :] - observation.points[None, None, :, :, :]
    residual = np.linalg.norm(delta, axis=-1)
    node_affinity = np.where(
        observation.present[None, None, :, :],
        _affinity(residual, cfg.node_tolerance_px, cfg.node_scale_px), 0.0,
    )
    per_node_gain = node_affinity * np.asarray(cfg.node_weights)
    node_gain = per_node_gain.sum(axis=-1)
    mask_gain = np.zeros((batch, 3, row_count), dtype=np.float64)
    best_masks = np.full((batch, 3, row_count), -1, dtype=np.int64)
    mask_quality = None
    distances = None
    if cfg.mask_weight and observation.fields is not None and row_count:
        distances = observation.fields.sample(projected)
        tolerance = np.array([cfg.endpoint_tolerance_px] * 3
                             + [cfg.interior_tolerance_px] * (samples.shape[2] - 3))
        weights = np.array([*cfg.node_weights] + [1.0] * (samples.shape[2] - 3))
        qualities = _affinity(distances, tolerance[None, None, :, None], cfg.mask_scale_px)
        mask_quality = (qualities * weights[None, None, :, None]).sum(axis=2) / weights.sum()
        for row in range(row_count):
            allowed = np.flatnonzero(observation.compatible_masks[row])
            if not len(allowed):
                continue
            quality = mask_quality[:, :, allowed]
            choice = np.argmax(quality, axis=-1)
            best = np.take_along_axis(quality, choice[:, :, None], axis=-1)[:, :, 0]
            mask_gain[:, :, row] = cfg.mask_weight * best
            best_masks[:, :, row] = np.where(
                best > cfg.tie_atol, observation.mask_indices[allowed[choice]], -1
            )
    gain = node_gain + mask_gain
    gain[~valid] = 0.0
    gain[np.abs(gain) <= cfg.tie_atol] = 0.0
    matched = np.full((batch, 3), -1, dtype=np.int64)
    margins = np.full((batch, 3), np.nan, dtype=np.float64)
    best_gain = np.zeros(batch)
    for item in np.flatnonzero(valid):
        best_gain[item], local_rows, margins[item] = _match(gain[item])
        positive = local_rows >= 0
        matched[item, positive] = observation.row_indices[local_rows[positive]]
    if np.any(best_gain > cfg.maximum_gain + cfg.tie_atol):
        raise RuntimeError("matching gain exceeds its fixed observation-independent ceiling")
    neutral = row_count == 0
    cost = np.zeros(batch) if neutral else np.maximum(cfg.maximum_gain - best_gain, 0.0)
    cost[~valid] = np.inf
    best_gain[~valid] = np.nan
    return {
        "neutral": neutral,
        "best_cost": cost,
        "best_gain": best_gain,
        "maximum_gain": cfg.maximum_gain,
        "assignment_by_body": matched,
        "assignment_margin_by_body": margins,
        "assignment_tied_by_body": np.isfinite(margins) & (margins <= cfg.tie_atol),
        "body_row_gain": gain,
        "node_gain": node_gain,
        "node_gain_by_node": per_node_gain,
        "mask_gain": mask_gain,
        "best_mask_by_body_row": best_masks,
        "body_mask_quality": mask_quality,
        "semantic_residuals_px": residual,
        "projected_samples": projected,
        "distances_by_body_sample_mask": distances,
        "candidate_projection_finite": valid,
        "observation_evidence": observation.evidence,
        "physical_IDs_assigned": False,
    }

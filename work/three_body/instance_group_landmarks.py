"""Experimental semantic node cost within frame-local mask/SLEAP body groups.

The frozen mask gate already identifies one complete SLEAP row per instance.
Use the M/C/T labels of that same row, including latent projected body nodes,
rather than selecting a different nearest row for each node. Recompute all six
whole-body bijections after adding the semantic cost to each body/mask pair.
This is a scoring component, not tracking, independent truth, or permanent ID.

Defaults are initial experimental choices fixed before candidate diagnostics:
12 px is the existing direct-support projection scale; bounded squared error
limits a single SLEAP mistake. No parameters are fitted to the reviewed images.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from instance_body_score import PERMUTATIONS


@dataclass(frozen=True)
class GroupLandmarkConfig:
    pixel_scale_px: float = 12.0
    per_node_cap: float = 9.0
    weight: float = 1.0
    tie_atol: float = 1e-12

    def __post_init__(self):
        for name in ("pixel_scale_px", "per_node_cap", "weight", "tie_atol"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise ValueError("finite numeric setting required: " + name)
            if not np.isfinite(value) or value < 0:
                raise ValueError("finite nonnegative setting required: " + name)
        if self.pixel_scale_px == 0 or self.per_node_cap == 0:
            raise ValueError("positive scale and cap required")


DEFAULT_CONFIG = GroupLandmarkConfig()


def add_group_landmark_cost(mask_result, sleap_points, config=DEFAULT_CONFIG):
    """Return a new combined score, retaining separate mask/semantic diagnostics.

    mask_result must come from the dense or sparse frozen body scorer on this same
    owned frame. This function never changes observation eligibility. The caller
    must bind the same-frame SLEAP source; coordinate values alone cannot prove it.
    """
    if not isinstance(config, GroupLandmarkConfig):
        raise ValueError("GroupLandmarkConfig required")
    result = dict(mask_result)
    result.update(
        group_landmark_policy=dict(
            schema="same_instance_SLEAP_landmarks_v1",
            **asdict(config),
            initial_experimental_choices=True,
            observed_flags_used=False,
            independent_ID_truth=False,
            group_assignments_are_frame_local=True,
            same_frame_source_binding_required=True,
        ),
        mask_only_best_cost=mask_result["best_cost"],
        mask_only_best_assignment=mask_result["best_assignment"],
        mask_only_assignment_costs=mask_result["assignment_costs"],
        mask_only_body_instance_costs=mask_result["body_instance_costs"],
        semantic_distances_by_body_node_instance_px=None,
        semantic_body_instance_costs=None,
        semantic_SL_row_by_instance=None,
    )
    if not mask_result["eligible"]:
        return result
    pair = np.asarray(mask_result["body_instance_costs"])
    projected = np.asarray(mask_result["projected_samples"])
    points = np.asarray(sleap_points)
    if (
        pair.ndim != 3
        or pair.shape[1:] != (3, 3)
        or pair.dtype.kind != "f"
        or projected.ndim != 4
        or projected.shape[:2] != pair.shape[:2]
        or projected.shape[2] < 3
        or projected.shape[3] != 2
        or projected.dtype.kind != "f"
        or points.ndim != 3
        or points.shape[1:] != (3, 2)
        or points.dtype.kind != "f"
    ):
        raise ValueError("body/mask costs, projected M/C/T, and floating SLEAP rows required")
    groups = mask_result["observation_evidence"]["qualifying_SL_rows"]
    if (
        not isinstance(groups, list)
        or len(groups) != 3
        or any(not isinstance(g, list) or len(g) != 1 for g in groups)
        or any(type(g[0]) is not int or not 0 <= g[0] < len(points) for g in groups)
    ):
        raise ValueError("one complete SLEAP row per eligible mask required")
    rows = np.array([g[0] for g in groups], dtype=np.int64)
    if len(np.unique(rows)) != 3 or not np.isfinite(points[rows]).all():
        raise ValueError("distinct finite same-instance SLEAP rows required")
    grouped = points[rows].astype(np.float64).transpose(1, 0, 2)
    delta = projected[:, :, :3, None, :].astype(np.float64) - grouped[None, None]
    distances = np.linalg.norm(delta, axis=-1)
    # Clip distance before squaring to avoid overflow; invalid candidates are
    # still +inf below, not made valid by the robust finite observation cap.
    bounded = np.minimum(distances / config.pixel_scale_px, np.sqrt(config.per_node_cap))
    semantic = np.mean(bounded * bounded, axis=2)
    valid = np.asarray(mask_result["candidate_projection_finite"])
    if valid.dtype != bool or valid.shape != (len(pair),):
        raise ValueError("candidate projection validity required")
    valid = valid & np.isfinite(projected).all(axis=(1, 2, 3))
    semantic[~valid] = np.inf
    result.update(
        semantic_distances_by_body_node_instance_px=distances,
        semantic_body_instance_costs=semantic,
        semantic_SL_row_by_instance=rows,
    )
    if config.weight == 0:
        # Preserve every frozen mask score/assignment byte when explicitly off.
        return result
    combined = pair + config.weight * semantic
    combined[~valid] = np.inf
    joint = np.stack(
        [combined[:, np.arange(3), permutation].sum(axis=1) for permutation in PERMUTATIONS],
        axis=-1,
    )
    selected = np.argmin(joint, axis=1)
    best = joint[np.arange(len(pair)), selected]
    assignment = PERMUTATIONS[selected].copy()
    assignment[~np.isfinite(best)] = -1
    result.update(
        best_cost=best,
        best_assignment=assignment,
        assignment_costs=joint,
        optimal_assignment_mask=(
            np.isfinite(joint) & np.isclose(joint, best[:, None], rtol=0, atol=config.tie_atol)
        ),
        body_instance_costs=combined,
    )
    return result

"""One eligibility ablation: separated instance masks do not require SLEAP groups.

This prepares the frozen sparse scorer's observation type. It changes no distance,
body projection, assignment, tolerance, or cost formula, and is not integrated into
tracking. Exactly three separated masks are not proof of three real fish: a
reflection can pass this gate while two real fish overlap in another mask.

No SLEAP input or instance-to-SLEAP correspondence is produced. In particular this
observation must not be treated as having the groups needed by a semantic-node
scorer. Availability is an image-geometry condition, not physical-ID evidence.
"""

from __future__ import annotations

import numpy as np
from instance_body_score import DEFAULT_CONFIG, InstanceBodyConfig
from sparse_instance_body_score import SparseDistanceFields, SparseInstanceObservation

REFERENCE_SPARSE_SHA256 = "68c41510d64a41098c6c1daf5e98621723f37c2f0f35730a68bd0e896517a634"
ELIGIBILITY_POLICY = "three_separated_masks_without_SLEAP_group_gate_v1"


def prepare_mask_only_instance_observation(
    masks: np.ndarray | None, config: InstanceBodyConfig = DEFAULT_CONFIG
) -> SparseInstanceObservation:
    """Retain all frozen mask gates; omit the independent SLEAP-group gate only.

    Missing, merged, overlapping, or insufficiently separated instances remain
    unavailable, with zero *unassessed* cost under score_sparse_instance_bodies.
    The original config is retained verbatim. No candidate pose participates in
    this gate and no confidence or semantic identity is inferred from a mask.
    """
    if not isinstance(config, InstanceBodyConfig):
        raise ValueError("InstanceBodyConfig required")
    evidence = dict(
        eligibility_policy=ELIGIBILITY_POLICY,
        candidate_pose_used_for_eligibility=False,
        detection_confidence_used=False,
        permanent_IDs_assigned=False,
        SLEAP_groups_used_for_eligibility=False,
        SLEAP_instance_correspondence_available=False,
        mask_instances_are_verified_fish=False,
        occlusion_or_reflection_resolved=False,
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
    return SparseInstanceObservation(
        True,
        "three_separated_mask_instances_without_SLEAP_group_requirement",
        shape,
        fields,
        None,
        evidence,
        config,
    )

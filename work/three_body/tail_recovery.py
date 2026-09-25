"""Pure current-frame tail recovery through a reversible node-axis adapter.

The frozen mouth generators see nodes (tail, chest, mouth) and reversed segment
lengths. Their exclusivity, triangulation, box, anatomy and strict-support gates
are unchanged. Native mouth/chest entries and other fish are preserved. This
helper applies no temporal gate, budget for the tracker, fit or identity rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import bounded_mouth_recovery as bounded
import node_recovery as legacy
import numpy as np
from node_geometry import RayObservations, supported_nodes

from medaka3d.types import VIEWS

REVERSE = (2, 1, 0)
VERSION = "tail_recovery_reversed_nodes_v1"


@dataclass(frozen=True)
class TailRecoveryProposal:
    pose_xyz: np.ndarray
    plan: np.ndarray
    fish: int
    node: int
    displacement_mm: float
    tail_pectoral_mm: float
    length_ratio: float
    ray_rms_mm: float
    supported_views: int
    seed_method: str


@dataclass(frozen=True)
class TailRecoveryResult:
    proposals: list[TailRecoveryProposal]
    diagnostics: dict


def reversed_inputs(scene_xyz, base_plan, rays, initial_lengths):
    pose, plan, lengths = map(np.asarray, (scene_xyz, base_plan, initial_lengths))
    if pose.shape != (3, 3, 3) or plan.shape != (3, 3, 3) or lengths.shape != (3, 2):
        raise ValueError("native three-fish pose/plan/length shapes required")
    if set(rays.points) != set(VIEWS) or len(rays.origins) != 3 or len(rays.directions) != 3:
        raise ValueError("explicit three-view rays required")
    for vi, view in enumerate(VIEWS):
        count = len(rays.points[view])
        if (
            rays.points[view].shape != (count, 3, 2)
            or rays.origins[vi].shape != (count, 3, 3)
            or rays.directions[vi].shape != (count, 3, 3)
        ):
            raise ValueError("native three-node ray shapes required")
    transformed = RayObservations(
        {v: rays.points[v][:, REVERSE].copy() for v in VIEWS},
        tuple(x[:, REVERSE].copy() for x in rays.origins),
        tuple(x[:, REVERSE].copy() for x in rays.directions),
    )
    return pose[:, REVERSE].copy(), plan[:, :, REVERSE].copy(), transformed, lengths[:, ::-1].copy()


def tail_recovery_proposals(
    scene_xyz, base_plan, rays, rig, initial_lengths, config=None, *, allow_bounded=True
):
    if type(allow_bounded) is not bool:
        raise ValueError("allow_bounded must be boolean")
    cfg = config or legacy.NodeRecoveryConfig()
    pose, plan, transformed, lengths = reversed_inputs(scene_xyz, base_plan, rays, initial_lengths)
    original = legacy.node_recovery_proposals(pose, plan, transformed, rig, lengths, cfg)
    extra = (
        bounded.bounded_mouth_recovery_proposals(pose, plan, transformed, rig, lengths, cfg)
        if allow_bounded
        else None
    )
    groups = [("free_intersection", original)]
    if extra is not None:
        groups.append(("bounded_intersection", extra))
    combined = []
    for method, result in groups:
        for candidate in result.proposals:
            restored_pose = candidate.pose_xyz[:, REVERSE].copy()
            restored_plan = candidate.plan[:, :, REVERSE].copy()
            sources, observed, _ = supported_nodes(restored_pose, restored_plan, rays, rig)
            fish = candidate.fish
            if (
                not observed[fish, 1:].all()
                or int((sources[fish, :, 2] >= 0).sum()) != candidate.supported_views
            ):
                raise AssertionError("reversed and native strict tail support differ")
            combined.append(
                TailRecoveryProposal(
                    restored_pose,
                    restored_plan,
                    fish,
                    2,
                    candidate.displacement_mm,
                    candidate.mouth_pectoral_mm,
                    candidate.length_ratio,
                    candidate.ray_rms_mm,
                    candidate.supported_views,
                    method,
                )
            )
    selected = []
    counts = []
    for fish in range(3):
        candidates = [p for p in combined if p.fish == fish]
        # Each delegate returns its own top K. Their combined top K is exactly
        # the top K of the union under this common key; free wins stable ties.
        candidates.sort(
            key=lambda p: (
                -p.supported_views,
                p.ray_rms_mm,
                p.displacement_mm,
                tuple(p.plan[fish, :, 2]),
            )
        )
        kept = candidates[: cfg.proposals_per_fish]
        selected.extend(kept)
        counts.append(dict(fish=fish, merged_candidates=len(candidates), returned=len(kept)))
    return TailRecoveryResult(
        selected,
        dict(
            schema=VERSION,
            target_node="tail",
            native_target_index=2,
            delegate_node_order=list(REVERSE),
            delegate_field_semantics="mouth in nested delegate diagnostics denotes native tail",
            original_delegate=original.diagnostics,
            bounded_delegate=None if extra is None else extra.diagnostics,
            fish=counts,
            allow_bounded=allow_bounded,
            age_gate_applied=False,
            tracker_budget_applied=False,
            no_angle_constraint_added=True,
            candidate_is_ID_confirmation=False,
        ),
    )

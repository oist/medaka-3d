"""Current-frame mouth alternatives around a directly supported pectoral node.

This pure CPU helper does not track, initialize, change identities or clear review.
Its proposals compete with the unchanged parent hypothesis in a future caller.
Healthy pectoral support and absent mouth support are measured at the supplied
current scene/plan; no temporal node counters are inferred from body-missing age.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from node_geometry import supported_nodes

from medaka3d.calib import triangulate
from medaka3d.types import VIEWS


@dataclass(frozen=True)
class NodeRecoveryConfig:
    ray_rms_mm: float = 1.5
    minimum_length_ratio: float = 0.5
    maximum_length_ratio: float = 1.5
    proposals_per_fish: int = 2
    maximum_combinations_per_fish: int = 4096
    lower_mm: tuple[float, float, float] = (0.0, 0.0, -6.0)
    upper_mm: tuple[float, float, float] = (192.0, 192.0, 186.0)

    def __post_init__(self):
        for name in ("ray_rms_mm", "minimum_length_ratio", "maximum_length_ratio"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_length_ratio > self.maximum_length_ratio:
            raise ValueError("length ratio bounds are reversed")
        for name in ("proposals_per_fish", "maximum_combinations_per_fish"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        lower, upper = np.asarray(self.lower_mm), np.asarray(self.upper_mm)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(lower >= upper)
        ):
            raise ValueError("engineering bounds require three finite ordered axes")


@dataclass(frozen=True)
class NodeRecoveryProposal:
    pose_xyz: np.ndarray
    plan: np.ndarray
    fish: int
    node: int
    displacement_mm: float
    mouth_pectoral_mm: float
    length_ratio: float
    ray_rms_mm: float
    supported_views: int


@dataclass(frozen=True)
class NodeRecoveryResult:
    proposals: list[NodeRecoveryProposal]
    diagnostics: dict


def _injective(plan):
    for view in range(3):
        for node in range(3):
            used = plan[:, view, node]
            used = used[used >= 0]
            if len(used) != len(np.unique(used)):
                return False
    return True


def node_recovery_proposals(scene_xyz, base_plan, rays, rig, initial_lengths, config=None):
    """Return mouth-only alternatives, preserving every other pose/plan entry.

    Inputs are scene (3 fish,3 nodes,xyz), injective plan (fish,view,node), current
    RayObservations, and initial segment lengths (fish,2). An unobserved center is
    never treated as an anatomical anchor. Even an unsupported other fish's
    base-plan mouth row is protected. Missing views and outside-box intersections
    are rejected without padding, extrapolation, clipping or widening the box.

    The existing supported_nodes check supplies the final 2-mm/12-pixel gate.
    The combination limit skips the entire affected fish with an explicit reason;
    it never takes a silent prefix of candidate detection rows. Ranking favors
    more directly supported views, then lower ray RMS, smaller displacement and
    lexicographic source rows. Eligibility/ranking is not identity confidence.
    """
    cfg = config or NodeRecoveryConfig()
    if tuple(rig.views) != tuple(VIEWS):
        raise ValueError("rig views must retain the explicit observation order")
    pose = np.asarray(scene_xyz, float)
    plan = np.asarray(base_plan)
    lengths = np.asarray(initial_lengths, float)
    lower, upper = np.asarray(cfg.lower_mm), np.asarray(cfg.upper_mm)
    if pose.shape != (3, 3, 3) or not np.isfinite(pose).all():
        raise ValueError("scene must contain three finite three-node poses")
    if np.any(pose < lower) or np.any(pose > upper):
        raise ValueError("current scene lies outside the configured engineering box")
    if plan.shape != (3, 3, 3) or plan.dtype.kind not in "iu" or np.any(plan < -1):
        raise ValueError("base plan must contain integer source rows or -1")
    if not _injective(plan):
        raise ValueError("base plan is not injective for each view/node")
    if lengths.shape != (3, 2) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("initial lengths must be finite positive (fish,2) segments")
    for view, name in enumerate(VIEWS):
        if np.any(plan[:, view] >= len(rays.points[name])):
            raise ValueError("base plan refers to a row outside current-frame detections")
    plan = plan.astype(np.int32, copy=False)
    _, observed, _ = supported_nodes(pose, plan, rays, rig)
    all_proposals, fish_diagnostics = [], []
    for fish in range(3):
        info = dict(
            fish=fish,
            pectoral_direct=bool(observed[fish, 1]),
            mouth_direct=bool(observed[fish, 0]),
            status="not_needed",
            combinations=0,
            admitted=0,
            returned=0,
        )
        fish_diagnostics.append(info)
        if observed[fish, 0]:
            continue
        if not observed[fish, 1]:
            info["status"] = "pectoral_not_directly_supported"
            continue
        free = []
        for view, name in enumerate(VIEWS):
            finite = np.isfinite(rays.points[name][:, 0]).all(-1)
            finite &= np.isfinite(rays.origins[view][:, 0]).all(-1)
            finite &= np.isfinite(rays.directions[view][:, 0]).all(-1)
            protected = plan[np.arange(3) != fish, view, 0]
            rows = np.flatnonzero(finite & ~np.isin(np.arange(len(finite)), protected))
            free.append(rows.tolist())
        info["free_mouth_rows_per_view"] = [len(rows) for rows in free]
        view_sets = [views for count in (3, 2) for views in combinations(range(3), count)]
        info["combinations"] = sum(
            int(np.prod([len(free[view]) for view in views])) for views in view_sets
        )
        if not info["combinations"]:
            info["status"] = "fewer_than_two_unclaimed_views"
            continue
        if info["combinations"] > cfg.maximum_combinations_per_fish:
            info["status"] = "combination_limit_exceeded"
            continue
        proposals = []
        rejection = dict(
            nonfinite_or_behind=0, ray_rms=0, outside_box=0, anatomy=0, strict_support=0
        )
        for views in view_sets:
            choices = list(product(*(free[v] for v in views)))
            if not choices:
                continue
            origins = np.array(
                [
                    [rays.origins[v][row, 0] for v, row in zip(views, rows, strict=True)]
                    for rows in choices
                ]
            )
            directions = np.array(
                [
                    [rays.directions[v][row, 0] for v, row in zip(views, rows, strict=True)]
                    for rows in choices
                ]
            )
            points, errors = triangulate(origins, directions)
            for index, (xyz, error) in enumerate(zip(points, errors, strict=True)):
                if (
                    not np.isfinite(xyz).all()
                    or not np.isfinite(error)
                    or np.any(((xyz - origins[index]) * directions[index]).sum(-1) < 0)
                ):
                    rejection["nonfinite_or_behind"] += 1
                    continue
                if error > cfg.ray_rms_mm:
                    rejection["ray_rms"] += 1
                    continue
                if np.any(xyz < lower) or np.any(xyz > upper):
                    rejection["outside_box"] += 1
                    continue
                length = float(np.linalg.norm(xyz - pose[fish, 1]))
                ratio = length / lengths[fish, 0]
                if not cfg.minimum_length_ratio <= ratio <= cfg.maximum_length_ratio:
                    rejection["anatomy"] += 1
                    continue
                candidate_pose, candidate_plan = pose.copy(), plan.copy()
                candidate_pose[fish, 0] = xyz
                candidate_plan[fish, :, 0] = -1
                candidate_plan[fish, list(views), 0] = choices[index]
                if not _injective(candidate_plan):
                    raise AssertionError("candidate stole a protected current-frame node row")
                sources, support, _ = supported_nodes(candidate_pose, candidate_plan, rays, rig)
                if not support[fish, 0] or not support[fish, 1]:
                    rejection["strict_support"] += 1
                    continue
                proposals.append(
                    NodeRecoveryProposal(
                        pose_xyz=candidate_pose,
                        plan=candidate_plan,
                        fish=fish,
                        node=0,
                        displacement_mm=float(np.linalg.norm(xyz - pose[fish, 0])),
                        mouth_pectoral_mm=length,
                        length_ratio=float(ratio),
                        ray_rms_mm=float(error),
                        supported_views=int((sources[fish, :, 0] >= 0).sum()),
                    )
                )
        proposals.sort(
            key=lambda item: (
                -item.supported_views,
                item.ray_rms_mm,
                item.displacement_mm,
                tuple(item.plan[fish, :, 0]),
            )
        )
        info.update(
            status="proposals_available" if proposals else "no_eligible_intersection",
            rejection_counts=rejection,
            admitted=len(proposals),
            returned=min(len(proposals), cfg.proposals_per_fish),
        )
        all_proposals.extend(proposals[: cfg.proposals_per_fish])
    return NodeRecoveryResult(
        all_proposals,
        dict(
            scope="current-frame competing mouth geometry only; no identity certification",
            target_node="mouth",
            fish=fish_diagnostics,
            base_direct_support=observed.tolist(),
        ),
    )

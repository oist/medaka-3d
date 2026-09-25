"""Whole-row multiview body proposals, complementary to independent node pools.

Each selected camera contributes a single SLEAP row to a body. Two-camera
subsets allow a bad third camera; two triangulated nodes suffice, with an
explicitly latent third node filled from the parent's predicted shape. No
segmentation, permanent identity, support flags, or temporal state is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from medaka3d.calib import triangulate
from medaka3d.types import VIEWS


@dataclass(frozen=True)
class BodyRowConfig:
    maximum_ray_rms_mm: float = 1.5
    maximum_pixel_error: float = 12.0
    minimum_triangulated_nodes: int = 2
    maximum_row_combinations: int = 10000
    per_slot_limit: int = 12
    scene_limit: int = 8
    minimum_length_ratio: float = 0.5
    maximum_length_ratio: float = 1.5
    motion_sigma_mm: float = 0.8
    ray_sigma_mm: float = 0.7
    shape_weight: float = 0.3
    motion_weight: float = 0.2

    def __post_init__(self):
        for key in (
            "maximum_ray_rms_mm",
            "maximum_pixel_error",
            "motion_sigma_mm",
            "ray_sigma_mm",
            "minimum_length_ratio",
            "maximum_length_ratio",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError("positive finite setting required: " + key)
        for key in ("maximum_row_combinations", "per_slot_limit", "scene_limit"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("positive integer setting required: " + key)
        if self.minimum_triangulated_nodes not in (2, 3):
            raise ValueError("two or three triangulated nodes required")
        if self.minimum_length_ratio >= self.maximum_length_ratio:
            raise ValueError("ordered length ratio bounds required")
        for key in ("shape_weight", "motion_weight"):
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError("nonnegative finite weight required: " + key)


DEFAULT_CONFIG = BodyRowConfig()


def _bounds(lower, upper):
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or not np.all(lower < upper)
    ):
        raise ValueError("finite ordered XYZ engineering bounds required")
    return lower, upper


def body_row_pool(rig, rays, lower, upper, config=DEFAULT_CONFIG):
    """Triangulate row bundles without allowing nodes to change rows per view.

    Every node with two or more finite rays must pass RMS, forward-ray, bounds,
    and all assigned-view reprojection checks. One-view nodes remain NaN and
    retain their proposed source row for later fitting. No point is silently
    dropped from a chosen row to improve a candidate's score.
    """
    if not isinstance(config, BodyRowConfig) or tuple(rig.views) != tuple(VIEWS):
        raise ValueError("BodyRowConfig and standard three-view rig required")
    lower, upper = _bounds(lower, upper)
    choices, usable = [], []
    for vi, view in enumerate(VIEWS):
        points = np.asarray(rays.points[view])
        origin, direction = np.asarray(rays.origins[vi]), np.asarray(rays.directions[vi])
        if (
            points.ndim != 3
            or points.shape[1:] != (3, 2)
            or origin.shape != (*points.shape[:-1], 3)
            or direction.shape != origin.shape
        ):
            raise ValueError("matching SLEAP row/node points and rays required")
        finite = (
            np.isfinite(points).all(-1)
            & np.isfinite(origin).all(-1)
            & np.isfinite(direction).all(-1)
        )
        norm = np.linalg.norm(direction, axis=-1)
        if np.any(finite & ~np.isclose(norm, 1.0, rtol=1e-5, atol=1e-7)):
            raise ValueError("finite observation rays must have unit directions")
        usable.append(finite)
        choices.append(np.flatnonzero(finite.any(-1)).tolist())
    tuples = []
    for count in (3, 2):
        for views in combinations(range(3), count):
            size = int(np.prod([len(choices[v]) for v in views]))
            if len(tuples) + size > config.maximum_row_combinations:
                raise ValueError("row combination bound exceeded; no silent truncation")
            for rows in product(*(choices[v] for v in views)):
                row = np.full(3, -1, np.int32)
                row[list(views)] = rows
                tuples.append(row)
    if not tuples:
        return [], dict(row_combinations=0, geometry_candidates=0, geometry_only=True)
    view_rows = np.stack(tuples)
    n = len(tuples)
    origins, directions = np.zeros((n, 3, 3, 3)), np.zeros((n, 3, 3, 3))
    held = np.zeros((n, 3, 3), bool)  # candidate,node,view
    source = np.full((n, 3, 3), -1, np.int32)  # candidate,view,node
    for vi in range(3):
        chosen = np.flatnonzero(view_rows[:, vi] >= 0)
        rows = view_rows[chosen, vi]
        origins[chosen, :, vi] = rays.origins[vi][rows]
        directions[chosen, :, vi] = rays.directions[vi][rows]
        held[chosen, :, vi] = usable[vi][rows]
        source[chosen, vi] = np.where(usable[vi][rows], rows[:, None], -1)
    xyz, rms = triangulate(origins, directions, held)
    measured = held.sum(-1) >= 2
    finite = np.isfinite(xyz).all(-1) & np.isfinite(rms)
    ray_forward = np.sum((xyz[:, :, None] - origins) * directions, axis=-1)
    good_node = (
        finite
        & (rms <= config.maximum_ray_rms_mm)
        & (xyz >= lower).all(-1)
        & (xyz <= upper).all(-1)
        & np.all(~held | (ray_forward >= 0), axis=-1)
    )
    projected = {}
    pixel_error = np.full((n, 3, 3), np.nan)
    for vi, view in enumerate(VIEWS):
        projected[view] = np.asarray(rig.camera(view).project(xyz), float)
        if projected[view].shape != (n, 3, 2):
            raise ValueError("projector must preserve candidate/node axes")
        for ci in np.flatnonzero(view_rows[:, vi] >= 0):
            row = view_rows[ci, vi]
            pixel_error[ci, vi] = np.linalg.norm(
                projected[view][ci] - rays.points[view][row], axis=-1
            )
    good_node &= np.all(
        ~held | (np.swapaxes(pixel_error, 1, 2) <= config.maximum_pixel_error), axis=-1
    )
    accepted = (measured.sum(-1) >= config.minimum_triangulated_nodes) & np.all(
        ~measured | good_node, axis=-1
    )
    pool = []
    for ci in np.flatnonzero(accepted):
        pool.append(
            dict(
                pool_index=len(pool),
                combination_index=int(ci),
                view_rows=view_rows[ci].copy(),
                pose_xyz=xyz[ci].copy(),
                triangulated_nodes=measured[ci].copy(),
                proposed_sources=source[ci].copy(),
                ray_rms_mm=rms[ci].copy(),
                reprojection_error_px=pixel_error[ci].copy(),
                used_ray_count=int(held[ci].sum()),
                same_row_within_each_view=True,
                direct_support_not_recomputed=True,
            )
        )
    return pool, dict(
        row_combinations=n,
        geometry_candidates=len(pool),
        geometry_only=True,
        minimum_triangulated_nodes=config.minimum_triangulated_nodes,
        row_combinations_silently_pruned=False,
        retained_SLEAP_rows_by_view={v: choices[i] for i, v in enumerate(VIEWS)},
        node_sources_are_proposals_not_IDs=True,
    )


def _minimum_rotation(source, target):
    """Minimum-angle proper rotation between nonzero vectors, deterministic at pi."""
    a, b = np.asarray(source, float), np.asarray(target, float)
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    if min(an, bn) < 1e-8:
        raise ValueError("degenerate two-node template or observation")
    a, b = a / an, b / bn
    cosine = np.clip(a @ b, -1.0, 1.0)
    cross = np.cross(a, b)
    sine = np.linalg.norm(cross)
    if sine < 1e-10:
        if cosine >= 0:
            return np.eye(3)
        basis = np.eye(3)[np.argmin(np.abs(a))]
        axis = np.cross(a, basis)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    x, y, z = cross
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def complete_from_prediction(body, predicted):
    """Fill exactly one latent node by rotating/translating the parent template."""
    pose = np.asarray(body["pose_xyz"], float).copy()
    measured = np.asarray(body["triangulated_nodes"], bool)
    predicted = np.asarray(predicted, float)
    if (
        pose.shape != (3, 3)
        or predicted.shape != (3, 3)
        or measured.shape != (3,)
        or measured.sum() < 2
        or not np.isfinite(predicted).all()
        or not np.isfinite(pose[measured]).all()
    ):
        raise ValueError("finite prediction and at least two measured nodes required")
    if measured.all():
        return pose
    first, second = np.flatnonzero(measured)
    rotation = _minimum_rotation(predicted[second] - predicted[first], pose[second] - pose[first])
    filled = (predicted - predicted[measured].mean(0)) @ rotation.T + pose[measured].mean(0)
    pose[~measured] = filled[~measured]
    return pose


def joint_body_row_scenes(pool, predicted, lengths, lower, upper, config=DEFAULT_CONFIG):
    """Bounded row-disjoint three-body scenes around one existing parent prediction.

    Ranking here is a proposal heuristic. Full image fitting and temporal beam
    selection remain separate. Per-slot pruning is reported, never called an
    exact global search. Retaining the original proposals is the caller's job.
    """
    lower, upper = _bounds(lower, upper)
    predicted, lengths = np.asarray(predicted, float), np.asarray(lengths, float)
    if (
        predicted.shape != (3, 3, 3)
        or lengths.shape != (3, 2)
        or not np.isfinite(predicted).all()
        or not np.isfinite(lengths).all()
        or np.any(lengths <= 0)
    ):
        raise ValueError("finite three-body prediction and positive fixed lengths required")
    options, counts = [], []
    for slot in range(3):
        choices = []
        for index, body in enumerate(pool):
            try:
                pose = complete_from_prediction(body, predicted[slot])
            except ValueError:
                continue
            if np.any(pose < lower) or np.any(pose > upper):
                continue
            segments = np.linalg.norm(np.diff(pose, axis=0), axis=-1)
            ratio = segments / lengths[slot]
            if np.any(ratio < config.minimum_length_ratio) or np.any(
                ratio > config.maximum_length_ratio
            ):
                continue
            measured = body["triangulated_nodes"]
            ray = np.mean((body["ray_rms_mm"][measured] / config.ray_sigma_mm) ** 2)
            shape = np.mean(((segments - lengths[slot]) / (0.2 * lengths[slot])) ** 2)
            motion = np.sum((pose[1] - predicted[slot, 1]) ** 2) / config.motion_sigma_mm**2
            relative_delta = (pose - pose[1]) - (predicted[slot] - predicted[slot, 1])
            orientation = np.mean(np.sum(relative_delta**2, axis=-1)) / 25.0
            score = float(
                ray
                + config.shape_weight * shape
                + config.motion_weight * (motion + 0.05 * orientation)
            )
            choices.append((score, index, pose))
        choices.sort(key=lambda item: (item[0], item[1]))
        counts.append(len(choices))
        options.append(choices[: config.per_slot_limit])
    scenes, conflicts = [], 0
    for selected in product(*options):
        rows = np.stack([pool[item[1]]["view_rows"] for item in selected])
        if any(len(r[r >= 0]) != len(set(r[r >= 0].tolist())) for r in rows.T):
            conflicts += 1
            continue
        scenes.append(
            dict(
                proposal_score=float(sum(item[0] for item in selected)),
                pool_indices=np.array([item[1] for item in selected], np.int32),
                pose_xyz=np.stack([item[2] for item in selected]),
                proposed_sources=np.stack([pool[item[1]]["proposed_sources"] for item in selected]),
                triangulated_nodes=np.stack(
                    [pool[item[1]]["triangulated_nodes"] for item in selected]
                ),
                source_rows=rows,
                source_rows_disjoint_within_each_view=True,
                physical_IDs_assigned=False,
                parent_prediction_used=True,
            )
        )
    scenes.sort(key=lambda s: (s["proposal_score"], tuple(s["pool_indices"])))
    return scenes[: config.scene_limit], dict(
        pool_size=len(pool),
        admissible_body_choices_by_slot=counts,
        retained_body_choices_by_slot=[len(c) for c in options],
        per_slot_pruned=[max(c - config.per_slot_limit, 0) for c in counts],
        conflicting_scene_combinations=conflicts,
        admissible_scenes=len(scenes),
        scenes_returned=min(len(scenes), config.scene_limit),
        exact_global_search=False,
        existing_proposals_must_be_retained=True,
    )

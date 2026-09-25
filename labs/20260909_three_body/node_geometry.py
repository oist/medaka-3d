"""Untracked SLEAP ray proposals; no segmentation or identity labels enter here."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from medaka3d.calib import triangulate
from medaka3d.types import VIEWS


@dataclass(frozen=True)
class RayObservations:
    points: dict[str, np.ndarray]
    origins: tuple[np.ndarray, ...]  # each (detections, nodes, xyz)
    directions: tuple[np.ndarray, ...]

    @classmethod
    def from_points(cls, rig, points):
        if set(points) != set(VIEWS):
            raise ValueError("all three views must be explicit, including empty detections")
        for view in VIEWS:
            value = np.asarray(points[view])
            if value.ndim != 3 or value.shape[1:] != (3, 2):
                raise ValueError(f"invalid SLEAP node array for {view}")
        rays = [rig.camera(v).rays(points[v]) for v in VIEWS]
        return cls(points, tuple(r[0] for r in rays), tuple(r[1] for r in rays))


def exclusive(cost, missing_cost, keep=2):
    """Best injective assignments over every finite candidate cheaper than missing."""
    cost = np.asarray(cost, float)
    choices = []
    for row in cost:
        valid = np.flatnonzero(np.isfinite(row) & (row < missing_cost))
        indexes = valid[np.argsort(row[valid], kind="stable")]
        choices.append([*indexes.tolist(), -1])
    results = []
    for assignment in product(*choices):
        used = [j for j in assignment if j >= 0]
        if len(used) != len(set(used)):
            continue
        value = sum(missing_cost if j < 0 else cost[i, j] for i, j in enumerate(assignment))
        results.append((float(value), np.array(assignment, np.int32)))
    results.sort(key=lambda x: (x[0], tuple(x[1])))
    return results[:keep]


def assignment_plans(pose, rays, ray_sigma=0.7, missing_cost=9.0, keep=2):
    base = np.full((3, 3, 3), -1, np.int32)  # fish, view, node
    alternatives, total = [], 0.0
    for view in range(3):
        for node in range(3):
            delta = pose[:, node, None] - rays.origins[view][None, :, node]
            direction = rays.directions[view][None, :, node]
            perpendicular = delta - np.sum(delta * direction, axis=-1, keepdims=True) * direction
            cost = np.sum(perpendicular**2, axis=-1) / ray_sigma**2
            choices = exclusive(cost, missing_cost, keep=2)
            base[:, view, node] = choices[0][1]
            total += choices[0][0]
            if len(choices) > 1:
                alternatives.append((choices[1][0] - choices[0][0], view, node, choices[1][1]))
    out = [(total, base)]
    for gap, view, node, alternative in sorted(alternatives, key=lambda x: x[0])[: keep - 1]:
        plan = base.copy()
        plan[:, view, node] = alternative
        out.append((total + gap, plan))
    return out


def assigned_rays(rays, plans):
    plans = np.asarray(plans, np.int32)
    if plans.ndim != 4 or plans.shape[1:] != (3, 3, 3):
        raise ValueError("plans must have shape (batch,fish,view,node)")
    origin = np.zeros((*plans.shape, 3), dtype=np.float32)
    direction = np.zeros_like(origin)
    valid = plans >= 0
    for view in range(3):
        for node in range(3):
            indices = plans[:, :, view, node]
            held = valid[:, :, view, node]
            if held.any():
                origin[:, :, view, node][held] = rays.origins[view][indices[held], node]
                direction[:, :, view, node][held] = rays.directions[view][indices[held], node]
    return origin, direction, valid


def supported_nodes(pose, plan, rays, rig, maximum_ray_mm=2.0, maximum_pixel_error=12.0):
    origin, direction, held = assigned_rays(rays, np.asarray(plan)[None])
    delta = pose[None, :, None] - origin
    distance = np.linalg.norm(
        delta - np.sum(delta * direction, axis=-1, keepdims=True) * direction, axis=-1
    )[0]
    valid = held[0] & (distance <= maximum_ray_mm)
    for view in range(3):
        projected = rig.camera(VIEWS[view]).project(pose)
        for fish, node in np.argwhere(valid[:, view]):
            row = plan[fish, view, node]
            error = np.linalg.norm(projected[fish, node] - rays.points[VIEWS[view]][row, node])
            valid[fish, view, node] &= np.isfinite(error) and error <= maximum_pixel_error
    sources = np.where(valid, plan, -1).astype(np.int32)
    return sources, valid.sum(axis=1) >= 2, distance


def node_pools(rig, rays, lower, upper, limit=24):
    """Current-frame two/three-view 3D hypotheses, allowing a bad third view."""
    pools = []
    for node in range(3):
        valid_rows = [
            np.flatnonzero(np.isfinite(rays.points[v][:, node]).all(axis=1)) for v in VIEWS
        ]
        plans = []
        for count in (3, 2):
            for views in combinations(range(3), count):
                for rows in product(*(valid_rows[v] for v in views)):
                    source = np.full(3, -1, np.int32)
                    source[list(views)] = rows
                    plans.append(source)
        if not plans:
            pools.append([])
            continue
        sources = np.stack(plans)
        origin = np.zeros((len(plans), 3, 3))
        direction = np.zeros_like(origin)
        held = sources >= 0
        for view in range(3):
            mask = held[:, view]
            origin[mask, view] = rays.origins[view][sources[mask, view], node]
            direction[mask, view] = rays.directions[view][sources[mask, view], node]
        xyz, residual = triangulate(origin, direction, held)
        good = (
            np.isfinite(xyz).all(axis=1)
            & (residual <= 1.5)
            & (xyz >= lower).all(axis=1)
            & (xyz <= upper).all(axis=1)
        )
        order = sorted(np.flatnonzero(good), key=lambda i: (-held[i].sum(), residual[i]))
        chosen = []
        for index in order:
            if any(
                np.linalg.norm(xyz[index] - item[0]) < 0.75
                and np.all((sources[index] < 0) | (sources[index] == item[1]))
                for item in chosen
            ):
                continue
            chosen.append((xyz[index], sources[index], float(residual[index])))
            if len(chosen) == limit:
                break
        pools.append(chosen)
    return pools


def recovery_poses(predicted, missing, lengths, pools, base_plan, lower, upper, keep=2):
    """Body-shaped alternatives around unused current observations, never past row IDs."""
    candidates = []
    for fish in np.flatnonzero(missing > 0):
        others = [f for f in range(3) if f != fish and missing[f] == 0]

        def free(source, node, others=others):
            return not any(
                np.any((source >= 0) & (source == base_plan[f, :, node])) for f in others
            )

        centers = [c for c in pools[1] if free(c[1], 1)]
        centers.sort(key=lambda c: np.linalg.norm(c[0] - predicted[fish, 1]))
        for center, _, residual in centers[:4]:
            choices = []
            for node, length in ((0, lengths[fish, 0]), (2, lengths[fish, 1])):
                options = []
                old = predicted[fish, node] - predicted[fish, 1]
                for point, source, error in pools[node]:
                    distance = np.linalg.norm(point - center)
                    if free(source, node) and 0.5 * length < distance < 1.5 * length:
                        cost = (distance - length) ** 2 + 0.03 * np.linalg.norm(
                            point - center - old
                        )
                        options.append((cost + error, point))
                options.sort(key=lambda x: x[0])
                choices.append(options[:2] or [(4.0, center + old)])
            for head, tail in product(*choices):
                pose = predicted.copy()
                pose[fish] = np.stack((head[1], center, tail[1]))
                if np.any(pose < lower) or np.any(pose > upper):
                    continue
                cost = (np.linalg.norm(center - predicted[fish, 1]) / (2 + missing[fish])) ** 2
                candidates.append((cost + residual + head[0] + tail[0], pose, int(fish)))
    candidates.sort(key=lambda x: x[0])
    return candidates[:keep]

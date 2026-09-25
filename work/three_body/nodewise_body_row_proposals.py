"""Add complete bodies whose nodes use different consistent camera subsets.

The original whole-row pool is preserved. Supplemental bodies keep a single
SLEAP row per camera but may exclude a different bad camera at each node. These
are geometric proposals, not confirmed observations or persistent identities.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from body_row_proposals import DEFAULT_CONFIG, _bounds, body_row_pool

from medaka3d.calib import triangulate
from medaka3d.types import VIEWS


def nodewise_body_row_pool(rig, rays, lower, upper, config=DEFAULT_CONFIG):
    """Preserve old proposals and add fully triangulated, row-consistent bodies.

    For each node, use all available views if they satisfy the original ray,
    forward, bounds and assigned-pixel gates. Only when three available views
    fail, keep every passing two-view alternative. A failed node is never made
    latent to admit a supplemental body. Excluded-view pixel errors are stored
    alongside the chosen source plan; they are not evidence of agreement.
    """
    pool, legacy_stats = body_row_pool(rig, rays, lower, upper, config)
    legacy_count = len(pool)
    lower, upper = _bounds(lower, upper)
    stats = dict(
        **legacy_stats,
        legacy_candidate_count=legacy_count,
        additional_candidate_count=0,
        nodewise_row_combinations=0,
        row_combinations_with_complete_node_options=0,
        duplicate_source_plans_removed=0,
        all_available_view_nodes_accepted=0,
        nodes_with_two_view_fallback=0,
        fallback_node_options=0,
        maximum_variants_per_row_tuple=27,
        supplemental_bodies_require_three_triangulated_nodes=True,
        physical_IDs_assigned=False,
        excluded_views_are_not_verified=True,
    )
    choices, usable = [], []
    for vi, view in enumerate(VIEWS):
        points = np.asarray(rays.points[view])
        valid = (
            np.isfinite(points).all(-1)
            & np.isfinite(rays.origins[vi]).all(-1)
            & np.isfinite(rays.directions[vi]).all(-1)
        )
        usable.append(valid)
        choices.append(np.flatnonzero(valid.any(-1)).tolist())
    rows = np.asarray(list(product(*choices)), dtype=np.int32).reshape(-1, 3)
    count = len(rows)
    stats["nodewise_row_combinations"] = count
    if count == 0:
        return pool, stats

    # Batch all row tuples, nodes and four candidate view sets. Option zero
    # means all finite views; the other three options are the camera pairs.
    masks = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], bool)
    origins = np.stack([np.asarray(rays.origins[v])[rows[:, v]] for v in range(3)], axis=2)
    directions = np.stack([np.asarray(rays.directions[v])[rows[:, v]] for v in range(3)], axis=2)
    available = np.stack([usable[v][rows[:, v]] for v in range(3)], axis=2)
    held = available[:, :, None, :] & masks[None, None, :, :]
    expanded_shape = (count, 3, 4, 3, 3)
    expanded_origins = np.broadcast_to(origins[:, :, None], expanded_shape)
    expanded_directions = np.broadcast_to(directions[:, :, None], expanded_shape)
    xyz, rms = triangulate(expanded_origins, expanded_directions, held)
    forward = np.sum((xyz[..., None, :] - expanded_origins) * expanded_directions, axis=-1)
    good = (
        (held.sum(-1) >= 2)
        & np.isfinite(xyz).all(-1)
        & np.isfinite(rms)
        & (rms <= config.maximum_ray_rms_mm)
        & (xyz >= lower).all(-1)
        & (xyz <= upper).all(-1)
        & np.all(~held | (forward >= 0), axis=-1)
    )
    errors = []
    for vi, view in enumerate(VIEWS):
        uv = np.asarray(rig.camera(view).project(xyz), float)
        if uv.shape != (count, 3, 4, 2):
            raise ValueError("projector must preserve row/node/view-option axes")
        observed = np.asarray(rays.points[view])[rows[:, vi]]
        errors.append(np.linalg.norm(uv - observed[:, :, None, :], axis=-1))
    pixel_error = np.stack(errors, axis=-1)  # row tuple,node,option,view
    good &= np.all(~held | (pixel_error <= config.maximum_pixel_error), axis=-1)

    seen = {np.asarray(body["proposed_sources"], dtype=np.int32).tobytes() for body in pool}
    nodes = np.arange(3)
    for ci, row_tuple in enumerate(rows):
        options = []
        for node in range(3):
            if good[ci, node, 0]:
                options.append([0])
                stats["all_available_view_nodes_accepted"] += 1
            elif available[ci, node].sum() == 3:
                passing = (np.flatnonzero(good[ci, node, 1:]) + 1).tolist()
                options.append(passing)
                if passing:
                    stats["nodes_with_two_view_fallback"] += 1
                    stats["fallback_node_options"] += len(passing)
            else:
                options.append([])
        if any(not x for x in options):
            continue
        stats["row_combinations_with_complete_node_options"] += 1
        for variant, selection in enumerate(product(*options)):
            selection = np.asarray(selection, np.int32)
            node_views = held[ci, nodes, selection]
            source = np.where(node_views.T, row_tuple[:, None], -1).astype(np.int32)
            key = source.tobytes()
            if key in seen:
                stats["duplicate_source_plans_removed"] += 1
                continue
            seen.add(key)
            effective_rows = np.where(node_views.any(0), row_tuple, -1).astype(np.int32)
            pool.append(
                dict(
                    pool_index=len(pool),
                    combination_index=int(ci),
                    nodewise_variant_index=int(variant),
                    view_rows=effective_rows,
                    observation_rows=row_tuple.copy(),
                    pose_xyz=xyz[ci, nodes, selection].copy(),
                    triangulated_nodes=np.ones(3, bool),
                    proposed_sources=source,
                    ray_rms_mm=rms[ci, nodes, selection].copy(),
                    reprojection_error_px=pixel_error[ci, nodes, selection].T.copy(),
                    used_ray_count=int(node_views.sum()),
                    node_view_mask=node_views.copy(),
                    available_node_views=available[ci].copy(),
                    excluded_available_node_views=(available[ci] & ~node_views).copy(),
                    node_view_option_indices=selection.copy(),
                    supplemental_nodewise_body=True,
                    all_nodes_triangulated_without_template=True,
                    same_row_within_each_view=True,
                    direct_support_not_recomputed=True,
                    physical_IDs_assigned=False,
                )
            )
    stats["additional_candidate_count"] = len(pool) - legacy_count
    stats["geometry_candidates"] = len(pool)
    return pool, stats

"""Pure geometry for fixed, initially qualified temporal node correspondences.

This is not a FlowObservations object or current SLEAP support. Initial labels
must be bound to the qualified query table by the caller. Reported may recover
later for an original query, but cannot introduce a new initial query. No FB,
texture quality or calibrated confidence is invented. All four subsets per node
are retained, including missing rays and every failed measured condition.

Slot bits describe unavailable inputs. Subset bits describe missing views,
the frozen triangulate determinant failure (abs(det)<=1e-6), nonfinite solution,
RMS>1mm, any forward depth<0, predecessor-node step>6mm, nonfinite projection,
or any reprojection error>3px. Equality at the numerical limits is accepted.
Native image bounds are continuous [0,1200), without rounding or clipping.
NaN means an unmeasurable value; attempted=False means triangulation was skipped.
Normal rank/condition are descriptive; only frozen triangulate chooses its solve.

The original reference predecessor is allowed for an explicitly conditional
saved-path diagnostic. A later tracker adapter must supply its actual live
state0 predecessor. Frame zero has no synthesized predecessor and is rejected.
"""

from __future__ import annotations

import sys
from enum import IntFlag
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "labs/20260909_three_body")]
from motion import FlowConfig  # noqa: E402

from medaka3d.calib import triangulate  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

CONFIG = FlowConfig()
SUBSETS = ((0, 1, 2), (0, 1), (0, 2), (1, 2))
SCHEMA = "semantic_node_observation_v1"


class SlotRejection(IntFlag):
    INITIAL_ABSENT = 1
    UNREPORTED = 2
    PIXEL_NONFINITE = 4
    PIXEL_OUTSIDE = 8
    RAY_NONFINITE = 16


class SubsetRejection(IntFlag):
    MISSING_VIEWS = 1
    NONFINITE_SYSTEM = 2
    TRIANGULATION_SINGULAR = 4
    NONFINITE_INTERSECTION = 8
    RAY_RMS = 16
    BEHIND_RAY = 32
    NODE_STEP = 64
    PROJECTION_NONFINITE = 128
    REPROJECTION = 256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bit_names(enum, bits):
    return [flag.name.lower() for flag in enum if int(bits) & int(flag)]


def validate_inputs(uv, reported, initial_selected, previous, rig, initial, prior, native, horizon):
    for value in (initial, prior, native, horizon):
        require(type(value) is int and 0 <= value < 2**63, "nonnegative native integer required")
    require(
        initial <= prior and native == prior + 1 and 1 <= native - initial <= 199,
        "consecutive predecessor required after anchor, within fixed200",
    )
    require(native <= horizon <= initial + 199, "release horizon outside observed200")
    require(
        tuple(rig.views) == tuple(VIEWS) and rig.world_units == "mm", "registered view/mm order"
    )
    require(
        isinstance(uv, np.ndarray) and uv.shape == (3, 3, 3, 2) and uv.dtype == np.float32,
        "native float32 body/view/node/xy required",
    )
    for value in (reported, initial_selected):
        require(
            isinstance(value, np.ndarray) and value.shape == (3, 3, 3) and value.dtype == bool,
            "boolean body/view/node flags required",
        )
    require(not np.any(reported & ~initial_selected), "reported cannot create an initial query")
    require(
        isinstance(previous, np.ndarray)
        and previous.shape == (3, 3, 3)
        and previous.dtype.kind == "f"
        and np.isfinite(previous).all(),
        "finite body/node/xyz predecessor required",
    )


def allocate():
    subset_shape = (3, 3, 4)
    return dict(
        subset_xyz=np.full((*subset_shape, 3), np.nan),
        ray_rms_mm=np.full(subset_shape, np.nan),
        forward_depth_mm=np.full((*subset_shape, 3), np.nan),
        perpendicular_distance_mm=np.full((*subset_shape, 3), np.nan),
        step_mm=np.full(subset_shape, np.nan),
        reprojected_uv=np.full((*subset_shape, 3, 2), np.nan),
        reprojection_error_px=np.full((*subset_shape, 3), np.nan),
        normal_matrix=np.full((*subset_shape, 3, 3), np.nan),
        normal_determinant=np.full(subset_shape, np.nan),
        normal_rank=np.full(subset_shape, -1, np.int8),
        normal_condition=np.full(subset_shape, np.nan),
        rejection_bits=np.zeros(subset_shape, np.uint16),
        attempted=np.zeros(subset_shape, bool),
        admitted=np.zeros(subset_shape, bool),
        chosen_subset=np.full((3, 3), -1, np.int8),
        pose_xyz=np.full((3, 3, 3), np.nan),
        node_valid=np.zeros((3, 3), bool),
        used_views=np.zeros((3, 3, 3), bool),
    )


def evaluate_subset(arrays, index, selected, origins, directions, uv, predecessor, cameras):
    """Measure all feasible gates even if an earlier finite condition already fails."""
    a, key = arrays, index
    a["attempted"][key] = True
    projector = np.eye(3) - directions[..., :, None] * directions[..., None, :]
    normal = projector.sum(axis=0)
    rhs = (projector @ origins[..., None]).sum(axis=0)[..., 0]
    a["normal_matrix"][key] = normal
    if not np.isfinite(normal).all() or not np.isfinite(rhs).all():
        a["rejection_bits"][key] = np.uint16(SubsetRejection.NONFINITE_SYSTEM)
        return
    determinant = float(np.linalg.det(normal))
    a["normal_determinant"][key] = determinant
    a["normal_rank"][key] = np.linalg.matrix_rank(normal)
    a["normal_condition"][key] = np.linalg.cond(normal)
    # Exactly the original backend: no least-squares fallback or ray normalization.
    xyz, residual = triangulate(origins, directions)
    a["subset_xyz"][key], a["ray_rms_mm"][key] = xyz, residual
    bits = SubsetRejection(0)
    if not np.isfinite(xyz).all() or not np.isfinite(residual):
        bits |= (
            SubsetRejection.TRIANGULATION_SINGULAR
            if abs(determinant) <= 1e-6
            else SubsetRejection.NONFINITE_INTERSECTION
        )
        a["rejection_bits"][key] = np.uint16(bits)
        return
    delta = xyz - origins
    depth = np.sum(delta * directions, axis=-1)
    perpendicular = np.linalg.norm(delta - depth[:, None] * directions, axis=-1)
    a["forward_depth_mm"][key][selected] = depth
    a["perpendicular_distance_mm"][key][selected] = perpendicular
    step = float(np.linalg.norm(xyz - predecessor))
    a["step_mm"][key] = step
    if residual > CONFIG.maximum_ray_residual_mm:
        bits |= SubsetRejection.RAY_RMS
    if np.any(depth < 0):
        bits |= SubsetRejection.BEHIND_RAY
    if step > CONFIG.maximum_node_step_mm:
        bits |= SubsetRejection.NODE_STEP
    for vi in selected:
        projected = np.asarray(cameras[vi].project(xyz.copy()))
        require(projected.shape == (2,) and projected.dtype.kind == "f", "camera projection schema")
        error = float(np.linalg.norm(projected - uv[vi]))
        a["reprojected_uv"][key][vi] = projected
        a["reprojection_error_px"][key][vi] = error
        if not np.isfinite(projected).all() or not np.isfinite(error):
            bits |= SubsetRejection.PROJECTION_NONFINITE
        elif error > CONFIG.maximum_reprojection_px:
            bits |= SubsetRejection.REPROJECTION
    a["rejection_bits"][key] = np.uint16(bits)
    a["admitted"][key] = not bits


def build_observation(
    uv,
    reported,
    initial_selected,
    rig,
    previous_pose_xyz,
    *,
    initial_frame_idx,
    previous_frame_idx,
    frame_idx,
    release_horizon,
):
    """Return owned arrays and metadata, with 27 slots and 36 subset records.

    uv/flags use (body,view,node); previous pose uses (body,node,xyz). Subset
    metrics use (body,node,subset[,view,...]); used_views retains body/view/node.
    Returned labels are initial conditional slots. No current SL sources,
    observed/review state, likelihood, quality, FB or fit result is returned.
    """
    validate_inputs(
        uv,
        reported,
        initial_selected,
        previous_pose_xyz,
        rig,
        initial_frame_idx,
        previous_frame_idx,
        frame_idx,
        release_horizon,
    )
    # Copies isolate all caller arrays even from camera implementations.
    xy, flags, initial = uv.copy(), reported.copy(), initial_selected.copy()
    previous = previous_pose_xyz.astype(np.float64, copy=True)
    cameras = tuple(rig.camera(v) for v in VIEWS)
    bits = np.zeros((3, 3, 3), np.uint16)
    bits[~initial] |= np.uint16(SlotRejection.INITIAL_ABSENT)
    bits[~flags] |= np.uint16(SlotRejection.UNREPORTED)
    finite = np.isfinite(xy).all(-1)
    inside = finite & ((xy >= 0) & (xy < 1200)).all(-1)
    bits[~finite] |= np.uint16(SlotRejection.PIXEL_NONFINITE)
    bits[finite & ~inside] |= np.uint16(SlotRejection.PIXEL_OUTSIDE)
    origins = np.full((3, 3, 3, 3), np.nan)
    directions = np.full_like(origins, np.nan)
    for vi, camera in enumerate(cameras):
        active = bits[:, vi] == 0
        if not active.any():
            continue
        ray = camera.rays(xy[:, vi][active].copy())
        require(
            len(ray) == 2 and all(np.asarray(x).shape == (int(active.sum()), 3) for x in ray),
            "camera ray schema",
        )
        origins[:, vi][active], directions[:, vi][active] = ray
        bad = active & (
            ~np.isfinite(origins[:, vi]).all(-1) | ~np.isfinite(directions[:, vi]).all(-1)
        )
        bits[:, vi][bad] |= np.uint16(SlotRejection.RAY_NONFINITE)
    available = bits == 0
    a = allocate()
    a.update(
        uv=xy,
        reported=flags,
        initial_selected=initial,
        previous_pose_xyz=previous,
        slot_rejection_bits=bits,
        ray_available=available,
        ray_origins=origins,
        ray_directions=directions,
    )
    subset_views = np.zeros((4, 3), bool)
    for i, subset in enumerate(SUBSETS):
        subset_views[i, list(subset)] = True
    a["subset_views"] = subset_views
    records = []
    for body in range(3):
        for node in range(3):
            for si, subset in enumerate(SUBSETS):
                selected, key = np.asarray(subset), (body, node, si)
                if not available[body, selected, node].all():
                    a["rejection_bits"][key] = np.uint16(SubsetRejection.MISSING_VIEWS)
                else:
                    evaluate_subset(
                        a,
                        key,
                        selected,
                        origins[body, selected, node].copy(),
                        directions[body, selected, node].copy(),
                        xy[body, :, node].copy(),
                        previous[body, node],
                        cameras,
                    )
                records.append(
                    dict(
                        body=body,
                        node=node,
                        subset=si,
                        views=list(subset),
                        attempted=bool(a["attempted"][key]),
                        admitted=bool(a["admitted"][key]),
                        reasons=bit_names(SubsetRejection, a["rejection_bits"][key]),
                    )
                )
            candidates = np.flatnonzero(a["admitted"][body, node])
            if len(candidates):
                chosen = min(
                    candidates,
                    key=lambda i: (-len(SUBSETS[i]), float(a["ray_rms_mm"][body, node, i]), int(i)),
                )
                a["chosen_subset"][body, node] = chosen
                a["pose_xyz"][body, node] = a["subset_xyz"][body, node, chosen]
                a["node_valid"][body, node] = True
                a["used_views"][body, list(SUBSETS[chosen]), node] = True
    slots = [
        dict(
            body=b,
            view=v,
            node=n,
            available=bool(available[b, v, n]),
            reasons=bit_names(SlotRejection, bits[b, v, n]),
        )
        for b in range(3)
        for v in range(3)
        for n in range(3)
    ]
    for array in a.values():
        array.setflags(write=False)
    metadata = dict(
        schema=SCHEMA,
        initial_frame_idx=initial_frame_idx,
        previous_frame_idx=previous_frame_idx,
        frame_idx=frame_idx,
        release_horizon=release_horizon,
        lookahead_frames=release_horizon - frame_idx,
        views=list(VIEWS),
        image_hw=[1200, 1200],
        slot_records=slots,
        subset_records=records,
        slot_rejection_bits={f.name.lower(): int(f) for f in SlotRejection},
        subset_rejection_bits={f.name.lower(): int(f) for f in SubsetRejection},
        geometry_limits=dict(
            ray_rms_mm=CONFIG.maximum_ray_residual_mm,
            reprojection_px=CONFIG.maximum_reprojection_px,
            predecessor_step_mm=CONFIG.maximum_node_step_mm,
            minimum_forward_depth_mm=0.0,
        ),
        subset_order=[list(s) for s in SUBSETS],
        ranking=["more_views", "lower_ray_rms", "subset_index"],
        normal_rank_condition_are_diagnostic=True,
        initial_query_qualification_by_caller=True,
        predecessor_origin_bound_by_caller=True,
        current_SL_support_established=False,
        biological_ID_truth=False,
        quality_or_FB_fabricated=False,
        model_or_fit_called=False,
    )
    return dict(arrays=a, metadata=metadata)

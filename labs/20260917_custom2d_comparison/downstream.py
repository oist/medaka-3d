"""Windowed reuse of the frozen C28 -> C33 -> C30v4 -> C34 geometry chain.

Inputs retain C01 detection row order. Native frames are rebased only inside
the legacy functions. No reference permutation is applied to estimated tracks.
Windowed Viterbi has different boundary conditions from a full-series solve.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import c01_s16_predictions as c01  # noqa: E402
import c25_variable_cardinality as c25  # noqa: E402
import c30_slot_stitch as c30  # noqa: E402
import c33_ray_rejection as c33  # noqa: E402
import c34_microflip as c34  # noqa: E402

from medaka3d.calib import triangulate  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

NODES = ("pectoral", "mouth")
POSE_NODES = ("mouth", "pectoral", "tail")
LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 0.767, 2.0, 8.0)


@dataclass
class PreparedWindow:
    frames: np.ndarray
    detections: dict
    rig: object
    label_at: dict
    row_at: dict
    counts: dict
    rays: dict
    base: int
    degraded: np.ndarray
    input_row_slices: dict
    prepare_seconds: float


def _slice_detections(detection, start, stop):
    lo, hi = np.searchsorted(detection.frame_idx, [start, stop])
    local = c01.Detections(
        experiment=detection.experiment,
        view=detection.view,
        frame_idx=detection.frame_idx[lo:hi].copy() - start,
        points=detection.points[lo:hi].copy(),
        score=detection.score[lo:hi].copy(),
        n_frames=stop - start,
        nodes=detection.nodes,
    )
    return local, slice(int(lo), int(hi))


def prepare_window(detections_by_view, labels_by_view, rig, start: int, stop: int):
    """Prepare native [start, stop), from full C01 Detections and per-row labels.

    Labels must be unique within each view/frame, or negative for untracked.
    Dense monotone renaming preserves equality and state order, while avoiding
    integer overflow when the legacy solver packs three view labels together.
    This does not relink detections, create points, or discard an extra fish.
    """
    began = time.perf_counter()
    if not isinstance(start, (int, np.integer)) or not isinstance(stop, (int, np.integer)):
        raise ValueError("Native interval bounds must be integers")
    total = {int(detections_by_view[v].n_frames) for v in VIEWS}
    if len(total) != 1 or not 0 <= start < stop <= next(iter(total)):
        raise ValueError("All three views must cover the requested nonempty native interval")
    detections, uv, label_at, row_at, counts, slices = {}, {}, {}, {}, {}, {}
    n_labels = 0
    # C30's successive +/-1 then +/-2 operations have effective support +/-3.
    wide_start, wide_stop = max(0, start - 3), min(next(iter(total)), stop + 3)
    degraded = np.zeros(wide_stop - wide_start, bool)
    for view in VIEWS:
        original = detections_by_view[view]
        labels = np.asarray(labels_by_view[view])
        if labels.shape != original.frame_idx.shape or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"{view}: integer labels must correspond to every C01 row")
        if np.any(np.diff(original.frame_idx) < 0):
            raise ValueError(f"{view}: detections must be ordered by native frame")
        if any(node not in original.nodes for node in POSE_NODES):
            raise ValueError(f"{view}: mouth, pectoral and tail nodes are required")
        local, row_slice = _slice_detections(original, start, stop)
        held = labels[row_slice].copy()
        live = held >= 0
        names, dense = np.unique(held[live], return_inverse=True)
        held[live] = dense
        pairs = np.column_stack([local.frame_idx[live], held[live]])
        if len(np.unique(pairs, axis=0)) != len(pairs):
            raise ValueError(f"{view}: duplicate 2D identity in a frame")
        n_labels = max(n_labels, len(names))
        detections[view], slices[view] = local, (row_slice.start, row_slice.stop)
        uv[view], label_at[view], row_at[view], counts[view] = c25.compact_layout(
            local, held, NODES
        )
        wide, _ = _slice_detections(original, wide_start, wide_stop)
        best = c30.best_scores(wide, wide.n_frames)
        degraded |= (best >= 0) & (best < c30.SCORE_DEGRADED)
    for h in (1, 2):
        widened = degraded.copy()
        widened[h:] |= degraded[:-h]
        widened[:-h] |= degraded[h:]
        degraded = widened
    return PreparedWindow(
        frames=np.arange(start, stop),
        detections=detections,
        rig=rig,
        label_at=label_at,
        row_at=row_at,
        counts=counts,
        rays=c25.compact_rays(rig, uv, NODES),
        base=n_labels + 2,
        degraded=degraded[start - wide_start : stop - wide_start].copy(),
        input_row_slices=slices,
        prepare_seconds=time.perf_counter() - began,
    )


def _stored_precision(arm):
    out = {key: np.array(value, copy=True) for key, value in arm.items()}
    for key in ("xyz", "residual_mm", "margin_mm"):
        if key in out:
            out[key] = out[key].astype(np.float32)
    return out


def attach_pose(arm, prepared):
    """Reconstruct display nodes from the chosen rows; preserve stored anchor xyz."""
    source = arm["source"]
    uv = np.full((*source.shape, len(POSE_NODES), 2), np.nan)
    origins = np.full((*source.shape, len(POSE_NODES), 3), np.nan)
    directions = np.full_like(origins, np.nan)
    for column, view in enumerate(VIEWS):
        det = prepared.detections[view]
        rows = source[:, :, column]
        frame, slot = np.where(rows >= 0)
        if not len(frame):
            continue
        if np.any(rows[frame, slot] >= det.counts()[frame]):
            raise ValueError("A chosen source row is outside its native frame")
        flat = det.start[frame] + rows[frame, slot]
        values = det.points[flat][:, [det.nodes.index(node) for node in POSE_NODES]]
        uv[frame, slot, column] = values
        finite = np.isfinite(values).all(axis=-1)
        ray_o, ray_d = prepared.rig.camera(view).rays(values[finite])
        local_o, local_d = (
            np.full((*values.shape[:-1], 3), np.nan),
            np.full((*values.shape[:-1], 3), np.nan),
        )
        local_o[finite], local_d[finite] = ray_o, ray_d
        origins[frame, slot, column], directions[frame, slot, column] = local_o, local_d
    origins, directions = origins.swapaxes(2, 3), directions.swapaxes(2, 3)
    valid = np.isfinite(origins).all(-1) & np.isfinite(directions).all(-1)
    pose, node_ray = triangulate(origins, directions, valid=valid)
    pose[:, :, POSE_NODES.index("pectoral")] = arm["xyz"]
    arm["pose"], arm["uv"] = pose, uv
    arm["node_ray_rms_mm"], arm["node_view_counts"] = node_ray, valid.sum(-1)
    # Match Reconstruction.resolved, used by the saved legacy label arms.
    resolved = arm["n_views_used"] >= 2
    arm["slot_track"] = np.where(resolved[:, None], np.arange(3), -1).astype(np.int32)
    return arm


def _native_events(events, offset):
    out = []
    for item in events:
        event = dict(item)
        for key in ("frame", "pre_frame", "post_frame", "first_frame", "last_frame"):
            if key in event:
                event[key] = int(event[key]) + offset
        if "zone" in event:
            event["zone"] = [int(frame) + offset for frame in event["zone"]]
        out.append(event)
    return out


def postprocess(c28, prepared):
    """Apply the original C33, C30v4 and C34 rules to one local reconstruction."""
    if (c30.VELOCITY_NEEDS_MEDIAN_MM, c30.BRIDGE_MARGIN_MM) != (2.5, 30.0):
        raise ValueError("The inherited C30v4 thresholds have changed")
    fixed, c33_stats = c33.repair(c28, prepared.detections, prepared.rig)
    fixed["margin_mm"] = c28["margin_mm"].copy()
    fixed["n_views_used"] = fixed["slot_rays"].min(axis=1).astype(np.int8)
    fixed = _stored_precision(fixed)
    xyz = fixed["xyz"].astype(np.float64)
    # These intentionally match the different original C30 and C34 guards.
    ok = (fixed["n_views_used"] >= 2) & np.isfinite(xyz).any(axis=(1, 2))
    permutation, joins, refusals = c30.stitch(
        xyz,
        np.flatnonzero(ok),
        c30.WINDOW,
        c30.MARGIN_MM,
        c30.GAP_FRAMES,
        velocity_window=c30.VELOCITY_WINDOW,
        degraded=prepared.degraded,
    )
    rows = np.arange(len(xyz))[:, None]
    for key in ("xyz", "residual_mm", "slot_rays", "source"):
        fixed[key] = fixed[key][rows, permutation]
    fixed = _stored_precision(fixed)
    xyz = fixed["xyz"].astype(np.float64)
    ok = (fixed["n_views_used"] >= 2) & np.isfinite(xyz).all(axis=(1, 2))
    windows = c34.find_windows(xyz, np.flatnonzero(ok))
    final = c34.apply_windows(
        {key: fixed[key] for key in ("xyz", "residual_mm", "slot_rays", "source")}, windows
    )
    final["margin_mm"], final["n_views_used"] = fixed["margin_mm"], fixed["n_views_used"]
    combined = c34.apply_windows({"permutation": permutation}, windows)["permutation"]
    final["slot_perm_from_c28"] = combined
    offset = int(prepared.frames[0])
    return attach_pose(_stored_precision(final), prepared), {
        "c33": c33_stats,
        "c30_rule": "v4: original median/velocity joins followed by clean-end zone audit",
        "c30_velocity_needs_median_mm": c30.VELOCITY_NEEDS_MEDIAN_MM,
        "c30_bridge_margin_mm": c30.BRIDGE_MARGIN_MM,
        "c30_score_dilation_effective_radius_frames": 3,
        "c30_joins": _native_events(joins, offset),
        "c30_refusals": _native_events(refusals, offset),
        "c34_windows": _native_events(windows, offset),
        "event_frame_coordinates": "native recording frames; zone and window ends inclusive",
    }


def compare_reference(arm, reference_c28, frames, core_interval=None):
    """Measure C28 reproduction after ONE constant permutation over the core.

    This diagnostic never changes the arm and is not an identity-accuracy score.
    Reference arrays can span the full recording or exactly the prepared window.
    """
    frames = np.asarray(frames)
    if core_interval is None:
        core_interval = (int(frames[0]), int(frames[-1] + 1))
    lo, hi = core_interval
    if not frames[0] <= lo < hi <= frames[-1] + 1:
        raise ValueError("Core interval must be contained in the prepared native window")
    core = (frames >= lo) & (frames < hi)
    reference = {}
    for key in ("source", "xyz"):
        values = np.asarray(reference_c28[key])
        if len(values) != len(frames):
            if len(values) <= frames[-1]:
                raise ValueError("Reference must span the recording or exactly this window")
            values = values[frames]
        reference[key] = values[core]
    source, xyz = arm["source"][core], arm["xyz"][core]
    scores = []
    for q in permutations(range(3)):
        s, x = source[:, q], xyz[:, q]
        comparable = np.isfinite(x).all(-1) & np.isfinite(reference["xyz"]).all(-1)
        distance = np.linalg.norm(x - reference["xyz"], axis=-1)[comparable]
        scores.append(
            {
                "permutation": list(q),
                "source_entry_mismatches": int((s != reference["source"]).sum()),
                "mean_paired_xyz_distance_mm": float(distance.mean()) if distance.size else None,
            }
        )
    best = min(
        scores,
        key=lambda row: (
            row["source_entry_mismatches"],
            row["mean_paired_xyz_distance_mm"]
            if row["mean_paired_xyz_distance_mm"] is not None
            else np.inf,
        ),
    )
    q = best["permutation"]
    s, x = source[:, q], xyz[:, q]
    matched = (s == reference["source"]).all(axis=(1, 2))
    supported = (reference["source"] >= 0).any(axis=(1, 2))
    comparable = np.isfinite(x).all(-1) & np.isfinite(reference["xyz"]).all(-1)
    distance = np.linalg.norm(x - reference["xyz"], axis=-1)[comparable]
    xyz_equal = np.array_equal(x, reference["xyz"], equal_nan=True)
    exact = bool(supported.any() and matched.all() and xyz_equal)
    return {
        "status": "exact_core_reproduction" if exact else "windowed_core_differs",
        "core_interval_half_open": [int(lo), int(hi)],
        "single_permutation_output_reference_slot_selects_window_slot": q,
        "permutation_applied_to_output": False,
        "all_six_permutation_scores": scores,
        "core_frames": int(core.sum()),
        "core_supported_frames": int(supported.sum()),
        "source_exact_frame_fraction": float(matched.mean()),
        "source_exact_entry_fraction": float((s == reference["source"]).mean()),
        "source_mismatch_native_frames": frames[core][~matched].tolist(),
        "xyz_exact_equal_nan": xyz_equal,
        "xyz_finite_mask_equal": bool(
            np.array_equal(np.isfinite(x), np.isfinite(reference["xyz"]))
        ),
        "xyz_max_paired_distance_mm": float(distance.max()) if distance.size else None,
        "limitation": "Windowed Viterbi boundaries differ; reproduction is not ID accuracy",
    }


def solve_window(prepared, lambda_mm=0.767, mu=4.0, core_interval=None, reference_c28=None):
    """Run one inherited lambda arm; reuse PreparedWindow for lambda sensitivity."""
    if not np.isfinite(lambda_mm) or lambda_mm < 0 or not np.isfinite(mu) or mu < 0:
        raise ValueError("Lambda and missing-node cost must be finite and nonnegative")
    began = time.perf_counter()
    local_frames = np.arange(len(prepared.frames))
    chosen, objective, inverse, tables, margin, pruned = c25.solve(
        prepared.rays,
        NODES,
        prepared.label_at,
        prepared.counts,
        local_frames,
        lambda_mm,
        mu,
        prepared.base,
    )
    xyz, residual, source, rays = c25.chosen_geometry(
        prepared.rays,
        NODES,
        tables,
        inverse,
        chosen,
        prepared.label_at,
        prepared.row_at,
        local_frames,
        "pectoral",
    )
    c28 = _stored_precision(
        {
            "xyz": xyz,
            "residual_mm": residual,
            "source": source,
            "slot_rays": rays,
            "margin_mm": margin * 3,
            "n_views_used": rays.min(axis=1).astype(np.int8),
            "state": chosen,
        }
    )
    final, diagnostics = postprocess(c28, prepared)
    attach_pose(c28, prepared)
    diagnostics.update(
        {
            "lambda_mm": float(lambda_mm),
            "missing_node_cost_mm": float(mu),
            "viterbi_objective": float(objective),
            "pruned_frames": len(pruned),
            "states_per_frame_mean": float(
                np.array([len(table) for table in tables])[inverse].mean()
            ),
            "states_per_frame_max": int(max(len(table) for table in tables)),
            "prepare_seconds": prepared.prepare_seconds,
            "solve_seconds": time.perf_counter() - began,
            "input_row_slices": prepared.input_row_slices,
            "pose_node_order": list(POSE_NODES),
            "source_view_order": list(VIEWS),
            "pose_anchor": "float32 pectoral xyz; other nodes triangulated from chosen rows",
            "scope": "windowed common legacy downstream; no additional 3D interval repair",
            "reference_reproduction": (
                compare_reference(c28, reference_c28, prepared.frames, core_interval)
                if reference_c28 is not None
                else {"status": "reference_not_supplied"}
            ),
        }
    )
    return {
        "frames": prepared.frames.copy(),
        "c28": c28,
        "final": final,
        "diagnostics": diagnostics,
    }


def run_downstream(detections_by_view, labels_by_view, rig, start, stop, **kwargs):
    """Convenience wrapper for a single lambda; see prepare_window/solve_window."""
    return solve_window(
        prepare_window(detections_by_view, labels_by_view, rig, start, stop), **kwargs
    )

"""Observation-aware diagnostics, independent of trajectory selection."""

from __future__ import annotations

from itertools import permutations

import numpy as np


def describe(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "n": int(values.size),
        **{
            key: float(function(values)) if values.size else None
            for key, function in (
                ("median", np.median),
                ("p95", lambda a: np.quantile(a, 0.95)),
                ("maximum", np.max),
            )
        },
    }


def tracklet_metrics(frame_idx, points, labels, clean_edge, start, stop, view):
    mask = (frame_idx >= start) & (frame_idx < stop)
    if labels.shape != frame_idx.shape or clean_edge.shape != frame_idx.shape:
        raise ValueError("Each input row needs its label and incoming-edge status")
    if np.any(labels < 0):
        raise ValueError("All input detections, including isolated rows, must be retained")
    if len(np.unique(np.column_stack([frame_idx, labels]), axis=0)) != len(labels):
        raise ValueError("A fragment has duplicate observations in a frame")
    previous, steps, intervals = {}, [], []
    for row, (frame, label) in enumerate(zip(frame_idx, labels, strict=True)):
        before = previous.get(int(label))
        if before is not None:
            dt = int(frame - frame_idx[before])
            if dt <= 0:
                raise ValueError("Tracklets must move forward in native frames")
            if mask[row]:
                d = np.linalg.norm(points[row, :2] - points[before, :2], axis=-1)
                finite = d[np.isfinite(d)]
                steps.append(float(np.max(finite)) if len(finite) else np.nan)
                intervals.append(dt)
        previous[int(label)] = row
    _, lengths = np.unique(labels[mask], return_counts=True)
    steps, intervals = np.asarray(steps), np.asarray(intervals)
    threshold = 60.0 if view == "top_cam" else 50.0
    return {
        "detection_rows": int(mask.sum()),
        "retained_rows": int(mask.sum()),
        "retention": 1.0 if mask.any() else None,
        "fragments": len(lengths),
        "singleton_rows": int((lengths == 1).sum()),
        "incoming_clean_edges": int(clean_edge[mask].sum()),
        "clean_edge_fraction_of_rows": float(clean_edge[mask].mean()) if mask.any() else None,
        "linked_edges": len(steps),
        "gap_links": int((intervals > 1).sum()),
        "large_links": int((steps > threshold * intervals).sum()),
        "step_px": describe(steps),
        "fragment_length_frames": describe(lengths),
        "scope": "Coverage and motion diagnostics; not ground-truth identity purity",
    }


def geometry_metrics(stage, mask):
    xyz = stage["xyz"][mask]
    present = np.isfinite(xyz).all(-1)
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
    source = stage["source"][mask]
    out = {
        "native_frames": len(xyz),
        "possible_fish_frames": int(present.size),
        "observed_fish_frames": int(present.sum()),
        "missing_fish_frames": int((~present).sum()),
        "availability": float(present.mean()),
        "observed_native_steps": int(np.isfinite(step).sum()),
        "steps_over_4mm": int((step > 4).sum()),
        "steps_over_10mm": int((step > 10).sum()),
        "step_mm": describe(step),
        "two_view_fish_frames": int((present & ((source >= 0).sum(-1) == 2)).sum()),
        "three_view_fish_frames": int((present & ((source >= 0).sum(-1) == 3)).sum()),
        "identity_review_fish_frames": int(
            stage.get("identity_review", np.zeros_like(mask)[:, None])[mask].sum()
        ),
        "scope": "Small displacements or abstentions do not establish fewer swaps",
    }
    if "residual_mm" in stage:
        out["residual_mm"] = describe(stage["residual_mm"][mask])
    if "pose" in stage:
        out["mouth_tail_length_mm"] = describe(
            np.linalg.norm(stage["pose"][mask, :, 0] - stage["pose"][mask, :, 2], axis=-1)
        )
    return out


def prefix_permutation(stage, reference, frames, core_start):
    """Return a fixed display permutation; never modify estimated arrays."""
    good = (
        (frames < core_start)
        & np.isfinite(stage["xyz"]).all((1, 2))
        & np.isfinite(reference["xyz"]).all((1, 2))
    )
    indices = np.flatnonzero(good)[:17]
    if len(indices) < 5:
        return {"valid": False, "permutation": [0, 1, 2], "frames_used": len(indices)}
    perms = list(permutations(range(3)))
    costs = [
        np.median(np.linalg.norm(reference["xyz"][indices] - stage["xyz"][indices][:, p], axis=-1))
        for p in perms
    ]
    rank = np.argsort(costs)
    return {
        "valid": True,
        "permutation": list(perms[rank[0]]),
        "frames_used": len(indices),
        "median_distance_mm": float(costs[rank[0]]),
        "runnerup_margin_mm": float(costs[rank[1]] - costs[rank[0]]),
        "scope": "Display names only: one permutation from pre-core context",
    }


def validate_sources(stage, frames, views):
    """Reject fabricated indices, repeated observations and misleading missingness."""
    for vi, view in enumerate(views):
        data = views[view]
        left = np.searchsorted(data["frame_idx"], frames)
        right = np.searchsorted(data["frame_idx"], frames + 1)
        rows = stage["source"][:, :, vi]
        if np.any(rows < -1) or np.any(rows >= (right - left)[:, None]):
            raise ValueError(f"Invalid measured source in {view}")
        for f in range(len(frames)):
            used = rows[f][rows[f] >= 0]
            if len(used) != len(np.unique(used)):
                raise ValueError(f"One detection used by multiple fish in {view}, f{frames[f]}")
        if "uv" in stage:
            present = rows >= 0
            if np.any(np.isfinite(stage["uv"][:, :, vi][~present])):
                raise ValueError("Predicted pixels were stored as measured observations")
            expected = data["points"][(left[:, None] + np.maximum(rows, 0))[present]]
            if not np.array_equal(stage["uv"][:, :, vi][present], expected, equal_nan=True):
                raise ValueError("Measured pixels differ from the original detection")
    resolved = np.isfinite(stage["xyz"]).all(-1)
    if np.any(resolved & ((stage["source"] >= 0).sum(-1) < 2)):
        raise ValueError("A measured 3D fish has fewer than two cameras")
    return {"status": "pass", "frames": len(frames), "no_invented_detections": True}

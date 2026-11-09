"""Experimental geometry-only warning detection and clean-end motion bridges."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from medaka3d.eval.review_identity import contiguous_runs

PERMS = np.array(list(permutations(range(3))), dtype=np.int8)


def signals(xyz, source, residual, local_tracks, margin, config):
    """No reference labels enter this function. Preserve missing-data signals."""
    n = len(xyz)
    valid = np.isfinite(xyz).all(axis=-1) & ((source >= 0).sum(axis=-1) >= 2)
    distance = np.linalg.norm(xyz[:, :, None] - xyz[:, None, :], axis=-1)
    pair_valid = valid[:, :, None] & valid[:, None, :]
    distance[~pair_valid] = np.inf
    distance[:, np.arange(3), np.arange(3)] = np.inf
    nearest = np.min(distance, axis=(1, 2))
    adjacent = valid[:-1] & valid[1:]
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
    large = np.r_[False, np.any(adjacent & (step > config["step_mm"]), axis=1)]
    rays = (source >= 0).sum(axis=-1)
    ray_change = np.r_[False, np.any(np.diff(rays, axis=0) != 0, axis=1)]
    track_change = np.r_[False, np.any(adjacent & (local_tracks[1:] != local_tracks[:-1]), axis=1)]
    d = np.linalg.norm(xyz[:-1, :, None] - xyz[1:, None, :], axis=-1)
    costs = np.stack([d[:, np.arange(3), q].sum(axis=1) for q in PERMS], axis=1)
    exchange = np.r_[False, (np.argmin(costs, axis=1) != 0) & adjacent.all(axis=1)]
    flags = {
        "missing_position": ~valid.all(axis=1),
        "large_step": large,
        "nonidentity_nearest": exchange,
        "ray_change": ray_change,
        "local_track_change": track_change,
        "residual_spike": np.any((rays == 3) & (residual > config["residual_mm"]), axis=1),
        "margin_low": np.isfinite(margin) & (margin < config["margin_mm"]),
    }
    assert all(v.shape == (n,) for v in flags.values())
    return flags, nearest, valid


def warning_intervals(flags, nearest, times, weights, proximity_mm, padding_s, broad=False):
    """Union padded warning cores without imposing an artificial maximum length."""
    active = {k: v for k, v in flags.items() if broad or k != "margin_low"}
    active["proximity"] = nearest < proximity_mm
    raw = np.logical_or.reduce(list(active.values())) & (weights > 0)
    expanded = []
    first = int(np.flatnonzero(weights > 0)[0])
    for lo, hi in contiguous_runs(raw):
        start = max(first, int(np.searchsorted(times, times[lo] - padding_s)))
        end = min(
            len(times),
            int(np.searchsorted(times, times[min(hi, len(times) - 1)] + padding_s, side="right")),
        )
        if expanded and start <= expanded[-1][1]:
            expanded[-1][1] = max(expanded[-1][1], end)
        else:
            expanded.append([start, end])
    mask = np.zeros(len(times), bool)
    rows = []
    for i, (lo, hi) in enumerate(expanded):
        mask[lo:hi] = True
        rows.append(
            {
                "interval_id": i,
                "start_frame": lo,
                "stop_frame_exclusive": hi,
                "start_s": float(times[lo]),
                "end_s": float(times[min(hi, len(times) - 1)]),
                "duration_s": float(weights[lo:hi].sum()),
                "missing_geometry_s": float(weights[lo:hi][flags["missing_position"][lo:hi]].sum()),
                "reasons": ",".join(k for k, v in active.items() if v[lo:hi].any()),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "interval_id",
            "start_frame",
            "stop_frame_exclusive",
            "start_s",
            "end_s",
            "duration_s",
            "missing_geometry_s",
            "reasons",
        ],
    ), mask


def _fit(xyz, tracks, times, indices, config):
    if len(indices) < config["minimum_samples"]:
        return None
    if times[indices[-1]] - times[indices[0]] < config["minimum_span_s"]:
        return None
    points = xyz[indices]
    if not np.isfinite(points).all() or (tracks[indices] < 0).any():
        return None
    if np.any(tracks[indices] != tracks[indices[0]]):
        return None
    t = times[indices]
    center = float(t.mean())
    dt = t - center
    mean = points.mean(axis=0)
    velocity = np.einsum("f,fij->ij", dt, points) / (dt @ dt)
    return center, mean, velocity


def _costs(left, right):
    return np.array([np.linalg.norm(left - right[q], axis=-1).sum() for q in PERMS])


def _winner(costs):
    order = np.argsort(costs)
    return int(order[0]), float(costs[order[1]] - costs[order[0]])


class MotionBridge:
    """Linker Protocol adapter; timestamps can be supplied independently of instances."""

    name = "geometry_warning_motion_bridge"

    def __init__(self, times, intervals, local_tracks, config, method="bidirectional"):
        self.times, self.intervals, self.tracks = times, intervals, local_tracks
        self.config, self.method = config, method
        self.decisions = []

    def link(self, instances):
        xyz = np.full((len(self.times), 3, 3), np.nan)
        for instance in instances:
            xyz[instance.frame, instance.slot] = instance.xyz
        perm, self.decisions = self.permutation(xyz)
        return np.take_along_axis(xyz, perm[:, :, None], axis=1)

    def permutation(self, xyz):
        """Apply net clean-end permutations; preserve all warning interiors in the audit."""
        times, cfg = self.times, self.config
        perm = np.tile(np.arange(3, dtype=np.int8), (len(times), 1))
        current = np.arange(3)
        decisions = []
        warning = np.zeros(len(times), bool)
        for interval in self.intervals.itertuples():
            warning[interval.start_frame : interval.stop_frame_exclusive] = True
        for interval in self.intervals.itertuples():
            a, b = int(interval.start_frame) - 1, int(interval.stop_frame_exclusive)
            if a < 0 or b >= len(times):
                continue
            row = {
                "interval_id": interval.interval_id,
                "before_frame": a,
                "after_frame": b,
                "start_s": float(times[a]),
                "end_s": float(times[b]),
                "method": self.method,
                "permutation": [0, 1, 2],
                "decision": "refused",
            }
            if times[b] - times[a] > cfg["max_interval_s"]:
                row["reason"] = "interval_too_long"
                decisions.append(row)
                continue
            pre = np.arange(np.searchsorted(times, times[a] - cfg["history_s"]), a + 1)
            post = np.arange(b, np.searchsorted(times, times[b] + cfg["history_s"], side="right"))
            if warning[pre].any() or warning[post].any():
                row["reason"] = "motion_window_overlaps_another_warning"
                decisions.append(row)
                continue
            left = _fit(xyz, self.tracks, times, pre, cfg)
            right = _fit(xyz, self.tracks, times, post, cfg)
            if left is None or right is None:
                row["reason"] = "endpoint_motion_incomplete_or_track_boundary"
                decisions.append(row)
                continue
            tl, pl, vl = left
            tr, pr, vr = right
            # Evaluate fits at actual clean endpoints, never at frame-count-derived times.
            p0 = pl + vl * (times[a] - tl)
            p1 = pr + vr * (times[b] - tr)
            position = _winner(_costs(p0, p1))
            forward = _winner(_costs(p0 + vl * (times[b] - times[a]), p1))
            backward = _winner(_costs(p0, p1 - vr * (times[b] - times[a])))
            for name, (best, margin) in zip(
                ("position", "forward", "backward"), (position, forward, backward), strict=True
            ):
                row[name + "_best"] = best
                row[name + "_margin_mm"] = margin
            evidence = {
                "position": [position],
                "bidirectional": [forward, backward],
                "consensus": [position, forward, backward],
            }[self.method]
            accepted = (
                len({v[0] for v in evidence}) == 1
                and min(v[1] for v in evidence) > cfg["assignment_margin_mm"]
            )
            if accepted:
                q = PERMS[evidence[0][0]]
                row["permutation"] = q.tolist()
                row["decision"] = "corrected" if evidence[0][0] else "retained"
                row["reason"] = "decisive_agreement"
                # q maps baseline before slots to baseline after slots; composition
                # maps the currently named output to baseline after slots.
                current = q[current]
                perm[b:] = current
            else:
                row["reason"] = "ambiguous_or_direction_disagreement"
            decisions.append(row)
        return perm, decisions

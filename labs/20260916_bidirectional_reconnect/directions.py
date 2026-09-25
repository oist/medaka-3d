"""Native-frame, fixed-coordinate directional diagnostics; no identity mutation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DirectionConfig:
    window: int = 17
    guard: int = 2
    angle_deg: float = 80.0
    distance_mm: float = 4.0
    angle_min_step_mm: float = 1.0
    min_direction_mm: float = 0.05
    minimum_fraction: float = 0.8


def diagnose(xyz, frames, config=None):
    cfg = config or DirectionConfig()
    x, frames = np.asarray(xyz, float), np.asarray(frames)
    if x.shape != (len(frames), 3, 3) or np.any(np.diff(frames) != 1):
        raise ValueError("Expected contiguous native frames with three 3D anchors")
    if cfg.window < 2 or cfg.guard < 0 or not 0 < cfg.minimum_fraction <= 1:
        raise ValueError("Invalid reference window configuration")
    d = np.diff(x, axis=0)
    length = np.linalg.norm(d, axis=-1)
    past = np.full_like(d, np.nan)
    future = np.full_like(d, np.nan)
    spread_past = np.full(length.shape, np.nan)
    spread_future = np.full(length.shape, np.nan)
    for i in range(len(d)):
        for destination, spread, lo, hi in (
            (past, spread_past, i - cfg.guard - cfg.window, i - cfg.guard),
            (future, spread_future, i + cfg.guard + 1, i + cfg.guard + 1 + cfg.window),
        ):
            if lo < 0 or hi > len(d):
                continue
            for slot in range(3):
                block = d[lo:hi, slot]
                block = block[np.isfinite(block).all(-1)]
                if len(block) < int(np.ceil(cfg.window * cfg.minimum_fraction)):
                    continue
                v = np.median(block, axis=0)
                if np.linalg.norm(v) < cfg.min_direction_mm:
                    continue
                destination[i, slot] = v
                spread[i, slot] = np.median(np.linalg.norm(block - v, axis=-1))

    def compare(v):
        norm = np.linalg.norm(v, axis=-1)
        denominator = length * norm
        cosine = np.full(length.shape, np.nan)
        np.divide(np.sum(d * v, axis=-1), denominator, out=cosine, where=denominator > 0)
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        parallel = np.sum(d * v, axis=-1) / np.where(norm > 0, norm, np.nan)
        perpendicular = np.sqrt(np.maximum(0, length**2 - parallel**2))
        ratio = length / np.where(norm > 0, norm, np.nan)
        return angle, perpendicular, ratio

    ap, pp, rp = compare(past)
    af, pf, rf = compare(future)
    distance = length > cfg.distance_mm
    forward = distance | ((length > cfg.angle_min_step_mm) & (ap >= cfg.angle_deg))
    bidirectional = forward | ((length > cfg.angle_min_step_mm) & (af >= cfg.angle_deg))
    return dict(
        frames=frames[1:],
        step_mm=length,
        past_angle_deg=ap,
        future_angle_deg=af,
        past_velocity=past,
        future_velocity=future,
        past_perpendicular_mm=pp,
        future_perpendicular_mm=pf,
        past_speed_ratio=rp,
        future_speed_ratio=rf,
        past_spread_mm=spread_past,
        future_spread_mm=spread_future,
        distance=distance,
        forward=forward,
        bidirectional=bidirectional,
    )


def episodes(result, start_frame, stop_frame, method="bidirectional", merge_frames=3, pad=2):
    frames = result["frames"]
    flags = result[method] & ((frames >= start_frame) & (frames < stop_frame))[:, None]
    indices = np.flatnonzero(flags.any(-1))
    groups = []
    for i in indices:
        frame = int(frames[i])
        if not groups or frame - groups[-1]["last_edge_frame"] > merge_frames:
            groups.append(dict(first_edge_frame=frame, last_edge_frame=frame, edges=[]))
        group = groups[-1]
        group["last_edge_frame"] = frame
        for slot in np.flatnonzero(flags[i]):
            group["edges"].append(
                {
                    "frame": frame,
                    "slot": int(slot),
                    **{
                        name: (
                            float(result[name][i, slot])
                            if np.isfinite(result[name][i, slot])
                            else None
                        )
                        for name in (
                            "step_mm",
                            "past_angle_deg",
                            "future_angle_deg",
                            "past_perpendicular_mm",
                            "future_perpendicular_mm",
                            "past_speed_ratio",
                            "future_speed_ratio",
                        )
                    },
                    "distance": bool(result["distance"][i, slot]),
                    "forward": bool(result["forward"][i, slot]),
                    "bidirectional_only": bool(
                        result["bidirectional"][i, slot] and not result["forward"][i, slot]
                    ),
                }
            )
    for i, group in enumerate(groups):
        group.update(
            index=i,
            start_frame=group["first_edge_frame"] - pad,
            stop_frame_exclusive=group["last_edge_frame"] + pad + 1,
        )
    return groups

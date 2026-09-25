"""Availability diagnostics for reacquisition, separate from identity accuracy."""

from __future__ import annotations

import numpy as np


def availability_episodes(xyz: np.ndarray, frames: np.ndarray) -> dict:
    """Report missing episodes without calling an observed return the same animal.

    Missing video frames split episodes; they are not silently filled or counted
    as recovered. Leading/trailing episodes are distinct from internally bounded
    gaps. Finite coordinates may still belong to the wrong animal.
    """
    points, frames = np.asarray(xyz), np.asarray(frames)
    if points.shape != (len(frames), 3, 3) or frames.ndim != 1:
        raise ValueError("expected frame indices and (frames, three slots, xyz) coordinates")
    if len(frames) and np.any(np.diff(frames) <= 0):
        raise ValueError("frame indices must be strictly increasing")
    finite = np.isfinite(points).all(axis=-1)
    slots = []
    for slot in range(3):
        episodes = []
        i = 0
        while i < len(frames):
            if finite[i, slot]:
                i += 1
                continue
            first = i
            while (i + 1 < len(frames) and not finite[i + 1, slot]
                   and frames[i + 1] == frames[i] + 1):
                i += 1
            last = i
            before = (first > 0 and frames[first - 1] == frames[first] - 1
                      and finite[first - 1, slot])
            after = (last + 1 < len(frames) and frames[last + 1] == frames[last] + 1
                     and finite[last + 1, slot])
            episodes.append({
                "start": int(frames[first]), "stop": int(frames[last] + 1),
                "frames": last - first + 1,
                "preceded_by_observation": bool(before),
                "followed_by_observation": bool(after),
                "internally_bounded": bool(before and after),
            })
            i += 1
        slots.append({
            "slot": slot, "available_frames": int(finite[:, slot].sum()),
            "total_frames": len(frames),
            "available_fraction": float(finite[:, slot].mean()) if len(frames) else None,
            "missing_episodes": episodes,
            "maximum_missing_frames": max((e["frames"] for e in episodes), default=0),
            "internally_bounded_gaps": sum(e["internally_bounded"] for e in episodes),
            "missing_at_end_frames": (episodes[-1]["frames"] if len(frames)
                                      and not finite[-1, slot] else 0),
        })
    return {
        "slots": slots,
        "all_three_available_frames": int(finite.all(axis=1).sum()),
        "all_three_available_fraction": float(finite.all(axis=1).mean()) if len(frames) else None,
        "scope": "availability only; an observed return does not certify identity recovery",
    }


def assess_availability(accepted_xyz: np.ndarray, frames: np.ndarray,
                        overall_threshold: float = .9,
                        per_slot_threshold: float = .8) -> dict:
    """Apply fixed engineering coverage gates; never turn these into an ID pass."""
    episodes = availability_episodes(accepted_xyz, frames)
    available = np.isfinite(accepted_xyz).all(axis=-1)
    overall = float(available.mean()) if available.size else None
    by_slot = [item["available_fraction"] for item in episodes["slots"]]
    passed = (overall is not None and overall >= overall_threshold
              and all(value is not None and value >= per_slot_threshold for value in by_slot))
    return {
        "overall_threshold": overall_threshold, "per_slot_threshold": per_slot_threshold,
        "overall_accepted_fraction": overall, "per_slot_accepted_fraction": by_slot,
        "availability_gate_passed": bool(passed),
        "all_three_identity_verdict": (
            "unverified: independent three-animal identity GT unavailable"),
        "scope": "engineering availability gate, not identity precision or swap-free validation",
    }


def nominal_tank_summary(pose_xyz: np.ndarray, observed: np.ndarray) -> dict:
    """Expose observations admitted outside the original nominal floor/world gauge."""
    pose, observed = np.asarray(pose_xyz, dtype=float), np.asarray(observed, dtype=bool)
    if pose.ndim != 4 or pose.shape[1:] != (3, 3, 3) or observed.shape != pose.shape[:-1]:
        raise ValueError("expected three-fish three-node poses and corresponding observation mask")
    finite = np.isfinite(pose).all(axis=-1)
    if np.any(observed & ~finite):
        raise ValueError("observed nodes must have finite coordinates")
    outside = finite & ((pose < [0, 0, 0]).any(axis=-1)
                        | (pose > [192, 192, 186]).any(axis=-1))
    below = finite & (pose[..., 2] < 0)
    observed_z = pose[..., 2][observed]
    latent_z = pose[..., 2][finite]
    return {
        "nominal_lower_mm": [0, 0, 0], "nominal_upper_mm": [192, 192, 186],
        "observed_nodes_outside_nominal": int((outside & observed).sum()),
        "observed_nodes_below_nominal_floor": int((below & observed).sum()),
        "observed_outside_by_slot": (outside & observed).sum(axis=(0, 2)).tolist(),
        "latent_nodes_outside_nominal": int(outside.sum()),
        "minimum_observed_z_mm": float(observed_z.min()) if observed_z.size else None,
        "minimum_latent_z_mm": float(latent_z.min()) if latent_z.size else None,
        "scope": "world-gauge sensitivity diagnostic; not a surveyed physical floor or ID score",
    }

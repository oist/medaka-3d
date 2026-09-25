"""Recompute chosen last-direct-observation links independently of the search graph."""

import numpy as np
from identity_output import build_identity_output


def chosen_memory_links(geometry, identity_to_raw_slot, *, scale_mm=.8, weight=.2):
    identity = build_identity_output(geometry, identity_to_raw_slot)
    frames = identity["frame_idx"]
    observed = identity["proposed_observed"][:, :, 1]
    xyz = identity["proposed_xyz"]
    mapping = identity["identity_to_raw_slot"]
    if not observed[0].all():
        raise ValueError("the first frame must directly observe all three pectoral nodes")
    for name, value in (("scale_mm", scale_mm), ("weight", weight)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or (value <= 0 if name == "scale_mm" else value < 0)):
            raise ValueError(f"invalid {name}")
    n = len(frames)
    previous_frame = np.full((n, 3), -1, np.int64)
    previous_slot = np.full((n, 3), -1, np.int64)
    gap_frames = np.zeros((n, 3), np.int64)
    displacement = np.full((n, 3, 3), np.nan)
    cost = np.zeros((n, 3))
    applied = np.zeros((n, 3), bool)
    last_frame = frames[0] + np.zeros(3, np.int64)
    last_slot, last_xyz = mapping[0].copy(), xyz[0].copy()
    for t in range(1, n):
        previous_frame[t], previous_slot[t] = last_frame, last_slot
        held = observed[t]
        dt = frames[t] - last_frame[held]
        dx = xyz[t, held] - last_xyz[held]
        gap_frames[t, held], displacement[t, held] = dt, dx
        applied[t, held] = True
        cost[t, held] = weight / 3 * (
            np.sum(dx * dx, axis=-1) / (scale_mm * scale_mm * dt) + 3 * np.log(dt)
        )
        last_frame[held] = frames[t]
        last_slot[held], last_xyz[held] = mapping[t, held], xyz[t, held]
    if not np.isfinite(cost).all():
        raise ValueError("chosen direct-event link cost overflowed")
    return {
        "frame_idx": frames.copy(), "identity_to_raw_slot": mapping.copy(),
        "direct_observed": observed.copy(), "previous_direct_frame": previous_frame,
        "previous_direct_raw_slot": previous_slot, "gap_frames_when_reobserved": gap_frames,
        "displacement_mm_when_reobserved": displacement, "motion_cost_applied": applied,
        "brownian_motion_cost_by_id": cost,
        "terminal_last_direct_frame": last_frame.copy(),
        "terminal_last_direct_raw_slot": last_slot.copy(),
        "terminal_unobserved_age_frames": frames[-1] - last_frame,
        "terminal_unsettled_gap": last_frame != frames[-1],
    }

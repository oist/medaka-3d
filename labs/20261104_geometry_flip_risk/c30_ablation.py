"""A single-parameter C30 ablation followed by the unchanged C34 rule."""

import c30_slot_stitch as c30
import c34_microflip as c34
import numpy as np


class PermissiveVelocity:
    """Linker adapter around the existing implementation; no copied pipeline."""

    name = "c30_velocity_without_median_prerequisite"

    def __init__(self, n_frames, resolved, degraded, median_prerequisite=0.0):
        self.n_frames, self.resolved, self.degraded = n_frames, resolved, degraded
        self.median_prerequisite = median_prerequisite

    def permutation(self, xyz):
        old = c30.VELOCITY_NEEDS_MEDIAN_MM
        try:
            c30.VELOCITY_NEEDS_MEDIAN_MM = self.median_prerequisite
            perm, joins, refusals = c30.stitch(xyz, self.resolved, degraded=self.degraded)
        finally:
            c30.VELOCITY_NEEDS_MEDIAN_MM = old
        linked = np.take_along_axis(xyz, perm[:, :, None], axis=1)
        # C30 accepts partially finite frames; C34 requires all three positions.
        # Preserve the distinct frame policies of the original stage entrypoints.
        c34_resolved = self.resolved[np.isfinite(linked[self.resolved]).all(axis=(1, 2))]
        windows = c34.find_windows(linked, c34_resolved)
        result = c34.apply_windows({"permutation": perm}, windows)["permutation"]
        return result, {"joins": joins, "refusals": refusals, "microflip_windows": windows}

    def link(self, instances):
        xyz = np.full((self.n_frames, 3, 3), np.nan)
        for instance in instances:
            xyz[instance.frame, instance.slot] = instance.xyz
        perm, _ = self.permutation(xyz)
        return np.take_along_axis(xyz, perm[:, :, None], axis=1)

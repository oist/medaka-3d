"""One prospective color-registration prefix, labeled by current selected slots.

The caller must observe the tracker sequence's first frame and each following
native frame exactly once. This accumulator never reads saved poses/templates,
revises earlier labels from a later beam traceback, or clears identity review.
Slot labels are model hypotheses, not biological identity truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from appearance_diagnostic import (
    EROSION_RADIUS,
    FEATURES,
    MIN_CORE_PIXELS,
    assign_instances,
    describe_core,
)

from medaka3d.types import VIEWS


@dataclass(frozen=True)
class OnlineAppearanceConfig:
    """Actual arm uses 166/5; smaller prefixes/sample counts support CPU tests."""

    training_frames: int = 166
    minimum_template_samples: int = 5

    def __post_init__(self):
        for name in ("training_frames", "minimum_template_samples"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.training_frames > 166:
            raise ValueError("registration cannot extend beyond the first 166 native frames")
        if self.minimum_template_samples > self.training_frames:
            raise ValueError("minimum_template_samples cannot exceed training_frames")


class OnlineAppearanceAccumulator:
    """Collect fixed instance-core descriptors from current, unreviewed slots.

    ``sources`` is fish,view,node in canonical VIEWS order; ``observed`` is the
    strict global node support mask. Both mouth and pectoral must be directly
    supported globally and have sources in the registering view. All latent
    competitors remain in instance exclusion, even if missing or reviewed.

    Templates are unavailable before the complete prefix. At its end a view is
    enabled only if every slot has enough samples. Disabled views expose NaN
    templates for all slots. Calls after freezing check only frame continuity;
    they do not read image/pose/source values or change the registration snapshot.
    Every public array or nested snapshot is an independent copy.
    """

    def __init__(self, rig, config=None):
        if tuple(rig.views) != VIEWS:
            raise ValueError("source view order must match canonical VIEWS")
        self.rig = rig
        self.config = config or OnlineAppearanceConfig()
        if not isinstance(self.config, OnlineAppearanceConfig):
            raise ValueError("OnlineAppearanceConfig required")
        self._last_seen = None
        self._frames = []
        self._features = []
        self._reasons = []
        self._instance_rows = []
        self._core_pixels = []
        self._online_pose_xyz = []
        self._online_sources = []
        self._online_observed = []
        self._online_identity_review = []
        self._counts = np.zeros((3, 3), np.int64)
        self._templates = np.full((3, 3, 6), np.nan)
        self._enabled_views = np.zeros(3, bool)
        self._frozen = False

    @property
    def frozen(self):
        return self._frozen

    @property
    def training_frames_seen(self):
        return len(self._frames)

    @property
    def counts(self):
        return self._counts.copy()

    @property
    def templates(self):
        return self._templates.copy()

    @property
    def enabled_views(self):
        return self._enabled_views.copy()

    def snapshot(self):
        """Training-only provenance arrays; post-freeze calls leave this unchanged."""
        n = len(self._frames)
        return {
            "config": asdict(self.config),
            "views": list(VIEWS),
            "feature_names": list(FEATURES),
            "erosion_radius_px": EROSION_RADIUS,
            "minimum_core_pixels": MIN_CORE_PIXELS,
            "frozen": self.frozen,
            "training_frames_seen": n,
            "first_frame_idx": self._frames[0] if n else None,
            "last_training_frame_idx": self._frames[-1] if n else None,
            "frame_idx": np.array(self._frames, dtype=np.int64),
            "features": np.array(self._features, dtype=float).reshape(n, 3, 3, 6),
            "reasons": np.array(self._reasons, dtype="U64").reshape(n, 3, 3),
            "instance_rows": np.array(self._instance_rows, dtype=np.int64).reshape(n, 3, 3),
            "core_pixels": np.array(self._core_pixels, dtype=np.int64).reshape(n, 3, 3),
            "online_pose_xyz": np.array(self._online_pose_xyz, dtype=np.float64).reshape(
                n, 3, 3, 3
            ),
            "online_sources": np.array(self._online_sources, dtype=np.int64).reshape(n, 3, 3, 3),
            "online_observed": np.array(self._online_observed, dtype=bool).reshape(n, 3, 3),
            "online_identity_review": np.array(self._online_identity_review, dtype=bool).reshape(
                n, 3
            ),
            "counts": self.counts,
            "templates": self.templates,
            "enabled_views": self.enabled_views,
            "scope": "Current selected model slots in the initial native prefix; not identity GT",
            "external_templates_or_saved_poses_used": False,
            "reviewed_slots_registered": False,
            "retroactive_beam_relabeling": False,
        }

    def observe(self, frame, best_pose, sources, observed, review):
        frame_idx = frame.frame_idx
        if (
            isinstance(frame_idx, (bool, np.bool_))
            or not isinstance(frame_idx, (int, np.integer)) or frame_idx < 0
        ):
            raise ValueError("frame_idx must be a nonnegative native integer")
        frame_idx = int(frame_idx)
        if self._last_seen is not None and frame_idx != self._last_seen + 1:
            raise ValueError("observe requires each continuous native frame exactly once")
        if self.frozen:
            self._last_seen = frame_idx
            return
        pose = np.asarray(best_pose)
        sources = np.asarray(sources)
        observed, review = np.asarray(observed), np.asarray(review)
        if pose.shape != (3, 3, 3) or not np.isfinite(pose).all():
            raise ValueError("best_pose must be finite fish,node,xyz with shape 3,3,3")
        if (
            sources.shape != (3, 3, 3) or not np.issubdtype(sources.dtype, np.integer)
            or np.any(sources < -1)
        ):
            raise ValueError("sources must be integer fish,view,node with values >= -1")
        if observed.shape != (3, 3) or observed.dtype != bool:
            raise ValueError("observed must be a boolean fish,node array")
        if review.shape != (3,) or review.dtype != bool:
            raise ValueError("review must be a boolean fish array")
        if np.any(observed & ((sources >= 0).sum(axis=1) < 2)):
            raise ValueError("direct observed nodes require sources from at least two views")
        features = np.full((3, 3, 6), np.nan)
        reasons = np.full((3, 3), "view_unavailable", dtype="U64")
        rows = np.full((3, 3), -1, np.int64)
        core_pixels = np.zeros((3, 3), np.int64)
        for vi, view in enumerate(VIEWS):
            masks = frame.masks.get(view)
            image = frame.images_bgr.get(view)
            if image is None:
                reasons[vi] = "image_unavailable"
            elif masks is not None:
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError("registration requires native uint8 BGR images")
                if masks.dtype != bool or masks.ndim != 3 or masks.shape[1:] != image.shape[:2]:
                    raise ValueError("registration requires full-image boolean instance masks")
                projected = np.asarray(self.rig.camera(view).project(pose[:, :2]))
                if projected.shape != (3, 2, 2):
                    raise ValueError("camera must project mouth/pectoral to fish,node,xy")
                direct = observed[:, :2].all(-1) & (sources[:, vi, :2] >= 0).all(-1)
                rows[vi], reasons[vi] = assign_instances(masks, projected, direct & ~review)
                for fish, row in enumerate(rows[vi]):
                    if row < 0:
                        continue
                    descriptor, core = describe_core(image, masks[row])
                    core_pixels[vi, fish] = int(core.sum())
                    if descriptor is None:
                        reasons[vi, fish] = "core_too_small"
                    else:
                        features[vi, fish] = descriptor
            # Review exclusion takes precedence in the registration denominator,
            # while the reviewed pose still participated in competitor exclusion.
            reasons[vi, review] = "identity_review"
        # Commit atomically after validating every supplied view.
        self._frames.append(frame_idx)
        self._features.append(features.copy())
        self._reasons.append(reasons.copy())
        self._instance_rows.append(rows.copy())
        self._core_pixels.append(core_pixels.copy())
        # Preserve the exact values supplied by the current online selected state;
        # later beam traceback and caller-owned buffer reuse cannot relabel them.
        self._online_pose_xyz.append(pose.copy())
        self._online_sources.append(sources.copy())
        self._online_observed.append(observed.copy())
        self._online_identity_review.append(review.copy())
        self._counts += np.isfinite(features).all(-1)
        self._last_seen = frame_idx
        if len(self._frames) == self.config.training_frames:
            self._enabled_views = np.all(
                self._counts >= self.config.minimum_template_samples, axis=1
            )
            all_features = np.asarray(self._features)
            for vi in np.flatnonzero(self._enabled_views):
                for fish in range(3):
                    values = all_features[:, vi, fish]
                    values = values[np.isfinite(values).all(-1)]
                    self._templates[vi, fish] = np.median(values, axis=0)
            self._frozen = True

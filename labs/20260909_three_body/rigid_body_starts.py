"""Place an unchanged three-node body into image-derived volume hypotheses.

This is a finite set of optimizer starts, not landmark reconstruction or identity
assignment. Every original scene is retained; trial bodies outside the registered
box are reported and excluded without clipping. A volume centroid is aligned to
an engineering Gaussian-volume centroid, never declared to be a pectoral node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RigidStartConfig:
    rolls: int = 4
    lower_mm: tuple = (0., 0., -6.)
    upper_mm: tuple = (192., 192., 186.)
    random_translation_sigma_mm: float = 12.

    def __post_init__(self):
        if type(self.rolls) is not int or self.rolls != 4:
            raise ValueError("this diagnostic fixes four rolls and two head-tail directions")
        lo, hi = np.asarray(self.lower_mm), np.asarray(self.upper_mm)
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all():
            raise ValueError("bounds must be finite XYZ triples")
        if np.any(lo >= hi):
            raise ValueError("lower bounds must precede upper bounds")
        if (not np.isfinite(self.random_translation_sigma_mm)
                or self.random_translation_sigma_mm <= 0):
            raise ValueError("random control translation scale must be positive")


def _body(value):
    body = np.asarray(value, dtype=np.float64)
    if body.shape != (3, 3) or not np.isfinite(body).all():
        raise ValueError("a body must contain three finite XYZ nodes")
    lengths = np.linalg.norm(np.diff(body, axis=0), axis=1)
    if np.any(lengths <= 0.05):
        raise ValueError("both original segments must exceed the renderer sigma floor")
    return body, lengths


def gaussian_volume_center(value):
    """Centroid of the existing eight Gaussian elements before occlusion.

Weights are sigma_axial * sigma_radial**2. Common opacity, width, total length
and Gaussian-integral constants cancel. This deliberately uses the fixed two
head / six tail elements and taper of RendererConfig's eight-element default.
It is an approximation to a body's volume, not an observed anatomical point.
"""
    body, lengths = _body(value)
    count = np.array([2, 6])
    t = np.r_[(np.arange(2) + .5) / 2, (np.arange(6) + .5) / 6]
    segment = np.r_[np.zeros(2, dtype=int), np.ones(6, dtype=int)]
    centers = body[:-1][segment] + np.diff(body, axis=0)[segment] * t[:, None]
    taper = np.r_[.45 + .55 * t[:2], 1. - .85 * t[2:] ** 1.2]
    # Keep the frozen renderer's sigma floor, including its default width .06.
    axial = np.maximum((lengths / count * .8)[segment], .05)
    radial = np.maximum(lengths.sum() * .06 * taper, .05)
    weight = axial * radial**2
    return np.sum(weight[:, None] * centers, axis=0) / weight.sum()


def body_frame(value):
    """Right-handed columns: mouth-to-tail axis, bend direction, perpendicular.

For an exactly straight body a deterministic world-axis fallback defines roll;
the complete four-roll set remains explicit. A coincident mouth and tail has
no usable longitudinal orientation and is rejected as a degenerate template.
"""
    body, _ = _body(value)
    axis = body[2] - body[0]
    norm = np.linalg.norm(axis)
    if norm <= 1e-10:
        raise ValueError("coincident mouth and tail have no longitudinal axis")
    axis /= norm
    bend = body[1] - body[0]
    bend -= np.dot(bend, axis) * axis
    if np.linalg.norm(bend) <= 1e-10:
        bend = np.eye(3)[np.argmin(np.abs(axis))]
        bend -= np.dot(bend, axis) * axis
    bend /= np.linalg.norm(bend)
    return np.column_stack((axis, bend, np.cross(axis, bend)))


def oriented_bodies(template, centroid, axes, config=None):
    """Eight rigid placements, with PCA axes supplied as orthonormal ROWS.

Returns bodies, rotations, and [longitudinal sign, quarter-roll] descriptors.
All pairwise landmark distances and chirality are preserved. No scale, bend,
width or anatomical-node inference is introduced by this placement operation.
"""
    cfg = config or RigidStartConfig()
    body, _ = _body(template)
    center, axes = np.asarray(centroid, float), np.asarray(axes, float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("volume centroid must be a finite XYZ point")
    if axes.shape != (3, 3) or not np.isfinite(axes).all():
        raise ValueError("PCA axes must be finite orthonormal rows")
    if not np.allclose(axes @ axes.T, np.eye(3), rtol=0, atol=1e-10):
        raise ValueError("PCA axes must be orthonormal rows")
    source = body_frame(body)
    relative = body - gaussian_volume_center(body)
    poses, rotations, metadata = [], [], []
    for sign in (1, -1):
        first, second = sign * axes[0], axes[1]
        third = np.cross(first, second)
        for roll in range(cfg.rolls):
            angle = 2 * np.pi * roll / cfg.rolls
            lateral = np.cos(angle) * second + np.sin(angle) * third
            target = np.column_stack((first, lateral, np.cross(first, lateral)))
            rotation = target @ source.T
            poses.append(relative @ rotation.T + center)
            rotations.append(rotation)
            metadata.append((sign, roll))
    return np.stack(poses), np.stack(rotations), np.asarray(metadata, np.int64)


def volume_scene_starts(scene, components, config=None):
    """Offer every volume to each raw body, including directly supported bodies.

``components`` is a sequence of mappings with ``centroid`` and ``axes``. Order
is retained in explicit component indices. Only one body changes in each start;
joint replacement of multiple missing or duplicate bodies is outside this arm.
All generated trials are returned with an admission mask and outside-node flags.
"""
    cfg = config or RigidStartConfig()
    scene = np.asarray(scene, float)
    if scene.shape != (3, 3, 3) or not np.isfinite(scene).all():
        raise ValueError("scene must contain three finite three-node bodies")
    if len(components) > 12:
        raise ValueError("the registered volume pool contains at most twelve components")
    for body in scene:
        body_frame(body)
    lo, hi = np.asarray(cfg.lower_mm), np.asarray(cfg.upper_mm)
    if np.any((scene < lo) | (scene > hi)):
        raise ValueError("preserved original scene lies outside the registered box")
    poses = [scene.copy()]
    labels = [(-1, -1, 0, 0)]
    for c, component in enumerate(components):
        for slot in range(3):
            bodies, _, orientation = oriented_bodies(
                scene[slot], component["centroid"], component["axes"], cfg
            )
            for body, (sign, roll) in zip(bodies, orientation, strict=True):
                candidate = scene.copy()
                candidate[slot] = body
                poses.append(candidate)
                labels.append((c, slot, sign, roll))
    pose = np.stack(poses)
    outside = (pose < lo) | (pose > hi)
    return dict(pose_xyz=pose, component_slot_sign_roll=np.asarray(labels, np.int64),
                outside_coordinates=outside, admissible=~outside.any(axis=(1, 2, 3)))


def random_scene_starts(scene, slot_sequence, *, seed, config=None):
    """Same number of raw-slot trials using fixed-scale random rigid starts.

No mask, component centre or inferred ID is consulted. Unbounded Gaussian
translations have sigma 12 mm; rotations use normalized isotropic Gaussian
quaternions. Rejected trials are retained and are NOT resampled to improve the
control. The original scene appears first in both arms.
"""
    cfg = config or RigidStartConfig()
    scene = np.asarray(scene, float)
    volume_scene_starts(scene, [], cfg)
    slots = np.asarray(slot_sequence)
    if slots.ndim == 1 and not len(slots):
        slots = slots.astype(np.int64)
    if (slots.ndim != 1 or not np.issubdtype(slots.dtype, np.integer)
            or np.any((slots < 0) | (slots > 2))):
        raise ValueError("trial slots must be an integer vector in 0..2")
    if type(seed) is not int or seed < 0:
        raise ValueError("control seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    poses = [scene.copy()]
    centers = np.stack([gaussian_volume_center(body) for body in scene])
    for slot in slots:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        rotation = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])
        center = centers[slot] + rng.normal(0, cfg.random_translation_sigma_mm, 3)
        candidate = scene.copy()
        candidate[slot] = (scene[slot] - centers[slot]) @ rotation.T + center
        poses.append(candidate)
    pose = np.stack(poses)
    outside = (pose < cfg.lower_mm) | (pose > cfg.upper_mm)
    return dict(pose_xyz=pose, raw_slot=np.r_[-1, slots], outside_coordinates=outside,
                admissible=~outside.any(axis=(1, 2, 3)), seed=seed)

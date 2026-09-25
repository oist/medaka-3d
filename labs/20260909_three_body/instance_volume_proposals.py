"""Image-only component hypotheses on an existing, fixed 2-mm refractive grid.

Top detector rows are independent proposals, not fish or identity labels. Their
volumes may overlap, split one fish, merge fish, or describe reflections/ghosts.
Centroids summarize occupied cells; they are not anatomical pectoral landmarks.
No SLEAP observations, previous poses, identity, rig, or optimizer is accepted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from mask_grid import bilinear_sample
from scipy import ndimage

from medaka3d.types import VIEWS


def _readonly(value):
    value = np.array(value, copy=True)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class InstanceVolumeConfig:
    """Fixed engineering settings; omission is reported, never silently accepted.

    The default retains even a one-cell component, which is explicitly PCA
    degenerate. Raising the minimum can discard thin fish; six-neighbour
    connectivity on a 2-mm grid can already fragment them at the default.
    """

    spacing_mm: float = 2.0
    support_threshold: float = 0.5
    min_component_voxels: int = 1
    max_components_per_top: int = 3
    max_total_components: int = 12
    pca_degeneracy_rtol: float = 1e-8

    def __post_init__(self):
        if isinstance(self.spacing_mm, bool) or self.spacing_mm != 2.0:
            raise ValueError("this component proposal uses the existing 2-mm grid only")
        if (isinstance(self.support_threshold, bool)
                or not np.isfinite(self.support_threshold)
                or not 0 < self.support_threshold <= 1):
            raise ValueError("support_threshold must be in (0,1]")
        for key, maximum in (("max_components_per_top", 3), ("max_total_components", 12)):
            if type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= maximum:
                raise ValueError(f"{key} must be a positive integer no greater than {maximum}")
        if type(self.min_component_voxels) is not int or self.min_component_voxels < 1:
            raise ValueError("min_component_voxels must be a positive integer")
        if (isinstance(self.pca_degeneracy_rtol, bool)
                or not np.isfinite(self.pca_degeneracy_rtol)
                or not 0 < self.pca_degeneracy_rtol < 1):
            raise ValueError("pca_degeneracy_rtol must be in (0,1)")


@dataclass(frozen=True)
class VolumeComponent:
    """One retained connected support region, with no anatomical or ID claim.

    ``pca_axes`` has shape (3,3): ROW i is the world-coordinate direction of
    eigenvalue i, ordered largest first. Axis signs are canonicalized, but the
    axes remain unoriented: PCA does not determine which end is a mouth.
    ``pca_extent_mm`` includes the full 2-mm cells along each PCA axis;
    eigenvalues describe the uniformly weighted cell CENTRES. Repeated or
    near-zero eigenvalues are marked explicitly. Their numerical axes must not
    be interpreted as a reliably measured orientation.
    """

    top_row: int
    component_label: int
    top_component_rank: int  # 1-based rank before the global round-robin cap
    voxel_indices: np.ndarray  # indices in the supplied fixed grid
    voxel_count: int
    volume_mm3: float
    centroid_xyz: np.ndarray  # 3, equally weighted cell centres
    pca_axes: np.ndarray  # 3,3 ROWS are world-coordinate axes
    pca_eigenvalues_mm2: np.ndarray  # 3 descending
    pca_extent_mm: np.ndarray  # 3, full-cell extent along each row axis
    axis_aligned_extent_mm: np.ndarray  # 3, full-cell world XYZ extent
    pca_rank: int
    pca_eigenvalue_tolerance_mm2: float
    pca_degenerate: bool
    principal_axis_defined: bool
    two_view_voxels: int  # exactly two supporting views
    three_view_voxels: int
    known_third_contradiction_voxels: int
    unknown_third_view_voxels: int
    view_supported_voxels: np.ndarray  # top/side1/side2
    view_known_voxels: np.ndarray
    mean_view_support: np.ndarray  # NaN only when that view is entirely unknown


@dataclass(frozen=True)
class InstanceVolumePool:
    components: tuple[VolumeComponent, ...]
    top_diagnostics: tuple[dict, ...]  # includes every original top row
    omitted: tuple[dict, ...]  # every detected but omitted 3D component
    view_available: np.ndarray
    view_has_foreground: np.ndarray
    diagnostics: dict


class InstanceVolumeProposer:
    """Reuse supplied native projections; never create or reproject a grid.

    ``xyz`` must be the C-flattened ``indexing='ij'`` regular XYZ grid with
    shape ``grid_shape`` and 2-mm spacing. ``projected_uv`` is N,3,2 in VIEWS
    order. Pixel (0,0) is the centre of the first native pixel. No resize,
    nearest-neighbour rounding, morphology or smoothing is performed.

    None and projections outside an image are unknown. An explicitly supplied
    empty/zero bool detector array is known background. This is a reachability
    diagnostic, separate from MaskGrid's empty-view-neutral likelihood policy.
    """

    def __init__(self, xyz, projected_uv, grid_shape, *, image_shape_hw=(1200, 1200), config=None):
        self.config = config or InstanceVolumeConfig()
        if not isinstance(self.config, InstanceVolumeConfig):
            raise TypeError("config must be InstanceVolumeConfig")
        shape = tuple(grid_shape)
        if len(shape) != 3 or any(type(v) is not int or v < 1 for v in shape):
            raise ValueError("grid_shape must contain three positive integers")
        image_shape = tuple(image_shape_hw)
        if len(image_shape) != 2 or any(type(v) is not int or v < 1 for v in image_shape):
            raise ValueError("image_shape_hw must contain two positive native dimensions")
        xyz, uv = np.asarray(xyz, dtype=float), np.asarray(projected_uv, dtype=float)
        if xyz.shape != (int(np.prod(shape)), 3) or not np.isfinite(xyz).all():
            raise ValueError("xyz must contain every finite fixed-grid cell in grid_shape order")
        if uv.shape != (len(xyz), 3, 2):
            raise ValueError("projected_uv must be N,3,2 in top/side1/side2 order")
        lattice = xyz.reshape(*shape, 3)
        for axis in range(3):
            expected = np.eye(3)[axis] * self.config.spacing_mm
            if not np.allclose(np.diff(lattice, axis=axis), expected, atol=1e-9, rtol=0):
                raise ValueError("xyz is not the supplied axis-aligned regular 2-mm XYZ grid")
        self.xyz, self.projected_uv = _readonly(xyz), _readonly(uv)
        self.grid_shape, self.image_shape_hw = shape, image_shape

    def _masks(self, masks_by_view):
        if not isinstance(masks_by_view, Mapping) or set(masks_by_view) != set(VIEWS):
            raise ValueError("provide exactly all three views, using None for missing views")
        masks, available, foreground = [], [], []
        for view in VIEWS:
            array = masks_by_view[view]
            available.append(array is not None)
            if array is None:
                masks.append(None)
                foreground.append(False)
                continue
            array = np.asarray(array)
            if (array.dtype != bool or array.ndim != 3
                    or array.shape[1:] != self.image_shape_hw):
                raise ValueError("D2 masks must be bool instances,H,W in declared native pixels")
            masks.append(array)
            foreground.append(bool(array.any()))
        return masks, np.asarray(available, dtype=bool), np.asarray(foreground, dtype=bool)

    def _component(self, row, label, rank, indices, values):
        points = self.xyz[indices]
        centroid = points.mean(axis=0)
        centered = points - centroid
        covariance = centered.T @ centered / len(points)
        eigenvalues, vectors = np.linalg.eigh(covariance)
        eigenvalues, axes = np.maximum(eigenvalues[::-1], 0), vectors[:, ::-1].T.copy()
        for axis in range(2):
            pivot = np.argmax(np.abs(axes[axis]))
            if axes[axis, pivot] < 0:
                axes[axis] *= -1
        axes[2] = np.cross(axes[0], axes[1])
        tolerance = self.config.pca_degeneracy_rtol * max(
            float(eigenvalues[0]), self.config.spacing_mm ** 2
        )
        rank_pca = int(np.count_nonzero(eigenvalues > tolerance))
        repeated = bool(np.any(np.abs(np.diff(eigenvalues)) <= tolerance))
        defined = bool(eigenvalues[0] > tolerance and eigenvalues[0] - eigenvalues[1] > tolerance)
        known, supported = np.isfinite(values), values >= self.config.support_threshold
        known_count, support_count = known.sum(1), supported.sum(1)
        mean = np.divide(np.nansum(values, axis=0), known.sum(0),
                         out=np.full(3, np.nan), where=known.sum(0) > 0)
        extent = np.ptp(centered @ axes.T, axis=0) + self.config.spacing_mm * np.abs(axes).sum(1)
        return VolumeComponent(
            row, label, rank, _readonly(indices), len(points),
            float(len(points) * self.config.spacing_mm ** 3), _readonly(centroid),
            _readonly(axes), _readonly(eigenvalues), _readonly(extent),
            _readonly(np.ptp(points, axis=0) + self.config.spacing_mm), rank_pca, tolerance,
            rank_pca < 3 or repeated, defined,
            int((support_count == 2).sum()), int((support_count == 3).sum()),
            int(((support_count == 2) & (known_count == 3)).sum()),
            int(((support_count == 2) & (known_count == 2)).sum()),
            _readonly(supported.sum(0)), _readonly(known.sum(0)), _readonly(mean),
        )

    def evaluate(self, masks_by_view) -> InstanceVolumePool:
        masks, available, foreground = self._masks(masks_by_view)
        side_values = np.full((len(self.xyz), 2), np.nan)
        for side in (1, 2):
            if masks[side] is not None:
                side_values[:, side - 1] = bilinear_sample(
                    masks[side].any(axis=0), self.projected_uv[:, side]
                )
        side_supported = side_values >= self.config.support_threshold
        side_any = side_supported.any(axis=1)
        eligible_by_row, top_diagnostics, omitted = [], [], []
        total_components = 0
        structure = ndimage.generate_binary_structure(3, 1)
        top_masks = masks[0] if masks[0] is not None else []
        for row, mask in enumerate(top_masks):
            top_values = bilinear_sample(mask, self.projected_uv[:, 0])
            top_supported = top_values >= self.config.support_threshold
            eligible = top_supported & side_any
            labels, count = ndimage.label(eligible.reshape(self.grid_shape), structure=structure)
            total_components += count
            flat = labels.ravel()
            sizes = np.bincount(flat, minlength=count + 1)
            strict = np.bincount(flat, weights=side_supported.all(1), minlength=count + 1)
            contradiction = np.bincount(
                flat, weights=(side_supported.sum(1) == 1) & np.isfinite(side_values).all(1),
                minlength=count + 1,
            )
            ranks = sorted(range(1, count + 1),
                           key=lambda label: (-sizes[label], -strict[label],
                                              contradiction[label], label))
            retained_here, row_omitted = [], []
            locations = ndimage.find_objects(labels, max_label=count)
            for rank, label in enumerate(ranks, start=1):
                reason = None
                if sizes[label] < self.config.min_component_voxels:
                    reason = "min_component_voxels"
                elif len(retained_here) >= self.config.max_components_per_top:
                    reason = "per_top_component_limit"
                if reason is not None:
                    row_omitted.append(dict(top_row=row, component_label=int(label),
                                            voxel_count=int(sizes[label]), reason=reason))
                    continue
                slices = locations[label - 1]
                local = np.nonzero(labels[slices] == label)
                indices = np.ravel_multi_index(
                    tuple(local[a] + slices[a].start for a in range(3)), self.grid_shape
                ).astype(np.int64)
                values = np.column_stack((top_values[indices], side_values[indices]))
                retained_here.append(self._component(row, int(label), rank, indices, values))
            status = "component_candidates"
            if not mask.any():
                status = "empty_top_instance"
            elif not top_supported.any():
                status = "top_mask_has_no_fixed_grid_support"
            elif not eligible.any():
                status = "no_top_plus_side_support"
            elif not retained_here:
                status = "all_components_below_minimum"
            top_diagnostics.append(dict(
                top_row=row, status=status, top_supported_voxels=int(top_supported.sum()),
                eligible_voxels=int(eligible.sum()), component_count=int(count),
                before_global_cap=len(retained_here), retained_components=0,
                omitted_components=len(row_omitted),
            ))
            digest = hashlib.sha256(np.packbits(mask, bitorder="little").tobytes()).hexdigest()
            eligible_by_row.append((digest, retained_here))
            omitted.extend(row_omitted)
        # One component per row per round, including small rows before another
        # row's second component. Content-derived ordering is independent of row IDs.
        selected = []
        ordered = sorted(eligible_by_row, key=lambda value: value[0])
        for rank in range(self.config.max_components_per_top):
            for _digest, candidates in ordered:
                if rank >= len(candidates):
                    continue
                component = candidates[rank]
                row = top_diagnostics[component.top_row]
                if len(selected) < self.config.max_total_components:
                    selected.append(component)
                    row["retained_components"] += 1
                else:
                    omitted.append(dict(
                        top_row=component.top_row, component_label=component.component_label,
                        voxel_count=component.voxel_count, reason="global_component_limit",
                    ))
                    row["omitted_components"] += 1
        diagnostics = dict(
            spacing_mm=self.config.spacing_mm, connectivity_neighbors=6,
            support_threshold=self.config.support_threshold,
            min_component_voxels=self.config.min_component_voxels,
            max_components_per_top=self.config.max_components_per_top,
            max_total_components=self.config.max_total_components,
            top_view_missing=masks[0] is None, top_instance_count=len(top_masks),
            total_components=int(total_components), retained_components=len(selected),
            omitted_components=len(omitted),
            omitted_by_reason={reason: sum(item["reason"] == reason for item in omitted)
                               for reason in ("min_component_voxels", "per_top_component_limit",
                                              "global_component_limit")},
            pca_axes_convention="rows, descending eigenvalue; unsigned directions",
            pca_degenerate_components=sum(c.pca_degenerate for c in selected),
            principal_axis_undefined_components=sum(not c.principal_axis_defined for c in selected),
            empty_available_view_policy="known background for support diagnostics",
            unknown_view_policy="None or nonfinite/outside native projection; never background",
            potential_thin_body_fragmentation=True,
            six_neighbor_two_mm_discretization_is_not_anatomical_connectivity=True,
            components_are_fish=False, centroids_are_pectoral_nodes=False,
            overlapping_components_and_duplicate_top_masks_are_not_deduplicated=True,
        )
        if len(selected) + len(omitted) != total_components:
            raise RuntimeError("component accounting did not preserve every omission")
        return InstanceVolumePool(tuple(selected), tuple(top_diagnostics), tuple(omitted),
                                  _readonly(available), _readonly(foreground), diagnostics)

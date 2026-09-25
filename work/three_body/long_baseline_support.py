"""Verified C34 label inputs for a separate common support classification.

No fitting, correspondence search, ID repair, detector or GT input. Saved
pectoral coordinates remain authoritative. Mouth/tail triangulations are
derived diagnostics, not the old method's saved three-node output. In
particular, finite_two_ray_observed is NOT a 2 mm / 12 pixel support verdict.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from medaka3d.calib import load_rig, triangulate
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
FRAME_COUNT = 348600
CATALOG_SHA256 = "7bcfcd453ad345d7a2ca6269554f0aae6f158c430f8ae8dd223382549325181f"
BASELINE = f"work/c34_microflip/{EXPERIMENT}__chain-c28-c33-c30v4-c34"
CALIBRATION = ("calibration/fixed_refractive_192x192x186/"
               "phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration")
NODES = ("mouth", "pectoral", "tail")
LAMBDAS = ("0", "0.1", "0.25", "0.5", "0.767", "2", "8")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(value):
    require(isinstance(value, str), "relative path must be text")
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    require(bool(path.parts) and not path.is_absolute() and ".." not in path.parts
            and ":" not in value, "unsafe registered relative path")
    return path.as_posix()


def native_frames(start, stop):
    require(type(start) is int and type(stop) is int
            and 0 <= start < stop <= FRAME_COUNT, "native range outside baseline recording")
    return np.arange(start, stop, dtype=np.int64)


class RegisteredNodes:
    """Same flat within-frame row order as the registered untracked SLEAP cache."""

    def __init__(self, paths, frames):
        self.frames, self.paths, self.data = frames.copy(), dict(paths), {}
        for view in VIEWS:
            with np.load(paths[view], allow_pickle=False) as data:
                require(str(data["experiment"]) == EXPERIMENT and str(data["view"]) == view
                        and tuple(data["nodes"]) == NODES, "wrong SLEAP experiment/view/nodes")
                index, points = data["frame_idx"], data["points"]
                require(index.ndim == 1 and index.dtype.kind in "iu"
                        and np.all(index < FRAME_COUNT) and np.all(index >= 0)
                        and np.all(index[1:] >= index[:-1]), "invalid SLEAP native index")
                require(points.shape == (len(index), 3, 2) and points.dtype.kind == "f"
                        and not np.isinf(points).any(), "invalid SLEAP point schema")
                lo, hi = np.searchsorted(index, [frames[0], frames[-1]+1])
                selected = index[lo:hi].astype(np.int64) - frames[0]
                counts = np.bincount(selected, minlength=len(frames))
                self.data[view] = (points[lo:hi].copy(), np.r_[0, np.cumsum(counts)])

    def at(self, view, frame):
        require(type(frame) is int and self.frames[0] <= frame <= self.frames[-1],
                "node request outside native interval")
        values, offsets = self.data[view]
        t = frame - int(self.frames[0])
        return values[offsets[t]:offsets[t+1]]


def reconstruct_sources(frames, source_rows, saved_xyz, nodes, rig, *, block_frames=1660):
    """Expand only existing instance rows; missing or degenerate nodes remain NaN.

    `sources` is an UNFILTERED candidate assignment for support classification,
    never a declaration that every stored row passes the common strict gates.
    All axes are T, fish, view, node (then xy) for sources/pixels, and T, fish,
    node, xyz for poses. No source row is inferred from coordinate proximity.
    """
    frames, source_rows, saved_xyz = map(np.asarray, (frames, source_rows, saved_xyz))
    require(frames.ndim == 1 and len(frames) > 0 and frames.dtype.kind in "iu"
            and frames.min() >= 0 and frames.max() < FRAME_COUNT
            and np.all(np.diff(frames.astype(np.int64)) == 1), "invalid contiguous native frames")
    require(source_rows.shape == (len(frames), 3, 3) and source_rows.dtype.kind == "i"
            and np.all(source_rows >= -1), "invalid original instance source rows")
    require(saved_xyz.shape == (len(frames), 3, 3) and saved_xyz.dtype.kind == "f"
            and not np.isinf(saved_xyz).any()
            and np.all(np.isfinite(saved_xyz).any(-1) == np.isfinite(saved_xyz).all(-1)),
            "saved pectorals must be whole finite or NaN points")
    require(tuple(rig.views) == tuple(VIEWS), "wrong calibration view order")
    require(type(block_frames) is int and block_frames > 0, "positive block_frames required")
    expanded = np.repeat(source_rows[..., None], 3, axis=-1)
    pose = np.full((len(frames), 3, 3, 3), np.nan)
    finite_rays = np.zeros(expanded.shape, dtype=bool)
    for start in range(0, len(frames), block_frames):
        stop = min(start + block_frames, len(frames))
        origins, directions = [], []
        for vi, view in enumerate(VIEWS):
            pixels = np.full((stop-start, 3, 3, 2), np.nan)
            for k, frame in enumerate(frames[start:stop]):
                points = np.asarray(nodes.at(view, int(frame)))
                require(points.ndim == 3 and points.shape[1:] == (3, 2),
                        "wrong per-frame node schema")
                rows = source_rows[start+k, :, vi]
                present = rows >= 0
                require(np.all(rows[present] < len(points)), "baseline source row out of range")
                require(len(np.unique(rows[present])) == int(present.sum()),
                        "baseline assigns one instance row to multiple fish")
                pixels[k, present] = points[rows[present]]
            origin, direction = rig.camera(view).rays(pixels)
            valid = (np.isfinite(pixels).all(-1) & np.isfinite(origin).all(-1)
                     & np.isfinite(direction).all(-1))
            finite_rays[start:stop, :, vi] = valid
            origins.append(origin)
            directions.append(direction)
        pose[start:stop], _ = triangulate(
            np.stack(origins, axis=-2), np.stack(directions, axis=-2),
            valid=np.moveaxis(finite_rays[start:stop], 2, -1))
    finite_two_ray = (finite_rays.sum(axis=2) >= 2) & np.isfinite(pose).all(-1)
    tested_pose = pose.copy()
    tested_pose[:, :, 1] = saved_xyz  # Never replace the original saved pectoral geometry.
    return dict(frame_idx=frames.copy(), original_source_rows=source_rows.copy(),
                sources=expanded, baseline_xyz=saved_xyz.copy(),
                reconstructed_pose_xyz=pose, pose_xyz=tested_pose,
                finite_ray_source=finite_rays, finite_two_ray_observed=finite_two_ray,
                observed=(finite_rays.sum(axis=2) >= 2) & np.isfinite(tested_pose).all(-1))


def classify_strict_support(arrays, nodes, rig, *, block_frames=1660):
    """New posthoc support; original arrays and old label acceptance are unchanged.

    Reproduces audit_persistent_minute.verify_all_support's float32 water-ray
    storage and fixed 2 mm / 12 native-pixel equations. Missing original points
    and whole-NaN reconstructed nodes simply fail support, without imputation.
    The four required output keys can be passed to verify_support_partitioned.
    """
    frames, pose, raw = (np.asarray(arrays[k]) for k in ("frame_idx", "pose_xyz", "sources"))
    require(frames.ndim == 1 and len(frames) > 0 and frames.dtype.kind in "iu"
            and frames.min() >= 0 and frames.max() < FRAME_COUNT
            and np.all(np.diff(frames.astype(np.int64)) == 1), "invalid support native frames")
    require(pose.shape == (len(frames), 3, 3, 3) and pose.dtype.kind == "f"
            and not np.isinf(pose).any()
            and np.all(np.isfinite(pose).any(-1) == np.isfinite(pose).all(-1)),
            "support pose must be whole finite or NaN")
    require(raw.shape == pose.shape and raw.dtype.kind == "i" and np.all(raw >= -1),
            "invalid raw node sources")
    require(tuple(rig.views) == tuple(VIEWS), "wrong calibration view order")
    require(type(block_frames) is int and block_frames > 0, "positive block_frames required")
    distance = np.full(raw.shape, np.nan)
    error = np.full(raw.shape, np.nan)
    valid = np.zeros(raw.shape, bool)
    for start in range(0, len(frames), block_frames):
        stop = min(start+block_frames, len(frames))
        xyz = pose[start:stop]
        for vi, view in enumerate(VIEWS):
            pixels = np.full((*xyz.shape[:-1], 2), np.nan)
            for t, frame in enumerate(frames[start:stop]):
                points = nodes.at(view, int(frame))
                for node in range(3):
                    rows = raw[start+t, :, vi, node]
                    present = rows >= 0
                    require(np.all(rows[present] < len(points)), "raw node source out of range")
                    require(len(np.unique(rows[present])) == int(present.sum()),
                            "raw source used by multiple fish for one node")
                    pixels[t, present, node] = points[rows[present], node]
            camera = rig.camera(view)
            origin, direction = camera.rays(pixels)
            origin, direction = origin.astype(np.float32), direction.astype(np.float32)
            delta = xyz - origin
            residual = np.linalg.norm(
                delta - np.sum(delta*direction, axis=-1, keepdims=True)*direction, axis=-1)
            pixels_error = np.linalg.norm(camera.project(xyz) - pixels, axis=-1)
            keep = ((raw[start:stop, :, vi] >= 0) & (residual <= 2.0)
                    & np.isfinite(pixels_error) & (pixels_error <= 12.0))
            distance[start:stop, :, vi] = residual
            error[start:stop, :, vi] = pixels_error
            valid[start:stop, :, vi] = keep
    return dict(frame_idx=frames.copy(), pose_xyz=pose.copy(),
                sources=np.where(valid, raw, -1), observed=valid.sum(axis=2) >= 2,
                ray_distance_mm=distance, native_pixel_error=error, valid_source=valid)


@dataclass
class BaselineSupportInputs:
    arrays: dict
    nodes: RegisteredNodes
    rig: object
    verified_hashes: dict
    metadata: dict

    def final_guard(self):
        for path, expected in self.verified_hashes.items():
            require(sha256(path) == expected, "baseline input changed: " + path)


def load_baseline_sources(input_root, lambda_value, start, stop, *,
                          expected_catalog_sha256=CATALOG_SHA256, block_frames=1660):
    """Verify registered bytes and return unfiltered arrays for strict classification.

    Only the chosen baseline, three raw SLEAP caches and four calibration JSONs
    are opened. Catalog metadata is read but no GT files/other predictions are.
    Caller owns source snapshots/output seals and must repeat `final_guard`.
    """
    frames = native_frames(start, stop)
    require(str(lambda_value) in LAMBDAS, "unregistered lambda")
    root = Path(input_root).resolve(strict=True)
    catalog_path = (root / "staging_manifest.json").resolve(strict=True)
    require(catalog_path.is_relative_to(root), "catalog escaped input root")
    require(isinstance(expected_catalog_sha256, str) and len(expected_catalog_sha256) == 64
            and sha256(catalog_path) == expected_catalog_sha256, "staging catalog SHA differs")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    entries = {}
    for item in catalog["files"]:
        name = relative(item["relative"])
        require(name not in entries, "duplicate catalog entry")
        entries[name] = item
    hashes = {str(catalog_path): expected_catalog_sha256}

    def verified(name):
        name = relative(name)
        require(name in entries, "required registered input absent: " + name)
        item, path = entries[name], (root / name).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file(), "registered input escaped root")
        require(type(item["bytes"]) is int and path.stat().st_size == item["bytes"]
                and sha256(path) == item["sha256"], "registered input SHA/size differs: " + name)
        hashes[str(path)] = item["sha256"]
        return path

    baseline = verified(f"{BASELINE}/C34-label__{EXPERIMENT}__lam{lambda_value}.npz")
    paths = {v: verified(f"work/c01_s16_predictions/bottomup/{EXPERIMENT}__{v}.npz")
             for v in VIEWS}
    calibration_manifest = verified(f"{CALIBRATION}/calibration_manifest.json")
    calibration = json.loads(calibration_manifest.read_text(encoding="utf-8"))
    require(tuple(calibration["view_order"]) == tuple(VIEWS)
            and len(calibration["model_files"]) == 3, "wrong calibration manifest views")
    for view, item in zip(VIEWS, calibration["model_files"], strict=True):
        require(item["view"] == view, "wrong calibration model view")
        verified(f"{CALIBRATION}/{relative(item['path'])}")
    rig = load_rig(calibration_manifest.parent)
    nodes = RegisteredNodes(paths, frames)
    with np.load(baseline, allow_pickle=False) as data:
        expected = {"slot_track", "xyz", "residual_mm", "margin_mm", "n_views_used", "source",
                    *(f"track__{v}" for v in VIEWS)}
        require(len(data.files) == len(expected) and set(data.files) == expected,
                "wrong C34 label schema")
        slot, xyz, source = data["slot_track"], data["xyz"], data["source"]
        require(slot.shape == (FRAME_COUNT, 3) and slot.dtype == np.int32
                and xyz.shape == (FRAME_COUNT, 3, 3) and xyz.dtype == np.float32
                and source.shape == (FRAME_COUNT, 3, 3) and source.dtype == np.int8,
                "wrong C34 native shape/dtype")
        slot = slot[start:stop].copy()
        require(np.all((slot == -1) | (slot == np.arange(3))),
                "gated/different identity IDs are not C34 label slots")
        arrays = reconstruct_sources(frames, source[start:stop], xyz[start:stop], nodes, rig,
                                     block_frames=block_frames)
        arrays.update(slot_track=slot, label_available=slot >= 0)
    result = BaselineSupportInputs(arrays, nodes, rig, hashes, dict(
        schema="c34_label_support_inputs_v1", experiment=EXPERIMENT, lambda_value=str(lambda_value),
        frame_range=[start, stop], fps=166, native_frames=len(frames),
        nodes=list(NODES), views=list(VIEWS), strict_support_computed=False,
        source_semantics="unfiltered original instance row repeated for its three nodes",
        identity_semantics="C34 label anonymous columns; not gated track IDs or biological truth",
        geometry_semantics="saved pectorals unchanged; mouth/tail derived source triangulations",
        missing_semantics="NaN retained in denominator; no latent point or source manufactured",
        acceptance_semantics="original label_available retained separately from strict support"))
    result.final_guard()
    return result

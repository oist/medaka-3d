"""Diverse image-space occupancy proposals joined into additional temporal starts.

The output is a set of optimizer initializations, not accepted tracks. A translated
body uses a grid centre as a trial centre without declaring it an observed pectoral
landmark. The saved twelve previous starts are neither opened nor modified here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations

import numpy as np
from temporal_refine import boundary_starts


@dataclass(frozen=True)
class MaskInitializationConfig:
    unweighted_candidates: int = 24
    weighted_candidates: int = 24
    nms_radius_mm: float = 2.0
    support_threshold: float = 0.5
    velocity_weight: float = 0.05
    velocity_scale_mm: float = 3.0
    mask_weight: float = 3.0
    fallback_cost: float = 3.0
    lower_mm: tuple = (0.0, 0.0, -6.0)
    upper_mm: tuple = (192.0, 192.0, 186.0)

    def __post_init__(self):
        counts = (self.unweighted_candidates, self.weighted_candidates)
        if any(type(n) is not int or n < 0 for n in counts) or not 1 <= sum(counts) <= 48:
            raise ValueError("the two candidate phases must total 1..48 nonnegative integer slots")
        for key in ("nms_radius_mm", "velocity_scale_mm"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("velocity_weight", "mask_weight", "fallback_cost"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not np.isfinite(self.support_threshold) or not 0 < self.support_threshold <= 1:
            raise ValueError("support threshold must be in (0,1]")
        lower, upper = np.asarray(self.lower_mm), np.asarray(self.upper_mm)
        if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all():
            raise ValueError("bounds must contain finite lower/upper XYZ values")
        if np.any(lower >= upper):
            raise ValueError("registered upper bounds must exceed lower bounds")


@dataclass(frozen=True)
class FrameMaskVoxels:
    frame_idx: int
    grid_indices: np.ndarray
    view_support: np.ndarray
    novelty_weight: np.ndarray

    @classmethod
    def from_saved(cls, arrays):
        return cls(
            int(arrays["frame_idx"]),
            arrays["two_view_or_more_voxel_indices"].copy(),
            arrays["active_view_support"].copy(),
            arrays["active_novelty_weight"].copy(),
        )


@dataclass(frozen=True)
class DiversePool:
    grid_indices: np.ndarray
    xyz: np.ndarray
    view_support: np.ndarray
    mean_known_support: np.ndarray
    novelty_weight: np.ndarray
    selection_phase: np.ndarray  # 0 image-only diversity, 1 soft-healthy diversity
    selection_score: np.ndarray
    known_third_contradiction: np.ndarray


@dataclass(frozen=True)
class MaskInitializationResult:
    pose_xyz: np.ndarray
    permutations: np.ndarray
    path_costs: np.ndarray
    selected_grid_indices: np.ndarray
    selected_kind: np.ndarray
    scene_mapping: np.ndarray
    missing_slot: np.ndarray
    metadata: dict
    arrays: dict
    starts: tuple


def diverse_pool(grid_xyz, evidence, config=None):
    cfg = config or MaskInitializationConfig()
    grid = np.asarray(grid_xyz, float)
    ids, values = np.asarray(evidence.grid_indices), np.asarray(evidence.view_support, float)
    novelty = np.asarray(evidence.novelty_weight, float)
    if grid.ndim != 2 or grid.shape[1] != 3 or not np.isfinite(grid).all():
        raise ValueError("fixed grid must be finite K,3 coordinates")
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("voxel indices must be an integer vector")
    if np.any(ids < 0) or np.any(ids >= len(grid)) or len(np.unique(ids)) != len(ids):
        raise ValueError("voxel indices must be distinct and within the fixed grid")
    if values.shape != (len(ids), 3) or novelty.shape != (len(ids),):
        raise ValueError("voxel support/novelty dimensions disagree")
    if np.isinf(values).any() or np.any((values < 0) | (values > 1)):
        raise ValueError("view support must be in [0,1], with NaN for unknown")
    if not np.isfinite(novelty).all() or np.any((novelty <= 0) | (novelty > 1)):
        raise ValueError("novelty is a strictly positive soft weight no greater than one")
    count = np.sum(values >= cfg.support_threshold, axis=1)
    known = np.isfinite(values).sum(axis=1)
    if np.any(count < 2):
        raise ValueError("initializer input must contain only the saved >=2-view voxels")
    quality = np.divide(np.nansum(values, axis=1), known, out=np.empty(len(ids)), where=known > 0)
    points = grid[ids]
    nearest = np.full(len(ids), np.inf)
    remaining = np.ones(len(ids), bool)
    selected, phases, scores = [], [], []
    for phase, limit in enumerate((cfg.unweighted_candidates, cfg.weighted_candidates)):
        weight = quality if phase == 0 else quality * novelty
        for _ in range(limit):
            choices = np.flatnonzero(remaining)
            if not len(choices):
                break
            score = weight[choices] * nearest[choices] if selected else weight[choices]
            order = np.lexsort((ids[choices], -score))
            index = choices[order[0]]
            selected.append(index)
            phases.append(phase)
            scores.append(score[order[0]])
            nearest = np.minimum(nearest, np.linalg.norm(points - points[index], axis=1))
            remaining &= nearest > cfg.nms_radius_mm
    selected = np.asarray(selected, np.int64)
    return DiversePool(
        ids[selected].copy(),
        points[selected].copy(),
        values[selected].copy(),
        quality[selected].copy(),
        novelty[selected].copy(),
        np.asarray(phases, np.int8),
        np.asarray(scores),
        ((count == 2) & (known == 3))[selected],
    )


def whole_pose_candidates(base_pose, missing_slot, pool, config=None):
    """Keep original base, add whole-body translations, reject outside without clipping."""
    cfg = config or MaskInitializationConfig()
    base = np.asarray(base_pose, float)
    lower, upper = np.asarray(cfg.lower_mm), np.asarray(cfg.upper_mm)
    if base.shape != (3, 3, 3) or not np.isfinite(base).all():
        raise ValueError("base scene must be a finite three-fish, three-node pose")
    if np.any((base < lower) | (base > upper)):
        raise ValueError("the original base scene lies outside registered bounds")
    if missing_slot is not None and missing_slot not in (0, 1, 2):
        raise ValueError("at most one missing pectoral slot can receive image proposals")
    if missing_slot is None:
        return dict(
            poses=base[None].copy(),
            emissions=np.zeros(1),
            grid_indices=np.array([-1]),
            kinds=np.asarray(["supported_base"]),
            rejected_grid_indices=np.empty(0, np.int64),
            rejected_outside=np.empty((0, 3, 3, 3), bool),
        )
    translated = np.repeat(base[None], len(pool.xyz), axis=0)
    translated[:, missing_slot] += (pool.xyz - base[missing_slot, 1])[:, None]
    outside = (translated < lower) | (translated > upper)
    allowed = ~outside.any(axis=(1, 2, 3))
    return dict(
        poses=np.concatenate((base[None], translated[allowed])),
        emissions=np.r_[
            cfg.fallback_cost, cfg.mask_weight * (1 - pool.mean_known_support[allowed])
        ],
        grid_indices=np.r_[-1, pool.grid_indices[allowed]],
        kinds=np.asarray(["neutral_base_fallback", *(["mask_grid"] * int(allowed.sum()))]),
        rejected_grid_indices=pool.grid_indices[~allowed].copy(),
        rejected_outside=outside[~allowed],
    )


def transition_cost(previous, current, config=None):
    cfg = config or MaskInitializationConfig()
    delta = current[:, None, :, 1] - previous[None, :, :, 1]
    return (
        cfg.velocity_weight * np.mean(np.sum(delta**2, axis=-1), axis=-1) / cfg.velocity_scale_mm**2
    )


def dynamic_path(candidates, config=None):
    """Exact first-order dynamic programming over all retained three-body scenes."""
    cfg = config or MaskInitializationConfig()
    if not candidates:
        raise ValueError("DP needs at least one complete native-frame candidate set")
    costs, predecessors = [], []
    for t, frame in enumerate(candidates):
        pose, emission = np.asarray(frame["poses"]), np.asarray(frame["emissions"])
        if pose.ndim != 4 or pose.shape[1:] != (3, 3, 3) or not len(pose):
            raise ValueError("each frame needs one or more complete three-body candidates")
        if (
            emission.shape != (len(pose),)
            or not np.isfinite(pose).all()
            or not np.isfinite(emission).all()
        ):
            raise ValueError("DP scene coordinates and emissions must be finite and consistent")
        if t == 0:
            costs.append(emission.copy())
            predecessors.append(np.full(len(pose), -1, np.int64))
            continue
        total = transition_cost(candidates[t - 1]["poses"], pose, cfg) + costs[-1][None]
        previous = np.argmin(total, axis=1)
        costs.append(emission + total[np.arange(len(pose)), previous])
        predecessors.append(previous)
    selected = np.empty(len(candidates), np.int64)
    selected[-1] = np.argmin(costs[-1])
    for t in range(len(candidates) - 1, 0, -1):
        selected[t - 1] = predecessors[t][selected[t]]
    return dict(
        path_indices=selected,
        path_cost=float(costs[-1][selected[-1]]),
        costs=costs,
        predecessors=predecessors,
    )


def scene_mappings(forward_pose, forward_observed, endpoint_permutation):
    """Same nearest-linear unordered-scene mapping as boundary_starts, plus source rows."""
    forward = np.asarray(forward_pose)
    fraction = np.linspace(0, 1, len(forward))[:, None, None, None]
    linear = (1 - fraction) * forward[0] + fraction * forward[-1, endpoint_permutation]
    possibilities = np.asarray(list(permutations(range(3))), int)
    mappings = []
    for t in range(len(forward)):
        distances = ((linear[t, :, None, 1] - forward[t, None, :, 1]) ** 2).sum(-1)
        cost = distances[np.arange(3)[None], possibilities].sum(-1)
        mappings.append(possibilities[np.argmin(cost)])
    mappings = np.stack(mappings)
    mappings[:2] = np.arange(3)
    mappings[-2:] = endpoint_permutation
    supported = forward_observed[np.arange(len(forward))[:, None], mappings, 1]
    return mappings, supported


def build_mask_initializations(
    forward_pose, forward_observed, grid_xyz, evidence_by_frame, config=None
):
    cfg = config or MaskInitializationConfig()
    forward, observed = np.asarray(forward_pose, float), np.asarray(forward_observed)
    if observed.dtype != bool:
        raise ValueError("forward support must be boolean")
    original = boundary_starts(forward, observed)
    n_frames = len(forward)
    if len(evidence_by_frame) != n_frames or not 5 <= n_frames <= 50:
        raise ValueError("one independent mask-voxel set is required per bounded native frame")
    frames = np.asarray([e.frame_idx for e in evidence_by_frame], np.int64)
    if np.any(np.diff(frames) != 1):
        raise ValueError("mask initialization cannot bridge undeclared frame gaps")
    if np.any((~observed[:, :, 1]).sum(axis=1) > 1):
        raise ValueError("prototype refuses more than one physically missing pectoral in a frame")
    pools = [diverse_pool(grid_xyz, evidence, cfg) for evidence in evidence_by_frame]
    scene_rows = np.flatnonzero(original[2] == "scene")
    trajectories, mappings_all, missing_all, solutions, sets_all = [], [], [], [], []
    for source_row in scene_rows:
        endpoint = original[1][source_row]
        base = original[0][source_row]
        mappings, supported = scene_mappings(forward, observed, endpoint)
        missing = np.full(n_frames, -1, np.int64)
        candidate_sets = []
        for t in range(n_frames):
            absent = np.flatnonzero(~supported[t])
            slot = int(absent[0]) if len(absent) else None
            if slot is not None:
                missing[t] = slot
            candidate_sets.append(whole_pose_candidates(base[t], slot, pools[t], cfg))
        solved = dynamic_path(candidate_sets, cfg)
        path = np.stack(
            [c["poses"][i] for c, i in zip(candidate_sets, solved["path_indices"], strict=True)]
        )
        boundary = original[3]
        if not np.array_equal(path[boundary], base[boundary]):
            raise ValueError("DP changed a fixed boundary scene")
        trajectories.append(path)
        mappings_all.append(mappings)
        missing_all.append(missing)
        solutions.append(solved)
        sets_all.extend(candidate_sets)
    if len(trajectories) != 6:
        raise ValueError("all six endpoint permutations must remain explicit")
    poses = np.stack(trajectories)
    selected = np.stack([s["path_indices"] for s in solutions])
    chosen_grid = np.asarray(
        [
            [sets_all[b * n_frames + t]["grid_indices"][selected[b, t]] for t in range(n_frames)]
            for b in range(6)
        ]
    )
    kinds = np.asarray(
        [
            [sets_all[b * n_frames + t]["kinds"][selected[b, t]] for t in range(n_frames)]
            for b in range(6)
        ]
    )
    offsets = np.r_[0, np.cumsum([len(c["poses"]) for c in sets_all])]
    rejected_offsets = np.r_[0, np.cumsum([len(c["rejected_grid_indices"]) for c in sets_all])]
    pool_offsets = np.r_[0, np.cumsum([len(pool.xyz) for pool in pools])]
    starts = (
        poses,
        original[1][scene_rows].copy(),
        np.full(6, "mask_grid"),
        original[3].copy(),
        original[4][scene_rows].copy(),
        original[5][scene_rows].copy(),
    )
    arrays = dict(
        frame_idx=frames,
        pose_xyz=poses,
        permutations=starts[1],
        start_kind=starts[2],
        path_costs=np.asarray([s["path_cost"] for s in solutions]),
        path_indices=selected,
        selected_grid_indices=chosen_grid,
        selected_kind=kinds,
        scene_mapping=np.stack(mappings_all),
        missing_slot=np.stack(missing_all),
        candidate_offsets=offsets,
        candidate_pose_xyz=np.concatenate([c["poses"] for c in sets_all]),
        candidate_emissions=np.concatenate([c["emissions"] for c in sets_all]),
        candidate_grid_indices=np.concatenate([c["grid_indices"] for c in sets_all]),
        candidate_kind=np.concatenate([c["kinds"] for c in sets_all]),
        dp_costs=np.concatenate([cost for s in solutions for cost in s["costs"]]),
        dp_predecessor_local_index=np.concatenate(
            [p for s in solutions for p in s["predecessors"]]
        ),
        rejected_offsets=rejected_offsets,
        rejected_grid_indices=np.concatenate([c["rejected_grid_indices"] for c in sets_all]),
        rejected_outside=np.concatenate([c["rejected_outside"] for c in sets_all]),
        pool_offsets=pool_offsets,
    )
    for key in DiversePool.__dataclass_fields__:
        arrays["pool_" + key] = np.concatenate([getattr(pool, key) for pool in pools])
    return MaskInitializationResult(
        pose_xyz=arrays["pose_xyz"],
        permutations=arrays["permutations"],
        path_costs=arrays["path_costs"],
        selected_grid_indices=arrays["selected_grid_indices"],
        selected_kind=arrays["selected_kind"],
        scene_mapping=arrays["scene_mapping"],
        missing_slot=arrays["missing_slot"],
        starts=starts,
        arrays=arrays,
        metadata={
            "config": asdict(cfg),
            "new_external_resets": 0,
            "accepted_identity_output": False,
            "source": "independent saved mask voxels and internal forward scene estimates",
            "permutation_semantics": "left identity -> original right-boundary slot",
            "candidate_offset_order": "permutation-major, then native time; length 6*T+1",
            "dp_predecessor_semantics": "local index at previous time, same permutation",
            "limitations": [
                "These six extra starts leave the original twelve external starts unchanged.",
                "Finite diversity/soft healthy weighting cannot guarantee every fish is covered.",
                "Known third-view contradictions and strict-three ghosts can remain.",
                "A grid centre is a trial translation target, not an anatomical observation.",
                "This prototype supports at most one physically missing pectoral per frame.",
            ],
        },
    )

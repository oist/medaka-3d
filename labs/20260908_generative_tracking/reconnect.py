"""Global ID reconnection on a fixed reconstructed three-fish sequence.

This pass optimizes six permutations, not the 3D coordinates. Complete geometry
frames delimit blocks; incomplete blocks additionally connect their observed
endpoints. Conditional block dynamic programs retain both boundary permutations,
so missing endpoints do not require a fictitious last-observation Markov state.
Margins are differences in model cost, not calibrated identity probabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np
from tracker import PERMUTATIONS, TrackResult

_PERMS = np.asarray(PERMUTATIONS, dtype=int)  # persistent ID -> raw geometry slot
_NODE_WEIGHTS = np.array([1.0, 1.0, 0.35])


@dataclass(frozen=True)
class ReconnectConfig:
    template_frames: int = 20
    minimum_quality: float = 0.25
    appearance_weight: float = 4.0
    length_weight: float = 0.05
    length_sigma_mm: float = 3.0
    motion_sigma_mm: float = 2.0
    gap_weight: float = 0.5
    gap_sigma_mm: float = 2.0
    review_margin: float = 2.0
    appearance_clear_margin: float = 2.0
    maximum_speed_mm_per_frame: float = 4.0
    position_slack_mm: float = 6.0
    template_separation: float = 0.05
    epoch_appearance_margin: float = 0.03
    epoch_minimum_samples: int = 3
    epoch_minimum_views: int = 2

    def __post_init__(self):
        if self.template_frames < 1:
            raise ValueError("template_frames must be positive")
        if self.epoch_minimum_samples < 1 or not 1 <= self.epoch_minimum_views <= 3:
            raise ValueError("epoch requires positive sample count and one to three views")
        if not 0 <= self.minimum_quality <= 1:
            raise ValueError("minimum_quality must lie in [0, 1]")
        for key in ("length_sigma_mm", "motion_sigma_mm", "gap_sigma_mm",
                    "maximum_speed_mm_per_frame", "position_slack_mm"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("appearance_weight", "length_weight", "gap_weight", "review_margin",
                    "appearance_clear_margin", "template_separation", "epoch_appearance_margin"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative and finite")


@dataclass(frozen=True)
class ReconnectResult(TrackResult):
    appearance: np.ndarray | None = None
    appearance_quality: np.ndarray | None = None
    geometry_review: np.ndarray | None = None
    identity_review: np.ndarray | None = None
    permutation: np.ndarray | None = None
    state_min_cost: np.ndarray | None = None
    appearance_margin_gain: np.ndarray | None = None
    epoch_appearance_margin: np.ndarray | None = None
    appearance_verified: np.ndarray | None = None
    connection_review: np.ndarray | None = None


def _array(result, name, shape, dtype):
    value = np.asarray(getattr(result, name), dtype=dtype)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return value


def _validate(result, initial):
    frames = np.asarray(result.frame_idx)
    if frames.ndim != 1 or not len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError("frame_idx must be a nonempty strictly increasing vector")
    if not np.issubdtype(frames.dtype, np.integer):
        raise ValueError("frame_idx must contain integer frame indexes")
    n = len(frames)
    pose = _array(result, "pose_xyz", (n, 3, 3, 3), float)
    observed = _array(result, "observed", (n, 3, 3), bool)
    source = _array(result, "node_source", (n, 3, 3, 3), int)
    _array(result, "xyz", (n, 3, 3), float)
    review = _array(result, "review", (n, 3), bool)
    if initial.shape != (3, 3, 3) or not np.isfinite(initial).all():
        raise ValueError("initial_xyz must be finite with shape (3, 3, 3)")
    if not np.isfinite(pose[observed]).all():
        raise ValueError("observed nodes require finite 3D coordinates")
    # Detector row numbers have meaning within this frame, view, and node only.
    for i in range(3):
        for j in range(i + 1, 3):
            if np.any((source[:, i] >= 0) & (source[:, i] == source[:, j])):
                raise ValueError("node_source is not exclusive within a frame/view/node")
    geometry = getattr(result, "geometry_review", None)
    identity = getattr(result, "identity_review", None)
    geometry = review.copy() if geometry is None else np.asarray(geometry, bool)
    identity = review.copy() if identity is None else np.asarray(identity, bool)
    if geometry.shape != (n, 3) or identity.shape != (n, 3):
        raise ValueError("geometry_review and identity_review must have shape (T, 3)")
    appearance = getattr(result, "appearance", None)
    quality = getattr(result, "appearance_quality", None)
    if (appearance is None) != (quality is None):
        raise ValueError("appearance and appearance_quality must be supplied together")
    if appearance is None:
        appearance, quality = np.zeros((n, 3, 3, 24)), np.zeros((n, 3, 3))
    else:
        appearance = np.asarray(appearance, float)
        quality = np.asarray(quality, float)
    if appearance.shape != (n, 3, 3, 24) or quality.shape != (n, 3, 3):
        raise ValueError("appearance requires (T, 3, 3, 24); quality requires (T, 3, 3)")
    if (not np.isfinite(appearance).all() or np.any(appearance < 0)
            or not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1))):
        raise ValueError("appearance must be nonnegative finite; quality must lie in [0, 1]")
    total = appearance.sum(axis=-1, keepdims=True)
    quality = quality * (total[..., 0] > 0) * observed[:, :, :2].all(axis=-1)[..., None]
    appearance = np.divide(appearance, total, out=np.zeros_like(appearance), where=total > 0)
    return frames, pose, observed, geometry, identity, appearance, quality


def _emissions(pose, observed, geometry, identity, hist, quality, initial, config):
    n = len(pose)
    templates = np.zeros((3, 3, 24))
    template_quality = np.zeros((3, 3))
    template_count = np.zeros((3, 3), int)
    # Stop each initial slot's prefix at the FIRST uncertain frame. No template
    # is updated after a gap, an overlap/identity review, or a later relabeling.
    for fish in range(3):
        reliable = observed[:, fish, :2].all(axis=1) & ~geometry[:, fish] & ~identity[:, fish]
        bad = np.flatnonzero(~reliable)
        stop = min(config.template_frames, int(bad[0]) if len(bad) else n, n)
        for view in range(3):
            good = quality[:stop, fish, view] >= config.minimum_quality
            good &= quality[:stop, fish, view] > 0
            weights = quality[:stop, fish, view][good]
            if len(weights):
                templates[fish, view] = np.average(hist[:stop, fish, view][good],
                                                  axis=0, weights=weights)
                template_quality[fish, view] = weights.mean()
                template_count[fish, view] = len(weights)
    appearance_cost = np.zeros((n, 3, 3))  # time, identity, raw slot
    shape_cost = np.zeros_like(appearance_cost)
    lengths = np.linalg.norm(initial[:, [0, 2]] - initial[:, 1, None], axis=-1)
    for fish in range(3):
        affinity = np.sqrt(hist * templates[fish][None, None]).sum(axis=-1)
        appearance_cost[:, fish] = config.appearance_weight * (
            (1 - np.clip(affinity, 0, 1)) * quality * template_quality[fish]
        ).sum(axis=-1)
        for seg, node in enumerate((0, 2)):
            distance = np.linalg.norm(pose[:, :, node] - pose[:, :, 1], axis=-1)
            error = np.nan_to_num((distance - lengths[fish, seg]) / config.length_sigma_mm)
            supported = observed[:, :, node] & observed[:, :, 1]
            shape_cost[:, fish] += config.length_weight * supported * 2 * (
                np.sqrt(1 + error**2) - 1)
    app = np.stack([sum(appearance_cost[:, f, p[f]] for f in range(3)) for p in _PERMS], 1)
    shape = np.stack([sum(shape_cost[:, f, p[f]] for f in range(3)) for p in _PERMS], 1)
    return shape + app, shape, template_count, templates


def _pair_cost(pose_a, pose_b, valid_a, valid_b, sigma, maximum_displacement):
    valid = valid_a[:, None] & valid_b[None]
    distance = np.linalg.norm(pose_a[:, None] - pose_b[None], axis=-1)
    error = np.nan_to_num(distance / sigma)
    weighted = valid * _NODE_WEIGHTS
    result = (2 * (np.sqrt(1 + error**2) - 1) * weighted).sum(axis=-1) / np.maximum(
        weighted.sum(axis=-1), 1)
    # The soft motion score cannot allow accumulated appearance evidence to
    # buy an instantaneous teleport. This is an engineering admission bound,
    # not a measured biological speed limit. Tail flexion is excluded.
    head_valid = valid[..., :2]
    displacement = np.nan_to_num(pose_b[None, :, :2] - pose_a[:, None, :2])
    head_shift = (displacement * head_valid[..., None]).sum(axis=2)
    head_shift /= np.maximum(head_valid.sum(axis=2, keepdims=True), 1)
    impossible = head_valid.any(axis=2) & (np.linalg.norm(head_shift, axis=2)
                                         > maximum_displacement)
    result[impossible] = np.inf
    return result


def _permutation_cost(pair):
    return sum(pair[_PERMS[:, f, None], _PERMS[None, :, f]] for f in range(3))


def _connections(frames, pose, observed, geometry, config):
    n = len(frames)
    reliable = observed & ~geometry[..., None]
    transitions = np.zeros((n, 6, 6))
    for t in range(1, n):
        dt = frames[t] - frames[t - 1]
        sigma = config.motion_sigma_mm * np.sqrt(dt)
        bound = config.maximum_speed_mm_per_frame * dt + config.position_slack_mm
        transitions[t] = _permutation_cost(_pair_cost(
            pose[t - 1], pose[t], observed[t - 1], observed[t], sigma, bound))
    complete = reliable[:, :, :2].all(axis=(1, 2))
    anchors = np.unique(np.r_[0, np.flatnonzero(complete), n - 1])
    gaps = []
    for a, b in pairwise(anchors):
        if b == a + 1:
            gaps.append(np.zeros((6, 6)))
        else:
            sigma = config.gap_sigma_mm * (1 + np.sqrt(frames[b] - frames[a]))
            bound = (config.maximum_speed_mm_per_frame * (frames[b] - frames[a])
                     + config.position_slack_mm)
            pair_cost = _permutation_cost(_pair_cost(
                pose[a], pose[b], reliable[a], reliable[b], sigma, bound))
            finite = np.isfinite(pair_cost)
            gaps.append(np.where(finite, config.gap_weight * np.where(finite, pair_cost, 0),
                                 np.inf))
    return anchors, transitions, gaps


def _solve(emission, transition, anchors, gaps):
    """Exact minimum and min-marginals for the stated block-endpoint objective."""
    n = len(emission)
    boundary = np.full((len(anchors), 6), np.inf)
    boundary[0, 0] = emission[0, 0]  # the sole fixed identity permutation
    blocks = []
    for block, (a, b) in enumerate(pairwise(anchors)):
        prefix = np.full((b - a + 1, 6, 6), np.inf)  # local time, start, current
        np.fill_diagonal(prefix[0], 0)
        for j, t in enumerate(range(a + 1, b + 1), 1):
            prefix[j] = (prefix[j - 1, :, :, None] + transition[t][None]).min(axis=1)
            prefix[j] += emission[t]
        kernel = prefix[-1] + gaps[block]
        boundary[block + 1] = (boundary[block, :, None] + kernel).min(axis=0)
        blocks.append((prefix, kernel))
    suffix_boundary = np.zeros_like(boundary)
    for block in range(len(blocks) - 1, -1, -1):
        suffix_boundary[block] = (blocks[block][1] + suffix_boundary[block + 1]).min(axis=1)
    optimum = boundary[-1].min()
    if not np.isfinite(optimum):
        # Preserve the raw geometry for inspection; no ID is accepted when
        # even one complete sequence cannot satisfy the engineering bound.
        cost = np.zeros((n, 6))
        cost[0, 1:] = np.inf
        return np.zeros(n, int), cost, float("inf")
    state_cost = np.full((n, 6), np.inf)
    state_cost[anchors] = boundary + suffix_boundary
    path = np.zeros(n, int)
    path[-1] = int(np.argmin(boundary[-1]))
    for block in range(len(blocks) - 1, -1, -1):
        a, b = anchors[block:block + 2]
        prefix, kernel = blocks[block]
        end = path[b]
        start = int(np.argmin(boundary[block] + kernel[:, end]))
        path[a] = start
        current = end
        for t in range(b, a + 1, -1):
            previous = int(np.argmin(prefix[t - a - 1, start] + transition[t, :, current]))
            path[t - 1] = previous
            current = previous
        suffix = np.full((b - a + 1, 6, 6), np.inf)  # time, current, end
        np.fill_diagonal(suffix[-1], 0)
        for t in range(b - 1, a, -1):
            suffix[t - a] = (transition[t + 1, :, :, None]
                             + emission[t + 1, None, :, None]
                             + suffix[t - a + 1][None]).min(axis=1)
            combined = (boundary[block, :, None, None]
                        + prefix[t - a, :, :, None] + gaps[block][:, None, :]
                        + suffix[t - a][None] + suffix_boundary[block + 1][None, None])
            state_cost[t] = combined.min(axis=(0, 2))
    return path, np.maximum(state_cost - optimum, 0), float(optimum)


def _identity_contrast(state_cost, permutation):
    result = np.zeros((len(state_cost), 3))
    for fish in range(3):
        same = _PERMS[None, :, fish] == permutation[:, fish, None]
        compatible = np.where(same, state_cost, np.inf).min(axis=1)
        other = np.where(~same, state_cost, np.inf).min(axis=1)
        result[:, fish] = other - compatible
    return result


def _appearance_epochs(hist, quality, supported, templates, counts, config):
    """Require separated templates and a mean advantage in multiple views.

    Epoch means, unlike a cumulative cost gap, cannot become large merely by
    repeating thousands of nearly indistinguishable image histograms.
    """
    n = len(hist)
    epoch_margin = np.full((n, 3, 3), np.nan)
    verified = np.zeros((n, 3), bool)
    for fish in range(3):
        others = [i for i in range(3) if i != fish]
        own = 1 - np.sqrt(hist[:, fish] * templates[fish]).sum(axis=-1)
        alternative = np.stack([
            1 - np.sqrt(hist[:, fish] * templates[other]).sum(axis=-1) for other in others
        ])
        separation = np.stack([
            1 - np.sqrt(templates[fish] * templates[other]).sum(axis=-1) for other in others
        ]).min(axis=0)
        template_valid = (counts > 0).all(axis=0) & (separation >= config.template_separation)
        per_frame_margin = alternative.min(axis=0) - own
        starts = np.flatnonzero(supported[:, fish] & ~np.r_[False, supported[:-1, fish]])
        ends = np.flatnonzero(supported[:, fish] & ~np.r_[supported[1:, fish], False]) + 1
        for start, end in zip(starts, ends, strict=True):
            agrees = np.zeros(3, bool)
            for view in range(3):
                good = quality[start:end, fish, view] >= config.minimum_quality
                good &= quality[start:end, fish, view] > 0
                if good.sum() < config.epoch_minimum_samples or not template_valid[view]:
                    continue
                weights = quality[start:end, fish, view][good]
                value = np.average(per_frame_margin[start:end, view][good], weights=weights)
                epoch_margin[start:end, fish, view] = value
                agrees[view] = value >= config.epoch_appearance_margin
            verified[start:end, fish] = agrees.sum() >= config.epoch_minimum_views
    return epoch_margin, verified


def _connection_review(frames, pose, observed, config):
    """Audit each chosen ID against its own last supported head, across gaps.

    The common block endpoints are a tractable objective, not a sufficient
    Markov state for every partially missing animal. A path that evades the
    adjacent-frame bound through a missing slot must remain unaccepted even
    when its appearance score is strong. This audit does not pretend to find
    the next best path under all these higher-order constraints.
    """
    review = np.zeros((len(frames), 3), bool)
    events = []
    for fish in range(3):
        previous = None
        for t in range(len(frames)):
            if not observed[t, fish, :2].all():
                continue
            if previous is not None:
                dt = int(frames[t] - frames[previous])
                displacement = float(np.linalg.norm(
                    (pose[t, fish, :2] - pose[previous, fish, :2]).mean(axis=0)))
                bound = config.maximum_speed_mm_per_frame * dt + config.position_slack_mm
                if displacement > bound:
                    review[t:, fish] = True
                    events.append({"id": fish, "previous_frame": int(frames[previous]),
                                   "frame": int(frames[t]), "displacement_mm": displacement,
                                   "engineering_bound_mm": bound})
            previous = t
    return review, events


def reconnect(result, initial_xyz, config: ReconnectConfig | None = None) -> ReconnectResult:
    """Relabel measured geometry with all available past/future image evidence.

    Input coordinates and observation sources are never fitted or interpolated.
    A missing-support history needs appearance evidence to release ID review,
    even when the raw beam erroneously reported no identity ambiguity.
    """
    config = config or ReconnectConfig()
    initial = np.asarray(initial_xyz, float)
    frames, pose, observed, geometry, identity, hist, quality = _validate(result, initial)
    emission, noappearance, counts, templates = _emissions(
        pose, observed, geometry, identity, hist, quality, initial, config)
    anchors, transitions, gaps = _connections(frames, pose, observed, geometry, config)
    path, state_cost, optimum = _solve(emission, transitions, anchors, gaps)
    _, noapp_state_cost, _ = _solve(noappearance, transitions, anchors, gaps)
    permutation = _PERMS[path]
    indexes = np.arange(len(frames))[:, None]

    def relabel(value):
        return np.asarray(value)[indexes, permutation].copy()

    margin = _identity_contrast(state_cost, permutation)
    geometry_margin = _identity_contrast(noapp_state_cost, permutation)
    gain = np.zeros_like(margin)
    finite = np.isfinite(margin) & np.isfinite(geometry_margin)
    gain[finite] = margin[finite] - geometry_margin[finite]
    chosen_observed = relabel(observed)
    geometry_review = relabel(geometry) | ~chosen_observed[:, :, :2].all(axis=2)
    # A frame-local row/slot number supplies no identity evidence. Loss history
    # follows the globally chosen persistent-ID path, not unchanged raw slots.
    interrupted = ~chosen_observed[:, :, :2].all(axis=2)
    interrupted[1:] |= permutation[1:] != permutation[:-1]
    unresolved = np.logical_or.accumulate(relabel(identity) | interrupted, axis=0)
    epoch_margin, appearance_verified = _appearance_epochs(
        relabel(hist), relabel(quality), chosen_observed[:, :, :2].all(axis=2),
        templates, counts, config)
    appearance_resolved = ((margin >= config.review_margin)
                           & (gain >= config.appearance_clear_margin) & appearance_verified)
    identity_review = ((margin < config.review_margin)
                       | (unresolved & ~appearance_resolved))
    connection_review, connection_events = _connection_review(
        frames, relabel(pose), chosen_observed, config)
    identity_review |= connection_review
    if not np.isfinite(optimum):
        identity_review[:] = True
    review = geometry_review | identity_review
    diagnostics = {
        "algorithm": "fixed-geometry-global-id-reconnection",
        "raw_algorithm": result.diagnostics,
        "config": asdict(config),
        "initializations": 1,
        "later_baseline_or_ground_truth_inputs": False,
        "coordinate_refitting_or_interpolation": False,
        "first_permutation_fixed": True,
        "template_counts": counts.tolist(),
        "template_histograms": templates.tolist(),
        "complete_geometry_anchors": len(anchors),
        "incomplete_blocks": int(np.sum(np.diff(anchors) > 1)),
        "global_cost": optimum,
        "adjacent_and_block_constraints_feasible": bool(np.isfinite(optimum)),
        "physically_feasible_global_path": bool(np.isfinite(optimum) and not connection_events),
        "per_id_gap_connection_audit_passed": not connection_events,
        "invalid_gap_connections": connection_events,
        "permuted_animal_frames": int((permutation != np.arange(3)).sum()),
        "identity_review_animal_frames": int(identity_review.sum()),
        "geometry_review_animal_frames": int(geometry_review.sum()),
        "identity_margin_kind": "global model cost gap on fixed geometry, not probability",
        "appearance_margin_gain_kind": "ID cost gap minus gap without appearance emissions",
        "appearance_release_rule": "Also requires separated initial templates and positive "
                                   "mean epoch margin in at least two camera views",
        "limitation": "Cannot restore missing/pruned geometry or prove biological identities; "
                      "same-appearance occluded animals can remain unidentifiable. "
                      "Individual gap reachability is audited after common-block optimization; "
                      "failed paths are reviewed, not claimed to be reoptimized exactly.",
    }
    xyz = relabel(result.xyz)
    xyz[~chosen_observed[:, :, 1]] = np.nan
    return ReconnectResult(
        frame_idx=frames.copy(), pose_xyz=relabel(pose), xyz=xyz,
        observed=chosen_observed, node_source=relabel(result.node_source), review=review,
        margin=np.sort(state_cost, axis=1)[:, 1], diagnostics=diagnostics,
        identity_margin=margin, appearance=relabel(hist), appearance_quality=relabel(quality),
        geometry_review=geometry_review, identity_review=identity_review,
        permutation=permutation, state_min_cost=state_cost, appearance_margin_gain=gain,
        epoch_appearance_margin=epoch_margin, appearance_verified=appearance_verified,
        connection_review=connection_review)

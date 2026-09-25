"""Conservative, fixed-coordinate *proposals* for three-slot reconnection.

These diagnostics never establish animal identity and never mutate tracking data.
``permutation[i]`` selects the raw right/window slot corresponding to left ID i.
All distances are mm and slopes mm/native frame. Image support, when supplied,
means only that a coordinate is supported; it is not independent identity evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations

import numpy as np

IDENTITY = (0, 1, 2)
PERMUTATIONS = tuple(permutations(range(3)))


@dataclass(frozen=True)
class ReconnectConfig:
    reference_frames: int = 17
    min_pair_separation: int = 3
    max_reference_step_mm: float = 4.0
    max_reference_median_residual_mm: float = 1.0
    max_suspect_frames: int = 83
    min_margin_mm: float = 5.0
    max_boundary_error_mm: float = 3.0

    def __post_init__(self):
        if self.reference_frames < 3:
            raise ValueError("reference_frames must be >= 3")
        if not 1 <= self.min_pair_separation < self.reference_frames:
            raise ValueError("min_pair_separation must be inside reference window")
        if self.max_suspect_frames < 1:
            raise ValueError("max_suspect_frames must be positive")
        thresholds = (
            self.max_reference_step_mm,
            self.max_reference_median_residual_mm,
            self.min_margin_mm,
            self.max_boundary_error_mm,
        )
        if not all(np.isfinite(v) and v > 0 for v in thresholds):
            raise ValueError("all distance thresholds must be positive and finite")


def _inputs(xyz, frame_idx, lo, hi, support):
    xyz = np.asarray(xyz, dtype=float)
    frame_idx = np.asarray(frame_idx)
    if xyz.ndim != 3 or xyz.shape[1:] != (3, 3):
        raise ValueError("xyz must have shape (T, 3, 3)")
    if frame_idx.shape != (len(xyz),) or not np.all(np.isfinite(frame_idx)):
        raise ValueError("frame_idx must contain T finite native frame indices")
    if np.any(frame_idx != np.floor(frame_idx)) or np.any(np.diff(frame_idx) <= 0):
        raise ValueError("frame_idx must be strictly increasing integer native indices")
    if not isinstance(lo, (int, np.integer)) or not isinstance(hi, (int, np.integer)):
        raise ValueError("suspect bounds must be integer row indices")
    if not 0 <= lo <= hi < len(xyz):
        raise ValueError("suspect row bounds outside xyz")
    available = np.isfinite(xyz).all(axis=2)
    if support is not None:
        support = np.asarray(support, dtype=bool)
        if support.shape != available.shape:
            raise ValueError("support must have shape (T, 3)")
        available &= support
    return xyz, frame_idx.astype(np.int64), available


def _reference(xyz, frame_idx, available, start, stop, config):
    """Fit exactly one contiguous flank, with no gaps or partial substitution."""
    report = {"row_start": int(start), "row_stop_exclusive": int(stop), "valid": False}
    if start < 0 or stop > len(xyz):
        report["reason"] = "incomplete_flank"
        return report, None
    frames = frame_idx[start:stop]
    report["frame_start"] = int(frames[0])
    report["frame_end"] = int(frames[-1])
    report["supported_counts"] = available[start:stop].sum(axis=0).tolist()
    if np.any(np.diff(frames) != 1):
        report["reason"] = "gapped_flank"
        return report, None
    if not np.all(available[start:stop]):
        report["reason"] = "missing_or_unsupported_flank"
        return report, None
    points = xyz[start:stop]
    center_frame = int(frames[len(frames) // 2])
    times = frames - center_frame
    first, second = np.triu_indices(len(frames), k=config.min_pair_separation)
    slopes = (points[second] - points[first]) / (times[second] - times[first])[:, None, None]
    velocity = np.median(slopes, axis=0)
    intercept = np.median(points - times[:, None, None] * velocity, axis=0)
    fitted = intercept + times[:, None, None] * velocity
    median_residual = np.median(np.linalg.norm(points - fitted, axis=2), axis=0)
    max_step = np.linalg.norm(np.diff(points, axis=0), axis=2).max(axis=0)
    report.update(
        center_frame=center_frame,
        velocity_mm_per_frame=velocity.tolist(),
        median_fit_residual_mm=median_residual.tolist(),
        max_native_step_mm=max_step.tolist(),
    )
    if np.any(max_step > config.max_reference_step_mm):
        report["reason"] = "reference_jump"
        return report, None
    if np.any(median_residual > config.max_reference_median_residual_mm):
        report["reason"] = "reference_not_constant_velocity"
        return report, None
    report.update(valid=True, reason="stable_contiguous_flank")
    return report, (center_frame, intercept, velocity)


def _predict(model, frame):
    center, position, velocity = model
    return position + (frame - center) * velocity


def _prepare(xyz, frame_idx, lo, hi, support, config, mode):
    xyz, frames, available = _inputs(xyz, frame_idx, lo, hi, support)
    result = {
        "mode": mode,
        "row_start": int(lo),
        "row_end_inclusive": int(hi),
        "frame_start": int(frames[lo]),
        "frame_end": int(frames[hi]),
        "config": asdict(config),
        "status": "not_scored",
        "reason": "",
        "suggested_permutation": None,
        "permutation_semantics": "output_left_id_i_selects_raw_slot_permutation_i",
        "scores": [],
        "independent_identity_validated": False,
        "image_support_provided": support is not None,
        "limitations": "Geometry only; every proposal requires raw-image identity review.",
    }
    if frames[hi] - frames[lo] + 1 > config.max_suspect_frames:
        result.update(status="rejected_long_window", reason="suspect_duration_exceeds_limit")
        return result, None
    left, lm = _reference(
        xyz, frames, available, lo - config.reference_frames, lo, config
    )
    right, rm = _reference(
        xyz, frames, available, hi + 1, hi + 1 + config.reference_frames, config
    )
    result["left_reference"] = left
    result["right_reference"] = right
    if lm is None or rm is None:
        result.update(status="insufficient_reference", reason="invalid_independent_flank")
        return result, None
    if np.any(np.diff(frames[lo - 1 : hi + 2]) != 1):
        result.update(status="rejected_frame_gap", reason="suspect_or_boundary_frames_not_native")
        return result, None
    return result, (xyz, frames, available, lm, rm)


def _score(permutation, forward_errors, backward_errors):
    # Shape (boundary count, output fish). For a transient, report both ends;
    # sum cost uses the per-fish mean across ends so the mm margin stays comparable.
    return {
        "permutation": list(permutation),
        "forward_error_mm": forward_errors.tolist(),
        "backward_error_mm": backward_errors.tolist(),
        "forward_sum_mm": float(forward_errors.mean(axis=0).sum()),
        "backward_sum_mm": float(backward_errors.mean(axis=0).sum()),
        "forward_max_fish_mm": float(forward_errors.max()),
        "backward_max_fish_mm": float(backward_errors.max()),
    }


def _verdict(result, available, lo, hi, config):
    ranked_f = sorted(result["scores"], key=lambda s: s["forward_sum_mm"])
    ranked_b = sorted(result["scores"], key=lambda s: s["backward_sum_mm"])
    best_f, best_b = ranked_f[0], ranked_b[0]
    margin_f = ranked_f[1]["forward_sum_mm"] - best_f["forward_sum_mm"]
    margin_b = ranked_b[1]["backward_sum_mm"] - best_b["backward_sum_mm"]
    result.update(
        forward_best=best_f["permutation"],
        backward_best=best_b["permutation"],
        forward_margin_mm=margin_f,
        backward_margin_mm=margin_b,
    )
    if best_f["permutation"] != best_b["permutation"]:
        result.update(status="ambiguous", reason="forward_backward_disagree")
    elif best_f["permutation"] == list(IDENTITY):
        result.update(status="identity_preferred", reason="no_geometric_reassignment_supported")
    elif margin_f < config.min_margin_mm or margin_b < config.min_margin_mm:
        result.update(status="ambiguous", reason="insufficient_margin_over_next_best")
    elif (
        best_f["forward_max_fish_mm"] > config.max_boundary_error_mm
        or best_b["backward_max_fish_mm"] > config.max_boundary_error_mm
    ):
        result.update(status="rejected_motion_mismatch", reason="per_fish_boundary_error_too_large")
    else:
        changed_slots = np.flatnonzero(np.asarray(best_f["permutation"]) != np.arange(3))
        if not np.all(available[lo : hi + 1, changed_slots]):
            result.update(status="rejected_support", reason="changed_slot_missing_or_unsupported")
        else:
            result.update(
                status="geometry_proposal_pending_review",
                reason="both_directions_decisive_and_close_requires_independent_image_review",
                suggested_permutation=best_f["permutation"],
            )
    return result


def audit_window(xyz, frame_idx, lo, hi, support=None, config=None):
    """Audit a persistent reconnection across inclusive suspect rows [lo, hi].

    Independent flanks immediately outside the window must each contain 17 native
    frames by default. Fit the left fish to the right boundary and the right fish
    back to the left boundary. A proposal connects these tracklets only: neither
    a change frame inside the window nor a suffix-wide correction is inferred.
    """
    config = config or ReconnectConfig()
    result, context = _prepare(
        xyz, frame_idx, lo, hi, support, config, "persistent_tracklet_connection"
    )
    if context is None:
        return result
    _, frames, available, lm, rm = context
    left_frame, right_frame = frames[lo - 1], frames[hi + 1]
    for p in PERMUTATIONS:
        forward = np.linalg.norm(
            _predict(lm, right_frame) - _predict(rm, right_frame)[list(p)], axis=1
        )
        backward = np.linalg.norm(
            _predict(lm, left_frame) - _predict(rm, left_frame)[list(p)], axis=1
        )
        result["scores"].append(_score(p, forward[None, :], backward[None, :]))
    result["application_scope"] = "tracklet_correspondence_only_change_frame_not_inferred"
    return _verdict(result, available, lo, hi, config)


def audit_transient(
    xyz, frame_idx, start, stop, support=None, config=None, *, reference_window=None
):
    """Audit one fixed permutation confined to EXACT suspect rows [start, stop].

    Unlike audit_window, exterior IDs are assumed to agree. Both exterior models
    predict the observed entry and exit of the interval, independently. Padded
    windows containing correct IDs at their endpoints will conservatively refuse
    a short inner swap; callers must supply precise boundaries to test that case.
    Optional ``reference_window=(lo, hi)`` supplies a larger exclusion window to
    keep both flanks clean while scanning exact subintervals. All excluded rows
    outside the proposed interval must then agree with both identity predictions.
    """
    config = config or ReconnectConfig()
    if reference_window is None:
        reference_window = (start, stop)
    lo, hi = reference_window
    if not lo <= start <= stop <= hi:
        raise ValueError("transient interval must lie inside reference_window")
    result, context = _prepare(
        xyz, frame_idx, lo, hi, support, config, "transient_exact_interval"
    )
    if context is None:
        return result
    xyz, frames, available, lm, rm = context
    result.update(
        reference_exclusion_row_start=int(lo),
        reference_exclusion_row_end_inclusive=int(hi),
        row_start=int(start),
        row_end_inclusive=int(stop),
        frame_start=int(frames[start]),
        frame_end=int(frames[stop]),
    )
    if not np.all(available[[start, stop]]):
        result.update(status="rejected_support", reason="transient_boundary_missing_or_unsupported")
        return result
    boundary_rows = list(dict.fromkeys((start, stop)))
    for p in PERMUTATIONS:
        forward = np.asarray([
            np.linalg.norm(_predict(lm, frames[row]) - xyz[row, list(p)], axis=1)
            for row in boundary_rows
        ])
        backward = np.asarray([
            np.linalg.norm(_predict(rm, frames[row]) - xyz[row, list(p)], axis=1)
            for row in boundary_rows
        ])
        result["scores"].append(_score(p, forward, backward))
    result["application_scope"] = "exact_interval_only_exterior_ids_assumed_unchanged"
    result = _verdict(result, available, start, stop, config)
    outside = [*range(lo, start), *range(stop + 1, hi + 1)]
    if outside:
        if not np.all(available[outside]):
            result["unchanged_rows_supported"] = False
            if result["suggested_permutation"] is not None:
                result.update(
                    status="rejected_support",
                    reason="unchanged_excluded_rows_missing_or_unsupported",
                    suggested_permutation=None,
                )
        else:
            max_error = max(
                float(np.linalg.norm(_predict(model, frames[row]) - xyz[row], axis=1).max())
                for row in outside
                for model in (lm, rm)
            )
            result["unchanged_rows_supported"] = True
            result["unchanged_rows_max_prediction_error_mm"] = max_error
            if (
                max_error > config.max_boundary_error_mm
                and result["suggested_permutation"] is not None
            ):
                result.update(
                    status="rejected_motion_mismatch",
                    reason="proposed_interval_leaves_other_excluded_rows_inconsistent",
                    suggested_permutation=None,
                )
    return result


def apply_permutation(array, start, stop, permutation):
    """Return a copy with slots selected only in inclusive rows [start, stop].

    This mechanical helper makes no decision and requires a separately reviewed
    mapping. Use identically for coordinates, all nodes, source and ID arrays.
    """
    array = np.asarray(array)
    permutation = tuple(permutation)
    if array.ndim < 2 or array.shape[1] != 3 or permutation not in PERMUTATIONS:
        raise ValueError("array needs slot axis 1 of size 3 and a valid permutation")
    if not 0 <= start <= stop < len(array):
        raise ValueError("permutation interval outside array")
    result = array.copy()
    result[start : stop + 1] = array[start : stop + 1, list(permutation)]
    return result

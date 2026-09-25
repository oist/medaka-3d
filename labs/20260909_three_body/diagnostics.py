"""Post-inference diagnostics, never identity truth or an inference input.

Historical baseline slots may themselves switch. Agreement, a returning
disagreement, stationarity and source-row changes are diagnostic observations,
not biological correctness labels. No hypothesis or threshold is fitted here.
"""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np

VIEWS = ("top_cam", "side1_cam", "side2_cam")
NODES = ("mouth", "pectoral", "tail")
PERMUTATIONS = np.asarray(list(permutations(range(3))), dtype=int)


def _runs(mask, frames):
    """Half-open frame intervals; missing comparisons must be false, not filled."""
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return [
        {"start": int(frames[lo]), "stop": int(frames[hi - 1]) + 1, "frames": int(hi - lo)}
        for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    ]


def _coverage(mask):
    return {"frames": int(np.count_nonzero(mask)), "fraction": float(np.mean(mask))}


def _finite_summary(values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    return {
        "finite_count": len(finite),
        "nonfinite_count": int(values.size - len(finite)),
        "minimum": float(finite.min()) if len(finite) else None,
        "maximum": float(finite.max()) if len(finite) else None,
        "mean": float(finite.mean()) if len(finite) else None,
    }


def _partial_baseline_disagreement(frames, xyz, reference, first_mapping, review, radius, margin):
    """Inspect an observed counterpart even when the third candidate is missing."""
    matches = np.full((len(frames), 3), -1, dtype=int)
    if first_mapping is not None:
        inverse = np.argsort(first_mapping)
        for t in range(len(frames)):
            if not np.isfinite(reference[t]).all():
                continue
            for fish in range(3):
                if not np.isfinite(xyz[t, fish]).all():
                    continue
                costs = ((reference[t] - xyz[t, fish]) ** 2).sum(-1)
                order = np.argsort(costs, kind="stable")
                best, second = order[:2]
                if costs[best] <= radius**2 and costs[second] - costs[best] > margin:
                    matches[t, fish] = inverse[best]
    compared = matches >= 0
    disagreed = compared & (matches != np.arange(3))
    episodes = {}
    for fish in range(3):
        rows = _runs(disagreed[:, fish], frames)
        for row in rows:
            lo, hi = row["start"] - int(frames[0]), row["stop"] - int(frames[0])
            row["unreviewed_frames"] = int((~review[lo:hi, fish]).sum())
        episodes[str(fish)] = rows
    duplicate_matches = []
    for a, b in combinations(range(3), 2):
        collision = compared[:, a] & compared[:, b] & (matches[:, a] == matches[:, b])
        duplicate_matches.extend({**event, "fish": [a, b]} for event in _runs(collision, frames))
    return {
        "scope": (
            "Independent nearest baseline position for each observed fish, even when another "
            "candidate is missing; NOT a joint assignment or identity GT."
        ),
        "reference_normalized_nearest_position_by_frame": matches.tolist(),
        "compared_frames_per_fish": compared.sum(axis=0).tolist(),
        "disagreement_frames_per_fish": disagreed.sum(axis=0).tolist(),
        "unreviewed_disagreement_frames_per_fish": (disagreed & ~review).sum(axis=0).tolist(),
        "episodes_per_fish": episodes,
        "two_candidates_nearest_to_same_baseline_position": duplicate_matches,
    }


def _baseline_disagreement(frames, xyz, baseline, review, radius, margin):
    if not np.array_equal(np.asarray(baseline["frame_idx"]), frames):
        raise ValueError("baseline frame_idx must exactly match result frame_idx")
    reference = np.asarray(baseline["xyz"], float)
    if reference.shape != xyz.shape:
        raise ValueError("baseline xyz must have shape (T,3,3)")
    finite = np.isfinite(xyz).all(axis=(1, 2)) & np.isfinite(reference).all(axis=(1, 2))
    compared = np.zeros(len(frames), bool)
    permutation = np.full((len(frames), 3), -1, dtype=int)
    maximum_distance = [None] * len(frames)
    gaps = [None] * len(frames)
    for t in np.flatnonzero(finite):
        squared = ((xyz[t][None] - reference[t][PERMUTATIONS]) ** 2).sum(-1)
        cost = squared.sum(-1)
        order = np.argsort(cost, kind="stable")
        best, second = order[:2]
        distance = float(np.sqrt(squared[best].max()))
        gap = float(cost[second] - cost[best])
        maximum_distance[t], gaps[t] = distance, gap
        if distance <= radius and gap > margin:
            compared[t] = True
            permutation[t] = PERMUTATIONS[best]
    valid_indices = np.flatnonzero(compared)
    normalized = np.full_like(permutation, -1)
    first_frame, first_mapping = None, None
    if len(valid_indices):
        first = int(valid_indices[0])
        first_frame = int(frames[first])
        first_mapping = permutation[first].copy()
        inverse = np.argsort(first_mapping)
        normalized[compared] = inverse[permutation[compared]]
    disagreed_fish = compared[:, None] & (normalized != np.arange(3))
    disagreed = disagreed_fish.any(axis=1)
    episodes = _runs(disagreed, frames)
    for event in episodes:
        lo, hi = event["start"] - int(frames[0]), event["stop"] - int(frames[0])
        event["affected_fish"] = np.flatnonzero(disagreed_fish[lo:hi].any(axis=0)).tolist()
        event["returns_to_reference_alignment"] = bool(
            lo > 0
            and hi < len(frames)
            and compared[lo - 1]
            and compared[hi]
            and not disagreed[lo - 1]
            and not disagreed[hi]
        )
        event["alignment_before"] = normalized[lo - 1].tolist() if lo > 0 else None
        event["alignment_after"] = normalized[hi].tolist() if hi < len(frames) else None
    return {
        "interpretation": "Position disagreement with one baseline; NOT identity GT.",
        "scope": "complete three-body observed pectoral positions; no latent fill or gap bridging",
        "compared": _coverage(compared),
        "uncompared_frames": int((~compared).sum()),
        "first_comparison_frame": first_frame,
        "initial_candidate_to_baseline_slots": None
        if first_mapping is None
        else first_mapping.tolist(),
        "candidate_to_baseline_slots_by_frame": permutation.tolist(),
        "reference_normalized_permutation_by_frame": normalized.tolist(),
        "maximum_matched_distance_mm_by_frame": maximum_distance,
        "next_permutation_cost_gap_mm2_by_frame": gaps,
        "disagreement_frames": int(disagreed.sum()),
        "per_fish_disagreement_frames": disagreed_fish.sum(axis=0).tolist(),
        "episodes": episodes,
        "returning_disagreement_episodes": sum(
            e["returns_to_reference_alignment"] for e in episodes
        ),
        "returning_event_caveat": (
            "Equal surrounding alignments would miss this transient disagreement. "
            "Either tracker or both may be wrong; this is not a measured swap count."
        ),
        "partial_counterpart_comparison": _partial_baseline_disagreement(
            frames, xyz, reference, first_mapping, review, radius, margin
        ),
    }


def diagnose(
    result,
    *,
    initial_pose_xyz=None,
    baseline_positions=None,
    fps=166.0,
    baseline_match_radius_mm=8.0,
    permutation_margin_mm2=1.0,
    stall_window_frames=25,
    stall_step_mm=0.001,
):
    """Produce JSON-safe QA for one completed native-frame tracking result.

    Each optional baseline is ``{frame_idx: (T,), xyz: (T,3,3)}``, with frames
    already aligned by the caller AFTER inference. Absolute baseline slot naming
    is factored out at the first unambiguous comparison, whose frame is recorded.
    The baseline only enters the disagreement section, never support or review.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(baseline_match_radius_mm) or baseline_match_radius_mm <= 0:
        raise ValueError("baseline_match_radius_mm must be finite and positive")
    if not np.isfinite(permutation_margin_mm2) or permutation_margin_mm2 < 0:
        raise ValueError("permutation_margin_mm2 must be finite and nonnegative")
    if (
        isinstance(stall_window_frames, bool)
        or not isinstance(stall_window_frames, int)
        or stall_window_frames < 2
    ):
        raise ValueError("stall_window_frames must be an integer >=2")
    if not np.isfinite(stall_step_mm) or stall_step_mm < 0:
        raise ValueError("stall_step_mm must be finite and nonnegative")
    frames = np.asarray(result["frame_idx"])
    if frames.ndim != 1 or not len(frames) or not np.issubdtype(frames.dtype, np.integer):
        raise ValueError("frame_idx must be nonempty integer frames")
    if np.any(np.diff(frames) != 1):
        raise ValueError("diagnostics require continuous native frames")
    count = len(frames)
    pose = np.asarray(result["pose_xyz"], float)
    observed = np.asarray(result["observed"], bool)
    review = np.asarray(result["identity_review"], bool)
    sources = np.asarray(result["sources"])
    body_supported = np.asarray(result["body_supported"], bool)
    if (
        pose.shape != (count, 3, 3, 3)
        or observed.shape != (count, 3, 3)
        or sources.shape != (count, 3, 3, 3)
        or review.shape != (count, 3)
        or body_supported.shape != (count, 3)
    ):
        raise ValueError("incompatible result array shapes")
    if not np.issubdtype(sources.dtype, np.integer) or np.any(sources < -1):
        raise ValueError("sources must be integer within-frame rows or -1")
    direct = np.isfinite(pose[:, :, 1]).all(-1) & observed[:, :, 1]
    accepted = direct & ~review
    accepted_consistency = None
    if "accepted_xyz" in result:
        stored = np.asarray(result["accepted_xyz"], float)
        if stored.shape != (count, 3, 3):
            raise ValueError("accepted_xyz must have shape (T,3,3)")
        stored_accepted = np.isfinite(stored).all(-1)
        accepted_consistency = int(np.count_nonzero(stored_accepted != accepted))
        accepted = stored_accepted
    xyz = np.where(direct[:, :, None], pose[:, :, 1], np.nan)
    identity = {
        "status": "independent_human_three_animal_truth_required",
        "observation_is_not_identity_accuracy": True,
        "observed_pectoral": [_coverage(direct[:, fish]) for fish in range(3)],
        "accepted_pectoral": [_coverage(accepted[:, fish]) for fish in range(3)],
        "accepted_mask_inconsistent_with_observed_and_review_count": accepted_consistency,
    }
    workload = {
        "all_fish_union": _runs(review.any(axis=1), frames),
        "per_fish": {str(f): _runs(review[:, f], frames) for f in range(3)},
        "co_reviewed_pairs": {
            f"{a}-{b}": _runs(review[:, a] & review[:, b], frames)
            for a, b in combinations(range(3), 2)
        },
        "pair_caveat": "Co-review is simultaneous withholding, not a pair-specific swap cause.",
        "review_fish_frames": int(review.sum()),
        "review_union_frames": int(review.any(axis=1).sum()),
        "review_union_seconds": float(review.any(axis=1).sum() / fps),
        "human_annotation_time": None,
    }
    present = sources >= 0
    row_changed = present[1:] & present[:-1] & (sources[1:] != sources[:-1])
    collisions = []
    for view, view_name in enumerate(VIEWS):
        for node, node_name in enumerate(NODES):
            for a, b in combinations(range(3), 2):
                collision = present[:, a, view, node] & (
                    sources[:, a, view, node] == sources[:, b, view, node]
                )
                for event in _runs(collision, frames):
                    collisions.append(
                        {**event, "view": view_name, "node": node_name, "fish": [a, b]}
                    )
    source_diagnostic = {
        "axes": ["fish", "view", "node"],
        "views": list(VIEWS),
        "nodes": list(NODES),
        "within_frame_row_changes": row_changed.sum(axis=0).tolist(),
        "became_missing": (present[:-1] & ~present[1:]).sum(axis=0).tolist(),
        "became_present": (~present[:-1] & present[1:]).sum(axis=0).tolist(),
        "exclusive_source_violations": collisions,
        "row_change_caveat": "Frame-local row changes do NOT measure animal switches.",
        "duplicate_instance_caveat": "Distinct rows can still be duplicate detections of one fish.",
    }
    stationarity = []
    for fish in range(3):
        for node in (0, 1):
            displacement = np.linalg.norm(np.diff(pose[:, fish, node], axis=0), axis=-1)
            stable = np.isfinite(displacement) & (displacement <= stall_step_mm)
            for edge_run in _runs(stable, frames[1:]):
                start, stop = edge_run["start"] - 1, edge_run["stop"]
                if stop - start < stall_window_frames:
                    continue
                lo, hi = start - int(frames[0]), stop - int(frames[0])
                stationarity.append(
                    {
                        "fish": fish,
                        "node": NODES[node],
                        "start": start,
                        "stop": stop,
                        "frames": stop - start,
                        "observed_fraction": float(observed[lo:hi, fish, node].mean()),
                        "review_fraction": float(review[lo:hi, fish].mean()),
                        "net_displacement_mm": float(
                            np.linalg.norm(pose[hi - 1, fish, node] - pose[lo, fish, node])
                        ),
                    }
                )
    reference = np.asarray(initial_pose_xyz, float) if initial_pose_xyz is not None else pose[0]
    if reference.shape != (3, 3, 3) or not np.isfinite(reference).all():
        raise ValueError("initial pose must be finite (3,3,3)")
    lengths = np.linalg.norm(np.diff(pose, axis=2), axis=-1)
    initial_lengths = np.linalg.norm(np.diff(reference, axis=1), axis=-1)
    if np.any(initial_lengths <= 0):
        raise ValueError("initial body segments must be positive")
    anatomy = []
    for fish in range(3):
        for segment in range(2):
            finite = lengths[:, fish, segment][np.isfinite(lengths[:, fish, segment])]
            minimum = float(finite.min()) if len(finite) else None
            maximum = float(finite.max()) if len(finite) else None
            anatomy.append(
                {
                    "fish": fish,
                    "segment": segment,
                    "initial_mm": float(initial_lengths[fish, segment]),
                    "minimum_mm": minimum,
                    "maximum_mm": maximum,
                    "maximum_initial_ratio": None
                    if maximum is None
                    else float(maximum / initial_lengths[fish, segment]),
                    "nonfinite_frames": int(count - len(finite)),
                }
            )
    bridge = body_supported & ~observed[:, :, 1]
    head_ray_count = present[:, :, :, :2].sum(axis=(2, 3))
    loss_summary = {
        key: _finite_summary(result[key]) for key in ("costs", "path_costs") if key in result
    }
    if "loss_parts" in result:
        parts = np.asarray(result["loss_parts"])
        if parts.shape != (count, 5):
            raise ValueError("loss_parts must be (T,5): node, shape, center, repulsion, silhouette")
        loss_summary["parts"] = {
            name: _finite_summary(parts[:, column])
            for column, name in enumerate(("node", "shape", "center", "repulsion", "silhouette"))
        }
        loss_summary["scope"] = (
            "Unweighted component summaries; orientation is not a saved part. "
            "Do not reconstruct total loss by simply adding these columns."
        )
    margin_summary = None
    if "score_margin" in result:
        margins = np.asarray(result["score_margin"], float)
        margin_summary = {
            **_finite_summary(margins),
            "positive_infinity_count": int(np.isposinf(margins).sum()),
            "negative_count": int((margins < 0).sum()),
            "scope": "Online gap, not confidence; infinity means no compared alternative.",
        }
    return {
        "kind": "post_inference_diagnostics_not_identity_ground_truth",
        "frame_range": [int(frames[0]), int(frames[-1]) + 1],
        "frames": count,
        "fps": fps,
        "settings": {
            "baseline_match_radius_mm": baseline_match_radius_mm,
            "permutation_margin_mm2": permutation_margin_mm2,
            "stationarity_window_frames": stall_window_frames,
            "stationarity_step_mm": stall_step_mm,
            "near_duplicate_pectoral_mm": 0.5,
        },
        "identity_evidence": identity,
        "review_workload": workload,
        "sources": source_diagnostic,
        "body_bridge_frames_per_fish": bridge.sum(axis=0).tolist(),
        "bridge_with_only_one_mouth_or_pectoral_ray_per_fish": (bridge & (head_ray_count == 1))
        .sum(axis=0)
        .tolist(),
        "body_support_caveat": "Mask support is an engineering bridge, not identity confirmation.",
        "stationary_pose_runs": stationarity,
        "stationarity_caveat": "Real fish can be stationary; these are inspection candidates.",
        "anatomy_reference": "supplied_initial_pose"
        if initial_pose_xyz is not None
        else "first_result_pose",
        "segment_lengths": anatomy,
        "near_duplicate_pectoral_pairs": {
            f"{a}-{b}": _runs(np.linalg.norm(pose[:, a, 1] - pose[:, b, 1], axis=-1) < 0.5, frames)
            for a, b in combinations(range(3), 2)
        },
        "near_duplicate_caveat": "Inspect close centers; contact does not prove duplication.",
        "loss_summary": loss_summary,
        "online_margin_summary": margin_summary,
        "baseline_disagreement": {
            name: _baseline_disagreement(
                frames, xyz, baseline, review, baseline_match_radius_mm, permutation_margin_mm2
            )
            for name, baseline in (baseline_positions or {}).items()
        },
    }

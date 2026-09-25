"""Retain the measured sources shared by every competitive joint assignment.

The frozen v1 estimator is preserved. Its physical gates, costs and 0.5 mm
margin are unchanged. The revised unit of abstention is a view observation:
uncertainty in the third view need not erase an unambiguous two-view point.
"""

from __future__ import annotations

from dataclasses import asdict
from itertools import pairwise

import multiview as base
import numpy as np

from medaka3d.types import VIEWS

MultiViewConfig = base.MultiViewConfig


def _joint_states(hypotheses, histories, votes, frame, config):
    """Expose v1's exact objective and feasible states without altering v1.

    This is the frozen `_assignment` score calculation, with its arrays returned
    for an observation-level consensus. A parity test pins winner and margins.
    No motion, geometry, missingness, or margin threshold is changed.
    """
    n = len(hypotheses["source"])
    costs = np.broadcast_to(hypotheses["cost"], (3, n)).copy()
    conflicted = (votes > 0).sum(-1) > 1
    costs[:, conflicted] = np.inf
    for slot in range(3):
        strong_other = (votes >= 2).any(-1) & (votes[:, slot] < 2)
        costs[slot, strong_other] = np.inf
        costs[slot] -= config.single_clean_vote_bonus_mm * (votes[:, slot] == 1)
        history = histories[slot]
        if not history:
            continue
        last_frame, last = history[-1]
        gap = int(frame - last_frame)
        velocity = np.zeros(3)
        if len(history) >= 2:
            velocity = np.median(
                [(b - a) / (fb - fa) for (fa, a), (fb, b) in pairwise(history)], axis=0
            )
        xyz = hypotheses["pose"][:, 1]
        distance = np.linalg.norm(xyz - (last + velocity * gap), axis=-1)
        step = np.linalg.norm(xyz - last, axis=-1) / gap
        if gap <= config.identity_gap_frames:
            gate = config.prediction_gate_mm + config.gap_prediction_growth_mm * (gap - 1)
            costs[slot, (distance > gate) | (step > config.maximum_speed_mm_per_frame)] = np.inf
            costs[slot] += config.prediction_weight * distance
        else:
            costs[slot] += config.prediction_weight * np.minimum(distance, 12.0)
            anchored = votes[:, slot] >= 2
            costs[slot, anchored & (step > config.maximum_speed_mm_per_frame)] = np.inf
    choices = [np.r_[np.flatnonzero(costs[s] <= config.absent_slot_cost_mm), n] for s in range(3)]
    states = np.stack(np.meshgrid(*choices, indexing="ij"), axis=-1).reshape(-1, 3)
    source = np.vstack([hypotheses["source"], np.full((1, 3), -1)])
    valid = np.ones(len(states), dtype=bool)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        left, right = source[states[:, a]], source[states[:, b]]
        valid &= ~np.any((left >= 0) & (left == right), axis=-1)
        if not histories[a] and not histories[b]:
            valid &= states[:, a] <= states[:, b]
    states = states[valid]
    padded = np.column_stack([costs, np.full(3, config.absent_slot_cost_mm)])
    total = padded[np.arange(3)[None, :], states].sum(-1)
    return states, total, costs, int(conflicted.sum())


def _consensus(hypotheses, histories, votes, frame, config):
    states, total, costs, conflicts = _joint_states(hypotheses, histories, votes, frame, config)
    n = len(hypotheses["source"])
    best_row = int(np.argmin(total))
    best = states[best_row]
    margins = np.full(3, np.inf)
    for slot in range(3):
        rival = states[:, slot] != best[slot]
        if rival.any():
            margins[slot] = float(total[rival].min() - total[best_row])
    # V1 accepts a margin equal to the threshold. Only strictly closer rivals
    # belong to the ambiguous set; include the winner even for a zero threshold.
    competitive = total - total[best_row] < config.correspondence_margin_mm
    competitive[best_row] = True
    near = states[competitive]
    source = np.vstack([hypotheses["source"], np.full((1, 3), -1)])
    near_sources = source[near]
    common = np.where(np.all(near_sources == near_sources[0], axis=0), near_sources[0], -1)
    selected = np.full(3, n, dtype=int)
    preserved = np.zeros(3, dtype=bool)
    withheld = np.zeros((3, 3), dtype=bool)
    for slot in range(3):
        if (common[slot] >= 0).sum() < 2:
            continue
        candidates = np.flatnonzero(np.all(hypotheses["source"] == common[slot], axis=1))
        if len(candidates) != 1:
            # The independently triangulated common tuple failed a frozen
            # geometry/tank gate; a fused three-view fit cannot vouch for it.
            continue
        candidate = int(candidates[0])
        if not np.isfinite(costs[slot, candidate]):
            continue
        if costs[slot, candidate] > config.absent_slot_cost_mm:
            continue
        selected[slot] = candidate
        preserved[slot] = margins[slot] < config.correspondence_margin_mm
        withheld[slot] = (source[best[slot]] >= 0) & (common[slot] < 0)
    return dict(
        selected=selected,
        best=best,
        margins=margins,
        preserved=preserved,
        withheld=withheld,
        missing=n,
        conflicts=conflicts,
        state_count=len(states),
        near_state_count=len(near),
        common_source=common,
    )


def reconstruct(frames, views, cameras, *, config=None):
    """Use unchanged v1 evidence with per-view consensus across near-best states."""
    config = MultiViewConfig() if config is None else config
    frames = np.asarray(frames, dtype=np.int64)
    if frames.ndim != 1 or (len(frames) > 1 and not np.all(np.diff(frames) == 1)):
        raise ValueError("frames must be unique contiguous native frames")
    camera_map = base._cameras(cameras)
    prepared = base._prepare(frames, views, camera_map)
    f = len(frames)
    pose = np.full((f, 3, 3, 3), np.nan)
    uv = np.full((f, 3, 3, 3, 2), np.nan)
    source = np.full((f, 3, 3), -1, dtype=np.int64)
    selected_rows = source.copy()
    residual = np.full((f, 3, 3), np.nan)
    raw_residual = residual.copy()
    tail_outside = np.zeros((f, 3), dtype=bool)
    tail_high_residual = tail_outside.copy()
    margins = np.full((f, 3), np.nan)
    identity_review = np.ones((f, 3), dtype=bool)
    consensus_preserved = np.zeros((f, 3), dtype=bool)
    withheld_view = np.zeros((f, 3, 3), dtype=bool)
    near_state_count = np.zeros(f, dtype=np.int64)
    sticky_review = np.zeros(3, dtype=bool)
    histories = [[] for _ in range(3)]
    owners = {view: {} for view in VIEWS}
    events = []
    max_hypotheses = max_states = 0
    for ti, frame in enumerate(frames):
        h = base._hypotheses(frame, prepared, config)
        if h is None:
            events.append(dict(frame=int(frame), reason="no_supported_two_view_geometry"))
            continue
        votes = base._vote_counts(h, prepared, owners)
        decision = _consensus(h, histories, votes, frame, config)
        missing = decision["missing"]
        max_hypotheses = max(max_hypotheses, missing)
        max_states = max(max_states, decision["state_count"])
        margins[ti] = decision["margins"]
        near_state_count[ti] = decision["near_state_count"]
        if decision["conflicts"]:
            events.append(
                dict(
                    frame=int(frame),
                    reason="contradicting_clean_anchors",
                    rejected_hypotheses=decision["conflicts"],
                )
            )
        for slot, index in enumerate(decision["selected"]):
            if index == missing:
                events.append(
                    dict(
                        frame=int(frame),
                        slot=slot,
                        reason="no_common_supported_two_view_observation",
                        complete_tuple_margin_mm=float(decision["margins"][slot]),
                    )
                )
                continue
            history = histories[slot]
            gap = int(frame - history[-1][0]) if history else 0
            if gap > config.identity_gap_frames and votes[index, slot] < 2:
                sticky_review[slot] = True
                events.append(
                    dict(
                        frame=int(frame),
                        slot=slot,
                        reason="long_gap_unanchored_identity_reacquisition",
                        gap_frames=gap,
                    )
                )
            identity_review[ti, slot] = sticky_review[slot]
            consensus_preserved[ti, slot] = decision["preserved"][slot]
            withheld_view[ti, slot] = decision["withheld"][slot]
            if decision["preserved"][slot]:
                events.append(
                    dict(
                        frame=int(frame),
                        slot=slot,
                        reason="common_two_view_consensus",
                        near_joint_assignments=decision["near_state_count"],
                        complete_tuple_margin_mm=float(decision["margins"][slot]),
                        retained_source=h["source"][index].tolist(),
                        withheld_views=[
                            VIEWS[v] for v in np.flatnonzero(decision["withheld"][slot])
                        ],
                    )
                )
            source[ti, slot] = h["source"][index]
            selected_rows[ti, slot] = h["rows"][index]
            pose[ti, slot] = h["pose"][index]
            uv[ti, slot] = h["uv"][index]
            residual[ti, slot] = h["residual"][index]
            raw_residual[ti, slot] = h["raw_residual"][index]
            tail_outside[ti, slot] = h["tail_outside"][index]
            tail_high_residual[ti, slot] = h["tail_high_residual"][index]
            history.append((int(frame), pose[ti, slot, 1].copy()))
            if len(history) > 4:
                del history[0]
            for vi, view in enumerate(VIEWS):
                row = selected_rows[ti, slot, vi]
                if row >= 0:
                    owners[view].setdefault(int(prepared[view]["labels"][row]), slot)
    joined, bridges = base._join_layer(
        frames, selected_rows, identity_review, prepared, camera_map, config
    )
    projection = np.stack([base._project(camera_map[v], pose) for v in VIEWS], axis=2)
    return dict(
        xyz=pose[:, :, 1].copy(),
        pose=pose,
        source=source,
        uv=uv,
        residual_mm=residual,
        raw_residual_mm=raw_residual,
        projected_uv=projection,
        identity_review=identity_review,
        assignment_margin_mm=margins,
        consensus_preserved=consensus_preserved,
        withheld_view=withheld_view,
        competitive_joint_state_count=near_state_count,
        labels_by_view={v: prepared[v]["labels"].copy() for v in VIEWS},
        clean_edge_by_view={v: prepared[v]["clean"].copy() for v in VIEWS},
        joined_labels_by_view=joined,
        bridges=bridges,
        events=events,
        metadata=dict(
            config=asdict(config),
            node_order=["mouth", "pectoral", "tail"],
            view_order=list(VIEWS),
            algorithm_revision="v2_common_observations_of_all_competitive_joint_states",
            input_detections={v: len(prepared[v]["frames"]) for v in VIEWS},
            selected_detections={v: int((source[:, :, i] >= 0).sum()) for i, v in enumerate(VIEWS)},
            clean_edges_unchanged=True,
            saved_temporal_ids_used=False,
            historical_reconstruction_used=False,
            identity_accuracy_claim=False,
            accepted_bridges=sum(e["accepted"] for e in bridges),
            tail_masked_fish_frames=int((tail_outside | tail_high_residual).sum()),
            tail_outside_tank_fish_frames=int(tail_outside.sum()),
            tail_high_ray_residual_fish_frames=int(tail_high_residual.sum()),
            maximum_frame_hypotheses=max_hypotheses,
            maximum_joint_assignments=max_states,
            consensus_preserved_fish_frames=int(consensus_preserved.sum()),
            withheld_view_observations=int(withheld_view.sum()),
            maximum_competitive_joint_assignments=int(near_state_count.max(initial=0)),
            margin_scope=(
                "assignment_margin_mm remains the original complete-tuple margin; "
                "accepted common two-view measurements may have a smaller value"
            ),
            consensus_scope=(
                "intersection over all joint states strictly within the unchanged "
                "margin; >=2 identical measured sources and unchanged gates required"
            ),
            bridge_layer="auxiliary 2D joins; initial clean edges and 3D observations unchanged",
            identity_review_rule=(
                "missing observations and sticky unanchored reacquisition after "
                f">{config.identity_gap_frames} frames"
            ),
            source_index_rule=(
                "original input row order within each native frame; "
                "-1 means no observation, including ambiguous withheld views"
            ),
        ),
    )

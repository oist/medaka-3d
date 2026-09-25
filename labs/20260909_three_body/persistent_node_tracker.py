"""Separate mouth recovery variant requiring direct observation history per beam parent.

The explicit track loop preserves the frozen tracker.py operation order. Its only
state additions are node missing ages and consecutive directly supported pectoral
ages. Mask/body support never increments the direct pectoral age. Ages follow
persistent ID slots along each temporal parent, without pose hashes or side maps.
"""

import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
from accelerated_tracker import AcceleratedThreeBodyTracker
from node_geometry import RayObservations, supported_nodes
from node_recovery import NodeRecoveryConfig, node_recovery_proposals
from observations import sample_pixels
from tracker import Proposal, State, ThreeBodyTracker

FROZEN_TRACKER_FILE_SHA256 = "f481e2d4105a0223cf24719482e1d057ba4e68094ff9df74af0a826a88d31c20"


@dataclass(frozen=True)
class PersistentNodeConfig:
    enabled: bool = True
    minimum_missing_frames: int = 4
    minimum_pectoral_frames: int = 4

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in ("minimum_missing_frames", "minimum_pectoral_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class PersistentNodeState(State):
    node_missing: np.ndarray  # fish,node; directly unsupported completed frames
    direct_pectoral_age: np.ndarray  # fish; consecutive directly supported completed frames


def updated_ages(parent, observed):
    """Update from this selected fitted state's strict support, not its proposal kind."""
    return (
        np.where(observed, 0, parent.node_missing + 1),
        np.where(observed[:, 1], parent.direct_pectoral_age + 1, 0),
    )


class PersistentNodeThreeBodyTracker(ThreeBodyTracker):
    """Original proposals plus age-eligible mouths from the first retained parent.

    Eligibility uses four *already completed* frames by default. Thus the earliest
    possible extra proposal is input frame five, with the current predicted scene
    also requiring strict pectoral support and a directly unsupported mouth. All
    original proposals retain their order and budget. At most six extra candidates
    are appended, as in the separate current-frame-only mouth recovery variant.

    No age threshold clears identity review. A geometrically supported replacement
    can still join different animals' parts; this gate only reduces transient triggers.
    """

    def __init__(self, rig, initial_pose_xyz, config=None, *, recovery_config=None,
                 persistence_config=None):
        super().__init__(rig, initial_pose_xyz, config)
        self.recovery_config = recovery_config or NodeRecoveryConfig()
        self.persistence_config = persistence_config or PersistentNodeConfig()
        if self.recovery_config.proposals_per_fish > 2:
            raise ValueError("this experimental tracker reserves at most two mouths per fish")
        if (not np.array_equal(self.lower, self.recovery_config.lower_mm)
                or not np.array_equal(self.upper, self.recovery_config.upper_mm)):
            raise ValueError("mouth recovery and inherited tracker must use the same box")
        self._recovery_calls = 0
        self._age_eligible_calls = 0
        self._recovery_proposed_calls = 0
        self._recovery_proposed_by_fish = np.zeros(3, np.int64)
        self._recovery_status_counts = Counter()

    def _age_eligible(self, state):
        cfg = self.persistence_config
        return ((state.node_missing[:, 0] >= cfg.minimum_missing_frames)
                & (state.direct_pectoral_age >= cfg.minimum_pectoral_frames))

    @property
    def node_recovery_diagnostics(self):
        return dict(
            calls=self._recovery_calls,
            calls_with_age_eligible_fish=self._age_eligible_calls,
            calls_with_additional_proposals=self._recovery_proposed_calls,
            proposed_by_fish=self._recovery_proposed_by_fish.tolist(),
            fish_status_counts=dict(self._recovery_status_counts),
            original_budget=self.config.max_hypotheses,
            additional_budget=3 * self.recovery_config.proposals_per_fish,
            scope="first-parent proposal counts, not selected recoveries or identity accuracy",
        )

    def _proposals(self, states, rays, flow=None):
        original = super()._proposals(states, rays, flow)
        self._recovery_calls += 1
        if not self.persistence_config.enabled:
            return original
        main = original[0]
        eligible = self._age_eligible(states[main.parent])
        if not eligible.any():
            return original
        self._age_eligible_calls += 1
        extra = node_recovery_proposals(
            main.pose, main.plan, rays, self.rig, self.lengths, self.recovery_config
        )
        self._recovery_status_counts.update(row["status"] for row in extra.diagnostics["fish"])
        added = []
        for candidate in extra.proposals:
            if not eligible[candidate.fish]:
                continue
            self._recovery_proposed_by_fish[candidate.fish] += 1
            added.append(Proposal(candidate.pose_xyz, main.predicted, candidate.plan, main.parent,
                                  "mouth_recovery", main.permutation))
        self._recovery_proposed_calls += bool(added)
        return original + added

    def track(self, frames):
        if self.initialize_count:
            raise RuntimeError("tracker instances are single-use: no hidden reinitialization")
        self.initialize_count = 1
        cfg = self.config
        states = [
            PersistentNodeState(self.initial.copy(), np.zeros((3, 3)), np.zeros(3, int),
                                np.zeros(3, bool), 0.0, np.zeros((3, 3), int), np.zeros(3, int))
        ]
        history, previous_frame = [], None
        previous_observation = None
        if cfg.flow_proposals:
            from motion import FlowObservations
        started = time.perf_counter()
        for index, frame in enumerate(frames):
            if previous_frame is not None and frame.frame_idx != previous_frame + 1:
                raise ValueError("tracking input must be continuous native frames")
            previous_frame = frame.frame_idx
            rays = RayObservations.from_points(self.rig, frame.detections)
            targets = (
                None
                if cfg.mode == "nodes"
                else sample_pixels(
                    frame, cfg.mode, cfg.positive_samples, cfg.near_samples, cfg.global_samples
                )
            )
            flow = None
            if cfg.flow_proposals and previous_observation is not None:
                flow = FlowObservations.from_frames(
                    previous_observation, frame, self.rig, states[0].pose
                )
            proposals = (
                self._proposals(states, rays)
                if flow is None
                else self._proposals(states, rays, flow)
            )
            poses, local_cost, parts, mask_support = self._fit(proposals, states, rays, targets)
            path_cost = local_cost + np.array([states[p.parent].cost for p in proposals])
            order = np.argsort(path_cost, kind="stable")
            selected = []
            for candidate in order:
                if any(
                    proposals[candidate].parent == proposals[j].parent
                    and proposals[candidate].permutation == proposals[j].permutation
                    and np.max(np.linalg.norm(poses[candidate] - poses[j], axis=-1)) < 0.5
                    for j in selected
                ):
                    continue
                selected.append(int(candidate))
                if len(selected) == cfg.beam_size:
                    break
            next_states, records = [], []
            for candidate in selected:
                proposal = proposals[candidate]
                parent = states[proposal.parent]
                pose = poses[candidate]
                sources, observed, _ = supported_nodes(pose, proposal.plan, rays, self.rig)
                has_head_ray = (sources[:, :, :2] >= 0).any(axis=(1, 2))
                body_supported = observed[:, 1] | (mask_support[candidate] & has_head_ray)
                missing = np.where(body_supported, 0, parent.missing + 1)
                review = parent.review | (missing >= cfg.review_gap_frames)
                identity_margin = float("inf")
                for other in order:
                    if other == candidate:
                        continue
                    affected = self._different_identity(pose, poses[other])
                    # At a spatial tie, permutation argmin alone chooses identity
                    # lexically. Preserve explicit same-parent symmetry candidates.
                    if proposals[other].parent == proposal.parent:
                        tied = (
                            np.max(np.linalg.norm(pose[:, 1] - poses[other, :, 1], axis=-1)) < 0.5
                        )
                        if tied:
                            affected |= np.asarray(proposal.permutation) != np.asarray(
                                proposals[other].permutation
                            )
                    if not affected.any():
                        continue
                    gap = float(path_cost[other] - path_cost[candidate])
                    identity_margin = min(identity_margin, gap)
                    if gap <= cfg.ambiguity_margin:
                        review |= affected
                displacement = pose[:, 1] - parent.pose[:, 1]
                velocity = 0.5 * parent.velocity + 0.5 * displacement
                speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
                velocity *= np.minimum(1.0, 2.0 / np.maximum(speed, 1e-12))
                velocity = np.where(body_supported[:, None], velocity, 0.8 * parent.velocity)
                node_missing, direct_pectoral_age = updated_ages(parent, observed)
                next_states.append(
                    PersistentNodeState(
                        pose, velocity, missing, review, float(path_cost[candidate]),
                        node_missing, direct_pectoral_age
                    )
                )
                records.append(
                    dict(
                        frame_idx=frame.frame_idx,
                        pose_xyz=pose,
                        observed=observed,
                        sources=sources,
                        identity_review=review,
                        mask_supported=mask_support[candidate],
                        body_supported=body_supported,
                        missing=missing,
                        costs=local_cost[candidate],
                        path_costs=path_cost[candidate],
                        loss_parts=parts[candidate],
                        score_margin=identity_margin,
                        parent=proposal.parent,
                        proposal_kind=proposal.kind,
                        hypotheses=len(proposals),
                        proposal_flow_nodes=(
                            flow.node_valid if flow is not None else np.zeros((3, 3), bool)
                        ),
                        proposal_flow_parent=(0 if flow is not None else -1),
                        node_missing=node_missing,
                        direct_pectoral_age=direct_pectoral_age,
                    )
                )
            history.append(records)
            states = next_states
            if cfg.flow_proposals:
                previous_observation = frame
            if index % 25 == 0:
                elapsed = time.perf_counter() - started
                direct = records[0]["observed"][:, 1].astype(int).tolist()
                print(
                    f"frame={frame.frame_idx} processed={index + 1} elapsed_s={elapsed:.1f} "
                    f"beam={len(states)} supported={direct} "
                    f"review={records[0]['identity_review'].astype(int).tolist()}",
                    flush=True,
                )
        if not history:
            raise ValueError("empty tracking sequence")
        chosen, state_index = [], 0
        for records in reversed(history):
            record = records[state_index]
            chosen.append(record)
            state_index = record["parent"]
        chosen.reverse()
        result = {
            key: np.asarray([record[key] for record in chosen])
            for key in chosen[0]
            if key != "parent"
        }
        result["xyz"] = np.where(
            result["observed"][:, :, 1, None], result["pose_xyz"][:, :, 1], np.nan
        )
        result["accepted_xyz"] = np.where(
            result["identity_review"][:, :, None], np.nan, result["xyz"]
        )
        result["body_xyz"] = np.where(
            result["body_supported"][:, :, None], result["pose_xyz"][:, :, 1], np.nan
        )
        return result


class AcceleratedPersistentNodeTracker(
    PersistentNodeThreeBodyTracker, AcceleratedThreeBodyTracker
):
    """The same explicit age-aware tracker with only the validated fitter accelerated."""

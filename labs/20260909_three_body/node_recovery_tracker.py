"""A separate proposal-only revision for mouths lost beside supported centers.

All original hypotheses compete unchanged. The additional mouth hypotheses do
not initialize a fish, certify identity or clear review. The inherited fit may
move any node as usual; mouth-only describes the additional starting proposals.
"""

from collections import Counter

import numpy as np
from accelerated_tracker import AcceleratedThreeBodyTracker
from node_recovery import NodeRecoveryConfig, node_recovery_proposals
from tracker import Proposal, ThreeBodyTracker


class NodeRecoveryThreeBodyTracker(ThreeBodyTracker):
    """Keep the original beam dynamics and add at most two mouths per fish.

    config.max_hypotheses remains the ORIGINAL proposal budget (18 in the
    registered arm). Recovery adds up to six candidates after that budget,
    retaining original hypotheses instead of replacing their alternatives.
    Additional proposals use the first parent's same motion prior and cost.
    Current-frame strict node support decides eligibility; there is no reset
    from archived poses, labels or a later frame.
    """

    def __init__(self, rig, initial_pose_xyz, config=None, *, recovery_config=None):
        super().__init__(rig, initial_pose_xyz, config)
        self.recovery_config = recovery_config or NodeRecoveryConfig()
        if self.recovery_config.proposals_per_fish > 2:
            raise ValueError("this experimental tracker reserves at most two mouths per fish")
        if (
            not np.array_equal(self.lower, self.recovery_config.lower_mm)
            or not np.array_equal(self.upper, self.recovery_config.upper_mm)
        ):
            raise ValueError("mouth recovery and inherited tracker must use the same box")
        self._recovery_calls = 0
        self._recovery_proposed_calls = 0
        self._recovery_proposed_by_fish = np.zeros(3, np.int64)
        self._recovery_status_counts = Counter()

    @property
    def node_recovery_diagnostics(self):
        return dict(
            calls=self._recovery_calls,
            calls_with_additional_proposals=self._recovery_proposed_calls,
            proposed_by_fish=self._recovery_proposed_by_fish.tolist(),
            fish_status_counts=dict(self._recovery_status_counts),
            original_budget=self.config.max_hypotheses,
            additional_budget=3 * self.recovery_config.proposals_per_fish,
            scope="proposal counts, not selected recoveries or identity accuracy",
        )

    def _proposals(self, states, rays, flow=None):
        original = super()._proposals(states, rays, flow)
        main = original[0]
        extra = node_recovery_proposals(
            main.pose, main.plan, rays, self.rig, self.lengths, self.recovery_config
        )
        self._recovery_calls += 1
        self._recovery_proposed_calls += bool(extra.proposals)
        self._recovery_status_counts.update(row["status"] for row in extra.diagnostics["fish"])
        added = []
        for candidate in extra.proposals:
            self._recovery_proposed_by_fish[candidate.fish] += 1
            added.append(
                Proposal(
                    candidate.pose_xyz, main.predicted, candidate.plan, main.parent,
                    "mouth_recovery", main.permutation,
                )
            )
        return original + added


class AcceleratedNodeRecoveryTracker(
    NodeRecoveryThreeBodyTracker, AcceleratedThreeBodyTracker
):
    """Same recovery variant, with the previously verified CUDA Graph fitter."""
